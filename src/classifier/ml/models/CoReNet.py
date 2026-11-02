"""Barista training / evaluation wrapper for the CoReNet network (``src.classifier.nn.corenet``).

One-to-one counterpart of ``src.classifier.ml.models.HCR``: the same stages, schedules, ghost-batch
schedule (``GBNSchedule``), benchmarks (``HCRBenchmarks``) and output columns, with the hard-coded HCR
network replaced by a ``CoReNet`` built from an event description.

Usage (from a model config, e.g. ``coffea4bees/classifier/config/model/CoReNet/_CoReNet.py``)::

    arch = CoReNetArch(loss=loss, network=<event description mapping>, inputs={"b": "CanJet", "other": "NotCanJet"},
                       architecture={"n_features": 4, "attention": True})
    CoReNetTraining(arch=arch, ghost_batch=GBNSchedule(), cross_validation=splitter,
                    training_schedule=..., benchmarks=HCRBenchmarks(rocs=...), model="FvT")
    CoReNetEvaluation(saved_model="<name>__<uuid>.pkl", cross_validation=splitter, output_definition=...)

Inputs
  * the event description (``network``): the CoReNet configuration grammar (objects, hypotheses,
    outputs, combine, training, ...). ``event:`` and ``labels:`` must NOT be set in it: barista fills
    ``event`` with ``InputBranch.feature_ancillary`` (in order) and ``labels`` with
    ``MultiClass.trainable_labels``. The name of the first output is a placeholder: ``outputs`` is
    rewritten to the single entry ``{labels[0]: <hypothesis of the first output>}``; the classes come
    from the dataset and no auxiliary heads are built.
  * ``inputs``: role name -> batch key (default ``{"b": Input.CanJet, "other": Input.NotCanJet}``).
  * ``architecture`` (optional, HCR-style overrides): ``n_features`` -> ``training.features``,
    ``use_attention_gate`` -> ``training.attention_gate``, ``use_kv_proj`` -> ``training.kv_projection``,
    ``attention: false`` removes every ``context`` of the hypotheses (``true`` keeps the file as is).

Batch-layout contract (``_CoReNetInput``): for each (role, key) of ``inputs`` the batch tensor
``batch[key]`` is viewed as ``(n, channels, slots)`` with ``slots = spec.roles[role].n`` (feature-major,
the layout HCR uses: CanJet -> (n, n_features, 4), NotCanJet -> (n, 5, 8)). With ``channels`` equal to
the number of features of the role's collection a presence channel of +1 is appended (HCR candidate jets
are always present); with one more channel the last one is used as the flag as given (e.g. isSelJet,
-1 for padding); any other count is an error. The event scalars are ``batch[Input.ancillary]``.

Eta reflection: CoReNet flips eta so that the leading slot of the first role (``leaf_order[0]``) has
eta >= 0, as HCR does with the leading candidate jet; the first role of the event description must
therefore be the first key of ``inputs``.

Outputs
  * training: ``Output.class_raw`` (class logits) and ``Output.quadjet_raw`` (view logits of the main
    hypothesis, when it has more than one view) in the batch, then the loss.
  * evaluation: ``p_<label>`` per trained label and, for a main hypothesis with V > 1 views, one view
    probability per view: ``q_`` + the 1-based slots of the view in order when every leaf of the views
    belongs to one role (HH -> 4b: q_1234, q_1324, q_1423, HCR's names and order), else ``q_<hyp>_<v>``.
  * checkpoint ``<name>__<uuid>.pkl`` (torch.save): ``model`` (state dict), ``metadata``, ``uuid``,
    ``label`` (MultiClass.trainable_labels), ``arch`` (CoReNetArch.save(): ``network``, ``inputs``,
    ``architecture`` and ``resolved``, the spec mapping the network was built from, so the checkpoint is
    self-contained), ``input`` (InputBranch feature_ancillary, feature_CanJet, feature_NotCanJet,
    n_NotCanJet; checked at evaluation as for HCR).
"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable

import fsspec
import torch
import torch.nn.functional as F
import torch.types as tt
from src.classifier.config.scheduler import SkimStep
from src.classifier.config.setting.HCR import Input, InputBranch, Output
from src.classifier.config.setting.ml import SplitterKeys
from src.classifier.config.state.label import MultiClass
from torch import Tensor

from ...algorithm.utils import Selector, map_batch, to_num
from ...nn.corenet import CoReNet
from ...nn.corenet.config import NetworkSpec, parse_config
from ...nn.schedule import Schedule
from ...utils import MemoryViewIO
from .. import BatchType
from ..evaluation import Evaluation, EvaluationStage
from ..skimmer import Skimmer, Splitter
from ..training import (
    BenchmarkStage,
    Model,
    MultiStageTraining,
    OutputStage,
    TrainingStage,
)
from .HCR import GBNSchedule, HCRBenchmarks, _amp_autocast

if TYPE_CHECKING:
    from src.storage.eos import PathLike

_ARCH_KEYS = ("n_features", "attention", "use_attention_gate", "use_kv_proj")
_FINETUNE_PREFIX = "head_main."  # the output layer: HCR's fine-tuning freeze set (parity test)


def _default_inputs():
    return {"b": Input.CanJet, "other": Input.NotCanJet}


@dataclass
class CoReNetArch:
    __skip_save = frozenset(("loss",))

    loss: Callable[[BatchType], Tensor] = None
    network: dict = None
    inputs: dict[str, str] = field(default_factory=_default_inputs)
    architecture: dict = None
    resolved: dict = None  # filled by the training from resolve_spec

    @classmethod
    def load(cls, saved: dict[str]):
        obj = cls()
        for k, v in saved.items():
            if k in cls.__annotations__:
                setattr(obj, k, v)
        return obj

    def save(self):
        return {
            k: getattr(self, k)
            for k in self.__annotations__
            if k not in self.__skip_save
        }


def resolve_spec(
    arch: CoReNetArch, labels: Iterable[str], event: Iterable[str]
) -> tuple[NetworkSpec, dict]:
    """The event description of ``arch`` completed by barista: ``event``, ``labels``, the single
    placeholder-free output and the ``architecture`` overrides. Returns ``(parse_config(spec), spec)``."""
    if not isinstance(arch.network, dict):
        raise ValueError(f"CoReNet: the event description must be a mapping, got {arch.network!r}")
    spec = copy.deepcopy(arch.network)
    labels, event = list(labels), list(event)
    for key in ("event", "labels"):
        if key in spec:
            raise ValueError(
                f"CoReNet: the event description sets '{key}:'; it is filled by barista "
                "(event = InputBranch.feature_ancillary, labels = MultiClass.trainable_labels)"
            )
    if not labels:
        raise ValueError("CoReNet: no trainable labels")
    inputs = dict(arch.inputs or {})
    if not inputs:
        raise ValueError("CoReNet: 'inputs' (role -> batch key) is empty")
    objects = spec.get("objects") or {}
    roles = [name for name, body in objects.items() if isinstance(body, str)]
    first_input = next(iter(inputs))
    if not roles or roles[0] != first_input:
        raise ValueError(
            f"CoReNet: the first role of the event description ({roles[0] if roles else None!r}) must be "
            f"the first key of inputs ({first_input!r}): its leading slot fixes the eta reflection, as the "
            "leading candidate jet does in HCR"
        )
    outputs = spec.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("CoReNet: the event description needs an 'outputs' mapping")
    spec["outputs"] = {labels[0]: next(iter(outputs.values()))}
    spec["event"] = event
    spec["labels"] = labels
    overrides = dict(arch.architecture or {})
    unknown = sorted(set(overrides) - set(_ARCH_KEYS))
    if unknown:
        raise ValueError(f"CoReNet: unknown architecture keys {unknown} (allowed: {list(_ARCH_KEYS)})")
    training = dict(spec.get("training") or {})
    if "n_features" in overrides:
        training["features"] = overrides["n_features"]
    if "use_attention_gate" in overrides:
        training["attention_gate"] = overrides["use_attention_gate"]
    if "use_kv_proj" in overrides:
        training["kv_projection"] = overrides["use_kv_proj"]
    spec["training"] = training
    if overrides.get("attention", True) is False:
        for body in (spec.get("hypotheses") or {}).values():
            if isinstance(body, dict):
                body.pop("context", None)
    parsed = parse_config(spec)
    missing = [r for r in parsed.roles if r not in inputs]
    unknown = [r for r in inputs if r not in parsed.leaves()]
    if missing or unknown:
        raise ValueError(
            f"CoReNet: inputs {inputs} do not match the roles {list(parsed.roles)} "
            f"(missing: {missing}, not a role: {unknown})"
        )
    return parsed, spec


def _CoReNetInput(
    batch: BatchType,
    device: tt.Device,
    spec: NetworkSpec,
    inputs: dict[str, str],
    selection: Tensor = None,
) -> dict[str, Tensor]:
    for k, v in batch.items():
        batch[k] = v.to(device, non_blocking=True)
    ancillary = batch[Input.ancillary]
    n = ancillary.shape[0]
    out: dict[str, Tensor] = {}
    for role, key in inputs.items():
        n_slots = spec.roles[role].n
        n_feat = len(spec.collection_of(role).features)
        x = batch[key].view(n, -1, n_slots)
        if x.shape[1] == n_feat:
            x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        elif x.shape[1] != n_feat + 1:
            raise ValueError(
                f'CoReNet input "{role}" from batch key "{key}": shape {tuple(batch[key].shape)} viewed as '
                f"{tuple(x.shape)}; expected {n_feat} or {n_feat + 1} channels of {n_slots} slots"
            )
        out[role] = x
    out["event"] = ancillary.view(n, -1)
    if selection is not None:
        selection = selection.to(device, non_blocking=True)
        out = {k: v[selection] for k, v in out.items()}
    return out


class _CoReNetSkim(Skimmer):
    def __init__(
        self,
        nn: CoReNet,
        device: tt.Device,
        splitter: Splitter,
        spec: NetworkSpec = None,
        inputs: dict[str, str] = None,
    ):
        self._nn = nn
        self._device = device
        self._splitter = splitter
        self._spec = spec
        self._inputs = inputs

    @torch.no_grad()
    def train(self, batch: BatchType):
        selections = self._splitter.step(batch)
        if self._nn is not None and selections[SplitterKeys.training].sum() > 0:
            self._nn.updateMeanStd(
                _CoReNetInput(
                    batch,
                    self._device,
                    self._spec,
                    self._inputs,
                    selections[SplitterKeys.training],
                )
            )
        return super().train(batch)


class CoReNetModel(Model):
    def __init__(
        self,
        device: tt.Device,
        arch: CoReNetArch,
        benchmarks: HCRBenchmarks,
        spec: NetworkSpec,
        resolved: dict,
    ):
        self._loss = arch.loss
        self._device = device
        self._gbn = None
        self._arch = arch
        self._spec = spec
        self._resolved = resolved
        self._inputs = dict(arch.inputs)
        self._nn = CoReNet(spec, device=device)
        self._main = spec.outputs[0].hypothesis
        self._benchmarks = benchmarks

    @property
    def ghost_batch(self):
        return self._gbn

    @ghost_batch.setter
    def ghost_batch(self, gbn: GBNSchedule):
        self._gbn = gbn
        if gbn is None:
            self._nn.setGhostBatches(0)
        else:
            self._gbn.reset()
            self._nn.setGhostBatches(self._gbn.n_batches)

    @property
    def hyperparameters(self) -> dict[str]:
        return {
            "n ghost batch": (
                self.ghost_batch.get_last_bs() if self.ghost_batch is not None else 0
            )
        }

    @property
    def nn(self):
        return self._nn

    def _forward(self, batch: BatchType):
        out = self._nn(_CoReNetInput(batch, self._device, self._spec, self._inputs))
        return out["logits"], out["views"].get(self._main)

    def train(self, batch: BatchType) -> Tensor:
        with _amp_autocast(self._device):
            c, q = self._forward(batch)
            batch[Output.class_raw] = c
            if q is not None:
                batch[Output.quadjet_raw] = q
            return self._loss(batch)

    def validate(self, batches: Iterable[BatchType]) -> dict[str]:
        weight = 0.0
        scalars = defaultdict(float)
        scalar_funcs = self._benchmarks.scalars
        rocs = [r.copy() for r in self._benchmarks.rocs]
        for batch in batches:
            with _amp_autocast(self._device):
                c, q = self._forward(batch)
            c = c.float()
            batch |= {
                Output.class_raw: c,
                Output.class_prob: F.softmax(c, dim=1),
            }
            if q is not None:
                batch[Output.quadjet_raw] = q.float()
            sumw = to_num(batch[Input.weight].sum())
            if scalar_funcs is None:
                if MultiClass.n_nontrainable() == 0:
                    scalars["loss"] += to_num(self._loss(batch)) * sumw
            else:
                for func in scalar_funcs:
                    measured = func(batch)
                    for name, value in measured.items():
                        scalars[name] += to_num(value) * sumw
            weight += sumw
            for roc in rocs:
                roc.update(batch)
        for k in scalars:
            scalars[k] /= weight
        return {"scalars": scalars, "roc": [r.to_json() for r in rocs]}

    def step(self, epoch: int = None):
        if self.ghost_batch is not None and self.ghost_batch.step(epoch):
            self._nn.setGhostBatches(self.ghost_batch.get_bs())


class CoReNetTraining(MultiStageTraining):
    def __init__(
        self,
        arch: CoReNetArch,
        ghost_batch: GBNSchedule,
        cross_validation: Splitter,
        training_schedule: Schedule,
        finetuning_schedule: Schedule = None,
        benchmarks: HCRBenchmarks = None,
        pretrained_weights: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._arch = arch
        self._ghost_batch = ghost_batch
        self._splitter = cross_validation
        self._training = training_schedule
        self._finetuning = finetuning_schedule
        self._benchmarks = benchmarks or HCRBenchmarks()
        self._pretrained_weights = pretrained_weights
        self._CoReNet: CoReNetModel = None

    def stages(self):
        spec, resolved = resolve_spec(
            self._arch, MultiClass.trainable_labels, InputBranch.feature_ancillary
        )
        self._arch.resolved = resolved
        self._CoReNet = CoReNetModel(
            device=self.device,
            arch=self._arch,
            benchmarks=self._benchmarks,
            spec=spec,
            resolved=resolved,
        )
        self._CoReNet.ghost_batch = self._ghost_batch
        self._CoReNet.to(self.device)
        if self._pretrained_weights is not None:
            if self._pretrained_weights.endswith(".pkl"):
                path = self._pretrained_weights
            else:
                import glob as _glob

                offset = self.metadata.get("offset", 0)
                pattern = f"{self._pretrained_weights}/*_offset{offset}_*.pkl"
                matches = _glob.glob(pattern)
                if not matches:
                    raise FileNotFoundError(
                        f"No pretrained weights found for offset {offset} in "
                        f"{self._pretrained_weights}. Pattern: {pattern}"
                    )
                if len(matches) > 1:
                    raise FileNotFoundError(
                        f"Ambiguous pretrained weights for offset {offset} in "
                        f"{self._pretrained_weights}: {matches}. "
                        f"Use a more specific directory or pass a direct .pkl path."
                    )
                path = matches[0]
            logging.info(f"Warm-starting from {path}")
            with fsspec.open(path, "rb") as f:
                saved = torch.load(f, map_location=self.device, weights_only=False)
            self._CoReNet.nn.load_state_dict(saved["model"])
        self._splitter.setup(self.dataset)
        skim = _CoReNetSkim(
            self._CoReNet.nn,
            self.device,
            self._splitter,
            spec=spec,
            inputs=self._CoReNet._inputs,
        )
        yield TrainingStage(
            name="Initialization",
            model=skim,
            schedule=SkimStep(),
            training=self.dataset,
        )
        self._CoReNet.nn.initMeanStd()
        validation_sets = self._splitter.get()
        training_set = validation_sets[SplitterKeys.training]
        yield BenchmarkStage(
            name="Baseline",
            model=self._CoReNet,
            validation=validation_sets,
        )
        yield TrainingStage(
            name="Training",
            model=self._CoReNet,
            schedule=self._training,
            training=training_set,
            validation=validation_sets,
        )
        self._CoReNet.ghost_batch = None
        if self._finetuning is not None:
            requires_grad = {
                name: p.requires_grad for name, p in self._CoReNet.nn.named_parameters()
            }
            for name, p in self._CoReNet.nn.named_parameters():
                if not name.startswith(_FINETUNE_PREFIX):
                    p.requires_grad = False
            yield TrainingStage(
                name="Finetuning",
                model=self._CoReNet,
                schedule=self._finetuning,
                training=training_set,
                validation=validation_sets,
            )
            self._CoReNet.ghost_batch = self._ghost_batch
            for name, p in self._CoReNet.nn.named_parameters():
                p.requires_grad = requires_grad[name]
        output_stage = OutputStage(name="Final", path=f"{self.name}__{self.uuid}.pkl")
        output_path = output_stage.absolute_path
        if not output_path.is_null:
            logging.info(f"Saving model to {output_path}")
            with fsspec.open(output_path, "wb") as f:
                torch.save(
                    {
                        "model": self._CoReNet.nn.state_dict(),
                        "metadata": self.metadata,
                        "uuid": self.uuid,
                        "label": MultiClass.trainable_labels,
                        "arch": self._arch.save(),
                        "input": {
                            k: getattr(InputBranch, k)
                            for k in (
                                "feature_ancillary",
                                "feature_CanJet",
                                "feature_NotCanJet",
                                "n_NotCanJet",
                            )
                        },
                    },
                    MemoryViewIO(f),
                )
            yield output_stage


def view_names(nn: CoReNet, hypothesis: str) -> list[str] | None:
    """Column names of the view probabilities of ``hypothesis`` in view-logit order, or None when it
    has a single view. One role over all views: ``q_`` + the 1-based slots of the view in order
    (HH -> 4b: q_1234, q_1324, q_1423); otherwise ``q_<hypothesis>_<v>``."""
    if hypothesis not in nn.selector:
        return None
    T = nn.tables[hypothesis]
    views = T.views
    if len(views) <= 1:
        return None
    roles = {role for view in views for assignment in view for role, _ in assignment}
    names: list[str] = [None] * len(views)
    for v, view in enumerate(views):
        column = int(T.view_index[v])
        if len(roles) == 1:
            names[column] = "q_" + "".join(
                str(slot + 1) for assignment in view for _, slot in assignment
            )
        else:
            names[column] = f"q_{hypothesis}_{v}"
    return names


class CoReNetModelEval(Model):
    def __init__(
        self,
        device: tt.Device,
        saved: dict[str],
        splitter: Splitter,
        mapping: Callable[[BatchType], BatchType],
    ):
        self._device = device
        self._splitter = splitter
        self._mapping = mapping
        self._classes = saved["label"]
        for k in saved["input"].keys():
            if getattr(InputBranch, k) != saved["input"][k]:
                raise ValueError(
                    f'Input features "{k}" mismatch: training={saved["input"][k]}, evaluation={getattr(InputBranch, k)}'
                )
        self._arch = CoReNetArch.load(saved["arch"])
        self._spec = parse_config(copy.deepcopy(saved["arch"]["resolved"]))
        if list(self._spec.classes) != list(self._classes):
            raise ValueError(
                f"CoReNet checkpoint: spec classes {self._spec.classes} differ from labels {self._classes}"
            )
        self._inputs = dict(self._arch.inputs)
        self._nn = CoReNet(self._spec, device=device)
        self._nn.load_state_dict(saved["model"])
        self._main = self._spec.outputs[0].hypothesis
        self._views = view_names(self._nn, self._main)

    @property
    def nn(self):
        return self._nn

    def evaluate(self, batch: BatchType) -> BatchType:
        selection = self._splitter.split(batch)[SplitterKeys.validation]
        selector = Selector(selection)
        with _amp_autocast(self._device):
            out = self._nn(
                _CoReNetInput(batch, self._device, self._spec, self._inputs, selection)
            )
        c = out["logits"].float()
        c_prob = F.softmax(c, dim=1).cpu()
        output = {}
        if self._views is not None:
            q_prob = F.softmax(out["views"][self._main].float(), dim=1).cpu()
            for i, name in enumerate(self._views):
                output[name] = q_prob[:, i]
        for i, label in enumerate(self._classes):
            output[f"p_{label}"] = c_prob[:, i]
        return selector.pad(map_batch(self._mapping, output))


class CoReNetEvaluation(Evaluation):
    def __init__(
        self,
        saved_model: PathLike,
        cross_validation: Splitter,
        output_definition: Callable[[BatchType], BatchType],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model = saved_model
        self._splitter = cross_validation
        self._mapping = output_definition

    def stages(self):
        with fsspec.open(self._model, "rb") as f:
            load_kw = {}
            if self.device.type == "cpu":
                load_kw["map_location"] = torch.device("cpu")
            saved = torch.load(f, weights_only=False, **load_kw)
        self._CoReNet = CoReNetModelEval(
            device=self.device,
            saved=saved,
            splitter=self._splitter,
            mapping=self._mapping,
        )
        self._CoReNet.to(self.device)
        yield EvaluationStage(
            name="Evaluation",
            model=self._CoReNet,
            dataset=self.dataset,
            dumper_kwargs={"name": self.name},
        )
