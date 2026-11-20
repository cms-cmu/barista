"""Training and evaluation of the HCR classifier.

The network is ``src.classifier.nn.blocks.HCR`` by default. With ``HCRArch.network`` set to a CoReNet
event description, the network is a ``CoReNet`` from the ``corenet`` package
(https://github.com/cms-cmu/CoReNet) and everything else (stages, schedules, ghost batches, benchmarks,
checkpoint layout, output columns) is unchanged. ``corenet`` is imported only in that case.
"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable

import fsspec
import torch
import torch.nn.functional as F
import torch.types as tt
from src.classifier.config.scheduler import SkimStep
from src.classifier.config.setting.HCR import Input, InputBranch, Output
from src.classifier.config.setting.ml import SplitterKeys, Training
from src.classifier.config.state.label import MultiClass
from torch import Tensor

from ...algorithm.utils import Selector, map_batch, to_num
from ...nn.blocks.HCR import HCR
from ...nn.schedule import MilestoneStep, Schedule
from ...utils import MemoryViewIO
from .. import BatchType
from ..benchmarks.multiclass import ROC
from ..evaluation import Evaluation, EvaluationStage
from ..skimmer import Skimmer, Splitter
from ..training import (
    BenchmarkStage,
    Model,
    MultiStageTraining,
    OutputStage,
    TrainingStage,
)

if TYPE_CHECKING:
    from src.storage.eos import PathLike


def _amp_autocast(device):
    """Mixed-precision autocast context, set by cfg.Training.precision on CUDA only.

    '' (default) returns a nullcontext -> exact fp32 behaviour. 'fp16'/'bf16' wrap the
    forward passes (train/validate/evaluate) to ~halve activation memory; the GBN
    running-stats update (updateMeanStd) is intentionally left outside. bf16 keeps the
    fp32 exponent range (no overflow) and is the choice for very wide nets.
    """
    p = Training.precision
    if p in ("fp16", "bf16") and getattr(device, "type", str(device)) == "cuda":
        dtype = torch.bfloat16 if p == "bf16" else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


@dataclass
class HCRArch:
    __skip_save = frozenset(("loss",))

    loss: Callable[[BatchType], Tensor] = None
    n_features: int = 8
    attention: bool = True
    use_attention_gate: bool = False
    use_kv_proj: bool = False
    # CoReNet instead of HCR: used only when ``network`` is set (see ``_CoReNetNetwork``)
    network: dict = None  # CoReNet event description (objects, hypotheses, outputs, ...)
    inputs: dict = None  # role -> batch key; default {"b": Input.CanJet, "other": Input.NotCanJet}
    architecture: dict = None  # overrides of the event description, keys of _CORENET_ARCH_KEYS
    resolved: dict = None  # the completed event description, filled by the training

    @classmethod
    def load(cls, saved: dict[str]):
        obj = cls()
        for k, v in saved.items():
            if k in cls.__annotations__:
                setattr(obj, k, v)
        return obj

    def save(self):
        corenet = self.network is not None
        return {
            k: getattr(self, k)
            for k in self.__annotations__
            if k not in self.__skip_save and (k in _CORENET_ARCH_FIELDS) == corenet
        }


@dataclass
class GBNSchedule(MilestoneStep):
    n_batches: int = 64
    milestones: list[int] = (1, 3, 6, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
    gamma: float = 0.25

    def __post_init__(self):
        self.milestones = sorted(self.milestones)
        self._last_bs = self.n_batches
        self.reset()

    def get_bs(self):
        self._last_bs = int(self.n_batches * (self.gamma**self.milestone))
        return self._last_bs

    def get_last_bs(self):
        return self._last_bs


@dataclass
class HCRBenchmarks:
    rocs: Iterable[ROC] = ()
    scalars: Iterable[Callable[[BatchType], dict[str, Tensor]]] = None
    fit_quality: Iterable = ()


def _HCRInput(batch: BatchType, device: tt.Device, selection: Tensor = None):
    for k, v in batch.items():
        batch[k] = v.to(device, non_blocking=True)
    inputs = [batch[k] for k in (Input.CanJet, Input.NotCanJet, Input.ancillary)]
    if selection is not None:
        selection = selection.to(device, non_blocking=True)
        inputs = [i[selection] for i in inputs]
    return inputs


_CORENET_ARCH_FIELDS = frozenset(("network", "inputs", "architecture", "resolved"))
_CORENET_ARCH_KEYS = ("n_features", "attention", "use_attention_gate", "use_kv_proj")
_CORENET_FINETUNE_PREFIX = "head_main."  # the output layer: what HCR's fine-tuning leaves trainable


def resolve_corenet_spec(arch: HCRArch, labels: Iterable[str], event: Iterable[str]):
    """The CoReNet event description of ``arch`` completed by barista. Returns
    ``(corenet NetworkSpec, completed mapping)``.

    ``event:`` and ``labels:`` must not be set in ``arch.network``: they are filled with ``event``
    (InputBranch.feature_ancillary, in order) and ``labels`` (MultiClass.trainable_labels). ``outputs``
    is rewritten to the single entry ``{labels[0]: <hypothesis of the first output>}``.
    ``arch.architecture`` overrides: ``n_features`` -> ``training.features``, ``use_attention_gate`` ->
    ``training.attention_gate``, ``use_kv_proj`` -> ``training.kv_projection``, ``attention: false``
    removes every ``context`` of the hypotheses. The first role of the event description must be the
    first key of ``arch.inputs``: its leading slot fixes the eta reflection, as the leading candidate
    jet does in HCR.
    """
    from corenet.models.config import parse_config

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
    if arch.inputs is None:
        arch.inputs = {"b": Input.CanJet, "other": Input.NotCanJet}
    inputs = dict(arch.inputs)
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
    unknown = sorted(set(overrides) - set(_CORENET_ARCH_KEYS))
    if unknown:
        raise ValueError(
            f"CoReNet: unknown architecture keys {unknown} (allowed: {list(_CORENET_ARCH_KEYS)})"
        )
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


class _CoReNetNetwork:
    """A ``CoReNet`` network and how barista batches go in and its outputs come out.

    Batch layout: for each (role, key) of ``inputs`` the tensor ``batch[key]`` is viewed as
    ``(n, channels, slots)`` with ``slots = spec.roles[role].n`` (the layout HCR uses: CanJet ->
    (n, n_features, 4), NotCanJet -> (n, 5, 8)). With as many channels as the role's collection has
    features a presence channel of +1 is appended; with one more, the last one is the flag as given;
    any other count is an error. The event scalars are ``batch[Input.ancillary]``.
    """

    def __init__(self, device: tt.Device, inputs: dict[str, str], spec):
        from corenet.models.network import CoReNet

        self.spec = spec
        self.roles = dict(inputs)
        self.nn = CoReNet(spec, device=device)
        self.main = spec.outputs[0].hypothesis

    def inputs(self, batch: BatchType, device: tt.Device, selection: Tensor = None):
        for k, v in batch.items():
            batch[k] = v.to(device, non_blocking=True)
        ancillary = batch[Input.ancillary]
        n = ancillary.shape[0]
        out: dict[str, Tensor] = {}
        for role, key in self.roles.items():
            n_slots = self.spec.roles[role].n
            n_feat = len(self.spec.collection_of(role).features)
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

    def forward(self, inputs: dict[str, Tensor]):
        """(class logits, view logits of the main hypothesis or None when it has a single view)"""
        out = self.nn(inputs)
        return out["logits"], out["views"].get(self.main)

    def view_names(self) -> list[str] | None:
        """Column names of the view probabilities in view-logit order, or None for a single view.
        One role over all views: ``q_`` + the 1-based slots of the view in order (HH -> 4b: q_1234,
        q_1324, q_1423, as HCR); otherwise ``q_<hypothesis>_<v>``."""
        if self.main not in self.nn.selector:
            return None
        T = self.nn.tables[self.main]
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
                names[column] = f"q_{self.main}_{v}"
        return names

    def freeze_for_finetuning(self) -> dict[str, bool]:
        """Freeze everything except the output layer; returns the previous state for ``unfreeze``."""
        previous = {name: p.requires_grad for name, p in self.nn.named_parameters()}
        for name, p in self.nn.named_parameters():
            if not name.startswith(_CORENET_FINETUNE_PREFIX):
                p.requires_grad = False
        return previous

    def unfreeze(self, previous: dict[str, bool]):
        for name, p in self.nn.named_parameters():
            p.requires_grad = previous[name]


class _HCRSkim(Skimmer):
    def __init__(
        self,
        nn: HCR,
        device: tt.Device,
        splitter: Splitter,
        corenet: _CoReNetNetwork = None,
    ):
        self._nn = nn
        self._device = device
        self._splitter = splitter
        self._corenet = corenet

    @torch.no_grad()
    def train(self, batch: BatchType):
        selections = self._splitter.step(batch)
        if self._nn is not None and selections[SplitterKeys.training].sum() > 0:
            if self._corenet is None:
                self._nn.updateMeanStd(
                    *_HCRInput(batch, self._device, selections[SplitterKeys.training])
                )
            else:
                self._nn.updateMeanStd(
                    self._corenet.inputs(
                        batch, self._device, selections[SplitterKeys.training]
                    )
                )
        return super().train(batch)


class HCRModel(Model):
    def __init__(
        self,
        device: tt.Device,
        arch: HCRArch,
        benchmarks: HCRBenchmarks,
        corenet_spec=None,
    ):
        self._loss = arch.loss
        self._device = device
        self._gbn = None
        self._arch = arch
        self._corenet: _CoReNetNetwork = None
        if arch.network is None:
            self._nn = HCR(
                dijetFeatures=arch.n_features,
                quadjetFeatures=arch.n_features,
                ancillaryFeatures=InputBranch.feature_ancillary,
                canJetFeatures=InputBranch.feature_CanJet,
                useOthJets=("attention" if arch.attention else ""),
                device=device,
                nClasses=MultiClass.n_trainable(),
                use_attention_gate=arch.use_attention_gate,
                use_kv_proj=arch.use_kv_proj,
            )
        else:
            self._corenet = _CoReNetNetwork(device, arch.inputs, corenet_spec)
            self._nn = self._corenet.nn
        self._benchmarks = benchmarks

    def _set_ghost_batches(self, n: int):
        if self._corenet is None:
            self._nn.setGhostBatches(n, False)
        else:
            self._nn.setGhostBatches(n)

    def _inputs(self, batch: BatchType):
        if self._corenet is None:
            return _HCRInput(batch, self._device)
        return self._corenet.inputs(batch, self._device)

    def _forward(self, inputs):
        """(class logits, quadjet/view logits); the second is None for a CoReNet with a single view"""
        if self._corenet is None:
            return self._nn(*inputs)
        return self._corenet.forward(inputs)

    @property
    def ghost_batch(self):
        return self._gbn

    @ghost_batch.setter
    def ghost_batch(self, gbn: GBNSchedule):
        self._gbn = gbn
        if gbn is None:
            self._set_ghost_batches(0)
        else:
            self._gbn.reset()
            self._set_ghost_batches(self._gbn.n_batches)

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

    def train(self, batch: BatchType) -> Tensor:
        with _amp_autocast(self._device):
            c, q = self._forward(self._inputs(batch))
            if q is not None:
                batch[Output.quadjet_raw] = q
            batch[Output.class_raw] = c
            return self._loss(batch)

    def validate(self, batches: Iterable[BatchType]) -> dict[str]:
        weight = 0.0
        scalars = defaultdict(float)
        scalar_funcs = self._benchmarks.scalars
        rocs = [r.copy() for r in (self._benchmarks.rocs or [])]
        fit_qualities = [f.copy() for f in (self._benchmarks.fit_quality or [])]
        for batch in batches:
            inputs = self._inputs(batch)
            if fit_qualities:
                # HCR.dataPrep rewrites the jet tensors in place (log(1+pt), eta sign); the
                # fit-quality benchmark reads the original kinematics from the batch
                if self._corenet is None:
                    inputs = [i.clone() for i in inputs]
                else:
                    inputs = {k: v.clone() for k, v in inputs.items()}
            with _amp_autocast(self._device):
                c, q = self._forward(inputs)
            c = c.float()
            if q is not None:
                batch[Output.quadjet_raw] = q.float()
            batch |= {
                Output.class_raw: c,
                Output.class_prob: F.softmax(c, dim=1),
            }
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
            for fq in fit_qualities:
                fq.update(batch)
        for k in scalars:
            scalars[k] /= weight
        res = {"scalars": scalars, "roc": [r.to_json() for r in rocs]}
        if fit_qualities:
            res["fit_quality"] = [fq.to_json() for fq in fit_qualities]
        return res

    def step(self, epoch: int = None):
        if self.ghost_batch is not None and self.ghost_batch.step(epoch):
            self._set_ghost_batches(self.ghost_batch.get_bs())


class HCRTraining(MultiStageTraining):
    def __init__(
        self,
        arch: HCRArch,
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
        self._HCR: HCRModel = None

    def stages(self):
        corenet_spec = None
        if self._arch.network is not None:
            corenet_spec, self._arch.resolved = resolve_corenet_spec(
                self._arch, MultiClass.trainable_labels, InputBranch.feature_ancillary
            )
        self._HCR = HCRModel(
            device=self.device,
            arch=self._arch,
            benchmarks=self._benchmarks,
            corenet_spec=corenet_spec,
        )
        self._HCR.ghost_batch = self._ghost_batch
        self._HCR.to(self.device)
        if self._pretrained_weights is not None:
            if self._pretrained_weights.endswith(".pkl"):
                path = self._pretrained_weights
            else:
                offset = self.metadata.get("offset", 0)
                # saved models are named model_<...>__offset_{N}__<...>.pkl (filename(metadata));
                # also accept the compact _offset{N}_ form. Resolved through fsspec, so the
                # directory can be remote (root://, e.g. an earlier roast's classifier/<label>)
                # as well as local; matches (files directly in the directory) are returned as
                # URLs under the given directory, whatever path form the filesystem's glob uses.
                url = self._pretrained_weights.rstrip("/")
                fs, root = fsspec.core.url_to_fs(url)
                root = root.rstrip("/")
                names = [f"*__offset_{offset}__*.pkl", f"*_offset{offset}_*.pkl"]
                pattern = " | ".join(f"{url}/{n}" for n in names)
                matches = sorted(
                    {f"{url}/{m.rstrip('/').rsplit('/', 1)[-1]}" for n in names for m in fs.glob(f"{root}/{n}")}
                )
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
            self._HCR.nn.load_state_dict(saved["model"])
        self._splitter.setup(self.dataset)
        corenet = self._HCR._corenet
        skim = _HCRSkim(self._HCR._nn, self.device, self._splitter, corenet=corenet)
        yield TrainingStage(
            name="Initialization",
            model=skim,
            schedule=SkimStep(),
            training=self.dataset,
            # the skim accumulates the input mean/std (not in any state dict): never resume it
            checkpoint=False,
        )
        self._HCR.nn.initMeanStd()
        validation_sets = self._splitter.get()
        training_set = validation_sets[SplitterKeys.training]
        yield BenchmarkStage(
            name="Baseline",
            model=self._HCR,
            validation=validation_sets,
        )
        yield TrainingStage(
            name="Training",
            model=self._HCR,
            schedule=self._training,
            training=training_set,
            validation=validation_sets,
        )
        self._HCR.ghost_batch = None
        if self._finetuning is not None:
            if corenet is None:
                layers = self._HCR._nn.layers
                layers.setLayerRequiresGrad(
                    requires_grad=False, index=self._HCR._nn.embedding_layers()
                )
            else:
                requires_grad = corenet.freeze_for_finetuning()
            yield TrainingStage(
                name="Finetuning",
                model=self._HCR,
                schedule=self._finetuning,
                training=training_set,
                validation=validation_sets,
            )
            self._HCR.ghost_batch = self._ghost_batch
            if corenet is None:
                layers.setLayerRequiresGrad(requires_grad=True)
            else:
                corenet.unfreeze(requires_grad)
        output_stage = OutputStage(name="Final", path=f"{self.name}__{self.uuid}.pkl")
        output_path = output_stage.absolute_path
        if not output_path.is_null:
            logging.info(f"Saving model to {output_path}")
            with fsspec.open(output_path, "wb") as f:
                torch.save(
                    {
                        "model": self._HCR.nn.state_dict(),
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


class HCRModelEval(Model):
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
        self._arch = HCRArch.load(saved["arch"])
        self._corenet: _CoReNetNetwork = None
        if self._arch.network is None:
            self._nn = HCR(
                dijetFeatures=self._arch.n_features,
                quadjetFeatures=self._arch.n_features,
                ancillaryFeatures=InputBranch.feature_ancillary,
                canJetFeatures=InputBranch.feature_CanJet,
                useOthJets=("attention" if self._arch.attention else ""),
                device=device,
                nClasses=len(self._classes),
                use_attention_gate=self._arch.use_attention_gate,
                use_kv_proj=self._arch.use_kv_proj,
            )
        else:
            from corenet.models.config import parse_config

            # the checkpoint carries the completed event description: no file is read here
            spec = parse_config(copy.deepcopy(self._arch.resolved))
            if list(spec.classes) != list(self._classes):
                raise ValueError(
                    f"CoReNet checkpoint: spec classes {spec.classes} differ from labels {self._classes}"
                )
            self._corenet = _CoReNetNetwork(device, self._arch.inputs, spec)
            self._nn = self._corenet.nn
        self._nn.load_state_dict(saved["model"])
        self._views = None if self._corenet is None else self._corenet.view_names()

    @property
    def nn(self):
        return self._nn

    def evaluate(self, batch: BatchType) -> BatchType:
        selection = self._splitter.split(batch)[SplitterKeys.validation]
        selector = Selector(selection)
        if self._corenet is not None:
            return selector.pad(self._evaluate_corenet(batch, selection))
        with _amp_autocast(self._device):
            c, q = self._nn(*_HCRInput(batch, self._device, selection))
        c, q = c.float(), q.float()
        c_prob = F.softmax(c, dim=1).cpu()
        q_prob = F.softmax(q, dim=1).cpu()
        output = {
            "q_1234": q_prob[:, 0],
            "q_1324": q_prob[:, 1],
            "q_1423": q_prob[:, 2],
        }
        for i, label in enumerate(self._classes):
            output[f"p_{label}"] = c_prob[:, i]
        return selector.pad(map_batch(self._mapping, output))

    def _evaluate_corenet(self, batch: BatchType, selection: Tensor) -> BatchType:
        with _amp_autocast(self._device):
            c, q = self._corenet.forward(
                self._corenet.inputs(batch, self._device, selection)
            )
        c_prob = F.softmax(c.float(), dim=1).cpu()
        output = {}
        if self._views is not None:
            q_prob = F.softmax(q.float(), dim=1).cpu()
            for i, name in enumerate(self._views):
                output[name] = q_prob[:, i]
        for i, label in enumerate(self._classes):
            output[f"p_{label}"] = c_prob[:, i]
        return map_batch(self._mapping, output)


class HCREvaluation(Evaluation):
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
        self._HCR = HCRModelEval(
            device=self.device,
            saved=saved,
            splitter=self._splitter,
            mapping=self._mapping,
        )
        self._HCR.to(self.device)
        yield EvaluationStage(
            name="Evaluation",
            model=self._HCR,
            dataset=self.dataset,
            dumper_kwargs={"name": self.name},
        )
