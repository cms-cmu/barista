from __future__ import annotations

import logging

import fsspec
import torch
import torch.nn.functional as F
from classifier.config.scheduler import SkimStep
from classifier.config.setting.HCR import Input, InputBranch, Output
from classifier.config.setting.ml import SplitterKeys
from classifier.config.state.label import MultiClass

from ...utils import MemoryViewIO
from .. import BatchType
from ..skimmer import Filter
from ..training import BenchmarkStage, OutputStage, TrainingStage
from .HCR import HCRModel, HCRTraining, _HCRSkim


def _remove_ZX(batch: BatchType):
    label = batch[Input.label]
    return ~torch.isin(label, label.new_tensor(MultiClass.indices("ZZ", "ZH")))


class loss_remove:
    def __init__(self, *labels: str):
        self.labels = labels

    def __call__(self, batch: BatchType):
        c_score = batch[Output.class_raw]
        weight = batch[Input.weight]
        label = batch[Input.label]
        idxs = sorted(MultiClass.indices(*self.labels))
        sliced = [i for i in range(MultiClass.n_trainable()) if i not in idxs]
        label = label.clone()
        for idx in idxs:
            label[label >= idx] -= 1
        # calculate loss
        cross_entropy = F.cross_entropy(c_score[:, sliced], label, reduction="none")
        loss = (cross_entropy * weight).sum() / weight.sum()
        return loss


class SvBTraining(HCRTraining):
    def stages(self):
        finetune_splitter = self._splitter + Filter(
            **{SplitterKeys.training: _remove_ZX}
        )
        self._HCR = HCRModel(
            device=self.device,
            arch=self._arch,
            benchmarks=self._benchmarks,
        )
        self._HCR.ghost_batch = self._ghost_batch
        self._HCR.to(self.device)
        # layers = self._HCR._nn.layers
        self._splitter.setup(self.dataset)
        yield TrainingStage(
            name="Initialization",
            model=_HCRSkim(self._HCR._nn, self.device, self._splitter),
            schedule=SkimStep(),
            training=self.dataset,
        )
        self._HCR.nn.initMeanStd()
        training_sets = self._splitter.get()
        yield BenchmarkStage(
            name="Baseline",
            model=self._HCR,
            validation=training_sets,
        )
        yield TrainingStage(
            name="Training",
            model=self._HCR,
            schedule=self._training,
            training=training_sets[SplitterKeys.training],
            validation=training_sets,
        )
        if self._finetuning is not None:
            finetune_splitter.setup(self.dataset)
            yield TrainingStage(
                name="Initialize Fintune",
                model=_HCRSkim(None, self.device, finetune_splitter),
                schedule=SkimStep(),
                training=self.dataset,
            )
            training_sets = finetune_splitter.get()
            self._HCR._loss = loss_remove("ZZ", "ZH")
            self._HCR.ghost_batch = None
            # layers.setLayerRequiresGrad(
            #     requires_grad=False, index=self._HCR._nn.embedding_layers()
            # )
            yield TrainingStage(
                name="Finetune ggF",
                model=self._HCR,
                schedule=self._finetuning,
                training=training_sets[SplitterKeys.training],
                validation=training_sets,
            )
            self._HCR.ghost_batch = self._ghost_batch
            # layers.setLayerRequiresGrad(requires_grad=True)
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
