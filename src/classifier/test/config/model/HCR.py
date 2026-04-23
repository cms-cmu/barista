from __future__ import annotations

from typing import TYPE_CHECKING

from src.classifier.config.model.HCR._HCR import (
    ROC_BIN,
    HCRTrain,
    HCREval,
    roc_nominal_selection,
)
from src.classifier.config.setting.HCR import Input, Output

if TYPE_CHECKING:
    from src.classifier.ml import BatchType


class SparseDenseTrain(HCRTrain):
    model = "test-SvD"

    @staticmethod
    def loss(batch: BatchType):
        import torch.nn.functional as F

        c_score = batch[Output.class_raw]
        weight = batch[Input.weight]
        label = batch[Input.label]

        # calculate loss
        cross_entropy = F.cross_entropy(c_score, label, reduction="none")
        loss = (cross_entropy * weight).sum() / weight.sum()
        return loss

    @property
    def rocs(self):
        from src.classifier.ml.benchmarks.multiclass import ROC

        rocs = [
            ROC(
                name="sparse vs dense",
                selection=roc_nominal_selection,
                bins=ROC_BIN,
                pos=["dense"],
            )
        ]

        return rocs


class SparseDenseEval(HCREval):
    model = "test-SvD"

    @staticmethod
    def output_definition(batch: BatchType):
        output = {
            "q_1234": ...,
            "q_1324": ...,
            "q_1423": ...,
            "p_sparse": ...,
            "p_dense": ...,
        }
        return output
