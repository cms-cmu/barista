from __future__ import annotations

from typing import TYPE_CHECKING

from src.classifier.config.setting.bbWWHCR import Input, MassRegion, Output
from src.classifier.config.state.label import MultiClass
from src.classifier.task import ArgParser

from src.classifier.config.model.bbWW.base._HCR import ROC_BIN, HCREval, HCRTrain

if TYPE_CHECKING:
    from src.classifier.ml import BatchType


def _roc_data_selection(batch: BatchType):
    import torch

    is_data = batch[Input.label].new(MultiClass.indices("data"))
    is_data = torch.isin(batch[Input.label], is_data)
    return {
        "y_pred": batch[Output.tt_prob][is_data],
        "y_true": batch[Input.label][is_data],
        "weight": batch[Input.weight][is_data],
    }


class Train(HCRTrain):
    argparser = ArgParser(description="Train dvtt")
    model = "dvtt"

    @staticmethod
    def loss(batch: BatchType):
        import torch
        import torch.nn.functional as F

        # get tensors
        tt_score = batch[Output.tt_raw]
        weight = batch[Input.weight]
        weight[weight < 0] = 0
        is_SR = MassRegion.SR

        # calculate loss
        cross_entropy = torch.zeros_like(weight)
        cross_entropy[~is_SR] = F.cross_entropy(
            c_score[~is_SR], batch[Input.label][~is_SR], reduction="none"
        )
        loss = (cross_entropy * weight).sum() / weight.sum()
        return loss
        
    @property
    def rocs(self):
        from src.classifier.ml.benchmarks.multiclass import ROC
        rocs = []
        
        if "ttbar" in MultiClass.labels:
            rocs.append(
                ROC(
                    name="ttbar vs data",
                    selection=roc_nominal_selection,
                    bins=ROC_BIN,
                    pos=["ttbar"],
                    neg=["data"],  # Explicitly specify data as negative class
                )
            )
        return rocs


class Eval(HCREval):
    model = "dvtt"

    @staticmethod
    def output_definition(batch: BatchType):
            ttbar_idx = MultiClass.indices("ttbar")[0]
            output = {
                "p_score": batch[Output.class_prob][:, ttbar_idx]  # ttbar probability
            }
            return output