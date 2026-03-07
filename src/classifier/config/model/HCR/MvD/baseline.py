from __future__ import annotations

from typing import TYPE_CHECKING

from src.classifier.config.setting.HCR import Input, MassRegion, Output
from src.classifier.config.state.label import MultiClass
from src.classifier.task import ArgParser

from .._HCR import ROC_BIN, HCREval, HCRTrain, roc_nominal_selection

if TYPE_CHECKING:
    from src.classifier.ml import BatchType


def _roc_mix4_selection(batch: BatchType):
    import torch

    is_mix4_or_d4 = batch[Input.label].new(MultiClass.indices("d4", "mix4"))
    is_mix4_or_d4 = torch.isin(batch[Input.label], is_mix4_or_d4)
    return {
        "y_pred": batch[Output.class_prob][is_mix4_or_d4],
        "y_true": batch[Input.label][is_mix4_or_d4],
        "weight": batch[Input.weight][is_mix4_or_d4],
    }


class Train(HCRTrain):
    argparser = ArgParser(description="Train MvD")
    model = "MvD"

    @staticmethod
    def loss(batch: BatchType):
        import torch
        import torch.nn.functional as F

        # get tensors
        c_score = batch[Output.class_raw]
        weight = batch[Input.weight]
        weight[weight < 0] = 0
        is_SR = (batch[Input.region] & MassRegion.SR) != 0

        # remove 4b detector data contribution from SR
        d4 = MultiClass.index("d4")
        no_d4_idx = [*range(len(MultiClass.trainable_labels))]
        no_d4_idx = no_d4_idx[:d4] + no_d4_idx[d4 + 1:]
        no_d4_y = batch[Input.label][is_SR]
        no_d4_y = torch.where(no_d4_y > d4, no_d4_y - 1, no_d4_y)

        # calculate loss
        cross_entropy = torch.zeros_like(weight)
        cross_entropy[~is_SR] = F.cross_entropy(
            c_score[~is_SR], batch[Input.label][~is_SR], reduction="none"
        )
        cross_entropy[is_SR] = F.cross_entropy(
            c_score[is_SR][:, no_d4_idx], no_d4_y, reduction="none"
        )
        loss = (cross_entropy * weight).sum() / weight.sum()
        return loss

    @property
    def rocs(self):
        from src.classifier.ml.benchmarks.multiclass import ROC

        rocs = [
            ROC(
                name="4b data vs mixed_all",
                selection=_roc_mix4_selection,
                bins=ROC_BIN,
                pos=("d4",),
            ),
        ]
        if "t4" in MultiClass.labels:
            rocs.append(
                ROC(
                    name="4b data vs mixed_all+ttbar",
                    selection=roc_nominal_selection,
                    bins=ROC_BIN,
                    pos=("d4",),
                )
            )
            rocs.append(
                ROC(
                    name="ttbar vs mixed_all",
                    selection=roc_nominal_selection,
                    bins=ROC_BIN,
                    pos=("t4",),
                )
            )
        return rocs


class Eval(HCREval):
    model = "MvD"

    @staticmethod
    def output_definition(batch: BatchType):
        output = {
            "q_1234": ...,
            "q_1324": ...,
            "q_1423": ...,
            "p_mix4": ...,
            "p_d4": ...,
        }
        if "p_t4" in batch:
            output |= {
                "p_t4": ...,
                "p_4b": batch["p_d4"] + batch["p_t4"],
                "MvD": (batch["p_d4"] - batch["p_t4"]) / batch["p_mix4"],
            }
        else:
            output |= {
                "MvD": batch["p_d4"] / batch["p_mix4"],
            }
        return output
