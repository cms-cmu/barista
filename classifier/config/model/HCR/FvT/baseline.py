from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.config.setting.HCR import Input, MassRegion, Output
from classifier.config.state.label import MultiClass
from classifier.task import ArgParser

from .._HCR import ROC_BIN, HCREval, HCRTrain, roc_nominal_selection

if TYPE_CHECKING:
    from classifier.ml import BatchType


def _roc_data_selection(batch: BatchType):
    import torch

    is_data = batch[Input.label].new(MultiClass.indices("d4", "d3"))
    is_data = torch.isin(batch[Input.label], is_data)
    return {
        "y_pred": batch[Output.class_prob][is_data],
        "y_true": batch[Input.label][is_data],
        "weight": batch[Input.weight][is_data],
    }


class Train(HCRTrain):
    argparser = ArgParser(description="Train FvT")
    model = "FvT"

    @staticmethod
    def loss(batch: BatchType):
        import torch
        import torch.nn.functional as F

        # get tensors
        c_score = batch[Output.class_raw]
        weight = batch[Input.weight]
        weight[weight < 0] = 0
        is_SR = (batch[Input.region] & MassRegion.SR) != 0

        # remove 4b data contribution from SR
        d4 = MultiClass.index("d4")
        no_d4_idx = [*range(len(MultiClass.trainable_labels))]
        no_d4_idx = no_d4_idx[:d4] + no_d4_idx[d4 + 1 :]
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
        from classifier.ml.benchmarks.multiclass import ROC

        rocs = [
            ROC(
                name="4b vs 3b data",
                selection=_roc_data_selection,
                bins=ROC_BIN,
                pos=("d4",),
            ),
        ]
        if "t4" in MultiClass.labels:
            rocs.append(
                ROC(
                    name="4b vs 3b",
                    selection=roc_nominal_selection,
                    bins=ROC_BIN,
                    pos=("d4", "t4"),
                )
            )
        if tts := [tt for tt in ("t3", "t4") if tt in MultiClass.labels]:
            rocs.append(
                ROC(
                    name="ttbar vs data",
                    selection=roc_nominal_selection,
                    bins=ROC_BIN,
                    pos=tts,
                )
            )
        return rocs


class Eval(HCREval):
    model = "FvT"

    @staticmethod
    def output_definition(batch: BatchType):
        output = {
            "q_1234": ...,
            "q_1324": ...,
            "q_1423": ...,
            "p_d4": ...,
            "p_d3": ...,
            "p_data": batch["p_d4"] + batch["p_d3"],
        }
        if "p_t4" in batch:
            p_m4 = batch["p_d4"] - batch["p_t4"]
            output |= {
                "p_t4": ...,
                "p_m4": p_m4,
                "p_4b": batch["p_d4"] + batch["p_t4"],
                "FvT": p_m4 / batch["p_d3"],
            }
            if "p_t3" in batch:
                output |= {
                    "p_t3": ...,
                    "p_m3": batch["p_d3"] - batch["p_t3"],
                    "p_ttbar": batch["p_t4"] + batch["p_t3"],
                    "p_3b": batch["p_d3"] + batch["p_t3"],
                }
            else:
                output |= {
                    "p_ttbar": batch["p_t4"],
                    "p_3b": batch["p_d3"],
                }
        else:
            output |= {
                "FvT": batch["p_d4"] / batch["p_d3"],
            }
        return output
