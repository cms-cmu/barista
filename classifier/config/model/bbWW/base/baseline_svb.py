from __future__ import annotations
from typing import TYPE_CHECKING
from classifier.config.setting.bbWWHCR import Input, Output
from classifier.config.state.label import MultiClass
from classifier.task import ArgParser
from classifier.config.model.bbWW.base._HCR import ROC_BIN, HCREval, HCRTrain

if TYPE_CHECKING:
    from classifier.ml import BatchType

_BKG = ("ttbar",)

class _roc_signal_selection:
    def __init__(self, sig: str):
        self.sig = sig

    def __call__(self, batch: BatchType):
        selected = self._select(batch)
        return {
            "y_pred": batch[Output.hh_prob][selected],  # Signal probability
            "y_true": batch[Input.label][selected],
            "weight": batch[Input.weight][selected],
        }

    def _select(self, batch: BatchType):
        import torch

        label = batch[Input.label]
        return torch.isin(label, label.new_tensor(MultiClass.indices(*_BKG, self.sig)))

class Train(HCRTrain):
    argparser = ArgParser(description="Train bbWW Model")
    model = "svb"

    @staticmethod
    def loss(batch: BatchType):
        import torch
        import torch.nn.functional as F
        
        # Simple binary classification
        logits = batch[Output.hh_raw]
        labels = batch[Input.label]  # 0 for background files, 1 for signal files
        weight = batch[Input.weight]
        weight[weight < 0] = 0
        
        cross_entropy = F.cross_entropy(logits, labels, reduction="none")
        loss = (cross_entropy * weight).sum() / weight.sum()
        return loss

    @property
    def rocs(self):
        from classifier.ml.benchmarks.multiclass import ROC
        
        return [
            ROC(
                name="Signal vs Background",
                selection=_roc_signal_selection("signal"),
                bins=ROC_BIN,
                pos=("signal",),  # Signal class
            ),
        ]

class Eval(HCREval):
    model = "svb"

    @staticmethod
    def output_definition(batch: BatchType):
        return {
            "signal_prob": batch["p_signal"], 
            "background_prob": batch["p_ttbar"], 
        }