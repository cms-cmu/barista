from __future__ import annotations
from typing import TYPE_CHECKING
from classifier.config.setting.bbWWHCR import Input, Output
from classifier.config.state.label import MultiClass
from classifier.task import ArgParser
from classifier.config.model.HCR._HCR import ROC_BIN, HCREval, HCRTrain

if TYPE_CHECKING:
    from classifier.ml import BatchType

def _roc_signal_selection(batch: BatchType):
    import torch
    signal = batch[Input.label].new(MultiClass.indices("signal"))
    signal = torch.isin(batch[Input.label], signal)
    return {
        "y_pred": batch[Output.class_prob][signal],  # Signal probability
        "y_true": batch[Input.label][signal],              # 0 or 1 labels
        "weight": batch[Input.weight][signal],
    }

class Train(HCRTrain):
    argparser = ArgParser(description="Train bbWW Model")
    model = "bbWW"

    @staticmethod
    def loss(batch: BatchType):
        import torch
        import torch.nn.functional as F
        
        # Simple binary classification
        logits = batch[Output.class_raw]
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
                selection=_roc_signal_selection,
                bins=ROC_BIN,
                pos=(1,),  # Signal class
            ),
        ]

class Eval(HCREval):
    model = "bbWW"

    @staticmethod
    def output_definition(batch: BatchType):
        return {
            "signal_prob": batch[Output.class_prob][:, 1],
            "background_prob": batch[Output.class_prob][:, 0],
            "HH_score": batch[Output.class_prob][:, 1],
        }