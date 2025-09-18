from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.config.setting.HCR import Input
from classifier.config.state.label import MultiClass
from classifier.task import ArgParser

from . import _remove_sig

if TYPE_CHECKING:
    from classifier.ml import BatchType


class _remove_ZX_ggF:
    def __init__(self, kl: float):
        self.kl = kl

    def __call__(self, batch: BatchType):
        import torch

        label = batch[Input.label]
        return ~(
            torch.isin(label, label.new_tensor(MultiClass.indices("ZZ", "ZH")))
            | ((label == MultiClass.index("ggF")) & (batch["kl"] != self.kl))
        )


class Train(_remove_sig.Train):
    model = "SvB_ggF-sm-no-ZX"
    argparser = ArgParser(description="Train SvB without ZZ and ZH.")

    def remover(self):
        return _remove_ZX_ggF(1.0)
