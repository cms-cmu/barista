from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.config.setting.HCR import Input
from classifier.config.state.label import MultiClass
from classifier.task import ArgParser

from . import _remove_sig

if TYPE_CHECKING:
    from classifier.ml import BatchType


class _remove_ZX:
    def __call__(self, batch: BatchType):
        import torch

        label = batch[Input.label]
        return ~torch.isin(label, label.new_tensor(MultiClass.indices("ZZ", "ZH")))


class Train(_remove_sig.Train):
    model = "SvB_ggF-all-no-ZX"
    argparser = ArgParser(description="Train SvB without ZZ and ZH.")

    def remover(self):
        return _remove_ZX()
