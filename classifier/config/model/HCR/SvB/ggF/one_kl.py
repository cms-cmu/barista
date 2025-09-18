from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.config.setting.HCR import Input
from classifier.config.state.label import MultiClass
from classifier.task import ArgParser

from . import _remove_sig

if TYPE_CHECKING:
    from classifier.ml import BatchType


class _remove_ggf:
    def __init__(self, kl: float):
        self.kl = kl

    def __call__(self, batch: BatchType):
        return ~(
            (batch[Input.label] == MultiClass.index("ggF")) & (batch["kl"] != self.kl)
        )


class Train(_remove_sig.Train):
    argparser = ArgParser(description="Train SvB with one of ggF signal.")
    argparser.add_argument(
        "--signal-ggf-kl", type=float, default=1.0, help="ggF signal used in training."
    )

    def remover(self):
        return _remove_ggf(self.opts.signal_ggf_kl)
