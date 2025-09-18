from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.task import ArgParser, parse

from ..._HCR import _SCHEDULER
from ..ggF import all_kl

if TYPE_CHECKING:
    from classifier.ml.skimmer import Splitter


class loss_valid:
    def __init__(self, loss):
        self.loss = loss

    def __call__(self, batch):
        return {"loss": self.loss(batch)}


class Train(all_kl.Train):
    argparser = ArgParser(description="Train SvB with modified architecture.")
    argparser.remove_argument("--training")
    argparser.remove_argument("--finetuning")
    argparser.remove_argument("--architecture")
    defaults = {
        "architecture": {"n_features": 16},
        "finetuning": [
            "FixedStep",
            "epoch: 10",
            "bs_init: 8192",
            "lr_init: 1.0e-3",
            "bs_milestones: [1, 2, 4, 7]",
            "lr_milestones: [5, 8, 10]",
        ],
        "training": [
            "FixedStep",
            "epoch: 5",
        ],
    }

    def initializer(self, splitter: Splitter, **kwargs):
        from classifier.ml.models.HCR import (
            GBNSchedule,
            HCRArch,
            HCRBenchmarks,
        )
        from classifier.ml.models.HCR_finetune_ggF import SvBTraining

        arch = HCRArch(**({"loss": self.loss} | self.opts.architecture))
        gbn = GBNSchedule(**self.opts.ghost_batch)
        training = parse.instance(self.opts.training, _SCHEDULER)
        finetuning = parse.instance(self.opts.finetuning, _SCHEDULER)

        return SvBTraining(
            arch=arch,
            ghost_batch=gbn,
            cross_validation=splitter,
            training_schedule=training,
            finetuning_schedule=finetuning,
            benchmarks=HCRBenchmarks(rocs=self.rocs, scalars=[loss_valid(self.loss)]),
            model=self.model,
            **kwargs,
        )
