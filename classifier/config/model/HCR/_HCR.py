from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable

from classifier.config.setting.HCR import Input, Output
from classifier.task import ArgParser, parse

from .._kfold import KFoldEval, KFoldTrain

_SCHEDULER = "classifier.config.scheduler"


if TYPE_CHECKING:
    from classifier.ml import BatchType
    from classifier.ml.benchmarks.multiclass import ROC
    from classifier.ml.skimmer import Splitter
    from torch import Tensor

ROC_BIN = (1000, 0, 1)


def roc_nominal_selection(batch: BatchType):
    return {
        "y_pred": batch[Output.class_prob],
        "y_true": batch[Input.label],
        "weight": batch[Input.weight],
    }


class HCRTrain(KFoldTrain):
    model: str
    loss: Callable[[BatchType], Tensor]
    rocs: Iterable[ROC] = ()

    argparser = ArgParser()
    argparser.add_argument(
        "--architecture",
        type=parse.mapping,
        default="",
        help=f"HCR architecture {parse.EMBED}",
    )
    argparser.add_argument(
        "--ghost-batch",
        type=parse.mapping,
        default="",
        help=f"ghost batch normalization configuration {parse.EMBED}",
    )
    argparser.add_argument(
        "--training",
        nargs="+",
        default=["FixedStep"],
        metavar=("CLASS", "KWARGS"),
        help=f"training scheduler {parse.EMBED}",
    )
    argparser.add_argument(
        "--finetuning",
        nargs="+",
        default=[],
        metavar=("CLASS", "KWARGS"),
        help=f"fine-tuning scheduler {parse.EMBED}",
    )

    def initializer(self, splitter: Splitter, **kwargs):
        from classifier.ml.models.HCR import (
            GBNSchedule,
            HCRArch,
            HCRBenchmarks,
            HCRTraining,
        )

        arch = HCRArch(**({"loss": self.loss} | self.opts.architecture))
        gbn = GBNSchedule(**self.opts.ghost_batch)
        training = parse.instance(self.opts.training, _SCHEDULER)
        finetuning = parse.instance(self.opts.finetuning, _SCHEDULER)

        return HCRTraining(
            arch=arch,
            ghost_batch=gbn,
            cross_validation=splitter,
            training_schedule=training,
            finetuning_schedule=finetuning,
            benchmarks=HCRBenchmarks(
                rocs=self.rocs,
            ),
            model=self.model,
            **kwargs,
        )


class HCREval(KFoldEval):
    model: str
    output_definition: Callable[[BatchType], BatchType]

    def initializer(self, model, splitter, **kwargs):
        from classifier.ml.models.HCR import HCREvaluation

        return HCREvaluation(
            saved_model=model,
            cross_validation=splitter,
            output_definition=self.output_definition,
            model=self.model,
            **kwargs,
        )
