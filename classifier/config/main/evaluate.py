from __future__ import annotations

import logging
import operator as op
from datetime import datetime
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING

from classifier.task import (
    ArgParser,
    Dataset,
    EntryPoint,
    Model,
    TaskOptions,
    converter,
    main,
)
from classifier.task.model import ModelRunner

from ..setting import ResultKey
from ._utils import SelectDevice, progress_advance

if TYPE_CHECKING:
    from classifier.nn.dataset.evaluation import EvalDatasetLike
    from classifier.process.device import Device


class _eval_model:
    def __init__(self, device: Device, dataset: EvalDatasetLike):
        self._device = device
        self._dataset = dataset

    def __call__(self, trainer: ModelRunner):
        return trainer(device=self._device, dataset=self._dataset)


class Main(SelectDevice, main.Main):
    argparser = ArgParser(
        prog="evaluate",
        description="Evaluate multiple models on the same dataset.",
        workflow=[
            (
                "main",
                "[blue]\[loader, ...]=dataset.evaluate()[/blue] initialize datasets",
            ),
            ("main", "[blue]loader()[/blue] prepare datasets"),
            (
                "main",
                "[blue]\[evaluator, ...]=model.evaluate()[/blue] initialize models",
            ),
            ("sub", "[blue]evaluator(device, dataset)[/blue] evaluate models"),
        ],
    )
    argparser.add_argument(
        "--max-evaluators",
        type=converter.int_pos,
        default=1,
        help="Maximum number of concurrent evaluators.",
    )

    def run(self, parser: EntryPoint):
        from concurrent.futures import ProcessPoolExecutor

        from classifier.monitor.progress import Progress
        from classifier.process import pool, status

        # prepare datasets
        tasks: list[Dataset] = parser.tasks[TaskOptions.dataset.name]
        timer = datetime.now()
        datasets = [*chain(*(t.evaluate() for t in tasks))]
        dataset: EvalDatasetLike = reduce(op.add, datasets)
        logging.info(
            f"Initialized {len(datasets)} datasets in {datetime.now() - timer}"
        )
        # initialize models
        models: list[Model] = parser.tasks[TaskOptions.model.name]
        timer = datetime.now()
        evaluators = [*chain(*(m.evaluate() for m in models))]
        logging.info(
            f"Initialized {len(evaluators)} models in {datetime.now() - timer}"
        )
        # evaluate models in parallel
        timer = datetime.now()
        with (
            ProcessPoolExecutor(
                max_workers=self.opts.max_evaluators,
                mp_context=status.context,
                initializer=status.initializer,
            ) as executor,
            Progress.new(
                total=len(evaluators), msg=("models", "Evaluating")
            ) as progress,
        ):
            results = [
                *pool.map_async(
                    executor,
                    _eval_model(self.device, dataset),
                    evaluators,
                    callbacks=[lambda _: progress_advance(progress)],
                )
            ]
        logging.info(f"Evaluated {len(evaluators)} models in {datetime.now() - timer}")
        return {ResultKey.predictions: results}
