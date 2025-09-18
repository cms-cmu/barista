from __future__ import annotations

import json
import logging
from datetime import datetime
from itertools import chain

import fsspec
from classifier.task import Analysis, ArgParser, EntryPoint, TaskOptions, main
from classifier.task.analysis import Analyzer

from .. import setting as cfg
from ._utils import progress_advance


class Main(main.Main):
    _no_state = True

    argparser = ArgParser(
        prog="analyze",
        description="Run standalone analysis.",
        workflow=[
            ("main", "[blue]\[analyzer, ...]=analysis.analyze()[/blue] initialize"),
            ("sub", "[blue]analyzer()[/blue] run"),
        ],
    )
    argparser.add_argument(
        "results",
        metavar="RESULT",
        nargs="*",
        help="path to result json files.",
    )

    @classmethod
    def prelude(cls):
        cfg.IO.monitor = None
        cfg.Analysis.enable = False

    def run(self, parser: EntryPoint):
        results = []
        for path in self.opts.results:
            with fsspec.open(path, "rt") as f:
                results.append(json.load(f))
        return run_analyzer(parser, results)


def _analyze(analyzer: Analyzer):
    return analyzer()


def run_analyzer(parser: EntryPoint, results: list[dict]):
    from concurrent.futures import ProcessPoolExecutor

    from classifier.monitor import Index
    from classifier.monitor.progress import Progress
    from classifier.process import pool, status

    analysis: list[Analysis] = parser.tasks[TaskOptions.analysis.name]
    analyzers = [*chain(*(a.analyze(results) for a in analysis))]
    if not analyzers:
        return None

    timer = datetime.now()
    with (
        ProcessPoolExecutor(
            max_workers=cfg.Analysis.max_workers,
            mp_context=status.context,
            initializer=status.initializer,
        ) as executor,
        Progress.new(total=len(analyzers), msg=("analysis", "Running")) as progress,
    ):
        outputs = [
            *pool.map_async(
                executor,
                _analyze,
                analyzers,
                callbacks=[lambda _: progress_advance(progress)],
            )
        ]
    Index.render()
    logging.info(f"Completed {len(outputs)} analysis in {datetime.now() - timer}")

    outputs = [*filter(lambda x: x is not None, outputs)]
    if outputs:
        return {cfg.ResultKey.analysis: outputs}
    else:
        return None
