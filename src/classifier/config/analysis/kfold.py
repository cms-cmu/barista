import operator as op
import sys
from collections import defaultdict
from functools import reduce
from typing import Iterable

from src.classifier.task import Analysis, ArgParser, converter

from ..setting import IO, ResultKey


class Merge(Analysis):
    argparser = ArgParser()
    argparser.add_argument(
        "--name",
        help="the name of the merged friend tree",
        default=None,
    )
    argparser.add_argument(
        "--base",
        default="chunks",
        help="the base path to store the evaluation results",
    )
    argparser.add_argument(
        "--naming",
        default=...,
        help="the rule to name friend tree files for evaluation",
    )
    argparser.add_argument(
        "--step",
        type=converter.int_pos,
        default=sys.maxsize,
        help="the number entries to load for each step",
    )
    argparser.add_argument(
        "--workers",
        type=converter.int_pos,
        default=1,
        help="the number of workers to run in parallel",
    )
    argparser.add_argument(
        "--stage",
        default=...,
        help="the name of the evaluation stage to merge",
    )
    argparser.add_argument(
        "--clean",
        action="store_true",
        help="remove the original friend trees after merging",
    )
    argparser.add_argument(
        "--optimize-step",
        type=converter.int_pos,
        default=10_000,
        help="the number of entries to load for each step when optimizing the merged friend tree.",
    )
    argparser.add_argument(
        "--std",
        action="extend",
        nargs="+",
        help="calculate the std of the specified branches",
        default=[],
    )

    def analyze(self, results: list[dict]):
        from src.classifier.root.kfold import MergeMean, MergeStd, merge_kfolds

        kfolds = _load_friends(self.opts.stage, results)
        if len(kfolds) > 1:
            methods = [MergeMean]
            if self.opts.std:
                methods.append(MergeStd(self.opts.std))
            return [
                merge_kfolds(
                    *kfolds,
                    methods=methods,
                    step=self.opts.step,
                    workers=self.opts.workers,
                    friend_name=self.opts.name,
                    dump_base_path=IO.output / self.opts.base,
                    dump_naming=self.opts.naming,
                    clean=self.opts.clean,
                    optimize=self.opts.optimize_step,
                )
            ]
        return []


def _load_friends(stage: str, results: list[dict]):
    from src.data_formats.root import Friend

    kfolds: list[Friend] = []
    for result in results:
        predictions: list[dict] = result.get(ResultKey.predictions)
        if predictions is None:
            continue
        for prediction in predictions:
            outputs: list[dict] = prediction.get("outputs")
            if not isinstance(outputs, Iterable):
                continue
            for output in outputs:
                if (output.get("stage") != "Evaluation") or (
                    (stage is not ...) and (output.get("name") != stage)
                ):
                    continue
                friends: dict = output.get("output")
                if not isinstance(friends, Iterable):
                    continue
                datasets: dict[str, list[Friend]] = defaultdict(list)
                for friend in friends:
                    dataset = None
                    if isinstance(friend, Friend):
                        dataset = friend
                    else:
                        try:
                            dataset = Friend.from_json(friend)
                        except Exception:
                            ...
                    if dataset is not None:
                        datasets[dataset.name].append(dataset)
                kfolds.extend(reduce(op.add, v) for v in datasets.values())
    return kfolds
