from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING

import fsspec
from classifier.task import ArgParser, Model, converter, parse
from rich.pretty import pretty_repr

from ..setting import ResultKey

if TYPE_CHECKING:
    from src.storage.eos import PathLike
    from classifier.ml.evaluation import Evaluation
    from classifier.ml.skimmer import Splitter
    from classifier.ml.training import MultiStageTraining


class KFoldTrain(ABC, Model):
    argparser = ArgParser()
    argparser.add_argument(
        "--no-kfold",
        action="store_true",
        help="disable kfold (equivalent to --kfolds 1)",
    )
    argparser.add_argument(
        "--kfolds",
        type=converter.bounded(int, lower=1),
        default=3,
        help="total number of folds (1 for no kfold)",
    )
    argparser.add_argument(
        "--kfold-offsets",
        action="extend",
        nargs="+",
        default=[],
        help="selected offsets, e.g. [yellow]--kfold-offsets 0-3 5[/yellow]",
    )
    argparser.add_argument(
        "--kfold-seed",
        action="extend",
        nargs="+",
        default=["kfold"],
        help="the random seed to shuffle the dataset",
    )
    argparser.add_argument(
        "--kfold-seed-offsets",
        action="extend",
        nargs="+",
        default=[],
        help="the offsets to generate new seeds, e.g. [yellow]--kfold-seed-offsets 0-3 5[/yellow]. If not given, the dataset will not be shuffled.",
    )

    def _get_offsets(self, name: str, max: int = None) -> list[int]:
        offsets = getattr(self.opts, name)
        if not offsets:
            return [*range(max)] if max else []
        else:
            return parse.intervals(offsets, max)

    @cached_property
    def kfolds(self) -> int:
        if self.opts.no_kfold:
            return 1
        return self.opts.kfolds

    @cached_property
    def offsets(self) -> list[int]:
        return self._get_offsets("kfold_offsets", self.kfolds)

    @cached_property
    def seeds(self) -> list[tuple[str]]:
        seed = self.opts.kfold_seed
        offsets = self._get_offsets("kfold_seed_offsets")
        if not offsets:
            return []
        return [(*seed, offset) for offset in offsets]

    @abstractmethod
    def initializer(self, splitter: Splitter, **kwargs) -> MultiStageTraining: ...

    def train(self):
        from classifier.ml.skimmer import KFold, RandomKFold

        if self.kfolds == 1:
            return [self.initializer(KFold(kfolds=1, offset=1)).train]
        elif not self.seeds:
            return [
                self.initializer(
                    KFold(self.kfolds, offset),
                    kfolds=self.kfolds,
                    offset=offset,
                ).train
                for offset in self.offsets
            ]
        else:
            return [
                self.initializer(
                    RandomKFold(seed=seed, kfolds=self.kfolds, offset=offset),
                    kfolds=self.kfolds,
                    offset=offset,
                    seed=seed,
                ).train
                for seed in self.seeds
                for offset in self.offsets
            ]


class KFoldEval(ABC, Model):
    argparser = ArgParser()
    argparser.add_argument(
        "--models",
        action="append",
        metavar=("NAME", "PATH"),
        nargs="+",
        default=[],
        help="name of the output stage and path to model json files",
    )
    argparser.add_argument(
        "--no-kfold",
        action="store_true",
        help="ignore the kfold used in training and evaluate all models",
    )

    @abstractmethod
    def initializer(
        self, model: PathLike, splitter: Splitter, **kwargs
    ) -> Evaluation: ...

    def evaluate(self):
        models = []
        metadatas = _find_models(self.opts.models, self.opts.no_kfold)
        for metadata in metadatas:
            models.append(self.initializer(**metadata).eval)
        if metadatas:
            logging.info(
                "The following models will be evaluated:",
                pretty_repr([m["model"] for m in metadatas]),
            )
        return models


def _find_models(args: list[list[str]], no_kfold: bool = False) -> list[dict]:
    from src.storage.eos import EOS
    from classifier.ml.skimmer import KFold, RandomKFold

    models = []
    for arg in args:
        name = arg[0]
        paths = arg[1:]
        for path in paths:
            with fsspec.open(path, "rt") as f:
                results: list[dict[str, dict]] = json.load(f).get(ResultKey.models, [])
            for result in results:
                metadata = result.get("metadata", {})
                m_path = None
                for stage in result.get("history", [])[::-1]:
                    if (stage.get("stage") == "Output") and (stage.get("name") == name):
                        m_path = stage["path"]
                        if stage.get("relative", False):
                            m_path = EOS(path).parent / m_path
                        break
                if m_path is None:
                    continue
                if ("kfolds" not in metadata) or ("offset" not in metadata):
                    continue
                kwargs = {
                    "kfolds": metadata["kfolds"],
                    "offset": metadata["offset"],
                }
                if "seed" in metadata:
                    kwargs["seed"] = metadata["seed"]
                    KFoldT = RandomKFold
                else:
                    KFoldT = KFold
                if no_kfold:
                    splitter = KFold(kfolds=1, offset=0)
                else:
                    splitter = KFoldT(**kwargs)
                models.append(dict(model=m_path, splitter=splitter, **kwargs))
    return models
