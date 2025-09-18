from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .special import interface
from .task import Task

if TYPE_CHECKING:
    from classifier.nn.dataset.evaluation import EvalDatasetLike
    from torch.utils.data import Dataset as TorchDataset


class Dataset(Task):
    @interface
    def train(self) -> list[TrainingSetLoader]:
        """
        Prepare training set loaders.
        """
        ...

    @interface
    def evaluate(self) -> list[EvaluationSetLoader]:
        """
        Prepare evaluation set loaders.
        """
        ...


class TrainingSetLoader(Protocol):
    def __call__(self) -> dict[str, TorchDataset]:
        """
        Load training set.
        """
        ...


class EvaluationSetLoader(Protocol):
    def __call__(self) -> EvalDatasetLike:
        """
        Generate evaluation set.
        """
        ...
