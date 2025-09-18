from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .special import interface
from .task import Task

if TYPE_CHECKING:
    from classifier.nn.dataset.evaluation import EvalDatasetLike
    from torch.utils.data import StackDataset

    from ..process.device import Device


class Model(Task):
    @interface
    def train(self) -> list[ModelTrainer]:
        """
        Pepare models for training.
        """
        ...

    @interface
    def evaluate(self) -> list[ModelRunner]:
        """
        Prepare models for evaluation.
        """
        ...


class ModelTrainer(Protocol):
    def __call__(self, device: Device, dataset: StackDataset) -> dict[str]:
        """
        Train model on dataset.
        """
        ...


class ModelRunner(Protocol):
    def __call__(self, device: Device, dataset: EvalDatasetLike) -> dict[str]:
        """
        Run model on dataset.
        """
        ...
