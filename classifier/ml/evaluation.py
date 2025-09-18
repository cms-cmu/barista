from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator

import torch
from classifier.config.setting import ml as cfg

from ..process.device import Device
from ..typetools import filename
from . import clear_cache
from .training import Model, Stage

if TYPE_CHECKING:
    from ..nn.dataset.evaluation import EvalDatasetLike


@dataclass(kw_only=True)
class EvaluationStage(Stage):
    model: Model
    dataset: EvalDatasetLike
    dumper_kwargs: dict[str]
    batch_size: int = ...

    @torch.no_grad()
    def run(self, _):
        batch_size = self.batch_size
        if batch_size is ...:
            batch_size = cfg.DataLoader.batch_eval
        self.model.nn.eval()
        loader = self.dataset.load(batch_size=batch_size, **self.dumper_kwargs)
        for dump, batch in loader:
            dump(self.model.evaluate(batch))
        return {
            "stage": self.stage,
            "name": self.name,
            "output": [*loader.result],
        }


class Evaluation(ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.metadata = kwargs
        self.name = filename(kwargs)
        self.device: torch.device = None
        self.dataset: EvalDatasetLike = None

    @property
    def min_memory(self):
        return 0

    def cleanup(self):
        clear_cache(self.device)

    @abstractmethod
    def stages(self) -> Generator[EvaluationStage, None, None]: ...

    def eval(self, device: Device, dataset: EvalDatasetLike):
        self.device = device.get(self.min_memory)
        self.dataset = dataset
        result = {
            "name": self.name,
            "metadata": self.metadata,
            "outputs": [],
        }
        outputs: list[dict] = result["outputs"]
        for stage in self.stages():
            outputs.append(stage.run(self))
        return result
