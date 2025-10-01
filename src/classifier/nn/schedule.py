from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from bisect import bisect_right
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, TypeVar

from ..typetools import nameof

if TYPE_CHECKING:
    from torch import optim
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader, Dataset


class BSScheduler(ABC):
    dataloader: DataLoader

    @abstractmethod
    def step(self, epoch: int = None): ...


_ValueT = TypeVar("_ValueT")


class Schedule(ABC):
    require_benchmark = False

    epoch: int
    epoch_key: Iterable[str] = ("hyperparameters", "epoch")

    @abstractmethod
    def optimizer(self, parameters, **kwargs) -> optim.Optimizer: ...

    @abstractmethod
    def bs_scheduler(self, dataset: Dataset, **kwargs) -> BSScheduler: ...

    @abstractmethod
    def lr_scheduler(self, optimizer: optim.Optimizer, **kwargs) -> LRScheduler: ...

    def _get_key(
        self,
        keys: Iterable[str],
        benchmark: dict,
        required: bool = False,
        value_type: Callable[[Any], _ValueT] = Any,
    ) -> Optional[_ValueT]:
        try:
            value = benchmark
            for key in keys:
                value = value[key]
            if value_type is not Any:
                value = value_type(value)
        except Exception as e:
            if required:
                value_typename = nameof(value_type) if value_type is not Any else ""
                logging.error(
                    f'Scheduler {nameof(self)}: required key {value_typename}"{key}" missing when reading',
                    "benchmark" + "".join(f"\[{k}]" for k in keys),
                    exc_info=e,
                )
                raise
            value = None
        return value

    def bs_step(self, bs: BSScheduler, benchmark: dict = None):
        bs.step(self._get_key(self.epoch_key, benchmark))

    def lr_step(self, lr: LRScheduler, benchmark: dict = None):
        lr.step(self._get_key(self.epoch_key, benchmark))

    def step(self, bs: BSScheduler, lr: LRScheduler, benchmark: dict = None):
        self.bs_step(bs, benchmark)
        self.lr_step(lr, benchmark)


class MilestoneStep:
    def __init__(self, milestones: Optional[Iterable[int]] = None):
        self.reset()
        self._milestones = []
        self.milestones = milestones

    @property
    def milestone(self):
        return self._milestone

    @property
    def milestones(self):
        return self._milestones

    @milestones.setter
    def milestones(self, milestones: Iterable[int] = None):
        self._milestones = sorted(milestones or [])

    def reset(self):
        self._step = 0
        self._milestone = 0

    def step(self, step: int = None):
        if step is None:
            self._step += 1
        else:
            self._step = step
        milestone = bisect_right(self.milestones, self._step)
        changed = milestone != self.milestone
        self._milestone = milestone
        return changed


class MultiStepBS(MilestoneStep, BSScheduler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        milestones: Optional[Iterable[int]] = None,
        gamma: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(milestones=milestones)
        self.dataset = dataset
        self.batch_size = batch_size
        self.gamma = gamma
        self.kwargs = kwargs

        self._bs = batch_size
        self._dataloader: DataLoader = None

    @property
    def dataloader(self):
        from .dataset import simple_loader

        if self._dataloader is None or self._dataloader.batch_size != self._bs:
            self._dataloader = simple_loader(
                self.dataset, batch_size=self._bs, **self.kwargs
            )
        return self._dataloader

    def step(self, epoch: int = None):
        super(MultiStepBS, self).step(epoch)
        self._bs = int(self.batch_size * (self.gamma**self.milestone))
