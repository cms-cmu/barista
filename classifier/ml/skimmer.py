from __future__ import annotations

from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Callable

import numpy.typing as npt
import torch
from src.math.random import SeedLike, Squares
from classifier.config.setting import ml as cfg
from torch import BoolTensor
from torch.utils.data import Dataset

from ..nn.dataset import subset
from ..utils import keep_fraction, noop
from . import BatchType
from .training import Model


class Skimmer(Model):
    @property
    def n_parameters(self) -> int:
        return 0

    @property
    def nn(self):
        return noop

    def train(self, _):
        return noop


class Splitter(ABC):
    @abstractmethod
    def split(self, batch: BatchType) -> dict[str, BoolTensor]: ...

    def setup(self, dataset: Dataset):
        self.reset()
        self.__dataset = dataset

    def step(self, batch: BatchType) -> dict[str, BoolTensor]:
        selected = self.split(batch)
        size = len(next(iter(selected.values())))
        if self.__selected is None:
            self.__selected = {
                k: torch.zeros(len(self.__dataset), dtype=torch.bool) for k in selected
            }
        for k, v in selected.items():
            self.__selected[k][self.__start : self.__start + size] = v
        self.__start += size
        return selected

    def reset(self):
        self.__dataset: Dataset = None
        self.__start: int = 0
        self.__selected: dict[str, BoolTensor] = None

    def get(self):
        if self.__start != len(self.__dataset):
            raise RuntimeError(f"{self.__class__.__name__} is not fully initialized")
        selected, dataset = self.__selected, self.__dataset
        self.reset()
        indices = torch.arange(len(dataset))
        return {k: subset(dataset, indices[v]) for k, v in selected.items()}

    def __add__(self, other: Splitter):
        if not isinstance(other, Splitter):
            return NotImplemented
        splitters = []
        for s in (self, other):
            if isinstance(s, ChainedSplitter):
                splitters.extend(s._splitters)
            else:
                splitters.append(s)
        return ChainedSplitter(*splitters)


class ChainedSplitter(Splitter):
    def __init__(self, *splitters: Splitter):
        self._splitters = splitters

    def split(self, batch: BatchType):
        merged = {}
        for s in self._splitters:
            for k, v in s.split(batch).items():
                if k in merged:
                    merged[k] &= v
                else:
                    merged[k] = v
        return merged


class KFold(Splitter):
    def __init__(self, kfolds: int, offset: int):
        self._k = kfolds
        self._i = offset

    def split(self, batch: BatchType):
        validation = torch.from_numpy((self._get_offset(batch) % self._k) == self._i)
        return {
            cfg.SplitterKeys.training: ~validation,
            cfg.SplitterKeys.validation: validation,
        }

    @classmethod
    def _get_offset(cls, batch: BatchType) -> npt.NDArray:
        return batch[cfg.KFold.offset].numpy().view(cfg.KFold.offset_dtype)


class RandomSubSample(KFold):
    def __init__(self, seed: SeedLike, fraction: str | Fraction):
        self._rng = Squares(seed)
        self._r = Fraction(fraction)

    def split(self, batch: BatchType):
        training = torch.from_numpy(keep_fraction(self._r, self._random_offset(batch)))
        return {
            cfg.SplitterKeys.training: training,
            cfg.SplitterKeys.validation: ~training,
        }

    def _random_offset(self, batch: BatchType) -> npt.NDArray:
        offset = self._get_offset(batch)
        offset = offset.reshape(offset.shape[0], -1)
        return self._rng.uint64(offset)


class RandomKFold(RandomSubSample):
    def __init__(self, seed: SeedLike, kfolds: int, offset: int):
        self._rng = Squares(seed)
        self._k = kfolds
        self._i = offset

    def split(self, batch: BatchType):
        validation = torch.from_numpy((self._random_offset(batch) % self._k) == self._i)
        return {
            cfg.SplitterKeys.training: ~validation,
            cfg.SplitterKeys.validation: validation,
        }


class Filter(Splitter):
    def __init__(self, **selection: Callable[[BatchType], BoolTensor]):
        self._selection = selection

    def split(self, batch: BatchType):
        return {k: v(batch) for k, v in self._selection.items()}
