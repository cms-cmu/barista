from __future__ import annotations

import math
import queue
import threading
from abc import ABC, abstractmethod
from typing import Generator, Generic, TypeVar

import numpy as np
import torch
from src.math_tools.random import SeedLike, _seed
from torch.utils.data import Dataset
from typing_extensions import Self

_BatchT = TypeVar("_BatchT")


class SliceableDataset(Dataset, ABC, Generic[_BatchT]):
    @abstractmethod
    def __getitem__(self, indices: torch.Tensor | slice) -> _BatchT: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @classmethod
    @abstractmethod
    def concat(cls, *datasets: Self) -> Self: ...

    @abstractmethod
    def subset(self, indices: torch.Tensor) -> Self: ...

    @property
    @abstractmethod
    def datasets(self): ...


class NamedTensorDataset(SliceableDataset[dict[str, torch.Tensor]]):
    def __init__(self, **tensors: torch.Tensor):
        sizes = {v.shape[0] for v in tensors.values()}
        if len(sizes) != 1:
            raise ValueError("All tensors must have the same length")
        self._size = sizes.pop()
        self._tensors = tensors

    def __len__(self):
        return self._size

    def __getitem__(self, indices: torch.Tensor | slice):
        batch = {k: v[indices] for k, v in self._tensors.items()}
        if isinstance(indices, slice):
            for k in batch:
                batch[k] = batch[k].clone()
        return batch

    @classmethod
    def concat(cls, *datasets: NamedTensorDataset):
        if not all(isinstance(d, NamedTensorDataset) for d in datasets):
            raise ValueError("Cannot concat datasets of different types")
        return NamedTensorDataset(
            **{
                k: torch.cat([t._tensors[k] for t in datasets])
                for k in datasets[0]._tensors
            }
        )

    def subset(self, indices: torch.Tensor):
        return NamedTensorDataset(**self[indices])

    @property
    def datasets(self):
        return self._tensors


_SENTINEL = object()


class SliceLoaderLite(Generic[_BatchT]):
    def __init__(
        self,
        dataset: SliceableDataset[_BatchT],
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: SeedLike = None,
        prefetch: int = 2,
        **kwargs,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed if seed is None else _seed(seed)
        self.prefetch = max(0, int(prefetch))
        self.kwargs = kwargs

    def _iter_batches(self) -> Generator[_BatchT, None, None]:
        data = self.dataset
        ndata = len(data)
        nbatch = self.batch_size
        indices = None
        if self.shuffle:
            rng = np.random.Generator(np.random.PCG64(self.seed))
            indices = torch.from_numpy(rng.permutation(ndata))
        for i in range(0, ndata, nbatch):
            if self.drop_last and (i + nbatch) > ndata:
                break
            if indices is not None:
                yield data[indices[i : i + nbatch]]
            else:
                yield data[i : i + nbatch]

    def __iter__(self) -> Generator[_BatchT, None, None]:
        if self.prefetch <= 0 or len(self.dataset) <= self.batch_size:
            yield from self._iter_batches()
            return

        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()

        def _producer():
            try:
                for batch in self._iter_batches():
                    while not stop.is_set():
                        try:
                            q.put((True, batch), timeout=0.1)
                            break
                        except queue.Full:
                            continue
                    if stop.is_set():
                        break
            except BaseException as exc:
                while not stop.is_set():
                    try:
                        q.put((False, exc), timeout=0.1)
                        break
                    except queue.Full:
                        continue
            finally:
                while not stop.is_set():
                    try:
                        q.put((True, _SENTINEL), timeout=0.1)
                        break
                    except queue.Full:
                        continue

        worker = threading.Thread(target=_producer, daemon=True)
        worker.start()
        try:
            while True:
                ok, payload = q.get()
                if not ok:
                    raise payload
                if payload is _SENTINEL:
                    break
                yield payload
        finally:
            stop.set()
            worker.join(timeout=1.0)

    def __len__(self):
        return (math.floor if self.drop_last else math.ceil)(
            len(self.dataset) / self.batch_size
        )
