from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from queue import Queue
from threading import Thread
from typing import (
    TYPE_CHECKING,
    Callable,
    Generator,
    Generic,
    Protocol,
    TypeVar,
    overload,
)

from classifier.config.setting.ml import DataLoader as cfg
from classifier.monitor.progress import MessageType, Progress
from classifier.process import status

if sys.version_info >= (3, 11):
    from typing import TypeVarTuple, Unpack
else:
    from typing_extensions import TypeVarTuple, Unpack


_ResultT = TypeVar("_ResultT")
_OtherResultT = TypeVar("_OtherResultT")
_ChainResultT = TypeVarTuple("_ChainResultT")
_OtherChainResultT = TypeVarTuple("_OtherChainResultT")

if TYPE_CHECKING:
    from classifier.ml import BatchType

    BatchLoader = Callable[[], BatchType]

    class BatchDumper(Protocol):
        def __init__(self, **kwargs): ...

        def __call__(self, batch: BatchType) -> _ResultT: ...

        def __len__(self) -> int: ...

    class EvalLoaderLike(Protocol):
        @property
        def result(self) -> tuple: ...

        def __iter__(self) -> Generator[tuple[BatchDumper, BatchType], None, None]: ...

    class EvalDatasetLike(Protocol):
        def load(self, batch_size: int, **kwargs) -> EvalLoaderLike: ...

        def __add__(self, other: EvalDatasetLike) -> EvalDatasetLike: ...

    _ToLoadQ = Queue[tuple[int, BatchLoader]]
    _LoadedQ = Queue[tuple[int, BatchType]]
    _EvaledQ = Queue[tuple[BatchDumper, BatchType]]
    _ResultQ = Queue[tuple[int, _ResultT]]


class EvalLoader(ABC, Generic[_ResultT]):
    __results: _ResultQ

    @staticmethod
    def _pool():
        return ProcessPoolExecutor(
            max_workers=cfg.num_workers * 2,
            mp_context=status.context,
            initializer=status.initializer,
        )

    def _init(
        self,
        batches: Generator[tuple[BatchDumper, BatchLoader], None, None],
        msg: MessageType,
    ):
        self.__batches = batches
        self.__progress_msg = msg
        self.__pool = None
        return self

    def _collect_result(self, nbatches: int, total: int):
        with Progress.new(total, self.__progress_msg) as progress:
            for _ in range(nbatches):
                size, result = self.__results.get()
                self.accumulate(result)
                progress.advance(size)

    def _recycle_pool(self, pool: ProcessPoolExecutor):
        self.__pool = pool

    @abstractmethod
    def accumulate(self, result: _ResultT): ...

    @property
    @abstractmethod
    def result(self) -> tuple[_ResultT]: ...

    def __enter__(self):
        return self.__pool

    def __exit__(self, *_):
        self.__pool = None

    def __iter__(self) -> Generator[tuple[BatchDumper, BatchType], None, None]:
        batches = [*self.__batches]
        nbatches = len(batches)
        total = sum(len(batch[0]) for batch in batches)
        collector = Thread(target=self._collect_result, args=(nbatches, total))
        if cfg.num_workers == 0:
            # collect results
            self.__results = Queue()
            collector.start()
            # generate batches
            for dumper, loader in batches:
                yield _blocking_dumper(dumper, self.__results), loader()
            collector.join()
        else:
            manager = status.context.Manager()
            toload_queue: _ToLoadQ = manager.Queue()
            loaded_queue: _LoadedQ = manager.Queue(maxsize=cfg.num_workers)
            evaled_queue: _EvaledQ = manager.Queue(maxsize=cfg.num_workers)
            # collect results
            self.__results = manager.Queue()
            collector.start()
            # generate batches
            for i, (_, loader) in enumerate(batches):
                toload_queue.put((i, loader))

            with self._pool() if self.__pool is None else self as pool:
                for _ in range(cfg.num_workers):
                    pool.submit(_load_worker, toload_queue, loaded_queue)
                for _ in range(cfg.num_workers):
                    pool.submit(_dump_worker, evaled_queue, self.__results)
                for _ in range(nbatches):
                    i, batch = loaded_queue.get()
                    yield _nonblocking_dumper(batches[i][0], evaled_queue), batch
                # wait for workers
                collector.join()
                toload_queue.put(None)
                evaled_queue.put(None)
        del self.__results


class EvalDataset(ABC, Generic[_ResultT]):
    __eval_loader__ = EvalLoader

    __progress_msg: MessageType = ("entries", "Evaluated")

    @abstractmethod
    def batches(
        self, batch_size: int, **kwargs
    ) -> Generator[tuple[BatchDumper, BatchLoader], None, None]: ...

    def setup(self, msg: MessageType = ...):
        if msg is not ...:
            self.__progress_msg = msg
        return self

    def load(self, batch_size: int, **kwargs) -> EvalLoader[_ResultT]:
        return self.__eval_loader__()._init(
            self.batches(batch_size, **kwargs), self.__progress_msg
        )

    def __add__(
        self, other: EvalDataset[_OtherResultT]
    ) -> ChainDataset[_ResultT, _OtherResultT]:
        if not isinstance(other, EvalDataset):
            return NotImplemented
        return ChainDataset(self, other)


def _load_worker(to_load: _ToLoadQ, loaded: _LoadedQ):
    while (job := to_load.get()) is not None:
        i, loader = job
        try:
            loaded.put((i, loader()))
        except Exception as e:
            logging.exception("when loading batch", exc_info=e)
            raise
    to_load.put(None)


def _dump_worker(evaluated: _EvaledQ, results: _ResultQ):
    while (job := evaluated.get()) is not None:
        dumper, batch = job
        try:
            results.put((len(dumper), dumper(batch)))
        except Exception as e:
            logging.exception("when dumping evaluated batch", exc_info=e)
            raise
    evaluated.put(None)


class _blocking_dumper:
    def __init__(self, dumper: BatchDumper, result: _ResultQ):
        self.dumper = dumper
        self.queue = result

    def __call__(self, batch: BatchType):
        self.queue.put((len(self.dumper), self.dumper(batch)))


class _nonblocking_dumper:
    def __init__(self, dumper: BatchDumper, evaled: _EvaledQ):
        self.dumper = dumper
        self.queue = evaled

    def __call__(self, batch: BatchType):
        self.queue.put((self.dumper, batch))


class AddableResultLoader(EvalLoader[_ResultT]):
    def __init__(self):
        self.__result = None

    def accumulate(self, result: _ResultT):
        if self.__result is None:
            self.__result = result
        else:
            self.__result += result

    @property
    def result(self) -> tuple[_ResultT]:
        return (self.__result,)


class ChainLoader(Generic[Unpack[_ChainResultT]]):
    def __init__(self, *loaders: EvalLoader, msg: MessageType):
        self.__loaders = loaders
        self.__msg = msg

    def __enter__(self):
        return None

    def __exit__(self, *_): ...

    @property
    def result(self) -> tuple[Unpack[_ChainResultT]]:
        return (*(loader.result for loader in self.__loaders),)

    def __iter__(self) -> Generator[tuple[BatchDumper, BatchType], None, None]:
        with (
            self if cfg.num_workers == 0 else EvalLoader._pool() as pool,
            Progress.new(len(self.__loaders), self.__msg) as progress,
        ):
            for loader in self.__loaders:
                loader._recycle_pool(pool)
                yield from loader
                progress.advance(1)


class ChainDataset(Generic[Unpack[_ChainResultT]]):
    __progress_msg: MessageType = ("Datasets", "Evaluated")

    def __init__(self, *datasets: EvalDataset):
        self.__datasets = datasets

    def setup(self, msg: MessageType = ...):
        if msg is not ...:
            self.__progress_msg = msg
        return self

    def load(self, batch_size: int, **kwargs) -> ChainLoader[Unpack[_ChainResultT]]:
        return ChainLoader(
            *(dataset.load(batch_size, **kwargs) for dataset in self.__datasets),
            msg=self.__progress_msg,
        )

    def __radd__(
        self, other: EvalDataset[_OtherResultT]
    ) -> ChainDataset[_OtherResultT, Unpack[_ChainResultT]]:
        return ChainDataset(other, *self.__datasets)

    @overload
    def __add__(
        self,
        other: EvalDataset[_OtherResultT],
    ) -> ChainDataset[Unpack[_ChainResultT], _OtherResultT]: ...
    @overload
    def __add__(
        self,
        other: ChainDataset[Unpack[_OtherChainResultT]],
    ) -> ChainDataset[Unpack[_ChainResultT], Unpack[_OtherChainResultT]]: ...
    def __add__(self, other: EvalDataset | ChainDataset):
        if isinstance(other, ChainDataset):
            return ChainDataset(*self.__datasets, *other.__datasets)
        return ChainDataset(*self.__datasets, other)
