from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Iterable

from src.data_formats.root import Chunk, Friend

from ..nn.dataset.evaluation import AddableResultLoader, EvalDataset

if TYPE_CHECKING:
    from src.data_formats.root.chain import NameMapping
    from src.data_formats.root.io import RecordLike
    from src.storage.eos import PathLike

    from ..ml import BatchType


def numpy_dumper(batch: BatchType) -> RecordLike:
    return {k: v.numpy(force=True) for k, v in batch.items()}


class FriendTreeEvalDataset(EvalDataset[Friend]):
    __eval_loader__ = AddableResultLoader[Friend]

    def __init__(
        self,
        chunks: Iterable[Chunk],
        load_method: Callable[[Chunk], BatchType],
        dump_method: Callable[[BatchType], RecordLike] = ...,
        dump_base_path: PathLike = ...,
        dump_naming: str | NameMapping = ...,
    ):
        self.__chunks = [*chunks]
        self.__loader = _chunk_loader(method=load_method)
        self.__dumper = _chunk_dumper(
            method=numpy_dumper if dump_method is ... else dump_method,
            base_path=dump_base_path,
            naming=dump_naming,
        )

    def batches(self, batch_size: int, name: str, **_):
        for chunk in Chunk.balance(batch_size, *self.__chunks):
            yield self.__dumper.new(chunk, name), self.__loader.new(chunk)


@dataclass(kw_only=True)
class _chunk_loader:
    chunk: Chunk = None
    method: Callable[[Chunk], BatchType]

    def __call__(self) -> BatchType:
        return self.method(self.chunk)

    def new(self, chunk: Chunk):
        return replace(self, chunk=chunk)


@dataclass(kw_only=True)
class _chunk_dumper:
    chunk: Chunk = None
    method: Callable[[BatchType], RecordLike]
    base_path: PathLike
    naming: str | NameMapping
    name: str = None

    def __len__(self):
        return len(self.chunk)

    def __call__(self, batch: BatchType) -> Friend:
        with Friend(name=self.name).auto_dump(
            base_path=self.base_path, naming=self.naming
        ) as friend:
            friend.add(self.chunk, self.method(batch))
            return friend

    def new(self, chunk: Chunk, name: str):
        return replace(self, chunk=chunk, name=name)
