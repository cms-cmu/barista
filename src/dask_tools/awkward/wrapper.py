from __future__ import annotations

from dataclasses import dataclass
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    Concatenate,
    Generic,
    Optional,
    ParamSpec,
    Protocol,
    TypeVar,
    overload,
)

import awkward as ak
import dask_awkward as dak
import dask_awkward.lib.core as dakcore
from dask.base import unpack_collections

from ._utils import is_typetracer, to_typetracer

T = TypeVar("T")
P = ParamSpec("P")

T2 = TypeVar("T2")
P2 = ParamSpec("P2")


@dataclass
class _RepackWrapper:
    fn: Callable
    args: Callable
    kwargs: Callable
    division: int
    meta: Optional[Callable]

    def __call__(self, *collections):
        typetracing = any(is_typetracer(c) for c in collections)
        result = (self.meta if typetracing and self.meta is not None else self.fn)(
            *self.args(collections[: self.division])[0],
            **self.kwargs(collections[self.division :])[0],
        )
        if typetracing and not is_typetracer(result):
            result = to_typetracer(result)
        return result


class _PartitionMappingDecorator(Protocol):
    def __call__(self, func: Callable[P, T]) -> _PartitionMappingWrapper[P, T]: ...


@dataclass
class _PartitionMappingWrapper(Generic[P, T]):
    func: Callable[P, T]
    label: Optional[str] = None
    token: Optional[str] = None
    meta: Callable[P, ak.Array] = None
    output_divisions: Optional[int] = None
    traverse: bool = True

    def __post_init__(self):
        if self.label is None:
            self.label = self.func.__name__

    def typetracer(self, func: Callable[P, ak.Array]):
        self.meta = func

    @overload
    def __get__(self, instance: None, owner) -> _PartitionMappingWrapper[P, T]: ...
    @overload
    def __get__(
        self: Callable[Concatenate[Any, P2], T2], instance: Any, owner
    ) -> Callable[P2, T2]: ...
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return partial(self, instance)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T | dak.Array:
        arg_col, arg_repack = unpack_collections(args, traverse=self.traverse)
        kwarg_col, kwarg_repack = unpack_collections(kwargs, traverse=self.traverse)
        if (len(arg_col) + len(kwarg_col)) == 0:
            return self.func(*args, **kwargs)
        func = _RepackWrapper(
            fn=self.func,
            args=arg_repack,
            kwargs=kwarg_repack,
            division=len(arg_col),
            meta=self.meta,
        )
        return dakcore._map_partitions(
            func,
            *arg_col,
            *kwarg_col,
            label=self.label,
            token=self.token,
            meta=None,
            output_divisions=self.output_divisions,
        )


@overload
def partition_mapping(
    func: Callable[P, T],
    /,
    typehint: None = None,
    label: Optional[str] = None,
    token: Optional[str] = None,
    meta: Optional[Callable[P, ak.Array]] = None,
    output_divisions: Optional[int] = None,
    traverse: bool = True,
) -> _PartitionMappingWrapper[P, T]: ...
@overload
def partition_mapping(
    func: None = None,
    /,
    typehint: None = None,
    label: Optional[str] = None,
    token: Optional[str] = None,
    meta: Optional[Callable[P, ak.Array]] = None,
    output_divisions: Optional[int] = None,
    traverse: bool = True,
) -> _PartitionMappingDecorator: ...
@overload
def partition_mapping(
    func: None = None,
    /,
    typehint: Callable[P, T] = ...,
    label: Optional[str] = None,
    token: Optional[str] = None,
    meta: Optional[Callable[P, ak.Array]] = None,
    output_divisions: Optional[int] = None,
    traverse: bool = True,
) -> Callable[[Callable], _PartitionMappingWrapper[P, T]]: ...
def partition_mapping(
    func=None,
    /,
    typehint=None,
    **kwargs,
):
    if func is None:
        return partial(partition_mapping, typehint=typehint, **kwargs)
    else:
        if typehint is None:
            typehint = func
        return wraps(typehint)(_PartitionMappingWrapper(func, **kwargs))
