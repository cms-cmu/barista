# TODO move to heptools.awkward
from __future__ import annotations

from functools import partial, reduce
from operator import add, and_, mul, or_
from typing import Any, Callable

import awkward as ak
import numpy as np
from awkward import Array

from .math_tools.partition import Partition
from .typetools import check_type
from .utils import astuple

__all__ = [
    "FieldLike",
    "RealNumber",
    "AnyInt",
    "AnyFloat",
    "has_record",
    "get_field",
    "set_field",
    "update_fields",
    "get_shape",
    "foreach",
    "partition_with_name",
    "between",
    "where",
    "sort",
    "or_arrays",
    "or_fields",
    "and_arrays",
    "and_fields",
    "add_arrays",
    "add_fields",
    "mul_arrays",
    "mul_arrays",
]

AnyInt = int | np.integer
AnyFloat = float | np.floating
RealNumber = AnyInt | AnyFloat

# field

FieldLike = str | tuple[str, ...]


def has_record(data: Array, field: FieldLike) -> tuple[str, ...]:
    parents = []
    for level in astuple(field):
        try:
            data = data[level]
            parents.append(level)
        except Exception:
            break
    return (*parents,)


def get_field(data: Array, field: FieldLike):
    if field is ...:
        try:
            return ak.num(data, axis=len(get_shape(data)) - 1)
        except Exception:
            return ak.Array(np.ones(len(data), dtype=np.int64))
    for level in astuple(field):
        if level in data.fields:
            data = data[level]
        else:
            data = getattr(data, level)
    return data


def set_field(data: Array, field: FieldLike, value: Array):
    field = astuple(field)
    parent = field[: len(has_record(data, field)) + 1]
    nested = field[len(parent) :]
    if nested:
        for level in reversed(nested):
            value = ak.zip({level: value})
    data[parent] = value


def cache_field(data: Array, field: FieldLike):
    field = astuple(field)
    if has_record(data, field) != field:
        set_field(data, field, get_field(data, field))


def update_fields(data: Array, new: Array, *fields: FieldLike):
    if not fields:
        fields = new.fields
    for field in fields:
        set_field(data, field, get_field(new, field))


# shape


def get_shape(data) -> list[str]:
    if isinstance(data, np.ndarray):
        return [str(i) for i in data.shape] + [str(data.dtype)]
    elif isinstance(data, Array):
        _type = ak.type(data)
        shape = [str(_type)]
        while _type is not None:
            try:
                _type = _type.type
                _str = str(_type)
                shape[-1] = shape[-1].removesuffix(f" * {_str}")
                shape.append(_str)
            except AttributeError:
                _type = None
        return shape
    else:
        return [type(data).__name__]


# slice


def foreach(data: Array) -> tuple[Array, ...]:
    dim = len(get_shape(data)) - 2
    count = np.unique(ak.ravel(ak.num(data, axis=dim)))
    if not len(count) == 1:
        raise IndexError(f"the length of the last axis must be uniform (got {count})")
    slices = tuple(slice(None) for _ in range(dim))
    return tuple(data[slices + (i,)] for i in range(count[0]))


def partition_with_name(data: Array, groups: int, members: int) -> tuple[Array, ...]:
    _sizes = ak.num(data)
    if not ak.any(_sizes >= groups * members):
        raise ValueError(f"not enough data to partition into {groups}×{members}")
    _combs = ak.Array(
        [
            Partition(i, groups, members).combinations[0]
            for i in range(ak.max(_sizes) + 1)
        ]
    )[_sizes]
    _combs = tuple(
        ak.unflatten(data[ak.flatten(_combs[:, :, :, i], axis=2)], groups, axis=1)
        for i in range(members)
    )
    return _combs


def partition_concatenated(data: Array, groups: int, members: int) -> Array:
    _sizes = ak.num(data)
    if not ak.any(_sizes >= groups * members):
        raise ValueError(f"not enough data to partition into {groups}×{members}")
    _combs = ak.Array(
        [
            Partition(i, groups, members).combinations[0]
            for i in range(ak.max(_sizes) + 1)
        ]
    )[_sizes]
    _combs = ak.unflatten(
        ak.unflatten(
            data[ak.flatten(ak.flatten(_combs[:, :, :, :], axis=3), axis=2)],
            members,
            axis=1,
        ),
        groups,
        axis=1,
    )
    return _combs


# reduce


def op_arrays(*arrays: Array, op: Callable[[Array, Array], Array]) -> Array:
    if arrays:
        return reduce(op, arrays)


def op_fields(data: Array, *fields: FieldLike, op: Callable[[Array, Array], Array]):
    return op_arrays(*(get_field(data, field) for field in fields), op=op)


or_arrays = partial(op_arrays, op=or_)
or_fields = partial(op_fields, op=or_)
and_arrays = partial(op_arrays, op=and_)
and_fields = partial(op_fields, op=and_)
add_arrays = partial(op_arrays, op=add)
add_fields = partial(op_fields, op=add)
mul_arrays = partial(op_arrays, op=mul)
mul_fields = partial(op_fields, op=mul)

# search, sort


def between(data: Array, range: tuple[float, float]) -> Array:
    return (data > range[0]) & (data < range[1])


def where(default: Array, *conditions: tuple[Array, Any]) -> Array:
    for condition, value in conditions:
        default = ak.where(condition, value, default)
    return default


def sort(
    data: Array,
    value: FieldLike | Callable[[Array], Array],
    axis: int = -1,
    ascending: bool = False,
) -> Array:
    if check_type(value, FieldLike):
        value = get_field(data, value)
    elif isinstance(value, Callable):
        value = value(data)
    else:
        raise TypeError(f"cannot sort by <{type(value).__name__}>")
    return data[ak.argsort(value, axis=axis, ascending=ascending)]
