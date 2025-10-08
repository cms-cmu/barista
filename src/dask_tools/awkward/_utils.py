from __future__ import annotations

from typing import TYPE_CHECKING

import awkward as ak
from dask_awkward.lib.core import (
    is_typetracer,
    make_unknown_length,
)

if TYPE_CHECKING:
    import numpy.typing as npt

to_typetracer = make_unknown_length


def maybe_typetracer(array: ak.Array) -> ak.Array:
    if is_typetracer(array):
        return ak.typetracer.length_zero_if_typetracer(array)
    return array


def len_maybe_typetracer(array: ak.Array, typetracer: int = 0) -> int:
    if is_typetracer(array):
        return typetracer
    return len(array)


def to_numpy_maybe_typetracer(
    array: ak.Array, allow_missing: bool = True
) -> npt.NDArray:
    return maybe_typetracer(array).to_numpy(allow_missing=allow_missing)
