from __future__ import annotations

from typing import TYPE_CHECKING, Any

import awkward as ak
import numpy as np

from .convert import to_numpy

if TYPE_CHECKING:
    import numpy.typing as npt


class selected:
    def __init__(
        self,
        padded_value: Any = 0,
        jagged_size: int = 0,
    ):
        self._value = padded_value
        self._size = jagged_size

    def _pad_regular(self, array: npt.NDArray, selection: npt.NDArray):
        padded = np.full(len(selection), self._value, dtype=array.dtype)
        if len(array) == len(selection):
            array = array[selection]
        padded[selection] = array
        return ak.Array(padded)

    def _pad_jagged(
        self, array: npt.NDArray, count: npt.NDArray, selection: npt.NDArray
    ) -> ak.Array:
        shape = (len(selection) - len(count)) * self._size + count.sum()
        padded_count = np.full(len(selection), self._size, dtype=count.dtype)
        padded_count[selection] = count
        padded = np.full(shape, self._value, dtype=array.dtype)
        padded[np.repeat(selection, padded_count)] = array
        return ak.unflatten(padded, padded_count)

    def _pad(
        self,
        array: npt.NDArray | tuple[npt.NDArray, npt.NDArray],
        selection: npt.NDArray,
    ):
        if isinstance(array, tuple):
            return self._pad_jagged(*array, selection)
        else:
            return self._pad_regular(array, selection)

    def __call__(self, array: ak.Array, selection: npt.ArrayLike) -> ak.Array:
        selection = np.asarray(selection, dtype=bool)
        if ak.fields(array):
            return ak.zip(
                {
                    name: self._pad(arr, selection)
                    for name, arr in to_numpy(array).items()
                }
            )
        else:
            return self._pad(to_numpy(array), selection)
