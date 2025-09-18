from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

import numpy as np
import numpy._typing as npt_
import numpy.typing as npt
import pandas as pd
import torch
from src.data_formats.root import Chain, Chunk, Friend

if TYPE_CHECKING:
    from .tools import DFProcessor


class FromRoot:
    def __init__(
        self,
        friends: Iterable[Friend] = None,
        branches: Callable[[set[str]], set[str]] = None,
        preprocessors: Iterable[DFProcessor] = None,
    ):
        self.chain = Chain()
        self.branches = branches
        self.preprocessors = [*(preprocessors or ())]

        if friends:
            self.chain += friends

    def __call__(self, chunk: Chunk) -> Optional[pd.DataFrame]:
        chain = self.chain.copy().add_chunk(chunk)
        df = chain.concat(library="pd", reader_options={"branch_filter": self.branches})
        for preprocessor in self.preprocessors:
            if len(df) == 0:
                return None
            df = preprocessor(df)
        return df


class ArrayOperation:
    def __init__(self):
        self.__ops = []

    def reshape(self, shape: npt_._ShapeLike):
        self.__ops.append(partial(np.reshape, newshape=shape))
        return self

    def moveaxis(self, source: npt_._ShapeLike, destination: npt_._ShapeLike):
        self.__ops.append(partial(np.moveaxis, source=source, destination=destination))
        return self

    def transpose(self, axes: npt_._ShapeLike = None):
        self.__ops.append(partial(np.transpose, axes=axes))
        return self

    def custom(self, func: Callable[[npt.NDArray], npt.NDArray]):
        self.__ops.append(func)
        return self

    def __call__(self, array: npt.NDArray) -> np.NDArray:
        for op in self.__ops:
            array = op(array)
        return array


class ToTensor:
    def __init__(self):
        self._columns: dict[
            str,
            tuple[
                npt.DTypeLike,
                list[tuple[str, Optional[int], Any, Optional[tuple[int, ...]]]],
            ],
        ] = {}
        self._current: str = None

    def remove(self, name: str):
        self._columns.pop(name, None)
        return self

    def add(
        self,
        name: str,
        dtype: npt.DTypeLike = None,
        transform: Callable[[npt.NDArray], npt.NDArray] = None,
    ):
        self._columns.setdefault(name, (dtype, transform, []))
        self._current = name
        return self

    def columns(
        self,
        *columns: str,
        target: int = None,
        pad_value: Any = 0,
        transform: Callable[[npt.NDArray], npt.NDArray] = None,
    ):
        if self._current is None:
            raise RuntimeError("Call add() to specify a name before adding columns")
        self._columns[self._current][-1].extend(
            (c, target, pad_value, transform) for c in columns
        )
        return self

    def tensor(self, data: pd.DataFrame):
        dataset: dict[str, torch.Tensor] = {}
        for name, (dtype, transform, columns) in self._columns.items():
            missing = [c for c, *_ in columns if c not in data]
            if missing:
                logging.warning(f"columns {missing} not found in dataframe")
                continue
            arrays = []
            for c, target, pad_value, col_transform in columns:
                if data[c].dtype == "awkward":
                    if target is None:
                        target = int(np.max(data[c].ak.num()))
                    array = np.ma.filled(
                        data[c].ak.pad_none(target, clip=True).ak.to_numpy(), pad_value
                    ).astype(dtype)
                else:
                    array = data[c].to_numpy(dtype)
                    if len(array.shape) == 1 and len(columns) > 1:
                        array = array[:, np.newaxis]
                if col_transform is not None:
                    array = col_transform(array)
                arrays.append(array)
            if len(arrays) == 1:
                to_tensor = arrays[0]
            else:
                to_tensor = np.concatenate(arrays, axis=-1)
            if to_tensor.dtype == np.uint64:  # workaround for no "uint64" in torch
                to_tensor = to_tensor.view(np.int64)
            if to_tensor.dtype == np.uint32:  # workaround for no "uint32" in torch
                to_tensor = to_tensor.view(np.int32)
            if transform is not None:
                to_tensor = transform(to_tensor)
            dataset[name] = torch.from_numpy(to_tensor)
        return dataset
