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
            if df is None or len(df) == 0:
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

    def _convert_group(
        self,
        data: pd.DataFrame,
        dtype: npt.DTypeLike,
        transform: Callable[[npt.NDArray], npt.NDArray] | None,
        columns: list[tuple[str, Optional[int], Any, Optional[tuple[int, ...]]]],
        ref_counts: dict[str, int] | None = None,
    ) -> np.ndarray:
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
            if ref_counts is not None:
                ref_counts[c] -= 1
                if ref_counts[c] <= 0:
                    del data[c]
            if col_transform is not None:
                array = col_transform(array)
            arrays.append(array)
        if len(arrays) == 1:
            out = arrays[0]
        else:
            out = np.concatenate(arrays, axis=-1)
        del arrays
        if out.dtype == np.uint64:  # workaround for no "uint64" in torch
            out = out.view(np.int64)
        if out.dtype == np.uint32:  # workaround for no "uint32" in torch
            out = out.view(np.int32)
        if transform is not None:
            out = transform(out)
        return out

    def to_numpy(
        self, data: pd.DataFrame, release: bool = False
    ) -> dict[str, np.ndarray]:
        ref_counts: dict[str, int] | None = None
        if release:
            ref_counts = {}
            for _, _, columns in self._columns.values():
                if any(c not in data for c, *_ in columns):
                    continue
                for c, *_ in columns:
                    ref_counts[c] = ref_counts.get(c, 0) + 1

        dataset: dict[str, np.ndarray] = {}
        for name, (dtype, transform, columns) in self._columns.items():
            missing = [c for c, *_ in columns if c not in data]
            if missing:
                logging.warning(f"columns {missing} not found in dataframe")
                continue
            dataset[name] = self._convert_group(
                data, dtype, transform, columns, ref_counts=ref_counts
            )
        return dataset

    def tensor(
        self, data: pd.DataFrame, release: bool = False
    ) -> dict[str, torch.Tensor]:
        arrays = self.to_numpy(data, release=release)
        return {k: torch.from_numpy(v) for k, v in arrays.items()}

    def concat_chunks(
        self,
        chunks: list[dict[str, np.ndarray]],
        keep_idx: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        if not chunks:
            return {}
        dataset: dict[str, torch.Tensor] = {}
        keys = list(chunks[0].keys())
        for k in keys:
            if len(chunks) == 1:
                out = chunks[0].pop(k)
            else:
                shapes = [c[k].shape for c in chunks]
                dtype = chunks[0][k].dtype
                total_n = sum(s[0] for s in shapes)
                tail0 = shapes[0][1:]
                uniform = all(s[1:] == tail0 for s in shapes)
                if uniform:
                    out = np.empty((total_n, *tail0), dtype=dtype)
                    pos = 0
                    for c in chunks:
                        arr = c.pop(k)
                        n = arr.shape[0]
                        out[pos : pos + n] = arr
                        pos += n
                        del arr
                else:
                    cols = self._columns.get(k, (None, None, []))[2]
                    pad_val = cols[0][2] if cols else 0
                    n_cols = len(cols) if cols else 1
                    max_tail = tuple(
                        max(s[d] for s in shapes) for d in range(1, len(shapes[0]))
                    )
                    out = np.full((total_n, *max_tail), pad_val, dtype=dtype)
                    pos = 0
                    if n_cols > 1 and len(shapes[0]) == 2 and max_tail[0] % n_cols == 0:
                        max_t = max_tail[0] // n_cols
                        for c in chunks:
                            arr = c.pop(k)
                            n = arr.shape[0]
                            t_i = arr.shape[1] // n_cols
                            for j in range(n_cols):
                                out[pos : pos + n, j * max_t : j * max_t + t_i] = arr[
                                    :, j * t_i : (j + 1) * t_i
                                ]
                            pos += n
                            del arr
                    else:
                        for c in chunks:
                            arr = c.pop(k)
                            n = arr.shape[0]
                            slices = (
                                slice(pos, pos + n),
                                *(slice(0, dim) for dim in arr.shape[1:]),
                            )
                            out[slices] = arr
                            pos += n
                            del arr
            if keep_idx is not None:
                out = np.ascontiguousarray(out[keep_idx])
            dataset[k] = torch.from_numpy(out)
        return dataset

    def refresh_from_df(
        self, dataset: dict[str, torch.Tensor], df: pd.DataFrame
    ) -> None:
        for name, (dtype, transform, columns) in self._columns.items():
            if not columns:
                continue
            if any(c not in df.columns or df[c].dtype == "awkward" for c, *_ in columns):
                continue
            dataset[name] = torch.from_numpy(
                self._convert_group(df, dtype, transform, columns)
            )
