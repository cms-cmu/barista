from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np
from src.classifier.df.tools import _iter_str, _map_str, _type_str
from src.classifier.task import parse
from src.classifier.typetools import new_TypedDict

if TYPE_CHECKING:
    import pandas as pd


class JCMColumnNames(TypedDict, total=False):
    weight: str = "weight"
    n_jets: str = "nSelJets"
    selected: str = "threeTag"


class apply_JCM_from_list:
    def __init__(self, path: str | list[str], start: int = 4, columns: JCMColumnNames = None):
        if isinstance(path, (list, tuple)):
            path = path[0] if len(path) > 0 else ""
        weights: list[float] = parse.mapping(path, "file") if path else []
        if weights is None:
            weights = []
        self._weights = np.ones(start + len(weights), dtype=float)
        self._weights[start:] = weights
        self._columns = new_TypedDict(JCMColumnNames, **(columns or {}))

    def __call__(self, df: pd.DataFrame):
        if df is None or len(df) == 0:
            return df
        if self._columns["weight"] not in df.columns:
            df[self._columns["weight"]] = 1.0
        if self._columns["selected"] not in df.columns or self._columns["n_jets"] not in df.columns:
            return df
        mask = df[self._columns["selected"]]
        if not np.any(mask):
            return df
        n_jets = df.loc[mask, self._columns["n_jets"]].to_numpy(dtype=int)
        df.loc[mask, self._columns["weight"]] *= np.take(
            self._weights, n_jets, mode="clip"
        )
        return df

    def __repr__(self):
        return (
            f"{_type_str(self)}({_map_str(self._columns)}) {_iter_str(self._weights)}"
        )
