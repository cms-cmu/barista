from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np
from classifier.df.tools import _iter_str, _map_str, _type_str
from classifier.task import parse
from classifier.typetools import new_TypedDict

if TYPE_CHECKING:
    import pandas as pd


class JCMColumnNames(TypedDict, total=False):
    weight: str = "weight"
    n_jets: str = "nSelJets"
    selected: str = "threeTag"


class apply_JCM_from_list:
    def __init__(self, path: str, start: int = 4, columns: JCMColumnNames = None):
        weights: list[float] = parse.mapping(path, "file")
        self._weights = np.ones(start + len(weights), dtype=float)
        self._weights[start:] = weights
        self._columns = new_TypedDict(JCMColumnNames, **(columns or {}))

    def __call__(self, df: pd.DataFrame):
        n_jets = df.loc[df[self._columns["selected"]], self._columns["n_jets"]]
        df.loc[df[self._columns["selected"]], self._columns["weight"]] *= np.take(
            self._weights, n_jets, mode="clip"
        )
        return df

    def __repr__(self):
        return (
            f"{_type_str(self)}({_map_str(self._columns)}) {_iter_str(self._weights)}"
        )
