from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.config.setting.df import Columns
from classifier.df.tools import _type_str

if TYPE_CHECKING:
    import pandas as pd


class apply_JCM:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, df: pd.DataFrame):
        df.loc[df["threeTag"], Columns.weight] *= df.loc[df["threeTag"], self.name]
        return df

    def __repr__(self):
        return f"{_type_str(self)} {self.name}"


class undo_JCM(apply_JCM):
    def __call__(self, df: pd.DataFrame):
        df.loc[df["threeTag"], Columns.weight] /= df.loc[df["threeTag"], self.name]
        return df
