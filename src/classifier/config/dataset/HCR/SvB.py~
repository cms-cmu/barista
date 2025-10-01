from __future__ import annotations

import operator as op
from functools import partial, reduce
from typing import TYPE_CHECKING

from classifier.config.setting.df import Columns
from classifier.config.state.label import MultiClass
from classifier.task import ArgParser, converter, parse

from . import _group, _picoAOD
from ._common import CommonEval, CommonTrain

if TYPE_CHECKING:
    import pandas as pd


def _reweight_bkg(df: pd.DataFrame):
    df.loc[:, "weight"] *= df["FvT"]
    return df


class _common_selection:
    ntags: str
    passHLT: bool = False

    def __init__(self, *regions: str):
        self.regions = regions

    def __call__(self, df: pd.DataFrame):
        selection = df[self.ntags] & reduce(op.or_, (df[r] for r in self.regions))
        if self.passHLT:
            selection &= df["passHLT"]
        return df[selection]

    def __repr__(self):
        from classifier.df.tools import _iter_str, _type_str

        selections = [self.ntags, *self.regions]
        if self.passHLT:
            selections.append("passHLT")
        return f"{_type_str(self)} {_iter_str(selections)}"


class _data_selection(_common_selection):
    ntags = "threeTag"
    passHLT = True


class _mc_selection(_common_selection):
    ntags = "fourTag"


def _remove_outlier(df: pd.DataFrame):
    # TODO: This is a temporary solution triggered by events with huge weights:
    return df.loc[df["weight"] < 1]


class _Train(CommonTrain):
    argparser = ArgParser()
    argparser.add_argument(
        "--regions",
        nargs="+",
        default=["SR"],
        help="Dijet mass regions",
    )

    def __init__(self):
        super().__init__()
        self.to_tensor.add("kl", "float32").columns("kl")

    def preprocess_by_group(self):
        import numpy as np

        ps = [
            _group.regex(
                "label:data",
                [
                    lambda: _data_selection(*self.opts.regions),
                    lambda: _reweight_bkg,
                ],
                [
                    lambda: _mc_selection(*self.opts.regions),
                ],
            ),
            _group.add_year(),
            _group.add_column(
                key="kl", pattern=r"kl:(?P<kl>.*)", default=np.nan, dtype=float
            ),
            _group.add_single_label({"data": "multijet"}),
            _group.regex(
                r"label:.*",
                [
                    lambda: _remove_outlier,
                ],
            ),
        ]

        return list(super().preprocess_by_group()) + ps


class Background(_picoAOD.Background, _Train):
    argparser = ArgParser()
    argparser.add_argument(
        "--norm",
        default=1.0,
        type=converter.float_pos,
        help="normalization factor",
    )

    def __init__(self):
        from classifier.df.tools import drop_columns

        super().__init__()
        self.postprocessors.insert(0, partial(self.normalize, norm=self.opts.norm))
        self.preprocessors.append(drop_columns("FvT"))

    def other_branches(self):
        return super().other_branches() | {"FvT"}

    @staticmethod
    def normalize(df: pd.DataFrame, norm: float):
        df.loc[:, "weight"] /= df["weight"].sum() / norm
        return df


def _norm(df: pd.DataFrame, norms: dict[int, float]):
    return df / (df.sum() / norms.get(df.name, 1.0))


class Signal(_picoAOD.Signal, _Train):
    argparser = ArgParser()
    argparser.add_argument(
        "--norm-ignore-kl",
        action="store_true",
        help="group the events by process regardless of kl and normalize each group to 1 (the events are still normalized by kl within each group)",
    )
    argparser.add_argument(
        "--norms-by-label",
        default=None,
        help="normalization factors for each label. if specified, --norm-ignore-kl will be enabled",
    )

    def __init__(self):
        super().__init__()
        norms = self.opts.norms_by_label
        ignore_kl = self.opts.norm_ignore_kl or (norms is not None)
        if norms is not None:
            norms = parse.mapping(norms)
        self.postprocessors.insert(
            0, partial(self.normalize, ignore_kl=ignore_kl, norms=norms)
        )

    def other_branches(self):
        return super().other_branches()

    @staticmethod
    def normalize(df: pd.DataFrame, ignore_kl: bool, norms: dict[str, float]):
        norms = {
            idx: norm
            for label, norm in (norms or {}).items()
            if (idx := MultiClass.index(label)) is not None
        }
        columns = [[Columns.label_index, "kl"]]
        if ignore_kl:
            columns.append([Columns.label_index])
        for col in columns:
            # fmt: off
            df.loc[:, "weight"] = (
                df
                .groupby(col, dropna=False)["weight"]
                .transform(partial(_norm, norms=norms))
            )
            # fmt: on
        return df


class Eval(
    _picoAOD.Background,
    _picoAOD.Signal,
    CommonEval,
): ...
