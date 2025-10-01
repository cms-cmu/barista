from __future__ import annotations
from typing import TYPE_CHECKING
from functools import partial, reduce

from src.classifier.task import ArgParser
from src.classifier.config.dataset.HCR import _group
from src.classifier.config.dataset.bbWW._common import CommonEval, CommonTrain
from src.classifier.config.dataset.bbWW import _picoAOD
from src.classifier.task import ArgParser, converter, parse
from src.classifier.config.setting.df import Columns

if TYPE_CHECKING:
    import pandas as pd

def _common_selection(df: pd.DataFrame):
    """Common selection for both signal and control regions"""

    return df["CR"] | df["SR"]

def _data_selection(df: pd.DataFrame):
    """Data selection excluding signal region events"""
    return df[_common_selection(df) & (~df["SR"])]

def _signal_selection(df: pd.DataFrame):
    """Signal selection for HH→bbWW analysis"""
    return df[_common_selection(df)]

def _select_sr(df: pd.DataFrame):
    """Select signal region events"""
    return df[df["SR"]]

def _select_cr(df: pd.DataFrame):
    """Select control region events"""
    return df[df["CR"]]

def _remove_sr(df: pd.DataFrame):
    """Remove signal region events"""
    return df[~df["SR"]]


def _remove_sr(df: pd.DataFrame):
    """Remove signal region events"""
    return df

def _norm(df: pd.DataFrame, norms: dict[int, float]):
    return df / (df.sum() / norms.get(df.name, 1.0))

class Train(CommonTrain):
    """Training dataset configuration for HH→bbWW classifier"""
    
    argparser = ArgParser()
    argparser.add_argument(
        "--no-SR",
        action="store_true",
        help="remove SR events from training",
    )

    argparser.add_argument(
        "--norm-by-label",
        action="store_true",
        help="normalize weights so each label sums to 1",
    )

    argparser.add_argument(
        "--norms",
        default=None,
        help="custom normalization factors per label (json format)",
    )

    def preprocess_by_group(self):
        from src.classifier.df.tools import add_label_index, add_label_index_from_column, prescale

        ps = []
        ps.append(
            _group.fullmatch(
                ("label:signal",),
                processors=[
                    lambda: _signal_selection,
                    lambda: add_label_index("signal"),
                ],
                name="HH signal selection",
            ),
        )
        if "ttbar" in self.mc_processes:
            ps.append(
                _group.fullmatch(
                    ("label:ttbar",),
                    processors=[
                        lambda: _signal_selection,
                        lambda: add_label_index("ttbar"),
                    ],
                    name="ttbar selection",
                ),
            )
        _group.add_year(),

        # Optional SR removal
        if self.opts.no_SR:
            ps.append(
                _group.fullmatch(
                    (),
                    processors=[
                        lambda: _remove_sr,
                    ],
                    name="remove signal region",
                )
            )

        return list(super().preprocess_by_group()) + ps

class Background(_picoAOD.Background, Train):
    argparser = ArgParser()
    argparser.add_argument(
        "--norm",
        default=1.0,
        type=converter.float_pos,
        help="normalization factor",
    )

    def __init__(self):
        from src.classifier.df.tools import drop_columns

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


class Signal(_picoAOD.Signal, Train):
    argparser = ArgParser()
    argparser.add_argument(
        "--norm-ignore-kl",
        default = True,
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
        ignore_kl = self.opts.norm_ignore_kl
        norms = self.opts.norms_by_label
        if norms is not None:
            norms = parse.mapping(norms)
        self.postprocessors.insert(
            0, partial(self.normalize,ignore_kl=ignore_kl, norms=norms)
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

        df.loc[:, "weight"] = (
            df
            .groupby([Columns.label_index], dropna=False)["weight"]
            .transform(partial(_norm, norms=norms))
        )
        
        return df

class TrainBaseline(_picoAOD.Signal, _picoAOD.Background, Train): 
    """Baseline training with signal and background processes"""
    ...

class Eval(_picoAOD.Signal, _picoAOD.Background, CommonEval): 
    """Evaluation dataset for HH→bbWW classifier"""
    ...