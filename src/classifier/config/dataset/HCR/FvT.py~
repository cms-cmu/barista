from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.task import ArgParser

from . import _group, _picoAOD
from ._common import CommonEval, CommonTrain

if TYPE_CHECKING:
    import pandas as pd


def _common_selection(df: pd.DataFrame):
    return (df["SB"] | df["SR"]) & (df["fourTag"] | df["threeTag"])


def _data_selection(df: pd.DataFrame):
    return df[_common_selection(df) & (~(df["SR"] & df["fourTag"])) & df["passHLT"]]


def _ttbar_selection(df: pd.DataFrame):
    return df[_common_selection(df)]


def _ttbar_3b_selection(df: pd.DataFrame):
    return df["threeTag"]


def _select_4b(df: pd.DataFrame):
    return df[df["fourTag"]]


def _select_3b(df: pd.DataFrame):
    return df[df["threeTag"]]


def _remove_sr(df: pd.DataFrame):
    return df[~df["SR"]]


class Train(CommonTrain):
    argparser = ArgParser()
    argparser.add_argument(
        "--no-SR",
        action="store_true",
        help="remove SR events",
    )
    argparser.add_argument(
        "--no-detector-4b",
        action="store_true",
        help="remove 4b detector data events",
    )
    argparser.add_argument(
        "--no-ttbar-3b",
        action="store_true",
        help="remove 3b ttbar events",
    )
    argparser.add_argument(
        "--ttbar-3b-prescale",
        default="10",
        help="prescale 3b ttbar events",
    )

    def preprocess_by_group(self):
        from classifier.df.tools import add_label_index_from_column, prescale

        ps = [
            _group.fullmatch(
                ("label:data",),
                processors=[
                    lambda: _data_selection,
                    lambda: add_label_index_from_column(threeTag="d3", fourTag="d4"),
                ],
                name="data selection",
            ),
            _group.fullmatch(
                ("source:mixed",),
                ("source:synthetic",),
                processors=[
                    lambda: _select_4b,
                ],
                name="remove 3b mixed/synthetic data",
            ),
            _group.add_year(),
        ]
        if self.opts.no_SR:
            ps.append(
                _group.fullmatch(
                    (),
                    processors=[
                        lambda: _remove_sr,
                    ],
                )
            )
        if self.opts.no_detector_4b:
            ps.append(
                _group.fullmatch(
                    ("source:detector",),
                    processors=[
                        lambda: _select_3b,
                    ],
                    name="remove 4b detector data",
                )
            )
        if not self.opts.no_ttbar_3b:
            ps.append(
                _group.fullmatch(
                    ("label:ttbar",),
                    processors=[
                        lambda: prescale(
                            scale=self.opts.ttbar_3b_prescale,
                            selection=_ttbar_3b_selection,
                            seed=("ttbar", 0),
                        ),
                        lambda: _ttbar_selection,
                        lambda: add_label_index_from_column(
                            threeTag="t3", fourTag="t4"
                        ),
                    ],
                    name="ttbar selection",
                )
            )
        else:
            ps.append(
                _group.fullmatch(
                    ("label:ttbar",),
                    processors=[
                        lambda: _select_4b,
                        lambda: _ttbar_selection,
                        lambda: add_label_index_from_column(fourTag="t4"),
                    ],
                    name="ttbar 4b selection",
                )
            )
        return list(super().preprocess_by_group()) + ps


class TrainBaseline(_picoAOD.Background, Train): ...


class Eval(_picoAOD.Data, CommonEval): ...
