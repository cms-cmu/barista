from __future__ import annotations

from typing import TYPE_CHECKING

from src.classifier.task import ArgParser

from . import _group, _picoAOD
from ._common import CommonEval, CommonTrain

if TYPE_CHECKING:
    import pandas as pd


def _common_selection(df: pd.DataFrame):
    return (df["SB"] | df["SR"]) & df["fourTag"]


def _data_selection(df: pd.DataFrame):
    return df[_common_selection(df) & (~(df["SR"] & df["fourTag"])) & df["passHLT"]]


def _mixed_all_selection(df: pd.DataFrame):
    return df[_common_selection(df)]


def _ttbar_selection(df: pd.DataFrame):
    return df[_common_selection(df) & df["passHLT"]]  # Use this line for Run3
    #return df[_common_selection(df)]


def _remove_sr(df: pd.DataFrame):
    return df[~df["SR"]]


def _fill_nan(df: pd.DataFrame):
    import logging

    import numpy as np

    for col in ["xW", "xbW"]:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                logging.warning(f"MvD: filling {n_nan} NaN values in '{col}' with 0")
                df[col] = df[col].fillna(0)

    for col in ["NotCanJet_mass"]:
        if col not in df.columns:
            continue
        if df[col].dtype == "awkward":
            import awkward as ak

            ak_arr = df[col].ak.array
            flat = ak.to_numpy(ak.flatten(ak_arr))
            n_nan = int(np.isnan(flat).sum())
            if n_nan > 0:
                logging.warning(f"MvD: filling {n_nan} NaN in jagged '{col}' with 0")
                from awkward_pandas import AwkwardExtensionArray

                filled_flat = np.where(np.isnan(flat), np.float32(0), flat)
                df[col] = AwkwardExtensionArray(ak.unflatten(filled_flat, ak.num(ak_arr)))
        else:
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                logging.warning(f"MvD: filling {n_nan} NaN values in '{col}' with 0")
                df[col] = df[col].fillna(0)

    return df


class Train(CommonTrain):
    argparser = ArgParser()
    argparser.add_argument(
        "--no-SR",
        action="store_true",
        help="remove SR events",
    )

    def preprocess_by_group(self):
        from functools import partial

        from src.classifier.df.tools import add_label_index_from_column
        from src.classifier.task import parse

        ps = []
        if self.opts.JCM_weight:
            from coffea4bees.classifier.compatibility.JCM.fit import apply_JCM_from_list

            for opts in self.opts.JCM_weight:
                ps.append(
                    _group.fullmatch(
                        parse.split_nonempty(opts[0], ","),
                        processors=[
                            partial(
                                apply_JCM_from_list,
                                path=opts[1],
                                selected_col="fourTag",
                            )
                        ],
                    )
                )

        ps += [
            _group.fullmatch(
                ("source:detector",),
                processors=[
                    lambda: _data_selection,
                    lambda: add_label_index_from_column(fourTag="d4"),
                ],
                name="detector data selection",
            ),
            _group.fullmatch(
                ("source:mixed_all",),
                processors=[
                    lambda: _fill_nan,
                    lambda: _mixed_all_selection,
                    lambda: add_label_index_from_column(fourTag="mix4"),
                ],
                name="mixed_all 4b selection",
            ),
            _group.fullmatch(
                ("label:ttbar",),
                processors=[
                    lambda: _ttbar_selection,
                    lambda: add_label_index_from_column(fourTag="t4"),
                ],
                name="ttbar 4b selection",
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
        return ps


class TrainBaseline(_picoAOD.MixedAllBackground, Train): ...


class Eval(_picoAOD.Data, CommonEval): ...
