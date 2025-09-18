from __future__ import annotations
from typing import TYPE_CHECKING
from classifier.task import ArgParser
from classifier.config.dataset.HCR import _group
from classifier.config.dataset.bbWW._common import CommonEval, CommonTrain
from . import _picoAOD

if TYPE_CHECKING:
    import pandas as pd

def _common_selection(df: pd.DataFrame):
    """Common selection for both signal and control regions"""
    return (df["CR"] | df["SR"])

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


class Train(CommonTrain):
    """Training dataset configuration for HH→bbWW classifier"""
    
    argparser = ArgParser()
    argparser.add_argument(
        "--no-SR",
        action="store_true",
        help="remove SR events from training",
    )
    argparser.add_argument(
        "--min-btags",
        type=int,
        default=2,
        help="minimum number of b-tagged jets",
    )
    argparser.add_argument(
        "--met-cut",
        type=float,
        default=50.0,
        help="minimum MET requirement (GeV)",
    )
    argparser.add_argument(
        "--lepton-pt-cut",
        type=float,
        default=20.0,
        help="minimum leading lepton pT (GeV)",
    )

    def preprocess_by_group(self):
        from classifier.df.tools import add_label_index_from_column, prescale

        ps = [
            _group.fullmatch(
                ("label:signal",),
                processors=[
                    lambda: _signal_selection,
                    lambda: add_label_index_from_column(CR="CR", SR="SR"),
                ],
                name="HH signal selection",
            ),

            _group.add_year(),
        ]

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

class TrainBaseline(_picoAOD.Signal, Train): 
    """Baseline training with background processes"""
    ...
class Eval(_picoAOD.Signal, CommonEval): 
    """Evaluation dataset for HH→bbWW classifier"""
    ...