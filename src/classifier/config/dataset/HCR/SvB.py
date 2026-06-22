from __future__ import annotations

import operator as op
from fractions import Fraction
from functools import partial, reduce
from typing import TYPE_CHECKING

from src.classifier.config.setting.df import Columns
from src.classifier.config.state.label import MultiClass
from src.classifier.task import ArgParser, converter, parse

from . import _group, _picoAOD
from ._common import CommonEval, CommonTrain

if TYPE_CHECKING:
    import pandas as pd


def _reweight_bkg(df: pd.DataFrame, branch: str = "FvT"):
    """Scale the event weight by ``df[branch]``.

    The background reweighting column is configurable (via functools.partial at
    the call site) so variants can reweight by a different per-event weight
    (e.g. MvD for the mixeddata_all background) instead of the default FvT.

    IMPORTANT: this MUST stay a module-level function (used with partial), not a
    closure. The loader sends per-group processors to a ProcessPoolExecutor,
    which pickles them; a nested-function closure is not picklable and silently
    hangs the loader's pool feeder during data loading.
    """
    df.loc[:, "weight"] *= df[branch]
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
        from src.classifier.df.tools import _iter_str, _type_str

        selections = [self.ntags, *self.regions]
        if self.passHLT:
            selections.append("passHLT")
        return f"{_type_str(self)} {_iter_str(selections)}"


class _data_selection(_common_selection):
    ntags = "threeTag"
    passHLT = True


class _mc_selection(_common_selection):
    ntags = "fourTag"


class _mixed_selection(_common_selection):
    # mixeddata_all background: 4-tag, and NO passHLT (mixeddata_all is already
    # HLT-filtered upstream, unlike the 3-tag detector-data selection).
    ntags = "fourTag"


def _remove_outlier(df: pd.DataFrame):
    import logging
    n_total = len(df)
    n_neg = (df["weight"] < 0).sum()
    n_pos = (df["weight"] >= 1).sum()
    if n_neg > 0 or n_pos > 0:
        logging.info(
            f"Outlier removal: removing {n_neg} events with negative weights (< 0) and "
            f"{n_pos} events with extreme weights (>= 1) out of {n_total} total events."
        )
    return df.loc[(df["weight"] >= 0) & (df["weight"] < 1)]


def _subsample(df: pd.DataFrame, fraction: float, seed: int):
    """Randomly keep ``fraction`` of the events (reproducible via ``seed``).

    Used for the statistics studies (C3/C4): applied uniformly to every group
    so signal, TT and data are all reduced by the same fraction, emulating a
    lower integrated luminosity. ``fraction >= 1`` is a no-op (keeps all rows
    unshuffled), so the nominal training is unaffected.

    MUST stay a module-level function (used with functools.partial); a closure
    would not pickle and would hang the loader's process pool.
    """
    if fraction >= 1.0:
        return df
    return df.sample(frac=fraction, random_state=seed)


class _Train(CommonTrain):
    # Background selection/reweighting knobs. Defaults reproduce the nominal
    # detector-3b + FvT behavior exactly; subclasses (e.g. BackgroundMixed)
    # override them for the mixeddata_all + MvD background.
    _data_selection_cls: type[_common_selection] = _data_selection
    _weight_branch: str = "FvT"

    argparser = ArgParser()
    argparser.add_argument(
        "--regions",
        nargs="+",
        default=["SR"],
        help="Dijet mass regions",
    )
    argparser.add_argument(
        "--subsample",
        default="1",
        help="fraction of events to randomly keep per group (statistics study)."
        " Accepts a float or fraction, e.g. 0.1 or 1/10. Default 1 = keep all.",
    )
    argparser.add_argument(
        "--subsample-seed",
        type=int,
        default=0,
        help="random seed for --subsample (reproducible subset)",
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
                    lambda: self._data_selection_cls(*self.opts.regions),
                    lambda: partial(_reweight_bkg, branch=self._weight_branch),
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
                    lambda: partial(
                        _subsample,
                        fraction=float(Fraction(self.opts.subsample)),
                        seed=self.opts.subsample_seed,
                    ),
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
        from src.classifier.df.tools import drop_columns

        super().__init__()
        self.postprocessors.insert(0, partial(self.normalize, norm=self.opts.norm))
        self.preprocessors.append(drop_columns(self._weight_branch))

    def other_branches(self):
        return super().other_branches() | {self._weight_branch}

    @staticmethod
    def normalize(df: pd.DataFrame, norm: float):
        df.loc[:, "weight"] /= df["weight"].sum() / norm
        return df


class BackgroundMixed(Background):
    """SvB background from mixeddata_all (4-tag) reweighted by MvD.

    Differs from the nominal FvT-based ``Background`` only in the selection
    (4-tag, no passHLT) and the per-event reweighting column (MvD). Everything
    else — normalization to ``--norm``, region selection, outlier removal — is
    inherited unchanged. Use with a train config that sets
    ``--data-source mixed_all`` and supplies the MvD weight friend.
    """

    _data_selection_cls = _mixed_selection
    _weight_branch = "MvD"


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
