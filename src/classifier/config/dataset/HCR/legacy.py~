# TODO: this should be removed after new skim
from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.task.special import Deprecated

from ...setting.df import Columns
from ...setting.HCR import Input, InputBranch
from . import _common, _group

if TYPE_CHECKING:
    import pandas as pd

_WEIGHT = "mcPseudoTagWeight"
_CANJETS = (
    *(
        f"canJet{i}_{f}"
        for f in ("pt", "eta", "phi", "m")
        for i in range(InputBranch.n_CanJet)
    ),
)
_NOTCANJETS = (*(f"notCanJet_{f}" for f in ("pt", "eta", "phi", "m", "isSelJet")),)


def _add_isSelJet(df: pd.DataFrame):
    if "notCanJet_isSelJet" not in df.columns:
        import numpy as np
        import pandas as pd
        from awkward_pandas import AwkwardExtensionArray

        pt = df["notCanJet_pt"].ak.array
        eta = df["notCanJet_eta"].ak.array
        df.loc[:, "notCanJet_isSelJet"] = pd.Series(
            AwkwardExtensionArray(1 * ((pt > 40) & (np.abs(eta) < 2.4))),
            index=df.index,
        )
    return df


class _Legacy(Deprecated, _common.Common):
    def __init__(self):
        super().__init__()

        # fmt: off
        (
            self.to_tensor
            .remove(Input.CanJet)
            .add(Input.CanJet, "float32").columns(*_CANJETS)
            .remove(Input.NotCanJet)
            .add(Input.NotCanJet, "float32").columns(*_NOTCANJETS, target=InputBranch.n_NotCanJet, pad_value=-1)
        )
        # fmt: on

        self.preprocessors.append(_add_isSelJet)

    def preprocess_by_group(self):
        from classifier.df.tools import add_columns, rename_columns

        return [
            _group.fullmatch(
                ("label:ttbar",),
                processors=[lambda: rename_columns(**{_WEIGHT: Columns.weight})],
            ),
            _group.fullmatch(
                ("label:data",),
                processors=[lambda: add_columns(**{Columns.weight: 1.0})],
            ),  # HACK: JCM weight for 3b data is not applied either. To apply, add --preprocess label:data
        ] + list(super().preprocess_by_group())

    def other_branches(self):
        branches = super().other_branches()
        branches -= {Columns.weight}
        branches |= {_WEIGHT}
        return branches.union(_CANJETS, _NOTCANJETS)


class _CommonTrain(_Legacy, _common.CommonTrain): ...


class Eval(_Legacy, _common.CommonEval):
    evaluable = True
