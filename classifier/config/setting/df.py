from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.task import GlobalSetting

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


class Columns(GlobalSetting):
    "Common column names."

    event: str = "event"
    weight: str = "weight"
    weight_raw: str = "weight_raw"

    label_index: str = "label_index"
    selection_index: str = "selection_index"

    index_dtype: DTypeLike = "uint8"
