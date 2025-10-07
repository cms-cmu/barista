from __future__ import annotations

import dask_awkward as dak
from hist.dask import Hist

from ..aktools import RealNumber
from ..hist_tools import H, Template
from ..hist_tools import hist as _h

__all__ = [
    "Collection",
    "Fill",
    "FillLike",
    "Template",
    "H",
]

FillLike = _h.LazyFill | RealNumber | bool | dak.Array


class Fill(_h._Fill[Hist]):
    class __backend__(_h._Fill.__backend__):
        ak = dak
        check_empty_mask = False
        anyarray = dak.Array


class Collection(_h._Collection[Hist, Fill]):
    class __backend__(_h._Collection.__backend__):
        fill = Fill
        hist = Hist
