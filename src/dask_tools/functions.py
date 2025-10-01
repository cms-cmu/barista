from __future__ import annotations

from functools import partial, reduce
from typing import TYPE_CHECKING, Any, Callable, Optional

import dask

from ..math_tools.utils import balance_split

if TYPE_CHECKING:
    from dask.base import DaskMethodsMixin


def reduction(
    func: Callable[[Any, Any], Any],
    *tasks: DaskMethodsMixin,
    split_every: Optional[int] = None,
):
    aggregate = dask.delayed(partial(reduce, func))
    if split_every is None:
        return aggregate(tasks)
    while steps := balance_split(len(tasks), split_every):
        start, reduced = 0, []
        for step in steps:
            reduced.append(aggregate(tasks[start : start + step]))
            start += step
        if len(reduced) == 1:
            return reduced[0]
        tasks = reduced
