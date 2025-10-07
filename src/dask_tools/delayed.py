from typing import TypeVar

from ..utils.wrapper import OptionalDecorator

_DelayedFuncT = TypeVar("_DelayedFuncT")


class _Delayed(OptionalDecorator):
    @property
    def _switch(self):
        return "dask"

    def _decorate(self, __func):
        from dask import delayed

        return delayed(__func)


def delayed(__func: _DelayedFuncT) -> _DelayedFuncT:
    return _Delayed(__func)
