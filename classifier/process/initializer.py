from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

from src.utils import unique

from ..utils import append_unique_instance

if TYPE_CHECKING:
    from . import Context


class _initializer:
    def __init__(self, *funcs: Callable):
        self._funcs: list[Callable] = []
        self.add(*funcs)

    def add(self, *funcs: Callable):
        self._funcs.extend(funcs)

    def add_unique(self, func):
        append_unique_instance(self._funcs, func)

    def __call__(self):
        for func in self._funcs:
            func()


class status:
    context: Context = None
    initializer = _initializer()


class _inherit_context_initializer:
    def __getstate__(self):
        return (status.context, status.initializer)

    def __setstate__(self, states: tuple[Context, _initializer]):
        self._context, self._initializer = states

    def __call__(self):
        status.context = self._context
        status.initializer = self._initializer


status.initializer.add_unique(_inherit_context_initializer)


class torch_set_sharing_strategy:
    def __init__(
        self, strategy: Literal["file_system", "file_descriptor"] = "file_system"
    ):
        import torch.multiprocessing as mp

        strategies = mp.get_all_sharing_strategies()
        if strategy not in strategies:
            raise ValueError(
                f'Unknown strategy "{strategy}", available strategies are {strategies}'
            )
        self.strategy = strategy

    def __call__(self):
        import torch.multiprocessing as mp

        mp.set_sharing_strategy(self.strategy)


def setup_context():
    from classifier.config.setting import Multiprocessing as cfg

    from . import get_context

    status.context = get_context(
        method=cfg.context_method,
        library=cfg.context_library,
        preload=unique(cfg.preload),
    )
    if cfg.context_library == "torch":
        status.initializer.add(torch_set_sharing_strategy(cfg.torch_sharing_strategy))
