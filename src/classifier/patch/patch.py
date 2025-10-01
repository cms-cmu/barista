from __future__ import annotations

import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Callable

from ..process import status
from ..task import GlobalState

if TYPE_CHECKING:
    from importlib.abc import Loader
    from types import ModuleType


class _PatchedLoader:
    def __init__(
        self,
        fullname: str,
        loader: Loader,
    ):
        self.fullname = fullname
        self.loader = loader

    def __getattr__(self, __name: str):
        return getattr(self.loader, __name)

    def exec_module(self, module: ModuleType):
        if hasattr(self.loader, "exec_module"):
            self.loader.exec_module(module)
        if self.fullname in Patch._post_import:
            for func in Patch._post_import[self.fullname]:
                func(module)


def _install_patch():
    if Patch not in sys.meta_path:
        sys.meta_path.insert(0, Patch)
        status.initializer.add_unique(_install_patch)


class Patch(GlobalState):
    _post_import: dict[str, list[Callable[[ModuleType], None]]] = defaultdict(list)

    @classmethod
    def find_spec(cls, fullname, path, target=None):
        if fullname in cls._post_import:
            for finder in sys.meta_path:
                if finder is not cls:
                    spec = finder.find_spec(fullname, path, target)
                    if spec is not None:
                        spec.loader = _PatchedLoader(fullname, spec.loader)
                        return spec
        return None

    @classmethod
    def register(cls, fullname: str):
        _install_patch()

        def wrapper(func):
            if func not in cls._post_import[fullname]:
                cls._post_import[fullname].append(func)
            return func

        return wrapper
