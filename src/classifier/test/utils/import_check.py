from __future__ import annotations

import builtins
import os
import pkgutil
import re
import sys
from pathlib import Path
from typing import Iterable
from unittest.mock import MagicMock


class ImportTracker:
    def __enter__(self):
        self.__import = builtins.__import__
        builtins.__import__ = self
        return self

    def __exit__(self, *_):
        builtins.__import__ = self.__import

    def __pattern(self, targets: Iterable[str]):
        return re.compile("|".join(rf"{t}|{t}\..*" for t in targets))

    def __init__(self, track: Iterable[str]):
        self.__sys = set(sys.modules)
        self.__track = self.__pattern(track)
        self.reset()

    def reset(self):
        self.imported = set()
        self.tracked = set()
        for key in [*sys.modules]:
            if key not in self.__sys:
                del sys.modules[key]

    def __call__(self, name, *args, **kwargs):
        self.imported.add(name)
        if self.__track.fullmatch(name) is not None:
            self.tracked.add(name)
            return MagicMock()
        try:
            return self.__import(name, *args, **kwargs)
        except ImportError:
            return MagicMock()
        except Exception:
            raise


def _is_private(modules: Iterable[str]):
    return any(m.startswith("_") for m in modules)


def walk_packages(
    module_path: str, import_path: str = None, skip_private: bool = False
):
    if import_path is None:
        import_p = Path(os.getcwd())
    else:
        import_p = Path(import_path)
    module_p = Path(module_path)
    if not module_p.is_absolute():
        module_p = import_p / module_p
    yield ".".join(module_p.relative_to(import_p).parts)
    for root, _, _ in os.walk(module_p):
        root = Path(root)
        parts = root.relative_to(import_p).parts
        for mod in pkgutil.iter_modules([str(root)]):
            module = parts + (mod.name,)
            if not (skip_private and _is_private(module)):
                yield ".".join(module)
