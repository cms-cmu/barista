"""Compatibility shim for unpickling numpy>=2.0 data in a numpy 1.x environment.

Files pickled under numpy>=2.0 (e.g. .coffea hist files produced inside the
analysis container) reference ``numpy._core.*``, which was named
``numpy.core.*`` in numpy 1.x. Without this shim, ``coffea.util.load`` /
``cloudpickle.load`` raises ``ModuleNotFoundError: No module named
'numpy._core'`` in numpy 1.x environments (coffea 0.7.x pins numpy<1.24, so
upgrading numpy is not an option there).

Importing this module aliases ``numpy._core[.sub]`` -> ``numpy.core[.sub]`` in
``sys.modules``; pickle's module lookup short-circuits on ``sys.modules``, so
no import-machinery hooks are needed. Under numpy>=2.0 (where ``numpy._core``
exists) this is a no-op.
"""

import importlib
import pkgutil
import sys

import numpy as np

if not hasattr(np, "_core"):
    import numpy.core

    sys.modules.setdefault("numpy._core", np.core)
    for _submodule in pkgutil.iter_modules(np.core.__path__):
        _name = f"numpy.core.{_submodule.name}"
        try:
            _mod = importlib.import_module(_name)
        except Exception:
            # some numpy.core submodules are build-time only (e.g. setup_common)
            continue
        sys.modules.setdefault(f"numpy._core.{_submodule.name}", _mod)
