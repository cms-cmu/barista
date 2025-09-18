import importlib

from classifier.config.state.static import MonitorExtensions

from ..process import status
from .backends import Platform
from .core import (
    Monitor,
    Recorder,
    connect_to_monitor,
    full_address,
    wait_for_monitor,
)
from .template import Index

__all__ = [
    "Platform",
    "Index",
    "Monitor",
    "Recorder",
    "connect_to_monitor",
    "wait_for_monitor",
    "setup_monitor",
    "disable_monitor",
    "setup_reporter",
    "full_address",
]

_PKG = "classifier.monitor"
_BACKENDS = "backends"


def setup_monitor():
    Monitor.start()

    for backend in MonitorExtensions.backends:
        mod = importlib.import_module(f"{_PKG}.{_BACKENDS}.{backend}")
        if hasattr(mod, "setup_backend"):
            mod.setup_backend()

    for component in MonitorExtensions.components:
        mod = importlib.import_module(f"{_PKG}.{component}")
        if hasattr(mod, "setup_monitor"):
            mod.setup_monitor()


def setup_reporter():
    for component in MonitorExtensions.components:
        mod = importlib.import_module(f"{_PKG}.{component}")
        if hasattr(mod, "setup_reporter"):
            mod.setup_reporter()


def disable_monitor():
    for backend in MonitorExtensions.backends:
        mod = importlib.import_module(f"{_PKG}.{_BACKENDS}.{backend}")
        if hasattr(mod, "disable_backend"):
            mod.disable_backend()

    for component in MonitorExtensions.components:
        mod = importlib.import_module(f"{_PKG}.{component}")
        if hasattr(mod, "disable_monitor"):
            mod.disable_monitor()


status.initializer.add_unique(setup_reporter)
