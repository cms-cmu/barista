from __future__ import annotations

import logging
import sys

import tblib.pickling_support
from classifier.config.setting import monitor as cfg

from ..core import Recorder
from ._redirect import MultiPlatformHandler


def _excepthook(exc_type, exc_value, exc_traceback):
    logging.error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback),
    )


def _common():
    tblib.pickling_support.install()
    if cfg.Log.forward_exception:
        sys.excepthook = _excepthook


def disable_monitor():
    logging.basicConfig(handlers=[logging.NullHandler()], level=None)


@cfg.check(cfg.Log, default=disable_monitor, is_callable=True)
def setup_reporter():
    _common()
    return logging.basicConfig(
        handlers=[MultiPlatformHandler()],
        level=cfg.Log.level,
    )


@cfg.check(cfg.Log, default=disable_monitor, is_callable=True)
def setup_monitor():
    _common()
    handlers = []
    if cfg.Console.enable:
        from ..backends.console import Dashboard as _CD
        from ._console import ConsoleDump, ConsoleHandler

        ConsoleDump.init()
        Recorder.to_dump(cfg.Log.file, ConsoleDump.serialize)
        handlers.append(ConsoleDump.handler)
        handlers.append(ConsoleHandler.new(_CD.console))
    if handlers:
        return logging.basicConfig(
            handlers=[MultiPlatformHandler.init(handlers=handlers)],
            level=cfg.Log.level,
        )
