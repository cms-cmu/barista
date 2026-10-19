from __future__ import annotations

import logging
import sys

import tblib.pickling_support
from src.classifier.config.setting import monitor as cfg

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


def _render_arg(arg) -> str:
    if isinstance(arg, str):
        return arg
    from rich.console import Console
    from rich.protocol import is_renderable

    if not is_renderable(arg):
        return repr(arg)
    console = Console(width=160, no_color=True, force_terminal=False)
    with console.capture() as capture:
        console.print(arg)
    return capture.get().rstrip()


class _PlainFormatter(logging.Formatter):
    """The classifier logs rich renderables as extra arguments, e.g. ``logging.info("title", table)``,
    which the monitor's handlers render after the message (see MultiPlatformHandler.emit). A plain
    handler would raise on ``msg % args`` ("--- Logging error ---"), so print them below the message."""

    def format(self, record: logging.LogRecord) -> str:
        if record.args and isinstance(record.msg, str):
            try:
                record.msg % record.args
            except (TypeError, ValueError):
                record = logging.makeLogRecord(record.__dict__)  # other handlers keep the original
                record.msg = "\n".join([record.msg, *map(_render_arg, record.args)])
                record.args = None
        return super().format(record)


def disable_monitor():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_PlainFormatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logging.basicConfig(handlers=[handler], level=cfg.Log.level)


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
