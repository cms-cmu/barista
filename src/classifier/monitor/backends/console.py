from __future__ import annotations

import atexit
import time
from threading import Lock, Thread
from typing import Callable

from rich.console import Console, RenderableType
from rich.live import Live

from ...config.setting import monitor as cfg


class UniqueGroup:
    def __init__(self):
        self._renderables: dict[int, RenderableType] = {}
        self._lock = Lock()

    def add(self, *renderables: RenderableType):
        with self._lock:
            for r in renderables:
                r_id = id(r)
                if r_id not in self._renderables:
                    self._renderables[r_id] = r

    def remove(self, *renderables: RenderableType):
        with self._lock:
            for renderable in renderables:
                self._renderables.pop(id(renderable), None)

    def __rich_console__(self, console, options):
        with self._lock:
            for renderable in self._renderables.values():
                yield renderable


class Dashboard:
    console: Console = None
    layout: UniqueGroup = None

    _lock: Lock = None
    _callbacks: list[Callable[[], None]] = []

    @classmethod
    def start(cls):
        with Live(
            cls.layout,
            refresh_per_second=cfg.Console.fps,
            transient=True,
            console=cls.console,
        ):
            while True:
                with cls._lock:
                    for callback in cls._callbacks:
                        callback()
                time.sleep(cfg.Console.interval)

    @classmethod
    @cfg.check(cfg.Console)
    def add(cls, callback: Callable[[], None]):
        with cls._lock:
            cls._callbacks.append(callback)


@cfg.check(cfg.Console)
def setup_backend():
    Dashboard.console = Console(markup=True)
    Dashboard.layout = UniqueGroup()
    Dashboard._lock = Lock()
    Thread(target=Dashboard.start, daemon=True).start()
    atexit.register(Dashboard.console.show_cursor)
