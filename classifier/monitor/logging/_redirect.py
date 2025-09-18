from __future__ import annotations

import logging
from typing import Iterable

from classifier.config.setting import monitor as cfg
from classifier.config.state.static import GitRepo

from ..backends import Platform
from ..core import Recorder, post_to_monitor


class MultiPlatformLogRecord(logging.LogRecord):
    msg: str | Platform

    def getMessage(self):
        return self.msg


class MultiPlatformHandler(logging.Handler):
    __instance: MultiPlatformHandler = None

    def __init__(self, handlers: Iterable[logging.Handler] = None):
        super().__init__()
        self._handlers = [*(handlers or ())]

    @classmethod
    def init(cls, **kwargs):
        if cls.__instance is None:
            cls.__instance = cls(**kwargs)
        return cls.__instance

    @cfg.check(cfg.Log)
    def emit(self, record: logging.LogRecord):
        record.__class__ = MultiPlatformLogRecord
        record.name = Recorder.name()
        record.pathname = GitRepo.get_url(record.pathname)
        if isinstance(record.msg, str) and record.args:
            try:
                record.msg = record.msg % record.args
                record.args = None
            except TypeError:
                ...
        self._emit(record)

    @post_to_monitor(max_retry=1)
    @cfg.check(cfg.Log)
    def _emit(self, record: MultiPlatformLogRecord):
        record.name = Recorder.registered(record.name)
        for handler in self._handlers:
            if hasattr(handler, "__platform__") and isinstance(record.msg, Platform):
                if handler.__platform__ not in record.msg:
                    continue
            handler.handle(record)
