from __future__ import annotations

import time
from copy import copy
from dataclasses import dataclass

from rich.progress import BarColumn, ProgressColumn, SpinnerColumn, TimeElapsedColumn
from rich.progress import Progress as _Bar

from ..config.setting import monitor as cfg
from ..typetools import PicklableLock, WithUUID
from ..utils import noop
from .core import CLIENT_NAME_WIDTH, MonitorProxy, Recorder, post_to_monitor

_FORMAT = "%H:%M:%S"
_UNKNOWN = "+--:--:--"

MessageType = str | tuple[str, ...]


@dataclass
class ProgressTracker(PicklableLock, WithUUID):
    msg: MessageType
    total: int

    def __post_init__(self):
        super().__init__()
        self.start_t = time.time()
        self.updated_t = self.start_t
        self.source = Recorder.name()
        self._completed = 0
        self._step = None
        self._update(updated=True)

    def _update(self, msg: MessageType = None, updated: bool = False, step: int = None):
        if (msg is not None) and (msg != self.msg):
            self.msg = msg
            updated = True
        if updated:
            new = copy(self)
            new.lock = None
            new._step = step
            Progress._update(new)

    def update(self, completed: int, msg: MessageType = None):
        with self.lock:
            updated = completed > self._completed
            if updated:
                self.updated_t = time.time()
                self._completed = completed
            self._update(msg, updated)

    def advance(
        self,
        step: int = 1,
        msg: MessageType = None,
        distributed: bool = False,
    ):
        with self.lock:
            updated = step > 0
            if updated:
                self.updated_t = time.time()
                self._completed += step
                if not distributed:
                    step = None
            self._update(msg, updated, step)

    def complete(self):
        self.update(self.total)

    @property
    def estimate(self):
        if self._completed > 0:
            return (
                self.updated_t - self.start_t
            ) / self._completed * self.total + self.start_t
        return None

    @property
    def is_finished(self):
        return self._completed >= self.total

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.complete()


class TimeRemainColumn(ProgressColumn):
    def render(self, task):
        estimate = task.fields.get("estimate")
        text = _UNKNOWN
        if estimate is not None:
            diff = estimate - time.time()
            if diff > 0:
                text = f"+{time.strftime(_FORMAT, time.gmtime(diff))}"
        return text


class Progress(MonitorProxy):
    _jobs: dict[tuple, ProgressTracker]
    _console_ids: dict[tuple, str]
    _console_bar: _Bar

    def __init__(self):
        self._jobs = {}
        self._console_bar = _Bar(
            SpinnerColumn(),
            TimeElapsedColumn(),
            TimeRemainColumn(),
            BarColumn(bar_width=None),
            "{task.completed}/{task.total} {task.description}",
            "\[{task.fields[source]}]",
            expand=True,
        )
        self._console_ids = {}

    @classmethod
    @cfg.check(cfg.Progress, default=noop)
    def new(cls, total: int, msg: MessageType = "") -> ProgressTracker:
        return ProgressTracker(msg=msg, total=total)

    @post_to_monitor(max_retry=1)
    @cfg.check(cfg.Progress)
    def _update(self, new: ProgressTracker):
        uuid = (new.source, new.uuid)
        old = self._jobs.get(uuid)
        if new._step is not None:
            if old is None:
                return
            new._completed = max(new._completed, old._completed + new._step)
        self._jobs[uuid] = new
        if new.is_finished:
            self._jobs.pop(uuid)

    @classmethod
    def _format_msg(cls, msg: MessageType):
        if isinstance(msg, str):
            return msg
        return "|".join(msg)

    @classmethod
    def _console_callback(cls):
        with cls.lock():
            jobs = cls._jobs.copy()

        for uuid, job in jobs.items():
            kwargs = {
                "description": cls._format_msg(job.msg),
                "completed": job._completed,
                "estimate": job.estimate,
                "source": f"{Recorder.registered(job.source):>{CLIENT_NAME_WIDTH}}",
            }
            if uuid not in cls._console_ids:
                cls._console_ids[uuid] = cls._console_bar.add_task(
                    total=job.total, **kwargs
                )
            else:
                cls._console_bar.update(task_id=cls._console_ids[uuid], **kwargs)
        for uuid in set(cls._console_ids) - set(jobs):
            try:
                cls._console_bar.remove_task(cls._console_ids.pop(uuid))
            except KeyError:
                ...


@cfg.check(cfg.Progress)
def setup_monitor():
    if cfg.Console.enable:
        from .backends.console import Dashboard as _CD

        _CD.layout.add(Progress._console_bar)
        _CD.add(Progress._console_callback)
