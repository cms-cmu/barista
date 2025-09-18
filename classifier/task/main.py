from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from collections import deque
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import fsspec

from ..utils import import_
from .analysis import Analysis
from .dataset import Dataset
from .model import Model
from .special import interface, new
from .state import GlobalSetting, _is_private
from .task import _DASH, Task

if TYPE_CHECKING:
    from types import ModuleType

    from classifier.config.state import Flags

_CLASSIFIER = "classifier"
_TEST = "test"
_CONFIG = "config"
_MAIN = "main"

_MODULE = "module"
_OPTION = "option"

_FROM = "from"
_TEMPLATE = "template"
_FLAG = "flag"


@dataclass
class _ModCtx:
    test: bool = False


class _Opts:
    opts: dict[str, type[Task]]

    @dataclass
    class T:
        type: type[Task]
        name: str = None

    def __init_subclass__(cls):
        cls.opts = {}
        for k, v in vars(cls).items():
            if isinstance(v, _Opts.T):
                v.name = k
                cls.opts[k] = v.type


class TaskOptions(_Opts):
    _T = _Opts.T

    setting = _T(GlobalSetting)
    dataset = _T(Dataset)
    model = _T(Model)
    analysis = _T(Analysis)


class EntryPoint:
    _mains = list(
        map(
            lambda x: x.removesuffix(".py"),
            filter(
                lambda x: x.endswith(".py") and not _is_private(x),
                os.listdir(Path(__file__).parent / f"../{_CONFIG}/{_MAIN}"),
            ),
        )
    )
    _tasks = TaskOptions.opts
    _reserved = [
        *(f"{_DASH}{k}" for k in chain(_tasks, (_FROM, _TEMPLATE, _FLAG))),
    ]

    @classmethod
    def _fetch_subargs(cls, args: deque):
        subargs = []
        while len(args) > 0 and args[0] not in cls._reserved:
            subargs.append(args.popleft())
        return subargs

    @classmethod
    def _fetch_config(cls, cat: str, ctx: _ModCtx):
        parts = [_CLASSIFIER, _CONFIG, cat]
        if ctx.test:
            parts.insert(1, _TEST)
        return parts

    @classmethod
    def __fetch_module_name(cls, module: str, cat: str, ctx: _ModCtx):
        mods = cls._fetch_config(cat, ctx) + module.split(".")
        return ".".join(mods[:-1]), mods[-1]

    @classmethod
    def _fetch_module(
        cls,
        module: str,
        cat: str,
        raise_error: bool = False,
        mock_flags: Flags = None,
        force_ctx: list[_ModCtx] = None,
    ) -> tuple[ModuleType, type[Task], tuple[str, str]]:
        from classifier.config.state import Flags

        if mock_flags is None:
            mock_flags = Flags

        if force_ctx is not None:
            ctxs = force_ctx
        else:
            ctxs = [_ModCtx()]
            if mock_flags.test:
                ctxs.insert(0, _ModCtx(test=True))

        for i, ctx in enumerate(ctxs):
            try:
                modname, clsname = cls.__fetch_module_name(module, cat, ctx)
                mods = import_(modname, clsname, True)
            except Exception:
                if raise_error and (i == len(ctxs) - 1):
                    raise
            else:
                return *mods, (modname, clsname)
        return None, None, (modname, clsname)

    def _fetch_all(self, *cats: str):
        from classifier.config.state import Flags

        self.tasks: dict[str, list[Task]] = {}
        for cat in cats:
            target = self._tasks[cat]
            self.tasks[cat] = []
            for imp, opts in self.args[cat]:
                _, cls, (_, clsname) = self._fetch_module(imp, cat, True)
                if not issubclass(cls, target):
                    raise TypeError(
                        f'Class "{clsname}" is not a subclass of "{target.__name__}"'
                    )
                else:
                    obj = new(cls, opts)
                    self.tasks[cat].append(obj)
                    if Flags.debug and obj.debug is not NotImplemented:
                        logging.debug(f"-{cat} {imp}", " ".join(opts))
                        try:
                            obj.debug()
                        except Exception as e:
                            logging.error(e, exc_info=e)

    @classmethod
    def _expand_module(cls, data: dict):
        import shlex

        from . import parse

        mod = data[_MODULE]
        opts = []
        for opt in data.get(_OPTION, []):
            if opt is None:
                opts.append("")
            elif isinstance(opt, str):
                opts.extend(shlex.split(opt))
            else:
                opts.append(parse.escape(opt))
        return mod, opts

    def _expand(self, *files: str, fetch_main: bool = False, formatter: str = None):
        from classifier.config.state import Flags

        from . import parse

        for file in files:
            args = parse.mapping(file, "file", formatter)
            if fetch_main and _MAIN in args:
                self.args[_MAIN] = self._expand_module(args[_MAIN])
            for cat in self._tasks:
                if cat in args:
                    for arg in args[cat]:
                        self.args[cat].append(self._expand_module(arg))
            if _FLAG in args:
                Flags.set(*args[_FLAG])

    def __init__(self, argv: list[str] = None, initializer: Callable[[], None] = None):
        from classifier.config.state import Flags, System

        if initializer is not None:
            initializer()
        if argv is None:
            argv = sys.argv.copy()

        self.entrypoint = Path(argv[0]).name
        self.cmd = " ".join(argv)
        self.args: dict[str, list[tuple[str, list[str]]]] = {k: [] for k in self._tasks}

        # fetch args for main task
        args = deque(argv[1:])
        if len(args) == 0:
            raise ValueError("No task specified")
        arg = args.popleft()
        self.args[_MAIN] = arg, self._fetch_subargs(args)
        if arg == _FROM:
            self._expand(*self.args[_MAIN][1], fetch_main=True)
        elif arg == _TEMPLATE:
            self._expand(
                *self.args[_MAIN][1][1:],
                fetch_main=True,
                formatter=self.args[_MAIN][1][0],
            )
        main: str = self.args[_MAIN][0]
        if main not in self._mains:
            raise ValueError(
                f'The first argument must be one of {self._mains}, got "{main}"'
            )
        System._init(main_task=main)

        # fetch args for other tasks
        while len(args) > 0:
            cat = args.popleft().removeprefix(_DASH)
            mod = args.popleft()
            opts = self._fetch_subargs(args)
            if cat == _FROM:
                self._expand(mod, *opts)
            elif cat == _TEMPLATE:
                self._expand(*opts, formatter=mod)
            elif cat == _FLAG:
                Flags.set(mod, *opts)
            else:
                self.args[cat].append((mod, opts))

        # fetch modules
        cls: type[Main] = self._fetch_module(
            f"{self.args[_MAIN][0]}.Main", _MAIN, True
        )[1]

        if cls.prelude is not NotImplemented:
            cls.prelude()

        all_cats = [*self._tasks]
        if not cls._no_init:
            self._fetch_all(all_cats[0])

        from ..config import setting as cfg
        from ..monitor import (
            connect_to_monitor,
            disable_monitor,
            full_address,
            setup_monitor,
            setup_reporter,
        )

        if cfg.Monitor.enable:
            if not cfg.Monitor.connect:
                setup_monitor()
                logging.info(f"Monitor is running at {full_address()}")
            else:
                connect_to_monitor()
                setup_reporter()
                logging.info(f"Connected to Monitor {full_address()}")

        else:
            disable_monitor()

        if not cls._no_init:
            from ..process import setup_context

            self._fetch_all(*all_cats[1:])
            setup_context()

        self.main: Main = new(cls, self.args[_MAIN][1])

    def run(self, reproducible: Callable = None):
        from classifier.config import setting as cfg
        from classifier.config.main.analyze import run_analyzer
        from classifier.config.state import System

        from ..monitor import Recorder, wait_for_monitor

        # run main task
        result = self.main.run(self)
        # run analysis on result
        if cfg.Analysis.enable:
            analysis = run_analyzer(self, [result])
            if analysis:
                result = (result or {}) | analysis
        logging.info(f"Total run time: {System.run_time()}")
        # wait for monitor
        wait_for_monitor()
        # dump state
        if (not self.main._no_state) and (not cfg.IO.states.is_null):
            cfg.save.parse([cfg.IO.states])
        # dump diagnostics
        Recorder.dump()
        # dump result
        if (result is not None) and (not cfg.IO.result.is_null):
            from src.utils.json import DefaultEncoder
            from classifier.config.setting import ResultKey

            result[ResultKey.uuid] = str(uuid.uuid1())
            result[ResultKey.command] = self.cmd
            if reproducible is not None:
                result[ResultKey.reproducible] = reproducible()
            serialized = json.dumps(result, cls=DefaultEncoder)
            with fsspec.open(cfg.IO.result, "wt") as f:
                f.write(serialized)


# main


class Main(Task):
    _no_state = False
    _no_init = False

    @classmethod
    @interface(optional=True)
    def prelude(cls): ...

    @interface
    def run(self, parser: EntryPoint) -> Optional[dict[str]]: ...
