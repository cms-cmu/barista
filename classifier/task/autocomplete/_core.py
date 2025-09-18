import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import inspect
import os
from collections import deque
from multiprocessing.connection import Listener
from typing import Iterable

from classifier.config.main.help import _walk_configs
from classifier.config.state import Flags

from .. import main as m
from ..special import TaskBase
from ._bind import ADDRESS, FILEDIR, is_exit

_TIMEOUT = 300  # seconds


class _filedir(Exception): ...


def _subcomplete(cls: type[TaskBase], args: list[str]):
    last = args[-1] if args else ""
    yield from (i for i in m.EntryPoint._reserved if i.startswith(last))
    if (
        isinstance(cls, type)
        and issubclass(cls, TaskBase)
        and cls.autocomplete is not NotImplemented
    ):
        yield from cls.autocomplete(args)


def _special(cat: str, args: list[str], is_main: bool = False):
    if not args:
        return
    last = args[-1]
    if last.startswith(m._DASH):
        yield from _subcomplete(None, [last])
        return
    match cat:
        case m._FROM:
            raise _filedir
        case m._TEMPLATE:
            if len(args) > 1:
                raise _filedir
        case m._FLAG:
            if not is_main:
                for flag in Flags.__annotations__:
                    if flag.startswith(last):
                        yield flag
                return


def autocomplete(args: Iterable[str]):
    flags = Flags()
    args = deque(args[1:])
    main = args.popleft() if args else ""
    if len(args) == 0:
        for part in m.EntryPoint._mains + [m._FROM, m._TEMPLATE]:
            if part.startswith(main):
                yield part
        return
    subargs = m.EntryPoint._fetch_subargs(args)
    if len(args) == 0:
        if main in m.EntryPoint._mains:
            yield from _subcomplete(
                m.EntryPoint._fetch_module(
                    f"{main}.Main",
                    m._MAIN,
                    mock_flags=flags,
                )[1],
                subargs,
            )
        else:
            yield from _special(main, subargs, is_main=True)
        return
    while len(args) > 0:
        cat = args.popleft().removeprefix(m._DASH)
        mod = args.popleft() if args else None
        if len(args) == 0:
            if cat in m.EntryPoint._tasks:
                mod = mod or ""
                target = m.EntryPoint._tasks[cat]
                for imp, ctx in _walk_configs(cat, flags.test):
                    if imp:
                        imp = f"{imp}."
                    _imp = f"{imp}*"
                    _mod, _, (modname, _) = m.EntryPoint._fetch_module(
                        _imp, cat, force_ctx=[ctx]
                    )
                    if _mod is not None:
                        for name, obj in inspect.getmembers(_mod, inspect.isclass):
                            if (
                                issubclass(obj, target)
                                and not m._is_private(name)
                                and obj.__module__.startswith(modname)
                                and (clsname := f"{imp}{name}").startswith(mod)
                            ):
                                yield clsname
                return
        subargs = m.EntryPoint._fetch_subargs(args)
        if cat == m._FLAG:
            flags._set(mod, *subargs)
        if len(args) == 0:
            if cat in m.EntryPoint._tasks:
                yield from _subcomplete(
                    m.EntryPoint._fetch_module(mod or "", cat, mock_flags=flags)[1],
                    subargs,
                )
            else:
                if mod is not None:
                    subargs.insert(0, mod)
                yield from _special(cat, subargs)
            return


def pipe_server():
    try:
        os.remove(ADDRESS)
    except FileNotFoundError:
        ...
    listener = Listener(ADDRESS)
    listener._listener._socket.settimeout(_TIMEOUT)
    while True:
        conn = None
        try:
            conn = listener.accept()
            args: list[str] = conn.recv()
            if is_exit(args):
                break
            conn.send(list(autocomplete(args)))
        except _filedir:
            conn.send(FILEDIR)
        except TimeoutError:
            break
        finally:
            if conn is not None:
                conn.close()
    try:
        listener.close()
    except OSError:
        ...


if __name__ == "__main__":
    pipe_server()
