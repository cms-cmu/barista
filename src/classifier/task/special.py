from __future__ import annotations

from typing import Any, Callable, Concatenate, Generator, ParamSpec, TypeVar, overload

from ..typetools import Method

__all__ = ["interface", "new", "TaskBase", "Static"]

_InterfaceP = ParamSpec("_InterfaceP")
_InterfaceReturnT = TypeVar("_InterfaceReturnT")


class InterfaceError(NotImplementedError):
    __module__ = NotImplementedError.__module__

    def __init__(self, owner, func):
        import inspect

        signature = str(inspect.signature(func)).replace("'", "").replace('"', "")
        super().__init__(
            f"Not implemented: {owner.__name__}.{func.__name__}{signature}"
        )


class _Interface:
    def __init__(self, func, optional: bool = True):
        self._func = func
        self._optional = optional

    def __get__(self, _, owner):
        if self._optional:
            return NotImplemented
        else:
            raise InterfaceError(owner, self._func)


@overload
def interface(
    func: Callable[Concatenate[Any, _InterfaceP], _InterfaceReturnT], /
) -> Method[_InterfaceP, _InterfaceReturnT]: ...
@overload
def interface(
    optional: bool = False,
) -> Callable[
    [Callable[Concatenate[Any, _InterfaceP], _InterfaceReturnT]],
    Method[_InterfaceP, _InterfaceReturnT],
]: ...
def interface(func=None, *, optional: bool = False):
    if func is None:
        return lambda func: _Interface(func, optional=optional)
    return _Interface(func, optional=optional)


class TaskBase:
    @interface
    def parse(self, opts: list[str]): ...

    @interface(optional=True)
    def debug(self): ...

    @classmethod
    @interface(optional=True)
    def autocomplete(cls, opts: list[str]) -> Generator[str, None, None]: ...

    @classmethod
    @interface
    def help(cls) -> str: ...


_TaskT = TypeVar("_TaskT", bound=TaskBase)


def new(cls: type[_TaskT], opts: list[str]) -> _TaskT:
    obj = cls.__new__(cls)
    obj.parse(opts)
    obj.__init__()
    return obj


class Static(TaskBase):
    @classmethod
    @interface
    def parse(cls, opts: list[str]): ...

    @classmethod
    @interface(optional=True)
    def debug(cls): ...

    def __new__(cls):
        return cls

    def __init__(): ...


class WorkInProgress: ...


class Deprecated: ...
