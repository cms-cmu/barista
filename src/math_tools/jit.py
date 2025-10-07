from inspect import getmro
from typing import TypeVar

_FuncT = TypeVar("_FuncT")


def allow_jit(func: _FuncT = None, **options) -> _FuncT:
    if func is None:
        return lambda f: allow_jit(f, **options)
    func.__numba_njit_options__ = options
    return func


class Compilable:
    @classmethod
    def jit(cls):
        from numba import njit

        mro = getmro(cls)
        for name in dir(cls):
            func = getattr(cls, name)
            if hasattr(func, "__numba_njit_options__"):
                opts = func.__numba_njit_options__
                del func.__numba_njit_options__
                for base in mro:
                    if name in vars(base):
                        setattr(base, name, njit(**opts)(func))
                        break
        return cls
