from __future__ import annotations

import re
from typing import Any, Callable, Generic, Iterable, Sized, TypeVar

__all__ = [
    "arg_set",
    "arg_new",
    "seqcall",
    "merge_op",
    "astuple",
    "unpack",
    "unique",
    "count",
    "Eval",
    "match_any",
]


def arg_set(arg, none=None, default=...):
    if arg is None:
        return none
    if arg is ...:
        return default
    return arg


def arg_new(
    arg,
    none: Callable[[]] = lambda: None,
    default: Callable[[]] = lambda: ...,
):
    if arg is None:
        return none()
    if arg is ...:
        return default()
    return arg


_SeqCallT = TypeVar("_SeqCallT")


class seqcall(Generic[_SeqCallT]):
    def __init__(self, *funcs: Callable[[_SeqCallT], _SeqCallT]):
        self._funcs = funcs

    def __call__(self, x: _SeqCallT) -> _SeqCallT:
        for func in self._funcs:
            x = func(x)
        return x


def merge_op(op, _x, _y):
    if _x is None:
        return _y
    elif _y is None:
        return _x
    else:
        return op(_x, _y)


def astuple(_o):
    return _o if isinstance(_o, tuple) else (_o,)


def unpack(__iter: Iterable) -> Any:
    __next = __iter
    while (
        isinstance(__next, Iterable)
        and isinstance(__next, Sized)
        and not isinstance(__next, str)
    ):
        if len(__next) == 1:
            __next = next(iter(__next))
        elif len(__next) == 0:
            return None
        else:
            return __iter
    return __next


_UniqueT = TypeVar("_UniqueT")


def unique(seq: Iterable[_UniqueT]):
    return list(dict.fromkeys(seq))


def count(seq: Iterable, value: Any) -> int:
    return sum(1 for i in seq if i == value)


_EvalT = TypeVar("_EvalT")


class Eval(Generic[_EvalT]):
    _quote_arg_pattern = re.compile(
        r"(?P<arg>"
        + r"|".join(
            [
                rf"((?<={i})[^\[\]\",=]*?(?={j}))"
                for i in [r"\[", ","]
                for j in [",", r"\]"]
            ]
        )
        + r")"
    )
    _eval_call_pattern = re.compile(r"\[(?P<arg>.*?)\]")

    def __init__(
        self, method: Callable[[], _EvalT] | dict[str, _EvalT], *args, **kwargs
    ):
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def __call__(self, expression: str) -> _EvalT:
        return eval(
            re.sub(
                self._eval_call_pattern,
                rf"self.method(\g<arg>,*self.args,**self.kwargs)",
                re.sub(self._quote_arg_pattern, r'"\g<arg>"', expression),
            )
        )

    def __getitem__(self, expression: str) -> _EvalT:
        return eval(
            re.sub(
                self._eval_call_pattern,
                rf"self.method[\g<arg>]",
                re.sub(self._quote_arg_pattern, r'"\g<arg>"', expression),
            )
        )


_TargetT = TypeVar("_TargetT")
_PatternT = TypeVar("_PatternT")


def match_any(
    target: _TargetT,
    patterns: _PatternT | Iterable[_PatternT],
    match: Callable[[_PatternT, _TargetT], bool],
):
    """
    Use :mod:`re` instead when matching :class:`str`.
    """
    if patterns is None:
        return False
    if patterns is ...:
        return True
    if not isinstance(patterns, Iterable) or isinstance(patterns, str):
        patterns = [patterns]
    for pattern in patterns:
        if match(pattern, target):
            return True
    return False


def map_executor(fn, *iterables):
    return (*map(fn, *iterables),)
