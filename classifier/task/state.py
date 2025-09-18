from __future__ import annotations

from textwrap import indent
from typing import Any

from ..process import status
from ..typetools import dict_proxy
from .special import Static
from .task import _INDENT

_MAX_WIDTH = 30

_RAW = "raw__"
_GET = "get__"
_SET = "set__"


def _is_private(name: str):
    return name.startswith("_")


def _is_special(name: str):
    return name.startswith("__") and name.endswith("__")


def _is_state(var: tuple[str, Any]):
    name, value = var
    return not _is_special(name) and not isinstance(value, classmethod)


class GlobalState:
    _states: list[type[GlobalState]] = []

    def __init_subclass__(cls):
        cls._states.append(cls)


class _share_global_state:
    def __getstate__(self):
        return (
            *filter(
                lambda x: x[1],
                (
                    (cls, dict(filter(_is_state, vars(cls).items())))
                    for cls in GlobalState._states
                ),
            ),
        )

    def __setstate__(self, states: tuple[tuple[type[GlobalState], dict[str]], ...]):
        self._states = states

    def __call__(self):
        for cls, vars in self._states:
            for k, v in vars.items():
                setattr(cls, k, v)
        del self._states


status.initializer.add_unique(_share_global_state)


class _ClassPropertyMeta(type):
    __cached_states__: dict

    def __getattribute__(cls, __name: str):
        if not _is_private(__name):
            if __name.startswith(_RAW):
                return vars(cls).get(__name.removeprefix(_RAW))
            value = vars(cls).get(__name)
            parser = vars(cls).get(f"{_GET}{__name}")
            if isinstance(parser, classmethod):
                if __name not in cls.__cached_states__:
                    cls.__cached_states__[__name] = parser.__func__(cls, value)
                return cls.__cached_states__[__name]
        return super().__getattribute__(__name)

    def __setattr__(cls, __name: str, __value: Any):
        __new = NotImplemented
        if not _is_private(__name):
            cls.__cached_states__.pop(__name, None)
            parser = vars(cls).get(f"{_SET}{__name}")
            if isinstance(parser, classmethod):
                __new = parser.__func__(cls, __value)
        if __new is not NotImplemented:
            __value = __new
        super().__setattr__(__name, __value)


class GlobalSetting(GlobalState, Static, metaclass=_ClassPropertyMeta):
    def __init_subclass__(cls):
        cls.__cached_states__ = {}
        super().__init_subclass__()
        for k, v in vars(cls).items():
            if _is_state((k, v)):
                setattr(cls, k, v)

    @classmethod
    def __mod_name__(cls):
        return ".".join(f"{cls.__module__}.{cls.__name__}".split(".")[3:])

    @classmethod
    def parse(cls, opts: list[str]):
        from . import parse

        proxy = dict_proxy(cls)
        for opt in opts:
            proxy.update(
                dict(filter(_is_state, dict_proxy(parse.mapping(opt)).items()))
            )

    @classmethod
    def autocomplete(cls, opts: list[str]):
        import json

        for k, v in vars(cls).items():
            if not _is_state((k, v)):
                continue
            try:
                parsed = json.dumps(v)
            except Exception:
                parsed = "null"
            opt = f'"{k}: {parsed}"'
            if opt.startswith(opts[-1]):
                yield opt

    @classmethod
    def _help_doc(cls):
        import inspect

        if doc := inspect.getdoc(cls):
            return indent(f"[yellow]{doc}[/yellow]", _INDENT)
        return ""

    @classmethod
    def help(cls):
        from src.typetools import get_partial_type_hints, type_name
        from rich.markup import escape

        from ..docstring import class_attribute_docstring
        from . import parse

        try:
            annotations = get_partial_type_hints(cls, include_extras=True)
        except Exception:
            annotations = cls.__annotations__
        keys = dict(filter(_is_state, vars(cls).items()))
        infos = [f"usage: {cls.__mod_name__()} OPTIONS [OPTIONS ...]", ""]
        if doc := cls._help_doc():
            infos.extend([doc, ""])
        docs = class_attribute_docstring(cls)
        infos.append(f"options: {parse.EMBED}")
        for k, v in keys.items():
            info = k
            if k in annotations:
                info += f": [green]{escape(type_name(annotations[k]))}[/green]"
            value = str(v)
            truncate = False
            if "\n" in value:
                value = value.split("\n", 1)[0]
                truncate = True
            if len(value) > _MAX_WIDTH:
                value = value[:_MAX_WIDTH]
                truncate = True
            info += f' = {value}{"..." if truncate else ""}'
            infos.append(indent(info, _INDENT))
            if docs.get(k):
                infos.append(indent(f"[yellow]{docs[k]}[/yellow]", _INDENT * 2))
        infos.append("")
        return "\n".join(infos)
