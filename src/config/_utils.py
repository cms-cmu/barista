from __future__ import annotations

import re
from os import get_terminal_size
from textwrap import indent
from typing import Iterable


def block_divider(size: int = 50):
    dash = "-" * max(min(size, get_terminal_size().columns) - 3, 1)
    return f">>>{dash}", f"{dash}<<<"


class _block_predicate:
    start = re.compile(r">>>-+\n")
    end = re.compile(r"-+<<<\n")

    def __init__(self):
        self._outside = True

    def __call__(self, text: str):
        if self.start.fullmatch(text):
            self._outside = False
            return False
        elif self.end.fullmatch(text):
            self._outside = True
            return False
        return self._outside


def block_indent(text: str, prefix: str):
    return indent(text, prefix, _block_predicate())


def format_repr(value, maxlines: int = None) -> str:
    if isinstance(value, dict):
        lines = []
        for k, v in value.items():
            line = format_repr(v)
            if isinstance(v, (dict, list)) or line.count("\n") > 0:
                line = f"{k}:\n{block_indent(line, '  ')}"
            else:
                line = f"{k}: {line}"
            lines.append(line)
        text = "\n".join(lines)
    elif isinstance(value, list):
        text = "\n".join("- " + format_repr(v).replace("\n", "\n  ") for v in value)
    elif isinstance(value, (str, int, float, bool, type(None))):
        text = str(value)
    else:
        text = repr(value)
    if maxlines is not None:
        lines = text.split("\n")
        if len(lines) > maxlines:
            lines[maxlines - 1] = f"+ {len(lines) - maxlines + 1} more lines"
        text = "\n".join(lines[:maxlines])
    return text


class check_reserved:
    def __init__(self, name: str, reserved: Iterable[str]):
        self.name = name
        self.reserved = set(reserved)

    def __call__(self, custom: Iterable[str]):
        if reserved := self.reserved.intersection(custom):
            raise ValueError(
                f"reserved {self.name} cannot be overridden: {', '.join(reserved)}"
            )


class SimpleTree:
    @classmethod
    def init(cls, tree: dict, keys: tuple[str, ...]):
        for key in keys:
            if key not in tree:
                tree[key] = {}
            tree = tree[key]
        return tree

    @classmethod
    def set(cls, tree: dict, keys: tuple[str, ...], value):
        cls.init(tree, keys[:-1])[keys[-1]] = value

    @classmethod
    def get(cls, tree: dict, keys: tuple[str, ...], default=None):
        for key in keys:
            if key not in tree:
                return default
            tree = tree[key]
        return tree
