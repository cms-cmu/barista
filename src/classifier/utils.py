from __future__ import annotations

import importlib
import io
import logging
from fractions import Fraction
from typing import TypeVar

import yaml

_ItemT = TypeVar("_ItemT")


class NOOP:
    def __bool__(self):
        return True

    def __len__(self):
        return 0

    def __getattr__(self, _):
        return self

    def __call__(self, *_, **__):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __repr__(self):
        return "N/A"

    def __format__(self, _):
        return repr(self)


class _NoopMeta(NOOP, type): ...


class noop(metaclass=_NoopMeta): ...


def import_(modname: str, clsname: str, raise_error: bool = False):
    _mod, _cls = None, None
    try:
        _mod = importlib.import_module(modname)
    except ModuleNotFoundError as e:
        if raise_error:
            raise e
    except Exception as e:
        if raise_error:
            raise e
        else:
            logging.error(e, exc_info=e)
    if _mod is not None and clsname != "*":
        try:
            _cls = getattr(_mod, clsname)
        except AttributeError as e:
            if raise_error:
                raise e
        except Exception as e:
            if raise_error:
                raise e
            else:
                logging.error(e, exc_info=e)
    return _mod, _cls


def append_unique_instance(collection: list[_ItemT], item: _ItemT | type[_ItemT]):
    if isinstance(item, type):
        for i in collection:
            if isinstance(i, item):
                return collection
        item = item()
    elif item in collection:
        return collection
    collection.append(item)
    return collection


def keep_fraction(fraction: Fraction, indices):
    return (indices % fraction.denominator) < fraction.numerator


class YamlIndentSequence(yaml.Dumper):
    "https://stackoverflow.com/questions/25108581/python-yaml-dump-bad-indentation"

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def not_none(*values):
    return next((v for v in values if v is not None), None)


class MemoryViewIO:
    def __init__(self, file: io.BytesIO):
        self.file = file

    def write(self, data):
        if isinstance(data, memoryview):
            data = data.tobytes()
        self.file.write(data)

    def flush(self):
        self.file.flush()
