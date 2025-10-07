import logging
from collections import defaultdict
from copy import deepcopy
from functools import cache
from io import StringIO
from pathlib import Path
from typing import Literal, overload

import fsspec

from ...typetools import dict_proxy

_SCHEMA = ":##"
_KEY = "@@"
_CACHED = ("file", "py")

SupportedSchema = Literal["yaml", "json", "csv", "file", "py"]


class DeserializationError(Exception):
    __module__ = Exception.__module__

    def __init__(self, msg):
        self.msg = msg


def _mapping_schema(arg: str):
    arg = arg.split(_SCHEMA, 1)
    if len(arg) == 1:
        return None, arg[0]
    else:
        return arg[0], arg[1]


def _mapping_nested_keys(arg: str):
    arg = arg.rsplit(_KEY, 1)
    if len(arg) == 1:
        return arg[0], None
    else:
        return arg[0], arg[1].split(".")


def _deserialize(data: str, protocol: str):
    match protocol:
        case "yaml":
            import yaml

            return yaml.safe_load(data)
        case "json":
            import json

            return json.loads(data)
        case "py":
            import importlib

            mods = data.split(".")
            try:
                mod = importlib.import_module(".".join(mods[:-1]))
                return getattr(mod, mods[-1])
            except Exception:
                raise DeserializationError(f'Failed to import "{data}"')
        case "csv":
            import pandas as pd

            return pd.read_csv(StringIO(data)).to_dict(orient="list")
        case _:
            raise DeserializationError(f'Unsupported protocol "{protocol}"')


@cache
def _deserialize_file(path: str, formatter: str):
    suffix = Path(path).suffix
    match suffix:
        case ".yml":
            protocol = "yaml"
        case ".yaml" | ".json" | ".csv":
            protocol = suffix[1:]
        case _:
            raise DeserializationError(f'Unsupported file "{path}"')
    try:
        with fsspec.open(path, "rt") as f:
            data = f.read()
        if formatter is not None:
            try:
                data = data.format(**mapping(formatter))
            except Exception as e:
                logging.error(exc_info=e)
                raise
        return _deserialize(data, protocol)
    except Exception:
        raise DeserializationError(f'Failed to read file "{path}"')


def _fetch_key(mapping, key):
    proxy = dict_proxy(mapping)
    try:
        return proxy[key]
    except Exception:
        ...
    try:
        return proxy[int(key)]
    except Exception:
        ...
    raise KeyError()


def escape(obj) -> str:
    if not isinstance(obj, str):
        import json

        obj = f"json{_SCHEMA}{json.dumps(obj)}"
    return obj


def mapping(
    arg: str,
    default: SupportedSchema = "yaml",
    formatter: str = None,
):
    """
    - `{data}`: parse as yaml
    - `yaml:##{data}`: parse as yaml
    - `json:##{data}`: parse as json
    - `csv:##{data}`: parse as csv
    - `file:##{path}`: read from file, support .yaml(.yml), .json .csv
    - `py:##{module.class}`: parse as python import

    `file`, `py` support an optional suffix `@@{key}.{key}...` to select a nested dict
    """
    if arg is None:
        return None
    if arg == "":
        return {}

    def warn(msg: str):
        logging.warning(f'{msg} when parsing "{arg}"')

    protocol, data = _mapping_schema(arg)
    if protocol is None:
        protocol = default
    keys = None
    if protocol in _CACHED:
        data, keys = _mapping_nested_keys(data)
    try:
        if protocol == "file":
            result = _deserialize_file(data, formatter)
        else:
            result = _deserialize(data, protocol)
    except DeserializationError as e:
        warn(e.msg)
        return
    if keys is not None:
        for i, k in enumerate(keys):
            try:
                result = _fetch_key(result, k)
            except KeyError:
                warn(f'Failed to select key "{".".join(keys[:i+1])}"')
                return
    if protocol in _CACHED:
        result = deepcopy(result)
    return result


def split_nonempty(text: str, sep: str):
    if text == "":
        return []
    return text.split(sep)


@overload
def grouped_mappings(
    opts: list[list[str]], sep: str
) -> dict[frozenset[str], list[str]]: ...
@overload
def grouped_mappings(
    opts: list[list[str]], sep: None = None
) -> dict[str, list[str]]: ...
def grouped_mappings(opts: list[list[str]], sep: str = None):
    result = defaultdict(list)
    for opt in opts:
        if len(opt) < 2:
            continue
        else:
            arg = opt[0]
            if sep is not None:
                arg = frozenset(split_nonempty(arg, sep))
            result[arg].extend(opt[1:])
    return result
