from __future__ import annotations

import operator as op
import sys
from collections import defaultdict
from functools import partial, reduce
from typing import TYPE_CHECKING, Callable, Literal, Protocol

if TYPE_CHECKING:
    import awkward
    import numpy
    import pandas

_UNKNOWN = "Unknown backend {library}."

Backends = Literal["ak", "np", "pd"]


class _NestedRecord(defaultdict):
    def __init__(self, *args, **kwargs):
        super().__init__(_NestedRecord, *args, **kwargs)

    def to_dict(self) -> dict:
        return {
            k: v.to_dict() if isinstance(v, _NestedRecord) else v
            for k, v in self.items()
        }

    def to_array(self) -> awkward.Array:
        import awkward as ak

        return ak.zip(
            {
                k: v.to_array() if isinstance(v, _NestedRecord) else v
                for k, v in self.items()
            },
            depth_limit=1,
        )

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            if len(key) > 1:
                self[key[0]][key[1:]] = value
                return
            else:
                key = key[0]
        super().__setitem__(key, value)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) > 1:
                return self[key[0]][key[1:]]
            else:
                key = key[0]
        return super().__getitem__(key)


class _Backends:
    backends = dict(ak="awkward", np="numpy", pd="pandas")
    ak: awkward
    np: numpy
    pd: pandas

    def __getattr__(self, name):
        try:
            return sys.modules[self.backends[name]]
        except KeyError:
            return self

    def check(self, __obj, __type):
        if __type is self:
            return False
        return isinstance(__obj, __type)


def record_backend(data, sequence=False):
    mod = _Backends()
    if sequence:
        backends = {*map(record_backend, data)}
        if len(backends) == 1:
            return backends.pop()
        else:
            raise ValueError(f"Inconsistent backends {backends}.")
    if isinstance(data, dict):
        backends = {*map(record_backend, data.values())}
        if len(backends) == 0:
            return "np"
        if len(backends) == 1:
            backends = backends.pop()
            if backends == "np.array":
                return "np"
            else:
                return f"dict.{backends}"
        return "dict"
    if mod.check(data, mod.ak.Array):
        return "ak"
    if mod.check(data, mod.np.ndarray):
        return "np.array"
    if mod.check(data, mod.pd.DataFrame):
        return "pd"
    return f"<{data.__module__}.{data.__class__.__name__}>"


def concat_record(data: list, library: Backends = ...):
    if library is ...:
        library = record_backend(data, sequence=True)
    if len(data) == 0:
        return None
    data = [data[0], *filter(partial(len_record, library=library), data[1:])]
    if len(data) == 1:
        return data[0]
    if library == "ak":
        import awkward as ak

        return ak.concatenate(data)
    elif library == "pd":
        import pandas as pd

        return pd.concat(data, ignore_index=True, sort=False, copy=False, axis=0)
    elif library == "np":
        import numpy as np

        result = {}
        for k in data[0].keys():
            result[k] = np.concatenate([d[k] for d in data])
        return result
    else:
        raise TypeError(_UNKNOWN.format(library=library))


def merge_record(data: list, library: Backends = ...):
    if library is ...:
        library = record_backend(data, sequence=True)
    if len(data) == 0:
        return None
    if len(data) == 1:
        return data[0]
    if library == "ak":
        import awkward as ak

        return ak.zip(
            reduce(op.or_, (dict(zip(ak.fields(arr), ak.unzip(arr))) for arr in data)),
            depth_limit=1,
        )
    elif library == "pd":
        import pandas as pd

        df = pd.concat(data, ignore_index=False, sort=False, copy=False, axis=1)
        return df.loc[:, ~df.columns.duplicated(keep="last")]
    elif library == "np" or library.startswith("dict"):
        return reduce(op.or_, data)
    else:
        raise TypeError(_UNKNOWN.format(library=library))


def slice_record(data, start: int, stop: int, library: Backends = ...):
    if library is ...:
        library = record_backend(data)
    if library in ("ak", "pd"):
        return data[start:stop]
    elif library == "np":
        return {k: v[start:stop] for k, v in data.items()}
    elif library.startswith("dict"):
        if library == "dict":
            return {k: slice_record(v, start, stop) for k, v in data.items()}
        else:
            content = library.removeprefix("dict.")
            return {
                k: slice_record(v, start, stop, library=content)
                for k, v in data.items()
            }
    else:
        raise TypeError(_UNKNOWN.format(library=library))


def len_record(data, library: Backends = ...):
    if library is ...:
        library = record_backend(data)
    if library in ("ak", "pd"):
        return len(data)
    elif library == "np" or library.startswith("dict"):
        if len(data) == 0:
            return 0
        return len(next(iter(data.values())))
    else:
        raise TypeError(_UNKNOWN.format(library=library))


def rename_record(
    data,
    mapping: Callable[[str], str | tuple[str, ...]],
    library: Backends = ...,
):
    if library is ...:
        library = record_backend(data)
    if library == "ak":
        import awkward as ak

        nested = _NestedRecord()
        for k, v in zip(ak.fields(data), ak.unzip(data)):
            nested[mapping(k)] = v
        return nested.to_array()
    elif library == "pd":
        import pandas as pd

        return pd.DataFrame({mapping(k): data.loc[:, k] for k in data.columns})
    elif library == "np" or library.startswith("dict"):
        nested = _NestedRecord()
        for k, v in data.items():
            nested[mapping(k)] = v
        return nested.to_dict()
    else:
        raise TypeError(_UNKNOWN.format(library=library))


def sizeof_record(data, library: Backends = ...):
    if library is ...:
        library = record_backend(data)
    if library in ("ak", "np.array"):
        return data.nbytes
    elif library == "pd":
        return data.memory_usage(index=True, deep=True).sum()
    elif library == "np" or library.startswith("dict"):
        if library == "dict":
            lib = ...
        elif library == "np":
            lib = "np.array"
        else:
            lib = library.removeprefix("dict.")
        return sum(sizeof_record(v, library=lib) for v in data.values())
    else:
        raise TypeError(_UNKNOWN.format(library=library))


def keyof_record(data, library: Backends = ...) -> list[str]:
    mod = _Backends()
    if library is ...:
        library = record_backend(data)
    if library == "ak":
        return list(mod.ak.fields(data))
    elif library == "pd":
        return list(data.columns)
    elif library == "np" or library.startswith("dict"):
        return list(data.keys())
    else:
        raise TypeError(_UNKNOWN.format(library=library))


class NameMapping(Protocol):
    def __call__(self, **keys: str) -> str | tuple[str, ...]: ...


def apply_naming(naming: str | NameMapping, keys: dict[str, str]):
    if isinstance(naming, str):
        return naming.format(**keys)
    elif isinstance(naming, Callable):
        return naming(**keys)
    else:
        raise TypeError(f'Unknown naming "{naming}"')


def materialize_record(data, library: Backends = ...):
    if library is ...:
        library = record_backend(data)
    if library == "ak":
        import awkward as ak

        if hasattr(ak, "materialized"):
            ak.materialized(data)
    elif library in ("np", "pd") or library.startswith("dict"):
        pass
    else:
        raise TypeError(_UNKNOWN.format(library=library))
