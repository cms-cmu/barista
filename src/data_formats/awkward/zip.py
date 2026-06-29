import re
from enum import IntEnum, auto
from itertools import groupby

import awkward as ak


def _unzip(array: ak.Array) -> dict[str, ak.Array]:
    return dict(zip(ak.fields(array), ak.unzip(array)))


class _Shape(IntEnum):
    JAGGED = auto()
    REGULAR = auto()


class NanoAOD:
    _count_pattern = re.compile(r"^n[A-Za-z]\w*$")

    def __init__(
        self,
        *selected: str,
        regular: bool = True,
        jagged: bool = True,
        cache: bool = True,
    ):
        self._selected = frozenset(selected)
        self._zip = {_Shape.JAGGED: jagged, _Shape.REGULAR: regular}
        self._cache: dict[frozenset[str], tuple[set[str], dict[str, set[str]]]] = (
            {} if cache else None
        )

    def _parse_fields(self, data: ak.Array):
        to_keep: set[str] = set(ak.fields(data))
        to_zip: dict[str, frozenset[str]] = {}
        if self._cache is not None:
            key = frozenset(to_keep)
            if key in self._cache:
                return self._cache[key]
        if any(self._zip.values()):
            fields = {
                k: list(v)
                for k, v in groupby(
                    sorted(
                        filter(
                            lambda x: len(x) > 1,
                            map(lambda x: x.split("_", 1), to_keep),
                        ),
                        key=lambda x: x[0],
                    ),
                    key=lambda x: x[0],
                )
            }
            keys = set(fields)
            jagged = set(
                filter(
                    lambda x: (x in keys) or (x in to_keep),
                    map(
                        lambda x: x[1:],
                        filter(
                            lambda x: self._count_pattern.match(x) is not None, to_keep
                        ),
                    ),
                )
            )
            regular = keys - jagged
            prefixes = {_Shape.JAGGED: jagged, _Shape.REGULAR: regular}
            for k in _Shape:
                if self._zip[k]:
                    if self._selected:
                        prefixes[k] &= self._selected
                    for prefix in prefixes[k]:
                        if prefix in fields:
                            to_zip[prefix] = frozenset(map("_".join, fields[prefix]))
                            to_keep -= to_zip[prefix]
                        if k == _Shape.JAGGED:
                            to_keep.remove(f"n{prefix}")
        if self._cache is not None:
            self._cache[key] = to_keep, to_zip
        return to_keep, to_zip

    def __call__(self, data: ak.Array):
        keep, to_zip = self._parse_fields(data)
        zipped = _unzip(data[keep]) if keep else {}
        for k, vs in to_zip.items():
            start = len(k) + 1
            zipped[k] = ak.zip({v[start:]: data[v] for v in vs})
        return ak.Array(zipped)
