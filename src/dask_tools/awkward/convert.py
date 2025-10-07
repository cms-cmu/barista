import dask_awkward as dak

from ... import awkward as akext


def from_jsonable(*jsonables, npartitions: int = ...) -> dak.Array:
    if npartitions is ...:
        npartitions = len(jsonables)
    return dak.from_awkward(akext.from_jsonable(*jsonables), npartitions=npartitions)
