import awkward as ak

from .wrapper import partition_mapping


def _to_backend_meta(array, *_, **__):
    return array


# operations

array = partition_mapping(typehint=ak.Array)(ak.Array)
to_backend = partition_mapping(typehint=ak.to_backend, meta=_to_backend_meta)(
    ak.to_backend
)
