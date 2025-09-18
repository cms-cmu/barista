from ._class import instance
from ._dict import (
    _deserialize_file,
    escape,
    grouped_mappings,
    mapping,
    split_nonempty,
)
from ._number import intervals

__all__ = [
    "escape",
    "mapping",
    "grouped_mappings",
    "split_nonempty",
    "intervals",
    "instance",
    "clear_cache",
]

EMBED = "[purple]\[embed][/purple]"


def clear_cache():
    _deserialize_file.cache_clear()
