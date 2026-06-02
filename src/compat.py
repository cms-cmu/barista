"""
Compatibility shim for coffea 0.7.x and coffea 2025.x.

Import COFFEA_2025 to branch on version, or use nano_from_root()
as a drop-in for NanoEventsFactory.from_root() that works on both.
"""
import coffea

COFFEA_2025 = not coffea.__version__.startswith("0.")


def nano_from_root(path, treename="Events", **kwargs):
    """
    Version-agnostic wrapper for NanoEventsFactory.from_root().

    Accepts path as either a string or a {path: treename} dict (coffea 2025 style).
    coffea 0.7: from_root(path_string, ...)
    coffea 2025: from_root({path_string: treename}, ...)
    """
    from coffea.nanoevents import NanoEventsFactory
    if isinstance(path, dict):
        path_str, treename = next(iter(path.items()))
    else:
        path_str = path
    src = {path_str: treename} if COFFEA_2025 else path_str
    return NanoEventsFactory.from_root(src, **kwargs)
