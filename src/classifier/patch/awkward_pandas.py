from __future__ import annotations

from typing import TYPE_CHECKING

from .patch import Patch
from .version import issue_version

if TYPE_CHECKING:
    import awkward_pandas as akpd


@classmethod
def _concat_same_type(cls, to_concat):
    import awkward as ak

    return cls(ak.concatenate([a._data for a in to_concat]))


@Patch.register("awkward_pandas")
def PR49(module: akpd):
    """
    Related:
    - [issue#48](https://github.com/intake/awkward-pandas/issues/48)
    - [PR#49](https://github.com/intake/awkward-pandas/pull/49)
    """
    if issue_version(module, "2024.3.0", "2023.8.0"):
        module.AwkwardExtensionArray._concat_same_type = _concat_same_type
