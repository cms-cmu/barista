import warnings

warnings.warn(
    "Importing HCR from 'src.classifier.nn.blocks.HCR' is deprecated and will be removed in a future release. "
    "Please import from 'coffea4bees.classifier.nn.blocks.HCR' instead.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from coffea4bees.classifier.nn.blocks.HCR import *  # noqa: F401, F403
    from coffea4bees.classifier.nn.blocks.HCR import __all__ as _coffea4bees_all
    __all__ = _coffea4bees_all
except ImportError:
    pass
