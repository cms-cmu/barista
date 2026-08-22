import warnings

warnings.warn(
    "Importing HCR from 'src.classifier.nn.blocks.HCR' is deprecated and will be removed in a future release. "
    "Please import from 'coffea4bees.classifier.nn.blocks.HCR' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from coffea4bees.classifier.nn.blocks.HCR import *  # noqa: F401, F403
