import warnings

warnings.warn(
    "Importing setting.HCR from 'src.classifier.config.setting.HCR' is deprecated. "
    "Please use 'coffea4bees.classifier.config.setting.HCR' instead.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from coffea4bees.classifier.config.setting.HCR import *  # noqa: F401, F403
    from coffea4bees.classifier.config.setting.HCR import __all__ as _coffea4bees_all
    __all__ = _coffea4bees_all
except ImportError:
    pass
