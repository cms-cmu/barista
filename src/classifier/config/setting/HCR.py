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
    from enum import IntEnum
    from src.classifier.task import GlobalSetting

    class InputBranch(GlobalSetting):
        "Name of branches in the input root file"

        feature_CanJet: list[str] = ["pt", "eta", "phi", "mass"]
        feature_NotCanJet: list[str] = feature_CanJet + ["isSelJet"]
        feature_ancillary: list[str] = ["year", "nSelJets", "xW", "xbW"]
        n_CanJet: int = 4
        n_NotCanJet: int = 8
        pad_value: float = -1

        @classmethod
        def get__feature_CanJet(cls, var: list[str]):
            return [f"CanJet_{f}" for f in var]

        @classmethod
        def get__feature_NotCanJet(cls, var: list[str]):
            return [f"NotCanJet_{f}" for f in var]

        @classmethod
        def get__feature_ancillary(cls, var: list[str]):
            return var.copy()

    class Input(GlobalSetting):
        "Name of the keys in the input batch."

        label: str = "label"
        region: str = "region"
        weight: str = "weight"
        ancillary: str = "ancillary"
        CanJet: str = "CanJet"
        NotCanJet: str = "NotCanJet"

    class Output(GlobalSetting):
        "Name of the keys in the output batch."

        class_raw: str = "class_raw"
        class_prob: str = "class_prob"
        quadjet_raw: str = "quadjet_raw"
        quadjet_prob: str = "quadjet_prob"

    class MassRegion(IntEnum):
        SB = 0b10
        ZZSR = 0b0101
        ZHSR = 0b1001
        HHSR = 0b1101
        SR = 0b01

    class NTag(IntEnum):
        fourTag = 0b10
        threeTag = 0b01
