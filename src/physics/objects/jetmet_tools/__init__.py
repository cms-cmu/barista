from src.compat import COFFEA_2025

if COFFEA_2025:
    from .CorrectedJetsFactory_coffea2025 import CorrectedJetsFactory
    from .correctionlib_adapters import (
        CorrectionLibJEC,
        CorrectionLibJER,
        CorrectionLibJERSF,
        CorrectionLibJUNC,
        CorrectionLibJECStack,
    )
else:
    from .CorrectedJetsFactory import FixedCorrectedJetsFactory as CorrectedJetsFactory

__all__ = [
    "CorrectedJetsFactory",
]
