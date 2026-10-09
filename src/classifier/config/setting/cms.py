from src.classifier.task import GlobalSetting


class CollisionData(GlobalSetting):
    "CMS collision data metadata"

    eras: dict[str, list[str]] = {
        "2023_BPix": ["D1", "D2"],
    }
    "eras for MC datasets"
    years: list[str] = ["2022", "2023", "2024", "2025", "2026"]
    "years for data"


class MC_TTbar(GlobalSetting):
    "Metadata for MC sample: TTbar"

    datasets: list[str] = ["TTToSemiLeptonic", "TTToHadronic", "TTTo2L2Nu"]
    "name of TTbar datasets"


class MC_HH_ggF(GlobalSetting):
    "Metadata for MC sample: ggF HH"

    kl: list[float] = [0.0, 1.0, 2.45, 5.0]
