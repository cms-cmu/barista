from classifier.task import GlobalSetting


class CollisionData(GlobalSetting):
    "CMS collision data metadata"

    eras: dict[str, list[str]] = {
        "2022_preEE": ["C", "D"],
        #"2022_EE": ["E", "F", "G"],
        #"2023_preBPix": ["C", "D"],
        #"2023_BPix" : ["E", "F", "G"]
    }
    "eras for MC datasets"
    years: list[str] = ["2022", "2023"]
    "years for data"


class MC_TTbar(GlobalSetting):
    "Metadata for MC sample: TTbar"

    datasets: list[str] = ["TTToSemiLeptonic"]#,"TTToHadronic", "TTo2L2Nu"]
    "name of TTbar datasets"


class MC_HH_ggF(GlobalSetting):
    "Metadata for MC sample: ggF HH"

    kl: list[float] = [0.0, 1.0, 2.45, 5.0]
