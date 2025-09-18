from __future__ import annotations

from classifier.task import GlobalSetting


class DataLoader(GlobalSetting):
    "DataLoader configuration"

    optimize_sliceable_dataset: bool = True
    "optimize for SliceableDataset. (best for small datasets that can be fully loaded into memory)"
    batch_skim: int = 2**17
    "batch size for skimming"
    batch_eval: int = 2**15
    "batch size for evaluation"

    pin_memory: bool = True
    "only for torch.DataLoader"
    num_workers: int = 0
    "only for torch.DataLoader"
    persistent_workers: bool = True
    "only for torch.DataLoader"

    @classmethod
    def get__persistent_workers(cls, value: bool) -> bool:
        if cls.num_workers == 0:
            return False
        return value


class KFold(GlobalSetting):
    "KFolding dataset splitter"

    offset: str = "offset"
    "key of the offset in the input batch"
    offset_dtype: str = "uint64"
    "dtype of the offset tensor"


class Training(GlobalSetting):
    "Multistage training"

    disable_benchmark: bool = False
    "disable unrequired benchmark steps"


class SplitterKeys(GlobalSetting):
    "keys in the splitter output"

    training: str = "training"
    validation: str = "validation"


class MultiClass(GlobalSetting):
    "Multiclass classification label manager"

    nontrainable_labels: set[str] = []

    @classmethod
    def set__nontrainable_labels(cls, value: list[str]):
        return set(value)
