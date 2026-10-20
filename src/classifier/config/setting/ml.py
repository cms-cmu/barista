from __future__ import annotations

from src.classifier.task import GlobalSetting


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
    "disable unnecessary benchmark steps"

    precision: str = ""
    "mixed-precision mode on CUDA: '' (off, exact fp32), 'fp16', or 'bf16'. "
    "Both halve activation memory (fit wider nets / larger batches in the 24 GB MPS "
    "cap) and use tensor cores. fp16 needs a GradScaler and OVERFLOWS for very wide "
    "nets (n_features>~60 diverges to nan); bf16 has the fp32 exponent range so it "
    "does NOT overflow (use it for big nets) at the cost of a little mantissa "
    "precision. NOTE: changes numerics at the ~1e-4 level, so do not mix precisions "
    "in a comparison finer than that."

    @classmethod
    def set__precision(cls, value: str):
        v = (value or "").lower()
        if v in ("", "fp32", "none", "off"):
            return ""
        assert v in ("fp16", "bf16"), f"unknown precision {value!r} (use fp16/bf16)"
        return v


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
