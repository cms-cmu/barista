from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ...config.setting.ml import DataLoader as cfg
from ...monitor.progress import MessageType, Progress
from .sliceable import SliceableDataset, SliceLoaderLite


def subset(dataset: Dataset, indices: torch.Tensor):
    if isinstance(dataset, SliceableDataset):
        return dataset.subset(indices)
    else:
        return Subset(dataset, indices)


def simple_loader(
    dataset: Dataset, report_progress: MessageType = None, **kwargs
) -> DataLoader | SliceLoaderLite:
    if isinstance(dataset, SliceableDataset) and cfg.optimize_sliceable_dataset:
        loader = SliceLoaderWithProgress(dataset, **kwargs)
    else:
        loader = DataLoaderWithProgress(
            dataset,
            **(
                dict(
                    num_workers=cfg.num_workers,
                    persistent_workers=cfg.persistent_workers,
                    pin_memory=cfg.pin_memory,
                )
                | kwargs
            ),
        )
        if loader.num_workers != 0:
            from ...process import status

            loader.multiprocessing_context = status.context
    loader._progress_msg = report_progress
    return loader


def skim_loader(dataset: Dataset, report_progress: MessageType = None, **kwargs):
    if "batch_size" not in kwargs:
        kwargs["batch_size"] = cfg.batch_skim
    return simple_loader(dataset, report_progress=report_progress, **kwargs)


class WithProgress:
    _progress_msg: MessageType

    def _progress_iter(self):
        progress = Progress.new(len(self), self._progress_msg)
        for i, data in enumerate(super().__iter__()):
            yield data
            progress.update(i + 1)

    def __iter__(self):
        if hasattr(self, "_progress_msg") and self._progress_msg is not None:
            return self._progress_iter()
        else:
            return super().__iter__()


class DataLoaderWithProgress(WithProgress, DataLoader): ...


class SliceLoaderWithProgress(WithProgress, SliceLoaderLite): ...
