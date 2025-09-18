from __future__ import annotations

import logging
from datetime import datetime
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING

from classifier.task import ArgParser, Dataset, EntryPoint, Main, TaskOptions, converter
from classifier.task.dataset import TrainingSetLoader

if TYPE_CHECKING:
    from classifier.monitor.progress import ProgressTracker


class progress_advance:
    def __init__(self, progress: ProgressTracker, step: int = 1):
        self._progress = progress
        self._step = step

    def __call__(self, _):
        self._progress.advance(step=self._step)


def _load_dataset(loader: TrainingSetLoader):
    return loader()


class SelectDevice(Main):
    argparser = ArgParser()
    argparser.add_argument(
        "--device",
        nargs="+",
        choices=["cpu", "cuda"],
        default=["cuda"],
        help="the device used for training",
    )

    @cached_property
    def device(self):
        from classifier.process.device import Device

        return Device(*self.opts.device)


class LoadTrainingSets(Main):
    _workflow = [
        ("main", "[blue]\[loader, ...]=dataset.train()[/blue] initialize datasets"),
        ("sub", "[blue]loader()[/blue] load datasets"),
    ]
    argparser = ArgParser()
    argparser.add_argument(
        "--max-loaders",
        type=converter.int_pos,
        default=1,
        help="the maximum number of datasets to load in parallel",
    )

    def load_training_sets(self, parser: EntryPoint):
        from concurrent.futures import ProcessPoolExecutor

        import torch
        from classifier.monitor.progress import Progress
        from classifier.nn.dataset.sliceable import NamedTensorDataset
        from classifier.process import pool, status
        from torch.utils.data import ConcatDataset, StackDataset

        # load datasets in parallel
        tasks: list[Dataset] = parser.tasks[TaskOptions.dataset.name]
        loaders = [*chain(*(k.train() for k in tasks))]
        if len(loaders) == 0:
            raise ValueError("No dataset to load")
        logging.info(f"Loading {len(loaders)} datasets")
        timer = datetime.now()
        with (
            ProcessPoolExecutor(
                max_workers=self.opts.max_loaders,
                mp_context=status.context,
                initializer=status.initializer,
            ) as executor,
            Progress.new(total=len(loaders), msg=("datasets", "Loading")) as progress,
        ):
            datasets = [
                *pool.map_async(
                    executor,
                    _load_dataset,
                    loaders,
                    callbacks=[lambda _: progress_advance(progress)],
                )
            ]
        logging.info(f"Loaded {len(loaders)} datasets in {datetime.now() - timer}")
        # concatenate datasets
        keys = [set(d.keys()) for d in datasets]
        kept = set.intersection(*keys)
        ignored = set.union(*keys) - kept
        kept = sorted(kept)
        logging.info(f"The following keys will be kept: {kept}")
        if ignored:
            logging.warning(f"The following keys will be ignored: {sorted(ignored)}")
        if all(isinstance(d[k], torch.Tensor) for k in kept for d in datasets):
            datasets = NamedTensorDataset.concat(
                *(NamedTensorDataset(**{k: d[k] for k in kept}) for d in datasets)
            )
        else:
            datasets = StackDataset(
                **{k: ConcatDataset(d[k] for d in datasets) for k in kept}
            )
        logging.info(f"Loaded {len(datasets)} data entries")
        return datasets
