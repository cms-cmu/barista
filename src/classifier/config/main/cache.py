from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import TYPE_CHECKING

import fsspec
from src.classifier.task import ArgParser, EntryPoint, converter

from ...utils import MemoryViewIO
from ..setting import IO as IOSetting
from ..setting import ResultKey
from ._utils import LoadTrainingSets, progress_advance

if TYPE_CHECKING:
    import numpy.typing as npt
    from storage.eos import EOS
    from torch.utils.data import StackDataset


class Main(LoadTrainingSets):
    argparser = ArgParser(
        prog="cache",
        description="Save datasets to disk. Use [blue]-dataset[/blue] [green]cache.Torch[/green] to load.",
        workflow=[
            *LoadTrainingSets._workflow,
            ("sub", "write chunks to disk"),
        ],
    )
    argparser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle the dataset before saving",
    )
    argparser.add_argument(
        "--nchunks",
        type=converter.int_pos,
        help="number of chunks",
    )
    argparser.add_argument(
        "--chunksize",
        type=converter.int_pos,
        help="size of each chunk, will be ignored if [yellow]--nchunks[/yellow] is given",
    )
    argparser.add_argument(
        "--compression",
        choices=fsspec.available_compressions(),
        help="compression algorithm to use",
    )
    argparser.add_argument(
        "--max-writers",
        type=converter.int_pos,
        default=1,
        help="the maximum number of files to write in parallel",
    )
    argparser.add_argument(
        "--states", action="extend", nargs="+", help="states to cache", default=[]
    )

    def run(self, parser: EntryPoint):
        from concurrent.futures import ProcessPoolExecutor

        import numpy as np
        from src.classifier.monitor.progress import Progress
        from src.classifier.process import pool, status

        # cache states
        states = dict.fromkeys(self.opts.states, None)
        if states:
            logging.info(f"The following states will be cached {sorted(states)}")
            for state in states:
                mod, var = state.rsplit(".", 1)
                mod = EntryPoint._fetch_module(mod, "state", True)[1]
                states[state] = getattr(mod, var)
        # cache datasets
        datasets = self.load_training_sets(parser)
        size = len(datasets)
        chunks = np.arange(size)
        if self.opts.shuffle:
            np.random.shuffle(chunks)
        if self.opts.nchunks is not None:
            chunksize = math.ceil(size / self.opts.nchunks)
        elif self.opts.chunksize is not None:
            chunksize = self.opts.chunksize
        else:
            chunksize = size
        chunks = [chunks[i : i + chunksize] for i in range(0, size, chunksize)]

        logging.info("Caching datasets...")
        timer = datetime.now()
        with (
            ProcessPoolExecutor(
                max_workers=self.opts.max_writers,
                mp_context=status.context,
                initializer=status.initializer,
            ) as executor,
            Progress.new(total=size, msg=("entries", "Caching")) as progress,
        ):
            tasks = pool.map_async(
                executor,
                _save_cache(datasets, IOSetting.output, self.opts.compression),
                range(len(chunks)),
                chunks,
                callbacks=[lambda _, idx: progress_advance(progress, len(idx))],
            )
            (*tasks,)  # wait for completion
        logging.info(
            f"Wrote {size} entries to {len(chunks)} files in {datetime.now() - timer}"
        )
        return {
            ResultKey.cache: {
                "size": size,
                "chunksize": chunksize,
                "shuffle": self.opts.shuffle,
                "compression": self.opts.compression,
                "states": states,
            }
        }


class _save_cache:
    def __init__(self, dataset: StackDataset, path: EOS, compression: str = None):
        self.dataset = dataset
        self.path = path
        self.compression = compression

    def __call__(self, chunk: int, indices: npt.ArrayLike):
        import torch
        from src.classifier.nn.dataset import skim_loader, subset
        from src.classifier.nn.dataset.sliceable import NamedTensorDataset

        dataset = subset(self.dataset, indices)
        if isinstance(dataset, NamedTensorDataset):
            data = dataset.datasets
        else:
            chunks = [*skim_loader(dataset)]
            data = {k: torch.cat([c[k] for c in chunks]) for k in self.dataset.datasets}
        with fsspec.open(
            self.path / f"chunk{chunk}.pt", "wb", compression=self.compression
        ) as f:
            torch.save(data, MemoryViewIO(f))
