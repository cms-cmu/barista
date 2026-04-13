from __future__ import annotations

import logging
from datetime import datetime
from itertools import chain
from typing import TYPE_CHECKING

from src.classifier.config.main._utils import _load_dataset, progress_advance
from src.classifier.task import (
    ArgParser,
    Dataset,
    EntryPoint,
    Main,
    TaskOptions,
    converter,
    parse,
)

if TYPE_CHECKING:
    import torch


class Main(Main):
    argparser = ArgParser(
        prog="compare_datasets",
        description="Save datasets to disk. Use [blue]-dataset[/blue] [green]cache.Torch[/green] to load.",
        workflow=[
            ("main", "[blue]\[loader, ...]=dataset.train()[/blue] initialize datasets"),
            ("sub", "[blue]loader()[/blue] load datasets"),
            ("main", "compare datasets"),
        ],
    )
    argparser.add_argument(
        "--max-loaders",
        type=converter.int_pos,
        default=1,
        help="the maximum number of datasets to load in parallel",
    )
    argparser.add_argument(
        "--target-groups",
        nargs="+",
        action="append",
        help="the groups of datasets to compare",
    )
    argparser.add_argument(
        "--atol",
        type=converter.float_pos,
        default=1e-8,
        help="absolute tolerance for comparing datasets.",
    )
    argparser.add_argument(
        "--rtol",
        type=converter.float_pos,
        default=1e-10,
        help="relative tolerance for comparing datasets.",
    )

    def run(self, parser: EntryPoint):
        from concurrent.futures import ProcessPoolExecutor

        import torch

        from src.classifier.monitor.progress import Progress
        from src.classifier.process import pool, status

        target_groups = [parse.intervals(g) for g in self.opts.target_groups or ()]

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
                    preserve_order=True,
                )
            ]
        logging.info(f"Loaded {len(loaders)} datasets in {datetime.now() - timer}")
        if not all(isinstance(v, torch.Tensor) for d in datasets for v in d.values()):
            raise ValueError("Only torch.Tensor can be compared.")
        logging.info(f"Comparing {len(datasets)} datasets")
        groups: list[list[int]] = []
        for i in range(len(datasets)):
            dataset = datasets[i]
            matched = False
            for group in groups:
                if _compare(
                    dataset,
                    datasets[group[0]],
                    atol=self.opts.atol,
                    rtol=self.opts.rtol,
                ):
                    group.append(i)
                    matched = True
                    break
            if not matched:
                groups.append([i])
        logging.info(f"Found the following groups {groups}")
        if target_groups:
            if _to_frozenset(groups) != _to_frozenset(target_groups):
                raise ValueError(f"Expected {target_groups}, but got {groups}")
            else:
                logging.info("Groups matched")
        return {
            "groups": groups,
        }


def _compare(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], **tolerance):
    import torch

    if not set(a.keys()) == set(b.keys()):
        return False
    for k in a.keys():
        if not torch.allclose(a[k], b[k], **tolerance):
            return False
    return True


def _to_frozenset(group: list[list[int]]):
    return frozenset(frozenset(g) for g in group)
