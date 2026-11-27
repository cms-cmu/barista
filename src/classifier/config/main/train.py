from __future__ import annotations

import logging
from datetime import datetime
from itertools import chain
from typing import TYPE_CHECKING

from src.classifier.task import ArgParser, EntryPoint, Model, TaskOptions, converter

from ..setting import ResultKey
from ._utils import LoadTrainingSets, SelectDevice, progress_advance

if TYPE_CHECKING:
    from src.classifier.process.device import Device
    from src.classifier.task.model import ModelTrainer
    from torch.utils.data import StackDataset


class _train_model:
    def __init__(self, device: Device, dataset: StackDataset):
        self._device = device
        self._dataset = dataset

    def __call__(self, trainer: ModelTrainer):
        return trainer(device=self._device, dataset=self._dataset)


def _share_dataset_memory(obj) -> None:
    """Move CPU dataset tensors into shared memory once so parallel trainer workers avoid copying."""
    import torch

    if isinstance(obj, torch.Tensor):
        if obj.device.type == "cpu":
            obj.share_memory_()
    elif isinstance(obj, dict):
        for v in obj.values():
            _share_dataset_memory(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _share_dataset_memory(v)
    elif hasattr(obj, "datasets"):
        _share_dataset_memory(obj.datasets)


class Main(SelectDevice, LoadTrainingSets):
    argparser = ArgParser(
        prog="train",
        description="Train multiple models with the same dataset.",
        workflow=[
            *LoadTrainingSets._workflow,
            ("main", "[blue]\[trainer, ...]=model.train()[/blue] initialize models"),
            ("sub", "[blue]trainer(device, dataset)[/blue] train models"),
        ],
    )
    argparser.add_argument(
        "--max-trainers",
        type=converter.int_pos,
        default=1,
        help="the maximum number of models to train in parallel",
    )

    def run(self, parser: EntryPoint):
        from concurrent.futures import ProcessPoolExecutor

        from src.classifier.monitor.progress import Progress
        from src.classifier.process import pool, status

        # load datasets in parallel
        datasets = self.load_training_sets(parser)
        # initialize models
        models: list[Model] = parser.tasks[TaskOptions.model.name]
        timer = datetime.now()
        trainers = [*chain(*(m.train() for m in models))]
        logging.info(f"Initialized {len(trainers)} models in {datetime.now() - timer}")
        # train models in parallel
        timer = datetime.now()
        if self.opts.max_trainers == 1:
            with Progress.new(total=len(trainers), msg=("models", "Training")) as progress:
                results = []
                trainer_func = _train_model(self.device, datasets)
                for trainer in trainers:
                    results.append(trainer_func(trainer))
                    progress_advance(progress)(None)
        else:
            _share_dataset_memory(datasets)
            with (
                ProcessPoolExecutor(
                    max_workers=self.opts.max_trainers,
                    mp_context=status.context,
                    initializer=status.initializer,
                ) as executor,
                Progress.new(total=len(trainers), msg=("models", "Training")) as progress,
            ):
                results = [
                    *pool.map_async(
                        executor,
                        _train_model(self.device, datasets),
                        trainers,
                        callbacks=[lambda _: progress_advance(progress)],
                    )
                ]
        logging.info(f"Trained {len(results)} models in {datetime.now() - timer}")
        return {ResultKey.models: results}
