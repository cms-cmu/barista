from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import fsspec
import torch
import torch.types as tt
from src.classifier.config.setting import IO as cfg_IO
from src.classifier.config.setting import ml as cfg
from rich.pretty import pretty_repr
from torch import Tensor, nn
from torch.utils.data import Dataset

from ..monitor.progress import MessageType, Progress
from ..monitor.usage import Usage
from ..nn.dataset import simple_loader
from ..nn.schedule import Schedule
from ..process.device import Device
from ..task.special import interface
from ..typetools import WithUUID, filename
from ..utils import MemoryViewIO
from . import BatchType, clear_cache

if TYPE_CHECKING:
    from src.storage.eos import PathLike


def _path_exists(p) -> bool:
    if p is None:
        return False
    try:
        if hasattr(p, "exists"):
            return p.exists() if callable(p.exists) else bool(p.exists)
        elif hasattr(p, "is_file"):
            return p.is_file()
    except Exception:
        pass
    return False


def _path_rm(p) -> None:
    if p is None:
        return
    try:
        if hasattr(p, "unlink"):
            p.unlink(missing_ok=True)
        elif hasattr(p, "rm"):
            p.rm()
    except Exception:
        pass


@dataclass(kw_only=True)
class Stage(ABC):
    name: str

    @abstractmethod
    def run(self, trainer: MultiStageTraining) -> dict[str]: ...

    @property
    def stage(self):
        return self.__class__.__name__.removesuffix("Stage")


@dataclass(kw_only=True)
class BenchmarkStage(Stage):
    model: Model
    validation: dict[str, Dataset]

    @staticmethod
    def _loader_benchmark(dataset: Dataset, msg: MessageType = None):
        return simple_loader(
            dataset,
            report_progress=msg,
            batch_size=cfg.DataLoader.batch_eval,
            shuffle=False,
            drop_last=False,
        )

    def _init_benchmark(self):
        return {
            k: self._loader_benchmark(v, ("batch", "benchmark", k))
            for k, v in self.validation.items()
        }

    def _iter_benchmark(
        self, trainer: MultiStageTraining, loaders: dict[str, Iterable[BatchType]]
    ):
        benchmark = {}
        self.model.nn.eval()
        with torch.no_grad():
            for k, v in loaders.items():
                if len(v) == 0:
                    continue
                trainer.cleanup()
                Usage.checkpoint(self.name, "benchmark", k, "start")
                benchmark[k] = self.model.validate(v)
                Usage.checkpoint(self.name, "benchmark", k, "finish")
        return benchmark

    @property
    def _skip_benchmark(self):
        return cfg.Training.disable_benchmark

    def run(self, trainer: MultiStageTraining):
        return {
            "stage": self.stage,
            "name": self.name,
            "benchmarks": (
                self._iter_benchmark(trainer, self._init_benchmark())
                if not self._skip_benchmark
                else {}
            ),
        }


@dataclass(kw_only=True)
class TrainingStage(BenchmarkStage):
    schedule: Schedule
    training: Dataset
    validation: dict[str, Dataset] = None

    @property
    def _skip_benchmark(self):
        return (not self.schedule.require_benchmark) and (
            super()._skip_benchmark
            or (self.validation is None)
            or (self.model.validate is NotImplemented)
        )

    def run(self, trainer: MultiStageTraining):
        history = {
            "stage": self.stage,
            "name": self.name,
            "parameters": self.model.n_parameters,
        }
        opt = self.schedule.optimizer(self.model.parameters())
        # GradScaler only for fp16 (bf16 has fp32 range, needs no loss scaling; off ->
        # disabled). When disabled, the loop below takes the exact original fp32 path.
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.Training.precision == "fp16")
        lr = self.schedule.lr_scheduler(opt)
        bs = self.schedule.bs_scheduler(
            self.training,
            shuffle=True,
            drop_last=True,
        )
        if not self._skip_benchmark:
            benchmarks = []
            validation = self._init_benchmark()
        else:
            benchmarks = None
        # training
        p_epoch = Progress.new(self.schedule.epoch, ("epoch", self.name))
        logging.info(f"Start {self.name}")
        start = datetime.now()

        # Checkpoint path for intermediate state recovery.
        # Prefer local CLASSIFIER_CHECKPOINT_DIR if set, otherwise cfg_IO.output.
        ckpt_dir = None
        if "CLASSIFIER_CHECKPOINT_DIR" in os.environ and os.environ["CLASSIFIER_CHECKPOINT_DIR"]:
            ckpt_dir = Path(os.environ["CLASSIFIER_CHECKPOINT_DIR"])
            try:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        if ckpt_dir is not None:
            ckpt_path = ckpt_dir / f"checkpoint_{trainer.name}_{self.name}.pt"
        elif not cfg_IO.output.is_null:
            ckpt_path = cfg_IO.output / f"checkpoint_{trainer.name}_{self.name}.pt"
        else:
            ckpt_path = None

        start_epoch = 0
        ckpt_to_load = None
        candidates = []
        if ckpt_path is not None:
            candidates.append(ckpt_path)
        if ckpt_dir is not None:
            candidates.append(ckpt_dir / f"checkpoint_{trainer.name}_{self.name}.pt")
        if not cfg_IO.output.is_null:
            remote_ckpt = cfg_IO.output / f"checkpoint_{trainer.name}_{self.name}.pt"
            if str(remote_ckpt) != str(ckpt_path):
                candidates.append(remote_ckpt)

        for cp in candidates:
            if _path_exists(cp):
                ckpt_to_load = cp
                break

        if ckpt_to_load is not None:
            try:
                logging.info(f"Loading checkpoint from {ckpt_to_load} to resume training...")
                with fsspec.open(ckpt_to_load, "rb") as f:
                    ckpt = torch.load(f, map_location=trainer.device, weights_only=False)
                self.model.nn.load_state_dict(ckpt["model_state_dict"])
                opt.load_state_dict(ckpt["opt_state_dict"])
                if "lr_state_dict" in ckpt:
                    lr.load_state_dict(ckpt["lr_state_dict"])
                if scaler.is_enabled() and "scaler_state_dict" in ckpt:
                    scaler.load_state_dict(ckpt["scaler_state_dict"])
                start_epoch = ckpt.get("epoch", 0)
                if benchmarks is not None and "benchmarks" in ckpt:
                    benchmarks = ckpt["benchmarks"]
                logging.info(f"Successfully resumed {self.name} from epoch {start_epoch}")
            except Exception as e:
                logging.warning(f"Could not load checkpoint from {ckpt_to_load}: {e}. Starting fresh.")
                start_epoch = 0

        if start_epoch >= self.schedule.epoch:
            logging.info(
                f"Stage {self.name} already completed all {self.schedule.epoch} epochs. Skipping."
            )
            if benchmarks is not None:
                history["training"] = benchmarks
            return history

        for epoch in range(start_epoch, self.schedule.epoch):
            epoch = epoch + 1
            trainer.cleanup()
            Usage.checkpoint(self.name, f"epoch{epoch}", "optimize")
            self.model.nn.train()
            if len(bs.dataloader) > 0:
                p_batch = Progress.new(len(bs.dataloader), "batch")
            for i, batch in enumerate(bs.dataloader):
                opt.zero_grad()
                loss = self.model.train(batch)
                if scaler.is_enabled() and isinstance(loss, torch.Tensor):
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                if isinstance(loss, torch.Tensor):
                    loss_val = loss.item()
                    p_batch.update(i + 1, ("batch", "training", f"loss={loss_val:.4g}"))
                else:
                    p_batch.update(i + 1, ("batch", "training", "skim"))
            benchmark = {
                "hyperparameters": {
                    "epoch": epoch,
                    "learning rate": lr.get_last_lr(),
                    "batch size": bs.dataloader.batch_size,
                }
                | self.model.hyperparameters,
            }
            if not self._skip_benchmark:
                benchmark["benchmarks"] = self._iter_benchmark(trainer, validation)
                benchmarks.append(benchmark)
            if epoch != self.schedule.epoch:
                self.schedule.step(bs, lr, benchmark)
                if self.model.step is not NotImplemented:
                    self.model.step()
            p_epoch.update(epoch)
            Usage.checkpoint(self.name, f"epoch{epoch}", "finish")

            # Save intermediate training checkpoint at end of each epoch
            if ckpt_path is not None:
                try:
                    ckpt_data = {
                        "epoch": epoch,
                        "model_state_dict": self.model.nn.state_dict(),
                        "opt_state_dict": opt.state_dict(),
                        "lr_state_dict": lr.state_dict(),
                        "benchmarks": benchmarks,
                    }
                    if scaler.is_enabled():
                        ckpt_data["scaler_state_dict"] = scaler.state_dict()
                    with fsspec.open(ckpt_path, "wb") as f:
                        torch.save(ckpt_data, MemoryViewIO(f))
                except Exception as e:
                    logging.warning(f"Failed to save intermediate checkpoint: {e}")

            if self.schedule.should_stop(benchmark):
                break

        logging.info(
            f"{self.name}: run {epoch} epochs in {datetime.now() - start}"
        )
        if benchmarks is not None:
            history["training"] = benchmarks
        return history


@dataclass(kw_only=True)
class OutputStage(Stage):
    path: PathLike
    relative: bool = True

    def run(self, _):
        return {
            "stage": self.stage,
            "name": self.name,
            "path": str(self.path),
            "relative": self.relative,
        }

    @property
    def absolute_path(self) -> PathLike:
        if self.relative:
            return cfg_IO.output / str(self.path)
        return self.path


class Model(ABC):
    def parameters(self):
        return self.nn.parameters()

    def to(self, device: tt.Device):
        self.nn.to(device)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.nn.parameters())

    @property
    def hyperparameters(self) -> dict[str]:
        return {}

    @property
    @abstractmethod
    def nn(self) -> nn.Module: ...

    @interface
    def train(self, batch: BatchType) -> Tensor: ...

    @interface(optional=True)
    def validate(self, batches: Iterable[BatchType]) -> dict: ...

    @interface
    def evaluate(self, batch: BatchType) -> BatchType: ...

    @interface(optional=True)
    def step(self, epoch: int = None): ...


class MultiStageTraining(WithUUID, ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.metadata = kwargs
        self.name = filename(kwargs)
        self.device: torch.device = None
        self.dataset: Dataset = None

    def cleanup(self):
        clear_cache(self.device)

    @property
    def min_memory(self):
        return 0

    @abstractmethod
    def stages(self) -> Iterable[Stage]: ...

    def find_existing_model(self):
        """Check if this model offset was already completely trained and saved."""
        if os.environ.get("CLASSIFIER_FORCE_RETRAIN"):
            return None
        if cfg_IO.output.is_null:
            return None
        try:
            if hasattr(cfg_IO.output, "ls"):
                files = cfg_IO.output.ls()
                for f in files:
                    fname = getattr(f, "name", str(f)).rsplit("/", 1)[-1]
                    if fname.startswith(f"{self.name}__") and fname.endswith(".pkl"):
                        try:
                            with fsspec.open(f, "rb") as fp:
                                meta = torch.load(fp, map_location="cpu", weights_only=False)
                                if "model" in meta and "uuid" in meta:
                                    return f
                        except Exception:
                            continue
        except Exception as e:
            logging.debug(f"Could not check existing model for {self.name}: {e}")
        return None

    def cleanup_checkpoints(self):
        """Remove intermediate stage checkpoints once the final model is safely saved."""
        dirs_to_clean = []
        if "CLASSIFIER_CHECKPOINT_DIR" in os.environ and os.environ["CLASSIFIER_CHECKPOINT_DIR"]:
            dirs_to_clean.append(Path(os.environ["CLASSIFIER_CHECKPOINT_DIR"]))
        if not cfg_IO.output.is_null:
            dirs_to_clean.append(cfg_IO.output)

        for d in dirs_to_clean:
            for stage_name in ("Training", "Finetuning", "Baseline"):
                cp = d / f"checkpoint_{self.name}_{stage_name}.pt"
                _path_rm(cp)

    def train(self, device: Device, dataset: Dataset):
        existing_model = self.find_existing_model()
        if existing_model is not None:
            logging.info(
                f"Model {self.name} already completed and saved at {existing_model}. Skipping training."
            )
            self.cleanup_checkpoints()
            return {
                "uuid": str(self.uuid),
                "name": self.name,
                "metadata": self.metadata,
                "history": [
                    {
                        "stage": "Output",
                        "name": "Final",
                        "path": str(existing_model),
                        "relative": False,
                    }
                ],
            }

        self.device = device.get(self.min_memory)
        self.dataset = dataset
        result = {
            "uuid": str(self.uuid),
            "name": self.name,
            "metadata": self.metadata,
            "history": [],
        }
        logging.info(
            f"Start training classifier {self.name}",
            pretty_repr(self.metadata),
        )
        history: list[dict] = result["history"]
        for stage in self.stages():
            Usage.checkpoint("stage", stage.name, "start")
            history.append(stage.run(self))
            Usage.checkpoint("stage", stage.name, "finish")
        self.cleanup_checkpoints()
        return result
