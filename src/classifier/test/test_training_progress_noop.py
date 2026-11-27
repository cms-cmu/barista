from __future__ import annotations

import torch
from torch import nn

from src.classifier.config.scheduler.basic import FixedStep
from src.classifier.config.setting import Monitor
from src.classifier.ml import BatchType
from src.classifier.ml.training import Model, MultiStageTraining, TrainingStage
from src.classifier.nn.dataset.sliceable import NamedTensorDataset


class _SpyLoss(torch.Tensor):
    item_calls = 0

    @staticmethod
    def __new__(cls, data):
        return torch.Tensor._make_subclass(cls, data, require_grad=data.requires_grad)

    def item(self):
        _SpyLoss.item_calls += 1
        return super().item()


class _DummyModel(Model):
    def __init__(self):
        self._nn = nn.Linear(4, 2)

    @property
    def nn(self) -> nn.Module:
        return self._nn

    def train(self, batch: BatchType) -> torch.Tensor:
        out = self._nn(batch["x"])
        loss = (out**2).mean()
        return _SpyLoss(loss)


class _DummyTrainer(MultiStageTraining):
    def stages(self):
        return ()


def test_training_stage_skips_loss_item_when_monitor_disabled():
    assert Monitor.enable is False
    _SpyLoss.item_calls = 0

    dataset = NamedTensorDataset(x=torch.randn(32, 4))
    trainer = _DummyTrainer(name="dummy")
    trainer.device = torch.device("cpu")

    stage = TrainingStage(
        name="Training",
        model=_DummyModel(),
        schedule=FixedStep(epoch=1, bs_init=8),
        training=dataset,
        checkpoint=False,
    )
    stage.run(trainer)

    assert _SpyLoss.item_calls == 0
