from dataclasses import dataclass

from .basic import AutoStep


@dataclass
class FinetuneStep(AutoStep):
    lr_init: float = 1.0e-4
    lr_scale: float = 0.5
    lr_threshold: float = 1e-3
    lr_patience: int = 0
    lr_cooldown: int = 0
    lr_min: float = 1e-9


class FinetuneStepSGD(FinetuneStep):  # PLAN test performance
    def optimizer(self, parameters, **kwargs):
        import torch.optim as optim

        return optim.SGD(parameters, lr=self.lr_init, **kwargs)
