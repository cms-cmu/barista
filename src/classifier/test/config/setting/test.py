import random

from src.classifier.task import GlobalSetting

from ..state.torch import DefaultSetting


class Randomness(GlobalSetting):
    seed: int = ...
    n_threads: int = ...
    deterministic_algorithm: bool = ...

    @classmethod
    def set__seed(cls, value: int):
        if value is ...:
            return None

        import numpy as np
        import torch

        random.seed(value)
        np.random.seed(value)
        if DefaultSetting.seed is None:
            DefaultSetting.seed = (torch.initial_seed(), torch.cuda.initial_seed())
        (cpu, cuda) = DefaultSetting.seed if value is None else (value, value)
        torch.manual_seed(cpu)
        torch.cuda.manual_seed_all(cuda)
        return value

    @classmethod
    def set__n_threads(cls, value: int):
        if value is ...:
            return None

        import torch

        if DefaultSetting.n_threads is None:
            DefaultSetting.n_threads = (
                torch.get_num_threads(),
                torch.get_num_interop_threads(),
            )
        (intraop, interop) = (
            DefaultSetting.n_threads if value is None else (value, value)
        )
        torch.set_num_threads(intraop)
        torch.set_num_interop_threads(interop)
        return value

    @classmethod
    def set__deterministic_algorithm(cls, value: bool):
        if value is ...:
            return False

        import torch

        torch.use_deterministic_algorithms(value)
        return value
