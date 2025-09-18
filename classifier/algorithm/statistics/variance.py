import torch
from src.math.statistics import Variance


class TensorVariance(Variance[torch.Tensor]):
    _t = torch.float32  # round-off error: ~1e-7
    _k = {"dim": 0, "keepdim": True, "dtype": _t}

    @classmethod
    @torch.no_grad()
    def compute(cls, data: torch.Tensor, weight: torch.Tensor = None):
        sumw = (
            torch.tensor(len(data), dtype=cls._t, device=data.device)
            if weight is None
            else weight.sum(dtype=cls._t)
        )
        sumw2 = sumw.clone() if weight is None else weight.pow(2).sum(dtype=cls._t)
        if weight is not None:
            for _ in range(len(data.shape) - 1):
                weight = weight.unsqueeze(-1)
        m1 = (
            data.mean(**cls._k)
            if weight is None
            else ((data * weight).sum(**cls._k) / sumw)
        )
        M2 = (
            (data - m1).pow(2).sum(**cls._k)
            if weight is None
            else ((data - m1).pow(2) * weight).sum(**cls._k)
        )
        return sumw, sumw2, m1, M2
