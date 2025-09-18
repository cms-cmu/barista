from typing import Callable, Iterable, Optional

import src.data_formats.numpy as npext
import numpy as np
import torch
import torch.types as tt
from torch import Tensor

from ..hist import FloatWeighted, RegularAxis
from ..utils import to_arr, to_num


def linear_differ(pos: Tensor, neg: Tensor):
    return (pos - neg) / 2 + 0.5


class FixedThresholdROC:
    def __init__(
        self,
        thresholds: RegularAxis | Iterable[float],
        positive_classes: Iterable[int],
        negative_classes: Iterable[int] = None,
        score_interpretation: Callable[[Tensor, Tensor], Tensor] = None,
    ):
        self._hist = FloatWeighted(thresholds, err=False)
        self._pos = sorted(set(positive_classes))
        self._has_neg = negative_classes is not None
        if self._has_neg:
            self._neg = sorted(set(negative_classes))
            self._map = score_interpretation
        self.reset()

    def reset(self):
        for k in ("t", "pos", "neg", "edge", "FP", "TP", "P", "N"):
            setattr(self, f"_{FixedThresholdROC.__name__}__{k}", None)

    def copy(self):
        new = super().__new__(self.__class__)
        new.__dict__ = self.__dict__.copy()
        new.reset()
        return new

    def _init(self, f: tt._dtype, i: tt._dtype, d: tt.Device):
        if self.__t is None:
            self.__t = {"dtype": f, "device": d}
            self.__pos = torch.as_tensor(self._pos, dtype=i, device=d)
            if self._has_neg:
                self.__neg = torch.as_tensor(self._neg, dtype=i, device=d)
            self.__FP = self._hist.copy(**self.__t)
            self.__TP = self._hist.copy(**self.__t)
            self.__P = torch.tensor(0, **self.__t)
            self.__N = torch.tensor(0, **self.__t)

    def _score(self, y_pred: Tensor):
        pos = y_pred[:, self._pos].sum(dim=-1)
        if (not self._has_neg) or (self._map is None):
            return pos
        else:
            neg = y_pred[:, self._neg].sum(dim=-1)
            return self._map(pos, neg)

    def _bounded(self, FPR: Tensor, TPR: Tensor):
        f, t = [FPR], [TPR]
        if FPR[0] != 1.0:
            f.insert(0, torch.tensor([1.0], **self.__t))
            t.insert(0, f[0])
        if FPR[-1] != 0.0:
            f.append(torch.tensor([0.0], **self.__t))
            t.append(f[-1])
        if len(f) > 1:
            return torch.cat(f), torch.cat(t)
        return FPR, TPR

    @staticmethod
    def _monotonic(new: Tensor, old: Tensor):
        new = new >= torch.cummax(new, dim=0)[0]
        if old is not None:
            new &= old
        return new

    @staticmethod
    def _check_shape(**tensors: tuple[Tensor, int]):
        sizes = {}
        for k, (v, dim) in tensors.items():
            if v.dim() > dim:
                raise ValueError(f"{k} must have at most {dim} dimensions")
            sizes[k] = len(v)
        if not len(set(sizes.values())) == 1:
            msg = ", ".join(f"{k}({v})" for k, v in sizes.items())
            raise ValueError(f"{msg} must have the same length")

    @torch.no_grad()
    def update(
        self,
        y_pred: Tensor = None,
        y_true: Tensor = None,
        weight: Optional[Tensor] = None,
    ):
        if y_pred is None or len(y_pred) == 0:
            return
        self._init(
            weight.dtype if weight is not None else y_pred.dtype,
            y_true.dtype,
            y_pred.device,
        )
        # prepare data
        self._check_shape(y_pred=(y_pred, 2), y_true=(y_true, 1))
        if y_pred.dim() == 2:
            y_pred = self._score(y_pred)
        if weight is None:
            weight = torch.ones_like(y_pred)
        else:
            self._check_shape(weight=(weight, 1), y_true=(y_true, 1))
        # update P, N, TP, FP
        p = torch.isin(y_true, self.__pos)
        n = torch.isin(y_true, self.__neg) if self._has_neg else ~p
        self.__P += weight[p].sum()
        self.__N += weight[n].sum()
        self.__TP.fill(y_pred[p], weight[p])
        self.__FP.fill(y_pred[n], weight[n])

    @torch.no_grad()
    def roc(self):
        if self.__t is None or self.__P == 0 or self.__N == 0:
            return torch.tensor([np.nan]), torch.tensor([np.nan]), torch.tensor(np.nan)
        __TP, _ = self.__TP.hist()
        __FP, _ = self.__FP.hist()
        fpr = torch.cumsum(__FP, dim=0) / self.__N
        tpr = torch.cumsum(__TP, dim=0) / self.__P
        # deal with negative weights
        monotonic = None
        if torch.any(__FP < 0.0):
            monotonic = self._monotonic(fpr, monotonic)
        if torch.any(__TP < 0.0):
            monotonic = self._monotonic(tpr, monotonic)
        if monotonic is not None:
            fpr, tpr = fpr[monotonic], tpr[monotonic]
        # add missing (0,0) or (1,1)
        fpr, tpr = self._bounded(1 - fpr, 1 - tpr)
        # AUC
        auc = -torch.trapz(tpr, fpr)
        self.reset()
        return fpr, tpr, auc

    def to_json(self):
        fpr, tpr, auc = self.roc()
        return {
            "FPR": npext.to.base64(to_arr(fpr)),
            "TPR": npext.to.base64(to_arr(tpr)),
            "AUC": to_num(auc),
        }
