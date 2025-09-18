from numbers import Real
from typing import Iterable

import src.data_formats.numpy as npext
import numpy as np
import torch
import torch.types as tt
from src.typetools import check_type
from torch import Tensor

from ...utils import not_none
from ..utils import to_arr

RegularAxis = tuple[int, Real, Real]


class FloatWeighted:
    def __init__(
        self,
        bins: RegularAxis | Iterable[float],
        err: bool = True,
        dtype: tt._dtype = None,
        device: tt.Device = None,
    ):
        self._err = err
        self._dtype = dtype
        self._device = device

        if check_type(bins, RegularAxis):
            self.__reg = bins
            step = (bins[2] - bins[1]) / bins[0]
            self._edge = (bins[0], step, bins[1] / step - 1)
            self._nbin = bins[0] + 2
        else:
            self.__reg = None
            self._edge = np.sort(bins)
            self._nbin = len(self._edge) + 1
        self.reset()

    @property
    def is_regular(self):
        return self.__reg is not None

    def copy(self, dtype: tt._dtype = None, device: tt.Device = None):
        new = super().__new__(self.__class__)
        new.__dict__ = self.__dict__.copy()
        new.reset()
        if dtype is not None:
            new._dtype = dtype
        if device is not None:
            new._device = device
        return new

    def reset(self):
        self.__bins: Tensor = None
        self.__errs: Tensor = None
        self.__edge: Tensor = None

    def _init(self, x: Tensor, weight: Tensor):
        _b = dict(
            dtype=not_none(self._dtype, weight.dtype),
            device=not_none(self._device, weight.device),
        )
        self.__bins = torch.zeros(self._nbin, **_b)
        if self._err:
            self.__errs = torch.zeros(self._nbin, **_b)
        _e = dict(
            dtype=not_none(self._dtype, x.dtype),
            device=not_none(self._device, x.device),
        )
        if self.is_regular:
            self.__edge = torch.tensor(self._edge[1:], **_e)
        else:
            self.__edge = torch.as_tensor(self._edge, **_e)

    @torch.no_grad()
    def fill(self, x: Tensor, weight: Tensor):
        # -inf < b[0] < e[0] <= b[1] < e[1] <= ... < e[-1] <= b[-1] < inf
        if self.__bins is None:
            self._init(x=x, weight=weight)
        e = self.__edge
        if self.is_regular:
            indices = torch.clip(x / e[0] - e[1], 0, self._edge[0] + 1).to(torch.int32)
        else:
            indices = torch.bucketize(x, e, right=True, out_int32=True)
        self.__bins.index_add_(0, indices, weight)
        if self._err:
            self.__errs.index_add_(0, indices, weight**2)

    @torch.no_grad()
    def hist(self):
        return self.__bins, torch.sqrt(self.__errs) if self._err else None

    def __repr__(self):
        if self.is_regular:
            b = self.__reg
            edges = np.linspace(b[1], b[2], b[0] + 1)
        else:
            edges = self._edge
        prev = "-\u221E"
        lines = []
        vals, errs = self.hist()
        for i, edge in enumerate([*map("{:.6g}".format, edges), "\u221E"]):
            line = "(" if i == 0 else "["
            line += f"{prev}, {edge}) {vals[i]:.6g}"
            if self._err:
                line += f" \u00B1 {errs[i]:.6g}"
            prev = edge
            lines.append(line)
        return "\n".join(lines)

    def to_json(self):
        vals, errs = self.hist()
        return {
            "values": npext.to.base64(to_arr(vals)),
            "errors": npext.to.base64(to_arr(errs)) if self._err else None,
            "edges": self.__reg if self.is_regular else npext.to.base64(self._edge),
        }
