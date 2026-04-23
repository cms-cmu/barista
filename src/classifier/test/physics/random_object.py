from dataclasses import dataclass
from typing import Optional

import awkward as ak
import numpy as np


def typed_uniform(
    rng: np.random.Generator, low: float, high: float, size: int, dtype: np.dtype
):
    # numpy.random.Generator.uniform does not support dtype
    data = rng.random(size, dtype=dtype)
    data *= high - low
    data += low
    return data


@dataclass
class RandomMPtEtaPhi:
    pt_range: tuple[float, float] = (40, 1000)
    mass_pt_ratio: tuple[float, float] = (0.03, 0.15)
    mass_range: tuple[Optional[float], Optional[float]] = (None, None)
    eta_range: tuple[float, float] = (-5, 5)
    phi_range: tuple[float, float] = (-np.pi, np.pi)
    count_range: tuple[int, int] = (4, 12)
    dtype: np.dtype = np.float32

    def generate(self, nevent: int, seed: int) -> ak.Array:
        rng = np.random.Generator(np.random.PCG64(seed))
        count = rng.integers(self.count_range[0], self.count_range[1] + 1, nevent)
        total = np.sum(count)
        pt = np.exp(typed_uniform(rng, *np.log(self.pt_range), total, dtype=self.dtype))
        mass = typed_uniform(
            rng,
            self.mass_pt_ratio[0] * pt,
            self.mass_pt_ratio[1] * pt,
            total,
            dtype=self.dtype,
        )
        np.clip(mass, *self.mass_range, out=mass)
        eta = typed_uniform(rng, *self.eta_range, total, dtype=self.dtype)
        phi = typed_uniform(rng, *self.phi_range, total, dtype=self.dtype)
        return ak.unflatten(
            ak.zip({"pt": pt, "mass": mass, "eta": eta, "phi": phi}),
            count,
        )
