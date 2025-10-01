from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Generic, Iterable, Literal, Optional, TypeVar, overload

import numpy as np
import numpy.typing as npt

_2_PI = 2 * np.pi
_UINT64_11 = np.uint64(11)
_UINT64_32 = np.uint64(32)
_BIT52_COUNT = np.float64(1 << 53)

SeedLike = int | str | Iterable[int | str]


def _str_to_entropy(__str: str) -> list[np.uint64]:
    return np.frombuffer(hashlib.md5(__str.encode()).digest(), dtype=np.uint64).tolist()


def _seed(*entropy: SeedLike) -> tuple[int, ...]:
    seeds = []
    for e in entropy:
        if isinstance(e, str):
            seeds.extend(_str_to_entropy(e))
        elif isinstance(e, Iterable):
            seeds.extend(_seed(*e))
        else:
            seeds.append(e)
    return (*seeds,)


_KeyT = TypeVar("_KeyT")


class CBRNG(ABC, Generic[_KeyT]):
    """
    Counter-based random number generator (CBRNG).
    """

    def __init__(self, *seed: SeedLike):
        self._seed = _seed(seed)
        self._keys: dict[int, _KeyT] = {}
        self._offset: int = None

    # bit generator
    @abstractmethod
    def bit32(
        self, counters: npt.NDArray[np.uint64], key: _KeyT
    ) -> npt.NDArray[np.uint32]: ...

    @abstractmethod
    def bit64(
        self, counters: npt.NDArray[np.uint64], key: _KeyT
    ) -> npt.NDArray[np.uint64]: ...

    @abstractmethod
    def key(self, generator: np.random.Generator) -> _KeyT: ...

    # key
    @property
    def _key(self) -> _KeyT:
        k, o = self._keys, self._offset
        if o not in k:
            s = self._seed
            if o is not None:
                s += (o,)
            k[o] = self.key(np.random.Generator(np.random.PCG64(s)))
        return k[o]

    def shift(self, offset: int = None):
        cls = self.__class__
        new = cls.__new__(cls)
        new._seed = self._seed
        new._keys = self._keys
        new._offset = offset
        return new

    # basic types
    @overload
    def uint(
        self, counters: npt.ArrayLike, bits: Literal[64] = 64
    ) -> npt.NDArray[np.uint64]: ...
    @overload
    def uint(
        self, counters: npt.ArrayLike, bits: Literal[32] = 32
    ) -> npt.NDArray[np.uint32]: ...
    def uint(
        self, counters: npt.ArrayLike, bits: Literal[32, 64] = 64
    ) -> npt.NDArray[np.uint]:
        counters = np.asarray(counters, dtype=np.uint64)
        match bits:
            case 32:
                return self.bit32(counters)
            case 64:
                return self.bit64(counters)
            case _:
                raise NotImplementedError

    def uint64(self, counters: npt.ArrayLike) -> npt.NDArray[np.uint64]:
        """
        Generate a random sequence by reducing the last dimension of the counters.
        """
        counters = np.asarray(counters, dtype=np.uint64)
        if counters.ndim == 1:
            return self.uint(counters, bits=64)
        while True:
            shape = counters.shape[-1]
            if shape == 1:
                return self.uint(counters).reshape(counters.shape[:-1])
            elif shape % 2 == 0:
                counters = self.uint(counters, 32).view(np.uint64)
            else:
                counters = np.concatenate(
                    [
                        counters[..., -1:],
                        self.uint(counters[..., :-1], 32).view(np.uint64),
                    ],
                    axis=-1,
                )

    def float64(self, counters: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        [0, 1) Based on `numpy.random._common.uint64_to_double`.
        """
        x = self.uint64(counters)
        x >>= _UINT64_11
        return x / _BIT52_COUNT

    # distributions
    def uniform(self, counters: npt.ArrayLike, low: float = 0.0, high: float = 1.0):
        """
        Generates from uniform distribution.

        Parameters
        ----------
        counters : ~numpy.typing.ArrayLike
            The counter array.
        low : float, optional
            The lower (closed) bound of the distribution.
        high : float, optional
            The upper (open) bound of the distribution.

        Returns
        -------
        ndarray
            The random sample.
        """
        x = self.float64(counters)
        x *= high - low
        x += low
        return x

    def normal(self, counters: npt.ArrayLike, loc: float = 0.0, scale: float = 1.0):
        """
        Generates from normal distribution using Box-Muller transform [1]_.

        .. warning::

            In an extremely rare case (:math:`1/2^{53}`), it may raise a "divide by zero" warning.

        Parameters
        ----------
        counters : ~numpy.typing.ArrayLike
            The counter array.
        loc : float, optional
            The mean of the normal distribution.
        scale : float, optional
            The standard deviation of the normal distribution.

        Returns
        -------
        ndarray
            The random sample.

        References
        ----------
        .. [1] `Box–Muller transform - Wikipedia <https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform>`_
        """
        o = self._offset
        o = 0 if o is None else o + 1
        x = self.float64(counters)
        y = self.shift(o).float64(counters)
        x = np.sqrt(-2.0 * np.log(x)) * np.cos(_2_PI * y)
        x *= scale
        x += loc
        return x

    def choice(
        self,
        counters: npt.ArrayLike,
        a: npt.ArrayLike | int,
        p: Optional[npt.ArrayLike] = None,
    ) -> npt.NDArray:
        R"""
        Generates a random sample from a given array.

        Parameters
        ----------
        counters : ~numpy.typing.ArrayLike
            The counter array.
        a : ~numpy.typing.ArrayLike or int
            If an array-like object, a random sample is generated by choosing from the first dimension. If an int, the random sample is generated as if it were ``np.arange(a)``.
        p : ~numpy.typing.ArrayLike, optional
            The probabilities associated with each entry in ``a``. If not given, the sample assumes a uniform distribution over all entries.

        Returns
        -------
        ndarray
            The random sample.

        Notes
        -----
        If the ``counters`` has shape :math:`[x_{1}, x_{2}, ..., x_{n}]` and ``a`` has shape :math:`[y_{1}, y_{2}, ..., y_{m}]`, the output shape is given as follows:
        
        .. math::

            &[x_{1}, x_{2}, ..., x_{n-1}, y_{2}, ..., y_{m}] & n > 1 \land m > 1\\
            &[x_{1}, x_{2}, ..., x_{n-1}] & n > 1 \land m = 1\\
            &[x_{1}, y_{2}, ..., y_{m}] & n = 1 \land m > 1\\
            &[x_{1}] & n = 1 \land m = 1
        """
        if isinstance(a, int):
            a = np.arange(a)
        else:
            a = np.asarray(a)
        length = a.shape[0]
        if p is not None:
            p = np.asarray(p)
            if p.ndim != 1:
                raise ValueError("p must be 1-dimensional")
            if p.shape[0] != length:
                raise ValueError("a and p must have same size")
            p = np.cumsum(p)
            p /= p[-1]
        if p is None:
            return a[self.uint64(counters) % length]
        else:
            return a[np.searchsorted(p, self.float64(counters))]


class Squares(CBRNG[np.uint64]):
    """
    Squares: a counter-based random number generator (CBRNG) [1]_.

    .. [1] https://arxiv.org/abs/2004.06278
    """

    def key(self, gen: np.random.Generator) -> np.uint64:
        bits = np.arange(1, 16, dtype=np.uint64)
        offsets = np.arange(0, 29, 4, dtype=np.uint64)
        lower8 = gen.choice(bits, 8, replace=False)
        for i in range(16):
            if lower8[i] % 2 == 1:
                lower8 = np.roll(lower8, -i)
                break
        higher8 = np.zeros(8, dtype=np.uint64)
        higher8[0:1] = gen.choice(np.delete(bits, int(lower8[-1]) - 1), 1)
        higher8[1:] = gen.choice(np.delete(bits, int(higher8[0]) - 1), 7, replace=False)
        return np.sum(lower8 << offsets) + (np.sum(higher8 << offsets) << _UINT64_32)

    def _round(self, lr: npt.NDArray, shift: npt.NDArray, last: bool = False):
        lr *= lr
        lr += shift
        if last:
            yield lr.copy()
        l = lr >> _UINT64_32
        lr <<= _UINT64_32
        lr |= l
        yield lr

    def bit32(self, ctrs: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint32]:
        x = ctrs * self._key
        y = x.copy()
        z = y + self._key
        # round 1-3
        for i in [y, z, y]:
            (_,) = self._round(x, i)
        # round 4
        x *= x
        x += z
        x >>= _UINT64_32
        return x.astype(np.uint32)

    def bit64(self, ctrs: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
        x = ctrs * self._key
        y = x.copy()
        z = y + self._key
        # round 1-3
        for i in [y, z, y]:
            (_,) = self._round(x, i)
        # round 4
        (t, _) = self._round(x, z, last=True)
        # round 5
        x *= x
        x += y
        x >>= _UINT64_32
        x ^= t
        return x


class Philox(CBRNG):
    """
    Philox: a counter-based random number generator (CBRNG) [1]_.

    .. [1] https://doi.org/10.1145/2063384.2063405
    """

    # TODO: implement Philox
