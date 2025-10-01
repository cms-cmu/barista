import operator as op
from typing import Any, Callable, Generic, TypeVar

_BoundedT = TypeVar("_BoundedT")


class bounded(Generic[_BoundedT]):
    msg = "Require a value {comp} {bound}, got {value}."
    _ops = {
        True: (op.gt, op.lt, ">=", "<=", "[", "]"),
        False: (op.ge, op.le, ">", "<", "(", ")"),
    }

    def __init__(
        self,
        type: Callable[[Any], _BoundedT],
        lower: _BoundedT = None,
        upper: _BoundedT = None,
        closed_lower: bool = True,
        closed_upper: bool = True,
    ):
        self._type = type
        self._lower = lower
        self._upper = upper
        self._comp_lower = self._ops[closed_lower]
        self._comp_upper = self._ops[closed_upper]

    def __call__(self, value) -> _BoundedT:
        num = self._type(value)
        ol, ou = self._comp_lower, self._comp_upper
        if self._lower is not None and ol[0](self._lower, num):
            raise ValueError(self.msg.format(comp=ol[2], bound=self._lower, value=num))
        if self._upper is not None and ou[1](self._upper, num):
            raise ValueError(self.msg.format(comp=ou[3], bound=self._upper, value=num))
        return num

    def __repr__(self):
        r = self._type.__name__
        ol, ou = self._comp_lower, self._comp_upper
        if self._lower is not None:
            r += f"{ol[4]}{self._lower}"
        else:
            r += "(-\u221E"
        r += ", "
        if self._upper is not None:
            r += f"{self._upper}{ou[5]}"
        else:
            r += "\u221E)"
        return r

    def __str__(self):
        return repr(self)


int_pos = bounded(int, lower=1)
int_neg = bounded(int, upper=-1)
float_pos = bounded(float, lower=0, closed_lower=False)
float_neg = bounded(float, upper=0, closed_upper=False)
