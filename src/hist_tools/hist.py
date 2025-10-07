from __future__ import annotations

import sys
from copy import deepcopy
from textwrap import indent
from typing import (
    Callable,
    Generic,
    Iterable,
    TypedDict,
    TypeVar,
    overload,
)

from packaging.version import Version
from rich.pretty import pretty_repr
from typing_extensions import Self  # DEPRECATE

import awkward as ak
import numpy as np
import numpy.typing as npt
from hist import Hist
from hist.axis import (
    AxesMixin,
    Boolean,
    IntCategory,
    Integer,
    Regular,
    StrCategory,
    Variable,
)

from ..aktools import (
    AnyInt,
    FieldLike,
    RealNumber,
    and_fields,
    get_field,
)
from ..config import Configurable, config
from ..data_formats import awkward as akext
from ..typetools import check_type, find_subclass
from . import template as _t

HistAxis = Boolean | IntCategory | Integer | Regular | StrCategory | Variable


class Label:
    @overload
    def __init__(self, label: LabelLike): ...
    @overload
    def __init__(self, code: str, display: str): ...
    def __init__(self, code: LabelLike, display: str = ...):
        if isinstance(code, Label):
            self.code = code.code
            self.display = code.display
        elif isinstance(code, tuple):
            self.code = code[0]
            self.display = code[1]
        elif isinstance(code, str):
            self.code = code
            self.display = code if display is ... else display

    def askwarg(self, code: str = "code", display: str = "display"):
        return {code: self.code, display: self.display}


def _fill_field(_s: str):
    return (*_s.split("."),)


def _fill_special(hist: str, axis: str):
    return f"{hist} \x00 {axis}"


def _fill_repr(fill: str):
    fills = fill.split(" \x00 ")
    if len(fills) == 1:
        return fills[0]
    else:
        return f'{fills[1]} (hist "{fills[0]}")'


def _create_axis(args: AxisLike) -> HistAxis:
    if isinstance(args, AxesMixin):
        return deepcopy(args)
    if len(args) == 0:
        raise HistError('require at least one argument "name" to create an axis')
    label = Label(args[-1]).askwarg("name", "label")
    if len(args) == 4:
        if (
            check_type(args[0], AnyInt)
            and args[0] > 0
            and all(check_type(arg, RealNumber) for arg in args[1:3])
        ):
            return Regular(*args[0:3], **label)
    elif len(args) == 3:
        if all(check_type(arg, AnyInt) for arg in args[0:2]) and args[0] <= args[1]:
            return Integer(*args[0:2], **label)
    elif len(args) == 2:
        if args[0] is ...:
            return Boolean(**label)
        elif isinstance(args[0], Iterable):
            if all(isinstance(arg, str) for arg in args[0]):
                return StrCategory(args[0], **label, growth=True)
            elif all(check_type(arg, AnyInt) for arg in args[0]):
                return IntCategory(args[0], **label, growth=True)
            elif all(check_type(arg, RealNumber) for arg in args[0]):
                return Variable(args[0], **label)
    elif len(args) == 1:
        return Boolean(**label)
    raise HistError(f'cannot create axis from arguments "{args}"')


LazyFill = FieldLike | Callable
FillLike = LazyFill | npt.ArrayLike | RealNumber | bool
LabelLike = str | tuple[str, str] | Label

RegularArgs = tuple[int, RealNumber, RealNumber]
RegularAxis = tuple[int, RealNumber, RealNumber, LabelLike] | Regular
IntegerArgs = tuple[int, int]
IntegerAxis = tuple[int, int, LabelLike] | Integer
BooleanArgs = tuple[Ellipsis] | type[Ellipsis]
BooleanAxis = tuple[Ellipsis, LabelLike] | tuple[LabelLike] | Boolean
StrCategoryArgs = tuple[Iterable[str]]
StrCategoryAxis = tuple[Iterable[str], LabelLike] | StrCategory
IntCategoryArgs = tuple[Iterable[int]]
IntCategoryAxis = tuple[Iterable[int], LabelLike] | IntCategory
VariableArgs = tuple[Iterable[RealNumber]]
VariableAxis = tuple[Iterable[RealNumber], LabelLike] | Variable

AxisArgs = (
    RegularArgs
    | IntegerArgs
    | BooleanArgs
    | StrCategoryArgs
    | IntCategoryArgs
    | VariableArgs
)
AxisLike = (
    RegularAxis
    | IntegerAxis
    | BooleanAxis
    | StrCategoryAxis
    | IntCategoryAxis
    | VariableAxis
)


class FillError(Exception): ...


class HistError(Exception): ...


HistType = TypeVar("HistType", bound=Hist)
FillType = TypeVar("FillType", bound="_Fill")


class _MissingFillValue: ...


class _Fill(Generic[HistType], Configurable, namespace="hist.Fill"):
    class __backend__:
        ak: ak
        check_empty_mask: bool
        anyarray: type
        broadcast_all: Callable[..., dict[str]] = None

        allow_str_array: bool = Version(ak.__version__) >= Version("2.0.0")

    allow_missing = config(True)

    def __init__(
        self,
        fills: dict[str, list[str]] = None,
        weight="weight",
        **fill_args: FillLike,
    ):
        self._fills = {} if fills is None else fills
        self._kwargs = fill_args | {"weight": weight}

    def __add__(self, other: _Fill | _t.Template) -> _Fill:
        if isinstance(other, _Fill):
            if (backend := find_subclass(self, other)) is None:
                raise FillError("Cannot merge fill using different backends")
            fills = other._fills | self._fills
            kwargs = other._kwargs | self._kwargs
            return backend(fills, **kwargs)
        elif isinstance(other, _t.Template):
            return self + other.new()
        return NotImplemented

    def __call__(
        self,
        events: ak.Array,
        hists: _Collection[HistType, Self] = ...,
        **fill_args: FillLike,
    ):
        self.fill(events, hists, **fill_args)

    def __setitem__(self, key: str, value: FillLike):
        self._kwargs[key] = value

    def __getitem__(self, key: str):
        return self._kwargs[key]

    def _get_fill_arg(self, method: Callable[[], FillLike]):
        try:
            return method()
        except Exception:
            if self.allow_missing:
                return _MissingFillValue
            raise

    def fill(
        self,
        events: ak.Array,
        hists: _Collection[HistType, Self] = ...,
        **fill_args: FillLike,
    ):
        _ak = self.__backend__.ak
        if hists is ...:
            if (hists := _Collection.current) is None:
                raise FillError("\nNo histogram collection is specified")
        if not isinstance(self, hists.__backend__.fill):
            raise FillError(
                "\nCannot fill a histogram collection with a different backend."
            )
        fill_args = self._kwargs | fill_args
        mask_categories = []
        for category in hists._categories:
            if category not in fill_args:
                field = _fill_field(category)
                if isinstance(hists._axes[category], StrCategory) and not (
                    self.__backend__.allow_str_array
                    and akext.is_array(get_field(events, field))
                ):
                    mask_categories.append(category)
                else:
                    fill_args[category] = field
        for fill_values in hists._generate_category_combinations(mask_categories):
            mask = and_fields(
                events, *(_fill_field(f"{k}.{v}") for k, v in fill_values.items())
            )
            masked = events if mask is None else events[mask]
            if self.__backend__.check_empty_mask and len(masked) == 0:
                continue
            for k, v in fill_args.items():
                try:
                    if (isinstance(v, str) and k in hists._categories) or isinstance(
                        v, bool | RealNumber
                    ):
                        fill_values[k] = v
                    elif check_type(v, FieldLike):
                        fill_values[k] = self._get_fill_arg(
                            lambda: get_field(masked, v)
                        )
                    elif isinstance(v, self.__backend__.anyarray):
                        fill_values[k] = v if mask is None else v[mask]
                    elif isinstance(v, Callable):
                        fill_values[k] = self._get_fill_arg(lambda: v(masked))
                    else:
                        raise TypeError("Unsupported fill value.")
                except Exception:
                    raise FillError(
                        f'\nWhile preparing fill value "{_fill_repr(k)}" from\n  {type(v)}\n{indent(pretty_repr(v), "    ")}\n the above error occurred.'
                    )
            for name in self._fills:
                hist_args = {}
                try:
                    fills = {
                        k: (
                            special
                            if (special := _fill_special(name, k)) in fill_values
                            else k
                        )
                        for k in self._fills[name]
                    }
                    if any(fill_values[v] is _MissingFillValue for v in fills.values()):
                        continue
                    arrays = {}
                    depths = {}
                    for v in fills.values():
                        fill = fill_values[v]
                        if isinstance(fill, self.__backend__.anyarray):
                            arrays[v] = fill
                            depths[v] = akext.max_depth(fill)
                    depths_set = set(depths.values())
                    if len(depths_set) > 1:
                        arrays = dict(
                            zip(arrays.keys(), _ak.broadcast_arrays(*arrays.values()))
                        )
                    if max(depths_set) > 1:
                        for k in arrays:
                            arrays[k] = _ak.ravel(arrays[k])
                    for k, v in fills.items():
                        if (fill := arrays.get(v)) is None:
                            fill = fill_values[v]
                        hist_args[k] = fill
                    # https://github.com/scikit-hep/boost-histogram/issues/452 #
                    if (self.__backend__.broadcast_all is not None) and all(
                        [
                            isinstance(axis, StrCategory)
                            for axis in hists._hists[name].axes
                        ]
                    ):
                        hist_args = self.__backend__.broadcast_all(**hist_args)
                    ############################################################
                    hists._hists[name].fill(**hist_args)
                    hists._filled.add(name)
                except Exception:
                    if hist_args:
                        msg = f'filling histogram "{name}", with\n" + {indent(pretty_repr(hist_args), "    ")}'
                    else:
                        msg = f'preparing the arguments for histogram "{name}"'
                    raise FillError(f"\nWhile {msg}\n the above exception occurred.")


if sys.version_info >= (3, 11):

    class CollectionOutput(Generic[HistType], TypedDict):
        hists: dict[str, HistType]
        categories: set[str]

else:

    class CollectionOutput(TypedDict):
        hists: dict[str, Hist]
        categories: set[str]


class _Collection(Generic[HistType, FillType]):
    class __backend__:
        hist: type[HistType]
        fill: type[FillType]

    current: Self

    def __init__(self, **categories):
        self._fills: dict[str, list[str]] = {}
        self._hists: dict[str, HistType] = {}
        self._categories = deepcopy(categories)
        self._axes: dict[str, HistAxis] = {
            k: _create_axis((*(v if isinstance(v, tuple) else (v,)), k))
            for k, v in self._categories.items()
        }
        self._filled: set[str] = set()
        self.cd()

    def add(self, name: str, *axes: AxisLike, **fill_args: FillLike):
        if name in self._hists:
            raise FillError(f'Histogram "{name}" already exists')
        axes = [_create_axis(axis) for axis in axes]
        self._fills[name] = [_axis.name for _axis in axes]
        self._hists[name] = self.__backend__.hist(
            *self._axes.values(), *axes, storage="weight", label="Events"
        )
        return self.auto_fill(name, **fill_args)

    def _generate_category_combinations(
        self, categories: list[str]
    ) -> list[dict[str, FillLike]]:
        if len(categories) == 0:
            return [{}]
        else:
            combs = np.stack(
                np.meshgrid(*(self._categories[category] for category in categories)),
                axis=-1,
            ).reshape((-1, len(categories)))
            return [dict(zip(categories, comb.tolist())) for comb in combs]

    def auto_fill(self, name: str, **fill_args: FillLike):
        default_args = {
            k: _fill_field(k) for k in self._fills[name] if k not in fill_args
        }
        fill_args = {
            _fill_special(name, k): v for k, v in fill_args.items()
        } | default_args
        fills = {name: self._fills[name] + [*self._categories] + ["weight"]}
        return self.__backend__.fill(fills, **fill_args)

    def duplicate_axes(self, name: str) -> list[HistAxis]:
        axes = []
        if name in self._hists:
            for axis in self._hists[name].axes:
                if axis not in self._axes.values():
                    axes.append(deepcopy(axis))
        return axes

    def cd(self):
        _Collection.current = self

    def to_dict(self, nonempty: bool = False) -> CollectionOutput[HistType]:
        if nonempty:
            hists = (k for k in self._hists if k in self._filled)  # preserve order
        else:
            hists = self._hists
        return {
            "hists": {k: self._hists[k] for k in hists},
            "categories": set(self._categories),
        }

    @property
    def output(self):
        return self.to_dict()


def _broadcast_all(weight: FillLike, **kwargs: FillLike):
    tobroadcast = None
    for k, v in kwargs.items():
        tobroadcast = k
        if isinstance(v, ak.Array | npt.NDArray) and len(v) == len(weight):
            tobroadcast = None
            break
    if tobroadcast is not None:
        kwargs[tobroadcast] = np.full(len(weight), kwargs[tobroadcast])
    kwargs["weight"] = weight
    return kwargs


class Fill(_Fill[Hist]):
    class __backend__(_Fill.__backend__):
        ak = ak
        check_empty_mask = True
        anyarray = ak.Array | np.ndarray
        broadcast_all = _broadcast_all


class Collection(_Collection[Hist, Fill]):
    class __backend__(_Collection.__backend__):
        hist = Hist
        fill = Fill
