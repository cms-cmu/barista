from __future__ import annotations

from copy import deepcopy
from typing import Callable, Iterable

from hist.axis import AxesMixin

from ..aktools import FieldLike, get_field
from ..typetools import check_type
from ..utils import astuple
from ..utils.regex import compile_any_wholeword, match_single
from . import hist as _h


class Template:
    class _Hist:
        def __init__(self, *axes: _h.AxisLike, **fill_args: _h.LazyFill):
            self._axes = [
                (
                    (
                        _h.Label(axis.name, axis.label)
                        if isinstance(axis, AxesMixin)
                        else _h.Label(axis[-1])
                    ),
                    axis,
                )
                for axis in axes
            ]
            self.fill_args = fill_args

        def axes(self, name: str, template: Template) -> list[_h.AxisLike]:
            _axes = []
            for label, axis in self._axes:
                args = template.rebin(name, label.code)
                if args is None:
                    _axes.append(axis)
                else:
                    if isinstance(args, tuple):
                        _axes.append((*args, label))
                    elif args is ...:
                        _axes.append((label,))
                    elif isinstance(args, AxesMixin):
                        args = deepcopy(args)
                        args.label = label.display
                        args.name = label.code
                        _axes.append(args)
            return _axes

    def __init__(
        self,
        name: _h.LabelLike,
        fill: FieldLike = ...,
        bins: dict[str | tuple[str, str], _h.AxisArgs] = None,
        skip: Iterable[str] = None,
        collection: _h._Collection = None,
        **fill_args: _h.LazyFill,
    ):
        self._name = _h.Label(name)
        self._data = fill
        self._bins = bins.copy() if bins is not None else {}
        self._skip = compile_any_wholeword(skip)
        self._collection = collection
        self._fill_args = fill_args

        self._fills = None
        self._instanced = False
        self._parent: Template = None

    def copy(self):
        return self.__class__(
            self._name, self._data, self._bins, self._skip, **self._fill_args
        )

    @property
    def collection(self):
        if self._parent is not None:
            return self._parent.collection
        if self._collection is not None:
            return self._collection
        return _h._Collection.current

    @property
    def data(self):
        if self._parent is not None:
            return self._parent.data + self._data
        return self._data

    @property
    def fill_args(self):
        if self._parent is not None:
            return self._fill_args | self._parent.fill_args
        return self._fill_args

    def hist_name(self, name: str, nested: bool = False):
        name = f"{self._name.code}.{name}"
        if nested and self._parent is not None:
            return self._parent.hist_name(name, nested)
        return name

    def axis_label(self, label: str):
        label = f"{self._name.display} {label}"
        if self._parent is not None:
            label = self._parent.axis_label(label)
        return label

    def rebin(self, name: str, axis: str):
        _axis = None
        if self._parent is not None:
            _axis = self._parent.rebin(self.hist_name(name), axis)
        if _axis is None:
            _axis = self._bins.get((name, axis), self._bins.get(axis))
        return _axis

    def skip(self, name: str):
        skip = False
        if self._parent is not None:
            skip |= self._parent.skip(self.hist_name(name))
        if not skip:
            skip |= match_single(self._skip, name)
        return skip

    def new(self, name: str = None, parent: Template = None):
        if not self._instanced:
            self._instanced = True
            self._parent = parent
            self._fills = _h._Fill()
            if name is not None:
                self._name.code = name
            self._data = astuple(
                self._data if self._data is not ... else _h._fill_field(self._name.code)
            )
            hists, templates = self.hists()
            for name, hist in hists.items():
                if not self.skip(name):
                    self._add(name, *hist.axes(name, self), **hist.fill_args)
            for name, template in templates.items():
                if template._instanced:
                    raise _h.HistError(
                        f'Template "{self.__class__.__name__}.{name}" has already been used'
                    )
                template = template.copy()
                self._fills += template.new(name, self)
        return self._fills

    def _add(self, name: str, *axes: _h.AxisLike, **fill_args: _h.LazyFill):
        _kwargs = {}
        fill_args = fill_args | self.fill_args
        axes = [_h._create_axis(axis) for axis in axes]
        data = self.data
        for axis in axes:
            axis.label = self.axis_label(axis.label)
            if axis.name in fill_args:
                _fill = fill_args[axis.name]
                if check_type(_fill, FieldLike):
                    _fill = data + astuple(_fill)
                elif check_type(_fill, Callable):
                    _fill = self._wrap(_fill)
                _kwargs[axis.name] = _fill
            else:
                _kwargs[axis.name] = data + _h._fill_field(axis.name)
        if "weight" in fill_args:
            _kwargs["weight"] = fill_args["weight"]
        self._fills += self.collection.add(
            self.hist_name(name, nested=True), *axes, **_kwargs
        )

    def _wrap(self, func: Callable):
        return lambda x: func(get_field(x, self.data))

    @classmethod
    def hists(cls) -> tuple[dict[str, _Hist], dict[str, Template]]:
        hists, templates = {}, {}
        for name in dir(cls):
            attr = getattr(cls, name)
            if isinstance(attr, cls._Hist):
                hists[name] = attr
            elif isinstance(attr, Template):
                templates[name] = attr
        return hists, templates

    def __add__(self, other: _h._Fill | Template) -> _h.Fill:
        if isinstance(other, Template):
            other = other.new()
        if isinstance(other, _h._Fill):
            return self.new() + other
        return NotImplemented


class Systematic(Template):
    def __init__(
        self,
        name: str,
        systs: Iterable[_h.LabelLike],
        *axes: AxesMixin | tuple,
        weight: FieldLike = "weight",
        collection: _h._Collection = None,
        **fill_args: FieldLike,
    ):
        super().__init__((name, ""), (), collection=collection, **fill_args)
        weight = astuple(weight)
        if len(axes) == 0:
            axes = self.collection.duplicate_axes(name)
        for _var in systs:
            _var = _h.Label(_var)
            self._name.display = f"({_var.display})"
            self._add(_var.code, *axes, weight=weight + _h._fill_field(_var.code))
        self._name.display = ""
