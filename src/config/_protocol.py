from __future__ import annotations

from functools import cached_property
from typing import Any, Generic, Optional, TypeVar, overload

from typing_extensions import Self  # DEPRECATE

from ._manager import _freeze_config, _status, _unset

T = TypeVar("T")


def _list_configs(cls: type[Configurable]):
    return {k: v for k in dir(cls) if isinstance(v := getattr(cls, k), config)}


class Configurable:
    __config_namespace__: tuple[str, ...]
    __config_attrs__: frozenset[tuple[str, ...]] = frozenset()
    __config_cache__: Optional[dict[tuple[str, ...], Any]]

    def __init_subclass__(cls, namespace: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if namespace is None:
            namespace = f"{cls.__module__}.{cls.__name__}"
        path = tuple(namespace.split("."))
        cls.__config_namespace__ = path
        cls.__config_cache__ = None
        for k, v in vars(cls).items():
            if isinstance(v, config):
                v._init(path + (k,))
        cls.__config_attrs__ = frozenset(v.name for v in _list_configs(cls).values())

    def __new__(cls, *_, **__):
        self = super().__new__(cls)
        if _status.frozen:
            _freeze_config(self)
        return self


class config(Generic[T]):
    @overload
    def __init__(self, value: T): ...
    @overload
    def __init__(self, name: str): ...
    @overload
    def __init__(self, value: T, name: str): ...
    def __init__(self, value: T = _unset, /, name: str = _unset):
        self.__name = tuple(name.split(".")) if name is not _unset else _unset
        self.__value = value

    @overload
    def __get__(self, instance: None, owner: type[Configurable]) -> Self[T]: ...
    @overload
    def __get__(self, instance: Configurable, owner: type[Configurable]) -> T: ...
    def __get__(self, instance: Configurable, _):
        if instance is None:
            return self
        if (cache := instance.__config_cache__) is not None:
            return cache[self.__name]
        return _status.get(self.__name)

    def __set__(self, instance: Configurable, value: T):
        if instance.__config_cache__ is not None:
            raise AttributeError("Cannot modify a frozen config.")
        self._set_data(value)

    def _init(self, name: tuple[str, ...]):
        if self.__name is _unset:
            self.__name = name
        if self.__value is not _unset:
            self._init_data(self.__value)
        del self.__value

    def _init_data(self, value: T):
        _status.default[self.name] = value

    def _set_data(self, value: T):
        _status.set(self.name, value)

    @property
    def name(self):
        return self.__name

    @cached_property
    def fullname(self):
        return ".".join(self.__name)

    @property
    def value(self) -> Any:
        return _status.get(self.__name)

    def set(self, value: T):
        self._set_data(value)

    def __repr__(self):
        return f"<{type(self).__name__}> {self.fullname} = {self.value}"


class const(config[T]):
    def _init_data(self, value: T):
        if self.name in _status.default:
            raise RuntimeError(f"Constant {self.fullname} is already initialized.")
        _status.default[self.name] = value

    def _set_data(self, _: T):
        raise RuntimeError(f"Constant {self.fullname} cannot be modified at runtime.")
