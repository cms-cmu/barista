from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, ContextManager, TypedDict, overload

from ._parser import ConfigParser
from ._utils import SimpleTree

if TYPE_CHECKING:
    from ._protocol import Configurable


class _unset: ...


class _status:
    updated: dict[str, Any] = {}
    default: dict[tuple[str, ...], Any] = {}
    frozen: bool = False
    context: list[tuple[str, Any]] = []

    @classmethod
    def get(cls, name: tuple[str, ...]):
        value = SimpleTree.get(cls.updated, name, _unset)
        if value is _unset:
            try:
                value = cls.default[name]
            except KeyError:
                raise AttributeError(f"Config {'.'.join(name)} is not set.")
        return value

    @classmethod
    def set(cls, name: tuple[str, ...], value: Any):
        SimpleTree.set(cls.updated, name, value)


def _freeze_config(config: Configurable):
    config.__config_cache__ = {
        k: deepcopy(_status.get(k)) for k in config.__config_attrs__
    }


@contextmanager
def _freeze(value: bool):
    cache = _status.frozen
    _status.frozen = value
    _status.context.append(("freeze", value))
    yield
    _status.frozen = cache
    _status.context.pop()


@contextmanager
def _override(path_or_dict: tuple, parser: ConfigParser):
    cache = _status.updated
    _status.updated = deepcopy(cache)
    _status.context.append(("override", path_or_dict))
    parser(*path_or_dict, result=_status.updated)
    yield
    _status.updated = cache
    _status.context.pop()


class _pickler:
    def __getstate__(self):
        return _status.updated

    def __setstate__(self, state):
        self.__state = state

    def __call__(self):
        _status.updated = self.__state
        del self.__state


class ConfigManager:
    __parser = ConfigParser(nested=True)

    @classmethod
    def update(
        cls, *path_or_dict: str | dict, parser: ConfigParser = None
    ) -> ContextManager[None]:
        parser = (parser or cls.__parser)(*path_or_dict, result=_status.updated)

    @classmethod
    def override(
        cls, *path_or_dict: str | dict, parser: ConfigParser = None
    ) -> ContextManager[None]:
        return _override(path_or_dict, parser=parser or cls.__parser)

    @overload
    @staticmethod
    def freeze() -> ContextManager[None]: ...
    @overload
    @staticmethod
    def freeze(*configs: Configurable) -> None: ...
    @staticmethod
    def freeze(*configs: Configurable):
        if not configs:
            return _freeze(True)
        for config in configs:
            _freeze_config(config)

    @overload
    @staticmethod
    def unfreeze() -> ContextManager[None]: ...
    @overload
    @staticmethod
    def unfreeze(*configs: Configurable) -> None: ...
    @staticmethod
    def unfreeze(*configs: Configurable):
        if not configs:
            return _freeze(False)
        for config in configs:
            config.__config_cache__ = None

    @staticmethod
    def take_snapshot():
        return deepcopy(_status.updated)

    @staticmethod
    def restore_snapshot(snapshot: dict[str, Any]):
        _status.updated = snapshot

    @staticmethod
    def initializer():
        return _pickler()

    @overload
    @staticmethod
    def inspect() -> ConfigManagerSummary: ...
    @overload
    @staticmethod
    def inspect(config: type[Configurable]) -> ConfigurableClassSummary: ...
    @overload
    @staticmethod
    def inspect(config: Configurable) -> ConfigurableObjectSummary: ...
    @staticmethod
    def inspect(config=None):
        from ._protocol import Configurable, _list_configs, const

        if config is None:
            return {
                "frozen": _status.frozen,
                "context_stack": _status.context,
                "default_configs": _status.default,
                "updated_configs": _status.updated,
            }
        elif isinstance(config, type) and issubclass(config, Configurable):
            return {
                "type": config,
                "namespace": config.__config_namespace__,
                "configs": [
                    {
                        "name": k,
                        "path": v.name,
                        "value": v.value,
                        "is_const": isinstance(v, const),
                    }
                    for k, v in _list_configs(config).items()
                ],
            }
        elif isinstance(config, Configurable):
            return {
                "instance": config,
                "frozen": config.__config_cache__ is not None,
                "configs": [
                    {
                        "name": k,
                        "path": v.name,
                        "value": getattr(config, k),
                        "is_const": isinstance(v, const),
                    }
                    for k, v in _list_configs(type(config)).items()
                ],
            }
        else:
            raise TypeError(f"Cannot inspect non-configurable: {config}")


class ConfigSummary(TypedDict):
    name: str
    path: tuple[str, ...]
    value: Any
    is_const: bool


class ConfigManagerSummary(TypedDict):
    frozen: bool
    context_stack: list[tuple[str, Any]]
    default_configs: dict[tuple[str, ...], Any]
    updated_configs: dict[str, Any]


class ConfigurableClassSummary(TypedDict):
    type: type[Configurable]
    namespace: tuple[str, ...]
    configs: list[ConfigSummary]
    bases: list[ConfigurableClassSummary]


class ConfigurableObjectSummary(TypedDict):
    instance: Configurable
    frozen: bool
    configs: list[ConfigSummary]
