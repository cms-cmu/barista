from _thread import LockType
from enum import Enum
from threading import Lock
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Mapping,
    ParamSpec,
    Protocol,
    TypeVar,
    runtime_checkable,
)
from uuid import uuid4

_MethodP = ParamSpec("_MethodP")
_MethodReturnT = TypeVar("_MethodReturnT")


class Method(Protocol, Generic[_MethodP, _MethodReturnT]):
    def __get__(
        self, instance: Any, owner: type | None = None
    ) -> Callable[_MethodP, _MethodReturnT]: ...
    def __call__(
        self_, self: Any, *args: _MethodP.args, **kwargs: _MethodP.kwargs
    ) -> _MethodReturnT: ...


class WithUUID:
    def __init__(self):
        super().__init__()
        self.uuid = uuid4()


class PicklableLock:
    def __init__(self):
        super().__init__()
        self.lock = Lock()

    def __copy__(self):
        new = self.__new__(self.__class__)
        new.__dict__ = self.__dict__.copy()
        return new

    def __getstate__(self):
        return self.__dict__ | {"lock": isinstance(self.lock, LockType)}

    def __setstate__(self, state):
        self.__dict__ = state
        self.lock = Lock() if self.lock else None


@runtime_checkable
class _MappingLike(Protocol):
    def __iter__(self): ...
    def __getitem__(self, __key): ...
    def __setitem__(self, __key, __value): ...
    def __delitem__(self, __key): ...
    def __contains__(self, __key): ...


class dict_proxy:
    def __new__(cls, obj):
        if isinstance(obj, _MappingLike):
            return super().__new__(_dictlike)
        return super().__new__(_classlike)

    def __init__(self, obj):
        self._obj = obj

    def items(self):
        for k in self:
            yield k, self[k]

    def update(self, *mappings: Mapping):
        for mapping in mappings:
            proxy = dict_proxy(mapping)
            for k in proxy:
                self[k] = proxy[k]
        return self


class _dictlike(dict_proxy):
    _obj: Mapping

    def __iter__(self):
        yield from self._obj

    def __getitem__(self, __key):
        return self._obj[__key]

    def __setitem__(self, __key, __value):
        self._obj[__key] = __value

    def __delitem__(self, __key):
        del self._obj[__key]

    def __contains__(self, __key):
        return __key in self._obj


class _classlike(dict_proxy):
    def __iter__(self):
        yield from dir(self._obj)

    def __getitem__(self, __key):
        return getattr(self._obj, __key)

    def __setitem__(self, __key, __value):
        setattr(self._obj, __key, __value)

    def __delitem__(self, __key):
        delattr(self._obj, __key)

    def __contains__(self, __key):
        return hasattr(self._obj, __key)


def enum_dict(enum: type[Enum]):
    return {i.name: i.value for i in enum}


@runtime_checkable
class FilenameProtocol(Protocol):
    def __filename__(self) -> str: ...


def filename(obj: Any) -> str:
    if isinstance(obj, FilenameProtocol):
        return obj.__filename__()
    elif isinstance(obj, Mapping):
        name = []
        for k, v in obj.items():
            name.append(f"{filename(k)}_{filename(v)}")
        return "__".join(name)
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, Iterable):
        return "-".join(map(filename, obj))
    else:
        return repr(obj)


def nameof(obj: Any) -> str:
    try:
        name = obj.__name__
    except Exception:
        name = type(obj).__name__
    return f"<{name}>"


def new_TypedDict(typed_dict: type, *args, **kwargs):
    obj = typed_dict(*args, **kwargs)
    defaults = vars(typed_dict)
    for k in typed_dict.__annotations__:
        if k not in obj and k in defaults:
            obj[k] = defaults[k]
    return obj
