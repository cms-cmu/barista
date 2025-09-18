from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import os
import socket
from dataclasses import dataclass
from enum import Flag
from threading import Lock
from typing import Callable, NamedTuple, TypeVar

import fsspec
from classifier.config import setting as cfg

from ..process import pipe_address
from ..process.initializer import status
from ..process.server import Client, Packet, PostBase, Server, post

__all__ = [
    "Monitor",
    "Reporter",
    "MonitorProxy",
    "post_to_monitor",
    "connect_to_monitor",
]

LOCALHOST = "localhost"
CLIENT_NAME_WIDTH = 7


def _get_host():
    return socket.gethostbyname(socket.gethostname())


class Node(NamedTuple):
    ip: str
    pid: int


class _start_reporter:
    def __getstate__(self):
        match _Status.now():
            case _Status.Monitor:
                return Monitor.current()._address
            case _Status.Reporter:
                return Reporter.current()._address

    def __setstate__(self, address: str):
        self._address = address

    def __call__(self):
        Reporter.init(self._address)


class _Status(Flag):
    Unknown = 0b000
    Monitor = 0b100
    Reporter = 0b010
    Fresh = 0b001

    @classmethod
    def now(cls):
        status = _Status.Unknown
        if Monitor.current() is not None:
            status |= cls.Monitor
        if Reporter.current() is not None:
            status |= cls.Reporter
        if status == _Status.Unknown:
            status |= cls.Fresh
        return status


_SingletonT = TypeVar("_SingletonT", bound="_Singleton")


class _Singleton:
    __allowed_process__ = _Status.Fresh | _Status.Monitor

    def __init_subclass__(cls) -> None:
        cls.__instance = None

    def __new__(cls, *_, **__):
        if cls.__instance is not None:
            raise RuntimeError(f"{cls.__name__} is already initialized")
        if _Status.now() not in cls.__allowed_process__:
            name = str(cls.__allowed_process__).removeprefix(_Status.__name__ + ".")
            raise RuntimeError(f"{cls.__name__} must be initialized in {name} process")
        cls.__instance = super().__new__(cls)
        return cls.__instance

    @classmethod
    def init(cls: type[_SingletonT], *args, **kwargs) -> _SingletonT:
        if cls.__instance is None:
            cls.__instance = cls(*args, **kwargs)
        return cls.__instance

    @classmethod
    def current(cls: type[_SingletonT]) -> _SingletonT:
        return cls.__instance

    @classmethod
    def reset(cls):
        cls.__instance = None


@dataclass
class _Packet(Packet):
    def __post_init__(self):
        self.retry = self.retry or cfg.Monitor.retry_max
        super().__post_init__()


class _PostToMonitor(PostBase):
    def __call__(self, cls: type[MonitorProxy], *args, **kwargs):
        packet = _Packet(
            obj=cls.init,
            func=self.name,
            args=args,
            kwargs=kwargs,
            wait=self.wait,
            lock=self.lock,
            retry=self.retry,
        )
        match _Status.now():
            case _Status.Monitor:
                if self.wait:
                    return packet()
                else:
                    return Monitor.current()._jobs.put(packet)
            case _Status.Reporter:
                return Reporter.current().send(packet)


def _post_method(*args, **kwargs):
    return classmethod(_PostToMonitor(*args, **kwargs))


class post_to_monitor(post):
    _method = _post_method


class Monitor(Server, _Singleton):
    __allowed_process__ = _Status.Fresh

    def __init__(self):
        # address
        host, port = cfg.Monitor.address
        if port is None:
            address = pipe_address(host)
            cfg.Monitor.address = address
        else:
            address = (LOCALHOST, port)
            cfg.Monitor.address = f"{LOCALHOST}:{port}"
        super().__init__(address=address)

    def _start(self):
        return super().start()

    def _stop(self):
        return super().stop()

    @classmethod
    def start(cls):
        self = cls.init()
        if self._start():
            status.initializer.add_unique(_start_reporter)

    @classmethod
    def stop(cls):
        cls.init()._stop()

    @classmethod
    def lock(cls):
        return cls.current()._lock


class Reporter(Client, _Singleton):
    __allowed_process__ = _Status.Fresh

    def __init__(self, address: tuple[str, int | None]):
        if address[1] is None:
            address = address[0]
        super().__init__(address)
        Recorder.register(Recorder.name())

    def _stop(self):
        return super().stop()

    @classmethod
    def stop(cls):
        cls.current()._stop()


class _ProxyMeta(type):
    def __getattr__(cls: type[MonitorProxy], name: str):
        return getattr(cls.init(), name)


class MonitorProxy(_Singleton, metaclass=_ProxyMeta):
    _lock: Lock = None

    @classmethod
    def lock(cls):
        if _Status.now() != _Status.Monitor:
            raise RuntimeError("lock can only be accessed in monitor process")
        if cls._lock is None:
            cls._lock = Lock()
        return cls._lock


class Recorder(MonitorProxy):
    _node = (_get_host(), os.getpid())
    _name = f"{_node[0]}/pid-{_node[1]}/{mp.current_process().name}"

    _reporters: dict[str, str]
    _data: list[tuple[str, Callable[[], bytes]]]

    def __init__(self):
        self._reporters = {self._name: "main"}
        self._data = [(cfg.Monitor.file, Recorder.serialize)]

    @classmethod
    def __register(cls, name: str):
        with cls.lock():
            index = f"#{len(cls._reporters)}"
            cls._reporters[name] = index
            return index

    @post_to_monitor
    def register(self, name: str):
        index = self.__register(name)
        if cfg.Monitor.log_show_connection:
            logging.info(
                f'"{name}" is registered as [repr.number]\[{index}][/repr.number]'
            )
        return index

    @classmethod
    def name(cls):
        return cls._name

    @classmethod
    def node(cls) -> Node:
        return cls._node

    @classmethod
    def registered(cls, name: str):
        index = cls._reporters.get(name)
        if index is None:
            index = cls.__register(name)
        return index

    @classmethod
    def to_dump(cls, file: str, func: Callable[[], bytes]):
        cls._data.append((file, func))

    @classmethod
    def serialize(cls):
        import json

        return json.dumps(cls._reporters, indent=4).encode()

    @classmethod
    def dump(cls):
        if (_Status.now() == _Status.Monitor) and (not cfg.IO.monitor.is_null):
            for file, func in cls._data:
                if file is not None:
                    with fsspec.open(cfg.IO.monitor / file, "wb") as f:
                        f.write(func())


def connect_to_monitor():
    Reporter.init(cfg.Monitor.address)
    status.initializer.add_unique(_start_reporter)
    atexit.register(Reporter.current().stop)


def wait_for_monitor():
    if _Status.now() is _Status.Monitor:
        Monitor.current().stop()


def full_address():
    address, port = cfg.Monitor.address
    if port is None:
        return address
    else:
        local = _get_host()
        if address == local or address == LOCALHOST:
            return f"{local}:{port}/{LOCALHOST}:{port}"
        return f"{address}:{port}"
