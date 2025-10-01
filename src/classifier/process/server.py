from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import wraps
from multiprocessing import connection as mpc
from queue import PriorityQueue
from threading import Lock, Thread
from types import MethodType
from typing import (
    Annotated,
    Any,
    Callable,
    Concatenate,
    Optional,
    ParamSpec,
    Protocol,
    TypeVar,
    overload,
)

from src.typetools import check_subclass, get_partial_type_hints
from src.classifier.typetools import Method

from ..utils import noop

__all__ = [
    "Server",
    "Client",
    "Proxy",
    "shared",
    "post",
]

_Address = str | tuple[str, int]
_RETRY_PRIORITY = int(1e9 * 1)  # seconds


class _ClientError(Exception):
    __module__ = Exception.__module__


def _close_connection(connection: mpc.Connection):
    try:
        connection.close()
    except Exception:
        pass


class ProxyLike(Protocol):
    def lock(self) -> Lock: ...


@dataclass
class Packet:
    obj: Callable[[], ProxyLike] = None
    func: str = ""
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    wait: bool = False
    lock: bool = False
    retry: int = 0

    def __post_init__(self):
        self._timestamp = time.time_ns()
        self._retried = 0

    def __call__(self, server: Server = None):
        if self.obj is ...:
            if server is not None:
                obj = server
            else:
                raise RuntimeError(f"Unable to identify method {self.func}")
        else:
            obj = self.obj()
        lock = obj.lock() if self.lock else noop
        try:
            with lock:
                p: PostBase = getattr(obj, self.func)
                return p.func(obj, *self.args, **self.kwargs)
        except Exception as e:
            logging.error(e, exc_info=e)

    def __lt__(self, other):
        if isinstance(other, Packet):
            if self.obj is None:
                return False
            elif other.obj is None:
                return True
            return self._priority < other._priority
        return NotImplemented

    @property
    def _priority(self):
        return self._retried * _RETRY_PRIORITY + self._timestamp


@dataclass
class PostBase:
    func: Callable
    wait: bool
    lock: bool
    retry: int

    def __post_init__(self):
        wraps(self.func)(self)
        self.name = self.func.__name__


class Server:
    def __init__(self, address: _Address):
        # address
        self._address = address

        # states
        self._accepting = False
        self._lock = Lock()

        # listener
        self._listener: tuple[mpc.Listener, Thread] = None
        self._runner: Thread = None

        # clients
        self._jobs: PriorityQueue[Packet] = PriorityQueue()
        self._connections: list[mpc.Connection] = []
        self._handlers: list[Thread] = []

    def start(self):
        if (self._listener is None) and (self._address is not None):
            self._listener = (
                mpc.Listener(self._address),
                Thread(target=self._listen, daemon=True),
            )
            self._listener[1].start()
            self._runner = Thread(target=self._run, daemon=True)
            self._runner.start()
            return True
        return False

    def stop(self):
        if self._listener is not None:
            # close listener
            listener, thread = self._listener
            self._listener = None
            if self._accepting:
                try:
                    mpc.Client(self._address).close()
                except ConnectionRefusedError:
                    pass
            listener.close()
            thread.join()
            # close runner
            self._jobs.put(Packet())
            self._runner.join()
            # close connections
            for connection in self._connections:
                _close_connection(connection)
            self._connections.clear()
            for handler in self._handlers:
                handler.join()
            self._handlers.clear()
            return True
        return False

    def _run(self):
        while (packet := self._jobs.get()).obj is not None:
            packet(self)

    def _listen(self):
        while True:
            try:
                self._accepting = True
                connection = self._listener[0].accept()
                self._accepting = False
                self._connections.append(connection)
                if self._listener is not None:
                    handler = Thread(
                        target=self._handle, args=(connection,), daemon=True
                    )
                    self._handlers.append(handler)
                    handler.start()
                else:
                    _close_connection(connection)
                    break
            except OSError:
                break
            finally:
                self._accepting = False

    def _handle(self, connection: mpc.Connection):
        while True:
            try:
                packet: Packet = connection.recv()
                if packet.wait:
                    connection.send(packet(self))
                else:
                    packet._retried = 0
                    self._jobs.put(packet)
            except Exception:
                _close_connection(connection)
                break

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


class Client:
    reconnect_delay = 1  # seconds

    def __init__(self, address: _Address):
        self._address = address
        self._lock = Lock()
        self._jobs: PriorityQueue[Packet] = PriorityQueue()
        self._sender: mpc.Connection = None
        self._thread = Thread(target=self._send_non_blocking, daemon=True)
        self._thread.start()

    def _send(self, packet: Packet):
        with self._lock:
            try:
                if self._sender is None:
                    self._sender = mpc.Client(self._address)
                self._sender.send(packet)
                if packet.wait:
                    return self._sender.recv()
            except Exception as e:
                if self._sender is not None:
                    _close_connection(self._sender)
                    self._sender = None
                if isinstance(e, ConnectionError):
                    raise
                raise _ClientError

    def _send_non_blocking(self):
        while (packet := self._jobs.get()).obj is not None:
            packet._retried += 1
            try:
                self._send(packet)
            except (_ClientError, ConnectionError) as e:
                if isinstance(e, ConnectionError):
                    if self._thread is None:
                        return
                    time.sleep(self.reconnect_delay)
                if packet._retried < packet.retry:
                    self._jobs.put(packet)

    def send(self, packet: Packet):
        if packet.wait:
            return self._send(packet)
        else:
            self._jobs.put(packet)

    def stop(self):
        if self._thread is not None:
            self._jobs.put(Packet())
            thread = self._thread
            self._thread = None
            thread.join()
            return True
        return False


class Proxy(Server):
    _client: Optional[Client] = None
    _address: shared[_Address]
    _init_local: shared[tuple[tuple, dict]]

    def __init_subclass__(cls):
        cls.__shared = []
        for k, v in get_partial_type_hints(cls, include_extras=True).items():
            if check_subclass(v, shared):
                cls.__shared.append(k)

    def __getstate__(self):
        return {k: getattr(self, k) for k in self.__shared if hasattr(self, k)}

    def __setstate__(self, __attrs: dict[str]):
        args, kwargs = __attrs.get("_init_local", ((), {}))
        self.init_local(*args, **kwargs)
        for k, v in __attrs.items():
            setattr(self, k, v)
        self._client = Client(self._address)

    def __init__(self, address: _Address, *args, **kwargs):
        super().__init__(address)
        self._init_local = (args, kwargs)
        self.init_local(*args, **kwargs)

    def init_local(self): ...

    def lock(self):
        return self._lock


_SharedT = TypeVar("_SharedT")
shared = Annotated[_SharedT, "shared"]


class _Post(PostBase):
    def __get__(self, obj: Proxy, cls=None):
        if obj is None:
            return self
        return MethodType(self, obj)

    def __call__(self, obj: Proxy, *args, **kwargs):
        packet = Packet(
            obj=...,
            func=self.name,
            args=args,
            kwargs=kwargs,
            wait=self.wait,
            lock=self.lock,
            retry=self.retry,
        )
        if obj._client is None:
            if self.wait:
                return packet(obj)
            else:
                obj._jobs.put(packet)
        else:
            return obj._client.send(packet)


_PostP = ParamSpec("_PostP")
_PostReturnT = TypeVar("_PostReturnT")


class _PostMeta(type):
    _method: type[PostBase] = None

    def __call__(
        cls,
        func=None,
        *,
        wait_for_return: bool = False,
        acquire_lock: bool = False,
        max_retry: int = None,
    ):
        if func is None:
            return lambda func: cls(
                func,
                wait_for_return=wait_for_return,
                acquire_lock=acquire_lock,
                max_retry=max_retry,
            )
        else:
            return cls._method(
                func=func,
                wait=wait_for_return,
                lock=acquire_lock,
                retry=max_retry,
            )


class post(metaclass=_PostMeta):
    _method = _Post

    @overload
    def __new__(
        cls, func: Callable[Concatenate[Any, _PostP], _PostReturnT], /
    ) -> Method[_PostP, _PostReturnT]: ...
    @overload
    def __new__(
        cls,
        wait_for_return: bool = False,
        acquire_lock: bool = False,
        max_retry: int = None,
    ) -> Callable[
        [Callable[Concatenate[Any, _PostP], _PostReturnT]],
        Method[_PostP, _PostReturnT],
    ]: ...
