from concurrent.futures import Executor, Future
from queue import Queue
from typing import Callable, Generator, Generic, Iterable, Optional, ParamSpec, TypeVar

_SubmitT = TypeVar("_SubmitT")
_SubmitP = ParamSpec("_SubmitP")


class _FuturePool(Generic[_SubmitT]):
    def __init__(self, timeout: Optional[float], count: int):
        self._timeout = timeout
        self._count = count
        self._tasks = Queue()

    def __len__(self):
        return self._count

    def __iter__(self):
        for _ in range(self._count):
            yield self._tasks.get()
        self._count = 0
        self._tasks = None

    def _result(self, task: Future[_SubmitT]):
        self._tasks.put(task.result(timeout=self._timeout))


class _OrderedFuturePool(Generic[_SubmitT]):
    def __init__(self, timeout: Optional[float]):
        self._timeout = timeout
        self._tasks: list[Future[_SubmitT]] = []

    def __len__(self):
        if self._tasks is None:
            return 0
        return len(self._tasks)

    def __iter__(self):
        for task in self._tasks:
            yield task.result(timeout=self._timeout)
        self._tasks = None


def map_async(
    e: Executor,
    fn: Callable[_SubmitP, _SubmitT],
    *args: Iterable,
    callbacks: Iterable[Callable[_SubmitP, Callable[[Future[_SubmitT]], None]]] = (),
    timeout: Optional[float] = None,
    preserve_order: bool = False,
) -> Generator[_SubmitT, None, None]:
    args = [*zip(*args)]
    callbacks = [*callbacks]
    if preserve_order:
        results = _OrderedFuturePool(timeout)
    else:
        results = _FuturePool(timeout, len(args))
        callbacks.append(lambda *_: results._result)
    for arg in args:
        task = e.submit(fn, *arg)
        for callback in callbacks:
            task.add_done_callback(callback(*arg))
        if preserve_order:
            results._tasks.append(task)
    return results


class CallbackExecutor(Generic[_SubmitT, _SubmitP]):
    def __init__(
        self,
        executor: Executor,
        *callbacks: Callable[_SubmitP, Callable[[Future[_SubmitT]], None]],
    ):
        self._executor = executor
        self._callbacks = callbacks

    def map(self, fn: Callable[_SubmitP, _SubmitT], *args):
        return map_async(self._executor, fn, *args, callbacks=self._callbacks)

    def submit(self, fn: Callable[_SubmitP, _SubmitT], *args):
        future = self._executor.submit(fn, *args)
        for callback in self._callbacks:
            future.add_done_callback(callback(*args))
        return future
