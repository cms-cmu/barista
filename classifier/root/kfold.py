from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterable, Protocol

import numpy as np
from src.data_formats.root import Chain, Chunk, Friend

from ..monitor.progress import Progress, ProgressTracker
from ..process import status
from ..process.pool import CallbackExecutor

if TYPE_CHECKING:
    from src.data_formats.root.chain import NameMapping
    from src.storage.eos import PathLike


class MergeMethod(Protocol):
    def __call__(self, branch: str, array: np.ndarray) -> dict[str, np.ndarray]: ...


def MergeMean(branch: str, array: np.ndarray):
    return {branch: np.nanmean(array, axis=1)}


class MergeStd:
    def __init__(self, branches: Iterable[str], suffix: str = "_std"):
        self.branches = set(branches)
        self.suffix = suffix

    def __call__(self, branch: str, array: np.ndarray):
        merged = {}
        if branch in self.branches:
            merged[branch + self.suffix] = np.nanstd(array, axis=1)
        return merged


@dataclass(kw_only=True)
class _merge_worker:
    chunk: Chunk = None
    chain: Chain
    methods: Iterable[MergeMethod]
    name: str
    base_path: PathLike
    naming: str | NameMapping

    def new(self, chunk: Chunk):
        return replace(self, chunk=chunk)

    def __call__(self) -> Friend:
        chain = self.chain.copy().add_chunk(self.chunk)
        data = chain.concat(library="pd", friend_only=True)
        merged: dict[str, np.ndarray] = {}
        for k in data.columns.get_level_values(0):
            array = data.loc[:, k]
            for method in self.methods:
                merged |= method(k, array)
        del data
        with Friend(name=self.name).auto_dump(
            base_path=self.base_path, naming=self.naming
        ) as friend:
            friend.add(self.chunk, merged)
        return friend


@dataclass
class _update_friend:
    friend: Friend = None

    def __call__(self, friend: Future[Friend]):
        friend = friend.result()
        if self.friend is None:
            self.friend = friend
        else:
            self.friend += friend


@dataclass
class _update_progress:
    progress: ProgressTracker
    step: int

    def __call__(self, *_):
        self.progress.advance(self.step)


@dataclass
class _optimize_progress:
    progress: ProgressTracker

    def __call__(self, result: Future[list[Chunk]]):
        self.progress.advance(sum(len(chunk) for chunk in result.result()))


class merge_kfolds:
    @staticmethod
    def _rename_column(friend: str, branch: str):
        return (branch, friend)

    def __init__(
        self,
        *friends: Friend,
        methods: Iterable[MergeMethod] = (MergeMean,),
        step: int,
        workers: int,
        friend_name: str,
        dump_base_path: PathLike,
        dump_naming: str | NameMapping = ...,
        clean: bool = False,
        optimize: int = None,
    ):
        self._friends = friends
        self._job = _merge_worker(
            chain=Chain().add_friend(*friends, renaming=self._rename_column),
            methods=methods,
            name=friend_name,
            base_path=dump_base_path,
            naming=dump_naming,
        )
        self._step = step
        self._workers = workers
        self._clean = clean
        self._optimize = optimize

    def __call__(self):
        # assume all friend trees have the same structure
        targets = [*self._friends[0].targets]
        n_entries = self._friends[0].n_entries
        result = _update_friend()
        with (
            ProcessPoolExecutor(
                max_workers=self._workers,
                mp_context=status.context,
                initializer=status.initializer,
            ) as pool,
        ):
            with Progress.new(
                total=n_entries,
                msg=("entries", "Merging", f"{len(self._friends)}-folds"),
            ) as progress:
                jobs = []
                for chunk in Chunk.balance(self._step, *targets):
                    job = pool.submit(self._job.new(chunk))
                    job.add_done_callback(result)
                    job.add_done_callback(_update_progress(progress, len(chunk)))
                    jobs.append(job)
                wait(jobs)
            if self._clean:
                with Progress.new(
                    total=sum(friend.n_fragments for friend in self._friends),
                    msg=("files", "Cleaning fragments"),
                ) as progress:
                    for friend in self._friends:
                        friend.reset(
                            confirm=False,
                            executor=CallbackExecutor(
                                pool, lambda *_: _update_progress(progress, 1)
                            ),
                        )
            if self._optimize is not None:
                with Progress.new(
                    total=n_entries,
                    msg=("entries", "Optimizing friend trees"),
                ) as progress:
                    result.friend = result.friend.merge(
                        step=self._optimize,
                        base_path=self._job.base_path,
                        executor=CallbackExecutor(
                            pool, lambda *_: _optimize_progress(progress)
                        ),
                    ).result()
        output = {"merged": result.friend}
        if not self._clean:
            output["original"] = [*self._friends]
        return output
