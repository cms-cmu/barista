from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import cached_property, reduce
from itertools import chain
from typing import TYPE_CHECKING, Iterable

from src.utils import unique
from classifier.config.main._utils import progress_advance
from classifier.task import ArgParser, Dataset, converter, parse

from ..setting import IO as IOSetting

if TYPE_CHECKING:
    import pandas as pd
    from src.data_formats.root import Chunk, Friend
    from classifier.df.io import FromRoot, ToTensor
    from classifier.df.tools import DFProcessor


class LoadRoot(ABC, Dataset):
    trainable: bool = False
    evaluable: bool = False

    argparser = ArgParser()
    argparser.add_argument(
        "--files",
        action="extend",
        nargs="+",
        default=[],
        help="the paths to the ROOT files",
    )
    argparser.add_argument(
        "--filelists",
        action="extend",
        nargs="+",
        default=[],
        help=f"the paths to the filelists {parse.EMBED}",
    )
    argparser.add_argument(
        "--friends",
        action="extend",
        nargs="+",
        default=[],
        help="the paths to the json files with friend tree metadata",
    )
    argparser.add_argument(
        "--max-workers",
        type=converter.int_pos,
        default=10,
        help="the maximum number of workers to fetch metadata and load training set",
    )
    argparser.add_argument(
        "--tree",
        default="Events",
        help="the name of the TTree",
    )
    argparser.add_argument(
        "--train-chunksize",
        type=converter.int_pos,
        default=1_000_000,
        help="the size of chunk to load training set",
        condition="trainable",
    )
    argparser.add_argument(
        "--eval-base",
        default="chunks",
        help="the base path to store the evaluation results",
        condition="evaluable",
    )
    argparser.add_argument(
        "--eval-naming",
        default=...,
        help="the rule to name friend tree files for evaluation",
        condition="evaluable",
    )

    def __init__(self):
        from classifier.df.io import ToTensor

        self._to_tensor = ToTensor()
        self._preprocessors: list[DFProcessor] = []
        self._postprocessors: list[DFProcessor] = []

    @property
    def to_tensor(self):
        return self._to_tensor

    @property
    def preprocessors(self):
        return self._preprocessors

    @property
    def postprocessors(self):
        return self._postprocessors

    def _parse_files(self, files: list[str], filelists: list[str]) -> list[str]:
        return unique(
            reduce(
                list.__add__,
                (parse.mapping(f, "file") or [] for f in filelists),
                files.copy(),
            )
        )

    def _parse_friends(self, friends: list[str]) -> list[Friend]:
        from src.data_formats.root import Friend

        return [Friend.from_json(parse.mapping(f, "file")) for f in friends]

    def _from_root(self):
        yield self.from_root(), self.files

    def train(self):
        if not self.trainable:
            raise NotImplementedError(
                f"{type(self).__name__} does not support training"
            )
        loader = _load_root(
            *self._from_root(),
            max_workers=self.opts.max_workers,
            chunksize=self.opts.train_chunksize,
            tree=self.opts.tree,
        )
        loader.to_tensor = self.to_tensor
        loader.postprocessors = self.postprocessors
        return [loader]

    def evaluate(self):
        if not self.evaluable:
            raise NotImplementedError(
                f"{type(self).__name__} does not support evaluation"
            )
        from concurrent.futures import ProcessPoolExecutor

        from src.data_formats.root import Chunk
        from classifier.monitor.progress import Progress
        from classifier.process import pool, status
        from classifier.root.dataset import FriendTreeEvalDataset

        from_roots = [*self._from_root()]
        with ProcessPoolExecutor(
            max_workers=self.opts.max_workers,
            mp_context=status.context,
            initializer=status.initializer,
        ) as executor:
            with Progress.new(
                total=sum(map(lambda x: len(x[1]), from_roots)),
                msg=("files", "Fetching metadata"),
            ) as progress:
                groups = [
                    (
                        from_root,
                        pool.map_async(
                            executor,
                            _fetch(tree=self.opts.tree),
                            files,
                            callbacks=[lambda _: progress_advance(progress)],
                        ),
                    )
                    for from_root, files in from_roots
                ]
                groups = [(from_root, [*files]) for from_root, files in groups]
                yield FriendTreeEvalDataset(
                    chunks=Chunk.common(*chain(*map(lambda x: x[1], groups))),
                    load_method=_eval_root(
                        *groups,
                        to_tensor=self.to_tensor,
                        postprocessors=self.postprocessors,
                    ),
                    dump_base_path=IOSetting.output / self.opts.eval_base,
                    dump_naming=self.opts.eval_naming,
                )

    @cached_property
    def files(self) -> list[str]:
        return self._parse_files(self.opts.files, self.opts.filelists)

    @cached_property
    def friends(self) -> list[Friend]:
        return self._parse_friends(self.opts.friends)

    @abstractmethod
    def from_root(self) -> FromRoot: ...


class LoadGroupedRoot(LoadRoot):
    argparser = ArgParser()
    argparser.add_argument(
        "--files",
        action="append",
        nargs="+",
        metavar=("GROUPS", "PATHS"),
        default=[],
        help="comma-separated groups and paths to the ROOT file",
    )
    argparser.add_argument(
        "--filelists",
        action="append",
        nargs="+",
        metavar=("GROUPS", "PATHS"),
        default=[],
        help=f"comma-separated groups and paths to the filelist {parse.EMBED}",
    )
    argparser.add_argument(
        "--friends",
        action="append",
        nargs="+",
        metavar=("GROUPS", "PATHS"),
        default=[],
        help="comma-separated groups and paths to the json file with the friend tree metadata",
    )

    def _from_root(self):
        files = self.files
        for k in files:
            yield self.from_root(k), files[k]

    @cached_property
    def files(self):
        files = parse.grouped_mappings(self.opts.files, ",")
        filelists = parse.grouped_mappings(self.opts.filelists, ",")
        return {
            k: self._parse_files(files.get(k, []), filelists.get(k, []))
            for k in set(files).union(filelists)
        }

    @cached_property
    def friends(self):
        return {
            k: self._parse_friends(v)
            for k, v in parse.grouped_mappings(self.opts.friends, ",").items()
        }

    @abstractmethod
    def from_root(self, groups: frozenset[str]) -> FromRoot: ...


class _fetch:
    def __init__(self, tree: str):
        self._tree = tree

    def __call__(self, path: str | Chunk):
        from src.data_formats.root import Chunk

        if isinstance(path, Chunk):
            return path
        chunk = Chunk(source=path, name=self._tree, fetch=True)
        return chunk


class _load_root:
    to_tensor: ToTensor
    postprocessors: list[DFProcessor]

    def __init__(
        self,
        *from_root: tuple[FromRoot, list[str]],
        max_workers: int,
        chunksize: int,
        tree: str,
    ):
        self._from_root = from_root
        self._max_workers = max_workers
        self._chunksize = chunksize
        self._tree = tree

    def __call__(self):
        data = self.load()
        for p in self.postprocessors:
            data = p(data)
        return self.to_tensor.tensor(data)

    def load(self) -> pd.DataFrame:
        from concurrent.futures import ProcessPoolExecutor

        import pandas as pd
        from src.data_formats.root import Chunk
        from classifier.monitor.progress import Progress
        from classifier.process import pool, status

        with ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=status.context,
            initializer=status.initializer,
        ) as executor:
            with Progress.new(
                total=sum(map(lambda x: len(x[1]), self._from_root)),
                msg=("files", "Fetching metadata"),
            ) as progress:
                chunks = [
                    list(
                        pool.map_async(
                            executor,
                            _fetch(tree=self._tree),
                            files,
                            callbacks=[lambda _: progress_advance(progress)],
                        )
                    )
                    for _, files in self._from_root
                ]
            with Progress.new(
                total=sum(map(len, chain(*chunks))),
                msg=("events", "Loading"),
            ) as progress:
                dfs = [
                    *chain(
                        *(
                            pool.map_async(
                                executor,
                                self._from_root[i][0],
                                Chunk.balance(
                                    self._chunksize,
                                    *chunks[i],
                                    common_branches=True,
                                ),
                                callbacks=[
                                    lambda x: progress_advance(progress, len(x))
                                ],
                            )
                            for i in range(len(chunks))
                        )
                    ),
                ]
        df = pd.concat(
            filter(lambda x: x is not None, dfs), ignore_index=True, copy=False
        )
        logging.info(
            "Loaded <DataFrame>:",
            f"entries: {len(df)}",
            f"columns: {sorted(df.columns)}",
        )
        return df


class _eval_root:
    def __init__(
        self,
        *from_root: tuple[FromRoot, Iterable[Chunk]],
        to_tensor: ToTensor,
        postprocessors: list[DFProcessor],
    ):
        self._from_roots: list[FromRoot] = []
        self._lookup: dict[Chunk, int] = {}
        for from_root, chunks in from_root:
            idx = len(self._from_roots)
            self._from_roots.append(from_root)
            for chunk in chunks:
                self._lookup[chunk.key()] = idx
        self._postprocessors = postprocessors
        self._to_tensor = to_tensor

    def __call__(self, chunk: Chunk):
        data = self._from_roots[self._lookup[chunk]](chunk)
        for p in self._postprocessors:
            data = p(data)
        return self._to_tensor.tensor(data)
