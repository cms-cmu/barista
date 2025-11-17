from __future__ import annotations

import logging
from concurrent.futures import Executor
from functools import partial
from typing import Iterable, Optional
from uuid import UUID

from ...math_tools.utils import balance_split
from ...storage.eos import EOS, PathLike
from ...typetools import check_type
from ...utils import map_executor


class _ChunkMeta(type):
    def _get(self, attr):
        if getattr(self, attr) is ...:
            self._fetch()
        return getattr(self, attr)

    def __new__(cls, name, bases, dic):
        for attr in ("branches", "num_entries", "uuid"):
            dic[attr] = property(partial(cls._get, attr=f"_{attr}"))
        return super().__new__(cls, name, bases, dic)


class Chunk(metaclass=_ChunkMeta):
    """
    A chunk of :class:`TTree` stored in a ROOT file.

    Parameters
    ----------
    source : PathLike or tuple[PathLike, ~uuid.UUID]
        Path to ROOT file with optional UUID
    name : str, optional, default='Events'
        Name of :class:`TTree`.
    branches : ~typing.Iterable[str], optional
        Name of branches. If not given, read from ``source``.
    num_entries : int, optional
        Number of entries. If not given, read from ``source``.
    entry_start : int, optional
        Start entry. If not given, set to ``0``.
    entry_stop : int, optional
        Stop entry. If not given, set to ``num_entries``.
    fetch : bool, optional, default=False
        Fetch missing metadata from ``source`` immediately after initialization.

    Notes
    -----
    The following special methods are implemented:

    - :meth:`__hash__`
    - :meth:`__eq__`
    - :meth:`__len__`
    - :meth:`__repr__`
    """

    path: EOS
    """~heptools.system.eos.EOS : Path to ROOT file."""
    uuid: UUID
    """~uuid.UUID : UUID of ROOT file."""
    name: str
    """str : Name of :class:`TTree`."""
    branches: frozenset[str]
    """frozenset[str] : Name of branches."""
    num_entries: int
    """int : Number of entries."""

    @property
    def entry_start(self):
        """int : Start entry."""
        if self._entry_start is ...:
            return 0
        return self._entry_start

    @property
    def entry_stop(self):
        """int : Stop entry."""
        if self._entry_stop is ...:
            return self.num_entries
        return self._entry_stop

    @property
    def offset(self):
        """int : Equal to ``entry_start``."""
        return self.entry_start

    def __init__(
        self,
        source: PathLike | tuple[PathLike, UUID] = None,
        name: str = "Events",
        branches: Iterable[str] = ...,
        num_entries: int = ...,
        entry_start: int = ...,
        entry_stop: int = ...,
        fetch: bool = False,
    ):
        if isinstance(branches, Iterable):
            branches = frozenset(branches)

        self.name = name
        self._entry_start = entry_start
        self._entry_stop = entry_stop

        self._uuid = ...
        self._branches = branches
        self._num_entries = num_entries

        if check_type(source, PathLike):
            self.path = EOS(source)
        elif check_type(source, tuple[PathLike, UUID]):
            self.path = EOS(source[0])
            self._uuid = source[1]
        elif source is None:
            self.path = source

        if fetch:
            self._fetch()

    @classmethod
    def _ignore(cls, value):
        return value is ... or value is None

    def integrity(self):
        """
        Check and report the following:

        - :data:`path` not exists
        - :data:`uuid` different from file
        - :data:`num_entries` different from file
        - :data:`branches` not in file
        - :data:`entry_start` out of range
        - :data:`entry_stop` out of range

        Returns
        -------
        Chunk or None
            A deep copy of ``self`` with corrected metadata. If file not exists, return ``None``.
        """
        chunk_name = f'chunk  "{self.path}"\n    '
        if not self.path.exists:
            logging.error(f"{chunk_name}file not exists")
            return None
        else:
            reloaded = Chunk(
                source=self.path,
                name=self.name,
                entry_start=self._entry_start,
                entry_stop=self._entry_stop,
                fetch=True,
            )
            if not self._ignore(self._uuid) and self._uuid != reloaded.uuid:
                logging.error(
                    f"{chunk_name}UUID {self._uuid}(stored) != {reloaded.uuid}(file)"
                )
            if (
                not self._ignore(self._num_entries)
                and self._num_entries != reloaded.num_entries
            ):
                logging.error(
                    f"{chunk_name}number of entries {self._num_entries}(stored) != {reloaded.num_entries}(file)"
                )
            if not self._ignore(self._branches):
                diff = self._branches - reloaded.branches
                if diff:
                    logging.error(f"{chunk_name}branches {diff} not in file")
            out_of_range = False
            if not self._ignore(self._entry_start):
                out_of_range |= (
                    self._entry_start < 0 or self._entry_start >= reloaded.num_entries
                )
            else:
                reloaded._entry_start = 0
            if not self._ignore(self._entry_stop):
                out_of_range |= (
                    self._entry_stop <= reloaded.entry_start
                    or self._entry_stop > reloaded.num_entries
                )
            else:
                reloaded._entry_stop = reloaded.num_entries
            if out_of_range:
                logging.warning(
                    f"{chunk_name}invalid entry range [0,{reloaded.num_entries}) -> [{reloaded.entry_start},{reloaded.entry_stop})"
                )
            return reloaded

    def _fetch(self):
        if any(v is ... for v in (self._branches, self._num_entries, self._uuid)):
            import uproot

            with uproot.open(self.path) as file:
                self._fetch_file(file)
        return self

    def _fetch_file(self, file):
        tree = file[self.name]
        if self._branches is ...:
            self._branches = frozenset(tree.keys())
        if self._num_entries is ...:
            self._num_entries = tree.num_entries
        if self._uuid is ...:
            self._uuid = file.file.uuid

    def __hash__(self):
        return hash((self.uuid, self.name))

    def __eq__(self, other):
        if isinstance(other, Chunk):
            return (self.uuid, self.name) == (other.uuid, other.name)
        return NotImplemented

    def __len__(self):
        return self.entry_stop - self.entry_start

    def __repr__(self):
        text = f"TTree:{self.path}"
        if not self._ignore(self._uuid):
            text += f"({self._uuid})"
        text += f":{self.name}"
        if not self._ignore(self._num_entries):
            text += f"[0,{self._num_entries})"
        if not self._ignore(self._entry_start) and not self._ignore(self._entry_stop):
            text += f" -> [{self._entry_start},{self._entry_stop})"
        return text

    def deepcopy(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs : dict, optional
            Override ``entry_start``, ``entry_stop`` or ``branches``.

        Returns
        -------
        Chunk
            A deep copy of ``self``.
        """
        path = self.path if self._uuid is ... else (self.path, self._uuid)
        return Chunk(
            source=path,
            name=self.name,
            num_entries=self._num_entries,
            branches=kwargs.get("branches", self._branches),
            entry_start=kwargs.get("entry_start", self._entry_start),
            entry_stop=kwargs.get("entry_stop", self._entry_stop),
        )

    def key(self):
        """
        Returns
        -------
        Chunk
            A deep copy of ``self`` that only keep the properties used by ``__hash__``.
        """
        return Chunk(
            source=(self.path, self.uuid),
            name=self.name,
            num_entries=None,
            branches=None,
            entry_start=None,
            entry_stop=None,
        )

    def slice(self, start: int, stop: int):
        """
        Parameters
        ----------
        start : int
            Entry start.
        stop : int
            Entry stop.

        Returns
        -------
        Chunk
            A sliced :meth:`deepcopy` of ``self`` from ``start`` + :data:`offset` to ``stop`` + :data:`offset`.
        """
        start += self.offset or 0
        stop += self.offset or 0
        chunk = self.deepcopy(entry_start=start, entry_stop=stop)
        return chunk

    @classmethod
    def from_path(cls, *paths: tuple[str, str], executor: Optional[Executor] = None):
        """
        Create :class:`Chunk` from ``paths`` and fetch metadata in parallel.

        Parameters
        ----------
        paths : tuple[tuple[str, str]
            Path to ROOT file and name of :class:`TTree`.
        executor: ~concurrent.futures.Executor, optional
            An executor with at least the :meth:`~concurrent.futures.Executor.map` method implemented. If not provided, the tasks will run sequentially in the current thread.

        Returns
        -------
        list[Chunk]
            List of chunks from ``paths``.
        """
        chunks = [Chunk(path, name) for path, name in paths]
        chunks = (map_executor if executor is None else executor.map)(
            Chunk._fetch, chunks
        )
        return [*chunks]

    @classmethod
    def common(
        cls,
        *chunks: Chunk,
    ) -> list[Chunk]:
        """
        Find common branches of ``chunks``.

        Parameters
        ----------
        chunks : tuple[Chunk]
            Chunks to select common branches.

        Returns
        -------
        list[Chunk]
            Deep copies of ``chunks`` with only common branches.
        """
        chunks = [chunk.deepcopy() for chunk in chunks]
        branches = set(chunk.branches for chunk in chunks)
        if len(branches) == 1:
            return chunks
        else:
            common = frozenset.intersection(*branches)
            for chunk in chunks:
                chunk._branches = common
            return chunks

    @classmethod
    def partition(
        cls,
        size: int,
        *chunks: Chunk,
        common_branches: bool = False,
    ):
        """
        Partition ``chunks`` into groups. The sum of entries in each group is equal to ``size`` except for the last one. The order of chunks is preserved.

        Parameters
        ----------
        size : int
            Size of each group.
        chunks : tuple[Chunk]
            Chunks to partition.
        common_branches : bool, optional, default=False
            If ``True``, only common branches of all chunks are kept.

        Yields
        ------
        list[Chunk]
            A group of chunks with total entries equal to ``size``.
        """
        i, start, remain = 0, 0, size
        group: list[Chunk] = []
        if common_branches:
            chunks = cls.common(*chunks)
        while i < len(chunks):
            chunk = min(remain, len(chunks[i]) - start)
            group.append(chunks[i].slice(start, start + chunk))
            remain -= chunk
            start += chunk
            if remain == 0:
                yield group
                group = []
                remain = size
            if start == len(chunks[i]):
                i += 1
                start = 0
        if group:
            yield group

    @classmethod
    def balance(
        cls,
        size: int,
        *chunks: Chunk,
        common_branches: bool = False,
    ):
        """
        Split ``chunks`` into smaller pieces with ``size`` entries in each. If not possible, will try to find another size minimizing the average deviation.

        Parameters
        ----------
        size : int
            Target number of entries in each chunk.
        chunks : tuple[Chunk]
            Chunks to balance.
        common_branches : bool, optional, default=False
            If ``True``, only common branches of all chunks are kept.

        Yields
        -------
        list[Chunk]
            Resized chunks with about ``size`` entries in each.
        """
        if common_branches:
            chunks = cls.common(*chunks)
        for chunk in chunks:
            total = len(chunk)
            start = 0
            for step in balance_split(total, size):
                yield chunk.slice(start, start + step)
                start += step

    def to_json(self):
        """
        Convert ``self`` to JSON data.

        Returns
        -------
        dict
            JSON data.
        """
        json_dict = {
            "path": str(self.path),
            "name": self.name,
        }
        json_dict["uuid"] = None if self._uuid is ... else str(self._uuid)
        if self._branches is not None:
            json_dict["branches"] = (
                None if self._branches is ... else list(self._branches)
            )
        if self._num_entries is not None:
            json_dict["num_entries"] = (
                None if self._num_entries is ... else self._num_entries
            )
        if self._entry_start is not None:
            json_dict["entry_start"] = (
                None if self._entry_start is ... else self._entry_start
            )
        if self._entry_stop is not None:
            json_dict["entry_stop"] = (
                None if self._entry_stop is ... else self._entry_stop
            )
        return json_dict

    @classmethod
    def from_json(cls, data: dict):
        """
        Create :class:`Chunk` from JSON data.

        Parameters
        ----------
        data : dict
            JSON data.

        Returns
        -------
        Chunk
            A :class:`Chunk` object from JSON data.
        """
        kwargs = {
            "name": data["name"],
        }
        uuid = data["uuid"]
        if uuid is not None:
            kwargs["source"] = (data["path"], UUID(uuid))
        else:
            kwargs["source"] = data["path"]
        for key in ("branches", "num_entries", "entry_start", "entry_stop"):
            if key in data:
                value = data[key]
                if value is None:
                    continue
            else:
                value = None
            kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_coffea_events(cls, events):
        """
        Create :class:`Chunk` when using ``coffea<=0.7.22``.

        Parameters
        ----------
        events
            Events generated by :class:`coffea.processor.Runner`.

        Returns
        -------
        Chunk
            Chunk from ``events``.
        """
        metadata = events.metadata
        return cls(
            source=(metadata["filename"], UUID(metadata["fileuuid"])),
            name=metadata["treename"],
            entry_start=metadata["entrystart"],
            entry_stop=metadata["entrystop"],
        )

    @classmethod
    def from_coffea_datasets(cls, datasets: dict) -> dict[str, list[Chunk]]:
        """
        Create :class:`Chunk` when using ``coffea>=2023.12.0``.

        Parameters
        ----------
        datasets
            Datasets generated by :func:`coffea.dataset_tools.preprocess`.

        Returns
        -------
        dict[str, list[Chunk]]
            A mapping from dataset names to lists of chunks using the partitions from ``datasets``.
        """
        import uproot

        partitions: dict[str, list[Chunk]] = {}
        for dataset, files in datasets.items():
            files = files["files"]
            partitions[dataset] = [
                cls(
                    source=(path, UUID(files[path]["uuid"])),
                    name=name,
                    entry_start=start,
                    entry_stop=stop,
                )
                for path, name, steps in uproot._util.regularize_files(
                    files, steps_allowed=True
                )
                for start, stop in steps
            ]
        return partitions
