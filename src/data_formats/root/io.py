"""
A wrapper for ROOT file I/O :func:`uproot.reading.open`, :func:`uproot._dask.dask` and :func:`uproot.writing.writable.recreate`.

.. note::
    Readers will use the following default options for :func:`uproot.open`:

    .. code-block:: python

        object_cache = None
        array_cache = None

    for :func:`uproot.dask`:

    .. code-block:: python

        open_files = False

    and for both:

    .. code-block:: python

        timeout = 180

.. warning::
    Writers will always overwrite the output file if it exists.

.. todo::
    Use :func:`dask_awkward.new_scalar_object` to return object.

"""

from __future__ import annotations

import logging
from numbers import Number
from typing import TYPE_CHECKING, Callable, Generator, Literal, TypedDict, overload

import uproot
from packaging.version import Version

from ...storage.eos import EOS, PathLike
from ._backend import (
    concat_record,
    len_record,
    materialize_record,
    record_backend,
    slice_record,
)
from .chunk import Chunk

if TYPE_CHECKING:
    import awkward as ak
    import dask.array as da
    import dask_awkward as dak
    import numpy as np
    import numpy.typing as npt
    import pandas as pd

if TYPE_CHECKING:
    RecordLike = ak.Array | pd.DataFrame | dict[str, np.ndarray]
    """
    ak.Array, pandas.DataFrame, dict[str, numpy.ndarray]: A mapping from string to array-like object. All arrays must have same lengths.
    """

if TYPE_CHECKING:
    DelayedRecordLike = dak.Array | dict[str, da.Array]
    """
    dask_awkward.Array, dict[str, dask.array.Array]: A mapping from string to array-like delayed object.  All arrays must have same lengths and partitions.
    """

if TYPE_CHECKING:
    UprootSupportedDtypes = str | Number | npt.ArrayLike
    """
    str, Number, ~numpy.typing.ArrayLike: dtypes supported by :mod:`uproot`
    """

_UTF8_NULL = "\x00"
_UTF8_CONT = b"\x80"


def _align_utf8(s: bytes, n: int) -> bytes:
    return s + _UTF8_CONT * ((n - len(s)) % n)


class _Reader:
    _open_options = {
        "object_cache": None,
        "array_cache": None,
    }
    _dask_options = {
        "open_files": False,
    }
    _default_options = {
        "timeout": 180,
    }

    def __init__(self, **options):
        self._dask_options = self._default_options | self._dask_options | options
        self._open_options = self._default_options | self._open_options | options


class WriterOptions(TypedDict, total=False):
    name: str
    parents: bool
    basket_size: int


class ReaderOptions(TypedDict, total=False):
    branch_filter: Callable[[set[str]], set[str]]
    transform: Callable[[RecordLike], RecordLike]


BRANCH_FILTER = "branch_filter"


class TreeWriter:
    """
    :func:`uproot.recreate` with remote file support and :class:`TBasket` size control.

    Parameters
    ----------
    name : str, optional, default='Events'
        Default name of tree.
    parents : bool, optional, default=True
        Create parent directories if not exist.
    basket_size : int, optional
        Size of :class:`TBasket`. If not given, a new :class:`TBasket` will be created for each :meth:`extend` call.
    **options: dict, optional
        Additional options passed to :func:`uproot.recreate`.
    Attributes
    ----------
    tree : ~heptools.root.chunk.Chunk or list[~heptools.root.chunk.Chunk]
        Created :class:`TTree`.
    """

    def __init__(
        self,
        name: str = "Events",
        parents: bool = True,
        basket_size: int = ...,
        **options,
    ):
        self._default_name = name
        self._parents = parents
        self._basket_size = basket_size
        self._options = options

        self.tree: Chunk | list[Chunk] = None
        self._reset()

    def __call__(self, path: PathLike, name: str = None):
        """
        Set output path.

        Parameters
        ----------
        path : PathLike
            Path to output ROOT file.
        name : str, optional
            Name of tree. If given, it will temporarily override the tree name.

        Returns
        -------
        self : TreeWriter
        """
        self._path = EOS(path)
        self._tree_name = name or self._default_name
        return self

    def __enter__(self):
        """
        Open a temporary local ROOT file for writing.

        Returns
        -------
        self : TreeWriter
        """
        self.tree = None
        self._temp = self._path.local_temp(dir=".")
        self._file = uproot.recreate(self._temp, **self._options)
        self._trees = {self._tree_name: 0}
        return self

    def __exit__(self, *exc):
        """
        If no exception is raised, move the temporary file to the output path and store :class:`~.chunk.Chunk` information to :data:`tree`.
        """
        if not any(exc):
            self._flush()
            self._file.close()
            self.tree = []
            with uproot.open(self._temp) as file:
                for tree, size in self._trees.items():
                    if size == 0 or size is None:
                        continue
                    chunk = Chunk(source=self._temp, name=tree)
                    chunk._fetch_file(file)
                    chunk.path = self._path
                    self.tree.append(chunk)
                    if chunk.num_entries != size:
                        raise RuntimeError(
                            f'Tree "{tree}" is corrupted. Expected {size} entries, got {chunk.num_entries}.'
                        )
            if len(self.tree) > 0:
                self._temp.move_to(self._path, parents=self._parents, overwrite=True)
                if len(self.tree) == 1:
                    self.tree = self.tree[0]
            else:
                self._temp.rm()
        else:
            self._file.close()
            self._temp.rm()
        self._reset()

    @property
    def _buffer_size(self):
        return sum(len_record(b, self._backend) for b in self._buffer)

    def _reset(self):
        self._path = None
        self._temp = None
        self._file = None
        self._buffer = None if self._basket_size is ... else []
        self._backend = None
        self._trees = None

    def _flush(self):
        if self._basket_size is ...:
            data = self._buffer
            self._buffer = None
        else:
            data = concat_record(self._buffer, library=self._backend)
            self._buffer = []
        if data is not None and len_record(data, library=self._backend) > 0:
            if self._backend == "ak":
                from .. import awkward as akext

                if akext.is_jagged(data):
                    data = {k: data[k] for k in data.fields}
            if self._tree_name not in self._file:
                if self._backend == "pd":
                    branch_types = {col: data[col].values.dtype for col in data.columns}
                elif self._backend == "np":
                    branch_types = {k: v.dtype for k, v in data.items()}
                elif hasattr(data, "fields"):
                    branch_types = {k: data[k].type for k in data.fields}
                elif isinstance(data, dict):
                    branch_types = {k: v.type if hasattr(v, "type") else v.dtype for k, v in data.items()}
                self._file.mktree(self._tree_name, branch_types)
            self._file[self._tree_name].extend(data)
        data = None

    def extend(self, data: RecordLike):
        """
        Extend the :class:`TTree` with ``data`` using :meth:`uproot.writing.writable.WritableTree.extend`.

        .. note::
                The `VirtualArray` in `awkward < 2.0` will be materialized.

        Parameters
        ----------
        data : RecordLike
            Data to extend.

        Returns
        -------
        self : TreeWriter
        """
        backend = record_backend(data)
        if backend not in ("ak", "pd", "np"):
            raise TypeError(f"Unsupported data backend {type(data)}.")
        if self._basket_size is ...:
            self._backend = backend
        else:
            if self._backend is None:
                self._backend = backend
            else:
                if backend != self._backend:
                    raise TypeError(
                        f"Inconsistent data backend, expected {self._backend}, given {backend}."
                    )
        size = len_record(data, self._backend)
        if size == 0:
            return
        elif size is None:
            raise ValueError("The extended data does not have a well-defined length.")
        materialize_record(data, backend)
        self._trees[self._tree_name] += size
        if self._basket_size is ...:
            self._backend = backend
            self._buffer = data
            self._flush()
        else:
            start = 0
            while start < len_record(data, self._backend):
                diff = self._basket_size - self._buffer_size
                self._buffer.append(
                    slice_record(data, start, start + diff, library=self._backend)
                )
                start += diff
                if self._buffer_size >= self._basket_size:
                    self._flush()
        return self

    def switch(self, name: str):
        """
        Switch to another tree in the same ROOT file.

        .. warning::

            The buffer will be immediately flushed regardless of the basket size.
            It is recommended to finish the current tree before switching.

        Parameters
        ----------
        name : str
            Name of tree.

        Returns
        -------
        self : TreeWriter
        """
        if name == self._tree_name:
            return self
        if self._trees.get(name, 0) is None:
            raise ValueError(f'Tree name "{name}" conflicts with metadata name.')
        self._flush()
        self._tree_name = name
        self._trees.setdefault(self._tree_name, 0)
        return self

    def save_metadata(self, name: str, metadata: dict[str, UprootSupportedDtypes]):
        """
        Save metadata to ROOT file.

        Parameters
        ----------
        name : str
            Name of metadata.
        metadata : dict[str, UprootSupportedDtypes]
            A dictionary of metadata.

        Returns
        -------
        self : TreeWriter
        """
        if name in self._trees:
            size = self._trees[name]
            if size is None:
                raise ValueError(f'Metadata name "{name}" already exists.')
            else:
                raise ValueError(f'Metadata name "{name}" conflicts with other trees.')
        else:
            self._trees[name] = None
        if Version(uproot.__version__) >= Version("5.0.0"):
            self._file[name] = {k: [v] for k, v in metadata.items()}
        else:
            import awkward as ak
            import numpy as np

            data = {}
            for k, v in metadata.items():
                if isinstance(v, str):
                    v = np.frombuffer(_align_utf8(v.encode("utf-8"), 8), dtype=np.int64)
                    k = f"{_UTF8_NULL}{k}"
                data[k] = ak.Array([v])
            self._file[name] = data
        return self


class TreeReader(_Reader):
    """
    Read data from :class:`~.chunk.Chunk`.

    Parameters
    ----------
    branch_filter : ~typing.Callable[[set[str]], set[str]], optional
        A function to select branches. If not given, all branches will be read.
    transform : ~typing.Callable[[RecordLike], RecordLike], optional
        A function to transform the data after reading. If not given, no transformation will be applied.
    **options : dict, optional
        Additional options passed to :func:`uproot.open`.
    """

    def __init__(
        self,
        branch_filter: Callable[[set[str]], set[str]] = None,
        transform: Callable[[RecordLike], RecordLike] = None,
        **options,
    ):
        super().__init__(**options)
        self._filter = branch_filter
        self._transform = transform

    @overload
    def arrays(
        self,
        source: Chunk,
        library: Literal["ak"] = "ak",
        **options,
    ) -> ak.Array: ...
    @overload
    def arrays(
        self,
        source: Chunk,
        library: Literal["pd"] = "pd",
        **options,
    ) -> pd.DataFrame: ...
    @overload
    def arrays(
        self,
        source: Chunk,
        library: Literal["np"] = "np",
        **options,
    ) -> dict[str, np.ndarray]: ...
    def arrays(
        self,
        source: Chunk,
        library: Literal["ak", "pd", "np"] = "ak",
        **options,
    ) -> RecordLike:
        """
        Read ``source`` into array.

        Parameters
        ----------
        source : ~heptools.root.chunk.Chunk
            Chunk of :class:`TTree`.
        library : ~typing.Literal['ak', 'np', 'pd'], optional, default='ak'
            The library used to represent arrays.

            - ``library='ak'``: return :class:`ak.Array`.
            - ``library='pd'``: return :class:`pandas.DataFrame`.
            - ``library='np'``: return :class:`dict` of :class:`numpy.ndarray`.
        **options : dict, optional
            Additional options passed to :meth:`uproot.behaviors.TBranch.HasBranches.arrays`.

        Returns
        -------
        RecordLike
            Data from :class:`TTree`.
        """
        options["library"] = library
        branches = source.branches
        if self._filter is not None:
            branches = self._filter(branches)
        try:
            with uproot.open(source.path, **self._open_options) as file:
                data = file[source.name].arrays(
                    expressions=branches,
                    entry_start=source.entry_start,
                    entry_stop=source.entry_stop,
                    **options,
                )
                if library == "pd":
                    data.reset_index(drop=True, inplace=True)
                if self._transform is not None:
                    data = self._transform(data)
                return data
        except Exception as e:
            logging.error(f"Failed to read {source.path}", exc_info=e)
            raise

    @overload
    def concat(
        self,
        *sources: Chunk,
        library: Literal["ak"] = "ak",
        **options,
    ) -> ak.Array: ...
    @overload
    def concat(
        self,
        *sources: Chunk,
        library: Literal["pd"] = "pd",
        **options,
    ) -> pd.DataFrame: ...
    @overload
    def concat(
        self,
        *sources: Chunk,
        library: Literal["np"] = "np",
        **options,
    ) -> dict[str, np.ndarray]: ...
    def concat(
        self,
        *sources: Chunk,
        library: Literal["ak", "pd", "np"] = "ak",
        **options,
    ) -> RecordLike:
        """
        Read ``sources`` into one array. The branches of ``sources`` must be the same after filtering.

        .. todo::
            Add :mod:`multiprocessing` support.

        Parameters
        ----------
        sources : tuple[~heptools.root.chunk.Chunk]
            One or more chunks of :class:`TTree`.
        library : ~typing.Literal['ak', 'np', 'pd'], optional, default='ak'
            The library used to represent arrays.
        **options : dict, optional
            Additional options passed to :meth:`arrays`.

        Returns
        -------
        RecordLike
            Concatenated data from :class:`TTree`.
        """
        options["library"] = library
        if len(sources) == 1:
            return self.arrays(sources[0], **options)
        if library in ("ak", "pd", "np"):
            sources = Chunk.common(*sources)
            return concat_record(
                [self.arrays(s, **options) for s in sources], library=library
            )
        else:
            raise ValueError(f"Unknown library {library}.")

    @overload
    def iterate(
        self,
        *sources: Chunk,
        step: int = ...,
        library: Literal["ak"] = "ak",
        mode: Literal["balance", "partition"] = "partition",
        **options,
    ) -> Generator[ak.Array, None, None]: ...
    @overload
    def iterate(
        self,
        *sources: Chunk,
        step: int = ...,
        library: Literal["pd"] = "pd",
        mode: Literal["balance", "partition"] = "partition",
        **options,
    ) -> Generator[pd.DataFrame, None, None]: ...
    @overload
    def iterate(
        self,
        *sources: Chunk,
        step: int = ...,
        library: Literal["np"] = "np",
        mode: Literal["balance", "partition"] = "partition",
        **options,
    ) -> Generator[dict[str, np.ndarray], None, None]: ...
    def iterate(
        self,
        *sources: Chunk,
        step: int = ...,
        library: Literal["ak", "pd", "np"] = "ak",
        mode: Literal["balance", "partition"] = "partition",
        **options,
    ) -> Generator[RecordLike, None, None]:
        """
        Iterate over ``sources``.

        Parameters
        ----------
        sources : tuple[~heptools.root.chunk.Chunk]
            One or more chunks of :class:`TTree`.
        step : int, optional
            Number of entries to read in each iteration step. If not given, the chunk size will be used and the ``mode`` will be ignored.
        library : ~typing.Literal['ak', 'np', 'pd'], optional, default='ak'
            The library used to represent arrays.
        mode : ~typing.Literal['balance', 'partition'], optional, default='partition'
            The mode to generate iteration steps.

            - ``mode='balance'``: use :meth:`~.chunk.Chunk.balance`. The length of output arrays is not guaranteed to be ``step`` but no need to concatenate.
            - ``mode='partition'``: use :meth:`~.chunk.Chunk.partition`. The length of output arrays is guaranteed to be ``step`` except for the last one but need to concatenate.
        **options : dict, optional
            Additional options passed to :meth:`arrays`.

        Yields
        ------
        RecordLike
            A chunk of data from :class:`TTree`.
        """
        options["library"] = library
        if step is ...:
            chunks = Chunk.common(*sources)
        elif mode == "partition":
            chunks = Chunk.partition(step, *sources, common_branches=True)
        elif mode == "balance":
            chunks = Chunk.balance(step, *sources, common_branches=True)
        else:
            raise ValueError(f'Unknown mode "{mode}".')
        for chunk in chunks:
            if not isinstance(chunk, list):
                chunk = (chunk,)
            yield self.concat(*chunk, **options)

    @overload
    def dask(
        self,
        *sources: Chunk,
        partition: int = ...,
        library: Literal["ak"] = "ak",
    ) -> dak.Array: ...
    @overload
    def dask(
        self,
        *sources: Chunk,
        partition: int = ...,
        library: Literal["np"] = "np",
    ) -> dict[str, da.Array]: ...
    def dask(
        self,
        *sources: Chunk,
        partition: int = ...,
        library: Literal["ak", "np"] = "ak",
    ) -> DelayedRecordLike:
        """
        Read ``sources`` into delayed arrays.

        Parameters
        ----------
        sources : tuple[~heptools.root.chunk.Chunk]
            One or more chunks of :class:`TTree`.
        partition: int, optional
            If given, the ``sources`` will be splitted into smaller chunks targeting ``partition`` entries.
        library : ~typing.Literal['ak', 'np'], optional, default='ak'
            The library used to represent arrays.

        Returns
        -------
        DelayedRecordLike
            Delayed data from :class:`TTree`.
        """
        if partition is ...:
            sources = Chunk.common(*sources)
        else:
            sources = [*Chunk.balance(partition, *sources, common_branches=True)]
        branches = sources[0].branches
        if self._filter is not None:
            branches = self._filter(branches)
        options = self._dask_options | {"library": library, "filter_name": branches}
        files = []
        for chunk in sources:
            files.append(
                {
                    chunk.path: {
                        "object_path": chunk.name,
                        "steps": [[chunk.entry_start, chunk.entry_stop]],
                    }
                }
            )
        return uproot.dask(files, **options)

    def load_metadata(
        self, name: str, source: Chunk, builtin_types: bool = False
    ) -> dict[str, UprootSupportedDtypes]:
        """
        Load metadata from ROOT file.

        Parameters
        ----------
        name : str
            Name of the metadata.
        source : Chunk
            The ROOT file source.
        builtin_types : bool, optional, default=False
            Convert numpy dtypes to builtin types.

        Returns
        -------
        dict[str, UprootSupportedDtypes]
            A dictionary of metadata.
        """
        with uproot.open(source.path, **self._open_options) as file:
            if (num_entries := file[name].num_entries) != 1:
                raise ValueError(
                    f"Expected one entry in {source.path}[{name}], got {num_entries}."
                )
            if Version(uproot.__version__) > Version("5.0.0"):
                metadata = {k: v[0] for k, v in file[name].arrays(library="np").items()}
            else:
                import awkward as ak

                from .. import awkward as akext

                metadata = {}
                counts = []
                array = file[name].arrays(library="ak")
                for k in array.fields:
                    v = array[k]
                    if akext.is_jagged(v):
                        counts.append(f"n{k}")
                    if k.startswith(_UTF8_NULL):
                        k = k.removeprefix(_UTF8_NULL)
                        metadata[k] = ak.to_numpy(v).tobytes().decode("utf-8", "ignore")
                    else:
                        metadata[k] = ak.to_numpy(v)[0]
                for k in counts:
                    del metadata[k]

            if builtin_types:
                for k, v in metadata.items():
                    match type(v).__module__:
                        case "numpy":
                            metadata[k] = v.item() if not v.shape else v.tolist()
            return metadata
