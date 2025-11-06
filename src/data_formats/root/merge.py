from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from ...dask_tools.delayed import delayed
from ...storage.eos import EOS, PathLike
from .chunk import Chunk
from .io import ReaderOptions, TreeReader, TreeWriter, WriterOptions

if TYPE_CHECKING:
    import awkward as ak


@delayed
def move(
    path: PathLike,
    source: Chunk,
    clean_source: bool = True,
    dask: bool = False,
):
    """
    Move ``source`` to ``path``.

    Parameters
    ----------
    path : PathLike
        Path to output ROOT file.
    source : ~heptools.root.chunk.Chunk
        Source chunk to move.
    clean_source : bool, optional, default=True
        If ``True``, remove the source chunk after moving.
    dask : bool, optional, default=False
        If ``True``, return a :class:`~dask.delayed.Delayed` object.

    Returns
    -------
    Chunk or Delayed
        Moved chunk.
    """
    source = source.deepcopy()
    source.path = (source.path.move_to if clean_source else source.path.copy_to)(
        path, overwrite=True, parents=True
    )
    return source


@delayed
def merge(
    path: PathLike,
    *sources: Chunk,
    step: int,
    writer_options: WriterOptions = None,
    reader_options: ReaderOptions = None,
    transform: Callable[[ak.Array], ak.Array] = None,
    dask: bool = False,
):
    """
    Merge ``sources`` into one :class:`~.chunk.Chunk`.

    Parameters
    ----------
    path : PathLike
        Path to output ROOT file.
    sources : tuple[~heptools.root.chunk.Chunk]
        Chunks to merge.
    step : int
        Number of entries to read and write in each iteration step.
    writer_options : dict, optional
        Additional options passed to :class:`~.io.TreeWriter`.
    reader_options : dict, optional
        Additional options passed to :class:`~.io.TreeReader`.
    transform : ~typing.Callable[[ak.Array], ak.Array], optional
        A function to transform the array before writing.
    dask : bool, optional, default=False
        If ``True``, return a :class:`~dask.delayed.Delayed` object.

    Returns
    -------
    Chunk or Delayed
        Merged chunk.
    """
    writer_options = writer_options or {}
    reader_options = reader_options or {}
    with TreeWriter(**writer_options)(path) as writer:
        for data in TreeReader(**reader_options).iterate(*sources, step=step):
            if transform:
                data = transform(data)
            writer.extend(data)
    return writer.tree


@delayed
def clean(
    source: list[Chunk],
    merged: list[Chunk],
    dask: bool = False,
):
    """
    Clean ``source`` after merging.

    Parameters
    ----------
    source : list[~heptools.root.chunk.Chunk]
        Source chunks to be cleaned.
    merged : list[~heptools.root.chunk.Chunk]
        Merged chunks.
    dask : bool, optional, default=False
        If ``True``, return a :class:`~dask.delayed.Delayed` object.

    Returns
    -------
    merged: list[Chunk] or Delayed
    """
    for chunk in source:
        try:
            chunk.path.rm()
        except Exception:
            ...
    return merged


def resize(
    path: PathLike,
    *sources: Chunk,
    step: int,
    chunk_size: int = ...,
    writer_options: WriterOptions = None,
    reader_options: ReaderOptions = None,
    clean_source: bool = True,
    transform: Callable[[ak.Array], ak.Array] = None,
    dask: bool = False,
):
    """
    :func:`merge` ``sources`` into :class:`~.chunk.Chunk` and :func:`clean` ``sources`` after merging.

    Parameters
    ----------
    path : PathLike
        Path to output ROOT file.
    sources : tuple[~heptools.root.chunk.Chunk]
        Chunks to merge.
    step : int
        Number of entries to read and write in each iteration step.
    chunk_size : int, optional
        Number of entries in each chunk. If not given, all entries will be merged into one chunk.
    writer_options : dict, optional
        Additional options passed to :class:`~.io.TreeWriter`.
    reader_options : dict, optional
        Additional options passed to :class:`~.io.TreeReader`.
    clean_source : bool, optional, default=True
        If ``True``, remove the source chunk after moving.
    transform : ~typing.Callable[[ak.Array], ak.Array], optional
        A function to transform the array before writing.
    dask : bool, optional, default=False
        If ``True``, return a :class:`~dask.delayed.Delayed` object.

    Returns
    -------
    list[Chunk] or Delayed
        Merged chunks.
    """
    path = EOS(path)
    results: list[Chunk] = []
    move_kws = dict(
        clean_source=clean_source,
        dask=dask,
    )
    merge_kws = dict(
        step=step,
        reader_options=reader_options,
        writer_options=writer_options,
        transform=transform,
        dask=dask,
    )
    to_clean = {(chunk.path, chunk.uuid): chunk for chunk in sources}
    if chunk_size is ...:
        if len(sources) == 1:
            results.append(move(path, sources[0], **move_kws))
            to_clean.clear()
        else:
            results.append(merge(path, *sources, **merge_kws))
    else:
        output = path
        parent = path.parent
        filename = f'{path.stem}.chunk{{index}}{"".join(path.suffixes)}'
        chunks = [*Chunk.partition(chunk_size, *sources, common_branches=True)]
        for index, to_merges in enumerate(chunks):
            if len(chunks) > 1:
                output = parent / filename.format(index=index)
            if (
                len(to_merges) == 1
                and (to_merge := to_merges[0]).entry_start == 0
                and to_merge.entry_stop == to_merge.num_entries
            ):
                results.append(move(output, to_merge, **move_kws))
                to_clean.pop((to_merge.path, to_merge.uuid), None)
            else:
                results.append(merge(output, *to_merges, **merge_kws))
    if clean_source and to_clean:
        results = clean(to_clean.values(), results, dask=dask)
    return results
