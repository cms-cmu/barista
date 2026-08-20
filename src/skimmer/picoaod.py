from __future__ import annotations

import gc
import logging
import re
import uuid
from abc import abstractmethod
from concurrent.futures import Future, ProcessPoolExecutor
from itertools import chain
from typing import Iterable

import awkward as ak
import numpy as np
import numpy.typing as npt
import uproot
from src.skimmer.cutflow import cutflow
from src.data_formats.awkward.zip import NanoAOD
from src.data_formats.root import Chunk, TreeReader, TreeWriter, merge
from src.storage.eos import EOS, PathLike
from src.utils.wrapper import retry
from coffea.processor import ProcessorABC

_PICOAOD = "picoAOD"
_ROOT = ".root"


class SkimmingError(Exception):
    __module__ = Exception.__module__


def _log_exception(e, *_):
    logging.error("The following exception occurred during skimming:", exc_info=e)
    return {}


_decompression_executor = uproot.ThreadPoolExecutor()


def _clear_cache(events: ak.Array):
    # Purge cached branches and virtual buffers
    for cache in getattr(events, "caches", []):
        cache.clear()
    behavior = getattr(events, "behavior", None)
    if isinstance(behavior, dict):
        for k in list(behavior.keys()):
            if isinstance(k, tuple) and len(k) > 0 and isinstance(k[0], str) and k[0].startswith("__"):
                behavior.pop(k, None)
    gc.collect()


def _branch_filter(collections: Iterable[str], branches: Iterable[str]):
    branches = chain(
        map("({}_.*)".format, collections or ()),
        map("(n{})".format, collections or ()),
        map("({})".format, branches or ()),
    )
    return rf'^(?!({"|".join(branches)})$).*$'


class PicoAOD(ProcessorABC):
    def __init__(
        self,
        base_path: PathLike,
        step: int,
        skip_collections: Iterable[str] = None,
        skip_branches: Iterable[str] = None,
        pico_base_name: str = _PICOAOD,
        campaign: str = ...,
    ):
        self._base = EOS(base_path)
        self._step = step
        self._pico_base_name = pico_base_name
        if campaign is ...:
            campaign = f"skim-{uuid.uuid4().hex[:8]}"
        if campaign is not None:
            logging.info(f"Using campaign name: {campaign}")
        self._campaign = campaign

        self._branch_filter = re.compile(
            _branch_filter(skip_collections, skip_branches)
        )
        self._transform = NanoAOD(regular=False, jagged=True)

    def _filter(self, branches: set[str]):
        return {*filter(self._branch_filter.match, branches)}

    def update_branch_filter(self, skip_collections, skip_branches):
        self._branch_filter = re.compile(
            _branch_filter(skip_collections, skip_branches)
        )

    @abstractmethod
    def select(
        self, events: ak.Array
    ) -> (
        npt.NDArray[np.bool_]
        | tuple[npt.NDArray[np.bool_], ak.Array]
        | tuple[npt.NDArray[np.bool_], ak.Array | None, dict]
    ):
        pass

    def preselect(self, events: ak.Array) -> npt.NDArray[np.bool_] | None:
        pass

    @property
    def preselected(self):
        return self._preselected

    # no retry, return empty dict if any exception
    @retry(1, handler=_log_exception, skip=(SkimmingError,))
    def process(self, events: ak.Array):
        EOS.set_retry(3, 10)  # 3 retries with 10 seconds interval

        # prepare
        dataset = events.metadata["dataset"]
        chunk = Chunk.from_coffea_events(events)
        source_chunk = {str(chunk.path): [(chunk.entry_start, chunk.entry_stop)]}
        path = (
            self._base
            / f"{dataset}/{self._pico_base_name}_{chunk.uuid}_{chunk.entry_start}_{chunk.entry_stop}{_ROOT}"
        )

        # check if chunks is already finished
        if self._campaign is not None:
            reader = TreeReader()
            try:
                cached = Chunk(path, fetch=True)
                metadata = reader.load_metadata(
                    self._campaign, cached, builtin_types=True
                )
                return {dataset: metadata | {"files": [cached], "source": source_chunk}}
            except Exception:
                pass

    # select events
    # self._cutFlow should be set by child classes, not here
        # preselect
        preselected = self.preselect(events)
        if preselected is None:
            self._preselected = np.ones(len(events), dtype=bool)
        else:
            self._preselected = np.asarray(preselected)
        self._preselected.setflags(write=False)

        # select
        selected = self.select(events)

        # Parse single-stream or multi-stream returns
        is_multi_stream = isinstance(selected, dict)
        streams: dict[str | None, dict] = {}

        if is_multi_stream:
            for stream_name, stream_val in selected.items():
                stream_added, stream_result = None, {}
                if isinstance(stream_val, tuple):
                    if len(stream_val) >= 2:
                        stream_added = stream_val[1]
                    if len(stream_val) >= 3:
                        stream_result = stream_val[2] or stream_result
                    stream_mask = stream_val[0]
                else:
                    stream_mask = stream_val
                if preselected is not None:
                    stream_mask = preselected & stream_mask
                streams[stream_name] = {
                    "selected": np.asarray(stream_mask),
                    "added": stream_added,
                    "result": stream_result,
                }
        else:
            added, result_extra = None, {}
            if isinstance(selected, tuple):
                if len(selected) >= 2:
                    added = selected[1]
                if len(selected) >= 3:
                    result_extra = selected[2] or result_extra
                selected = selected[0]
            if preselected is not None:
                selected = preselected & selected
            streams[None] = {
                "selected": np.asarray(selected),
                "added": added,
                "result": result_extra,
            }

        # Calculate union of all selected streams
        union_selected = np.zeros(len(events), dtype=bool)
        for s in streams.values():
            union_selected |= s["selected"]

        # Common weights calculation
        weights = {}
        if "genWeight" in events.fields:
            genWeight = events.genWeight
            if preselected is not None:
                genWeight = genWeight[preselected]
                outliers = events[~preselected]
                if len(outliers) > 0:
                    weights["outliers"] = [int(e) for e in outliers.event]
                    weights["sumw_diff"] = float(np.sum(outliers.genWeight))
                    weights["sumw2_diff"] = float(np.sum(outliers.genWeight**2))
            weights["sumw"] = float(np.sum(genWeight))
            weights["sumw2"] = float(np.sum(genWeight**2))
        else:
            nevents = len(events) if preselected is None else float(np.sum(preselected))
            weights["sumw"] = nevents
            weights["sumw2"] = nevents

        result = {}
        stream_metadata = {}
        for stream_name, s in streams.items():
            stream_dataset = dataset if stream_name is None else f"{dataset}_{stream_name}"
            stream_saved = int(np.sum(s["selected"]))
            metadata = (
                {
                    "total_events": len(events),
                    "saved_events": stream_saved,
                }
                | weights
                | s["result"]
            )
            stream_metadata[stream_name] = metadata
            result[stream_dataset] = metadata | {
                "files": [],
                "source": source_chunk,
            }
            if hasattr(self, "_cutFlow") and self._cutFlow is not None:
                self._cutFlow.addOutputSkim(result, stream_dataset)
                if "genWeight" not in events.fields:
                    self._cutFlow.addOutputLumisProcessed(
                        result, stream_dataset, events.run, events.luminosityBlock
                    )

            # Sanity check
            if s["added"] is not None and (size := len(s["added"])) != stream_saved:
                raise SkimmingError(
                    f"Length of additional branches ({size}) does not match the number of selected events ({stream_saved}) for stream {stream_name}"
                )

        # Clear cache
        _clear_cache(events)

        # Save selected events
        total_saved = sum(result[k]["saved_events"] for k in result)
        if total_saved > 0:
            reader = TreeReader(self._filter)
            active_writers: dict[str | None, TreeWriter] = {}
            saved_counters: dict[str | None, int] = {k: 0 for k in streams}

            for stream_name, s in streams.items():
                stream_dataset = dataset if stream_name is None else f"{dataset}_{stream_name}"
                if result[stream_dataset]["saved_events"] > 0:
                    stream_path = (
                        self._base
                        / f"{stream_dataset}/{self._pico_base_name}_{chunk.uuid}_{chunk.entry_start}_{chunk.entry_stop}{_ROOT}"
                    )
                    writer = TreeWriter()(stream_path)
                    writer.__enter__()
                    active_writers[stream_name] = writer

            try:
                with reader._open_with_retry(chunk.path) as file:
                    tree = file[chunk.name]
                    branches = chunk.branches
                    if self._filter is not None:
                        branches = self._filter(branches)
                    for i, chunks in enumerate(
                        Chunk.partition(self._step, chunk, common_branches=True)
                    ):
                        _union = union_selected[i * self._step : (i + 1) * self._step]
                        _range = np.arange(len(_union))[_union]
                        if len(_range) == 0:
                            continue
                        _start, _stop = int(_range[0]), int(_range[-1] + 1)
                        _entry_start = (chunks[0].entry_start or 0) + _start
                        _entry_stop = (chunks[0].entry_start or 0) + _stop
                        data = tree.arrays(
                            expressions=branches,
                            entry_start=_entry_start,
                            entry_stop=_entry_stop,
                            library="ak",
                            decompression_executor=_decompression_executor,
                        )

                        for stream_name, writer in active_writers.items():
                            s = streams[stream_name]
                            _stream_sel = s["selected"][i * self._step + _start : i * self._step + _stop]
                            if not np.any(_stream_sel):
                                continue
                            stream_data = data[_stream_sel]
                            if s["added"] is not None:
                                n_sel = int(np.sum(_stream_sel))
                                cur_saved = saved_counters[stream_name]
                                _added = s["added"][cur_saved : cur_saved + n_sel]
                                for k in s["added"].fields:
                                    stream_data[k] = _added[k]
                                saved_counters[stream_name] = cur_saved + n_sel
                            stream_data = self._transform(stream_data)
                            writer.extend(stream_data)
            finally:
                for stream_name, writer in active_writers.items():
                    stream_dataset = dataset if stream_name is None else f"{dataset}_{stream_name}"
                    if self._campaign is not None:
                        writer.save_metadata(self._campaign, stream_metadata[stream_name])
                    writer.__exit__(None, None, None)
                    if writer.tree is not None:
                        result[stream_dataset]["files"].append(writer.tree)

        return result

    def postprocess(self, accumulator):
        pass


def _fetch_metadata(dataset: str, path: PathLike, dask: bool = False):
    try:
        with uproot.open(path) as f:
            if "genEventCount" in f["Runs"].keys():
                data = f["Runs"].arrays(
                    ["genEventCount", "genEventSumw", "genEventSumw2"]
                )
                return {
                    dataset: {
                        "count": int(np.sum(data["genEventCount"])),
                        "sumw_raw": float(
                            np.sum(data["genEventSumw"].to_numpy().astype(np.float64))
                        ),
                        "sumw2_raw": float(
                            np.sum(data["genEventSumw2"].to_numpy().astype(np.float64))
                        ),
                    }
                }
            else:
                data = f["Events"].arrays(["event"])
                return {dataset: {"count": float(ak.num(data["event"], axis=0))}}
    except:
        return {dataset: {"bad_files": [str(EOS(path))]}}


def fetch_metadata(
    fileset: dict[str, dict[str, list[str]]], n_process: int = None, dask: bool = True
) -> list[dict[str, dict[str]]]:
    if not dask:
        with ProcessPoolExecutor(max_workers=n_process) as executor:
            tasks: list[Future] = []
            for dataset, files in fileset.items():
                for file in files["files"]:
                    tasks.append(
                        executor.submit(_fetch_metadata, dataset, file, dask=dask)
                    )
            results = [task.result() for task in tasks]
    else:
        from dask import delayed

        func = delayed(_fetch_metadata)
        results = []
        for dataset, files in fileset.items():
            for file in files["files"]:
                results.append(func(dataset, file, dask=dask))
    return results


def integrity_check(
    fileset: dict[str, dict[str, list[str]]],
    output: dict[str, dict[str, dict[str, list[tuple[int, int]]]]],
    num_entries: dict[str, dict[str, int]] = None,
):
    complete = True
    logging.info("Checking integrity of the picoAOD...")
    diff = set(fileset) - set(output)
    miss_dict = {}
    if diff:
        logging.error(f"The whole dataset is missing: {diff}")
        complete = False
        miss_dict["dataset_missing"] = list(diff)
    for dataset in output:
        if len(output[dataset]["files"]) == 0:
            logging.warning(f'No file is saved for "{dataset}"')
        inputs = map(EOS, fileset[dataset]["files"])
        outputs = {EOS(k): v for k, v in output[dataset]["source"].items()}
        ns = (
            None
            if num_entries is None
            else {EOS(k): v for k, v in num_entries[dataset].items()}
        )
        file_missing = []
        chunk_missing = []
        for file in inputs:
            if file not in outputs:
                logging.error(f'The whole file is missing in outputs: "{file}"')
                complete = False
                file_missing.append(str(file))
            else:
                chunks = sorted(outputs[file], key=lambda x: x[0])
                if ns is not None:
                    chunks.append((ns[file], ns[file]))
                merged = []
                start, stop = 0, 0
                for _start, _stop in chunks:
                    if _start != stop:
                        if start != stop:
                            merged.append([str(start), str(stop)])
                        start = _start
                        logging.error(f'Missing chunk: [{stop}, {_start}) in "{file}"')
                        complete = False
                        chunk_missing.append(f'[{stop}, {_start}) in "{file}"')
                    stop = _stop
                if start != stop:
                    merged.append([start, stop])
        if file_missing:
            miss_dict["file_missing"] = file_missing
        if chunk_missing:
            miss_dict["chunk_missing"] = chunk_missing
        output[dataset].pop("source")
        output[dataset]["missing"] = miss_dict
    return output, complete


def resize(
    base_path: PathLike,
    output: dict[str, dict[str, list[Chunk]]],
    step: int,
    chunk_size: int,
    dask: bool = True,
    pico_base_name: str = _PICOAOD,
):
    base = EOS(base_path)
    transform = NanoAOD(regular=False, jagged=True)
    for dataset, chunks in output.items():
        if len(chunks["files"]) > 0:
            output[dataset]["files"] = merge.resize(
                base / dataset / f"{pico_base_name}{_ROOT}",
                *chunks["files"],
                step=step,
                chunk_size=chunk_size,
                reader_options={"transform": transform},
                dask=dask,
            )
    return output
