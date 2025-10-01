from __future__ import annotations

import logging
from functools import cached_property
from inspect import getmro
from typing import Callable, Iterable

from src.classifier.config.setting.cms import CollisionData, MC_HH_ggF, MC_TTbar
from src.classifier.config.state import Flags
from src.classifier.task import ArgParser, Dataset, parse


class _PicoAOD(Dataset):
    pico_filelists: Iterable[Callable[[str], Iterable[list[str]]]]
    pico_files: Iterable[Callable[[str], Iterable[list[str]]]]

    argparser = ArgParser()
    argparser.remove_argument("--files", "--filelists")
    argparser.add_argument(
        "--metadata",
        nargs="*",
        default=["datasets"],
        help="names of the metadata files.",
    )

    def __init__(self):
        super().__init__()
        if not hasattr(self.opts, "filelists"):
            self.opts.filelists = []
        if not hasattr(self.opts, "files"):
            self.opts.files = []
        for metadata in self.opts.metadata:
            self.opts.filelists.extend(
                self._filelists(f"{metadata}.yml@@datasets")
            )
        for metadata in self.opts.metadata:
            self.opts.files.extend(self._files(f"{metadata}.yml@@datasets"))

    def _iter(self, name: str):
        for base in getmro(self.__class__):
            if issubclass(base, _PicoAOD) and (
                (datasets := vars(base).get(name)) is not None
            ):
                yield from datasets

    def _load(self, name: str, metadata: str):
        filelists = []
        for dataset in self._iter(name):
            filelists.extend(dataset(self, metadata))
        return filelists

    def _files(self, metadata: str):
        return self._load("pico_files", metadata)

    def _filelists(self, metadata: str):
        return self._load("pico_filelists", metadata)


class _MCDataset:
    processes: tuple[str, ...]


class _ttbar(_MCDataset):
    processes = ("ttbar",)

    def __new__(cls, self: MC, metadata: str):
        filelists = []
        if "ttbar" in self.mc_processes:
            for year in CollisionData.eras:
                filelists.append(
                    [
                        f"label:ttbar,year:{year}",
                        *(
                            metadata + f".{tt}.{year}.picoAOD.files"
                            for tt in MC_TTbar.datasets
                        ),
                    ]
                )
        return filelists




class _signal(_MCDataset):
    processes = ("GluGluToHHTo2B2VLNu2J",)

    @classmethod
    def __c2str(cls, coupling: float):
        return f"{coupling:.6g}".replace(".", "p")

    @classmethod
    def __cs2label(cls, couplings: dict[str, float]):
        return ",".join(f"{k}:{v:.6g}" for k, v in couplings.items())

    def __new__(cls, self: MC, metadata: str):
        filelists = []
        
        # This will be "GluGluToHHTo2B2VLNu2J" based on your command
        process_name = "GluGluToHHTo2B2VLNu2J" 

        if process_name in self.mc_processes:
            for year in CollisionData.eras:
                # The label for the data being loaded
                label = f"label:signal,year:{year}"
                
                # The exact path it will look for in the YAML
                lookup_path = f"{metadata}.{process_name}.{year}.picoAOD.files"
                
                filelists.append([label, lookup_path])
                
        return filelists


def _data(self: Data, metadata: str):
    filelists = []
    if "detector" in self.data_sources:
        for year, eras in CollisionData.eras.items():
            filelists.append(
                [
                    f"label:data,year:{year},source:detector",
                    *(metadata + f".data.{year}.picoAOD.{e}.files" for e in eras),
                ]
            )
    return filelists


class Data(_PicoAOD):
    pico_filelists = (_data,)
    pico_files = ()

    argparser = ArgParser()
    argparser.add_argument(
        "--data-source",
        metavar="SOURCE",
        default=["detector"],
        choices=("detector", "mixed", "synthetic"),
        help="choose the source of the data",
        nargs="*",
    )

    @cached_property
    def data_sources(self) -> set[str]:
        return {*self.opts.data_source}


class MC(_PicoAOD):
    argparser = ArgParser()
    argparser.add_argument(
        "--mc-processes",
        metavar="PROCESS",
        nargs="*",
        default=None,
        help="list of MC processes. If not specified, all processes are used",
    )

    @cached_property
    def mc_processes(self) -> set[str]:
        selected = self.mc_processes_all
        if self.opts.mc_processes is not None:
            selected = selected.intersection(self.opts.mc_processes)
        if Flags.debug:
            logging.debug(
                "The following MC processes are selected:",
                f"{sorted(selected)} of {sorted(self.mc_processes_all)}",
            )
        return selected

    @cached_property
    def mc_processes_all(self) -> set[str]:
        processes = set()
        for dataset in self._iter("pico_filelists"):
            if isinstance(dataset, type) and issubclass(dataset, _MCDataset):
                processes.update(dataset.processes)
        return processes


class Background(MC):
    pico_filelists = (_ttbar,)


class Signal(MC):
    pico_filelists = (_signal,)
