from __future__ import annotations

import pickle
import socket
import warnings
from typing import TYPE_CHECKING, Literal

import fsspec
from src.classifier.task.state import GlobalSetting, _share_global_state

from src.classifier.config.state import System

if TYPE_CHECKING:
    from src.storage.eos import EOS


class save(GlobalSetting):
    "Save global states to file."

    @classmethod
    def parse(cls, opts: list[str]):
        with fsspec.open(opts[0], "wb") as f:
            pickle.dump(_share_global_state(), f)

    @classmethod
    def help(cls):
        infos = [
            f"usage: {cls.__mod_name__()} OUTPUT",
            "",
            cls._help_doc(),
            "",
        ]
        return "\n".join(infos)


class load(GlobalSetting):
    "Load global states from files."

    @classmethod
    def parse(cls, opts: list[str]):
        for opt in opts:
            with fsspec.open(opt, "rb") as f:
                pickle.load(f)()

    @classmethod
    def help(cls):
        infos = [
            f"usage: {cls.__mod_name__()} INPUT [INPUT ...]",
            "",
            cls._help_doc(),
            "",
        ]
        return "\n".join(infos)


class IO(GlobalSetting):
    "Basic I/O settings."

    timestamp: str = "%Y-%m-%dT%H-%M-%S"
    "timestamp format"

    output: EOS = "./{main}-{timestamp}/"
    "base directory for all outputs"
    monitor: EOS = "diagnostics"
    "name of the folder for monitor outputs"
    report: EOS = "report"
    "name of the folder for report outputs"

    states: EOS = "states.pkl"
    "name of the state file"
    result: EOS = "result.json"
    "name of the result file"

    @classmethod
    def _generate_path(cls, value: str):
        return cls.output / value

    @classmethod
    def get__output(cls, value: str):
        from src.storage.eos import EOS

        if isinstance(value, str):
            value = value.format(
                user=System.user_name,
                main=System.main_task,
                timestamp=System.startup_time.strftime(cls.timestamp),
            )
        return EOS(value)

    @classmethod
    def get__monitor(cls, value: str):
        return cls._generate_path(value)

    @classmethod
    def get__report(cls, value: str):
        return cls._generate_path(value)

    @classmethod
    def get__profiler(cls, value: str):
        return cls._generate_path(value)

    @classmethod
    def get__states(cls, value: str):
        return cls.output / value

    @classmethod
    def get__result(cls, value: str):
        return cls.output / value


class Monitor(GlobalSetting):
    "Basic monitor settings."

    enable: bool = True
    "enable monitor system"
    file: str = "meta.json"
    "name of the monitor metadata file"
    address: tuple[str, int] = ":10200"
    """address of the monitor to start on/connect to ("ip:port" for TCP socket or any other string for UNIX domain socket)"""
    connect: bool = False
    "connect to the monitor instead of starting a new one"

    # logging
    log_show_connection: bool = False
    "show a message when connected to the monitor"

    # performance
    retry_max: int = 1
    "default max retries when sending packets"

    # builtins
    socket_timeout: float = None
    "default timeout for python socket"
    warnings_ignore: bool = True
    "ignore python warnings"

    @classmethod
    def set__socket_timeout(cls, value: float):
        socket.setdefaulttimeout(value)
        return NotImplemented

    @classmethod
    def set__warnings_ignore(cls, value: bool):
        if value:
            warnings.filterwarnings("ignore")
        else:
            warnings.filterwarnings("default")
        return NotImplemented

    @classmethod
    def get__address(cls, value: int | str):
        if isinstance(value, int):
            return None, value
        if value is None:
            return None, None
        parts = value.rsplit(":", 1)
        if len(parts) == 2:
            try:
                port = int(parts[1])
                host = parts[0] or None
                return host, port
            except Exception:
                pass
        return value or None, None


class Analysis(GlobalSetting):
    "Analysis job settings."

    enable: bool = True
    "enable analysis jobs"
    max_workers: int = 1
    "maximum number of workers"


class Multiprocessing(GlobalSetting):
    "Multiprocessing settings."

    context_method: Literal["fork", "forkserver", "spawn"] = "forkserver"
    "method to create subprocesses"
    context_library: Literal["torch", "builtins"] = "torch"
    "library for multiprocessing"
    preload: list[str] = ["torch"]
    "list of modules to load in forkserver"
    torch_sharing_strategy: Literal["file_system", "file_descriptor"] = "file_system"
    "strategy for pytorch shared memory"


class ResultKey(GlobalSetting):
    "Keys in result dict."

    uuid: str = "uuid"
    "result UUID"
    command: str = "command"
    "python command"
    reproducible: str = "reproducible"
    "config to reproduce the result"

    # analyze
    analysis: str = "analysis"
    "analysis outputs"

    # cache
    cache: str = "cache"
    "metadata of the cached datasets"

    # evaluate
    predictions: str = "predictions"
    "metadata of the evaluated predictions"

    # train
    models: str = "models"
    "metadata of the trained models"


class ROOT(GlobalSetting):
    "ROOT related settings."

    friend_allow_missing: bool = True
    "allow missing friend tree chunks"

    @classmethod
    def set__friend_allow_missing(cls, value: bool):
        # placeholder for compatibility
        return NotImplemented
