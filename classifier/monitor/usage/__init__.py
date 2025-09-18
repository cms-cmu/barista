import json
import time
from collections import defaultdict
from threading import Lock, Thread
from typing import TypedDict

import psutil
from classifier.config.setting import monitor as cfg
from classifier.config.state import System

from ..core import MonitorProxy, Node, Recorder, post_to_monitor

_MIB = 2**20
_CUDA = {"12.1": 254}  # MiB


class Resource(TypedDict):
    time: int  # ns
    cpu: dict[int, float]  # pid: cpu %
    memory: dict[int, float]  # pid: memory MiB
    gpu: dict[int, float]  # pid: gpu memory MiB


class Checkpoint(TypedDict):
    time: float  # s
    name: tuple[str, ...]
    pids: list[int]


class ProcessInfo(TypedDict):
    pid: int
    parent: int
    start: float  # s
    cmd: list[str]


class Usage(MonitorProxy):
    _records_local: list[Resource] = []
    _processes_local: set[int] = set()
    _lock = Lock()
    _tracker: Thread = None

    # GPU
    _n_gpu: int = None
    _torch_calibration: int = None  # MiB
    _pynvml_handles: list = None
    _pynvml_unavailable = System.in_singularity

    # state
    _running = False

    def __init__(self):
        self._records: dict[Node, list[Resource]] = defaultdict(list)
        self._processes: dict[str, list[ProcessInfo]] = defaultdict(list)
        self._checkpoints: dict[Node, list[Checkpoint]] = defaultdict(list)

    @classmethod
    @cfg.check(cfg.Usage)
    def start(cls):
        cls._running = True
        cls._tracker = Thread(target=cls._track, daemon=True)
        cls._tracker.start()

    @classmethod
    def stop(cls):
        if cls._tracker is not None:
            cls._running = False
            cls._tracker.join()
            cls._tracker = None
            cls._records_local = []

    @classmethod
    @cfg.check(cfg.Usage)
    def checkpoint(cls, *tags: str):
        if not cls._running:
            return
        start_t = time.time_ns()
        with cls._lock:
            records = cls._records_local
            cls._records_local = []
        end_t = time.time_ns()
        cls._checkpoint(
            Recorder.node(), {"time": (start_t + end_t) // 2, "name": tags}, records
        )

    @post_to_monitor
    @cfg.check(cfg.Usage)
    def _checkpoint(self, node: Node, checkpoint: Checkpoint, records: list[Resource]):
        self._records[node].extend(records)
        self._checkpoints[node].append(checkpoint)

    @post_to_monitor
    @cfg.check(cfg.Usage)
    def _send_pinfo(self, ip: str, info: ProcessInfo):
        self._processes[ip].append(info)

    @classmethod
    def _pinfo(self, process: psutil.Process):
        if process.pid not in self._processes_local:
            self._processes_local.add(process.pid)
            info = {
                "pid": process.pid,
                "parent": process.ppid(),
                "start": process.create_time(),
                "cmd": process.cmdline(),
            }
            self._send_pinfo(Recorder.node()[0], info)

    @classmethod
    def _track(cls):
        while cls._running:
            start_t = time.time_ns()
            p = psutil.Process()
            cls._pinfo(p)
            # CPU, memory
            cpu = {p.pid: p.cpu_percent(cfg.Usage.interval)}
            mem = {p.pid: p.memory_info().rss / _MIB}
            for c in p.children(recursive=True):
                try:
                    cls._pinfo(c)
                    cpu[c.pid] = c.cpu_percent(cfg.Usage.interval)
                    mem[c.pid] = c.memory_info().rss / _MIB
                except psutil.NoSuchProcess:
                    cpu.pop(c.pid, None)
                    mem.pop(c.pid, None)
            # GPU
            if cfg.Usage.gpu:
                if cfg.Usage.gpu_force_torch or cls._pynvml_unavailable:
                    gpu = cls._gpu_torch(p.pid)
                else:
                    gpu = cls._gpu_nvml(p.pid, *cpu)
                    if gpu is None:
                        cls._pynvml_unavailable = True
                        gpu = cls._gpu_torch(p.pid)
            else:
                gpu = {}
            end_t = time.time_ns()
            with cls._lock:
                cls._records_local.append(
                    {
                        "time": (start_t + end_t) // 2,
                        "cpu": cpu,
                        "memory": mem,
                        "gpu": gpu,
                    }
                )
            remain_t = cfg.Usage.interval - (end_t - start_t) / 1e9
            if remain_t > 0:
                time.sleep(remain_t)

    @classmethod
    def _gpu_nvml(cls, *pids: int) -> dict[int, float]:
        import pynvml

        if cls._n_gpu is None:
            try:
                pynvml.nvmlInit()
                cls._n_gpu = pynvml.nvmlDeviceGetCount()
            except pynvml.NVMLError:
                return None
        if cls._n_gpu > 0:
            if cls._pynvml_handles is None:
                cls._pynvml_handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(cls._n_gpu)
                ]
            gpu = defaultdict(float)
            pids = set(pids)
            for handle in cls._pynvml_handles:
                for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    pid, mem = p.pid, p.usedGpuMemory
                    if (pid in pids) and (mem is not None):
                        gpu[pid] += mem / _MIB
            return gpu
        return {}

    @classmethod
    def _gpu_torch(cls, pid: int) -> dict[int, float]:
        import torch

        if cls._n_gpu is None:
            cls._n_gpu = torch.cuda.device_count()
        if cls._n_gpu > 0:
            if cls._torch_calibration is None:
                cls._torch_calibration = _CUDA.get(torch.version.cuda, 0)
            gpu = 0.0
            for i in range(cls._n_gpu):
                reserved = torch.cuda.memory_reserved(i)
                if reserved > 0:
                    gpu += (reserved / _MIB) + cls._torch_calibration
            return {pid: gpu}
        return {}

    @classmethod
    @cfg.check(cfg.Usage)
    def _serialize(cls):
        usage = defaultdict(defaultdict[dict])
        for node in set(cls._checkpoints).intersection(cls._records):
            usage[node[0]][node[1]] = {
                "checkpoints": cls._checkpoints[node],
                "records": cls._records[node],
            }
        return {"usage": usage, "process": cls._processes}

    @classmethod
    def serialize(cls):
        return json.dumps(cls._serialize()).encode()


@cfg.check(cfg.Usage)
def setup_reporter():
    Usage.start()


@cfg.check(cfg.Usage)
def setup_monitor():
    Usage.start()
    Recorder.to_dump(cfg.Usage.file, Usage.serialize)
