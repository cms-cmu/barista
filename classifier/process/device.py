from __future__ import annotations

import logging
from typing import Optional

import torch


class Device:
    def __init__(
        self,
        *devices: str,
    ):
        self._devices: dict[str, Optional[set[int]]] = {}

        for device in devices:
            try:
                d = torch.device(device)
            except RuntimeError as e:
                logging.error(e, exc_info=e)
                continue
            if d.type not in self._devices:
                match d.type:
                    case "cuda":
                        self._devices[d.type] = set()
                    case "cpu":
                        self._devices[d.type] = None
                    case _:
                        logging.error(f'"{d.type}" is not supported')
            if self._devices.get(d.type) is not None:
                if d.index is None:
                    self._devices[d.type] = None
                else:
                    self._devices[d.type].add(d.index)

        if not self._devices:
            raise ValueError("No valid device found")

    def _balance_cuda_device(self, *indices: int, min_memory):
        ram = [(i, torch.cuda.mem_get_info(f"cuda:{i}")[0]) for i in indices]
        ram = list(filter(lambda x: x[1] >= min_memory, ram))
        if not ram:
            return None, None
        else:
            return sorted(ram, key=lambda x: x[1])[0]

    def get(self, cuda_min_memory: int = 0) -> torch.device:
        if "cuda" in self._devices:
            if torch.cuda.is_available():
                indices = self._devices["cuda"]
                count = torch.cuda.device_count()
                if indices is None:
                    available = list(range(count))
                else:
                    available = list(filter(lambda x: x < count, indices))
                    if len(available) == 0:
                        logging.warning(
                            f"Only {count} CUDA devices available on this system, got indices {list(sorted(indices))}"
                        )
                if len(available) > 0:
                    i, mem = self._balance_cuda_device(
                        *available, min_memory=cuda_min_memory
                    )
                    if i is not None:
                        d = torch.device(f"cuda:{i}")
                        stat = torch.cuda.get_device_properties(d)
                        logging.info(
                            f"Found {count} CUDA devices, using device {i} {stat.name} with {mem//1024**2}/{stat.total_memory//1024**2} MiB free/total memory"
                        )
                        return d
                    else:
                        logging.warning(
                            f"No CUDA device with at least {self.cuda_min_memory//1024**2} MiB of memory available"
                        )
        if "cpu" in self._devices:
            return torch.device("cpu")
        raise ValueError("No valid device found")
