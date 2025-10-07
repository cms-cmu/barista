import gc

import torch

BatchType = dict[str, torch.Tensor]


def batch_size(batch: BatchType) -> int:
    sizes = {len(v) for v in batch.values()}
    if len(sizes) > 1:
        sizes = {k: len(v) for k, v in batch.items()}
        raise ValueError(f"Inconsistent batch sizes {sizes}")
    return sizes.pop()


def clear_cache(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
