from __future__ import annotations

import torch
from torch.utils.data import StackDataset

from src.classifier.config.main.train import _share_dataset_memory
from src.classifier.nn.dataset.sliceable import NamedTensorDataset, SliceLoaderLite


def test_sliceloader_prefetch_matches_sync_deterministically():
    ds = NamedTensorDataset(
        x=torch.arange(100, dtype=torch.float32).unsqueeze(1),
        y=torch.arange(100, dtype=torch.int64),
    )

    for shuffle in (False, True):
        for drop_last in (False, True):
            sync_loader = SliceLoaderLite(
                ds, batch_size=16, shuffle=shuffle, drop_last=drop_last, seed=42, prefetch=0
            )
            async_loader = SliceLoaderLite(
                ds, batch_size=16, shuffle=shuffle, drop_last=drop_last, seed=42, prefetch=2
            )
            sync_batches = list(sync_loader)
            async_batches = list(async_loader)
            assert len(sync_batches) == len(async_batches) == len(async_loader)
            for b_sync, b_async in zip(sync_batches, async_batches):
                assert torch.equal(b_sync["x"], b_async["x"])
                assert torch.equal(b_sync["y"], b_async["y"])


def test_sliceloader_prefetch_early_break():
    ds = NamedTensorDataset(x=torch.arange(64, dtype=torch.float32))
    loader = SliceLoaderLite(ds, batch_size=8, shuffle=True, seed=7, prefetch=2)
    it = iter(loader)
    first = next(it)
    assert len(first["x"]) == 8
    del it


def test_share_dataset_memory_marks_stack_tensors_shared():
    ds_a = NamedTensorDataset(a=torch.randn(10, 3))
    ds_b = NamedTensorDataset(b=torch.randint(0, 5, (10,)))
    stack = StackDataset(a=ds_a, b=ds_b)

    _share_dataset_memory(stack)

    assert ds_a.datasets["a"].is_shared()
    assert ds_b.datasets["b"].is_shared()
