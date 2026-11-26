from __future__ import annotations

from functools import partial

import awkward as ak
import awkward_pandas as akpd
import numpy as np
import pandas as pd
import pytest
import torch

from src.classifier.config.dataset._root import _load_chunk, _load_root
from src.classifier.df.io import ToTensor


def _make_sample_df(n: int, seed: int, max_notcan: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, max_notcan + 1, size=n)
    notcan_pt = ak.Array([rng.uniform(20.0, 100.0, size=int(c)).tolist() for c in counts])
    notcan_eta = ak.Array([rng.uniform(-2.4, 2.4, size=int(c)).tolist() for c in counts])

    can_pt = ak.Array([rng.uniform(40.0, 150.0, size=4).tolist() for _ in range(n)])

    return pd.DataFrame(
        {
            "event": rng.integers(1, 10**9, size=n, dtype=np.uint64),
            "label_index": rng.integers(0, 4, size=n, dtype=np.int64),
            "region_index": rng.choice([1, 2], size=n).astype(np.int64),
            "weight": rng.uniform(0.1, 2.0, size=n).astype(np.float32),
            "xW": rng.uniform(0.0, 3.0, size=n).astype(np.float32),
            "year": np.full(n, 240, dtype=np.int64),
            "year_2024": np.full(n, 1.0, dtype=np.float32),
            "year_2022_preEE": np.full(n, 0.0, dtype=np.float32),
            "CanJet_pt": akpd.from_awkward(can_pt),
            "NotCanJet_pt": akpd.from_awkward(notcan_pt),
            "NotCanJet_eta": akpd.from_awkward(notcan_eta),
        }
    )


def _make_to_tensor(fixed_target: bool = True) -> ToTensor:
    tt = ToTensor()
    (
        tt.add("event", "uint64")
        .columns("event")
        .add("label", "int64")
        .columns("label_index")
        .add("region", "int64")
        .columns("region_index")
        .add("weight", "float32")
        .columns("weight")
        .add("ancillary", "float32")
        .columns("xW", "year_2024", "year_2022_preEE")
        .add("CanJet", "float32")
        .columns("CanJet_pt", target=4)
        .add("NotCanJet", "float32")
        .columns(
            "NotCanJet_pt",
            "NotCanJet_eta",
            target=6 if fixed_target else None,
            pad_value=-1.0,
        )
    )
    return tt


class _FakeFromRoot:
    def __init__(self, mapping: dict[int, pd.DataFrame | None]):
        self._mapping = mapping

    def __call__(self, chunk: int):
        df = self._mapping[chunk]
        return df.copy() if df is not None else None


@pytest.mark.parametrize("fixed_target", [True, False])
def test_streaming_concat_matches_baseline_bit_for_bit(fixed_target: bool):
    df0 = _make_sample_df(12, seed=1, max_notcan=2)
    df1 = _make_sample_df(15, seed=2, max_notcan=5)

    tt_base = _make_to_tensor(fixed_target=fixed_target)
    full_df = pd.concat([df0, df1], ignore_index=True)
    expected = tt_base.tensor(full_df)

    tt_stream = _make_to_tensor(fixed_target=fixed_target)
    loader0 = _load_chunk(_FakeFromRoot({0: df0}), tt_stream, keep_meta=False)
    loader1 = _load_chunk(_FakeFromRoot({1: df1}), tt_stream, keep_meta=False)

    r0 = loader0(0)
    r1 = loader1(1)
    assert r0 is not None and r1 is not None
    assert r0[2] is None and r1[2] is None

    actual = tt_stream.concat_chunks([r0[1], r1[1]])
    assert set(actual.keys()) == set(expected.keys())
    for k in expected:
        assert actual[k].dtype == expected[k].dtype
        assert actual[k].shape == expected[k].shape
        assert torch.equal(actual[k], expected[k]), f"Mismatch on {k}"


def test_streaming_with_postprocessors_yields_and_row_filter():
    df0 = _make_sample_df(10, seed=10)
    df1 = _make_sample_df(10, seed=11)
    # Mark two rows negative weight to verify row filtering
    df0.loc[2, "weight"] = -1.0
    df1.loc[5, "weight"] = -1.0

    tt_base = _make_to_tensor()
    tt_stream = _make_to_tensor()

    seen_years = {}

    def normalize(df: pd.DataFrame, norm: float) -> pd.DataFrame:
        df = df[df["weight"] > 0]
        df.loc[:, "weight"] = df["weight"] / (df["weight"].sum() / norm)
        seen_years.update(df.groupby("year")["weight"].sum().to_dict())
        return df

    post = partial(normalize, norm=3.0)

    # Baseline path
    full_df = post(pd.concat([df0, df1], ignore_index=True))
    expected = tt_base.tensor(full_df)

    # Streaming path
    r0 = _load_chunk(_FakeFromRoot({0: df0}), tt_stream, keep_meta=True)(0)
    r1 = _load_chunk(_FakeFromRoot({1: df1}), tt_stream, keep_meta=True)(1)
    assert "year_2024" not in r0[2].columns
    assert "NotCanJet_pt" not in r0[2].columns
    assert "year" in r0[2].columns

    meta_df = pd.concat([r0[2], r1[2]], ignore_index=True)
    meta_df = post(meta_df)
    keep_idx = meta_df.index.to_numpy()
    actual = tt_stream.concat_chunks([r0[1], r1[1]], keep_idx=keep_idx)
    tt_stream.refresh_from_df(actual, meta_df)

    assert seen_years[240] == pytest.approx(3.0, rel=1e-5)
    for k in expected:
        assert torch.allclose(actual[k].float(), expected[k].float())
        assert actual[k].dtype == expected[k].dtype


def test_to_numpy_release_frees_columns_and_empty_chunks():
    df = _make_sample_df(5, seed=7)
    tt = _make_to_tensor()
    out = tt.tensor(df, release=True)
    assert list(df.columns) == ["year"]
    assert out["CanJet"].shape == (5, 4)

    empty_df = df.iloc[0:0].copy()
    assert _load_chunk(_FakeFromRoot({0: empty_df}), tt, keep_meta=False)(0) is None
    assert _load_chunk(_FakeFromRoot({0: None}), tt, keep_meta=False)(0) is None


def test_load_root_zero_events_raises(monkeypatch):
    class _DummyChunk:
        def __len__(self):
            return 0

    monkeypatch.setattr("src.data_formats.root.Chunk.balance", lambda *_a, **_kw: [])
    loader = _load_root(
        (_FakeFromRoot({}), []),
        max_workers=1,
        chunksize=100,
        tree="Events",
        preserve_order=False,
    )
    loader.to_tensor = _make_to_tensor()
    loader.postprocessors = []
    with pytest.raises(ValueError, match="Dataset loaded 0 events"):
        loader()
