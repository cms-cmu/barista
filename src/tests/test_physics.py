"""Unit tests for src.physics selection and weighting helpers."""

import json
import os
import tempfile

import awkward as ak
import numpy as np

from src.physics.common import mask_event_decision
from src.physics.event_selection import apply_event_selection
from src.physics.event_weights import add_weights


def _make_events(n=4):
    return ak.Array(
        {
            "run": np.full(n, 315257, dtype=np.int32),
            "luminosityBlock": np.array([1, 2, 10, 20], dtype=np.int32),
            "HLT": ak.zip(
                {
                    "TrigA": np.array([True, False, False, True]),
                    "TrigB": np.array([False, True, False, False]),
                }
            ),
            "Flag": ak.zip(
                {
                    "Filter1": np.array([True, True, False, True]),
                    "Filter2": np.array([True, False, True, True]),
                }
            ),
            "genWeight": np.array([1.0, -1.0, 2.0, 1.0], dtype=np.float32),
        }
    )


def test_mask_event_decision_or_and_skip():
    events = _make_events()
    or_mask = mask_event_decision(
        events, decision="OR", branch="HLT", list_to_mask=["TrigA", "TrigB"]
    )
    assert np.array_equal(or_mask, np.array([True, True, False, True]))

    and_mask = mask_event_decision(
        events,
        decision="AND",
        branch="Flag",
        list_to_mask=["Filter1", "Filter2", "OptionalMissing"],
        list_to_skip=["OptionalMissing"],
    )
    assert np.array_equal(and_mask, np.array([True, False, False, True]))


def test_mask_event_decision_missing_branch_returns_false():
    events = _make_events()
    mask = mask_event_decision(
        events, decision="OR", branch="NonExistent", list_to_mask=["Foo"]
    )
    assert mask.dtype == bool
    assert not np.any(mask)


def test_apply_event_selection_with_golden_json():
    events = ak.with_parameter(
        _make_events(), "metadata", {"trigger": ["TrigA"]}
    )
    # Use a minimal Record-like metadata wrapper via ak.Array behavior?
    # apply_event_selection reads event.metadata.get(...), so attach a SimpleNamespace/dict.
    events.layout.parameters["__record__"] = "Event"

    with tempfile.TemporaryDirectory() as tmpdir:
        golden = os.path.join(tmpdir, "golden.json")
        with open(golden, "w") as f:
            json.dump({"315257": [[1, 5]]}, f)

        # Wrap in a lightweight proxy that provides .metadata alongside awkward fields
        class _EventsProxy:
            def __init__(self, arr, meta):
                self._arr = arr
                self.metadata = meta

            def __getattr__(self, name):
                return getattr(self._arr, name)

            def __getitem__(self, key):
                return self._arr[key]

            def __setitem__(self, key, val):
                self._arr[key] = val

            def __len__(self):
                return len(self._arr)

            @property
            def fields(self):
                return self._arr.fields

        proxy = _EventsProxy(ak.Array(_make_events()), {"trigger": ["TrigA"]})
        out = apply_event_selection(
            proxy,
            {"goldenJSON": golden, "NoiseFilter": ["Filter1", "Filter2"]},
            cut_on_lumimask=True,
        )
        assert np.array_equal(np.asarray(out["lumimask"]), [True, True, False, False])
        assert np.array_equal(np.asarray(out["passHLT"]), [True, False, False, True])
        assert np.array_equal(np.asarray(out["passNoiseFilter"]), [True, False, False, True])


def test_add_weights_data_and_mc_genweight():
    class _EventsProxy:
        def __init__(self, arr, meta):
            self._arr = arr
            self.metadata = meta

        def __getattr__(self, name):
            return getattr(self._arr, name)

        def __len__(self):
            return len(self._arr)

        @property
        def fields(self):
            return self._arr.fields

    arr = _make_events()
    data_proxy = _EventsProxy(arr, {})
    w_data, names_data = add_weights(
        data_proxy,
        do_MC_weights=False,
        corrections_metadata={},
        apply_trigWeight=False,
    )
    assert names_data == ["data"]
    assert np.allclose(w_data.weight(), np.ones(len(arr)))

    mc_proxy = _EventsProxy(
        arr,
        {"lumi": 10.0, "xs": 2.0, "kFactor": 0.5, "genEventSumw": 5.0},
    )
    w_mc, names_mc = add_weights(
        mc_proxy,
        do_MC_weights=True,
        corrections_metadata={},
        apply_trigWeight=False,
        run_systematics=False,
    )
    assert names_mc == ["genweight"]
    # scale = 10.0 * 2.0 * 0.5 / 5.0 = 2.0
    assert np.allclose(w_mc.weight(), np.array([2.0, -2.0, 4.0, 2.0]))
