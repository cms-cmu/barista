"""A ratio whose denominator is ONE stack component (`type: stack` + `key`).

Needed to ratio an overlay to the part of the stack it should follow -- e.g. ttbar pseudodata to
the ttbar MC component of a multijet + ttbar stack -- in the same plot where data is ratioed to
the whole stack. Without a key the stack is summed, as before.
"""
import numpy as np

from src.plotting.helpers_make_plot import _resolve_hist_source
from src.plotting.helpers_make_plot_dict import HistSource


def _entry(values):
    values = np.asarray(values, float)
    return {"values": values.tolist(), "variances": values.tolist(), "centers": [0.5, 1.5]}


PLOT_DATA = {
    "hists": {"psdata": _entry([2.0, 3.0])},
    "stack": {"Multijet": _entry([10.0, 20.0]), "TTbar": _entry([1.0, 2.0])},
}


def test_stack_without_key_is_summed():
    values, variances, centers, _ = _resolve_hist_source(HistSource(source="stack", key=None), PLOT_DATA)
    assert values.tolist() == [11.0, 22.0]
    assert variances.tolist() == [11.0, 22.0]
    assert centers == [0.5, 1.5]


def test_stack_with_key_is_one_component():
    values, variances, _, entry = _resolve_hist_source(HistSource(source="stack", key="TTbar"), PLOT_DATA)
    assert values.tolist() == [1.0, 2.0]
    assert variances.tolist() == [1.0, 2.0]
    assert entry is PLOT_DATA["stack"]["TTbar"]


def test_hists_source_unchanged():
    values, _, _, _ = _resolve_hist_source(HistSource(source="hists", key="psdata"), PLOT_DATA)
    assert values.tolist() == [2.0, 3.0]
