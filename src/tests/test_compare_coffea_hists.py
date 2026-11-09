#!/usr/bin/env python
"""Unit tests for src/tools/compare_coffea_hists.py."""
import os
import pickle
import sys
import tempfile
import unittest
import numpy as np

# Add repo root to path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.tools.compare_coffea_hists import (
    _find_var_axis,
    _project_1d,
    compare_hist_objects,
    run_comparison,
    generate_text_report,
    generate_html_report,
)


class MockAxis:
    def __init__(self, name, values_list=None):
        self.name = name
        self._values = values_list or []

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


class MockHist1D:
    """Mock projected 1D histogram."""
    def __init__(self, values):
        self._vals = np.asarray(values, dtype=float)

    def values(self):
        return self._vals


class MockHistMultiD:
    """Mock multi-dimensional histogram supporting dict-slicing and .project()."""
    def __init__(self, axes, data_dict):
        self.axes = axes
        self.data_dict = data_dict  # (proc, tag, reg) -> 1D numpy array

    def __getitem__(self, loc_dict):
        proc = loc_dict.get("process")
        tag = loc_dict.get("tag")
        reg = loc_dict.get("region")
        key = (proc, tag, reg)
        if key not in self.data_dict:
            raise KeyError(f"Key {key} not found")
        # Return a sliced object that can project
        var_axis_name = self.axes[-1].name
        return MockSlicedHist(var_axis_name, self.data_dict[key])


class MockSlicedHist:
    def __init__(self, var_name, values):
        self.axes = [MockAxis(var_name)]
        self._vals = values

    def project(self, var_name):
        return MockHist1D(self._vals)


class TestCompareCoffeaHists(unittest.TestCase):
    def setUp(self):
        self.axes = [
            MockAxis("process", ["data", "TTbar4b_from_d3"]),
            MockAxis("tag", ["threeTag", "fourTag"]),
            MockAxis("region", ["SB", "SR"]),
            MockAxis("v4j.mass"),
        ]

    def test_find_var_axis(self):
        hist_obj = MockSlicedHist("v4j.mass", [1, 2, 3])
        self.assertEqual(_find_var_axis(hist_obj), "v4j.mass")

    def test_compare_hist_objects_matching(self):
        vals = np.array([10.0, 25.5, 30.2, 5.1])
        data = {
            ("data", "fourTag", "SB"): vals,
            ("data", "threeTag", "SB"): vals * 2,
        }
        h_a = MockHistMultiD(self.axes, data)
        h_b = MockHistMultiD(self.axes, data)

        results = compare_hist_objects(
            h_a, h_b, "v4j.mass",
            processes=["data"],
            tags=["threeTag", "fourTag"],
            regions=["SB"],
            tolerance=1e-4,
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["status"] == "PASS" for r in results))
        self.assertEqual(results[0]["max_diff"], 0.0)

    def test_compare_hist_objects_divergent(self):
        data_a = {("data", "fourTag", "SB"): np.array([10.0, 20.0, 30.0])}
        data_b = {("data", "fourTag", "SB"): np.array([10.0, 25.0, 30.0])}
        h_a = MockHistMultiD(self.axes, data_a)
        h_b = MockHistMultiD(self.axes, data_b)

        results = compare_hist_objects(
            h_a, h_b, "v4j.mass",
            processes=["data"],
            tags=["fourTag"],
            regions=["SB"],
            tolerance=1e-4,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "FAIL")
        self.assertAlmostEqual(results[0]["max_diff"], 5.0)

    def test_run_comparison_file_io(self):
        vals = np.array([5.0, 15.0, 25.0])
        data = {
            ("data", "fourTag", "SB"): vals,
            ("data", "threeTag", "SB"): vals,
        }
        h = MockHistMultiD(self.axes, data)
        dict_a = {"hists": {"v4j.mass": h, "selJets.n": h}}
        dict_b = {"hists": {"v4j.mass": h, "selJets.n": h}}

        with tempfile.NamedTemporaryFile(suffix=".coffea", delete=False) as f_a:
            pickle.dump(dict_a, f_a)
            path_a = f_a.name
        with tempfile.NamedTemporaryFile(suffix=".coffea", delete=False) as f_b:
            pickle.dump(dict_b, f_b)
            path_b = f_b.name

        try:
            passed, results, meta = run_comparison(
                path_a, path_b,
                variables=["v4j.mass", "selJets.n"],
                regions=["SB"],
                tags=["fourTag"],
                tolerance=1e-4,
            )
            self.assertTrue(passed)
            self.assertEqual(len(results), 4)

            txt = generate_text_report(passed, results, meta)
            self.assertTrue(txt.startswith("PASS"))

            html = generate_html_report(passed, results, meta)
            self.assertIn("badge-success", html)
            self.assertIn("v4j.mass", html)
        finally:
            if os.path.exists(path_a):
                os.unlink(path_a)
            if os.path.exists(path_b):
                os.unlink(path_b)


if __name__ == "__main__":
    unittest.main()
