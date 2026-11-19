"""Tests of the b-tagging SF calibration: derivation, file format, evaluation and closure.

    ./run_container python -m src.tests.btag_calibration
"""

import os
import tempfile
import unittest

import hist
import numpy as np

from src.physics.btag_calibration import (
    btag_sf_calibration_samples,
    build_btag_sf_calibration,
    evaluate_btag_sf_calibration,
    load_btag_sf_calibration,
)
from src.tools.compute_btagSF_calibration import CalibrationInput, compute_ratio, fold_flow, validate, write_calibration

NJET_EDGES = [4, 5, 6, 8, 20]
HT_EDGES = [0, 500, 1000, 5000]
VARIATIONS = ["central", "up_hf", "down_hf"]


def _toy_hists(n=20000, seed=1):
    rng = np.random.default_rng(seed)
    njet = rng.integers(4, 25, n)  # includes overflow (>= 20)
    ht = rng.uniform(100, 6000, n)  # includes overflow (>= 5000)
    w = rng.uniform(0.5, 1.5, n)
    x = rng.uniform(0, 1, n)
    sf = {
        "central": 0.8 + 0.02 * njet + 1e-4 * ht * x,
        "up_hf": 0.9 + 0.02 * njet + 1e-4 * ht * x,
        "down_hf": 0.7 + 0.02 * njet + 1e-4 * ht * x,
    }

    def cats():
        return (
            hist.axis.StrCategory([], name="process", growth=True),
            hist.axis.StrCategory([], name="year", growth=True),
            hist.axis.StrCategory([], name="variation", growth=True),
        )

    def map_axes():
        return (hist.axis.Variable(NJET_EDGES, name="calib_nJet"), hist.axis.Variable(HT_EDGES, name="calib_HT"))

    hists = {
        "Njet_HT_noBtagSF": hist.Hist(*cats(), *map_axes(), storage="weight"),
        "Njet_HT_btagSF": hist.Hist(*cats(), *map_axes(), storage="weight"),
        "Njet_HT_btagSFcross": hist.Hist(*cats(), *map_axes(), storage="weight"),
        "x_noBtagSF": hist.Hist(*cats(), hist.axis.Regular(10, 0, 1, name="x"), *map_axes(), storage="weight"),
        "x_btagSF": hist.Hist(*cats(), hist.axis.Regular(10, 0, 1, name="x"), *map_axes(), storage="weight"),
    }
    common = dict(process="TT", year="UL18", calib_nJet=njet, calib_HT=ht)
    hists["Njet_HT_noBtagSF"].fill(variation="central", weight=w, **common)
    hists["Njet_HT_btagSFcross"].fill(variation="central", weight=w * np.sqrt(sf["central"]), **common)
    for v in VARIATIONS:
        hists["Njet_HT_btagSF"].fill(variation=v, weight=w * sf[v], **common)
    hists["x_noBtagSF"].fill(variation="central", x=x, weight=w, **common)
    hists["x_btagSF"].fill(variation="central", x=x, weight=w * sf["central"], **common)
    return hists, dict(njet=njet, ht=ht, w=w, x=x, sf=sf)


class BtagSFCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.hists, cls.events = _toy_hists()
        cls.inp = CalibrationInput(cls.hists, "Njet_HT")
        cls.maps = write_calibration(cls.inp, "UL18", cls.tmp.name, min_neff=0)
        cls.path = os.path.join(cls.tmp.name, "btagSF_calibrationSF_UL18.json")
        cls.text = load_btag_sf_calibration(cls.path)
        cls.correction = build_btag_sf_calibration(cls.text)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fold_flow(self):
        a = np.arange(20.0).reshape(4, 5)
        folded = fold_flow(a)
        self.assertEqual(folded.shape, (2, 3))
        self.assertEqual(folded.sum(), a.sum())

    def test_ratio_empty_bins(self):
        ratio, err = compute_ratio(np.array([0.0, 2.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        np.testing.assert_array_equal(ratio, [1.0, 1.0])
        np.testing.assert_array_equal(err, [0.0, 0.0])

    def test_correlated_uncertainty_is_smaller(self):
        num, num_var, den, den_var, cross_var = self.inp.yields("TT", "UL18")
        _, err = compute_ratio(num, num_var, den, den_var, cross_var)
        _, err_uncorr = compute_ratio(num, num_var, den, den_var)
        self.assertTrue(np.all(err < err_uncorr))

    def test_samples(self):
        self.assertEqual(btag_sf_calibration_samples(self.text), ["TT"])

    def test_evaluation_matches_maps(self):
        njet = np.array([4, 5, 7, 12])
        ht = np.array([100.0, 600.0, 2000.0, 400.0])
        for v in VARIATIONS:
            expected = self.maps[("TT", v)][0][[0, 1, 2, 3], [0, 1, 2, 0]]
            np.testing.assert_allclose(evaluate_btag_sf_calibration(self.correction, "TT", v, njet, ht), expected)

    def test_clamp(self):
        central = self.maps[("TT", "central")][0]
        out = evaluate_btag_sf_calibration(self.correction, "TT", "central", np.array([30]), np.array([9000.0]))
        np.testing.assert_allclose(out, central[-1, -1])

    def test_unknown_sample(self):
        with self.assertRaises(KeyError):
            evaluate_btag_sf_calibration(self.correction, "missing", "central", np.array([4]), np.array([100.0]))

    def test_split_sample(self):
        njet = np.array([4, 5, 7, 12])
        ht = np.array([100.0, 600.0, 2000.0, 400.0])
        mask = np.array([True, False, True, False])
        expected = evaluate_btag_sf_calibration(self.correction, "TT", "central", njet, ht)
        out = evaluate_btag_sf_calibration(self.correction, {"TT": np.ones(4, bool)}, "central", njet, ht)
        np.testing.assert_allclose(out, expected)
        # masks that do not cover all the events are rejected
        with self.assertRaises(ValueError):
            evaluate_btag_sf_calibration(self.correction, {"TT": mask}, "central", njet, ht)

    def test_event_level_closure(self):
        """Applying the calibration event by event restores the no-SF yield in every map bin."""
        ev = self.events
        for v in VARIATIONS:
            calib = evaluate_btag_sf_calibration(self.correction, "TT", v, ev["njet"], ev["ht"])
            closed = hist.Hist(
                hist.axis.Variable(NJET_EDGES, name="n"), hist.axis.Variable(HT_EDGES, name="ht"), storage="weight"
            )
            closed.fill(n=ev["njet"], ht=ev["ht"], weight=ev["w"] * ev["sf"][v] * calib)
            num, *_ = self.inp.yields("TT", "UL18")
            np.testing.assert_allclose(fold_flow(closed.values(flow=True)), num, rtol=1e-10)

    def test_validate(self):
        summary = validate(self.inp, os.path.join(self.tmp.name, "validation"))
        entry = summary["UL18"]["TT"]
        self.assertNotAlmostEqual(entry["btagSF_over_noBtagSF"], 1.0, places=2)
        self.assertAlmostEqual(entry["btagSF_calib_over_noBtagSF"], 1.0, places=10)
        self.assertTrue(
            os.path.exists(os.path.join(self.tmp.name, "validation", "closure", "UL18", "x", "x_TT_UL18_calibrated.png"))
        )


if __name__ == "__main__":
    unittest.main()
