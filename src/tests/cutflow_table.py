#!/usr/bin/env python3
"""Tests for src/tools/cutflow_table.py dataset-name parsing.

parse_dataset_name decides which year bucket every cutflow entry lands in, so a name it fails to
parse does not raise -- it falls through to ("Unknown") and quietly disappears from the closure
table. That is how the first Run 3 production came to show two year tables instead of four, with
2023_BPix listing ttbar and no data: the year alternation covered (_preEE|_postEE|_BPix) only, so
2022_EE and 2023_preBPix never matched, and the era group was a bare [A-H] so the digit-carrying
Run 3 eras (C01, C3, D1) did not match either.
"""

import unittest

from src.tools.cutflow_table import parse_dataset_name


class TestParseDatasetNameRun3(unittest.TestCase):
    """Every year and era actually present in metadata/datasets/data.yml for Run 3."""

    def test_data_eras(self):
        cases = {
            "data_2022_preEEB": ("data", "2022_preEE", "2022_preEE_B"),
            "data_2022_preEEC": ("data", "2022_preEE", "2022_preEE_C"),
            "data_2022_preEED": ("data", "2022_preEE", "2022_preEE_D"),
            "data_2022_EEE": ("data", "2022_EE", "2022_EE_E"),
            "data_2022_EEF": ("data", "2022_EE", "2022_EE_F"),
            "data_2022_EEG": ("data", "2022_EE", "2022_EE_G"),
            "data_2023_preBPixC01": ("data", "2023_preBPix", "2023_preBPix_C01"),
            "data_2023_preBPixC02": ("data", "2023_preBPix", "2023_preBPix_C02"),
            "data_2023_preBPixC11": ("data", "2023_preBPix", "2023_preBPix_C11"),
            "data_2023_preBPixC12": ("data", "2023_preBPix", "2023_preBPix_C12"),
            "data_2023_preBPixC3": ("data", "2023_preBPix", "2023_preBPix_C3"),
            "data_2023_preBPixC4": ("data", "2023_preBPix", "2023_preBPix_C4"),
            "data_2023_BPixD1": ("data", "2023_BPix", "2023_BPix_D1"),
            "data_2023_BPixD2": ("data", "2023_BPix", "2023_BPix_D2"),
        }
        for name, expected in cases.items():
            with self.subTest(name):
                self.assertEqual(parse_dataset_name(name), expected)

    def test_mc_per_year(self):
        for proc in ("TTToHadronic", "TTToSemiLeptonic", "TTTo2L2Nu"):
            for year in ("2022_preEE", "2022_EE", "2023_preBPix", "2023_BPix"):
                with self.subTest(f"{proc}_{year}"):
                    self.assertEqual(parse_dataset_name(f"{proc}_{year}"), (proc, year, year))

    def test_no_year_is_lost(self):
        """The whole point: all four years survive, none lands in Unknown."""
        names = [f"data_2022_preEE{e}" for e in ("B", "C", "D")]
        names += [f"data_2022_EE{e}" for e in ("E", "F", "G")]
        names += [f"data_2023_preBPix{e}" for e in ("C01", "C02", "C11", "C12", "C3", "C4")]
        names += [f"data_2023_BPix{e}" for e in ("D1", "D2")]
        years = {parse_dataset_name(n)[1] for n in names}
        self.assertEqual(years, {"2022_preEE", "2022_EE", "2023_preBPix", "2023_BPix"})
        self.assertNotIn("Unknown", years)
        # and every one is still recognised as data, not a mangled process name
        self.assertEqual({parse_dataset_name(n)[0] for n in names}, {"data"})

    def test_double_underscore(self):
        self.assertEqual(
            parse_dataset_name("data__2023_preBPix_C01"), ("data", "2023_preBPix", "2023_preBPix_C01")
        )
        self.assertEqual(
            parse_dataset_name("TTToHadronic__2022_EE"), ("TTToHadronic", "2022_EE", "2022_EE")
        )

    def test_2024_bare_year(self):
        self.assertEqual(parse_dataset_name("TTToHadronic_2024"), ("TTToHadronic", "2024", "2024"))


class TestParseDatasetNameRun2(unittest.TestCase):
    """Run 2 must keep working -- the Run 3 fix shares the same regexes."""

    def test_ul_years(self):
        cases = {
            "data_UL16_preVFPC": ("data", "UL16_preVFP", "UL16_preVFPC"),
            "data_UL16_postVFPF": ("data", "UL16_postVFP", "UL16_postVFPF"),
            "data_UL17C": ("data", "UL17", "UL17C"),
            "data_UL18A": ("data", "UL18", "UL18A"),
            "ttHbb_UL16_preVFP": ("ttHbb", "UL16_preVFP", "UL16_preVFP"),
            "TTToHadronic_UL18": ("TTToHadronic", "UL18", "UL18"),
        }
        for name, expected in cases.items():
            with self.subTest(name):
                self.assertEqual(parse_dataset_name(name), expected)

    def test_merge_ttbar(self):
        self.assertEqual(parse_dataset_name("TTToHadronic_UL18", merge_ttbar=True)[0], "TTbar")
        self.assertEqual(parse_dataset_name("TTTo2L2Nu_2023_BPix", merge_ttbar=True)[0], "TTbar")
        self.assertEqual(parse_dataset_name("data_UL18A", merge_ttbar=True)[0], "data")

    def test_unparseable_falls_back(self):
        self.assertEqual(parse_dataset_name("something_odd"), ("something_odd", "Unknown", "something_odd"))


if __name__ == "__main__":
    unittest.main()
