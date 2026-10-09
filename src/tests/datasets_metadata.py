#!/usr/bin/env python3
"""Tests for src/runner/dataset.py:load_datasets_metadata (runner.py -m).

`-m` used to be one local path, so the only way to make a new dataset visible to runner.py was to
write its YAML into coffea4bees/metadata/datasets/. It now takes several sources, local or remote
(fsspec), so a consumer can read a dataset another roast published to EOS in place. The risk that
comes with merging is two productions of one dataset key -- a stale mixeddata_4b.yml in the
default directory and a roast's published one -- combining silently. These tests pin the rule:
fields merge, a field defined differently in two sources raises.
"""

import os
import tempfile
import unittest

import yaml

from src.runner.dataset import load_datasets_metadata

DEFAULT_DIR = "coffea4bees/metadata/datasets/"


def _write(dirname, name, data):
    path = os.path.join(dirname, name)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return path


def _entry(year, files, n_samples=None):
    e = {year: {"picoAOD": {"files_template": files}}}
    if n_samples is not None:
        e["nSamples"] = n_samples
    return e


class TestSingleSource(unittest.TestCase):

    def test_string_and_list_are_equivalent(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(d, "a.yml", {"mixeddata_4b": _entry("2022_EE", ["a_vXXX.root"], 16)})
            self.assertEqual(load_datasets_metadata(p), load_datasets_metadata([p]))

    def test_file_with_datasets_wrapper(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(d, "a.yml", {"datasets": {"data": _entry("2022_EE", ["d.root"])}})
            self.assertEqual(list(load_datasets_metadata(p)["datasets"]), ["data"])

    def test_directory_merges_its_files(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, "a.yml", {"data": _entry("2022_EE", ["d.root"])})
            _write(d, "b.yml", {"TTToHadronic": _entry("2022_EE", ["t.root"])})
            _write(d, "notes.txt", {"ignored": 1})
            self.assertEqual(sorted(load_datasets_metadata(d)["datasets"]), ["TTToHadronic", "data"])

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                load_datasets_metadata(d)

    def test_file_url_goes_through_fsspec(self):
        """A URL, not a bare path: the remote code path, exercised without EOS."""
        with tempfile.TemporaryDirectory() as d:
            p = _write(d, "a.yml", {"mixeddata_all": _entry("2022_EE", ["m.root"])})
            out = load_datasets_metadata(f"file://{p}")
            self.assertIn("mixeddata_all", out["datasets"])


class TestMultipleSources(unittest.TestCase):

    def test_disjoint_datasets(self):
        with tempfile.TemporaryDirectory() as d:
            a = _write(d, "a.yml", {"data": _entry("2022_EE", ["d.root"])})
            b = _write(d, "b.yml", {"mixeddata_all": _entry("2022_EE", ["m.root"])})
            self.assertEqual(sorted(load_datasets_metadata([a, b])["datasets"]), ["data", "mixeddata_all"])

    def test_one_dataset_split_across_runs(self):
        """Run 2 and Run 3 productions of one key contribute different years."""
        with tempfile.TemporaryDirectory() as d:
            a = _write(d, "r2.yml", {"mixeddata_4b": _entry("UL18", ["r2_vXXX.root"], 16)})
            b = _write(d, "r3.yml", {"mixeddata_4b": _entry("2022_EE", ["r3_vXXX.root"], 16)})
            ds = load_datasets_metadata([a, b])["datasets"]["mixeddata_4b"]
            self.assertEqual(sorted(k for k in ds if k != "nSamples"), ["2022_EE", "UL18"])
            self.assertEqual(ds["nSamples"], 16)

    def test_identical_redefinition_is_fine(self):
        with tempfile.TemporaryDirectory() as d:
            a = _write(d, "a.yml", {"data": _entry("2022_EE", ["d.root"])})
            b = _write(d, "b.yml", {"data": _entry("2022_EE", ["d.root"])})
            self.assertIn("data", load_datasets_metadata([a, b])["datasets"])

    def test_conflicting_nsamples_raises(self):
        with tempfile.TemporaryDirectory() as d:
            a = _write(d, "a.yml", {"mixeddata_4b": _entry("UL18", ["r2_vXXX.root"], 15)})
            b = _write(d, "b.yml", {"mixeddata_4b": _entry("2022_EE", ["r3_vXXX.root"], 16)})
            with self.assertRaises(ValueError) as cm:
                load_datasets_metadata([a, b])
            self.assertIn("mixeddata_4b.nSamples", str(cm.exception))
            self.assertIn(a, str(cm.exception))
            self.assertIn(b, str(cm.exception))

    def test_conflicting_year_raises(self):
        """Two productions of the same dataset-year: never merged silently."""
        with tempfile.TemporaryDirectory() as d:
            a = _write(d, "old.yml", {"mixeddata_all": _entry("2022_EE", ["old.root"])})
            b = _write(d, "new.yml", {"mixeddata_all": _entry("2022_EE", ["new.root"])})
            with self.assertRaises(ValueError) as cm:
                load_datasets_metadata([a, b])
            self.assertIn("mixeddata_all.2022_EE", str(cm.exception))

    def test_inputs_not_mutated(self):
        with tempfile.TemporaryDirectory() as d:
            a = _write(d, "a.yml", {"mixeddata_4b": _entry("UL18", ["x"], 16)})
            b = _write(d, "b.yml", {"mixeddata_4b": _entry("2022_EE", ["y"], 16)})
            first = load_datasets_metadata([a])
            load_datasets_metadata([a, b])
            self.assertEqual(first, load_datasets_metadata([a]))


@unittest.skipUnless(os.path.isdir(DEFAULT_DIR), "run from the barista root")
class TestDefaultDirectoryUnchanged(unittest.TestCase):
    """`-m coffea4bees/metadata/datasets/` must give exactly what the old code gave."""

    def test_matches_legacy_loader(self):
        from omegaconf import OmegaConf
        files = [OmegaConf.load(os.path.join(DEFAULT_DIR, f)) for f in os.listdir(DEFAULT_DIR)
                 if f.endswith(('.yaml', '.yml'))]
        legacy = OmegaConf.to_container(OmegaConf.create({'datasets': OmegaConf.merge(*files)}), resolve=True)
        self.assertEqual(load_datasets_metadata(DEFAULT_DIR), legacy)


if __name__ == "__main__":
    unittest.main()
