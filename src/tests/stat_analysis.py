import os
import sys
import unittest
from types import SimpleNamespace

# Add current directory and submit wrapper directory to sys.path
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "software/snakemake/scripts"))

# Mock snakemake modules to avoid dependency errors in non-snakemake environments
from unittest.mock import MagicMock
sys.modules['snakemake'] = MagicMock()
sys.modules['snakemake.utils'] = MagicMock()

from src.stat_analysis.helpers import (
    make_poi_maps,
    get_default_othersignals,
    get_grid_split_points,
    get_likelihood_scan_chunks
)
import submit_wrapper


class TestStatAnalysisHelpers(unittest.TestCase):

    def test_make_poi_maps_string(self):
        signals = "sigA sigB"
        res = make_poi_maps(signals, "1,-10,10")
        self.assertEqual(res, "--PO 'map=.*/sigA:rsigA[1,-10,10]' --PO 'map=.*/sigB:rsigB[1,-10,10]'")

    def test_make_poi_maps_dict(self):
        signals = {"sigA": "1,-5,5", "sigB": "1,-10,10"}
        res = make_poi_maps(signals)
        self.assertEqual(res, "--PO 'map=.*/sigA:rsigA[1,-5,5]' --PO 'map=.*/sigB:rsigB[1,-10,10]'")

    def test_make_poi_maps_list(self):
        signals = ["sigA", "sigB"]
        res = make_poi_maps(signals, ["1,-5,5", "1,-10,10"])
        self.assertEqual(res, "--PO 'map=.*/sigA:rsigA[1,-5,5]' --PO 'map=.*/sigB:rsigB[1,-10,10]'")

    def test_get_default_othersignals(self):
        config = {
            "channels": {
                "HH4b": {
                    "signallabel": "ggHH",
                    "othersignal": "sigA sigB"
                }
            }
        }
        # 1. By wildcards.path containing channel
        wildcards_path = SimpleNamespace(path="output/v4/HH4b/workspace/datacard")
        self.assertEqual(get_default_othersignals(wildcards_path, config), ["sigA", "sigB"])

        # 2. By wildcards.signallabel
        wildcards_label = SimpleNamespace(signallabel="ggHH")
        self.assertEqual(get_default_othersignals(wildcards_label, config), ["sigA", "sigB"])

    def test_get_grid_split_points(self):
        config = {
            "likelihood_scan_points": "50",
            "likelihood_scan_split_size": "10"
        }
        wildcards = SimpleNamespace(split_index="0")
        self.assertEqual(get_grid_split_points(wildcards, config), (0, 9))

        wildcards = SimpleNamespace(split_index="4")
        self.assertEqual(get_grid_split_points(wildcards, config), (40, 49))

    def test_get_likelihood_scan_chunks(self):
        config = {
            "likelihood_scan_points": "25",
            "likelihood_scan_split_size": "10"
        }
        wildcards = SimpleNamespace(path="output/scan", signallabel="ggHH")
        chunks = get_likelihood_scan_chunks(wildcards, config)
        self.assertEqual(chunks, [
            "output/scan_likelihood_scan_chunk_0__ggHH.root",
            "output/scan_likelihood_scan_chunk_1__ggHH.root",
            "output/scan_likelihood_scan_chunk_2__ggHH.root"
        ])


class TestSubmitWrapper(unittest.TestCase):

    def test_get_output_base_dir(self):
        job_props = {"log": ["output/v4_systematics_test/HH4b/logs/workspace.log"]}
        self.assertEqual(submit_wrapper.get_output_base_dir(job_props), "output/v4_systematics_test/HH4b")

        job_props_out = {"output": ["output/v4/HH4b/workspace/datacard.root"]}
        self.assertEqual(submit_wrapper.get_output_base_dir(job_props_out), "output/v4/HH4b")

    def test_format_template(self):
        template = "run --input {input} --output {output} --sig {params.signallabel}"
        job_props = {
            "rule": "test_rule",
            "input": ["in.root"],
            "output": ["out.root"],
            "params": {"signallabel": "ggHH"},
            "wildcards": {},
            "log": ["log.txt"],
            "threads": 1
        }
        res = submit_wrapper.format_template(template, job_props)
        self.assertEqual(res, "run --input in.root --output out.root --sig ggHH")


if __name__ == "__main__":
    unittest.main()
