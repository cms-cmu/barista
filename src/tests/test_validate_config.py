#!/usr/bin/env python3
"""Unit tests for src/tools/validate_config.py."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import yaml

from tools.validate_config import (
    Issue,
    ValidationReport,
    audit_datasets_consistency,
    audit_filesystem_references,
    audit_flags_and_duplicates,
    audit_producer_consumer_paths,
    audit_schema_and_typos,
    audit_year_eras_consistency,
    format_report,
    format_summary_matrix,
    resolve_config_chain,
    validate_config,
)


class TestConfigLayeringAndResolution(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.root = Path(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_direct_config_loading(self):
        cfg_file = self.root / "config.yml"
        cfg_file.write_text("analysis: HH4b\nyear: 2024\n")
        data, bases, issues = resolve_config_chain(cfg_file, repo_root=self.root)
        self.assertEqual(data["analysis"], "HH4b")
        self.assertEqual(data["year"], 2024)
        self.assertEqual(len(bases), 0)
        self.assertEqual(len(issues), 0)

    def test_base_inheritance_merging(self):
        base_file = self.root / "base.yml"
        base_file.write_text(
            "analysis: HH4b\n"
            "year_eras:\n"
            "  UL18: [A, B]\n"
            "classifier_setting:\n"
            "  epochs: 20\n"
        )
        child_file = self.root / "child.yml"
        child_file.write_text(
            "base: base.yml\n"
            "year_eras:\n"
            "  UL18: [C, D]\n"
            "classifier_setting:\n"
            "  lr: 0.001\n"
        )
        data, bases, issues = resolve_config_chain(child_file, repo_root=self.root)
        self.assertEqual(data["analysis"], "HH4b")
        self.assertEqual(data["year_eras"]["UL18"], ["C", "D"])
        self.assertEqual(data["classifier_setting"]["epochs"], 20)
        self.assertEqual(data["classifier_setting"]["lr"], 0.001)
        self.assertEqual(len(bases), 1)
        self.assertEqual(len(issues), 0)

    def test_circular_base_detection(self):
        a_file = self.root / "a.yml"
        b_file = self.root / "b.yml"
        a_file.write_text("base: b.yml\nfoo: 1\n")
        b_file.write_text("base: a.yml\nbar: 2\n")

        data, bases, issues = resolve_config_chain(a_file, repo_root=self.root)
        errs = [i for i in issues if i.severity == "ERROR"]
        self.assertTrue(any("Cyclic config inheritance detected" in e.message for e in errs))

    def test_missing_base_detection(self):
        a_file = self.root / "a.yml"
        a_file.write_text("base: nonexistent_base.yml\n")
        data, bases, issues = resolve_config_chain(a_file, repo_root=self.root)
        errs = [i for i in issues if i.severity == "ERROR"]
        self.assertTrue(any("not found" in e.message for e in errs))


class TestDuplicateAndDivergenceAudits(unittest.TestCase):
    def test_dataset_synchronization_detects_missing_signal_in_training(self):
        cfg = {
            "dataset": ["GluGluToHHTo4B_cHHH1", "data"],
            "classifier_inputs": {
                "datasets": ["TTTo2L2Nu"],  # Missing GluGluToHHTo4B_cHHH1
            },
        }
        issues: list[Issue] = []
        audit_datasets_consistency(cfg, issues)
        errs = [i for i in issues if i.severity == "ERROR"]
        self.assertTrue(any("GluGluToHHTo4B_cHHH1" in e.message for e in errs))

    def test_dataset_synchronization_matches_cleanly(self):
        cfg = {
            "dataset": ["GluGluToHHTo4B_cHHH1", "data"],
            "classifier_inputs": {
                "datasets": ["GluGluToHHTo4B_cHHH1", "TTTo2L2Nu"],
            },
        }
        issues: list[Issue] = []
        audit_datasets_consistency(cfg, issues)
        errs = [i for i in issues if i.severity == "ERROR"]
        self.assertEqual(len(errs), 0)

    def test_era_synchronization_divergence(self):
        cfg = {
            "year_eras": {
                "UL17": ["C", "D", "E", "F"],  # Missing B
            },
            "classifier_setting": {
                "option": [
                    {
                        "eras": {
                            "UL17": ["B", "C", "D", "E", "F"],  # Has B
                        }
                    }
                ]
            },
        }
        issues: list[Issue] = []
        audit_year_eras_consistency(cfg, issues)
        self.assertTrue(len(issues) > 0)
        self.assertTrue(any("UL17" in i.message and "B" in i.message for i in issues))

    def test_flag_conflicts_detection(self):
        cfg = {
            "fvt_closure": {"config": {"require_trigWeight": True}},
            "classifier_inputs": {"config": {"require_trigWeight": False}},
        }
        issues: list[Issue] = []
        audit_flags_and_duplicates(cfg, issues)
        warns = [i for i in issues if i.severity == "WARNING"]
        self.assertEqual(len(warns), 1)
        self.assertIn("Conflicting trigger weight flags", warns[0].message)

    def test_redundant_keys_informational(self):
        cfg = {
            "sec1": {"require_trigWeight": True},
            "sec2": {"require_trigWeight": True},
        }
        issues: list[Issue] = []
        audit_flags_and_duplicates(cfg, issues)
        infos = [i for i in issues if i.severity == "INFO"]
        self.assertEqual(len(infos), 1)
        self.assertIn("redundantly defined in 2 sections", infos[0].message)

    def test_friend_path_producer_consumer_mismatch(self):
        cfg = {
            "fvt": {
                "label": "FvT_v1",
            },
            "analysis_config": {
                "config": {
                    "friends": {
                        "FvT": "root://cmseos.fnal.gov/store/user/test/FvT_v2/friend.json",
                    }
                }
            },
        }
        issues: list[Issue] = []
        audit_producer_consumer_paths(cfg, issues)
        warns = [i for i in issues if i.severity == "WARNING"]
        self.assertEqual(len(warns), 1)
        self.assertIn("Producer Phase C label is 'FvT_v1'", warns[0].message)


class TestFullValidator(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.root = Path(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_validate_clean_config(self):
        cfg_file = self.root / "valid.yml"
        cfg_file.write_text(yaml.dump({
            "analysis": "HH4b",
            "year_eras": {"UL18": ["C", "D"]},
            "dataset": ["GluGluToHHTo4B_cHHH1", "data"],
            "classifier_inputs": {
                "datasets": ["GluGluToHHTo4B_cHHH1", "data"],
            },
        }))
        report = validate_config(cfg_file, repo_root=self.root)
        self.assertTrue(report.is_valid(strict=False))
        self.assertEqual(len(report.errors), 0)

    def test_format_report_and_summary_matrix(self):
        rep = ValidationReport(
            config_path=Path("dummy.yml"),
            bases=[Path("dummy.yml")],
            issues=[
                Issue(severity="ERROR", category="CONSISTENCY", key="test", message="bad divergence"),
                Issue(severity="WARNING", category="SCHEMA", key="test_w", message="missing field"),
                Issue(severity="INFO", category="DUPLICATE", key="test_i", message="redundant"),
            ],
        )
        self.assertFalse(rep.is_valid())
        out = format_report(rep, color=False)
        self.assertIn("Status: FAILED", out)
        self.assertIn("bad divergence", out)

        matrix = format_summary_matrix([rep], color=False)
        self.assertIn("Total Configs: 1 | Errors: 1 | Warnings: 1", matrix)


if __name__ == "__main__":
    unittest.main()
