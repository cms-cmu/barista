import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.classifier import plot_weights


class TestPlotWeights(unittest.TestCase):
    def test_auto_selects_fvt_label_map_and_maps_all_classes(self):
        patterns = plot_weights._load_label_patterns(None, ["t4", "t3", "d4", "d3"])
        self.assertEqual(
            plot_weights.sample_to_labels("data_UL18_A", patterns),
            ["d3", "d4"],
        )
        self.assertEqual(
            plot_weights.sample_to_labels("TTToHadronic_stitched_UL18", patterns),
            ["t3", "t4"],
        )

    def test_auto_selects_hh4b_label_map_including_tthbb(self):
        patterns = plot_weights._load_label_patterns(None, ["ttHbb", "ttbar", "multijet"])
        self.assertEqual(
            plot_weights.sample_to_labels("ttHbb_UL18", patterns),
            ["ttHbb"],
        )
        self.assertEqual(
            plot_weights.sample_to_labels("data_UL18_A", patterns),
            ["multijet"],
        )
        self.assertEqual(
            plot_weights.sample_to_labels("TTToSemiLeptonic_stitched_UL18", patterns),
            ["ttbar"],
        )

    def test_load_train_yaml_preserves_template_fvt_over_friend_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            wfs_dir = Path(tmp)
            (wfs_dir / "train.yml").write_text(
                """
dataset:
- module: HCR.SvB.Background
  option:
  - --no-SR
  - --JCM-weight year:UL18 root://eos/jcm_UL18.yml@@JCM_weights
  - --friends "" {FvT}_input@@HCR_input
  - --friends label:data {FvT}/result.json@@analysis.0.merged
"""
            )
            friends_args, jcm_args, no_sr = plot_weights.load_train_yaml(
                str(wfs_dir),
                template_str="model: root://eos/m, FvT: root://eos/friend/FvT_run",
                friend_path="root://eos/friend/SvB_run",
            )
            self.assertTrue(no_sr)
            self.assertEqual(len(jcm_args), 1)
            self.assertIn("root://eos/friend/FvT_run/result.json@@analysis.0.merged", " ".join(friends_args))
            self.assertNotIn("SvB_run", " ".join(friends_args))

    def test_d4_blinding_never_includes_sr_even_when_no_sr_false(self):
        # Events: [0]=3b+SB+HLT, [1]=4b+SB+HLT, [2]=4b+SR+HLT, [3]=4b+SB+noHLT, [4]=3b+SR+HLT
        ntags3 = np.array([True, False, False, False, True])
        ntags4 = np.array([False, True, True, True, False])
        sb = np.array([True, True, False, True, False])
        sr = np.array([False, False, True, False, True])
        pass_hlt = np.array([True, True, True, False, True])

        d4_mask = plot_weights._build_label_mask("d4", ntags3, ntags4, sb, sr, pass_hlt=pass_hlt, no_sr=False)
        self.assertEqual(d4_mask.tolist(), [False, True, False, False, False])

        d3_all = plot_weights._build_label_mask("d3", ntags3, ntags4, sb, sr, pass_hlt=pass_hlt, no_sr=False)
        self.assertEqual(d3_all.tolist(), [True, False, False, False, True])

        d3_sb_only = plot_weights._build_label_mask("d3", ntags3, ntags4, sb, sr, pass_hlt=pass_hlt, no_sr=True)
        self.assertEqual(d3_sb_only.tolist(), [True, False, False, False, False])

        t4_mask = plot_weights._build_label_mask("t4", ntags3, ntags4, sb, sr, pass_hlt=pass_hlt, no_sr=False)
        self.assertEqual(t4_mask.tolist(), [False, True, True, True, False])

    def test_match_jcm_for_path_prefers_longest_year_key(self):
        jcm_by_year = {"UL16_preVFP": "pre", "UL16_postVFP": "post", "UL18": "u18", "2022": "22", "2022_EE": "22ee"}
        self.assertEqual(
            plot_weights._match_jcm_for_path("root://eos/data_UL16_preVFP_B/picoAOD.root", jcm_by_year),
            "pre",
        )
        self.assertEqual(
            plot_weights._match_jcm_for_path("root://eos/data_2022_EE_E/picoAOD.root", jcm_by_year),
            "22ee",
        )


if __name__ == "__main__":
    unittest.main()
