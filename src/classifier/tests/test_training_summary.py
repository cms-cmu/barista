import importlib.util
import tempfile
import unittest
from pathlib import Path

from src.classifier import training_summary


def _make_model_entry(root: Path, name: str, rel_dir: str, val_loss: float, gap: float, fq: dict) -> dict:
    mdir = root if rel_dir == "." else (root / rel_dir)
    (mdir / "plots" / "analyze" / "fit_quality").mkdir(parents=True, exist_ok=True)
    return {
        "index": 0,
        "name": name,
        "dir": mdir,
        "rel_dir": rel_dir,
        "config": {
            "prescale": None,
            "no_sr": False,
            "kfolds": 3,
            "epochs": 20,
            "ft_epochs": 1,
            "bs_init": 1024,
            "ft_bs": 16384,
            "jcm_weights": [],
        },
        "yields": {
            "d4_count": 1000,
            "d4_weight": 1000.0,
            "d3_count": 2000,
            "d3_weight": 1000.0,
            "t4_count": 100,
            "t4_weight": 50.0,
            "t3_count": 500,
            "t3_weight": 150.0,
        },
        "metrics": {
            "kfolds": 3,
            "best_epoch": 18,
            "val_loss": val_loss,
            "tr_loss": val_loss - gap,
            "gap": gap,
            "fold_sigma": 0.002,
            "z": gap / 0.002,
            "status_text": "Optimal",
            "status_badge": "optimal",
            "auc_4b_vs_3b": 0.6900,
            "auc_4b_vs_3b_err": 0.0010,
            "auc_4b_vs_3b_data": 0.6920,
            "auc_4b_vs_3b_data_err": 0.0015,
            "auc_ttbar_vs_data": 0.8610,
            "auc_ttbar_vs_data_err": 0.0008,
        },
        "fit_quality": fq,
    }


class TestTrainingSummary(unittest.TestCase):
    def test_generate_html_columns_ensemble_summary_gap_sign_and_section_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m0 = _make_model_entry(
                root,
                "mix_0",
                "models/mix_0",
                val_loss=0.6770,
                gap=-0.00050,
                fq={
                    "chi2_ndf_pt4": 100.0,
                    "chi2_ndf_pt1": 1.40,
                    "chi2_ndf_fvt": 1.50,
                    "chi2_ndf_svb_sr": 1.30,
                    "chi2_ndf_svb_sr_gt0p8": 1.20,
                    "composite_chi2_ndf": 34.30,
                    "neg_fraction": 0.0030,
                },
            )
            m1 = _make_model_entry(
                root,
                "mix_1",
                "models/mix_1",
                val_loss=0.6780,
                gap=-0.00080,
                fq={
                    "chi2_ndf_pt4": 110.0,
                    "chi2_ndf_pt1": 1.20,
                    "chi2_ndf_fvt": 1.30,
                    "chi2_ndf_svb_sr": 1.50,
                    "chi2_ndf_svb_sr_gt0p8": 1.40,
                    "composite_chi2_ndf": 37.50,
                    "neg_fraction": 0.0040,
                },
            )
            (m0["dir"] / "plots" / "analyze" / "fit_quality" / "svb_sr.png").write_bytes(b"png")

            out_html = root / "training_summary.html"
            training_summary.generate_html([m0, m1], out_html, "Test Ensemble Summary")
            text = out_html.read_text(encoding="utf-8")

            for col in (
                "<th>χ²/ndf (pt4)</th>",
                "<th>χ²/ndf (pt1)</th>",
                "<th>χ²/ndf (FvT)</th>",
                "<th>χ²/ndf (SvB SR)</th>",
                "<th>χ²/ndf (SvB &gt; 0.8)</th>",
                "<th>Score (χ²)</th>",
                "<th>f(w &lt; 0)</th>",
            ):
                self.assertIn(col, text)

            self.assertIn("-0.00065", text)
            self.assertNotIn("+-0.00065", text)
            self.assertIn("Cross-Sample Diagnostic Plots", text)
            self.assertNotIn('{"Cross-Sample', text)

            summary_row = text.split('<tr class="summary-row">', 1)[1].split("</tr>", 1)[0]
            self.assertIn("105.00 ± 7.07", summary_row)
            self.assertIn("1.30 ± 0.14", summary_row)
            self.assertIn("1.40 ± 0.14", summary_row)
            self.assertIn("0.35% ± 0.07%", summary_row)

            self.assertIn("tab-closure-dist", text)
            self.assertIn("models/mix_0/plots/analyze/fit_quality/svb_sr.png", text)

            out_single = root / "single_summary.html"
            training_summary.generate_html([m0], out_single, "Single Model Summary")
            single_text = out_single.read_text(encoding="utf-8")
            self.assertIn("Diagnostic & Metric Plots", single_text)
            self.assertNotIn('{"Cross-Sample', single_text)

    @unittest.skipIf(importlib.util.find_spec("matplotlib") is None, "matplotlib not installed")
    def test_generate_fit_quality_plots_writes_pt4_and_svb_sr(self):
        fq = {
            "source": "d3",
            "histograms": {
                "pt4_sb": {
                    "bin_edges": [20.0, 40.0, 60.0],
                    "target": [10.0, 200.0],
                    "target_sumw2": [10.0, 200.0],
                    "model": [15.0, 195.0],
                    "model_sumw2": [3.0, 25.0],
                },
                "svb_sr": {
                    "bin_edges": [0.0, 0.5, 1.0],
                    "target": [100.0, 50.0],
                    "target_sumw2": [100.0, 50.0],
                    "model": [105.0, 48.0],
                    "model_sumw2": [12.0, 8.0],
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "fit_quality"
            generated = training_summary.generate_fit_quality_plots([fq], out_dir, source="d3")
            self.assertIn("pt4_sb", generated)
            self.assertIn("svb_sr", generated)
            self.assertTrue((out_dir / "pt4_sb.png").is_file())
            self.assertTrue((out_dir / "svb_sr.png").is_file())


if __name__ == "__main__":
    unittest.main()
