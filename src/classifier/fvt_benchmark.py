#!/usr/bin/env python
"""Evaluate and summarize FvT training fit quality benchmarks.

Usage:
  python -m src.classifier.fvt_benchmark --result output/FvT/result.json
  python -m src.classifier.fvt_benchmark --result output/FvT/result.json --reference roasts/nominal/result.json
"""
import argparse
import json
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result", required=True, help="Path to result.json from FvT training")
    p.add_argument("--reference", default=None, help="Optional path to nominal reference result.json")
    p.add_argument("--json", action="store_true", help="Print summary as JSON")
    return p.parse_args()


def extract_benchmark(result_path: str) -> dict:
    with open(result_path) as f:
        data = json.load(f)

    models = data.get("models", [])
    if not models:
        return {}

    summary = {
        "models": [],
        "avg_val_loss": 0.0,
        "avg_auc_4b_3b": 0.0,
        "avg_auc_ttbar": 0.0,
        "fit_quality": {},
    }

    losses = []
    aucs_4b = []
    aucs_tt = []
    fit_qualities = []

    for m in models:
        m_info = {"name": m.get("name")}
        for stage in m.get("history", []):
            if stage.get("name") in ["Training", "Finetuning"] and "training" in stage:
                last_epoch = stage["training"][-1]
                benchmarks = last_epoch.get("benchmarks", {})
                val_sc = benchmarks.get("validation", {}).get("scalars", {})
                val_rocs = benchmarks.get("validation", {}).get("roc", [])
                val_fq = benchmarks.get("validation", {}).get("fit_quality", [])

                if "loss" in val_sc:
                    losses.append(val_sc["loss"])
                for r in val_rocs:
                    if r.get("name") in ["4b vs 3b data", "4b vs 3b"]:
                        if r.get("AUC") is not None:
                            aucs_4b.append(r["AUC"])
                    elif r.get("name") == "ttbar vs data":
                        if r.get("AUC") is not None:
                            aucs_tt.append(r["AUC"])
                if val_fq:
                    fit_qualities.append(val_fq[0])

    if losses:
        summary["avg_val_loss"] = sum(losses) / len(losses)
    if aucs_4b:
        summary["avg_auc_4b_3b"] = sum(aucs_4b) / len(aucs_4b)
    if aucs_tt:
        summary["avg_auc_ttbar"] = sum(aucs_tt) / len(aucs_tt)

    if fit_qualities:
        # Average fit qualities across folds
        avg_fq = {
            "weights": {
                "neg_fraction": sum(fq["weights"]["neg_fraction"] for fq in fit_qualities) / len(fit_qualities),
                "p99": sum(fq["weights"]["p99"] for fq in fit_qualities) / len(fit_qualities),
                "max": max(fq["weights"]["max"] for fq in fit_qualities),
            },
            "ttbar_mc_comparison": {},
            "data_calibration": {},
            "closure": {
                "chi2_ndf_pt4": sum(fq["closure"]["chi2_ndf_pt4"] for fq in fit_qualities) / len(fit_qualities),
                "chi2_ndf_pt1": sum(fq["closure"]["chi2_ndf_pt1"] for fq in fit_qualities) / len(fit_qualities),
                "composite_chi2_ndf": sum(fq["closure"]["composite_chi2_ndf"] for fq in fit_qualities) / len(fit_qualities),
            },
            "verdict": {
                "all_gates_passed": all(fq["verdict"]["all_gates_passed"] for fq in fit_qualities),
                "primary_score": sum(fq["verdict"]["primary_score"] for fq in fit_qualities) / len(fit_qualities),
            },
        }
        if "ttbar_mc_comparison" in fit_qualities[0]:
            for k in fit_qualities[0]["ttbar_mc_comparison"]:
                avg_fq["ttbar_mc_comparison"][k] = sum(fq["ttbar_mc_comparison"][k] for fq in fit_qualities) / len(fit_qualities)
        if "data_calibration" in fit_qualities[0]:
            for k in fit_qualities[0]["data_calibration"]:
                avg_fq["data_calibration"][k] = sum(fq["data_calibration"][k] for fq in fit_qualities) / len(fit_qualities)
        elif "calibration" in fit_qualities[0]:
            avg_fq["calibration"] = {
                "ratio_pd4_pd3": sum(fq["calibration"].get("ratio_pd4_pd3", 1.0) for fq in fit_qualities) / len(fit_qualities),
                "ratio_pt4_pt3": sum(fq["calibration"].get("ratio_pt4_pt3", 1.0) for fq in fit_qualities) / len(fit_qualities),
            }
        summary["fit_quality"] = avg_fq

    return summary


def print_table(summary: dict, ref_summary: dict = None):
    print("\n" + "=" * 70)
    print("           FvT TRAINING FIT QUALITY BENCHMARK")
    print("=" * 70)

    fq = summary.get("fit_quality", {})
    weights = fq.get("weights", {})
    ttbar_mc = fq.get("ttbar_mc_comparison", {})
    data_calib = fq.get("data_calibration", {})
    calib = fq.get("calibration", {})
    closure = fq.get("closure", {})
    verdict = fq.get("verdict", {})

    ref_fq = ref_summary.get("fit_quality", {}) if ref_summary else {}
    ref_weights = ref_fq.get("weights", {})
    ref_ttbar_mc = ref_fq.get("ttbar_mc_comparison", {})
    ref_data_calib = ref_fq.get("data_calibration", {})
    ref_calib = ref_fq.get("calibration", {})
    ref_closure = ref_fq.get("closure", {})

    header = f"{'Metric':<35} | {'Current':<15}"
    if ref_summary:
        header += f" | {'Reference':<15}"
    print(header)
    print("-" * len(header))

    def row(label, val, ref_val=None, fmt=""):
        s_val = f"{val:{fmt}}" if isinstance(val, (int, float)) else str(val)
        line = f"{label:<35} | {s_val:<15}"
        if ref_summary:
            s_ref = f"{ref_val:{fmt}}" if isinstance(ref_val, (int, float)) else (str(ref_val) if ref_val is not None else "-")
            line += f" | {s_ref:<15}"
        print(line)

    print("[ TIER 1: HARD GATES ]")
    neg_f = weights.get("neg_fraction", 0.0) * 100
    ref_neg_f = ref_weights.get("neg_fraction", 0.0) * 100 if ref_weights else None
    row("  Negative weight fraction", f"{neg_f:.3f}%", f"{ref_neg_f:.3f}%" if ref_neg_f is not None else "-", "")
    row("  99th percentile weight (p99)", weights.get("p99", "-"), ref_weights.get("p99", "-"))
    row("  Maximum weight (max)", weights.get("max", "-"), ref_weights.get("max", "-"))
    row("  All Gates Passed?", verdict.get("all_gates_passed", "-"))

    print("\n[ TIER 2: PRIMARY OPTIMIZATION (CLOSURE) ]")
    row("  chi2/ndf: pt4 (4th jet pt)", closure.get("chi2_ndf_pt4", "-"), ref_closure.get("chi2_ndf_pt4", "-"))
    row("  chi2/ndf: pt1 (Lead jet pt)", closure.get("chi2_ndf_pt1", "-"), ref_closure.get("chi2_ndf_pt1", "-"))
    row("  Composite chi2/ndf (Score)", closure.get("composite_chi2_ndf", "-"), ref_closure.get("composite_chi2_ndf", "-"))

    print("\n[ TIER 3: DISCRIMINATION & LOSS ]")
    row("  Validation Loss", summary.get("avg_val_loss", 0.0), ref_summary.get("avg_val_loss") if ref_summary else None, ".4f")
    row("  AUC (4b vs 3b data)", summary.get("avg_auc_4b_3b", 0.0), ref_summary.get("avg_auc_4b_3b") if ref_summary else None, ".4f")
    row("  AUC (ttbar vs data)", summary.get("avg_auc_ttbar", 0.0), ref_summary.get("avg_auc_ttbar") if ref_summary else None, ".4f")

    if ttbar_mc:
        print("\n[ TTBAR MC PRESERVATION & CALIBRATION ]")
        row("  True MC t4 yield", ttbar_mc.get("true_mc_t4_weight", "-"), ref_ttbar_mc.get("true_mc_t4_weight", "-"))
        row("  True MC t3 yield", ttbar_mc.get("true_mc_t3_weight", "-"), ref_ttbar_mc.get("true_mc_t3_weight", "-"))
        row("  True MC t4 / t3 ratio", ttbar_mc.get("true_ratio_t4_t3", "-"), ref_ttbar_mc.get("true_ratio_t4_t3", "-"))
        row("  Pred t4 / t3 ratio", ttbar_mc.get("pred_ratio_t4_t3", "-"), ref_ttbar_mc.get("pred_ratio_t4_t3", "-"))
        row("  Calib: Pred t4 prob / MC t4", ttbar_mc.get("calib_t4_prob_to_mc", "-"), ref_ttbar_mc.get("calib_t4_prob_to_mc", "-"))
        row("  Calib: Pred t3 prob / MC t3", ttbar_mc.get("calib_t3_prob_to_mc", "-"), ref_ttbar_mc.get("calib_t3_prob_to_mc", "-"))
        row("  Transfer: d3->t4 / MC t4", ttbar_mc.get("transfer_d3_to_t4_to_mc", "-"), ref_ttbar_mc.get("transfer_d3_to_t4_to_mc", "-"))
        row("  Transfer: d3->t3 / MC t3", ttbar_mc.get("transfer_d3_to_t3_to_mc", "-"), ref_ttbar_mc.get("transfer_d3_to_t3_to_mc", "-"))

    if calib:
        print("\n[ CALIBRATION PROBABILITIES ]")
        row("  Ratio pd4 / pd3", calib.get("ratio_pd4_pd3", "-"), ref_calib.get("ratio_pd4_pd3", "-"))
        row("  Ratio pt4 / pt3", calib.get("ratio_pt4_pt3", "-"), ref_calib.get("ratio_pt4_pt3", "-"))
    print("=" * 70 + "\n")


def main():
    args = parse_args()
    summary = extract_benchmark(args.result)
    ref_summary = extract_benchmark(args.reference) if args.reference else None

    if args.json:
        out = {"current": summary}
        if ref_summary:
            out["reference"] = ref_summary
        print(json.dumps(out, indent=2))
    else:
        print_table(summary, ref_summary)


if __name__ == "__main__":
    main()
