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
    p.add_argument(
        "--stage",
        choices=["final", "baseline", "all"],
        default="final",
        help="which benchmark to summarize: final epoch of the last training stage (default), "
             "the pre-training Baseline (fit-quality replay of an existing model), or every stage",
    )
    return p.parse_args()


def _stats(arr):
    if not arr:
        return 0.0, 0.0
    mean = sum(arr) / len(arr)
    std = (sum((x - mean) ** 2 for x in arr) / len(arr)) ** 0.5 if len(arr) > 1 else 0.0
    return mean, std


def _variance(arr):
    if not arr or len(arr) < 2:
        return 0.0
    mean = sum(arr) / len(arr)
    return sum((x - mean) ** 2 for x in arr) / (len(arr) - 1)


def _select_benchmarks(models: list, stage: str = "final"):
    """Yield one ``{"training": ..., "validation": ...}`` benchmark dict per selected stage.

    final:    last epoch of each model's last Training/Finetuning stage (one entry per fold)
    baseline: the pre-training "Baseline" benchmark (one per fold); with --pretrained-weights
              and zero epochs this is the fit quality of an existing model (fit_quality_replay)
    all:      last epoch of every Training/Finetuning stage (pre-Nov 2026 behaviour: pools
              Training and Finetuning, so a 3-fold training gives 6 entries)
    """
    for m in models:
        history = m.get("history", [])
        if stage == "baseline":
            for st in history:
                if st.get("name") == "Baseline" and st.get("benchmarks"):
                    yield st["benchmarks"]
            continue
        trained = [
            st for st in history
            if st.get("name") in ["Training", "Finetuning"] and st.get("training")
        ]
        if stage == "final":
            trained = trained[-1:]
        for st in trained:
            yield st["training"][-1].get("benchmarks", {})


def extract_benchmark(result_path: str, stage: str = "final") -> dict:
    with open(result_path) as f:
        data = json.load(f)

    models = data.get("models", [])
    if not models:
        return {}

    summary = {
        "n_models": len(models),
        "avg_val_loss": 0.0,
        "val_loss_std": 0.0,
        "avg_tr_loss": 0.0,
        "tr_loss_std": 0.0,
        "loss_gap": 0.0,
        "avg_auc_4b_3b": 0.0,
        "val_auc_4b_3b_std": 0.0,
        "avg_tr_auc_4b_3b": 0.0,
        "tr_auc_4b_3b_std": 0.0,
        "avg_auc_ttbar": 0.0,
        "val_auc_ttbar_std": 0.0,
        "avg_tr_auc_ttbar": 0.0,
        "tr_auc_ttbar_std": 0.0,
        "fit_quality": {},
    }

    val_losses = []
    tr_losses = []
    val_loss_sb = []
    tr_loss_sb = []
    val_loss_sr = []
    tr_loss_sr = []

    val_aucs_4b = []
    tr_aucs_4b = []
    val_aucs_4b_sb = []
    tr_aucs_4b_sb = []
    val_aucs_tt = []
    tr_aucs_tt = []
    val_aucs_tt_sb = []
    tr_aucs_tt_sb = []
    val_aucs_tt_sr = []
    tr_aucs_tt_sr = []

    val_chi2_fvt_sb = []
    tr_chi2_fvt_sb = []
    val_chi2_t3_sb = []
    val_chi2_t4_sb = []
    val_chi2_t3_sr = []
    val_chi2_t4_sr = []

    fit_qualities = []

    for benchmarks in _select_benchmarks(models, stage):

        # Validation
        val_sc = benchmarks.get("validation", {}).get("scalars", {})
        val_rocs = benchmarks.get("validation", {}).get("roc", [])
        val_fq = benchmarks.get("validation", {}).get("fit_quality", [])

        if "loss" in val_sc:
            val_losses.append(val_sc["loss"])
        for r in val_rocs:
            rname = r.get("name")
            rauc = r.get("AUC")
            if rauc is None:
                continue
            if rname in ["4b vs 3b data SB", "4b vs 3b SB"]:
                val_aucs_4b_sb.append(rauc)
            elif rname in ["4b vs 3b data", "4b vs 3b"]:
                val_aucs_4b.append(rauc)
            elif rname == "ttbar vs data SB":
                val_aucs_tt_sb.append(rauc)
            elif rname == "ttbar vs data SR":
                val_aucs_tt_sr.append(rauc)
            elif rname == "ttbar vs data":
                val_aucs_tt.append(rauc)

        if val_fq:
            v_prox = val_fq[0].get("proxies", {})
            if v_prox.get("loss_sb") is not None: val_loss_sb.append(v_prox["loss_sb"])
            if v_prox.get("loss_sr") is not None: val_loss_sr.append(v_prox["loss_sr"])
            if v_prox.get("chi2_fvt_sb") is not None: val_chi2_fvt_sb.append(v_prox["chi2_fvt_sb"])
            if v_prox.get("chi2_t3_sb") is not None: val_chi2_t3_sb.append(v_prox["chi2_t3_sb"])
            if v_prox.get("chi2_t4_sb") is not None: val_chi2_t4_sb.append(v_prox["chi2_t4_sb"])
            if v_prox.get("chi2_t3_sr") is not None: val_chi2_t3_sr.append(v_prox["chi2_t3_sr"])
            if v_prox.get("chi2_t4_sr") is not None: val_chi2_t4_sr.append(v_prox["chi2_t4_sr"])
            fit_qualities.append(val_fq[0])

        # Training
        tr_sc = benchmarks.get("training", {}).get("scalars", {})
        tr_rocs = benchmarks.get("training", {}).get("roc", [])
        if "loss" in tr_sc:
            tr_losses.append(tr_sc["loss"])
        for r in tr_rocs:
            rname = r.get("name")
            rauc = r.get("AUC")
            if rauc is None:
                continue
            if rname in ["4b vs 3b data SB", "4b vs 3b SB"]:
                tr_aucs_4b_sb.append(rauc)
            elif rname in ["4b vs 3b data", "4b vs 3b"]:
                tr_aucs_4b.append(rauc)
            elif rname == "ttbar vs data SB":
                tr_aucs_tt_sb.append(rauc)
            elif rname == "ttbar vs data SR":
                tr_aucs_tt_sr.append(rauc)
            elif rname == "ttbar vs data":
                tr_aucs_tt.append(rauc)

        tr_fq = benchmarks.get("training", {}).get("fit_quality", [])
        if tr_fq:
            t_prox = tr_fq[0].get("proxies", {})
            if t_prox.get("loss_sb") is not None: tr_loss_sb.append(t_prox["loss_sb"])
            if t_prox.get("loss_sr") is not None: tr_loss_sr.append(t_prox["loss_sr"])
            if t_prox.get("chi2_fvt_sb") is not None: tr_chi2_fvt_sb.append(t_prox["chi2_fvt_sb"])

    summary["avg_val_loss"], summary["val_loss_std"] = _stats(val_losses)
    summary["avg_tr_loss"], summary["tr_loss_std"] = _stats(tr_losses)
    if val_losses and tr_losses:
        summary["loss_gap"] = summary["avg_val_loss"] - summary["avg_tr_loss"]

    summary["avg_auc_4b_3b"], summary["val_auc_4b_3b_std"] = _stats(val_aucs_4b)
    summary["avg_tr_auc_4b_3b"], summary["tr_auc_4b_3b_std"] = _stats(tr_aucs_4b)

    summary["avg_auc_ttbar"], summary["val_auc_ttbar_std"] = _stats(val_aucs_tt)
    summary["avg_tr_auc_ttbar"], summary["tr_auc_ttbar_std"] = _stats(tr_aucs_tt)

    if fit_qualities:
        # Average fit qualities across folds
        avg_fq = {
            "weights": {
                "neg_fraction": sum(fq.get("weights", {}).get("neg_fraction", 0.0) for fq in fit_qualities) / len(fit_qualities),
                "p99": sum(fq.get("weights", {}).get("p99", 0.0) for fq in fit_qualities) / len(fit_qualities),
                "max": max(fq.get("weights", {}).get("max", 0.0) for fq in fit_qualities),
            },
            "ttbar_mc_comparison": {},
            "data_calibration": {},
            "closure": {},
            "verdict": {
                "all_gates_passed": all(fq.get("verdict", {}).get("all_gates_passed", False) for fq in fit_qualities),
                "primary_score": (lambda v: sum(v) / len(v) if v else None)([fq.get("verdict", {}).get("primary_score") for fq in fit_qualities if fq.get("verdict", {}).get("primary_score") is not None]),
            },
        }

        # Closure metrics
        for cl_key in ["chi2_ndf_pt4", "chi2_ndf_pt1", "chi2_ndf_fvt", "norm_closure_sb", "composite_chi2_ndf"]:
            vals = [fq.get("closure", {}).get(cl_key) for fq in fit_qualities if fq.get("closure", {}).get(cl_key) is not None]
            if vals:
                avg_fq["closure"][cl_key] = sum(vals) / len(vals)

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

    # Extract Online Proxies (Stage F2 closure predictors)
    avg_val_loss_sb, _ = _stats(val_loss_sb)
    avg_tr_loss_sb, _ = _stats(tr_loss_sb)
    avg_val_loss_sr, _ = _stats(val_loss_sr)
    avg_tr_loss_sr, _ = _stats(tr_loss_sr)

    delta_loss_sb = (avg_val_loss_sb - avg_tr_loss_sb) if (val_loss_sb and tr_loss_sb) else 0.0
    delta_loss_sr = (avg_val_loss_sr - avg_tr_loss_sr) if (val_loss_sr and tr_loss_sr) else 0.0

    avg_auc_4b_sb, _ = _stats(val_aucs_4b_sb)
    avg_auc_tt_sb, _ = _stats(val_aucs_tt_sb)
    avg_auc_tt_sr, _ = _stats(val_aucs_tt_sr)

    avg_chi2_fvt_sb, _ = _stats(val_chi2_fvt_sb)
    avg_chi2_t3_sb, _ = _stats(val_chi2_t3_sb)
    avg_chi2_t4_sb, _ = _stats(val_chi2_t4_sb)
    avg_chi2_t3_sr, _ = _stats(val_chi2_t3_sr)
    avg_chi2_t4_sr, _ = _stats(val_chi2_t4_sr)

    var_chi2_fvt_sb_val = _variance(val_chi2_fvt_sb)
    var_chi2_fvt_sb_tr = _variance(tr_chi2_fvt_sb)
    var_loss_sb_val = _variance(val_loss_sb)
    var_loss_sb_tr = _variance(tr_loss_sb)
    var_loss_sr_val = _variance(val_loss_sr)
    var_loss_sr_tr = _variance(tr_loss_sr)
    var_auc_4b_sb_val = _variance(val_aucs_4b_sb)
    var_auc_tt_sb_val = _variance(val_aucs_tt_sb)
    var_auc_tt_sr_val = _variance(val_aucs_tt_sr)

    gate_bias = bool(delta_loss_sb < 1.0e-3 and delta_loss_sr < 0.9e-3) if val_loss_sb else None
    gate_fit_quality = bool(avg_val_loss_sr < 1.020 and avg_auc_tt_sr >= 0.835) if (val_loss_sr and avg_auc_tt_sr > 0) else None
    gate_variance = bool(var_auc_tt_sr_val < 1.0e-6) if len(val_aucs_tt_sr) >= 2 else None
    all_proxy_gates = (gate_bias and gate_fit_quality and gate_variance) if (gate_bias is not None and gate_fit_quality is not None and gate_variance is not None) else None

    summary["online_proxies"] = {
        "fit_quality": {
            "loss_sb": avg_val_loss_sb,
            "loss_sr": avg_val_loss_sr,
            "auc_4b_3b_sb": avg_auc_4b_sb,
            "auc_tt_sb": avg_auc_tt_sb,
            "auc_tt_sr": avg_auc_tt_sr,
            "chi2_fvt_sb": avg_chi2_fvt_sb,
            "chi2_t3_sb": avg_chi2_t3_sb,
            "chi2_t4_sb": avg_chi2_t4_sb,
            "chi2_t3_sr": avg_chi2_t3_sr,
            "chi2_t4_sr": avg_chi2_t4_sr,
        },
        "bias": {
            "delta_loss_sb": delta_loss_sb,
            "delta_loss_sr": delta_loss_sr,
        },
        "variance": {
            "var_chi2_fvt_sb_val": var_chi2_fvt_sb_val,
            "var_chi2_fvt_sb_tr": var_chi2_fvt_sb_tr,
            "var_loss_sb_val": var_loss_sb_val,
            "var_loss_sb_tr": var_loss_sb_tr,
            "var_loss_sr_val": var_loss_sr_val,
            "var_loss_sr_tr": var_loss_sr_tr,
            "var_auc_4b_3b_sb_val": var_auc_4b_sb_val,
            "var_auc_tt_sb_val": var_auc_tt_sb_val,
            "var_auc_tt_sr_val": var_auc_tt_sr_val,
        },
        "verdict": {
            "gate_bias": gate_bias,
            "gate_fit_quality": gate_fit_quality,
            "gate_variance": gate_variance,
            "all_gates_passed": all_proxy_gates,
        },
    }

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
    if "chi2_ndf_fvt" in closure or "chi2_ndf_fvt" in ref_closure:
        row("  chi2/ndf: FvT (Sideband)", closure.get("chi2_ndf_fvt", "-"), ref_closure.get("chi2_ndf_fvt", "-"))
    if "norm_closure_sb" in closure or "norm_closure_sb" in ref_closure:
        row("  Sideband Norm (3b*w/4b)", closure.get("norm_closure_sb", "-"), ref_closure.get("norm_closure_sb", "-"))
    row("  Composite chi2/ndf (Score)", closure.get("composite_chi2_ndf", "-"), ref_closure.get("composite_chi2_ndf", "-"))

    print("\n[ TIER 3: DISCRIMINATION & LOSS (TRAIN vs VAL) ]")
    row("  Validation Loss", summary.get("avg_val_loss", 0.0), ref_summary.get("avg_val_loss") if ref_summary else None, ".4f")
    row("  Training Loss", summary.get("avg_tr_loss", 0.0), ref_summary.get("avg_tr_loss") if ref_summary else None, ".4f")
    row("  Loss Gap (Val - Tr)", summary.get("loss_gap", 0.0), ref_summary.get("loss_gap") if ref_summary else None, "+.4f")
    if summary.get("val_loss_std", 0.0) > 0 or (ref_summary and ref_summary.get("val_loss_std", 0.0) > 0):
        row("  Val Loss Fold Spread (σ)", summary.get("val_loss_std", 0.0), ref_summary.get("val_loss_std") if ref_summary else None, ".4f")
        row("  Tr Loss Fold Spread (σ)", summary.get("tr_loss_std", 0.0), ref_summary.get("tr_loss_std") if ref_summary else None, ".4f")
    row("  Validation AUC (4b vs 3b)", summary.get("avg_auc_4b_3b", 0.0), ref_summary.get("avg_auc_4b_3b") if ref_summary else None, ".4f")
    row("  Training AUC (4b vs 3b)", summary.get("avg_tr_auc_4b_3b", 0.0), ref_summary.get("avg_tr_auc_4b_3b") if ref_summary else None, ".4f")
    row("  Validation AUC (ttbar vs data)", summary.get("avg_auc_ttbar", 0.0), ref_summary.get("avg_auc_ttbar") if ref_summary else None, ".4f")
    row("  Training AUC (ttbar vs data)", summary.get("avg_tr_auc_ttbar", 0.0), ref_summary.get("avg_tr_auc_ttbar") if ref_summary else None, ".4f")

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

    prox = summary.get("online_proxies", {})
    ref_prox = ref_summary.get("online_proxies", {}) if ref_summary else {}
    if prox:
        print("\n[ ONLINE PROXIES: FIT QUALITY ]")
        fq_p = prox.get("fit_quality", {})
        rfq_p = ref_prox.get("fit_quality", {})
        row("  Loss (SB)", fq_p.get("loss_sb", 0.0), rfq_p.get("loss_sb") if ref_summary else None, ".4f")
        row("  Loss (SR)", fq_p.get("loss_sr", 0.0), rfq_p.get("loss_sr") if ref_summary else None, ".4f")
        row("  AUC 4b vs 3b (SB)", fq_p.get("auc_4b_3b_sb", 0.0), rfq_p.get("auc_4b_3b_sb") if ref_summary else None, ".4f")
        row("  AUC ttbar vs data (SB)", fq_p.get("auc_tt_sb", 0.0), rfq_p.get("auc_tt_sb") if ref_summary else None, ".4f")
        row("  AUC ttbar vs data (SR)", fq_p.get("auc_tt_sr", 0.0), rfq_p.get("auc_tt_sr") if ref_summary else None, ".4f")
        row("  chi2 binned FvT (SB)", fq_p.get("chi2_fvt_sb", 0.0), rfq_p.get("chi2_fvt_sb") if ref_summary else None, ".3f")
        row("  chi2 t3 -> d3 (SB)", fq_p.get("chi2_t3_sb", 0.0), rfq_p.get("chi2_t3_sb") if ref_summary else None, ".3f")
        row("  chi2 t4 -> d3 (SB)", fq_p.get("chi2_t4_sb", 0.0), rfq_p.get("chi2_t4_sb") if ref_summary else None, ".3f")
        row("  chi2 t3 -> d3 (SR)", fq_p.get("chi2_t3_sr", 0.0), rfq_p.get("chi2_t3_sr") if ref_summary else None, ".3f")
        row("  chi2 t4 -> d3 (SR)", fq_p.get("chi2_t4_sr", 0.0), rfq_p.get("chi2_t4_sr") if ref_summary else None, ".3f")

        print("\n[ ONLINE PROXIES: BIAS (GENERALIZATION GAPS) ]")
        bias_p = prox.get("bias", {})
        rbias_p = ref_prox.get("bias", {})
        row("  Delta Loss SB (Val - Tr)", bias_p.get("delta_loss_sb", 0.0), rbias_p.get("delta_loss_sb") if ref_summary else None, "+.5f")
        row("  Delta Loss SR (Val - Tr)", bias_p.get("delta_loss_sr", 0.0), rbias_p.get("delta_loss_sr") if ref_summary else None, "+.5f")

        print("\n[ ONLINE PROXIES: ENSEMBLE VARIANCE (ACROSS FOLDS) ]")
        var_p = prox.get("variance", {})
        rvar_p = ref_prox.get("variance", {})
        row("  Var chi2 FvT SB (Val)", var_p.get("var_chi2_fvt_sb_val", 0.0), rvar_p.get("var_chi2_fvt_sb_val") if ref_summary else None, ".2f")
        row("  Var Loss SB (Val)", var_p.get("var_loss_sb_val", 0.0), rvar_p.get("var_loss_sb_val") if ref_summary else None, ".2e")
        row("  Var Loss SR (Val)", var_p.get("var_loss_sr_val", 0.0), rvar_p.get("var_loss_sr_val") if ref_summary else None, ".2e")
        row("  Var AUC ttbar SR (Val)", var_p.get("var_auc_tt_sr_val", 0.0), rvar_p.get("var_auc_tt_sr_val") if ref_summary else None, ".2e")

        print("\n[ ONLINE PROXIES: 3-TIER GATE VERDICT ]")
        verd = prox.get("verdict", {})
        rverd = ref_prox.get("verdict", {})
        row("  Gate 1: Bias (Delta Loss)", "PASS" if verd.get("gate_bias") else "FAIL", "PASS" if rverd.get("gate_bias") else ("FAIL" if ref_summary else "-"))
        row("  Gate 2: Fit Quality (SR Loss & Top Rej)", "PASS" if verd.get("gate_fit_quality") else "FAIL", "PASS" if rverd.get("gate_fit_quality") else ("FAIL" if ref_summary else "-"))
        row("  Gate 3: Variance (Fold Stability)", "PASS" if verd.get("gate_variance") else "FAIL", "PASS" if rverd.get("gate_variance") else ("FAIL" if ref_summary else "-"))
        row("  ALL GATES PASSED?", "YES" if verd.get("all_gates_passed") else "NO", "YES" if rverd.get("all_gates_passed") else ("NO" if ref_summary else "-"))
    print("=" * 70 + "\n")


def main():
    args = parse_args()
    summary = extract_benchmark(args.result, args.stage)
    ref_summary = extract_benchmark(args.reference, args.stage) if args.reference else None

    if args.json:
        out = {"current": summary}
        if ref_summary:
            out["reference"] = ref_summary
        print(json.dumps(out, indent=2))
    else:
        print_table(summary, ref_summary)


if __name__ == "__main__":
    main()
