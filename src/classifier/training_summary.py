#!/usr/bin/env python3
"""Cross-sample training summary HTML dashboard for Stage C FvT models.

Aggregates the 16 independent subsample trainings (mix_0..mix_15) in Step C:
- Extracts training parameters and options from wfs/train.yml
- Extracts loaded event counts and effective weights from yields.json
- Parses validation/training losses, fold spread, divergence Z, and status from analyze.log
- Parses test AUC metrics (4b vs 3b, 4b vs 3b data, ttbar vs data) from plots/analyze/index.html
- Computes ensemble-wide statistics (mean, sample std, min, max, ensemble pull)
- Renders an interactive HTML dashboard with scorecards, summary tables, and a lightbox plot gallery
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create FvT training summary HTML dashboard.")
    parser.add_argument("--models-dir", "-m", required=True, help="Path to models directory (e.g. output/.../bkg_syst_C_FvT/models)")
    parser.add_argument("--output", "-o", required=True, help="Path to output HTML file")
    parser.add_argument("--title", "-t", default=None, help="Dashboard title")
    return parser.parse_args()


def parse_train_yml(yml_path: Path) -> dict:
    """Parse key parameters from wfs/train.yml without requiring PyYAML."""
    data = {
        "prescale": None,
        "no_sr": False,
        "kfolds": 3,
        "epochs": 20,
        "bs_init": 1024,
        "ft_epochs": 1,
        "ft_bs": 16384,
        "jcm_weights": [],
    }
    if not yml_path.exists():
        return data

    try:
        content = yml_path.read_text(encoding="utf-8", errors="replace")
        m_ps = re.search(r"--ttbar-3b-prescale\s+(\d+)", content)
        if m_ps:
            data["prescale"] = int(m_ps.group(1))

        if "--no-SR" in content:
            data["no_sr"] = True

        m_kf = re.search(r"--kfolds\s+(\d+)", content)
        if m_kf:
            data["kfolds"] = int(m_kf.group(1))

        # epoch and bs_init
        epochs = re.findall(r"epoch:\s*(\d+)", content)
        bs = re.findall(r"bs_init:\s*(\d+)", content)
        if epochs:
            data["epochs"] = int(epochs[0])
            if len(epochs) > 1:
                data["ft_epochs"] = int(epochs[1])
        if bs:
            data["bs_init"] = int(bs[0])
            if len(bs) > 1:
                data["ft_bs"] = int(bs[1])

        # JCM weights
        jcms = re.findall(r"--JCM-weight\s+(\S+)\s+(\S+)", content)
        for year, fpath in jcms:
            fname = Path(fpath.split("@@")[0]).name
            data["jcm_weights"].append((year, fname))
    except Exception as e:
        print(f"Warning: Failed to parse {yml_path}: {e}", file=sys.stderr)

    return data


def parse_yields_json(json_path: Path) -> dict:
    """Parse loaded class counts and weights from yields.json."""
    data = {
        "d4_count": None, "d4_weight": None,
        "d3_count": None, "d3_weight": None,
        "t4_count": None, "t4_weight": None,
        "t3_count": None, "t3_weight": None,
    }
    if not json_path.exists():
        return data

    try:
        y = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
        # Regions -> SB
        sb = y.get("regions", {}).get("SB", {})
        if not sb and "years" in y:
            # sum over years if SB not at top level
            pass
        if sb:
            for cls in ("d4", "d3", "t4", "t3"):
                if cls in sb:
                    data[f"{cls}_count"] = sb[cls].get("count")
                    data[f"{cls}_weight"] = sb[cls].get("weight")
    except Exception as e:
        print(f"Warning: Failed to parse {json_path}: {e}", file=sys.stderr)

    return data


def fetch_remote_file(src_url: str, dst_path: Path) -> bool:
    """Fetch a remote file (e.g. root://...) using fsspec or xrdcp."""
    if not src_url.startswith("root://"):
        if Path(src_url).exists():
            if str(dst_path.resolve()) != str(Path(src_url).resolve()):
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_url, dst_path)
            return True
        return False

    # 1. Try fsspec (Python-native XRootD)
    try:
        import fsspec
        with fsspec.open(src_url, "rb") as fin:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dst_path, "wb") as fout:
                fout.write(fin.read())
        if dst_path.exists() and dst_path.stat().st_size > 0:
            return True
    except Exception:
        pass

    # 2. Try xrdcp CLI if available
    xrdcp_bin = shutil.which("xrdcp")
    if not xrdcp_bin and os.path.exists("/cvmfs/cms.cern.ch/common/xrdcp"):
        xrdcp_bin = "/cvmfs/cms.cern.ch/common/xrdcp"
    if xrdcp_bin:
        try:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [xrdcp_bin, "-s", "-f", src_url, str(dst_path)]
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if dst_path.exists() and dst_path.stat().st_size > 0:
                return True
        except Exception:
            pass

    return False


def locate_or_fetch_result_json(p: Path) -> Path | None:
    """Locate result.json locally or fetch from remote EOS storage if referenced in train.yml."""
    if not p.is_dir():
        return None
    res_file = p / "result.json"
    if res_file.exists():
        return res_file
    if (p / "models" / "result.json").exists():
        return p / "models" / "result.json"

    train_yml = p / "wfs" / "train.yml"
    if train_yml.exists():
        try:
            content = train_yml.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"output:\s*['\"]?(root://[^\s'\"]+|[^\s'\"]+)", content)
            if m:
                out_target = m.group(1).rstrip("/")
                cand = f"{out_target}/result.json"
                if "{" in out_target:
                    for cfg_path in (
                        list(p.glob("../../roasts/*/config.yml"))
                        + list(p.glob("../../../roasts/*/config.yml"))
                        + list(Path.cwd().glob("roasts/*/config.yml"))
                    ):
                        try:
                            cfg_text = cfg_path.read_text(encoding="utf-8", errors="replace")
                            m_fvt = re.search(r"model:\s*['\"]?([^\s'\"]+)", cfg_text)
                            if m_fvt:
                                m_tmpl = m_fvt.group(1)
                                eos_p = "root://cmseos.fnal.gov//store/user/algomez/HH4b_prod"
                                rid = cfg_path.parent.name
                                m_url = m_tmpl.replace("{eos_prod}", eos_p)
                                m_url = m_url.replace("{eos_base}", f"{eos_p}/{rid}")
                                m_url = m_url.replace("{roast_id}", rid)
                                m_url = m_url.replace("{label}", p.name)
                                if m_url.startswith("root://"):
                                    cand = f"{m_url.rstrip('/')}/result.json"
                                    break
                        except Exception:
                            pass
                if cand.startswith("root://"):
                    if fetch_remote_file(cand, p / "result.json"):
                        return p / "result.json"
                elif Path(cand).exists():
                    return Path(cand)
        except Exception:
            pass

    return None


def parse_fit_quality_json(result_path: Path | None) -> dict:
    """Parse fit quality benchmark (weights health, calibration, closure) from result.json."""
    data = {
        "neg_fraction": None,
        "p99": None,
        "max": None,
        "mean": None,
        "std": None,
        "ratio_pd4_pd3": None,
        "ratio_pt4_pt3": None,
        "true_ratio_t4_t3": None,
        "pred_ratio_t4_t3": None,
        "transfer_d3_to_t4_to_mc": None,
        "transfer_d3_to_t3_to_mc": None,
        "chi2_ndf_pt4": None,
        "chi2_ndf_pt1": None,
        "chi2_ndf_fvt": None,
        "chi2_ndf_svb_sr": None,
        "norm_closure_sb": None,
        "composite_chi2_ndf": None,
        "all_gates_passed": None,
        "loss_sb": None,
        "loss_sr": None,
        "chi2_fvt_sb": None,
        "chi2_t3_sb": None,
        "chi2_t4_sb": None,
        "chi2_t3_sr": None,
        "chi2_t4_sr": None,
        "fqs": [],
    }
    if not result_path.exists():
        return data

    try:
        y = json.loads(result_path.read_text(encoding="utf-8", errors="replace"))
        models = y.get("models", [])
        fqs = []
        for m in models:
            trained = [
                s for s in m.get("history", [])
                if s.get("name") in ["Training", "Finetuning"] and s.get("training")
            ]
            if trained:
                last_epoch = trained[-1]["training"][-1]
                val_fq = last_epoch.get("benchmarks", {}).get("validation", {}).get("fit_quality", [])
                if val_fq:
                    fqs.append(val_fq[0])
        if fqs:
            data["fqs"] = fqs
            data["neg_fraction"] = sum(fq.get("weights", {}).get("neg_fraction", 0.0) for fq in fqs) / len(fqs)
            data["p99"] = sum(fq.get("weights", {}).get("p99", 0.0) for fq in fqs) / len(fqs)
            data["max"] = max(fq.get("weights", {}).get("max", 0.0) for fq in fqs)
            data["mean"] = sum(fq.get("weights", {}).get("mean", 0.0) for fq in fqs) / len(fqs)
            data["std"] = sum(fq.get("weights", {}).get("std", 0.0) for fq in fqs) / len(fqs)

            # Calibration & Transfer (support both new and legacy keys)
            pd4_vals = [fq.get("data_calibration", {}).get("calib_d4_prob_to_data") or fq.get("calibration", {}).get("ratio_pd4_pd3") for fq in fqs if (fq.get("data_calibration", {}).get("calib_d4_prob_to_data") or fq.get("calibration", {}).get("ratio_pd4_pd3")) is not None]
            data["ratio_pd4_pd3"] = (sum(pd4_vals) / len(pd4_vals)) if pd4_vals else None

            pt4_vals = [fq.get("ttbar_mc_comparison", {}).get("pred_ratio_t4_t3") or fq.get("calibration", {}).get("ratio_pt4_pt3") for fq in fqs if (fq.get("ttbar_mc_comparison", {}).get("pred_ratio_t4_t3") or fq.get("calibration", {}).get("ratio_pt4_pt3")) is not None]
            data["ratio_pt4_pt3"] = (sum(pt4_vals) / len(pt4_vals)) if pt4_vals else None

            true_ratios = [fq.get("ttbar_mc_comparison", {}).get("true_ratio_t4_t3") for fq in fqs if fq.get("ttbar_mc_comparison", {}).get("true_ratio_t4_t3") is not None]
            data["true_ratio_t4_t3"] = (sum(true_ratios) / len(true_ratios)) if true_ratios else None
            data["pred_ratio_t4_t3"] = data["ratio_pt4_pt3"]

            t_t4_vals = [fq.get("ttbar_mc_comparison", {}).get("transfer_d3_to_t4_to_mc") for fq in fqs if fq.get("ttbar_mc_comparison", {}).get("transfer_d3_to_t4_to_mc") is not None]
            data["transfer_d3_to_t4_to_mc"] = (sum(t_t4_vals) / len(t_t4_vals)) if t_t4_vals else None

            t_t3_vals = [fq.get("ttbar_mc_comparison", {}).get("transfer_d3_to_t3_to_mc") for fq in fqs if fq.get("ttbar_mc_comparison", {}).get("transfer_d3_to_t3_to_mc") is not None]
            data["transfer_d3_to_t3_to_mc"] = (sum(t_t3_vals) / len(t_t3_vals)) if t_t3_vals else None

            # Closure
            chi2_pt4_vals = [fq.get("closure", {}).get("chi2_ndf_pt4") for fq in fqs if fq.get("closure", {}).get("chi2_ndf_pt4") is not None]
            data["chi2_ndf_pt4"] = (sum(chi2_pt4_vals) / len(chi2_pt4_vals)) if chi2_pt4_vals else None

            chi2_pt1_vals = [fq.get("closure", {}).get("chi2_ndf_pt1") for fq in fqs if fq.get("closure", {}).get("chi2_ndf_pt1") is not None]
            data["chi2_ndf_pt1"] = (sum(chi2_pt1_vals) / len(chi2_pt1_vals)) if chi2_pt1_vals else None

            chi2_fvt_vals = [fq.get("closure", {}).get("chi2_ndf_fvt") for fq in fqs if fq.get("closure", {}).get("chi2_ndf_fvt") is not None]
            data["chi2_ndf_fvt"] = (sum(chi2_fvt_vals) / len(chi2_fvt_vals)) if chi2_fvt_vals else None

            chi2_svb_vals = [fq.get("closure", {}).get("chi2_ndf_svb_sr") for fq in fqs if fq.get("closure", {}).get("chi2_ndf_svb_sr") is not None]
            data["chi2_ndf_svb_sr"] = (sum(chi2_svb_vals) / len(chi2_svb_vals)) if chi2_svb_vals else None

            chi2_svb_gt0p8_vals = [fq.get("closure", {}).get("chi2_ndf_svb_sr_gt0p8") for fq in fqs if fq.get("closure", {}).get("chi2_ndf_svb_sr_gt0p8") is not None]
            data["chi2_ndf_svb_sr_gt0p8"] = (sum(chi2_svb_gt0p8_vals) / len(chi2_svb_gt0p8_vals)) if chi2_svb_gt0p8_vals else None

            norm_sb_vals = [fq.get("closure", {}).get("norm_closure_sb") for fq in fqs if fq.get("closure", {}).get("norm_closure_sb") is not None]
            data["norm_closure_sb"] = (sum(norm_sb_vals) / len(norm_sb_vals)) if norm_sb_vals else None

            comp_vals = [fq.get("closure", {}).get("composite_chi2_ndf") for fq in fqs if fq.get("closure", {}).get("composite_chi2_ndf") is not None]
            data["composite_chi2_ndf"] = (sum(comp_vals) / len(comp_vals)) if comp_vals else None

            # Online Proxies
            for prx_key in ["loss_sb", "loss_sr", "chi2_fvt_sb", "chi2_t3_sb", "chi2_t4_sb", "chi2_t3_sr", "chi2_t4_sr"]:
                prx_vals = [fq.get("proxies", {}).get(prx_key) for fq in fqs if fq.get("proxies", {}).get(prx_key) is not None]
                data[prx_key] = (sum(prx_vals) / len(prx_vals)) if prx_vals else None

            data["all_gates_passed"] = all(fq.get("verdict", {}).get("all_gates_passed", False) for fq in fqs)
    except Exception as e:
        print(f"Warning: Failed to parse fit quality from {result_path}: {e}", file=sys.stderr)

    return data


def generate_fit_quality_plots(fqs: list[dict], out_dir: Path, source: str = "d3") -> dict:
    """Generate ratio comparison plots for fit quality closure histograms."""
    if not fqs:
        return {}
    summed_hists = {}
    for fq in fqs:
        for k, h in (fq.get("histograms") or {}).items():
            if k not in summed_hists:
                summed_hists[k] = {key: list(h[key]) for key in h if isinstance(h[key], list)}
                continue
            for key in ("target", "model", "target_sumw2", "model_sumw2"):
                if key in h and key in summed_hists[k]:
                    summed_hists[k][key] = [a + b for a, b in zip(summed_hists[k][key], h[key])]

    if not summed_hists:
        return {}

    try:
        from src.classifier.fit_quality_report import _plot
    except Exception:
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    generated = {}
    for hname, h in summed_hists.items():
        if not h.get("target") or sum(h["target"]) <= 0:
            continue
        png_path = out_dir / f"{hname}.png"
        try:
            _plot(hname, h, source, str(png_path))
            generated[hname] = png_path
        except Exception as e:
            print(f"Warning: Failed to plot {hname} in {out_dir}: {e}", file=sys.stderr)

    return generated


def parse_analyze_log(log_path: Path) -> dict:
    """Parse convergence metrics and divergence Z from analyze.log."""
    data = {
        "kfolds": None,
        "best_epoch": None,
        "val_loss": None,
        "tr_loss": None,
        "gap": None,
        "fold_sigma": None,
        "z": None,
        "status_text": None,
        "status_badge": "unknown",
    }
    if not log_path.exists():
        return data

    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        # Example line:
        # [K-Fold Loss Analysis Group 1] K=3 folds: Best Ep=20, Val Loss=0.56580, Tr Loss=0.56279, Gap=+0.00301, Fold σ=0.00182, Z=+1.66σ (Mild Divergence (1.0σ < Z ≤ 2.0σ))
        pattern = r"K=(\d+)\s+folds:\s+Best Ep=(\d+),\s+Val Loss=([0-9.]+),\s+Tr Loss=([0-9.]+),\s+Gap=([+-]?[0-9.]+),\s+Fold σ=([0-9.]+),\s+Z=([+-]?[0-9.]+)σ\s+\((.*)\)"
        m = re.search(pattern, text)
        if m:
            data["kfolds"] = int(m.group(1))
            data["best_epoch"] = int(m.group(2))
            data["val_loss"] = float(m.group(3))
            data["tr_loss"] = float(m.group(4))
            data["gap"] = float(m.group(5))
            data["fold_sigma"] = float(m.group(6))
            data["z"] = float(m.group(7))
            status_raw = m.group(8).strip()
            data["status_text"] = status_raw

            z = data["z"]
            if z <= 1.0:
                data["status_badge"] = "optimal"
            elif z <= 2.0:
                data["status_badge"] = "mild"
            else:
                data["status_badge"] = "overtrained"
    except Exception as e:
        print(f"Warning: Failed to parse {log_path}: {e}", file=sys.stderr)

    return data


def parse_analyze_html(html_path: Path) -> dict:
    """Parse validation AUC metrics from plots/analyze/index.html."""
    data = {
        "auc_4b_vs_3b": None,
        "auc_4b_vs_3b_err": None,
        "auc_4b_vs_3b_data": None,
        "auc_4b_vs_3b_data_err": None,
        "auc_ttbar_vs_data": None,
        "auc_ttbar_vs_data_err": None,
    }
    if not html_path.exists():
        return data

    try:
        text = html_path.read_text(encoding="utf-8", errors="replace")
        # Look for table entries
        # <td><b>Validation AUC: 4b vs 3b</b></td>
        # <td>0.8813 ± 0.0015</td>
        pattern = r"<td><b>Validation AUC:\s*([^<]+)</b></td>\s*<td>\s*([0-9.]+)(?:\s*±\s*([0-9.]+))?"
        matches = re.findall(pattern, text)
        for name, val, err in matches:
            name_clean = name.strip()
            val_f = float(val) if val else None
            err_f = float(err) if err else None

            if name_clean == "4b vs 3b":
                data["auc_4b_vs_3b"] = val_f
                data["auc_4b_vs_3b_err"] = err_f
            elif name_clean == "4b vs 3b data":
                data["auc_4b_vs_3b_data"] = val_f
                data["auc_4b_vs_3b_data_err"] = err_f
            elif name_clean == "ttbar vs data":
                data["auc_ttbar_vs_data"] = val_f
                data["auc_ttbar_vs_data_err"] = err_f
    except Exception as e:
        print(f"Warning: Failed to parse {html_path}: {e}", file=sys.stderr)

    return data


def compute_stats(values: list[float]) -> dict:
    """Compute mean, sample std, min, and max for a list of values."""
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    n = len(valid)
    mean = sum(valid) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in valid) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return {
        "mean": mean,
        "std": std,
        "min": min(valid),
        "max": max(valid),
        "n": n,
    }


def generate_html(models_data: list[dict], out_path: Path, title: str):
    base_dir = out_path.parent

    # Compute ensemble statistics
    val_losses = [m["metrics"]["val_loss"] for m in models_data if m["metrics"]["val_loss"] is not None]
    tr_losses = [m["metrics"]["tr_loss"] for m in models_data if m["metrics"]["tr_loss"] is not None]
    gaps = [m["metrics"]["gap"] for m in models_data if m["metrics"]["gap"] is not None]
    aucs_4b = [m["metrics"]["auc_4b_vs_3b"] for m in models_data if m["metrics"]["auc_4b_vs_3b"] is not None]
    aucs_4b_data = [m["metrics"]["auc_4b_vs_3b_data"] for m in models_data if m["metrics"]["auc_4b_vs_3b_data"] is not None]
    aucs_ttbar = [m["metrics"]["auc_ttbar_vs_data"] for m in models_data if m["metrics"]["auc_ttbar_vs_data"] is not None]

    val_stat = compute_stats(val_losses)
    tr_stat = compute_stats(tr_losses)
    gap_stat = compute_stats(gaps)
    auc_4b_stat = compute_stats(aucs_4b)
    auc_4b_data_stat = compute_stats(aucs_4b_data)
    auc_ttbar_stat = compute_stats(aucs_ttbar)

    chi2_vals = [m["fit_quality"]["composite_chi2_ndf"] for m in models_data if m.get("fit_quality") and m["fit_quality"].get("composite_chi2_ndf") is not None]
    chi2_stat = compute_stats(chi2_vals)
    svb_chi2_vals = [m["fit_quality"]["chi2_ndf_svb_sr"] for m in models_data if m.get("fit_quality") and m["fit_quality"].get("chi2_ndf_svb_sr") is not None]
    svb_chi2_stat = compute_stats(svb_chi2_vals)
    svb_gt0p8_chi2_vals = [m["fit_quality"]["chi2_ndf_svb_sr_gt0p8"] for m in models_data if m.get("fit_quality") and m["fit_quality"].get("chi2_ndf_svb_sr_gt0p8") is not None]
    svb_gt0p8_chi2_stat = compute_stats(svb_gt0p8_chi2_vals)

    # Compute ensemble pull for each model
    for m in models_data:
        vl = m["metrics"]["val_loss"]
        if vl is not None and val_stat["mean"] is not None and val_stat["std"] and val_stat["std"] > 0:
            m["ensemble_pull"] = (vl - val_stat["mean"]) / val_stat["std"]
        else:
            m["ensemble_pull"] = None

    # Max pull in ensemble
    pulls = [abs(m["ensemble_pull"]) for m in models_data if m["ensemble_pull"] is not None]
    max_pull = max(pulls) if pulls else 0.0

    # Hyperparameter summary from first model
    first_cfg = models_data[0]["config"] if models_data else {}
    prescale_str = f"ps{first_cfg.get('prescale')}" if first_cfg.get("prescale") else "Nominal"
    no_sr_str = "No-SR (Sideband Only)" if first_cfg.get("no_sr") else "Standard"
    epochs_str = f"{first_cfg.get('epochs', 20)} Epochs (+{first_cfg.get('ft_epochs', 1)} FT)"
    kfolds_str = f"K={first_cfg.get('kfolds', 3)} Folds"

    # Status counts
    n_optimal = sum(1 for m in models_data if m["metrics"]["status_badge"] == "optimal")
    n_mild = sum(1 for m in models_data if m["metrics"]["status_badge"] == "mild")
    n_over = sum(1 for m in models_data if m["metrics"]["status_badge"] == "overtrained")
    n_total = len(models_data)

    doc_title = title or f"Stage C FvT Training Summary ({prescale_str})"

    # Format numbers safely
    def f_val(val, fmt=".5f", prefix=""):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "—"
        return f"{prefix}{val:{fmt}}"

    lines = []
    lines.append(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(doc_title)}</title>
<style>
:root {{
  --bg: #f8fafc;
  --fg: #0f172a;
  --muted: #64748b;
  --card: #ffffff;
  --line: #e2e8f0;
  --primary: #2563eb;
  --success: #16a34a;
  --warning: #d97706;
  --danger: #dc2626;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}}
header {{
  position: sticky;
  top: 0;
  z-index: 100;
  background: var(--card);
  border-bottom: 1px solid var(--line);
  padding: 12px 24px;
  display: flex;
  flex-wrap: wrap;
  gap: 12px 20px;
  align-items: center;
  box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}}
.roast-back-btn {{
  text-decoration: none;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-weight: 600;
  color: var(--primary);
  background: #eff6ff;
  padding: 6px 14px;
  border-radius: 6px;
  border: 1px solid #bfdbfe;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.15s ease;
}}
.roast-back-btn:hover {{
  background: #dbeafe;
  color: #1d4ed8;
}}
header h1 {{
  font-size: 18px;
  margin: 0;
  font-weight: 700;
  color: #1e293b;
  display: flex;
  align-items: center;
  gap: 8px;
}}
header .badge {{
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 9999px;
  font-weight: 600;
  background: #eff6ff;
  color: var(--primary);
  border: 1px solid #bfdbfe;
}}
header .meta {{ color: var(--muted); font-size: 13px; }}
main {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}

/* Top Scorecards */
.scorecards {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 16px;
  margin-bottom: 28px;
}}
.metric-card {{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px 20px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}}
.metric-card .title {{ font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
.metric-card .val {{ font-size: 22px; font-weight: 700; margin: 6px 0 2px; color: var(--fg); }}
.metric-card .sub {{ font-size: 12px; color: var(--muted); }}

.status-pill {{
  display: inline-block;
  padding: 2px 8px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
}}
.status-optimal {{ background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }}
.status-mild {{ background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }}
.status-overtrained {{ background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }}
.status-neutral {{ background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }}

/* Table Styling */
.table-container {{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 32px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}}
.table-header {{
  padding: 14px 20px;
  border-bottom: 1px solid var(--line);
  font-weight: 600;
  font-size: 15px;
  background: #f8fafc;
  display: flex;
  justify-content: space-between;
  align-items: center;
}}
.table-header .note {{ font-size: 12px; font-weight: 400; color: var(--muted); }}
table.report-table {{
  width: 100%;
  border-collapse: collapse;
  text-align: left;
  font-size: 13px;
}}
table.report-table th, table.report-table td {{
  padding: 10px 14px;
  border-bottom: 1px solid var(--line);
  vertical-align: middle;
}}
table.report-table th {{ background: #f8fafc; color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.03em; }}
table.report-table tr:hover td {{ background: #f1f5f9; }}
table.report-table tr.summary-row {{ background: #f8fafc; font-weight: 600; border-top: 2px solid var(--line); }}
table.report-table tr.summary-row td {{ border-bottom: none; }}

/* Links */
a.link-btn {{
  display: inline-block;
  padding: 2px 7px;
  font-size: 11px;
  font-weight: 600;
  text-decoration: none;
  border-radius: 4px;
  background: #eff6ff;
  color: var(--primary);
  border: 1px solid #bfdbfe;
  margin-right: 4px;
  transition: all 0.1s ease;
}}
a.link-btn:hover {{ background: #dbeafe; }}
a.link-subtle {{
  color: var(--primary);
  text-decoration: none;
  font-weight: 600;
}}
a.link-subtle:hover {{ text-decoration: underline; }}

/* Gallery Grid */
.section-title {{
  font-size: 16px;
  font-weight: 700;
  margin: 32px 0 14px;
  color: #1e293b;
  display: flex;
  align-items: center;
  gap: 10px;
}}
.tabs {{
  display: flex;
  gap: 8px;
  margin-bottom: 16px;
  border-bottom: 1px solid var(--line);
  padding-bottom: 8px;
}}
.tab-btn {{
  background: none;
  border: none;
  padding: 6px 14px;
  font-size: 13px;
  font-weight: 600;
  color: var(--muted);
  cursor: pointer;
  border-radius: 6px;
  transition: all 0.15s ease;
}}
.tab-btn.active {{
  background: #eff6ff;
  color: var(--primary);
}}
.tab-pane {{
  display: none;
}}
.tab-pane.active {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
}}
.card {{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  cursor: zoom-in;
  transition: transform 0.1s ease, box-shadow 0.1s ease;
}}
.card:hover {{
  transform: translateY(-2px);
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}}
.card img {{
  display: block;
  width: 100%;
  height: auto;
  aspect-ratio: 4 / 3;
  object-fit: contain;
  background: #ffffff;
}}
.card .cap {{
  padding: 8px 12px;
  font-size: 12px;
  border-top: 1px solid var(--line);
  background: #ffffff;
  color: var(--muted);
  display: flex;
  justify-content: space-between;
  align-items: center;
}}
.card .cap b {{ color: var(--fg); font-weight: 600; }}

/* Lightbox Modal */
#lb {{
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.95);
  display: none;
  z-index: 1000;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}}
#lb.on {{ display: flex; }}
#lb img {{
  max-width: 95vw;
  max-height: 86vh;
  background: #ffffff;
  border-radius: 6px;
  box-shadow: 0 20px 25px -5px rgba(0,0,0,0.5);
}}
#lb .cap {{ color: #f8fafc; margin-top: 12px; font-size: 14px; }}
#lb .x {{ position: absolute; top: 16px; right: 24px; color: #94a3b8; font-size: 32px; cursor: pointer; }}
#lb .nav {{ position: absolute; top: 50%; transform: translateY(-50%); color: #94a3b8; font-size: 48px; cursor: pointer; padding: 0 20px; user-select: none; }}
#lb .prev {{ left: 0; }} #lb .next {{ right: 0; }}
#lb .x:hover, #lb .nav:hover {{ color: #ffffff; }}
</style>
</head>
<body>

<header>
  <a href="#" onclick="goBackToRoastIndex(); return false;" class="roast-back-btn">← Roast Index</a>
  <h1>{html.escape(doc_title)} <span class="badge">{n_total} {"Subsamples" if n_total > 1 else "Model"}</span></h1>
  <span class="meta">{no_sr_str} · Prescale {prescale_str} · {kfolds_str} · {epochs_str}</span>
</header>

<main>
  <!-- Scorecards -->
  <div class="scorecards">
    <div class="metric-card">
      <div class="title">{"Ensemble " if n_total > 1 else ""}Validation Loss</div>
      <div class="val">{f_val(val_stat['mean'])} {f'<span style="font-size:14px; font-weight:400; color:var(--muted)">± {f_val(val_stat["std"])}</span>' if n_total > 1 else ''}</div>
      <div class="sub">{f"Min: {f_val(val_stat['min'])} · Max: {f_val(val_stat['max'])}" if n_total > 1 else "K-Fold Mean Loss"}</div>
    </div>
    
    <div class="metric-card">
      <div class="title">Generalization Gap ⟨ΔL⟩</div>
      <div class="val">{f_val(gap_stat['mean'], prefix='+')} {f'<span style="font-size:14px; font-weight:400; color:var(--muted)">± {f_val(gap_stat["std"])}</span>' if n_total > 1 else ''}</div>
      <div class="sub">{"⟨L_val⟩ - ⟨L_train⟩ across " + str(n_total) + " models" if n_total > 1 else "L_val - L_train"}</div>
    </div>
    
    <div class="metric-card">
      <div class="title">{"Ensemble " if n_total > 1 else ""}AUC (4b vs 3b)</div>
      <div class="val">{f_val(auc_4b_stat['mean'], fmt='.4f')} {f'<span style="font-size:14px; font-weight:400; color:var(--muted)">± {f_val(auc_4b_stat["std"], fmt=".4f")}</span>' if n_total > 1 else ''}</div>
      <div class="sub">{f"Min: {f_val(auc_4b_stat['min'], fmt='.4f')} · Max: {f_val(auc_4b_stat['max'], fmt='.4f')}" if n_total > 1 else "Cross-validation AUC"}</div>
    </div>

    <div class="metric-card">
      <div class="title">{"Ensemble Compatibility" if n_total > 1 else "Model Status"}</div>
      <div class="val">
        <span class="status-pill status-optimal">{n_optimal} Optimal</span>
        {f'<span class="status-pill status-mild">{n_mild} Mild</span>' if (n_mild or n_total > 1) else ''}
        {f'<span class="status-pill status-overtrained">{n_over} Overtrained</span>' if n_over else ''}
      </div>
      <div class="sub">{f"Max Ensemble Pull: {max_pull:.2f}σ (All models compatible)" if n_total > 1 else "Overfitting Diagnostics"}</div>
    </div>

    <div class="metric-card">
      <div class="title">Kinematic Closure ⟨χ²/ndf⟩</div>
      <div class="val">{f_val(chi2_stat['mean'], fmt='.2f') if chi2_stat['mean'] is not None else '—'} <span style="font-size:14px; font-weight:400; color:var(--muted)">{f"± {f_val(chi2_stat['std'], fmt='.2f')}" if (chi2_stat['std'] and n_total > 1) else ''}</span></div>
      <div class="sub">{f"Min: {f_val(chi2_stat['min'], fmt='.2f')} · Max: {f_val(chi2_stat['max'], fmt='.2f')}" if (chi2_stat['min'] is not None and n_total > 1) else "Target: → 1.00 (pt4 & pt1)"}</div>
    </div>
  </div>

  <!-- Convergence & Loss Statistics Table -->
  <div class="table-container">
    <div class="table-header">
      <span>{"Cross-Sample Convergence, Fit Quality & Loss Statistics" if n_total > 1 else "Model Convergence, Fit Quality & Loss Statistics"}</span>
      <span class="note">Validation loss, test AUC, and kinematic reweighting closure evaluate model quality</span>
    </div>
    <table class="report-table">
      <thead>
        <tr>
          <th>{"Subsample" if n_total > 1 else "Model"}</th>
          <th>Best Ep</th>
          <th>Val Loss ⟨L_val⟩</th>
          <th>Train Loss ⟨L_tr⟩</th>
          <th>Gap ΔL</th>
          <th>Fold Spread σ</th>
          <th>Fold Pull Z</th>
          <th>Ensemble Pull</th>
          <th>Fold Status</th>
          <th>AUC (4b vs 3b)</th>
          <th>AUC (4b vs 3b data)</th>
          <th>AUC (ttbar vs data)</th>
          <th>χ²/ndf (pt4)</th>
          <th>χ²/ndf (SvB SR)</th>
          <th>χ²/ndf (SvB &gt; 0.8)</th>
          <th>Score (χ²)</th>
          <th>f(w &lt; 0)</th>
          <th>Quick Links</th>
        </tr>
      </thead>
      <tbody>
""")

    for m in models_data:
        m_name = m["name"]
        met = m["metrics"]
        rel_model = "" if m["rel_dir"] in (".", "") else f"{m['rel_dir'].rstrip('/')}/"

        ep = met["best_epoch"] if met["best_epoch"] is not None else "—"
        vl = f_val(met["val_loss"])
        tl = f_val(met["tr_loss"])
        gap = f_val(met["gap"], prefix="+" if (met["gap"] or 0) > 0 else "")
        s_fold = f_val(met["fold_sigma"])
        z_str = f_val(met["z"], fmt=".2f", prefix="+" if (met["z"] or 0) > 0 else "") + "σ" if met["z"] is not None else "—"

        ens_pull = m.get("ensemble_pull")
        ens_pull_str = f_val(ens_pull, fmt=".2f", prefix="+" if (ens_pull or 0) > 0 else "") + "σ" if ens_pull is not None else "—"
        pull_color = "var(--fg)"
        if ens_pull is not None:
            if abs(ens_pull) > 2.5:
                pull_color = "var(--danger)"
            elif abs(ens_pull) > 1.8:
                pull_color = "var(--warning)"

        badge_cls = f"status-{met['status_badge']}"
        badge_text = met["status_text"] or met["status_badge"].title()

        auc_4b = f"{met['auc_4b_vs_3b']:.4f} ± {met['auc_4b_vs_3b_err']:.4f}" if met["auc_4b_vs_3b"] is not None and met["auc_4b_vs_3b_err"] is not None else f_val(met["auc_4b_vs_3b"], fmt=".4f")
        auc_4b_d = f"{met['auc_4b_vs_3b_data']:.4f} ± {met['auc_4b_vs_3b_data_err']:.4f}" if met["auc_4b_vs_3b_data"] is not None and met["auc_4b_vs_3b_data_err"] is not None else f_val(met["auc_4b_vs_3b_data"], fmt=".4f")
        auc_tt = f"{met['auc_ttbar_vs_data']:.4f} ± {met['auc_ttbar_vs_data_err']:.4f}" if met["auc_ttbar_vs_data"] is not None and met["auc_ttbar_vs_data_err"] is not None else f_val(met["auc_ttbar_vs_data"], fmt=".4f")

        fq = m.get("fit_quality") or {}
        chi2_pt4_str = f_val(fq.get("chi2_ndf_pt4"), fmt=".2f")
        chi2_svb_str = f_val(fq.get("chi2_ndf_svb_sr"), fmt=".2f")
        chi2_svb_gt0p8_str = f_val(fq.get("chi2_ndf_svb_sr_gt0p8"), fmt=".2f")
        score_str = f_val(fq.get("composite_chi2_ndf"), fmt=".2f")
        neg_frac = fq.get("neg_fraction")
        neg_str = f"{neg_frac * 100:.2f}%" if neg_frac is not None else "—"

        link_analyze = f"{rel_model}plots/analyze/index.html"
        link_gallery = f"{rel_model}plots/index.html"
        link_yields = f"{rel_model}yields.html"

        lines.append(f"""        <tr>
          <td><a class="link-subtle" href="{link_analyze}">{m_name}</a></td>
          <td>{ep}</td>
          <td><b>{vl}</b></td>
          <td>{tl}</td>
          <td>{gap}</td>
          <td style="color:var(--muted)">{s_fold}</td>
          <td>{z_str}</td>
          <td style="color:{pull_color}; font-weight:600">{ens_pull_str}</td>
          <td><span class="status-pill {badge_cls}">{badge_text}</span></td>
          <td>{auc_4b}</td>
          <td>{auc_4b_d}</td>
          <td>{auc_tt}</td>
          <td>{chi2_pt4_str}</td>
          <td>{chi2_svb_str}</td>
          <td>{chi2_svb_gt0p8_str}</td>
          <td><b>{score_str}</b></td>
          <td>{neg_str}</td>
          <td>
            <a class="link-btn" href="{link_analyze}">Report</a>
            <a class="link-btn" href="{link_gallery}">Gallery</a>
            <a class="link-btn" href="{link_yields}">Yields</a>
          </td>
        </tr>""")

    # Ensemble Summary Row
    mean_vl_str = f_val(val_stat["mean"]) + f" ± {val_stat['std']:.5f}" if val_stat["mean"] is not None else "—"
    mean_tl_str = f_val(tr_stat["mean"]) + f" ± {tr_stat['std']:.5f}" if tr_stat["mean"] is not None else "—"
    mean_gap_str = (f"+{gap_stat['mean']:.5f}" if (gap_stat['mean'] or 0) > 0 else f"{gap_stat['mean']:.5f}") + f" ± {gap_stat['std']:.5f}" if gap_stat["mean"] is not None else "—"
    mean_auc_4b = f"{auc_4b_stat['mean']:.4f} ± {auc_4b_stat['std']:.4f}" if auc_4b_stat["mean"] is not None else "—"
    mean_auc_4b_d = f"{auc_4b_data_stat['mean']:.4f} ± {auc_4b_data_stat['std']:.4f}" if auc_4b_data_stat["mean"] is not None else "—"
    mean_auc_tt = f"{auc_ttbar_stat['mean']:.4f} ± {auc_ttbar_stat['std']:.4f}" if auc_ttbar_stat["mean"] is not None else "—"
    mean_chi2_str = f"{chi2_stat['mean']:.2f} ± {chi2_stat['std']:.2f}" if chi2_stat["mean"] is not None else "—"
    mean_svb_chi2_str = f"{svb_chi2_stat['mean']:.2f} ± {svb_chi2_stat['std']:.2f}" if svb_chi2_stat["mean"] is not None else "—"
    mean_svb_gt0p8_chi2_str = f"{svb_gt0p8_chi2_stat['mean']:.2f} ± {svb_gt0p8_chi2_stat['std']:.2f}" if svb_gt0p8_chi2_stat["mean"] is not None else "—"

    lines.append(f"""        <tr class="summary-row">
          <td><b>Ensemble Mean ± Std</b></td>
          <td>—</td>
          <td><b>{mean_vl_str}</b></td>
          <td>{mean_tl_str}</td>
          <td>{mean_gap_str}</td>
          <td>—</td>
          <td>—</td>
          <td>0.00σ</td>
          <td><span class="status-pill status-neutral">{n_total} Models</span></td>
          <td><b>{mean_auc_4b}</b></td>
          <td><b>{mean_auc_4b_d}</b></td>
          <td><b>{mean_auc_tt}</b></td>
          <td>—</td>
          <td><b>{mean_svb_chi2_str}</b></td>
          <td><b>{mean_svb_gt0p8_chi2_str}</b></td>
          <td><b>{mean_chi2_str}</b></td>
          <td>—</td>
          <td>—</td>
        </tr>
      </tbody>
    </table>
  </div>

  <!-- Loaded Dataset Yields Table -->
  <div class="table-container">
    <div class="table-header">
      <span>Dataset Statistics & Loaded Yields</span>
      <span class="note">Event counts and effective weights per class loaded into HCR dataloaders</span>
    </div>
    <table class="report-table">
      <thead>
        <tr>
          <th>Subsample</th>
          <th>Prescale</th>
          <th>Selection</th>
          <th>Data 4b (d4) Count</th>
          <th>Data 4b (d4) Weight</th>
          <th>Data 3b (d3) Count</th>
          <th>Data 3b (d3) Weight</th>
          <th>ttbar 4b (t4) Count</th>
          <th>ttbar 4b (t4) Weight</th>
          <th>ttbar 3b (t3) Count</th>
          <th>ttbar 3b (t3) Weight</th>
          <th>Data 4b/3b Ratio</th>
          <th>ttbar 4b/3b Ratio</th>
        </tr>
      </thead>
      <tbody>
""")

    for m in models_data:
        m_name = m["name"]
        cfg = m["config"]
        yd = m["yields"]

        ps = f"ps{cfg['prescale']}" if cfg.get("prescale") else "Nominal"
        sel = "SB Only (--no-SR)" if cfg.get("no_sr") else "Full Region"

        d4_c = f"{yd['d4_count']:,}" if yd.get("d4_count") is not None else "—"
        d4_w = f"{yd['d4_weight']:,.1f}" if yd.get("d4_weight") is not None else "—"
        d3_c = f"{yd['d3_count']:,}" if yd.get("d3_count") is not None else "—"
        d3_w = f"{yd['d3_weight']:,.1f}" if yd.get("d3_weight") is not None else "—"
        t4_c = f"{yd['t4_count']:,}" if yd.get("t4_count") is not None else "—"
        t4_w = f"{yd['t4_weight']:,.1f}" if yd.get("t4_weight") is not None else "—"
        t3_c = f"{yd['t3_count']:,}" if yd.get("t3_count") is not None else "—"
        t3_w = f"{yd['t3_weight']:,.1f}" if yd.get("t3_weight") is not None else "—"

        d_ratio = f"{yd['d4_weight'] / yd['d3_weight']:.4f}" if yd.get("d4_weight") and yd.get("d3_weight") else "—"
        t_ratio = f"{yd['t4_weight'] / yd['t3_weight']:.4f}" if yd.get("t4_weight") and yd.get("t3_weight") else "—"

        lines.append(f"""        <tr>
          <td><b>{m_name}</b></td>
          <td>{ps}</td>
          <td>{sel}</td>
          <td>{d4_c}</td>
          <td>{d4_w}</td>
          <td>{d3_c}</td>
          <td>{d3_w}</td>
          <td>{t4_c}</td>
          <td>{t4_w}</td>
          <td>{t3_c}</td>
          <td>{t3_w}</td>
          <td><b>{d_ratio}</b></td>
          <td><b>{t_ratio}</b></td>
        </tr>""")

    lines.append("""      </tbody>
    </table>
  </div>

  <!-- Diagnostic & Metric Plots Gallery -->
  <div class="section-title">
    <span>{"Cross-Sample Diagnostic Plots" if n_total > 1 else "Diagnostic & Metric Plots"}</span>
  </div>
""")

    tab_specs = [
        {
            "id": "tab-closure-evo",
            "name": "Kinematic Closure Evolution",
            "plots": [
                ("plots/analyze/HCR/closure-evolution-fvt_sb-1.png", "Closure Evolution: FvT Weight (SB)", "χ²/ndf: fvt_sb"),
                ("plots/analyze/HCR/closure-evolution-pt4_sb-1.png", "Closure Evolution: 4th Jet pT (SB)", "χ²/ndf: pt4_sb"),
                ("plots/analyze/HCR/closure-evolution-pt1_sb-1.png", "Closure Evolution: Leading Jet pT (SB)", "χ²/ndf: pt1_sb"),
            ],
        },
        {
            "id": "tab-closure-dist",
            "name": "Closure Histograms & Ratios",
            "plots": [
                ("plots/analyze/fit_quality/pt4_sb.png", "4th Jet pT (SB) [Data vs Model]", "pt4_sb"),
                ("plots/analyze/fit_quality/pt1_sb.png", "Leading Jet pT (SB) [Data vs Model]", "pt1_sb"),
                ("plots/analyze/fit_quality/fvt_sb.png", "Transfer Weight w (SB) [Data vs Model]", "fvt_sb"),
                ("plots/analyze/fit_quality/t3_sb.png", "pt3 (SB) [ttbar MC vs Source x Prob]", "t3_sb"),
                ("plots/analyze/fit_quality/t4_sb.png", "pt4 (SB) [ttbar MC vs Source x Prob]", "t4_sb"),
                ("plots/analyze/fit_quality/t3_sr.png", "pt3 (SR) [ttbar MC vs Source x Prob]", "t3_sr"),
                ("plots/analyze/fit_quality/t4_sr.png", "pt4 (SR) [ttbar MC vs Source x Prob]", "t4_sr"),
            ],
        },
        {
            "id": "tab-loss",
            "name": "Loss & Convergence",
            "plots": [
                ("plots/analyze/HCR/kfold-loss-divergence-1.png", "Loss Convergence & Divergence Pull", "Loss Divergence"),
                ("plots/analyze/HCR/learning-rate-1.png", "Learning Rate Schedule", "LR Schedule"),
                ("plots/analyze/HCR/fold-consistency-1.png", "Fold Consistency", "Fold Consistency"),
            ],
        },
        {
            "id": "tab-roc",
            "name": "ROC Discrimination Curves",
            "plots": [
                ("plots/analyze/HCR/roc-4b_vs_3b-1.png", "ROC: 4b vs 3b (Val)", "4b vs 3b"),
                ("plots/analyze/HCR/roc-4b_vs_3b_data-1.png", "ROC: 4b vs 3b Data", "4b vs 3b Data"),
                ("plots/analyze/HCR/roc-4b_vs_3b_data_sb-1.png", "ROC: 4b vs 3b Data (SB)", "4b vs 3b Data SB"),
                ("plots/analyze/HCR/roc-ttbar_vs_data-1.png", "ROC: ttbar vs Data", "ttbar vs Data"),
                ("plots/analyze/HCR/roc-ttbar_vs_data_sb-1.png", "ROC: ttbar vs Data (SB)", "ttbar vs Data SB"),
                ("plots/analyze/HCR/roc-ttbar_vs_data_sr-1.png", "ROC: ttbar vs Data (SR)", "ttbar vs Data SR"),
            ],
        },
        {
            "id": "tab-auc",
            "name": "AUC Evolution Across Epochs",
            "plots": [
                ("plots/analyze/HCR/auc-evolution-4b_vs_3b-1.png", "AUC Evolution: 4b vs 3b", "4b vs 3b"),
                ("plots/analyze/HCR/auc-evolution-4b_vs_3b_data-1.png", "AUC Evolution: 4b vs 3b Data", "4b vs 3b Data"),
                ("plots/analyze/HCR/auc-evolution-4b_vs_3b_data_sb-1.png", "AUC Evolution: 4b vs 3b Data (SB)", "4b vs 3b Data SB"),
                ("plots/analyze/HCR/auc-evolution-ttbar_vs_data-1.png", "AUC Evolution: ttbar vs Data", "ttbar vs Data"),
                ("plots/analyze/HCR/auc-evolution-ttbar_vs_data_sb-1.png", "AUC Evolution: ttbar vs Data (SB)", "ttbar vs Data SB"),
                ("plots/analyze/HCR/auc-evolution-ttbar_vs_data_sr-1.png", "AUC Evolution: ttbar vs Data (SR)", "ttbar vs Data SR"),
            ],
        },
    ]

    tab_cards = {}
    for spec in tab_specs:
        cards = []
        if n_total == 1:
            m = models_data[0]
            rel = "" if m["rel_dir"] in (".", "") else f"{m['rel_dir'].rstrip('/')}/"
            for rel_img, title, subtitle in spec["plots"]:
                if (m["dir"] / rel_img).exists():
                    cards.append({
                        "src": f"{rel}{rel_img}",
                        "title": title,
                        "sub": subtitle,
                    })
        else:
            for rel_img, title, subtitle in spec["plots"]:
                for m in models_data:
                    rel = "" if m["rel_dir"] in (".", "") else f"{m['rel_dir'].rstrip('/')}/"
                    if (m["dir"] / rel_img).exists():
                        cards.append({
                            "src": f"{rel}{rel_img}",
                            "title": f"{m['name']} — {subtitle}",
                            "sub": title,
                        })
        tab_cards[spec["id"]] = cards

    active_tabs = [spec for spec in tab_specs if tab_cards.get(spec["id"])]
    if not active_tabs:
        active_tabs = tab_specs[:1]

    lines.append('  <div class="tabs">')
    for i, spec in enumerate(active_tabs):
        cls = "tab-btn active" if i == 0 else "tab-btn"
        lines.append(f'    <button class="{cls}" onclick="switchTab(\'{spec["id"]}\', this)">{html.escape(spec["name"])}</button>')
    lines.append('  </div>\n')

    for i, spec in enumerate(active_tabs):
        cls = "tab-pane active" if i == 0 else "tab-pane"
        lines.append(f'  <!-- {spec["name"]} -->')
        lines.append(f'  <div id="{spec["id"]}" class="{cls}">')
        for c in tab_cards.get(spec["id"], []):
            lines.append(f"""    <div class="card" onclick="openLightbox('{c['src']}', '{html.escape(c['title'])}')">
      <img src="{c['src']}" alt="{html.escape(c['title'])}" loading="lazy">
      <div class="cap">
        <b>{html.escape(c['title'])}</b>
        <span>{html.escape(c['sub'])}</span>
      </div>
    </div>""")
        lines.append('  </div>\n')

    lines.append("""  </div>
</main>

<!-- Lightbox Modal -->
<div id="lb" onclick="if(event.target===this) closeLightbox()">
  <span class="x" onclick="closeLightbox()">&times;</span>
  <span class="nav prev" onclick="prevLightbox(event)">&lsaquo;</span>
  <img id="lb-img" src="" alt="Enlarged Plot">
  <div id="lb-cap" class="cap"></div>
  <span class="nav next" onclick="nextLightbox(event)">&rsaquo;</span>
</div>

<script>
function switchTab(tabId, btn) {
  document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  btn.classList.add('active');
}

let activeImages = [];
let currentIndex = 0;

function updateActiveImages() {
  const currentPane = document.querySelector('.tab-pane.active');
  activeImages = Array.from(currentPane.querySelectorAll('.card')).map(card => {
    return {
      src: card.querySelector('img').getAttribute('src'),
      cap: card.querySelector('.cap').innerText.replace('\\n', ' — ')
    };
  });
}

function openLightbox(src, cap) {
  updateActiveImages();
  currentIndex = activeImages.findIndex(item => item.src === src);
  if (currentIndex === -1) currentIndex = 0;
  showLightbox();
}

function showLightbox() {
  const item = activeImages[currentIndex];
  document.getElementById('lb-img').src = item.src;
  document.getElementById('lb-cap').innerText = item.cap;
  document.getElementById('lb').classList.add('on');
}

function closeLightbox() {
  document.getElementById('lb').classList.remove('on');
}

function prevLightbox(e) {
  e.stopPropagation();
  currentIndex = (currentIndex - 1 + activeImages.length) % activeImages.length;
  showLightbox();
}

function nextLightbox(e) {
  e.stopPropagation();
  currentIndex = (currentIndex + 1) % activeImages.length;
  showLightbox();
}

function goBackToRoastIndex() {
  if (document.referrer && (document.referrer.includes('index.html') || document.referrer.endsWith('/'))) {
    window.location.href = document.referrer;
    return;
  }
  const href = window.location.href.split('?')[0].split('#')[0];
  if (href.includes('/output/')) {
    window.location.href = href.split('/output/')[0] + '/index.html';
    return;
  }
  const m = href.match(new RegExp('(.*\\/[^\\/]+_\\d{8}_[0-9a-f]{7}-[0-9a-f]{7})\\/'));
  if (m) {
    window.location.href = m[1] + '/index.html';
    return;
  }
  window.location.href = '../index.html';
}

document.addEventListener('keydown', e => {
  if (!document.getElementById('lb').classList.contains('on')) return;
  if (e.key === 'Escape') closeLightbox();
  if (e.key === 'ArrowLeft') prevLightbox(e);
  if (e.key === 'ArrowRight') nextLightbox(e);
});
</script>

</body>
</html>
""")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated training summary dashboard at: {out_path}")


def main():
    args = parse_args()
    models_dir = Path(args.models_dir).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Scan for mix_* subdirectories
    mix_dirs = []
    for p in models_dir.glob("mix_*"):
        if p.is_dir():
            # sort key based on integer index
            name = p.name
            m_idx = re.search(r"mix_(\d+)", name)
            idx = int(m_idx.group(1)) if m_idx else 999
            mix_dirs.append((idx, p))

    mix_dirs.sort(key=lambda x: x[0])
    if not mix_dirs:
        # Check if models_dir itself is a single model directory (e.g. stat-only Phase C)
        if (
            (models_dir / "analyze.log").exists()
            or (models_dir / "wfs" / "train.yml").exists()
            or (models_dir / "train.done").exists()
            or (models_dir / "yields.html").exists()
            or (models_dir / "models").is_dir()
        ):
            mix_dirs = [(0, models_dir)]
        else:
            print(f"Error: No mix_* directories or model artifacts found in {models_dir}", file=sys.stderr)
            sys.exit(1)

    models_data = []
    base_dir = out_path.parent

    for idx, p in mix_dirs:
        m_name = p.name
        rel_dir = os.path.relpath(p, base_dir)

        cfg = parse_train_yml(p / "wfs" / "train.yml")
        yd = parse_yields_json(p / "yields.json")
        met_log = parse_analyze_log(p / "analyze.log")
        met_html = parse_analyze_html(p / "plots" / "analyze" / "index.html")

        res_file = locate_or_fetch_result_json(p)
        fq = parse_fit_quality_json(res_file)
        if fq.get("fqs"):
            fq_plot_dir = p / "plots" / "analyze" / "fit_quality"
            src = fq.get("fqs", [{}])[0].get("source", "d3")
            generate_fit_quality_plots(fq["fqs"], fq_plot_dir, source=src)

        # Merge metrics
        metrics = {}
        metrics.update(met_log)
        metrics.update(met_html)

        models_data.append({
            "index": idx,
            "name": m_name,
            "dir": p,
            "rel_dir": rel_dir,
            "config": cfg,
            "yields": yd,
            "metrics": metrics,
            "fit_quality": fq,
        })

    title = args.title or f"FvT Training Summary — {models_dir.parent.parent.name if models_dir.parent.name == 'bkg_syst_C_FvT' else models_dir.name}"
    generate_html(models_data, out_path, title)


if __name__ == "__main__":
    main()
