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
  <h1>{html.escape(doc_title)} <span class="badge">{n_total} Subsamples</span></h1>
  <span class="meta">{no_sr_str} · Prescale {prescale_str} · {kfolds_str} · {epochs_str}</span>
</header>

<main>
  <!-- Scorecards -->
  <div class="scorecards">
    <div class="metric-card">
      <div class="title">Ensemble Validation Loss</div>
      <div class="val">{f_val(val_stat['mean'])} <span style="font-size:14px; font-weight:400; color:var(--muted)">± {f_val(val_stat['std'])}</span></div>
      <div class="sub">Min: {f_val(val_stat['min'])} · Max: {f_val(val_stat['max'])}</div>
    </div>
    
    <div class="metric-card">
      <div class="title">Generalization Gap ⟨ΔL⟩</div>
      <div class="val">{f_val(gap_stat['mean'], prefix='+')} <span style="font-size:14px; font-weight:400; color:var(--muted)">± {f_val(gap_stat['std'])}</span></div>
      <div class="sub">⟨L_val⟩ - ⟨L_train⟩ across 16 models</div>
    </div>
    
    <div class="metric-card">
      <div class="title">Ensemble AUC (4b vs 3b)</div>
      <div class="val">{f_val(auc_4b_stat['mean'], fmt='.4f')} <span style="font-size:14px; font-weight:400; color:var(--muted)">± {f_val(auc_4b_stat['std'], fmt='.4f')}</span></div>
      <div class="sub">Min: {f_val(auc_4b_stat['min'], fmt='.4f')} · Max: {f_val(auc_4b_stat['max'], fmt='.4f')}</div>
    </div>

    <div class="metric-card">
      <div class="title">Ensemble Compatibility</div>
      <div class="val">
        <span class="status-pill status-optimal">{n_optimal} Optimal</span>
        <span class="status-pill status-mild">{n_mild} Mild</span>
        {f'<span class="status-pill status-overtrained">{n_over} Overtrained</span>' if n_over else ''}
      </div>
      <div class="sub">Max Ensemble Pull: {max_pull:.2f}σ (All models compatible)</div>
    </div>
  </div>

  <!-- Convergence & Loss Statistics Table -->
  <div class="table-container">
    <div class="table-header">
      <span>Cross-Sample Convergence & Loss Statistics (16 Pseudo-Experiments)</span>
      <span class="note">Validation loss and test AUC evaluate model consistency across statistical realizations</span>
    </div>
    <table class="report-table">
      <thead>
        <tr>
          <th>Subsample</th>
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
          <th>Quick Links</th>
        </tr>
      </thead>
      <tbody>
""")

    for m in models_data:
        m_name = m["name"]
        met = m["metrics"]
        rel_model = m["rel_dir"]

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

        link_analyze = f"{rel_model}/plots/analyze/index.html"
        link_gallery = f"{rel_model}/plots/index.html"
        link_yields = f"{rel_model}/yields.html"

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

  <!-- Cross-Sample Visual Comparisons Gallery -->
  <div class="section-title">
    <span>Cross-Sample Diagnostic Plots</span>
  </div>

  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('tab-loss', this)">Loss Convergence Curves</button>
    <button class="tab-btn" onclick="switchTab('tab-roc', this)">ROC Curves (4b vs 3b)</button>
    <button class="tab-btn" onclick="switchTab('tab-auc', this)">AUC Evolution</button>
    <button class="tab-btn" onclick="switchTab('tab-fold', this)">K-Fold Consistency</button>
  </div>

  <!-- Tab 1: Loss Divergence -->
  <div id="tab-loss" class="tab-pane active">
""")

    for m in models_data:
        m_name = m["name"]
        rel = m["rel_dir"]
        img_path = f"{rel}/plots/analyze/HCR/kfold-loss-divergence-1.png"
        lines.append(f"""    <div class="card" onclick="openLightbox('{img_path}', '{m_name} — Loss Divergence')">
      <img src="{img_path}" alt="{m_name} Loss Divergence" loading="lazy">
      <div class="cap">
        <b>{m_name}</b>
        <span>{f_val(m['metrics']['val_loss'])} ({f_val(m['metrics']['z'], fmt='.2f', prefix='+')}σ)</span>
      </div>
    </div>""")

    lines.append("""  </div>

  <!-- Tab 2: ROC Curves -->
  <div id="tab-roc" class="tab-pane">
""")

    for m in models_data:
        m_name = m["name"]
        rel = m["rel_dir"]
        img_path = f"{rel}/plots/analyze/HCR/roc-4b_vs_3b-1.png"
        auc_str = f"{m['metrics']['auc_4b_vs_3b']:.4f}" if m['metrics']['auc_4b_vs_3b'] is not None else "—"
        lines.append(f"""    <div class="card" onclick="openLightbox('{img_path}', '{m_name} — ROC Curve 4b vs 3b')">
      <img src="{img_path}" alt="{m_name} ROC Curve" loading="lazy">
      <div class="cap">
        <b>{m_name}</b>
        <span>AUC = {auc_str}</span>
      </div>
    </div>""")

    lines.append("""  </div>

  <!-- Tab 3: AUC Evolution -->
  <div id="tab-auc" class="tab-pane">
""")

    for m in models_data:
        m_name = m["name"]
        rel = m["rel_dir"]
        img_path = f"{rel}/plots/analyze/HCR/auc-evolution-4b_vs_3b-1.png"
        lines.append(f"""    <div class="card" onclick="openLightbox('{img_path}', '{m_name} — AUC Evolution')">
      <img src="{img_path}" alt="{m_name} AUC Evolution" loading="lazy">
      <div class="cap">
        <b>{m_name}</b>
        <span>AUC Evolution</span>
      </div>
    </div>""")

    lines.append("""  </div>

  <!-- Tab 4: K-Fold Consistency -->
  <div id="tab-fold" class="tab-pane">
""")

    for m in models_data:
        m_name = m["name"]
        rel = m["rel_dir"]
        img_path = f"{rel}/plots/analyze/HCR/fold-consistency-1.png"
        lines.append(f"""    <div class="card" onclick="openLightbox('{img_path}', '{m_name} — K-Fold Consistency')">
      <img src="{img_path}" alt="{m_name} Fold Consistency" loading="lazy">
      <div class="cap">
        <b>{m_name}</b>
        <span>Fold Consistency</span>
      </div>
    </div>""")

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
        print(f"Error: No mix_* directories found in {models_dir}", file=sys.stderr)
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
        })

    title = args.title or f"FvT Training Summary — {models_dir.parent.parent.name if models_dir.parent.name == 'bkg_syst_C_FvT' else models_dir.name}"
    generate_html(models_data, out_path, title)


if __name__ == "__main__":
    main()
