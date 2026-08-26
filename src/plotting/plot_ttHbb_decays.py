#!/usr/bin/env python3
"""
Diagnostic and visualization tool for ttH(bb) ttbar decay mode analysis.
Reads .coffea output produced by ttHbbProcessor and generates:
- Formatted summary tables (raw and weighted yields, percentages, retention efficiencies)
- 3-panel publication-quality pie charts (Inclusive, Selected SR or SB, Selected SR only)
- Detailed channel comparison bar charts
- Markdown summary reports
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import numpy as np

# Ensure repository root is in sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from coffea.util import load


STAGE_LABELS = {
    "inclusive": "Inclusive (All Processed)",
    "selected_SR_or_SB": "Selected (SR or SB)",
    "selected_SR_only": "Selected (SR Only)",
    "selected_SR_or_SB_gt6": "Selected (SR or SB, $n_{\\mathrm{jets}} > 6$)",
    "selected_SR_only_gt6": "Selected (SR Only, $n_{\\mathrm{jets}} > 6$)",
}

GROUPED_COLORS = {
    "Hadronic": "#3b82f6",            # Blue
    "Semileptonic (e/mu)": "#10b981", # Green
    "Dileptonic (e/mu)": "#ef4444",   # Red
    "Tau decays": "#8b5cf6",          # Purple
    "Other": "#f59e0b",               # Amber
}

DETAILED_COLORS = {
    "Hadronic": "#2563eb",
    "Semileptonic (e)": "#059669",
    "Semileptonic (mu)": "#10b981",
    "Dileptonic (ee)": "#dc2626",
    "Dileptonic (mumu)": "#ef4444",
    "Dileptonic (emu)": "#f87171",
    "Semileptonic (tau)": "#7c3aed",
    "Dileptonic (tau+tau)": "#8b5cf6",
    "Dileptonic (tau+lep)": "#a78bfa",
    "Other": "#d97706",
}


def find_decay_hist(data):
    """Recursively search for ttbar_decay_study histogram in loaded coffea data."""
    if isinstance(data, dict):
        if "ttbar_decay_study" in data:
            return data["ttbar_decay_study"]
        if "hists" in data and isinstance(data["hists"], dict) and "ttbar_decay_study" in data["hists"]:
            return data["hists"]["ttbar_decay_study"]
        for v in data.values():
            found = find_decay_hist(v)
            if found is not None:
                return found
    return None


def extract_stage_data(decay_hist, stage, weight_type="weighted"):
    """Extract category dictionary for a given stage and weight type."""
    stage_data = {}
    categories = decay_hist.axes["category"]
    for cat in categories:
        try:
            val = float(np.sum(decay_hist[stage, cat, :, weight_type].values()))
        except Exception:
            val = 0.0
        stage_data[cat] = val
    return stage_data


def extract_detail_data(decay_hist, stage, weight_type="weighted"):
    """Extract detailed channel dictionary for a given stage and weight type."""
    detail_data = {}
    details = decay_hist.axes["detail"]
    for det in details:
        try:
            val = float(np.sum(decay_hist[stage, :, det, weight_type].values()))
        except Exception:
            val = 0.0
        detail_data[det] = val
    return detail_data


def print_summary_table(decay_hist, out_file=None):
    """Print and optionally save formatted summary tables."""
    lines = []
    lines.append("=" * 95)
    lines.append(f"{'ttH(bb) ttbar Decay Mode Composition Analysis':^95}")
    lines.append("=" * 95)

    stages = ["inclusive", "selected_SR_or_SB", "selected_SR_only"]
    categories = [c for c in decay_hist.axes["category"] if c != "Other"] + ["Other"]

    # Table 1: Grouped Categories
    lines.append("\n[1] GROUPED DECAY CATEGORIES")
    lines.append("-" * 95)
    lines.append(f"{'Stage':<24} | {'Category':<22} | {'Raw Events':>12} | {'Raw %':>8} | {'Weighted':>12} | {'Wtd %':>8}")
    lines.append("-" * 95)

    inc_raw_tot = sum(extract_stage_data(decay_hist, "inclusive", "raw").values())
    inc_wtd_tot = sum(extract_stage_data(decay_hist, "inclusive", "weighted").values())

    table_data = {}

    for stage in stages:
        raw_data = extract_stage_data(decay_hist, stage, "raw")
        wtd_data = extract_stage_data(decay_hist, stage, "weighted")
        tot_raw = sum(raw_data.values())
        tot_wtd = sum(wtd_data.values())

        table_data[stage] = {
            "raw": raw_data,
            "weighted": wtd_data,
            "tot_raw": tot_raw,
            "tot_wtd": tot_wtd,
        }

        stage_label = STAGE_LABELS.get(stage, stage)
        first = True
        for cat in categories:
            r = raw_data.get(cat, 0)
            w = wtd_data.get(cat, 0)
            r_pct = (r / tot_raw * 100) if tot_raw > 0 else 0.0
            w_pct = (w / tot_wtd * 100) if tot_wtd > 0 else 0.0

            prefix = stage_label if first else ""
            first = False
            lines.append(f"{prefix:<24} | {cat:<22} | {int(r):>12,d} | {r_pct:>7.2f}% | {w:>12.2f} | {w_pct:>7.2f}%")
        
        lines.append(f"{'':<24} | {'TOTAL':<22} | {int(tot_raw):>12,d} | {100.0:>7.2f}% | {tot_wtd:>12.2f} | {100.0:>7.2f}%")
        lines.append("-" * 95)

    # Table 2: Retention Efficiencies
    lines.append("\n[2] SELECTION RETENTION EFFICIENCY (Relative to Inclusive)")
    lines.append("-" * 95)
    lines.append(f"{'Category':<22} | {'SR or SB Eff (Raw)':>18} | {'SR or SB Eff (Wtd)':>18} | {'SR Only Eff (Raw)':>18} | {'SR Only Eff (Wtd)':>18}")
    lines.append("-" * 95)

    inc_raw = table_data["inclusive"]["raw"]
    inc_wtd = table_data["inclusive"]["weighted"]
    srosb_raw = table_data["selected_SR_or_SB"]["raw"]
    srosb_wtd = table_data["selected_SR_or_SB"]["weighted"]
    sronly_raw = table_data["selected_SR_only"]["raw"]
    sronly_wtd = table_data["selected_SR_only"]["weighted"]

    for cat in categories:
        e_srosb_r = (srosb_raw.get(cat, 0) / inc_raw.get(cat, 1) * 100) if inc_raw.get(cat, 0) > 0 else 0.0
        e_srosb_w = (srosb_wtd.get(cat, 0) / inc_wtd.get(cat, 1) * 100) if inc_wtd.get(cat, 0) > 0 else 0.0
        e_sronly_r = (sronly_raw.get(cat, 0) / inc_raw.get(cat, 1) * 100) if inc_raw.get(cat, 0) > 0 else 0.0
        e_sronly_w = (sronly_wtd.get(cat, 0) / inc_wtd.get(cat, 1) * 100) if inc_wtd.get(cat, 0) > 0 else 0.0

        lines.append(f"{cat:<22} | {e_srosb_r:>17.3f}% | {e_srosb_w:>17.3f}% | {e_sronly_r:>17.3f}% | {e_sronly_w:>17.3f}%")

    tot_e_srosb_r = (table_data["selected_SR_or_SB"]["tot_raw"] / inc_raw_tot * 100) if inc_raw_tot > 0 else 0.0
    tot_e_srosb_w = (table_data["selected_SR_or_SB"]["tot_wtd"] / inc_wtd_tot * 100) if inc_wtd_tot > 0 else 0.0
    tot_e_sronly_r = (table_data["selected_SR_only"]["tot_raw"] / inc_raw_tot * 100) if inc_raw_tot > 0 else 0.0
    tot_e_sronly_w = (table_data["selected_SR_only"]["tot_wtd"] / inc_wtd_tot * 100) if inc_wtd_tot > 0 else 0.0
    lines.append("-" * 95)
    lines.append(f"{'OVERALL':<22} | {tot_e_srosb_r:>17.3f}% | {tot_e_srosb_w:>17.3f}% | {tot_e_sronly_r:>17.3f}% | {tot_e_sronly_w:>17.3f}%")
    lines.append("=" * 95)

    table_text = "\n".join(lines)
    print(table_text)

    if out_file:
        with open(out_file, "w") as f:
            f.write(table_text)
            f.write("\n")

    return table_data


def plot_pie_charts(decay_hist, output_dir, weight_type="weighted"):
    """Generate 3-panel pie chart comparison across the three stages."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = ["inclusive", "selected_SR_or_SB", "selected_SR_only"]
    categories = [c for c in decay_hist.axes["category"] if c != "Other"] + ["Other"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=300)
    fig.suptitle(r"$\mathbf{CMS}$ $\mathrm{Simulation\ Preliminary}$" + f"\n$t\\bar{{t}}H(b\\bar{{b}})$ $t\\bar{{t}}$ Decay Mode Fractions ({weight_type.capitalize()})", fontsize=16, y=1.02)

    inc_tot = sum(extract_stage_data(decay_hist, "inclusive", weight_type).values())

    for idx, stage in enumerate(stages):
        ax = axes[idx]
        data = extract_stage_data(decay_hist, stage, weight_type)
        tot = sum(data.values())

        # Filter out zero values for clean pie display
        active_cats = [c for c in categories if data.get(c, 0) > 0]
        vals = [data[c] for c in active_cats]
        colors = [GROUPED_COLORS.get(c, "#cccccc") for c in active_cats]

        stage_title = STAGE_LABELS.get(stage, stage)
        if stage == "inclusive":
            subtitle = f"Total: {tot:,.1f}" if weight_type == "weighted" else f"Total: {int(tot):,d}"
        else:
            eff = (tot / inc_tot * 100) if inc_tot > 0 else 0.0
            subtitle = f"Total: {tot:,.1f} (Eff: {eff:.2f}%)" if weight_type == "weighted" else f"Total: {int(tot):,d} (Eff: {eff:.2f}%)"

        if tot > 0:
            wedges, texts, autotexts = ax.pie(
                vals,
                labels=None,
                autopct=lambda pct: f"{pct:.1f}%" if pct >= 1.0 else "",
                pctdistance=0.75,
                colors=colors,
                startangle=140,
                wedgeprops=dict(width=0.85, edgecolor="white", linewidth=1.5),
            )
            for autotext in autotexts:
                autotext.set_fontsize(11)
                autotext.set_fontweight("bold")
                autotext.set_color("white")
        else:
            ax.text(0, 0, "No events", ha="center", va="center", fontsize=12)

        ax.set_title(f"{stage_label_short(stage)}\n{subtitle}", fontsize=13, pad=10, fontweight="semibold")

    # Common legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=GROUPED_COLORS.get(cat, "#cccccc"), label=cat)
        for cat in categories if any(extract_stage_data(decay_hist, s, weight_type).get(cat, 0) > 0 for s in stages)
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles), fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    png_path = os.path.join(output_dir, f"ttHbb_decay_fractions_pie_{weight_type}.png")
    pdf_path = os.path.join(output_dir, f"ttHbb_decay_fractions_pie_{weight_type}.pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved pie chart to: {png_path} and {pdf_path}")


def stage_label_short(stage):
    if stage == "inclusive":
        return "1. Inclusive (All Processed)"
    elif stage == "selected_SR_or_SB":
        return "2. Selected (SR or SB, Baseline)"
    elif stage == "selected_SR_only":
        return "3. Selected (SR Only, Baseline)"
    elif stage == "selected_SR_or_SB_gt6":
        return r"Selected (SR or SB, $n_{\mathrm{jets}} > 6$)"
    elif stage == "selected_SR_only_gt6":
        return r"Selected (SR Only, $n_{\mathrm{jets}} > 6$)"
    return stage


def plot_pie_charts_gt6(decay_hist, output_dir, weight_type="weighted"):
    """Generate 3-panel pie chart comparison for nJets > 6 (Inclusive, Selected SR or SB gt6, Selected SR Only gt6)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = ["inclusive", "selected_SR_or_SB_gt6", "selected_SR_only_gt6"]
    categories = [c for c in decay_hist.axes["category"] if c != "Other"] + ["Other"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=300)
    fig.suptitle(r"$\mathbf{CMS}$ $\mathrm{Simulation\ Preliminary}$" + f"\n$t\\bar{{t}}H(b\\bar{{b}})$ $t\\bar{{t}}$ Decay Mode Fractions with $n_{{\\mathrm{{jets}}}} > 6$ ({weight_type.capitalize()})", fontsize=16, y=1.02)

    inc_tot = sum(extract_stage_data(decay_hist, "inclusive", weight_type).values())

    for idx, stage in enumerate(stages):
        ax = axes[idx]
        data = extract_stage_data(decay_hist, stage, weight_type)
        tot = sum(data.values())

        active_cats = [c for c in categories if data.get(c, 0) > 0]
        vals = [data[c] for c in active_cats]
        colors = [GROUPED_COLORS.get(c, "#cccccc") for c in active_cats]

        if stage == "inclusive":
            subtitle = f"Total: {tot:,.1f}" if weight_type == "weighted" else f"Total: {int(tot):,d}"
        else:
            eff = (tot / inc_tot * 100) if inc_tot > 0 else 0.0
            subtitle = f"Total: {tot:,.1f} (Eff: {eff:.2f}%)" if weight_type == "weighted" else f"Total: {int(tot):,d} (Eff: {eff:.2f}%)"

        if tot > 0:
            wedges, texts, autotexts = ax.pie(
                vals,
                labels=None,
                autopct=lambda pct: f"{pct:.1f}%" if pct >= 1.0 else "",
                pctdistance=0.75,
                colors=colors,
                startangle=140,
                wedgeprops=dict(width=0.85, edgecolor="white", linewidth=1.5),
            )
            for autotext in autotexts:
                autotext.set_fontsize(11)
                autotext.set_fontweight("bold")
                autotext.set_color("white")
        else:
            ax.text(0, 0, "No events", ha="center", va="center", fontsize=12)

        ax.set_title(f"{stage_label_short(stage)}\n{subtitle}", fontsize=13, pad=10, fontweight="semibold")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=GROUPED_COLORS.get(cat, "#cccccc"), label=cat)
        for cat in categories if any(extract_stage_data(decay_hist, s, weight_type).get(cat, 0) > 0 for s in stages)
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles), fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    png_path = os.path.join(output_dir, f"ttHbb_decay_fractions_pie_njets_gt6_{weight_type}.png")
    pdf_path = os.path.join(output_dir, f"ttHbb_decay_fractions_pie_njets_gt6_{weight_type}.pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved pie chart to: {png_path} and {pdf_path}")


def plot_bar_comparison(decay_hist, output_dir, weight_type="weighted"):
    """Generate detailed subcategory bar chart comparing stages."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = ["inclusive", "selected_SR_or_SB", "selected_SR_only"]
    details = [d for d in decay_hist.axes["detail"] if d != "Other"] + ["Other"]

    # Filter details that have > 0 counts
    active_details = [d for d in details if any(extract_detail_data(decay_hist, s, weight_type).get(d, 0) > 0 for s in stages)]

    x = np.arange(len(active_details))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

    for i, stage in enumerate(stages):
        detail_data = extract_detail_data(decay_hist, stage, weight_type)
        tot = sum(detail_data.values())
        fractions = [(detail_data.get(d, 0) / tot * 100) if tot > 0 else 0.0 for d in active_details]
        
        offset = (i - 1) * width
        label = STAGE_LABELS.get(stage, stage)
        rects = ax.bar(x + offset, fractions, width, label=label, edgecolor="black", linewidth=0.8, alpha=0.85)

        for rect in rects:
            h = rect.get_height()
            if h >= 0.5:
                ax.annotate(f"{h:.1f}%",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, rotation=0)

    ax.set_ylabel(f"Fraction of Events (%) [{weight_type.capitalize()}]", fontsize=12)
    ax.set_title(r"$\mathbf{CMS}$ $\mathrm{Simulation\ Preliminary}$" + f"\n$t\\bar{{t}}H(b\\bar{{b}})$ Detailed $t\\bar{{t}}$ Channel Breakdown", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(active_details, rotation=25, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(50, ax.get_ylim()[1] * 1.15))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    png_path = os.path.join(output_dir, f"ttHbb_decay_detailed_bars_{weight_type}.png")
    pdf_path = os.path.join(output_dir, f"ttHbb_decay_detailed_bars_{weight_type}.pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bar chart to: {png_path} and {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze and plot ttbar decay mode fractions for ttH(bb).")
    parser.add_argument("-i", "--input", required=True, help="Path to input .coffea file")
    parser.add_argument("-o", "--output-dir", default="output/ttHbb/plots/decay_study", help="Directory to save output plots and reports")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading coffea file: {args.input}")
    output = load(args.input)

    decay_hist = find_decay_hist(output)
    if decay_hist is None:
        print(f"ERROR: 'ttbar_decay_study' histogram not found in {args.input}!")
        print("Available keys in root:", list(output.keys()) if isinstance(output, dict) else type(output))
        if isinstance(output, dict) and "hists" in output:
            print("Available keys in hists:", list(output["hists"].keys()))
        sys.exit(1)

    print("Found 'ttbar_decay_study' histogram successfully.")
    
    # 1. Summary tables
    summary_txt = os.path.join(args.output_dir, "ttHbb_decay_summary.txt")
    print_summary_table(decay_hist, out_file=summary_txt)

    # 2. Plots for weighted and raw
    has_gt6 = "selected_SR_only_gt6" in decay_hist.axes["stage"]
    for w_type in ["weighted", "raw"]:
        plot_pie_charts(decay_hist, args.output_dir, weight_type=w_type)
        if has_gt6:
            plot_pie_charts_gt6(decay_hist, args.output_dir, weight_type=w_type)
        plot_bar_comparison(decay_hist, args.output_dir, weight_type=w_type)

    print(f"\nAll plots and summary tables saved to: {args.output_dir}\n")


if __name__ == "__main__":
    main()

