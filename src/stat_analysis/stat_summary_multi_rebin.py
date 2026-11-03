#!/usr/bin/env python3
"""
Multi-rebin HTML/Markdown summary for Combine statistical results.
Scans all stat_analysis_rebin* (or stat_analysis_unblinded_mixeddata_rebin*)
subdirectories under a master directory and builds an interactive, single-page
summary with a master comparative table, dynamic per-rebin detail panels,
and direct impacts links for non-stat-only models.
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
from collections import OrderedDict


# Ensure barista root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.stat_analysis.stat_summary import (
    collect_channel,
    discover_channels,
    flim,
    fnum,
    fscan,
    fsig,
    short_proc,
)

logger = logging.getLogger("stat_summary_multi_rebin")


def discover_rebin_dirs(master_dir: str) -> list[tuple[int, str]]:
    """
    Finds all rebin directories under master_dir and extracts their integer rebin values.
    Returns sorted list of (rebin_val, dir_path).
    """
    rebin_dirs = []
    pattern = os.path.join(master_dir, "*rebin*")
    for d in sorted(glob.glob(pattern)):
        if not os.path.isdir(d):
            continue
        bname = os.path.basename(d)
        m = re.search(r"rebin(\d+)", bname)
        if m:
            rebin_val = int(m.group(1))
            rebin_dirs.append((rebin_val, d))
    rebin_dirs.sort(key=lambda x: x[0])
    return rebin_dirs


def build_multi_rebin_summary(master_dir: str, title: str, label: str, is_blind: bool = True) -> tuple[str, str]:
    rebin_dirs = discover_rebin_dirs(master_dir)
    if not rebin_dirs:
        logger.warning(f"No rebin directories found in {master_dir}")
        return "<p>No rebin directories found.</p>", "No rebin directories found."

    e = html.escape
    rebin_data = []

    for rebin_val, rdir in rebin_dirs:
        channels = discover_channels(rdir)
        ch_list = []
        for ch, sig in channels.items():
            info = collect_channel(rdir, ch, sig, variable=None)
            
            # Check impacts pdf relative to master_dir
            impacts_pdf = os.path.join(rdir, ch, "impacts", f"datacard_impacts__{sig}.pdf")
            is_stat_only = "_stat_only" in ch or ch.endswith("_stat")
            impacts_pngs = []
            if os.path.exists(impacts_pdf):
                rel_impacts = os.path.relpath(impacts_pdf, master_dir)
                if not is_stat_only:
                    pngs = sorted(glob.glob(os.path.join(rdir, ch, "impacts", "impacts_page-*.png")))
                    if not pngs and shutil.which("pdftoppm"):
                        try:
                            subprocess.run(
                                ["pdftoppm", "-png", "-r", "150", impacts_pdf, os.path.join(rdir, ch, "impacts", "impacts_page")],
                                check=False, capture_output=True
                            )
                            pngs = sorted(glob.glob(os.path.join(rdir, ch, "impacts", "impacts_page-*.png")))
                        except Exception as e:
                            logger.warning(f"Could not convert impacts PDF to PNG: {e}")
                    impacts_pngs = [os.path.relpath(p, master_dir) for p in pngs]
            else:
                rel_impacts = None

            ch_list.append({
                "info": info,
                "is_stat_only": is_stat_only,
                "rel_impacts": rel_impacts,
                "impacts_pngs": impacts_pngs,
                "rdir": rdir,
            })

        rebin_data.append((rebin_val, rdir, ch_list))

    # Determine default active rebin
    default_rebin = rebin_data[0][0] if rebin_data else None

    # Render HTML
    h = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8'>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"  <title>{e(title)}</title>",
        "  <style>",
        "    :root {",
        "      --bg: #f8fafc;",
        "      --fg: #0f172a;",
        "      --muted: #64748b;",
        "      --card: #ffffff;",
        "      --line: #e2e8f0;",
        "      --accent: #2563eb;",
        "      --accent-hover: #1d4ed8;",
        "      --accent-soft: #eff6ff;",
        "      --pass-bg: #dcfce7;",
        "      --pass-fg: #15803d;",
        "      --pass-border: #bbf7d0;",
        "      --warn-bg: #fef3c7;",
        "      --warn-fg: #b45309;",
        "      --warn-border: #fde68a;",
        "      --danger-bg: #fee2e2;",
        "      --danger-fg: #b91c1c;",
        "      --danger-border: #fecaca;",
        "      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;",
        "      --sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;",
        "    }",
        "    * { box-sizing: border-box; margin: 0; padding: 0; }",
        "    body { font-family: var(--sans); background: var(--bg); color: var(--fg); font-size: 14px; line-height: 1.5; padding: 24px 32px 64px; }",
        "    .container { max-width: 1400px; margin: 0 auto; display: flex; flex-direction: column; gap: 18px; }",
        "    header { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 18px 24px; }",
        "    h1 { color: var(--fg); font-size: 1.35rem; font-weight: 700; margin-bottom: 6px; display: flex; align-items: center; gap: 10px; }",
        "    h2 { color: var(--fg); font-size: 1.1rem; font-weight: 600; margin: 18px 0 10px 0; border-bottom: 1px solid var(--line); padding-bottom: 6px; }",
        "    h3 { color: var(--fg); font-size: 0.95rem; font-weight: 600; margin: 16px 0 8px; }",
        "    h4 { color: var(--fg); font-size: 13.5px; font-weight: 600; margin: 12px 0 6px; }",
        "    .meta { font-size: 13px; color: var(--muted); display: flex; flex-wrap: wrap; align-items: center; gap: 6px 12px; }",
        "    .card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 18px; }",
        "    .badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; font-size: 11.5px; font-weight: 600; border-radius: 4px; white-space: nowrap; }",
        "    .badge-primary { background: var(--accent-soft); color: var(--accent); border: 1px solid #bfdbfe; font-family: var(--mono); }",
        "    .badge-blind { background: var(--warn-bg); color: var(--warn-fg); border: 1px solid var(--warn-border); }",
        "    .badge-unblind { background: var(--pass-bg); color: var(--pass-fg); border: 1px solid var(--pass-border); }",
        "    .badge-impacts { display: inline-flex; align-items: center; gap: 4px; padding: 3px 8px; font-size: 12px; font-weight: 500; border-radius: 4px; background: var(--accent-soft); color: var(--accent); text-decoration: none; border: 1px solid #bfdbfe; transition: all 0.15s ease; }",
        "    .badge-impacts:hover { background: var(--accent); color: #fff; }",
        "    .muted { color: var(--muted); }",
        "    .wrap-table { overflow-x: auto; background: var(--card); border: 1px solid var(--line); border-radius: 8px; margin: 8px 0; }",
        "    table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }",
        "    th { background: #f8fafc; color: var(--muted); text-align: right; padding: 8px 12px; border-bottom: 1px solid var(--line); font-size: 11.5px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.03em; }",
        "    td { padding: 8px 12px; border-bottom: 1px solid var(--line); text-align: right; }",
        "    th.l, td.l { text-align: left; }",
        "    .rebin-row { cursor: pointer; transition: background-color 0.15s ease; }",
        "    .rebin-row:hover { background-color: var(--accent-soft) !important; }",
        "    .active-row { background-color: var(--accent-soft) !important; font-weight: 500; }",
        "    .active-row td:first-child { border-left: 3px solid var(--accent); }",
        "    .mono { font-family: var(--mono); font-size: 12.5px; }",
        "    .band { color: var(--muted); font-size: 12px; margin-left: 4px; }",
        "    .tabs-container { display: flex; gap: 6px; margin: 16px 0 10px 0; border-bottom: 1px solid var(--line); padding-bottom: 8px; }",
        "    .rebin-tab-btn { background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 6px 14px; font-size: 13px; font-weight: 600; cursor: pointer; color: var(--fg); transition: all 0.15s ease; }",
        "    .rebin-tab-btn:hover { background: var(--accent-soft); color: var(--accent); border-color: #bfdbfe; }",
        "    .active-tab { background: var(--accent) !important; color: #fff !important; border-color: var(--accent) !important; }",
        "    .rebin-detail-pane { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 20px; margin-top: 15px; }",
        "    .figs { display: flex; flex-wrap: wrap; gap: 14px; margin: 12px 0; }",
        "    .figs figure { margin: 0; width: 340px; border: 1px solid var(--line); border-radius: 8px; background: var(--card); overflow: hidden; padding: 6px; }",
        "    .figs img { width: 100%; border-radius: 6px; display: block; }",
        "    .figs figcaption { font-size: 12px; color: var(--muted); text-align: center; padding: 6px 4px 2px; }",
        "    .links { margin: 10px 0; display: flex; flex-wrap: wrap; gap: 8px; }",
        "    .links a { display: inline-flex; align-items: center; gap: 4px; padding: 4px 10px; border-radius: 6px; font-size: 12.5px; font-weight: 500; text-decoration: none; color: var(--accent); background: var(--card); border: 1px solid var(--line); transition: all 0.15s ease; }",
        "    .links a:hover { background: var(--accent-soft); border-color: #bfdbfe; }",
        "    .btn-detail { background: var(--accent); color: #fff; border: none; border-radius: 4px; padding: 4px 10px; font-size: 12px; font-weight: 600; cursor: pointer; }",
        "    .btn-detail:hover { background: var(--accent-hover); }",
        "  </style>",
        "  <script>",
        "    function showRebinDetails(rebinVal) {",
        "      document.querySelectorAll('.rebin-detail-pane').forEach(function(el) { el.style.display = 'none'; });",
        "      document.querySelectorAll('.rebin-tab-btn').forEach(function(el) { el.classList.remove('active-tab'); });",
        "      document.querySelectorAll('.rebin-row').forEach(function(el) { el.classList.remove('active-row'); });",
        "      var pane = document.getElementById('details-rebin-' + rebinVal);",
        "      if (pane) { pane.style.display = 'block'; }",
        "      var btn = document.getElementById('tab-btn-' + rebinVal);",
        "      if (btn) { btn.classList.add('active-tab'); }",
        "      var rows = document.querySelectorAll('.rebin-row-' + rebinVal);",
        "      rows.forEach(function(r) { r.classList.add('active-row'); });",
        "      var container = document.getElementById('details-section');",
        "      if (container) { container.scrollIntoView({ behavior: 'smooth', block: 'start' }); }",
        "    }",
        "  </script>",
        "</head>",
        "<body>",
        "<div class='container'>",
        "  <header>",
        f"    <h1>{e(title)} <span class='badge badge-primary'>{e(label)}</span></h1>",
        f"    <div class='meta'>Combine Statistical Inference across evaluated Rebins. Mode: <span class='badge {'badge-blind' if is_blind else 'badge-unblind'}'>{'Blinded' if is_blind else 'Unblinded'}</span> &middot; Click any table row or rebin tab to reveal details below.</div>",
        "  </header>",
        "  <div class='card'>",
        "    <h2>Comparative Rebin Summary</h2>",
        "    <div class='wrap-table'>",
        "    <table>",
        "      <thead>",
        "        <tr>",
        "          <th class='l'>Rebin</th>",
        "          <th class='l'>Channel</th>",
        "          <th>Exp. 95% CL Limit</th>",
    ]

    if not is_blind:
        h.append("        <th>Obs. Limit</th>")
    h.append("        <th>Exp. Significance</th>")
    if not is_blind:
        h.append("        <th>Obs. Significance</th>")
    h.extend([
        "        <th>Likelihood Scan (r)</th>",
        "        <th>Impacts</th>",
        "        <th style='text-align: center;'>Action</th>",
        "      </tr>",
        "    </thead>",
        "    <tbody>",
    ])

    rows_md = []
    for rebin_val, rdir, ch_list in rebin_data:
        is_first_ch = True
        for ch_dict in ch_list:
            info = ch_dict["info"]
            is_stat_only = ch_dict["is_stat_only"]
            rel_impacts = ch_dict["rel_impacts"]

            ch_name = info["channel"]
            sig_name = info["signal"]
            lim = info.get("limits")
            sig_info = info.get("significance")
            scan = info.get("scan")

            med, band1, band2 = flim(lim)
            obs_lim = fnum(lim.get("obs"), 2) if (lim and lim.get("obs") is not None and not is_blind) else "-"
            exp_sig = f"{fnum(sig_info.get('expected'), 2)} σ" if (sig_info and sig_info.get("expected") is not None) else "-"
            obs_sig = f"{fnum(sig_info.get('observed'), 2)} σ" if (sig_info and sig_info.get("observed") is not None and not is_blind) else "-"
            scan_str = fscan(scan)

            if not is_stat_only and rel_impacts:
                impacts_cell = f"<a href='{e(rel_impacts)}' target='_blank' class='badge-impacts' onclick='event.stopPropagation();'>Impacts PDF &nearr;</a>"
                impacts_md = f"[Impacts PDF]({rel_impacts})"
            else:
                impacts_cell = "<span class='muted'>—</span>"
                impacts_md = "—"

            active_cls = " active-row" if rebin_val == default_rebin else ""
            h.append(f"      <tr class='rebin-row rebin-row-{rebin_val}{active_cls}' onclick='showRebinDetails({rebin_val});'>")
            h.append(f"        <td class='l mono'><strong>rebin {rebin_val}</strong></td>")
            h.append(f"        <td class='l mono'><code>{e(ch_name)}</code></td>")
            h.append(f"        <td class='mono'><b>{med}</b> <span class='band'>{band1}</span></td>")
            if not is_blind:
                h.append(f"        <td class='mono'><b>{obs_lim}</b></td>")
            h.append(f"        <td class='mono'>{exp_sig}</td>")
            if not is_blind:
                h.append(f"        <td class='mono'><b>{obs_sig}</b></td>")
            h.append(f"        <td class='mono'><b>{e(scan_str)}</b></td>")
            h.append(f"        <td style='text-align: center;'>{impacts_cell}</td>")
            if is_first_ch:
                h.append(f"        <td style='text-align: center;'><button class='btn-detail' onclick='event.stopPropagation(); showRebinDetails({rebin_val});'>View Details &darr;</button></td>")
                is_first_ch = False
            else:
                h.append(f"        <td style='text-align: center;'><button class='btn-detail' onclick='event.stopPropagation(); showRebinDetails({rebin_val});'>View Details &darr;</button></td>")
            h.append("      </tr>")

            rows_md.append({
                "rebin": rebin_val,
                "channel": ch_name,
                "exp_limit": f"**{med}** {band1}",
                "obs_limit": f"**{obs_lim}**" if not is_blind else "-",
                "exp_sig": exp_sig,
                "obs_sig": f"**{obs_sig}**" if not is_blind else "-",
                "scan": f"`{scan_str}`",
                "impacts": impacts_md,
            })

    h.extend([
        "    </tbody>",
        "  </table>",
        "  </div>",
        "  </div>",
        "  <div id='details-section' style='margin-top: 18px;'></div>",
        "  <div class='card'>",
        "  <h2>Interactive Rebin Details</h2>",
        "  <div class='tabs-container'>",
    ])

    for rebin_val, _, _ in rebin_data:
        active_tab_cls = " active-tab" if rebin_val == default_rebin else ""
        h.append(f"    <button id='tab-btn-{rebin_val}' class='rebin-tab-btn{active_tab_cls}' onclick='showRebinDetails({rebin_val});'>Rebin {rebin_val}</button>")

    h.append("  </div>")

    # Render Detail Panes for each rebin
    for rebin_val, rdir, ch_list in rebin_data:
        display_style = "block" if rebin_val == default_rebin else "none"
        h.append(f"  <div id='details-rebin-{rebin_val}' class='rebin-detail-pane' style='display: {display_style};'>")
        h.append(f"    <h3 style='margin-top: 0; color: #1971c2;'>Full Breakdown: Rebin {rebin_val}</h3>")

        for ch_dict in ch_list:
            c = ch_dict["info"]
            is_stat_only = ch_dict["is_stat_only"]
            rel_impacts = ch_dict["rel_impacts"]
            ch_name = c["channel"]
            sig_name = c["signal"]

            h.append(f"    <h4>Channel: <code>{e(ch_name)}</code> &mdash; Signal: <code>{e(sig_name)}</code></h4>")

            # Impacts banner if available
            if not is_stat_only and rel_impacts:
                h.append(f"    <div style='margin: 8px 0;'><a href='{e(rel_impacts)}' target='_blank' class='badge-impacts'>&nearr; Open Parameter Impacts PDF for Rebin {rebin_val} ({e(ch_name)})</a></div>")

            # Yields table
            if c.get("years"):
                h.append("    <h5>Datacard yields per year</h5>")
                h.append("    <table><tr><th class='l'>year</th><th>observation</th>"
                         + "".join(f"<th>{e(short_proc(p))}</th>" for p in c["processes"]) + "</tr>")
                for year, card in c["years"].items():
                    h.append(f"    <tr><td class='l'>{e(year)}</td><td>{fnum(card['observation'])}</td>"
                             + "".join(f"<td>{fnum(card['rates'].get(p), 2)}</td>" for p in c["processes"]) + "</tr>")
                h.append(f"    <tr><th class='l'>all</th><th>{fnum(c['observation'])}</th>"
                         + "".join(f"<th>{fnum(c['totals'].get(p), 2)}</th>" for p in c["processes"]) + "</tr></table>")

            # Figures
            figs_found = []
            scan_img = os.path.join(rdir, ch_name, "likelihood_scan", "scan_plot.png")
            if os.path.exists(scan_img):
                figs_found.append(("likelihood scan", os.path.relpath(scan_img, master_dir)))

            for p in sorted(glob.glob(os.path.join(rdir, ch_name, "postfit", "plots", f"postfitplots__{sig_name}__*.png"))):
                lbl = f"postfit {os.path.basename(p).split('__')[-1][:-4]}"
                figs_found.append((lbl, os.path.relpath(p, master_dir)))

            corr_img = os.path.join(rdir, ch_name, "postfit", "plots", "correlation_fit_s.png")
            if os.path.exists(corr_img):
                figs_found.append(("correlation (physics)", os.path.relpath(corr_img, master_dir)))

            # If not stat-only and impacts exist, add page 1 to Key Figures with other plots
            if not is_stat_only and ch_dict.get("impacts_pngs"):
                page1 = ch_dict["impacts_pngs"][0]
                figs_found.append(("impacts", page1))

            if figs_found:
                h.append("    <h5>Key Figures</h5><div class='figs'>")
                for caption, fig_rel in figs_found:
                    h.append(f"      <figure><a href='{e(fig_rel)}' target='_blank'><img src='{e(fig_rel)}' alt='{e(caption)}' loading='lazy'></a><figcaption>{e(caption)}</figcaption></figure>")
                h.append("    </div>")

            # Artifact Links


            h.append("    <h5>Artifact Links</h5><div class='links'>")
            for label, rel_target in c.get("links", {}).items():
                target_from_master = os.path.relpath(os.path.join(rdir, rel_target), master_dir)
                h.append(f"      <a href='{e(target_from_master)}' target='_blank'>{e(label)}</a>")
            if not is_stat_only and rel_impacts and "impacts pdf" not in c.get("links", {}):
                h.append(f"      <a href='{e(rel_impacts)}' target='_blank' style='font-weight: bold; background: #e8eaf6;'>impacts pdf</a>")
            h.append("    </div><hr style='border: 0; border-top: 1px dashed #dee2e6; margin: 20px 0;'>")

        h.append("  </div>")

    h.extend([
        "  </div>",
        "</div>",
        "</body>",
        "</html>",
    ])

    # Render Markdown
    md_lines = [
        f"# {title} — {label}",
        "",
        f"Mode: **{'Blinded' if is_blind else 'Unblinded'}**",
        "",
    ]
    if is_blind:
        md_lines.append("| Rebin | Channel | Exp. Limit (95% CL) | Exp. Significance | Likelihood Scan | Impacts |")
        md_lines.append("|---|---|---|---|---|---|")
        for r in rows_md:
            md_lines.append(f"| rebin {r['rebin']} | `{r['channel']}` | {r['exp_limit']} | {r['exp_sig']} | {r['scan']} | {r['impacts']} |")
    else:
        md_lines.append("| Rebin | Channel | Exp. Limit | Obs. Limit | Exp. Sig. | Obs. Sig. | Likelihood Scan | Impacts |")
        md_lines.append("|---|---|---|---|---|---|---|---|")
        for r in rows_md:
            md_lines.append(f"| rebin {r['rebin']} | `{r['channel']}` | {r['exp_limit']} | {r['obs_limit']} | {r['exp_sig']} | {r['obs_sig']} | {r['scan']} | {r['impacts']} |")

    return "\n".join(h), "\n".join(md_lines)


def main():
    parser = argparse.ArgumentParser(description="Generate interactive multi-rebin summary page.")
    parser.add_argument("-d", "--dir", required=True, help="Master stat analysis directory containing rebin subdirectories.")
    parser.add_argument("-o", "--output", required=True, help="Output HTML summary file path.")
    parser.add_argument("--md", default=None, help="Output Markdown summary file path.")
    parser.add_argument("--title", default="Combine Statistical Interpretation Across Rebins", help="Page title.")
    parser.add_argument("--label", default="", help="Label for analysis run.")
    parser.add_argument("--unblind", action="store_true", help="Display observed limits and significance.")
    args = parser.parse_args()

    html_content, md_content = build_multi_rebin_summary(
        args.dir,
        title=args.title,
        label=args.label,
        is_blind=not args.unblind
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html_content)
    print(f"Wrote interactive HTML summary: {args.output}")

    if args.md:
        os.makedirs(os.path.dirname(os.path.abspath(args.md)), exist_ok=True)
        with open(args.md, "w") as f:
            f.write(md_content)
        print(f"Wrote Markdown summary: {args.md}")


if __name__ == "__main__":
    main()
