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
        f"  <title>{e(title)}</title>",
        "  <style>",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 25px; background: #f8f9fa; color: #212529; }",
        "    .container { max-width: 1400px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }",
        "    h1 { color: #1a2530; margin-top: 0; font-size: 24px; border-bottom: 2px solid #e9ecef; padding-bottom: 12px; }",
        "    h2 { color: #2c3e50; font-size: 19px; margin-top: 25px; border-bottom: 1px solid #dee2e6; padding-bottom: 8px; }",
        "    h3 { color: #34495e; font-size: 16px; margin-top: 20px; }",
        "    h4 { color: #495057; font-size: 14px; margin-top: 15px; }",
        "    .meta { font-size: 14px; color: #6c757d; margin-bottom: 20px; }",
        "    .badge { display: inline-block; padding: 4px 8px; font-size: 12px; font-weight: 600; border-radius: 4px; }",
        "    .badge-primary { background: #e3f2fd; color: #0d47a1; margin-left: 8px; }",
        "    .badge-blind { background: #fff3e0; color: #e65100; font-weight: bold; }",
        "    .badge-unblind { background: #e8f5e9; color: #1b5e20; font-weight: bold; }",
        "    .badge-impacts { display: inline-block; padding: 3px 8px; font-size: 12px; font-weight: 600; border-radius: 4px; background: #e8eaf6; color: #283593; text-decoration: none; border: 1px solid #c5cae9; }",
        "    .badge-impacts:hover { background: #3f51b5; color: #fff; text-decoration: none; }",
        "    .muted { color: #adb5bd; }",
        "    table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13.5px; }",
        "    th { background: #f1f3f5; color: #495057; text-align: right; padding: 9px 12px; border: 1px solid #dee2e6; }",
        "    td { padding: 9px 12px; border: 1px solid #dee2e6; text-align: right; }",
        "    th.l, td.l { text-align: left; }",
        "    .rebin-row { cursor: pointer; transition: background-color 0.15s ease; }",
        "    .rebin-row:hover { background-color: #f1f8ff !important; }",
        "    .active-row { background-color: #e7f5ff !important; font-weight: 500; }",
        "    .active-row td:first-child { border-left: 4px solid #1971c2; }",
        "    .mono { font-family: SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12.5px; }",
        "    .band { color: #6c757d; font-size: 12px; margin-left: 4px; }",
        "    .tabs-container { display: flex; gap: 8px; margin: 20px 0 10px 0; border-bottom: 2px solid #dee2e6; padding-bottom: 8px; }",
        "    .rebin-tab-btn { background: #e9ecef; border: 1px solid #ced4da; border-radius: 6px; padding: 7px 16px; font-size: 14px; font-weight: 600; cursor: pointer; color: #495057; transition: all 0.2s ease; }",
        "    .rebin-tab-btn:hover { background: #dee2e6; color: #212529; }",
        "    .active-tab { background: #1971c2 !important; color: #fff !important; border-color: #1864ab !important; }",
        "    .rebin-detail-pane { background: #ffffff; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin-top: 15px; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }",
        "    .figs { display: flex; flex-wrap: wrap; gap: 14px; margin: 12px 0; }",
        "    .figs figure { margin: 0; width: 320px; }",
        "    .figs img { width: 100%; border: 1px solid #ced4da; border-radius: 4px; background: #fff; }",
        "    .figs figcaption { font-size: 12px; color: #6c757d; text-align: center; margin-top: 4px; }",
        "    .links { margin: 10px 0; display: flex; flex-wrap: wrap; gap: 10px; font-size: 13px; }",
        "    .links a { background: #f8f9fa; border: 1px solid #dee2e6; padding: 4px 10px; border-radius: 4px; color: #1971c2; text-decoration: none; font-weight: 500; }",
        "    .links a:hover { background: #e7f5ff; border-color: #74c0fc; }",
        "    .btn-detail { background: #1971c2; color: #fff; border: none; border-radius: 4px; padding: 4px 10px; font-size: 12px; font-weight: 600; cursor: pointer; }",
        "    .btn-detail:hover { background: #1864ab; }",
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
        f"  <h1>{e(title)} <span class='badge badge-primary'>{e(label)}</span></h1>",
        f"  <div class='meta'>Combine Statistical Inference across evaluated Rebins. Mode: <span class='badge {'badge-blind' if is_blind else 'badge-unblind'}'>{'Blinded' if is_blind else 'Unblinded'}</span> &middot; Click any table row or rebin tab to reveal details below.</div>",
        "  <h2>Comparative Rebin Summary</h2>",
        "  <table>",
        "    <thead>",
        "      <tr>",
        "        <th class='l'>Rebin</th>",
        "        <th class='l'>Channel</th>",
        "        <th>Exp. 95% CL Limit</th>",
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
        "  <div id='details-section' style='margin-top: 35px;'></div>",
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
