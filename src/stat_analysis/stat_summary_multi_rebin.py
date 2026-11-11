#!/usr/bin/env python3
"""
Multi-rebin HTML/Markdown summary for Combine statistical results.
Scans all stat_analysis_rebin* (or stat_analysis_unblinded_mixeddata_rebin*)
subdirectories under a master directory and builds a unified summary matching
the rich layout, tables, systematic impacts, and figures of stat_summary.py.
"""
from __future__ import annotations

import argparse
import glob
import html
import logging
import math
import os
import re
import sys
from collections import OrderedDict

# Ensure barista root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.stat_analysis.stat_summary import (
    CSS,
    collect_channel,
    discover_channels,
    flim,
    fnum,
    fscan,
    fsig,
    md_page,
    render_channel_card,
    render_header,
    render_overview_card,
    render_roast_script,
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
        msg = f"<p>No rebin directories found under <code>{html.escape(master_dir)}</code>.</p>"
        return f"<!doctype html><html><body>{msg}</body></html>", "No rebin directories found."

    e = html.escape
    rebin_data = []

    for rebin_val, rdir in rebin_dirs:
        channels = discover_channels(rdir)
        ch_list = []
        prefix = os.path.relpath(os.path.abspath(rdir), os.path.abspath(master_dir))
        for ch, sig in channels.items():
            info = collect_channel(rdir, ch, sig, variable=None)
            # Prefix links and figures to be relative to master_dir
            if prefix not in (".", ""):
                info["links"] = OrderedDict((k, os.path.join(prefix, v)) for k, v in info["links"].items())
                info["figures"] = OrderedDict((k, os.path.join(prefix, v)) for k, v in info["figures"].items())
            ch_list.append(info)
        rebin_data.append((rebin_val, rdir, ch_list))

    # Case 1: Exactly 1 rebin directory exists -> Render single-rebin direct format
    if len(rebin_data) == 1:
        rebin_val, rdir, ch_list = rebin_data[0]
        h = [
            "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
            f"<title>{e(title)}</title><style>{CSS}</style></head><body>",
            "<div class='container'>",
            render_header(title, master_dir, is_blind, f"{label} (rebin {rebin_val})" if label else f"rebin {rebin_val}"),
            render_overview_card(ch_list, blind=is_blind, title=f"Overview (Rebin {rebin_val})"),
        ]
        for c in ch_list:
            h.append(render_channel_card(c, blind=is_blind))
        h.append(render_roast_script())
        h.append("</div></body></html>")
        
        md_text = md_page(f"{title} — {label} (rebin {rebin_val})" if label else f"{title} (rebin {rebin_val})", ch_list, is_blind)
        return "\n".join(h), md_text

    # Case 2: Multiple rebin directories exist -> Comparative table + Rebin detail tabs
    default_rebin = rebin_data[0][0]

    h = [
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{e(title)}</title><style>{CSS}</style></head><body>",
        "<div class='container'>",
        render_header(title, master_dir, is_blind, label),
        "<div class='card'><h2>Comparative Rebin Summary</h2>",
        "<div class='wrap-table'><table>",
        "<tr><th class='l'>rebin</th><th class='l'>channel</th><th class='l'>signal</th>"
        "<th>exp. significance</th>" + ("" if is_blind else "<th>obs. significance</th>") +
        "<th>exp. 95% CL limit (median)</th><th>&plusmn;1&sigma;</th><th>&plusmn;2&sigma;</th>" + ("" if is_blind else "<th>obs. limit</th>") +
        "<th>bkg yield</th><th>signal yield</th><th>S/&radic;B</th><th>likelihood scan (68% CL)</th><th>nuisances</th><th>top impact (&Delta;r)</th></tr>",
    ]

    md_lines = [
        f"# {title} — {label}",
        "",
        "blinded (Asimov data_obs)" if is_blind else "unblinded",
        "",
        "| rebin | channel | signal | exp. significance | exp. limit median | ±1σ | ±2σ | bkg yield | signal yield | scan | top impact |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for rebin_val, rdir, ch_list in rebin_data:
        active_cls = " active-row" if rebin_val == default_rebin else ""
        for c in ch_list:
            med, one, two = flim(c.get("limits"))
            sig = c.get("significance") or {}
            totals = c.get("totals", {})
            signal_name = c.get("signal", "")
            bkg = sum(v for p, v in totals.items() if p != signal_name)
            s = totals.get(signal_name, float("nan"))
            soverb = s / math.sqrt(bkg) if bkg and not math.isnan(s) else float("nan")
            sc = c.get("scan")
            
            top_imp_cell = "<span class='muted'>—</span>"
            top_imp_md = "—"
            if c.get("impacts"):
                top_imp = c["impacts"][0]
                top_imp_cell = f"<code>{e(top_imp['name'])}</code> (<b>{top_imp['impact']:+.4f}</b>)"
                top_imp_md = f"`{top_imp['name']}` ({top_imp['impact']:+.4f})"

            h.append(
                f"<tr class='rebin-row rebin-row-{rebin_val}{active_cls}' onclick='showRebinDetails({rebin_val});'>"
                f"<td class='l'><b>rebin {rebin_val}</b></td>"
                f"<td class='l'>{e(c.get('channel', ''))}</td>"
                f"<td class='l'><code>{e(signal_name)}</code></td>"
                f"<td>{fsig(sig.get('expected'))}</td>" + ("" if is_blind else f"<td>{fsig(sig.get('observed'))}</td>") +
                f"<td><b>{med}</b></td><td>{one}</td><td>{two}</td>" + ("" if is_blind else f"<td>{fnum((c.get('limits') or {}).get('obs'), 2)}</td>") +
                f"<td>{fnum(bkg)}</td><td>{fnum(s, 2)}</td><td>{fnum(soverb, 3)}</td>"
                f"<td><b>{e(fscan(sc))}</b></td><td>{c.get('nuisances', 0)}</td>"
                f"<td>{top_imp_cell}</td></tr>"
            )

            md_lines.append(
                f"| rebin {rebin_val} | {c.get('channel', '')} | `{signal_name}` | {fsig(sig.get('expected'))} | "
                f"{med} | {one} | {two} | {fnum(bkg)} | {fnum(s, 2)} | {fscan(sc)} | {top_imp_md} |"
            )

    h.append("</table></div>")
    h.append("<p class='small'>Click any row above or rebin tab below to view detailed datacards, limits, systematic impacts, and figures.</p>")
    h.append("</div>")  # close comparative card

    # Interactive Rebin Details Section
    h.append("<div id='details-section' class='card'><h2>Rebin Details</h2>")
    h.append("<div class='tabs-container'>")
    for rebin_val, _, _ in rebin_data:
        active_tab_cls = " active-tab" if rebin_val == default_rebin else ""
        h.append(f"<button id='tab-btn-{rebin_val}' class='rebin-tab-btn{active_tab_cls}' onclick='showRebinDetails({rebin_val});'>Rebin {rebin_val}</button>")
    h.append("</div>")

    # Detail panes
    for rebin_val, rdir, ch_list in rebin_data:
        display_style = "flex" if rebin_val == default_rebin else "none"
        h.append(f"<div id='details-rebin-{rebin_val}' class='rebin-detail-pane' style='display: {display_style};'>")
        h.append(render_overview_card(ch_list, blind=is_blind, title=f"Overview — Rebin {rebin_val}"))
        for c in ch_list:
            h.append(render_channel_card(c, blind=is_blind, header_prefix=f"[Rebin {rebin_val}] "))
        h.append("</div>")

    h.append("</div>")  # close details-section card

    # JavaScript for Rebin Details and Roast Index back button
    h.append("""
<script>
function showRebinDetails(rebinVal) {
  document.querySelectorAll('.rebin-detail-pane').forEach(function(el) { el.style.display = 'none'; });
  document.querySelectorAll('.rebin-tab-btn').forEach(function(el) { el.classList.remove('active-tab'); });
  document.querySelectorAll('.rebin-row').forEach(function(el) { el.classList.remove('active-row'); });
  var pane = document.getElementById('details-rebin-' + rebinVal);
  if (pane) { pane.style.display = 'flex'; }
  var btn = document.getElementById('tab-btn-' + rebinVal);
  if (btn) { btn.classList.add('active-tab'); }
  var rows = document.querySelectorAll('.rebin-row-' + rebinVal);
  rows.forEach(function(r) { r.classList.add('active-row'); });
  var container = document.getElementById('details-section');
  if (container) { container.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
}
</script>
""")
    h.append(render_roast_script())
    h.append("</div></body></html>")

    # Markdown Top Systematic Impacts Breakdown
    md_lines.append("\n## Top systematic impacts\n")
    for rebin_val, _, ch_list in rebin_data:
        for c in ch_list:
            if c.get("impacts"):
                md_lines.append(f"### Rebin {rebin_val} — {c.get('channel', '')} (`{c.get('signal', '')}`)\n")
                md_lines.append("| nuisance parameter | impact on r | type |")
                md_lines.append("|---|---|---|")
                for imp in c["impacts"][:10]:
                    md_lines.append(f"| `{imp.get('name', '')}` | **{imp.get('impact', 0.0):+.4f}** | {imp.get('type', '')} |")
                md_lines.append("")

    return "\n".join(h), "\n".join(md_lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate unified multi-rebin summary page matching stat_summary format.")
    parser.add_argument("-d", "--dir", required=True, help="Master stat analysis directory containing rebin subdirectories.")
    parser.add_argument("-o", "--output", required=True, help="Output HTML summary file path.")
    parser.add_argument("--md", default=None, help="Output Markdown summary file path.")
    parser.add_argument("--title", default="Statistical analysis summary", help="Page title.")
    parser.add_argument("--label", default="", help="Label for analysis run.")
    parser.add_argument("--unblind", action="store_true", help="Display observed limits and significance.")
    args = parser.parse_args()

    html_content, md_content = build_multi_rebin_summary(
        args.dir,
        title=args.title,
        label=args.label,
        is_blind=not args.unblind
    )

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html_content)
    print(f"Wrote unified HTML summary: {args.output}")

    if args.md:
        md_dir = os.path.dirname(os.path.abspath(args.md)) or "."
        os.makedirs(md_dir, exist_ok=True)
        with open(args.md, "w") as f:
            f.write(md_content)
        print(f"Wrote Markdown summary: {args.md}")


if __name__ == "__main__":
    main()
