#!/usr/bin/env python3
"""
Standalone Impact & Pull Plotter for CMS Combine.
Uses the publication-grade PyROOT styling from the CMS Di-Higgs combination tool,
completely self-contained within barista (zero dependency on law or external checkouts).

Features:
- Exact CMS HH combination styling (ROOT canvas, colors, fonts, ticks).
- Dynamically handles any POI (e.g. r, rttHbb, rggHH, kl, etc.).
- Multi-page PDF generation with configurable parameters per page.
- Direct per-page PNG export eliminating all blank A4 border spaces.
"""

import os
import sys
import json
import math
import argparse

# Ensure dhi_core and plotlib are on sys.path
plots_dir = os.path.dirname(os.path.abspath(__file__))
if plots_dir not in sys.path:
    sys.path.insert(0, plots_dir)

import dhi_core
import dhi_core.config as dhi_cfg
from dhi_core.util import DotDict, import_ROOT
import dhi_core.plots.pulls_impacts as pi


def ensure_poi_registered(poi_name):
    """Ensure the given POI has a display label in dhi_cfg.poi_data."""
    if poi_name not in dhi_cfg.poi_data:
        # Build clean ROOT LaTeX label (e.g. 'rttHbb' -> 'r_{t#bar{t}H}', 'r' -> 'r')
        if poi_name.startswith("r"):
            suffix = poi_name[1:]
            if "ttH" in suffix or "tth" in suffix:
                label = "r_{t#bar{t}H}"
            elif "HH" in suffix or "hh" in suffix:
                label = f"r_{{{suffix}}}"
            elif suffix:
                label = f"r_{{{suffix}}}"
            else:
                label = "r"
        else:
            label = poi_name
        dhi_cfg.poi_data[poi_name] = DotDict(label=label, str=poi_name)


def main():
    parser = argparse.ArgumentParser(description="Generate publication-grade CMS Combine Impacts plot.")
    parser.add_argument("-i", "--input", required=True, help="Input impacts JSON file")
    parser.add_argument("-o", "--output", required=True, help="Output plot file (.pdf or .png)")
    parser.add_argument("-p", "--poi", default=None, help="POI name (default: auto-detected from JSON)")
    parser.add_argument("--per-page", type=int, default=20, help="Parameters per page (default: 20, -1 for all)")
    parser.add_argument("--page", type=int, default=-1, help="Specific page (0-indexed, default: -1 for all)")
    parser.add_argument("--order-by-impact", action="store_true", default=True, help="Sort by absolute impact (default: True)")
    parser.add_argument("--no-order-by-impact", dest="order_by_impact", action="store_false", help="Preserve original order")
    parser.add_argument("--pull-range", type=int, default=2, help="Symmetric pull range on bottom axis (default: 2)")
    parser.add_argument("--impact-range", type=float, default=-1.0, help="Symmetric impact range on top axis (default: auto)")
    parser.add_argument("--campaign", default="Run2", help="LPC/CMS campaign label key (default: 'Run2')")
    parser.add_argument("--cms-postfix", default="Preliminary", help="CMS header postfix (default: 'Preliminary')")
    parser.add_argument("--labels", default=None, help="JSON file with parameter name -> ROOT LaTeX remappings")
    parser.add_argument("--export-page-pngs", action="store_true", default=True, help="Export tight PNG for each page without A4 whitespace")
    args = parser.parse_args()

    # Load impacts data to determine POI and number of parameters
    with open(args.input, "r") as f:
        data = json.load(f)

    if not args.poi:
        if "POIs" in data and data["POIs"]:
            poi_name = data["POIs"][0]["name"]
        else:
            raise ValueError(f"No POIs found in {args.input}")
    else:
        poi_name = args.poi

    ensure_poi_registered(poi_name)

    total_params = len(data.get("params", []))
    per_page = args.per_page if args.per_page > 0 else total_params
    n_pages = max(1, math.ceil(float(total_params) / per_page))

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    base_name, ext = os.path.splitext(os.path.abspath(args.output))

    labels_map = None
    if args.labels and os.path.exists(args.labels):
        with open(args.labels, "r") as f:
            labels_map = json.load(f)

    ROOT = import_ROOT()
    ROOT.gROOT.SetBatch(True)

    # 1. If output is PDF (or includes all pages), generate multi-page PDF
    if ext.lower() == ".pdf":
        print(f"Generating multi-page PDF ({n_pages} page(s)): {args.output}")
        pi.plot_pulls_impacts(
            paths=[args.output],
            data=data,
            poi=poi_name,
            order_by_impact=args.order_by_impact,
            parameters_per_page=per_page,
            selected_page=args.page,
            pull_range=args.pull_range,
            impact_range=args.impact_range,
            campaign=args.campaign,
            cms_postfix=args.cms_postfix,
            labels=labels_map,
            style="dhi_default",
        )

    # 2. Direct per-page PNG export (completely eliminating blank A4 paper margins)
    if args.export_page_pngs or ext.lower() == ".png":
        pages_to_render = range(n_pages) if args.page < 0 else [args.page]
        for p_idx in pages_to_render:
            if n_pages > 1:
                page_png = f"{base_name}_page{p_idx + 1}.png"
            else:
                page_png = f"{base_name}.png"
            
            print(f"Generating tight PNG for Page {p_idx + 1}: {page_png}")
            pi.plot_pulls_impacts(
                paths=[page_png],
                data=data,
                poi=poi_name,
                order_by_impact=args.order_by_impact,
                parameters_per_page=per_page,
                selected_page=p_idx,
                pull_range=args.pull_range,
                impact_range=args.impact_range,
                campaign=args.campaign,
                cms_postfix=args.cms_postfix,
                labels=labels_map,
                style="dhi_default",
            )
            
            # Also create standard preview symlink or copy if multi-page
            if n_pages > 1 and p_idx == 0 and ext.lower() == ".pdf":
                p1_alias = f"{base_name}.png"
                if os.path.exists(page_png) and not os.path.exists(p1_alias):
                    try:
                        os.symlink(os.path.basename(page_png), p1_alias)
                    except OSError:
                        pass

    print("Impacts plotting complete!")


if __name__ == "__main__":
    main()
