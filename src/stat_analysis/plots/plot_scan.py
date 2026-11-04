#!/usr/bin/env python3
"""
Standalone 1D & 2D Likelihood Scan Plotter for CMS Combine.
Uses the publication-grade PyROOT styling from the CMS Di-Higgs combination tool,
completely self-contained within barista (zero dependency on law or external checkouts).

Features:
- Exact CMS HH combination styling (ROOT canvas, spline curves, threshold lines).
- Computes exact crossing points for 1, 2, 3, 5 sigma and prints intervals.
- Supports overlaying Observed, Expected, and custom scans (e.g. Stat-only).
- Automatically exports both PDF and high-res PNG.
"""

import os
import sys
import argparse
import numpy as np

# Ensure dhi_core and plotlib are on sys.path
plots_dir = os.path.dirname(os.path.abspath(__file__))
if plots_dir not in sys.path:
    sys.path.insert(0, plots_dir)

import dhi_core
import dhi_core.config as dhi_cfg
from dhi_core.util import DotDict, import_ROOT
import dhi_core.plots.likelihoods as pl


def ensure_poi_registered(poi_name):
    """Ensure the given POI has a display label in dhi_cfg.poi_data."""
    if poi_name not in dhi_cfg.poi_data:
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


def load_root_scan(root_path, poi_name=None):
    """Reads MultiDimFit ROOT file, deduplicates and sorts points."""
    ROOT = import_ROOT()
    f = ROOT.TFile.Open(root_path)
    if not f or f.IsZombie():
        raise FileNotFoundError(f"Cannot open ROOT file: {root_path}")
    tree = f.Get("limit")
    if not tree:
        f.Close()
        raise ValueError(f"Tree 'limit' not found in {root_path}")

    # Determine POI branch if not set
    branches = [b.GetName() for b in tree.GetListOfBranches()]
    if not poi_name:
        candidates = [b for b in branches if b.startswith("r") and b not in ("quantileExpected", "real")]
        if not candidates:
            ignore = {"limit", "limitErr", "mh", "syst", "iToy", "iSeed", "iChannel", "t_cpu", "t_real", "quantileExpected", "deltaNLL"}
            candidates = [b for b in branches if b not in ignore]
        if candidates:
            poi_name = candidates[0]
        else:
            raise ValueError(f"Could not auto-detect POI in: {branches}")

    points = {}
    best_val = None
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        r = float(getattr(tree, poi_name))
        dnll2 = 2.0 * float(tree.deltaNLL)
        if i == 0 and dnll2 == 0.0:
            best_val = r
        if r in points:
            points[r] = min(points[r], dnll2)
        else:
            points[r] = dnll2
    f.Close()

    r_sorted = np.array(sorted(points.keys()), dtype=float)
    dnll2_sorted = np.array([points[k] for k in r_sorted], dtype=float)

    # Shift minimum to 0.0
    min_val = np.min(dnll2_sorted)
    dnll2_sorted = dnll2_sorted - min_val

    return poi_name, r_sorted, dnll2_sorted, best_val


def main():
    parser = argparse.ArgumentParser(description="Generate publication-grade CMS Combine Likelihood Scan plot.")
    parser.add_argument("--obs", default=None, help="Observed MultiDimFit ROOT file")
    parser.add_argument("--exp", default=None, help="Expected MultiDimFit ROOT file")
    parser.add_argument("--others", nargs="*", default=[], help="Other ROOT files in format 'file.root:label'")
    parser.add_argument("-p", "--poi", default=None, help="POI name (default: auto-detected)")
    parser.add_argument("-o", "--output", required=True, help="Output plot file (.pdf or .png)")
    parser.add_argument("--x-min", type=float, default=None, help="Minimum POI value on x-axis")
    parser.add_argument("--x-max", type=float, default=None, help="Maximum POI value on x-axis")
    parser.add_argument("--y-max", type=float, default=15.0, help="Maximum y-axis value (default: 15.0)")
    parser.add_argument("--campaign", default="Run2", help="LPC/CMS campaign label key (default: 'Run2')")
    parser.add_argument("--cms-postfix", default="Preliminary", help="CMS header postfix (default: 'Preliminary')")
    parser.add_argument("--show-best-fit", action="store_true", default=True, help="Display best fit value with uncertainties")
    parser.add_argument("--show-points", action="store_true", default=True, help="Display scan sample points")
    args = parser.parse_args()

    data_entries = []
    poi_name = args.poi

    # Add observed if provided
    if args.obs and os.path.exists(args.obs):
        p_name, r_vals, dnll2_vals, best_pt = load_root_scan(args.obs, poi_name)
        poi_name = p_name
        data_entries.append({
            "values": {poi_name: r_vals, "dnll2": dnll2_vals},
            "poi_min": best_pt,
            "name": "Observed",
        })

    # Add expected if provided
    if args.exp and os.path.exists(args.exp):
        p_name, r_vals, dnll2_vals, best_pt = load_root_scan(args.exp, poi_name)
        poi_name = p_name
        data_entries.append({
            "values": {poi_name: r_vals, "dnll2": dnll2_vals},
            "poi_min": best_pt,
            "name": "Expected",
        })

    # Add others (e.g. Stat-only)
    for item in args.others:
        parts = item.split(":")
        fpath = parts[0]
        lbl = parts[1] if len(parts) > 1 else os.path.basename(fpath)
        if os.path.exists(fpath):
            p_name, r_vals, dnll2_vals, best_pt = load_root_scan(fpath, poi_name)
            poi_name = p_name
            data_entries.append({
                "values": {poi_name: r_vals, "dnll2": dnll2_vals},
                "poi_min": best_pt,
                "name": lbl,
            })

    if not data_entries:
        raise ValueError("No valid input ROOT files provided (--obs, --exp, or --others).")

    ensure_poi_registered(poi_name)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    base_name, ext = os.path.splitext(os.path.abspath(args.output))

    out_pdf = f"{base_name}.pdf"
    out_png = f"{base_name}.png"

    print(f"Generating likelihood scan plot for POI '{poi_name}'...")
    pl.plot_likelihood_scans_1d(
        paths=[out_pdf, out_png],
        poi=poi_name,
        data=data_entries,
        campaign=args.campaign,
        cms_postfix=args.cms_postfix,
        cms_layout="outside_horizontal",
        style="dhi_default",
        show_best_fit=args.show_best_fit,
        show_points=args.show_points,
        x_min=args.x_min,
        x_max=args.x_max,
        y_max=args.y_max,
    )
    print(f"Successfully generated: {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
