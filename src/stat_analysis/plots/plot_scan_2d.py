#!/usr/bin/env python3
"""
Standalone 2D Likelihood Scan Plotter for CMS Combine.
Uses the publication-grade PyROOT styling from the CMS Di-Higgs combination tool,
completely self-contained within barista (zero dependency on law or external checkouts).

Plots 2D contour maps with 68% and 95% CL contours, best-fit cross, and SM reference point.

Example usage:
    python3 src/stat_analysis/plots/plot_scan_2d.py -i scan2d.root --poi1 kl --poi2 kt -o scan2d.pdf
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
        if poi_name == "kl":
            label = "#kappa_{#lambda}"
        elif poi_name == "kt":
            label = "#kappa_{t}"
        elif poi_name == "CV" or poi_name == "kV":
            label = "#kappa_{V}"
        elif poi_name == "C2V" or poi_name == "k2V":
            label = "#kappa_{2V}"
        elif poi_name.startswith("r"):
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


def load_root_scan_2d(root_path, poi1, poi2):
    """Reads MultiDimFit ROOT file with 2 POIs, extracts unique grid points."""
    ROOT = import_ROOT()
    f = ROOT.TFile.Open(root_path)
    if not f or f.IsZombie():
        raise FileNotFoundError(f"Cannot open ROOT file: {root_path}")
    tree = f.Get("limit")
    if not tree:
        f.Close()
        raise ValueError(f"Tree 'limit' not found in {root_path}")

    branches = [b.GetName() for b in tree.GetListOfBranches()]
    if poi1 not in branches or poi2 not in branches:
        f.Close()
        raise ValueError(f"Tree must contain branches '{poi1}' and '{poi2}'. Found: {branches}")

    has_dnll = "deltaNLL" in branches

    x_vals = []
    y_vals = []
    dnll_vals = []

    for entry in tree:
        x = getattr(entry, poi1)
        y = getattr(entry, poi2)
        nll = 2.0 * getattr(entry, "deltaNLL") if has_dnll else 0.0
        x_vals.append(x)
        y_vals.append(y)
        dnll_vals.append(nll)

    f.Close()

    x_arr = np.array(x_vals, dtype=float)
    y_arr = np.array(y_vals, dtype=float)
    dnll_arr = np.array(dnll_vals, dtype=float)

    # Shift minimum to 0
    if len(dnll_arr) > 0 and np.min(dnll_arr) < 0:
        dnll_arr -= np.min(dnll_arr)

    return {poi1: x_arr, poi2: y_arr, "dnll2": dnll_arr}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="ROOT file from Combine MultiDimFit (grid scan)")
    parser.add_argument("-o", "--output", required=True, help="Output plot path (.pdf or .png)")
    parser.add_argument("--poi1", required=True, help="Name of first POI (X axis)")
    parser.add_argument("--poi2", required=True, help="Name of second POI (Y axis)")
    parser.add_argument("--x-min", type=float, default=None, help="X axis min")
    parser.add_argument("--x-max", type=float, default=None, help="X axis max")
    parser.add_argument("--y-min", type=float, default=None, help="Y axis min")
    parser.add_argument("--y-max", type=float, default=None, help="Y axis max")
    parser.add_argument("--cms-postfix", default="Preliminary", help="CMS label postfix")
    parser.add_argument("--campaign", default="Run 2 / Run 3", help="Campaign / lumi label")
    parser.add_argument("--show-best-fit", action="store_true", default=True, help="Show best fit point marker")
    parser.add_argument("--show-sm-point", action="store_true", default=True, help="Show SM (1,1) point marker")

    args = parser.parse_args()

    ensure_poi_registered(args.poi1)
    ensure_poi_registered(args.poi2)

    data = load_root_scan_2d(args.input, args.poi1, args.poi2)

    out_base = os.path.splitext(args.output)[0]
    out_pdf = f"{out_base}.pdf"
    out_png = f"{out_base}.png"
    out_paths = [out_pdf, out_png]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"Generating 2D scan plot for {args.poi1} vs {args.poi2} ({len(data['dnll2'])} points)...")

    pl.plot_likelihood_scan_2d(
        paths=out_paths,
        poi1=args.poi1,
        poi2=args.poi2,
        values=data,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        show_best_fit=args.show_best_fit,
        show_sm_point=args.show_sm_point,
        cms_postfix=args.cms_postfix,
        campaign=args.campaign,
        cms_layout="outside_horizontal",
    )

    print(f"Saved 2D scan plots:\n  {out_pdf}\n  {out_png}")


if __name__ == "__main__":
    main()
