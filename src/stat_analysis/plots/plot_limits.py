#!/usr/bin/env python3
"""
Standalone Upper Limits & Brazil Bands Plotter for CMS Combine.
Uses the publication-grade PyROOT styling from the CMS Di-Higgs combination tool,
completely self-contained within barista (zero dependency on law or external checkouts).

Reads datacard_limits.json (from AsymptoticLimits) and produces Brazil flag plots.
"""

import os
import sys
import json
import argparse
import numpy as np

# Ensure dhi_core and plotlib are on sys.path
plots_dir = os.path.dirname(os.path.abspath(__file__))
if plots_dir not in sys.path:
    sys.path.insert(0, plots_dir)

import dhi_core
import dhi_core.config as dhi_cfg
from dhi_core.util import DotDict, import_ROOT
import dhi_core.plots.limits as plim


def ensure_poi_registered(poi_name):
    if poi_name not in dhi_cfg.poi_data:
        dhi_cfg.poi_data[poi_name] = DotDict(label=poi_name, str=poi_name)


def main():
    parser = argparse.ArgumentParser(description="Generate publication-grade CMS Combine Upper Limits / Brazil Band plot.")
    parser.add_argument("-i", "--input", required=True, help="Input limits JSON file")
    parser.add_argument("-o", "--output", required=True, help="Output plot file (.pdf or .png)")
    parser.add_argument("-p", "--poi", default="r", help="POI name (default: 'r')")
    parser.add_argument("--campaign", default="Run2", help="LPC/CMS campaign label key (default: 'Run2')")
    parser.add_argument("--cms-postfix", default="Preliminary", help="CMS header postfix (default: 'Preliminary')")
    parser.add_argument("--unblinded", action="store_true", help="Display observed limits")
    args = parser.parse_args()

    ensure_poi_registered(args.poi)

    with open(args.input, "r") as f:
        data = json.load(f)

    # Unwrap mass key if present (e.g. {"120.0": {...}})
    if len(data) == 1 and any(isinstance(v, dict) for v in data.values()):
        first_key = list(data.keys())[0]
        data = data[first_key]

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    base_name, _ = os.path.splitext(os.path.abspath(args.output))

    out_pdf = f"{base_name}.pdf"
    out_png = f"{base_name}.png"

    # Extract limit quantiles
    exp0 = float(data.get("exp0", data.get("0.5", data.get("exp", 1.0))))
    exp_p1 = float(data.get("exp+1", data.get("0.84", exp0 * 1.5)))
    exp_m1 = float(data.get("exp-1", data.get("0.16", exp0 * 0.7)))
    exp_p2 = float(data.get("exp+2", data.get("0.975", exp0 * 2.0)))
    exp_m2 = float(data.get("exp-2", data.get("0.025", exp0 * 0.5)))

    # Format into limit points for plim.plot_limit_points
    # Expected: list of 5 values [exp-2, exp-1, exp0, exp+1, exp+2]
    point_dict = {
        "name": args.poi,
        "expected": [exp_m2, exp_m1, exp0, exp_p1, exp_p2],
    }
    if args.unblinded and ("obs" in data or "-1.0" in data):
        obs_val = float(data.get("obs", data.get("-1.0", exp0)))
        point_dict["observed"] = obs_val

    points = [point_dict]

    print(f"Generating upper limits plot for POI '{args.poi}'...")
    plim.plot_limit_points(
        paths=[out_pdf, out_png],
        poi=args.poi,
        data=points,
        campaign=args.campaign,
        cms_postfix=args.cms_postfix,
        cms_layout="outside_horizontal",
        style="dhi_default",
    )
    print(f"Successfully generated: {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
