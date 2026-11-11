#!/usr/bin/env python3
# coding: utf-8

"""
Standalone script to plot limit comparison points across multiple channels, categories, or eras.
Uses the publication-grade PyROOT styling from the CMS Di-Higgs combination package,
completely decoupled within barista.

Example usage:
    python3 src/stat_analysis/plots/plot_limit_points.py -i limits_HH4b.json limits_ttHbb.json -o limits_comparison.pdf --poi r
    python3 src/stat_analysis/plots/plot_limit_points.py -i all_limits.json -o limits_comparison.pdf --poi r --sort
"""

import os
import sys
import json
import argparse

plots_dir = os.path.dirname(os.path.abspath(__file__))
if plots_dir not in sys.path:
    sys.path.insert(0, plots_dir)

import dhi_core
import dhi_core.config as dhi_cfg
from dhi_core.util import DotDict
import dhi_core.plots.limits as dhi_limits


def ensure_poi_registered(poi_name):
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


def parse_limit_dict(d, name):
    # Extract expected 5 values: -2, -1, 0, +1, +2
    exp_m2 = d.get("exp-2", d.get("0.025", 0.0))
    exp_m1 = d.get("exp-1", d.get("0.160", 0.0))
    exp_0 = d.get("exp0", d.get("0.500", 0.0))
    exp_p1 = d.get("exp+1", d.get("0.840", 0.0))
    exp_p2 = d.get("exp+2", d.get("0.975", 0.0))
    obs = d.get("obs", None)

    entry = {
        "name": name,
        "expected": (exp_0, exp_p1, exp_m1, exp_p2, exp_m2),
    }
    if obs is not None:
        entry["observed"] = float(obs)
    return entry


def load_limits_data(input_paths):
    data_points = []
    for path in input_paths:
        with open(path, "r") as f:
            raw = json.load(f)

        base_name = os.path.splitext(os.path.basename(path))[0].replace("datacard_limits__", "")
        # Check if file has multiple channel keys
        if any(isinstance(v, dict) for v in raw.values()):
            for k, v in raw.items():
                if isinstance(v, dict) and any(x in v for x in ["exp0", "0.500"]):
                    data_points.append(parse_limit_dict(v, k))
        else:
            data_points.append(parse_limit_dict(raw, base_name))
    return data_points


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--inputs", nargs="+", required=True, help="Input limit JSON file(s)")
    parser.add_argument("-o", "--output", required=True, help="Output plot path (.pdf or .png)")
    parser.add_argument("--poi", default="r", help="POI name (default: 'r')")
    parser.add_argument("--sort", action="store_true", help="Sort entries by expected limit")
    parser.add_argument("--x-min", type=float, help="Min limit value on X axis")
    parser.add_argument("--x-max", type=float, help="Max limit value on X axis")
    parser.add_argument("--x-log", action="store_true", help="Logarithmic X axis")
    parser.add_argument("--campaign", default="Run 2 / Run 3", help="Campaign / lumi label")
    parser.add_argument("--cms-postfix", default="Preliminary", help="CMS postfix label")

    args = parser.parse_args()

    ensure_poi_registered(args.poi)

    data = load_limits_data(args.inputs)
    if not data:
        raise ValueError(f"No valid limit entries found in: {args.inputs}")

    out_base = os.path.splitext(args.output)[0]
    out_pdf = f"{out_base}.pdf"
    out_png = f"{out_base}.png"
    out_paths = [out_pdf, out_png]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"Generating limit comparison plot for {len(data)} channels/categories...")

    dhi_limits.plot_limit_points(
        paths=out_paths,
        poi=args.poi,
        data=data,
        sort_by="expected" if args.sort else None,
        x_min=args.x_min,
        x_max=args.x_max,
        x_log=args.x_log,
        campaign=args.campaign,
        cms_postfix=args.cms_postfix,
        cms_layout="outside_horizontal",
    )

    print(f"Saved limit comparison plots:\n  {out_pdf}\n  {out_png}")


if __name__ == "__main__":
    main()
