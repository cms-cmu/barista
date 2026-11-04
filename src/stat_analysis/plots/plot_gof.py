#!/usr/bin/env python3
"""
Standalone Goodness-of-Fit (GoF) Plotter for CMS Combine.
Uses the publication-grade PyROOT styling from the CMS Di-Higgs combination tool,
completely self-contained within barista (zero dependency on law or external checkouts).

Reads GoF JSON / ROOT outputs from combine -M GoodnessOfFit --algo saturated
and plots the toy test-statistic distribution with observed arrow and empirical p-value.
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
import dhi_core.plots.gof as pgof


def load_gof_data(json_or_root_path):
    """Loads observed value and toy values from JSON or ROOT file."""
    if json_or_root_path.endswith(".json"):
        with open(json_or_root_path, "r") as f:
            data = json.load(f)
        obs = float(data.get("data", data.get("observed", 0.0)))
        toys = [float(v) for v in data.get("toys", [])]
        return obs, np.array(toys, dtype=float)

    ROOT = import_ROOT()
    f = ROOT.TFile.Open(json_or_root_path)
    t = f.Get("limit")
    obs = None
    toys = []
    for i in range(t.GetEntries()):
        t.GetEntry(i)
        val = float(t.limit)
        if getattr(t, "iToy", 0) == 0 and obs is None:
            obs = val
        else:
            toys.append(val)
    f.Close()
    return obs, np.array(toys, dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Generate publication-grade CMS Combine Goodness-of-Fit plot.")
    parser.add_argument("-i", "--input", required=True, help="Input GoF JSON or merged toys ROOT file")
    parser.add_argument("--data", default=None, help="Optional separate data GoF ROOT file")
    parser.add_argument("-o", "--output", required=True, help="Output plot file (.pdf or .png)")
    parser.add_argument("--campaign", default="Run2", help="LPC/CMS campaign label key (default: 'Run2')")
    parser.add_argument("--cms-postfix", default="Preliminary", help="CMS header postfix (default: 'Preliminary')")
    args = parser.parse_args()

    obs, toys = load_gof_data(args.input)
    if args.data:
        obs_separate, _ = load_gof_data(args.data)
        if obs_separate is not None:
            obs = obs_separate

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    base_name, _ = os.path.splitext(os.path.abspath(args.output))

    out_pdf = f"{base_name}.pdf"
    out_png = f"{base_name}.png"

    print(f"Generating Goodness-of-Fit distribution plot ({len(toys)} toys, obs={obs:.2f})...")
    pgof.plot_gof_distribution(
        paths=[out_pdf, out_png],
        data=obs,
        toys=toys,
        algorithm="saturated",
        campaign=args.campaign,
        cms_postfix=args.cms_postfix,
        style="dhi_default",
    )
    print(f"Successfully generated: {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
