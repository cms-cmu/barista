#!/usr/bin/env python3
# coding: utf-8

"""
Standalone script to create post-fit signal-over-background (S/sqrt(B) or S/B)
summary plots combined over all categories and bins in a Combine fit.
Uses the publication-grade PyROOT styling from the CMS Di-Higgs combination package,
completely decoupled within barista.

Example usage:
    python3 src/stat_analysis/plots/plot_s_over_b.py -i fitDiagnostics.root -o s_over_b.pdf --poi rttHbb
    python3 src/stat_analysis/plots/plot_s_over_b.py -i fitDiagnostics.root -o s_over_b.pdf --poi r --bins 10
"""

import os
import sys
import argparse

plots_dir = os.path.dirname(os.path.abspath(__file__))
if plots_dir not in sys.path:
    sys.path.insert(0, plots_dir)

import dhi_core
import dhi_core.config as dhi_cfg
from dhi_core.util import DotDict
import dhi_core.plots.postfit_shapes as ps


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="Input ROOT file from Combine FitDiagnostics")
    parser.add_argument("-o", "--output", required=True, help="Output plot path (.pdf or .png)")
    parser.add_argument("--poi", default="r", help="Name of parameter of interest (default: 'r')")
    parser.add_argument("--bins", type=int, default=8, help="Number of S/B bins (default: 8)")
    parser.add_argument("--order-without-sqrt", action="store_true", help="Order by log(S/B) instead of log(S/sqrt(B))")
    parser.add_argument("--signal-scale", type=float, default=1.0, help="Scale factor for signal visualization")
    parser.add_argument("--signal-superimposed", action="store_true", help="Draw signal superimposed rather than stacked")
    parser.add_argument("--campaign", default="Run 2 / Run 3", help="Campaign or lumi string for top-right label")
    parser.add_argument("--cms-postfix", default="Preliminary", help="CMS label postfix")
    parser.add_argument("--prefit", action="store_true", help="Plot pre-fit rather than post-fit shapes")
    parser.add_argument("--unblinded", action="store_true", help="Show observed data points")

    args = parser.parse_args()

    ensure_poi_registered(args.poi)

    out_base = os.path.splitext(args.output)[0]
    out_pdf = f"{out_base}.pdf"
    out_png = f"{out_base}.png"
    out_paths = [out_pdf, out_png]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"Generating S/sqrt(B) postfit plot for POI '{args.poi}' from: {args.input}...")

    ps.plot_s_over_b(
        paths=out_paths,
        poi=args.poi,
        fit_diagnostics_path=os.path.abspath(args.input),
        bins=args.bins,
        order_without_sqrt=args.order_without_sqrt,
        signal_superimposed=args.signal_superimposed,
        signal_scale=args.signal_scale,
        campaign=args.campaign,
        cms_postfix=args.cms_postfix,
        prefit=args.prefit,
        unblinded=args.unblinded,
    )

    print(f"Saved S/B postfit plots:\n  {out_pdf}\n  {out_png}")


if __name__ == "__main__":
    main()
