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
import json
import math
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


def compute_uncertainty_breakdown(ranges_path, poi_name):
    if not ranges_path or not os.path.exists(ranges_path):
        return
    try:
        with open(ranges_path) as f:
            data = json.load(f)

        components = {}
        for k, v in data.items():
            if not isinstance(v, dict) or "uncertainty" not in v:
                continue
            unc = v.get("uncertainty", [])
            if not unc or len(unc) < 1:
                continue
            hi, lo = unc[0]
            hi_val = float(hi) if hi is not None else float("nan")
            lo_val = abs(float(lo)) if lo is not None else float("nan")
            best = float(v.get("best_fit", 1.0))
            components[k] = {"hi": hi_val, "lo": lo_val, "best": best}

        total_info = None
        bkg_info = None
        sig_info = None
        stat_info = None
        for k, c in components.items():
            lbl = k.split("__", 1)[-1].lower() if "__" in k else k.lower()
            if "bkg" in lbl:
                bkg_info = c
            elif "sig" in lbl:
                sig_info = c
            elif "stat" in lbl:
                stat_info = c
            elif "exp" in lbl or "total" in lbl or "obs" in lbl:
                if total_info is None:
                    total_info = c

        decomp = {}
        if total_info:
            decomp["total"] = {"hi": total_info["hi"], "lo": total_info["lo"], "best": total_info["best"]}
        if stat_info:
            decomp["stat"] = {
                "hi": stat_info["hi"],
                "lo": stat_info["lo"],
                "avg": (stat_info["hi"] + stat_info["lo"]) / 2.0,
            }
        if bkg_info and stat_info:
            bkg_hi = math.sqrt(max(0.0, bkg_info["hi"] ** 2 - stat_info["hi"] ** 2))
            bkg_lo = math.sqrt(max(0.0, bkg_info["lo"] ** 2 - stat_info["lo"] ** 2))
            decomp["bkg"] = {
                "hi": bkg_hi,
                "lo": bkg_lo,
                "avg": (bkg_hi + bkg_lo) / 2.0,
                "raw_hi": bkg_info["hi"],
                "raw_lo": bkg_info["lo"],
            }
        elif total_info and stat_info and not sig_info:
            # For bkg-syst-only workflows where total is bkg+stat
            bkg_hi = math.sqrt(max(0.0, total_info["hi"] ** 2 - stat_info["hi"] ** 2))
            bkg_lo = math.sqrt(max(0.0, total_info["lo"] ** 2 - stat_info["lo"] ** 2))
            decomp["bkg"] = {
                "hi": bkg_hi,
                "lo": bkg_lo,
                "avg": (bkg_hi + bkg_lo) / 2.0,
                "raw_hi": total_info["hi"],
                "raw_lo": total_info["lo"],
            }
        if sig_info and stat_info:
            sig_hi = math.sqrt(max(0.0, sig_info["hi"] ** 2 - stat_info["hi"] ** 2))
            sig_lo = math.sqrt(max(0.0, sig_info["lo"] ** 2 - stat_info["lo"] ** 2))
            decomp["sig"] = {
                "hi": sig_hi,
                "lo": sig_lo,
                "avg": (sig_hi + sig_lo) / 2.0,
                "raw_hi": sig_info["hi"],
                "raw_lo": sig_info["lo"],
            }

        out_struct = {
            "poi": poi_name,
            "ranges": data,
            "decomposition": decomp,
        }
        with open(ranges_path, "w") as f:
            json.dump(out_struct, f, indent=2)

        if decomp:
            print("\n" + "=" * 60)
            print(f" Likelihood Fit Decomposition for {poi_name}:")
            if "total" in decomp:
                print(f"   Total uncertainty : +{decomp['total']['hi']:.4f} / -{decomp['total']['lo']:.4f}")
            if "stat" in decomp:
                print(f"   Statistical only  : +{decomp['stat']['hi']:.4f} / -{decomp['stat']['lo']:.4f} (avg: ±{decomp['stat']['avg']:.4f})")
            if "bkg" in decomp:
                print(f"   Background syst   : +{decomp['bkg']['hi']:.4f} / -{decomp['bkg']['lo']:.4f} (avg: ±{decomp['bkg']['avg']:.4f})")
            if "sig" in decomp:
                print(f"   Signal syst       : +{decomp['sig']['hi']:.4f} / -{decomp['sig']['lo']:.4f} (avg: ±{decomp['sig']['avg']:.4f})")
            print("=" * 60 + "\n")
    except Exception as e:
        print(f"Warning: could not compute uncertainty breakdown: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate publication-grade CMS Combine Likelihood Scan plot.")
    parser.add_argument("--obs", default=None, help="Observed MultiDimFit ROOT file")
    parser.add_argument("--exp", default=None, help="Expected MultiDimFit ROOT file")
    parser.add_argument("--others", nargs="*", default=[], help="Other ROOT files in format 'file.root:label'")
    parser.add_argument("--ranges-path", default=None, help="Save parameter ranges and uncertainties to JSON file")
    parser.add_argument("--exp-label", default=None, help="Legend label for expected scan curve")
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
        exp_lbl = args.exp_label or ("Total (all syst)" if args.others else "Expected")
        data_entries.append({
            "values": {poi_name: r_vals, "dnll2": dnll2_vals},
            "poi_min": best_pt,
            "name": exp_lbl,
        })

    # Add others (e.g. Stat-only, Bkg-only, Sig-only)
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
        ranges_path=args.ranges_path,
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

    if args.ranges_path:
        compute_uncertainty_breakdown(args.ranges_path, poi_name)


if __name__ == "__main__":
    main()
