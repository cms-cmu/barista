#!/usr/bin/env python
"""One-page HTML summary of a Combine statistical analysis directory.

Reads what the ``combine.smk`` rules leave under ``<stat_dir>/<channel>/``:

* ``limits/datacard_limits__<signal>.json``          AsymptoticLimits (exp-2 … exp+2, obs)
* ``significance/datacard_significance__<signal>.json``  observed / expected significance
* ``datacards/datacard_<channel>_<year>.txt``       per-bin observation and process rates
* ``likelihood_scan/higgsCombine_merged_<signal>*.root`` scan points (valid / failed), ``scan_plot.png``
* ``postfit/<channel>_<year>_CMS_th1x_fit_s.png``, ``postfit/covariance_fit_s.png``

and writes a self-contained ``summary.html`` (plus an optional Markdown twin) with an overview
table over all channels, and per channel the yields per year, the figures and links to the raw
outputs. All links are relative to the summary file, so the page works wherever the directory is
copied (e.g. roast publish).

Usage::

    python src/stat_analysis/stat_summary.py output/TESTRun2/stat_analysis -o output/TESTRun2/stat_analysis/summary.html
        [--channel HH4b=ggHH_kl_1_kt_1_13p0TeV_hbbhbb --channel ZZ4b=ZZ_bbbb ...]   # default: discover
        [--variable HH4b=SvB_MA.ps_hh_fine ...] [--title T] [--blind] [--md summary.md]
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


logger = logging.getLogger("stat_summary")


# --------------------------------------------------------------------------- readers
def read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"cannot read {path}: {e}")
        return None


def read_limits(path: str) -> dict | None:
    """{'exp-2','exp-1','exp0','exp+1','exp+2','obs'} for the (single) mass point."""
    d = read_json(path)
    if not d:
        return None
    if isinstance(d, dict) and d and all(isinstance(v, dict) for v in d.values()):
        d = next(iter(d.values()))  # {"120.0": {...}}
    return d if isinstance(d, dict) else None


def read_significance(path: str) -> dict | None:
    d = read_json(path)
    if not d:
        return None
    out = {}
    for k in ("observed", "expected"):
        v = d.get(k)
        out[k] = v.get("significance") if isinstance(v, dict) else v
    return out


def read_datacard(path: str) -> dict | None:
    """Parse a single-bin datacard: {'bin', 'observation', 'rates': {process: rate}}."""
    try:
        lines = [ln.rstrip("\n") for ln in open(path)]
    except OSError:
        return None
    out = {"bin": None, "observation": None, "rates": OrderedDict(), "nuisances": 0}
    procs = None
    for ln in lines:
        t = ln.split()
        if not t:
            continue
        if t[0] == "bin" and len(t) == 2 and out["bin"] is None:
            out["bin"] = t[1]
        elif t[0] == "observation" and len(t) >= 2:
            try:
                out["observation"] = float(t[1])
            except ValueError:
                pass
        elif t[0] == "process" and procs is None:
            procs = t[1:]
        elif t[0] == "rate" and procs:
            for p, r in zip(procs, t[1:]):
                try:
                    out["rates"][p] = float(r)
                except ValueError:
                    out["rates"][p] = float("nan")
        elif len(t) >= 2 and t[1] in ("lnN", "shape", "shapeN2", "lnU", "gmN", "rateParam", "param"):
            out["nuisances"] += 1
    return out


def read_scan(path: str, poi: str) -> dict | None:
    """Extract best fit and 68% CL crossing from a merged MultiDimFit tree."""
    try:
        import uproot  # noqa: F401
    except Exception:  # noqa: BLE001
        return None
    try:
        t = uproot.open(path)["limit"]
        branches = set(t.keys())
        pb = poi if poi in branches else next((b for b in branches if b.startswith("r") and b not in ("run",)), None)
        if pb is None:
            return None
        a = t.arrays([pb, "deltaNLL", "quantileExpected"], library="np")
        best = [float(r) for r, q in zip(a[pb], a["quantileExpected"]) if q < 0]
        best_r = best[0] if best else float("nan")

        pts = [
            (float(r), float(2 * d))
            for r, d, q in zip(a[pb], a["deltaNLL"], a["quantileExpected"])
            if q >= 0 and d < 9000 and not (math.isnan(d) or math.isinf(d))
        ]
        pts.sort(key=lambda x: x[0])

        r_lo, r_hi = None, None
        left = [p for p in pts if p[0] <= best_r]
        right = [p for p in pts if p[0] >= best_r]

        # Interpolate left crossing where 2*deltaNLL == 1.0
        if any(p[1] >= 1.0 for p in left):
            for i in range(len(left) - 1):
                (r1, d1), (r2, d2) = left[i], left[i + 1]
                if (d1 >= 1.0 and d2 <= 1.0) or (d1 <= 1.0 and d2 >= 1.0):
                    r_lo = r1 + (1.0 - d1) * (r2 - r1) / (d2 - d1) if d2 != d1 else r1

        # Interpolate right crossing where 2*deltaNLL == 1.0
        if any(p[1] >= 1.0 for p in right):
            for i in range(len(right) - 1):
                (r1, d1), (r2, d2) = right[i], right[i + 1]
                if (d1 <= 1.0 and d2 >= 1.0) or (d1 >= 1.0 and d2 <= 1.0):
                    r_hi = r1 + (1.0 - d1) * (r2 - r1) / (d2 - d1) if d2 != d1 else r2

        err_lo = (best_r - r_lo) if (r_lo is not None and not math.isnan(best_r)) else None
        err_hi = (r_hi - best_r) if (r_hi is not None and not math.isnan(best_r)) else None

        return {
            "poi": pb,
            "bestfit": best_r,
            "err_lo": err_lo,
            "err_hi": err_hi,
            "total": len(pts),
            "valid": len(pts),
        }
    except Exception as e:  # noqa: BLE001
        logger.warning(f"cannot read scan {path}: {e}")
        return None


def read_impacts(path: str, poi: str = "r") -> list[dict] | None:
    """Read nuisance parameter impacts from a combineTool Impacts json output."""
    d = read_json(path)
    if not d:
        return None
    params = d.get("params", [])
    out = []
    for p in params:
        name = p.get("name", "")
        fit = p.get("fit", [0, 0, 0])
        # combineTool: impact_<poi> is a number, <poi> is [lo, nominal, hi]. A genuine 0.0 impact is
        # falsy, so only fall back when the key is missing, and turn a [lo, nom, hi] into (hi - lo) / 2.
        impact = p.get(f"impact_{poi}")
        if impact is None and poi in p:
            impact = p[poi]
        if impact is None:
            for k, v in p.items():
                if (k.startswith("impact_r") or k.startswith("r")) and isinstance(v, (int, float)):
                    impact = v
                    break
        if isinstance(impact, (list, tuple)):
            impact = (impact[-1] - impact[0]) / 2 if len(impact) >= 2 else (impact[0] if impact else 0.0)
        out.append({
            "name": name,
            "fit": fit,
            "impact": float(impact) if impact is not None else 0.0,
            "type": p.get("type", ""),
        })
    out.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return out


# --------------------------------------------------------------------------- discovery
def discover_channels(stat_dir: str) -> "OrderedDict[str, str]":
    """{channel: signallabel} from <stat_dir>/<channel>/limits/datacard_limits__<signal>.json."""
    found = OrderedDict()
    for p in sorted(glob.glob(os.path.join(stat_dir, "*", "limits", "datacard_limits__*.json"))):
        ch = os.path.basename(os.path.dirname(os.path.dirname(p)))
        sig = os.path.basename(p)[len("datacard_limits__"):-len(".json")]
        found.setdefault(ch, sig)
    return found


def collect_channel(stat_dir: str, channel: str, signal: str, variable: str | None) -> dict:
    d = os.path.join(stat_dir, channel)
    rel = lambda *p: os.path.relpath(os.path.join(d, *p), stat_dir)  # noqa: E731
    exists = lambda *p: os.path.exists(os.path.join(d, *p))  # noqa: E731
    info = {"channel": channel, "signal": signal, "variable": variable, "links": OrderedDict(), "figures": OrderedDict()}

    info["limits"] = read_limits(os.path.join(d, "limits", f"datacard_limits__{signal}.json"))
    info["significance"] = read_significance(os.path.join(d, "significance", f"datacard_significance__{signal}.json"))

    years = OrderedDict()
    cards = sorted(glob.glob(os.path.join(d, "datacards", f"datacard_{channel}_*.txt")))
    if not cards:
        cards = sorted([
            p for p in glob.glob(os.path.join(d, "datacards", "datacard_*.txt"))
            if not os.path.basename(p).startswith("datacard__")
        ])
    for p in cards:
        bname = os.path.basename(p)
        if bname.startswith(f"datacard_{channel}_"):
            year = bname[len(f"datacard_{channel}_"):-len(".txt")]
        elif bname.startswith(f"datacard_{signal}_"):
            year = bname[len(f"datacard_{signal}_"):-len(".txt")]
        else:
            year = bname[len("datacard_"):-len(".txt")]
        card = read_datacard(p)
        if card:
            years[year] = card
    info["years"] = years
    # process columns: keep datacard order, signals first as they appear
    procs = []
    for card in years.values():
        for p in card["rates"]:
            if p not in procs:
                procs.append(p)
    info["processes"] = procs
    info["totals"] = {p: sum(c["rates"].get(p, 0.0) for c in years.values()) for p in procs}
    info["observation"] = sum(c["observation"] or 0.0 for c in years.values())
    info["nuisances"] = max((c["nuisances"] for c in years.values()), default=0)

    scan_json_path = os.path.join(d, "likelihood_scan", f"datacard_likelihood_scan__{signal}.json")
    if os.path.exists(scan_json_path):
        info["scan_json"] = read_json(scan_json_path)

    sjson = info.get("scan_json")
    if sjson and isinstance(sjson, dict) and "decomposition" in sjson and "total" in sjson["decomposition"]:
        tot = sjson["decomposition"]["total"]
        info["scan"] = {
            "poi": sjson.get("poi", f"r{signal}"),
            "bestfit": float(tot.get("best", 1.0)),
            "err_lo": float(tot.get("lo", 0.0)),
            "err_hi": float(tot.get("hi", 0.0)),
        }
    else:
        # Prefer nominal/total expected or base scan root file over intermediate fit slices (_bkg, _sig, _stat)
        merged_exp = sorted(glob.glob(os.path.join(d, "likelihood_scan", f"higgsCombine_merged_{signal}_exp*.root")))
        merged_base = sorted(glob.glob(os.path.join(d, "likelihood_scan", f"higgsCombine_merged_{signal}.*.root")))
        merged_all = sorted(glob.glob(os.path.join(d, "likelihood_scan", f"higgsCombine_merged_{signal}*.root")))
        merged_target = merged_exp or merged_base or merged_all
        info["scan"] = read_scan(merged_target[0], f"r{signal}") if merged_target else None

    impacts_json_path = os.path.join(d, "impacts", f"impacts_combine_datacard__{signal}_exp.json")
    if not os.path.exists(impacts_json_path):
        cands = sorted(glob.glob(os.path.join(d, "impacts", "impacts_combine_*.json")))
        if cands:
            impacts_json_path = cands[0]
    info["impacts"] = read_impacts(impacts_json_path, f"r{signal}") if os.path.exists(impacts_json_path) else None

    # links
    for label, *p in (
        ("combined datacard", "datacards", f"datacard__{channel}.txt"),
        ("workspace", "workspace", f"datacard__{signal}.root"),
        ("limits json", "limits", f"datacard_limits__{signal}.json"),
        ("limits log", "limits", f"datacard_limits__{signal}.txt"),
        ("significance log", "significance", f"datacard_significance__{signal}.log"),
        ("likelihood scan pdf", "likelihood_scan", f"datacard_likelihood_scan__{signal}.pdf"),
        ("postfit pdf", "postfit", f"datacard_postfit__{signal}.pdf"),
        ("impacts pdf", "impacts", f"datacard_impacts__{signal}.pdf"),
        ("impacts json", "impacts", os.path.basename(impacts_json_path) if os.path.exists(impacts_json_path) else f"impacts_combine_datacard__{signal}_exp.json"),
        ("fitDiagnostics (b-only)", "postfit", f"datacard_fitDiagnostics_bonly__{signal}.root"),
        ("diffNuisances (b-only)", "postfit", f"datacard_diffNuisances_bonly__{signal}.root"),
        ("correlation (physics) pdf", "postfit", "plots", "correlation_fit_s.pdf"),
        ("correlation (all) pdf", "postfit", "plots", "correlation_fit_s_all.pdf"),
        ("systematics gallery", "plots_systematics", "index.html"),
        ("likelihood scan json", "likelihood_scan", f"datacard_likelihood_scan__{signal}.json"),
    ):
        if exists(*p):
            info["links"][label] = rel(*p)
    for year, card in years.items():
        info["links"][f"datacard {year}"] = rel("datacards", f"datacard_{channel}_{year}.txt")

    # figures
    for scan_candidate in (f"datacard_likelihood_scan__{signal}.png", "scan_plot.png"):
        if exists("likelihood_scan", scan_candidate):
            info["figures"]["likelihood scan"] = rel("likelihood_scan", scan_candidate)
            break
    for p in sorted(glob.glob(os.path.join(d, "postfit", "plots", f"postfitplots__{signal}__*.png"))):
        info["figures"][f"postfit {os.path.basename(p).split('__')[-1][:-4]}"] = rel("postfit", "plots", os.path.basename(p))
    if exists("postfit", "plots", "correlation_fit_s.png"):
        info["figures"]["correlation (physics)"] = rel("postfit", "plots", "correlation_fit_s.png")
    if exists("postfit", "plots", "correlation_fit_s_all.png"):
        info["figures"]["correlation (all)"] = rel("postfit", "plots", "correlation_fit_s_all.png")
    elif exists("postfit", "covariance_fit_s.png"):
        info["figures"]["covariance (fit_s)"] = rel("postfit", "covariance_fit_s.png")

    impacts_pdf = os.path.join(d, "impacts", f"datacard_impacts__{signal}.pdf")
    is_stat_only = "_stat_only" in channel or channel.endswith("_stat")
    if os.path.exists(impacts_pdf) and not is_stat_only:
        split_dir = os.path.join(d, "impacts", f"datacard_impacts_split__{signal}")
        if not os.path.isdir(split_dir):
            split_dirs = glob.glob(os.path.join(d, "impacts", "*impacts_split*"))
            if split_dirs:
                split_dir = split_dirs[0]
        impacts_pngs = sorted(glob.glob(os.path.join(split_dir, "*.png"))) if os.path.isdir(split_dir) else []
        if not impacts_pngs:
            cands = sorted([p for p in glob.glob(os.path.join(d, "impacts", f"datacard_impacts__{signal}_page*.png")) if not os.path.islink(p)])
            if cands:
                impacts_pngs = cands
            else:
                cands2 = sorted([p for p in glob.glob(os.path.join(d, "impacts", "impacts_page-*.png")) if not os.path.islink(p)])
                if cands2:
                    impacts_pngs = cands2
                else:
                    impacts_pngs = sorted([p for p in glob.glob(os.path.join(d, "impacts", "*.png")) if not os.path.islink(p)])
        if not impacts_pngs and shutil.which("pdftoppm"):
            try:
                subprocess.run(
                    ["pdftoppm", "-png", "-r", "150", impacts_pdf, os.path.join(d, "impacts", "impacts_page")],
                    check=False, capture_output=True
                )
                impacts_pngs = sorted(glob.glob(os.path.join(d, "impacts", "impacts_page-*.png")))
            except Exception as e:
                logger.warning(f"Could not convert impacts PDF to PNG: {e}")
        for i, png in enumerate(impacts_pngs):
            label = "impacts" if len(impacts_pngs) == 1 else f"impacts (page {i+1})"
            info["figures"][label] = os.path.relpath(png, stat_dir)

    return info




# --------------------------------------------------------------------------- formatting
def fnum(v, digits=1) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{digits}f}" if abs(v) >= 1 else f"{v:.3g}"


def flim(lim: dict | None) -> tuple[str, str, str]:
    """(median, ±1σ band, ±2σ band) strings for the expected limit."""
    if not lim:
        return "-", "-", "-"
    g = lambda k: lim.get(k)  # noqa: E731
    med = fnum(g("exp0"), 2)
    one = f"[{fnum(g('exp-1'), 2)}, {fnum(g('exp+1'), 2)}]" if g("exp-1") is not None else "-"
    two = f"[{fnum(g('exp-2'), 2)}, {fnum(g('exp+2'), 2)}]" if g("exp-2") is not None else "-"
    return med, one, two


def fsig(s) -> str:
    return fnum(s, 2) if isinstance(s, (int, float)) else "-"


def fscan(sc: dict | None) -> str:
    if not sc or math.isnan(sc.get("bestfit", float("nan"))):
        return "-"
    best = sc["bestfit"]
    elo = sc.get("err_lo")
    ehi = sc.get("err_hi")
    if elo is not None and ehi is not None:
        if abs(elo - ehi) < 0.005:
            return f"{best:.2f} ± {ehi:.2f}"
        return f"{best:.2f} +{ehi:.2f}/−{elo:.2f}"
    elif ehi is not None:
        return f"{best:.2f} +{ehi:.2f}"
    elif elo is not None:
        return f"{best:.2f} −{elo:.2f}"
    return f"{best:.2f}"


def fscan_decomp(sc: dict | None, sjson: dict | None) -> str:
    base = fscan(sc)
    if not sjson or not isinstance(sjson, dict):
        return f"<b>{html.escape(base)}</b>"
    decomp = sjson.get("decomposition", {})
    if not decomp:
        return f"<b>{html.escape(base)}</b>"
    parts = []
    if "stat" in decomp:
        parts.append(f"stat ±{decomp['stat']['avg']:.2f}")
    if "bkg" in decomp:
        parts.append(f"bkg ±{decomp['bkg']['avg']:.2f}")
    if "sig" in decomp:
        parts.append(f"sig ±{decomp['sig']['avg']:.2f}")
    if parts:
        return f"<b>{html.escape(base)}</b><br><span class='small' style='white-space:nowrap;'>({', '.join(parts)})</span>"
    return f"<b>{html.escape(base)}</b>"


def short_proc(p: str) -> str:
    return p.replace("_13p0TeV_hbbhbb", "").replace("ggHH_kl_", "ggHH kl=").replace("_kt_1", "")


CSS = """
:root {
  --bg: #f8fafc;
  --fg: #0f172a;
  --muted: #64748b;
  --card: #ffffff;
  --line: #e2e8f0;
  --accent: #2563eb;
  --accent-hover: #1d4ed8;
  --accent-soft: #eff6ff;
  --pass-bg: #dcfce7;
  --pass-fg: #15803d;
  --pass-border: #bbf7d0;
  --warn-bg: #fef3c7;
  --warn-fg: #b45309;
  --warn-border: #fde68a;
  --danger-bg: #fee2e2;
  --danger-fg: #b91c1c;
  --danger-border: #fecaca;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: var(--sans);
  background: var(--bg);
  color: var(--fg);
  line-height: 1.5;
  font-size: 14px;
  padding: 24px 32px 64px;
}
.container { max-width: 1400px; margin: 0 auto; display: flex; flex-direction: column; gap: 18px; }
header {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 18px 24px;
}
h1 { font-size: 1.35rem; font-weight: 700; margin-bottom: 6px; }
h2 { font-size: 1.1rem; font-weight: 600; margin: 18px 0 10px 0; border-bottom: 1px solid var(--line); padding-bottom: 6px; display: flex; align-items: center; gap: 8px; }
h3 { font-size: 0.95rem; font-weight: 600; margin: 14px 0 6px 0; color: var(--fg); }
.meta { color: var(--muted); font-size: 13px; display: flex; flex-wrap: wrap; align-items: center; gap: 6px 12px; }
code { font-family: var(--mono); font-size: 0.88em; background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 18px; }
.wrap-table { overflow-x: auto; background: var(--card); border: 1px solid var(--line); border-radius: 8px; margin: 8px 0 12px 0; }
table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--line); padding: 8px 12px; text-align: right; }
th { background: #f8fafc; color: var(--muted); font-size: 11.5px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.03em; }
td:first-child, th:first-child { text-align: left; }
td.l, th.l { text-align: left; }
tr:hover td { background: #f8fafc; }
.badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 4px; font-size: 11.5px; font-weight: 600; white-space: nowrap; }
.blind { background: var(--warn-bg); color: var(--warn-fg); border: 1px solid var(--warn-border); }
.warn { background: var(--warn-bg); color: var(--warn-fg); border: 1px solid var(--warn-border); }
.ok { background: var(--pass-bg); color: var(--pass-fg); border: 1px solid var(--pass-border); }
.figs { display: flex; flex-wrap: wrap; gap: 14px; margin: 12px 0; }
.figs figure { margin: 0; width: 340px; border: 1px solid var(--line); border-radius: 8px; background: var(--card); overflow: hidden; padding: 6px; }
.figs img { width: 100%; border-radius: 6px; display: block; }
.figs figcaption { font-size: 12px; color: var(--muted); text-align: center; padding: 6px 4px 2px; }
.links { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
.links a { display: inline-flex; align-items: center; gap: 4px; padding: 4px 10px; border-radius: 6px; font-size: 12.5px; font-weight: 500; text-decoration: none; color: var(--accent); background: var(--card); border: 1px solid var(--line); transition: all 0.15s ease; }
.links a:hover { background: var(--accent-soft); border-color: #bfdbfe; }
.roast-back-btn { text-decoration: none; display: inline-flex; align-items: center; gap: 6px; font-weight: 600; color: var(--accent); background: #eff6ff; padding: 5px 12px; border-radius: 6px; border: 1px solid #bfdbfe; font-size: 13px; cursor: pointer; margin-right: 12px; }
.roast-back-btn:hover { background: #dbeafe; }
.small { font-size: 12px; color: var(--muted); margin-top: 6px; }
.sig td:first-child { font-weight: 600; }
.tabs-container { display: flex; gap: 8px; margin: 12px 0 16px 0; border-bottom: 1px solid var(--line); padding-bottom: 8px; flex-wrap: wrap; }
.rebin-tab-btn { background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 6px 14px; font-size: 13px; font-weight: 600; cursor: pointer; color: var(--fg); transition: all 0.15s ease; }
.rebin-tab-btn:hover { background: var(--accent-soft); color: var(--accent); border-color: #bfdbfe; }
.active-tab { background: var(--accent) !important; color: #fff !important; border-color: var(--accent) !important; }
.rebin-row { cursor: pointer; transition: background-color 0.15s ease; }
.rebin-row:hover { background-color: var(--accent-soft) !important; }
.active-row { background-color: var(--accent-soft) !important; font-weight: 500; }
.active-row td:first-child { border-left: 3px solid var(--accent); }
.rebin-detail-pane { display: flex; flex-direction: column; gap: 18px; }
"""


def render_header(title: str, stat_dir: str, blind: bool, label: str | None = None) -> str:
    e = html.escape
    return (
        f"<header><div style='display:flex;align-items:center;gap:12px;'>"
        f"<a href='#' onclick='goBackToRoastIndex(); return false;' class='roast-back-btn'>&larr; Roast Index</a>"
        f"<h1>{e(title)}</h1></div>"
        f"<div class='meta'>{e(label) + ' &middot; ' if label else ''}<code>{e(stat_dir)}</code> &middot; "
        + (f"<span class='badge blind'>blinded</span> data_obs = sum of backgrounds (Asimov); \"observed\" numbers are not data"
           if blind else "<span class='badge ok'>unblinded</span>")
        + "</div></header>"
    )


def render_overview_card(chans: list[dict], blind: bool = True, title: str = "Overview") -> str:
    e = html.escape
    h = [
        f"<div class='card'><h2>{e(title)}</h2>",
        "<div class='wrap-table'><table><tr><th class='l'>channel</th><th class='l'>signal</th><th class='l'>variable</th>"
        "<th>exp. significance</th>" + ("" if blind else "<th>obs. significance</th>") +
        "<th>exp. 95% CL limit (median)</th><th>&plusmn;1&sigma;</th><th>&plusmn;2&sigma;</th>" + ("" if blind else "<th>obs. limit</th>") +
        "<th>bkg yield</th><th>signal yield</th><th>S/&radic;B</th><th>likelihood scan (68% CL)</th><th>nuisances</th></tr>"
    ]
    for c in chans:
        med, one, two = flim(c.get("limits"))
        sig = c.get("significance") or {}
        totals = c.get("totals", {})
        signal_name = c.get("signal", "")
        bkg = sum(v for p, v in totals.items() if p != signal_name)
        s = totals.get(signal_name, float("nan"))
        soverb = s / math.sqrt(bkg) if bkg and not math.isnan(s) else float("nan")
        sc = c.get("scan")
        h.append(
            f"<tr class='sig'><td>{e(c.get('channel', ''))}</td><td class='l'><code>{e(signal_name)}</code></td><td class='l'>{e(c.get('variable') or '-')}</td>"
            f"<td>{fsig(sig.get('expected'))}</td>" + ("" if blind else f"<td>{fsig(sig.get('observed'))}</td>") +
            f"<td><b>{med}</b></td><td>{one}</td><td>{two}</td>" + ("" if blind else f"<td>{fnum((c.get('limits') or {}).get('obs'), 2)}</td>") +
            f"<td>{fnum(bkg)}</td><td>{fnum(s, 2)}</td><td>{fnum(soverb, 3)}</td><td>{fscan_decomp(sc, c.get('scan_json'))}</td><td>{c.get('nuisances', 0)}</td></tr>"
        )
    h.append("</table></div>")
    h.append("<p class='small'>Limits: AsymptoticLimits on the signal strength r of the listed signal, other signals fixed to 0. "
             "Significance: <code>-M Significance</code>, expected from the Asimov dataset with r = 1. Yields: datacard rates summed over years "
             "(the fit variable's histogram in the SR). Likelihood scan: best-fit r with 68% CL (&plusmn;1&sigma;) interval.</p>")
    h.append("</div>")
    return "\n".join(h)


def render_channel_card(c: dict, blind: bool = True, header_prefix: str = "") -> str:
    e = html.escape
    ch_name = c.get("channel", "")
    sig_name = c.get("signal", "")
    h = [f"<div class='card'><h2>{e(header_prefix)}{e(ch_name)} &mdash; <code>{e(sig_name)}</code></h2>"]

    years = c.get("years", {})
    procs = c.get("processes", [])
    if years:
        h.append("<h3>Datacard yields per year</h3><div class='wrap-table'><table><tr><th class='l'>year</th><th>observation</th>"
                 + "".join(f"<th>{e(short_proc(p))}</th>" for p in procs) + "</tr>")
        for year, card in years.items():
            h.append(f"<tr><td>{e(year)}</td><td>{fnum(card.get('observation'))}</td>"
                     + "".join(f"<td>{fnum(card.get('rates', {}).get(p), 2)}</td>" for p in procs) + "</tr>")
        totals = c.get("totals", {})
        h.append(f"<tr><th class='l'>all</th><th>{fnum(c.get('observation'))}</th>"
                 + "".join(f"<th>{fnum(totals.get(p), 2)}</th>" for p in procs) + "</tr></table></div>")

    lim = c.get("limits")
    if lim:
        h.append("<h3>Limits and significance</h3><div class='wrap-table'><table><tr><th class='l'>quantity</th><th>value</th></tr>")
        for k, lab in (("exp-2", "expected −2σ"), ("exp-1", "expected −1σ"), ("exp0", "expected median"),
                       ("exp+1", "expected +1σ"), ("exp+2", "expected +2σ"), ("obs", "observed" + (" (Asimov)" if blind else ""))):
            if k in lim:
                h.append(f"<tr><td>95% CL limit, {e(lab)}</td><td>{fnum(lim[k], 3)}</td></tr>")
        sig = c.get("significance") or {}
        h.append(f"<tr><td>expected significance</td><td>{fsig(sig.get('expected'))}</td></tr>")
        h.append(f"<tr><td>observed significance{' (Asimov)' if blind else ''}</td><td>{fsig(sig.get('observed'))}</td></tr>")
        if c.get("scan"):
            h.append(f"<tr><td>likelihood scan (best fit r, 68% CL)</td><td><b>{e(fscan(c['scan']))}</b></td></tr>")
        h.append("</table></div>")

    sjson = c.get("scan_json", {})
    decomp = sjson.get("decomposition", {}) if isinstance(sjson, dict) else {}
    if decomp:
        h.append("<h3>Likelihood fit &amp; uncertainty decomposition</h3><div class='wrap-table'><table><tr><th class='l'>fit component</th><th>best-fit r</th><th>68% CL interval</th><th>isolated uncertainty</th></tr>")
        if "total" in decomp:
            tot = decomp["total"]
            h.append(f"<tr><td class='l'><b>Total (all systematics)</b></td><td><b>{tot.get('best', 1.0):.2f}</b></td><td><b>+{tot['hi']:.3f} / −{tot['lo']:.3f}</b></td><td>&plusmn;{(tot['hi']+tot['lo'])/2.0:.3f} (total)</td></tr>")
        if "bkg" in decomp:
            bkg = decomp["bkg"]
            h.append(f"<tr><td class='l'>Bkg. syst. + stat</td><td>-</td><td>+{bkg['raw_hi']:.3f} / −{bkg['raw_lo']:.3f}</td><td><b>&plusmn;{bkg['avg']:.3f} (background)</b></td></tr>")
        if "sig" in decomp:
            sig = decomp["sig"]
            h.append(f"<tr><td class='l'>Sig. syst. + stat</td><td>-</td><td>+{sig['raw_hi']:.3f} / −{sig['raw_lo']:.3f}</td><td><b>&plusmn;{sig['avg']:.3f} (signal)</b></td></tr>")
        if "stat" in decomp:
            st = decomp["stat"]
            h.append(f"<tr><td class='l'>Stat. only</td><td>-</td><td>+{st['hi']:.3f} / −{st['lo']:.3f}</td><td><b>&plusmn;{st['avg']:.3f} (statistical)</b></td></tr>")
        h.append("</table></div>")

    if c.get("impacts"):
        top_imps = c["impacts"][:15]
        h.append("<h3>Top systematic impacts on &Delta;r</h3><div class='wrap-table'><table><tr><th class='l'>nuisance parameter</th><th>impact on r</th><th>type</th></tr>")
        for imp in top_imps:
            h.append(f"<tr><td><code>{e(imp.get('name', ''))}</code></td><td><b>{imp.get('impact', 0.0):+.4f}</b></td><td>{e(imp.get('type', ''))}</td></tr>")
        h.append("</table></div>")

    figs = c.get("figures", {})
    if figs:
        h.append("<div class='figs'>")
        for cap, src in figs.items():
            h.append(f"<figure><a href='{e(src)}' target='_blank'><img src='{e(src)}' alt='{e(cap)}' loading='lazy'></a><figcaption>{e(cap)}</figcaption></figure>")
        h.append("</div>")

    links = c.get("links", {})
    if links:
        h.append("<div class='links'>" + " ".join(f"<a href='{e(p)}' target='_blank'>{e(lab)} ↗</a>" for lab, p in links.items()) + "</div>")

    h.append("</div>")
    return "\n".join(h)


def render_roast_script() -> str:
    return """<script>
function goBackToRoastIndex() {
  if (document.referrer && (document.referrer.includes('index.html') || document.referrer.endsWith('/'))) {
    window.location.href = document.referrer;
    return;
  }
  const href = window.location.href.split('?')[0].split('#')[0];
  if (href.includes('/output/')) {
    window.location.href = href.split('/output/')[0] + '/index.html';
    return;
  }
  const m = href.match(new RegExp('(.*\\/[^\\/]+_\\d{8}_[0-9a-f]{7}-[0-9a-f]{7})\\/'));
  if (m) {
    window.location.href = m[1] + '/index.html';
    return;
  }
  window.location.href = '../index.html';
}
</script>"""


def html_page(title: str, stat_dir: str, chans: list[dict], blind: bool, label: str | None) -> str:
    e = html.escape
    h = [
        f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>{e(title)}</title><style>{CSS}</style></head><body>",
        "<div class='container'>",
        render_header(title, stat_dir, blind, label),
        render_overview_card(chans, blind),
    ]
    for c in chans:
        h.append(render_channel_card(c, blind))
    h.append(render_roast_script())
    h.append("</div></body></html>")
    return "\n".join(h)


def md_page(title: str, chans: list[dict], blind: bool) -> str:
    out = [f"# {title}", "", "blinded (Asimov data_obs)" if blind else "unblinded", "",
           "| channel | signal | exp. significance | exp. limit median | ±1σ | ±2σ | bkg yield | signal yield | likelihood scan (68% CL) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for c in chans:
        med, one, two = flim(c["limits"])
        sig = c["significance"] or {}
        bkg = sum(v for p, v in c["totals"].items() if p != c["signal"])
        out.append(f"| {c['channel']} | `{c['signal']}` | {fsig(sig.get('expected'))} | {med} | {one} | {two} | "
                   f"{fnum(bkg)} | {fnum(c['totals'].get(c['signal'], float('nan')), 2)} | {fscan(c['scan'])} |")
    if any(c.get("impacts") for c in chans):
        out.append("\n## Top systematic impacts\n")
        for c in chans:
            if c.get("impacts"):
                out.append(f"### {c['channel']} (`{c['signal']}`)\n")
                out.append("| nuisance parameter | impact on r | type |")
                out.append("|---|---|---|")
                for imp in c["impacts"][:10]:
                    out.append(f"| `{imp['name']}` | **{imp['impact']:+.4f}** | {imp['type']} |")
                out.append("")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stat_dir", help="stat_analysis directory (contains one sub-directory per channel)")
    ap.add_argument("-o", "--output", required=True, help="output HTML (links are relative to its directory)")
    ap.add_argument("--md", default=None, help="also write a Markdown summary table")
    ap.add_argument("--channel", action="append", default=[], metavar="CHANNEL=SIGNALLABEL",
                    help="channel and its signal label (repeat); default: every <stat_dir>/*/limits/datacard_limits__*.json")
    ap.add_argument("--variable", action="append", default=[], metavar="CHANNEL=VARIABLE", help="fit variable per channel (display only)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--label", default=None, help="production label shown in the header")
    ap.add_argument("--blind", action="store_true", help="the cards were made with --blind (data_obs = Asimov)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")

    chans = OrderedDict()
    for spec in args.channel:
        ch, _, sig = spec.partition("=")
        chans[ch] = sig
    if not chans:
        chans = discover_channels(args.stat_dir)
    if not chans:
        raise SystemExit(f"no channels found under {args.stat_dir} (no */limits/datacard_limits__*.json) and none given with --channel")
    variables = dict(spec.partition("=")[::2] for spec in args.variable)

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    # links are made relative to stat_dir; if the page lives elsewhere, prefix the relative path
    prefix = os.path.relpath(os.path.abspath(args.stat_dir), out_dir)
    infos = []
    for ch, sig in chans.items():
        info = collect_channel(args.stat_dir, ch, sig, variables.get(ch))
        if prefix not in (".", ""):
            info["links"] = OrderedDict((k, os.path.join(prefix, v)) for k, v in info["links"].items())
            info["figures"] = OrderedDict((k, os.path.join(prefix, v)) for k, v in info["figures"].items())
        infos.append(info)

    title = args.title or f"Statistical analysis summary{' — ' + args.label if args.label else ''}"
    with open(args.output, "w") as f:
        f.write(html_page(title, args.stat_dir, infos, args.blind, args.label))
    if args.md:
        with open(args.md, "w") as f:
            f.write(md_page(title, infos, args.blind))
    for c in infos:
        med, one, _ = flim(c["limits"])
        sig = (c["significance"] or {}).get("expected")
        print(f"{c['channel']:8s} {c['signal']:34s} exp.sig={fsig(sig):>6s}  exp.limit={med:>6s} {one:18s} scan={fscan(c['scan'])}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
