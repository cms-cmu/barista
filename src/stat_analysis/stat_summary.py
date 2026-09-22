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
    """Count valid / failed grid points in a merged MultiDimFit tree (needs uproot)."""
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
        pts = [(float(r), float(d)) for r, d, q in zip(a[pb], a["deltaNLL"], a["quantileExpected"]) if q >= 0]
        good = [(r, d) for r, d in pts if d < 9000 and not math.isnan(d)]
        best = [float(r) for r, q in zip(a[pb], a["quantileExpected"]) if q < 0]
        return {"poi": pb, "total": len(pts), "valid": len(good), "failed": len(pts) - len(good),
                "rmin": min((r for r, _ in good), default=float("nan")),
                "rmax": max((r for r, _ in good), default=float("nan")),
                "bestfit": best[0] if best else float("nan")}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"cannot read scan {path}: {e}")
        return None


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
    for p in sorted(glob.glob(os.path.join(d, "datacards", f"datacard_{channel}_*.txt"))):
        year = os.path.basename(p)[len(f"datacard_{channel}_"):-len(".txt")]
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

    merged = sorted(glob.glob(os.path.join(d, "likelihood_scan", f"higgsCombine_merged_{signal}*.root")))
    info["scan"] = read_scan(merged[0], f"r{signal}") if merged else None

    # links
    for label, *p in (
        ("combined datacard", "datacards", f"datacard__{channel}.txt"),
        ("workspace", "workspace", f"datacard__{signal}.root"),
        ("limits json", "limits", f"datacard_limits__{signal}.json"),
        ("limits log", "limits", f"datacard_limits__{signal}.txt"),
        ("significance log", "significance", f"datacard_significance__{signal}.log"),
        ("likelihood scan pdf", "likelihood_scan", f"datacard_likelihood_scan__{signal}.pdf"),
        ("postfit pdf", "postfit", f"datacard_postfit__{signal}.pdf"),
        ("fitDiagnostics (b-only)", "postfit", f"datacard_fitDiagnostics_bonly__{signal}.root"),
        ("diffNuisances (b-only)", "postfit", f"datacard_diffNuisances_bonly__{signal}.root"),
    ):
        if exists(*p):
            info["links"][label] = rel(*p)
    for year, card in years.items():
        info["links"][f"datacard {year}"] = rel("datacards", f"datacard_{channel}_{year}.txt")

    # figures
    if exists("likelihood_scan", "scan_plot.png"):
        info["figures"]["likelihood scan"] = rel("likelihood_scan", "scan_plot.png")
    for year in years:
        for fit in ("fit_s", "fit_b"):
            if exists("postfit", f"{channel}_{year}_CMS_th1x_{fit}.png"):
                info["figures"][f"postfit {year} ({fit})"] = rel("postfit", f"{channel}_{year}_CMS_th1x_{fit}.png")
                break
    if exists("postfit", "covariance_fit_s.png"):
        info["figures"]["covariance (fit_s)"] = rel("postfit", "covariance_fit_s.png")
    for p in sorted(glob.glob(os.path.join(d, "postfit", "plots", f"postfitplots__{signal}__*.png"))):
        info["figures"][f"postfit {os.path.basename(p).split('__')[-1][:-4]}"] = rel("postfit", "plots", os.path.basename(p))
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
    if not sc:
        return "-"
    return f"{sc['valid']}/{sc['total']} valid" + (f" (r {fnum(sc['rmin'], 2)} … {fnum(sc['rmax'], 2)})" if sc["valid"] else "")


def short_proc(p: str) -> str:
    return p.replace("_13p0TeV_hbbhbb", "").replace("ggHH_kl_", "ggHH kl=").replace("_kt_1", "")


CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:1.5rem;color:#222;background:#fff;max-width:1400px}
h1{font-size:1.4rem;margin:0 0 .3rem} h2{font-size:1.15rem;margin:1.6rem 0 .4rem;border-bottom:1px solid #ddd;padding-bottom:.2rem}
h3{font-size:1rem;margin:1rem 0 .3rem}
.meta{color:#555;font-size:.9rem;margin-bottom:.8rem} code{background:#f3f3f3;padding:0 .25em;font-size:.9em}
table{border-collapse:collapse;font-size:.88rem;font-variant-numeric:tabular-nums;margin:.3rem 0}
th,td{border:1px solid #ddd;padding:.25rem .55rem;text-align:right} th{background:#f5f5f5}
td:first-child,th:first-child{text-align:left} td.l,th.l{text-align:left}
.badge{display:inline-block;padding:.1rem .5rem;border-radius:.3rem;font-size:.8rem;font-weight:600;color:#fff}
.blind{background:#ef6c00} .warn{background:#c62828} .ok{background:#2e7d32}
.figs{display:flex;flex-wrap:wrap;gap:.8rem;margin:.5rem 0} .figs figure{margin:0;width:320px}
.figs img{width:100%;border:1px solid #ddd;background:#fff} .figs figcaption{font-size:.8rem;color:#555;text-align:center}
.links a{margin-right:.9rem;font-size:.88rem;white-space:nowrap} .small{font-size:.85rem;color:#555}
.sig td:first-child{font-weight:600}
"""


def html_page(title: str, stat_dir: str, chans: list[dict], blind: bool, label: str | None) -> str:
    e = html.escape
    h = [f"<!doctype html><html><head><meta charset='utf-8'><title>{e(title)}</title><style>{CSS}</style></head><body>",
         f"<h1>{e(title)}</h1>",
         f"<div class='meta'>{e(label) + ' &middot; ' if label else ''}<code>{e(stat_dir)}</code> &middot; "
         + (f"<span class='badge blind'>blinded</span> data_obs = sum of backgrounds (Asimov); \"observed\" numbers are not data"
            if blind else "unblinded") + "</div>"]

    # ---- overview
    h.append("<h2>Overview</h2>")
    h.append("<table><tr><th class='l'>channel</th><th class='l'>signal</th><th class='l'>variable</th>"
             "<th>exp. significance</th>" + ("" if blind else "<th>obs. significance</th>") +
             "<th>exp. 95% CL limit (median)</th><th>&plusmn;1&sigma;</th><th>&plusmn;2&sigma;</th>" + ("" if blind else "<th>obs. limit</th>") +
             "<th>bkg yield</th><th>signal yield</th><th>S/&radic;B</th><th>likelihood scan</th><th>nuisances</th></tr>")
    for c in chans:
        med, one, two = flim(c["limits"])
        sig = c["significance"] or {}
        bkg = sum(v for p, v in c["totals"].items() if p in ("multijet", "tt") or p.startswith(("multijet", "tt", "TT", "bkg")))
        s = c["totals"].get(c["signal"], float("nan"))
        soverb = s / math.sqrt(bkg) if bkg and not math.isnan(s) else float("nan")
        sc = c["scan"]
        scan_cls = "" if not sc or sc["failed"] == 0 else " class='warn badge'"
        h.append(f"<tr class='sig'><td>{e(c['channel'])}</td><td class='l'><code>{e(c['signal'])}</code></td><td class='l'>{e(c['variable'] or '-')}</td>"
                 f"<td>{fsig(sig.get('expected'))}</td>" + ("" if blind else f"<td>{fsig(sig.get('observed'))}</td>") +
                 f"<td><b>{med}</b></td><td>{one}</td><td>{two}</td>" + ("" if blind else f"<td>{fnum((c['limits'] or {}).get('obs'), 2)}</td>") +
                 f"<td>{fnum(bkg)}</td><td>{fnum(s, 2)}</td><td>{fnum(soverb, 3)}</td><td><span{scan_cls}>{e(fscan(sc))}</span></td><td>{c['nuisances']}</td></tr>")
    h.append("</table>")
    h.append("<p class='small'>Limits: AsymptoticLimits on the signal strength r of the listed signal, other signals fixed to 0. "
             "Significance: <code>-M Significance</code>, expected from the Asimov dataset with r = 1. Yields: datacard rates summed over years "
             "(the fit variable's histogram in the SR). Likelihood scan: grid points with a converged fit; points fail where the total pdf goes "
             "negative (r &lt; 0 with near-empty background bins).</p>")

    # ---- per channel
    for c in chans:
        h.append(f"<h2>{e(c['channel'])} &mdash; <code>{e(c['signal'])}</code></h2>")
        if c["years"]:
            h.append("<h3>Datacard yields per year</h3><table><tr><th class='l'>year</th><th>observation</th>"
                     + "".join(f"<th>{e(short_proc(p))}</th>" for p in c["processes"]) + "</tr>")
            for year, card in c["years"].items():
                h.append(f"<tr><td>{e(year)}</td><td>{fnum(card['observation'])}</td>"
                         + "".join(f"<td>{fnum(card['rates'].get(p), 2)}</td>" for p in c["processes"]) + "</tr>")
            h.append(f"<tr><th class='l'>all</th><th>{fnum(c['observation'])}</th>"
                     + "".join(f"<th>{fnum(c['totals'].get(p), 2)}</th>" for p in c["processes"]) + "</tr></table>")
        if c["limits"]:
            lim = c["limits"]
            h.append("<h3>Limits and significance</h3><table><tr><th class='l'>quantity</th><th>value</th></tr>")
            for k, lab in (("exp-2", "expected −2σ"), ("exp-1", "expected −1σ"), ("exp0", "expected median"),
                           ("exp+1", "expected +1σ"), ("exp+2", "expected +2σ"), ("obs", "observed" + (" (Asimov)" if blind else ""))):
                if k in lim:
                    h.append(f"<tr><td>95% CL limit, {e(lab)}</td><td>{fnum(lim[k], 3)}</td></tr>")
            sig = c["significance"] or {}
            h.append(f"<tr><td>expected significance</td><td>{fsig(sig.get('expected'))}</td></tr>")
            h.append(f"<tr><td>observed significance{' (Asimov)' if blind else ''}</td><td>{fsig(sig.get('observed'))}</td></tr>")
            if c["scan"]:
                sc = c["scan"]
                h.append(f"<tr><td>likelihood scan points</td><td>{e(fscan(sc))}, best fit r = {fnum(sc['bestfit'], 3)}</td></tr>")
            h.append("</table>")
        if c["figures"]:
            h.append("<div class='figs'>")
            for cap, src in c["figures"].items():
                h.append(f"<figure><a href='{e(src)}'><img src='{e(src)}' alt='{e(cap)}' loading='lazy'></a><figcaption>{e(cap)}</figcaption></figure>")
            h.append("</div>")
        if c["links"]:
            h.append("<div class='links'>" + " ".join(f"<a href='{e(p)}'>{e(lab)}</a>" for lab, p in c["links"].items()) + "</div>")
    h.append("</body></html>")
    return "\n".join(h)


def md_page(title: str, chans: list[dict], blind: bool) -> str:
    out = [f"# {title}", "", "blinded (Asimov data_obs)" if blind else "unblinded", "",
           "| channel | signal | exp. significance | exp. limit median | ±1σ | ±2σ | bkg yield | signal yield | scan |",
           "|---|---|---|---|---|---|---|---|---|"]
    for c in chans:
        med, one, two = flim(c["limits"])
        sig = c["significance"] or {}
        bkg = sum(v for p, v in c["totals"].items() if p.startswith(("multijet", "tt", "TT", "bkg")))
        out.append(f"| {c['channel']} | `{c['signal']}` | {fsig(sig.get('expected'))} | {med} | {one} | {two} | "
                   f"{fnum(bkg)} | {fnum(c['totals'].get(c['signal'], float('nan')), 2)} | {fscan(c['scan'])} |")
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
