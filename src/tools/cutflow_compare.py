#!/usr/bin/env python
"""Compare two cutflow dumps (``dumpCutFlow.py`` yml or ``.coffea``) dataset by dataset.

Meant for cross-phase consistency checks of a production: e.g. the Phase F.1 main
analysis pass against the Phase C.4 FvT-closure pass, which run the same processor on
the same data with the same JCM x FvT weights and must therefore give identical
``counts3``/``counts4`` (weighted) and ``*_unit`` (raw) cutflows for every dataset
they share. Datasets or cuts present in only one dump are reported, not failed.

Usage::

    python src/tools/cutflow_compare.py A.yml B.yml -o compare.html [--txt compare.txt]
        [--label-a "F.1"] [--label-b "C.4"] [--title T] [--tolerance 1e-3]
        [--sections counts3 counts4 counts3_unit counts4_unit] [--cuts SR SB ...]
        [--ignore 'data*:counts4*' ...] [--strict]

``--ignore`` takes ``<dataset glob>:<section glob>`` patterns for known, accepted
differences (e.g. blinded 4b data). The first line of the text output is the verdict
(``PASS``/``FAIL``); the exit code is 0 unless ``--strict`` and the verdict is FAIL.
"""
from __future__ import annotations

import argparse
import fnmatch
import html
import logging
import math
import os
import sys
from collections import OrderedDict, defaultdict

if __name__ == "__main__" and __package__ is None:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.tools.cutflow_closure import load_counts, year_sort_key  # noqa: E402
from src.tools.cutflow_table import parse_dataset_name  # noqa: E402

logger = logging.getLogger("cutflow_compare")

SECTIONS = ["counts4", "counts3", "counts4_unit", "counts3_unit"]
SECTION_LABEL = {
    "counts4": "4b, weighted", "counts3": "3b, weighted",
    "counts4_unit": "4b, raw", "counts3_unit": "3b, raw",
}


# --------------------------------------------------------------------------- comparison
def rel_diff(a: float, b: float) -> float:
    m = max(abs(a), abs(b))
    return abs(a - b) / m if m > 0 else 0.0


def compare(A: dict, B: dict, sections: list, cuts: list | None, tol: float, ignore: list):
    """Return (rows, only_a, only_b, ignored) where rows = list of dicts per
    (section, dataset, cut) with a, b, rel, ok."""
    rows, only_a, only_b, ignored = [], [], [], []
    for sec in sections:
        da, db = A.get(sec) or {}, B.get(sec) or {}
        for ds in sorted(set(da) | set(db), key=dataset_sort_key):
            if ds not in db:
                if da[ds]:
                    only_a.append((sec, ds))
                continue
            if ds not in da:
                if db[ds]:
                    only_b.append((sec, ds))
                continue
            skip = any(fnmatch.fnmatch(ds, dp) and fnmatch.fnmatch(sec, sp) for dp, sp in ignore)
            ca, cb = da[ds] or {}, db[ds] or {}
            for cut in (cuts or [c for c in ca if c in cb]):
                if cut not in ca or cut not in cb:
                    continue
                a, b = float(ca[cut]), float(cb[cut])
                rel = rel_diff(a, b)
                ok = rel <= tol
                if skip and not ok:
                    ignored.append({"section": sec, "dataset": ds, "cut": cut, "a": a, "b": b, "rel": rel})
                    continue
                rows.append({"section": sec, "dataset": ds, "cut": cut, "a": a, "b": b, "rel": rel, "ok": ok})
    return rows, only_a, only_b, ignored


def dataset_sort_key(ds: str):
    process, year, sub = parse_dataset_name(ds)
    return (process, year_sort_key(year), sub)


def aggregate(rows: list) -> "OrderedDict[tuple, dict]":
    """Sum over datasets: {(section, process, cut): {a, b, n, nbad}}."""
    agg = OrderedDict()
    for r in rows:
        process, _, _ = parse_dataset_name(r["dataset"])
        key = (r["section"], process, r["cut"])
        cell = agg.setdefault(key, {"a": 0.0, "b": 0.0, "n": 0, "nbad": 0})
        cell["a"] += r["a"]
        cell["b"] += r["b"]
        cell["n"] += 1
        cell["nbad"] += 0 if r["ok"] else 1
    return agg


# --------------------------------------------------------------------------- formatting
def fnum(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.1f}"


def fratio(a: float, b: float) -> str:
    return f"{a / b:.4f}" if b else ("1.0000" if a == 0 else "inf")


def frel(rel: float) -> str:
    return f"{100 * rel:.3g}%" if rel else "0"


def text_report(title, la, lb, rows, only_a, only_b, ignored, tol, files) -> str:
    bad = [r for r in rows if not r["ok"]]
    verdict = "PASS" if not bad else "FAIL"
    out = [verdict, f"# {title}", "",
           f"A = {la}: {files[0]}", f"B = {lb}: {files[1]}",
           f"tolerance: relative {tol:g}; compared {len(rows)} (section, dataset, cut) entries, "
           f"{len(bad)} differ" + (f", {len(ignored)} ignored" if ignored else ""), ""]
    out.append("## totals per process (summed over datasets)")
    out.append(f"{'section':14s} {'process':30s} {'cut':16s} {'A':>14s} {'B':>14s} {'A/B':>8s}  datasets")
    for (sec, proc, cut), c in aggregate(rows).items():
        flag = "" if c["nbad"] == 0 else f"  <-- {c['nbad']} differ"
        out.append(f"{sec:14s} {proc:30s} {cut:16s} {fnum(c['a']):>14s} {fnum(c['b']):>14s} {fratio(c['a'], c['b']):>8s}  {c['n']}{flag}")
    out.append("")
    if bad:
        out.append(f"## differences ({len(bad)})")
        out.append(f"{'section':14s} {'dataset':30s} {'cut':16s} {'A':>14s} {'B':>14s} {'A/B':>8s} {'rel.diff':>9s}")
        for r in bad:
            out.append(f"{r['section']:14s} {r['dataset']:30s} {r['cut']:16s} {fnum(r['a']):>14s} {fnum(r['b']):>14s} "
                       f"{fratio(r['a'], r['b']):>8s} {frel(r['rel']):>9s}")
        out.append("")
    if ignored:
        out.append(f"## ignored differences ({len(ignored)}, --ignore)")
        for r in ignored:
            out.append(f"{r['section']:14s} {r['dataset']:30s} {r['cut']:16s} {fnum(r['a']):>14s} {fnum(r['b']):>14s} {frel(r['rel']):>9s}")
        out.append("")
    for name, lst in ((f"only in A ({la})", only_a), (f"only in B ({lb})", only_b)):
        if lst:
            out.append(f"## datasets {name}: {len(lst)}")
            by_sec = defaultdict(list)
            for sec, ds in lst:
                by_sec[sec].append(ds)
            for sec, dss in by_sec.items():
                out.append(f"  {sec}: {', '.join(dss)}")
            out.append("")
    return "\n".join(out) + "\n"


CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:1.5rem;color:#222;background:#fff}
h1{font-size:1.3rem;margin:0 0 .3rem} h2{font-size:1.05rem;margin:1.4rem 0 .4rem}
.meta{color:#555;font-size:.9rem;margin-bottom:.8rem} .meta code{background:#f3f3f3;padding:0 .25em}
.verdict{display:inline-block;padding:.25rem .7rem;border-radius:.3rem;font-weight:700;color:#fff}
.pass{background:#2e7d32} .fail{background:#c62828}
table{border-collapse:collapse;font-size:.85rem;font-variant-numeric:tabular-nums}
th,td{border:1px solid #ddd;padding:.2rem .5rem;text-align:right} th{background:#f5f5f5;position:sticky;top:0}
td:nth-child(1),td:nth-child(2),td:nth-child(3),th:nth-child(1),th:nth-child(2),th:nth-child(3){text-align:left}
tr.bad td{background:#fdecea} tr.ign td{background:#fff8e1;color:#666}
details{margin-top:1rem} summary{cursor:pointer;font-weight:600}
.small{font-size:.85rem;color:#555}
"""


def html_report(title, la, lb, rows, only_a, only_b, ignored, tol, files) -> str:
    bad = [r for r in rows if not r["ok"]]
    verdict = "PASS" if not bad else "FAIL"
    e = html.escape
    h = [f"<!doctype html><html><head><meta charset='utf-8'><title>{e(title)}</title><style>{CSS}</style></head><body>",
         f"<h1>{e(title)}</h1>",
         f"<div class='meta'>A = <b>{e(la)}</b> <code>{e(files[0])}</code><br>B = <b>{e(lb)}</b> <code>{e(files[1])}</code><br>"
         f"relative tolerance {tol:g}; {len(rows)} (section, dataset, cut) entries compared, {len(bad)} differ"
         + (f", {len(ignored)} ignored" if ignored else "") + "</div>",
         f"<span class='verdict {'pass' if verdict == 'PASS' else 'fail'}'>{verdict}</span>"]

    h.append("<h2>Totals per process (summed over datasets)</h2>")
    h.append("<table><tr><th>section</th><th>process</th><th>cut</th><th>A</th><th>B</th><th>A / B</th><th>datasets</th><th>differ</th></tr>")
    for (sec, proc, cut), c in aggregate(rows).items():
        cls = " class='bad'" if c["nbad"] else ""
        h.append(f"<tr{cls}><td>{e(SECTION_LABEL.get(sec, sec))}</td><td>{e(proc)}</td><td>{e(cut)}</td><td>{fnum(c['a'])}</td>"
                 f"<td>{fnum(c['b'])}</td><td>{fratio(c['a'], c['b'])}</td><td>{c['n']}</td><td>{c['nbad'] or ''}</td></tr>")
    h.append("</table>")

    if bad:
        h.append(f"<h2>Differences ({len(bad)})</h2>")
        h.append("<table><tr><th>section</th><th>dataset</th><th>cut</th><th>A</th><th>B</th><th>A / B</th><th>rel. diff</th></tr>")
        for r in bad:
            h.append(f"<tr class='bad'><td>{e(SECTION_LABEL.get(r['section'], r['section']))}</td><td>{e(r['dataset'])}</td><td>{e(r['cut'])}</td>"
                     f"<td>{fnum(r['a'])}</td><td>{fnum(r['b'])}</td><td>{fratio(r['a'], r['b'])}</td><td>{frel(r['rel'])}</td></tr>")
        h.append("</table>")
    if ignored:
        h.append(f"<h2>Ignored differences ({len(ignored)}, <code>--ignore</code>)</h2>")
        h.append("<table><tr><th>section</th><th>dataset</th><th>cut</th><th>A</th><th>B</th><th>rel. diff</th></tr>")
        for r in ignored:
            h.append(f"<tr class='ign'><td>{e(SECTION_LABEL.get(r['section'], r['section']))}</td><td>{e(r['dataset'])}</td><td>{e(r['cut'])}</td>"
                     f"<td>{fnum(r['a'])}</td><td>{fnum(r['b'])}</td><td>{frel(r['rel'])}</td></tr>")
        h.append("</table>")
    for name, lst in ((f"only in A ({la})", only_a), (f"only in B ({lb})", only_b)):
        if lst:
            by_sec = defaultdict(list)
            for sec, ds in lst:
                by_sec[sec].append(ds)
            h.append(f"<h2>Datasets {e(name)}: {len(lst)}</h2><div class='small'>")
            h += [f"<b>{e(sec)}</b>: {e(', '.join(dss))}<br>" for sec, dss in by_sec.items()]
            h.append("</div>")

    h.append(f"<details><summary>All compared entries ({len(rows)})</summary>")
    h.append("<table><tr><th>section</th><th>dataset</th><th>cut</th><th>A</th><th>B</th><th>A / B</th><th>rel. diff</th></tr>")
    for r in rows:
        cls = "" if r["ok"] else " class='bad'"
        h.append(f"<tr{cls}><td>{e(SECTION_LABEL.get(r['section'], r['section']))}</td><td>{e(r['dataset'])}</td><td>{e(r['cut'])}</td>"
                 f"<td>{fnum(r['a'])}</td><td>{fnum(r['b'])}</td><td>{fratio(r['a'], r['b'])}</td><td>{frel(r['rel'])}</td></tr>")
    h.append("</table></details></body></html>")
    return "\n".join(h)


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="cutflow dump A (yml or .coffea)")
    ap.add_argument("b", help="cutflow dump B (yml or .coffea)")
    ap.add_argument("-o", "--output", required=True, help="output HTML file")
    ap.add_argument("--txt", default=None, help="also write a plain-text report (first line: PASS/FAIL)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--tolerance", type=float, default=1e-3, help="max relative difference (default 1e-3)")
    ap.add_argument("--sections", nargs="+", default=SECTIONS, help=f"blocks to compare (default: {' '.join(SECTIONS)})")
    ap.add_argument("--cuts", nargs="+", default=None, help="cuts to compare (default: every cut present in both)")
    ap.add_argument("--ignore", nargs="*", default=[], metavar="DATASET_GLOB:SECTION_GLOB",
                    help="accepted differences, e.g. 'data*:counts4*' for blinded 4b data")
    ap.add_argument("--strict", action="store_true", help="exit 1 on FAIL")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")

    A, B = load_counts(args.a), load_counts(args.b)
    ignore = []
    for pat in args.ignore:
        dp, _, sp = pat.partition(":")
        ignore.append((dp or "*", sp or "*"))
    rows, only_a, only_b, ignored = compare(A, B, args.sections, args.cuts, args.tolerance, ignore)
    title = args.title or f"cutflow comparison: {args.label_a} vs {args.label_b}"
    files = (args.a, args.b)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html_report(title, args.label_a, args.label_b, rows, only_a, only_b, ignored, args.tolerance, files))
    text = text_report(title, args.label_a, args.label_b, rows, only_a, only_b, ignored, args.tolerance, files)
    if args.txt:
        with open(args.txt, "w") as f:
            f.write(text)
    verdict = text.splitlines()[0]
    nbad = sum(1 for r in rows if not r["ok"])
    print(f"{verdict}: {len(rows)} entries compared, {nbad} differ, {len(ignored)} ignored; "
          f"{len(only_a)} only in A, {len(only_b)} only in B -> {args.output}")
    if args.strict and verdict != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
