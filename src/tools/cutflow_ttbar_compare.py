#!/usr/bin/env python
"""ttbar MC vs a data-driven ttbar estimate, cut by cut: cuts as rows.

Reads the same cutflow dump as ``cutflow_closure.py`` (``counts3`` / ``counts4``
[+ ``*_unit``] per ``<process>_<year><era>`` dataset, or a ``.coffea`` file) and
writes a self-contained HTML page (plus an optional plain-text version) with

    cut | tt 4b MC | tt 4b estimate | estimate / MC

summed over all years and per year.  A "detailed" toggle adds the MC
components and the same comparison for 3b.  The default estimate is
``TTbar_from_d3`` (processor_HH4b ``plot_ttbar_with_weights``: 3b data weighted
by the FvT d3_to_t4 / d3_to_t3, filled into the 4b / 3b cutflows).

The estimate / MC error is approximate: it propagates the raw (unit-weight)
entry counts of both sides, ignoring the spread of the weights.

Usage:
    python src/tools/cutflow_ttbar_compare.py output/.../cutflow_FvT_closure.yml \
        -o output/.../cutflow_ttbar_MC_vs_d3.html [--txt ...txt] [--title T] \
        [--mc TTToHadronic TTToSemiLeptonic TTTo2L2Nu] [--estimate TTbar_from_d3] \
        [--cuts passPreSel passDiJetMass SB SR]
"""

from __future__ import annotations

import argparse
import html
import logging
import math
import os
import sys
from collections import OrderedDict, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.getcwd(), os.path.abspath(os.path.join(_HERE, "..", ".."))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.tools.cutflow_table import parse_dataset_name  # noqa: E402
from src.tools.cutflow_closure import (  # noqa: E402
    DEFAULT_TTBAR, TT_SHORT, load_counts, cut_order, year_sort_key, cut_label, fnum, text_table,
)
from src.tools.cutflow_closure import fraw as _fraw  # noqa: E402

logger = logging.getLogger("cutflow_ttbar_compare")


def fraw(v: float) -> str:
    """raw entry count in parentheses; nothing for a cut the source is not filled at"""
    return "" if math.isnan(v) else _fraw(v)

DEFAULT_ESTIMATE = "TTbar_from_d3"


# --------------------------------------------------------------------------- aggregation
def aggregate(counts: dict, mc: list, estimate: str, cuts: list):
    """Return (years, table) with table[year][cut] = {"mc3": {comp: v}, "mc4": ..., "est3": v, "est4": v,
    and the *_raw unit-weight counts}."""
    years = OrderedDict()
    table = defaultdict(lambda: defaultdict(lambda: {
        "mc3": defaultdict(float), "mc4": defaultdict(float),
        "mc3_raw": defaultdict(float), "mc4_raw": defaultdict(float),
        "est3": 0.0, "est4": 0.0, "est3_raw": 0.0, "est4_raw": 0.0, "est3_seen": False, "est4_seen": False}))
    for key, block in (("counts3", "3"), ("counts4", "4")):
        for ds, per_cut in counts[key].items():
            if not per_cut:
                continue
            process, year, _ = parse_dataset_name(ds)
            if process not in mc and process != estimate:
                continue
            years.setdefault(year, None)
            unit = counts.get(f"counts{block}_unit", {}).get(ds) or {}
            for cut in cuts:
                if cut not in per_cut:
                    continue
                v = float(per_cut[cut])
                raw = float(unit.get(cut, 0.0))
                for y in (year, "all"):
                    cell = table[y][cut]
                    if process == estimate:
                        cell[f"est{block}"] += v
                        cell[f"est{block}_seen"] = True
                        cell[f"est{block}_raw"] += raw
                    else:
                        cell[f"mc{block}"][process] += v
                        cell[f"mc{block}_raw"][process] += raw
    return sorted(years, key=year_sort_key), table


def compare(cell: dict, block: str, mc: list) -> dict:
    """MC total, estimate and estimate / MC (with the approximate raw-count error) for one tag block."""
    nan = float("nan")
    m = sum(cell[f"mc{block}"].get(p, 0.0) for p in mc)
    m_raw = sum(cell[f"mc{block}_raw"].get(p, 0.0) for p in mc)
    # a cut the estimate is never filled at (e.g. the *_woTrig rows: no trigger weight on data) is
    # "-", not 0 and a ratio of 0
    if not cell[f"est{block}_seen"]:
        return {"mc": m, "mc_raw": m_raw, "est": nan, "est_raw": nan, "ratio": nan, "err": nan}
    e, e_raw = cell[f"est{block}"], cell[f"est{block}_raw"]
    ratio = e / m if m else nan
    err = ratio * math.sqrt(1 / m_raw + 1 / e_raw) if (m and m_raw > 0 and e_raw > 0) else nan
    return {"mc": m, "mc_raw": m_raw, "est": e, "est_raw": e_raw, "ratio": ratio, "err": err}


def is_alltag(cell: dict, mc: list) -> bool:
    """Early cuts (before the tag split) fill both tag cutflows identically and the estimate not at all."""
    m3 = sum(cell["mc3"].get(p, 0.0) for p in mc)
    m4 = sum(cell["mc4"].get(p, 0.0) for p in mc)
    return m3 == m4 and m3 > 0 and not cell["est4"]


# --------------------------------------------------------------------------- formatting
def fratio(r: float, e: float) -> str:
    if r is None or math.isnan(r):
        return "-"
    return f"{r:.3f} ± {e:.3f}" if not math.isnan(e) else f"{r:.3f}"


def headers(mc: list, estimate: str, detailed: bool) -> list:
    h = ["cut", "tt 4b MC"]
    if detailed:
        h += [f"tt 4b MC {TT_SHORT.get(p, p)}" for p in mc]
    h += [f"tt 4b {estimate}", "est / MC 4b"]
    if detailed:
        h += ["tt 3b MC", f"tt 3b {estimate}", "est / MC 3b"]
    return h


def row_values(cut: str, cell: dict, mc: list, estimate: str, detailed: bool) -> list:
    n = len(headers(mc, estimate, detailed)) - 1
    if is_alltag(cell, mc):
        return [f"{cut_label(cut)} (all tags)"] + ["-"] * n
    w = (lambda v, raw: f"{fnum(v)} {fraw(raw)}") if detailed else (lambda v, raw: fnum(v))
    c4 = compare(cell, "4", mc)
    r = [cut_label(cut), w(c4["mc"], c4["mc_raw"])]
    if detailed:
        r += [w(cell["mc4"].get(p, 0.0), cell["mc4_raw"].get(p, 0.0)) for p in mc]
    r += [w(c4["est"], c4["est_raw"]), fratio(c4["ratio"], c4["err"])]
    if detailed:
        c3 = compare(cell, "3", mc)
        r += [w(c3["mc"], c3["mc_raw"]), w(c3["est"], c3["est_raw"]), fratio(c3["ratio"], c3["err"])]
    return r


def render_text(title: str, years: list, table, cuts: list, mc: list, estimate: str) -> str:
    out = [f"# {title}", ""]
    for detailed in (False, True):
        out.append("## " + ("detailed (MC components, 3b)" if detailed else "summary"))
        out.append("")
        hdr = headers(mc, estimate, detailed)
        for y in ["all"] + years:
            rows = [row_values(c, table[y][c], mc, estimate, detailed) for c in cuts if c in table[y]]
            out.append(text_table("all years" if y == "all" else y, hdr, rows))
    return "\n".join(out)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
body{margin:0;padding:16px 20px 60px;background:#f7f7f8;color:#1b1b1f;font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
h1{font-size:18px;margin:0 0 4px} .sub{color:#6b6b76;margin-bottom:14px}
label{color:#6b6b76;margin-right:16px} h2{font-size:15px;margin:22px 0 8px}
.wrap{overflow-x:auto;background:#fff;border:1px solid #e2e2e8;border-radius:8px}
table{border-collapse:collapse;white-space:nowrap;min-width:100%}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid #eee} th:first-child,td:first-child{text-align:left;font-weight:500}
thead th{background:#f0f0f4;position:sticky;top:0;font-weight:600}
tbody tr:hover{background:#fafaff}
.det{display:none;color:#555} body.detailed .det{display:table-cell}
span.raw{display:none;color:#8a8a94;font-size:12px;margin-left:4px} body.showraw span.raw{display:inline}
th.grp{background:#e8eef9} td.ratio{font-weight:600}
td.ok{background:#e6f6ea} td.warn{background:#fff3d6} td.bad{background:#fde2e2}
.note{color:#6b6b76;font-size:12px;margin-top:10px}
</style></head><body>
<h1>__TITLE__</h1><div class="sub">__SUB__</div>
<label><input type="checkbox" id="det" onchange="document.body.classList.toggle('detailed',this.checked)"> show MC components and the 3b comparison</label>
<label><input type="checkbox" id="rawbox" onchange="document.body.classList.toggle('showraw',this.checked)"> show unweighted entry counts in parentheses</label>
__TABLES__
<div class="note">MC counts are weighted (xsec &times; lumi &times; SFs); the estimate is 3b data weighted by JCM &times; the FvT
d3_to_t4 (4b) / d3_to_t3 (3b) probabilities. The optional parentheses are raw entry counts (unit weight): for the estimate, the number of
3b data events. est / MC error from the raw counts of both sides only (weight spread ignored).
Ratio cell shading: |est/MC &minus; 1| &lt; 5% green, &lt; 20% amber, else red.</div>
</body></html>
"""


def ratio_td(c: dict, extra: str = "") -> str:
    r = c["ratio"]
    dev = abs(r - 1) if not math.isnan(r) else float("nan")
    cls = ("ratio " + ("ok" if dev < 0.05 else "warn" if dev < 0.20 else "bad")) if not math.isnan(dev) else "ratio"
    return f'<td class="{cls} {extra}">{fratio(r, c["err"])}</td>'


def html_table(title: str, cuts: list, cells: dict, mc: list, estimate: str) -> str:
    def th(label, cls=""):
        return f'<th class="{cls}">{html.escape(label)}</th>'

    def wr(v, raw, cls=""):
        return f'<td class="{cls}">{fnum(v)}<span class="raw">{fraw(raw)}</span></td>'

    hdr = [th("cut"), th("tt 4b MC", "grp")] + [th(f"tt 4b MC {TT_SHORT.get(p, p)}", "det") for p in mc]
    hdr += [th(f"tt 4b {estimate}", "grp"), th("est / MC 4b", "grp")]
    hdr += [th("tt 3b MC", "det"), th(f"tt 3b {estimate}", "det"), th("est / MC 3b", "det")]
    rows = []
    for c in cuts:
        if c not in cells:
            continue
        cell = cells[c]
        if is_alltag(cell, mc):
            tds = [f"<td>{html.escape(cut_label(c))} <span style='color:#999'>(all tags)</span></td>", "<td>-</td>"]
            tds += ['<td class="det">-</td>'] * len(mc) + ["<td>-</td>"] * 2 + ['<td class="det">-</td>'] * 3
            rows.append("<tr>" + "".join(tds) + "</tr>")
            continue
        c4, c3 = compare(cell, "4", mc), compare(cell, "3", mc)
        tds = [f"<td>{html.escape(cut_label(c))}</td>", wr(c4["mc"], c4["mc_raw"])]
        tds += [wr(cell["mc4"].get(p, 0.0), cell["mc4_raw"].get(p, 0.0), "det") for p in mc]
        tds += [wr(c4["est"], c4["est_raw"]), ratio_td(c4)]
        tds += [wr(c3["mc"], c3["mc_raw"], "det"), wr(c3["est"], c3["est_raw"], "det"), ratio_td(c3, "det")]
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return (f"<h2>{html.escape(title)}</h2><div class=\"wrap\"><table><thead><tr>{''.join(hdr)}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def render_html(title: str, source: str, years: list, table, cuts: list, mc: list, estimate: str) -> str:
    tables = [html_table("all years", cuts, table["all"], mc, estimate)] + [html_table(y, cuts, table[y], mc, estimate) for y in years]
    sub = (f"source: {html.escape(source)} &middot; MC = {html.escape(', '.join(mc))} &middot; estimate = {html.escape(estimate)}"
           f" &middot; years: {html.escape(', '.join(years))}")
    return PAGE.replace("__TITLE__", html.escape(title)).replace("__SUB__", sub).replace("__TABLES__", "\n".join(tables))


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="cutflow yml (counts3/counts4[/…_unit]) or a .coffea file")
    ap.add_argument("-o", "--output", required=True, help="output HTML file")
    ap.add_argument("--txt", default=None, help="also write a plain-text version here")
    ap.add_argument("--title", default=None)
    ap.add_argument("--mc", nargs="+", default=DEFAULT_TTBAR, help="ttbar MC process names")
    ap.add_argument("--estimate", default=DEFAULT_ESTIMATE, help="process name of the data-driven ttbar estimate")
    ap.add_argument("--cuts", nargs="+", default=None, help="cuts (rows) in order; default: order found in the input")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    counts = load_counts(args.input)
    cuts = cut_order(counts, args.cuts)
    years, table = aggregate(counts, args.mc, args.estimate, cuts)
    if not years:
        raise SystemExit(f"no ttbar MC ({', '.join(args.mc)}) or '{args.estimate}' datasets found in {args.input}")
    if not any(table["all"][c]["est4"] for c in cuts):
        logger.warning(f"no '{args.estimate}' counts in {args.input} (was plot_ttbar_with_weights on?)")
    title = args.title or os.path.splitext(os.path.basename(args.input))[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(render_html(title, args.input, years, table, cuts, args.mc, args.estimate))
    logger.info(f"wrote {args.output}: {len(cuts)} cuts x {len(years)} years")
    text = render_text(title, years, table, cuts, args.mc, args.estimate)
    if args.txt:
        with open(args.txt, "w") as f:
            f.write(text)
        logger.info(f"wrote {args.txt}")
    print(text_table("all years", headers(args.mc, args.estimate, False),
                     [row_values(c, table["all"][c], args.mc, args.estimate, False) for c in cuts if c in table["all"]]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
