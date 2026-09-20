#!/usr/bin/env python
"""Background-closure cutflow tables: cuts as rows, background model as columns.

Reads a cutflow dump (``cutflow_*.yml`` as written by ``run-cutflow.sh`` /
``dumpCutFlow.py``: ``counts3``, ``counts4`` and the ``*_unit`` raw counts per
``<process>_<year><era>`` dataset) or a ``.coffea`` file with the
``cutFlowThreeTag`` / ``cutFlowFourTag`` dictionaries, and writes a
self-contained HTML page (plus an optional plain-text version) with

    cut | data 3b | tt 3b | Multijet | tt 4b | Bkg | data 4b | data/Bkg

where Multijet = data 3b - tt 3b and Bkg = Multijet + tt 4b, once summed over
all years and once per year.  A "detailed" toggle (always present in the text
output) breaks the ttbar columns down into their components and adds the
tt 3b / data 3b fraction.  The data/Bkg statistical error uses the raw 4b data
count when the ``*_unit`` counts are present.

Usage:
    python src/tools/cutflow_closure.py output/.../cutflow_wJCM.yml \
        -o output/.../cutflow_wJCM.html [--txt output/.../cutflow_wJCM_table.txt] \
        [--title T] [--data data] [--ttbar TTToHadronic TTToSemiLeptonic TTTo2L2Nu] \
        [--cuts passJetMult passPreSel passDiJetMass SB SR]

Notes: with the pre-JCM (unweighted 3b) cutflow the Multijet column is not
normalised and data/Bkg is ~mu_qcd; with JCM applied it is the closure test.
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

logger = logging.getLogger("cutflow_closure")

DEFAULT_TTBAR = ["TTToHadronic", "TTToSemiLeptonic", "TTTo2L2Nu"]
TT_SHORT = {"TTToHadronic": "had", "TTToSemiLeptonic": "semi", "TTTo2L2Nu": "dilep"}


# --------------------------------------------------------------------------- input
def load_counts(path: str) -> dict:
    """Return {'counts3': {ds: {cut: v}}, 'counts4': ..., 'counts3_unit': ..., 'counts4_unit': ...}."""
    if path.endswith(".coffea"):
        from coffea.util import load
        h = load(path)
        out = {
            "counts3": h["cutFlowThreeTag"],
            "counts4": h["cutFlowFourTag"],
            "counts3_unit": h.get("cutFlowThreeTagUnitWeight", {}),
            "counts4_unit": h.get("cutFlowFourTagUnitWeight", {}),
        }
        return {k: {ds: {c: float(v) for c, v in cuts.items()} for ds, cuts in d.items()} for k, d in out.items()}
    import yaml
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    for k in ("counts3", "counts4"):
        if k not in d:
            raise SystemExit(f"{path}: missing '{k}' block")
    d.setdefault("counts3_unit", {})
    d.setdefault("counts4_unit", {})
    return d


def cut_order(counts: dict, requested: list | None) -> list:
    if requested:
        return requested
    seen = OrderedDict()
    for ds in counts["counts4"].values():
        for c in (ds or {}):
            seen.setdefault(c, None)
    return list(seen)


# --------------------------------------------------------------------------- aggregation
def aggregate(counts: dict, data_name: str, ttbar: list, cuts: list):
    """Return (years, table) with table[year][cut] = dict of sums:
    data3, data4, data4_raw, tt3[comp], tt4[comp]."""
    years = OrderedDict()
    table = defaultdict(lambda: defaultdict(lambda: {
        "data3": 0.0, "data4": 0.0, "data4_raw": 0.0, "data3_raw": 0.0,
        "tt3": defaultdict(float), "tt4": defaultdict(float),
        "tt3_raw": defaultdict(float), "tt4_raw": defaultdict(float)}))
    unknown = set()
    for key, block in (("counts3", "3"), ("counts4", "4")):
        for ds, per_cut in counts[key].items():
            if not per_cut:  # empty block (e.g. a process with no filled cuts)
                continue
            process, year, _ = parse_dataset_name(ds)
            if process == data_name:
                kind = "data"
            elif process in ttbar:
                kind = process
            else:
                unknown.add(process)
                continue
            years.setdefault(year, None)
            unit = counts.get(f"counts{block}_unit", {}).get(ds) or {}
            for cut in cuts:
                if cut not in per_cut:
                    continue
                v = float(per_cut[cut])
                for y in (year, "all"):
                    cell = table[y][cut]
                    if kind == "data":
                        cell[f"data{block}"] += v
                        cell[f"data{block}_raw"] += float(unit.get(cut, v))
                    else:
                        cell[f"tt{block}"][kind] += v
                        cell[f"tt{block}_raw"][kind] += float(unit.get(cut, 0.0))
    if unknown:
        logger.info(f"ignoring processes not in the background model: {sorted(unknown)}")
    return sorted(years, key=year_sort_key), table


def year_sort_key(year: str):
    """Chronological order: UL16_preVFP < UL16_postVFP < UL17 < UL18 < 2022_preEE < 2022_EE < 2023_preBPix < 2023_BPix < 2024."""
    import re
    m = re.match(r"^(?:UL)?(\d{2,4})(?:_(.*))?$", year)
    if not m:
        return (9999, 9, year)
    num = int(m.group(1))
    num = 2000 + num if num < 100 else num
    suffix = (m.group(2) or "").lower()
    sub = 0 if suffix.startswith("pre") else 1 if suffix else 0
    return (num, sub, year)


def derived(cell: dict, ttbar: list) -> dict:
    tt3 = sum(cell["tt3"].get(p, 0.0) for p in ttbar)
    tt4 = sum(cell["tt4"].get(p, 0.0) for p in ttbar)
    nan = float("nan")
    # Early cuts (before the tag split) are filled identically into both tag cutflows
    # ("allTag" fills): no background model there, show the totals only.
    alltag = cell["data3"] == cell["data4"] and tt3 == tt4 and cell["data3"] > 0
    if alltag:
        return {"tt3": tt3, "tt4": nan, "mj": nan, "bkg": nan, "ratio": nan, "err": nan, "tt3frac": nan, "alltag": True}
    mj = cell["data3"] - tt3
    bkg = mj + tt4
    ratio = cell["data4"] / bkg if bkg else nan
    n_raw = cell["data4_raw"]
    err = ratio / math.sqrt(n_raw) if (bkg and n_raw > 0) else nan
    frac = tt3 / cell["data3"] if cell["data3"] else nan
    return {"tt3": tt3, "tt4": tt4, "mj": mj, "bkg": bkg, "ratio": ratio, "err": err, "tt3frac": frac, "alltag": False}


# --------------------------------------------------------------------------- formatting
def fraw(v: float) -> str:
    """Raw (unit-weight) entry count, for display in parentheses."""
    return f"({v:,.0f})"


def tt_raw(cell: dict, block: str, ttbar: list) -> float:
    return sum(cell[f"tt{block}_raw"].get(p, 0.0) for p in ttbar)


def cut_label(cut: str) -> str:
    """Human label for a cut name: `<cut>_woTrig` (filled without the MC trigger weight)
    becomes `<cut> (before trig. weight)`."""
    if cut.endswith("_woTrig"):
        return f"{cut[:-len('_woTrig')]} (before trig. weight)"
    return cut


def fnum(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:,.1f}"


def fratio(r: float, e: float) -> str:
    if r is None or math.isnan(r):
        return "-"
    return f"{r:.3f} ± {e:.3f}" if not math.isnan(e) else f"{r:.3f}"


def ffrac(x: float) -> str:
    return "-" if math.isnan(x) else f"{100 * x:.0f}%"


def headers(ttbar: list, detailed: bool) -> list:
    h = ["cut", "data 3b", "tt 3b"]
    if detailed:
        h += [f"tt 3b {TT_SHORT.get(p, p)}" for p in ttbar] + ["tt 3b / data 3b"]
    h += ["Multijet", "tt 4b"]
    if detailed:
        h += [f"tt 4b {TT_SHORT.get(p, p)}" for p in ttbar]
    h += ["Bkg", "data 4b", "data / Bkg"]
    return h


def row_values(cut: str, cell: dict, ttbar: list, detailed: bool) -> list:
    d = derived(cell, ttbar)
    if d["alltag"]:
        r = [f"{cut_label(cut)} (all tags)", fnum(cell["data3"]), fnum(d["tt3"])]
        if detailed:
            r += [fnum(cell["tt3"].get(p, 0.0)) for p in ttbar] + ["-"]
        r += ["-", "-"]
        if detailed:
            r += ["-"] * len(ttbar)
        return r + ["-", "-", "-"]
    # detailed text view: weighted value followed by the raw entry count in parentheses
    w = (lambda v, raw: f"{fnum(v)} {fraw(raw)}") if detailed else (lambda v, raw: fnum(v))
    r = [cut_label(cut), w(cell["data3"], cell["data3_raw"]), w(d["tt3"], tt_raw(cell, "3", ttbar))]
    if detailed:
        r += [w(cell["tt3"].get(p, 0.0), cell["tt3_raw"].get(p, 0.0)) for p in ttbar] + [ffrac(d["tt3frac"])]
    r += [fnum(d["mj"]), w(d["tt4"], tt_raw(cell, "4", ttbar))]
    if detailed:
        r += [w(cell["tt4"].get(p, 0.0), cell["tt4_raw"].get(p, 0.0)) for p in ttbar]
    r += [fnum(d["bkg"]), w(cell["data4"], cell["data4_raw"]), fratio(d["ratio"], d["err"])]
    return r


def text_table(title: str, hdr: list, rows: list) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(hdr, *rows)]
    fmt = lambda r: "  ".join(str(x).ljust(w) if i == 0 else str(x).rjust(w) for i, (x, w) in enumerate(zip(r, widths)))
    lines = [title, fmt(hdr), "  ".join("-" * w for w in widths)] + [fmt(r) for r in rows]
    return "\n".join(lines) + "\n"


def render_text(title: str, years: list, table, cuts: list, ttbar: list) -> str:
    out = [f"# {title}", ""]
    for detailed in (False, True):
        out.append("## " + ("detailed (ttbar components)" if detailed else "summary"))
        out.append("")
        hdr = headers(ttbar, detailed)
        for y in ["all"] + years:
            rows = [row_values(c, table[y][c], ttbar, detailed) for c in cuts if c in table[y]]
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
<label><input type="checkbox" id="det" onchange="document.body.classList.toggle('detailed',this.checked)"> show ttbar components and tt 3b / data 3b</label>
<label><input type="checkbox" id="rawbox" onchange="document.body.classList.toggle('showraw',this.checked)"> show unweighted entry counts in parentheses</label>
__TABLES__
<div class="note">Counts are weighted (xsec &times; lumi &times; SFs, and JCM on 3b where applied); the optional parentheses are raw entry counts (unit weight).
Multijet = data 3b &minus; tt 3b; Bkg = Multijet + tt 4b; data / Bkg error from the raw 4b data count only.
Ratio cell shading: |data/Bkg &minus; 1| &lt; 5% green, &lt; 20% amber, else red.</div>
</body></html>
"""


def html_table(title: str, cuts: list, cells: dict, ttbar: list) -> str:
    def th(label, cls=""):
        return f'<th class="{cls}">{html.escape(label)}</th>'

    def wr(v, raw, cls=""):
        """weighted value with the raw entry count in a toggleable span"""
        return f'<td class="{cls}">{fnum(v)}<span class="raw">{fraw(raw)}</span></td>'
    hdr = [th("cut"), th("data 3b"), th("tt 3b")]
    hdr += [th(f"tt 3b {TT_SHORT.get(p, p)}", "det") for p in ttbar] + [th("tt 3b / data 3b", "det")]
    hdr += [th("Multijet"), th("tt 4b")] + [th(f"tt 4b {TT_SHORT.get(p, p)}", "det") for p in ttbar]
    hdr += [th("Bkg", "grp"), th("data 4b", "grp"), th("data / Bkg", "grp")]
    rows = []
    for c in cuts:
        if c not in cells:
            continue
        cell = cells[c]
        d = derived(cell, ttbar)
        if d["alltag"]:
            n_det = 2 * len(ttbar) + 1
            tds = [f"<td>{html.escape(cut_label(c))} <span style='color:#999'>(all tags)</span></td>", wr(cell['data3'], cell['data3_raw']),
                   wr(d['tt3'], tt_raw(cell, '3', ttbar))] + ['<td class="det">-</td>'] * n_det + ["<td>-</td>"] * 5
            rows.append("<tr>" + "".join(tds) + "</tr>")
            continue
        dev = abs(d["ratio"] - 1) if not math.isnan(d["ratio"]) else float("nan")
        cls = "ratio " + ("ok" if dev < 0.05 else "warn" if dev < 0.20 else "bad") if not math.isnan(dev) else "ratio"
        tds = [f"<td>{html.escape(cut_label(c))}</td>", wr(cell['data3'], cell['data3_raw']), wr(d['tt3'], tt_raw(cell, '3', ttbar))]
        tds += [wr(cell["tt3"].get(p, 0.0), cell["tt3_raw"].get(p, 0.0), "det") for p in ttbar] + [f'<td class="det">{ffrac(d["tt3frac"])}</td>']
        tds += [f"<td>{fnum(d['mj'])}</td>", wr(d['tt4'], tt_raw(cell, '4', ttbar))]
        tds += [wr(cell["tt4"].get(p, 0.0), cell["tt4_raw"].get(p, 0.0), "det") for p in ttbar]
        tds += [f"<td>{fnum(d['bkg'])}</td>", wr(cell['data4'], cell['data4_raw']), f'<td class="{cls}">{fratio(d["ratio"], d["err"])}</td>']
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return (f"<h2>{html.escape(title)}</h2><div class=\"wrap\"><table><thead><tr>{''.join(hdr)}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def render_html(title: str, source: str, years: list, table, cuts: list, ttbar: list) -> str:
    tables = [html_table("all years", cuts, table["all"], ttbar)] + [html_table(y, cuts, table[y], ttbar) for y in years]
    sub = f"source: {html.escape(source)} &middot; ttbar = {html.escape(', '.join(ttbar))} &middot; years: {html.escape(', '.join(years))}"
    return (PAGE.replace("__TITLE__", html.escape(title)).replace("__SUB__", sub)
            .replace("__TABLES__", "\n".join(tables)))


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="cutflow yml (counts3/counts4[/…_unit]) or a .coffea file")
    ap.add_argument("-o", "--output", required=True, help="output HTML file")
    ap.add_argument("--txt", default=None, help="also write a plain-text version here")
    ap.add_argument("--title", default=None)
    ap.add_argument("--data", default="data", help="data process name (default: data)")
    ap.add_argument("--ttbar", nargs="+", default=DEFAULT_TTBAR, help="ttbar process names (3b subtraction + 4b component)")
    ap.add_argument("--cuts", nargs="+", default=None, help="cuts (rows) in order; default: order found in the input")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    counts = load_counts(args.input)
    cuts = cut_order(counts, args.cuts)
    years, table = aggregate(counts, args.data, args.ttbar, cuts)
    if not years:
        raise SystemExit(f"no '{args.data}' or ttbar datasets found in {args.input}")
    title = args.title or os.path.splitext(os.path.basename(args.input))[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(render_html(title, args.input, years, table, cuts, args.ttbar))
    logger.info(f"wrote {args.output}: {len(cuts)} cuts x {len(years)} years")
    text = render_text(title, years, table, cuts, args.ttbar)
    if args.txt:
        with open(args.txt, "w") as f:
            f.write(text)
        logger.info(f"wrote {args.txt}")
    # summary table on stdout so it lands in the workflow log
    print(text_table("all years", headers(args.ttbar, False), [row_values(c, table["all"][c], args.ttbar, False) for c in cuts if c in table["all"]]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
