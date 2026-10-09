"""Per-class yields a classifier training loaded, as a small HTML page + JSON.

The HCR datasets write the counts and weights of every class by region (and by year and region)
to $HCR_YIELDS_FILE at the end of dataset loading, one dataset module at a time (merged) (coffea4bees/classifier/config/dataset/HCR/
_common.py: _log_yields). Trainings logged before that existed only printed the per-region tables
("In region SB:" + a rich table), which this also parses from the training log.

A ratio compares the data with the background model the training starts from, e.g.
    --ratio "d4 / t4 + mix4"          MvD: four-tag data vs ttbar + mixed x JCM
    --ratio "d4 / d3 - t3 + t4"       FvT: four-tag data vs JCM x (3b data - 3b ttbar) + 4b ttbar
Terms are class labels with + or -; a class missing from a region makes that ratio "-".

Usage (the classifier workflow's `yields` rule):
    python3 src/classifier/yields_report.py --yields output/MvD/yields_loaded.json \
        --log output/MvD/train.log --ratio "d4 / t4 + mix4" --title MvD \
        -o output/MvD/yields.html --json output/MvD/yields.json
"""


import argparse
import html
import json
import os
import re
import sys

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)")
_ROW = re.compile(r"^\s*[│┃|]\s*([^│┃|\s]+)\s*[│┃|]\s*(\d+)\s*[│┃|]\s*([-+0-9.eE]+|nan)\s*[│┃|]")
_REGION = re.compile(r"In region (\S+?):")


def parse_log(path: str) -> dict:
    """The per-region tables in a training log -> {"regions": {...}, "years": {}}. Each dataset
    module prints its own set (SvB: the signal, then multijet + ttbar), so the tables are merged by
    region; a class printed twice (a retried training appends to the same log) keeps the last."""
    with open(path, errors="replace") as f:
        text = _ANSI.sub("", f.read().replace("\r", "\n"))
    regions, region = {}, None
    for line in text.splitlines():
        m = _REGION.search(line)
        if m:
            region = m.group(1)
            regions.setdefault(region, {})
            continue
        m = _ROW.match(line)
        if m and region is not None and m.group(1) != "Class":
            regions[region][m.group(1)] = {"count": int(m.group(2)), "weight": float(m.group(3))}
    return {"regions": {r: c for r, c in regions.items() if c}, "years": {}}


def parse_ratio(spec: str) -> tuple:
    """"d4 / t4 + mix4" -> (spec, [("d4")], [(+1, "t4"), (+1, "mix4")])."""
    if "/" not in spec:
        raise ValueError(f"ratio {spec!r}: expected 'numerator / denominator'")
    num, den = spec.split("/", 1)

    def terms(side):
        out = []
        for sign, label in re.findall(r"([+-]?)\s*([A-Za-z_][\w]*)", side):
            out.append((-1 if sign == "-" else 1, label))
        if not out:
            raise ValueError(f"ratio {spec!r}: empty side {side!r}")
        return out

    return spec.strip(), terms(num), terms(den)


def ratio_value(classes: dict, num, den):
    def total(terms):
        if any(label not in classes for _, label in terms):
            return None
        return sum(sign * classes[label]["weight"] for sign, label in terms)

    n, d = total(num), total(den)
    if n is None or d is None or d == 0:
        return n, d, None
    return n, d, n / d


def _table(title: str, regions: dict, ratios) -> str:
    rows = []
    for region in sorted(regions):
        classes = regions[region]
        rows.append(f'<tr><th colspan="3" class="region">{html.escape(region)}</th></tr>')
        for label in sorted(classes):
            c = classes[label]
            rows.append(f"<tr><td>{html.escape(label)}</td><td>{c['count']:,}</td><td>{c['weight']:,.1f}</td></tr>")
        for spec, num, den in ratios:
            n, d, r = ratio_value(classes, num, den)
            val = "-" if r is None else f"{r:.4f}"
            detail = "" if r is None else f" <span class='muted'>({n:,.1f} / {d:,.1f})</span>"
            rows.append(f"<tr class='ratio'><td>{html.escape(spec)}</td><td></td><td><b>{val}</b>{detail}</td></tr>")
    return (f"<h3>{html.escape(title)}</h3><table><tr><th>class</th><th>count</th><th>weight</th></tr>"
            + "".join(rows) + "</table>")


def render(yields: dict, ratios, title: str, source: str) -> str:
    parts = [f"<h1>{html.escape(title)}: loaded yields</h1>",
             f"<p class='muted'>Per-class event counts and summed training weights at the end of "
             f"dataset loading (before the k-fold split), by region. Source: {html.escape(source)}.</p>",
             _table("All years", yields.get("regions", {}), ratios)]
    for year in sorted(yields.get("years", {})):
        parts.append(_table(f"Year {year}", yields["years"][year], ratios))
    style = ("body{font-family:sans-serif;margin:1.5em;max-width:1000px}"
             "table{border-collapse:collapse;margin:0.3em 2em 1.2em 0;display:inline-table;vertical-align:top}"
             "td,th{border:1px solid #ccc;padding:2px 10px;text-align:right}th{background:#eee}"
             "th.region{text-align:left;background:#f6f6f6}td:first-child{text-align:left}"
             "tr.ratio td{background:#fbf8e8}.muted{color:#777;font-size:12px}")
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{html.escape(title)} yields</title>"
            f"<style>{style}</style></head><body>" + "".join(parts) + "</body></html>")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yields", help="JSON written by the training ($HCR_YIELDS_FILE)")
    p.add_argument("--log", help="training log (fallback: parse its per-region tables)")
    p.add_argument("--ratio", action="append", default=[], help="'num / den' of class labels")
    p.add_argument("--title", default="classifier")
    p.add_argument("-o", "--output", required=True, help="HTML page")
    p.add_argument("--json", help="write the yields (+ ratios) as JSON too")
    a = p.parse_args(argv)

    if a.yields and os.path.exists(a.yields):
        with open(a.yields) as f:
            yields, source = json.load(f), os.path.basename(a.yields)
    elif a.log and os.path.exists(a.log):
        yields, source = parse_log(a.log), f"tables in {os.path.basename(a.log)}"
    else:
        sys.exit(f"no yields: neither {a.yields} nor {a.log} exists")
    if not yields.get("regions"):
        print(f"warning: no per-class yields found in {source}", file=sys.stderr)
        yields = {"regions": {}, "years": {}}
    ratios = [parse_ratio(s) for s in a.ratio]

    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    with open(a.output, "w") as f:
        f.write(render(yields, ratios, a.title, source))
    if a.json:
        out = dict(yields)
        out["ratios"] = {
            scope: {region: {spec: ratio_value(cl, num, den)[2] for spec, num, den in ratios}
                    for region, cl in regs.items()}
            for scope, regs in [("all", yields.get("regions", {}))]
            + [(y, r) for y, r in sorted(yields.get("years", {}).items())]
        }
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
    for region, classes in sorted(yields["regions"].items()):
        line = ", ".join(f"{k} {v['weight']:.1f}" for k, v in sorted(classes.items()))
        rs = "; ".join(f"{spec} = {ratio_value(classes, n, d)[2] if ratio_value(classes, n, d)[2] is not None else '-'}"
                       for spec, n, d in ratios)
        print(f"{region}: {line}" + (f"  |  {rs}" if rs else ""))


if __name__ == "__main__":
    main()
