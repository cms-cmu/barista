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
        rows.append(f'<tr><th colspan="3" class="region"><span class="region-badge">{html.escape(region)}</span></th></tr>')
        for label in sorted(classes):
            c = classes[label]
            rows.append(f"<tr><td><code>{html.escape(label)}</code></td><td>{c['count']:,}</td><td>{c['weight']:,.1f}</td></tr>")
        for spec, num, den in ratios:
            n, d, r = ratio_value(classes, num, den)
            val = "-" if r is None else f"{r:.4f}"
            detail = "" if r is None else f" <span class='muted'>({n:,.1f} / {d:,.1f})</span>"
            rows.append(f"<tr class='ratio'><td><strong>{html.escape(spec)}</strong></td><td></td><td><b>{val}</b>{detail}</td></tr>")
    return (f"<div class='card'><h3>{html.escape(title)}</h3>"
            f"<div class='wrap-table'><table><thead><tr><th class='l'>class</th><th>count</th><th>weight</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div></div>")


def render(yields: dict, ratios, title: str, source: str) -> str:
    parts = [f"<header><h1>{html.escape(title)}: loaded yields</h1>"
             f"<p class='muted'>Per-class event counts and summed training weights at the end of "
             f"dataset loading (before the k-fold split), by region. Source: <code>{html.escape(source)}</code></p></header>",
             "<div class='grid-tables'>",
             _table("All years", yields.get("regions", {}), ratios)]
    for year in sorted(yields.get("years", {})):
        parts.append(_table(f"Year {year}", yields["years"][year], ratios))
    parts.append("</div>")
    style = """
    :root {
      --bg: #f8fafc;
      --fg: #0f172a;
      --muted: #64748b;
      --card: #ffffff;
      --line: #e2e8f0;
      --accent: #2563eb;
      --accent-soft: #eff6ff;
      --warn-bg: #fef3c7;
      --warn-fg: #b45309;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: var(--sans); background: var(--bg); color: var(--fg); font-size: 14px; line-height: 1.5; padding: 24px 32px 64px; }
    .container { max-width: 1200px; margin: 0 auto; display: flex; flex-direction: column; gap: 18px; }
    header { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 18px 24px; }
    h1 { font-size: 1.35rem; font-weight: 700; margin-bottom: 6px; }
    h3 { font-size: 1rem; font-weight: 600; margin-bottom: 12px; }
    .muted { color: var(--muted); font-size: 13px; }
    code { font-family: var(--mono); font-size: 0.88em; background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }
    .grid-tables { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; align-items: start; }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 18px; }
    .wrap-table { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
    th, td { border-bottom: 1px solid var(--line); padding: 7px 10px; text-align: right; }
    th { background: #f8fafc; color: var(--muted); font-size: 11.5px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.03em; }
    th.l, td:first-child { text-align: left; }
    th.region { text-align: left; background: #f1f5f9; padding: 6px 10px; }
    .region-badge { display: inline-block; font-weight: 700; color: var(--fg); text-transform: none; font-size: 12px; }
    tr.ratio td { background: var(--accent-soft); }
    tbody tr:hover td { background: #f8fafc; }
    """
    return (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{html.escape(title)} yields</title><style>{style}</style></head><body><div class='container'>"
            + "".join(parts) + "</div></body></html>")


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
