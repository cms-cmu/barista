#!/usr/bin/env python3
# coding: utf-8

"""
Standalone script to extract and format yield tables directly from a Combine datacard and shapes.
Can output plain text, markdown, or LaTeX tables.

Example usage:
    python3 src/stat_analysis/tools/yield_table.py datacard.txt
    python3 src/stat_analysis/tools/yield_table.py datacard.txt --table-fmt latex
    python3 src/stat_analysis/tools/yield_table.py datacard.txt --bins "pass.*" "fail.*"
"""

import os
import sys
import re
import argparse
from optparse import OptionParser


def parse_datacard(datacard_path, mass=125.0):
    from HiggsAnalysis.CombinedLimit.DatacardParser import parseCard, addDatacardParserOptions
    from HiggsAnalysis.CombinedLimit.ShapeTools import ShapeBuilder

    parser = OptionParser()
    addDatacardParserOptions(parser)
    options = parser.parse_args([])[0]
    options.mass = mass
    options.fileName = datacard_path

    with open(datacard_path, "r") as f:
        dc = parseCard(f, options)
    sb = ShapeBuilder(dc, options)
    return dc, sb


class YieldValue:
    def __init__(self, val=0.0, err=0.0):
        self.val = float(val)
        self.err = float(err)

    def __add__(self, other):
        import math
        return YieldValue(self.val + other.val, math.sqrt(self.err**2 + other.err**2))

    def format(self, precision=2, style="plain"):
        fmt = f"{{:.{precision}f}}"
        v_str = fmt.format(self.val)
        e_str = fmt.format(self.err)
        if style == "latex":
            return f"${v_str} \\pm {e_str}$"
        return f"{v_str} +/- {e_str}"


def build_yield_table(datacard_path, bin_patterns=None, table_fmt="github", data_name="data_obs", precision=2):
    import ROOT
    ROOT.gROOT.SetBatch(True)

    dc, sb = parse_datacard(datacard_path)

    if not bin_patterns:
        bin_patterns = [".+"]

    groups = {}
    for pat in bin_patterns:
        regex = re.compile(pat)
        matched = [b for b in dc.bins if regex.match(b)]
        if matched:
            groups[pat] = matched

    if not groups:
        print(f"Warning: No bins matched patterns: {bin_patterns}. Available bins: {dc.bins}")
        return

    use_latex = "latex" in table_fmt.lower()
    style = "latex" if use_latex else "plain"

    processes = list(dc.processes)
    # Add data_obs if present
    if data_name not in processes:
        processes.append(data_name)

    rows = []
    total_bkg_per_group = {g: YieldValue(0.0, 0.0) for g in groups}

    for p in processes:
        is_data = (p == data_name)
        is_signal = (p in dc.isSignal and dc.isSignal[p])
        row = [str(p)]

        for g, cats in groups.items():
            tot = YieldValue(0.0, 0.0)
            for c in cats:
                try:
                    shape = sb.getShape(c, p)
                    err = ROOT.Double(0.0)
                    rate = shape.IntegralAndError(0, shape.GetNbinsX() + 1, err)
                except Exception:
                    # Try reading rate from dc.exp if shape isn't available
                    rate = dc.exp.get(c, {}).get(p, 0.0)
                    err = 0.0
                tot = tot + YieldValue(rate, err)

            if not is_data and not is_signal:
                total_bkg_per_group[g] = total_bkg_per_group[g] + tot

            row.append(tot.format(precision=precision, style=style))
        rows.append(row)

    # Add Total Background row
    tot_bkg_row = ["Total Bkg"]
    for g in groups:
        tot_bkg_row.append(total_bkg_per_group[g].format(precision=precision, style=style))
    rows.append(tot_bkg_row)

    headers = ["Process"] + list(groups.keys())

    # Try tabulate
    try:
        import tabulate
        if use_latex:
            tabulate.LATEX_ESCAPE_RULES.pop("\\", None)
            tabulate.LATEX_ESCAPE_RULES.pop("$", None)
        table_str = tabulate.tabulate(rows, headers=headers, tablefmt=table_fmt, colalign=("left",) + ("right",) * len(groups))
        print(table_str)
    except ImportError:
        # Fallback simple formatter
        col_widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
        fmt_row = " | ".join(f"{{:<{w}}}" if i == 0 else f"{{:>{w}}}" for i, w in enumerate(col_widths))
        sep_row = "-+-".join("-" * w for w in col_widths)
        print(fmt_row.format(*headers))
        print(sep_row)
        for r in rows:
            print(fmt_row.format(*r))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to datacard .txt")
    parser.add_argument("--bins", nargs="+", default=[".+"], help="Regex patterns to group bins")
    parser.add_argument("--table-fmt", default="github", help="Table format (github, grid, latex, pipe, etc.)")
    parser.add_argument("--precision", type=int, default=2, help="Decimal precision")
    parser.add_argument("--data-name", default="data_obs", help="Data process name")

    args = parser.parse_args()
    build_yield_table(
        datacard_path=os.path.abspath(args.input),
        bin_patterns=args.bins,
        table_fmt=args.table_fmt,
        data_name=args.data_name,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()
