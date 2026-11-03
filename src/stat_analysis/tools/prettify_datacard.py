#!/usr/bin/env python3
# coding: utf-8

"""
Standalone script to format and prettify CMS combine datacards into cleanly aligned columns.
Zero dependencies outside the Python standard library.

Example usage:
    python3 src/stat_analysis/tools/prettify_datacard.py datacard.txt
    python3 src/stat_analysis/tools/prettify_datacard.py datacard.txt -o datacard_pretty.txt
"""

import os
import sys
import argparse
from collections import OrderedDict, defaultdict
import fnmatch

parameter_directives = [
    "lnN",
    "lnU",
    "gmN",
    "trG",
    "unif",
    "dFD",
    "dFD2",
    "constr",
    "shape*",
    "discrete",
    "param",
    "rateParam",
    "flatParam",
    "extArg",
]

columnar_parameter_directives = ["lnN", "lnU", "gmN", "shape*"]


def multi_match(val, patterns):
    if isinstance(patterns, str):
        patterns = [patterns]
    for p in patterns:
        if fnmatch.fnmatch(val, p):
            return True
    return False


def read_datacard_blocks(datacard_path):
    fields = [
        "preamble",
        "counts",
        "shapes",
        "observations",
        "rates",
        "parameters",
        "groups",
        "auto_mc_stats",
        "nuisance_edits",
        "unknown",
    ]
    blocks = OrderedDict((field, []) for field in fields)

    with open(datacard_path, "r") as f:
        lines = []
        for line in f.readlines():
            line = line.strip().lstrip("- ")
            if line and not line.startswith(("#", "//")):
                lines.append(line)

    for preamble_offset, line in enumerate(lines):
        if line.startswith(("imax", "jmax", "kmax")):
            break
    else:
        raise ValueError(f"Datacard {datacard_path} contains no counter section (imax|jmax|kmax)")

    blocks["preamble"].extend(lines[:preamble_offset])
    del lines[:preamble_offset]

    # Observations
    for obs_offset in range(len(lines) - 1):
        line = lines[obs_offset]
        if not line.startswith("bin "):
            continue
        next_line = lines[obs_offset + 1]
        if next_line.startswith("observation "):
            blocks["observations"].extend([line, next_line])
            del lines[obs_offset:obs_offset + 2]
            break

    # Rates
    for rate_offset in range(len(lines) - 3):
        line = lines[rate_offset]
        if not line.startswith("bin "):
            continue
        next_lines = lines[rate_offset + 1:rate_offset + 4]
        if (
            next_lines[0].startswith("process ") and
            next_lines[1].startswith("process ") and
            next_lines[2].startswith("rate ")
        ):
            blocks["rates"].extend([line] + next_lines)
            del lines[rate_offset:rate_offset + 4]
            break

    # Categorize remaining lines
    for line in lines:
        words = line.split()
        if not words:
            continue
        field = "unknown"
        if words[0] in ("imax", "jmax", "kmax"):
            field = "counts"
        elif words[0] == "shapes":
            field = "shapes"
        elif len(words) >= 2:
            if words[1] == "autoMCStats":
                field = "auto_mc_stats"
            elif words[1] == "group":
                field = "groups"
            elif words[1] == "edit" and words[0] == "nuisance":
                field = "nuisance_edits"
            elif multi_match(words[1], parameter_directives):
                field = "parameters"
        blocks[field].append(line)

    return blocks


def write_datacard_pretty(f, blocks, skip_preamble=False):
    spacing = "  "
    sep = "-" * 80 + "\n"

    def write(l):
        if not isinstance(l, str):
            l = spacing.join(map(str, l))
        f.write(l.strip() + "\n")

    def align(lines, n_cols=None):
        rows = [(line.strip().split() if isinstance(line, str) else line) for line in lines]
        if not rows:
            return []
        if not n_cols:
            n_cols = max([len(row) for row in rows] + [0])
        for row in rows:
            diff = n_cols - len(row)
            if diff > 0:
                row.extend(diff * [""])
            elif diff < 0:
                del row[n_cols:]
        widths = [max([len(row[i]) for row in rows] + [0]) for i in range(n_cols)]
        return [
            spacing.join(val.ljust(width) for val, width in zip(row, widths)) for row in rows
        ]

    # Preamble
    if not skip_preamble and blocks.get("preamble"):
        write("\n".join(blocks["preamble"]))
        f.write(sep)

    # Counts
    if blocks.get("counts"):
        for line in blocks["counts"]:
            write(line.split()[:2])
        f.write(sep)

    # Shapes
    if blocks.get("shapes"):
        for line in align(blocks["shapes"], n_cols=6):
            write(line)
        f.write(sep)

    # Observations
    if blocks.get("observations"):
        for line in align(blocks["observations"]):
            write(line)
        f.write(sep)

    # Columnar parameters vs other parameters
    parameter_lines = blocks.get("parameters", [])
    columnar_parameter_lines = []
    other_parameter_lines = defaultdict(list)
    for line in parameter_lines:
        parts = line.strip().split()
        param_type = "missing" if len(parts) < 2 else parts[1]
        if multi_match(param_type, columnar_parameter_directives):
            columnar_parameter_lines.append(parts)
        else:
            other_parameter_lines[param_type].append(parts)

    rate_lines = []
    for rate_line in blocks.get("rates", []):
        parts = rate_line.strip().split()
        if columnar_parameter_lines:
            parts = parts[:1] + [""] + parts[1:]
        rate_lines.append(parts)

    aligned_lines = align(rate_lines + columnar_parameter_lines)
    rate_lines = aligned_lines[:len(rate_lines)]
    columnar_parameter_lines = aligned_lines[len(rate_lines):]

    if rate_lines:
        for line in rate_lines:
            write(line)
        f.write(sep)

    for line in columnar_parameter_lines:
        write(line)

    if columnar_parameter_lines and other_parameter_lines:
        f.write(sep)

    for p_type, lines in other_parameter_lines.items():
        for line in align(lines):
            write(line)

    if blocks.get("auto_mc_stats"):
        f.write(sep)
        for line in align(blocks["auto_mc_stats"]):
            write(line)

    if blocks.get("groups"):
        f.write(sep)
        for line in align(blocks["groups"]):
            write(line)

    if blocks.get("nuisance_edits"):
        f.write(sep)
        for line in align(blocks["nuisance_edits"]):
            write(line)

    if blocks.get("unknown"):
        f.write(sep)
        for line in blocks["unknown"]:
            write(line)


def prettify_datacard(input_card, output_card=None, skip_preamble=False):
    input_path = os.path.abspath(input_card)
    blocks = read_datacard_blocks(input_path)

    out_path = os.path.abspath(output_card) if output_card else input_path
    temp_path = out_path + ".tmp"
    with open(temp_path, "w") as f:
        write_datacard_pretty(f, blocks, skip_preamble=skip_preamble)
    os.replace(temp_path, out_path)
    print(f"Prettified datacard written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to the input datacard .txt")
    parser.add_argument("-o", "--output", help="Path to write the prettified datacard (defaults to in-place update)")
    parser.add_argument("--no-preamble", action="store_true", help="Remove preambles before counts")

    args = parser.parse_args()
    prettify_datacard(args.input, args.output, skip_preamble=args.no_preamble)


if __name__ == "__main__":
    main()
