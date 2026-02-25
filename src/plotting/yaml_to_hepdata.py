#!/usr/bin/env python3
"""
Convert barista/coffea4bees plot YAML files to HEPData submission format.

Uses hepdata_lib — run via the pixi hepdata environment:

    pixi run -e hepdata python src/plotting/yaml_to_hepdata.py \
        coffea4bees/archive/plotsForPaper/SvB_MA_postfitplots_fit_b_138.yaml

    # Multiple files (all go into the same submission):
    pixi run -e hepdata python src/plotting/yaml_to_hepdata.py \
        coffea4bees/archive/plotsForPaper/SvB_MA_postfitplots_fit_b_138.yaml \
        coffea4bees/archive/plotsForPaper/SvB_MA_postfitplots_prefit_138.yaml

    # Custom output directory:
    pixi run -e hepdata python src/plotting/yaml_to_hepdata.py \
        -o /some/other/dir \
        coffea4bees/archive/plotsForPaper/SvB_MA_postfitplots_fit_b_138.yaml

    # Validate:
    pixi run -e hepdata hepdata-validate -d plotsForPaper/hepdata_submission/

    # Create tar for upload:
    tar czf hepdata_submission.tar.gz plotsForPaper/hepdata_submission/

By default, output is written to plotsForPaper/hepdata_submission/ relative
to the current working directory.
"""

import argparse
import math
import os
import sys
from pathlib import Path

import yaml

try:
    from hepdata_lib import Submission, Table, Variable, Uncertainty
except ImportError:
    print(
        "ERROR: hepdata_lib is not installed.\n"
        "Run this script via the pixi hepdata environment:\n"
        "  pixi run -e hepdata python src/plotting/yaml_to_hepdata.py ..."
    )
    sys.exit(1)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_yaml(filepath: str) -> dict:
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def variance_to_sqrt(variances: list) -> list:
    return [math.sqrt(v) for v in variances]


# ── metadata ─────────────────────────────────────────────────────────────────

def extract_metadata(data: dict) -> dict:
    kwargs = data.get("kwargs", {})
    meta = {
        "variable": data.get("var", ""),
        "cut": data.get("cut", ""),
        "region": data.get("region", ""),
        "year_str": kwargs.get("year_str", ""),
        "ylabel": kwargs.get("ylabel", ""),
        "file_name": data.get("file_name", ""),
    }
    for key, val in kwargs.get("text", {}).items():
        if isinstance(val, dict) and "text" in val:
            meta[key] = val["text"]
    return meta


def build_table_name(meta: dict, filename: str) -> str:
    parts = []
    if meta.get("region_text"):
        parts.append(meta["region_text"])
    if meta.get("fit_text"):
        parts.append(meta["fit_text"])
    if meta.get("variable"):
        parts.append(meta["variable"])
    return " | ".join(parts) if parts else Path(filename).stem


def build_table_description(meta: dict) -> str:
    lines = []
    if meta.get("fit_text"):
        lines.append(meta["fit_text"])
    if meta.get("region"):
        lines.append(f"Region: {meta['region']}")
    if meta.get("cut"):
        lines.append(f"Selection: {meta['cut']}")
    if meta.get("variable"):
        lines.append(f"Variable: {meta['variable']}")
    if meta.get("year_str"):
        lines.append(f"Luminosity: {meta['year_str']}")
    return ". ".join(lines) if lines else "Distribution"


# ── variable builders ────────────────────────────────────────────────────────

def make_x_variable(edges: list, x_label: str = "") -> Variable:
    """Create the binned independent x-axis variable."""
    label = x_label if x_label else "x"
    x_var = Variable(label, is_independent=True, is_binned=True, units="")
    x_var.values = list(zip(edges[:-1], edges[1:]))
    return x_var


def add_dependent_variable(
    table: Table,
    name: str,
    values: list,
    variances: list = None,
    units: str = "Events",
    region: str = "",
    process: str = "",
) -> None:
    """Add a dependent variable (y-axis) with optional stat uncertainty."""
    dep = Variable(name, is_independent=False, is_binned=False, units=units)
    dep.values = values

    if region:
        dep.add_qualifier("Region", region)
    if process:
        dep.add_qualifier("Process", process)

    if variances:
        unc = Uncertainty("stat", is_symmetric=True)
        unc.values = variance_to_sqrt(variances)
        dep.add_uncertainty(unc)

    table.add_variable(dep)


def add_ratio_variable(
    table: Table,
    name: str,
    values: list,
    errors: list = None,
    error_label: str = "stat",
    region: str = "",
) -> None:
    """Add a ratio variable (dimensionless) with optional uncertainty."""
    dep = Variable(name, is_independent=False, is_binned=False, units="")
    dep.values = values

    if region:
        dep.add_qualifier("Region", region)

    if errors:
        unc = Uncertainty(error_label, is_symmetric=True)
        unc.values = errors
        dep.add_uncertainty(unc)

    table.add_variable(dep)


# ── main conversion ─────────────────────────────────────────────────────────

def convert_single_file(
    filepath: str,
    submission: Submission,
    reaction: str,
    cmenergies: float,
) -> None:
    """Convert one barista plot YAML into a HEPData Table."""
    data = load_yaml(filepath)
    meta = extract_metadata(data)
    src_filename = os.path.basename(filepath)

    # Use the input filename (without .yaml) as the table name for hepdata_lib,
    # which will also use it as the output data file name.
    stem = Path(filepath).stem
    table_name = build_table_name(meta, src_filename)

    table = Table(stem)
    table.description = build_table_description(meta)
    table.location = f"Source: {src_filename}"

    # Keywords (required by HEPData schema)
    table.keywords["reactions"] = [reaction]
    table.keywords["cmenergies"] = [cmenergies]
    table.keywords["observables"] = ["N"]

    # ── find bin edges ───────────────────────────────────────────────────
    edges = None
    x_label = ""
    for section_key in ("hists", "stack"):
        for _, hd in data.get(section_key, {}).items():
            if "edges" in hd:
                edges = hd["edges"]
                x_label = hd.get("x_label", "")
                break
        if edges:
            break

    if edges is None:
        print(f"  WARNING: no bin edges in {src_filename}, skipping.")
        return

    # ── independent variable (x-axis) ────────────────────────────────────
    table.add_variable(make_x_variable(edges, x_label))

    region = meta.get("region", "")

    # ── data histograms ──────────────────────────────────────────────────
    for hist_name, hd in data.get("hists", {}).items():
        label = hd.get("label", hist_name)
        process = hd.get("process", hist_name)
        add_dependent_variable(
            table, name=label,
            values=hd["values"], variances=hd.get("variances"),
            region=region, process=process,
        )

    # ── stacked backgrounds ──────────────────────────────────────────────
    for stack_name, sd in data.get("stack", {}).items():
        label = sd.get("label", stack_name)
        process = sd.get("process", stack_name)
        if isinstance(process, list):
            process = " + ".join(process)
        add_dependent_variable(
            table, name=label,
            values=sd["values"], variances=sd.get("variances"),
            region=region, process=process,
        )

    # ── ratio (data / prediction) ────────────────────────────────────────
    ratio_section = data.get("ratio", {})
    for ratio_name, rd in ratio_section.items():
        if rd.get("type") == "band":
            continue
        add_ratio_variable(
            table, name="Data / Prediction",
            values=rd.get("ratio", []),
            errors=rd.get("error"),
            error_label="stat",
            region=region,
        )

    # ── background uncertainty band ──────────────────────────────────────
    for ratio_name, rd in ratio_section.items():
        if rd.get("type") != "band":
            continue
        band_label = rd.get("label", "Background uncertainty")
        add_ratio_variable(
            table, name=f"Ratio {band_label}",
            values=rd.get("ratio", []),
            errors=rd.get("error"),
            error_label=band_label,
            region=region,
        )

    submission.add_table(table)
    print(f"  Added table: '{table_name}' -> {stem}.yaml")


def main():
    parser = argparse.ArgumentParser(
        description="Convert barista plot YAML files to HEPData submission format."
    )
    parser.add_argument(
        "input_files", nargs="+",
        help="One or more barista plot YAML files to convert.",
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Output directory. Default: plotsForPaper_output/hepdata_submission/ "
             "in the current working directory.",
    )
    parser.add_argument(
        "--comment",
        default="Post-fit distributions from the HH->4b analysis.",
        help="Comment/description for the HEPData submission.",
    )
    parser.add_argument(
        "--reaction",
        default="P P --> H H --> B B B B",
        help="Reaction string for HEPData keywords.",
    )
    parser.add_argument(
        "--cmenergies",
        type=float, default=13.0,
        help="Center-of-mass energy in TeV (default: 13).",
    )
    args = parser.parse_args()

    for f in args.input_files:
        if not os.path.isfile(f):
            print(f"ERROR: File not found: {f}")
            sys.exit(1)

    # Default output dir: plotsForPaper_output/hepdata_submission/ in cwd
    if args.output_dir is None:
        output_dir = os.path.join("plotsForPaper_output", "hepdata_submission")
    else:
        output_dir = args.output_dir

    # Create submission
    submission = Submission()
    submission.comment = args.comment

    print(f"Converting {len(args.input_files)} file(s) to HEPData format...\n")

    for filepath in args.input_files:
        print(f"Processing: {filepath}")
        convert_single_file(filepath, submission, args.reaction, args.cmenergies)

    # Write output (creates submission.yaml + per-table yaml)
    os.makedirs(output_dir, exist_ok=True)
    submission.create_files(output_dir)

    print(f"\nHEPData submission written to: {output_dir}/")
    print(f"Validate with:  pixi run -e hepdata hepdata-validate -d {output_dir}/")


if __name__ == "__main__":
    main()
