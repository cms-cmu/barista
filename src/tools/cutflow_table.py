#!/usr/bin/env python3
"""
Cutflow Table Extraction and Formatting Tool.

Extracts cutflow tables from Coffea output files (.coffea) and formats them
into Markdown, LaTeX, Terminal (Rich/Tabulate), or CSV tables with support
for process/era aggregation, step filtering, and selection efficiencies.
"""

import argparse
import math
import os
import re
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Any

try:
    import coffea.util as coffea_util
except ImportError:
    coffea_util = None

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

try:
    from rich.console import Console
    from rich.table import Table as RichTable
except ImportError:
    Console = None
    RichTable = None


def parse_dataset_name(dataset: str, merge_ttbar: bool = False) -> Tuple[str, str, str]:
    """
    Parses dataset name into (process, year, sub_era).
    Supports Run 2 (UL16_preVFP, UL17, etc.) and Run 3 (2022_preEE, 2023, etc.)
    formats with single '_' or double '__' separators.
    
    Examples:
      'data_UL16_preVFPC' -> ('data', 'UL16_preVFP', 'UL16_preVFPC')
      'data_UL17C'        -> ('data', 'UL17', 'UL17C')
      'ttHbb_UL16_preVFP' -> ('ttHbb', 'UL16_preVFP', 'UL16_preVFP')
      'TTToHadronic__2022_preEE' -> ('TTToHadronic' or 'TTbar', '2022_preEE', '2022_preEE')
      'data__2022_preEE_B' -> ('data', '2022_preEE', '2022_preEE_B')
    """
    # Check double underscore format first: <process>__<year/era> or <process>__<sample>__<year>
    if "__" in dataset:
        parts = [p for p in dataset.split("__") if p]
        if len(parts) == 2:
            process, era_part = parts[0], parts[1]
        elif len(parts) >= 3:
            process, era_part = parts[0] if parts[0] != "histAll" else parts[1], parts[-1]
        else:
            process, era_part = dataset, "Unknown"

        # Check for year/era in era_part
        m = re.match(r"^((?:UL\d{2}(?:_preVFP|_postVFP)?|\d{4}(?:_preEE|_postEE|_BPix)?))(_[A-H]|[A-H])?$", era_part)
        if m:
            year = m.group(1)
            sub_era = era_part
        else:
            year = era_part
            sub_era = era_part

        if merge_ttbar and any(t in process for t in ["TTToHadronic", "TTToSemiLeptonic", "TTTo2L2Nu", "TTbar"]):
            process = "TTbar"
        return process, year, sub_era

    # Run 2 single underscore format: e.g. data_UL16_preVFPC, ttHbb_UL17, TTbar_from_d3_UL17D
    pattern_run2 = r"^(.*?)_(UL\d{2}(?:_preVFP|_postVFP)?)([A-H])?$"
    match = re.match(pattern_run2, dataset)
    if match:
        process = match.group(1)
        year = match.group(2)
        letter = match.group(3) or ""
        sub_era = f"{year}{letter}"
        if merge_ttbar and any(t in process for t in ["TTToHadronic", "TTToSemiLeptonic", "TTTo2L2Nu", "TTbar"]):
            process = "TTbar"
        return process, year, sub_era

    # Run 3 single underscore format: e.g. data_2022_preEEB, data_2022_preEE_B
    pattern_run3 = r"^(.*?)_(\d{4}(?:_preEE|_postEE|_BPix)?)(?:_?([A-H]))?$"
    match_r3 = re.match(pattern_run3, dataset)
    if match_r3:
        process = match_r3.group(1)
        year = match_r3.group(2)
        letter = match_r3.group(3) or ""
        sub_era = f"{year}_{letter}" if letter else year
        if merge_ttbar and any(t in process for t in ["TTToHadronic", "TTToSemiLeptonic", "TTTo2L2Nu", "TTbar"]):
            process = "TTbar"
        return process, year, sub_era

    # Fallback
    proc = dataset
    if merge_ttbar and any(t in proc for t in ["TTToHadronic", "TTToSemiLeptonic", "TTTo2L2Nu", "TTbar"]):
        proc = "TTbar"
    return proc, "Unknown", dataset


def load_coffea_file(filepath: str) -> Dict[str, Any]:
    """Loads a .coffea file using coffea.util."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    if coffea_util is None:
        raise ImportError("coffea package is required to load .coffea files.")
    return coffea_util.load(filepath)


def get_available_cutflow_keys(data: Dict[str, Any]) -> List[str]:
    """Finds all keys in the data dictionary that contain cutflow information."""
    cutflow_keys = []
    for k in data.keys():
        if "cutflow" in k.lower() or "cut_flow" in k.lower():
            cutflow_keys.append(k)
    return cutflow_keys


def format_number(val: Optional[float], is_data: bool = False, precision: int = 2) -> str:
    """Formats a number safely for table display."""
    if val is None:
        return "-"
    if isinstance(val, (int, float)):
        if math.isnan(val):
            return "NaN"
        if math.isinf(val):
            return "Inf" if val > 0 else "-Inf"
        if is_data or (abs(val - round(val)) < 1e-5 and val >= 1):
            return f"{int(round(val)):,}"
        if val == 0:
            return "0"
        if 0 < abs(val) < 0.01 or abs(val) >= 1e7:
            return f"{val:.{precision}e}"
        return f"{val:,.{precision}f}"
    return str(val)


def format_efficiency(eff: Optional[float]) -> str:
    """Formats efficiency as percentage."""
    if eff is None:
        return "-"
    if math.isnan(eff) or math.isinf(eff):
        return "-"
    return f"{eff * 100:.2f}%"


def build_cutflow_table(
    cf_dict: Dict[str, Dict[str, float]],
    group_by: str = "process",
    include_aux: bool = False,
    only_aux: bool = False,
    clean_names: bool = False,
    include_efficiencies: bool = False,
    show_empty: bool = False,
    merge_ttbar: bool = False,
    precision: int = 2,
) -> Tuple[List[str], List[List[str]]]:
    """
    Processes the raw cutflow dictionary into table headers and row records.
    """
    # 1. Determine the ordered list of cut steps across all datasets
    ordered_steps: OrderedDict[str, None] = OrderedDict()
    for ds, steps in cf_dict.items():
        if isinstance(steps, dict):
            for s in steps.keys():
                ordered_steps[s] = None
    all_steps = list(ordered_steps.keys())

    # Filter steps according to aux flags
    selected_steps = []
    for s in all_steps:
        is_aux = s.endswith("_woTrig") or "_woTrig" in s
        if only_aux:
            if is_aux:
                selected_steps.append(s)
        elif not include_aux:
            if not is_aux:
                selected_steps.append(s)
        else:
            selected_steps.append(s)

    if not selected_steps:
        selected_steps = all_steps

    # 2. Group datasets and sum yields
    columns_data: OrderedDict[str, Dict[str, float]] = OrderedDict()
    is_data_col: Dict[str, bool] = {}

    if group_by == "process":
        # Group by base process name (e.g. data, ttHbb, TTbar)
        process_groups: Dict[str, List[str]] = OrderedDict()
        for ds, steps in cf_dict.items():
            if not show_empty and (not steps or (isinstance(steps, dict) and len(steps) == 0)):
                continue
            proc, _, _ = parse_dataset_name(ds, merge_ttbar=merge_ttbar)
            process_groups.setdefault(proc, []).append(ds)

        # Sort so data comes first, then TTbar / signal / MC
        def sort_key(p: str):
            pl = p.lower()
            if "data" in pl:
                return (0, p)
            elif "ttbar" in pl or "ttto" in pl:
                return (1, p)
            return (2, p)

        sorted_procs = sorted(process_groups.keys(), key=sort_key)

        for proc in sorted_procs:
            ds_list = process_groups[proc]
            col_name = f"{proc} (Total)"
            is_data_col[col_name] = "data" in proc.lower()
            columns_data[col_name] = {s: 0.0 for s in selected_steps}
            for ds in ds_list:
                ds_dict = cf_dict.get(ds, {})
                if isinstance(ds_dict, dict):
                    for s in selected_steps:
                        val = ds_dict.get(s, 0.0)
                        if val is not None and not math.isnan(val):
                            columns_data[col_name][s] += val

    elif group_by == "year":
        # Group by (Process, Year) + Total per process
        proc_year_groups: Dict[Tuple[str, str], List[str]] = OrderedDict()
        all_procs: OrderedDict[str, None] = OrderedDict()
        all_years: OrderedDict[str, None] = OrderedDict()

        for ds, steps in cf_dict.items():
            if not show_empty and (not steps or (isinstance(steps, dict) and len(steps) == 0)):
                continue
            proc, year, _ = parse_dataset_name(ds, merge_ttbar=merge_ttbar)
            proc_year_groups.setdefault((proc, year), []).append(ds)
            all_procs[proc] = None
            all_years[year] = None

        def sort_key(p: str):
            pl = p.lower()
            if "data" in pl:
                return (0, p)
            elif "ttbar" in pl or "ttto" in pl:
                return (1, p)
            return (2, p)

        sorted_procs = sorted(all_procs.keys(), key=sort_key)
        sorted_years = sorted(all_years.keys())

        for proc in sorted_procs:
            proc_is_data = "data" in proc.lower()
            # Add year columns
            for yr in sorted_years:
                if (proc, yr) in proc_year_groups:
                    col_name = f"{proc} [{yr}]"
                    is_data_col[col_name] = proc_is_data
                    columns_data[col_name] = {s: 0.0 for s in selected_steps}
                    for ds in proc_year_groups[(proc, yr)]:
                        ds_dict = cf_dict.get(ds, {})
                        if isinstance(ds_dict, dict):
                            for s in selected_steps:
                                val = ds_dict.get(s, 0.0)
                                if val is not None and not math.isnan(val):
                                    columns_data[col_name][s] += val

            # Add process total column
            total_col_name = f"{proc} [Total]"
            is_data_col[total_col_name] = proc_is_data
            columns_data[total_col_name] = {s: 0.0 for s in selected_steps}
            for yr in sorted_years:
                if (proc, yr) in proc_year_groups:
                    for ds in proc_year_groups[(proc, yr)]:
                        ds_dict = cf_dict.get(ds, {})
                        if isinstance(ds_dict, dict):
                            for s in selected_steps:
                                val = ds_dict.get(s, 0.0)
                                if val is not None and not math.isnan(val):
                                    columns_data[total_col_name][s] += val

    else:  # 'dataset' or 'none'
        # Keep granular dataset columns
        filtered_ds = [
            ds for ds, v in cf_dict.items()
            if show_empty or (v and isinstance(v, dict) and len(v) > 0)
        ]
        sorted_datasets = sorted(
            filtered_ds,
            key=lambda d: (0 if "data" in d.lower() else (1 if "tt" in d.lower() else 2), d),
        )
        for ds in sorted_datasets:
            is_data_col[ds] = "data" in ds.lower()
            ds_dict = cf_dict.get(ds, {})
            columns_data[ds] = {
                s: (ds_dict.get(s, 0.0) if isinstance(ds_dict, dict) else 0.0)
                for s in selected_steps
            }

    # 3. Construct headers and table rows
    headers = ["Cut / Selection Step"]
    col_names = list(columns_data.keys())

    for col in col_names:
        headers.append(col)
        if include_efficiencies:
            headers.append(f"{col} (Rel %)")
            headers.append(f"{col} (Cum %)")

    rows: List[List[str]] = []
    prev_yields: Dict[str, Optional[float]] = {col: None for col in col_names}
    initial_yields: Dict[str, Optional[float]] = {col: None for col in col_names}

    for step in selected_steps:
        display_step = step.replace("_woTrig", "") if clean_names else step
        row = [display_step]
        for col in col_names:
            y = columns_data[col].get(step, 0.0)
            if initial_yields[col] is None:
                initial_yields[col] = y

            row.append(format_number(y, is_data=is_data_col[col], precision=precision))

            if include_efficiencies:
                prev_y = prev_yields[col]
                rel_eff = (y / prev_y) if (prev_y is not None and prev_y > 0) else (1.0 if prev_y is not None else None)
                init_y = initial_yields[col]
                cum_eff = (y / init_y) if (init_y is not None and init_y > 0) else None

                row.append(format_efficiency(rel_eff))
                row.append(format_efficiency(cum_eff))

            prev_yields[col] = y
        rows.append(row)

    return headers, rows


def render_markdown_table(headers: List[str], rows: List[List[str]], title: Optional[str] = None) -> str:
    """Renders table in GitHub Flavored Markdown."""
    lines = []
    if title:
        lines.append(f"### {title}\n")
    if tabulate:
        lines.append(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))
        
        header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
        sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
        lines.append(header_line)
        lines.append(sep_line)
        for row in rows:
            line = "| " + " | ".join(row[i].ljust(col_widths[i]) if i == 0 else row[i].rjust(col_widths[i]) for i in range(len(row))) + " |"
            lines.append(line)
    return "\n".join(lines)


def render_latex_table(
    headers: List[str],
    rows: List[List[str]],
    title: Optional[str] = None,
    label: str = "tab:cutflow",
) -> str:
    """Renders table in LaTeX format using booktabs."""
    num_cols = len(headers)
    col_spec = "l" + "r" * (num_cols - 1)
    
    def escape_latex(text: str) -> str:
        t = text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
        return t

    latex_headers = [escape_latex(h) for h in headers]
    
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{escape_latex(title) if title else 'Cutflow Summary'}}}" if title else r"  \caption{Cutflow Summary}",
        f"  \\label{{{label}}}",
        r"  \resizebox{\textwidth}{!}{",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        r"      \toprule",
        "      " + " & ".join(latex_headers) + r" \\",
        r"      \midrule",
    ]

    for row in rows:
        escaped_row = [escape_latex(row[0])] + [escape_latex(c) for c in row[1:]]
        lines.append("      " + " & ".join(escaped_row) + r" \\")

    lines.extend([
        r"      \bottomrule",
        r"    \end{tabular}",
        r"  }",
        r"\end{table}",
    ])
    return "\n".join(lines)


def render_terminal_table(headers: List[str], rows: List[List[str]], title: Optional[str] = None) -> str:
    """Renders table for terminal display."""
    if RichTable and Console:
        console = Console(record=True, width=200)
        table = RichTable(title=title or "Cutflow Table", show_header=True, header_style="bold cyan")
        table.add_column(headers[0], justify="left", style="bold white")
        for h in headers[1:]:
            table.add_column(h, justify="right", style="green" if "data" in h.lower() else "yellow")
        for r in rows:
            table.add_row(*r)
        with console.capture() as capture:
            console.print(table)
        return capture.get()
    elif tabulate:
        return tabulate(rows, headers=headers, tablefmt="fancy_grid")
    else:
        return render_markdown_table(headers, rows, title=title)


def render_csv_table(headers: List[str], rows: List[List[str]]) -> str:
    """Renders table as CSV."""
    import csv
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description="Extract and format cutflow tables from Coffea output files (.coffea)."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the .coffea file."
    )
    parser.add_argument(
        "-k", "--key", default="cutFlowFourTag",
        help="Cutflow accumulator key to read (default: cutFlowFourTag)."
    )
    parser.add_argument(
        "--list-keys", action="store_true",
        help="List all available cutflow keys in the input file and exit."
    )
    parser.add_argument(
        "-f", "--format", choices=["markdown", "latex", "terminal", "csv"], default="markdown",
        help="Output format (default: markdown)."
    )
    parser.add_argument(
        "-g", "--group-by", choices=["process", "year", "dataset"], default="process",
        help="Aggregation level for datasets (default: process)."
    )
    parser.add_argument(
        "--wo-trig", "--only-aux", dest="wo_trig", action="store_true",
        help="Use without-trigger numbers (_woTrig cuts) instead of standard cuts."
    )
    parser.add_argument(
        "--include-aux", action="store_true",
        help="Include auxiliary steps alongside standard steps (e.g. both standard and _woTrig cuts)."
    )
    parser.add_argument(
        "--clean-names", action="store_true",
        help="Clean cut names in display (strip '_woTrig' suffix when --wo-trig is active)."
    )
    parser.add_argument(
        "--show-empty", action="store_true",
        help="Show datasets/processes even if they contain 0 events / empty dictionaries."
    )
    parser.add_argument(
        "--merge-ttbar", action="store_true",
        help="Merge individual TTbar channels (TTToHadronic, TTToSemiLeptonic, TTTo2L2Nu) into a single TTbar process."
    )
    parser.add_argument(
        "-e", "--efficiencies", action="store_true",
        help="Include relative (step-by-step) and cumulative selection efficiencies."
    )
    parser.add_argument(
        "-p", "--precision", type=int, default=2,
        help="Decimal precision for floating-point / weighted yields (default: 2)."
    )
    parser.add_argument(
        "-o", "--output", help="Optional output file path to write the table to."
    )
    parser.add_argument(
        "--title", help="Optional custom title/caption for the table."
    )

    args = parser.parse_args()

    # 1. Load data
    try:
        data = load_coffea_file(args.input)
    except Exception as e:
        print(f"Error loading coffea file: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Key listing mode
    available_keys = get_available_cutflow_keys(data)
    if args.list_keys:
        print(f"Available cutflow keys in '{args.input}':")
        for k in available_keys:
            print(f"  - {k}")
        sys.exit(0)

    # 3. Validate cutflow key
    if args.key not in data:
        print(
            f"Error: Key '{args.key}' not found in '{args.input}'.",
            file=sys.stderr
        )
        if available_keys:
            print(f"Available cutflow keys: {', '.join(available_keys)}", file=sys.stderr)
        sys.exit(1)

    cf_dict = data[args.key]
    if not isinstance(cf_dict, dict):
        print(f"Error: Key '{args.key}' is not a dictionary (type: {type(cf_dict)}).", file=sys.stderr)
        sys.exit(1)

    # 4. Build table data
    table_title = args.title or f"Cutflow Table: {args.key} ({os.path.basename(args.input)})"
    headers, rows = build_cutflow_table(
        cf_dict=cf_dict,
        group_by=args.group_by,
        include_aux=args.include_aux,
        only_aux=args.wo_trig,
        clean_names=args.clean_names,
        include_efficiencies=args.efficiencies,
        show_empty=args.show_empty,
        merge_ttbar=args.merge_ttbar,
        precision=args.precision,
    )

    # 5. Render output
    if args.format == "markdown":
        rendered = render_markdown_table(headers, rows, title=table_title)
    elif args.format == "latex":
        rendered = render_latex_table(headers, rows, title=table_title)
    elif args.format == "terminal":
        rendered = render_terminal_table(headers, rows, title=table_title)
    elif args.format == "csv":
        rendered = render_csv_table(headers, rows)
    else:
        rendered = render_markdown_table(headers, rows, title=table_title)

    # 6. Output to file or stdout
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(rendered + "\n")
        print(f"Table saved to: {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
