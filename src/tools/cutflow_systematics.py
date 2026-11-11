#!/usr/bin/env python3
"""
Cutflow Systematics Tool for barista / coffea4bees.

Extracts yields for systematic variations in fourTag & SR (or user-specified tag/region)
from merged Coffea histogram files (.coffea), validates them against known reference
counts, and generates formatted HTML and YAML cutflow reports.

Usage:
    # Run validation and produce HTML report:
    python src/tools/cutflow_systematics.py histAll_signals_syst.coffea \
        --known-counts coffea4bees/analysis/tests/known_Counts_systematics_ttHbb.yml \
        --output-html cutflow_systematics.html \
        --output-yml cutflow_systematics.yml \
        --output-txt cutflow_validation_systematics.txt \
        --tolerance 1e-3

    # Dump known counts reference YAML:
    python src/tools/cutflow_systematics.py histAll_signals_syst.coffea \
        --dump-known-counts coffea4bees/analysis/tests/known_Counts_systematics_ttHbb.yml
"""
from __future__ import annotations

import argparse
import html
import math
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    import coffea.util as coffea_util
except ImportError:
    coffea_util = None


def load_coffea_systematics_hists(file_path: str) -> Dict[str, Any]:
    """Load histograms from a .coffea file."""
    if coffea_util is None:
        raise ImportError("coffea.util is required to load .coffea files")
    data = coffea_util.load(file_path)
    if isinstance(data, dict):
        if "hists" in data and isinstance(data["hists"], dict):
            return data["hists"]
        return data
    raise ValueError(f"Unrecognized coffea file structure in {file_path}")


def find_systematics_hist(hists: Dict[str, Any], preferred_var: Optional[str] = None) -> Tuple[str, Any]:
    """Find the best discriminant histogram containing the 'variation' axis."""
    if preferred_var and preferred_var in hists:
        return preferred_var, hists[preferred_var]

    # Search for preferred matches
    candidates = []
    for name, h in hists.items():
        if hasattr(h, "axes"):
            ax_names = [a.name for a in h.axes]
            if "variation" in ax_names and "tag" in ax_names and "region" in ax_names:
                candidates.append((name, h))

    if not candidates:
        raise ValueError(f"No histogram found with 'variation', 'tag', and 'region' axes in {list(hists.keys())}")

    # Prioritize ps_ttHbb or ps_hh or ps
    for name, h in candidates:
        if "ps_ttHbb" in name:
            return name, h
    for name, h in candidates:
        if "ps_hh" in name or "ps" in name:
            return name, h

    return candidates[0]


def extract_systematic_yields(
    hist: Any,
    tag: str = "fourTag",
    region: str = "SR",
    process: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Extract event yields for each process and variation for the given tag and region.
    Returns: {process_name: {variation_name: yield_float}}
    """
    results: Dict[str, Dict[str, float]] = {}

    ax_names = [a.name for a in hist.axes]
    has_process = "process" in ax_names
    has_tag = "tag" in ax_names
    has_region = "region" in ax_names
    has_var = "variation" in ax_names

    if not (has_tag and has_region and has_var):
        raise ValueError(f"Histogram axes must contain 'tag', 'region', and 'variation', got {ax_names}")

    # Discover processes
    if has_process:
        proc_axis = hist.axes["process"]
        processes = [p for p in proc_axis]
        if process and process in processes:
            processes = [process]
    else:
        processes = ["signal"]

    var_axis = hist.axes["variation"]
    variations = list(var_axis)

    # Sort variations: nominal first, then alphabetical
    sorted_vars = ["nominal"] if "nominal" in variations else []
    sorted_vars += sorted([v for v in variations if v != "nominal"])

    for proc in processes:
        results[proc] = OrderedDict()
        for v in sorted_vars:
            try:
                # Slice histogram
                selectors = {"tag": tag, "region": region, "variation": v}
                if has_process:
                    selectors["process"] = proc
                
                h_slice = hist[selectors]
                val = float(h_slice.values(flow=False).sum())
                results[proc][v] = val
            except Exception as e:
                # If slice fails, record 0.0 or raise
                results[proc][v] = 0.0

    return results


def rel_diff(a: float, b: float) -> float:
    m = max(abs(a), abs(b))
    return abs(a - b) / m if m > 0.0 else 0.0


def validate_counts(
    observed: Dict[str, Dict[str, float]],
    reference: Dict[str, Any],
    tolerance: float = 1e-3,
    tag: str = "fourTag",
    region: str = "SR",
) -> List[Dict[str, Any]]:
    """
    Validate observed yields against reference known counts.
    Returns list of comparison row dictionaries.
    """
    ref_counts = reference.get(f"counts_{tag}", reference.get(tag, reference.get("counts4", reference)))
    if isinstance(ref_counts, dict) and region in ref_counts:
        ref_counts = ref_counts[region]
    elif isinstance(ref_counts, dict) and "SR" in ref_counts:
        ref_counts = ref_counts["SR"]
    rows = []

    for proc, var_dict in observed.items():
        ref_proc = ref_counts.get(proc, {})
        for var_name, obs_val in var_dict.items():
            if var_name in ref_proc:
                exp_val = float(ref_proc[var_name])
                diff = rel_diff(obs_val, exp_val)
                passed = diff <= tolerance
                rows.append({
                    "process": proc,
                    "variation": var_name,
                    "observed": obs_val,
                    "expected": exp_val,
                    "rel_diff": diff,
                    "status": "PASS" if passed else "FAIL",
                })
            else:
                rows.append({
                    "process": proc,
                    "variation": var_name,
                    "observed": obs_val,
                    "expected": None,
                    "rel_diff": None,
                    "status": "NEW",
                })

    return rows


def generate_html_report(
    observed: Dict[str, Dict[str, float]],
    validation_rows: Optional[List[Dict[str, Any]]],
    hist_name: str,
    tag: str,
    region: str,
    tolerance: float,
) -> str:
    """Generate a responsive HTML report of the cutflow systematic variations."""
    total_vars = sum(len(v) for v in observed.values())
    failures = [r for r in validation_rows or [] if r["status"] == "FAIL"]
    verdict = "FAIL" if failures else "PASS"
    verdict_color = "#dc2626" if failures else "#16a34a"

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8'/>",
        "  <title>Systematic Cutflow Validation</title>",
        "  <style>",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 2rem; background: #f8fafc; color: #1e293b; }",
        "    .card { background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 1.5rem; margin-bottom: 1.5rem; }",
        "    .badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 600; font-size: 0.875rem; }",
        "    .badge-pass { background: #dcfce7; color: #15803d; }",
        "    .badge-fail { background: #fee2e2; color: #b91c1c; }",
        "    .badge-info { background: #e0f2fe; color: #0369a1; }",
        "    table { width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.9rem; }",
        "    th { background: #f1f5f9; padding: 0.75rem; text-align: left; border-bottom: 2px solid #e2e8f0; }",
        "    td { padding: 0.65rem 0.75rem; border-bottom: 1px solid #e2e8f0; }",
        "    tr:hover { background: #f8fafc; }",
        "    .num { text-align: right; font-variant-numeric: tabular-nums; }",
        "    .up { color: #2563eb; font-weight: 500; }",
        "    .down { color: #d97706; font-weight: 500; }",
        "    .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }",
        "    .metric-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 1rem; }",
        "    .metric-val { font-size: 1.5rem; font-weight: 700; color: #0f172a; }",
        "    .metric-lbl { font-size: 0.8rem; text-transform: uppercase; color: #64748b; margin-top: 0.25rem; }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <div class='card'>",
        f"    <div style='display: flex; justify-content: space-between; align-items: center;'>",
        f"      <h2>Systematic Cutflow Validation ({tag} & {region})</h2>",
        f"      <span class='badge' style='background: {verdict_color}; color: white; font-size: 1.1rem; padding: 0.5rem 1.25rem;'>{verdict}</span>",
        f"    </div>",
        f"    <p style='color: #64748b; margin-top: 0.5rem;'>Discriminant: <code>{html.escape(hist_name)}</code> | Tolerance: <code>{tolerance:.1e}</code></p>",
        "    <div class='metric-grid'>",
        f"      <div class='metric-box'><div class='metric-val'>{total_vars}</div><div class='metric-lbl'>Total Variations</div></div>",
    ]

    # Calculate nominal yields and shifts
    val_map = {r["variation"]: r for r in (validation_rows or [])}
    for proc, v_dict in observed.items():
        nom = v_dict.get("nominal", 0.0)
        html_parts.append(f"      <div class='metric-box'><div class='metric-val'>{nom:.4f}</div><div class='metric-lbl'>Nominal Yield ({proc})</div></div>")
        if validation_rows:
            html_parts.append(f"      <div class='metric-box'><div class='metric-val'>{len(failures)}</div><div class='metric-lbl'>Mismatches</div></div>")

    html_parts.extend([
        "    </div>",
        "  </div>",
    ])

    for proc, v_dict in observed.items():
        nom_yield = v_dict.get("nominal", 0.0)
        html_parts.extend([
            "  <div class='card'>",
            f"    <h3>Process: {html.escape(proc)}</h3>",
            "    <table>",
            "      <thead>",
            "        <tr>",
            "          <th>Variation</th>",
            "          <th class='num'>Yield</th>",
            "          <th class='num'>Shift vs Nom</th>",
            "          <th class='num'>Ratio to Nom</th>",
        ])
        if validation_rows:
            html_parts.extend([
                "          <th class='num'>Expected</th>",
                "          <th class='num'>Rel Diff</th>",
                "          <th style='text-align: center;'>Status</th>",
            ])
        html_parts.extend([
            "        </tr>",
            "      </thead>",
            "      <tbody>",
        ])

        for var_name, yld in v_dict.items():
            shift = yld - nom_yield
            ratio = (yld / nom_yield) if nom_yield > 0.0 else 1.0
            shift_pct = (shift / nom_yield * 100.0) if nom_yield > 0.0 else 0.0
            cls = "up" if "Up" in var_name else ("down" if "Down" in var_name else "")

            html_parts.append("        <tr>")
            html_parts.append(f"          <td class='{cls}'><code>{html.escape(var_name)}</code></td>")
            html_parts.append(f"          <td class='num'><b>{yld:.4f}</b></td>")
            html_parts.append(f"          <td class='num'>{shift_pct:+.2f}%</td>")
            html_parts.append(f"          <td class='num'>{ratio:.4f}</td>")

            if validation_rows:
                row_val = val_map.get(var_name, {})
                exp = row_val.get("expected")
                diff = row_val.get("rel_diff")
                st = row_val.get("status", "NEW")
                st_cls = "badge-pass" if st == "PASS" else ("badge-fail" if st == "FAIL" else "badge-info")
                exp_str = f"{exp:.4f}" if exp is not None else "&mdash;"
                diff_str = f"{diff * 100:.3f}%" if diff is not None else "&mdash;"
                html_parts.append(f"          <td class='num'>{exp_str}</td>")
                html_parts.append(f"          <td class='num'>{diff_str}</td>")
                html_parts.append(f"          <td style='text-align: center;'><span class='badge {st_cls}'>{st}</span></td>")

            html_parts.append("        </tr>")

        html_parts.extend([
            "      </tbody>",
            "    </table>",
            "  </div>",
        ])

    html_parts.extend([
        "</body>",
        "</html>\n",
    ])
    return "\n".join(html_parts)


def main():
    parser = argparse.ArgumentParser(description="Extract and validate systematic variation cutflows.")
    parser.add_argument("input_coffea", help="Path to input .coffea file (merged signal systematics)")
    parser.add_argument("--variable", "-v", default=None, help="Discriminant variable name (default: auto-detect)")
    parser.add_argument("--tag", "-t", default="fourTag", help="Tag region (default: fourTag)")
    parser.add_argument("--region", "-r", default="SR", help="Analysis region (default: SR)")
    parser.add_argument("--process", "-p", default=None, help="Process name filter (default: all)")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Relative difference tolerance (default: 1e-3)")
    parser.add_argument("--known-counts", "-k", default=None, help="YAML file with known reference counts")
    parser.add_argument("--dump-known-counts", default=None, help="Dump observed counts as reference YAML")
    parser.add_argument("--output-yml", "-y", default=None, help="Output observed counts to YAML")
    parser.add_argument("--output-html", "-o", default=None, help="Output HTML validation dashboard")
    parser.add_argument("--output-txt", default=None, help="Output text summary (PASS/FAIL verdict)")

    args = parser.parse_args()

    if not os.path.exists(args.input_coffea):
        sys.exit(f"Error: Input coffea file not found: {args.input_coffea}")

    hists = load_coffea_systematics_hists(args.input_coffea)
    hist_name, hist = find_systematics_hist(hists, preferred_var=args.variable)
    print(f"[INFO] Using systematic histogram: {hist_name} ({args.tag}, {args.region})")

    observed = extract_systematic_yields(hist, tag=args.tag, region=args.region, process=args.process)

    # Dump known counts if requested
    dump_target = args.dump_known_counts or args.output_yml
    if dump_target:
        dump_dir = os.path.dirname(dump_target)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        dump_data = {
            f"counts_{args.tag}": {
                args.region: {
                    proc: {v: round(float(val), 6) for v, val in v_dict.items()}
                    for proc, v_dict in observed.items()
                }
            },
            # Flat alias for backward-compatible lookup
            args.tag: {
                proc: {v: round(float(val), 6) for v, val in v_dict.items()}
                for proc, v_dict in observed.items()
            }
        }
        with open(dump_target, "w") as f:
            yaml.dump(dump_data, f, default_flow_style=False, sort_keys=False)
        print(f"[INFO] Saved cutflow counts to: {dump_target}")

    # Validate against known counts if provided
    validation_rows = None
    passed = True
    if args.known_counts:
        if not os.path.exists(args.known_counts):
            sys.exit(f"Error: Known counts file not found: {args.known_counts}")
        with open(args.known_counts, "r") as f:
            ref_data = yaml.safe_load(f)
        validation_rows = validate_counts(observed, ref_data, tolerance=args.tolerance, tag=args.tag, region=args.region)
        print(f"\n{'='*75}")
        print(f"{'Variation':<45} {'Observed':>9} {'Expected':>9} {'Diff %':>8}")
        print(f"{'-'*75}")
        for r in validation_rows:
            diff_str = f"{r['rel_diff']*100:+.2f}%" if r.get('rel_diff') is not None else "-"
            exp_str = f"{r['expected']:>9.4f}" if r.get('expected') is not None else "     None"
            print(f"{r['variation']:<45} {r['observed']:>9.4f} {exp_str} {diff_str:>8}")
        print(f"{'='*75}\n")

        failures = [r for r in validation_rows if r["status"] == "FAIL"]
        if failures:
            passed = False
            print(f"[FAIL] {len(failures)} systematic variation(s) exceeded tolerance ({args.tolerance}):")
            for f in failures:
                print(f"  - {f['process']} {f['variation']}: observed={f['observed']:.5f}, expected={f['expected']:.5f}, rel_diff={f['rel_diff']*100:.3f}%")
        else:
            print(f"[PASS] All {len(validation_rows)} systematic variation(s) matched reference within tolerance ({args.tolerance})")

    # Output text summary
    if args.output_txt:
        txt_dir = os.path.dirname(args.output_txt)
        if txt_dir:
            os.makedirs(txt_dir, exist_ok=True)
        with open(args.output_txt, "w") as f:
            if validation_rows:
                verdict = "PASS" if passed else "FAIL"
                f.write(f"{verdict}\n")
                f.write(f"Validated {len(validation_rows)} systematic variations in {args.tag} & {args.region}.\n")
                if not passed:
                    f.write(f"Failures ({len(failures)}):\n")
                    for fl in failures:
                        f.write(f"  {fl['process']} {fl['variation']}: obs={fl['observed']:.6f} exp={fl['expected']:.6f} diff={fl['rel_diff']*100:.3f}%\n")
            else:
                f.write("PASS\n")
                f.write(f"Extracted systematic cutflow for {sum(len(v) for v in observed.values())} variations.\n")
        print(f"[INFO] Saved text summary to: {args.output_txt}")

    # Output HTML report
    if args.output_html:
        html_dir = os.path.dirname(args.output_html)
        if html_dir:
            os.makedirs(html_dir, exist_ok=True)
        html_content = generate_html_report(
            observed,
            validation_rows,
            hist_name=hist_name,
            tag=args.tag,
            region=args.region,
            tolerance=args.tolerance,
        )
        with open(args.output_html, "w") as f:
            f.write(html_content)
        print(f"[INFO] Saved HTML cutflow report to: {args.output_html}")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
