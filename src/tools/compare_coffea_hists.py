#!/usr/bin/env python
"""Compare histograms from two .coffea files (e.g. Phase F vs Phase C.4).

Verifies that control kinematic distributions (e.g. selJets.n, v4j.mass) produced
in Phase F match Phase C.4 exactly in unblinded regions (e.g. Sideband).

Usage:
    python src/tools/compare_coffea_hists.py file_f.coffea file_c.coffea \
        -o hist_crosscheck.html --txt hist_crosscheck.txt \
        --label-a "Phase F" --label-b "Phase C.4" \
        --variables selJets.n,v4j.mass --regions SB --tolerance 1e-4

Exit codes:
    0: Test executed (PASS, or FAIL without --strict)
    1: FAIL with --strict, or fatal error
"""
from __future__ import annotations

import argparse
import html
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("compare_coffea_hists")


def _find_var_axis(hist_obj) -> Optional[str]:
    """Find the primary non-metadata variable axis of a 1D histogram."""
    meta_axes = {"process", "year", "tag", "region", "variation", "systematic"}
    cand_axes = [
        a.name for a in hist_obj.axes
        if a.name not in meta_axes and type(a).__name__ != "Boolean"
    ]
    if cand_axes:
        return cand_axes[0]
    # Fallback to the last axis
    return hist_obj.axes[-1].name if len(hist_obj.axes) > 0 else None


def _get_axis_values(hist_obj, axis_name: str) -> list:
    """Safely get values of an axis by name across Hist and mock containers."""
    try:
        return list(hist_obj.axes[axis_name])
    except Exception:
        for a in hist_obj.axes:
            if getattr(a, "name", None) == axis_name:
                return list(a)
    return []


def _project_1d(hist_obj, proc: str, tag: str, reg: str):
    """Slice histogram by process, tag, region and project onto the 1D variable axis."""
    loc_dict = {}
    axis_names = [a.name for a in hist_obj.axes]

    if "process" in axis_names:
        procs = _get_axis_values(hist_obj, "process")
        if proc in procs:
            loc_dict["process"] = proc
        else:
            return None
    if "tag" in axis_names:
        tags = _get_axis_values(hist_obj, "tag")
        if tag in tags:
            loc_dict["tag"] = tag
        else:
            return None
    if "region" in axis_names:
        regs = _get_axis_values(hist_obj, "region")
        if reg in regs:
            loc_dict["region"] = reg
        else:
            return None
    if "variation" in axis_names:
        vars_list = _get_axis_values(hist_obj, "variation")
        if "nominal" in vars_list:
            loc_dict["variation"] = "nominal"
    if "systematic" in axis_names:
        systs_list = _get_axis_values(hist_obj, "systematic")
        if "nominal" in systs_list:
            loc_dict["systematic"] = "nominal"

    try:
        sliced = hist_obj[loc_dict]
    except Exception as e:
        logger.debug(f"Failed slicing {loc_dict}: {e}")
        return None

    var_axis = _find_var_axis(sliced)
    if not var_axis:
        return None

    try:
        # Projecting down to 1D variable axis
        proj = sliced.project(var_axis)
        return proj
    except Exception as e:
        logger.debug(f"Failed projecting onto {var_axis}: {e}")
        return None


def compare_hist_objects(
    h_a,
    h_b,
    var_name: str,
    processes: List[str],
    tags: List[str],
    regions: List[str],
    tolerance: float = 1e-4,
) -> List[Dict[str, Any]]:
    """Compare two histogram objects across all process/tag/region combinations."""
    import numpy as np

    results = []

    for proc in processes:
        for tag in tags:
            for reg in regions:
                entry = {
                    "variable": var_name,
                    "process": proc,
                    "tag": tag,
                    "region": reg,
                    "status": "SKIP",
                    "reason": "",
                    "yield_a": 0.0,
                    "yield_b": 0.0,
                    "max_diff": 0.0,
                    "rel_diff": 0.0,
                    "n_bins": 0,
                }

                p_a = _project_1d(h_a, proc, tag, reg)
                p_b = _project_1d(h_b, proc, tag, reg)

                if p_a is None or p_b is None:
                    missing = []
                    if p_a is None:
                        missing.append("A")
                    if p_b is None:
                        missing.append("B")
                    entry["reason"] = f"Slice not found in {','.join(missing)}"
                    results.append(entry)
                    continue

                vals_a = np.asarray(p_a.values())
                vals_b = np.asarray(p_b.values())

                if vals_a.shape != vals_b.shape:
                    entry["status"] = "FAIL"
                    entry["reason"] = f"Shape mismatch: {vals_a.shape} vs {vals_b.shape}"
                    results.append(entry)
                    continue

                y_a = float(np.sum(vals_a))
                y_b = float(np.sum(vals_b))
                abs_diff = np.abs(vals_a - vals_b)
                max_diff = float(np.max(abs_diff)) if len(abs_diff) > 0 else 0.0

                denom = max(abs(y_a), abs(y_b), 1e-9)
                rel_diff = abs(y_a - y_b) / denom

                entry["yield_a"] = y_a
                entry["yield_b"] = y_b
                entry["max_diff"] = max_diff
                entry["rel_diff"] = rel_diff
                entry["n_bins"] = len(vals_a)

                # Check numerical agreement
                close = np.allclose(vals_a, vals_b, rtol=tolerance, atol=tolerance)
                if close:
                    entry["status"] = "PASS"
                else:
                    entry["status"] = "FAIL"
                    entry["reason"] = f"Max diff {max_diff:.3e} exceeds tolerance {tolerance:.3e}"

                results.append(entry)

    return results


def run_comparison(
    file_a: str,
    file_b: str,
    variables: List[str],
    regions: List[str],
    tags: Optional[List[str]] = None,
    processes: Optional[List[str]] = None,
    tolerance: float = 1e-4,
) -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
    """Loads coffea files and runs comparison across variables and categories."""
    try:
        from coffea.util import load
    except ImportError:
        import pickle
        def load(p):
            with open(p, "rb") as f:
                return pickle.load(f)

    if tags is None:
        tags = ["threeTag", "fourTag"]

    data_a = load(file_a)
    data_b = load(file_b)

    hists_a = data_a.get("hists", data_a) if isinstance(data_a, dict) else data_a
    hists_b = data_b.get("hists", data_b) if isinstance(data_b, dict) else data_b

    all_results = []
    meta = {
        "file_a": file_a,
        "file_b": file_b,
        "tolerance": tolerance,
        "variables": variables,
        "regions": regions,
        "tags": tags,
    }

    # Discover shared processes if not provided
    if processes is None:
        # Check first variable in hists_a
        procs_a = set()
        for v in variables:
            if v in hists_a and hasattr(hists_a[v], "axes") and "process" in [a.name for a in hists_a[v].axes]:
                procs_a.update(_get_axis_values(hists_a[v], "process"))
        procs_b = set()
        for v in variables:
            if v in hists_b and hasattr(hists_b[v], "axes") and "process" in [a.name for a in hists_b[v].axes]:
                procs_b.update(_get_axis_values(hists_b[v], "process"))
        shared_procs = sorted(procs_a & procs_b)
        # Prioritize data and ttbar
        preferred = ["data", "TTbar4b_from_d3", "TTbar3b_from_d3"]
        processes = [p for p in preferred if p in shared_procs] + [p for p in shared_procs if p not in preferred]
        if not processes:
            processes = ["data"]

    meta["processes"] = processes

    for var in variables:
        if var not in hists_a or var not in hists_b:
            missing = []
            if var not in hists_a:
                missing.append("A")
            if var not in hists_b:
                missing.append("B")
            all_results.append({
                "variable": var,
                "process": "*",
                "tag": "*",
                "region": "*",
                "status": "SKIP",
                "reason": f"Variable missing from {','.join(missing)}",
                "yield_a": 0.0,
                "yield_b": 0.0,
                "max_diff": 0.0,
                "rel_diff": 0.0,
                "n_bins": 0,
            })
            continue

        h_a = hists_a[var]
        h_b = hists_b[var]
        var_res = compare_hist_objects(
            h_a, h_b, var, processes, tags, regions, tolerance=tolerance
        )
        all_results.extend(var_res)

    has_fail = any(r["status"] == "FAIL" for r in all_results)
    passed_entries = [r for r in all_results if r["status"] == "PASS"]
    overall_pass = (not has_fail) and (len(passed_entries) > 0)

    return overall_pass, all_results, meta


def generate_text_report(
    overall_pass: bool,
    results: List[Dict[str, Any]],
    meta: Dict[str, Any],
    label_a: str = "Phase F",
    label_b: str = "Phase C.4",
) -> str:
    """Generate plaintext summary table."""
    status_str = "PASS" if overall_pass else "FAIL"
    lines = [
        f"{status_str}: Histogram consistency cross-check between {label_a} and {label_b}",
        f"File A ({label_a}): {meta.get('file_a', '')}",
        f"File B ({label_b}): {meta.get('file_b', '')}",
        f"Tolerance: {meta.get('tolerance', 1e-4)}",
        "",
        f"{'Variable':<15} {'Process':<18} {'Tag':<10} {'Region':<8} {'Status':<6} {'Yield A':<12} {'Yield B':<12} {'Max Diff':<12} {'Reason'}",
        "-" * 105,
    ]
    for r in results:
        v = r["variable"]
        p = r["process"]
        t = r["tag"]
        reg = r["region"]
        st = r["status"]
        ya = f"{r['yield_a']:.2f}" if r["n_bins"] > 0 else "—"
        yb = f"{r['yield_b']:.2f}" if r["n_bins"] > 0 else "—"
        md = f"{r['max_diff']:.2e}" if r["n_bins"] > 0 else "—"
        reason = r.get("reason", "")
        lines.append(f"{v:<15} {p:<18} {t:<10} {reg:<8} {st:<6} {ya:<12} {yb:<12} {md:<12} {reason}")

    return "\n".join(lines) + "\n"


def generate_html_report(
    overall_pass: bool,
    results: List[Dict[str, Any]],
    meta: Dict[str, Any],
    label_a: str = "Phase F",
    label_b: str = "Phase C.4",
    title: Optional[str] = None,
) -> str:
    """Generate rich standalone HTML dashboard."""
    doc_title = title or f"Histogram Cross-Check — {label_a} vs {label_b}"
    badge_cls = "badge-success" if overall_pass else "badge-danger"
    badge_text = "● PASS" if overall_pass else "● FAIL"
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    skip_count = sum(1 for r in results if r["status"] == "SKIP")

    rows = []
    for r in results:
        st = r["status"]
        st_cls = "chip-pass" if st == "PASS" else ("chip-danger" if st == "FAIL" else "chip-neutral")
        ya = f"{r['yield_a']:.4f}" if r["n_bins"] > 0 else "—"
        yb = f"{r['yield_b']:.4f}" if r["n_bins"] > 0 else "—"
        md = f"{r['max_diff']:.2e}" if r["n_bins"] > 0 else "—"
        rd = f"{r['rel_diff']:.2e}" if r["n_bins"] > 0 else "—"
        reason = html.escape(str(r.get("reason", "")))

        rows.append(f"""
        <tr>
            <td><strong><code>{html.escape(r['variable'])}</code></strong></td>
            <td><code>{html.escape(r['process'])}</code></td>
            <td>{html.escape(r['tag'])}</td>
            <td><span class="badge badge-host">{html.escape(r['region'])}</span></td>
            <td><span class="chip {st_cls}">{st}</span></td>
            <td>{ya}</td>
            <td>{yb}</td>
            <td><code>{md}</code></td>
            <td><code>{rd}</code></td>
            <td class="muted">{reason}</td>
        </tr>""")

    table_rows = "\n".join(rows) if rows else "<tr><td colspan='10' class='muted'>No results</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{html.escape(doc_title)}</title>
    <style>
        :root {{
            --bg: #f8fafc;
            --fg: #0f172a;
            --muted: #64748b;
            --card: #ffffff;
            --line: #e2e8f0;
            --accent: #2563eb;
            --accent-soft: #eff6ff;
            --pass-bg: #dcfce7;
            --pass-fg: #15803d;
            --pass-border: #bbf7d0;
            --danger-bg: #fee2e2;
            --danger-fg: #b91c1c;
            --danger-border: #fecaca;
            --warn-bg: #fef3c7;
            --warn-fg: #b45309;
            --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--fg);
            font-family: var(--sans);
            font-size: 14px;
            line-height: 1.5;
            padding: 24px 32px 64px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; display: flex; flex-direction: column; gap: 16px; }}
        header {{
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .header-main h1 {{
            font-size: 1.3rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .header-sub {{ color: var(--muted); margin-top: 4px; font-size: 13px; }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}
        .badge-success {{ background: var(--pass-bg); color: var(--pass-fg); border: 1px solid var(--pass-border); }}
        .badge-danger {{ background: var(--danger-bg); color: var(--danger-fg); border: 1px solid var(--danger-border); }}
        .badge-host {{ background: #f1f5f9; color: var(--fg); border: 1px solid var(--line); font-family: var(--mono); text-transform: none; }}
        .card {{
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }}
        .card-header {{
            padding: 12px 18px;
            border-bottom: 1px solid var(--line);
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #fafbfc;
        }}
        .card-header h2 {{ font-size: 15px; font-weight: 600; }}
        .card-body {{ padding: 18px; }}
        .grid-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }}
        .stat-box {{
            border: 1px solid var(--line);
            border-radius: 6px;
            padding: 12px 16px;
            background: #ffffff;
        }}
        .stat-val {{ font-size: 20px; font-weight: 700; }}
        .stat-label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; font-weight: 600; }}
        table.data-table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }}
        table.data-table th, table.data-table td {{ padding: 8px 12px; border-bottom: 1px solid var(--line); }}
        table.data-table th {{ background: #f8fafc; color: var(--muted); font-size: 11.5px; text-transform: uppercase; font-weight: 600; }}
        table.data-table tr:hover td {{ background: #f8fafc; }}
        .chip {{ display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 4px; font-size: 11.5px; font-weight: 600; }}
        .chip-pass {{ background: var(--pass-bg); color: var(--pass-fg); border: 1px solid var(--pass-border); }}
        .chip-danger {{ background: var(--danger-bg); color: var(--danger-fg); border: 1px solid var(--danger-border); }}
        .chip-neutral {{ background: #f1f5f9; color: var(--muted); border: 1px solid var(--line); }}
        code {{ font-family: var(--mono); font-size: 0.9em; background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }}
        .muted {{ color: var(--muted); }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-main">
                <h1>{html.escape(doc_title)}</h1>
                <div class="header-sub">
                    Direct validation of control kinematics ({", ".join(html.escape(v) for v in meta.get('variables', []))}) between {html.escape(label_a)} and {html.escape(label_b)}
                </div>
            </div>
            <div>
                <span class="badge {badge_cls}">{badge_text}</span>
            </div>
        </header>

        <div class="card">
            <div class="card-body">
                <div class="grid-stats">
                    <div class="stat-box">
                        <div class="stat-label">Passed Slices</div>
                        <div class="stat-val" style="color: var(--pass-fg);">{pass_count}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Failed Slices</div>
                        <div class="stat-val" style="color: {'var(--danger-fg)' if fail_count > 0 else 'var(--muted)'};">{fail_count}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Skipped Slices</div>
                        <div class="stat-val" style="color: var(--muted);">{skip_count}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Tolerance</div>
                        <div class="stat-val"><code>{meta.get('tolerance', 1e-4)}</code></div>
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <h2>Comparison Details by Variable & Slice</h2>
                <span class="muted">Regions: {", ".join(meta.get('regions', []))}</span>
            </div>
            <div class="card-body" style="padding: 0; overflow-x: auto;">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Variable</th>
                            <th>Process</th>
                            <th>Tag</th>
                            <th>Region</th>
                            <th>Status</th>
                            <th>Yield ({html.escape(label_a)})</th>
                            <th>Yield ({html.escape(label_b)})</th>
                            <th>Max Bin Diff</th>
                            <th>Rel Diff</th>
                            <th>Notes</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_a", help="Primary .coffea file (e.g. Phase F)")
    parser.add_argument("file_b", help="Reference .coffea file (e.g. Phase C.4)")
    parser.add_argument("-o", "--output", default="hist_crosscheck.html", help="Path to output HTML report")
    parser.add_argument("--txt", default="hist_crosscheck.txt", help="Path to output TXT report")
    parser.add_argument("--label-a", default="Phase F", help="Label for file A")
    parser.add_argument("--label-b", default="Phase C.4", help="Label for file B")
    parser.add_argument("--title", default=None, help="Report title")
    parser.add_argument("--variables", default="selJets.n,v4j.mass", help="Comma-separated variables to compare")
    parser.add_argument("--regions", default="SB", help="Comma-separated regions to compare (default: SB)")
    parser.add_argument("--tags", default="threeTag,fourTag", help="Comma-separated tags to compare")
    parser.add_argument("--processes", default=None, help="Comma-separated processes to compare (default: auto)")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Tolerance for allclose comparison")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if FAIL")

    args = parser.parse_args()

    vars_list = [v.strip() for v in args.variables.replace("+", ",").split(",") if v.strip()]
    regs_list = [r.strip() for r in args.regions.replace("+", ",").split(",") if r.strip()]
    tags_list = [t.strip() for t in args.tags.replace("+", ",").split(",") if t.strip()]
    procs_list = [p.strip() for p in args.processes.replace("+", ",").split(",") if p.strip()] if args.processes else None

    if not os.path.exists(args.file_a):
        logger.error(f"File A not found: {args.file_a}")
        sys.exit(1)
    if not os.path.exists(args.file_b):
        logger.error(f"File B not found: {args.file_b}")
        sys.exit(1)

    overall_pass, results, meta = run_comparison(
        args.file_a,
        args.file_b,
        variables=vars_list,
        regions=regs_list,
        tags=tags_list,
        processes=procs_list,
        tolerance=args.tolerance,
    )

    txt_content = generate_text_report(overall_pass, results, meta, label_a=args.label_a, label_b=args.label_b)
    if args.txt:
        os.makedirs(os.path.dirname(os.path.abspath(args.txt)), exist_ok=True)
        with open(args.txt, "w") as f:
            f.write(txt_content)
        print(f"Wrote text report to {args.txt}")

    html_content = generate_html_report(overall_pass, results, meta, label_a=args.label_a, label_b=args.label_b, title=args.title)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(html_content)
        print(f"Wrote HTML report to {args.output}")

    print(txt_content.splitlines()[0])

    if args.strict and not overall_pass:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
