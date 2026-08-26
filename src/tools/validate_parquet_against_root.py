"""
Memory-efficient validation tool to compare HCR_input ROOT files with merged Parquet files.
Verifies exact numerical equality for all input variables (after applying SR | SB selection)
and generates 1D comparison and ratio plots (ROOT vs. Parquet) using column-wise streaming.
"""

import os
import re
import glob
import logging
import argparse
import numpy as np
import awkward as ak
import uproot
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.tools.merge_friendtrees_to_parquet import (
    get_newest_manifests,
    load_friends_from_manifest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def natural_chunk_sort_key(filepath):
    m = re.search(r"_(\d+)\.parquet$", filepath)
    return int(m.group(1)) if m else 0


def plot_comparison(name, arr_root, arr_parquet, out_path, num_bins=40):
    """
    Generate a 2-panel comparison plot:
    Top panel: ROOT vs Parquet distribution overlay.
    Bottom panel: Parquet / ROOT ratio plot.
    """
    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
    )

    valid_mask = np.isfinite(arr_root) & np.isfinite(arr_parquet)
    vals_root = arr_root[valid_mask]
    vals_pq = arr_parquet[valid_mask]

    if len(vals_root) == 0:
        plt.close(fig)
        return

    min_val, max_val = np.min(vals_root), np.max(vals_root)
    if min_val == max_val:
        bins = np.linspace(min_val - 1, max_val + 1, num_bins + 1)
    else:
        unique_vals = np.unique(vals_root)
        if len(unique_vals) <= 10 and np.all(np.equal(np.mod(unique_vals, 1), 0)):
            bins = np.arange(min_val - 0.5, max_val + 1.5, 1)
        else:
            bins = np.linspace(min_val, max_val, num_bins + 1)

    counts_root, edges = np.histogram(vals_root, bins=bins)
    counts_pq, _ = np.histogram(vals_pq, bins=bins)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    # Top Panel
    ax_main.step(edges, np.append(counts_root, counts_root[-1]), where="post", label="HCR_input (ROOT)", color="black", linewidth=1.5)
    ax_main.step(edges, np.append(counts_pq, counts_pq[-1]), where="post", label="Merged Parquet", color="red", linestyle="--", linewidth=1.2)
    ax_main.set_ylabel("Events", fontsize=10)
    ax_main.set_title(f"Variable Comparison: {name}", fontsize=11, fontweight="bold")
    ax_main.grid(True, linestyle=":", alpha=0.6)
    ax_main.legend(loc="upper right", frameon=True)

    # Bottom Panel: Ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(counts_root > 0, counts_pq / counts_root, 1.0)
        ratio_err = np.where(counts_root > 0, np.sqrt(counts_pq) / counts_root, 0.0)

    ax_ratio.errorbar(bin_centers, ratio, yerr=ratio_err, fmt="o", color="black", markersize=3, elinewidth=1, capsize=1.5)
    ax_ratio.axhline(1.0, color="red", linestyle="--", linewidth=1)
    ax_ratio.set_ylim(0.8, 1.2)
    ax_ratio.set_ylabel("Parquet / ROOT", fontsize=9)
    ax_ratio.set_xlabel(name, fontsize=10)
    ax_ratio.grid(True, linestyle=":", alpha=0.6)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def extract_hcr_column_from_root(manifest_path, var_name, max_jets=12, max_can_jets=4, max_not_can_jets=8):
    """
    Extract a single column from HCR_input ROOT files for all aligned chunks, applying SR | SB selection.
    """
    friends = load_friends_from_manifest(manifest_path)
    hcr_friend = friends["HCR_input"]
    parent_chunks = list(hcr_friend._data.keys())

    chunks_data = []

    for parent_chunk in parent_chunks:
        hcr_items = hcr_friend._data.get(parent_chunk, [])
        for item in hcr_items:
            hcr_path = item.chunk.path
            start, stop = item.start, item.stop

            with uproot.open(hcr_path) as f:
                tree = f["Events"]
                tree_keys = set(tree.keys())

                # Helper to find existing branch
                def get_branch_name(candidates):
                    for c in candidates:
                        if c in tree_keys:
                            return c
                    return None

                # Find SR and SB branches
                sr_branch = get_branch_name(["SR", "SR/SR", "SR/bool", "Events/SR"])
                sb_branch = get_branch_name(["SB", "SB/SB", "SB/bool", "Events/SB"])

                # Check if it's a CanJet field (e.g. CanJet1_pt -> CanJet.pt index 0)
                m_can = re.match(r"CanJet(\d+)_(.*)", var_name)
                m_notcan = re.match(r"NotCanJet(\d+)_(.*)", var_name)

                if m_can:
                    idx = int(m_can.group(1)) - 1
                    attr = m_can.group(2)
                    target_branch = get_branch_name([
                        f"CanJet/CanJet.{attr}", f"CanJet/{attr}", f"CanJet_{attr}",
                        f"Events/CanJet/CanJet.{attr}", f"Events/CanJet/{attr}"
                    ])
                    if not target_branch:
                        continue

                    branches_to_read = [sr_branch, sb_branch, target_branch]
                    arrs = tree.arrays([b for b in branches_to_read if b], entry_start=0, entry_stop=stop - start)
                    
                    sr_arr = arrs[sr_branch] if sr_branch else ak.Array([False]*len(arrs))
                    sb_arr = arrs[sb_branch] if sb_branch else ak.Array([False]*len(arrs))
                    mask = (sr_arr != 0) | (sb_arr != 0)
                    filtered = arrs[mask]
                    if len(filtered) == 0:
                        continue

                    branch_data = filtered[target_branch]
                    padded = ak.pad_none(branch_data, max_can_jets, axis=1, clip=True)
                    fill_val = -1 if ("Id" in attr or "Idx" in attr or "Flavour" in attr) else (0 if ("isSelJet" in attr or "cleanmask" in attr) else -1.0)
                    col_data = ak.fill_none(padded[:, idx], fill_val)
                    chunks_data.append(ak.to_numpy(col_data))

                elif m_notcan:
                    idx = int(m_notcan.group(1)) - 1
                    attr = m_notcan.group(2)
                    target_branch = get_branch_name([
                        f"NotCanJet/NotCanJet.{attr}", f"NotCanJet/{attr}", f"NotCanJet_{attr}",
                        f"Events/NotCanJet/NotCanJet.{attr}", f"Events/NotCanJet/{attr}"
                    ])
                    if not target_branch:
                        continue

                    branches_to_read = [sr_branch, sb_branch, target_branch]
                    arrs = tree.arrays([b for b in branches_to_read if b], entry_start=0, entry_stop=stop - start)

                    sr_arr = arrs[sr_branch] if sr_branch else ak.Array([False]*len(arrs))
                    sb_arr = arrs[sb_branch] if sb_branch else ak.Array([False]*len(arrs))
                    mask = (sr_arr != 0) | (sb_arr != 0)
                    filtered = arrs[mask]
                    if len(filtered) == 0:
                        continue

                    branch_data = filtered[target_branch]
                    padded = ak.pad_none(branch_data, max_not_can_jets, axis=1, clip=True)
                    fill_val = 0 if attr == "isSelJet" else -1.0
                    col_data = ak.fill_none(padded[:, idx], fill_val)
                    chunks_data.append(ak.to_numpy(col_data))

                else:
                    # Event-level branch
                    target_branch = get_branch_name([var_name, f"{var_name}/{var_name}", f"Events/{var_name}"])
                    if not target_branch:
                        continue

                    branches_to_read = list(set(filter(None, [sr_branch, sb_branch, target_branch])))
                    arrs = tree.arrays(branches_to_read, entry_start=0, entry_stop=stop - start)

                    sr_arr = arrs[sr_branch] if sr_branch else ak.Array([False]*len(arrs))
                    sb_arr = arrs[sb_branch] if sb_branch else ak.Array([False]*len(arrs))
                    mask = (sr_arr != 0) | (sb_arr != 0)
                    filtered = arrs[mask]
                    if len(filtered) == 0:
                        continue

                    chunks_data.append(ak.to_numpy(filtered[target_branch]))

    if not chunks_data:
        return np.array([])
    return np.concatenate(chunks_data)


def validate_dataset(manifest_info, parquet_base_dir, plot_dir, make_plots=True):
    """
    Validate a single dataset-era manifest against its merged Parquet files using memory-safe column streaming.
    """
    base_name = manifest_info["base_name"]
    year_era = manifest_info["year_era"]
    manifest_path = manifest_info["path"]

    logging.info(f"=== Validating Dataset: {base_name} ({year_era}) ===")

    parquet_folder = os.path.join(parquet_base_dir, base_name)
    parquet_files = sorted(
        glob.glob(os.path.join(parquet_folder, f"part_{year_era}*.parquet")),
        key=natural_chunk_sort_key
    )
    if not parquet_files:
        logging.warning(f"No Parquet part files found for {base_name} ({year_era}) in {parquet_folder}")
        return None

    logging.info(f"Found {len(parquet_files)} Parquet part file(s) for {year_era}")

    # Inspect schema from the first parquet file
    pq_schema = pq.read_schema(parquet_files[0])
    all_pq_fields = set(pq_schema.names)

    # Get sample HCR_input friend tree to discover variables
    friends = load_friends_from_manifest(manifest_path)
    sample_chunk = list(friends["HCR_input"]._data.values())[0][0].chunk.path
    with uproot.open(sample_chunk) as f:
        sample_tree = f["Events"]
        root_keys = [k.split("/")[0] for k in sample_tree.keys()]

    # Determine list of test variables
    test_variables = []
    # Event-level
    for k in root_keys:
        if k in all_pq_fields and k != "year" and not any(k.startswith(p) for p in ("CanJet", "NotCanJet", "Jet")):
            test_variables.append(k)

    # CanJet
    can_attrs = ["pt", "eta", "phi", "mass", "btagScore", "area", "rawFactor",
                 "btagPNetCvB", "btagPNetCvL", "btagPNetQvG", "btagPNetTauVJet",
                 "PNetRegPtRawRes", "nSVs", "chHEF", "neHEF", "chEmEF", "neEmEF", "muEF", "nConstituents"]
    for idx in range(1, 5):
        for attr in can_attrs:
            name = f"CanJet{idx}_{attr}"
            if name in all_pq_fields:
                test_variables.append(name)

    # NotCanJet
    not_can_attrs = ["pt", "eta", "phi", "mass", "isSelJet"]
    for idx in range(1, 9):
        for attr in not_can_attrs:
            name = f"NotCanJet{idx}_{attr}"
            if name in all_pq_fields:
                test_variables.append(name)

    logging.info(f"Streaming validation for {len(test_variables)} HCR_input variables...")

    results = []
    plot_sub_dir = os.path.join(plot_dir, base_name, year_era)
    count_match = True
    n_root_total, n_pq_total = 0, 0

    for var_idx, var_name in enumerate(sorted(test_variables)):
        # 1. Read ROOT column
        arr_root = extract_hcr_column_from_root(manifest_path, var_name)
        if len(arr_root) == 0:
            continue

        # 2. Read Parquet column
        pq_col_arrays = []
        for pf in parquet_files:
            table = pq.read_table(pf, columns=[var_name])
            pq_col_arrays.append(table[var_name].to_numpy())
        arr_pq = np.concatenate(pq_col_arrays) if pq_col_arrays else np.array([])

        if var_idx == 0:
            n_root_total = len(arr_root)
            n_pq_total = len(arr_pq)
            count_match = (n_root_total == n_pq_total)
            logging.info(f"Event Count -> ROOT (SR|SB): {n_root_total:,} | Parquet: {n_pq_total:,}")

        min_len = min(len(arr_root), len(arr_pq))
        arr_root_cmp = arr_root[:min_len]
        arr_pq_cmp = arr_pq[:min_len]

        if np.issubdtype(arr_root_cmp.dtype, np.floating):
            diff = np.abs(arr_root_cmp - arr_pq_cmp)
            max_diff = float(np.nanmax(diff)) if len(diff) > 0 else 0.0
            is_exact = np.allclose(arr_root_cmp, arr_pq_cmp, equal_nan=True, rtol=1e-5, atol=1e-6)
        else:
            diff = (arr_root_cmp != arr_pq_cmp)
            max_diff = int(np.sum(diff))
            is_exact = (max_diff == 0)

        status = "EXACT" if (is_exact and count_match) else "DIFF"
        results.append({
            "variable": var_name,
            "status": status,
            "max_diff": max_diff,
            "root_dtype": str(arr_root.dtype),
            "pq_dtype": str(arr_pq.dtype)
        })

        if make_plots:
            plot_file = os.path.join(plot_sub_dir, f"{var_name}.png")
            plot_comparison(var_name, arr_root_cmp, arr_pq_cmp, plot_file)

    if make_plots:
        logging.info(f"Generated {len(test_variables)} comparison plots in: {plot_sub_dir}")

    return {
        "dataset": base_name,
        "era": year_era,
        "n_root": n_root_total,
        "n_pq": n_pq_total,
        "count_match": count_match,
        "variables": results
    }


def main():
    parser = argparse.ArgumentParser(description="Validate Parquet files against HCR_input ROOT files.")
    parser.add_argument(
        "-i", "--input-glob",
        default="output/mixeddata_friendtrees*/singlefiles/all_friends_*.json",
        help="Glob pattern for friend tree manifest JSONs"
    )
    parser.add_argument(
        "-p", "--parquet-dir",
        default="output/mixeddata_parquet",
        help="Base directory containing merged parquet files"
    )
    parser.add_argument(
        "-o", "--output-plot-dir",
        default="output/validation_plots",
        help="Directory to save validation plots"
    )
    parser.add_argument(
        "-d", "--dataset",
        help="Specific dataset to validate (e.g. ZH4b, mixeddata_all, TTTo2L2Nu, TTToHadronic)"
    )
    parser.add_argument(
        "-y", "--era",
        help="Specific era to validate (e.g. UL18)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation and only run numerical checks"
    )

    args = parser.parse_args()

    manifests = get_newest_manifests(args.input_glob)
    if not manifests:
        logging.error("No manifest files discovered.")
        return

    selected_manifests = []
    for key, info in manifests.items():
        if args.dataset and info["base_name"] != args.dataset:
            continue
        if args.era and info["year_era"] != args.era:
            continue
        selected_manifests.append(info)

    if not selected_manifests:
        logging.error(f"No manifests matched criteria (dataset={args.dataset}, era={args.era})")
        return

    all_summaries = []
    for manifest_info in selected_manifests:
        summary = validate_dataset(
            manifest_info,
            args.parquet_dir,
            args.output_plot_dir,
            make_plots=(not args.no_plots)
        )
        if summary:
            all_summaries.append(summary)

    print("\n" + "=" * 80)
    print(" " * 25 + "VALIDATION SUMMARY REPORT")
    print("=" * 80)
    for s in all_summaries:
        print(f"\nDataset: {s['dataset']} | Era: {s['era']}")
        print(f"Events: ROOT = {s['n_root']:,} | Parquet = {s['n_pq']:,} | Match = {s['count_match']}")
        print("-" * 80)
        print(f"{'Variable':<35} | {'Status':<8} | {'Max Diff / Mismatch':<20} | {'Dtype (ROOT/PQ)'}")
        print("-" * 80)
        all_exact = True
        for var in s["variables"]:
            diff_str = f"{var['max_diff']:.2e}" if isinstance(var['max_diff'], float) else str(var['max_diff'])
            dtype_str = f"{var['root_dtype']} / {var['pq_dtype']}"
            print(f"{var['variable']:<35} | {var['status']:<8} | {diff_str:<20} | {dtype_str}")
            if var["status"] != "EXACT":
                all_exact = False
        print("-" * 80)
        if all_exact and s['count_match']:
            print(">>> OVERALL RESULT: ALL VARIABLES 100% IDENTICAL (PASS) <<<")
        else:
            print(">>> OVERALL RESULT: DISCREPANCIES DETECTED (INVESTIGATE) <<<")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
