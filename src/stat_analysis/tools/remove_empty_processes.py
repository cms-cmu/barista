#!/usr/bin/env python3
# coding: utf-8

"""
Standalone script to remove processes from datacard bins whose rate is below a given threshold.
Cleans up the process rows, rates, and all columnar nuisance parameter values.
Zero dependencies outside the Python standard library.

Example usage:
    # Remove any process with rate < 0.001 from all bins
    python3 src/stat_analysis/tools/remove_empty_processes.py datacard.txt --threshold 0.001 -o cleaned.txt

    # Remove process 'vjets' from 'pass.*' bins if rate < 0.1
    python3 src/stat_analysis/tools/remove_empty_processes.py datacard.txt --bin-pattern "pass.*" --process-pattern "vjets" --threshold 0.1 -o cleaned.txt
"""

import os
import sys
import re
import fnmatch
import argparse
from collections import OrderedDict


def remove_empty_processes(datacard_path, output_path=None, threshold=0.0, bin_pattern="*", process_pattern="*"):
    input_path = os.path.abspath(datacard_path)
    output_path = os.path.abspath(output_path) if output_path else input_path

    with open(input_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    # Locate rate table block
    # Follows: bin, process, process, rate
    rate_indices = []
    for i, line in enumerate(lines[:-3]):
        if (
            line.startswith("bin ") and
            lines[i+1].startswith("process ") and
            lines[i+2].startswith("process ") and
            lines[i+3].startswith("rate ")
        ):
            rate_indices = [i, i+1, i+2, i+3]
            break

    if not rate_indices:
        print("Could not find standard 4-line rate table (bin / process / process / rate). Card unchanged.")
        return

    b_idx, p_idx, pnum_idx, r_idx = rate_indices
    bins = lines[b_idx].split()[1:]
    processes = lines[p_idx].split()[1:]
    proc_nums = lines[pnum_idx].split()[1:]
    rates = lines[r_idx].split()[1:]

    n_cols = len(bins)
    if not (len(processes) == len(proc_nums) == len(rates) == n_cols):
        print(f"Column count mismatch: bins={len(bins)}, proc={len(processes)}, rates={len(rates)}. Card unchanged.")
        return

    keep_cols = []
    removed_pairs = []

    for col in range(n_cols):
        b = bins[col]
        p = processes[col]
        try:
            r = float(rates[col])
        except ValueError:
            r = 0.0

        matches_filter = fnmatch.fnmatch(b, bin_pattern) and fnmatch.fnmatch(p, process_pattern)
        if matches_filter and r <= threshold:
            removed_pairs.append((b, p, r))
        else:
            keep_cols.append(col)

    if not removed_pairs:
        print(f"No processes matched threshold <= {threshold}. Datacard unchanged.")
        return

    print(f"Removing {len(removed_pairs)} bin-process pairs with rate <= {threshold}:")
    for b, p, r in removed_pairs:
        print(f"  - Bin: {b:<25} Process: {p:<25} Rate: {r}")

    # Reconstruct rate block
    new_bins = [bins[c] for c in keep_cols]
    new_procs = [processes[c] for c in keep_cols]
    new_pnums = [proc_nums[c] for c in keep_cols]
    new_rates = [rates[c] for c in keep_cols]

    lines[b_idx] = "bin         " + "  ".join(new_bins)
    lines[p_idx] = "process     " + "  ".join(new_procs)
    lines[pnum_idx] = "process     " + "  ".join(new_pnums)
    lines[r_idx] = "rate        " + "  ".join(new_rates)

    # Update jmax count
    unique_bkg = set(p for p, num in zip(new_procs, new_pnums) if int(num) > 0)
    for i, line in enumerate(lines[:b_idx]):
        if line.startswith("jmax "):
            lines[i] = f"jmax {len(unique_bkg)}"

    # Update columnar systematics (lines after rate block)
    columnar_directives = ["lnN", "lnU", "gmN", "shape", "shape*"]
    for i in range(r_idx + 1, len(lines)):
        words = lines[i].split()
        if len(words) >= 2 and any(words[1].startswith(d) for d in columnar_directives):
            param_name = words[0]
            param_type = words[1]
            param_vals = words[2:]
            if len(param_vals) == n_cols:
                new_vals = [param_vals[c] for c in keep_cols]
                # If all values are '-', we could keep or drop; keeping ensures schema preservation
                lines[i] = f"{param_name:<35} {param_type:<8} " + "  ".join(new_vals)

    with open(output_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Cleaned datacard written to: {output_path} ({len(keep_cols)} columns remaining, originally {n_cols})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to input datacard .txt")
    parser.add_argument("-o", "--output", help="Path to output datacard (defaults to in-place update)")
    parser.add_argument("-t", "--threshold", type=float, default=0.0, help="Rate threshold to remove (default: 0.0)")
    parser.add_argument("-b", "--bin-pattern", default="*", help="Bin wildcard pattern (default: '*')")
    parser.add_argument("-p", "--process-pattern", default="*", help="Process wildcard pattern (default: '*')")

    args = parser.parse_args()
    remove_empty_processes(
        args.input,
        output_path=args.output,
        threshold=args.threshold,
        bin_pattern=args.bin_pattern,
        process_pattern=args.process_pattern,
    )


if __name__ == "__main__":
    main()
