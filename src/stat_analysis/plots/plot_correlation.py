#!/usr/bin/env python3
# coding: utf-8

"""
Script to plot post-fit nuisance parameter correlation matrices from Combine fits.
Supports input directly from FitDiagnostics ROOT files (RooFitResult) or extracted JSON files.
Filters parameters to show only those with significant correlation (|rho| >= threshold).

Example usage:
    python3 src/stat_analysis/plots/plot_correlation.py -i fitDiagnostics.root -o correlation_matrix.pdf --threshold 0.3
    python3 src/stat_analysis/plots/plot_correlation.py -i fit_result.json -o correlation_matrix.pdf --top 25
"""

import os
import sys
import json
import argparse
import numpy as np


def load_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    labels = data.get("matrix_labels", [])
    corr = np.array(data.get("correlation", []), dtype=float)
    return labels, corr


def load_from_root(root_path, obj_name="fit_s"):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    f = ROOT.TFile.Open(os.path.abspath(root_path), "READ")
    if not f or f.IsZombie():
        raise FileNotFoundError(f"Cannot open ROOT file: {root_path}")
    res = f.Get(obj_name)
    if not res or not isinstance(res, ROOT.RooFitResult):
        f.Close()
        raise ValueError(f"RooFitResult '{obj_name}' not found in {root_path}")

    pars = res.floatParsFinal()
    labels = [pars[i].GetName() for i in range(len(pars))]
    matrix = res.correlationMatrix()
    cls = matrix.ClassName()
    dtype = np.float64 if "double" in cls.lower() else np.float32
    corr = np.frombuffer(matrix.GetMatrixArray(), dtype=dtype, count=matrix.GetNoElements()).reshape((matrix.GetNrows(), matrix.GetNcols()))
    f.Close()
    return labels, corr


def plot_correlation_matrix(
    input_file,
    output_path,
    obj_name="fit_s",
    threshold=0.3,
    top_n=30,
    keep_patterns=None,
    skip_patterns=None,
    title=None,
):
    if input_file.endswith(".json"):
        labels, corr = load_from_json(input_file)
    else:
        labels, corr = load_from_root(input_file, obj_name)

    if len(labels) == 0 or corr.size == 0:
        print("Warning: Correlation matrix is empty.")
        return

    # Filter out parameters by keep/skip patterns if requested
    import fnmatch
    valid_indices = []
    for idx, lbl in enumerate(labels):
        keep = True
        if keep_patterns:
            keep = any(fnmatch.fnmatch(lbl, p) for p in keep_patterns)
        if keep and skip_patterns:
            keep = not any(fnmatch.fnmatch(lbl, p) for p in skip_patterns)
        if keep:
            valid_indices.append(idx)

    labels = [labels[i] for i in valid_indices]
    corr = corr[np.ix_(valid_indices, valid_indices)]

    # Find parameters that have at least one off-diagonal correlation >= threshold
    n = len(labels)
    off_diag_mask = ~np.eye(n, dtype=bool)
    max_off_diag = np.max(np.abs(corr * off_diag_mask), axis=1) if n > 1 else np.zeros(n)

    sig_indices = [i for i in range(n) if max_off_diag[i] >= threshold]

    if len(sig_indices) == 0:
        print(f"No parameters with off-diagonal |rho| >= {threshold}. Using top {min(top_n, n)} correlated parameters.")
        sig_indices = np.argsort(max_off_diag)[::-1][:min(top_n, n)].tolist()
    elif len(sig_indices) > top_n:
        print(f"Found {len(sig_indices)} parameters above threshold {threshold}; selecting top {top_n}.")
        sorted_by_max = sorted(sig_indices, key=lambda i: max_off_diag[i], reverse=True)
        sig_indices = sorted_by_max[:top_n]

    # Re-order alphabetically or by correlation
    sig_indices = sorted(sig_indices, key=lambda i: labels[i])
    sub_labels = [labels[i] for i in sig_indices]
    sub_corr = corr[np.ix_(sig_indices, sig_indices)]

    # Plot using matplotlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dim = max(8, len(sub_labels) * 0.45 + 3)
    fig, ax = plt.subplots(figsize=(fig_dim, fig_dim * 0.9))

    cmap = plt.cm.get_cmap("coolwarm").copy()
    im = ax.imshow(sub_corr, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Linear Correlation Coefficient", rotation=270, labelpad=20, fontsize=12)

    ax.set_xticks(np.arange(len(sub_labels)))
    ax.set_yticks(np.arange(len(sub_labels)))
    ax.set_xticklabels(sub_labels, rotation=90, ha="right", fontsize=9)
    ax.set_yticklabels(sub_labels, fontsize=9)

    # Annotate values
    for i in range(len(sub_labels)):
        for j in range(len(sub_labels)):
            val = sub_corr[i, j]
            if abs(val) >= 0.05:
                text_color = "white" if abs(val) > 0.65 else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", color=text_color, fontsize=7)

    plot_title = title or f"Post-fit Correlation Matrix ({obj_name}, |#rho| #geq {threshold})"
    ax.set_title(plot_title, fontsize=13, pad=15)
    plt.tight_layout()

    out_base = os.path.splitext(output_path)[0]
    out_pdf = f"{out_base}.pdf"
    out_png = f"{out_base}.png"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved correlation plots:\n  {out_pdf}\n  {out_png}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="Input ROOT (fitDiagnostics) or JSON file")
    parser.add_argument("-o", "--output", required=True, help="Output plot path (.pdf or .png)")
    parser.add_argument("--fit", default="fit_s", help="Fit result name in ROOT file ('fit_s' or 'fit_b')")
    parser.add_argument("--threshold", type=float, default=0.3, help="Min |correlation| to display")
    parser.add_argument("--top", type=int, default=30, help="Max parameters to display")
    parser.add_argument("--keep", nargs="+", help="Wildcard patterns to keep")
    parser.add_argument("--skip", nargs="+", help="Wildcard patterns to skip")
    parser.add_argument("--title", help="Custom plot title")

    args = parser.parse_args()
    plot_correlation_matrix(
        input_file=args.input,
        output_path=args.output,
        obj_name=args.fit,
        threshold=args.threshold,
        top_n=args.top,
        keep_patterns=args.keep,
        skip_patterns=args.skip,
        title=args.title,
    )


if __name__ == "__main__":
    main()
