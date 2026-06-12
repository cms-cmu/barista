#!/usr/bin/env python
"""Plot HCR classifier input features split by physics label.

Plot set 1 (--mode raw):      raw branches from friend tree ROOT files
Plot set 2 (--mode dataprep):  features after model.inputEmbed.dataPrep()

Usage:
  ./run_container classifier_cpu python src/classifier/plot_inputs.py \
      --mode raw \
      --checkpoint root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/classifier/SvB_lowpt/ \
      --metadata coffea4bees/metadata/datasets_HH4b_Run2/2024_v2/classifier_inputs_lowpt_wlowptJCM.json@@HCR_input_lowpt \
      --output-dir output/classifier_input_plots
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

# Add repo root to sys.path so `from src.xxx import ...` works
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import fsspec
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Sample directory name → physics label mapping
# --------------------------------------------------------------------------- #
# Built-in defaults for known analyses. Each entry maps a regex pattern
# (matched against sample directory names in the metadata paths) to a
# physics label. Extend this dict for new analyses, or pass --label-map
# to override at runtime with a JSON file.
BUILTIN_LABEL_MAPS = {
    "HH4b": [
        (r"^data_",              "multijet"),
        (r"^TTTo",               "ttbar"),
        (r"^ZZ4b",               "ZZ"),
        (r"^(ZH4b|ggZH4b)",     "ZH"),
        (r"^GluGluToHHTo4B",     "ggF"),
    ],
}


def _load_label_patterns(label_map_path, checkpoint_labels):
    """Build [(compiled_regex, label)] from a JSON file or built-in defaults.

    JSON format: {"regex_pattern": "label", ...}
    e.g. {"^data_": "multijet", "^TTTo": "ttbar"}

    If no --label-map is given, tries each built-in map and picks the one
    whose labels best overlap with the checkpoint's labels.
    """
    if label_map_path:
        with open(label_map_path) as f:
            mapping = json.load(f)
        return [(re.compile(pat), lbl) for pat, lbl in mapping.items()]

    # Auto-select best built-in map
    label_set = set(checkpoint_labels)
    best_name, best_overlap = None, -1
    for name, patterns in BUILTIN_LABEL_MAPS.items():
        map_labels = {lbl for _, lbl in patterns}
        overlap = len(label_set & map_labels)
        if overlap > best_overlap:
            best_name, best_overlap = name, overlap

    if best_name and best_overlap > 0:
        print(f"Using built-in label map: {best_name}")
        return [(re.compile(pat), lbl) for pat, lbl in BUILTIN_LABEL_MAPS[best_name]]

    print("Warning: no matching built-in label map found. Use --label-map to provide one.")
    return []


def sample_to_label(sample_name, patterns):
    for pattern, label in patterns:
        if pattern.search(sample_name):
            return label
    return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--mode", choices=["raw", "dataprep"], required=True,
        help="'raw' = plot input branches; 'dataprep' = plot after dataPrep transform",
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="path to model directory (contains *.pkl) or direct .pkl file",
    )
    p.add_argument(
        "--metadata", required=True,
        help="path@@key to the metadata JSON (e.g. path/to/file.json@@HCR_input_lowpt)",
    )
    p.add_argument(
        "--output-dir", default="output/classifier_input_plots",
        help="directory to save plots (local path or root:// EOS path)",
    )
    p.add_argument(
        "--max-files", type=int, default=2,
        help="max friend tree files to read per label (default: 2)",
    )
    p.add_argument("--nbins", type=int, default=50, help="histogram bins")
    p.add_argument(
        "--label-map", default=None,
        help='JSON file mapping regex patterns to labels, e.g. {"^data_": "multijet", "^TTTo": "ttbar"}. '
             "If not given, auto-selects from built-in maps based on checkpoint labels.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #
def load_checkpoint(path):
    """Load a checkpoint .pkl from a file or directory (picks latest).

    Returns the saved dict with keys: label, input, arch, model.
    """
    import torch

    if path.endswith(".pkl"):
        ckpt_path = path
    else:
        # Directory — list model_*.pkl files (exclude states.pkl etc.)
        if path.startswith("root://"):
            from src.storage.eos import EOS
            entries = EOS(path).ls()
            pkls = sorted(
                str(f) for f in entries
                if str(f).endswith(".pkl") and "/model_" in str(f)
            )
        else:
            pkls = sorted(str(p) for p in Path(path).glob("model_*.pkl"))
        if not pkls:
            sys.exit(f"No model_*.pkl checkpoints found in {path}")
        ckpt_path = pkls[0]

    print(f"Loading checkpoint: {ckpt_path}")
    with fsspec.open(ckpt_path, "rb") as f:
        saved = torch.load(f, map_location="cpu", weights_only=False)
    return saved


# --------------------------------------------------------------------------- #
# Metadata → {label: [friend_tree_paths]}
# --------------------------------------------------------------------------- #
def parse_metadata_arg(arg):
    """Split 'path/to/file.json@@key' into (path, key)."""
    parts = arg.split("@@", 1)
    if len(parts) != 2:
        sys.exit(f"--metadata must be path@@key, got: {arg}")
    return parts[0], parts[1]


def load_friend_files_by_label(metadata_path, meta_key, labels, max_files, label_patterns):
    """Read metadata JSON and group friend tree paths by physics label.

    Only returns labels present in the `labels` list (from checkpoint).
    """
    with open(metadata_path) as f:
        meta = json.load(f)[meta_key]

    label_set = set(labels)
    files = {label: [] for label in labels}
    for entry in meta["data"]:
        pico_path = entry[0]["path"]
        # extract sample dir name from path: .../something/<sample>/file.root
        parts = pico_path.rsplit("/", 2)
        if len(parts) < 3:
            continue
        sample_name = parts[-2]
        label = sample_to_label(sample_name, label_patterns)
        if label is None or label not in label_set:
            continue
        for chunk_info in entry[1]:
            files[label].append(chunk_info["chunk"]["path"])

    import random
    rng = random.Random(42)
    for label in files:
        rng.shuffle(files[label])

    if max_files:
        for label in files:
            files[label] = files[label][:max_files]
    return files


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    saved = load_checkpoint(args.checkpoint)

    labels = saved["label"]
    input_cfg = saved["input"]
    print(f"Labels: {labels}")
    print(f"Ancillary features: {input_cfg['feature_ancillary']}")

    label_patterns = _load_label_patterns(args.label_map, labels)

    meta_path, meta_key = parse_metadata_arg(args.metadata)
    files_by_label = load_friend_files_by_label(
        meta_path, meta_key, labels, args.max_files, label_patterns
    )
    for label, paths in files_by_label.items():
        print(f"  {label}: {len(paths)} files")

    if args.mode == "raw":
        plot_raw(args, saved, files_by_label)
    else:
        plot_dataprep(args, saved, files_by_label)


LABEL_COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]


def _branch_groups(input_cfg):
    """Build {group_name: [branch_names]} from checkpoint input config.

    The checkpoint stores branch names already prefixed (e.g. "CanJet_pt",
    not "pt"), because InputBranch.get__feature_CanJet applies the prefix
    before saving.
    """
    return {
        "CanJet": list(input_cfg["feature_CanJet"]),
        "NotCanJet": list(input_cfg["feature_NotCanJet"]),
        "Ancillary": list(input_cfg["feature_ancillary"]),
    }


def read_branches(file_paths, branches, n_canjet=4, n_notcanjet=8):
    """Read branches from friend tree ROOT files, return {branch: np.array}.

    Multi-jet branches are padded and returned as 2D arrays (nevents, n_jets)
    so that per-jet distributions can be plotted.
    Scalar branches are returned as 1D arrays.
    Events with fewer than n_canjet CanJets are filtered out (they didn't
    pass event selection and have empty jet arrays in the friend tree).
    """
    import uproot
    import awkward as ak

    arrays = {b: [] for b in branches}
    for path in file_paths:
        try:
            with uproot.open(path + ":Events") as tree:
                # Filter: only events with enough CanJets (others failed selection)
                ncanjet = tree["nCanJet"].array(library="np")
                mask = ncanjet >= n_canjet

                for branch in branches:
                    if branch == "year":
                        match = re.search(r"UL(\d{2})", path)
                        if not match:
                            match = re.search(r"20(\d{2})", path)
                        if match:
                            y_val = float(match.group(1))
                        else:
                            raise ValueError(f"Could not parse year from path: {path}")
                        n_events = np.sum(mask)
                        arr = np.full(n_events, y_val, dtype=np.float32)
                        arrays[branch].append(arr)
                    else:
                        arr = tree[branch].array(library="ak")[mask]
                        if "CanJet" in branch and "NotCanJet" not in branch:
                            padded = ak.fill_none(ak.pad_none(arr, n_canjet, clip=True), np.nan)
                            arrays[branch].append(ak.to_numpy(padded))
                        elif "NotCanJet" in branch:
                            padded = ak.fill_none(ak.pad_none(arr, n_notcanjet, clip=True), np.nan)
                            arrays[branch].append(ak.to_numpy(padded))
                        else:
                            arrays[branch].append(ak.to_numpy(arr))
        except Exception as e:
            print(f"  Warning: skipping {path}: {e}")
    result = {}
    for b, vs in arrays.items():
        if vs:
            result[b] = np.concatenate(vs)
        else:
            result[b] = np.array([])
    return result


def _shared_bins(arrays, nbins):
    """Compute shared bin edges across multiple arrays."""
    all_vals = [v for v in arrays if len(v) > 0]
    if not all_vals:
        return np.linspace(0, 1, nbins + 1)
    combined = np.concatenate(all_vals)
    lo, hi = np.nanpercentile(combined, [0.1, 99.9])
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    return np.linspace(lo, hi, nbins + 1)


def _save_fig(fig, output_dir, filename):
    """Save figure to local or EOS path via fsspec."""
    full = os.path.join(output_dir, filename)
    if full.startswith("root://"):
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        buf.seek(0)
        with fsspec.open(full, "wb") as f:
            f.write(buf.read())
    else:
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        fig.savefig(full, dpi=120)


def plot_raw(args, saved, files_by_label):
    """Plot raw input branches from friend tree ROOT files, split by label.

    For multi-jet branches (CanJet, NotCanJet), produces one plot per jet index.
    For scalar branches (Ancillary), produces one plot per branch.
    """
    labels = saved["label"]
    input_cfg = saved["input"]
    groups = _branch_groups(input_cfg)
    all_branches = [b for bs in groups.values() for b in bs]
    n_canjet = 4
    n_notcanjet = input_cfg.get("n_NotCanJet", 8)

    data_by_label = {}
    for label in labels:
        paths = files_by_label.get(label, [])
        if not paths:
            print(f"  No files for {label}, skipping")
            continue
        print(f"Reading {len(paths)} files for {label}...")
        data_by_label[label] = read_branches(
            paths, all_branches, n_canjet=n_canjet, n_notcanjet=n_notcanjet
        )

    raw_dir = os.path.join(args.output_dir, "raw")
    if not raw_dir.startswith("root://"):
        os.makedirs(raw_dir, exist_ok=True)

    for group_name, branches in groups.items():
        for branch in branches:
            # Determine number of jets for this branch
            if group_name == "CanJet":
                n_jets = n_canjet
            elif group_name == "NotCanJet":
                n_jets = n_notcanjet
            else:
                n_jets = 0  # scalar

            if n_jets > 0:
                # Per-jet plots
                for jet_idx in range(n_jets):
                    # Collect values per label for shared binning
                    vals_per_label = {}
                    for label in labels:
                        if label not in data_by_label:
                            continue
                        arr = data_by_label[label].get(branch, np.array([]))
                        if arr.size == 0 or arr.ndim < 2:
                            continue
                        vals = arr[:, jet_idx]
                        vals = vals[np.isfinite(vals)]
                        if group_name == "NotCanJet":
                            vals = vals[vals != -1]
                        if len(vals) > 0:
                            vals_per_label[label] = vals
                    bins = _shared_bins(list(vals_per_label.values()), args.nbins)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    for i, label in enumerate(labels):
                        if label not in vals_per_label:
                            continue
                        ax.hist(
                            vals_per_label[label], bins=bins, histtype="step", density=True,
                            label=label, color=LABEL_COLORS[i % len(LABEL_COLORS)],
                            linewidth=1.5,
                        )
                    ax.set_xlabel(f"{branch}[{jet_idx}]")
                    ax.set_ylabel("Density")
                    ax.set_title(f"Raw: {branch} — jet {jet_idx}")
                    ax.legend()
                    fig.tight_layout()
                    _save_fig(fig, raw_dir, f"{branch}_jet{jet_idx}.png")
                    plt.close(fig)
                    print(f"  Saved {branch}_jet{jet_idx}.png")
            else:
                # Scalar branch — single plot
                vals_per_label = {}
                for label in labels:
                    if label not in data_by_label:
                        continue
                    vals = data_by_label[label].get(branch, np.array([]))
                    if vals.size == 0:
                        continue
                    vals = vals[np.isfinite(vals)]
                    if len(vals) > 0:
                        vals_per_label[label] = vals
                bins = _shared_bins(list(vals_per_label.values()), args.nbins)
                fig, ax = plt.subplots(figsize=(8, 5))
                for i, label in enumerate(labels):
                    if label not in vals_per_label:
                        continue
                    ax.hist(
                        vals_per_label[label], bins=bins, histtype="step", density=True,
                        label=label, color=LABEL_COLORS[i % len(LABEL_COLORS)],
                        linewidth=1.5,
                    )
                ax.set_xlabel(branch)
                ax.set_ylabel("Density")
                ax.set_title(f"Raw: {branch}")
                ax.legend()
                fig.tight_layout()
                _save_fig(fig, raw_dir, f"{branch}.png")
                plt.close(fig)
                print(f"  Saved {branch}.png")

    print(f"Raw plots saved to {raw_dir}/")


def _build_tensors(file_paths, input_cfg):
    """Read friend tree ROOT files and build (j, o, a) tensors for dataPrep.

    Layout matches what dataPrep expects before reshape:
      j: (N, n_CanJet_features * 4)  — e.g. (N, 16) for [pt,eta,phi,mass]×4
      o: (N, n_NotCanJet_features * n_NotCanJet) — e.g. (N, 40)
      a: (N, n_ancillary)
    Features are concatenated in the order given by input_cfg, matching
    the reshape in dataPrep: j.view(n, n_features, 4), o.view(n, n_features, -1).
    """
    import torch
    import uproot
    import awkward as ak

    canjet_branches = list(input_cfg["feature_CanJet"])
    notcanjet_branches = list(input_cfg["feature_NotCanJet"])
    ancillary_branches = list(input_cfg["feature_ancillary"])
    n_notcanjet = input_cfg["n_NotCanJet"]

    n_canjet = 4
    j_list, o_list, a_list = [], [], []
    for path in file_paths:
        try:
            with uproot.open(path + ":Events") as tree:
                # Filter: only events with enough CanJets
                ncanjet = tree["nCanJet"].array(library="np")
                mask = ncanjet >= n_canjet

                # CanJet: jagged → pad to exactly 4 jets
                j_parts = []
                for b in canjet_branches:
                    arr = tree[b].array(library="ak")[mask]
                    padded = ak.fill_none(ak.pad_none(arr, n_canjet, clip=True), 0)
                    j_parts.append(ak.to_numpy(padded))
                j_list.append(np.concatenate(j_parts, axis=1))

                # NotCanJet: jagged → pad to n_notcanjet
                o_parts = []
                for b in notcanjet_branches:
                    arr = tree[b].array(library="ak")[mask]
                    padded = ak.fill_none(ak.pad_none(arr, n_notcanjet, clip=True), -1)
                    o_parts.append(ak.to_numpy(padded))
                o_list.append(np.concatenate(o_parts, axis=1))

                # Ancillary: scalars
                a_parts = []
                for b in ancillary_branches:
                    if b == "year":
                        match = re.search(r"UL(\d{2})", path)
                        if not match:
                            match = re.search(r"20(\d{2})", path)
                        if match:
                            y_val = float(match.group(1))
                        else:
                            raise ValueError(f"Could not parse year from path: {path}")
                        n_events = np.sum(mask)
                        arr = np.full((n_events, 1), y_val, dtype=np.float32)
                    else:
                        arr = ak.to_numpy(tree[b].array(library="ak")[mask])
                        if arr.ndim == 1:
                            arr = arr[:, np.newaxis]
                    a_parts.append(arr)
                a_list.append(np.concatenate(a_parts, axis=1))
        except Exception as e:
            print(f"  Warning: skipping {path}: {e}")

    if not j_list:
        return None, None, None

    return (
        torch.tensor(np.concatenate(j_list), dtype=torch.float32),
        torch.tensor(np.concatenate(o_list), dtype=torch.float32),
        torch.tensor(np.concatenate(a_list), dtype=torch.float32),
    )


def _build_model(saved):
    """Reconstruct HCR model from checkpoint, return the nn module."""
    import torch
    from src.classifier.nn.blocks.HCR import HCR as HCRNet
    from src.classifier.ml.models.HCR import HCRArch

    arch = HCRArch.load(saved["arch"])
    nn = HCRNet(
        dijetFeatures=arch.n_features,
        quadjetFeatures=arch.n_features,
        ancillaryFeatures=saved["input"]["feature_ancillary"],
        useOthJets="attention" if arch.attention else "",
        device="cpu",
        nClasses=len(saved["label"]),
    )
    nn.load_state_dict(saved["model"])
    nn.eval()
    return nn


def _ancillary_post_name(raw_name):
    """Map raw ancillary feature name to its post-dataPrep description."""
    if "nSelJets" in raw_name:
        return f"log({raw_name})"
    return raw_name


def plot_dataprep(args, saved, files_by_label):
    """Run dataPrep on raw inputs and plot the transformed features."""
    import torch

    nn = _build_model(saved)
    input_embed = nn.inputEmbed
    labels = saved["label"]
    input_cfg = saved["input"]
    ancillary_names = list(input_cfg["feature_ancillary"])

    # dataPrep output tensor descriptions
    # Feature names are fixed by the dataPrep transform logic
    tensor_info = {
        "j": ["log(1+pt)", "eta", "deltaPhi", "log(1+mass)"],
        "d": ["log(1+pt)", "eta", "deltaPhi", "log(1+mass)"],
        "q": ["log(1+pt)", "eta", "log(1+mass)"],
        "a": [_ancillary_post_name(n) for n in ancillary_names],
        "o": ["log(1+pt)", "eta", "log(1+mass)", "isSelJet/isCanJet"],
    }

    # Run dataPrep per label
    results = {}
    for label in labels:
        paths = files_by_label.get(label, [])
        if not paths:
            continue
        print(f"Building tensors for {label} ({len(paths)} files)...")
        j, o, a = _build_tensors(paths, input_cfg)
        if j is None:
            print(f"  No data loaded for {label}, skipping")
            continue
        print(f"  Running dataPrep ({j.shape[0]} events)...")
        with torch.no_grad():
            j_out, d_out, q_out, a_out, o_out, *_ = input_embed.dataPrep(j, o, a)
        results[label] = {
            "j": j_out.numpy(),
            "d": d_out.numpy(),
            "q": q_out.numpy(),
            "a": a_out.numpy(),
            "o": o_out.numpy() if o_out is not None else None,
        }

    dp_dir = os.path.join(args.output_dir, "dataprep")
    if not dp_dir.startswith("root://"):
        os.makedirs(dp_dir, exist_ok=True)

    for tensor_name, feat_names in tensor_info.items():
        for feat_idx, feat_name in enumerate(feat_names):
            # Collect values per label for shared binning
            vals_per_label = {}
            for label in labels:
                if label not in results:
                    continue
                tensor = results[label][tensor_name]
                if tensor is None:
                    continue
                vals = tensor[:, feat_idx, :].flatten()
                vals = vals[np.isfinite(vals)]
                if len(vals) > 0:
                    vals_per_label[label] = vals
            bins = _shared_bins(list(vals_per_label.values()), args.nbins)
            fig, ax = plt.subplots(figsize=(8, 5))
            for i, label in enumerate(labels):
                if label not in vals_per_label:
                    continue
                ax.hist(
                    vals_per_label[label], bins=bins, histtype="step", density=True,
                    label=label, color=LABEL_COLORS[i % len(LABEL_COLORS)],
                    linewidth=1.5,
                )
            ax.set_xlabel(f"{tensor_name}[{feat_idx}]: {feat_name}")
            ax.set_ylabel("Density")
            ax.set_title(f"dataPrep: {tensor_name} — {feat_name}")
            ax.legend()
            fig.tight_layout()
            safe_name = feat_name.replace("(", "").replace(")", "").replace("+", "p").replace("/", "_").replace(" ", "_")
            _save_fig(fig, dp_dir, f"{tensor_name}_{feat_idx}_{safe_name}.png")
            plt.close(fig)
            print(f"  Saved {tensor_name}_{feat_idx}_{safe_name}.png")

    print(f"DataPrep plots saved to {dp_dir}/")


if __name__ == "__main__":
    main()
