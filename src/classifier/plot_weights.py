#!/usr/bin/env python
"""Compute weight statistics and plot event weight diagnostics for classifier training.

This script parses metadata, loads event weight branches (weight, FvT, pseudoTagWeight)
alongside key kinematics (CanJet_pt[0] and selected jet counts), computes weight
statistics (including effective sample size), and generates diagnostic plots.

Usage:
  ./run_container classifier_cpu python src/classifier/plot_weights.py \
      --checkpoint output/some_model/ \
      --metadata coffea4bees/metadata/datasets_HH4b_Run2/2024_v2/classifier_inputs_lowpt_tightWP_wlowptJCM.json@@HCR_input_lowpt \
      --output-dir output/weight_debug
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

from src.data_formats.root import Friend, Chunk
from src.classifier.config.dataset.HCR.SvB import _remove_outlier
from src.classifier.task import parse

# --------------------------------------------------------------------------- #
# Sample directory name → physics label mapping
# --------------------------------------------------------------------------- #
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
    """Build [(compiled_regex, label)] from a JSON file or built-in defaults."""
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
        "--checkpoint", required=True,
        help="path to model directory (contains *.pkl) or direct .pkl file",
    )
    p.add_argument(
        "--metadata", required=True,
        help="path@@key to the metadata JSON (e.g. path/to/file.json@@HCR_input_lowpt)",
    )
    p.add_argument(
        "--output-dir", default="output/weight_diagnostics",
        help="directory to save plots and summary (local path or root:// EOS path)",
    )
    p.add_argument(
        "--max-files", type=int, default=10,
        help="max friend tree files to read per label (default: 10)",
    )
    p.add_argument("--nbins", type=int, default=50, help="histogram bins")
    p.add_argument(
        "--label-map", default=None,
        help='JSON file mapping regex patterns to labels, e.g. {"^data_": "multijet", "^TTTo": "ttbar"}. '
             "If not given, auto-selects from built-in maps based on checkpoint labels.",
    )
    p.add_argument(
        "--friends", nargs="*", default=[],
        help="friend trees mapping, e.g. 'label:data path/to/result.json@@key'"
    )
    p.add_argument(
        "--JCM-weight", nargs="*", default=[],
        help="JCM weight configuration, e.g. 'label:data path/to/JCM.yml@@JCM_weights'"
    )
    p.add_argument(
        "--wfs-base", default="",
        help="path to training workflow configuration folder (contains train.yml)"
    )
    p.add_argument(
        "--template-str", default="",
        help="template formatting string (e.g. 'model: path, FvT: path')"
    )
    p.add_argument(
        "--friend-path", default="",
        help="resolved path to the FvT friend trees"
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #
def load_checkpoint(path):
    """Load a checkpoint .pkl from a file or directory (picks latest)."""
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


def parse_metadata_arg(arg):
    """Split 'path/to/file.json@@key' into (path, key)."""
    parts = arg.split("@@", 1)
    if len(parts) != 2:
        sys.exit(f"--metadata must be path@@key, got: {arg}")
    return parts[0], parts[1]


def load_friend_files_by_label(metadata_path, meta_key, labels, max_files, label_patterns):
    """Read metadata JSON and group friend tree paths by physics label."""
    with open(metadata_path) as f:
        meta = json.load(f)[meta_key]

    label_set = set(labels)
    files = {label: [] for label in labels}
    for entry in meta["data"]:
        pico_path = entry[0]["path"]
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
# Helper Matchers and Parsers
# --------------------------------------------------------------------------- #
def parse_friend_args(friends_list):
    """Parse a list of friends arguments into a dictionary mapping label_group -> (path, key)."""
    parsed = {}
    all_tokens = []
    for item in friends_list:
        all_tokens.extend(item.split())
        
    for i in range(0, len(all_tokens), 2):
        if i + 1 >= len(all_tokens):
            break
        group = all_tokens[i]
        path_arg = all_tokens[i+1]
        path, key_path = parse_metadata_arg(path_arg)
        parsed[group] = (path, [k for k in key_path.split(".") if k])
    return parsed


def parse_jcm_args(jcm_list):
    """Parse a list of JCM arguments into a dictionary mapping label_group -> (path, key)."""
    parsed = {}
    all_tokens = []
    for item in jcm_list:
        all_tokens.extend(item.split())
        
    for i in range(0, len(all_tokens), 2):
        if i + 1 >= len(all_tokens):
            break
        group = all_tokens[i]
        path_arg = all_tokens[i+1]
        path, key_path = parse_metadata_arg(path_arg)
        parsed[group] = (path, [k for k in key_path.split(".") if k])
    return parsed


def matches_group(label, group):
    """Check if the label matches the configuration group (e.g. label:data -> multijet)."""
    if not group or group == '""' or group == "all":
        return True
    if ":" in group:
        g_type, g_val = group.split(":", 1)
        if g_type == "label":
            if g_val == "data" and label == "multijet":
                return True
            return g_val == label
    return group in label


def load_train_yaml(wfs_base, template_str, friend_path):
    """Load train.yml, format it, and extract friends and JCM-weight options."""
    train_yml_path = os.path.join(wfs_base, "train.yml")
    if not os.path.exists(train_yml_path):
        print(f"Warning: train.yml not found at {train_yml_path}")
        return [], []
        
    print(f"Automatically parsing training configuration from {train_yml_path}...")
    import yaml
    with open(train_yml_path, "r") as f:
        content = f.read()
        
    format_dict = {}
    if template_str:
        # Parse template_str like "model: path, FvT: path"
        parts = template_str.split(",")
        for part in parts:
            if ":" in part:
                k, v = part.split(":", 1)
                format_dict[k.strip()] = v.strip()
    
    if friend_path:
        format_dict["FvT"] = friend_path
        
    if format_dict:
        try:
            content = content.format(**format_dict)
        except Exception as e:
            print(f"Warning: could not format train.yml with templates: {e}")
            
    train_data = yaml.safe_load(content)
    datasets = train_data.get("dataset", [])
    
    friends_args = []
    jcm_args = []
    
    for ds in datasets:
        opts = ds.get("option", [])
        i = 0
        while i < len(opts):
            opt = opts[i]
            if opt.startswith("--friends"):
                parts = opt.split(maxsplit=1)
                if len(parts) > 1:
                    friends_args.append(parts[1])
                elif i + 1 < len(opts):
                    friends_args.append(opts[i+1])
                    i += 1
            elif opt.startswith("--JCM-weight"):
                parts = opt.split(maxsplit=1)
                if len(parts) > 1:
                    jcm_args.append(parts[1])
                elif i + 1 < len(opts):
                    jcm_args.append(opts[i+1])
                    i += 1
            i += 1
            
    return friends_args, jcm_args


# --------------------------------------------------------------------------- #
# Reading Data
# --------------------------------------------------------------------------- #
def read_data(file_paths, label, friend_obj, jcm_processor, n_canjet=4):
    """Read branches and apply JCM, FvT reweighting, and outlier removal."""
    import uproot
    import awkward as ak
    import pandas as pd
    
    if not file_paths:
        return None
        
    data = {
        "pt": [],
        "pt4": [],
        "njets": [],
        "weight": []
    }
    
    for path in file_paths:
        try:
            # Create target Chunk object to match friend tree
            target_chunk = Chunk(source=path, name="Events", fetch=True)
            num_entries = target_chunk.num_entries
            
            with uproot.open(path + ":Events") as tree:
                keys = list(tree.keys())
                
                # Filter: nCanJet >= 4
                ncanjet = tree["nCanJet"].array(library="np")
                mask = ncanjet >= n_canjet
                if np.sum(mask) == 0:
                    continue
                
                # Select variables
                if "nSelJets_lowpt" in keys:
                    nseljets_col = "nSelJets_lowpt"
                    selected_col = "lowpt_threeTag"
                else:
                    nseljets_col = "nSelJets"
                    selected_col = "threeTag"
                
                # Build pandas DataFrame for preprocessing
                df_data = {
                    "weight": tree["weight"].array(library="np"),
                    nseljets_col: tree[nseljets_col].array(library="np"),
                    selected_col: tree[selected_col].array(library="np"),
                }
                
                # Check for FvT branch from friend tree
                if friend_obj is not None:
                    try:
                        fvt_df = friend_obj.arrays(target_chunk, library="pd", reader_options={"branches": ["FvT"]})
                        if fvt_df is not None and "FvT" in fvt_df.columns:
                            df_data["FvT"] = fvt_df["FvT"].to_numpy()
                        else:
                            df_data["FvT"] = np.ones(num_entries, dtype=np.float32)
                    except Exception as e:
                        print(f"  Warning: could not read FvT from friend tree for {path}: {e}")
                        df_data["FvT"] = np.ones(num_entries, dtype=np.float32)
                else:
                    df_data["FvT"] = np.ones(num_entries, dtype=np.float32)
                
                df = pd.DataFrame(df_data)
                
                # Apply JCM weight if processor exists
                if jcm_processor is not None:
                    df = jcm_processor(df)
                    
                # Apply FvT reweighting if friends loaded
                if friend_obj is not None:
                    df["weight"] *= df["FvT"]
                    
                # Apply outlier and negative weight removal (which logs counts!)
                df = _remove_outlier(df)
                
                # Mask to keep events passing nCanJet selection
                passing_idx = df.index[mask[df.index]]
                df = df.loc[passing_idx]
                
                if len(df) == 0:
                    continue
                
                # Read CanJet_pt and get CanJet_pt[0] and CanJet_pt[3] for the remaining events
                if "CanJet_pt" in keys:
                    pt_arr = tree["CanJet_pt"].array(library="ak")[passing_idx]
                    padded_pt = ak.fill_none(ak.pad_none(pt_arr, 4, clip=True), 0.0)
                    pt_np = ak.to_numpy(padded_pt)[:, 0]
                    pt4_np = ak.to_numpy(padded_pt)[:, 3]
                    data["pt"].append(pt_np)
                    data["pt4"].append(pt4_np)
                else:
                    data["pt"].append(np.full(len(df), np.nan))
                    data["pt4"].append(np.full(len(df), np.nan))
                
                data["njets"].append(df[nseljets_col].to_numpy())
                data["weight"].append(df["weight"].to_numpy())
                
        except Exception as e:
            print(f"  Warning: skipping {path}: {e}")
            import traceback
            traceback.print_exc()
            
    result = {}
    for key, arrays in data.items():
        if arrays:
            result[key] = np.concatenate(arrays)
        else:
            result[key] = np.array([])
    return result


def _shared_bins(arrays, nbins):
    """Compute shared bin edges across multiple arrays."""
    combined = np.concatenate([arr for arr in arrays if len(arr) > 0])
    if len(combined) == 0:
        return np.linspace(0, 1, nbins + 1)
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


def _save_text(content, output_dir, filename):
    """Save text file to local or EOS path via fsspec."""
    full = os.path.join(output_dir, filename)
    if full.startswith("root://"):
        with fsspec.open(full, "w") as f:
            f.write(content)
    else:
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w") as f:
            f.write(content)


def compute_profile(x, y, bins):
    """Compute the mean and standard error of y in bins of x."""
    bin_indices = np.digitize(x, bins) - 1
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_means = []
    bin_sems = []
    
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        y_bin = y[mask]
        y_bin = y_bin[np.isfinite(y_bin)]
        if len(y_bin) > 0:
            mean = np.mean(y_bin)
            sem = np.std(y_bin) / np.sqrt(len(y_bin)) if len(y_bin) > 1 else 0.0
        else:
            mean = np.nan
            sem = np.nan
        bin_means.append(mean)
        bin_sems.append(sem)
        
    return bin_centers, np.array(bin_means), np.array(bin_sems)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    saved = load_checkpoint(args.checkpoint)
    labels = saved["label"]
    print(f"Labels in checkpoint: {labels}")

    label_patterns = _load_label_patterns(args.label_map, labels)
    meta_path, meta_key = parse_metadata_arg(args.metadata)
    files_by_label = load_friend_files_by_label(
        meta_path, meta_key, labels, args.max_files, label_patterns
    )
    
    for label, paths in files_by_label.items():
        print(f"  {label}: {len(paths)} files")

    # Determine friends & JCM arguments (CLI override or auto-load from YAML)
    friends_list = args.friends
    jcm_list = args.JCM_weight
    if not friends_list and not jcm_list and args.wfs_base:
        friends_list, jcm_list = load_train_yaml(
            args.wfs_base, args.template_str, args.friend_path
        )

    # Parse configurations
    friends_map = parse_friend_args(friends_list)
    jcm_map = parse_jcm_args(jcm_list)

    # Diagnostics loop
    summary_data = []
    
    for label in labels:
        paths = files_by_label.get(label, [])
        if not paths:
            print(f"No files for label: {label}, skipping diagnostics")
            continue
            
        # 1. Load Friend Trees if configured
        friend_obj = None
        for group, (json_path, key_path) in friends_map.items():
            if matches_group(label, group):
                print(f"Loading friend tree for group {group} from {json_path}@@{'.'.join(key_path)}...")
                with fsspec.open(json_path, "r") as f:
                    res = json.load(f)
                for k in key_path:
                    if isinstance(res, list):
                        res = res[int(k)]
                    else:
                        res = res[k]
                friend_obj = Friend.from_json(res)
                break
                
        # 2. Load JCM weights if configured
        jcm_processor = None
        for group, (json_path, key_path) in jcm_map.items():
            if matches_group(label, group):
                print(f"Loading JCM weights for group {group} from {json_path}@@{'.'.join(key_path)}...")
                from coffea4bees.classifier.compatibility.JCM.fit import apply_JCM_from_list
                jcm_processor = apply_JCM_from_list(
                    path=json_path + f"@@{'.'.join(key_path)}",
                    n_jets_col="nSelJets_lowpt" if "lowpt" in json_path else "nSelJets",
                    selected_col="lowpt_threeTag" if "lowpt" in json_path else "threeTag",
                    start=1,
                )
                break

        # Read and preprocess the data
        print(f"Reading and preprocessing files for {label}...")
        data = read_data(paths, label, friend_obj, jcm_processor)
        if not data:
            print(f"No data parsed for label: {label}, skipping")
            continue
            
        w = data["weight"]
        N = len(w)
        if N > 0:
            w_min = np.min(w)
            w_max = np.max(w)
            w_mean = np.mean(w)
            w_std = np.std(w)
            sum_w = np.sum(w)
            sum_w2 = np.sum(w**2)
            neff = (sum_w**2) / sum_w2 if sum_w2 > 0 else 0.0
            ratio = neff / N if N > 0 else 0.0
        else:
            w_min = w_max = w_mean = w_std = neff = ratio = 0.0
            
        summary_data.append({
            "label": label,
            "branch": "weight_reweighted" if (friend_obj or jcm_processor) else "weight_raw",
            "nraw": N,
            "min": w_min,
            "max": w_max,
            "mean": w_mean,
            "std": w_std,
            "neff": neff,
            "ratio": ratio
        })

        # 1. Log-scale weight distribution histogram
        print(f"Plotting weight distributions for {label}...")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_yscale("log")
        w_finite = w[np.isfinite(w)]
        if len(w_finite) > 0:
            bins = _shared_bins([w_finite], args.nbins)
            ax.hist(
                w_finite, bins=bins, histtype="step", density=True,
                label=label, linewidth=1.5, alpha=0.85
            )
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Density (log scale)")
        ax.set_title(f"Weight Distributions — {label}")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        fig.tight_layout()
        _save_fig(fig, args.output_dir, f"weights_dist_{label}.png")
        plt.close(fig)

        # 2. Kinematic profile vs leading jet pt (candJet0)
        pt = data["pt"]
        finite_pt = np.isfinite(pt)
        if np.any(finite_pt) and len(w) > 0:
            print(f"Plotting weight profile vs leading jet pt for {label}...")
            pt_finite = pt[finite_pt]
            pt_min, pt_max = np.percentile(pt_finite, [1, 99])
            if pt_min == pt_max:
                pt_min, pt_max = 0, 500
            bins_pt = np.linspace(pt_min, pt_max, 21)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            w_aligned = w[finite_pt]
            bin_centers, bin_means, bin_sems = compute_profile(pt_finite, w_aligned, bins_pt)
            valid = ~np.isnan(bin_means)
            ax.errorbar(
                bin_centers[valid], bin_means[valid], yerr=bin_sems[valid],
                fmt="o-", capsize=3, label=label, markersize=4, elinewidth=1.5
            )
            ax.set_xlabel("Leading Jet $p_T$ [GeV]")
            ax.set_ylabel("Mean Event Weight")
            ax.set_title(f"Weight Profile vs Leading Jet $p_T$ — {label}")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.tight_layout()
            _save_fig(fig, args.output_dir, f"profile_pt1_{label}.png")
            plt.close(fig)

        # 2b. Kinematic profile vs 4th leading jet pt (candJet3)
        pt4 = data["pt4"]
        finite_pt4 = np.isfinite(pt4)
        if np.any(finite_pt4) and len(w) > 0:
            print(f"Plotting weight profile vs 4th leading jet pt for {label}...")
            pt4_finite = pt4[finite_pt4]
            pt4_min, pt4_max = np.percentile(pt4_finite, [1, 99])
            if pt4_min == pt4_max:
                pt4_min, pt4_max = 0, 150
            bins_pt4 = np.linspace(pt4_min, pt4_max, 21)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            w_aligned = w[finite_pt4]
            bin_centers, bin_means, bin_sems = compute_profile(pt4_finite, w_aligned, bins_pt4)
            valid = ~np.isnan(bin_means)
            ax.errorbar(
                bin_centers[valid], bin_means[valid], yerr=bin_sems[valid],
                fmt="o-", capsize=3, label=label, markersize=4, elinewidth=1.5
            )
            ax.set_xlabel("4th Leading Jet $p_T$ [GeV]")
            ax.set_ylabel("Mean Event Weight")
            ax.set_title(f"Weight Profile vs 4th Leading Jet $p_T$ — {label}")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.tight_layout()
            _save_fig(fig, args.output_dir, f"profile_pt4_{label}.png")
            plt.close(fig)

        # 3. Kinematic profile vs njets
        njets = data["njets"]
        finite_njets = np.isfinite(njets)
        if np.any(finite_njets) and len(w) > 0:
            print(f"Plotting weight profile vs select jet count for {label}...")
            njets_finite = njets[finite_njets]
            njets_min = int(np.min(njets_finite))
            njets_max = int(np.max(njets_finite))
            njets_bins = np.arange(njets_min - 0.5, njets_max + 1.5, 1.0)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            w_aligned = w[finite_njets]
            bin_centers, bin_means, bin_sems = compute_profile(njets_finite, w_aligned, njets_bins)
            valid = ~np.isnan(bin_means)
            ax.errorbar(
                bin_centers[valid], bin_means[valid], yerr=bin_sems[valid],
                fmt="o-", capsize=3, label=label, markersize=4, elinewidth=1.5
            )
            ax.set_xlabel("$N_{\\text{selected jets}}$")
            ax.set_ylabel("Mean Event Weight")
            ax.set_title(f"Weight Profile vs $N_{{\\text{{selected jets}}}}$ — {label}")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.tight_layout()
            _save_fig(fig, args.output_dir, f"profile_njets_{label}.png")
            plt.close(fig)

    # 4. Generate summary table in Markdown
    print("Generating weight_stats.md table...")
    md_lines = [
        "# Event Weight Diagnostics Summary",
        "",
        "| Label | Weight Branch | $N_{\\text{raw}}$ | Min | Max | Mean | Std Dev | $N_{\\text{eff}}$ | $N_{\\text{eff}} / N_{\\text{raw}}$ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary_data:
        md_lines.append(
            f"| {row['label']} | {row['branch']} | {row['nraw']} | {row['min']:.4g} | {row['max']:.4g} | {row['mean']:.4g} | {row['std']:.4g} | {row['neff']:.1f} | {row['ratio']*100:.2f}% |"
        )
    
    md_content = "\n".join(md_lines) + "\n"
    _save_text(md_content, args.output_dir, "weight_stats.md")
    print(f"Diagnostics complete. Summary saved to {args.output_dir}/weight_stats.md")


if __name__ == "__main__":
    main()
