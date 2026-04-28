from __future__ import annotations
import sys
# Remove the directory containing this file from sys.path to prevent local
# random.py from shadowing stdlib random (used transitively by tempfile)
sys.path = [p for p in sys.path if p != __file__.rsplit('/', 1)[0]]

import os
import re
import pickle
import tempfile
import argparse
import numpy as np
import awkward as ak
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
plt.rcParams["figure.figsize"] = [8,8]
plt.rcParams["font.size"] = 18

REGIONS = ("nominal_4j2b", "lowpt_4j2b")
DEFAULT_N_BINS = 5  # start of n_bins scan
# Filename pattern written by bbreww processor:
#   phh_hist_{dataset}__{year}_{chunk_id}.pkl
_PHH_FILE_RE = re.compile(r"^phh_hist_(?P<dataset>.+?)__(?P<year>.+)_(?P<chunk>[0-9a-f]{8})\.pkl$")
# taken from https://github.com/mmarchegiani/ttHbb_SPANet/blob/main/scripts/quantile_regression.py

class WeightedQuantileTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_quantiles=1000, output_distribution='normal'):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(np.array([self.quantiles_, self.reference_quantiles_]), f)

    def load(self, filename):
        extension = os.path.splitext(filename)[1]
        if not extension == '.pkl':
            raise ValueError(f"Invalid file extension '{os.path.splitext(filename)[1]}'. Only '.pkl' files are supported.")
        self.quantiles_, self.reference_quantiles_ = np.load(filename, allow_pickle=True)

    def _weighted_quantiles(self, X, weights):
        # Filter out NaN/Inf values — np.argsort puts NaN at the end,
        # which causes the upper quantiles to become NaN
        valid = np.isfinite(X) & np.isfinite(weights)
        X = X[valid]
        weights = weights[valid]

        # Calculate weighted quantiles
        sorted_indices = np.argsort(X)
        X_sorted = X[sorted_indices]
        weights_sorted = weights[sorted_indices]
        cum_weights = np.cumsum(weights_sorted) / np.sum(weights_sorted)

        # Interpolate to get quantiles
        quantiles = np.interp(np.linspace(0, 1, self.n_quantiles), cum_weights, X_sorted)
        return quantiles
    
    def fit(self, X, y=None, sample_weight=None):
        if sample_weight is None:
            raise ValueError("Sample weights must be provided.")
        
        self.quantiles_ = self._weighted_quantiles(X, sample_weight)
        
        if self.output_distribution == 'normal':
            self.reference_quantiles_ = norm.ppf(np.linspace(0, 1, self.n_quantiles))
        elif self.output_distribution == 'uniform':
            self.reference_quantiles_ = np.linspace(0, 1, self.n_quantiles)
        else:
            raise ValueError(f"Unknown output distribution '{self.output_distribution}'.")
        
        return self
    
    def transform(self, X):
        # Interpolate based on weighted quantiles (NaN inputs produce NaN outputs)
        transformed_X = np.where(
            np.isfinite(X),
            np.interp(X, self.quantiles_, self.reference_quantiles_),
            np.nan
        )
        return transformed_X

def plot_score(X, W, transformer, label, output_dir):
    transformed_score = transformer.transform(X)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original score
    ax1.hist(X, weights=W, bins=100, histtype='step', label=label)
    ax1.set_xlabel("Original Score")
    ax1.set_ylabel("Counts")
    ax1.legend()
    
    # Transformed score - use fixed 0-1 range to verify flatness
    ax2.hist(transformed_score, weights=W, bins=np.linspace(0, 1, 101), histtype='step', label=f"{label} transformed")
    ax2.set_xlabel("Transformed Score")
    ax2.set_ylabel("Counts")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{label}_score.png", dpi=300)


def _load_pickle(path):
    """Load a pickle from local path or EOS URL."""
    from src.storage.eos import EOS
    eos_path = EOS(path)
    if eos_path.is_local:
        with open(str(eos_path.path), 'rb') as f:
            return pickle.load(f)
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        tmp = f.name
    try:
        eos_path.copy_to(EOS(tmp), overwrite=True)
        with open(tmp, 'rb') as f:
            return pickle.load(f)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_phh_from_directory(input_dir):
    """Glob `phh_hist_*.pkl` files in an EOS or local directory and group by dataset.

    Returns a nested dict: {dataset_name: {region: {'phh': array, 'weight': array}}}.
    All eras and chunks of the same dataset are concatenated.
    """
    from src.storage.eos import EOS
    dir_eos = EOS(input_dir)
    entries = dir_eos.ls()
    grouped = {}  # dataset -> region -> (list[phh], list[weight])
    n_files = 0
    for entry in entries:
        fname = os.path.basename(str(entry.path))
        m = _PHH_FILE_RE.match(fname)
        if not m:
            continue
        dataset = m.group("dataset")
        data = _load_pickle(str(entry))
        n_files += 1
        slot = grouped.setdefault(dataset, {r: ([], []) for r in REGIONS})
        for region in REGIONS:
            if region not in data:
                continue
            slot[region][0].append(np.asarray(data[region]['phh']))
            slot[region][1].append(np.asarray(data[region]['weight']))
    out = {}
    for dataset, regions in grouped.items():
        out[dataset] = {}
        for region, (phh_list, w_list) in regions.items():
            if not phh_list:
                continue
            out[dataset][region] = {
                'phh': np.concatenate(phh_list),
                'weight': np.concatenate(w_list),
            }
    print(f"Loaded {n_files} chunk pickles across {len(out)} datasets from {input_dir}")
    return out


def split_signal_background(grouped):
    """Separate merged dataset dict into signal / background arrays per region.

    - signal: dataset name contains 'GluGlu'
    - data:   dataset name contains 'data' (case-insensitive) — excluded
    - background: everything else
    """
    sig = {r: {'phh': [], 'weight': []} for r in REGIONS}
    bkg = {r: {'phh': [], 'weight': []} for r in REGIONS}
    for dataset, regions in grouped.items():
        name_lc = dataset.lower()
        if 'data' in name_lc:
            continue
        target = sig if 'GluGlu' in dataset else bkg
        for region, arrs in regions.items():
            target[region]['phh'].append(arrs['phh'])
            target[region]['weight'].append(arrs['weight'])
    def _concat(d):
        return {r: {k: (np.concatenate(v) if v else np.array([]))
                    for k, v in arrs.items()}
                for r, arrs in d.items()}
    return _concat(sig), _concat(bkg)


def _quantile_bin_edges(transformer, n_bins):
    """Bin edges in original score space giving ~equal signal yield per bin."""
    edges = np.interp(np.linspace(0, 1, n_bins + 1),
                      transformer.reference_quantiles_, transformer.quantiles_)
    # guarantee monotonic, cover full [0, 1]
    edges[0] = min(edges[0], 0.0)
    edges[-1] = max(edges[-1], 1.0)
    return edges


def compute_significance(sig_phh, sig_w, bkg_phh, bkg_w, bin_edges,
                         min_unweighted_bkg=5):
    """Asimov-style binned significance accounting for MC stat uncertainty:

        Z = sqrt(sum_i s_i^2 / (b_i + sigma_b_i^2))

    where sigma_b_i^2 = sum of squared background weights in bin i (the
    statistical uncertainty on the background prediction from finite MC).
    Without this term, the score blows up in narrow bins where b is tiny
    and biases the scan toward absurdly many bins.

    Bins with unweighted background count below `min_unweighted_bkg` invalidate
    the binning (returns (nan, n_bad)). Bins with b_i <= 0 are skipped.
    """
    s, _     = np.histogram(sig_phh, bins=bin_edges, weights=sig_w)
    b, _     = np.histogram(bkg_phh, bins=bin_edges, weights=bkg_w)
    b_var, _ = np.histogram(bkg_phh, bins=bin_edges, weights=bkg_w ** 2)
    b_raw, _ = np.histogram(bkg_phh, bins=bin_edges)
    n_bad = int(np.sum(b_raw < min_unweighted_bkg))
    if n_bad > 0:
        return np.nan, n_bad
    denom = b + b_var
    valid = denom > 0
    z2 = np.sum(np.where(valid, s ** 2 / np.where(valid, denom, 1.0), 0.0))
    return float(np.sqrt(z2)), 0


def stability_split_scan(transformer, sig_phh, sig_w, bkg_phh, bkg_w,
                         n_bins_default, n_bins_max, min_unweighted_bkg):
    """Run the bin scan on two independent halves (odd/even event index) of
    signal AND background. If the 'best n_bins' is driven by real structure,
    both halves should peak near the same n. If it's driven by statistical
    fluctuations, the two halves will disagree.
    """
    results = {}
    for label, idx_fn in (("halfA", lambda n: np.arange(n) % 2 == 0),
                          ("halfB", lambda n: np.arange(n) % 2 == 1)):
        s_mask = idx_fn(sig_phh.size)
        b_mask = idx_fn(bkg_phh.size)
        # scale weights by 2 so each half represents the full luminosity
        scan = optimize_n_bins(transformer,
                               sig_phh[s_mask], 2.0 * sig_w[s_mask],
                               bkg_phh[b_mask], 2.0 * bkg_w[b_mask],
                               n_bins_default=n_bins_default,
                               n_bins_max=n_bins_max,
                               min_unweighted_bkg=min_unweighted_bkg)
        results[label] = scan
    return results


def optimize_n_bins(transformer, sig_phh, sig_w, bkg_phh, bkg_w,
                    n_bins_default=DEFAULT_N_BINS, n_bins_max=30,
                    min_unweighted_bkg=5):
    """Scan n_bins from default to n_bins_max. Stop when the min-bkg floor is hit.

    Returns dict with per-n_bins results and the best n_bins.
    """
    results = []  # list of (n_bins, Z, n_bad)
    z_default = None
    best_z = -np.inf
    best_n = None
    stopped_at = None
    for n in range(n_bins_default, n_bins_max + 1):
        edges = _quantile_bin_edges(transformer, n)
        z, n_bad = compute_significance(sig_phh, sig_w, bkg_phh, bkg_w, edges,
                                        min_unweighted_bkg=min_unweighted_bkg)
        results.append((n, z, n_bad))
        if n == n_bins_default:
            z_default = z
        if np.isnan(z):
            stopped_at = n
            break
        if z > best_z:
            best_z = z
            best_n = n
    return {
        'results': results,
        'n_bins_default': n_bins_default,
        'z_default': z_default,
        'best_n_bins': best_n,
        'best_z': best_z if best_n is not None else None,
        'stopped_at': stopped_at,
    }


def run_bin_optimization(input_dir, output_dir, n_quantiles=10000,
                         n_bins_default=DEFAULT_N_BINS, n_bins_max=30,
                         min_unweighted_bkg=5):
    """End-to-end: load directory, split sig/bkg, fit transformer, scan bins.

    Results are printed per region and the fitted transformer is saved (per
    region) to `output_dir/quantiles_regressed_{region}.pkl`.
    """
    os.makedirs(output_dir, exist_ok=True)
    grouped = load_phh_from_directory(input_dir)
    sig, bkg = split_signal_background(grouped)

    summary = {}
    for region in REGIONS:
        s_phh, s_w = sig[region]['phh'], sig[region]['weight']
        b_phh, b_w = bkg[region]['phh'], bkg[region]['weight']
        if s_phh.size == 0 or b_phh.size == 0:
            print(f"[{region}] no signal or background events — skipping")
            continue
        print(f"\n=== {region} ===")
        print(f"  signal:     {s_phh.size} events, sum(w) = {s_w.sum():.3f}")
        print(f"  background: {b_phh.size} events, sum(w) = {b_w.sum():.3f}")

        transformer = WeightedQuantileTransformer(n_quantiles=n_quantiles,
                                                  output_distribution='uniform')
        transformer.fit(s_phh, sample_weight=s_w)
        plot_score(s_phh, s_w, transformer, f"HH_{region}", output_dir)
        transformer.save(os.path.join(output_dir,
                                      f"quantiles_regressed_{region}.pkl"))

        scan = optimize_n_bins(transformer, s_phh, s_w, b_phh, b_w,
                               n_bins_default=n_bins_default,
                               n_bins_max=n_bins_max,
                               min_unweighted_bkg=min_unweighted_bkg)
        summary[region] = scan

        best_n = scan['best_n_bins']
        if best_n is not None:
            edges = _quantile_bin_edges(transformer, best_n)
            edges_path = os.path.join(output_dir, f"bin_edges_{region}.txt")
            with open(edges_path, 'w') as f:
                f.write(f"# region={region}  n_bins={best_n}  Z={scan['best_z']:.4f}\n")
                f.write(", ".join(f"{e:.6f}" for e in edges) + "\n")
            print(f"  bin edges written to {edges_path}")

        halves = stability_split_scan(transformer, s_phh, s_w, b_phh, b_w,
                                      n_bins_default=n_bins_default,
                                      n_bins_max=n_bins_max,
                                      min_unweighted_bkg=min_unweighted_bkg)
        print(f"  stability split (two independent halves, weights x2):")
        for label, hscan in halves.items():
            print(f"    {label}: best n_bins = {hscan['best_n_bins']}, "
                  f"Z = {hscan['best_z']:.4f}"
                  if hscan['best_n_bins'] is not None
                  else f"    {label}: no valid binning")

        print(f"  default n_bins = {scan['n_bins_default']}: "
              f"Z = {scan['z_default']}")
        if scan['stopped_at'] is not None:
            print(f"  stopped at n_bins = {scan['stopped_at']}: "
                  f"a background bin had < {min_unweighted_bkg} unweighted events")
        if scan['best_n_bins'] is not None:
            print(f"  best n_bins    = {scan['best_n_bins']}: "
                  f"Z = {scan['best_z']:.4f}")
        best_n = scan['best_n_bins']
        if best_n is not None:
            window = [(n, z, nb) for n, z, nb in scan['results']
                      if abs(n - best_n) <= 10]
            print(f"  scan (n_bins, Z, n_bad_bkg_bins) within +/-10 of best n={best_n}:")
        else:
            window = scan['results']
            print(f"  scan (n_bins, Z, n_bad_bkg_bins):")
        for n, z, nb in window:
            z_s = f"{z:.4f}" if not np.isnan(z) else "NaN (stopped)"
            print(f"    n_bins={n:3d}  Z={z_s}  n_bad_bkg_bins={nb}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantile regression of ML classifier HH score")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("-i", "--input", nargs="+", help="Input pkl files (single-sample legacy mode)")
    src.add_argument("--input-dir", help=(
        "Directory (local or EOS root:// URL) containing phh_hist_*.pkl chunk "
        "files. Files are grouped by dataset, split into signal (GluGlu*) and "
        "background (everything else except 'data'), and the bin-count "
        "optimization is run per region."))
    parser.add_argument("-o", "--output", help="Output folder for fitted quantile transformer", required=True)
    parser.add_argument("-n", "--n_quantiles", type=int, default=10000, help="Number of quantiles", required=False)
    parser.add_argument("-b", "--n_bins", type=int, default=DEFAULT_N_BINS,
                        help="Default/starting number of equal-probability bins")
    parser.add_argument("--n-bins-max", type=int, default=30,
                        help="Upper end of the bin-count scan (--input-dir mode)")
    parser.add_argument("--min-bkg", type=int, default=5,
                        help="Minimum unweighted background events required per bin")
    parser.add_argument("-r", "--region", choices=list(REGIONS), default="nominal_4j2b",
                        help="Region to use for fitting in legacy -i mode")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.input_dir is not None:
        run_bin_optimization(
            input_dir=args.input_dir,
            output_dir=args.output,
            n_quantiles=args.n_quantiles,
            n_bins_default=args.n_bins,
            n_bins_max=args.n_bins_max,
            min_unweighted_bkg=args.min_bkg,
        )
    else:
        # Legacy single-sample mode: `-i` files are assumed to be signal.
        for file in args.input:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Input file '{file}' does not exist.")
        output_file = os.path.join(args.output, "quantiles_regressed.pkl")

        phh_list = []
        weight_list = []

        print("--- File Validation ---")
        for file in args.input:
            with open(file, 'rb') as f:
                data = pickle.load(f)

            phh = data[args.region]['phh']
            weights = data[args.region]['weight']

            phh_list.append(phh)
            weight_list.append(weights)

        X = np.concatenate(phh_list)
        W = np.concatenate(weight_list)

        print(f"Loaded {len(X)} events from {len(args.input)} files")
        print(f"Using region: {args.region}")

        transformer = WeightedQuantileTransformer(n_quantiles=args.n_quantiles, output_distribution='uniform')

        print("Fitting quantile transformer on signal sample...")
        transformer.fit(X, sample_weight=W)
        plot_score(X, W, transformer, "HH", args.output) # plot signal score to verify it's flat

        print("Saving the fitted quantiles to", output_file)
        transformer.save(output_file)

        # Print custom bin edges in original score space
        bin_edges = np.interp(np.linspace(0, 1, args.n_bins + 1), transformer.reference_quantiles_, transformer.quantiles_)
        print(f"\n{args.n_bins} equal-probability bin edges in original score space:")
        print(np.array2string(bin_edges, separator=', '))