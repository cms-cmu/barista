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
from typing import Optional
import numpy as np
import awkward as ak
from scipy.stats import norm
try:
    from numpy.typing import NDArray
except ImportError:
    NDArray = None  # annotations are never evaluated (PEP 563)
try:
    # sklearn only supplies the (optional) estimator/transformer mixins; none of
    # the math below depends on it, so fall back to plain object if it's absent
    # (e.g. the bare coffea container, which has no sklearn).
    from sklearn.base import BaseEstimator, TransformerMixin
except ModuleNotFoundError:
    class BaseEstimator:  # noqa: D401 - minimal stand-in
        pass
    class TransformerMixin:  # noqa: D401 - minimal stand-in
        pass
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
plt.rcParams["figure.figsize"] = [8,8]
plt.rcParams["font.size"] = 18

REGIONS = ("nominal_4j2b", "lowpt_4j2b", "incl_3j2b")
DEFAULT_MAX_BINS = 30    # k_max: maximum number of final bins the DP may choose
DEFAULT_M_FINE = 250     # number of fine uniform-in-score bins the DP merges
DEFAULT_MIN_NEFF = 10.5  # minimum effective unweighted background events per bin
DEFAULT_Z2_MIN = 1e-5    # per-bin z^2 floor; forbids thin near-zero-signal bins
DEFAULT_EPS = 0.01       # knee tolerance: smallest k within (1+eps) of best mu_UL

# Major-background families whose yield must be strictly positive in every bin
# (positivity constraint of the DP). Patterns are matched against dataset names.
MAJOR_BKG_PATTERNS = {
    'tt':    ('TTTo',),
    'wjets': ('WtoLNu-2Jets',),
    'tW':    ('TbarWplus', 'TWminus'),
}


def _major_bkg_group(dataset):
    """Return the major-background family ('tt'/'wjets'/'tW') a dataset belongs
    to, or None if it is not one of the constrained major backgrounds."""
    for group, patterns in MAJOR_BKG_PATTERNS.items():
        if any(p in dataset for p in patterns):
            return group
    return None

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


def split_background_by_process(grouped):
    """Split the merged dataset dict into per-major-process background arrays.

    Returns {region: {group: {'phh': array, 'weight': array}}} where `group` is
    one of the major-background families ('tt', 'wjets', 'tW') or 'other' for any
    background dataset not matching a major family. Signal ('GluGlu') and data are
    excluded. Used by the DP binning to enforce per-process positivity (constraint
    a), which requires keeping the major backgrounds separate rather than summed.
    """
    groups = list(MAJOR_BKG_PATTERNS.keys()) + ['other']
    out = {r: {g: {'phh': [], 'weight': []} for g in groups} for r in REGIONS}
    for dataset, regions in grouped.items():
        name_lc = dataset.lower()
        if 'data' in name_lc or 'GluGlu' in dataset:
            continue
        group = _major_bkg_group(dataset) or 'other'
        for region, arrs in regions.items():
            out[region][group]['phh'].append(arrs['phh'])
            out[region][group]['weight'].append(arrs['weight'])
    return {r: {g: {k: (np.concatenate(v) if v else np.array([]))
                    for k, v in arrs.items()}
                for g, arrs in gd.items()}
            for r, gd in out.items()}


# ── DP binning: maximize Sum z^2 over k contiguous bins ───────────────────────
# Implements the HH-combine binning strategy: partition a finely-binned score
# axis into at most k_max bins maximizing Z_A^2 = Sum s^2/(s+b+sigma_s^2+sigma_b^2)
# subject to per-bin constraints, then pick the smallest k whose estimated
# mu_UL = 1 + 1.64/sqrt(Z_A^2) is within (1+eps) of the best.

_DP_NEG = -1e18


def _extract_dp_arrays(
    sig_phh: NDArray, sig_w: NDArray, bkg_phh: NDArray, bkg_w: NDArray,
    m_fine: int
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    '''
    Histogram the signal and background event arrays onto `m_fine` uniform fine
    bins on [0, 1]. Returns per-fine-bin signal yield s, background yield b,
    background variance vb (sum w^2), signal variance vs, and the fine bin edges.
    '''
    bin_edges = np.linspace(0.0, 1.0, m_fine + 1)
    s, _  = np.histogram(sig_phh, bins=bin_edges, weights=sig_w)
    vs, _ = np.histogram(sig_phh, bins=bin_edges, weights=sig_w ** 2)
    b, _  = np.histogram(bkg_phh, bins=bin_edges, weights=bkg_w)
    vb, _ = np.histogram(bkg_phh, bins=bin_edges, weights=bkg_w ** 2)
    return s, b, vb, vs, bin_edges


def _build_segment_tables(
    s: NDArray, b: NDArray, vb: NDArray, vs: NDArray,
    neff_thresh: float, z2_min: float, pos_prefix: list[NDArray]
) -> tuple[list, list]:
    '''
    Per-segment z^2 and feasibility tables for the DP.

    For every `end` in [1, nbins], seg_z2[end][start] is the z^2 of the merged
    segment [start, end) and seg_ok[end][start] its feasibility:
        z^2  = s^2 / (s + b + sigma_s^2 + sigma_b^2)
        ok   = (neff = b^2/sigma_b^2 >= neff_thresh)
             & (z^2 >= z2_min)
             & (every pos_prefix process has yield > 0 over the segment)
    All quantities are computed from prefix sums, so building the tables is
    O(nbins^2) with vectorized inner arrays.
    '''
    nbins = len(s)
    S  = np.concatenate([[0.0], np.cumsum(s)])
    B  = np.concatenate([[0.0], np.cumsum(b)])
    VS = np.concatenate([[0.0], np.cumsum(vs)])
    VB = np.concatenate([[0.0], np.cumsum(vb)])
    seg_z2 = [np.array([])] * (nbins + 1)
    seg_ok = [np.array([], dtype=bool)] * (nbins + 1)
    for end in range(1, nbins + 1):
        starts = np.arange(end)
        seg_s  = S[end]  - S[starts]
        seg_b  = B[end]  - B[starts]
        seg_vs = VS[end] - VS[starts]
        seg_vb = VB[end] - VB[starts]
        denom = seg_s + seg_b + seg_vs + seg_vb
        z2 = np.where(denom > 0,
                      seg_s ** 2 / np.where(denom > 0, denom, 1.0), 0.0)
        neff = np.where(seg_vb > 0,
                        seg_b ** 2 / np.where(seg_vb > 0, seg_vb, 1.0), 0.0)
        ok = (neff >= neff_thresh) & (z2 >= z2_min)
        for P in pos_prefix:
            ok &= (P[end] - P[starts]) > 0
        seg_z2[end] = z2
        seg_ok[end] = ok
    return seg_z2, seg_ok


def _run_dp(
    seg_z2: list[NDArray], seg_ok: list[NDArray],
    max_bins: int, nbins: int
) -> tuple[NDArray, NDArray]:
    '''
    DP forward pass: Build a table `best_total[k, end]` containing sum(z^2)
    `[k, end]` represents the optimal partitioning of [0, end) fine bins into
    `k` segments, using `seg_z2` and `seg_ok`. Hence, `best_total` is upper
    -triangular. This is built from the starting point `best_total[0,0]=0.0`
    iteratively.
    `split[k, end]` it also built iteratively, and holds the optimal
    starting bin number `start` for the final partition of `[0, end)` into `k`
    bins. The full optimal bin edges can be extracted by recursively querying
    `split[k, end]->split[k-1,split[k,end]]->...`.
    '''
    best_total = np.full((max_bins + 1, nbins + 1), _DP_NEG)
    split = np.zeros((max_bins + 1, nbins + 1), dtype=int)
    best_total[0, 0] = 0.0
    for k in range(1, max_bins + 1):
        prev_row = best_total[k - 1]
        for end in range(1, nbins + 1):
            candidates = np.where(
                seg_ok[end], prev_row[:end] + seg_z2[end], _DP_NEG
            )
            start = int(np.argmax(candidates))
            best_total[k, end] = candidates[start]
            split[k, end] = start
    return best_total, split


def _select_k_at_knee(dp_curve: NDArray, eps: float) -> int:
    '''
    Knee selection on the estimated mu=1 upper limit mu_UL = 1 + 1.64/sqrt(Sum z2),
    where 1.64 is the one-sided 95% normal quantile. Return the smallest feasible
    k whose mu_UL is within (1+eps) of the best.
    NaN entries from infeasible k fail the comparison.
    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        ul_curve = 1.0 + 1.64 / np.sqrt(dp_curve)
    ul_best = np.nanmin(ul_curve)
    return int(np.argmax(ul_curve <= (1.0 + eps) * ul_best)) + 1 # First occurrence, minimum k


def _backtrack_dp_boundaries(split: NDArray, k_best: int, nbins: int) -> list:
    '''Walk the DP split table to recover the boundary fine-bin indices.'''
    boundaries = [nbins]
    end = nbins
    for k in range(k_best, 0, -1):
        end = split[k][end]
        boundaries.append(end)
    return boundaries[::-1]


def _backtrack_dp_edges(split: NDArray, k_best: int, nbins: int, bin_edges: NDArray) -> NDArray:
    '''Walk the DP split table to recover the boundary fine-bin indices as edge values.'''
    boundaries = _backtrack_dp_boundaries(split, k_best, nbins)
    quants = bin_edges[boundaries]
    quants[0], quants[-1] = 0.0, 1.0
    return quants


def _sb_monotonicity_penalty(sb: NDArray) -> int:
    '''
    Departure from a monotonically increasing s/b spectrum. Walks the coarse bins
    left to right, tracking a running streak of consecutive decreases: the first
    decrease adds 1, a second consecutive decrease adds 2, and so on, so that long
    downward runs are penalized more heavily than isolated dips.
    '''
    penalty = 0
    streak = 0
    for i in range(1, len(sb)):
        if sb[i] < sb[i - 1]:
            streak += 1
            penalty += streak
        else:
            streak = 0
    return penalty


def _dp_sb_monotonicity_curve(
    split: NDArray, dp_curve: NDArray, s: NDArray, b: NDArray, nbins: int
) -> NDArray:
    '''
    s/b non-monotonicity penalty for the optimal partition at each bin count k.
    Backtracks the DP split table for every feasible k, forms the per-coarse-bin
    s/b spectrum, and scores it with _sb_monotonicity_penalty.  NaN where k is
    infeasible.
    '''
    S = np.concatenate([[0.0], np.cumsum(s)])
    B = np.concatenate([[0.0], np.cumsum(b)])
    mono_curve = np.full(len(dp_curve), np.nan)
    for k in range(1, len(dp_curve) + 1):
        if np.isnan(dp_curve[k - 1]):
            continue
        bounds = np.array(_backtrack_dp_boundaries(split, k, nbins))
        seg_s = S[bounds[1:]] - S[bounds[:-1]]
        seg_b = B[bounds[1:]] - B[bounds[:-1]]
        with np.errstate(divide='ignore', invalid='ignore'):
            sb = np.where(seg_b > 0, seg_s / seg_b, np.inf)
        mono_curve[k - 1] = _sb_monotonicity_penalty(sb)
    return mono_curve


def plot_dp_ul_curve(
    dp_curve: NDArray, k_best: int, eps: float, plot_path,
    mono_curve: Optional[NDArray]=None
) -> None:
    '''
    Diagnostic plot of the estimated mu=1 upper limit mu_UL = 1 + 1.64/sqrt(Sum z2)
    versus the number of bins k.  Reference lines mark the (1+eps)*mu_UL_best knee
    threshold and the selected k_best.  When mono_curve is given, a lower panel
    shows the s/b non-monotonicity penalty versus k (0 = perfectly monotonic).
    '''
    ks = np.arange(1, len(dp_curve) + 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        mu_ul = 1.0 + 1.64 / np.sqrt(np.asarray(dp_curve, dtype=float))
    ul_best = float(np.nanmin(mu_ul))

    if mono_curve is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig, (ax, ax_mono) = plt.subplots(
            2, 1, figsize=(8, 9), sharex=True,
            gridspec_kw={'height_ratios': [2, 1]},
        )

    ax.plot(ks, mu_ul, 'o-')
    ax.axhline((1.0 + eps) * ul_best, color='C1', ls=':',
               label=f'{eps:.0%} degradation threshold')
    ax.axvline(k_best, color='C3', ls='--', label=f'chosen $k$ = {k_best}')
    ax.set_ylabel(r'estimated $\mu_{UL}$')
    ax.set_ylim(0.98*ul_best, 1.1*ul_best)
    ax.legend(title=r'best $\mu_{UL}$' + f' = {ul_best:.3g}')

    if mono_curve is None:
        ax.set_xlabel(r'number of bins $k$')
    else:
        ax_mono.plot(ks, np.asarray(mono_curve, dtype=float), 'o-', color='C2')
        ax_mono.axvline(k_best, color='C3', ls='--')
        ax_mono.set_xlabel(r'number of bins $k$')
        ax_mono.set_ylabel('s/b non-monotonicity')
        ax_mono.set_ylim(bottom=0)

    fig.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)


def get_dp_optimal_bin_edges(
    sig_phh: NDArray, sig_w: NDArray, bkg_phh: NDArray, bkg_w: NDArray,
    max_bins: int=DEFAULT_MAX_BINS, neff_thresh: float=DEFAULT_MIN_NEFF,
    z2_min: float=DEFAULT_Z2_MIN, eps: float=DEFAULT_EPS,
    m_fine: int=DEFAULT_M_FINE, pos_events: list=None
) -> tuple[NDArray, Optional[NDArray], Optional[NDArray]]:
    '''
    Partitions the fine-binned score axis into at most `max_bins` contiguous bins
    so as to maximize the total Asimov mu=1 sensitivity Sum z2, where each bin's
    z2 = s^2 / (s + b + sigma_s^2 + sigma_b^2) is the expected significance
    squared for an Asimov mu=1 measurement including both signal and background
    MC statistical uncertainties. Every bin is constrained to have
    neff = b^2/sigma_b^2 >= `neff_thresh` for Barlow-Beeston-lite nuisances in
    Combine and z2 >= `z2_min`; the latter floor forbids thin, marginal bins
    (mostly at low score) that make the spectrum jagged without meaningfully
    improving sensitivity.

    The objective is additive over contiguous segments and every constraint is
    per-segment, so the DP returns the exact global optimum for this objective.

    Among the feasible bin counts k <= `max_bins`, the smallest k whose estimated
    upper limit mu_UL = 1 + 1.64/sqrt(sum z2) is within a fraction `eps` of the
    best is selected.

    Args:
        sig_phh, sig_w (NDArray): signal event scores and weights (weights
            include lumi/xsec normalization)
        bkg_phh, bkg_w (NDArray): summed-background event scores and weights
        max_bins (int): maximum number of bins
        neff_thresh (float): minimum effective background MC events per bin
        z2_min (float): minimum z2 contribution required of every bin
        eps (float): knee tolerance; pick the smallest k whose estimated mu_UL
            is within (1+eps) of the best mu_UL
        m_fine (int): number of uniform fine bins on [0, 1] the DP merges
        pos_events (list): optional list of (phh, weight) event-array pairs, one
            per process; a segment [j, m) is invalid unless every process's
            yield summed over that range is > 0.  Used to require individual
            background processes to be present in every bin.
    Returns:
        quants (NDArray): bin edges, endpoints fixed at 0 and 1
        dp_curve (NDArray): total z2 for each k = 1 .. max_bins,
            with NaN where k is infeasible
        mono_curve (NDArray): s/b non-monotonicity penalty for the optimal
            partition at each k = 1 .. max_bins, with NaN where k is infeasible
    '''
    s, b, vb, vs, bin_edges = _extract_dp_arrays(sig_phh, sig_w,
                                                 bkg_phh, bkg_w, m_fine)
    nbins = len(s)

    pos_prefix = []
    for phh, w in (pos_events or []):
        a, _ = np.histogram(phh, bins=bin_edges, weights=w)
        pos_prefix.append(np.concatenate(([0.0], np.cumsum(a))))
    max_bins = min(max_bins, nbins)

    seg_z2, seg_ok = _build_segment_tables(s, b, vb, vs, neff_thresh, z2_min, pos_prefix)
    best_total, split = _run_dp(seg_z2, seg_ok, max_bins, nbins)

    final_scores = best_total[1:max_bins + 1, nbins]
    if np.all(final_scores <= _DP_NEG / 2):
        return np.array([0., 1.]), None, None
    dp_curve = np.where(final_scores <= _DP_NEG / 2, np.nan, final_scores)

    k_best = _select_k_at_knee(dp_curve, eps)
    quants = _backtrack_dp_edges(split, k_best, nbins, bin_edges)
    mono_curve = _dp_sb_monotonicity_curve(split, dp_curve, s, b, nbins)

    return quants, dp_curve, mono_curve


def run_bin_optimization(input_dir, output_dir, n_quantiles=10000,
                         max_bins=DEFAULT_MAX_BINS, m_fine=DEFAULT_M_FINE,
                         min_neff_bkg=DEFAULT_MIN_NEFF, z2_min=DEFAULT_Z2_MIN,
                         eps=DEFAULT_EPS):
    """End-to-end: load directory, split sig/bkg, run the DP binning per region.

    For each region the summed-signal and summed-background score distributions
    are histogrammed onto `m_fine` uniform fine bins and `get_dp_optimal_bin_edges`
    picks the optimal <= `max_bins` coarse bins (max Sum z^2, knee rule at `eps`)
    subject to n_eff >= `min_neff_bkg`, z^2 >= `z2_min`, and strictly positive
    tt/wjets/tW yield per bin. The chosen edges go to
    `output_dir/bin_edges_{region}.txt`; a mu_UL-vs-k diagnostic plot goes to
    `output_dir/dp_ul_curve_{region}.png`. The fitted quantile transformer is
    still saved to `output_dir/quantiles_regressed_{region}.pkl` (diagnostic
    only — the DP does not use it).
    """
    os.makedirs(output_dir, exist_ok=True)
    grouped = load_phh_from_directory(input_dir)
    sig, bkg = split_signal_background(grouped)
    bkg_by_proc = split_background_by_process(grouped)

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
        for group in MAJOR_BKG_PATTERNS:
            g = bkg_by_proc[region][group]
            print(f"    {group:6s}: {g['phh'].size} events, "
                  f"sum(w) = {g['weight'].sum():.3f}")

        # Diagnostic only: flat-signal check + transformer pickle (not used by DP)
        transformer = WeightedQuantileTransformer(n_quantiles=n_quantiles,
                                                  output_distribution='uniform')
        transformer.fit(s_phh, sample_weight=s_w)
        plot_score(s_phh, s_w, transformer, f"HH_{region}", output_dir)
        transformer.save(os.path.join(output_dir,
                                      f"quantiles_regressed_{region}.pkl"))

        pos_events = [(bkg_by_proc[region][g]['phh'],
                       bkg_by_proc[region][g]['weight'])
                      for g in MAJOR_BKG_PATTERNS]

        edges, dp_curve, mono_curve = get_dp_optimal_bin_edges(
            s_phh, s_w, b_phh, b_w,
            max_bins=max_bins, neff_thresh=min_neff_bkg,
            z2_min=z2_min, eps=eps, m_fine=m_fine,
            pos_events=pos_events)

        if dp_curve is None:
            print(f"  WARNING: no feasible binning at any k <= {max_bins} — "
                  f"even a single bin violates the constraints "
                  f"(n_eff >= {min_neff_bkg} or process positivity). "
                  f"Writing a single [0, 1] bin.")
            k_best = 1
        else:
            k_best = len(edges) - 1
            za2 = dp_curve[k_best - 1]
            mu_ul = 1.0 + 1.64 / np.sqrt(za2)
            print(f"  chosen k = {k_best} (knee rule, eps = {eps:.0%})")
            print(f"  Z_A^2 = {za2:.4f}  ->  estimated mu_UL = {mu_ul:.3f}")
            print(f"  s/b non-monotonicity penalty = {mono_curve[k_best - 1]:.0f}")
            plot_path = os.path.join(output_dir, f"dp_ul_curve_{region}.png")
            plot_dp_ul_curve(dp_curve, k_best, eps, plot_path, mono_curve)
            print(f"  mu_UL-vs-k plot written to {plot_path}")

        edges_path = os.path.join(output_dir, f"bin_edges_{region}.txt")
        with open(edges_path, 'w') as f:
            f.write(f"# region={region}  n_bins={k_best}  max_bins={max_bins}  "
                    f"m_fine={m_fine}  min_neff_bkg={min_neff_bkg}  "
                    f"z2_min={z2_min}  eps={eps}\n")
            f.write(", ".join(f"{e:.6f}" for e in edges) + "\n")
        print(f"  bin edges written to {edges_path}")
        summary[region] = {
            'edges': edges,
            'k_best': k_best,
            'dp_curve': dp_curve,
            'mono_curve': mono_curve,
        }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantile regression of ML classifier HH score")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("-i", "--input", nargs="+", help="Input pkl files (single-sample legacy mode)")
    src.add_argument("--input-dir", help=(
        "Directory (local or EOS root:// URL) containing phh_hist_*.pkl chunk "
        "files. Files are grouped by dataset, split into signal (GluGlu*) and "
        "background (everything else except 'data'), and the DP bin "
        "optimization is run per region."))
    parser.add_argument("-o", "--output", help="Output folder for fitted quantile transformer", required=True)
    parser.add_argument("-n", "--n_quantiles", type=int, default=10000, help="Number of quantiles", required=False)
    parser.add_argument("-b", "--max-bins", dest="max_bins", type=int,
                        default=DEFAULT_MAX_BINS,
                        help="Maximum number of final bins k_max for the DP "
                             "(--input-dir mode); in legacy -i mode, the number "
                             "of equal-probability bin edges printed")
    parser.add_argument("--m-fine", type=int, default=DEFAULT_M_FINE,
                        help="Number of uniform fine bins on [0, 1] the DP merges")
    parser.add_argument("--min-neff", type=float, default=DEFAULT_MIN_NEFF,
                        help="Minimum effective unweighted background events "
                             "n_eff=(sum w)^2/(sum w^2) required per bin")
    parser.add_argument("--z2-min", type=float, default=DEFAULT_Z2_MIN,
                        help="Minimum per-bin z^2 contribution")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS,
                        help="Knee tolerance: smallest k with estimated mu_UL "
                             "within (1+eps) of the best")
    parser.add_argument("-r", "--region", choices=list(REGIONS), default="nominal_4j2b",
                        nargs="+" , help="Region to use for fitting in legacy -i mode")
    args = parser.parse_args()

    if args.max_bins < 1:
        parser.error(f"-b/--max-bins must be >= 1, got {args.max_bins}")

    os.makedirs(args.output, exist_ok=True)

    if args.input_dir is not None:
        run_bin_optimization(
            input_dir=args.input_dir,
            output_dir=args.output,
            n_quantiles=args.n_quantiles,
            max_bins=args.max_bins,
            m_fine=args.m_fine,
            min_neff_bkg=args.min_neff,
            z2_min=args.z2_min,
            eps=args.eps,
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
        bin_edges = np.interp(np.linspace(0, 1, args.max_bins + 1), transformer.reference_quantiles_, transformer.quantiles_)
        print(f"\n{args.max_bins} equal-probability bin edges in original score space:")
        print(np.array2string(bin_edges, separator=', '))
