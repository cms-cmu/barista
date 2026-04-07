import os
import pickle
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantile regression of ML classifier HH score")
    parser.add_argument("-i", "--input", nargs="+", help="Input pkl files", required=True)
    parser.add_argument("-o", "--output", help="Output folder for fitted quantile transformer", required=True)
    parser.add_argument("-n", "--n_quantiles", type=int, default=10000, help="Number of quantiles", required=False)
    parser.add_argument("-b", "--n_bins", type=int, default=50, help="Number of equal-probability bins to print")
    parser.add_argument("-r", "--region", choices=["nominal_4j2b", "lowpt_4j2b"], default="nominal_4j2b",
                        help="Region to use for fitting (default: nominal_4j2b)")
    args = parser.parse_args()

    for file in args.input:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Input file '{file}' does not exist.")
    os.makedirs(args.output, exist_ok=True)
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