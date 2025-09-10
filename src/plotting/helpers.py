import hist
import numpy as np
import os
import yaml
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from hist.intervals import ratio_uncertainty
from src.physics.dihiggs.di_higgs import Coupling, ggF

# Constants
EPSILON = 0.001  # Small value to avoid division by zero in ratio calculations
PHI = (1 + np.sqrt(5)) / 2

# Color palette for plots
COLORS = [
    "xkcd:black", "xkcd:red", "xkcd:off green", "xkcd:blue",
    "xkcd:orange", "xkcd:violet", "xkcd:grey", "xkcd:pink",
    "xkcd:pale blue",
    "xkcd:black", "xkcd:red", "xkcd:off green", "xkcd:blue",
    "xkcd:orange", "xkcd:violet", "xkcd:grey", "xkcd:pink",
]

# Dictionary and List Operations
def get_value_nested_dict(nested_dict: Dict[str, Any], target_key: str, debug: bool = False) -> Any:
    """Return the first value matching the target key from a nested dictionary.

    Args:
        nested_dict: The nested dictionary to search through
        target_key: The key to find in the dictionary

    Returns:
        The value associated with the target key

    Raises:
        ValueError: If the target key is not found in the dictionary
    """
    if debug: print("Searching in nested_dict:", nested_dict)

    for k, v in nested_dict.items():
        if k == target_key:
            return v

        if isinstance(v, dict):
            try:
                return get_value_nested_dict(v, target_key, debug=debug)
            except ValueError:
                continue

    raise ValueError(f"target_key {target_key} not in nested_dict")

def compare_dict_keys_with_list(dict1: Dict[str, Any], list2: List[str]) -> Tuple[Set[str], Set[str]]:
    """Compare the keys of a dictionary with the elements of a list.

    Args:
        dict1: The dictionary to compare
        list2: The list to compare against

    Returns:
        A tuple containing:
            - common_keys: Set of keys present in both dictionary and list
            - unique_to_dict1: Set of keys only in the dictionary
    """
    keys1 = set(dict1.keys())
    list2_set = set(list2)

    common_keys = keys1.intersection(list2_set)
    unique_to_dict1 = keys1.difference(list2_set)

    return common_keys, unique_to_dict1

# Histogram Creation
def make_hist(
    *,
    edges: np.ndarray,
    values: np.ndarray,
    variances: np.ndarray,
    x_label: str,
    under_flow: float,
    over_flow: float,
    add_flow: bool
) -> hist.Hist:
    """Create a 1D histogram with weighted storage.

    Args:
        edges: Bin edges for the histogram
        values: Bin values
        variances: Bin variances
        x_label: Label for the x-axis
        under_flow: Value for underflow bin
        over_flow: Value for overflow bin
        add_flow: Whether to add under/overflow values

    Returns:
        A hist.Hist object with the specified configuration
    """
    hist_obj = hist.Hist(
        hist.axis.Variable(edges, name=x_label),
        storage=hist.storage.Weight()
    )

    if add_flow:
        values[0] += under_flow
        values[-1] += over_flow

    hist_obj[...] = np.array(
        list(zip(values, variances)),
        dtype=[("value", "f8"), ("variance", "f8")]
    )

    return hist_obj

def make_2d_hist(
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    values: np.ndarray,
    variances: np.ndarray,
    x_label: str,
    y_label: str
) -> hist.Hist:
    """Create a 2D histogram with weighted storage.

    Args:
        x_edges: Bin edges for the x-axis
        y_edges: Bin edges for the y-axis
        values: Bin values
        variances: Bin variances
        x_label: Label for the x-axis
        y_label: Label for the y-axis

    Returns:
        A hist.Hist object with the specified configuration
    """
    hist_obj = hist.Hist(
        hist.axis.Variable(x_edges, name=x_label),
        hist.axis.Variable(y_edges, name=y_label),
        storage=hist.storage.Weight()
    )

    hist_obj[...] = np.array(
        list(zip(np.ravel(values), np.ravel(variances))),
        dtype=[("value", "f8"), ("variance", "f8")]
    ).reshape(len(x_edges) - 1, len(y_edges) - 1)

    return hist_obj

# Physics-specific Functions
def make_klambda_hist(kl_value: str, plot_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Create a k-lambda histogram from plot data.

    Args:
        kl_value: The k-lambda value as a string (e.g., "HH4b_kl1")
        plot_data: Dictionary containing plot data for different k-lambda values

    Returns:
        Dictionary containing weighted plot data for the specified k-lambda value
    """
    kl_target = float(kl_value.replace("HH4b_kl", ""))

    plot_data_0 = get_value_nested_dict(plot_data, "HH4b_kl0")
    plot_data_1 = get_value_nested_dict(plot_data, "HH4b_kl1")
    plot_data_2_45 = get_value_nested_dict(plot_data, "HH4b_kl2p45")
    plot_data_5 = get_value_nested_dict(plot_data, "HH4b_kl5")

    basis = ggF(Coupling(dict(kl=0.0), dict(kl=1.0), dict(kl=2.45), dict(kl=5.0)))
    target_weights = basis.weight(Coupling(kl=kl_target))[0]

    w_0, w_1, w_2_45, w_5 = target_weights

    plot_data_kl = {}
    for key in ["values", "variances", "under_flow", "over_flow"]:
        plot_data_kl[key] = (
            w_0 * np.array(plot_data_0[key]) +
            w_1 * np.array(plot_data_1[key]) +
            w_2_45 * np.array(plot_data_2_45[key]) +
            w_5 * np.array(plot_data_5[key])
        )

    return plot_data_kl

# File Operations
def savefig(fig: Any, file_name: Union[str, List[str]], *args: Any) -> None:
    """Save a figure to a PDF file.

    Args:
        fig: The figure object to save
        file_name: Name of the output file (string or list of strings)
        *args: Additional path components
    """
    args_str = []
    for arg in args:
        args_str.append("_vs_".join(arg) if isinstance(arg, list) else arg)

    output_path = "/".join(args_str)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_name = file_name if isinstance(file_name, str) else "_vs_".join(file_name)
    file_path = f"{output_path}/{file_name.replace('.', '_').replace('/', '_')}.pdf"
    print(f"wrote pdf: {file_path}")
    fig.savefig(file_path)

def save_yaml(plot_data: Dict[str, Any], var: Union[str, List[str]], *args: Any) -> None:
    """Save plot data to a YAML file.

    Args:
        plot_data: Data to save
        var: Variable name (string or list of strings)
        *args: Additional path components
    """
    args_str = []
    for arg in args:
        args_str.append("_vs_".join(arg) if isinstance(arg, list) else arg)

    output_path = "/".join(args_str)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    var_str = var if isinstance(var, str) else "_vs_".join(var)
    file_name = f"{output_path}/{var_str.replace('.', '_').replace('/', '_')}.yaml"
    print(f"wrote yaml: {file_name}")

    def clean_for_yaml(obj):
        """Recursively clean object for safe YAML serialization."""
        if obj == sum:  # Functions, classes, etc.
            return "sum"  # Convert to string representation
        elif isinstance(obj, list):
            return [clean_for_yaml(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: clean_for_yaml(v) for k, v in obj.items()}
        else:
            return obj

    cleaned_data = clean_for_yaml(plot_data)

    with open(file_name, "w") as yfile:
        yaml.safe_dump(cleaned_data, yfile, default_flow_style=False, sort_keys=False)

# Utility Functions
def get_cut_dict(cut: str, cut_list: List[str]) -> Dict[str, Any]:
    """Create a dictionary of cuts with sum as default value.

    Args:
        cut: The main cut to set as True
        cut_list: List of all cuts

    Returns:
        Dictionary with cuts as keys and sum as default value
    """
    cut_dict = {c: sum for c in cut_list}
    cut_dict[cut] = True
    return cut_dict

def get_label(default_str: str, override_list: Optional[List[str]], i: int) -> str:
    """Get a label from override list or use default.

    Args:
        default_str: Default label to use if override is not available
        override_list: Optional list of override labels
        i: Index of the label to get

    Returns:
        The label to use
    """
    return override_list[i] if (override_list and len(override_list) > i) else default_str

def make_ratio(
    num_values: np.ndarray,
    num_vars: np.ndarray,
    den_values: np.ndarray,
    den_vars: np.ndarray,
    epsilon: float = EPSILON,  # Use constant EPSILON as default
    **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate ratio and uncertainty between two histograms.

    Args:
        num_values: Numerator values
        num_vars: Numerator variances
        den_values: Denominator values
        den_vars: Denominator variances
        epsilon: Small value to avoid division by zero (defaults to EPSILON)
        **kwargs: Additional options (e.g., norm for normalization)

    Returns:
        Tuple of (ratios, ratio_uncertainties)
    """
    ratios = num_values / den_values
    ratios[np.isnan(ratios)] = 0

    if kwargs.get("norm", False):
        num_sf = np.sum(num_values, axis=0)
        den_sf = np.sum(den_values, axis=0)
        ratios *= den_sf / num_sf

    ratios[ratios == 0] = np.nan
    ratios[np.isinf(ratios)] = np.nan
    ratios[den_values == 0] = np.nan

    num_values[num_values == 0] = epsilon
    ratio_uncert = np.abs(ratios) * np.sqrt(num_vars * np.power(num_values, -2.0))
    ratio_uncert = np.nan_to_num(ratio_uncert, nan=1)

    return ratios, ratio_uncert

def get_year_str(year: Union[str, List[str]]) -> str:
    """Convert year format to string.

    Args:
        year: Year or list of years to convert

    Returns:
        Formatted year string
    """
    if year == sum:
        return "sum"
    if isinstance(year, list):
        return "_vs_".join(year)
    return year.replace("UL", "20")

def get_axis_str(axis: Union[str, List[str]]) -> str:
    """Convert axis format to string.

    Args:
        region: Region or list of regions to convert

    Returns:
        Formatted region string
    """
    if isinstance(axis, list):
        str_axis = ["sum" if i == sum else i for i in axis]
        return " vs ".join(str_axis)
    if axis == sum:
        return "sum"
    return axis
