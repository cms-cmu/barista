"""
Helper functions for creating plot dictionaries from histogram data.

This module provides functions to:
- Extract histogram data from input files
- Create plot configurations for different types of plots (1D, 2D, ratios)
- Handle stacked and unstacked histograms
- Manage plot metadata and styling
"""

from typing import Dict, List, NamedTuple, Optional, Union, Any, Tuple
import copy
import logging
import src.plotting.helpers as plot_helpers
from src.plotting.plot_types import PlotData, RatioSpec, HistSource
import hist
import numpy as np
from rich.pretty import pretty_repr

logger = logging.getLogger(__name__)


def print_list_debug_info(process, cut, axis_opts):
    print(f" hist process={process}, "
          f"cut={cut}, "
          f"axis_opts={axis_opts}")


#
#  Get hist values — private helpers
#

# Config keys that control styling, not histogram indexing
_STYLE_KEYS = frozenset([
    "process", "scalefactor", "label", "fillcolor", "edgecolor",
    "histtype", "alpha", "linewidth", "linestyle", "zorder", "year",
])


def _normalize_year(year: str):
    """Convert multi-year aliases (RunII, Run2, …) to the hist sum sentinel."""
    return sum if year in ("RunII", "Run2", "Run3", "RunIII") else year


def _build_hist_opts(
    process: str, year, config: Dict, axis_opts: Dict,
    cut: Optional[str], cfg: Any, debug: bool
) -> Tuple[Dict, Dict]:
    """Assemble the hist indexing dict from config, axis_opts, and cut; return (hist_opts, cut_dict)."""
    opts: Dict = {"process": process, "year": year}
    for c_key, c_val in config.items():
        if c_key not in _STYLE_KEYS:
            if debug:
                print(f"Adding to hist_opts: {c_key} = {c_val}")
            opts[c_key] = c_val
    opts = opts | axis_opts

    cut_dict: Dict = {}
    if cut is not None:
        try:
            cut_dict = plot_helpers.get_cut_dict(cut, cfg.cutList)
        except (AttributeError, KeyError) as e:
            raise AttributeError(f"Failed to get cut dictionary: {str(e)}")
        opts = opts | cut_dict

    return opts, cut_dict


def _find_hist_obj(
    cfg: Any, var: str, process: str, hist_opts: Dict, file_index: Optional[int]
) -> hist.Hist:
    """Locate the hist.Hist object and prune hist_opts of keys unknown to that file."""
    hist_key     = cfg.hist_key     if hasattr(cfg, 'hist_key')     else 'hists'
    category_key = cfg.category_key if hasattr(cfg, 'category_key') else 'categories'

    if len(cfg.hists) > 1 and not cfg.combine_input_files:
        if file_index is None:
            raise ValueError("Must provide file_index when using multiple input files without combine_input_files")
        try:
            _, unique_to_dict = plot_helpers.compare_dict_keys_with_list(hist_opts, cfg.hists[file_index][category_key])
        except (KeyError, IndexError) as e:
            raise ValueError(f"Failed to compare dictionary keys (multiple hists): {str(e)}")
        for _key in unique_to_dict:
            hist_opts.pop(_key)
        try:
            hist_obj = cfg.hists[file_index][hist_key][var]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Failed to get histogram for var {var}: {str(e)}")
        if "variation" in cfg.hists[file_index][category_key]:
            hist_opts["variation"] = "nominal"
        return hist_obj

    hist_obj = None
    for _input_data in cfg.hists:
        try:
            _, unique_to_dict = plot_helpers.compare_dict_keys_with_list(hist_opts, _input_data[category_key])
        except (KeyError, AttributeError) as e:
            raise ValueError(f"Failed to compare dictionary keys: {str(e)}")
        for _key in unique_to_dict:
            hist_opts.pop(_key)
        if var not in _input_data[hist_key]:
            continue
        try:
            available_processes = list(_input_data[hist_key][var].axes["process"])
        except (KeyError, StopIteration):
            available_processes = None
        if available_processes is not None and process in available_processes:
            if "variation" in _input_data[category_key]:
                hist_opts["variation"] = "nominal"
            hist_obj = _input_data[hist_key][var]

    if hist_obj is None:
        try:
            avail = list(_input_data[hist_key][var].axes["process"]) if var in _input_data[hist_key] else None
        except Exception:
            avail = None
        avail_str = f" (available processes: {avail})" if avail is not None else f" (var '{var}' not found in {hist_key})"
        raise ValueError(f"get_hist_data Could not find histogram for var {var} with process {process} in inputs{avail_str}")
    return hist_obj


def _apply_intcategory_compat(
    hist_obj: hist.Hist, hist_opts: Dict, axis_opts: Dict, cfg: Any, config: Dict
) -> None:
    """Translate string tag/region values to IntCategory hist.loc() lookups (in-place)."""
    try:
        for axis in hist_obj.axes:
            if axis.name == "tag" and isinstance(axis, hist.axis.IntCategory):
                hist_opts['tag'] = hist.loc(cfg.plotConfig["codes"]["tag"][config["tag"]])
            if axis.name == "region" and isinstance(axis, hist.axis.IntCategory):
                region_val = axis_opts.get('region', None)
                if isinstance(region_val, list):
                    hist_opts['region'] = [hist.loc(cfg.plotConfig["codes"]["region"][i]) for i in hist_opts['region']]
                elif region_val and region_val not in ("sum", sum):
                    hist_opts['region'] = hist.loc(cfg.plotConfig["codes"]["region"][region_val])
    except (KeyError, AttributeError) as e:
        raise ValueError(f"Failed to handle axis compatibility: {str(e)}")


def _remove_missing_cut_keys(
    hist_obj: hist.Hist, hist_opts: Dict, cut_dict: Dict, debug: bool
) -> None:
    """Drop cut axes from hist_opts that are absent from this histogram (in-place)."""
    for cut_key in cut_dict:
        if debug:
            print(f"Checking cut_key {cut_key} in hist_obj.axes {hist_obj.axes.name}")
        if cut_key not in hist_obj.axes.name and cut_key in hist_opts:
            if debug:
                print(f"Removing cut_key {cut_key} from hist_opts {hist_opts}")
            hist_opts.pop(cut_key)


def _select_hist(
    hist_obj: hist.Hist, hist_opts: Dict, rebin: int, do2d: bool, debug: bool
) -> hist.Hist:
    """Apply rebin + index to produce the selected histogram slice."""
    if not do2d:
        var_name = hist_obj.axes[-1].name
        hist_opts = hist_opts | {var_name: hist.rebin(rebin)}
    try:
        if debug:
            print(f"hist_opts are {hist_opts}")
        return hist_obj[hist_opts]
    except Exception as e:
        raise ValueError(
            f"helpers_make_plot_dict::Failed to select histogram: {str(e)} hist_opts was {hist_opts}"
        )


def _squeeze_hist(selected_hist: hist.Hist, do2d: bool) -> hist.Hist:
    """Collapse any extra leading dimension left after process/year selection."""
    if do2d:
        if len(selected_hist.shape) == 3:
            return selected_hist[sum, :, :]
    else:
        if len(selected_hist.shape) == 2:
            return selected_hist[sum, :]
    return selected_hist


#
#  Get hist values — public entry point
#
def get_hist_data(*, process: str, cfg: Any, config: Dict, var: str, cut: Optional[str], rebin: int, year: str, axis_opts: Dict, do2d: bool = False, file_index: Optional[int] = None, debug: bool = False) -> hist.Hist:
    """
    Extract histogram data for a given process and configuration.

    Args:
        process: Name of the process to extract
        cfg: Configuration object containing histogram data
        config: Dictionary of plot configuration options
        var: Variable to plot
        cut: Selection cut to apply
        rebin: Rebinning factor
        year: Data taking year
        do2d: Whether to extract 2D histogram data
        axis_opts: Axis options for histogram selection
        file_index: Index of input file to use (if multiple files)
        debug: Enable debug output

    Returns:
        hist.Hist: Extracted histogram data

    Raises:
        ValueError: If histogram data cannot be found
        TypeError: If input parameters are of incorrect type
        AttributeError: If required configuration attributes are missing
    """
    if not isinstance(process, str):
        raise TypeError(f"get_hist_data::process must be a string, got {type(process)}")
    if not isinstance(var, str):
        raise TypeError(f"get_hist_data::var must be a string, got {type(var)} = {var}")
    if cut is not None and not isinstance(cut, str):
        raise TypeError(f"get_hist_data::cut must be a string or None, got {type(cut)}")
    if not isinstance(rebin, int):
        raise TypeError(f"get_hist_data::rebin must be an integer, got {type(rebin)}")
    if rebin < 1:
        raise ValueError(f"rebin must be positive, got {rebin}")

    year = _normalize_year(year)
    if debug:
        print(f" in get_hist_data: hist process={process}, axis_opts={axis_opts}, year={year}, var={var}")

    hist_opts, cut_dict = _build_hist_opts(process, year, config, axis_opts, cut, cfg, debug)
    hist_obj             = _find_hist_obj(cfg, var, process, hist_opts, file_index)
    _apply_intcategory_compat(hist_obj, hist_opts, axis_opts, cfg, config)
    _remove_missing_cut_keys(hist_obj, hist_opts, cut_dict, debug)
    selected_hist        = _select_hist(hist_obj, hist_opts, rebin, do2d, debug)
    selected_hist        = _squeeze_hist(selected_hist, do2d)
    selected_hist       *= config.get("scalefactor", 1.0)

    if debug:
        print(f"helpers_make_plot_dict::get_hist_data Leaving")
    return selected_hist



#
def get_hist_data_list(*, proc_list: List[str], cfg: Any, config: Dict, var: str, cut: Optional[str], rebin: int, year: str, axis_opts: Dict, do2d: bool, file_index: Optional[int], debug) -> hist.Hist:
    """
    Extract and combine histogram data for a list of processes.

    Args:
        proc_list: List of process names to combine
        cfg: Configuration object containing histogram data
        config: Dictionary of plot configuration options
        var: Variable to plot
        cut: Selection cut to apply
        rebin: Rebinning factor
        year: Data taking year
        axis_opts: Axis options for histogram selection
        do2d: Whether to extract 2D histogram data
        file_index: Index of input file to use (if multiple files)
        debug: Enable debug output

    Returns:
        hist.Hist: Combined histogram data
    """
    if debug:
        print(f"In get_hist_data_list proc_list={proc_list} \n")


    selected_hist = None
    for _proc in proc_list:

        if debug:
            print(f" \t  process {_proc} \n")

        if isinstance(_proc, list):
            _selected_hist =  get_hist_data_list(proc_list=_proc, cfg=cfg, config=config, var=var,
                                                 cut=cut, rebin=rebin, year=year, do2d=do2d, axis_opts=axis_opts, file_index=file_index, debug=debug)
        else:
            _selected_hist = get_hist_data(process=_proc, cfg=cfg, config=config, var=var,
                                           cut=cut, rebin=rebin, year=year, axis_opts=axis_opts, do2d=do2d, file_index=file_index, debug=debug)

        if selected_hist is None:
            selected_hist = _selected_hist
        else:
            selected_hist += _selected_hist

    if debug:
        print(f"Leaving get_hist_data_list\n")

    return selected_hist


#
#  Get hist from input file(s)
#
def add_hist_data(*, cfg, config, var, cut, rebin, year, axis_opts, do2d=False, file_index=None, debug=False):

    if debug:
        print(f"In add_hist_data {config['process']} \n")

    proc_list = config['process'] if isinstance(config['process'], list) else [config['process']]

    selected_hist = get_hist_data_list(proc_list=proc_list, cfg=cfg, config=config, var=var,
                                       cut=cut, rebin=rebin, year=year, do2d=do2d, axis_opts=axis_opts, file_index=file_index, debug=debug)

    if do2d:

        # Extract counts and variances
        try:
            config["values"]    = selected_hist.view(flow=False)["value"].tolist()  # Bin counts (array)
            config["variances"] = selected_hist.view(flow=False)["variance"].tolist()  # Bin variances (array)
        except IndexError:
            config["values"]    = selected_hist.values()  # Bin counts (array)
            config["variances"] = selected_hist.variances()  # Bin variances (array)
        if config["variances"] is None:
            config["variances"] = np.zeros_like(config["values"])

        config["x_edges"]   = selected_hist.axes[0].edges.tolist()  # X-axis edges
        config["y_edges"]   = selected_hist.axes[1].edges.tolist()  # Y-axis edges
        config["x_label"]   = selected_hist.axes[0].label  # X-axis label
        config["y_label"]   = selected_hist.axes[1].label  # Y-axis label

    else:
        if debug: print(f"fetching the data\n")
        try:
            config["values"]     = selected_hist.values().tolist()
            config["variances"]  = selected_hist.variances().tolist()
            config["centers"]    = selected_hist.axes[0].centers.tolist()
            config["edges"]      = selected_hist.axes[0].edges.tolist()
            config["x_label"]    = selected_hist.axes[0].label
            config["under_flow"] = float(selected_hist.view(flow=True)["value"][0])
            config["over_flow"]  = float(selected_hist.view(flow=True)["value"][-1])
        except Exception as e:
            raise ValueError(f"Failed to extract histogram data in add_hist_data: {e}")
        if debug: print(f"DONE fetching the data\n")

    if debug: print(f"Leaving add_hist_data\n")

    return



def _create_base_plot_dict(var: str, cut: Any, axis_opts: Dict, **kwargs) -> Dict:
    """Create the base plot dictionary structure."""
    return {
        "hists": {},
        "stack": {},
        "ratio": {},
        "var": var,
        "cut": cut,
        "axis_opts": axis_opts,
        "kwargs": kwargs,
    }

def _get_proc_id(process_config: Dict) -> str:
    """Return a unique string identifier for a process config."""
    return process_config["label"] if isinstance(process_config["process"], list) else process_config["process"]

def _setup_overlay_config(process_config: Dict, item: Any, index: int,
                           label_override: Optional[List[str]], set_edge_color: bool = False) -> Dict:
    """Return a deep copy of process_config styled for one item in an overlay list."""
    config = copy.deepcopy(process_config)
    config["fillcolor"] = plot_helpers.COLORS[index]
    if set_edge_color:
        config["edgecolor"] = plot_helpers.COLORS[index]
    config["label"] = plot_helpers.get_label(f"{process_config['label']} {item}", label_override, index)
    config["histtype"] = "errorbar"
    return config

class LoadSpec(NamedTuple):
    """Everything needed to load one histogram entry.

    Produced by _entries_* functions and consumed by _load_hists.
    """
    config: Dict                   # per-item process config (deep-copied, styled)
    var: str                       # variable to fetch
    cut: Optional[str]             # cut to apply
    year: Any                      # str | builtins.sum
    axis_opts: Dict                # axis opts for this entry
    file_index: Optional[int]      # which input file (None = auto)
    key: str                       # key for plot_data["hists"]
    hist_key_override: Optional[str] = None  # if set, calls cfg.set_hist_key first


def _load_hists(plot_data: Dict, cfg: Any, entries: List[LoadSpec], *,
                rebin: int, do2d: bool, debug: bool) -> None:
    """Load a list of LoadSpec entries into plot_data["hists"].

    Single location that calls add_hist_data and stores each result, replacing
    the per-handler add_hist_data + dict-insertion pattern.
    """
    for entry in entries:
        if entry.hist_key_override is not None:
            cfg.set_hist_key(entry.hist_key_override)
        add_hist_data(cfg=cfg, config=entry.config, var=entry.var, cut=entry.cut,
                      rebin=rebin, year=entry.year, axis_opts=entry.axis_opts,
                      do2d=do2d, file_index=entry.file_index, debug=debug)
        plot_data["hists"][entry.key] = entry.config


def _entries_overlay(
    *,
    process_config: Dict,
    items: List,
    base_var: str,
    base_cut,
    base_year,
    base_axis_opts: Dict,
    label_fn=str,
    item_is_var: bool = False,
    item_is_cut: bool = False,
    item_is_year: bool = False,
    axis_opt_key: Optional[str] = None,
    set_edge_color: bool = False,
    label_override: Optional[List[str]] = None,
    hist_key_list: Optional[List[str]] = None,
) -> List[LoadSpec]:
    """Core loop shared by all single-process, iterate-one-dimension overlay builders.

    Exactly one of item_is_var / item_is_cut / item_is_year / axis_opt_key should be
    set to describe which dimension varies across items. All other dimensions are fixed
    at their base_* values.
    """
    proc_id = _get_proc_id(process_config)
    entries = []
    for i, item in enumerate(items):
        _config     = _setup_overlay_config(process_config, label_fn(item), i, label_override, set_edge_color)
        _var        = item if item_is_var  else base_var
        _cut        = item if item_is_cut  else base_cut
        _year       = item if item_is_year else base_year
        if axis_opt_key is not None:
            _axis_opts = copy.deepcopy(base_axis_opts)
            _axis_opts[axis_opt_key] = item
        else:
            _axis_opts = base_axis_opts
        entries.append(LoadSpec(
            config=_config, var=_var, cut=_cut, year=_year,
            axis_opts=_axis_opts, file_index=None,
            key=f"{proc_id}{item}{i}",
            hist_key_override=hist_key_list[i] if hist_key_list else None,
        ))
    return entries


def _entries_cut_list(*, process_config: Dict, var_to_plot: str, axis_opts: Dict,
                      cut_list: List[str], year: str,
                      label_override: Optional[List[str]] = None,
                      hist_key_list: Optional[List[str]] = None,
                      debug: bool = False) -> List[LoadSpec]:
    """Build LoadSpec entries for a list of cuts (one hist per cut)."""
    return _entries_overlay(
        process_config=process_config, items=cut_list,
        base_var=var_to_plot, base_cut=None, base_year=year, base_axis_opts=axis_opts,
        label_fn=plot_helpers.cut_to_label, item_is_cut=True,
        label_override=label_override, hist_key_list=hist_key_list,
    )


def _entries_axis_opts_list(*, process_config: Dict, var_to_plot: str, cut: Any,
                            axis_list_name: str, axis_list_values: List, axis_opts: Dict,
                            year: str, label_override: Optional[List[str]] = None,
                            debug: bool = False) -> List[LoadSpec]:
    """Build LoadSpec entries for a list of axis_opts values (one hist per value)."""
    return _entries_overlay(
        process_config=process_config, items=axis_list_values,
        base_var=var_to_plot, base_cut=cut, base_year=year, base_axis_opts=axis_opts,
        axis_opt_key=axis_list_name,
        label_override=label_override,
    )

def _add_ratio_plots(plot_data: Dict, **kwargs) -> None:
    """
    Add ratio plots to the plot configuration.

    Args:
        plot_data: Plot data dictionary
        **kwargs: Additional plotting options including do2d
    """
    do2d = kwargs.get("do2d", False)
    if do2d:
        _add_2d_ratio_plots(plot_data, **kwargs)
    else:
        _add_1d_ratio_plots(plot_data, **kwargs)

def get_plot_dict_from_list(*, cfg: Any, var: str, cut: str, axis_opts: Dict, process: Any, **kwargs) -> PlotData:
    """Create a plot dictionary from lists of processes, cuts, axis_opts, etc."""
    debug = kwargs.get("debug", False)
    rebin = kwargs.get("rebin", 1)
    do2d = kwargs.get("do2d", False)
    var_over_ride = kwargs.get("var_over_ride", {})
    label_override = kwargs.get("labels", None)
    year = kwargs.get("year", "RunII")
    file_labels = kwargs.get("fileLabels", [])
    hist_key_list = kwargs.get("hist_key_list", None)

    plot_data = _create_base_plot_dict(var, cut, axis_opts, **kwargs)
    plot_data["process"] = process

    # Resolve process config(s) from plotConfig
    if isinstance(process, list):
        var_to_plot = var
        process_config = []
        for p in process:
            try:
                process_config.append(plot_helpers.get_value_nested_dict(cfg.plotConfig, p))
            except ValueError:
                if "HH4b" in p:
                    process_config.append(plot_helpers.make_klambda_hist(p, cfg.plotConfig))
    else:
        try:
            process_config = plot_helpers.get_value_nested_dict(cfg.plotConfig, process)
        except ValueError:
            raise ValueError(f"\t ERROR process = {process} not in plotConfig! \n")
        var_to_plot = var_over_ride.get(process, var)

    # Detect active list dimension and build entries
    axis_opts_list = next(((k, v) for k, v in axis_opts.items() if isinstance(v, list)), None)

    if isinstance(cut, list):
        entries = _entries_cut_list(
            process_config=process_config, var_to_plot=var_to_plot,
            axis_opts=axis_opts, cut_list=cut, year=year,
            label_override=label_override, hist_key_list=hist_key_list, debug=debug,
        )
    elif len(cfg.hists) > 1 and not cfg.combine_input_files and isinstance(process, list):
        entries = _entries_process_list_multi_file(
            process_config=process_config, cfg=cfg, var=var,
            axis_opts=axis_opts, cut=cut, year=year,
            var_over_ride=var_over_ride, label_override=label_override,
            file_labels=file_labels, debug=debug,
        )
    elif len(cfg.hists) > 1 and not cfg.combine_input_files:
        entries = _entries_input_files(
            process_config=process_config, cfg=cfg, var_to_plot=var_to_plot,
            axis_opts=axis_opts, cut=cut, year=year,
            label_override=label_override, file_labels=file_labels, debug=debug,
        )
    elif isinstance(process, list):
        entries = _entries_process_list(
            process_config=process_config, var=var,
            axis_opts=axis_opts, cut=cut, year=year,
            var_over_ride=var_over_ride, label_override=label_override, debug=debug,
        )
    elif isinstance(var, list):
        entries = _entries_var_list(
            process_config=process_config, var_list=var,
            axis_opts=axis_opts, cut=cut, year=year,
            label_override=label_override, debug=debug,
        )
    elif isinstance(year, list):
        entries = _entries_year_list(
            process_config=process_config, var=var_to_plot,
            axis_opts=axis_opts, cut=cut, year_list=year,
            label_override=label_override, debug=debug,
        )
    elif axis_opts_list:
        axis_list_name, _ = axis_opts_list
        axis_list_values = axis_opts[axis_list_name]
        entries = _entries_axis_opts_list(
            process_config=process_config, var_to_plot=var_to_plot,
            axis_opts=axis_opts, cut=cut, year=year,
            axis_list_name=axis_list_name, axis_list_values=axis_list_values,
            label_override=label_override, debug=debug,
        )
    else:
        raise ValueError("Error: At least one parameter must be a list!")

    _load_hists(plot_data, cfg, entries, rebin=rebin, do2d=do2d, debug=debug)

    if kwargs.get("doRatio", True):
        _add_ratio_plots(plot_data, **kwargs)

    return plot_data


def load_stack_config(*, cfg: Any, stack_config: Dict, var: str, cut: str, axis_opts: Dict,  **kwargs) -> Dict:
    """
    Load and process stack configuration for plotting.

    Args:
        cfg: Configuration object
        stack_config: Dictionary of stack configuration options
        var: Variable to plot
        cut: Selection cut
        axis_opts: Axis options for histogram selection
        **kwargs: Additional plotting options

    Returns:
        Dict: Processed stack configuration
    """
    stack_dict = {}
    var_over_ride = kwargs.get("var_over_ride", {})
    rebin = kwargs.get("rebin", 1)
    year = kwargs.get("year", "RunII")
    debug = kwargs.get("debug", False)
    do2d = kwargs.get("do2d", False)

    for _proc_name, _proc_config in stack_config.items():
        proc_config = copy.deepcopy(_proc_config)
        var_to_plot = var_over_ride.get(_proc_name, var)

        if debug:
            print(f"stack_process is {_proc_name} var is {var_to_plot}")

        if proc_config.get("process", None):
            add_hist_data(cfg=cfg, config=proc_config,
                         var=var_to_plot, cut=cut, rebin=rebin, year=year,
                         axis_opts=axis_opts, do2d=do2d, debug=debug)
            stack_dict[_proc_name] = proc_config
        elif proc_config.get("sum", None):
            _handle_stack_sum(proc_config=proc_config, cfg=cfg, var_to_plot=var_to_plot, cut=cut, rebin=rebin, year=year, axis_opts=axis_opts, do2d=do2d, debug=debug, var_over_ride=var_over_ride)
            stack_dict[_proc_name] = proc_config
        else:
            raise ValueError("Error: Stack component must have either 'process' or 'sum' configuration")
    logger.debug(f"stack_dict {stack_dict}")
    return stack_dict

def _handle_stack_sum(*, proc_config: Dict, cfg: Any, var_to_plot: str,
                     cut: str, rebin: int, year: str, axis_opts: Dict, do2d: bool, debug: bool,
                     var_over_ride: Dict) -> None:
    """Handle stack components that are sums of processes."""
    for sum_proc_name, sum_proc_config in proc_config["sum"].items():
        sum_proc_config["year"] = proc_config["year"]
        var_to_plot = var_over_ride.get(sum_proc_name, var_to_plot)

        add_hist_data(cfg=cfg, config=sum_proc_config,
                     var=var_to_plot, cut=cut, rebin=rebin, year=year,
                     axis_opts=axis_opts, do2d=do2d, debug=debug)

    # Combine  and variances
    stack_values = [v["values"] for _, v in proc_config["sum"].items()]
    proc_config["values"] = np.sum(stack_values, axis=0).tolist()

    stack_variances = [v["variances"] for _, v in proc_config["sum"].items()]
    proc_config["variances"] = np.sum(stack_variances, axis=0).tolist()

    # Copy metadata from first sum component
    first_sum_entry = next(iter(proc_config["sum"].values()))
    proc_config["centers"] = first_sum_entry["centers"]
    proc_config["edges"] = first_sum_entry["edges"]
    proc_config["x_label"] = first_sum_entry["x_label"]

    # Combine under/overflow
    stack_under_flow = [v["under_flow"] for _, v in proc_config["sum"].items()]
    proc_config["under_flow"] = float(np.sum(stack_under_flow, axis=0).tolist())

    stack_over_flow = [v["over_flow"] for _, v in proc_config["sum"].items()]
    proc_config["over_flow"] = float(np.sum(stack_over_flow, axis=0))


def get_values_variances_centers_from_dict(hist_config: Dict, plot_data: Dict) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Extract values, variances and centers from a histogram source config.

    Args:
        hist_config: Dict with keys ``type`` ("hists" or "stack") and
            optionally ``key`` (histogram name within the source).
        plot_data: PlotData dict containing "hists" and "stack".

    Returns:
        Tuple of (values, variances, centers) as numpy arrays / list.

    Raises:
        ValueError: If ``hist_config["type"]`` is not "hists" or "stack".
    """
    if hist_config["type"] == "hists":
        num_data = plot_data["hists"][hist_config["key"]]
        return np.array(num_data["values"]), np.array(num_data["variances"]), num_data["centers"]

    if hist_config["type"] == "stack":
        return_values = np.sum([v["values"] for _, v in plot_data["stack"].items()], axis=0)
        return_variances = np.sum([v["variances"] for _, v in plot_data["stack"].items()], axis=0)
        centers = next(iter(plot_data["stack"].values()))["centers"]
        return return_values, return_variances, centers

    raise ValueError("ERROR: ratio needs to be of type 'hists' or 'stack'")


def add_ratio_plots(ratio_config: Dict, plot_data: Dict, **kwargs) -> None:
    """Populate plot_data["ratio_specs"] from an explicit YAML ratio config.

    Actual computation is deferred to _resolve_ratio_specs at render time.
    """
    for r_name, _r_config in ratio_config.items():
        r_config = copy.deepcopy(_r_config)
        num_cfg = r_config.get("numerator", {})
        den_cfg = r_config.get("denominator", {})

        # Everything except source-pointer and band config is render style.
        skip = {"numerator", "denominator", "bkg_err_band", "norm"}
        style = {k: v for k, v in r_config.items() if k not in skip}

        plot_data.setdefault("ratio_specs", []).append(RatioSpec(
            name=f"ratio_{r_name}",
            numerator=HistSource(source=num_cfg.get("type", "hists"), key=num_cfg.get("key")),
            denominator=HistSource(source=den_cfg.get("type", "hists"), key=den_cfg.get("key")),
            norm=kwargs.get("norm", r_config.get("norm", False)),
            style=style,
        ))

        default_band_config = {"color": "k", "type": "band", "hatch": "\\\\\\"}
        _band_config = r_config.get("bkg_err_band", default_band_config)
        if _band_config:
            band_style = copy.deepcopy(_band_config)
            band_style.setdefault("type", "band")
            plot_data["ratio_specs"].append(RatioSpec(
                name=f"band_{r_name}",
                numerator=None,
                denominator=HistSource(source=den_cfg.get("type", "hists"), key=den_cfg.get("key")),
                style=band_style,
            ))

def get_plot_dict_from_config(*, cfg: Any, var: str = 'selJets.pt',
                              cut: Optional[str] = None, axis_opts: Dict, **kwargs) -> PlotData:
    """
    Create a plot dictionary from configuration.

    Args:
        cfg: Configuration object
        var: Variable to plot
        cut: Selection cut
        axis_opts: Axis options for histogram selection
        **kwargs: Additional plotting options

    Returns:
        Dict: Plot configuration dictionary

    Raises:
        AttributeError: If cut is not in cutList
    """
    process = kwargs.get("process", None)
    year = kwargs.get("year", "RunII")
    rebin = kwargs.get("rebin", 1)
    do2d = kwargs.get("do2d", False)
    debug = kwargs.get("debug", False)

    if debug:
        print(f"in get_plot_dict_from_config hist process={process}, cut={cut}")


    # Make process a list if it exists and isn't one already
    if process is not None and not isinstance(process, list):
        process = [process]

    var_over_ride = kwargs.get("var_over_ride", {})

    if cut:
        _bare_cut = cut.lstrip("~")
        if _bare_cut not in cfg.cutList:
            raise AttributeError(f"{cut} not in cutList {cfg.cutList}")

    plot_data = _create_base_plot_dict(var, cut, axis_opts, **kwargs)
    if do2d:
        plot_data["process"] = process[0]
        plot_data["is_2d_hist"] = True

    # Get histogram configuration
    hist_config = cfg.plotConfig["hists"]
    if process is not None:
        hist_config = {key: hist_config[key] for key in process if key in hist_config}

    # Process each histogram
    for _proc_name, _proc_config in hist_config.items():
        proc_config = copy.deepcopy(_proc_config)
        proc_config["name"] = _proc_name
        var_to_plot = var_over_ride.get(_proc_name, var)

        add_hist_data(cfg=cfg, config=proc_config,
                     var=var_to_plot, cut=cut, rebin=rebin, year=year,
                     axis_opts=axis_opts, do2d=do2d, debug=debug)
        plot_data["hists"][_proc_name] = proc_config

    # Process stack configuration
    stack_config = cfg.plotConfig.get("stack", {})
    if process is not None:
        stack_config = {key: stack_config[key] for key in process if key in stack_config}

    plot_data["stack"] = load_stack_config(cfg=cfg, stack_config=stack_config,
                                           var=var, cut=cut, axis_opts=axis_opts,  **kwargs)

    # Add ratio plots if requested
    if kwargs.get("doRatio", True) and not do2d:
        ratio_config = cfg.plotConfig.get("ratios", {})
        if ratio_config:
            add_ratio_plots(ratio_config, plot_data, **kwargs)

    return plot_data

def _entries_input_files(*, process_config: Dict, cfg: Any, var_to_plot: str,
                         axis_opts: Dict, cut: Any, year: str,
                         label_override: Optional[List[str]] = None,
                         file_labels: Optional[List[str]] = None,
                         debug: bool = False) -> List[LoadSpec]:
    """Build LoadSpec entries for multiple input files (one hist per file)."""
    file_labels = file_labels or []
    proc_id = process_config["label"] if isinstance(process_config["process"], list) else process_config["process"]
    entries = []
    for iF in range(len(cfg.hists)):
        _process_config = copy.deepcopy(process_config)
        _process_config["fillcolor"] = plot_helpers.COLORS[iF]
        _process_config["histtype"] = "errorbar"
        if label_override:
            _process_config["label"] = label_override[iF]
        elif iF < len(file_labels):
            _process_config["label"] = f"{_process_config['label']} {file_labels[iF]}"
        else:
            _process_config["label"] = f"{_process_config['label']} file{iF + 1}"
        entries.append(LoadSpec(
            config=_process_config, var=var_to_plot, cut=cut, year=year,
            axis_opts=axis_opts, file_index=iF, key=f"{proc_id}file{iF}",
        ))
    return entries

def get_var_to_plot(var, var_over_ride: Dict, proc_id: str, iP: int, debug: bool) -> str:
        """Get the variable to plot, considering overrides."""
        this_var = var
        if isinstance(var, list):
            this_var = var[iP]

        return var_over_ride.get(proc_id, this_var)


def _prepare_process_config(proc_conf: Dict):
    """Deepcopy proc_conf, set histtype to errorbar, and return (config, proc_id)."""
    _process_config = copy.deepcopy(proc_conf)
    _process_config["fillcolor"] = proc_conf.get("fillcolor", None)
    _process_config["histtype"] = "errorbar"
    _proc_id = proc_conf["label"] if isinstance(proc_conf["process"], list) else proc_conf["process"]
    return _process_config, _proc_id


def _entries_process_list(*, process_config: List[Dict], var: str, axis_opts: Dict,
                          cut: Any, year: str, var_over_ride: Dict,
                          label_override: Optional[List[str]] = None,
                          debug: bool = False) -> List[LoadSpec]:
    """Build LoadSpec entries for a list of processes (one hist per process)."""
    entries = []
    for iP, _proc_conf in enumerate(process_config):
        _process_config, _proc_id = _prepare_process_config(_proc_conf)
        var_to_plot = get_var_to_plot(var, var_over_ride, _proc_id, iP, debug)
        entries.append(LoadSpec(
            config=_process_config, var=var_to_plot, cut=cut, year=year,
            axis_opts=axis_opts, file_index=None, key=f"{_proc_id}{iP}",
        ))
    return entries

def _entries_process_list_multi_file(*, process_config: List[Dict], cfg: Any, var: str,
                                     axis_opts: Dict, cut: Any, year: str, var_over_ride: Dict,
                                     label_override: Optional[List[str]] = None,
                                     file_labels: Optional[List[str]] = None,
                                     debug: bool = False) -> List[LoadSpec]:
    """Build LoadSpec entries pairing each process with its corresponding input file."""
    file_labels = file_labels or []
    entries = []
    for iP, _proc_conf in enumerate(process_config):
        _process_config, _proc_id = _prepare_process_config(_proc_conf)
        if label_override and iP < len(label_override):
            _process_config["label"] = label_override[iP]
        elif iP < len(file_labels):
            _process_config["label"] = f"{_process_config['label']} {file_labels[iP]}"
        var_to_plot = get_var_to_plot(var, var_over_ride, _proc_id, iP, debug)
        entries.append(LoadSpec(
            config=_process_config, var=var_to_plot, cut=cut, year=year,
            axis_opts=axis_opts, file_index=min(iP, len(cfg.hists) - 1),
            key=f"{_proc_id}{iP}",
        ))
    return entries

def _entries_var_list(*, process_config: Dict, var_list: List[str], axis_opts: Dict,
                      cut: Any, year: str, label_override: Optional[List[str]] = None,
                      debug: bool = False) -> List[LoadSpec]:
    """Build LoadSpec entries for a list of variables (one hist per variable)."""
    return _entries_overlay(
        process_config=process_config, items=var_list,
        base_var=None, base_cut=cut, base_year=year, base_axis_opts=axis_opts,
        item_is_var=True, set_edge_color=True,
        label_override=label_override,
    )


def _entries_year_list(*, process_config: Dict, var: str, axis_opts: Dict,
                       cut: Any, year_list: List[str],
                       label_override: Optional[List[str]] = None,
                       debug: bool = False) -> List[LoadSpec]:
    """Build LoadSpec entries for a list of years (one hist per year)."""
    return _entries_overlay(
        process_config=process_config, items=year_list,
        base_var=var, base_cut=cut, base_year=None, base_axis_opts=axis_opts,
        item_is_year=True, set_edge_color=True,
        label_override=label_override,
    )

def _add_2d_ratio_plots(plot_data: Dict, **kwargs) -> None:
    """Populate plot_data["ratio_specs"] for a 2-D ratio plot.

    Convention: first histogram is denominator, second is numerator.
    Actual computation is deferred to _resolve_ratio_specs at render time.
    """
    hist_keys = list(plot_data["hists"].keys())
    if len(hist_keys) < 2:
        raise ValueError("Need at least two histograms for 2D ratio plot")

    den_key = hist_keys[0]
    num_key = hist_keys[1]

    plot_data.setdefault("ratio_specs", []).append(RatioSpec(
        name=f"ratio_{num_key}_to_{den_key}",
        denominator=HistSource(source="hists", key=den_key),
        numerator=HistSource(source="hists", key=num_key),
        norm=kwargs.get("norm", False),
        is_2d=True,
        style={},
    ))

def _add_1d_ratio_plots(plot_data: Dict, **kwargs) -> None:
    """Populate plot_data["ratio_specs"] for a 1-D ratio panel.

    Convention: the first histogram is the denominator; all others are
    numerators.  A band-at-1 entry (denominator uncertainty) is always added.
    Actual computation is deferred to _resolve_ratio_specs at render time.
    """
    hist_keys = list(plot_data["hists"].keys())
    if len(hist_keys) < 2:
        logger.debug("_add_1d_ratio_plots: only one histogram — skipping ratio")
        return

    den_key = hist_keys[0]
    den_source = HistSource(source="hists", key=den_key)

    plot_data.setdefault("ratio_specs", []).append(RatioSpec(
        name="bkg_band",
        denominator=den_source,
        numerator=None,
        style={"color": "k", "type": "band", "hatch": "\\\\"},
    ))

    for iH, num_key in enumerate(hist_keys[1:]):
        color = plot_data["hists"][num_key].get("edgecolor", plot_helpers.COLORS[iH])
        plot_data["ratio_specs"].append(RatioSpec(
            name=f"ratio_{num_key}_to_{den_key}_{iH}",
            denominator=den_source,
            numerator=HistSource(source="hists", key=num_key),
            norm=kwargs.get("norm", False),
            style={"color": color, "marker": "o"},
        ))
