"""
Helper functions for creating plot dictionaries from histogram data.

This module provides functions to:
- Extract histogram data from input files
- Create plot configurations for different types of plots (1D, 2D, ratios)
- Handle stacked and unstacked histograms
- Manage plot metadata and styling
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import copy
import src.plotting.helpers as plot_helpers
import hist
import numpy as np


def print_list_debug_info(process, cut, axis_opts):
    print(f" hist process={process}, "
          f"cut={cut}, "
          f"axis_opts={axis_opts}")


#
#  Get hist values
#
def get_hist_data(*, process: str, cfg: Any, config: Dict, var: str, cut: str, rebin: int, year: str, axis_opts : Dict, do2d: bool = False, file_index: Optional[int] = None, debug: bool = False) -> hist.Hist:
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
    # Input validation
    if not isinstance(process, str):
        raise TypeError(f"process must be a string, got {type(process)}")
    if not isinstance(var, str):
        raise TypeError(f"var must be a string, got {type(var)}")
    if not isinstance(cut, str):
        raise TypeError(f"cut must be a string, got {type(cut)}")
    if not isinstance(rebin, int):
        raise TypeError(f"rebin must be an integer, got {type(rebin)}")
    if rebin < 1:
        raise ValueError(f"rebin must be positive, got {rebin}")

    if year in ["RunII", "Run2", "Run3", "RunIII"]:
        year = sum

    if debug:
        print(f" in get_hist_data: hist process={process}, "
              f"axis_opts={axis_opts}, year={year}, var={var}")

    hist_opts = {
        "process": process,
        "year": year,
    }

    for c_key, c_val in config.items():
        if c_key in ["process", "scalefactor", "label", "fillcolor", "edgecolor", "histtype", "alpha", "linewidth", "linestyle", "zorder"]:
            continue
        if debug: print(f"Adding to hist_opts: {c_key} = {c_val}")
        hist_opts[c_key] = c_val

    hist_opts = hist_opts | axis_opts


    try:
        cut_dict = plot_helpers.get_cut_dict(cut, cfg.cutList)
    except (AttributeError, KeyError) as e:
        raise AttributeError(f"Failed to get cut dictionary: {str(e)}")

    hist_opts = hist_opts | cut_dict

    hist_obj = None
    if len(cfg.hists) > 1 and not cfg.combine_input_files:
        if file_index is None:
            raise ValueError("Must provide file_index when using multiple input files without combine_input_files")

        try:
            common, unique_to_dict = plot_helpers.compare_dict_keys_with_list(hist_opts, cfg.hists[file_index]['categories'])
        except (KeyError, IndexError) as e:
            raise ValueError(f"Failed to compare dictionary keys: {str(e)}")

        if len(unique_to_dict) > 0:
            for _key in unique_to_dict:
                hist_opts.pop(_key)

        try:
            hist_obj = cfg.hists[file_index]['hists'][var]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Failed to get histogram for var {var}: {str(e)}")

        if "variation" in cfg.hists[file_index]["categories"]:
            hist_opts = hist_opts | {"variation": "nominal"}

    else:
        for _input_data in cfg.hists:
            try:
                common, unique_to_dict = plot_helpers.compare_dict_keys_with_list(hist_opts, _input_data['categories'])
            except (KeyError, AttributeError) as e:
                raise ValueError(f"Failed to compare dictionary keys: {str(e)}")

            if len(unique_to_dict) > 0:
                for _key in unique_to_dict:
                    hist_opts.pop(_key)

            if var in _input_data['hists'] and process in _input_data['hists'][var].axes["process"]:
                if "variation" in _input_data["categories"]:
                    hist_opts = hist_opts | {"variation": "nominal"}
                hist_obj = _input_data['hists'][var]

    if hist_obj is None:
        raise ValueError(f"get_hist_data Could not find histogram for var {var} with process {process} in inputs")

    # Handle backwards compatibility
    try:
        for axis in hist_obj.axes:
            if (axis.name == "tag") and isinstance(axis, hist.axis.IntCategory):
                hist_opts['tag'] = hist.loc(cfg.plotConfig["codes"]["tag"][config["tag"]])
            if (axis.name == "region") and isinstance(axis, hist.axis.IntCategory):
                 if isinstance(axis_opts.get('region',None), list):
                     hist_opts['region'] = [hist.loc(cfg.plotConfig["codes"]["region"][i]) for i in hist_opts['region']]
                 elif axis_opts.get('region',None) and  axis_opts.get('region',None) not in ["sum", sum]:
                     hist_opts['region'] = hist.loc(cfg.plotConfig["codes"]["region"][region])
    except (KeyError, AttributeError) as e:
        raise ValueError(f"Failed to handle axis compatibility: {str(e)}")

    # Add rebin options
    varName = hist_obj.axes[-1].name
    if not do2d:
        var_dict = {varName: hist.rebin(rebin)}
        hist_opts = hist_opts | var_dict

    # Do the hist selection/binning
    try:
        selected_hist = hist_obj[hist_opts]
    except Exception as e:
        raise ValueError(f"Failed to select histogram: {str(e)}")

    # Handle shape differences
    if do2d:
        if len(selected_hist.shape) == 3:  # for 2D plots
            selected_hist = selected_hist[sum, :, :]
    else:
        if len(selected_hist.shape) == 2:
            selected_hist = selected_hist[sum, :]

    # Apply scale factor
    try:
        selected_hist *= config.get("scalefactor", 1.0)
    except Exception as e:
        raise ValueError(f"Failed to apply scale factor: {str(e)}")

    return selected_hist



#
def get_hist_data_list(*, proc_list: List[str], cfg: Any, config: Dict, var: str, cut: str, rebin: int, year: str, axis_opts: Dict, do2d: bool, file_index: Optional[int], debug) -> hist.Hist:
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

        if type(_proc) is list:
            _selected_hist =  get_hist_data_list(proc_list=_proc, cfg=cfg, config=config, var=var,
                                                 cut=cut, rebin=rebin, year=year, do2d=do2d, axis_opts=axis_opts, file_index=file_index, debug=debug)
        else:
            _selected_hist = get_hist_data(process=_proc, cfg=cfg, config=config, var=var,
                                           cut=cut, rebin=rebin, year=year, axis_opts=axis_opts, do2d=do2d, file_index=file_index, debug=debug)

        if selected_hist is None:
            selected_hist = _selected_hist
        else:
            selected_hist += _selected_hist

    return selected_hist


#
#  Get hist from input file(s)
#
def add_hist_data(*, cfg, config, var, cut, rebin, year, axis_opts, do2d=False, file_index=None, debug=False):

    if debug:
        print(f"In add_hist_data {config['process']} \n")

    proc_list = config['process'] if type(config['process']) is list else [config['process']]

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
        config["values"]     = selected_hist.values().tolist()
        config["variances"]  = selected_hist.variances().tolist()
        config["centers"]    = selected_hist.axes[0].centers.tolist()
        config["edges"]      = selected_hist.axes[0].edges.tolist()
        config["x_label"]    = selected_hist.axes[0].label
        config["under_flow"] = float(selected_hist.view(flow=True)["value"][0])
        config["over_flow"]  = float(selected_hist.view(flow=True)["value"][-1])

    return



def _create_base_plot_dict(var: str, cut: str, axis_opts: Dict, process: Any, **kwargs) -> Dict:
    """Create the base plot dictionary structure."""
    plot_data = {
        "hists": {},
        "stack": {},
        "ratio": {},
        "var": var,
        "cut": cut,
        "axis_opts": axis_opts,
        "kwargs": kwargs,
        "process": process
    }
    return plot_data

def _handle_cut_list(*, plot_data: Dict, process_config: Dict, cfg: Any, var_to_plot: str,
                     axis_opts: Dict, cut_list: List[str], rebin: int, year: str, do2d: bool,
                     label_override: Optional[List[str]] = None, debug: bool = False) -> None:
    """Handle plotting multiple cuts."""
    if debug:
        print(f"in _handle_cut_list cut_list={cut_list}")

    for ic, _cut in enumerate(cut_list):
        if debug:
            print_list_debug_info(process_config["process"], _cut, axis_opts)

        _process_config = copy.deepcopy(process_config)
        _process_config["fillcolor"] = plot_helpers.COLORS[ic]
        _process_config["label"] = plot_helpers.get_label(f"{process_config['label']} {_cut}", label_override, ic)
        _process_config["histtype"] = "errorbar"

        add_hist_data(cfg=cfg, config=_process_config,
                      var=var_to_plot, axis_opts=axis_opts, cut=_cut, rebin=rebin, year=year,
                      do2d=do2d, debug=debug)

        proc_id = process_config["label"] if isinstance(process_config["process"], list) else process_config["process"]
        plot_data["hists"][f"{proc_id}{_cut}{ic}"] = _process_config

def _handle_axis_opts_list(*, plot_data: Dict, process_config: Dict, cfg: Any, var_to_plot: str,
                           cut: str, axis_list_name: str, axis_list_values: List[str],
                           axis_opts: Dict, rebin: int, year: str, do2d: bool,
                           label_override: Optional[List[str]] = None, debug: bool = False) -> None:
    """Handle plotting multiple axis opts."""
    for ia, _axis_val in enumerate(axis_list_values):
        _axis_opts = copy.deepcopy(axis_opts)
        _axis_opts[axis_list_name] = _axis_val

        if debug:
            print_list_debug_info(process_config["process"], cut, _axis_opts)

        _process_config = copy.deepcopy(process_config)
        _process_config["fillcolor"] = plot_helpers.COLORS[ia]
        _process_config["label"] = plot_helpers.get_label(f"{process_config['label']} {_axis_val}", label_override, ia)
        _process_config["histtype"] = "errorbar"

        add_hist_data(cfg=cfg, config=_process_config,
                     var=var_to_plot, axis_opts=_axis_opts, cut=cut, rebin=rebin, year=year,
                     do2d=do2d, debug=debug)

        proc_id = process_config["label"] if isinstance(process_config["process"], list) else process_config["process"]
        plot_data["hists"][f"{proc_id}{_axis_val}{ia}"] = _process_config

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

def get_plot_dict_from_list(*, cfg: Any, var: str, cut: str, axis_opts: Dict, process: Any, **kwargs) -> Dict:
    """
    Create a plot dictionary from lists of processes, cuts, axis_opts, etc.

    Args:
        cfg: Configuration object
        var: Variable to plot
        cut: Selection cut
        axis_opts: Axis options for histogram selection
        process: Process or list of processes
        **kwargs: Additional plotting options

    Returns:
        Dict: Plot configuration dictionary
    """
    debug = kwargs.get("debug", False)
    if debug:
        print(f"in get_plot_dict_from_list hist process={process}, cut={cut}")

    rebin = kwargs.get("rebin", 1)
    do2d = kwargs.get("do2d", False)
    var_over_ride = kwargs.get("var_over_ride", {})
    label_override = kwargs.get("labels", None)
    year = kwargs.get("year", "RunII")
    file_labels = kwargs.get("fileLabels", [])

    plot_data = _create_base_plot_dict(var, cut, axis_opts, process, **kwargs)

    # Parse process configuration
    if isinstance(process, list):
        var_to_plot = var
        process_config = []
        for p in process:
            try:
                _p_config = plot_helpers.get_value_nested_dict(cfg.plotConfig, p)
                process_config.append( _p_config )
            except ValueError:
                if not p.find("HH4b") == -1:
                    print(f"Trying HH4b {p}")
                    _p_config = plot_helpers.make_klambda_hist(p, cfg.plotConfig)
                    breakpoint()
                    #print(f"Trying HH4b {p}")
                    #kl_value = p.replace("HH4b_kl","")
                    #_p_config = copy.deepcopy(plot_helpers.get_value_nested_dict(cfg.plotConfig, "HH4b_kl1"))
                    #_p_config["label"] = f'HH4b (kl={kl_value})'
                    #_p_config["process"] = f'HH4b (kl={kl_value})'
                    process_config.append( _p_config )

    else:
        try:
            process_config = plot_helpers.get_value_nested_dict(cfg.plotConfig, process)
            proc_id = process_config["label"] if isinstance(process_config["process"], list) else process_config["process"]
        except ValueError:
            raise ValueError(f"\t ERROR process = {process} not in plotConfig! \n")

        var_to_plot = var_over_ride.get(process, var)


    axis_opts_list = False
    axis_list_name = None
    for k, v in axis_opts.items():
        if type(v) is list:
            axis_opts_list = True
            axis_list_name = k
            break

    opts_dict = {"plot_data":plot_data,
                 "process_config":process_config,
                 "cfg":cfg,
                 "axis_opts":axis_opts,
                 "cut":cut,
                 "rebin":rebin,
                 "year":year,
                 "do2d":do2d,
                 "label_override":label_override,
                 "debug":debug}


    # Handle different types of lists
    if isinstance(cut, list):
        if debug: print(f"cut is a list {cut}")
        opts_dict.pop("cut")
        _handle_cut_list(**opts_dict, cut_list=cut, var_to_plot=var_to_plot)

    elif len(cfg.hists) > 1 and not cfg.combine_input_files:
        if debug: print(f"hist is a list {process}")
        _handle_input_files(**opts_dict, var_to_plot=var_to_plot, file_labels=file_labels)

    elif isinstance(process, list):
        if debug: print(f"process is a list {process}")
        _handle_process_list(**opts_dict, var=var, var_over_ride=var_over_ride)

    elif isinstance(var, list):
        if debug: print(f"var is a list {var}")
        _handle_var_list(**opts_dict, var_list=var)

    elif isinstance(year, list):
        if debug: print(f"year is a list {year}")
        opts_dict.pop("year")
        _handle_year_list(**opts_dict, var=var_to_plot, year_list=year)

    elif axis_opts_list:
        if debug: print(f"One of the axis_opts is a list: {axis_list_name} {axis_opts[axis_list_name]}")
        axis_list_values = opts_dict["axis_opts"].pop(axis_list_name)
        _handle_axis_opts_list(**opts_dict, axis_list_name=axis_list_name, axis_list_values=axis_list_values, var_to_plot=var_to_plot)
    else:
        raise ValueError("Error: At least one parameter must be a list!")

    # Handle ratio plots if requested
    if kwargs.get("doRatio", kwargs.get("doratio", False)):
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
            _handle_stack_sum(proc_config, cfg, var_to_plot, cut, rebin, year, axis_opts, do2d, debug, var_over_ride)
            stack_dict[_proc_name] = proc_config
        else:
            raise ValueError("Error: Stack component must have either 'process' or 'sum' configuration")

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
    """
    Extract values, variances and centers from histogram configuration.

    Args:
        hist_config: Histogram configuration dictionary
        plot_data: Plot data dictionary

    Returns:
        Tuple containing values, variances and centers arrays

    Raises:
        ValueError: If histogram type is invalid
    """
    if hist_config["type"] == "hists":
        num_data = plot_data["hists"][hist_config["key"]]
        return np.array(num_data["values"]), np.array(num_data["variances"]), num_data["centers"]

    if hist_config["type"] == "stack":
        return_values = [v["values"] for _, v in plot_data["stack"].items()]
        return_values = np.sum(return_values, axis=0)

        return_variances = [v["variances"] for _, v in plot_data["stack"].items()]
        return_variances = np.sum(return_variances, axis=0)

        centers = next(iter(plot_data["stack"].values()))["centers"]
        return return_values, return_variances, centers

    raise ValueError("ERROR: ratio needs to be of type 'hists' or 'stack'")

def add_ratio_plots(ratio_config: Dict, plot_data: Dict, **kwargs) -> None:
    """
    Add ratio plots to the plot configuration.

    Args:
        ratio_config: Ratio plot configuration
        plot_data: Plot data dictionary
        **kwargs: Additional plotting options
    """
    for r_name, _r_config in ratio_config.items():
        r_config = copy.deepcopy(_r_config)

        num_values, num_vars, num_centers = get_values_variances_centers_from_dict(r_config.get("numerator"), plot_data)
        den_values, den_vars, _ = get_values_variances_centers_from_dict(r_config.get("denominator"), plot_data)

        if kwargs.get("norm", False):
            r_config["norm"] = True

        # Add ratio plot
        ratios, ratio_uncert = plot_helpers.make_ratio(num_values, num_vars, den_values, den_vars, **r_config)
        r_config["ratio"] = ratios.tolist()
        r_config["error"] = ratio_uncert.tolist()
        r_config["centers"] = num_centers
        plot_data["ratio"][f"ratio_{r_name}"] = r_config

        # Add background error band
        default_band_config = {"color": "k", "type": "band", "hatch": "\\\\\\"}
        _band_config = r_config.get("bkg_err_band", default_band_config)

        if _band_config:
            band_config = copy.deepcopy(_band_config)
            band_config["ratio"] = np.ones(len(num_centers)).tolist()
            den_values[den_values == 0] = plot_helpers.EPSILON
            band_config["error"] = np.sqrt(den_vars * np.power(den_values, -2.0)).tolist()
            band_config["centers"] = list(num_centers)
            plot_data["ratio"][f"band_{r_name}"] = band_config

def get_plot_dict_from_config(*, cfg: Any, var: str = 'selJets.pt',
                              cut: str = "passPreSel", axis_opts: Dict, **kwargs) -> Dict:
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

    if cut and cut not in cfg.cutList:
        raise AttributeError(f"{cut} not in cutList {cfg.cutList}")

    # Initialize plot data structure
    plot_data = {
        "hists": {},
        "stack": {},
        "ratio": {},
        "var": var,
        "cut": cut,
        "axis_opts": axis_opts,
        "kwargs": kwargs
    }
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
    if kwargs.get("doRatio", kwargs.get("doratio", False)) and not do2d:
        ratio_config = cfg.plotConfig["ratios"]
        add_ratio_plots(ratio_config, plot_data, **kwargs)

    return plot_data

def _handle_input_files(plot_data: Dict, process_config: Dict, cfg: Any, var_to_plot: str,
                        axis_opts: Dict, cut: str, rebin: int, year: str, do2d: bool,
                        label_override: Optional[List[str]] = None, debug: bool = False,
                        file_labels: Optional[List[str]] = None) -> None:
    """Handle plotting from multiple input files."""
    if debug:
        print_list_debug_info(process_config["process"], cut, axis_opts)

    file_labels = file_labels or []
    proc_id = process_config["label"] if isinstance(process_config["process"], list) else process_config["process"]

    for iF, _input_file in enumerate(cfg.hists):
        _process_config = copy.deepcopy(process_config)
        _process_config["fillcolor"] = plot_helpers.COLORS[iF]

        if label_override:
            _process_config["label"] = label_override[iF]
        elif iF < len(file_labels):
            _process_config["label"] = f"{_process_config['label']} {file_labels[iF]}"
        else:
            _process_config["label"] = f"{_process_config['label']} file{iF + 1}"

        _process_config["histtype"] = "errorbar"

        add_hist_data(cfg=cfg, config=_process_config,
                     var=var_to_plot, axis_opts=axis_opts, cut=cut, rebin=rebin, year=year,
                     do2d=do2d, file_index=iF, debug=debug)

        plot_data["hists"][f"{proc_id}file{iF}"] = _process_config

def _handle_process_list(*, plot_data: Dict, process_config: List[Dict], cfg: Any, var: str,
                         axis_opts: Dict, cut: str, rebin: int, year: str, do2d: bool,
                         var_over_ride: Dict, label_override: Optional[List[str]] = None, debug: bool = False) -> None:
    """Handle plotting multiple processes."""
    for iP, _proc_conf in enumerate(process_config):
        if debug:
            print_list_debug_info(_proc_conf["process"],  cut, axis_opts)

        _process_config = copy.deepcopy(_proc_conf)
        _process_config["fillcolor"] = _proc_conf.get("fillcolor", None)
        _process_config["histtype"] = "errorbar"

        _proc_id = _proc_conf["label"] if isinstance(_proc_conf["process"], list) else _proc_conf["process"]
        var_to_plot = var_over_ride.get(_proc_id, var)

        add_hist_data(cfg=cfg, config=_process_config,
                     var=var_to_plot, axis_opts=axis_opts, cut=cut, rebin=rebin, year=year,
                     do2d=do2d, debug=debug)

        plot_data["hists"][f"{_proc_id}{iP}"] = _process_config

def _handle_var_list(*, plot_data: Dict, process_config: Dict, cfg: Any, var_list: List[str],
                     axis_opts: Dict, cut: str, rebin: int, year: str, do2d: bool,
                     label_override: Optional[List[str]] = None, debug: bool = False) -> None:
    """Handle plotting multiple variables."""
    proc_id = process_config["label"] if isinstance(process_config["process"], list) else process_config["process"]

    for iv, _var in enumerate(var_list):
        if debug:
            print_list_debug_info(process_config["process"],  cut, axis_opts)

        _process_config = copy.deepcopy(process_config)
        _process_config["fillcolor"] = plot_helpers.COLORS[iv]
        _process_config["label"] = plot_helpers.get_label(f"{process_config['label']} {_var}", label_override, iv)
        _process_config["histtype"] = "errorbar"

        add_hist_data(cfg=cfg, config=_process_config,
                     var=_var, axis_opts=axis_opts, cut=cut, rebin=rebin, year=year,
                     do2d=do2d, debug=debug)

        plot_data["hists"][f"{proc_id}{_var}{iv}"] = _process_config

def _handle_year_list(*, plot_data: Dict, process_config: Dict, cfg: Any, var: str,
                     axis_opts: Dict, cut: str, rebin: int, year_list: List[str], do2d: bool,
                     label_override: Optional[List[str]] = None, debug: bool = False) -> None:
    """Handle plotting multiple years."""
    proc_id = process_config["label"] if isinstance(process_config["process"], list) else process_config["process"]

    for iy, _year in enumerate(year_list):
        if debug:
            print_list_debug_info(process_config["process"],  cut, axis_opts)

        _process_config = copy.copy(process_config)
        _process_config["fillcolor"] = plot_helpers.COLORS[iy]
        _process_config["label"] = plot_helpers.get_label(f"{process_config['label']} {_year}", label_override, iy)
        _process_config["histtype"] = "errorbar"

        add_hist_data(cfg=cfg, config=_process_config,
                      var=var, axis_opts=axis_opts, cut=cut, rebin=rebin, year=_year,
                      do2d=do2d, debug=debug)

        plot_data["hists"][f"{proc_id}{_year}{iy}"] = _process_config

def _add_2d_ratio_plots(plot_data: Dict, **kwargs) -> None:
    """
    Add 2D ratio plots to the plot configuration.

    Args:
        plot_data: Plot data dictionary containing histogram data
        **kwargs: Additional plotting options

    Raises:
        ValueError: If insufficient histograms for ratio calculation
        KeyError: If required histogram data is missing
    """
    hist_keys = list(plot_data["hists"].keys())
    if len(hist_keys) < 2:
        raise ValueError("Need at least two histograms for 2D ratio plot")

    try:
        den_key = hist_keys.pop(0)
        den_values = np.array(plot_data["hists"][den_key]["values"])
        den_vars = plot_data["hists"][den_key]["variances"]
        den_values[den_values == 0] = plot_helpers.EPSILON

        num_key = hist_keys.pop(0)
        num_values = np.array(plot_data["hists"][num_key]["values"])
        num_vars = plot_data["hists"][num_key]["variances"]

        ratio_config = {}
        ratios, ratio_uncert = plot_helpers.make_ratio(num_values, num_vars, den_values, den_vars, **kwargs)
        ratio_config["ratio"] = ratios.tolist()
        ratio_config["error"] = ratio_uncert.tolist()
        plot_data["ratio"][f"ratio_{num_key}_to_{den_key}"] = ratio_config
    except (KeyError, IndexError) as e:
        raise ValueError(f"Failed to create 2D ratio plot: {str(e)}")

def _add_1d_ratio_plots(plot_data: Dict, **kwargs) -> None:
    """
    Add 1D ratio plots to the plot configuration.

    Args:
        plot_data: Plot data dictionary containing histogram data
        **kwargs: Additional plotting options

    Raises:
        ValueError: If insufficient histograms for ratio calculation
        KeyError: If required histogram data is missing
    """
    hist_keys = list(plot_data["hists"].keys())
    if len(hist_keys) < 1:
        raise ValueError("Need at least one histogram for 1D ratio plot")

    try:
        den_key = hist_keys.pop(0)
        den_values = np.array(plot_data["hists"][den_key]["values"])
        den_vars = plot_data["hists"][den_key]["variances"]
        den_centers = plot_data["hists"][den_key]["centers"]

        den_values[den_values == 0] = plot_helpers.EPSILON

        # Add background error band
        band_ratios = np.ones(len(den_centers))
        band_uncert = np.sqrt(den_vars * np.power(den_values, -2.0))
        band_config = {
            "color": "k",
            "type": "band",
            "hatch": "\\\\",
            "ratio": band_ratios.tolist(),
            "error": band_uncert.tolist(),
            "centers": list(den_centers)
        }
        plot_data["ratio"]["bkg_band"] = band_config

        # Add ratio plots for each histogram
        for iH, _num_key in enumerate(hist_keys):
            num_values = np.array(plot_data["hists"][_num_key]["values"])
            num_vars = plot_data["hists"][_num_key]["variances"]

            ratio_config = {
                "color": plot_data["hists"][_num_key].get("edgecolor",plot_helpers.COLORS[iH]),
                "marker": "o"
            }
            ratios, ratio_uncert = plot_helpers.make_ratio(num_values, num_vars, den_values, den_vars, **kwargs)
            ratio_config["ratio"] = ratios.tolist()
            ratio_config["error"] = ratio_uncert.tolist()
            ratio_config["centers"] = den_centers

            plot_data["ratio"][f"ratio_{_num_key}_to_{den_key}_{iH}"] = ratio_config
    except (KeyError, IndexError) as e:
        raise ValueError(f"Failed to create 1D ratio plot: {str(e)}")
