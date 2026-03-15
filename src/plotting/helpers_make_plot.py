import matplotlib.pyplot as plt
import mplhep as hep  # HEP (CMS) extensions/styling on top of mpl
import matplotlib.patches as mpatches
import src.plotting.helpers as plot_helpers
import hist
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Plot styling configuration
plt.style.use([hep.style.CMS, {'font.size': 22}])

# Constants for plotting
DEFAULT_FIGURE_SIZE = 7
DEFAULT_LINEWIDTH = 2
DEFAULT_MARKERSIZE = 12
DEFAULT_COLOR = 'red'
DEFAULT_LINESTYLE = '-'

# Constants for border plotting
BORDER_COLOR = 'orangered'
BORDER_LINESTYLE = 'dashed'
BORDER_LINEWIDTH = 5

# Constants for ratio plot configuration
RATIO_GRID_CONFIG = {
    'hspace': 0.06,
    'height_ratios': [3, 1],
    'left': 0.1,
    'right': 0.95,
    'top': 0.95,
    'bottom': 0.1
}

# Constants for standard plot configuration
STANDARD_AXES_CONFIG = {
    'left': 0.1,
    'bottom': 0.15,
    'width': 0.85,
    'height': 0.8
}

def _plot_kinematic_cut_lines(funcs: list) -> None:
    """Plot a list of kinematic cut line functions over x in [100, 1100]."""
    x = np.linspace(100, 1100, 50)
    for func in funcs:
        plt.plot(x, [func(xi) for xi in x],
                 color=DEFAULT_COLOR, linestyle=DEFAULT_LINESTYLE, linewidth=DEFAULT_LINEWIDTH)

def plot_leadst_lines() -> None:
    """Plot leading-jet pt cut lines: (360/x)-0.5 and max(1.5, (650/x)+0.5)."""
    _plot_kinematic_cut_lines([
        lambda x: (360/x) - 0.5,
        lambda x: max(1.5, (650/x) + 0.5),
    ])

def plot_sublst_lines() -> None:
    """Plot subleading-jet pt cut lines: 235/x and max(1.5, (650/x)+0.7)."""
    _plot_kinematic_cut_lines([
        lambda x: 235/x,
        lambda x: max(1.5, (650/x) + 0.7),
    ])


def _higgs_mass_ellipse(X, Y, cx: float, cy: float):
    """Compute the Higgs-mass ellipse distance: ((X-cx)/(0.1*X))^2 + ((Y-cy)/(0.1*Y))^2."""
    return ((X - cx) / (0.1 * X))**2 + ((Y - cy) / (0.1 * Y))**2

def plot_border_SR() -> None:
    """Plot the HH→4b signal region borders as four Higgs-mass ellipse contours."""
    x = np.linspace(0, 250, 500)
    X, Y = np.meshgrid(x, x)
    # (cx, cy, level) for each SR ellipse
    ellipses = [
        (127.5, 122.5, 1.90**2),
        (127.5,  89.18, 1.90**2),
        ( 92.82, 122.5, 1.90**2),
        ( 92.82,  89.18, 2.60**2),
    ]
    for cx, cy, level in ellipses:
        plt.contour(X, Y, _higgs_mass_ellipse(X, Y, cx, cy),
                    levels=[level], colors=BORDER_COLOR,
                    linestyles=BORDER_LINESTYLE, linewidths=BORDER_LINEWIDTH)


def _draw_stack(stack_dict: Dict, uniform_bins: bool, norm: bool, add_flow: bool,
                plot_data: Dict, ax) -> None:
    """Draw the stacked histogram. Stores uniform-bin metadata in plot_data when needed."""
    if not stack_dict:
        return
    if uniform_bins:
        stack_keys = list(stack_dict.keys())
        stack_values_list = []
        stack_edges = None
        for k in stack_keys:
            v = stack_dict[k]
            vals = np.array(v["values"], dtype=float)
            if add_flow:
                vals = vals.copy()
                vals[0] += v["under_flow"]
                vals[-1] += v["over_flow"]
            if norm:
                total = vals.sum()
                if total > 0:
                    vals = vals / total
            stack_values_list.append(vals)
            if stack_edges is None:
                stack_edges = np.array(v["edges"])
        n = len(stack_values_list[0])
        uniform_edges = np.arange(n + 1) - 0.5
        bottoms = np.zeros(n)
        fill_colors = [stack_dict[k].get("fillcolor") for k in stack_keys]
        edge_colors = [stack_dict[k].get("edgecolor") for k in stack_keys]
        for i, vals in enumerate(stack_values_list):
            ax.stairs(vals + bottoms, uniform_edges,
                      color=fill_colors[i], fill=True, baseline=bottoms, linewidth=0)
            ax.stairs(vals + bottoms, uniform_edges,
                      color=edge_colors[i], linewidth=1.0, baseline=bottoms)
            bottoms += vals
        plot_data["_uniform_bin_edges"] = stack_edges
        plot_data["_uniform_n_bins"] = n
    else:
        stack_dict_for_hist = {}
        for k, v in stack_dict.items():
            stack_dict_for_hist[k] = plot_helpers.make_hist(
                edges=v["edges"], values=v["values"], variances=v["variances"],
                x_label=v["x_label"], under_flow=v["under_flow"], over_flow=v["over_flow"],
                add_flow=add_flow
            )
        fill_colors = [v.get("fillcolor") for _, v in stack_dict.items()]
        edge_colors = [v.get("edgecolor") for _, v in stack_dict.items()]
        if stack_dict_for_hist:
            s = hist.Stack.from_dict(stack_dict_for_hist)
            s.plot(stack=True, histtype="fill", color=fill_colors, label=None, density=norm)
            s.plot(stack=True, histtype="step", color=edge_colors, label=None, density=norm)


def _build_stack_legend_patches(stack_dict: Dict) -> List:
    """Build legend patch handles for the stack components."""
    patches = []
    for _, data in stack_dict.items():
        label = data.get("label")
        if label in ["None"]:
            continue
        patches.append(mpatches.Patch(
            facecolor=data.get("fillcolor"),
            edgecolor=data.get("edgecolor"),
            label=label,
        ))
    return patches


def _draw_hists(hists_dict: Dict, uniform_bins: bool, norm: bool, add_flow: bool,
                plot_data: Dict, **kwargs) -> None:
    """Draw individual histograms onto the current axes."""
    ax = plt.gca()
    for _, hist_data in hists_dict.items():
        vals = np.array(hist_data["values"], dtype=float)
        varis = np.array(hist_data["variances"], dtype=float)
        edges = np.array(hist_data["edges"])
        label = hist_data.get("label", "")
        color = hist_data.get("fillcolor", "k")
        histtype = kwargs.get("histtype", hist_data.get("histtype", "errorbar"))
        lw = kwargs.get("linewidth", hist_data.get("linewidth", DEFAULT_LINEWIDTH))

        if add_flow:
            vals = vals.copy()
            vals[0] += hist_data["under_flow"]
            vals[-1] += hist_data["over_flow"]

        if uniform_bins:
            if norm:
                total = vals.sum()
                if total > 0:
                    varis = varis / (total ** 2)
                    vals = vals / total
            n = len(vals)
            x_positions = np.arange(n)
            if histtype == "errorbar":
                ax.errorbar(x_positions, vals, yerr=np.sqrt(varis),
                            fmt="o", color=color, label=label,
                            markersize=DEFAULT_MARKERSIZE, linewidth=lw)
            elif histtype == "step":
                ax.stairs(vals, np.arange(n + 1) - 0.5,
                          color=color, label=label, linewidth=lw)
            else:
                ax.bar(x_positions, vals, width=0.9, color=color, label=label, linewidth=lw)
            plot_data["_uniform_bin_edges"] = edges
            plot_data["_uniform_n_bins"] = n
        else:
            hist_obj = plot_helpers.make_hist(
                edges=hist_data["edges"], values=hist_data["values"],
                variances=hist_data["variances"], x_label=hist_data["x_label"],
                under_flow=hist_data["under_flow"], over_flow=hist_data["over_flow"],
                add_flow=add_flow,
            )
            plot_opts = {
                "density": norm,
                "label": label,
                "color": color,
                "histtype": histtype,
                "linewidth": lw,
                "yerr": histtype == "errorbar",
            }
            if histtype == "errorbar":
                plot_opts["markersize"] = DEFAULT_MARKERSIZE
            hist_obj.plot(**plot_opts)


def _configure_main_axes(stack_patches: List, norm: bool, **kwargs) -> None:
    """Set axis labels, scales, legend, limits, and text annotations on the current axes."""
    # Labels
    if kwargs.get("xlabel"):
        plt.xlabel(kwargs["xlabel"])
    plt.xlabel(plt.gca().get_xlabel(), loc="right", fontsize=kwargs.get("xlabel_fontsize", 30))

    if kwargs.get("ylabel"):
        plt.ylabel(kwargs["ylabel"])
    if norm:
        plt.ylabel(plt.gca().get_ylabel() + " (normalized)")
    plt.ylabel(plt.gca().get_ylabel(), loc="top",
               fontsize=kwargs.get("ylabel_fontsize", 30),
               labelpad=kwargs.get("ylabel_labelpad", -4))

    # Scales
    if kwargs.get("yscale"):
        plt.yscale(kwargs["yscale"])
    if kwargs.get("xscale"):
        plt.xscale(kwargs["xscale"])

    # Legend
    if kwargs.get("legend", True):
        handles = list(stack_patches)
        labels = [p.get_label() for p in stack_patches]
        data_handles, data_labels = plt.gca().get_legend_handles_labels()
        handles.extend(data_handles)
        labels.extend(data_labels)

        legend_reverse = True
        if kwargs.get("legend_order"):
            legend_reverse = False
            ordered_handles, ordered_labels = [], []
            for item in kwargs["legend_order"]:
                print(item)
                ordered_handles.append(handles[labels.index(item)])
                ordered_labels.append(item)
            handles, labels = ordered_handles, ordered_labels

        plt.legend(handles=handles, labels=labels,
                   loc=kwargs.get("legend_loc", "best"),
                   frameon=False, reverse=legend_reverse,
                   fontsize=kwargs.get("legend_fontsize", 22))

    # Limits
    if kwargs.get("ylim"):
        plt.ylim(*kwargs["ylim"])
    if kwargs.get("xlim"):
        plt.xlim(*kwargs["xlim"])

    # Text annotations
    for _, t in kwargs.get("text", {}).items():
        plt.text(t["xpos"], t["ypos"], t["text"],
                 horizontalalignment=t.get("horizontalalignment", "center"),
                 verticalalignment="top",
                 transform=plt.gca().transAxes,
                 fontsize=t.get("fontsize", 22),
                 weight=t.get("weight", "normal"))


def _draw_plot_from_dict(plot_data: Dict[str, Any], **kwargs) -> None:
    """
    Draw stack, individual histograms, and configure axes from a plot data dictionary.

    Args:
        plot_data: Dictionary with 'stack', 'hists', and display metadata.
        **kwargs: norm, uniform_bins, add_flow, xlabel, ylabel, yscale, xscale,
                  legend, ylim, xlim, text, debug, and forwarded hist/stack options.
    """
    if kwargs.get("debug"):
        logger.info(f'\t in _draw_plot ... kwargs = {kwargs}')

    norm = kwargs.get("norm", False)
    uniform_bins = kwargs.get("uniform_bins", False)
    add_flow = kwargs.get("add_flow", False)
    ax = plt.gca()

    stack_dict = plot_data.get("stack", {})
    _draw_stack(stack_dict, uniform_bins, norm, add_flow, plot_data, ax)
    stack_patches = _build_stack_legend_patches(stack_dict)
    _draw_hists(plot_data.get("hists", {}), uniform_bins, norm, add_flow, plot_data, **kwargs)
    _configure_main_axes(stack_patches, norm, **kwargs)


def _setup_figure(do_ratio: bool, **kwargs) -> Tuple[plt.Figure, plt.Axes, Any]:
    """Create figure and main axes. Returns (fig, main_ax, gridspec_or_None)."""
    fig = plt.figure()
    if do_ratio:
        grid = fig.add_gridspec(2, 1, **kwargs.get("ratio_grid_config", RATIO_GRID_CONFIG))
        main_ax = fig.add_subplot(grid[0])
        return fig, main_ax, grid
    else:
        fig.add_axes((0.1, 0.15, 0.85, 0.8))
        return fig, fig.gca(), None


def _draw_ratio_panel(ratio_ax, plot_data: Dict, top_xlabel: str,
                      uniform_bins: bool, **kwargs) -> None:
    """Draw the ratio subplot content and set its labels/limits/legend."""
    ratio_ax.axhline(kwargs.get("ratio_line_value", 1.0),
                     color="black", linestyle="dashed", linewidth=2.0)
    legend_handles = {}

    for ratio_name, ratio_data in plot_data["ratio"].items():
        error_bar_type = ratio_data.get("type", "bar")
        label = ratio_data.get("label", ratio_name)

        if error_bar_type == "band":
            if uniform_bins:
                for i, (yi, err) in enumerate(zip(ratio_data["ratio"], ratio_data["error"])):
                    plt.fill_between([i - 0.5, i + 0.5], yi - err, yi + err,
                                     hatch=ratio_data.get("hatch", "/"),
                                     edgecolor=ratio_data.get("color", "black"),
                                     facecolor=ratio_data.get("facecolor", "none"),
                                     linewidth=0.0, zorder=1)
            else:
                bin_width = ratio_data["centers"][1] - ratio_data["centers"][0]
                for xi, yi, err in zip(ratio_data["centers"], ratio_data["ratio"], ratio_data["error"]):
                    plt.fill_between([xi - bin_width/2, xi + bin_width/2], yi - err, yi + err,
                                     hatch=ratio_data.get("hatch", "/"),
                                     edgecolor=ratio_data.get("color", "black"),
                                     facecolor=ratio_data.get("facecolor", "none"),
                                     linewidth=0.0, zorder=1)
            from matplotlib.patches import Rectangle
            legend_handles[label] = Rectangle(
                (0, 0), 1, 1,
                hatch=ratio_data.get("hatch", "/"),
                edgecolor=ratio_data.get("color", "black"),
                facecolor=ratio_data.get("facecolor", "none"),
                linewidth=0.0)

        elif error_bar_type in ["step", "fill"]:
            lw = kwargs.get("linewidth", ratio_data.get("linewidth", DEFAULT_LINEWIDTH))
            if uniform_bins:
                vals = np.array(ratio_data["ratio"], dtype=float)
                n = len(vals)
                uniform_edges = np.arange(n + 1) - 0.5
                color = ratio_data.get("fillcolor", "k")
                if error_bar_type == "fill":
                    stairs_artist = ratio_ax.stairs(vals, uniform_edges,
                                                    color=color, fill=True, linewidth=0)
                    legend_handles[label] = stairs_artist
                    ratio_ax.stairs(vals, uniform_edges,
                                    color=ratio_data.get("edgecolor", "k"), linewidth=lw)
                else:
                    stairs_artist = ratio_ax.stairs(vals, uniform_edges,
                                                    color=color, linewidth=lw)
                    legend_handles[label] = stairs_artist
            else:
                hist_obj = plot_helpers.make_hist(
                    edges=ratio_data["edges"], values=ratio_data["ratio"],
                    variances=ratio_data["variances"], x_label=ratio_data.get("x_label", ""),
                    under_flow=ratio_data.get("under_flow", 0),
                    over_flow=ratio_data.get("over_flow", 0),
                    add_flow=kwargs.get("add_flow", False)
                )
                plot_opts = {"density": False, "label": ratio_data.get("label", ""),
                             "color": ratio_data.get("fillcolor", "k"),
                             "histtype": ratio_data.get("type", "errorbar"),
                             "linewidth": lw, "yerr": False}
                plot_result = hist_obj.plot(**plot_opts)
                if plot_result and len(plot_result) > 0:
                    artist = plot_result[0]
                    if hasattr(artist, "stairs") and artist.stairs is not None:
                        legend_handles[label] = artist.stairs
                if ratio_data.get("type") == "fill":
                    hist_obj.plot(**{**plot_opts,
                                    "color": ratio_data.get("edgecolor", "k"),
                                    "histtype": "step"})

        else:
            positions = np.arange(len(ratio_data["ratio"])) if uniform_bins else ratio_data["centers"]
            handle = ratio_ax.errorbar(
                positions, ratio_data["ratio"], yerr=ratio_data["error"],
                color=ratio_data.get("color", "black"),
                marker=ratio_data.get("marker", "o"),
                linestyle=ratio_data.get("linestyle", "none"),
                markersize=ratio_data.get("markersize", 4),
            )
            legend_handles[label] = handle

    plt.ylabel(kwargs.get("rlabel", "Ratio"))
    plt.ylabel(plt.gca().get_ylabel(), loc="center", fontsize=kwargs.get("rlabel_fontsize", 30))
    plt.xlabel(kwargs.get("xlabel", top_xlabel), loc="right", fontsize=kwargs.get("xlabel_fontsize", 30))
    plt.ylim(*kwargs.get("rlim", [0, 2]))

    if kwargs.get("ratio_legend_order"):
        ordered_handles, ordered_labels = [], []
        for rl in kwargs["ratio_legend_order"]:
            if rl in legend_handles:
                ordered_labels.append(rl)
                ordered_handles.append(legend_handles[rl])
        print(ordered_handles, ordered_labels)
        ratio_ax.legend(ordered_handles, ordered_labels, ncol=2,
                        loc=kwargs.get("ratio_legend_loc", "upper left"))


def _apply_uniform_bin_ticks(bottom_ax, plot_data: Dict, **kwargs) -> None:
    """Set integer tick labels on bottom_ax when uniform_bins mode is active."""
    n = plot_data["_uniform_n_bins"]
    tick_step = kwargs.get("uniform_bins_tick_step", max(1, n // 10))
    tick_positions = list(range(0, n, tick_step))
    bottom_ax.set_xticks(tick_positions)
    bottom_ax.set_xticklabels([str(i) for i in tick_positions],
                               fontsize=kwargs.get("uniform_bins_fontsize", 14))
    bottom_ax.set_xlim(-0.5, n - 0.5)
    if not kwargs.get("xlabel"):
        bottom_ax.set_xlabel("Bin index", loc="right",
                              fontsize=kwargs.get("xlabel_fontsize", 30))
    del plot_data["_uniform_bin_edges"]
    del plot_data["_uniform_n_bins"]


def _plot_from_dict(plot_data: Dict[str, Any], **kwargs) -> Tuple[plt.Figure, plt.Axes, Optional[plt.Axes]]:
    """
    Create a 1D plot (with optional ratio panel) from a plot data dictionary.

    Returns (fig, main_ax, ratio_ax). ratio_ax is None when no ratio is drawn.
    """
    if kwargs.get("debug"):
        logger.info(f'\t in plot ... kwargs = {kwargs}')

    do_ratio = len(plot_data.get("ratio", {}))
    fig, main_ax, grid = _setup_figure(do_ratio, **kwargs)

    year_str = plot_helpers.get_year_str(year=kwargs.get("year_str", kwargs.get("year", "RunII")))
    hep.cms.label(kwargs.get("CMSText", "Internal"), data=True,
                  year=year_str, loc=0, ax=main_ax)

    if kwargs.get("do_title", True) and "region" in plot_data["axis_opts"]:
        main_ax.set_title(plot_helpers.get_axis_str(plot_data["axis_opts"]["region"]))

    _draw_plot_from_dict(plot_data, **kwargs)

    uniform_bins = kwargs.get("uniform_bins", False)
    ratio_ax = None

    if do_ratio:
        top_xlabel = plt.gca().get_xlabel()
        plt.xlabel("")
        ratio_ax = fig.add_subplot(grid[1], sharex=main_ax)
        plt.setp(main_ax.get_xticklabels(), visible=False)
        _draw_ratio_panel(ratio_ax, plot_data, top_xlabel, uniform_bins, **kwargs)

    if uniform_bins and "_uniform_bin_edges" in plot_data:
        bottom_ax = ratio_ax if do_ratio else main_ax
        _apply_uniform_bin_ticks(bottom_ax, plot_data, **kwargs)

    return fig, main_ax, ratio_ax


def make_plot_from_dict(plot_data: Dict[str, Any], *, do2d: bool = False) -> Tuple[plt.Figure, Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]]:
    """
    Create a plot from a dictionary of plot data.

    Args:
        plot_data: Dictionary containing plot data and configuration
        do2d: Whether to create a 2D plot

    Returns:
        Tuple containing:
            - Figure object
            - Axes object or tuple of axes objects (main and ratio)

    Raises:
        ValueError: If the plotting operation fails
        KeyError: If required data is missing from plot_data
    """
    try:
        kwargs = plot_data.get("kwargs", {})
        if do2d:
            fig, ax = _plot2d_from_dict(plot_data, **kwargs)
        else:
            fig, main_ax, ratio_ax = _plot_from_dict(plot_data, **kwargs)
            ax = (main_ax, ratio_ax)

        if kwargs.get("outputFolder", None):
            try:
                # Determine tag name
                if isinstance(plot_data.get("process", ""), list):
                    tagName = "_vs_".join(plot_data["process"])
                else:
                    try:
                        tagName = plot_helpers.get_value_nested_dict(plot_data, "tag")
                        if isinstance(tagName, hist.loc):
                            tagName = str(tagName.value)
                    except ValueError:
                        pass

                # Construct output path
                try:

                    output_path = [
                        kwargs.get("outputFolder"),
                        kwargs.get("year", "RunII"),
                        plot_data["cut"],
                    ]

                    for k in sorted(plot_data["axis_opts"].keys()):
                        if k in ["name"]:
                            continue
                        v = plot_data["axis_opts"][k]
                        output_path.append(f"{k}_{plot_helpers.get_axis_str(v).replace(' ','_')}")

                    output_path.append(plot_data.get("process", ""))

                except (NameError, KeyError):
                    print("Setting output path to outputFolder only")
                    output_path = [kwargs.get("outputFolder")]

                # Determine file name
                file_name = plot_data.get("file_name", plot_data["var"])
                if kwargs.get("yscale", None) == "log":
                    file_name += "_logy"

                # Save plot
                plot_helpers.savefig(fig, file_name, *output_path)

                # Save YAML if requested
                if kwargs.get("write_yaml", False):
                    plot_helpers.save_yaml(plot_data, file_name, *output_path)

            except Exception as e:
                logger.error(f"Error saving plot: {str(e)}")
                raise ValueError(f"Failed to save plot: {str(e)}")

        return fig, ax

    except Exception as e:
        logger.error(f"Error in make_plot_from_dict: {str(e)}")
        raise ValueError(f"Failed to create plot: {str(e)}")


def _plot2d_from_dict(plot_data: Dict[str, Any], **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a 2D plot from a dictionary of plot data.

    Args:
        plot_data: Dictionary containing plot data and configuration
        **kwargs: Additional plotting options:
            - debug: bool - Whether to print debug information
            - year: str - Year for CMS label
            - rlim: List[float] - Limits for ratio plot
            - full: bool - Whether to create full plot
            - plot_contour: bool - Whether to plot contour
            - plot_leadst_lines: bool - Whether to plot leading lines
            - plot_sublst_lines: bool - Whether to plot subleading lines

    Returns:
        Tuple containing:
            - Figure object
            - Axes object

    Raises:
        ValueError: If the plotting operation fails
        KeyError: If required data is missing from plot_data
    """
    try:
        if kwargs.get("debug", False):
            logger.info(f'\t in _plot2d_from_dict ... kwargs = {kwargs}')

        if len(plot_data.get("ratio", {})):
            if kwargs.get("debug", False):
                logger.info(f'\t doing ratio')

            # Plot ratios
            key_iter = iter(plot_data["hists"])
            num_key = next(key_iter)
            num_hist_data = plot_data["hists"][num_key]

            den_key = next(key_iter)
            den_hist_data = plot_data["hists"][den_key]

            ratio_key = next(iter(plot_data["ratio"]))

            # Mask 0s
            hd = np.array(plot_data["ratio"][ratio_key]["ratio"])
            hd[hd < 0.001] = np.nan

            hist_obj_2d = plot_helpers.make_2d_hist(
                x_edges=num_hist_data["x_edges"],
                y_edges=num_hist_data["y_edges"],
                values=hd,
                variances=num_hist_data["variances"],
                x_label=num_hist_data["x_label"],
                y_label=num_hist_data["y_label"]
            )

            scale = 2
            fig = plt.figure(figsize=(10*scale, 6*scale))
            gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.4)
            ax_big = fig.add_subplot(gs[:, 0])

            hist_obj_2d.plot2d(
                cmap="turbo",
                cmin=kwargs.get("rlim", [None, None])[0],
                cmax=kwargs.get("rlim", [None, None])[1]
            )

            ax_top_right = fig.add_subplot(gs[0, 1])

            num_hd = np.array(num_hist_data["values"])
            num_hd[num_hd < 0.001] = np.nan

            num_hist_obj_2d = plot_helpers.make_2d_hist(
                x_edges=num_hist_data["x_edges"],
                y_edges=num_hist_data["y_edges"],
                values=num_hd,
                variances=num_hist_data["variances"],
                x_label=num_hist_data["x_label"],
                y_label=num_hist_data["y_label"]
            )

            num_hist_obj_2d.plot2d(cmap="turbo",
                                   cmin=kwargs.get("zlim", [None, None])[0],
                                   cmax=kwargs.get("zlim", [None, None])[1])

            ax_bottom_right = fig.add_subplot(gs[1, 1])
            den_hd = np.array(den_hist_data["values"])
            den_hd[den_hd < 0.001] = np.nan

            den_hist_obj_2d = plot_helpers.make_2d_hist(
                x_edges=den_hist_data["x_edges"],
                y_edges=den_hist_data["y_edges"],
                values=den_hd,
                variances=den_hist_data["variances"],
                x_label=den_hist_data["x_label"],
                y_label=den_hist_data["y_label"]
            )

            den_hist_obj_2d.plot2d(cmap="turbo",
                                   cmin=kwargs.get("zlim", [None, None])[0],
                                   cmax=kwargs.get("zlim", [None, None])[1])


            axis_list = [ax_big, ax_top_right, ax_bottom_right]
            if kwargs.get('xlim', False):
                for ax in axis_list:
                    ax.set_xlim(*kwargs.get('xlim'))

            if kwargs.get('ylim', False):
                for ax in axis_list:
                    ax.set_ylim(*kwargs.get('ylim'))


        else:
            if len(plot_data.get("hists", {})):
                key = next(iter(plot_data["hists"]))
                hist_data = plot_data["hists"][key]
            elif len(plot_data.get("stack", {})):
                key = next(iter(plot_data["stack"]))
                hist_data = plot_data["stack"][key]
            else:
                raise ValueError("No valid data found in plot_data")

            if kwargs.get("full", False):
                hist_obj_2d = plot_helpers.make_2d_hist(
                    x_edges=hist_data["x_edges"],
                    y_edges=hist_data["y_edges"],
                    values=hist_data["values"],
                    variances=hist_data["variances"],
                    x_label=hist_data["x_label"],
                    y_label=hist_data["y_label"]
                )

                fig = plt.figure()
                val = hist_obj_2d.plot2d_full(
                    main_cmap="jet",
                    top_color="k",
                    top_lw=2,
                    side_lw=2,
                    side_color="k",
                )
            else:
                # Mask 0s
                hd = np.array(hist_data["values"])
                hd[hd < 0.001] = np.nan

                hist_obj_2d = plot_helpers.make_2d_hist(
                    x_edges=hist_data["x_edges"],
                    y_edges=hist_data["y_edges"],
                    values=hd,
                    variances=hist_data["variances"],
                    x_label=hist_data["x_label"],
                    y_label=hist_data["y_label"]
                )

                fig = plt.figure()
                fig.add_axes((0.1, 0.15, 0.85, 0.8))
                hist_obj_2d.plot2d(cmap="turbo",
                                   cmin=kwargs.get("zlim", [None, None])[0],
                                   cmax=kwargs.get("zlim", [None, None])[1])

            ax = fig.gca()
            if kwargs.get('xlim', False):
                ax.set_xlim(*kwargs.get('xlim'))

            if kwargs.get('ylim', False):
                ax.set_ylim(*kwargs.get('ylim'))


            # Add additional plot elements if requested
            if kwargs.get("plot_contour", False):
                plot_border_SR()
            if kwargs.get("plot_leadst_lines", False):
                plot_leadst_lines()
            if kwargs.get("plot_sublst_lines", False):
                plot_sublst_lines()

        ax = fig.gca()


        hep.cms.label(kwargs.get('CMSText', "Internal"), data=True,
                      year=kwargs.get('year', "RunII").replace("UL", "20"), loc=0, ax=ax)

        if 'region' in plot_data["axis_opts"]:
            ax.set_title(f"{plot_data['axis_opts']['region']}  ({plot_data['cut']})", fontsize=16)
        else:
            ax.set_title(f"                       ({plot_data['cut']})", fontsize=16)


        return fig, ax

    except Exception as e:
        logger.error(f"Error in _plot2d_from_dict: {str(e)}")
        raise ValueError(f"Failed to create 2D plot: {str(e)}")
