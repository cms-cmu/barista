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
plt.style.use([hep.style.CMS, {'font.size': 16}])

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

def plot_leadst_lines() -> None:
    """
    Plot leading and subleading lines for visualization.

    This function plots two mathematical functions:
    1. f(x) = (360/x) - 0.5
    2. f(x) = max(1.5, (650/x) + 0.5)

    The lines are plotted in red with a solid line style and default linewidth.

    Raises:
        ValueError: If the plotting operation fails
    """
    try:
        def func4(x: float) -> float:
            """Calculate the first function value.

            Args:
                x: Input value

            Returns:
                float: Function value at x
            """
            return (360/x) - 0.5

        def func6(x: float) -> float:
            """Calculate the second function value.

            Args:
                x: Input value

            Returns:
                float: Function value at x
            """
            return max(1.5, (650/x) + 0.5)

        # Plot func4 as a line plot
        x_func4 = np.linspace(100, 1100, 50)
        y_func4 = func4(x_func4)
        plt.plot(x_func4, y_func4,
                 color=DEFAULT_COLOR,
                 linestyle=DEFAULT_LINESTYLE,
                 linewidth=DEFAULT_LINEWIDTH)

        # Plot func6 as a line plot
        x_func6 = np.linspace(100, 1100, 50)
        y_func6 = [func6(x) for x in x_func6]
        plt.plot(x_func6, y_func6,
                 color=DEFAULT_COLOR,
                 linestyle=DEFAULT_LINESTYLE,
                 linewidth=DEFAULT_LINEWIDTH)

    except Exception as e:
        logger.error(f"Error in plot_leadst_lines: {str(e)}")
        raise ValueError(f"Failed to plot leading and subleading lines: {str(e)}")

def plot_sublst_lines() -> None:
    """
    Plot subleading lines for visualization.

    This function plots two mathematical functions:
    1. f(x) = (235/x)
    2. f(x) = max(1.5, (650/x) + 0.7)

    The lines are plotted in red with a solid line style and default linewidth.

    Raises:
        ValueError: If the plotting operation fails
    """
    try:
        def func4(x: float) -> float:
            """Calculate the first function value.

            Args:
                x: Input value

            Returns:
                float: Function value at x
            """
            return (235/x)

        def func6(x: float) -> float:
            """Calculate the second function value.

            Args:
                x: Input value

            Returns:
                float: Function value at x
            """
            return max(1.5, (650/x) + 0.7)

        # Plot func4 as a line plot
        x_func4 = np.linspace(100, 1100, 50)
        y_func4 = func4(x_func4)
        plt.plot(x_func4, y_func4,
                 color=DEFAULT_COLOR,
                 linestyle=DEFAULT_LINESTYLE,
                 linewidth=DEFAULT_LINEWIDTH)

        # Plot func6 as a line plot
        x_func6 = np.linspace(100, 1100, 50)
        y_func6 = [func6(x) for x in x_func6]
        plt.plot(x_func6, y_func6,
                 color=DEFAULT_COLOR,
                 linestyle=DEFAULT_LINESTYLE,
                 linewidth=DEFAULT_LINEWIDTH)

    except Exception as e:
        logger.error(f"Error in plot_sublst_lines: {str(e)}")
        raise ValueError(f"Failed to plot subleading lines: {str(e)}")


def plot_border_SR() -> None:
    """
    Plot the border of the signal region using contour plots.

    This function creates four contour plots representing the boundaries
    of the signal region using mathematical functions. The contours are
    plotted with a dashed orange-red line style.

    Raises:
        ValueError: If the plotting operation fails
    """
    try:
        def func0(x: float, y: float) -> float:
            """Calculate the first boundary function value.

            Args:
                x: x-coordinate
                y: y-coordinate

            Returns:
                float: Function value at (x,y)
            """
            return (((x - 127.5) / (0.1 * x)) ** 2 + ((y - 122.5) / (0.1 * y)) ** 2)

        def func1(x: float, y: float) -> float:
            """Calculate the second boundary function value.

            Args:
                x: x-coordinate
                y: y-coordinate

            Returns:
                float: Function value at (x,y)
            """
            return (((x - 127.5) / (0.1 * x)) ** 2 + ((y - 89.18) / (0.1 * y)) ** 2)

        def func2(x: float, y: float) -> float:
            """Calculate the third boundary function value.

            Args:
                x: x-coordinate
                y: y-coordinate

            Returns:
                float: Function value at (x,y)
            """
            return (((x - 92.82) / (0.1 * x)) ** 2 + ((y - 122.5) / (0.1 * y)) ** 2)

        def func3(x: float, y: float) -> float:
            """Calculate the fourth boundary function value.

            Args:
                x: x-coordinate
                y: y-coordinate

            Returns:
                float: Function value at (x,y)
            """
            return (((x - 92.82) / (0.1 * x)) ** 2 + ((y - 89.18) / (0.1 * y)) ** 2)

        # Create a grid of x and y values
        x = np.linspace(0, 250, 500)
        y = np.linspace(0, 250, 500)
        X, Y = np.meshgrid(x, y)

        # Compute the function values on the grid
        Z0 = func0(X, Y)
        Z1 = func1(X, Y)
        Z2 = func2(X, Y)
        Z3 = func3(X, Y)

        # Create the plot
        plt.contour(X, Y, Z0, levels=[1.90*1.90],
                    colors=BORDER_COLOR,
                    linestyles=BORDER_LINESTYLE,
                    linewidths=BORDER_LINEWIDTH)
        plt.contour(X, Y, Z1, levels=[1.90*1.90],
                    colors=BORDER_COLOR,
                    linestyles=BORDER_LINESTYLE,
                    linewidths=BORDER_LINEWIDTH)
        plt.contour(X, Y, Z2, levels=[1.90*1.90],
                    colors=BORDER_COLOR,
                    linestyles=BORDER_LINESTYLE,
                    linewidths=BORDER_LINEWIDTH)
        plt.contour(X, Y, Z3, levels=[2.60*2.60],
                    colors=BORDER_COLOR,
                    linestyles=BORDER_LINESTYLE,
                    linewidths=BORDER_LINEWIDTH)

    except Exception as e:
        logger.error(f"Error in plot_border_SR: {str(e)}")
        raise ValueError(f"Failed to plot signal region border: {str(e)}")


def _draw_plot_from_dict(plot_data: Dict[str, Any], **kwargs) -> None:
    """
    Draw a plot from a dictionary of plot data.

    Args:
        plot_data: Dictionary containing plot data including stack and histograms
        **kwargs: Additional plotting options:
            - norm: bool - Whether to normalize the plot
            - debug: bool - Whether to print debug information
            - xlabel: str - X-axis label
            - ylabel: str - Y-axis label
            - yscale: str - Y-axis scale ('log' or None)
            - xscale: str - X-axis scale ('log' or None)
            - legend: bool - Whether to show legend
            - ylim: List[float] - Y-axis limits [min, max]
            - xlim: List[float] - X-axis limits [min, max]
            - add_flow: bool - Whether to add under/overflow bins

    Raises:
        ValueError: If the plotting operation fails
        KeyError: If required data is missing from plot_data
    """
    try:
        if kwargs.get("debug", False):
            logger.info(f'\t in _draw_plot ... kwargs = {kwargs}')
        norm = kwargs.get("norm", False)

        # Draw the stack
        stack_dict = plot_data.get("stack", {})
        if not stack_dict:
            logger.warning("No stack data provided in plot_data")

        stack_dict_for_hist = {}
        for k, v in stack_dict.items():
            try:
                stack_dict_for_hist[k] = plot_helpers.make_hist(
                    edges=v["edges"],
                    values=v["values"],
                    variances=v["variances"],
                    x_label=v["x_label"],
                    under_flow=v["under_flow"],
                    over_flow=v["over_flow"],
                    add_flow=kwargs.get("add_flow", False)
                )
            except KeyError as e:
                logger.error(f"Missing required key in stack data: {e}")
                raise

        stack_colors_fill = [v.get("fillcolor") for _, v in stack_dict.items()]
        stack_colors_edge = [v.get("edgecolor") for _, v in stack_dict.items()]

        if len(stack_dict_for_hist):
            s = hist.Stack.from_dict(stack_dict_for_hist)

            s.plot(stack=True, histtype="fill",
                   color=stack_colors_fill,
                   label=None,
                   density=norm)

            s.plot(stack=True, histtype="step",
                   color=stack_colors_edge,
                   label=None,
                   density=norm)

        stack_patches = []

        # Add the stack components to the legend
        for _, stack_proc_data in stack_dict.items():
            _label = stack_proc_data.get('label')

            if _label in ["None"]:
                continue

            stack_patches.append(mpatches.Patch(
                facecolor=stack_proc_data.get("fillcolor"),
                edgecolor=stack_proc_data.get("edgecolor"),
                label=_label
            ))

        # Draw the hists
        hist_artists = {}
        for hist_proc_name, hist_data in plot_data.get("hists", {}).items():
            try:
                hist_obj = plot_helpers.make_hist(
                    edges=hist_data["edges"],
                    values=hist_data["values"],
                    variances=hist_data["variances"],
                    x_label=hist_data["x_label"],
                    under_flow=hist_data["under_flow"],
                    over_flow=hist_data["over_flow"],
                    add_flow=kwargs.get("add_flow", False)
                )

                _plot_options = {
                    "density": norm,
                    "label": hist_data.get("label", ""),
                    "color": hist_data.get('fillcolor', 'k'),
                    "histtype": kwargs.get("histtype", hist_data.get("histtype", "errorbar")),
                    "linewidth": kwargs.get("linewidth", hist_data.get("linewidth", 2)),
                    "yerr": False,
                }

                if kwargs.get("histtype", hist_data.get("histtype", "errorbar")) in ["errorbar"]:
                    _plot_options["markersize"] = 12
                    _plot_options["yerr"] = True

                hist_artists[hist_data.get("label")] = hist_obj.plot(**_plot_options)[0]

            except KeyError as e:
                logger.error(f"Missing required key in histogram data: {e}")
                raise



        # Set labels
        if kwargs.get("xlabel", None):
            plt.xlabel(kwargs.get("xlabel"))
        plt.xlabel(plt.gca().get_xlabel(), loc='right', fontsize=kwargs.get('xlabel_fontsize', 22))

        if kwargs.get("ylabel", None):
            plt.ylabel(kwargs.get("ylabel"))
        if norm:
            plt.ylabel(plt.gca().get_ylabel() + " (normalized)")
        plt.ylabel(plt.gca().get_ylabel(), loc='top', fontsize=kwargs.get('ylabel_fontsize', 22))

        # Set scales
        if kwargs.get("yscale", None):
            plt.yscale(kwargs.get('yscale'))
        if kwargs.get("xscale", None):
            plt.xscale(kwargs.get('xscale'))

        # Add legend
        if kwargs.get('legend', True):
            handles = []
            labels = []

            for s in stack_patches:
                handles.append(s)
                labels.append(s.get_label())

            data_handles, data_labels = plt.gca().get_legend_handles_labels()
            for h, l in zip(data_handles, data_labels):
                handles.append(h)
                labels.append(l)

            #
            legend_reverse = True
            if kwargs.get("legend_order", None):
                legend_reverse = False
                # Sort handles and labels based on legend_order
                sorted_labels =  []
                sorted_handles =  []
                for i in kwargs.get("legend_order"):
                    print(i)
                    sorted_labels.append(i)
                    sorted_handles.append(handles[labels.index(i)])


                handles = sorted_handles
                labels = sorted_labels



            plt.legend(
                handles=handles,
                labels=labels,
                loc=kwargs.get("legend_loc",'best'),
                frameon=False,
                reverse=legend_reverse,
                fontsize=kwargs.get('legend_fontsize', 22),
            )




        # Set limits
        if kwargs.get('ylim', False):
            plt.ylim(*kwargs.get('ylim'))
        if kwargs.get('xlim', False):
            plt.xlim(*kwargs.get('xlim'))

        # Add text annotations
        for _, _text_info in kwargs.get("text", {}).items():
            plt.text(_text_info["xpos"], _text_info["ypos"], _text_info["text"],
                     horizontalalignment=_text_info.get("horizontalalignment",'center'),
                     verticalalignment='top',
                     transform=plt.gca().transAxes,
                     fontsize=_text_info.get("fontsize", 22),
                     weight = _text_info.get("weight", 'normal'),
                     )


    except Exception as e:
        logger.error(f"Error in _draw_plot_from_dict: {str(e)}")
        raise ValueError(f"Failed to draw plot from dictionary: {str(e)}")


def _plot_from_dict(plot_data: Dict[str, Any], **kwargs) -> Tuple[plt.Figure, plt.Axes, Optional[plt.Axes]]:
    """
    Create a plot from a dictionary of plot data.

    Args:
        plot_data: Dictionary containing plot data including stack, histograms, and ratio
        **kwargs: Additional plotting options:
            - debug: bool - Whether to print debug information
            - year: str - Year for CMS label
            - ratio_line_value: float - Value for ratio plot line
            - rlabel: str - Label for ratio plot
            - rlim: List[float] - Limits for ratio plot
            - xlabel: str - X-axis label

    Returns:
        Tuple containing:
            - Figure object
            - Main axes object
            - Ratio axes object (if ratio plot is created, otherwise None)

    Raises:
        ValueError: If the plotting operation fails
        KeyError: If required data is missing from plot_data
    """
    try:
        if kwargs.get("debug", False):
            logger.info(f'\t in plot ... kwargs = {kwargs}')

        size = 7
        fig = plt.figure()

        do_ratio = len(plot_data.get("ratio", {}))
        if do_ratio:
            grid = fig.add_gridspec(2, 1, **kwargs.get("ratio_grid_config",RATIO_GRID_CONFIG))
            main_ax = fig.add_subplot(grid[0])
        else:
            fig.add_axes((0.1, 0.15, 0.85, 0.8))
            main_ax = fig.gca()
            ratio_ax = None

        year_str = plot_helpers.get_year_str(year=kwargs.get("year_str",kwargs.get('year', "RunII")))

        hep.cms.label("Internal", data=True,
                      year=year_str, loc=0, ax=main_ax)

        if kwargs.get("do_title", True) and 'region' in plot_data["axis_opts"] :
            main_ax.set_title(f"{plot_helpers.get_axis_str(plot_data['axis_opts']['region'])}")

        _draw_plot_from_dict(plot_data, **kwargs)

        if do_ratio:
            top_xlabel = plt.gca().get_xlabel()
            plt.xlabel("")

            ratio_ax = fig.add_subplot(grid[1], sharex=main_ax)
            plt.setp(main_ax.get_xticklabels(), visible=False)

            central_value_artist = ratio_ax.axhline(
                kwargs.get("ratio_line_value", 1.0),
                color="black",
                linestyle="dashed",
                linewidth=2.0
            )

            legend_handles = {}  # Store handles for legend

            for ratio_name, ratio_data in plot_data["ratio"].items():
                try:
                    error_bar_type = ratio_data.get("type", "bar")
                    label = ratio_data.get("label", ratio_name)  # Use ratio_name as fallback

                    if error_bar_type == "band":
                        # Only works for constant bin size
                        bin_width = (ratio_data["centers"][1] - ratio_data["centers"][0])

                        # Create hatched error regions using fill_between
                        for xi, yi, err in zip(ratio_data["centers"], ratio_data["ratio"], ratio_data["error"]):
                            plt.fill_between(
                                [xi - bin_width/2, xi + bin_width/2],
                                yi - err,
                                yi + err,
                                hatch=ratio_data.get("hatch", "/"),
                                edgecolor=ratio_data.get("color", "black"),
                                facecolor=ratio_data.get("facecolor", 'none'),
                                linewidth=0.0,
                                zorder=1
                            )

                        # Create a proxy artist for the legend
                        from matplotlib.patches import Rectangle
                        proxy = Rectangle((0, 0), 1, 1,
                                          hatch=ratio_data.get("hatch", "/"),
                                          edgecolor=ratio_data.get("color", "black"),
                                          facecolor=ratio_data.get("facecolor", 'none'),
                                          linewidth=0.0)
                        legend_handles[label]  = proxy


                    elif error_bar_type in ["step","fill"]:

                        hist_obj = plot_helpers.make_hist(
                            edges=ratio_data["edges"],
                            values=ratio_data["ratio"],
                            variances=ratio_data["variances"],
                            x_label=ratio_data.get("x_label",""),
                            under_flow=ratio_data.get("under_flow",0),
                            over_flow=ratio_data.get("over_flow", 0),
                            add_flow=kwargs.get("add_flow", False)
                        )

                        _plot_options = {
                            "density": False,
                            "label": ratio_data.get("label", ""),
                            "color": ratio_data.get('fillcolor', 'k'),
                            "histtype": ratio_data.get("type", "errorbar"),
                            "linewidth": kwargs.get("linewidth", ratio_data.get("linewidth", 2)),
                            "yerr": False,
                        }

                        #if kwargs.get("histtype", hist_data.get("histtype", "errorbar")) in ["errorbar"]:
                        #    _plot_options["markersize"] = 12
                        #    _plot_options["yerr"] = True

                        # Capture the plot handle - extract the stairs from StairsArtists
                        plot_result = hist_obj.plot(**_plot_options)
                        if plot_result and len(plot_result) > 0:
                            stairs_artist = plot_result[0]
                            # Extract the actual stairs object for the legend
                            if hasattr(stairs_artist, 'stairs') and stairs_artist.stairs is not None:
                                legend_handles[label] = stairs_artist.stairs

                        if ratio_data.get("type") == "fill":
                            _plot_options_edge = {
                                "density": False,
                                "label": ratio_data.get("label", ""),
                                "color": ratio_data.get('edgecolor', 'k'),
                                "histtype": "step",
                                "linewidth": kwargs.get("linewidth", ratio_data.get("linewidth", 2)),
                                "yerr": False,
                            }

                            #if kwargs.get("histtype", hist_data.get("histtype", "errorbar")) in ["errorbar"]:
                            #    _plot_options["markersize"] = 12
                            #    _plot_options["yerr"] = True

                            # Capture the plot handle - extract the stairs from StairsArtists
                            plot_result = hist_obj.plot(**_plot_options_edge)



                    else:
                        handle = ratio_ax.errorbar(
                            ratio_data["centers"],
                            ratio_data["ratio"],
                            yerr=ratio_data["error"],
                            color=ratio_data.get("color", "black"),
                            marker=ratio_data.get("marker", "o"),
                            linestyle=ratio_data.get("linestyle", "none"),
                            markersize=ratio_data.get("markersize", 4),
                        )
                        legend_handles[label] = handle


                except KeyError as e:
                    logger.error(f"Missing required key in ratio data: {e}")
                    raise

            # Set ratio plot labels and limits
            plt.ylabel(kwargs.get("rlabel", "Ratio"))
            plt.ylabel(plt.gca().get_ylabel(), loc='center', fontsize=kwargs.get('rlabel_fontsize', 22))
            plt.xlabel(kwargs.get("xlabel", top_xlabel), loc='right', fontsize=kwargs.get('xlabel_fontsize', 22))
            plt.ylim(*kwargs.get('rlim', [0, 2]))

            if kwargs.get("ratio_legend_order", {}):
                handles = []
                labels = []
                for _r_label in kwargs.get("ratio_legend_order", {}):
                    if _r_label in legend_handles:
                        labels.append(_r_label)
                        handles.append(legend_handles[_r_label])

                print(handles, labels)
                ratio_ax.legend(handles, labels, ncol=2, loc=kwargs.get("ratio_legend_loc",'upper left'))  # or specify location like loc='upper right'

        return fig, main_ax, ratio_ax

    except Exception as e:
        logger.error(f"Error in _plot_from_dict: {str(e)}")
        raise ValueError(f"Failed to create plot from dictionary: {str(e)}")


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

                except NameError:
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


        hep.cms.label("Internal", data=True,
                      year=kwargs.get('year', "RunII").replace("UL", "20"), loc=0, ax=ax)

        if 'region' in plot_data["axis_opts"]:
            ax.set_title(f"{plot_data['axis_opts']['region']}  ({plot_data['cut']})", fontsize=16)
        else:
            ax.set_title(f"                       ({plot_data['cut']})", fontsize=16)


        return fig, ax

    except Exception as e:
        logger.error(f"Error in _plot2d_from_dict: {str(e)}")
        raise ValueError(f"Failed to create 2D plot: {str(e)}")
