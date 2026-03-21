"""
Typed contracts for the plotting pipeline.

  RenderOptions  — all options that control how a plot is rendered.
  PlotData       — the intermediate dict passed from builders to renderers.
  HistEntry      — the shape of a single histogram entry inside PlotData.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keys consumed exclusively by data-extraction functions (builder side).
# These appear in the raw kwargs dict but are NOT render options, so no
# warning is emitted when from_kwargs encounters them.
# ---------------------------------------------------------------------------
_DATA_ONLY_KEYS: frozenset = frozenset({
    'rebin', 'process', 'var_over_ride', 'hist_key_list', 'do2d',
    'fileLabels', 'labels', 'combine_input_files',
})


# ---------------------------------------------------------------------------
# RenderOptions
# ---------------------------------------------------------------------------

@dataclass
class RenderOptions:
    """All options that control how a 1-D or 2-D plot is rendered.

    Construct from a raw kwargs dict with::

        opts = RenderOptions.from_kwargs(kwargs)

    Unknown keys that are neither RenderOptions fields nor known
    data-extraction keys produce a logger.warning and are ignored.
    """

    # --- Axis labels ---------------------------------------------------------
    xlabel: Optional[str] = None
    xlabel_fontsize: int = 30
    ylabel: Optional[str] = None
    ylabel_fontsize: int = 30
    ylabel_labelpad: int = -4

    # --- Scales and limits ---------------------------------------------------
    yscale: Optional[str] = None
    xscale: Optional[str] = None
    ylim: Optional[Tuple] = None
    xlim: Optional[Tuple] = None

    # --- Normalisation and flow ----------------------------------------------
    norm: bool = False
    add_flow: bool = False

    # --- Uniform-bins mode ---------------------------------------------------
    uniform_bins: bool = False
    uniform_bins_tick_step: Optional[int] = None
    uniform_bins_fontsize: int = 14

    # --- Per-histogram style overrides ---------------------------------------
    # None means "fall through to the per-hist value stored in HistEntry".
    histtype: Optional[str] = None
    linewidth: Optional[int] = None

    # --- Legend --------------------------------------------------------------
    legend: bool = True
    legend_loc: str = "best"
    legend_fontsize: int = 22
    legend_order: Optional[List[str]] = None

    # --- Ratio panel ---------------------------------------------------------
    doRatio: bool = True
    rlabel: str = "Ratio"
    rlabel_fontsize: int = 30
    rlim: Tuple = (0, 2)
    ratio_line_value: float = 1.0
    ratio_grid_config: Optional[Dict] = None
    ratio_legend_order: Optional[List[str]] = None
    ratio_legend_loc: str = "upper left"

    # --- CMS label / title ---------------------------------------------------
    CMSText: str = "Internal"
    year: str = "RunII"
    year_str: Optional[str] = None   # overrides year for the CMS label only
    do_title: bool = True

    # --- Text annotations ----------------------------------------------------
    text: Dict = field(default_factory=dict)

    # --- Output --------------------------------------------------------------
    outputFolder: Optional[str] = None
    fmt: str = "pdf"
    dpi: Optional[int] = None
    write_yaml: bool = False

    # --- 2-D specific --------------------------------------------------------
    zlim: Tuple = (None, None)
    full: bool = False
    plot_contour: bool = False
    plot_leadst_lines: bool = False
    plot_sublst_lines: bool = False

    # --- Debug ---------------------------------------------------------------
    debug: bool = False

    # -------------------------------------------------------------------------

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]) -> RenderOptions:
        """Build a RenderOptions from a raw kwargs dict.

        Keys that are neither RenderOptions fields nor known data-extraction
        keys produce a warning but do not raise.
        """
        known_render = {f.name for f in dataclass_fields(cls)}
        render_kwargs: Dict[str, Any] = {}
        unknown: List[str] = []
        for k, v in kwargs.items():
            if k in known_render:
                render_kwargs[k] = v
            elif k not in _DATA_ONLY_KEYS:
                unknown.append(k)
        if unknown:
            logger.warning(
                f"RenderOptions.from_kwargs: unrecognized keys {unknown!r} — ignoring"
            )
        return cls(**render_kwargs)


# ---------------------------------------------------------------------------
# HistSource / RatioSpec
# ---------------------------------------------------------------------------


@dataclass
class HistSource:
    """Pointer to a histogram dataset within PlotData."""
    source: str = "hists"      # "hists" | "stack"
    key: Optional[str] = None  # key into plot_data[source]; None = sum all (stack)


@dataclass
class RatioSpec:
    """Specification for one entry in the ratio panel.

    Builder functions populate plot_data["ratio_specs"] with RatioSpec objects
    instead of computing ratio values directly.  The renderer resolves them via
    _resolve_ratio_specs just before drawing, consolidating what were previously
    three scattered compute sites into one pure function.

    Attributes:
        name: Key used in the resolved ratio dict.
        denominator: Source histogram to use as denominator.
        numerator: Source histogram for the numerator.  None means "band-at-1"
            — draw denominator uncertainty at y=1, no actual ratio computed.
        norm: Pass norm=True to make_ratio (scale num/den to unit area first).
        is_2d: True for 2-D ratio plots.
        style: Remaining render keys forwarded to _draw_ratio_panel
            (type, color, marker, hatch, label, ...).
    """
    name: str
    denominator: HistSource
    numerator: Optional[HistSource] = None
    norm: bool = False
    is_2d: bool = False
    style: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PlotData / HistEntry TypedDicts
# ---------------------------------------------------------------------------

from typing import TypedDict  # noqa: E402  (keep with its usage context)


class HistEntry(TypedDict, total=False):
    """Shape of a single histogram entry in PlotData['hists'] or ['stack']."""
    values: list
    variances: list
    edges: list
    centers: list
    under_flow: float
    over_flow: float
    x_label: str
    fillcolor: str
    edgecolor: str
    label: str
    histtype: str
    linewidth: float
    # 2-D fields
    x_edges: list
    y_edges: list
    y_label: str


class PlotData(TypedDict, total=False):
    """Intermediate data structure produced by builder functions and consumed
    by renderer functions.  All fields are optional (total=False) to
    accommodate both 1-D and 2-D paths and the gradual population pattern
    used by the builders."""
    hists: Dict[str, HistEntry]
    stack: Dict[str, HistEntry]
    ratio: Dict[str, Any]
    ratio_specs: List  # List[RatioSpec]
    var: Any          # str or list[str]
    cut: Optional[str]
    axis_opts: Dict[str, Any]
    kwargs: Dict[str, Any]   # raw kwargs; consumed by RenderOptions.from_kwargs
    process: Any             # str or list[str], used in 2-D path
    is_2d_hist: bool
    file_name: str
    tag: Any
