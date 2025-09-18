import operator as op
from functools import reduce
from itertools import chain
from typing import TypedDict

import pandas as pd
from src.hist import Label, LabelLike
from src.utils import unique
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    CustomJS,
    Div,
    HoverTool,
    Legend,
    LegendItem,
    ScrollBox,
    Slider,
    Toggle,
    UIElement,
)
from bokeh.plotting import figure

from ..template import SimpleImporter

_NA = "N/A"
_ARROW = " ⇒ "

_SCATTER_SIZE = 10
_VSPAN_WIDTH = 3
_VSPAN_KWARGS = {
    "color": "green",
    "alpha": 0.3,
}
_SCALAR_FIGURE = {
    "height": 300,
    "toolbar_location": "left",
    "tools": "xpan,xwheel_zoom,reset,save",
    "sizing_mode": "stretch_width",
}
_CURVE_FIGURE = {
    "toolbar_location": "left",
    "tools": "pan,wheel_zoom,reset,save",
}
_SLIDER_KWARGS = {
    "start": 1,
    "step": 1,
    "margin": (0, 20, 0, 20),
    "sizing_mode": "stretch_width",
}
_LEGEND_KWARGS = {
    "location": "top_center",
    "orientation": "vertical",
    "click_policy": "hide",
}

code = SimpleImporter(__file__)


class StyleDict(TypedDict):
    line: dict[str, dict[str]]
    scatter: dict[str, dict[str]]


def generate_toggles(keys: set[str], category: dict[str, list[str]]):
    category, keys = category.copy(), keys.copy()
    category[None] = keys
    toggles: dict[str, Toggle] = dict()
    layout = []
    for k, v in category.items():
        group: list[Toggle] = []
        for vv in sorted(v):
            if vv in keys:
                keys.remove(vv)
                toggles[vv] = Toggle(label=vv, active=True, button_type="primary")
                group.append(toggles[vv])
        if group:
            if k is None:
                k = "other"
            group.insert(0, Toggle(label=k, active=True, button_type="success"))
            for toggle in group[1:]:
                group[0].js_link("active", toggle, "active")
            layout.extend(group)
    return toggles, row(*layout, sizing_mode="stretch_width")


def generate_layout(toggles: UIElement, plots: list[UIElement]):
    return column(
        toggles,
        ScrollBox(
            child=column(plots, sizing_mode="stretch_width"),
            sizing_mode="stretch_both",
        ),
        sizing_mode="stretch_both",
    )


def plot_multiphase_scalar(
    *,
    plot: list[str],
    plot_data: dict[tuple[str, ...], pd.DataFrame],
    phase: pd.DataFrame,
    phase_milestone: list[str],
    style: StyleDict,
    category: dict[str, list[str]],
):
    layout = []
    milestone = phase_milestone
    # generate toggles
    cat_toggles, toggle_row = generate_toggles(
        set(chain.from_iterable(plot_data.keys())), category
    )
    # generate phase separator
    separators = {"x": [], "label": []}
    _nulls = phase.isnull()
    _phase = phase.loc[:, milestone].astype(str, copy=True)
    # https://github.com/pandas-dev/pandas/issues/20442
    _phase[_nulls] = _NA
    _shifted = _phase.shift(1, fill_value=_NA)
    _changed = _phase != _shifted
    _selection = _changed.any(axis=1)
    _phase, _shifted = _phase[_selection], _shifted[_selection]
    _labels = pd.DataFrame(index=_phase.index)
    for p in _phase.columns:
        _labels[p] = _shifted[p].str.cat(_phase[p], sep=_ARROW)
    _labels[~_changed] = None
    separators = {"x": [], "label": [], "width": []}
    for i, r in _labels.iterrows():
        r = r.dropna()
        separators["x"].append(i - 0.5)
        separators["label"].append(
            "\n".join(code.html("custom_tooltip", key=k, value=v) for k, v in r.items())
        )
        separators["width"].append(len(r) * _VSPAN_WIDTH)
    del _phase, _nulls, _shifted, _changed, _labels, _selection
    # plot data
    dfs = {k: pd.concat([plot_data[k], phase], axis=1) for k in sorted(plot_data)}
    columns = sorted(
        unique(chain.from_iterable(map(lambda x: x.columns, dfs.values())))
    )
    scatter_hover = HoverTool(
        tooltips=[(k, "@{" + k + "}") for k in columns],
    )
    scatter_hover.renderers = []
    phase_hover = HoverTool(tooltips="@label")
    phase_hover.renderers = []
    shared_x_range = None
    for p in plot:
        fig = figure(
            title=p,
            y_axis_label=p,
            **_SCALAR_FIGURE,
        )
        fig.add_tools(scatter_hover, phase_hover)
        if shared_x_range is None:
            shared_x_range = fig.x_range
        else:
            fig.x_range = shared_x_range
        legends = []
        for k, df in dfs.items():
            # glyph
            curve = fig.line(
                source=ColumnDataSource(data=df),
                x="index",
                y=p,
                **reduce(op.or_, (style["line"].get(tag, {}) for tag in k)),
            )
            scatter = fig.scatter(
                source=ColumnDataSource(data=df),
                x="index",
                y=p,
                **reduce(op.or_, (style["scatter"].get(tag, {}) for tag in k)),
                size=_SCATTER_SIZE,
            )
            legends.append(LegendItem(label=",".join(k), renderers=[curve, scatter]))
            # widgets
            togs = [cat_toggles[kk] for kk in k]
            for tog in togs:
                tog.js_on_change(
                    "active",
                    CustomJS(
                        args=dict(toggles=togs.copy(), curves=[curve, scatter]),
                        code=code.js("curve_toggle_visibility"),
                    ),
                )
            scatter_hover.renderers.append(scatter)
        fig.add_layout(Legend(items=legends, **_LEGEND_KWARGS), "right")
        vs = fig.vspan(
            x="x",
            width="width",
            source=ColumnDataSource(data=separators),
            **_VSPAN_KWARGS,
        )
        phase_hover.renderers.append(vs)
        layout.append(fig)
    return generate_layout(toggle_row, layout)


def list_last_scalar(
    *,
    plot_data: dict[tuple[str, ...], pd.DataFrame],
    phase: pd.DataFrame,
    **_,
):

    dfs = {k: pd.concat([plot_data[k], phase], axis=1) for k in sorted(plot_data)}
    columns = sorted(
        unique(chain.from_iterable(map(lambda x: x.columns, dfs.values())))
    )
    layout = []
    for k, df in dfs.items():
        table = "".join(
            f"<tr><td>{col}</td><td>{df[col].iloc[-1]}</td></tr>"
            for col in columns
            if col in df.columns
        )
        layout.append(Div(text=f"<h3>{k}</h3><table>{table}</table>"))
    return column(
        ScrollBox(
            child=column(layout, sizing_mode="stretch_width"),
            sizing_mode="stretch_both",
        ),
        sizing_mode="stretch_both",
    )


def plot_multiphase_curve(
    *,
    phase: pd.DataFrame,
    data: dict[str, dict[tuple[str, ...], list[pd.DataFrame]]],
    style: StyleDict,
    category: dict[str, list[str]],
    x_axis: LabelLike = "x",
    y_axis: LabelLike = "y",
    figure_kwargs: dict[str] = None,
):
    x_axis = Label(x_axis)
    y_axis = Label(y_axis)
    figure_kwargs = figure_kwargs or {}
    layout = []
    # generate toggles
    cat_toggles, toggle_row = generate_toggles(
        set(
            chain.from_iterable(
                chain.from_iterable(map(lambda x: x.keys(), data.values()))
            )
        ),
        category,
    )
    # generate phases
    phases = {}
    for i, r in phase.astype(str).iterrows():
        phases[i + 1] = "\n".join(
            code.html("custom_tooltip", key=k, value=v) for k, v in r.items()
        )
    # plot data
    shared_slider = None
    for plot, curves in data.items():
        n_phase = len(phases)
        # sanity check
        if not all(len(v) == n_phase for v in curves.values()):
            raise ValueError("Number of phases and curves do not match.")
        # generate components
        banner = Div(text=phases[n_phase])
        slider = Slider(
            end=n_phase,
            value=n_phase,
            title=f"{plot} step",
            **_SLIDER_KWARGS,
        )
        if shared_slider is None:
            shared_slider = slider
        else:
            slider.js_link("value", shared_slider, "value")
            shared_slider.js_link("value", slider, "value")
        fig = figure(
            title=plot,
            x_axis_label=x_axis.display,
            y_axis_label=y_axis.display,
            **_CURVE_FIGURE | figure_kwargs,
        )
        dfs = {
            k: pd.concat(v, axis=1, keys=[*map(str, range(1, n_phase + 1))])
            for k, v in curves.items()
        }
        slider.js_on_change(
            "value",
            CustomJS(
                args=dict(source=phases, slider=slider, div=banner),
                code=code.js("div_slider_switch_content"),
            ),
        )
        legends = []
        for k, df in dfs.items():
            # glyph
            source = ColumnDataSource(data=df)
            source.data["x"] = source.data[f"{n_phase}_{x_axis.code}"]
            source.data["y"] = source.data[f"{n_phase}_{y_axis.code}"]
            curve = fig.line(
                source=source,
                x="x",
                y="y",
                **reduce(op.or_, (style["line"].get(tag, {}) for tag in k)),
            )
            legends.append(LegendItem(label=",".join(k), renderers=[curve]))
            # widgets
            slider.js_on_change(
                "value",
                CustomJS(
                    args=dict(
                        source=source, slider=slider, x=x_axis.code, y=y_axis.code
                    ),
                    code=code.js("curve_slider_switch_data"),
                ),
            )
            togs = [cat_toggles[kk] for kk in k]
            for tog in togs:
                tog.js_on_change(
                    "active",
                    CustomJS(
                        args=dict(toggles=togs.copy(), curves=[curve]),
                        code=code.js("curve_toggle_visibility"),
                    ),
                )
        fig.add_layout(Legend(items=legends, **_LEGEND_KWARGS), "right")
        layout.append(column(slider, row(fig, banner), sizing_mode="stretch_width"))

    return generate_layout(toggle_row, layout)
