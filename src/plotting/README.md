# `src/plotting` — Barista Plotting Library

Analysis-agnostic plotting infrastructure built on [hist](https://hist.readthedocs.io) and [mplhep](https://mplhep.readthedocs.io). Used by **coffea4bees** (HH→4b), **bbreww** (bbWW), and any other analysis package that loads its histograms in the standard coffea format.

---

## Module Map

```
src/plotting/
├── plots.py                  # Public API: makePlot(), make2DPlot(), load_hists(), parse_args()
├── iPlot_config.py           # plot_config dataclass (the "cfg" object threaded everywhere)
├── helpers.py                # Utilities: colors, ratio math, hist construction, file I/O
├── helpers_make_plot_dict.py # Data layer: hist → plot_data dict
├── helpers_make_plot.py      # Render layer: plot_data dict → matplotlib figure
├── plot_from_yaml.py         # Drive plots from a YAML file instead of Python
├── pb_pdf_to_png.py          # Batch PDF→PNG conversion
└── yaml_to_hepdata.py        # Convert plot YAML to HEPData format
```

Analysis packages add a thin wrapper on top (e.g. `coffea4bees/plots/plots.py:load_config_4b`).

---

## Architecture: Two-Layer Design

```
makePlot(cfg, var, cut, axis_opts, **kwargs)
         │
         ▼
 ┌───────────────────────────────┐
 │  Data layer                   │  helpers_make_plot_dict.py
 │  hist.Hist → plot_data dict   │
 └───────────┬───────────────────┘
             │  plot_data = {"hists":{}, "stack":{}, "ratio":{}, ...}
             ▼
 ┌───────────────────────────────┐
 │  Render layer                 │  helpers_make_plot.py
 │  plot_data dict → fig, ax     │
 └───────────────────────────────┘
```

The **data layer** knows about histogram axes, cuts, processes, years, and stacking rules. It extracts numpy arrays from `hist.Hist` objects and bundles them into a plain dict (`plot_data`).

The **render layer** knows nothing about the physics — it only reads `plot_data` and drives matplotlib/mplhep. This makes it straightforward to test rendering independently.

---

## The `plot_config` Object (`cfg`)

Everything flows through a single `plot_config` instance (defined in `iPlot_config.py`). Analysis scripts populate it once at startup; the plotting functions read it.

| Attribute | Type | Purpose |
|---|---|---|
| `hists` | `list[dict]` | One entry per input file. Each dict has `hists[var]` → `hist.Hist` and `categories` → axis name list. |
| `plotConfig` | `dict` | YAML-loaded process/stack/ratio definitions (see below). |
| `axisLabelsDict` | `dict` | `{hist_key: {axis_name: [values]}}` — populated by `read_axes_and_cuts()`. |
| `cutListDict` | `dict` | `{hist_key: [cut_names]}` — Boolean axes become cuts. |
| `axisLabels` | `dict` | Alias into `axisLabelsDict[hist_key]` (current active key). |
| `cutList` | `list` | Alias into `cutListDict[hist_key]`. |
| `hist_key` | `str` | Active histogram collection, typically `"hists"` or `"hists_ttbar"`. Switch with `cfg.set_hist_key(key)`. |
| `category_key` | `str` | Derived from `hist_key` — name of the axis-name list in each hist entry. |
| `outputFolder` | `str\|None` | If set, plots are auto-saved here. |
| `fileLabels` | `list[str]` | Labels for each input file when comparing multiple files. |
| `combine_input_files` | `bool` | If True, sum histograms across all input files instead of overlaying them. |

### Initialization (analysis-side)

```python
from src.plotting.iPlot_config import plot_config
from src.plotting.plots import load_hists, read_axes_and_cuts

cfg = plot_config()
cfg.plotConfig = load_config_4b("coffea4bees/plots/metadata/plotsAll.yml")
cfg.hists = load_hists(["hists.coffea"])
cfg.axisLabelsDict, cfg.cutListDict = read_axes_and_cuts(
    cfg.hists, cfg.plotConfig, hist_keys=["hists", "hists_ttbar"]
)
cfg.set_hist_key("hists")
```

---

## Public API (`plots.py`)

### `makePlot(cfg, var, cut, axis_opts, **kwargs) → (fig, ax)`

Make a 1D plot. Automatically routes to the list-overlay path or the config-stack path.

```python
from src.plotting.plots import makePlot

fig, ax = makePlot(cfg, "v4j.mass", cut="passPreSel", axis_opts={"region": "SR"})
```

### `make2DPlot(cfg, process, var, cut, axis_opts, **kwargs) → (fig, ax)`

Make a 2D histogram plot.

```python
fig, ax = make2DPlot(cfg, "Multijet", var="quadJet_selected.lead_vs_subl_m",
                     cut="failSvB", axis_opts={"region": "SR"})
```

### Routing logic in `makePlot`

If **any** of these is a list, the function calls `get_plot_dict_from_list` (overlay mode):
- `cut`
- `var`
- `process` (kwarg)
- `year` (kwarg)
- any value in `axis_opts`
- `len(cfg.hists) > 1` and `not cfg.combine_input_files`

Otherwise it calls `get_plot_dict_from_config` (stack+config mode).

---

## `kwargs` Reference

These keyword arguments are accepted by `makePlot`, `make2DPlot`, and flow through to the render layer.

| kwarg | Type | Default | Effect |
|---|---|---|---|
| `process` | `str\|list` | None | Restrict to one or more processes; pass a list to overlay |
| `year` | `str\|list` | `"RunII"` | Year selection; `"RunII"` → sum over all years. Pass list to overlay. |
| `rebin` | `int` | `1` | Rebin factor applied to the last axis |
| `norm` | `bool` | `False` | Normalize each histogram to unit area |
| `doRatio` / `doratio` | `bool` | `False` | Add a ratio subplot |
| `yscale` | `str` | None | `"log"` for log y-axis |
| `xscale` | `str` | None | `"log"` for log x-axis |
| `xlim` | `[lo, hi]` | None | x-axis range |
| `ylim` | `[lo, hi]` | None | y-axis range |
| `rlim` | `[lo, hi]` | `[0,2]` | Ratio panel y-range |
| `rlabel` | `str` | `"Ratio"` | Ratio panel y-label |
| `xlabel` | `str` | (from hist axis) | Override x-axis label |
| `ylabel` | `str` | (from hist axis) | Override y-axis label |
| `legend` | `bool` | `True` | Show legend |
| `legend_loc` | `str` | `"best"` | `plt.legend` loc |
| `legend_order` | `list[str]` | None | Explicit label ordering in legend |
| `legend_fontsize` | `int` | `22` | Legend font size |
| `add_flow` | `bool` | `False` | Fold under/overflow into first/last bins |
| `uniform_bins` | `bool` | `False` | Plot with integer bin indices (useful for MVA score or combined-variable axes) |
| `uniform_bins_tick_step` | `int` | `max(1, n//10)` | Tick spacing for uniform-bin x-axis |
| `labels` | `list[str]` | None | Override auto-generated overlay labels |
| `fileLabels` | `list[str]` | `cfg.fileLabels` | Labels when overlaying multiple input files |
| `var_over_ride` | `dict` | `{}` | Map `{process: var}` to plot different variables per process |
| `hist_key_list` | `list[str]` | None | Switch hist_key per entry when overlaying cuts (e.g. `["hists","hists_ttbar"]`) |
| `outputFolder` | `str` | None | Save PDFs here (organized by year/cut/region) |
| `write_yaml` | `bool` | `False` | Also save the `plot_data` dict as YAML alongside the PDF |
| `CMSText` | `str` | `"Internal"` | CMS label text |
| `year_str` | `str` | (from `year`) | Override the year string shown on the CMS label |
| `do_title` | `bool` | `True` | Show region name as subplot title |
| `debug` | `bool` | `False` | Verbose logging |
| `text` | `dict` | `{}` | Extra text annotations: `{"label": {"xpos":0.5, "ypos":0.9, "text":"...", "fontsize":18}}` |

---

## The Overlay Mechanism

When any axis is a list, `get_plot_dict_from_list` dispatches to one of these handlers:

| Handler | What varies | Sets `edgecolor`? |
|---|---|---|
| `_handle_cut_list` | `cut` per entry | No |
| `_handle_axis_opts_list` | one `axis_opts` key per entry | No |
| `_handle_process_list` | `process` per entry | No (uses per-process color from plotConfig) |
| `_handle_var_list` | `var` per entry | Yes |
| `_handle_year_list` | `year` per entry | Yes |
| `_handle_input_files` | input file per entry | No |

Each handler uses `_setup_overlay_config()` to deep-copy the base process config and assign sequential colors from `helpers.COLORS`. Labels are auto-generated as `"{base_label} {item}"` unless `labels=` is passed.

---

## The `plot_data` Dict (Intermediate Representation)

Both data-layer functions return a `plot_data` dict with this structure:

```python
plot_data = {
    "var":       str,            # variable name
    "cut":       str,            # cut name
    "axis_opts": dict,           # e.g. {"region": "SR", "tag": "fourTag"}
    "process":   str|list,       # set for 2D plots
    "kwargs":    dict,           # all kwargs forwarded to the render layer

    "hists": {                   # individual (non-stacked) histograms
        "proc_key": {
            "process":    str|list,
            "label":      str,
            "fillcolor":  str,
            "edgecolor":  str,
            "histtype":   str,     # "errorbar", "step", "fill"
            "values":     list[float],
            "variances":  list[float],
            "centers":    list[float],
            "edges":      list[float],
            "x_label":    str,
            "under_flow": float,
            "over_flow":  float,
        },
        ...
    },

    "stack": {                   # stacked background components
        "proc_key": { same fields as hists entry },
        ...
    },

    "ratio": {                   # ratio panel entries
        "ratio_<key>": {
            "ratio":   list[float],
            "error":   list[float],
            "centers": list[float],
            "color":   str,
            "type":    str,        # "bar" (errorbar), "band", "step", "fill"
            "hatch":   str,        # for "band" type
        },
        ...
    },
}
```

For **2D** histograms, `hists` entries use `x_edges`, `y_edges`, `x_label`, `y_label` instead of `edges`/`centers`/`x_label`.

---

## The `plotConfig` YAML Structure

The `plotConfig` dict (loaded from YAML by the analysis package) drives `get_plot_dict_from_config`. Minimum required structure:

```yaml
hists:
  data:
    process: data
    label: Data
    fillcolor: "xkcd:black"
    edgecolor: "xkcd:black"
    histtype: errorbar
  HH4b:
    process: HH4b_kl1
    label: "HH4b (kl=1)"
    fillcolor: "xkcd:red"
    edgecolor: "xkcd:red"
    histtype: step

stack:
  Multijet:
    process: Multijet
    label: Multijet
    fillcolor: "xkcd:light blue"
    edgecolor: "xkcd:blue"
    year: RunII
  TTbar:
    process: [TTToHadronic, TTTo2L2Nu, TTToSemiLeptonic]
    label: t#bar{t}
    fillcolor: "xkcd:green"
    edgecolor: "xkcd:dark green"
    year: RunII

ratios:
  data_over_bkg:
    numerator:   {type: hists, key: data}
    denominator: {type: stack}
    label: Data/Bkg
    bkg_err_band: {color: k, type: band, hatch: "\\\\\\"}

# Optional: integer-coded region/tag axes (backwards compatibility)
codes:
  region: {SR: 2, SB: 1, other: 0}
  tag:    {fourTag: 4, threeTag: 3, other: 0}
```

Stack components can also be **sums** of sub-processes, each fetched separately and combined:

```yaml
stack:
  TTbar:
    label: t#bar{t}
    fillcolor: "xkcd:green"
    edgecolor: "xkcd:dark green"
    year: RunII
    sum:
      TTToHadronic: {process: TTToHadronic, label: TTToHadronic, fillcolor: ...}
      TTTo2L2Nu:    {process: TTTo2L2Nu,    label: TTTo2L2Nu,    fillcolor: ...}
```

The analysis-specific `load_config_4b()` also expands **process templates** — entries with `XXX` in the name and an `nSamples` field expand into N individual entries with sequential colors. Used for hemisphere-mixed pseudo-data samples.

---

## Interactive Plotting (`iPlot`)

`coffea4bees/plots/iPlot.py` provides a REPL-friendly wrapper. Start it with:

```bash
./run_container python -i coffea4bees/plots/iPlot.py hists.coffea -m coffea4bees/plots/metadata/plotsAll.yml
```

Then in the Python REPL:

```python
# Discover what's available
ls()                         # list variables
ls("region")                 # list regions
info()                       # dump full config summary
examples()                   # print usage examples

# Basic plots
plot("v4j.mass", cut="passPreSel", region="SR")
plot("v4j.mass", cut="passPreSel", region="SR", doRatio=1, rebin=4, yscale="log")

# Overlay regions
plot("v4j.mass", cut="passPreSel", region=["SR","SB"], process="data", doRatio=1)

# Overlay cuts
plot("v4j.mass", cut=["passPreSel","passSvB","failSvB"], region="SR", process="data", norm=1)

# Overlay processes
plot("v4j.mass", cut="passPreSel", region="SR", process=["data","Multijet","HH4b"], norm=1)

# Overlay years
plot("v4j.mass", cut="passPreSel", region="SR", process="data",
     year=["UL16_preVFP","UL16_postVFP","UL17","UL18"], doRatio=1)

# 2D
plot2d("quadJet_selected.lead_vs_subl_m", process="Multijet", cut="failSvB", region="SR")
plot2d("quadJet_selected.lead_vs_subl_m", process="Multijet", cut="failSvB", region="SR", full=True)

# Compare two input files
# (launch with two input files: iPlot.py fileA.coffea fileB.coffea -l labelA labelB)
plot("v4j.mass", cut="passPreSel", region="SR", process="data")
```

`iPlot` automatically switches `cfg.hist_key` to `"hists_ttbar"` when `cut` is `"passMuMu"` or `"passElMu"`, enabling ttbar-from-data plots.

Wildcard variable matching: `plot("canJet*", ...)` prints matching variable names instead of plotting.

---

## Batch Plotting (`makePlots.py`)

`coffea4bees/plots/makePlots.py` runs all standard plots for a given input file. Driven by two YAMLs:

- **`plotsAll.yml`** (or similar) — `plotConfig` defining processes, stacks, ratios
- **`plotModifiers.yml`** — per-variable overrides (rebin, xlim, ylim, etc.)

```bash
./run_container python coffea4bees/plots/makePlots.py hists.coffea \
    -m coffea4bees/plots/metadata/plotsAll.yml \
    --modifiers coffea4bees/plots/metadata/plotModifiers.yml \
    -o plots/
```

---

## HH→4b Physics Overlays

`helpers_make_plot.py` has three physics-specific functions for 2D Higgs mass plane plots:

```python
plot_border_SR()       # Four elliptic SR contours in (m_lead, m_subl) plane
plot_leadst_lines()    # Leading-jet pt cut lines
plot_sublst_lines()    # Subleading-jet pt cut lines
```

These are activated via kwargs in `make2DPlot`:

```python
make2DPlot(..., plot_contour=True, plot_leadst_lines=True, plot_sublst_lines=True)
```

The SR ellipses are defined by four Higgs mass targets (in GeV):
- `(127.5, 122.5)` — HH node (both on-shell H)
- `(127.5,  89.18)` — ZH node (lead on-shell H, subl on-shell Z)
- `( 92.82, 122.5)` — ZH node (lead on-shell Z, subl on-shell H)
- `( 92.82,  89.18)` — ZZ node

The ellipse radius is `((m-m0)/(0.1*m))² < threshold²` where threshold is 1.90 for the H nodes and 2.60 for the ZZ node.

---

## κλ Reweighting

Arbitrary HH κλ coupling hypotheses can be plotted without generating new MC samples. The function `helpers.make_klambda_hist(kl_value, plot_data)` combines the four EFT basis samples (`HH4b_kl0`, `HH4b_kl1`, `HH4b_kl2p45`, `HH4b_kl5`) using the `ggF` interference weight matrix from `src/physics/dihiggs/di_higgs.py`.

Usage: pass `process="HH4b_kl2"` (or any numeric κλ) to `iPlot.plot()` — the routing code in `get_plot_dict_from_list` calls `make_klambda_hist` automatically when the process name is not found in `plotConfig`.

---

## Output

When `outputFolder` is set, plots are saved as PDFs with a structured path:

```
<outputFolder>/<year>/<cut>/<region_tag>/<process>/<var>.pdf
```

e.g. `plots/RunII/passPreSel/region_SR/data/v4j_mass.pdf`

If `write_yaml=True`, the full `plot_data` dict is also serialized alongside the PDF — useful for debugging or reproducing a plot without re-running the analysis.

---

## Color Palette

`helpers.COLORS` is a 16-color list (cycling black, red, green, blue, orange, violet, grey, pink) from the XKCD color vocabulary. Used in order for overlays. Customize by passing `fillcolor` directly in plotConfig or overriding with `labels=`.

---

## Adding a New Analysis Package

To use this library from a new analysis package:

1. Implement `load_config_<analysis>(metadata_yaml)` that returns a `plotConfig` dict.
2. Populate a `plot_config` instance (see initialization snippet above).
3. Call `makePlot` / `make2DPlot` — everything else is handled by barista.

The only analysis-specific knowledge needed is:
- Process names that exist in your histograms
- The axis names for region/tag/cut selection
- The YAML metadata describing how to draw the stack and ratios

No changes to `src/plotting/` are needed for a new analysis.

---

## Known Limitations / Technical Debt

- `read_axes_and_cuts` skips axes with `extent > 20` with a `# HACK` comment — variable-bin axes used as category axes will be silently ignored.
- Region/tag axes stored as `hist.axis.IntCategory` (old format) require a `codes` dict in `plotConfig` for backwards compatibility. Newer histograms use string categories and don't need this.
- The `axis_opts` dict is mutated in-place in some call paths; `copy.deepcopy` guards are in the overlay handlers but callers should not rely on `axis_opts` being unchanged after a `makePlot` call.
