# Plotting Module — Refactoring Notes

These are architectural observations from a code review of `src/plotting/`.
The current code works and is flexible, but has accumulated implicit contracts
that make it fragile to extend. This doc tracks the proposed changes.

---

## ✅ 1. `plot_data` dict is an implicit schema

The central data structure passed between the builder (`helpers_make_plot_dict.py`)
and renderer (`helpers_make_plot.py`) is a plain dict with implicit keys:
`hists`, `stack`, `ratio`, `kwargs`, `var`, `cut`, `axis_opts`. Its shape is
never validated; both sides just have to "know" what's in it.

**Fix:** Replace with a `PlotData` dataclass or `TypedDict`. Makes the contract
explicit and turns `KeyError` at render time into a construction-time failure.

**Done (branch `plot_types`, MR #77):** `PlotData` and `HistEntry` TypedDicts
defined in `src/plotting/plot_types.py`. Builder return types annotated.

---

## ✅ 2. `kwargs` threads two unrelated concerns through 5+ layers

Data-extraction options (`rebin`, `year`, `process`) and rendering options
(`xlabel`, `norm`, `yscale`, `doRatio`, `ylim`, ...) are bundled in the same
`kwargs` dict and passed from `makePlot` all the way down to `_draw_hists`.
No individual function documents which keys it consumes.

**Fix:** Split into `DataOptions` and `RenderOptions` — even simple dataclasses.
Makes per-function dependencies explicit and eliminates the "pass everything
and hope" pattern.

**Done (branch `plot_types`, MR #77):** `RenderOptions` dataclass in
`src/plotting/plot_types.py`. All render functions now take `opts: RenderOptions`
instead of `**kwargs`. `from_kwargs()` warns on unrecognized keys.
Public API (`makePlot`/`make2DPlot`) unchanged.
`DataOptions` deferred — data-extraction kwargs are low-churn and the
render-side was the pressing problem.

---

## ✅ 6. Ratio computation coupled to dict construction

Ratio values were computed and embedded in `plot_data["ratio"]` during dict
building. Any change to ratio behavior (normalization, uncertainty treatment)
touched both the builder and the renderer. Three separate compute sites:
`_add_1d_ratio_plots`, `_add_2d_ratio_plots`, `add_ratio_plots`.

**Done (branch `plot_types`, MR #77):**
- `HistSource` + `RatioSpec` dataclasses added to `plot_types.py`.
- All three builder functions now populate `plot_data["ratio_specs"]` with
  `RatioSpec` objects instead of computing values.
- `_resolve_hist_source`, `_compute_ratio_entry`, `_resolve_ratio_specs` added
  to `helpers_make_plot.py`; called at the start of `_plot_from_dict` and
  `_plot2d_from_dict`, consolidating computation into one place.
- Dead helper `get_values_variances_centers_from_dict` removed from builder.

---

## 3. Two completely parallel builder paths that share no code

`get_plot_dict_from_config` (YAML-driven, explicit ratio config) and
`get_plot_dict_from_list` (infers overlay from list inputs) are separate
top-to-bottom pipelines. The branch at the `makePlot` entry point is
effectively "which pipeline am I in?"

**Fix:** Normalize all inputs to lists up front and run one pipeline that
handles both cases. Ratio specification should be explicit in both paths.

**Note:** Best tackled after item 6 — once ratio specs are explicit the
two paths become much more similar and the merge is straightforward.

---

## 4. `if/elif` list-type explosion in `get_plot_dict_from_list`

Seven dispatch cases: cut list, process list, multi-file, process+multi-file,
var list, year list, axis_opts list. Each is a hand-written handler with its
own iteration. Grown organically one feature at a time.

**Fix:** Table-driven or strategy-pattern dispatch — each "list dimension"
registers how to iterate over itself. Adding a new list type becomes trivial.

**Note:** Tackle together with item 3.

---

## 5. `cfg` is a duck-typed global state bag

`cfg` is passed everywhere and accessed duck-typed: `cfg.hists`,
`cfg.plotConfig`, `cfg.cutList`, `cfg.hist_key`, `cfg.combine_input_files`,
etc. Different call paths require different subsets of these attributes with
no enforced contract.

**Fix:** Define a proper `AnalysisConfig` dataclass with explicit fields.
Each function signature then declares what it actually needs.

**Note:** Requires auditing all `cfg` consumers in coffea4bees before
touching — do last.

---

## Order of attack

| Step | Items | Status |
|------|-------|--------|
| 1 | Typed `PlotData` + `RenderOptions` | ✅ done (MR #77) |
| 2 | Decouple ratio spec from computation | ✅ done (MR #77) |
| 3 | Unify builder paths + list dispatch | ⬜ after step 2 |
| 4 | `cfg` → `AnalysisConfig` dataclass | ⬜ last |
