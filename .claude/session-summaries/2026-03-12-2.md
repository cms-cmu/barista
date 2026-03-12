# Session — 2026-03-12-2

## What we did
- Added `TTbar4b_from_MvD` histogram plots using MvD weights (`p_t4/p_mix4`), analogous to existing FvT `plot_ttbar_with_weights` feature
- Computed `weight_mix4_to_t4_MvD` in `event_weights.py` using `weights.partial_weight(exclude=["MvD"])` as base (includes JCM) multiplied by `p_t4/p_mix4` for fourTag events
- Added `_fill_ttbar_MvD_detailed_cutflows` method in processor filling only fourTag (no threeTag/twoTag)
- Fixed early-return guard in `histograms` to condition on both `plot_ttbar_with_weights` and `plot_ttbar_with_MvD_weights`
- Refactored FvT and MvD TTbar hist filling blocks to be independent (not nested)
- Enabled feature in `HH4b_MvD.yml` with `plot_ttbar_with_MvD_weights: true`
- Confirmed test passes: `JOB EXECUTION COMPLETED SUCCESSFULLY` (28004 events)

## Decisions
- Weight name `weight_mix4_to_t4_MvD` — user preference over `weight_d4_to_t4_MvD`
- Base weight excludes MvD but includes JCM — fourTag events already 4b-tagged so JCM applies; MvD is the reweighting factor itself
- `p_mix4 == 0` → weight set to 0.0 (not inf/nan)
- Cutflow output key: `TTbar4b_from_MvD_{era}` where era = `dataset.removeprefix("mixeddata_all_")`
- Only fourTag filled in cutflow and histograms — no threeTag component for MvD (unlike FvT which has both 3b and 4b)

## Files changed
- `coffea4bees/analysis/helpers/event_weights.py` — compute and store `weight_mix4_to_t4_MvD` in MvD branch of `add_pseudotagweights`
- `coffea4bees/analysis/processors/processor_HH4b.py` — new flag, cutflow init, `_fill_ttbar_MvD_detailed_cutflows` method, fixed early-return guard, MvD histogram call
- `coffea4bees/analysis/metadata/HH4b_MvD.yml` — added `plot_ttbar_with_MvD_weights: true`

## Open threads
- Full condor job not yet submitted — needs to be run from a login node with port forwarding
