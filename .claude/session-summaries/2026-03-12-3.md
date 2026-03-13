# Session — 2026-03-12-3

## What we did
- Fixed `mixeddata_all` eras combining into separate histogram categories (e.g. `mixeddata_allG`) by excluding `processName == "mixeddata_all"` from the era-stripping override in processor
- Added `TTbar4b_from_MvD` histogram plots using MvD weights (`p_t4/p_mix4`) — new flag `plot_ttbar_with_MvD_weights`, weight field `weight_mix4_to_t4_MvD`, fill method `_fill_ttbar_MvD_detailed_cutflows`
- Fixed the FvT/MvD TTbar histogram early-return guard and refactored to independent blocks
- Diagnosed NaN values in MvD histogram counts: traced to NaN `p_t4`/`p_mix4` in the MvD friend tree
- Added NaN diagnostic in `event_weights.py` (later moved to processor post-selev), printing run/event/nJet_selected for NaN events
- Confirmed NaN events in selev all have `nJet_selected >= 5` (3-jet hypothesis ruled out for selected events)
- Printed selJet 4-vectors for NaN events: found `Jet_mass=NaN` directly in the picoAOD for soft forward jets
- Verified NaN jet mass in the picoAOD using uproot lookup by run/event number
- Wrote `check_picoAOD_nans.py` in `hemisphere_mixing/` — reports fraction of jets with NaN per 4-vector field; found 0.18% of jets have NaN `Jet_mass` (only mass affected, not pt/eta/phi)
- Root cause confirmed: hemisphere mixing boost produces jets with NaN mass (soft, forward jets: pt 12–25 GeV, |eta| > 1.8)

## Decisions
- NaN diagnostic moved to processor after `selev = event[analysis_selections]` — only check selected events to avoid noise from unselected events
- Used `print()` not `logging.warning()` for per-event output to avoid logger line-length truncation
- `check_picoAOD_nans.py` uses `ak.flatten` on uproot jagged arrays — plain numpy `isnan` fails on object arrays

## Files changed
- `coffea4bees/analysis/processors/processor_HH4b.py` — fixed processName override for `mixeddata_all`; added `plot_ttbar_with_MvD_weights` flag, cutflow, fill method, histogram call; added post-selev NaN diagnostic
- `coffea4bees/analysis/helpers/event_weights.py` — added `weight_mix4_to_t4_MvD` computation; NaN diagnostic removed (moved to processor)
- `coffea4bees/analysis/metadata/HH4b_MvD.yml` — added `plot_ttbar_with_MvD_weights: true`
- `coffea4bees/hemisphere_mixing/check_picoAOD_nans.py` — new script to check picoAOD for NaN jet 4-vectors

## Open threads
- NaN `Jet_mass` in mixed picoAODs needs a fix — hemisphere mixing boost producing imaginary mass for soft forward jets; fix options: (1) guard in `buildTop` against NaN jet mass, (2) fix in the mixing algorithm itself, (3) filter NaN-mass jets before top reconstruction
- Full condor MvD job not yet submitted
- Temporary NaN diagnostic prints in `processor_HH4b.py` should be removed once the root cause is fixed
