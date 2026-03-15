# Session — 2026-03-12-4

## What we did
- Identified root cause of NaN `Jet_mass` in mixed picoAODs: `boost_jets_along_z` was recomputing `boosted_jets.mass` after a Lorentz boost via `sqrt(E^2 - |p|^2)`, which goes NaN for soft/forward jets due to floating-point cancellation
- Fixed by replacing `boosted_jets.mass` with `jets.mass` (original mass) in the output zip — mass is Lorentz-invariant so it is unchanged by the boost
- Confirmed fix eliminates NaN `Jet_mass` in `output/test_mixeddata_make_dataset_Run3/data_2022_EEE/picoAOD_mixed_all.root`

## Decisions
- Use `jets.mass` (pre-boost) instead of `boosted_jets.mass` (post-boost recompute) — mass is a Lorentz invariant; recomputing it introduces numerical error for soft/forward jets where `E ≈ |p|`

## Files changed
- `coffea4bees/hemisphere_mixing/mixing_helpers.py` — `boost_jets_along_z`: changed `"mass": boosted_jets.mass` → `"mass": jets.mass`

## Open threads
- Full condor MvD job not yet submitted
- Temporary NaN diagnostic prints in `processor_HH4b.py` should be removed now that root cause is fixed
