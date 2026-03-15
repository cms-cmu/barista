# Session — mvd-jcm-offset-fix (2026-03-13)

## What we did
- Investigated where JCM weights are computed for `mixeddata_all` during MvD classifier training
- Added `n_jets_offset: int = 0` parameter to `apply_JCM_from_list` in `coffea4bees/classifier/compatibility/JCM/fit.py`
- Passed `n_jets_offset=1` in the MvD dataset config so training adds 1 to `nSelJets` when looking up JCM weights for `mixeddata_all` fourTag events
- Wrote a README summarizing the MvD training/evaluation workflow in the MvD config directory

## Decisions
- Add offset as a parameter (`n_jets_offset`) rather than hardcoding — keeps it backward-compatible (default=0) and FvT unaffected
- Offset applied in `coffea4bees` (analysis-specific code), not barista base class — appropriate since the offset is MvD-specific
- README placed in `coffea4bees/classifier/config/workflows/HH4b_Run3/MvD/` to complement the existing `apply_MvD.md` (which covers analysis-side usage); README focuses on training/eval workflow

## Files changed
- `coffea4bees/classifier/compatibility/JCM/fit.py` — added `n_jets_offset` param; `n_jets` lookup now uses `df[n_jets_col] + self._n_jets_offset`
- `src/classifier/config/dataset/HCR/MvD.py` — passes `n_jets_offset=1` in the `partial(apply_JCM_from_list, ...)` call for MvD JCM weight application
- `coffea4bees/classifier/config/workflows/HH4b_Run3/MvD/README.md` — new file documenting the MvD fit workflow (classification task, model inputs, training/eval config, key code locations)

## Open threads
- MvD training needs to be re-run with the new offset to produce updated classifier weights and friend trees
