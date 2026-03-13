# Session — 2026-03-11

## What we did
- Diagnosed root cause of `IndexError: cannot slice ListOffsetArray (of length 0)` during MvD classifier evaluation on `mixed_all` picoAODs
- Added diagnostic logging to `src/data_formats/root/io.py` and `src/data_formats/root/_backend.py` to identify which files trigger 0-row reads and size mismatches in `merge_record`
- Confirmed via diagnostics that batches at `[66667, 100000)` were being attempted on HCR_input files with only 60269 entries
- Traced the crash chain: picoAOD (100000 events) → 3 balanced batches → batch 3 exceeds HCR_input size → 0 rows returned → `pd.concat(axis=1)` crash with awkward extension array
- Identified that 14 of 264 mixed_all HCR_input entries in `classifier_inputs_MvD_Run3.json` have mismatched entry counts
- Discovered root cause is **path collisions**: multiple eras within `2023_BPix` (BPixD, BPixE, etc.) all write HCR_input files with the same chunk number to the same shared directory `mixeddata_all_2023_BPix/`, overwriting each other; the JSON records different UUIDs/stop values for the same physical path
- Confirmed path collision pattern affects both 2022 and 2023 era groupings
- Fixed `mixeddata_all.yml` to have per-era sub-keys (`E/F/G`, `B/C/D`, `D/E`, `A/B/C/D/E/F`) matching `common.yml` era definitions, for both `mixeddata_all` and `mixeddata_all_noTT` sections
- Updated `_mixeddata_all` in `_picoAOD.py` to mirror `_data`: iterates `CollisionData.eras.items()` and reads `.mixeddata_all.{year}.picoAOD.{e}.files` per era

## Decisions
- Keep HCR_input as friend (not remove it) — `xW` and `xbW` are not in the picoAODs, only in HCR_input
- Mirror `data.yml` era structure in `mixeddata_all.yml` — cleanest fix, consistent with existing detector data approach; `common.yml` already defines the correct era letters (D/E for BPix, A-F for preBPix)
- Use diagnostic logging rather than silently skipping size mismatches — preserves crash visibility while adding context

## Files changed
- `src/data_formats/root/io.py` — added WARNING when uproot returns 0 rows at non-zero entry offset
- `src/data_formats/root/_backend.py` — added WARNING before `pd.concat` when DataFrames have mismatched lengths
- `coffea4bees/metadata/datasets_HH4b_Run3/mixeddata_all.yml` — restructured both `mixeddata_all` and `mixeddata_all_noTT` to per-era sub-keys under `picoAOD`
- `src/classifier/config/dataset/HCR/_picoAOD.py` — `_mixeddata_all` now iterates per era like `_data`

## Open threads
- Re-run `coffea4bees/scripts/classifier-inputs-mixeddata-all-Run3.sh` for the 14 affected era/chunk combinations to regenerate HCR_input files with unique era-specific output directories
- Regenerate `coffea4bees/metadata/datasets_HH4b_Run3/classifier_inputs_MvD_Run3.json` after HCR_input regeneration
- Re-run MvD evaluation to confirm the crash is resolved
- The `datasets_HH4b_Run3_2025_Run3_skims.yml` (used by evaluate.yml) still has flat `mixeddata_all` file lists — may need updating to match the new per-era structure if it's used for training/eval file loading
