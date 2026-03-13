# Session — claude-md-refactor (2026-03-12)

## What we did
- Moved TTbar estimation and cutflow implementation notes out of `coffea4bees/CLAUDE.md` into a new reference file `coffea4bees/analysis/processors/ttbar-estimation.md`
- Added pointer to hemisphere mixing README (`hemisphere_mixing/README.md`) in `coffea4bees/CLAUDE.md`
- Created `coffea4bees/analysis/processors/apply_MvD.md` reference doc from `apply_MvD_plan.md` session summary
- Added pointer to `apply_MvD.md` in `coffea4bees/CLAUDE.md`
- Created `src/tools/make_dataset_yml.py` — converts picoaod_datasets output yml to analysis datasets yml (works for mixeddata, JetDeClustered, 4b skims, Run2+Run3)
- Created `src/tools/README.md` covering all tools in the directory
- Fixed `yaml.safe_load` → `yaml.full_load` to handle `!!python/tuple` tags in picoaod_datasets files
- Tested script successfully on `output/mixeddata_make_dataset_Run3_all/picoaod_datasets_mixeddata_Run3_noTT_pz.yml`

## Decisions
- Use separate reference `.md` files (not skills) for domain knowledge that's only sometimes relevant — skills are better for procedural workflows; plain path references are lazy-loaded
- `make_dataset_yml.py` goes in `src/tools/` not `coffea4bees/analysis/tools/` — no coffea4bees-specific logic, reusable across analysis packages
- Renamed from `make_mixeddata_dataset_yml.py` to `make_dataset_yml.py` — same script works for any picoaod_datasets output

## Files changed
- `coffea4bees/CLAUDE.md` — replaced TTbar/cutflow sections with pointer; added hemisphere mixing and MvD pointers
- `coffea4bees/analysis/processors/ttbar-estimation.md` — new reference doc for TTbar estimation and cutflow structure
- `coffea4bees/analysis/processors/apply_MvD.md` — new reference doc for MvD classifier application
- `src/tools/make_dataset_yml.py` — new script to convert picoaod_datasets yml to datasets yml
- `src/tools/README.md` — new README covering all tools in the directory

## Open threads
- `mixeddata_all_new.yml` was written as a test output; user may want to review and replace `mixeddata_all.yml` with it
