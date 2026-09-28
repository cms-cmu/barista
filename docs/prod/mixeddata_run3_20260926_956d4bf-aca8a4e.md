# mixeddata_run3_20260926_956d4bf-aca8a4e

*mixeddata_run3 — roasted by John Alison on 2026-09-26*

| | |
|---|---|
| barista | [`956d4bf895a8`](https://gitlab.cern.ch/cms-cmu/barista/-/commit/956d4bf895a8c6e66cc9a6e637dec502ded0edb6) |
| coffea4bees | [`aca8a4e1346a`](https://gitlab.cern.ch/cms-cmu/coffea4bees/-/commit/aca8a4e1346a07083b2a8ce9faab575451bab8bb) |
| config | `coffea4bees/workflows/config/mixeddata_run3.yml` (sha256 `2a536743dfe8`) — [captured copy](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_20260926_956d4bf-aca8a4e/roasts/mixeddata_run3_20260926_956d4bf-aca8a4e/config.yml) |
| results | [https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_20260926_956d4bf-aca8a4e/](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_20260926_956d4bf-aca8a4e/) |
| manifest | [roast.json](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_20260926_956d4bf-aca8a4e/roasts/mixeddata_run3_20260926_956d4bf-aca8a4e/roast.json) |
| archive (EOS) | `root://cmseos.fnal.gov//store/user/jda102/HH4b_prod/mixeddata_run3_20260926_956d4bf-aca8a4e/` (list: `xrdfs root://cmseos.fnal.gov ls -R /store/user/jda102/HH4b_prod/mixeddata_run3_20260926_956d4bf-aca8a4e`) |

## Steps

| step | host | snakefile | state | log |
|---|---|---|---|---|
| MakeMixedData | cmslpc | `coffea4bees/workflows/Snakefile_MakeMixedData.smk`  | submitted 2026-09-27 | [MakeMixedData.log](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_20260926_956d4bf-aca8a4e/logs/MakeMixedData.log) |

## Pages

| workflow | page |
|---|---|
| M6 | [cutflow_validation](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_20260926_956d4bf-aca8a4e/output/mixeddata_run3_20260926_956d4bf-aca8a4e/M6/cutflow_validation.html) |
| M6 | [plots gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_20260926_956d4bf-aca8a4e/output/mixeddata_run3_20260926_956d4bf-aca8a4e/M6/plots/index.html) |
| M6 | [study gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_20260926_956d4bf-aca8a4e/output/mixeddata_run3_20260926_956d4bf-aca8a4e/M6/study/index.html) |

## Notes

Run 3 mixed-data datasets from the non-tight production nominal_run3_nontight_20260925_0c53380-193851f (JCM, jcm_hists, FvT). Supersedes mixeddata_run3_20260925_56690d5-1a7e467 (library built with the tight nominal FvT). HOTPATCHED 2026-09-26: mixer JCM lookup + empty-dataset guards (see history). HOTPATCH 4 (2026-09-26): mixeddata_4b + ttbar PSData, PSData unit weight, M.6 validation (see history). HOTPATCH 5 (2026-09-26): mixed-JCM weight in M.6 + PSData jets as stored (see history). HOTPATCH 6: study index page. HOTPATCH 7: equal-probability subsample wrap (see history).

## History

- 2026-09-26 08:26:20 new 
- 2026-09-26 08:28:13 checkout host=cmslpc
- 2026-09-26 08:28:24 submit step=MakeMixedData host=cmslpc
- 2026-09-26 08:31:04 submit step=MakeMixedData host=cmslpc
- 2026-09-26 09:01:37 submit step=MakeMixedData host=cmslpc
- 2026-09-26 09:38:16 hotpatch host=cmslpc files=['coffea4bees/skimmer/processor/make_mixed_data.py', 'coffea4bees/workflows/Snakefile_MakeMixedData.smk', 'coffea4bees/workflows/Snakefile_MakeMixedData_2_mix.smk', 'coffea4bees/workflows/Snakefile_MakeMixedData_5_ttbar_psdata.smk'] what=mixer JCM lookup (explicit JCM_file stored under 'default' but looked up by year -> None -> every chunk failed on nJet_ps_and_tag, skipbadfiles hid it, empty mixeddata_all published); new M2_check/M5_check refuse to publish an empty dataset. git diff on top of aca8a4e1.
- 2026-09-26 09:38:18 resume step=MakeMixedData host=cmslpc
- 2026-09-26 10:10:57 submit step=MakeMixedData host=cmslpc
- 2026-09-26 10:12:07 resume step=MakeMixedData host=cmslpc
- 2026-09-26 11:38:45 submit step=MakeMixedData host=cmslpc
- 2026-09-26 12:02:59 submit step=MakeMixedData host=cmslpc
- 2026-09-26 12:08:53 submit step=MakeMixedData host=cmslpc
- 2026-09-26 13:30:26 hotpatch host=cmslpc files=['coffea4bees/workflows/Snakefile_MakeMixedData_4_subsample.smk'] what=M4 split: runner picosize 1e9 -> one file per (subsample, era). With 100k the overlap-inflated subsamples got more chunks (v0 27 files vs v6 24) and M4_dataset_yml (correctly) refused: no single vXXX template fits. Applied on top of the earlier hotpatch.
- 2026-09-26 13:30:28 resume step=MakeMixedData host=cmslpc
- 2026-09-26 14:42:49 submit step=MakeMixedData host=cmslpc
- 2026-09-26 14:44:00 publish host=cmslpc ok=True
- 2026-09-26 14:44:02 archive host=cmslpc ok=True
- 2026-09-26 15:36:06 hotpatch host=cmslpc files=['coffea4bees/analysis/helpers/processor_config.py', 'coffea4bees/workflows/Snakefile_MakeMixedData.smk', 'coffea4bees/workflows/Snakefile_MakeMixedData_4_subsample.smk', 'coffea4bees/workflows/Snakefile_MakeMixedData_5_ttbar_psdata.smk', 'coffea4bees/workflows/Snakefile_MakeMixedData_6_validation.smk (new)', 'coffea4bees/plots/metadata/plotsMixedData_validation.yml (new)', 'coffea4bees/workflows/scripts/mixeddata_validation_report.py (new)', 'src/plotting/helpers_make_plot.py', 'src/plotting/helpers_make_plot_dict.py'] what=mixeddata_4b folds in the ttbar_PSData files (John); ttbar_PSData processed as unit-weight pseudodata (processor_config); new M.6 validation step (plots, cutflow page, study plots, overlap matrix); ratio denominator can be one stack component. Files copied (worktree state) into both checkouts; shared Dask daemon was not running.
- 2026-09-26 15:36:07 submit step=MakeMixedData host=cmslpc
- 2026-09-26 15:36:27 submit step=MakeMixedData host=cmslpc
- 2026-09-26 15:48:06 hotpatch host=cmslpc files=['coffea4bees/analysis/processors/processor_HH4b.py', 'coffea4bees/workflows/Snakefile_MakeMixedData_6_validation.smk'] what=M.6 mixed hists: apply_MvD true + apply_MvD_weight false (the only path that applies the JCM to 4b mixed events; first M.6 pass had mixed ~10x data). Run 3: ttbar_PSData jets taken as stored (isPSData joins isSyntheticData) instead of re-corrected as data (PSData/TT MC 0.85-0.90 after the dijet-mass cut).
- 2026-09-26 15:48:08 submit step=MakeMixedData host=cmslpc
- 2026-09-26 15:59:57 publish host=cmslpc ok=True
- 2026-09-26 15:59:58 archive host=cmslpc ok=True
- 2026-09-26 16:45:04 hotpatch host=cmslpc files=['coffea4bees/workflows/scripts/mixeddata_validation_report.py', 'coffea4bees/workflows/Snakefile_MakeMixedData_6_validation.smk'] what=M.6 study/index.html page (numbers + all study plots by topic). Presentation only.
- 2026-09-26 16:45:05 submit step=MakeMixedData host=cmslpc
- 2026-09-26 16:45:24 submit step=MakeMixedData host=cmslpc
- 2026-09-26 16:46:24 publish host=cmslpc ok=True
- 2026-09-26 22:19:25 hotpatch host=cmslpc files=['coffea4bees/hemisphere_mixing/mixing_helpers.py'] what=assign_mixed_subsamples: overflow wraps modulo floor(1/w) (the event's valid slices) instead of a fixed 9. With w up to 0.43 the %9 targets above 1 dropped large-w (high jet multiplicity) events from high-v subsamples: sizes v0-v3 2.02 M vs v6-v15 1.77 M and a composition bias. M.3 study, M.4 splits and M.6 rerun.
- 2026-09-26 22:19:27 submit step=MakeMixedData host=cmslpc
- 2026-09-26 22:19:50 submit step=MakeMixedData host=cmslpc
- 2026-09-27 00:15:59 publish host=cmslpc ok=True
- 2026-09-27 00:16:01 archive host=cmslpc ok=True
- 2026-09-27 19:48:09 hotpatch host=cmslpc files=['coffea4bees/analysis/helpers/event_weights.py'] what=git apply of coffea4bees a7de466da's event_weights.py hunk: mixed-data JCM argument nSelJets-3 (the fit's variable) instead of nUntaggedLoose+1, which under-weighted mixed events with >=5 loose tags (MvD roast: -4.7% of the four-tag SB yield). Only M6_hists_mixed uses this path; force-rerun, M.6 downstream rebuilt.
- 2026-09-27 19:48:10 submit step=MakeMixedData host=cmslpc
- 2026-09-27 20:49:43 hotpatch host=cmslpc files=['coffea4bees/hemisphere_mixing/mixing_helpers.py'] what=update_pseudoTagWeight_of_mixed_data (M.4 splitter + M.3 study): mixed-data JCM argument nSelJets-3 instead of nUntaggedLoose+1, the same fix as event_weights.py -- the subsamples were drawn with too small a probability for >=5-loose-tag events (M.6 after the processor fix: subsample/model 0.96 Run 3, 0.985 Run 2). Force-rerun M3_study + M4_split; M4 republishes handoff/mixeddata_4b.yml, M6 closure rebuilt.
- 2026-09-27 20:49:45 submit step=MakeMixedData host=cmslpc
- 2026-09-27 23:17:41 resume step=MakeMixedData host=cmslpc
- 2026-09-27 23:18:36 submit step=MakeMixedData host=cmslpc
- 2026-09-27 23:19:01 submit step=MakeMixedData host=cmslpc
- 2026-09-28 00:15:34 publish host=cmslpc ok=True
- 2026-09-28 00:15:37 archive host=cmslpc ok=True
