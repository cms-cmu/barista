# declustered_run3_20260927_7e9990e-f0d7cdf

*declustered_run3 — roasted by John Alison on 2026-09-27*

| | |
|---|---|
| barista | [`7e9990e8a45f`](https://gitlab.cern.ch/cms-cmu/barista/-/commit/7e9990e8a45fde0b08033ffef747ff997b6a19cd) |
| coffea4bees | [`f0d7cdf72580`](https://gitlab.cern.ch/cms-cmu/coffea4bees/-/commit/f0d7cdf72580bfbaa36700afab576dcbe8e66760) |
| config | `coffea4bees/workflows/config/declustered_run3.yml` (sha256 `733a7d89c835`) — [captured copy](https://johnda.web.cern.ch/johnda/HH4b/prod/declustered_run3_20260927_7e9990e-f0d7cdf/roasts/declustered_run3_20260927_7e9990e-f0d7cdf/config.yml) |
| results | [https://johnda.web.cern.ch/johnda/HH4b/prod/declustered_run3_20260927_7e9990e-f0d7cdf/](https://johnda.web.cern.ch/johnda/HH4b/prod/declustered_run3_20260927_7e9990e-f0d7cdf/) |
| manifest | [roast.json](https://johnda.web.cern.ch/johnda/HH4b/prod/declustered_run3_20260927_7e9990e-f0d7cdf/roasts/declustered_run3_20260927_7e9990e-f0d7cdf/roast.json) |

| archive (EOS) | `root://cmseos.fnal.gov//store/user/jda102/HH4b_prod/declustered_run3_20260927_7e9990e-f0d7cdf/` (list: `xrdfs root://cmseos.fnal.gov ls -R /store/user/jda102/HH4b_prod/declustered_run3_20260927_7e9990e-f0d7cdf`) |

## Steps

| step | host | snakefile | state | log |
|---|---|---|---|---|
| DeClustered | cmslpc | `coffea4bees/workflows/Snakefile_DeClustered.smk`  | submitted 2026-09-27 | [DeClustered.log](https://johnda.web.cern.ch/johnda/HH4b/prod/declustered_run3_20260927_7e9990e-f0d7cdf/logs/DeClustered.log) |

## Pages

| workflow | page |
|---|---|
| D2 | [D2 gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/declustered_run3_20260927_7e9990e-f0d7cdf/output/declustered_run3_20260927_7e9990e-f0d7cdf/D2/index.html) |
| D5 | [cutflow_monitoring](https://johnda.web.cern.ch/johnda/HH4b/prod/declustered_run3_20260927_7e9990e-f0d7cdf/output/declustered_run3_20260927_7e9990e-f0d7cdf/D5/cutflow_monitoring.html) |
| D5 | [plots gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/declustered_run3_20260927_7e9990e-f0d7cdf/output/declustered_run3_20260927_7e9990e-f0d7cdf/D5/plots/index.html) |
| superseded_d3_to_t3 | [cutflow_monitoring](https://johnda.web.cern.ch/johnda/HH4b/prod/declustered_run3_20260927_7e9990e-f0d7cdf/output/declustered_run3_20260927_7e9990e-f0d7cdf/superseded_d3_to_t3/D5/cutflow_monitoring.html) |
| superseded_d3_to_t3 | [plots gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/declustered_run3_20260927_7e9990e-f0d7cdf/output/declustered_run3_20260927_7e9990e-f0d7cdf/superseded_d3_to_t3/D5/plots/index.html) |

## Notes

Run 3 DeClustered dataset, 1 seed, ttbar subtracted (non-tight FvT): synthetic_data_multijet + synthetic_data_4b (multijet + ttbar_PSData from mixeddata_run3_20260926_956d4bf-aca8a4e). D.1-D.5. Hists/config/FvT from nominal_run3_nontight_20260925_0c53380-193851f. Hot-patch 2026-09-27 (after the run): D5/cutflow_monitoring.{html,txt} regenerated with barista src/tools/cutflow_closure.py from e5b762af (--multijet-process syn_v0 --pseudodata ttbar_PSData), the shared closure-table tool D.5 uses from coffea4bees 1d13a5c6; counts unchanged. Rerun from D.3 (2026-09-27/28): first pass subtracted ttbar in the DeClusterer with FvT d3_to_t3 (helper default) on 4b events, removing 200k 4b events = 1.44x ttbar MC (D.1, correct d4_to_t4: 150k = 1.08x); data/model at passPreSel was 1.070. Hot-patched into the checkout from coffea4bees 40e6b7fd4: skimmer/processor/make_declustered_data_4b.py (the fix) + the D.5 closure-table files (rules/analysis.smk, Snakefile_DeClustered_5_monitoring.smk, scripts/declustered_validation_report.py); NOT the origin/master merge on that branch. First-pass D3/D4/D5 kept in output/<id>/superseded_d3_to_t3/; EOS picoAODs + handoff YAMLs overwritten by the rerun.

## History

- 2026-09-27 11:14:12 new 
- 2026-09-27 11:15:11 checkout host=cmslpc
- 2026-09-27 11:15:12 submit step=DeClustered host=cmslpc
- 2026-09-27 11:20:34 submit step=DeClustered host=cmslpc
- 2026-09-27 22:57:41 publish host=cmslpc ok=True
- 2026-09-27 23:09:07 publish host=cmslpc ok=True
- 2026-09-27 23:21:23 submit step=DeClustered host=cmslpc
- 2026-09-27 23:21:41 submit step=DeClustered host=cmslpc
- 2026-09-28 07:20:16 publish host=cmslpc ok=True
- 2026-09-28 07:32:28 archive host=cmslpc ok=True
- 2026-09-28 07:32:45 publish host=cmslpc ok=True
