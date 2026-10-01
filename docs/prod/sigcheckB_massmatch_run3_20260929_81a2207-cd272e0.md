# sigcheckB_massmatch_run3_20260929_81a2207-cd272e0

*sigcheckB_massmatch_run3 — roasted by John Alison on 2026-09-29*

| | |
|---|---|
| barista | [`81a22074b0a6`](https://gitlab.cern.ch/cms-cmu/barista/-/commit/81a22074b0a679364653ba1da1cb422e8d205555) |
| coffea4bees | [`cd272e08ec3a`](https://gitlab.cern.ch/cms-cmu/coffea4bees/-/commit/cd272e08ec3a99b63491d9246fe287f5e37d53b8) |
| config | `coffea4bees/workflows/config/declustered_run3_library_sigcheck_massmatch.yml` (sha256 `199ad351fe34`) — [captured copy](https://johnda.web.cern.ch/johnda/HH4b/prod/sigcheckB_massmatch_run3_20260929_81a2207-cd272e0/roasts/sigcheckB_massmatch_run3_20260929_81a2207-cd272e0/config.yml) |
| results | [https://johnda.web.cern.ch/johnda/HH4b/prod/sigcheckB_massmatch_run3_20260929_81a2207-cd272e0/](https://johnda.web.cern.ch/johnda/HH4b/prod/sigcheckB_massmatch_run3_20260929_81a2207-cd272e0/) |
| manifest | [roast.json](https://johnda.web.cern.ch/johnda/HH4b/prod/sigcheckB_massmatch_run3_20260929_81a2207-cd272e0/roasts/sigcheckB_massmatch_run3_20260929_81a2207-cd272e0/roast.json) |



## Steps

| step | host | snakefile | state | log |
|---|---|---|---|---|
| DeClustered | cmslpc | `coffea4bees/workflows/Snakefile_DeClustered.smk` all_D6 | submitted 2026-09-29 | [DeClustered.log](https://johnda.web.cern.ch/johnda/HH4b/prod/sigcheckB_massmatch_run3_20260929_81a2207-cd272e0/logs/DeClustered.log) |

## Notes

D.6 signal scrambling check (max_chunks 20/year) on the declustered_run3_library roast's library, variant B (mass_match_weight 1.0 for <2b groups)

## History

- 2026-09-29 21:33:06 new 
- 2026-09-29 21:39:31 checkout host=cmslpc
- 2026-09-29 21:39:43 submit step=DeClustered host=cmslpc
- 2026-09-30 03:09:51 publish host=cmslpc ok=True
