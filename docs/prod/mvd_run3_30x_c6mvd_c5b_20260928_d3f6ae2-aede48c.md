# mvd_run3_30x_c6mvd_c5b_20260928_d3f6ae2-aede48c

*mvd_run3_30x_c6mvd_c5b — roasted by John Alison on 2026-09-28*

| | |
|---|---|
| barista | [`d3f6ae22cc3b`](https://gitlab.cern.ch/cms-cmu/barista/-/commit/d3f6ae22cc3bc6be56c20b757afee1d884bf047a) |
| coffea4bees | [`aede48c9cf89`](https://gitlab.cern.ch/cms-cmu/coffea4bees/-/commit/aede48c9cf89f9f59c705d3e50071bd49e03cec6) |
| config | `coffea4bees/workflows/config/mvd_run3_30x_c6mvd_c5b.yml` (sha256 `dcd8f231ed53`) — [captured copy](https://johnda.web.cern.ch/johnda/HH4b/prod/mvd_run3_30x_c6mvd_c5b_20260928_d3f6ae2-aede48c/roasts/mvd_run3_30x_c6mvd_c5b_20260928_d3f6ae2-aede48c/config.yml) |
| results | [https://johnda.web.cern.ch/johnda/HH4b/prod/mvd_run3_30x_c6mvd_c5b_20260928_d3f6ae2-aede48c/](https://johnda.web.cern.ch/johnda/HH4b/prod/mvd_run3_30x_c6mvd_c5b_20260928_d3f6ae2-aede48c/) |
| manifest | [roast.json](https://johnda.web.cern.ch/johnda/HH4b/prod/mvd_run3_30x_c6mvd_c5b_20260928_d3f6ae2-aede48c/roasts/mvd_run3_30x_c6mvd_c5b_20260928_d3f6ae2-aede48c/roast.json) |

| archive (EOS) | `root://cmseos.fnal.gov//store/user/jda102/HH4b_prod/mvd_run3_30x_c6mvd_c5b_20260928_d3f6ae2-aede48c/` (list: `xrdfs root://cmseos.fnal.gov ls -R /store/user/jda102/HH4b_prod/mvd_run3_30x_c6mvd_c5b_20260928_d3f6ae2-aede48c`) |

## Steps

| step | host | snakefile | state | log |
|---|---|---|---|---|
| MvD_3_svb | falcon | `coffea4bees/workflows/Snakefile_MvD_3_svb.smk`  | submitted 2026-09-29 | [MvD_3_svb.log](https://johnda.web.cern.ch/johnda/HH4b/prod/mvd_run3_30x_c6mvd_c5b_20260928_d3f6ae2-aede48c/logs/MvD_3_svb.log) |
| MvD | cmslpc | `coffea4bees/workflows/Snakefile_MvD.smk`  | submitted 2026-09-29 | [MvD.log](https://johnda.web.cern.ch/johnda/HH4b/prod/mvd_run3_30x_c6mvd_c5b_20260928_d3f6ae2-aede48c/logs/MvD.log) |

## Pages

| workflow | page |
|---|---|
| SvB_MvD_30x_c5b_c6mvd | [yields](https://johnda.web.cern.ch/johnda/HH4b/prod/mvd_run3_30x_c6mvd_c5b_20260928_d3f6ae2-aede48c/output/SvB_MvD_30x_c5b_c6mvd/yields.html) |
| V4 | [plots_MvD_30x_c6mvd_c5b gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/mvd_run3_30x_c6mvd_c5b_20260928_d3f6ae2-aede48c/output/mvd_run3_30x_c6mvd_c5b_20260928_d3f6ae2-aede48c/V4/plots_MvD_30x_c6mvd_c5b/index.html) |
| V4 | [summary](https://johnda.web.cern.ch/johnda/HH4b/prod/mvd_run3_30x_c6mvd_c5b_20260928_d3f6ae2-aede48c/output/mvd_run3_30x_c6mvd_c5b_20260928_d3f6ae2-aede48c/V4/stat_analysis/summary.html) |

## Notes

c5b SvB against the c6-feature 30x MvD (mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c)

## History

- 2026-09-28 19:16:45 new 
- 2026-09-28 19:17:47 checkout host=cmslpc
- 2026-09-28 19:17:57 checkout host=falcon
- 2026-09-29 00:26:42 submit step=MvD_3_svb host=falcon
- 2026-09-29 00:28:01 submit step=MvD_3_svb host=falcon
- 2026-09-29 02:15:05 hotpatch host=cmslpc files=['roasts/mvd_run3_30x_c6mvd_c5b_20260928_d3f6ae2-aede48c/config.yml (mtime kept)'] what=mvd_pmix4_floor -> 0.01 for the c6 MvD (floor scan on this MvD's V.2c: 0.05 SR/SB 1.043/1.059, 0.01 1.006/1.019, none 1.004/1.015; 0.05 clipped real weight of the sharper c6 MvD). Proposed by Claude overnight, pending John's confirmation.
- 2026-09-29 09:23:03 config-edited sha256=dcd8f231ed53
- 2026-09-29 09:23:03 submit step=MvD host=cmslpc
- 2026-09-29 09:24:46 submit step=MvD host=cmslpc
- 2026-09-29 11:26:52 publish host=cmslpc ok=True
- 2026-09-29 15:40:23 publish host=cmslpc ok=True
- 2026-09-29 15:40:38 publish host=falcon ok=True
- 2026-09-29 15:40:39 archive host=cmslpc ok=True
- 2026-09-29 15:40:42 archive host=falcon ok=True
