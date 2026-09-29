# mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c

*mvd_run3_30x_c6mvd — roasted by John Alison on 2026-09-28*

| | |
|---|---|
| barista | [`d3f6ae22cc3b`](https://gitlab.cern.ch/cms-cmu/barista/-/commit/d3f6ae22cc3bc6be56c20b757afee1d884bf047a) |
| coffea4bees | [`aede48c9cf89`](https://gitlab.cern.ch/cms-cmu/coffea4bees/-/commit/aede48c9cf89f9f59c705d3e50071bd49e03cec6) |
| config | `coffea4bees/workflows/config/mvd_run3_30x_c6mvd.yml` (sha256 `2fad86282eea`) — [captured copy](https://johnda.web.cern.ch/johnda/HH4b/prod/mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c/roasts/mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c/config.yml) |
| results | [https://johnda.web.cern.ch/johnda/HH4b/prod/mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c/](https://johnda.web.cern.ch/johnda/HH4b/prod/mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c/) |
| manifest | [roast.json](https://johnda.web.cern.ch/johnda/HH4b/prod/mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c/roasts/mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c/roast.json) |


## Steps

| step | host | snakefile | state | log |
|---|---|---|---|---|
| MvD_2_train | falcon | `coffea4bees/workflows/Snakefile_MvD_2_train.smk`  | submitted 2026-09-28 (resumed) | [MvD_2_train.log](https://johnda.web.cern.ch/johnda/HH4b/prod/mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c/logs/MvD_2_train.log) |
| MvD | cmslpc | `coffea4bees/workflows/Snakefile_MvD.smk`  | submitted 2026-09-29 | [MvD.log](https://johnda.web.cern.ch/johnda/HH4b/prod/mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c/logs/MvD.log) |
| MvD_3_svb | falcon | `coffea4bees/workflows/Snakefile_MvD_3_svb.smk`  | submitted 2026-09-29 (resumed) | [MvD_3_svb.log](https://johnda.web.cern.ch/johnda/HH4b/prod/mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c/logs/MvD_3_svb.log) |

## Pages

| workflow | page |
|---|---|
| V2c | [cutflow_MvD_30x_c6_closure](https://johnda.web.cern.ch/johnda/HH4b/prod/mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c/output/mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c/V2c/cutflow_MvD_30x_c6_closure.html) |
| V2c | [plots_MvD_30x_c6_closure gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c/output/mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c/V2c/plots_MvD_30x_c6_closure/index.html) |
| V4 | [plots_MvD_30x_c6mvd_kin gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c/output/mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c/V4/plots_MvD_30x_c6mvd_kin/index.html) |
| V4 | [summary](https://johnda.web.cern.ch/johnda/HH4b/prod/mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c/output/mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c/V4/stat_analysis/summary.html) |

## Notes

30x MvD with the 16 c6 jet features (MvD >= SvB in features); kinematic SvB against it; c5/c5b/c6 SvB overlays reuse this MvD

## History

- 2026-09-28 19:12:21 new 
- 2026-09-28 19:13:30 checkout host=cmslpc
- 2026-09-28 19:13:43 checkout host=falcon
- 2026-09-28 19:14:02 submit step=MvD_2_train host=falcon
- 2026-09-28 19:14:53 submit step=MvD_2_train host=falcon
- 2026-09-28 22:40:23 config-edited sha256=6917d8a844f0
- 2026-09-28 22:40:23 resume step=MvD_2_train host=falcon
- 2026-09-28 22:40:45 resume step=MvD_2_train host=falcon
- 2026-09-29 00:26:40 submit step=MvD_3_svb host=falcon
- 2026-09-29 00:27:59 submit step=MvD_3_svb host=falcon
- 2026-09-29 00:28:06 submit step=MvD host=cmslpc
- 2026-09-29 00:29:32 submit step=MvD host=cmslpc
- 2026-09-29 00:58:14 hotpatch host=cmslpc files=['roasts/mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c/config.yml (mtime kept)'] what=floor study: mvd_pmix4_floor 0.05 -> 0.01 (cap MvD ~100). The c6 MvD is sharper (57 % of mixed x JCM SB weight at MvD<0.5, 3.2 % >5; kin 30x: 33 % / 1.8 %) and the 0.05 cap clipped real weight: <MvD> 0.954, closure SB 1.059 / SR 1.043. Floored-0.05 V2c outputs kept as *_floor0p05. V2c_config force-rerun.
- 2026-09-29 00:58:26 config-edited sha256=2fad86282eea
- 2026-09-29 00:58:26 submit step=MvD host=cmslpc
- 2026-09-29 01:25:27 hotpatch host=cmslpc files=['roasts/mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c/config.yml (mtime kept)'] what=floor study: 0.01 -> 0 (no floor). 0.01 gave closure SB 1.019 / SR 1.006 (0.05: 1.059 / 1.043). 0.01 outputs kept as *_floor0p01.
- 2026-09-29 01:25:29 config-edited sha256=563983d34b70
- 2026-09-29 01:25:29 submit step=MvD host=cmslpc
- 2026-09-29 02:14:12 resume step=MvD_3_svb host=falcon
- 2026-09-29 02:14:38 resume step=MvD_3_svb host=falcon
- 2026-09-29 02:15:05 hotpatch host=cmslpc files=['roasts/mvd_run3_30x_c6mvd_20260928_d3f6ae2-aede48c/config.yml (mtime kept)'] what=mvd_pmix4_floor -> 0.01 for the c6 MvD (floor scan on this MvD's V.2c: 0.05 SR/SB 1.043/1.059, 0.01 1.006/1.019, none 1.004/1.015; 0.05 clipped real weight of the sharper c6 MvD). Proposed by Claude overnight, pending John's confirmation.
- 2026-09-29 02:15:10 config-edited sha256=2fad86282eea
- 2026-09-29 02:15:10 submit step=MvD host=cmslpc
- 2026-09-29 06:22:40 publish host=cmslpc ok=True
- 2026-09-29 07:30:10 submit step=MvD host=cmslpc
- 2026-09-29 07:30:34 submit step=MvD host=cmslpc
- 2026-09-29 07:57:06 publish host=cmslpc ok=True
