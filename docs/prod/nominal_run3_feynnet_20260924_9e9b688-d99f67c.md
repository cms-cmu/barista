# nominal_run3_feynnet_20260924_9e9b688-d99f67c

*nominal_run3_feynnet — roasted by John Alison on 2026-09-24*

| | |
|---|---|
| barista | [`9e9b6882cc98`](https://gitlab.cern.ch/cms-cmu/barista/-/commit/9e9b6882cc98caac69d7883346719e46367feb4f) |
| coffea4bees | [`d99f67cbe954`](https://gitlab.cern.ch/cms-cmu/coffea4bees/-/commit/d99f67cbe95403af4ee809ec8da1321ec31a4a38) |
| config | `coffea4bees/workflows/config/nominal_run3_feynnet.yml` (sha256 `0d10f0c774d8`) — [captured copy](https://johnda.web.cern.ch/johnda/HH4b/prod/nominal_run3_feynnet_20260924_9e9b688-d99f67c/roasts/nominal_run3_feynnet_20260924_9e9b688-d99f67c/config.yml) |
| results | [https://johnda.web.cern.ch/johnda/HH4b/prod/nominal_run3_feynnet_20260924_9e9b688-d99f67c/](https://johnda.web.cern.ch/johnda/HH4b/prod/nominal_run3_feynnet_20260924_9e9b688-d99f67c/) |
| manifest | [roast.json](https://johnda.web.cern.ch/johnda/HH4b/prod/nominal_run3_feynnet_20260924_9e9b688-d99f67c/roasts/nominal_run3_feynnet_20260924_9e9b688-d99f67c/roast.json) |
| archive (EOS) | `root://cmseos.fnal.gov//store/user/jda102/HH4b_prod/nominal_run3_feynnet_20260924_9e9b688-d99f67c/` (list: `xrdfs root://cmseos.fnal.gov ls -R /store/user/jda102/HH4b_prod/nominal_run3_feynnet_20260924_9e9b688-d99f67c`) |

## Steps

| step | host | snakefile | state | log |
|---|---|---|---|---|
| F | cmslpc | `coffea4bees/workflows/Snakefile_PhaseF.smk`  | submitted 2026-09-24 | [F.log](https://johnda.web.cern.ch/johnda/HH4b/prod/nominal_run3_feynnet_20260924_9e9b688-d99f67c/logs/F.log) |

## Pages

| workflow | page |
|---|---|
| Run3_nominal_feynnet | [cutflow_Run3_nominal_feynnet](https://johnda.web.cern.ch/johnda/HH4b/prod/nominal_run3_feynnet_20260924_9e9b688-d99f67c/output/Run3_nominal_feynnet/cutflow_Run3_nominal_feynnet.html) |
| Run3_nominal_feynnet | [cutflow_crosscheck_Run3_nominal_feynnet](https://johnda.web.cern.ch/johnda/HH4b/prod/nominal_run3_feynnet_20260924_9e9b688-d99f67c/output/Run3_nominal_feynnet/cutflow_crosscheck_Run3_nominal_feynnet.html) |
| plots_Run3_nominal_feynnet | [plots_Run3_nominal_feynnet gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/nominal_run3_feynnet_20260924_9e9b688-d99f67c/output/Run3_nominal_feynnet/plots_Run3_nominal_feynnet/index.html) |
| stat_analysis | [summary](https://johnda.web.cern.ch/johnda/HH4b/prod/nominal_run3_feynnet_20260924_9e9b688-d99f67c/output/Run3_nominal_feynnet/stat_analysis/summary.html) |

## Notes

Phase F only: adds SvB_FeynNet.p_ggHH_vs_bkg_fine (240 bins) channel; HH4b and 50-bin HH4b_FeynNet must reproduce nominal_run3_feynnet_20260923_d54d644-c24101b (3.70/0.80, 5.08/0.39). Upstream pinned to nominal_run3_20260922_3f9e199-1e0504f

## History

- 2026-09-24 08:50:58 new 
- 2026-09-24 08:53:58 checkout host=cmslpc
- 2026-09-24 08:54:09 submit step=F host=cmslpc
- 2026-09-24 09:05:43 submit step=F host=cmslpc
- 2026-09-24 10:15:08 publish host=cmslpc ok=False
- 2026-09-24 10:19:49 publish host=cmslpc ok=True
- 2026-09-24 10:25:39 archive host=cmslpc ok=True
