# declustered_run2_20260928_a4d420a-cb9b5c3

*declustered_run2 — roasted by John Alison on 2026-09-28*

| | |
|---|---|
| barista | [`a4d420a4bce1`](https://gitlab.cern.ch/cms-cmu/barista/-/commit/a4d420a4bce14448708d59dc8ada8313a2d2ebc6) |
| coffea4bees | [`cb9b5c311a08`](https://gitlab.cern.ch/cms-cmu/coffea4bees/-/commit/cb9b5c311a082b5baf42991a2c2f66425e26e697) |
| config | `coffea4bees/workflows/config/declustered_run2.yml` (sha256 `197b95945acf`) — [captured copy](https://johnda.web.cern.ch/johnda/HH4b/prod/declustered_run2_20260928_a4d420a-cb9b5c3/roasts/declustered_run2_20260928_a4d420a-cb9b5c3/config.yml) |
| results | [https://johnda.web.cern.ch/johnda/HH4b/prod/declustered_run2_20260928_a4d420a-cb9b5c3/](https://johnda.web.cern.ch/johnda/HH4b/prod/declustered_run2_20260928_a4d420a-cb9b5c3/) |
| manifest | [roast.json](https://johnda.web.cern.ch/johnda/HH4b/prod/declustered_run2_20260928_a4d420a-cb9b5c3/roasts/declustered_run2_20260928_a4d420a-cb9b5c3/roast.json) |

| archive (EOS) | `root://cmseos.fnal.gov//store/user/jda102/HH4b_prod/declustered_run2_20260928_a4d420a-cb9b5c3/` (list: `xrdfs root://cmseos.fnal.gov ls -R /store/user/jda102/HH4b_prod/declustered_run2_20260928_a4d420a-cb9b5c3`) |

## Steps

| step | host | snakefile | state | log |
|---|---|---|---|---|
| DeClustered | cmslpc | `coffea4bees/workflows/Snakefile_DeClustered.smk`  | submitted 2026-09-28 | [DeClustered.log](https://johnda.web.cern.ch/johnda/HH4b/prod/declustered_run2_20260928_a4d420a-cb9b5c3/logs/DeClustered.log) |

## Pages

| workflow | page |
|---|---|
| D2 | [D2 gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/declustered_run2_20260928_a4d420a-cb9b5c3/output/declustered_run2_20260928_a4d420a-cb9b5c3/D2/index.html) |
| D5 | [cutflow_monitoring](https://johnda.web.cern.ch/johnda/HH4b/prod/declustered_run2_20260928_a4d420a-cb9b5c3/output/declustered_run2_20260928_a4d420a-cb9b5c3/D5/cutflow_monitoring.html) |
| D5 | [plots gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/declustered_run2_20260928_a4d420a-cb9b5c3/output/declustered_run2_20260928_a4d420a-cb9b5c3/D5/plots/index.html) |

## Notes

Run 2 DeClustered dataset, 1 seed, ttbar subtracted (FvT d4_to_t4): synthetic_data_multijet + synthetic_data_4b (multijet + ttbar_PSData from mixeddata_run2_20260927_c39c264-0ce4d59). D.1-D.5. FvT + B.1 hists/config from nominal_run2_20260919_a180b1a-a16559c (blinded: no data SR).

## History

- 2026-09-28 07:43:21 new 
- 2026-09-28 07:47:59 checkout host=cmslpc
- 2026-09-28 07:48:00 submit step=DeClustered host=cmslpc
- 2026-09-28 07:54:34 submit step=DeClustered host=cmslpc
- 2026-09-28 15:13:10 publish host=cmslpc ok=True
- 2026-09-28 15:13:12 archive host=cmslpc ok=True
