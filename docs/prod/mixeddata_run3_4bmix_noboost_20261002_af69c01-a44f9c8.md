# mixeddata_run3_4bmix_noboost_20261002_af69c01-a44f9c8

*mixeddata_run3_4bmix_noboost — roasted by John Alison on 2026-10-02*

| | |
|---|---|
| barista | [`af69c0149c3e`](https://gitlab.cern.ch/cms-cmu/barista/-/commit/af69c0149c3e6f34bddedf4d5e2889b8c05bb6b8) |
| coffea4bees | [`a44f9c8950b2`](https://gitlab.cern.ch/cms-cmu/coffea4bees/-/commit/a44f9c8950b2010cf4eb12c60c3517eb2bab5df3) |
| config | `coffea4bees/workflows/config/mixeddata_run3_4bmix_noboost.yml` (sha256 `941d97c9ca85`) — [captured copy](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_noboost_20261002_af69c01-a44f9c8/roasts/mixeddata_run3_4bmix_noboost_20261002_af69c01-a44f9c8/config.yml) |
| results | [https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_noboost_20261002_af69c01-a44f9c8/](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_noboost_20261002_af69c01-a44f9c8/) |
| manifest | [roast.json](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_noboost_20261002_af69c01-a44f9c8/roasts/mixeddata_run3_4bmix_noboost_20261002_af69c01-a44f9c8/roast.json) |
| production area | `root://cmseos.fnal.gov//store/user/jda102/HH4b_prod/mixeddata_run3_4bmix_noboost_20261002_af69c01-a44f9c8/` |
| archive (EOS) | `root://cmseos.fnal.gov//store/user/jda102/HH4b_prod/mixeddata_run3_4bmix_noboost_20261002_af69c01-a44f9c8/` (list: `xrdfs root://cmseos.fnal.gov ls -R /store/user/jda102/HH4b_prod/mixeddata_run3_4bmix_noboost_20261002_af69c01-a44f9c8`) |

## Steps

| step | host | snakefile | state | log |
|---|---|---|---|---|
| MakeMixedData | cmslpc | `coffea4bees/workflows/Snakefile_MakeMixedData.smk`  | submitted 2026-10-02 (resumed) | [MakeMixedData.log](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_noboost_20261002_af69c01-a44f9c8/logs/MakeMixedData.log) |

## Pages

| workflow | page |
|---|---|
| M6 | [cutflow_validation](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_noboost_20261002_af69c01-a44f9c8/output/mixeddata_run3_4bmix_noboost_20261002_af69c01-a44f9c8/M6/cutflow_validation.html) |
| M6 | [plots gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_noboost_20261002_af69c01-a44f9c8/output/mixeddata_run3_4bmix_noboost_20261002_af69c01-a44f9c8/M6/plots/index.html) |
| M6 | [study gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_noboost_20261002_af69c01-a44f9c8/output/mixeddata_run3_4bmix_noboost_20261002_af69c01-a44f9c8/M6/study/index.html) |
| M7 | [plots_mixeddata gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_noboost_20261002_af69c01-a44f9c8/output/mixeddata_run3_4bmix_noboost_20261002_af69c01-a44f9c8/M7/plots_mixeddata/index.html) |
| M7 | [plots_signal gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_noboost_20261002_af69c01-a44f9c8/output/mixeddata_run3_4bmix_noboost_20261002_af69c01-a44f9c8/M7/plots_signal/index.html) |

## Notes

4b mixing, NO z boost (4D match incl. pz); reads the 4b skim of mixeddata_run3_4bmix_20261001_af69c01-28f390d and the 3b roast's hemilib; A/B partner of …28f390d

## History

- 2026-10-02 00:38:53 new 
- 2026-10-02 00:39:56 checkout host=cmslpc
- 2026-10-02 00:40:30 repinned from=jda102@cmslpc364.fnal.gov to=jda102@cmslpc335.fnal.gov why=checkout landed on cmslpc364 next to the running mixeddata_run3_4bmix_20261001_af69c01-28f390d; two MakeMixedData roasts on one interactive node OOM'd cmslpc307 on 2026-10-01
- 2026-10-02 00:40:31 submit step=MakeMixedData host=cmslpc
- 2026-10-02 00:42:01 submit step=MakeMixedData host=cmslpc
- 2026-10-02 09:07:41 resume step=MakeMixedData host=cmslpc
- 2026-10-02 09:07:59 resume step=MakeMixedData host=cmslpc
- 2026-10-02 14:32:06 hotpatch what=M.7 4b-mode fix (coffea4bees b21663540, git apply of a44f9c895..b21663540 on the checkout): in 4b mixing M.7's mixed-data pass takes no M.3 JCM (unit weights, plots x 1/N), the report runs orig:-:mixed4b (no 3b-origin sample), plots without the 3b curves; M.3 drops out of the 4b DAG. Before: M7_report crashed on synthetic_mc_3b_* and the mixed-data overlay was weighted with a meaningless M.3 JCM fit.
- 2026-10-02 14:33:36 resume step=MakeMixedData host=cmslpc
- 2026-10-02 14:34:20 resume step=MakeMixedData host=cmslpc
- 2026-10-02 14:34:43 resume step=MakeMixedData host=cmslpc
- 2026-10-02 18:21:44 publish host=cmslpc ok=True
- 2026-10-02 18:21:46 archive host=cmslpc ok=True
