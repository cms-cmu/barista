# mixeddata_run3_4bmix_20261001_af69c01-28f390d

*mixeddata_run3_4bmix — roasted by John Alison on 2026-10-01*

| | |
|---|---|
| barista | [`af69c0149c3e`](https://gitlab.cern.ch/cms-cmu/barista/-/commit/af69c0149c3e6f34bddedf4d5e2889b8c05bb6b8) |
| coffea4bees | [`28f390dbac26`](https://gitlab.cern.ch/cms-cmu/coffea4bees/-/commit/28f390dbac266658dacfbcb32655e69adbe54fc9) |
| config | `coffea4bees/workflows/config/mixeddata_run3_4bmix.yml` (sha256 `7153888712c8`) — [captured copy](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_20261001_af69c01-28f390d/roasts/mixeddata_run3_4bmix_20261001_af69c01-28f390d/config.yml) |
| results | [https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_20261001_af69c01-28f390d/](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_20261001_af69c01-28f390d/) |
| manifest | [roast.json](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_20261001_af69c01-28f390d/roasts/mixeddata_run3_4bmix_20261001_af69c01-28f390d/roast.json) |
| production area | `root://cmseos.fnal.gov//store/user/jda102/HH4b_prod/mixeddata_run3_4bmix_20261001_af69c01-28f390d/` |
| archive (EOS) | `root://cmseos.fnal.gov//store/user/jda102/HH4b_prod/mixeddata_run3_4bmix_20261001_af69c01-28f390d/` (list: `xrdfs root://cmseos.fnal.gov ls -R /store/user/jda102/HH4b_prod/mixeddata_run3_4bmix_20261001_af69c01-28f390d`) |

## Steps

| step | host | snakefile | state | log |
|---|---|---|---|---|
| MakeMixedData | cmslpc | `coffea4bees/workflows/Snakefile_MakeMixedData.smk`  | submitted 2026-10-02 (resumed) | [MakeMixedData.log](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_20261001_af69c01-28f390d/logs/MakeMixedData.log) |

## Pages

| workflow | page |
|---|---|
| M6 | [cutflow_validation](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_20261001_af69c01-28f390d/output/mixeddata_run3_4bmix_20261001_af69c01-28f390d/M6/cutflow_validation.html) |
| M6 | [plots gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_20261001_af69c01-28f390d/output/mixeddata_run3_4bmix_20261001_af69c01-28f390d/M6/plots/index.html) |
| M6 | [study gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_20261001_af69c01-28f390d/output/mixeddata_run3_4bmix_20261001_af69c01-28f390d/M6/study/index.html) |
| M7 | [plots_mixeddata gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_20261001_af69c01-28f390d/output/mixeddata_run3_4bmix_20261001_af69c01-28f390d/M7/plots_mixeddata/index.html) |
| M7 | [plots_signal gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_20261001_af69c01-28f390d/output/mixeddata_run3_4bmix_20261001_af69c01-28f390d/M7/plots_signal/index.html) |

## Notes

4b mixing: M.2a 4b pre-skim (builds data__4bskim, shared with the no-boost roast) + boost acceptance retry 2.4, 6 GB workers for all steps; hemilib from mixeddata_run3_20260926_956d4bf-aca8a4e

## History

- 2026-10-01 22:49:19 new 
- 2026-10-01 22:50:12 checkout host=cmslpc
- 2026-10-01 22:50:23 submit step=MakeMixedData host=cmslpc
- 2026-10-01 22:52:29 submit step=MakeMixedData host=cmslpc
- 2026-10-02 14:31:54 hotpatch what=M.7 4b-mode fix (coffea4bees b21663540, git apply of a44f9c895..b21663540 on the checkout): in 4b mixing M.7's mixed-data pass takes no M.3 JCM (unit weights, plots x 1/N), the report runs orig:-:mixed4b (no 3b-origin sample), plots without the 3b curves; M.3 drops out of the 4b DAG. Before: M7_report crashed on synthetic_mc_3b_* and the mixed-data overlay was weighted with a meaningless M.3 JCM fit.
- 2026-10-02 14:32:35 resume step=MakeMixedData host=cmslpc
- 2026-10-02 14:34:41 resume step=MakeMixedData host=cmslpc
- 2026-10-02 16:19:50 publish host=cmslpc ok=True
- 2026-10-02 16:32:53 archive host=cmslpc ok=True
