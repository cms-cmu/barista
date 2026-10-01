# mixeddata_run3_4bmix_20260930_2e4bcf8-2f1d90b

*mixeddata_run3_4bmix — roasted by John Alison on 2026-09-30*

| | |
|---|---|
| barista | [`2e4bcf8fe1e2`](https://gitlab.cern.ch/cms-cmu/barista/-/commit/2e4bcf8fe1e2e2f9f174096bef56028382861e96) |
| coffea4bees | [`2f1d90b45cfa`](https://gitlab.cern.ch/cms-cmu/coffea4bees/-/commit/2f1d90b45cfa0ce0728d6695a5ef96de1a7bf27e) |
| config | `coffea4bees/workflows/config/mixeddata_run3_4bmix.yml` (sha256 `4a1910666d9d`) — [captured copy](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_20260930_2e4bcf8-2f1d90b/roasts/mixeddata_run3_4bmix_20260930_2e4bcf8-2f1d90b/config.yml) |
| results | [https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_20260930_2e4bcf8-2f1d90b/](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_20260930_2e4bcf8-2f1d90b/) |
| manifest | [roast.json](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_20260930_2e4bcf8-2f1d90b/roasts/mixeddata_run3_4bmix_20260930_2e4bcf8-2f1d90b/roast.json) |
| production area | `root://cmseos.fnal.gov//store/user/jda102/HH4b_prod/mixeddata_run3_4bmix_20260930_2e4bcf8-2f1d90b/` |
| archive (EOS) | `root://cmseos.fnal.gov//store/user/jda102/HH4b_prod/mixeddata_run3_4bmix_20260930_2e4bcf8-2f1d90b/` (list: `xrdfs root://cmseos.fnal.gov ls -R /store/user/jda102/HH4b_prod/mixeddata_run3_4bmix_20260930_2e4bcf8-2f1d90b`) |

## Steps

| step | host | snakefile | state | log |
|---|---|---|---|---|
| MakeMixedData | cmslpc | `coffea4bees/workflows/Snakefile_MakeMixedData.smk`  | submitted 2026-09-30 | [MakeMixedData.log](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_20260930_2e4bcf8-2f1d90b/logs/MakeMixedData.log) |

## Pages

| workflow | page |
|---|---|
| M6 | [cutflow_validation](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_20260930_2e4bcf8-2f1d90b/output/mixeddata_run3_4bmix_20260930_2e4bcf8-2f1d90b/M6/cutflow_validation.html) |
| M6 | [plots gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_20260930_2e4bcf8-2f1d90b/output/mixeddata_run3_4bmix_20260930_2e4bcf8-2f1d90b/M6/plots/index.html) |
| M6 | [study gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_20260930_2e4bcf8-2f1d90b/output/mixeddata_run3_4bmix_20260930_2e4bcf8-2f1d90b/M6/study/index.html) |
| M7 | [M7 gallery](https://johnda.web.cern.ch/johnda/HH4b/prod/mixeddata_run3_4bmix_20260930_2e4bcf8-2f1d90b/output/mixeddata_run3_4bmix_20260930_2e4bcf8-2f1d90b/M7/index.html) |

## Notes

4b-mixing variant (rerun with condor restored, c4b 2f1d90b45): 4b data mixed (own hemis vetoed), 16 seeds random rank top-10, library pinned from mixeddata_run3_20260926_956d4bf-aca8a4e

## History

- 2026-09-30 20:45:45 new 
- 2026-09-30 20:47:14 checkout host=cmslpc
- 2026-09-30 20:47:24 submit step=MakeMixedData host=cmslpc
- 2026-09-30 20:48:52 submit step=MakeMixedData host=cmslpc
- 2026-10-01 07:35:00 publish host=cmslpc ok=True
- 2026-10-01 08:37:30 archive host=cmslpc ok=True
