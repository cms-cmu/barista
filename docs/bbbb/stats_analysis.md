#  statistical analysis

This part of the analysis uses [coffea](https://coffeateam.github.io/coffea/), [ROOT](https://root.cern/) and the cms package [combine](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/). Therefore we need different environments to run the different steps.

We need two sets of environemnts. One with `coffea` to take the outputs of analysis and convert them into a json format and another with `root` and `combine`. For the first one, you can use the `` container. For the second follow the next instructions.

## Install Combine and CombineHarvester

### With container

We can use the combine/combineHarvester container. For that you can run:
```
cd /python/  ### if you are not there
./../shell_combine combine --help
```

The script `./shell_combine` runs your command inside a container, for simplicity.

## Convert hist to json

Using the  container:
```
cd python/stats_analysis/
python convert_hist_to_json.py -o histos/histAll.json -i ../analysis/hists/histAll.coffea
```

## Convert json to root (for combine)

With the combine container:
```
cd python/
../shell_combine python3 stats_analysis/convert_json_to_root.py --classifier SvB_MA SvB -f histos/histAll.json --merge2016 --output_dir stats_analysis/datacards/ --plot
```

## How to run make_variable_binning:

Using the combine container:
```
cd python/
../shell_combine python3 stats_analysis/make_variable_binning.py -i hists/test_/histAll.json -t 10 -o stats_analysis/tmp/histAll_rebinned.root
```
`-i` can take json or root files. The output is a root file.


## How to run for `m4j`
Note: the location of shell_combine may need to be modified in the command
```
for var_type in m4j_zz m4j_zh m4j_hh; do
    ./shell_combine python3 stats_analysis/runTwoStageClosure.py  --var $var_type  \
        --classifier SvB  --m4j_xmin 230  --m4j_xmax 1200 \
        --rebin 16 --outputPath stats_analysis/sysana/histsLocalResonance \
        --input_file_data3b analysis/hists/sysana/histMixedBkg_data_3b_for_mixed_kfold.root \
        --input_file_TT     analysis/hists/sysana/histMixedBkg_TT.root \
        --input_file_mix    analysis/hists/sysana/histMixedData.root \
        --input_file_sig    analysis/hists/sysana/histSignal_UL.root \
        --use_kfold   
```