#  skimmer

To run the skimmer, remember first to set the coffea environment and your grid certificate. If you followed the instructions in the [README.md](../../README.md), the `set_shell.sh` file must be located right after the package, and then:
```
voms-proxy-init -rfc -voms cms --valid 168:00
cd / ## Or go to this directory
source set_shell.sh
```

## In this folder

Here you find the code to create `picoAODs` (skims from nanoAOD)

Each folder contains:
 - [metadata](./metadata/): yml files to run the processors
 - [processors](./processors/): python files with the processors for each skimms.

Then, the run-all script is called `runner.py` and it is one directory below (in [python/](../../python/)). This script will run local or condor depending on the flag used. To learn all the options of the script, just run:
```
# (inside //python/)
python runner.py --help
```

## Run Analysis

### Example to run the analysis

For example, to run a processor you can do:
```
#  (inside //python/)
python runner.py -s -p skimmer/processor/skimmer_4b.py -c skimmer/metadata/HH4b.yml -y UL18 -d TTTo2L2Nu -t
```

The output file of this process will be located under `python/skimmer/test/`.


### Check that all events have been processed

Compares processed lumi sections against those expected in json
```
python python/skimmer/metadata/get_das_info.py -d python/metadata/datasets_HH4b_Run3.yml 
python skimmer/metadata/check_event_counts.py -y skimmer/metadata/picoaod_datasets_data_2023_BPix.yml
```

Compares processed lumi sections against those expected in json
```
python skimmer/metadata/check_lumi_sections.py -j base_class/data/goldenJSON/Cert_Collisions2023_366442_370790_Golden.json -y skimmer/metadata/picoaod_datasets_data_2023_BPix.yml
```


### Add skims to dataset 

Add output of skims to input data sets

```
python src/scripts/merge_yaml_datasets.py -m metadata/datasets_HH4b_Run3.yml -o metadata/datasets_HH4b_Run3_merged.yml -f metadata/archive/skims_Run3_2024_v2/picoaod_datasets_data_202*
```