# Tools in Barista

## How to compute integrated luminosity for data

The script `src/tools/compute_lumi_processes.py` reads a YAML file that is the output of the skimming steps, containing `lumis_processed` data (run numbers and lumi sections), converts it to the JSON format required by [brilcalc](https://cms-opendata-workshop.github.io/workshop-lesson-luminosity/03-calculating-luminosity/index.html), and automatically runs brilcalc to compute the integrated luminosity.

### Usage

**Basic usage** (prints luminosity to console):
```bash
./run_container brilcalc python src/tools/compute_lumi_processes.py -i <input.yml>
```

**Save output to file**:
```bash
./run_container brilcalc python src/tools/compute_lumi_processes.py -i <input.yml> -o <output.txt>
```

### Command-line Options

- `-i, --input`: **(Required)** Input YAML file with lumis_processed data
- `-o, --output`: **(Optional)** File to save brilcalc output. If not specified, output is printed to console

## Check that all events have been processed

Compares processed lumi sections against those expected in json

```
python src/tools/get_das_info.py -d coffea4bees/metadata/datasets_HH4b_Run3.yml 
python src/tools/check_event_counts.py -y skimmer/metadata/picoaod_datasets_data_2023_BPix.yml
```

Compares processed lumi sections against those expected in json

```
python src/tools/check_lumi_sections.py -j src/data/goldenJSON/Cert_Collisions2023_366442_370790_Golden.json -y skimmer/metadata/picoaod_datasets_data_2023_BPix.yml
```

## Add skims to dataset 

Add output of skims to input data sets

```
python src/tools/merge_yaml_datasets.py -m metadata/datasets_HH4b_Run3.yml -o metadata/datasets_HH4b_Run3_merged.yml -f metadata/archive/skims_Run3_2024_v2/picoaod_datasets_data_202*
```