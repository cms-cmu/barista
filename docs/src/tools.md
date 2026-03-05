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

## Replicate files to CMU EOS (`cmsdata`)

The script `src/tools/replicate_to_cmsdata.py` copies HH4b picoAOD and classifier
files from FNAL EOS personal user areas to the CMU EOS group area at
`root://cmsdata.phys.cmu.edu//store/group/HH4b/`.

Two sources of files are combined:

1. picoAOD / friend-tree files explicitly listed in the dataset YML files
2. `FvT*.root` / `SvB*.root` companion files discovered by scanning the same source directories on FNAL EOS

All paths are flattened: user-specific prefixes are stripped and files are placed
under the era-specific group area (`Run2/` or `Run3/`).

### Usage

**Generate commands file and SLURM script, then submit:**
```bash
python src/tools/replicate_to_cmsdata.py --era Run2 --submit --proxy proxy/x509_proxy
python src/tools/replicate_to_cmsdata.py --era Run3 --submit --proxy proxy/x509_proxy
```

**Force-overwrite existing files** (e.g. to fix truncated copies):
```bash
python src/tools/replicate_to_cmsdata.py --era Run2 --submit --force --proxy proxy/x509_proxy
```

**Skip companion file scan** (copy only files listed in YMLs):
```bash
python src/tools/replicate_to_cmsdata.py --era Run2 --no-companions --submit --proxy proxy/x509_proxy
```

**Dry run** (no files written, no job submitted):
```bash
python src/tools/replicate_to_cmsdata.py --era Run2 --dry-run
```

### Command-line Options

| Option | Description |
|--------|-------------|
| `--era` | **(Required)** Era to replicate: `Run2` or `Run3` |
| `--submit` | Submit a SLURM array job after generating the commands file |
| `--proxy` | Path to VOMS proxy file relative to the submit directory (default: `proxy/x509_proxy`) |
| `--force` | Pass `--force` to `xrdcp` to overwrite existing destination files |
| `--no-companions` | Skip scanning for `FvT`/`SvB` companion files |
| `--output-dir` | Directory for commands file, SLURM script, and logs (default: `slurm_replication/<era>/`) |
| `--workers` | Parallel workers for companion file directory scanning (default: 16) |
| `--batch-size` | Files per SLURM array task (default: 49) |
| `--dry-run` | Print summary without writing any files or submitting jobs |

### Output

All output is written to `slurm_replication/<era>/`:

- `commands.txt` — tab-separated `src dest` pairs for all files to copy
- `submit.sh` — generated SLURM array job script
- `logs/` — per-task stdout/stderr and `failures_<taskid>.log` for failed copies

### Storage remap for running from CMU EOS

After replication, use the `--storage-remap` flag in `runner.py` to redirect file
paths at load time without modifying the YML files. Remap configs live in
`coffea4bees/metadata/`:

```bash
# Run2 analysis from CMU EOS
python runner.py ... --storage-remap coffea4bees/metadata/storage_remaps_run2_cmu.yml

# Run3 analysis from CMU EOS
python runner.py ... --storage-remap coffea4bees/metadata/storage_remaps_run3_cmu.yml
```

### Source prefix mapping

| Era | FNAL EOS source prefix | CMU EOS destination |
|-----|------------------------|---------------------|
| Run2 | `.../jda102/condor/ZH4b/ULTrig/` | `Run2/` |
| Run2 | `.../algomez/XX4b/` | `Run2/` |
| Run2 | `.../jda102/XX4b/` | `Run2/` |
| Run2 | `.../smurthy/XX4b/` | `Run2/` |
| Run3 | `.../jda102/XX4b/` | `Run3/` |
| Run3 | `.../smurthy/XX4b/` | `Run3/` |

## Convert plot YAML to HEPData submission

The script `src/plotting/yaml_to_hepdata.py` converts barista plot YAML files (the ones produced by the plotting code) into the [HEPData](https://www.hepdata.net/) submission format. It uses `hepdata_lib` and runs in a dedicated pixi environment.

### Setup (first time only)

The hepdata tools are managed via a pixi environment defined in `software/pixi/pixi.toml`. The environment is installed automatically on first use:

```bash
pixi run -e hepdata python --version   # triggers install if needed
```

### Usage

**Single file** (output goes to `plotsForPaper/hepdata_submission/` in the current directory):
```bash
pixi run -e hepdata python src/plotting/yaml_to_hepdata.py \
    coffea4bees/archive/plotsForPaper/SvB_MA_postfitplots_fit_b_138.yaml
```

**Multiple files** (all tables go into the same submission):
```bash
pixi run -e hepdata python src/plotting/yaml_to_hepdata.py \
    coffea4bees/archive/plotsForPaper/SvB_MA_postfitplots_fit_b_138.yaml \
    coffea4bees/archive/plotsForPaper/SvB_MA_postfitplots_prefit_138.yaml
```

**Custom output directory**:
```bash
pixi run -e hepdata python src/plotting/yaml_to_hepdata.py \
    -o my_hepdata_output/ \
    coffea4bees/archive/plotsForPaper/SvB_MA_postfitplots_fit_b_138.yaml
```

### Validation

Validate the submission using `hepdata-validate` (also available in the pixi environment):

```bash
pixi run -e hepdata hepdata-validate -d plotsForPaper/hepdata_submission/
```

### Creating a tar file for upload

HEPData expects a `.tar.gz` archive for upload. After generating and validating the submission:

```bash
tar czf hepdata_submission.tar.gz plotsForPaper/hepdata_submission/
```

### Command-line Options

| Option | Description |
|--------|-------------|
| `input_files` | One or more barista plot YAML files to convert (positional) |
| `-o, --output-dir` | Output directory. Default: `plotsForPaper/hepdata_submission/` in the current working directory |
| `--comment` | Description for the HEPData submission |
| `--reaction` | Reaction string for HEPData keywords (default: `P P --> H H --> B B B B`) |
| `--cmenergies` | Center-of-mass energy in TeV (default: `13.0`) |

### Output

The script produces a `plotsForPaper/hepdata_submission/` directory containing:

- `submission.yaml` — HEPData submission manifest with all table entries
- One `.yaml` data file per input file (same name as the input), containing the binned data, backgrounds, and data/prediction ratios with statistical uncertainties
```