# src/tools

Standalone utility scripts for dataset management, file replication, and analysis bookkeeping. Run inside the analysis container unless noted.

## make_dataset_yml.py

Converts a `picoaod_datasets` output file (produced by `runner.py -o`) into a datasets YAML suitable for use with `runner.py -m`.

Works for any picoaod_datasets output: mixeddata_all, JetDeClustered, 4b skims, etc.

```bash
python src/tools/make_dataset_yml.py \
    -i output/mixeddata_make_dataset_Run3_all/picoaod_datasets_mixeddata_Run3_noTT_pz.yml \
    -o coffea4bees/metadata/datasets_HH4b_Run3/mixeddata_all.yml \
    -n mixeddata_all_noTT
```

| Argument          | Description                                            |
|-------------------|--------------------------------------------------------|
| `-i` / `--input`  | Input `picoaod_datasets` yml from runner.py output     |
| `-o` / `--output` | Output datasets yml to write                           |
| `-n` / `--name`   | Top-level dataset name (default: `mixeddata_all_noTT`) |

The script parses dataset keys like `data_2022_EEE` or `JetDeClustered_2023_BPixD1` into `year` / `era` components, extracts the `files:` list (ignoring `bad_files:`), and writes the nested `{name} → {year} → picoAOD → {era} → files` structure. Supported year prefixes: `2022_EE`, `2022_preEE`, `2023_BPix`, `2023_preBPix`, `UL16_preVFP`, `UL16_postVFP`, `UL17`, `UL18`.

## convert_coffea_to_json.py

Converts coffea `.coffea` histogram files to a nested JSON format suitable for `make_combine_inputs.py` or general inspection. Works generically for any analysis — axis names and category values are discovered automatically.

- **StrCategory axes** (e.g. `process`, `year`, `channel`, `flavor`) iterate naturally as strings.
- **IntCategory axes** (e.g. `tag`, `region` if stored as integers) can be mapped to human-readable string keys via `--mapping-config`.
- **Boolean axes** (e.g. `passPreSel`) are summed over by default, or fixed to `True`/`False` via `--select`.

```bash
# bbreww — StrCategory axes, no mapping needed
python src/tools/convert_coffea_to_json.py \
    -i bbreww/analysis/hists/histAll.coffea \
    -o bbreww/stats_analysis/histos/histAll.json \
    --histos SvB.phh

# coffea4bees — fix Boolean axis passPreSel to True
python src/tools/convert_coffea_to_json.py \
    -i coffea4bees/analysis/hists/histAll.coffea \
    -o coffea4bees/stats_analysis/histos/histAll.json \
    --histos SvB_MA.ps_hh_fine \
    --select passPreSel=True

# Use only SR
python src/tools/convert_coffea_to_json.py \
    -i analysis/hists/histAll.coffea -o histos/histAll.json \
    --select region=SR

# Merge SR+CR into one histogram (sum over region axis)
python src/tools/convert_coffea_to_json.py \
    -i analysis/hists/histAll.coffea -o histos/histAll.json \
    --sum region

# Both: fix passPreSel=True and merge regions
python src/tools/convert_coffea_to_json.py \
    -i analysis/hists/histAll.coffea -o histos/histAll.json \
    --select passPreSel=True --sum region

# IntCategory axes with integer bin values — provide a mapping file
python src/tools/convert_coffea_to_json.py \
    -i analysis/hists/histAll.coffea \
    -o histos/histAll.json \
    --mapping-config my_mapping.json
```

Example `--mapping-config` JSON:
```json
{
    "tag":    {"0": "threeTag", "1": "fourTag", "2": "other"},
    "region": {"0": "SR",       "1": "SB",      "2": "other"}
}
```

| Argument | Description |
|---|---|
| `-i` / `--input_file` | Input `.coffea` file |
| `-o` / `--output` | Output JSON file |
| `--histos` | Histogram names to convert (default: all) |
| `--select AXIS=VALUE` | Fix an axis to one value, e.g. `region=SR` or `passPreSel=True` |
| `--sum AXIS` | Sum over all values of an axis, collapsing it (e.g. `--sum region` merges SR+CR) |
| `--mapping-config` | JSON file mapping `IntCategory` int values to string keys |
| `-v` / `--verbose` | Debug-level logging |

`--select` and `--sum` are mutually exclusive per axis. Boolean axes are always summed unless listed in `--select`.

## convert_json_to_root.py

Converts a JSON histogram file (produced by `convert_coffea_to_json.py` or similar) into a ROOT file of `TH1F` histograms. Supports arbitrarily deep JSON nesting, uniform and variable rebinning, and optional ROOT subdirectory organisation. The `json_to_TH1()` function is also imported directly by `make_combine_inputs.py` in analysis packages.

```bash
# Uniform rebin by factor 5, flat ROOT file
python src/tools/convert_json_to_root.py \
    -f histos/histAll.json -o output/histAll.root --rebin 5

# Variable binning (provide bin edges)
python src/tools/convert_json_to_root.py \
    -f histos/histAll.json -o output/histAll.root \
    --rebin-edges 0.0 0.2 0.4 0.6 0.8 1.0

# Write nesting levels as ROOT subdirectories
python src/tools/convert_json_to_root.py \
    -f histos/histAll.json -o output/histAll.root --dirs

# Convert only selected top-level keys
python src/tools/convert_json_to_root.py \
    -f histos/histAll.json -o output/histAll.root \
    --histos SvB.phh SvB.ptt
```

| Argument | Description |
|---|---|
| `-f` / `--file` | Input JSON file |
| `-o` / `--output` | Output ROOT file (overwritten if exists) |
| `--histos` | Top-level keys to convert (default: all) |
| `-r` / `--rebin` | Uniform rebin factor (default: 1) |
| `--rebin-edges` | Variable bin edges (overrides `--rebin`) |
| `--dirs` | Write nesting levels as ROOT subdirectories |
| `-v` / `--verbose` | Debug-level logging |

## merge_yaml_datasets.py

Merges picoAOD file lists from one or more datasets YMLs into a main datasets file.

```bash
python src/tools/merge_yaml_datasets.py \
    -m coffea4bees/metadata/datasets_HH4b.yml \
    -f output/picoaod_datasets_UL18.yml \
    -o coffea4bees/metadata/datasets_HH4b_merged.yml
```

## merge_coffea_files.py

Merges multiple `.coffea` histogram output files into one.

```bash
python src/tools/merge_coffea_files.py -i file1.coffea file2.coffea -o merged.coffea
```

## check_event_counts.py

Compares event counts between a datasets YAML and processed output to check for missing or duplicate processing.

## check_lumi_sections.py

Checks luminosity sections in a golden JSON against the lumi sections present in processed data files.

## compute_lumi_processes.py

Converts a YAML lumi file to JSON and runs `brilcalc` to compute integrated luminosity. Run inside the `brilcalc` container.

## get_das_info.py

Queries DAS (`dasgoclient`) for dataset summary information and writes results to JSON.

## replicate_to_cmsdata.py

Replicates picoAOD and classifier (FvT/SvB) files from FNAL EOS to CMU EOS via SLURM array jobs.

```bash
# Generate commands file only
python src/tools/replicate_to_cmsdata.py --era Run2

# Generate and submit SLURM array job
python src/tools/replicate_to_cmsdata.py --era Run2 --submit --proxy proxy/x509_proxy
```

## compute_combine_limits.py

Helper functions for combining expected limits from multiple sources.

## condor_monitor.py

Monitors HTCondor jobs in the terminal, showing the last line of stdout for each running job (via `condor_tail`). Re-queries the queue every 10 seconds and exits when all jobs are done.

```bash
python src/tools/condor_monitor.py              # monitor all your jobs
python src/tools/condor_monitor.py HH4b         # filter by batch name / args
python src/tools/condor_monitor.py 2294883      # filter by cluster ID
```

**Display columns:** `<cluster.proc>  <status>  <batch name>  <last stdout line>`

Status codes: `I`dle, `R`unning, `C`omplete, `H`eld, `X` Removed, `T` Transferring, `S` Suspended.

The optional grep argument is matched against the job's `JobBatchName`, `Arguments`, and `ClusterId`. Up to 16 `condor_tail` fetches run concurrently. No external dependencies — runs directly on the host (does not require the analysis container).
