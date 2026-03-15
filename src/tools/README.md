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
