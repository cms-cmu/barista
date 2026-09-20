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
    -i bbreww/output/histAll.coffea \
    -o bbreww/stats_analysis/histos/histAll.json \
    --histos SvB.phh

# coffea4bees — fix Boolean axis passPreSel to True
python src/tools/convert_coffea_to_json.py \
    -i coffea4bees/output/histAll.coffea \
    -o coffea4bees/stats_analysis/histos/histAll.json \
    --histos SvB_MA.ps_hh_fine \
    --select passPreSel=True

# Use only SR
python src/tools/convert_coffea_to_json.py \
    -i output/histAll.coffea -o histos/histAll.json \
    --select region=SR

# Merge SR+CR into one histogram (sum over region axis)
python src/tools/convert_coffea_to_json.py \
    -i output/histAll.coffea -o histos/histAll.json \
    --sum region

# Both: fix passPreSel=True and merge regions
python src/tools/convert_coffea_to_json.py \
    -i output/histAll.coffea -o histos/histAll.json \
    --select passPreSel=True --sum region

# IntCategory axes with integer bin values — provide a mapping file
python src/tools/convert_coffea_to_json.py \
    -i output/histAll.coffea \
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

## roast.py

Reproducible production runs ("roasts") of the Snakemake workflows, keyed on git hashes. A roast pins a barista sha, a coffea4bees sha and a captured `--configfile`, gets an **isolated checkout on each host** (outside the mutagen-synced dev trees), runs each step in a detached tmux window, and publishes results to the owner's CERNBox www area plus a catalogue page in the docs site (`docs/prod/`, "Cupping notes").

```bash
bin/roast init --cmslpc-user jda102 --falcon-user jalison --cern-user johnda   # once, writes ~/.config/roast/config.json
bin/roast proxy                      # voms-proxy-init on cmslpc (interactive, weekly); --check shows time left
bin/roast new --config coffea4bees/workflows/config/nominal_run2.yml --phases B,C,D,F
bin/roast checkout <id>              # clone + checkout pinned shas on cmslpc and falcon
bin/roast submit <id> --step B -n    # dry run (snakemake -n): plan only
bin/roast submit <id> --step B -t    # test slice (--config test=true), runs locally on the node
bin/roast submit <id> --step B       # Phase B for real on cmslpc, in tmux session "roast"
bin/roast status <id>                # per-step exit codes, snakemake progress, condor / GPU
bin/roast attach <id> [--step B]     # ssh -t into the host's roast tmux session on that window
bin/roast submit <id> --step C       # when B is done: falcon
bin/roast resume <id> --step C       # after a dead driver (reboot / oomd): --unlock + --rerun-incomplete, same args as last submit
bin/roast publish <id>               # xrdcp small artefacts to CERNBox, write docs/prod/<id>.md + index.md
bin/roast archive <id>               # xrdcp heavy products (.coffea/.root/yml/json) to FNAL EOS eos.path/<id>/
bin/roast pull <id> [--only '*.coffea']  # rsync a roast's output/ to output/roasts/<id>/ on this machine
bin/roast ls | show <id> | index
```

Workflow configs may use the placeholder `{roast_id}` in paths (e.g. `make_classifier_input: root://cmseos.fnal.gov//store/user/<you>/HH4b_prod/{roast_id}/classifier_inputs/` in `nominal_run2.yml`); `roast submit` passes `--config roast_id=<id>` and `helpers/common.smk` resolves the placeholder (defaulting to the config `label` outside roast), so each production run writes to its own EOS directory. Heavy products go to FNAL EOS with `archive` (rules in an `archive` block, defaults `*.coffea *.root *.yml *.yaml *.json *.pkl`, no size cap; destination `eos.url` + `eos.path/<id>/`, e.g. `root://cmseos.fnal.gov//store/user/<you>/HH4b_prod/<id>/output/...`). What `publish` ships is controlled by a `publish` block (`include` filename globs, `exclude` path globs, `max_mb`) in `~/.config/roast/config.json`, overridable per roast under `publish_rules` in `roast.json`. Defaults: pdf/png/svg/yml/json/txt/log/md/csv/tex under 50 MB, excluding `*_test/`, Dask reports and `performance/` profiles; `logs/` and `roasts/<id>/` always go. `publish -n` lists the selection without copying; reruns skip files already on CERNBox with the same size.

Ids are `<label>_<YYYYMMDD>_<barista7>-<coffea4bees7>`; any unique prefix works. The cmslpc ssh target follows `host_file` (`~/.cmslpc-claude-host`) when present, so re-pinning after a dead node is one file edit.

Phases map to hosts as in `coffea4bees/workflows/README.md` (A, B, E, F on cmslpc; C, D on falcon). Non-phase workflows use `--step host:Snakefile[:targets]`, e.g. `--step falcon:coffea4bees/workflows/Snakefile_Run3_SvB_training.smk:output/Run3_quadjet_run2/SvB/train.done`. Chaining across hosts is manual: submit the next step when `status` shows the previous one at `exit=0`. Stdlib only; commit `roasts/<id>/` and `docs/prod/` so the Pages site picks them up.
