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

### Commands

```bash
bin/roast init --cmslpc-user jda102 --falcon-user jalison --cern-user johnda   # once, writes ~/.config/roast/config.json
bin/roast proxy [--host cmslpc|falcon] [--check]   # voms-proxy-init on that host (interactive, weekly); default host cmslpc
bin/roast new --config coffea4bees/workflows/config/nominal_run2.yml --phases B,C,C4,D,F
bin/roast checkout <id> [--host falcon]  # clone + checkout pinned shas on every host of the roast (or one)
bin/roast submit <id> --step B -n        # dry run (snakemake -n): plan only — ALWAYS do this first
bin/roast submit <id> --step B -t        # test slice (--config test=true), runs locally on the node
bin/roast submit <id> --step B           # the real thing, tmux session "roast", window <label>_<date>_B
bin/roast submit <id> --step B --extra "--touch"   # any extra snakemake args
bin/roast status <id>                    # per-step state + batch-system detail for this roast's jobs
bin/roast attach <id> [--step B]         # ssh -t into the host's roast tmux session on that window
bin/roast resume <id> --step B           # after a failure / dead driver: --unlock + --rerun-incomplete, same args as last submit
bin/roast publish <id> [-n] [--docs-only]   # xrdcp small artefacts to CERNBox, write docs/prod/<id>.md + index.md
bin/roast archive <id>                   # xrdcp heavy products (.coffea/.root/yml/json) to FNAL EOS eos.path/<id>/
bin/roast pull <id> [--only '*.coffea']  # rsync a roast's output/ to output/roasts/<id>/ on this machine
bin/roast pourover <id> --step F     # pull that step's merged histograms + its plot config, serve pourOver locally
bin/roast rm <id> [--yes]                # delete a roast everywhere (dry run unless --yes): local, host checkouts, EOS archive, CERNBox
bin/roast ls | show <id> | index
```

Step keys (`PHASES` in `roast.py`): `A` `B` `E` `F` `C4` on cmslpc, `C` `D` on falcon. `C4` is the FvT closure
(`Snakefile_PhaseC_4_FvT_closure.smk`, a processor pass, hence cmslpc). Non-phase workflows use
`--step host:Snakefile[:targets]`. Chaining across hosts is manual: submit the next step when `status` shows the
previous one at `exit=0`. Steps not listed at `new` can be added later by appending to `steps` in
`roasts/<id>/roast.json` (then `checkout --host <host>` if it is a new host).

### The nominal_run2 sequence (Run 2, config `coffea4bees/workflows/config/nominal_run2.yml`)

Each step below is `-n` first, then for real; `status` until `exit=0`. Between phases there are **handoffs**:
files the next phase reads must be committed to coffea4bees and shipped into the checkouts (see next section),
and the first run of every cutflow check only *dumps* its counts — bless the dump as the reference and ship it.

| step | host | what runs | handoff afterwards |
|---|---|---|---|
| `B` | cmslpc | B.1: processor data+ttbar without JCM → fit JCM → rerun with JCM → NoFvT plots + gallery, cutflow checks (`known_fullCounts_JCM_{NoJCM,wJCM}.yml`), closure tables. B.2: classifier-input friend trees for `classifier_inputs.datasets` (data, ttbar, signals) → EOS `HH4b_prod/<id>/classifier_inputs/` | copy `output/<label>/computeJCM/JCM_<tag>/jetCombinatoricModel_SB_<tag>.yml` → `coffea4bees/metadata/weights/JCM/<id>/`; copy `output/<label>/classifier_inputs/classifier_inputs_friends.json` → `coffea4bees/metadata/datasets/classifier_inputs_<id>.json`; bless the two JCM cutflow dumps |
| `C` | falcon | FvT: train (3-fold, GPU), analyze, evaluate → EOS `classifier/FvT_nominal`, `friend/FvT_nominal`; input/weight plots → CERNBox | none (the `fvt`/`svb`/`analysis_config` blocks reference `{eos_base}/friend/FvT_nominal` via `{roast_id}`) |
| `C4` | cmslpc | FvT closure: processor on data with JCM×FvT (ttbar merged from B.1's wJCM singlefiles), `plotsAll` plots + gallery, cutflow check (`known_fullCounts_FvT_closure.yml`), closure table (Multijet = 3b data) | bless the cutflow dump; look at SR/SB data/Bkg in the closure table before training the SvB |
| `D` | falcon | SvB: train (signal vs 3b×JCM×FvT + ttbar), analyze, evaluate → `classifier/SvB_nominal`, `friend/SvB_nominal` | none (Phase F reads `friends.SvB_MA` = `friend/SvB_nominal`) |
| `F` | cmslpc | F.1: processor on `dataset:` (data, ttbar, signals) with JCM+FvT+SvB friends, blinding → `histAll_<label>.coffea`, cutflow check (`known_fullCounts_<label>.yml`), `plotsAll_ttbarWeights` plots + gallery. F.2: Combine inputs, workspaces, fits, limits, scans per channel | bless the cutflow dump; `publish`; `archive` |

Handoff files are per roast (paths contain `{roast_id}`), so nothing in the analysis metadata points at another
production by accident; `fvt.workflow_overrides` / `svb.workflow_overrides` replace only the input flags of the
checked-in classifier templates (`helpers/common.smk: write_workflow_overrides`).

```bash
ID=<roast id>
CK=~/nobackup/HH4b/prod/$ID/barista            # cmslpc checkout (prod_root in ~/.config/roast/config.json)
# --- after B ---
scp cmslpc:$CK/output/TESTRun2/computeJCM/JCM_2024_v2/jetCombinatoricModel_SB_2024_v2.yml coffea4bees/metadata/weights/JCM/$ID/
scp cmslpc:$CK/output/TESTRun2/classifier_inputs/classifier_inputs_friends.json coffea4bees/metadata/datasets/classifier_inputs_$ID.json
scp cmslpc:$CK/output/TESTRun2/computeJCM/cutflow_NoJCM.yml coffea4bees/analysis/tests/known_fullCounts_JCM_NoJCM.yml   # bless
scp cmslpc:$CK/output/TESTRun2/computeJCM/cutflow_wJCM.yml  coffea4bees/analysis/tests/known_fullCounts_JCM_wJCM.yml
# --- after C4 / F ---
scp cmslpc:$CK/output/TESTRun2/FvT_closure/cutflow_FvT_closure.yml coffea4bees/analysis/tests/known_fullCounts_FvT_closure.yml
scp cmslpc:$CK/output/TESTRun2/cutflow_TESTRun2.yml coffea4bees/analysis/tests/known_fullCounts_TESTRun2.yml
```
(when a check *fails*, the dump survives as `cutflow_<pass>_failed.yml` and the verdict as
`cutflow_validation_<pass>_result.txt`; both are published.)

### Updating code in a live roast

A roast pins shas; the checkouts do **not** track a branch. To run newer commits (a fix, a blessed reference, a
handoff file) without a new roast, ship them the way `checkout` does — a git push straight into the checkout,
then detach it there. Do this for every host that will run the affected step:

```bash
git -C coffea4bees push jda102@cmslpc307.fnal.gov:$CK/coffea4bees "<sha>:refs/roasts/$ID-fix"
ssh cmslpc307.fnal.gov "cd $CK/coffea4bees && git checkout -q --detach <sha>"
# same with ~/work/prod/$ID/barista on falcon (jalison@falcon.phys.cmu.edu), and for barista itself without the /coffea4bees
```
`roast.json` keeps the *original* shas; the step log records the shas that actually ran (`=== barista … coffea4bees … ===`).
Quote the refspec if the sha is in a variable (`"${SHA}:refs/…"` — zsh eats `$SHA:r`).

### Gotchas

* **The roast runs its captured `roasts/<id>/config.yml`, not the repo file.** After editing
  `coffea4bees/workflows/config/nominal_run2.yml`, copy it over the captured one (and commit `roasts/`); `submit`
  re-ships the roast dir (`scp -p`, mtimes preserved) and logs `config-edited`.
* **Changing the config mid-roast reruns processor jobs.** `config.yml` is an input of the `create_*_config`
  rules; a newer config regenerates them and every processor job downstream re-runs. If the generated processor
  config is unaffected (or you patched it by hand on the host), `touch` the finished outputs
  (`output/<label>/**/singlefiles/*.coffea` and the generated `*_config.yml`) before resuming, and check with `-n`.
  `--rerun-triggers` without `mtime` does not prevent this in snakemake 9.25.
* **Never run `submit -n` while that step is running.** It regenerates the step script; bash reads scripts
  incrementally and the live run derails (no exit marker). The guard refuses only when a snakemake driver process
  for the roast is alive; a leftover window with no driver is closed automatically.
* **falcon** runs snakemake through the SLURM profile (`software/snakemake/profiles/falcon`); `roast` passes
  `--jobs` for that. It needs its own proxy: `bin/roast proxy --host falcon`. Training jobs land on the GPU nodes
  (`squeue -u <user>`), the FvT/SvB take ~1.5 h each.
* **cmslpc processor steps** need a valid proxy in the checkout (`proxy/x509_proxy`, seeded from `/tmp/x509up_u<uid>` by
  the step script) and use `--shared-dask --condor` (one long-lived dask daemon that tars `src/` + `coffea4bees/` once —
  kill it after shipping code: `ps -u $USER -o pid=,args= | grep start-cluster-daemon`).
* **Missing trigger weights are an error** (`require_trigWeight`, default true in `processor_HH4b`). A file with no
  entry in the trigger-weight friend index stops the job with the dataset name; declare the exception
  (`require_trigWeight: false` + comment) or regenerate the friends (Phase A.2, e.g. `trigweights_ZZ4b_UL16.yml`).
* `publish` records the shipped HTML pages (galleries, cutflow closure tables) in `roast.json` and the docs page
  links them under "Pages". Commit `roasts/<id>/` and `docs/prod/` afterwards so the Pages site picks them up.

### Config knobs

Workflow configs may use the placeholder `{roast_id}` in paths (e.g. `make_classifier_input: root://cmseos.fnal.gov//store/user/<you>/HH4b_prod/{roast_id}/classifier_inputs/` in `nominal_run2.yml`); `roast submit` passes `--config roast_id=<id>` and `helpers/common.smk` resolves the placeholder (defaulting to the config `label` outside roast), so each production run writes to its own EOS directory. Heavy products go to FNAL EOS with `archive` (rules in an `archive` block, defaults `*.coffea *.root *.yml *.yaml *.json *.pkl`, no size cap; destination `eos.url` + `eos.path/<id>/`, e.g. `root://cmseos.fnal.gov//store/user/<you>/HH4b_prod/<id>/output/...`). What `publish` ships is controlled by a `publish` block (`include` filename globs, `exclude` path globs, `max_mb`) in `~/.config/roast/config.json`, overridable per roast under `publish_rules` in `roast.json`. Defaults: pdf/png/svg/html/yml/json/txt/log/md/csv/tex under 50 MB, excluding `*_test/`, Dask reports and `performance/` profiles; `logs/` and `roasts/<id>/` always go. `publish -n` lists the selection without copying; reruns skip files already on CERNBox with the same size.

`pourover` closes the loop on a finished step: it reads the `makePlots` command the step actually ran (the launcher passes `--printshellcmds`, so the resolved command is in `logs/<step>.log`), rsyncs that histogram file and that plot config into `output/roasts/<id>/`, and starts pourOver against them. rsync is the cache, so a second call skips the transfer. `--list` shows the commands a step ran, `--which N` / `--match SUBSTR` pick among them, `--coffea` / `--config` override either input, `--no-pull` works offline, and `--extra` passes further pourOver flags. The interpreter comes from `pourover.python` in the config (default `~/python-environments/pourover/bin/python`).

`status` reports each step as `not started`, `running` (a snakemake driver process is alive), `error` (the driver died after a job failed), `stalled` (died with no error) or `exit=N`, plus snakemake progress and the last log line. It also lists this roast's batch jobs, matched by the scheduler's record of the submitting directory: HTCondor batches by state on cmslpc, and on falcon each slurm job with its rule name, state, elapsed/limit, node, cpus/mem/gres and the last line of that job's own slurm log (training loss, batch counter), followed by jobs that finished in the last two days and a one-line cluster summary.

Ids are `<label>_<YYYYMMDD>_<barista7>-<coffea4bees7>`; any unique prefix works. The cmslpc ssh target follows `host_file` (`~/.cmslpc-claude-host`) when present, so re-pinning after a dead node is one file edit. Stdlib only.
