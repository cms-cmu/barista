# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


## Project Overview

**Barista** is a base class library for CMS physics analysis, built on the [Coffea](https://coffeateam.github.io/coffea/) framework. Python 3.10+ with Snakemake workflow orchestration.

Analysis packages built on barista include **coffea4bees** (`cms-cmu/coffea4bees`, HH→4b) and **bbreww** (`cms-cmu/bbreww`, bbWW). These are separate CERN GitLab repos cloned into the barista workspace at development and CI time.

Repository: `gitlab.cern.ch/cms-cmu/barista` (CERN GitLab)

## Skills 

Do not call /done or /start — these are top-level skills that write to ~/ClaudeBrain/ and are reserved for the home directory Claude instance.
Use the /session-summary skill to document progress or save state.


## Running Commands

All Python analysis code runs inside Apptainer/Singularity containers via the `run_container` script. Do not run analysis code directly on the host.

When testing changes use the skills /test-barista and /test-coffea4bees


```bash
# Interactive analysis shell
./run_container

# Run analysis (processor, config, and metadata come from the analysis package)
./run_container python runner.py -p <processor.py> -c <config.yml> -m <datasets.yml> -y UL18 -t

# Other containers
./run_container combine        # CMS Combine (statistical analysis)
./run_container classifier     # ML training/inference

# Snakemake workflows (uses Pixi, not container)
./run_container snakemake --snakefile <workflow.smk> --cores 4

# CERN lxplus: --condor auto-detects the site and uses dask_lxplus (CERN HTCondor); GPU trainings
# submitted with `./run_container classifier <cmd>` become HTCondor GPU jobs. One-time setup:
./run_container lxplus-setup
./run_container voms-proxy-init -voms cms -rfc --valid 168:00 -out proxy/x509_proxy
```

### Running Tests Locally

```bash
# Unit tests (run inside container)
./run_container python -m src.tests.hist_collection
./run_container python -m src.tests.kappa_framework
./run_container python -m pytest src/plotting/tests/

# Local CI emulation (from coffea4bees/ directory)
source scripts/run-local-ci.sh NAME_OF_CI_JOB
```

### Classifier Training Workflow

```bash
# Train/evaluate/analyze an HCR classifier (uses Snakemake + classifier container)
./run_container snakemake \
    --snakefile src/classifier/workflow/Snakefile \
    --configfile <path/to/workflow_config.yml> \
    --cores 1

# Dry-run to preview DAG
./run_container snakemake --snakefile src/classifier/workflow/Snakefile \
    --configfile <path/to/workflow_config.yml> -np
```

Example `workflow_config.yml` files live in `coffea4bees/classifier/config/workflows/`.
The DAG is: `train → evaluate` and `train → analyze` (parallel after training).


## Roast Production Workflow System (`bin/roast`)

Production analyses and multi-phase pipelines must be executed via `bin/roast` rather than manual ad-hoc scripts. `roast` enforces reproducibility by pinning exact git SHAs (`barista` and `coffea4bees`), capturing merged configurations, creating isolated checkouts per cluster, and running in detached `tmux` sessions.

### Production Workflow Types & Manifest Creation

Roast standardizes three core workflow families defined in `coffea4bees/workflows/workflows.yml`:

1. **Stat-Only (Nominal) Pipeline:**
   Runs baseline event selection, FvT background estimation, SvB classifier evaluation, data/MC comparison plots, and stat-only Combine datacards.
   ```bash
   bin/roast new --config coffea4bees/workflows/config/nominal_run2.yml --phases B,C,C4,D,F --label nominal_run2
   # or starting from raw nanoAOD skimming: --phases A,B,C,C4,D,F
   ```

2. **Background Systematics Pipeline (`bkg_syst`):**
   Generates mixed-data subsamples, calibrates independent JCM transfer functions, trains 16 FvT models on GPUs (Falcon/Bridges-2), executes two-stage background closure (variance, bias, spurious signal), and runs Combine statistical inference.
   ```bash
   bin/roast new --config coffea4bees/workflows/config/ttHbb_bkg_syst.yml --phases BKG_C,BKG_F --label ttHbb_bkg_syst
   # or full chain including subsample slicing: --phases BKG_AB,BKG_C,BKG_F
   ```

3. **Full Systematics Pipeline (`full_syst`):**
   Integrates signal systematic variations (`others` friend weights, `jes` PyTorch kinematic shifts) with data-driven background closure models from `bkg_syst`. Validates nominal distributions against upstream stat-only, validates systematic cutflows against reference counts, plots shape variations with interactive HTML galleries, and runs complete Combine inference (limits, significance, likelihood scans, postfit, and impacts).
   ```bash
   # Config points to upstream stat-only (inputs.nominal_hists) and bkg_syst (inputs.bkg_syst_file)
   bin/roast new --config coffea4bees/workflows/config/ttHbb_full_syst.yml --phases FULL_SYST --label ttHbb_full_syst
   ```

### Mandatory Roast Sequence:
1. **Preflight Configuration Validation:**
   ```bash
   bin/roast validate coffea4bees/workflows/config/<config>.yml
   ```
   Audits for parameter divergence, duplicate key definitions across sections, invalid filesystem paths, and era mismatches. (`bin/roast new` also invokes this check automatically; bypass with `--no-validate` only if strictly needed).
2. **Create Roast Manifest:**
   ```bash
   bin/roast new --config coffea4bees/workflows/config/<config>.yml --phases <phases> --label <label> [--nautilus | --bridges2]
   ```
   Pins the current git HEADs, merges any `base:` configs, and records validation status in `roasts/<id>/roast.json`.
3. **Cluster Checkouts:**
   ```bash
   bin/roast checkout <id> --host cmslpc
   bin/roast checkout <id> --host falcon    # or nautilus / bridges2
   ```
4. **Dry-run Each Step:**
   ```bash
   bin/roast submit <id> --step <step> -n
   ```
5. **Execute Step:**
   ```bash
   bin/roast submit <id> --step <step>
   ```
   Automatically generates visual Snakemake rulegraph DAG (`logs/<step>_rulegraph.svg`) before execution and execution summary (`logs/<step>_summary.txt`) on exit 0.
6. **Track Status:**
   ```bash
   bin/roast status <id>
   bin/roast monitor
   ```
7. **Step C Rule:**
   Phase C consists of **BOTH** `C` (GPU training & evaluation) and `C4` (CPU FvT data closure on `cmslpc`). Always run both. FvT is strictly trained in SB, never in SR.
8. **Copy & Sync Products:**
   ```bash
   bin/roast copy <id>
   ```
9. **Publish Cupping Notes & Dashboard:**
   ```bash
   bin/roast publish <id>
   ```
   Publishes results, cupping notes, and an interactive CERNBox web index rendering inline SVG DAG rulegraphs and execution metrics.
10. **Archive & Clean:**
   ```bash
   bin/roast archive <id>
   bin/roast clean <id> --yes
   ```


## Architecture

### Key Entry Points

- `runner.py` - Main analysis runner (~1000 lines). Handles dataset loading, processor execution, Dask/HTCondor submission, output saving, and git hash tracking.
- `run_container` - Bash script that dispatches to the correct container (Coffea, Combine, Classifier, Snakemake, Brilcalc) and handles host-specific configuration (LPC, lxplus, falcon, bridges2)
- `src/runner/cluster.py` - Dask cluster backends: `setup_condor_cluster` dispatches on `detect_condor_site()` to `lpcjobqueue` (FNAL LPC) or `dask_lxplus` (CERN lxplus); SLURM (falcon/bridges2) and local clusters live here too
- `software/condor/submit_classifier_lxplus.sh`, `software/snakemake/scripts/lxplus_condor_submit.py` - HTCondor GPU job submission on lxplus (direct command / Snakemake `lxplus_gpu` profile)

### `src/` - Barista Base Class Library

Code here is analysis-agnostic and should not depend on any specific analysis package.

Dependency flow: `utils → (storage, data_formats, math) → (hist_tools, physics, dask_tools) → (plotting, skimmer) → scripts`

| Module                 | Purpose                                                                 |
|------------------------|-------------------------------------------------------------------------|
| `data_formats/`        | Conversions between Awkward Array, NumPy, ROOT formats; friend tree I/O |
| `config/`              | YAML/OmegaConf config loading, parsing, patching                        |
| `dask_tools/`          | Distributed computing with Dask; delayed histogram ops                  |
| `storage/`             | EOS/XRootD file system abstraction                                      |
| `math_tools/`          | Numba JIT-compiled math, statistics, partitioning                       |
| `hist_tools/`          | Histogram creation, merging, templates                                  |
| `physics/`             | Jet corrections (JES/JER), event selection, kinematic calculations      |
| `skimmer/`             | NanoAOD→PicoAOD event filtering (integrity checks, metadata)            |
| `friendtrees/`         | Friend tree creation and management                                     |
| `plotting/`            | Matplotlib-based visualization                                          |
| `classifier/`          | HCR ensemble ML models; `workflow/` holds the Snakemake training DAG    |
| `data/`                | Physics corrections: b-tagging SFs, JEC, pileup weights, golden JSONs   |
| `system/`              | EOS filesystem utilities (overlaps with `storage/`)                     |
| `tests/`               | Module-level integration tests (`hist_collection`, `kappa_framework`)   |

### Container Architecture

Three containers for different tasks:
- **Coffea** (`barista:latest`) - Main analysis (default)
- **Combine** (`combine-container:CMSSW_11_3_4-combine_v9.1.0`) - Statistical analysis
- **Classifier** (`barista:classifier_latest`) - ML training/inference

Different pixi environments:

- **Snakemake** (`snakemake`) - Workflow execution
- **Brilcalc** (`brilcalc`) - Luminosity calculations
- **Reana** (`reana`) - Reana workflow execution)

## CI/CD

GitLab CI stages: `build → setup → code → skimmer → friendtree → analysis → tools → plot → cutflow → deploy → validation → pages → classifier`

Pipeline rules:
- Skips on markdown-only changes, tags, and branches containing 'test'
- Runs on merge requests and pushes to branches without open MRs
- Container rebuild triggered only by branches starting with `container_`

## Development Workflow

- `master` branch is protected; work on feature branches and create merge requests
- Grid proxy required for remote file access: `voms-proxy-init -rfc -voms cms --valid 168:00`
- Pixi (`pixi.toml` / `software/pixi/`) manages the Snakemake environment; container manages the analysis environment
- `coffea4bees/` is a **separate git repository** cloned into the workspace. Changes there won't appear in `git diff` for barista — commit and push from within `coffea4bees/` separately.

## Automated Issue Bot

When invoked headlessly by the issue-bot (running on falcon.phys.cmu.edu):

- Do NOT run git commands — branching, committing, and pushing are handled externally
- No interactive input is available — if the issue is ambiguous, make changes using your best judgement and list any judgement calls made
- Use `/test-local-CI` to validate changes before finishing

## Where to Make Changes

For issues filed against **coffea4bees**, start in `coffea4bees/`. If the root cause lies in the barista base library (`src/`), fix it there — follow the root cause wherever it leads.

For issues filed against **barista**, edit in `src/`. Do not add analysis-specific logic to `src/`.

## Validating a Fix

After making changes, run `/test-barista`. A successful fix means all previously passing tests still pass with no new failures. Report which tests passed and which (if any) failed.
