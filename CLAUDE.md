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


## Architecture

### Key Entry Points

- `runner.py` - Main analysis runner (~1000 lines). Handles dataset loading, processor execution, Dask/HTCondor submission, output saving, and git hash tracking.
- `run_container` - Bash script that dispatches to the correct container (Coffea, Combine, Classifier, Snakemake, Brilcalc) and handles host-specific configuration (LPC, lxplus, etc.)

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
