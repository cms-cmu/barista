# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Barista** is a base class library for CMS physics analysis at Carnegie Mellon University, built on the [Coffea](https://coffeateam.github.io/coffea/) framework. Python 3.10+ with Snakemake workflow orchestration.

Analysis packages built on barista include **coffea4bees** (`cms-cmu/coffea4bees`, HH→4b) and **bbreww** (`cms-cmu/bbreww`, bbWW). These are separate CERN GitLab repos cloned into the barista workspace at development and CI time.

Repository: `gitlab.cern.ch/cms-cmu/barista` (CERN GitLab)

### Data-Taking Periods

- **Run 2**: years `UL16` (has `preVFP` and `postVFP` eras), `UL17`, `UL18`
- **Run 3**: years `2022`, `2023`

These are passed to `runner.py` via the `-y`/`--years` flag (e.g., `--years 2022 2023`).

## Running Commands

All Python analysis code runs inside Apptainer/Singularity containers via the `run_container` script. Do not run analysis code directly on the host.

```bash
# Interactive analysis shell
./run_container

# Run analysis (processor, config, and metadata come from the analysis package)
./run_container python runner.py -p <processor.py> -c <config.yml> -m <datasets.yml> -y UL18 -t

# Run in test mode (limited files)
./run_container python runner.py --test

# Run with Dask distributed
./run_container python runner.py --dask

# Other containers
./run_container combine        # CMS Combine (statistical analysis)
./run_container classifier     # ML training/inference
./run_container brilcalc       # Luminosity calculations

# Snakemake workflows (uses Pixi, not container)
./run_container snakemake --snakefile <workflow.smk> --cores 4
```

### Tests

Barista base class tests are shell scripts in `src/scripts/`, run inside the analysis container:
- `code-hist-collection.sh` - Histogram collection tests
- `code-kappa-framework.sh` - Framework tests
- `skimmer-basic-test.sh` - Skimming tests

CI validation uses `src/tests/check_yaml.py` for YAML comparison with tolerance.

Analysis packages have their own test scripts (e.g., `coffea4bees/scripts/`).

## Architecture

### Key Entry Points

- `runner.py` - Main analysis runner (~1000 lines). Handles dataset loading, processor execution, Dask/HTCondor submission, output saving, and git hash tracking.
- `dask_run.py` - Dask-specific distributed runner
- `run_container` - Bash script that dispatches to the correct container (Coffea, Combine, Classifier, Snakemake, Brilcalc) and handles host-specific configuration (LPC, lxplus, etc.)

### `src/` - Barista Base Class Library

Code here is analysis-agnostic and should not depend on any specific analysis package.

Dependency flow: `utils → (storage, data_formats, math) → (hist, physics, dask) → (plotting, skimmer) → scripts`

| Module | Purpose |
|--------|---------|
| `data_formats/` | Conversions between Awkward Array, NumPy, ROOT formats; friend tree I/O |
| `config/` | YAML/OmegaConf config loading, parsing, patching |
| `dask/`, `dask_tools/` | Distributed computing with Dask; delayed histogram ops |
| `storage/` | EOS/XRootD file system abstraction |
| `math/` | Numba JIT-compiled math, statistics, partitioning |
| `hist/`, `hist_tools/` | Histogram creation, merging, templates |
| `physics/` | Jet corrections (JES/JER), event selection, kinematic calculations |
| `skimmer/` | NanoAOD→PicoAOD event filtering (integrity checks, metadata) |
| `friendtrees/` | Friend tree creation and management |
| `plotting/` | Matplotlib-based visualization |
| `classifier/` | HCR ensemble ML models |
| `data/` | Physics corrections: b-tagging SFs, JEC, pileup weights, golden JSONs |

### Container Architecture

Five containers for different tasks:
- **Coffea** (`barista:latest`) - Main analysis (default)
- **Combine** (`combine-container:CMSSW_11_3_4-combine_v9.1.0`) - Statistical analysis
- **Classifier** (`chuyuanliu/heptools:ml`) - ML training/inference
- **Snakemake** (`barista:reana_latest`) - Workflow execution
- **Brilcalc** - Luminosity calculations

CVMFS paths are auto-resolved when available for faster startup. Host-specific bind mounts are configured automatically (LPC: `/uscmst1b_scratch`, `/uscms_data`; lxplus: `/afs`, `/eos`).

## CI/CD

GitLab CI stages: `build → setup → code → skimmer → friendtree → analysis → tools → plot → cutflow → validation → pages`

Pipeline rules:
- Skips on markdown-only changes, tags, and branches containing 'test'
- Runs on merge requests and pushes to branches without open MRs
- Container rebuild triggered only by branches starting with `container_`

CI uses **dynamic branch synchronization**: if a branch with the same name exists in dependent repos (coffea4bees, bbreww), those branches are tested together; otherwise it falls back to `master`.

CI stage definitions are modular, split across:
- `src/workflows/gitlab-CI/stages_*.yml`
- External includes from analysis package repos

## Development Workflow

- `master` branch is protected; work on feature branches and create merge requests
- Grid proxy required for remote file access: `voms-proxy-init -rfc -voms cms --valid 168:00`
- Git hash and diff are tracked automatically in outputs for reproducibility
- The Dockerfile is at `software/dockerfiles/Dockerfile_analysis` (base: `coffeateam/coffea-base-almalinux8`)
- Pixi (`pixi.toml` / `software/pixi/`) manages the Snakemake environment; container manages the analysis environment
- Documentation auto-deploys to `barista.docs.cern.ch` via MkDocs on push to master
