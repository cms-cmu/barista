# Barista

[![pipeline status](https://gitlab.cern.ch/cms-cmu/barista/badges/master/pipeline.svg)](https://gitlab.cern.ch/cms-cmu/barista/-/commits/master)

**Barista** is a toolkit for performing physics analyses at Carnegie Mellon University (CMU), built on top of the [Coffea](https://coffeateam.github.io/coffea/) framework.  

It streamlines data processing, analysis workflows, and reproducibility for CMS experiments.

- 📖 Full documentation: [barista.docs.cern.ch](https://barista.docs.cern.ch/)

## Table of Contents

- [Directory Structure](#directory-structure)

- [Installation](#installation)
- [How to run the python files](#how-to-run-the-python-files)
- [Conda environment](#conda-environment)
- [How to contribute](#how-to-contribute)
- [How to run the container for the code](#how-to-run-the-container-for-the-code)
- [Documentation](https://barista.docs.cern.ch/)

## Directory Structure

```text
barista/
├── src/              # Core utilities and modules
├── docs/             # Documentation and website files
├── software/         # Container, conda, and environment setup
├── dask_run.py       # Dask runner script
├── runner.py         # Main analysis runner script
├── README.md         # Project overview and instructions
└── ...               # Additional scripts, configs, and data
```

One can clone individual repositories for each analysis as need on top of the barista folder.

To get started quickly:

1. **Clone the repository**

```bash
git clone ssh://git@gitlab.cern.ch:7999/cms-cmu/barista.git
cd barista
```

2. **Initialize your proxy (for remote file access)**

  ```bash
  voms-proxy-init -rfc -voms cms --valid 168:00
  ```

3. **Run the analysis container**

  ```bash
  ./run_container
  ```

4. **Start an analysis (example)**

  ```bash
  ./run_container python runner.py --help
  ```

For more details, see the sections below.

## Installation

### How to run the python files

This repository assumes that you are running in a machine that has access to [cvmfs](https://cernvm.cern.ch/fs/). Then you can clone this repository as:

```bash
git clone ssh://git@gitlab.cern.ch:7999/cms-cmu/barista.git
```

The software required to run this package is encapsulated within a container. Additional information about the container can be found in the [Dockerfile](software/dockerfiles/Dockerfile_analysis).

In addition, dont forget to run your voms-proxy to have access to remote files:

```bash
voms-proxy-init -rfc -voms cms --valid 168:00
```

#### Conda environment (not recommended)

In case you want to run the package using a conda environmnent, you can use the [environment.yml](software/conda/environment.yml) file. Notice however that there are some libraries missing in case you want to run the full framework.

## How to contribute

If you want to submit your changes to the code to the **main repository** (aka cms-cmu gitlab user), you can create a new branch in your local machine and then push to the main repository. For example:

```bash
git checkout -b my_branch
git add file1 file2
git commit -m 'new changes'
git push origin my_branch
```

The `master` branch is protected, ensuring that users cannot accidentally modify its content. Once you are satisfied with your changes, push them to your branch. After your branch successfully passes the pipeline tests, you can create a merge request on the GitLab website to merge your changes into the main repository.


## How to run the container for the code

This project uses containers to simplify environment setup and ensure reproducibility. There are three main containers/environments, each for a specific purpose:


- **Analysis Container**: Coffea-based, for skimming, analysis, jet clustering, and histogram generation.
- **Combine Container**: Official CMS Combine, for statistical analysis.
- **Snakemake Environment**: For running workflows and automation.


You can run the required software interactively or as a job using the [run_container](./run_container) script:

- **Interactive Mode**: Container stays open for manual commands.
- **Job Mode**: Container runs a specific job and exits automatically.


```bash
Usage: ./run_container [command] [options]

Commands:
  [command...]         Run commands inside the analysis container.
                       Opens an interactive shell if no commands are given.
                       (Interactive shell is the only option to run on LPC HTCondor).
  combine [command...] Run commands inside the combine container.
                       Opens an interactive shell if no commands are given.
  snakemake [options]  Run snakemake with the specified options.
                       Requires --snakefile argument.
  --help               Show this help message.

Examples:
  source run_container
  # Open an interactive shell in the analysis container (HTCondor jobs)
  ./run_container
  ./run_container combine
  # Open an interactive shell in the combine container
  ./run_container combine combine -M AsymptoticLimits
  # Run snakemake with the specified Snakefile
  ./run_container snakemake --snakefile coffea4bees/workflows/Snakefile --cores 4
```


## How to run the coffea part of the code

The main entry point for analysis and skimming is `runner.py`, which should be run inside the Coffea container using the `run_container` script.

### Usage

```bash
./run_container python runner.py [OPTIONS]
```

### Main options

- `-p, --processor`: Path to the processor Python file (default: `coffea4bees/analysis/processors/processor_HH4b.py`)
- `-c, --configs`: Path to the main configuration YAML file (default: `coffea4bees/analysis/metadata/HH4b.yml`)
- `-m, --metadata`: Path to the datasets metadata YAML file (default: `coffea4bees/metadata/datasets_HH4b.yml`)
- `-o, --output`: Name of the output file (default: `hists.coffea`)
- `-op, --output-path`: Directory path for output files (default: `hists/`)
- `-y, --years`: Year(s) of data to process (e.g., `--years UL17 UL18`)
- `-d, --datasets`: Dataset name(s) to process (e.g., `--datasets HH4b ZZ4b`)
- `-e, --eras`: Data era(s) to process (data only, e.g., `--eras A B C`)
- `-s, --skimming`: Run in skimming mode instead of analysis mode
- `-t, --test`: Run in test mode with limited number of files
- `--systematics`: List of systematics to apply (e.g., `--systematics jes all`)
- `--dask`: Use Dask for distributed processing
- `--condor`: Submit jobs to HTCondor cluster
- `--debug`: Enable debug mode with verbose logging
- `--check-input-files`: Check input files for corruption before processing
- `--githash`: Override git hash for reproducibility tracking
- `--gitdiff`: Override git diff for reproducibility tracking

### Example commands

- Show help:

```bash
./run_container python runner.py --help
```

All arguments are documented in the help message (`--help`). For advanced configuration, see the processor and config YAML files.


## Information for continuous integration (CI)

The CI workflow is defined in the [gitlab-ci.yml](.gitlab-ci.yml) file. When you push your code to the main repository, the pipeline is triggered automatically.

If you have forked the repository (NOT recommended), the GitLab CI pipeline requires your grid certificate to function. To run the GitLab CI workflow in your private fork, you must first configure specific variables to set up your voms-proxy. Follow [these instructions](https://awesome-workshop.github.io/gitlab-cms/03-vomsproxy/index.html) (excluding the final section, "Using the grid proxy") to complete the setup.

### CI/CD for Multi-Repository Development

Our analysis framework is distributed across multiple Git repositories to maintain a clear separation of concerns. The **`barista`** repository contains the core physics analysis framework, while repositories like **`bbreww`** and **`coffea4bees`** contain specific analysis code and configurations that depend on `barista`. To ensure that changes across these repositories are tested together correctly, we employ a dynamic CI/CD workflow within GitLab. This system is designed to solve the common "chicken-and-egg" problem where a feature in one repository depends on an unmerged feature in another.

### Dynamic Branch Synchronization

The core of our CI/CD strategy is **dynamic branch synchronization**. The pipeline is intelligently configured to use the branch name of the commit that triggered the process (e.g., `feature/new-analysis`) as the target branch for its dependencies. When a pipeline runs in the `bbreww` repository, it will first attempt to find and use a branch with the exact same name in the central `barista` repository. This allows developers to work on features across the entire software stack simultaneously, confident that the CI will integrate and test the corresponding feature branches together.

To make this process robust, a crucial **fallback mechanism** is in place. If the pipeline is triggered on a branch in `bbreww` that does not exist in `barista`, it will automatically default to using `barista`'s stable `master` branch. This ensures that changes isolated to a single repository are always tested against the latest stable version of the core framework. The `setup_workspace` job in the pipeline handles this logic by first checking for the feature branch's existence before cloning the `barista` code and then assembling a complete workspace for subsequent testing and validation stages. This automated, flexible approach enables a streamlined development workflow while maintaining the integrity of our protected branches.

### To run the CI workflow locally

Within the [coffea4bees/scripts/](coffea4bees/scripts/) directory, there is a script named `run-local-ci.sh` that facilitates running a Snakemake workflow ([`Snakefile_testCI`](coffea4bees/workflows/Snakefile_testCI)) locally, emulating the GitLab CI process. This script provides a convenient way to execute the CI workflow locally. To run it, navigate to the `coffea4bees/` directory and execute:

```bash
source scripts/run-local-ci.sh NAME_OF_CI_JOB
```

For those interested in Snakemake, the `Snakefile_testCI` defines "rules" (jobs) similar to those in the GitLab CI workflow. The inclusion of rules in the workflow depends on the inputs specified in `rule all`. Rules can be defined anywhere after `rule all`, but they will only execute if their output files are listed in `rule all`, or if you call directly the name of the rule.

## Information about the container

This package uses its own container. It is based on `coffeateam/coffea-base-almalinux8:0.7.23-py3.10` including some additional python packages. This container is created automatically in the GitLab CI step **if** the name of the branch (and the merging branch in the case of a pull request to the master) starts with `container_`. Additionally, you can review the file [software/dockerfiles/Dockerfile_analysis](software/dockerfiles/Dockerfile_analysis), which is used to create the container.
