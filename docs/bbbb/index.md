
# Coffea4bees Python Package

[![pipeline status](https://gitlab.cern.ch/cms-cmu/coffea4bees/badges/master/pipeline.svg)](https://gitlab.cern.ch/cms-cmu/coffea4bees/-/commits/master)

This directory contains the main Python code for the Coffea4bees project, built on top of the [barista](https://gitlab.cern.ch/cms-cmu/barista) framework for high-energy physics analyses.

## Purpose

This folder provides all analysis, skimming, machine learning, plotting, and workflow automation tools for 4b physics analyses. It is the main entry point for running and developing new analysis features.

## Quickstart

To run analysis or skimming from this directory:

1. Clone the `barista` repository and then this repository.

```bash
git clone ssh://git@gitlab.cern.ch:7999/cms-cmu/barista.git
cd barista
git clone ssh://git@gitlab.cern.ch:7999/cms-cmu/coffea4bees.git
```

2. Run the main analysis script:

```bash
python runner.py --help
```

3. Explore subfolders for specialized tasks (skimming, ML, plotting, etc.).


## Folder Overview

- **analysis/**: Main analysis processors, helpers, metadata, tools, and tests for physics analysis.
- **analysis_dask/**: Dask-based analysis modules and configuration for distributed processing.
- **archive/**: Archived datasets, plots, and skims from previous runs.
- **classifier/**: Machine learning models, utilities, and scripts for classification tasks.
- **examples/**: Example scripts for analysis and meta-data rescue.
- **jet_clustering/**: Jet clustering algorithms, studies, and synthetic data generation.
- **metadata/**: Datasets, cross-sections, triggers, and luminosity files for analyses.
- **plots/**: Plotting scripts, styles, and metadata for visualizations.
- **scripts/**: Shell scripts for running, testing, and automating analysis jobs.
- **skimmer/**: Processors for filtering NanoAOD files and saving skimmed (picoAOD) files.
- **stats_analysis/**: Statistical analysis scripts and Combine framework integration.
- **workflows/**: Snakemake workflows and rules for automating analysis pipelines.

For more details about each component, refer to the `README.md` file in the respective folder.

## REANA Integration

[![Launch with Snakemake on REANA](https://www.reana.io/static/img/badges/launch-on-reana.svg)](https://reana.cern.ch/launch?name=Coffea4bees&specification=reana.yml&url=https%3A%2F%2Fgitlab.cern.ch%2Fcms-cmu%2Fcoffea4bees)

This package supports running workflows on [REANA](https://reana.cern.ch/). The REANA workflow is triggered manually via the GitLab CI pipeline or automatically every Saturday.

Workflow outputs (plots, files) are available at [https://plotsalgomez.webtest.cern.ch/HH4b/reana/](https://plotsalgomez.webtest.cern.ch/HH4b/reana/).

Each output folder is named with the REANA job execution date and the corresponding Git commit hash. Folders are only copied here if the REANA job completes successfully.