# bbreww

[![pipeline status](https://gitlab.cern.ch/cms-cmu/bbreww/badges/master/pipeline.svg)](https://gitlab.cern.ch/cms-cmu/bbreww/-/commits/master)

`bbreww` is a CMS analysis package built on top of the [Barista](https://barista.docs.cern.ch/) framework, designed for streamlined physics workflows at CMU.

## Overview

This repository contains analysis code, workflows, and CI and Snakemake configurations for the bbWW analysis. It leverages Barista and Coffea for scalable, reproducible data processing.

## Installation

To run the analysis, you need to have the `barista` framework installed. This can be done by cloning the repository and installing the required dependencies.
To install the necessary dependencies, run:

```bash
git clone ssh://git@gitlab.cern.ch:7999/cms-cmu/barista.git
cd barista/
git clone ssh://git@gitlab.cern.ch:7999/cms-cmu/bbreww.git
```

For more details on environment setup, see the [Barista documentation](https://barista.docs.cern.ch/).

## Contributing

If you want to submit your changes to the code, create a new branch in your local machine and push it to the main repository. For example:

```bash
git checkout -b my_feature_branch
git add file1 file2
git commit -m 'Describe your changes'
git push origin my_feature_branch
```

The `master` branch is protected to prevent accidental modifications. Once your branch passes pipeline tests, create a merge request on GitLab to propose merging your changes.

**General guidelines:**
- Do not push directly to `master`.
- Make sure your changes are well-documented and tested.
- Follow the Barista contribution guidelines if applicable.
- Use descriptive commit messages and branch names.
- Review pipeline results before requesting a merge.

For more details, see the [Barista documentation](https://barista.docs.cern.ch/) and contribution guide.

## Gitlab CI tests

The repository includes a Gitlab CI configuration file (`.gitlab-ci.yml`) that runs tests on the code. These tests are run on every commit and merge request to ensure that the code is working correctly.

To run the tests locally, you can use the snakemake workflow defined in `bbreww/workflows/Snakemake_CI`. You can run the tests by executing the following command:

```bash
./run_container snakemake -s bbreww/workflows/Snakemake_CI --use-apptainer XXXXX
```

Replace `XXXXX` with the name of the job you want to run from the gitlab CI configuration file. For example, to run the `analysis_test` job, you can use:

```bash
./run_container snakemake -s bbreww/workflows/Snakemake_CI --use-apptainer analysis_test
```

The output of the tests will be saved in the `CI_output/` directory.

## Directory Structure

Some key folders in this repository:

- `analysis/` — Analysis scripts and helpers
- `workflows/` — Snakemake and CI workflows
- `scripts/` — Utility scripts for running and testing
- `tests/` — Unit and integration tests

## Documentation

For more information about the Barista framework and analysis workflows, visit the [Barista documentation](https://barista.docs.cern.ch/).
