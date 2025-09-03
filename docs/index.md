# 

[![pipeline status](https://gitlab.cern.ch/cms-cmu//badges/master/pipeline.svg)](https://gitlab.cern.ch/cms-cmu//-/commits/master)

This is the repository for the 4b analyses at CMU based in coffea.

There is a website dopcumenting what this package does on this link [https://.docs.cern.ch/](https://.docs.cern.ch/).

Information about the analysis steps can be found in the [README](python/analysis/README.md) of the analysis folder.

## Installation

### How to run the python files

This repository assumes that you are running in a machine that has access to [cvmfs](https://cernvm.cern.ch/fs/). Then you can clone this repository as:

```bash
git clone ssh://git@gitlab.cern.ch:7999/cms-cmu/.git --recursive
```

The software required to run this package is encapsulated within a container. A script located in the `python/` folder simplifies the process of running the container seamlessly. For more details, refer to the [python/README.md](python/README.md). Additional information about the container can be found in the [Dockerfile](Dockerfile).

In addition, dont forget to run your voms-proxy to have access to remote files:

```bash
voms-proxy-init -rfc -voms cms --valid 168:00
```

#### Conda environment

In case you want to run the package using a conda environmnent, you can use the [environment.yml](environment.yml) file. Notice however that there are some libraries missing in case you want to run the full framework.

## How to contribute

If you want to submit your changes to the code to the **main repository** (aka cms-cmu gitlab user), you can create a new branch in your local machine and then push to the main repository. For example:

```bash
git checkout -b my_branch
git add file1 file2
git commit -m 'new changes'
git push origin my_branch
```

The `master` branch is protected, ensuring that users cannot accidentally modify its content. Once you are satisfied with your changes, push them to your branch. After your branch successfully passes the pipeline tests, you can create a merge request on the GitLab website to merge your changes into the main repository.

## REANA

[![Launch with Snakemake on REANA](https://www.reana.io/static/img/badges/launch-on-reana.svg)]($https://reana.cern.ch/launch?name=&specification=reana.yml&url=https%3A%2F%2Fgitlab.cern.ch%2Fcms-cmu%2F)

This package supports running workflows on [REANA](https://reana.cern.ch/). The REANA workflow is triggered manually via the GitLab CI pipeline or automatically every Saturday.

The output of the REANA workflow, including plots and files, is available at [https://plotsalgomez.webtest.cern.ch/HH4b/reana/](https://plotsalgomez.webtest.cern.ch/HH4b/reana/).

Each folder in this directory is named with the date the REANA job was executed and the corresponding Git commit hash. Unlike previous setups, folders are only copied to this location if the REANA job completes successfully.

## Information for continuos integration (CI)

The CI workflow is defined in the [gitlab-ci.yml](.gitlab-ci.yml) file. When you push your code to the main repository, the pipeline is triggered automatically.

If you have forked the repository, the GitLab CI pipeline requires your grid certificate to function. To run the GitLab CI workflow in your private fork, you must first configure specific variables to set up your voms-proxy. Follow [these instructions](https://awesome-workshop.github.io/gitlab-cms/03-vomsproxy/index.html) (excluding the final section, "Using the grid proxy") to complete the setup.

### To run the ci workflow locally in your machine

We use [Snakemake](https://snakemake.readthedocs.io/en/stable/) to replicate the workflow executed in the GitLab CI. Snakemake is the workflow management system utilized by REANA to submit jobs.

Within the [python/scripts/](python/scripts/) directory, there is a script named `run_local_ci.sh` that facilitates running a Snakemake workflow ([`Snakefile_testCI`](python/workflows/Snakefile_testCI)) locally, emulating the GitLab CI process. This script provides a convenient way to execute the CI workflow locally. To run it, navigate to the `python/` directory and execute:

```bash
source scripts/run_local_ci.sh NAME_OF_CI_JOB
```

Here, `NAME_OF_CI_JOB` corresponds to the specific job name in the GitLab CI workflow. The script will automatically execute the relevant part of the CI workflow. All output files will be stored in the `CI_output/` directory, with each job creating a separate subdirectory named after the job.

Keep in mind that Snakemake has a **feature** where it checks for the existence of output files before running a job. If the output files already exist, the job will be skipped, and the workflow will proceed to the next step. If you are debugging and need to rerun a specific job, you must manually delete the folder containing the existing output files.

For those interested in Snakemake, the `Snakefile_testCI` defines "rules" (jobs) similar to those in the GitLab CI workflow. The inclusion of rules in the workflow depends on the inputs specified in `rule all`. Rules can be defined anywhere after `rule all`, but they will only execute if their output files are listed in `rule all`. Unlike GitLab CI, where output files **should** be listed, in Snakemake, the output files must define the subsequent rule to execute.

## Information about the container

This packages uses its own container. It is based on `coffeateam/coffea-base-almalinux8:0.7.23-py3.10` including some additional python packages. This container is created automatically in the gitlab CI step **IF** the name of the branch (and the merging branch in the case of a pull request to the master) starts with `container_`. Additionally, one can take a look at the file [software/dockerfiles/Dockerfile_analysis](software/dockerfiles/Dockerfile_analysis) which is the one used to create the container.
