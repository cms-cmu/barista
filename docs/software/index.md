# Software Environment & Container Setup

This repository provides a comprehensive containerized software environment for high-energy physics analysis workflows. The setup ensures reproducible, portable execution across different computing environments including local machines, CERN computing facilities (LPC, lxplus), and distributed computing systems.

## Overview

The repository uses a layered approach to software management:

- **Containerization**: Apptainer/Docker containers for isolated execution environments
- **Package Management**: Multiple options including Pixi, Conda, and pip for dependency management
- **Workflow Orchestration**: Integrated support for Snakemake workflows
- **Multi-Environment Support**: Automated configuration for different computing clusters

## Quick Start

Most analysis workflows are executed using the `run_container` script:

```bash
# Interactive shell in analysis container
./run_container

# Run specific commands
./run_container python analysis_script.py

# Use combine container for statistical analysis
./run_container combine

# Execute Snakemake workflows
./run_container snakemake --snakefile my_workflow.snakefile
```

## Container Images

The repository supports multiple specialized containers (also available in `cvmfs`):

### Analysis Container (Default)

- **Image**: `gitlab-registry.cern.ch/cms-cmu/barista:latest`
- **Purpose**: Primary analysis environment with Coffea, Awkward Array, and HEP tools
- **Usage**: Default container for most physics analysis tasks

### Combine Container

- **Image**: `gitlab-registry.cern.ch/cms-analysis/general/combine-container:CMSSW_11_3_4-combine_v9.1.0-harvester_v2.1.0`
- **Purpose**: Statistical analysis using the CMS Combine tool
- **Usage**: Limit setting, significance calculations, and statistical inference

## Environment Management

### Pixi Environment (`software/pixi/`)

Modern package manager with cross-platform support:

- **Configuration**: `requirements.txt` with minimal Python and Snakemake dependencies
- **Features**: Fast dependency resolution, reproducible environments
- **Installation**: Automatically handled by `run_container` script

### Conda Environment (`software/conda/`)

Traditional conda environment with comprehensive package list:

- **Configuration**: `environment.yml` with 400+ scientific computing packages
- **Purpose**: Legacy support and detailed dependency specification
- **Includes**: Machine learning libraries, data analysis tools, visualization packages

This environment is NOT recommended, but is maintained for cases where singularity is not available.

### Docker Images (`software/dockerfiles/`)

Container definitions for various use cases:

- `Dockerfile_analysis`: Primary analysis container
- `Dockerfile_analysis_reana`: REANA workflow execution
- `ml/`: Machine learning specific containers

## Computing Environment Support

The `run_container` script automatically detects and configures for different computing environments:

### CERN LPC (cmslpc)

- **Storage Binding**: `/uscmst1b_scratch`, `/uscms_data/`
- **Pixi Location**: `/uscms_data/d3/${USER}/.pixi` (persistent storage)
- **Special Features**: HTCondor job support with `.shell` script

### CERN lxplus

- **Storage Binding**: `/afs`, `/eos`, `/cvmfs`
- **Grid Security**: Automatic mounting of grid certificates
- **CVMFS Access**: Full access to CERN software repositories
- **HTCondor (Dask workers)**: `runner.py --condor` auto-detects lxplus and submits Dask workers to
  CERN HTCondor with [`dask_lxplus`](https://github.com/cernops/dask-lxplus) (`CernCluster`). Workers run
  inside the analysis image (`MY.SingularityImage`, `+JobFlavour`, `MY.SendCredential`). The code tarball
  is staged once on EOS and fetched by URL; worker `stdout`/`stderr` are delivered to
  `<lxplus_eos_scratch>/condor_logs/<run>/` (default `/eos/cms/store/group/phys_higgs/ttHbb/$USER/4b/barista_scratch`).
  Runner config keys: `condor_site`, `lxplus_job_flavour` (`workday`), `lxplus_disk_per_worker` (`10GB`),
  `lxplus_death_timeout` (`3600`), `lxplus_scheduler_port` (`8786`), `lxplus_batch_name`,
  `lxplus_worker_image`, `lxplus_eos_scratch`, `lxplus_send_credential`. `--condor-site lpc|lxplus` forces a backend.
- **Condor client inside the container**: `run_container` binds `/etc/condor` and `/etc/sysconfig/ngbauth-submit`
  (not `/etc/krb5.conf`: the host file needs an unbound `includedir` and breaks Kerberos in the image), skips the
  host-only `myschedd` hook (`SKIP_LOCAL_CONFIG_FILE=TRUE`, explicit `_CONDOR_SCHEDD_HOST`/`_CONDOR_CREDD_HOST`),
  copies your Kerberos cache to `/tmp/$USER/krb5cc_barista`
  (it outlives the login session, unlike `/run/user/<uid>`) and stores the batch credential with
  `condor_store_cred` so `MY.SendCredential` jobs can be submitted from the container.
- **GPU jobs**: `./run_container classifier <cmd>` submits `<cmd>` as an HTCondor GPU job in the classifier
  image (`software/condor/submit_classifier_lxplus.sh`; knobs `CONDOR_JOB_FLAVOUR`, `CONDOR_REQUEST_GPUS`,
  `CONDOR_REQUEST_CPUS`, `CONDOR_REQUEST_MEMORY`, `CONDOR_GOOD_GPUS`, `CONDOR_INTERACTIVE`, `CONDOR_DRY_RUN`,
  `CONDOR_LOG_DIR`). Snakemake training workflows are auto-dispatched to the `lxplus_gpu` profile, which sends
  GPU rules to HTCondor through `software/snakemake/scripts/lxplus_condor_submit.py` and runs the rest locally.
- **Setup**: `./run_container lxplus-setup` (EOS scratch dirs, credential, interim `dask_lxplus` install when
  the image predates it) and `./run_container voms-proxy-init -voms cms -rfc --valid 168:00 -out proxy/x509_proxy`.
- **Kerberos lifetime**: the copied ticket is valid ~24 h. For longer shared-dask daemons renew it
  (`kinit -R` or a fresh `kinit`, then any `./run_container` call refreshes the copy); without a valid ticket the
  daemon cannot submit or remove workers. Spooled worker logs that did not reach EOS can be fetched with
  `condor_transfer_data <cluster>`.
- **Pixi**: installed under `/afs/cern.ch/work/<u>/<user>/.pixi` when the work volume exists (`BARISTA_PIXI_DIR` overrides).

### Local/Custom Environments

- **Flexible Binding**: Configurable mount points
- **CVMFS Support**: Optional access to distributed software

## Key Features

### Automatic Image Resolution

- **CVMFS Integration**: Uses unpacked images from `/cvmfs/unpacked.cern.ch` when available
- **Fallback**: Downloads from registry if CVMFS unavailable
- **Performance**: Faster startup with pre-cached images

### Workflow Integration

- **Snakemake Support**: Built-in workflow execution with proper container integration
- **Job Queue**: LPC-specific job submission with `lpcjobqueue`
- **Batch Processing**: Support for distributed computing systems


## Usage Examples

### Basic Analysis

```bash
# Start interactive session
./run_container

# Run analysis processor
./run_container python -m coffea4bees.analysis.processors.my_processor


### Statistical Analysis

```bash
# Combine tools
./run_container combine combine -M AsymptoticLimits datacard.txt

# Interactive combine session
./run_container combine
```

### Workflow Execution

```bash
# Run complete analysis pipeline
./run_container snakemake --snakefile workflows/analysis.smk --cores 8

# Test workflow
./run_container snakemake --snakefile workflows/test.smk --dry-run
```
