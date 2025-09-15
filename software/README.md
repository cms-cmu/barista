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
