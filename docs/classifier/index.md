# Classifier

## Folder Structure

- `classifier/`: main package for classifier
    - Machine Learning
        - `ml/`: high-level ML workflows and utilities
        - `nn/`: neural network models
    - [Task System](task.md)
        - `task/`: task protocols and command-line interface
        - `config/`: task configurations
        - `test/`: task configurations for testing
    - [Monitor System](monitor.md)
        - `monitor/`: monitor core and components
    - Others
        - `data/`: model data archives
        - `algorithm/`: algorithms implemented with `torch.Tensor`
        - `compatibility/`: 4b analysis related modules
        - `root/`: `ROOT` I/O utilities
        - `df/`: `pd.DataFrame` utilities
        - `process/`: multiprocessing utilities
        - `patch/`: unreleased critical bug fixes
- `pyml.py`: run the classifier jobs, can be used as an executable.

## Getting Started

### Setup Environment

!!! note

    You are assumed to be in the `barista/` directory to run the following commands.

!!! warning

    All of the `./pyml.py` should be replaced by `./src/pyml.py`

#### Container Setup and Usage

The classifier runs in dedicated Apptainer containers with different modes for GPU and CPU workloads.

##### Container Types

- **GPU Container** (`classifier`): For training and GPU-accelerated inference
- **CPU Container** (`classifier_cpu`): For data processing, evaluation, and CPU-only tasks

##### Cluster-Specific Behavior & Container Resolution

- **FALCON cluster (`falcon.phys.cmu.edu`)**:
  - Automatically resolves unpacked container images directly from CVMFS (`/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest`).
  - Snakemake orchestrator runs on the login node and uses `snakemake-executor-plugin-slurm` to submit GPU and CPU rules to SLURM queues.
- **PSC Bridges-2 cluster (`bridges2.psc.edu`)**:
  - Since `/cvmfs` is not mounted on Bridges-2, containers resolve to pre-extracted SIF images (`/ocean/projects/phy260026p/shared/images/barista_latest.sif`) or Apptainer image cache.
  - Executing `./run_container snakemake ...` on the login node automatically allocates an interactive CPU session (`srun` on `RM-shared`) for the Snakemake controller, which then submits worker jobs to SLURM.

#### Running Classifier Snakemake Workflows

##### On FALCON

```bash
./run_container snakemake \
    --profile software/snakemake/profiles/falcon \
    --snakefile src/classifier/workflow/Snakefile \
    --configfile coffea4bees/classifier/config/workflows/HH4b_2024_v2/SvB/workflow_config.yml \
    --jobs 5 \
    --use-apptainer -p
```

##### On PSC Bridges-2

```bash
./run_container snakemake \
    --profile software/snakemake/profiles/bridges2 \
    --snakefile src/classifier/workflow/Snakefile \
    --configfile coffea4bees/classifier/config/workflows/HH4b_2024_v2/SvB/workflow_config.yml \
    --jobs 5 \
    --use-apptainer -p
```

!!! note "PSC Bridges-2 Resource Allocation Rules"
    - **GPU partition (`GPU-shared`)**: Maximum memory per allocated GPU is **22,750 MB (22.75 GB)**. Ensure `mem_mb` in `bridges2` profile is $\le 20000$.
    - **CPU partition (`RM-shared`)**: Enforces a strict **2,000 MB (2 GB) memory per core** limit. CPU rules (e.g., input plotting) must not request more memory per core than allocated (e.g. `mem_mb: 3800` for 2 CPUs).

#### Slurm Behavior and Job Monitoring

* To check status of jobs in queue: `squeue --me`
* To monitor logs of submitted jobs:

```bash
tail -f slurm_logs/rule_train/<job_id>.log
```

## Command-line Interface

See the [Task System](task.md) for details.

### Setup Auto-completion

To register the auto-completion for the current shell session, run the following command:

```bash
source classifier/install.sh
```

To unregister the auto-completion, run:

```bash
source classifier/uninstall.sh
```

The auto-completion will be triggered when the command starts with `./pyml.py` and the `<tab>` key is pressed. It will dynamically search for available tasks in the `classifier/config` directory and hint for the task name or the arguments.

![Auto-completion example](./images/autocompletion.gif)

### Help

Use the following command to print help for all tasks:

```bash
./pyml.py help --all
```

## Training and Evaluation

See the [HCR Training](hcr.md) for a complete example to train and evaluate a HCR model for SvB and FvT.

## Monitor

A monitor is provided to collect logs, progresses, resource metrics and other information from worker processes/nodes. See the [Monitor System](monitor.md) for details.
