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

#### Container Setup and Usage

The classifier runs in dedicated Apptainer containers with different modes for GPU and CPU workloads.

##### Container Types

- **GPU Container** (`classifier`): For training and GPU-accelerated inference
- **CPU Container** (`classifier_cpu`): For data processing, evaluation, and CPU-only tasks

##### Cluster-Specific Behavior

**On FALCON cluster:**

- All jobs with commands automatically submit to SLURM batch queue
- Interactive shells (CPU only) launch in an interactive SLURM session (`srun`)

**On other clusters (LPC, lxplus, etc.):**

- Jobs run locally in the container without SLURM submission

##### Usage Examples

**GPU Container**

```bash
# Run training script (submits to SLURM on FALCON, runs locally elsewhere)
./run_container classifier python train_model.py

# GPU interactive mode is NOT supported (security restriction)
```

!!! warning "GPU Interactive Mode"
    Interactive shells are disabled for the GPU container to prevent accidental resource consumption on GPU nodes.

**CPU Container**

```bash
# Run processing script (submits to SLURM on FALCON, runs locally elsewhere)
./run_container classifier_cpu python process_data.py

# Interactive shell on FALCON (auto-submits interactive SLURM job)
./run_container classifier_cpu

# Interactive shell on other clusters (opens immediately)
./run_container classifier_cpu
```

!!! tip "Checking Interactive Mode"
    When in an interactive SLURM session on FALCON, check the job ID:
    ```bash
    echo $SLURM_JOB_ID  # Non-empty = you're in a SLURM job
    ```

#### Slurm Behavior and Job Monitoring (on Falcon)

* to check status of the job, use squeue
* the slurm configuration is inside `barista/software/slurm/slurm.conf` if you need to request more resources
* to check the progress of the submitted job:

```
tail -f slurm_logs/classifier_batch_<job_id>.out
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

## Histogram

The histogramming is handled by `dask` processors for better performance and compatibility. See the [Histogram](histogram.md) for details.
