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

    You are assumed to be in the `/python/` directory to run the following commands.

#### Use Container (Recommended)

!!! warning

    You may need to change the apptainer cache and temp directory before pulling any image, especially when the home directory has a limited quota. The directories are controlled by the following environment variables:
    ```bash

    export APPTAINER_CACHEDIR=
    export APPTAINER_TMPDIR=
    ```

The docker image is available as:

- `docker://chuyuanliu/heptools:ml`
- `/cvmfs/unpacked.cern.ch/registry.hub.docker.com/chuyuanliu/heptools:ml` (only when CVMFS is available)

The image is built from the following configurations:

- [`base.Dockerfile`](https://github.com/chuyuanliu/heptools/blob/be5a3122dd506a3899d1a69bf48770e1569bfeed/docker/base.Dockerfile): base image
- [`ml.Dockerfile`](https://github.com/chuyuanliu/heptools/blob/be5a3122dd506a3899d1a69bf48770e1569bfeed/docker/ml.Dockerfile): ml image derived from base image
- [`base.yml`](https://github.com/chuyuanliu/heptools/blob/be5a3122dd506a3899d1a69bf48770e1569bfeed/docker/base.yml): used by `base.Dockerfile`
- [`base-linux.yml`](https://github.com/chuyuanliu/heptools/blob/be5a3122dd506a3899d1a69bf48770e1569bfeed/docker/base-linux.yml): used by `base.Dockerfile`
- [`ml.yml`](https://github.com/chuyuanliu/heptools/blob/be5a3122dd506a3899d1a69bf48770e1569bfeed/docker/ml.yml): used by `ml.Dockerfile`

Run the following command to start an interactive shell:

```bash
apptainer exec \
    -B .:/srv \
    --nv \
    --pwd /srv \
    docker://chuyuanliu/heptools:ml \
    bash --init-file /entrypoint.sh
```

where:

- `-B .:/srv` mount the current directory to `/srv`
- `--nv` enable GPU
- `--pwd /srv` equivalent to `cd /srv` when starting the container
- `bash --init-file /entrypoint.sh` (**important**) start a bash shell and run the initialization script.

#### Use Conda

The conda environment can be created from the `base.yml`, `base-linux.yml` and `ml.yml` files listed above.
`classifier/env.yml` is deprecated and not actively maintained.

#### `rogue01/rogue02` specific

- change the cache and temp directory for apptainer:

    1. `mkdir -p /mnt/scratch/${USER}/.apptainer`
    2. add the following to `~/.bashrc`

        ```bash
        export APPTAINER_TMPDIR=/mnt/scratch/${USER}/.apptainer/
        export APPTAINER_CACHEDIR=/mnt/scratch/${USER}/.apptainer/
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
