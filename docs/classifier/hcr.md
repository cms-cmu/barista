# Hierarchical Combinatoric ResNet (HCR)

This tutorial will work through a complete example of training baseline FvT and SvB classifier for HH4b analysis using skim datasets_HH4b_2024_v2 on rogue.

## Setup environment

- (optional) setup apptaine cache and temp directory
- start a container and enter `/coffea4bees/` directory
- (optional) setup the base path for workflow files

```bash
export WFS="classifier/config/workflows/HH4b_2024_v2"
```

See [Overview](index.md#setup-environment) for details.

## FvT Training and Evaluation

- set the following variables in the `${WFS}/FvT/run.sh` script:
    - `MODEL`: the base path to store the FvT model
    - `FvT`: the base path to store the FvT friend trees
    - `PLOT`: the base path to store benchmark plots
    - all other variables are optional
- run the following command:

```bash
source ${WFS}/FvT/run.sh
```

To understand the details of the whole workflow, check the comments in the following files by order:

- `${WFS}/FvT/train.yml`
- `${WFS}/FvT/evaluate.yml`
- `${WFS}/common.yml`
- `${WFS}/FvT/run.sh`

## SvB Training and Evaluation

- set the following variables in the `${WFS}/SvB/run.sh` script:
    - `MODEL`: the base path to store the SvB models
    - `SvB`: the base path to store the SvB friend trees
    - `FvT`: the base path to the FvT friend trees (should be the same as the FvT training)
    - `PLOT`: the base path to store benchmark plots
    - all other variables are optional
- run the following command:

```bash
source ${WFS}/SvB/run.sh
```

To understand the details of the whole workflow, check the comments in the following files by order (assuming you have already checked the FvT config files):

- `${WFS}/SvB/train.yml`
- `${WFS}/SvB/evaluate.yml` (basically the same as the FvT evaluation)
- `${WFS}/SvB/run.sh`

## Plotting

- make a local copy of config `analysis_dask/config/userdata.cfg.yml` and fill all required fields
- make a local copy of config `analysis_dask/config/classifier_plot_vars.cfg.yml` and change the SvB and FvT friend tree paths in `classifier_outputs<var>` according to the evaluation scripts and modify the `classifier_datasets<var>` to match the datasets you want to plot.
- run the following command:

```bash
python dask_run.py \
    analysis_dask/config/userdata.local.cfg.yml \
    analysis_dask/config/cluster.cfg.yml#rogue_local_huge \ 
    analysis_dask/config/classifier_plot_vars.local.cfg.yml#2024_v2 \
    analysis_dask/config/classifier_plot.cfg.yml#2024_v2
```

- the output will be available as `{output_dir}/classifier_plot_2024_v2_{timestamp}/hists/classifier_basic.coffea`

See [Histogram](histogram.md#classifier-plot-configurations) for details.

## Tips on Performance

- Training:
    - in main task `train`, consider increasing `--max-trainers` to parallel multiple models (CPU, GPU, memory bounded)
    - in `-dataset HCR.*`, consider increasing `--max-workers` (IO and CPU bounded, require extra memory)
    - in `-setting ml.DataLoader`
        - always set `optimize_sliceable_dataset` to `True` if the dataset fits in memory. This option enables a custom data loader that makes use of `torch`'s c++ based parallel slicing, which is significantly faster and more memory efficient than the default `torch.utils.data.DataLoader`.
        - if `optimize_sliceable_dataset` is disabled, consider increasing `num_workers` to speed up batch generation (mainly CPU bounded, require extra memory)
        - consider increasing `batch_eval` to speed up evaluation (mainly GPU memory bounded)
    - in `-setting torch.Training`, consider using `disable_benchmark` to skip all benchmark steps.
- Evaluation:
    - in main task `evaluate`, consider increasing `--max-evaluators` to parallel multiple models (CPU, GPU, memory bounded)
    - in `-setting torch.DataLoader`, consider increasing `num_workers` and `batch_eval`. (IO and CPU bounded, require extra memory)
- Merging k-folds:
    - in `-analysis kfold.Merge`,
        - consider increasing `--workers` (IO and CPU bounded, require extra memory)
        - consider using a finite `--step` to split root files into smaller chunks.
