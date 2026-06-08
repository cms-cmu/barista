# Generic Classifier Training Workflow

A single Snakemake workflow for training, evaluating, and analyzing HCR classifiers.
Each model is described by a `workflow_config.yml` file — no edits to the Snakefile needed.

## DAG

```
train → evaluate        (GPU, ~4h)
      → analyze         (CPU, ~1h)
      → plot_inputs_raw      (CPU, optional)
      → plot_inputs_dataprep (CPU, optional)
      → plot_weights         (CPU, optional)
```

`analyze` and `evaluate` run in parallel after `train` completes.
`plot_inputs_*` and `plot_weights` rules are only scheduled when `plot_inputs: true` or `plot_weights: true` respectively in the config.

## How to run

From the barista root directory:

```bash
## open a tmux session
./run_container snakemake \
    --snakefile src/classifier/workflow/Snakefile \
    --configfile <path/to/workflow_config.yml> 
    --cores 1 \
    ## --logger snkmt # optional: if you want a dashboard
```

Dry-run to preview the DAG without executing:

```bash
./run_container snakemake \
    --snakefile src/classifier/workflow/Snakefile \
    --configfile <path/to/workflow_config.yml> \
    -np
```

Run a single rule (e.g. only train):

```bash
./run_container snakemake \
    --snakefile src/classifier/workflow/Snakefile \
    --configfile <path/to/workflow_config.yml> \
    --cores 1 \
    output/my_model/train.done
```

If you want to see the dashboard, add `--logger snkmt` to the above commands and then outside your tmux session, run:

```bash
pixi run snkmt console
```

## Config file schema

Each `workflow_config.yml` describes one model. The `{eos_base}` and `{label}` placeholders
are resolved by the Snakefile and can be used in `output_dir`, `model`, `friend`,
`train_template`, and `eval_template`.

Example:

```yaml
# Paths
eos_base:  "root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2"
plot_base: "root://eosuser.cern.ch//eos/user/a/algomez/www/HH4b/classifier_plots"

# Workflow settings
classifier_config_paths: coffea4bees   # passed as CLASSIFIER_CONFIG_PATHS env var to pyml.py
wfs_base: coffea4bees/classifier/config/workflows/HH4b_2024_v2_lowpt/FvT
                                       # directory containing train.yml, evaluate.yml, and ../common.yml
label: lowpt_FvT                       # identifies this run; used in output paths and plot directories
output_dir: output/{label}/            # where flag files and logs are written

# Optional features
plot_inputs: true   # set to false (or omit) to skip plot_inputs_raw and plot_inputs_dataprep
plot_weights: true  # set to false (or omit) to skip plot_weights event diagnostics

# Model paths — {eos_base} and {label} placeholders are resolved
model:  "{eos_base}/classifier/{label}"
friend: "{eos_base}/friend/{label}"

# Template strings passed to: ./src/pyml.py template "{<template>}" train.yml / evaluate.yml
train_template: "model: {eos_base}/classifier/{label}"
eval_template:  "model: {eos_base}/classifier/{label}, FvT: {eos_base}/friend/{label}"

# Path to classifier inputs JSON (only needed when plot_inputs: true)
metadata: "coffea4bees/metadata/datasets_HH4b_Run2/2024_v2/classifier_inputs_lowpt_wlowptJCM.json@@HCR_input_lowpt"
```

Plots are written to `{plot_base}/{DATE}_{label}/analyze`, `.../inputs_raw`, and `.../inputs_dataprep`.
To keep outputs from a separate run, change `label` (and optionally `output_dir`).

### Required keys

| Key | Description |
| --- | --- |
| `eos_base` | Base EOS path for model inputs/outputs |
| `plot_base` | Base EOS path for plots |
| `label` | Short identifier for this run (used in output paths and plot dirs) |
| `classifier_config_paths` | Value for `CLASSIFIER_CONFIG_PATHS` env var |
| `wfs_base` | Directory with `train.yml`, `evaluate.yml`, and `../common.yml` |
| `output_dir` | Output directory for logs and flag files |
| `model` | Path to model output directory |
| `train_template` | Template string for training |
| `eval_template` | Template string for evaluation |

### Optional keys

| Key | Default | Description |
| --- | --- | --- |
| `plot_inputs` | `false` | Whether to run input feature plots after training |
| `plot_weights` | `false` | Whether to run event weight diagnostic plotting and summary |
| `metadata` | `""` | Classifier inputs JSON path, required when `plot_inputs: true` or `plot_weights: true` |
| `friend` | — | Friend tree path (used in templates, not directly by the Snakefile) |

## Example configs

Ready-to-use configs live alongside their `train.yml` / `evaluate.yml`:

- `coffea4bees/classifier/config/workflows/HH4b_2024_v2_lowpt/FvT/workflow_config.yml`
- `coffea4bees/classifier/config/workflows/HH4b_2024_v2_lowpt/SvB/workflow_config.yml`
- `coffea4bees/classifier/config/workflows/HH4b_Run3/MvD/workflow_config.yml`

## Event Weight Diagnostics

The `plot_weights` rule runs the `plot_weights.py` script. It automatically parses `train.yml` (using `--wfs-base` and template parameters) to load any FvT friend trees and JCM weights, applies them, executes the outlier/negative-weight removal filter, and computes statistics and diagnostic plots.

Here is how to interpret the outputs:

### 1. The Summary Table (`weight_stats.md`)
* **Effective Sample Size ($N_{\text{eff}}$)**: Computed as $N_{\text{eff}} = \frac{(\sum w)^2}{\sum w^2}$. The ratio $N_{\text{eff}} / N_{\text{raw}}$ measures weight uniformity. A ratio close to 100% means weights are flat. A ratio below 10% - 20% indicates that a tiny fraction of events dominates the loss and training gradients, leading to severe overfitting.
* **Negative Weights ($Min < 0$)**: Outlier removal automatically filters out negative weights (`weight < 0`) during dataset loading to prevent gradient direction flipping. Confirm that `Min >= 0` for all samples in this table.
* **Extreme Weights ($Max \gg Mean$)**: The outlier removal clips weights $\ge 1.0$. Confirm that `Max < 1.0` for all samples.

### 2. Weight Distributions (`weights_dist_<label>.png`)
* **Tails**: Look for long, heavy right-hand tails (plotted in log scale) of events with very high weights. These can cause gradient dominance.
* **Zero Spikes**: A high population at zero means those events do not contribute to gradients, resulting in wasted training time.

### 3. Kinematic Profile Plots
The script outputs three profile plots of average weight vs. kinematics:
* **Leading Jet $p_T$ (`profile_pt1_<label>.png`)**: Check for weight slopes and fluctuations in the leading jet $p_T$ spectrum.
* **4th Leading Jet $p_T$ (`profile_pt4_<label>.png`)**: Particularly important in the low-$p_T$ phase space to check for weight explosions or high uncertainties at the turn-on threshold.
* **Selected Jet Count (`profile_njets_<label>.png`)**: Check for spikes at specific jet multiplicities (e.g. ggF signal spikes due to b-tagging Scale Factor products).
* **Uncertainties**: Large error bars show statistical fluctuations in the weights. Smooth profiles with small error bars are desired.


