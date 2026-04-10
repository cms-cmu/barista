# Generic Classifier Training Workflow

A single Snakemake workflow for training, evaluating, and analyzing HCR classifiers.
Each model is described by a `workflow_config.yml` file — no edits to the Snakefile needed.

## DAG

```
train → evaluate        (GPU, ~4h)
      → analyze         (CPU, ~1h)
      → plot_inputs_raw      (CPU, optional)
      → plot_inputs_dataprep (CPU, optional)
```

`analyze` and `evaluate` run in parallel after `train` completes.
`plot_inputs_*` rules are only scheduled when `plot_inputs: true` in the config.

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

Each `workflow_config.yml` describes one model. Example:

```yaml
# Identity
lpc_user: algomez               # LPC username, used to build eos_base
cern_user: a/algomez            # CERN username (first-letter/username), used for EOS www plots

# Paths — {lpc_user} and {eos_base} are resolved automatically
eos_base: "root://cmseos.fnal.gov//store/user/{lpc_user}/XX4b/2024_v2"

# Workflow settings
classifier_config_paths: coffea4bees   # passed as CLASSIFIER_CONFIG_PATHS env var to pyml.py
wfs_base: coffea4bees/classifier/config/workflows/HH4b_2024_v2_lowpt/FvT
                                       # directory containing train.yml, evaluate.yml, and ../common.yml
output_dir: output/lowpt_FvT/          # where flag files and logs are written; change this to keep outputs from separate runs

# Optional features
plot_inputs: true   # set to false (or omit) to skip plot_inputs_raw and plot_inputs_dataprep

# Model paths — {eos_base} placeholder is resolved
model:          "{eos_base}/classifier/FvT_lowpt"
friend:         "{eos_base}/friend/FvT_lowpt"

# Template strings passed to: ./src/pyml.py template "{<template>}" train.yml / evaluate.yml
train_template: "model: {eos_base}/classifier/FvT_lowpt"
eval_template:  "model: {eos_base}/classifier/FvT_lowpt, FvT: {eos_base}/friend/FvT_lowpt"

# Path to classifier inputs JSON (only needed when plot_inputs: true)
metadata: "coffea4bees/metadata/datasets_HH4b_Run2/2024_v2/classifier_inputs_lowpt_wlowptJCM.json@@HCR_input_lowpt"
```

### Required keys

| Key | Description |
|-----|-------------|
| `lpc_user` | LPC username |
| `cern_user` | CERN username in `x/username` format |
| `eos_base` | Base EOS path (may use `{lpc_user}`) |
| `classifier_config_paths` | Value for `CLASSIFIER_CONFIG_PATHS` env var |
| `wfs_base` | Directory with `train.yml`, `evaluate.yml`, and `../common.yml` |
| `output_dir` | Output directory for logs and flag files |
| `model` | Path to model output directory |
| `train_template` | Template string for training |
| `eval_template` | Template string for evaluation |

### Optional keys

| Key | Default | Description |
|-----|---------|-------------|
| `plot_inputs` | `false` | Whether to run input feature plots after training |
| `metadata` | `""` | Classifier inputs JSON path, required when `plot_inputs: true` |
| `friend` | — | Friend tree path (used in templates, not directly by the Snakefile) |

## Example configs

Ready-to-use configs live alongside their `train.yml` / `evaluate.yml`:

- `coffea4bees/classifier/config/workflows/HH4b_2024_v2_lowpt/FvT/workflow_config.yml`
- `coffea4bees/classifier/config/workflows/HH4b_2024_v2_lowpt/SvB/workflow_config.yml`
- `coffea4bees/classifier/config/workflows/HH4b_Run3/MvD/workflow_config.yml`

To run a separate job without overwriting existing outputs, copy a config and change `output_dir`.
