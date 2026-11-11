# Statistical Analysis Combine Snakemake Workflow

This directory contains the central Snakemake workflow (`combine.smk`) and helper utilities for running statistical analysis (limits, significance, goodness-of-fit, and nuisance impacts) using the CMS CombinedLimit tool.

## Execution Modes

There are two primary ways to run this workflow:

### 1. Direct Execution
Run Snakemake directly on the `combine.smk` file by passing your custom configuration YAML file. This mode generates and runs the entire standard analysis DAG automatically.

```bash
./run_container snakemake \
    --snakefile src/stat_analysis/combine.smk \
    --configfile path/to/your_config.yml \
    --cores 4
```

To preview the execution plan (dry-run):
```bash
./run_container snakemake \
    --snakefile src/stat_analysis/combine.smk \
    --configfile path/to/your_config.yml \
    -n
```

### 2. Modular Inclusion
Import the module into your custom analysis pipeline's Snakefile. This allows you to selectively instantiate, customize, and chain the rules:

```python
module combine:
    snakefile: "src/stat_analysis/combine.smk"
    config: config

# Example: Custom instantiation of the workspace rule
use rule workspace from combine with:
    input: "path/to/datacard.txt"
    output: "path/to/workspace.root"
```

---

## Example Configuration File

Save a file as `your_config.yml` (e.g. under `config/stats/`):

```yaml
# Output directory base path
output_path: "output/v4_systematics_test/HH4b/"

# Bypass impacts and GoF fits (run only limits/significance/scans)
stat_only: false

# Mass parameter passed to Combine commands
mass: "125"

# Likelihood scan settings
likelihood_scan_points: 50
likelihood_scan_split_size: 10

# Goodness of Fit settings
gof_algo: "saturated"
num_toy_jobs: 10
toys_per_job: 50

# Channel definitions mapping signals
channels:
  HH4b:
    signallabel: "ggHH_kl_1_kt_1_13p0TeV_hbbhbb"
    othersignal: "ggHH_kl_0_kt_1_13p0TeV_hbbhbb ggHH_kl_2p45_kt_1_13p0TeV_hbbhbb ggHH_kl_5_kt_1_13p0TeV_hbbhbb"
    signal: "GluGluToHHTo4B_cHHH1"
```

---

## Configuration Settings Summary

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stat_only` | `bool` | `False` | When `True`, automatically bypasses time-consuming Goodness-of-Fit (GoF) and Systematic Impacts fits. |
| `output_path` | `str` | `output/v4_systematics_test/HH4b/` | Base directory path where limits, scans, and fits will be output. |
| `channels` | `dict` | `{}` | A dictionary defining signals and channels. |

---

## Rule Parallelization Details

To optimize CPU utilization, the fits have been parallelized to run concurrently across HTCondor slots:
- **Fit Diagnostics** is split into parallel B-only (`fit_diagnostics_bonly`) and S+B (`fit_diagnostics_sb`) fit rules.
- **Impacts** is split into `impacts_initial_fit` (initial fit), `impacts_do_fits` (systematic fits run in parallel on HTCondor), and `impacts_collect` (a localrule that merges fits and plots the final PDF).

---

## Standalone Plotters & Analysis Tools

All plotting routines and datacard inspection utilities run standalone within the Combine container with zero external dependencies on `law`:

### Publication Plotters (`src/stat_analysis/plots/`)
- `plot_impacts.py`: Nuisance impacts and pulls with tight zero-margin PNG rendering.
- `plot_scan.py`: 1D profile likelihood scans with cubic spline interpolation and $1\sigma, 2\sigma, 3\sigma, 5\sigma$ crossing intervals.
- `plot_scan_2d.py`: 2D likelihood contour scans ($68\%$ and $95\%$ CL contours, SM and best-fit markers).
- `plot_limits.py`: Brazil band upper limit plots directly from Combine limit JSON.
- `plot_limit_points.py`: Summary limit comparison points across channels, eras, or categories.
- `plot_gof.py`: Goodness-of-fit toy distributions with empirical $p$-value calculation.
- `plot_s_over_b.py`: Aggregated post-fit $S/\sqrt{B}$ (or $S/B$) summary distribution.
- `plot_correlation.py`: Post-fit nuisance correlation matrix heatmap.

### Datacard Manipulation Tools (`src/stat_analysis/tools/`)
- `prettify_datacard.py`: Aligns datacards into clean, human-readable columnar format.
- `yield_table.py`: Generates Markdown, ASCII, or LaTeX yield tables from datacards and shapes.
- `bundle_datacard.py`: Copies a datacard and its referenced ROOT shapes into a self-contained folder.
- `remove_empty_processes.py`: Prunes empty/negligible processes from datacards to stabilize fits.
- `rename_parameters.py`: Renames nuisance parameters across datacard text and shape files.
- `extract_fit_result.py`: Dumps fit values, errors, and covariance/correlation matrices to JSON.
- `web/view_cov_json.html`: Interactive browser viewer for post-fit correlation matrices.
- `web/view_datacard.html`: Interactive browser viewer for exploring datacards.

---

## Rule Graph

Below is the dependency graph of the workflow rules:

![Rule Graph](rulegraph.png)
