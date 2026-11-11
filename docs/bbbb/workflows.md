#  workflows

## List of workflows

This is a list of the workflows using Snakemake that we are using:

 * `Snakefile_full_syst`: Master coordinator workflow for the full systematics pipeline (Stage F.1 + Stage F.2 + remote EOS synchronization).
 * `Snakefile_full_syst_1_analysis`: Stage F.1 signal systematic variations (`others` friend weights, `jes` PyTorch kinematic shifts), singlefile generation, merging, nominal integrity validation against upstream stat-only, and systematic cutflow table extraction.
 * `Snakefile_full_syst_2_stats`: Stage F.2 Combine datacard creation integrating signal systematics and background closure models (`bkg_syst_file`), shape variation plotting (`make_syst_plots.py`), interactive HTML gallery (`make_gallery.py`), and Combine statistical inference (limits, significance, scans, postfit, and impacts).
 * `Snakefile_bkg_syst`: Complete background systematics workflow (Mixed Data Creation, Subsamples, JCM, FvT Neural Networks, Closure & Stats).
 * `Snakefile_reana`: The workflow we use to launch the reana jobs.
 * `Snakefile_binoptimization`: workflow to test the bin optimization of the SvB variable.

## Useful commands

Snakemake workflows can be launch using the [run_snakemake](../run_snakemake) script, which internally using a snakemake container, then you can use it as:

```
cd coffea4bees/
./run_snakemake --snakefile workflows/Snakefile_binoptimization 
```

Useful snakemake arguments: 
    * `--dry-run` checks if all the rules are created correctly. Try to always run it as a test.
    * `--printshellcmds` along dry-run, it prints all the commands that the snakefile will execute
    * `--rerun-incomplete` to run partially workflows
    * `--force` in case some outputs are present, it will replace them by running the rules.
    * `--cores X` specifies the number of cores used.
    * `--use-singularity` if the rules uses a container

