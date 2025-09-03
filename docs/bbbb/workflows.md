#  workflows

## List of workflows

This is a list of the workflows using Snakemake that we are using:

 * Snakefile_reana: The workflow we use to launch the reana jobs.
 * Snakefile_binoptimization: workflow to test the bin optimization of the SvB variable.

## Useful commands

Snakemake workflows can be launch using the [run_snakemake](../run_snakemake) script, which internally using a snakemake container, then you can use it as:

```
cd python/
./run_snakemake --snakefile workflows/Snakefile_binoptimization 
```

Useful snakemake arguments: 
    * `--dry-run` checks if all the rules are created correctly. Try to always run it as a test.
    * `--printshellcmds` along dry-run, it prints all the commands that the snakefile will execute
    * `--rerun-incomplete` to run partially workflows
    * `--force` in case some outputs are present, it will replace them by running the rules.
    * `--cores X` specifies the number of cores used.
    * `--use-singularity` if the rules uses a container

