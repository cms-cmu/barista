#!/bin/bash
#SBATCH -p RM-shared
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=7800M
#SBATCH -A phy260026p
#SBATCH -J fvt_c_bkg_syst
#SBATCH -o logs_bkg_syst_C.log
#SBATCH -e logs_bkg_syst_C.log

set -eo pipefail

cd /ocean/projects/phy260026p/tgomezes/HH4b/barista-ttHbb-bkg-syst

# Ensure proxy is set and has correct permissions
export X509_USER_PROXY="$PWD/proxy/x509_proxy"
export PATH="$PWD/bin:$PATH"
chmod 600 "$X509_USER_PROXY"

echo "=== Starting Stage C Pipeline on Bridges-2 at $(date) ==="
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Proxy: $X509_USER_PROXY"

# Clear any previous stale directory locks
~/.pixi/bin/pixi run snakemake \
    -s coffea4bees/workflows/Snakefile_bkg_syst_C.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml \
    --unlock

# Run main workflow with mtime rerun-triggers
~/.pixi/bin/pixi run snakemake \
    --profile software/snakemake/profiles/bridges2 \
    -s coffea4bees/workflows/Snakefile_bkg_syst_C.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml \
    --rerun-triggers mtime \
    --jobs 32 \
    --keep-going \
    all_bkg_syst_C

echo "=== Finished Stage C at $(date) with exit code $? ==="
