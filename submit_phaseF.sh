#!/bin/bash
#SBATCH --job-name=phaseF_ttHbb
#SBATCH --partition=RM-shared
#SBATCH --account=phy260026p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/phaseF_%j.log
#SBATCH --error=slurm_logs/phaseF_%j.log

cd /ocean/projects/phy260026p/tgomezes/HH4b/barista
./run_container snakemake --profile software/snakemake/profiles/bridges2 -s coffea4bees/workflows/Snakefile_PhaseF.smk --configfile coffea4bees/workflows/config/analysis_ttHbb_Run3.yml --jobs 4 --latency-wait 60 --rerun-incomplete --rerun-triggers mtime
