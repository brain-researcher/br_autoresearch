#!/bin/bash
#SBATCH --job-name=ep02_narps_aggregate
#SBATCH --partition=russpold
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/zijiao/episode02_narps_analysis/logs/%j_aggregate.out
#SBATCH --error=/scratch/users/zijiao/episode02_narps_analysis/logs/%j_aggregate.err

set -euo pipefail
module load python/3.12.1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd /oak/stanford/groups/russpold/users/zijiao/autoresearch/episode02_narps_analysis
/home/users/zijiao/pcvenv/bin/python outputs/code/aggregate_results.py
