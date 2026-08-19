#!/bin/bash
#SBATCH --job-name=ep02_narps_fit
#SBATCH --partition=russpold
#SBATCH --array=0-107%16
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/zijiao/episode02_narps_analysis/logs/%A_%a.out
#SBATCH --error=/scratch/users/zijiao/episode02_narps_analysis/logs/%A_%a.err

set -euo pipefail
module load python/3.12.1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd /oak/stanford/groups/russpold/users/zijiao/autoresearch/episode02_narps_analysis
/home/users/zijiao/pcvenv/bin/python outputs/code/fit_subject.py "${SLURM_ARRAY_TASK_ID}"
