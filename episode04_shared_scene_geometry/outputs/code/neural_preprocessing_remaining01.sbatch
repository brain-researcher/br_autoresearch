#!/bin/bash
#SBATCH --job-name=ep04_preprocess_remaining
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:35:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/neural_preprocessing/20261003-remaining-folds01/logs/slurm-%j.out
#SBATCH --error=outputs/runs/neural_preprocessing/20261003-remaining-folds01/logs/slurm-%j.err

set -euo pipefail
hostname
printf 'SLURM_JOB_ID=%s SLURM_CPUS_PER_TASK=%s\n' "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}"
cd "${OAK:?}/users/zijiao/br_autoresearch/episode04_shared_scene_geometry"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec python3.12 outputs/code/preprocess_development_neural_remaining01.py
