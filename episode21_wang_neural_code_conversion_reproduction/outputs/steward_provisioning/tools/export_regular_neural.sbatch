#!/usr/bin/env bash
# Prepared allocation proposal; do not submit without the separate approval record.
#SBATCH --job-name=ep21_regular_neural_export
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00

set -euo pipefail
source /share/software/user/open/lmod/lmod/init/bash
module purge >/dev/null 2>&1
module load devel math python/3.12.1 py-numpy/2.2.6_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_episode="$OAK/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
task_runtime="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$SCRATCH/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/neural-export-$SLURM_JOB_ID/cache"
cd "$task_episode"
exec "$task_runtime/venv/bin/python" -B outputs/steward_provisioning/tools/export_regular_neural.py
