#!/usr/bin/env bash
# Approved combined public/lock prelude and regular-only feature extraction.
#SBATCH --job-name=ep21_regular_vgg19
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -G 1
#SBATCH --time=08:00:00
set -euo pipefail
source /share/software/user/open/lmod/lmod/init/bash
module purge >/dev/null 2>&1
module load devel math viz python/3.12.1 py-numpy/1.26.3_py312
module load py-h5py/3.10.0_py312 py-pillow/10.2.0_py312 py-pytorch/2.4.1_py312
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
task_episode="$OAK/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
# Feature-owned cache/tmp are created after validating its real context.
cd "$task_episode"
task_runtime="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules"
# Use the same interpreter/dependencies for public conformance and features.
exec "$task_runtime/venv/bin/python" -B outputs/steward_provisioning/tools/export_regular_vgg19.py "$@"
