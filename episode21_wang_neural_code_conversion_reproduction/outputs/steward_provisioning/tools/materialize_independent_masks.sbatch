#!/usr/bin/env bash
#SBATCH --job-name=ep21-native-masks
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=outputs/steward_provisioning/masks-%j.log
set -euo pipefail
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel math viz python/3.12.1 py-numpy/1.26.3_py312 py-matplotlib/3.8.3_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$SCRATCH/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/independent-masks-$SLURM_JOB_ID/mpl-cache"
task_astra_runtime="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules"
"$task_astra_runtime/venv/bin/python" -B outputs/steward_provisioning/tools/materialize_independent_masks.py
