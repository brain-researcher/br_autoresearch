#!/bin/bash
#SBATCH --job-name=ep11-mixed56
#SBATCH --partition=russpold
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode11_projection_types_vs_gradients/mixed_support_56d/slurm-%j.out

set -euo pipefail

EP11_REPO_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch
EP11_EPISODE_ROOT=${EP11_REPO_ROOT}/episode11_projection_types_vs_gradients
EP11_RUN_ROOT=${SCRATCH}/br_autoresearch/episode11_projection_types_vs_gradients/mixed_support_56d
EP11_RUN_DIR=${EP11_RUN_ROOT}/run_${SLURM_JOB_ID}
EP11_PYTHON=${SCRATCH}/br_autoresearch/episode11_projection_types_vs_gradients/ccf_env_py311/bin/python

mkdir -p "${EP11_RUN_DIR}"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/tools/run_mixed_support_56d_qualification.py" "${EP11_RUN_DIR}/"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/whole_cortex_ssptr_inventory.csv" "${EP11_RUN_DIR}/"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/allen_target_vocabulary.csv" "${EP11_RUN_DIR}/"

"${EP11_PYTHON}" \
  "${EP11_RUN_DIR}/run_mixed_support_56d_qualification.py" \
  --inventory "${EP11_RUN_DIR}/whole_cortex_ssptr_inventory.csv" \
  --vocabulary "${EP11_RUN_DIR}/allen_target_vocabulary.csv" \
  --output-dir "${EP11_RUN_DIR}/results" \
  --replicates 20 \
  --target-margin-multipliers 0,0.5,1.0,1.5,2.0 \
  --calibration-amplitude-grid 0,0.025,0.05,0.075,0.10,0.15,0.20,0.30,0.45,0.65,0.90,1.25,1.75,2.50 \
  --calibration-replicates 8 \
  --seed 20260926 \
  --margin 0.05

echo "EP11_MIXED_SUPPORT_56D_RUN_DIR=${EP11_RUN_DIR}"
