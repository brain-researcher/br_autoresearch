#!/bin/bash
#SBATCH --job-name=ep11-group-module
#SBATCH --partition=russpold
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode11_projection_types_vs_gradients/group_module_ablation/slurm-%j.out

set -euo pipefail

EP11_REPO_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch
EP11_EPISODE_ROOT=${EP11_REPO_ROOT}/episode11_projection_types_vs_gradients
EP11_RUN_ROOT=${SCRATCH}/br_autoresearch/episode11_projection_types_vs_gradients/group_module_ablation
EP11_RUN_DIR=${EP11_RUN_ROOT}/run_${SLURM_JOB_ID}

ml purge
ml load devel math
ml load python/3.12.1 py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-pandas/2.2.1_py312

mkdir -p "${EP11_RUN_DIR}"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/tools/run_mixed_support_56d_qualification.py" "${EP11_RUN_DIR}/"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/tools/run_mixed_support_design_sensitivity.py" "${EP11_RUN_DIR}/"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/tools/run_mixed_support_group_module_ablation.py" "${EP11_RUN_DIR}/"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/whole_cortex_ssptr_inventory.csv" "${EP11_RUN_DIR}/"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/allen_target_vocabulary.csv" "${EP11_RUN_DIR}/"

python3 "${EP11_RUN_DIR}/run_mixed_support_56d_qualification.py" \
  --inventory "${EP11_RUN_DIR}/whole_cortex_ssptr_inventory.csv" \
  --vocabulary "${EP11_RUN_DIR}/allen_target_vocabulary.csv" \
  --self-test > "${EP11_RUN_DIR}/core_self_test.json"

python3 "${EP11_RUN_DIR}/run_mixed_support_group_module_ablation.py" \
  --inventory "${EP11_RUN_DIR}/whole_cortex_ssptr_inventory.csv" \
  --vocabulary "${EP11_RUN_DIR}/allen_target_vocabulary.csv" \
  --self-test > "${EP11_RUN_DIR}/module_self_test.json"

python3 "${EP11_RUN_DIR}/run_mixed_support_group_module_ablation.py" \
  --inventory "${EP11_RUN_DIR}/whole_cortex_ssptr_inventory.csv" \
  --vocabulary "${EP11_RUN_DIR}/allen_target_vocabulary.csv" \
  --output-dir "${EP11_RUN_DIR}/results" \
  --batch-size 32 \
  --maximum-replicates 256 \
  --integration-replicates 32 \
  --target-contrast-mcse 0.005 \
  --calibration-replicates 32 \
  --seed 20260927 \
  --margin 0.05

echo "EP11_GROUP_MODULE_RUN_DIR=${EP11_RUN_DIR}"
