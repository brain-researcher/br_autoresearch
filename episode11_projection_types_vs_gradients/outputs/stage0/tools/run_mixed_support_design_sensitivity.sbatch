#!/bin/bash
#SBATCH --job-name=ep11-design-sens
#SBATCH --partition=russpold
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode11_projection_types_vs_gradients/design_sensitivity/slurm-%j.out

set -euo pipefail

EP11_REPO_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch
EP11_EPISODE_ROOT=${EP11_REPO_ROOT}/episode11_projection_types_vs_gradients
EP11_RUN_ROOT=${SCRATCH}/br_autoresearch/episode11_projection_types_vs_gradients/design_sensitivity
EP11_RUN_DIR=${EP11_RUN_ROOT}/run_${SLURM_JOB_ID}

ml purge
ml load devel math
ml load python/3.12.1 py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-pandas/2.2.1_py312

mkdir -p "${EP11_RUN_DIR}"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/tools/run_mixed_support_56d_qualification.py" "${EP11_RUN_DIR}/"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/tools/run_mixed_support_design_sensitivity.py" "${EP11_RUN_DIR}/"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/whole_cortex_ssptr_inventory.csv" "${EP11_RUN_DIR}/"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/allen_target_vocabulary.csv" "${EP11_RUN_DIR}/"

python3 \
  "${EP11_RUN_DIR}/run_mixed_support_design_sensitivity.py" \
  --inventory "${EP11_RUN_DIR}/whole_cortex_ssptr_inventory.csv" \
  --vocabulary "${EP11_RUN_DIR}/allen_target_vocabulary.csv" \
  --output-dir "${EP11_RUN_DIR}/results" \
  --replicates 32 \
  --outer-cv-replicates 0 \
  --calibration-replicates 32 \
  --group-scenarios reusable_groups \
  --designs nested_baseline_12_dev_8_audit_16_unassigned,nested_estimator_ceiling_27_dev_8_audit_1_unassigned,nested_precision_ceiling_12_dev_23_audit_1_unassigned \
  --seed 20260926 \
  --margin 0.05

echo "EP11_DESIGN_SENSITIVITY_RUN_DIR=${EP11_RUN_DIR}"
