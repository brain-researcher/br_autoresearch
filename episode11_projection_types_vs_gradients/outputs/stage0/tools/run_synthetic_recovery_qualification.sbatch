#!/bin/bash
#SBATCH --job-name=ep11-syn-recovery
#SBATCH --partition=russpold
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode11_projection_types_vs_gradients/synthetic_recovery/slurm-%j.out

set -euo pipefail

EP11_REPO_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch
EP11_EPISODE_ROOT=${EP11_REPO_ROOT}/episode11_projection_types_vs_gradients
EP11_RUN_ROOT=${SCRATCH}/br_autoresearch/episode11_projection_types_vs_gradients/synthetic_recovery
EP11_RUN_DIR=${EP11_RUN_ROOT}/run_${SLURM_JOB_ID}
EP11_PYTHON=${SCRATCH}/br_autoresearch/episode11_projection_types_vs_gradients/ccf_env_py311/bin/python

mkdir -p "${EP11_RUN_DIR}"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/whole_cortex_ssptr_inventory.csv" "${EP11_RUN_DIR}/inventory.csv"

"${EP11_PYTHON}" \
  "${EP11_EPISODE_ROOT}/outputs/stage0/tools/run_synthetic_recovery_qualification.py" \
  --inventory "${EP11_RUN_DIR}/inventory.csv" \
  --output-csv "${EP11_RUN_DIR}/synthetic_recovery_qualification.csv" \
  --output-md "${EP11_RUN_DIR}/synthetic_recovery_qualification.md" \
  --replicate-output "${EP11_RUN_DIR}/synthetic_recovery_replicates.csv" \
  --replicates 200 \
  --oracle-draws 16 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --seed 20260926 \
  --margin 0.05

echo "EP11_SYNTHETIC_RECOVERY_RUN_DIR=${EP11_RUN_DIR}"
