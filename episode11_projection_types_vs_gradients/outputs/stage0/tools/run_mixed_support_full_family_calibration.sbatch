#!/bin/bash
#SBATCH --job-name=ep11-full-family-cal
#SBATCH --partition=russpold
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode11_projection_types_vs_gradients/full-family-calibration-%j.out

set -euo pipefail

EP11_REPO_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch
EP11_EPISODE_ROOT=${EP11_REPO_ROOT}/episode11_projection_types_vs_gradients
EP11_RUN_ROOT=${SCRATCH}/br_autoresearch/episode11_projection_types_vs_gradients/full_family_calibration
EP11_RUN_DIR=${EP11_RUN_ROOT}/run_${SLURM_JOB_ID}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ml purge
ml load devel math
ml load python/3.12.1 py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-pandas/2.2.1_py312

mkdir -p "${EP11_RUN_DIR}"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/tools/run_mixed_support_56d_qualification.py" "${EP11_RUN_DIR}/"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/tools/run_mixed_support_full_family_calibration.py" "${EP11_RUN_DIR}/"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/whole_cortex_ssptr_inventory.csv" "${EP11_RUN_DIR}/"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/allen_target_vocabulary.csv" "${EP11_RUN_DIR}/"

python3 "${EP11_RUN_DIR}/run_mixed_support_full_family_calibration.py" \
  --inventory "${EP11_RUN_DIR}/whole_cortex_ssptr_inventory.csv" \
  --vocabulary "${EP11_RUN_DIR}/allen_target_vocabulary.csv" \
  --self-test > "${EP11_RUN_DIR}/full_family_self_test.json"

python3 "${EP11_RUN_DIR}/run_mixed_support_full_family_calibration.py" \
  --inventory "${EP11_RUN_DIR}/whole_cortex_ssptr_inventory.csv" \
  --vocabulary "${EP11_RUN_DIR}/allen_target_vocabulary.csv" \
  --output-dir "${EP11_RUN_DIR}/results" \
  --amplitude-grid "${EP11_AMPLITUDE_GRID:-0.0,0.10,0.20,0.30,0.45,0.65,0.90,1.25,1.75,2.50}" \
  --fit-replicates "${EP11_FIT_REPLICATES:-32}" \
  --curve-starts "${EP11_CURVE_STARTS:-2}" \
  --final-starts "${EP11_FINAL_STARTS:-3}" \
  --max-iterations "${EP11_MAX_ITERATIONS:-1200}" \
  --batch-size "${EP11_BATCH_SIZE:-32}" \
  --maximum-verification-replicates "${EP11_MAXIMUM_VERIFICATION_REPLICATES:-256}" \
  --target-mcse "${EP11_TARGET_MCSE:-0.005}" \
  --target-gain "${EP11_TARGET_GAIN:-0.100}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --seed "${EP11_SEED:-20260929}"

echo "EP11_FULL_FAMILY_CALIBRATION_RUN_DIR=${EP11_RUN_DIR}"
