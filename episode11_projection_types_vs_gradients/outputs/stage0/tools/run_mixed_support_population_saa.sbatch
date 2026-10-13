#!/bin/bash
#SBATCH --job-name=ep11-pop-saa
#SBATCH --partition=russpold
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode11_projection_types_vs_gradients/population-saa-%j.out

set -euo pipefail

EP11_REPO_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch
EP11_EPISODE_ROOT=${EP11_REPO_ROOT}/episode11_projection_types_vs_gradients
EP11_RUN_ROOT=${SCRATCH}/br_autoresearch/episode11_projection_types_vs_gradients/population_saa
EP11_RUN_DIR=${EP11_RUN_ROOT}/run_${SLURM_JOB_ID}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ml purge
ml load devel math
ml load python/3.12.1 py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-pandas/2.2.1_py312

mkdir -p "${EP11_RUN_DIR}/memmaps"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/tools/run_mixed_support_56d_qualification.py" "${EP11_RUN_DIR}/"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/tools/run_mixed_support_full_family_calibration.py" "${EP11_RUN_DIR}/"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/tools/run_mixed_support_population_saa.py" "${EP11_RUN_DIR}/"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/whole_cortex_ssptr_inventory.csv" "${EP11_RUN_DIR}/"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/allen_target_vocabulary.csv" "${EP11_RUN_DIR}/"

python3 "${EP11_RUN_DIR}/run_mixed_support_population_saa.py" \
  --inventory "${EP11_RUN_DIR}/whole_cortex_ssptr_inventory.csv" \
  --vocabulary "${EP11_RUN_DIR}/allen_target_vocabulary.csv" \
  --fit-stream "${EP11_FIT_STREAM:-5101}" \
  --validation-stream "${EP11_VALIDATION_STREAM:-5201}" \
  --terminal-stream "${EP11_TERMINAL_STREAM:-5301}" \
  --anchor-stream "${EP11_ANCHOR_STREAM:-5401}" \
  --self-test > "${EP11_RUN_DIR}/population_saa_self_test.json"

python3 "${EP11_RUN_DIR}/run_mixed_support_population_saa.py" \
  --inventory "${EP11_RUN_DIR}/whole_cortex_ssptr_inventory.csv" \
  --vocabulary "${EP11_RUN_DIR}/allen_target_vocabulary.csv" \
  --output-dir "${EP11_RUN_DIR}/results" \
  --scratch-dir "${EP11_RUN_DIR}/memmaps" \
  --amplitude "${EP11_AMPLITUDE:-1.25}" \
  --saa-ladder "${EP11_SAA_LADDER:-128,512}" \
  --chunk-replicates "${EP11_CHUNK_REPLICATES:-32}" \
  --preconditioner-replicates "${EP11_PRECONDITIONER_REPLICATES:-32}" \
  --initialization-replicates "${EP11_INITIALIZATION_REPLICATES:-16}" \
  --starts "${EP11_STARTS:-2}" \
  --first-level-fresh-starts "${EP11_FIRST_LEVEL_FRESH_STARTS:-3}" \
  --start-workers "${EP11_START_WORKERS:-${SLURM_CPUS_PER_TASK}}" \
  --max-iterations "${EP11_MAX_ITERATIONS:-1200}" \
  --polish-iterations "${EP11_POLISH_ITERATIONS:-600}" \
  --validation-replicates "${EP11_VALIDATION_REPLICATES:-128}" \
  --terminal-replicates "${EP11_TERMINAL_REPLICATES:-512}" \
  --anchor-replicates "${EP11_ANCHOR_REPLICATES:-2}" \
  --target-mcse "${EP11_TARGET_MCSE:-0.005}" \
  --start-stability "${EP11_START_STABILITY:-0.002}" \
  --ladder-stability "${EP11_LADDER_STABILITY:-0.005}" \
  --terminal-stability "${EP11_TERMINAL_STABILITY:-0.005}" \
  --polish-improvement "${EP11_POLISH_IMPROVEMENT:-0.005}" \
  --fit-stream "${EP11_FIT_STREAM:-5101}" \
  --validation-stream "${EP11_VALIDATION_STREAM:-5201}" \
  --terminal-stream "${EP11_TERMINAL_STREAM:-5301}" \
  --anchor-stream "${EP11_ANCHOR_STREAM:-5401}" \
  --seed "${EP11_SEED:-20261001}"

echo "EP11_POPULATION_SAA_RUN_DIR=${EP11_RUN_DIR}"
