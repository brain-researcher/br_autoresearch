#!/bin/bash
#SBATCH --job-name=ep11-pop-h
#SBATCH --partition=russpold
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode11_projection_types_vs_gradients/population-h-recovery-%j.out

set -euo pipefail

EP11_REPO_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch
EP11_EPISODE_ROOT=${EP11_REPO_ROOT}/episode11_projection_types_vs_gradients
EP11_RUN_ROOT=${SCRATCH}/br_autoresearch/episode11_projection_types_vs_gradients/population_h_recovery
EP11_RUN_DIR=${EP11_RUN_ROOT}/run_${SLURM_JOB_ID}
EP11_C_RESULTS_DIR=${1:-${EP11_C_RESULTS_DIR:-}}
EP11_START_POLICY=${2:-${EP11_START_POLICY:-raw_empirical_v1}}
EP11_H_OPTIONS=(--start-policy "${EP11_START_POLICY}")
case "${EP11_START_POLICY}" in
  raw_empirical_v1|empirical_perturbation_v1) ;;
  *) echo "Unknown H start policy: ${EP11_START_POLICY}" >&2; exit 2 ;;
esac
if [[ -n "${EP11_VERIFICATION_STREAM:-}" ]]; then
  EP11_H_OPTIONS+=(--verification-stream "${EP11_VERIFICATION_STREAM}")
fi

if [[ -z "${EP11_C_RESULTS_DIR}" ]]; then
  echo "Usage: sbatch run_mixed_support_population_h_recovery.sbatch /absolute/path/to/passing_C_results" >&2
  exit 2
fi
if [[ "${EP11_C_RESULTS_DIR}" != /* ]]; then
  echo "The fitted-C result directory must be an absolute path" >&2
  exit 2
fi
if [[ ! -f "${EP11_C_RESULTS_DIR}/population_saa_pilot_summary.json" ]]; then
  echo "Missing fitted-C summary in ${EP11_C_RESULTS_DIR}" >&2
  exit 2
fi
if [[ ! -f "${EP11_C_RESULTS_DIR}/population_saa_C_model_state.npz" ]]; then
  echo "Missing fitted-C model state in ${EP11_C_RESULTS_DIR}" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ml purge
ml load devel math
ml load python/3.12.1 py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-pandas/2.2.1_py312

mkdir -p "${EP11_RUN_DIR}/c_input" "${EP11_RUN_DIR}/memmaps"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/tools/run_mixed_support_56d_qualification.py" "${EP11_RUN_DIR}/"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/tools/run_mixed_support_full_family_calibration.py" "${EP11_RUN_DIR}/"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/tools/run_mixed_support_population_saa.py" "${EP11_RUN_DIR}/"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/tools/run_mixed_support_population_h_recovery.py" "${EP11_RUN_DIR}/"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/whole_cortex_ssptr_inventory.csv" "${EP11_RUN_DIR}/"
cp "${EP11_EPISODE_ROOT}/outputs/stage0/allen_target_vocabulary.csv" "${EP11_RUN_DIR}/"
cp "${EP11_C_RESULTS_DIR}/population_saa_pilot_summary.json" "${EP11_RUN_DIR}/c_input/"
cp "${EP11_C_RESULTS_DIR}/population_saa_C_model_state.npz" "${EP11_RUN_DIR}/c_input/"

if [[ "${EP11_START_POLICY}" == raw_empirical_v1 ]]; then
  python3 "${EP11_RUN_DIR}/run_mixed_support_population_h_recovery.py" \
    --inventory "${EP11_RUN_DIR}/whole_cortex_ssptr_inventory.csv" \
    --vocabulary "${EP11_RUN_DIR}/allen_target_vocabulary.csv" \
    --self-test > "${EP11_RUN_DIR}/population_h_recovery_self_test.json"
fi

python3 "${EP11_RUN_DIR}/run_mixed_support_population_h_recovery.py" \
  --inventory "${EP11_RUN_DIR}/whole_cortex_ssptr_inventory.csv" \
  --vocabulary "${EP11_RUN_DIR}/allen_target_vocabulary.csv" \
  --c-summary "${EP11_RUN_DIR}/c_input/population_saa_pilot_summary.json" \
  --c-model-state "${EP11_RUN_DIR}/c_input/population_saa_C_model_state.npz" \
  --output-dir "${EP11_RUN_DIR}/results" \
  --scratch-dir "${EP11_RUN_DIR}/memmaps" \
  --chunk-replicates "${EP11_CHUNK_REPLICATES:-32}" \
  --preconditioner-replicates "${EP11_PRECONDITIONER_REPLICATES:-32}" \
  --initialization-replicates "${EP11_INITIALIZATION_REPLICATES:-16}" \
  --start-workers "${EP11_START_WORKERS:-${SLURM_CPUS_PER_TASK}}" \
  --verification-replicates "${EP11_VERIFICATION_REPLICATES:-512}" \
  --verification-workers "${EP11_VERIFICATION_WORKERS:-${SLURM_CPUS_PER_TASK}}" \
  "${EP11_H_OPTIONS[@]}" \
  --target-mcse "${EP11_TARGET_MCSE:-0.005}" \
  --start-stability "${EP11_START_STABILITY:-0.002}" \
  --truth-h-upper "${EP11_TRUTH_H_UPPER:-0.010}"

echo "EP11_POPULATION_H_RECOVERY_RUN_DIR=${EP11_RUN_DIR}"
