#!/usr/bin/env bash
#SBATCH --job-name=ep10-m1-1t-gate-v3
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=russpold,normal
#SBATCH --constraint=CPU_SKU:7502

set -euo pipefail
umask 007

: "${SCRATCH:?SCRATCH is required}"
: "${OAK:?OAK is required}"
: "${GROUP_HOME:?GROUP_HOME is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

python_bin="$GROUP_HOME/users/zijiao/envs/ep10_targets_py312/bin/python"
project="$OAK/users/zijiao/br_autoresearch/episode10_single_cell_coprojection"
stage0="$project/outputs/stage0"
engine="$stage0/tools/fit_v3_m1.py"
wrapper="$stage0/tools/run_v3_m1_synthetic_warm_zero_single_thread_v3.py"
validator="$stage0/tools/validate_v3_m1_numerical_execution_v3.py"
engine_test="$stage0/tests/test_fit_v3_m1.py"
execution_test="$stage0/tests/test_v3_m1_numerical_execution_v3.py"
contract="$stage0/v3_m1_model_contract.yaml"
synthetic_design_addendum="$stage0/v3_m1_synthetic_design_addendum_v1.yaml"
synthetic_recovery_addendum="$stage0/v3_m1_synthetic_recovery_addendum_v2.yaml"
solver_addendum_v1="$stage0/v3_m1_solver_addendum_v1.yaml"
solver_addendum="$stage0/v3_m1_solver_addendum_v2.yaml"
execution_addendum="$stage0/v3_m1_numerical_execution_addendum_v3.yaml"
scratch_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection"
output_root="$scratch_root/v3_m1_synthetic_warm_zero_diagnostic_v2_job_${SLURM_JOB_ID}"
runtime_log="${output_root}.log"

test -x "$python_bin"
test -d "$scratch_root"
test ! -e "$output_root"
test ! -L "$output_root"
test ! -e "$runtime_log"
test ! -L "$runtime_log"

for required in \
  "$engine" \
  "$wrapper" \
  "$validator" \
  "$engine_test" \
  "$execution_test" \
  "$contract" \
  "$synthetic_design_addendum" \
  "$synthetic_recovery_addendum" \
  "$solver_addendum_v1" \
  "$solver_addendum" \
  "$execution_addendum"; do
  test -f "$required"
  test ! -L "$required"
done

printf 'EP10 M1 frozen one-thread 68-row gate log: %s\n' "$runtime_log"
exec >>"$runtime_log" 2>&1

ml purge
ml load devel python/3.12.1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
for numerical_override in \
  OPENBLAS_CORETYPE \
  NPY_DISABLE_CPU_FEATURES \
  NPY_ENABLE_CPU_FEATURES; do
  if [[ -v "$numerical_override" ]]; then
    printf 'ERROR: inherited numerical backend override %s is forbidden\n' \
      "$numerical_override" >&2
    exit 72
  fi
done

"$python_bin" --version
"$python_bin" "$validator" \
  --addendum "$execution_addendum" \
  --runtime
"$python_bin" "$engine_test"
# Backend-policy regressions already pass and their code is unchanged.
# The per-process runtime attestation above remains mandatory on every launch.
"$python_bin" "$wrapper" \
  --contract "$contract" \
  --synthetic-design-addendum "$synthetic_design_addendum" \
  --synthetic-recovery-addendum "$synthetic_recovery_addendum" \
  --solver-addendum "$solver_addendum" \
  --execution-addendum "$execution_addendum" \
  --output-root "$output_root"
