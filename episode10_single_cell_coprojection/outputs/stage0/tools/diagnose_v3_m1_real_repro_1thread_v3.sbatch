#!/usr/bin/env bash
#SBATCH --job-name=ep10-v3-m1-real-repro-1t-v3
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
runner="$stage0/tools/fit_v3_m1_real_v1.py"
diagnostic_v1="$stage0/tools/diagnose_v3_m1_real_failure_v1.py"
diagnostic_v2="$stage0/tools/diagnose_v3_m1_real_repro_v2.py"
diagnostic="$stage0/tools/diagnose_v3_m1_real_repro_1thread_v3.py"
runner_test_file="$stage0/tests/test_fit_v3_m1_real_v1.py"
diagnostic_v2_test_file="$stage0/tests/test_diagnose_v3_m1_real_repro_v2.py"
diagnostic_test_file="$stage0/tests/test_diagnose_v3_m1_real_repro_1thread_v3.py"
contract="$stage0/v3_m1_model_contract.yaml"
scratch_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection"
interface_root="$scratch_root/v3_model_interface_preflight_v1"
geometry_root="$scratch_root/v3_m1_geometry_v1"
target_root="$scratch_root/gao_mop_development_targets_v3_ep10_assets"
actual_k_root="$scratch_root/v3_m1_actual_k_gate_v1_job_45602325"
prior_fit_root="$scratch_root/v3_m1_real_fit_v1_job_45609750"
output_root="$scratch_root/v3_m1_real_warm_repro_1thread_v3_job_${SLURM_JOB_ID}"
runtime_log="${output_root}.log"

test -x "$python_bin"
test -d "$scratch_root"
test ! -e "$output_root"
test ! -e "$runtime_log"

for required in \
  "$runner" \
  "$diagnostic_v1" \
  "$diagnostic_v2" \
  "$diagnostic" \
  "$runner_test_file" \
  "$diagnostic_v2_test_file" \
  "$diagnostic_test_file" \
  "$contract" \
  "$stage0/tools/fit_v3_m1.py" \
  "$stage0/readiness_status.yaml" \
  "$actual_k_root/v3_m1_actual_k_gate_summary.json" \
  "$prior_fit_root/v3_m1_fit_summary.json" \
  "$interface_root/v3_model_interface.csv" \
  "$interface_root/v3_model_interface_summary.json" \
  "$geometry_root/atlas_target_anchors.csv" \
  "$geometry_root/cell_target_geometry.csv" \
  "$geometry_root/target_pair_geometry.csv" \
  "$geometry_root/geometry_summary.json" \
  "$target_root/primary_16_target_calls.csv" \
  "$target_root/v3_gate_summary.json"; do
  test -f "$required"
  test ! -L "$required"
done

exec >>"$runtime_log" 2>&1
printf 'EP10 M1 AMD single-thread two-repeat diagnostic v3 log: %s\n' "$runtime_log"

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
"$python_bin" "$runner_test_file"
"$python_bin" "$diagnostic_v2_test_file"
"$python_bin" "$diagnostic_test_file"
"$python_bin" "$diagnostic" \
  --contract "$contract" \
  --interface-root "$interface_root" \
  --geometry-root "$geometry_root" \
  --actual-k-root "$actual_k_root" \
  --target-root "$target_root" \
  --prior-fit-root "$prior_fit_root" \
  --output-root "$output_root"
