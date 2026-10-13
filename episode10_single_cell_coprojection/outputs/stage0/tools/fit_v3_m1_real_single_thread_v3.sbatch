#!/usr/bin/env bash
#SBATCH --job-name=ep10-v3-m1-real-1t-v3
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=russpold,normal
#SBATCH --constraint=CPU_SKU:7502

set -euo pipefail
umask 007

: "${SCRATCH:?SCRATCH is required}"
: "${OAK:?OAK is required}"
: "${GROUP_HOME:?GROUP_HOME is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
: "${EP10_M1_PRODUCTION_RESOURCE_JSON:?Set the approved clean-refit resource assessment}"

python_bin="$GROUP_HOME/users/zijiao/envs/ep10_targets_py312/bin/python"
project="$OAK/users/zijiao/br_autoresearch/episode10_single_cell_coprojection"
stage0="$project/outputs/stage0"
target_free_engine="$stage0/tools/fit_v3_m1.py"
runner="$stage0/tools/fit_v3_m1_real_v1.py"
validator="$stage0/tools/validate_v3_m1_numerical_execution_v3.py"
test_file="$stage0/tests/test_fit_v3_m1_real_v1.py"
contract="$stage0/v3_m1_model_contract.yaml"
execution_addendum="$stage0/v3_m1_numerical_execution_addendum_v3.yaml"
resource_amendment="$stage0/v3_m1_production_resource_amendment_v4.yaml"
scratch_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection"
interface_root="$scratch_root/v3_model_interface_preflight_v1"
geometry_root="$scratch_root/v3_m1_geometry_v1"
target_root="$scratch_root/gao_mop_development_targets_v3_ep10_assets"
actual_k_root="$scratch_root/v3_m1_actual_k_gate_v1_job_45602325"
qualification_root="$scratch_root/v3_m1_synthetic_warm_zero_diagnostic_v2_job_46328672"
affected_refresh_root="$scratch_root/v3_m1_affected_numerics_refresh_v1_job_46330399"
output_root="$scratch_root/v3_m1_real_fit_single_thread_v3_job_${SLURM_JOB_ID}"
runtime_log="${output_root}.log"

test -x "$python_bin"
test -d "$scratch_root"
if test -e "$output_root"; then
  test -d "$output_root"
  test ! -L "$output_root"
fi
if test -e "$runtime_log"; then
  test -f "$runtime_log"
  test ! -L "$runtime_log"
fi

for required in \
  "$target_free_engine" \
  "$runner" \
  "$validator" \
  "$test_file" \
  "$contract" \
  "$execution_addendum" \
  "$resource_amendment" \
  "$EP10_M1_PRODUCTION_RESOURCE_JSON" \
  "$stage0/v3_m1_wall_limit_authorization_20261002.json" \
  "$stage0/v3_m1_synthetic_design_addendum_v1.yaml" \
  "$stage0/v3_m1_synthetic_recovery_addendum_v2.yaml" \
  "$stage0/v3_m1_solver_addendum_v1.yaml" \
  "$stage0/v3_m1_solver_addendum_v2.yaml" \
  "$stage0/readiness_status.yaml" \
  "$actual_k_root/v3_m1_actual_k_gate_summary.json" \
  "$qualification_root/v3_m1_synthetic_warm_zero_diagnostic.json" \
  "$affected_refresh_root/v3_m1_affected_numerics_refresh.json" \
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
printf 'EP10 M1 clean one-thread development fit v3 log: %s\n' "$runtime_log"

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
"$python_bin" "$test_file"
"$python_bin" "$runner" \
  --contract "$contract" \
  --execution-addendum "$execution_addendum" \
  --qualification-root "$qualification_root" \
  --affected-refresh-root "$affected_refresh_root" \
  --resource-amendment "$resource_amendment" \
  --resource-assessment "$EP10_M1_PRODUCTION_RESOURCE_JSON" \
  --interface-root "$interface_root" \
  --geometry-root "$geometry_root" \
  --actual-k-root "$actual_k_root" \
  --target-root "$target_root" \
  --output-root "$output_root"
