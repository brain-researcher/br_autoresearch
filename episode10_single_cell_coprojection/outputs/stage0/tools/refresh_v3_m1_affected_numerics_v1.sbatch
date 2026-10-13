#!/usr/bin/env bash
#SBATCH --job-name=ep10-m1-affected-refresh-v1
#SBATCH --time=08:00:00
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
: "${EP10_M1_REFRESH_RESOURCE_JSON:?Set current accounting with this request reserved}"

project="$OAK/users/zijiao/br_autoresearch/episode10_single_cell_coprojection"
stage0="$project/outputs/stage0"
python_bin="$GROUP_HOME/users/zijiao/envs/ep10_targets_py312/bin/python"
output_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/v3_m1_affected_numerics_refresh_v1_job_${SLURM_JOB_ID}"
runtime_log="${output_root}.log"
durable_root="$stage0/v3_m1_affected_numerics_refresh_v1_job_${SLURM_JOB_ID}"
test ! -e "$output_root"
test ! -e "$runtime_log"
test ! -e "$durable_root"
test -f "$EP10_M1_REFRESH_RESOURCE_JSON"
test ! -L "$EP10_M1_REFRESH_RESOURCE_JSON"

# Preserve actual numerical JSON and the log after either success or failure.
# This small bounded copy does not write to any provisioned input payload.
retain_result() {
  run_status=$?
  trap - EXIT
  if mkdir "$durable_root"; then
    if test -f "$output_root/v3_m1_affected_numerics_refresh.json"; then
      cp "$output_root/v3_m1_affected_numerics_refresh.json" "$durable_root/" || \
        printf 'RETENTION_ERROR: numerical JSON remains at %s\n' "$output_root" >&2
    fi
    if test -f "$runtime_log"; then
      cp "$runtime_log" "$durable_root/runtime.log" || \
        printf 'RETENTION_ERROR: runtime log remains at %s\n' "$runtime_log" >&2
    fi
  else
    printf 'RETENTION_ERROR: durable directory creation failed; source remains %s\n' "$output_root" >&2
  fi
  exit "$run_status"
}
trap retain_result EXIT
trap 'exit 143' TERM
exec >>"$runtime_log" 2>&1
ml purge
ml load devel python/3.12.1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
for numerical_override in OPENBLAS_CORETYPE NPY_DISABLE_CPU_FEATURES NPY_ENABLE_CPU_FEATURES; do
  if [[ -v "$numerical_override" ]]; then
    printf 'ERROR: inherited numerical backend override %s is forbidden\n' "$numerical_override" >&2
    exit 72
  fi
done
"$python_bin" "$stage0/tools/refresh_v3_m1_affected_numerics_v1.py" \
  --output-root "$output_root" \
  --resource-accounting "$EP10_M1_REFRESH_RESOURCE_JSON"
