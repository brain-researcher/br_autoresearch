#!/usr/bin/env bash
#SBATCH --job-name=ep10-v3-m1-real-v1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=russpold,normal

set -euo pipefail
umask 007

: "${SCRATCH:?SCRATCH is required}"
: "${OAK:?OAK is required}"
: "${GROUP_HOME:?GROUP_HOME is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

python_bin="$GROUP_HOME/users/zijiao/envs/ep10_targets_py312/bin/python"
project="$OAK/users/zijiao/br_autoresearch/episode10_single_cell_coprojection"
stage0="$project/outputs/stage0"
target_free_engine="$stage0/tools/fit_v3_m1.py"
runner="$stage0/tools/fit_v3_m1_real_v1.py"
test_file="$stage0/tests/test_fit_v3_m1_real_v1.py"
contract="$stage0/v3_m1_model_contract.yaml"
scratch_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection"
interface_root="$scratch_root/v3_model_interface_preflight_v1"
geometry_root="$scratch_root/v3_m1_geometry_v1"
target_root="$scratch_root/gao_mop_development_targets_v3_ep10_assets"
actual_k_root="$scratch_root/v3_m1_actual_k_gate_v1_job_45602325"
output_root="$scratch_root/v3_m1_real_fit_v1_job_${SLURM_JOB_ID}"
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
  "$test_file" \
  "$contract" \
  "$stage0/v3_m1_synthetic_design_addendum_v1.yaml" \
  "$stage0/v3_m1_synthetic_recovery_addendum_v2.yaml" \
  "$stage0/v3_m1_solver_addendum_v1.yaml" \
  "$stage0/v3_m1_solver_addendum_v2.yaml" \
  "$stage0/readiness_status.yaml" \
  "$actual_k_root/v3_m1_actual_k_gate_summary.json" \
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
printf 'EP10 M1 development-only real fit v1 log: %s\n' "$runtime_log"

ml purge
ml load devel python/3.12.1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS=1

"$python_bin" --version
"$python_bin" "$test_file"
"$python_bin" "$runner" \
  --contract "$contract" \
  --interface-root "$interface_root" \
  --geometry-root "$geometry_root" \
  --actual-k-root "$actual_k_root" \
  --target-root "$target_root" \
  --output-root "$output_root"
