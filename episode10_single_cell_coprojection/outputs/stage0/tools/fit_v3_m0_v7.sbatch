#!/usr/bin/env bash
#SBATCH --job-name=ep10-v3-m0-v7
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
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
engine="$stage0/tools/fit_v3_m0_v7.py"
contract="$stage0/v3_m0_model_contract.yaml"
solver_addendum="$stage0/v3_m0_solver_addendum_v7.yaml"
scratch_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection"
target_root="$scratch_root/gao_mop_development_targets_v3_ep10_assets"
interface_root="$scratch_root/v3_model_interface_preflight_v1"
output_root="$scratch_root/v3_m0_fit_v7"
runtime_log="${output_root}.log"

test -d "$scratch_root"
test ! -e "$output_root"
test ! -e "$runtime_log"

printf 'EP10 M0 v7 production log: %s\n' "$runtime_log"
exec > >(tee "$runtime_log") 2>&1

ml purge
ml load devel python/3.12.1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

test -x "$python_bin"
for required in \
  "$engine" \
  "$contract" \
  "$solver_addendum" \
  "$stage0/tools/fit_v3_m0.py" \
  "$stage0/tools/fit_v3_m0_v2.py" \
  "$stage0/tools/fit_v3_m0_v3.py" \
  "$stage0/tools/fit_v3_m0_v4.py" \
  "$stage0/tools/fit_v3_m0_v5.py" \
  "$stage0/tools/fit_v3_m0_v6.py" \
  "$stage0/v3_m0_solver_addendum_v2.yaml" \
  "$stage0/v3_m0_solver_addendum_v3.yaml" \
  "$stage0/v3_m0_solver_addendum_v4.yaml" \
  "$stage0/v3_m0_solver_addendum_v5.yaml" \
  "$stage0/v3_m0_solver_addendum_v6.yaml" \
  "$target_root/primary_16_target_calls.csv" \
  "$target_root/v3_gate_summary.json" \
  "$interface_root/v3_model_interface.csv" \
  "$interface_root/v3_model_interface_summary.json"; do
  test -f "$required"
  test ! -L "$required"
done

"$python_bin" --version
"$python_bin" "$engine" \
  --unit-test \
  --contract "$contract" \
  --solver-addendum "$solver_addendum"

"$python_bin" "$engine" \
  --execute \
  --contract "$contract" \
  --solver-addendum "$solver_addendum" \
  --target-root "$target_root" \
  --interface-root "$interface_root" \
  --output-root "$output_root"
