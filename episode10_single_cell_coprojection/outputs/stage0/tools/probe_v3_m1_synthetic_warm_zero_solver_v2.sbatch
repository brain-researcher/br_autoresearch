#!/usr/bin/env bash
#SBATCH --job-name=ep10-m1-polish-v2
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
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
engine="$stage0/tools/fit_v3_m1.py"
test_file="$stage0/tests/test_fit_v3_m1.py"
contract="$stage0/v3_m1_model_contract.yaml"
synthetic_design_addendum="$stage0/v3_m1_synthetic_design_addendum_v1.yaml"
synthetic_recovery_addendum="$stage0/v3_m1_synthetic_recovery_addendum_v2.yaml"
solver_addendum_v1="$stage0/v3_m1_solver_addendum_v1.yaml"
solver_addendum="$stage0/v3_m1_solver_addendum_v2.yaml"
scratch_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection"
output_root="$scratch_root/v3_m1_synthetic_warm_zero_diagnostic_v2_job_${SLURM_JOB_ID}"
runtime_log="${output_root}.log"

test -x "$python_bin"
test -d "$scratch_root"
test ! -e "$output_root"
test ! -e "$runtime_log"

for required in \
  "$engine" \
  "$test_file" \
  "$contract" \
  "$synthetic_design_addendum" \
  "$synthetic_recovery_addendum" \
  "$solver_addendum_v1" \
  "$solver_addendum"; do
  test -f "$required"
  test ! -L "$required"
done

printf 'EP10 M1 one-shot-confirmation warm/zero probe log: %s\n' "$runtime_log"
exec > >(tee "$runtime_log") 2>&1

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
"$python_bin" "$engine" \
  --synthetic-warm-zero-diagnostic \
  --contract "$contract" \
  --synthetic-design-addendum "$synthetic_design_addendum" \
  --synthetic-recovery-addendum "$synthetic_recovery_addendum" \
  --solver-addendum "$solver_addendum" \
  --output-root "$output_root"
