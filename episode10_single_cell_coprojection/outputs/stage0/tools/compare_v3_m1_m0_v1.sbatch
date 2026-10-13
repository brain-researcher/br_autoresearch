#!/usr/bin/env bash
#SBATCH --job-name=ep10-v3-m1-m0-compare-v1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
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
runner="$stage0/tools/compare_v3_m1_m0_v1.py"
test_file="$stage0/tests/test_compare_v3_m1_m0_v1.py"
contract="$stage0/v3_m1_model_contract.yaml"
readiness="$stage0/readiness_status.yaml"
scratch_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection"
m1_root="$scratch_root/v3_m1_real_fit_single_thread_v3_job_46332190"
m0_root="$scratch_root/v3_m0_fit_v7"
output_root="$scratch_root/v3_m1_compare_m0_v1_job_${SLURM_JOB_ID}"
runtime_log="${output_root}.log"

test -x "$python_bin"
test -f "$runner"
test ! -L "$runner"
test -f "$test_file"
test ! -L "$test_file"
test -f "$contract"
test ! -L "$contract"
test -f "$readiness"
test ! -L "$readiness"
test ! -e "$output_root"
test ! -e "$runtime_log"

exec >>"$runtime_log" 2>&1
printf 'EP10 paired M1-minus-M0 comparison log: %s\n' "$runtime_log"

ml purge
ml load devel python/3.12.1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$python_bin" --version
"$python_bin" "$test_file"
"$python_bin" "$runner" \
  --execute \
  --contract "$contract" \
  --readiness "$readiness" \
  --m1-root "$m1_root" \
  --m0-root "$m0_root" \
  --output-root "$output_root"
