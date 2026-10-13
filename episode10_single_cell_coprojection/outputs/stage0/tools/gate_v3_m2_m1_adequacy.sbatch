#!/usr/bin/env bash
#SBATCH --job-name=ep10-v3-m2-m1-gate
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
tool="$stage0/tools/gate_v3_m2_m1_adequacy.py"
test_file="$stage0/tests/test_gate_v3_m2_m1_adequacy.py"
scratch_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection"
m1_root="$scratch_root/v3_m1_real_fit_single_thread_v3_job_46332190"
output_root="$scratch_root/v3_m2_m1_adequacy_gate_job_${SLURM_JOB_ID}"
runtime_log="${output_root}.log"

test -x "$python_bin"
test -f "$tool"
test -f "$test_file"
test -d "$m1_root"
test ! -L "$m1_root"
test ! -e "$output_root"
test ! -e "$runtime_log"
mkdir -m 2770 "$output_root"

exec >>"$runtime_log" 2>&1
printf 'EP10 v3 M2 M1 adequacy gate log: %s\n' "$runtime_log"

ml purge
ml load devel python/3.12.1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$python_bin" "$test_file"
"$python_bin" "$tool" \
  --m1-root "$m1_root" \
  --output "$output_root/v3_m2_m1_adequacy.json"

printf 'EP10 v3 M2 M1 adequacy gate: PASS\n'
