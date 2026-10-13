#!/usr/bin/env bash
#SBATCH --job-name=ep10-v3-m2-qual
#SBATCH --time=02:00:00
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
tool="$stage0/tools/qualify_v3_m2_core.py"
core_test_file="$stage0/tests/test_fit_v3_m2.py"
qualification_test_file="$stage0/tests/test_fit_v3_m2_contract_qualification.py"
scratch_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection"
output_root="$scratch_root/v3_m2_core_contract_qualification_job_${SLURM_JOB_ID}"
runtime_log="${output_root}.log"

test -x "$python_bin"
test -f "$tool"
test -f "$core_test_file"
test -f "$qualification_test_file"
test ! -e "$output_root"
test ! -e "$runtime_log"
mkdir -m 2770 "$output_root"

exec >>"$runtime_log" 2>&1
printf 'EP10 v3 M2 frozen target-free core qualification log: %s\n' "$runtime_log"

ml purge
ml load devel python/3.12.1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$python_bin" "$core_test_file"
"$python_bin" "$tool" \
  --output "$output_root/v3_m2_core_contract_qualification.json"
"$python_bin" "$qualification_test_file" \
  --report "$output_root/v3_m2_core_contract_qualification.json"

printf 'EP10 v3 M2 frozen target-free core qualification: PASS\n'
