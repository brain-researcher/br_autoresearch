#!/usr/bin/env bash
#SBATCH --job-name=ep10-v3-m2-optqual
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=9
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
tool="$stage0/tools/qualify_v3_m2_optional.py"
resume_cases="$stage0/v3_m2_optional_family_qualification_job_46115632/completed_cases"
scratch_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection"
output_root="$scratch_root/v3_m2_optional_family_qualification_job_${SLURM_JOB_ID}"
runtime_log="${output_root}.log"
report="$output_root/v3_m2_optional_family_qualification.json"

test -x "$python_bin"
test -f "$tool"
test -d "$resume_cases"
test ! -e "$output_root"
test ! -e "$runtime_log"
mkdir -m 2770 "$output_root"

exec >>"$runtime_log" 2>&1
printf 'EP10 v3 M2 target-free optional-family qualification log: %s\n' "$runtime_log"

ml purge
ml load devel python/3.12.1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Reuse the unchanged numerical test passes from job46115632 and the focused
# resume regressions run before this submission; do not repeat test cycles.
"$python_bin" --version
"$python_bin" "$tool" --output "$report" --workers 9 \
  --resume-completed-cases "$resume_cases"

printf 'EP10 v3 M2 target-free optional-family qualification: PASS\n'
