#!/usr/bin/env bash
#SBATCH --job-name=ep10-v4-falsifier-time
#SBATCH --partition=russpold,normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/scratch/users/%u/%x-%j.out
#SBATCH --error=/scratch/users/%u/%x-%j.err

set -euo pipefail
umask 007

: "${SCRATCH:?SCRATCH is required}"
: "${OAK:?OAK is required}"
: "${GROUP_HOME:?GROUP_HOME is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required}"

python_bin="$GROUP_HOME/users/zijiao/envs/ep10_targets_py312/bin/python"
episode="$OAK/users/zijiao/br_autoresearch/episode10_single_cell_coprojection"
stage0="$episode/outputs/stage0"
tool="$stage0/tools/benchmark_v3_m2_falsifier_timing_v1.py"
scratch_episode="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection"
run_root="$scratch_episode/v3_m2_falsifier_timing_v1_job_${SLURM_JOB_ID}"
report="$run_root/v3_m2_falsifier_timing_supplement.json"

test -x "$python_bin"
test -f "$tool"
test ! -L "$tool"
test -d "$scratch_episode"
test ! -L "$scratch_episode"
test ! -e "$run_root"
mkdir -p "$run_root"

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

"$python_bin" --version
"$python_bin" "$stage0/tools/validate_v3_m1_numerical_execution_v3.py" --runtime > "$run_root/numeric_backend.json"
# Focused timing orchestration regressions run before submission.
"$python_bin" "$tool" --output "$report" --workers 8

printf 'EP10 v4 target-free falsifier timing supplement: PASS\n'
printf 'aggregate report: %s\n' "$report"
