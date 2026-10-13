#!/usr/bin/env bash
#SBATCH --job-name=ep10-v4-permutation-diag
#SBATCH --partition=russpold,normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/scratch/users/%u/%x-%j.out
#SBATCH --error=/scratch/users/%u/%x-%j.err

set -euo pipefail
umask 007
: "${SCRATCH:?SCRATCH is required}"
: "${OAK:?OAK is required}"
: "${GROUP_HOME:?GROUP_HOME is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

python_bin="$GROUP_HOME/users/zijiao/envs/ep10_targets_py312/bin/python"
episode="$OAK/users/zijiao/br_autoresearch/episode10_single_cell_coprojection"
tool="$episode/outputs/stage0/tools/diagnose_v3_m2_falsifier_permutation_v1.py"
scratch_episode="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection"
run_root="$scratch_episode/v3_m2_falsifier_permutation_diagnostic_job_${SLURM_JOB_ID}"
test -x "$python_bin"
test -f "$tool"
test ! -L "$tool"
test -d "$scratch_episode"
test ! -L "$scratch_episode"
test ! -e "$run_root"
mkdir -m 2770 "$run_root"

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
"$python_bin" "$tool" --output-root "$run_root"
