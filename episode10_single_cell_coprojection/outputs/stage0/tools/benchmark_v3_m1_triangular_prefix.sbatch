#!/usr/bin/env bash
#SBATCH --job-name=ep10-triangular-prefix
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/users/%u/%x-%j.out
#SBATCH --error=/scratch/users/%u/%x-%j.err

set -euo pipefail
umask 007
: "${SCRATCH:?SCRATCH is required}"
: "${OAK:?OAK is required}"
: "${GROUP_HOME:?GROUP_HOME is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

episode="$OAK/users/zijiao/br_autoresearch/episode10_single_cell_coprojection"
python_bin="$GROUP_HOME/users/zijiao/envs/ep10_targets_py312/bin/python"
run_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/v3_m1_triangular_prefix_job_${SLURM_JOB_ID}"
durable="$episode/outputs/stage0/v3_m1_triangular_prefix_job_${SLURM_JOB_ID}"
test ! -e "$run_root"
test ! -e "$durable"
mkdir -p "$run_root"

# Copy only this small target-free report directory on every ordinary exit.
# The retained scratch report also survives a scheduler timeout.
preserve_reports() {
    cp -an "$run_root" "$durable"
}
trap preserve_reports EXIT

ml purge
ml load devel python/3.12.1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$episode"
hostname
printf 'SLURM_JOB_ID=%s\n' "$SLURM_JOB_ID"
"$python_bin" outputs/stage0/tools/validate_v3_m1_numerical_execution_v3.py --runtime > "$run_root/numeric_backend.json"
"$python_bin" outputs/stage0/tests/test_v3_m1_triangular_solve.py > "$run_root/numerical_tests.txt" 2>&1
"$python_bin" outputs/stage0/tools/benchmark_v3_m1_triangular_prefix.py --output-root "$run_root/paired"
