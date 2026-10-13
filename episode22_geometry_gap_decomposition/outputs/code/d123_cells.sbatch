#!/bin/bash
#SBATCH --job-name=ep22-d123
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode22_geometry_gap_decomposition
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode22_geometry_gap_decomposition/outputs/runs/d123_development/20261012-attempt02/slurm-%A_%a.out
# EP22 D1–D3, one participant × fold per array task (0–24, participant-major).
# Development responses only, from EP04's saved mean patterns; no OOD values.
set -uo pipefail
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-none} start=$(date -u +%FT%TZ)"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scipy/1.12.0_py312
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK \
       MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONDONTWRITEBYTECODE=1
timeout --signal=TERM --kill-after=10s 3300s python3.12 -u -B outputs/code/d123_cell.py \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --out-dir outputs/runs/d123_development/20261012-attempt02/cells
status=$?
echo "cell_exit=$status end=$(date -u +%FT%TZ)"
exit $status
