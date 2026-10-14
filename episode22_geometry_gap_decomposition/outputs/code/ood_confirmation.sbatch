#!/bin/bash
#SBATCH --job-name=ep22-ood-confirm
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --chdir=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode22_geometry_gap_decomposition
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode22_geometry_gap_decomposition/outputs/runs/ood_confirmation/20261013-opening01/slurm-%j.out
# EP22 single procedurally sealed OOD opening. The evaluator verifies the
# write-once lock, keeps OOD values in memory and writes one report.
set -uo pipefail
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ)"
module purge
module load devel python/3.12.1 py-h5py/3.10.0_py312 math py-numpy/1.26.3 py-scipy/1.12.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python3.12 -u -B outputs/code/ood_confirmation.py --mode confirm --workers 5 \
  --out-dir outputs/runs/ood_confirmation/20261013-opening01 \
  --approval-record "outputs/experiment_log.md#2026-10-13-0032-utc--ood_confirmation--scientist-approved-the-single-ood-opening"
echo "confirm_exit=$? end=$(date -u +%FT%TZ)"
