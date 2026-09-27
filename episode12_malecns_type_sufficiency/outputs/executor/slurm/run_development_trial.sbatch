#!/bin/bash
#SBATCH --job-name=ep12-real-trial
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/outputs/development_run001/slurm-trial-%j.out

set -euo pipefail

EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_SCRATCH=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency

cd "${EP12_ROOT}"
module purge
module load python/3.12.1 \
  py-numpy/1.26.3_py312 \
  py-scipy/1.12.0_py312 \
  py-scikit-learn/1.5.1_py312 \
  py-pandas/2.2.1_py312 \
  py-pyarrow/18.1.0_py312

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${EP12_ROOT}/outputs/executor:${PYTHONPATH:-}"
export TMPDIR="${EP12_SCRATCH}"

srun --ntasks=1 python3 -m ep12_executor run-development-trial \
  --output-dir "${EP12_ROOT}/outputs/development_run001/trial001" \
  --snapshot-dir "${EP12_SCRATCH}/development_snapshot_v1"
