#!/bin/bash
#SBATCH --job-name=ep12-margin
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/prelaunch/slurm-margin-diagnostic-%j.out

set -euo pipefail

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
export PYTHONPATH="outputs/executor:${PYTHONPATH:-}"

EP12_TMP_ROOT=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
mkdir -p "${EP12_TMP_ROOT}"
export TMPDIR="${EP12_TMP_ROOT}"

python3 -m ep12_executor diagnose-margin \
  --output-dir outputs/prelaunch/margin_compatibility
