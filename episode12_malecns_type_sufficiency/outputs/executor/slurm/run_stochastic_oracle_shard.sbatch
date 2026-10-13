#!/bin/bash
#SBATCH --job-name=ep12-oracle
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/oracle-%A_%a.out

. /etc/profile >/dev/null
set -euo pipefail

EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_TIER=${1:?tier is required}
EP12_LOOK=${2:?look index is required}
EP12_BANK_ROOT=${3:?bank root is required}
EP12_CELL_INDEX=${SLURM_ARRAY_TASK_ID:?submit as an explicit cell array}

cd "${EP12_ROOT}"
module purge
module load python/3.12.1 \
  py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 \
  py-scikit-learn/1.5.1_py312 py-pandas/2.2.1_py312 \
  py-pyarrow/18.1.0_py312
export PYTHONPATH="${EP12_ROOT}/outputs/executor${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python3 -m ep12_executor run-oracle-shard \
  --tier "${EP12_TIER}" \
  --look-index "${EP12_LOOK}" \
  --cell-index "${EP12_CELL_INDEX}" \
  --output-dir "${EP12_BANK_ROOT}/look_$(printf '%02d' "${EP12_LOOK}")" \
  --contract "${EP12_ROOT}/outputs/executor/SCIENTIFIC_CONTRACT_POST36_V3.yaml" \
  --qualification-spec "${EP12_ROOT}/outputs/executor/SYNTHETIC_QUALIFICATION_SPEC_POST36_V3.yaml"
