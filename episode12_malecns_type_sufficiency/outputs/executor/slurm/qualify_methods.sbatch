#!/bin/bash
#SBATCH --job-name=ep12-qualify
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/qualify-%j.out

set -euo pipefail

EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_TIER=${1:-qualification}
EP12_OUTPUT=${2:-${EP12_ROOT}/outputs/scientific_qualification/${EP12_TIER}}

cd "${EP12_ROOT}"
module purge
module load python/3.12.1 \
  py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 \
  py-scikit-learn/1.5.1_py312 py-pandas/2.2.1_py312 \
  py-pyarrow/18.1.0_py312
export PYTHONPATH="${EP12_ROOT}/outputs/executor${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python3 -m ep12_executor.cli qualify-methods \
  --tier "${EP12_TIER}" \
  --output-dir "${EP12_OUTPUT}" \
  --policy "${EP12_ROOT}/SEARCH_POLICY.yaml" \
  --contract "${EP12_ROOT}/outputs/executor/SCIENTIFIC_CONTRACT_POST36.yaml" \
  --qualification-spec "${EP12_ROOT}/outputs/executor/SYNTHETIC_QUALIFICATION_SPEC_POST36.yaml"
