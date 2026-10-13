#!/bin/bash
#SBATCH --job-name=ep12-qualify
#SBATCH --partition=normal
# The qualification loop is deliberately serial and pins numerical libraries
# to one thread. Historical execution used about 126 MiB RSS, so retain ample
# headroom without triggering Sherlock's extra-CPU memory allocation.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/qualify-%j.out

# Exact-once launchers use a sanitized environment; initialize Sherlock Lmod
# explicitly before enabling nounset.
# shellcheck source=/etc/profile
. /etc/profile >/dev/null
set -euo pipefail

EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_TIER=${1:-qualification}
EP12_OUTPUT=${2:-${EP12_ROOT}/outputs/scientific_qualification/${EP12_TIER}}
EP12_ORACLE_AGGREGATE=${3:?completed stochastic-oracle aggregate is required}

cd "${EP12_ROOT}"
module purge
module load python/3.12.1 \
  py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 \
  py-scikit-learn/1.5.1_py312 py-pandas/2.2.1_py312 \
  py-pyarrow/18.1.0_py312
export PYTHONPATH="${EP12_ROOT}/outputs/executor${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python3 -m ep12_executor qualify-methods \
  --tier "${EP12_TIER}" \
  --output-dir "${EP12_OUTPUT}" \
  --policy "${EP12_ROOT}/SEARCH_POLICY.yaml" \
  --resource-authorization "${EP12_ROOT}/outputs/executor/DEVELOPMENT_BUDGET_OVERRIDE.yaml" \
  --contract "${EP12_ROOT}/outputs/executor/SCIENTIFIC_CONTRACT_POST36_V3.yaml" \
  --qualification-spec "${EP12_ROOT}/outputs/executor/SYNTHETIC_QUALIFICATION_SPEC_POST36_V3.yaml" \
  --oracle-aggregate "${EP12_ORACLE_AGGREGATE}"
