#!/bin/bash
#SBATCH --job-name=ep12-readiness
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/readiness-%j.out

set -euo pipefail

EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_RUN_ROOT=${EP12_ROOT}/outputs/development_run002
EP12_QUALIFICATION_DIR=${1:-${EP12_ROOT}/outputs/scientific_qualification/qualification}
EP12_OUTPUT_DIR=${2:-${EP12_RUN_ROOT}/production_readiness}

cd "${EP12_ROOT}"
module purge
module load python/3.12.1 py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 \
  py-scikit-learn/1.5.1_py312 py-pandas/2.2.1_py312 py-pyarrow/18.1.0_py312
export PYTHONPATH="${EP12_ROOT}/outputs/executor${PYTHONPATH:+:${PYTHONPATH}}"

python3 -m ep12_executor.production_readiness \
  --qualification-dir "${EP12_QUALIFICATION_DIR}" \
  --observed-run-root "${EP12_RUN_ROOT}" \
  --observed-finalization-dir "${EP12_RUN_ROOT}/observed_finalization" \
  --falsifier-plan "${EP12_ROOT}/outputs/executor/OBSERVED_POST36_TAIL_PLAN.json" \
  --policy "${EP12_ROOT}/SEARCH_POLICY.yaml" \
  --prefix-contract "${EP12_ROOT}/outputs/executor/SCIENTIFIC_CONTRACT.yaml" \
  --active-contract "${EP12_ROOT}/outputs/executor/SCIENTIFIC_CONTRACT_POST36.yaml" \
  --budget-override "${EP12_ROOT}/outputs/executor/DEVELOPMENT_BUDGET_OVERRIDE.yaml" \
  --output-dir "${EP12_OUTPUT_DIR}" \
  --require-for observed_tail
