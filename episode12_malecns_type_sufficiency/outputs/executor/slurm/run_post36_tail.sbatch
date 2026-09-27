#!/bin/bash
#SBATCH --job-name=ep12-post36-tail
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/outputs/development_run002/slurm-post36-tail-%j.out

# One invocation realizes at most one valid development-only slot. A successful
# continuation submits the same bounded controller as an afterok child; stops,
# failures, and wall-deadline decisions never submit another slot.
set -euo pipefail

EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_SCRATCH=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_RUN_ROOT=${EP12_ROOT}/outputs/development_run002

cd "${EP12_ROOT}"
module purge
module load python/3.12.1 \
  py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 \
  py-scikit-learn/1.5.1_py312 py-pandas/2.2.1_py312 \
  py-pyarrow/18.1.0_py312
export PYTHONPATH="${EP12_ROOT}/outputs/executor${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${EP12_SCRATCH}/development_run002/post36_tail"

set +e
python3 -m ep12_executor.post36_tail run-one \
  --plan "${EP12_ROOT}/outputs/executor/OBSERVED_POST36_TAIL_PLAN.json" \
  --policy "${EP12_ROOT}/SEARCH_POLICY.yaml" \
  --prefix-contract "${EP12_ROOT}/outputs/executor/SCIENTIFIC_CONTRACT.yaml" \
  --active-contract "${EP12_ROOT}/outputs/executor/SCIENTIFIC_CONTRACT_POST36.yaml" \
  --readiness-dir "${EP12_RUN_ROOT}/production_readiness" \
  --run-root "${EP12_RUN_ROOT}" \
  --observed-finalization-dir "${EP12_RUN_ROOT}/observed_finalization" \
  --tail-root "${EP12_RUN_ROOT}/post36_tail" \
  --snapshot-dir "${EP12_SCRATCH}/development_snapshot_v1" \
  --scratch-root "${EP12_SCRATCH}/development_run002/post36_tail"
EP12_CONTROLLER_RC=$?
set -e

if [[ "${EP12_CONTROLLER_RC}" -eq 10 ]]; then
  : "${SLURM_JOB_ID:?continuation requires a Slurm job ID}"
  sbatch --dependency="afterok:${SLURM_JOB_ID}" "$0"
  exit 0
fi
if [[ "${EP12_CONTROLLER_RC}" -eq 0 || "${EP12_CONTROLLER_RC}" -eq 12 ]]; then
  exit 0
fi
exit "${EP12_CONTROLLER_RC}"
