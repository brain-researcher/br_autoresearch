#!/bin/bash
#SBATCH --job-name=ep12-plan-next
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/outputs/development_run002/slurm-plan-successor-%j.out

set -euo pipefail

: "${EP12_CYCLE:?EP12_CYCLE must be 1 or 2}"
if [[ "${EP12_CYCLE}" != "1" && "${EP12_CYCLE}" != "2" ]]; then
  echo "EP12_CYCLE must be 1 or 2" >&2
  exit 2
fi

EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_SCRATCH=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_RUN_ROOT="${EP12_ROOT}/outputs/development_run002"

mkdir -p "${EP12_RUN_ROOT}/adaptive/cycle${EP12_CYCLE}"
mkdir -p "${EP12_SCRATCH}/development_run002"
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

EP12_SOURCE_ARGS=(
  --source
  "${EP12_RUN_ROOT}/search_plan/coverage_configurations.json"
  "${EP12_RUN_ROOT}/observed"
)
if [[ "${EP12_CYCLE}" == "2" ]]; then
  EP12_SOURCE_ARGS+=(
    --source
    "${EP12_RUN_ROOT}/adaptive/cycle1/plan/successor_configurations.json"
    "${EP12_RUN_ROOT}/adaptive/cycle1/observed"
  )
fi

srun --ntasks=1 python3 -m ep12_executor.adaptive_successors \
  --cycle "${EP12_CYCLE}" \
  "${EP12_SOURCE_ARGS[@]}" \
  --output-dir "${EP12_RUN_ROOT}/adaptive/cycle${EP12_CYCLE}/plan"
