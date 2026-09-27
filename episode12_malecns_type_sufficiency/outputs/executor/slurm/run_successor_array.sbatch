#!/bin/bash
#SBATCH --job-name=ep12-successor
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/outputs/development_run002/slurm-successor-%A_%a.out

set -euo pipefail

: "${EP12_CYCLE:?EP12_CYCLE must be 1 or 2}"
if [[ "${EP12_CYCLE}" != "1" && "${EP12_CYCLE}" != "2" ]]; then
  echo "EP12_CYCLE must be 1 or 2" >&2
  exit 2
fi

EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_SCRATCH=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_RUN_ROOT="${EP12_ROOT}/outputs/development_run002"
EP12_TRIAL_ID=$(printf 'adaptive_c%d_%03d' "${EP12_CYCLE}" "$((SLURM_ARRAY_TASK_ID + 1))")
EP12_WORK_DIR="${EP12_SCRATCH}/development_run002/adaptive/cycle${EP12_CYCLE}/${EP12_TRIAL_ID}.${SLURM_ARRAY_JOB_ID}"
EP12_FINAL_DIR="${EP12_RUN_ROOT}/adaptive/cycle${EP12_CYCLE}/observed/${EP12_TRIAL_ID}"
EP12_STAGE_DIR="${EP12_FINAL_DIR}.staging-${SLURM_ARRAY_JOB_ID}"
EP12_CONFIG_PATH="${EP12_RUN_ROOT}/adaptive/cycle${EP12_CYCLE}/plan/successor_configurations.json"

if [[ -e "${EP12_FINAL_DIR}" || -e "${EP12_STAGE_DIR}" ]]; then
  echo "refusing to overwrite an existing durable successor directory" >&2
  exit 2
fi
if [[ -d "${EP12_WORK_DIR}" ]] && [[ -n "$(ls -A "${EP12_WORK_DIR}")" ]]; then
  echo "refusing to reuse a nonempty successor scratch directory" >&2
  exit 2
fi
mkdir -p "${EP12_WORK_DIR}"
mkdir -p "${EP12_RUN_ROOT}/adaptive/cycle${EP12_CYCLE}/observed"
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

srun --ntasks=1 python3 -m ep12_executor.adaptive_trial \
  --configuration-path "${EP12_CONFIG_PATH}" \
  --config-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir "${EP12_WORK_DIR}" \
  --snapshot-dir "${EP12_SCRATCH}/development_snapshot_v1"

mkdir "${EP12_STAGE_DIR}"
cp -a "${EP12_WORK_DIR}/." "${EP12_STAGE_DIR}/"
mv "${EP12_STAGE_DIR}" "${EP12_FINAL_DIR}"
