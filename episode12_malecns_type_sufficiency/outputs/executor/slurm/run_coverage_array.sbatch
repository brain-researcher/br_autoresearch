#!/bin/bash
#SBATCH --job-name=ep12-coverage
#SBATCH --partition=normal
#SBATCH --array=0-17
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/outputs/development_run002/slurm-coverage-%A_%a.out

set -euo pipefail

EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_SCRATCH=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_TRIAL_ID=$(printf 'coverage_%03d' "$((SLURM_ARRAY_TASK_ID + 1))")
EP12_WORK_DIR="${EP12_SCRATCH}/development_run002/observed/${EP12_TRIAL_ID}.${SLURM_ARRAY_JOB_ID}"
EP12_FINAL_DIR="${EP12_ROOT}/outputs/development_run002/observed/${EP12_TRIAL_ID}"
EP12_STAGE_DIR="${EP12_FINAL_DIR}.staging-${SLURM_ARRAY_JOB_ID}"

if [[ -e "${EP12_FINAL_DIR}" || -e "${EP12_STAGE_DIR}" ]]; then
  echo "refusing to overwrite an existing durable trial directory" >&2
  exit 2
fi
if [[ -d "${EP12_WORK_DIR}" ]] && [[ -n "$(ls -A "${EP12_WORK_DIR}")" ]]; then
  echo "refusing to reuse a nonempty scratch trial directory" >&2
  exit 2
fi
mkdir -p "${EP12_WORK_DIR}"
mkdir -p "${EP12_ROOT}/outputs/development_run002/observed"
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
  --output-dir "${EP12_WORK_DIR}" \
  --snapshot-dir "${EP12_SCRATCH}/development_snapshot_v1" \
  --config-manifest "${EP12_ROOT}/outputs/development_run002/search_plan" \
  --config-index "${SLURM_ARRAY_TASK_ID}"

mkdir "${EP12_STAGE_DIR}"
cp -a "${EP12_WORK_DIR}/." "${EP12_STAGE_DIR}/"
mv "${EP12_STAGE_DIR}" "${EP12_FINAL_DIR}"
