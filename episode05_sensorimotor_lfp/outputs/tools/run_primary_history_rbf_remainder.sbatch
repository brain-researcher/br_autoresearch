#!/bin/bash
#SBATCH --job-name=ep05-hw105
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:35:00
#SBATCH --no-requeue
#SBATCH --chdir=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode05_sensorimotor_lfp
#SBATCH --output=outputs/runs/adaptive_search/primary_history_rbf_submission_attempt001/slurm_%j.out
#SBATCH --error=outputs/runs/adaptive_search/primary_history_rbf_submission_attempt001/slurm_%j.err

# Exactly fifteen fresh session banks plus the preserved fixed pilot; no retry.
set -eo pipefail
EP05_HW105_HOST=$(hostname)
printf 'hostname=%s SLURM_JOB_ID=%s SLURM_CPUS_PER_TASK=%s\n' \
  "$EP05_HW105_HOST" "${SLURM_JOB_ID-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP05_HW105_HOST" == *-ln* ]]; then
  printf 'HW105 execution requires its verified Slurm compute allocation.\n' >&2
  exit 2
fi
if [[ "${SLURM_CPUS_PER_TASK-}" != 1 || -n "${SLURM_ARRAY_TASK_ID-}" ]]; then
  printf 'This bounded remainder requires one CPU and is not an array.\n' >&2
  exit 2
fi
: "${OAK:?OAK must be set by Sherlock}"
cd "$OAK/users/zijiao/br_autoresearch/episode05_sensorimotor_lfp"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

source /etc/profile >/dev/null
set -u
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scipy/1.12.0_py312 py-scikit-learn/1.5.1_py312 py-h5py/3.10.0_py312
EP05_HW105_PYTHON=$(command -v python3.12)
module list
printf 'selected_python=%s\n' "$EP05_HW105_PYTHON"
date -u '+remainder_start_utc=%Y-%m-%dT%H:%M:%SZ'
exec timeout --signal=TERM --kill-after=10s 2040s \
  "$EP05_HW105_PYTHON" outputs/tools/run_primary_history_rbf_panel.py remainder \
  --source "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --trial-table outputs/runs/source_support/attempt002/trial_identity.csv \
  --scalar-bank-root outputs/runs/adaptive_search/primary_reference_attempt001 \
  --hw-bank-root outputs/runs/adaptive_search/primary_history_repeatability_weight_control_attempt002 \
  --k-bank-root outputs/runs/adaptive_search/primary_capacity_attempt001 \
  --blend-bank-root outputs/runs/adaptive_search/primary_history_blend_attempt001 \
  --hw-summary outputs/runs/adaptive_search/primary_history_repeatability_weight_summary_attempt002/primary_history_repeatability_weight.json \
  --k-summary outputs/runs/adaptive_search/primary_capacity_summary_attempt001/primary_capacity.json \
  --blend-summary outputs/runs/adaptive_search/primary_history_blend_summary_attempt001/primary_history_blend.json \
  --bank-root outputs/runs/adaptive_search/primary_history_rbf_attempt001 \
  --summary-out outputs/runs/adaptive_search/primary_history_rbf_summary_attempt001 \
  --out outputs/runs/adaptive_search/primary_history_rbf_remainder_attempt001 \
  --workers 1 --max-seconds 1980
