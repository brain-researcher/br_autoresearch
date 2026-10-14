#!/bin/bash
# Resource-balanced replacements for original array indices 27 and 28 only.
# Root verifies both original workers are terminal before releasing this array.
# Same frozen mapping, model, private checkpoint paths and scoring; no subset.
#SBATCH --job-name=ep18-core-balanced
#SBATCH --partition=russpold,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=27-28%2
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/baseline_models/core_direct_balanced_operations001/slurm-%A_%a.out
#SBATCH --error=outputs/runs/baseline_models/core_direct_balanced_operations001/slurm-%A_%a.err
set -euo pipefail
umask 077
[[ $# -eq 0 ]] || exit 2
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_ARRAY_JOB_ID:-}" && "$(hostname)" != *-ln* ]] || exit 2
[[ "${SLURM_ARRAY_TASK_ID:-}" == 27 || "${SLURM_ARRAY_TASK_ID:-}" == 28 ]] || exit 2
[[ "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 32768 ]] || exit 2
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]] || exit 2
: "${OAK:?}" "${GROUP_HOME:?}" "${L_SCRATCH_JOB:?}"
ep18_balanced_episode="$OAK/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_balanced_code="$ep18_balanced_episode/outputs/code"
ep18_balanced_operations="$ep18_balanced_episode/outputs/runs/baseline_models/core_direct_balanced_operations001"
ep18_balanced_local="$L_SCRATCH_JOB/ep18-core-balanced-${SLURM_ARRAY_TASK_ID}"
mkdir -m 700 "$ep18_balanced_local"
mkdir -m 700 "$ep18_balanced_local/tmp" "$ep18_balanced_local/cache"
export TMPDIR="$ep18_balanced_local/tmp" XDG_CACHE_HOME="$ep18_balanced_local/cache"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8 BLIS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1
unset PYTHONPATH PYTHONHOME
ep18_balanced_python="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/bin/python"
[[ -x "$ep18_balanced_python" ]] || exit 2
ep18_balanced_deadline=$((${SLURM_JOB_END_TIME:?}-90))
ep18_balanced_timeout=$((ep18_balanced_deadline-$(date +%s)+15))
(( ep18_balanced_timeout > 20 )) || exit 124
printf 'hostname=%s\njob_id=%s\narray_index=%s\nallocated_cpus=%s\nmemory_mib=%s\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$SLURM_CPUS_PER_TASK" "$SLURM_MEM_PER_NODE"
cd "$ep18_balanced_local"
# Reuse passed backend checks. The unchanged worker verifies settings/recipe
# bindings and reuses only complete inner-pair and outer-fold checkpoints.
timeout --signal=TERM --kill-after=15s "${ep18_balanced_timeout}s" "$ep18_balanced_python" -B \
  "$ep18_balanced_code/production_core_direct_array.py" \
  --initialized-root "$ep18_balanced_episode/outputs/cache/private/core_reference_20261013_001" \
  --run-root "$ep18_balanced_episode/outputs/cache/private/core_direct001" \
  --public-root "$ep18_balanced_operations" --deadline-epoch "$ep18_balanced_deadline" \
  --threads 8 --rss-limit-bytes 30064771072
