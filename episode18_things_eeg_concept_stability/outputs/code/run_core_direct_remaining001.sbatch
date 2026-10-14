#!/bin/bash
# Root selects final walltime from the completed first-pair measurement.
# 29 frozen development participants, excluding separately owned sub-07.
#SBATCH --job-name=ep18-core-remaining
#SBATCH --partition=russpold,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-28%6
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/baseline_models/core_direct_remaining_operations001/slurm-%A_%a.out
#SBATCH --error=outputs/runs/baseline_models/core_direct_remaining_operations001/slurm-%A_%a.err
set -euo pipefail
umask 077
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_ARRAY_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]]
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < 29 ))
[[ "${SLURM_CPUS_PER_TASK:-}" == 16 && "${SLURM_MEM_PER_NODE:-}" == 32768 ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]]
: "${OAK:?}" "${GROUP_HOME:?}" "${L_SCRATCH_JOB:?}"
ep18_episode="$OAK/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_code="$ep18_episode/outputs/code"
ep18_operations="$ep18_episode/outputs/runs/baseline_models/core_direct_remaining_operations001"
ep18_local="$L_SCRATCH_JOB/ep18-core-remaining-${SLURM_ARRAY_TASK_ID}"
mkdir -m 700 "$ep18_local"
mkdir -m 700 "$ep18_local/tmp" "$ep18_local/cache"
export TMPDIR="$ep18_local/tmp" XDG_CACHE_HOME="$ep18_local/cache"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16 BLIS_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1
unset PYTHONPATH PYTHONHOME
ep18_python="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/bin/python"
[[ -x "$ep18_python" ]]
ep18_deadline=$((${SLURM_JOB_END_TIME:?}-90))
ep18_timeout=$((ep18_deadline-$(date +%s)+15))
(( ep18_timeout > 20 ))
printf 'hostname=%s\njob_id=%s\narray_index=%s\nallocated_cpus=%s\nmemory_mib=%s\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$SLURM_CPUS_PER_TASK" "$SLURM_MEM_PER_NODE"
cd "$ep18_local"
# Reuse the first job's completed backend checks; each worker retains exactly
# the original scientific analysis and owns only its selected participant.
timeout --signal=TERM --kill-after=15s "${ep18_timeout}s" "$ep18_python" -B \
  "$ep18_code/production_core_direct_array.py" \
  --initialized-root "$ep18_episode/outputs/cache/private/core_reference_20261013_001" \
  --run-root "$ep18_episode/outputs/cache/private/core_direct001" \
  --public-root "$ep18_operations" --deadline-epoch "$ep18_deadline" \
  --threads 16 --rss-limit-bytes 30064771072
