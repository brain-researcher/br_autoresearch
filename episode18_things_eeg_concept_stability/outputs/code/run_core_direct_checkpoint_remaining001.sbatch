#!/bin/bash
# Coordinator verifies each selected predecessor is terminal and incomplete.
# Original array indices 9,11,13,15,17,26 map to sub-22,25,27,29,31,47.
# Narrow --array at submission if any finishes; completed people must not rerun.
# Same frozen mapping, all models/folds, and original private checkpoint root.
#SBATCH --job-name=ep18-core-checkpoint
#SBATCH --partition=russpold,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=9,11,13,15,17,26%6
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/baseline_models/core_direct_checkpoint_operations001/slurm-%A_%a.out
#SBATCH --error=outputs/runs/baseline_models/core_direct_checkpoint_operations001/slurm-%A_%a.err
set -euo pipefail
umask 077
[[ $# -eq 0 ]] || exit 2
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_ARRAY_JOB_ID:-}" && "$(hostname)" != *-ln* ]] || exit 2
case "${SLURM_ARRAY_TASK_ID:-}" in
  9|11|13|15|17|26) ;;
  *) exit 2 ;;
esac
[[ "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 32768 ]] || exit 2
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]] || exit 2
: "${OAK:?}" "${GROUP_HOME:?}" "${L_SCRATCH_JOB:?}"
ep18_checkpoint_episode="$OAK/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_checkpoint_code="$ep18_checkpoint_episode/outputs/code"
ep18_checkpoint_operations="$ep18_checkpoint_episode/outputs/runs/baseline_models/core_direct_checkpoint_operations001"
ep18_checkpoint_local="$L_SCRATCH_JOB/ep18-core-checkpoint-${SLURM_ARRAY_TASK_ID}"
mkdir -m 700 "$ep18_checkpoint_local"
mkdir -m 700 "$ep18_checkpoint_local/tmp" "$ep18_checkpoint_local/cache"
export TMPDIR="$ep18_checkpoint_local/tmp" XDG_CACHE_HOME="$ep18_checkpoint_local/cache"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8 BLIS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1
unset PYTHONPATH PYTHONHOME
ep18_checkpoint_python="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/bin/python"
[[ -x "$ep18_checkpoint_python" ]] || exit 2
ep18_checkpoint_deadline=$((${SLURM_JOB_END_TIME:?}-90))
ep18_checkpoint_timeout=$((ep18_checkpoint_deadline-$(date +%s)+15))
(( ep18_checkpoint_timeout > 20 )) || exit 124
printf 'hostname=%s\njob_id=%s\narray_index=%s\nallocated_cpus=%s\nmemory_mib=%s\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$SLURM_CPUS_PER_TASK" "$SLURM_MEM_PER_NODE"
cd "$ep18_checkpoint_local"
# Passed backend checks are reused. The unchanged worker checks original
# participant/settings/recipe bindings before consuming saved pair/fold fits.
timeout --signal=TERM --kill-after=15s "${ep18_checkpoint_timeout}s" "$ep18_checkpoint_python" -B \
  "$ep18_checkpoint_code/production_core_direct_array.py" \
  --initialized-root "$ep18_checkpoint_episode/outputs/cache/private/core_reference_20261013_001" \
  --run-root "$ep18_checkpoint_episode/outputs/cache/private/core_direct001" \
  --public-root "$ep18_checkpoint_operations" --deadline-epoch "$ep18_checkpoint_deadline" \
  --threads 8 --rss-limit-bytes 30064771072
