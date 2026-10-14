#!/bin/bash
# Dependency closes all original writers before this batch may start.
# Pass potential remaining IDs explicitly: sub-27, sub-31, and/or sub-47.
# A complete.json file skips only fitting; the reporter validates all 30.
# One real Slurm job runs them sequentially, retaining every model and fold.
#SBATCH --job-name=ep18-core-checkpoint-batch
#SBATCH --partition=russpold,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --dependency=afterany:47717611_13:47717611_17:47717611_26
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/baseline_models/core_direct_checkpoint_batch_operations001/slurm-%j.out
#SBATCH --error=outputs/runs/baseline_models/core_direct_checkpoint_batch_operations001/slurm-%j.err
set -euo pipefail
umask 077
(( $# >= 1 && $# <= 3 )) || exit 2
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" && "$(hostname)" != *-ln* ]] || exit 2
[[ "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 32768 ]] || exit 2
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]] || exit 2
declare -A ep18_batch_seen=()
for ep18_batch_person in "$@"; do
  case "$ep18_batch_person" in
    sub-27|sub-31|sub-47) ;;
    *) exit 2 ;;
  esac
  [[ -z "${ep18_batch_seen[$ep18_batch_person]:-}" ]] || exit 2
  ep18_batch_seen[$ep18_batch_person]=1
done
: "${OAK:?}" "${GROUP_HOME:?}" "${L_SCRATCH_JOB:?}"
ep18_batch_episode="$OAK/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_batch_code="$ep18_batch_episode/outputs/code"
ep18_batch_operations="$ep18_batch_episode/outputs/runs/baseline_models/core_direct_checkpoint_batch_operations001"
ep18_batch_report="$ep18_batch_episode/outputs/runs/baseline_models/core_direct_checkpoint_batch_report001"
ep18_batch_figures="$ep18_batch_episode/outputs/figures/core_direct_checkpoint_batch001"
ep18_batch_local="$L_SCRATCH_JOB/ep18-core-checkpoint-batch"
mkdir -m 700 "$ep18_batch_local"
mkdir -m 700 "$ep18_batch_local/tmp" "$ep18_batch_local/cache" "$ep18_batch_local/matplotlib"
export TMPDIR="$ep18_batch_local/tmp" XDG_CACHE_HOME="$ep18_batch_local/cache"
export MPLCONFIGDIR="$ep18_batch_local/matplotlib"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8 BLIS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1
unset PYTHONPATH PYTHONHOME
ep18_batch_python="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/bin/python"
[[ -x "$ep18_batch_python" ]] || exit 2
ep18_batch_deadline=$((${SLURM_JOB_END_TIME:?}-90))
printf 'hostname=%s\njob_id=%s\nallocated_cpus=%s\nmemory_mib=%s\nshared_deadline=%s\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK" "$SLURM_MEM_PER_NODE" "$ep18_batch_deadline"
cd "$ep18_batch_local"
for ep18_batch_person in "$@"; do
  # File-existence metadata only: do not open participant losses here. The
  # dependency ensures the prior writer has ended before this decision.
  if [[ -f "$ep18_batch_episode/outputs/cache/private/core_direct001/$ep18_batch_person/complete.json" ]]; then
    printf 'skipped_completed_participant=%s\n' "$ep18_batch_person"
    continue
  fi
  # Recalculate remaining time; no participant receives a restarted clock.
  ep18_batch_timeout=$((ep18_batch_deadline-$(date +%s)+15))
  (( ep18_batch_timeout > 20 )) || exit 124
  printf 'continuing_participant=%s\n' "$ep18_batch_person"
  # The unchanged worker validates every retained pair/fold against the
  # original settings/recipe binding; no models or folds are removed.
  timeout --signal=TERM --kill-after=15s "${ep18_batch_timeout}s" "$ep18_batch_python" -B \
    "$ep18_batch_code/production_core_direct.py" \
    --initialized-root "$ep18_batch_episode/outputs/cache/private/core_reference_20261013_001" \
    --run-root "$ep18_batch_episode/outputs/cache/private/core_direct001" \
    --public-root "$ep18_batch_operations" --participant-id "$ep18_batch_person" \
    --deadline-epoch "$ep18_batch_deadline" --threads 8 --rss-limit-bytes 30064771072
done
# The coordinator records this contingent all-30 release before submission.
# The unchanged reporter first requires every original completion file, then
# validates each participant's complete original six-fold/model record before
# producing descriptive participant intervals, one figure and three lines.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 viz py-matplotlib/3.8.3_py312
ep18_batch_report_timeout=$((ep18_batch_deadline-$(date +%s)))
(( ep18_batch_report_timeout > 0 )) || exit 124
if (( ep18_batch_report_timeout > 540 )); then ep18_batch_report_timeout=540; fi
mkdir -m 700 "$ep18_batch_report"
timeout --signal=TERM --kill-after=5s "${ep18_batch_report_timeout}s" python3.12 -B \
  "$ep18_batch_code/report_core_direct.py" \
  --initialized-root "$ep18_batch_episode/outputs/cache/private/core_reference_20261013_001" \
  --run-root "$ep18_batch_episode/outputs/cache/private/core_direct001" \
  --output-root "$ep18_batch_report" --figure-root "$ep18_batch_figures" \
  > "$ep18_batch_report/execution.txt" 2>&1
printf 'completed_all30_report=%s\n' "$ep18_batch_report"
