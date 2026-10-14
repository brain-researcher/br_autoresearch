#!/bin/bash
# INACTIVE template. Requires separately adopted AUTHORITY_JSON as sole argument.
# No authority is created here; no automatic retry, successor or requeue.
#SBATCH --job-name=ep18-false-bounded
#SBATCH --partition=russpold,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --no-requeue
#SBATCH --output=ep18-false-bounded-%j.out
#SBATCH --error=ep18-false-bounded-%j.err
set -euo pipefail
umask 077
[[ $# == 1 && "$1" == /* && -f "$1" ]]
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
: "${OAK:?}" "${GROUP_HOME:?}" "${L_SCRATCH_JOB:?}"
ep18_authority="$1"
ep18_episode="$OAK/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_adapter="$ep18_episode/outputs/proposed/false_partition_continuation/continue_search.py"
ep18_python="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/bin/python"
[[ -x "$ep18_python" ]]
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
ulimit -c 0
ep18_start=${SLURM_JOB_START_TIME:?}
ep18_end=${SLURM_JOB_END_TIME:?}
ep18_deadline=$("$ep18_python" -B - "$ep18_authority" "$ep18_start" "$ep18_end" <<'DEADLINE'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
assert p.stat().st_size <= 65536
a = json.loads(p.read_text())
assert a['status'] == 'adopted' and a['scope'] == 'bounded_generator_continuation'
assert type(a['new_absolute_deadline_unix']) is int
print(min(int(sys.argv[2])+32400, int(sys.argv[3])-60, a['new_absolute_deadline_unix']-60))
DEADLINE
)
ep18_now=$(date +%s)
(( ep18_deadline-ep18_now > 60 ))
ep18_local="$L_SCRATCH_JOB/ep18-false-bounded-${SLURM_JOB_ID}"
mkdir -m 700 "$ep18_local"
mkdir -m 700 "$ep18_local/tmp" "$ep18_local/cache" "$ep18_local/hf"
export TMPDIR="$ep18_local/tmp" XDG_CACHE_HOME="$ep18_local/cache" HF_HOME="$ep18_local/hf"
ep18_operations="$ep18_episode/outputs/runs/false_partition_preparation/bounded_continuation_operations"
mkdir -p -m 700 "$ep18_operations"
ep18_ops="$ep18_operations/segment_${SLURM_JOB_ID}"
mkdir -m 700 "$ep18_ops"
ep18_stage=setup
ep18_search_exit=-1
finish() {
  ep18_exit=$?
  trap - EXIT
  printf 'exit=%s\nstage=%s\nsearch_exit=%s\nend=%s\n' \
    "$ep18_exit" "$ep18_stage" "$ep18_search_exit" "$(date -Is)" >> "$ep18_ops/status.txt"
  exit "$ep18_exit"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
printf 'host=%s job=%s requested_CPUs=%s actual_CPUs=%s start=%s end=%s deadline=%s authority=%s\n' \
  "$(hostname)" "$SLURM_JOB_ID" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_JOB_CPUS_PER_NODE:?}" \
  "$ep18_start" "$ep18_end" "$ep18_deadline" "$ep18_authority" > "$ep18_ops/status.txt"
cd "$ep18_local"
ep18_now=$(date +%s)
(( ep18_deadline-ep18_now > 60 ))
ep18_stage=authority_admission_and_search
set +e
timeout --signal=TERM --kill-after=5s "$((ep18_deadline+40-ep18_now))s" \
  "$ep18_python" -B "$ep18_adapter" --authority "$ep18_authority" \
  --output "$ep18_ops/search" --deadline "$ep18_deadline" \
  > "$ep18_ops/search.txt" 2> "$ep18_ops/search_stderr.txt"
ep18_search_exit=$?
set -e
case "$ep18_search_exit" in
  0) ep18_stage=image_context_candidate_bank_complete ;;
  75) ep18_stage=resource_incomplete_review_checkpoint_availability ;;
  2) ep18_stage=frozen_generator_exhausted ;;
  124|137) ep18_stage=outer_timeout_requires_saved_report_review ;;
  *) ep18_stage=refused_or_failed_requires_diagnosis ;;
esac
exit "$ep18_search_exit"
