#!/bin/bash
# Proposed explicit continuation of resource-incomplete search47672861.
# Root confirms predecessor terminal, selects, and submits once. Reuse the
# recorded passing search/checkpoint checks; no tests or preparation here.
#SBATCH --job-name=ep18-false-resume
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/false_partition_preparation/search_continuation_operations_20261013_001/%x-%j.out
#SBATCH --error=outputs/runs/false_partition_preparation/search_continuation_operations_20261013_001/%x-%j.err
set -euo pipefail
umask 077
[[ $# == 0 ]]
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
: "${OAK:?}" "${GROUP_HOME:?}" "${L_SCRATCH_JOB:?}"
ep18_cpus=${SLURM_CPUS_PER_TASK:?Actual allocated CPUs required}
[[ "$ep18_cpus" =~ ^[0-9]+$ ]]
(( ep18_cpus >= 13 && ep18_cpus <= 16 ))
[[ "${SLURM_MEM_PER_NODE:-}" == 98304 && "${SLURM_JOB_NUM_NODES:-}" == 1 ]]
[[ "${SLURM_JOB_CPUS_PER_NODE:-}" == "$ep18_cpus" && "${SLURM_NTASKS:-}" == 1 ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
ep18_episode="$OAK/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_code="$ep18_episode/outputs/code"
ep18_config="$ep18_episode/outputs/config/false_partition_sub07_20261012_001.json"
ep18_cache="$ep18_episode/outputs/cache/false_partition_sub07_20261012_001"
ep18_prior="$ep18_episode/outputs/runs/false_partition_preparation/search_operations_20261013_001/initial_47672861/search"
ep18_operations_root="$ep18_episode/outputs/runs/false_partition_preparation/search_continuation_operations_20261013_001"
[[ -d "$ep18_operations_root" && -f "$ep18_cache/manifest.json" ]]
[[ -f "$ep18_prior/checkpoint.npz" && -f "$ep18_prior/report.json" ]]
[[ ! -e "$ep18_cache/progress.json.pending" && ! -e "$ep18_prior/checkpoint.npz.pending" ]]
ep18_ops="$ep18_operations_root/segment_${SLURM_JOB_ID}"
mkdir -m 700 "$ep18_ops"
ep18_started=$SECONDS
ep18_stage=allocation_admission
ep18_search_exit=-1
ep18_deadline=0
finish() {
  ep18_exit=$?
  trap - EXIT
  printf '{"stage":"%s","exit_code":%s,"elapsed_seconds":%s,"job_id":"%s","search_predecessor_job_id":"47672861","deadline_unix":%s,"allocated_cpus":%s,"memory_mib":98304,"numerical_workers":1,"reserved_CPU_seconds":%s,"search_exit":%s,"RSS_guard_bytes":96636764160,"array_admission_bytes":85899345920,"logical_counters_reset":false,"preparation_repeated":false,"tests_executed":false,"automatic_retry":false,"automatic_successor":false,"scientific_release_authorized":false}\n' \
    "$ep18_stage" "$ep18_exit" "$((SECONDS-ep18_started))" "$SLURM_JOB_ID" \
    "$ep18_deadline" "$ep18_cpus" "$((ep18_cpus*3600))" "$ep18_search_exit" > "$ep18_ops/exit.json"
  exit "$ep18_exit"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
ulimit -c 0
ep18_start=${SLURM_JOB_START_TIME:?Scheduler start epoch required}
ep18_end=${SLURM_JOB_END_TIME:?Scheduler end epoch required}
[[ "$ep18_start" =~ ^[0-9]+$ && "$ep18_end" =~ ^[0-9]+$ ]]
ep18_now=$(date +%s)
(( ep18_start > 0 && ep18_start <= ep18_now && ep18_end > ep18_now ))
ep18_deadline=$((ep18_start+3540))
if (( ep18_end-60 < ep18_deadline )); then ep18_deadline=$((ep18_end-60)); fi
if (( 1791926940 < ep18_deadline )); then ep18_deadline=1791926940; fi
(( ep18_deadline-ep18_now > 60 ))
printf 'hostname=%s job_id=%s start=%s end=%s deadline=%s CPUs=%s numerical_workers=1 memory_mib=98304 predecessor=47672861\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$ep18_start" "$ep18_end" "$ep18_deadline" "$ep18_cpus"

ep18_stage=selected_runtime
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1
unset PYTHONPATH PYTHONHOME
ep18_python="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/bin/python"
[[ -x "$ep18_python" ]]
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
ep18_local="$L_SCRATCH_JOB/ep18-false-search-continue001-${SLURM_JOB_ID}"
mkdir -m 700 "$ep18_local"
mkdir -m 700 "$ep18_local/tmp" "$ep18_local/cache" "$ep18_local/hf"
export TMPDIR="$ep18_local/tmp" XDG_CACHE_HOME="$ep18_local/cache" HF_HOME="$ep18_local/hf"
cd "$ep18_local"
ep18_now=$(date +%s)
(( ep18_deadline-ep18_now > 20 ))
# The first segment wrote its cooperative stop report7seconds after deadline;
# its5second external tail produced timeout124 despite a complete checkpoint.
# Permit40seconds here plus5seconds kill grace, still inside the60second tail
# reserved before both the actual allocation end and original episode end.
ep18_timeout=$((ep18_deadline+40-ep18_now))
ep18_stage=resume_saved_search_state
set +e
timeout --signal=TERM --kill-after=5s "${ep18_timeout}s" "$ep18_python" -B \
  "$ep18_code/false_partition_search.py" search --config "$ep18_config" \
  --output "$ep18_ops/search" --resume "$ep18_prior/checkpoint.npz" \
  --deadline "$ep18_deadline" --rss-byte-cap 96636764160 --array-byte-cap 85899345920 \
  > "$ep18_ops/search.txt" 2> "$ep18_ops/search_stderr.txt"
ep18_search_exit=$?
set -e
case "$ep18_search_exit" in
  0) ep18_stage=image_context_candidate_bank_complete ;;
  75) ep18_stage=search_resource_incomplete_checkpoint_if_written ;;
  2) ep18_stage=frozen_search_budget_exhausted ;;
  124|137) ep18_stage=outer_timeout_requires_saved_report_review ;;
  *) ep18_stage=search_failed_requires_diagnosis ;;
esac
exit "$ep18_search_exit"
