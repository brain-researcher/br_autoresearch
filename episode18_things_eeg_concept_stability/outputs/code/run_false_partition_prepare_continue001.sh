#!/bin/bash
# Selected continuation; coordinator owns submission and accounting.
# inspect: metadata counts only, in any coordinator-owned compute allocation.
# prepare: one explicitly selected 4CPU/28GiB/4h allocation, one worker/thread.
# No search, tests, input recreation, automatic retry or successor is invoked.
# Coordinator must verify the predecessor terminal and sole cache ownership.
set -euo pipefail
umask 077
[[ $# == 1 && ( "$1" == inspect || "$1" == prepare ) ]]
ep18_mode=$1
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
: "${OAK:?}" "${GROUP_HOME:?}"
ep18_episode="$OAK/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_code="$ep18_episode/outputs/code"
ep18_config="$ep18_episode/outputs/config/false_partition_sub07_20261012_001.json"
ep18_cache="$ep18_episode/outputs/cache/false_partition_sub07_20261012_001"
ep18_prior="$ep18_episode/outputs/runs/false_partition_preparation/false_partition_operations_20261012_001/initial_47584692"

source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1
ep18_python="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/bin/python"
[[ -x "$ep18_python" ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''

# No array/archive, source event, EEG, feature matrix or candidate score is read.
# The existing config digest includes the unchanged producer's source hash.
# Only complete-count/row-count and operational resource facts are projected.
project_metadata() {
  local ep18_projection_seconds=$1
  timeout --signal=TERM --kill-after=5s "${ep18_projection_seconds}s" "$ep18_python" -B - \
    "$ep18_config" "$ep18_cache" "$ep18_prior" "$ep18_code/false_partition_search.py" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

config_path, cache, prior, producer = map(Path, sys.argv[1:])
def read(path):
    with path.open('rb') as source:
        data = source.read(8*1024**2+1)
    if len(data) > 8*1024**2:
        raise ValueError('bounded_metadata_required')
    return json.loads(data)

config = read(config_path)
with producer.open('rb') as source:
    code = source.read(1024**2+1)
if len(code) > 1024**2:
    raise ValueError('bounded_existing_producer_required')
config['code_sha256'] = hashlib.sha256(code).hexdigest()
if config['code_sha256'] != 'c8cfdb1703711d81ad9e3f4d1ed774f45b7958b654eeba1a4fd9d87ca60c0c3d':
    raise ValueError('unchanged_original_producer_required')
saved = read(prior/'prepare'/'settings.json')
original = read(prior/'prepare'/'report.json')
exit_record = read(prior/'exit.json')
if not (saved['phase'] == 'prepare' and saved['config'] == config
        and config['cache_dir'] == str(cache) and config['participant_id'] == 'sub-07'
        and config['not_after_unix'] == 1791927000
        and config['cache_byte_cap'] == 77309411328
        and config['segment_byte_cap'] == 67108864 and config['pair_block_rows'] == 128
        and original['status'] == 'resource_incomplete'
        and original['budget_stop_reason'] == 'time_or_signal'
        and original['phase'] == 'prepare' and original['EEG_read'] is False
        and exit_record['job_id'] == '47584692' and exit_record['exit_code'] == 75
        and exit_record['prepare_exit'] == 75 and exit_record['search_exit'] == -1):
    raise ValueError('diagnosed_original_preparation_stop_required')
config_id = hashlib.sha256((json.dumps(config,sort_keys=True,separators=(',', ':'),
                                      allow_nan=False)+'\n').encode()).hexdigest()
settings, progress = read(cache/'settings.json'), read(cache/'progress.json')
if not (settings['config'] == config and settings['config_id'] == config_id
        and progress['config_id'] == config_id):
    raise ValueError('original_cache_configuration_binding_required')
pairs, contexts = progress['completed_pairs'], progress['completed_contexts']
row, active = progress['active_pair_row'], progress['active_pair'] is not None
if not (isinstance(pairs,dict) and isinstance(contexts,dict)
        and len(pairs) <= 36 and len(contexts) <= 6 and type(row) is int
        and 0 <= row <= 22248 and (active or row == 0)):
    raise ValueError('bounded_original_progress_counts_required')
print(json.dumps({'completed_pair_caches':len(pairs),'required_pair_caches':36,
    'active_pair_present':active,'active_pair_completed_rows':row,
    'rows_per_pair':22248,'completed_context_folds':len(contexts),'required_context_folds':6,
    'complete_cache_manifest_present':(cache/'manifest.json').exists(),
    'pending_progress_metadata_present':(cache/'progress.json.pending').exists(),
    'prior_prepare_seconds':original['elapsed_seconds'],
    'prior_peak_rss_bytes':original['process_peak_rss_bytes'],
    'original_config_and_producer_unchanged':True,
    'payload_arrays_read':False,'EEG_read':False,'scientific_result':False},sort_keys=True))
PY
}

if [[ "$ep18_mode" == inspect ]]; then
  project_metadata 25
  exit 0
fi

[[ "${SLURM_CPUS_PER_TASK:-}" == 4 && "${SLURM_MEM_PER_NODE:-}" == 28672 ]]
[[ "${SLURM_NTASKS:-}" == 1 && "${SLURM_JOB_NUM_NODES:-}" == 1 && "${SLURM_JOB_CPUS_PER_NODE:-}" == 4 ]]
[[ -z "${SLURM_ARRAY_TASK_ID:-}" && -z "${SLURM_ARRAY_JOB_ID:-}" ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" && "${SLURM_GPUS:-0}" == 0 ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 && "$SLURM_JOB_ID" =~ ^[1-9][0-9]*$ ]]
: "${L_SCRATCH_JOB:?}" "${SLURM_JOB_START_TIME:?}" "${SLURM_JOB_END_TIME:?}"
[[ "$SLURM_JOB_START_TIME" =~ ^[0-9]+$ && "$SLURM_JOB_END_TIME" =~ ^[0-9]+$ ]]
ep18_now=$(date +%s)
(( SLURM_JOB_START_TIME > 0 && SLURM_JOB_START_TIME <= ep18_now && ep18_now < SLURM_JOB_END_TIME ))
ep18_deadline=$((SLURM_JOB_START_TIME+14400-60))
if (( SLURM_JOB_END_TIME-60 < ep18_deadline )); then ep18_deadline=$((SLURM_JOB_END_TIME-60)); fi
if (( 1791926940 < ep18_deadline )); then ep18_deadline=1791926940; fi
(( ep18_deadline-ep18_now > 60 ))
ep18_operations_root="$ep18_episode/outputs/runs/false_partition_preparation/prepare_continuation_operations_20261013_001"
[[ -d "$ep18_operations_root" && -d "$L_SCRATCH_JOB" && -d "$ep18_cache" ]]
ep18_ops="$ep18_operations_root/segment_${SLURM_JOB_ID}"
mkdir -m 700 "$ep18_ops"
ep18_stage=continuation_admission
ep18_started=$SECONDS
ep18_prepare_exit=-1
ep18_projection_exit=-1
finish() {
  ep18_exit=$?
  trap - EXIT
  printf '{"stage":"%s","exit_code":%s,"elapsed_seconds":%s,"job_id":"%s","predecessor_job_id":"47584692","deadline_unix":%s,"allocated_cpus":4,"memory_mib":28672,"numerical_workers":1,"reserved_CPU_seconds":57600,"prepare_exit":%s,"post_projection_exit":%s,"RSS_guard_bytes":25769803776,"array_admission_bytes":8589934592,"search_launched":false,"tests_executed":false,"automatic_retry":false,"automatic_successor":false,"scientific_release_authorized":false}\n' \
    "$ep18_stage" "$ep18_exit" "$((SECONDS-ep18_started))" "$SLURM_JOB_ID" "$ep18_deadline" \
    "$ep18_prepare_exit" "$ep18_projection_exit" > "$ep18_ops/exit.json"
  exit "$ep18_exit"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
ulimit -c 0
# One durable claim prevents accidental duplicate continuation writers. It is
# retained after any failure; it is not a supervisor or authorization receipt.
mkdir -m 700 "$ep18_operations_root/sole_continuation_claim"
printf '%s\n' "$SLURM_JOB_ID" > "$ep18_operations_root/sole_continuation_claim/job_id.txt"

ep18_local="$L_SCRATCH_JOB/ep18-false-prepare-continue001-${SLURM_JOB_ID}"
mkdir -m 700 "$ep18_local"
mkdir -m 700 "$ep18_local/tmp" "$ep18_local/cache" "$ep18_local/hf"
export TMPDIR="$ep18_local/tmp" XDG_CACHE_HOME="$ep18_local/cache" HF_HOME="$ep18_local/hf"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
cd "$ep18_local"

ep18_stage=metadata_before_continuation
project_metadata 25 > "$ep18_ops/progress_before.json"
[[ ! -e "$ep18_cache/manifest.json" && ! -e "$ep18_cache/progress.json.pending" ]]
# Changed experiment-program validation is bounded and non-gating.
ep18_astra="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep18_astra_deps="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
ep18_astra_rc=0
env PYTHONPATH="$ep18_astra/lib/python3.12/site-packages:$ep18_astra_deps" \
  timeout --signal=TERM --kill-after=5s 30s "$ep18_astra/bin/astra-tools" \
  validate "$ep18_episode/outputs/astra/v0.0.14/astra.yaml" --json \
  > "$ep18_ops/astra.json" 2> "$ep18_ops/astra_stderr.txt" || ep18_astra_rc=$?
printf '%s\n' "$ep18_astra_rc" > "$ep18_ops/astra_exit.txt"
ep18_now=$(date +%s)
(( ep18_deadline-ep18_now > 20 ))
ep18_timeout=$((ep18_deadline+5-ep18_now))
ep18_stage=resume_original_cache_preparation
set +e
timeout --signal=TERM --kill-after=5s "${ep18_timeout}s" "$ep18_python" -B \
  "$ep18_code/false_partition_search.py" prepare --config "$ep18_config" \
  --output "$ep18_ops/prepare" --deadline "$ep18_deadline" \
  --rss-byte-cap 25769803776 --array-byte-cap 8589934592 \
  > "$ep18_ops/cache_preparation.txt" 2> "$ep18_ops/cache_preparation_stderr.txt"
ep18_prepare_exit=$?
# At most30seconds after the preparation timeout (25seconds plus5-second
# kill grace); the60-second reserved tail also covers the final exit record.
project_metadata 25 > "$ep18_ops/progress_after.json" 2> "$ep18_ops/projection_stderr.txt"
ep18_projection_exit=$?
set -e
case "$ep18_prepare_exit" in
  0) ep18_stage=cache_preparation_complete_search_not_started ;;
  75) ep18_stage=cache_preparation_resource_incomplete_search_not_started ;;
  *) ep18_stage=cache_preparation_failure_requires_diagnosis ;;
esac
if (( ep18_projection_exit != 0 && ep18_prepare_exit == 0 )); then
  ep18_stage=cache_complete_post_projection_failed
  exit "$ep18_projection_exit"
fi
exit "$ep18_prepare_exit"
