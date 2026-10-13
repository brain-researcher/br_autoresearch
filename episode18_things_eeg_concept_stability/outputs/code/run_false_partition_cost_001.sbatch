#!/bin/bash
# Submit only after coordinator review, from the EP18 directory, with the
# existing image assembler dependency. Request 13 CPUs/96GiB for 52 CPU-hours:
# observed site memory normalization increased the earlier 12-CPU request to
# 13 CPUs. Allow allocation rounding through 16 CPUs and record actual CPUs
# times four hours (at most64 CPU-hours). Exactly one worker/thread runs.
# The coordinator precreates only the operations root and its parent below.
# Input/cache directories must be absent. No retry, requeue or successor.
#SBATCH --job-name=ep18-false-cost
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --dependency=afterok:47551339
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/false_partition_preparation/false_partition_operations_20261012_001/%x-%j.out
#SBATCH --error=outputs/runs/false_partition_preparation/false_partition_operations_20261012_001/%x-%j.err
set -euo pipefail
umask 077
[[ $# == 0 ]]
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
ep18_allocated_cpus=${SLURM_CPUS_PER_TASK:?Actual task CPU allocation is required}
[[ "$ep18_allocated_cpus" =~ ^[0-9]+$ ]]
(( ep18_allocated_cpus >= 13 && ep18_allocated_cpus <= 16 ))
[[ "${SLURM_MEM_PER_NODE:-}" == 98304 && "${SLURM_JOB_NUM_NODES:-}" == 1 ]]
[[ "${SLURM_JOB_CPUS_PER_NODE:-}" == "$ep18_allocated_cpus" ]]
ep18_reserved_cpu_hours=$((4*ep18_allocated_cpus))
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]]
: "${OAK:?}" "${GROUP_HOME:?}" "${L_SCRATCH_JOB:?}"
ep18_episode="$OAK/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_code="$ep18_episode/outputs/code"
ep18_release="$ep18_episode/outputs/config/false_partition_release_20261012_001.json"
ep18_config="$ep18_episode/outputs/config/false_partition_sub07_20261012_001.json"
ep18_inputs="$ep18_episode/outputs/runs/false_partition_preparation/inputs_20261012_001"
ep18_cache="$ep18_episode/outputs/cache/false_partition_sub07_20261012_001"
ep18_operations_root="$ep18_episode/outputs/runs/false_partition_preparation/false_partition_operations_20261012_001"
[[ -d "$ep18_operations_root" ]]
ep18_operations="$ep18_operations_root/initial_${SLURM_JOB_ID}"
mkdir -m 700 "$ep18_operations"
ep18_stage=allocation_admission
ep18_started=$SECONDS
ep18_deadline=0
ep18_prepare_exit=-1
ep18_search_exit=-1
finish() {
  ep18_exit=$?
  trap - EXIT
  printf '{"stage":"%s","exit_code":%s,"elapsed_seconds":%s,"job_id":"%s","deadline_unix":%s,"allocated_cpus":%s,"numerical_workers":1,"reserved_cpu_hours":%s,"prepare_exit":%s,"search_exit":%s,"automatic_retry":false,"automatic_successor":false}\n' \
    "$ep18_stage" "$ep18_exit" "$((SECONDS-ep18_started))" "$SLURM_JOB_ID" "$ep18_deadline" \
    "$ep18_allocated_cpus" "$ep18_reserved_cpu_hours" \
    "$ep18_prepare_exit" "$ep18_search_exit" > "$ep18_operations/exit.json"
  exit "$ep18_exit"
}
trap finish EXIT

# Fixed actual allocation clock. The coordinator checks the requested Slurm
# TimeLimit; do not equate End-Start to exactly four hours or poll Slurm here.
ep18_start=${SLURM_JOB_START_TIME:?Scheduler start epoch is required}
ep18_end=${SLURM_JOB_END_TIME:?Scheduler end epoch is required}
[[ "$ep18_start" =~ ^[0-9]+$ && "$ep18_end" =~ ^[0-9]+$ ]]
ep18_now=$(date +%s)
(( ep18_start > 0 && ep18_start <= ep18_now && ep18_end > ep18_start ))
ep18_deadline=$((ep18_start+14400-60))
if (( ep18_end-60 < ep18_deadline )); then ep18_deadline=$((ep18_end-60)); fi
if (( 1791927000 < ep18_deadline )); then ep18_deadline=1791927000; fi

# Each finite command leaves five seconds for forced termination inside the
# common deadline, and ten seconds between its cooperative stop and timeout.
# Search saves its unchanged logical state at that cooperative deadline.
phase_window() {
  local ep18_cap=$1 ep18_current ep18_outer_end
  ep18_current=$(date +%s)
  ep18_outer_end=$((ep18_current+ep18_cap))
  if (( ep18_deadline-5 < ep18_outer_end )); then ep18_outer_end=$((ep18_deadline-5)); fi
  ep18_phase_seconds=$((ep18_outer_end-ep18_current))
  (( ep18_phase_seconds > 20 )) || return 75
  ep18_phase_deadline=$((ep18_outer_end-10))
}
phase_window 60
printf 'hostname=%s job_id=%s start=%s end=%s deadline=%s allocated_cpus=%s reserved_cpu_hours=%s workers=1 memory_mib=98304\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$ep18_start" "$ep18_end" "$ep18_deadline" \
  "$ep18_allocated_cpus" "$ep18_reserved_cpu_hours"

ep18_stage=fresh_initial_attempt
[[ ! -e "$ep18_inputs" && ! -e "$ep18_cache" ]]
ep18_stage=selected_runtime
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1
unset PYTHONPATH PYTHONHOME
ep18_python="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/bin/python"
[[ -x "$ep18_python" ]]
ep18_local="$L_SCRATCH_JOB/ep18-false-cost-${SLURM_JOB_ID}"
mkdir -m 700 "$ep18_local"
mkdir -m 700 "$ep18_local/tmp" "$ep18_local/cache" "$ep18_local/hf"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$ep18_local/tmp" XDG_CACHE_HOME="$ep18_local/cache" HF_HOME="$ep18_local/hf"
cd "$ep18_episode"

ep18_stage=completed_assembly_and_release_admission
phase_window 60
timeout --signal=TERM --kill-after=5s "${ep18_phase_seconds}s" "$ep18_python" -B - \
  "$ep18_release" "$ep18_config" "$ep18_code" "$ep18_inputs" "$ep18_cache" \
  > "$ep18_operations/input_admission.json" <<'PY'
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[3])
from false_partition_search import load_config, search_array_bytes, context_schema
from prepare_false_partition_inputs import feature_binding, load_release, require
release, people, event_paths, join_path = load_release(sys.argv[1])
config = load_config(sys.argv[2])
ids, families = feature_binding(release['feature_bank'])
require(config['inputs_dir'] == sys.argv[4] and config['cache_dir'] == sys.argv[5]
        and config['participant_id'] == people[0] == 'sub-07'
        and config['not_after_unix'] == 1791927000
        and config['pair_block_rows'] == 128
        and config['cache_byte_cap'] == 77309411328
        and config['segment_byte_cap'] == 67108864
        and config['recorded_span_recipe'] == {
            'cue_anchor': 'preceding_recorded_E3', 'post_last_inclusive_samples': 800,
            'score_first_offset_samples': -800, 'score_last_inclusive_samples': 800}
        and config['filter_left_support_samples'] == 128
        and config['filter_right_support_samples'] == 0)
horizon, edges, names, rate = context_schema(config)
array_bytes, descriptors = search_array_bytes(len(names))
require(horizon == 928 and len(names) == 855 and array_bytes <= 85899345920)
print(json.dumps({'assembly_complete': True, 'feature_images': len(ids),
    'families': len(families), 'development_participants': len(people),
    'first_production_participant': 'sub-07', 'context_channels': len(names),
    'search_array_admission_bytes': array_bytes,
    'EEG_read': False, 'scientific_subset_selected': False}))
PY

ep18_stage=astra_validation
phase_window 60
ep18_astra_env="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep18_astra="$ep18_astra_env/bin/astra-tools"
env PYTHONPATH="$ep18_astra_env/lib/python3.12/site-packages:$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages" \
  timeout --signal=TERM --kill-after=5s "${ep18_phase_seconds}s" "$ep18_astra" validate \
  outputs/astra/v0.0.14/astra.yaml --json > "$ep18_operations/astra_validation.json"

# Complete all30 development input records; only the existing binary target
# and response flags are selected, never RT/accuracy. The search's explicit
# coarse-button context locates the flag at its source stimulus onset.
ep18_stage=complete_development_input_binding
phase_window 300
timeout --signal=TERM --kill-after=5s "${ep18_phase_seconds}s" "$ep18_python" -B \
  "$ep18_code/prepare_false_partition_inputs.py" --release-manifest "$ep18_release" \
  --output-dir "$ep18_inputs" --deadline-unix "$ep18_phase_deadline" \
  --rss-bytes 2147483648 --durable-bytes 134217728 \
  > "$ep18_operations/input_preparation.txt" 2> "$ep18_operations/input_preparation_stderr.txt"

# This cache is ordinary production work for the first recorded participant,
# not a reduced image panel or a primary analysis subset. Cache progress and
# any partial files remain for explicitly reviewed continuation after a stop.
ep18_stage=complete_sub07_cache_preparation
phase_window 10800
set +e
timeout --signal=TERM --kill-after=5s "${ep18_phase_seconds}s" "$ep18_python" -B \
  "$ep18_code/false_partition_search.py" prepare --config "$ep18_config" \
  --output "$ep18_operations/prepare" --deadline "$ep18_phase_deadline" \
  --rss-byte-cap 96636764160 --array-byte-cap 8589934592 \
  > "$ep18_operations/cache_preparation.txt" 2> "$ep18_operations/cache_preparation_stderr.txt"
ep18_prepare_exit=$?
set -e
if (( ep18_prepare_exit != 0 )); then
  ep18_stage=cache_incomplete_or_failed_search_not_started
  exit "$ep18_prepare_exit"
fi

# The search itself requires the completed cache manifest. Its initial bank
# uses unchanged seed20261012,8members,64starts/member,100000swaps/start.
# Every continuation is separately selected and uses its saved checkpoint;
# this wrapper neither resets counters nor submits a successor.
ep18_stage=bounded_full_budget_search_segment
phase_window 3600
set +e
timeout --signal=TERM --kill-after=5s "${ep18_phase_seconds}s" "$ep18_python" -B \
  "$ep18_code/false_partition_search.py" search --config "$ep18_config" \
  --output "$ep18_operations/search" --deadline "$ep18_phase_deadline" \
  --rss-byte-cap 96636764160 --array-byte-cap 85899345920 \
  > "$ep18_operations/search.txt" 2> "$ep18_operations/search_stderr.txt"
ep18_search_exit=$?
set -e
case "$ep18_search_exit" in
  0) ep18_stage=image_context_candidate_bank_complete ;;
  75) ep18_stage=search_resource_incomplete_checkpoint_if_written ;;
  2) ep18_stage=frozen_search_budget_exhausted ;;
  *) ep18_stage=search_failed_requires_diagnosis ;;
esac
exit "$ep18_search_exit"
