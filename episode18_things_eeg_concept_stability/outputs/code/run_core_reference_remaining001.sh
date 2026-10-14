#!/bin/bash
# UNUSED: no resource selection or submission is made by this source.
# Submit only after complete first-person cost evidence and a recorded
# coordinator selection. Every sbatch invocation must explicitly supply
# --time, --cpus-per-task=2, --mem=15G, --nodes=1, --ntasks=1,
# --no-requeue and the coordinator-verified phase dependency/array options.
# Arguments: participant|last ALLOCATION_SECONDS ARRAY_CONCURRENCY
#            OTHER_RESERVED_EP18_CPUS REMAINING_EP18_CPU_SECONDS
#            none|PREDECESSOR_ARRAY_JOB_ID
# One28-person array is followed by the substantive sub49 fit plus finalization.
# No tests, reinitialization, scheduler calls, retries or successors are created.
set -euo pipefail
umask 077
[[ $# == 6 ]]
ep18_phase=$1
ep18_allocation_seconds=$2
ep18_concurrency=$3
ep18_other_cpus=$4
ep18_remaining_ledger_seconds=$5
ep18_predecessor=$6
[[ "$ep18_phase" == participant || "$ep18_phase" == last ]]
[[ "$ep18_allocation_seconds" =~ ^[1-9][0-9]{2,6}$ ]]
[[ "$ep18_concurrency" =~ ^[1-9][0-9]?$ ]]
[[ "$ep18_other_cpus" =~ ^(0|[1-9][0-9]{0,2})$ ]]
[[ "$ep18_remaining_ledger_seconds" =~ ^[1-9][0-9]{0,7}$ ]]
(( ep18_allocation_seconds >= 600 && ep18_allocation_seconds <= 1209600 ))
(( ep18_concurrency <= 28 && ep18_other_cpus <= 128 ))
(( 2*ep18_concurrency+ep18_other_cpus <= 128 ))
(( ep18_remaining_ledger_seconds <= 43200000 ))
ep18_reserved_cpu_seconds=$((29*2*ep18_allocation_seconds))
(( ep18_reserved_cpu_seconds <= ep18_remaining_ledger_seconds ))

[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_JOB_ID}" =~ ^[1-9][0-9]*$ ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 2 && "${SLURM_MEM_PER_NODE:-}" == 15360 ]]
[[ "${SLURM_NTASKS:-}" == 1 && "${SLURM_JOB_NUM_NODES:-}" == 1 && "${SLURM_JOB_CPUS_PER_NODE:-}" == 2 ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" && "${SLURM_GPUS:-0}" == 0 ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
: "${OAK:?}" "${GROUP_HOME:?}" "${L_SCRATCH_JOB:?}" "${SLURM_JOB_START_TIME:?}" "${SLURM_JOB_END_TIME:?}"
[[ "$SLURM_JOB_START_TIME" =~ ^[0-9]+$ && "$SLURM_JOB_END_TIME" =~ ^[0-9]+$ ]]
ep18_now=$(date +%s)
(( SLURM_JOB_START_TIME > 0 && SLURM_JOB_START_TIME <= ep18_now && ep18_now < SLURM_JOB_END_TIME ))
ep18_deadline=$((SLURM_JOB_START_TIME+ep18_allocation_seconds-60))
if (( SLURM_JOB_END_TIME-60 < ep18_deadline )); then ep18_deadline=$((SLURM_JOB_END_TIME-60)); fi
if (( 1791926990 < ep18_deadline )); then ep18_deadline=1791926990; fi
(( ep18_now < ep18_deadline ))

ep18_people=(sub-08 sub-09 sub-10 sub-11 sub-12 sub-13 sub-14 sub-20
  sub-21 sub-22 sub-24 sub-25 sub-26 sub-27 sub-28 sub-29 sub-30 sub-31
  sub-32 sub-34 sub-35 sub-38 sub-43 sub-44 sub-45 sub-46 sub-47 sub-48)
if [[ "$ep18_phase" == participant ]]; then
  [[ "$ep18_predecessor" == none && "${SLURM_ARRAY_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]]
  [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^(0|[1-9][0-9]?)$ ]]
  [[ "${SLURM_ARRAY_TASK_COUNT:-}" == 28 && "${SLURM_ARRAY_TASK_MIN:-}" == 0
     && "${SLURM_ARRAY_TASK_MAX:-}" == 27 && "${SLURM_ARRAY_TASK_STEP:-}" == 1 ]]
  (( SLURM_ARRAY_TASK_ID <= 27 ))
  ep18_person=${ep18_people[$SLURM_ARRAY_TASK_ID]}
else
  [[ "$ep18_predecessor" =~ ^[1-9][0-9]*$ && "$ep18_predecessor" != "$SLURM_JOB_ID" ]]
  [[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]]
  ep18_person=sub-49
fi

ep18_episode="$OAK/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_code="$ep18_episode/outputs/code"
ep18_run="$ep18_episode/outputs/cache/private/core_reference_20261013_001"
ep18_cost_ops="$ep18_episode/outputs/runs/baseline_models/core_reference_operations_20261013_001/initial_47591674"
ep18_operations_root="$ep18_episode/outputs/runs/baseline_models/core_reference_remaining_operations_20261013_001"
[[ -d "$ep18_operations_root" && -d "$L_SCRATCH_JOB" ]]
ep18_ops="$ep18_operations_root/${ep18_phase}_${ep18_person}_${SLURM_JOB_ID}"
mkdir -m 700 "$ep18_ops"
ep18_stage=allocation_admission
ep18_started=$SECONDS
finish() {
  ep18_exit=$?
  trap - EXIT
  printf '{"phase":"%s","participant_id":"%s","stage":"%s","exit_code":%s,"elapsed_seconds":%s,"job_id":"%s","deadline_epoch":%s,"allocated_cpus":2,"memory_mib":15360,"numerical_threads":1,"this_parent_reserved_CPU_seconds":%s,"whole_remaining_cohort_reserved_CPU_seconds":%s,"coordinator_selected_array_concurrency":%s,"coordinator_reserved_other_EP18_CPUs":%s,"coordinator_remaining_ledger_CPU_seconds":%s,"predecessor_array_job_id":"%s","scheduler_bindings_verified_by":"coordinator","RSS_guard":"observed_high_water_at_budget_checks","automatic_retry":false,"current_tests_executed":false,"scientific_release_authorized":false}\n' \
    "$ep18_phase" "$ep18_person" "$ep18_stage" "$ep18_exit" "$((SECONDS-ep18_started))" \
    "$SLURM_JOB_ID" "$ep18_deadline" "$((2*ep18_allocation_seconds))" "$ep18_reserved_cpu_seconds" \
    "$ep18_concurrency" "$ep18_other_cpus" "$ep18_remaining_ledger_seconds" "$ep18_predecessor" > "$ep18_ops/exit.json"
  exit "$ep18_exit"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
ulimit -c 0
[[ -f "$ep18_run/inputs_complete.json" && -f "$ep18_run/settings.json" ]]
[[ ! -e "$ep18_run/$ep18_person" ]]
[[ ! -e "$ep18_run/core_reference.complete.json" && ! -e "$ep18_run/core_reference.complete.json.partial" ]]

ep18_local="$L_SCRATCH_JOB/ep18-core-remaining001-${ep18_phase}-${ep18_person}"
mkdir -m 700 "$ep18_local"
mkdir -m 700 "$ep18_local/tmp" "$ep18_local/cache"
export TMPDIR="$ep18_local/tmp" XDG_CACHE_HOME="$ep18_local/cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1
unset PYTHONPATH PYTHONHOME
ep18_python="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/bin/python"
[[ -x "$ep18_python" ]]
cd "$ep18_local"

remaining_seconds() {
  local ep18_left
  ep18_left=$((ep18_deadline-$(date +%s)))
  (( ep18_left > 0 )) || return 2
  printf '%s\n' "$ep18_left"
}

# Reuse actual public cost/check evidence; no private scores or new tests here.
ep18_stage=completed_first_person_cost_binding
ep18_bound_seconds=$(remaining_seconds)
if (( ep18_bound_seconds > 15 )); then ep18_bound_seconds=15; fi
timeout --signal=TERM --kill-after=5s "${ep18_bound_seconds}s" "$ep18_python" -B - "$ep18_cost_ops" <<'PY' > "$ep18_ops/cost_evidence.json"
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
def read(name):
    with (root / name).open('rb') as source:
        value = source.read(65537)
    if len(value) > 65536:
        raise ValueError('bounded_existing_cost_record_required')
    return value.decode('utf-8')

exit_record = json.loads(read('exit.json'))
person = json.loads(read('participant_stdout.txt'))
checks = read('core_reference_checks.txt')
if not (exit_record.get('job_id') == '47591674' and exit_record.get('exit_code') == 0
        and exit_record.get('stage') == 'complete_sub07_cost_attempt'
        and exit_record.get('cost_only') is True
        and person.get('status') == 'complete_core_reference_person'
        and person.get('participant_id') == 'sub-07' and person.get('cost_only') is True
        and person.get('scientific_release_authorized') is False
        and person.get('progress', {}).get('inner_splits_completed') == 30
        and person.get('progress', {}).get('outer_splits_completed') == 6
        and 'error_class' not in person
        and re.fullmatch(r'[0-9a-f]{64}', person.get('settings_id', ''))):
    raise ValueError('complete_original_sub07_cost_required')
methods = [line for line in checks.splitlines() if line.startswith('test_')]
if not (len(methods) == len(set(methods)) == 2
        and all(line.endswith(' ... ok') for line in methods)
        and len(re.findall(r'^Ran 2 tests in [0-9.]+s$', checks, re.MULTILINE)) == 1
        and checks.rstrip().splitlines()[-1] == 'OK'):
    raise ValueError('reuse_exact_two_passed_core_checks_required')
print(json.dumps({'status': 'complete_first_person_cost_evidence_bound',
    'source_job_id': '47591674', 'prior_methods': 2, 'current_tests_executed': False,
    'settings_id': person['settings_id']}, sort_keys=True))
PY

if [[ "$ep18_phase" == last ]]; then
  # Only the coordinator-selected afterok parent may own the final writer.
  # This permanent claim also prevents accidental replacement of a failed last
  # attempt. It does not submit a job or infer predecessor success from its ID.
  mkdir -m 700 "$ep18_operations_root/finalizer_claim"
  printf 'job_id=%s\npredecessor_array_job_id=%s\n' "$SLURM_JOB_ID" "$ep18_predecessor" \
    > "$ep18_operations_root/finalizer_claim/owner.txt"
fi

ep18_stage=complete_participant_reference
ep18_person_deadline=$ep18_deadline
if [[ "$ep18_phase" == last ]]; then ep18_person_deadline=$((ep18_deadline-60)); fi
ep18_now=$(date +%s)
(( ep18_now < ep18_person_deadline ))
ep18_timeout=$((ep18_person_deadline+5-ep18_now))
timeout --signal=TERM --kill-after=5s "${ep18_timeout}s" "$ep18_python" -B \
  "$ep18_code/production_core_reference.py" participant \
  --run-dir "$ep18_run" --participant-id "$ep18_person" \
  --deadline-epoch "$ep18_person_deadline" --rss-limit-bytes 12884901888 \
  --durable-byte-limit 2147483648 --record-byte-limit 134217728 --threads 1 \
  > "$ep18_ops/participant_stdout.txt" 2> "$ep18_ops/participant_stderr.txt"

if [[ "$ep18_phase" == last ]]; then
  ep18_stage=all30_reference_finalization
  ep18_remaining=$(remaining_seconds)
  ep18_timeout=$((ep18_remaining+5))
  # All30 complete+operational records are required by the original finalizer.
  # This1MiB final record allowance is additional to the unchanged per-person
  # 2GiB, within the same substantive parent and its original deadline.
  timeout --signal=TERM --kill-after=5s "${ep18_timeout}s" "$ep18_python" -B \
    "$ep18_code/production_core_reference.py" finalize --run-dir "$ep18_run" \
    --deadline-epoch "$ep18_deadline" --rss-limit-bytes 12884901888 \
    --durable-byte-limit 1048576 --record-byte-limit 134217728 --threads 1 \
    > "$ep18_ops/finalize_stdout.txt" 2> "$ep18_ops/finalize_stderr.txt"
fi
ep18_stage=completed
