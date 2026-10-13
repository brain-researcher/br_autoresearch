#!/bin/bash
# Focused deadline arithmetic only; run in coordinator-owned compute.
# Extract only the marked production block. Never execute modules, srun,
# scientific programs, scheduler queries, or the rest of the wrapper.
set -euo pipefail
EP_TEST_HOST=$(hostname)
if [[ -z "${SLURM_JOB_ID-}" || "$EP_TEST_HOST" == *-ln* ]]; then
  printf 'Run this focused check inside the coordinated compute allocation\n' >&2
  exit 2
fi
EP_TEST_DIRECTORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EP_TEST_BLOCK=$(sed -n '/^# BEGIN OPTIONAL_PARENT_DEADLINE$/,/^# END OPTIONAL_PARENT_DEADLINE$/p' \
  "$EP_TEST_DIRECTORY/mz_twice_rival_wave.sbatch")
if [[ "$EP_TEST_BLOCK" != '# BEGIN OPTIONAL_PARENT_DEADLINE'* || \
      "$EP_TEST_BLOCK" != *'# END OPTIONAL_PARENT_DEADLINE' ]]; then
  printf 'The production deadline block was not found\n' >&2
  exit 1
fi

check_deadline() (
  EP_TEST_CASE=$1
  EP_TWICE_ENTERED=$2
  EP_TWICE_PARENT=$3
  EP_TEST_START=$4
  EP_TEST_END=$5
  EP_TEST_EXPECTED=$6
  EP_TEST_EVIDENCE=$7
  EP_TEST_OBSERVED=$8
  unset SLURM_JOB_START_TIME SLURM_JOB_END_TIME
  if [[ "$EP_TEST_START" != absent ]]; then SLURM_JOB_START_TIME=$EP_TEST_START; fi
  if [[ "$EP_TEST_END" != absent ]]; then SLURM_JOB_END_TIME=$EP_TEST_END; fi
  eval "$EP_TEST_BLOCK"
  if [[ "$EP_TWICE_DEADLINE" != "$EP_TEST_EXPECTED" || \
        "$EP_TWICE_TIME_EVIDENCE" != "$EP_TEST_EVIDENCE" || \
        "$EP_TWICE_OBSERVED_PARENT" != "$EP_TEST_OBSERVED" || \
        "$EP_TWICE_OBSERVED_START" != "${SLURM_JOB_START_TIME-unavailable}" || \
        "$EP_TWICE_OBSERVED_END" != "${SLURM_JOB_END_TIME-unavailable}" ]] || \
        (( EP_TWICE_DECLARED_DEADLINE != EP_TWICE_ENTERED + EP_TWICE_PARENT || \
           EP_TWICE_DEADLINE > EP_TWICE_DECLARED_DEADLINE )); then
    printf 'FAIL %s: deadline=%s evidence=%s observed_parent=%s\n' \
      "$EP_TEST_CASE" "$EP_TWICE_DEADLINE" "$EP_TWICE_TIME_EVIDENCE" "$EP_TWICE_OBSERVED_PARENT" >&2
    exit 1
  fi
  printf 'PASS %s\n' "$EP_TEST_CASE"
)

# Both actually observed +4-second differences must be accepted without
# extending the entered+declared deadline or changing either component cap.
check_deadline mvz_plus_four 1000 14880 1000 15884 15880 \
  declared_parent_with_slurm_end_observed 14884
check_deadline mvc_plus_four 1000 7500 1000 8504 8500 \
  declared_parent_with_slurm_end_observed 7504
check_deadline absent_metadata 1000 7500 absent absent 8500 \
  coordinator_verified_declared_parent unavailable
check_deadline earlier_end 1000 7500 900 8400 8400 \
  declared_parent_shortened_by_slurm_end 7500
check_deadline earlier_end_without_start 1000 7500 absent 8400 8400 \
  declared_parent_shortened_by_slurm_end unavailable
check_deadline malformed_metadata 1000 7500 invalid invalid 8500 \
  coordinator_verified_declared_parent unavailable
