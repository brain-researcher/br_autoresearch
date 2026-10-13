#!/bin/bash
# One bounded collection/screen pass in a root-owned compute allocation.
# Uses the existing EP04 runtime; no submission, retry, lock, or OOD successor.
set -euo pipefail
set -C
[[ $# -eq 0 ]] || exit 2

ep04_collection_host=$(hostname)
ep04_collection_root="${OAK:?}/users/zijiao/br_autoresearch/episode04_shared_scene_geometry"
printf 'HOST=%s JOB=%s CPUS=%s\n' "$ep04_collection_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}"
[[ "$ep04_collection_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 \
   && "${SLURM_JOB_CPUS_PER_NODE:?}" == 1 && "${SLURM_NNODES:?}" == 1 ]]
[[ "$(pwd -P)" == "$ep04_collection_root" ]]
for ep04_collection_thread in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS \
  NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS BLIS_NUM_THREADS; do
  [[ "${!ep04_collection_thread:-}" == 1 ]]
done
command -v python3.12 >/dev/null
command -v timeout >/dev/null
export PYTHONDONTWRITEBYTECODE=1

ep04_collection_lists=outputs/code/remaining_controls_selected/20261005-full01
ep04_collection_observed=outputs/runs/observed_native_panel/20261005-observed01/observed_native_panel.json
ep04_collection_null=outputs/runs/registered_null_collection/20261005-full01
ep04_collection_encoding=outputs/runs/identical_fold_encoding_collection/20261005-full01
ep04_collection_decision=outputs/runs/panel_decision/registered_panel_decision.json
[[ -f "$ep04_collection_lists/null_reports.json" \
   && -f "$ep04_collection_lists/encoding_reports.json" \
   && -f "$ep04_collection_observed" \
   && -f outputs/runs/native_sensitivities/20261005-all01/native_sensitivities.json ]]
for ep04_collection_destination in "$ep04_collection_null" "$ep04_collection_encoding" \
  "$ep04_collection_decision"; do
  [[ ! -e "$ep04_collection_destination" && ! -L "$ep04_collection_destination" ]]
done

# A fresh operational ledger records pending/started/exited stages even when a
# collector returns a scientific not-evaluable result. Never truncate an old pass.
exec {ep04_collection_status_fd}> "$ep04_collection_lists/collection_action_status.tsv"
printf 'stage\tstate\texit_status\nnull_collection\tpending\t\nencoding_collection\tpending\t\npanel_screen\tpending\t\n' \
  >&"$ep04_collection_status_fd"

ep04_collection_stage() {
  local ep04_collection_name=$1 ep04_collection_exit
  shift
  printf '%s\tstarted\t\n' "$ep04_collection_name" >&"$ep04_collection_status_fd"
  printf '{"stage":"%s","state":"started"}\n' "$ep04_collection_name"
  if timeout --signal=TERM --kill-after=10s 610s python3.12 -u -B "$@"; then
    ep04_collection_exit=0
  else
    ep04_collection_exit=$?
  fi
  printf '%s\texited\t%s\n' "$ep04_collection_name" "$ep04_collection_exit" >&"$ep04_collection_status_fd"
  printf '{"stage":"%s","state":"exited","exit_status":%s}\n' \
    "$ep04_collection_name" "$ep04_collection_exit"
  # In particular, encoding exit 2 is retained as-is, including not_evaluable.
  # No later stage runs on nonzero exit; existing reports remain untouched.
  return "$ep04_collection_exit"
}

ep04_collection_stage null_collection outputs/code/collect_registered_nulls.py \
  --report-list "$ep04_collection_lists/null_reports.json" \
  --observed-report "$ep04_collection_observed" \
  --output-dir "$ep04_collection_null" --max-seconds 600

ep04_collection_stage encoding_collection outputs/code/collect_identical_fold_encoding.py \
  --report-list "$ep04_collection_lists/encoding_reports.json" \
  --observed-report "$ep04_collection_observed" \
  --output-dir "$ep04_collection_encoding" --max-seconds 600

ep04_collection_stage panel_screen outputs/code/screen_registered_panel.py \
  --null-report "$ep04_collection_null/registered_null_collection.json" \
  --encoding-report "$ep04_collection_encoding/identical_fold_encoding_collection.json" \
  --max-seconds 600

exec {ep04_collection_status_fd}>&-
