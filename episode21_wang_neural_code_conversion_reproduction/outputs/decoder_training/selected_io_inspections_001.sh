#!/bin/bash
# Selected child of the substantive work5 remainder parent only. No submissions.
# Reuse the 16 passing inspection-helper tests from job46949521. Three serial
# scans use the parent's existing 1CPU/7Gi/8h reservation and preserve quarantine.
set -euo pipefail
[[ $# -eq 0 && "${SLURM_ARRAY_TASK_ID:-}" == 4 ]]
[[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 1 && "${SLURM_MEM_PER_NODE:-}" == 7168 ]]
[[ "$(hostname)" != *-ln* ]]
[[ "$PWD" == "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction" ]]
python3.12 -u -B - <<'EP21_SELECTED_INSPECTIONS'
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, "outputs/decoder_training/tools")
import decoder_execution_contract as contract

WORK = ("work_0061", "work_0062", "work_0047")
QUARANTINED = ["work_0032", "work_0033", "work_0047", "work_0061", "work_0062"]
BASE = contract.BASE
require = contract.require
require(Path.cwd().resolve() == BASE, "exact_EP21_working_directory_required")
manifest = contract.prefix.read_json(contract.MANIFEST)
items = [item for item in manifest["work_items"] if item["work_item_id"] == "work_0005"]
require(len(items) == 1, "one_immutable_work5_assignment_required")
allocation = contract.verify_allocation(items[0]["resources"])
verified_at = time.monotonic()
require(allocation["time_limit_seconds"] == 8 * 3600, "selected_work5_eight_hour_parent_required")
end_epoch = allocation["job_end_epoch"]
directory = BASE / "outputs/decoder_training" / ("selected_io_inspection_job_" + allocation["job_id"])
directory.mkdir(mode=0o700)  # Exclusive evidence path; never overwrite a prior child.
status_path = directory / "status.jsonl"


def record(value):
    with status_path.open("a") as stream:
        stream.write(json.dumps(value) + "\n")
        stream.flush()
    print(json.dumps(value), flush=True)


def validate_report(path, work, code):
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 16 * 1024**2,
            "bounded_fresh_inspection_report_required")
    report = json.loads(path.read_text())
    require(report.get("schema_version") == "ep21.closed_io_fragment_inspection.v1"
            and report.get("requested_work_items") == [work]
            and report.get("unrequested_work_items") == [item for item in QUARANTINED if item != work]
            and report.get("quarantined_work_items") == QUARANTINED
            and report.get("allocation", {}).get("job_id") == allocation["job_id"]
            and report["allocation"].get("hostname") == allocation["hostname"]
            and report["allocation"].get("caller_verified_end_epoch") == end_epoch,
            "matching_selected_inspection_identity_required")
    require(all(report.get(key) is False for key in
                ("original_artifacts_modified", "repair_coverage_written", "retry_authorized", "reuse_authorized"))
            and report.get("all_five_fragments_completely_checked") is False,
            "inspection_must_preserve_all_quarantine_boundaries")
    fragments = report.get("fragments")
    require(isinstance(fragments, list) and len(fragments) <= 1
            and all(row.get("work_item_id") == work for row in fragments)
            and "earlier_error" not in report
            and all("earlier_error" not in row and row.get("status") in
                    {"consistent", "corrupt", "timed_out"} for row in fragments),
            "storage_resource_or_unknown_failure_cannot_continue_fit")
    attempted = [row["work_item_id"] for row in fragments]
    checked = [row["work_item_id"] for row in fragments if row.get("inspection_complete") is True]
    require(report.get("attempted_work_items") == attempted
            and report.get("completely_checked_work_items") == checked
            and report.get("not_inspected_requested_work_items") == ([] if attempted else [work])
            and report.get("all_requested_fragments_completely_checked") is (checked == [work]),
            "explicit_inspection_completeness_required")
    if code == 0:
        require(attempted == [work]
                and report.get("status") in {"consistent_observations_only", "inconsistent_observations"}
                and all(row["status"] in {"consistent", "corrupt"} for row in fragments),
                "completed_diagnostic_report_required_for_exit_zero")
    else:
        require(code == 3 and report.get("status") == "stopped_timed_out",
                "only_clean_internal_timeout_may_continue_fit")
    return {"work_item_id": work, "exit_code": code, "report_status": report["status"],
            "completely_checked_work_items": checked, "reuse_authorized": False,
            "retry_authorized": False}


record({"event": "selected_inspection_child_started", "allocation": allocation,
        "selected_work_items": list(WORK), "parent_charged_once_to": "EP21",
        "inspection_checks_reused_from_job": "46949521", "separate_CPU_reservation": False})
try:
    for work in WORK:
        # Skip optional inspection if its full bound plus closure reserve no
        # longer fits. The substantive worker retains its own actual-end guard.
        if end_epoch - time.time() < 900:
            record({"event": "selected_inspections_deferred_for_fit", "remaining_work_items":
                    list(WORK[WORK.index(work):]), "reason": "less_than_600_seconds_plus_300_second_reserve"})
            break
        report_path = directory / (work + ".json")
        error_path = directory / (work + ".err")
        command = ["timeout", "--signal=TERM", "--kill-after=5s", "605s", "python3.12", "-u", "-B",
                   "outputs/decoder_training/tools/inspect_closed_io_fragments.py", "--authorization",
                   "outputs/decoder_training_full_authorization.json", "--allocation-end-epoch",
                   str(end_epoch), "--max-seconds", "600", "--work-items", work]
        with report_path.open("x") as output, error_path.open("x") as errors:
            result = subprocess.run(command, stdout=output, stderr=errors, timeout=620, check=False)
        record({"event": "selected_inspection_exited", "work_item_id": work, "exit_code": result.returncode})
        require(result.returncode in (0, 3), "inspection_failure_or_external_timeout_stops_affected_parent")
        record({"event": "selected_inspection_report_checked", **validate_report(report_path, work, result.returncode)})
    # The subsequent worker verifies its actual allocation independently. Avoid
    # a second scheduler observation inside 60s if semantic scans return early.
    spacing = 60 - (time.monotonic() - verified_at)
    if spacing > 0:
        time.sleep(spacing)
    record({"event": "selected_inspection_child_finished", "fit_may_proceed": True,
            "reuse_authorized": False, "retry_authorized": False, "repair_coverage_written": False})
except Exception as error:
    record({"event": "selected_inspection_child_stopped", "error_type": type(error).__name__,
            "error": str(error), "fit_may_proceed": False, "automatic_retry": False})
    raise SystemExit(2)
EP21_SELECTED_INSPECTIONS
