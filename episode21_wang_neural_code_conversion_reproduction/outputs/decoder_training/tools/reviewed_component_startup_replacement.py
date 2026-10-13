"""One manual work48 replacement after unsupported tee startup syntax.

Pure metadata: retain the failed 39-second charge and the original work48
storage reservation. This authorizes no fragment reuse or general retry.
"""
from datetime import datetime


FIELD = "reviewed_component_startup_replacement"
JOB = "47080265_47"
WORK = "work_0048"
ROOT = "outputs/decoder_training/full_continuation_001"
REVIEW_RELATIVE = f"{ROOT}/reviewed_component_startup_replacement_001.json"
PARTIAL_FIELD = "reviewed_deadline_partials"
QUARANTINE_FIELD = "reviewed_io_quarantine"
WRAPPER_RELATIVE = "outputs/decoder_training/full_workers_parallel_003.sbatch"
PREVIOUS_WRAPPER_RELATIVE = "outputs/decoder_training/full_workers_parallel_002.sbatch"
ALLOCATION = {"logical_id": WORK, "array_index": 47, "reserved_CPU_seconds": 28800}
FIXED = {
    "schema_version": "ep21.reviewed_component_startup_replacement.v1",
    "reviewer": "coordinating_agent",
    "predecessor_job_id": JOB,
    "predecessor_allocation_job_id": "47080265",
    "work_item_id": WORK,
    "array_index": 47,
    "failure_reason": "unsupported_tee_output_error_option",
    "replacement_wrapper": WRAPPER_RELATIVE,
    "deadline_partial_review": f"{ROOT}/reviewed_deadline_partials_002.json",
    "io_quarantine_review": f"{ROOT}/reviewed_io_quarantine_001.json",
    "required_checks_passed": 74,
    "optional_pipeline_started": True,
    "optional_component_exit": 141,
    "optional_log_write_exit": 1,
    "shard_directory_absent": True,
    "request_or_status_created": False,
    "fragment_created": False,
    "scientific_payload_admitted": False,
    "scoring_admitted": False,
    "additional_durable_bytes": 0,
    "automatic_retry": False,
}
TEXT_FIELDS = {"reviewed_utc", "fix_evidence", "terminal_evidence",
               "absence_evidence", "user_authorization_context"}
FORBIDDEN_FIELDS = {
    "reviewed_pre_admission_replacement", "reviewed_startup_replacement",
    "reviewed_remainder_bundle_startup_replacement", "reviewed_io_recovery",
    "reviewed_deadline_remainder", "grouped_remainder_work_items",
    "clean_work5_remainder_under_quarantine",
}


class ComponentStartupReplacementError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ComponentStartupReplacementError(message)


def validate_review(review):
    require(isinstance(review, dict) and set(review) == set(FIXED) | TEXT_FIELDS
            and all(type(review[key]) is type(value) and review[key] == value
                    for key, value in FIXED.items())
            and all(isinstance(review[key], str) and review[key].strip()
                    for key in TEXT_FIELDS),
            "exact_work48_component_startup_review_required")
    try:
        observed = datetime.fromisoformat(review["reviewed_utc"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ComponentStartupReplacementError("dated_component_startup_review_required") from exc
    require(observed.utcoffset() is not None,
            "timezone_bearing_component_startup_review_required")


def _wrapper_is(command, relative):
    return (isinstance(command, list) and bool(command)
            and isinstance(command[-1], str)
            and (command[-1] == relative or command[-1].endswith("/" + relative)))


def validate_intent(row, jobs, batches):
    """Return only the exact failed parent bypassed by this reviewed intent."""
    if FIELD not in row:
        return None
    validate_review(row[FIELD])
    require(row.get("phase") == "workers" and row.get("manual_worker_refill") is True
            and row.get("allocations") == [ALLOCATION]
            and not (FORBIDDEN_FIELDS & set(row))
            and _wrapper_is(row.get("command"), WRAPPER_RELATIVE),
            "component_replacement_requires_only_work48_and_wrapper003")
    predecessor = jobs.get(JOB)
    require(isinstance(predecessor, dict) and predecessor.get("phase") == "workers"
            and predecessor.get("logical_id") == WORK
            and predecessor.get("reserved_CPU_seconds") == 28800,
            "component_predecessor_must_be_registered_work48")
    expected = {"allocation_job_id": "47080265", "allocated_cpus": 1,
                "elapsed_seconds": 39, "state": "FAILED", "exit_code": "74:0"}
    terminal = predecessor.get("terminal")
    require(isinstance(terminal, dict)
            and all(type(terminal.get(key)) is type(value) and terminal[key] == value
                    for key, value in expected.items()),
            "exact_component_startup_39_second_failure_must_be_charged")
    previous = batches.get(predecessor.get("batch_id"))
    require(isinstance(previous, dict) and previous.get("accepted_id") == "47080265"
            and previous.get("not_submitted") is False
            and previous.get("phase") == "workers"
            and previous.get("manual_worker_refill") is True
            and previous.get("allocations") == [ALLOCATION]
            and _wrapper_is(previous.get("command"), PREVIOUS_WRAPPER_RELATIVE),
            "component_replacement_requires_exact_accepted_wrapper002_predecessor")
    partial = row.get(PARTIAL_FIELD)
    quarantine = row.get(QUARANTINE_FIELD)
    require(isinstance(partial, dict) and isinstance(quarantine, dict)
            and partial == previous.get(PARTIAL_FIELD)
            and quarantine == previous.get(QUARANTINE_FIELD)
            and partial.get("reviewed_utc") == "2026-10-07T18:07:40Z"
            and isinstance(partial.get("partials"), list)
            and all(isinstance(item, dict) for item in partial["partials"])
            and [item.get("work_item_id") for item in partial["partials"]]
                == ["work_0005", "work_0006", "work_0007", "work_0013"],
            "component_replacement_preserves_partial002_and_original_quarantine")
    require(not any(FIELD in batch for batch in batches.values())
            and {key for key, job in jobs.items() if job.get("logical_id") == WORK} == {JOB},
            "one_manual_component_replacement_only_no_chaining")
    require(all(batch.get("accepted_id") is not None
                and batch.get("not_submitted") is False for batch in batches.values()),
            "component_replacement_requires_reconciled_accepted_history")
    return JOB
