"""Manual replacement of three named failures before decoder fitting began.

The third attempt read quarantined payload during a bounded inspection.
Pure metadata: preserve all 25/21/625-second charges and logical items.
No general retry, fragment reuse, inspection, scheduler call or file mutation.
"""
from datetime import datetime

import decoder_remainder_bundle as bundle


FIELD = "reviewed_remainder_bundle_startup_replacement"
PREDECESSOR = "46942785_5"
REVIEW_RELATIVE = ("outputs/decoder_training/full_continuation_001/"
                   "reviewed_bundle_startup_replacement_001.json")
QUARANTINE_FIELD = "reviewed_io_quarantine"
FIXED = {
    "schema_version": "ep21.reviewed_bundle_startup_replacement.v1",
    "reviewer": "coordinating_agent",
    "predecessor_job_id": PREDECESSOR,
    "predecessor_allocation_job_id": "46942785",
    "logical_work_items": list(bundle.MEMBERS),
    "array_index": bundle.ARRAY_INDEX,
    "remainder_plan": bundle.PLAN_RELATIVE,
    "failure_reason": "required_quarantine_fixtures_nested_autospec",
    "scientific_output_directories_absent": True,
    "request_or_status_created": False,
    "scientific_payload_admitted": False,
    "io_fragment_inspection_started": False,
    "optional_children_started": False,
    "automatic_retry": False,
}
PREDECESSOR_002 = "46943996_5"
REVIEW_RELATIVE_002 = ("outputs/decoder_training/full_continuation_001/"
                       "reviewed_bundle_startup_replacement_002.json")
FIXED_002 = {
    **FIXED,
    "schema_version": "ep21.reviewed_bundle_startup_replacement.v2",
    "predecessor_job_id": PREDECESSOR_002,
    "predecessor_allocation_job_id": "46943996",
    "failure_reason": "required_startup_fixture_mutated_shared_terminal_history",
    "previous_startup_review": REVIEW_RELATIVE,
}
PREDECESSOR_003 = "46945868_5"
REVIEW_RELATIVE_003 = ("outputs/decoder_training/full_continuation_001/"
                       "reviewed_bundle_startup_replacement_003.json")
FIXED_003 = {
    **{key: value for key, value in FIXED_002.items() if key not in {
        "scientific_output_directories_absent", "request_or_status_created",
        "scientific_payload_admitted"}},
    "schema_version": "ep21.reviewed_bundle_startup_replacement.v3",
    "predecessor_job_id": PREDECESSOR_003,
    "predecessor_allocation_job_id": "46945868",
    "failure_reason": "bounded_quarantine_inspection_timeout_before_decoder_fits",
    "previous_startup_review": REVIEW_RELATIVE_002,
    "scientific_fit_children_absent": True,
    "decoder_fit_output_directories_absent": True,
    "decoder_fit_request_or_status_created": False,
    "read_only_quarantine_payload_inspected": True,
    "io_fragment_inspection_started": True,
    "io_fragment_inspection_outcome": "bounded_timeout",
    "quarantine_inspection_required_before_unrelated_fits": False,
}
TEXT_FIELDS = {"reviewed_utc", "fix_evidence", "terminal_evidence",
               "absence_evidence", "user_authorization_context"}


class BundleStartupReplacementError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise BundleStartupReplacementError(message)


def validate_review(review):
    fixed = (FIXED_003 if isinstance(review, dict)
             and review.get("schema_version") == FIXED_003["schema_version"] else
             FIXED_002 if isinstance(review, dict)
             and review.get("schema_version") == FIXED_002["schema_version"] else FIXED)
    require(isinstance(review, dict) and set(review) == set(fixed) | TEXT_FIELDS
            and all(type(review[key]) is type(value) and review[key] == value
                    for key, value in fixed.items())
            and all(isinstance(review[key], str) and review[key].strip()
                    for key in TEXT_FIELDS), "exact_preempirical_bundle_startup_review_required")
    try:
        observed = datetime.fromisoformat(review["reviewed_utc"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise BundleStartupReplacementError("dated_bundle_startup_review_required") from exc
    require(observed.utcoffset() is not None, "timezone_bearing_bundle_startup_review_required")


def review_relative(review):
    validate_review(review)
    return {FIXED["schema_version"]: REVIEW_RELATIVE,
            FIXED_002["schema_version"]: REVIEW_RELATIVE_002,
            FIXED_003["schema_version"]: REVIEW_RELATIVE_003}[review["schema_version"]]


def validate_intent(row, jobs, batches):
    """Return the one bypassed startup parent only for this exact new intent."""
    if FIELD not in row:
        return None
    validate_review(row[FIELD])
    supplement = row[FIELD]["schema_version"] == FIXED_002["schema_version"]
    inspection_supplement = row[FIELD]["schema_version"] == FIXED_003["schema_version"]
    predecessor_id = (PREDECESSOR_003 if inspection_supplement else
                      PREDECESSOR_002 if supplement else PREDECESSOR)
    raw = "46945868" if inspection_supplement else "46943996" if supplement else "46942785"
    elapsed = 625 if inspection_supplement else 21 if supplement else 25
    bundle.validate_bundle(row)
    require(QUARANTINE_FIELD in row, "bundle_startup_replacement_preserves_existing_quarantine")
    predecessor = jobs.get(predecessor_id)
    require(predecessor is not None and predecessor.get("phase") == "workers"
            and predecessor.get("logical_id") == bundle.ANCHOR
            and predecessor.get("reserved_CPU_seconds") == bundle.RESERVED_CPU_SECONDS,
            "bundle_startup_predecessor_must_be_registered_two_hour_parent")
    expected = {"allocation_job_id": raw, "allocated_cpus": 1,
                "elapsed_seconds": elapsed, "state": "FAILED", "exit_code": "1:0"}
    terminal = predecessor.get("terminal")
    require(terminal is not None and all(type(terminal.get(key)) is type(value)
            and terminal[key] == value for key, value in expected.items()),
            f"exact_bundle_startup_{elapsed}_second_failure_must_be_charged")
    previous = batches.get(predecessor.get("batch_id"))
    require(previous is not None and previous.get("accepted_id") == raw
            and previous.get("not_submitted") is False,
            "bundle_startup_requires_exact_accepted_predecessor_batch")
    bundle.validate_bundle(previous)
    require(previous.get(bundle.PLAN_FIELD) == row[bundle.PLAN_FIELD]
            and previous.get(QUARANTINE_FIELD) == row[QUARANTINE_FIELD]
            and previous.get("allocations") == row["allocations"],
            "bundle_startup_must_preserve_plan_items_reservation_and_quarantine")
    if inspection_supplement:
        prior = [batch for batch in batches.values() if FIELD in batch]
        require(len(prior) == 2 and prior[-1] is previous
                and [batch[FIELD].get("schema_version") for batch in prior]
                    == [FIXED["schema_version"], FIXED_002["schema_version"]]
                and [batch.get("accepted_id") for batch in prior] == ["46943996", "46945868"],
                "exact_third_startup_supplement_only_no_further_chaining")
        for old_batch in prior:
            validate_review(old_batch[FIELD])
        for job_id, old_raw, old_seconds in ((PREDECESSOR, "46942785", 25),
                                            (PREDECESSOR_002, "46943996", 21)):
            old_terminal = jobs.get(job_id, {}).get("terminal")
            old_expected = {"allocation_job_id": old_raw, "allocated_cpus": 1,
                            "elapsed_seconds": old_seconds, "state": "FAILED", "exit_code": "1:0"}
            require(isinstance(old_terminal, dict) and all(type(old_terminal.get(key)) is type(value)
                    and old_terminal[key] == value for key, value in old_expected.items()),
                    "third_startup_preserves_known_25_and_21_second_charges")
    elif supplement:
        prior = [batch for batch in batches.values() if FIELD in batch]
        require(len(prior) == 1 and prior[0] is previous
                and previous[FIELD].get("schema_version") == FIXED["schema_version"],
                "exact_second_startup_supplement_only_no_further_chaining")
        validate_review(previous[FIELD])
        original = jobs.get(PREDECESSOR, {}).get("terminal")
        first_expected = {"allocation_job_id": "46942785", "allocated_cpus": 1,
                          "elapsed_seconds": 25, "state": "FAILED", "exit_code": "1:0"}
        require(isinstance(original, dict) and all(type(original.get(key)) is type(value)
                and original[key] == value for key, value in first_expected.items()),
                "second_startup_preserves_original_25_second_charge")
    else:
        require(not any(FIELD in batch for batch in batches.values()),
                "one_manual_bundle_startup_replacement_only_no_chaining")
    require(all(batch.get("accepted_id") is not None and batch.get("not_submitted") is False
                for batch in batches.values()),
            "bundle_startup_replacement_requires_reconciled_accepted_history")
    return predecessor_id


def superseded_jobs(batches):
    """Only accepted replacements suppress the known parent's result lookup.

    Replay validates the complete intent first. The original allocation stays
    in jobs with its unchanged terminal charge; an intent alone suppresses none.
    """
    result = set()
    for batch in batches.values():
        if FIELD in batch and batch.get("accepted_id") is not None:
            validate_review(batch[FIELD])
            bundle.validate_bundle(batch)
            require(batch.get("not_submitted") is False,
                    "accepted_bundle_startup_replacement_cannot_be_rejected")
            result.add(batch[FIELD]["predecessor_job_id"])
    return result
