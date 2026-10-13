"""One reviewed Oak incident: quarantine five assignments, never reuse them.

Metadata only. This disposition admits unrelated manual work, not a retry,
coverage claim, storage release, fragment repair, or assembly exception. A
separate exact recovery review may accompany it without changing this history.
"""
from datetime import datetime

import decoder_remainder_bundle as bundle


FIELD = "reviewed_io_quarantine"
CLEAN_WORK5_FIELD = "clean_work5_remainder_under_quarantine"
ROOT = "outputs/decoder_training/full_continuation_001"
REVIEW_RELATIVE = f"{ROOT}/reviewed_io_quarantine_001.json"
KNOWN = {
    "work_0032": ("46918887_31", "46919303", 12689),
    "work_0033": ("46918887_32", "46919304", 12689),
    "work_0047": ("46918887_46", "46919447", 12689),
    "work_0061": ("46918155_60", "46918155", 13322),
    "work_0062": ("46918887_61", "46918887", 12398),
}
UNCERTAIN = {"work_0032": (35, 17920, 18176), "work_0061": (12, 19968, 20224)}
FIXED = {
    "schema_version": "ep21.reviewed_io_quarantine.v1",
    "reviewer": "coordinating_agent",
    "cause": "reviewed_2026_10_07_Oak_input_output_error",
    "allows_only": "fresh_unassigned_manual_workers_or_existing_work6_work13_bundle",
    "automatic_retry": False,
    "preserve_original_fragments_assignments_and_charges": True,
    "validated_coverage_credit": 0,
    "allow_fragment_reuse": False,
    "allow_failed_work_retry": False,
    "allow_failed_work_remainder": False,
    "allow_assembly_exception": False,
    "release_durable_reservation": False,
}


class QuarantineError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise QuarantineError(message)


def validate_review(review, jobs, manifest):
    require(isinstance(review, dict)
            and set(review) == set(FIXED) | {"reviewed_utc", "evidence", "failures"}
            and all(type(review[key]) is type(value) and review[key] == value
                    for key, value in FIXED.items()), "exact_incident_quarantine_review_required")
    try:
        observed = datetime.fromisoformat(review["reviewed_utc"].replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise QuarantineError("dated_quarantine_review_required") from exc
    require(observed.utcoffset() is not None and isinstance(review["evidence"], str)
            and review["evidence"].strip(), "dated_sourced_quarantine_review_required")
    entries = review["failures"]
    require(isinstance(entries, list) and len(entries) == len(KNOWN),
            "all_five_exact_incident_failures_required")
    items = {item["work_item_id"]: item for item in manifest["work_items"]}
    reviewed = {}
    for entry, (work, (job_id, raw, elapsed)) in zip(entries, KNOWN.items()):
        expected = {"work_item_id": work, "job_id": job_id, "allocation_job_id": raw,
                    "elapsed_seconds": elapsed,
                    "status_path": f"{ROOT}/shards/{work}/work_item_status.json"}
        require(isinstance(entry, dict) and set(entry) == set(expected)
                and all(type(entry[key]) is type(value) and entry[key] == value
                        for key, value in expected.items()) and work in items,
                "only_exact_five_reviewed_io_allocations_may_be_quarantined")
        job = jobs.get(job_id)
        require(job is not None and job["phase"] == "workers" and job["logical_id"] == work,
                "quarantine_requires_registered_original_worker")
        terminal = job["terminal"]
        required = {"state": "FAILED", "exit_code": "2:0", "allocated_cpus": 1,
                    "allocation_job_id": raw, "elapsed_seconds": elapsed}
        require(terminal is not None and all(type(terminal.get(key)) is type(value)
                and terminal[key] == value for key, value in required.items())
                and job["reserved_CPU_seconds"] == 28800,
                "quarantine_requires_exact_existing_failed_terminal_charge")
        reviewed[job_id] = entry
    return reviewed



def validate_clean_work5_remainder(batch):
    """One separately selected clean remainder; the five failed jobs stay closed.

    The caller still validates the complete immutable plan, authentic original
    closeout, first remainder attempt and aggregate resources. This marker adds
    no permissions to the historical quarantine review itself.
    """
    require(batch.get(CLEAN_WORK5_FIELD) is True
            and batch.get("phase") == "workers" and batch.get("manual_worker_refill") is True
            and bundle.FIELD not in batch
            and not any(key in batch for key in ("reviewed_pre_admission_replacement",
                "reviewed_startup_replacement", "reviewed_remainder_bundle_startup_replacement")),
            "explicit_ungrouped_clean_work5_remainder_marker_required")
    expected_allocation = {"logical_id": "work_0005", "array_index": 4, "reserved_CPU_seconds": 28800}
    allocations = batch.get("allocations")
    require(isinstance(allocations, list) and len(allocations) == 1
            and allocations[0] == expected_allocation
            and all(type(allocations[0][key]) is type(value) for key, value in expected_allocation.items()),
            "clean_work5_requires_one_index4_eight_hour_parent")
    plan = batch.get(bundle.PLAN_FIELD)
    require(isinstance(plan, dict)
            and plan.get("schema_version") == "ep21.decoder_deadline_remainders.v1"
            and plan.get("original_work_manifest") == f"{ROOT}/work_manifest.json"
            and plan.get("reviewed_partials") == f"{ROOT}/reviewed_deadline_partials_002.json"
            and plan.get("automatic_retry") is False and plan.get("preserve_original_fragments") is True
            and plan.get("scope") == "one_manual_exact_remainder_attempt_per_listed_work_item",
            "clean_work5_requires_unchanged_cumulative_plan002")
    items = plan.get("work_items")
    require(isinstance(items, list) and all(isinstance(item, dict) for item in items)
            and [item.get("work_item_id") for item in items]
                == ["work_0005", "work_0006", "work_0007", "work_0013"],
            "clean_work5_preserves_complete_plan002_members")
    item = items[0]
    expected = {"work_item_id": "work_0005", "participant": "sub-01", "array_index": 4,
                "predecessor_job_id": "46839670_4", "predecessor_allocation_job_id": "46840029",
                "predecessor_status": f"{ROOT}/shards/work_0005/work_item_status.json",
                "output_directory": f"{ROOT}/remainder_shards_001/work_0005", "units_requested": 246656}
    require(all(type(item.get(key)) is type(value) and item[key] == value for key, value in expected.items()),
            "clean_work5_preserves_original_identity_path_and_246656_remaining_units")
    resources = item.get("resources", {})
    require(all(type(resources.get(key)) is int and resources[key] == value for key, value in
                {"cpus": 1, "memory_gib": 7, "gpus": 0, "minutes": 480, "scratch_bytes": 2_000_000_000}.items()),
            "clean_work5_preserves_original_resource_limits")
    return ("work_0005",)


def validate_intent(batch, jobs, manifest, batches):
    """An explicit disposition is immutable and must accompany later intents."""
    prior = [value[FIELD] for value in batches.values() if FIELD in value]
    if prior:
        require(all(value == prior[0] for value in prior) and batch.get(FIELD) == prior[0],
                "published_io_quarantine_cannot_change_narrow_or_disappear")
    require(CLEAN_WORK5_FIELD not in batch or FIELD in batch,
            "clean_work5_requires_unchanged_incident_quarantine")
    # Local import avoids the helper's deliberate dependency on this immutable
    # incident diagnosis. This separate review alone can authorize its exact
    # complements or final assembly; historical quarantine permissions stay false.
    import reviewed_io_recovery as recovery
    require(recovery.FIELD not in batch or FIELD in batch,
            "IO_recovery_requires_unchanged_incident_quarantine")
    if FIELD not in batch:
        return {}
    if recovery.FIELD in batch:
        reviewed = validate_review(batch[FIELD], jobs, manifest)
        recovery.validate_intent(batch, jobs, batches, manifest)
        return reviewed
    # A separate review repairs only the named pre-fit logging failure. The
    # original five-fragment quarantine retains all its unchanged restrictions.
    import reviewed_component_startup_replacement as component_startup
    if component_startup.FIELD in batch:
        reviewed = validate_review(batch[FIELD], jobs, manifest)
        component_startup.validate_intent(batch, jobs, batches)
        return reviewed
    require(batch.get("phase") == "workers" and batch.get("manual_worker_refill") is True
            and not any(key in batch for key in (
                "reviewed_pre_admission_replacement", "reviewed_startup_replacement")),
            "io_quarantine_only_for_explicit_unrelated_manual_workers")
    reviewed = validate_review(batch[FIELD], jobs, manifest)
    if CLEAN_WORK5_FIELD in batch:
        selected = set(validate_clean_work5_remainder(batch))
    elif bundle.PLAN_FIELD in batch or bundle.FIELD in batch:
        selected = set(bundle.validate_bundle(batch))
    else:
        selected = {item["logical_id"] for item in batch["allocations"]}
        require(selected and not selected.intersection(job["logical_id"] for job in jobs.values()),
                "io_quarantine_grants_no_retry_or_reassignment")
    require(not selected.intersection(KNOWN), "quarantined_work_cannot_be_selected")
    return reviewed


def validate_closed_record(entry, record, item, *, authorization_path, manifest_path, shared_state_path):
    """Check incident identity; deliberately do not qualify fragment contents."""
    work = entry["work_item_id"]
    require(record.get("schema_version") == "ep21.decoder_work_item.v1"
            and record.get("status") == "partial_work_item_stopped"
            and record.get("error_type") == "OSError"
            and record.get("error") == "[Errno 5] Input/output error",
            "quarantine_requires_observed_Oak_io_failure")
    require(record.get("work_item_id") == item["work_item_id"] == work
            and record.get("participant") == item["participant"]
            and record.get("resources") == item["resources"]
            and record.get("requested_spans") == item["spans"]
            and record.get("units_requested") == item["units_requested"]
            and record.get("authorization") == authorization_path
            and record.get("work_manifest") == manifest_path
            and record.get("shared_state_reference") == shared_state_path
            and record.get("fragment") == "channels.h5",
            "quarantine_original_scope_or_resource_mismatch")
    allocation = record.get("allocation", {})
    expected = {"job_id": entry["allocation_job_id"], "hostname": "sh04-04n21",
                "allocated_cpus": 1, "memory_gib": 7, "gpus": 0, "time_limit_seconds": 28800}
    require(all(type(allocation.get(key)) is type(value) and allocation[key] == value
                for key, value in expected.items()), "quarantine_original_allocation_mismatch")
    require(record.get("training_role") == "subject_unique" and record.get("training_images") == 4712
            and record.get("original_neural_values_read") is False
            and record.get("retained_neural_state_read") is True
            and record.get("original_budget_consumption_preserved") is True
            and all(record.get(key) is False for key in (
                "predictions_or_scores_computed", "tau_train_values_read", "tau_test_values_read",
                "OOD_access", "complete_decoder", "successor_authorized", "automatic_retry")),
            "quarantine_does_not_waive_data_or_execution_boundaries")
    errors = record.get("closeout_errors")
    if work == "work_0062":
        require(errors == [], "quarantine_work62_closed_metadata_changed")
    else:
        require(isinstance(errors, list) and len(errors) == 3
                and [error.get("stage") for error in errors]
                    == ["fragment_fsync", "fragment_close", "fragment_accounting"]
                and all(error.get("affects_fragment") is True for error in errors),
                "quarantine_requires_recorded_fragment_closeout_failure")
    if work in UNCERTAIN:
        channel, start, stop = UNCERTAIN[work]
        require(record.get("uncertain_publication") == {
            "participant": item["participant"], "layer": "conv1_1", "channel": channel,
            "spatial_Fortran_interval": [start, stop], "phase": "completion_marker_fsync",
            "error_type": "OSError"}, "quarantine_marker_uncertainty_changed")
    else:
        require("uncertain_publication" not in record, "quarantine_marker_evidence_changed")
