"""Metadata-only admission of reviewed deadline partials for unrelated workers.

No file reads, retries, coverage reassignment, storage reclamation or completion
claims. A review may accompany only a fresh manual worker submission. Original
assignments and physical allocation charges remain in the unchanged journal.
"""
from datetime import datetime
import math


FIELD = "reviewed_deadline_partials"
ROOT = "outputs/decoder_training/full_continuation_001"


class ReviewError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ReviewError(message)


def validate_intent(batch, jobs, manifest):
    require(batch.get("phase") == "workers" and batch.get("manual_worker_refill") is True
            and "reviewed_pre_admission_replacement" not in batch,
            "deadline_partial_review_only_for_fresh_manual_workers")
    review = batch.get(FIELD)
    exact = {"schema_version": "ep21.reviewed_deadline_partials.v1",
             "reviewer": "coordinating_agent", "cause": "ordinary_work_item_deadline",
             "automatic_retry": False, "allows_only": "fresh_unassigned_manual_workers"}
    require(isinstance(review, dict) and set(review) == set(exact) | {"reviewed_utc", "partials"}
            and all(type(review[key]) is type(value) and review[key] == value
                    for key, value in exact.items()), "exact_deadline_partial_review_required")
    try:
        observed = datetime.fromisoformat(review["reviewed_utc"].replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ReviewError("dated_deadline_partial_review_required") from exc
    require(observed.utcoffset() is not None, "timezone_bearing_deadline_partial_review_required")
    entries = review["partials"]
    require(isinstance(entries, list) and entries, "explicit_reviewed_partial_entries_required")
    items = {item["work_item_id"]: item for item in manifest["work_items"]}
    reviewed, logical = {}, set()
    selected = {item["logical_id"] for item in batch["allocations"]}
    for entry in entries:
        require(isinstance(entry, dict) and set(entry) == {
            "work_item_id", "job_id", "allocation_job_id", "elapsed_seconds", "completed_units", "status_path"},
            "exact_deadline_partial_entry_required")
        job_id, work = entry["job_id"], entry["work_item_id"]
        require(isinstance(job_id, str) and isinstance(work, str)
                and job_id not in reviewed and work not in logical and work in items
                and work not in selected, "unique_unselected_reviewed_partial_required")
        job = jobs.get(job_id)
        require(job is not None and job["phase"] == "workers" and job["logical_id"] == work,
                "reviewed_partial_must_be_registered_worker")
        terminal = job["terminal"]
        require(terminal is not None and terminal["state"] == "FAILED"
                and terminal["exit_code"] == "2:0" and terminal["allocated_cpus"] == 1
                and terminal["allocation_job_id"] == entry["allocation_job_id"]
                and type(entry["elapsed_seconds"]) is int and entry["elapsed_seconds"] > 0
                and terminal["elapsed_seconds"] == entry["elapsed_seconds"]
                and terminal["elapsed_seconds"] <= job["reserved_CPU_seconds"],
                "reviewed_deadline_partial_requires_exact_terminal_charge")
        require(type(entry["completed_units"]) is int
                and 0 < entry["completed_units"] < items[work]["units_requested"]
                and entry["status_path"] == f"{ROOT}/shards/{work}/work_item_status.json",
                "reviewed_partial_count_and_fixed_status_path_required")
        reviewed[job_id] = entry
        logical.add(work)
    return reviewed


def _spans(rows):
    require(isinstance(rows, list), "explicit_partial_span_list_required")
    result = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {
            "participant", "layer", "channel", "spatial_Fortran_interval"}, "exact_partial_span_fields_required")
        interval = row["spatial_Fortran_interval"]
        require(isinstance(row["participant"], str) and isinstance(row["layer"], str)
                and type(row["channel"]) is int and row["channel"] >= 0
                and isinstance(interval, list) and len(interval) == 2
                and all(type(value) is int for value in interval) and 0 <= interval[0] < interval[1],
                "exact_partial_span_coordinates_required")
        key = row["participant"], row["layer"], row["channel"]
        require(key not in result, "duplicate_partial_channel_interval")
        result[key] = tuple(interval)
    return result


def validate_closed_record(entry, record, item, *, authorization_path, manifest_path, shared_state_path):
    require(record.get("schema_version") == "ep21.decoder_work_item.v1"
            and record.get("status") == "partial_work_item_stopped"
            and record.get("error_type") == "Stop" and record.get("error") == "work_item_deadline_reached",
            "only_diagnosed_ordinary_work_item_deadline_is_reviewable")
    require(record.get("work_item_id") == item["work_item_id"] == entry["work_item_id"]
            and record.get("participant") == item["participant"]
            and record.get("resources") == item["resources"]
            and record.get("authorization") == authorization_path
            and record.get("work_manifest") == manifest_path
            and record.get("shared_state_reference") == shared_state_path
            and record.get("fragment") == "channels.h5", "reviewed_partial_scope_or_resource_mismatch")
    allocation = record.get("allocation", {})
    cap = item["resources"]
    require(allocation.get("job_id") == entry["allocation_job_id"]
            and allocation.get("allocated_cpus") == cap["cpus"] == 1
            and allocation.get("memory_gib") == cap["memory_gib"] == 7
            and allocation.get("gpus") == cap["gpus"] == 0
            and allocation.get("time_limit_seconds") == cap["minutes"]*60
            and isinstance(allocation.get("hostname"), str) and allocation["hostname"]
            and "-ln" not in allocation["hostname"]
            and type(allocation.get("job_end_epoch")) in (int, float)
            and math.isfinite(allocation["job_end_epoch"]) and allocation["job_end_epoch"] > 0,
            "reviewed_partial_allocation_mismatch")
    require(record.get("training_role") == "subject_unique" and record.get("training_images") == 4712
            and record.get("original_neural_values_read") is False
            and record.get("retained_neural_state_read") is True
            and record.get("original_budget_consumption_preserved") is True
            and all(record.get(key) is False for key in (
                "predictions_or_scores_computed", "tau_train_values_read", "tau_test_values_read",
                "OOD_access", "complete_decoder", "successor_authorized", "automatic_retry")),
            "reviewed_partial_data_and_execution_boundaries_required")
    errors = record.get("closeout_errors")
    require(record.get("remaining_interval_status") == "exact_uncommitted_coverage"
            and record.get("completed_units_is_lower_bound") is False
            and "uncertain_publication" not in record
            and isinstance(errors, list)
            and all(isinstance(error, dict) and error.get("affects_fragment") is False for error in errors),
            "uncertain_or_unfinalized_partial_cannot_release_fresh_work")
    require(record.get("requested_spans") == item["spans"]
            and type(record.get("units_requested")) is int
            and record["units_requested"] == item["units_requested"]
            and type(record.get("completed_units")) is int
            and record["completed_units"] == entry["completed_units"],
            "reviewed_partial_request_or_count_mismatch")
    requested = _spans(item["spans"])
    committed = _spans(record.get("committed_spans"))
    remaining = _spans(record.get("remaining_spans"))
    require(remaining and set(committed) <= set(requested) and set(remaining) <= set(requested),
            "partial_intervals_must_belong_to_exact_request")
    require(list(committed) == [key for key in requested if key in committed]
            and list(remaining) == [key for key in requested if key in remaining],
            "partial_intervals_must_keep_manifest_order")
    for key, (start, stop) in requested.items():
        done = committed.get(key)
        cursor = start if done is None else done[1]
        require((done is None or done[0] == start) and start <= cursor <= stop
                and remaining.get(key) == ((cursor, stop) if cursor < stop else None),
                "committed_and_remaining_must_exactly_partition_each_requested_span")
    require(sum(stop-start for start, stop in requested.values()) == item["units_requested"]
            and sum(stop-start for start, stop in committed.values()) == entry["completed_units"]
            and sum(stop-start for start, stop in remaining.values())
                == item["units_requested"]-entry["completed_units"], "partial_coverage_counts_mismatch")
