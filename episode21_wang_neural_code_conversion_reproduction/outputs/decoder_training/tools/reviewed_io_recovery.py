"""One explicit five-fragment I/O salvage decision; pure metadata only.

The original failures, quarantine, assignments and CPU charges remain intact.
A valid sidecar does not activate a ledger or submit a job. Callers load the
named metadata and perform ordinary payload checks in an admitted allocation.
"""
from copy import deepcopy
from datetime import datetime
from pathlib import PurePosixPath

import reviewed_deadline_partials as partials
import reviewed_io_quarantine as quarantine

FIELD = "reviewed_io_recovery"
ROOT = quarantine.ROOT
REVIEW_RELATIVE = ROOT + "/reviewed_io_salvage_001.json"
WORK_IDS = tuple(quarantine.KNOWN)
METADATA_BYTES = 64 * 1024**2
GROSS_DURABLE_BYTES = 10_631_803_136
RESERVATION_REDUCTION_BYTES = 2_087_397_376
NET_DURABLE_BYTES = GROSS_DURABLE_BYTES - RESERVATION_REDUCTION_BYTES
# Participant, first conv1_1 channel, exact reported/inspected prefix units.
KNOWN = {
    "work_0032": ("sub-05", 20, 770560),
    "work_0033": ("sub-05", 40, 632320),
    "work_0047": ("sub-06", 20, 723712),
    "work_0061": ("sub-07", 0, 622080),
    "work_0062": ("sub-07", 20, 571904),
}
LIMITS = {"maximum_allocated_CPU_seconds": 1800000, "maximum_concurrent_jobs": 42,
    "peak_transient_bytes": 16000000000, "durable_bytes": 500000000000,
    "fresh_parent_CPU_seconds_each": 28800, "all_five_fresh_parent_CPU_seconds": 144000,
    "admission_requires_current_CPU_count_and_scratch_reservations": True}
ASSEMBLY = {
    "exception_scope": "exact_reviewed_committed_prefixes_only",
    "historical_IO_error_and_fragment_closeout_veto_may_be_overridden": True,
    "historical_uncertain_publication_veto_may_be_overridden_only_for_committed_prefix": True,
    "historical_failures_and_uncertainty_preserved": True,
    "uncertain_rows_are_excluded_from_prefix_and_included_in_fresh_remainder": True,
    "ordinary_payload_identity_marker_coefficient_and_normalization_checks_required": True,
    "ordinary_nonoverlap_and_complete_coverage_checks_required": True,
    "ordinary_separate_assembly_admission_required": True,
    "no_other_fragment_or_stage_exception": True}
TOTALS = {"original_and_existing_remainder_reserved_bytes": 489756203706,
    "fresh_remainder_reserved_bytes": GROSS_DURABLE_BYTES,
    "retained_original_reservation_reduction_bytes": RESERVATION_REDUCTION_BYTES,
    "net_additional_reserved_bytes": NET_DURABLE_BYTES,
    "prospective_reserved_bytes": 498300609466, "maximum_durable_bytes": 500000000000,
    "remaining_unreserved_bytes": 1699390534}
FIXED = {"schema_version": "ep21.reviewed_io_salvage.v1", "reviewer": "coordinating_agent",
    "cause": quarantine.FIXED["cause"],
    "scope": "accept_exact_inspected_prefixes_and_one_manual_fresh_exact_complement_per_listed_work_item",
    "original_work_manifest": ROOT + "/work_manifest.json",
    "original_quarantine_review": quarantine.REVIEW_RELATIVE,
    "original_quarantine_unchanged": True, "automatic_retry": False, "automatic_extension": False,
    "preserve_original_fragments_assignments_failures_and_charges": True,
    "ledger_activated": False, "submission_authorized_by_this_file": False,
    "accepted_committed_units": 3320576, "fresh_remaining_units": 1697024,
    "training_role": "subject_unique", "training_images": 4712,
    "scientific_fit_roles_and_kernels_unchanged": True,
    "allow_prediction_or_scoring": False, "allow_tau_train_values": False,
    "allow_tau_test_values": False, "allow_ood": False, "allow_converters": False,
    "allow_reconstruction": False, "execution_limits": LIMITS,
    "storage_totals": TOTALS, "assembly_policy": ASSEMBLY}


class RecoveryError(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise RecoveryError(reason)


def _same(left, right):
    """Exact JSON types too: True cannot substitute for 1."""
    if type(left) is not type(right):
        return False
    if isinstance(right, dict):
        return set(left) == set(right) and all(_same(left[key], value) for key, value in right.items())
    if isinstance(right, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _dated(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as error:
        raise RecoveryError("timezone_bearing_review_evidence_required") from error
    require(result.utcoffset() is not None, "timezone_bearing_review_evidence_required")
    return result


def _span(participant, channel, start, stop):
    return {"participant": participant, "layer": "conv1_1", "channel": channel,
            "spatial_Fortran_interval": [start, stop]}


def _partition(work):
    participant, first, saved = KNOWN[work]
    requested, committed, fresh = [], [], []
    for index, channel in enumerate(range(first, first + 20)):
        requested.append(_span(participant, channel, 0, 50176))
        stop = min(50176, max(0, saved - index * 50176))
        if stop:
            committed.append(_span(participant, channel, 0, stop))
        if stop < 50176:
            fresh.append(_span(participant, channel, stop, 50176))
    return requested, committed, fresh


def _resources(units, spans):
    return {"cpus": 1, "memory_gib": 7, "gpus": 0, "minutes": 480,
            "durable_bytes": units * 6045 + spans * 1024**2 + METADATA_BYTES,
            "scratch_bytes": 2000000000}


def _original(work):
    requested, _, _ = _partition(work)
    return {"work_item_id": work, "participant": KNOWN[work][0],
            "spans": requested,
            "units_requested": 1003520, "resources": {"cpus": 1, "memory_gib": 7, "gpus": 0,
                "minutes": 480, "durable_bytes": 6087397376, "scratch_bytes": 2000000000}}


def _inspection_identity(work):
    old = work in ("work_0032", "work_0033")
    return ("outputs/decoder_training/remainder_bundle_checks_job_46945868/closed_io_inspection.json"
            if old else f"outputs/decoder_training/selected_io_inspection_job_46949821/{work}.json",
            "46945868" if old else "46949821",
            "stopped_timed_out" if old else "consistent_observations_only")


def _validate_storage(value):
    directory = ROOT + "/shards/work_0062/"
    files = {directory + name: {"st_size": size, "file_type": "regular", "st_nlink": 1,
                               "is_symlink": False} for name, size in
             (("channels.h5", 3639836056), ("requested_units.json", 8433), ("work_item_status.json", 12914))}
    expected = {"work_item_id": "work_0062", "original_reserved_bytes": 6087397376,
        "retained_reserved_bytes": 4000000000, "reservation_reduction_bytes": RESERVATION_REDUCTION_BYTES,
        "files_deleted": False, "original_artifacts_modified": False,
        "stat_observer": "coordinating_agent",
        "stat_method": "exact_three_file_lstat_no_directory_scan_no_payload_read",
        "files": files, "observed_file_bytes": 3639857403,
        "metadata_allowance_bytes": METADATA_BYTES, "observed_plus_metadata_bytes": 3706966267,
        "activation_requires_tested_journal_admission": True}
    require(isinstance(value, dict) and set(value) == set(expected) | {"stat_observed_utc"}
            and all(_same(value[key], item) for key, item in expected.items()),
            "only_exact_work62_four_GB_retained_reservation_is_reviewed")
    _dated(value["stat_observed_utc"])
    require(sum(row["st_size"] for row in files.values()) + METADATA_BYTES < 4000000000,
            "observed_closed_files_plus_metadata_must_fit_retained_reservation")


def validate_review_structure(review, manifest=None):
    """Validate the fixed review, exact intervals and reservation arithmetic."""
    require(isinstance(review, dict) and set(review) == set(FIXED) |
            {"reviewed_utc", "decision", "work_items", "retained_original_reservation"}
            and all(_same(review[key], value) for key, value in FIXED.items()),
            "exact_five_fragment_IO_recovery_decision_required")
    reviewed_at = _dated(review["reviewed_utc"])
    require(isinstance(review["decision"], str) and review["decision"].strip(), "explicit_recovery_decision_required")
    _validate_storage(review["retained_original_reservation"])
    require(reviewed_at >= _dated(review["retained_original_reservation"]["stat_observed_utc"]),
            "review_must_follow_closed_storage_evidence")
    rows = review["work_items"]
    require(isinstance(rows, list) and len(rows) == 5
            and all(isinstance(row, dict) for row in rows)
            and [row.get("work_item_id") for row in rows] == list(WORK_IDS),
            "all_five_ordered_recovery_items_required")
    originals = None
    if manifest is not None:
        originals = manifest["work_items"]
        require(len(originals) == 75 and [row["work_item_id"] for row in originals]
                == [f"work_{index:04d}" for index in range(1, 76)], "unchanged_75_assignments_required")
    for item in rows:
        work = item["work_item_id"]
        participant, _, saved = KNOWN[work]
        job, raw, elapsed = quarantine.KNOWN[work]
        requested, committed, fresh = _partition(work)
        for spans in (item.get("committed_spans"), item.get("spans")):
            partials._spans(spans)
        directory = ROOT + "/shards/" + work
        expected = {"work_item_id": work, "participant": participant, "array_index": int(work[-4:]) - 1,
            "predecessor_job_id": job, "predecessor_allocation_job_id": raw,
            "predecessor_elapsed_seconds": elapsed, "predecessor_status": directory + "/work_item_status.json",
            "predecessor_fragment": directory + "/channels.h5", "predecessor_request": directory + "/requested_units.json",
            "original_units_requested": 1003520, "committed_units": saved, "committed_spans": committed,
            "spans": fresh, "units_requested": 1003520 - saved,
            "output_directory": ROOT + "/io_remainder_shards_001/" + work,
            "resources": _resources(1003520 - saved, len(fresh)), "effective_scratch_bytes": 1012826112,
            "uncertain_rows_in_fresh_remainder": 256 if work in quarantine.UNCERTAIN else 0,
            "uncertain_observed_marker_2_units": 0}
        variable = {"inspection", "historical_closeout_errors", "historical_uncertain_publication"}
        require(set(item) == set(expected) | variable
                and all(_same(item[key], value) for key, value in expected.items()),
                "only_exact_inspected_prefix_and_disjoint_fresh_complement_required")
        errors = item["historical_closeout_errors"]
        require(errors == [] if work == "work_0062" else isinstance(errors, list) and len(errors) == 3
                and all(isinstance(row, dict) for row in errors)
                and [row.get("stage") for row in errors] == ["fragment_fsync", "fragment_close", "fragment_accounting"]
                and all(row.get("affects_fragment") is True for row in errors), "original_closeout_failure_must_remain_recorded")
        uncertain = None
        if work in quarantine.UNCERTAIN:
            channel, start, stop = quarantine.UNCERTAIN[work]
            uncertain = {**_span(participant, channel, start, stop), "phase": "completion_marker_fsync", "error_type": "OSError"}
        require(_same(item["historical_uncertain_publication"], uncertain), "original_uncertainty_must_remain_outside_reused_prefix")
        path, inspection_job, report_status = _inspection_identity(work)
        evidence = {"report": path, "report_status": report_status, "allocation_job_id": inspection_job,
            "work_item_id": work, "fragment_status": "consistent", "inspection_complete": True,
            "reported_committed_units_checked": saved, "observed_marker_2_units": saved,
            "ambiguous_marker_2_units_checked": 0}
        require(isinstance(item["inspection"], dict)
                and set(item["inspection"]) == set(evidence) | {"report_observed_utc"}
                and all(_same(item["inspection"][key], value) for key, value in evidence.items()),
                "exact_completed_inspection_identity_and_prefix_count_required")
        require(reviewed_at >= _dated(item["inspection"]["report_observed_utc"]), "review_must_follow_each_inspection")
        if originals is not None:
            original = originals[item["array_index"]]
            require(all(_same(original.get(key), value) for key, value in _original(work).items()),
                    "review_must_match_original_manifest_assignment")
    require(sum(item["resources"]["durable_bytes"] for item in rows) == GROSS_DURABLE_BYTES,
            "all_five_fresh_outputs_reserved_exactly_once")
    return {item["work_item_id"]: item for item in rows}


def _entry(item):
    return {"work_item_id": item["work_item_id"], "job_id": item["predecessor_job_id"],
            "allocation_job_id": item["predecessor_allocation_job_id"],
            "elapsed_seconds": item["predecessor_elapsed_seconds"], "status_path": item["predecessor_status"]}


def _shared(base_directory, participant):
    return str(PurePosixPath(base_directory) / ROOT / "shared" / participant / "shared_training_state.npz")


def prefix_allowance(review, work, original_record, *, base_directory):
    """Waive only historical closeout/uncertainty for this exact saved prefix.

    The caller still checks original payload identity, markers, coefficients and
    normalization. Returned coverage excludes every formerly uncertain row.
    """
    items = validate_review_structure(review)
    require(work in items, "only_reviewed_IO_prefix_may_receive_exception")
    item, base = items[work], PurePosixPath(base_directory)
    quarantine.validate_closed_record(_entry(item), original_record, _original(work),
        authorization_path=str(base / "outputs/decoder_training_full_authorization.json"),
        manifest_path=str(base / ROOT / "work_manifest.json"), shared_state_path=_shared(base, item["participant"]))
    require(_same(original_record.get("committed_spans"), item["committed_spans"])
            and _same(original_record.get("remaining_spans"), item["spans"])
            and _same(original_record.get("completed_units"), item["committed_units"])
            and _same(original_record.get("closeout_errors"), item["historical_closeout_errors"])
            and _same(original_record.get("uncertain_publication"), item["historical_uncertain_publication"]),
            "current_original_status_must_match_exact_reviewed_prefix_and_history")
    require(original_record.get("completed_units_is_lower_bound") is (work in quarantine.UNCERTAIN)
            and original_record.get("remaining_interval_status") ==
                ("not_authoritative_for_repair_due_to_uncertain_marker_publication" if work in quarantine.UNCERTAIN
                 else "exact_uncommitted_coverage"), "original_coverage_uncertainty_flags_must_remain_recorded")
    return deepcopy(item["committed_spans"])


def _validate_jobs(review, manifest, jobs):
    # Reuse the fixed incident's terminal/identity validator; this manufactured
    # metadata view is not a changed historical quarantine or an authority event.
    quarantine.validate_review({**quarantine.FIXED, "reviewed_utc": review["reviewed_utc"],
        "evidence": "Terminal identity checks for the separate explicit I/O recovery review",
        "failures": [_entry(item) for item in review["work_items"]]}, jobs, manifest)


def validate_review(review, manifest, jobs, closed_records, inspection_reports, *, base_directory):
    """Callers supply closed records keyed by work and reports keyed by path."""
    items = validate_review_structure(review, manifest)
    _validate_jobs(review, manifest, jobs)
    require(set(closed_records) == set(WORK_IDS) and set(inspection_reports)
            == {item["inspection"]["report"] for item in items.values()}, "exact_review_metadata_inputs_required")
    for work, item in items.items():
        prefix_allowance(review, work, closed_records[work], base_directory=base_directory)
        evidence = item["inspection"]
        report = inspection_reports[evidence["report"]]
        require(report.get("schema_version") == "ep21.closed_io_fragment_inspection.v1"
                and report.get("observed_utc") == evidence["report_observed_utc"]
                and report.get("status") == evidence["report_status"]
                and report.get("allocation", {}).get("job_id") == evidence["allocation_job_id"]
                and "earlier_error" not in report
                and all(report.get(key) is False for key in ("original_artifacts_modified",
                    "repair_coverage_written", "retry_authorized", "reuse_authorized")),
                "inspection_is_bound_evidence_not_prior_reuse_authority")
        matches = [row for row in report.get("fragments", []) if row.get("work_item_id") == work]
        require(len(matches) == 1, "one_exact_complete_inspection_fragment_required")
        row = matches[0]
        require(row.get("fragment") == item["predecessor_fragment"]
                and row.get("status") == "consistent" and row.get("inspection_complete") is True
                and "error" not in row and "earlier_error" not in row
                and all(_same(row.get(key), value) for key, value in {
                    "reported_committed_units": item["committed_units"],
                    "reported_committed_units_checked": item["committed_units"],
                    "observed_marker_2_units": item["committed_units"], "ambiguous_marker_2_units_checked": 0,
                    "historical_closeout_errors": item["historical_closeout_errors"],
                    "historical_uncertain_publication": item["historical_uncertain_publication"]}.items())
                and all(row.get(key) is False for key in ("original_artifacts_modified",
                    "repair_coverage_written", "retry_authorized", "reuse_authorized")),
                "complete_consistent_exact_prefix_observation_required")
    return items


def validate_intent(batch, jobs, batches, manifest):
    """No retry chain: one new physical attempt per listed original assignment."""
    review = batch.get(FIELD)
    items = validate_review_structure(review, manifest)
    _validate_jobs(review, manifest, jobs)
    require(quarantine.FIELD in batch, "original_quarantine_must_accompany_recovery")
    quarantine.validate_review(batch[quarantine.FIELD], jobs, manifest)
    for previous in batches.values():
        if quarantine.FIELD in previous:
            require(_same(batch[quarantine.FIELD], previous[quarantine.FIELD]), "original_quarantine_is_immutable")
        if FIELD in previous:
            require(_same(review, previous[FIELD]), "published_IO_recovery_review_is_immutable")
    require(not any(key in batch for key in ("reviewed_pre_admission_replacement", "reviewed_startup_replacement",
        "reviewed_remainder_bundle_startup_replacement", quarantine.bundle.FIELD, quarantine.CLEAN_WORK5_FIELD)),
        "IO_recovery_is_not_a_replacement_or_bundle")
    phase = batch.get("phase")
    if phase == "workers":
        require(batch.get("manual_worker_refill") is True and "reviewed_deadline_remainder" not in batch,
                "explicit_manual_IO_complement_intent_required")
        allocations = batch.get("allocations")
        require(isinstance(allocations, list) and allocations and all(isinstance(row, dict) for row in allocations),
                "explicit_IO_complement_allocations_required")
        selected = [row.get("logical_id") for row in allocations]
        require(len(set(selected)) == len(selected) and set(selected) <= set(WORK_IDS), "only_unique_five_IO_complements_may_be_selected")
        for allocation, work in zip(allocations, selected):
            require(_same(allocation, {"logical_id": work, "array_index": items[work]["array_index"],
                                      "reserved_CPU_seconds": 28800}), "exact_original_index_and_new_eight_hour_charge_required")
            attempts = [job for job in jobs.values() if job["phase"] == "workers" and job["logical_id"] == work]
            require(len(attempts) == 1 and attempts[0]["job_id"] == items[work]["predecessor_job_id"],
                    "one_original_failed_attempt_only_before_IO_complement")
            require(not any(any(row.get("logical_id") == work for row in old.get("allocations", []))
                            for old in batches.values() if FIELD in old and old.get("phase") == "workers"),
                    "one_IO_complement_intent_only_no_automatic_retry")
    else:
        require(phase == "assembly" and "manual_worker_refill" not in batch,
                "only_manual_complement_or_separate_final_assembly_intent")
        require(all(old.get("accepted_id") is not None and not old.get("not_submitted") for old in batches.values())
                and all(job.get("terminal") is not None for job in jobs.values()), "assembly_requires_closed_accounted_history")
        account = {"jobs": jobs, "batches": batches}
        for work in WORK_IDS:
            registered_remainder_job(account, review, work, require_terminal=True)
    return items


def registered_remainder_job(account, review, work, *, require_terminal=False):
    items = validate_review_structure(review)
    require(work in items, "registered_IO_complement_must_belong_to_fixed_review")
    candidates = [job for job in account["jobs"].values() if job.get("phase") == "workers"
        and job.get("logical_id") == work and job.get("batch_id") in account["batches"]
        and _same(account["batches"][job["batch_id"]].get(FIELD), review)
        and account["batches"][job["batch_id"]].get("accepted_id") is not None]
    require(len(candidates) == 1, "one_exact_accepted_IO_complement_required")
    job = candidates[0]
    batch = account["batches"][job["batch_id"]]
    require(job.get("job_id") == str(batch["accepted_id"]) + "_" + str(items[work]["array_index"])
            and batch.get("phase") == "workers" and batch.get("manual_worker_refill") is True
            and {"logical_id": work, "array_index": items[work]["array_index"], "reserved_CPU_seconds": 28800}
                in batch.get("allocations", [])
            and job.get("reserved_CPU_seconds") == 28800, "exact_registered_IO_array_and_reservation_required")
    terminal = job.get("terminal")
    if require_terminal:
        require(terminal is not None and terminal.get("state") == "COMPLETED"
                and terminal.get("exit_code") == "0:0" and terminal.get("allocated_cpus") == 1,
                "complete_charged_IO_complement_required")
    else:
        require(terminal is None, "IO_complement_worker_must_be_outstanding")
    return job


def validate_completed_remainder(item, record, registered_job, *, base_directory):
    """Bind a fresh successful complement to its separately charged allocation."""
    work = item.get("work_item_id")
    require(work in KNOWN, "only_fixed_IO_complement_can_complete")
    _, _, fresh = _partition(work)
    require(_same(item.get("spans"), fresh)
            and _same(item.get("units_requested"), 1003520 - KNOWN[work][2])
            and _same(item.get("resources"), _resources(1003520 - KNOWN[work][2], len(fresh)))
            and item.get("effective_scratch_bytes") == 1012826112
            and item.get("predecessor_status") == ROOT + "/shards/" + work + "/work_item_status.json"
            and item.get("output_directory") == ROOT + "/io_remainder_shards_001/" + work,
            "exact_fresh_IO_complement_required")
    terminal = registered_job.get("terminal")
    require(registered_job.get("phase") == "workers" and registered_job.get("logical_id") == work
            and registered_job.get("reserved_CPU_seconds") == 28800
            and terminal is not None and terminal.get("state") == "COMPLETED"
            and terminal.get("exit_code") == "0:0" and terminal.get("allocated_cpus") == 1,
            "separately_charged_successful_IO_complement_required")
    base = PurePosixPath(base_directory)
    expected = {"schema_version": "ep21.decoder_work_item.v1", "status": "complete_work_item_only",
        "work_item_id": work, "participant": KNOWN[work][0], "resources": item["resources"],
        "authorization": str(base / "outputs/decoder_training_full_authorization.json"),
        "work_manifest": str(base / ROOT / "work_manifest.json"), "io_recovery_review": str(base / REVIEW_RELATIVE),
        "predecessor_job_id": quarantine.KNOWN[work][0], "predecessor_status": item["predecessor_status"],
        "shared_state_reference": _shared(base, KNOWN[work][0]), "fragment": "channels.h5",
        "requested_spans": fresh, "committed_spans": fresh, "remaining_spans": [],
        "units_requested": item["units_requested"], "completed_units": item["units_requested"],
        "remaining_interval_status": "exact_uncommitted_coverage", "completed_units_is_lower_bound": False,
        "training_role": "subject_unique", "training_images": 4712, "original_neural_values_read": False,
        "retained_neural_state_read": True, "original_budget_consumption_preserved": True,
        "effective_scratch_bytes": item["effective_scratch_bytes"], "parallelism_amendment_id": "parallelism_amendment"}
    expected.update({key: False for key in ("predictions_or_scores_computed", "tau_train_values_read",
        "tau_test_values_read", "OOD_access", "complete_decoder", "successor_authorized", "automatic_retry")})
    require(all(_same(record.get(key), value) for key, value in expected.items()), "completed_IO_complement_identity_or_coverage_mismatch")
    allocation = record.get("allocation", {})
    require(all(_same(allocation.get(key), value) for key, value in {"job_id": terminal["allocation_job_id"],
        "allocated_cpus": 1, "memory_gib": 7, "gpus": 0, "time_limit_seconds": 28800}.items()),
        "completed_IO_complement_allocation_mismatch")
    require("uncertain_publication" not in record and isinstance(record.get("closeout_errors"), list)
            and all(isinstance(row, dict) and row.get("affects_fragment") is False for row in record["closeout_errors"]),
            "fresh_IO_complement_requires_ordinary_successful_closeout")


def _published(batches):
    reviews = [batch[FIELD] for batch in batches.values() if FIELD in batch]
    if not reviews:
        return None
    for review in reviews:
        validate_review_structure(review)
    require(all(_same(review, reviews[0]) for review in reviews), "published_IO_recovery_review_is_immutable")
    return reviews[0]


def reserved_durable_bytes(batches):
    """Full five-output reservation once, including an unresolved first intent."""
    return 0 if _published(batches) is None else GROSS_DURABLE_BYTES


def net_additional_durable_bytes(review):
    validate_review_structure(review)
    return NET_DURABLE_BYTES
