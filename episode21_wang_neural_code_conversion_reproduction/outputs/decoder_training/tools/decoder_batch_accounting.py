"""EP21 finite-batch accounting, with no scheduler or scientific I/O.

Events are appended under one file lock; existing events are never rewritten.
An intent reserves its full request even when scheduler acceptance is unknown.
Terminal charges are allocated CPUs times elapsed parent seconds, not task CPU
utilization. This module does not retry, infer completion or assign new units.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal

import reviewed_deadline_partials as deadline_partials
import decoder_deadline_remainders as deadline_remainders
import decoder_remainder_bundle as remainder_bundle
import reviewed_io_quarantine as io_quarantine
import reviewed_io_recovery as io_recovery
import reviewed_bundle_startup_replacement as bundle_startup
import reviewed_component_startup_replacement as component_startup
import decoder_parallelism_amendment as parallelism

MODE = "actual_allocated_with_full_outstanding_reservations"
JOURNAL_BYTES = 64 * 1024**2
TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
REPLACEMENT_FIELD = "reviewed_pre_admission_replacement"
MANUAL_REFILL_FIELD = "manual_worker_refill"
REPLACED_JOB = "46839670_7"
REPLACED_WORK = "work_0008"
STARTUP_REPLACEMENT_FIELD = "reviewed_startup_replacement"
STARTUP_REPLACED_JOB = "46880582_12"
STARTUP_REPLACED_WORK = "work_0013"
COEXISTING_WORKERS = frozenset(f"46839670_{index}" for index in range(7))


class AccountingError(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise AccountingError(message)


def budget_definition(authority):
    need(authority.get("cpu_budget_mode") == MODE, "explicit_actual_allocation_budget_mode_required")
    caps = authority["resources"]
    cpu, durable = caps["maximum_allocated_CPU_seconds"], caps["durable_bytes"]
    need(type(cpu) is int and 0 < cpu <= 500 * 3600
         and type(durable) is int and 0 < durable <= 500_000_000_000,
         "finite_500_CPU_hour_and_500_GB_ceiling_required")
    need("included_layer_job" in authority, "explicit_included_layer_job_or_null_required")
    layer = authority["included_layer_job"]
    if layer is not None:
        need(isinstance(layer, dict) and set(layer) == {"job_id", "reserved_CPU_seconds", "durable_bytes"}
             and isinstance(layer["job_id"], str) and re.fullmatch(r"[1-9][0-9]*", layer["job_id"])
             and type(layer["reserved_CPU_seconds"]) is int and layer["reserved_CPU_seconds"] == 86400
             and type(layer["durable_bytes"]) is int and layer["durable_bytes"] == 25_000_000_000,
             "exact_one_CPU_24h_25GB_included_layer_reservation_required")
    return {"cpu_seconds": cpu, "durable_bytes": durable, "included_layer_job": layer}


def durable_reservations(authority, manifest, batches, *, proposed=None):
    """Reserve exact recovery outputs once against the full immutable basis.

    The historical quarantine remains unchanged. Only the separate fixed
    recovery review reduces work62's original reservation; no file is deleted,
    no other unused reservation is released, and unresolved intents count too.
    Legacy histories keep their existing durable-admission behavior.
    """
    combined = dict(batches)
    if proposed is not None:
        combined["__prospective_IO_storage__"] = proposed
    deadline_bytes = deadline_remainders.reserved_durable_bytes(combined)
    gross = io_recovery.reserved_durable_bytes(combined)
    result = {"additional_remainder_durable_bytes": deadline_bytes,
              "additional_io_recovery_durable_bytes": 0,
              "io_recovery_gross_durable_bytes": gross,
              "retained_original_reservation_reduction_bytes": 0,
              "projected_durable_bytes": None}
    if not gross:
        return result
    reviews = [row[io_recovery.FIELD] for row in combined.values() if io_recovery.FIELD in row]
    review = reviews[0]  # reserved_durable_bytes already requires identical reviews.
    io_recovery.validate_review_structure(review, manifest)
    for row in combined.values():
        if deadline_remainders.FIELD in row:
            deadline_remainders.validate_plan_structure(row[deadline_remainders.FIELD], authority, manifest)
    original = (sum(item["resources"]["durable_bytes"] for item in manifest["work_items"])
        + authority["state_build_resources"]["durable_bytes"]
        + authority["assembly_resources"]["durable_bytes"] + JOURNAL_BYTES
        + authority["closed_included_layer_storage"]["charged_bytes"])
    need(original + deadline_bytes == review["storage_totals"]["original_and_existing_remainder_reserved_bytes"],
         "reviewed_IO_storage_must_match_full_manifest_and_existing_remainders")
    net = io_recovery.net_additional_durable_bytes(review)
    total = original + deadline_bytes + net
    need(total <= budget_definition(authority)["durable_bytes"] <= 500_000_000_000,
         "full_manifest_plus_IO_recovery_exceeds_durable_cap")
    result.update(additional_io_recovery_durable_bytes=net,
                  retained_original_reservation_reduction_bytes=gross-net,
                  projected_durable_bytes=total)
    return result


def _allocation(logical, job_id, cpu_seconds, *, phase, batch_id=None):
    return {"logical_id": logical, "job_id": job_id, "reserved_CPU_seconds": cpu_seconds,
            "phase": phase, "batch_id": batch_id, "terminal": None}


def allocations_for(authority, manifest, phase, selected=()):
    """Bind each full reservation to its immutable global manifest index."""
    items = manifest["work_items"]
    ids = [item["work_item_id"] for item in items]
    need(ids == [f"work_{i + 1:04d}" for i in range(len(ids))] and ids,
         "canonical_ordered_array_work_item_ids_required")
    if phase == "workers":
        need(selected and len(selected) == len(set(selected)) and set(selected) <= set(ids),
             "explicit_unique_manifest_work_items_required")
        return [{"logical_id": item["work_item_id"], "array_index": i,
                 "reserved_CPU_seconds": item["resources"]["cpus"] * item["resources"]["minutes"] * 60}
                for i, item in enumerate(items) if item["work_item_id"] in selected]
    need(phase in {"states", "assembly"} and not selected, "exact_finite_phase_required")
    cap = authority["state_build_resources" if phase == "states" else "assembly_resources"]
    return [{"logical_id": phase, "reserved_CPU_seconds": cap["cpus"] * cap["minutes"] * 60}]


def validate_replacement_review(review):
    """One coordinator-reviewed mechanical repair, not a general retry grant."""
    exact = {"schema_version": "ep21.reviewed_pre_admission_replacement.v1",
             "reviewer": "coordinating_agent", "predecessor_job_id": REPLACED_JOB,
             "predecessor_allocation_job_id": "46839670", "work_item_id": REPLACED_WORK,
             "array_index": 7, "failure_reason": "actual_work_item_allocation_mismatch",
             "shard_directory_absent": True, "request_or_status_created": False,
             "scientific_payload_admitted": False, "automatic_retry": False}
    text_fields = {"reviewed_utc", "fix_evidence", "terminal_evidence", "absence_evidence",
                   "user_authorization_context"}
    need(isinstance(review, dict) and set(review) == set(exact) | text_fields,
         "exact_dated_pre_admission_replacement_review_required")
    need(all(type(review[key]) is type(value) and review[key] == value for key, value in exact.items())
         and all(isinstance(review[key], str) and review[key].strip() for key in text_fields),
         "only_reviewed_46839670_7_pre_admission_failure_may_be_replaced")
    try:
        observed = datetime.fromisoformat(review["reviewed_utc"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise AccountingError("dated_timezone_bearing_replacement_review_required") from exc
    need(observed.utcoffset() is not None, "dated_timezone_bearing_replacement_review_required")


def validate_replacement_intent(row, jobs, batches):
    need(MANUAL_REFILL_FIELD not in row, "reviewed_replacement_cannot_use_manual_refill")
    validate_replacement_review(row.get(REPLACEMENT_FIELD))
    need(row.get("phase") == "workers" and row.get("allocations") == [
             {"logical_id": REPLACED_WORK, "array_index": 7, "reserved_CPU_seconds": 28800}],
         "replacement_must_preserve_exact_single_work_0008_assignment")
    predecessor = jobs.get(REPLACED_JOB)
    need(predecessor is not None and predecessor["phase"] == "workers"
         and predecessor["logical_id"] == REPLACED_WORK,
         "reviewed_predecessor_must_be_an_accepted_worker")
    terminal = predecessor["terminal"]
    expected = {"allocation_job_id": "46839670", "allocated_cpus": 1,
                "elapsed_seconds": 15, "state": "FAILED", "exit_code": "2:0"}
    need(terminal is not None and all(terminal.get(key) == value for key, value in expected.items()),
         "reviewed_predecessor_15_second_failure_must_already_be_charged")
    need(not any(REPLACEMENT_FIELD in batch for batch in batches.values())
         and [job["job_id"] for job in jobs.values() if job["logical_id"] == REPLACED_WORK] == [REPLACED_JOB],
         "one_manual_replacement_only_no_chaining_or_repeat")
    need(all(batch["accepted_id"] is not None and not batch["not_submitted"] for batch in batches.values()),
         "replacement_requires_reconciled_accepted_history")


def validate_startup_replacement_review(review):
    exact = {"schema_version": "ep21.reviewed_startup_replacement.v1",
             "reviewer": "coordinating_agent", "predecessor_job_id": STARTUP_REPLACED_JOB,
             "predecessor_allocation_job_id": "46880582", "work_item_id": STARTUP_REPLACED_WORK,
             "array_index": 12, "failure_reason": "required_deadline_partial_fixture_precondition",
             "shard_directory_absent": True, "request_or_status_created": False,
             "scientific_payload_admitted": False, "optional_children_started": False,
             "automatic_retry": False}
    text_fields = {"reviewed_utc", "fix_evidence", "terminal_evidence", "absence_evidence", "user_authorization_context"}
    need(isinstance(review, dict) and set(review) == set(exact) | text_fields
         and all(type(review[key]) is type(value) and review[key] == value for key, value in exact.items())
         and all(isinstance(review[key], str) and review[key].strip() for key in text_fields),
         "exact_reviewed_work_0013_startup_failure_required")
    try:
        observed = datetime.fromisoformat(review["reviewed_utc"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise AccountingError("dated_startup_replacement_review_required") from exc
    need(observed.utcoffset() is not None, "timezone_bearing_startup_replacement_review_required")


def validate_startup_replacement_intent(row, jobs, batches):
    validate_startup_replacement_review(row.get(STARTUP_REPLACEMENT_FIELD))
    need(row.get("phase") == "workers" and row.get(MANUAL_REFILL_FIELD) is True
         and REPLACEMENT_FIELD not in row and deadline_partials.FIELD in row
         and row.get("allocations") == [{"logical_id": STARTUP_REPLACED_WORK,
                                         "array_index": 12, "reserved_CPU_seconds": 28800}],
         "startup_replacement_must_preserve_single_work_0013_and_partial_review")
    predecessor = jobs.get(STARTUP_REPLACED_JOB)
    need(predecessor is not None and predecessor["phase"] == "workers"
         and predecessor["logical_id"] == STARTUP_REPLACED_WORK,
         "startup_predecessor_must_be_registered_worker")
    terminal = predecessor["terminal"]
    expected = {"allocation_job_id": "46880582", "allocated_cpus": 1,
                "elapsed_seconds": 12, "state": "FAILED", "exit_code": "1:0"}
    need(terminal is not None and all(terminal.get(key) == value for key, value in expected.items()),
         "startup_predecessor_12_second_failure_must_be_charged")
    need(not any(STARTUP_REPLACEMENT_FIELD in batch for batch in batches.values())
         and [job["job_id"] for job in jobs.values() if job["logical_id"] == STARTUP_REPLACED_WORK]
             == [STARTUP_REPLACED_JOB], "one_exact_startup_replacement_only_no_chaining")
    need(all(batch["accepted_id"] is not None and not batch["not_submitted"] for batch in batches.values()),
         "startup_replacement_requires_reconciled_accepted_history")


def validate_manual_refill_intent(row, jobs, batches, authority, *, manifest=None, effective_parallelism=None):
    """Bound one operator-selected fresh array across all outstanding arrays.

    Scheduler absence does not release a slot or its full CPU reservation.
    Scientific closeout and the single queue observation belong to the caller.
    """
    need(row.get(MANUAL_REFILL_FIELD) is True and row.get("phase") == "workers"
         and REPLACEMENT_FIELD not in row, "manual_refill_only_for_ordinary_workers")
    need(all(batch["accepted_id"] is not None and not batch["not_submitted"] for batch in batches.values()),
         "manual_refill_requires_reconciled_accepted_history")
    states = [job for job in jobs.values() if job["phase"] == "states"]
    need(len(states) == 1 and states[0]["terminal"] is not None
         and states[0]["terminal"]["state"] == "COMPLETED"
         and states[0]["terminal"]["exit_code"] == "0:0"
         and states[0]["terminal"]["allocated_cpus"] == 1
         and not any(job["phase"] == "assembly" for job in jobs.values()),
         "manual_refill_requires_completed_states_and_no_assembly")
    need(any(job["phase"] == "workers" and job["terminal"] is not None
             and job["terminal"]["state"] == "COMPLETED" and job["terminal"]["exit_code"] == "0:0"
             and job["terminal"]["allocated_cpus"] == 1 for job in jobs.values()),
         "manual_refill_requires_a_successfully_closed_worker")
    outstanding = {key: job for key, job in jobs.items() if job["terminal"] is None}
    need(all(job["phase"] == "workers" and job["batch_id"] in batches
             and batches[job["batch_id"]]["accepted_id"] is not None
             for job in outstanding.values()), "manual_refill_outstanding_jobs_must_be_known_workers")
    if effective_parallelism is None:
        maximum = authority["resources"]["maximum_concurrent_jobs"]
        need(type(maximum) is int and maximum > 0
             and len(outstanding) + len(row["allocations"]) <= min(8, maximum),
             "manual_refill_outstanding_plus_new_exceeds_worker_cap")
    else:
        need(manifest is not None, "amended_parallelism_requires_bound_manifest")
        parallelism.validate_reservation(row, jobs, batches, manifest, effective_parallelism)
    return outstanding


def validate_remainder_assembly_intent(row, jobs, batches, authority, manifest):
    """Assembly joins every reviewed remainder, after all fitting is charged."""
    need(row.get("phase") == "assembly" and not any(field in row for field in
         (MANUAL_REFILL_FIELD, REPLACEMENT_FIELD, STARTUP_REPLACEMENT_FIELD, deadline_partials.FIELD)),
         "remainder_assembly_requires_its_own_explicit_final_intent")
    if io_recovery.FIELD in row:
        io_recovery.validate_intent(row, jobs, batches, manifest)
    else:
        need(not any(io_recovery.FIELD in prior for prior in batches.values())
             and not any(job_id in jobs for job_id, _, _ in io_quarantine.KNOWN.values()),
             "remainder_assembly_requires_explicit_IO_recovery_for_incident_parents")
    plan = row[deadline_remainders.FIELD]
    items = deadline_remainders.validate_plan_structure(plan, authority, manifest)
    previous = [batch[deadline_remainders.FIELD] for batch in batches.values()
                if deadline_remainders.FIELD in batch]
    need(previous and all(deadline_remainders.plans_compatible(value, plan) for value in previous),
         "published_remainder_plan_is_immutable")
    need(all(batch["accepted_id"] is not None and not batch["not_submitted"] for batch in batches.values())
         and all(job["terminal"] is not None for job in jobs.values()),
         "remainder_assembly_requires_all_registered_jobs_charged")
    review = deadline_remainders.latest_published_review({"batches": batches})
    reviewed = deadline_partials.validate_intent({"phase": "workers", MANUAL_REFILL_FIELD: True,
        "allocations": [], deadline_partials.FIELD: review}, jobs, manifest)
    need(set(reviewed) == {items[work]["predecessor_job_id"] for work in items},
         "remainder_assembly_requires_every_reviewed_predecessor")
    for work in items:
        deadline_remainders.registered_remainder_job({"jobs": jobs, "batches": batches}, plan, work,
                                                    require_terminal=True)


def replay(events, authority, manifest=None):
    """Reconstruct exact accepted jobs and unresolved full reservations."""
    budget = budget_definition(authority)
    effective_parallelism = None
    need(events and events[0].get("event") == "budget_opened"
         and events[0].get("budget") == budget, "immutable_budget_basis_mismatch")
    jobs, batches, assigned, basis = {}, {}, set(), None
    layer = budget["included_layer_job"]
    if layer:
        jobs[layer["job_id"]] = _allocation("included_layer", layer["job_id"],
                layer["reserved_CPU_seconds"], phase="included_layer")
    for row in events[1:]:
        kind = row.get("event")
        if kind == "manifest_bound":
            need(basis is None and not batches, "manifest_can_be_bound_only_once_before_submission")
            basis = row
            need(row.get("authority") == authority, "submitted_authority_is_immutable")
        elif kind == parallelism.EVENT:
            need(basis is not None and effective_parallelism is None,
                 "one_parallelism_amendment_after_immutable_manifest_only")
            effective_parallelism = parallelism.validate_event(row, authority, basis["manifest"], jobs, batches)
            charged = sum(job["terminal"]["allocated_cpus"] * job["terminal"]["elapsed_seconds"]
                          for job in jobs.values() if job["terminal"] is not None)
            reserved = sum(job["reserved_CPU_seconds"] for job in jobs.values() if job["terminal"] is None)
            need(charged + reserved <= budget["cpu_seconds"], "parallelism_cannot_reset_or_exceed_CPU_cap")
        elif kind == "submission_intent":
            need(basis is not None, "full_manifest_required_before_intent")
            need("effective_parallelism" not in row, "intent_parallelism_policy_is_derived_from_prior_event")
            io_quarantine.validate_intent(row, jobs, basis["manifest"], batches)
            bundle_startup.validate_intent(row, jobs, batches)
            component_replacement = component_startup.validate_intent(row, jobs, batches)
            identifier = row["batch_id"]
            need(identifier == f"batch_{len(batches) + 1:04d}" and identifier not in batches,
                 "unique_sequential_batch_id_required")
            units = row["allocations"]
            need(isinstance(units, list) and units and row["phase"] in {"states", "workers", "assembly"},
                 "nonempty_finite_batch_required")
            selected = [item["logical_id"] for item in units] if row["phase"] == "workers" else ()
            if remainder_bundle.FIELD in row:
                remainder_bundle.validate_bundle(row)
            else:
                need(units == allocations_for(authority, basis["manifest"], row["phase"], selected),
                     "intent_reservations_must_match_immutable_manifest")
            replacement = REPLACEMENT_FIELD in row
            if replacement:
                validate_replacement_intent(row, jobs, batches)
            startup_replacement = STARTUP_REPLACEMENT_FIELD in row
            if startup_replacement:
                validate_startup_replacement_intent(row, jobs, batches)
            if effective_parallelism is not None and row["phase"] == "workers":
                need(row.get(MANUAL_REFILL_FIELD) is True, "amended_workers_require_manual_refill")
            if MANUAL_REFILL_FIELD in row:
                validate_manual_refill_intent(row, jobs, batches, authority,
                    manifest=basis["manifest"], effective_parallelism=effective_parallelism)
            if effective_parallelism is not None and row["phase"] == "workers":
                charged = sum(job["terminal"]["allocated_cpus"] * job["terminal"]["elapsed_seconds"]
                              for job in jobs.values() if job["terminal"] is not None)
                reserved = sum(job["reserved_CPU_seconds"] for job in jobs.values() if job["terminal"] is None)
                need(charged + reserved + sum(item["reserved_CPU_seconds"] for item in units)
                     <= budget["cpu_seconds"], "parallelism_actual_plus_reserved_exceeds_CPU_cap")
            recovery = io_recovery.FIELD in row
            if recovery:
                # Quarantine validation has already checked the exact review,
                # original failed charges, and first-only fresh assignments.
                need(effective_parallelism is not None,
                     "IO_recovery_requires_published_parallelism_amendment")
                if row["phase"] == "assembly":
                    need(deadline_remainders.FIELD in row,
                         "IO_recovery_assembly_requires_existing_deadline_plan")
                charged = sum(job["terminal"]["allocated_cpus"] * job["terminal"]["elapsed_seconds"]
                              for job in jobs.values() if job["terminal"] is not None)
                reserved = sum(job["reserved_CPU_seconds"] for job in jobs.values() if job["terminal"] is None)
                need(charged + reserved + sum(item["reserved_CPU_seconds"] for item in units)
                     <= budget["cpu_seconds"], "IO_recovery_actual_plus_reserved_exceeds_CPU_cap")
            durable_reservations(authority, basis["manifest"], batches, proposed=row)
            remainder = deadline_remainders.FIELD in row
            if remainder:
                need(not replacement and not startup_replacement, "remainder_cannot_be_a_startup_replacement")
                if row["phase"] == "workers":
                    deadline_remainders.validate_remainder_intent(row, jobs, batches, authority, basis["manifest"])
                else:
                    validate_remainder_assembly_intent(row, jobs, batches, authority, basis["manifest"])
                # Additional physical attempts retain every original charge and
                # reserve their full new request before scheduler acceptance.
                charged = sum(job["terminal"]["allocated_cpus"] * job["terminal"]["elapsed_seconds"]
                              for job in jobs.values() if job["terminal"] is not None)
                reserved = sum(job["reserved_CPU_seconds"] for job in jobs.values() if job["terminal"] is None)
                need(charged + reserved + sum(item["reserved_CPU_seconds"] for item in units)
                     <= budget["cpu_seconds"], "remainder_actual_plus_reserved_exceeds_CPU_cap")
            if deadline_partials.FIELD in row:
                deadline_partials.validate_intent(row, jobs, basis["manifest"])
                prior_reviews = [other[deadline_partials.FIELD] for other in batches.values()
                                 if deadline_partials.FIELD in other]
                current_review = row[deadline_partials.FIELD]
                if prior_reviews and current_review != prior_reviews[-1]:
                    previous = prior_reviews[-1]
                    fixed = set(previous) - {"reviewed_utc", "partials"}
                    need(all(current_review[key] == previous[key] for key in fixed)
                         and len(current_review["partials"]) > len(previous["partials"])
                         and current_review["partials"][:len(previous["partials"])] == previous["partials"]
                         and datetime.fromisoformat(current_review["reviewed_utc"].replace("Z", "+00:00"))
                             > datetime.fromisoformat(previous["reviewed_utc"].replace("Z", "+00:00")),
                         "published_deadline_partial_review_is_immutable_except_newer_appended_partials")
            for item in units:
                need((item["logical_id"] not in assigned or replacement or startup_replacement or component_replacement
                      or (remainder or recovery) and row["phase"] == "workers")
                     and type(item["reserved_CPU_seconds"]) is int
                     and item["reserved_CPU_seconds"] > 0, "work_or_phase_cannot_be_submitted_twice")
                assigned.add(item["logical_id"])
            batches[identifier] = {**row, "accepted_id": None, "not_submitted": False}
            if effective_parallelism is not None:
                batches[identifier]["effective_parallelism"] = dict(effective_parallelism)
        elif kind == "submission_accepted":
            batch = batches[row["batch_id"]]
            job = row["job_id"]
            need(batch["accepted_id"] is None and not batch["not_submitted"]
                 and isinstance(job, str) and re.fullmatch(r"[1-9][0-9]*", job),
                 "one_explicit_scheduler_acceptance_required")
            need(all(other["accepted_id"] != job for other in batches.values())
                 and (layer is None or layer["job_id"] != job), "duplicate_scheduler_parent_identity")
            batch["accepted_id"] = job
            for item in batch["allocations"]:
                physical = f"{job}_{item['array_index']}" if batch["phase"] == "workers" else job
                need(physical not in jobs, "duplicate_scheduler_parent_identity")
                jobs[physical] = _allocation(item["logical_id"], physical, item["reserved_CPU_seconds"],
                                            phase=batch["phase"], batch_id=row["batch_id"])
        elif kind == "submission_uncertain":
            need(row["batch_id"] in batches, "uncertainty_requires_recorded_intent")
        elif kind == "submission_not_accepted":
            batch = batches[row["batch_id"]]
            need(batch["accepted_id"] is None and not batch["not_submitted"] and row.get("evidence"),
                 "explicit_nonacceptance_reconciliation_required")
            batch["not_submitted"] = True
        elif kind == "terminal_charge":
            job = jobs.get(row.get("job_id"))
            need(job is not None and job["terminal"] is None, "terminal_parent_must_be_known_and_charged_once")
            cpus, elapsed = row.get("allocated_cpus"), row.get("elapsed_seconds")
            need(type(cpus) is int and cpus in {0, 1} and type(elapsed) is int and elapsed >= 0
                 and (cpus == 1 or elapsed == 0) and row.get("state") in TERMINAL
                 and isinstance(row.get("exit_code"), str) and re.fullmatch(r"[0-9]+:[0-9]+", row["exit_code"])
                 and isinstance(row.get("allocation_job_id"), str)
                 and re.fullmatch(r"[1-9][0-9]*", row["allocation_job_id"])
                 and isinstance(row.get("observed_utc"), str) and row["observed_utc"] and row.get("evidence"),
                 "explicit_terminal_scheduler_parent_accounting_required")
            need("_" in job["job_id"] or row["allocation_job_id"] == job["job_id"],
                 "terminal_allocation_identity_mismatch")
            need(all(other["terminal"] is None or
                     other["terminal"]["allocation_job_id"] != row["allocation_job_id"]
                     for other in jobs.values()), "one_physical_allocation_cannot_be_charged_twice")
            job["terminal"] = row
        else:
            raise AccountingError("unrecognized_or_incomplete_accounting_event")
    if basis is not None:
        need(basis["authority"] == authority, "submitted_authority_is_immutable")
        if manifest is not None:
            need(basis["manifest"] == manifest, "submitted_manifest_is_immutable")
    unknown = [b for b in batches.values() if b["accepted_id"] is None and not b["not_submitted"]]
    actual = sum(j["terminal"]["allocated_cpus"] * j["terminal"]["elapsed_seconds"]
                 for j in jobs.values() if j["terminal"] is not None)
    reserved = sum(j["reserved_CPU_seconds"] for j in jobs.values() if j["terminal"] is None)
    reserved += sum(item["reserved_CPU_seconds"] for b in unknown for item in b["allocations"])
    # This changes readiness only after a new physical allocation is accepted.
    # Both physical attempts remain in jobs and every terminal charge is kept.
    superseded = {REPLACED_JOB for b in batches.values()
                  if REPLACEMENT_FIELD in b and b["accepted_id"] is not None}
    superseded.update(STARTUP_REPLACED_JOB for b in batches.values()
                      if STARTUP_REPLACEMENT_FIELD in b and b["accepted_id"] is not None)
    superseded.update(bundle_startup.superseded_jobs(batches))
    superseded.update(component_startup.JOB for b in batches.values()
                      if component_startup.FIELD in b and b["accepted_id"] is not None)
    return {"budget": budget, "jobs": jobs, "batches": batches, "assigned": assigned,
            "basis": basis, "uncertain_batches": unknown, "actual_CPU_seconds": actual,
            "effective_parallelism": effective_parallelism,
            "outstanding_reserved_CPU_seconds": reserved,
            "superseded_pre_admission_jobs": superseded,
            **durable_reservations(authority, basis["manifest"] if basis else manifest, batches),
            "rejected_batch": any(b["not_submitted"] for b in batches.values())}


def admit_reservation(account, requested_seconds):
    need(not account["uncertain_batches"], "uncertain_submission_stops_further_dispatch")
    need(not account["rejected_batch"], "rejected_attempt_requires_separate_review_no_automatic_retry")
    need(type(requested_seconds) is int and requested_seconds > 0, "positive_new_reservation_required")
    need(account["actual_CPU_seconds"] + account["outstanding_reserved_CPU_seconds"] + requested_seconds
         <= account["budget"]["cpu_seconds"], "actual_plus_outstanding_plus_new_exceeds_CPU_cap")


def append_parallelism_amendment(rows, append, authority, manifest, *, observed_utc, user_instruction):
    """Caller holds the exclusive journal lock; validate before appending once."""
    event = parallelism.make_event(observed_utc=observed_utc, user_instruction=user_instruction)
    account = replay([*rows, event], authority, manifest)
    append(event)
    return account


@contextmanager
def locked_events(path, *, create=False, shared=False):
    """Lock one append-only journal; torn/invalid history stops, never repairs."""
    path = Path(path)
    need(not path.is_symlink(), "accounting_journal_must_not_be_symlink")
    flags = os.O_RDONLY if shared else os.O_RDWR | os.O_APPEND
    if create:
        flags |= os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "r" if shared else "a+", encoding="utf-8") as stream:
        # A worker may start while sbatch is returning. One blocking read lock
        # waits for its 30-second response and acceptance fsync, at most 45s.
        # This is one kernel wait, not a filesystem/scheduler polling loop.
        if shared:
            need(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "shared_lock_requires_no_existing_alarm")
            previous = signal.getsignal(signal.SIGALRM)
            def timeout(_signum, _frame):
                raise AccountingError("submission_journal_lock_wait_exceeded_45_seconds")
            signal.signal(signal.SIGALRM, timeout)
            signal.setitimer(signal.ITIMER_REAL, 45)
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        need(os.fstat(stream.fileno()).st_size <= JOURNAL_BYTES, "accounting_journal_byte_cap")
        stream.seek(0)
        raw = stream.read()
        need(not raw or raw.endswith("\n"), "incomplete_accounting_record_requires_reconciliation")
        rows = [json.loads(line) for line in raw.splitlines()]
        def append(row):
            need(not shared, "read_only_accounting_view")
            event = {"utc": datetime.now(timezone.utc).isoformat(), **row}
            encoded = json.dumps(event, sort_keys=True) + "\n"
            need(os.fstat(stream.fileno()).st_size + len(encoded.encode("utf-8")) <= JOURNAL_BYTES,
                 "accounting_journal_byte_cap")
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            rows.append(event)
        yield rows, append
