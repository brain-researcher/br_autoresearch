"""Operator-driven EP21 finite batches; no approval, payload reads or polling.

The fixed authority and complete work manifest become immutable at the first
submission. One append-only journal records budget, intent, acceptance and
operator-observed terminal allocation charges. Uncertain scheduler acceptance
stops new submissions. A failed or partial parent requires separate review; this
helper never automatically retries, cancels, extends, or edits assignments.
One explicit reviewed pre-admission repair is supported for task 46839670_7.
An explicit manual worker refill may coexist with registered ongoing workers;
it requires exact terminal results and a reconciled queue within the global cap.
The fixed five-parent Oak incident may be explicitly quarantined for independent
manual fitting, without crediting coverage or authorizing reuse of those files.
Each invocation
submits at most one state parent, one finite fit array, or one assembly parent.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess

import decoder_batch_accounting as accounting
import decoder_execution_contract as contract
import participant_training_state as state
import reviewed_deadline_partials as deadline_partials
import decoder_deadline_remainders as deadline_remainders
import decoder_remainder_bundle as remainder_bundle
import reviewed_io_quarantine as io_quarantine
import reviewed_bundle_startup_replacement as bundle_startup
import reviewed_component_startup_replacement as component_startup
import decoder_parallelism_amendment as parallelism
import reviewed_io_recovery as io_recovery

require, Stop = contract.require, contract.Stop
BASE = contract.BASE
JOURNAL = contract.ROOT / "submission.jsonl"
BATCH_ROOT = BASE / "outputs/decoder_training"
REPLACEMENT_REVIEW = contract.ROOT / "reviewed_pre_admission_replacement_001.json"
REPLACEMENT_WRAPPER = BATCH_ROOT / "full_workers_admission_repair_001.sbatch"
REFILL_WRAPPER = BATCH_ROOT / "full_workers_refill_001.sbatch"
PARALLEL_WRAPPER = BATCH_ROOT / "full_workers_parallel_001.sbatch"
COMPONENT_WRAPPER = BATCH_ROOT / "full_workers_parallel_002.sbatch"
COMPONENT_REPAIR_WRAPPER = BATCH_ROOT / "full_workers_parallel_003.sbatch"
COMPONENT_STARTUP_REVIEW = BASE / component_startup.REVIEW_RELATIVE
IO_REMAINDER_WRAPPER = BATCH_ROOT / "full_io_remainders_001.sbatch"
IO_ASSEMBLY_WRAPPER = BATCH_ROOT / "full_assembly_io_recovery_001.sbatch"
DEADLINE_PARTIAL_REVIEW = contract.ROOT / "reviewed_deadline_partials_001.json"
DEADLINE_PARTIAL_REVIEW_002 = contract.ROOT / "reviewed_deadline_partials_002.json"
STARTUP_REPLACEMENT_REVIEW = contract.ROOT / "reviewed_startup_replacement_001.json"
REMAINDER_PLAN = BASE / deadline_remainders.PLAN_RELATIVE
REMAINDER_PLAN_002 = BASE / deadline_remainders.PLAN_RELATIVE_002
REMAINDER_WRAPPER = BATCH_ROOT / "full_remainders_001.sbatch"
WORK5_REMAINDER_WRAPPER = BATCH_ROOT / "full_remainder_work5_001.sbatch"
REMAINDER_BUNDLE_WRAPPER = BATCH_ROOT / "full_remainders_002_bundle.sbatch"
REMAINDER_BUNDLE_WRAPPER_003 = BATCH_ROOT / "full_remainders_003_bundle.sbatch"
REMAINDER_ASSEMBLY_WRAPPER = BATCH_ROOT / "full_assembly_remainders_001.sbatch"
REMAINDER_ASSEMBLY_WRAPPER_002 = BATCH_ROOT / "full_assembly_remainders_002.sbatch"
IO_QUARANTINE_REVIEW = BASE / io_quarantine.REVIEW_RELATIVE
IO_SALVAGE_REVIEW = BASE / io_recovery.REVIEW_RELATIVE
IO_RECOVERY_WRAPPER = BATCH_ROOT / "full_workers_io_recovery_001.sbatch"
BUNDLE_STARTUP_REVIEW = BASE / bundle_startup.REVIEW_RELATIVE
BUNDLE_STARTUP_REVIEW_002 = BASE / bundle_startup.REVIEW_RELATIVE_002
BUNDLE_STARTUP_REVIEW_003 = BASE / bundle_startup.REVIEW_RELATIVE_003
WORKER_CPU_CONSTRAINT = "CPU_GEN:SIE"  # Coordinator verified with sh_node_feat.
REFILL_LIVE_STATES = frozenset({"PENDING", "CONFIGURING", "RUNNING", "COMPLETING"})


def build_batch(authority, manifest, phase, selected, partitions, *, constraint=None, manual_refill=False,
                deadline_partial_review=None, effective_parallelism=None):
    require(isinstance(partitions, str) and re.fullmatch(r"[a-zA-Z0-9_-]+(?:,[a-zA-Z0-9_-]+)*", partitions),
            "explicit_discovered_partition_list_required")
    require("owners" not in partitions.split(","), "preemptible_owners_not_in_this_proposal")
    require(constraint is None or constraint == WORKER_CPU_CONSTRAINT,
            "only_verified_CPU_GEN_SIE_constraint_supported")
    require(constraint is None or phase == "workers", "CPU_constraint_only_for_ordinary_workers")
    require(type(manual_refill) is bool and (not manual_refill or phase == "workers"),
            "manual_refill_only_for_ordinary_workers")
    require(deadline_partial_review is None or manual_refill and phase == "workers",
            "deadline_partial_review_only_for_fresh_manual_workers")
    allocations = accounting.allocations_for(authority, manifest, phase, selected)
    items = {item["work_item_id"]: item for item in manifest["work_items"]}
    caps = ([items[item["logical_id"]]["resources"] for item in allocations] if phase == "workers" else
            [authority["state_build_resources" if phase == "states" else "assembly_resources"]])
    minutes = 480 if phase == "workers" else 240
    require(all(cap["cpus"] == 1 and cap["memory_gib"] == 7 and cap["gpus"] == 0
                and cap["minutes"] == minutes for cap in caps), "fixed_wrapper_limits_must_match_every_authorized_job")
    stem = f"full_{phase}001-" + ("%A_%a" if phase == "workers" else "%j")
    command = ["sbatch", "--parsable", f"--partition={partitions}", "--nodes=1", "--ntasks=1",
               "--cpus-per-task=1", "--mem=7G", f"--time={minutes}", "--no-requeue",
               f"--chdir={BASE}", "--export=ALL", f"--job-name=ep21-full-{phase}-001",
               f"--output=outputs/decoder_training/{stem}.out", f"--error=outputs/decoder_training/{stem}.err"]
    if phase == "workers":
        if constraint is not None:
            command.append(f"--constraint={constraint}")
        concurrency = min(parallelism.effective_maximum(authority, effective_parallelism), len(allocations))
        require(type(concurrency) is int and concurrency > 0, "positive_authorized_array_concurrency_required")
        command.append("--array=" + ",".join(str(item["array_index"]) for item in allocations) + f"%{concurrency}")
    wrapper = REFILL_WRAPPER if manual_refill else BATCH_ROOT / f"full_{phase}_001.sbatch"
    if effective_parallelism is not None:
        require(manual_refill and phase == "workers", "parallelism_wrapper_requires_fresh_manual_workers")
        wrapper = PARALLEL_WRAPPER
    command.append(str(wrapper))
    batch = {"phase": phase, "allocations": allocations, "command": command}
    if manual_refill:
        batch[accounting.MANUAL_REFILL_FIELD] = True
    if deadline_partial_review is not None:
        batch[deadline_partials.FIELD] = deadline_partial_review
    return batch


def require_no_live_episode_job(runner):
    result = runner(["squeue", "--me", "--noheader", "--format=%i|%j|%T"],
                    capture_output=True, text=True, timeout=30, check=True)
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("|")]
        require(len(fields) == 3, "unrecognized_queue_observation")
        if fields[1].lower().startswith("ep21"):
            raise Stop(f"existing_EP21_job_requires_reconciliation:{fields[0]}:{fields[2]}")


def require_reviewed_replacement_queue(account, authority, runner):
    """One expanded observation; allow only the seven original other writers."""
    outstanding = {key for key, job in account["jobs"].items() if job["terminal"] is None}
    maximum = authority["resources"]["maximum_concurrent_jobs"]
    require(outstanding <= accounting.COEXISTING_WORKERS
            and len(outstanding) + 1 <= min(8, maximum), "replacement_would_exceed_known_writer_envelope")
    result = runner(["squeue", "--me", "--array", "--noheader", "--format=%i|%j|%T"],
                    capture_output=True, text=True, timeout=30, check=True)
    seen = set()
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("|")]
        require(len(fields) == 3, "unrecognized_queue_observation")
        job_id, name, _ = fields
        if name.lower().startswith("ep21") or job_id in accounting.COEXISTING_WORKERS:
            require(job_id in outstanding and job_id not in seen and name == "ep21-full-workers-001",
                    "replacement_queue_contains_unreviewed_or_duplicate_EP21_writer")
            seen.add(job_id)


def require_manual_refill_queue(account, authority, batch, runner):
    """Exactly reconcile every reserved worker in one expanded observation."""
    outstanding = accounting.validate_manual_refill_intent(batch, account["jobs"], account["batches"], authority,
        manifest=account["basis"]["manifest"], effective_parallelism=account.get("effective_parallelism"))
    result = runner(["squeue", "--me", "--array", "--noheader", "--format=%i|%j|%T"],
                    capture_output=True, text=True, timeout=30, check=True)
    seen = set()
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("|")]
        require(len(fields) == 3, "unrecognized_queue_observation")
        job_id, name, live_state = fields
        if name.lower().startswith("ep21") or job_id in account["jobs"]:
            require(job_id in outstanding and job_id not in seen
                    and name == "ep21-full-workers-001" and live_state in REFILL_LIVE_STATES,
                    "manual_refill_queue_contains_unregistered_duplicate_or_unready_EP21_job")
            seen.add(job_id)
    require(seen == set(outstanding), "manual_refill_missing_reserved_worker_requires_terminal_reconciliation")


def require_previous_results(account, manifest, *, reader=contract.prefix.read_json, reviewed_partials=None,
                             reviewed_quarantines=None):
    """Terminal scheduler success alone cannot imply successful scientific work."""
    items = {item["work_item_id"]: item for item in manifest["work_items"]}
    for job in account["jobs"].values():
        if job["job_id"] in account.get("superseded_pre_admission_jobs", set()):
            continue
        terminal = job["terminal"]
        require(terminal is not None, "accepted_parent_terminal_accounting_missing")
        if job["phase"] == "included_layer":
            # Exact complete/partial layer validity and resume cause are already
            # checked by validate_work_manifest, before any batch admission.
            continue
        if job["job_id"] in (reviewed_quarantines or {}):
            entry = reviewed_quarantines[job["job_id"]]
            path = BASE / entry["status_path"]
            require(path.is_file() and not path.is_symlink(), "quarantine_closed_status_required")
            item = items[job["logical_id"]]
            io_quarantine.validate_closed_record(entry, reader(path), item,
                authorization_path=str(state.FULL_AUTH), manifest_path=str(contract.MANIFEST),
                shared_state_path=str(state.state_directory(item["participant"]) / "shared_training_state.npz"))
            continue  # Original assignment remains incomplete; no fragment read or reuse.
        registered_batch = account["batches"].get(job["batch_id"], {})
        if job["phase"] == "workers" and io_recovery.FIELD in registered_batch:
            review = registered_batch[io_recovery.FIELD]
            item = io_recovery.validate_review_structure(review, manifest)[job["logical_id"]]
            registered = io_recovery.registered_remainder_job(account, review, job["logical_id"],
                                                               require_terminal=True)
            path = BASE / item["output_directory"] / "work_item_status.json"
            require(path.is_file() and not path.is_symlink() and (path.parent / "channels.h5").is_file()
                    and not (path.parent / "channels.h5").is_symlink(),
                    "closed_IO_remainder_status_and_fragment_required")
            io_recovery.validate_completed_remainder(item, reader(path), registered, base_directory=str(BASE))
            continue
        if job["phase"] == "workers" and deadline_remainders.FIELD in registered_batch:
            plan = registered_batch[deadline_remainders.FIELD]
            remainder_items = deadline_remainders.validate_plan_structure(plan, account["basis"]["authority"], manifest)
            members = remainder_bundle.work_ids_for_job(job, account["batches"])
            for work in members:
                item = remainder_items[work]
                registered = deadline_remainders.registered_remainder_job(account, plan, work, require_terminal=True)
                path = BASE / item["output_directory"] / "work_item_status.json"
                require(path.is_file() and not path.is_symlink() and (path.parent / "channels.h5").is_file()
                        and not (path.parent / "channels.h5").is_symlink(), "closed_remainder_status_and_fragment_required")
                record = reader(path)
                deadline_remainders.validate_completed_remainder(item, record, registered, base_directory=str(BASE),
                    remainder_plan_relative=deadline_remainders.plan_relative(plan), registered_work_items=members)
                if registered_batch.get(io_quarantine.CLEAN_WORK5_FIELD) is True:
                    require(record.get(io_quarantine.CLEAN_WORK5_FIELD) is True,
                            "clean_work5_remainder_record_required")
                if registered_batch.get("effective_parallelism") is not None and remainder_bundle.FIELD not in registered_batch:
                    require(record.get("effective_scratch_bytes") == parallelism.work_scratch_bytes(manifest, [work])
                            and record.get("parallelism_amendment_id") == parallelism.EVENT,
                            "amended_remainder_effective_scratch_record_required")
            continue
        if job["job_id"] in (reviewed_partials or {}):
            entry = reviewed_partials[job["job_id"]]
            path = BASE / entry["status_path"]
            require(path.is_file() and not path.is_symlink()
                    and (path.parent / "channels.h5").is_file()
                    and not (path.parent / "channels.h5").is_symlink(),
                    "reviewed_partial_closed_status_and_fragment_required")
            item = items[job["logical_id"]]
            deadline_partials.validate_closed_record(entry, reader(path), item,
                authorization_path=str(state.FULL_AUTH), manifest_path=str(contract.MANIFEST),
                shared_state_path=str(state.state_directory(item["participant"]) / "shared_training_state.npz"))
            continue  # Remains assigned and incomplete; no replay or assembly permission.
        require(terminal["state"] == "COMPLETED" and terminal["exit_code"] == "0:0"
                and terminal["allocated_cpus"] == 1, "failed_parent_requires_review_no_automatic_successor")
        if job["phase"] == "states":
            path = contract.ROOT / "state_preparation_001/state_parent_status.json"
            expected = "complete_four_participant_training_states_only"
        elif job["phase"] == "workers":
            path = contract.ROOT / "shards" / job["logical_id"] / "work_item_status.json"
            expected = "complete_work_item_only"
        else:
            path = contract.ROOT / "assembled_decoder_manifest.json"
            expected = "complete_fixed_decoder_collection"
        require(path.is_file() and not path.is_symlink(), "accepted_task_result_missing")
        record = reader(path)
        allocation = record.get("assembly_allocation" if job["phase"] == "assembly" else "allocation", {})
        require(record.get("status") == expected
                and allocation.get("job_id") == terminal["allocation_job_id"],
                "completed_task_result_or_allocation_mismatch")
        if job["phase"] == "states":
            require(record.get("participants_completed") == list(state.NEW_PARTICIPANTS)
                    and record.get("participants_remaining") == [], "incomplete_participant_states")
        elif job["phase"] == "workers":
            item = items[job["logical_id"]]
            require(record.get("work_item_id") == item["work_item_id"]
                    and record.get("participant") == item["participant"]
                    and record.get("resources") == item["resources"]
                    and record.get("requested_spans") == item["spans"]
                    and record.get("committed_spans") == item["spans"]
                    and record.get("units_requested") == record.get("completed_units") == item["units_requested"]
                    and record.get("remaining_spans") == []
                    and record.get("completed_units_is_lower_bound") is False
                    and "uncertain_publication" not in record
                    and not any(error.get("affects_fragment") is not False
                                for error in record.get("closeout_errors", [{"affects_fragment": True}])),
                    "partial_or_uncertain_work_requires_separate_review")
            if registered_batch.get("effective_parallelism") is not None:
                require(record.get("effective_scratch_bytes")
                        == parallelism.work_scratch_bytes(manifest, [item["work_item_id"]])
                        and record.get("parallelism_amendment_id") == parallelism.EVENT,
                        "amended_worker_effective_scratch_record_required")


def build_reviewed_replacement(authority, manifest, review, partitions):
    accounting.validate_replacement_review(review)
    batch = build_batch(authority, manifest, "workers", [accounting.REPLACED_WORK], partitions)
    batch["command"][-1] = str(REPLACEMENT_WRAPPER)
    batch[accounting.REPLACEMENT_FIELD] = review
    return batch


def build_startup_replacement(authority, manifest, review, partial_review, partitions, *, constraint=None):
    accounting.validate_startup_replacement_review(review)
    batch = build_batch(authority, manifest, "workers", [accounting.STARTUP_REPLACED_WORK], partitions,
                        constraint=constraint, manual_refill=True, deadline_partial_review=partial_review)
    batch[accounting.STARTUP_REPLACEMENT_FIELD] = review
    return batch


def build_deadline_remainders(authority, manifest, plan, selected, partitions, *, constraint=None,
                             clean_work5=False):
    items = deadline_remainders.validate_plan_structure(plan, authority, manifest)
    require(selected and set(selected) <= set(items),
            "explicit_known_deadline_remainder_selection_required")
    require(type(clean_work5) is bool, "explicit_clean_work5_route_required")
    require(deadline_remainders.plan_relative(plan) == deadline_remainders.PLAN_RELATIVE
            if not clean_work5 else tuple(selected) == ("work_0005",)
            and deadline_remainders.plan_relative(plan) == deadline_remainders.PLAN_RELATIVE_002,
            "supplemental_remainder_requires_explicit_substantive_bundle")
    batch = build_batch(authority, manifest, "workers", selected, partitions,
                        constraint=constraint, manual_refill=True)
    batch["command"][-1] = str(WORK5_REMAINDER_WRAPPER if clean_work5 else REMAINDER_WRAPPER)
    batch[deadline_remainders.FIELD] = plan
    if clean_work5:
        batch[io_quarantine.CLEAN_WORK5_FIELD] = True
        batch["command"] = [
            "--output=outputs/decoder_training/full_remainder_work5_001-%A_%a.out" if value.startswith("--output=") else
            "--error=outputs/decoder_training/full_remainder_work5_001-%A_%a.err" if value.startswith("--error=") else value
            for value in batch["command"]]
        io_quarantine.validate_clean_work5_remainder(batch)
    return batch


def build_remainder_assembly(authority, manifest, plan, partitions):
    deadline_remainders.validate_plan_structure(plan, authority, manifest)
    batch = build_batch(authority, manifest, "assembly", (), partitions)
    batch["command"][-1] = str(REMAINDER_ASSEMBLY_WRAPPER if deadline_remainders.plan_relative(plan)
                               == deadline_remainders.PLAN_RELATIVE else REMAINDER_ASSEMBLY_WRAPPER_002)
    batch[deadline_remainders.FIELD] = plan
    return batch


def build_remainder_bundle(authority, manifest, plan, partitions, *, constraint=None):
    """Exactly work6 then work13, one physical two-hour worker; no general bundling."""
    deadline_remainders.validate_plan_structure(plan, authority, manifest)
    batch = build_batch(authority, manifest, "workers", [remainder_bundle.ANCHOR], partitions,
                        constraint=constraint, manual_refill=True)
    batch["allocations"][0]["reserved_CPU_seconds"] = remainder_bundle.RESERVED_CPU_SECONDS
    batch["command"] = ["--time=120" if value == "--time=480" else value for value in batch["command"]]
    batch["command"] = [
        "--output=outputs/decoder_training/full_remainders002_bundle-%A_%a.out" if value.startswith("--output=") else
        "--error=outputs/decoder_training/full_remainders002_bundle-%A_%a.err" if value.startswith("--error=") else value
        for value in batch["command"]]
    batch["command"][-1] = str(REMAINDER_BUNDLE_WRAPPER)
    batch[deadline_remainders.FIELD] = plan
    batch[remainder_bundle.FIELD] = list(remainder_bundle.MEMBERS)
    remainder_bundle.validate_bundle(batch)
    return batch


def build_io_remainders(authority, manifest, review, selected, partitions, *, constraint=None,
                        effective_parallelism=None):
    items = io_recovery.validate_review_structure(review, manifest)
    require(selected and len(set(selected)) == len(selected) and set(selected) <= set(items),
            "explicit_unique_reviewed_IO_complement_selection_required")
    parallelism.validate_policy(effective_parallelism)
    batch = build_batch(authority, manifest, "workers", selected, partitions,
        constraint=constraint, manual_refill=True, effective_parallelism=effective_parallelism)
    batch[io_recovery.FIELD] = review
    batch["command"][-1] = str(IO_REMAINDER_WRAPPER)
    batch["command"] = [
        "--output=outputs/decoder_training/full_io_remainders001-%A_%a.out" if value.startswith("--output=") else
        "--error=outputs/decoder_training/full_io_remainders001-%A_%a.err" if value.startswith("--error=") else value
        for value in batch["command"]]
    return batch


def build_io_assembly(authority, manifest, plan, review, partitions):
    require(deadline_remainders.plan_relative(plan) == deadline_remainders.PLAN_RELATIVE_002,
            "IO_assembly_requires_cumulative_deadline_plan002")
    io_recovery.validate_review_structure(review, manifest)
    batch = build_remainder_assembly(authority, manifest, plan, partitions)
    batch[io_recovery.FIELD] = review
    batch["command"][-1] = str(IO_ASSEMBLY_WRAPPER)
    return batch


def select_component_wrapper(batch, effective_parallelism):
    """One substantive unassigned work48; optional checks grant no salvage access."""
    parallelism.validate_policy(effective_parallelism)
    require(batch.get("phase") == "workers" and batch.get(accounting.MANUAL_REFILL_FIELD) is True
            and batch.get("allocations") == [{"logical_id": "work_0048", "array_index": 47,
                                             "reserved_CPU_seconds": 28800}]
            and io_quarantine.FIELD in batch
            and not any(field in batch for field in (io_recovery.FIELD, deadline_remainders.FIELD,
                accounting.REPLACEMENT_FIELD, accounting.STARTUP_REPLACEMENT_FIELD, bundle_startup.FIELD)),
            "component_checks_only_in_fresh_amended_work48_parent")
    batch["command"][-1] = str(COMPONENT_WRAPPER)


def select_component_replacement_wrapper(batch, review, effective_parallelism):
    """Explicit one-time replacement of the known portable-tee startup failure."""
    select_component_wrapper(batch, effective_parallelism)
    component_startup.validate_review(review)
    batch[component_startup.FIELD] = review
    batch["command"][-1] = str(COMPONENT_REPAIR_WRAPPER)


def load_io_recovery_context(account, manifest, review, *, reader=contract.prefix.read_json):
    """Closed metadata and three exact stats only; no failed HDF5 payload read."""
    items = io_recovery.validate_review_structure(review, manifest)
    records, reports = {}, {}
    for work, item in items.items():
        path = BASE / item["predecessor_status"]
        require(path.is_file() and not path.is_symlink(), "closed_IO_original_status_required")
        records[work] = reader(path)
        report = item["inspection"]["report"]
        if report not in reports:
            path = BASE / report
            require(path.is_file() and not path.is_symlink(), "closed_IO_inspection_report_required")
            reports[report] = reader(path)
    io_recovery.validate_review(review, manifest, account["jobs"], records, reports, base_directory=str(BASE))
    storage = review["retained_original_reservation"]
    observed = 0
    for relative, evidence in storage["files"].items():
        info = (BASE / relative).lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == evidence["st_nlink"]
                and info.st_size == evidence["st_size"], "reviewed_closed_work62_file_accounting_changed")
        observed += info.st_size
    require(observed + storage["metadata_allowance_bytes"] <= storage["retained_reserved_bytes"],
            "retained_work62_reservation_does_not_cover_current_closed_artifacts")
    return items


def load_remainder_context(account, manifest, plan, *, reader=contract.prefix.read_json):
    """Bind the plan's immutable review; admit unrelated failures by latest review."""
    items = deadline_remainders.validate_plan_structure(plan, account["basis"]["authority"], manifest)
    review_path = BASE / plan["reviewed_partials"]
    require(review_path.is_file() and not review_path.is_symlink(),
            "fixed_published_partial_review_required_for_remainder")
    review = reader(review_path)
    records = {}
    for work, item in items.items():
        path = BASE / item["predecessor_status"]
        require(path.is_file() and not path.is_symlink() and (path.parent / "channels.h5").is_file()
                and not (path.parent / "channels.h5").is_symlink(), "closed_original_fragment_and_status_required")
        records[work] = reader(path)
    deadline_remainders.validate_plan(plan, account["basis"]["authority"], manifest, review, records,
                                      account, base_directory=str(BASE))
    latest = deadline_remainders.latest_published_review(account)
    return deadline_partials.validate_intent({"phase": "workers", "manual_worker_refill": True,
        "allocations": [], deadline_partials.FIELD: latest}, account["jobs"], manifest)


def admit_reviewed_replacement(account, manifest, batch, *, results=require_previous_results):
    accounting.admit_reservation(account, sum(item["reserved_CPU_seconds"] for item in batch["allocations"]))
    accounting.validate_replacement_intent(batch, account["jobs"], account["batches"])
    require("states" in account["assigned"] and "assembly" not in account["assigned"],
            "states_required_and_assembly_is_final")
    remaining = {}
    for key, job in account["jobs"].items():
        if key == accounting.REPLACED_JOB:
            continue
        if job["terminal"] is None:
            require(key in accounting.COEXISTING_WORKERS and job["phase"] == "workers"
                    and job["logical_id"] == f"work_{int(key.rsplit('_', 1)[1]) + 1:04d}",
                    "only_original_other_workers_may_remain_active_during_repair")
        else:
            remaining[key] = job
    # Completed prerequisites/results still use the ordinary exact checks.
    # Only the seven explicitly identified ongoing writers are deferred.
    results({**account, "jobs": remaining}, manifest)


def admit_batch(account, manifest, batch, *, results=require_previous_results):
    reviewed = None
    if account.get("basis") is not None:
        accounting.durable_reservations(account["basis"]["authority"], manifest, account["batches"], proposed=batch)
    quarantined = io_quarantine.validate_intent(batch, account["jobs"], manifest, account["batches"])
    replaced_bundle_startup = bundle_startup.validate_intent(batch, account["jobs"], account["batches"])
    replaced_component_startup = component_startup.validate_intent(batch, account["jobs"], account["batches"])
    remainder = deadline_remainders.FIELD in batch
    recovery = io_recovery.FIELD in batch
    if recovery:
        require(account["basis"] is not None and account.get("effective_parallelism") is not None,
                "IO_recovery_requires_bound_manifest_and_explicit_parallelism_policy")
        io_recovery.validate_intent(batch, account["jobs"], account["batches"], manifest)
        load_io_recovery_context(account, manifest, batch[io_recovery.FIELD])
        latest = deadline_remainders.latest_published_review(account)
        reviewed = deadline_partials.validate_intent({"phase": "workers", "manual_worker_refill": True,
            "allocations": [], deadline_partials.FIELD: latest}, account["jobs"], manifest)
    if remainder:
        require(account["basis"] is not None, "remainder_requires_bound_original_manifest")
        authority = account["basis"]["authority"]
        require(not any(field in batch for field in (accounting.REPLACEMENT_FIELD,
                accounting.STARTUP_REPLACEMENT_FIELD, deadline_partials.FIELD)),
                "remainder_requires_separate_explicit_intent")
        if batch["phase"] == "workers":
            deadline_remainders.validate_remainder_intent(batch, account["jobs"], account["batches"], authority, manifest)
        else:
            accounting.validate_remainder_assembly_intent(batch, account["jobs"], account["batches"], authority, manifest)
        reviewed = load_remainder_context(account, manifest, batch[deadline_remainders.FIELD])
    if deadline_partials.FIELD in batch:
        reviewed = deadline_partials.validate_intent(batch, account["jobs"], manifest)
    if accounting.REPLACEMENT_FIELD in batch:
        return admit_reviewed_replacement(account, manifest, batch, results=results)
    startup_replacement = accounting.STARTUP_REPLACEMENT_FIELD in batch
    if startup_replacement:
        accounting.validate_startup_replacement_intent(batch, account["jobs"], account["batches"])
    accounting.admit_reservation(account, sum(item["reserved_CPU_seconds"] for item in batch["allocations"]))
    require(startup_replacement or replaced_component_startup or (remainder or recovery) and batch["phase"] == "workers"
            or not any(item["logical_id"] in account["assigned"] for item in batch["allocations"]),
            "work_or_phase_cannot_be_submitted_twice")
    if accounting.MANUAL_REFILL_FIELD in batch:
        require(account["basis"] is not None, "manual_refill_requires_bound_manifest")
        accounting.validate_manual_refill_intent(batch, account["jobs"], account["batches"],
            account["basis"]["authority"], manifest=manifest,
            effective_parallelism=account.get("effective_parallelism"))
        # Defer only registered ongoing worker results. Every terminal result,
        # including earlier arrays, retains the exact ordinary closeout checks.
        closed = {key: job for key, job in account["jobs"].items() if job["terminal"] is not None
                  and key != replaced_bundle_startup
                  and key != replaced_component_startup
                  and not (startup_replacement and key == accounting.STARTUP_REPLACED_JOB)}
        if quarantined:
            results({**account, "jobs": closed}, manifest, reviewed_partials=reviewed,
                    reviewed_quarantines=quarantined)
        elif reviewed is None:
            results({**account, "jobs": closed}, manifest)
        else:
            results({**account, "jobs": closed}, manifest, reviewed_partials=reviewed)
    else:
        if recovery:
            results(account, manifest, reviewed_partials=reviewed, reviewed_quarantines=quarantined)
        elif remainder:
            results(account, manifest, reviewed_partials=reviewed)
        else:
            results(account, manifest)
    phase, assigned = batch["phase"], account["assigned"]
    if phase == "states":
        require(not assigned, "states_must_be_first_full_stage_batch")
    else:
        require("states" in assigned and "assembly" not in assigned, "states_required_and_assembly_is_final")
    if phase == "assembly":
        require({item["work_item_id"] for item in manifest["work_items"]} <= assigned,
                "all_required_work_items_must_complete_before_assembly")


def submit_one(batch, rows, append, authority, manifest, *, runner=subprocess.run,
               results=require_previous_results):
    """Caller holds the journal lock through intent and bounded scheduler reply."""
    account = accounting.replay(rows, authority, manifest)
    admit_batch(account, manifest, batch, results=results)
    identifier = f"batch_{len(account['batches']) + 1:04d}"
    event = {"event": "submission_intent", "batch_id": identifier, **batch}
    # Validate the exact prospective event before its durable publication.
    accounting.replay([*rows, event], authority, manifest)
    append(event)
    environment = {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}
    try:
        response = runner(batch["command"], capture_output=True, text=True, timeout=30,
                          check=False, env=environment)
    except (subprocess.TimeoutExpired, OSError) as exc:
        append({"event": "submission_uncertain", "batch_id": identifier,
                "error_type": type(exc).__name__, "error": str(exc), "automatic_retry": False})
        raise Stop("submission_response_uncertain_reconcile_before_any_dispatch") from exc
    match = re.fullmatch(r"([1-9][0-9]*)(?:;[a-zA-Z0-9_.-]+)?", response.stdout.strip())
    if response.returncode != 0 or match is None:
        append({"event": "submission_uncertain", "batch_id": identifier,
                "returncode": response.returncode, "stdout": response.stdout, "stderr": response.stderr,
                "automatic_retry": False})
        raise Stop("submission_response_uncertain_reconcile_before_any_dispatch")
    accepted = {"event": "submission_accepted", "batch_id": identifier, "job_id": match.group(1),
                "scheduler_response": response.stdout.strip()}
    accounting.replay([*rows, accepted], authority, manifest)
    append(accepted)
    return {"batch_id": identifier, "job_id": match.group(1), "phase": batch["phase"],
            "parent_allocations": len(batch["allocations"]), "automatic_successor": False}


def summary(account):
    return {"actual_CPU_seconds": account["actual_CPU_seconds"],
            "effective_parallelism": account.get("effective_parallelism"),
            "outstanding_reserved_CPU_seconds": account["outstanding_reserved_CPU_seconds"],
            "cap_CPU_seconds": account["budget"]["cpu_seconds"],
            "unresolved_batches": [batch["batch_id"] for batch in account["uncertain_batches"]],
            "accepted_parents": list(account["jobs"]),
            "reviewed_extra_engineering_attempts": sum(
                accounting.REPLACEMENT_FIELD in batch or accounting.STARTUP_REPLACEMENT_FIELD in batch
                or bundle_startup.FIELD in batch or component_startup.FIELD in batch
                for batch in account["batches"].values()),
            "superseded_pre_admission_jobs": sorted(account.get("superseded_pre_admission_jobs", set())),
            "additional_remainder_durable_bytes": account.get("additional_remainder_durable_bytes", 0),
            "additional_io_recovery_durable_bytes": account.get("additional_io_recovery_durable_bytes", 0),
            "io_recovery_gross_durable_bytes": account.get("io_recovery_gross_durable_bytes", 0),
            "retained_original_reservation_reduction_bytes": account.get("retained_original_reservation_reduction_bytes", 0),
            "accepted_remainder_attempts": sum(len(batch.get(remainder_bundle.FIELD, batch["allocations"]))
                for batch in account["batches"].values()
                if (deadline_remainders.FIELD in batch or io_recovery.FIELD in batch) and batch["phase"] == "workers"
                and batch["accepted_id"] is not None),
            "accepted_remainder_parent_allocations": sum(len(batch["allocations"])
                for batch in account["batches"].values()
                if (deadline_remainders.FIELD in batch or io_recovery.FIELD in batch) and batch["phase"] == "workers"
                and batch["accepted_id"] is not None),
            "rejected_batch_requires_review": account["rejected_batch"]}


def run(args):
    require(Path.cwd().resolve() == BASE, "exact_episode_working_directory_required")
    require("-ln" in socket.gethostname() and not os.environ.get("SLURM_JOB_ID"),
            "coordinator_login_submission_required_not_worker_submission")
    authority = state.authorize_full(args.authorization)
    accounting.budget_definition(authority)
    require(contract.ROOT.is_dir() and not contract.ROOT.is_symlink(), "prepared_full_stage_directory_required")
    if args.action == "status":
        with accounting.locked_events(JOURNAL, shared=True) as (rows, _):
            return summary(accounting.replay(rows, authority))
    manifest = None
    if args.action in {"submit", "replace-unstarted-worker", "replace-startup-worker",
                       "submit-deadline-remainders", "submit-remainder-bundle", "assemble-with-remainders",
                       "amend-parallelism", "submit-io-remainders", "assemble-with-io-recovery"}:
        require(Path(args.manifest).absolute() == contract.MANIFEST and not contract.MANIFEST.is_symlink(),
                "fixed_full_stage_work_manifest_required")
        manifest = contract.prefix.read_json(contract.MANIFEST)
        contract.validate_work_manifest(manifest, authority)
    with accounting.locked_events(JOURNAL, create=args.action == "initialize") as (rows, append):
        if args.action == "initialize":
            require(not rows, "existing_budget_journal_cannot_be_reinitialized")
            append({"event": "budget_opened", "budget": accounting.budget_definition(authority)})
            return summary(accounting.replay(rows, authority))
        account = accounting.replay(rows, authority, manifest)
        if args.action == "amend-parallelism":
            updated = accounting.append_parallelism_amendment(rows, append, authority, manifest,
                observed_utc=args.observed_utc, user_instruction=args.user_instruction)
            return summary(updated)
        if args.action in {"terminal", "resolve-submission"}:
            if args.action == "terminal":
                event = {"event": "terminal_charge", "job_id": args.job_id,
                         "allocation_job_id": args.allocation_job_id, "allocated_cpus": args.allocated_cpus,
                         "elapsed_seconds": args.elapsed_seconds, "state": args.state,
                         "exit_code": args.exit_code, "observed_utc": args.observed_utc, "evidence": args.evidence}
            elif args.not_accepted:
                event = {"event": "submission_not_accepted", "batch_id": args.batch_id, "evidence": args.evidence}
            else:
                event = {"event": "submission_accepted", "batch_id": args.batch_id,
                         "job_id": args.job_id, "evidence": args.evidence}
            updated = accounting.replay([*rows, event], authority)
            append(event)
            return summary(updated)
        require(args.action in {"submit", "replace-unstarted-worker", "replace-startup-worker",
                               "submit-deadline-remainders", "submit-remainder-bundle", "assemble-with-remainders",
                               "submit-io-remainders", "assemble-with-io-recovery"},
                "recognized_operator_action_required")
        replacement = args.action == "replace-unstarted-worker"
        if args.action in {"submit-io-remainders", "assemble-with-io-recovery"}:
            require(Path(args.review).absolute() == IO_SALVAGE_REVIEW
                    and IO_SALVAGE_REVIEW.is_file() and not IO_SALVAGE_REVIEW.is_symlink(),
                    "fixed_explicit_IO_salvage_review_required")
            review = contract.prefix.read_json(IO_SALVAGE_REVIEW)
            if args.action == "submit-io-remainders":
                batch = build_io_remainders(authority, manifest, review, args.work_items, args.partitions,
                    constraint=args.constraint, effective_parallelism=account.get("effective_parallelism"))
            else:
                require(Path(args.remainder_manifest).absolute() == REMAINDER_PLAN_002
                        and REMAINDER_PLAN_002.is_file() and not REMAINDER_PLAN_002.is_symlink(),
                        "IO_assembly_requires_fixed_deadline_plan002")
                batch = build_io_assembly(authority, manifest, contract.prefix.read_json(REMAINDER_PLAN_002),
                                          review, args.partitions)
        elif args.action in {"submit-deadline-remainders", "submit-remainder-bundle", "assemble-with-remainders"}:
            remainder_path = Path(args.remainder_manifest).absolute()
            require(remainder_path in (REMAINDER_PLAN, REMAINDER_PLAN_002)
                    and remainder_path.is_file() and not remainder_path.is_symlink(),
                    "fixed_exact_deadline_remainder_sidecar_required")
            plan = contract.prefix.read_json(remainder_path)
            require(BASE / deadline_remainders.plan_relative(plan) == remainder_path,
                    "remainder_sidecar_content_identity_mismatch")
            if args.action == "submit-deadline-remainders":
                batch = build_deadline_remainders(authority, manifest, plan, args.work_items,
                    args.partitions, constraint=args.constraint,
                    clean_work5=getattr(args, "io_quarantine_review", None) is not None)
            elif args.action == "submit-remainder-bundle":
                batch = build_remainder_bundle(authority, manifest, plan, args.partitions, constraint=args.constraint)
            else:
                batch = build_remainder_assembly(authority, manifest, plan, args.partitions)
        elif replacement:
            require(Path(args.review).absolute() == REPLACEMENT_REVIEW
                    and REPLACEMENT_REVIEW.is_file() and not REPLACEMENT_REVIEW.is_symlink(),
                    "exact_coordinator_replacement_review_sidecar_required")
            batch = build_reviewed_replacement(authority, manifest,
                    contract.prefix.read_json(REPLACEMENT_REVIEW), args.partitions)
        else:
            review = None
            if args.deadline_partial_review is not None:
                review_path = Path(args.deadline_partial_review).absolute()
                require(review_path == DEADLINE_PARTIAL_REVIEW
                        or review_path == DEADLINE_PARTIAL_REVIEW_002
                        and args.action == "submit" and args.phase == "workers" and args.manual_refill,
                        "fixed_coordinator_deadline_partial_review_required")
                require(review_path.is_file() and not review_path.is_symlink(),
                        "fixed_coordinator_deadline_partial_review_required")
                review = contract.prefix.read_json(review_path)
            if args.action == "replace-startup-worker":
                require(Path(args.review).absolute() == STARTUP_REPLACEMENT_REVIEW
                        and STARTUP_REPLACEMENT_REVIEW.is_file() and not STARTUP_REPLACEMENT_REVIEW.is_symlink(),
                        "exact_coordinator_startup_replacement_review_required")
                batch = build_startup_replacement(authority, manifest,
                    contract.prefix.read_json(STARTUP_REPLACEMENT_REVIEW), review, args.partitions,
                    constraint=args.constraint)
            else:
                batch = build_batch(authority, manifest, args.phase, args.work_items or (), args.partitions,
                                    constraint=args.constraint, manual_refill=args.manual_refill,
                                    deadline_partial_review=review,
                                    effective_parallelism=(account.get("effective_parallelism")
                                        if args.phase == "workers" and args.manual_refill else None))
        quarantine_path = getattr(args, "io_quarantine_review", None)
        if quarantine_path is not None:
            require((args.action in {"submit", "submit-remainder-bundle", "submit-deadline-remainders",
                                    "submit-io-remainders"}
                    and batch["phase"] == "workers" and batch.get(accounting.MANUAL_REFILL_FIELD) is True)
                    or (args.action == "assemble-with-io-recovery" and batch["phase"] == "assembly"
                        and io_recovery.FIELD in batch),
                    "io_quarantine_only_for_explicit_manual_worker_or_fixed_bundle")
            require(Path(quarantine_path).absolute() == IO_QUARANTINE_REVIEW
                    and IO_QUARANTINE_REVIEW.is_file() and not IO_QUARANTINE_REVIEW.is_symlink(),
                    "fixed_coordinator_io_quarantine_review_required")
            batch[io_quarantine.FIELD] = contract.prefix.read_json(IO_QUARANTINE_REVIEW)
            if args.action == "submit":
                batch["command"][-1] = str(PARALLEL_WRAPPER if account.get("effective_parallelism") is not None
                                           else IO_RECOVERY_WRAPPER)
        if getattr(args, "component_checks", False):
            require(args.action == "submit", "component_wrapper_only_for_explicit_fresh_submission")
            select_component_wrapper(batch, account.get("effective_parallelism"))
        component_review_path = getattr(args, "component_startup_review", None)
        if component_review_path is not None:
            require(args.action == "submit" and not getattr(args, "component_checks", False)
                    and Path(component_review_path).absolute() == COMPONENT_STARTUP_REVIEW
                    and COMPONENT_STARTUP_REVIEW.is_file() and not COMPONENT_STARTUP_REVIEW.is_symlink(),
                    "fixed_explicit_component_startup_review_required")
            select_component_replacement_wrapper(batch, contract.prefix.read_json(COMPONENT_STARTUP_REVIEW),
                                                 account.get("effective_parallelism"))
        startup_review_path = getattr(args, "startup_review", None)
        if startup_review_path is not None:
            review_path = Path(startup_review_path).absolute()
            require(args.action == "submit-remainder-bundle"
                    and review_path in (BUNDLE_STARTUP_REVIEW, BUNDLE_STARTUP_REVIEW_002, BUNDLE_STARTUP_REVIEW_003)
                    and review_path.is_file() and not review_path.is_symlink(),
                    "fixed_bundle_startup_review_required")
            startup_review = contract.prefix.read_json(review_path)
            require(BASE / bundle_startup.review_relative(startup_review) == review_path,
                    "bundle_startup_sidecar_content_identity_mismatch")
            batch[bundle_startup.FIELD] = startup_review
            if review_path == BUNDLE_STARTUP_REVIEW_003:
                batch["command"][-1] = str(REMAINDER_BUNDLE_WRAPPER_003)
                batch["command"] = [
                    "--output=outputs/decoder_training/full_remainders003_bundle-%A_%a.out" if value.startswith("--output=") else
                    "--error=outputs/decoder_training/full_remainders003_bundle-%A_%a.err" if value.startswith("--error=") else value
                    for value in batch["command"]]
        admit_batch(account, manifest, batch)
        require(Path(batch["command"][-1]).is_file(), "prepared_full_stage_batch_wrapper_required")
        if batch["phase"] == "states":
            fresh = [contract.ROOT / "state_preparation_001", contract.ROOT / "shards",
                     contract.ROOT / "assembled_decoder_manifest.json"]
            fresh.extend(state.state_directory(person) for person in state.NEW_PARTICIPANTS)
        elif batch["phase"] == "workers":
            if io_recovery.FIELD in batch:
                items = io_recovery.validate_review_structure(batch[io_recovery.FIELD], manifest)
                fresh = [BASE / items[item["logical_id"]]["output_directory"] for item in batch["allocations"]]
            elif deadline_remainders.FIELD in batch:
                remainder_items = {item["work_item_id"]: item for item in batch[deadline_remainders.FIELD]["work_items"]}
                members = batch.get(remainder_bundle.FIELD, [item["logical_id"] for item in batch["allocations"]])
                fresh = [BASE / remainder_items[work]["output_directory"] for work in members]
            else:
                fresh = [contract.ROOT / "shards" / item["logical_id"] for item in batch["allocations"]]
        else:
            fresh = [contract.ROOT / "assembled_decoder_manifest.json"]
        require(all(not path.exists() and not path.is_symlink() for path in fresh), "attempt_outputs_must_be_fresh")
        if replacement:
            require_reviewed_replacement_queue(account, authority, subprocess.run)
        elif accounting.MANUAL_REFILL_FIELD in batch:
            require_manual_refill_queue(account, authority, batch, subprocess.run)
        else:
            require_no_live_episode_job(subprocess.run)  # One observation; no polling or automatic retry.
        if account["basis"] is None:
            append({"event": "manifest_bound", "authority": authority, "manifest": manifest})
        return submit_one(batch, rows, append, authority, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("initialize")
    sub.add_parser("status")
    amendment = sub.add_parser("amend-parallelism", help="Record the explicit user concurrency amendment without changing budgets or assignments")
    amendment.add_argument("--manifest", required=True)
    amendment.add_argument("--observed-utc", required=True)
    amendment.add_argument("--user-instruction", required=True,
                           help="Exact user instruction authorizing greater concurrent execution")
    launch = sub.add_parser("submit")
    launch.add_argument("--manifest", required=True)
    launch.add_argument("--phase", choices=("states", "workers", "assembly"), required=True)
    launch.add_argument("--work-items", nargs="+")
    launch.add_argument("--partitions", required=True)
    launch.add_argument("--constraint", choices=(WORKER_CPU_CONSTRAINT,),
                        help="Optional discovered CPU feature for ordinary worker batches only; no default constraint")
    launch.add_argument("--manual-refill", action="store_true",
                        help="Explicitly admit fresh workers alongside reconciled outstanding workers within the effective global cap")
    launch.add_argument("--deadline-partial-review",
                        help="Explicit coordinator review admitting unrelated fresh manual workers; no partial retry")
    launch.add_argument("--io-quarantine-review",
                        help="Exact reviewed Oak incident: unrelated manual work only; no failed-fragment reuse")
    launch.add_argument("--component-checks", action="store_true",
                        help="Bounded pending recovery and EP19 checks in fresh work48 only; grants no recovery execution")
    launch.add_argument("--component-startup-review",
                        help="One manual work48 replacement after the exact charged portable-tee startup failure")
    repair = sub.add_parser("replace-unstarted-worker", help="One reviewed pre-admission replacement of 46839670_7")
    repair.add_argument("--manifest", required=True)
    repair.add_argument("--review", required=True)
    repair.add_argument("--partitions", required=True)
    startup = sub.add_parser("replace-startup-worker", help="One reviewed pre-empirical replacement of 46880582_12")
    startup.add_argument("--manifest", required=True)
    startup.add_argument("--review", required=True)
    startup.add_argument("--deadline-partial-review", required=True)
    startup.add_argument("--partitions", required=True)
    startup.add_argument("--constraint", choices=(WORKER_CPU_CONSTRAINT,))
    remainders = sub.add_parser("submit-deadline-remainders", help="One manual exact remainder per reviewed work 5, 6 or 7")
    remainders.add_argument("--manifest", required=True)
    remainders.add_argument("--remainder-manifest", required=True)
    remainders.add_argument("--work-items", nargs="+", choices=deadline_remainders.WORK_IDS, required=True)
    remainders.add_argument("--partitions", required=True)
    remainders.add_argument("--constraint", choices=(WORKER_CPU_CONSTRAINT,))
    remainders.add_argument("--io-quarantine-review",
                           help="Exact work5-only plan002 remainder route; preserves the separate five-fragment quarantine")
    bundle = sub.add_parser("submit-remainder-bundle", help="One manual two-hour work6/work13 serial remainder bundle")
    bundle.add_argument("--manifest", required=True)
    bundle.add_argument("--remainder-manifest", required=True)
    bundle.add_argument("--partitions", required=True)
    bundle.add_argument("--constraint", choices=(WORKER_CPU_CONSTRAINT,))
    bundle.add_argument("--io-quarantine-review",
                        help="Exact reviewed Oak incident, separate from the unchanged work6/work13 predecessors")
    bundle.add_argument("--startup-review",
                        help="Explicit review001/002/003 of named failures before decoder fitting began")
    remainder_assembly = sub.add_parser("assemble-with-remainders", help="Explicitly assemble the original and complete remainder fragments")
    remainder_assembly.add_argument("--manifest", required=True)
    remainder_assembly.add_argument("--remainder-manifest", required=True)
    remainder_assembly.add_argument("--partitions", required=True)
    io_workers = sub.add_parser("submit-io-remainders", help="One manual exact fresh complement per inspected I/O original")
    io_workers.add_argument("--manifest", required=True)
    io_workers.add_argument("--io-recovery-review", dest="review", required=True)
    io_workers.add_argument("--io-quarantine-review", required=True)
    io_workers.add_argument("--work-items", nargs="+", choices=io_recovery.WORK_IDS, required=True)
    io_workers.add_argument("--partitions", required=True)
    io_workers.add_argument("--constraint", choices=(WORKER_CPU_CONSTRAINT,))
    io_assembly = sub.add_parser("assemble-with-io-recovery", help="Complete assembly with explicit deadline and inspected I/O complements")
    io_assembly.add_argument("--manifest", required=True)
    io_assembly.add_argument("--remainder-manifest", required=True)
    io_assembly.add_argument("--io-recovery-review", dest="review", required=True)
    io_assembly.add_argument("--io-quarantine-review", required=True)
    io_assembly.add_argument("--partitions", required=True)
    terminal = sub.add_parser("terminal")
    terminal.add_argument("--job-id", required=True, help="Numeric parent, or arrayparent_index; never a step")
    terminal.add_argument("--allocation-job-id", required=True, help="Numeric SLURM_JOB_ID recorded by that task")
    terminal.add_argument("--allocated-cpus", type=int, required=True)
    terminal.add_argument("--elapsed-seconds", type=int, required=True)
    terminal.add_argument("--state", choices=sorted(accounting.TERMINAL), required=True)
    terminal.add_argument("--exit-code", required=True)
    terminal.add_argument("--observed-utc", required=True)
    terminal.add_argument("--evidence", required=True, help="Coordinator's terminal parent accounting observation")
    resolve = sub.add_parser("resolve-submission")
    resolve.add_argument("--batch-id", required=True)
    accepted = resolve.add_mutually_exclusive_group(required=True)
    accepted.add_argument("--job-id")
    accepted.add_argument("--not-accepted", action="store_true")
    resolve.add_argument("--evidence", required=True)
    try:
        print(json.dumps(run(parser.parse_args()), sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "dispatch_stopped", "error_type": type(exc).__name__,
                          "error": str(exc), "automatic_retry": False}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
