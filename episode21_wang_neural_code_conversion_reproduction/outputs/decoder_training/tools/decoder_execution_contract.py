"""Unactivated full-decoder work assignments, coverage and finite resources.

Importing this module performs no scientific import, payload read or execution.
It does not authorize or submit jobs. The separate full-stage authority is
owned by participant_training_state; callers admit it before using a work item.
"""
from __future__ import annotations

import json
from datetime import datetime
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import time

import decoder_coverage as coverage
import train_layer_continuation as layer

prefix = layer.prefix
require, Stop = prefix.require, prefix.Stop
BASE = prefix.BASE
ROOT = BASE / "outputs/decoder_training/full_continuation_001"
MANIFEST = ROOT / "work_manifest.json"
MANIFEST_RELATIVE = str(MANIFEST.relative_to(BASE))
TERMINAL_STATES = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY",
                   "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
RESOURCE_KEYS = {"cpus", "memory_gib", "gpus", "minutes", "durable_bytes", "scratch_bytes"}
CLOSED_LAYER_METADATA_BYTES = 64 * 1024**2
RESUMABLE_LAYER_ERRORS = {"layer_deadline_reached", "continuation_deadline_reached", "termination_requested",
                          "worker_RSS_closure_threshold", "new_layer_durable_byte_cap",
                          "channel_scratch_cap", "node_local_capacity_required"}


def validate_resources(resources):
    require(isinstance(resources, dict) and set(resources) == RESOURCE_KEYS,
            "explicit_exact_job_resource_fields_required")
    require(all(type(resources[key]) is int for key in RESOURCE_KEYS)
            and resources["cpus"] == 1 and resources["memory_gib"] == 7 and resources["gpus"] == 0
            and resources["minutes"] > 5 and resources["durable_bytes"] > 0
            and resources["scratch_bytes"] > 0, "finite_one_CPU_7GiB_resource_caps_required")
    return resources


def parse_span(record):
    require(isinstance(record, dict)
            and set(record) == {"participant", "layer", "channel", "spatial_Fortran_interval"},
            "exact_span_fields_required")
    interval = record["spatial_Fortran_interval"]
    require(isinstance(interval, list) and len(interval) == 2, "explicit_spatial_interval_required")
    return coverage.Span(record["participant"], record["layer"], record["channel"], *interval).validate()


def _safe_file(path):
    require(path.is_file() and not path.is_symlink(), "retained_regular_file_required")


def read_retained_coverage(retained_entries):
    """Read terminal metadata only; never turn a planned layer into retained work.

The two consumed legacy jobs are mandatory. An optional later layer can add
exact committed rows, including a terminal partial result. Uncertain marker
publication needs reconciliation before it can supply a new work assignment.
The coordinator supplies its independently observed closed-parent evidence.
Actual fragment arrays/markers are subsequently checked by the assembler.
"""
    require(isinstance(retained_entries, list) and 2 <= len(retained_entries) <= 3,
            "two_legacy_references_and_optional_layer_required")
    expected_paths = {
        "retained_prefix": "outputs/decoder_training/retained_prefix_001/prefix_status.json",
        "channel_continuation": "outputs/decoder_training/continuation_001/continuation_status.json",
        "layer_continuation": "outputs/decoder_training/layer_continuation_001/continuation_status.json",
    }
    entries = {}
    for item in retained_entries:
        require(isinstance(item, dict) and item.get("kind") in expected_paths,
                "recognized_retained_format_required")
        kind = item["kind"]
        require(kind not in entries and item.get("status") == expected_paths[kind],
                "unique_fixed_retained_reference_required")
        _safe_file(BASE / item["status"])
        entries[kind] = item
    require({"retained_prefix", "channel_continuation"} <= set(entries),
            "completed_legacy_work_cannot_be_omitted")
    layer_approved = layer.AUTH.is_file() and prefix.read_json(layer.AUTH).get("approved") is True
    if layer_approved or any((layer.OUTPUT / name).exists()
                             for name in ("requested_units.json", "continuation_status.json")):
        require("layer_continuation" in entries,
                "existing_layer_attempt_must_be_reconciled_before_new_assignments")
    binding = prefix.read_json(prefix.OUTPUT / "training_row_binding.json")
    layer.admit_completed(binding["training_image_ids"])
    for name in prefix.LAYERS:
        _safe_file(prefix.OUTPUT / (name + ".h5"))
    spans = list(coverage.retained_spans())
    if "layer_continuation" not in entries:
        return tuple(spans)
    entry = entries["layer_continuation"]
    report = prefix.read_json(BASE / entry["status"])
    closed = entry.get("closed_parent", {})
    require(isinstance(closed, dict) and isinstance(closed.get("job_id"), str)
            and closed["job_id"].isdigit() and closed.get("state") in TERMINAL_STATES
            and isinstance(closed.get("exit_code"), str) and ":" in closed["exit_code"]
            and isinstance(closed.get("observed_utc"), str) and bool(closed["observed_utc"])
            and report.get("allocation", {}).get("job_id") == closed["job_id"],
            "coordinator_observed_terminal_parent_required")
    require(report.get("schema_version") == "ep21.decoder_layer_continuation.v1"
            and report.get("status") in {"complete_layer_continuation_only",
                "partial_layer_continuation_stopped", "all_units_committed_layer_manifest_publication_failed"}
            and report.get("participant") == "sub-01" and report.get("layer") == "conv1_1"
            and report.get("training_role") == "subject_unique" and report.get("training_images") == 4712
            and report.get("training_repetitions") == 18843
            and report.get("shared_state_reference") == str(prefix.OUTPUT / "shared_training_state.npz")
            and report.get("units_requested") == 3_161_088
            and report.get("retained_completed_units") == 51328
            and report.get("fragment") == "channels.h5", "recognized_terminal_layer_scope_required")
    require(report.get("remaining_interval_status") == "exact_uncommitted_coverage"
            and report.get("completed_units_is_lower_bound") is False
            and "uncertain_publication" not in report
            and isinstance(report.get("closeout_errors"), list)
            and not any(item.get("affects_fragment") is not False for item in report["closeout_errors"]),
            "uncertain_fragment_cannot_supply_remaining_work")
    require(all(report.get(key) is False for key in ("predictions_or_scores_computed", "tau_train_values_read",
                "tau_test_values_read", "OOD_access", "complete_decoder", "successor_authorized")),
            "retained_layer_data_boundaries_required")
    requested = tuple(parse_span(item) for item in report["requested_spans"])
    committed = tuple(parse_span(item) for item in report["committed_spans"])
    remaining = tuple(parse_span(item) for item in report["remaining_spans"])
    require(requested == layer.SPANS, "fixed_layer_request_required")
    allowed_keys = {item.key for item in requested}
    require(all(item.key in allowed_keys for item in (*committed, *remaining)),
            "retained_layer_span_outside_request")
    coverage.verify_complete_coverage([coverage.Span("sub-01", "conv1_1", 0, 0, 50176),
                                      *committed, *remaining], participants=("sub-01",), layers=("conv1_1",))
    require(type(report.get("completed_units")) is int
            and sum(item.count for item in committed) == report["completed_units"],
            "retained_committed_count_mismatch")
    if report["status"] == "complete_layer_continuation_only":
        require(not remaining and report.get("complete_layer") is True,
                "complete_layer_report_must_cover_every_unit")
    if committed:
        _safe_file((BASE / entry["status"]).parent / "channels.h5")
    return tuple([*spans, *committed])


def validate_retained_continuation_causes(retained_entries, authority):
    """An exact coverage subtraction cannot itself resolve a failed fit.

All committed rows remain retained even after a scientific failure, but this
general worker cannot resume its failed target unchanged. A diagnosed bounded
resource/deadline stop needs the new full authority to name that exact stop.
Invalid target scales, nonfinite data/solves and contract defects require their
own correction/amendment before a different runnable plan can be implemented.
"""
    for entry in retained_entries:
        if entry["kind"] != "layer_continuation":
            continue
        report = prefix.read_json(BASE / entry["status"])
        if report["completed_units"] == report["units_requested"]:
            continue  # No failed layer target is being retried by the new jobs.
        require(report.get("error_type") == "Stop" and report.get("error") in RESUMABLE_LAYER_ERRORS,
                "failed_required_layer_target_needs_correction_or_scientific_amendment")
        expected = {"job_id": report["allocation"]["job_id"], "error": report["error"],
                    "cause": "resource_or_deadline"}
        require(authority.get("resume_partial_layer") == expected,
                "explicit_new_authority_for_exact_partial_layer_stop_required")


def included_layer_storage_charge(authority, included, retained_entries):
    """Reclaim only an explicitly bound, closed layer's unused reservation.

    Called after terminal coverage/cause validation. Only named file sizes and
    terminal JSON are read; no fragment payload is opened. The original 25-GB
    included-job identity and CPU journal remain unchanged. The fixed reserve
    covers named scheduler/ASTRA closeout metadata, not another writer or retry.
    """
    bound = authority.get("closed_included_layer_storage")
    if bound is None:
        return included["durable_bytes"] if included else 0
    require(included is not None and isinstance(bound, dict)
            and set(bound) == {"job_id", "files", "metadata_allowance_bytes", "charged_bytes"}
            and bound["job_id"] == included["job_id"]
            and type(bound["metadata_allowance_bytes"]) is int
            and bound["metadata_allowance_bytes"] == CLOSED_LAYER_METADATA_BYTES,
            "explicit_closed_included_layer_storage_bound_required")
    entries = [entry for entry in retained_entries if entry["kind"] == "layer_continuation"]
    require(len(entries) == 1 and entries[0].get("closed_parent", {}).get("job_id") == included["job_id"]
            and entries[0]["closed_parent"].get("state") in TERMINAL_STATES,
            "storage_reclamation_requires_closed_included_parent")
    directory = BASE / "outputs/decoder_training/layer_continuation_001"
    status_path = directory / "continuation_status.json"
    require(entries[0].get("status") == str(status_path.relative_to(BASE))
            and directory.is_dir() and not directory.is_symlink(),
            "fixed_closed_layer_storage_directory_required")
    names = {"channels.h5", "requested_units.json", "continuation_status.json"}
    optional = directory / "complete_layer_manifest.json"
    if optional.exists() or optional.is_symlink():
        names.add(optional.name)
    expected = {str((directory / name).relative_to(BASE)) for name in names}
    files = bound["files"]
    require(isinstance(files, dict) and set(files) == expected,
            "all_named_closed_layer_files_required")
    for relative, size in files.items():
        path = BASE / relative
        _safe_file(path)
        require(type(size) is int and size > 0 and path.stat().st_size == size,
                "closed_layer_file_size_changed_or_undercharged")
    report = prefix.read_json(status_path)
    require(report.get("allocation", {}).get("job_id") == included["job_id"]
            and report.get("resources", {}).get("durable_bytes") == included["durable_bytes"]
            and report.get("status") in {"complete_layer_continuation_only",
                "partial_layer_continuation_stopped", "all_units_committed_layer_manifest_publication_failed"}
            and report.get("completed_units_is_lower_bound") is False
            and report.get("remaining_interval_status") == "exact_uncommitted_coverage"
            and report.get("closeout_errors") == []
            and type(report.get("durable_bytes_before_final_status")) is int
            and report["durable_bytes_before_final_status"]
                == sum(files.values()) - files[str(status_path.relative_to(BASE))],
            "closed_layer_storage_must_match_terminal_accounting")
    log_root = BASE / "outputs/decoder_training"
    metadata = 0
    for suffix in ("out", "err"):
        path = log_root / f"continue_layer001-{included['job_id']}.{suffix}"
        _safe_file(path)
        metadata += path.stat().st_size
    astra = log_root / f"continue_layer001-astra-{included['job_id']}.json"
    if astra.exists() or astra.is_symlink():
        _safe_file(astra)
        metadata += astra.stat().st_size
    require(metadata <= CLOSED_LAYER_METADATA_BYTES,
            "closed_layer_metadata_allowance_exceeded")
    require(type(bound["charged_bytes"]) is int
            and bound["charged_bytes"] == sum(files.values()) + CLOSED_LAYER_METADATA_BYTES
            and bound["charged_bytes"] <= included["durable_bytes"],
            "closed_layer_charge_must_cover_files_and_fixed_metadata_allowance")
    return bound["charged_bytes"]


def validate_work_manifest(manifest, authority):
    """Require exact all-five coverage and finite aggregate caps before dispatch.

Concurrency remains a coordinator/scheduler responsibility. This verifies that
the declared cap fits the aggregate transient allowance; it starts no workers.
"""
    require(authority.get("work_manifest") == MANIFEST_RELATIVE,
            "fixed_full_stage_work_manifest_required")
    require(manifest.get("schema_version") == "ep21.decoder_work_manifest.v1"
            and manifest.get("participants") == list(coverage.PARTICIPANTS)
            and manifest.get("total_required_units") == coverage.TOTAL
            and manifest.get("training_role") == "subject_unique"
            and manifest.get("training_images") == 4712
            and manifest.get("numerical_block_units") == 64, "unchanged_full_decoder_target_required")
    retained = read_retained_coverage(manifest["retained"])
    validate_retained_continuation_causes(manifest["retained"], authority)
    require(type(authority.get("reuse_completed_units")) is int
            and authority["reuse_completed_units"] == sum(span.count for span in retained),
            "authorized_retained_units_must_match_actual_coverage")
    assignments, all_spans, resources = {}, list(retained), []
    require(isinstance(manifest.get("work_items"), list) and manifest["work_items"],
            "explicit_nonempty_work_assignments_required")
    for item in manifest["work_items"]:
        require(isinstance(item, dict) and set(item) == {"work_item_id", "participant", "spans", "resources", "units_requested"},
                "exact_work_item_fields_required")
        identifier = item["work_item_id"]
        require(isinstance(identifier, str) and re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", identifier)
                and identifier not in assignments, "unique_safe_work_item_id_required")
        spans = tuple(parse_span(row) for row in item["spans"])
        require(spans and item["participant"] in coverage.PARTICIPANTS
                and all(span.participant == item["participant"] for span in spans)
                and len({span.key for span in spans}) == len(spans),
                "one_participant_one_interval_per_channel_in_work_item_required")
        require(type(item["units_requested"]) is int
                and item["units_requested"] == sum(span.count for span in spans),
                "assigned_unit_count_mismatch")
        # Stable order is mathematical layer order, not alphabetic conv/FC order.
        order = lambda span: (prefix.LAYERS.index(span.layer), span.channel, span.start)
        require(list(spans) == sorted(spans, key=order), "fixed_layer_channel_traversal_required")
        resource_cap = validate_resources(item["resources"])
        largest_bank = max(4712 * (span.shape[1] * span.shape[2] if len(span.shape) == 3
                                  else span.shape[0]) * 4 for span in spans)
        require(largest_bank <= resource_cap["scratch_bytes"]
                and sum(span.count for span in spans) * 6045 + len(spans) * 1_048_576
                    <= resource_cap["durable_bytes"], "work_item_storage_caps_cannot_cover_request")
        assignments[identifier] = item
        all_spans.extend(spans)
        resources.append(resource_cap)
    require(coverage.verify_complete_coverage(all_spans) == coverage.TOTAL,
            "full_remaining_target_cannot_be_shrunk_or_replayed")
    stage = authority.get("resources", {})
    required = {"worker_jobs", "state_build_jobs", "assembly_jobs", "maximum_concurrent_jobs",
                "maximum_allocated_CPU_seconds", "durable_bytes", "peak_transient_bytes"}
    require(isinstance(stage, dict) and set(stage) == required
            and all(type(stage[key]) is int and stage[key] > 0 for key in required)
            and stage["worker_jobs"] == len(assignments)
            and stage["state_build_jobs"] == stage["assembly_jobs"] == 1,
            "explicit_finite_full_stage_job_caps_required")
    resources.extend((validate_resources(authority["state_build_resources"]),
                      validate_resources(authority["assembly_resources"])))
    import decoder_batch_accounting as accounting
    extra_durable = 0
    batch_mode = authority.get("cpu_budget_mode") == accounting.MODE
    if batch_mode:
        budget = accounting.budget_definition(authority)
        included = budget["included_layer_job"]
        layers = [entry for entry in manifest["retained"] if entry["kind"] == "layer_continuation"]
        require((not layers and included is None) or
                (len(layers) == 1 and included is not None
                 and layers[0]["closed_parent"]["job_id"] == included["job_id"]),
                "first_layer_cost_must_be_included_with_retained_layer")
        extra_durable = accounting.JOURNAL_BYTES + included_layer_storage_charge(
            authority, included, manifest["retained"])
    require(stage["maximum_concurrent_jobs"] <= min(8, len(resources))
            and (batch_mode or sum(cap["cpus"] * cap["minutes"] * 60 for cap in resources)
                <= stage["maximum_allocated_CPU_seconds"])
            and sum(cap["durable_bytes"] for cap in resources) + extra_durable <= stage["durable_bytes"]
            and sum(sorted((cap["scratch_bytes"] for cap in resources), reverse=True)
                    [:stage["maximum_concurrent_jobs"]]) <= stage["peak_transient_bytes"],
            "aggregate_full_stage_caps_exceeded")
    return assignments


def admit_registered_allocation(authority, manifest, logical_id, allocation):
    """Admit a registered allocation and return its optional effective scratch cap.

    The original resource record remains unchanged. Only an accepted worker
    whose intent follows the explicit parallelism amendment receives the new
    cap; running legacy workers retain their original admission and records.
    """
    import decoder_batch_accounting as accounting
    if authority.get("cpu_budget_mode") != accounting.MODE:
        return
    with accounting.locked_events(ROOT / "submission.jsonl", shared=True) as (events, _):
        account = accounting.replay(events, authority, manifest)
    job_id = allocation["job_id"]
    if logical_id.startswith("work_"):
        parent, index = os.environ.get("SLURM_ARRAY_JOB_ID", ""), os.environ.get("SLURM_ARRAY_TASK_ID", "")
        require(parent.isdigit() and re.fullmatch(r"0|[1-9][0-9]*", index),
                "registered_array_allocation_required")
        job_id = f"{parent}_{index}"
    job = account["jobs"].get(job_id)
    require(job is not None and job["logical_id"] == logical_id and job["terminal"] is None
            and allocation["allocated_cpus"] * allocation["time_limit_seconds"] <= job["reserved_CPU_seconds"]
            and not account["uncertain_batches"] and not account["rejected_batch"]
            and account["actual_CPU_seconds"] + account["outstanding_reserved_CPU_seconds"]
                <= account["budget"]["cpu_seconds"], "unregistered_or_unfunded_full_stage_allocation")
    batch = account["batches"].get(job["batch_id"], {})
    if logical_id.startswith("work_") and batch.get("effective_parallelism") is not None:
        import decoder_parallelism_amendment as parallelism
        cap = parallelism.work_scratch_bytes(manifest, [logical_id])
        item = next(item for item in manifest["work_items"] if item["work_item_id"] == logical_id)
        require(0 < cap <= item["resources"]["scratch_bytes"],
                "amended_scratch_cap_must_fit_original_worker_bound")
        return {"effective_scratch_bytes": cap, "parallelism_amendment_id": parallelism.EVENT}


def verify_allocation(resources):
    validate_resources(resources)
    host = socket.gethostname().split(".")[0]
    job = os.environ.get("SLURM_JOB_ID", "")
    require(job.isdigit() and "-ln" not in host, "actual_compute_allocation_required")
    require(all(os.environ.get(name) == "1" for name in prefix.THREADS)
            and os.environ.get("SLURM_CPUS_PER_TASK") == "1"
            and os.environ.get("SLURM_MEM_PER_NODE") == "7168", "one_CPU_7GiB_six_thread_caps_required")
    array_parent = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    in_array = array_parent is not None or array_task is not None
    selector = job
    if in_array:
        require(isinstance(array_parent, str) and re.fullmatch(r"[1-9][0-9]*", array_parent)
                and isinstance(array_task, str) and re.fullmatch(r"0|[1-9][0-9]*", array_task),
                "explicit_array_allocation_identity_required")
        # The array master's numeric JobId can also belong to one task. A bare
        # lookup then returns every task, not just that numeric allocation.
        selector = f"{array_parent}_{array_task}"
    result = subprocess.run(["scontrol", "show", "job", selector, "-o"], check=True,
                            capture_output=True, text=True, timeout=30)
    records = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    require(len(records) == 1, "actual_scheduler_single_job_record_required")
    pairs = [token.split("=", 1) for token in records[0].split() if "=" in token]
    require(sum(key == "JobId" for key, _ in pairs) == 1
            and sum(key == "ArrayJobId" for key, _ in pairs) == int(in_array)
            and sum(key == "ArrayTaskId" for key, _ in pairs) == int(in_array),
            "actual_scheduler_unambiguous_job_identity_required")
    fields = dict(pairs)
    tres = dict(token.split("=", 1) for token in fields.get("AllocTRES", "").split(",") if "=" in token)
    require(fields.get("JobId") == job and fields.get("JobState") == "RUNNING"
            and (not in_array or (fields["ArrayJobId"] == array_parent and fields["ArrayTaskId"] == array_task))
            and fields.get("NumCPUs") == "1" and fields.get("NumNodes") == "1"
            and fields.get("NodeList") == host and tres.get("cpu") == "1"
            and tres.get("mem") in ("7G", "7168M") and not any(key.startswith("gres/gpu") for key in tres),
            "actual_work_item_allocation_mismatch")
    seconds = prefix.seconds(fields["TimeLimit"])
    # scontrol reports the current allocation end in this shell's local time.
    # Do not require an optional job environment snapshot to be present/current.
    try:
        end = datetime.fromisoformat(fields["EndTime"]).timestamp()
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise Stop("actual_scheduler_end_time_required") from exc
    require(0 < seconds <= resources["minutes"] * 60 and end > time.time() + 300,
            "authorized_walltime_and_closure_reserve_required")
    return {"job_id": job, "hostname": host, "allocated_cpus": 1, "memory_gib": 7,
            "gpus": 0, "time_limit_seconds": seconds, "job_end_epoch": end}


class WorkBudget(prefix.Budget):
    def __init__(self, allocation, resources, max_seconds=None):
        self.allocation = allocation
        self.resources = validate_resources(resources)
        internal = resources["minutes"] * 60 - 300
        if max_seconds is not None:
            require(type(max_seconds) is int and 0 < max_seconds <= internal,
                    "bounded_work_item_internal_seconds_required")
            internal = max_seconds
        super().__init__(allocation, internal)

    def check(self):
        require(time.monotonic() < self.deadline, "work_item_deadline_reached")
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 6.5 * 1024**3,
                "worker_RSS_closure_threshold")

    def room(self, extra):
        require(sum(self.files.values()) + extra <= self.resources["durable_bytes"],
                "work_item_durable_byte_cap")
