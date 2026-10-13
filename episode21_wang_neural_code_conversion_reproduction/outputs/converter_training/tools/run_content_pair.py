"""Inactive first full content-pair entrypoint; no resume or automatic retry.

The separately selected grant must supply measured finite resource limits.
Only stdlib imports precede admission and the exclusive attempt claim.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import socket
import sys
import time

from run_brain_pair import _time_seconds
from run_content_measurement import (
    AdmissionError, THREADS, _bytes_under, _memory_bytes, _read_text, _require, _same,
)

BASE = Path(__file__).resolve().parents[3]
PROFILE = BASE / "outputs/content_pair_training_authorization.json"
RUN_DIR = BASE / "outputs/runs/content_pair_training/attempt001"
FINAL = RUN_DIR / "final_content_pair.pt"
PROGRESS = RUN_DIR / "progress.jsonl"
PROGRESS_BYTES = 262144
SOURCE, TARGET = "sub-03", "sub-01"
FIXED = {"schema_version": "ep21.content_pair_training.v1", "operation": "train_and_publish_full_pair",
    "jobs": 1, "partition": "russpold", "source": SOURCE, "target": TARGET, "pair_index": 4,
    "training_images": 4712, "layers": 19, "iterations": 1024,
    "source_eligible_voxels": 12897, "target_eligible_voxels": 12294,
    "device": "cpu", "backend": "sparse", "block_units": 256, "row_chunk_size": 512,
    "feature_read_mode": "bounded_column_stripe", "feature_stripe_bytes": 67108864,
    "automatic_retry": False, "production_training": True, "model_publication": True, "scoring": False,
    "correction_report": "outputs/pipeline_conformance/converter_loss_correction_001/report.json"}
INPUT_SCOPE = {"normalization_participants": ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"],
    "source_training": SOURCE, "target_decoder": TARGET, "source_features": SOURCE,
    "target_response_matrix": False, "tau": False, "OOD": False}
FIXED_RESOURCES = {"gpus": 0, "durable_bytes": 2147483648,
    "scratch_bytes": 268435456, "final_artifact_bytes": 1073741824}
RESOURCE_KEYS = set(FIXED_RESOURCES) | {
    "cpus", "memory_gib", "minutes", "deadline_seconds", "rss_limit_bytes", "allocated_cpu_hours"}


def _profile():
    value = json.loads(_read_text(PROFILE))
    _require(isinstance(value, dict) and value.get("authorized") is True, "profile_not_activated")
    _require(isinstance(value.get("approval_source"), str) and bool(value["approval_source"].strip()),
             "separate_content_pair_selection_required")
    for key, expected in FIXED.items():
        _require(_same(value.get(key), expected), "content_pair_scope_changed")
    _require(_same(value.get("input_scope"), INPUT_SCOPE), "content_pair_roles_changed")
    resources = value.get("resources")
    _require(isinstance(resources, dict) and resources.keys() == RESOURCE_KEYS,
             "complete_explicit_content_resources_required")
    for key, expected in FIXED_RESOURCES.items():
        _require(_same(resources[key], expected), "content_storage_or_GPU_scope_changed")
    for key in ("cpus", "memory_gib", "minutes", "deadline_seconds", "rss_limit_bytes"):
        _require(type(resources[key]) is int and resources[key] > 0,
                 "positive_explicit_integer_resources_required")
    hours = resources["allocated_cpu_hours"]
    _require(type(hours) in (int, float) and math.isfinite(hours) and hours > 0,
             "finite_explicit_allocated_CPU_hours_required")
    _require(resources["minutes"] <= 10080, "observed_partition_seven_day_limit")
    _require(resources["deadline_seconds"] + 30 <= resources["minutes"] * 60,
             "deadline_requires_closeout_time")
    _require(resources["rss_limit_bytes"] <= resources["memory_gib"] * 1024**3,
             "RSS_limit_exceeds_allocation")
    _require(hours >= resources["cpus"] * resources["minutes"] / 60,
             "full_allocated_CPU_hours_reservation_required")
    return {**FIXED, "input_scope": dict(INPUT_SCOPE), "resources": dict(resources),
            "authorized": True, "approval_source": value["approval_source"]}


def _allocation(profile):
    resources = profile["resources"]
    fields = dict(token.split("=", 1) for token in _read_text(RUN_DIR / "allocation.txt").split()
                  if "=" in token)
    host, job = socket.gethostname().split(".")[0], os.environ.get("SLURM_JOB_ID", "")
    _require(job.isdigit() and "-ln" not in host and fields.get("JobId") == job,
             "actual_compute_job_required")
    _require(fields.get("NodeList", "").split(".")[0] == host, "allocation_node_mismatch")
    for key, expected in {"JobState": "RUNNING", "NumCPUs": str(resources["cpus"]),
                          "NumTasks": "1", "NumNodes": "1", "Partition": FIXED["partition"]}.items():
        _require(fields.get(key) == expected, "allocation_resource_mismatch")
    _require(_time_seconds(fields.get("TimeLimit", "")) == resources["minutes"] * 60,
             "allocation_time_mismatch")
    memory = resources["memory_gib"] * 1024**3
    _require(_memory_bytes(fields.get("MinMemoryNode", "")) == memory, "allocation_memory_mismatch")
    tres = dict(item.split("=", 1) for item in fields.get("AllocTRES", "").split(",") if "=" in item)
    _require(tres.get("cpu") == str(resources["cpus"]) and tres.get("node") == "1"
             and _memory_bytes(tres.get("mem", "")) == memory, "allocated_TRES_mismatch")
    _require(not any("gpu" in key.lower() for key in tres), "gpu_allocation_forbidden")
    for key, item in fields.items():
        if key.startswith(("TresPer", "Gres", "GRES")):
            _require("gpu" not in item.lower(), "gpu_allocation_forbidden")
    _require(os.environ.get("SLURM_CPUS_PER_TASK") == str(resources["cpus"])
             and os.environ.get("SLURM_MEM_PER_NODE") == str(resources["memory_gib"] * 1024),
             "allocation_environment_mismatch")
    for key in ("SLURM_NTASKS", "SLURM_NNODES"):
        _require(os.environ.get(key, "1") == "1", "allocation_environment_mismatch")
    _require(all(os.environ.get(key) == "1" for key in THREADS), "one_thread_environment_required")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
             and all(os.environ.get(key, "") in ("", "0") for key in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE"))
             and all(not os.environ.get(key, "") for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS")),
             "cpu_only_environment_required")
    end_value = datetime.fromisoformat(fields.get("EndTime", ""))
    _require(end_value.tzinfo is None or end_value.utcoffset().total_seconds() == 0,
             "UTC_allocation_end_required")
    end = end_value.replace(tzinfo=timezone.utc).timestamp()
    _require(end - time.time() >= resources["deadline_seconds"] + 30,
             "insufficient_actual_remaining_allocation")
    return {"job_id": job, "hostname": host, "job_end_epoch": end,
            "cpus": resources["cpus"], "memory_gib": resources["memory_gib"],
            "partition": FIXED["partition"], "minutes": resources["minutes"], "gpus": 0}


def _scratch():
    parent = os.environ.get("L_SCRATCH_JOB", "")
    _require(bool(parent) and Path(parent).is_absolute(), "node_local_job_scratch_required")
    expected = Path(parent) / "ep21_content_pair_attempt001"
    _require(os.environ.get("TMPDIR") == str(expected) and expected.is_dir()
             and not expected.is_symlink(), "fixed_content_pair_scratch_required")
    return expected


@dataclass(frozen=True)
class ContentPairContext:
    profile: dict
    allocation: dict
    scratch: Path


class ContentPairBudget:
    """Separate finite production reservation, with no decoder/measurement balance."""
    def __init__(self, context):
        self.context = context
        self.allocation = dict(context.allocation)
        self.resources = dict(context.profile["resources"])
        self.deadline = time.monotonic() + self.resources["deadline_seconds"]
        self.check()

    def check(self):
        _require(os.environ.get("SLURM_JOB_ID") == self.allocation["job_id"]
                 and socket.gethostname().split(".")[0] == self.allocation["hostname"],
                 "allocation_identity_changed")
        if time.monotonic() >= self.deadline or time.time() >= self.allocation["job_end_epoch"]:
            raise TimeoutError("content_pair_deadline")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > self.resources["rss_limit_bytes"]:
            raise MemoryError("content_pair_RSS_limit")

    def check_storage(self):
        _bytes_under(RUN_DIR, self.resources["durable_bytes"])
        _bytes_under(self.context.scratch, self.resources["scratch_bytes"])


def admit():
    profile = _profile()  # Inactive or unset resources stop before allocation/imports/inputs.
    context = ContentPairContext(profile, _allocation(profile), _scratch())
    budget = ContentPairBudget(context)
    budget.check_storage()
    return context, budget


def _execution_options(profile):
    return {"device": FIXED["device"], "block_units": FIXED["block_units"],
            "row_chunk_size": FIXED["row_chunk_size"], "backend": FIXED["backend"],
            "final_path": FINAL, "maximum_bytes": profile["resources"]["final_artifact_bytes"]}


def resolver(context, *, operation, source, target, budget, execution_options):
    _require(isinstance(context, ContentPairContext) and isinstance(budget, ContentPairBudget)
             and budget.context is context and budget.allocation == context.allocation,
             "same_content_pair_context_required")
    _require(_profile() == context.profile and budget.resources == context.profile["resources"],
             "content_pair_admission_changed")
    budget.check()
    budget.check_storage()
    _require(source == SOURCE and target == TARGET, "fixed_content_pair_required")
    if operation == "bind_content_inputs":
        _require(execution_options is None, "binding_has_no_execution_options")
    elif operation == FIXED["operation"]:
        _require(execution_options is not None and _same(dict(execution_options), _execution_options(context.profile)),
                 "fixed_content_pair_publication_options_required")
    else:
        raise AdmissionError("operation_not_authorized")


def _load_operations():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from empirical_content_inputs import bind_content_inputs, train_and_publish_full_pair
    import numpy as np
    return bind_content_inputs, train_and_publish_full_pair, {
        "python": sys.version.split()[0], "numpy": str(np.__version__), "torch": str(torch.__version__)}


def _bound_scope(bound):
    _require(bound.source.axis.participant == SOURCE and bound.target == TARGET
             and len(bound.source.axis.eligible_native_ids) == FIXED["source_eligible_voxels"]
             and len(bound.pair.target_axis.eligible_native_ids) == FIXED["target_eligible_voxels"]
             and len(bound.source.image_ids) == FIXED["training_images"]
             and bound.source.normalized_X.shape == (4712, 12897), "fixed_full_content_input_scope_required")
    _require(len(bound.panel.axes) == 5 and len(bound.panel.initializations) == 20,
             "complete_canonical_panel_initialization_required")
    initial = bound.panel.initializations[FIXED["pair_index"]]
    _require(initial.pair_index == 4 and initial.source == SOURCE and initial.target == TARGET
             and initial.input_width == 12897 and initial.output_width == 12294,
             "canonical_pair_four_initialization_required")
    channels = bound.panel.channel_schedule.channels_by_layer
    _require(len(channels) == FIXED["layers"] and all(len(values) == FIXED["iterations"] for values in channels),
             "complete_shared_19_layer_1024_step_schedule_required")


def _publish(path, record, budget):
    payload = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _require(len(payload) <= 65536, "operational_record_size_limit")
    cap = budget.resources["durable_bytes"]
    _require(_bytes_under(RUN_DIR, cap) + len(payload) <= cap, "operational_output_limit")
    with open(path, "xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def run_once():
    context, budget = admit()
    for path in (FINAL, FINAL.with_name(FINAL.name + ".partial"), RUN_DIR / "report.json", PROGRESS):
        _require(not os.path.lexists(path), "existing_attempt_output_no_retry")
    _publish(RUN_DIR / "claim.json", {"operation": FIXED["operation"], "source": SOURCE, "target": TARGET,
        "pair_index": 4, "allocation": context.allocation, "resources": budget.resources,
        "approval_source": context.profile["approval_source"], "automatic_retry": False}, budget)
    stage, interrupted = "load_operations", False
    started_wall, started_cpu = time.monotonic(), time.process_time()
    report = {"kind": "ep21_content_pair_operational_report", "source": SOURCE, "target": TARGET,
        "pair_index": 4, "requested_updates": 1024, "training_images": 4712, "layers": 19,
        "backend": "sparse", "observed_completed_updates": 0, "completed_updates": None,
        "model_published": False, "published_path": None, "resumable": False,
        "automatic_retry": False, "runtime": None,
        "allocation_identity": {key: context.allocation[key] for key in ("job_id", "hostname")}}

    def on_iteration(event):
        # The existing event also contains loss/rate; read neither into this report.
        completed = event["completed_iterations"]
        _require(type(completed) is int and completed == report["observed_completed_updates"] + 1
                 and completed <= 1024, "consecutive_completed_update_count_required")
        report["observed_completed_updates"] = completed
        payload = (json.dumps({"completed_update": completed,
            "elapsed_wall_seconds": time.monotonic() - started_wall}, allow_nan=False) + "\n").encode("utf-8")
        _require(progress.tell() + len(payload) <= PROGRESS_BYTES, "bounded_operational_progress_required")
        cap = budget.resources["durable_bytes"]
        _require(_bytes_under(RUN_DIR, cap) + len(payload) <= cap, "operational_output_limit")
        progress.write(payload)
        progress.flush()
        budget.check()

    try:
        with open(PROGRESS, "xb") as progress, open(os.devnull, "w") as quiet, \
                redirect_stdout(quiet), redirect_stderr(quiet):
            bind, train, runtime = _load_operations()
            report["runtime"] = runtime
            stage = "bind_inputs"
            bound = bind(context, require_admitted_context=resolver, source=SOURCE, target=TARGET, budget=budget)
            _bound_scope(bound)
            stage = "fit_and_publish"
            result = train(bound, on_iteration=on_iteration, **_execution_options(context.profile))
            completion = result.completion
            _require(result.source == SOURCE and result.target == TARGET and type(result.pair_index) is int
                     and result.pair_index == 4 and result.path == FINAL
                     and type(completion.iterations) is int and completion.iterations == 1024
                     and type(completion.rows) is int and completion.rows == 4712
                     and completion.backend == "sparse" and FINAL.is_file(),
                     "complete_fixed_content_pair_publication_required")
            report.update(completed_updates=1024, model_published=True, published_path=str(FINAL))
            stage = "closeout"
            budget.check_storage()
        report.update(status="complete_content_pair")
    except BaseException as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        report.update(status="incomplete", error_type=type(exc).__name__, failed_stage=stage)
        if stage == "fit_and_publish":
            report["model_published"] = None
    # Never serialize completion or callback events: both contain private losses.
    report.update(elapsed_wall_seconds=time.monotonic() - started_wall,
        process_cpu_seconds=time.process_time() - started_cpu,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    _publish(RUN_DIR / "report.json", report, budget)
    return 130 if interrupted else 0 if report["status"] == "complete_content_pair" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-admission", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.check_admission:
            context, _ = admit()
            print(json.dumps({"status": "content_pair_admitted", "job_id": context.allocation["job_id"],
                              "payload_read": False, "attempt_claimed": False}))
            return 0
        return run_once()
    except BaseException as exc:
        print(json.dumps({"status": "content_pair_entrypoint_refused", "error_type": type(exc).__name__}))
        return 130 if isinstance(exc, KeyboardInterrupt) else 2


if __name__ == "__main__":
    raise SystemExit(main())
