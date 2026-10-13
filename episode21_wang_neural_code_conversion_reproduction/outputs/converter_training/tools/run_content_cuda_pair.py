"""Dormant first full cached CUDA content pair; no automatic retry or resume.

An explicit production grant must select finite allocation and accounting
limits before any CUDA initialization or scientific input access. Only stdlib
imports and inert entrypoint helpers precede admission. Existing measurement
and brain jobs, grants and output paths are separate and remain unchanged.
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

from run_content_measurement import (
    AdmissionError, THREADS, _bytes_under, _memory_bytes, _read_text, _require, _same,
)
from run_content_pair import _bound_scope, _time_seconds
from run_content_cuda_measurement import _initialize_cuda


BASE = Path(__file__).resolve().parents[3]
PROFILE = BASE / "outputs/content_cuda_pair_training_authorization.json"
RUN_DIR = BASE / "outputs/runs/content_pair_training/cuda001"
FINAL = RUN_DIR / "final_content_pair.pt"
PROGRESS = RUN_DIR / "progress.jsonl"
PROGRESS_BYTES = 262144
SCRATCH_NAME = "ep21_content_cuda_pair001"
SOURCE, TARGET = "sub-03", "sub-01"
FIXED = {"schema_version": "ep21.content_cuda_pair_training.v1",
    "operation": "train_and_publish_cached_cuda_full_pair", "jobs": 1, "partition": "gpu",
    "source": SOURCE, "target": TARGET, "pair_index": 4,
    "training_images": 4712, "layers": 19, "iterations": 1024,
    "source_eligible_voxels": 12897, "target_eligible_voxels": 12294,
    "device": "cuda:0", "backend": "sparse", "block_units": 256, "row_chunk_size": 512,
    "device_name": "Tesla V100-SXM2-32GB",
    "scheduler_constraint": "GPU_SKU:V100_SXM2&GPU_MEM:32GB",
    "feature_read_mode": "bounded_channel_two_pass_cache",
    "maximum_channel_bytes": 1073741824, "maximum_retained_bytes": 4294967296,
    "amp": False, "tf32": False,
    "automatic_retry": False, "production_training": True, "model_publication": True, "scoring": False,
    "correction_report": "outputs/pipeline_conformance/converter_loss_correction_001/report.json"}
INPUT_SCOPE = {"normalization_participants": ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"],
    "source_training": SOURCE, "target_decoder": TARGET, "source_features": SOURCE,
    "target_response_matrix": False, "tau": False, "OOD": False}
FIXED_RESOURCES = {"cpus": 2, "memory_gib": 16, "gpus": 1,
    "rss_limit_bytes": 16106127360, "minimum_device_bytes": 23622320128,
    "gpu_memory_limit_bytes": 21474836480, "durable_bytes": 2147483648,
    "scratch_bytes": 268435456, "final_artifact_bytes": 1073741824}
RESOURCE_KEYS = set(FIXED_RESOURCES) | {
    "minutes", "deadline_seconds", "allocated_cpu_hours", "allocated_gpu_hours"}


def _profile():
    value = json.loads(_read_text(PROFILE))
    _require(isinstance(value, dict) and value.get("authorized") is True, "profile_not_activated")
    _require(isinstance(value.get("approval_source"), str) and bool(value["approval_source"].strip()),
             "separate_CUDA_production_selection_required")
    for key, expected in FIXED.items():
        _require(_same(value.get(key), expected), "CUDA_content_pair_scope_changed")
    _require(_same(value.get("input_scope"), INPUT_SCOPE), "CUDA_content_pair_roles_changed")
    resources = value.get("resources")
    _require(isinstance(resources, dict) and resources.keys() == RESOURCE_KEYS,
             "complete_explicit_CUDA_production_resources_required")
    for key, expected in FIXED_RESOURCES.items():
        _require(_same(resources[key], expected), "fixed_CUDA_production_resources_changed")
    for key in ("minutes", "deadline_seconds"):
        _require(type(resources[key]) is int and resources[key] > 0,
                 "positive_explicit_integer_runtime_required")
    _require(resources["minutes"] <= 2880, "observed_GPU_partition_two_day_limit")
    _require(resources["deadline_seconds"] + 60 <= resources["minutes"] * 60,
             "absolute_deadline_requires_closeout_time")
    for key, count in (("allocated_cpu_hours", 2), ("allocated_gpu_hours", 1)):
        hours = resources[key]
        _require(type(hours) in (int, float) and math.isfinite(hours) and hours > 0,
                 "finite_explicit_allocated_hours_required")
        _require(hours >= count * resources["minutes"] / 60,
                 "full_CPU_and_GPU_allocation_reservations_required")
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
    for key, expected in {"JobState": "RUNNING", "NumCPUs": "2", "NumTasks": "1",
                          "NumNodes": "1", "Partition": "gpu"}.items():
        _require(fields.get(key) == expected, "allocation_resource_mismatch")
    _require(_time_seconds(fields.get("TimeLimit", "")) == resources["minutes"] * 60,
             "allocation_time_mismatch")
    _require(_memory_bytes(fields.get("MinMemoryNode", "")) == 16 * 1024**3,
             "allocation_memory_mismatch")
    tres = dict(item.split("=", 1) for item in fields.get("AllocTRES", "").split(",") if "=" in item)
    _require(tres.get("cpu") == "2" and tres.get("node") == "1"
             and _memory_bytes(tres.get("mem", "")) == 16 * 1024**3, "allocated_TRES_mismatch")
    _require(tres.get("gres/gpu") == "1"
             and all(value == "1" for key, value in tres.items() if key.startswith("gres/gpu:")),
             "exactly_one_allocated_GPU_required")
    _require(os.environ.get("SLURM_CPUS_PER_TASK") == "2"
             and os.environ.get("SLURM_MEM_PER_NODE") == "16384", "allocation_environment_mismatch")
    for key in ("SLURM_NTASKS", "SLURM_NNODES"):
        _require(os.environ.get(key, "1") == "1", "allocation_environment_mismatch")
    _require(all(os.environ.get(key) == "1" for key in THREADS), "one_thread_environment_required")
    _require(os.environ.get("SLURM_GPUS_ON_NODE") == "1"
             and os.environ.get("SLURM_GPUS", "1") == "1", "one_GPU_environment_required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _require(bool(visible) and visible not in ("-1", "NoDevFiles") and "," not in visible,
             "one_visible_owned_GPU_required")
    for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
        value = os.environ.get(key, "")
        _require(not value or ("," not in value and value not in ("-1", "NoDevFiles")),
                 "one_GPU_identity_required")
    epochs = {}
    for key in ("StartTime", "EndTime"):
        parsed = datetime.fromisoformat(fields.get(key, ""))
        _require(parsed.tzinfo is None or parsed.utcoffset().total_seconds() == 0,
                 "UTC_allocation_times_required")
        epochs[key] = parsed.replace(tzinfo=timezone.utc).timestamp()
    start, end, now = epochs["StartTime"], epochs["EndTime"], time.time()
    _require(start <= now < end and start < end, "invalid_actual_allocation_interval")
    deadline = min(start + resources["deadline_seconds"], end - 60)
    _require(deadline - now >= 1, "insufficient_actual_remaining_allocation")
    return {"job_id": job, "hostname": host, "job_start_epoch": start, "job_end_epoch": end,
        "deadline_epoch": deadline, "cpus": 2, "memory_gib": 16, "partition": "gpu",
        "minutes": resources["minutes"], "gpus": 1}


def _scratch():
    parent = os.environ.get("L_SCRATCH_JOB", "")
    _require(bool(parent) and Path(parent).is_absolute(), "node_local_job_scratch_required")
    path = Path(parent) / SCRATCH_NAME
    _require(os.environ.get("TMPDIR") == str(path) and path.is_dir() and not path.is_symlink(),
             "fixed_CUDA_content_pair_scratch_required")
    return path


@dataclass(frozen=True)
class ContentCudaPairContext:
    profile: dict
    allocation: dict
    scratch: Path


class ContentCudaPairBudget:
    """One finite production parent; no decoder or measurement account reuse."""
    def __init__(self, context):
        self.context = context
        self.allocation = dict(context.allocation)
        self.resources = dict(context.profile["resources"])
        self.absolute_deadline = self.allocation["deadline_epoch"]
        self.deadline = time.monotonic() + self.absolute_deadline - time.time()
        self.torch = self.device_info = None
        self.check()

    def check(self):
        _require(os.environ.get("SLURM_JOB_ID") == self.allocation["job_id"]
                 and socket.gethostname().split(".")[0] == self.allocation["hostname"],
                 "allocation_identity_changed")
        if time.monotonic() >= self.deadline or time.time() >= self.absolute_deadline:
            raise TimeoutError("CUDA_content_pair_deadline")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > self.resources["rss_limit_bytes"]:
            raise MemoryError("CUDA_content_pair_host_RSS_limit")
        if self.torch is not None:
            cuda = self.torch.cuda
            _require(cuda.current_device() == 0, "fixed_cuda_zero_device_required")
            for counter in (cuda.memory_allocated, cuda.memory_reserved,
                            cuda.max_memory_allocated, cuda.max_memory_reserved):
                if counter(0) > self.resources["gpu_memory_limit_bytes"]:
                    raise MemoryError("CUDA_content_pair_allocator_memory_limit")

    def check_storage(self):
        _bytes_under(RUN_DIR, self.resources["durable_bytes"])
        _bytes_under(self.context.scratch, self.resources["scratch_bytes"])

    def remaining_seconds(self):
        self.check()
        remaining = int(min(self.deadline - time.monotonic(), self.absolute_deadline - time.time()))
        _require(remaining > 0, "insufficient_actual_remaining_allocation")
        return remaining


def admit():
    # Inactive or incomplete profiles stop before actual allocation or imports.
    profile = _profile()
    context = ContentCudaPairContext(profile, _allocation(profile), _scratch())
    budget = ContentCudaPairBudget(context)
    budget.check_storage()
    # Reuses the measurement helper only for identical fixed device caps and
    # float32/TF32/thread settings, never its context, grant, clock or operation.
    _initialize_cuda(budget)
    _require(budget.device_info["name"] == FIXED["device_name"],
             "measured_V100_SXM2_32GB_device_required")
    return context, budget


def _training_options():
    return {"device": "cuda:0", "block_units": 256, "row_chunk_size": 512,
            "backend": "sparse", "final_path": FINAL, "maximum_bytes": 1073741824}


def _execution_options():
    return {**_training_options(), **{key: FIXED[key] for key in
        ("feature_read_mode", "maximum_channel_bytes", "maximum_retained_bytes")}}


def resolver(context, *, operation, source, target, budget, execution_options):
    _require(isinstance(context, ContentCudaPairContext) and isinstance(budget, ContentCudaPairBudget)
             and budget.context is context and budget.allocation == context.allocation,
             "same_CUDA_content_pair_context_required")
    _require(_profile() == context.profile and budget.resources == context.profile["resources"],
             "CUDA_content_pair_admission_changed")
    budget.check()
    budget.check_storage()
    _require(source == SOURCE and target == TARGET, "fixed_CUDA_content_pair_required")
    if operation == "bind_content_inputs":
        _require(execution_options is None, "binding_has_no_execution_options")
    elif operation == FIXED["operation"]:
        _require(execution_options is not None and _same(dict(execution_options), _execution_options()),
                 "fixed_CUDA_content_pair_publication_options_required")
    else:
        raise AdmissionError("operation_not_authorized")


def _load_operations():
    from empirical_content_inputs import bind_content_inputs
    from content_pair_cuda_training import train_and_publish_cached_cuda_full_pair
    import numpy as np
    import torch
    return bind_content_inputs, train_and_publish_cached_cuda_full_pair, {
        "python": sys.version.split()[0], "numpy": str(np.__version__), "torch": str(torch.__version__)}


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
        "admitted_device": budget.device_info, "approval_source": context.profile["approval_source"],
        "automatic_retry": False}, budget)
    stage, interrupted = "load_operations", False
    started_wall, started_cpu = time.monotonic(), time.process_time()
    report = {"kind": "ep21_content_cuda_pair_operational_report", "source": SOURCE, "target": TARGET,
        "pair_index": 4, "requested_updates": 1024, "training_images": 4712, "layers": 19,
        "device": "cuda:0", "backend": "sparse", "feature_read_mode": FIXED["feature_read_mode"],
        "maximum_channel_bytes": FIXED["maximum_channel_bytes"],
        "maximum_retained_bytes": FIXED["maximum_retained_bytes"],
        "observed_completed_updates": 0, "completed_updates": None,
        "model_published": False, "published_path": None, "resumable": False,
        "automatic_retry": False, "runtime": None, "admitted_device": budget.device_info,
        "allocation_identity": {key: context.allocation[key] for key in ("job_id", "hostname")}}

    def on_iteration(event):
        # Never serialize the supplied event: it also contains private losses.
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
            result = train(bound, on_iteration=on_iteration, **_training_options())
            completion = result.completion
            _require(result.source == SOURCE and result.target == TARGET and type(result.pair_index) is int
                     and result.pair_index == 4 and result.path == FINAL
                     and type(completion.iterations) is int and completion.iterations == 1024
                     and type(completion.rows) is int and completion.rows == 4712
                     and completion.backend == "sparse" and report["observed_completed_updates"] == 1024
                     and FINAL.is_file(), "complete_fixed_CUDA_content_pair_publication_required")
            report.update(completed_updates=1024, model_published=True, published_path=str(FINAL))
            stage = "closeout"
            budget.check_storage()
        report.update(status="complete_content_pair")
    except BaseException as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        report.update(status="incomplete", error_type=type(error).__name__, failed_stage=stage)
        if stage == "fit_and_publish":
            report["model_published"] = None
    report.update(elapsed_wall_seconds=time.monotonic() - started_wall,
        process_cpu_seconds=time.process_time() - started_cpu,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    _publish(RUN_DIR / "report.json", report, budget)
    return 130 if interrupted else 0 if report["status"] == "complete_content_pair" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check-admission", action="store_true")
    action.add_argument("--remaining-seconds", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.check_admission or args.remaining_seconds:
            context, budget = admit()
            if args.remaining_seconds:
                print(budget.remaining_seconds())
                return 0
            print(json.dumps({"status": "CUDA_content_pair_admitted", "job_id": context.allocation["job_id"],
                "device": budget.device_info, "payload_read": False, "attempt_claimed": False}))
            return 0
        return run_once()
    except BaseException as error:
        print(json.dumps({"status": "CUDA_content_pair_entrypoint_refused", "error_type": type(error).__name__}))
        return 130 if isinstance(error, KeyboardInterrupt) else 2


if __name__ == "__main__":
    raise SystemExit(main())
