"""Separately admitted one-attempt CUDA measurement; never production fitting.

Only stdlib imports precede active-profile and actual allocation verification.
Owned CUDA initialization then verifies the visible device and allocator cap.
Scientific inputs are opened only after the exclusive claim. The parent batch
must first pass the selected GPU loss/adjoint fixture. This entrypoint creates
no grant, scheduler job, retry, model publication or evaluation access.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path
import resource
import socket
import time

from run_content_measurement import (
    AdmissionError, _require, _read_text, _memory_bytes, _same, _bytes_under, _time_seconds,
    THREADS,
)

BASE = Path(__file__).resolve().parents[3]
PROFILE = BASE / "outputs/content_cuda_measurement_authorization.json"
RUN_DIR = BASE / "outputs/runs/content_resource_measurement/cuda001"
SCRATCH_NAME = "ep21_content_cuda_measurement001"
CACHE_PROFILE = BASE / "outputs/content_cuda_cache_authorization.json"
CACHE_RUN_DIR = BASE / "outputs/runs/content_resource_measurement/cuda_cache001"
CACHE_SCRATCH_NAME = "ep21_content_cuda_cache001"
CACHE_REPAIR_PROFILE = BASE / "outputs/content_cuda_cache_repair_authorization.json"
CACHE_REPAIR_RUN_DIR = BASE / "outputs/runs/content_resource_measurement/cuda_cache002"
CACHE_REPAIR_SCRATCH_NAME = "ep21_content_cuda_cache002"
STAGES = ("cuda001", "cache-001", "cache-002")
SOURCE, TARGET = "sub-03", "sub-01"
RESOURCES = {"cpus": 2, "memory_gib": 16, "minutes": 120, "gpus": 1,
    "deadline_seconds": 6300, "rss_limit_bytes": 16106127360,
    "minimum_device_bytes": 23622320128, "gpu_memory_limit_bytes": 21474836480,
    "durable_bytes": 33554432, "scratch_bytes": 268435456}
FIXED = {"schema_version": "ep21.content_cuda_measurement.v1",
    "operation": "measure_content_pair_cuda_four_updates", "jobs": 1,
    "partition": "gpu", "source": SOURCE, "target": TARGET,
    "updates": 4, "block_units": 256, "row_chunk_size": 512,
    "backend": "sparse", "device": "cuda:0", "feature_read_mode": "bounded_column_stripe",
    "feature_stripe_bytes": 67108864, "automatic_retry": False,
    "production_training": False, "model_publication": False, "amp": False, "tf32": False,
    "correction_report": "outputs/pipeline_conformance/converter_loss_correction_001/report.json"}
INPUT_SCOPE = {"normalization_participants": ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"],
    "source_training": SOURCE, "target_decoder": TARGET, "source_features": SOURCE,
    "target_response_matrix": False, "tau": False, "OOD": False}


@dataclass(frozen=True)
class MeasurementStage:
    name: str
    profile_path: Path
    run_dir: Path
    scratch_name: str
    allocation_start_deadline: bool

    @property
    def cached(self):
        return self.name in ("cache-001", "cache-002")

    @property
    def fixed(self):
        if self.cached:
            return {**{key: value for key, value in FIXED.items() if key != "feature_stripe_bytes"},
                "operation": "measure_content_pair_cuda_cached_four_updates",
                "feature_read_mode": "bounded_channel_two_pass_cache",
                "maximum_channel_bytes": 1073741824, "maximum_retained_bytes": 4294967296}
        return dict(FIXED)


def _stage(name="cuda001"):
    _require(name in STAGES, "unknown_CUDA_measurement_stage")
    if name == "cache-002":
        return MeasurementStage(name, CACHE_REPAIR_PROFILE, CACHE_REPAIR_RUN_DIR,
                                CACHE_REPAIR_SCRATCH_NAME, True)
    if name == "cache-001":
        return MeasurementStage(name, CACHE_PROFILE, CACHE_RUN_DIR, CACHE_SCRATCH_NAME, True)
    return MeasurementStage(name, PROFILE, RUN_DIR, SCRATCH_NAME, False)


def _profile(stage=None):
    stage = _stage() if stage is None else stage
    value = json.loads(_read_text(stage.profile_path))
    _require(isinstance(value, dict) and value.get("authorized") is True, "profile_not_activated")
    _require(isinstance(value.get("approval_source"), str) and bool(value["approval_source"].strip()),
             "separate_CUDA_measurement_approval_required")
    for key, expected in stage.fixed.items():
        _require(_same(value.get(key), expected), "CUDA_measurement_scope_changed")
    _require(_same(value.get("resources"), RESOURCES) and _same(value.get("input_scope"), INPUT_SCOPE),
             "CUDA_measurement_resources_or_roles_changed")
    return {**stage.fixed, "resources": dict(RESOURCES), "input_scope": dict(INPUT_SCOPE),
            "authorized": True, "approval_source": value["approval_source"]}


def _allocation(stage=None):
    stage = _stage() if stage is None else stage
    fields = dict(token.split("=", 1) for token in _read_text(stage.run_dir / "allocation.txt").split()
                  if "=" in token)
    host, job = socket.gethostname().split(".")[0], os.environ.get("SLURM_JOB_ID", "")
    _require(job.isdigit() and "-ln" not in host and fields.get("JobId") == job,
             "actual_compute_job_required")
    _require(fields.get("NodeList", "").split(".")[0] == host, "allocation_node_mismatch")
    for key, expected in {"JobState": "RUNNING", "NumCPUs": "2", "NumTasks": "1",
                          "NumNodes": "1", "Partition": "gpu"}.items():
        _require(fields.get(key) == expected, "allocation_resource_mismatch")
    _require(_time_seconds(fields.get("TimeLimit", "")) == 7200, "allocation_time_mismatch")
    _require(_memory_bytes(fields.get("MinMemoryNode", "")) == 16 * 1024**3,
             "allocation_memory_mismatch")
    tres = dict(item.split("=", 1) for item in fields.get("AllocTRES", "").split(",") if "=" in item)
    _require(tres.get("cpu") == "2" and tres.get("node") == "1"
             and _memory_bytes(tres.get("mem", "")) == 16 * 1024**3,
             "allocated_TRES_mismatch")
    _require(tres.get("gres/gpu") == "1"
             and all(value == "1" for key, value in tres.items() if key.startswith("gres/gpu:")),
             "exactly_one_allocated_GPU_required")
    _require(os.environ.get("SLURM_CPUS_PER_TASK") == "2"
             and os.environ.get("SLURM_MEM_PER_NODE") == "16384",
             "allocation_environment_mismatch")
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
    # The batch captures scontrol under TZ=UTC. Honor an explicit offset if present.
    end_time = datetime.fromisoformat(fields.get("EndTime", ""))
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    end = end_time.timestamp()
    result = {"job_id": job, "hostname": host, "job_end_epoch": end,
        "cpus": 2, "memory_gib": 16, "partition": "gpu", "minutes": 120, "gpus": 1}
    now = time.time()
    if stage.allocation_start_deadline:
        start_time = datetime.fromisoformat(fields.get("StartTime", ""))
        _require(start_time.tzinfo is None or start_time.utcoffset().total_seconds() == 0,
                 "UTC_allocation_start_required")
        _require(end_time.utcoffset().total_seconds() == 0, "UTC_allocation_end_required")
        start = start_time.replace(tzinfo=timezone.utc).timestamp()
        _require(start <= now < end and start < end, "invalid_actual_allocation_interval")
        deadline = min(start + RESOURCES["deadline_seconds"], end - 60)
        _require(deadline - now >= 1, "insufficient_actual_remaining_allocation")
        result.update(job_start_epoch=start, deadline_epoch=deadline)
    else:
        _require(end - now >= 6330, "insufficient_actual_remaining_allocation")
    return result


def _scratch(stage=None):
    stage = _stage() if stage is None else stage
    parent = os.environ.get("L_SCRATCH_JOB", "")
    _require(bool(parent) and Path(parent).is_absolute(), "node_local_job_scratch_required")
    path = Path(parent) / stage.scratch_name
    _require(os.environ.get("TMPDIR") == str(path) and path.is_dir() and not path.is_symlink(),
             "fixed_CUDA_measurement_scratch_required")
    return path


@dataclass(frozen=True)
class MeasurementContext:
    profile: dict
    allocation: dict
    scratch: Path
    stage: MeasurementStage


class MeasurementBudget:
    """Host and PyTorch allocator limits; not a meter of all physical GPU use."""
    def __init__(self, context):
        self.context = context
        self.allocation = dict(context.allocation)
        self.absolute_deadline = (self.allocation["deadline_epoch"]
            if context.stage.allocation_start_deadline else self.allocation["job_end_epoch"])
        remaining = (self.absolute_deadline - time.time()
            if context.stage.allocation_start_deadline else RESOURCES["deadline_seconds"])
        self.deadline = time.monotonic() + remaining
        self.torch = None
        self.device_info = None
        self.check()

    def check(self):
        _require(os.environ.get("SLURM_JOB_ID") == self.allocation["job_id"]
                 and socket.gethostname().split(".")[0] == self.allocation["hostname"],
                 "allocation_identity_changed")
        if time.monotonic() >= self.deadline or time.time() >= self.absolute_deadline:
            raise TimeoutError("CUDA_measurement_deadline")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > RESOURCES["rss_limit_bytes"]:
            raise MemoryError("CUDA_measurement_host_RSS_limit")
        if self.torch is not None:
            cuda = self.torch.cuda
            _require(cuda.current_device() == 0, "fixed_cuda_zero_device_required")
            for counter in (cuda.memory_allocated, cuda.memory_reserved,
                            cuda.max_memory_allocated, cuda.max_memory_reserved):
                if counter(0) > RESOURCES["gpu_memory_limit_bytes"]:
                    raise MemoryError("CUDA_measurement_allocator_memory_limit")

    def check_storage(self):
        _bytes_under(self.context.stage.run_dir, RESOURCES["durable_bytes"])
        _bytes_under(self.context.scratch, RESOURCES["scratch_bytes"])

    def remaining_seconds(self):
        self.check()
        remaining = int(min(self.deadline - time.monotonic(), self.absolute_deadline - time.time()))
        _require(remaining > 0, "insufficient_actual_remaining_allocation")
        return remaining


def _initialize_cuda(budget):
    # Called only after active fixed profile, actual job, host budget and storage.
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
             "exactly_one_available_CUDA_device_required")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    total = int(properties.total_memory)
    _require(total >= RESOURCES["minimum_device_bytes"], "CUDA_device_memory_below_admitted_minimum")
    torch.cuda.set_per_process_memory_fraction(RESOURCES["gpu_memory_limit_bytes"] / total, 0)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _require(not torch.is_autocast_enabled(), "CUDA_autocast_forbidden")
    budget.torch = torch
    budget.device_info = {"device": "cuda:0", "name": str(properties.name),
        "total_memory_bytes": total, "visible_device_count": 1,
        "pytorch_allocator_limit_bytes": RESOURCES["gpu_memory_limit_bytes"],
        "pytorch_allocator_fraction": RESOURCES["gpu_memory_limit_bytes"] / total,
        "allocator_limit_is_not_total_physical_GPU_usage": True}
    budget.check()


def admit(stage_name="cuda001"):
    # Inactive profiles cannot inspect allocation, initialize CUDA or read inputs.
    stage = _stage(stage_name)
    profile = _profile(stage)
    context = MeasurementContext(profile, _allocation(stage), _scratch(stage), stage)
    budget = MeasurementBudget(context)
    budget.check_storage()
    _initialize_cuda(budget)
    return context, budget


def resolver(context, *, operation, source, target, budget, execution_options):
    _require(isinstance(context, MeasurementContext) and isinstance(budget, MeasurementBudget)
             and budget.context is context and budget.allocation == context.allocation,
             "same_CUDA_measurement_context_required")
    _require(context.stage == _stage(context.stage.name)
             and _profile(context.stage) == context.profile, "CUDA_measurement_admission_changed")
    budget.check()
    _require(source == SOURCE and target == TARGET, "fixed_CUDA_measurement_pair_required")
    if operation == "bind_content_inputs":
        _require(execution_options is None, "binding_has_no_execution_options")
    elif operation == context.stage.fixed["operation"]:
        expected = {"device": "cuda:0", "block_units": 256, "row_chunk_size": 512,
            "backend": "sparse", "updates": 4, "source_rows": 4712, "layer_count": 19,
            "final_path": context.stage.run_dir / "reserved_final_model.pt", "maximum_bytes": 1}
        if context.stage.cached:
            expected.update({key: context.stage.fixed[key] for key in
                ("feature_read_mode", "maximum_channel_bytes", "maximum_retained_bytes")})
        _require(execution_options is not None and _same(dict(execution_options), expected),
                 "fixed_CUDA_measurement_options_required")
    else:
        raise AdmissionError("operation_not_authorized")


def _load_operations(stage=None):
    stage = _stage() if stage is None else stage
    import empirical_content_inputs
    import content_pair_cuda_measurement
    if stage.cached:
        measure = partial(content_pair_cuda_measurement.measure_content_pair_cuda,
                          feature_read_mode=stage.fixed["feature_read_mode"])
    else:
        import decoder_interface
        _require(decoder_interface.FEATURE_STRIPE_BYTES == FIXED["feature_stripe_bytes"],
                 "fixed_column_stripe_reader_required")
        measure = content_pair_cuda_measurement.measure_content_pair_cuda
    return empirical_content_inputs.bind_content_inputs, measure


def _publish(path, record, stage=None):
    stage = _stage() if stage is None else stage
    payload = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    used = _bytes_under(stage.run_dir, RESOURCES["durable_bytes"])
    _require(used + len(payload) <= RESOURCES["durable_bytes"], "operational_output_limit")
    with open(path, "xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def run_once(stage_name="cuda001"):
    context, budget = admit(stage_name)
    stage = context.stage
    _publish(stage.run_dir / "claim.json", {"operation": stage.fixed["operation"], "stage": stage.name,
        "allocation": context.allocation, "device": budget.device_info,
        "approval_source": context.profile["approval_source"],
        "automatic_retry": False, "production_training": False, "model_publication": False}, stage)
    report, interrupted = None, False
    try:
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
            bind, measure = _load_operations(stage)
            def prepare_inputs():
                return bind(context, require_admitted_context=resolver, source=SOURCE,
                            target=TARGET, budget=budget)
            report = measure(context, require_admitted_context=resolver, prepare_inputs=prepare_inputs,
                budget=budget, reserved_final_path=stage.run_dir / "reserved_final_model.pt")
        budget.check_storage()
        _require(isinstance(report, dict) and report.get("status") in ("complete_measurement", "incomplete")
                 and report.get("model_published") is False and report.get("production_fit_completed") is False,
                 "operational_CUDA_measurement_report_required")
    except BaseException as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        if not isinstance(report, dict):
            report = {"kind": "ep21_content_pair_cuda_resource_measurement", "source": SOURCE,
                "target": TARGET, "completed_updates": None, "production_fit_completed": False,
                "model_published": False, "resumable": False,
                "allocation_identity": {key: context.allocation[key] for key in ("job_id", "hostname")}}
        # Preserve any confirmed completed-update count if only closeout failed.
        report.update(status="incomplete", error_type=type(error).__name__, failed_stage="CUDA_measurement_entrypoint")
    report.update(stage=stage.name, admitted_device=budget.device_info,
                  feature_read_mode=stage.fixed["feature_read_mode"])
    for key in ("feature_stripe_bytes", "maximum_channel_bytes", "maximum_retained_bytes"):
        if key in stage.fixed:
            report[key] = stage.fixed[key]
    _publish(stage.run_dir / "report.json", report, stage)
    return 130 if interrupted else (0 if report["status"] == "complete_measurement" else 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="cuda001")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check-admission", action="store_true")
    action.add_argument("--remaining-seconds", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.check_admission or args.remaining_seconds:
            context, budget = admit(args.stage)
            if args.remaining_seconds:
                print(budget.remaining_seconds())
                return 0
            print(json.dumps({"status": "CUDA_measurement_admitted", "job_id": context.allocation["job_id"],
                "stage": context.stage.name, "device": budget.device_info,
                "payload_read": False, "attempt_claimed": False}))
            return 0
        return run_once(args.stage)
    except BaseException as error:
        print(json.dumps({"status": "CUDA_measurement_entrypoint_refused", "error_type": type(error).__name__}))
        return 130 if isinstance(error, KeyboardInterrupt) else 2


if __name__ == "__main__":
    raise SystemExit(main())
