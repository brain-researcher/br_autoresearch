"""Separate finite full-fit entrypoint for sixteen original CUDA pairs.

Only sixteen literal stages exist. An active per-stage grant and actual allocation
precede CUDA initialization and scientific imports. The running pair-4 launcher
is reused only for its inert helpers and generic budget checks, without global
mutation. Shared checks may be reused by the coordinator; this module never
submits, retries, resumes, scores, or advances to another pair.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time

import run_content_cuda_pair as core


BASE = Path(__file__).resolve().parents[3]
AdmissionError, THREADS = core.AdmissionError, core.THREADS
_require, _same, _read_text = core._require, core._same, core._read_text
_bytes_under, _memory_bytes, _time_seconds = core._bytes_under, core._memory_bytes, core._time_seconds
_initialize_cuda = core._initialize_cuda
OPERATION = "train_and_publish_cached_cuda_remaining_pair"
PROGRESS_BYTES = 262144
RESOURCES = {"cpus": 2, "memory_gib": 16, "minutes": 2880, "gpus": 1,
    "deadline_seconds": 169200, "allocated_cpu_hours": 96, "allocated_gpu_hours": 48,
    "rss_limit_bytes": 16106127360, "minimum_device_bytes": 23622320128,
    "gpu_memory_limit_bytes": 21474836480, "durable_bytes": 2147483648,
    "scratch_bytes": 1073741824, "final_artifact_bytes": 1073741824}
FIXED = {"schema_version": "ep21.content_cuda_remaining_training.v1", "operation": OPERATION,
    "jobs": 1, "partition": "gpu",
    "training_images": 4712, "layers": 19, "iterations": 1024,
    "device": "cuda:0", "backend": "sparse", "block_units": 256, "row_chunk_size": 512,
    "device_name": "Tesla V100-SXM2-32GB",
    "scheduler_constraint": "GPU_SKU:V100_SXM2&GPU_MEM:32GB",
    "feature_read_mode": "bounded_channel_two_pass_cache",
    "maximum_channel_bytes": 1073741824, "maximum_retained_bytes": 4294967296,
    "amp": False, "tf32": False, "automatic_retry": False,
    "production_training": True, "model_publication": True, "scoring": False,
    "correction_report": "outputs/pipeline_conformance/converter_loss_correction_001/report.json"}


@dataclass(frozen=True)
class CudaRemainingStage:
    name: str
    source: str
    target: str
    pair_index: int
    source_width: int
    target_width: int

    @property
    def profile_path(self):
        return BASE / f"outputs/content_cuda_remaining_{self.pair_index:03d}_authorization.json"

    @property
    def run_dir(self):
        return BASE / ("outputs/runs/content_pair_training/"
                       f"sub{self.source[-2:]}_to_sub{self.target[-2:]}_cuda001")

    @property
    def final_path(self):
        return self.run_dir / "final_content_pair.pt"

    @property
    def scratch_name(self):
        return f"ep21_content_cuda_remaining{self.pair_index:03d}"


STAGES = {
    "remaining-000": CudaRemainingStage("remaining-000", "sub-01", "sub-03", 0, 12294, 12897),
    "remaining-001": CudaRemainingStage("remaining-001", "sub-01", "sub-05", 1, 12294, 10891),
    "remaining-002": CudaRemainingStage("remaining-002", "sub-01", "sub-06", 2, 12294, 12023),
    "remaining-003": CudaRemainingStage("remaining-003", "sub-01", "sub-07", 3, 12294, 12057),
    "remaining-008": CudaRemainingStage("remaining-008", "sub-05", "sub-01", 8, 10891, 12294),
    "remaining-009": CudaRemainingStage("remaining-009", "sub-05", "sub-03", 9, 10891, 12897),
    "remaining-010": CudaRemainingStage("remaining-010", "sub-05", "sub-06", 10, 10891, 12023),
    "remaining-011": CudaRemainingStage("remaining-011", "sub-05", "sub-07", 11, 10891, 12057),
    "remaining-012": CudaRemainingStage("remaining-012", "sub-06", "sub-01", 12, 12023, 12294),
    "remaining-013": CudaRemainingStage("remaining-013", "sub-06", "sub-03", 13, 12023, 12897),
    "remaining-014": CudaRemainingStage("remaining-014", "sub-06", "sub-05", 14, 12023, 10891),
    "remaining-015": CudaRemainingStage("remaining-015", "sub-06", "sub-07", 15, 12023, 12057),
    "remaining-016": CudaRemainingStage("remaining-016", "sub-07", "sub-01", 16, 12057, 12294),
    "remaining-017": CudaRemainingStage("remaining-017", "sub-07", "sub-03", 17, 12057, 12897),
    "remaining-018": CudaRemainingStage("remaining-018", "sub-07", "sub-05", 18, 12057, 10891),
    "remaining-019": CudaRemainingStage("remaining-019", "sub-07", "sub-06", 19, 12057, 12023),
}


HARDWARE_AMENDMENT_RELATIVE = "outputs/content_cuda_remaining_hardware_amendment.json"
HARDWARE_AMENDMENT_FIXED = {
    "schema_version": "ep21.content_cuda_remaining_hardware.v1",
    "authorized": True,
    "operation": "extend_pending_remaining_content_hardware_only",
    "selected_at": "2026-10-12 (before first execution)",
    "pending_stage_jobs": {
        "remaining-000": "47377474",
        "remaining-001": "47382245", "remaining-002": "47382246", "remaining-003": "47382247",
        "remaining-008": "47382248", "remaining-009": "47382249", "remaining-010": "47382250",
        "remaining-011": "47382251", "remaining-012": "47382252", "remaining-013": "47382253",
        "remaining-014": "47382254", "remaining-015": "47382255", "remaining-016": "47382256",
        "remaining-017": "47382257", "remaining-018": "47382258", "remaining-019": "47382259",
    },
    "allowed_device_names": ["Tesla V100-SXM2-32GB", "NVIDIA H100 80GB HBM3"],
    "scheduler_constraint": "(GPU_SKU:V100_SXM2&GPU_MEM:32GB)|(GPU_SKU:H100_SXM5&GPU_MEM:80GB)",
    "original_grants_resources_science_unchanged": True,
    "selected_before_first_execution": True,
    "automatic_retry": False,
}


def _hardware_amendment():
    path = BASE / HARDWARE_AMENDMENT_RELATIVE
    if not os.path.lexists(path):
        return None
    value = json.loads(_read_text(path))
    _require(isinstance(value, dict)
             and value.keys() == HARDWARE_AMENDMENT_FIXED.keys() | {"approval_source"},
             "complete_remaining_hardware_amendment_required")
    for key, expected in HARDWARE_AMENDMENT_FIXED.items():
        _require(_same(value[key], expected), "remaining_hardware_amendment_scope_changed")
    _require(isinstance(value["approval_source"], str) and bool(value["approval_source"].strip()),
             "separate_remaining_hardware_selection_required")
    return value


def _stage(name):
    _require(isinstance(name, str) and name in STAGES, "fixed_CUDA_remaining_stage_required")
    return STAGES[name]


def _fixed(stage):
    return {**FIXED, "stage": stage.name, "source": stage.source, "target": stage.target,
            "pair_index": stage.pair_index, "source_eligible_voxels": stage.source_width,
            "target_eligible_voxels": stage.target_width}


def _input_scope(stage):
    return {"normalization_participants": ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"],
            "source_training": stage.source, "target_decoder": stage.target, "source_features": stage.source,
            "target_response_matrix": False, "tau": False, "OOD": False}


def _profile(stage):
    value = json.loads(_read_text(stage.profile_path))
    _require(isinstance(value, dict) and value.get("authorized") is True, "profile_not_activated")
    _require(isinstance(value.get("approval_source"), str) and bool(value["approval_source"].strip()),
             "separate_CUDA_remaining_selection_required")
    for key, expected in _fixed(stage).items():
        _require(_same(value.get(key), expected), "CUDA_remaining_scope_changed")
    _require(_same(value.get("input_scope"), _input_scope(stage)), "CUDA_remaining_roles_changed")
    _require(_same(value.get("resources"), RESOURCES), "fixed_CUDA_remaining_resources_required")
    profile = {**_fixed(stage), "authorized": True, "approval_source": value["approval_source"],
               "input_scope": _input_scope(stage), "resources": dict(RESOURCES)}
    amendment = _hardware_amendment()
    if amendment is not None:
        profile["hardware_eligibility_amendment"] = amendment
    return profile


def _allocation(stage, profile):
    fields = dict(token.split("=", 1) for token in _read_text(stage.run_dir / "allocation.txt").split()
                  if "=" in token)
    host, job = socket.gethostname().split(".")[0], os.environ.get("SLURM_JOB_ID", "")
    _require(job.isdigit() and "-ln" not in host and fields.get("JobId") == job, "actual_compute_job_required")
    _require(fields.get("NodeList", "").split(".")[0] == host, "allocation_node_mismatch")
    for key, expected in {"JobState": "RUNNING", "NumCPUs": "2", "NumTasks": "1",
                          "NumNodes": "1", "Partition": "gpu"}.items():
        _require(fields.get(key) == expected, "allocation_resource_mismatch")
    _require(_time_seconds(fields.get("TimeLimit", "")) == profile["resources"]["minutes"] * 60,
             "allocation_time_mismatch")
    _require(_memory_bytes(fields.get("MinMemoryNode", "")) == 16 * 1024**3, "allocation_memory_mismatch")
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
    _require(os.environ.get("SLURM_GPUS_ON_NODE") == "1" and os.environ.get("SLURM_GPUS", "1") == "1",
             "one_GPU_environment_required")
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
        _require(parsed.tzinfo is None or parsed.utcoffset().total_seconds() == 0, "UTC_allocation_times_required")
        epochs[key] = parsed.replace(tzinfo=timezone.utc).timestamp()
    start, end, now = epochs["StartTime"], epochs["EndTime"], time.time()
    _require(start <= now < end and start < end, "invalid_actual_allocation_interval")
    deadline = min(start + profile["resources"]["deadline_seconds"], end - 60)
    _require(deadline - now >= 1, "insufficient_actual_remaining_allocation")
    return {"job_id": job, "hostname": host, "job_start_epoch": start, "job_end_epoch": end,
        "deadline_epoch": deadline, "cpus": 2, "memory_gib": 16, "partition": "gpu", "minutes": 2880, "gpus": 1}


def _scratch(stage):
    parent = os.environ.get("L_SCRATCH_JOB", "")
    _require(bool(parent) and Path(parent).is_absolute(), "node_local_job_scratch_required")
    path = Path(parent) / stage.scratch_name
    _require(os.environ.get("TMPDIR") == str(path) and path.is_dir() and not path.is_symlink(),
             "fixed_CUDA_remaining_scratch_required")
    return path


@dataclass(frozen=True)
class CudaRemainingContext:
    stage: CudaRemainingStage
    profile: dict
    allocation: dict
    scratch: Path


class CudaRemainingBudget(core.ContentCudaPairBudget):
    """Reuse original allocation/deadline/RSS/device checks, own storage paths."""
    def check_storage(self):
        _bytes_under(self.context.stage.run_dir, self.resources["durable_bytes"])
        _bytes_under(self.context.scratch, self.resources["scratch_bytes"])


def admit(stage_name):
    stage = _stage(stage_name)
    profile = _profile(stage)
    allocation = _allocation(stage, profile)
    amendment = profile.get("hardware_eligibility_amendment")
    if amendment is not None:
        _require(allocation["job_id"] == amendment["pending_stage_jobs"][stage.name],
                 "selected_pending_remaining_job_required")
    context = CudaRemainingContext(stage, profile, allocation, _scratch(stage))
    budget = CudaRemainingBudget(context)
    budget.check_storage()
    _initialize_cuda(budget)
    allowed = amendment["allowed_device_names"] if amendment is not None else [FIXED["device_name"]]
    _require(budget.device_info["name"] in allowed, "qualified_remaining_CUDA_device_required")
    return context, budget


def _training_options(stage):
    return {"device": "cuda:0", "block_units": 256, "row_chunk_size": 512, "backend": "sparse",
            "final_path": stage.final_path, "maximum_bytes": 1073741824}


def _execution_options(stage):
    return {**_training_options(stage), **{key: FIXED[key] for key in
        ("feature_read_mode", "maximum_channel_bytes", "maximum_retained_bytes")}}


def resolver(context, *, operation, source, target, budget, execution_options):
    _require(isinstance(context, CudaRemainingContext) and _stage(context.stage.name) is context.stage
             and isinstance(budget, CudaRemainingBudget) and budget.context is context
             and budget.allocation == context.allocation, "same_CUDA_remaining_context_required")
    stage = context.stage
    _require(_profile(stage) == context.profile and budget.resources == context.profile["resources"],
             "CUDA_remaining_admission_changed")
    budget.check()
    budget.check_storage()
    _require(source == stage.source and target == stage.target, "fixed_CUDA_remaining_pair_required")
    if operation == "bind_content_inputs":
        _require(execution_options is None, "binding_has_no_execution_options")
    elif operation == OPERATION:
        _require(execution_options is not None and _same(dict(execution_options), _execution_options(stage)),
                 "fixed_CUDA_remaining_publication_options_required")
    else:
        raise AdmissionError("operation_not_authorized")


def _bound_scope(bound, stage):
    _require(bound.source.axis.participant == stage.source and bound.target == stage.target
             and len(bound.source.axis.eligible_native_ids) == stage.source_width
             and len(bound.pair.target_axis.eligible_native_ids) == stage.target_width
             and len(bound.source.image_ids) == 4712
             and bound.source.normalized_X.shape == (4712, stage.source_width), "fixed_full_content_remaining_scope_required")
    _require(len(bound.panel.axes) == 5 and len(bound.panel.initializations) == 20,
             "complete_canonical_panel_initialization_required")
    initial = bound.panel.initializations[stage.pair_index]
    _require(initial.pair_index == stage.pair_index and initial.source == stage.source and initial.target == stage.target
             and initial.input_width == stage.source_width and initial.hidden_width == stage.source_width // 2
             and initial.output_width == stage.target_width,
             "canonical_remaining_pair_initialization_required")
    channels = bound.panel.channel_schedule.channels_by_layer
    _require(len(channels) == 19 and all(len(values) == 1024 for values in channels),
             "complete_shared_19_layer_1024_step_schedule_required")


def _load_operations():
    from empirical_content_inputs import bind_content_inputs
    from content_pair_cuda_remaining import train_and_publish_cached_cuda_remaining_pair
    import numpy as np
    import torch
    return bind_content_inputs, train_and_publish_cached_cuda_remaining_pair, {
        "python": sys.version.split()[0], "numpy": str(np.__version__), "torch": str(torch.__version__)}


def _publish(path, record, budget):
    payload = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _require(len(payload) <= 65536, "operational_record_size_limit")
    cap = budget.resources["durable_bytes"]
    _require(_bytes_under(budget.context.stage.run_dir, cap) + len(payload) <= cap, "operational_output_limit")
    with open(path, "xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def run_once(stage_name):
    context, budget = admit(stage_name)
    stage = context.stage
    final, run = stage.final_path, stage.run_dir
    progress_path = run / "progress.jsonl"
    for path in (final, final.with_name(final.name + ".partial"), run / "report.json", progress_path):
        _require(not os.path.lexists(path), "existing_attempt_output_no_retry")
    _publish(run / "claim.json", {"operation": OPERATION, "stage": stage.name, "source": stage.source,
        "target": stage.target, "pair_index": stage.pair_index, "allocation": context.allocation,
        "resources": budget.resources, "admitted_device": budget.device_info,
        "approval_source": context.profile["approval_source"], "automatic_retry": False,
        "hardware_eligibility_amendment": context.profile.get("hardware_eligibility_amendment")}, budget)
    phase, interrupted = "load_operations", False
    started_wall, started_cpu = time.monotonic(), time.process_time()
    report = {"kind": "ep21_content_cuda_remaining_operational_report", "stage": stage.name,
        "source": stage.source, "target": stage.target, "pair_index": stage.pair_index,
        "requested_updates": 1024, "training_images": 4712, "layers": 19,
        "device": "cuda:0", "backend": "sparse", "feature_read_mode": FIXED["feature_read_mode"],
        "maximum_channel_bytes": FIXED["maximum_channel_bytes"],
        "maximum_retained_bytes": FIXED["maximum_retained_bytes"],
        "observed_completed_updates": 0, "completed_updates": None,
        "model_published": False, "published_path": None, "resumable": False,
        "automatic_retry": False, "runtime": None, "admitted_device": budget.device_info,
        "allocation_identity": {key: context.allocation[key] for key in ("job_id", "hostname")}}

    def on_iteration(event):
        completed = event["completed_iterations"]
        _require(type(completed) is int and completed == report["observed_completed_updates"] + 1
                 and completed <= 1024, "consecutive_completed_update_count_required")
        report["observed_completed_updates"] = completed
        payload = (json.dumps({"completed_update": completed,
            "elapsed_wall_seconds": time.monotonic() - started_wall}, allow_nan=False) + "\n").encode("utf-8")
        _require(progress.tell() + len(payload) <= PROGRESS_BYTES, "bounded_operational_progress_required")
        cap = budget.resources["durable_bytes"]
        _require(_bytes_under(run, cap) + len(payload) <= cap, "operational_output_limit")
        progress.write(payload)
        progress.flush()
        budget.check()

    try:
        with open(progress_path, "xb") as progress, open(os.devnull, "w") as quiet, \
                redirect_stdout(quiet), redirect_stderr(quiet):
            bind, train, runtime = _load_operations()
            report["runtime"] = runtime
            phase = "bind_inputs"
            bound = bind(context, require_admitted_context=resolver, source=stage.source, target=stage.target, budget=budget)
            _bound_scope(bound, stage)
            phase = "fit_and_publish"
            result = train(bound, stage=stage.name, on_iteration=on_iteration, **_training_options(stage))
            completion = result.completion
            _require(result.source == stage.source and result.target == stage.target and type(result.pair_index) is int
                     and result.pair_index == stage.pair_index and result.path == final
                     and type(completion.iterations) is int and completion.iterations == 1024
                     and type(completion.rows) is int and completion.rows == 4712
                     and completion.backend == "sparse" and report["observed_completed_updates"] == 1024
                     and final.is_file(), "complete_fixed_CUDA_remaining_publication_required")
            report.update(completed_updates=1024, model_published=True, published_path=str(final))
            phase = "closeout"
            budget.check_storage()
        report.update(status="complete_content_pair")
    except BaseException as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        report.update(status="incomplete", error_type=type(error).__name__, failed_stage=phase)
        if phase == "fit_and_publish":
            report["model_published"] = None
    report.update(elapsed_wall_seconds=time.monotonic() - started_wall,
        process_cpu_seconds=time.process_time() - started_cpu,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    _publish(run / "report.json", report, budget)
    return 130 if interrupted else 0 if report["status"] == "complete_content_pair" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=tuple(STAGES))
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
            print(json.dumps({"status": "CUDA_remaining_admitted", "stage": args.stage,
                "job_id": context.allocation["job_id"], "device": budget.device_info,
                "payload_read": False, "attempt_claimed": False}))
            return 0
        return run_once(args.stage)
    except BaseException as error:
        print(json.dumps({"status": "CUDA_remaining_entrypoint_refused", "stage": args.stage,
                          "error_type": type(error).__name__}))
        return 130 if isinstance(error, KeyboardInterrupt) else 2


if __name__ == "__main__":
    raise SystemExit(main())
