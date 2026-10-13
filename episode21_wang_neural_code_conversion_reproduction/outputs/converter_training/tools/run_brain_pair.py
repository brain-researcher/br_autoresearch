"""Fixed separately admitted, single-attempt full tau-train brain-pair stages.

Only stdlib imports precede explicit admission and the exclusive attempt claim.
No automatic retry, resume, panel expansion, held-out scoring or successor is provided.
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

from run_content_measurement import (
    AdmissionError, THREADS, _bytes_under, _memory_bytes, _read_text, _require, _same,
)

BASE = Path(__file__).resolve().parents[3]
PROFILE = BASE / "outputs/brain_pair_proposal.json"
RUN_DIR = BASE / "outputs/runs/brain_pair_training/attempt001"
FINAL = RUN_DIR / "final_brain_pair.pt"
NEXT_PROFILE = BASE / "outputs/brain_pair_next_authorization.json"
NEXT_RUN_DIR = BASE / "outputs/runs/brain_pair_training/sub03_to_sub05_001"
MEMORY_PROFILE = BASE / "outputs/brain_pair_memory_authorization.json"
MEMORY_RUN_DIR = BASE / "outputs/runs/brain_pair_training/sub03_to_sub05_mem16_001"
MEMORY_REPAIR_PROFILE = BASE / "outputs/brain_pair_memory_repair_authorization.json"
MEMORY_REPAIR_RUN_DIR = BASE / "outputs/runs/brain_pair_training/sub03_to_sub05_mem16_002"
STAGES = ("original", "next-001", "next-memory-001", "next-memory-002")
SOURCE, TARGET = "sub-03", "sub-01"
RESOURCES = {"cpus": 1, "memory_gib": 7, "minutes": 1440, "gpus": 0,
    "deadline_seconds": 84600, "rss_limit_bytes": 6979321856,
    "durable_bytes": 2147483648, "scratch_bytes": 268435456,
    "final_artifact_bytes": 805306368}
FIXED = {"schema_version": "ep21.brain_pair_proposal.v1", "operation": "train_and_publish_brain_pair",
    "jobs": 1, "partition": "russpold", "source": SOURCE, "target": TARGET,
    "training_images": 897, "cv_fits": 50, "final_fits": 1,
    "source_native_voxels": 12897, "target_native_voxels": 12295,
    "automatic_retry": False, "production_training": True, "model_publication": True,
    "scoring": False}
INPUT_SCOPE = {"source_tau_train": SOURCE, "target_tau_train": TARGET,
    "tau_test": False, "OOD": False, "subject_unique": False,
    "features": False, "decoder_payloads": False}


@dataclass(frozen=True)
class BrainPairStage:
    name: str
    profile_path: Path
    run_dir: Path
    final_path: Path
    scratch_name: str
    source: str
    target: str
    source_native_voxels: int
    target_native_voxels: int
    pair_index: int
    allocation_start_deadline: bool
    cpus: int
    memory_gib: int
    rss_limit_bytes: int

    @property
    def resources(self):
        # Each stage owns its resource values; the historical defaults stay fixed.
        return {**RESOURCES, "cpus": self.cpus, "memory_gib": self.memory_gib,
                "rss_limit_bytes": self.rss_limit_bytes}


def _stage(name="original"):
    _require(name in STAGES, "unknown_brain_pair_stage")
    if name == "next-memory-002":
        return BrainPairStage(name, MEMORY_REPAIR_PROFILE, MEMORY_REPAIR_RUN_DIR,
            MEMORY_REPAIR_RUN_DIR / "final_brain_pair.pt", "ep21_brain_pair_sub03_to_sub05_mem16_002",
            "sub-03", "sub-05", 12897, 10891, 5, True, 3, 16, 16106127360)
    if name == "next-memory-001":
        return BrainPairStage(name, MEMORY_PROFILE, MEMORY_RUN_DIR,
            MEMORY_RUN_DIR / "final_brain_pair.pt", "ep21_brain_pair_sub03_to_sub05_mem16_001",
            "sub-03", "sub-05", 12897, 10891, 5, True, 2, 16, 16106127360)
    if name == "next-001":
        return BrainPairStage(name, NEXT_PROFILE, NEXT_RUN_DIR,
            NEXT_RUN_DIR / "final_brain_pair.pt", "ep21_brain_pair_sub03_to_sub05_001",
            "sub-03", "sub-05", 12897, 10891, 5, True, 1, 7, 6979321856)
    return BrainPairStage(name, PROFILE, RUN_DIR, FINAL, "ep21_brain_pair_attempt001",
        SOURCE, TARGET, 12897, 12295, 4, False, 1, 7, 6979321856)


def _profile(stage=None):
    stage = _stage() if stage is None else stage
    value = json.loads(_read_text(stage.profile_path))
    _require(isinstance(value, dict) and value.get("authorized") is True, "profile_not_activated")
    _require(isinstance(value.get("approval_source"), str) and bool(value["approval_source"].strip()),
             "separate_brain_pair_approval_required")
    fixed = {**FIXED, "source": stage.source, "target": stage.target,
             "source_native_voxels": stage.source_native_voxels,
             "target_native_voxels": stage.target_native_voxels}
    scope = {**INPUT_SCOPE, "source_tau_train": stage.source, "target_tau_train": stage.target}
    for key, expected in fixed.items():
        _require(_same(value.get(key), expected), "brain_pair_scope_changed")
    _require(_same(value.get("resources"), stage.resources) and _same(value.get("input_scope"), scope),
             "brain_pair_resources_or_roles_changed")
    return {**fixed, "resources": stage.resources, "input_scope": scope,
            "authorized": True, "approval_source": value["approval_source"]}


def _time_seconds(value):
    # Slurm formats a full day either as24:00:00 or1-00:00:00.
    days = 0
    if "-" in value:
        day, separator, value = value.partition("-")
        _require(separator == "-" and day.isdigit(), "allocation_time_format")
        days = int(day)
    parts = value.split(":")
    _require(len(parts) == 3 and all(part.isdigit() for part in parts), "allocation_time_format")
    hours, minutes, seconds = map(int, parts)
    _require(minutes < 60 and seconds < 60, "allocation_time_format")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _allocation(stage=None):
    stage = _stage() if stage is None else stage
    resources = stage.resources
    fields = dict(token.split("=", 1) for token in _read_text(stage.run_dir / "allocation.txt").split()
                  if "=" in token)
    host, job = socket.gethostname().split(".")[0], os.environ.get("SLURM_JOB_ID", "")
    _require(job.isdigit() and "-ln" not in host and fields.get("JobId") == job,
             "actual_compute_job_required")
    _require(fields.get("NodeList", "").split(".")[0] == host, "allocation_node_mismatch")
    for key, expected in {"JobState": "RUNNING", "NumCPUs": str(resources["cpus"]), "NumTasks": "1",
                          "NumNodes": "1", "Partition": "russpold"}.items():
        _require(fields.get(key) == expected, "allocation_resource_mismatch")
    _require(_time_seconds(fields.get("TimeLimit", "")) == resources["minutes"] * 60, "allocation_time_mismatch")
    _require(_memory_bytes(fields.get("MinMemoryNode", "")) == resources["memory_gib"] * 1024**3,
             "allocation_memory_mismatch")
    tres = dict(item.split("=", 1) for item in fields.get("AllocTRES", "").split(",") if "=" in item)
    _require(tres.get("cpu") == str(resources["cpus"]) and tres.get("node") == "1"
             and _memory_bytes(tres.get("mem", "")) == resources["memory_gib"] * 1024**3,
             "allocated_TRES_mismatch")
    _require(not any("gpu" in key.lower() for key in tres), "gpu_allocation_forbidden")
    for key, value in fields.items():
        if key.startswith(("TresPer", "Gres", "GRES")):
            _require("gpu" not in value.lower(), "gpu_allocation_forbidden")
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
    # The batch captures scontrol under TZ=UTC; an explicit offset, if present,
    # must agree rather than being silently discarded.
    end_value = datetime.fromisoformat(fields.get("EndTime", ""))
    _require(end_value.tzinfo is None or end_value.utcoffset().total_seconds() == 0,
             "UTC_allocation_end_required")
    end = end_value.replace(tzinfo=timezone.utc).timestamp()
    result = {"job_id": job, "hostname": host, "job_end_epoch": end,
              "cpus": resources["cpus"], "memory_gib": resources["memory_gib"],
              "partition": "russpold", "minutes": resources["minutes"], "gpus": 0}
    now = time.time()
    if stage.allocation_start_deadline:
        start_value = datetime.fromisoformat(fields.get("StartTime", ""))
        _require(start_value.tzinfo is None or start_value.utcoffset().total_seconds() == 0,
                 "UTC_allocation_start_required")
        start = start_value.replace(tzinfo=timezone.utc).timestamp()
        _require(start <= now < end and start < end, "invalid_actual_allocation_interval")
        deadline = min(start + resources["deadline_seconds"], end - 60)
        _require(deadline - now >= 1, "insufficient_actual_remaining_allocation")
        result.update(job_start_epoch=start, deadline_epoch=deadline)
    else:
        _require(end - now >= 84630, "insufficient_actual_remaining_allocation")
    return result


def _scratch(stage=None):
    stage = _stage() if stage is None else stage
    parent = os.environ.get("L_SCRATCH_JOB", "")
    _require(bool(parent) and Path(parent).is_absolute(), "node_local_job_scratch_required")
    expected = Path(parent) / stage.scratch_name
    _require(os.environ.get("TMPDIR") == str(expected) and expected.is_dir()
             and not expected.is_symlink(), "fixed_brain_pair_scratch_required")
    return expected


@dataclass(frozen=True)
class BrainPairContext:
    profile: dict
    allocation: dict
    scratch: Path
    stage: BrainPairStage


class BrainPairBudget:
    """One separately admitted finite parent, preserving all prior accounts."""
    def __init__(self, context):
        self.context = context
        self.allocation = dict(context.allocation)
        self.resources = context.stage.resources
        self.absolute_deadline = (self.allocation["deadline_epoch"]
            if context.stage.allocation_start_deadline else self.allocation["job_end_epoch"])
        remaining = (self.absolute_deadline - time.time()
            if context.stage.allocation_start_deadline else self.resources["deadline_seconds"])
        self.deadline = time.monotonic() + remaining
        self.check()

    def check(self):
        _require(os.environ.get("SLURM_JOB_ID") == self.allocation["job_id"]
                 and socket.gethostname().split(".")[0] == self.allocation["hostname"],
                 "allocation_identity_changed")
        if time.monotonic() >= self.deadline or time.time() >= self.absolute_deadline:
            raise TimeoutError("brain_pair_deadline")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > self.resources["rss_limit_bytes"]:
            raise MemoryError("brain_pair_RSS_limit")

    def check_storage(self):
        _bytes_under(self.context.stage.run_dir, self.resources["durable_bytes"])
        _bytes_under(self.context.scratch, self.resources["scratch_bytes"])


    def remaining_seconds(self):
        self.check()
        remaining = int(min(self.deadline - time.monotonic(), self.absolute_deadline - time.time()))
        _require(remaining > 0, "insufficient_actual_remaining_allocation")
        return remaining


def admit(stage_name="original"):
    stage = _stage(stage_name)
    profile = _profile(stage)  # Inactive proposals never inspect allocation or inputs.
    context = BrainPairContext(profile, _allocation(stage), _scratch(stage), stage)
    budget = BrainPairBudget(context)
    budget.check_storage()
    return context, budget


def resolver(context, *, operation, source, target, budget, execution_options):
    _require(isinstance(context, BrainPairContext) and isinstance(budget, BrainPairBudget)
             and budget.context is context and budget.allocation == context.allocation,
             "same_brain_pair_context_required")
    _require(context.stage == _stage(context.stage.name)
             and _profile(context.stage) == context.profile, "brain_pair_admission_changed")
    budget.check()
    budget.check_storage()
    _require(source == context.stage.source and target == context.stage.target, "fixed_brain_pair_required")
    if operation == "bind_brain_inputs":
        _require(execution_options is None, "binding_has_no_execution_options")
    elif operation == FIXED["operation"]:
        expected = {"final_path": context.stage.final_path,
                    "maximum_bytes": context.stage.resources["final_artifact_bytes"]}
        _require(execution_options is not None and _same(dict(execution_options), expected),
                 "fixed_brain_pair_publication_options_required")
    else:
        raise AdmissionError("operation_not_authorized")


def _load_operations():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from empirical_brain_inputs import bind_brain_inputs
    from brain_pair_driver import train_and_publish_brain_pair
    import numpy as np
    return bind_brain_inputs, train_and_publish_brain_pair, {
        "python": sys.version.split()[0], "numpy": str(np.__version__), "torch": str(torch.__version__)}


def _bound_scope(bound, stage=None):
    stage = _stage() if stage is None else stage
    for response, participant, width in ((bound.source, stage.source, stage.source_native_voxels),
                                         (bound.target, stage.target, stage.target_native_voxels)):
        _require(response.participant == participant and response.role == "tau_train"
                 and len(response.image_ids) == 897 and len(response.native_voxel_ids) == width
                 and response.values.shape == (897, width), "fixed_full_native_brain_input_scope_required")


def _publish(path, record, stage=None):
    stage = _stage() if stage is None else stage
    payload = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _require(len(payload) <= 65536, "operational_record_size_limit")
    _require(_bytes_under(stage.run_dir, stage.resources["durable_bytes"]) + len(payload)
             <= stage.resources["durable_bytes"],
             "operational_output_limit")
    with open(path, "xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _brain_failure_details(exc, stage=None):
    """Expose fixed failure identifiers, never arbitrary exception text.

    Inspect only the class already loaded by the admitted operations. In
    particular, failure reporting cannot import scientific code before admission.
    A missing fold in the original message stays unspecified here.
    """
    module = sys.modules.get("brain_loss")
    if (type(exc) is not getattr(module, "BrainLossNotEvaluable", None)
            or len(exc.args) != 1 or type(exc.args[0]) is not str or len(exc.args[0]) > 256):
        return {}
    stage = _stage() if stage is None else stage
    known = {code: {"reason_code": code} for code in (
        "nonfinite_float32_primal_system", "required_float32_primal_solve_failed",
        "nonfinite_float32_brain_mapping", "nonfinite_brain_validation_prediction",
        "incomplete_or_nonfinite_required_CV_scores", "nonfinite_image_weighted_CV_score")}
    for role, participant in (("source", stage.source), ("target", stage.target)):
        for code in ("nonfinite_required_tau_train_response", "nonfinite_required_tau_train_export",
                     "nonfinite_required_tau_train_repetition", "nonfinite_required_tau_train_image_mean"):
            known[f"{code}:{participant}"] = {"reason_code": code, "role": role}
        code = "zero_or_nonfinite_required_training_scale"
        for fold in range(5):
            known[f"{code}:{role}_fold_{fold}"] = {
                "reason_code": code, "role": role, "context": "fold_training", "fold": fold}
        known[f"{code}:{role}_final"] = {"reason_code": code, "role": role, "context": "final"}
        code = "nonfinite_required_standardized_response"
        for context in ("fit", "validation", "final"):
            known[f"{code}:{role}_{context}"] = {"reason_code": code, "role": role, "context": context}
    code = "nonfinite_required_validation_MSE"
    for fold in range(5):
        for alpha in (620, 640, 660, 740, 780, 800, 820, 1000, 1200, 1400):
            known[f"{code}:fold_{fold}:alpha_{alpha}"] = {"reason_code": code, "fold": fold, "alpha": alpha}
    details = known.get(exc.args[0])
    return {} if details is None else {"brain_loss_failure": details}


def run_once(stage_name="original"):
    context, budget = admit(stage_name)
    selected = context.stage
    final, run_dir = selected.final_path, selected.run_dir
    for path in (final, final.with_name(final.name + ".partial"), run_dir / "report.json"):
        _require(not os.path.lexists(path), "existing_attempt_output_no_retry")
    _publish(run_dir / "claim.json", {"operation": FIXED["operation"],
        "source": selected.source, "target": selected.target,
        "pair_stage": selected.name, "pair_index": selected.pair_index,
        "allocation": context.allocation, "approval_source": context.profile["approval_source"],
        "automatic_retry": False}, selected)
    stage, interrupted = "load_operations", False
    started_wall, started_cpu = time.monotonic(), time.process_time()
    report = {"kind": "ep21_brain_pair_operational_report",
              "source": selected.source, "target": selected.target,
              "pair_stage": selected.name, "pair_index": selected.pair_index,
              "completed_fits": None, "model_published": False, "published_path": None,
              "resumable": False, "automatic_retry": False,
              "runtime": None,
              "allocation_identity": {key: context.allocation[key] for key in ("job_id", "hostname")}}
    try:
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
            bind, train, runtime = _load_operations()
            report["runtime"] = runtime
            stage = "bind_inputs"
            bound = bind(context, require_admitted_context=resolver,
                         source=selected.source, target=selected.target, budget=budget)
            _bound_scope(bound, selected)
            stage = "fit_and_publish"
            result = train(bound, final_path=final, maximum_bytes=selected.resources["final_artifact_bytes"])
            _require(result.source == selected.source and result.target == selected.target and result.path == final
                     and result.summary.available is True and type(result.summary.completed_fits) is int
                     and result.summary.completed_fits == 51 and final.is_file(),
                     "complete_fixed_brain_pair_publication_required")
            report.update(completed_fits=51, model_published=True, published_path=str(final))
            stage = "closeout"
            budget.check_storage()
        report.update(status="complete_brain_pair")
    except BaseException as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        report.update(status="incomplete", error_type=type(exc).__name__, failed_stage=stage)
        report.update(_brain_failure_details(exc, selected))
        if stage == "fit_and_publish":
            report["model_published"] = None
    # Only allowlisted failure identifiers; no values, scores, free text or traceback.
    report.update(elapsed_wall_seconds=time.monotonic() - started_wall,
                  process_cpu_seconds=time.process_time() - started_cpu,
                  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    _publish(run_dir / "report.json", report, selected)
    return 130 if interrupted else 0 if report["status"] == "complete_brain_pair" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="original")
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
            print(json.dumps({"status": "brain_pair_admitted", "job_id": context.allocation["job_id"],
                              "payload_read": False, "attempt_claimed": False}))
            return 0
        return run_once(args.stage)
    except BaseException as exc:
        print(json.dumps({"status": "brain_pair_entrypoint_refused", "error_type": type(exc).__name__}))
        return 130 if isinstance(exc, KeyboardInterrupt) else 2


if __name__ == "__main__":
    raise SystemExit(main())
