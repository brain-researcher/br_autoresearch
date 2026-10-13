"""Separate, initially inactive, one-attempt EP21 resource-measurement entrypoint.

Stdlib-only admission precedes scientific imports and all payload access. This
module admits binding and four disposable updates, never full fit/publication.
The enclosing Slurm batch owns the 2h allocation and optional companions.
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
import stat
import time

BASE = Path(__file__).resolve().parents[3]
PROFILE = BASE / "outputs/converter_measurement_proposal.json"
RUN_DIR = BASE / "outputs/runs/content_resource_measurement/attempt002"
STRIPE_PROFILE = BASE / "outputs/converter_io_measurement_authorization.json"
STRIPE_RUN_DIR = BASE / "outputs/runs/content_resource_measurement/column_stripe001"
STRIPE_REPAIR_PROFILE = BASE / "outputs/converter_io_measurement_repair_authorization.json"
STRIPE_REPAIR_RUN_DIR = BASE / "outputs/runs/content_resource_measurement/column_stripe002"
STAGES = ("original", "column-stripe-001", "column-stripe-002")
SOURCE, TARGET = "sub-03", "sub-01"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
RESOURCES = {"cpus": 3, "memory_gib": 16, "minutes": 120, "gpus": 0,
             "deadline_seconds": 6300, "rss_limit_bytes": 15 * 1024**3,
             "durable_bytes": 16 * 1024**2, "scratch_bytes": 256 * 1024**2}
FIXED = {"schema_version": "ep21.content_measurement_proposal.v1",
         "operation": "measure_content_pair_four_updates", "jobs": 1,
         "partition": "russpold", "source": SOURCE, "target": TARGET,
         "updates": 4, "block_units": 256, "row_chunk_size": 512,
         "backend": "sparse", "device": "cpu", "automatic_retry": False,
         "production_training": False, "model_publication": False,
         "correction_report": "outputs/pipeline_conformance/converter_loss_correction_001/report.json"}
INPUT_SCOPE = {"normalization_participants": ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"],
               "source_training": SOURCE, "target_decoder": TARGET,
               "source_features": SOURCE, "tau": False, "OOD": False}


class AdmissionError(ValueError):
    """Fixed operational refusal; never include empirical exception text."""


def _require(condition, code):
    if not condition:
        raise AdmissionError(code)


@dataclass(frozen=True)
class MeasurementStage:
    name: str
    profile_path: Path
    run_dir: Path
    scratch_name: str
    extra_scope: tuple


def _stage(name="original"):
    """Fixed attempt identities only; never accept caller-supplied paths/grants."""
    _require(name in STAGES, "unknown_measurement_stage")
    if name == "column-stripe-001":
        return MeasurementStage(name, STRIPE_PROFILE, STRIPE_RUN_DIR,
            "ep21_content_measurement_column_stripe001",
            (("feature_read_mode", "bounded_column_stripe"),
             ("feature_stripe_bytes", 67108864)))
    if name == "column-stripe-002":
        return MeasurementStage(name, STRIPE_REPAIR_PROFILE, STRIPE_REPAIR_RUN_DIR,
            "ep21_content_measurement_column_stripe002",
            (("feature_read_mode", "bounded_column_stripe"),
             ("feature_stripe_bytes", 67108864)))
    return MeasurementStage(name, PROFILE, RUN_DIR,
        "ep21_content_measurement_attempt002", ())


def _read_text(path):
    with open(path, "rb") as stream:
        value = stream.read(65537)
    _require(len(value) <= 65536, "metadata_size_limit")
    return value.decode("utf-8")


def _same(actual, expected):
    # bool must not stand in for an integer resource count.
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_same(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_same(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _profile(stage):
    value = json.loads(_read_text(stage.profile_path))
    _require(isinstance(value, dict) and value.get("authorized") is True, "profile_not_activated")
    _require(isinstance(value.get("approval_source"), str) and bool(value["approval_source"].strip()),
             "separate_measurement_approval_required")
    fixed = {**FIXED, **dict(stage.extra_scope)}
    for key, expected in fixed.items():
        _require(_same(value.get(key), expected), "measurement_scope_changed")
    _require(_same(value.get("resources"), RESOURCES) and _same(value.get("input_scope"), INPUT_SCOPE),
             "measurement_resources_or_roles_changed")
    # Descriptive companion/accounting fields are allowed, not new authority.
    return {**fixed, "resources": dict(RESOURCES), "input_scope": INPUT_SCOPE,
            "authorized": True, "approval_source": value["approval_source"]}


def _memory_bytes(value):
    for suffix, scale in (("G", 1024**3), ("M", 1024**2), ("K", 1024)):
        if value.endswith(suffix):
            number = float(value[:-1])
            _require(math.isfinite(number) and number >= 0, "invalid_allocation_memory")
            return number * scale
    # Bare scontrol MinMemoryNode is expressed in MiB.
    if value.isdigit():
        return int(value) * 1024**2
    raise AdmissionError("allocation_memory_units_required")


def _time_seconds(value):
    parts = value.split(":")
    _require(len(parts) == 3 and all(part.isdigit() for part in parts), "allocation_time_format")
    hours, minutes, seconds = map(int, parts)
    _require(minutes < 60 and seconds < 60, "allocation_time_format")
    return hours * 3600 + minutes * 60 + seconds


def _allocation(stage):
    fields = dict(token.split("=", 1) for token in _read_text(stage.run_dir / "allocation.txt").split()
                  if "=" in token)
    host, job = socket.gethostname().split(".")[0], os.environ.get("SLURM_JOB_ID", "")
    _require(job.isdigit() and "-ln" not in host and fields.get("JobId") == job,
             "actual_compute_job_required")
    _require(fields.get("NodeList", "").split(".")[0] == host, "allocation_node_mismatch")
    for key, expected in {"JobState": "RUNNING", "NumCPUs": "3", "NumTasks": "1",
                          "NumNodes": "1", "Partition": "russpold"}.items():
        _require(fields.get(key) == expected, "allocation_resource_mismatch")
    _require(_time_seconds(fields.get("TimeLimit", "")) == 7200, "allocation_time_mismatch")
    _require(_memory_bytes(fields.get("MinMemoryNode", "")) == 16 * 1024**3,
             "allocation_memory_mismatch")
    tres = dict(item.split("=", 1) for item in fields.get("AllocTRES", "").split(",") if "=" in item)
    _require(tres.get("cpu") == "3" and tres.get("node") == "1"
             and _memory_bytes(tres.get("mem", "")) == 16 * 1024**3, "allocated_TRES_mismatch")
    _require(not any("gpu" in key.lower() for key in tres), "gpu_allocation_forbidden")
    for key, value in fields.items():
        if key.startswith(("TresPer", "Gres", "GRES")):
            _require("gpu" not in value.lower(), "gpu_allocation_forbidden")
    _require(os.environ.get("SLURM_CPUS_PER_TASK") == "3"
             and os.environ.get("SLURM_MEM_PER_NODE") == "16384", "allocation_environment_mismatch")
    for key in ("SLURM_NTASKS", "SLURM_NNODES"):
        _require(os.environ.get(key, "1") == "1", "allocation_environment_mismatch")
    _require(all(os.environ.get(key) == "1" for key in THREADS), "one_thread_environment_required")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and
             all(os.environ.get(key, "") in ("", "0") for key in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE"))
             and all(not os.environ.get(key, "") for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS")),
             "cpu_only_environment_required")
    end = datetime.fromisoformat(fields.get("EndTime", "")).replace(tzinfo=timezone.utc).timestamp()
    _require(end - time.time() >= 6330, "insufficient_actual_remaining_allocation")
    return {"job_id": job, "hostname": host, "job_end_epoch": end,
            "cpus": 3, "memory_gib": 16, "partition": "russpold", "minutes": 120, "gpus": 0}


def _scratch(stage):
    parent = os.environ.get("L_SCRATCH_JOB", "")
    _require(bool(parent) and Path(parent).is_absolute(), "node_local_job_scratch_required")
    expected = Path(parent) / stage.scratch_name
    _require(os.environ.get("TMPDIR") == str(expected) and expected.is_dir()
             and not expected.is_symlink(), "fixed_measurement_scratch_required")
    return expected


def _bytes_under(path, cap):
    """Bounded accounting of this tiny attempt directory, not project scans."""
    pending, entries, total = [(Path(path), 0)], 0, 0
    while pending:
        directory, depth = pending.pop()
        _require(depth <= 4 and not directory.is_symlink(), "attempt_directory_shape_limit")
        with os.scandir(directory) as stream:
            for entry in stream:
                entries += 1
                _require(entries <= 128, "attempt_directory_entry_limit")
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    pending.append((Path(entry.path), depth + 1))
                else:
                    _require(stat.S_ISREG(info.st_mode), "attempt_regular_files_only")
                    total += info.st_size
                    _require(total <= cap, "attempt_byte_limit")
    return total


@dataclass(frozen=True)
class MeasurementContext:
    profile: dict
    allocation: dict
    scratch: Path
    stage: MeasurementStage


class MeasurementBudget:
    """Separate 16GiB-stage budget; no decoder ledger or old 7GiB grant."""
    def __init__(self, context):
        self.context = context
        self.allocation = dict(context.allocation)
        self.deadline = time.monotonic() + RESOURCES["deadline_seconds"]
        self.check()

    def check(self):
        _require(os.environ.get("SLURM_JOB_ID") == self.allocation["job_id"]
                 and socket.gethostname().split(".")[0] == self.allocation["hostname"],
                 "allocation_identity_changed")
        if time.monotonic() >= self.deadline or time.time() >= self.allocation["job_end_epoch"]:
            raise TimeoutError("measurement_deadline")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > RESOURCES["rss_limit_bytes"]:
            raise MemoryError("measurement_RSS_limit")

    def check_storage(self):
        _bytes_under(self.context.stage.run_dir, RESOURCES["durable_bytes"])
        _bytes_under(self.context.scratch, RESOURCES["scratch_bytes"])


def admit(*, stage="original"):
    selected = _stage(stage)
    # Must precede allocation: inactive profile never reads source/allocation.
    profile = _profile(selected)
    context = MeasurementContext(profile, _allocation(selected), _scratch(selected), selected)
    budget = MeasurementBudget(context)
    budget.check_storage()
    return context, budget


def resolver(context, *, operation, source, target, budget, execution_options):
    _require(isinstance(context, MeasurementContext) and isinstance(budget, MeasurementBudget)
             and budget.context is context and budget.allocation == context.allocation,
             "same_measurement_context_required")
    _require(_profile(context.stage) == context.profile, "measurement_admission_changed")
    budget.check()
    _require(source == SOURCE and target == TARGET, "fixed_measurement_pair_required")
    if operation == "bind_content_inputs":
        _require(execution_options is None, "binding_has_no_execution_options")
    elif operation == FIXED["operation"]:
        expected = {"device": "cpu", "block_units": 256, "row_chunk_size": 512,
                    "backend": "sparse", "updates": 4, "source_rows": 4712,
                    "layer_count": 19, "final_path": context.stage.run_dir / "reserved_final_model.pt", "maximum_bytes": 1}
        _require(execution_options is not None and _same(dict(execution_options), expected),
                 "fixed_measurement_options_required")
    else:
        raise AdmissionError("operation_not_authorized")
    return None


def _load_operations():
    # Nothing scientific is imported before separate admission and claim.
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import empirical_content_inputs
    import content_pair_measurement
    return empirical_content_inputs.bind_content_inputs, content_pair_measurement.measure_content_pair


def _publish(path, record, *, stage=None):
    selected = _stage() if stage is None else stage
    payload = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    used = _bytes_under(selected.run_dir, RESOURCES["durable_bytes"])
    _require(used + len(payload) <= RESOURCES["durable_bytes"], "operational_output_limit")
    with open(path, "xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def run_once(*, stage="original"):
    context, budget = admit(stage=stage)
    selected, run_dir = context.stage, context.stage.run_dir
    # Batch owns this directory; only this exclusive file claims the one run.
    _publish(run_dir / "claim.json", {"operation": FIXED["operation"], "stage": selected.name,
        **dict(selected.extra_scope),
        "allocation": context.allocation, "approval_source": context.profile["approval_source"],
        "automatic_retry": False, "production_training": False, "model_publication": False}, stage=selected)
    report, interrupted = None, False
    try:
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
            bind, measure = _load_operations()
            def prepare_inputs():
                return bind(context, require_admitted_context=resolver, source=SOURCE,
                            target=TARGET, budget=budget)
            report = measure(context, require_admitted_context=resolver, prepare_inputs=prepare_inputs,
                             budget=budget, reserved_final_path=run_dir / "reserved_final_model.pt")
        budget.check_storage()
        _require(isinstance(report, dict) and report.get("status") in ("complete_measurement", "incomplete")
                 and report.get("model_published") is False and report.get("production_fit_completed") is False,
                 "operational_measurement_report_required")
    except BaseException as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        if not isinstance(report, dict):
            report = {"kind": "ep21_content_pair_resource_measurement", "source": SOURCE, "target": TARGET,
                      "completed_updates": None, "production_fit_completed": False,
                      "model_published": False, "resumable": False,
                      "allocation_identity": {k: context.allocation[k] for k in ("job_id", "hostname")}}
        # A storage failure after completed updates must not erase their counts.
        report.update(status="incomplete", error_type=type(exc).__name__, failed_stage="measurement_entrypoint")
    report.update(stage=selected.name, **dict(selected.extra_scope))
    _publish(run_dir / "report.json", report, stage=selected)
    return 130 if interrupted else (0 if report["status"] == "complete_measurement" else 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-admission", action="store_true")
    parser.add_argument("--stage", choices=STAGES, default="original")
    args = parser.parse_args(argv)
    try:
        if args.check_admission:
            context, _ = admit(stage=args.stage)
            print(json.dumps({"status": "measurement_admitted", "job_id": context.allocation["job_id"],
                              "stage": context.stage.name,
                              "payload_read": False, "attempt_claimed": False}))
            return 0
        return run_once(stage=args.stage)
    except BaseException as exc:
        # No exception message/traceback: empirical libraries can include values.
        print(json.dumps({"status": "measurement_entrypoint_refused", "error_type": type(exc).__name__}))
        return 130 if isinstance(exc, KeyboardInterrupt) else 2


if __name__ == "__main__":
    raise SystemExit(main())
