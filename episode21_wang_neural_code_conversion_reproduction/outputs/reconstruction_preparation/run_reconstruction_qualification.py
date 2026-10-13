"""One bounded public-model qualification, with independently admitted companions.

Only standard-library imports precede profile and actual CPU allocation checks.
This module submits no job and never grants neural access. The separately
authorized brain-prefix child reuses allocation proof, not the public input scope.
"""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time

BASE = Path(__file__).resolve().parents[2]
CONVERTER_TOOLS = BASE / "outputs/converter_training/tools"
if str(CONVERTER_TOOLS) not in sys.path:
    sys.path.insert(0, str(CONVERTER_TOOLS))
from run_content_measurement import _read_text, _require, _same, _memory_bytes, _time_seconds, _bytes_under, THREADS

PROFILE = BASE / "outputs/reconstruction_qualification_authorization.json"
RUN_DIR = BASE / "outputs/runs/reconstruction_qualification/attempt001"
SCRATCH_NAME = "ep21_reconstruction_qualification001"
RESOURCES = {"cpus": 1, "memory_gib": 7, "minutes": 120, "gpus": 0,
    "numeric_threads": 1, "deadline_seconds": 6900, "rss_limit_bytes": 6979321856,
    "durable_bytes": 1073741824, "scratch_bytes": 4294967296,
    "separate_diagnostic_durable_bytes": 1048576}
FIXTURE = {"count": 1, "features": "C_order_float32_((unit_index_modulo_257)-128)/128_each_full_layer",
    "decoder_means": "float32_zeros", "layers": 19, "updates": 200, "seed": 42,
    "method": "within_source", "source": "protocol-neutral-fixture", "device": "cpu"}
INPUT_SCOPE = {"generator_figshare_file": 52399790, "generator_archive_bytes": 832692892,
    "generator_archive_md5": "8be66997045859a49b1d2bcef6640490",
    "generator_members": ["generator_relu7.pt", "relu7_act_range_3x.txt"],
    "reuse_VGG_archive": True,
    "VGG_members": ["VGG_ILSVRC_19_layers.pt", "estimated_cnn_feat_std_VGG_ILSVRC_19_layers_ImgSize_224x224_chwise_dof1.mat"],
    "neural_values": False, "LAION_images": False, "decoded_features": False, "tau": False, "OOD": False}
FIXED = {"schema_version": "ep21.reconstruction_qualification.v1",
    "operation": "qualify_public_reconstruction_runtime", "jobs": 1, "partition": "russpold",
    "automatic_retry": False, "empirical_reconstruction": False, "scientific_scoring": False,
    "image_publication": False, "protocol_lock_mutation": False,
    "serialized_companions": {"focused_checks_and_ASTRA": True,
        "brain_prefix_separate_authorization": "outputs/brain_prefix_diagnostic_reconstruction_authorization.json",
        "charge_parent_once": True}}


def _profile():
    value = json.loads(_read_text(PROFILE))
    _require(value.get("authorized") is True and isinstance(value.get("approval_source"), str)
             and bool(value["approval_source"].strip()), "public_qualification_authority_required")
    for key, expected in {**FIXED, "resources": RESOURCES, "fixture": FIXTURE, "input_scope": INPUT_SCOPE}.items():
        _require(_same(value.get(key), expected), "public_qualification_scope_changed")
    return value


def _epoch(value):
    stamp = datetime.fromisoformat(value)
    return stamp.replace(tzinfo=timezone.utc).timestamp() if stamp.tzinfo is None else stamp.timestamp()


def _allocation(component):
    _require(component in ("reconstruction", "brain_prefix"), "fixed_parent_component_required")
    fields = dict(item.split("=", 1) for item in _read_text(RUN_DIR / "allocation.txt").split() if "=" in item)
    host, job = socket.gethostname().split(".")[0], os.environ.get("SLURM_JOB_ID", "")
    _require(job.isdigit() and "-ln" not in host and fields.get("JobId") == job
             and fields.get("NodeList", "").split(".")[0] == host, "actual_owned_compute_allocation_required")
    for key, expected in {"JobState": "RUNNING", "NumCPUs": "1", "NumTasks": "1",
                          "NumNodes": "1", "Partition": "russpold"}.items():
        _require(fields.get(key) == expected, "CPU_qualification_allocation_mismatch")
    _require(_time_seconds(fields.get("TimeLimit", "")) == 7200
             and _memory_bytes(fields.get("MinMemoryNode", "")) == 7 * 1024**3,
             "CPU_qualification_time_or_memory_mismatch")
    tres = dict(item.split("=", 1) for item in fields.get("AllocTRES", "").split(",") if "=" in item)
    _require(tres.get("cpu") == "1" and tres.get("node") == "1"
             and _memory_bytes(tres.get("mem", "")) == 7 * 1024**3
             and not any(key.startswith("gres/gpu") and value != "0" for key, value in tres.items()),
             "CPU_only_allocated_TRES_required")
    _require(os.environ.get("SLURM_CPUS_PER_TASK") == "1"
             and os.environ.get("SLURM_MEM_PER_NODE") == "7168"
             and all(os.environ.get(key, "1") == "1" for key in ("SLURM_NTASKS", "SLURM_NNODES")),
             "CPU_qualification_environment_mismatch")
    _require(all(os.environ.get(key) == "1" for key in THREADS)
             and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
             and all(os.environ.get(key, "0") in ("", "0") for key in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE")),
             "CPU_only_single_thread_environment_required")
    start, end = _epoch(fields.get("StartTime", "")), _epoch(fields.get("EndTime", ""))
    deadline = min(start + RESOURCES["deadline_seconds"], end - 60)
    _require(start <= time.time() and deadline - time.time() >= (330 if component == "brain_prefix" else 600),
             "insufficient_actual_parent_time_remaining")
    return {"job_id": job, "hostname": host, "job_start_epoch": start, "job_end_epoch": end,
            "deadline_epoch": deadline, "cpus": 1, "memory_gib": 7, "minutes": 120,
            "gpus": 0, "partition": "russpold"}


@dataclass(frozen=True)
class QualificationContext:
    profile: dict
    allocation: dict
    scratch: Path
    component: str


class QualificationBudget:
    def __init__(self, context):
        self.context = context
        self.allocation = dict(context.allocation)
        self.check()

    def check(self):
        _require(os.environ.get("SLURM_JOB_ID") == self.allocation["job_id"]
                 and socket.gethostname().split(".")[0] == self.allocation["hostname"],
                 "qualification_allocation_identity_changed")
        if time.time() >= self.allocation["deadline_epoch"]:
            raise TimeoutError("public_qualification_parent_deadline")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > RESOURCES["rss_limit_bytes"]:
            raise MemoryError("public_qualification_RSS_limit")

    def check_storage(self):
        _bytes_under(RUN_DIR, RESOURCES["durable_bytes"])
        _bytes_under(self.context.scratch, RESOURCES["scratch_bytes"])


def admit(*, component="reconstruction"):
    profile = _profile()
    allocation = _allocation(component)
    parent = os.environ.get("L_SCRATCH_JOB", "")
    _require(bool(parent) and Path(parent).is_absolute(), "node_local_job_scratch_required")
    scratch = Path(parent) / SCRATCH_NAME
    _require(scratch.is_dir() and not scratch.is_symlink() and os.environ.get("TMPDIR") == str(scratch / "tmp"),
             "fixed_qualification_job_scratch_required")
    context = QualificationContext(profile, allocation, scratch, component)
    budget = QualificationBudget(context)
    budget.check_storage()
    return context, budget


def _publish(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _require(len(payload) <= 262144 and _bytes_under(RUN_DIR, RESOURCES["durable_bytes"])
             + len(payload) <= RESOURCES["durable_bytes"], "qualification_output_limit")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _configure_runtime():
    # Import only inside the admitted child, before model work can start.
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def run_once():
    context, budget = admit()
    _require(not os.path.lexists(RUN_DIR / "report.json"), "qualification_attempt_already_closed")
    _publish(RUN_DIR / "claim.json", {"operation": FIXED["operation"], "allocation": budget.allocation,
        "fixture": FIXTURE, "automatic_retry": False, "empirical_execution": False})
    started = time.monotonic()
    report = {"kind": "ep21_public_reconstruction_qualification", "status": "incomplete",
        "allocation_identity": {key: budget.allocation[key] for key in ("job_id", "hostname")},
        "empirical_execution": False, "neural_or_LAION_image_values_read": False,
        "tau_or_OOD_access": False, "image_or_scientific_score_published": False,
        "protocol_lock_changed": False, "automatic_retry": False}
    interrupted = False
    try:
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
            _configure_runtime()
            from qualify_reconstruction import qualify_reconstruction
            result = qualify_reconstruction(budget=budget, scratch=context.scratch,
                retained_archive_dir=RUN_DIR / "assets")
            _require(result.get("status") == "completed_public_runtime_qualification"
                     and result.get("fixture_count") == 1
                     and result.get("reconstruction", {}).get("completed_updates") == 200,
                     "one_complete_public_reconstruction_required")
            budget.check()
            budget.check_storage()
            _require(_profile() == context.profile, "qualification_profile_changed_during_run")
            report["qualification"] = result
            report["status"] = "complete_public_runtime_qualification"
    except BaseException as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        report["error_type"] = type(error).__name__
        stage = getattr(error, "stage", None)
        if stage in ("admitted_budget_and_directories", "generator_archive_download",
                     "generator_member_authentication_and_staging", "cached_VGG_selected_member_staging",
                     "public_model_runtime_imports", "VGG_strict_load", "generator_strict_load",
                     "public_statistics_and_latent_bounds", "single_fixed_non_neural_fixture",
                     "one_full_200_update_reconstruction", "in_memory_output_metadata"):
            report["failed_stage"] = stage
        failure_type = getattr(error, "failure_type", None)
        if failure_type in ("ValueError", "RuntimeError", "TimeoutError", "MemoryError", "OSError",
                            "FileNotFoundError", "FileExistsError", "KeyError", "TypeError",
                            "UnpicklingError", "BadZipFile", "HTTPError", "URLError"):
            report["cause_type"] = failure_type
    report.update(elapsed_seconds=time.monotonic() - started,
        peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    _publish(RUN_DIR / "report.json", report)
    return 130 if interrupted else (0 if report["status"] == "complete_public_runtime_qualification" else 1)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        choice = parser.add_mutually_exclusive_group()
        choice.add_argument("--check-admission", action="store_true")
        choice.add_argument("--remaining-seconds", action="store_true")
        args = parser.parse_args()
        if args.check_admission or args.remaining_seconds:
            context, budget = admit()
            if args.remaining_seconds:
                print(max(1, int(budget.allocation["deadline_epoch"] - time.time())))
            else:
                print(json.dumps({"status": "admitted_CPU_public_qualification", "allocation": budget.allocation}))
            result = 0
        else:
            result = run_once()
    except BaseException as error:
        print(json.dumps({"status": "reconstruction_qualification_entrypoint_refused", "error_type": type(error).__name__}))
        result = 130 if isinstance(error, KeyboardInterrupt) else 2
    raise SystemExit(result)
