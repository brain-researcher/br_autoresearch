"""Fixed first original Finding A readout, with stdlib-first admission.

The separately selected batch owns dependency47370327, prerequisite checks and
this isolated empirical child. The child establishes its fatal40-GiB file cap
and default SIGXFSZ disposition before scientific imports or input access.
SIGXFSZ cannot be recovered inside the child: the parent must record that
resource failure and preserve partial files. There is no retry or resume.
Successful record publication still leaves the complete quantitative Finding A
not evaluable because required original pairs are absent.

Context/operation, actual allocation, deadline, RSS and fatal file policy are
checked on every resolver call. Grant, completed-check records and aggregate
storage are checked at each operation's first call and on the next callback
after a one-second interval thereafter. This operational polling avoids metadata
I/O on every per-draw numerical block; it changes no scientific calculation.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

import run_brain_pair as core


BASE = Path(__file__).resolve().parents[3]
SOURCE, TARGET = "sub-03", "sub-01"
RESOURCES = {"cpus": 3, "memory_gib": 16, "minutes": 4320, "gpus": 0,
    "deadline_seconds": 255600, "rss_limit_bytes": 16106127360,
    "durable_bytes": 51539607552, "scratch_bytes": 4294967296,
    "collection_bytes": 42949672960}
FIXED = {"schema_version": "ep21.finding_a_first_readout.v1",
    "operation": "run_finding_a_first_readout", "jobs": 1, "partition": "russpold",
    "source": SOURCE, "target": TARGET, "original_content_pair_index": 4,
    "dependency_job_id": "47370327", "test_images": 224, "layers": 19,
    "decoded_slots": 38, "total_slots": 855, "image_bootstrap_draws": 1000,
    "dyadic_bootstrap_draws": 1000, "draw_batch_size": 16, "automatic_retry": False,
    "production_training": False, "model_publication": False,
    "decoded_feature_publication": True, "scientific_scoring": True, "model_policy": "original_locked"}
INPUT_SCOPE = {"source_tau_test": SOURCE, "target_tau_test": False, "tau_train": False,
    "decoder_payload_participants": [SOURCE, TARGET],
    "normalization_participants": ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"],
    "common_true_features": "tau_test",
    "original_content_model": "outputs/runs/content_pair_training/cuda001/final_content_pair.pt",
    "source_training_response_matrix": False, "target_response_matrix": False,
    "regular_images": False, "OOD": False}
OPERATIONS = (FIXED["operation"], "bind_finding_a_inputs", "read_finding_a_tau_test", "read_finding_a_truth")
CHECK_RECORDS = ("empirical_input_checks_exit", "readout_composition_checks_exit",
                 "batched_inference_checks_exit", "readout_admission_checks_exit")
AdmissionError, _require, _same = core.AdmissionError, core._require, core._same


@dataclass(frozen=True)
class ReadoutStage:
    profile_path: Path
    run_dir: Path
    final_path: Path
    scratch_name: str = "ep21_finding_a_first001"
    allocation_start_deadline: bool = True

    @property
    def resources(self):
        return dict(RESOURCES)

    @property
    def publication_paths(self):
        return (self.final_path,
                self.final_path.with_suffix(".decoded_feature_manifest.json"),
                self.final_path.with_suffix(".finding_a_report.json"))


def _stage():
    run = BASE / "outputs/runs/finding_a_feature_decoding/first001"
    return ReadoutStage(BASE / "outputs/finding_a_first_readout_authorization.json",
                        run, run / "finding_a_collection.h5")


def _profile(stage):
    value = json.loads(core._read_text(stage.profile_path))
    _require(isinstance(value, dict) and value.get("authorized") is True, "readout_profile_not_activated")
    _require(isinstance(value.get("approval_source"), str) and bool(value["approval_source"].strip()),
             "explicit_first_readout_approval_required")
    for key, expected in FIXED.items():
        _require(_same(value.get(key), expected), "fixed_first_readout_scope_changed")
    _require(_same(value.get("resources"), RESOURCES) and _same(value.get("input_scope"), INPUT_SCOPE),
             "first_readout_resources_or_roles_changed")
    return {**FIXED, "resources": dict(RESOURCES), "input_scope": INPUT_SCOPE,
            "authorized": True, "approval_source": value["approval_source"]}


def _checks_completed(stage):
    found = {key: [] for key in CHECK_RECORDS}
    for line in core._read_text(stage.run_dir / "batch_status.txt").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in found:
            found[key].append(value)
    _require(all(values == ["0"] for values in found.values()),
             "all_eleven_selected_checks_must_pass_once_before_payload")


def _require_file_policy():
    soft, _ = resource.getrlimit(resource.RLIMIT_FSIZE)
    _require(soft == RESOURCES["collection_bytes"], "exact_fatal_collection_file_cap_required")
    _require(signal.getsignal(signal.SIGXFSZ) == signal.SIG_DFL,
             "default_fatal_collection_signal_required")


def _configure_file_policy():
    """Only this isolated empirical CLI child changes its own process policy."""
    _, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    _require(hard == resource.RLIM_INFINITY or hard >= RESOURCES["collection_bytes"],
             "parent_hard_file_limit_cannot_admit_collection")
    resource.setrlimit(resource.RLIMIT_FSIZE, (RESOURCES["collection_bytes"], hard))
    signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
    _require_file_policy()


@dataclass(frozen=True)
class ReadoutContext:
    profile: dict
    allocation: dict
    scratch: Path
    stage: ReadoutStage


class ReadoutBudget(core.BrainPairBudget):
    """Reuse actual-allocation absolute deadline, RSS and aggregate storage checks."""
    def __init__(self, context):
        super().__init__(context)
        self._next_policy_check = 0.
        self._checked_operations = set()

    def check_policy(self, operation):
        now = time.monotonic()
        if operation not in self._checked_operations or now >= self._next_policy_check:
            _require(_profile(self.context.stage) == self.context.profile, "readout_admission_changed")
            _checks_completed(self.context.stage)
            self.check_storage()
            self._checked_operations.add(operation)
            self._next_policy_check = now + 1.


def admit():
    stage = _stage()
    profile = _profile(stage)
    context = ReadoutContext(profile, core._allocation(stage), core._scratch(stage), stage)
    budget = ReadoutBudget(context)
    budget.check_storage()
    return context, budget


def resolver(context, *, operation, source, target, budget, execution_options):
    _require(isinstance(context, ReadoutContext) and isinstance(budget, ReadoutBudget)
             and budget.context is context and budget.allocation == context.allocation
             and context.stage == _stage(), "same_fixed_readout_context_required")
    _require(source == SOURCE and target == TARGET and operation in OPERATIONS,
             "fixed_readout_pair_and_operation_required")
    if operation == FIXED["operation"]:
        expected = {"collection_path": context.stage.final_path, "maximum_bytes": RESOURCES["collection_bytes"]}
        _require(execution_options is not None and _same(dict(execution_options), expected),
                 "fixed_readout_collection_options_required")
    else:
        _require(execution_options is None, "fixed_input_operations_have_no_options")
    budget.check()
    _require_file_policy()
    budget.check_policy(operation)


def _load_operation():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import numpy as np
    import h5py
    from finding_a_first_readout import run_finding_a_first_readout
    return run_finding_a_first_readout, {"python": sys.version.split()[0],
        "numpy": str(np.__version__), "torch": str(torch.__version__), "h5py": str(h5py.__version__)}


def _publish(path, value, budget):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _require(len(data) <= 65536, "bounded_operational_readout_record_required")
    _require(core._bytes_under(budget.context.stage.run_dir, RESOURCES["durable_bytes"]) + len(data)
             <= RESOURCES["durable_bytes"], "readout_operational_record_exceeds_total_cap")
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def run_once():
    context, budget = admit()
    stage = context.stage
    for path in (*stage.publication_paths, stage.run_dir / "report.json", stage.run_dir / "claim.json"):
        _require(not os.path.lexists(path) and not os.path.lexists(str(path) + ".partial"),
                 "existing_first_readout_attempt_no_retry")
    _checks_completed(stage)
    _configure_file_policy()
    options = {"collection_path": stage.final_path, "maximum_bytes": RESOURCES["collection_bytes"]}
    resolver(context, operation=FIXED["operation"], source=SOURCE, target=TARGET,
             budget=budget, execution_options=options)
    _publish(stage.run_dir / "claim.json", {"operation": FIXED["operation"], "source": SOURCE, "target": TARGET,
        "allocation": context.allocation, "approval_source": context.profile["approval_source"],
        "model_policy": "original_locked", "collection_path": str(stage.final_path),
        "automatic_retry": False, "resumable": False}, budget)
    phase, interrupted = "load_operation", False
    started_wall, started_cpu = time.monotonic(), time.process_time()
    report = {"kind": "ep21_first_finding_a_readout_operational_report", "source": SOURCE, "target": TARGET,
        "model_policy": "original_locked", "status": "incomplete", "artifacts_published": False,
        "published_paths": None, "decoded_slots": None, "unavailable_slots": None,
        "quantitative_finding": "not_evaluable", "comparability": "not_evaluable",
        "model_training_performed": False, "automatic_retry": False, "resumable": False,
        "runtime": None, "allocation_identity": {key: context.allocation[key] for key in ("job_id", "hostname")}}
    try:
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
            operation, runtime = _load_operation()
            report["runtime"] = runtime
            phase = "decode_infer_publish"
            result = operation(context, require_admitted_context=resolver, budget=budget, **options)
            _require(isinstance(result, tuple) and result == stage.publication_paths
                     and all(path.is_file() and not path.is_symlink() for path in result),
                     "complete_fixed_readout_publication_required")
            report.update(artifacts_published=True, published_paths=list(map(str, result)),
                          decoded_slots=38, unavailable_slots=817)
            phase = "closeout"
            budget.check()
            budget.check_storage()
        report["status"] = "complete_requested_readout_record"
    except BaseException as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        report.update(error_type=type(error).__name__, failed_stage=phase,
                      resource_limit_failure=isinstance(error, (TimeoutError, MemoryError)))
        if phase == "decode_infer_publish":
            report["artifacts_published"] = None
    report.update(elapsed_wall_seconds=time.monotonic() - started_wall,
                  process_cpu_seconds=time.process_time() - started_cpu,
                  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    _publish(stage.run_dir / "report.json", report, budget)
    return 130 if interrupted else 0 if report["status"] == "complete_requested_readout_record" else 1


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
            else:
                print(json.dumps({"status": "first_readout_allocation_admitted", "job_id": context.allocation["job_id"],
                                  "payload_read": False, "attempt_claimed": False}))
            return 0
        return run_once()
    except BaseException as error:
        print(json.dumps({"status": "first_readout_entrypoint_refused", "error_type": type(error).__name__}))
        return 130 if isinstance(error, KeyboardInterrupt) else 2


if __name__ == "__main__":
    raise SystemExit(main())
