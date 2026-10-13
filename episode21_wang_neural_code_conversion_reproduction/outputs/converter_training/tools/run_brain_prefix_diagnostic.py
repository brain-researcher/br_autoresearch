"""Separately admitted failure diagnosis inside one fixed parent stage.

The parent proves allocation identity only; this profile alone permits the
original pair's tau-train prefix. No fit, solve, score or model is requested.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import json
import os
from pathlib import Path
import resource
import sys
import time

import run_content_cuda_measurement as parent
from run_content_measurement import _read_text, _require, _same, _bytes_under
from run_brain_pair import _brain_failure_details, _bound_scope

BASE = Path(__file__).resolve().parents[3]
PROFILE = BASE / "outputs/brain_prefix_diagnostic_authorization.json"
RUN_DIR = BASE / "outputs/runs/brain_pair_prefix_diagnostic/attempt001"
RECONSTRUCTION_PROFILE = BASE / "outputs/brain_prefix_diagnostic_reconstruction_authorization.json"
RECONSTRUCTION_RUN_DIR = BASE / "outputs/runs/brain_pair_prefix_diagnostic/attempt002"
STAGES = ("original", "reconstruction-002")
FIXED = {"schema_version": "ep21.brain_prefix_diagnostic.v1",
    "operation": "diagnose_brain_first_fold", "attempts": 1,
    "source": "sub-03", "target": "sub-01", "training_images": 897,
    "source_native_voxels": 12897, "target_native_voxels": 12295,
    "fits": 0, "scoring": False, "model_publication": False, "automatic_retry": False,
    "parent_authorization": "outputs/content_cuda_measurement_authorization.json"}
RESOURCES = {"deadline_seconds": 300, "rss_limit_bytes": 6979321856, "durable_bytes": 1048576}
INPUT_SCOPE = {"source_tau_train": "sub-03", "target_tau_train": "sub-01",
    "tau_test": False, "OOD": False, "subject_unique": False,
    "features": False, "decoder_payloads": False}


@dataclass(frozen=True)
class DiagnosticStage:
    name: str
    profile_path: Path
    run_dir: Path
    parent_authorization: str


def _stage(name="original"):
    _require(name in STAGES, "unknown_brain_diagnostic_stage")
    if name == "reconstruction-002":
        return DiagnosticStage(name, RECONSTRUCTION_PROFILE, RECONSTRUCTION_RUN_DIR,
            "outputs/reconstruction_qualification_authorization.json")
    return DiagnosticStage(name, PROFILE, RUN_DIR, FIXED["parent_authorization"])


def _profile(stage):
    value = json.loads(_read_text(stage.profile_path))
    _require(value.get("authorized") is True and isinstance(value.get("approval_source"), str)
             and bool(value["approval_source"].strip()), "separate_brain_diagnostic_approval_required")
    fixed = {**FIXED, "parent_authorization": stage.parent_authorization}
    for key, expected in {**fixed, "resources": RESOURCES, "input_scope": INPUT_SCOPE}.items():
        _require(_same(value.get(key), expected), "brain_diagnostic_scope_changed")
    return {**fixed, "resources": dict(RESOURCES), "input_scope": dict(INPUT_SCOPE),
            "approval_source": value["approval_source"], "authorized": True}


def _admit_parent(stage):
    if stage.name == "original":
        return parent.admit()
    # This branch is reached only after the separate tau-prefix grant passes.
    # The reconstruction parent performs CPU-only allocation admission; it
    # does not initialize CUDA or confer neural access itself.
    preparation = BASE / "outputs/reconstruction_preparation"
    if str(preparation) not in sys.path:
        sys.path.insert(0, str(preparation))
    from run_reconstruction_qualification import admit
    return admit(component="brain_prefix")


@dataclass(frozen=True)
class DiagnosticContext:
    profile: dict
    parent_context: object
    stage: DiagnosticStage


class DiagnosticBudget:
    def __init__(self, context, parent_budget):
        self.context, self.parent_budget = context, parent_budget
        self.allocation = dict(parent_budget.allocation)
        self.deadline = time.monotonic() + RESOURCES["deadline_seconds"]
        self.check()

    def check(self):
        self.parent_budget.check()
        if time.monotonic() >= self.deadline:
            raise TimeoutError("brain_prefix_diagnostic_deadline")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > RESOURCES["rss_limit_bytes"]:
            raise MemoryError("brain_prefix_diagnostic_RSS_limit")

    def check_storage(self):
        self.parent_budget.check_storage()
        _bytes_under(self.context.stage.run_dir, RESOURCES["durable_bytes"])


def resolver(context, *, operation, source, target, budget, execution_options):
    _require(isinstance(context, DiagnosticContext) and isinstance(budget, DiagnosticBudget)
             and budget.context is context and budget.parent_budget.context is context.parent_context,
             "separate_brain_diagnostic_context_required")
    _require(context.stage == _stage(context.stage.name)
             and _profile(context.stage) == context.profile, "brain_diagnostic_admission_changed")
    _require(source == "sub-03" and target == "sub-01"
             and operation in ("bind_brain_inputs", "diagnose_brain_first_fold")
             and execution_options is None, "brain_diagnostic_operation_not_authorized")
    budget.check()


def _publish(path, value, *, run_dir):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    _require(len(raw) <= 65536 and _bytes_under(run_dir, RESOURCES["durable_bytes"])
             + len(raw) <= RESOURCES["durable_bytes"], "brain_diagnostic_output_limit")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def run_once(stage_name="original"):
    selected = _stage(stage_name)
    profile = _profile(selected)  # Separate tau authority precedes either parent's admission.
    parent_context, parent_budget = _admit_parent(selected)
    context = DiagnosticContext(profile, parent_context, selected)
    budget = DiagnosticBudget(context, parent_budget)
    budget.check_storage()
    run_dir = selected.run_dir
    _require(not os.path.lexists(run_dir / "report.json"), "diagnostic_attempt_already_closed")
    _publish(run_dir / "claim.json", {"operation": FIXED["operation"],
        "diagnostic_stage": selected.name, "parent_authorization": selected.parent_authorization,
        "approval_source": profile["approval_source"], "allocation": budget.allocation,
        "automatic_retry": False, "additional_allocation_charge": False}, run_dir=run_dir)
    report = {"kind": "ep21_brain_first_fold_diagnostic", "source": "sub-03", "target": "sub-01",
        "diagnostic_stage": selected.name, "parent_authorization": selected.parent_authorization,
        "fits_performed": 0, "scores_produced": False, "model_published": False,
        "allocation_identity": {key: budget.allocation[key] for key in ("job_id", "hostname")}}
    stage, started = "load_operations", time.monotonic()
    interrupted = False
    try:
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
            from empirical_brain_inputs import bind_brain_inputs
            from brain_prefix_diagnostic import diagnose_brain_first_fold
            budget.check()
            stage = "bind_inputs"
            bound = bind_brain_inputs(context, require_admitted_context=resolver,
                source="sub-03", target="sub-01", budget=budget)
            _bound_scope(bound)
            stage = "first_fold_normalization"
            report.update(diagnose_brain_first_fold(bound))
            budget.check()
            budget.check_storage()
    except BaseException as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        details = _brain_failure_details(error)
        report.update(status="localized" if details else "incomplete",
            error_type=type(error).__name__, failed_stage=stage, **details)
    report.update(elapsed_seconds=time.monotonic() - started,
                  peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    _publish(run_dir / "report.json", report, run_dir=run_dir)
    return 130 if interrupted else (0 if report["status"] in ("localized", "not_localized") else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="original")
    args = parser.parse_args()
    try:
        result = run_once(args.stage)
    except BaseException as error:
        print(json.dumps({"status": "brain_diagnostic_entrypoint_refused", "error_type": type(error).__name__}))
        result = 130 if isinstance(error, KeyboardInterrupt) else 2
    raise SystemExit(result)
