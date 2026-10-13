"""Nine separately granted original brain pairs; one exclusive attempt each.

This entrypoint reuses the existing fixed brain-pair admission/resource helpers
and scientific binder/driver without modifying their live files. Only canonical
pairs 9/10/11/13/14/15/17/18/19 can be selected. Each stage has its own six-hour
allocation and absolute 21,000-second work deadline, including earlier checks.
No arbitrary pair, resource override, automatic continuation, retry, resume or
evaluation. This launcher changes no training, normalization or model policy.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import json
import os
import resource
import time

import run_brain_pair as core


BASE = core.BASE
STAGES = ("remaining-009", "remaining-010", "remaining-011", "remaining-013",
          "remaining-014", "remaining-015", "remaining-017", "remaining-018",
          "remaining-019")


@dataclass(frozen=True)
class BrainPairStage(core.BrainPairStage):
    @property
    def resources(self):
        return {**super().resources, "minutes": 360, "deadline_seconds": 21000}


BrainPairContext = core.BrainPairContext
BrainPairBudget = core.BrainPairBudget
_require = core._require
_same = core._same
_profile = core._profile
_allocation = core._allocation
_scratch = core._scratch
_publish = core._publish
_bound_scope = core._bound_scope
_brain_failure_details = core._brain_failure_details
_load_operations = core._load_operations


def _stage(name):
    _require(name in STAGES, "unknown_fixed_brain_remaining_stage")
    # Original itertools.permutations participant order; full native axes from
    # the existing regular_neural_structural_report tau_train metadata.
    index, source, target, source_width, target_width, stem = {
        "remaining-009": (9, "sub-05", "sub-03", 10891, 12897, "sub05_to_sub03_001"),
        "remaining-010": (10, "sub-05", "sub-06", 10891, 12023, "sub05_to_sub06_001"),
        "remaining-011": (11, "sub-05", "sub-07", 10891, 12057, "sub05_to_sub07_001"),
        "remaining-013": (13, "sub-06", "sub-03", 12023, 12897, "sub06_to_sub03_001"),
        "remaining-014": (14, "sub-06", "sub-05", 12023, 10891, "sub06_to_sub05_001"),
        "remaining-015": (15, "sub-06", "sub-07", 12023, 12057, "sub06_to_sub07_001"),
        "remaining-017": (17, "sub-07", "sub-03", 12057, 12897, "sub07_to_sub03_001"),
        "remaining-018": (18, "sub-07", "sub-05", 12057, 10891, "sub07_to_sub05_001"),
        "remaining-019": (19, "sub-07", "sub-06", 12057, 12023, "sub07_to_sub06_001"),
    }[name]
    profile = BASE / f"outputs/brain_pair_remaining_{index:03d}_authorization.json"
    run_dir = BASE / "outputs/runs/brain_pair_training" / stem
    return BrainPairStage(name, profile, run_dir, run_dir / "final_brain_pair.pt",
        "ep21_brain_pair_" + stem, source, target, source_width, target_width, index,
        True, 3, 16, 16106127360)


def admit(stage_name):
    selected = _stage(stage_name)
    profile = _profile(selected)  # Inactive/changed grants refuse before allocation/imports.
    context = BrainPairContext(profile, _allocation(selected), _scratch(selected), selected)
    budget = BrainPairBudget(context)
    budget.check_storage()
    return context, budget


def resolver(context, *, operation, source, target, budget, execution_options):
    _require(isinstance(context, BrainPairContext) and isinstance(budget, BrainPairBudget)
             and budget.context is context and budget.allocation == context.allocation,
             "same_brain_roster_context_required")
    _require(context.stage == _stage(context.stage.name)
             and _profile(context.stage) == context.profile, "brain_roster_admission_changed")
    budget.check()
    budget.check_storage()
    _require(source == context.stage.source and target == context.stage.target,
             "fixed_brain_roster_pair_required")
    if operation == "bind_brain_inputs":
        _require(execution_options is None, "binding_has_no_execution_options")
    elif operation == core.FIXED["operation"]:
        expected = {"final_path": context.stage.final_path,
                    "maximum_bytes": context.stage.resources["final_artifact_bytes"]}
        _require(execution_options is not None and _same(dict(execution_options), expected),
                 "fixed_brain_roster_publication_options_required")
    else:
        raise core.AdmissionError("operation_not_authorized")


def run_once(stage_name):
    context, budget = admit(stage_name)
    selected = context.stage
    final, run_dir = selected.final_path, selected.run_dir
    for path in (final, final.with_name(final.name + ".partial"), run_dir / "report.json"):
        _require(not os.path.lexists(path), "existing_attempt_output_no_retry")
    _publish(run_dir / "claim.json", {"operation": core.FIXED["operation"],
        "source": selected.source, "target": selected.target,
        "pair_stage": selected.name, "pair_index": selected.pair_index,
        "allocation": context.allocation, "approval_source": context.profile["approval_source"],
        "automatic_retry": False}, selected)
    phase, interrupted = "load_operations", False
    started_wall, started_cpu = time.monotonic(), time.process_time()
    report = {"kind": "ep21_brain_pair_operational_report",
        "source": selected.source, "target": selected.target,
        "pair_stage": selected.name, "pair_index": selected.pair_index,
        "completed_fits": None, "model_published": False, "published_path": None,
        "resumable": False, "automatic_retry": False, "runtime": None,
        "allocation_identity": {key: context.allocation[key] for key in ("job_id", "hostname")}}
    try:
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
            bind, train, runtime = _load_operations()
            report["runtime"] = runtime
            phase = "bind_inputs"
            bound = bind(context, require_admitted_context=resolver,
                         source=selected.source, target=selected.target, budget=budget)
            _bound_scope(bound, selected)
            phase = "fit_and_publish"
            result = train(bound, final_path=final, maximum_bytes=selected.resources["final_artifact_bytes"])
            _require(result.source == selected.source and result.target == selected.target
                     and result.pair_index == selected.pair_index and result.path == final
                     and result.summary.available is True and type(result.summary.completed_fits) is int
                     and result.summary.completed_fits == 51 and final.is_file(),
                     "complete_fixed_brain_roster_publication_required")
            report.update(completed_fits=51, model_published=True, published_path=str(final))
            phase = "closeout"
            budget.check_storage()
        report.update(status="complete_brain_pair")
    except BaseException as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        report.update(status="incomplete", error_type=type(exc).__name__, failed_stage=phase)
        report.update(_brain_failure_details(exc, selected))
        if phase == "fit_and_publish":
            report["model_published"] = None
    # Preserve the existing categorical-only reporting and conservative fit count.
    report.update(elapsed_wall_seconds=time.monotonic() - started_wall,
        process_cpu_seconds=time.process_time() - started_cpu,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    _publish(run_dir / "report.json", report, selected)
    return 130 if interrupted else 0 if report["status"] == "complete_brain_pair" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
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
