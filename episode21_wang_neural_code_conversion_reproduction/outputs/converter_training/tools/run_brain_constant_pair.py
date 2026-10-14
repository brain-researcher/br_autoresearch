"""Separately admitted complete constant-coordinate comparator, one pair/job.

The original comparator, its artifacts and held readout are unchanged. Only
stdlib and existing stdlib admission helpers load before explicit admission.
All20 ordered pairs use fresh adapted fits; no original artifact is relabeled.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import itertools
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time
from types import MappingProxyType

import run_brain_pair as original


BASE = original.BASE
PROFILE = BASE / "outputs/brain_constant_panel_authorization.json"
GUARD = BASE / "outputs/runs/brain_constant_discriminator/attempt002/report.json"
OPERATION = "train_and_publish_brain_constant_pair"
PARTICIPANTS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
PAIRS = tuple(itertools.permutations(PARTICIPANTS, 2))
WIDTHS = dict(zip(PARTICIPANTS, (12295, 12897, 10891, 12023, 12057)))
RESOURCES = dict(cpus=3, memory_gib=16, minutes=360, gpus=0,
    deadline_seconds=21000, rss_limit_bytes=16106127360,
    durable_bytes=2147483648, scratch_bytes=268435456, final_artifact_bytes=805306368)
require = original._require


@dataclass(frozen=True)
class ConstantStage(original.BrainPairStage):
    @property
    def resources(self):
        return dict(RESOURCES)


def stage(index):
    require(type(index) is int and 0 <= index < 20, "original_pair_index_required")
    source, target = PAIRS[index]
    stem = f"pair{index:03d}_{source.replace('-', '')}_to_{target.replace('-', '')}_001"
    run = BASE / "outputs/runs/brain_constant_training" / stem
    return ConstantStage(f"constant-{index:03d}", PROFILE, run,
        run / "final_brain_constant_pair.pt", "ep21_brain_constant_" + stem,
        source, target, WIDTHS[source], WIDTHS[target], index, True, 3, 16, RESOURCES["rss_limit_bytes"])


def profile():
    value = json.loads(original._read_text(PROFILE))
    expected = dict(schema_version="ep21.brain_constant_panel.v1", authorized=True,
        state="separately_regenerated_adapted_training_lock",
        normalization_policy="float32_ddof1_exact_constant_exact_mean_zero_sd_effective_one",
        inherited_lock="outputs/protocol_lock.yaml",
        operation=OPERATION, training_images=897, cv_fits=50, final_fits=1,
        original_pair_order=[list(pair) for pair in PAIRS], native_widths=WIDTHS,
        resources=RESOURCES, fresh_amended_fits=20, original_artifact_reuse=False,
        production_training=True, model_publication=True, scoring=False,
        tau_test_access=False, OOD_access=False, automatic_retry=False)
    require(all(original._same(value.get(key), expected_value)
                for key, expected_value in expected.items()), "constant_comparator_grant_changed_or_inactive")
    require(isinstance(value.get("approval_source"), str) and bool(value["approval_source"]),
            "explicit_constant_comparator_selection_required")
    guard = json.loads(original._read_text(GUARD))
    require(guard.get("kind") == "ep21_brain_constant_discriminator"
        and guard.get("status") == "completed" and guard.get("guard_satisfied") is True
        and guard.get("first_fold_only") is True and guard.get("fits_performed") == 0
        and guard.get("tau_test_accessed") is False, "completed_adopted_guard_required")
    return value


def allocation():
    """Bind actual Slurm environment; array workers never poll the scheduler."""
    host = socket.gethostname().split(".")[0]
    job = os.environ.get("SLURM_JOB_ID", "")
    require(job.isdigit() and "-ln" not in host, "actual_compute_allocation_required")
    for key, expected in dict(SLURM_CPUS_PER_TASK="3", SLURM_MEM_PER_NODE="16384",
            SLURM_JOB_NUM_NODES="1", SLURM_JOB_CPUS_PER_NODE="3",
            SLURM_NTASKS="1", SLURM_JOB_PARTITION="russpold").items():
        require(os.environ.get(key) == expected, "actual_constant_fit_resources_required")
    require(all(os.environ.get(key) == "1" for key in original.THREADS), "one_numerical_thread_required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not os.environ.get("SLURM_JOB_GPUS", "")
        and not os.environ.get("SLURM_STEP_GPUS", "")
        and all(os.environ.get(key, "0") in ("", "0") for key in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE")),
        "CPU_only_constant_fit_required")
    start, end = (int(os.environ[key]) for key in ("SLURM_JOB_START_TIME", "SLURM_JOB_END_TIME"))
    # The submitted script fixes six hours; Slurm may add seconds of EndTime
    # slack. Internal start+21000 remains stricter than that reservation.
    require(start <= time.time() < end and start < end,
            "bounded_actual_allocation_clock_required")
    return dict(job_id=job, hostname=host, job_start_epoch=start, job_end_epoch=end,
        deadline_epoch=min(start + 21000, end - 60), cpus=3, memory_gib=16,
        minutes=360, partition="russpold", gpus=0)


def admit(index):
    selected, authority = stage(index), profile()
    identity = allocation()
    scratch = original._scratch(selected)
    context = original.BrainPairContext(authority, identity, scratch, selected)
    budget = original.BrainPairBudget(context)
    budget.check_storage()
    return context, budget


def resolver(context, *, operation, source, target, budget, execution_options):
    require(isinstance(context, original.BrainPairContext)
        and isinstance(budget, original.BrainPairBudget) and budget.context is context
        and budget.allocation == context.allocation, "same_constant_fit_context_required")
    require(context.stage == stage(context.stage.pair_index) and profile() == context.profile,
            "constant_fit_selection_changed")
    require((source, target) == (context.stage.source, context.stage.target), "selected_full_pair_required")
    budget.check()
    budget.check_storage()
    if operation == "bind_brain_inputs":
        require(execution_options is None, "binding_has_no_fit_options")
    elif operation == OPERATION:
        require(execution_options is not None and original._same(dict(execution_options),
            dict(final_path=context.stage.final_path, maximum_bytes=RESOURCES["final_artifact_bytes"])),
            "selected_adapted_publication_required")
    else:
        raise original.AdmissionError("operation_not_authorized")


def train_and_publish(bound, *, final_path, maximum_bytes):
    import empirical_brain_inputs as inputs
    import brain_loss_constant as brain
    import final_brain_constant_artifact as final
    require(isinstance(bound, inputs.BoundBrainInputs), "bound_tau_train_inputs_required")
    pair = bound.source.participant, bound.target.participant
    options = MappingProxyType(dict(final_path=Path(final_path), maximum_bytes=maximum_bytes))

    def check_operation():
        result = bound.require_admitted_context(bound.execution_context, operation=OPERATION,
            source=pair[0], target=pair[1], budget=bound.budget, execution_options=options)
        require(result is None, "resolver_must_raise_or_return_None")
        inputs.assembly._require_reader_context(bound.budget)
        require((bound.budget.allocation["job_id"], bound.budget.allocation["hostname"])
            == bound.allocation_identity, "constant_fit_cannot_move_allocations")

    check_operation()
    path = final._file_context(options["final_path"], bound.budget, maximum_bytes)
    require(path.is_absolute() and path.parent.is_dir(), "explicit_existing_publication_parent_required")
    require(type(maximum_bytes) is int and maximum_bytes == RESOURCES["final_artifact_bytes"],
            "fixed_adapted_artifact_cap_required")
    for candidate in (path, path.with_name(path.name + ".partial")):
        require(not os.path.lexists(candidate), "existing_model_no_retry")
    fitted = brain.fit_brain_pair(bound.source, bound.target, bound.folds,
        source_native_ids=bound.source.native_voxel_ids,
        target_native_ids=bound.target.native_voxel_ids, check=bound.budget.check)
    check_operation()
    artifact = final.prepare_final_brain_constant_artifact(fitted, expected_pair=pair,
        authenticated_tau_train_image_ids=bound.folds.image_ids,
        source_native_ids=bound.source.native_voxel_ids,
        target_native_ids=bound.target.native_voxel_ids, check=bound.budget.check)
    summary = fitted.summary
    del fitted
    check_operation()
    published = final.write_final_brain_constant_artifact(path, artifact,
        budget=bound.budget, maximum_bytes=maximum_bytes)
    return published, summary


def failure_details(exc, selected):
    """Allowlisted reason/role/fold only; never expose values or free text."""
    inherited = original._brain_failure_details(exc, selected)
    if inherited:
        return inherited
    module = sys.modules.get("brain_loss")
    if (type(exc) is not getattr(module, "BrainLossNotEvaluable", None)
            or len(exc.args) != 1 or type(exc.args[0]) is not str or len(exc.args[0]) > 256):
        return {}
    for code in ("nonfinite_required_training_response", "nonfinite_or_negative_required_training_scale",
                 "zero_sd_nonconstant_required_training_column", "zero_sd_constant_mean_mismatch"):
        for role in ("source", "target"):
            for fold in range(5):
                if exc.args[0] == f"{code}:{role}_fold_{fold}":
                    return {"brain_loss_failure": dict(reason_code=code, role=role,
                                                       context="fold_training", fold=fold)}
            if exc.args[0] == f"{code}:{role}_final":
                return {"brain_loss_failure": dict(reason_code=code, role=role, context="final")}
    return {}


def run_once(index):
    context, budget = admit(index)
    selected = context.stage
    for path in (selected.run_dir / "claim.json", selected.run_dir / "report.json",
                 selected.final_path, selected.final_path.with_name(selected.final_path.name + ".partial")):
        require(not os.path.lexists(path), "existing_attempt_no_retry")
    original._publish(selected.run_dir / "claim.json", dict(operation=OPERATION,
        source=selected.source, target=selected.target, pair_index=index,
        allocation=context.allocation, approval_source=context.profile["approval_source"],
        automatic_retry=False), selected)
    started, cpu = time.monotonic(), time.process_time()
    phase, interrupted = "scientific_imports", False
    report = dict(kind="ep21_brain_constant_operational_report", source=selected.source,
        target=selected.target, pair_index=index, model_published=False, completed_fits=None,
        original_artifact_reused=False, tau_test_accessed=False, scoring=False,
        automatic_retry=False, allocation_identity=context.allocation)
    try:
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
            import torch
            import numpy as np
            import empirical_brain_inputs as inputs
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            report["runtime"] = dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__)
            phase = "bind_inputs"
            bound = inputs.bind_brain_inputs(context, require_admitted_context=resolver,
                source=selected.source, target=selected.target, budget=budget)
            original._bound_scope(bound, selected)
            phase = "fit_and_publish"
            path, summary = train_and_publish(bound, final_path=selected.final_path,
                maximum_bytes=RESOURCES["final_artifact_bytes"])
            require(path == selected.final_path and path.is_file()
                and summary.available is True and type(summary.completed_fits) is int
                and summary.completed_fits == 51, "complete_adapted_pair_required")
            report.update(model_published=True, completed_fits=51, published_path=str(path))
            phase = "closeout"
            budget.check_storage()
        report["status"] = "complete_brain_constant_pair"
    except BaseException as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        report.update(status="incomplete", error_type=type(exc).__name__, failed_stage=phase)
        report.update(failure_details(exc, selected))
        if phase == "fit_and_publish":
            report["model_published"] = None
    report.update(elapsed_wall_seconds=time.monotonic() - started,
        process_cpu_seconds=time.process_time() - cpu,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    original._publish(selected.run_dir / "report.json", report, selected)
    return 130 if interrupted else 0 if report["status"] == "complete_brain_constant_pair" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-index", type=int, choices=range(20), required=True)
    parser.add_argument("--check-admission", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.check_admission:
            context, _ = admit(args.pair_index)
            print(json.dumps(dict(status="admitted", payload_read=False, job_id=context.allocation["job_id"])))
            return 0
        return run_once(args.pair_index)
    except BaseException as exc:
        print(json.dumps(dict(status="entrypoint_refused", error_type=type(exc).__name__)))
        return 130 if isinstance(exc, KeyboardInterrupt) else 2


if __name__ == "__main__":
    raise SystemExit(main())
