"""Inactive stdlib-first caller for the separately admitted full amended panel.

No grant, resource values or scheduler bindings are selected here. A future
explicit activated phase grant must name finite resources and forty distinct
actual model jobs (array children individually, never an array-parent shorthand).
The existing amended loaders then bind all20 fresh constant-rule brain artifacts,
all20 original content artifacts and five within-source streams exactly.

Reuse existing allocation, scratch, monotonic/absolute deadline, RSS, bounded
inode accounting and exclusive operational publication primitives. All original
readout callers/grants stay untouched. Every operation revalidates allocation
and fatal file policy; grant, dependencies, checks and bounded storage are
refreshed on first use and at most once per second. No retry or resume exists.
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

import run_finding_a_full_readout as original


BASE = Path(__file__).resolve().parents[3]
core = original.core
AdmissionError, _require, _same = original.AdmissionError, original._require, original._same
_text, _status, _job_id = original._text, original._status, original._job_id
_bytes_unique, _publish = original._bytes_unique, original._publish
PARTICIPANTS, PAIRS, PHASES = original.PARTICIPANTS, original.PAIRS, original.PHASES
MODEL_POLICY = "exact_constant_brain_comparator_all20_fresh_fits"
OPERATIONS = {"decode": "run_amended_finding_a_decoded_point",
              "draw": "run_amended_finding_a_draw_batch",
              "finalize": "finalize_amended_finding_a_readout"}
FIXED = {**original.FIXED, "schema_version": "ep21.amended_full_readout.v1",
    "operations": OPERATIONS, "decoded_slots": 855, "unavailable_slots": 0,
    "model_policy": MODEL_POLICY}
# Concurrency and resources are explicit future selections, never old defaults.
del FIXED["maximum_concurrent_draw_tasks"]
INPUT_SCOPE = {"decode": {**original.INPUT_SCOPE["decode"],
    "fresh_constant_rule_brain_models": 20, "original_brain_artifact_reuse": False},
    "draw": dict(original.INPUT_SCOPE["draw"]), "finalize": dict(original.INPUT_SCOPE["finalize"])}
del INPUT_SCOPE["decode"]["original_non_sub01_brain_models"]
RESOURCE_FIELDS = frozenset(("cpus", "memory_gib", "minutes", "gpus", "deadline_seconds",
    "rss_limit_bytes", "collection_bytes", "aggregate_bytes", "durable_bytes", "scratch_bytes", "batch_bytes"))
CHECK_RECORDS = ("amended_readout_checks_exit", "amended_readout_check_methods")


@dataclass(frozen=True)
class AmendedStage(original.FullReadoutStage):
    resource_items: tuple = ()

    @property
    def resources(self):
        _require(bool(self.resource_items), "explicit_selected_phase_resources_required")
        return dict(self.resource_items)


def _stage(phase, batch_index=None, resources=None):
    _require(phase in PHASES, "fixed_amended_readout_phase_required")
    _require((phase == "draw" and type(batch_index) is int and 0 <= batch_index < 63)
        or (phase != "draw" and batch_index is None), "exact_draw_index_or_no_index_required")
    root = BASE / "outputs/runs/finding_a_feature_decoding/amended_full001"
    name = f"draw_{batch_index:03d}" if phase == "draw" else phase
    run = root / "draws" / f"batch_{batch_index:03d}" if phase == "draw" else root / phase
    return AmendedStage(phase, batch_index, BASE / "outputs/amended_full_readout_authorization.json",
        root, run, "ep21_amended_full001_" + name, True,
        () if resources is None else tuple(sorted(resources.items())))


def _resource_plan(resources):
    _require(isinstance(resources, dict) and set(resources) == set(PHASES), "all_three_explicit_resource_phases_required")
    result = {}
    for phase in PHASES:
        values = resources[phase]
        _require(isinstance(values, dict) and set(values) == RESOURCE_FIELDS
            and type(values["gpus"]) is int and values["gpus"] == 0
            and all(type(value) is int and value > 0 for key, value in values.items() if key != "gpus"),
            "finite_positive_CPU_phase_resources_required")
        _require(values["deadline_seconds"] <= values["minutes"] * 60 - 60
            and values["rss_limit_bytes"] < values["memory_gib"] * 1024**3
            and values["batch_bytes"] <= values["collection_bytes"] <= values["aggregate_bytes"]
            and values["durable_bytes"] <= values["aggregate_bytes"], "inconsistent_phase_resource_bounds")
        result[phase] = dict(values)
    for key in ("collection_bytes", "aggregate_bytes", "batch_bytes"):
        _require(len({result[phase][key] for phase in PHASES}) == 1, "shared_publication_caps_must_agree")
    return result


def _profile(stage):
    value = json.loads(_text(stage.profile_path))
    _require(isinstance(value, dict) and value.get("authorized") is True, "amended_readout_profile_not_activated")
    _require(isinstance(value.get("approval_source"), str) and value["approval_source"].strip(),
             "separate_amended_readout_approval_required")
    for key, expected in FIXED.items():
        _require(_same(value.get(key), expected), "fixed_amended_readout_scope_changed")
    _require(_same(value.get("input_scope"), INPUT_SCOPE), "amended_readout_input_roles_changed")
    parallel = value.get("maximum_concurrent_draw_tasks")
    _require(type(parallel) is int and 1 <= parallel <= 63, "explicit_finite_draw_concurrency_required")
    return {**FIXED, "resources": _resource_plan(value.get("resources")), "input_scope": INPUT_SCOPE,
        "maximum_concurrent_draw_tasks": parallel, "authorized": True, "approval_source": value["approval_source"]}


def _bindings(stage, allocation):
    """Existing procedural dependency format, now forty distinct actual parents.

    The coordinator verifies all completed model jobs and captures individual
    array-child job identities; forty repeated references to a parent array
    cannot satisfy this binding. Model artifact identity is checked by loaders.
    """
    value = json.loads(_text(stage.root / "dependency_bindings.json"))
    _require(isinstance(value, dict) and set(value) == set(PHASES), "exact_phase_dependency_binding_required")
    decode = value["decode"]
    _require(isinstance(decode, dict) and set(decode) == {"job_id", "afterok_job_ids", "completed_model_job_ids"},
             "decode_dependency_binding_required")
    active, completed = decode["afterok_job_ids"], decode["completed_model_job_ids"]
    _require(isinstance(active, list) and isinstance(completed, list), "active_and_completed_model_jobs_required")
    parents = active + completed
    _require(_job_id(decode["job_id"]) and len(parents) == len(set(parents)) == 40
        and all(_job_id(job) for job in parents) and decode["job_id"] not in parents,
        "40_distinct_actual_model_jobs_required_no_array_parent_shorthand")
    selected = {"decode": decode}
    if stage.phase != "decode":
        draw = value["draw"]
        _require(isinstance(draw, dict) and set(draw) == {"array_job_id", "afterok_job_id"}
            and _job_id(draw["array_job_id"]) and draw["afterok_job_id"] == decode["job_id"]
            and draw["array_job_id"] not in parents + [decode["job_id"]], "draw_afterok_decode_binding_required")
        selected["draw"] = draw
    if stage.phase == "finalize":
        final = value["finalize"]
        _require(isinstance(final, dict) and set(final) == {"job_id", "afterok_job_id"}
            and _job_id(final["job_id"]) and final["afterok_job_id"] == draw["array_job_id"]
            and final["job_id"] not in parents + [decode["job_id"], draw["array_job_id"]],
            "finalize_afterok_whole_array_binding_required")
        selected["finalize"] = final
    if stage.phase == "draw":
        fields = dict(token.split("=", 1) for token in _text(stage.run_dir / "allocation.txt").split() if "=" in token)
        _require(os.environ.get("SLURM_ARRAY_JOB_ID") == draw["array_job_id"]
            and os.environ.get("SLURM_ARRAY_TASK_ID") == str(stage.batch_index)
            and fields.get("ArrayJobId") == draw["array_job_id"]
            and fields.get("ArrayTaskId") == str(stage.batch_index), "actual_fixed_array_child_required")
        _require(all(os.environ.get(key) == expected for key, expected in {
            "SLURM_ARRAY_TASK_COUNT": "63", "SLURM_ARRAY_TASK_MIN": "0",
            "SLURM_ARRAY_TASK_MAX": "62", "SLURM_ARRAY_TASK_STEP": "1"}.items()),
            "exact_63_element_unit_step_array_required")
    else:
        _require(all(not os.environ.get(key, "") for key in ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"))
            and allocation["job_id"] == selected[stage.phase]["job_id"], "actual_phase_job_binding_required")
    return selected


def _checks_completed(stage):
    _require(_status(stage.root / "decode/batch_status.txt", CHECK_RECORDS)
        == {"amended_readout_checks_exit": ["0"], "amended_readout_check_methods": ["8"]},
        "exact_eight_affected_methods_must_pass_once")


def _completed(stage, binding):
    if stage.phase == "decode":
        return
    decode = json.loads(_text(stage.root / "decode/report.json"))
    _require(decode.get("phase") == "decode" and decode.get("status") == "complete_decode"
        and decode.get("model_policy") == MODEL_POLICY and decode.get("artifacts_published") is True
        and decode.get("decoded_slots") == 855 and decode.get("unavailable_slots") == 0
        and decode.get("allocation_identity", {}).get("job_id") == binding["decode"]["job_id"],
        "successful_bound_amended_decode_report_required")
    _require(_status(stage.root / "decode/batch_status.txt", ("phase_exit", "status"))
        == {"phase_exit": ["0"], "status": ["closed"]}, "successful_closed_decode_parent_required")
    if stage.phase == "draw":
        _require(not os.path.lexists(stage.stage_h5.with_suffix(".finalization_started.json")),
            "draw_forbidden_after_finalization_claim")
        return
    jobs = set()
    for index in range(63):
        directory = stage.root / "draws" / f"batch_{index:03d}"
        result = json.loads(_text(directory / "report.json"))
        job = result.get("allocation_identity", {}).get("job_id")
        _require(result.get("phase") == "draw" and type(result.get("batch_index")) is int
            and result["batch_index"] == index and result.get("status") == "complete_draw"
            and result.get("model_policy") == MODEL_POLICY and result.get("artifacts_published") is True
            and result.get("readers_closed") is True and result.get("array_job_id") == binding["draw"]["array_job_id"]
            and _job_id(job) and job not in jobs,
            "all_63_actual_successful_closed_amended_draw_children_required")
        jobs.add(job)
        _require(_status(directory / "batch_status.txt", ("phase_exit", "status"))
            == {"phase_exit": ["0"], "status": ["closed"]}, "all_63_closed_successful_draw_parents_required")


def _require_file_policy(stage):
    _require(resource.getrlimit(resource.RLIMIT_FSIZE)[0] == stage.resources["collection_bytes"]
        and signal.getsignal(signal.SIGXFSZ) == signal.SIG_DFL, "exact_fatal_collection_file_policy_required")


def _configure_file_policy(stage):
    _, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    cap = stage.resources["collection_bytes"]
    _require(hard == resource.RLIM_INFINITY or hard >= cap, "parent_hard_limit_cannot_admit_collection")
    resource.setrlimit(resource.RLIMIT_FSIZE, (cap, hard))
    signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
    _require_file_policy(stage)


@dataclass(frozen=True)
class AmendedContext:
    profile: dict
    allocation: dict
    scratch: Path
    stage: AmendedStage
    binding: dict


class AmendedBudget(core.BrainPairBudget):
    def __init__(self, context):
        super().__init__(context)
        self._next_policy_check, self._checked_operations = 0., set()

    def check_storage(self):
        stage = self.context.stage
        _bytes_unique(stage.root, self.resources["aggregate_bytes"])
        _bytes_unique(stage.run_dir, self.resources["durable_bytes"])
        _bytes_unique(self.context.scratch, self.resources["scratch_bytes"])

    def check_policy(self, operation):
        now = time.monotonic()
        if operation not in self._checked_operations or now >= self._next_policy_check:
            _require(_profile(self.context.stage) == self.context.profile, "amended_readout_grant_changed")
            _require(_bindings(self.context.stage, self.allocation) == self.context.binding, "active_dependency_binding_changed")
            _checks_completed(self.context.stage)
            _completed(self.context.stage, self.context.binding)
            self.check_storage()
            self._checked_operations.add(operation)
            self._next_policy_check = now + 1.


def admit(phase, batch_index=None):
    initial = _stage(phase, batch_index)
    profile = _profile(initial)  # Missing/inactive new grant refuses before allocation or imports.
    stage = _stage(phase, batch_index, profile["resources"][phase])
    allocation, scratch = core._allocation(stage), core._scratch(stage)
    context = AmendedContext(profile, allocation, scratch, stage, _bindings(stage, allocation))
    budget = AmendedBudget(context)
    budget.check_storage()
    return context, budget


def _options(stage):
    return {**original._options(stage), "model_policy": MODEL_POLICY}


def resolver(context, *, operation, source, target, budget, execution_options):
    _require(isinstance(context, AmendedContext) and isinstance(budget, AmendedBudget)
        and budget.context is context and budget.allocation == context.allocation
        and context.stage == _stage(context.stage.phase, context.stage.batch_index,
                                   context.profile["resources"][context.stage.phase]), "same_fixed_amended_context_required")
    stage = context.stage
    if operation == OPERATIONS[stage.phase]:
        _require(source is None and target is None and execution_options is not None
            and _same(dict(execution_options), _options(stage)), "exact_amended_phase_options_required")
    else:
        _require(stage.phase == "decode" and execution_options is None, "operation_not_authorized_in_phase")
        if operation in ("bind_full_finding_a_inputs", "read_full_finding_a_truth", "prepare_amended_full_finding_a_models"):
            _require(source is None and target is None, "input_operation_has_no_pair")
        elif operation == "read_full_finding_a_tau_test":
            _require(source in PARTICIPANTS and target is None, "fixed_tau_test_owner_required")
        elif operation in ("read_full_finding_a_content_model", "read_amended_full_finding_a_brain_model"):
            _require((source, target) in PAIRS, "all_twenty_original_ordered_pairs_required")
        else:
            raise AdmissionError("operation_not_authorized")
    budget.check()
    _require_file_policy(stage)
    budget.check_policy(operation)


def _load_operation(phase):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import numpy as np
    import h5py
    import amended_distributed_readout as implementation
    return getattr(implementation, {"decode": "run_decoded_point", "draw": "run_draw_batch",
        "finalize": "run_finalizer"}[phase]), implementation.AmendedPhaseResult, {
        "python": sys.version.split()[0], "numpy": str(np.__version__),
        "torch": str(torch.__version__), "h5py": str(h5py.__version__)}


def _tau_train_ids(context, budget):
    """Same original metadata authentication, under the distinct new resolver."""
    resolver(context, operation=OPERATIONS["decode"], source=None, target=None,
        budget=budget, execution_options=_options(context.stage))
    sys.path.insert(0, str(BASE / "outputs/pipeline_conformance/tools"))
    import feature_protocol
    _, roles, alignment, _ = feature_protocol.image_roles()
    lock = json.loads(_text(BASE / "outputs/protocol_lock.yaml", maximum=10_000_000))
    _require(lock.get("state") == "valid_prospective_lock" and lock.get("job_id") == "46424955"
        and lock.get("regular_image_binding") == alignment, "original_locked_image_role_binding_required")
    ids = tuple(row["image_id"] for _, row in roles["tau_train"])
    _require(len(ids) == len(set(ids)) == 897 and all(isinstance(value, str) and value for value in ids),
        "897_authenticated_tau_train_IDs_required")
    budget.check()
    return tuple(sorted(ids, key=lambda value: value.encode("utf-8")))


def run_once(phase, batch_index=None):
    context, budget = admit(phase, batch_index)
    stage = context.stage
    paths = (*stage.publication_paths, stage.run_dir / "claim.json", stage.run_dir / "report.json")
    if phase == "finalize":
        paths += (stage.stage_h5.with_suffix(".finalization_started.json"),)
    for path in paths:
        _require(not os.path.lexists(path) and not os.path.lexists(str(path) + ".partial"),
            "existing_amended_readout_phase_no_retry")
    _checks_completed(stage)
    _completed(stage, context.binding)
    _configure_file_policy(stage)
    options = _options(stage)
    resolver(context, operation=OPERATIONS[phase], source=None, target=None,
        budget=budget, execution_options=options)
    _publish(stage.run_dir / "claim.json", {"phase": phase, "batch_index": batch_index,
        "operation": OPERATIONS[phase], "model_policy": MODEL_POLICY, "allocation": context.allocation,
        "binding": context.binding, "approval_source": context.profile["approval_source"], "automatic_retry": False}, budget)
    started_wall, started_cpu = time.monotonic(), time.process_time()
    step, interrupted = "load_operation", False
    report = {"kind": "ep21_amended_full_finding_a_phase_report", "phase": phase, "batch_index": batch_index,
        "status": "incomplete", "artifacts_published": False, "published_paths": None,
        "readers_closed": False, "decoded_slots": None, "unavailable_slots": None,
        "array_job_id": context.binding.get("draw", {}).get("array_job_id") if phase == "draw" else None,
        "allocation_identity": {key: context.allocation[key] for key in ("job_id", "hostname")},
        "quantitative_finding": "not_inspected_in_operational_report", "comparability": "not_evaluable",
        "model_policy": MODEL_POLICY, "model_training_performed": False,
        "automatic_retry": False, "resumable": False, "runtime": None}
    try:
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
            operation, result_type, report["runtime"] = _load_operation(phase)
            kwargs = dict(require_admitted_context=resolver, budget=budget)
            if phase == "decode":
                step = "authenticate_tau_ID_metadata"
                kwargs.update(stage_h5=stage.stage_h5, maximum_bytes=stage.resources["collection_bytes"],
                    authenticated_tau_train_image_ids=_tau_train_ids(context, budget))
            else:
                kwargs.update({key: value for key, value in options.items() if key != "model_policy"})
            if phase == "finalize":
                def readers_closed():
                    budget.check()
                    _require(_bindings(stage, context.allocation) == context.binding, "dependency_binding_changed")
                    _completed(stage, context.binding)
                kwargs["require_readers_closed"] = readers_closed
            step = "execute_phase"
            result = operation(context, **kwargs)
            _require(isinstance(result, result_type) and result.phase == phase and result.model_policy == MODEL_POLICY
                and isinstance(result.artifacts, tuple) and result.artifacts == stage.publication_paths
                and all(path.is_file() and path.resolve() == path.absolute() for path in result.artifacts),
                "complete_typed_amended_phase_publication_required")
            report.update(artifacts_published=True, published_paths=list(map(str, result.artifacts)),
                readers_closed=phase == "draw", decoded_slots=855, unavailable_slots=0)
            step = "closeout"
            budget.check(); budget.check_storage()
        report["status"] = "complete_" + phase
    except BaseException as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        report.update(error_type=type(error).__name__, failed_stage=step,
            resource_limit_failure=isinstance(error, (TimeoutError, MemoryError)))
        if step == "execute_phase":
            report["artifacts_published"] = None
    report.update(elapsed_wall_seconds=time.monotonic() - started_wall,
        process_cpu_seconds=time.process_time() - started_cpu,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    _publish(stage.run_dir / "report.json", report, budget)
    return 130 if interrupted else 0 if report["status"] == "complete_" + phase else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--batch-index", type=int)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check-admission", action="store_true")
    action.add_argument("--remaining-seconds", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.check_admission or args.remaining_seconds:
            context, budget = admit(args.phase, args.batch_index)
            if args.remaining_seconds:
                print(budget.remaining_seconds())
            else:
                print(json.dumps({"status": "amended_readout_allocation_admitted", "phase": args.phase,
                    "batch_index": args.batch_index, "model_policy": MODEL_POLICY,
                    "job_id": context.allocation["job_id"], "payload_read": False, "attempt_claimed": False}))
            return 0
        return run_once(args.phase, args.batch_index)
    except BaseException as error:
        print(json.dumps({"status": "amended_readout_entrypoint_refused", "error_type": type(error).__name__}))
        return 130 if isinstance(error, KeyboardInterrupt) else 2


if __name__ == "__main__":
    raise SystemExit(main())
