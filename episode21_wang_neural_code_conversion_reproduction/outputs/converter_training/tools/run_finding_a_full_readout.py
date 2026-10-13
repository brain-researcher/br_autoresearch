"""Dormant, stdlib-first admission for the three original full-readout phases.

Only a separately activated fixed grant and coordinator-supplied Slurm binding
admit execution. The binding records submitted afterok dependencies; it is a
procedural prerequisite, not an OS isolation mechanism or scheduler query.
The isolated empirical child establishes RLIMIT_FSIZE and SIGXFSZ=SIG_DFL.
Fatal file-cap failures belong to the parent; partials are never resumed.

Context, allocation, deadline, RSS and file policy are checked every callback.
Grant, dependency/check records and bounded inode-aware storage are rechecked
at each operation's first callback and at most once per second thereafter.
No existing first-readout, scientific implementation or grant is modified.
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
import stat
import sys
import time

import run_brain_pair as core


BASE = Path(__file__).resolve().parents[3]
GIB, MIB = 1024**3, 1024**2
PARTICIPANTS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
PAIRS = tuple((s, t) for s in PARTICIPANTS for t in PARTICIPANTS if s != t)
PHASES = ("decode", "draw", "finalize")
OPERATIONS = {"decode": "run_full_finding_a_decoded_point",
              "draw": "run_full_finding_a_draw_batch", "finalize": "finalize_full_finding_a_readout"}
COMMON_RESOURCES = {"gpus": 0, "collection_bytes": 768 * GIB, "aggregate_bytes": 776 * GIB,
                    "durable_bytes": 64 * MIB, "scratch_bytes": 4 * GIB, "batch_bytes": 32 * MIB}
RESOURCES = {
    "decode": {**COMMON_RESOURCES, "cpus": 3, "memory_gib": 16, "minutes": 4320,
               "deadline_seconds": 255600, "rss_limit_bytes": 15 * GIB},
    "draw": {**COMMON_RESOURCES, "cpus": 3, "memory_gib": 16, "minutes": 1440,
             "deadline_seconds": 84600, "rss_limit_bytes": 15 * GIB},
    "finalize": {**COMMON_RESOURCES, "cpus": 1, "memory_gib": 7, "minutes": 120,
                 "deadline_seconds": 6900, "rss_limit_bytes": 13 * GIB // 2}}
FIXED = {"schema_version": "ep21.finding_a_full_readout.v1", "partition": "russpold",
    "phases": list(PHASES), "operations": OPERATIONS, "jobs": 65, "draw_tasks": 63,
    "maximum_concurrent_draw_tasks": 4,
    "numeric_threads": 1, "test_images": 224, "layers": 19, "total_slots": 855,
    "decoded_slots": 703, "unavailable_slots": 152, "image_bootstrap_draws": 1000,
    "dyadic_bootstrap_draws": 1000, "draw_batch_size": 16, "model_policy": "original_locked",
    "production_training": False, "model_publication": False,
    "decoded_feature_publication": True, "scientific_scoring": True,
    "automatic_retry": False, "resumable": False}
INPUT_SCOPE = {
    "decode": {"tau_test_participants": list(PARTICIPANTS), "common_true_features": "tau_test",
        "decoder_payload_participants": list(PARTICIPANTS), "normalization_participants": list(PARTICIPANTS),
        "original_content_models": 20, "original_non_sub01_brain_models": 12,
        "tau_train_image_ID_metadata": True, "training_response_matrices": False,
        "tau_train_response_values": False, "regular_images": False, "OOD": False},
    "draw": {"published_decoded_point": True, "common_true_features": "tau_test",
        "models": False, "neural_values": False, "decoder_coefficients": False, "OOD": False},
    "finalize": {"published_decoded_point": True, "all_63_draw_batches": True,
        "models": False, "neural_values": False, "decoder_coefficients": False,
        "true_feature_values": False, "OOD": False}}
CHECK_RECORDS = ("full_input_checks_exit", "full_model_checks_exit", "full_composition_checks_exit",
    "draw_batch_checks_exit", "grouped_collection_checks_exit", "draw_artifact_checks_exit",
    "staged_collection_checks_exit", "distributed_composition_checks_exit",
    "full_readout_admission_checks_exit", "astra_exit")
AdmissionError, _require, _same = core.AdmissionError, core._require, core._same


@dataclass(frozen=True)
class FullReadoutStage:
    phase: str
    batch_index: int | None
    profile_path: Path
    root: Path
    run_dir: Path
    scratch_name: str
    allocation_start_deadline: bool = True

    @property
    def resources(self):
        return dict(RESOURCES[self.phase])

    @property
    def stage_h5(self):
        return self.root / "decoded_point.h5"

    @property
    def final_h5(self):
        return self.root / "finding_a_collection.h5"

    @property
    def batch_path(self):
        _require(self.phase == "draw", "draw_phase_required")
        return self.root / "draw_batches" / f"draw_batch_{self.batch_index:03d}.json"

    @property
    def publication_paths(self):
        if self.phase == "decode":
            return self.stage_h5, self.stage_h5.with_suffix(".decoded_point.json")
        if self.phase == "draw":
            return (self.batch_path,)
        return (self.final_h5, self.final_h5.with_suffix(".decoded_feature_manifest.json"),
                self.final_h5.with_suffix(".finding_a_report.json"))


def _stage(phase, batch_index=None):
    _require(phase in PHASES, "fixed_full_readout_phase_required")
    _require((phase == "draw" and type(batch_index) is int and 0 <= batch_index < 63)
             or (phase != "draw" and batch_index is None), "exact_draw_index_or_no_index_required")
    root = BASE / "outputs/runs/finding_a_feature_decoding/full001"
    name = f"draw_{batch_index:03d}" if phase == "draw" else phase
    run = root / "draws" / f"batch_{batch_index:03d}" if phase == "draw" else root / phase
    return FullReadoutStage(phase, batch_index, BASE / "outputs/finding_a_full_readout_authorization.json",
                            root, run, "ep21_finding_a_full001_" + name)


def _text(path, maximum=65536):
    path = Path(path)
    _require(path.resolve() == path.absolute() and path.is_file(), "fixed_regular_metadata_required")
    with path.open("rb") as stream:
        value = stream.read(maximum + 1)
    _require(len(value) <= maximum, "bounded_operational_metadata_required")
    return value.decode("utf-8")


def _profile(stage):
    value = json.loads(_text(stage.profile_path))
    _require(isinstance(value, dict) and value.get("authorized") is True, "full_readout_profile_not_activated")
    _require(isinstance(value.get("approval_source"), str) and value["approval_source"].strip(),
             "explicit_full_readout_approval_required")
    for key, expected in FIXED.items():
        _require(_same(value.get(key), expected), "fixed_full_readout_scope_changed")
    _require(_same(value.get("resources"), RESOURCES) and _same(value.get("input_scope"), INPUT_SCOPE),
             "full_readout_resources_or_roles_changed")
    return {**FIXED, "resources": RESOURCES, "input_scope": INPUT_SCOPE,
            "authorized": True, "approval_source": value["approval_source"]}


def _job_id(value):
    return isinstance(value, str) and value.isdigit() and int(value) > 0


def _bindings(stage, allocation):
    """Bind actual jobs to the coordinator's submitted afterok graph, without polling."""
    value = json.loads(_text(stage.root / "dependency_bindings.json"))
    _require(isinstance(value, dict) and set(value) == set(PHASES), "exact_phase_dependency_binding_required")
    decode = value["decode"]
    _require(isinstance(decode, dict) and set(decode) == {"job_id", "afterok_job_ids", "completed_model_job_ids"},
             "decode_dependency_binding_required")
    active, completed = decode["afterok_job_ids"], decode["completed_model_job_ids"]
    _require(isinstance(active, list) and isinstance(completed, list), "declared_active_and_completed_model_jobs_required")
    # The coordinator supplies accounting-verified completed parents separately:
    # old jobs may no longer be admissible Slurm afterok dependencies. The exact
    # original model loaders still bind every artifact; this is not a model ledger.
    parents = active + completed
    _require(_job_id(decode["job_id"]) and len(parents) == 32
             and all(_job_id(job) for job in parents) and len(set(parents)) == 32
             and decode["job_id"] not in parents, "32_distinct_original_model_parent_jobs_required")
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
                 and fields.get("ArrayTaskId") == str(stage.batch_index), "actual_fixed_array_element_required")
        _require(all(os.environ.get(key) == expected for key, expected in {
            "SLURM_ARRAY_TASK_COUNT": "63", "SLURM_ARRAY_TASK_MIN": "0",
            "SLURM_ARRAY_TASK_MAX": "62", "SLURM_ARRAY_TASK_STEP": "1"}.items()),
            "exact_63_element_unit_step_array_required")
    else:
        _require(all(not os.environ.get(key, "") for key in ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"))
                 and allocation["job_id"] == selected[stage.phase]["job_id"], "actual_phase_job_binding_required")
    return selected


def _status(path, keys):
    found = {key: [] for key in keys}
    for line in _text(path).splitlines():
        key, separator, value = line.partition("=")
        if separator and key in found:
            found[key].append(value)
    return found


def _checks_completed(stage):
    found = _status(stage.root / "decode/batch_status.txt", CHECK_RECORDS)
    _require(all(values == ["0"] for values in found.values()), "23_components_three_admission_and_ASTRA_must_pass_once")


def _completed(stage, binding):
    """Later phases require successful closed predecessors, never mere file presence."""
    if stage.phase == "decode":
        return
    decode = json.loads(_text(stage.root / "decode/report.json"))
    _require(decode.get("phase") == "decode" and decode.get("status") == "complete_decode"
             and decode.get("artifacts_published") is True
             and decode.get("allocation_identity", {}).get("job_id") == binding["decode"]["job_id"],
             "successful_bound_decode_report_required")
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
                 and result.get("artifacts_published") is True and result.get("readers_closed") is True
                 and result.get("array_job_id") == binding["draw"]["array_job_id"]
                 and _job_id(job) and job not in jobs, "all_63_distinct_successful_closed_draw_reports_required")
        jobs.add(job)
        _require(_status(directory / "batch_status.txt", ("phase_exit", "status"))
                 == {"phase_exit": ["0"], "status": ["closed"]}, "all_63_closed_successful_draw_parents_required")


def _bytes_unique(path, cap):
    """Stat only the admitted bounded tree; count each encountered file inode once."""
    path = Path(path)
    _require(path.resolve() == path.absolute() and path.is_dir(), "fixed_storage_root_required")
    pending, sizes, entries, total = [(path, 0)], {}, 0, 0
    while pending:
        directory, depth = pending.pop()
        _require(depth <= 4, "bounded_storage_depth_required")
        with os.scandir(directory) as stream:
            for entry in stream:
                entries += 1
                _require(entries <= 4096, "bounded_full_readout_entry_count_required")
                try:
                    info = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue  # A peer may exclusively publish and unlink its partial.
                if stat.S_ISDIR(info.st_mode):
                    pending.append((Path(entry.path), depth + 1))
                else:
                    _require(stat.S_ISREG(info.st_mode), "regular_storage_files_only")
                    key = info.st_dev, info.st_ino
                    previous = sizes.get(key, 0)
                    sizes[key] = max(previous, info.st_size)
                    total += sizes[key] - previous
                    _require(total <= cap, "full_readout_storage_cap")
    return total


def _require_file_policy():
    _require(resource.getrlimit(resource.RLIMIT_FSIZE)[0] == COMMON_RESOURCES["collection_bytes"]
             and signal.getsignal(signal.SIGXFSZ) == signal.SIG_DFL, "exact_fatal_collection_file_policy_required")


def _configure_file_policy():
    _, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    cap = COMMON_RESOURCES["collection_bytes"]
    _require(hard == resource.RLIM_INFINITY or hard >= cap, "parent_hard_limit_cannot_admit_collection")
    resource.setrlimit(resource.RLIMIT_FSIZE, (cap, hard))
    signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
    _require_file_policy()


@dataclass(frozen=True)
class FullReadoutContext:
    profile: dict
    allocation: dict
    scratch: Path
    stage: FullReadoutStage
    binding: dict


class FullReadoutBudget(core.BrainPairBudget):
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
            _require(_profile(self.context.stage) == self.context.profile, "full_readout_grant_changed")
            _require(_bindings(self.context.stage, self.allocation) == self.context.binding,
                     "active_dependency_binding_changed")
            _checks_completed(self.context.stage)
            _completed(self.context.stage, self.context.binding)
            self.check_storage()
            self._checked_operations.add(operation)
            self._next_policy_check = now + 1.


def admit(phase, batch_index=None):
    stage = _stage(phase, batch_index)
    profile = _profile(stage)  # Inactive proposal stops before allocation/imports/inputs.
    allocation, scratch = core._allocation(stage), core._scratch(stage)
    context = FullReadoutContext(profile, allocation, scratch, stage, _bindings(stage, allocation))
    budget = FullReadoutBudget(context)
    budget.check_storage()
    return context, budget


def _options(stage):
    cap = stage.resources["collection_bytes"]
    if stage.phase == "decode":
        return {"collection_path": stage.stage_h5, "maximum_bytes": cap}
    if stage.phase == "draw":
        return {"stage_h5": stage.stage_h5, "maximum_collection_bytes": cap,
                "batch_index": stage.batch_index, "batch_path": stage.batch_path,
                "maximum_batch_bytes": stage.resources["batch_bytes"]}
    return {"stage_h5": stage.stage_h5, "final_h5": stage.final_h5,
            "batch_directory": stage.root / "draw_batches", "maximum_collection_bytes": cap,
            "maximum_batch_bytes": stage.resources["batch_bytes"]}


def resolver(context, *, operation, source, target, budget, execution_options):
    _require(isinstance(context, FullReadoutContext) and isinstance(budget, FullReadoutBudget)
             and budget.context is context and budget.allocation == context.allocation
             and context.stage == _stage(context.stage.phase, context.stage.batch_index),
             "same_fixed_phase_context_required")
    stage = context.stage
    if operation == OPERATIONS[stage.phase]:
        _require(source is None and target is None and execution_options is not None
                 and _same(dict(execution_options), _options(stage)), "exact_phase_options_required")
    else:
        _require(stage.phase == "decode" and execution_options is None, "operation_not_authorized_in_phase")
        if operation in ("bind_full_finding_a_inputs", "read_full_finding_a_truth", "prepare_full_finding_a_models"):
            _require(source is None and target is None, "full_input_operation_has_no_pair")
        elif operation == "read_full_finding_a_tau_test":
            _require(source in PARTICIPANTS and target is None, "fixed_tau_test_owner_required")
        elif operation in ("read_full_finding_a_content_model", "read_full_finding_a_brain_model"):
            _require((source, target) in PAIRS and (operation != "read_full_finding_a_brain_model"
                     or "sub-01" not in (source, target)), "original_available_model_pair_required")
        else:
            raise AdmissionError("operation_not_authorized")
    budget.check()
    _require_file_policy()
    budget.check_policy(operation)


def _load_operation(phase):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import numpy as np
    import h5py
    import finding_a_distributed_readout as implementation
    return getattr(implementation, {"decode": "run_decoded_point", "draw": "run_draw_batch",
                                  "finalize": "run_finalizer"}[phase]), {
        "python": sys.version.split()[0], "numpy": str(np.__version__),
        "torch": str(torch.__version__), "h5py": str(h5py.__version__)}


def _tau_train_ids(context, budget):
    """Existing CSV/export/provider metadata authentication; no response/image arrays."""
    resolver(context, operation=OPERATIONS["decode"], source=None, target=None,
             budget=budget, execution_options=_options(context.stage))
    sys.path.insert(0, str(BASE / "outputs/pipeline_conformance/tools"))
    import feature_protocol
    _, roles, alignment, _ = feature_protocol.image_roles()
    # This existing .yaml is a JSON document, as used by the empirical binder.
    # Do not introduce a PyYAML dependency into the qualified numerical runtime.
    lock = json.loads(_text(BASE / "outputs/protocol_lock.yaml", maximum=10_000_000))
    _require(lock.get("state") == "valid_prospective_lock"
             and lock.get("job_id") == "46424955" and lock.get("regular_image_binding") == alignment,
             "original_locked_image_role_binding_required")
    ids = tuple(row["image_id"] for _, row in roles["tau_train"])
    _require(len(ids) == len(set(ids)) == 897 and all(isinstance(x, str) and x for x in ids),
             "897_authenticated_tau_train_IDs_required")
    budget.check()
    return tuple(sorted(ids, key=lambda value: value.encode("utf-8")))


def _publish(path, value, budget):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    _require(len(data) <= 65536, "bounded_operational_record_required")
    stage = budget.context.stage
    for root, cap in ((stage.root, stage.resources["aggregate_bytes"]),
                      (stage.run_dir, stage.resources["durable_bytes"])):
        _require(_bytes_unique(root, cap) + len(data) <= cap, "operational_record_exceeds_storage_cap")
    with path.open("xb") as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())


def run_once(phase, batch_index=None):
    context, budget = admit(phase, batch_index)
    stage = context.stage
    paths = (*stage.publication_paths, stage.run_dir / "claim.json", stage.run_dir / "report.json")
    if phase == "finalize":
        paths += (stage.stage_h5.with_suffix(".finalization_started.json"),)
    for path in paths:
        _require(not os.path.lexists(path) and not os.path.lexists(str(path) + ".partial"),
                 "existing_full_readout_phase_no_retry")
    _checks_completed(stage)
    _completed(stage, context.binding)
    _configure_file_policy()
    options = _options(stage)
    resolver(context, operation=OPERATIONS[phase], source=None, target=None,
             budget=budget, execution_options=options)
    _publish(stage.run_dir / "claim.json", {"phase": phase, "batch_index": batch_index,
        "operation": OPERATIONS[phase], "allocation": context.allocation, "binding": context.binding,
        "approval_source": context.profile["approval_source"], "automatic_retry": False}, budget)
    started_wall, started_cpu = time.monotonic(), time.process_time()
    step, interrupted = "load_operation", False
    report = {"kind": "ep21_full_finding_a_phase_report", "phase": phase, "batch_index": batch_index,
        "status": "incomplete", "artifacts_published": False, "published_paths": None,
        "readers_closed": False, "decoded_slots": None, "unavailable_slots": None,
        "array_job_id": context.binding.get("draw", {}).get("array_job_id") if phase == "draw" else None,
        "allocation_identity": {key: context.allocation[key] for key in ("job_id", "hostname")},
        "quantitative_finding": "not_evaluable", "comparability": "not_evaluable",
        "model_policy": "original_locked", "model_training_performed": False,
        "automatic_retry": False, "resumable": False, "runtime": None}
    try:
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
            operation, report["runtime"] = _load_operation(phase)
            kwargs = dict(require_admitted_context=resolver, budget=budget)
            if phase == "decode":
                step = "authenticate_tau_ID_metadata"
                kwargs.update(stage_h5=stage.stage_h5, maximum_bytes=stage.resources["collection_bytes"],
                              authenticated_tau_train_image_ids=_tau_train_ids(context, budget))
            else:
                kwargs.update(options)
            if phase == "finalize":
                def readers_closed():
                    budget.check()
                    _require(_bindings(stage, context.allocation) == context.binding, "dependency_binding_changed")
                    _completed(stage, context.binding)
                kwargs["require_readers_closed"] = readers_closed
            step = "execute_phase"
            result = operation(context, **kwargs)
            actual = (result,) if phase == "draw" else result
            _require(isinstance(actual, tuple) and actual == stage.publication_paths
                     and all(path.is_file() and path.resolve() == path.absolute() for path in actual),
                     "complete_fixed_phase_publication_required")
            report.update(artifacts_published=True, published_paths=list(map(str, actual)),
                          readers_closed=phase == "draw", decoded_slots=703, unavailable_slots=152)
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
                print(json.dumps({"status": "full_readout_allocation_admitted", "phase": args.phase,
                    "batch_index": args.batch_index, "job_id": context.allocation["job_id"],
                    "payload_read": False, "attempt_claimed": False}))
            return 0
        return run_once(args.phase, args.batch_index)
    except BaseException as error:
        print(json.dumps({"status": "full_readout_entrypoint_refused", "error_type": type(error).__name__}))
        return 130 if isinstance(error, KeyboardInterrupt) else 2


if __name__ == "__main__":
    raise SystemExit(main())
