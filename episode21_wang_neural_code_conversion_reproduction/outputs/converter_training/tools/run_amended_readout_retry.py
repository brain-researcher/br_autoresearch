"""Inactive, single-use fresh amended_full002 readout after diagnosed failure.

No authorization file or resource selection accompanies this implementation.
The separate selection references the operative fixed-science profile, records
the coordinator's terminal diagnosis, and binds fresh phase jobs to the same
forty producers. It never opens the old HDF or resumes a partial collection.

For this caller resources.deadline_seconds is the HARD admitted budget. Each
phase stops scientific work its explicit cleanup_reserve_seconds earlier. The
outer wrapper must bind --hard-stop-epoch, never the work budget clock.
Every cleanup reserve must exceed its maximum ten-second TERM-to-KILL grace;
TERM therefore follows scientific work stop and KILL is due at the hard epoch.
Terminal observations and preservation of the frozen old attempt are the
coordinator's responsibility; this caller performs no scheduler queries.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import resource
import sys
import time

import run_amended_full_readout as amended
from amended_decode_lifecycle import DecodeLifecycle


BASE = Path(__file__).resolve().parents[3]
ATTEMPT = "amended_full002"
ROOT = BASE / "outputs/runs/finding_a_feature_decoding" / ATTEMPT
PROFILE = BASE / "outputs/amended_readout_retry_authorization.json"
SCIENCE_PROFILE = "outputs/amended_full_readout_authorization.json"
PRIOR_ROOT = BASE / "outputs/runs/finding_a_feature_decoding/amended_full001"
PHASES, MODEL_POLICY = amended.PHASES, amended.MODEL_POLICY
OPERATIONS = {**amended.OPERATIONS, "decode": "run_checkpointed_amended_finding_a_decoded_point"}
_require, _text, _same = amended._require, amended._text, amended._same
PROFILE_KEYS = frozenset(("schema_version", "authorized", "approval_source", "attempt",
    "science_profile", "resources", "cleanup_reserve_seconds", "maximum_concurrent_draw_tasks",
    "prior_decode_terminal", "retained_attempt_bytes", "combined_storage_cap_bytes"))


def _stage(phase, batch_index=None, resources=None):
    _require(phase in PHASES, "fixed_retry_phase_required")
    _require((phase == "draw" and type(batch_index) is int and 0 <= batch_index < 63)
        or (phase != "draw" and batch_index is None), "exact_draw_index_or_no_index_required")
    name = f"draw_{batch_index:03d}" if phase == "draw" else phase
    run = ROOT / "draws" / f"batch_{batch_index:03d}" if phase == "draw" else ROOT / phase
    return amended.AmendedStage(phase, batch_index, PROFILE, ROOT, run,
        "ep21_amended_full002_" + name, True,
        () if resources is None else tuple(sorted(resources.items())))


def _profile(stage):
    # This read and activation check precede allocation, old-attempt inspection,
    # scientific imports, claims and payload access.
    value = json.loads(_text(stage.profile_path))
    _require(isinstance(value, dict) and value.get("authorized") is True,
        "fresh_retry_selection_not_activated")
    _require(set(value) == PROFILE_KEYS and value["schema_version"] == "ep21.amended_readout_retry.v1"
        and value["attempt"] == ATTEMPT and value["science_profile"] == SCIENCE_PROFILE,
        "exact_separate_retry_selection_required")
    _require(isinstance(value["approval_source"], str) and bool(value["approval_source"].strip()),
        "coordinator_retry_selection_required")
    # Reuse the operative validator rather than copying scientific fields.
    science = amended._profile(amended._stage("decode"))
    resources = amended._resource_plan(value["resources"])
    reserve = value["cleanup_reserve_seconds"]
    _require(isinstance(reserve, dict) and set(reserve) == set(PHASES)
        and all(type(reserve[p]) is int and 10 < reserve[p] < resources[p]["deadline_seconds"]
                for p in PHASES), "cleanup_reserve_must_exceed_ten_second_wrapper_grace")
    parallel = value["maximum_concurrent_draw_tasks"]
    _require(type(parallel) is int and 1 <= parallel <= 63, "finite_draw_concurrency_required")
    terminal = value["prior_decode_terminal"]
    _require(isinstance(terminal, dict) and set(terminal) ==
        {"job_id", "state", "exit_code", "observed_at", "diagnosis"}
        and amended._job_id(terminal["job_id"])
        and terminal["state"] in {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED", "BOOT_FAIL", "DEADLINE"}
        and isinstance(terminal["exit_code"], str) and len(terminal["exit_code"].split(":")) == 2
        and all(piece.isdigit() for piece in terminal["exit_code"].split(":"))
        and isinstance(terminal["observed_at"], str)
        and isinstance(terminal["diagnosis"], str) and bool(terminal["diagnosis"].strip()),
        "diagnosed_terminal_failure_required_not_running_or_success")
    observed = datetime.fromisoformat(terminal["observed_at"])
    _require(observed.tzinfo is not None and observed.timestamp() <= time.time(),
        "past_timezone_aware_terminal_observation_required")
    retained, combined = value["retained_attempt_bytes"], value["combined_storage_cap_bytes"]
    _require(type(retained) is int and retained >= 0 and type(combined) is int
        and combined >= retained + resources["decode"]["aggregate_bytes"],
        "retained_plus_fresh_attempt_storage_required")
    return {**value, "resources": resources, "science": science}


def _bindings(stage, allocation, profile):
    selected = amended._bindings(stage, allocation)
    fresh = json.loads(_text(stage.root / "dependency_bindings.json"))
    old = json.loads(_text(PRIOR_ROOT / "dependency_bindings.json"))
    _require(isinstance(old, dict) and set(old) == set(PHASES), "preserved_original_phase_binding_required")
    prior = old["decode"]
    prior_producers = prior["afterok_job_ids"] + prior["completed_model_job_ids"]
    producers = fresh["decode"]["afterok_job_ids"] + fresh["decode"]["completed_model_job_ids"]
    _require(len(prior_producers) == len(set(prior_producers)) == 40
        and all(amended._job_id(job) for job in prior_producers)
        and set(producers) == set(prior_producers), "same_forty_original_producer_jobs_required")
    _require(profile["prior_decode_terminal"]["job_id"] == prior["job_id"],
        "terminal_failure_must_identify_original_decode")
    # Every phase is freshly bound even when admitting decode. Original helpers
    # still authenticate the actual current phase/array child against allocation.
    draw, final = fresh["draw"], fresh["finalize"]
    _require(isinstance(draw, dict) and set(draw) == {"array_job_id", "afterok_job_id"}
        and isinstance(final, dict) and set(final) == {"job_id", "afterok_job_id"}
        and draw["afterok_job_id"] == fresh["decode"]["job_id"]
        and final["afterok_job_id"] == draw["array_job_id"], "fresh_complete_afterok_chain_required")
    jobs = (fresh["decode"]["job_id"], draw["array_job_id"], final["job_id"])
    previous_jobs = (prior["job_id"], old["draw"]["array_job_id"], old["finalize"]["job_id"])
    _require(all(amended._job_id(job) for job in jobs + previous_jobs)
        and len(set(jobs)) == 3 and not set(jobs).intersection((*previous_jobs, *producers)),
        "all_retry_phase_jobs_must_be_fresh")
    return selected


def _checks_completed():
    # Prior evidence is validated in place, never copied as a current test PASS.
    amended._checks_completed(amended._stage("decode"))


@dataclass(frozen=True)
class RetryContext:
    profile: dict
    allocation: dict
    scratch: Path
    stage: amended.AmendedStage
    binding: dict
    retained_actual_bytes: int


class RetryBudget(amended.core.BrainPairBudget):
    def __init__(self, context):
        self.lifecycle = None
        super().__init__(context)
        # This separate original budget is used only for operational closeout.
        self.cleanup_budget = amended.core.BrainPairBudget(context)
        reserve = context.profile["cleanup_reserve_seconds"][context.stage.phase]
        self.absolute_deadline -= reserve
        self.deadline -= reserve
        self._next_policy_check, self._checked_operations = 0., set()
        if context.stage.phase == "decode":
            self.lifecycle = DecodeLifecycle(job_id=context.allocation["job_id"],
                collection_path=context.stage.stage_h5, record_path=context.stage.run_dir / "closure.json",
                work_stop_epoch=self.absolute_deadline,
                hard_stop_epoch=self.cleanup_budget.absolute_deadline)
        self.check()

    def check(self):
        if self.lifecycle is not None:
            self.lifecycle.check_work()
        super().check()

    def hard_remaining_seconds(self):
        return self.cleanup_budget.remaining_seconds()

    def check_storage(self):
        stage, profile = self.context.stage, self.context.profile
        current = amended._bytes_unique(stage.root, self.resources["aggregate_bytes"])
        _require(profile["retained_attempt_bytes"] + current <= profile["combined_storage_cap_bytes"],
            "preserved_plus_current_attempt_storage_cap")
        amended._bytes_unique(stage.run_dir, self.resources["durable_bytes"])
        amended._bytes_unique(self.context.scratch, self.resources["scratch_bytes"])

    def check_policy(self, operation):
        now = time.monotonic()
        if operation not in self._checked_operations or now >= self._next_policy_check:
            _require(_profile(self.context.stage) == self.context.profile, "retry_selection_or_original_science_changed")
            _require(_bindings(self.context.stage, self.allocation, self.context.profile) == self.context.binding,
                "retry_dependency_binding_changed")
            _checks_completed()
            amended._completed(self.context.stage, self.context.binding)
            self.check_storage()
            self._checked_operations.add(operation)
            self._next_policy_check = now + 1.


def admit(phase, batch_index=None):
    profile = _profile(_stage(phase, batch_index))
    stage = _stage(phase, batch_index, profile["resources"][phase])
    allocation, scratch = amended.core._allocation(stage), amended.core._scratch(stage)
    binding = _bindings(stage, allocation, profile)
    # The original attempt is terminal and frozen by coordinator selection.
    # Measure its bounded inode total once; never scan it in operation callbacks.
    retained = amended._bytes_unique(PRIOR_ROOT, profile["retained_attempt_bytes"])
    context = RetryContext(profile, allocation, scratch, stage, binding, retained)
    budget = RetryBudget(context)
    budget.check_storage()
    return context, budget


def _options(stage, budget):
    options = amended._options(stage)
    if stage.phase == "decode":
        lifecycle = budget.lifecycle
        options.update(work_stop_epoch=lifecycle.work_stop_epoch, hard_stop_epoch=lifecycle.hard_stop_epoch,
            closure_record_path=lifecycle.record_path)
    return options


def resolver(context, *, operation, source, target, budget, execution_options):
    _require(isinstance(context, RetryContext) and isinstance(budget, RetryBudget)
        and budget.context is context and budget.allocation == context.allocation
        and context.stage == _stage(context.stage.phase, context.stage.batch_index,
            context.profile["resources"][context.stage.phase]), "same_fresh_retry_context_required")
    stage = context.stage
    if operation == OPERATIONS[stage.phase]:
        _require(source is None and target is None and execution_options is not None
            and _same(dict(execution_options), _options(stage, budget)), "exact_retry_phase_options_required")
    else:
        _require(stage.phase == "decode" and execution_options is None, "operation_not_authorized_in_retry_phase")
        if operation in ("bind_full_finding_a_inputs", "read_full_finding_a_truth", "prepare_amended_full_finding_a_models"):
            _require(source is None and target is None, "input_operation_has_no_pair")
        elif operation == "read_full_finding_a_tau_test":
            _require(source in amended.PARTICIPANTS and target is None, "fixed_tau_test_owner_required")
        elif operation in ("read_full_finding_a_content_model", "read_amended_full_finding_a_brain_model"):
            _require((source, target) in amended.PAIRS, "all_twenty_original_ordered_pairs_required")
        else:
            raise amended.AdmissionError("operation_not_authorized")
    budget.check()
    amended._require_file_policy(stage)
    budget.check_policy(operation)


def _tau_train_ids(context, budget):
    resolver(context, operation=OPERATIONS["decode"], source=None, target=None,
        budget=budget, execution_options=_options(context.stage, budget))
    # Same original metadata-only authentication, using this caller's resolver.
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


def _load_operation(phase):
    if phase == "decode":
        import amended_checkpointed_decode as implementation
        return implementation.run_decoded_point, None, {"python": sys.version.split()[0],
            "scientific_runtime": "loaded_lazily_inside_checkpointed_composition"}
    return amended._load_operation(phase)


def run_once(phase, batch_index=None):
    context, budget = admit(phase, batch_index)
    stage = context.stage
    paths = (*stage.publication_paths, stage.run_dir / "claim.json", stage.run_dir / "report.json")
    if phase == "finalize":
        paths += (stage.stage_h5.with_suffix(".finalization_started.json"),)
    for path in paths:
        _require(not os.path.lexists(path) and not os.path.lexists(str(path) + ".partial"),
            "existing_fresh_retry_phase_no_resume")
    _checks_completed()
    amended._completed(stage, context.binding)
    amended._configure_file_policy(stage)
    options = _options(stage, budget)
    resolver(context, operation=OPERATIONS[phase], source=None, target=None,
        budget=budget, execution_options=options)
    amended._publish(stage.run_dir / "claim.json", {"phase": phase, "batch_index": batch_index,
        "attempt": ATTEMPT, "operation": OPERATIONS[phase], "model_policy": MODEL_POLICY,
        "allocation": context.allocation, "binding": context.binding,
        "approval_source": context.profile["approval_source"], "automatic_retry": False,
        "prior_decode_terminal": context.profile["prior_decode_terminal"],
        "retained_actual_bytes": context.retained_actual_bytes,
        "checks_evidence_source": str(PRIOR_ROOT / "decode/batch_status.txt"),
        "checks_execution": "prior_evidence_reused_no_current_test_execution",
        "work_stop_epoch": budget.absolute_deadline,
        "hard_stop_epoch": budget.cleanup_budget.absolute_deadline}, budget.cleanup_budget)
    started_wall, started_cpu = time.monotonic(), time.process_time()
    step, interrupted = "load_operation", False
    report = {"kind": "ep21_amended_readout_retry_phase_report", "phase": phase, "batch_index": batch_index,
        "attempt": ATTEMPT, "status": "incomplete", "artifacts_published": False, "published_paths": None,
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
                    authenticated_tau_train_image_ids=_tau_train_ids(context, budget),
                    lifecycle=budget.lifecycle, handle_sigterm=True)
            else:
                kwargs.update({key: value for key, value in options.items() if key != "model_policy"})
            if phase == "finalize":
                def readers_closed():
                    budget.check()
                    _require(_bindings(stage, context.allocation, context.profile) == context.binding,
                        "dependency_binding_changed")
                    amended._completed(stage, context.binding)
                kwargs["require_readers_closed"] = readers_closed
            step = "execute_phase"
            result = operation(context, **kwargs)
            if phase == "decode":
                import amended_distributed_readout
                result_type = amended_distributed_readout.AmendedPhaseResult
                _require(budget.lifecycle.record_published and budget.lifecycle.close_state == "succeeded",
                    "observed_successful_decode_close_required")
            _require(isinstance(result, result_type) and result.phase == phase and result.model_policy == MODEL_POLICY
                and isinstance(result.artifacts, tuple) and result.artifacts == stage.publication_paths
                and all(path.is_file() and path.resolve() == path.absolute() for path in result.artifacts),
                "complete_typed_amended_phase_publication_required")
            report.update(artifacts_published=True, published_paths=list(map(str, result.artifacts)),
                readers_closed=phase == "draw", decoded_slots=855, unavailable_slots=0)
            step = "closeout"
            budget.cleanup_budget.check()
            budget.check_storage()
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
    budget.cleanup_budget.check()
    budget.check_storage()
    amended._publish(stage.run_dir / "report.json", report, budget.cleanup_budget)
    return 130 if interrupted else 0 if report["status"] == "complete_" + phase else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--batch-index", type=int)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check-admission", action="store_true")
    action.add_argument("--hard-remaining-seconds", action="store_true")
    action.add_argument("--hard-stop-epoch", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.check_admission or args.hard_remaining_seconds or args.hard_stop_epoch:
            context, budget = admit(args.phase, args.batch_index)
            if args.hard_stop_epoch:
                budget.cleanup_budget.check()
                print(int(budget.cleanup_budget.absolute_deadline))
            elif args.hard_remaining_seconds:
                print(budget.hard_remaining_seconds())
            else:
                print(json.dumps({"status": "fresh_retry_allocation_admitted", "phase": args.phase,
                    "batch_index": args.batch_index, "attempt": ATTEMPT, "model_policy": MODEL_POLICY,
                    "job_id": context.allocation["job_id"], "payload_read": False, "attempt_claimed": False}))
            return 0
        return run_once(args.phase, args.batch_index)
    except BaseException as error:
        print(json.dumps({"status": "amended_readout_retry_refused", "error_type": type(error).__name__}))
        return 130 if isinstance(error, KeyboardInterrupt) else 2


if __name__ == "__main__":
    raise SystemExit(main())
