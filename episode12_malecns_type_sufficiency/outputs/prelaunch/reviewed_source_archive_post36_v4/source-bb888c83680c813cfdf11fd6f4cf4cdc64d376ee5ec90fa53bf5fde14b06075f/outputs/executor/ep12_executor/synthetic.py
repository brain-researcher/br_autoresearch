"""Deterministic synthetic qualification program for the EP12 controller."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .controller import (
    AuditError,
    Episode12Controller,
    SimulatedInterruption,
    SyntheticSourceManifest,
    WholeTypeRoleManifest,
    balanced_coverage_configs,
)
from .journal import HashChainedJournal
from .policy import EpisodePolicy, canonical_json, digest_bytes, digest_object


EPISODE_ROOT = Path(
    "/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/"
    "episode12_malecns_type_sufficiency"
)
SCRATCH_ROOT = Path(
    "/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency"
)


def executor_code_hash() -> str:
    package_root = Path(__file__).resolve().parent
    material = b"".join(
        path.name.encode("utf-8") + b"\0" + path.read_bytes()
        for path in sorted(package_root.glob("*.py"))
    )
    return digest_bytes(material)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass
class SyntheticAuditVault:
    """Idempotent synthetic evaluator with a durable access receipt."""

    path: Path
    result: Mapping[str, Any]

    def _load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text(encoding="utf-8"))

    @property
    def access_count(self) -> int:
        record = self._load()
        return int(record["access_count"]) if record else 0

    @property
    def evaluator_invocation_count(self) -> int:
        record = self._load()
        return int(record["evaluator_invocation_count"]) if record else 0

    def reconcile(self, operation_id: str, lock_hash: str) -> Mapping[str, Any] | None:
        record = self._load()
        if record is None:
            return None
        if record["operation_id"] != operation_id or record["lock_hash"] != lock_hash:
            raise AuditError("synthetic evaluator receipt belongs to another operation")
        return record["result"]

    def evaluate(self, operation_id: str, lock_hash: str) -> Mapping[str, Any]:
        existing = self._load()
        if existing is not None:
            if existing["operation_id"] != operation_id or existing["lock_hash"] != lock_hash:
                raise AuditError("synthetic final vault already consumed")
            return existing["result"]
        record = {
            "operation_id": operation_id,
            "lock_hash": lock_hash,
            "access_count": 1,
            "evaluator_invocation_count": 1,
            "result_hash": digest_object(self.result),
            "result": dict(self.result),
        }
        atomic_json(self.path, record)
        return record["result"]


def role_manifest() -> WholeTypeRoleManifest:
    return WholeTypeRoleManifest(
        development_types=tuple(f"synthetic_dev_type_{index:03d}" for index in range(1, 17)),
        final_types=tuple(f"synthetic_final_type_{index:03d}" for index in range(1, 5)),
    )


def make_controller(run_dir: Path, policy_path: Path) -> Episode12Controller:
    run_dir = run_dir.resolve()
    policy_path = policy_path.resolve()
    if policy_path != (EPISODE_ROOT / "SEARCH_POLICY.yaml").resolve():
        raise ValueError("executor policy path must be canonical EP12 SEARCH_POLICY.yaml")
    allowed_roots = (SCRATCH_ROOT.resolve(), (EPISODE_ROOT / "outputs").resolve())
    if not any(root == run_dir or root in run_dir.parents for root in allowed_roots):
        raise ValueError("synthetic run directory must stay in EP12 scratch or outputs")
    run_dir.mkdir(parents=True, exist_ok=True)
    return Episode12Controller.load_policy(
        policy_path=policy_path,
        source_manifest=SyntheticSourceManifest(source_id="ep12-generated-qualification-v1"),
        split_manifest=role_manifest(),
        journal_path=run_dir / "trial_ledger.jsonl",
    )


def _evaluation(
    controller: Episode12Controller,
    *,
    selection_statistic: float,
    cpu_core_hours: float = 0.0,
    generator_kind: str = "adequate_template",
    falsifier_id: str | None = None,
    comparison_gate_overrides: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    comparison_gates = {
        gate: False for gate in controller.PRE_NULL_COMPARISON_GATES
    }
    simulated_gate_outcomes = {
        gate: "simulated_pass_for_state_machine_only"
        for gate in controller.PRE_NULL_COMPARISON_GATES
    }
    for gate, simulated_pass in (comparison_gate_overrides or {}).items():
        simulated_gate_outcomes[gate] = (
            "simulated_pass_for_state_machine_only"
            if simulated_pass
            else "simulated_fail_for_state_machine_only"
        )
    cpu_cores = 32.0 if cpu_core_hours else 0.0
    wall_hours = cpu_core_hours / cpu_cores if cpu_cores else 0.0
    controls: dict[str, Any] = {
        "endpoint_mass_preserved": True,
        "complete_triplet": True,
        "synthetic_fixture_only": True,
        "scientific_pre_null_gates_passed": False,
        "scientific_model_qualification": False,
    }
    if falsifier_id is not None:
        controls["falsifier_id"] = falsifier_id
        controls["falsifier_execution_status"] = "simulated_not_executed"
    return {
        "models_completed": list(controller.policy.model_triplet),
        "accessed_types": list(controller.role_manifest.development_types),
        "incoming_accessed": False,
        "provider_group_accessed": False,
        "provider_instance_accessed": False,
        "metrics": {
            "selection_statistic": selection_statistic,
            "bilateral_predictive_quality": selection_statistic,
            "synthetic_generator_kind": generator_kind,
            "scientific_model_qualification": False,
        },
        "comparison_gates": comparison_gates,
        "simulated_pre_null_gate_outcomes": simulated_gate_outcomes,
        "controls": controls,
        "resources": {
            "cpu_core_hours": cpu_core_hours,
            "gpu_hours": 0.0,
            "cpu_cores": cpu_cores,
            "memory_gib": 1.0,
            "wall_hours": wall_hours,
            "measurement": "synthetic_fixture_not_runtime_profile",
        },
    }


def run_observed_search(
    controller: Episode12Controller,
    *,
    total_trials: int = 52,
    interrupt_trial: int | None = None,
    cpu_core_hours_per_trial: float = 0.0,
) -> dict[str, Any]:
    policy = controller.policy
    coverage = balanced_coverage_configs(policy)
    if total_trials < 1 or total_trials > policy.maximum_valid_trials:
        raise ValueError("invalid synthetic trial count")
    selected_trial = min(36, total_trials)
    interrupted_once = False

    for index in range(1, total_trials + 1):
        trial_id = f"trial-{index:04d}"
        config = dict(coverage[(index - 1) % len(coverage)])
        if index <= len(coverage):
            stage = "coverage"
            proposal_mode = "prespecified_coverage"
            parents: list[str] = []
            evidence: list[str] = []
            cycle = None
        elif 21 <= index <= 34:
            stage = "falsification"
            proposal_mode = "mandated_falsifier"
            parents = [f"trial-{index - 1:04d}"]
            evidence = [f"trial:trial-{index - 1:04d}:scored"]
            cycle = None
        else:
            stage = "adaptive"
            proposal_mode = "outcome_adaptive_successor"
            parents = [f"trial-{index - 1:04d}"]
            evidence = [f"trial:trial-{index - 1:04d}:scored"]
            cycle = f"synthetic-cycle-{index:04d}"

        context_hash = controller.journal.prefix_hash()
        registered_falsifier = (
            policy.required_falsifiers[(index - 21) % len(policy.required_falsifiers)]
            if stage == "falsification"
            else "synthetic fixture-specific negative control"
        )
        controller.register_hypothesis(
            trial_id=trial_id,
            parent_trial_ids=parents,
            proposal_mode=proposal_mode,
            proposal_context_hash=context_hash,
            outcome_evidence_refs=evidence,
            successor_cycle_id=cycle,
            prediction="synthetic bilateral score follows the fixture",
            falsifier=registered_falsifier,
            changed_operator="covering_array_factor" if stage == "coverage" else stage,
            unchanged_operators=("T_U_M_triplet", "whole_type_roles"),
            expected_information_gain="exercise a declared controller branch",
            expected_cost={"synthetic_core_hours": cpu_core_hours_per_trial},
            config=config,
            stage=stage,
        )
        if index <= selected_trial:
            selection_statistic = index / 1000.0
        else:
            selection_statistic = selected_trial / 1000.0 - (index - selected_trial) / 1_000_000.0
        evaluation = _evaluation(
            controller,
            selection_statistic=selection_statistic,
            cpu_core_hours=cpu_core_hours_per_trial,
            generator_kind=(
                "side_artifact" if stage == "falsification" else "adequate_template"
            ),
            falsifier_id=registered_falsifier if stage == "falsification" else None,
        )
        if interrupt_trial == index and not interrupted_once:
            try:
                controller.run_development_trial(
                    trial_id, evaluation, interrupt_after_running=True
                )
            except SimulatedInterruption:
                interrupted_once = True
                controller = make_controller(controller.journal.path.parent, policy.path)
        controller.run_development_trial(trial_id, evaluation)
        if index <= selected_trial:
            disposition = "INCUMBENT"
            rationale = "synthetic score improved"
        else:
            disposition = "RETIRED"
            rationale = "synthetic patience tail did not improve"
        controller.update_archive(
            trial_id,
            disposition,
            rationale=rationale,
            decision=index <= 2 or index == selected_trial,
        )

    return {
        "controller": controller,
        "selected_trial_id": f"trial-{selected_trial:04d}",
        "interruption_injected": interrupted_once,
    }


def null_receipts(
    controller: Episode12Controller,
    *,
    observed_statistic: float,
    selection_program_hash: str,
) -> list[dict[str, Any]]:
    valid_trials = controller.replay().valid_trial_count
    receipts: list[dict[str, Any]] = []
    for index in range(controller.policy.null_replicates):
        selected = observed_statistic - 0.01 - index / 1_000_000.0
        trajectory = {
            "replicate_index": index,
            "seed": 120000 + index,
            "valid_trials": valid_trials,
            "selected_statistic": selected,
            "selection_program_hash": selection_program_hash,
        }
        receipts.append(
            {
                **trajectory,
                "initial_ledger_empty": True,
                "complete_controller_rerun": False,
                "simulated_complete_controller_rerun": True,
                "nested_null_searches": 0,
                "deterministic_replay_passed": False,
                "simulated_deterministic_replay": True,
                "policy_hash": controller.policy.policy_hash,
                "model_family_fits": valid_trials * 3,
                "trajectory_hash": digest_object(trajectory),
                "receipt_scope": "asserted_synthetic_not_executed",
            }
        )
    return receipts


def final_result(kind: str, controller: Episode12Controller) -> dict[str, Any]:
    support = {key: True for key in controller.RESIDUAL_SUPPORT_KEYS}
    common: dict[str, Any] = {
        "valid_final": True,
        "comparison_valid": True,
        "every_final_type_reported": True,
        "reported_final_types": list(controller.role_manifest.final_types),
        "per_type_results": [
            {
                "type": type_name,
                "directions": {
                    "left_to_right": {"status": "scored"},
                    "right_to_left": {"status": "scored"},
                },
            }
            for type_name in controller.role_manifest.final_types
        ],
        "target_side_refit": False,
        "source_side_fit_only": True,
        "reference_calibrated": True,
        "adequacy_sensitivity_qualified": True,
        "residual_support": support,
        "followup_started": False,
        "paper_level_claim_emitted": False,
    }
    if kind == "positive":
        common["directions"] = {
            "left_to_right": {
                "gain": 0.040,
                "simultaneous_interval": [0.025, 0.055],
            },
            "right_to_left": {
                "gain": 0.038,
                "simultaneous_interval": [0.024, 0.052],
            },
        }
    elif kind == "adequate":
        common["directions"] = {
            "left_to_right": {
                "gain": 0.003,
                "simultaneous_interval": [-0.009, 0.015],
            },
            "right_to_left": {
                "gain": 0.002,
                "simultaneous_interval": [-0.010, 0.014],
            },
        }
    elif kind == "unresolved":
        common["directions"] = {
            "left_to_right": {
                "gain": 0.035,
                "simultaneous_interval": [0.021, 0.050],
            },
            "right_to_left": {
                "gain": 0.005,
                "simultaneous_interval": [-0.018, 0.028],
            },
        }
    else:
        raise ValueError(f"unknown final fixture kind: {kind}")
    return common


def run_complete_terminal_scenario(
    *,
    kind: str,
    run_dir: Path,
    policy_path: Path,
    inject_trial_interruption: bool = False,
    inject_lost_final_acknowledgement: bool = False,
    final_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    controller = make_controller(run_dir, policy_path)
    search = run_observed_search(
        controller,
        interrupt_trial=7 if inject_trial_interruption else None,
    )
    controller = search["controller"]
    action, reason = controller.check_stop()
    if (action, reason) != ("stop", "patience_exhausted"):
        raise RuntimeError(f"synthetic search did not reach frozen patience stop: {(action, reason)}")
    controller.record_stop(reason)
    selection_program_hash = digest_object(
        {
            "id": "synthetic_qualification_only",
            "policy_hash": controller.policy.policy_hash,
        }
    )
    observed = 0.036
    receipts = null_receipts(
        controller,
        observed_statistic=observed,
        selection_program_hash=selection_program_hash,
    )
    receipt_path = run_dir / "synthetic_null_receipts.json"
    atomic_json(receipt_path, receipts)
    null_record = controller.run_full_search_null(
        generator_lock_hash=digest_object({"synthetic_generator": "U"}),
        replicate_manifest=receipts,
        selected_trial_id=search["selected_trial_id"],
        observed_statistic=observed,
        selection_program_hash=selection_program_hash,
    )
    lock_hash = controller.lock_configuration(
        search["selected_trial_id"],
        synthetic_selection_rule_id="synthetic_qualification_only",
        final_output_allowlist=controller.SYNTHETIC_FINAL_OUTPUT_FIELDS,
    )
    vault = SyntheticAuditVault(
        run_dir / "synthetic_final_vault.json",
        dict(final_payload) if final_payload is not None else final_result(kind, controller),
    )
    operation_id = f"synthetic-final-{kind}"
    lost_ack_observed = False
    if inject_lost_final_acknowledgement:
        try:
            controller.run_audit(
                lock_hash,
                vault,
                operation_id=operation_id,
                simulate_interruption_after_evaluator_commit=True,
            )
        except SimulatedInterruption:
            lost_ack_observed = True
            controller = make_controller(run_dir, policy_path)
    terminal = controller.run_audit(
        lock_hash,
        vault,
        operation_id=operation_id,
    )
    return {
        "scenario": kind,
        "terminal": terminal,
        "status": controller.status(),
        "lock_hash": lock_hash,
        "null_p": null_record["metrics"]["monte_carlo_p"],
        "null_replicates": null_record["metrics"]["replicates"],
        "protocol_scale_model_family_fit_count": null_record["metrics"]
        ["protocol_scale_model_family_fit_count"],
        "trial_interruption_recovered": search["interruption_injected"],
        "lost_acknowledgement_reconciled": lost_ack_observed,
        "evaluator_invocation_count": vault.evaluator_invocation_count,
        "evaluator_access_count": vault.access_count,
        "journal_path": str(controller.journal.path),
        "synthetic_null_receipts_path": str(receipt_path),
        "synthetic_null_receipts_sha256": digest_bytes(receipt_path.read_bytes()),
    }


def run_no_valid_comparison(run_dir: Path, policy_path: Path) -> dict[str, Any]:
    controller = make_controller(run_dir, policy_path)
    for index, config in enumerate(balanced_coverage_configs(controller.policy), start=1):
        trial_id = f"rejected-{index:04d}"
        controller.register_hypothesis(
            trial_id=trial_id,
            parent_trial_ids=(),
            proposal_mode="prespecified_coverage",
            proposal_context_hash=controller.journal.prefix_hash(),
            outcome_evidence_refs=(),
            successor_cycle_id=None,
            prediction="synthetic readiness branch",
            falsifier="not executable under fixture",
            changed_operator="covering_array_factor",
            unchanged_operators=("T_U_M_triplet",),
            expected_information_gain="readiness logging",
            expected_cost={"synthetic_core_hours": 0.0},
            config=config,
            stage="coverage",
        )
        evaluation = _evaluation(
            controller,
            selection_statistic=0.0,
            comparison_gate_overrides={"T_U_M_capacity_comparison_is_fair": False},
        )
        controller.run_development_trial(trial_id, evaluation)
    controller.record_stop("no_executable_branch")
    terminal = controller.close_without_audit(
        "closed_no_valid_comparison",
        "all attempted synthetic comparisons failed a declared validity gate",
    )
    return {
        "scenario": "no_valid_comparison",
        "terminal": terminal,
        "status": controller.status(),
        "journal_path": str(controller.journal.path),
    }


def run_budget_exhaustion(run_dir: Path, policy_path: Path) -> dict[str, Any]:
    controller = make_controller(run_dir, policy_path)
    search = run_observed_search(
        controller,
        total_trials=12,
        cpu_core_hours_per_trial=84.0,
    )
    controller = search["controller"]
    action, reason = controller.check_stop()
    if (action, reason) != ("stop", "cpu_budget_exhausted"):
        raise RuntimeError("budget fixture did not exhaust the policy ceiling")
    controller.record_stop(reason)
    terminal = controller.close_without_audit(
        "incomplete_search",
        "synthetic CPU budget exhausted before minimum trials and coverage",
    )
    return {
        "scenario": "budget_exhaustion",
        "terminal": terminal,
        "status": controller.status(),
        "journal_path": str(controller.journal.path),
    }


def run_input_failure(run_dir: Path, policy_path: Path) -> dict[str, Any]:
    controller = make_controller(run_dir, policy_path)
    controller.record_input_failure("synthetic source-manifest hash mismatch")
    terminal = controller.close_without_audit(
        "technical_failure",
        "synthetic source-manifest hash mismatch",
    )
    return {
        "scenario": "input_failure",
        "terminal": terminal,
        "status": controller.status(),
        "journal_path": str(controller.journal.path),
    }


def bootstrap_fixture_matrix() -> list[dict[str, Any]]:
    """List required scientific-shape tests that remain deliberately pending."""

    return [
        {
            "fixture": "adequate_single_template",
            "expected_family": "T_type_template",
            "expected_controller_class": "adequacy_capable",
            "implemented": False,
            "counts_toward_qualification": False,
        },
        {
            "fixture": "continuous_unimodal_variation",
            "expected_family": "U_flexible_unimodal",
            "expected_controller_class": "continuous_readout",
            "implemented": False,
            "counts_toward_qualification": False,
        },
        {
            "fixture": "true_residual_groups",
            "expected_family": "M_residual_modes",
            "expected_controller_class": "positive_capable",
            "implemented": False,
            "counts_toward_qualification": False,
        },
        {
            "fixture": "weak_signal",
            "expected_family": None,
            "expected_controller_class": "unresolved_capable",
            "implemented": False,
            "counts_toward_qualification": False,
        },
        {
            "fixture": "side_artifact",
            "expected_family": None,
            "expected_controller_class": "control_rejection_capable",
            "implemented": False,
            "counts_toward_qualification": False,
        },
    ]


def _resolve_durable_artifact(output_dir: Path, raw_path: Any, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise FileExistsError(f"completed qualification is missing {label}")
    path = Path(raw_path).resolve()
    if output_dir != path and output_dir not in path.parents:
        raise FileExistsError(f"completed qualification {label} escapes its output directory")
    if not path.is_file():
        raise FileExistsError(f"completed qualification {label} is missing: {path}")
    return path


def _verify_completed_artifacts(output_dir: Path, report: Mapping[str, Any]) -> None:
    scenarios = report.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 6:
        raise FileExistsError("completed qualification has an invalid scenario inventory")
    policy_summary = report.get("policy")
    if not isinstance(policy_summary, Mapping):
        raise FileExistsError("completed qualification is missing its policy summary")
    expected_null_replicates = policy_summary.get("null_replicates")
    for scenario in scenarios:
        if not isinstance(scenario, Mapping):
            raise FileExistsError("completed qualification contains malformed scenario metadata")
        journal_path = _resolve_durable_artifact(
            output_dir, scenario.get("journal_path"), "scenario journal"
        )
        journal_hash = digest_object(
            journal_path.read_text(encoding="utf-8").splitlines()
        )
        if journal_hash != scenario.get("journal_hash"):
            raise FileExistsError(f"scenario journal hash mismatch: {journal_path}")
        HashChainedJournal(journal_path).verify()

        complete_search = scenario.get("scenario") in {"positive", "adequate", "unresolved"}
        vault_raw = scenario.get("synthetic_final_vault_path")
        if complete_search and vault_raw is None:
            raise FileExistsError("complete-search scenario is missing its synthetic final vault")
        if vault_raw is not None:
            vault_path = _resolve_durable_artifact(
                output_dir, vault_raw, "synthetic final vault"
            )
            if digest_bytes(vault_path.read_bytes()) != scenario.get(
                "synthetic_final_vault_sha256"
            ):
                raise FileExistsError(f"synthetic final vault hash mismatch: {vault_path}")

        receipts_raw = scenario.get("synthetic_null_receipts_path")
        if complete_search and receipts_raw is None:
            raise FileExistsError("complete-search scenario is missing synthetic null receipts")
        if receipts_raw is not None:
            receipts_path = _resolve_durable_artifact(
                output_dir, receipts_raw, "synthetic null receipts"
            )
            if digest_bytes(receipts_path.read_bytes()) != scenario.get(
                "synthetic_null_receipts_sha256"
            ):
                raise FileExistsError(f"synthetic null receipt hash mismatch: {receipts_path}")
            receipts = json.loads(receipts_path.read_text(encoding="utf-8"))
            if len(receipts) != expected_null_replicates or any(
                receipt.get("receipt_scope") != "asserted_synthetic_not_executed"
                or receipt.get("complete_controller_rerun") is not False
                or receipt.get("deterministic_replay_passed") is not False
                for receipt in receipts
            ):
                raise FileExistsError(
                    f"synthetic null receipt assertions are malformed: {receipts_path}"
                )


def qualify_all(
    *,
    episode_root: Path,
    scratch_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    episode_root = episode_root.resolve()
    scratch_root = scratch_root.resolve()
    output_dir = output_dir.resolve()
    expected_output_root = (episode_root / "outputs").resolve()
    if episode_root != EPISODE_ROOT.resolve():
        raise ValueError(f"episode root must be canonical EP12 root: {EPISODE_ROOT}")
    if output_dir != expected_output_root and expected_output_root not in output_dir.parents:
        raise ValueError("durable qualification output must stay under EP12 outputs/")
    if scratch_root == episode_root or episode_root in scratch_root.parents:
        raise ValueError("scratch root must be separate from the durable episode root")
    if scratch_root != SCRATCH_ROOT.resolve():
        raise ValueError(f"qualification scratch root must be {SCRATCH_ROOT}")
    policy_path = episode_root / "SEARCH_POLICY.yaml"
    policy = EpisodePolicy.load(policy_path)
    code_hash = executor_code_hash()
    completion_path = output_dir / "qualification_manifest.json"
    report_path = output_dir / "qualification_report.json"
    incomplete_marker = output_dir / ".ep12_qualification_incomplete.json"
    if completion_path.is_file() and report_path.is_file():
        manifest = json.loads(completion_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            manifest.get("policy_hash") != policy.policy_hash
            or manifest.get("executor_code_hash") != code_hash
            or manifest.get("report_hash") != digest_object(report)
        ):
            raise FileExistsError(
                "completed qualification belongs to different code/policy; choose a new output path"
            )
        if manifest.get("scenario_count") != len(report.get("scenarios", [])):
            raise FileExistsError("completed qualification scenario count does not match report")
        _verify_completed_artifacts(output_dir, report)
        if incomplete_marker.exists():
            incomplete_marker.unlink()
        return report
    if output_dir.exists() and any(output_dir.iterdir()) and not incomplete_marker.is_file():
        raise FileExistsError(
            f"durable qualification directory is unmanaged/nonempty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        incomplete_marker,
        {
            "managed_by": "ep12_executor",
            "qualification_scope": "synthetic_control_plane_only",
            "policy_hash": policy.policy_hash,
            "executor_code_hash": code_hash,
        },
    )
    work_dir = scratch_root / "qualification_work"
    if work_dir.exists():
        marker = work_dir / ".ep12_synthetic_qualification_work"
        if not marker.is_file():
            raise ValueError(f"refusing to replace unmarked scratch directory: {work_dir}")
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    (work_dir / ".ep12_synthetic_qualification_work").write_text(
        "EP12 generated synthetic qualification workspace\n", encoding="utf-8"
    )

    scenarios = [
        run_complete_terminal_scenario(
            kind="positive",
            run_dir=work_dir / "positive",
            policy_path=policy_path,
            inject_trial_interruption=True,
            inject_lost_final_acknowledgement=True,
        ),
        run_complete_terminal_scenario(
            kind="adequate",
            run_dir=work_dir / "adequate",
            policy_path=policy_path,
        ),
        run_complete_terminal_scenario(
            kind="unresolved",
            run_dir=work_dir / "unresolved",
            policy_path=policy_path,
        ),
        run_no_valid_comparison(work_dir / "no_valid", policy_path),
        run_budget_exhaustion(work_dir / "budget", policy_path),
        run_input_failure(work_dir / "input_failure", policy_path),
    ]

    expected = {
        "positive": ("candidate_ready_residual_modes", "candidate_ready"),
        "adequate": (
            "closed_single_population_adequate_within_margin",
            "closed_no_candidate",
        ),
        "unresolved": ("closed_unresolved", "closed_no_candidate"),
        "no_valid_comparison": ("closed_no_valid_comparison", "closed_no_candidate"),
        "budget_exhaustion": ("incomplete_search", "closed_no_candidate"),
        "input_failure": ("technical_failure", "technical_failure"),
    }
    checks: list[dict[str, Any]] = []
    for scenario in scenarios:
        expected_detailed, expected_outer = expected[scenario["scenario"]]
        terminal = scenario["terminal"]
        passed = (
            terminal["detailed_outcome"] == expected_detailed
            and terminal["outer_status"] == expected_outer
            and scenario["status"]["real_connectivity_access_count"] == 0
        )
        checks.append(
            {
                "check": f"terminal:{scenario['scenario']}",
                "passed": passed,
                "observed": terminal,
            }
        )
        if scenario["scenario"] in {"positive", "adequate", "unresolved"}:
            checks.append(
                {
                    "check": f"complete_search_invariants:{scenario['scenario']}",
                    "passed": scenario["status"]["valid_trials"] >= policy.minimum_valid_trials
                    and scenario["status"]["coverage_complete"]
                    and scenario["status"]["required_falsifier_ids_exercised"]
                    and scenario["status"]["final_open_count"] == 1
                    and scenario["null_replicates"] == policy.null_replicates,
                }
            )
        else:
            checks.append(
                {
                    "check": f"failure_final_remains_unopened:{scenario['scenario']}",
                    "passed": scenario["status"]["final_open_count"] == 0,
                }
            )
    positive = scenarios[0]
    checks.extend(
        [
            {
                "check": "interruption_resume_no_double_count",
                "passed": positive["trial_interruption_recovered"]
                and positive["status"]["valid_trials"] == 52,
            },
            {
                "check": "lost_final_ack_exactly_once",
                "passed": positive["lost_acknowledgement_reconciled"]
                and positive["status"]["final_open_count"] == 1
                and positive["evaluator_invocation_count"] == 1
                and positive["evaluator_access_count"] == 1,
            },
            {
                "check": "asserted_null_receipt_protocol_arithmetic",
                "passed": all(
                    scenario.get("null_replicates") == policy.null_replicates
                    and scenario.get("protocol_scale_model_family_fit_count", 0)
                    >= policy.minimum_valid_trials * 3 * (policy.null_replicates + 1)
                    for scenario in scenarios[:3]
                ),
            },
            {
                "check": "no_real_connectivity_access",
                "passed": all(
                    scenario["status"]["real_connectivity_access_count"] == 0
                    for scenario in scenarios
                ),
            },
        ]
    )
    fixtures = bootstrap_fixture_matrix()
    passed = all(check["passed"] for check in checks)

    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_output = output_dir / "scenarios"
    scenario_output.mkdir(parents=True, exist_ok=True)
    durable_scenarios: list[dict[str, Any]] = []
    for scenario in scenarios:
        source = Path(scenario["journal_path"])
        destination_dir = scenario_output / scenario["scenario"]
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / "trial_ledger.jsonl"
        shutil.copy2(source, destination)
        durable = dict(scenario)
        durable["journal_path"] = str(destination)
        durable["journal_hash"] = digest_object(
            destination.read_text(encoding="utf-8").splitlines()
        )
        vault_source = source.parent / "synthetic_final_vault.json"
        if vault_source.is_file():
            vault_destination = destination_dir / "synthetic_final_vault.json"
            shutil.copy2(vault_source, vault_destination)
            durable["synthetic_final_vault_path"] = str(vault_destination)
            durable["synthetic_final_vault_sha256"] = digest_bytes(
                vault_destination.read_bytes()
            )
        receipts_source = source.parent / "synthetic_null_receipts.json"
        if receipts_source.is_file():
            receipts_destination = destination_dir / "synthetic_null_receipts.json"
            shutil.copy2(receipts_source, receipts_destination)
            durable["synthetic_null_receipts_path"] = str(receipts_destination)
            durable["synthetic_null_receipts_sha256"] = digest_bytes(
                receipts_destination.read_bytes()
            )
        durable_scenarios.append(durable)

    report = {
        "schema_version": "ep12.executor_qualification.v1",
        "qualification_passed": passed,
        "qualification_scope": "synthetic_control_plane_only",
        "scientific_model_qualification": False,
        "required_scientific_shape_qualification_passed": False,
        "real_data_execution_enabled": False,
        "scientific_interpretation_permitted": False,
        "executed_full_search_null_completed": False,
        "episode": {
            "episode_id": "ep12",
            "episode_root": str(episode_root),
            "repository_root": str(episode_root.parent),
            "durable_output_root": str(episode_root / "outputs"),
            "read_only_input_root": str(episode_root / "inputs"),
            "temporary_root": str(scratch_root),
            "policy_path": str(policy_path),
            "common_protocol_path": str(episode_root.parent / "ADAPTIVE_SEARCH_PROTOCOL.md"),
            "controller_interface_path": str(
                episode_root.parent / "ADAPTIVE_CONTROLLER_INTERFACE.md"
            ),
        },
        "policy": policy.contract_summary(),
        "executor_code_hash": code_hash,
        "checks": checks,
        "pending_scientific_shape_tests": fixtures,
        "scenarios": durable_scenarios,
        "connectivity_outcomes_inspected": False,
        "male_cns_source_accessed": False,
        "followup_started": False,
        "canonical_review_or_reward_requested": False,
        "limitations": [
            "No real MaleCNS connectivity, body-statistic, synapse, or neurotransmitter value was opened.",
            "The T/U/M scientific estimators, capacity matching, uncertainty implementation, and power grid remain unset.",
            "Fixture routing is not sensitivity or scientific-model qualification.",
            "The 99 null records are asserted synthetic receipt fixtures, not executed full controller reruns or a cost profile.",
            "The executor intentionally exposes no real-data command.",
        ],
    }
    _verify_completed_artifacts(output_dir, report)
    atomic_json(output_dir / "qualification_report.json", report)
    manifest = {
        "report": "qualification_report.json",
        "report_hash": digest_object(report),
        "qualification_passed": passed,
        "qualification_scope": "synthetic_control_plane_only",
        "scientific_model_qualification": False,
        "required_scientific_shape_qualification_passed": False,
        "executed_full_search_null_completed": False,
        "scientific_interpretation_permitted": False,
        "real_data_execution_enabled": False,
        "scenario_count": len(durable_scenarios),
        "connectivity_outcomes_inspected": False,
        "policy_hash": policy.policy_hash,
        "executor_code_hash": code_hash,
    }
    atomic_json(output_dir / "qualification_manifest.json", manifest)
    incomplete_marker.unlink()
    directory_fd = os.open(str(output_dir), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return report
