"""Fail-closed EP12 adaptive-controller control plane.

This module intentionally has no MaleCNS loader.  It qualifies the controller
with generated fixtures while consequential estimator and eligibility choices
remain unset.  Adding a real-data backend is a separate, explicit step.
"""

from __future__ import annotations

import itertools
import math
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .journal import HashChainedJournal
from .policy import (
    ConfigurationError,
    EpisodePolicy,
    digest_bytes,
    digest_object,
)


class ControllerError(RuntimeError):
    """Base controller error."""


class StateTransitionError(ControllerError):
    """A requested transition is not legal in the replayed state."""


class RoleLeakageError(ControllerError):
    """Synthetic access attempted to cross the frozen whole-type boundary."""


class AuditError(ControllerError):
    """Final evaluation could not satisfy the exactly-once contract."""


class SimulatedInterruption(ControllerError):
    """Fault injection used only by synthetic qualification."""


@dataclass(frozen=True)
class SyntheticSourceManifest:
    source_id: str
    provider_id: str = "synthetic:ep12-qualification"
    kind: str = "generated_synthetic"
    contains_real_connectivity: bool = False
    real_connectivity_access_count: int = 0
    male_cns_declared_source_accessed: bool = False

    def validate(self) -> None:
        if self.kind != "generated_synthetic":
            raise ControllerError("only a generated-synthetic source is enabled")
        if self.contains_real_connectivity or self.real_connectivity_access_count != 0:
            raise ControllerError("synthetic qualification may not contain real connectivity")
        if self.male_cns_declared_source_accessed:
            raise ControllerError("MaleCNS must remain unopened during qualification")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "provider_id": self.provider_id,
            "kind": self.kind,
            "contains_real_connectivity": self.contains_real_connectivity,
            "real_connectivity_access_count": self.real_connectivity_access_count,
            "male_cns_declared_source_accessed": self.male_cns_declared_source_accessed,
        }


@dataclass(frozen=True)
class WholeTypeRoleManifest:
    development_types: tuple[str, ...]
    final_types: tuple[str, ...]
    assignment_unit: str = "whole_provider_type"
    assignment_inputs: tuple[str, ...] = ("synthetic_annotation",)
    connectivity_used_for_assignment: bool = False
    deterministic_seed: int = 12012

    def validate(self) -> None:
        development = set(self.development_types)
        final = set(self.final_types)
        if self.assignment_unit != "whole_provider_type":
            raise RoleLeakageError("roles must be assigned by whole provider type")
        if not development or not final:
            raise RoleLeakageError("synthetic role manifest needs development and final types")
        if len(development) != len(self.development_types):
            raise RoleLeakageError("duplicate development type")
        if len(final) != len(self.final_types):
            raise RoleLeakageError("duplicate final type")
        overlap = development & final
        if overlap:
            raise RoleLeakageError(
                f"whole types cannot cross roles: {', '.join(sorted(overlap))}"
            )
        if self.connectivity_used_for_assignment:
            raise RoleLeakageError("connectivity cannot be used for role assignment")
        forbidden = {"provider_group", "provider_instance", "body_statistics", "weights"}
        if forbidden.intersection(self.assignment_inputs):
            raise RoleLeakageError("role assignment requested a forbidden field")
        if len(development) != 4 * len(final):
            raise RoleLeakageError("synthetic whole-type roles must preserve the frozen 80/20 split")
        if not isinstance(self.deterministic_seed, int):
            raise RoleLeakageError("whole-type assignment requires a deterministic seed")

    def as_dict(self) -> dict[str, Any]:
        return {
            "development_types": list(self.development_types),
            "final_types": list(self.final_types),
            "assignment_unit": self.assignment_unit,
            "assignment_inputs": list(self.assignment_inputs),
            "connectivity_used_for_assignment": self.connectivity_used_for_assignment,
            "deterministic_seed": self.deterministic_seed,
        }


class AuditEvaluator(Protocol):
    def reconcile(self, operation_id: str, lock_hash: str) -> Mapping[str, Any] | None:
        ...

    def evaluate(self, operation_id: str, lock_hash: str) -> Mapping[str, Any]:
        ...


@dataclass
class ReplayState:
    records: list[dict[str, Any]]
    trials: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    stop_reason: str | None = None
    null_record: dict[str, Any] | None = None
    lock_record: dict[str, Any] | None = None
    audit_open_record: dict[str, Any] | None = None
    audit_result_record: dict[str, Any] | None = None
    terminal_record: dict[str, Any] | None = None
    input_failure_record: dict[str, Any] | None = None

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> "ReplayState":
        state = cls(records=records)
        for record in records:
            kind = record.get("record_kind")
            if kind == "trial_transition":
                state.trials.setdefault(record["trial_id"], []).append(record)
            elif kind == "stop":
                state.stop_reason = record["disposition"]
            elif kind == "full_search_null":
                state.null_record = record
            elif kind == "configuration_lock":
                state.lock_record = record
            elif kind == "audit_open":
                if state.audit_open_record is not None:
                    raise StateTransitionError("more than one final opening is recorded")
                state.audit_open_record = record
            elif kind == "audit_result":
                state.audit_result_record = record
            elif kind == "terminal":
                state.terminal_record = record
            elif kind == "input_failure":
                state.input_failure_record = record
        state._validate_trial_transitions()
        if state.audit_result_record and not state.audit_open_record:
            raise StateTransitionError("audit result exists without an opening")
        if state.lock_record and state.stop_reason is None:
            raise StateTransitionError("configuration lock exists without a stop record")
        return state

    def _validate_trial_transitions(self) -> None:
        allowed: dict[str | None, set[str]] = {
            None: {"PROPOSED"},
            "PROPOSED": {"READINESS_REJECTED", "RUNNING"},
            "RUNNING": {"ENGINEERING_FAILED", "INVALID", "SCORED"},
            "SCORED": {"RETIRED", "CHALLENGER", "INCUMBENT"},
            "READINESS_REJECTED": set(),
            "ENGINEERING_FAILED": set(),
            "INVALID": set(),
            "RETIRED": set(),
            "CHALLENGER": set(),
            "INCUMBENT": set(),
        }
        for trial_id, transitions in self.trials.items():
            previous: str | None = None
            for transition in transitions:
                current = transition["state"]
                if current not in allowed.get(previous, set()):
                    raise StateTransitionError(
                        f"illegal transition for {trial_id}: {previous!r} -> {current!r}"
                    )
                previous = current

    @property
    def audit_open_count(self) -> int:
        return int(self.audit_open_record is not None)

    @property
    def valid_trial_ids(self) -> tuple[str, ...]:
        return tuple(
            trial_id
            for trial_id, records in self.trials.items()
            if any(record["state"] == "SCORED" for record in records)
        )

    @property
    def valid_trial_count(self) -> int:
        return len(self.valid_trial_ids)

    def latest_trial_record(self, trial_id: str) -> dict[str, Any] | None:
        records = self.trials.get(trial_id, [])
        return records[-1] if records else None

    def proposed_trial_record(self, trial_id: str) -> dict[str, Any] | None:
        for record in self.trials.get(trial_id, []):
            if record["state"] == "PROPOSED":
                return record
        return None

    def scored_trial_record(self, trial_id: str) -> dict[str, Any] | None:
        for record in self.trials.get(trial_id, []):
            if record["state"] == "SCORED":
                return record
        return None

    @property
    def disposition_records(self) -> list[dict[str, Any]]:
        return [
            records[-1]
            for records in self.trials.values()
            if records[-1]["state"] in {"RETIRED", "CHALLENGER", "INCUMBENT"}
        ]


class Episode12Controller:
    """Event-sourced controller restricted to synthetic EP12 qualification."""

    COVERAGE_FACTORS = (
        "partner_vocabulary",
        "representation",
        "covariance",
        "component_count",
    )
    RESIDUAL_SUPPORT_KEYS = (
        "separated_and_prevalent_modes",
        "differentiating_partner_preferences",
        "cross_side_structure_matches",
        "anatomical_and_technical_controls_pass",
        "gain_not_concentrated",
        "selection_and_influence_controls_pass",
    )
    PRE_NULL_COMPARISON_GATES = (
        "data_roles_and_quality_pass",
        "T_U_M_capacity_comparison_is_fair",
        "T_U_M_calibration_qualified",
        "numerical_unimodality_for_U",
        "endpoint_mass_preserved",
        "synthetic_qualification_passed",
        "deterministic_replay_passed",
    )
    SYNTHETIC_FINAL_OUTPUT_FIELDS = (
        "valid_final",
        "comparison_valid",
        "directions",
        "every_final_type_reported",
        "reported_final_types",
        "per_type_results",
        "target_side_refit",
        "source_side_fit_only",
        "reference_calibrated",
        "adequacy_sensitivity_qualified",
        "residual_support",
        "followup_started",
        "paper_level_claim_emitted",
    )

    def __init__(
        self,
        *,
        policy: EpisodePolicy,
        journal_path: Path,
        source_manifest: SyntheticSourceManifest,
        role_manifest: WholeTypeRoleManifest,
    ):
        source_manifest.validate()
        role_manifest.validate()
        self.policy = policy
        self.journal = HashChainedJournal(journal_path)
        self.source_manifest = source_manifest
        self.role_manifest = role_manifest
        self.source_hash = digest_object(source_manifest.as_dict())
        self.split_hash = digest_object(role_manifest.as_dict())
        self.environment_hash = digest_object(
            {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
            }
        )
        package_root = Path(__file__).resolve().parent
        code_material = b"".join(
            path.name.encode("utf-8") + b"\0" + path.read_bytes()
            for path in sorted(package_root.glob("*.py"))
        )
        self.code_hash = digest_bytes(code_material)
        self._validate_existing_identity()

    def _validate_existing_identity(self) -> None:
        expected = {
            "policy": self.policy.policy_hash,
            "code": self.code_hash,
            "environment": self.environment_hash,
            "source": self.source_hash,
            "split": self.split_hash,
        }
        for record in self.journal.read():
            hashes = record.get("hashes")
            if not isinstance(hashes, Mapping):
                raise ControllerError("journal record is missing identity hashes")
            mismatched = [key for key, value in expected.items() if hashes.get(key) != value]
            if mismatched:
                raise ControllerError(
                    "journal identity mismatch on resume: " + ", ".join(mismatched)
                )

    @classmethod
    def load_policy(
        cls,
        policy_path: Path,
        source_manifest: SyntheticSourceManifest,
        split_manifest: WholeTypeRoleManifest,
        journal_path: Path,
    ) -> "Episode12Controller":
        return cls(
            policy=EpisodePolicy.load(policy_path),
            journal_path=journal_path,
            source_manifest=source_manifest,
            role_manifest=split_manifest,
        )

    def replay(self) -> ReplayState:
        return ReplayState.from_records(self.journal.read())

    def status(self) -> dict[str, Any]:
        state = self.replay()
        integrity = self.journal.verify()
        coverage = self.coverage_status(state)
        return {
            "episode_id": self.policy.episode_id,
            "mode": "synthetic_qualification_only",
            "qualification_scope": "synthetic_control_plane_only",
            "scientific_interpretation_permitted": False,
            "real_data_execution_enabled": False,
            "scientific_model_qualification": False,
            "executed_full_search_null_completed": False,
            "trial_count_semantics": "synthetic_state_machine_fixture_count",
            "valid_trials": state.valid_trial_count,
            "coverage_complete": coverage["complete"],
            "adaptive_cycles": self._adaptive_cycle_count(state),
            "incumbent_challenger_decisions": self._decision_count(state),
            "post_coverage_falsifier_fraction": self._falsifier_fraction(state),
            "required_falsifier_ids_exercised": self._required_falsifier_status(state)[
                "complete"
            ],
            "cpu_core_hours_recorded": self._resource_usage(state),
            "wall_hours_recorded": self._wall_usage(state),
            "stop_reason": state.stop_reason,
            "null_completed": state.null_record is not None,
            "configuration_locked": state.lock_record is not None,
            "final_open_count": state.audit_open_count,
            "terminal": (
                state.terminal_record.get("terminal") if state.terminal_record else None
            ),
            "real_connectivity_access_count": 0,
            **integrity,
        }

    def _timestamp(self) -> str:
        offset = len(self.journal.read())
        value = datetime(2026, 9, 24, tzinfo=timezone.utc) + timedelta(seconds=offset)
        return value.isoformat().replace("+00:00", "Z")

    def _hashes(self, config: Mapping[str, Any]) -> dict[str, str]:
        return {
            "policy": self.policy.policy_hash,
            "code": self.code_hash,
            "environment": self.environment_hash,
            "source": self.source_hash,
            "split": self.split_hash,
            "config": digest_object(config),
        }

    def _record_payload(
        self,
        *,
        record_kind: str,
        trial_id: str,
        proposal_mode: str,
        stage: str,
        state: str,
        config: Mapping[str, Any] | None = None,
        parent_trial_ids: Sequence[str] = (),
        proposal_context_hash: str | None = None,
        outcome_evidence_refs: Sequence[str] = (),
        successor_cycle_id: str | None = None,
        configuration_lock_hash: str | None = None,
        hypothesis: str = "EP12 synthetic controller qualification",
        directional_prediction: str = "fixture-declared synthetic prediction",
        falsifier: str = "fixture-declared synthetic falsifier",
        changed_operator: str = "synthetic_fixture",
        unchanged_operators: Sequence[str] = ("T_U_M_triplet",),
        expected_information_gain: str = "controller qualification only",
        expected_cost: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        controls: Mapping[str, Any] | None = None,
        resources: Mapping[str, Any] | None = None,
        belief_update: str = "none; synthetic qualification is not scientific evidence",
        successor_rationale: str = "not applicable",
        disposition: str = "recorded",
        incumbent_decision: Mapping[str, Any] | None = None,
        audit_receipt: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        config_dict = dict(config or {})
        payload: dict[str, Any] = {
            "record_kind": record_kind,
            "qualification_scope": "synthetic_control_plane_only",
            "scientific_interpretation_permitted": False,
            "real_data_execution_enabled": False,
            "scientific_model_qualification": False,
            "trial_id": trial_id,
            "parent_trial_ids": list(parent_trial_ids),
            "proposal_mode": proposal_mode,
            "proposal_context_hash": proposal_context_hash or self.journal.prefix_hash(),
            "outcome_evidence_refs": list(outcome_evidence_refs),
            "successor_cycle_id": successor_cycle_id,
            "configuration_lock_hash": configuration_lock_hash,
            "stage": stage,
            "hypothesis": hypothesis,
            "directional_prediction": directional_prediction,
            "falsifier": falsifier,
            "changed_operator": changed_operator,
            "unchanged_operators": list(unchanged_operators),
            "expected_information_gain": expected_information_gain,
            "expected_cost": dict(expected_cost or {"synthetic_core_hours": 0.0}),
            "config": config_dict,
            "hashes": self._hashes(config_dict),
            "state": state,
            "metrics": dict(metrics) if metrics is not None else None,
            "controls": dict(controls) if controls is not None else None,
            "resources": dict(resources or {"cpu_core_hours": 0.0, "gpu_hours": 0.0}),
            "belief_update": belief_update,
            "successor_rationale": successor_rationale,
            "disposition": disposition,
            "incumbent_decision": (
                dict(incumbent_decision) if incumbent_decision is not None else None
            ),
            "audit_receipt": dict(audit_receipt) if audit_receipt is not None else None,
        }
        if extra:
            payload.update(extra)
        return payload

    def _append(self, event_id: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.journal.append(
            event_id=event_id, timestamp_utc=self._timestamp(), payload=payload
        ).record

    def _assert_development_mutable(
        self,
        *,
        allow_stopped: bool = False,
        allow_active_stop: bool = False,
    ) -> None:
        state = self.replay()
        if state.lock_record or state.audit_open_record or state.terminal_record:
            raise StateTransitionError("development is immutable after lock/final/terminal")
        if state.input_failure_record:
            raise StateTransitionError("run is closed by an input failure")
        if state.stop_reason and not allow_stopped:
            raise StateTransitionError("development is immutable after a stop is recorded")
        if not allow_active_stop and not state.stop_reason:
            action, reason = self.check_stop()
            if action == "stop":
                raise StateTransitionError(
                    f"development cannot continue after active stop condition: {reason}"
                )

    def validate_config(self, config: Mapping[str, Any]) -> tuple[bool, str | None]:
        try:
            self.policy.validate_config(config)
        except ConfigurationError as exc:
            return False, str(exc)
        return True, None

    def register_hypothesis(
        self,
        *,
        trial_id: str,
        parent_trial_ids: Sequence[str],
        proposal_mode: str,
        proposal_context_hash: str,
        outcome_evidence_refs: Sequence[str],
        successor_cycle_id: str | None,
        prediction: str,
        falsifier: str,
        changed_operator: str,
        unchanged_operators: Sequence[str],
        expected_information_gain: str,
        expected_cost: Mapping[str, Any],
        config: Mapping[str, Any],
        stage: str,
    ) -> Mapping[str, Any]:
        sequence_fields = {
            "parent_trial_ids": parent_trial_ids,
            "outcome_evidence_refs": outcome_evidence_refs,
            "unchanged_operators": unchanged_operators,
        }
        if not isinstance(trial_id, str) or not trial_id.strip() or trial_id.startswith("__"):
            raise StateTransitionError("trial id must be a nonempty, non-reserved string")
        if not isinstance(proposal_context_hash, str) or len(proposal_context_hash) < 32:
            raise StateTransitionError("proposal context hash must contain at least 32 characters")
        for field_name, values in sequence_fields.items():
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise StateTransitionError(f"{field_name} must be a sequence of strings")
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise StateTransitionError(f"{field_name} contains an empty or non-string value")
        text_fields = {
            "prediction": prediction,
            "falsifier": falsifier,
            "changed_operator": changed_operator,
            "expected_information_gain": expected_information_gain,
        }
        for field_name, value in text_fields.items():
            if not isinstance(value, str) or not value.strip():
                raise StateTransitionError(f"{field_name} must be a nonempty string")
        if not unchanged_operators:
            raise StateTransitionError("unchanged_operators must contain at least one item")
        if not isinstance(expected_cost, Mapping) or not expected_cost:
            raise StateTransitionError("expected_cost must be a nonempty object")
        if any(not isinstance(key, str) or not key for key in expected_cost):
            raise StateTransitionError("expected_cost keys must be nonempty strings")
        if not isinstance(config, Mapping):
            raise ConfigurationError("configuration must be an object")

        stage_modes = {
            "readiness": "readiness_or_baseline",
            "coverage": "prespecified_coverage",
            "adaptive": "outcome_adaptive_successor",
            "falsification": "mandated_falsifier",
        }
        if stage not in stage_modes:
            raise StateTransitionError(f"unsupported development stage: {stage!r}")
        if proposal_mode != stage_modes[stage]:
            raise StateTransitionError(
                f"proposal mode {proposal_mode!r} is not valid for stage {stage!r}"
            )
        if successor_cycle_id is not None and (
            not isinstance(successor_cycle_id, str) or not successor_cycle_id.strip()
        ):
            raise StateTransitionError("successor cycle id must be null or a nonempty string")
        if stage == "adaptive" and successor_cycle_id is None:
            raise StateTransitionError("adaptive successors require a cycle id")
        if stage != "adaptive" and successor_cycle_id is not None:
            raise StateTransitionError("only adaptive successors may name a cycle id")
        if stage in {"readiness", "coverage"} and (
            parent_trial_ids or outcome_evidence_refs
        ):
            raise StateTransitionError(
                "readiness and prespecified coverage proposals cannot cite outcomes"
            )

        self._assert_development_mutable()
        accepted, reason = self.validate_config(config)
        if not accepted:
            raise ConfigurationError(reason or "configuration rejected")
        state = self.replay()
        existing = state.proposed_trial_record(trial_id)
        if existing:
            same_request = (
                existing["hashes"]["config"] == digest_object(config)
                and existing["parent_trial_ids"] == list(parent_trial_ids)
                and existing["proposal_mode"] == proposal_mode
                and existing["proposal_context_hash"] == proposal_context_hash
                and existing["outcome_evidence_refs"] == list(outcome_evidence_refs)
                and existing["successor_cycle_id"] == successor_cycle_id
                and existing["stage"] == stage
                and existing["directional_prediction"] == prediction
                and existing["falsifier"] == falsifier
                and existing["changed_operator"] == changed_operator
                and existing["unchanged_operators"] == list(unchanged_operators)
                and existing["expected_information_gain"] == expected_information_gain
                and existing["expected_cost"] == dict(expected_cost)
            )
            if not same_request:
                raise StateTransitionError("trial id reused with a changed proposal request")
            return existing
        actual_prefix = self.journal.prefix_hash()
        if proposal_context_hash != actual_prefix:
            raise StateTransitionError("proposal context does not match visible ledger prefix")
        if stage == "adaptive":
            if not parent_trial_ids or not outcome_evidence_refs or not successor_cycle_id:
                raise StateTransitionError("adaptive successors need parents, evidence, and cycle")
            scored_records = {
                record["event_id"]
                for records in state.trials.values()
                for record in records
                if record["state"] == "SCORED"
            }
            if not set(outcome_evidence_refs).issubset(scored_records):
                raise StateTransitionError("adaptive evidence must reference prior scored records")
            if any(state.scored_trial_record(parent) is None for parent in parent_trial_ids):
                raise StateTransitionError("adaptive parent must already be scored")
        if stage == "falsification" and falsifier not in self.policy.required_falsifiers:
            raise StateTransitionError("falsification trials require a registered falsifier id")

        payload = self._record_payload(
            record_kind="trial_transition",
            trial_id=trial_id,
            proposal_mode=proposal_mode,
            stage=stage,
            state="PROPOSED",
            config=config,
            parent_trial_ids=parent_trial_ids,
            proposal_context_hash=proposal_context_hash,
            outcome_evidence_refs=outcome_evidence_refs,
            successor_cycle_id=successor_cycle_id,
            hypothesis=f"synthetic hypothesis for {trial_id}",
            directional_prediction=prediction,
            falsifier=falsifier,
            changed_operator=changed_operator,
            unchanged_operators=unchanged_operators,
            expected_information_gain=expected_information_gain,
            expected_cost=expected_cost,
            disposition="awaiting execution",
        )
        return self._append(f"trial:{trial_id}:proposed", payload)

    def readiness_reject(self, trial_id: str, reason: str) -> Mapping[str, Any]:
        self._assert_development_mutable()
        state = self.replay()
        proposed = state.proposed_trial_record(trial_id)
        latest = state.latest_trial_record(trial_id)
        if proposed is None:
            raise StateTransitionError("readiness rejection requires a proposal")
        if latest and latest["state"] == "READINESS_REJECTED":
            return latest
        if latest != proposed:
            raise StateTransitionError("readiness rejection is no longer legal")
        payload = self._transition_payload(
            proposed, state="READINESS_REJECTED", disposition=reason
        )
        return self._append(f"trial:{trial_id}:readiness-rejected", payload)

    def _transition_payload(
        self,
        proposed: Mapping[str, Any],
        *,
        state: str,
        metrics: Mapping[str, Any] | None = None,
        controls: Mapping[str, Any] | None = None,
        resources: Mapping[str, Any] | None = None,
        disposition: str,
        incumbent_decision: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._record_payload(
            record_kind="trial_transition",
            trial_id=proposed["trial_id"],
            proposal_mode=proposed["proposal_mode"],
            stage=proposed["stage"],
            state=state,
            config=proposed["config"],
            parent_trial_ids=proposed["parent_trial_ids"],
            proposal_context_hash=proposed["proposal_context_hash"],
            outcome_evidence_refs=proposed["outcome_evidence_refs"],
            successor_cycle_id=proposed["successor_cycle_id"],
            hypothesis=proposed["hypothesis"],
            directional_prediction=proposed["directional_prediction"],
            falsifier=proposed["falsifier"],
            changed_operator=proposed["changed_operator"],
            unchanged_operators=proposed["unchanged_operators"],
            expected_information_gain=proposed["expected_information_gain"],
            expected_cost=proposed["expected_cost"],
            metrics=metrics,
            controls=controls,
            resources=resources,
            disposition=disposition,
            incumbent_decision=incumbent_decision,
            extra=extra,
        )

    def _validate_evaluation_access(self, evaluation: Mapping[str, Any]) -> None:
        accesses = evaluation.get("accessed_types")
        if not isinstance(accesses, list):
            raise RoleLeakageError("evaluation must declare accessed synthetic types")
        allowed = set(self.role_manifest.development_types)
        unexpected = set(map(str, accesses)) - allowed
        if unexpected:
            raise RoleLeakageError(
                f"development evaluation accessed reserved types: {sorted(unexpected)}"
            )
        if evaluation.get("incoming_accessed"):
            raise RoleLeakageError("incoming profiles are forbidden in the outgoing round")
        if evaluation.get("provider_group_accessed") or evaluation.get(
            "provider_instance_accessed"
        ):
            raise RoleLeakageError("forbidden provider annotations were accessed")

    def _record_integrity_failure(self, trial_id: str, reason: str) -> None:
        payload = self._record_payload(
            record_kind="input_failure",
            trial_id=f"__integrity__:{trial_id}",
            proposal_mode="readiness_or_baseline",
            stage="readiness",
            state="ENGINEERING_FAILED",
            disposition=reason,
            extra={"affected_trial_id": trial_id},
        )
        self._append(f"operation:integrity-failure:{trial_id}", payload)

    def _validate_trial_resources(self, resources: Mapping[str, Any]) -> None:
        try:
            numeric_fields = {
                "cpu_core_hours": float(resources.get("cpu_core_hours", 0.0)),
                "gpu_hours": float(resources.get("gpu_hours", 0.0)),
                "cpu_cores": float(resources.get("cpu_cores", 0.0)),
                "memory_gib": float(resources.get("memory_gib", 0.0)),
                "wall_hours": float(resources.get("wall_hours", 0.0)),
            }
        except (TypeError, ValueError) as exc:
            raise ControllerError("trial resource receipt must be numeric") from exc
        if any(not math.isfinite(value) or value < 0 for value in numeric_fields.values()):
            raise ControllerError("trial resource receipt must be finite and nonnegative")
        if numeric_fields["gpu_hours"] > self.policy.gpu_hour_ceiling:
            raise ControllerError("GPU use exceeds the frozen zero-GPU budget")
        if numeric_fields["cpu_cores"] > self.policy.cpu_cores_per_trial_maximum:
            raise ControllerError("trial CPU-core request exceeds the policy")
        if numeric_fields["memory_gib"] > self.policy.memory_gib_per_trial_maximum:
            raise ControllerError("trial memory use exceeds the policy")
        if numeric_fields["wall_hours"] > self.policy.wall_hours_per_trial_maximum:
            raise ControllerError("trial wall time exceeds the policy")
        available_core_hours = numeric_fields["cpu_cores"] * numeric_fields["wall_hours"]
        if numeric_fields["cpu_core_hours"] > available_core_hours + 1e-12:
            raise ControllerError("CPU-core-hour receipt exceeds cores times wall time")

    def run_development_trial(
        self,
        trial_id: str,
        evaluation: Mapping[str, Any],
        *,
        interrupt_after_running: bool = False,
    ) -> Mapping[str, Any]:
        state = self.replay()
        proposed = state.proposed_trial_record(trial_id)
        latest = state.latest_trial_record(trial_id)
        if proposed is None:
            raise StateTransitionError("trial was not registered")
        evaluation_hash = digest_object(evaluation)
        prior_evaluation_hashes = {
            record["evaluation_hash"]
            for record in state.trials.get(trial_id, [])
            if "evaluation_hash" in record
        }
        if prior_evaluation_hashes and prior_evaluation_hashes != {evaluation_hash}:
            raise StateTransitionError("trial retry changed the evaluator payload")
        if latest and latest["state"] in {
            "SCORED",
            "RETIRED",
            "CHALLENGER",
            "INCUMBENT",
            "INVALID",
            "ENGINEERING_FAILED",
        }:
            scored = state.scored_trial_record(trial_id)
            return scored or latest
        self._assert_development_mutable()
        if latest and latest["state"] == "PROPOSED":
            running_payload = self._transition_payload(
                proposed,
                state="RUNNING",
                disposition="synthetic evaluator started",
                extra={"evaluation_hash": evaluation_hash},
            )
            self._append(f"trial:{trial_id}:running", running_payload)
        if interrupt_after_running:
            raise SimulatedInterruption(f"interrupted after RUNNING for {trial_id}")

        try:
            self._validate_evaluation_access(evaluation)
        except RoleLeakageError as exc:
            self._record_integrity_failure(trial_id, f"role leakage: {exc}")
            raise
        resources = dict(evaluation.get("resources") or {})
        resources.setdefault("cpu_core_hours", 0.0)
        resources.setdefault("gpu_hours", 0.0)
        try:
            self._validate_trial_resources(resources)
        except ControllerError as exc:
            self._record_integrity_failure(trial_id, f"resource integrity failure: {exc}")
            raise
        models = evaluation.get("models_completed")
        if tuple(models or ()) != self.policy.model_triplet:
            payload = self._transition_payload(
                proposed,
                state="INVALID",
                metrics=None,
                controls=evaluation.get("controls"),
                resources=resources,
                disposition="incomplete or reordered T/U/M triplet",
                extra={"evaluation_hash": evaluation_hash},
            )
            return self._append(f"trial:{trial_id}:invalid", payload)
        metrics = evaluation.get("metrics")
        if not isinstance(metrics, Mapping) or "selection_statistic" not in metrics:
            payload = self._transition_payload(
                proposed,
                state="INVALID",
                resources=resources,
                disposition="missing predeclared development metrics",
                extra={"evaluation_hash": evaluation_hash},
            )
            return self._append(f"trial:{trial_id}:invalid", payload)
        try:
            selection_statistic = float(metrics["selection_statistic"])
        except (TypeError, ValueError):
            selection_statistic = math.nan
        if not math.isfinite(selection_statistic):
            payload = self._transition_payload(
                proposed,
                state="INVALID",
                metrics=None,
                controls=evaluation.get("controls"),
                resources=resources,
                disposition="selection statistic is nonnumeric or nonfinite",
                extra={"evaluation_hash": evaluation_hash},
            )
            return self._append(f"trial:{trial_id}:invalid", payload)
        simulated_gates = evaluation.get("simulated_pre_null_gate_outcomes")
        if not isinstance(simulated_gates, Mapping):
            failed_gates = list(self.PRE_NULL_COMPARISON_GATES)
        else:
            failed_gates = [
                gate
                for gate in self.PRE_NULL_COMPARISON_GATES
                if simulated_gates.get(gate) != "simulated_pass_for_state_machine_only"
            ]
        if proposed["stage"] == "falsification":
            controls = evaluation.get("controls")
            if not isinstance(controls, Mapping) or controls.get("falsifier_id") != proposed["falsifier"]:
                failed_gates.append("registered_falsifier_result")
        if failed_gates:
            payload = self._transition_payload(
                proposed,
                state="INVALID",
                metrics=metrics,
                controls={
                    **dict(evaluation.get("controls") or {}),
                    "failed_simulated_pre_null_gates": sorted(set(failed_gates)),
                    "scientific_pre_null_gates_passed": False,
                },
                resources=resources,
                disposition="pre-null comparison validity gates failed",
                extra={"evaluation_hash": evaluation_hash},
            )
            return self._append(f"trial:{trial_id}:invalid", payload)
        payload = self._transition_payload(
            proposed,
            state="SCORED",
            metrics=metrics,
            controls=evaluation.get("controls"),
            resources=resources,
            disposition="complete synthetic T/U/M comparison scored",
            extra={"evaluation_hash": evaluation_hash},
        )
        return self._append(f"trial:{trial_id}:scored", payload)

    def apply_falsifiers(self, trial_id: str) -> Mapping[str, Any] | None:
        scored = self.replay().scored_trial_record(trial_id)
        if scored is None:
            raise StateTransitionError("falsifiers require a scored trial")
        return scored.get("controls")

    def score_trial(self, trial_id: str) -> Mapping[str, Any]:
        scored = self.replay().scored_trial_record(trial_id)
        if scored is None:
            raise StateTransitionError("trial is not scored")
        return scored["metrics"]

    def update_archive(
        self,
        trial_id: str,
        disposition: str,
        *,
        rationale: str,
        decision: bool,
    ) -> Mapping[str, Any]:
        self._assert_development_mutable(allow_active_stop=True)
        normalized = disposition.upper()
        if normalized not in {"RETIRED", "CHALLENGER", "INCUMBENT"}:
            raise ValueError("invalid archive disposition")
        state = self.replay()
        latest = state.latest_trial_record(trial_id)
        proposed = state.proposed_trial_record(trial_id)
        scored = state.scored_trial_record(trial_id)
        if latest and latest["state"] in {"RETIRED", "CHALLENGER", "INCUMBENT"}:
            expected_decision = {
                "is_decision": bool(decision),
                "result": normalized.lower(),
                "rationale": rationale,
            }
            if (
                latest["state"] != normalized
                or latest["disposition"] != rationale
                or latest.get("incumbent_decision") != expected_decision
            ):
                raise StateTransitionError("archive retry changed the frozen disposition")
            return latest
        if proposed is None or scored is None or latest != scored:
            raise StateTransitionError("archive update requires a newly scored trial")
        incumbent_decision = {
            "is_decision": bool(decision),
            "result": normalized.lower(),
            "rationale": rationale,
        }
        payload = self._transition_payload(
            proposed,
            state=normalized,
            metrics=scored["metrics"],
            controls=scored["controls"],
            resources=scored["resources"],
            disposition=rationale,
            incumbent_decision=incumbent_decision,
        )
        return self._append(f"trial:{trial_id}:archive", payload)

    def coverage_status(self, state: ReplayState | None = None) -> dict[str, Any]:
        state = state or self.replay()
        coverage_records = [
            state.scored_trial_record(trial_id)
            for trial_id in state.valid_trial_ids
            if state.proposed_trial_record(trial_id)["stage"] == "coverage"
        ]
        coverage_records = [record for record in coverage_records if record is not None]
        observed: set[tuple[str, Any, str, Any]] = set()
        for record in coverage_records:
            config = record["config"]
            for left_index, left in enumerate(self.COVERAGE_FACTORS):
                for right in self.COVERAGE_FACTORS[left_index + 1 :]:
                    observed.add((left, config[left], right, config[right]))
        expected: set[tuple[str, Any, str, Any]] = set()
        for left_index, left in enumerate(self.COVERAGE_FACTORS):
            for right in self.COVERAGE_FACTORS[left_index + 1 :]:
                for left_value, right_value in itertools.product(
                    self.policy.grammar_choices[left],
                    self.policy.grammar_choices[right],
                ):
                    expected.add((left, left_value, right, right_value))
        missing = expected - observed
        complete = len(coverage_records) >= self.policy.coverage_minimum and not missing
        return {
            "complete": complete,
            "valid_coverage_trials": len(coverage_records),
            "covered_pair_count": len(observed),
            "required_pair_count": len(expected),
            "missing_pair_count": len(missing),
        }

    def _attempted_coverage_complete(self, state: ReplayState) -> bool:
        proposed = [
            records[0]
            for records in state.trials.values()
            if records[0]["stage"] == "coverage"
        ]
        if len(proposed) < self.policy.coverage_minimum:
            return False
        observed: set[tuple[str, Any, str, Any]] = set()
        expected: set[tuple[str, Any, str, Any]] = set()
        for record in proposed:
            config = record["config"]
            for left_index, left in enumerate(self.COVERAGE_FACTORS):
                for right in self.COVERAGE_FACTORS[left_index + 1 :]:
                    observed.add((left, config[left], right, config[right]))
        for left_index, left in enumerate(self.COVERAGE_FACTORS):
            for right in self.COVERAGE_FACTORS[left_index + 1 :]:
                for left_value, right_value in itertools.product(
                    self.policy.grammar_choices[left], self.policy.grammar_choices[right]
                ):
                    expected.add((left, left_value, right, right_value))
        return expected.issubset(observed)

    def _adaptive_cycle_count(self, state: ReplayState) -> int:
        return len(
            {
                record["successor_cycle_id"]
                for trial_id in state.valid_trial_ids
                for record in [state.proposed_trial_record(trial_id)]
                if record["stage"] == "adaptive" and record["successor_cycle_id"]
            }
        )

    def _decision_count(self, state: ReplayState) -> int:
        return sum(
            bool(record.get("incumbent_decision", {}).get("is_decision"))
            for record in state.disposition_records
            if record.get("incumbent_decision")
        )

    def _falsifier_fraction(self, state: ReplayState) -> float:
        ordered_ids = list(state.valid_trial_ids)
        coverage_boundary: int | None = None
        for index in range(len(ordered_ids)):
            prefix_ids = ordered_ids[: index + 1]
            coverage_records = [
                state.scored_trial_record(trial_id)
                for trial_id in prefix_ids
                if state.proposed_trial_record(trial_id)["stage"] == "coverage"
            ]
            coverage_records = [record for record in coverage_records if record is not None]
            if len(coverage_records) < self.policy.coverage_minimum:
                continue
            observed: set[tuple[str, Any, str, Any]] = set()
            expected: set[tuple[str, Any, str, Any]] = set()
            for record in coverage_records:
                config = record["config"]
                for left_index, left in enumerate(self.COVERAGE_FACTORS):
                    for right in self.COVERAGE_FACTORS[left_index + 1 :]:
                        observed.add((left, config[left], right, config[right]))
            for left_index, left in enumerate(self.COVERAGE_FACTORS):
                for right in self.COVERAGE_FACTORS[left_index + 1 :]:
                    for left_value, right_value in itertools.product(
                        self.policy.grammar_choices[left],
                        self.policy.grammar_choices[right],
                    ):
                        expected.add((left, left_value, right, right_value))
            if expected.issubset(observed):
                coverage_boundary = index
                break
        if coverage_boundary is None:
            return 0.0
        post_coverage_ids = ordered_ids[coverage_boundary + 1 :]
        if not post_coverage_ids:
            return 0.0
        falsifiers = sum(
            state.proposed_trial_record(trial_id)["stage"] == "falsification"
            for trial_id in post_coverage_ids
        )
        return falsifiers / len(post_coverage_ids)

    def _required_falsifier_status(self, state: ReplayState) -> dict[str, Any]:
        observed = {
            proposed["falsifier"]
            for trial_id in state.valid_trial_ids
            for proposed in [state.proposed_trial_record(trial_id)]
            if proposed["stage"] == "falsification"
            and state.scored_trial_record(trial_id)["controls"].get("falsifier_id")
            == proposed["falsifier"]
        }
        required = set(self.policy.required_falsifiers)
        return {
            "complete": required.issubset(observed),
            "observed": sorted(observed),
            "missing": sorted(required - observed),
        }

    def _trailing_non_improvements(self, state: ReplayState) -> int:
        ordered = [
            state.scored_trial_record(trial_id)
            for trial_id in state.valid_trial_ids
        ]
        ordered = [record for record in ordered if record is not None]
        if not ordered:
            return 0
        values = [float(record["metrics"]["selection_statistic"]) for record in ordered]
        best_so_far = -math.inf
        last_improvement_index = -1
        for index, value in enumerate(values):
            if value > best_so_far:
                best_so_far = value
                last_improvement_index = index
        return len(values) - last_improvement_index - 1

    def _resource_usage(self, state: ReplayState) -> float:
        return sum(
            float(records[-1].get("resources", {}).get("cpu_core_hours", 0.0))
            for records in state.trials.values()
            if records[-1]["state"]
            in {"INVALID", "ENGINEERING_FAILED", "SCORED", "RETIRED", "CHALLENGER", "INCUMBENT"}
        )

    def _wall_usage(self, state: ReplayState) -> float:
        return sum(
            float(records[-1].get("resources", {}).get("wall_hours", 0.0))
            for records in state.trials.values()
            if records[-1]["state"]
            in {"INVALID", "ENGINEERING_FAILED", "SCORED", "RETIRED", "CHALLENGER", "INCUMBENT"}
        )

    def check_stop(self) -> tuple[str, str | None]:
        state = self.replay()
        if state.input_failure_record:
            return "stop", "technical_failure"
        if state.valid_trial_count >= self.policy.maximum_valid_trials:
            return "stop", "valid_trials_max_reached"
        if self._resource_usage(state) >= self.policy.cpu_core_hour_ceiling:
            return "stop", "cpu_budget_exhausted"
        if self._wall_usage(state) >= float(
            self.policy.raw["budgets"]["wall_clock_hour_ceiling"]
        ):
            return "stop", "wall_clock_budget_exhausted"
        if (
            state.valid_trial_count >= self.policy.patience_active_after_trial
            and self.coverage_status(state)["complete"]
            and self._trailing_non_improvements(state) >= self.policy.patience_valid_trials
        ):
            return "stop", "patience_exhausted"
        return "continue", None

    def record_stop(self, reason: str) -> Mapping[str, Any]:
        state = self.replay()
        if state.stop_reason:
            if state.stop_reason != reason:
                raise StateTransitionError("stop reason is already frozen")
            return next(record for record in state.records if record.get("record_kind") == "stop")
        self._assert_development_mutable(allow_active_stop=True)
        computed_action, computed_reason = self.check_stop()
        allowed_manual = {"admissible_configuration_space_exhausted", "no_executable_branch"}
        if not (computed_action == "stop" and computed_reason == reason) and reason not in allowed_manual:
            raise StateTransitionError(f"stop reason {reason!r} is not currently active")
        if reason == "no_executable_branch":
            attempts = [
                records for records in state.trials.values() if records[0]["stage"] == "coverage"
            ]
            terminal_attempts = {
                "READINESS_REJECTED",
                "INVALID",
                "ENGINEERING_FAILED",
            }
            if (
                len(attempts) < self.policy.coverage_minimum
                or any(records[-1]["state"] not in terminal_attempts for records in attempts)
                or state.valid_trial_count != 0
                or not self._attempted_coverage_complete(state)
            ):
                raise StateTransitionError(
                    "no-executable-branch requires complete logged coverage attempts"
                )
        if reason == "admissible_configuration_space_exhausted" and (
            state.valid_trial_count < self.policy.minimum_valid_trials
            or not self.coverage_status(state)["complete"]
        ):
            raise StateTransitionError(
                "configuration-space exhaustion requires minimum valid search evidence"
            )
        payload = self._record_payload(
            record_kind="stop",
            trial_id="__stop__",
            proposal_mode="mandated_falsifier",
            stage="falsification",
            state="RETIRED",
            disposition=reason,
            extra={"valid_trial_count": state.valid_trial_count},
        )
        return self._append("operation:stop", payload)

    def run_full_search_null(
        self,
        generator_lock_hash: str,
        replicate_manifest: Sequence[Mapping[str, Any]],
        *,
        selected_trial_id: str,
        observed_statistic: float,
        selection_program_hash: str,
    ) -> Mapping[str, Any]:
        state = self.replay()
        receipts_hash = digest_object(list(replicate_manifest))
        if state.null_record is not None:
            existing = state.null_record
            same_request = (
                existing["metrics"]["generator_lock_hash"] == generator_lock_hash
                and existing["metrics"]["selected_trial_id"] == selected_trial_id
                and existing["metrics"]["selection_program_hash"] == selection_program_hash
                and float(existing["metrics"]["observed_statistic"])
                == float(observed_statistic)
                and existing["replicate_receipts_hash"] == receipts_hash
            )
            if not same_request:
                raise StateTransitionError("full-search-null is already committed differently")
            return existing
        self._assert_development_mutable(allow_stopped=True)
        if state.stop_reason is None:
            raise StateTransitionError("observed search must stop before null calibration")
        if state.valid_trial_count == 0:
            raise StateTransitionError("null calibration needs a complete comparison")
        selected_trial = state.scored_trial_record(selected_trial_id)
        if selected_trial is None:
            raise StateTransitionError("null calibration must bind a scored comparison")
        selected_statistic = float(selected_trial["metrics"]["selection_statistic"])
        if not math.isfinite(observed_statistic) or observed_statistic != selected_statistic:
            raise StateTransitionError(
                "observed null statistic must equal the frozen selected comparison"
            )
        if len(replicate_manifest) != self.policy.null_replicates:
            raise StateTransitionError(
                f"expected {self.policy.null_replicates} complete null searches"
            )
        seen_indices: set[int] = set()
        seen_seeds: set[int] = set()
        null_statistics: list[float] = []
        null_fit_count = 0
        for receipt in replicate_manifest:
            index = int(receipt.get("replicate_index", -1))
            if index in seen_indices or index < 0:
                raise StateTransitionError("duplicate or invalid null replicate index")
            seen_indices.add(index)
            seed = receipt.get("seed")
            if not isinstance(seed, int) or seed in seen_seeds:
                raise StateTransitionError("null seeds must be present and unique integers")
            seen_seeds.add(seed)
            required_truths = (
                receipt.get("initial_ledger_empty") is True,
                receipt.get("complete_controller_rerun") is False,
                receipt.get("simulated_complete_controller_rerun") is True,
                receipt.get("nested_null_searches") == 0,
                receipt.get("deterministic_replay_passed") is False,
                receipt.get("simulated_deterministic_replay") is True,
                receipt.get("policy_hash") == self.policy.policy_hash,
                receipt.get("selection_program_hash") == selection_program_hash,
                receipt.get("receipt_scope") == "asserted_synthetic_not_executed",
            )
            if not all(required_truths):
                raise StateTransitionError(f"null replicate {index} is incomplete or mismatched")
            valid_trials = int(receipt.get("valid_trials", 0))
            model_fits = int(receipt.get("model_family_fits", 0))
            if valid_trials < self.policy.minimum_valid_trials or model_fits != valid_trials * 3:
                raise StateTransitionError(f"null replicate {index} did not rerun full T/U/M search")
            null_fit_count += model_fits
            statistic = float(receipt["selected_statistic"])
            if not math.isfinite(statistic):
                raise StateTransitionError(f"null replicate {index} has a nonfinite statistic")
            null_statistics.append(statistic)
            trajectory = {
                "replicate_index": index,
                "seed": seed,
                "valid_trials": valid_trials,
                "selected_statistic": statistic,
                "selection_program_hash": selection_program_hash,
            }
            if receipt.get("trajectory_hash") != digest_object(trajectory):
                raise StateTransitionError(f"null replicate {index} trajectory hash mismatch")
        expected_indices = set(range(self.policy.null_replicates))
        if seen_indices != expected_indices:
            raise StateTransitionError("null replicate indices are not the frozen complete set")
        exceedances = sum(value >= observed_statistic for value in null_statistics)
        monte_carlo_p = (1 + exceedances) / (self.policy.null_replicates + 1)
        total_model_fits = state.valid_trial_count * 3 + null_fit_count
        minimum_protocol_fits = (
            self.policy.minimum_valid_trials * 3 * (self.policy.null_replicates + 1)
        )
        if total_model_fits < minimum_protocol_fits:
            raise StateTransitionError("null program is below declared protocol scale")
        metrics = {
            "generator_lock_hash": generator_lock_hash,
            "selected_trial_id": selected_trial_id,
            "selection_program_hash": selection_program_hash,
            "observed_statistic": observed_statistic,
            "null_statistics": null_statistics,
            "exceedances": exceedances,
            "monte_carlo_p": monte_carlo_p,
            "replicates": len(replicate_manifest),
            "protocol_scale_model_family_fit_count": total_model_fits,
            "minimum_protocol_model_family_fits": minimum_protocol_fits,
            "count_is_measured_runtime": False,
            "qualification_scope": "synthetic_receipt_validation_only",
            "executed_full_search_null_completed": False,
        }
        payload = self._record_payload(
            record_kind="full_search_null",
            trial_id="__full_search_null__",
            proposal_mode="mandated_falsifier",
            stage="falsification",
            state="SCORED",
            metrics=metrics,
            controls={
                "pre_outcome_cost_profile_passed": False,
                "synthetic_protocol_arithmetic_passed": True,
            },
            disposition="99 synthetic null receipts validated; reruns not executed",
            extra={"replicate_receipts_hash": receipts_hash},
        )
        return self._append("operation:full-search-null", payload)

    def _search_gates(self, state: ReplayState) -> dict[str, bool]:
        return {
            "minimum_valid_trials": state.valid_trial_count >= self.policy.minimum_valid_trials,
            "branch_coverage": self.coverage_status(state)["complete"],
            "adaptive_cycles": self._adaptive_cycle_count(state)
            >= self.policy.minimum_adaptive_cycles,
            "incumbent_challenger_decisions": self._decision_count(state)
            >= self.policy.minimum_incumbent_decisions,
            "post_coverage_falsifier_fraction": self._falsifier_fraction(state)
            >= self.policy.falsifier_fraction_minimum,
            "required_falsifier_ids_exercised": self._required_falsifier_status(state)[
                "complete"
            ],
            "synthetic_null_receipts_validated": state.null_record is not None,
            "journal_integrity": bool(self.journal.verify()),
            "stop_recorded": state.stop_reason is not None,
        }

    def lock_configuration(
        self,
        incumbent_id: str,
        *,
        synthetic_selection_rule_id: str,
        final_output_allowlist: Sequence[str],
    ) -> str:
        state = self.replay()
        if state.lock_record is not None:
            manifest = state.lock_record["lock_manifest"]
            same_request = (
                manifest["selected_trial_id"] == incumbent_id
                and manifest["synthetic_selection_rule_id"] == synthetic_selection_rule_id
                and manifest["final_output_allowlist"] == list(final_output_allowlist)
            )
            if not same_request:
                raise StateTransitionError("configuration is already locked to another request")
            return state.lock_record["configuration_lock_hash"]
        self._assert_development_mutable(allow_stopped=True)
        gates = self._search_gates(state)
        failed = sorted(key for key, passed in gates.items() if not passed)
        if failed:
            raise StateTransitionError(f"cannot lock; failed search gates: {', '.join(failed)}")
        selected = state.scored_trial_record(incumbent_id)
        if selected is None:
            raise StateTransitionError("selected comparison is not a valid scored triplet")
        if state.null_record["metrics"]["selected_trial_id"] != incumbent_id:
            raise StateTransitionError("lock selection differs from the null-calibrated comparison")
        latest_selected = state.latest_trial_record(incumbent_id)
        if latest_selected is None or latest_selected["state"] != "INCUMBENT":
            raise StateTransitionError("selected comparison is not the archived incumbent")
        if synthetic_selection_rule_id != "synthetic_qualification_only":
            raise StateTransitionError("real scientific selection rule remains intentionally unset")
        if tuple(final_output_allowlist) != self.SYNTHETIC_FINAL_OUTPUT_FIELDS:
            raise StateTransitionError(
                "synthetic final output allowlist must exactly match the qualified schema"
            )
        manifest = {
            "episode_id": self.policy.episode_id,
            "mode": "synthetic_qualification_only",
            "policy_hash": self.policy.policy_hash,
            "code_hash": self.code_hash,
            "environment_hash": self.environment_hash,
            "source_hash": self.source_hash,
            "split_hash": self.split_hash,
            "selected_trial_id": incumbent_id,
            "selected_config_hash": selected["hashes"]["config"],
            "synthetic_selection_rule_id": synthetic_selection_rule_id,
            "meaningful_margin": self.policy.meaningful_margin,
            "null_record_hash": state.null_record["record_hash"],
            "final_output_allowlist": list(final_output_allowlist),
            "ledger_prefix_hash": self.journal.prefix_hash(),
            "retry_rule": "same_operation_id_reconcile_only",
            "real_data_enabled": False,
            "scientific_model_qualification": False,
            "executed_full_search_null_completed": False,
            "scientific_interpretation_permitted": False,
        }
        lock_hash = digest_object(manifest)
        payload = self._record_payload(
            record_kind="configuration_lock",
            trial_id="__configuration_lock__",
            proposal_mode="lock",
            stage="lock",
            state="LOCKED",
            config=selected["config"],
            configuration_lock_hash=lock_hash,
            disposition="synthetic configuration locked",
            extra={"lock_manifest": manifest, "search_gates": gates},
        )
        self._append("operation:configuration-lock", payload)
        return lock_hash

    def _final_result_errors(
        self,
        result: Mapping[str, Any],
        declared_fields: Sequence[str] | None = None,
    ) -> list[str]:
        errors: list[str] = []
        declared = tuple(declared_fields or self.SYNTHETIC_FINAL_OUTPUT_FIELDS)
        if set(result) != set(declared):
            missing = sorted(set(declared) - set(result))
            extra = sorted(set(result) - set(declared))
            if missing:
                errors.append("missing final fields: " + ", ".join(missing))
            if extra:
                errors.append("undeclared final fields: " + ", ".join(extra))

        required_directions = ("left_to_right", "right_to_left")
        directions = result.get("directions")
        if not isinstance(directions, Mapping) or set(directions) != set(required_directions):
            errors.append("aggregate reciprocal directions are incomplete")
        else:
            for direction in required_directions:
                summary = directions[direction]
                if not isinstance(summary, Mapping) or set(summary) != {
                    "gain",
                    "simultaneous_interval",
                }:
                    errors.append(f"malformed aggregate direction: {direction}")
                    continue
                interval = summary.get("simultaneous_interval")
                try:
                    gain = float(summary.get("gain"))
                    lower, upper = map(float, interval)
                except (TypeError, ValueError):
                    errors.append(f"nonnumeric aggregate direction: {direction}")
                    continue
                if not all(math.isfinite(value) for value in (gain, lower, upper)):
                    errors.append(f"nonfinite aggregate direction: {direction}")
                elif not lower <= gain <= upper:
                    errors.append(f"inconsistent aggregate interval: {direction}")

        expected_types = list(self.role_manifest.final_types)
        reported_types = result.get("reported_final_types")
        if not isinstance(reported_types, list) or reported_types != expected_types:
            errors.append("reported final types do not exactly match the frozen manifest")
        if result.get("every_final_type_reported") is not True:
            errors.append("every_final_type_reported must be explicitly true")
        if result.get("target_side_refit") is not False:
            errors.append("target_side_refit must be explicitly false")
        if result.get("source_side_fit_only") is not True:
            errors.append("source_side_fit_only must be explicitly true")

        per_type = result.get("per_type_results")
        if not isinstance(per_type, list) or len(per_type) != len(expected_types):
            errors.append("per-type final accounting has the wrong length")
        else:
            observed_types: list[Any] = []
            for item in per_type:
                if not isinstance(item, Mapping) or set(item) != {"type", "directions"}:
                    errors.append("malformed per-type final record")
                    continue
                observed_types.append(item["type"])
                item_directions = item["directions"]
                if not isinstance(item_directions, Mapping) or set(item_directions) != set(
                    required_directions
                ):
                    errors.append(f"missing reciprocal direction for type {item['type']}")
                    continue
                for direction in required_directions:
                    direction_record = item_directions[direction]
                    if (
                        not isinstance(direction_record, Mapping)
                        or set(direction_record) != {"status"}
                        or direction_record.get("status") != "scored"
                    ):
                        errors.append(
                            f"invalid synthetic accounting for {item['type']} {direction}"
                        )
            if observed_types != expected_types:
                errors.append("per-type records do not exactly match the frozen type order")

        support = result.get("residual_support")
        if not isinstance(support, Mapping) or set(support) != set(self.RESIDUAL_SUPPORT_KEYS):
            errors.append("residual-support result schema is incomplete")
        elif any(not isinstance(support[key], bool) for key in self.RESIDUAL_SUPPORT_KEYS):
            errors.append("residual-support results must be booleans")
        for name in (
            "valid_final",
            "comparison_valid",
            "reference_calibrated",
            "adequacy_sensitivity_qualified",
            "followup_started",
            "paper_level_claim_emitted",
        ):
            if not isinstance(result.get(name), bool):
                errors.append(f"{name} must be an explicit boolean")
        if result.get("followup_started") is not False:
            errors.append("primary executor may not start the follow-up")
        if result.get("paper_level_claim_emitted") is not False:
            errors.append("synthetic final may not emit a paper-level claim")
        return errors

    def _decide_terminal(
        self, result: Mapping[str, Any], null_p: float
    ) -> tuple[str, str]:
        if (
            not math.isfinite(float(null_p))
            or not 0.0 <= float(null_p) <= 1.0
            or self._final_result_errors(result)
            or result.get("valid_final") is not True
            or result.get("comparison_valid") is not True
        ):
            detailed = "technical_failure"
            return detailed, self.policy.terminal_mapping[detailed]
        directions = result["directions"]
        required_directions = ("left_to_right", "right_to_left")

        lower_bounds_above_margin = all(
            float(directions[direction]["simultaneous_interval"][0])
            > self.policy.meaningful_margin
            for direction in required_directions
        )
        support = result["residual_support"]
        residual_support_passes = all(support[key] for key in self.RESIDUAL_SUPPORT_KEYS)
        if (
            lower_bounds_above_margin
            and residual_support_passes
            and null_p <= self.policy.null_pass_threshold
        ):
            detailed = "candidate_ready_residual_modes"
            return detailed, self.policy.terminal_mapping[detailed]

        upper_bounds_below_margin = all(
            float(directions[direction]["simultaneous_interval"][1])
            < self.policy.meaningful_margin
            for direction in required_directions
        )
        if (
            upper_bounds_below_margin
            and result["adequacy_sensitivity_qualified"] is True
            and result["reference_calibrated"] is True
        ):
            detailed = "closed_single_population_adequate_within_margin"
            return detailed, self.policy.terminal_mapping[detailed]
        detailed = "closed_unresolved"
        return detailed, self.policy.terminal_mapping[detailed]

    def run_audit(
        self,
        lock_hash: str,
        evaluator: AuditEvaluator,
        *,
        operation_id: str,
        simulate_lost_acknowledgement: bool = False,
        simulate_interruption_after_evaluator_commit: bool = False,
    ) -> Mapping[str, Any]:
        state = self.replay()
        if state.lock_record is None:
            raise AuditError("configuration must be locked before final evaluation")
        frozen_lock = state.lock_record["configuration_lock_hash"]
        if lock_hash != frozen_lock:
            raise AuditError("configuration lock hash mismatch")
        if state.terminal_record:
            terminal_operation = state.terminal_record.get("operation_id")
            if terminal_operation != operation_id:
                raise AuditError("a different final operation is already terminal")
            return state.terminal_record["terminal"]
        if state.audit_open_record:
            prior_operation = state.audit_open_record["audit_receipt"]["operation_id"]
            if prior_operation != operation_id:
                raise AuditError("the single final opening is already reserved")
        else:
            if state.audit_open_count >= self.policy.maximum_open_count:
                raise AuditError("maximum final opening count reached")
            open_receipt = {
                "operation_id": operation_id,
                "lock_hash": lock_hash,
                "open_index": 1,
                "opening_consumed": True,
                "outcome_emitted_at_open_record": False,
            }
            payload = self._record_payload(
                record_kind="audit_open",
                trial_id="__audit__",
                proposal_mode="audit",
                stage="audit",
                state="AUDIT_OPENED",
                configuration_lock_hash=lock_hash,
                disposition="single synthetic final opening consumed",
                audit_receipt=open_receipt,
                extra={"operation_id": operation_id},
            )
            self._append("operation:audit-open", payload)

        state = self.replay()
        if state.audit_result_record is None:
            declared_fields = tuple(state.lock_record["lock_manifest"]["final_output_allowlist"])
            validation_errors: list[str] = []
            raw_result: Mapping[str, Any] | None = None
            try:
                reconciled = evaluator.reconcile(operation_id, lock_hash)
                candidate = (
                    reconciled
                    if reconciled is not None
                    else evaluator.evaluate(operation_id, lock_hash)
                )
                if simulate_interruption_after_evaluator_commit:
                    raise SimulatedInterruption(
                        "synthetic evaluator committed before controller result journal"
                    )
                if not isinstance(candidate, Mapping):
                    validation_errors.append("evaluator did not return a result mapping")
                else:
                    raw_result = candidate
                    validation_errors.extend(
                        self._final_result_errors(candidate, declared_fields)
                    )
            except SimulatedInterruption:
                raise
            except Exception as exc:  # trusted evaluator failure becomes durable technical failure
                validation_errors.append(
                    f"evaluator failure after opening: {type(exc).__name__}: {exc}"
                )

            safe_result = {
                field: raw_result[field]
                for field in declared_fields
                if raw_result is not None and field in raw_result
            }
            try:
                raw_result_hash = digest_object(
                    raw_result if raw_result is not None else validation_errors
                )
                digest_object(safe_result)
            except Exception as exc:  # serialization must never strand a consumed opening
                validation_errors.append(
                    f"evaluator result is not canonically serializable: {type(exc).__name__}"
                )
                raw_result_hash = digest_object(
                    {
                        "serialization_failure": type(exc).__name__,
                        "operation_id": operation_id,
                    }
                )
                safe_result = {}
            null_p = float(state.null_record["metrics"]["monte_carlo_p"])
            if validation_errors:
                detailed = "technical_failure"
                outer = self.policy.terminal_mapping[detailed]
            else:
                detailed, outer = self._decide_terminal(safe_result, null_p)
            result_receipt = {
                "operation_id": operation_id,
                "lock_hash": lock_hash,
                "opening_consumed": True,
                "result_hash": raw_result_hash,
                "result": safe_result,
                "result_validation_errors": validation_errors,
                "detailed_outcome": detailed,
                "outer_status": outer,
            }
            payload = self._record_payload(
                record_kind="audit_result",
                trial_id="__audit__",
                proposal_mode="audit",
                stage="audit",
                state="AUDITED",
                configuration_lock_hash=lock_hash,
                metrics={"null_p": null_p},
                disposition="synthetic final result committed",
                audit_receipt=result_receipt,
                extra={"operation_id": operation_id},
            )
            self._append("operation:audit-result", payload)
            state = self.replay()

        audit_receipt = state.audit_result_record["audit_receipt"]
        terminal = {
            "detailed_outcome": audit_receipt["detailed_outcome"],
            "outer_status": audit_receipt["outer_status"],
            "final_evaluation_completed": audit_receipt["detailed_outcome"]
            != "technical_failure",
            "final_evaluation_valid": audit_receipt["detailed_outcome"]
            != "technical_failure",
            "final_open_count": state.audit_open_count,
            "result_hash": audit_receipt["result_hash"],
            "real_connectivity_access_count": 0,
            "qualification_scope": "synthetic_control_plane_only",
            "scientific_interpretation_permitted": False,
            "real_data_execution_enabled": False,
            "scientific_model_qualification": False,
            "executed_full_search_null_completed": False,
        }
        terminal_payload = self._record_payload(
            record_kind="terminal",
            trial_id="__terminal__",
            proposal_mode="audit",
            stage="audit",
            state="AUDITED",
            configuration_lock_hash=lock_hash,
            disposition=terminal["detailed_outcome"],
            audit_receipt=audit_receipt,
            extra={"terminal": terminal, "operation_id": operation_id},
        )
        self._append("operation:terminal", terminal_payload)
        if simulate_lost_acknowledgement:
            raise SimulatedInterruption("synthetic final result committed before acknowledgement")
        return terminal

    def close_without_audit(self, detailed_outcome: str, reason: str) -> Mapping[str, Any]:
        state = self.replay()
        allowed = {"closed_no_valid_comparison", "incomplete_search", "technical_failure"}
        if detailed_outcome not in allowed:
            raise StateTransitionError("this terminal requires a final evaluation")
        if state.audit_open_record:
            raise StateTransitionError("cannot use an unopened-final terminal after final access")
        if state.terminal_record:
            return state.terminal_record["terminal"]
        if detailed_outcome == "closed_no_valid_comparison" and state.valid_trial_count:
            raise StateTransitionError("valid scored comparisons exist")
        if detailed_outcome == "closed_no_valid_comparison" and (
            state.stop_reason != "no_executable_branch"
            or not self._attempted_coverage_complete(state)
        ):
            raise StateTransitionError(
                "no-valid-comparison requires completed, logged development attempts"
            )
        if detailed_outcome == "incomplete_search":
            gates = self._search_gates(state)
            if all(gates.values()):
                raise StateTransitionError("search is complete, not incomplete")
        terminal = {
            "detailed_outcome": detailed_outcome,
            "outer_status": self.policy.terminal_mapping[detailed_outcome],
            "final_evaluation_completed": False,
            "final_open_count": 0,
            "reason": reason,
            "real_connectivity_access_count": 0,
            "qualification_scope": "synthetic_control_plane_only",
            "scientific_interpretation_permitted": False,
            "real_data_execution_enabled": False,
            "scientific_model_qualification": False,
            "executed_full_search_null_completed": False,
        }
        terminal_state = "ENGINEERING_FAILED" if detailed_outcome == "technical_failure" else "RETIRED"
        payload = self._record_payload(
            record_kind="terminal",
            trial_id="__terminal__",
            proposal_mode="mandated_falsifier",
            stage="falsification",
            state=terminal_state,
            disposition=detailed_outcome,
            extra={"terminal": terminal, "operation_id": None},
        )
        self._append("operation:terminal", payload)
        return terminal

    def record_input_failure(self, reason: str) -> Mapping[str, Any]:
        state = self.replay()
        if state.records:
            raise StateTransitionError("input integrity must be established before trials")
        payload = self._record_payload(
            record_kind="input_failure",
            trial_id="__input__",
            proposal_mode="readiness_or_baseline",
            stage="readiness",
            state="ENGINEERING_FAILED",
            disposition=reason,
        )
        return self._append("operation:input-failure", payload)


def balanced_coverage_configs(policy: EpisodePolicy) -> list[dict[str, Any]]:
    """Return the frozen 18-row synthetic covering array.

    The construction covers every pair among three 3-level factors and one
    2-level factor.  It is an executor fixture, not the still-unfrozen real
    scientific covering array.
    """

    vocabularies = policy.grammar_choices["partner_vocabulary"]
    representations = policy.grammar_choices["representation"]
    covariances = policy.grammar_choices["covariance"]
    components = policy.grammar_choices["component_count"]
    if not (len(vocabularies) == len(representations) == len(covariances) == 3):
        raise ConfigurationError("synthetic covering construction expects three 3-level factors")
    if len(components) != 2:
        raise ConfigurationError("synthetic covering construction expects one 2-level factor")
    rows: list[dict[str, Any]] = []
    for vocabulary_index, representation_index, component_index in itertools.product(
        range(3), range(3), range(2)
    ):
        covariance_index = (vocabulary_index + representation_index + component_index) % 3
        config = {
            "models": list(policy.model_triplet),
            "partner_vocabulary": vocabularies[vocabulary_index],
            "partner_minimum_development_type_count": policy.grammar_choices[
                "partner_minimum_development_type_count"
            ][0],
            "rare_partner_pooling_mass_fraction": policy.grammar_choices[
                "rare_partner_pooling_mass_fraction"
            ][0],
            "partner_hierarchy_depth": policy.grammar_choices["partner_hierarchy_depth"][0],
            "representation": representations[representation_index],
            "rank": policy.grammar_choices["rank"][0],
            "composition_pseudocount": policy.grammar_choices["composition_pseudocount"][0],
            "nuisance": list(policy.nuisance_terms),
            "nuisance_spline_df": policy.grammar_choices["nuisance_spline_df"][0],
            "covariance": covariances[covariance_index],
            "covariance_rank": policy.grammar_choices["covariance_rank"][0],
            "component_count": components[component_index],
            "regularization": policy.grammar_choices["regularization"][0],
            "data_mode": "synthetic",
        }
        policy.validate_config(config)
        rows.append(config)
    return rows
