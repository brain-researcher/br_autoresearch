#!/usr/bin/env python3
"""Chronology-owning observed-controller session for the EP10 v3 M2 search.

The core executor authenticates individual executions.  That is deliberately
not enough to prove that a caller ran one prospective controller history: an
individual execution can be repeated and otherwise-valid objects can be
cherry-picked after the fact.  This module owns the missing serial state.

There is no generic production external-adapter registrar here.  The optional
family has one concrete named adapter with its own one-shot capability and
detailed execution receipt; other production external scopes remain closed
until their own named factories exist.  The explicitly named synthetic
qualification API exists only for target-free unit tests and rejects sealed
production caches.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import weakref
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


SCRIPT_PATH = Path(os.path.abspath(__file__))
TOOLS_DIR = SCRIPT_PATH.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import v3_m2_optional_diagnostic_v1 as diagnostic  # noqa: E402
import v3_m2_trial_executor_v1 as executor  # noqa: E402
import v3_search_controller_v1 as controller  # noqa: E402


RUNNING = "running"
ORDINARY_PATIENCE_STOP = "ordinary_patience_stop"
INCOMPLETE_BUDGET_STOP = "incomplete_budget_stop"
TECHNICAL_RETRY_STOP = "technical_stop_invalid_retry_ceiling"
TECHNICAL_RESOURCE_STOP = "technical_stop_resource"
TECHNICAL_INTEGRITY_STOP = "technical_stop_integrity"
ORDINARY_CONCLUSION = "ordinary_search_complete"
INCOMPLETE_CONCLUSION = "incomplete_not_success_not_negative"
OPTIONAL_FAMILIES = frozenset({"low_rank", "sparse_plus_low_rank"})
RADEMACHER_REFERENCE_SEEDS = tuple(range(2026092801, 2026092808))
RADEMACHER_SPECIFICITY_MARGIN_BITS = 0.002

_SESSION_SEAL = object()
_ADAPTER_AUTHORITY_SEAL = object()
_TERMINAL_AUTHORIZATION_SEAL = object()
_SELECTED_REFIT_AUTHORIZATION_SEAL = object()
_SELECTED_REFIT_ISSUANCE_LOCK = threading.RLock()


class ObservedSessionError(RuntimeError):
    """The observed controller session failed a chronology or authority check."""


class ObservedResourceStop(RuntimeError):
    """A trusted adapter reports a resource stop, which can never select a refit."""


def _make_weak_identity_registry(label: str) -> tuple[Any, Any]:
    records: dict[int, tuple[weakref.ReferenceType[Any], Any]] = {}

    def register(instance: Any, metadata: Any) -> None:
        identity = id(instance)

        def remove(reference: weakref.ReferenceType[Any]) -> None:
            current = records.get(identity)
            if current is not None and current[0] is reference:
                records.pop(identity, None)

        records[identity] = (weakref.ref(instance, remove), metadata)

    def lookup(instance: Any) -> Any:
        record = records.get(id(instance))
        if record is None or record[0]() is not instance:
            raise ObservedSessionError(f"{label} was not issued by this module")
        return record[1]

    return register, lookup


_register_session, _lookup_session = _make_weak_identity_registry(
    "controller run session"
)
_register_adapter, _lookup_adapter = _make_weak_identity_registry(
    "adapter authority"
)
_register_terminal, _lookup_terminal = _make_weak_identity_registry(
    "terminal selection authorization"
)
_register_selected_refit, _lookup_selected_refit = _make_weak_identity_registry(
    "selected-refit procedure authorization"
)


@dataclass(frozen=True)
class SyntheticTrialOutcome:
    """Explicitly synthetic scored outcome used only by qualification tests."""

    result: controller.TrialResult
    equal_group_pair_contrasts: tuple[float, ...] = (0.0,) * controller.PAIR_COUNT

    def __post_init__(self) -> None:
        if type(self.result) is not controller.TrialResult:
            raise ObservedSessionError("synthetic outcome lacks an exact TrialResult")
        contrasts = self.equal_group_pair_contrasts
        if (
            type(contrasts) is not tuple
            or len(contrasts) != controller.PAIR_COUNT
            or any(type(value) is not float or not math.isfinite(value) for value in contrasts)
        ):
            raise ObservedSessionError(
                "synthetic outcome contrasts must be exactly 120 finite floats"
            )


class ControllerRunSession:
    """Opaque, once-only serial controller session."""

    __slots__ = ("_seal", "__weakref__")

    def __init__(self, *, _seal: object | None = None) -> None:
        if _seal is not _SESSION_SEAL:
            raise ObservedSessionError("sessions must be issued by a start function")
        object.__setattr__(self, "_seal", _seal)

    @property
    def next_trial(self) -> controller.Trial | None:
        state = _require_session(self, deep_cache=False)
        if state.status != RUNNING:
            return None
        return state.schedule[len(state.results)]

    @property
    def status(self) -> str:
        return _require_session(self, deep_cache=False).status

    @property
    def scientific_conclusion(self) -> str | None:
        return _require_session(self, deep_cache=False).scientific_conclusion

    @property
    def valid_trial_count(self) -> int:
        return len(_require_session(self, deep_cache=False).results)

    @property
    def invalid_attempt_count(self) -> int:
        return _require_session(self, deep_cache=False).invalid_attempts

    def run_next_core(self) -> controller.TrialResult:
        return _run_next_core(self)

    def run_next_external(self, authority: AdapterAuthority) -> controller.TrialResult:
        return _run_next_external(self, authority, required_mode="production")

    def run_next_optional(self, authority: AdapterAuthority) -> controller.TrialResult:
        """Run the exact next production optional-family attempt."""
        return _run_next_external(
            self,
            authority,
            required_mode="production",
            required_scope="optional",
        )

    def run_next_falsifier(self, authority: AdapterAuthority) -> controller.TrialResult:
        """Run the exact next named production falsifier attempt."""
        return _run_next_external(
            self,
            authority,
            required_mode="production",
            required_scope="falsifier",
        )

    def run_next_synthetic_qualification(
        self,
        authority: AdapterAuthority,
    ) -> controller.TrialResult:
        return _run_next_external(self, authority, required_mode="synthetic")

    def seal_resource_stop(self, reason: str) -> None:
        _seal_resource_stop(self, reason)

    def __copy__(self) -> None:
        raise ObservedSessionError("controller sessions cannot be copied")

    def __deepcopy__(self, memo: Any) -> None:
        del memo
        raise ObservedSessionError("controller sessions cannot be copied")

    def __reduce__(self) -> None:
        raise ObservedSessionError("controller sessions cannot be serialized")


class AdapterAuthority:
    """Opaque authority for one exact, session-bound external adapter."""

    __slots__ = ("_seal", "__weakref__")

    def __init__(self, *, _seal: object | None = None) -> None:
        if _seal is not _ADAPTER_AUTHORITY_SEAL:
            raise ObservedSessionError("adapter authorities require an approved issuer")
        object.__setattr__(self, "_seal", _seal)

    def __copy__(self) -> None:
        raise ObservedSessionError("adapter authorities cannot be copied")

    def __deepcopy__(self, memo: Any) -> None:
        del memo
        raise ObservedSessionError("adapter authorities cannot be copied")

    def __reduce__(self) -> None:
        raise ObservedSessionError("adapter authorities cannot be serialized")


class TerminalSelectionAuthorization:
    """Opaque authorization derived only from a sealed serial session."""

    __slots__ = ("_seal", "__weakref__")

    def __init__(self, *, _seal: object | None = None) -> None:
        if _seal is not _TERMINAL_AUTHORIZATION_SEAL:
            raise ObservedSessionError(
                "terminal selection authorizations require a sealed session"
            )
        object.__setattr__(self, "_seal", _seal)

    @property
    def disposition(self) -> str:
        snapshot, _ = _validated_terminal_snapshot(self)
        return snapshot.disposition

    @property
    def scientific_conclusion(self) -> str:
        snapshot, _ = _validated_terminal_snapshot(self)
        return snapshot.scientific_conclusion

    @property
    def winner_result(self) -> controller.TrialResult:
        _, winner = _validated_terminal_snapshot(self)
        return winner

    @property
    def selected_configuration(self) -> controller.Configuration:
        _, winner = _validated_terminal_snapshot(self)
        configuration = winner.trial.configuration
        if type(configuration) is not controller.Configuration:
            raise ObservedSessionError("terminal configuration no longer exists")
        return configuration

    @property
    def valid_trial_count(self) -> int:
        snapshot, _ = _validated_terminal_snapshot(self)
        return snapshot.valid_trial_count

    @property
    def invalid_attempt_count(self) -> int:
        snapshot, _ = _validated_terminal_snapshot(self)
        return snapshot.invalid_attempt_count

    def __copy__(self) -> None:
        raise ObservedSessionError("terminal authorizations cannot be copied")

    def __deepcopy__(self, memo: Any) -> None:
        del memo
        raise ObservedSessionError("terminal authorizations cannot be copied")

    def __reduce__(self) -> None:
        raise ObservedSessionError("terminal authorizations cannot be serialized")


class SelectedRefitProcedureAuthorization:
    """Opaque one-shot authority for terminal full-development completion."""

    __slots__ = ("_seal", "__weakref__")

    def __init__(self, *, _seal: object | None = None) -> None:
        if _seal is not _SELECTED_REFIT_AUTHORIZATION_SEAL:
            raise ObservedSessionError(
                "selected-refit authorities require terminal issuance"
            )
        object.__setattr__(self, "_seal", _seal)

    def __copy__(self) -> None:
        raise ObservedSessionError("selected-refit authorities cannot be copied")

    def __deepcopy__(self, memo: Any) -> None:
        del memo
        raise ObservedSessionError("selected-refit authorities cannot be copied")

    def __reduce__(self) -> None:
        raise ObservedSessionError(
            "selected-refit authorities cannot be serialized"
        )


@dataclass(frozen=True)
class _Attempt:
    ordinal: int
    trial: controller.Trial
    outcome: str
    entry_identity: int | None
    invalid_attempts_after: int
    authority_identity: int | None
    error_type: str | None
    execution_capability: object | None
    execution_capability_digest: bytes | None
    production_attempt_ordinal: int | None


@dataclass
class _SessionState:
    owner: executor.CoreTrialExecutor
    session_reference: weakref.ReferenceType[ControllerRunSession]
    cache_reference: weakref.ReferenceType[executor.PreparedTrialCache]
    maximum_iterations: int
    mode: str
    schedule: list[controller.Trial]
    schedule_digests: list[bytes]
    history: list[executor.TrialExecution | executor.ExternalExecutionReceipt]
    history_digests: list[bytes]
    results: list[controller.TrialResult]
    invalid_counts_at_results: list[int]
    attempts: list[_Attempt]
    attempt_digests: list[bytes]
    authority_bindings: dict[
        str,
        tuple[int, weakref.ReferenceType[AdapterAuthority]],
    ]
    production_claim: executor.ProductionCacheClaim | None = None
    production_attempts: int = 0
    optional_production_attempts: int = 0
    permutation_production_attempts: int = 0
    rademacher_production_attempts: int = 0
    invalid_attempts: int = 0
    coverage_evidence: controller.CoverageEvidence | None = None
    coverage_evidence_digest: bytes | None = None
    coverage_length: int | None = None
    cycle_one_configurations: tuple[controller.Configuration, ...] | None = None
    status: str = RUNNING
    scientific_conclusion: str | None = None
    version: int = 0
    in_flight: bool = False
    capability_issuance_open: bool = False
    optional_binding_open: bool = False
    optional_capability_issuance_open: bool = False
    permutation_binding_open: bool = False
    permutation_capability_issuance_open: bool = False
    rademacher_binding_open: bool = False
    rademacher_capability_issuance_open: bool = False
    synthetic_diagnostic: controller.OptionalDiagnostics | None = None
    terminal_reference: weakref.ReferenceType[TerminalSelectionAuthorization] | None = None
    terminal_issued: bool = False
    selected_refit_source: _SelectedRefitSource | None = None
    selected_refit_authority_issued: bool = False
    selected_refit_authority_reference: (
        weakref.ReferenceType[SelectedRefitProcedureAuthorization] | None
    ) = None


@dataclass(frozen=True)
class _SelectedRefitSource:
    cells: tuple[Any, ...]
    moments: tuple[Any, ...]
    candidates_reference: weakref.ReferenceType[Any]
    geometry_reference: weakref.ReferenceType[Any]
    pairs_reference: weakref.ReferenceType[Any]
    cells_digest: bytes
    moments_digest: bytes


@dataclass
class _SelectedRefitBinding:
    session_reference: weakref.ReferenceType[ControllerRunSession]
    owner_reference: weakref.ReferenceType[executor.CoreTrialExecutor]
    terminal_reference: weakref.ReferenceType[TerminalSelectionAuthorization]
    winner_reference: weakref.ReferenceType[controller.TrialResult]
    source: _SelectedRefitSource
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    phase: str = "issued_for_full_m1"
    full_objective_reference: weakref.ReferenceType[Any] | None = None
    full_objective_guard: Any | None = None
    fixed_m1_reference: weakref.ReferenceType[Any] | None = None
    prepared_reference: weakref.ReferenceType[Any] | None = None
    full_objective_digest: bytes | None = None
    fixed_m1_digest: bytes | None = None
    prepared_digest: bytes | None = None
    audit_digest: bytes | None = None
    hybrid_reserved: bool = False
    hybrid_receipt_reference: weakref.ReferenceType[Any] | None = None
    hybrid_receipt_guard: Any | None = None
    hybrid_receipt_digest: bytes | None = None
    precursor_reference: weakref.ReferenceType[Any] | None = None
    precursor_guard: Any | None = None
    precursor_digest: bytes | None = None
    cold_reference: weakref.ReferenceType[Any] | None = None
    cold_guard: Any | None = None
    cold_digest: bytes | None = None
    warm_reference: weakref.ReferenceType[Any] | None = None
    warm_guard: Any | None = None
    warm_digest: bytes | None = None


class _ExternalProducer:
    def __init__(self) -> None:
        self.results: dict[int, controller.TrialResult] = {}


class _ExternalValidator:
    def __call__(
        self,
        producer: _ExternalProducer,
        result: controller.TrialResult,
    ) -> bool:
        return producer.results.get(result.trial.valid_index) is result


@dataclass(frozen=True)
class _AdapterState:
    session_reference: weakref.ReferenceType[ControllerRunSession]
    callback_reference: weakref.ReferenceType[Any] | None
    named_adapter_reference: weakref.ReferenceType[Any] | None
    adapter_kind: str
    mode: str
    scopes: tuple[str, ...]
    producer: Any
    validator: Any


@dataclass(frozen=True)
class _TerminalSnapshot:
    session_reference: weakref.ReferenceType[ControllerRunSession]
    owner_reference: weakref.ReferenceType[executor.CoreTrialExecutor]
    cache_reference: weakref.ReferenceType[executor.PreparedTrialCache]
    session_version: int
    maximum_iterations: int
    history_identities: tuple[int, ...]
    history_digests: tuple[bytes, ...]
    schedule_digests: tuple[bytes, ...]
    evidence_reference: weakref.ReferenceType[controller.CoverageEvidence]
    evidence_digest: bytes
    invalid_counts_at_results: tuple[int, ...]
    attempt_digests: tuple[bytes, ...]
    disposition: str
    scientific_conclusion: str
    winner_entry_reference: weakref.ReferenceType[Any]
    winner_result_reference: weakref.ReferenceType[controller.TrialResult]
    configuration_reference: weakref.ReferenceType[controller.Configuration]
    configuration_signature: tuple[object, ...]
    valid_trial_count: int
    invalid_attempt_count: int


def _executor_claim_request(
    session: ControllerRunSession,
    owner: executor.CoreTrialExecutor,
) -> bool:
    if (
        type(session) is not ControllerRunSession
        or session._seal is not _SESSION_SEAL
        or type(owner) is not executor.CoreTrialExecutor
    ):
        raise ObservedSessionError("executor claim request lacks exact session authority")
    state = _lookup_session(session)
    if (
        state.owner is not owner
        or state.cache_reference() is not owner.prepared
        or state.mode != "production"
        or state.status != RUNNING
        or state.in_flight
        or state.attempts
        or state.results
        or state.production_claim is not None
        or state.production_attempts != 0
    ):
        raise ObservedSessionError("executor claim request is not a pristine session")
    return True


def _executor_capability_request(
    session: ControllerRunSession,
    owner: executor.CoreTrialExecutor,
    trial: controller.Trial,
) -> tuple[int, int]:
    if type(session) is not ControllerRunSession or type(owner) is not executor.CoreTrialExecutor:
        raise ObservedSessionError("executor capability request has the wrong types")
    state = _lookup_session(session)
    if (
        state.owner is not owner
        or state.mode != "production"
        or state.status != RUNNING
        or not state.in_flight
        or not state.capability_issuance_open
        or state.production_claim is None
        or trial is not state.schedule[len(state.results)]
    ):
        raise ObservedSessionError("executor capability request is off chronology")
    return len(state.attempts) + 1, state.version


def _executor_ack_request(
    session: ControllerRunSession,
    owner: executor.CoreTrialExecutor,
    capability: executor.ProductionExecutionCapability,
    attempt: _Attempt,
    binding_digest: bytes,
    session_attempt_ordinal: int,
    session_version: int,
) -> bool:
    if type(session) is not ControllerRunSession or type(owner) is not executor.CoreTrialExecutor:
        raise ObservedSessionError("executor acknowledgment request has the wrong types")
    state = _lookup_session(session)
    if (
        state.owner is not owner
        or state.mode != "production"
        or state.status not in {RUNNING, ORDINARY_PATIENCE_STOP, INCOMPLETE_BUDGET_STOP}
        or not state.in_flight
        or not state.attempts
        or state.attempts[-1] is not attempt
        or state.attempt_digests[-1] != executor._deep_integrity_digest(attempt)
        or attempt.ordinal != session_attempt_ordinal
        or state.version != session_version + 1
        or attempt.execution_capability is not capability
        or attempt.execution_capability_digest != binding_digest
    ):
        raise ObservedSessionError("executor acknowledgment differs from session ledger")
    return True


def _optional_adapter_module() -> Any:
    import v3_m2_optional_adapter_v1 as optional_adapter

    expected = TOOLS_DIR / "v3_m2_optional_adapter_v1.py"
    if Path(os.path.abspath(os.fspath(optional_adapter.SCRIPT_PATH))) != expected:
        raise ObservedSessionError("named optional adapter path changed")
    return optional_adapter


def _permutation_adapter_module() -> Any:
    import v3_m2_permutation_production_adapter_v1 as permutation_adapter

    expected = TOOLS_DIR / "v3_m2_permutation_production_adapter_v1.py"
    if Path(os.path.abspath(os.fspath(permutation_adapter.SCRIPT_PATH))) != expected:
        raise ObservedSessionError("named permutation adapter path changed")
    return permutation_adapter


def _rademacher_adapter_module() -> Any:
    import v3_m2_rademacher_production_adapter_v1 as rademacher_adapter

    expected = TOOLS_DIR / "v3_m2_rademacher_production_adapter_v1.py"
    if Path(os.path.abspath(os.fspath(rademacher_adapter.SCRIPT_PATH))) != expected:
        raise ObservedSessionError("named Rademacher adapter path changed")
    return rademacher_adapter


def _optional_adapter_bind_request(
    session: ControllerRunSession,
    authority: AdapterAuthority,
    adapter: object,
    owner: executor.CoreTrialExecutor,
) -> bool:
    optional_adapter = _optional_adapter_module()
    if (
        type(session) is not ControllerRunSession
        or type(authority) is not AdapterAuthority
        or type(adapter) is not optional_adapter.OptionalFamilyAdapter
        or type(owner) is not executor.CoreTrialExecutor
    ):
        raise ObservedSessionError("optional adapter bind request has the wrong types")
    state = _lookup_session(session)
    record = _lookup_adapter(authority)
    if (
        state.owner is not owner
        or state.mode != "production"
        or state.status != RUNNING
        or state.in_flight
        or state.attempts
        or state.results
        or not state.optional_binding_open
        or record.session_reference() is not session
        or record.named_adapter_reference is None
        or record.named_adapter_reference() is not adapter
        or record.adapter_kind != "optional_family_v1"
        or record.mode != "production"
        or record.scopes != ("optional",)
    ):
        raise ObservedSessionError("optional adapter bind request is off chronology")
    return True


def _optional_adapter_capability_request(
    session: ControllerRunSession,
    authority: AdapterAuthority,
    adapter: object,
    owner: executor.CoreTrialExecutor,
    trial: controller.Trial,
    session_attempt_ordinal: int,
    session_version: int,
) -> bool:
    optional_adapter = _optional_adapter_module()
    if type(adapter) is not optional_adapter.OptionalFamilyAdapter:
        raise ObservedSessionError("optional capability request has the wrong adapter")
    state = _lookup_session(session)
    record = _lookup_adapter(authority)
    if (
        state.owner is not owner
        or state.mode != "production"
        or state.status != RUNNING
        or not state.in_flight
        or not state.optional_capability_issuance_open
        or trial is not state.schedule[len(state.results)]
        or _trial_scope(trial) != "optional"
        or session_attempt_ordinal != len(state.attempts) + 1
        or session_version != state.version
        or record.session_reference() is not session
        or record.named_adapter_reference is None
        or record.named_adapter_reference() is not adapter
        or record.adapter_kind != "optional_family_v1"
    ):
        raise ObservedSessionError("optional capability request is off chronology")
    return True


def _optional_adapter_execute_request(
    session: ControllerRunSession,
    authority: AdapterAuthority,
    adapter: object,
    owner: executor.CoreTrialExecutor,
    trial: controller.Trial,
    capability: object,
    binding_digest: bytes,
) -> bool:
    optional_adapter = _optional_adapter_module()
    state = _lookup_session(session)
    record = _lookup_adapter(authority)
    if (
        type(adapter) is not optional_adapter.OptionalFamilyAdapter
        or not optional_adapter._is_optional_attempt_capability(capability)
        or type(binding_digest) is not bytes
        or state.owner is not owner
        or state.mode != "production"
        or state.status != RUNNING
        or not state.in_flight
        or state.optional_capability_issuance_open
        or trial is not state.schedule[len(state.results)]
        or _trial_scope(trial) != "optional"
        or record.session_reference() is not session
        or record.named_adapter_reference is None
        or record.named_adapter_reference() is not adapter
    ):
        raise ObservedSessionError("optional capability use is off chronology")
    return True


def _optional_adapter_ack_request(
    session: ControllerRunSession,
    authority: AdapterAuthority,
    adapter: object,
    capability: object,
    attempt: _Attempt,
    binding_digest: bytes,
    session_attempt_ordinal: int,
    session_version: int,
) -> bool:
    optional_adapter = _optional_adapter_module()
    state = _lookup_session(session)
    record = _lookup_adapter(authority)
    if (
        type(adapter) is not optional_adapter.OptionalFamilyAdapter
        or not optional_adapter._is_optional_attempt_capability(capability)
        or state.mode != "production"
        or state.status
        not in {RUNNING, ORDINARY_PATIENCE_STOP, INCOMPLETE_BUDGET_STOP, TECHNICAL_INTEGRITY_STOP}
        or not state.in_flight
        or not state.attempts
        or state.attempts[-1] is not attempt
        or state.attempt_digests[-1] != executor._deep_integrity_digest(attempt)
        or attempt.ordinal != session_attempt_ordinal
        or state.version != session_version + 1
        or attempt.execution_capability is not capability
        or attempt.execution_capability_digest != binding_digest
        or record.session_reference() is not session
        or record.named_adapter_reference is None
        or record.named_adapter_reference() is not adapter
    ):
        raise ObservedSessionError("optional acknowledgment differs from session ledger")
    return True


def _permutation_adapter_bind_request(
    session: ControllerRunSession,
    authority: AdapterAuthority,
    adapter: object,
    owner: executor.CoreTrialExecutor,
    source_cache: executor.PreparedTrialCache,
) -> bool:
    permutation_adapter = _permutation_adapter_module()
    if (
        type(session) is not ControllerRunSession
        or type(authority) is not AdapterAuthority
        or type(adapter) is not permutation_adapter.WholeSetPermutationAdapter
        or type(owner) is not executor.CoreTrialExecutor
        or type(source_cache) is not executor.PreparedTrialCache
    ):
        raise ObservedSessionError("permutation adapter bind request has wrong types")
    state = _lookup_session(session)
    record = _lookup_adapter(authority)
    if (
        state.owner is not owner
        or owner.prepared is not source_cache
        or state.cache_reference() is not source_cache
        or state.mode != "production"
        or state.status != RUNNING
        or state.in_flight
        or state.attempts
        or state.results
        or not state.permutation_binding_open
        or record.session_reference() is not session
        or record.named_adapter_reference is None
        or record.named_adapter_reference() is not adapter
        or record.adapter_kind != "whole_set_permutation_v1"
        or record.mode != "production"
        or record.scopes != ("permutation",)
    ):
        raise ObservedSessionError("permutation adapter bind request is off chronology")
    return True


def _permutation_adapter_capability_request(
    session: ControllerRunSession,
    authority: AdapterAuthority,
    adapter: object,
    owner: executor.CoreTrialExecutor,
    source_cache: executor.PreparedTrialCache,
    trial: controller.Trial,
    coverage_binding: controller.CoverageBinding,
    null_replicate_index: int | None,
    session_attempt_ordinal: int,
    session_version: int,
) -> bool:
    permutation_adapter = _permutation_adapter_module()
    state = _lookup_session(session)
    record = _lookup_adapter(authority)
    evidence = state.coverage_evidence
    if (
        type(adapter) is not permutation_adapter.WholeSetPermutationAdapter
        or type(owner) is not executor.CoreTrialExecutor
        or type(source_cache) is not executor.PreparedTrialCache
        or type(trial) is not controller.Trial
        or type(coverage_binding) is not controller.CoverageBinding
        or state.owner is not owner
        or owner.prepared is not source_cache
        or state.cache_reference() is not source_cache
        or state.mode != "production"
        or state.status != RUNNING
        or not state.in_flight
        or not state.permutation_capability_issuance_open
        or trial is not state.schedule[len(state.results)]
        or trial.kind != "permutation"
        or _trial_scope(trial) != "falsifier"
        or type(evidence) is not controller.CoverageEvidence
        or coverage_binding is not trial.coverage_binding
        or coverage_binding is not evidence.binding
        or null_replicate_index is not None
        or session_attempt_ordinal != len(state.attempts) + 1
        or session_version != state.version
        or record.session_reference() is not session
        or record.named_adapter_reference is None
        or record.named_adapter_reference() is not adapter
        or record.adapter_kind != "whole_set_permutation_v1"
        or record.scopes != ("permutation",)
    ):
        raise ObservedSessionError("permutation capability request is off chronology")
    return True


def _permutation_adapter_execute_request(
    session: ControllerRunSession,
    authority: AdapterAuthority,
    adapter: object,
    owner: executor.CoreTrialExecutor,
    source_cache: executor.PreparedTrialCache,
    trial: controller.Trial,
    coverage_binding: controller.CoverageBinding,
    null_replicate_index: int | None,
    capability: object,
    binding_digest: bytes,
    session_attempt_ordinal: int,
    session_version: int,
) -> bool:
    permutation_adapter = _permutation_adapter_module()
    state = _lookup_session(session)
    record = _lookup_adapter(authority)
    evidence = state.coverage_evidence
    if (
        type(adapter) is not permutation_adapter.WholeSetPermutationAdapter
        or not permutation_adapter._is_permutation_attempt_capability(capability)
        or type(binding_digest) is not bytes
        or state.owner is not owner
        or owner.prepared is not source_cache
        or state.cache_reference() is not source_cache
        or state.mode != "production"
        or state.status != RUNNING
        or not state.in_flight
        or state.permutation_capability_issuance_open
        or trial is not state.schedule[len(state.results)]
        or trial.kind != "permutation"
        or type(evidence) is not controller.CoverageEvidence
        or coverage_binding is not trial.coverage_binding
        or coverage_binding is not evidence.binding
        or null_replicate_index is not None
        or session_attempt_ordinal != len(state.attempts) + 1
        or session_version != state.version
        or record.session_reference() is not session
        or record.named_adapter_reference is None
        or record.named_adapter_reference() is not adapter
        or record.adapter_kind != "whole_set_permutation_v1"
    ):
        raise ObservedSessionError("permutation capability use is off chronology")
    return True


def _permutation_adapter_ack_request(
    session: ControllerRunSession,
    authority: AdapterAuthority,
    adapter: object,
    owner: executor.CoreTrialExecutor,
    source_cache: executor.PreparedTrialCache,
    trial: controller.Trial,
    coverage_binding: controller.CoverageBinding,
    null_replicate_index: int | None,
    capability: object,
    binding_digest: bytes,
    adapter_attempt_ordinal: int,
    session_attempt_ordinal: int,
    session_version: int,
    completion: str,
    error_type: str | None,
    ledger_attempt: object,
) -> bool:
    permutation_adapter = _permutation_adapter_module()
    state = _lookup_session(session)
    record = _lookup_adapter(authority)
    evidence = state.coverage_evidence
    expected_outcome = "valid" if completion == "success" else None
    if completion == "error":
        expected_outcome = getattr(ledger_attempt, "outcome", None)
        if expected_outcome not in {"invalid", "resource"}:
            expected_outcome = None
    if (
        type(adapter) is not permutation_adapter.WholeSetPermutationAdapter
        or not permutation_adapter._is_permutation_attempt_capability(capability)
        or type(binding_digest) is not bytes
        or completion not in {"success", "error"}
        or (completion == "success" and error_type is not None)
        or (completion == "error" and (type(error_type) is not str or not error_type))
        or type(ledger_attempt) is not _Attempt
        or state.owner is not owner
        or owner.prepared is not source_cache
        or state.cache_reference() is not source_cache
        or state.mode != "production"
        or state.status
        not in {RUNNING, ORDINARY_PATIENCE_STOP, INCOMPLETE_BUDGET_STOP, TECHNICAL_RETRY_STOP}
        or not state.in_flight
        or not state.attempts
        or state.attempts[-1] is not ledger_attempt
        or state.attempt_digests[-1] != executor._deep_integrity_digest(ledger_attempt)
        or ledger_attempt.trial is not trial
        or ledger_attempt.ordinal != session_attempt_ordinal
        or ledger_attempt.outcome != expected_outcome
        or ledger_attempt.error_type != error_type
        or ledger_attempt.authority_identity != id(authority)
        or ledger_attempt.execution_capability is not capability
        or ledger_attempt.execution_capability_digest != binding_digest
        or ledger_attempt.production_attempt_ordinal != adapter_attempt_ordinal
        or adapter_attempt_ordinal != state.permutation_production_attempts + 1
        or state.version != session_version + 1
        or trial.kind != "permutation"
        or type(evidence) is not controller.CoverageEvidence
        or coverage_binding is not trial.coverage_binding
        or coverage_binding is not evidence.binding
        or null_replicate_index is not None
        or record.session_reference() is not session
        or record.named_adapter_reference is None
        or record.named_adapter_reference() is not adapter
        or record.adapter_kind != "whole_set_permutation_v1"
    ):
        raise ObservedSessionError("permutation acknowledgment differs from session ledger")
    return True


def _rademacher_adapter_bind_request(
    session: ControllerRunSession,
    authority: AdapterAuthority,
    adapter: object,
    owner: executor.CoreTrialExecutor,
    source_cache: executor.PreparedTrialCache,
) -> bool:
    rademacher_adapter = _rademacher_adapter_module()
    if (
        type(session) is not ControllerRunSession
        or type(authority) is not AdapterAuthority
        or type(adapter) is not rademacher_adapter.RademacherReferenceAdapter
        or type(owner) is not executor.CoreTrialExecutor
        or type(source_cache) is not executor.PreparedTrialCache
    ):
        raise ObservedSessionError("Rademacher adapter bind request has wrong types")
    state = _lookup_session(session)
    record = _lookup_adapter(authority)
    if (
        state.owner is not owner
        or owner.prepared is not source_cache
        or state.cache_reference() is not source_cache
        or state.mode != "production"
        or state.status != RUNNING
        or state.in_flight
        or state.attempts
        or state.results
        or not state.rademacher_binding_open
        or record.session_reference() is not session
        or record.named_adapter_reference is None
        or record.named_adapter_reference() is not adapter
        or record.adapter_kind != "rademacher_reference_v1"
        or record.mode != "production"
        or record.scopes != ("rademacher",)
    ):
        raise ObservedSessionError("Rademacher adapter bind request is off chronology")
    return True


def _rademacher_adapter_capability_request(
    session: ControllerRunSession,
    authority: AdapterAuthority,
    adapter: object,
    owner: executor.CoreTrialExecutor,
    source_cache: executor.PreparedTrialCache,
    trial: controller.Trial,
    coverage_evidence: controller.CoverageEvidence,
    null_replicate_index: int | None,
    session_attempt_ordinal: int,
    session_version: int,
) -> bool:
    rademacher_adapter = _rademacher_adapter_module()
    state = _lookup_session(session)
    record = _lookup_adapter(authority)
    evidence = state.coverage_evidence
    if (
        type(adapter) is not rademacher_adapter.RademacherReferenceAdapter
        or type(owner) is not executor.CoreTrialExecutor
        or type(source_cache) is not executor.PreparedTrialCache
        or type(trial) is not controller.Trial
        or type(coverage_evidence) is not controller.CoverageEvidence
        or state.owner is not owner
        or owner.prepared is not source_cache
        or state.cache_reference() is not source_cache
        or state.mode != "production"
        or state.status != RUNNING
        or not state.in_flight
        or not state.rademacher_capability_issuance_open
        or trial is not state.schedule[len(state.results)]
        or trial.kind != "rademacher_noise"
        or _trial_scope(trial) != "falsifier"
        or coverage_evidence is not evidence
        or trial.coverage_binding is not coverage_evidence.binding
        or null_replicate_index is not None
        or session_attempt_ordinal != len(state.attempts) + 1
        or session_version != state.version
        or record.session_reference() is not session
        or record.named_adapter_reference is None
        or record.named_adapter_reference() is not adapter
        or record.adapter_kind != "rademacher_reference_v1"
        or record.scopes != ("rademacher",)
    ):
        raise ObservedSessionError("Rademacher capability request is off chronology")
    return True


def _rademacher_adapter_execute_request(
    session: ControllerRunSession,
    authority: AdapterAuthority,
    adapter: object,
    owner: executor.CoreTrialExecutor,
    source_cache: executor.PreparedTrialCache,
    trial: controller.Trial,
    coverage_evidence: controller.CoverageEvidence,
    null_replicate_index: int | None,
    capability: object,
    binding_digest: bytes,
) -> bool:
    rademacher_adapter = _rademacher_adapter_module()
    state = _lookup_session(session)
    record = _lookup_adapter(authority)
    evidence = state.coverage_evidence
    if (
        type(adapter) is not rademacher_adapter.RademacherReferenceAdapter
        or not rademacher_adapter._is_rademacher_attempt_capability(capability)
        or type(binding_digest) is not bytes
        or state.owner is not owner
        or owner.prepared is not source_cache
        or state.cache_reference() is not source_cache
        or state.mode != "production"
        or state.status != RUNNING
        or not state.in_flight
        or state.rademacher_capability_issuance_open
        or trial is not state.schedule[len(state.results)]
        or trial.kind != "rademacher_noise"
        or coverage_evidence is not evidence
        or trial.coverage_binding is not coverage_evidence.binding
        or null_replicate_index is not None
        or record.session_reference() is not session
        or record.named_adapter_reference is None
        or record.named_adapter_reference() is not adapter
        or record.adapter_kind != "rademacher_reference_v1"
    ):
        raise ObservedSessionError("Rademacher capability use is off chronology")
    return True


def _rademacher_adapter_ack_request(
    session: ControllerRunSession,
    authority: AdapterAuthority,
    adapter: object,
    owner: executor.CoreTrialExecutor,
    source_cache: executor.PreparedTrialCache,
    trial: controller.Trial,
    coverage_evidence: controller.CoverageEvidence,
    null_replicate_index: int | None,
    capability: object,
    binding_digest: bytes,
    adapter_attempt_ordinal: int,
    session_attempt_ordinal: int,
    session_version: int,
    completion: str,
    error_type: str | None,
    ledger_attempt: object,
) -> bool:
    rademacher_adapter = _rademacher_adapter_module()
    state = _lookup_session(session)
    record = _lookup_adapter(authority)
    evidence = state.coverage_evidence
    expected_outcome = "valid" if completion == "success" else None
    if completion == "error":
        expected_outcome = getattr(ledger_attempt, "outcome", None)
        if expected_outcome not in {"invalid", "resource"}:
            expected_outcome = None
    if (
        type(adapter) is not rademacher_adapter.RademacherReferenceAdapter
        or not rademacher_adapter._is_rademacher_attempt_capability(capability)
        or type(binding_digest) is not bytes
        or completion not in {"success", "error"}
        or (completion == "success" and error_type is not None)
        or (completion == "error" and (type(error_type) is not str or not error_type))
        or type(ledger_attempt) is not _Attempt
        or state.owner is not owner
        or owner.prepared is not source_cache
        or state.cache_reference() is not source_cache
        or state.mode != "production"
        or state.status
        not in {RUNNING, ORDINARY_PATIENCE_STOP, INCOMPLETE_BUDGET_STOP, TECHNICAL_RETRY_STOP}
        or not state.in_flight
        or not state.attempts
        or state.attempts[-1] is not ledger_attempt
        or state.attempt_digests[-1] != executor._deep_integrity_digest(ledger_attempt)
        or ledger_attempt.trial is not trial
        or ledger_attempt.ordinal != session_attempt_ordinal
        or ledger_attempt.outcome != expected_outcome
        or ledger_attempt.error_type != error_type
        or ledger_attempt.authority_identity != id(authority)
        or ledger_attempt.execution_capability is not capability
        or ledger_attempt.execution_capability_digest != binding_digest
        or ledger_attempt.production_attempt_ordinal != adapter_attempt_ordinal
        or adapter_attempt_ordinal != state.rademacher_production_attempts + 1
        or state.version != session_version + 1
        or trial.kind != "rademacher_noise"
        or coverage_evidence is not evidence
        or trial.coverage_binding is not coverage_evidence.binding
        or null_replicate_index is not None
        or record.session_reference() is not session
        or record.named_adapter_reference is None
        or record.named_adapter_reference() is not adapter
        or record.adapter_kind != "rademacher_reference_v1"
    ):
        raise ObservedSessionError("Rademacher acknowledgment differs from session ledger")
    return True


def _start_session(
    owner: executor.CoreTrialExecutor,
    *,
    mode: str,
) -> ControllerRunSession:
    if type(owner) is not executor.CoreTrialExecutor:
        raise ObservedSessionError("session owner must be an exact CoreTrialExecutor")
    executor._require_prepared_cache(owner.prepared)
    if mode == "production":
        if owner.prepared.mode != executor.SEALED_PRODUCTION_MODE:
            raise ObservedSessionError("observed sessions require sealed production cache")
    elif mode == "synthetic":
        if owner.prepared.mode != executor.SYNTHETIC_TEST_MODE:
            raise ObservedSessionError(
                "synthetic qualification sessions reject sealed production cache"
            )
    else:
        raise ObservedSessionError("unknown controller session mode")
    session = ControllerRunSession(_seal=_SESSION_SEAL)
    initial_schedule = list(controller.coverage_trials())
    state = _SessionState(
        owner=owner,
        session_reference=weakref.ref(session),
        cache_reference=weakref.ref(owner.prepared),
        maximum_iterations=owner.maximum_iterations,
        mode=mode,
        schedule=initial_schedule,
        schedule_digests=[
            executor._deep_integrity_digest(trial) for trial in initial_schedule
        ],
        history=[],
        history_digests=[],
        results=[],
        invalid_counts_at_results=[],
        attempts=[],
        attempt_digests=[],
        authority_bindings={},
    )
    _register_session(session, state)
    if mode == "production":
        try:
            state.production_claim = executor._claim_production_cache(owner, session)
        except executor.ProductionExecutionAuthorityError as error:
            raise ObservedSessionError("production cache claim failed closed") from error
    return session


def start_observed_session(
    owner: executor.CoreTrialExecutor,
    *,
    selected_refit_inputs: tuple[
        Sequence[Any],
        Sequence[Any],
        Any,
        Any,
        Any,
    ]
    | None = None,
) -> ControllerRunSession:
    """Start production and bind same-pass physical inputs for final refit."""
    session = _start_session(owner, mode="production")
    if selected_refit_inputs is None:
        return session
    if type(selected_refit_inputs) is not tuple or len(selected_refit_inputs) != 5:
        raise ObservedSessionError("selected-refit source inputs are malformed")
    cells_raw, moments_raw, candidates, geometry, pairs = selected_refit_inputs
    cells = tuple(cells_raw)
    moments = tuple(moments_raw)
    cache = executor._require_prepared_cache(owner.prepared)
    if (
        len(cells) != executor.EXPECTED_PRODUCTION_ROWS
        or len(moments) != executor.EXPECTED_PRODUCTION_ROWS
        or candidates is not cache.candidates
        or geometry is not cache.geometry
        or pairs is not cache.pairs
        or tuple((cell.identifier, cell.group) for cell in cells)
        != cache.cell_inventory
    ):
        raise ObservedSessionError(
            "selected-refit source differs from the observed cache"
        )
    import fit_v3_m2 as m2

    try:
        for fold in cache.folds:
            m2._require_physical_fold_provenance(
                fold.training,
                cells,
                moments,
                candidates,
                geometry,
                pairs,
                fold.spec.training_indexes,
            )
    except m2.M2Error as error:
        raise ObservedSessionError(
            "selected-refit inputs lack same-pass physical provenance"
        ) from error
    state = _lookup_session(session)
    state.selected_refit_source = _SelectedRefitSource(
        cells=cells,
        moments=moments,
        candidates_reference=weakref.ref(candidates),
        geometry_reference=weakref.ref(geometry),
        pairs_reference=weakref.ref(pairs),
        cells_digest=executor._deep_integrity_digest(cells),
        moments_digest=executor._deep_integrity_digest(moments),
    )
    return session


def start_synthetic_qualification_session(
    owner: executor.CoreTrialExecutor,
) -> ControllerRunSession:
    """Start the unmistakably synthetic, target-free test-only session path."""
    return _start_session(owner, mode="synthetic")


def _require_session(
    session: ControllerRunSession,
    *,
    deep_cache: bool = True,
) -> _SessionState:
    if type(session) is not ControllerRunSession or session._seal is not _SESSION_SEAL:
        raise ObservedSessionError("controller session lacks module authority")
    state = _lookup_session(session)
    if (
        state.session_reference() is not session
        or state.cache_reference() is not state.owner.prepared
    ):
        raise ObservedSessionError("controller session owner/cache binding changed")
    if (
        type(state.owner.maximum_iterations) is not int
        or state.owner.maximum_iterations != state.maximum_iterations
    ):
        raise ObservedSessionError("controller session executor settings changed")
    if state.capability_issuance_open and not state.in_flight:
        raise ObservedSessionError("production capability gate is open off-attempt")
    if state.optional_binding_open:
        raise ObservedSessionError("optional adapter binding gate leaked outside issuance")
    if state.optional_capability_issuance_open and not state.in_flight:
        raise ObservedSessionError("optional capability gate is open off-attempt")
    if state.permutation_binding_open:
        raise ObservedSessionError(
            "permutation adapter binding gate leaked outside issuance"
        )
    if state.permutation_capability_issuance_open and not state.in_flight:
        raise ObservedSessionError("permutation capability gate is open off-attempt")
    if state.rademacher_binding_open:
        raise ObservedSessionError(
            "Rademacher adapter binding gate leaked outside issuance"
        )
    if state.rademacher_capability_issuance_open and not state.in_flight:
        raise ObservedSessionError("Rademacher capability gate is open off-attempt")
    if deep_cache:
        executor._require_prepared_cache(state.owner.prepared)
    if state.mode == "production":
        if state.production_claim is None:
            raise ObservedSessionError("production session lacks its permanent cache claim")
        try:
            executor._require_production_cache_claim(
                state.owner,
                session,
                state.production_claim,
                state.production_attempts,
                allow_pending=state.in_flight,
            )
        except executor.ProductionExecutionAuthorityError as error:
            raise ObservedSessionError(
                "production cache claim or attempt ledger changed"
            ) from error
        binding = state.authority_bindings.get("optional")
        if binding is not None:
            authority = binding[1]()
            if authority is None:
                raise ObservedSessionError("optional authority no longer exists")
            record = _lookup_adapter(authority)
            adapter = (
                None
                if record.named_adapter_reference is None
                else record.named_adapter_reference()
            )
            if adapter is None:
                raise ObservedSessionError("named optional adapter no longer exists")
            try:
                _optional_adapter_module()._require_optional_attempt_count(
                    adapter,
                    session,
                    authority,
                    state.optional_production_attempts,
                    allow_pending=state.in_flight,
                )
            except Exception as error:
                raise ObservedSessionError(
                    "optional adapter attempt ledger changed"
                ) from error
        binding = state.authority_bindings.get("permutation")
        if binding is not None:
            authority = binding[1]()
            if authority is None:
                raise ObservedSessionError("permutation authority no longer exists")
            record = _lookup_adapter(authority)
            adapter = (
                None
                if record.named_adapter_reference is None
                else record.named_adapter_reference()
            )
            if adapter is None:
                raise ObservedSessionError("named permutation adapter no longer exists")
            try:
                _permutation_adapter_module()._require_permutation_attempt_count(
                    adapter,
                    session,
                    authority,
                    state.permutation_production_attempts,
                    allow_pending=state.in_flight,
                )
            except Exception as error:
                raise ObservedSessionError(
                    "permutation adapter attempt ledger changed"
                ) from error
        binding = state.authority_bindings.get("rademacher")
        if binding is not None:
            authority = binding[1]()
            if authority is None:
                raise ObservedSessionError("Rademacher authority no longer exists")
            record = _lookup_adapter(authority)
            adapter = (
                None
                if record.named_adapter_reference is None
                else record.named_adapter_reference()
            )
            if adapter is None:
                raise ObservedSessionError("named Rademacher adapter no longer exists")
            try:
                _rademacher_adapter_module()._require_rademacher_attempt_count(
                    adapter,
                    session,
                    authority,
                    state.rademacher_production_attempts,
                    allow_pending=state.in_flight,
                )
            except Exception as error:
                raise ObservedSessionError(
                    "Rademacher adapter attempt ledger changed"
                ) from error
    _reauthenticate_history(state)
    return state


def _configuration_signature(
    configuration: controller.Configuration,
) -> tuple[object, ...]:
    if type(configuration) is not controller.Configuration:
        raise ObservedSessionError("selected configuration has the wrong exact type")
    if type(configuration.family) is not str:
        raise ObservedSessionError("selected family has the wrong exact type")
    if type(configuration.rho) is not float or not math.isfinite(configuration.rho):
        raise ObservedSessionError("selected rho has the wrong exact type or value")
    if configuration.alpha is None:
        alpha: tuple[object, ...] = (type(None), None)
    else:
        if type(configuration.alpha) is not float or not math.isfinite(configuration.alpha):
            raise ObservedSessionError("selected alpha has the wrong exact type or value")
        alpha = (float, configuration.alpha.hex())
    if configuration.rank is None:
        rank: tuple[object, ...] = (type(None), None)
    else:
        if type(configuration.rank) is not int:
            raise ObservedSessionError("selected rank has the wrong exact type")
        rank = (int, configuration.rank)
    if configuration.sparse_rho is None:
        sparse_rho: tuple[object, ...] = (type(None), None)
    else:
        if (
            type(configuration.sparse_rho) is not float
            or not math.isfinite(configuration.sparse_rho)
        ):
            raise ObservedSessionError(
                "selected sparse rho has the wrong exact type or value"
            )
        sparse_rho = (float, configuration.sparse_rho.hex())
    if type(configuration.sparse_edges) is not tuple or any(
        type(edge) is not int for edge in configuration.sparse_edges
    ):
        raise ObservedSessionError("selected sparse edges have the wrong exact types")
    return (
        controller.Configuration,
        (str, configuration.family),
        (float, configuration.rho.hex()),
        alpha,
        rank,
        sparse_rho,
        tuple((int, edge) for edge in configuration.sparse_edges),
    )


def _trial_scope(trial: controller.Trial) -> str:
    if trial.falsifier:
        return "falsifier"
    configuration = trial.configuration
    if configuration is not None and configuration.family in OPTIONAL_FAMILIES:
        return "optional"
    return "core"


def _authority_scope(trial: controller.Trial, mode: str) -> str:
    scope = _trial_scope(trial)
    if mode == "production" and scope == "falsifier":
        if trial.kind == "permutation":
            return "permutation"
        if trial.kind == "rademacher_noise":
            return "rademacher"
    return scope


def _issue_adapter_authority(
    session: ControllerRunSession,
    callback: Callable[[controller.Trial], SyntheticTrialOutcome],
    *,
    mode: str,
    scopes: Sequence[str],
    synthetic_diagnostic: controller.OptionalDiagnostics | None = None,
) -> AdapterAuthority:
    state = _require_session(
        session,
        deep_cache=mode != "synthetic",
    )
    if mode != "synthetic":
        raise ObservedSessionError(
            "generic production adapter issuance is intentionally unavailable"
        )
    if state.status != RUNNING or state.attempts:
        raise ObservedSessionError("adapter authority must be bound before the first attempt")
    if state.in_flight:
        raise ObservedSessionError("cannot issue authority during an in-flight attempt")
    if mode != state.mode:
        raise ObservedSessionError("adapter authority mode differs from its session")
    if not callable(callback):
        raise ObservedSessionError("adapter callback is not callable")
    try:
        callback_reference = weakref.ref(callback)
    except TypeError as error:
        raise ObservedSessionError("adapter callback needs a stable weak identity") from error
    scope_tuple = tuple(scopes)
    allowed = {"core", "optional", "falsifier"}
    if (
        not scope_tuple
        or len(set(scope_tuple)) != len(scope_tuple)
        or any(type(scope) is not str or scope not in allowed for scope in scope_tuple)
    ):
        raise ObservedSessionError("adapter scope inventory is malformed")
    if mode == "production" and "core" in scope_tuple:
        raise ObservedSessionError("production core execution is owned by the core executor")
    if any(scope in state.authority_bindings for scope in scope_tuple):
        raise ObservedSessionError("adapter scopes already have exact bound authorities")
    if synthetic_diagnostic is not None:
        if mode != "synthetic" or type(synthetic_diagnostic) is not controller.OptionalDiagnostics:
            raise ObservedSessionError(
                "only synthetic qualification may override its diagnostic"
            )
        if (
            state.synthetic_diagnostic is not None
            and state.synthetic_diagnostic != synthetic_diagnostic
        ):
            raise ObservedSessionError("two synthetic diagnostic authorities differ")
        state.synthetic_diagnostic = synthetic_diagnostic
    authority = AdapterAuthority(_seal=_ADAPTER_AUTHORITY_SEAL)
    _register_adapter(
        authority,
        _AdapterState(
            session_reference=weakref.ref(session),
            callback_reference=callback_reference,
            named_adapter_reference=None,
            adapter_kind="synthetic_callback",
            mode=mode,
            scopes=scope_tuple,
            producer=_ExternalProducer(),
            validator=_ExternalValidator(),
        ),
    )
    for scope in scope_tuple:
        state.authority_bindings[scope] = (id(authority), weakref.ref(authority))
    return authority


def issue_synthetic_qualification_authority(
    session: ControllerRunSession,
    callback: Callable[[controller.Trial], SyntheticTrialOutcome],
    *,
    scopes: Sequence[str] = ("core", "optional", "falsifier"),
    diagnostic_override: controller.OptionalDiagnostics | None = None,
) -> AdapterAuthority:
    """Issue only a target-free synthetic test authority.

    This public factory rejects sealed-production sessions.  There is
    intentionally no corresponding generic production factory.
    """
    return _issue_adapter_authority(
        session,
        callback,
        mode="synthetic",
        scopes=scopes,
        synthetic_diagnostic=diagnostic_override,
    )


def issue_optional_family_authority(
    session: ControllerRunSession,
    adapter: object,
) -> AdapterAuthority:
    """Bind the exact named production optional adapter before any attempt."""

    optional_adapter = _optional_adapter_module()
    state = _require_session(session)
    if (
        state.mode != "production"
        or state.status != RUNNING
        or state.in_flight
        or state.attempts
        or state.results
    ):
        raise ObservedSessionError(
            "optional authority must bind to a pristine production session"
        )
    if type(adapter) is not optional_adapter.OptionalFamilyAdapter:
        raise ObservedSessionError("optional authority requires the exact named adapter")
    if "optional" in state.authority_bindings:
        raise ObservedSessionError("optional scope already has a bound authority")
    producer, validator = optional_adapter._adapter_producer_and_validator(adapter)
    authority = AdapterAuthority(_seal=_ADAPTER_AUTHORITY_SEAL)
    _register_adapter(
        authority,
        _AdapterState(
            session_reference=weakref.ref(session),
            callback_reference=None,
            named_adapter_reference=weakref.ref(adapter),
            adapter_kind="optional_family_v1",
            mode="production",
            scopes=("optional",),
            producer=producer,
            validator=validator,
        ),
    )
    state.optional_binding_open = True
    try:
        optional_adapter._bind_to_observed_session(
            adapter,
            session,
            authority,
            state.owner,
        )
    except Exception as error:
        raise ObservedSessionError("named optional adapter binding failed") from error
    finally:
        state.optional_binding_open = False
    state.authority_bindings["optional"] = (id(authority), weakref.ref(authority))
    return authority


def issue_whole_set_permutation_authority(
    session: ControllerRunSession,
    adapter: object,
) -> AdapterAuthority:
    """Bind the exact named permutation adapter before any attempt."""

    permutation_adapter = _permutation_adapter_module()
    state = _require_session(session)
    if (
        state.mode != "production"
        or state.status != RUNNING
        or state.in_flight
        or state.attempts
        or state.results
    ):
        raise ObservedSessionError(
            "permutation authority must bind to a pristine production session"
        )
    if type(adapter) is not permutation_adapter.WholeSetPermutationAdapter:
        raise ObservedSessionError(
            "permutation authority requires the exact named adapter"
        )
    if "permutation" in state.authority_bindings:
        raise ObservedSessionError("permutation scope already has a bound authority")
    producer, validator = permutation_adapter._adapter_producer_and_validator(adapter)
    authority = AdapterAuthority(_seal=_ADAPTER_AUTHORITY_SEAL)
    _register_adapter(
        authority,
        _AdapterState(
            session_reference=weakref.ref(session),
            callback_reference=None,
            named_adapter_reference=weakref.ref(adapter),
            adapter_kind="whole_set_permutation_v1",
            mode="production",
            scopes=("permutation",),
            producer=producer,
            validator=validator,
        ),
    )
    state.permutation_binding_open = True
    try:
        permutation_adapter._bind_to_observed_session(
            adapter,
            session,
            authority,
            state.owner,
        )
    except Exception as error:
        raise ObservedSessionError("named permutation adapter binding failed") from error
    finally:
        state.permutation_binding_open = False
    state.authority_bindings["permutation"] = (
        id(authority),
        weakref.ref(authority),
    )
    return authority


def issue_rademacher_reference_authority(
    session: ControllerRunSession,
    adapter: object,
) -> AdapterAuthority:
    """Bind the exact named Rademacher adapter before any attempt."""

    rademacher_adapter = _rademacher_adapter_module()
    state = _require_session(session)
    if (
        state.mode != "production"
        or state.status != RUNNING
        or state.in_flight
        or state.attempts
        or state.results
    ):
        raise ObservedSessionError(
            "Rademacher authority must bind to a pristine production session"
        )
    if type(adapter) is not rademacher_adapter.RademacherReferenceAdapter:
        raise ObservedSessionError(
            "Rademacher authority requires the exact named adapter"
        )
    if "rademacher" in state.authority_bindings:
        raise ObservedSessionError("Rademacher scope already has a bound authority")
    producer, validator = rademacher_adapter._adapter_producer_and_validator(adapter)
    authority = AdapterAuthority(_seal=_ADAPTER_AUTHORITY_SEAL)
    _register_adapter(
        authority,
        _AdapterState(
            session_reference=weakref.ref(session),
            callback_reference=None,
            named_adapter_reference=weakref.ref(adapter),
            adapter_kind="rademacher_reference_v1",
            mode="production",
            scopes=("rademacher",),
            producer=producer,
            validator=validator,
        ),
    )
    state.rademacher_binding_open = True
    try:
        rademacher_adapter._bind_to_observed_session(
            adapter,
            session,
            authority,
            state.owner,
        )
    except Exception as error:
        raise ObservedSessionError("named Rademacher adapter binding failed") from error
    finally:
        state.rademacher_binding_open = False
    state.authority_bindings["rademacher"] = (
        id(authority),
        weakref.ref(authority),
    )
    return authority


def _diagnostic_rows(state: _SessionState) -> tuple[diagnostic.PairResidualRow, ...]:
    rows: list[diagnostic.PairResidualRow] = []
    for fold in state.owner.prepared.folds:
        for local_index, cell in enumerate(fold.validation.cells):
            membership = state.owner.prepared.candidates.basis(int(cell.k)).membership[
                int(cell.observed_index)
            ]
            rows.append(
                diagnostic.PairResidualRow(
                    str(cell.identifier),
                    str(cell.group),
                    int(cell.k),
                    membership,
                    fold.baseline_pair_probabilities[local_index],
                )
            )
    return tuple(rows)


def _bind_coverage(state: _SessionState) -> None:
    if len(state.results) != controller.COVERAGE_TRIALS:
        raise ObservedSessionError("coverage binding occurred off trial 16")
    core_results = tuple(state.results)
    incumbent = controller.choose_incumbent(core_results)
    incumbent_entry = state.history[incumbent.trial.valid_index - 1]
    if type(incumbent_entry) is not executor.TrialExecution:
        raise ObservedSessionError("core incumbent lacks a core-executor execution")
    if state.synthetic_diagnostic is None:
        diagnostics = diagnostic.diagnose_optional_families(
            _diagnostic_rows(state)
        ).diagnostics
    else:
        diagnostics = state.synthetic_diagnostic
    evidence = controller.bind_coverage_evidence(
        core_results,
        diagnostics,
        incumbent_entry.equal_group_pair_contrasts,
    )
    state.coverage_evidence = evidence
    state.coverage_evidence_digest = executor._deep_integrity_digest(evidence)
    state.results[:] = list(evidence.core_results)


def _append_first_cycle(state: _SessionState) -> None:
    assert state.coverage_evidence is not None
    assert state.coverage_length is not None
    coverage_results = tuple(state.results[: state.coverage_length])
    configurations = controller.successor_configurations(
        coverage_results,
        coverage_evidence=state.coverage_evidence,
    )
    state.cycle_one_configurations = configurations
    _append_schedule(
        state,
        controller._minimum_cycle_trials(
            configurations,
            cycle=1,
            start_valid_index=state.coverage_length + 1,
            coverage_binding=state.coverage_evidence.binding,
        ),
    )


def _append_schedule(
    state: _SessionState,
    trials: Sequence[controller.Trial],
) -> None:
    supplied = tuple(trials)
    if not supplied or any(type(trial) is not controller.Trial for trial in supplied):
        raise ObservedSessionError("derived schedule extension is empty or malformed")
    expected_start = len(state.schedule) + 1
    if tuple(trial.valid_index for trial in supplied) != tuple(
        range(expected_start, expected_start + len(supplied))
    ):
        raise ObservedSessionError("derived schedule extension is not contiguous")
    state.schedule.extend(supplied)
    state.schedule_digests.extend(
        executor._deep_integrity_digest(trial) for trial in supplied
    )


def _extend_schedule(state: _SessionState) -> None:
    if state.status != RUNNING or len(state.schedule) != len(state.results):
        return
    if state.coverage_evidence is None:
        if len(state.results) != controller.COVERAGE_TRIALS:
            raise ObservedSessionError("core schedule exhausted before coverage binding")
        _bind_coverage(state)

    assert state.coverage_evidence is not None
    count = len(state.results)
    if state.coverage_length is None:
        if count == controller.COVERAGE_TRIALS:
            coverage = controller.coverage_trials(
                coverage_evidence=state.coverage_evidence
            )
            if len(coverage) > count:
                _append_schedule(state, coverage[count:])
                return
            state.coverage_length = count
        elif count == controller.COVERAGE_TRIALS + 8:
            if state.coverage_evidence.activation.sparse_plus_low_rank:
                coverage = controller.coverage_trials(
                    coverage_evidence=state.coverage_evidence,
                    low_rank_anchor_results=tuple(
                        state.results[
                            controller.COVERAGE_TRIALS : controller.COVERAGE_TRIALS + 8
                        ]
                    ),
                )
                if len(coverage) > count:
                    _append_schedule(state, coverage[count:])
                    return
            state.coverage_length = count
        elif count == controller.COVERAGE_TRIALS + 12:
            state.coverage_length = count
        else:
            raise ObservedSessionError("optional coverage exhausted off its frozen boundary")

    assert state.coverage_length is not None
    if count == state.coverage_length:
        _append_first_cycle(state)
        return
    if count == state.coverage_length + 12:
        compact_cycle_one = (
            *state.results[: state.coverage_length],
            *state.results[state.coverage_length + 4 : state.coverage_length + 11],
        )
        configurations = controller.successor_configurations(
            compact_cycle_one,
            coverage_evidence=state.coverage_evidence,
        )
        _append_schedule(
            state,
            controller._minimum_cycle_trials(
                configurations,
                cycle=2,
                start_valid_index=count + 1,
                coverage_binding=state.coverage_evidence.binding,
            ),
        )
        return
    if count >= state.coverage_length + 24:
        _append_schedule(
            state,
            controller.continuation_block(
                count + 1,
                tuple(state.results),
                coverage_length=state.coverage_length,
                coverage_evidence=state.coverage_evidence,
            ),
        )
        return
    raise ObservedSessionError("minimum schedule exhausted off a cycle boundary")


def _stop_after_success(state: _SessionState) -> None:
    evidence = state.coverage_evidence
    try:
        disposition = controller.stop_decision(
            tuple(state.results),
            state.invalid_attempts,
            evidence,
        )
    except (controller.ControllerError, TypeError, ValueError) as error:
        raise ObservedSessionError("controller rejected the serial result prefix") from error
    if disposition == ORDINARY_PATIENCE_STOP:
        if len(state.results) >= controller.MAXIMUM_VALID_TRIALS:
            raise ObservedSessionError("ordinary patience stop occurred at the trial cap")
        state.status = disposition
        state.scientific_conclusion = ORDINARY_CONCLUSION
    elif disposition == INCOMPLETE_BUDGET_STOP:
        if len(state.results) != controller.MAXIMUM_VALID_TRIALS:
            raise ObservedSessionError("budget stop occurred before the exact trial cap")
        state.status = disposition
        state.scientific_conclusion = INCOMPLETE_CONCLUSION
    elif disposition == TECHNICAL_RETRY_STOP:
        raise ObservedSessionError("a valid success followed an unsealed retry ceiling")
    elif disposition == "continue" or disposition.startswith("continue_"):
        return
    else:
        raise ObservedSessionError("controller returned an unknown stop disposition")


def _checkpoint(state: _SessionState) -> tuple[object, ...]:
    return (
        list(state.schedule),
        list(state.schedule_digests),
        list(state.history),
        list(state.history_digests),
        list(state.results),
        list(state.invalid_counts_at_results),
        list(state.attempts),
        list(state.attempt_digests),
        state.coverage_evidence,
        state.coverage_evidence_digest,
        state.coverage_length,
        state.cycle_one_configurations,
        state.status,
        state.scientific_conclusion,
        state.version,
        state.invalid_attempts,
        state.production_attempts,
        state.optional_production_attempts,
        state.permutation_production_attempts,
        state.rademacher_production_attempts,
    )


def _restore(state: _SessionState, checkpoint: tuple[object, ...]) -> None:
    (
        schedule,
        schedule_digests,
        history,
        history_digests,
        results,
        invalid_counts,
        attempts,
        attempt_digests,
        evidence,
        evidence_digest,
        coverage_length,
        cycle_one,
        status,
        conclusion,
        version,
        invalid_attempts,
        production_attempts,
        optional_production_attempts,
        permutation_production_attempts,
        rademacher_production_attempts,
    ) = checkpoint
    state.schedule[:] = schedule  # type: ignore[index]
    state.schedule_digests[:] = schedule_digests  # type: ignore[index]
    state.history[:] = history  # type: ignore[index]
    state.history_digests[:] = history_digests  # type: ignore[index]
    state.results[:] = results  # type: ignore[index]
    state.invalid_counts_at_results[:] = invalid_counts  # type: ignore[index]
    state.attempts[:] = attempts  # type: ignore[index]
    state.attempt_digests[:] = attempt_digests  # type: ignore[index]
    state.coverage_evidence = evidence  # type: ignore[assignment]
    state.coverage_evidence_digest = evidence_digest  # type: ignore[assignment]
    state.coverage_length = coverage_length  # type: ignore[assignment]
    state.cycle_one_configurations = cycle_one  # type: ignore[assignment]
    state.status = status  # type: ignore[assignment]
    state.scientific_conclusion = conclusion  # type: ignore[assignment]
    state.version = version  # type: ignore[assignment]
    state.invalid_attempts = invalid_attempts  # type: ignore[assignment]
    state.production_attempts = production_attempts  # type: ignore[assignment]
    state.optional_production_attempts = optional_production_attempts  # type: ignore[assignment]
    state.permutation_production_attempts = permutation_production_attempts  # type: ignore[assignment]
    state.rademacher_production_attempts = rademacher_production_attempts  # type: ignore[assignment]


def _record_invalid(
    state: _SessionState,
    trial: controller.Trial,
    authority_identity: int | None,
    error: BaseException,
    capability: object | None = None,
) -> None:
    production_ordinal, capability_digest, completion_error = (
        _completed_capability_binding(state, capability, "error")
    )
    observed_error = f"{type(error).__module__}.{type(error).__qualname__}"
    if completion_error is not None and completion_error != observed_error:
        raise ObservedSessionError("executor/session failure classifications differ")
    checkpoint = _checkpoint(state)
    try:
        state.invalid_attempts += 1
        state.version += 1
        attempt = _Attempt(
            ordinal=len(state.attempts) + 1,
            trial=trial,
            outcome="invalid",
            entry_identity=None,
            invalid_attempts_after=state.invalid_attempts,
            authority_identity=authority_identity,
            error_type=observed_error,
            execution_capability=capability,
            execution_capability_digest=capability_digest,
            production_attempt_ordinal=production_ordinal,
        )
        state.attempts.append(attempt)
        state.attempt_digests.append(executor._deep_integrity_digest(attempt))
        _acknowledge_capability(state, attempt)
        if state.invalid_attempts >= controller.MAX_INVALID_RETRIES:
            state.status = TECHNICAL_RETRY_STOP
            state.scientific_conclusion = None
    except Exception:
        _restore(state, checkpoint)
        raise


def _completed_capability_binding(
    state: _SessionState,
    capability: object | None,
    expected_completion: str,
) -> tuple[int | None, bytes | None, str | None]:
    if capability is None:
        return None, None, None
    claim = state.production_claim
    if state.mode != "production" or claim is None:
        raise ObservedSessionError("synthetic attempt carried a production capability")
    if type(capability) is executor.ProductionExecutionCapability:
        try:
            ordinal, digest, completion, error_type = (
                executor._completed_production_execution_binding(
                    state.owner,
                    _session_for_state(state),
                    claim,
                    capability,
                )
            )
        except executor.ProductionExecutionAuthorityError as error:
            raise ObservedSessionError(
                "production capability did not complete exactly once"
            ) from error
        if completion != expected_completion:
            raise ObservedSessionError("production capability completion kind changed")
        return ordinal, digest, error_type
    permutation_adapter = _permutation_adapter_module()
    if permutation_adapter._is_permutation_attempt_capability(capability):
        _, authority, adapter = _permutation_attempt_context(state, capability)
        try:
            ordinal, digest, error_type, _ = (
                permutation_adapter._completed_permutation_attempt_binding(
                    adapter,
                    _session_for_state(state),
                    authority,
                    capability,
                    expected_completion,
                )
            )
        except Exception as error:
            raise ObservedSessionError(
                "permutation capability did not complete exactly once"
            ) from error
        return ordinal, digest, error_type
    rademacher_adapter = _rademacher_adapter_module()
    if rademacher_adapter._is_rademacher_attempt_capability(capability):
        _, authority, adapter = _rademacher_attempt_context(state, capability)
        try:
            ordinal, digest, error_type, _ = (
                rademacher_adapter._completed_rademacher_attempt_binding(
                    adapter,
                    _session_for_state(state),
                    authority,
                    capability,
                    expected_completion,
                )
            )
        except Exception as error:
            raise ObservedSessionError(
                "Rademacher capability did not complete exactly once"
            ) from error
        return ordinal, digest, error_type
    optional_adapter, authority, adapter = _optional_attempt_context(state, capability)
    try:
        ordinal, digest, error_type = (
            optional_adapter._completed_optional_attempt_binding(
                adapter,
                capability,
                expected_completion,
            )
        )
    except Exception as error:
        raise ObservedSessionError(
            "optional capability did not complete exactly once"
        ) from error
    del authority
    return ordinal, digest, error_type


def _session_for_state(state: _SessionState) -> ControllerRunSession:
    session = state.session_reference()
    if type(session) is not ControllerRunSession or _lookup_session(session) is not state:
        raise ObservedSessionError("session state lost its exact issued identity")
    return session


def _optional_attempt_context(
    state: _SessionState,
    capability: object,
) -> tuple[Any, AdapterAuthority, object]:
    optional_adapter = _optional_adapter_module()
    if not optional_adapter._is_optional_attempt_capability(capability):
        raise ObservedSessionError("attempt capability has an unknown exact type")
    binding = state.authority_bindings.get("optional")
    if binding is None:
        raise ObservedSessionError("optional capability lacks a bound authority")
    authority = binding[1]()
    if authority is None:
        raise ObservedSessionError("optional authority no longer exists")
    record = _lookup_adapter(authority)
    adapter = (
        None
        if record.named_adapter_reference is None
        else record.named_adapter_reference()
    )
    if adapter is None:
        raise ObservedSessionError("named optional adapter no longer exists")
    return optional_adapter, authority, adapter


def _permutation_attempt_context(
    state: _SessionState,
    capability: object,
) -> tuple[Any, AdapterAuthority, object]:
    permutation_adapter = _permutation_adapter_module()
    if not permutation_adapter._is_permutation_attempt_capability(capability):
        raise ObservedSessionError("attempt capability has an unknown exact type")
    binding = state.authority_bindings.get("permutation")
    if binding is None:
        raise ObservedSessionError("permutation capability lacks a bound authority")
    authority = binding[1]()
    if authority is None:
        raise ObservedSessionError("permutation authority no longer exists")
    record = _lookup_adapter(authority)
    adapter = (
        None
        if record.named_adapter_reference is None
        else record.named_adapter_reference()
    )
    if adapter is None:
        raise ObservedSessionError("named permutation adapter no longer exists")
    return permutation_adapter, authority, adapter


def _rademacher_attempt_context(
    state: _SessionState,
    capability: object,
) -> tuple[Any, AdapterAuthority, object]:
    rademacher_adapter = _rademacher_adapter_module()
    if not rademacher_adapter._is_rademacher_attempt_capability(capability):
        raise ObservedSessionError("attempt capability has an unknown exact type")
    binding = state.authority_bindings.get("rademacher")
    if binding is None:
        raise ObservedSessionError("Rademacher capability lacks a bound authority")
    authority = binding[1]()
    if authority is None:
        raise ObservedSessionError("Rademacher authority no longer exists")
    record = _lookup_adapter(authority)
    adapter = (
        None
        if record.named_adapter_reference is None
        else record.named_adapter_reference()
    )
    if adapter is None:
        raise ObservedSessionError("named Rademacher adapter no longer exists")
    return rademacher_adapter, authority, adapter


def _acknowledge_capability(state: _SessionState, attempt: _Attempt) -> None:
    capability = attempt.execution_capability
    if capability is None:
        if (
            attempt.execution_capability_digest is not None
            or attempt.production_attempt_ordinal is not None
        ):
            raise ObservedSessionError("nonproduction attempt carries capability fields")
        return
    session = _session_for_state(state)
    if type(capability) is executor.ProductionExecutionCapability:
        claim = state.production_claim
        if claim is None:
            raise ObservedSessionError("production capability lacks its cache claim")
        try:
            ordinal, digest = executor._acknowledge_production_execution_attempt(
                state.owner,
                session,
                claim,
                capability,
                attempt,
            )
        except executor.ProductionExecutionAuthorityError as error:
            raise ObservedSessionError(
                "production capability could not be acknowledged by the exact ledger"
            ) from error
        if (
            ordinal != state.production_attempts + 1
            or ordinal != attempt.production_attempt_ordinal
            or digest != attempt.execution_capability_digest
        ):
            raise ObservedSessionError("production capability ordinal or digest changed")
        state.production_attempts = ordinal
        return
    permutation_adapter = _permutation_adapter_module()
    if permutation_adapter._is_permutation_attempt_capability(capability):
        _, authority, adapter = _permutation_attempt_context(state, capability)
        try:
            ordinal, digest = permutation_adapter._acknowledge_permutation_attempt(
                adapter,
                session,
                authority,
                capability,
                attempt,
            )
        except Exception as error:
            raise ObservedSessionError(
                "permutation capability could not be acknowledged by the exact ledger"
            ) from error
        if (
            ordinal != state.permutation_production_attempts + 1
            or ordinal != attempt.production_attempt_ordinal
            or digest != attempt.execution_capability_digest
        ):
            raise ObservedSessionError(
                "permutation capability ordinal or digest changed"
            )
        state.permutation_production_attempts = ordinal
        return
    rademacher_adapter = _rademacher_adapter_module()
    if rademacher_adapter._is_rademacher_attempt_capability(capability):
        _, authority, adapter = _rademacher_attempt_context(state, capability)
        try:
            ordinal, digest = rademacher_adapter._acknowledge_rademacher_attempt(
                adapter,
                session,
                authority,
                capability,
                attempt,
            )
        except Exception as error:
            raise ObservedSessionError(
                "Rademacher capability could not be acknowledged by the exact ledger"
            ) from error
        if (
            ordinal != state.rademacher_production_attempts + 1
            or ordinal != attempt.production_attempt_ordinal
            or digest != attempt.execution_capability_digest
        ):
            raise ObservedSessionError(
                "Rademacher capability ordinal or digest changed"
            )
        state.rademacher_production_attempts = ordinal
        return
    optional_adapter, authority, adapter = _optional_attempt_context(state, capability)
    try:
        ordinal, digest = optional_adapter._acknowledge_optional_attempt(
            adapter,
            session,
            authority,
            capability,
            attempt,
        )
    except Exception as error:
        raise ObservedSessionError(
            "optional capability could not be acknowledged by the exact ledger"
        ) from error
    if (
        ordinal != state.optional_production_attempts + 1
        or ordinal != attempt.production_attempt_ordinal
        or digest != attempt.execution_capability_digest
    ):
        raise ObservedSessionError("optional capability ordinal or digest changed")
    state.optional_production_attempts = ordinal


def _record_resource_attempt(
    state: _SessionState,
    trial: controller.Trial,
    authority_identity: int | None,
    error: BaseException,
    capability: object | None = None,
) -> None:
    production_ordinal, capability_digest, completion_error = (
        _completed_capability_binding(state, capability, "error")
    )
    observed_error = f"{type(error).__module__}.{type(error).__qualname__}"
    if completion_error is not None and completion_error != observed_error:
        raise ObservedSessionError("resource-stop classifications differ")
    checkpoint = _checkpoint(state)
    try:
        state.version += 1
        attempt = _Attempt(
            ordinal=len(state.attempts) + 1,
            trial=trial,
            outcome="resource",
            entry_identity=None,
            invalid_attempts_after=state.invalid_attempts,
            authority_identity=authority_identity,
            error_type=observed_error,
            execution_capability=capability,
            execution_capability_digest=capability_digest,
            production_attempt_ordinal=production_ordinal,
        )
        state.attempts.append(attempt)
        state.attempt_digests.append(executor._deep_integrity_digest(attempt))
        _acknowledge_capability(state, attempt)
    except Exception:
        _restore(state, checkpoint)
        raise


def _commit_valid(
    state: _SessionState,
    entry: executor.TrialExecution | executor.ExternalExecutionReceipt,
    result: controller.TrialResult,
    authority_identity: int | None,
    capability: object | None = None,
) -> controller.TrialResult:
    if state.status != RUNNING:
        raise ObservedSessionError("a sealed session cannot accept another result")
    trial = state.schedule[len(state.results)]
    if (
        type(result) is not controller.TrialResult
        or not result.valid
        or result.trial is not trial
    ):
        raise ObservedSessionError("adapter result is not the exact expected valid trial")
    production_ordinal, capability_digest, _ = _completed_capability_binding(
        state, capability, "success"
    )
    checkpoint = _checkpoint(state)
    try:
        state.history.append(entry)
        state.history_digests.append(executor._deep_integrity_digest(entry))
        state.results.append(result)
        state.invalid_counts_at_results.append(state.invalid_attempts)
        state.version += 1
        attempt = _Attempt(
            ordinal=len(state.attempts) + 1,
            trial=trial,
            outcome="valid",
            entry_identity=id(entry),
            invalid_attempts_after=state.invalid_attempts,
            authority_identity=authority_identity,
            error_type=None,
            execution_capability=capability,
            execution_capability_digest=capability_digest,
            production_attempt_ordinal=production_ordinal,
        )
        state.attempts.append(attempt)
        state.attempt_digests.append(executor._deep_integrity_digest(attempt))
        _acknowledge_capability(state, attempt)
        if len(state.results) == controller.COVERAGE_TRIALS:
            _bind_coverage(state)
        _stop_after_success(state)
        if state.status == RUNNING:
            _extend_schedule(state)
        _reauthenticate_history(state)
    except Exception:
        _restore(state, checkpoint)
        raise
    return state.results[-1]


def _successor_authorization(
    state: _SessionState,
    trial: controller.Trial,
) -> executor.SuccessorAuthorization | None:
    if trial.kind != "successor" or _trial_scope(trial) != "core":
        return None
    if state.coverage_evidence is None:
        raise ObservedSessionError("core successor precedes coverage evidence")
    if state.mode == "synthetic":
        _reauthenticate_history(state)
        if trial is not state.schedule[len(state.results)]:
            raise ObservedSessionError("synthetic successor differs from exact next trial")
        return None
    return executor.authorize_successor(
        state.owner,
        trial,
        tuple(state.history),
        state.coverage_evidence,
    )


def _run_next_core(session: ControllerRunSession) -> controller.TrialResult:
    state = _require_session(session)
    if state.status != RUNNING:
        raise ObservedSessionError("a sealed session has no next core trial")
    if state.in_flight:
        raise ObservedSessionError("controller session already has an in-flight attempt")
    trial = state.schedule[len(state.results)]
    if _trial_scope(trial) != "core":
        raise ObservedSessionError("next trial belongs to an external adapter")
    state.in_flight = True
    capability: executor.ProductionExecutionCapability | None = None
    try:
        try:
            authorization = _successor_authorization(state, trial)
            if state.mode == "production":
                claim = state.production_claim
                if claim is None:
                    raise ObservedSessionError("production session lost its cache claim")
                state.capability_issuance_open = True
                try:
                    capability = executor._issue_production_execution_capability(
                        state.owner,
                        session,
                        claim,
                        trial,
                        authorization,
                    )
                finally:
                    state.capability_issuance_open = False
        except Exception as error:
            _seal_integrity_stop(state)
            raise ObservedSessionError(
                "core preflight or one-shot capability issuance failed integrity validation"
            ) from error
        try:
            entry = state.owner.execute(
                trial,
                authorization=authorization,
                _session_capability=capability,
            )
        except executor.ProductionExecutionAuthorityError as error:
            _seal_integrity_stop(state)
            raise ObservedSessionError(
                "production execution capability failed before numerical retry"
            ) from error
        except ObservedResourceStop as error:
            try:
                _record_resource_attempt(state, trial, None, error, capability)
            except Exception as ledger_error:
                _seal_integrity_stop(state)
                raise ObservedSessionError(
                    "resource attempt failed capability acknowledgment"
                ) from ledger_error
            _seal_resource_stop(session, "core adapter resource stop", _internal=True)
            raise
        except BaseException as error:
            try:
                if state.status == RUNNING:
                    _record_invalid(state, trial, None, error, capability)
            except Exception as ledger_error:
                _seal_integrity_stop(state)
                raise ObservedSessionError(
                    "failed core attempt was not ledger-acknowledged"
                ) from ledger_error
            if isinstance(error, Exception):
                raise ObservedSessionError("core trial attempt failed") from error
            raise
        try:
            result = executor._require_owned_execution(state.owner, entry)
            return _commit_valid(state, entry, result, None, capability)
        except Exception as error:
            _seal_integrity_stop(state)
            raise ObservedSessionError(
                "returned core result failed atomic commit integrity"
            ) from error
    finally:
        state.capability_issuance_open = False
        state.in_flight = False


def _require_adapter(
    session: ControllerRunSession,
    state: _SessionState,
    authority: AdapterAuthority,
    *,
    required_mode: str,
    scope: str,
) -> _AdapterState:
    if type(authority) is not AdapterAuthority or authority._seal is not _ADAPTER_AUTHORITY_SEAL:
        raise ObservedSessionError("external attempt lacks adapter authority")
    record = _lookup_adapter(authority)
    if record.session_reference() is not session:
        raise ObservedSessionError("adapter authority belongs to another session")
    if record.mode != required_mode or scope not in record.scopes:
        raise ObservedSessionError("adapter authority has the wrong mode or scope")
    binding = state.authority_bindings.get(scope)
    if (
        binding is None
        or binding[0] != id(authority)
        or binding[1]() is not authority
    ):
        raise ObservedSessionError("adapter differs from the frozen scope authority")
    if record.adapter_kind == "synthetic_callback":
        if record.callback_reference is None or record.callback_reference() is None:
            raise ObservedSessionError("authorized adapter callback no longer exists")
    elif record.adapter_kind == "optional_family_v1":
        if (
            record.named_adapter_reference is None
            or record.named_adapter_reference() is None
            or scope != "optional"
            or required_mode != "production"
        ):
            raise ObservedSessionError("named optional adapter no longer exists")
    elif record.adapter_kind == "whole_set_permutation_v1":
        if (
            record.named_adapter_reference is None
            or record.named_adapter_reference() is None
            or scope != "permutation"
            or required_mode != "production"
        ):
            raise ObservedSessionError("named permutation adapter no longer exists")
    elif record.adapter_kind == "rademacher_reference_v1":
        if (
            record.named_adapter_reference is None
            or record.named_adapter_reference() is None
            or scope != "rademacher"
            or required_mode != "production"
        ):
            raise ObservedSessionError("named Rademacher adapter no longer exists")
    else:
        raise ObservedSessionError("external adapter kind is not approved")
    return record


def _run_next_permutation(
    session: ControllerRunSession,
    state: _SessionState,
    authority: AdapterAuthority,
    record: _AdapterState,
    trial: controller.Trial,
) -> controller.TrialResult:
    permutation_adapter = _permutation_adapter_module()
    adapter = (
        None
        if record.named_adapter_reference is None
        else record.named_adapter_reference()
    )
    evidence = state.coverage_evidence
    if (
        adapter is None
        or type(adapter) is not permutation_adapter.WholeSetPermutationAdapter
        or type(evidence) is not controller.CoverageEvidence
        or trial.kind != "permutation"
        or trial.coverage_binding is not evidence.binding
    ):
        _seal_integrity_stop(state)
        raise ObservedSessionError(
            "permutation attempt lacks its adapter or coverage evidence"
        )
    capability: object | None = None
    try:
        state.permutation_capability_issuance_open = True
        try:
            capability = permutation_adapter._issue_permutation_attempt_capability(
                adapter,
                session,
                authority,
                trial,
                null_replicate_index=None,
                session_attempt_ordinal=len(state.attempts) + 1,
                session_version=state.version,
            )
        finally:
            state.permutation_capability_issuance_open = False
    except Exception as error:
        _seal_integrity_stop(state)
        raise ObservedSessionError(
            "permutation one-shot capability issuance failed integrity validation"
        ) from error
    try:
        outcome = permutation_adapter.execute_permutation_attempt(
            adapter,
            capability,
        )
    except ObservedResourceStop as error:
        try:
            _record_resource_attempt(
                state,
                trial,
                id(authority),
                error,
                capability,
            )
        except Exception as ledger_error:
            _seal_integrity_stop(state)
            raise ObservedSessionError(
                "permutation resource attempt failed ledger acknowledgment"
            ) from ledger_error
        _seal_resource_stop(
            session,
            "permutation adapter resource stop",
            _internal=True,
        )
        raise
    except BaseException as error:
        try:
            if state.status == RUNNING:
                _record_invalid(
                    state,
                    trial,
                    id(authority),
                    error,
                    capability,
                )
        except Exception as ledger_error:
            _seal_integrity_stop(state)
            raise ObservedSessionError(
                "failed permutation attempt was not ledger-acknowledged"
            ) from ledger_error
        if isinstance(error, permutation_adapter.PermutationExecutionAuthorityError):
            _seal_integrity_stop(state)
        if isinstance(error, Exception):
            raise ObservedSessionError("permutation trial attempt failed") from error
        raise
    try:
        result = permutation_adapter._require_registered_execution(
            adapter,
            outcome,
        )
        if result.trial is not trial or not result.valid:
            raise ObservedSessionError(
                "named permutation adapter returned the wrong trial result"
            )
        entry = executor.authenticate_external_execution(
            state.owner,
            record.producer,
            record.validator,
            result,
            execution_detail=outcome,
        )
        return _commit_valid(
            state,
            entry,
            result,
            id(authority),
            capability,
        )
    except Exception as error:
        _seal_integrity_stop(state)
        raise ObservedSessionError(
            "returned permutation execution failed atomic commit integrity"
        ) from error


def _run_next_rademacher(
    session: ControllerRunSession,
    state: _SessionState,
    authority: AdapterAuthority,
    record: _AdapterState,
    trial: controller.Trial,
) -> controller.TrialResult:
    rademacher_adapter = _rademacher_adapter_module()
    adapter = (
        None
        if record.named_adapter_reference is None
        else record.named_adapter_reference()
    )
    evidence = state.coverage_evidence
    if (
        adapter is None
        or type(adapter) is not rademacher_adapter.RademacherReferenceAdapter
        or type(evidence) is not controller.CoverageEvidence
        or trial.kind != "rademacher_noise"
        or trial.coverage_binding is not evidence.binding
    ):
        _seal_integrity_stop(state)
        raise ObservedSessionError(
            "Rademacher attempt lacks its adapter or coverage evidence"
        )
    capability: object | None = None
    try:
        state.rademacher_capability_issuance_open = True
        try:
            capability = rademacher_adapter._issue_rademacher_attempt_capability(
                adapter,
                session,
                authority,
                trial,
                evidence,
                tuple(state.history),
                tuple(state.results),
                null_replicate_index=None,
                session_attempt_ordinal=len(state.attempts) + 1,
                session_version=state.version,
            )
        finally:
            state.rademacher_capability_issuance_open = False
    except Exception as error:
        _seal_integrity_stop(state)
        raise ObservedSessionError(
            "Rademacher one-shot capability issuance failed integrity validation"
        ) from error
    try:
        outcome = rademacher_adapter.execute_rademacher_attempt(
            adapter,
            capability,
        )
    except ObservedResourceStop as error:
        try:
            _record_resource_attempt(
                state,
                trial,
                id(authority),
                error,
                capability,
            )
        except Exception as ledger_error:
            _seal_integrity_stop(state)
            raise ObservedSessionError(
                "Rademacher resource attempt failed ledger acknowledgment"
            ) from ledger_error
        _seal_resource_stop(
            session,
            "Rademacher adapter resource stop",
            _internal=True,
        )
        raise
    except BaseException as error:
        try:
            if state.status == RUNNING:
                _record_invalid(
                    state,
                    trial,
                    id(authority),
                    error,
                    capability,
                )
        except Exception as ledger_error:
            _seal_integrity_stop(state)
            raise ObservedSessionError(
                "failed Rademacher attempt was not ledger-acknowledged"
            ) from ledger_error
        if isinstance(error, rademacher_adapter.RademacherExecutionAuthorityError):
            _seal_integrity_stop(state)
        if isinstance(error, Exception):
            raise ObservedSessionError("Rademacher trial attempt failed") from error
        raise
    try:
        result = rademacher_adapter._require_registered_execution(
            adapter,
            outcome,
        )
        if result.trial is not trial or not result.valid:
            raise ObservedSessionError(
                "named Rademacher adapter returned the wrong trial result"
            )
        entry = executor.authenticate_external_execution(
            state.owner,
            record.producer,
            record.validator,
            result,
            execution_detail=outcome,
        )
        return _commit_valid(
            state,
            entry,
            result,
            id(authority),
            capability,
        )
    except Exception as error:
        _seal_integrity_stop(state)
        raise ObservedSessionError(
            "returned Rademacher execution failed atomic commit integrity"
        ) from error


def _run_next_external(
    session: ControllerRunSession,
    authority: AdapterAuthority,
    *,
    required_mode: str,
    required_scope: str | None = None,
) -> controller.TrialResult:
    state = _require_session(
        session,
        deep_cache=required_mode != "synthetic",
    )
    if state.status != RUNNING:
        raise ObservedSessionError("a sealed session has no next external trial")
    if state.in_flight:
        raise ObservedSessionError("controller session already has an in-flight attempt")
    trial = state.schedule[len(state.results)]
    scope = _trial_scope(trial)
    authority_scope = _authority_scope(trial, required_mode)
    if required_scope is not None and scope != required_scope:
        raise ObservedSessionError("next external trial has the wrong named scope")
    if required_mode == "production" and scope == "core":
        raise ObservedSessionError("production core trials must use run_next_core")
    try:
        record = _require_adapter(
            session,
            state,
            authority,
            required_mode=required_mode,
            scope=authority_scope,
        )
    except Exception:
        # An untrusted/cross-session object never becomes a controller attempt.
        raise
    state.in_flight = True
    try:
        if required_mode == "production":
            if authority_scope == "permutation":
                return _run_next_permutation(
                    session,
                    state,
                    authority,
                    record,
                    trial,
                )
            if authority_scope == "rademacher":
                return _run_next_rademacher(
                    session,
                    state,
                    authority,
                    record,
                    trial,
                )
            if scope != "optional" or record.adapter_kind != "optional_family_v1":
                raise ObservedSessionError(
                    "this production external scope lacks a named adapter"
                )
            optional_adapter = _optional_adapter_module()
            adapter = (
                None
                if record.named_adapter_reference is None
                else record.named_adapter_reference()
            )
            evidence = state.coverage_evidence
            if adapter is None or type(evidence) is not controller.CoverageEvidence:
                _seal_integrity_stop(state)
                raise ObservedSessionError(
                    "optional attempt lacks its adapter or coverage evidence"
                )
            capability: object | None = None
            try:
                state.optional_capability_issuance_open = True
                try:
                    capability = optional_adapter._issue_optional_attempt_capability(
                        adapter,
                        session,
                        authority,
                        state.owner,
                        trial,
                        evidence,
                        tuple(state.history),
                        tuple(state.results),
                        len(state.attempts) + 1,
                        state.version,
                    )
                finally:
                    state.optional_capability_issuance_open = False
            except Exception as error:
                _seal_integrity_stop(state)
                raise ObservedSessionError(
                    "optional one-shot capability issuance failed integrity validation"
                ) from error
            try:
                outcome = optional_adapter._execute_optional_attempt(
                    adapter,
                    capability,
                )
            except ObservedResourceStop as error:
                try:
                    _record_resource_attempt(
                        state,
                        trial,
                        id(authority),
                        error,
                        capability,
                    )
                except Exception as ledger_error:
                    _seal_integrity_stop(state)
                    raise ObservedSessionError(
                        "optional resource attempt failed ledger acknowledgment"
                    ) from ledger_error
                _seal_resource_stop(
                    session,
                    "optional adapter resource stop",
                    _internal=True,
                )
                raise
            except BaseException as error:
                try:
                    if state.status == RUNNING:
                        _record_invalid(
                            state,
                            trial,
                            id(authority),
                            error,
                            capability,
                        )
                except Exception as ledger_error:
                    _seal_integrity_stop(state)
                    raise ObservedSessionError(
                        "failed optional attempt was not ledger-acknowledged"
                    ) from ledger_error
                if isinstance(error, optional_adapter.OptionalExecutionAuthorityError):
                    _seal_integrity_stop(state)
                if isinstance(error, Exception):
                    raise ObservedSessionError("optional trial attempt failed") from error
                raise
            try:
                result = optional_adapter._require_registered_execution(
                    adapter,
                    outcome,
                )
                if result.trial is not trial or not result.valid:
                    raise ObservedSessionError(
                        "named optional adapter returned the wrong trial result"
                    )
                entry = executor.authenticate_external_execution(
                    state.owner,
                    record.producer,
                    record.validator,
                    result,
                    execution_detail=outcome,
                )
                return _commit_valid(
                    state,
                    entry,
                    result,
                    id(authority),
                    capability,
                )
            except Exception as error:
                _seal_integrity_stop(state)
                raise ObservedSessionError(
                    "returned optional execution failed atomic commit integrity"
                ) from error

        if record.adapter_kind != "synthetic_callback":
            raise ObservedSessionError("synthetic execution lacks its callback adapter")
        callback = (
            None if record.callback_reference is None else record.callback_reference()
        )
        if callback is None:
            raise ObservedSessionError("synthetic callback no longer exists")
        if scope == "core" and trial.kind == "successor":
            try:
                _successor_authorization(state, trial)
            except Exception as error:
                _seal_integrity_stop(state)
                raise ObservedSessionError(
                    "synthetic core successor preflight failed integrity validation"
                ) from error
        try:
            outcome = callback(trial)
        except ObservedResourceStop as error:
            try:
                _record_resource_attempt(state, trial, id(authority), error)
            except Exception as ledger_error:
                _seal_integrity_stop(state)
                raise ObservedSessionError(
                    "external resource attempt failed ledger commit"
                ) from ledger_error
            _seal_resource_stop(session, "external adapter resource stop", _internal=True)
            raise
        except Exception as error:
            if state.status == RUNNING:
                _record_invalid(state, trial, id(authority), error)
            raise ObservedSessionError("external trial attempt failed") from error
        if type(outcome) is not SyntheticTrialOutcome:
            error = ObservedSessionError(
                "adapter must return an exact SyntheticTrialOutcome receipt"
            )
            if type(outcome) is controller.TrialResult:
                _seal_integrity_stop(state)
            else:
                _record_invalid(state, trial, id(authority), error)
            raise error
        try:
            result = outcome.result
            if result.trial is not trial or not result.valid:
                raise ObservedSessionError(
                    "adapter returned the wrong or invalid trial result"
                )
            if scope == "core":
                if required_mode != "synthetic":
                    raise ObservedSessionError("production core callback is forbidden")
                entry = executor.TrialExecution(
                    trial=trial,
                    folds=(),
                    pair_contrast_rows=(),
                    equal_group_pair_contrasts=outcome.equal_group_pair_contrasts,
                    result=result,
                )
                executor._register_owned_execution(state.owner, entry)
            else:
                record.producer.results[trial.valid_index] = result
                if required_mode == "synthetic":
                    executor._run_external_validator(
                        record.producer,
                        record.validator,
                        result,
                    )
                    entry = executor.ExternalExecutionReceipt(
                        result,
                        _seal=executor._EXTERNAL_EXECUTION_RECEIPT_SEAL,
                    )
                    executor._register_external_receipt_identity(
                        entry,
                        (
                            weakref.ref(state.owner),
                            weakref.ref(state.owner.prepared),
                            weakref.ref(record.producer),
                            weakref.ref(record.validator),
                            executor._deep_integrity_digest(entry),
                        ),
                    )
                else:
                    entry = executor.authenticate_external_execution(
                        state.owner,
                        record.producer,
                        record.validator,
                        result,
                    )
            return _commit_valid(state, entry, result, id(authority))
        except Exception as error:
            _seal_integrity_stop(state)
            raise ObservedSessionError(
                "returned scored result failed atomic commit integrity"
            ) from error
    finally:
        state.optional_capability_issuance_open = False
        state.permutation_capability_issuance_open = False
        state.rademacher_capability_issuance_open = False
        state.in_flight = False


def _seal_integrity_stop(state: _SessionState) -> None:
    if state.status != TECHNICAL_INTEGRITY_STOP:
        state.version += 1
        state.status = TECHNICAL_INTEGRITY_STOP
        state.scientific_conclusion = None


def _seal_resource_stop(
    session: ControllerRunSession,
    reason: str,
    *,
    _internal: bool = False,
) -> None:
    state = _require_session(session)
    if state.status != RUNNING:
        raise ObservedSessionError("controller session is already sealed")
    if state.in_flight and not _internal:
        raise ObservedSessionError("cannot externally seal during an in-flight attempt")
    if type(reason) is not str or not reason.strip():
        raise ObservedSessionError("resource stop reason must be a nonempty string")
    state.version += 1
    state.status = TECHNICAL_RESOURCE_STOP
    state.scientific_conclusion = None


def _normalized_history_results(
    state: _SessionState,
) -> tuple[controller.TrialResult, ...]:
    raw = tuple(
        _authenticated_entry_result(state, entry)
        for entry in state.history
    )
    if len(raw) < controller.COVERAGE_TRIALS:
        return raw
    evidence = state.coverage_evidence
    if type(evidence) is not controller.CoverageEvidence:
        raise ObservedSessionError("post-core history lacks exact coverage evidence")
    bound_sixteenth = replace(
        raw[controller.COVERAGE_TRIALS - 1],
        coverage_binding=evidence.binding,
    )
    if (
        *raw[: controller.COVERAGE_TRIALS - 1],
        bound_sixteenth,
    ) != evidence.core_results:
        raise ObservedSessionError("coverage evidence differs from session-owned core")
    incumbent = controller.choose_incumbent(evidence.core_results)
    incumbent_entry = state.history[incumbent.trial.valid_index - 1]
    if (
        type(incumbent_entry) is not executor.TrialExecution
        or incumbent_entry.equal_group_pair_contrasts
        != evidence.equal_group_pair_contrasts
    ):
        raise ObservedSessionError("coverage contrasts differ from the owned core incumbent")
    return (*evidence.core_results, *raw[controller.COVERAGE_TRIALS :])


def _authenticated_entry_result(
    state: _SessionState,
    entry: executor.TrialExecution | executor.ExternalExecutionReceipt,
) -> controller.TrialResult:
    if state.mode != "synthetic":
        return executor._authenticated_history_result(state.owner, entry)
    if type(entry) is executor.TrialExecution:
        owner_reference, cache_reference, expected_digest = (
            executor._lookup_execution_identity(entry)
        )
        if (
            owner_reference() is not state.owner
            or cache_reference() is not state.owner.prepared
            or executor._deep_integrity_digest(entry) != expected_digest
        ):
            raise ObservedSessionError("synthetic owned execution changed authority")
        return entry.result
    if type(entry) is executor.ExternalExecutionReceipt:
        (
            owner_reference,
            cache_reference,
            producer_reference,
            validator_reference,
            expected_digest,
        ) = executor._lookup_external_receipt_identity(entry)
        producer = producer_reference()
        validator = validator_reference()
        if (
            owner_reference() is not state.owner
            or cache_reference() is not state.owner.prepared
            or producer is None
            or validator is None
            or executor._deep_integrity_digest(entry) != expected_digest
        ):
            raise ObservedSessionError("synthetic external receipt changed authority")
        executor._run_external_validator(
            producer,
            validator,
            entry.result,
            execution_detail=entry.execution_detail,
        )
        if executor._deep_integrity_digest(entry) != expected_digest:
            raise ObservedSessionError("synthetic receipt changed during revalidation")
        return entry.result
    raise ObservedSessionError("session history contains an unauthenticated entry")


def _reauthenticate_history(state: _SessionState) -> None:
    for scope, (identity, reference) in state.authority_bindings.items():
        authority = reference()
        if authority is None or id(authority) != identity:
            raise ObservedSessionError(
                f"{scope} adapter authority no longer has its issued identity"
            )
        record = _lookup_adapter(authority)
        bound_session = record.session_reference()
        if (
            bound_session is None
            or _lookup_session(bound_session) is not state
            or scope not in record.scopes
        ):
            raise ObservedSessionError(
                f"{scope} adapter session binding no longer exists"
            )
        if record.adapter_kind == "synthetic_callback":
            if record.callback_reference is None or record.callback_reference() is None:
                raise ObservedSessionError(
                    f"{scope} adapter callback no longer exists"
                )
        elif record.adapter_kind == "optional_family_v1":
            if (
                scope != "optional"
                or record.named_adapter_reference is None
                or record.named_adapter_reference() is None
            ):
                raise ObservedSessionError("named optional adapter no longer exists")
        elif record.adapter_kind == "whole_set_permutation_v1":
            if (
                scope != "permutation"
                or record.named_adapter_reference is None
                or record.named_adapter_reference() is None
            ):
                raise ObservedSessionError("named permutation adapter no longer exists")
        elif record.adapter_kind == "rademacher_reference_v1":
            if (
                scope != "rademacher"
                or record.named_adapter_reference is None
                or record.named_adapter_reference() is None
            ):
                raise ObservedSessionError("named Rademacher adapter no longer exists")
        else:
            raise ObservedSessionError("session contains an unknown adapter kind")
    if len(state.schedule) != len(state.schedule_digests) or not state.schedule:
        raise ObservedSessionError("derived schedule signature inventory changed")
    for index, (trial, expected_digest) in enumerate(
        zip(state.schedule, state.schedule_digests, strict=True),
        start=1,
    ):
        if (
            type(trial) is not controller.Trial
            or trial.valid_index != index
            or executor._deep_integrity_digest(trial) != expected_digest
        ):
            raise ObservedSessionError("derived schedule changed after issuance")
    if not len(state.results) <= len(state.schedule):
        raise ObservedSessionError("session has more results than derived trials")
    if not (
        len(state.history)
        == len(state.history_digests)
        == len(state.results)
        == len(state.invalid_counts_at_results)
    ):
        raise ObservedSessionError("session ledger lengths diverged")
    if tuple(result.trial.valid_index for result in state.results) != tuple(
        range(1, len(state.results) + 1)
    ):
        raise ObservedSessionError("session result history is not contiguous")
    if any(
        result.trial is not state.schedule[index]
        for index, result in enumerate(state.results)
    ):
        raise ObservedSessionError("session results differ from the derived schedule")
    for entry, expected_digest in zip(
        state.history,
        state.history_digests,
        strict=True,
    ):
        if executor._deep_integrity_digest(entry) != expected_digest:
            raise ObservedSessionError("a committed execution changed after issuance")
    normalized = _normalized_history_results(state)
    if len(normalized) != len(state.results) or any(
        observed is not expected
        for observed, expected in zip(state.results, normalized, strict=True)
    ):
        raise ObservedSessionError("canonical session results changed identity")
    if state.coverage_evidence is not None:
        if (
            state.coverage_evidence_digest is None
            or executor._deep_integrity_digest(state.coverage_evidence)
            != state.coverage_evidence_digest
        ):
            raise ObservedSessionError("coverage evidence changed after binding")
    if len(state.attempts) != len(state.attempt_digests):
        raise ObservedSessionError("attempt ledger signature inventory changed")
    valid_index = 0
    invalid_count = 0
    production_attempt_count = 0
    optional_production_attempt_count = 0
    permutation_production_attempt_count = 0
    rademacher_production_attempt_count = 0
    for ordinal, (attempt, expected_digest) in enumerate(
        zip(state.attempts, state.attempt_digests, strict=True),
        start=1,
    ):
        if (
            type(attempt) is not _Attempt
            or attempt.ordinal != ordinal
            or executor._deep_integrity_digest(attempt) != expected_digest
            or attempt.outcome not in {"valid", "invalid", "resource"}
            or attempt.trial is not state.schedule[valid_index]
        ):
            raise ObservedSessionError("attempt chronology or signature changed")
        scope = _trial_scope(attempt.trial)
        authority_scope = _authority_scope(attempt.trial, state.mode)
        binding = state.authority_bindings.get(authority_scope)
        expected_authority = None if binding is None else binding[0]
        if attempt.authority_identity != expected_authority:
            raise ObservedSessionError("attempt used a different scope authority")
        if state.mode == "production" and scope == "core":
            capability = attempt.execution_capability
            claim = state.production_claim
            if (
                type(capability) is not executor.ProductionExecutionCapability
                or claim is None
                or type(attempt.execution_capability_digest) is not bytes
                or attempt.production_attempt_ordinal != production_attempt_count + 1
            ):
                raise ObservedSessionError(
                    "production core attempt lacks its exact one-shot capability"
                )
            try:
                observed_ordinal, observed_digest = (
                    executor._require_acknowledged_production_execution(
                        state.owner,
                        _session_for_state(state),
                        claim,
                        capability,
                        attempt,
                    )
                )
            except executor.ProductionExecutionAuthorityError as error:
                raise ObservedSessionError(
                    "production attempt capability acknowledgment changed"
                ) from error
            if (
                observed_ordinal != attempt.production_attempt_ordinal
                or observed_digest != attempt.execution_capability_digest
            ):
                raise ObservedSessionError(
                    "production attempt capability binding changed"
                )
            production_attempt_count += 1
        elif state.mode == "production" and scope == "optional":
            capability = attempt.execution_capability
            optional_adapter, authority, adapter = _optional_attempt_context(
                state,
                capability,
            )
            if (
                not optional_adapter._is_optional_attempt_capability(capability)
                or type(attempt.execution_capability_digest) is not bytes
                or attempt.production_attempt_ordinal
                != optional_production_attempt_count + 1
            ):
                raise ObservedSessionError(
                    "production optional attempt lacks its exact one-shot capability"
                )
            try:
                observed_ordinal, observed_digest = (
                    optional_adapter._require_acknowledged_optional_attempt(
                        adapter,
                        _session_for_state(state),
                        authority,
                        capability,
                        attempt,
                    )
                )
            except Exception as error:
                raise ObservedSessionError(
                    "optional attempt capability acknowledgment changed"
                ) from error
            if (
                observed_ordinal != attempt.production_attempt_ordinal
                or observed_digest != attempt.execution_capability_digest
            ):
                raise ObservedSessionError(
                    "optional attempt capability binding changed"
                )
            optional_production_attempt_count += 1
        elif state.mode == "production" and authority_scope == "permutation":
            capability = attempt.execution_capability
            permutation_adapter, authority, adapter = _permutation_attempt_context(
                state,
                capability,
            )
            if (
                not permutation_adapter._is_permutation_attempt_capability(capability)
                or type(attempt.execution_capability_digest) is not bytes
                or attempt.production_attempt_ordinal
                != permutation_production_attempt_count + 1
            ):
                raise ObservedSessionError(
                    "production permutation attempt lacks its exact one-shot capability"
                )
            try:
                observed_ordinal, observed_digest = (
                    permutation_adapter._require_acknowledged_permutation_attempt(
                        adapter,
                        _session_for_state(state),
                        authority,
                        capability,
                        attempt,
                    )
                )
            except Exception as error:
                raise ObservedSessionError(
                    "permutation attempt capability acknowledgment changed"
                ) from error
            if (
                observed_ordinal != attempt.production_attempt_ordinal
                or observed_digest != attempt.execution_capability_digest
            ):
                raise ObservedSessionError(
                    "permutation attempt capability binding changed"
                )
            permutation_production_attempt_count += 1
        elif state.mode == "production" and authority_scope == "rademacher":
            capability = attempt.execution_capability
            rademacher_adapter, authority, adapter = _rademacher_attempt_context(
                state,
                capability,
            )
            if (
                not rademacher_adapter._is_rademacher_attempt_capability(capability)
                or type(attempt.execution_capability_digest) is not bytes
                or attempt.production_attempt_ordinal
                != rademacher_production_attempt_count + 1
            ):
                raise ObservedSessionError(
                    "production Rademacher attempt lacks its exact one-shot capability"
                )
            try:
                observed_ordinal, observed_digest = (
                    rademacher_adapter._require_acknowledged_rademacher_attempt(
                        adapter,
                        _session_for_state(state),
                        authority,
                        capability,
                        attempt,
                    )
                )
            except Exception as error:
                raise ObservedSessionError(
                    "Rademacher attempt capability acknowledgment changed"
                ) from error
            if (
                observed_ordinal != attempt.production_attempt_ordinal
                or observed_digest != attempt.execution_capability_digest
            ):
                raise ObservedSessionError(
                    "Rademacher attempt capability binding changed"
                )
            rademacher_production_attempt_count += 1
        elif (
            attempt.execution_capability is not None
            or attempt.execution_capability_digest is not None
            or attempt.production_attempt_ordinal is not None
        ):
            raise ObservedSessionError(
                "nonproduction or external attempt carries a core capability"
            )
        if attempt.outcome == "invalid":
            invalid_count += 1
            if (
                attempt.entry_identity is not None
                or attempt.error_type is None
                or attempt.invalid_attempts_after != invalid_count
            ):
                raise ObservedSessionError("invalid attempt record is malformed")
            continue
        if attempt.outcome == "resource":
            if (
                attempt.entry_identity is not None
                or attempt.error_type is None
                or attempt.invalid_attempts_after != invalid_count
                or ordinal != len(state.attempts)
                or state.status not in {RUNNING, TECHNICAL_RESOURCE_STOP}
                or (state.status == RUNNING and not state.in_flight)
            ):
                raise ObservedSessionError("resource attempt record is malformed")
            continue
        if valid_index >= len(state.history):
            raise ObservedSessionError("attempt ledger has an extra valid result")
        entry = state.history[valid_index]
        if (
            attempt.entry_identity != id(entry)
            or attempt.error_type is not None
            or attempt.invalid_attempts_after != invalid_count
            or state.invalid_counts_at_results[valid_index] != invalid_count
        ):
            raise ObservedSessionError("valid attempt interleaving changed")
        valid_index += 1
    if valid_index != len(state.history) or invalid_count != state.invalid_attempts:
        raise ObservedSessionError("attempt totals differ from the committed history")
    if production_attempt_count != state.production_attempts:
        raise ObservedSessionError(
            "production capability count differs from the session attempt ledger"
        )
    if optional_production_attempt_count != state.optional_production_attempts:
        raise ObservedSessionError(
            "optional capability count differs from the session attempt ledger"
        )
    if permutation_production_attempt_count != state.permutation_production_attempts:
        raise ObservedSessionError(
            "permutation capability count differs from the session attempt ledger"
        )
    if rademacher_production_attempt_count != state.rademacher_production_attempts:
        raise ObservedSessionError(
            "Rademacher capability count differs from the session attempt ledger"
        )


def _terminal_replay(
    state: _SessionState,
) -> tuple[
    str,
    str,
    controller.TrialResult,
    executor.TrialExecution | executor.ExternalExecutionReceipt,
    controller.Configuration,
]:
    _reauthenticate_history(state)
    if state.status not in {ORDINARY_PATIENCE_STOP, INCOMPLETE_BUDGET_STOP}:
        raise ObservedSessionError("session did not end in a selectable disposition")
    if state.scientific_conclusion not in {
        ORDINARY_CONCLUSION,
        INCOMPLETE_CONCLUSION,
    }:
        raise ObservedSessionError("terminal scientific conclusion is missing")
    evidence = state.coverage_evidence
    if type(evidence) is not controller.CoverageEvidence:
        raise ObservedSessionError("terminal history lacks coverage evidence")
    results = tuple(state.results)
    try:
        controller.validate_result_history(results, coverage_evidence=evidence)
    except (controller.ControllerError, TypeError, ValueError) as error:
        raise ObservedSessionError("terminal history fails controller validation") from error
    for length in range(controller.MINIMUM_VALID_TRIALS, len(results)):
        disposition = controller.stop_decision(
            results[:length],
            state.invalid_counts_at_results[length - 1],
            evidence,
        )
        if not (disposition == "continue" or disposition.startswith("continue_")):
            raise ObservedSessionError("history continued past its earliest stop")
    disposition = controller.stop_decision(
        results,
        state.invalid_counts_at_results[-1],
        evidence,
    )
    if disposition != state.status:
        raise ObservedSessionError("sealed disposition differs from controller replay")
    if disposition == ORDINARY_PATIENCE_STOP:
        if len(results) >= controller.MAXIMUM_VALID_TRIALS:
            raise ObservedSessionError("ordinary terminal history reached the cap")
        conclusion = ORDINARY_CONCLUSION
    elif disposition == INCOMPLETE_BUDGET_STOP:
        if len(results) != controller.MAXIMUM_VALID_TRIALS:
            raise ObservedSessionError("incomplete terminal history is not exact-cap")
        conclusion = INCOMPLETE_CONCLUSION
    else:
        raise ObservedSessionError("technical or continuing history cannot select a refit")
    if conclusion != state.scientific_conclusion:
        raise ObservedSessionError("scientific conclusion changed after sealing")
    winner = controller.choose_incumbent(results)
    rademacher_results = tuple(
        result for result in results if result.trial.kind == "rademacher_noise"
    )
    if (
        len(rademacher_results) != len(RADEMACHER_REFERENCE_SEEDS)
        or tuple(result.trial.seed for result in rademacher_results)
        != RADEMACHER_REFERENCE_SEEDS
        or any(not result.valid or not result.trial.falsifier for result in rademacher_results)
    ):
        raise ObservedSessionError(
            "terminal history lacks the exact seven valid Rademacher references"
        )
    maximum_rademacher_gain = max(
        result.gain_bits_per_cell for result in rademacher_results
    )
    if (
        winner.gain_bits_per_cell - maximum_rademacher_gain
        < RADEMACHER_SPECIFICITY_MARGIN_BITS
    ):
        raise ObservedSessionError(
            "selected association does not clear the frozen Rademacher specificity margin"
        )
    configuration = winner.trial.configuration
    if type(configuration) is not controller.Configuration:
        raise ObservedSessionError("controller incumbent lacks a configuration")
    winner_entry = state.history[winner.trial.valid_index - 1]
    return disposition, conclusion, winner, winner_entry, configuration


def authorize_terminal_selection(
    session: ControllerRunSession,
) -> TerminalSelectionAuthorization:
    """Issue a once-only terminal selection from the exact sealed ledger."""
    state = _require_session(session)
    if state.in_flight:
        raise ObservedSessionError("cannot authorize during an in-flight attempt")
    if state.terminal_issued:
        raise ObservedSessionError("terminal selection was already authorized")
    disposition, conclusion, winner, winner_entry, configuration = _terminal_replay(
        state
    )
    evidence = state.coverage_evidence
    assert evidence is not None and state.coverage_evidence_digest is not None
    authorization = TerminalSelectionAuthorization(
        _seal=_TERMINAL_AUTHORIZATION_SEAL
    )
    snapshot = _TerminalSnapshot(
        session_reference=weakref.ref(session),
        owner_reference=weakref.ref(state.owner),
        cache_reference=weakref.ref(state.owner.prepared),
        session_version=state.version,
        maximum_iterations=state.maximum_iterations,
        history_identities=tuple(id(entry) for entry in state.history),
        history_digests=tuple(state.history_digests),
        schedule_digests=tuple(state.schedule_digests),
        evidence_reference=weakref.ref(evidence),
        evidence_digest=state.coverage_evidence_digest,
        invalid_counts_at_results=tuple(state.invalid_counts_at_results),
        attempt_digests=tuple(state.attempt_digests),
        disposition=disposition,
        scientific_conclusion=conclusion,
        winner_entry_reference=weakref.ref(winner_entry),
        winner_result_reference=weakref.ref(winner),
        configuration_reference=weakref.ref(configuration),
        configuration_signature=_configuration_signature(configuration),
        valid_trial_count=len(state.results),
        invalid_attempt_count=state.invalid_attempts,
    )
    _register_terminal(authorization, snapshot)
    state.terminal_reference = weakref.ref(authorization)
    state.terminal_issued = True
    return authorization


def require_terminal_selection(
    owner: executor.CoreTrialExecutor,
    authorization: TerminalSelectionAuthorization,
) -> controller.TrialResult:
    """Reauthenticate a terminal ledger and return its canonical incumbent."""
    if type(owner) is not executor.CoreTrialExecutor:
        raise ObservedSessionError("terminal selection owner has the wrong exact type")
    if (
        type(authorization) is not TerminalSelectionAuthorization
        or authorization._seal is not _TERMINAL_AUTHORIZATION_SEAL
    ):
        raise ObservedSessionError("terminal selection lacks module authority")
    snapshot = _lookup_terminal(authorization)
    session = snapshot.session_reference()
    if session is None:
        raise ObservedSessionError("terminal controller session no longer exists")
    state = _require_session(session)
    if (
        snapshot.owner_reference() is not owner
        or state.owner is not owner
        or snapshot.cache_reference() is not owner.prepared
        or snapshot.session_version != state.version
        or snapshot.maximum_iterations != state.maximum_iterations
        or snapshot.history_identities != tuple(id(entry) for entry in state.history)
        or snapshot.history_digests != tuple(state.history_digests)
        or snapshot.schedule_digests != tuple(state.schedule_digests)
        or snapshot.invalid_counts_at_results
        != tuple(state.invalid_counts_at_results)
        or snapshot.attempt_digests != tuple(state.attempt_digests)
        or snapshot.valid_trial_count != len(state.results)
        or snapshot.invalid_attempt_count != state.invalid_attempts
    ):
        raise ObservedSessionError("terminal selection owner, cache, or ledger changed")
    evidence = snapshot.evidence_reference()
    if (
        evidence is None
        or evidence is not state.coverage_evidence
        or executor._deep_integrity_digest(evidence) != snapshot.evidence_digest
    ):
        raise ObservedSessionError("terminal selection evidence changed")
    disposition, conclusion, winner, winner_entry, configuration = _terminal_replay(
        state
    )
    if (
        disposition != snapshot.disposition
        or conclusion != snapshot.scientific_conclusion
        or snapshot.winner_entry_reference() is not winner_entry
        or snapshot.winner_result_reference() is not winner
        or snapshot.configuration_reference() is not configuration
        or snapshot.configuration_signature != _configuration_signature(configuration)
    ):
        raise ObservedSessionError("terminal incumbent or configuration changed")
    return winner


def require_selected_refit_authorization(
    owner: executor.CoreTrialExecutor,
    authorization: TerminalSelectionAuthorization,
) -> controller.TrialResult:
    """Return a refit incumbent only for an ordinary prospective stop.

    An exact-cap authorization remains useful for reporting its canonical
    incumbent and resource statistics, but it is explicitly incomplete and is
    not a scientific selected-refit capability.
    """
    winner = require_terminal_selection(owner, authorization)
    snapshot = _lookup_terminal(authorization)
    if (
        snapshot.disposition != ORDINARY_PATIENCE_STOP
        or snapshot.scientific_conclusion != ORDINARY_CONCLUSION
    ):
        raise ObservedSessionError(
            "incomplete or technical disposition cannot authorize selected refit"
        )
    return winner


def _selected_refit_binding(
    owner: executor.CoreTrialExecutor,
    terminal: TerminalSelectionAuthorization,
    authority: SelectedRefitProcedureAuthorization,
) -> tuple[_SelectedRefitBinding, controller.TrialResult]:
    winner = require_selected_refit_authorization(owner, terminal)
    if (
        type(authority) is not SelectedRefitProcedureAuthorization
        or authority._seal is not _SELECTED_REFIT_AUTHORIZATION_SEAL
    ):
        raise ObservedSessionError("selected-refit procedure authority is invalid")
    binding = _lookup_selected_refit(authority)
    if type(binding) is not _SelectedRefitBinding:
        raise ObservedSessionError("selected-refit procedure binding is malformed")
    session = binding.session_reference()
    source = binding.source
    if session is None:
        raise ObservedSessionError("selected-refit controller session is unavailable")
    state = _require_session(session)
    if (
        binding.owner_reference() is not owner
        or binding.terminal_reference() is not terminal
        or binding.winner_reference() is not winner
        or state.owner is not owner
        or state.selected_refit_source is not source
        or state.selected_refit_authority_reference is None
        or state.selected_refit_authority_reference() is not authority
        or source.candidates_reference() is not owner.prepared.candidates
        or source.geometry_reference() is not owner.prepared.geometry
        or source.pairs_reference() is not owner.prepared.pairs
        or executor._deep_integrity_digest(source.cells) != source.cells_digest
        or executor._deep_integrity_digest(source.moments) != source.moments_digest
    ):
        raise ObservedSessionError(
            "selected-refit procedure source or terminal binding changed"
        )
    return binding, winner


def _selected_refit_binding_identity(
    owner: executor.CoreTrialExecutor,
    terminal: TerminalSelectionAuthorization,
    authority: SelectedRefitProcedureAuthorization,
) -> _SelectedRefitBinding:
    """Resolve only the unforgeable registry identity used to claim a stage.

    Deep source, terminal, and result integrity checks deliberately happen
    *after* the matching phase is atomically moved to ``*_in_flight``.  Thus a
    genuine capability cannot be restored and retried after a detected
    mutation.
    """

    if (
        type(authority) is not SelectedRefitProcedureAuthorization
        or authority._seal is not _SELECTED_REFIT_AUTHORIZATION_SEAL
    ):
        raise ObservedSessionError("selected-refit procedure authority is invalid")
    binding = _lookup_selected_refit(authority)
    if (
        type(binding) is not _SelectedRefitBinding
        or binding.owner_reference() is not owner
        or binding.terminal_reference() is not terminal
    ):
        raise ObservedSessionError(
            "selected-refit procedure authority belongs to another terminal"
        )
    return binding


def _selected_refit_module_request(name: str, *arguments: Any) -> Any:
    module = sys.modules.get("v3_m2_selected_refit_v1")
    expected_path = TOOLS_DIR / "v3_m2_selected_refit_v1.py"
    if (
        module is None
        or Path(os.path.abspath(os.fspath(getattr(module, "SCRIPT_PATH", ""))))
        != expected_path
    ):
        raise ObservedSessionError(
            "selected-refit objective requires the named coordinator module"
        )
    callback = getattr(module, name, None)
    if not callable(callback):
        raise ObservedSessionError(
            "selected-refit objective authority handshake is unavailable"
        )
    try:
        return callback(*arguments)
    except ObservedSessionError:
        raise
    except Exception as error:
        raise ObservedSessionError(
            "selected-refit coordinator rejected its objective token"
        ) from error


def authorize_selected_refit_procedure(
    owner: executor.CoreTrialExecutor,
    terminal: TerminalSelectionAuthorization,
) -> SelectedRefitProcedureAuthorization:
    """Atomically reserve the only completion capability for one terminal."""

    winner = require_selected_refit_authorization(owner, terminal)
    snapshot = _lookup_terminal(terminal)
    session = snapshot.session_reference()
    if session is None:
        raise ObservedSessionError("terminal controller session no longer exists")
    state = _require_session(session)
    source = state.selected_refit_source
    if type(source) is not _SelectedRefitSource:
        raise ObservedSessionError(
            "observed session lacks prospectively bound selected-refit inputs"
        )
    with _SELECTED_REFIT_ISSUANCE_LOCK:
        if state.selected_refit_authority_issued:
            raise ObservedSessionError(
                "selected-refit procedure authority was already issued"
            )
        authority = SelectedRefitProcedureAuthorization(
            _seal=_SELECTED_REFIT_AUTHORIZATION_SEAL
        )
        binding = _SelectedRefitBinding(
            session_reference=weakref.ref(session),
            owner_reference=weakref.ref(owner),
            terminal_reference=weakref.ref(terminal),
            winner_reference=weakref.ref(winner),
            source=source,
        )
        _register_selected_refit(authority, binding)
        state.selected_refit_authority_reference = weakref.ref(authority)
        state.selected_refit_authority_issued = True
    return authority


def _begin_selected_refit_full_m1_request(
    owner: executor.CoreTrialExecutor,
    terminal: TerminalSelectionAuthorization,
    authority: SelectedRefitProcedureAuthorization,
) -> bytes:
    """Consume the prospective gate before any fresh M1 work begins."""

    binding = _selected_refit_binding_identity(owner, terminal, authority)
    with binding.lock:
        if binding.phase != "issued_for_full_m1":
            if binding.phase != "complete":
                binding.phase = "burned"
                binding.full_objective_guard = None
                binding.hybrid_receipt_guard = None
                binding.precursor_guard = None
                binding.cold_guard = None
                binding.warm_guard = None
            raise ObservedSessionError(
                "fresh full-M1 request was duplicated or is off chronology"
            )
        binding.phase = "full_m1_in_flight"
    try:
        authenticated, winner = _selected_refit_binding(
            owner, terminal, authority
        )
        if authenticated is not binding:
            raise ObservedSessionError("selected-refit procedure binding changed")
        source = binding.source
        return executor._authority_binding_digest(
            "selected_refit_full_m1_in_flight_v1",
            id(owner),
            id(terminal),
            id(authority),
            id(winner),
            source.cells_digest,
            source.moments_digest,
            id(source.candidates_reference()),
            id(source.geometry_reference()),
            id(source.pairs_reference()),
        )
    except BaseException:
        with binding.lock:
            binding.phase = "burned"
        raise


def _selected_refit_source_request(
    owner: executor.CoreTrialExecutor,
    terminal: TerminalSelectionAuthorization,
    authority: SelectedRefitProcedureAuthorization,
) -> tuple[tuple[Any, ...], tuple[Any, ...], Any, Any, Any]:
    """Return only the source already bound before observed search began."""

    binding, _ = _selected_refit_binding(owner, terminal, authority)
    source = binding.source
    with binding.lock:
        if binding.phase != "full_m1_in_flight":
            raise ObservedSessionError(
                "selected-refit source is unavailable outside fresh-M1 execution"
            )
        candidates = source.candidates_reference()
        geometry = source.geometry_reference()
        pairs = source.pairs_reference()
        if candidates is None or geometry is None or pairs is None:
            raise ObservedSessionError("selected-refit source libraries are unavailable")
        return source.cells, source.moments, candidates, geometry, pairs


def _complete_selected_refit_full_m1_request(
    owner: executor.CoreTrialExecutor,
    terminal: TerminalSelectionAuthorization,
    authority: SelectedRefitProcedureAuthorization,
    full_objective: object,
) -> bytes:
    """Register only the objective produced after the prospective begin gate."""

    import fit_v3_m1 as m1
    import fit_v3_m2 as m2

    binding, winner = _selected_refit_binding(owner, terminal, authority)
    source = binding.source
    fixed_m1_theta, prepared, audit_evidence, token_digest = (
        _selected_refit_module_request(
            "_full_objective_authority_record",
            full_objective,
            authority,
            owner,
            terminal,
        )
    )
    candidates = source.candidates_reference()
    geometry = source.geometry_reference()
    pairs = source.pairs_reference()
    if candidates is None or geometry is None or pairs is None:
        raise ObservedSessionError("selected-refit source libraries are unavailable")
    try:
        m2._require_physical_fold_provenance(
            prepared,
            source.cells,
            source.moments,
            candidates,
            geometry,
            pairs,
            tuple(range(executor.EXPECTED_PRODUCTION_ROWS)),
        )
    except m2.M2Error as error:
        raise ObservedSessionError(
            "full objective lacks same-pass physical provenance"
        ) from error
    with binding.lock:
        if binding.phase != "full_m1_in_flight":
            raise ObservedSessionError(
                "fresh full-M1 completion is outside its in-flight phase"
            )
        if (
            type(fixed_m1_theta) is not np.ndarray
            or type(prepared) is not m2.PreparedM2
            or len(prepared.cells) != executor.EXPECTED_PRODUCTION_ROWS
            or prepared.transform.rank != executor.EXPECTED_PRODUCTION_RANK
            or prepared.m1_data.cells is not prepared.cells
            or any(
                supplied is not expected
                for supplied, expected in zip(
                    prepared.cells, source.cells, strict=True
                )
            )
            or getattr(fixed_m1_theta, "shape", None) != (m1.PARAMETER_DIMENSION,)
        ):
            raise ObservedSessionError(
                "fresh full-M1 completion produced a malformed objective"
            )
        digest = executor._authority_binding_digest(
            "selected_refit_full_objective_v1",
            id(owner),
            id(terminal),
            id(authority),
            id(winner),
            id(full_objective),
            token_digest,
            id(fixed_m1_theta),
            id(prepared),
            executor._deep_integrity_digest(fixed_m1_theta),
            executor._deep_integrity_digest(prepared),
            executor._deep_integrity_digest(audit_evidence),
        )
        binding.full_objective_reference = weakref.ref(full_objective)
        binding.full_objective_guard = full_objective
        binding.fixed_m1_reference = weakref.ref(fixed_m1_theta)
        binding.prepared_reference = weakref.ref(prepared)
        binding.full_objective_digest = digest
        binding.fixed_m1_digest = executor._deep_integrity_digest(fixed_m1_theta)
        binding.prepared_digest = executor._deep_integrity_digest(prepared)
        binding.audit_digest = executor._deep_integrity_digest(audit_evidence)
        binding.phase = "full_m1_ready"
        return digest


def _require_selected_refit_full_objective(
    owner: executor.CoreTrialExecutor,
    terminal: TerminalSelectionAuthorization,
    authority: SelectedRefitProcedureAuthorization,
    full_objective: object,
) -> bytes:
    binding, winner = _selected_refit_binding(owner, terminal, authority)
    fixed_m1_theta, prepared, audit_evidence, token_digest = (
        _selected_refit_module_request(
            "_full_objective_authority_record",
            full_objective,
            authority,
            owner,
            terminal,
        )
    )
    with binding.lock:
        if binding.phase not in {
            "full_m1_ready",
            "precursor_in_flight",
            "precursor_ready",
            "cold_in_flight",
            "cold_ready",
            "warm_in_flight",
            "warm_ready",
            "complete",
        }:
            raise ObservedSessionError(
                "selected-refit full objective is not ready"
            )
        if (
            binding.full_objective_reference is None
            or binding.full_objective_guard is not full_objective
            or binding.fixed_m1_reference is None
            or binding.prepared_reference is None
            or binding.full_objective_reference() is not full_objective
            or binding.fixed_m1_reference() is not fixed_m1_theta
            or binding.prepared_reference() is not prepared
            or binding.fixed_m1_digest
            != executor._deep_integrity_digest(fixed_m1_theta)
            or binding.prepared_digest != executor._deep_integrity_digest(prepared)
            or binding.audit_digest != executor._deep_integrity_digest(audit_evidence)
        ):
            raise ObservedSessionError(
                "selected-refit full objective identity or integrity changed"
            )
        expected = executor._authority_binding_digest(
            "selected_refit_full_objective_v1",
            id(owner),
            id(terminal),
            id(authority),
            id(winner),
            id(full_objective),
            token_digest,
            id(fixed_m1_theta),
            id(prepared),
            binding.fixed_m1_digest,
            binding.prepared_digest,
            binding.audit_digest,
        )
        if expected != binding.full_objective_digest:
            raise ObservedSessionError("selected-refit objective authority changed")
        return expected


def _bind_selected_refit_hybrid_receipt(
    owner: executor.CoreTrialExecutor,
    terminal: TerminalSelectionAuthorization,
    authority: SelectedRefitProcedureAuthorization,
    full_objective: object,
    receipt: object,
    authorization_digest: bytes,
) -> None:
    _require_selected_refit_full_objective(
        owner,
        terminal,
        authority,
        full_objective,
    )
    binding, _ = _selected_refit_binding(owner, terminal, authority)
    with binding.lock:
        if (
            binding.phase != "full_m1_ready"
            or not binding.hybrid_reserved
            or binding.hybrid_receipt_reference is not None
            or binding.hybrid_receipt_guard is not None
            or type(authorization_digest) is not bytes
        ):
            raise ObservedSessionError(
                "terminal hybrid receipt binding is invalid or duplicated"
            )
        binding.hybrid_receipt_reference = weakref.ref(receipt)
        binding.hybrid_receipt_guard = receipt
        binding.hybrid_receipt_digest = executor._authority_binding_digest(
            "selected_refit_hybrid_receipt_v1",
            id(receipt),
            authorization_digest,
        )


def _begin_selected_refit_fit_stage(
    owner: executor.CoreTrialExecutor,
    terminal: TerminalSelectionAuthorization,
    authority: SelectedRefitProcedureAuthorization,
    full_objective: object,
    stage: str,
    *,
    hybrid_receipt: object | None = None,
    prior_result: object | None = None,
) -> None:
    binding = _selected_refit_binding_identity(owner, terminal, authority)
    with binding.lock:
        if stage not in {"precursor", "cold", "warm"}:
            raise ObservedSessionError("unknown selected-refit fit stage")
        phase_before_claim = binding.phase
        if stage == "precursor":
            phase_is_expected = phase_before_claim == "full_m1_ready"
            claimed_phase = "precursor_in_flight"
        elif stage == "cold":
            phase_is_expected = phase_before_claim in {
                "full_m1_ready",
                "precursor_ready",
            }
            claimed_phase = "cold_in_flight"
        else:
            phase_is_expected = phase_before_claim == "cold_ready"
            claimed_phase = "warm_in_flight"
        if not phase_is_expected:
            if binding.phase != "complete":
                binding.phase = "burned"
            raise ObservedSessionError(
                "selected-refit fit stage is off chronology"
            )
        binding.phase = claimed_phase

    try:
        authenticated, winner = _selected_refit_binding(
            owner, terminal, authority
        )
        if authenticated is not binding:
            raise ObservedSessionError("selected-refit procedure binding changed")
        _require_selected_refit_full_objective(
            owner,
            terminal,
            authority,
            full_objective,
        )
        is_hybrid = (
            winner.trial.configuration.family == "sparse_plus_low_rank"
        )
        if is_hybrid != binding.hybrid_reserved:
            raise ObservedSessionError(
                "selected-refit family and reserved authority disagree"
            )
        if is_hybrid:
            if (
                binding.hybrid_receipt_reference is None
                or binding.hybrid_receipt_reference() is not hybrid_receipt
                or binding.hybrid_receipt_guard is not hybrid_receipt
                or binding.hybrid_receipt_digest is None
            ):
                raise ObservedSessionError(
                    "selected hybrid stage lacks its exact reserved receipt"
                )
        elif hybrid_receipt is not None:
            raise ObservedSessionError(
                "selected nonhybrid stage carries hybrid authority"
            )
        if stage == "precursor":
            if not is_hybrid or prior_result is not None:
                raise ObservedSessionError(
                    "hybrid precursor has the wrong family or initializer"
                )
        elif stage == "cold":
            if is_hybrid:
                if phase_before_claim != "precursor_ready":
                    raise ObservedSessionError(
                        "selected hybrid cold fit preceded its precursor"
                    )
                if (
                    binding.precursor_reference is None
                    or binding.precursor_reference() is not prior_result
                    or binding.precursor_guard is not prior_result
                    or binding.precursor_digest
                    != executor._deep_integrity_digest(prior_result)
                ):
                    raise ObservedSessionError(
                        "selected hybrid cold fit lacks its exact precursor"
                    )
            elif phase_before_claim != "full_m1_ready" or prior_result is not None:
                raise ObservedSessionError(
                    "selected nonhybrid cold fit has an alternate initializer"
                )
        else:
            if (
                binding.cold_reference is None
                or binding.cold_reference() is not prior_result
                or binding.cold_guard is not prior_result
                or binding.cold_digest
                != executor._deep_integrity_digest(prior_result)
            ):
                raise ObservedSessionError(
                    "selected warm refit lacks its exact cold result"
                )
    except BaseException:
        with binding.lock:
            binding.phase = "burned"
        raise


def _finish_selected_refit_fit_stage(
    authority: SelectedRefitProcedureAuthorization,
    stage: str,
    result: object | None,
) -> None:
    if (
        type(authority) is not SelectedRefitProcedureAuthorization
        or authority._seal is not _SELECTED_REFIT_AUTHORIZATION_SEAL
    ):
        raise ObservedSessionError("selected-refit procedure authority is invalid")
    binding = _lookup_selected_refit(authority)
    with binding.lock:
        expected = f"{stage}_in_flight"
        if binding.phase != expected:
            raise ObservedSessionError("selected-refit fit completion changed chronology")
        if result is None:
            binding.phase = "burned"
            binding.full_objective_guard = None
            binding.hybrid_receipt_guard = None
            binding.precursor_guard = None
            binding.cold_guard = None
            binding.warm_guard = None
            return
        digest = executor._deep_integrity_digest(result)
        reference = weakref.ref(result)
        if stage == "precursor":
            binding.precursor_reference = reference
            binding.precursor_guard = result
            binding.precursor_digest = digest
            binding.phase = "precursor_ready"
        elif stage == "cold":
            binding.cold_reference = reference
            binding.cold_guard = result
            binding.cold_digest = digest
            binding.phase = "cold_ready"
        elif stage == "warm":
            binding.warm_reference = reference
            binding.warm_guard = result
            binding.warm_digest = digest
            binding.phase = "warm_ready"
        else:
            binding.phase = "burned"
            raise ObservedSessionError("unknown selected-refit fit stage")


def _require_selected_refit_fit_pair(
    authority: SelectedRefitProcedureAuthorization,
    cold_result: object,
    warm_result: object,
) -> None:
    """Read-only final integrity check for every selected family fit pair."""

    if (
        type(authority) is not SelectedRefitProcedureAuthorization
        or authority._seal is not _SELECTED_REFIT_AUTHORIZATION_SEAL
    ):
        raise ObservedSessionError("selected-refit procedure authority is invalid")
    binding = _lookup_selected_refit(authority)
    with binding.lock:
        if (
            binding.phase != "warm_ready"
            or binding.cold_guard is not cold_result
            or binding.cold_reference is None
            or binding.cold_reference() is not cold_result
            or binding.cold_digest
            != executor._deep_integrity_digest(cold_result)
            or binding.warm_guard is not warm_result
            or binding.warm_reference is None
            or binding.warm_reference() is not warm_result
            or binding.warm_digest
            != executor._deep_integrity_digest(warm_result)
        ):
            raise ObservedSessionError(
                "selected-refit cold/warm results changed before report commit"
            )


def _burn_selected_refit_procedure(
    authority: SelectedRefitProcedureAuthorization,
) -> None:
    """Irrevocably burn a genuine authority without deep reauthentication."""

    if (
        type(authority) is not SelectedRefitProcedureAuthorization
        or authority._seal is not _SELECTED_REFIT_AUTHORIZATION_SEAL
    ):
        raise ObservedSessionError("selected-refit procedure authority is invalid")
    binding = _lookup_selected_refit(authority)
    with binding.lock:
        if binding.phase != "complete":
            binding.phase = "burned"
            binding.full_objective_guard = None
            binding.hybrid_receipt_guard = None
            binding.precursor_guard = None
            binding.cold_guard = None
            binding.warm_guard = None


def _complete_selected_refit_procedure(
    authority: SelectedRefitProcedureAuthorization,
    result_linkage: str,
) -> None:
    binding = _lookup_selected_refit(authority)
    with binding.lock:
        try:
            owner = binding.owner_reference()
            terminal = binding.terminal_reference()
            full_objective = binding.full_objective_guard
            cold = binding.cold_guard
            warm = binding.warm_guard
            if (
                binding.phase != "warm_ready"
                or result_linkage != "selected_refit_report.json"
                or owner is None
                or terminal is None
                or full_objective is None
                or binding.full_objective_reference is None
                or binding.full_objective_reference() is not full_objective
                or cold is None
                or binding.cold_reference is None
                or binding.cold_reference() is not cold
                or binding.cold_digest
                != executor._deep_integrity_digest(cold)
                or warm is None
                or binding.warm_reference is None
                or binding.warm_reference() is not warm
                or binding.warm_digest
                != executor._deep_integrity_digest(warm)
            ):
                raise ObservedSessionError(
                    "selected-refit completion lacks its guarded results or fixed linkage"
                )
            _require_selected_refit_full_objective(
                owner,
                terminal,
                authority,
                full_objective,
            )
            if binding.hybrid_reserved:
                import fit_v3_m2_optional as optional

                receipt = binding.hybrid_receipt_guard
                precursor = binding.precursor_guard
                if (
                    receipt is None
                    or binding.hybrid_receipt_reference is None
                    or binding.hybrid_receipt_reference() is not receipt
                    or binding.hybrid_receipt_digest is None
                    or precursor is None
                    or binding.precursor_reference is None
                    or binding.precursor_reference() is not precursor
                    or binding.precursor_digest
                    != executor._deep_integrity_digest(precursor)
                ):
                    raise ObservedSessionError(
                        "selected hybrid completion lost its receipt or precursor"
                    )
                try:
                    receipt_binding = (
                        optional._require_terminal_full_development_hybrid_receipt(
                            receipt
                        )
                    )
                    cold_binding = optional._require_registered_hybrid_result(
                        cold, "production"
                    )
                    warm_binding = optional._require_registered_hybrid_result(
                        warm, "production"
                    )
                except optional.LowRankError as error:
                    raise ObservedSessionError(
                        "selected hybrid completion failed receipt reauthentication"
                    ) from error
                if (
                    receipt_binding.phase != "warm_ready"
                    or receipt_binding.selected_anchor_reference is None
                    or receipt_binding.selected_anchor_reference() is not precursor
                    or receipt_binding.cold_result_reference is None
                    or receipt_binding.cold_result_reference() is not cold
                    or receipt_binding.warm_result_reference is None
                    or receipt_binding.warm_result_reference() is not warm
                    or cold_binding.selected_anchor_reference() is not precursor
                    or warm_binding.selected_anchor_reference() is not precursor
                    or cold_binding.production_receipt_reference is None
                    or cold_binding.production_receipt_reference() is not receipt
                    or warm_binding.production_receipt_reference is None
                    or warm_binding.production_receipt_reference() is not receipt
                ):
                    raise ObservedSessionError(
                        "selected hybrid completion chain changed after fitting"
                    )
            elif any(
                value is not None
                for value in (
                    binding.hybrid_receipt_guard,
                    binding.hybrid_receipt_reference,
                    binding.precursor_guard,
                    binding.precursor_reference,
                )
            ):
                raise ObservedSessionError(
                    "selected nonhybrid completion acquired hybrid state"
                )
            binding.phase = "complete"
        except BaseException:
            if binding.phase != "complete":
                binding.phase = "burned"
                binding.full_objective_guard = None
                binding.hybrid_receipt_guard = None
                binding.precursor_guard = None
                binding.cold_guard = None
                binding.warm_guard = None
            raise


def _selected_refit_provenance(
    owner: executor.CoreTrialExecutor,
    authorization: TerminalSelectionAuthorization,
) -> tuple[controller.CoverageEvidence, tuple[controller.TrialResult, ...]]:
    """Return the live coverage objects needed by the selected-refit bridge.

    This is deliberately not a serialized handoff.  The returned objects remain
    owned by the exact terminal session and are reauthenticated again when a
    terminal hybrid receipt is issued or used.
    """

    require_selected_refit_authorization(owner, authorization)
    snapshot = _lookup_terminal(authorization)
    session = snapshot.session_reference()
    if session is None:
        raise ObservedSessionError("terminal controller session no longer exists")
    state = _require_session(session)
    evidence = state.coverage_evidence
    if type(evidence) is not controller.CoverageEvidence:
        raise ObservedSessionError("selected refit lacks exact coverage evidence")
    low_rank_results: tuple[controller.TrialResult, ...] = ()
    if evidence.activation.sparse_plus_low_rank:
        start = controller.COVERAGE_TRIALS
        stop = start + 8
        low_rank_results = tuple(state.results[start:stop])
        try:
            controller.best_low_rank_anchor_rank(low_rank_results, evidence)
        except (controller.ControllerError, TypeError, ValueError) as error:
            raise ObservedSessionError(
                "selected refit lacks its exact low-rank coverage history"
            ) from error
    return evidence, low_rank_results


def _selected_refit_outer_execution(
    owner: executor.CoreTrialExecutor,
    authorization: TerminalSelectionAuthorization,
) -> object:
    """Return the authenticated winner's immutable outer-CV execution."""

    winner = require_selected_refit_authorization(owner, authorization)
    snapshot = _lookup_terminal(authorization)
    entry = snapshot.winner_entry_reference()
    if entry is None:
        raise ObservedSessionError("selected outer-CV execution is unavailable")
    if type(entry) is executor.TrialExecution:
        owned = executor._require_owned_execution(owner, entry)
        if owned is not winner:
            evidence = snapshot.evidence_reference()
            if not (
                winner.trial.valid_index == controller.COVERAGE_TRIALS
                and type(evidence) is controller.CoverageEvidence
                and evidence.core_results[-1] is winner
                and owned.trial is winner.trial
                and owned.valid is winner.valid
                and owned.gain_bits_per_cell == winner.gain_bits_per_cell
                and owned.active_pair_degrees == winner.active_pair_degrees
                and owned.coverage_binding is None
                and winner.coverage_binding is evidence.binding
            ):
                raise ObservedSessionError("selected core outer execution changed")
        execution = entry
    elif type(entry) is executor.ExternalExecutionReceipt:
        if executor._require_external_execution_receipt(owner, entry) is not winner:
            raise ObservedSessionError("selected optional outer receipt changed")
        execution = entry.execution_detail
    else:
        raise ObservedSessionError("selected outer execution has an invalid type")
    if (
        execution is None
        or getattr(execution, "trial", None) is not winner.trial
        or len(getattr(execution, "folds", ())) != 12
        or len(getattr(execution, "pair_contrast_rows", ()))
        != executor.EXPECTED_PRODUCTION_ROWS
    ):
        raise ObservedSessionError(
            "selected outer execution is incomplete or belongs to another winner"
        )
    execution_result = getattr(execution, "result", None)
    if execution_result is not winner and not (
        type(entry) is executor.TrialExecution
        and winner.trial.valid_index == controller.COVERAGE_TRIALS
        and getattr(execution_result, "trial", None) is winner.trial
        and execution_result.gain_bits_per_cell == winner.gain_bits_per_cell
        and execution_result.active_pair_degrees == winner.active_pair_degrees
    ):
        raise ObservedSessionError("selected outer execution result changed")
    return execution


def _terminal_full_refit_hybrid_inputs_request(
    owner: executor.CoreTrialExecutor,
    authorization: TerminalSelectionAuthorization,
    procedure_authority: SelectedRefitProcedureAuthorization,
    full_objective: object,
) -> tuple[
    controller.TrialResult,
    controller.CoverageEvidence,
    tuple[controller.TrialResult, ...],
    controller.Configuration,
    controller.Configuration,
]:
    """Resolve, never nominate, the exact terminal hybrid and precursor."""

    _require_selected_refit_full_objective(
        owner,
        authorization,
        procedure_authority,
        full_objective,
    )
    winner = require_selected_refit_authorization(owner, authorization)
    target = winner.trial.configuration
    evidence, anchors = _selected_refit_provenance(owner, authorization)
    if (
        type(target) is not controller.Configuration
        or target.family != "sparse_plus_low_rank"
        or len(anchors) != 8
    ):
        raise ObservedSessionError("terminal winner is not an authorized hybrid")
    import fit_v3_m2_optional as optional

    try:
        initializer = optional.derive_hybrid_low_rank_initializer_configuration(
            evidence,
            anchors,
            target,
        )
    except (optional.LowRankError, controller.ControllerError) as error:
        raise ObservedSessionError(
            "terminal hybrid precursor derivation failed"
        ) from error
    return winner, evidence, anchors, target, initializer


def _terminal_full_refit_hybrid_request(
    *,
    reserve: bool,
    owner: executor.CoreTrialExecutor,
    authorization: TerminalSelectionAuthorization,
    procedure_authority: SelectedRefitProcedureAuthorization,
    winner: controller.TrialResult,
    coverage_evidence: controller.CoverageEvidence,
    low_rank_anchor_results: Sequence[controller.TrialResult],
    target_configuration: controller.Configuration,
    initializer_configuration: controller.Configuration,
    full_objective: object,
    specification: object,
) -> bytes:
    """Reserve once or read-only reauthenticate one selected hybrid target."""

    canonical_winner = require_selected_refit_authorization(owner, authorization)
    if canonical_winner is not winner:
        raise ObservedSessionError("terminal full refit winner identity changed")
    evidence, canonical_anchors = _selected_refit_provenance(owner, authorization)
    supplied_anchors = tuple(low_rank_anchor_results)
    if (
        coverage_evidence is not evidence
        or len(supplied_anchors) != 8
        or len(canonical_anchors) != 8
        or any(
            supplied is not canonical
            for supplied, canonical in zip(
                supplied_anchors, canonical_anchors, strict=True
            )
        )
        or winner.trial.configuration is not target_configuration
        or target_configuration.family != "sparse_plus_low_rank"
    ):
        raise ObservedSessionError(
            "terminal full refit hybrid provenance differs from the live ledger"
        )

    # Import lazily to preserve the observed-session/optional-engine boundary.
    import fit_v3_m2 as core
    import fit_v3_m2_optional as optional

    try:
        derived_initializer = optional.derive_hybrid_low_rank_initializer_configuration(
            evidence,
            canonical_anchors,
            target_configuration,
        )
    except (optional.LowRankError, TypeError, ValueError) as error:
        raise ObservedSessionError(
            "terminal hybrid initializer derivation failed"
        ) from error
    if derived_initializer is not initializer_configuration:
        raise ObservedSessionError(
            "terminal hybrid full refit used an alternate precursor"
        )
    fixed_m1_theta, full_prepared, audit_evidence, _ = (
        _selected_refit_module_request(
            "_full_objective_authority_record",
            full_objective,
            procedure_authority,
            owner,
            authorization,
        )
    )
    objective_digest = _require_selected_refit_full_objective(
        owner,
        authorization,
        procedure_authority,
        full_objective,
    )
    binding, _ = _selected_refit_binding(
        owner, authorization, procedure_authority
    )
    candidates = binding.source.candidates_reference()
    geometry = binding.source.geometry_reference()
    pairs = binding.source.pairs_reference()
    if (
        type(full_prepared) is not core.PreparedM2
        or len(full_prepared.cells) != executor.EXPECTED_PRODUCTION_ROWS
        or full_prepared.transform.rank != executor.EXPECTED_PRODUCTION_RANK
        or candidates is not owner.prepared.candidates
        or geometry is not owner.prepared.geometry
        or pairs is not owner.prepared.pairs
    ):
        raise ObservedSessionError(
            "terminal hybrid full refit objective is not the exact production objective"
        )
    cached_by_identifier = {
        cell.identifier: cell
        for fold in owner.prepared.folds
        for cell in fold.validation.cells
    }
    if (
        tuple((cell.identifier, cell.group) for cell in full_prepared.cells)
        != owner.prepared.cell_inventory
        or len(cached_by_identifier) != executor.EXPECTED_PRODUCTION_ROWS
        or any(
            cached_by_identifier.get(cell.identifier) is not cell
            for cell in full_prepared.cells
        )
    ):
        raise ObservedSessionError(
            "terminal hybrid full refit cells differ from the selected session"
        )
    digest = executor._authority_binding_digest(
        "terminal_selected_full_development_hybrid_v1",
        objective_digest,
        id(owner),
        id(authorization),
        id(procedure_authority),
        id(winner),
        id(evidence),
        tuple(id(result) for result in canonical_anchors),
        id(target_configuration),
        id(initializer_configuration),
        id(full_objective),
        id(fixed_m1_theta),
        id(full_prepared),
        id(candidates),
        id(geometry),
        id(pairs),
        id(specification),
        executor._deep_integrity_digest(fixed_m1_theta),
        executor._deep_integrity_digest(full_prepared),
        executor._deep_integrity_digest(specification),
    )
    with binding.lock:
        if reserve:
            if binding.phase != "full_m1_ready" or binding.hybrid_reserved:
                raise ObservedSessionError(
                    "terminal hybrid authority was already reserved"
                )
            binding.hybrid_reserved = True
        elif (
            not binding.hybrid_reserved
            or binding.phase
            not in {
                "full_m1_ready",
                "precursor_in_flight",
                "precursor_ready",
                "cold_in_flight",
                "cold_ready",
                "warm_in_flight",
                "warm_ready",
            }
        ):
            raise ObservedSessionError(
                "terminal hybrid authority was not prospectively reserved"
            )
    return digest


def _reserve_terminal_full_refit_hybrid_request(
    owner: executor.CoreTrialExecutor,
    authorization: TerminalSelectionAuthorization,
    procedure_authority: SelectedRefitProcedureAuthorization,
    winner: controller.TrialResult,
    coverage_evidence: controller.CoverageEvidence,
    low_rank_anchor_results: Sequence[controller.TrialResult],
    target_configuration: controller.Configuration,
    initializer_configuration: controller.Configuration,
    full_objective: object,
    specification: object,
) -> bytes:
    return _terminal_full_refit_hybrid_request(
        reserve=True,
        owner=owner,
        authorization=authorization,
        procedure_authority=procedure_authority,
        winner=winner,
        coverage_evidence=coverage_evidence,
        low_rank_anchor_results=low_rank_anchor_results,
        target_configuration=target_configuration,
        initializer_configuration=initializer_configuration,
        full_objective=full_objective,
        specification=specification,
    )


def _reauthenticate_terminal_full_refit_hybrid_request(
    owner: executor.CoreTrialExecutor,
    authorization: TerminalSelectionAuthorization,
    procedure_authority: SelectedRefitProcedureAuthorization,
    winner: controller.TrialResult,
    coverage_evidence: controller.CoverageEvidence,
    low_rank_anchor_results: Sequence[controller.TrialResult],
    target_configuration: controller.Configuration,
    initializer_configuration: controller.Configuration,
    full_objective: object,
    specification: object,
) -> bytes:
    return _terminal_full_refit_hybrid_request(
        reserve=False,
        owner=owner,
        authorization=authorization,
        procedure_authority=procedure_authority,
        winner=winner,
        coverage_evidence=coverage_evidence,
        low_rank_anchor_results=low_rank_anchor_results,
        target_configuration=target_configuration,
        initializer_configuration=initializer_configuration,
        full_objective=full_objective,
        specification=specification,
    )


def _validated_terminal_snapshot(
    authorization: TerminalSelectionAuthorization,
) -> tuple[_TerminalSnapshot, controller.TrialResult]:
    snapshot = _lookup_terminal(authorization)
    owner = snapshot.owner_reference()
    if owner is None:
        raise ObservedSessionError("terminal selection owner no longer exists")
    winner = require_terminal_selection(owner, authorization)
    return snapshot, winner


__all__ = (
    "AdapterAuthority",
    "ControllerRunSession",
    "INCOMPLETE_BUDGET_STOP",
    "INCOMPLETE_CONCLUSION",
    "ORDINARY_CONCLUSION",
    "ORDINARY_PATIENCE_STOP",
    "ObservedResourceStop",
    "ObservedSessionError",
    "SelectedRefitProcedureAuthorization",
    "SyntheticTrialOutcome",
    "TECHNICAL_RESOURCE_STOP",
    "TECHNICAL_RETRY_STOP",
    "TECHNICAL_INTEGRITY_STOP",
    "TerminalSelectionAuthorization",
    "authorize_terminal_selection",
    "authorize_selected_refit_procedure",
    "issue_optional_family_authority",
    "issue_rademacher_reference_authority",
    "issue_whole_set_permutation_authority",
    "issue_synthetic_qualification_authority",
    "require_terminal_selection",
    "require_selected_refit_authorization",
    "start_observed_session",
    "start_synthetic_qualification_session",
)
