#!/usr/bin/env python3
"""Named production adapter for EP10 v3 optional M2 families.

This adapter is the only production bridge from the chronology-owning observed
session to the low-rank and sparse-plus-low-rank numerical engines.  It owns a
one-shot attempt capability, executes all twelve frozen outer folds, and keeps
the complete 194-cell score/physical-pair-contrast record in an immutable
``OptionalTrialExecution``.  It has no filesystem reader and no command-line
entry point.

The adapter is intentionally process-local.  Authorities, capabilities,
hybrid initializer receipts, optimizer results, the observed session, and the
prepared cache must all remain the exact live objects that issued the record.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(os.path.abspath(__file__))
TOOLS_DIR = SCRIPT_PATH.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m2 as core  # noqa: E402
import fit_v3_m2_optional as optional  # noqa: E402
import v3_m2_trial_executor_v1 as executor  # noqa: E402
import v3_search_controller_v1 as controller  # noqa: E402


OPTIONAL_FAMILIES = frozenset({"low_rank", "sparse_plus_low_rank"})
RULE_VERSION = optional.HYBRID_INITIALIZER_RULE_VERSION
LN2 = math.log(2.0)

_ADAPTER_SEAL = object()
_CAPABILITY_SEAL = object()
_HYBRID_AUTHORITY_SEAL = object()


class OptionalAdapterError(core.M2Error):
    """The named optional adapter rejected an execution or numerical record."""


class OptionalExecutionAuthorityError(OptionalAdapterError):
    """A session/cache/trial/one-shot authority failed closed."""


def _make_weak_identity_registry(label: str) -> tuple[Any, Any]:
    records: dict[int, tuple[weakref.ReferenceType[Any], Any]] = {}
    lock = threading.RLock()

    def register(instance: Any, metadata: Any) -> None:
        identity = id(instance)

        def remove(reference: weakref.ReferenceType[Any]) -> None:
            with lock:
                current = records.get(identity)
                if current is not None and current[0] is reference:
                    records.pop(identity, None)

        with lock:
            existing = records.get(identity)
            if existing is not None and existing[0]() is not instance:
                raise OptionalExecutionAuthorityError(f"{label} identity collision")
            records[identity] = (weakref.ref(instance, remove), metadata)

    def lookup(instance: Any) -> Any:
        with lock:
            record = records.get(id(instance))
            if record is None or record[0]() is not instance:
                raise OptionalExecutionAuthorityError(
                    f"{label} was not issued by the named optional adapter"
                )
            return record[1]

    return register, lookup


_register_adapter, _lookup_adapter = _make_weak_identity_registry(
    "optional-family adapter"
)
_register_capability, _lookup_capability = _make_weak_identity_registry(
    "optional attempt capability"
)
_register_execution, _lookup_execution = _make_weak_identity_registry(
    "optional trial execution"
)
_register_hybrid_authority, _lookup_hybrid_authority = _make_weak_identity_registry(
    "session hybrid solver authority"
)


class OptionalFamilyAdapter:
    """Opaque adapter permanently bound to one executor/cache and one session."""

    __slots__ = ("_seal", "__weakref__")

    def __init__(self, *, _seal: object | None = None) -> None:
        if _seal is not _ADAPTER_SEAL:
            raise OptionalExecutionAuthorityError(
                "optional adapters must be issued by create_optional_family_adapter"
            )
        object.__setattr__(self, "_seal", _seal)

    def __copy__(self) -> None:
        raise OptionalExecutionAuthorityError("optional adapters cannot be copied")

    def __deepcopy__(self, memo: Any) -> None:
        del memo
        raise OptionalExecutionAuthorityError("optional adapters cannot be copied")

    def __reduce__(self) -> None:
        raise OptionalExecutionAuthorityError("optional adapters cannot be serialized")


class OptionalAttemptCapability:
    """Opaque one-shot authority for one exact optional trial attempt."""

    __slots__ = ("_seal", "__weakref__")

    def __init__(self, *, _seal: object | None = None) -> None:
        if _seal is not _CAPABILITY_SEAL:
            raise OptionalExecutionAuthorityError(
                "optional attempt capabilities require observed-session issuance"
            )
        object.__setattr__(self, "_seal", _seal)

    def __copy__(self) -> None:
        raise OptionalExecutionAuthorityError(
            "optional attempt capabilities cannot be copied"
        )

    def __deepcopy__(self, memo: Any) -> None:
        del memo
        raise OptionalExecutionAuthorityError(
            "optional attempt capabilities cannot be copied"
        )

    def __reduce__(self) -> None:
        raise OptionalExecutionAuthorityError(
            "optional attempt capabilities cannot be serialized"
        )


class _SessionHybridAuthority:
    """Opaque per-fold authority for the v3 prescribed hybrid initializer."""

    __slots__ = ("_seal", "__weakref__")

    def __init__(self, *, _seal: object | None = None) -> None:
        if _seal is not _HYBRID_AUTHORITY_SEAL:
            raise OptionalExecutionAuthorityError(
                "session hybrid authority requires named-adapter issuance"
            )
        object.__setattr__(self, "_seal", _seal)

    def __copy__(self) -> None:
        raise OptionalExecutionAuthorityError("hybrid authorities cannot be copied")

    def __deepcopy__(self, memo: Any) -> None:
        del memo
        raise OptionalExecutionAuthorityError("hybrid authorities cannot be copied")

    def __reduce__(self) -> None:
        raise OptionalExecutionAuthorityError("hybrid authorities cannot be serialized")


@dataclass(frozen=True)
class OptionalFoldTrialExecution:
    """One optional-family outer-fold fit and its unique held-out predictions."""

    spec: executor.FoldSpec
    specification: optional.LowRankSpec | optional.HybridSpec
    initializer_configuration: controller.Configuration | None
    initializer_fit: optional.LowRankOptimizerResult | None
    hybrid_authority: object | None
    hybrid_receipt: object | None
    fit: optional.LowRankOptimizerResult | optional.HybridOptimizerResult
    gain_bits_per_cell: float
    active_pair_degrees: int
    cells: tuple[executor.CellTrialResult, ...]


@dataclass(frozen=True)
class OptionalTrialExecution:
    """Complete immutable detail for one session-issued optional trial."""

    trial: controller.Trial
    folds: tuple[OptionalFoldTrialExecution, ...]
    pair_contrast_rows: tuple[controller.PairContrastRow, ...]
    equal_group_pair_contrasts: tuple[float, ...]
    result: controller.TrialResult
    mode: str
    session_identity: int
    owner_identity: int
    cache_identity: int
    authority_identity: int
    attempt_ordinal: int


@dataclass
class _AdapterState:
    owner_reference: weakref.ReferenceType[executor.CoreTrialExecutor]
    cache_reference: weakref.ReferenceType[executor.PreparedTrialCache]
    lock: threading.RLock
    validator: _OptionalExecutionValidator | None = None
    session_reference: weakref.ReferenceType[Any] | None = None
    authority_reference: weakref.ReferenceType[Any] | None = None
    current_capability: OptionalAttemptCapability | None = None
    acknowledged_attempts: int = 0
    result_records: dict[
        int,
        tuple[
            weakref.ReferenceType[controller.TrialResult],
            weakref.ReferenceType[OptionalTrialExecution],
        ],
    ] | None = None


@dataclass
class _CapabilityRecord:
    adapter_reference: weakref.ReferenceType[OptionalFamilyAdapter]
    session_reference: weakref.ReferenceType[Any]
    authority_reference: weakref.ReferenceType[Any]
    owner_reference: weakref.ReferenceType[executor.CoreTrialExecutor]
    cache_reference: weakref.ReferenceType[executor.PreparedTrialCache]
    trial_reference: weakref.ReferenceType[controller.Trial]
    trial_digest: bytes
    coverage_reference: weakref.ReferenceType[controller.CoverageEvidence]
    coverage_digest: bytes
    history_references: tuple[weakref.ReferenceType[Any], ...]
    history_digests: tuple[bytes, ...]
    result_references: tuple[weakref.ReferenceType[controller.TrialResult], ...]
    result_digests: tuple[bytes, ...]
    session_attempt_ordinal: int
    session_version: int
    optional_attempt_ordinal: int
    binding_digest: bytes
    phase: str = "issued"
    completion: str | None = None
    error_type: str | None = None
    execution_reference: weakref.ReferenceType[OptionalTrialExecution] | None = None
    execution_digest: bytes | None = None
    attempt_reference: weakref.ReferenceType[Any] | None = None
    attempt_digest: bytes | None = None


@dataclass(frozen=True)
class _ExecutionBinding:
    adapter_reference: weakref.ReferenceType[OptionalFamilyAdapter]
    session_reference: weakref.ReferenceType[Any]
    authority_reference: weakref.ReferenceType[Any]
    owner_reference: weakref.ReferenceType[executor.CoreTrialExecutor]
    cache_reference: weakref.ReferenceType[executor.PreparedTrialCache]
    capability_reference: weakref.ReferenceType[OptionalAttemptCapability]
    capability_digest: bytes
    execution_digest: bytes


@dataclass(frozen=True)
class _HybridAuthorityBinding:
    capability_reference: weakref.ReferenceType[OptionalAttemptCapability]
    trial_reference: weakref.ReferenceType[controller.Trial]
    coverage_reference: weakref.ReferenceType[controller.CoverageEvidence]
    coverage_digest: bytes
    result_references: tuple[weakref.ReferenceType[controller.TrialResult], ...]
    result_digests: tuple[bytes, ...]
    target_configuration_reference: weakref.ReferenceType[controller.Configuration]
    target_configuration_digest: bytes
    initializer_configuration_reference: weakref.ReferenceType[controller.Configuration]
    initializer_configuration_digest: bytes
    initializer_fit_reference: weakref.ReferenceType[optional.LowRankOptimizerResult]
    initializer_fit_digest: bytes
    specification_reference: weakref.ReferenceType[optional.HybridSpec]
    specification_digest: bytes
    prepared_fold_reference: weakref.ReferenceType[executor.PreparedFold]
    fold_ordinal: int
    binding_digest: bytes


class _OptionalExecutionValidator:
    """Stable weak-referenceable validator retained by the adapter state."""

    __slots__ = ("adapter_reference", "__weakref__")

    def __init__(self, adapter: OptionalFamilyAdapter) -> None:
        self.adapter_reference = weakref.ref(adapter)

    def __call__(
        self,
        producer: OptionalFamilyAdapter,
        result: controller.TrialResult,
    ) -> bool:
        adapter = self.adapter_reference()
        if (
            adapter is None
            or producer is not adapter
            or type(result) is not controller.TrialResult
        ):
            return False
        state = _require_adapter(adapter)
        assert state.result_records is not None
        record = state.result_records.get(result.trial.valid_index)
        if record is None or record[0]() is not result:
            return False
        execution = record[1]()
        return execution is not None and execution.result is result

    def validate_execution_detail(
        self,
        producer: OptionalFamilyAdapter,
        result: controller.TrialResult,
        execution_detail: object | None,
    ) -> bool:
        adapter = self.adapter_reference()
        if (
            adapter is None
            or producer is not adapter
            or type(execution_detail) is not OptionalTrialExecution
            or execution_detail.result is not result
        ):
            return False
        return _require_registered_execution(adapter, execution_detail) is result


def _validate_named_external_execution_detail(
    producer: object,
    validator: object,
    result: controller.TrialResult,
    execution_detail: object,
) -> bool:
    """Exact executor handshake for one named production execution record."""

    if (
        type(producer) is not OptionalFamilyAdapter
        or type(validator) is not _OptionalExecutionValidator
    ):
        return False
    state = _require_adapter(producer)
    if state.validator is not validator:
        return False
    return (
        _OptionalExecutionValidator.__call__(validator, producer, result) is True
        and _OptionalExecutionValidator.validate_execution_detail(
            validator,
            producer,
            result,
            execution_detail,
        )
        is True
    )


def _observed_request(name: str, *arguments: Any) -> Any:
    module = sys.modules.get("v3_m2_observed_session_v1")
    expected = TOOLS_DIR / "v3_m2_observed_session_v1.py"
    if (
        module is None
        or Path(os.path.abspath(os.fspath(getattr(module, "SCRIPT_PATH", ""))))
        != expected
    ):
        raise OptionalExecutionAuthorityError(
            "optional production execution requires the exact observed session"
        )
    callback = getattr(module, name, None)
    if not callable(callback):
        raise OptionalExecutionAuthorityError(
            "observed-session optional handshake is unavailable"
        )
    try:
        return callback(*arguments)
    except OptionalExecutionAuthorityError:
        raise
    except Exception as error:
        raise OptionalExecutionAuthorityError(
            "observed-session optional handshake failed"
        ) from error


def _require_adapter(
    adapter: OptionalFamilyAdapter,
    *,
    deep_cache: bool = True,
) -> _AdapterState:
    if type(adapter) is not OptionalFamilyAdapter or adapter._seal is not _ADAPTER_SEAL:
        raise OptionalExecutionAuthorityError("optional adapter lacks exact module authority")
    state = _lookup_adapter(adapter)
    if type(state) is not _AdapterState:
        raise OptionalExecutionAuthorityError("optional adapter state is malformed")
    owner = state.owner_reference()
    cache = state.cache_reference()
    if (
        type(owner) is not executor.CoreTrialExecutor
        or cache is None
        or owner.prepared is not cache
    ):
        raise OptionalExecutionAuthorityError("optional adapter owner/cache is unavailable")
    if deep_cache:
        executor._require_prepared_cache(cache)
    if cache.mode != executor.SEALED_PRODUCTION_MODE:
        raise OptionalExecutionAuthorityError(
            "named optional adapter accepts only sealed-production caches"
        )
    if cache.mode == "sealed_production" and (
        len(cache.cell_inventory) != executor.EXPECTED_PRODUCTION_ROWS
        or len(cache.folds) != executor.EXPECTED_FOLDS
    ):
        raise OptionalExecutionAuthorityError(
            "sealed optional cache is not the exact 194-cell 12-fold inventory"
        )
    if state.result_records is None or state.validator is None:
        raise OptionalExecutionAuthorityError("optional adapter registries are unavailable")
    return state


def create_optional_family_adapter(
    owner: executor.CoreTrialExecutor,
) -> OptionalFamilyAdapter:
    """Create one named adapter for one exact unmodified production executor."""

    if type(owner) is not executor.CoreTrialExecutor:
        raise OptionalExecutionAuthorityError("optional adapter owner has the wrong type")
    cache = executor._require_prepared_cache(owner.prepared)
    if cache.mode != executor.SEALED_PRODUCTION_MODE:
        raise OptionalExecutionAuthorityError(
            "optional adapter creation requires a sealed-production cache"
        )
    adapter = OptionalFamilyAdapter(_seal=_ADAPTER_SEAL)
    state = _AdapterState(
        owner_reference=weakref.ref(owner),
        cache_reference=weakref.ref(cache),
        lock=threading.RLock(),
        result_records={},
    )
    state.validator = _OptionalExecutionValidator(adapter)
    _register_adapter(adapter, state)
    return adapter


def _bind_to_observed_session(
    adapter: OptionalFamilyAdapter,
    session: object,
    authority: object,
    owner: executor.CoreTrialExecutor,
) -> bool:
    """Bind the adapter once; callable only during named session issuance."""

    state = _require_adapter(adapter)
    if state.owner_reference() is not owner:
        raise OptionalExecutionAuthorityError("optional adapter belongs to another owner")
    accepted = _observed_request(
        "_optional_adapter_bind_request",
        session,
        authority,
        adapter,
        owner,
    )
    if accepted is not True:
        raise OptionalExecutionAuthorityError("observed session rejected adapter binding")
    with state.lock:
        if state.session_reference is not None or state.authority_reference is not None:
            raise OptionalExecutionAuthorityError("optional adapter was already bound")
        try:
            state.session_reference = weakref.ref(session)
            state.authority_reference = weakref.ref(authority)
        except TypeError as error:
            raise OptionalExecutionAuthorityError(
                "session and authority require stable weak identities"
            ) from error
    return True


def _adapter_producer_and_validator(
    adapter: OptionalFamilyAdapter,
) -> tuple[OptionalFamilyAdapter, _OptionalExecutionValidator]:
    state = _require_adapter(adapter)
    assert state.validator is not None
    return adapter, state.validator


def _capability_binding(record: _CapabilityRecord) -> bytes:
    return executor._authority_binding_digest(
        id(record.adapter_reference()),
        id(record.session_reference()),
        id(record.authority_reference()),
        id(record.owner_reference()),
        id(record.cache_reference()),
        id(record.trial_reference()),
        record.trial_digest,
        id(record.coverage_reference()),
        record.coverage_digest,
        tuple(id(reference()) for reference in record.history_references),
        record.history_digests,
        tuple(id(reference()) for reference in record.result_references),
        record.result_digests,
        record.session_attempt_ordinal,
        record.session_version,
        record.optional_attempt_ordinal,
        RULE_VERSION,
    )


def _history_matches_results(
    history: Sequence[executor.TrialExecution | executor.ExternalExecutionReceipt],
    results: Sequence[controller.TrialResult],
    evidence: controller.CoverageEvidence,
) -> bool:
    """Match raw entries to the canonical stream with trial-16 binding."""

    entries = tuple(history)
    canonical = tuple(results)
    if len(entries) != len(canonical):
        return False
    for index, (entry, result) in enumerate(zip(entries, canonical, strict=True)):
        if index != controller.COVERAGE_TRIALS - 1:
            if entry.result is not result:
                return False
            continue
        if (
            result is not evidence.core_results[-1]
            or entry.result.trial is not result.trial
            or entry.result.gain_bits_per_cell != result.gain_bits_per_cell
            or entry.result.active_pair_degrees != result.active_pair_degrees
            or entry.result.valid != result.valid
            or entry.result.coverage_binding is not None
            or result.coverage_binding != evidence.binding
        ):
            return False
    return True


def _require_live_capability(
    capability: OptionalAttemptCapability,
) -> tuple[_CapabilityRecord, _AdapterState]:
    if (
        type(capability) is not OptionalAttemptCapability
        or capability._seal is not _CAPABILITY_SEAL
    ):
        raise OptionalExecutionAuthorityError(
            "optional attempt capability lacks exact opaque authority"
        )
    record = _lookup_capability(capability)
    if type(record) is not _CapabilityRecord:
        raise OptionalExecutionAuthorityError("optional capability state is malformed")
    adapter = record.adapter_reference()
    if adapter is None:
        raise OptionalExecutionAuthorityError("optional adapter died during capability use")
    state = _require_adapter(adapter)
    if _capability_binding(record) != record.binding_digest:
        raise OptionalExecutionAuthorityError("optional capability binding changed")
    return record, state


def _issue_optional_attempt_capability(
    adapter: OptionalFamilyAdapter,
    session: object,
    authority: object,
    owner: executor.CoreTrialExecutor,
    trial: controller.Trial,
    coverage_evidence: controller.CoverageEvidence,
    history: Sequence[executor.TrialExecution | executor.ExternalExecutionReceipt],
    results: Sequence[controller.TrialResult],
    session_attempt_ordinal: int,
    session_version: int,
) -> OptionalAttemptCapability:
    """Issue one exact capability while the observed chronology gate is open."""

    state = _require_adapter(adapter)
    cache = state.cache_reference()
    immutable_history = tuple(history)
    immutable_results = tuple(results)
    if (
        state.owner_reference() is not owner
        or cache is not owner.prepared
        or state.session_reference is None
        or state.session_reference() is not session
        or state.authority_reference is None
        or state.authority_reference() is not authority
        or type(trial) is not controller.Trial
        or type(trial.configuration) is not controller.Configuration
        or trial.configuration.family not in OPTIONAL_FAMILIES
        or trial.falsifier
        or type(coverage_evidence) is not controller.CoverageEvidence
        or len(immutable_history) != len(immutable_results)
        or not _history_matches_results(
            immutable_history,
            immutable_results,
            coverage_evidence,
        )
        or type(session_attempt_ordinal) is not int
        or type(session_version) is not int
    ):
        raise OptionalExecutionAuthorityError(
            "optional capability inputs differ from the bound session chronology"
        )
    accepted = _observed_request(
        "_optional_adapter_capability_request",
        session,
        authority,
        adapter,
        owner,
        trial,
        session_attempt_ordinal,
        session_version,
    )
    if accepted is not True:
        raise OptionalExecutionAuthorityError("observed session rejected capability issuance")
    with state.lock:
        current = state.current_capability
        if current is not None:
            prior = _lookup_capability(current)
            if prior.phase != "ledger_acked":
                raise OptionalExecutionAuthorityError(
                    "prior optional capability is not ledger-acknowledged"
                )
        capability = OptionalAttemptCapability(_seal=_CAPABILITY_SEAL)
        record = _CapabilityRecord(
            adapter_reference=weakref.ref(adapter),
            session_reference=weakref.ref(session),
            authority_reference=weakref.ref(authority),
            owner_reference=weakref.ref(owner),
            cache_reference=weakref.ref(cache),
            trial_reference=weakref.ref(trial),
            trial_digest=executor._deep_integrity_digest(trial),
            coverage_reference=weakref.ref(coverage_evidence),
            coverage_digest=executor._deep_integrity_digest(coverage_evidence),
            history_references=tuple(weakref.ref(entry) for entry in immutable_history),
            history_digests=tuple(
                executor._deep_integrity_digest(entry) for entry in immutable_history
            ),
            result_references=tuple(weakref.ref(result) for result in immutable_results),
            result_digests=tuple(
                executor._deep_integrity_digest(result) for result in immutable_results
            ),
            session_attempt_ordinal=session_attempt_ordinal,
            session_version=session_version,
            optional_attempt_ordinal=state.acknowledged_attempts + 1,
            binding_digest=b"",
        )
        record.binding_digest = _capability_binding(record)
        _register_capability(capability, record)
        state.current_capability = capability
        return capability


def _require_capability_context(
    capability: OptionalAttemptCapability,
    *,
    expected_phase: str,
) -> tuple[_CapabilityRecord, _AdapterState]:
    record, state = _require_live_capability(capability)
    adapter = record.adapter_reference()
    session = record.session_reference()
    authority = record.authority_reference()
    owner = record.owner_reference()
    cache = record.cache_reference()
    trial = record.trial_reference()
    evidence = record.coverage_reference()
    history = tuple(reference() for reference in record.history_references)
    results = tuple(reference() for reference in record.result_references)
    if (
        adapter is None
        or session is None
        or authority is None
        or owner is None
        or cache is None
        or trial is None
        or evidence is None
        or any(entry is None for entry in history)
        or any(result is None for result in results)
        or state.current_capability is not capability
        or record.phase != expected_phase
        or owner.prepared is not cache
        or state.owner_reference() is not owner
        or state.session_reference is None
        or state.session_reference() is not session
        or state.authority_reference is None
        or state.authority_reference() is not authority
        or executor._deep_integrity_digest(trial) != record.trial_digest
        or executor._deep_integrity_digest(evidence) != record.coverage_digest
        or tuple(executor._deep_integrity_digest(entry) for entry in history)
        != record.history_digests
        or tuple(executor._deep_integrity_digest(result) for result in results)
        != record.result_digests
        or not _history_matches_results(history, results, evidence)
    ):
        raise OptionalExecutionAuthorityError(
            "optional capability session/cache/trial/history binding changed"
        )
    return record, state


def _low_rank_coverage(
    record: _CapabilityRecord,
) -> tuple[
    controller.CoverageEvidence,
    tuple[controller.TrialResult, ...],
    tuple[OptionalTrialExecution, ...],
]:
    evidence = record.coverage_reference()
    if type(evidence) is not controller.CoverageEvidence:
        raise OptionalExecutionAuthorityError("optional attempt lost coverage evidence")
    results = tuple(reference() for reference in record.result_references)
    history = tuple(reference() for reference in record.history_references)
    start = controller.COVERAGE_TRIALS
    stop = start + 8
    if len(results) < stop or len(history) < stop:
        raise OptionalExecutionAuthorityError(
            "hybrid execution precedes all eight low-rank score results"
        )
    score_results = results[start:stop]
    executions: list[OptionalTrialExecution] = []
    for entry, result in zip(history[start:stop], score_results, strict=True):
        if (
            type(entry) is not executor.ExternalExecutionReceipt
            or type(entry.execution_detail) is not OptionalTrialExecution
            or entry.result is not result
        ):
            raise OptionalExecutionAuthorityError(
                "low-rank coverage lacks detailed optional executions"
            )
        execution = entry.execution_detail
        adapter = record.adapter_reference()
        assert adapter is not None
        _require_registered_execution(adapter, execution)
        executions.append(execution)
    try:
        controller.best_low_rank_anchor_rank(score_results, evidence)
    except (controller.ControllerError, TypeError, ValueError) as error:
        raise OptionalExecutionAuthorityError(
            "low-rank score-bearing coverage no longer validates"
        ) from error
    return evidence, score_results, tuple(executions)


def _hybrid_authority_digest(binding: _HybridAuthorityBinding) -> bytes:
    return executor._authority_binding_digest(
        id(binding.capability_reference()),
        id(binding.trial_reference()),
        id(binding.coverage_reference()),
        binding.coverage_digest,
        tuple(id(reference()) for reference in binding.result_references),
        binding.result_digests,
        id(binding.target_configuration_reference()),
        binding.target_configuration_digest,
        id(binding.initializer_configuration_reference()),
        binding.initializer_configuration_digest,
        id(binding.initializer_fit_reference()),
        binding.initializer_fit_digest,
        id(binding.specification_reference()),
        binding.specification_digest,
        id(binding.prepared_fold_reference()),
        binding.fold_ordinal,
        RULE_VERSION,
    )


def _issue_hybrid_solver_authority(
    capability: OptionalAttemptCapability,
    fold: executor.PreparedFold,
    evidence: controller.CoverageEvidence,
    score_results: tuple[controller.TrialResult, ...],
    target_configuration: controller.Configuration,
    initializer_configuration: controller.Configuration,
    initializer_fit: optional.LowRankOptimizerResult,
    specification: optional.HybridSpec,
) -> _SessionHybridAuthority:
    record, _ = _require_capability_context(capability, expected_phase="in_flight")
    trial = record.trial_reference()
    if trial is None or trial.configuration is not target_configuration:
        raise OptionalExecutionAuthorityError("hybrid target is not the capability trial")
    authority = _SessionHybridAuthority(_seal=_HYBRID_AUTHORITY_SEAL)
    binding = _HybridAuthorityBinding(
        capability_reference=weakref.ref(capability),
        trial_reference=weakref.ref(trial),
        coverage_reference=weakref.ref(evidence),
        coverage_digest=executor._deep_integrity_digest(evidence),
        result_references=tuple(weakref.ref(result) for result in score_results),
        result_digests=tuple(
            executor._deep_integrity_digest(result) for result in score_results
        ),
        target_configuration_reference=weakref.ref(target_configuration),
        target_configuration_digest=executor._deep_integrity_digest(target_configuration),
        initializer_configuration_reference=weakref.ref(initializer_configuration),
        initializer_configuration_digest=executor._deep_integrity_digest(
            initializer_configuration
        ),
        initializer_fit_reference=weakref.ref(initializer_fit),
        initializer_fit_digest=executor._deep_integrity_digest(initializer_fit),
        specification_reference=weakref.ref(specification),
        specification_digest=executor._deep_integrity_digest(specification),
        prepared_fold_reference=weakref.ref(fold),
        fold_ordinal=fold.spec.ordinal,
        binding_digest=b"",
    )
    object.__setattr__(binding, "binding_digest", _hybrid_authority_digest(binding))
    _register_hybrid_authority(authority, binding)
    return authority


def _require_hybrid_solver_authority(
    authority: object,
    coverage_evidence: object,
    low_rank_anchor_results: Sequence[object],
    selected_low_rank_anchor: optional.LowRankOptimizerResult,
    target_configuration: object,
    specification: optional.HybridSpec,
    initializer_configuration: object,
) -> bytes:
    """Reauthenticate the exact live v3 initializer receipt at solver use time."""

    if (
        type(authority) is not _SessionHybridAuthority
        or authority._seal is not _HYBRID_AUTHORITY_SEAL
    ):
        raise OptionalExecutionAuthorityError("hybrid solver authority is invalid")
    binding = _lookup_hybrid_authority(authority)
    if type(binding) is not _HybridAuthorityBinding:
        raise OptionalExecutionAuthorityError("hybrid solver binding is malformed")
    capability = binding.capability_reference()
    trial = binding.trial_reference()
    evidence = binding.coverage_reference()
    results = tuple(reference() for reference in binding.result_references)
    target = binding.target_configuration_reference()
    initializer = binding.initializer_configuration_reference()
    initializer_fit = binding.initializer_fit_reference()
    target_specification = binding.specification_reference()
    fold = binding.prepared_fold_reference()
    supplied_results = tuple(low_rank_anchor_results)
    if (
        capability is None
        or trial is None
        or evidence is not coverage_evidence
        or any(result is None for result in results)
        or len(results) != 8
        or len(supplied_results) != 8
        or any(
            supplied is not registered
            for supplied, registered in zip(
                supplied_results,
                results,
                strict=True,
            )
        )
        or target is not target_configuration
        or initializer is not initializer_configuration
        or initializer_fit is not selected_low_rank_anchor
        or target_specification is not specification
        or fold is None
        or trial.configuration is not target_configuration
        or executor._deep_integrity_digest(evidence) != binding.coverage_digest
        or tuple(executor._deep_integrity_digest(result) for result in results)
        != binding.result_digests
        or executor._deep_integrity_digest(target)
        != binding.target_configuration_digest
        or executor._deep_integrity_digest(initializer)
        != binding.initializer_configuration_digest
        or executor._deep_integrity_digest(initializer_fit)
        != binding.initializer_fit_digest
        or executor._deep_integrity_digest(specification)
        != binding.specification_digest
        or fold.spec.ordinal != binding.fold_ordinal
        or _hybrid_authority_digest(binding) != binding.binding_digest
    ):
        raise OptionalExecutionAuthorityError(
            "hybrid solver evidence/fold/initializer/specification binding changed"
        )
    capability_record, _ = _require_live_capability(capability)
    if capability_record.phase not in {"in_flight", "completed_unacked", "ledger_acked"}:
        raise OptionalExecutionAuthorityError("hybrid authority outlived its attempt")
    try:
        derived = optional.derive_hybrid_low_rank_initializer_configuration(
            evidence,
            results,
            target,
        )
    except optional.LowRankError as error:
        raise OptionalExecutionAuthorityError(
            "hybrid initializer no longer follows the v3 rule"
        ) from error
    if derived is not initializer:
        raise OptionalExecutionAuthorityError("hybrid initializer identity changed")
    _, canonical_results, coverage_executions = _low_rank_coverage(capability_record)
    if (
        len(canonical_results) != 8
        or len(results) != 8
        or any(
            canonical is not registered
            for canonical, registered in zip(
                canonical_results,
                results,
                strict=True,
            )
        )
    ):
        raise OptionalExecutionAuthorityError("hybrid score-result inventory changed")
    expected_execution = tuple(
        execution
        for execution in coverage_executions
        if execution.trial.configuration is initializer
    )
    if len(expected_execution) != 1:
        raise OptionalExecutionAuthorityError(
            "hybrid initializer execution is not unique in coverage history"
        )
    if not 1 <= binding.fold_ordinal <= len(expected_execution[0].folds):
        raise OptionalExecutionAuthorityError("hybrid initializer fold is unavailable")
    expected_fold = expected_execution[0].folds[binding.fold_ordinal - 1]
    if (
        expected_fold.spec is not fold.spec
        or expected_fold.fit is not initializer_fit
        or type(expected_fold.fit) is not optional.LowRankOptimizerResult
    ):
        raise OptionalExecutionAuthorityError(
            "hybrid initializer is not the exact registered same-fold coverage fit"
        )
    return binding.binding_digest


def _build_low_rank_specification(
    fold: executor.PreparedFold,
    configuration: controller.Configuration,
    cache: executor.PreparedTrialCache,
) -> optional.LowRankSpec:
    if configuration.rank is None:
        raise OptionalAdapterError("low-rank configuration lacks a rank")
    scales = optional.fold_low_rank_scales(
        fold.fixed_m1.theta,
        fold.training,
        cache.candidates,
        cache.geometry,
        cache.pairs,
    )
    return optional.LowRankSpec(
        int(configuration.rank),
        float(configuration.rho),
        float(scales.lambda_reference),
    )


def _build_hybrid_specification(
    fold: executor.PreparedFold,
    configuration: controller.Configuration,
    cache: executor.PreparedTrialCache,
) -> optional.HybridSpec:
    if configuration.rank is None or configuration.sparse_rho is None:
        raise OptionalAdapterError("hybrid configuration lacks rank or sparse rho")
    scales = optional.fold_low_rank_scales(
        fold.fixed_m1.theta,
        fold.training,
        cache.candidates,
        cache.geometry,
        cache.pairs,
    )
    maximum = executor._validate_lambda_maximum(
        core.fold_lambda_maximum(
            fold.fixed_m1.theta,
            fold.training,
            cache.candidates,
            cache.geometry,
            cache.pairs,
            "lasso",
        )
    )
    try:
        named_edges = tuple(core.PAIR_INDEX[index] for index in configuration.sparse_edges)
    except (IndexError, TypeError) as error:
        raise OptionalAdapterError("hybrid named-edge indexes are invalid") from error
    return optional.HybridSpec(
        int(configuration.rank),
        float(configuration.rho),
        float(configuration.sparse_rho),
        float(scales.lambda_reference),
        float(maximum.value),
        named_edges,
    )


def _score_fold(
    fold: executor.PreparedFold,
    beta: np.ndarray,
    cache: executor.PreparedTrialCache,
) -> tuple[float, int, tuple[executor.CellTrialResult, ...], tuple[controller.PairContrastRow, ...]]:
    prediction = core.evaluate_m2(
        beta,
        fold.fixed_m1.theta,
        fold.validation,
        cache.candidates,
        cache.geometry,
        cache.pairs,
        need_predictions=True,
    )
    if prediction.log_probability_nats is None:
        raise OptionalAdapterError("optional held-out log probabilities were omitted")
    log_probability = executor._readonly_copy(
        prediction.log_probability_nats,
        shape=(len(fold.validation.cells),),
        label="optional held-out log probability",
    )
    fitted_pairs = executor._pair_probabilities(
        beta,
        fold.fixed_m1.theta,
        fold.validation,
        cache.candidates,
        cache.geometry,
        cache.pairs,
    )
    pair_contrasts = fitted_pairs - fold.baseline_pair_probabilities
    gains = (log_probability - fold.baseline_log_probability_nats) / LN2
    if not np.all(np.isfinite(gains)):
        raise OptionalAdapterError("optional held-out gains are nonfinite")
    cells: list[executor.CellTrialResult] = []
    rows: list[controller.PairContrastRow] = []
    for local_index, (global_index, cell) in enumerate(
        zip(fold.spec.validation_indexes, fold.validation.cells, strict=True)
    ):
        if cell.k in {0, core.m1.PANEL_SIZE} and float(
            log_probability[local_index]
        ) != float(fold.baseline_log_probability_nats[local_index]):
            raise OptionalAdapterError("K in {0,16} must have exact-zero optional gain")
        contrast = np.asarray(pair_contrasts[local_index], dtype=float)
        if cell.k in {0, 1, core.m1.PANEL_SIZE}:
            contrast = np.zeros(core.PAIR_DIMENSION, dtype=float)
        values = tuple(float(value) for value in contrast)
        row = controller.make_pair_contrast_row(
            cell_key=str(cell.identifier),
            group=str(cell.group),
            k=int(cell.k),
            values=values,
        )
        rows.append(row)
        cells.append(
            executor.CellTrialResult(
                global_index=int(global_index),
                cell_key=str(cell.identifier),
                group=str(cell.group),
                k=int(cell.k),
                m1_log_probability_nats=float(
                    fold.baseline_log_probability_nats[local_index]
                ),
                m2_log_probability_nats=float(log_probability[local_index]),
                pair_probability_contrasts=values,
            )
        )
    return (
        math.fsum(float(value) for value in gains) / len(gains),
        int(np.count_nonzero(np.asarray(beta, dtype=float))),
        tuple(cells),
        tuple(rows),
    )


def _execute_numerical_attempt(
    capability: OptionalAttemptCapability,
    record: _CapabilityRecord,
) -> OptionalTrialExecution:
    adapter = record.adapter_reference()
    session = record.session_reference()
    authority = record.authority_reference()
    owner = record.owner_reference()
    cache = record.cache_reference()
    trial = record.trial_reference()
    if (
        adapter is None
        or session is None
        or authority is None
        or owner is None
        or cache is None
        or trial is None
        or type(trial.configuration) is not controller.Configuration
    ):
        raise OptionalExecutionAuthorityError("optional attempt context is unavailable")
    configuration = trial.configuration
    fold_results: list[OptionalFoldTrialExecution] = []
    rows: list[controller.PairContrastRow] = []
    seen_indexes: set[int] = set()

    evidence: controller.CoverageEvidence | None = None
    score_results: tuple[controller.TrialResult, ...] = ()
    coverage_executions: tuple[OptionalTrialExecution, ...] = ()
    initializer_configuration: controller.Configuration | None = None
    if configuration.family == "sparse_plus_low_rank":
        evidence, score_results, coverage_executions = _low_rank_coverage(record)
        initializer_configuration = optional.derive_hybrid_low_rank_initializer_configuration(
            evidence,
            score_results,
            configuration,
        )

    for fold_index, fold in enumerate(cache.folds):
        hybrid_authority: object | None = None
        hybrid_receipt: object | None = None
        initializer_fit: optional.LowRankOptimizerResult | None = None
        if configuration.family == "low_rank":
            specification: optional.LowRankSpec | optional.HybridSpec = (
                _build_low_rank_specification(fold, configuration, cache)
            )
            fit: optional.LowRankOptimizerResult | optional.HybridOptimizerResult = (
                optional.fit_low_rank(
                    fold.fixed_m1.theta,
                    fold.training,
                    cache.candidates,
                    cache.geometry,
                    cache.pairs,
                    specification,
                    maximum_iterations=owner.maximum_iterations,
                )
            )
            fold_initializer = None
        elif configuration.family == "sparse_plus_low_rank":
            assert evidence is not None and initializer_configuration is not None
            matching = tuple(
                execution
                for execution in coverage_executions
                if execution.trial.configuration is initializer_configuration
            )
            if len(matching) != 1:
                raise OptionalExecutionAuthorityError(
                    "hybrid initializer lacks one exact low-rank coverage execution"
                )
            initializer_execution = matching[0]
            if len(initializer_execution.folds) != executor.EXPECTED_FOLDS:
                raise OptionalExecutionAuthorityError(
                    "low-rank initializer execution lost its fold inventory"
                )
            initializer_fold = initializer_execution.folds[fold_index]
            if (
                initializer_fold.spec is not fold.spec
                or type(initializer_fold.fit) is not optional.LowRankOptimizerResult
            ):
                raise OptionalExecutionAuthorityError(
                    "hybrid initializer is from another outer fold"
                )
            initializer_fit = initializer_fold.fit
            specification = _build_hybrid_specification(fold, configuration, cache)
            hybrid_authority = _issue_hybrid_solver_authority(
                capability,
                fold,
                evidence,
                score_results,
                configuration,
                initializer_configuration,
                initializer_fit,
                specification,
            )
            hybrid_receipt = optional._authorize_session_production_hybrid(
                evidence,
                score_results,
                initializer_fit,
                configuration,
                specification,
                hybrid_authority,
            )
            fit = optional.fit_sparse_plus_low_rank(
                fold.fixed_m1.theta,
                fold.training,
                cache.candidates,
                cache.geometry,
                cache.pairs,
                specification,
                initializer_fit,
                production_receipt=hybrid_receipt,
                maximum_sweeps=owner.maximum_iterations,
            )
            fold_initializer = initializer_configuration
        else:  # pragma: no cover - issuance rejects this first
            raise OptionalAdapterError("optional adapter received a nonoptional family")

        gain, active_degrees, cells, fold_rows = _score_fold(
            fold,
            fit.beta,
            cache,
        )
        for cell in cells:
            if cell.global_index in seen_indexes:
                raise OptionalAdapterError("an optional cell received two held-out scores")
            seen_indexes.add(cell.global_index)
        rows.extend(fold_rows)
        fold_results.append(
            OptionalFoldTrialExecution(
                spec=fold.spec,
                specification=specification,
                initializer_configuration=fold_initializer,
                initializer_fit=initializer_fit,
                hybrid_authority=hybrid_authority,
                hybrid_receipt=hybrid_receipt,
                fit=fit,
                gain_bits_per_cell=float(gain),
                active_pair_degrees=active_degrees,
                cells=cells,
            )
        )

    if seen_indexes != set(range(len(cache.cell_inventory))):
        raise OptionalAdapterError("optional outer cross-fit did not score every cell once")
    rows.sort(key=lambda row: row.cell_key)
    contrasts = executor._aggregate_pair_contrasts(rows)
    gain = math.fsum(fold.gain_bits_per_cell for fold in fold_results) / len(fold_results)
    active_degrees = max(fold.active_pair_degrees for fold in fold_results)
    result = controller.TrialResult(
        trial=trial,
        gain_bits_per_cell=float(gain),
        active_pair_degrees=active_degrees,
    )
    return OptionalTrialExecution(
        trial=trial,
        folds=tuple(fold_results),
        pair_contrast_rows=tuple(rows),
        equal_group_pair_contrasts=contrasts,
        result=result,
        mode=executor.SEALED_PRODUCTION_MODE,
        session_identity=id(session),
        owner_identity=id(owner),
        cache_identity=id(cache),
        authority_identity=id(authority),
        attempt_ordinal=record.optional_attempt_ordinal,
    )


def _execute_optional_attempt(
    adapter: OptionalFamilyAdapter,
    capability: OptionalAttemptCapability,
) -> OptionalTrialExecution:
    """Consume one capability before solver entry and burn it on BaseException."""

    record, state = _require_capability_context(capability, expected_phase="issued")
    if record.adapter_reference() is not adapter:
        raise OptionalExecutionAuthorityError("optional capability belongs to another adapter")
    session = record.session_reference()
    authority = record.authority_reference()
    owner = record.owner_reference()
    trial = record.trial_reference()
    accepted = _observed_request(
        "_optional_adapter_execute_request",
        session,
        authority,
        adapter,
        owner,
        trial,
        capability,
        record.binding_digest,
    )
    if accepted is not True:
        raise OptionalExecutionAuthorityError("observed session rejected capability use")
    with state.lock:
        if state.current_capability is not capability or record.phase != "issued":
            raise OptionalExecutionAuthorityError("optional capability was already consumed")
        record.phase = "in_flight"
    try:
        execution = _execute_numerical_attempt(capability, record)
        _register_completed_execution(adapter, capability, execution)
    except BaseException as error:
        with state.lock:
            if state.current_capability is not capability or record.phase != "in_flight":
                raise OptionalExecutionAuthorityError(
                    "optional capability could not be burned after failure"
                ) from error
            record.completion = "error"
            record.error_type = f"{type(error).__module__}.{type(error).__qualname__}"
            record.phase = "completed_unacked"
        raise
    with state.lock:
        if state.current_capability is not capability or record.phase != "in_flight":
            raise OptionalExecutionAuthorityError(
                "optional capability completion is not atomic"
            )
        record.completion = "success"
        record.execution_reference = weakref.ref(execution)
        record.execution_digest = executor._deep_integrity_digest(execution)
        record.phase = "completed_unacked"
    return execution


def _register_completed_execution(
    adapter: OptionalFamilyAdapter,
    capability: OptionalAttemptCapability,
    execution: OptionalTrialExecution,
) -> None:
    record, state = _require_capability_context(capability, expected_phase="in_flight")
    if (
        type(execution) is not OptionalTrialExecution
        or execution.trial is not record.trial_reference()
        or execution.attempt_ordinal != record.optional_attempt_ordinal
    ):
        raise OptionalExecutionAuthorityError("optional execution differs from capability")
    binding = _ExecutionBinding(
        adapter_reference=weakref.ref(adapter),
        session_reference=record.session_reference,
        authority_reference=record.authority_reference,
        owner_reference=record.owner_reference,
        cache_reference=record.cache_reference,
        capability_reference=weakref.ref(capability),
        capability_digest=record.binding_digest,
        execution_digest=executor._deep_integrity_digest(execution),
    )
    _register_execution(execution, binding)
    assert state.result_records is not None
    state.result_records[execution.trial.valid_index] = (
        weakref.ref(execution.result),
        weakref.ref(execution),
    )


def _completed_optional_attempt_binding(
    adapter: OptionalFamilyAdapter,
    capability: OptionalAttemptCapability,
    expected_completion: str,
) -> tuple[int, bytes, str | None]:
    record, state = _require_live_capability(capability)
    if (
        record.adapter_reference() is not adapter
        or state.current_capability is not capability
        or record.phase != "completed_unacked"
        or record.completion != expected_completion
    ):
        raise OptionalExecutionAuthorityError(
            "optional capability is not ready for ledger acknowledgment"
        )
    if expected_completion == "success":
        execution = None if record.execution_reference is None else record.execution_reference()
        if (
            execution is None
            or record.execution_digest != executor._deep_integrity_digest(execution)
        ):
            raise OptionalExecutionAuthorityError("optional execution changed before commit")
    return record.optional_attempt_ordinal, record.binding_digest, record.error_type


def _acknowledge_optional_attempt(
    adapter: OptionalFamilyAdapter,
    session: object,
    authority: object,
    capability: OptionalAttemptCapability,
    attempt: object,
) -> tuple[int, bytes]:
    record, state = _require_live_capability(capability)
    accepted = _observed_request(
        "_optional_adapter_ack_request",
        session,
        authority,
        adapter,
        capability,
        attempt,
        record.binding_digest,
        record.session_attempt_ordinal,
        record.session_version,
    )
    expected = {"valid"} if record.completion == "success" else {"invalid", "resource"}
    with state.lock:
        if (
            accepted is not True
            or record.adapter_reference() is not adapter
            or record.session_reference() is not session
            or record.authority_reference() is not authority
            or state.current_capability is not capability
            or record.phase != "completed_unacked"
            or getattr(attempt, "outcome", None) not in expected
            or getattr(attempt, "trial", None) is not record.trial_reference()
            or getattr(attempt, "execution_capability", None) is not capability
            or getattr(attempt, "execution_capability_digest", None)
            != record.binding_digest
            or getattr(attempt, "production_attempt_ordinal", None)
            != record.optional_attempt_ordinal
            or record.optional_attempt_ordinal != state.acknowledged_attempts + 1
        ):
            raise OptionalExecutionAuthorityError(
                "observed ledger did not exactly acknowledge optional capability"
            )
        record.attempt_reference = weakref.ref(attempt)
        record.attempt_digest = executor._deep_integrity_digest(attempt)
        record.phase = "ledger_acked"
        state.acknowledged_attempts += 1
        return record.optional_attempt_ordinal, record.binding_digest


def _require_acknowledged_optional_attempt(
    adapter: OptionalFamilyAdapter,
    session: object,
    authority: object,
    capability: OptionalAttemptCapability,
    attempt: object,
) -> tuple[int, bytes]:
    record, state = _require_live_capability(capability)
    if (
        record.adapter_reference() is not adapter
        or record.session_reference() is not session
        or record.authority_reference() is not authority
        or record.phase != "ledger_acked"
        or record.attempt_reference is None
        or record.attempt_reference() is not attempt
        or record.attempt_digest != executor._deep_integrity_digest(attempt)
        or getattr(attempt, "execution_capability_digest", None)
        != record.binding_digest
    ):
        raise OptionalExecutionAuthorityError(
            "optional capability ledger acknowledgment changed"
        )
    return record.optional_attempt_ordinal, record.binding_digest


def _require_optional_attempt_count(
    adapter: OptionalFamilyAdapter,
    session: object,
    authority: object,
    acknowledged_attempts: int,
    *,
    allow_pending: bool,
) -> None:
    state = _require_adapter(adapter)
    with state.lock:
        if (
            state.session_reference is None
            or state.session_reference() is not session
            or state.authority_reference is None
            or state.authority_reference() is not authority
            or state.acknowledged_attempts != acknowledged_attempts
        ):
            raise OptionalExecutionAuthorityError(
                "optional adapter/session attempt count changed"
            )
        current = state.current_capability
        if current is None:
            return
        record = _lookup_capability(current)
        allowed = {"ledger_acked"}
        if allow_pending:
            allowed.update({"issued", "in_flight", "completed_unacked"})
        if record.phase not in allowed or _capability_binding(record) != record.binding_digest:
            raise OptionalExecutionAuthorityError(
                "optional adapter has an unacknowledged off-chronology capability"
            )


def _require_exact_hybrid_execution_binding(
    fold_execution: OptionalFoldTrialExecution,
    configuration: controller.Configuration,
    specification: optional.HybridSpec,
    fit_binding: optional._HybridResultBinding,
    receipt_binding: optional._ProductionHybridBinding,
) -> None:
    """Bind retained hybrid output to its exact reported session initializer."""

    if (
        type(fold_execution) is not OptionalFoldTrialExecution
        or type(configuration) is not controller.Configuration
        or type(specification) is not optional.HybridSpec
        or type(fit_binding) is not optional._HybridResultBinding
        or type(receipt_binding) is not optional._ProductionHybridBinding
    ):
        raise OptionalExecutionAuthorityError(
            "hybrid execution binding has the wrong exact type"
        )
    initializer_configuration = fold_execution.initializer_configuration
    initializer_fit = fold_execution.initializer_fit
    hybrid_authority = fold_execution.hybrid_authority
    hybrid_receipt = fold_execution.hybrid_receipt
    fit_receipt_reference = fit_binding.production_receipt_reference
    receipt_authority_reference = receipt_binding.session_authority_reference
    evidence = receipt_binding.coverage_reference()
    anchor_results = tuple(
        reference() for reference in receipt_binding.low_rank_result_references
    )
    receipt_anchor = receipt_binding.selected_anchor_reference()
    target_configuration = receipt_binding.target_configuration_reference()
    target_specification = receipt_binding.target_specification_reference()
    if (
        type(initializer_configuration) is not controller.Configuration
        or type(initializer_fit) is not optional.LowRankOptimizerResult
        or fit_binding.selected_anchor_reference() is not initializer_fit
        or fit_receipt_reference is None
        or fit_receipt_reference() is not hybrid_receipt
        or receipt_anchor is not initializer_fit
        or receipt_authority_reference is None
        or receipt_authority_reference() is not hybrid_authority
        or evidence is None
        or len(anchor_results) != 8
        or any(result is None for result in anchor_results)
        or target_configuration is not configuration
        or target_specification is not specification
    ):
        raise OptionalExecutionAuthorityError(
            "hybrid fit/receipt/authority/initializer identity changed"
        )
    try:
        derived_initializer = (
            optional.derive_hybrid_low_rank_initializer_configuration(
                evidence,
                anchor_results,
                target_configuration,
            )
        )
    except optional.LowRankError as error:
        raise OptionalExecutionAuthorityError(
            "hybrid receipt initializer derivation no longer validates"
        ) from error
    if derived_initializer is not initializer_configuration:
        raise OptionalExecutionAuthorityError(
            "hybrid receipt does not bind the exact reported initializer configuration"
        )


def _require_registered_execution(
    adapter: OptionalFamilyAdapter,
    execution: OptionalTrialExecution,
) -> controller.TrialResult:
    if type(execution) is not OptionalTrialExecution:
        raise OptionalExecutionAuthorityError(
            "optional history lacks an OptionalTrialExecution"
        )
    binding = _lookup_execution(execution)
    if type(binding) is not _ExecutionBinding:
        raise OptionalExecutionAuthorityError("optional execution provenance is malformed")
    owner = binding.owner_reference()
    cache = binding.cache_reference()
    session = binding.session_reference()
    authority = binding.authority_reference()
    capability = binding.capability_reference()
    state = _require_adapter(adapter)
    configuration = execution.trial.configuration
    if (
        binding.adapter_reference() is not adapter
        or owner is None
        or cache is None
        or session is None
        or authority is None
        or capability is None
        or state.owner_reference() is not owner
        or owner.prepared is not cache
        or state.session_reference is None
        or state.session_reference() is not session
        or state.authority_reference is None
        or state.authority_reference() is not authority
        or executor._deep_integrity_digest(execution) != binding.execution_digest
        or execution.mode != executor.SEALED_PRODUCTION_MODE
        or execution.session_identity != id(session)
        or execution.owner_identity != id(owner)
        or execution.cache_identity != id(cache)
        or execution.authority_identity != id(authority)
        or execution.result.trial is not execution.trial
        or not execution.result.valid
        or execution.result.coverage_binding is not None
        or type(configuration) is not controller.Configuration
        or configuration.family not in OPTIONAL_FAMILIES
        or len(execution.folds) != executor.EXPECTED_FOLDS
        or any(
            observed.spec is not prepared.spec
            for observed, prepared in zip(
                execution.folds,
                cache.folds,
                strict=True,
            )
        )
        or sum(len(fold.cells) for fold in execution.folds) != len(cache.cell_inventory)
        or len(execution.pair_contrast_rows) != len(cache.cell_inventory)
        or len(execution.equal_group_pair_contrasts) != core.PAIR_DIMENSION
    ):
        raise OptionalExecutionAuthorityError(
            "optional execution owner/cache/session/fold/detail binding changed"
        )
    capability_record, _ = _require_live_capability(capability)
    if (
        capability_record.binding_digest != binding.capability_digest
        or execution.attempt_ordinal != capability_record.optional_attempt_ordinal
        or capability_record.execution_reference is None
        or capability_record.execution_reference() is not execution
        or capability_record.execution_digest != binding.execution_digest
        or capability_record.phase not in {"completed_unacked", "ledger_acked"}
    ):
        raise OptionalExecutionAuthorityError(
            "optional execution capability provenance changed"
        )
    expected_indexes = set(range(len(cache.cell_inventory)))
    observed_indexes = {
        cell.global_index for fold in execution.folds for cell in fold.cells
    }
    if observed_indexes != expected_indexes:
        raise OptionalExecutionAuthorityError("optional execution cell inventory changed")
    all_cells: list[executor.CellTrialResult] = []
    for prepared_fold, fold_execution in zip(cache.folds, execution.folds, strict=True):
        fit = fold_execution.fit
        specification = fold_execution.specification
        if type(fit) is optional.LowRankOptimizerResult:
            if (
                configuration.family != "low_rank"
                or type(specification) is not optional.LowRankSpec
                or specification
                != _build_low_rank_specification(prepared_fold, configuration, cache)
                or fit.state.rank != specification.rank
                or fit.telemetry.get("objective_mode") != "production"
                or fit.telemetry.get("rank") != specification.rank
                or fit.telemetry.get("rho") != float(specification.rho)
                or fit.telemetry.get("lambda_reference")
                != float(specification.lambda_reference)
            ):
                raise OptionalExecutionAuthorityError("low-rank fold specification changed")
            fit_binding = optional._require_registered_low_rank_result(fit, "production")
            optional._require_same_objective(
                fit_binding.objective,
                prepared_fold.fixed_m1.theta,
                prepared_fold.training,
                cache.candidates,
                cache.geometry,
                cache.pairs,
                optional._low_rank_specification_signature(specification),
                None,
            )
            if any(
                value is not None
                for value in (
                    fold_execution.initializer_configuration,
                    fold_execution.initializer_fit,
                    fold_execution.hybrid_authority,
                    fold_execution.hybrid_receipt,
                )
            ):
                raise OptionalExecutionAuthorityError(
                    "low-rank execution carries hybrid provenance"
                )
        elif type(fit) is optional.HybridOptimizerResult:
            if (
                configuration.family != "sparse_plus_low_rank"
                or type(specification) is not optional.HybridSpec
                or specification
                != _build_hybrid_specification(prepared_fold, configuration, cache)
                or type(fold_execution.initializer_configuration)
                is not controller.Configuration
                or type(fold_execution.initializer_fit)
                is not optional.LowRankOptimizerResult
                or fold_execution.hybrid_authority is None
                or fold_execution.hybrid_receipt is None
                or fit.telemetry.get("objective_mode") != "production"
                or fit.telemetry.get("rank") != specification.rank
                or fit.telemetry.get("rho_low") != float(specification.rho_low)
                or fit.telemetry.get("rho_sparse") != float(specification.rho_sparse)
                or fit.telemetry.get("lambda_reference")
                != float(specification.lambda_reference)
                or fit.telemetry.get("core_lasso_lambda_maximum")
                != float(specification.core_lasso_lambda_maximum)
                or fit.telemetry.get("named_edges") != specification.named_edges
            ):
                raise OptionalExecutionAuthorityError("hybrid fold provenance is incomplete")
            fit_binding = optional._require_registered_hybrid_result(fit, "production")
            optional._require_same_objective(
                fit_binding.objective,
                prepared_fold.fixed_m1.theta,
                prepared_fold.training,
                cache.candidates,
                cache.geometry,
                cache.pairs,
                optional._hybrid_specification_signature(specification),
                None,
            )
            receipt_binding = optional._require_production_hybrid_receipt(
                fold_execution.hybrid_receipt,
                specification,
                fold_execution.initializer_fit,
            )
            _require_exact_hybrid_execution_binding(
                fold_execution,
                configuration,
                specification,
                fit_binding,
                receipt_binding,
            )
        else:
            raise OptionalExecutionAuthorityError("optional fold fit type changed")

        expected_cells = tuple(
            zip(
                prepared_fold.spec.validation_indexes,
                prepared_fold.validation.cells,
                prepared_fold.baseline_log_probability_nats,
                strict=True,
            )
        )
        if len(fold_execution.cells) != len(expected_cells):
            raise OptionalExecutionAuthorityError(
                "optional fold score inventory changed"
            )
        for cell_result, (global_index, cell, baseline) in zip(
            fold_execution.cells,
            expected_cells,
            strict=True,
        ):
            values = cell_result.pair_probability_contrasts
            if (
                type(cell_result) is not executor.CellTrialResult
                or cell_result.global_index != global_index
                or cell_result.cell_key != str(cell.identifier)
                or cell_result.group != str(cell.group)
                or cell_result.k != int(cell.k)
                or cell_result.m1_log_probability_nats != float(baseline)
                or not math.isfinite(cell_result.m2_log_probability_nats)
                or type(values) is not tuple
                or len(values) != core.PAIR_DIMENSION
                or any(type(value) is not float for value in values)
                or not all(math.isfinite(value) for value in values)
                or (
                    cell_result.k in {0, 1, core.m1.PANEL_SIZE}
                    and any(value != 0.0 for value in values)
                )
                or (
                    cell_result.k in {0, core.m1.PANEL_SIZE}
                    and cell_result.m2_log_probability_nats
                    != cell_result.m1_log_probability_nats
                )
            ):
                raise OptionalExecutionAuthorityError(
                    "optional held-out cell score or physical contrast changed"
                )
        expected_gain = math.fsum(
            (cell.m2_log_probability_nats - cell.m1_log_probability_nats) / LN2
            for cell in fold_execution.cells
        ) / len(fold_execution.cells)
        if (
            fold_execution.gain_bits_per_cell != expected_gain
            or fold_execution.active_pair_degrees
            != int(np.count_nonzero(np.asarray(fit.beta, dtype=float)))
        ):
            raise OptionalExecutionAuthorityError(
                "optional fold summary differs from its retained scores and fit"
            )
        all_cells.extend(fold_execution.cells)

    ordered_cells = tuple(sorted(all_cells, key=lambda cell: cell.cell_key))
    if len({cell.cell_key for cell in ordered_cells}) != len(ordered_cells):
        raise OptionalExecutionAuthorityError("optional cell keys are not unique")
    for row, cell in zip(
        execution.pair_contrast_rows,
        ordered_cells,
        strict=True,
    ):
        try:
            controller._require_canonical_pair_contrast_row(row)
        except (controller.ControllerError, TypeError, ValueError) as error:
            raise OptionalExecutionAuthorityError(
                "optional physical pair-contrast row lost controller authority"
            ) from error
        if (
            row.cell_key != cell.cell_key
            or row.group != cell.group
            or row.k != cell.k
            or row.values != cell.pair_probability_contrasts
        ):
            raise OptionalExecutionAuthorityError(
                "optional physical pair-contrast rows differ from retained cells"
            )
    if (
        executor._aggregate_pair_contrasts(execution.pair_contrast_rows)
        != execution.equal_group_pair_contrasts
        or execution.result.gain_bits_per_cell
        != math.fsum(fold.gain_bits_per_cell for fold in execution.folds)
        / executor.EXPECTED_FOLDS
        or execution.result.active_pair_degrees
        != max(fold.active_pair_degrees for fold in execution.folds)
    ):
        raise OptionalExecutionAuthorityError(
            "optional trial summary differs from its exact 12-fold detail"
        )
    return execution.result


def _require_registered_result(
    adapter: OptionalFamilyAdapter,
    result: controller.TrialResult,
) -> OptionalTrialExecution:
    state = _require_adapter(adapter)
    assert state.result_records is not None
    record = state.result_records.get(result.trial.valid_index)
    if record is None or record[0]() is not result:
        raise OptionalExecutionAuthorityError(
            "optional result was not produced by this adapter"
        )
    execution = record[1]()
    if execution is None or execution.result is not result:
        raise OptionalExecutionAuthorityError("optional detailed execution is unavailable")
    _require_registered_execution(adapter, execution)
    return execution


def _execution_from_receipt(
    adapter: OptionalFamilyAdapter,
    receipt: executor.ExternalExecutionReceipt,
) -> OptionalTrialExecution:
    if (
        type(receipt) is not executor.ExternalExecutionReceipt
        or type(receipt.execution_detail) is not OptionalTrialExecution
    ):
        raise OptionalExecutionAuthorityError(
            "optional session receipt lacks immutable execution detail"
        )
    execution = receipt.execution_detail
    if execution.result is not receipt.result:
        raise OptionalExecutionAuthorityError("optional receipt result/detail diverged")
    _require_registered_execution(adapter, execution)
    return execution


def _is_optional_attempt_capability(value: object) -> bool:
    return type(value) is OptionalAttemptCapability and getattr(
        value, "_seal", None
    ) is _CAPABILITY_SEAL


__all__ = (
    "OptionalAdapterError",
    "OptionalExecutionAuthorityError",
    "OptionalFamilyAdapter",
    "OptionalFoldTrialExecution",
    "OptionalTrialExecution",
    "create_optional_family_adapter",
)
