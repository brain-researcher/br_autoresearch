#!/usr/bin/env python3
"""Named production authority for EP10 v3 Rademacher-reference trials.

The lower-level :mod:`v3_m2_rademacher_adapter_v1` owns the exact immutable
dictionary and the fold transform.  This module adds the process-local
controller authority: it binds one sealed core cache, its exact live source
cells and uniform moments, one dictionary per controller trial objective, a
one-shot attempt, and the complete twelve-fold/194-score execution record.

Production has no callback surface.  A deliberately private synthetic issuer
exists only for lightweight, target-free authority tests.  It is never
accepted by the observed-session hooks or by the sealed-production validator.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(os.path.abspath(__file__))
TOOLS_DIR = SCRIPT_PATH.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m1 as m1  # noqa: E402
import fit_v3_m2 as m2  # noqa: E402
import fit_v3_m2_optional as optional  # noqa: E402
import v3_m2_rademacher_adapter_v1 as rademacher  # noqa: E402
import v3_m2_trial_executor_v1 as executor  # noqa: E402
import v3_search_controller_v1 as controller  # noqa: E402


EXPECTED_ROWS = executor.EXPECTED_PRODUCTION_ROWS
EXPECTED_FOLDS = executor.EXPECTED_FOLDS
EXPECTED_RANK = rademacher.EXPECTED_PROJECTED_RANK
ALL_MATCHED_FAMILIES = frozenset(
    (*controller.CORE_FAMILIES, "low_rank", "sparse_plus_low_rank")
)
LN2 = math.log(2.0)

_ADAPTER_SEAL = object()
_CAPABILITY_SEAL = object()
_PRECURSOR_RECEIPT_SEAL = object()


class RademacherProductionError(m2.M2Error):
    """A named Rademacher numerical execution failed its frozen contract."""


class RademacherExecutionAuthorityError(RademacherProductionError):
    """A live adapter/session/cache/trial/attempt binding failed closed."""


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

        try:
            reference = weakref.ref(instance, remove)
        except TypeError as error:
            raise RademacherExecutionAuthorityError(
                f"{label} does not support live identity binding"
            ) from error
        with lock:
            current = records.get(identity)
            if current is not None and current[0]() is not instance:
                raise RademacherExecutionAuthorityError(f"{label} identity collision")
            records[identity] = (reference, metadata)

    def lookup(instance: Any) -> Any:
        with lock:
            current = records.get(id(instance))
            if current is None or current[0]() is not instance:
                raise RademacherExecutionAuthorityError(
                    f"{label} lacks named-adapter issuance"
                )
            return current[1]

    return register, lookup


_register_adapter, _lookup_adapter = _make_weak_identity_registry(
    "Rademacher-reference adapter"
)
_register_capability, _lookup_capability = _make_weak_identity_registry(
    "Rademacher attempt capability"
)
_register_execution, _lookup_execution = _make_weak_identity_registry(
    "Rademacher trial execution"
)
_register_precursor_receipt, _lookup_precursor_receipt = (
    _make_weak_identity_registry("Rademacher hybrid precursor receipt")
)


def _weak(value: Any, label: str) -> weakref.ReferenceType[Any]:
    try:
        return weakref.ref(value)
    except TypeError as error:
        raise RademacherExecutionAuthorityError(
            f"{label} does not support process-local authority"
        ) from error


class RademacherReferenceAdapter:
    """Opaque named authority for one exact core executor and source view."""

    __slots__ = ("_seal", "__weakref__")

    def __init__(self, *, _seal: object | None = None) -> None:
        if _seal is not _ADAPTER_SEAL:
            raise RademacherExecutionAuthorityError(
                "Rademacher adapters require named factory issuance"
            )
        object.__setattr__(self, "_seal", _seal)

    def __copy__(self) -> None:
        raise RademacherExecutionAuthorityError("Rademacher adapters cannot be copied")

    def __deepcopy__(self, memo: Any) -> None:
        del memo
        raise RademacherExecutionAuthorityError("Rademacher adapters cannot be copied")

    def __reduce__(self) -> None:
        raise RademacherExecutionAuthorityError(
            "Rademacher adapters cannot be serialized"
        )


class RademacherAttemptCapability:
    """Opaque consume-once authority for one exact controller attempt."""

    __slots__ = ("_seal", "__weakref__")

    def __init__(self, *, _seal: object | None = None) -> None:
        if _seal is not _CAPABILITY_SEAL:
            raise RademacherExecutionAuthorityError(
                "Rademacher capabilities require named session issuance"
            )
        object.__setattr__(self, "_seal", _seal)

    def __copy__(self) -> None:
        raise RademacherExecutionAuthorityError(
            "Rademacher capabilities cannot be copied"
        )

    def __deepcopy__(self, memo: Any) -> None:
        del memo
        raise RademacherExecutionAuthorityError(
            "Rademacher capabilities cannot be copied"
        )

    def __reduce__(self) -> None:
        raise RademacherExecutionAuthorityError(
            "Rademacher capabilities cannot be serialized"
        )


class _RademacherHybridPrecursorReceipt:
    """Opaque same-objective precursor binding retained by one hybrid fold."""

    __slots__ = ("_seal", "__weakref__")

    def __init__(self, *, _seal: object | None = None) -> None:
        if _seal is not _PRECURSOR_RECEIPT_SEAL:
            raise RademacherExecutionAuthorityError(
                "Rademacher precursor receipts require adapter issuance"
            )
        object.__setattr__(self, "_seal", _seal)

    def __copy__(self) -> None:
        raise RademacherExecutionAuthorityError("precursor receipts cannot be copied")

    def __deepcopy__(self, memo: Any) -> None:
        del memo
        raise RademacherExecutionAuthorityError("precursor receipts cannot be copied")

    def __reduce__(self) -> None:
        raise RademacherExecutionAuthorityError(
            "precursor receipts cannot be serialized"
        )


@dataclass(frozen=True)
class RademacherCellScore:
    """One held-out noise-objective score; no biological pair output exists."""

    global_index: int
    cell_key: str
    group: str
    k: int
    m1_log_probability_nats: float
    m2_log_probability_nats: float
    gain_bits: float


@dataclass(frozen=True)
class RademacherFoldExecution:
    """One exact physical fold transformed and fit on the shared dictionary."""

    spec: executor.FoldSpec
    fixed_m1: executor.FixedM1Offset
    rademacher_fit: rademacher.RademacherFoldFit
    validation: m2.PreparedM2
    specification: executor.PenaltyMapping | optional.LowRankSpec | optional.HybridSpec
    precursor_configuration: controller.Configuration | None
    precursor_specification: optional.LowRankSpec | None
    precursor_fit: optional.LowRankOptimizerResult | None
    precursor_receipt: object | None
    fit: m2.M2OptimizerResult | optional.LowRankOptimizerResult | optional.HybridOptimizerResult
    gain_bits_per_cell: float
    active_pair_degrees: int
    cells: tuple[RademacherCellScore, ...]


@dataclass(frozen=True)
class RademacherTrialExecution:
    """Complete immutable record for one selector-ineligible Rademacher trial."""

    trial: controller.Trial
    matched_configuration: controller.Configuration
    seed_route: rademacher.RademacherSeedRoute
    features: rademacher.RademacherFeatureLibrary
    folds: tuple[RademacherFoldExecution, ...]
    result: controller.TrialResult
    mode: str
    session_identity: int
    owner_identity: int
    cache_identity: int
    authority_identity: int
    attempt_ordinal: int
    session_attempt_ordinal: int
    null_replicate_index: int | None
    selector_eligible: bool


@dataclass
class _DictionaryRecord:
    trial_reference: weakref.ReferenceType[controller.Trial]
    trial_digest: bytes
    null_replicate_index: int | None
    route: rademacher.RademacherSeedRoute
    route_digest: bytes
    features: Any
    features_digest: bytes


@dataclass
class _AdapterState:
    owner_reference: weakref.ReferenceType[executor.CoreTrialExecutor] | None
    cache_reference: weakref.ReferenceType[Any]
    candidates_reference: weakref.ReferenceType[m1.CandidateLibrary]
    geometry_reference: weakref.ReferenceType[m1.GeometryEngine]
    pairs_reference: weakref.ReferenceType[m2.PairFeatureLibrary]
    source_cells: tuple[m1.Cell, ...]
    source_moments: tuple[m1.UniformCellMoments, ...]
    source_cells_digest: bytes
    source_moments_digest: bytes
    mode: str
    maximum_iterations: int
    lock: threading.RLock
    synthetic_feature_factory: Callable[[m1.CandidateLibrary, rademacher.RademacherSeedRoute], Any] | None = None
    synthetic_runner: Callable[[RademacherAttemptCapability, Any, Any], tuple[tuple[RademacherFoldExecution, ...], controller.TrialResult]] | None = None
    session_reference: weakref.ReferenceType[Any] | None = None
    authority_reference: weakref.ReferenceType[Any] | None = None
    observed_hooks: tuple[Any, Any, Any, Any] | None = None
    dictionaries: dict[int, _DictionaryRecord] | None = None
    current_capability: RademacherAttemptCapability | None = None
    issued_attempts: int = 0
    acknowledged_attempts: int = 0
    last_session_attempt_ordinal: int = 0
    result_records: dict[int, tuple[weakref.ReferenceType[Any], weakref.ReferenceType[Any]]] | None = None
    validator: _RademacherExecutionValidator | None = None


@dataclass
class _CapabilityRecord:
    adapter_reference: weakref.ReferenceType[RademacherReferenceAdapter]
    session_reference: weakref.ReferenceType[Any]
    authority_reference: weakref.ReferenceType[Any]
    owner_reference: weakref.ReferenceType[executor.CoreTrialExecutor] | None
    cache_reference: weakref.ReferenceType[Any]
    trial_reference: weakref.ReferenceType[controller.Trial]
    trial_digest: bytes
    coverage_reference: weakref.ReferenceType[Any]
    coverage_digest: bytes
    history_references: tuple[weakref.ReferenceType[Any], ...]
    history_digests: tuple[bytes, ...]
    result_references: tuple[weakref.ReferenceType[controller.TrialResult], ...]
    result_digests: tuple[bytes, ...]
    dictionary_reference: weakref.ReferenceType[Any]
    dictionary_digest: bytes
    route_reference: weakref.ReferenceType[rademacher.RademacherSeedRoute]
    route_digest: bytes
    null_replicate_index: int | None
    attempt_ordinal: int
    session_attempt_ordinal: int
    session_version: int
    binding_digest: bytes
    phase: str = "issued"
    completion: str | None = None
    error_type: str | None = None
    execution_reference: weakref.ReferenceType[RademacherTrialExecution] | None = None
    execution_digest: bytes | None = None
    ledger_attempt_reference: weakref.ReferenceType[Any] | None = None
    ledger_attempt_digest: bytes | None = None


@dataclass(frozen=True)
class _ExecutionBinding:
    adapter_reference: weakref.ReferenceType[RademacherReferenceAdapter]
    session_reference: weakref.ReferenceType[Any]
    authority_reference: weakref.ReferenceType[Any]
    owner_reference: weakref.ReferenceType[executor.CoreTrialExecutor] | None
    cache_reference: weakref.ReferenceType[Any]
    capability_reference: weakref.ReferenceType[RademacherAttemptCapability]
    capability_digest: bytes
    execution_digest: bytes


@dataclass(frozen=True)
class _PrecursorBinding:
    capability_reference: weakref.ReferenceType[RademacherAttemptCapability]
    trial_reference: weakref.ReferenceType[controller.Trial]
    coverage_reference: weakref.ReferenceType[controller.CoverageEvidence]
    coverage_digest: bytes
    result_references: tuple[weakref.ReferenceType[controller.TrialResult], ...]
    result_digests: tuple[bytes, ...]
    feature_reference: weakref.ReferenceType[rademacher.RademacherFeatureLibrary]
    rademacher_fold_reference: weakref.ReferenceType[rademacher.RademacherFoldFit]
    physical_fold_reference: weakref.ReferenceType[executor.PreparedFold]
    target_configuration_reference: weakref.ReferenceType[controller.Configuration]
    initializer_configuration_reference: weakref.ReferenceType[controller.Configuration]
    precursor_specification_reference: weakref.ReferenceType[optional.LowRankSpec]
    hybrid_specification_reference: weakref.ReferenceType[optional.HybridSpec]
    precursor_fit_reference: weakref.ReferenceType[optional.LowRankOptimizerResult]
    fold_ordinal: int
    binding_digest: bytes


class _RademacherExecutionValidator:
    """Weak, exact validator retained by one named adapter."""

    __slots__ = ("adapter_reference", "__weakref__")

    def __init__(self, adapter: RademacherReferenceAdapter) -> None:
        self.adapter_reference = weakref.ref(adapter)

    def __call__(
        self,
        producer: RademacherReferenceAdapter,
        result: controller.TrialResult,
    ) -> bool:
        adapter = self.adapter_reference()
        if adapter is None or producer is not adapter or type(result) is not controller.TrialResult:
            return False
        try:
            state = _require_adapter(adapter)
            assert state.result_records is not None
            record = state.result_records.get(result.trial.valid_index)
            if record is None or record[0]() is not result:
                return False
            detail = record[1]()
            return detail is not None and _require_registered_execution(adapter, detail) is result
        except (RademacherProductionError, executor.CoreTrialExecutorError):
            return False

    def validate_execution_detail(
        self,
        producer: RademacherReferenceAdapter,
        result: controller.TrialResult,
        execution_detail: object | None,
    ) -> bool:
        adapter = self.adapter_reference()
        if (
            adapter is None
            or producer is not adapter
            or type(execution_detail) is not RademacherTrialExecution
            or execution_detail.result is not result
        ):
            return False
        try:
            return _require_registered_execution(adapter, execution_detail) is result
        except (RademacherProductionError, executor.CoreTrialExecutorError):
            return False


_VALIDATOR_CALL = _RademacherExecutionValidator.__call__
_VALIDATOR_DETAIL = _RademacherExecutionValidator.validate_execution_detail


def _validate_named_external_execution_detail(
    producer: object,
    validator: object,
    result: controller.TrialResult,
    execution_detail: object,
) -> bool:
    """Exact handshake for executor receipt integration; bare results fail."""
    if (
        type(producer) is not RademacherReferenceAdapter
        or type(validator) is not _RademacherExecutionValidator
        or type(execution_detail) is not RademacherTrialExecution
    ):
        return False
    state = _require_adapter(producer)
    if state.mode != executor.SEALED_PRODUCTION_MODE or state.validator is not validator:
        return False
    return (
        _VALIDATOR_CALL(validator, producer, result) is True
        and _VALIDATOR_DETAIL(validator, producer, result, execution_detail) is True
    )


_OBSERVED_BIND_HOOK = "_rademacher_adapter_bind_request"
_OBSERVED_CAPABILITY_HOOK = "_rademacher_adapter_capability_request"
_OBSERVED_EXECUTE_HOOK = "_rademacher_adapter_execute_request"
_OBSERVED_ACK_HOOK = "_rademacher_adapter_ack_request"
_OBSERVED_HOOK_NAMES = (
    _OBSERVED_BIND_HOOK,
    _OBSERVED_CAPABILITY_HOOK,
    _OBSERVED_EXECUTE_HOOK,
    _OBSERVED_ACK_HOOK,
)


def _observed_hook(name: str) -> Any:
    """Resolve one exact observed-session hook without invoking it."""
    if name not in _OBSERVED_HOOK_NAMES:
        raise RademacherExecutionAuthorityError(
            "unknown observed-session Rademacher hook"
        )
    module_name = "v3_m2_observed_session_v1"
    module = sys.modules.get(module_name)
    expected = TOOLS_DIR / f"{module_name}.py"
    module_path = getattr(module, "SCRIPT_PATH", None)
    module_file = getattr(module, "__file__", None)
    file_path = None if module_file is None else Path(os.path.abspath(os.fspath(module_file)))
    if (
        module is None
        or getattr(module, "__name__", None) != module_name
        or module_path is None
        or Path(os.path.abspath(os.fspath(module_path))) != expected
        or not expected.is_file()
        or expected.is_symlink()
        or file_path != expected
        or file_path.resolve(strict=True) != expected.resolve(strict=True)
    ):
        raise RademacherExecutionAuthorityError(
            "production Rademacher execution requires the exact observed session"
        )
    callback = getattr(module, name, None)
    code = getattr(callback, "__code__", None)
    if (
        not callable(callback)
        or getattr(callback, "__module__", None) != module_name
        or code is None
        or Path(os.path.abspath(code.co_filename)) != expected
    ):
        raise RademacherExecutionAuthorityError(
            f"observed-session Rademacher hook {name!r} is unavailable"
        )
    return callback


_OBSERVED_HOOK = _observed_hook


def _observed_request(
    name: str,
    *arguments: Any,
    expected_callback: Any | None = None,
) -> Any:
    """Call one exact observed-session hook; never fall back generically."""
    if _observed_hook is not _OBSERVED_HOOK:
        raise RademacherExecutionAuthorityError(
            "observed-session Rademacher hook resolver was rebound"
        )
    callback = _OBSERVED_HOOK(name)
    if expected_callback is not None and callback is not expected_callback:
        raise RademacherExecutionAuthorityError(
            f"observed-session Rademacher hook {name!r} changed after binding"
        )
    try:
        return callback(*arguments)
    except RademacherExecutionAuthorityError:
        raise
    except Exception as error:
        raise RademacherExecutionAuthorityError(
            f"observed-session Rademacher hook {name!r} rejected its binding"
        ) from error


_OBSERVED_REQUEST = _observed_request


def _require_observed_dispatch() -> None:
    if (
        _observed_hook is not _OBSERVED_HOOK
        or _observed_request is not _OBSERVED_REQUEST
    ):
        raise RademacherExecutionAuthorityError(
            "observed-session Rademacher request dispatch was rebound"
        )


def _capture_observed_hooks() -> tuple[Any, Any, Any, Any]:
    _require_observed_dispatch()
    return (
        _OBSERVED_HOOK(_OBSERVED_BIND_HOOK),
        _OBSERVED_HOOK(_OBSERVED_CAPABILITY_HOOK),
        _OBSERVED_HOOK(_OBSERVED_EXECUTE_HOOK),
        _OBSERVED_HOOK(_OBSERVED_ACK_HOOK),
    )


def _require_observed_hooks(
    callbacks: tuple[Any, Any, Any, Any] | None,
) -> tuple[Any, Any, Any, Any]:
    _require_observed_dispatch()
    if type(callbacks) is not tuple or len(callbacks) != 4:
        raise RademacherExecutionAuthorityError(
            "Rademacher adapter lacks its exact observed-session hooks"
        )
    for name, expected in zip(_OBSERVED_HOOK_NAMES, callbacks, strict=True):
        if _OBSERVED_HOOK(name) is not expected:
            raise RademacherExecutionAuthorityError(
                f"observed-session Rademacher hook {name!r} changed after binding"
            )
    return callbacks


def _request_observed_bind(
    session: object,
    authority: object,
    adapter: RademacherReferenceAdapter,
    owner: executor.CoreTrialExecutor,
    source_cache: executor.PreparedTrialCache,
) -> tuple[Any, Any, Any, Any]:
    callbacks = _capture_observed_hooks()
    accepted = _OBSERVED_REQUEST(
        _OBSERVED_BIND_HOOK,
        session,
        authority,
        adapter,
        owner,
        source_cache,
        expected_callback=callbacks[0],
    )
    if accepted is not True:
        raise RademacherExecutionAuthorityError(
            "observed session rejected Rademacher adapter binding"
        )
    return callbacks


def _request_observed_capability(
    callbacks: tuple[Any, Any, Any, Any] | None,
    session: object,
    authority: object,
    adapter: RademacherReferenceAdapter,
    owner: executor.CoreTrialExecutor,
    source_cache: executor.PreparedTrialCache,
    trial: controller.Trial,
    coverage_evidence: controller.CoverageEvidence,
    null_replicate_index: int | None,
    session_attempt_ordinal: int,
    session_version: int,
) -> None:
    exact = _require_observed_hooks(callbacks)
    accepted = _OBSERVED_REQUEST(
        _OBSERVED_CAPABILITY_HOOK,
        session,
        authority,
        adapter,
        owner,
        source_cache,
        trial,
        coverage_evidence,
        null_replicate_index,
        session_attempt_ordinal,
        session_version,
        expected_callback=exact[1],
    )
    if accepted is not True:
        raise RademacherExecutionAuthorityError(
            "observed session rejected Rademacher capability issuance"
        )


def _request_observed_execute(
    callbacks: tuple[Any, Any, Any, Any] | None,
    session: object,
    authority: object,
    adapter: RademacherReferenceAdapter,
    owner: executor.CoreTrialExecutor,
    source_cache: executor.PreparedTrialCache,
    trial: controller.Trial,
    coverage_evidence: controller.CoverageEvidence,
    null_replicate_index: int | None,
    capability: RademacherAttemptCapability,
    binding_digest: bytes,
) -> None:
    exact = _require_observed_hooks(callbacks)
    accepted = _OBSERVED_REQUEST(
        _OBSERVED_EXECUTE_HOOK,
        session,
        authority,
        adapter,
        owner,
        source_cache,
        trial,
        coverage_evidence,
        null_replicate_index,
        capability,
        binding_digest,
        expected_callback=exact[2],
    )
    if accepted is not True:
        raise RademacherExecutionAuthorityError(
            "observed session rejected Rademacher capability use"
        )


def _request_observed_ack(
    callbacks: tuple[Any, Any, Any, Any] | None,
    session: object,
    authority: object,
    adapter: RademacherReferenceAdapter,
    owner: executor.CoreTrialExecutor,
    source_cache: executor.PreparedTrialCache,
    trial: controller.Trial,
    coverage_evidence: controller.CoverageEvidence,
    null_replicate_index: int | None,
    capability: RademacherAttemptCapability,
    binding_digest: bytes,
    adapter_attempt_ordinal: int,
    session_attempt_ordinal: int,
    session_version: int,
    completion: str,
    error_type: str | None,
    ledger_attempt: object,
) -> None:
    exact = _require_observed_hooks(callbacks)
    accepted = _OBSERVED_REQUEST(
        _OBSERVED_ACK_HOOK,
        session,
        authority,
        adapter,
        owner,
        source_cache,
        trial,
        coverage_evidence,
        null_replicate_index,
        capability,
        binding_digest,
        adapter_attempt_ordinal,
        session_attempt_ordinal,
        session_version,
        completion,
        error_type,
        ledger_attempt,
        expected_callback=exact[3],
    )
    if accepted is not True:
        raise RademacherExecutionAuthorityError(
            "observed session rejected Rademacher ledger acknowledgment"
        )


_REQUEST_OBSERVED_BIND = _request_observed_bind
_REQUEST_OBSERVED_CAPABILITY = _request_observed_capability
_REQUEST_OBSERVED_EXECUTE = _request_observed_execute
_REQUEST_OBSERVED_ACK = _request_observed_ack


def _source_cells(cache: executor.PreparedTrialCache) -> tuple[m1.Cell, ...]:
    rows: list[m1.Cell | None] = [None] * len(cache.cell_inventory)
    for fold in cache.folds:
        for index, cell in zip(
            fold.spec.validation_indexes, fold.validation.cells, strict=True
        ):
            if not 0 <= index < len(rows) or rows[index] is not None:
                raise RademacherExecutionAuthorityError(
                    "source cache lacks one unique validation owner per cell"
                )
            rows[index] = cell
    if any(cell is None for cell in rows):
        raise RademacherExecutionAuthorityError("source cache cell inventory is incomplete")
    exact = tuple(cell for cell in rows if cell is not None)
    if tuple((str(cell.identifier), str(cell.group)) for cell in exact) != cache.cell_inventory:
        raise RademacherExecutionAuthorityError(
            "source cache cell order differs from its execution inventory"
        )
    return exact


_PHYSICAL_PROVENANCE = m2._require_physical_fold_provenance
_FEATURE_FACTORY = rademacher.RademacherFeatureLibrary
_FEATURE_AUTHENTICATE = rademacher._require_issued_features
_FIT_RADEMACHER_FOLD = rademacher.fit_rademacher_fold
_APPLY_RADEMACHER_VALIDATION = rademacher.apply_rademacher_validation
_CORE_LAMBDA_MAXIMUM = m2.fold_lambda_maximum
_CORE_FIT = m2.fit_m2
_CORE_EVALUATE = m2.evaluate_m2
_CORE_VALIDATE_FIT = executor._validate_optimizer_result
_CORE_VALIDATE_MAXIMUM = executor._validate_lambda_maximum
_CORE_PENALTY = executor.penalty_from_configuration
_OPTIONAL_SCALES = optional.fold_low_rank_scales
_OPTIONAL_FIT_LOW_RANK = optional.fit_low_rank
_OPTIONAL_FIT_HYBRID = optional._fit_sparse_plus_low_rank_impl
_OPTIONAL_DERIVE_PRECURSOR = optional.derive_hybrid_low_rank_initializer_configuration


def _require_fixed_production_functions() -> None:
    if (
        m2._require_physical_fold_provenance is not _PHYSICAL_PROVENANCE
        or rademacher.RademacherFeatureLibrary is not _FEATURE_FACTORY
        or rademacher._require_issued_features is not _FEATURE_AUTHENTICATE
        or rademacher.fit_rademacher_fold is not _FIT_RADEMACHER_FOLD
        or rademacher.apply_rademacher_validation is not _APPLY_RADEMACHER_VALIDATION
        or m2.fold_lambda_maximum is not _CORE_LAMBDA_MAXIMUM
        or m2.fit_m2 is not _CORE_FIT
        or m2.evaluate_m2 is not _CORE_EVALUATE
        or executor._validate_optimizer_result is not _CORE_VALIDATE_FIT
        or executor._validate_lambda_maximum is not _CORE_VALIDATE_MAXIMUM
        or executor.penalty_from_configuration is not _CORE_PENALTY
        or optional.fold_low_rank_scales is not _OPTIONAL_SCALES
        or optional.fit_low_rank is not _OPTIONAL_FIT_LOW_RANK
        or optional._fit_sparse_plus_low_rank_impl is not _OPTIONAL_FIT_HYBRID
        or optional.derive_hybrid_low_rank_initializer_configuration
        is not _OPTIONAL_DERIVE_PRECURSOR
        or _RademacherExecutionValidator.__call__ is not _VALIDATOR_CALL
        or _RademacherExecutionValidator.validate_execution_detail is not _VALIDATOR_DETAIL
        or _run_production_numerical is not _PRODUCTION_NUMERICAL_RUNNER
        or _score_fold is not _PRODUCTION_SCORE_FOLD
        or _core_specification is not _PRODUCTION_CORE_SPECIFICATION
        or _low_rank_specification is not _PRODUCTION_LOW_RANK_SPECIFICATION
        or _hybrid_specification is not _PRODUCTION_HYBRID_SPECIFICATION
        or _issue_precursor_receipt is not _PRODUCTION_PRECURSOR_ISSUER
        or _observed_hook is not _OBSERVED_HOOK
        or _observed_request is not _OBSERVED_REQUEST
        or _request_observed_bind is not _REQUEST_OBSERVED_BIND
        or _request_observed_capability is not _REQUEST_OBSERVED_CAPABILITY
        or _request_observed_execute is not _REQUEST_OBSERVED_EXECUTE
        or _request_observed_ack is not _REQUEST_OBSERVED_ACK
    ):
        raise RademacherExecutionAuthorityError(
            "named Rademacher production functions were rebound"
        )


def _make_state(
    *,
    owner: executor.CoreTrialExecutor | None,
    cache: Any,
    cells: Sequence[m1.Cell],
    moments: Sequence[m1.UniformCellMoments],
    candidates: m1.CandidateLibrary,
    geometry: m1.GeometryEngine,
    pairs: m2.PairFeatureLibrary,
    mode: str,
    maximum_iterations: int,
    synthetic_feature_factory: Callable[[m1.CandidateLibrary, rademacher.RademacherSeedRoute], Any] | None = None,
    synthetic_runner: Callable[[RademacherAttemptCapability, Any, Any], tuple[tuple[RademacherFoldExecution, ...], controller.TrialResult]] | None = None,
) -> _AdapterState:
    rows = tuple(cells)
    uniform = tuple(moments)
    if not rows or len(rows) != len(uniform):
        raise RademacherExecutionAuthorityError("source cells and moments are not aligned")
    return _AdapterState(
        owner_reference=None if owner is None else _weak(owner, "source executor"),
        cache_reference=_weak(cache, "source cache"),
        candidates_reference=_weak(candidates, "source candidates"),
        geometry_reference=_weak(geometry, "source geometry"),
        pairs_reference=_weak(pairs, "source physical pair library"),
        source_cells=rows,
        source_moments=uniform,
        source_cells_digest=executor._deep_integrity_digest(rows),
        source_moments_digest=executor._deep_integrity_digest(uniform),
        mode=mode,
        maximum_iterations=maximum_iterations,
        lock=threading.RLock(),
        synthetic_feature_factory=synthetic_feature_factory,
        synthetic_runner=synthetic_runner,
        dictionaries={},
        result_records={},
    )


def _require_source_binding(state: _AdapterState, *, deep: bool = True) -> None:
    cache = state.cache_reference()
    candidates = state.candidates_reference()
    geometry = state.geometry_reference()
    pairs = state.pairs_reference()
    if cache is None or candidates is None or geometry is None or pairs is None:
        raise RademacherExecutionAuthorityError(
            "Rademacher source producer is no longer live"
        )
    if (
        executor._deep_integrity_digest(state.source_cells) != state.source_cells_digest
        or executor._deep_integrity_digest(state.source_moments)
        != state.source_moments_digest
        or geometry.library is not candidates
        or pairs.candidates is not candidates
    ):
        raise RademacherExecutionAuthorityError(
            "Rademacher source cells, moments, or libraries changed"
        )
    if state.mode == executor.SEALED_PRODUCTION_MODE:
        owner = None if state.owner_reference is None else state.owner_reference()
        if type(owner) is not executor.CoreTrialExecutor or owner.prepared is not cache:
            raise RademacherExecutionAuthorityError(
                "Rademacher source executor/cache binding is unavailable"
            )
        exact = executor._require_prepared_cache(cache)
        if (
            exact.mode != executor.SEALED_PRODUCTION_MODE
            or exact.expected_rank != EXPECTED_RANK
            or exact.execution_inventory is None
            or owner.maximum_iterations != executor.FROZEN_MAXIMUM_ITERATIONS
            or len(exact.folds) != EXPECTED_FOLDS
            or len(exact.cell_inventory) != EXPECTED_ROWS
            or any(
                current is not issued
                for current, issued in zip(_source_cells(exact), state.source_cells, strict=True)
            )
        ):
            raise RademacherExecutionAuthorityError(
                "sealed Rademacher source cache changed after issuance"
            )
        if deep:
            for fold in exact.folds:
                try:
                    provenance = _PHYSICAL_PROVENANCE(
                        fold.training,
                        state.source_cells,
                        state.source_moments,
                        candidates,
                        geometry,
                        pairs,
                        fold.spec.training_indexes,
                    )
                except m2.M2Error as error:
                    raise RademacherExecutionAuthorityError(
                        "physical fold lost exact live cell/moment provenance"
                    ) from error
                if provenance.physical_rank != EXPECTED_RANK:
                    raise RademacherExecutionAuthorityError(
                        "physical fold rank changed before Rademacher execution"
                    )
    elif (
        state.mode != executor.SYNTHETIC_TEST_MODE
        or state.owner_reference is not None
        or state.synthetic_feature_factory is None
        or state.synthetic_runner is None
    ):
        raise RademacherExecutionAuthorityError("Rademacher adapter mode is invalid")


def _require_adapter(
    adapter: RademacherReferenceAdapter,
    *,
    require_bound: bool = True,
    deep_cache: bool = True,
) -> _AdapterState:
    if type(adapter) is not RademacherReferenceAdapter or adapter._seal is not _ADAPTER_SEAL:
        raise RademacherExecutionAuthorityError(
            "Rademacher adapter lacks exact named-module authority"
        )
    state = _lookup_adapter(adapter)
    if type(state) is not _AdapterState:
        raise RademacherExecutionAuthorityError("Rademacher adapter state is malformed")
    _require_source_binding(state, deep=deep_cache)
    if require_bound and (
        state.session_reference is None
        or state.session_reference() is None
        or state.authority_reference is None
        or state.authority_reference() is None
    ):
        raise RademacherExecutionAuthorityError(
            "Rademacher adapter lacks its live session authority"
        )
    if require_bound:
        if state.mode == executor.SEALED_PRODUCTION_MODE:
            _require_observed_hooks(state.observed_hooks)
        elif state.observed_hooks is not None:
            raise RademacherExecutionAuthorityError(
                "synthetic Rademacher adapter carries production session hooks"
            )
    if state.dictionaries is None or state.result_records is None or state.validator is None:
        raise RademacherExecutionAuthorityError("Rademacher registries are unavailable")
    return state


def create_rademacher_reference_adapter(
    owner: executor.CoreTrialExecutor,
    cells: Sequence[m1.Cell],
    moments: Sequence[m1.UniformCellMoments],
) -> RademacherReferenceAdapter:
    """Bind the named production adapter to exact live source rows and moments."""
    _require_fixed_production_functions()
    if type(owner) is not executor.CoreTrialExecutor:
        raise RademacherExecutionAuthorityError("Rademacher source owner has wrong type")
    cache = executor._require_prepared_cache(owner.prepared)
    rows = tuple(cells)
    cached_rows = _source_cells(cache)
    if (
        cache.mode != executor.SEALED_PRODUCTION_MODE
        or owner.maximum_iterations != executor.FROZEN_MAXIMUM_ITERATIONS
        or len(cached_rows) != len(rows)
        or any(left is not right for left, right in zip(cached_rows, rows, strict=True))
    ):
        raise RademacherExecutionAuthorityError(
            "Rademacher production factory requires the exact sealed source cache rows"
        )
    adapter = RademacherReferenceAdapter(_seal=_ADAPTER_SEAL)
    state = _make_state(
        owner=owner,
        cache=cache,
        cells=rows,
        moments=moments,
        candidates=cache.candidates,
        geometry=cache.geometry,
        pairs=cache.pairs,
        mode=executor.SEALED_PRODUCTION_MODE,
        maximum_iterations=executor.FROZEN_MAXIMUM_ITERATIONS,
    )
    state.validator = _RademacherExecutionValidator(adapter)
    _register_adapter(adapter, state)
    _require_source_binding(state, deep=True)
    return adapter


def _create_synthetic_qualification_adapter(
    cache_authority: object,
    cells: Sequence[m1.Cell],
    moments: Sequence[m1.UniformCellMoments],
    candidates: m1.CandidateLibrary,
    geometry: m1.GeometryEngine,
    pairs: m2.PairFeatureLibrary,
    *,
    session: object,
    authority: object,
    feature_factory: Callable[[m1.CandidateLibrary, rademacher.RademacherSeedRoute], Any],
    numerical_runner: Callable[[RademacherAttemptCapability, Any, Any], tuple[tuple[RademacherFoldExecution, ...], controller.TrialResult]],
    maximum_iterations: int = 1,
) -> RademacherReferenceAdapter:
    """Unmistakably synthetic, target-free issuer for lightweight tests only."""
    if (
        not callable(feature_factory)
        or not callable(numerical_runner)
        or isinstance(maximum_iterations, bool)
        or not isinstance(maximum_iterations, int)
        or not 1 <= maximum_iterations <= executor.FROZEN_MAXIMUM_ITERATIONS
    ):
        raise RademacherExecutionAuthorityError("synthetic qualification inputs are invalid")
    adapter = RademacherReferenceAdapter(_seal=_ADAPTER_SEAL)
    state = _make_state(
        owner=None,
        cache=cache_authority,
        cells=cells,
        moments=moments,
        candidates=candidates,
        geometry=geometry,
        pairs=pairs,
        mode=executor.SYNTHETIC_TEST_MODE,
        maximum_iterations=maximum_iterations,
        synthetic_feature_factory=feature_factory,
        synthetic_runner=numerical_runner,
    )
    state.session_reference = _weak(session, "synthetic Rademacher session")
    state.authority_reference = _weak(authority, "synthetic Rademacher authority")
    state.validator = _RademacherExecutionValidator(adapter)
    _register_adapter(adapter, state)
    return adapter


def _bind_to_observed_session(
    adapter: RademacherReferenceAdapter,
    session: object,
    authority: object,
    owner: executor.CoreTrialExecutor,
) -> bool:
    """Bind once, only through the observed session's pristine-bind gate."""
    state = _require_adapter(adapter, require_bound=False)
    if state.mode != executor.SEALED_PRODUCTION_MODE:
        raise RademacherExecutionAuthorityError(
            "observed-session binding rejects synthetic Rademacher adapters"
        )
    callbacks = _REQUEST_OBSERVED_BIND(
        session,
        authority,
        adapter,
        owner,
        state.cache_reference(),
    )
    with state.lock:
        if (
            state.owner_reference is None
            or state.owner_reference() is not owner
            or owner.prepared is not state.cache_reference()
            or state.session_reference is not None
            or state.authority_reference is not None
            or state.observed_hooks is not None
            or state.issued_attempts != 0
            or state.dictionaries
            or state.result_records
        ):
            raise RademacherExecutionAuthorityError(
                "Rademacher adapter must bind to one pristine source session"
            )
        state.session_reference = _weak(session, "observed Rademacher session")
        state.authority_reference = _weak(authority, "observed Rademacher authority")
        state.observed_hooks = callbacks
    return True


def _require_trial(trial: controller.Trial) -> controller.Trial:
    if type(trial) is not controller.Trial:
        raise RademacherExecutionAuthorityError("Rademacher trial has wrong exact type")
    try:
        controller.Trial.__post_init__(trial)
    except (controller.ControllerError, TypeError, ValueError) as error:
        raise RademacherExecutionAuthorityError("Rademacher trial no longer validates") from error
    configuration = trial.matched_configuration
    if (
        trial.kind != "rademacher_noise"
        or trial.role != "reference"
        or trial.falsifier is not True
        or type(trial.seed) is not int
        or type(configuration) is not controller.Configuration
        or configuration.family not in ALL_MATCHED_FAMILIES
        or type(trial.coverage_binding) is not controller.CoverageBinding
    ):
        raise RademacherExecutionAuthorityError(
            "trial is not an exact capacity-matched Rademacher reference"
        )
    try:
        controller.Configuration.__post_init__(configuration)
    except (controller.ControllerError, TypeError, ValueError) as error:
        raise RademacherExecutionAuthorityError(
            "matched Rademacher configuration changed"
        ) from error
    return trial


def _history_matches_results(
    history: Sequence[executor.TrialExecution | executor.ExternalExecutionReceipt],
    results: Sequence[controller.TrialResult],
    evidence: controller.CoverageEvidence,
) -> bool:
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


def _seed_route(
    trial: controller.Trial,
    null_replicate_index: int | None,
    *,
    synthetic: bool,
) -> rademacher.RademacherSeedRoute:
    assert trial.seed is not None
    if null_replicate_index is None:
        return rademacher.RademacherSeedRoute(
            "target_free_qualification" if synthetic else "observed",
            int(trial.seed),
        )
    if (
        isinstance(null_replicate_index, bool)
        or not isinstance(null_replicate_index, int)
        or not 1 <= null_replicate_index <= len(controller.NULL_SEEDS)
    ):
        raise RademacherExecutionAuthorityError(
            "Rademacher null replicate index is outside 1..99"
        )
    replay = controller.new_null_replay(null_replicate_index)
    return rademacher.RademacherSeedRoute(
        rademacher.NULL_REPLAY_MODE,
        int(trial.seed),
        null_replicate_seed=int(replay.seed),
        valid_index=int(trial.valid_index),
    )


def _dictionary_record(
    state: _AdapterState,
    trial: controller.Trial,
    null_replicate_index: int | None,
) -> _DictionaryRecord:
    assert state.dictionaries is not None
    existing = state.dictionaries.get(trial.valid_index)
    if existing is not None:
        if (
            existing.trial_reference() is not trial
            or existing.trial_digest != executor._deep_integrity_digest(trial)
            or existing.null_replicate_index != null_replicate_index
            or existing.route_digest != executor._deep_integrity_digest(existing.route)
            or existing.features_digest != executor._deep_integrity_digest(existing.features)
        ):
            raise RademacherExecutionAuthorityError(
                "Rademacher retry differs from its frozen objective or dictionary"
            )
        if state.mode == executor.SEALED_PRODUCTION_MODE:
            _FEATURE_AUTHENTICATE(existing.features)
        return existing
    route = _seed_route(
        trial,
        null_replicate_index,
        synthetic=state.mode == executor.SYNTHETIC_TEST_MODE,
    )
    candidates = state.candidates_reference()
    if candidates is None:
        raise RademacherExecutionAuthorityError("Rademacher candidate producer died")
    if state.mode == executor.SEALED_PRODUCTION_MODE:
        features = _FEATURE_FACTORY(candidates, route)
        _FEATURE_AUTHENTICATE(features)
    else:
        assert state.synthetic_feature_factory is not None
        features = state.synthetic_feature_factory(candidates, route)
    try:
        _weak(features, "Rademacher dictionary")
    except RademacherExecutionAuthorityError:
        raise
    record = _DictionaryRecord(
        trial_reference=_weak(trial, "Rademacher trial"),
        trial_digest=executor._deep_integrity_digest(trial),
        null_replicate_index=null_replicate_index,
        route=route,
        route_digest=executor._deep_integrity_digest(route),
        features=features,
        features_digest=executor._deep_integrity_digest(features),
    )
    state.dictionaries[trial.valid_index] = record
    return record


def _capability_digest(record: _CapabilityRecord) -> bytes:
    adapter = record.adapter_reference()
    session = record.session_reference()
    authority = record.authority_reference()
    owner = None if record.owner_reference is None else record.owner_reference()
    cache = record.cache_reference()
    trial = record.trial_reference()
    coverage = record.coverage_reference()
    dictionary = record.dictionary_reference()
    route = record.route_reference()
    if any(value is None for value in (adapter, session, authority, cache, trial, coverage, dictionary, route)):
        raise RademacherExecutionAuthorityError(
            "Rademacher capability producer is no longer live"
        )
    if record.owner_reference is not None and owner is None:
        raise RademacherExecutionAuthorityError("Rademacher source owner died")
    return executor._authority_binding_digest(
        "rademacher_reference_v1",
        id(adapter),
        id(session),
        id(authority),
        0 if owner is None else id(owner),
        id(cache),
        id(trial),
        record.trial_digest,
        id(coverage),
        record.coverage_digest,
        tuple(id(reference()) for reference in record.history_references),
        record.history_digests,
        tuple(id(reference()) for reference in record.result_references),
        record.result_digests,
        id(dictionary),
        record.dictionary_digest,
        id(route),
        record.route_digest,
        record.null_replicate_index,
        record.attempt_ordinal,
        record.session_attempt_ordinal,
        record.session_version,
    )


def _issue_rademacher_attempt_capability(
    adapter: RademacherReferenceAdapter,
    session: object,
    authority: object,
    trial: controller.Trial,
    coverage_evidence: controller.CoverageEvidence | None,
    history: Sequence[executor.TrialExecution | executor.ExternalExecutionReceipt],
    results: Sequence[controller.TrialResult],
    *,
    null_replicate_index: int | None,
    session_attempt_ordinal: int,
    session_version: int,
) -> RademacherAttemptCapability:
    """Issue after the observed in-flight gate; freeze/reuse the dictionary."""
    state = _require_adapter(adapter)
    checked = _require_trial(trial)
    immutable_history = tuple(history)
    immutable_results = tuple(results)
    if (
        state.session_reference is None
        or state.session_reference() is not session
        or state.authority_reference is None
        or state.authority_reference() is not authority
        or isinstance(session_attempt_ordinal, bool)
        or not isinstance(session_attempt_ordinal, int)
        or session_attempt_ordinal <= state.last_session_attempt_ordinal
        or isinstance(session_version, bool)
        or not isinstance(session_version, int)
        or session_version < 0
    ):
        raise RademacherExecutionAuthorityError(
            "Rademacher capability session/authority/attempt binding is invalid"
        )
    if state.mode == executor.SEALED_PRODUCTION_MODE:
        _require_fixed_production_functions()
        if (
            type(coverage_evidence) is not controller.CoverageEvidence
            or checked.coverage_binding is not coverage_evidence.binding
            or len(immutable_history) != len(immutable_results)
            or not _history_matches_results(
                immutable_history, immutable_results, coverage_evidence
            )
        ):
            raise RademacherExecutionAuthorityError(
                "Rademacher coverage/history binding is not exact"
            )
        try:
            controller.validate_available_result_prefix(
                immutable_results, coverage_evidence
            )
        except (controller.ControllerError, TypeError, ValueError) as error:
            raise RademacherExecutionAuthorityError(
                "Rademacher history is not a canonical controller prefix"
            ) from error
        owner = None if state.owner_reference is None else state.owner_reference()
        if type(owner) is not executor.CoreTrialExecutor:
            raise RademacherExecutionAuthorityError("Rademacher source owner died")
        _REQUEST_OBSERVED_CAPABILITY(
            state.observed_hooks,
            session,
            authority,
            adapter,
            owner,
            state.cache_reference(),
            checked,
            coverage_evidence,
            null_replicate_index,
            session_attempt_ordinal,
            session_version,
        )
        coverage_object: Any = coverage_evidence
    else:
        if coverage_evidence is not None:
            coverage_object = coverage_evidence
        else:
            coverage_object = checked.coverage_binding
        if immutable_history or immutable_results:
            raise RademacherExecutionAuthorityError(
                "synthetic Rademacher qualification requires an empty controller history"
            )
    with state.lock:
        if session_attempt_ordinal <= state.last_session_attempt_ordinal:
            raise RademacherExecutionAuthorityError(
                "Rademacher session attempt chronology changed"
            )
        current = state.current_capability
        if current is not None:
            prior = _lookup_capability(current)
            if prior.phase not in {"completed_acked", "burned_acked"}:
                raise RademacherExecutionAuthorityError(
                    "a prior Rademacher capability is not ledger-terminal"
                )
        assert state.result_records is not None
        if checked.valid_index in state.result_records:
            raise RademacherExecutionAuthorityError(
                "a successful Rademacher objective cannot be committed twice"
            )
        dictionary = _dictionary_record(state, checked, null_replicate_index)
        capability = RademacherAttemptCapability(_seal=_CAPABILITY_SEAL)
        record = _CapabilityRecord(
            adapter_reference=_weak(adapter, "Rademacher adapter"),
            session_reference=_weak(session, "Rademacher session"),
            authority_reference=_weak(authority, "Rademacher authority"),
            owner_reference=state.owner_reference,
            cache_reference=state.cache_reference,
            trial_reference=_weak(checked, "Rademacher trial"),
            trial_digest=dictionary.trial_digest,
            coverage_reference=_weak(coverage_object, "Rademacher coverage"),
            coverage_digest=executor._deep_integrity_digest(coverage_object),
            history_references=tuple(_weak(entry, "Rademacher history entry") for entry in immutable_history),
            history_digests=tuple(executor._deep_integrity_digest(entry) for entry in immutable_history),
            result_references=tuple(_weak(result, "Rademacher history result") for result in immutable_results),
            result_digests=tuple(executor._deep_integrity_digest(result) for result in immutable_results),
            dictionary_reference=_weak(dictionary.features, "Rademacher dictionary"),
            dictionary_digest=dictionary.features_digest,
            route_reference=_weak(dictionary.route, "Rademacher seed route"),
            route_digest=dictionary.route_digest,
            null_replicate_index=null_replicate_index,
            attempt_ordinal=state.issued_attempts + 1,
            session_attempt_ordinal=session_attempt_ordinal,
            session_version=session_version,
            binding_digest=b"",
        )
        record.binding_digest = _capability_digest(record)
        _register_capability(capability, record)
        state.current_capability = capability
        state.issued_attempts += 1
        state.last_session_attempt_ordinal = session_attempt_ordinal
        return capability


def _require_capability(
    capability: RademacherAttemptCapability,
    *,
    require_current: bool,
) -> tuple[_CapabilityRecord, _AdapterState]:
    if (
        type(capability) is not RademacherAttemptCapability
        or capability._seal is not _CAPABILITY_SEAL
    ):
        raise RademacherExecutionAuthorityError(
            "Rademacher capability lacks exact module authority"
        )
    record = _lookup_capability(capability)
    if type(record) is not _CapabilityRecord:
        raise RademacherExecutionAuthorityError("Rademacher capability record is malformed")
    adapter = record.adapter_reference()
    if adapter is None:
        raise RademacherExecutionAuthorityError("Rademacher adapter producer is dead")
    state = _require_adapter(adapter)
    trial = record.trial_reference()
    coverage = record.coverage_reference()
    features = record.dictionary_reference()
    route = record.route_reference()
    history = tuple(reference() for reference in record.history_references)
    results = tuple(reference() for reference in record.result_references)
    if (
        trial is None
        or coverage is None
        or features is None
        or route is None
        or any(entry is None for entry in history)
        or any(result is None for result in results)
        or state.session_reference is None
        or state.session_reference() is not record.session_reference()
        or state.authority_reference is None
        or state.authority_reference() is not record.authority_reference()
        or state.cache_reference() is not record.cache_reference()
        or state.owner_reference is not record.owner_reference
        or executor._deep_integrity_digest(trial) != record.trial_digest
        or executor._deep_integrity_digest(coverage) != record.coverage_digest
        or tuple(executor._deep_integrity_digest(entry) for entry in history)
        != record.history_digests
        or tuple(executor._deep_integrity_digest(result) for result in results)
        != record.result_digests
        or executor._deep_integrity_digest(features) != record.dictionary_digest
        or executor._deep_integrity_digest(route) != record.route_digest
        or record.binding_digest != _capability_digest(record)
        or (require_current and state.current_capability is not capability)
    ):
        raise RademacherExecutionAuthorityError(
            "Rademacher capability live context changed"
        )
    if state.mode == executor.SEALED_PRODUCTION_MODE:
        if type(coverage) is not controller.CoverageEvidence:
            raise RademacherExecutionAuthorityError("Rademacher coverage producer changed")
        if not _history_matches_results(history, results, coverage):
            raise RademacherExecutionAuthorityError("Rademacher history/result binding changed")
        _FEATURE_AUTHENTICATE(features)
    return record, state


def _low_rank_coverage(
    record: _CapabilityRecord,
) -> tuple[controller.CoverageEvidence, tuple[controller.TrialResult, ...]]:
    evidence = record.coverage_reference()
    results = tuple(reference() for reference in record.result_references)
    if type(evidence) is not controller.CoverageEvidence or any(result is None for result in results):
        raise RademacherExecutionAuthorityError("hybrid Rademacher coverage is unavailable")
    start = controller.COVERAGE_TRIALS
    stop = start + 8
    if len(results) < stop:
        raise RademacherExecutionAuthorityError(
            "hybrid Rademacher execution precedes eight low-rank coverage scores"
        )
    anchors = tuple(results[start:stop])
    if any(type(result) is not controller.TrialResult for result in anchors):
        raise RademacherExecutionAuthorityError("low-rank coverage result type changed")
    target = record.trial_reference()
    assert target is not None and target.matched_configuration is not None
    try:
        _OPTIONAL_DERIVE_PRECURSOR(evidence, anchors, target.matched_configuration)
    except optional.LowRankError as error:
        raise RademacherExecutionAuthorityError(
            "hybrid Rademacher precursor configuration no longer derives"
        ) from error
    return evidence, anchors


def _precursor_digest(binding: _PrecursorBinding) -> bytes:
    return executor._authority_binding_digest(
        "rademacher_hybrid_precursor_v1",
        id(binding.capability_reference()),
        id(binding.trial_reference()),
        id(binding.coverage_reference()),
        binding.coverage_digest,
        tuple(id(reference()) for reference in binding.result_references),
        binding.result_digests,
        id(binding.feature_reference()),
        id(binding.rademacher_fold_reference()),
        id(binding.physical_fold_reference()),
        id(binding.target_configuration_reference()),
        id(binding.initializer_configuration_reference()),
        id(binding.precursor_specification_reference()),
        id(binding.hybrid_specification_reference()),
        id(binding.precursor_fit_reference()),
        binding.fold_ordinal,
    )


def _issue_precursor_receipt(
    capability: RademacherAttemptCapability,
    physical_fold: executor.PreparedFold,
    rademacher_fold: rademacher.RademacherFoldFit,
    evidence: controller.CoverageEvidence,
    anchor_results: tuple[controller.TrialResult, ...],
    target_configuration: controller.Configuration,
    initializer_configuration: controller.Configuration,
    precursor_specification: optional.LowRankSpec,
    hybrid_specification: optional.HybridSpec,
    precursor_fit: optional.LowRankOptimizerResult,
) -> _RademacherHybridPrecursorReceipt:
    record, state = _require_capability(capability, require_current=True)
    if record.phase != "consumed":
        raise RademacherExecutionAuthorityError(
            "Rademacher precursor can be issued only inside the consumed attempt"
        )
    if state.mode == executor.SEALED_PRODUCTION_MODE:
        registered_results = tuple(
            reference() for reference in record.result_references
        )
        registered_anchors = registered_results[
            controller.COVERAGE_TRIALS : controller.COVERAGE_TRIALS + 8
        ]
        if (
            record.coverage_reference() is not evidence
            or len(registered_anchors) != 8
            or any(
                supplied is not registered
                for supplied, registered in zip(
                    anchor_results,
                    registered_anchors,
                    strict=True,
                )
            )
        ):
            raise RademacherExecutionAuthorityError(
                "hybrid precursor is not bound to exact physical coverage"
            )
    receipt = _RademacherHybridPrecursorReceipt(_seal=_PRECURSOR_RECEIPT_SEAL)
    binding = _PrecursorBinding(
        capability_reference=_weak(capability, "Rademacher capability"),
        trial_reference=record.trial_reference,
        coverage_reference=_weak(evidence, "Rademacher coverage"),
        coverage_digest=executor._deep_integrity_digest(evidence),
        result_references=tuple(_weak(result, "low-rank coverage result") for result in anchor_results),
        result_digests=tuple(executor._deep_integrity_digest(result) for result in anchor_results),
        feature_reference=record.dictionary_reference,
        rademacher_fold_reference=_weak(rademacher_fold, "Rademacher fold"),
        physical_fold_reference=_weak(physical_fold, "physical fold"),
        target_configuration_reference=_weak(target_configuration, "hybrid target"),
        initializer_configuration_reference=_weak(initializer_configuration, "hybrid precursor configuration"),
        precursor_specification_reference=_weak(precursor_specification, "precursor specification"),
        hybrid_specification_reference=_weak(hybrid_specification, "hybrid specification"),
        precursor_fit_reference=_weak(precursor_fit, "Rademacher precursor fit"),
        fold_ordinal=physical_fold.spec.ordinal,
        binding_digest=b"",
    )
    object.__setattr__(binding, "binding_digest", _precursor_digest(binding))
    _register_precursor_receipt(receipt, binding)
    return receipt


def _require_precursor_receipt(
    receipt: object,
    fold_execution: RademacherFoldExecution,
    capability: RademacherAttemptCapability,
) -> _PrecursorBinding:
    if (
        type(receipt) is not _RademacherHybridPrecursorReceipt
        or getattr(receipt, "_seal", None) is not _PRECURSOR_RECEIPT_SEAL
    ):
        raise RademacherExecutionAuthorityError("hybrid precursor receipt is invalid")
    binding = _lookup_precursor_receipt(receipt)
    if type(binding) is not _PrecursorBinding:
        raise RademacherExecutionAuthorityError("hybrid precursor binding is malformed")
    record, state = _require_capability(capability, require_current=False)
    trial = record.trial_reference()
    evidence = binding.coverage_reference()
    results = tuple(reference() for reference in binding.result_references)
    if (
        binding.capability_reference() is not capability
        or binding.trial_reference() is not trial
        or evidence is None
        or len(results) != 8
        or any(result is None for result in results)
        or binding.feature_reference() is not fold_execution.rademacher_fit.features
        or binding.rademacher_fold_reference() is not fold_execution.rademacher_fit
        or binding.physical_fold_reference() is None
        or binding.target_configuration_reference() is not trial.matched_configuration
        or binding.initializer_configuration_reference() is not fold_execution.precursor_configuration
        or binding.precursor_specification_reference() is not fold_execution.precursor_specification
        or binding.hybrid_specification_reference() is not fold_execution.specification
        or binding.precursor_fit_reference() is not fold_execution.precursor_fit
        or binding.fold_ordinal != fold_execution.spec.ordinal
        or executor._deep_integrity_digest(evidence) != binding.coverage_digest
        or tuple(executor._deep_integrity_digest(result) for result in results)
        != binding.result_digests
        or _precursor_digest(binding) != binding.binding_digest
    ):
        raise RademacherExecutionAuthorityError(
            "hybrid precursor seed/fold/cache/configuration/fit binding changed"
        )
    if state.mode == executor.SEALED_PRODUCTION_MODE:
        registered = tuple(reference() for reference in record.result_references)[
            controller.COVERAGE_TRIALS : controller.COVERAGE_TRIALS + 8
        ]
        if (
            record.coverage_reference() is not evidence
            or len(registered) != 8
            or any(
                supplied is not current
                for supplied, current in zip(results, registered, strict=True)
            )
        ):
            raise RademacherExecutionAuthorityError(
                "hybrid precursor physical coverage binding changed"
            )
    assert trial is not None and trial.matched_configuration is not None
    derived = _OPTIONAL_DERIVE_PRECURSOR(
        evidence, results, trial.matched_configuration
    )
    if derived is not fold_execution.precursor_configuration:
        raise RademacherExecutionAuthorityError(
            "hybrid precursor configuration no longer follows the v3 derivation"
        )
    return binding


def _core_specification(
    fixed_m1: executor.FixedM1Offset,
    prepared: m2.PreparedM2,
    candidates: m1.CandidateLibrary,
    geometry: m1.GeometryEngine,
    features: rademacher.RademacherFeatureLibrary,
    configuration: controller.Configuration,
) -> executor.PenaltyMapping:
    maximum = _CORE_VALIDATE_MAXIMUM(
        _CORE_LAMBDA_MAXIMUM(
            fixed_m1.theta,
            prepared,
            candidates,
            geometry,
            features,
            configuration.family,
            l1_ratio=(float(configuration.alpha) if configuration.family == "elastic_net" else 1.0),
        )
    )
    return _CORE_PENALTY(configuration, maximum)


def _low_rank_specification(
    fixed_m1: executor.FixedM1Offset,
    prepared: m2.PreparedM2,
    candidates: m1.CandidateLibrary,
    geometry: m1.GeometryEngine,
    features: rademacher.RademacherFeatureLibrary,
    configuration: controller.Configuration,
) -> optional.LowRankSpec:
    if configuration.rank is None:
        raise RademacherProductionError("low-rank Rademacher configuration lacks rank")
    scales = _OPTIONAL_SCALES(
        fixed_m1.theta, prepared, candidates, geometry, features
    )
    return optional.LowRankSpec(
        int(configuration.rank),
        float(configuration.rho),
        float(scales.lambda_reference),
    )


def _hybrid_specification(
    fixed_m1: executor.FixedM1Offset,
    prepared: m2.PreparedM2,
    candidates: m1.CandidateLibrary,
    geometry: m1.GeometryEngine,
    features: rademacher.RademacherFeatureLibrary,
    configuration: controller.Configuration,
) -> optional.HybridSpec:
    if configuration.rank is None or configuration.sparse_rho is None:
        raise RademacherProductionError("hybrid Rademacher configuration is incomplete")
    scales = _OPTIONAL_SCALES(
        fixed_m1.theta, prepared, candidates, geometry, features
    )
    maximum = _CORE_VALIDATE_MAXIMUM(
        _CORE_LAMBDA_MAXIMUM(
            fixed_m1.theta,
            prepared,
            candidates,
            geometry,
            features,
            "lasso",
        )
    )
    try:
        edges = tuple(m2.PAIR_INDEX[index] for index in configuration.sparse_edges)
    except (IndexError, TypeError) as error:
        raise RademacherProductionError("hybrid Rademacher edge indexes are invalid") from error
    return optional.HybridSpec(
        int(configuration.rank),
        float(configuration.rho),
        float(configuration.sparse_rho),
        float(scales.lambda_reference),
        float(maximum.value),
        edges,
    )


def _score_fold(
    physical_fold: executor.PreparedFold,
    validation: m2.PreparedM2,
    beta: np.ndarray,
    features: rademacher.RademacherFeatureLibrary,
    candidates: m1.CandidateLibrary,
    geometry: m1.GeometryEngine,
) -> tuple[float, tuple[RademacherCellScore, ...]]:
    zero = np.zeros(m2.PAIR_DIMENSION, dtype=float)
    nested = _CORE_EVALUATE(
        zero,
        physical_fold.fixed_m1.theta,
        validation,
        candidates,
        geometry,
        features,
        need_predictions=True,
    )
    prediction = _CORE_EVALUATE(
        beta,
        physical_fold.fixed_m1.theta,
        validation,
        candidates,
        geometry,
        features,
        need_predictions=True,
    )
    if nested.log_probability_nats is None or prediction.log_probability_nats is None:
        raise RademacherProductionError("Rademacher held-out predictions were omitted")
    if not np.array_equal(
        nested.log_probability_nats, physical_fold.baseline_log_probability_nats
    ):
        raise RademacherProductionError("Rademacher beta zero is not exact same-fold M1")
    scores: list[RademacherCellScore] = []
    for local_index, (global_index, cell) in enumerate(
        zip(
            physical_fold.spec.validation_indexes,
            validation.cells,
            strict=True,
        )
    ):
        baseline = float(physical_fold.baseline_log_probability_nats[local_index])
        observed = float(prediction.log_probability_nats[local_index])
        if not math.isfinite(observed):
            raise RademacherProductionError("Rademacher held-out score is nonfinite")
        if cell.k in {0, m1.PANEL_SIZE} and observed != baseline:
            raise RademacherProductionError("K in {0,16} must have exact-zero noise gain")
        scores.append(
            RademacherCellScore(
                global_index=int(global_index),
                cell_key=str(cell.identifier),
                group=str(cell.group),
                k=int(cell.k),
                m1_log_probability_nats=baseline,
                m2_log_probability_nats=observed,
                gain_bits=(observed - baseline) / LN2,
            )
        )
    return math.fsum(score.gain_bits for score in scores) / len(scores), tuple(scores)


def _run_production_numerical(
    capability: RademacherAttemptCapability,
    record: _CapabilityRecord,
    state: _AdapterState,
) -> tuple[tuple[RademacherFoldExecution, ...], controller.TrialResult]:
    cache = record.cache_reference()
    trial = record.trial_reference()
    features = record.dictionary_reference()
    candidates = state.candidates_reference()
    geometry = state.geometry_reference()
    physical_pairs = state.pairs_reference()
    if (
        type(cache) is not executor.PreparedTrialCache
        or trial is None
        or type(features) is not rademacher.RademacherFeatureLibrary
        or candidates is None
        or geometry is None
        or physical_pairs is None
        or type(trial.matched_configuration) is not controller.Configuration
    ):
        raise RademacherExecutionAuthorityError("Rademacher numerical context is unavailable")
    configuration = trial.matched_configuration
    evidence: controller.CoverageEvidence | None = None
    anchors: tuple[controller.TrialResult, ...] = ()
    precursor_configuration: controller.Configuration | None = None
    if configuration.family == "sparse_plus_low_rank":
        evidence, anchors = _low_rank_coverage(record)
        precursor_configuration = _OPTIONAL_DERIVE_PRECURSOR(
            evidence, anchors, configuration
        )

    folds: list[RademacherFoldExecution] = []
    for physical_fold in cache.folds:
        rfit = _FIT_RADEMACHER_FOLD(
            state.source_cells,
            state.source_moments,
            candidates,
            geometry,
            features,
            physical_fold.training,
            physical_pairs,
            physical_fold.spec.training_indexes,
            expected_rank=EXPECTED_RANK,
        )
        validation_cells = tuple(
            state.source_cells[index] for index in physical_fold.spec.validation_indexes
        )
        validation_moments = tuple(
            state.source_moments[index] for index in physical_fold.spec.validation_indexes
        )
        validation = _APPLY_RADEMACHER_VALIDATION(
            rfit, validation_cells, validation_moments
        )
        precursor_specification = None
        precursor_fit = None
        precursor_receipt = None
        if configuration.family in controller.CORE_FAMILIES:
            specification: executor.PenaltyMapping | optional.LowRankSpec | optional.HybridSpec = _core_specification(
                physical_fold.fixed_m1,
                rfit.prepared,
                candidates,
                geometry,
                features,
                configuration,
            )
            fit: m2.M2OptimizerResult | optional.LowRankOptimizerResult | optional.HybridOptimizerResult = _CORE_VALIDATE_FIT(
                _CORE_FIT(
                    physical_fold.fixed_m1.theta,
                    rfit.prepared,
                    candidates,
                    geometry,
                    features,
                    specification.specification,
                    maximum_iterations=state.maximum_iterations,
                ),
                state.maximum_iterations,
            )
            if configuration.rho == 1.0 and np.count_nonzero(fit.beta) != 0:
                raise RademacherProductionError(
                    "rho=1 Rademacher core fit did not retain beta zero"
                )
        elif configuration.family == "low_rank":
            specification = _low_rank_specification(
                physical_fold.fixed_m1,
                rfit.prepared,
                candidates,
                geometry,
                features,
                configuration,
            )
            fit = _OPTIONAL_FIT_LOW_RANK(
                physical_fold.fixed_m1.theta,
                rfit.prepared,
                candidates,
                geometry,
                features,
                specification,
                maximum_iterations=state.maximum_iterations,
            )
        elif configuration.family == "sparse_plus_low_rank":
            assert evidence is not None and precursor_configuration is not None
            precursor_specification = _low_rank_specification(
                physical_fold.fixed_m1,
                rfit.prepared,
                candidates,
                geometry,
                features,
                precursor_configuration,
            )
            precursor_fit = _OPTIONAL_FIT_LOW_RANK(
                physical_fold.fixed_m1.theta,
                rfit.prepared,
                candidates,
                geometry,
                features,
                precursor_specification,
                maximum_iterations=state.maximum_iterations,
            )
            specification = _hybrid_specification(
                physical_fold.fixed_m1,
                rfit.prepared,
                candidates,
                geometry,
                features,
                configuration,
            )
            precursor_receipt = _issue_precursor_receipt(
                capability,
                physical_fold,
                rfit,
                evidence,
                anchors,
                configuration,
                precursor_configuration,
                precursor_specification,
                specification,
                precursor_fit,
            )
            # This private numerical body is used deliberately: the public
            # hybrid wrapper authorizes physical optional-session anchors,
            # whereas v4 requires a fresh same-Rademacher-objective precursor.
            fit = _OPTIONAL_FIT_HYBRID(
                physical_fold.fixed_m1.theta,
                rfit.prepared,
                candidates,
                geometry,
                features,
                specification,
                precursor_fit,
                linear_correction=None,
                production_receipt=precursor_receipt,
                maximum_sweeps=state.maximum_iterations,
            )
        else:  # pragma: no cover - _require_trial rejects this first
            raise RademacherProductionError("unsupported matched Rademacher family")

        gain, scores = _score_fold(
            physical_fold,
            validation,
            fit.beta,
            features,
            candidates,
            geometry,
        )
        folds.append(
            RademacherFoldExecution(
                spec=physical_fold.spec,
                fixed_m1=physical_fold.fixed_m1,
                rademacher_fit=rfit,
                validation=validation,
                specification=specification,
                precursor_configuration=(
                    precursor_configuration
                    if configuration.family == "sparse_plus_low_rank"
                    else None
                ),
                precursor_specification=precursor_specification,
                precursor_fit=precursor_fit,
                precursor_receipt=precursor_receipt,
                fit=fit,
                gain_bits_per_cell=float(gain),
                active_pair_degrees=int(np.count_nonzero(np.asarray(fit.beta, dtype=float))),
                cells=scores,
            )
        )
    result = controller.TrialResult(
        trial=trial,
        gain_bits_per_cell=math.fsum(fold.gain_bits_per_cell for fold in folds)
        / len(folds),
        active_pair_degrees=max(fold.active_pair_degrees for fold in folds),
    )
    return tuple(folds), result


_PRODUCTION_NUMERICAL_RUNNER = _run_production_numerical
_PRODUCTION_SCORE_FOLD = _score_fold
_PRODUCTION_CORE_SPECIFICATION = _core_specification
_PRODUCTION_LOW_RANK_SPECIFICATION = _low_rank_specification
_PRODUCTION_HYBRID_SPECIFICATION = _hybrid_specification
_PRODUCTION_PRECURSOR_ISSUER = _issue_precursor_receipt


def execute_rademacher_attempt(
    adapter: RademacherReferenceAdapter,
    capability: RademacherAttemptCapability,
) -> RademacherTrialExecution:
    """Consume before any solver entry and burn on every ``BaseException``."""
    record, state = _require_capability(capability, require_current=True)
    if record.adapter_reference() is not adapter:
        raise RademacherExecutionAuthorityError(
            "Rademacher capability belongs to another adapter"
        )
    if record.phase != "issued":
        raise RademacherExecutionAuthorityError("Rademacher capability was already consumed")
    if state.mode == executor.SEALED_PRODUCTION_MODE:
        _require_fixed_production_functions()
        _REQUEST_OBSERVED_EXECUTE(
            state.observed_hooks,
            record.session_reference(),
            record.authority_reference(),
            adapter,
            record.owner_reference(),
            record.cache_reference(),
            record.trial_reference(),
            record.coverage_reference(),
            record.null_replicate_index,
            capability,
            record.binding_digest,
        )
    with state.lock:
        if state.current_capability is not capability or record.phase != "issued":
            raise RademacherExecutionAuthorityError("Rademacher capability was already consumed")
        record.phase = "consumed"
    commit_key: int | None = None
    commit_value: tuple[weakref.ReferenceType[Any], weakref.ReferenceType[Any]] | None = None
    try:
        if state.mode == executor.SEALED_PRODUCTION_MODE:
            folds, result = _PRODUCTION_NUMERICAL_RUNNER(capability, record, state)
        else:
            assert state.synthetic_runner is not None
            folds, result = state.synthetic_runner(capability, record, state)
        trial = record.trial_reference()
        route = record.route_reference()
        features = record.dictionary_reference()
        if trial is None or route is None or features is None or result.trial is not trial:
            raise RademacherExecutionAuthorityError(
                "Rademacher numerical runner returned another objective"
            )
        execution = RademacherTrialExecution(
            trial=trial,
            matched_configuration=trial.matched_configuration,
            seed_route=route,
            features=features,
            folds=tuple(folds),
            result=result,
            mode=state.mode,
            session_identity=id(record.session_reference()),
            owner_identity=(0 if record.owner_reference is None else id(record.owner_reference())),
            cache_identity=id(record.cache_reference()),
            authority_identity=id(record.authority_reference()),
            attempt_ordinal=record.attempt_ordinal,
            session_attempt_ordinal=record.session_attempt_ordinal,
            null_replicate_index=record.null_replicate_index,
            selector_eligible=False,
        )
        execution_digest = executor._deep_integrity_digest(execution)
        binding = _ExecutionBinding(
            adapter_reference=record.adapter_reference,
            session_reference=record.session_reference,
            authority_reference=record.authority_reference,
            owner_reference=record.owner_reference,
            cache_reference=record.cache_reference,
            capability_reference=_weak(capability, "Rademacher capability"),
            capability_digest=record.binding_digest,
            execution_digest=execution_digest,
        )
        _register_execution(execution, binding)
        record.execution_reference = _weak(execution, "Rademacher execution")
        record.execution_digest = execution_digest
        _require_registered_execution(adapter, execution)
        with state.lock:
            assert state.result_records is not None
            if trial.valid_index in state.result_records:
                raise RademacherExecutionAuthorityError(
                    "Rademacher result would duplicate a prior commit"
                )
            commit_key = trial.valid_index
            commit_value = (
                _weak(result, "Rademacher result"),
                _weak(execution, "Rademacher execution"),
            )
            state.result_records[commit_key] = commit_value
            record.phase = "completed_unacked"
            record.completion = "success"
    except BaseException as error:
        with state.lock:
            if (
                commit_key is not None
                and commit_value is not None
                and state.result_records is not None
                and state.result_records.get(commit_key) is commit_value
            ):
                state.result_records.pop(commit_key, None)
            record.execution_reference = None
            record.execution_digest = None
            record.phase = "burned_unacked"
            record.completion = "error"
            record.error_type = f"{type(error).__module__}.{type(error).__qualname__}"
        raise
    return execution


def _validate_fold_detail(
    physical: executor.PreparedFold,
    observed: RademacherFoldExecution,
    configuration: controller.Configuration,
    features: rademacher.RademacherFeatureLibrary,
    capability: RademacherAttemptCapability,
    maximum_iterations: int,
) -> None:
    if (
        type(observed) is not RademacherFoldExecution
        or observed.spec is not physical.spec
        or observed.fixed_m1 is not physical.fixed_m1
        or observed.rademacher_fit.features is not features
        or observed.rademacher_fit.physical_prepared is not physical.training
        or observed.validation.transform is not observed.rademacher_fit.transform
        or len(observed.validation.cells) != len(physical.validation.cells)
        or any(
            current is not expected
            for current, expected in zip(
                observed.validation.cells,
                physical.validation.cells,
                strict=True,
            )
        )
    ):
        raise RademacherExecutionAuthorityError(
            "Rademacher fold/physical transform identity changed"
        )
    rademacher._require_issued_fold(observed.rademacher_fit)
    rademacher._require_issued_prepared(observed.rademacher_fit, observed.validation)
    fit = observed.fit
    if configuration.family in controller.CORE_FAMILIES:
        expected_ratio = (
            float(configuration.alpha)
            if configuration.family == "elastic_net"
            else 1.0
        )
        if (
            type(observed.specification) is not executor.PenaltyMapping
            or observed.specification.family != configuration.family
            or observed.specification.rho != float(configuration.rho)
            or observed.specification.penalty_kind != configuration.family
            or observed.specification.l1_ratio != expected_ratio
            or observed.specification.specification.kind != configuration.family
            or observed.specification.specification.l1_ratio != expected_ratio
            or observed.specification.specification.strength
            != float(configuration.rho)
            * float(observed.specification.lambda_maximum.value)
            or type(fit) is not m2.M2OptimizerResult
            or any(
                value is not None
                for value in (
                    observed.precursor_configuration,
                    observed.precursor_specification,
                    observed.precursor_fit,
                    observed.precursor_receipt,
                )
            )
        ):
            raise RademacherExecutionAuthorityError(
                "Rademacher core family configuration changed"
            )
        _CORE_VALIDATE_MAXIMUM(observed.specification.lambda_maximum)
        _CORE_VALIDATE_FIT(fit, maximum_iterations)
    elif configuration.family == "low_rank":
        if (
            type(observed.specification) is not optional.LowRankSpec
            or type(fit) is not optional.LowRankOptimizerResult
            or observed.specification.rank != configuration.rank
            or observed.specification.rho != float(configuration.rho)
            or fit.state.rank != configuration.rank
            or fit.telemetry.get("rank") != configuration.rank
            or fit.telemetry.get("rho") != float(configuration.rho)
            or fit.telemetry.get("lambda_reference")
            != float(observed.specification.lambda_reference)
            or any(
                value is not None
                for value in (
                    observed.precursor_configuration,
                    observed.precursor_specification,
                    observed.precursor_fit,
                    observed.precursor_receipt,
                )
            )
        ):
            raise RademacherExecutionAuthorityError(
                "Rademacher low-rank configuration changed"
            )
        binding = optional._require_registered_low_rank_result(fit, "production")
        state = _lookup_adapter(_lookup_capability(capability).adapter_reference())
        geometry = state.geometry_reference()
        if geometry is None:
            raise RademacherExecutionAuthorityError("Rademacher geometry producer died")
        optional._require_same_objective(
            binding.objective,
            physical.fixed_m1.theta,
            observed.rademacher_fit.prepared,
            features.candidates,
            geometry,
            features,
            optional._low_rank_specification_signature(observed.specification),
            None,
        )
    else:
        if (
            configuration.family != "sparse_plus_low_rank"
            or type(observed.specification) is not optional.HybridSpec
            or type(observed.precursor_configuration) is not controller.Configuration
            or type(observed.precursor_specification) is not optional.LowRankSpec
            or type(observed.precursor_fit) is not optional.LowRankOptimizerResult
            or type(fit) is not optional.HybridOptimizerResult
            or observed.precursor_receipt is None
            or observed.precursor_specification.rank
            != observed.precursor_configuration.rank
            or observed.precursor_specification.rho
            != float(observed.precursor_configuration.rho)
            or observed.precursor_specification.lambda_reference
            != observed.specification.lambda_reference
            or observed.specification.rank != configuration.rank
            or observed.specification.rho_low != float(configuration.rho)
            or observed.specification.rho_sparse
            != float(configuration.sparse_rho)
            or observed.specification.edge_indexes
            != tuple(configuration.sparse_edges)
            or observed.precursor_fit.state.rank
            != observed.precursor_configuration.rank
            or fit.telemetry.get("rank") != configuration.rank
            or fit.telemetry.get("rho_low") != float(configuration.rho)
            or fit.telemetry.get("rho_sparse")
            != float(configuration.sparse_rho)
            or fit.telemetry.get("lambda_reference")
            != float(observed.specification.lambda_reference)
            or fit.telemetry.get("core_lasso_lambda_maximum")
            != float(observed.specification.core_lasso_lambda_maximum)
            or fit.telemetry.get("named_edges")
            != observed.specification.named_edges
        ):
            raise RademacherExecutionAuthorityError(
                "Rademacher hybrid precursor/configuration record changed"
            )
        _require_precursor_receipt(observed.precursor_receipt, observed, capability)
        precursor_binding = optional._require_registered_low_rank_result(
            observed.precursor_fit, "production"
        )
        hybrid_binding = optional._require_registered_hybrid_result(fit, "production")
        if (
            hybrid_binding.selected_anchor_reference() is not observed.precursor_fit
            or hybrid_binding.production_receipt_reference is None
            or hybrid_binding.production_receipt_reference() is not observed.precursor_receipt
        ):
            raise RademacherExecutionAuthorityError(
                "Rademacher hybrid did not use its exact fresh precursor"
            )
        state = _lookup_adapter(_lookup_capability(capability).adapter_reference())
        geometry = state.geometry_reference()
        if geometry is None:
            raise RademacherExecutionAuthorityError("Rademacher geometry producer died")
        optional._require_same_objective(
            precursor_binding.objective,
            physical.fixed_m1.theta,
            observed.rademacher_fit.prepared,
            features.candidates,
            geometry,
            features,
            optional._low_rank_specification_signature(observed.precursor_specification),
            None,
        )
        optional._require_same_objective(
            hybrid_binding.objective,
            physical.fixed_m1.theta,
            observed.rademacher_fit.prepared,
            features.candidates,
            geometry,
            features,
            optional._hybrid_specification_signature(observed.specification),
            None,
        )
    expected = tuple(
        zip(
            physical.spec.validation_indexes,
            physical.validation.cells,
            physical.baseline_log_probability_nats,
            strict=True,
        )
    )
    if len(observed.cells) != len(expected):
        raise RademacherExecutionAuthorityError("Rademacher fold score inventory changed")
    for score, (global_index, cell, baseline) in zip(observed.cells, expected, strict=True):
        if (
            type(score) is not RademacherCellScore
            or score.global_index != global_index
            or score.cell_key != str(cell.identifier)
            or score.group != str(cell.group)
            or score.k != int(cell.k)
            or score.m1_log_probability_nats != float(baseline)
            or not math.isfinite(score.m2_log_probability_nats)
            or score.gain_bits
            != (score.m2_log_probability_nats - score.m1_log_probability_nats) / LN2
            or (
                score.k in {0, m1.PANEL_SIZE}
                and score.m2_log_probability_nats != score.m1_log_probability_nats
            )
        ):
            raise RademacherExecutionAuthorityError(
                "Rademacher held-out score or M1 baseline changed"
            )
    if (
        observed.gain_bits_per_cell
        != math.fsum(score.gain_bits for score in observed.cells) / len(observed.cells)
        or observed.active_pair_degrees
        != int(np.count_nonzero(np.asarray(fit.beta, dtype=float)))
    ):
        raise RademacherExecutionAuthorityError(
            "Rademacher fold summary differs from its retained scores and fit"
        )


def _require_registered_execution(
    adapter: RademacherReferenceAdapter,
    execution: RademacherTrialExecution,
) -> controller.TrialResult:
    if type(execution) is not RademacherTrialExecution:
        raise RademacherExecutionAuthorityError(
            "Rademacher history lacks exact detailed execution"
        )
    binding = _lookup_execution(execution)
    if type(binding) is not _ExecutionBinding:
        raise RademacherExecutionAuthorityError("Rademacher execution binding is malformed")
    state = _require_adapter(adapter)
    capability = binding.capability_reference()
    if capability is None:
        raise RademacherExecutionAuthorityError("Rademacher capability producer is dead")
    record, capability_state = _require_capability(capability, require_current=False)
    trial = record.trial_reference()
    route = record.route_reference()
    features = record.dictionary_reference()
    owner = None if record.owner_reference is None else record.owner_reference()
    cache = record.cache_reference()
    if (
        capability_state is not state
        or binding.adapter_reference() is not adapter
        or binding.session_reference() is not record.session_reference()
        or binding.authority_reference() is not record.authority_reference()
        or binding.owner_reference is not record.owner_reference
        or binding.cache_reference() is not cache
        or binding.capability_digest != record.binding_digest
        or binding.execution_digest != executor._deep_integrity_digest(execution)
        or record.execution_reference is None
        or record.execution_reference() is not execution
        or record.execution_digest != binding.execution_digest
        or record.phase not in {"consumed", "completed_unacked", "completed_acked"}
        or trial is None
        or route is None
        or features is None
        or execution.trial is not trial
        or execution.matched_configuration is not trial.matched_configuration
        or execution.seed_route is not route
        or execution.features is not features
        or execution.mode != state.mode
        or execution.session_identity != id(record.session_reference())
        or execution.owner_identity != (0 if owner is None else id(owner))
        or execution.cache_identity != id(cache)
        or execution.authority_identity != id(record.authority_reference())
        or execution.attempt_ordinal != record.attempt_ordinal
        or execution.session_attempt_ordinal != record.session_attempt_ordinal
        or execution.null_replicate_index != record.null_replicate_index
        or execution.selector_eligible is not False
        or execution.result.trial is not trial
        or not execution.result.valid
        or execution.result.coverage_binding is not None
    ):
        raise RademacherExecutionAuthorityError(
            "Rademacher session/cache/trial/dictionary execution binding changed"
        )
    _require_trial(trial)
    configuration = trial.matched_configuration
    assert configuration is not None
    expected_folds = EXPECTED_FOLDS if state.mode == executor.SEALED_PRODUCTION_MODE else len(execution.folds)
    expected_rows = EXPECTED_ROWS if state.mode == executor.SEALED_PRODUCTION_MODE else len(state.source_cells)
    if len(execution.folds) != expected_folds or expected_folds == 0:
        raise RademacherExecutionAuthorityError("Rademacher fold inventory changed")
    all_scores: list[RademacherCellScore] = []
    if state.mode == executor.SEALED_PRODUCTION_MODE:
        assert type(cache) is executor.PreparedTrialCache
        for physical, observed in zip(cache.folds, execution.folds, strict=True):
            _validate_fold_detail(
                physical,
                observed,
                configuration,
                features,
                capability,
                state.maximum_iterations,
            )
            all_scores.extend(observed.cells)
    else:
        for observed in execution.folds:
            if type(observed) is not RademacherFoldExecution:
                raise RademacherExecutionAuthorityError("synthetic fold detail type changed")
            all_scores.extend(observed.cells)
    if (
        len(all_scores) != expected_rows
        or {score.global_index for score in all_scores} != set(range(expected_rows))
        or len({score.cell_key for score in all_scores}) != expected_rows
        or any(type(score) is not RademacherCellScore for score in all_scores)
        or execution.result.gain_bits_per_cell
        != math.fsum(fold.gain_bits_per_cell for fold in execution.folds) / len(execution.folds)
        or execution.result.active_pair_degrees
        != max(fold.active_pair_degrees for fold in execution.folds)
    ):
        raise RademacherExecutionAuthorityError(
            "Rademacher execution lacks its exact unique score inventory"
        )
    return execution.result


def _completed_rademacher_attempt_binding(
    adapter: RademacherReferenceAdapter,
    session: object,
    authority: object,
    capability: RademacherAttemptCapability,
    expected_completion: str,
) -> tuple[int, bytes, str | None, RademacherTrialExecution | None]:
    record, _ = _require_capability(capability, require_current=True)
    if (
        expected_completion not in {"success", "error"}
        or record.adapter_reference() is not adapter
        or record.session_reference() is not session
        or record.authority_reference() is not authority
        or record.phase not in {"completed_unacked", "burned_unacked"}
        or record.completion != expected_completion
    ):
        raise RademacherExecutionAuthorityError(
            "Rademacher attempt is not terminal in this exact session"
        )
    detail = None if record.execution_reference is None else record.execution_reference()
    if expected_completion == "success":
        if (
            record.error_type is not None
            or detail is None
            or _require_registered_execution(adapter, detail) is not detail.result
        ):
            raise RademacherExecutionAuthorityError(
                "successful Rademacher attempt lost detailed execution"
            )
    elif detail is not None or type(record.error_type) is not str or not record.error_type:
        raise RademacherExecutionAuthorityError(
            "failed Rademacher attempt lost exact error classification"
        )
    return record.attempt_ordinal, record.binding_digest, record.error_type, detail


def _acknowledge_rademacher_attempt(
    adapter: RademacherReferenceAdapter,
    session: object,
    authority: object,
    capability: RademacherAttemptCapability,
    ledger_attempt: object,
) -> tuple[int, bytes]:
    """Bind terminal capability to the exact latest observed-ledger attempt."""
    record, state = _require_capability(capability, require_current=True)
    trial = record.trial_reference()
    expected_outcomes = {"valid"} if record.completion == "success" else {
        "invalid",
        "resource",
    }
    if (
        record.adapter_reference() is not adapter
        or record.session_reference() is not session
        or record.authority_reference() is not authority
        or record.phase not in {"completed_unacked", "burned_unacked"}
        or record.completion not in {"success", "error"}
        or trial is None
        or getattr(ledger_attempt, "ordinal", None) != record.session_attempt_ordinal
        or getattr(ledger_attempt, "trial", None) is not trial
        or getattr(ledger_attempt, "outcome", None) not in expected_outcomes
        or getattr(ledger_attempt, "error_type", None) != record.error_type
        or getattr(ledger_attempt, "execution_capability", None) is not capability
        or getattr(ledger_attempt, "execution_capability_digest", None)
        != record.binding_digest
        or getattr(ledger_attempt, "production_attempt_ordinal", None)
        != record.attempt_ordinal
    ):
        raise RademacherExecutionAuthorityError(
            "session ledger attempt differs from Rademacher capability"
        )
    if state.mode == executor.SEALED_PRODUCTION_MODE:
        _REQUEST_OBSERVED_ACK(
            state.observed_hooks,
            session,
            authority,
            adapter,
            record.owner_reference(),
            record.cache_reference(),
            trial,
            record.coverage_reference(),
            record.null_replicate_index,
            capability,
            record.binding_digest,
            record.attempt_ordinal,
            record.session_attempt_ordinal,
            record.session_version,
            record.completion,
            record.error_type,
            ledger_attempt,
        )
    with state.lock:
        if record.attempt_ordinal != state.acknowledged_attempts + 1:
            raise RademacherExecutionAuthorityError(
                "Rademacher acknowledgment chronology changed"
            )
        record.ledger_attempt_reference = _weak(ledger_attempt, "Rademacher ledger attempt")
        record.ledger_attempt_digest = executor._deep_integrity_digest(ledger_attempt)
        record.phase = (
            "completed_acked" if record.completion == "success" else "burned_acked"
        )
        state.acknowledged_attempts += 1
    return record.attempt_ordinal, record.binding_digest


def _require_acknowledged_rademacher_attempt(
    adapter: RademacherReferenceAdapter,
    session: object,
    authority: object,
    capability: RademacherAttemptCapability,
    ledger_attempt: object,
) -> tuple[int, bytes]:
    record, _ = _require_capability(capability, require_current=False)
    if (
        record.adapter_reference() is not adapter
        or record.session_reference() is not session
        or record.authority_reference() is not authority
        or record.phase not in {"completed_acked", "burned_acked"}
        or record.ledger_attempt_reference is None
        or record.ledger_attempt_reference() is not ledger_attempt
        or record.ledger_attempt_digest != executor._deep_integrity_digest(ledger_attempt)
    ):
        raise RademacherExecutionAuthorityError(
            "Rademacher attempt ledger acknowledgment changed"
        )
    return record.attempt_ordinal, record.binding_digest


def _require_rademacher_attempt_count(
    adapter: RademacherReferenceAdapter,
    session: object,
    authority: object,
    acknowledged_attempts: int,
    *,
    allow_pending: bool,
) -> None:
    state = _require_adapter(adapter)
    if (
        state.session_reference is None
        or state.session_reference() is not session
        or state.authority_reference is None
        or state.authority_reference() is not authority
        or state.acknowledged_attempts != acknowledged_attempts
    ):
        raise RademacherExecutionAuthorityError(
            "Rademacher adapter/session attempt count changed"
        )
    current = state.current_capability
    if current is None:
        return
    record = _lookup_capability(current)
    allowed = {"completed_acked", "burned_acked"}
    if allow_pending:
        allowed.update({"issued", "consumed", "completed_unacked", "burned_unacked"})
    if record.phase not in allowed or record.binding_digest != _capability_digest(record):
        raise RademacherExecutionAuthorityError(
            "Rademacher adapter has an invalid pending attempt"
        )


def _adapter_producer_and_validator(
    adapter: RademacherReferenceAdapter,
) -> tuple[RademacherReferenceAdapter, _RademacherExecutionValidator]:
    # The observed-session owner obtains this pair immediately before the
    # adapter's one allowed pristine production bind request.
    state = _require_adapter(adapter, require_bound=False)
    assert state.validator is not None
    return adapter, state.validator


def _adapter_owner_and_cache(
    adapter: RademacherReferenceAdapter,
) -> tuple[executor.CoreTrialExecutor, executor.PreparedTrialCache]:
    """Expose exact identities for the later sealed-receipt owner check."""
    state = _require_adapter(adapter)
    owner = None if state.owner_reference is None else state.owner_reference()
    cache = state.cache_reference()
    if (
        state.mode != executor.SEALED_PRODUCTION_MODE
        or type(owner) is not executor.CoreTrialExecutor
        or type(cache) is not executor.PreparedTrialCache
        or owner.prepared is not cache
    ):
        raise RademacherExecutionAuthorityError(
            "Rademacher adapter lacks its exact production owner/cache"
        )
    return owner, cache


def _execution_from_receipt(
    adapter: RademacherReferenceAdapter,
    receipt: executor.ExternalExecutionReceipt,
) -> RademacherTrialExecution:
    if (
        type(receipt) is not executor.ExternalExecutionReceipt
        or type(receipt.execution_detail) is not RademacherTrialExecution
        or receipt.execution_detail.result is not receipt.result
    ):
        raise RademacherExecutionAuthorityError(
            "Rademacher session receipt lacks immutable execution detail"
        )
    _require_registered_execution(adapter, receipt.execution_detail)
    return receipt.execution_detail


def _require_registered_result(
    adapter: RademacherReferenceAdapter,
    result: controller.TrialResult,
) -> RademacherTrialExecution:
    state = _require_adapter(adapter)
    assert state.result_records is not None
    record = state.result_records.get(result.trial.valid_index)
    if record is None or record[0]() is not result:
        raise RademacherExecutionAuthorityError(
            "Rademacher result was not produced by this adapter"
        )
    execution = record[1]()
    if execution is None or execution.result is not result:
        raise RademacherExecutionAuthorityError(
            "Rademacher detailed execution is unavailable"
        )
    _require_registered_execution(adapter, execution)
    return execution


def _is_rademacher_attempt_capability(value: object) -> bool:
    return type(value) is RademacherAttemptCapability and getattr(
        value, "_seal", None
    ) is _CAPABILITY_SEAL


__all__ = (
    "RademacherAttemptCapability",
    "RademacherCellScore",
    "RademacherExecutionAuthorityError",
    "RademacherFoldExecution",
    "RademacherProductionError",
    "RademacherReferenceAdapter",
    "RademacherTrialExecution",
    "create_rademacher_reference_adapter",
    "execute_rademacher_attempt",
)
