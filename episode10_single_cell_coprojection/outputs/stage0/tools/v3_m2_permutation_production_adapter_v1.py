#!/usr/bin/env python3
"""Named production authority for EP10 whole-set-permutation trials.

This module is the narrow production layer above the frozen mapping primitive
and the fresh 66-path/12-refit permutation-M1 implementation.  It owns one
mapping per exact controller objective, process-local one-shot attempt
capabilities, the fresh permuted M2 cache, and the complete immutable 12-fold,
194-score execution record.

The observed-session owner is deliberately not modified here.  Private
handshake functions expose exact objects for that later single-owner
integration; no generic numerical callback, checkpoint loader, M0 path, or
full-development M1 fit is accepted by this module.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import weakref
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCRIPT_PATH = Path(os.path.abspath(__file__))
TOOLS_DIR = SCRIPT_PATH.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m1 as m1  # noqa: E402
import fit_v3_m2 as m2  # noqa: E402
import v3_m2_permutation_adapter_v1 as mapping  # noqa: E402
import v3_m2_permutation_m1_v1 as permutation_m1  # noqa: E402
import v3_m2_trial_executor_v1 as executor  # noqa: E402
import v3_search_controller_v1 as controller  # noqa: E402


EXPECTED_ROWS = mapping.EXPECTED_ROWS
EXPECTED_FOLDS = executor.EXPECTED_FOLDS
LN2 = math.log(2.0)

_DEEP_INTEGRITY_DIGEST = executor._deep_integrity_digest
_AUTHORITY_BINDING_DIGEST = executor._authority_binding_digest

_ADAPTER_SEAL = object()
_CAPABILITY_SEAL = object()


class PermutationProductionError(m2.M2Error):
    """The named permutation adapter rejected numerical execution detail."""


class PermutationExecutionAuthorityError(PermutationProductionError):
    """A live session/cache/trial/mapping/attempt authority failed closed."""


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
            raise PermutationExecutionAuthorityError(
                f"{label} does not support live identity binding"
            ) from error
        with lock:
            current = records.get(identity)
            if current is not None and current[0]() is not instance:
                raise PermutationExecutionAuthorityError(f"{label} identity collision")
            records[identity] = (reference, metadata)

    def lookup(instance: Any) -> Any:
        with lock:
            current = records.get(id(instance))
            if current is None or current[0]() is not instance:
                raise PermutationExecutionAuthorityError(
                    f"{label} lacks named-adapter issuance"
                )
            return current[1]

    return register, lookup


_register_adapter, _lookup_adapter = _make_weak_identity_registry(
    "whole-set-permutation adapter"
)
_register_capability, _lookup_capability = _make_weak_identity_registry(
    "permutation attempt capability"
)
_register_execution, _lookup_execution = _make_weak_identity_registry(
    "permutation trial execution"
)


class WholeSetPermutationAdapter:
    """Opaque named adapter bound to one exact source cache and session."""

    __slots__ = ("_seal", "__weakref__")

    def __init__(self, *, _seal: object | None = None) -> None:
        if _seal is not _ADAPTER_SEAL:
            raise PermutationExecutionAuthorityError(
                "whole-set-permutation adapters require named factory issuance"
            )
        object.__setattr__(self, "_seal", _seal)

    def __copy__(self) -> None:
        raise PermutationExecutionAuthorityError("permutation adapters cannot be copied")

    def __deepcopy__(self, memo: Any) -> None:
        del memo
        raise PermutationExecutionAuthorityError("permutation adapters cannot be copied")

    def __reduce__(self) -> None:
        raise PermutationExecutionAuthorityError(
            "permutation adapters cannot be serialized"
        )


class PermutationAttemptCapability:
    """Opaque one-shot authority for one exact permutation trial attempt."""

    __slots__ = ("_seal", "__weakref__")

    def __init__(self, *, _seal: object | None = None) -> None:
        if _seal is not _CAPABILITY_SEAL:
            raise PermutationExecutionAuthorityError(
                "permutation capabilities require named session issuance"
            )
        object.__setattr__(self, "_seal", _seal)

    def __copy__(self) -> None:
        raise PermutationExecutionAuthorityError(
            "permutation capabilities cannot be copied"
        )

    def __deepcopy__(self, memo: Any) -> None:
        del memo
        raise PermutationExecutionAuthorityError(
            "permutation capabilities cannot be copied"
        )

    def __reduce__(self) -> None:
        raise PermutationExecutionAuthorityError(
            "permutation capabilities cannot be serialized"
        )


@dataclass(frozen=True)
class PermutationCellScore:
    """One unique held-out permuted-M2 score against its own permuted M1."""

    global_index: int
    cell_key: str
    group: str
    k: int
    m1_log_probability_nats: float
    m2_log_probability_nats: float
    gain_bits: float


@dataclass(frozen=True)
class PermutationFoldExecution:
    """One configured physical-M2 fit and its held-out cell scores."""

    spec: executor.FoldSpec
    fixed_m1: executor.FixedM1Offset
    transform: m2.ResidualTransform
    penalty: executor.PenaltyMapping
    fit: m2.M2OptimizerResult
    gain_bits_per_cell: float
    active_pair_degrees: int
    cells: tuple[PermutationCellScore, ...]


@dataclass(frozen=True)
class PermutationTrialExecution:
    """Complete immutable detail retained for one selector-ineligible trial."""

    trial: controller.Trial
    configuration: controller.Configuration
    assignment: mapping.PermutationAssignment
    m1_execution: permutation_m1.PermutationM1Execution
    prepared_cache: executor.PreparedTrialCache
    folds: tuple[PermutationFoldExecution, ...]
    result: controller.TrialResult
    recipient_to_donor: tuple[int, ...]
    m1_fit_identities: tuple[tuple[str, str, float], ...]
    mode: str
    session_identity: int
    owner_identity: int
    source_cache_identity: int
    authority_identity: int
    attempt_ordinal: int
    session_attempt_ordinal: int
    null_replicate_index: int | None
    selector_eligible: bool


@dataclass
class _MappingRecord:
    trial_reference: weakref.ReferenceType[controller.Trial]
    trial_digest: bytes
    null_replicate_index: int | None
    assignment: mapping.PermutationAssignment
    assignment_digest: bytes


@dataclass
class _AdapterState:
    owner_reference: weakref.ReferenceType[executor.CoreTrialExecutor] | None
    source_cache_reference: weakref.ReferenceType[Any]
    candidates_reference: weakref.ReferenceType[m1.CandidateLibrary]
    geometry_reference: weakref.ReferenceType[m1.GeometryEngine]
    source_cells: tuple[m1.Cell, ...]
    source_cells_digest: bytes
    candidate_anchors: tuple[Any, ...]
    geometry_anchor: Any
    mode: str
    maximum_iterations: int
    expected_rank: int | None
    lock: threading.RLock
    session_reference: weakref.ReferenceType[Any] | None = None
    authority_reference: weakref.ReferenceType[Any] | None = None
    observed_hooks: tuple[Any, Any, Any, Any] | None = None
    mappings: dict[int, _MappingRecord] | None = None
    current_capability: PermutationAttemptCapability | None = None
    issued_attempts: int = 0
    acknowledged_attempts: int = 0
    last_session_attempt_ordinal: int = 0
    result_records: dict[
        int,
        tuple[
            weakref.ReferenceType[controller.TrialResult],
            weakref.ReferenceType[PermutationTrialExecution],
        ],
    ] | None = None
    validator: _PermutationExecutionValidator | None = None


@dataclass
class _CapabilityRecord:
    adapter_reference: weakref.ReferenceType[WholeSetPermutationAdapter]
    session_reference: weakref.ReferenceType[Any]
    authority_reference: weakref.ReferenceType[Any]
    owner_reference: weakref.ReferenceType[executor.CoreTrialExecutor] | None
    source_cache_reference: weakref.ReferenceType[Any]
    trial_reference: weakref.ReferenceType[controller.Trial]
    trial_digest: bytes
    assignment_reference: weakref.ReferenceType[mapping.PermutationAssignment]
    assignment_digest: bytes
    null_replicate_index: int | None
    attempt_ordinal: int
    session_attempt_ordinal: int
    session_version: int
    binding_digest: bytes
    phase: str = "issued"
    completion: str | None = None
    error_type: str | None = None
    m1_execution_reference: weakref.ReferenceType[Any] | None = None
    prepared_cache_reference: weakref.ReferenceType[Any] | None = None
    execution_reference: weakref.ReferenceType[PermutationTrialExecution] | None = None
    execution_digest: bytes | None = None
    ledger_attempt_reference: weakref.ReferenceType[Any] | None = None
    ledger_attempt_digest: bytes | None = None


@dataclass(frozen=True)
class _ExecutionBinding:
    adapter_reference: weakref.ReferenceType[WholeSetPermutationAdapter]
    session_reference: weakref.ReferenceType[Any]
    authority_reference: weakref.ReferenceType[Any]
    owner_reference: weakref.ReferenceType[executor.CoreTrialExecutor] | None
    source_cache_reference: weakref.ReferenceType[Any]
    capability_reference: weakref.ReferenceType[PermutationAttemptCapability]
    capability_digest: bytes
    execution_digest: bytes


class _PermutationExecutionValidator:
    """Exact named validator retained for later executor/session integration."""

    __slots__ = ("adapter_reference", "__weakref__")

    def __init__(self, adapter: WholeSetPermutationAdapter) -> None:
        self.adapter_reference = weakref.ref(adapter)

    def __call__(
        self,
        producer: WholeSetPermutationAdapter,
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
        except (PermutationProductionError, executor.CoreTrialExecutorError):
            return False

    def validate_execution_detail(
        self,
        producer: WholeSetPermutationAdapter,
        result: controller.TrialResult,
        execution_detail: object | None,
    ) -> bool:
        adapter = self.adapter_reference()
        if (
            adapter is None
            or producer is not adapter
            or type(execution_detail) is not PermutationTrialExecution
            or execution_detail.result is not result
        ):
            return False
        try:
            return _require_registered_execution(adapter, execution_detail) is result
        except (PermutationProductionError, executor.CoreTrialExecutorError):
            return False


_VALIDATOR_CALL = _PermutationExecutionValidator.__call__
_VALIDATOR_DETAIL = _PermutationExecutionValidator.validate_execution_detail


_OBSERVED_BIND_HOOK = "_permutation_adapter_bind_request"
_OBSERVED_CAPABILITY_HOOK = "_permutation_adapter_capability_request"
_OBSERVED_EXECUTE_HOOK = "_permutation_adapter_execute_request"
_OBSERVED_ACK_HOOK = "_permutation_adapter_ack_request"
_OBSERVED_HOOK_NAMES = (
    _OBSERVED_BIND_HOOK,
    _OBSERVED_CAPABILITY_HOOK,
    _OBSERVED_EXECUTE_HOOK,
    _OBSERVED_ACK_HOOK,
)


def _observed_hook(name: str) -> Any:
    """Resolve one exact observed-session hook without invoking it."""
    if name not in _OBSERVED_HOOK_NAMES:
        raise PermutationExecutionAuthorityError(
            "unknown observed-session permutation hook"
        )
    module_name = "v3_m2_observed_session_v1"
    module = sys.modules.get(module_name)
    expected = TOOLS_DIR / f"{module_name}.py"
    module_path = getattr(module, "SCRIPT_PATH", None)
    module_file = getattr(module, "__file__", None)
    file_path = (
        None
        if module_file is None
        else Path(os.path.abspath(os.fspath(module_file)))
    )
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
        raise PermutationExecutionAuthorityError(
            "production permutation execution requires the exact observed session"
        )
    callback = getattr(module, name, None)
    code = getattr(callback, "__code__", None)
    if (
        not callable(callback)
        or getattr(callback, "__module__", None) != module_name
        or code is None
        or Path(os.path.abspath(code.co_filename)) != expected
    ):
        raise PermutationExecutionAuthorityError(
            f"observed-session permutation hook {name!r} is unavailable"
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
        raise PermutationExecutionAuthorityError(
            "observed-session permutation hook resolver was rebound"
        )
    callback = _OBSERVED_HOOK(name)
    if expected_callback is not None and callback is not expected_callback:
        raise PermutationExecutionAuthorityError(
            f"observed-session permutation hook {name!r} changed after binding"
        )
    try:
        return callback(*arguments)
    except PermutationExecutionAuthorityError:
        raise
    except Exception as error:
        raise PermutationExecutionAuthorityError(
            f"observed-session permutation hook {name!r} rejected its binding"
        ) from error


_OBSERVED_REQUEST = _observed_request


def _require_observed_dispatch() -> None:
    if (
        _observed_hook is not _OBSERVED_HOOK
        or _observed_request is not _OBSERVED_REQUEST
    ):
        raise PermutationExecutionAuthorityError(
            "observed-session permutation request dispatch was rebound"
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
        raise PermutationExecutionAuthorityError(
            "permutation adapter lacks its exact observed-session hooks"
        )
    for name, expected in zip(_OBSERVED_HOOK_NAMES, callbacks, strict=True):
        if _OBSERVED_HOOK(name) is not expected:
            raise PermutationExecutionAuthorityError(
                f"observed-session permutation hook {name!r} changed after binding"
            )
    return callbacks


def _request_observed_bind(
    session: object,
    authority: object,
    adapter: WholeSetPermutationAdapter,
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
        raise PermutationExecutionAuthorityError(
            "observed session rejected permutation adapter binding"
        )
    return callbacks


def _request_observed_capability(
    callbacks: tuple[Any, Any, Any, Any] | None,
    session: object,
    authority: object,
    adapter: WholeSetPermutationAdapter,
    owner: executor.CoreTrialExecutor,
    source_cache: executor.PreparedTrialCache,
    trial: controller.Trial,
    coverage_binding: controller.CoverageBinding,
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
        coverage_binding,
        null_replicate_index,
        session_attempt_ordinal,
        session_version,
        expected_callback=exact[1],
    )
    if accepted is not True:
        raise PermutationExecutionAuthorityError(
            "observed session rejected permutation capability issuance"
        )


def _request_observed_execute(
    callbacks: tuple[Any, Any, Any, Any] | None,
    session: object,
    authority: object,
    adapter: WholeSetPermutationAdapter,
    owner: executor.CoreTrialExecutor,
    source_cache: executor.PreparedTrialCache,
    trial: controller.Trial,
    coverage_binding: controller.CoverageBinding,
    null_replicate_index: int | None,
    capability: PermutationAttemptCapability,
    binding_digest: bytes,
    session_attempt_ordinal: int,
    session_version: int,
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
        coverage_binding,
        null_replicate_index,
        capability,
        binding_digest,
        session_attempt_ordinal,
        session_version,
        expected_callback=exact[2],
    )
    if accepted is not True:
        raise PermutationExecutionAuthorityError(
            "observed session rejected permutation capability use"
        )


def _request_observed_ack(
    callbacks: tuple[Any, Any, Any, Any] | None,
    session: object,
    authority: object,
    adapter: WholeSetPermutationAdapter,
    owner: executor.CoreTrialExecutor,
    source_cache: executor.PreparedTrialCache,
    trial: controller.Trial,
    coverage_binding: controller.CoverageBinding,
    null_replicate_index: int | None,
    capability: PermutationAttemptCapability,
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
        coverage_binding,
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
        raise PermutationExecutionAuthorityError(
            "observed session rejected permutation ledger acknowledgment"
        )


_REQUEST_OBSERVED_BIND = _request_observed_bind
_REQUEST_OBSERVED_CAPABILITY = _request_observed_capability
_REQUEST_OBSERVED_EXECUTE = _request_observed_execute
_REQUEST_OBSERVED_ACK = _request_observed_ack


def _weak(value: Any, label: str) -> weakref.ReferenceType[Any]:
    try:
        return weakref.ref(value)
    except TypeError as error:
        raise PermutationExecutionAuthorityError(
            f"{label} does not support process-local authority"
        ) from error


def _source_cells(cache: executor.PreparedTrialCache) -> tuple[m1.Cell, ...]:
    rows: list[m1.Cell | None] = [None] * len(cache.cell_inventory)
    for fold in cache.folds:
        for index, cell in zip(
            fold.spec.validation_indexes,
            fold.validation.cells,
            strict=True,
        ):
            if not 0 <= index < len(rows) or rows[index] is not None:
                raise PermutationExecutionAuthorityError(
                    "source cache does not have one unique validation owner per cell"
                )
            rows[index] = cell
    if any(cell is None for cell in rows):
        raise PermutationExecutionAuthorityError(
            "source cache does not cover its complete cell inventory"
        )
    exact = tuple(cell for cell in rows if cell is not None)
    if tuple((cell.identifier, cell.group) for cell in exact) != cache.cell_inventory:
        raise PermutationExecutionAuthorityError(
            "source cache cell order differs from its execution inventory"
        )
    return exact


def _same_array_values_exact(left: object, right: object) -> bool:
    observed = np.asarray(left)
    expected = np.asarray(right)
    return (
        observed.dtype == expected.dtype
        and observed.shape == expected.shape
        and np.ascontiguousarray(observed).tobytes(order="C")
        == np.ascontiguousarray(expected).tobytes(order="C")
    )


def _same_scalar_tree_exact(left: object, right: object) -> bool:
    """Compare scalar/dataclass metadata without bool/int or signed-zero aliases."""
    if type(left) is not type(right):
        return False
    if left is None:
        return True
    if type(left) is float:
        return left.hex() == right.hex()  # type: ignore[union-attr]
    if type(left) in {bool, int, str, bytes}:
        return left == right
    if type(left) is tuple:
        return len(left) == len(right) and all(  # type: ignore[arg-type]
            _same_scalar_tree_exact(observed, expected)
            for observed, expected in zip(left, right, strict=True)  # type: ignore[arg-type]
        )
    if is_dataclass(left) and not isinstance(left, type):
        return all(
            _same_scalar_tree_exact(
                getattr(left, descriptor.name), getattr(right, descriptor.name)
            )
            for descriptor in fields(left)
        )
    return left is right


def _make_state(
    *,
    owner: executor.CoreTrialExecutor | None,
    source_cache: Any,
    cells: Sequence[m1.Cell],
    candidates: m1.CandidateLibrary,
    geometry: m1.GeometryEngine,
    mode: str,
    maximum_iterations: int,
    expected_rank: int | None,
) -> _AdapterState:
    _require_fixed_mapping_functions()
    rows, _, _, _ = _MAPPING_CANONICAL_CELLS(cells)
    permutation_m1._validate_exact_library_methods(candidates, geometry)
    ks = tuple(sorted({cell.k for cell in rows}))
    return _AdapterState(
        owner_reference=None if owner is None else _weak(owner, "source executor"),
        source_cache_reference=_weak(source_cache, "source cache authority"),
        candidates_reference=_weak(candidates, "source candidates"),
        geometry_reference=_weak(geometry, "source geometry"),
        source_cells=rows,
        source_cells_digest=_DEEP_INTEGRITY_DIGEST(rows),
        candidate_anchors=permutation_m1._candidate_basis_anchors(candidates, ks),
        geometry_anchor=permutation_m1._geometry_anchor(geometry, ks),
        mode=mode,
        maximum_iterations=maximum_iterations,
        expected_rank=expected_rank,
        lock=threading.RLock(),
        mappings={},
        result_records={},
    )


def create_whole_set_permutation_adapter(
    owner: executor.CoreTrialExecutor,
) -> WholeSetPermutationAdapter:
    """Create the named adapter for one exact sealed-production source cache."""
    if type(owner) is not executor.CoreTrialExecutor:
        raise PermutationExecutionAuthorityError(
            "permutation adapter source owner has the wrong exact type"
        )
    cache = executor._require_prepared_cache(owner.prepared)
    if (
        cache.mode != executor.SEALED_PRODUCTION_MODE
        or owner.maximum_iterations != executor.FROZEN_MAXIMUM_ITERATIONS
        or cache.execution_inventory is None
        or cache.expected_rank != executor.EXPECTED_PRODUCTION_RANK
    ):
        raise PermutationExecutionAuthorityError(
            "permutation adapter requires the exact sealed-production cache"
        )
    adapter = WholeSetPermutationAdapter(_seal=_ADAPTER_SEAL)
    state = _make_state(
        owner=owner,
        source_cache=cache,
        cells=_source_cells(cache),
        candidates=cache.candidates,
        geometry=cache.geometry,
        mode=executor.SEALED_PRODUCTION_MODE,
        maximum_iterations=executor.FROZEN_MAXIMUM_ITERATIONS,
        expected_rank=executor.EXPECTED_PRODUCTION_RANK,
    )
    state.validator = _PermutationExecutionValidator(adapter)
    _register_adapter(adapter, state)
    return adapter


def _create_synthetic_qualification_adapter(
    cells: Sequence[m1.Cell],
    candidates: m1.CandidateLibrary,
    geometry: m1.GeometryEngine,
    *,
    session: object,
    authority: object,
    cache_authority: object,
    maximum_iterations: int = 1,
) -> WholeSetPermutationAdapter:
    """Unmistakably synthetic issuer used only by target-free qualification."""
    if (
        isinstance(maximum_iterations, bool)
        or not isinstance(maximum_iterations, int)
        or not 1 <= maximum_iterations <= executor.FROZEN_MAXIMUM_ITERATIONS
    ):
        raise PermutationExecutionAuthorityError(
            "synthetic maximum iterations are outside the frozen bound"
        )
    adapter = WholeSetPermutationAdapter(_seal=_ADAPTER_SEAL)
    state = _make_state(
        owner=None,
        source_cache=cache_authority,
        cells=cells,
        candidates=candidates,
        geometry=geometry,
        mode=executor.SYNTHETIC_TEST_MODE,
        maximum_iterations=maximum_iterations,
        expected_rank=None,
    )
    state.session_reference = _weak(session, "synthetic session")
    state.authority_reference = _weak(authority, "synthetic authority")
    state.validator = _PermutationExecutionValidator(adapter)
    _register_adapter(adapter, state)
    return adapter


def _require_adapter(
    adapter: WholeSetPermutationAdapter,
    *,
    require_bound: bool = True,
) -> _AdapterState:
    if type(adapter) is not WholeSetPermutationAdapter or adapter._seal is not _ADAPTER_SEAL:
        raise PermutationExecutionAuthorityError(
            "permutation adapter lacks exact named-module authority"
        )
    state = _lookup_adapter(adapter)
    if type(state) is not _AdapterState:
        raise PermutationExecutionAuthorityError("permutation adapter state is malformed")
    cache = state.source_cache_reference()
    candidates = state.candidates_reference()
    geometry = state.geometry_reference()
    if cache is None or candidates is None or geometry is None:
        raise PermutationExecutionAuthorityError(
            "permutation adapter source producer is no longer live"
        )
    if _DEEP_INTEGRITY_DIGEST(state.source_cells) != state.source_cells_digest:
        raise PermutationExecutionAuthorityError(
            "permutation source cells changed after adapter creation"
        )
    permutation_m1._require_library_anchors(
        candidates,
        geometry,
        state.candidate_anchors,
        state.geometry_anchor,
    )
    if state.mode == executor.SEALED_PRODUCTION_MODE:
        owner = None if state.owner_reference is None else state.owner_reference()
        if type(owner) is not executor.CoreTrialExecutor or owner.prepared is not cache:
            raise PermutationExecutionAuthorityError(
                "permutation source executor/cache binding is unavailable"
            )
        exact_cache = executor._require_prepared_cache(cache)
        if (
            exact_cache.mode != executor.SEALED_PRODUCTION_MODE
            or owner.maximum_iterations != executor.FROZEN_MAXIMUM_ITERATIONS
            or any(
                current is not issued
                for current, issued in zip(
                    _source_cells(exact_cache), state.source_cells, strict=True
                )
            )
        ):
            raise PermutationExecutionAuthorityError(
                "permutation source cache changed after named issuance"
            )
    elif state.mode != executor.SYNTHETIC_TEST_MODE or state.owner_reference is not None:
        raise PermutationExecutionAuthorityError("permutation adapter mode is invalid")
    if require_bound:
        if (
            state.session_reference is None
            or state.session_reference() is None
            or state.authority_reference is None
            or state.authority_reference() is None
        ):
            raise PermutationExecutionAuthorityError(
                "permutation adapter lacks its live session authority"
            )
        if state.mode == executor.SEALED_PRODUCTION_MODE:
            _require_observed_hooks(state.observed_hooks)
        elif state.observed_hooks is not None:
            raise PermutationExecutionAuthorityError(
                "synthetic permutation adapter carries production session hooks"
            )
    if state.mappings is None or state.result_records is None or state.validator is None:
        raise PermutationExecutionAuthorityError("permutation adapter registries are absent")
    return state


def _bind_to_observed_session(
    adapter: WholeSetPermutationAdapter,
    session: object,
    authority: object,
    owner: executor.CoreTrialExecutor,
) -> bool:
    """Private one-time hook for the later observed-session single owner."""
    state = _require_adapter(adapter, require_bound=False)
    if state.mode != executor.SEALED_PRODUCTION_MODE:
        raise PermutationExecutionAuthorityError(
            "observed-session binding rejects synthetic adapters"
        )
    callbacks = _REQUEST_OBSERVED_BIND(
        session,
        authority,
        adapter,
        owner,
        state.source_cache_reference(),
    )
    with state.lock:
        if (
            state.owner_reference is None
            or state.owner_reference() is not owner
            or owner.prepared is not state.source_cache_reference()
            or state.session_reference is not None
            or state.authority_reference is not None
            or state.observed_hooks is not None
            or state.issued_attempts != 0
            or state.mappings
            or state.result_records
        ):
            raise PermutationExecutionAuthorityError(
                "permutation adapter must bind to one pristine exact source session"
            )
        state.session_reference = _weak(session, "observed permutation session")
        state.authority_reference = _weak(authority, "observed permutation authority")
        state.observed_hooks = callbacks
    return True


_MAPPING_REQUIRE_TRIAL = mapping._require_trial
_MAPPING_CANONICAL_CELLS = mapping._canonical_cells
_MAPPING_ASSIGNMENT_VALIDATE = mapping.PermutationAssignment._validate
_MAPPING_ASSIGNMENT_APPLY = mapping.PermutationAssignment.apply
_MAPPING_APPLY_PERMUTATION = mapping.apply_permutation


def _require_fixed_mapping_functions() -> None:
    if (
        mapping._require_trial is not _MAPPING_REQUIRE_TRIAL
        or mapping._canonical_cells is not _MAPPING_CANONICAL_CELLS
        or mapping.PermutationAssignment._validate
        is not _MAPPING_ASSIGNMENT_VALIDATE
        or mapping.PermutationAssignment.apply is not _MAPPING_ASSIGNMENT_APPLY
        or mapping.apply_permutation is not _MAPPING_APPLY_PERMUTATION
    ):
        raise PermutationExecutionAuthorityError(
            "named permutation mapping functions were rebound"
        )


def _require_trial(trial: controller.Trial) -> controller.Trial:
    _require_fixed_mapping_functions()
    checked = _MAPPING_REQUIRE_TRIAL(trial)
    configuration = checked.configuration
    if (
        type(configuration) is not controller.Configuration
        or configuration.family not in controller.CORE_FAMILIES
        or type(checked.coverage_binding) is not controller.CoverageBinding
    ):
        raise PermutationExecutionAuthorityError(
            "whole-set permutation configuration is outside controller core families"
        )
    return checked


_BUILD_OBSERVED_MAPPING = mapping.build_observed_permutation_assignment
_BUILD_NULL_MAPPING = mapping.build_null_permutation_assignment
_FIT_PERMUTED_M1 = permutation_m1.fit_permuted_outer_m1
_PREPARE_PERMUTED_CACHE = permutation_m1.prepare_permuted_m2_cache
_REQUIRE_PERMUTED_M1_EXECUTION = permutation_m1._require_execution
_FRESH_PAIR_FEATURE_LIBRARY = permutation_m1._fresh_pair_feature_library
_CORE_EXECUTOR_CLASS = executor.CoreTrialExecutor
_CORE_EXECUTOR_INIT = executor.CoreTrialExecutor.__init__
_REQUIRE_PREPARED_CACHE = executor._require_prepared_cache
_VALIDATE_LAMBDA_MAXIMUM = executor._validate_lambda_maximum
_PENALTY_FROM_CONFIGURATION = executor.penalty_from_configuration
_VALIDATE_OPTIMIZER_RESULT = executor._validate_optimizer_result
_CORE_FOLD_LAMBDA_MAXIMUM = m2.fold_lambda_maximum
_CORE_FIT_M2 = m2.fit_m2
_CORE_EVALUATE_M2 = m2.evaluate_m2
_PREPARE_TRIAL_CACHE = executor.prepare_trial_cache


def _require_fixed_production_functions() -> None:
    _require_fixed_mapping_functions()
    if (
        mapping.build_observed_permutation_assignment is not _BUILD_OBSERVED_MAPPING
        or mapping.build_null_permutation_assignment is not _BUILD_NULL_MAPPING
        or permutation_m1.fit_permuted_outer_m1 is not _FIT_PERMUTED_M1
        or permutation_m1.prepare_permuted_m2_cache is not _PREPARE_PERMUTED_CACHE
        or permutation_m1._require_execution is not _REQUIRE_PERMUTED_M1_EXECUTION
        or permutation_m1._fresh_pair_feature_library
        is not _FRESH_PAIR_FEATURE_LIBRARY
        or executor.CoreTrialExecutor is not _CORE_EXECUTOR_CLASS
        or executor.CoreTrialExecutor.__init__ is not _CORE_EXECUTOR_INIT
        or executor._require_prepared_cache is not _REQUIRE_PREPARED_CACHE
        or executor._validate_lambda_maximum is not _VALIDATE_LAMBDA_MAXIMUM
        or executor.penalty_from_configuration is not _PENALTY_FROM_CONFIGURATION
        or executor._validate_optimizer_result is not _VALIDATE_OPTIMIZER_RESULT
        or executor._deep_integrity_digest is not _DEEP_INTEGRITY_DIGEST
        or executor._authority_binding_digest is not _AUTHORITY_BINDING_DIGEST
        or m2.fold_lambda_maximum is not _CORE_FOLD_LAMBDA_MAXIMUM
        or m2.fit_m2 is not _CORE_FIT_M2
        or m2.evaluate_m2 is not _CORE_EVALUATE_M2
        or executor.prepare_trial_cache is not _PREPARE_TRIAL_CACHE
        or _PermutationExecutionValidator.__call__ is not _VALIDATOR_CALL
        or _PermutationExecutionValidator.validate_execution_detail
        is not _VALIDATOR_DETAIL
        or _CONFIGURED_M2_RUNNER is not _execute_configured_core_m2
        or _observed_hook is not _OBSERVED_HOOK
        or _observed_request is not _OBSERVED_REQUEST
        or _request_observed_bind is not _REQUEST_OBSERVED_BIND
        or _request_observed_capability is not _REQUEST_OBSERVED_CAPABILITY
        or _request_observed_execute is not _REQUEST_OBSERVED_EXECUTE
        or _request_observed_ack is not _REQUEST_OBSERVED_ACK
    ):
        raise PermutationExecutionAuthorityError(
            "named permutation implementation functions were rebound"
        )


def _require_mapping_record(
    state: _AdapterState,
    trial: controller.Trial,
    null_replicate_index: int | None,
) -> _MappingRecord:
    assert state.mappings is not None
    if null_replicate_index is not None and (
        type(null_replicate_index) is not int
        or not 1 <= null_replicate_index <= len(controller.NULL_SEEDS)
    ):
        raise PermutationExecutionAuthorityError(
            "null permutation replicate index is outside exact range 1..99"
        )
    current = state.mappings.get(trial.valid_index)
    if current is not None:
        if (
            current.trial_reference() is not trial
            or current.trial_digest != _DEEP_INTEGRITY_DIGEST(trial)
            or type(current.null_replicate_index) is not type(null_replicate_index)
            or current.null_replicate_index != null_replicate_index
            or current.assignment_digest
            != _DEEP_INTEGRITY_DIGEST(current.assignment)
        ):
            raise PermutationExecutionAuthorityError(
                "permutation retry differs from its frozen trial or mapping"
            )
        _MAPPING_ASSIGNMENT_VALIDATE(current.assignment)
        return current

    if null_replicate_index is None:
        assignment = _BUILD_OBSERVED_MAPPING(state.source_cells, trial)
    else:
        assignment = _BUILD_NULL_MAPPING(
            state.source_cells,
            trial,
            null_replicate_index=null_replicate_index,
        )
    current = _MappingRecord(
        trial_reference=_weak(trial, "permutation trial"),
        trial_digest=_DEEP_INTEGRITY_DIGEST(trial),
        null_replicate_index=null_replicate_index,
        assignment=assignment,
        assignment_digest=_DEEP_INTEGRITY_DIGEST(assignment),
    )
    _MAPPING_ASSIGNMENT_VALIDATE(assignment)
    state.mappings[trial.valid_index] = current
    return current


def _capability_digest(record: _CapabilityRecord) -> bytes:
    adapter = record.adapter_reference()
    session = record.session_reference()
    authority = record.authority_reference()
    owner = (
        None if record.owner_reference is None else record.owner_reference()
    )
    source_cache = record.source_cache_reference()
    trial = record.trial_reference()
    assignment = record.assignment_reference()
    if any(
        value is None
        for value in (adapter, session, authority, source_cache, trial, assignment)
    ) or (record.owner_reference is not None and owner is None):
        raise PermutationExecutionAuthorityError(
            "permutation capability producer is no longer live"
        )
    return _AUTHORITY_BINDING_DIGEST(
        "whole_set_permutation_v1",
        id(adapter),
        id(session),
        id(authority),
        0 if owner is None else id(owner),
        id(source_cache),
        id(trial),
        record.trial_digest,
        id(assignment),
        record.assignment_digest,
        record.null_replicate_index,
        record.attempt_ordinal,
        record.session_attempt_ordinal,
        record.session_version,
    )


def _issue_permutation_attempt_capability(
    adapter: WholeSetPermutationAdapter,
    session: object,
    authority: object,
    trial: controller.Trial,
    *,
    null_replicate_index: int | None,
    session_attempt_ordinal: int,
    session_version: int,
) -> PermutationAttemptCapability:
    """Issue one capability after freezing/reusing the exact pre-M1 mapping."""
    state = _require_adapter(adapter)
    checked = _require_trial(trial)
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
        raise PermutationExecutionAuthorityError(
            "permutation capability session/authority/attempt binding is invalid"
        )
    owner = None if state.owner_reference is None else state.owner_reference()
    if state.mode == executor.SEALED_PRODUCTION_MODE:
        if type(owner) is not executor.CoreTrialExecutor:
            raise PermutationExecutionAuthorityError(
                "permutation capability lost its exact source owner"
            )
        _REQUEST_OBSERVED_CAPABILITY(
            state.observed_hooks,
            session,
            authority,
            adapter,
            owner,
            state.source_cache_reference(),
            checked,
            checked.coverage_binding,
            null_replicate_index,
            session_attempt_ordinal,
            session_version,
        )
    elif owner is not None:
        raise PermutationExecutionAuthorityError(
            "synthetic permutation capability unexpectedly has a source owner"
        )
    with state.lock:
        if session_attempt_ordinal <= state.last_session_attempt_ordinal:
            raise PermutationExecutionAuthorityError(
                "permutation capability session attempt chronology changed"
            )
        current = state.current_capability
        if current is not None:
            prior = _lookup_capability(current)
            if prior.phase not in {"completed_acked", "burned_acked"}:
                raise PermutationExecutionAuthorityError(
                    "a prior permutation capability is not ledger-terminal"
                )
        assert state.result_records is not None
        existing = state.result_records.get(checked.valid_index)
        if existing is not None:
            raise PermutationExecutionAuthorityError(
                "a successful permutation trial cannot be attempted or committed twice"
            )
        mapping_record = _require_mapping_record(
            state, checked, null_replicate_index
        )
        capability = PermutationAttemptCapability(_seal=_CAPABILITY_SEAL)
        record = _CapabilityRecord(
            adapter_reference=_weak(adapter, "permutation adapter"),
            session_reference=_weak(session, "permutation session"),
            authority_reference=_weak(authority, "permutation authority"),
            owner_reference=state.owner_reference,
            source_cache_reference=state.source_cache_reference,
            trial_reference=_weak(checked, "permutation trial"),
            trial_digest=mapping_record.trial_digest,
            assignment_reference=_weak(
                mapping_record.assignment, "permutation assignment"
            ),
            assignment_digest=mapping_record.assignment_digest,
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
    capability: PermutationAttemptCapability,
    *,
    require_current: bool,
) -> tuple[_CapabilityRecord, _AdapterState]:
    if (
        type(capability) is not PermutationAttemptCapability
        or capability._seal is not _CAPABILITY_SEAL
    ):
        raise PermutationExecutionAuthorityError(
            "permutation attempt capability lacks exact module authority"
        )
    record = _lookup_capability(capability)
    if type(record) is not _CapabilityRecord:
        raise PermutationExecutionAuthorityError("permutation capability record is malformed")
    adapter = record.adapter_reference()
    if adapter is None:
        raise PermutationExecutionAuthorityError("permutation adapter producer is dead")
    state = _require_adapter(adapter)
    trial = record.trial_reference()
    assignment = record.assignment_reference()
    owner = None if record.owner_reference is None else record.owner_reference()
    if (
        trial is None
        or assignment is None
        or state.session_reference is None
        or state.session_reference() is not record.session_reference()
        or state.authority_reference is None
        or state.authority_reference() is not record.authority_reference()
        or record.owner_reference is not state.owner_reference
        or state.source_cache_reference() is not record.source_cache_reference()
        or record.trial_digest != _DEEP_INTEGRITY_DIGEST(trial)
        or record.assignment_digest != _DEEP_INTEGRITY_DIGEST(assignment)
        or record.binding_digest != _capability_digest(record)
        or (require_current and state.current_capability is not capability)
    ):
        raise PermutationExecutionAuthorityError(
            "permutation capability live binding changed"
        )
    if state.mode == executor.SEALED_PRODUCTION_MODE:
        if (
            type(owner) is not executor.CoreTrialExecutor
            or owner.prepared is not record.source_cache_reference()
        ):
            raise PermutationExecutionAuthorityError(
                "permutation capability source owner/cache binding changed"
            )
    elif owner is not None:
        raise PermutationExecutionAuthorityError(
            "synthetic permutation capability acquired a source owner"
        )
    _MAPPING_ASSIGNMENT_VALIDATE(assignment)
    return record, state


def _identity_prefix(record: _CapabilityRecord) -> str:
    trial = record.trial_reference()
    assignment = record.assignment_reference()
    assert trial is not None and assignment is not None and trial.seed is not None
    null = "none" if record.null_replicate_index is None else str(record.null_replicate_index)
    return (
        f"permutation.{assignment.rng_route}.null_{null}.trial_{trial.valid_index}."
        f"seed_{trial.seed}.attempt_{record.attempt_ordinal}"
    )


def _validate_permuted_cache(
    m1_execution: permutation_m1.PermutationM1Execution,
    cache: executor.PreparedTrialCache,
    source_cache: object,
) -> executor.PreparedTrialCache:
    _REQUIRE_PERMUTED_M1_EXECUTION(m1_execution)
    exact = _REQUIRE_PREPARED_CACHE(cache)
    source_transforms = (
        ()
        if type(source_cache) is not executor.PreparedTrialCache
        else tuple(fold.training.transform for fold in source_cache.folds)
    )
    if (
        exact is source_cache
        or (
            type(source_cache) is executor.PreparedTrialCache
            and exact.pairs is source_cache.pairs
        )
        or exact.candidates is not m1_execution.candidates
        or exact.geometry is not m1_execution.geometry
        or exact.geometry.library is not exact.candidates
        or exact.pairs.candidates is not exact.candidates
        or type(exact.pairs) is not m2.PairFeatureLibrary
        or any(
            name in vars(exact.pairs)
            for name in ("basis", "statistics")
        )
        or len(exact.folds) != EXPECTED_FOLDS
        or exact.cell_inventory
        != tuple(
            (cell.identifier, cell.group) for cell in m1_execution.permuted_cells
        )
    ):
        raise PermutationProductionError(
            "permuted M2 cache is not bound to its exact fresh M1 execution"
        )
    seen: set[int] = set()
    for fold in exact.folds:
        spec = fold.spec
        offset = m1_execution.fixed_outer_offsets.get(spec.held_out_group)
        expected_cells = tuple(
            m1_execution.permuted_cells[index]
            for index in spec.validation_indexes
        )
        expected_training_cells = tuple(
            m1_execution.permuted_cells[index]
            for index in spec.training_indexes
        )
        expected_baseline = m1_execution.baseline_log_probability_nats[
            list(spec.validation_indexes)
        ]
        observed_cells = tuple(fold.validation.cells)
        observed_training_cells = tuple(fold.training.cells)
        if (
            fold.fixed_m1 is not offset
            or fold.training.transform is not fold.validation.transform
            or any(
                fold.training.transform is source_transform
                for source_transform in source_transforms
            )
            or len(observed_cells) != len(expected_cells)
            or any(
                observed is not expected
                for observed, expected in zip(
                    observed_cells, expected_cells, strict=True
                )
            )
            or len(observed_training_cells) != len(expected_training_cells)
            or any(
                observed is not expected
                for observed, expected in zip(
                    observed_training_cells,
                    expected_training_cells,
                    strict=True,
                )
            )
            or not _same_array_values_exact(
                fold.baseline_log_probability_nats, expected_baseline
            )
        ):
            raise PermutationProductionError(
                "permuted M2 fold lost its M1 offset, transform, or own baseline"
            )
        seen.update(spec.validation_indexes)
    if seen != set(range(EXPECTED_ROWS)):
        raise PermutationProductionError(
            "permuted M2 cache does not cover 194 unique held-out rows"
        )
    return exact


def _validate_applied_mapping(
    source_cells: tuple[m1.Cell, ...],
    assignment: mapping.PermutationAssignment,
    permuted_cells: tuple[m1.Cell, ...],
) -> None:
    """Independently bind the M1 rows to the frozen complete-set mapping."""
    if len(source_cells) != EXPECTED_ROWS or len(permuted_cells) != EXPECTED_ROWS:
        raise PermutationProductionError(
            "permutation M1 row inventory changed during mapping application"
        )
    for recipient_index, (recipient, observed) in enumerate(
        zip(source_cells, permuted_cells, strict=True)
    ):
        donor_index = assignment.recipient_to_donor[recipient_index]
        donor = source_cells[donor_index]
        if (
            type(observed) is not m1.Cell
            or not _same_scalar_tree_exact(
                observed.identifier, recipient.identifier
            )
            or not _same_scalar_tree_exact(observed.group, recipient.group)
            or not _same_scalar_tree_exact(observed.layer, recipient.layer)
            or not _same_scalar_tree_exact(
                observed.hemisphere, recipient.hemisphere
            )
            or not _same_scalar_tree_exact(observed.soma_um, recipient.soma_um)
            or type(observed.soma_distance_mm)
            is not type(recipient.soma_distance_mm)
            or not _same_array_values_exact(
                observed.soma_distance_mm, recipient.soma_distance_mm
            )
            or not _same_scalar_tree_exact(observed.k, recipient.k)
            or not _same_scalar_tree_exact(
                observed.segment_unmapped, recipient.segment_unmapped
            )
            or not _same_scalar_tree_exact(
                observed.sampling_uniform, recipient.sampling_uniform
            )
            or not _same_scalar_tree_exact(
                observed.observed_index, donor.observed_index
            )
        ):
            raise PermutationProductionError(
                "permutation M1 rows differ from the exact recipient-to-donor mapping"
            )


def _execute_configured_core_m2(
    cache: executor.PreparedTrialCache,
    trial: controller.Trial,
    maximum_iterations: int,
) -> tuple[tuple[PermutationFoldExecution, ...], controller.TrialResult]:
    """Fixed named 12-fold runner for the controller's permutation families."""
    exact = _REQUIRE_PREPARED_CACHE(cache)
    checked = _require_trial(trial)
    configuration = checked.configuration
    assert configuration is not None
    owner = _CORE_EXECUTOR_CLASS(
        exact, maximum_iterations=maximum_iterations
    )
    if type(owner) is not _CORE_EXECUTOR_CLASS:
        raise PermutationExecutionAuthorityError(
            "configured permutation M2 executor type changed"
        )
    folds: list[PermutationFoldExecution] = []
    seen: set[int] = set()
    for prepared in exact.folds:
        if configuration.family == "elastic_net":
            assert configuration.alpha is not None
            ratio = float(configuration.alpha)
        else:
            ratio = 1.0
        maximum = _VALIDATE_LAMBDA_MAXIMUM(
            _CORE_FOLD_LAMBDA_MAXIMUM(
                prepared.fixed_m1.theta,
                prepared.training,
                exact.candidates,
                exact.geometry,
                exact.pairs,
                configuration.family,
                l1_ratio=ratio,
            )
        )
        penalty = _PENALTY_FROM_CONFIGURATION(configuration, maximum)
        fit = _VALIDATE_OPTIMIZER_RESULT(
            _CORE_FIT_M2(
                prepared.fixed_m1.theta,
                prepared.training,
                exact.candidates,
                exact.geometry,
                exact.pairs,
                penalty.specification,
                maximum_iterations=maximum_iterations,
            ),
            maximum_iterations,
        )
        if configuration.rho == 1.0 and np.count_nonzero(fit.beta) != 0:
            raise PermutationProductionError(
                "rho=1 permutation fit did not retain beta zero"
            )
        prediction = _CORE_EVALUATE_M2(
            fit.beta,
            prepared.fixed_m1.theta,
            prepared.validation,
            exact.candidates,
            exact.geometry,
            exact.pairs,
            need_predictions=True,
        )
        if prediction.log_probability_nats is None:
            raise PermutationProductionError(
                "configured permutation M2 omitted held-out scores"
            )
        log_probability = np.asarray(prediction.log_probability_nats, dtype=float)
        if (
            log_probability.shape != prepared.baseline_log_probability_nats.shape
            or not np.all(np.isfinite(log_probability))
        ):
            raise PermutationProductionError(
                "configured permutation M2 held-out scores are malformed"
            )
        cell_scores: list[PermutationCellScore] = []
        for local, (global_index, cell) in enumerate(
            zip(
                prepared.spec.validation_indexes,
                prepared.validation.cells,
                strict=True,
            )
        ):
            if global_index in seen:
                raise PermutationProductionError(
                    "configured permutation M2 scored one cell twice"
                )
            seen.add(global_index)
            baseline = float(prepared.baseline_log_probability_nats[local])
            observed = float(log_probability[local])
            if cell.k in {0, m1.PANEL_SIZE} and observed != baseline:
                raise PermutationProductionError(
                    "K in {0,16} changed from its exact permuted M1 baseline"
                )
            gain = (observed - baseline) / LN2
            if not math.isfinite(gain):
                raise PermutationProductionError(
                    "configured permutation M2 produced a nonfinite gain"
                )
            cell_scores.append(
                PermutationCellScore(
                    global_index=global_index,
                    cell_key=str(cell.identifier),
                    group=str(cell.group),
                    k=int(cell.k),
                    m1_log_probability_nats=baseline,
                    m2_log_probability_nats=observed,
                    gain_bits=float(gain),
                )
            )
        fold_gain = math.fsum(cell.gain_bits for cell in cell_scores) / len(
            cell_scores
        )
        folds.append(
            PermutationFoldExecution(
                spec=prepared.spec,
                fixed_m1=prepared.fixed_m1,
                transform=prepared.training.transform,
                penalty=penalty,
                fit=fit,
                gain_bits_per_cell=float(fold_gain),
                active_pair_degrees=int(np.count_nonzero(fit.beta)),
                cells=tuple(cell_scores),
            )
        )
    if seen != set(range(EXPECTED_ROWS)):
        raise PermutationProductionError(
            "configured permutation M2 did not score exactly 194 unique cells"
        )
    result = controller.TrialResult(
        trial=checked,
        gain_bits_per_cell=float(
            math.fsum(fold.gain_bits_per_cell for fold in folds) / EXPECTED_FOLDS
        ),
        active_pair_degrees=max(fold.active_pair_degrees for fold in folds),
    )
    return tuple(folds), result


_CONFIGURED_M2_RUNNER = _execute_configured_core_m2


def _m1_identities(
    execution: permutation_m1.PermutationM1Execution,
) -> tuple[tuple[str, str, float], ...]:
    return tuple(
        (
            group,
            execution.fixed_outer_offsets[group].fit_identity,
            float(execution.selected_lambdas[group]),
        )
        for group in m1.GROUP_ORDER
    )


def execute_permutation_attempt(
    adapter: WholeSetPermutationAdapter,
    capability: PermutationAttemptCapability,
) -> PermutationTrialExecution:
    """Consume before M1 entry and burn the capability on every BaseException."""
    record, state = _require_capability(capability, require_current=True)
    if record.adapter_reference() is not adapter:
        raise PermutationExecutionAuthorityError(
            "permutation capability belongs to another adapter"
        )
    _require_fixed_mapping_functions()
    _require_observed_dispatch()
    if state.mode == executor.SEALED_PRODUCTION_MODE:
        _require_fixed_production_functions()
        session = record.session_reference()
        authority = record.authority_reference()
        owner = None if record.owner_reference is None else record.owner_reference()
        source_cache = record.source_cache_reference()
        trial = record.trial_reference()
        _REQUEST_OBSERVED_EXECUTE(
            state.observed_hooks,
            session,
            authority,
            adapter,
            owner,
            source_cache,
            trial,
            trial.coverage_binding if trial is not None else None,
            record.null_replicate_index,
            capability,
            record.binding_digest,
            record.session_attempt_ordinal,
            record.session_version,
        )
    with state.lock:
        if record.phase != "issued":
            raise PermutationExecutionAuthorityError(
                "permutation capability was already consumed"
            )
        record.phase = "consumed"
    commit_key: int | None = None
    commit_value: tuple[
        weakref.ReferenceType[Any], weakref.ReferenceType[Any]
    ] | None = None
    try:
        if state.mode == executor.SEALED_PRODUCTION_MODE:
            _require_fixed_production_functions()
        trial = record.trial_reference()
        assignment = record.assignment_reference()
        candidates = state.candidates_reference()
        geometry = state.geometry_reference()
        if trial is None or assignment is None or candidates is None or geometry is None:
            raise PermutationExecutionAuthorityError(
                "permutation numerical producer died before execution"
            )
        m1_execution = _FIT_PERMUTED_M1(
            state.source_cells,
            assignment,
            candidates,
            geometry,
            identity_prefix=_identity_prefix(record),
        )
        _REQUIRE_PERMUTED_M1_EXECUTION(m1_execution)
        _validate_applied_mapping(
            state.source_cells,
            assignment,
            m1_execution.permuted_cells,
        )
        if (
            m1_execution.assignment is not assignment
            or m1_execution.candidates is not candidates
            or m1_execution.geometry is not geometry
            or m1_execution.warm_path_count != permutation_m1.EXPECTED_WARM_PATHS
            or m1_execution.zero_refit_count != permutation_m1.EXPECTED_ZERO_REFITS
            or m1_execution.full_development_refit_run is not False
        ):
            raise PermutationProductionError(
                "fresh permutation M1 detail is incomplete or relabelled"
            )
        source_cache = state.source_cache_reference()
        if source_cache is None:
            raise PermutationExecutionAuthorityError(
                "permutation source cache died before M2 preparation"
            )
        if state.mode == executor.SEALED_PRODUCTION_MODE:
            assert type(source_cache) is executor.PreparedTrialCache
            mode = executor.SEALED_PRODUCTION_MODE
            inventory = source_cache.execution_inventory
            expected_rank = executor.EXPECTED_PRODUCTION_RANK
        else:
            mode = executor.SYNTHETIC_TEST_MODE
            inventory = None
            expected_rank = None
        cache = _PREPARE_PERMUTED_CACHE(
            m1_execution,
            candidates,
            geometry,
            mode=mode,
            execution_inventory=inventory,
            expected_rank=expected_rank,
        )
        cache = _validate_permuted_cache(m1_execution, cache, source_cache)
        folds, result = _CONFIGURED_M2_RUNNER(
            cache, trial, state.maximum_iterations
        )
        configuration = trial.configuration
        assert configuration is not None
        execution = PermutationTrialExecution(
            trial=trial,
            configuration=configuration,
            assignment=assignment,
            m1_execution=m1_execution,
            prepared_cache=cache,
            folds=folds,
            result=result,
            recipient_to_donor=assignment.recipient_to_donor,
            m1_fit_identities=_m1_identities(m1_execution),
            mode=state.mode,
            session_identity=id(record.session_reference()),
            owner_identity=(
                0
                if record.owner_reference is None
                else id(record.owner_reference())
            ),
            source_cache_identity=id(record.source_cache_reference()),
            authority_identity=id(record.authority_reference()),
            attempt_ordinal=record.attempt_ordinal,
            session_attempt_ordinal=record.session_attempt_ordinal,
            null_replicate_index=record.null_replicate_index,
            selector_eligible=False,
        )
        execution_digest = _DEEP_INTEGRITY_DIGEST(execution)
        binding = _ExecutionBinding(
            adapter_reference=record.adapter_reference,
            session_reference=record.session_reference,
            authority_reference=record.authority_reference,
            owner_reference=record.owner_reference,
            source_cache_reference=record.source_cache_reference,
            capability_reference=_weak(capability, "permutation capability"),
            capability_digest=record.binding_digest,
            execution_digest=execution_digest,
        )
        _register_execution(execution, binding)
        record.m1_execution_reference = _weak(
            m1_execution, "permutation M1 execution"
        )
        record.prepared_cache_reference = _weak(cache, "permuted M2 cache")
        record.execution_reference = _weak(execution, "permutation execution")
        record.execution_digest = execution_digest
        _require_registered_execution(adapter, execution)
        with state.lock:
            assert state.result_records is not None
            if execution.trial.valid_index in state.result_records:
                raise PermutationExecutionAuthorityError(
                    "permutation result would duplicate an existing commit"
                )
            commit_key = execution.trial.valid_index
            commit_value = (
                _weak(execution.result, "permutation result"),
                _weak(execution, "permutation execution"),
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
            record.m1_execution_reference = None
            record.prepared_cache_reference = None
            record.execution_reference = None
            record.execution_digest = None
            record.phase = "burned_unacked"
            record.completion = "error"
            record.error_type = (
                f"{type(error).__module__}.{type(error).__qualname__}"
            )
        raise
    return execution


def _validate_fold_detail(
    prepared: executor.PreparedFold,
    observed: PermutationFoldExecution,
    configuration: controller.Configuration,
    maximum_iterations: int,
) -> None:
    if (
        type(observed) is not PermutationFoldExecution
        or observed.spec is not prepared.spec
        or observed.fixed_m1 is not prepared.fixed_m1
        or observed.transform is not prepared.training.transform
        or prepared.validation.transform is not observed.transform
        or type(observed.penalty) is not executor.PenaltyMapping
        or type(observed.fit) is not m2.M2OptimizerResult
        or type(observed.cells) is not tuple
    ):
        raise PermutationExecutionAuthorityError(
            "permutation fold configuration, M1 identity, or transform changed"
        )
    _VALIDATE_LAMBDA_MAXIMUM(observed.penalty.lambda_maximum)
    expected_penalty = _PENALTY_FROM_CONFIGURATION(
        configuration, observed.penalty.lambda_maximum
    )
    if not _same_scalar_tree_exact(observed.penalty, expected_penalty):
        raise PermutationExecutionAuthorityError(
            "permutation fold penalty mapping differs from its exact configuration"
        )
    _VALIDATE_OPTIMIZER_RESULT(observed.fit, maximum_iterations)
    expected = tuple(
        zip(
            prepared.spec.validation_indexes,
            prepared.validation.cells,
            prepared.baseline_log_probability_nats,
            strict=True,
        )
    )
    if len(observed.cells) != len(expected):
        raise PermutationExecutionAuthorityError(
            "permutation fold score inventory changed"
        )
    for score, (global_index, cell, baseline) in zip(
        observed.cells, expected, strict=True
    ):
        if (
            type(score) is not PermutationCellScore
            or not _same_scalar_tree_exact(score.global_index, int(global_index))
            or not _same_scalar_tree_exact(score.cell_key, str(cell.identifier))
            or not _same_scalar_tree_exact(score.group, str(cell.group))
            or not _same_scalar_tree_exact(score.k, int(cell.k))
            or not _same_scalar_tree_exact(
                score.m1_log_probability_nats, float(baseline)
            )
            or type(score.m2_log_probability_nats) is not float
            or not math.isfinite(score.m2_log_probability_nats)
            or not _same_scalar_tree_exact(
                score.gain_bits,
                (score.m2_log_probability_nats - score.m1_log_probability_nats)
                / LN2,
            )
            or (
                score.k in {0, m1.PANEL_SIZE}
                and not _same_scalar_tree_exact(
                    score.m2_log_probability_nats,
                    score.m1_log_probability_nats,
                )
            )
        ):
            raise PermutationExecutionAuthorityError(
                "permutation held-out score changed or lost its own M1 baseline"
            )
    expected_gain = math.fsum(score.gain_bits for score in observed.cells) / len(
        observed.cells
    )
    if (
        not _same_scalar_tree_exact(observed.gain_bits_per_cell, expected_gain)
        or not _same_scalar_tree_exact(
            observed.active_pair_degrees,
            int(np.count_nonzero(np.asarray(observed.fit.beta, dtype=float))),
        )
    ):
        raise PermutationExecutionAuthorityError(
            "permutation fold summary differs from its retained scores and fit"
        )


def _require_registered_execution(
    adapter: WholeSetPermutationAdapter,
    execution: PermutationTrialExecution,
) -> controller.TrialResult:
    if type(execution) is not PermutationTrialExecution:
        raise PermutationExecutionAuthorityError(
            "permutation history lacks exact detailed execution"
        )
    binding = _lookup_execution(execution)
    if type(binding) is not _ExecutionBinding:
        raise PermutationExecutionAuthorityError(
            "permutation execution binding is malformed"
        )
    state = _require_adapter(adapter)
    capability = binding.capability_reference()
    if capability is None:
        raise PermutationExecutionAuthorityError(
            "permutation execution capability producer is dead"
        )
    record, capability_state = _require_capability(
        capability, require_current=False
    )
    assignment = record.assignment_reference()
    trial = record.trial_reference()
    owner = None if record.owner_reference is None else record.owner_reference()
    source_cache = record.source_cache_reference()
    if (
        capability_state is not state
        or binding.adapter_reference() is not adapter
        or binding.session_reference() is not record.session_reference()
        or binding.authority_reference() is not record.authority_reference()
        or binding.owner_reference is not record.owner_reference
        or binding.source_cache_reference() is not source_cache
        or binding.capability_digest != record.binding_digest
        or binding.execution_digest != _DEEP_INTEGRITY_DIGEST(execution)
        or record.execution_reference is None
        or record.execution_reference() is not execution
        or record.execution_digest != binding.execution_digest
        or record.phase not in {"consumed", "completed_unacked", "completed_acked"}
        or trial is None
        or assignment is None
        or execution.trial is not trial
        or execution.configuration is not trial.configuration
        or execution.assignment is not assignment
        or execution.recipient_to_donor is not assignment.recipient_to_donor
        or execution.mode != state.mode
        or execution.session_identity != id(record.session_reference())
        or execution.owner_identity != (0 if owner is None else id(owner))
        or execution.source_cache_identity != id(source_cache)
        or execution.authority_identity != id(record.authority_reference())
        or execution.attempt_ordinal != record.attempt_ordinal
        or execution.session_attempt_ordinal != record.session_attempt_ordinal
        or execution.null_replicate_index != record.null_replicate_index
        or execution.selector_eligible is not False
        or type(execution.folds) is not tuple
        or type(execution.result) is not controller.TrialResult
        or execution.result.trial is not trial
        or execution.result.valid is not True
        or execution.result.coverage_binding is not None
    ):
        raise PermutationExecutionAuthorityError(
            "permutation execution session/cache/trial/mapping binding changed"
        )
    _require_trial(trial)
    _MAPPING_ASSIGNMENT_VALIDATE(assignment)
    if _DEEP_INTEGRITY_DIGEST(assignment) != record.assignment_digest:
        raise PermutationExecutionAuthorityError(
            "permutation execution mapping changed after issuance"
        )
    m1_execution = _REQUIRE_PERMUTED_M1_EXECUTION(execution.m1_execution)
    _validate_applied_mapping(
        state.source_cells,
        assignment,
        m1_execution.permuted_cells,
    )
    cache = _validate_permuted_cache(
        m1_execution,
        execution.prepared_cache,
        source_cache,
    )
    if (
        record.m1_execution_reference is None
        or record.m1_execution_reference() is not m1_execution
        or record.prepared_cache_reference is None
        or record.prepared_cache_reference() is not cache
        or m1_execution.assignment is not assignment
        or execution.m1_fit_identities != _m1_identities(m1_execution)
        or len(execution.folds) != EXPECTED_FOLDS
    ):
        raise PermutationExecutionAuthorityError(
            "permutation M1/cache/fold identities changed"
        )
    configuration = execution.configuration
    assert configuration is not None
    all_scores: list[PermutationCellScore] = []
    for prepared, observed in zip(cache.folds, execution.folds, strict=True):
        _validate_fold_detail(
            prepared, observed, configuration, state.maximum_iterations
        )
        all_scores.extend(observed.cells)
    if (
        len(all_scores) != EXPECTED_ROWS
        or {score.global_index for score in all_scores} != set(range(EXPECTED_ROWS))
        or len({score.cell_key for score in all_scores}) != EXPECTED_ROWS
        or not _same_scalar_tree_exact(
            execution.result.gain_bits_per_cell,
            math.fsum(fold.gain_bits_per_cell for fold in execution.folds)
            / EXPECTED_FOLDS,
        )
        or not _same_scalar_tree_exact(
            execution.result.active_pair_degrees,
            max(fold.active_pair_degrees for fold in execution.folds),
        )
    ):
        raise PermutationExecutionAuthorityError(
            "permutation execution does not retain 12 folds and 194 unique scores"
        )
    return execution.result


def _completed_permutation_attempt_binding(
    adapter: WholeSetPermutationAdapter,
    session: object,
    authority: object,
    capability: PermutationAttemptCapability,
    expected_completion: str,
) -> tuple[int, bytes, str | None, PermutationTrialExecution | None]:
    record, _ = _require_capability(capability, require_current=True)
    if (
        expected_completion not in {"success", "error"}
        or record.adapter_reference() is not adapter
        or record.session_reference() is not session
        or record.authority_reference() is not authority
        or record.phase not in {"completed_unacked", "burned_unacked"}
        or record.completion != expected_completion
    ):
        raise PermutationExecutionAuthorityError(
            "permutation attempt is not terminal in this exact session"
        )
    detail = None if record.execution_reference is None else record.execution_reference()
    if record.completion == "success":
        if (
            record.error_type is not None
            or detail is None
            or _require_registered_execution(adapter, detail) is not detail.result
        ):
            raise PermutationExecutionAuthorityError(
                "successful permutation attempt lost its detailed execution"
            )
    elif (
        detail is not None
        or type(record.error_type) is not str
        or not record.error_type
    ):
        raise PermutationExecutionAuthorityError(
            "failed permutation attempt lost its exact error classification"
        )
    return record.attempt_ordinal, record.binding_digest, record.error_type, detail


def _acknowledge_permutation_attempt(
    adapter: WholeSetPermutationAdapter,
    session: object,
    authority: object,
    capability: PermutationAttemptCapability,
    ledger_attempt: object,
) -> tuple[int, bytes]:
    """Bind one terminal capability to the exact later session-ledger attempt."""
    record, state = _require_capability(capability, require_current=True)
    expected_outcomes = {"valid"} if record.completion == "success" else {
        "invalid",
        "resource",
    }
    trial = record.trial_reference()
    if (
        record.adapter_reference() is not adapter
        or record.session_reference() is not session
        or record.authority_reference() is not authority
        or record.phase not in {"completed_unacked", "burned_unacked"}
        or record.completion not in {"success", "error"}
        or trial is None
        or getattr(ledger_attempt, "ordinal", None)
        != record.session_attempt_ordinal
        or getattr(ledger_attempt, "trial", None) is not trial
        or getattr(ledger_attempt, "outcome", None) not in expected_outcomes
        or getattr(ledger_attempt, "error_type", None) != record.error_type
        or getattr(ledger_attempt, "execution_capability", None) is not capability
        or getattr(ledger_attempt, "execution_capability_digest", None)
        != record.binding_digest
        or getattr(ledger_attempt, "production_attempt_ordinal", None)
        != record.attempt_ordinal
    ):
        raise PermutationExecutionAuthorityError(
            "session ledger attempt differs from permutation capability"
        )
    if state.mode == executor.SEALED_PRODUCTION_MODE:
        owner = None if record.owner_reference is None else record.owner_reference()
        _REQUEST_OBSERVED_ACK(
            state.observed_hooks,
            session,
            authority,
            adapter,
            owner,
            record.source_cache_reference(),
            trial,
            trial.coverage_binding,
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
            raise PermutationExecutionAuthorityError(
                "permutation attempt acknowledgment chronology changed"
            )
        record.ledger_attempt_reference = _weak(
            ledger_attempt, "permutation ledger attempt"
        )
        record.ledger_attempt_digest = _DEEP_INTEGRITY_DIGEST(ledger_attempt)
        record.phase = (
            "completed_acked" if record.completion == "success" else "burned_acked"
        )
        state.acknowledged_attempts += 1
    return record.attempt_ordinal, record.binding_digest


def _require_acknowledged_permutation_attempt(
    adapter: WholeSetPermutationAdapter,
    session: object,
    authority: object,
    capability: PermutationAttemptCapability,
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
        or record.ledger_attempt_digest
        != _DEEP_INTEGRITY_DIGEST(ledger_attempt)
    ):
        raise PermutationExecutionAuthorityError(
            "permutation attempt ledger acknowledgment changed"
        )
    return record.attempt_ordinal, record.binding_digest


def _require_permutation_attempt_count(
    adapter: WholeSetPermutationAdapter,
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
            or type(acknowledged_attempts) is not int
            or acknowledged_attempts < 0
            or state.acknowledged_attempts != acknowledged_attempts
            or type(allow_pending) is not bool
        ):
            raise PermutationExecutionAuthorityError(
                "permutation adapter/session attempt count changed"
            )
        current = state.current_capability
        if current is None:
            return
        record = _lookup_capability(current)
        allowed = {"completed_acked", "burned_acked"}
        if allow_pending:
            allowed.update(
                {"issued", "consumed", "completed_unacked", "burned_unacked"}
            )
        if record.phase not in allowed:
            raise PermutationExecutionAuthorityError(
                "permutation adapter has an invalid pending attempt"
            )


def _adapter_producer_and_validator(
    adapter: WholeSetPermutationAdapter,
) -> tuple[WholeSetPermutationAdapter, _PermutationExecutionValidator]:
    # The observed-session owner obtains this exact producer/validator pair
    # immediately before the one allowed production bind request.
    state = _require_adapter(adapter, require_bound=False)
    assert state.validator is not None
    return adapter, state.validator


def _adapter_owner_and_cache(
    adapter: WholeSetPermutationAdapter,
) -> tuple[executor.CoreTrialExecutor, executor.PreparedTrialCache]:
    """Expose exact identities for the later sealed-receipt owner check."""
    state = _require_adapter(adapter)
    owner = None if state.owner_reference is None else state.owner_reference()
    cache = state.source_cache_reference()
    if (
        state.mode != executor.SEALED_PRODUCTION_MODE
        or type(owner) is not executor.CoreTrialExecutor
        or type(cache) is not executor.PreparedTrialCache
        or owner.prepared is not cache
    ):
        raise PermutationExecutionAuthorityError(
            "permutation adapter lacks its exact production owner/cache"
        )
    return owner, cache


def _validate_named_external_execution_detail(
    producer: object,
    validator: object,
    result: controller.TrialResult,
    execution_detail: object,
) -> bool:
    """Narrow exact handshake for later executor receipt integration."""
    if (
        type(producer) is not WholeSetPermutationAdapter
        or type(validator) is not _PermutationExecutionValidator
    ):
        return False
    try:
        state = _require_adapter(producer)
    except (PermutationProductionError, executor.CoreTrialExecutorError):
        return False
    if (
        state.mode != executor.SEALED_PRODUCTION_MODE
        or state.owner_reference is None
        or state.owner_reference() is None
        or state.validator is not validator
    ):
        return False
    return (
        _VALIDATOR_CALL(validator, producer, result) is True
        and _VALIDATOR_DETAIL(
            validator, producer, result, execution_detail
        )
        is True
    )


def _execution_from_receipt(
    adapter: WholeSetPermutationAdapter,
    receipt: executor.ExternalExecutionReceipt,
) -> PermutationTrialExecution:
    state = _require_adapter(adapter)
    if (
        state.mode != executor.SEALED_PRODUCTION_MODE
        or type(receipt) is not executor.ExternalExecutionReceipt
        or type(receipt.execution_detail) is not PermutationTrialExecution
        or receipt.execution_detail.result is not receipt.result
    ):
        raise PermutationExecutionAuthorityError(
            "permutation receipt lacks its complete immutable execution detail"
        )
    _require_registered_execution(adapter, receipt.execution_detail)
    return receipt.execution_detail


def _is_permutation_attempt_capability(value: object) -> bool:
    return type(value) is PermutationAttemptCapability and getattr(
        value, "_seal", None
    ) is _CAPABILITY_SEAL


__all__ = (
    "PermutationCellScore",
    "PermutationExecutionAuthorityError",
    "PermutationFoldExecution",
    "PermutationProductionError",
    "PermutationTrialExecution",
    "WholeSetPermutationAdapter",
    "create_whole_set_permutation_adapter",
    "execute_permutation_attempt",
)
