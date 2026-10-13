#!/usr/bin/env python3
"""Contract-fixed optional low-rank engine for EP10 v3 M2.

This module is deliberately a thin, target-free layer around
``fit_v3_m2``.  It maps the strict upper triangle of

    W = U diag(d) U.T,  U.T U = I,

through the training-fold M2 range projector and delegates every exact
fixed-K likelihood, gradient, and probability calculation to the core
engine.  It implements the prospectively frozen low-rank family and the v2
clarified sparse-plus-low-rank family in raw named physical-edge coordinates.
The prospective v3 clarification additionally fixes which already-scored
low-rank coverage fit initializes a hybrid whose rank differs from the
coverage-selected rank.
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

import fit_v3_m2 as core


LOW_RANKS = (1, 2, 3, 4, 6, 8)
RHO_GRID = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
LAMBDA_REFERENCE_FLOOR = 1.0e-12
ARMIJO_INITIAL_STEP = 1.0
ARMIJO_SHRINK_FACTOR = 0.5
ARMIJO_C1 = 1.0e-4
ARMIJO_MAXIMUM_LINE_TRIALS = 80
LOW_RANK_MAXIMUM_ITERATIONS = 20_000
LOW_RANK_GRADIENT_INFINITY_TOLERANCE = 1.0e-6
LOW_RANK_RELATIVE_OBJECTIVE_TOLERANCE = 1.0e-9
STIEFEL_ORTHOGONALITY_TOLERANCE = 1.0e-10
EIGENPAIR_ROUNDING_GUARD_FACTOR = 64.0 * core.m1.PANEL_SIZE
CLUSTER_GRAM_SCHMIDT_RESIDUAL_THRESHOLD = (
    256.0 * core.m1.PANEL_SIZE * np.finfo(float).eps
)
CLUSTER_CERTIFICATE_INFINITY_TOLERANCE = (
    1024.0 * core.m1.PANEL_SIZE * np.finfo(float).eps
)
HYBRID_MAXIMUM_SWEEPS = 20_000
HYBRID_JOINT_STATIONARITY_TOLERANCE = 1.0e-6
HYBRID_RELATIVE_OBJECTIVE_TOLERANCE = 1.0e-9
SPARSE_BACKTRACKING_MULTIPLIER = 2.0
SPARSE_MAXIMUM_LINE_TRIALS = 80
EXPECTED_FULL_DEVELOPMENT_ROWS = 194
EXPECTED_FULL_DEVELOPMENT_GROUPS = 12
PAIR_TO_INDEX = {edge: index for index, edge in enumerate(core.PAIR_INDEX)}
HYBRID_INITIALIZER_RULE_VERSION = "v3_target_rank_coverage_anchor"


class LowRankError(core.M2Error):
    """A low-rank mathematical or numerical contract failed."""


def _make_weak_identity_registry(label: str):
    """Return closure-private identity registration and lookup functions."""

    records: dict[int, tuple[weakref.ReferenceType[Any], Any]] = {}

    def register(instance: object, metadata: Any) -> None:
        identity = id(instance)
        existing = records.get(identity)
        if existing is not None and existing[0]() is not instance:
            raise LowRankError(f"{label} identity collision")

        def remove(reference: weakref.ReferenceType[Any]) -> None:
            current = records.get(identity)
            if current is not None and current[0] is reference:
                records.pop(identity, None)

        records[identity] = (weakref.ref(instance, remove), metadata)

    def lookup(instance: object) -> Any:
        record = records.get(id(instance))
        if record is None or record[0]() is not instance:
            raise LowRankError(f"{label} was not issued by this solver process")
        return record[1]

    return register, lookup


_register_low_rank_result_identity, _lookup_low_rank_result_identity = (
    _make_weak_identity_registry("low-rank optimizer result")
)
_register_hybrid_result_identity, _lookup_hybrid_result_identity = (
    _make_weak_identity_registry("hybrid optimizer result")
)
_register_hybrid_receipt_identity, _lookup_hybrid_receipt_identity = (
    _make_weak_identity_registry("production hybrid provenance receipt")
)
(
    _register_terminal_full_refit_receipt_identity,
    _lookup_terminal_full_refit_receipt_identity,
) = _make_weak_identity_registry("terminal full-development hybrid receipt")
_LOW_RANK_RESULT_SEAL = object()
_PRODUCTION_HYBRID_RECEIPT_SEAL = object()
_TERMINAL_FULL_REFIT_HYBRID_RECEIPT_SEAL = object()


def _readonly(array: np.ndarray, label: str) -> np.ndarray:
    result = np.array(array, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise LowRankError(f"{label} is nonfinite")
    result.setflags(write=False)
    return result


def _validate_rank(rank: int) -> int:
    value = int(rank)
    if value != rank or value not in LOW_RANKS:
        raise LowRankError(f"low-rank rank must be one of {LOW_RANKS}")
    return value


@dataclass(frozen=True)
class LowRankSpec:
    """One frozen low-rank rank/rho node in fold-local penalty units."""

    rank: int
    rho: float
    lambda_reference: float

    def __post_init__(self) -> None:
        _validate_rank(self.rank)
        if not math.isfinite(float(self.rho)) or float(self.rho) not in RHO_GRID:
            raise LowRankError("low-rank rho is outside the frozen grid")
        if (
            not math.isfinite(float(self.lambda_reference))
            or float(self.lambda_reference) < LAMBDA_REFERENCE_FLOOR
        ):
            raise LowRankError(
                "low-rank lambda reference must be finite and at least its frozen floor"
            )

    @property
    def strength(self) -> float:
        return float(self.rho) * float(self.lambda_reference)


@dataclass(frozen=True)
class LowRankState:
    """A finite Stiefel factor and its signed diagonal eigenvalues."""

    U: np.ndarray
    d: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.U, dtype=float)
        diagonal = np.asarray(self.d, dtype=float)
        if diagonal.ndim != 1:
            raise LowRankError("low-rank d must be one dimensional")
        rank = _validate_rank(int(diagonal.size))
        if matrix.shape != (core.m1.PANEL_SIZE, rank):
            raise LowRankError("low-rank U has the wrong shape")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(diagonal)):
            raise LowRankError("low-rank state is nonfinite")
        error = float(
            np.max(np.abs(matrix.T @ matrix - np.eye(rank, dtype=float)))
        )
        if error > STIEFEL_ORTHOGONALITY_TOLERANCE:
            raise LowRankError(
                "low-rank U does not have orthonormal columns: "
                f"maximum_error={error:.17g}"
            )
        object.__setattr__(self, "U", _readonly(matrix, "low-rank U"))
        object.__setattr__(self, "d", _readonly(diagonal, "low-rank d"))

    @property
    def rank(self) -> int:
        return int(self.d.size)


@dataclass(frozen=True)
class DeterministicEigensystem:
    values: np.ndarray
    vectors: np.ndarray
    raw_values: np.ndarray
    eigenpair_residuals: np.ndarray
    interval_radii: np.ndarray
    rounding_guard: float
    cluster_sizes: tuple[int, ...]
    cluster_means: np.ndarray


@dataclass(frozen=True)
class LowRankFoldScales:
    lambda_reference: float
    raw_gradient_spectral_norm: float
    smooth_lipschitz_upper_bound: float
    null_gradient: np.ndarray
    null_gradient_matrix: np.ndarray


@dataclass(frozen=True)
class LowRankInitialization:
    state: LowRankState
    fold_scales: LowRankFoldScales
    negative_gradient_step: float
    selected_gradient_eigenvalues: np.ndarray


@dataclass(frozen=True)
class LowRankEvaluation:
    objective: float
    smooth_objective: float
    penalty_value: float
    beta: np.ndarray
    physical_matrix: np.ndarray
    gradient_U: np.ndarray
    gradient_d: np.ndarray
    gradient_infinity: float
    core_evaluation: core.M2Evaluation


@dataclass(frozen=True)
class LowRankOptimizerResult:
    state: LowRankState
    beta: np.ndarray
    objective: float
    smooth_objective: float
    penalty_value: float
    iterations: int
    gradient_infinity: float
    relative_objective_change: float
    status: str
    telemetry: Mapping[str, Any]


@dataclass(frozen=True)
class HybridSpec:
    """One frozen sparse-plus-low-rank node in fold-local penalty units."""

    rank: int
    rho_low: float
    rho_sparse: float
    lambda_reference: float
    core_lasso_lambda_maximum: float
    named_edges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        _validate_rank(self.rank)
        for label, value in (
            ("rho_low", self.rho_low),
            ("rho_sparse", self.rho_sparse),
        ):
            if not math.isfinite(float(value)) or float(value) not in RHO_GRID:
                raise LowRankError(f"hybrid {label} is outside the frozen grid")
        if (
            not math.isfinite(float(self.lambda_reference))
            or float(self.lambda_reference) < LAMBDA_REFERENCE_FLOOR
        ):
            raise LowRankError(
                "hybrid lambda reference must be finite and at least its frozen floor"
            )
        if (
            not math.isfinite(float(self.core_lasso_lambda_maximum))
            or float(self.core_lasso_lambda_maximum) <= 0.0
        ):
            raise LowRankError(
                "hybrid core lasso lambda maximum must be finite and positive"
            )

        try:
            supplied_edges = tuple(self.named_edges)
        except TypeError as error:
            raise LowRankError("hybrid named edges are not iterable") from error
        if not 1 <= len(supplied_edges) <= 16:
            raise LowRankError("hybrid requires between 1 and 16 named edges")
        normalized: list[tuple[int, int]] = []
        for edge in supplied_edges:
            if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                raise LowRankError("each hybrid named edge must be a target pair")
            left_raw, right_raw = edge
            if (
                isinstance(left_raw, (bool, np.bool_))
                or isinstance(right_raw, (bool, np.bool_))
            ):
                raise LowRankError("hybrid named-edge coordinates must be integers")
            left = int(left_raw)
            right = int(right_raw)
            if left != left_raw or right != right_raw:
                raise LowRankError("hybrid named-edge coordinates must be integers")
            pair = (left, right)
            if pair not in PAIR_TO_INDEX:
                raise LowRankError("hybrid named edge is outside the physical panel")
            normalized.append(pair)
        indexes = tuple(PAIR_TO_INDEX[pair] for pair in normalized)
        if any(right <= left for left, right in zip(indexes, indexes[1:])):
            raise LowRankError(
                "hybrid named edges must be unique and in frozen lexicographic order"
            )
        object.__setattr__(self, "named_edges", tuple(normalized))

    @property
    def low_strength(self) -> float:
        return float(self.rho_low) * float(self.lambda_reference)

    @property
    def sparse_strength(self) -> float:
        return float(self.rho_sparse) * float(self.core_lasso_lambda_maximum)

    @property
    def edge_indexes(self) -> tuple[int, ...]:
        return tuple(PAIR_TO_INDEX[pair] for pair in self.named_edges)


@dataclass(frozen=True)
class HybridState:
    low_rank: LowRankState
    sparse_coefficients: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.low_rank, LowRankState):
            raise LowRankError("hybrid low-rank state is malformed")
        sparse = np.asarray(self.sparse_coefficients, dtype=float)
        if sparse.ndim != 1 or not 1 <= sparse.size <= 16:
            raise LowRankError("hybrid sparse coefficients have the wrong shape")
        if not np.all(np.isfinite(sparse)):
            raise LowRankError("hybrid sparse coefficients are nonfinite")
        object.__setattr__(
            self,
            "sparse_coefficients",
            _readonly(sparse, "hybrid sparse coefficients"),
        )


@dataclass(frozen=True)
class HybridEvaluation:
    objective: float
    smooth_objective: float
    low_rank_penalty_value: float
    sparse_penalty_value: float
    beta: np.ndarray
    physical_coefficient: np.ndarray
    gradient_U: np.ndarray
    gradient_d: np.ndarray
    gradient_a: np.ndarray
    sparse_stationarity_mapping: np.ndarray
    joint_stationarity_infinity: float
    core_evaluation: core.M2Evaluation


@dataclass(frozen=True)
class HybridOptimizerResult:
    state: HybridState
    beta: np.ndarray
    objective: float
    smooth_objective: float
    low_rank_penalty_value: float
    sparse_penalty_value: float
    sweeps: int
    joint_stationarity_infinity: float
    relative_objective_change: float
    sparse_lipschitz: float
    status: str
    telemetry: Mapping[str, Any]


class _ProductionHybridReceipt:
    """Opaque process-local capability issued by ``authorize_production_hybrid``."""

    __slots__ = ("_seal", "__weakref__")

    def __init__(self, *, _seal: object | None = None) -> None:
        if _seal is not _PRODUCTION_HYBRID_RECEIPT_SEAL:
            raise LowRankError(
                "production hybrid receipts must be issued by "
                "authorize_production_hybrid"
            )
        self._seal = _seal

    def __copy__(self):
        raise LowRankError("production hybrid receipts cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise LowRankError("production hybrid receipts cannot be copied")

    def __reduce__(self):
        raise LowRankError("production hybrid receipts cannot be serialized")


class _TerminalFullRefitHybridReceipt:
    """Opaque live-terminal authority for one cold/warm full hybrid pair."""

    __slots__ = ("_seal", "__weakref__")

    def __init__(self, *, _seal: object | None = None) -> None:
        if _seal is not _TERMINAL_FULL_REFIT_HYBRID_RECEIPT_SEAL:
            raise LowRankError(
                "terminal full-refit hybrid receipts require named issuance"
            )
        self._seal = _seal

    def __copy__(self):
        raise LowRankError("terminal full-refit hybrid receipts cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise LowRankError("terminal full-refit hybrid receipts cannot be copied")

    def __reduce__(self):
        raise LowRankError(
            "terminal full-refit hybrid receipts cannot be serialized"
        )


@dataclass(frozen=True)
class _ProductionHybridBinding:
    coverage_reference: weakref.ReferenceType[Any]
    low_rank_result_references: tuple[weakref.ReferenceType[Any], ...]
    selected_anchor_reference: weakref.ReferenceType[LowRankOptimizerResult]
    coverage_signature: tuple[object, ...]
    low_rank_result_signatures: tuple[tuple[object, ...], ...]
    selected_anchor_signature: tuple[object, ...]
    selected_configuration_key: str
    selected_rank: int
    selected_rho: float
    localized_edge_indexes: tuple[int, ...]
    target_configuration_reference: weakref.ReferenceType[Any]
    target_configuration_signature: tuple[object, ...]
    target_specification_reference: weakref.ReferenceType[HybridSpec]
    target_specification_signature: tuple[object, ...]
    initializer_configuration_key: str
    initializer_rule_version: str
    session_authority_reference: weakref.ReferenceType[Any] | None
    session_authority_digest: bytes | None


@dataclass(frozen=True)
class _ObjectiveBinding:
    """Process-local identity and integrity binding for one exact objective."""

    prepared_reference: weakref.ReferenceType[core.PreparedM2]
    candidates_reference: weakref.ReferenceType[core.CandidateLibrary]
    geometry_reference: weakref.ReferenceType[core.GeometryEngine]
    pairs_reference: weakref.ReferenceType[core.PairFeatureLibrary]
    fixed_m1_reference: weakref.ReferenceType[np.ndarray]
    fixed_m1_signature: tuple[object, ...]
    prepared_signature: tuple[object, ...]
    specification_signature: tuple[object, ...]
    linear_correction_signature: tuple[object, ...] | None


@dataclass
class _TerminalFullRefitHybridBinding:
    owner_reference: weakref.ReferenceType[Any]
    terminal_reference: weakref.ReferenceType[Any]
    procedure_authority_reference: weakref.ReferenceType[Any]
    full_objective_reference: weakref.ReferenceType[Any]
    winner_reference: weakref.ReferenceType[Any]
    coverage_reference: weakref.ReferenceType[Any]
    low_rank_result_references: tuple[weakref.ReferenceType[Any], ...]
    target_configuration_reference: weakref.ReferenceType[Any]
    initializer_configuration_reference: weakref.ReferenceType[Any]
    precursor_specification: LowRankSpec
    # Unlike controller configurations, the target HybridSpec is minted by
    # this authorization call and has no independent session/history owner.
    # The receipt binding must therefore keep the canonical object strongly
    # live while the parallel weak reference continues to prove identity.
    target_specification: HybridSpec
    target_specification_reference: weakref.ReferenceType[HybridSpec]
    coverage_signature: tuple[object, ...]
    low_rank_result_signatures: tuple[tuple[object, ...], ...]
    target_configuration_signature: tuple[object, ...]
    initializer_configuration_signature: tuple[object, ...]
    target_specification_signature: tuple[object, ...]
    objective: _ObjectiveBinding
    authorization_digest: bytes
    lock: threading.RLock
    phase: str = "issued_for_precursor"
    selected_anchor_reference: weakref.ReferenceType[LowRankOptimizerResult] | None = None
    selected_anchor_signature: tuple[object, ...] | None = None
    cold_result_reference: weakref.ReferenceType[Any] | None = None
    warm_result_reference: weakref.ReferenceType[Any] | None = None


@dataclass(frozen=True)
class _LowRankResultBinding:
    objective: _ObjectiveBinding
    objective_mode: str
    result_signature: tuple[object, ...]


@dataclass(frozen=True)
class _HybridResultBinding:
    objective: _ObjectiveBinding
    objective_mode: str
    selected_anchor_reference: weakref.ReferenceType[LowRankOptimizerResult]
    selected_anchor_signature: tuple[object, ...]
    production_receipt_reference: weakref.ReferenceType[object] | None
    result_signature: tuple[object, ...]


def _array_integrity_signature(array: np.ndarray) -> tuple[object, ...]:
    value = np.asarray(array)
    return (
        "array",
        tuple(value.shape),
        value.dtype.str,
        tuple(value.strides),
        bool(value.flags.writeable),
        value.tobytes(order="A"),
    )


def _immutable_integrity_signature(value: Any) -> object:
    if value is None or isinstance(value, (str, bytes, bool, np.bool_, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise LowRankError("integrity-bound metadata contains a nonfinite float")
        return ("float", number.hex())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _array_integrity_signature(value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_immutable_integrity_signature(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_immutable_integrity_signature(item) for item in value))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise LowRankError("integrity-bound metadata keys must be strings")
        return (
            "mapping",
            tuple(
                (key, _immutable_integrity_signature(value[key]))
                for key in sorted(value)
            ),
        )
    raise LowRankError(
        "integrity-bound metadata contains an unsupported mutable object"
    )


def _weak_reference(value: Any, label: str) -> weakref.ReferenceType[Any]:
    try:
        return weakref.ref(value)
    except TypeError as error:
        raise LowRankError(
            f"{label} must have a stable weak identity for a reproducible refit"
        ) from error


def _low_rank_specification_signature(
    specification: LowRankSpec,
) -> tuple[object, ...]:
    if type(specification) is not LowRankSpec:
        raise LowRankError("low-rank specification has the wrong concrete type")
    return (
        specification.rank,
        float(specification.rho).hex(),
        float(specification.lambda_reference).hex(),
    )


def _hybrid_specification_signature(
    specification: HybridSpec,
) -> tuple[object, ...]:
    if type(specification) is not HybridSpec:
        raise LowRankError("hybrid specification has the wrong concrete type")
    return (
        specification.rank,
        float(specification.rho_low).hex(),
        float(specification.rho_sparse).hex(),
        float(specification.lambda_reference).hex(),
        float(specification.core_lasso_lambda_maximum).hex(),
        specification.named_edges,
    )


def _cell_integrity_signature(cell: Any) -> tuple[object, ...]:
    try:
        return (
            type(cell),
            str(cell.identifier),
            str(cell.group),
            str(cell.layer),
            str(cell.hemisphere),
            tuple(float(value).hex() for value in cell.soma_um),
            _array_integrity_signature(np.asarray(cell.soma_distance_mm)),
            int(cell.k),
            int(cell.observed_index),
            int(cell.segment_unmapped),
            float(cell.sampling_uniform).hex(),
        )
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise LowRankError("prepared objective contains a malformed cell") from error


def _prepared_integrity_signature(
    prepared: core.PreparedM2,
) -> tuple[object, ...]:
    if type(prepared) is not core.PreparedM2:
        raise LowRankError("prepared M2 objective has the wrong concrete type")
    data = prepared.m1_data
    transform = prepared.transform
    try:
        signature = (
            tuple(_cell_integrity_signature(cell) for cell in data.cells),
            _array_integrity_signature(data.x),
            _array_integrity_signature(data.geometry_centers),
            _array_integrity_signature(data.geometry_scales),
            _array_integrity_signature(data.weights),
            _array_integrity_signature(data.design_transform.soma_mean),
            _array_integrity_signature(data.design_transform.soma_scale),
            data.design_transform is transform.design_transform,
            _array_integrity_signature(transform.geometry_scales),
            _array_integrity_signature(transform.projection_f_to_h),
            _array_integrity_signature(transform.pair_residual_scales),
            _array_integrity_signature(transform.range_basis),
            _array_integrity_signature(transform.range_projector),
            _array_integrity_signature(transform.residual_gram),
            _array_integrity_signature(transform.scaled_residual_gram),
            _immutable_integrity_signature(transform.diagnostics),
        )
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise LowRankError("prepared M2 objective is malformed") from error
    return signature


def _make_objective_binding(
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    specification_signature: tuple[object, ...],
    linear_correction: np.ndarray | None,
) -> _ObjectiveBinding:
    if geometry.library is not candidates or pairs.candidates is not candidates:
        raise LowRankError(
            "reproducible refits require one shared candidate/geometry/pair cache"
        )
    theta = np.asarray(fixed_m1_theta)
    if theta is not fixed_m1_theta:
        raise LowRankError(
            "fixed M1 theta must be an ndarray with stable objective identity"
        )
    correction_signature = (
        None
        if linear_correction is None
        else _array_integrity_signature(linear_correction)
    )
    return _ObjectiveBinding(
        prepared_reference=_weak_reference(prepared, "prepared M2 objective"),
        candidates_reference=_weak_reference(candidates, "candidate library"),
        geometry_reference=_weak_reference(geometry, "geometry engine"),
        pairs_reference=_weak_reference(pairs, "pair-feature library"),
        fixed_m1_reference=_weak_reference(fixed_m1_theta, "fixed M1 theta"),
        fixed_m1_signature=_array_integrity_signature(fixed_m1_theta),
        prepared_signature=_prepared_integrity_signature(prepared),
        specification_signature=specification_signature,
        linear_correction_signature=correction_signature,
    )


def _require_full_development_prepared(prepared: core.PreparedM2) -> None:
    import v3_search_controller_v1 as controller

    if type(prepared) is not core.PreparedM2:
        raise LowRankError("full-development refit requires an exact PreparedM2")
    cells = tuple(prepared.cells)
    inventory = tuple((str(cell.identifier), str(cell.group)) for cell in cells)
    data = prepared.m1_data
    if (
        len(cells) != EXPECTED_FULL_DEVELOPMENT_ROWS
        or inventory != controller.FROZEN_CELL_INVENTORY.entries
        or len({identifier for identifier, _ in inventory})
        != EXPECTED_FULL_DEVELOPMENT_ROWS
        or len({group for _, group in inventory})
        != EXPECTED_FULL_DEVELOPMENT_GROUPS
        or np.asarray(data.x).shape
        != (EXPECTED_FULL_DEVELOPMENT_ROWS, core.m1.DESIGN_DIMENSION)
        or np.asarray(data.geometry_centers).shape
        != (EXPECTED_FULL_DEVELOPMENT_ROWS, core.m1.GEOMETRY_DIMENSION)
        or np.asarray(data.weights).shape != (EXPECTED_FULL_DEVELOPMENT_ROWS,)
    ):
        raise LowRankError(
            "selected-model warm refit requires the exact aligned frozen "
            "194-row, 12-group full-development objective, not an outer fold"
        )


def _require_same_objective(
    binding: _ObjectiveBinding,
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    specification_signature: tuple[object, ...],
    linear_correction: np.ndarray | None,
) -> None:
    if type(binding) is not _ObjectiveBinding:
        raise LowRankError("optimizer result objective provenance is malformed")
    if (
        binding.prepared_reference() is not prepared
        or binding.candidates_reference() is not candidates
        or binding.geometry_reference() is not geometry
        or binding.pairs_reference() is not pairs
        or binding.fixed_m1_reference() is not fixed_m1_theta
    ):
        raise LowRankError(
            "converged result belongs to a different prepared objective or fixed M1"
        )
    correction_signature = (
        None
        if linear_correction is None
        else _array_integrity_signature(linear_correction)
    )
    if (
        binding.fixed_m1_signature
        != _array_integrity_signature(fixed_m1_theta)
        or binding.prepared_signature != _prepared_integrity_signature(prepared)
        or binding.specification_signature != specification_signature
        or binding.linear_correction_signature != correction_signature
        or geometry.library is not candidates
        or pairs.candidates is not candidates
    ):
        raise LowRankError(
            "converged result objective, specification, or inputs changed after fitting"
        )


def _low_rank_result_integrity_signature(
    result: LowRankOptimizerResult,
) -> tuple[object, ...]:
    if type(result) is not LowRankOptimizerResult:
        raise LowRankError("selected low-rank result has the wrong concrete type")
    return (
        _array_integrity_signature(result.state.U),
        _array_integrity_signature(result.state.d),
        _array_integrity_signature(result.beta),
        float(result.objective).hex(),
        float(result.smooth_objective).hex(),
        float(result.penalty_value).hex(),
        result.iterations,
        float(result.gradient_infinity).hex(),
        float(result.relative_objective_change).hex(),
        result.status,
        _immutable_integrity_signature(result.telemetry),
    )


def _hybrid_result_integrity_signature(
    result: HybridOptimizerResult,
) -> tuple[object, ...]:
    if type(result) is not HybridOptimizerResult:
        raise LowRankError("converged hybrid result has the wrong concrete type")
    return (
        _array_integrity_signature(result.state.low_rank.U),
        _array_integrity_signature(result.state.low_rank.d),
        _array_integrity_signature(result.state.sparse_coefficients),
        _array_integrity_signature(result.beta),
        float(result.objective).hex(),
        float(result.smooth_objective).hex(),
        float(result.low_rank_penalty_value).hex(),
        float(result.sparse_penalty_value).hex(),
        result.sweeps,
        float(result.joint_stationarity_infinity).hex(),
        float(result.relative_objective_change).hex(),
        float(result.sparse_lipschitz).hex(),
        result.status,
        _immutable_integrity_signature(result.telemetry),
    )


def _require_registered_low_rank_result(
    result: LowRankOptimizerResult, expected_mode: str
) -> _LowRankResultBinding:
    if type(result) is not LowRankOptimizerResult:
        raise LowRankError("selected low-rank result has the wrong concrete type")
    binding = _lookup_low_rank_result_identity(result)
    if type(binding) is not _LowRankResultBinding:
        raise LowRankError("selected low-rank result provenance is malformed")
    if binding.objective_mode != expected_mode:
        raise LowRankError("selected low-rank result objective mode is incompatible")
    if _low_rank_result_integrity_signature(result) != binding.result_signature:
        raise LowRankError("selected low-rank result changed after fitting")
    return binding


def _require_registered_hybrid_result(
    result: HybridOptimizerResult, expected_mode: str
) -> _HybridResultBinding:
    if type(result) is not HybridOptimizerResult:
        raise LowRankError("converged hybrid result has the wrong concrete type")
    binding = _lookup_hybrid_result_identity(result)
    if type(binding) is not _HybridResultBinding:
        raise LowRankError("converged hybrid result provenance is malformed")
    if binding.objective_mode != expected_mode:
        raise LowRankError("converged hybrid result objective mode is incompatible")
    if _hybrid_result_integrity_signature(result) != binding.result_signature:
        raise LowRankError("converged hybrid result changed after fitting")
    return binding


def _controller_binding_signature(binding: Any) -> object:
    if binding is None:
        return None
    return (
        _immutable_integrity_signature(binding.core_signature),
        _immutable_integrity_signature(binding.diagnostic_signature),
        _immutable_integrity_signature(binding.contrast_signature),
    )


def _controller_configuration_signature(configuration: Any) -> object:
    if configuration is None:
        return None
    return (
        configuration.family,
        float(configuration.rho).hex(),
        None
        if configuration.alpha is None
        else float(configuration.alpha).hex(),
        configuration.rank,
        None
        if configuration.sparse_rho is None
        else float(configuration.sparse_rho).hex(),
        tuple(configuration.sparse_edges),
        configuration.key,
    )


def _controller_trial_signature(trial: Any) -> tuple[object, ...]:
    return (
        trial.valid_index,
        trial.role,
        trial.kind,
        trial.phase,
        trial.falsifier,
        _controller_configuration_signature(trial.configuration),
        trial.seed,
        trial.cycle,
        _controller_configuration_signature(trial.matched_configuration),
        _controller_binding_signature(trial.coverage_binding),
    )


def _controller_result_signature(result: Any) -> tuple[object, ...]:
    return (
        _controller_trial_signature(result.trial),
        float(result.gain_bits_per_cell).hex(),
        result.active_pair_degrees,
        result.valid,
        _controller_binding_signature(result.coverage_binding),
    )


def _coverage_evidence_signature(evidence: Any) -> tuple[object, ...]:
    diagnostic = evidence.diagnostic
    return (
        tuple(_controller_result_signature(result) for result in evidence.core_results),
        (
            float(diagnostic.leading_energy).hex(),
            float(diagnostic.leading_edge_participation_ratio).hex(),
            float(diagnostic.median_leave_one_group_cosine).hex(),
            diagnostic.leave_one_group_cosines_at_least_0_70,
        ),
        tuple(float(value).hex() for value in evidence.equal_group_pair_contrasts),
        _controller_binding_signature(evidence.binding),
    )


def derive_hybrid_low_rank_initializer_configuration(
    coverage_evidence: object,
    low_rank_anchor_results: Sequence[object],
    target_configuration: object,
) -> object:
    """Derive the v3-prescribed low-rank initializer configuration.

    This function is a target-free policy derivation, not an execution
    authorization.  A production successor still needs the future named
    optional adapter to authenticate the exact session-issued target and its
    fold/cache entry.  Callers cannot nominate an initializer: the returned
    configuration is derived from the exact eight score-bearing coverage
    results and the target hybrid rank.
    """

    import v3_search_controller_v1 as controller

    if type(coverage_evidence) is not controller.CoverageEvidence:
        raise LowRankError("hybrid initializer derivation requires CoverageEvidence")
    anchors = tuple(low_rank_anchor_results)
    if len(anchors) != 8 or any(
        type(result) is not controller.TrialResult for result in anchors
    ):
        raise LowRankError(
            "hybrid initializer derivation requires the exact eight low-rank results"
        )
    if type(target_configuration) is not controller.Configuration:
        raise LowRankError("hybrid target configuration has the wrong concrete type")
    try:
        controller.CoverageEvidence.__post_init__(coverage_evidence)
        for result in anchors:
            controller.TrialResult.__post_init__(result)
        controller.Configuration.__post_init__(target_configuration)
    except (controller.ControllerError, TypeError, ValueError) as error:
        raise LowRankError(
            "hybrid initializer evidence, scores, or target changed after issuance"
        ) from error
    if (
        target_configuration.family != "sparse_plus_low_rank"
        or tuple(target_configuration.sparse_edges)
        != tuple(coverage_evidence.localized.edges)
    ):
        raise LowRankError(
            "hybrid target configuration does not match the activated edge identity"
        )
    try:
        selected_rank = controller.best_low_rank_anchor_rank(
            anchors, coverage_evidence
        )
        winner = controller.choose_incumbent(anchors)
    except (controller.ControllerError, TypeError, ValueError) as error:
        raise LowRankError(
            "controller rejected the hybrid initializer coverage provenance"
        ) from error
    selected = winner.trial.configuration
    if (
        type(selected) is not controller.Configuration
        or selected.family != "low_rank"
        or selected.rank != selected_rank
    ):
        raise LowRankError("controller did not identify one fixed low-rank winner")
    if target_configuration.rank == selected_rank:
        return selected

    prescribed = tuple(
        result.trial.configuration
        for result in anchors
        if result.trial.configuration is not None
        and result.trial.configuration.family == "low_rank"
        and result.trial.configuration.rank == target_configuration.rank
        and result.trial.configuration.rho == 0.1
    )
    if len(prescribed) != 1:
        raise LowRankError(
            "target rank lacks its unique mandatory rho-0.1 coverage anchor"
        )
    return prescribed[0]


def _require_hybrid_target_matches_specification(
    target_configuration: object,
    specification: HybridSpec,
) -> None:
    import v3_search_controller_v1 as controller

    if type(target_configuration) is not controller.Configuration:
        raise LowRankError("hybrid target configuration has the wrong concrete type")
    if type(specification) is not HybridSpec:
        raise LowRankError("hybrid target specification has the wrong concrete type")
    if (
        target_configuration.family != "sparse_plus_low_rank"
        or target_configuration.rank != specification.rank
        or float(target_configuration.rho) != float(specification.rho_low)
        or float(target_configuration.sparse_rho) != float(specification.rho_sparse)
        or tuple(target_configuration.sparse_edges) != specification.edge_indexes
    ):
        raise LowRankError(
            "hybrid target configuration and HybridSpec fields disagree"
        )


def _optional_adapter_request(name: str, *arguments: Any) -> Any:
    """Call only the exact episode-local named optional adapter.

    The import is deliberately lazy: the adapter uses this numerical module,
    while this module consults the adapter only for a live session-issued
    hybrid authority.  A public caller cannot substitute a generic callback or
    nominate a rank-changing initializer.
    """

    module = sys.modules.get("v3_m2_optional_adapter_v1")
    expected_path = Path(os.path.abspath(__file__)).parent / "v3_m2_optional_adapter_v1.py"
    if (
        module is None
        or Path(os.path.abspath(os.fspath(getattr(module, "SCRIPT_PATH", ""))))
        != expected_path
    ):
        raise LowRankError("session hybrid authorization requires the named optional adapter")
    callback = getattr(module, name, None)
    if not callable(callback):
        raise LowRankError("named optional adapter hybrid handshake is unavailable")
    try:
        return callback(*arguments)
    except LowRankError:
        raise
    except Exception as error:
        raise LowRankError("named optional adapter rejected hybrid provenance") from error


def _observed_session_request(name: str, *arguments: Any) -> Any:
    """Call only the live episode-local observed-session authority module."""

    module = sys.modules.get("v3_m2_observed_session_v1")
    expected_path = Path(os.path.abspath(__file__)).parent / "v3_m2_observed_session_v1.py"
    if (
        module is None
        or Path(os.path.abspath(os.fspath(getattr(module, "SCRIPT_PATH", ""))))
        != expected_path
    ):
        raise LowRankError(
            "terminal full refit requires the exact observed-session module"
        )
    callback = getattr(module, name, None)
    if not callable(callback):
        raise LowRankError("terminal full-refit authority handshake is unavailable")
    try:
        return callback(*arguments)
    except LowRankError:
        raise
    except Exception as error:
        raise LowRankError(
            "observed session rejected terminal full-refit provenance"
        ) from error


def _selected_refit_request(name: str, *arguments: Any) -> Any:
    """Call only the exact named selected-refit coordinator module."""

    module = sys.modules.get("v3_m2_selected_refit_v1")
    expected_path = Path(os.path.abspath(__file__)).parent / "v3_m2_selected_refit_v1.py"
    if (
        module is None
        or Path(os.path.abspath(os.fspath(getattr(module, "SCRIPT_PATH", ""))))
        != expected_path
    ):
        raise LowRankError("terminal hybrid requires the named selected-refit module")
    callback = getattr(module, name, None)
    if not callable(callback):
        raise LowRankError("selected-refit objective handshake is unavailable")
    try:
        return callback(*arguments)
    except LowRankError:
        raise
    except Exception as error:
        raise LowRankError("selected-refit objective authority was rejected") from error


def authorize_production_hybrid(
    coverage_evidence: object,
    low_rank_anchor_results: Sequence[object],
    selected_low_rank_anchor: LowRankOptimizerResult,
    target_configuration: object,
    specification: HybridSpec,
    *,
    _session_authority: object | None = None,
) -> object:
    """Issue a process-local receipt for the controller-selected hybrid inputs.

    The public path is deliberately limited to the four mandatory
    selected-rank hybrid coverage targets.  A private opaque authority from the
    named observed optional adapter may additionally authorize the exact live
    session target (including an adjacent-rank successor).  Raw callers still
    cannot nominate a target, initializer, session, fold, or cache owner.
    """

    import v3_search_controller_v1 as controller

    if type(coverage_evidence) is not controller.CoverageEvidence:
        raise LowRankError("production hybrid authorization requires CoverageEvidence")
    anchors = tuple(low_rank_anchor_results)
    if len(anchors) != 8 or any(
        type(result) is not controller.TrialResult for result in anchors
    ):
        raise LowRankError(
            "production hybrid authorization requires the exact eight low-rank results"
        )
    if not coverage_evidence.activation.sparse_plus_low_rank:
        raise LowRankError("coverage evidence did not activate the hybrid family")
    _require_hybrid_target_matches_specification(
        target_configuration, specification
    )
    try:
        selected_rank = controller.best_low_rank_anchor_rank(
            anchors, coverage_evidence
        )
        winner = controller.choose_incumbent(anchors)
        bound_configurations = controller.bound_optional_anchor_configurations(
            coverage_evidence, anchors
        )
    except (controller.ControllerError, TypeError, ValueError) as error:
        raise LowRankError(
            "controller rejected the hybrid production provenance"
        ) from error
    configuration = winner.trial.configuration
    if (
        configuration is None
        or configuration.family != "low_rank"
        or configuration.rank != selected_rank
        or len(bound_configurations) != 12
    ):
        raise LowRankError("controller did not identify one fixed low-rank winner")
    localized_edges = tuple(coverage_evidence.localized.edges)
    hybrid_configurations = bound_configurations[8:]
    if any(
        item.family != "sparse_plus_low_rank"
        or item.rank != selected_rank
        or tuple(item.sparse_edges) != localized_edges
        for item in hybrid_configurations
    ):
        raise LowRankError("controller hybrid rank or localized selector changed")
    initializer_configuration = derive_hybrid_low_rank_initializer_configuration(
        coverage_evidence,
        anchors,
        target_configuration,
    )
    session_authority_digest: bytes | None = None
    if _session_authority is None:
        if initializer_configuration is not configuration:
            raise LowRankError(
                "rank-changing production hybrid requires named session adapter authority"
            )
        target_matches = tuple(
            item for item in hybrid_configurations if item.key == target_configuration.key
        )
        if len(target_matches) != 1 or target_matches[0] != target_configuration:
            raise LowRankError(
                "hybrid successor target requires named session adapter authority"
            )
    else:
        session_authority_digest = _optional_adapter_request(
            "_require_hybrid_solver_authority",
            _session_authority,
            coverage_evidence,
            anchors,
            selected_low_rank_anchor,
            target_configuration,
            specification,
            initializer_configuration,
        )
        if type(session_authority_digest) is not bytes:
            raise LowRankError("named optional adapter returned a malformed authority digest")

    _require_registered_low_rank_result(selected_low_rank_anchor, "production")
    telemetry = selected_low_rank_anchor.telemetry
    if (
        telemetry.get("objective_mode") != "production"
        or telemetry.get("rank") != initializer_configuration.rank
        or telemetry.get("rho") != float(initializer_configuration.rho)
        or telemetry.get("lambda_reference")
        != float(specification.lambda_reference)
        or selected_low_rank_anchor.state.rank != initializer_configuration.rank
    ):
        raise LowRankError(
            "solver anchor telemetry does not match the controller-selected winner"
        )

    receipt = _ProductionHybridReceipt(_seal=_PRODUCTION_HYBRID_RECEIPT_SEAL)
    binding = _ProductionHybridBinding(
        coverage_reference=weakref.ref(coverage_evidence),
        low_rank_result_references=tuple(weakref.ref(result) for result in anchors),
        selected_anchor_reference=weakref.ref(selected_low_rank_anchor),
        coverage_signature=_coverage_evidence_signature(coverage_evidence),
        low_rank_result_signatures=tuple(
            _controller_result_signature(result) for result in anchors
        ),
        selected_anchor_signature=_low_rank_result_integrity_signature(
            selected_low_rank_anchor
        ),
        selected_configuration_key=configuration.key,
        selected_rank=int(configuration.rank),
        selected_rho=float(configuration.rho),
        localized_edge_indexes=localized_edges,
        target_configuration_reference=_weak_reference(
            target_configuration, "hybrid target configuration"
        ),
        target_configuration_signature=_controller_configuration_signature(
            target_configuration
        ),
        target_specification_reference=_weak_reference(
            specification, "hybrid target specification"
        ),
        target_specification_signature=_hybrid_specification_signature(
            specification
        ),
        initializer_configuration_key=initializer_configuration.key,
        initializer_rule_version=HYBRID_INITIALIZER_RULE_VERSION,
        session_authority_reference=(
            None
            if _session_authority is None
            else _weak_reference(_session_authority, "session hybrid authority")
        ),
        session_authority_digest=session_authority_digest,
    )
    _register_hybrid_receipt_identity(receipt, binding)
    return receipt


def _authorize_terminal_full_development_hybrid_impl(
    owner: object,
    terminal_authorization: object,
    procedure_authority: object,
    full_objective: object,
) -> object:
    """Prospectively reserve the terminal precursor/cold/warm hybrid chain."""

    (
        winner,
        coverage_evidence,
        anchors,
        target_configuration,
        initializer_configuration,
    ) = _observed_session_request(
        "_terminal_full_refit_hybrid_inputs_request",
        owner,
        terminal_authorization,
        procedure_authority,
        full_objective,
    )
    fixed_m1_theta, prepared, _, _ = _selected_refit_request(
        "_full_objective_authority_record",
        full_objective,
        procedure_authority,
        owner,
        terminal_authorization,
    )
    candidates = owner.prepared.candidates
    geometry = owner.prepared.geometry
    pairs = owner.prepared.pairs
    _require_full_development_prepared(prepared)
    scales = fold_low_rank_scales(
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
    )
    import v3_m2_trial_executor_v1 as trial_executor

    lasso_maximum = trial_executor._validate_lambda_maximum(
        core.fold_lambda_maximum(
            fixed_m1_theta,
            prepared,
            candidates,
            geometry,
            pairs,
            "lasso",
        )
    )
    try:
        named_edges = tuple(
            core.PAIR_INDEX[index] for index in target_configuration.sparse_edges
        )
    except (IndexError, TypeError) as error:
        raise LowRankError("terminal hybrid localized edges are malformed") from error
    specification = HybridSpec(
        int(target_configuration.rank),
        float(target_configuration.rho),
        float(target_configuration.sparse_rho),
        float(scales.lambda_reference),
        float(lasso_maximum.value),
        named_edges,
    )
    _require_hybrid_target_matches_specification(target_configuration, specification)
    precursor_specification = LowRankSpec(
        int(initializer_configuration.rank),
        float(initializer_configuration.rho),
        float(specification.lambda_reference),
    )
    authorization_digest = _observed_session_request(
        "_reserve_terminal_full_refit_hybrid_request",
        owner,
        terminal_authorization,
        procedure_authority,
        winner,
        coverage_evidence,
        anchors,
        target_configuration,
        initializer_configuration,
        full_objective,
        specification,
    )
    if type(authorization_digest) is not bytes:
        raise LowRankError("terminal full-refit authority digest is malformed")
    receipt = _TerminalFullRefitHybridReceipt(
        _seal=_TERMINAL_FULL_REFIT_HYBRID_RECEIPT_SEAL
    )
    binding = _TerminalFullRefitHybridBinding(
        owner_reference=_weak_reference(owner, "terminal refit owner"),
        terminal_reference=_weak_reference(
            terminal_authorization, "terminal refit authorization"
        ),
        procedure_authority_reference=_weak_reference(
            procedure_authority, "selected-refit procedure authority"
        ),
        full_objective_reference=_weak_reference(
            full_objective, "selected-refit full objective"
        ),
        winner_reference=_weak_reference(winner, "terminal refit winner"),
        coverage_reference=_weak_reference(
            coverage_evidence, "terminal refit coverage evidence"
        ),
        low_rank_result_references=tuple(
            _weak_reference(result, "terminal refit low-rank score")
            for result in anchors
        ),
        target_configuration_reference=_weak_reference(
            target_configuration, "terminal refit target configuration"
        ),
        initializer_configuration_reference=_weak_reference(
            initializer_configuration, "terminal refit precursor configuration"
        ),
        precursor_specification=precursor_specification,
        target_specification=specification,
        target_specification_reference=_weak_reference(
            specification, "terminal refit hybrid specification"
        ),
        coverage_signature=_coverage_evidence_signature(coverage_evidence),
        low_rank_result_signatures=tuple(
            _controller_result_signature(result) for result in anchors
        ),
        target_configuration_signature=_controller_configuration_signature(
            target_configuration
        ),
        initializer_configuration_signature=_controller_configuration_signature(
            initializer_configuration
        ),
        target_specification_signature=_hybrid_specification_signature(
            specification
        ),
        objective=_make_objective_binding(
            fixed_m1_theta,
            prepared,
            candidates,
            geometry,
            pairs,
            _hybrid_specification_signature(specification),
            None,
        ),
        authorization_digest=authorization_digest,
        lock=threading.RLock(),
    )
    _register_terminal_full_refit_receipt_identity(receipt, binding)
    _observed_session_request(
        "_bind_selected_refit_hybrid_receipt",
        owner,
        terminal_authorization,
        procedure_authority,
        full_objective,
        receipt,
        authorization_digest,
    )
    return receipt


def authorize_terminal_full_development_hybrid(
    owner: object,
    terminal_authorization: object,
    procedure_authority: object,
    full_objective: object,
) -> object:
    """Reserve one receipt before its precursor; any failed use burns it."""

    try:
        return _authorize_terminal_full_development_hybrid_impl(
            owner,
            terminal_authorization,
            procedure_authority,
            full_objective,
        )
    except BaseException:
        try:
            _observed_session_request(
                "_burn_selected_refit_procedure", procedure_authority
            )
        except BaseException:
            pass
        raise


def _require_terminal_full_development_hybrid_receipt(
    receipt: object,
) -> _TerminalFullRefitHybridBinding:
    if (
        type(receipt) is not _TerminalFullRefitHybridReceipt
        or getattr(receipt, "_seal", None)
        is not _TERMINAL_FULL_REFIT_HYBRID_RECEIPT_SEAL
    ):
        raise LowRankError("terminal full-refit hybrid receipt is invalid")
    binding = _lookup_terminal_full_refit_receipt_identity(receipt)
    if type(binding) is not _TerminalFullRefitHybridBinding:
        raise LowRankError("terminal full-refit hybrid binding is malformed")
    owner = binding.owner_reference()
    terminal = binding.terminal_reference()
    procedure_authority = binding.procedure_authority_reference()
    full_objective = binding.full_objective_reference()
    winner = binding.winner_reference()
    evidence = binding.coverage_reference()
    anchors = tuple(reference() for reference in binding.low_rank_result_references)
    target = binding.target_configuration_reference()
    initializer = binding.initializer_configuration_reference()
    bound_specification = binding.target_specification
    if (
        owner is None
        or terminal is None
        or procedure_authority is None
        or full_objective is None
        or winner is None
        or evidence is None
        or any(result is None for result in anchors)
        or target is None
        or initializer is None
        or binding.target_specification_reference() is not bound_specification
    ):
        raise LowRankError("terminal full-refit hybrid provenance is unavailable")
    if (
        _coverage_evidence_signature(evidence) != binding.coverage_signature
        or tuple(_controller_result_signature(result) for result in anchors)
        != binding.low_rank_result_signatures
        or _controller_configuration_signature(target)
        != binding.target_configuration_signature
        or _controller_configuration_signature(initializer)
        != binding.initializer_configuration_signature
        or _hybrid_specification_signature(bound_specification)
        != binding.target_specification_signature
    ):
        raise LowRankError("terminal full-refit hybrid provenance changed")
    fixed_m1_theta = binding.objective.fixed_m1_reference()
    prepared = binding.objective.prepared_reference()
    candidates = binding.objective.candidates_reference()
    geometry = binding.objective.geometry_reference()
    pairs = binding.objective.pairs_reference()
    if any(
        value is None
        for value in (fixed_m1_theta, prepared, candidates, geometry, pairs)
    ):
        raise LowRankError("terminal full-refit objective is unavailable")
    _require_same_objective(
        binding.objective,
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        _hybrid_specification_signature(bound_specification),
        None,
    )
    anchor = (
        None
        if binding.selected_anchor_reference is None
        else binding.selected_anchor_reference()
    )
    if (anchor is None) != (binding.selected_anchor_signature is None):
        raise LowRankError("terminal hybrid precursor binding is inconsistent")
    if anchor is not None and (
        _low_rank_result_integrity_signature(anchor)
        != binding.selected_anchor_signature
    ):
        raise LowRankError("terminal hybrid precursor changed")
    observed_digest = _observed_session_request(
        "_reauthenticate_terminal_full_refit_hybrid_request",
        owner,
        terminal,
        procedure_authority,
        winner,
        evidence,
        anchors,
        target,
        initializer,
        full_objective,
        bound_specification,
    )
    if observed_digest != binding.authorization_digest:
        raise LowRankError("terminal full-refit hybrid authority changed")
    return binding


def _authorize_session_production_hybrid(
    coverage_evidence: object,
    low_rank_anchor_results: Sequence[object],
    selected_low_rank_anchor: LowRankOptimizerResult,
    target_configuration: object,
    specification: HybridSpec,
    session_authority: object,
) -> object:
    """Internal named-adapter path for one exact session-issued hybrid target."""

    return authorize_production_hybrid(
        coverage_evidence,
        low_rank_anchor_results,
        selected_low_rank_anchor,
        target_configuration,
        specification,
        _session_authority=session_authority,
    )


def _require_production_hybrid_receipt(
    receipt: object,
    specification: HybridSpec,
    selected_low_rank_anchor: LowRankOptimizerResult,
) -> _ProductionHybridBinding:
    import v3_search_controller_v1 as controller

    if (
        type(receipt) is not _ProductionHybridReceipt
        or getattr(receipt, "_seal", None) is not _PRODUCTION_HYBRID_RECEIPT_SEAL
    ):
        raise LowRankError("production hybrid provenance receipt is invalid")
    binding = _lookup_hybrid_receipt_identity(receipt)
    if type(binding) is not _ProductionHybridBinding:
        raise LowRankError("production hybrid provenance binding is malformed")
    evidence = binding.coverage_reference()
    anchor_results = tuple(
        reference() for reference in binding.low_rank_result_references
    )
    registered_anchor = binding.selected_anchor_reference()
    target_configuration = binding.target_configuration_reference()
    target_specification = binding.target_specification_reference()
    if (
        evidence is None
        or any(result is None for result in anchor_results)
        or registered_anchor is not selected_low_rank_anchor
        or target_configuration is None
        or target_specification is not specification
    ):
        raise LowRankError("production hybrid provenance objects are unavailable")
    if type(evidence) is not controller.CoverageEvidence or any(
        type(result) is not controller.TrialResult for result in anchor_results
    ):
        raise LowRankError("production hybrid provenance types changed")
    if _coverage_evidence_signature(evidence) != binding.coverage_signature:
        raise LowRankError("coverage evidence changed after hybrid authorization")
    if (
        binding.initializer_rule_version != HYBRID_INITIALIZER_RULE_VERSION
        or _controller_configuration_signature(target_configuration)
        != binding.target_configuration_signature
        or _hybrid_specification_signature(specification)
        != binding.target_specification_signature
    ):
        raise LowRankError(
            "hybrid target configuration or specification changed after authorization"
        )
    _require_hybrid_target_matches_specification(
        target_configuration, specification
    )
    current_result_signatures = tuple(
        _controller_result_signature(result) for result in anchor_results
    )
    if current_result_signatures != binding.low_rank_result_signatures:
        raise LowRankError("low-rank score results changed after hybrid authorization")
    _require_registered_low_rank_result(selected_low_rank_anchor, "production")
    if (
        _low_rank_result_integrity_signature(selected_low_rank_anchor)
        != binding.selected_anchor_signature
    ):
        raise LowRankError("selected low-rank anchor changed after authorization")

    try:
        selected_rank = controller.best_low_rank_anchor_rank(
            anchor_results, evidence
        )
        winner = controller.choose_incumbent(anchor_results)
        bound_configurations = controller.bound_optional_anchor_configurations(
            evidence, anchor_results
        )
    except (controller.ControllerError, TypeError, ValueError) as error:
        raise LowRankError("controller provenance no longer validates") from error
    configuration = winner.trial.configuration
    if (
        configuration is None
        or configuration.key != binding.selected_configuration_key
        or configuration.rank != binding.selected_rank
        or float(configuration.rho) != binding.selected_rho
        or selected_rank != binding.selected_rank
        or len(bound_configurations) != 12
    ):
        raise LowRankError("controller-selected low-rank anchor changed")
    hybrid_configurations = bound_configurations[8:]
    session_authority = (
        None
        if binding.session_authority_reference is None
        else binding.session_authority_reference()
    )
    if binding.session_authority_reference is None:
        if binding.session_authority_digest is not None:
            raise LowRankError("public hybrid receipt carries session authority metadata")
        if not any(
            item.key == target_configuration.key and item == target_configuration
            for item in hybrid_configurations
        ):
            raise LowRankError(
                "hybrid target is no longer an exact mandatory coverage configuration"
            )
    else:
        if session_authority is None or type(binding.session_authority_digest) is not bytes:
            raise LowRankError("session hybrid authority is no longer available")
    localized_edges = tuple(evidence.localized.edges)
    if localized_edges != binding.localized_edge_indexes:
        raise LowRankError("controller localized sparse selector changed")
    initializer_configuration = derive_hybrid_low_rank_initializer_configuration(
        evidence,
        anchor_results,
        target_configuration,
    )
    if (
        specification.rank != target_configuration.rank
        or specification.edge_indexes != binding.localized_edge_indexes
        or initializer_configuration.key != binding.initializer_configuration_key
    ):
        raise LowRankError(
            "hybrid specification does not match its controller-bound target and initializer"
        )
    telemetry = selected_low_rank_anchor.telemetry
    if (
        telemetry.get("objective_mode") != "production"
        or telemetry.get("rank") != initializer_configuration.rank
        or telemetry.get("rho") != float(initializer_configuration.rho)
        or telemetry.get("lambda_reference")
        != float(specification.lambda_reference)
    ):
        raise LowRankError(
            "selected low-rank anchor telemetry does not match the hybrid specification"
        )
    if session_authority is not None:
        observed_digest = _optional_adapter_request(
            "_require_hybrid_solver_authority",
            session_authority,
            evidence,
            anchor_results,
            selected_low_rank_anchor,
            target_configuration,
            specification,
            initializer_configuration,
        )
        if observed_digest != binding.session_authority_digest:
            raise LowRankError("session hybrid authority changed after receipt issuance")
    return binding


def physical_gradient_matrix(edge_gradient: np.ndarray) -> np.ndarray:
    """Return the symmetric matrix adjoint of strict-upper extraction.

    Each physical edge gradient is split equally between its two symmetric
    matrix entries, as required by the frozen numerical addendum.
    """

    vector = np.asarray(edge_gradient, dtype=float)
    if vector.shape != (core.PAIR_DIMENSION,) or not np.all(np.isfinite(vector)):
        raise LowRankError("physical edge gradient is malformed or nonfinite")
    matrix = np.zeros((core.m1.PANEL_SIZE, core.m1.PANEL_SIZE), dtype=float)
    for edge_index, (left, right) in enumerate(core.PAIR_INDEX):
        value = 0.5 * float(vector[edge_index])
        matrix[left, right] = value
        matrix[right, left] = value
    return _readonly(matrix, "physical gradient matrix")


def offdiagonal_vector(matrix: np.ndarray) -> np.ndarray:
    """Extract the frozen lexicographic strict upper triangle."""

    value = np.asarray(matrix, dtype=float)
    expected = (core.m1.PANEL_SIZE, core.m1.PANEL_SIZE)
    if value.shape != expected or not np.all(np.isfinite(value)):
        raise LowRankError("physical low-rank matrix is malformed or nonfinite")
    symmetry_error = float(np.max(np.abs(value - value.T)))
    if symmetry_error > 32.0 * np.finfo(float).eps * max(
        1.0, float(np.max(np.abs(value)))
    ):
        raise LowRankError("physical low-rank matrix is not symmetric")
    result = np.asarray([value[left, right] for left, right in core.PAIR_INDEX])
    return _readonly(result, "physical off-diagonal vector")


def low_rank_matrix(state: LowRankState) -> np.ndarray:
    """Materialize ``U diag(d) U.T`` in binary64."""

    matrix = (state.U * state.d[np.newaxis, :]) @ state.U.T
    matrix = 0.5 * (matrix + matrix.T)
    return _readonly(matrix, "physical low-rank matrix")


def low_rank_effective_beta(
    state: LowRankState, transform: core.ResidualTransform
) -> np.ndarray:
    """Map a physical low-rank matrix into the identified fold range."""

    physical = offdiagonal_vector(low_rank_matrix(state))
    beta = core._range_project_vector(physical, transform)
    return _readonly(beta, "effective low-rank beta")


def _embed_named_sparse(
    sparse_coefficients: np.ndarray, specification: HybridSpec
) -> np.ndarray:
    sparse = np.asarray(sparse_coefficients, dtype=float)
    if sparse.shape != (len(specification.named_edges),) or not np.all(
        np.isfinite(sparse)
    ):
        raise LowRankError("hybrid sparse coefficient vector is malformed or nonfinite")
    embedded = np.zeros(core.PAIR_DIMENSION, dtype=float)
    embedded[np.asarray(specification.edge_indexes, dtype=int)] = sparse
    return _readonly(embedded, "embedded named sparse coefficients")


def hybrid_effective_beta(
    state: HybridState,
    specification: HybridSpec,
    transform: core.ResidualTransform,
) -> np.ndarray:
    """Return ``P[offdiag(U diag(d) U.T) + E_C a]`` exactly once."""

    if state.low_rank.rank != specification.rank:
        raise LowRankError("hybrid state and specification ranks disagree")
    if state.sparse_coefficients.size != len(specification.named_edges):
        raise LowRankError("hybrid state and named edge set disagree")
    physical = offdiagonal_vector(low_rank_matrix(state.low_rank))
    physical = physical + _embed_named_sparse(
        state.sparse_coefficients, specification
    )
    beta = core._range_project_vector(physical, transform)
    return _readonly(beta, "effective sparse-plus-low-rank beta")


def _soft_threshold(vector: np.ndarray, threshold: float) -> np.ndarray:
    source = np.asarray(vector, dtype=float)
    if not np.all(np.isfinite(source)):
        raise LowRankError("hybrid soft-threshold source is nonfinite")
    if not math.isfinite(float(threshold)) or float(threshold) < 0.0:
        raise LowRankError("hybrid soft-threshold amount is invalid")
    result = np.sign(source) * np.maximum(np.abs(source) - float(threshold), 0.0)
    return _readonly(result, "hybrid soft-threshold result")


def _initial_sparse_lipschitz(
    specification: HybridSpec,
    transform: core.ResidualTransform,
    core_lipschitz_upper_bound: float,
) -> tuple[float, float]:
    bound = float(core_lipschitz_upper_bound)
    if not math.isfinite(bound) or bound < 0.0:
        raise LowRankError("hybrid core smooth Lipschitz bound is invalid")
    indexes = np.asarray(specification.edge_indexes, dtype=int)
    basis = np.asarray(transform.range_basis, dtype=float)
    projected_embedding = basis @ basis[indexes, :].T
    try:
        singular_values = np.linalg.svd(projected_embedding, compute_uv=False)
    except np.linalg.LinAlgError as error:
        raise LowRankError("hybrid projected named-edge norm failed") from error
    if not np.all(np.isfinite(singular_values)):
        raise LowRankError("hybrid projected named-edge norm is nonfinite")
    spectral_norm = float(singular_values[0]) if singular_values.size else 0.0
    initial = max(1.0, bound * spectral_norm * spectral_norm)
    if not math.isfinite(initial) or initial <= 0.0:
        raise LowRankError("hybrid initial sparse Lipschitz state is invalid")
    return float(initial), spectral_norm


def _evaluate_hybrid_impl(
    state: HybridState,
    specification: HybridSpec,
    sparse_lipschitz: float,
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    *,
    need_predictions: bool,
    linear_correction: np.ndarray | None,
) -> HybridEvaluation:
    if state.low_rank.rank != specification.rank:
        raise LowRankError("hybrid state and specification ranks disagree")
    if state.sparse_coefficients.size != len(specification.named_edges):
        raise LowRankError("hybrid state and named edge set disagree")
    lipschitz = float(sparse_lipschitz)
    if not math.isfinite(lipschitz) or lipschitz <= 0.0:
        raise LowRankError("hybrid sparse Lipschitz state must be finite and positive")

    physical_matrix = low_rank_matrix(state.low_rank)
    physical_coefficient = offdiagonal_vector(physical_matrix)
    physical_coefficient = physical_coefficient + _embed_named_sparse(
        state.sparse_coefficients, specification
    )
    beta = core._range_project_vector(physical_coefficient, prepared.transform)
    evaluation = core.evaluate_m2(
        beta,
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        need_predictions=need_predictions,
        linear_correction=linear_correction,
    )
    projected_gradient = core._range_project_vector(
        evaluation.gradient, prepared.transform
    )
    gradient_matrix = physical_gradient_matrix(projected_gradient)
    matrix_times_U = gradient_matrix @ state.low_rank.U
    ambient_U = 2.0 * matrix_times_U * state.low_rank.d[np.newaxis, :]
    gradient_U = stiefel_tangent_projection(state.low_rank.U, ambient_U)
    gradient_d = np.einsum(
        "ij,ij->j", state.low_rank.U, matrix_times_U, optimize=True
    ) + specification.low_strength * state.low_rank.d
    gradient_a = projected_gradient[
        np.asarray(specification.edge_indexes, dtype=int)
    ]

    sparse_prox = _soft_threshold(
        state.sparse_coefficients - gradient_a / lipschitz,
        specification.sparse_strength / lipschitz,
    )
    sparse_mapping = lipschitz * (state.sparse_coefficients - sparse_prox)
    low_penalty = 0.5 * float(np.dot(state.low_rank.d, state.low_rank.d))
    sparse_penalty = float(np.sum(np.abs(state.sparse_coefficients), dtype=float))
    objective = (
        float(evaluation.objective)
        + specification.low_strength * low_penalty
        + specification.sparse_strength * sparse_penalty
    )
    joint_stationarity = max(
        float(np.max(np.abs(gradient_U))),
        float(np.max(np.abs(gradient_d))),
        float(np.max(np.abs(sparse_mapping))),
    )
    if not all(
        math.isfinite(value)
        for value in (
            objective,
            low_penalty,
            sparse_penalty,
            joint_stationarity,
        )
    ) or not (
        np.all(np.isfinite(gradient_d))
        and np.all(np.isfinite(gradient_a))
        and np.all(np.isfinite(sparse_mapping))
    ):
        raise LowRankError("hybrid objective, penalty, or stationarity is nonfinite")
    return HybridEvaluation(
        objective=float(objective),
        smooth_objective=float(evaluation.objective),
        low_rank_penalty_value=low_penalty,
        sparse_penalty_value=sparse_penalty,
        beta=_readonly(beta, "hybrid effective beta"),
        physical_coefficient=_readonly(
            physical_coefficient, "hybrid physical coefficient"
        ),
        gradient_U=gradient_U,
        gradient_d=_readonly(gradient_d, "hybrid d gradient"),
        gradient_a=_readonly(gradient_a, "hybrid sparse gradient"),
        sparse_stationarity_mapping=_readonly(
            sparse_mapping, "hybrid sparse stationarity mapping"
        ),
        joint_stationarity_infinity=joint_stationarity,
        core_evaluation=evaluation,
    )


def evaluate_sparse_plus_low_rank(
    state: HybridState,
    specification: HybridSpec,
    sparse_lipschitz: float,
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    *,
    need_predictions: bool = False,
) -> HybridEvaluation:
    """Evaluate the production hybrid objective without a synthetic correction."""

    return _evaluate_hybrid_impl(
        state,
        specification,
        sparse_lipschitz,
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        need_predictions=need_predictions,
        linear_correction=None,
    )


def evaluate_sparse_plus_low_rank_synthetic_qualification(
    state: HybridState,
    specification: HybridSpec,
    sparse_lipschitz: float,
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    *,
    linear_correction: np.ndarray,
    need_predictions: bool = False,
) -> HybridEvaluation:
    """Evaluate an explicitly target-free synthetic hybrid objective."""

    correction = np.asarray(linear_correction, dtype=float)
    if correction.shape != (core.PAIR_DIMENSION,) or not np.all(
        np.isfinite(correction)
    ):
        raise LowRankError(
            "synthetic hybrid linear correction is malformed or nonfinite"
        )
    return _evaluate_hybrid_impl(
        state,
        specification,
        sparse_lipschitz,
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        need_predictions=need_predictions,
        linear_correction=correction,
    )


def canonicalize_state(state: LowRankState) -> LowRankState:
    """Apply the representation-only frozen order and column-sign rules."""

    order = sorted(
        range(state.rank),
        key=lambda index: (
            -abs(float(state.d[index])),
            -float(state.d[index]),
            index,
        ),
    )
    U = np.array(state.U[:, order], dtype=float, copy=True)
    d = np.array(state.d[order], dtype=float, copy=True)
    for column in range(state.rank):
        pivot = int(np.argmax(np.abs(U[:, column])))
        if U[pivot, column] < 0.0:
            U[:, column] *= -1.0
    return LowRankState(U, d)


def stiefel_tangent_projection(U: np.ndarray, ambient: np.ndarray) -> np.ndarray:
    """Embedded-Euclidean projection onto the Stiefel tangent space."""

    matrix = np.asarray(U, dtype=float)
    source = np.asarray(ambient, dtype=float)
    if matrix.ndim != 2 or source.shape != matrix.shape:
        raise LowRankError("Stiefel projection operands have incompatible shapes")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(source)):
        raise LowRankError("Stiefel projection operands are nonfinite")
    cross = matrix.T @ source
    symmetric = 0.5 * (cross + cross.T)
    result = source - matrix @ symmetric
    return _readonly(result, "Stiefel tangent gradient")


def qr_retraction(U: np.ndarray, tangent: np.ndarray, step: float) -> np.ndarray:
    """Reduced-QR Stiefel retraction with deterministic diagonal signs."""

    matrix = np.asarray(U, dtype=float)
    direction = np.asarray(tangent, dtype=float)
    if matrix.ndim != 2 or direction.shape != matrix.shape:
        raise LowRankError("QR retraction operands have incompatible shapes")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(direction)):
        raise LowRankError("QR retraction operands are nonfinite")
    if not math.isfinite(float(step)) or float(step) < 0.0:
        raise LowRankError("QR retraction step must be finite and nonnegative")
    try:
        Q, R = np.linalg.qr(matrix + float(step) * direction, mode="reduced")
    except np.linalg.LinAlgError as error:
        raise LowRankError("QR retraction failed") from error
    diagonal = np.diag(R)
    if np.any(diagonal == 0.0):
        raise LowRankError("QR retraction encountered an exactly rank-deficient input")
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    Q = Q * signs[np.newaxis, :]
    error = float(np.max(np.abs(Q.T @ Q - np.eye(Q.shape[1]))))
    if error > STIEFEL_ORTHOGONALITY_TOLERANCE:
        raise LowRankError("QR retraction did not return orthonormal columns")
    return _readonly(Q, "QR-retracted Stiefel factor")


def _canonicalize_vector_sign(vector: np.ndarray) -> np.ndarray:
    result = np.array(vector, dtype=float, copy=True)
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0.0:
        result *= -1.0
    return result


def deterministic_symmetric_eigh(matrix: np.ndarray) -> DeterministicEigensystem:
    """Return the frozen backward-error-clustered initialization eigensystem."""

    source = np.asarray(matrix, dtype=float)
    expected = (core.m1.PANEL_SIZE, core.m1.PANEL_SIZE)
    if source.shape != expected or not np.all(np.isfinite(source)):
        raise LowRankError("deterministic eigensystem input is malformed or nonfinite")
    symmetric = 0.5 * (source + source.T)
    try:
        values, vectors = np.linalg.eigh(symmetric)
    except np.linalg.LinAlgError as error:
        raise LowRankError("deterministic symmetric eigensystem failed") from error
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(vectors)):
        raise LowRankError("deterministic symmetric eigensystem is nonfinite")

    if np.any(np.diff(values) < 0.0):
        raise LowRankError("symmetric eigensolver did not return increasing eigenvalues")

    spectral_norm = float(np.max(np.abs(values))) if values.size else 0.0
    rounding_guard = (
        EIGENPAIR_ROUNDING_GUARD_FACTOR
        * np.finfo(float).eps
        * max(1.0, spectral_norm)
    )
    raw_residual_matrix = symmetric @ vectors - vectors * values[np.newaxis, :]
    eigenpair_residuals = np.linalg.norm(raw_residual_matrix, axis=0)
    interval_radii = eigenpair_residuals + rounding_guard
    if not (
        math.isfinite(rounding_guard)
        and rounding_guard > 0.0
        and np.all(np.isfinite(eigenpair_residuals))
        and np.all(np.isfinite(interval_radii))
    ):
        raise LowRankError("eigensolver backward-error intervals are invalid")

    lower = values - interval_radii
    upper = values + interval_radii
    increasing_clusters: list[tuple[int, int]] = []
    start = 0
    cluster_upper = float(upper[0])
    for index in range(1, values.size):
        if float(lower[index]) <= cluster_upper:
            cluster_upper = max(cluster_upper, float(upper[index]))
            continue
        increasing_clusters.append((start, index))
        start = index
        cluster_upper = float(upper[index])
    increasing_clusters.append((start, values.size))

    cluster_records: list[tuple[float, int, np.ndarray]] = []
    for start, stop in increasing_clusters:
        cardinality = stop - start
        cluster_values = np.asarray(values[start:stop], dtype=float)
        cluster_mean = math.fsum(float(value) for value in cluster_values) / int(
            cardinality
        )
        if not math.isfinite(cluster_mean):
            raise LowRankError("numerical eigencluster mean is nonfinite")

        cluster_vectors = np.asarray(vectors[:, start:stop], dtype=float)
        raw_projector = cluster_vectors @ cluster_vectors.T
        projector = 0.5 * (raw_projector + raw_projector.T)
        if not np.all(np.isfinite(projector)):
            raise LowRankError("numerical eigencluster projector is nonfinite")

        if cardinality == 1:
            canonical_basis = _canonicalize_vector_sign(cluster_vectors[:, 0])[
                :, np.newaxis
            ]
        else:
            selected: list[np.ndarray] = []
            for coordinate in range(core.m1.PANEL_SIZE):
                residual = np.array(projector[:, coordinate], dtype=float, copy=True)
                for _ in range(2):
                    for accepted in selected:
                        residual -= float(np.dot(accepted, residual)) * accepted
                norm = float(np.linalg.norm(residual))
                if not math.isfinite(norm):
                    raise LowRankError(
                        "numerical eigencluster Gram-Schmidt residual is nonfinite"
                    )
                if norm <= CLUSTER_GRAM_SCHMIDT_RESIDUAL_THRESHOLD:
                    continue
                accepted = _canonicalize_vector_sign(residual / norm)
                selected.append(accepted)
                if len(selected) == cardinality:
                    break
            if len(selected) != cardinality:
                raise LowRankError(
                    "ordered coordinate Gram-Schmidt did not reach the exact "
                    "numerical eigencluster cardinality"
                )
            canonical_basis = np.column_stack(selected)

        orthogonality_error = float(
            np.max(
                np.abs(
                    canonical_basis.T @ canonical_basis
                    - np.eye(cardinality, dtype=float)
                )
            )
        )
        projector_error = float(
            np.max(np.abs(canonical_basis @ canonical_basis.T - projector))
        )
        if (
            not math.isfinite(orthogonality_error)
            or not math.isfinite(projector_error)
            or orthogonality_error > CLUSTER_CERTIFICATE_INFINITY_TOLERANCE
            or projector_error > CLUSTER_CERTIFICATE_INFINITY_TOLERANCE
        ):
            raise LowRankError(
                "numerical eigencluster failed its orthogonality/projector certificate"
            )

        common_value_bound = max(
            abs(float(value) - cluster_mean) for value in cluster_values
        ) + 4.0 * float(np.max(interval_radii[start:stop]))
        common_value_residuals = np.linalg.norm(
            symmetric @ canonical_basis - cluster_mean * canonical_basis,
            axis=0,
        )
        if (
            not math.isfinite(common_value_bound)
            or not np.all(np.isfinite(common_value_residuals))
            or np.any(common_value_residuals > common_value_bound)
        ):
            raise LowRankError(
                "numerical eigencluster failed its common-value residual certificate"
            )
        cluster_records.append((cluster_mean, start, canonical_basis))

    ordered_clusters = sorted(
        cluster_records,
        key=lambda record: (-abs(record[0]), -record[0], record[1]),
    )
    ordered_values = np.concatenate(
        [
            np.full(basis.shape[1], mean, dtype=float)
            for mean, _, basis in ordered_clusters
        ]
    )
    ordered_vectors = np.column_stack(
        [basis for _, _, basis in ordered_clusters]
    )
    cluster_sizes = tuple(int(basis.shape[1]) for _, _, basis in ordered_clusters)
    cluster_means = np.asarray(
        [mean for mean, _, _ in ordered_clusters], dtype=float
    )
    full_orthogonality_error = float(
        np.max(
            np.abs(
                ordered_vectors.T @ ordered_vectors
                - np.eye(values.size, dtype=float)
            )
        )
    )
    if (
        not math.isfinite(full_orthogonality_error)
        or full_orthogonality_error > CLUSTER_CERTIFICATE_INFINITY_TOLERANCE
    ):
        raise LowRankError(
            "clustered deterministic eigensystem failed its orthogonality certificate"
        )
    return DeterministicEigensystem(
        values=_readonly(ordered_values, "ordered clustered eigenvalues"),
        vectors=_readonly(ordered_vectors, "ordered clustered eigenvectors"),
        raw_values=_readonly(values, "raw increasing eigenvalues"),
        eigenpair_residuals=_readonly(
            eigenpair_residuals, "eigenpair backward errors"
        ),
        interval_radii=_readonly(interval_radii, "eigenpair interval radii"),
        rounding_guard=float(rounding_guard),
        cluster_sizes=cluster_sizes,
        cluster_means=_readonly(cluster_means, "ordered cluster means"),
    )


def core_smooth_lipschitz_upper_bound(
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
) -> float:
    """Return a deterministic global upper bound for the core smooth Hessian.

    For each cell, ``Cov(phi) <= E[phi phi.T]`` in Loewner order and the
    latter's spectral norm is bounded by ``max_S ||phi(S)||^2``.  Summing
    these bounds with the already-frozen equal-group weights gives a valid
    fold-local bound without using any realized outcome identity.
    """

    terms: list[float] = []
    for cell_index, cell in enumerate(prepared.cells):
        basis = candidates.basis(cell.k)
        if basis.count == 1:
            terms.append(0.0)
            continue
        maximum_squared_norm = 0.0
        for start in range(0, basis.count, core.m1.BATCH_MAXIMUM):
            stop = min(start + core.m1.BATCH_MAXIMUM, basis.count)
            feature = core.residual_pair_batch(
                prepared, cell_index, start, stop, candidates, geometry, pairs
            )
            squared = np.einsum("ij,ij->i", feature, feature, optimize=True)
            maximum_squared_norm = max(
                maximum_squared_norm, float(np.max(squared))
            )
        terms.append(float(prepared.weights[cell_index]) * maximum_squared_norm)
    bound = math.fsum(terms)
    if not math.isfinite(bound) or bound < 0.0:
        raise LowRankError("core smooth Lipschitz upper bound is invalid")
    return float(bound)


def _validated_synthetic_correction(linear_correction: np.ndarray) -> np.ndarray:
    correction = np.asarray(linear_correction, dtype=float)
    if (
        correction.shape != (core.PAIR_DIMENSION,)
        or not np.all(np.isfinite(correction))
    ):
        raise LowRankError(
            "synthetic qualification linear correction is malformed or nonfinite"
        )
    return _readonly(correction, "synthetic qualification linear correction")


def _fold_low_rank_scales_impl(
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    *,
    linear_correction: np.ndarray | None,
    smooth_lipschitz_upper_bound: float | None = None,
) -> LowRankFoldScales:
    """Shared implementation behind separated production/synthetic APIs."""

    if smooth_lipschitz_upper_bound is None:
        lipschitz = core_smooth_lipschitz_upper_bound(
            prepared, candidates, geometry, pairs
        )
    else:
        lipschitz = float(smooth_lipschitz_upper_bound)
        if not math.isfinite(lipschitz) or lipschitz < 0.0:
            raise LowRankError("supplied core smooth Lipschitz bound is invalid")
    null = core.evaluate_m2(
        np.zeros(core.PAIR_DIMENSION, dtype=float),
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        linear_correction=linear_correction,
    )
    projected_gradient = core._range_project_vector(
        null.gradient, prepared.transform
    )
    gradient_matrix = physical_gradient_matrix(projected_gradient)
    eigenvalues = np.linalg.eigvalsh(gradient_matrix)
    if not np.all(np.isfinite(eigenvalues)):
        raise LowRankError("null-gradient spectral norm is nonfinite")
    raw = float(np.max(np.abs(eigenvalues)))
    reference = max(LAMBDA_REFERENCE_FLOOR, raw)
    return LowRankFoldScales(
        lambda_reference=reference,
        raw_gradient_spectral_norm=raw,
        smooth_lipschitz_upper_bound=lipschitz,
        null_gradient=_readonly(projected_gradient, "projected null gradient"),
        null_gradient_matrix=gradient_matrix,
    )


def fold_low_rank_scales(
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    *,
    smooth_lipschitz_upper_bound: float | None = None,
) -> LowRankFoldScales:
    """Compute production fold-local scales on the observed objective."""

    return _fold_low_rank_scales_impl(
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        linear_correction=None,
        smooth_lipschitz_upper_bound=smooth_lipschitz_upper_bound,
    )


def fold_low_rank_scales_synthetic_qualification(
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    *,
    linear_correction: np.ndarray,
    smooth_lipschitz_upper_bound: float | None = None,
) -> LowRankFoldScales:
    """Compute fold-local scales for an explicit synthetic objective."""

    correction = _validated_synthetic_correction(linear_correction)
    return _fold_low_rank_scales_impl(
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        linear_correction=correction,
        smooth_lipschitz_upper_bound=smooth_lipschitz_upper_bound,
    )


def low_rank_lambda_reference(
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
) -> float:
    """Return the production ridge reference scale."""

    return fold_low_rank_scales(
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
    ).lambda_reference


def low_rank_lambda_reference_synthetic_qualification(
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    *,
    linear_correction: np.ndarray,
) -> float:
    """Return the ridge reference scale for an explicit synthetic objective."""

    return fold_low_rank_scales_synthetic_qualification(
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        linear_correction=linear_correction,
    ).lambda_reference


def _initialize_low_rank_impl(
    rank: int,
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    *,
    linear_correction: np.ndarray | None,
    fold_scales: LowRankFoldScales | None = None,
) -> LowRankInitialization:
    """Shared initialization behind separated production/synthetic APIs."""

    selected_rank = _validate_rank(rank)
    scales = fold_scales
    if scales is None:
        scales = _fold_low_rank_scales_impl(
            fixed_m1_theta,
            prepared,
            candidates,
            geometry,
            pairs,
            linear_correction=linear_correction,
        )
    eigensystem = deterministic_symmetric_eigh(scales.null_gradient_matrix)
    step = 1.0 / max(1.0, scales.smooth_lipschitz_upper_bound)
    U = eigensystem.vectors[:, :selected_rank]
    d = -step * eigensystem.values[:selected_rank]
    state = canonicalize_state(LowRankState(U, d))
    return LowRankInitialization(
        state=state,
        fold_scales=scales,
        negative_gradient_step=float(step),
        selected_gradient_eigenvalues=_readonly(
            eigensystem.values[:selected_rank], "selected gradient eigenvalues"
        ),
    )


def initialize_low_rank(
    rank: int,
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    *,
    fold_scales: LowRankFoldScales | None = None,
) -> LowRankInitialization:
    """Construct the sole production null-gradient initialization."""

    return _initialize_low_rank_impl(
        rank,
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        linear_correction=None,
        fold_scales=fold_scales,
    )


def initialize_low_rank_synthetic_qualification(
    rank: int,
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    *,
    linear_correction: np.ndarray,
    fold_scales: LowRankFoldScales | None = None,
) -> LowRankInitialization:
    """Construct the sole initialization for an explicit synthetic objective."""

    correction = _validated_synthetic_correction(linear_correction)
    return _initialize_low_rank_impl(
        rank,
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        linear_correction=correction,
        fold_scales=fold_scales,
    )


def _evaluate_low_rank_impl(
    state: LowRankState,
    specification: LowRankSpec,
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    *,
    need_predictions: bool = False,
    linear_correction: np.ndarray | None,
) -> LowRankEvaluation:
    """Shared evaluation behind separated production/synthetic APIs."""

    if state.rank != specification.rank:
        raise LowRankError("low-rank state and specification ranks disagree")
    physical = low_rank_matrix(state)
    beta = low_rank_effective_beta(state, prepared.transform)
    evaluation = core.evaluate_m2(
        beta,
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        need_predictions=need_predictions,
        linear_correction=linear_correction,
    )
    projected_gradient = core._range_project_vector(
        evaluation.gradient, prepared.transform
    )
    gradient_matrix = physical_gradient_matrix(projected_gradient)
    matrix_times_U = gradient_matrix @ state.U
    ambient_U = 2.0 * matrix_times_U * state.d[np.newaxis, :]
    gradient_U = stiefel_tangent_projection(state.U, ambient_U)
    gradient_d = np.einsum(
        "ij,ij->j", state.U, matrix_times_U, optimize=True
    ) + specification.strength * state.d
    penalty = 0.5 * float(np.dot(state.d, state.d))
    objective = float(evaluation.objective) + specification.strength * penalty
    gradient_infinity = max(
        float(np.max(np.abs(gradient_U))),
        float(np.max(np.abs(gradient_d))),
    )
    if not all(
        math.isfinite(value)
        for value in (objective, penalty, gradient_infinity)
    ) or not np.all(np.isfinite(gradient_d)):
        raise LowRankError("low-rank objective or gradient is nonfinite")
    return LowRankEvaluation(
        objective=objective,
        smooth_objective=float(evaluation.objective),
        penalty_value=penalty,
        beta=beta,
        physical_matrix=physical,
        gradient_U=gradient_U,
        gradient_d=_readonly(gradient_d, "low-rank d gradient"),
        gradient_infinity=gradient_infinity,
        core_evaluation=evaluation,
    )


def evaluate_low_rank(
    state: LowRankState,
    specification: LowRankSpec,
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    *,
    need_predictions: bool = False,
) -> LowRankEvaluation:
    """Evaluate the production penalized low-rank objective and gradients."""

    return _evaluate_low_rank_impl(
        state,
        specification,
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        need_predictions=need_predictions,
        linear_correction=None,
    )


def evaluate_low_rank_synthetic_qualification(
    state: LowRankState,
    specification: LowRankSpec,
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    *,
    linear_correction: np.ndarray,
    need_predictions: bool = False,
) -> LowRankEvaluation:
    """Evaluate an explicitly synthetic penalized low-rank objective."""

    correction = _validated_synthetic_correction(linear_correction)
    return _evaluate_low_rank_impl(
        state,
        specification,
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        need_predictions=need_predictions,
        linear_correction=correction,
    )


def low_rank_candidate_distribution(
    state: LowRankState,
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    cell_index: int,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
) -> core.ConditionalDistribution:
    """Return one exact fixed-K distribution for a low-rank state."""

    return core.candidate_distribution(
        low_rank_effective_beta(state, prepared.transform),
        fixed_m1_theta,
        prepared,
        cell_index,
        candidates,
        geometry,
        pairs,
    )


def compare_low_rank_probabilities(
    left: LowRankState,
    right: LowRankState,
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
) -> dict[str, float]:
    """Compare only identified probabilities, never factor representations."""

    return core.compare_conditional_probabilities(
        low_rank_effective_beta(left, prepared.transform),
        low_rank_effective_beta(right, prepared.transform),
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
    )


def _optimizer_result(
    state: LowRankState,
    evaluation: LowRankEvaluation,
    iteration: int,
    relative_change: float,
    *,
    specification: LowRankSpec,
    objective_binding: _ObjectiveBinding,
    objective_mode: str,
    initialization: str,
    line_trials: int,
    fixed_point_rechecks: int = 0,
    _seal: object | None = None,
) -> LowRankOptimizerResult:
    if _seal is not _LOW_RANK_RESULT_SEAL:
        raise LowRankError("low-rank results must be issued by the optimizer")
    result = LowRankOptimizerResult(
        state=state,
        beta=evaluation.beta,
        objective=evaluation.objective,
        smooth_objective=evaluation.smooth_objective,
        penalty_value=evaluation.penalty_value,
        iterations=iteration,
        gradient_infinity=evaluation.gradient_infinity,
        relative_objective_change=relative_change,
        status="pass",
        telemetry={
            "objective_mode": objective_mode,
            "rank": specification.rank,
            "rho": float(specification.rho),
            "lambda_reference": float(specification.lambda_reference),
            "initialization": initialization,
            "Armijo_initial_step": ARMIJO_INITIAL_STEP,
            "Armijo_shrink_factor": ARMIJO_SHRINK_FACTOR,
            "Armijo_c1": ARMIJO_C1,
            "maximum_line_trials": ARMIJO_MAXIMUM_LINE_TRIALS,
            "total_line_trials": line_trials,
            "fixed_point_recheck": fixed_point_rechecks > 0,
            "fixed_point_rechecks": fixed_point_rechecks,
            "QR_retraction": True,
            "embedded_Euclidean_Stiefel_metric": True,
            "global_optimum_claimed": False,
            "all_finite": True,
        },
    )
    _register_low_rank_result_identity(
        result,
        _LowRankResultBinding(
            objective=objective_binding,
            objective_mode=objective_mode,
            result_signature=_low_rank_result_integrity_signature(result),
        ),
    )
    return result


def _fit_low_rank_impl(
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    specification: LowRankSpec,
    *,
    linear_correction: np.ndarray | None,
    warm_start: LowRankOptimizerResult | None = None,
    maximum_iterations: int = LOW_RANK_MAXIMUM_ITERATIONS,
) -> LowRankOptimizerResult:
    """Shared deterministic solver behind sealed production/synthetic APIs.

    Failure to find an Armijo step or simultaneous convergence is an invalid
    fit and raises ``LowRankError``.  There is no alternate initialization,
    relaxed tolerance, or multistart path.
    """

    if (
        type(maximum_iterations) is not int
        or maximum_iterations <= 0
        or maximum_iterations > LOW_RANK_MAXIMUM_ITERATIONS
    ):
        raise LowRankError(
            "low-rank maximum iterations must be an integer in 1..20000"
        )
    objective_mode = (
        "production" if linear_correction is None else "synthetic_qualification"
    )
    correction = (
        None
        if linear_correction is None
        else _validated_synthetic_correction(linear_correction)
    )
    registered_warm: _LowRankResultBinding | None = None
    if warm_start is not None:
        _require_full_development_prepared(prepared)
        registered_warm = _require_registered_low_rank_result(
            warm_start, objective_mode
        )
        _require_same_objective(
            registered_warm.objective,
            fixed_m1_theta,
            prepared,
            candidates,
            geometry,
            pairs,
            _low_rank_specification_signature(specification),
            correction,
        )
        if (
            warm_start.status != "pass"
            or warm_start.telemetry.get("initialization")
            != "deterministic_null_gradient_eigentree"
            or warm_start.gradient_infinity
            > LOW_RANK_GRADIENT_INFINITY_TOLERANCE
            or warm_start.relative_objective_change
            > LOW_RANK_RELATIVE_OBJECTIVE_TOLERANCE
        ):
            raise LowRankError(
                "low-rank warm refit requires an exactly registered converged result"
            )

    if correction is None:
        computed_scales = fold_low_rank_scales(
            fixed_m1_theta,
            prepared,
            candidates,
            geometry,
            pairs,
        )

        def initialize() -> LowRankInitialization:
            return initialize_low_rank(
                specification.rank,
                fixed_m1_theta,
                prepared,
                candidates,
                geometry,
                pairs,
                fold_scales=computed_scales,
            )

        def evaluate(state: LowRankState) -> LowRankEvaluation:
            return evaluate_low_rank(
                state,
                specification,
                fixed_m1_theta,
                prepared,
                candidates,
                geometry,
                pairs,
            )

    else:
        computed_scales = fold_low_rank_scales_synthetic_qualification(
            fixed_m1_theta,
            prepared,
            candidates,
            geometry,
            pairs,
            linear_correction=correction,
        )

        def initialize() -> LowRankInitialization:
            return initialize_low_rank_synthetic_qualification(
                specification.rank,
                fixed_m1_theta,
                prepared,
                candidates,
                geometry,
                pairs,
                linear_correction=correction,
                fold_scales=computed_scales,
            )

        def evaluate(state: LowRankState) -> LowRankEvaluation:
            return evaluate_low_rank_synthetic_qualification(
                state,
                specification,
                fixed_m1_theta,
                prepared,
                candidates,
                geometry,
                pairs,
                linear_correction=correction,
            )

    # Recompute both fold-local scales even if a caller computed them earlier,
    # and require bit-exact agreement with the trial specification.  An
    # ordinary fit has the sole deterministic null-gradient initialization;
    # the repeat path below accepts only an exact registered converged result
    # bound to this same objective.
    if computed_scales.lambda_reference != float(specification.lambda_reference):
        raise LowRankError(
            "low-rank specification does not use the recomputed fold lambda reference"
        )
    objective_binding = _make_objective_binding(
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        _low_rank_specification_signature(specification),
        correction,
    )
    if warm_start is None:
        initialization = initialize()
        state = initialization.state
        initialization_label = "deterministic_null_gradient_eigentree"
        current = evaluate(state)
    else:
        assert registered_warm is not None
        state = LowRankState(warm_start.state.U, warm_start.state.d)
        initialization_label = "exact_registered_converged_low_rank_result"
        current = evaluate(state)
        if (
            not np.array_equal(current.beta, warm_start.beta)
            or current.objective != float(warm_start.objective)
            or current.smooth_objective != float(warm_start.smooth_objective)
            or current.penalty_value != float(warm_start.penalty_value)
            or current.gradient_infinity != float(warm_start.gradient_infinity)
        ):
            raise LowRankError(
                "registered low-rank result does not reproduce on its bound objective"
            )
    total_line_trials = 0
    fixed_point_rechecks = 0

    def repeat_exact_stationary_point(
        stationary_state: LowRankState,
        stationary_evaluation: LowRankEvaluation,
    ) -> tuple[LowRankEvaluation, float]:
        """Evaluate an exact-zero fixed point again without an Armijo trial."""

        repeated = evaluate(stationary_state)
        relative_change = abs(
            stationary_evaluation.objective - repeated.objective
        ) / max(
            1.0,
            abs(stationary_evaluation.objective),
            abs(repeated.objective),
        )
        if not math.isfinite(relative_change):
            raise LowRankError(
                "low-rank fixed-point relative objective change is nonfinite"
            )
        if (
            repeated.gradient_infinity != 0.0
            or relative_change > LOW_RANK_RELATIVE_OBJECTIVE_TOLERANCE
        ):
            raise LowRankError(
                "low-rank exact-zero fixed point did not repeat within the frozen "
                "objective tolerance"
            )
        return repeated, relative_change

    if current.gradient_infinity == 0.0:
        current, fixed_point_relative_change = repeat_exact_stationary_point(
            state, current
        )
        fixed_point_rechecks += 1
        return _optimizer_result(
            state,
            current,
            0,
            fixed_point_relative_change,
            specification=specification,
            objective_binding=objective_binding,
            objective_mode=objective_mode,
            initialization=initialization_label,
            line_trials=total_line_trials,
            fixed_point_rechecks=fixed_point_rechecks,
            _seal=_LOW_RANK_RESULT_SEAL,
        )

    last_relative_change = math.inf
    for iteration in range(1, maximum_iterations + 1):
        direction_U = -np.asarray(current.gradient_U)
        direction_d = -np.asarray(current.gradient_d)
        directional_derivative = -(
            float(np.sum(current.gradient_U * current.gradient_U, dtype=float))
            + float(np.dot(current.gradient_d, current.gradient_d))
        )
        if not math.isfinite(directional_derivative) or directional_derivative >= 0.0:
            raise LowRankError("low-rank descent direction is not strictly descending")

        accepted_state: LowRankState | None = None
        accepted_evaluation: LowRankEvaluation | None = None
        step = ARMIJO_INITIAL_STEP
        for _ in range(ARMIJO_MAXIMUM_LINE_TRIALS):
            total_line_trials += 1
            trial_U = qr_retraction(state.U, direction_U, step)
            trial_d = state.d + step * direction_d
            trial_state = canonicalize_state(LowRankState(trial_U, trial_d))
            trial = evaluate(trial_state)
            armijo_bound = (
                current.objective
                + ARMIJO_C1 * step * directional_derivative
            )
            if trial.objective <= armijo_bound:
                accepted_state = trial_state
                accepted_evaluation = trial
                break
            step *= ARMIJO_SHRINK_FACTOR
        if accepted_state is None or accepted_evaluation is None:
            raise LowRankError(
                "low-rank Armijo search failed within the frozen line-trial cap"
            )

        last_relative_change = abs(
            current.objective - accepted_evaluation.objective
        ) / max(
            1.0,
            abs(current.objective),
            abs(accepted_evaluation.objective),
        )
        if not math.isfinite(last_relative_change):
            raise LowRankError("low-rank relative objective change is nonfinite")
        state = accepted_state
        current = accepted_evaluation
        if (
            current.gradient_infinity <= LOW_RANK_GRADIENT_INFINITY_TOLERANCE
            and last_relative_change <= LOW_RANK_RELATIVE_OBJECTIVE_TOLERANCE
        ):
            return _optimizer_result(
                state,
                current,
                iteration,
                last_relative_change,
                specification=specification,
                objective_binding=objective_binding,
                objective_mode=objective_mode,
                initialization=initialization_label,
                line_trials=total_line_trials,
                fixed_point_rechecks=fixed_point_rechecks,
                _seal=_LOW_RANK_RESULT_SEAL,
            )
        if current.gradient_infinity == 0.0:
            current, fixed_point_relative_change = repeat_exact_stationary_point(
                state, current
            )
            fixed_point_rechecks += 1
            return _optimizer_result(
                state,
                current,
                iteration,
                fixed_point_relative_change,
                specification=specification,
                objective_binding=objective_binding,
                objective_mode=objective_mode,
                initialization=initialization_label,
                line_trials=total_line_trials,
                fixed_point_rechecks=fixed_point_rechecks,
                _seal=_LOW_RANK_RESULT_SEAL,
            )

    raise LowRankError(
        "low-rank optimizer reached its iteration cap without both gradient "
        "and relative-objective convergence; "
        f"rank={specification.rank} rho={specification.rho:.17g} "
        f"iterations={maximum_iterations} "
        f"gradient_U_infinity={float(np.max(np.abs(current.gradient_U))):.17g} "
        f"gradient_d_infinity={float(np.max(np.abs(current.gradient_d))):.17g} "
        f"relative_objective_change={last_relative_change:.17g} "
        f"last_accepted_step={step:.17g} "
        f"total_line_trials={total_line_trials}"
    )


def fit_low_rank(
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    specification: LowRankSpec,
    *,
    maximum_iterations: int = LOW_RANK_MAXIMUM_ITERATIONS,
) -> LowRankOptimizerResult:
    """Fit the production low-rank family on the observed exact objective.

    This API intentionally has no linear-correction argument.  Synthetic
    population qualification must use the explicitly named qualification
    entry point below, so a production caller cannot silently alter the
    observed objective.
    """

    return _fit_low_rank_impl(
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        specification,
        linear_correction=None,
        maximum_iterations=maximum_iterations,
    )


def fit_low_rank_synthetic_qualification(
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    specification: LowRankSpec,
    *,
    linear_correction: np.ndarray,
    maximum_iterations: int = LOW_RANK_MAXIMUM_ITERATIONS,
) -> LowRankOptimizerResult:
    """Fit a target-free synthetic population qualification objective only."""

    correction = _validated_synthetic_correction(linear_correction)
    return _fit_low_rank_impl(
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        specification,
        linear_correction=correction,
        maximum_iterations=maximum_iterations,
    )


def refit_low_rank_from_converged(
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    specification: LowRankSpec,
    converged_result: LowRankOptimizerResult,
    *,
    maximum_iterations: int = LOW_RANK_MAXIMUM_ITERATIONS,
) -> LowRankOptimizerResult:
    """Repeat a production fit from its exact registered converged solution."""

    return _fit_low_rank_impl(
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        specification,
        linear_correction=None,
        warm_start=converged_result,
        maximum_iterations=maximum_iterations,
    )


def refit_low_rank_from_converged_synthetic_qualification(
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    specification: LowRankSpec,
    converged_result: LowRankOptimizerResult,
    *,
    linear_correction: np.ndarray,
    maximum_iterations: int = LOW_RANK_MAXIMUM_ITERATIONS,
) -> LowRankOptimizerResult:
    """Repeat an explicit synthetic fit from its registered converged result."""

    correction = _validated_synthetic_correction(linear_correction)
    return _fit_low_rank_impl(
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        specification,
        linear_correction=correction,
        warm_start=converged_result,
        maximum_iterations=maximum_iterations,
    )


def _validated_selected_low_rank_anchor(
    selected_anchor: LowRankOptimizerResult,
    expected_mode: str,
    expected_lambda_reference: float,
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    *,
    linear_correction: np.ndarray | None,
) -> LowRankState:
    """Bind the required selected anchor result to this fold and objective."""

    registered = _require_registered_low_rank_result(selected_anchor, expected_mode)
    telemetry = selected_anchor.telemetry
    if not isinstance(telemetry, Mapping):
        raise LowRankError("selected low-rank anchor telemetry is malformed")
    if telemetry.get("objective_mode") != expected_mode:
        raise LowRankError("selected low-rank anchor objective mode is incompatible")
    try:
        rank = int(telemetry["rank"])
        rho = float(telemetry["rho"])
        lambda_reference = float(telemetry["lambda_reference"])
    except (KeyError, TypeError, ValueError) as error:
        raise LowRankError("selected low-rank anchor provenance is incomplete") from error
    if (
        rank != selected_anchor.state.rank
        or lambda_reference != float(expected_lambda_reference)
        or selected_anchor.status != "pass"
        or not math.isfinite(float(selected_anchor.objective))
        or not math.isfinite(float(selected_anchor.gradient_infinity))
        or not math.isfinite(float(selected_anchor.relative_objective_change))
        or selected_anchor.gradient_infinity
        > LOW_RANK_GRADIENT_INFINITY_TOLERANCE
        or selected_anchor.relative_objective_change
        > LOW_RANK_RELATIVE_OBJECTIVE_TOLERANCE
    ):
        raise LowRankError("selected low-rank anchor is invalid or unconverged")
    anchor_specification = LowRankSpec(rank, rho, lambda_reference)
    _require_same_objective(
        registered.objective,
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        _low_rank_specification_signature(anchor_specification),
        linear_correction,
    )
    if linear_correction is None:
        repeated = evaluate_low_rank(
            selected_anchor.state,
            anchor_specification,
            fixed_m1_theta,
            prepared,
            candidates,
            geometry,
            pairs,
        )
    else:
        repeated = evaluate_low_rank_synthetic_qualification(
            selected_anchor.state,
            anchor_specification,
            fixed_m1_theta,
            prepared,
            candidates,
            geometry,
            pairs,
            linear_correction=linear_correction,
        )
    beta_error = float(np.max(np.abs(repeated.beta - selected_anchor.beta)))
    objective_error = abs(repeated.objective - float(selected_anchor.objective))
    if (
        beta_error > 1.0e-12
        or objective_error > 1.0e-12
        or repeated.gradient_infinity > LOW_RANK_GRADIENT_INFINITY_TOLERANCE
    ):
        raise LowRankError(
            "selected low-rank anchor does not reproduce on the current fold objective"
        )
    return LowRankState(selected_anchor.state.U, selected_anchor.state.d)


def _hybrid_optimizer_result(
    state: HybridState,
    evaluation: HybridEvaluation,
    sweep: int,
    relative_change: float,
    sparse_lipschitz: float,
    specification: HybridSpec,
    objective_binding: _ObjectiveBinding,
    objective_mode: str,
    selected_low_rank_anchor: LowRankOptimizerResult,
    production_receipt: object | None,
    *,
    initialization: str,
    initial_sparse_lipschitz: float,
    projected_embedding_spectral_norm: float,
    low_rank_line_trials: int,
    sparse_line_trials: int,
    sparse_backtracking_steps: int,
) -> HybridOptimizerResult:
    result = HybridOptimizerResult(
        state=state,
        beta=evaluation.beta,
        objective=evaluation.objective,
        smooth_objective=evaluation.smooth_objective,
        low_rank_penalty_value=evaluation.low_rank_penalty_value,
        sparse_penalty_value=evaluation.sparse_penalty_value,
        sweeps=sweep,
        joint_stationarity_infinity=evaluation.joint_stationarity_infinity,
        relative_objective_change=relative_change,
        sparse_lipschitz=float(sparse_lipschitz),
        status="pass",
        telemetry={
            "objective_mode": objective_mode,
            "initialization": initialization,
            "rank": specification.rank,
            "rho_low": float(specification.rho_low),
            "rho_sparse": float(specification.rho_sparse),
            "lambda_reference": float(specification.lambda_reference),
            "core_lasso_lambda_maximum": float(
                specification.core_lasso_lambda_maximum
            ),
            "named_edges": specification.named_edges,
            "block_order": ("low_rank", "sparse"),
            "initial_sparse_lipschitz": float(initial_sparse_lipschitz),
            "final_sparse_lipschitz": float(sparse_lipschitz),
            "projected_embedding_spectral_norm": float(
                projected_embedding_spectral_norm
            ),
            "low_rank_line_trials": low_rank_line_trials,
            "sparse_line_trials": sparse_line_trials,
            "sparse_backtracking_steps": sparse_backtracking_steps,
            "post_sweep_evaluations": sweep,
            "global_optimum_claimed": False,
            "all_finite": True,
        },
    )
    receipt_reference = (
        None
        if production_receipt is None
        else _weak_reference(production_receipt, "production hybrid receipt")
    )
    _register_hybrid_result_identity(
        result,
        _HybridResultBinding(
            objective=objective_binding,
            objective_mode=objective_mode,
            selected_anchor_reference=_weak_reference(
                selected_low_rank_anchor, "selected low-rank anchor"
            ),
            selected_anchor_signature=_low_rank_result_integrity_signature(
                selected_low_rank_anchor
            ),
            production_receipt_reference=receipt_reference,
            result_signature=_hybrid_result_integrity_signature(result),
        ),
    )
    return result


def _fit_sparse_plus_low_rank_impl(
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    specification: HybridSpec,
    selected_low_rank_anchor: LowRankOptimizerResult,
    *,
    linear_correction: np.ndarray | None,
    production_receipt: object | None = None,
    warm_start: HybridOptimizerResult | None = None,
    maximum_sweeps: int = HYBRID_MAXIMUM_SWEEPS,
) -> HybridOptimizerResult:
    """Run the frozen low-rank-then-sparse deterministic block solver."""

    if (
        type(maximum_sweeps) is not int
        or maximum_sweeps <= 0
        or maximum_sweeps > HYBRID_MAXIMUM_SWEEPS
    ):
        raise LowRankError("hybrid maximum sweeps must be an integer in 1..20000")
    objective_mode = (
        "production" if linear_correction is None else "synthetic_qualification"
    )
    correction = (
        None
        if linear_correction is None
        else _validated_synthetic_correction(linear_correction)
    )
    registered_warm: _HybridResultBinding | None = None
    if warm_start is not None:
        _require_full_development_prepared(prepared)
        registered_warm = _require_registered_hybrid_result(
            warm_start, objective_mode
        )
        _require_same_objective(
            registered_warm.objective,
            fixed_m1_theta,
            prepared,
            candidates,
            geometry,
            pairs,
            _hybrid_specification_signature(specification),
            correction,
        )
        if (
            registered_warm.selected_anchor_reference()
            is not selected_low_rank_anchor
            or registered_warm.selected_anchor_signature
            != _low_rank_result_integrity_signature(selected_low_rank_anchor)
        ):
            raise LowRankError(
                "converged hybrid result belongs to a different selected low-rank anchor"
            )
        registered_receipt = registered_warm.production_receipt_reference
        if objective_mode == "production":
            if (
                production_receipt is None
                or registered_receipt is None
                or registered_receipt() is not production_receipt
            ):
                raise LowRankError(
                    "converged hybrid result belongs to a different production receipt"
                )
        elif production_receipt is not None or registered_receipt is not None:
            raise LowRankError(
                "synthetic hybrid warm refit cannot use production provenance"
            )
        if (
            warm_start.status != "pass"
            or warm_start.telemetry.get("initialization")
            != "selected_low_rank_anchor_plus_exact_zero_sparse"
            or warm_start.joint_stationarity_infinity
            > HYBRID_JOINT_STATIONARITY_TOLERANCE
            or warm_start.relative_objective_change
            > HYBRID_RELATIVE_OBJECTIVE_TOLERANCE
            or not math.isfinite(float(warm_start.sparse_lipschitz))
            or float(warm_start.sparse_lipschitz) <= 0.0
        ):
            raise LowRankError(
                "hybrid warm refit requires an exactly registered converged result"
            )

    if correction is None:
        computed_scales = fold_low_rank_scales(
            fixed_m1_theta,
            prepared,
            candidates,
            geometry,
            pairs,
        )
    else:
        computed_scales = fold_low_rank_scales_synthetic_qualification(
            fixed_m1_theta,
            prepared,
            candidates,
            geometry,
            pairs,
            linear_correction=correction,
        )
    if computed_scales.lambda_reference != float(specification.lambda_reference):
        raise LowRankError(
            "hybrid specification does not use the recomputed fold lambda reference"
        )
    lambda_maximum = core.fold_lambda_maximum(
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        "lasso",
        linear_correction=correction,
    )
    if lambda_maximum.value != float(specification.core_lasso_lambda_maximum):
        raise LowRankError(
            "hybrid specification does not use the recomputed core lasso lambda maximum"
        )

    objective_binding = _make_objective_binding(
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        _hybrid_specification_signature(specification),
        correction,
    )

    anchor_state = _validated_selected_low_rank_anchor(
        selected_low_rank_anchor,
        objective_mode,
        computed_scales.lambda_reference,
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        linear_correction=correction,
    )
    if anchor_state.rank != specification.rank:
        raise LowRankError("selected low-rank anchor rank and hybrid rank disagree")
    default_sparse_lipschitz, projected_embedding_norm = _initial_sparse_lipschitz(
        specification,
        prepared.transform,
        computed_scales.smooth_lipschitz_upper_bound,
    )
    if warm_start is None:
        state = HybridState(
            anchor_state,
            np.zeros(len(specification.named_edges), dtype=float),
        )
        sparse_lipschitz = default_sparse_lipschitz
        initialization_label = "selected_low_rank_anchor_plus_exact_zero_sparse"
    else:
        assert registered_warm is not None
        state = HybridState(
            LowRankState(
                warm_start.state.low_rank.U,
                warm_start.state.low_rank.d,
            ),
            warm_start.state.sparse_coefficients,
        )
        sparse_lipschitz = float(warm_start.sparse_lipschitz)
        initialization_label = "exact_registered_converged_hybrid_result"
    initial_sparse_lipschitz = sparse_lipschitz
    current = _evaluate_hybrid_impl(
        state,
        specification,
        sparse_lipschitz,
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        need_predictions=False,
        linear_correction=correction,
    )
    if warm_start is not None and (
        not np.array_equal(current.beta, warm_start.beta)
        or current.objective != float(warm_start.objective)
        or current.smooth_objective != float(warm_start.smooth_objective)
        or current.low_rank_penalty_value
        != float(warm_start.low_rank_penalty_value)
        or current.sparse_penalty_value
        != float(warm_start.sparse_penalty_value)
        or current.joint_stationarity_infinity
        != float(warm_start.joint_stationarity_infinity)
    ):
        raise LowRankError(
            "registered hybrid result does not reproduce on its bound objective"
        )
    low_rank_line_trials = 0
    sparse_line_trials = 0
    sparse_backtracking_steps = 0

    for sweep in range(1, maximum_sweeps + 1):
        previous_sweep_objective = current.objective

        low_gradient_infinity = max(
            float(np.max(np.abs(current.gradient_U))),
            float(np.max(np.abs(current.gradient_d))),
        )
        if low_gradient_infinity > 0.0:
            direction_U = -np.asarray(current.gradient_U)
            direction_d = -np.asarray(current.gradient_d)
            directional_derivative = -(
                float(np.sum(current.gradient_U * current.gradient_U, dtype=float))
                + float(np.dot(current.gradient_d, current.gradient_d))
            )
            if (
                not math.isfinite(directional_derivative)
                or directional_derivative >= 0.0
            ):
                raise LowRankError(
                    "hybrid low-rank block direction is not strictly descending"
                )
            accepted_low_state: HybridState | None = None
            accepted_low_evaluation: HybridEvaluation | None = None
            step = ARMIJO_INITIAL_STEP
            for _ in range(ARMIJO_MAXIMUM_LINE_TRIALS):
                low_rank_line_trials += 1
                trial_U = qr_retraction(state.low_rank.U, direction_U, step)
                trial_d = state.low_rank.d + step * direction_d
                trial_low_rank = canonicalize_state(
                    LowRankState(trial_U, trial_d)
                )
                trial_state = HybridState(
                    trial_low_rank, state.sparse_coefficients
                )
                trial = _evaluate_hybrid_impl(
                    trial_state,
                    specification,
                    sparse_lipschitz,
                    fixed_m1_theta,
                    prepared,
                    candidates,
                    geometry,
                    pairs,
                    need_predictions=False,
                    linear_correction=correction,
                )
                armijo_bound = (
                    current.objective
                    + ARMIJO_C1 * step * directional_derivative
                )
                if trial.objective <= armijo_bound:
                    accepted_low_state = trial_state
                    accepted_low_evaluation = trial
                    break
                step *= ARMIJO_SHRINK_FACTOR
            if accepted_low_state is None or accepted_low_evaluation is None:
                raise LowRankError(
                    "hybrid low-rank Armijo search exhausted its frozen line cap"
                )
            state = accepted_low_state
            current = accepted_low_evaluation

        accepted_sparse_state: HybridState | None = None
        accepted_sparse_evaluation: HybridEvaluation | None = None
        candidate_lipschitz = sparse_lipschitz
        for _ in range(SPARSE_MAXIMUM_LINE_TRIALS):
            sparse_line_trials += 1
            proposal = _soft_threshold(
                state.sparse_coefficients
                - current.gradient_a / candidate_lipschitz,
                specification.sparse_strength / candidate_lipschitz,
            )
            displacement = proposal - state.sparse_coefficients
            trial_state = HybridState(state.low_rank, proposal)
            trial = _evaluate_hybrid_impl(
                trial_state,
                specification,
                candidate_lipschitz,
                fixed_m1_theta,
                prepared,
                candidates,
                geometry,
                pairs,
                need_predictions=False,
                linear_correction=correction,
            )
            majorizer = (
                current.smooth_objective
                + float(np.dot(current.gradient_a, displacement))
                + 0.5
                * candidate_lipschitz
                * float(np.dot(displacement, displacement))
            )
            if not math.isfinite(majorizer):
                raise LowRankError("hybrid sparse majorizer is nonfinite")
            smooth_tolerance = core._objective_roundoff_tolerance(majorizer)
            monotone_tolerance = core._objective_roundoff_tolerance(
                current.objective
            )
            if (
                trial.smooth_objective <= majorizer + smooth_tolerance
                and trial.objective <= current.objective + monotone_tolerance
            ):
                accepted_sparse_state = trial_state
                accepted_sparse_evaluation = trial
                sparse_lipschitz = candidate_lipschitz
                break
            candidate_lipschitz *= SPARSE_BACKTRACKING_MULTIPLIER
            sparse_backtracking_steps += 1
            if not math.isfinite(candidate_lipschitz):
                raise LowRankError(
                    "hybrid sparse Lipschitz state became nonfinite"
                )
        if accepted_sparse_state is None or accepted_sparse_evaluation is None:
            raise LowRankError(
                "hybrid sparse backtracking exhausted its frozen line cap"
            )
        state = accepted_sparse_state

        # The convergence certificate is always evaluated after both blocks at
        # the same U, d, a, and carried sparse Lipschitz state.
        current = _evaluate_hybrid_impl(
            state,
            specification,
            sparse_lipschitz,
            fixed_m1_theta,
            prepared,
            candidates,
            geometry,
            pairs,
            need_predictions=False,
            linear_correction=correction,
        )
        relative_change = abs(
            previous_sweep_objective - current.objective
        ) / max(
            1.0,
            abs(previous_sweep_objective),
            abs(current.objective),
        )
        if not math.isfinite(relative_change):
            raise LowRankError("hybrid full-sweep objective change is nonfinite")
        if (
            current.joint_stationarity_infinity
            <= HYBRID_JOINT_STATIONARITY_TOLERANCE
            and relative_change <= HYBRID_RELATIVE_OBJECTIVE_TOLERANCE
        ):
            return _hybrid_optimizer_result(
                state,
                current,
                sweep,
                relative_change,
                sparse_lipschitz,
                specification,
                objective_binding,
                objective_mode,
                selected_low_rank_anchor,
                production_receipt,
                initialization=initialization_label,
                initial_sparse_lipschitz=initial_sparse_lipschitz,
                projected_embedding_spectral_norm=projected_embedding_norm,
                low_rank_line_trials=low_rank_line_trials,
                sparse_line_trials=sparse_line_trials,
                sparse_backtracking_steps=sparse_backtracking_steps,
            )

    raise LowRankError(
        "hybrid optimizer reached its sweep cap without joint stationarity and "
        "full-sweep objective convergence"
    )


def _terminal_receipt_minimal(
    receipt: object,
) -> _TerminalFullRefitHybridBinding:
    if (
        type(receipt) is not _TerminalFullRefitHybridReceipt
        or receipt._seal is not _TERMINAL_FULL_REFIT_HYBRID_RECEIPT_SEAL
    ):
        raise LowRankError("terminal full-refit hybrid receipt is invalid")
    binding = _lookup_terminal_full_refit_receipt_identity(receipt)
    if type(binding) is not _TerminalFullRefitHybridBinding:
        raise LowRankError("terminal full-refit hybrid binding is malformed")
    return binding


def _terminal_stage_context(
    binding: _TerminalFullRefitHybridBinding,
) -> tuple[Any, Any, Any, Any]:
    owner = binding.owner_reference()
    terminal = binding.terminal_reference()
    authority = binding.procedure_authority_reference()
    full_objective = binding.full_objective_reference()
    if any(value is None for value in (owner, terminal, authority, full_objective)):
        raise LowRankError("terminal full-refit authority objects are unavailable")
    return owner, terminal, authority, full_objective


def _terminal_objective(
    binding: _TerminalFullRefitHybridBinding,
) -> tuple[np.ndarray, core.PreparedM2, Any, Any, Any, HybridSpec]:
    theta = binding.objective.fixed_m1_reference()
    prepared = binding.objective.prepared_reference()
    candidates = binding.objective.candidates_reference()
    geometry = binding.objective.geometry_reference()
    pairs = binding.objective.pairs_reference()
    specification = binding.target_specification
    if any(
        value is None
        for value in (theta, prepared, candidates, geometry, pairs, specification)
    ) or binding.target_specification_reference() is not specification:
        raise LowRankError("terminal full-refit objective is unavailable")
    return theta, prepared, candidates, geometry, pairs, specification


def fit_terminal_full_development_hybrid_precursor(
    receipt: object,
    *,
    maximum_iterations: int = LOW_RANK_MAXIMUM_ITERATIONS,
) -> LowRankOptimizerResult:
    """Consume the receipt's one prescribed deterministic precursor fit."""

    binding = _terminal_receipt_minimal(receipt)
    owner, terminal, authority, full_objective = _terminal_stage_context(binding)
    began = False
    try:
        _observed_session_request(
            "_begin_selected_refit_fit_stage",
            owner,
            terminal,
            authority,
            full_objective,
            "precursor",
            hybrid_receipt=receipt,
        )
        began = True
        with binding.lock:
            if binding.phase != "issued_for_precursor":
                raise LowRankError("terminal hybrid precursor was already consumed")
            binding.phase = "precursor_in_flight"
        binding = _require_terminal_full_development_hybrid_receipt(receipt)
        theta, prepared, candidates, geometry, pairs, _ = _terminal_objective(binding)
        result = _fit_low_rank_impl(
            theta,
            prepared,
            candidates,
            geometry,
            pairs,
            binding.precursor_specification,
            linear_correction=None,
            maximum_iterations=maximum_iterations,
        )
        with binding.lock:
            if binding.phase != "precursor_in_flight":
                raise LowRankError("terminal hybrid precursor chronology changed")
            binding.selected_anchor_reference = _weak_reference(
                result, "terminal full-refit precursor result"
            )
            binding.selected_anchor_signature = _low_rank_result_integrity_signature(
                result
            )
            binding.phase = "precursor_ready"
        _observed_session_request(
            "_finish_selected_refit_fit_stage", authority, "precursor", result
        )
        return result
    except BaseException:
        with binding.lock:
            binding.phase = "burned"
        if began:
            try:
                _observed_session_request(
                    "_finish_selected_refit_fit_stage",
                    authority,
                    "precursor",
                    None,
                )
            except BaseException:
                _observed_session_request(
                    "_burn_selected_refit_procedure", authority
                )
        raise


def fit_terminal_full_development_hybrid_cold(
    receipt: object,
    *,
    maximum_sweeps: int = HYBRID_MAXIMUM_SWEEPS,
) -> HybridOptimizerResult:
    """Run the selected hybrid once from precursor plus exact-zero sparse."""

    binding = _terminal_receipt_minimal(receipt)
    owner, terminal, authority, full_objective = _terminal_stage_context(binding)
    anchor = (
        None
        if binding.selected_anchor_reference is None
        else binding.selected_anchor_reference()
    )
    began = False
    try:
        _observed_session_request(
            "_begin_selected_refit_fit_stage",
            owner,
            terminal,
            authority,
            full_objective,
            "cold",
            hybrid_receipt=receipt,
            prior_result=anchor,
        )
        began = True
        with binding.lock:
            if binding.phase != "precursor_ready" or anchor is None:
                raise LowRankError("terminal hybrid cold fit lacks its precursor")
            binding.phase = "cold_in_flight"
        binding = _require_terminal_full_development_hybrid_receipt(receipt)
        theta, prepared, candidates, geometry, pairs, specification = (
            _terminal_objective(binding)
        )
        assert anchor is not None
        result = _fit_sparse_plus_low_rank_impl(
            theta,
            prepared,
            candidates,
            geometry,
            pairs,
            specification,
            anchor,
            linear_correction=None,
            production_receipt=receipt,
            maximum_sweeps=maximum_sweeps,
        )
        with binding.lock:
            if binding.phase != "cold_in_flight":
                raise LowRankError(
                    "terminal hybrid cold publication lost its in-flight authority"
                )
            binding.cold_result_reference = _weak_reference(
                result, "terminal full-refit cold hybrid result"
            )
            binding.phase = "cold_ready"
        _observed_session_request(
            "_finish_selected_refit_fit_stage", authority, "cold", result
        )
        return result
    except BaseException:
        with binding.lock:
            binding.phase = "burned"
        if began:
            try:
                _observed_session_request(
                    "_finish_selected_refit_fit_stage", authority, "cold", None
                )
            except BaseException:
                _observed_session_request(
                    "_burn_selected_refit_procedure", authority
                )
        raise


def fit_terminal_full_development_hybrid_warm(
    receipt: object,
    *,
    maximum_sweeps: int = HYBRID_MAXIMUM_SWEEPS,
) -> HybridOptimizerResult:
    """Repeat only from the exact receipt-registered cold hybrid result."""

    binding = _terminal_receipt_minimal(receipt)
    owner, terminal, authority, full_objective = _terminal_stage_context(binding)
    cold = (
        None
        if binding.cold_result_reference is None
        else binding.cold_result_reference()
    )
    anchor = (
        None
        if binding.selected_anchor_reference is None
        else binding.selected_anchor_reference()
    )
    began = False
    try:
        _observed_session_request(
            "_begin_selected_refit_fit_stage",
            owner,
            terminal,
            authority,
            full_objective,
            "warm",
            hybrid_receipt=receipt,
            prior_result=cold,
        )
        began = True
        with binding.lock:
            if binding.phase != "cold_ready" or cold is None or anchor is None:
                raise LowRankError("terminal hybrid warm fit lacks its cold result")
            binding.phase = "warm_in_flight"
        binding = _require_terminal_full_development_hybrid_receipt(receipt)
        theta, prepared, candidates, geometry, pairs, specification = (
            _terminal_objective(binding)
        )
        result = _fit_sparse_plus_low_rank_impl(
            theta,
            prepared,
            candidates,
            geometry,
            pairs,
            specification,
            anchor,
            linear_correction=None,
            production_receipt=receipt,
            warm_start=cold,
            maximum_sweeps=maximum_sweeps,
        )
        with binding.lock:
            if binding.phase != "warm_in_flight":
                raise LowRankError(
                    "terminal hybrid warm publication lost its in-flight authority"
                )
            binding.warm_result_reference = _weak_reference(
                result, "terminal full-refit warm hybrid result"
            )
            binding.phase = "warm_ready"
        _observed_session_request(
            "_finish_selected_refit_fit_stage", authority, "warm", result
        )
        return result
    except BaseException:
        with binding.lock:
            binding.phase = "burned"
        if began:
            try:
                _observed_session_request(
                    "_finish_selected_refit_fit_stage", authority, "warm", None
                )
            except BaseException:
                _observed_session_request(
                    "_burn_selected_refit_procedure", authority
                )
        raise


def fit_sparse_plus_low_rank(
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    specification: HybridSpec,
    selected_low_rank_anchor: LowRankOptimizerResult,
    *,
    production_receipt: object,
    maximum_sweeps: int = HYBRID_MAXIMUM_SWEEPS,
) -> HybridOptimizerResult:
    """Fit a controller-authorized production hybrid from its selected anchor."""

    _require_production_hybrid_receipt(
        production_receipt,
        specification,
        selected_low_rank_anchor,
    )
    return _fit_sparse_plus_low_rank_impl(
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        specification,
        selected_low_rank_anchor,
        linear_correction=None,
        production_receipt=production_receipt,
        maximum_sweeps=maximum_sweeps,
    )


def fit_sparse_plus_low_rank_synthetic_qualification(
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    specification: HybridSpec,
    selected_low_rank_anchor: LowRankOptimizerResult,
    *,
    linear_correction: np.ndarray,
    maximum_sweeps: int = HYBRID_MAXIMUM_SWEEPS,
) -> HybridOptimizerResult:
    """Fit an explicitly target-free synthetic hybrid objective."""

    correction = _validated_synthetic_correction(linear_correction)
    return _fit_sparse_plus_low_rank_impl(
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        specification,
        selected_low_rank_anchor,
        linear_correction=correction,
        maximum_sweeps=maximum_sweeps,
    )


def refit_sparse_plus_low_rank_from_converged(
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    specification: HybridSpec,
    selected_low_rank_anchor: LowRankOptimizerResult,
    converged_result: HybridOptimizerResult,
    *,
    production_receipt: object,
    maximum_sweeps: int = HYBRID_MAXIMUM_SWEEPS,
) -> HybridOptimizerResult:
    """Repeat a production hybrid from its exact registered converged state."""

    _require_production_hybrid_receipt(
        production_receipt,
        specification,
        selected_low_rank_anchor,
    )
    return _fit_sparse_plus_low_rank_impl(
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        specification,
        selected_low_rank_anchor,
        linear_correction=None,
        production_receipt=production_receipt,
        warm_start=converged_result,
        maximum_sweeps=maximum_sweeps,
    )


def refit_sparse_plus_low_rank_from_converged_synthetic_qualification(
    fixed_m1_theta: np.ndarray,
    prepared: core.PreparedM2,
    candidates: core.CandidateLibrary,
    geometry: core.GeometryEngine,
    pairs: core.PairFeatureLibrary,
    specification: HybridSpec,
    selected_low_rank_anchor: LowRankOptimizerResult,
    converged_result: HybridOptimizerResult,
    *,
    linear_correction: np.ndarray,
    maximum_sweeps: int = HYBRID_MAXIMUM_SWEEPS,
) -> HybridOptimizerResult:
    """Repeat a synthetic hybrid from its exact registered converged state."""

    correction = _validated_synthetic_correction(linear_correction)
    return _fit_sparse_plus_low_rank_impl(
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        specification,
        selected_low_rank_anchor,
        linear_correction=correction,
        warm_start=converged_result,
        maximum_sweeps=maximum_sweeps,
    )


def benchmark_scale_units(context: Any) -> Mapping[str, Mapping[str, Any]]:
    """Expose the optional view of one context-bound central benchmark run."""

    import v3_m2_scale_benchmark_units_v1 as bridge

    return bridge.optional_scale_units(context)
