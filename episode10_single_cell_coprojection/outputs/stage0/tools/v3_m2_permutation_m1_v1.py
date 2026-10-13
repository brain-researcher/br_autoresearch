#!/usr/bin/env python3
"""Fresh leakage-free outer M1 orchestration for EP10 permutation trials.

The v4 falsifier clarification requires every whole-set permutation to run
the complete 66-path nested M1 selection and the 12 selected-lambda outer
zero refits afresh.  This module implements exactly that outcome-dependent
prefix in memory.  It deliberately has no full-development M1 refit, no M0
fit, no checkpoint loader, and no data loader.

Production one-shot authority, attempt persistence, and the configured M2
fit are layered on top by the named whole-set-permutation adapter.  This
module never accepts a generic production callback factory.
"""

from __future__ import annotations

import math
import os
import re
import sys
import threading
import weakref
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(os.path.abspath(__file__))
TOOLS_DIR = SCRIPT_PATH.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m1 as m1  # noqa: E402
import fit_v3_m1_real_v1 as real_m1  # noqa: E402
import fit_v3_m2 as m2  # noqa: E402
import v3_m2_permutation_adapter_v1 as permutation  # noqa: E402
import v3_m2_trial_executor_v1 as executor  # noqa: E402


EXPECTED_ROWS = 194
EXPECTED_FOLDS = 12
EXPECTED_WARM_PATHS = math.comb(EXPECTED_FOLDS, 2)
EXPECTED_ZERO_REFITS = EXPECTED_FOLDS
_NO_CHECKPOINT_IO = object()
_EXECUTION_SEAL = object()
_IDENTITY_PATTERN = re.compile(r"[A-Za-z0-9_.:-]+\Z")
_EXPECTED_WARM_KEYS = frozenset(
    f"leave_two__{left}__{right}"
    for left_index, left in enumerate(m1.GROUP_ORDER)
    for right in m1.GROUP_ORDER[left_index + 1 :]
)
_EXPECTED_ZERO_KEYS = frozenset(
    f"outer_refit__{group}" for group in m1.GROUP_ORDER
)
_EXECUTION_REGISTRY_LOCK = threading.RLock()
_EXECUTION_REGISTRY: weakref.WeakKeyDictionary[Any, tuple[Any, ...]] = (
    weakref.WeakKeyDictionary()
)


class PermutationM1Error(real_m1.RealM1Error):
    """A frozen per-permutation M1 invariant failed closed."""


@dataclass(frozen=True)
class _ArrayAnchor:
    reference: np.ndarray
    snapshot: np.ndarray


@dataclass(frozen=True)
class _CandidateBasisAnchor:
    k: int
    reference: m1.CandidateBasis
    membership: _ArrayAnchor
    contrasts: _ArrayAnchor


@dataclass(frozen=True)
class _GeometryAnchor:
    pairs_reference: m1.PairGeometry
    distance: _ArrayAnchor
    cross_side: _ArrayAnchor
    homotopic: _ArrayAnchor
    ordered_edges: tuple[tuple[float, int, int], ...]
    shared_features: tuple[tuple[int, _ArrayAnchor], ...]


def _readonly_float(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise PermutationM1Error(f"{label} is malformed or nonfinite")
    contiguous = np.ascontiguousarray(array)
    frozen = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=contiguous.dtype
    ).reshape(contiguous.shape, order="C")
    if frozen.flags.writeable:
        raise PermutationM1Error(f"{label} could not be made immutable")
    return frozen


def _immutable_array(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if type(value) is not np.ndarray or array.dtype.hasobject:
        raise PermutationM1Error(f"{label} is not an exact numeric array")
    contiguous = np.ascontiguousarray(array)
    frozen = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=contiguous.dtype
    ).reshape(contiguous.shape, order="C")
    if frozen.flags.writeable:
        raise PermutationM1Error(f"{label} snapshot could not be made immutable")
    return frozen


def _array_anchor(value: Any, label: str) -> _ArrayAnchor:
    if type(value) is not np.ndarray:
        raise PermutationM1Error(f"{label} is not an exact ndarray")
    return _ArrayAnchor(value, _immutable_array(value, label))


def _require_array_anchor(value: Any, anchor: _ArrayAnchor, label: str) -> None:
    if (
        value is not anchor.reference
        or type(value) is not np.ndarray
        or value.dtype != anchor.snapshot.dtype
        or value.shape != anchor.snapshot.shape
        or np.ascontiguousarray(value).tobytes(order="C")
        != anchor.snapshot.tobytes(order="C")
    ):
        raise PermutationM1Error(f"{label} changed after permutation M1 fitting")


_CANDIDATE_BASIS_METHOD = m1.CandidateLibrary.basis
_GEOMETRY_RAW_BATCH_METHOD = m1.GeometryEngine.raw_batch
_GEOMETRY_SHARED_FEATURES_METHOD = m1.GeometryEngine.shared_features
_GEOMETRY_UNIFORM_MOMENTS_METHOD = m1.GeometryEngine.uniform_moments
_PAIR_FEATURE_LIBRARY_CLASS = m2.PairFeatureLibrary
_PAIR_FEATURE_INIT_METHOD = m2.PairFeatureLibrary.__init__
_PAIR_FEATURE_BASIS_METHOD = m2.PairFeatureLibrary.basis
_PAIR_FEATURE_STATISTICS_METHOD = m2.PairFeatureLibrary.statistics


def _validate_exact_library_methods(
    candidates: m1.CandidateLibrary,
    geometry: m1.GeometryEngine,
) -> None:
    if (
        type(candidates) is not m1.CandidateLibrary
        or type(geometry) is not m1.GeometryEngine
        or geometry.library is not candidates
        or "basis" in vars(candidates)
        or any(
            name in vars(geometry)
            for name in ("raw_batch", "shared_features", "uniform_moments")
        )
        or m1.CandidateLibrary.basis is not _CANDIDATE_BASIS_METHOD
        or m1.GeometryEngine.raw_batch is not _GEOMETRY_RAW_BATCH_METHOD
        or m1.GeometryEngine.shared_features is not _GEOMETRY_SHARED_FEATURES_METHOD
        or m1.GeometryEngine.uniform_moments is not _GEOMETRY_UNIFORM_MOMENTS_METHOD
    ):
        raise PermutationM1Error(
            "permutation M1 requires the exact physical candidate/geometry methods"
        )


def _candidate_basis_anchors(
    candidates: m1.CandidateLibrary,
    ks: tuple[int, ...],
) -> tuple[_CandidateBasisAnchor, ...]:
    anchors: list[_CandidateBasisAnchor] = []
    for k in ks:
        basis = candidates.basis(k)
        if type(basis) is not m1.CandidateBasis or basis.k != k:
            raise PermutationM1Error("permutation candidate basis identity changed")
        anchors.append(
            _CandidateBasisAnchor(
                k=k,
                reference=basis,
                membership=_array_anchor(
                    basis.membership, f"permutation candidate membership K={k}"
                ),
                contrasts=_array_anchor(
                    basis.contrasts, f"permutation candidate contrasts K={k}"
                ),
            )
        )
    return tuple(anchors)


def _geometry_anchor(
    geometry: m1.GeometryEngine,
    ks: tuple[int, ...],
) -> _GeometryAnchor:
    pairs = geometry.pairs
    if type(pairs) is not m1.PairGeometry:
        raise PermutationM1Error("permutation geometry pair authority changed")
    ordered_edges = tuple(
        (float(distance), int(left), int(right))
        for distance, left, right in pairs.ordered_edges
    )
    if ordered_edges != pairs.ordered_edges:
        raise PermutationM1Error("permutation geometry edge order is malformed")
    return _GeometryAnchor(
        pairs_reference=pairs,
        distance=_array_anchor(pairs.distance, "permutation pair distance"),
        cross_side=_array_anchor(pairs.cross_side, "permutation pair cross-side"),
        homotopic=_array_anchor(pairs.homotopic, "permutation pair homotopic"),
        ordered_edges=ordered_edges,
        shared_features=tuple(
            (
                k,
                _array_anchor(
                    geometry.shared_features(k),
                    f"permutation shared geometry K={k}",
                ),
            )
            for k in ks
        ),
    )


def _require_library_anchors(
    candidates: m1.CandidateLibrary,
    geometry: m1.GeometryEngine,
    candidate_anchors: tuple[_CandidateBasisAnchor, ...],
    geometry_anchor: _GeometryAnchor,
) -> None:
    _validate_exact_library_methods(candidates, geometry)
    for anchor in candidate_anchors:
        basis = candidates.basis(anchor.k)
        if basis is not anchor.reference or basis.k != anchor.k:
            raise PermutationM1Error(
                f"permutation candidate basis K={anchor.k} was relabelled"
            )
        _require_array_anchor(
            basis.membership,
            anchor.membership,
            f"permutation candidate membership K={anchor.k}",
        )
        _require_array_anchor(
            basis.contrasts,
            anchor.contrasts,
            f"permutation candidate contrasts K={anchor.k}",
        )
    pairs = geometry.pairs
    if pairs is not geometry_anchor.pairs_reference or type(pairs) is not m1.PairGeometry:
        raise PermutationM1Error("permutation geometry pair object was relabelled")
    _require_array_anchor(
        pairs.distance, geometry_anchor.distance, "permutation pair distance"
    )
    _require_array_anchor(
        pairs.cross_side, geometry_anchor.cross_side, "permutation pair cross-side"
    )
    _require_array_anchor(
        pairs.homotopic, geometry_anchor.homotopic, "permutation pair homotopic"
    )
    if pairs.ordered_edges != geometry_anchor.ordered_edges:
        raise PermutationM1Error("permutation geometry edge order changed")
    for k, anchor in geometry_anchor.shared_features:
        _require_array_anchor(
            geometry.shared_features(k),
            anchor,
            f"permutation shared geometry K={k}",
        )


def _fresh_pair_feature_library(
    candidates: m1.CandidateLibrary,
) -> m2.PairFeatureLibrary:
    cls = _PAIR_FEATURE_LIBRARY_CLASS
    if (
        m2.PairFeatureLibrary is not cls
        or cls.__init__ is not _PAIR_FEATURE_INIT_METHOD
        or cls.basis is not _PAIR_FEATURE_BASIS_METHOD
        or cls.statistics is not _PAIR_FEATURE_STATISTICS_METHOD
    ):
        raise PermutationM1Error("permutation pair-feature class or methods changed")
    pairs = cls(candidates)
    if (
        type(pairs) is not cls
        or pairs.candidates is not candidates
        or any(name in vars(pairs) for name in ("basis", "statistics"))
    ):
        raise PermutationM1Error("permutation pair-feature authority changed")
    return pairs


def _validate_identity_prefix(value: object) -> str:
    if type(value) is not str or not value or _IDENTITY_PATTERN.fullmatch(value) is None:
        raise PermutationM1Error("permutation M1 identity prefix is malformed")
    return value


def _validate_factory_key(key: object, expected: frozenset[str]) -> str:
    value = str(key)
    if value not in expected or real_m1.CHECKPOINT_KEY_PATTERN.fullmatch(value) is None:
        raise PermutationM1Error("permutation M1 factory key is outside its frozen set")
    return value


def _fit_warm_path(
    key: str,
    indexes: Sequence[int],
    cells: Sequence[m1.Cell],
    moments: Sequence[m1.UniformCellMoments],
    library: m1.CandidateLibrary,
    geometry: m1.GeometryEngine,
) -> real_m1.TrainingBundle:
    """Fit one complete descending 12-lambda path without checkpoint I/O."""
    fit_key = _validate_factory_key(key, _EXPECTED_WARM_KEYS)
    selected = tuple(int(index) for index in indexes)
    prepared = real_m1._prepare_training(cells, moments, selected)
    try:
        fits = m1.fit_path(prepared, library, geometry)
    except m1.M1Error as error:
        error.production_fit_kind = "permutation_in_memory_warm"
        error.production_fit_key = fit_key
        raise
    if tuple(fits) != m1.LAMBDAS_DESCENDING:
        raise PermutationM1Error("permutation warm path lambda order changed")
    for ridge_lambda, result in fits.items():
        real_m1._validate_optimizer_result(result, float(ridge_lambda))
    return real_m1.TrainingBundle(fit_key, selected, prepared, fits, False)


def _fit_zero_outer(
    key: str,
    indexes: Sequence[int],
    ridge_lambda: float,
    cells: Sequence[m1.Cell],
    moments: Sequence[m1.UniformCellMoments],
    library: m1.CandidateLibrary,
    geometry: m1.GeometryEngine,
) -> real_m1.TrainingBundle:
    """Fit one selected outer model from all-zero 261-vector initialization."""
    fit_key = _validate_factory_key(key, _EXPECTED_ZERO_KEYS)
    ridge = float(ridge_lambda)
    if ridge not in m1.LAMBDAS_DESCENDING:
        raise PermutationM1Error("permutation outer lambda is outside the frozen grid")
    selected = tuple(int(index) for index in indexes)
    prepared = real_m1._prepare_training(cells, moments, selected)
    try:
        result = m1.fit_newton(prepared, library, geometry, ridge, initial=None)
    except m1.M1Error as error:
        error.production_fit_kind = "permutation_in_memory_zero"
        error.production_fit_key = fit_key
        error.production_fit_lambda = ridge
        raise
    real_m1._validate_optimizer_result(result, ridge)
    return real_m1.TrainingBundle(
        fit_key, selected, prepared, MappingProxyType({ridge: result}), False
    )


@dataclass
class _FactoryAccounting:
    warm_keys: list[str] = field(default_factory=list)
    zero_keys: list[str] = field(default_factory=list)

    def record_warm(self, key: object, checkpoint_root: object) -> str:
        if checkpoint_root is not _NO_CHECKPOINT_IO:
            raise PermutationM1Error("permutation warm path received checkpoint I/O")
        value = _validate_factory_key(key, _EXPECTED_WARM_KEYS)
        self.warm_keys.append(value)
        return value

    def record_zero(self, key: object, checkpoint_root: object) -> str:
        if checkpoint_root is not _NO_CHECKPOINT_IO:
            raise PermutationM1Error("permutation zero refit received checkpoint I/O")
        value = _validate_factory_key(key, _EXPECTED_ZERO_KEYS)
        self.zero_keys.append(value)
        return value

    def validate_complete(self) -> None:
        if (
            len(self.warm_keys) != EXPECTED_WARM_PATHS
            or len(set(self.warm_keys)) != EXPECTED_WARM_PATHS
            or frozenset(self.warm_keys) != _EXPECTED_WARM_KEYS
        ):
            raise PermutationM1Error(
                "permutation M1 did not run exactly 66 unique leave-two warm paths"
            )
        if (
            len(self.zero_keys) != EXPECTED_ZERO_REFITS
            or len(set(self.zero_keys)) != EXPECTED_ZERO_REFITS
            or frozenset(self.zero_keys) != _EXPECTED_ZERO_KEYS
        ):
            raise PermutationM1Error(
                "permutation M1 did not run exactly 12 unique outer zero refits"
            )


@dataclass(frozen=True, init=False, eq=False)
class PermutationM1Execution:
    """Immutable aggregate of one fresh, outer-only permuted M1 execution."""

    assignment: permutation.PermutationAssignment
    candidates: m1.CandidateLibrary
    geometry: m1.GeometryEngine
    permuted_cells: tuple[m1.Cell, ...]
    uniform_moments: tuple[m1.UniformCellMoments, ...]
    selected_lambdas: Mapping[str, float]
    fixed_outer_offsets: Mapping[str, executor.FixedM1Offset]
    baseline_log_probability_nats: np.ndarray
    baseline_marginals: np.ndarray
    boundary_count: int
    warm_path_count: int
    zero_refit_count: int
    full_development_refit_run: bool
    identity_prefix: str
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        assignment: permutation.PermutationAssignment,
        candidates: m1.CandidateLibrary,
        geometry: m1.GeometryEngine,
        permuted_cells: tuple[m1.Cell, ...],
        uniform_moments: tuple[m1.UniformCellMoments, ...],
        selected_lambdas: Mapping[str, float],
        fixed_outer_offsets: Mapping[str, executor.FixedM1Offset],
        baseline_log_probability_nats: np.ndarray,
        baseline_marginals: np.ndarray,
        boundary_count: int,
        identity_prefix: str,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _EXECUTION_SEAL:
            raise PermutationM1Error(
                "permutation M1 executions must be issued by fit_permuted_outer_m1"
            )
        object.__setattr__(self, "assignment", assignment)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "permuted_cells", permuted_cells)
        object.__setattr__(self, "uniform_moments", uniform_moments)
        object.__setattr__(
            self, "selected_lambdas", MappingProxyType(dict(selected_lambdas))
        )
        object.__setattr__(
            self, "fixed_outer_offsets", MappingProxyType(dict(fixed_outer_offsets))
        )
        object.__setattr__(
            self, "baseline_log_probability_nats", baseline_log_probability_nats
        )
        object.__setattr__(self, "baseline_marginals", baseline_marginals)
        object.__setattr__(self, "boundary_count", int(boundary_count))
        object.__setattr__(self, "warm_path_count", EXPECTED_WARM_PATHS)
        object.__setattr__(self, "zero_refit_count", EXPECTED_ZERO_REFITS)
        object.__setattr__(self, "full_development_refit_run", False)
        object.__setattr__(self, "identity_prefix", identity_prefix)
        object.__setattr__(self, "_seal", _seal)


def _register_execution(execution: PermutationM1Execution) -> None:
    ks = tuple(sorted({cell.k for cell in execution.permuted_cells}))
    _validate_exact_library_methods(execution.candidates, execution.geometry)
    candidate_anchors = _candidate_basis_anchors(execution.candidates, ks)
    geometry_anchor = _geometry_anchor(execution.geometry, ks)
    with _EXECUTION_REGISTRY_LOCK:
        _EXECUTION_REGISTRY[execution] = (
            weakref.ref(execution.assignment),
            weakref.ref(execution.candidates),
            weakref.ref(execution.geometry),
            execution.permuted_cells,
            execution.uniform_moments,
            execution.baseline_log_probability_nats,
            execution.baseline_marginals,
            candidate_anchors,
            geometry_anchor,
        )


def _require_execution(
    execution: PermutationM1Execution,
) -> PermutationM1Execution:
    if (
        type(execution) is not PermutationM1Execution
        or execution._seal is not _EXECUTION_SEAL
    ):
        raise PermutationM1Error("permutation M1 execution authority changed")
    with _EXECUTION_REGISTRY_LOCK:
        try:
            (
                assignment_reference,
                candidates_reference,
                geometry_reference,
                cells,
                moments,
                baseline,
                marginals,
                candidate_anchors,
                geometry_anchor,
            ) = _EXECUTION_REGISTRY[execution]
        except KeyError as error:
            raise PermutationM1Error(
                "permutation M1 execution lacks live producer authority"
            ) from error
    if (
        assignment_reference() is not execution.assignment
        or candidates_reference() is not execution.candidates
        or geometry_reference() is not execution.geometry
        or cells is not execution.permuted_cells
        or moments is not execution.uniform_moments
        or baseline is not execution.baseline_log_probability_nats
        or marginals is not execution.baseline_marginals
    ):
        raise PermutationM1Error("permutation M1 execution changed after issuance")
    _require_library_anchors(
        execution.candidates,
        execution.geometry,
        candidate_anchors,
        geometry_anchor,
    )
    return execution


def _validate_cross_fit(
    cross_fit: real_m1.CrossFitResult, cells: tuple[m1.Cell, ...]
) -> tuple[np.ndarray, np.ndarray]:
    log_probability = _readonly_float(
        cross_fit.prediction.log_probability_nats,
        (EXPECTED_ROWS,),
        "permutation outer M1 log probability",
    )
    marginals = _readonly_float(
        cross_fit.prediction.marginals,
        (EXPECTED_ROWS, m1.PANEL_SIZE),
        "permutation outer M1 marginals",
    )
    expected_k = np.asarray([cell.k for cell in cells], dtype=float)
    if not (
        np.all(log_probability <= 1.0e-12)
        and np.all(marginals >= -1.0e-12)
        and np.all(marginals <= 1.0 + 1.0e-12)
        and np.allclose(
            np.sum(marginals, axis=1), expected_k, rtol=0.0, atol=1.0e-10
        )
    ):
        raise PermutationM1Error("permutation outer M1 predictions are malformed")
    return log_probability, marginals


def fit_permuted_outer_m1(
    cells: Sequence[m1.Cell],
    assignment: permutation.PermutationAssignment,
    candidates: m1.CandidateLibrary,
    geometry: m1.GeometryEngine,
    *,
    identity_prefix: str,
) -> PermutationM1Execution:
    """Run exactly 66 warm paths and 12 zero-start outer fits on one mapping."""
    prefix = _validate_identity_prefix(identity_prefix)
    if type(assignment) is not permutation.PermutationAssignment:
        raise PermutationM1Error("permutation M1 lacks a frozen assignment")
    _validate_exact_library_methods(candidates, geometry)
    applied_cells = assignment.apply(cells)
    if len(applied_cells) != EXPECTED_ROWS:
        raise PermutationM1Error("permutation M1 row count changed")
    permuted_cells = tuple(
        replace(
            cell,
            soma_distance_mm=_readonly_float(
                cell.soma_distance_mm,
                (m1.PANEL_SIZE,),
                "permutation M1 soma-distance row",
            ),
        )
        for cell in applied_cells
    )
    raw_moments = tuple(geometry.uniform_moments(cell) for cell in permuted_cells)
    moments = tuple(
        m1.UniformCellMoments(
            _readonly_float(
                moment.geometry_mean,
                (m1.GEOMETRY_DIMENSION,),
                "permutation M1 uniform geometry mean",
            ),
            _readonly_float(
                moment.raw_joint_covariance,
                (m1.JOINT_CELL_DIMENSION, m1.JOINT_CELL_DIMENSION),
                "permutation M1 uniform joint covariance",
            ),
        )
        for moment in raw_moments
    )
    accounting = _FactoryAccounting()

    def warm_factory(
        key: str,
        indexes: Sequence[int],
        factory_cells: Sequence[m1.Cell],
        factory_moments: Sequence[m1.UniformCellMoments],
        library: m1.CandidateLibrary,
        factory_geometry: m1.GeometryEngine,
        checkpoint_root: object,
    ) -> real_m1.TrainingBundle:
        accounting.record_warm(key, checkpoint_root)
        return _fit_warm_path(
            key,
            indexes,
            factory_cells,
            factory_moments,
            library,
            factory_geometry,
        )

    def zero_factory(
        key: str,
        indexes: Sequence[int],
        ridge_lambda: float,
        factory_cells: Sequence[m1.Cell],
        factory_moments: Sequence[m1.UniformCellMoments],
        library: m1.CandidateLibrary,
        factory_geometry: m1.GeometryEngine,
        checkpoint_root: object,
    ) -> real_m1.TrainingBundle:
        accounting.record_zero(key, checkpoint_root)
        return _fit_zero_outer(
            key,
            indexes,
            ridge_lambda,
            factory_cells,
            factory_moments,
            library,
            factory_geometry,
        )

    selection = real_m1.select_outer_lambdas(
        permuted_cells,
        moments,
        candidates,
        geometry,
        _NO_CHECKPOINT_IO,  # type: ignore[arg-type]
        warm_factory=warm_factory,
        predictor=real_m1._production_predictor,
    )
    cross_fit = real_m1.refit_outer_and_predict(
        permuted_cells,
        moments,
        candidates,
        geometry,
        _NO_CHECKPOINT_IO,  # type: ignore[arg-type]
        selection,
        zero_factory=zero_factory,
        predictor=real_m1._production_predictor,
    )
    accounting.validate_complete()
    if (
        set(selection.warm_cache) != _EXPECTED_WARM_KEYS
        or set(cross_fit.warm_cache) != _EXPECTED_WARM_KEYS
        or set(selection.selected_lambdas) != set(m1.GROUP_ORDER)
        or set(cross_fit.outer_refits) != set(m1.GROUP_ORDER)
        or cross_fit.boundary_count != selection.boundary_count
    ):
        raise PermutationM1Error("permutation M1 orchestration inventory changed")
    log_probability, marginals = _validate_cross_fit(cross_fit, permuted_cells)

    offsets: dict[str, executor.FixedM1Offset] = {}
    selected_lambdas: dict[str, float] = {}
    for ordinal, group in enumerate(m1.GROUP_ORDER, start=1):
        expected_indexes = tuple(
            index for index, cell in enumerate(permuted_cells) if cell.group != group
        )
        bundle = cross_fit.outer_refits[group]
        ridge = float(selection.selected_lambdas[group])
        if (
            bundle.key != f"outer_refit__{group}"
            or bundle.indexes != expected_indexes
            or bundle.checkpoint_reused is not False
            or set(bundle.fits) != {ridge}
            or ridge not in m1.LAMBDAS_DESCENDING
        ):
            raise PermutationM1Error(
                "permutation M1 outer refit is not the exact fresh fold-bound fit"
            )
        result = bundle.fits[ridge]
        theta = _readonly_float(
            result.theta, (m1.PARAMETER_DIMENSION,), "permutation M1 theta"
        )
        design_mean = _readonly_float(
            bundle.prepared.design_transform.soma_mean,
            (9,),
            "permutation M1 soma mean",
        )
        design_scale = _readonly_float(
            bundle.prepared.design_transform.soma_scale,
            (9,),
            "permutation M1 soma scale",
        )
        geometry_scales = _readonly_float(
            bundle.prepared.geometry_scales,
            (m1.GEOMETRY_DIMENSION,),
            "permutation M1 geometry scale",
        )
        if np.any(design_scale <= 0.0) or np.any(geometry_scales <= 0.0):
            raise PermutationM1Error("permutation M1 fitted scales are not positive")
        offsets[group] = executor.FixedM1Offset(
            held_out_group=group,
            training_indexes=expected_indexes,
            theta=theta,
            design_transform=m1.DesignTransform(design_mean, design_scale),
            geometry_scales=geometry_scales,
            fit_identity=f"{prefix}__outer_{ordinal:02d}_{group}",
        )
        selected_lambdas[group] = ridge

    execution = PermutationM1Execution(
        assignment=assignment,
        candidates=candidates,
        geometry=geometry,
        permuted_cells=permuted_cells,
        uniform_moments=moments,
        selected_lambdas=selected_lambdas,
        fixed_outer_offsets=offsets,
        baseline_log_probability_nats=log_probability,
        baseline_marginals=marginals,
        boundary_count=selection.boundary_count,
        identity_prefix=prefix,
        _seal=_EXECUTION_SEAL,
    )
    _register_execution(execution)
    return execution


def prepare_permuted_m2_cache(
    execution: PermutationM1Execution,
    candidates: m1.CandidateLibrary,
    geometry: m1.GeometryEngine,
    *,
    mode: str,
    execution_inventory: executor.FrozenExecutionInventory | None,
    expected_rank: int | None,
) -> executor.PreparedTrialCache:
    """Build the physical M2 cache and rebind its baseline to the fresh M1."""
    _require_execution(execution)
    if (
        candidates is not execution.candidates
        or geometry is not execution.geometry
        or geometry.library is not candidates
    ):
        raise PermutationM1Error("permutation M1 execution/cache authority changed")
    # Each permutation gets a fresh pair-feature cache; accepting a mutable
    # external cache would reopen the M1-to-M2 binding after reauthentication.
    pairs = _fresh_pair_feature_library(candidates)
    cache = executor.prepare_trial_cache(
        execution.permuted_cells,
        execution.uniform_moments,
        candidates,
        geometry,
        pairs,
        execution.fixed_outer_offsets,
        mode=mode,
        execution_inventory=execution_inventory,
        expected_rank=expected_rank,
    )
    for fold in cache.folds:
        expected = execution.baseline_log_probability_nats[
            list(fold.spec.validation_indexes)
        ]
        if not np.array_equal(fold.baseline_log_probability_nats, expected):
            raise PermutationM1Error(
                "permuted M2 cache did not retain its own outer-cross-fitted M1 baseline"
            )
    return cache


__all__ = [
    "EXPECTED_WARM_PATHS",
    "EXPECTED_ZERO_REFITS",
    "PermutationM1Error",
    "PermutationM1Execution",
    "fit_permuted_outer_m1",
    "prepare_permuted_m2_cache",
]
