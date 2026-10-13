#!/usr/bin/env python3
"""Target-free exact residual-pair engine for EP10 v3 M2.

This module composes around :mod:`fit_v3_m1`.  It has no real-data loader and
its command line exposes only a deterministic synthetic self-test.  A caller
must supply already-constructed ``Cell`` objects, uniform candidate moments,
and a fixed M1 parameter vector.

For every cell, both the 120 unordered-pair indicators H and the full 261-D
M1 sufficient statistic F are centered under the uniform distribution on the
cell's fixed-K candidate set.  Training folds fit the equal-group projection
of H on F and the residual RMS scales.  Validation folds only apply that
training transform.  M2 is the exact conditional softmax tilt

    fixed_M1_score(candidate) + beta @ scaled_residual_pair(candidate),

with beta constrained to the numerically identified range of the training
residual Gram matrix.  The supported penalties are lasso, elastic net, and a
36-block bilateral anatomical group lasso in the original lexicographic edge
coordinate (never in a rotated eigenbasis).
"""

from __future__ import annotations

import argparse
import itertools
import json
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

import fit_v3_m1 as m1  # noqa: E402


CandidateLibrary = m1.CandidateLibrary
GeometryEngine = m1.GeometryEngine
Cell = m1.Cell
UniformCellMoments = m1.UniformCellMoments

PAIR_INDEX = tuple(itertools.combinations(range(m1.PANEL_SIZE), 2))
PAIR_DIMENSION = len(PAIR_INDEX)
if PAIR_DIMENSION != 120:
    raise RuntimeError("the frozen 16-target panel must have 120 unordered pairs")

PAIR_TO_INDEX = {pair: index for index, pair in enumerate(PAIR_INDEX)}
ANATOMICAL_BASES = tuple(
    m1.PANEL[2 * index].split("__", 1)[0] for index in range(8)
)


def _build_bilateral_groups() -> tuple[tuple[str, tuple[int, ...]], ...]:
    groups: list[tuple[str, tuple[int, ...]]] = []
    for left_base, right_base in itertools.combinations(range(8), 2):
        edges = sorted(
            PAIR_TO_INDEX[tuple(sorted((2 * left_base + left_side, 2 * right_base + right_side)))]
            for left_side in range(2)
            for right_side in range(2)
        )
        groups.append(
            (
                f"{ANATOMICAL_BASES[left_base]}__{ANATOMICAL_BASES[right_base]}",
                tuple(edges),
            )
        )
    for base_index, base in enumerate(ANATOMICAL_BASES):
        groups.append(
            (
                f"{base}__homotopic",
                (PAIR_TO_INDEX[(2 * base_index, 2 * base_index + 1)],),
            )
        )
    flattened = [edge for _, edges in groups for edge in edges]
    if (
        len(groups) != 36
        or sorted(len(edges) for _, edges in groups) != [1] * 8 + [4] * 28
        or sorted(flattened) != list(range(PAIR_DIMENSION))
    ):
        raise RuntimeError("bilateral anatomical groups do not partition 120 pairs")
    return tuple(groups)


BILATERAL_GROUPS = _build_bilateral_groups()
RANK_EPSILON_MULTIPLIER = PAIR_DIMENSION
PROJECTION_TOLERANCE = 1.0e-10
PROJECTOR_TOLERANCE = 1.0e-12
PROX_TOLERANCE = 1.0e-12
PROX_MAXIMUM_ITERATIONS = 10_000
ZERO_KKT_ACCELERATED_MAXIMUM_ITERATIONS = 512
ZERO_KKT_MINIMAX_MAXIMUM_ITERATIONS = 1_024
ZERO_KKT_MINIMAX_SMOOTHING_MINIMUM_EXPONENT = 1
ZERO_KKT_MINIMAX_SMOOTHING_MAXIMUM_EXPONENT = 18
ZERO_KKT_MINIMAX_NEWTON_STEPS_PER_SMOOTHING = 40
ZERO_KKT_ACTIVE_FACE_ROUNDOFF_MULTIPLIER = 64.0
ZERO_KKT_ACTIVE_FACE_BACKWARD_ERROR_MULTIPLIER = 1_024.0
KKT_TOLERANCE = 1.0e-7
RELATIVE_OBJECTIVE_TOLERANCE = 1.0e-10
FISTA_MAXIMUM_ITERATIONS = 20_000
FISTA_INITIAL_LIPSCHITZ = 1.0
BACKTRACKING_MULTIPLIER = 2.0
MAJORIZATION_EPSILON_MULTIPLIER = 64.0
# This is only a safety cap; the purely relative stopping rule normally exits
# much earlier.  The larger cap keeps nonzero binary64 gradients far below the
# former absolute zero floor representable without changing the frozen
# expansion ceiling.
LAMBDA_MAXIMUM_BISECTIONS = 256
LAMBDA_MAXIMUM_INITIAL_UPPER = 1.0
LAMBDA_MAXIMUM_MAXIMUM_EXPANSIONS = 80
LAMBDA_MAXIMUM_RELATIVE_BRACKET_TOLERANCE = 1.0e-12


class M2Error(RuntimeError):
    """A target-free M2 mathematical or numerical contract failed."""


class _ZeroKKTIndeterminate(M2Error):
    """The bounded solver proved neither side of a zero-KKT query."""


@dataclass(frozen=True)
class PairBasis:
    k: int
    indicators: np.ndarray

    @property
    def count(self) -> int:
        return int(self.indicators.shape[0])


@dataclass(frozen=True)
class PairUniformStatistics:
    mean: np.ndarray
    covariance: np.ndarray
    contrast_cross_covariance: np.ndarray


class PairFeatureLibrary:
    """Panel-wide, bounded cache of lexicographic pair indicators and moments."""

    def __init__(self, candidates: CandidateLibrary) -> None:
        self.candidates = candidates
        self._bases: dict[int, PairBasis] = {}
        self._statistics: dict[int, PairUniformStatistics] = {}
        self.left = np.asarray([pair[0] for pair in PAIR_INDEX], dtype=int)
        self.right = np.asarray([pair[1] for pair in PAIR_INDEX], dtype=int)

    def basis(self, k: int) -> PairBasis:
        key = int(k)
        if key not in self._bases:
            membership = self.candidates.basis(key).membership
            indicators = (
                membership[:, self.left] * membership[:, self.right]
            ).astype(np.uint8, copy=False)
            expected = math.comb(key, 2)
            if indicators.shape != (membership.shape[0], PAIR_DIMENSION):
                raise M2Error("pair-indicator cache has the wrong shape")
            if not np.all(np.sum(indicators, axis=1, dtype=int) == expected):
                raise M2Error("candidate pair count differs from choose(K,2)")
            self._bases[key] = PairBasis(key, indicators)
        return self._bases[key]

    def statistics(self, k: int) -> PairUniformStatistics:
        key = int(k)
        if key not in self._statistics:
            pair = self.basis(key).indicators.astype(float, copy=False)
            contrast = self.candidates.basis(key).contrasts
            mean_h = np.mean(pair, axis=0)
            mean_f = np.mean(contrast, axis=0)
            centered_h = pair - mean_h
            centered_f = contrast - mean_f
            denominator = float(pair.shape[0])
            covariance = centered_h.T @ centered_h / denominator
            cross = centered_f.T @ centered_h / denominator
            self._statistics[key] = PairUniformStatistics(
                mean=_finite(mean_h, "uniform pair mean"),
                covariance=_finite(
                    0.5 * (covariance + covariance.T),
                    "uniform pair covariance",
                ),
                contrast_cross_covariance=_finite(
                    cross, "uniform target-contrast/pair cross covariance"
                ),
            )
        return self._statistics[key]

    def all_counts(self) -> dict[int, int]:
        return {k: self.basis(k).count for k in range(m1.PANEL_SIZE + 1)}


@dataclass(frozen=True)
class ResidualTransform:
    design_transform: m1.DesignTransform
    geometry_scales: np.ndarray
    projection_f_to_h: np.ndarray
    pair_residual_scales: np.ndarray
    range_basis: np.ndarray
    range_projector: np.ndarray
    residual_gram: np.ndarray
    scaled_residual_gram: np.ndarray
    diagnostics: Mapping[str, Any]

    @property
    def rank(self) -> int:
        return int(self.range_basis.shape[1])


@dataclass(frozen=True)
class PreparedM2:
    m1_data: m1.PreparedData
    transform: ResidualTransform

    @property
    def cells(self) -> tuple[Cell, ...]:
        return self.m1_data.cells

    @property
    def weights(self) -> np.ndarray:
        return self.m1_data.weights


@dataclass(frozen=True)
class _ArrayAnchor:
    reference: Any
    snapshot: np.ndarray


@dataclass(frozen=True)
class _CellAnchor:
    reference: Any
    scalar_fields: tuple[Any, ...]
    soma_distance_mm: _ArrayAnchor


@dataclass(frozen=True)
class _MomentAnchor:
    reference: Any
    geometry_mean: _ArrayAnchor
    raw_joint_covariance: _ArrayAnchor


@dataclass(frozen=True)
class _CandidateBasisAnchor:
    k: int
    reference: Any
    membership: _ArrayAnchor
    contrasts: _ArrayAnchor


@dataclass(frozen=True)
class _PhysicalFoldRecord:
    prepared_reference: Any
    m1_data_reference: Any
    transform_reference: Any
    candidates_reference: Any
    geometry_reference: Any
    pairs_reference: Any
    candidate_bases: tuple[_CandidateBasisAnchor, ...]
    selected_indexes: tuple[int, ...]
    cells: tuple[_CellAnchor, ...]
    moments: tuple[_MomentAnchor, ...]
    x: _ArrayAnchor
    geometry_centers: _ArrayAnchor
    prepared_geometry_scales: _ArrayAnchor
    weights: _ArrayAnchor
    design_transform_reference: Any
    soma_mean: _ArrayAnchor
    soma_scale: _ArrayAnchor
    transform_geometry_scales: _ArrayAnchor
    projection_f_to_h: _ArrayAnchor
    pair_residual_scales: _ArrayAnchor
    range_basis: _ArrayAnchor
    range_projector: _ArrayAnchor
    residual_gram: _ArrayAnchor
    scaled_residual_gram: _ArrayAnchor
    gram_ff: np.ndarray
    inverse_ff: np.ndarray
    ff_tolerance: float
    ff_rank: int
    physical_rank: int
    training_groups: int


@dataclass(frozen=True)
class _PhysicalFoldProvenance:
    gram_ff: np.ndarray
    inverse_ff: np.ndarray
    ff_tolerance: float
    ff_rank: int
    physical_rank: int
    training_groups: int
    selected_indexes: tuple[int, ...]
    candidate_ks: tuple[int, ...]
    candidates: CandidateLibrary
    geometry: GeometryEngine
    pairs: PairFeatureLibrary


@dataclass(frozen=True)
class _RademacherTransformRecord:
    transform_reference: Any
    fitted_reference: Any
    features_reference: Any


@dataclass(frozen=True)
class _RademacherPreparedRecord:
    prepared_reference: Any
    transform_reference: Any
    fitted_reference: Any
    features_reference: Any


_PRIVATE_REGISTRY_LOCK = threading.RLock()
_PHYSICAL_FOLD_REGISTRY: dict[int, tuple[Any, _PhysicalFoldRecord]] = {}
_RADEMACHER_TRANSFORM_REGISTRY: dict[
    int, tuple[Any, _RademacherTransformRecord]
] = {}
_RADEMACHER_PREPARED_REGISTRY: dict[
    int, tuple[Any, _RademacherPreparedRecord]
] = {}
_RADEMACHER_MODULE_NAME = "v3_m2_rademacher_adapter_v1"
_RADEMACHER_MODULE_PATH = TOOLS_DIR / f"{_RADEMACHER_MODULE_NAME}.py"
_RADEMACHER_FEATURE_KIND = "centered_Rademacher_noise_pseudofeatures"


def _weak_reference(value: object, label: str) -> Any:
    try:
        return weakref.ref(value)
    except TypeError as error:
        raise M2Error(f"{label} does not support identity-bound provenance") from error


def _register_identity(
    registry: dict[int, tuple[Any, Any]],
    value: object,
    record: Any,
) -> None:
    key = id(value)

    def remove(reference: Any, *, identity: int = key) -> None:
        with _PRIVATE_REGISTRY_LOCK:
            current = registry.get(identity)
            if current is not None and current[0] is reference:
                registry.pop(identity, None)

    reference = weakref.ref(value, remove)
    with _PRIVATE_REGISTRY_LOCK:
        registry[key] = (reference, record)


def _lookup_identity(
    registry: Mapping[int, tuple[Any, Any]],
    value: object,
) -> Any | None:
    with _PRIVATE_REGISTRY_LOCK:
        entry = registry.get(id(value))
    if entry is None or entry[0]() is not value:
        return None
    return entry[1]


def _immutable_snapshot(array: np.ndarray, label: str) -> np.ndarray:
    source = np.asarray(array)
    if source.dtype.hasobject:
        raise M2Error(f"{label} cannot use object dtype")
    contiguous = np.ascontiguousarray(source)
    result = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=contiguous.dtype
    ).reshape(contiguous.shape, order="C")
    if result.flags.writeable:
        raise M2Error(f"{label} snapshot is unexpectedly mutable")
    return result


def _array_anchor(array: np.ndarray, label: str) -> _ArrayAnchor:
    if type(array) is not np.ndarray:
        raise M2Error(f"{label} is not an exact ndarray")
    return _ArrayAnchor(
        reference=_weak_reference(array, label),
        snapshot=_immutable_snapshot(array, label),
    )


def _require_array_anchor(
    array: np.ndarray,
    anchor: _ArrayAnchor,
    label: str,
) -> None:
    if array is not anchor.reference():
        raise M2Error(f"{label} was relabelled after physical preparation")
    if (
        array.dtype != anchor.snapshot.dtype
        or array.shape != anchor.snapshot.shape
        or not np.array_equal(array, anchor.snapshot)
    ):
        raise M2Error(f"{label} changed after physical preparation")


def _require_byte_exact_array_anchor(
    array: np.ndarray,
    anchor: _ArrayAnchor,
    label: str,
) -> None:
    if array is not anchor.reference():
        raise M2Error(f"{label} was relabelled after physical preparation")
    if (
        array.dtype != anchor.snapshot.dtype
        or array.shape != anchor.snapshot.shape
        or array.strides != anchor.snapshot.strides
        or not np.array_equal(
            array.view(np.uint8).reshape(-1),
            anchor.snapshot.view(np.uint8).reshape(-1),
        )
    ):
        raise M2Error(f"{label} changed bytewise after physical preparation")


def _cell_scalar_fields(cell: Cell) -> tuple[Any, ...]:
    return (
        cell.identifier,
        cell.group,
        cell.layer,
        cell.hemisphere,
        tuple(cell.soma_um),
        cell.k,
        cell.observed_index,
        cell.segment_unmapped,
        cell.sampling_uniform,
    )


def _cell_anchor(cell: Cell) -> _CellAnchor:
    if type(cell) is not Cell:
        raise M2Error("physical cell is not the exact Cell type")
    return _CellAnchor(
        reference=_weak_reference(cell, "physical cell"),
        scalar_fields=_cell_scalar_fields(cell),
        soma_distance_mm=_array_anchor(
            cell.soma_distance_mm, "physical cell soma-distance array"
        ),
    )


def _require_cell_anchor(cell: Cell, anchor: _CellAnchor) -> None:
    if cell is not anchor.reference():
        raise M2Error("physical cell identity changed after preparation")
    if _cell_scalar_fields(cell) != anchor.scalar_fields:
        raise M2Error("physical cell fields changed after preparation")
    _require_array_anchor(
        cell.soma_distance_mm,
        anchor.soma_distance_mm,
        "physical cell soma-distance array",
    )


def _moment_anchor(moment: UniformCellMoments) -> _MomentAnchor:
    if type(moment) is not UniformCellMoments:
        raise M2Error("physical uniform moments have the wrong type")
    return _MomentAnchor(
        reference=_weak_reference(moment, "physical uniform moments"),
        geometry_mean=_array_anchor(
            moment.geometry_mean, "physical uniform geometry mean"
        ),
        raw_joint_covariance=_array_anchor(
            moment.raw_joint_covariance,
            "physical uniform joint covariance",
        ),
    )


def _require_moment_anchor(
    moment: UniformCellMoments,
    anchor: _MomentAnchor,
) -> None:
    if moment is not anchor.reference():
        raise M2Error("physical uniform-moment identity changed after preparation")
    _require_array_anchor(
        moment.geometry_mean,
        anchor.geometry_mean,
        "physical uniform geometry mean",
    )
    _require_array_anchor(
        moment.raw_joint_covariance,
        anchor.raw_joint_covariance,
        "physical uniform joint covariance",
    )


def _candidate_basis_anchor(
    candidates: CandidateLibrary,
    k: int,
) -> _CandidateBasisAnchor:
    basis = candidates.basis(k)
    if type(basis) is not m1.CandidateBasis or basis.k != k:
        raise M2Error("physical candidate basis has the wrong exact type or K")
    return _CandidateBasisAnchor(
        k=k,
        reference=_weak_reference(basis, "physical candidate basis"),
        membership=_array_anchor(
            basis.membership, f"physical K={k} candidate membership"
        ),
        contrasts=_array_anchor(
            basis.contrasts, f"physical K={k} candidate contrasts"
        ),
    )


def _require_candidate_basis_anchor(
    candidates: CandidateLibrary,
    anchor: _CandidateBasisAnchor,
) -> None:
    basis = candidates.basis(anchor.k)
    if basis is not anchor.reference():
        raise M2Error(
            f"physical K={anchor.k} candidate basis was relabelled after fitting"
        )
    _require_byte_exact_array_anchor(
        basis.membership,
        anchor.membership,
        f"physical K={anchor.k} candidate membership",
    )
    _require_byte_exact_array_anchor(
        basis.contrasts,
        anchor.contrasts,
        f"physical K={anchor.k} candidate contrasts",
    )


@dataclass(frozen=True)
class M2Evaluation:
    objective: float
    gradient: np.ndarray
    log_probability_nats: np.ndarray | None
    marginals: np.ndarray | None
    candidate_visits: int


@dataclass(frozen=True)
class ConditionalDistribution:
    probabilities: np.ndarray
    log_probabilities: np.ndarray
    marginals: np.ndarray


@dataclass(frozen=True)
class PenaltySpec:
    kind: str
    strength: float
    l1_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in {"lasso", "elastic_net", "group_lasso"}:
            raise M2Error(f"unsupported M2 penalty {self.kind!r}")
        if not math.isfinite(float(self.strength)) or self.strength < 0.0:
            raise M2Error("penalty strength must be finite and nonnegative")
        if self.kind == "elastic_net":
            if self.l1_ratio not in {0.25, 0.5, 0.75}:
                raise M2Error("elastic-net l1_ratio must be one of 0.25, 0.5, 0.75")
        elif self.l1_ratio != 1.0:
            raise M2Error("l1_ratio applies only to elastic net")


@dataclass(frozen=True)
class ProxResult:
    vector: np.ndarray
    iterations: int
    fixed_point_residual: float
    range_residual: float
    invariant_residual: float
    certified: bool


@dataclass(frozen=True)
class _DykstraWarmState:
    amount: float
    penalty_kind: str
    l1_ratio: float
    penalty_correction: np.ndarray
    range_correction: np.ndarray


@dataclass(frozen=True)
class LambdaMaximumResult:
    value: float
    lower_bracket: float
    upper_bracket: float
    zero_kkt_residual: float
    bisections: int


@dataclass(frozen=True)
class ZeroKKTCertificate:
    residual: float
    stationarity: float
    iterations: int
    certified: bool
    feasible: bool
    separation_margin: float
    correction: tuple[float, ...]
    active_face_growths: int = 0
    active_face_qp_events: int = 0
    active_face_size: int = 0
    active_face_support_size: int = 0
    active_face_maximum_weight: float = 0.0
    active_face_effective_constraint_count: float = 0.0


@dataclass(frozen=True)
class M2OptimizerResult:
    beta: np.ndarray
    objective: float
    smooth_objective: float
    penalty_value: float
    iterations: int
    lipschitz: float
    kkt_residual: float
    relative_objective_change: float
    range_residual: float
    status: str
    telemetry: Mapping[str, Any]


def _finite(array: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(result)):
        raise M2Error(f"{label} is nonfinite")
    return result


def _symmetric_eigensystem(matrix: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    symmetric = np.asarray(matrix, dtype=float)
    if symmetric.ndim != 2 or symmetric.shape[0] != symmetric.shape[1]:
        raise M2Error(f"{label} is not square")
    symmetric = _finite(0.5 * (symmetric + symmetric.T), label)
    values, vectors = np.linalg.eigh(symmetric)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(vectors)):
        raise M2Error(f"{label} eigensystem is nonfinite")
    return values, vectors


def _deterministic_pseudoinverse(matrix: np.ndarray, label: str) -> tuple[np.ndarray, float, int]:
    values, vectors = _symmetric_eigensystem(matrix, label)
    largest = float(np.max(np.abs(values))) if values.size else 0.0
    tolerance = max(matrix.shape) * np.finfo(float).eps * max(1.0, largest)
    if values.size and float(np.min(values)) < -10.0 * tolerance:
        raise M2Error(f"{label} is materially indefinite")
    keep = values > tolerance
    inverse = np.zeros_like(values)
    inverse[keep] = 1.0 / values[keep]
    result = (vectors * inverse) @ vectors.T
    return _finite(0.5 * (result + result.T), f"{label} pseudoinverse"), tolerance, int(np.sum(keep))


def _register_physical_fold_provenance(
    prepared: PreparedM2,
    cells: Sequence[Cell],
    moments: Sequence[UniformCellMoments],
    selected: tuple[int, ...],
    candidates: CandidateLibrary,
    geometry: GeometryEngine,
    pairs: PairFeatureLibrary,
    gram_ff: np.ndarray,
    inverse_ff: np.ndarray,
    ff_tolerance: float,
    ff_rank: int,
) -> None:
    if type(prepared) is not PreparedM2 or type(prepared.m1_data) is not m1.PreparedData:
        raise M2Error("physical PreparedM2 issuance is malformed")
    if type(prepared.transform) is not ResidualTransform:
        raise M2Error("physical residual transform issuance is malformed")
    if type(candidates) is not CandidateLibrary:
        raise M2Error("physical candidate library has the wrong type")
    if type(geometry) is not GeometryEngine or geometry.library is not candidates:
        raise M2Error("physical geometry is not bound to the candidate library")
    if type(pairs) is not PairFeatureLibrary or pairs.candidates is not candidates:
        raise M2Error("physical pair features are not bound to the candidate library")
    selected_cells = tuple(cells[index] for index in selected)
    selected_moments = tuple(moments[index] for index in selected)
    if len(prepared.cells) != len(selected_cells) or any(
        prepared.cells[index] is not selected_cells[index]
        for index in range(len(selected_cells))
    ):
        raise M2Error("physical PreparedM2 cells changed before provenance issuance")
    prepared_m1 = prepared.m1_data
    transform = prepared.transform
    design = prepared_m1.design_transform
    if transform.design_transform is not design:
        raise M2Error("physical transform lost its M1 design-transform identity")
    record = _PhysicalFoldRecord(
        prepared_reference=_weak_reference(prepared, "physical PreparedM2"),
        m1_data_reference=_weak_reference(prepared_m1, "physical M1 preparation"),
        transform_reference=_weak_reference(transform, "physical residual transform"),
        candidates_reference=_weak_reference(candidates, "physical candidate library"),
        geometry_reference=_weak_reference(geometry, "physical geometry"),
        pairs_reference=_weak_reference(pairs, "physical pair-feature library"),
        candidate_bases=tuple(
            _candidate_basis_anchor(candidates, k)
            for k in sorted({cell.k for cell in cells})
        ),
        selected_indexes=selected,
        cells=tuple(_cell_anchor(cell) for cell in selected_cells),
        moments=tuple(_moment_anchor(moment) for moment in selected_moments),
        x=_array_anchor(prepared_m1.x, "physical prepared design"),
        geometry_centers=_array_anchor(
            prepared_m1.geometry_centers, "physical geometry centers"
        ),
        prepared_geometry_scales=_array_anchor(
            prepared_m1.geometry_scales, "physical prepared geometry scales"
        ),
        weights=_array_anchor(prepared_m1.weights, "physical equal-group weights"),
        design_transform_reference=_weak_reference(
            design, "physical design transform"
        ),
        soma_mean=_array_anchor(design.soma_mean, "physical soma mean"),
        soma_scale=_array_anchor(design.soma_scale, "physical soma scale"),
        transform_geometry_scales=_array_anchor(
            transform.geometry_scales, "physical transform geometry scales"
        ),
        projection_f_to_h=_array_anchor(
            transform.projection_f_to_h, "physical F-to-H projection"
        ),
        pair_residual_scales=_array_anchor(
            transform.pair_residual_scales, "physical pair RMS"
        ),
        range_basis=_array_anchor(transform.range_basis, "physical range basis"),
        range_projector=_array_anchor(
            transform.range_projector, "physical range projector"
        ),
        residual_gram=_array_anchor(
            transform.residual_gram, "physical residual Gram"
        ),
        scaled_residual_gram=_array_anchor(
            transform.scaled_residual_gram, "physical scaled residual Gram"
        ),
        gram_ff=_immutable_snapshot(gram_ff, "same-pass physical G_FF"),
        inverse_ff=_immutable_snapshot(
            inverse_ff, "same-pass physical G_FF pseudoinverse"
        ),
        ff_tolerance=float(ff_tolerance),
        ff_rank=int(ff_rank),
        physical_rank=transform.rank,
        training_groups=len({cell.group for cell in selected_cells}),
    )
    _register_identity(_PHYSICAL_FOLD_REGISTRY, prepared, record)


def _reauthenticate_physical_fold_provenance(
    prepared: PreparedM2,
) -> _PhysicalFoldProvenance:
    if type(prepared) is not PreparedM2:
        raise M2Error("physical prepared fold is malformed")
    record = _lookup_identity(_PHYSICAL_FOLD_REGISTRY, prepared)
    if record is None or prepared is not record.prepared_reference():
        raise M2Error("physical prepared fold has no same-pass provenance")
    prepared_m1 = prepared.m1_data
    transform = prepared.transform
    if prepared_m1 is not record.m1_data_reference():
        raise M2Error("physical M1 preparation was relabelled after fitting")
    if transform is not record.transform_reference():
        raise M2Error("physical residual transform was relabelled after fitting")
    candidates = record.candidates_reference()
    geometry = record.geometry_reference()
    pairs = record.pairs_reference()
    if candidates is None or geometry is None or pairs is None:
        raise M2Error("physical fold provenance objects are no longer live")
    if geometry.library is not candidates or pairs.candidates is not candidates:
        raise M2Error("physical feature-library bindings changed after fitting")
    for anchor in record.candidate_bases:
        _require_candidate_basis_anchor(candidates, anchor)
    if len(prepared.cells) != len(record.cells):
        raise M2Error("physical prepared-cell count changed after fitting")
    for cell, anchor in zip(prepared.cells, record.cells, strict=True):
        _require_cell_anchor(cell, anchor)
    for anchor in record.moments:
        moment = anchor.reference()
        if moment is None:
            raise M2Error("physical uniform moments are no longer live")
        _require_moment_anchor(moment, anchor)
    _require_array_anchor(prepared_m1.x, record.x, "physical prepared design")
    _require_array_anchor(
        prepared_m1.geometry_centers,
        record.geometry_centers,
        "physical geometry centers",
    )
    _require_array_anchor(
        prepared_m1.geometry_scales,
        record.prepared_geometry_scales,
        "physical prepared geometry scales",
    )
    _require_array_anchor(
        prepared_m1.weights, record.weights, "physical equal-group weights"
    )
    design = prepared_m1.design_transform
    if design is not record.design_transform_reference():
        raise M2Error("physical design transform was relabelled after fitting")
    if transform.design_transform is not design:
        raise M2Error("physical transform lost its M1 design-transform identity")
    _require_array_anchor(design.soma_mean, record.soma_mean, "physical soma mean")
    _require_array_anchor(design.soma_scale, record.soma_scale, "physical soma scale")
    _require_array_anchor(
        transform.geometry_scales,
        record.transform_geometry_scales,
        "physical transform geometry scales",
    )
    _require_array_anchor(
        transform.projection_f_to_h,
        record.projection_f_to_h,
        "physical F-to-H projection",
    )
    _require_array_anchor(
        transform.pair_residual_scales,
        record.pair_residual_scales,
        "physical pair RMS",
    )
    _require_array_anchor(transform.range_basis, record.range_basis, "physical range basis")
    _require_array_anchor(
        transform.range_projector,
        record.range_projector,
        "physical range projector",
    )
    _require_array_anchor(
        transform.residual_gram, record.residual_gram, "physical residual Gram"
    )
    _require_array_anchor(
        transform.scaled_residual_gram,
        record.scaled_residual_gram,
        "physical scaled residual Gram",
    )
    if transform.rank != record.physical_rank:
        raise M2Error("physical residual rank changed after fitting")
    return _PhysicalFoldProvenance(
        gram_ff=record.gram_ff,
        inverse_ff=record.inverse_ff,
        ff_tolerance=record.ff_tolerance,
        ff_rank=record.ff_rank,
        physical_rank=record.physical_rank,
        training_groups=record.training_groups,
        selected_indexes=record.selected_indexes,
        candidate_ks=tuple(anchor.k for anchor in record.candidate_bases),
        candidates=candidates,
        geometry=geometry,
        pairs=pairs,
    )


def _require_physical_fold_provenance(
    prepared: PreparedM2,
    cells: Sequence[Cell],
    moments: Sequence[UniformCellMoments],
    candidates: CandidateLibrary,
    geometry: GeometryEngine,
    pairs: PairFeatureLibrary,
    indexes: Sequence[int],
) -> _PhysicalFoldProvenance:
    provenance = _reauthenticate_physical_fold_provenance(prepared)
    selected = tuple(int(index) for index in indexes)
    if selected != provenance.selected_indexes:
        raise M2Error("physical fold indexes differ from same-pass provenance")
    if (
        candidates is not provenance.candidates
        or geometry is not provenance.geometry
        or pairs is not provenance.pairs
    ):
        raise M2Error("physical fold libraries differ from same-pass provenance")
    if len(cells) != len(moments):
        raise M2Error("physical cells and moments are not aligned")
    record = _lookup_identity(_PHYSICAL_FOLD_REGISTRY, prepared)
    assert record is not None
    for local_index, source_index in enumerate(selected):
        if source_index < 0 or source_index >= len(cells):
            raise M2Error("physical fold index is outside the supplied rows")
        _require_cell_anchor(cells[source_index], record.cells[local_index])
        _require_moment_anchor(moments[source_index], record.moments[local_index])
    return provenance


def _exact_rademacher_module() -> Any:
    module = sys.modules.get(_RADEMACHER_MODULE_NAME)
    if module is None or getattr(module, "__name__", None) != _RADEMACHER_MODULE_NAME:
        raise M2Error("exact Rademacher adapter module is not loaded")
    module_file = getattr(module, "__file__", None)
    if module_file is None or Path(os.path.abspath(module_file)) != _RADEMACHER_MODULE_PATH:
        raise M2Error("Rademacher adapter module path is not exact")
    if (
        not _RADEMACHER_MODULE_PATH.is_file()
        or _RADEMACHER_MODULE_PATH.is_symlink()
        or Path(module_file).resolve(strict=True)
        != _RADEMACHER_MODULE_PATH.resolve(strict=True)
    ):
        raise M2Error("Rademacher adapter module path is not canonical")
    return module


def _exact_rademacher_function(module: Any, name: str) -> Any:
    function = getattr(module, name, None)
    code = getattr(function, "__code__", None)
    if (
        not callable(function)
        or getattr(function, "__module__", None) != _RADEMACHER_MODULE_NAME
        or code is None
        or Path(os.path.abspath(code.co_filename)) != _RADEMACHER_MODULE_PATH
    ):
        raise M2Error(f"Rademacher adapter handshake {name} is not exact")
    return function


def _require_rademacher_transform_record(
    transform: ResidualTransform,
) -> _RademacherTransformRecord | None:
    record = _lookup_identity(_RADEMACHER_TRANSFORM_REGISTRY, transform)
    marker = False
    try:
        marker = transform.diagnostics.get("feature_kind") == _RADEMACHER_FEATURE_KIND
    except AttributeError:
        marker = False
    if record is None:
        if marker:
            raise M2Error("Rademacher transform lacks exact registered provenance")
        return None
    module = _exact_rademacher_module()
    fitted = record.fitted_reference()
    features = record.features_reference()
    if (
        fitted is None
        or features is None
        or transform is not record.transform_reference()
        or type(fitted) is not module.RademacherFoldFit
        or type(features) is not module.RademacherFeatureLibrary
        or fitted.transform is not transform
        or fitted.features is not features
    ):
        raise M2Error("registered Rademacher transform identity changed")
    authenticate = _exact_rademacher_function(module, "_core_authenticate_fold_fit")
    if authenticate(fitted) is not True:
        raise M2Error("registered Rademacher fold failed exact authentication")
    return record


def _register_exact_rademacher_fold(fitted: object) -> None:
    module = _exact_rademacher_module()
    if type(fitted) is not module.RademacherFoldFit:
        raise M2Error("Rademacher fold registration has the wrong exact type")
    authenticate = _exact_rademacher_function(module, "_core_authenticate_fold_fit")
    if authenticate(fitted) is not True:
        raise M2Error("Rademacher fold was not issued by the exact adapter")
    transform = fitted.transform
    prepared = fitted.prepared
    features = fitted.features
    if type(transform) is not ResidualTransform or type(prepared) is not PreparedM2:
        raise M2Error("Rademacher fold registration is malformed")
    transform_record = _RademacherTransformRecord(
        transform_reference=_weak_reference(transform, "Rademacher transform"),
        fitted_reference=_weak_reference(fitted, "Rademacher fold fit"),
        features_reference=_weak_reference(features, "Rademacher feature library"),
    )
    _register_identity(_RADEMACHER_TRANSFORM_REGISTRY, transform, transform_record)
    _register_exact_rademacher_prepared(fitted, prepared)


def _register_exact_rademacher_prepared(
    fitted: object,
    prepared: PreparedM2,
) -> None:
    module = _exact_rademacher_module()
    if type(fitted) is not module.RademacherFoldFit or type(prepared) is not PreparedM2:
        raise M2Error("Rademacher prepared registration has the wrong exact type")
    transform_record = _require_rademacher_transform_record(fitted.transform)
    if transform_record is None:
        raise M2Error("Rademacher prepared registration lacks its fold transform")
    authenticate = _exact_rademacher_function(module, "_core_authenticate_prepared")
    if authenticate(fitted, prepared) is not True:
        raise M2Error("Rademacher PreparedM2 was not issued by the exact adapter")
    record = _RademacherPreparedRecord(
        prepared_reference=_weak_reference(prepared, "Rademacher PreparedM2"),
        transform_reference=_weak_reference(fitted.transform, "Rademacher transform"),
        fitted_reference=_weak_reference(fitted, "Rademacher fold fit"),
        features_reference=_weak_reference(fitted.features, "Rademacher features"),
    )
    _register_identity(_RADEMACHER_PREPARED_REGISTRY, prepared, record)


def _require_rademacher_problem(
    prepared: PreparedM2,
    candidates: CandidateLibrary,
    geometry: GeometryEngine,
    pairs: PairFeatureLibrary,
) -> None:
    transform_record = _require_rademacher_transform_record(prepared.transform)
    module = sys.modules.get(_RADEMACHER_MODULE_NAME)
    exact_features = False
    if module is not None:
        module = _exact_rademacher_module()
        exact_features = type(pairs) is module.RademacherFeatureLibrary
    if transform_record is None:
        if exact_features:
            raise M2Error("Rademacher features require a registered Rademacher fold")
        return
    assert module is not None
    record = _lookup_identity(_RADEMACHER_PREPARED_REGISTRY, prepared)
    if record is None or prepared is not record.prepared_reference():
        raise M2Error("Rademacher PreparedM2 lacks exact registered provenance")
    fitted = record.fitted_reference()
    features = record.features_reference()
    if (
        fitted is None
        or features is None
        or type(pairs) is not module.RademacherFeatureLibrary
        or pairs is not features
        or prepared.transform is not record.transform_reference()
        or fitted.transform is not prepared.transform
    ):
        raise M2Error("Rademacher fold, preparation, or dictionary identity changed")
    authenticate = _exact_rademacher_function(module, "_core_authenticate_problem")
    if authenticate(fitted, prepared, candidates, geometry, pairs) is not True:
        raise M2Error("Rademacher problem failed exact fold/dictionary authentication")


def fit_residual_transform(
    cells: Sequence[Cell],
    moments: Sequence[UniformCellMoments],
    candidates: CandidateLibrary,
    geometry: GeometryEngine,
    pairs: PairFeatureLibrary,
    indexes: Sequence[int] | None = None,
    *,
    expected_rank: int | None = None,
) -> tuple[ResidualTransform, PreparedM2]:
    """Fit the training-only uniform-Omega F projection and residual scaling."""
    if len(cells) != len(moments):
        raise M2Error("cells and M1 uniform moments are not aligned")
    selected = (
        tuple(range(len(cells)))
        if indexes is None
        else tuple(int(index) for index in indexes)
    )
    if not selected or len(set(selected)) != len(selected):
        raise M2Error("training indexes must be nonempty and unique")
    if any(index < 0 or index >= len(cells) for index in selected):
        raise M2Error("training index is outside the supplied cells")

    gram_ff, prepared_m1 = m1.fold_gram(cells, moments, selected)
    gram_fh = np.zeros((m1.PARAMETER_DIMENSION, PAIR_DIMENSION), dtype=float)
    gram_hh = np.zeros((PAIR_DIMENSION, PAIR_DIMENSION), dtype=float)

    for local_index, cell in enumerate(prepared_m1.cells):
        weight = float(prepared_m1.weights[local_index])
        statistics = pairs.statistics(cell.k)
        x_row = prepared_m1.x[local_index]
        target_cross = np.einsum(
            "a,rp->arp",
            x_row,
            statistics.contrast_cross_covariance,
            optimize=True,
        ).reshape(m1.B_PARAMETER_DIMENSION, PAIR_DIMENSION)
        gram_fh[: m1.B_PARAMETER_DIMENSION] += weight * target_cross

        basis = candidates.basis(cell.k)
        geometry_cross = np.zeros((m1.GEOMETRY_DIMENSION, PAIR_DIMENSION), dtype=float)
        for start in range(0, basis.count, m1.BATCH_MAXIMUM):
            stop = min(start + m1.BATCH_MAXIMUM, basis.count)
            raw = geometry.raw_batch(cell, start, stop)
            z = (
                raw - prepared_m1.geometry_centers[local_index]
            ) / prepared_m1.geometry_scales
            h_centered = (
                pairs.basis(cell.k).indicators[start:stop].astype(float, copy=False)
                - statistics.mean
            )
            geometry_cross += z.T @ h_centered
        geometry_cross /= float(basis.count)
        gram_fh[m1.B_PARAMETER_DIMENSION :] += weight * geometry_cross
        gram_hh += weight * statistics.covariance

    gram_ff = _finite(0.5 * (gram_ff + gram_ff.T), "G_FF")
    gram_fh = _finite(gram_fh, "G_FH")
    gram_hh = _finite(0.5 * (gram_hh + gram_hh.T), "G_HH")
    inverse_ff, ff_tolerance, ff_rank = _deterministic_pseudoinverse(
        gram_ff, "G_FF"
    )
    projection = inverse_ff @ gram_fh
    orthogonality = gram_fh - gram_ff @ projection
    orthogonality_error = float(np.max(np.abs(orthogonality)))
    if orthogonality_error > PROJECTION_TOLERANCE:
        raise M2Error("uniform residual pair features are not orthogonal to F")

    residual_gram_schur = (
        gram_hh
        - projection.T @ gram_fh
        - gram_fh.T @ projection
        + projection.T @ gram_ff @ projection
    )
    residual_gram_schur = _finite(
        0.5 * (residual_gram_schur + residual_gram_schur.T),
        "Schur residual pair Gram",
    )
    # Accumulate R.T R directly rather than relying on cancellation between
    # the four Schur-complement terms.  This is mathematically identical but
    # keeps exact-null directions at roundoff-squared scale, which is needed
    # for the prospective tau/10,tau,10tau rank-stability diagnostic.
    residual_gram = np.zeros((PAIR_DIMENSION, PAIR_DIMENSION), dtype=float)
    projection_target = projection[: m1.B_PARAMETER_DIMENSION].reshape(
        m1.DESIGN_DIMENSION,
        m1.TARGET_CONTRAST_DIMENSION,
        PAIR_DIMENSION,
    )
    projection_geometry = projection[m1.B_PARAMETER_DIMENSION :]
    for local_index, cell in enumerate(prepared_m1.cells):
        basis = candidates.basis(cell.k)
        statistics = pairs.statistics(cell.k)
        contrast_mean = np.mean(basis.contrasts, axis=0)
        cell_target_projection = np.einsum(
            "a,arp->rp",
            prepared_m1.x[local_index],
            projection_target,
            optimize=True,
        )
        cell_second = np.zeros((PAIR_DIMENSION, PAIR_DIMENSION), dtype=float)
        for start in range(0, basis.count, m1.BATCH_MAXIMUM):
            stop = min(start + m1.BATCH_MAXIMUM, basis.count)
            raw = geometry.raw_batch(cell, start, stop)
            z = (
                raw - prepared_m1.geometry_centers[local_index]
            ) / prepared_m1.geometry_scales
            h_centered = (
                pairs.basis(cell.k).indicators[start:stop].astype(float, copy=False)
                - statistics.mean
            )
            a_centered = basis.contrasts[start:stop] - contrast_mean
            residual = h_centered - (
                a_centered @ cell_target_projection
                + z @ projection_geometry
            )
            cell_second += residual.T @ residual
        residual_gram += (
            float(prepared_m1.weights[local_index])
            * cell_second
            / float(basis.count)
        )
    residual_gram = _finite(
        0.5 * (residual_gram + residual_gram.T), "direct residual pair Gram"
    )
    schur_direct_error = float(
        np.max(np.abs(residual_gram - residual_gram_schur))
    )
    residual_values, _ = _symmetric_eigensystem(residual_gram, "residual pair Gram")
    residual_largest = float(np.max(np.abs(residual_values)))
    residual_tolerance = (
        PAIR_DIMENSION
        * np.finfo(float).eps
        * max(1.0, residual_largest)
    )
    if float(np.min(residual_values)) < -10.0 * residual_tolerance:
        raise M2Error("residual pair Gram is materially indefinite")
    diagonal = np.diag(residual_gram).copy()
    if np.any(diagonal <= residual_tolerance):
        raise M2Error("a physical pair coordinate has zero residual RMS")
    scales = np.sqrt(diagonal)
    scaled_gram = residual_gram / scales[:, None] / scales[None, :]
    scaled_gram = _finite(
        0.5 * (scaled_gram + scaled_gram.T), "scaled residual pair Gram"
    )
    eigenvalues, eigenvectors = _symmetric_eigensystem(
        scaled_gram, "scaled residual pair Gram"
    )
    largest = float(np.max(eigenvalues))
    rank_tolerance = (
        RANK_EPSILON_MULTIPLIER
        * np.finfo(float).eps
        * max(1.0, largest)
    )
    if float(np.min(eigenvalues)) < -10.0 * rank_tolerance:
        raise M2Error("scaled residual pair Gram is materially indefinite")
    ranks = {
        "tau_over_10": int(np.sum(eigenvalues > rank_tolerance / 10.0)),
        "tau": int(np.sum(eigenvalues > rank_tolerance)),
        "ten_tau": int(np.sum(eigenvalues > 10.0 * rank_tolerance)),
    }
    if len(set(ranks.values())) != 1:
        raise M2Error(
            "residual pair numerical rank is unstable across tau/10,tau,10tau: "
            f"{ranks}"
        )
    keep = eigenvalues > rank_tolerance
    range_basis = eigenvectors[:, keep]
    range_projector = range_basis @ range_basis.T
    range_projector = _finite(
        0.5 * (range_projector + range_projector.T), "pair range projector"
    )
    projector_symmetry = float(
        np.max(np.abs(range_projector - range_projector.T))
    )
    projector_idempotence = float(
        np.max(np.abs(range_projector @ range_projector - range_projector))
    )
    if (
        projector_symmetry > PROJECTOR_TOLERANCE
        or projector_idempotence > PROJECTOR_TOLERANCE
    ):
        raise M2Error(
            "pair range projector failed its symmetry/idempotence certificate"
        )
    rank = int(np.sum(keep))
    expected_match = expected_rank is None or rank == int(expected_rank)
    if expected_rank is not None and not expected_match:
        raise M2Error(
            f"residual pair rank {rank} differs from expected rank {expected_rank}"
        )

    diagnostics: dict[str, Any] = {
        "training_cells": len(selected),
        "training_groups": len({cells[index].group for index in selected}),
        "uniform_Omega_centering": True,
        "equal_group_weighting": True,
        "G_FF_rank": ff_rank,
        "G_FF_tolerance": ff_tolerance,
        "G_FH_projection_maximum_absolute_orthogonality_error": orthogonality_error,
        "Schur_vs_direct_residual_Gram_maximum_absolute_error": schur_direct_error,
        "pair_residual_RMS_minimum": float(np.min(scales)),
        "pair_residual_RMS_maximum": float(np.max(scales)),
        "rank_tolerance": rank_tolerance,
        "rank_at_tau_over_10_tau_ten_tau": ranks,
        "rank_stable": True,
        "rank": rank,
        "expected_rank": expected_rank,
        "expected_rank_match": expected_match,
        "range_projector_symmetry_maximum_absolute_error": projector_symmetry,
        "range_projector_idempotence_maximum_absolute_error": projector_idempotence,
    }
    transform = ResidualTransform(
        design_transform=prepared_m1.design_transform,
        geometry_scales=prepared_m1.geometry_scales.copy(),
        projection_f_to_h=_finite(projection, "F-to-H projection"),
        pair_residual_scales=_finite(scales, "pair residual scales"),
        range_basis=_finite(range_basis, "pair range basis"),
        range_projector=range_projector,
        residual_gram=residual_gram,
        scaled_residual_gram=scaled_gram,
        diagnostics=diagnostics,
    )
    prepared = PreparedM2(prepared_m1, transform)
    _register_physical_fold_provenance(
        prepared,
        cells,
        moments,
        selected,
        candidates,
        geometry,
        pairs,
        gram_ff,
        inverse_ff,
        ff_tolerance,
        ff_rank,
    )
    return transform, prepared


def apply_residual_transform(
    cells: Sequence[Cell],
    moments: Sequence[UniformCellMoments],
    transform: ResidualTransform,
) -> PreparedM2:
    """Apply a frozen training transform to validation cells without refitting."""
    if len(cells) != len(moments) or not cells:
        raise M2Error("validation cells and moments must be nonempty and aligned")
    prepared = m1.apply_fixed_transforms(
        cells,
        moments,
        transform.design_transform,
        transform.geometry_scales,
    )
    return PreparedM2(prepared, transform)


def _m1_parameter_blocks(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(theta, dtype=float)
    if vector.shape != (m1.PARAMETER_DIMENSION,) or not np.all(np.isfinite(vector)):
        raise M2Error("fixed M1 parameter vector is malformed or nonfinite")
    return (
        vector[: m1.B_PARAMETER_DIMENSION].reshape(
            m1.DESIGN_DIMENSION, m1.TARGET_CONTRAST_DIMENSION
        ),
        vector[m1.B_PARAMETER_DIMENSION :],
    )


def _range_project_vector(vector: np.ndarray, transform: ResidualTransform) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    if array.shape != (PAIR_DIMENSION,):
        raise M2Error("pair vector has the wrong shape")
    if _require_rademacher_transform_record(transform) is not None:
        return transform.range_projector @ array
    basis = transform.range_basis
    return basis @ (basis.T @ array)


def _range_project_rows(matrix: np.ndarray, transform: ResidualTransform) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[1] != PAIR_DIMENSION:
        raise M2Error("pair-feature batch has the wrong shape")
    if _require_rademacher_transform_record(transform) is not None:
        return array @ transform.range_projector
    basis = transform.range_basis
    return (array @ basis) @ basis.T


def _fixed_m1_score_batch(
    fixed_m1_theta: np.ndarray,
    prepared: PreparedM2,
    cell_index: int,
    start: int,
    stop: int,
    candidates: CandidateLibrary,
    geometry: GeometryEngine,
) -> np.ndarray:
    b_matrix, gamma = _m1_parameter_blocks(fixed_m1_theta)
    cell = prepared.cells[cell_index]
    basis = candidates.basis(cell.k)
    raw = geometry.raw_batch(cell, start, stop)
    z = (
        raw - prepared.m1_data.geometry_centers[cell_index]
    ) / prepared.m1_data.geometry_scales
    target_coefficient = b_matrix.T @ prepared.m1_data.x[cell_index]
    score = (
        basis.contrasts[start:stop] @ target_coefficient
        + z @ gamma
    )
    return _finite(score, "fixed M1 candidate score")


def residual_pair_batch(
    prepared: PreparedM2,
    cell_index: int,
    start: int,
    stop: int,
    candidates: CandidateLibrary,
    geometry: GeometryEngine,
    pairs: PairFeatureLibrary,
) -> np.ndarray:
    """Return frozen, scaled, range-projected residual pair features."""
    _require_rademacher_problem(prepared, candidates, geometry, pairs)
    cell = prepared.cells[cell_index]
    basis = candidates.basis(cell.k)
    if not (0 <= start <= stop <= basis.count) or stop - start > m1.BATCH_MAXIMUM:
        raise M2Error("residual pair batch is outside the frozen bounds")
    statistics = pairs.statistics(cell.k)
    h_centered = (
        pairs.basis(cell.k).indicators[start:stop].astype(float, copy=False)
        - statistics.mean
    )
    contrast_mean = np.mean(basis.contrasts, axis=0)
    a_centered = basis.contrasts[start:stop] - contrast_mean
    raw = geometry.raw_batch(cell, start, stop)
    z = (
        raw - prepared.m1_data.geometry_centers[cell_index]
    ) / prepared.m1_data.geometry_scales

    projection = prepared.transform.projection_f_to_h
    projection_target = projection[: m1.B_PARAMETER_DIMENSION].reshape(
        m1.DESIGN_DIMENSION,
        m1.TARGET_CONTRAST_DIMENSION,
        PAIR_DIMENSION,
    )
    cell_target_projection = np.einsum(
        "a,arp->rp",
        prepared.m1_data.x[cell_index],
        projection_target,
        optimize=True,
    )
    fitted = (
        a_centered @ cell_target_projection
        + z @ projection[m1.B_PARAMETER_DIMENSION :]
    )
    residual = (h_centered - fitted) / prepared.transform.pair_residual_scales
    return _finite(
        _range_project_rows(residual, prepared.transform),
        "scaled residual pair feature",
    )


def _validate_problem_inputs(
    beta: np.ndarray,
    fixed_m1_theta: np.ndarray,
    prepared: PreparedM2,
) -> np.ndarray:
    vector = np.asarray(beta, dtype=float)
    if vector.shape != (PAIR_DIMENSION,) or not np.all(np.isfinite(vector)):
        raise M2Error("M2 beta is malformed or nonfinite")
    _m1_parameter_blocks(fixed_m1_theta)
    if not prepared.cells:
        raise M2Error("M2 evaluation requires at least one cell")
    return vector


def candidate_distribution(
    beta: np.ndarray,
    fixed_m1_theta: np.ndarray,
    prepared: PreparedM2,
    cell_index: int,
    candidates: CandidateLibrary,
    geometry: GeometryEngine,
    pairs: PairFeatureLibrary,
) -> ConditionalDistribution:
    """Exhaustively return one cell's exact fixed-K conditional distribution."""
    vector = _validate_problem_inputs(beta, fixed_m1_theta, prepared)
    _require_rademacher_problem(prepared, candidates, geometry, pairs)
    if not 0 <= int(cell_index) < len(prepared.cells):
        raise M2Error("cell index is outside prepared data")
    cell = prepared.cells[int(cell_index)]
    basis = candidates.basis(cell.k)
    if basis.count == 1:
        return ConditionalDistribution(
            probabilities=np.ones(1, dtype=float),
            log_probabilities=np.zeros(1, dtype=float),
            marginals=basis.membership[0].astype(float),
        )

    beta_is_zero = bool(np.count_nonzero(vector) == 0)
    maximum = -math.inf
    for start in range(0, basis.count, m1.BATCH_MAXIMUM):
        stop = min(start + m1.BATCH_MAXIMUM, basis.count)
        score = _fixed_m1_score_batch(
            fixed_m1_theta, prepared, cell_index, start, stop, candidates, geometry
        )
        if not beta_is_zero:
            score = score + residual_pair_batch(
                prepared, cell_index, start, stop, candidates, geometry, pairs
            ) @ vector
        maximum = max(maximum, float(np.max(score)))
    denominator = 0.0
    stored: list[tuple[int, int, np.ndarray]] = []
    for start in range(0, basis.count, m1.BATCH_MAXIMUM):
        stop = min(start + m1.BATCH_MAXIMUM, basis.count)
        score = _fixed_m1_score_batch(
            fixed_m1_theta, prepared, cell_index, start, stop, candidates, geometry
        )
        if not beta_is_zero:
            score = score + residual_pair_batch(
                prepared, cell_index, start, stop, candidates, geometry, pairs
            ) @ vector
        weights = np.exp(score - maximum)
        denominator += float(np.sum(weights, dtype=float))
        stored.append((start, stop, weights))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise M2Error("exact M2 normalizer is invalid")
    probabilities = np.empty(basis.count, dtype=float)
    for start, stop, weights in stored:
        probabilities[start:stop] = weights / denominator
    log_probabilities = np.log(probabilities)
    marginals = probabilities @ basis.membership
    if not (
        np.all(np.isfinite(probabilities))
        and np.all(np.isfinite(log_probabilities))
        and np.all(np.isfinite(marginals))
    ):
        raise M2Error("conditional probabilities or marginals are nonfinite")
    return ConditionalDistribution(probabilities, log_probabilities, marginals)


def evaluate_m2(
    beta: np.ndarray,
    fixed_m1_theta: np.ndarray,
    prepared: PreparedM2,
    candidates: CandidateLibrary,
    geometry: GeometryEngine,
    pairs: PairFeatureLibrary,
    *,
    need_predictions: bool = False,
    linear_correction: np.ndarray | None = None,
) -> M2Evaluation:
    """Evaluate exact equal-group M2 negative log likelihood and gradient."""
    vector = _validate_problem_inputs(beta, fixed_m1_theta, prepared)
    _require_rademacher_problem(prepared, candidates, geometry, pairs)
    correction: np.ndarray | None = None
    if linear_correction is not None:
        correction = np.asarray(linear_correction, dtype=float)
        if correction.shape != (PAIR_DIMENSION,) or not np.all(np.isfinite(correction)):
            raise M2Error("M2 linear correction is malformed or nonfinite")
    beta_is_zero = bool(np.count_nonzero(vector) == 0)
    objective_terms: list[float] = []
    gradient = np.zeros(PAIR_DIMENSION, dtype=float)
    log_probability = (
        np.zeros(len(prepared.cells), dtype=float) if need_predictions else None
    )
    marginals = (
        np.zeros((len(prepared.cells), m1.PANEL_SIZE), dtype=float)
        if need_predictions
        else None
    )
    candidate_visits = 0

    for cell_index, cell in enumerate(prepared.cells):
        basis = candidates.basis(cell.k)
        if not 0 <= cell.observed_index < basis.count:
            raise M2Error("observed candidate index is outside fixed-K Omega")
        if basis.count == 1:
            candidate_visits += 2
            if need_predictions:
                assert log_probability is not None and marginals is not None
                log_probability[cell_index] = 0.0
                marginals[cell_index] = basis.membership[0]
            continue

        maximum = -math.inf
        for start in range(0, basis.count, m1.BATCH_MAXIMUM):
            stop = min(start + m1.BATCH_MAXIMUM, basis.count)
            score = _fixed_m1_score_batch(
                fixed_m1_theta,
                prepared,
                cell_index,
                start,
                stop,
                candidates,
                geometry,
            )
            if not beta_is_zero:
                score = score + residual_pair_batch(
                    prepared, cell_index, start, stop, candidates, geometry, pairs
                ) @ vector
            maximum = max(maximum, float(np.max(score)))
            candidate_visits += stop - start
        if not math.isfinite(maximum):
            raise M2Error("M2 candidate-score maximum is nonfinite")

        denominator = 0.0
        feature_numerator = np.zeros(PAIR_DIMENSION, dtype=float)
        marginal_numerator = (
            np.zeros(m1.PANEL_SIZE, dtype=float) if need_predictions else None
        )
        observed_score = math.nan
        observed_feature: np.ndarray | None = None
        for start in range(0, basis.count, m1.BATCH_MAXIMUM):
            stop = min(start + m1.BATCH_MAXIMUM, basis.count)
            pair_feature = residual_pair_batch(
                prepared, cell_index, start, stop, candidates, geometry, pairs
            )
            score = _fixed_m1_score_batch(
                fixed_m1_theta,
                prepared,
                cell_index,
                start,
                stop,
                candidates,
                geometry,
            )
            if not beta_is_zero:
                score = score + pair_feature @ vector
            weight = np.exp(score - maximum)
            denominator += float(np.sum(weight, dtype=float))
            feature_numerator += weight @ pair_feature
            if need_predictions:
                assert marginal_numerator is not None
                marginal_numerator += weight @ basis.membership[start:stop]
            if start <= cell.observed_index < stop:
                offset = cell.observed_index - start
                observed_score = float(score[offset])
                observed_feature = pair_feature[offset].copy()
            candidate_visits += stop - start
        if (
            not math.isfinite(denominator)
            or denominator <= 0.0
            or observed_feature is None
            or not math.isfinite(observed_score)
        ):
            raise M2Error("M2 exact normalizer or observed feature is invalid")
        cell_nll = maximum + math.log(denominator) - observed_score
        weight = float(prepared.weights[cell_index])
        objective_terms.append(weight * cell_nll)
        gradient += weight * (feature_numerator / denominator - observed_feature)
        if need_predictions:
            assert (
                log_probability is not None
                and marginals is not None
                and marginal_numerator is not None
            )
            log_probability[cell_index] = -cell_nll
            marginals[cell_index] = marginal_numerator / denominator

    objective = math.fsum(objective_terms)
    if correction is not None:
        objective += float(np.dot(vector, correction))
        gradient += correction
    gradient = _range_project_vector(gradient, prepared.transform)
    if not math.isfinite(objective) or not np.all(np.isfinite(gradient)):
        raise M2Error("M2 objective or gradient is nonfinite")
    return M2Evaluation(
        objective=float(objective),
        gradient=gradient,
        log_probability_nats=log_probability,
        marginals=marginals,
        candidate_visits=candidate_visits,
    )


def population_linear_correction(
    planted_beta: np.ndarray,
    fixed_m1_theta: np.ndarray,
    prepared: PreparedM2,
    candidates: CandidateLibrary,
    geometry: GeometryEngine,
    pairs: PairFeatureLibrary,
) -> np.ndarray:
    """Return a synthetic-only correction making planted probabilities exact."""
    planted = _validate_problem_inputs(planted_beta, fixed_m1_theta, prepared)
    correction = np.zeros(PAIR_DIMENSION, dtype=float)
    for cell_index, cell in enumerate(prepared.cells):
        basis = candidates.basis(cell.k)
        if basis.count == 1:
            continue
        distribution = candidate_distribution(
            planted,
            fixed_m1_theta,
            prepared,
            cell_index,
            candidates,
            geometry,
            pairs,
        )
        expected = np.zeros(PAIR_DIMENSION, dtype=float)
        observed: np.ndarray | None = None
        for start in range(0, basis.count, m1.BATCH_MAXIMUM):
            stop = min(start + m1.BATCH_MAXIMUM, basis.count)
            feature = residual_pair_batch(
                prepared, cell_index, start, stop, candidates, geometry, pairs
            )
            expected += distribution.probabilities[start:stop] @ feature
            if start <= cell.observed_index < stop:
                observed = feature[cell.observed_index - start].copy()
        if observed is None:
            raise M2Error("synthetic population correction missed observed candidate")
        correction += float(prepared.weights[cell_index]) * (observed - expected)
    return _finite(
        _range_project_vector(correction, prepared.transform),
        "synthetic population linear correction",
    )


def compare_conditional_probabilities(
    left_beta: np.ndarray,
    right_beta: np.ndarray,
    fixed_m1_theta: np.ndarray,
    prepared: PreparedM2,
    candidates: CandidateLibrary,
    geometry: GeometryEngine,
    pairs: PairFeatureLibrary,
) -> dict[str, float]:
    """Compare identified predictions rather than non-unique raw coefficients."""
    kl_terms: list[float] = []
    probability_errors: list[float] = []
    marginal_errors: list[float] = []
    for cell_index in range(len(prepared.cells)):
        left = candidate_distribution(
            left_beta, fixed_m1_theta, prepared, cell_index, candidates, geometry, pairs
        )
        right = candidate_distribution(
            right_beta, fixed_m1_theta, prepared, cell_index, candidates, geometry, pairs
        )
        kl_terms.append(
            float(
                np.sum(
                    left.probabilities
                    * (left.log_probabilities - right.log_probabilities)
                )
            )
        )
        probability_errors.append(
            float(np.max(np.abs(left.probabilities - right.probabilities)))
        )
        marginal_errors.append(float(np.mean(np.abs(left.marginals - right.marginals))))
    return {
        "mean_KL_nats": math.fsum(kl_terms) / len(kl_terms),
        "maximum_candidate_probability_absolute_error": max(probability_errors),
        "mean_inclusion_marginal_absolute_error": math.fsum(marginal_errors)
        / len(marginal_errors),
    }


def penalty_value(beta: np.ndarray, specification: PenaltySpec) -> float:
    """Return J(beta), excluding the scalar penalty strength."""
    vector = np.asarray(beta, dtype=float)
    if vector.shape != (PAIR_DIMENSION,) or not np.all(np.isfinite(vector)):
        raise M2Error("penalty vector is malformed or nonfinite")
    if specification.kind == "lasso":
        value = float(np.sum(np.abs(vector), dtype=float))
    elif specification.kind == "elastic_net":
        alpha = float(specification.l1_ratio)
        value = alpha * float(np.sum(np.abs(vector), dtype=float))
        value += 0.5 * (1.0 - alpha) * float(np.dot(vector, vector))
    else:
        components = [
            math.sqrt(float(len(indexes))) * float(np.linalg.norm(vector[list(indexes)]))
            for _, indexes in BILATERAL_GROUPS
        ]
        value = math.fsum(components)
    if not math.isfinite(value):
        raise M2Error("penalty value is nonfinite")
    return value


def _soft_threshold(vector: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(vector) * np.maximum(np.abs(vector) - threshold, 0.0)


def _penalty_prox_unconstrained(
    vector: np.ndarray,
    amount: float,
    specification: PenaltySpec,
) -> np.ndarray:
    if not math.isfinite(float(amount)) or amount < 0.0:
        raise M2Error("proximal penalty amount must be finite and nonnegative")
    source = _finite(np.asarray(vector, dtype=float), "proximal source")
    if source.shape != (PAIR_DIMENSION,):
        raise M2Error("proximal source has the wrong shape")
    if amount == 0.0:
        return source.copy()
    if specification.kind == "lasso":
        result = _soft_threshold(source, amount)
    elif specification.kind == "elastic_net":
        alpha = float(specification.l1_ratio)
        result = _soft_threshold(source, amount * alpha) / (
            1.0 + amount * (1.0 - alpha)
        )
    else:
        result = np.zeros_like(source)
        for _, indexes in BILATERAL_GROUPS:
            index = np.asarray(indexes, dtype=int)
            block = source[index]
            norm = float(np.linalg.norm(block))
            threshold = amount * math.sqrt(float(len(indexes)))
            if norm > threshold:
                result[index] = (1.0 - threshold / norm) * block
    return _finite(result, "unconstrained penalty proximal result")


def _prox_penalty_range_with_state(
    vector: np.ndarray,
    amount: float,
    specification: PenaltySpec,
    transform: ResidualTransform,
    *,
    tolerance: float = PROX_TOLERANCE,
    maximum_iterations: int = PROX_MAXIMUM_ITERATIONS,
    warm_state: _DykstraWarmState | None = None,
) -> tuple[ProxResult, _DykstraWarmState | None, bool]:
    """Dykstra prox plus private deterministic dual-correction warm state."""
    source = np.asarray(vector, dtype=float)
    if source.shape != (PAIR_DIMENSION,) or not np.all(np.isfinite(source)):
        raise M2Error("Dykstra source is malformed or nonfinite")
    if not math.isfinite(float(tolerance)) or tolerance <= 0.0:
        raise M2Error("Dykstra tolerance must be finite and positive")
    if maximum_iterations <= 0:
        raise M2Error("Dykstra maximum iterations must be positive")
    # The frozen Dykstra stopping rule is absolute in the physical
    # coefficient coordinates.  Scaling it by the source magnitude would make
    # an identical iterate pass or fail solely because the caller rescaled the
    # problem, and would no longer implement the declared 1e-12 tolerance.
    threshold = float(tolerance)

    if amount == 0.0:
        result = _range_project_vector(source, transform)
        range_residual = float(
            np.linalg.norm(result - _range_project_vector(result, transform), ord=np.inf)
        )
        return (
            ProxResult(
                result, 1, 0.0, range_residual, 0.0, range_residual <= threshold
            ),
            None,
            False,
        )

    if transform.rank == PAIR_DIMENSION:
        result = _penalty_prox_unconstrained(source, amount, specification)
        range_residual = float(
            np.linalg.norm(result - _range_project_vector(result, transform), ord=np.inf)
        )
        certified = bool(range_residual <= threshold)
        if not certified:
            raise M2Error("full-rank proximal result failed range certification")
        return ProxResult(result, 1, 0.0, range_residual, 0.0, True), None, False

    warm_started = bool(
        warm_state is not None
        and warm_state.amount == float(amount)
        and warm_state.penalty_kind == specification.kind
        and warm_state.l1_ratio == float(specification.l1_ratio)
    )
    if warm_started:
        assert warm_state is not None
        penalty_correction = np.asarray(
            warm_state.penalty_correction, dtype=float
        ).copy()
        range_correction = np.asarray(warm_state.range_correction, dtype=float).copy()
        if (
            penalty_correction.shape != (PAIR_DIMENSION,)
            or range_correction.shape != (PAIR_DIMENSION,)
            or not np.all(np.isfinite(penalty_correction))
            or not np.all(np.isfinite(range_correction))
        ):
            raise M2Error("Dykstra warm corrections are malformed or nonfinite")
        # Keep source = primal + both dual corrections when the proximal
        # source changes between outer iterations.
        x = source - penalty_correction - range_correction
    else:
        x = source.copy()
        penalty_correction = np.zeros(PAIR_DIMENSION, dtype=float)
        range_correction = np.zeros(PAIR_DIMENSION, dtype=float)
    fixed_residual = math.inf
    range_residual = math.inf
    invariant_residual = math.inf
    for iteration in range(1, maximum_iterations + 1):
        penalty_input = x + penalty_correction
        y = _penalty_prox_unconstrained(
            penalty_input, amount, specification
        )
        new_penalty_correction = penalty_input - y
        range_input = y + range_correction
        new_x = _range_project_vector(range_input, transform)
        new_range_correction = range_input - new_x

        fixed_residual = float(np.linalg.norm(new_x - y, ord=np.inf))
        change = float(np.linalg.norm(new_x - x, ord=np.inf))
        range_residual = float(
            np.linalg.norm(
                new_x - _range_project_vector(new_x, transform), ord=np.inf
            )
        )
        invariant_residual = float(
            np.linalg.norm(
                source
                - new_x
                - new_penalty_correction
                - new_range_correction,
                ord=np.inf,
            )
        )
        if not all(
            math.isfinite(value)
            for value in (
                fixed_residual,
                change,
                range_residual,
                invariant_residual,
            )
        ):
            raise M2Error("Dykstra proximal iteration became nonfinite")
        x = new_x
        penalty_correction = new_penalty_correction
        range_correction = new_range_correction
        if (
            fixed_residual <= threshold
            and change <= threshold
            and range_residual <= threshold
            and invariant_residual <= threshold
        ):
            state = _DykstraWarmState(
                amount=float(amount),
                penalty_kind=specification.kind,
                l1_ratio=float(specification.l1_ratio),
                penalty_correction=_finite(
                    penalty_correction.copy(), "Dykstra penalty correction"
                ),
                range_correction=_finite(
                    range_correction.copy(), "Dykstra range correction"
                ),
            )
            return (
                ProxResult(
                    _finite(x, "Dykstra proximal result"),
                    iteration,
                    fixed_residual,
                    range_residual,
                    invariant_residual,
                    True,
                ),
                state,
                warm_started,
            )
    raise M2Error(
        "Dykstra proximal map did not converge and certify within the iteration cap: "
        f"fixed={fixed_residual:.17g}, range={range_residual:.17g}, "
        f"invariant={invariant_residual:.17g}, threshold={threshold:.17g}"
    )


def prox_penalty_range(
    vector: np.ndarray,
    amount: float,
    specification: PenaltySpec,
    transform: ResidualTransform,
    *,
    tolerance: float = PROX_TOLERANCE,
    maximum_iterations: int = PROX_MAXIMUM_ITERATIONS,
) -> ProxResult:
    """Dykstra prox of ``amount*J + indicator_Range(P)`` with certification."""
    result, _, _ = _prox_penalty_range_with_state(
        vector,
        amount,
        specification,
        transform,
        tolerance=tolerance,
        maximum_iterations=maximum_iterations,
    )
    return result


def _zero_kkt_residual(
    gradient_at_zero: np.ndarray,
    strength: float,
    shape: PenaltySpec,
    transform: ResidualTransform,
) -> tuple[float, ProxResult]:
    specification = PenaltySpec(shape.kind, strength, shape.l1_ratio)
    prox = prox_penalty_range(
        -np.asarray(gradient_at_zero, dtype=float),
        strength,
        specification,
        transform,
    )
    residual = float(np.linalg.norm(prox.vector, ord=np.inf))
    return residual, prox


def _dual_ball_projection(
    vector: np.ndarray,
    specification: PenaltySpec,
) -> np.ndarray:
    """Project onto ``strength * partial J(0)`` in physical edge coordinates."""
    source = np.asarray(vector, dtype=float)
    if source.shape != (PAIR_DIMENSION,) or not np.all(np.isfinite(source)):
        raise M2Error("zero-KKT dual source is malformed or nonfinite")
    if specification.kind == "lasso":
        radius = float(specification.strength)
        result = np.clip(source, -radius, radius)
    elif specification.kind == "elastic_net":
        # The smooth L2 component has zero derivative at beta=0.
        radius = float(specification.strength) * float(specification.l1_ratio)
        result = np.clip(source, -radius, radius)
    else:
        result = np.zeros_like(source)
        for _, indexes in BILATERAL_GROUPS:
            index = np.asarray(indexes, dtype=int)
            block = source[index]
            norm = float(math.hypot(*(float(value) for value in block)))
            norm_upper = (
                math.nextafter(norm, math.inf) if norm > 0.0 else 0.0
            )
            radius = float(specification.strength) * math.sqrt(float(len(indexes)))
            if norm_upper <= radius or norm == 0.0:
                result[index] = block
            elif radius > 0.0:
                # Round boundary projections inward.  A dual vector a few ulps
                # outside its group ball cannot be used as a feasible witness.
                scale = math.nextafter(radius / norm_upper, 0.0)
                projected = scale * block
                for _ in range(8):
                    projected_norm = float(
                        math.hypot(*(float(value) for value in projected))
                    )
                    projected_norm_upper = (
                        math.nextafter(projected_norm, math.inf)
                        if projected_norm > 0.0
                        else 0.0
                    )
                    if projected_norm_upper <= radius:
                        break
                    projected = np.nextafter(projected, np.zeros_like(projected))
                else:
                    raise M2Error(
                        "zero-KKT group-ball projection did not round inward"
                    )
                result[index] = projected
    return _finite(result, "zero-KKT dual-ball projection")


def _dual_ball_support(
    direction: np.ndarray,
    specification: PenaltySpec,
) -> float:
    """Support function of ``strength * partial J(0)``."""
    vector = np.asarray(direction, dtype=float)
    if vector.shape != (PAIR_DIMENSION,) or not np.all(np.isfinite(vector)):
        raise M2Error("zero-KKT separating direction is malformed or nonfinite")
    if specification.kind == "lasso":
        support = float(specification.strength) * float(np.sum(np.abs(vector)))
    elif specification.kind == "elastic_net":
        support = (
            float(specification.strength)
            * float(specification.l1_ratio)
            * float(np.sum(np.abs(vector)))
        )
    else:
        support = 0.0
        for _, indexes in BILATERAL_GROUPS:
            index = np.asarray(indexes, dtype=int)
            support += (
                float(specification.strength)
                * math.sqrt(float(len(indexes)))
                * float(np.linalg.norm(vector[index]))
            )
    if not math.isfinite(support) or support < 0.0:
        raise M2Error("zero-KKT dual-ball support is invalid")
    return float(support)


def _zero_kkt_separation_margin(
    source: np.ndarray,
    dual: np.ndarray,
    specification: PenaltySpec,
    transform: ResidualTransform,
    decision_tolerance: float,
    *,
    refine_active_normal: bool,
    additional_direction: np.ndarray | None = None,
    additional_direction_diagnostics: dict[str, float] | None = None,
) -> float:
    """Return a guarded lower-bound margin for infinity-norm infeasibility.

    For any nonzero ``y`` in Range(P), positivity of
    ``y @ source - support_C(y) - delta*||y||_1`` proves that the infinity
    distance from ``source`` to ``C + Null(P)`` exceeds ``delta``.  The
    returned value is a binary64-error-guarded lower bound on that expression.
    """
    range_residual = _range_project_vector(source - dual, transform)
    basis = transform.range_basis

    def guarded_margin(
        candidate: np.ndarray,
        *,
        use_negative_prefilter: bool = True,
        diagnostics: dict[str, float] | None = None,
    ) -> float:
        # Represent the direction by stored range-basis coefficients.  In exact
        # real arithmetic B@a is therefore in Range(B); bound the binary64
        # matrix-vector error coordinatewise instead of assuming the rounded
        # vector is literally in the span.
        if basis.shape[1] == 0:
            return -math.inf
        rough_direction = np.asarray(candidate, dtype=float)
        rough_l1 = float(np.sum(np.abs(rough_direction)))
        rough_support = _dual_ball_support(rough_direction, specification)
        rough_inner_product = float(np.dot(source, rough_direction))
        rough_absolute_products = float(
            np.sum(np.abs(source * rough_direction))
        )
        epsilon = np.finfo(float).eps
        reduction_gamma = (
            PAIR_DIMENSION
            * epsilon
            / (1.0 - PAIR_DIMENSION * epsilon)
        )
        rough_gap = (
            rough_inner_product
            - rough_support
            - float(decision_tolerance) * rough_l1
        )
        rough_error = 8.0 * reduction_gamma * (
            rough_absolute_products
            + abs(rough_support)
            + float(decision_tolerance) * rough_l1
        )
        # This is only a cheap negative prefilter.  It can never certify
        # infeasibility; every nonnegative or roundoff-adjacent candidate is
        # rebuilt and checked by the conservative calculation below.
        if not math.isfinite(rough_gap) or (
            use_negative_prefilter and rough_gap < -rough_error
        ):
            if diagnostics is not None:
                diagnostics["guarded_lower_margin"] = -math.inf
            return -math.inf

        coefficients = basis.T @ rough_direction
        direction = np.empty(PAIR_DIMENSION, dtype=float)
        coordinate_error = np.empty(PAIR_DIMENSION, dtype=float)
        multiplication_factor = epsilon / (1.0 - epsilon)
        underflow_unit = math.ulp(0.0)
        for coordinate, row in enumerate(basis):
            products = [
                float(left) * float(right)
                for left, right in zip(row, coefficients, strict=True)
            ]
            observed = math.fsum(products)
            absolute_sum = math.fsum(abs(product) for product in products)
            absolute_sum_upper = math.nextafter(
                absolute_sum + 2.0 * math.ulp(absolute_sum), math.inf
            )
            product_error = (
                multiplication_factor * absolute_sum_upper
                + len(products) * underflow_unit
            )
            summation_error = 2.0 * math.ulp(observed)
            direction[coordinate] = observed
            coordinate_error[coordinate] = math.nextafter(
                product_error + summation_error, math.inf
            )
        if (
            not np.all(np.isfinite(direction))
            or not np.all(np.isfinite(coordinate_error))
            or np.count_nonzero(direction) == 0
        ):
            if diagnostics is not None:
                diagnostics["guarded_lower_margin"] = -math.inf
            return -math.inf

        absolute_products = math.fsum(
            abs(float(left) * float(right))
            for left, right in zip(source, direction, strict=True)
        )
        source_inner_product = math.fsum(
            float(left) * float(right)
            for left, right in zip(source, direction, strict=True)
        )
        product_underflow = PAIR_DIMENSION * underflow_unit
        dot_roundoff = (
            4.0 * reduction_gamma * absolute_products + product_underflow
        )
        span_weight = math.fsum(
            abs(float(value)) * float(error)
            for value, error in zip(source, coordinate_error, strict=True)
        )
        span_roundoff = math.nextafter(
            span_weight / (1.0 - reduction_gamma) + product_underflow,
            math.inf,
        )
        dot_error = math.nextafter(dot_roundoff + span_roundoff, math.inf)

        l1_direction = math.fsum(abs(float(value)) for value in direction)
        l1_span_error = math.fsum(float(error) for error in coordinate_error)
        l1_upper = math.nextafter(
            math.nextafter(l1_direction, math.inf)
            + math.nextafter(
                l1_span_error / (1.0 - reduction_gamma), math.inf
            ),
            math.inf,
        )
        if specification.kind == "lasso":
            radius = float(specification.strength)
            support_upper = math.nextafter(radius * l1_upper, math.inf)
        elif specification.kind == "elastic_net":
            radius = float(specification.strength) * float(
                specification.l1_ratio
            )
            support_upper = math.nextafter(radius * l1_upper, math.inf)
        else:
            support_terms: list[float] = []
            for _, indexes in BILATERAL_GROUPS:
                index = np.asarray(indexes, dtype=int)
                block_norm = float(
                    math.hypot(*(float(value) for value in direction[index]))
                )
                block_error = float(
                    math.hypot(*(float(value) for value in coordinate_error[index]))
                )
                block_upper = math.nextafter(
                    math.nextafter(block_norm, math.inf)
                    + math.nextafter(block_error, math.inf),
                    math.inf,
                )
                radius = float(specification.strength) * math.sqrt(
                    float(len(indexes))
                )
                support_terms.append(
                    math.nextafter(radius * block_upper, math.inf)
                )
            support_upper = math.nextafter(math.fsum(support_terms), math.inf)
        decision_upper = math.nextafter(
            float(decision_tolerance) * l1_upper, math.inf
        )
        if not all(
            math.isfinite(value)
            for value in (
                source_inner_product,
                dot_error,
                support_upper,
                decision_upper,
            )
        ):
            if diagnostics is not None:
                diagnostics["guarded_lower_margin"] = -math.inf
            return -math.inf
        # Downward rounding covers the three final subtractions.  A positive
        # value is then a conservative lower bound for
        # y@s - support_C(y) - decision_tolerance*||y||_1.
        margin = math.nextafter(source_inner_product - dot_error, -math.inf)
        margin = math.nextafter(margin - support_upper, -math.inf)
        margin = math.nextafter(margin - decision_upper, -math.inf)
        if diagnostics is not None:
            nominal_support = _dual_ball_support(direction, specification)
            nominal_decision = float(decision_tolerance) * l1_direction
            nominal_gap = (
                source_inner_product - nominal_support - nominal_decision
            )
            diagnostics.update(
                {
                    "nominal_gap": float(nominal_gap),
                    "guarded_lower_margin": float(margin),
                    "guard_loss": float(nominal_gap - margin),
                    "dot_and_span_error": float(dot_error),
                    "support_inflation": float(
                        support_upper - nominal_support
                    ),
                    "decision_inflation": float(
                        decision_upper - nominal_decision
                    ),
                }
            )
        return float(margin)

    margin = guarded_margin(range_residual)
    if additional_direction is not None:
        # A smooth-minimax KKT direction is only approximately in Range(B).
        # Its small raw null component can make the reject-only shortcut above
        # negative even when the exact stored projection B@(B.T@candidate) is
        # a valid separator.  Skip only that shortcut; the same conservative
        # reconstruction and full arithmetic lower bound still gate acceptance.
        margin = max(
            margin,
            guarded_margin(
                additional_direction,
                use_negative_prefilter=False,
                diagnostics=additional_direction_diagnostics,
            ),
        )
    if not refine_active_normal:
        return margin

    if basis.shape[1] == 0:
        return margin
    constraint_rows: list[np.ndarray] = []
    boundary_epsilon = 64.0 * np.finfo(float).eps
    if specification.kind in {"lasso", "elastic_net"}:
        radius = float(specification.strength)
        if specification.kind == "elastic_net":
            radius *= float(specification.l1_ratio)
        boundary_guard = boundary_epsilon * max(1.0, abs(radius))
        inactive = np.flatnonzero(np.abs(dual) < radius - boundary_guard)
        if inactive.size:
            constraint_rows.append(basis[inactive, :])
    else:
        for _, indexes in BILATERAL_GROUPS:
            index = np.asarray(indexes, dtype=int)
            block_basis = basis[index, :]
            block = dual[index]
            block_norm = float(np.linalg.norm(block))
            radius = float(specification.strength) * math.sqrt(float(len(indexes)))
            boundary_guard = boundary_epsilon * max(1.0, abs(radius), block_norm)
            if block_norm < radius - boundary_guard or block_norm == 0.0:
                constraint_rows.append(block_basis)
            else:
                unit = block / block_norm
                constraint_rows.append(
                    block_basis - np.outer(unit, unit @ block_basis)
                )
    if constraint_rows:
        constraints = np.vstack(constraint_rows)
        _, singular_values, right_vectors = np.linalg.svd(
            constraints, full_matrices=True
        )
        singular_maximum = (
            float(singular_values[0]) if singular_values.size else 0.0
        )
        rank_tolerance = (
            max(constraints.shape)
            * np.finfo(float).eps
            * max(1.0, singular_maximum)
        )
        constraint_rank = int(np.count_nonzero(singular_values > rank_tolerance))
        coefficient_null = right_vectors[constraint_rank:, :].T
    else:
        coefficient_null = np.eye(basis.shape[1], dtype=float)
    if coefficient_null.shape[1] == 0:
        return margin
    seed_coefficients = basis.T @ range_residual
    cleaned_coefficients = coefficient_null @ (
        coefficient_null.T @ seed_coefficients
    )
    if float(np.linalg.norm(cleaned_coefficients)) == 0.0:
        cleaned_coefficients = coefficient_null[:, 0]
    cleaned_direction = basis @ cleaned_coefficients
    return max(margin, guarded_margin(cleaned_direction))


def zero_kkt_certificate(
    gradient_at_zero: np.ndarray,
    specification: PenaltySpec,
    transform: ResidualTransform,
    *,
    tolerance: float = PROX_TOLERANCE,
    maximum_iterations: int = PROX_MAXIMUM_ITERATIONS,
    decision_tolerance: float | None = None,
    initial_correction: np.ndarray | None = None,
) -> ZeroKKTCertificate:
    """Certify ``0 in g + strength*dJ(0) + Range(P)^perp``.

    The null-space correction is optimized by monotone accelerated gradient
    on the squared distance to the separable dual ball.  Each return carries
    either a direct feasible witness or a guarded separating-hyperplane
    certificate of infeasibility; stationarity alone is never a certificate.
    """
    gradient = np.asarray(gradient_at_zero, dtype=float)
    if gradient.shape != (PAIR_DIMENSION,) or not np.all(np.isfinite(gradient)):
        raise M2Error("zero-KKT gradient is malformed or nonfinite")
    if not math.isfinite(float(tolerance)) or tolerance <= 0.0:
        raise M2Error("zero-KKT tolerance must be finite and positive")
    if maximum_iterations <= 0:
        raise M2Error("zero-KKT maximum iterations must be positive")
    source = -gradient
    scale = max(1.0, float(np.linalg.norm(source, ord=np.inf)))
    threshold = float(tolerance) * scale
    decision_threshold = (
        threshold if decision_tolerance is None else float(decision_tolerance)
    )
    if not math.isfinite(decision_threshold) or decision_threshold <= 0.0:
        raise M2Error("zero-KKT decision tolerance must be finite and positive")

    def null_project(vector: np.ndarray) -> np.ndarray:
        return vector - _range_project_vector(vector, transform)

    def freeze_correction(vector: np.ndarray) -> tuple[float, ...]:
        return tuple(float(value) for value in vector)

    def evaluate_null(
        correction: np.ndarray,
        *,
        refine_active_normal: bool = False,
        additional_direction: np.ndarray | None = None,
        additional_direction_diagnostics: dict[str, float] | None = None,
    ) -> tuple[float, np.ndarray, float, float, float]:
        dual = _dual_ball_projection(source - correction, specification)
        residual_vector = source - correction - dual
        objective = 0.5 * float(np.dot(residual_vector, residual_vector))
        null_gradient = -null_project(residual_vector)
        # The dual vector itself gives a direct zero-KKT witness: P(source-dual).
        kkt = float(
            np.linalg.norm(
                _range_project_vector(source - dual, transform), ord=np.inf
            )
        )
        stationarity = float(np.linalg.norm(null_gradient, ord=np.inf))
        if kkt <= decision_threshold:
            separation_margin = -math.inf
        else:
            separation_margin = _zero_kkt_separation_margin(
                source,
                dual,
                specification,
                transform,
                decision_threshold,
                refine_active_normal=refine_active_normal,
                additional_direction=additional_direction,
                additional_direction_diagnostics=(
                    additional_direction_diagnostics
                ),
            )
        return objective, null_gradient, kkt, stationarity, separation_margin

    if initial_correction is None:
        correction = np.zeros(PAIR_DIMENSION, dtype=float)
    else:
        supplied_correction = np.asarray(initial_correction, dtype=float)
        if (
            supplied_correction.shape != (PAIR_DIMENSION,)
            or not np.all(np.isfinite(supplied_correction))
        ):
            raise M2Error("zero-KKT initial correction is malformed or nonfinite")
        correction = null_project(supplied_correction)
    extrapolated = correction.copy()
    acceleration = 1.0
    (
        current_objective,
        _,
        current_kkt,
        current_stationarity,
        current_separation_margin,
    ) = evaluate_null(correction)
    if current_kkt <= decision_threshold:
        return ZeroKKTCertificate(
            current_kkt,
            current_stationarity,
            0,
            True,
            True,
            0.0,
            freeze_correction(correction),
        )
    if current_separation_margin > 0.0:
        return ZeroKKTCertificate(
            current_kkt,
            current_stationarity,
            0,
            True,
            False,
            current_separation_margin,
            freeze_correction(correction),
        )
    last_relative_change = math.inf
    minimax_iterations = 0
    minimax_last_maximum_constraint = math.nan
    minimax_last_smoothing = math.nan
    minimax_last_gradient_infinity = math.nan
    minimax_last_separator_null_infinity = math.nan
    minimax_last_projected_separator_gap = math.nan
    minimax_last_projected_separator_gap_per_l1 = math.nan
    minimax_last_affine_range_residual = math.nan
    minimax_last_affine_ball_violation = math.nan
    minimax_last_affine_ball_violation_guarded_upper = math.nan
    minimax_last_maximum_weight = math.nan
    minimax_last_effective_constraint_count = math.nan
    minimax_last_guarded_lower_margin = math.nan
    minimax_last_nominal_gap = math.nan
    minimax_last_guard_loss = math.nan
    minimax_last_dot_and_span_error = math.nan
    minimax_last_support_inflation = math.nan
    minimax_last_decision_inflation = math.nan
    minimax_last_combined_guarded_margin = math.nan
    minimax_active_face_growths = 0
    minimax_active_face_qp_events = 0
    minimax_active_face_maximum_size = 0
    minimax_active_face_best_null_infinity = math.inf
    minimax_active_face_best_guarded_margin = -math.inf
    minimax_active_face_last_support_size = 0
    minimax_active_face_last_maximum_weight = 0.0
    minimax_active_face_last_effective_constraint_count = 0.0
    fallback_iteration = 0
    accelerated_iterations = min(
        maximum_iterations, ZERO_KKT_ACCELERATED_MAXIMUM_ITERATIONS
    )
    for iteration in range(1, accelerated_iterations + 1):
        accepted = False
        for restart_attempt in range(2):
            (
                extrapolated_objective,
                extrapolated_gradient,
                _,
                _,
                _,
            ) = evaluate_null(extrapolated)
            trial = null_project(extrapolated - extrapolated_gradient)
            (
                trial_objective,
                _,
                trial_kkt,
                trial_stationarity,
                trial_separation_margin,
            ) = evaluate_null(trial)
            monotone_tolerance = 1.0e-15 * max(1.0, abs(current_objective))
            if trial_objective <= current_objective + monotone_tolerance:
                accepted = True
                break
            if restart_attempt == 0:
                extrapolated = correction.copy()
                acceleration = 1.0
        if not accepted:
            raise M2Error("zero-KKT dual solver failed its monotone restart")
        last_relative_change = abs(current_objective - trial_objective) / max(
            1.0, abs(current_objective), abs(trial_objective)
        )
        if trial_kkt <= decision_threshold:
            return ZeroKKTCertificate(
                trial_kkt,
                trial_stationarity,
                iteration,
                True,
                True,
                0.0,
                freeze_correction(trial),
            )
        if trial_separation_margin > 0.0:
            return ZeroKKTCertificate(
                trial_kkt,
                trial_stationarity,
                iteration,
                True,
                False,
                trial_separation_margin,
                freeze_correction(trial),
            )
        next_acceleration = 0.5 * (
            1.0 + math.sqrt(1.0 + 4.0 * acceleration * acceleration)
        )
        next_extrapolated = trial + (
            (acceleration - 1.0) / next_acceleration
        ) * (trial - correction)
        correction = trial
        extrapolated = null_project(next_extrapolated)
        acceleration = next_acceleration
        current_objective = trial_objective
        current_kkt = trial_kkt
        current_stationarity = trial_stationarity
        current_separation_margin = trial_separation_margin

    remaining_iterations = maximum_iterations - accelerated_iterations
    basis = transform.range_basis
    if remaining_iterations > 0 and basis.shape[1] < PAIR_DIMENSION:
        _, _, right_vectors = np.linalg.svd(basis.T, full_matrices=True)
        null_basis = right_vectors[basis.shape[1] :, :].T
        coefficients = null_basis.T @ correction

        # Near a tangential intersection of the affine null-space translate and
        # the separable dual ball, squared-distance iterations can be arbitrarily
        # slow.  A dimensionless smooth minimax phase solves the equivalent
        # convex quadratic-constraint problem in the (small) null coordinates.
        # Its iterate never certifies anything by itself: feasibility is still
        # checked by a direct dual witness, and its softmax KKT direction is
        # passed through the same guarded range-space separator used above.
        if specification.kind in {"lasso", "elastic_net"}:
            constraint_indexes = tuple(
                np.asarray((index,), dtype=int)
                for index in range(PAIR_DIMENSION)
            )
            scalar_radius = float(specification.strength)
            if specification.kind == "elastic_net":
                scalar_radius *= float(specification.l1_ratio)
            constraint_radii = tuple(
                scalar_radius for _ in constraint_indexes
            )
        else:
            constraint_indexes = tuple(
                np.asarray(indexes, dtype=int)
                for _, indexes in BILATERAL_GROUPS
            )
            constraint_radii = tuple(
                float(specification.strength) * math.sqrt(float(len(indexes)))
                for indexes in constraint_indexes
            )
        minimax_scale = max(
            float(np.linalg.norm(source, ord=np.inf)),
            max(constraint_radii, default=0.0),
            np.finfo(float).tiny,
        )
        scaled_source = source / minimax_scale
        scaled_radii = tuple(
            radius / minimax_scale for radius in constraint_radii
        )
        constraint_null_blocks = tuple(
            null_basis[index, :] for index in constraint_indexes
        )
        constraint_hessians = tuple(
            2.0 * (block.T @ block) for block in constraint_null_blocks
        )
        minimax_coordinates = coefficients / minimax_scale
        minimax_budget = min(
            ZERO_KKT_MINIMAX_MAXIMUM_ITERATIONS, remaining_iterations
        )

        def minimax_state(
            coordinates: np.ndarray,
            smoothing: float,
        ) -> tuple[
            float,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]:
            scaled_dual = scaled_source - null_basis @ coordinates
            constraints: list[float] = []
            gradients: list[np.ndarray] = []
            for index, radius, block in zip(
                constraint_indexes,
                scaled_radii,
                constraint_null_blocks,
                strict=True,
            ):
                values = scaled_dual[index]
                constraints.append(float(np.dot(values, values)) - radius * radius)
                gradients.append(-2.0 * (block.T @ values))
            constraint_values = np.asarray(constraints, dtype=float)
            gradient_rows = np.asarray(gradients, dtype=float)
            shift = float(np.max(constraint_values))
            exponentials = np.exp((constraint_values - shift) / smoothing)
            weights = exponentials / float(np.sum(exponentials))
            gradient_value = weights @ gradient_rows
            hessian_value = np.zeros(
                (null_basis.shape[1], null_basis.shape[1]), dtype=float
            )
            second_moment = np.zeros_like(hessian_value)
            for weight, hessian, row in zip(
                weights,
                constraint_hessians,
                gradient_rows,
                strict=True,
            ):
                hessian_value += float(weight) * hessian
                second_moment += float(weight) * np.outer(row, row)
            hessian_value += (
                second_moment - np.outer(gradient_value, gradient_value)
            ) / smoothing
            hessian_value = 0.5 * (hessian_value + hessian_value.T)
            objective_value = shift + smoothing * math.log(
                float(np.sum(exponentials))
            )
            separating_direction = np.zeros(PAIR_DIMENSION, dtype=float)
            for weight, index in zip(weights, constraint_indexes, strict=True):
                separating_direction[index] = float(weight) * scaled_dual[index]
            return (
                float(objective_value),
                constraint_values,
                gradient_value,
                hessian_value,
                scaled_dual,
                separating_direction,
            )

        def minimum_norm_simplex_weights(
            rows: np.ndarray,
            original_indexes: np.ndarray,
            initial_weights: np.ndarray | None,
            event_budget: int,
        ) -> tuple[np.ndarray, int, bool]:
            """Deterministic Wolfe active set for a minimum-norm mixture."""
            face_rows = np.asarray(rows, dtype=float)
            face_original = np.asarray(original_indexes, dtype=int)
            count = int(face_rows.shape[0])
            if (
                face_rows.ndim != 2
                or count == 0
                or face_original.shape != (count,)
                or not np.all(np.isfinite(face_rows))
                or event_budget <= 0
            ):
                return np.zeros(max(count, 0), dtype=float), 0, False
            gram = face_rows @ face_rows.T
            gram = 0.5 * (gram + gram.T)
            if not np.all(np.isfinite(gram)):
                return np.zeros(count, dtype=float), 0, False

            weights = np.zeros(count, dtype=float)
            events = 0
            if initial_weights is not None:
                supplied = np.asarray(initial_weights, dtype=float)
                if supplied.shape == (count,) and np.all(np.isfinite(supplied)):
                    supplied = np.maximum(supplied, 0.0)
                    supplied_sum = math.fsum(float(value) for value in supplied)
                    if supplied_sum > 0.0:
                        weights = supplied / supplied_sum
            if np.count_nonzero(weights) == 0:
                row_norms = np.einsum("ij,ij->i", face_rows, face_rows)
                first = min(
                    range(count),
                    key=lambda index: (
                        float(row_norms[index]),
                        int(face_original[index]),
                    ),
                )
                weights[first] = 1.0
                events += 1
                if events > event_budget:
                    return weights, event_budget, False

            seen_supports: set[tuple[int, ...]] = set()
            epsilon = np.finfo(float).eps
            while events < event_budget:
                support = tuple(
                    int(index) for index in np.flatnonzero(weights > 0.0)
                )
                if not support or support in seen_supports:
                    return weights, events, False
                seen_supports.add(support)
                active = np.asarray(support, dtype=int)
                active_gram = gram[np.ix_(active, active)]
                size = int(active.size)
                kkt_matrix = np.empty((size + 1, size + 1), dtype=float)
                kkt_matrix[:size, :size] = active_gram
                kkt_matrix[:size, size] = 1.0
                kkt_matrix[size, :size] = 1.0
                kkt_matrix[size, size] = 0.0
                right_hand_side = np.zeros(size + 1, dtype=float)
                right_hand_side[size] = 1.0
                events += 1
                solution, _, _, _ = np.linalg.lstsq(
                    kkt_matrix, right_hand_side, rcond=None
                )
                if not np.all(np.isfinite(solution)):
                    return weights, events, False
                backward_residual = float(
                    np.linalg.norm(
                        kkt_matrix @ solution - right_hand_side,
                        ord=np.inf,
                    )
                )
                backward_scale = max(
                    1.0,
                    float(np.linalg.norm(kkt_matrix, ord=np.inf))
                    * float(np.linalg.norm(solution, ord=np.inf)),
                )
                if backward_residual > (
                    ZERO_KKT_ACTIVE_FACE_BACKWARD_ERROR_MULTIPLIER
                    * epsilon
                    * backward_scale
                ):
                    return weights, events, False
                affine_weights = solution[:size]
                weight_tolerance = (
                    ZERO_KKT_ACTIVE_FACE_ROUNDOFF_MULTIPLIER
                    * epsilon
                    * max(
                    1.0, float(np.max(np.abs(affine_weights)))
                    )
                )
                if float(np.min(affine_weights)) < -weight_tolerance:
                    current = weights[active]
                    negative = np.flatnonzero(affine_weights < 0.0)
                    steps = [
                        float(current[index])
                        / float(current[index] - affine_weights[index])
                        for index in negative
                    ]
                    boundary_step = min(steps)
                    if (
                        not math.isfinite(boundary_step)
                        or boundary_step < 0.0
                        or boundary_step > 1.0
                    ):
                        return weights, events, False
                    boundary_weights = current + boundary_step * (
                        affine_weights - current
                    )
                    next_weights = np.zeros(count, dtype=float)
                    next_weights[active] = np.maximum(boundary_weights, 0.0)
                    drop_tolerance = (
                        ZERO_KKT_ACTIVE_FACE_ROUNDOFF_MULTIPLIER
                        * epsilon
                        * max(
                            1.0, float(np.max(np.abs(next_weights)))
                        )
                    )
                    dropped = active[
                        next_weights[active] <= drop_tolerance
                    ]
                    if dropped.size == 0:
                        return weights, events, False
                    if events + int(dropped.size) > event_budget:
                        return weights, event_budget, False
                    next_weights[dropped] = 0.0
                    events += int(dropped.size)
                    weight_sum = math.fsum(
                        float(value) for value in next_weights
                    )
                    if weight_sum <= 0.0 or not math.isfinite(weight_sum):
                        return weights, events, False
                    weights = next_weights / weight_sum
                    continue

                polished = np.maximum(affine_weights, 0.0)
                polished_sum = math.fsum(float(value) for value in polished)
                if polished_sum <= 0.0 or not math.isfinite(polished_sum):
                    return weights, events, False
                polished /= polished_sum
                next_weights = np.zeros(count, dtype=float)
                next_weights[active] = polished
                zeroed = active[next_weights[active] <= weight_tolerance]
                if zeroed.size:
                    if events + int(zeroed.size) > event_budget:
                        return weights, event_budget, False
                    next_weights[zeroed] = 0.0
                    events += int(zeroed.size)
                    weight_sum = math.fsum(
                        float(value) for value in next_weights
                    )
                    if weight_sum <= 0.0 or not math.isfinite(weight_sum):
                        return weights, events, False
                    weights = next_weights / weight_sum
                    continue
                weights = next_weights

                point = weights @ face_rows
                directional_values = face_rows @ point
                nu = float(np.dot(point, point))
                qp_tolerance = (
                    ZERO_KKT_ACTIVE_FACE_ROUNDOFF_MULTIPLIER
                    * epsilon
                    * max(
                        1.0,
                        float(np.linalg.norm(gram, ord=np.inf)),
                        abs(nu),
                    )
                )
                excluded = np.flatnonzero(weights == 0.0)
                violating = [
                    int(index)
                    for index in excluded
                    if float(directional_values[index]) < nu - qp_tolerance
                ]
                if not violating:
                    return weights, events, True
                entering = min(
                    violating,
                    key=lambda index: (
                        float(directional_values[index]),
                        int(face_original[index]),
                    ),
                )
                if events + 1 > event_budget:
                    return weights, event_budget, False
                # A zero-weight entry denotes membership in the next corral.
                # Give it the smallest positive binary64 weight so the support
                # signature includes the deterministic entering vertex.
                weights[entering] = math.ulp(0.0)
                weight_sum = math.fsum(float(value) for value in weights)
                weights /= weight_sum
                events += 1
            return weights, events, False

        for smoothing_exponent in range(
            ZERO_KKT_MINIMAX_SMOOTHING_MINIMUM_EXPONENT,
            ZERO_KKT_MINIMAX_SMOOTHING_MAXIMUM_EXPONENT + 1,
        ):
            if minimax_iterations >= minimax_budget:
                break
            smoothing = 10.0 ** (-smoothing_exponent)
            for _ in range(ZERO_KKT_MINIMAX_NEWTON_STEPS_PER_SMOOTHING):
                if minimax_iterations >= minimax_budget:
                    break
                (
                    smooth_objective,
                    _,
                    smooth_gradient,
                    smooth_hessian,
                    _,
                    _,
                ) = minimax_state(minimax_coordinates, smoothing)
                eigenvalues, eigenvectors = np.linalg.eigh(smooth_hessian)
                largest_eigenvalue = float(np.max(np.abs(eigenvalues)))
                eigen_floor = max(
                    null_basis.shape[1]
                    * np.finfo(float).eps
                    * largest_eigenvalue,
                    np.finfo(float).tiny,
                )
                inverse_values = 1.0 / np.maximum(eigenvalues, eigen_floor)
                newton_direction = -(
                    (eigenvectors * inverse_values)
                    @ (eigenvectors.T @ smooth_gradient)
                )
                directional_derivative = float(
                    np.dot(smooth_gradient, newton_direction)
                )
                if (
                    not np.all(np.isfinite(newton_direction))
                    or not math.isfinite(directional_derivative)
                    or directional_derivative >= 0.0
                ):
                    break
                accepted = False
                step = 1.0
                for _ in range(80):
                    trial_coordinates = (
                        minimax_coordinates + step * newton_direction
                    )
                    trial_objective = minimax_state(
                        trial_coordinates, smoothing
                    )[0]
                    if trial_objective <= (
                        smooth_objective
                        + 1.0e-4 * step * directional_derivative
                    ):
                        accepted = True
                        break
                    step *= 0.5
                minimax_iterations += 1
                if not accepted:
                    break
                minimax_coordinates = trial_coordinates
                if float(np.linalg.norm(smooth_gradient, ord=np.inf)) <= (
                    64.0 * np.finfo(float).eps
                ):
                    break

            (
                _,
                minimax_constraints,
                minimax_gradient,
                _,
                minimax_scaled_dual,
                minimax_separator,
            ) = minimax_state(minimax_coordinates, smoothing)
            minimax_last_maximum_constraint = float(
                np.max(minimax_constraints)
            )
            minimax_last_smoothing = smoothing
            minimax_last_gradient_infinity = float(
                np.linalg.norm(minimax_gradient, ord=np.inf)
            )
            minimax_last_separator_null_infinity = float(
                np.linalg.norm(null_project(minimax_separator), ord=np.inf)
            )
            projected_separator = _range_project_vector(
                minimax_separator, transform
            )
            projected_separator_l1 = float(
                np.sum(np.abs(projected_separator))
            )
            projected_separator_gap_without_decision = (
                math.fsum(
                    float(left) * float(right)
                    for left, right in zip(
                        source, projected_separator, strict=True
                    )
                )
                - _dual_ball_support(projected_separator, specification)
            )
            minimax_last_projected_separator_gap = (
                projected_separator_gap_without_decision
                - decision_threshold * projected_separator_l1
            )
            if projected_separator_l1 > 0.0:
                minimax_last_projected_separator_gap_per_l1 = (
                    projected_separator_gap_without_decision
                    / projected_separator_l1
                )
            affine_candidate = minimax_scale * minimax_scaled_dual
            minimax_last_affine_range_residual = float(
                np.linalg.norm(
                    _range_project_vector(source - affine_candidate, transform),
                    ord=np.inf,
                )
            )
            if specification.kind in {"lasso", "elastic_net"}:
                affine_radius = float(specification.strength)
                if specification.kind == "elastic_net":
                    affine_radius *= float(specification.l1_ratio)
                affine_maximum_absolute = float(
                    np.max(np.abs(affine_candidate))
                )
                minimax_last_affine_ball_violation = float(
                    affine_maximum_absolute - affine_radius
                )
                minimax_last_affine_ball_violation_guarded_upper = (
                    math.nextafter(
                        math.nextafter(affine_maximum_absolute, math.inf)
                        - affine_radius,
                        math.inf,
                    )
                )
            else:
                affine_group_violations: list[float] = []
                affine_group_guarded_violations: list[float] = []
                for _, indexes in BILATERAL_GROUPS:
                    index = np.asarray(indexes, dtype=int)
                    affine_group_norm = float(
                        math.hypot(
                            *(float(value) for value in affine_candidate[index])
                        )
                    )
                    affine_group_radius = float(
                        specification.strength
                    ) * math.sqrt(float(len(indexes)))
                    affine_group_violations.append(
                        affine_group_norm - affine_group_radius
                    )
                    affine_group_guarded_violations.append(
                        math.nextafter(
                            math.nextafter(affine_group_norm, math.inf)
                            - affine_group_radius,
                            math.inf,
                        )
                    )
                minimax_last_affine_ball_violation = max(
                    affine_group_violations
                )
                minimax_last_affine_ball_violation_guarded_upper = max(
                    affine_group_guarded_violations
                )
            minimax_weight_shift = float(np.max(minimax_constraints))
            minimax_weight_numerators = np.exp(
                (minimax_constraints - minimax_weight_shift) / smoothing
            )
            minimax_weights = minimax_weight_numerators / float(
                np.sum(minimax_weight_numerators)
            )
            minimax_last_maximum_weight = float(np.max(minimax_weights))
            minimax_last_effective_constraint_count = 1.0 / float(
                np.dot(minimax_weights, minimax_weights)
            )
            minimax_correction = null_project(
                minimax_scale * (null_basis @ minimax_coordinates)
            )
            minimax_direction_diagnostics: dict[str, float] = {}
            (
                current_objective,
                current_null_gradient,
                current_kkt,
                current_stationarity,
                current_separation_margin,
            ) = evaluate_null(
                minimax_correction,
                additional_direction=minimax_separator,
                additional_direction_diagnostics=(
                    minimax_direction_diagnostics
                ),
            )
            minimax_last_guarded_lower_margin = (
                minimax_direction_diagnostics.get(
                    "guarded_lower_margin", math.nan
                )
            )
            minimax_last_nominal_gap = minimax_direction_diagnostics.get(
                "nominal_gap", math.nan
            )
            minimax_last_guard_loss = minimax_direction_diagnostics.get(
                "guard_loss", math.nan
            )
            minimax_last_dot_and_span_error = (
                minimax_direction_diagnostics.get(
                    "dot_and_span_error", math.nan
                )
            )
            minimax_last_support_inflation = (
                minimax_direction_diagnostics.get(
                    "support_inflation", math.nan
                )
            )
            minimax_last_decision_inflation = (
                minimax_direction_diagnostics.get(
                    "decision_inflation", math.nan
                )
            )
            minimax_last_combined_guarded_margin = (
                current_separation_margin
            )
            total_iteration = accelerated_iterations + minimax_iterations
            if current_kkt <= decision_threshold:
                return ZeroKKTCertificate(
                    current_kkt,
                    current_stationarity,
                    total_iteration,
                    True,
                    True,
                    0.0,
                    freeze_correction(minimax_correction),
                )
            if current_separation_margin > 0.0:
                return ZeroKKTCertificate(
                    current_kkt,
                    current_stationarity,
                    total_iteration,
                    True,
                    False,
                    current_separation_margin,
                    freeze_correction(minimax_correction),
                )

        # At very small smoothing, binary64 softmax weights can collapse onto
        # one member of a genuinely nonsmooth active face.  Polish only the
        # candidate normal multipliers at the final primal coordinates.  This
        # phase never accepts its QP or constraint diagnostics: each proposed
        # normal still has to pass the unchanged fully guarded separator.
        if minimax_iterations < minimax_budget:
            active_face_gradient_rows = np.asarray(
                [
                    -2.0
                    * (
                        block.T
                        @ minimax_scaled_dual[index]
                    )
                    for index, block in zip(
                        constraint_indexes,
                        constraint_null_blocks,
                        strict=True,
                    )
                ],
                dtype=float,
            )
            active_face_norm_squares = np.asarray(
                [
                    float(np.dot(minimax_scaled_dual[index], minimax_scaled_dual[index]))
                    for index in constraint_indexes
                ],
                dtype=float,
            )
            active_face_scale = max(
                1.0,
                max(
                    float(norm_square) + float(radius) * float(radius)
                    for norm_square, radius in zip(
                        active_face_norm_squares,
                        scaled_radii,
                        strict=True,
                    )
                ),
            )
            active_face_tolerance = (
                ZERO_KKT_ACTIVE_FACE_ROUNDOFF_MULTIPLIER
                * np.finfo(float).eps
                * active_face_scale
            )
            active_face_order = np.argsort(
                -minimax_constraints,
                kind="stable",
            )
            previous_full_weights = np.zeros(
                len(constraint_indexes), dtype=float
            )
            face_end = 0
            while (
                face_end < len(constraint_indexes)
                and minimax_iterations < minimax_budget
            ):
                band_value = float(
                    minimax_constraints[active_face_order[face_end]]
                )
                next_end = face_end + 1
                while (
                    next_end < len(constraint_indexes)
                    and float(
                        minimax_constraints[active_face_order[next_end]]
                    )
                    >= band_value - active_face_tolerance
                ):
                    next_end += 1
                face_end = next_end
                minimax_iterations += 1
                minimax_active_face_growths += 1
                minimax_active_face_maximum_size = max(
                    minimax_active_face_maximum_size, face_end
                )
                if minimax_iterations >= minimax_budget:
                    break
                allowed = active_face_order[:face_end]
                initial_face_weights = previous_full_weights[allowed]
                if float(np.sum(initial_face_weights)) <= 0.0:
                    initial_face_weights = None
                event_budget = minimax_budget - minimax_iterations
                (
                    face_weights,
                    face_events,
                    face_converged,
                ) = minimum_norm_simplex_weights(
                    active_face_gradient_rows[allowed],
                    allowed,
                    initial_face_weights,
                    event_budget,
                )
                minimax_iterations += face_events
                minimax_active_face_qp_events += face_events
                previous_full_weights = np.zeros(
                    len(constraint_indexes), dtype=float
                )
                previous_full_weights[allowed] = face_weights
                if (
                    not face_converged
                    or minimax_iterations >= minimax_budget
                ):
                    continue
                positive_face_weights = previous_full_weights[
                    previous_full_weights > 0.0
                ]
                minimax_active_face_last_support_size = int(
                    positive_face_weights.size
                )
                minimax_active_face_last_maximum_weight = float(
                    np.max(positive_face_weights)
                )
                minimax_active_face_last_effective_constraint_count = (
                    1.0
                    / float(
                        np.dot(
                            positive_face_weights,
                            positive_face_weights,
                        )
                    )
                )
                polished_separator = np.zeros(
                    PAIR_DIMENSION, dtype=float
                )
                for weight, index in zip(
                    previous_full_weights,
                    constraint_indexes,
                    strict=True,
                ):
                    polished_separator[index] = (
                        float(weight) * minimax_scaled_dual[index]
                    )
                polished_null_infinity = float(
                    np.linalg.norm(
                        null_project(polished_separator), ord=np.inf
                    )
                )
                minimax_active_face_best_null_infinity = min(
                    minimax_active_face_best_null_infinity,
                    polished_null_infinity,
                )
                minimax_iterations += 1
                polished_diagnostics: dict[str, float] = {}
                (
                    current_objective,
                    current_null_gradient,
                    current_kkt,
                    current_stationarity,
                    current_separation_margin,
                ) = evaluate_null(
                    minimax_correction,
                    additional_direction=polished_separator,
                    additional_direction_diagnostics=polished_diagnostics,
                )
                polished_guarded_margin = polished_diagnostics.get(
                    "guarded_lower_margin", -math.inf
                )
                minimax_active_face_best_guarded_margin = max(
                    minimax_active_face_best_guarded_margin,
                    polished_guarded_margin,
                )
                total_iteration = accelerated_iterations + minimax_iterations
                if current_kkt <= decision_threshold:
                    return ZeroKKTCertificate(
                        current_kkt,
                        current_stationarity,
                        total_iteration,
                        True,
                        True,
                        0.0,
                        freeze_correction(minimax_correction),
                        active_face_growths=minimax_active_face_growths,
                        active_face_qp_events=minimax_active_face_qp_events,
                        active_face_size=minimax_active_face_maximum_size,
                        active_face_support_size=(
                            minimax_active_face_last_support_size
                        ),
                        active_face_maximum_weight=(
                            minimax_active_face_last_maximum_weight
                        ),
                        active_face_effective_constraint_count=(
                            minimax_active_face_last_effective_constraint_count
                        ),
                    )
                if current_separation_margin > 0.0:
                    return ZeroKKTCertificate(
                        current_kkt,
                        current_stationarity,
                        total_iteration,
                        True,
                        False,
                        current_separation_margin,
                        freeze_correction(minimax_correction),
                        active_face_growths=minimax_active_face_growths,
                        active_face_qp_events=minimax_active_face_qp_events,
                        active_face_size=minimax_active_face_maximum_size,
                        active_face_support_size=(
                            minimax_active_face_last_support_size
                        ),
                        active_face_maximum_weight=(
                            minimax_active_face_last_maximum_weight
                        ),
                        active_face_effective_constraint_count=(
                            minimax_active_face_last_effective_constraint_count
                        ),
                    )

        correction = null_project(
            minimax_scale * (null_basis @ minimax_coordinates)
        )
        coefficients = null_basis.T @ correction
        remaining_iterations -= minimax_iterations
        inverse_hessian = np.eye(null_basis.shape[1], dtype=float)
        (
            current_objective,
            current_null_gradient,
            current_kkt,
            current_stationarity,
            current_separation_margin,
        ) = evaluate_null(null_basis @ coefficients)
        for fallback_iteration in range(1, remaining_iterations + 1):
            total_iteration = (
                accelerated_iterations
                + minimax_iterations
                + fallback_iteration
            )
            if (
                current_kkt > decision_threshold
                and current_separation_margin <= 0.0
                and current_stationarity
                <= 1.0e-1 * max(current_kkt, decision_threshold)
                and (fallback_iteration == 1 or fallback_iteration % 8 == 0)
            ):
                (
                    current_objective,
                    current_null_gradient,
                    current_kkt,
                    current_stationarity,
                    current_separation_margin,
                ) = evaluate_null(
                    null_basis @ coefficients,
                    refine_active_normal=True,
                )
            if current_kkt <= decision_threshold:
                return ZeroKKTCertificate(
                    current_kkt,
                    current_stationarity,
                    total_iteration,
                    True,
                    True,
                    0.0,
                    freeze_correction(null_basis @ coefficients),
                )
            if current_separation_margin > 0.0:
                return ZeroKKTCertificate(
                    current_kkt,
                    current_stationarity,
                    total_iteration,
                    True,
                    False,
                    current_separation_margin,
                    freeze_correction(null_basis @ coefficients),
                )
            coordinate_gradient = null_basis.T @ current_null_gradient
            direction = -(inverse_hessian @ coordinate_gradient)
            directional_derivative = float(np.dot(coordinate_gradient, direction))
            if (
                not np.all(np.isfinite(direction))
                or not math.isfinite(directional_derivative)
                or directional_derivative >= 0.0
            ):
                inverse_hessian = np.eye(null_basis.shape[1], dtype=float)
                direction = -coordinate_gradient
                directional_derivative = -float(
                    np.dot(coordinate_gradient, coordinate_gradient)
                )
            step = 1.0
            accepted = False
            for _ in range(80):
                trial_coefficients = coefficients + step * direction
                (
                    trial_objective,
                    trial_null_gradient,
                    trial_kkt,
                    trial_stationarity,
                    trial_separation_margin,
                ) = evaluate_null(null_basis @ trial_coefficients)
                armijo_bound = (
                    current_objective
                    + 1.0e-4 * step * directional_derivative
                    + 64.0
                    * np.finfo(float).eps
                    * max(abs(current_objective), abs(trial_objective))
                )
                if trial_objective <= armijo_bound:
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
            last_relative_change = abs(
                current_objective - trial_objective
            ) / max(1.0, abs(current_objective), abs(trial_objective))
            trial_coordinate_gradient = null_basis.T @ trial_null_gradient
            coefficient_change = trial_coefficients - coefficients
            gradient_change = trial_coordinate_gradient - coordinate_gradient
            curvature = float(np.dot(coefficient_change, gradient_change))
            curvature_floor = (
                64.0
                * np.finfo(float).eps
                * float(np.linalg.norm(coefficient_change))
                * float(np.linalg.norm(gradient_change))
            )
            if math.isfinite(curvature) and curvature > curvature_floor:
                rho = 1.0 / curvature
                identity = np.eye(null_basis.shape[1], dtype=float)
                left = identity - rho * np.outer(
                    coefficient_change, gradient_change
                )
                right = identity - rho * np.outer(
                    gradient_change, coefficient_change
                )
                inverse_hessian = (
                    left @ inverse_hessian @ right
                    + rho * np.outer(coefficient_change, coefficient_change)
                )
                if not np.all(np.isfinite(inverse_hessian)):
                    inverse_hessian = np.eye(null_basis.shape[1], dtype=float)
            else:
                inverse_hessian = np.eye(null_basis.shape[1], dtype=float)
            coefficients = trial_coefficients
            current_objective = trial_objective
            current_null_gradient = trial_null_gradient
            current_kkt = trial_kkt
            current_stationarity = trial_stationarity
            current_separation_margin = trial_separation_margin
        # The loop checks the certificate before each quasi-Newton update.  Check
        # the state produced by the final permitted update as well, without
        # spending an additional solver iteration beyond the frozen cap.
        final_correction = null_basis @ coefficients
        (
            current_objective,
            current_null_gradient,
            current_kkt,
            current_stationarity,
            current_separation_margin,
        ) = evaluate_null(final_correction, refine_active_normal=True)
        final_iteration = (
            accelerated_iterations + minimax_iterations + fallback_iteration
        )
        if current_kkt <= decision_threshold:
            return ZeroKKTCertificate(
                current_kkt,
                current_stationarity,
                final_iteration,
                True,
                True,
                0.0,
                freeze_correction(final_correction),
            )
        if current_separation_margin > 0.0:
            return ZeroKKTCertificate(
                current_kkt,
                current_stationarity,
                final_iteration,
                True,
                False,
                current_separation_margin,
                freeze_correction(final_correction),
            )
    raise _ZeroKKTIndeterminate(
        "zero-KKT dual feasibility solver did not converge within the iteration cap: "
        f"residual={current_kkt:.17g}, stationarity={current_stationarity:.17g}, "
        f"relative_change={last_relative_change:.17g}, threshold={threshold:.17g}, "
        f"decision_tolerance={decision_threshold:.17g}, "
        f"separation_margin={current_separation_margin:.17g}, "
        f"strength={float(specification.strength):.17g}, "
        f"minimax_iterations={minimax_iterations}, "
        f"minimax_maximum_constraint={minimax_last_maximum_constraint:.17g}, "
        f"minimax_smoothing={minimax_last_smoothing:.17g}, "
        f"minimax_gradient_infinity={minimax_last_gradient_infinity:.17g}, "
        f"minimax_separator_null_infinity="
        f"{minimax_last_separator_null_infinity:.17g}, "
        f"minimax_projected_separator_gap="
        f"{minimax_last_projected_separator_gap:.17g}, "
        f"minimax_projected_separator_gap_per_l1="
        f"{minimax_last_projected_separator_gap_per_l1:.17g}, "
        f"minimax_guarded_lower_margin="
        f"{minimax_last_guarded_lower_margin:.17g}, "
        f"minimax_nominal_gap={minimax_last_nominal_gap:.17g}, "
        f"minimax_guard_loss={minimax_last_guard_loss:.17g}, "
        f"minimax_dot_and_span_error="
        f"{minimax_last_dot_and_span_error:.17g}, "
        f"minimax_support_inflation="
        f"{minimax_last_support_inflation:.17g}, "
        f"minimax_decision_inflation="
        f"{minimax_last_decision_inflation:.17g}, "
        f"minimax_affine_range_residual="
        f"{minimax_last_affine_range_residual:.17g}, "
        f"minimax_affine_ball_violation="
        f"{minimax_last_affine_ball_violation:.17g}, "
        f"minimax_affine_ball_violation_guarded_upper="
        f"{minimax_last_affine_ball_violation_guarded_upper:.17g}, "
        f"minimax_maximum_weight={minimax_last_maximum_weight:.17g}, "
        f"minimax_effective_constraint_count="
        f"{minimax_last_effective_constraint_count:.17g}, "
        f"minimax_combined_guarded_margin="
        f"{minimax_last_combined_guarded_margin:.17g}, "
        f"minimax_active_face_growths={minimax_active_face_growths}, "
        f"minimax_active_face_qp_events={minimax_active_face_qp_events}, "
        f"minimax_active_face_maximum_size="
        f"{minimax_active_face_maximum_size}, "
        f"minimax_active_face_best_null_infinity="
        f"{minimax_active_face_best_null_infinity:.17g}, "
        f"minimax_active_face_best_guarded_margin="
        f"{minimax_active_face_best_guarded_margin:.17g}, "
        f"minimax_active_face_last_support_size="
        f"{minimax_active_face_last_support_size}, "
        f"minimax_active_face_last_maximum_weight="
        f"{minimax_active_face_last_maximum_weight:.17g}, "
        f"minimax_active_face_last_effective_constraint_count="
        f"{minimax_active_face_last_effective_constraint_count:.17g}, "
        f"fallback_iterations={fallback_iteration}, "
        f"total_iterations_consumed="
        f"{accelerated_iterations + minimax_iterations + fallback_iteration}"
    )


def fold_lambda_maximum(
    fixed_m1_theta: np.ndarray,
    prepared: PreparedM2,
    candidates: CandidateLibrary,
    geometry: GeometryEngine,
    pairs: PairFeatureLibrary,
    penalty_kind: str,
    *,
    l1_ratio: float = 1.0,
    linear_correction: np.ndarray | None = None,
) -> LambdaMaximumResult:
    """Find the constrained fold-local zero-KKT lambda by deterministic bisection."""
    shape = PenaltySpec(penalty_kind, 0.0, l1_ratio)
    zero = np.zeros(PAIR_DIMENSION, dtype=float)
    evaluation = evaluate_m2(
        zero,
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        linear_correction=linear_correction,
    )
    gradient = evaluation.gradient
    gradient_norm = float(np.linalg.norm(gradient, ord=np.inf))
    # Only a bit-exact zero gradient denotes the exact nested M1 model.  A
    # tolerance branch here would manufacture lambda_max=0 for small but real
    # M2 gradients and collapse every rho to the M1 fit.
    if np.count_nonzero(gradient) == 0:
        return LambdaMaximumResult(0.0, 0.0, 0.0, 0.0, 0)
    # The parent contract defines lambda_max by exact zero-KKT feasibility.
    # Use a dimension-aware binary64 backward-error scale, capped by the
    # frozen absolute proximal tolerance, rather than a positive modeling
    # tolerance that would create an avoidable feasibility gray zone.
    epsilon = np.finfo(float).eps
    roundoff_gamma = PAIR_DIMENSION * epsilon / (1.0 - PAIR_DIMENSION * epsilon)
    zero_threshold = min(
        PROX_TOLERANCE,
        roundoff_gamma * gradient_norm,
    )
    if zero_threshold <= 0.0:
        raise M2Error("nonzero lambda-max gradient underflowed its decision scale")

    upper = LAMBDA_MAXIMUM_INITIAL_UPPER
    upper_specification = PenaltySpec(penalty_kind, upper, l1_ratio)
    certificate = zero_kkt_certificate(
        gradient,
        upper_specification,
        prepared.transform,
        decision_tolerance=zero_threshold,
    )
    residual = certificate.residual
    expansions = 0
    while not certificate.feasible:
        if expansions >= LAMBDA_MAXIMUM_MAXIMUM_EXPANSIONS:
            raise M2Error("could not bracket fold-local lambda_max")
        upper *= 2.0
        expansions += 1
        if not math.isfinite(upper):
            raise M2Error("could not bracket fold-local lambda_max")
        certificate = zero_kkt_certificate(
            gradient,
            PenaltySpec(penalty_kind, upper, l1_ratio),
            prepared.transform,
            decision_tolerance=zero_threshold,
            initial_correction=np.asarray(certificate.correction, dtype=float),
        )
        residual = certificate.residual

    lower = 0.0
    completed_bisections = 0

    def bracket_probe(strength: float) -> ZeroKKTCertificate:
        nonlocal completed_bisections
        if completed_bisections >= LAMBDA_MAXIMUM_BISECTIONS:
            raise M2Error(
                "fold-local lambda_max exhausted its zero-KKT oracle-call cap"
            )
        completed_bisections += 1
        observed = zero_kkt_certificate(
            gradient,
            PenaltySpec(penalty_kind, strength, l1_ratio),
            prepared.transform,
            decision_tolerance=zero_threshold,
            # Only the certified feasible upper endpoint supplies warm state.
            # An indeterminate query is never assigned to either bracket side.
            initial_correction=np.asarray(certificate.correction, dtype=float),
        )
        if not observed.certified:
            raise _ZeroKKTIndeterminate(
                "zero-KKT oracle returned an uncertified bracket probe"
            )
        return observed

    while (
        upper - lower
        > LAMBDA_MAXIMUM_RELATIVE_BRACKET_TOLERANCE * upper
    ):
        old_lower = lower
        old_upper = upper
        midpoint = old_lower + 0.5 * (old_upper - old_lower)
        if not old_lower < midpoint < old_upper:
            raise M2Error(
                "fold-local lambda_max bracket has no interior midpoint"
            )
        try:
            midpoint_certificate = bracket_probe(midpoint)
        except _ZeroKKTIndeterminate as midpoint_error:
            # A finite-precision oracle can be unable to classify a point lying
            # essentially on the feasibility transition.  Keep both certified
            # endpoints and probe away from that point.  The unknown midpoint
            # itself never moves the bracket or supplies a warm correction.
            quarter_strengths = (
                old_lower + 0.25 * (old_upper - old_lower),
                old_lower + 0.75 * (old_upper - old_lower),
            )
            quarter_certificates: list[
                tuple[float, ZeroKKTCertificate]
            ] = []
            last_indeterminate: _ZeroKKTIndeterminate = midpoint_error
            for quarter_strength in quarter_strengths:
                if not old_lower < quarter_strength < old_upper:
                    continue
                try:
                    quarter_certificates.append(
                        (quarter_strength, bracket_probe(quarter_strength))
                    )
                except _ZeroKKTIndeterminate as quarter_error:
                    last_indeterminate = quarter_error
            if not quarter_certificates:
                raise M2Error(
                    "fold-local lambda_max could not shrink an indeterminate "
                    "zero-KKT interval with certified quarter probes"
                ) from last_indeterminate

            infeasible_strengths = [
                strength
                for strength, observed in quarter_certificates
                if not observed.feasible
            ]
            feasible_certificates = [
                (strength, observed)
                for strength, observed in quarter_certificates
                if observed.feasible
            ]
            if (
                infeasible_strengths
                and feasible_certificates
                and min(strength for strength, _ in feasible_certificates)
                < max(infeasible_strengths)
            ):
                raise M2Error(
                    "zero-KKT quarter probes contradicted monotone feasibility"
                )
            if infeasible_strengths:
                lower = max(lower, max(infeasible_strengths))
            if feasible_certificates:
                feasible_strength, feasible_certificate = min(
                    feasible_certificates,
                    key=lambda item: item[0],
                )
                upper = min(upper, feasible_strength)
                residual = feasible_certificate.residual
                certificate = feasible_certificate
            if lower <= old_lower and upper >= old_upper:
                raise M2Error(
                    "certified quarter probes did not shrink the lambda_max bracket"
                )
            continue

        midpoint_residual = midpoint_certificate.residual
        if midpoint_certificate.feasible:
            upper = midpoint
            residual = midpoint_residual
            certificate = midpoint_certificate
        else:
            lower = midpoint
    final_certificate = zero_kkt_certificate(
        gradient,
        PenaltySpec(penalty_kind, upper, l1_ratio),
        prepared.transform,
        decision_tolerance=zero_threshold,
        initial_correction=np.asarray(certificate.correction, dtype=float),
    )
    final_residual = final_certificate.residual
    if not final_certificate.feasible or final_residual > zero_threshold:
        raise M2Error("fold-local lambda_max failed its zero-KKT certificate")
    return LambdaMaximumResult(
        value=float(upper),
        lower_bracket=float(lower),
        upper_bracket=float(upper),
        zero_kkt_residual=float(final_residual),
        bisections=completed_bisections,
    )


def _objective_with_penalty(
    evaluation: M2Evaluation,
    beta: np.ndarray,
    specification: PenaltySpec,
) -> tuple[float, float]:
    raw_penalty = penalty_value(beta, specification)
    total = evaluation.objective + specification.strength * raw_penalty
    if not math.isfinite(total):
        raise M2Error("penalized M2 objective is nonfinite")
    return float(total), float(raw_penalty)


def _proximal_gradient_residual(
    beta: np.ndarray,
    gradient: np.ndarray,
    lipschitz: float,
    specification: PenaltySpec,
    transform: ResidualTransform,
    warm_state: _DykstraWarmState | None = None,
) -> tuple[float, ProxResult, _DykstraWarmState | None, bool]:
    prox, next_warm_state, warm_started = _prox_penalty_range_with_state(
        beta - gradient / lipschitz,
        specification.strength / lipschitz,
        specification,
        transform,
        warm_state=warm_state,
    )
    residual = lipschitz * float(np.linalg.norm(beta - prox.vector, ord=np.inf))
    return residual, prox, next_warm_state, warm_started


def _objective_roundoff_tolerance(value: float) -> float:
    return (
        MAJORIZATION_EPSILON_MULTIPLIER
        * np.finfo(float).eps
        * max(1.0, abs(float(value)))
    )


def fit_m2(
    fixed_m1_theta: np.ndarray,
    prepared: PreparedM2,
    candidates: CandidateLibrary,
    geometry: GeometryEngine,
    pairs: PairFeatureLibrary,
    specification: PenaltySpec,
    *,
    initial: np.ndarray | None = None,
    linear_correction: np.ndarray | None = None,
    maximum_iterations: int = FISTA_MAXIMUM_ITERATIONS,
) -> M2OptimizerResult:
    """Deterministic monotone FISTA with certified range/penalty proximal maps."""
    if maximum_iterations <= 0:
        raise M2Error("FISTA maximum iterations must be positive")
    if initial is None:
        x = np.zeros(PAIR_DIMENSION, dtype=float)
    else:
        source = np.asarray(initial, dtype=float)
        if source.shape != (PAIR_DIMENSION,) or not np.all(np.isfinite(source)):
            raise M2Error("initial beta is malformed or nonfinite")
        x = _range_project_vector(source, prepared.transform)
    y = x.copy()
    acceleration = 1.0
    current_evaluation = evaluate_m2(
        x,
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        linear_correction=linear_correction,
    )
    current_objective, current_penalty = _objective_with_penalty(
        current_evaluation, x, specification
    )
    lipschitz = FISTA_INITIAL_LIPSCHITZ
    backtracking_steps = 0
    restart_count = 0
    maximum_prox_iterations = 0
    optimizer_dykstra_state: _DykstraWarmState | None = None
    kkt_dykstra_state: _DykstraWarmState | None = None
    dykstra_warm_start_calls = 0

    zero_certificate: ZeroKKTCertificate | None = None
    if np.count_nonzero(x) == 0:
        zero_certificate = zero_kkt_certificate(
            current_evaluation.gradient,
            specification,
            prepared.transform,
            decision_tolerance=KKT_TOLERANCE,
        )
        initial_kkt = zero_certificate.residual
        maximum_prox_iterations = max(
            maximum_prox_iterations, zero_certificate.iterations
        )
    else:
        (
            initial_kkt,
            initial_prox,
            kkt_dykstra_state,
            initial_kkt_warm_started,
        ) = _proximal_gradient_residual(
            x,
            current_evaluation.gradient,
            lipschitz,
            specification,
            prepared.transform,
            kkt_dykstra_state,
        )
        dykstra_warm_start_calls += int(initial_kkt_warm_started)
        maximum_prox_iterations = max(
            maximum_prox_iterations, initial_prox.iterations
        )
    if initial_kkt <= KKT_TOLERANCE:
        range_residual = float(
            np.linalg.norm(x - _range_project_vector(x, prepared.transform), ord=np.inf)
        )
        return M2OptimizerResult(
            beta=x,
            objective=current_objective,
            smooth_objective=current_evaluation.objective,
            penalty_value=current_penalty,
            iterations=0,
            lipschitz=lipschitz,
            kkt_residual=initial_kkt,
            relative_objective_change=0.0,
            range_residual=range_residual,
            status="pass",
            telemetry={
                "zero_solution_KKT_certified": True,
                "zero_solution_dual_stationarity": (
                    zero_certificate.stationarity
                    if zero_certificate is not None
                    else None
                ),
                "backtracking_steps": 0,
                "monotone_restarts": 0,
                "maximum_Dykstra_iterations": maximum_prox_iterations,
                "Dykstra_warm_start_calls": dykstra_warm_start_calls,
                "initial_Lipschitz_estimate": FISTA_INITIAL_LIPSCHITZ,
                "majorization_epsilon_multiplier": MAJORIZATION_EPSILON_MULTIPLIER,
            },
        )

    last_relative_change = math.inf
    last_kkt = initial_kkt
    for iteration in range(1, maximum_iterations + 1):
        accepted = False
        candidate = x
        candidate_evaluation = current_evaluation
        candidate_objective = current_objective
        candidate_penalty = current_penalty
        accepted_prox: ProxResult | None = None
        for restart_attempt in range(2):
            y_evaluation = evaluate_m2(
                y,
                fixed_m1_theta,
                prepared,
                candidates,
                geometry,
                pairs,
                linear_correction=linear_correction,
            )
            for _ in range(80):
                prox, proposed_dykstra_state, warm_started = (
                    _prox_penalty_range_with_state(
                        y - y_evaluation.gradient / lipschitz,
                        specification.strength / lipschitz,
                        specification,
                        prepared.transform,
                        warm_state=optimizer_dykstra_state,
                    )
                )
                dykstra_warm_start_calls += int(warm_started)
                maximum_prox_iterations = max(
                    maximum_prox_iterations, prox.iterations
                )
                trial = prox.vector
                trial_evaluation = evaluate_m2(
                    trial,
                    fixed_m1_theta,
                    prepared,
                    candidates,
                    geometry,
                    pairs,
                    linear_correction=linear_correction,
                )
                displacement = trial - y
                majorizer = (
                    y_evaluation.objective
                    + float(np.dot(y_evaluation.gradient, displacement))
                    + 0.5 * lipschitz * float(np.dot(displacement, displacement))
                )
                if (
                    trial_evaluation.objective
                    <= majorizer + _objective_roundoff_tolerance(majorizer)
                ):
                    break
                lipschitz *= BACKTRACKING_MULTIPLIER
                backtracking_steps += 1
                if not math.isfinite(lipschitz):
                    raise M2Error("FISTA Lipschitz estimate became nonfinite")
            else:
                raise M2Error("FISTA backtracking did not find a majorizing step")

            trial_objective, trial_penalty = _objective_with_penalty(
                trial_evaluation, trial, specification
            )
            monotone_tolerance = _objective_roundoff_tolerance(current_objective)
            if trial_objective <= current_objective + monotone_tolerance:
                candidate = trial
                candidate_evaluation = trial_evaluation
                candidate_objective = trial_objective
                candidate_penalty = trial_penalty
                accepted_prox = prox
                optimizer_dykstra_state = proposed_dykstra_state
                accepted = True
                break
            if restart_attempt == 0:
                y = x.copy()
                acceleration = 1.0
                restart_count += 1
        if not accepted or accepted_prox is None:
            raise M2Error("monotone FISTA restart failed to decrease the objective")

        last_relative_change = abs(current_objective - candidate_objective) / max(
            1.0, abs(current_objective), abs(candidate_objective)
        )
        (
            last_kkt,
            kkt_prox,
            kkt_dykstra_state,
            kkt_warm_started,
        ) = _proximal_gradient_residual(
            candidate,
            candidate_evaluation.gradient,
            lipschitz,
            specification,
            prepared.transform,
            kkt_dykstra_state,
        )
        dykstra_warm_start_calls += int(kkt_warm_started)
        maximum_prox_iterations = max(
            maximum_prox_iterations, kkt_prox.iterations
        )
        range_residual = float(
            np.linalg.norm(
                candidate - _range_project_vector(candidate, prepared.transform),
                ord=np.inf,
            )
        )
        if not all(
            math.isfinite(value)
            for value in (
                candidate_objective,
                last_relative_change,
                last_kkt,
                range_residual,
            )
        ):
            raise M2Error("FISTA convergence diagnostics became nonfinite")
        if (
            last_kkt <= KKT_TOLERANCE
            and last_relative_change <= RELATIVE_OBJECTIVE_TOLERANCE
        ):
            return M2OptimizerResult(
                beta=_finite(candidate, "fitted M2 beta"),
                objective=candidate_objective,
                smooth_objective=candidate_evaluation.objective,
                penalty_value=candidate_penalty,
                iterations=iteration,
                lipschitz=lipschitz,
                kkt_residual=last_kkt,
                relative_objective_change=last_relative_change,
                range_residual=range_residual,
                status="pass",
                telemetry={
                    "zero_solution_KKT_certified": False,
                    "backtracking_steps": backtracking_steps,
                    "monotone_restarts": restart_count,
                    "maximum_Dykstra_iterations": maximum_prox_iterations,
                    "Dykstra_warm_start_calls": dykstra_warm_start_calls,
                    "initial_Lipschitz_estimate": FISTA_INITIAL_LIPSCHITZ,
                    "majorization_epsilon_multiplier": MAJORIZATION_EPSILON_MULTIPLIER,
                    "all_finite": True,
                },
            )

        next_acceleration = 0.5 * (
            1.0 + math.sqrt(1.0 + 4.0 * acceleration * acceleration)
        )
        next_y = candidate + (
            (acceleration - 1.0) / next_acceleration
        ) * (candidate - x)
        x = candidate
        y = _range_project_vector(next_y, prepared.transform)
        acceleration = next_acceleration
        current_evaluation = candidate_evaluation
        current_objective = candidate_objective
        current_penalty = candidate_penalty

    raise M2Error(
        "monotone FISTA reached its iteration cap without both KKT and relative-objective convergence"
    )


def synthetic_k2_problem(
    *, rows: int = 24
) -> tuple[
    tuple[Cell, ...],
    tuple[UniformCellMoments, ...],
    PreparedM2,
    CandidateLibrary,
    GeometryEngine,
    PairFeatureLibrary,
    np.ndarray,
]:
    """Build a small deterministic target-free problem for tests only."""
    if rows < 20:
        raise M2Error("synthetic K=2 problem requires at least 20 rows")
    (
        transform_cells,
        transform_moments,
        _,
        candidates,
        geometry,
        _,
    ) = m1.generate_synthetic_dataset(
        groups=1, rows_per_group=17, seed=2026092801
    )
    pair_features = PairFeatureLibrary(candidates)
    transform, _ = fit_residual_transform(
        transform_cells,
        transform_moments,
        candidates,
        geometry,
        pair_features,
    )
    target_coordinates = m1.synthetic_target_coordinates()
    cells: list[Cell] = []
    for row in range(rows):
        phase = float(row + 1)
        ap = 3.0 + 0.55 * math.sin(0.73 * phase) + 0.013 * phase
        dv = 2.4 + 0.47 * math.cos(0.51 * phase) + 0.007 * phase * phase
        depth = 0.65 + 0.21 * math.sin(0.37 * phase + 0.4) + 0.003 * phase
        hemisphere = "left" if (row // 3) % 2 == 0 else "right"
        ml = 5.7 - depth if hemisphere == "left" else 5.7 + depth
        soma_canonical = np.asarray((ap, dv, abs(depth)), dtype=float)
        distances = np.sqrt(
            np.sum((target_coordinates - soma_canonical) ** 2, axis=1)
        )
        cells.append(
            Cell(
                identifier=f"m2_synthetic_{row:04d}",
                group=f"synthetic_group_{row % 4}",
                layer=m1.LAYERS[row % len(m1.LAYERS)],
                hemisphere=hemisphere,
                soma_um=(1000.0 * ap, 1000.0 * dv, 1000.0 * ml),
                soma_distance_mm=distances,
                k=2,
                observed_index=(17 * row + 3) % math.comb(m1.PANEL_SIZE, 2),
                segment_unmapped=int(row % 5 == 0),
                sampling_uniform=0.5,
            )
        )
    moments = tuple(geometry.uniform_moments(cell) for cell in cells)
    prepared = apply_residual_transform(
        cells, moments, transform
    )
    if transform.rank <= 0 or transform.rank >= PAIR_DIMENSION:
        raise M2Error("synthetic residual range should be nonempty and nontrivial")
    fixed_m1_theta = 0.15 * m1.planted_parameters()
    return (
        tuple(cells),
        moments,
        prepared,
        candidates,
        geometry,
        pair_features,
        fixed_m1_theta,
    )


def run_self_test() -> dict[str, Any]:
    """Run target-free synthetic conformance checks; never accepts data paths."""
    (
        cells,
        moments,
        prepared,
        candidates,
        geometry,
        pairs,
        fixed_m1_theta,
    ) = synthetic_k2_problem(rows=24)
    expected_counts = {k: math.comb(m1.PANEL_SIZE, k) for k in range(17)}
    if pairs.all_counts() != expected_counts:
        raise M2Error("all-K pair enumeration count changed")

    zero = np.zeros(PAIR_DIMENSION, dtype=float)
    m2_zero = evaluate_m2(
        zero,
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
        need_predictions=True,
    )
    m1_zero = m1.evaluate_model(
        fixed_m1_theta,
        prepared.m1_data,
        candidates,
        geometry,
        0.0,
        need_hessian=False,
        need_predictions=True,
    )
    assert m2_zero.log_probability_nats is not None
    assert m2_zero.marginals is not None
    assert m1_zero.log_probability_nats is not None
    assert m1_zero.marginals is not None
    nesting_log_error = float(
        np.max(
            np.abs(m2_zero.log_probability_nats - m1_zero.log_probability_nats)
        )
    )
    nesting_marginal_error = float(
        np.max(np.abs(m2_zero.marginals - m1_zero.marginals))
    )
    if nesting_log_error != 0.0 or nesting_marginal_error != 0.0:
        raise M2Error("beta=0 did not exactly reproduce fixed M1 predictions")

    direction = _range_project_vector(
        np.sin(np.arange(1, PAIR_DIMENSION + 1, dtype=float)), prepared.transform
    )
    direction /= float(np.linalg.norm(direction))
    beta = 0.03 * direction
    analytic = evaluate_m2(
        beta,
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
    )
    step = 1.0e-6
    plus = evaluate_m2(
        beta + step * direction,
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
    )
    minus = evaluate_m2(
        beta - step * direction,
        fixed_m1_theta,
        prepared,
        candidates,
        geometry,
        pairs,
    )
    finite_difference = (plus.objective - minus.objective) / (2.0 * step)
    analytic_direction = float(np.dot(analytic.gradient, direction))
    derivative_error = abs(finite_difference - analytic_direction)
    if derivative_error > 2.0e-7:
        raise M2Error("M2 analytic gradient failed its finite-difference check")

    lambda_checks: dict[str, Any] = {}
    for kind, ratio in (("lasso", 1.0), ("elastic_net", 0.5), ("group_lasso", 1.0)):
        maximum = fold_lambda_maximum(
            fixed_m1_theta,
            prepared,
            candidates,
            geometry,
            pairs,
            kind,
            l1_ratio=ratio,
        )
        specification = PenaltySpec(kind, maximum.value, ratio)
        fitted = fit_m2(
            fixed_m1_theta,
            prepared,
            candidates,
            geometry,
            pairs,
            specification,
            maximum_iterations=20,
        )
        if np.count_nonzero(fitted.beta) != 0 or fitted.kkt_residual > KKT_TOLERANCE:
            raise M2Error(f"{kind} lambda_max did not certify beta=0")
        lambda_checks[kind] = {
            "lambda_max": maximum.value,
            "zero_KKT_residual": fitted.kkt_residual,
        }

    return {
        "status": "pass",
        "stage": "v3_M2_target_free_synthetic_self_test",
        "cells": len(cells),
        "groups": len({cell.group for cell in cells}),
        "pair_dimension": PAIR_DIMENSION,
        "bilateral_groups": len(BILATERAL_GROUPS),
        "residual_rank": prepared.transform.rank,
        "rank_stable": prepared.transform.diagnostics["rank_stable"],
        "projection_orthogonality_maximum_absolute_error": prepared.transform.diagnostics[
            "G_FH_projection_maximum_absolute_orthogonality_error"
        ],
        "beta_zero_log_probability_maximum_absolute_error": nesting_log_error,
        "beta_zero_marginal_maximum_absolute_error": nesting_marginal_error,
        "gradient_directional_maximum_absolute_error": derivative_error,
        "lambda_maximum": lambda_checks,
        "real_target_membership_or_effects_opened": False,
        "final_or_unassigned_or_SWC_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if not arguments.self_test:
        parser.error("the target-free M2 CLI exposes only --self-test")
    try:
        result = run_self_test()
    except (
        M2Error,
        m1.M1Error,
        AssertionError,
        ArithmeticError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
