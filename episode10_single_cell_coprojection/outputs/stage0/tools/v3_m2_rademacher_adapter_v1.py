#!/usr/bin/env python3
"""Target-free capacity-matched Rademacher features for EP10 v3 M2.

This is a lower-level numerical adapter.  It has no data loader, command-line
entry point, controller authority, or persistence surface.  A caller supplies
an already-fitted physical ``PreparedM2`` fold, its exact physical pair-feature
library, and target-free ``Cell`` objects.

The one trial-level :class:`RademacherFeatureLibrary` owns the exact frozen
65,536-by-120 dictionary and is reused by every fold.  Each fold refits only
the noise-specific projection of that dictionary on the M1 sufficient
statistic.  The physical fold's design transform, geometry scales, pair RMS
vector, range basis, and range projector are reused unchanged.
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

import fit_v3_m1 as m1  # noqa: E402
import fit_v3_m2 as m2  # noqa: E402


ROW_COUNT = 65_536
COLUMN_COUNT = m2.PAIR_DIMENSION
RAW_WORD_COUNT = 122_880
DECODED_BIT_COUNT = ROW_COUNT * COLUMN_COUNT
NULL_STAGE_CODE_M2_FIT = 3
PURPOSE_CODE_RADEMACHER_DESIGN = 3
NONAPPLICABLE_FOLD_ORDINAL = 0
EXPECTED_PROJECTED_RANK = 104
DIRECT_SEED_MODES = frozenset({"observed", "target_free_qualification"})
NULL_REPLAY_MODE = "null_replay"

if (
    COLUMN_COUNT != 120
    or DECODED_BIT_COUNT != 7_864_320
    or RAW_WORD_COUNT * 64 != DECODED_BIT_COUNT
):
    raise RuntimeError("the frozen Rademacher dictionary dimensions changed")


class RademacherAdapterError(m2.M2Error):
    """A target-free Rademacher numerical contract failed closed."""


@dataclass(frozen=True)
class _StatisticsIssuance:
    statistics_reference: Any
    mean_reference: Any
    mean_snapshot: np.ndarray
    covariance_reference: Any
    covariance_snapshot: np.ndarray
    cross_reference: Any
    cross_snapshot: np.ndarray


@dataclass(frozen=True)
class _CandidateBasisSnapshot:
    k: int
    basis_reference: Any
    membership_reference: Any
    membership_snapshot: np.ndarray
    contrasts_reference: Any
    contrasts_snapshot: np.ndarray


@dataclass(frozen=True)
class _FeatureIssuance:
    feature_reference: Any
    candidates_reference: Any
    seed_route_reference: Any
    matrix_reference: Any
    seed_route_record: tuple[Any, ...]
    offsets: tuple[tuple[int, int, int], ...]
    statistics: dict[int, _StatisticsIssuance]


@dataclass(frozen=True)
class _PreparedSnapshot:
    prepared_reference: Any
    m1_data_reference: Any
    transform_reference: Any
    cell_references: tuple[Any, ...]
    cell_fields: tuple[tuple[Any, ...], ...]
    soma_distance_references: tuple[Any, ...]
    soma_distance_snapshots: tuple[np.ndarray, ...]
    moment_references: tuple[Any, ...]
    moment_geometry_references: tuple[Any, ...]
    moment_geometry_snapshots: tuple[np.ndarray, ...]
    moment_covariance_references: tuple[Any, ...]
    moment_covariance_snapshots: tuple[np.ndarray, ...]
    x_reference: Any
    x_snapshot: np.ndarray
    geometry_centers_reference: Any
    geometry_centers_snapshot: np.ndarray
    geometry_scales_reference: Any
    geometry_scales_snapshot: np.ndarray
    weights_reference: Any
    weights_snapshot: np.ndarray
    design_transform_reference: Any
    soma_mean_reference: Any
    soma_mean_snapshot: np.ndarray
    soma_scale_reference: Any
    soma_scale_snapshot: np.ndarray
    candidates_reference: Any
    candidate_bases: tuple[_CandidateBasisSnapshot, ...]


@dataclass(frozen=True)
class _FoldIssuance:
    fitted_reference: Any
    features_reference: Any
    physical_prepared_reference: Any
    transform_reference: Any
    prepared_reference: Any
    candidates_reference: Any
    geometry_reference: Any
    physical_pairs_reference: Any
    selected_indexes: tuple[int, ...]
    physical_candidate_ks: tuple[int, ...]
    projection_reference: Any
    projection_snapshot: np.ndarray
    residual_gram_reference: Any
    residual_gram_snapshot: np.ndarray
    scaled_gram_reference: Any
    scaled_gram_snapshot: np.ndarray
    prepared_snapshot: _PreparedSnapshot


@dataclass(frozen=True)
class _ValidationIssuance:
    prepared_reference: Any
    fitted_reference: Any
    prepared_snapshot: _PreparedSnapshot


_ISSUANCE_LOCK = threading.RLock()
_FEATURE_ISSUANCE: dict[int, tuple[Any, _FeatureIssuance]] = {}
_FOLD_ISSUANCE: dict[int, tuple[Any, _FoldIssuance]] = {}
_PREPARED_ISSUANCE: dict[
    int, tuple[Any, _FoldIssuance | _ValidationIssuance]
] = {}


def _weak_reference(value: object, label: str) -> Any:
    try:
        return weakref.ref(value)
    except TypeError as error:
        raise RademacherAdapterError(
            f"{label} does not support identity-bound issuance"
        ) from error


def _register_identity(
    registry: dict[int, tuple[Any, Any]],
    value: object,
    record: Any,
) -> None:
    key = id(value)

    def remove(reference: Any, *, identity: int = key) -> None:
        with _ISSUANCE_LOCK:
            current = registry.get(identity)
            if current is not None and current[0] is reference:
                registry.pop(identity, None)

    reference = weakref.ref(value, remove)
    with _ISSUANCE_LOCK:
        registry[key] = (reference, record)


def _lookup_identity(registry: Mapping[int, tuple[Any, Any]], value: object) -> Any | None:
    with _ISSUANCE_LOCK:
        entry = registry.get(id(value))
    if entry is None or entry[0]() is not value:
        return None
    return entry[1]


def _exact_nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RademacherAdapterError(f"{label} must be an exact integer")
    result = int(value)
    if result < 0:
        raise RademacherAdapterError(f"{label} must be nonnegative")
    return result


def _readonly(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    # Back the public array by immutable bytes.  Merely clearing NumPy's
    # WRITEABLE flag on an owning array can be reversed by its caller.
    result = np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
    ).reshape(contiguous.shape, order="C")
    if result.flags.writeable:
        raise RademacherAdapterError("failed to make an immutable array")
    return result


@dataclass(frozen=True)
class RademacherSeedRoute:
    """Exact observed/qualification or null-replay RNG route.

    ``trial_seed`` remains part of null trial identity, but, prospectively, it
    does not control null-replay draws.  Those draws use the replicate seed and
    the frozen integer-only spawn key.
    """

    mode: str
    trial_seed: int
    null_replicate_seed: int | None = None
    valid_index: int | None = None

    def __post_init__(self) -> None:
        trial_seed = _exact_nonnegative_integer(self.trial_seed, "trial seed")
        object.__setattr__(self, "trial_seed", trial_seed)
        if self.mode in DIRECT_SEED_MODES:
            if self.null_replicate_seed is not None or self.valid_index is not None:
                raise RademacherAdapterError(
                    "direct Rademacher seed routes cannot carry null fields"
                )
            return
        if self.mode != NULL_REPLAY_MODE:
            raise RademacherAdapterError("unknown Rademacher seed-route mode")
        replicate_seed = _exact_nonnegative_integer(
            self.null_replicate_seed, "null replicate seed"
        )
        valid_index = _exact_nonnegative_integer(
            self.valid_index, "null trial valid index"
        )
        if valid_index < 1 or valid_index > 80:
            raise RademacherAdapterError(
                "null trial valid index is outside the frozen 1..80 range"
            )
        object.__setattr__(self, "null_replicate_seed", replicate_seed)
        object.__setattr__(self, "valid_index", valid_index)

    @property
    def spawn_key(self) -> tuple[int, int, int, int] | None:
        if self.mode != NULL_REPLAY_MODE:
            return None
        assert self.valid_index is not None
        return (
            NULL_STAGE_CODE_M2_FIT,
            NONAPPLICABLE_FOLD_ORDINAL,
            self.valid_index,
            PURPOSE_CODE_RADEMACHER_DESIGN,
        )

    def bit_generator(self) -> np.random.PCG64:
        if self.mode in DIRECT_SEED_MODES:
            return np.random.PCG64(self.trial_seed)
        assert self.null_replicate_seed is not None
        assert self.spawn_key is not None
        sequence = np.random.SeedSequence(
            self.null_replicate_seed,
            spawn_key=self.spawn_key,
        )
        return np.random.PCG64(sequence)

    def record(self) -> Mapping[str, Any]:
        return {
            "mode": self.mode,
            "trial_seed": self.trial_seed,
            "null_replicate_seed": self.null_replicate_seed,
            "valid_index": self.valid_index,
            "spawn_key": self.spawn_key,
        }


def _seed_route_identity(route: RademacherSeedRoute) -> tuple[Any, ...]:
    return (
        route.mode,
        route.trial_seed,
        route.null_replicate_seed,
        route.valid_index,
        route.spawn_key,
    )


def _require_issued_features(features: object) -> _FeatureIssuance:
    if type(features) is not RademacherFeatureLibrary:
        raise RademacherAdapterError(
            "Rademacher features have the wrong exact type"
        )
    issuance = _lookup_identity(_FEATURE_ISSUANCE, features)
    if issuance is None or features is not issuance.feature_reference():
        raise RademacherAdapterError(
            "Rademacher features lack exact issuance provenance"
        )
    candidates = issuance.candidates_reference()
    route = issuance.seed_route_reference()
    matrix = issuance.matrix_reference()
    if (
        candidates is None
        or route is None
        or matrix is None
        or features.candidates is not candidates
        or features.seed_route is not route
        or features.matrix is not matrix
    ):
        raise RademacherAdapterError(
            "Rademacher feature provenance was relabelled"
        )
    if type(candidates) is not m1.CandidateLibrary or type(route) is not RademacherSeedRoute:
        raise RademacherAdapterError("Rademacher feature provenance type changed")
    if _seed_route_identity(route) != issuance.seed_route_record:
        raise RademacherAdapterError("Rademacher seed route changed after issuance")
    if (
        matrix.shape != (ROW_COUNT, COLUMN_COUNT)
        or matrix.dtype != np.dtype(np.float64)
        or not matrix.flags.c_contiguous
        or matrix.flags.writeable
    ):
        raise RademacherAdapterError("Rademacher matrix changed after issuance")
    offsets = tuple(
        (key, bounds[0], bounds[1])
        for key, bounds in sorted(features._offsets.items())
    )
    if offsets != issuance.offsets:
        raise RademacherAdapterError("Rademacher K-block offsets changed after issuance")
    for key, statistics_issuance in tuple(issuance.statistics.items()):
        statistics = features._statistics.get(key)
        if (
            statistics is None
            or statistics is not statistics_issuance.statistics_reference()
        ):
            raise RademacherAdapterError(
                "Rademacher uniform-statistics cache was relabelled"
            )
        _require_byte_exact_array(
            statistics.mean,
            statistics_issuance.mean_reference,
            statistics_issuance.mean_snapshot,
            "Rademacher uniform mean",
        )
        _require_byte_exact_array(
            statistics.covariance,
            statistics_issuance.covariance_reference,
            statistics_issuance.covariance_snapshot,
            "Rademacher uniform covariance",
        )
        _require_byte_exact_array(
            statistics.contrast_cross_covariance,
            statistics_issuance.cross_reference,
            statistics_issuance.cross_snapshot,
            "Rademacher uniform cross covariance",
        )
    return issuance


def _centered_matrix_from_raw_words(words: np.ndarray) -> np.ndarray:
    """Decode exact low-to-high bits, reshape in C order, and center by K."""
    raw = np.asarray(words)
    if raw.dtype != np.dtype(np.uint64) or raw.shape != (RAW_WORD_COUNT,):
        raise RademacherAdapterError(
            f"Rademacher raw draw must contain exactly {RAW_WORD_COUNT} uint64 words"
        )

    # Force little-endian byte storage before unpacking.  With bitorder=little,
    # this is numeric word bit 0, ..., bit 63 even on a big-endian host.
    little_endian_words = raw.astype(np.dtype("<u8"), copy=False)
    decoded = np.unpackbits(
        little_endian_words.view(np.uint8),
        bitorder="little",
    )
    if decoded.shape != (DECODED_BIT_COUNT,):
        raise RademacherAdapterError("Rademacher raw-word decoding changed size")
    signs = decoded.astype(np.int8, copy=False)
    signs *= 2
    signs -= 1
    signed_matrix = signs.reshape((ROW_COUNT, COLUMN_COUNT), order="C")
    centered = signed_matrix.astype(np.float64)

    start = 0
    for k in range(m1.PANEL_SIZE + 1):
        count = math.comb(m1.PANEL_SIZE, k)
        stop = start + count
        integer_sum = np.sum(
            signed_matrix[start:stop], axis=0, dtype=np.int64
        )
        mean = integer_sum.astype(np.float64) / float(count)
        centered[start:stop] -= mean
        start = stop
    if start != ROW_COUNT:
        raise RademacherAdapterError("Rademacher K-block row count changed")
    if not np.all(np.isfinite(centered)):
        raise RademacherAdapterError("centered Rademacher dictionary is nonfinite")
    return _readonly(centered)


class RademacherFeatureLibrary:
    """One immutable trial-level dictionary in CandidateLibrary row order."""

    __slots__ = (
        "_candidates",
        "_seed_route",
        "_matrix",
        "_offsets",
        "_bases",
        "_statistics",
        "_sealed",
        "__weakref__",
    )

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise RademacherAdapterError(
                "issued Rademacher feature-library fields are immutable"
            )
        object.__setattr__(self, name, value)

    def __init__(
        self,
        candidates: m1.CandidateLibrary,
        seed_route: RademacherSeedRoute,
    ) -> None:
        if type(candidates) is not m1.CandidateLibrary:
            raise RademacherAdapterError(
                "Rademacher rows require the exact existing CandidateLibrary"
            )
        if type(seed_route) is not RademacherSeedRoute:
            raise RademacherAdapterError("Rademacher seed route is malformed")
        object.__setattr__(self, "_sealed", False)
        object.__setattr__(self, "_candidates", candidates)
        object.__setattr__(self, "_seed_route", seed_route)
        words = seed_route.bit_generator().random_raw(RAW_WORD_COUNT)
        object.__setattr__(self, "_matrix", _centered_matrix_from_raw_words(words))
        object.__setattr__(self, "_offsets", {})
        object.__setattr__(self, "_bases", {})
        object.__setattr__(self, "_statistics", {})
        start = 0
        for k in range(m1.PANEL_SIZE + 1):
            count = math.comb(m1.PANEL_SIZE, k)
            if candidates.basis(k).count != count:
                raise RademacherAdapterError(
                    "CandidateLibrary K-block order or count changed"
                )
            self._offsets[k] = (start, start + count)
            start += count
        if start != ROW_COUNT:
            raise RademacherAdapterError("candidate row universe is not 65,536")
        issuance = _FeatureIssuance(
            feature_reference=_weak_reference(self, "Rademacher feature library"),
            candidates_reference=_weak_reference(candidates, "Rademacher candidates"),
            seed_route_reference=_weak_reference(seed_route, "Rademacher seed route"),
            matrix_reference=_weak_reference(self._matrix, "Rademacher matrix"),
            seed_route_record=_seed_route_identity(seed_route),
            offsets=tuple(
                (key, bounds[0], bounds[1])
                for key, bounds in sorted(self._offsets.items())
            ),
            statistics={},
        )
        _register_identity(_FEATURE_ISSUANCE, self, issuance)
        object.__setattr__(self, "_sealed", True)

    @property
    def candidates(self) -> m1.CandidateLibrary:
        return self._candidates

    @property
    def seed_route(self) -> RademacherSeedRoute:
        return self._seed_route

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    def basis(self, k: int) -> m2.PairBasis:
        _require_issued_features(self)
        key = _exact_nonnegative_integer(k, "Rademacher K")
        if key not in self._offsets:
            raise RademacherAdapterError("Rademacher K is outside 0..16")
        if key not in self._bases:
            start, stop = self._offsets[key]
            block = self._matrix[start:stop]
            if block.shape != (math.comb(m1.PANEL_SIZE, key), COLUMN_COUNT):
                raise RademacherAdapterError("Rademacher K block has wrong shape")
            if block.flags.writeable or not np.shares_memory(block, self._matrix):
                raise RademacherAdapterError(
                    "Rademacher K block is not a read-only shared view"
                )
            self._bases[key] = m2.PairBasis(key, block)
        result = self._bases[key]
        start, stop = self._offsets[key]
        expected_block = self._matrix[start:stop]
        if (
            type(result) is not m2.PairBasis
            or result.k != key
            or result.indicators.shape != (stop - start, COLUMN_COUNT)
            or result.indicators.dtype != expected_block.dtype
            or result.indicators.strides != expected_block.strides
            or result.indicators.flags.writeable
            or not np.shares_memory(result.indicators, self._matrix)
            or result.indicators.__array_interface__["data"][0]
            != expected_block.__array_interface__["data"][0]
        ):
            raise RademacherAdapterError("Rademacher K-block cache lost provenance")
        return result

    def statistics(self, k: int) -> m2.PairUniformStatistics:
        feature_issuance = _require_issued_features(self)
        key = _exact_nonnegative_integer(k, "Rademacher K")
        if key not in self._statistics:
            if key in feature_issuance.statistics:
                raise RademacherAdapterError(
                    "Rademacher statistics cache was removed after issuance"
                )
            r_centered = self.basis(key).indicators
            contrasts = self.candidates.basis(key).contrasts
            contrast_centered = contrasts - np.mean(contrasts, axis=0)
            denominator = float(r_centered.shape[0])
            covariance = r_centered.T @ r_centered / denominator
            cross = contrast_centered.T @ r_centered / denominator
            mean = _readonly(np.zeros(COLUMN_COUNT, dtype=float))
            covariance = _readonly(
                m2._finite(
                    0.5 * (covariance + covariance.T),
                    "uniform Rademacher covariance",
                )
            )
            cross = _readonly(
                m2._finite(cross, "uniform target-contrast/Rademacher cross covariance")
            )
            self._statistics[key] = m2.PairUniformStatistics(
                mean=mean,
                covariance=covariance,
                contrast_cross_covariance=cross,
            )
            result = self._statistics[key]
            statistics_issuance = _StatisticsIssuance(
                statistics_reference=_weak_reference(
                    result, "Rademacher uniform statistics"
                ),
                mean_reference=_weak_reference(
                    result.mean, "Rademacher uniform mean"
                ),
                mean_snapshot=_readonly(result.mean),
                covariance_reference=_weak_reference(
                    result.covariance, "Rademacher uniform covariance"
                ),
                covariance_snapshot=_readonly(result.covariance),
                cross_reference=_weak_reference(
                    result.contrast_cross_covariance,
                    "Rademacher uniform cross covariance",
                ),
                cross_snapshot=_readonly(result.contrast_cross_covariance),
            )
            with _ISSUANCE_LOCK:
                feature_issuance.statistics[key] = statistics_issuance
        result = self._statistics[key]
        statistics_issuance = feature_issuance.statistics.get(key)
        if (
            statistics_issuance is None
            or result is not statistics_issuance.statistics_reference()
        ):
            raise RademacherAdapterError(
                "Rademacher uniform-statistics object lost issuance provenance"
            )
        if (
            type(result) is not m2.PairUniformStatistics
            or result.mean.shape != (COLUMN_COUNT,)
            or result.covariance.shape != (COLUMN_COUNT, COLUMN_COUNT)
            or result.contrast_cross_covariance.shape
            != (m1.TARGET_CONTRAST_DIMENSION, COLUMN_COUNT)
            or result.mean.flags.writeable
            or result.covariance.flags.writeable
            or result.contrast_cross_covariance.flags.writeable
        ):
            raise RademacherAdapterError(
                "Rademacher uniform-statistics cache lost provenance"
            )
        _require_byte_exact_array(
            result.mean,
            statistics_issuance.mean_reference,
            statistics_issuance.mean_snapshot,
            "Rademacher uniform mean",
        )
        _require_byte_exact_array(
            result.covariance,
            statistics_issuance.covariance_reference,
            statistics_issuance.covariance_snapshot,
            "Rademacher uniform covariance",
        )
        _require_byte_exact_array(
            result.contrast_cross_covariance,
            statistics_issuance.cross_reference,
            statistics_issuance.cross_snapshot,
            "Rademacher uniform cross covariance",
        )
        return result

    def all_counts(self) -> dict[int, int]:
        return {
            k: self.basis(k).count for k in range(m1.PANEL_SIZE + 1)
        }


def _cell_fields(cell: m1.Cell) -> tuple[Any, ...]:
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


def _candidate_basis_snapshot(
    candidates: m1.CandidateLibrary,
    k: int,
) -> _CandidateBasisSnapshot:
    basis = candidates.basis(k)
    if type(basis) is not m1.CandidateBasis or basis.k != k:
        raise RademacherAdapterError("Rademacher candidate basis is malformed")
    return _CandidateBasisSnapshot(
        k=k,
        basis_reference=_weak_reference(basis, "Rademacher candidate basis"),
        membership_reference=_weak_reference(
            basis.membership, "Rademacher candidate membership"
        ),
        membership_snapshot=_readonly(basis.membership),
        contrasts_reference=_weak_reference(
            basis.contrasts, "Rademacher candidate contrasts"
        ),
        contrasts_snapshot=_readonly(basis.contrasts),
    )


def _snapshot_prepared(
    prepared: m2.PreparedM2,
    cells: Sequence[m1.Cell],
    moments: Sequence[m1.UniformCellMoments],
    candidates: m1.CandidateLibrary,
) -> _PreparedSnapshot:
    if type(prepared) is not m2.PreparedM2 or type(prepared.m1_data) is not m1.PreparedData:
        raise RademacherAdapterError("Rademacher PreparedM2 is malformed")
    exact_cells = tuple(cells)
    exact_moments = tuple(moments)
    if len(exact_cells) != len(exact_moments) or len(exact_cells) != len(prepared.cells):
        raise RademacherAdapterError("Rademacher prepared rows are not aligned")
    if any(
        prepared.cells[index] is not exact_cells[index]
        for index in range(len(exact_cells))
    ):
        raise RademacherAdapterError("Rademacher prepared cell identity changed")
    data = prepared.m1_data
    design = data.design_transform
    return _PreparedSnapshot(
        prepared_reference=_weak_reference(prepared, "Rademacher PreparedM2"),
        m1_data_reference=_weak_reference(data, "Rademacher M1 preparation"),
        transform_reference=_weak_reference(prepared.transform, "Rademacher transform"),
        cell_references=tuple(
            _weak_reference(cell, "Rademacher prepared cell") for cell in exact_cells
        ),
        cell_fields=tuple(_cell_fields(cell) for cell in exact_cells),
        soma_distance_references=tuple(
            _weak_reference(cell.soma_distance_mm, "Rademacher cell distance")
            for cell in exact_cells
        ),
        soma_distance_snapshots=tuple(
            _readonly(cell.soma_distance_mm) for cell in exact_cells
        ),
        moment_references=tuple(
            _weak_reference(moment, "Rademacher uniform moments")
            for moment in exact_moments
        ),
        moment_geometry_references=tuple(
            _weak_reference(moment.geometry_mean, "Rademacher geometry mean")
            for moment in exact_moments
        ),
        moment_geometry_snapshots=tuple(
            _readonly(moment.geometry_mean) for moment in exact_moments
        ),
        moment_covariance_references=tuple(
            _weak_reference(moment.raw_joint_covariance, "Rademacher joint covariance")
            for moment in exact_moments
        ),
        moment_covariance_snapshots=tuple(
            _readonly(moment.raw_joint_covariance) for moment in exact_moments
        ),
        x_reference=_weak_reference(data.x, "Rademacher prepared design"),
        x_snapshot=_readonly(data.x),
        geometry_centers_reference=_weak_reference(
            data.geometry_centers, "Rademacher geometry centers"
        ),
        geometry_centers_snapshot=_readonly(data.geometry_centers),
        geometry_scales_reference=_weak_reference(
            data.geometry_scales, "Rademacher prepared geometry scales"
        ),
        geometry_scales_snapshot=_readonly(data.geometry_scales),
        weights_reference=_weak_reference(data.weights, "Rademacher weights"),
        weights_snapshot=_readonly(data.weights),
        design_transform_reference=_weak_reference(
            design, "Rademacher design transform"
        ),
        soma_mean_reference=_weak_reference(design.soma_mean, "Rademacher soma mean"),
        soma_mean_snapshot=_readonly(design.soma_mean),
        soma_scale_reference=_weak_reference(design.soma_scale, "Rademacher soma scale"),
        soma_scale_snapshot=_readonly(design.soma_scale),
        candidates_reference=_weak_reference(candidates, "Rademacher candidates"),
        candidate_bases=tuple(
            _candidate_basis_snapshot(candidates, k)
            for k in sorted({cell.k for cell in exact_cells})
        ),
    )


def _require_exact_array(
    current: np.ndarray,
    reference: Any,
    snapshot: np.ndarray,
    label: str,
) -> None:
    if current is not reference():
        raise RademacherAdapterError(f"{label} was relabelled after issuance")
    if (
        current.dtype != snapshot.dtype
        or current.shape != snapshot.shape
        or not np.array_equal(current, snapshot)
    ):
        raise RademacherAdapterError(f"{label} changed after issuance")


def _require_byte_exact_array(
    current: np.ndarray,
    reference: Any,
    snapshot: np.ndarray,
    label: str,
) -> None:
    if current is not reference():
        raise RademacherAdapterError(f"{label} was relabelled after issuance")
    if (
        current.dtype != snapshot.dtype
        or current.shape != snapshot.shape
        or current.strides != snapshot.strides
        or not np.array_equal(
            current.view(np.uint8).reshape(-1),
            snapshot.view(np.uint8).reshape(-1),
        )
    ):
        raise RademacherAdapterError(f"{label} changed bytewise after issuance")


def _require_candidate_basis_snapshot(
    candidates: m1.CandidateLibrary,
    snapshot: _CandidateBasisSnapshot,
) -> None:
    basis = candidates.basis(snapshot.k)
    if basis is not snapshot.basis_reference():
        raise RademacherAdapterError(
            f"Rademacher K={snapshot.k} candidate basis was relabelled"
        )
    _require_byte_exact_array(
        basis.membership,
        snapshot.membership_reference,
        snapshot.membership_snapshot,
        f"Rademacher K={snapshot.k} candidate membership",
    )
    _require_byte_exact_array(
        basis.contrasts,
        snapshot.contrasts_reference,
        snapshot.contrasts_snapshot,
        f"Rademacher K={snapshot.k} candidate contrasts",
    )


def _require_prepared_snapshot(
    prepared: m2.PreparedM2,
    snapshot: _PreparedSnapshot,
) -> None:
    if prepared is not snapshot.prepared_reference():
        raise RademacherAdapterError("Rademacher PreparedM2 identity changed")
    data = prepared.m1_data
    if data is not snapshot.m1_data_reference():
        raise RademacherAdapterError("Rademacher M1 preparation was relabelled")
    if prepared.transform is not snapshot.transform_reference():
        raise RademacherAdapterError("Rademacher transform was relabelled")
    candidates = snapshot.candidates_reference()
    if candidates is None:
        raise RademacherAdapterError("Rademacher candidates are no longer live")
    for basis_snapshot in snapshot.candidate_bases:
        _require_candidate_basis_snapshot(candidates, basis_snapshot)
    if len(prepared.cells) != len(snapshot.cell_references):
        raise RademacherAdapterError("Rademacher prepared-cell count changed")
    for index, cell in enumerate(prepared.cells):
        if cell is not snapshot.cell_references[index]():
            raise RademacherAdapterError("Rademacher prepared-cell identity changed")
        if _cell_fields(cell) != snapshot.cell_fields[index]:
            raise RademacherAdapterError("Rademacher prepared-cell fields changed")
        _require_exact_array(
            cell.soma_distance_mm,
            snapshot.soma_distance_references[index],
            snapshot.soma_distance_snapshots[index],
            "Rademacher cell soma-distance array",
        )
    for index, reference in enumerate(snapshot.moment_references):
        moment = reference()
        if moment is None:
            raise RademacherAdapterError("Rademacher uniform moments are no longer live")
        _require_exact_array(
            moment.geometry_mean,
            snapshot.moment_geometry_references[index],
            snapshot.moment_geometry_snapshots[index],
            "Rademacher uniform geometry mean",
        )
        _require_exact_array(
            moment.raw_joint_covariance,
            snapshot.moment_covariance_references[index],
            snapshot.moment_covariance_snapshots[index],
            "Rademacher uniform covariance",
        )
    _require_exact_array(data.x, snapshot.x_reference, snapshot.x_snapshot, "Rademacher design")
    _require_exact_array(
        data.geometry_centers,
        snapshot.geometry_centers_reference,
        snapshot.geometry_centers_snapshot,
        "Rademacher geometry centers",
    )
    _require_exact_array(
        data.geometry_scales,
        snapshot.geometry_scales_reference,
        snapshot.geometry_scales_snapshot,
        "Rademacher prepared geometry scales",
    )
    _require_exact_array(
        data.weights,
        snapshot.weights_reference,
        snapshot.weights_snapshot,
        "Rademacher weights",
    )
    design = data.design_transform
    if design is not snapshot.design_transform_reference():
        raise RademacherAdapterError("Rademacher design transform was relabelled")
    _require_exact_array(
        design.soma_mean,
        snapshot.soma_mean_reference,
        snapshot.soma_mean_snapshot,
        "Rademacher soma mean",
    )
    _require_exact_array(
        design.soma_scale,
        snapshot.soma_scale_reference,
        snapshot.soma_scale_snapshot,
        "Rademacher soma scale",
    )


@dataclass(frozen=True)
class RademacherFoldFit:
    """One noise-specific fold transform bound to its shared dictionary."""

    features: RademacherFeatureLibrary
    physical_prepared: m2.PreparedM2
    transform: m2.ResidualTransform
    prepared: m2.PreparedM2

    @property
    def physical_transform(self) -> m2.ResidualTransform:
        return self.physical_prepared.transform

    def __post_init__(self) -> None:
        if self.prepared.transform is not self.transform:
            raise RademacherAdapterError(
                "Rademacher prepared fold lost its transform identity"
            )
        if self.prepared.m1_data is not self.physical_prepared.m1_data:
            raise RademacherAdapterError(
                "Rademacher fold did not reuse the exact physical M1 preparation"
            )
        physical_transform = self.physical_transform
        if (
            self.transform.design_transform is not physical_transform.design_transform
            or self.transform.geometry_scales is not physical_transform.geometry_scales
            or self.transform.pair_residual_scales
            is not physical_transform.pair_residual_scales
            or self.transform.range_basis is not physical_transform.range_basis
            or self.transform.range_projector
            is not physical_transform.range_projector
        ):
            raise RademacherAdapterError(
                "Rademacher fold did not retain exact physical normalization objects"
            )


def _require_issued_fold(fitted: object) -> _FoldIssuance:
    if type(fitted) is not RademacherFoldFit:
        raise RademacherAdapterError("Rademacher fold fit has the wrong exact type")
    issuance = _lookup_identity(_FOLD_ISSUANCE, fitted)
    if issuance is None or fitted is not issuance.fitted_reference():
        raise RademacherAdapterError("Rademacher fold fit lacks exact issuance")
    _require_issued_features(fitted.features)
    if (
        fitted.features is not issuance.features_reference()
        or fitted.physical_prepared is not issuance.physical_prepared_reference()
        or fitted.transform is not issuance.transform_reference()
        or fitted.prepared is not issuance.prepared_reference()
    ):
        raise RademacherAdapterError("Rademacher fold fields were relabelled")
    try:
        physical = m2._reauthenticate_physical_fold_provenance(
            fitted.physical_prepared
        )
    except m2.M2Error as error:
        raise RademacherAdapterError(
            "Rademacher physical fold provenance changed"
        ) from error
    if (
        physical.candidates is not issuance.candidates_reference()
        or physical.geometry is not issuance.geometry_reference()
        or physical.pairs is not issuance.physical_pairs_reference()
        or physical.selected_indexes != issuance.selected_indexes
        or physical.candidate_ks != issuance.physical_candidate_ks
    ):
        raise RademacherAdapterError("Rademacher physical-fold binding changed")
    physical_transform = fitted.physical_transform
    transform = fitted.transform
    if (
        transform.design_transform is not physical_transform.design_transform
        or transform.geometry_scales is not physical_transform.geometry_scales
        or transform.pair_residual_scales
        is not physical_transform.pair_residual_scales
        or transform.range_basis is not physical_transform.range_basis
        or transform.range_projector is not physical_transform.range_projector
    ):
        raise RademacherAdapterError(
            "Rademacher physical normalization identity changed"
        )
    _require_exact_array(
        transform.projection_f_to_h,
        issuance.projection_reference,
        issuance.projection_snapshot,
        "Rademacher F-to-R projection",
    )
    _require_exact_array(
        transform.residual_gram,
        issuance.residual_gram_reference,
        issuance.residual_gram_snapshot,
        "Rademacher residual Gram",
    )
    _require_exact_array(
        transform.scaled_residual_gram,
        issuance.scaled_gram_reference,
        issuance.scaled_gram_snapshot,
        "Rademacher projected Gram",
    )
    if (
        transform.diagnostics.get("feature_kind")
        != "centered_Rademacher_noise_pseudofeatures"
        or transform.diagnostics.get("seed_route")
        != fitted.features.seed_route.record()
    ):
        raise RademacherAdapterError("Rademacher transform provenance changed")
    _require_prepared_snapshot(fitted.prepared, issuance.prepared_snapshot)
    return issuance


def _require_issued_prepared(
    fitted: RademacherFoldFit,
    prepared: m2.PreparedM2,
) -> _PreparedSnapshot:
    fold = _require_issued_fold(fitted)
    issuance = _lookup_identity(_PREPARED_ISSUANCE, prepared)
    if issuance is None:
        raise RademacherAdapterError("Rademacher PreparedM2 lacks exact issuance")
    if isinstance(issuance, _FoldIssuance):
        if issuance is not fold or prepared is not fitted.prepared:
            raise RademacherAdapterError("Rademacher training preparation was relabelled")
        snapshot = issuance.prepared_snapshot
    else:
        if (
            prepared is not issuance.prepared_reference()
            or fitted is not issuance.fitted_reference()
        ):
            raise RademacherAdapterError("Rademacher validation binding changed")
        snapshot = issuance.prepared_snapshot
    if prepared.transform is not fitted.transform:
        raise RademacherAdapterError("Rademacher preparation lost its fold transform")
    _require_prepared_snapshot(prepared, snapshot)
    return snapshot


def _core_authenticate_fold_fit(fitted: object) -> bool:
    try:
        _require_issued_fold(fitted)
    except (RademacherAdapterError, m2.M2Error):
        return False
    return True


def _core_authenticate_prepared(
    fitted: object,
    prepared: object,
) -> bool:
    try:
        if type(fitted) is not RademacherFoldFit or type(prepared) is not m2.PreparedM2:
            return False
        _require_issued_prepared(fitted, prepared)
    except (RademacherAdapterError, m2.M2Error):
        return False
    return True


def _core_authenticate_problem(
    fitted: object,
    prepared: object,
    candidates: object,
    geometry: object,
    features: object,
) -> bool:
    try:
        if type(fitted) is not RademacherFoldFit or type(prepared) is not m2.PreparedM2:
            return False
        issuance = _require_issued_fold(fitted)
        _require_issued_prepared(fitted, prepared)
        if (
            candidates is not issuance.candidates_reference()
            or geometry is not issuance.geometry_reference()
            or features is not fitted.features
            or features is not issuance.features_reference()
        ):
            return False
        _require_issued_features(features)
    except (RademacherAdapterError, m2.M2Error):
        return False
    return True


def fit_rademacher_fold(
    cells: Sequence[m1.Cell],
    moments: Sequence[m1.UniformCellMoments],
    candidates: m1.CandidateLibrary,
    geometry: m1.GeometryEngine,
    features: RademacherFeatureLibrary,
    physical_prepared: m2.PreparedM2,
    physical_pairs: m2.PairFeatureLibrary,
    indexes: Sequence[int] | None = None,
    *,
    expected_rank: int = EXPECTED_PROJECTED_RANK,
) -> RademacherFoldFit:
    """Fit ``A_R`` from the exact same-pass physical ``G_FF`` and pseudoinverse."""
    if len(cells) != len(moments):
        raise RademacherAdapterError("cells and uniform moments are not aligned")
    _require_issued_features(features)
    if features.candidates is not candidates:
        raise RademacherAdapterError(
            "Rademacher dictionary and fold use different CandidateLibrary objects"
        )
    selected = (
        tuple(range(len(cells)))
        if indexes is None
        else tuple(
            _exact_nonnegative_integer(index, "Rademacher training index")
            for index in indexes
        )
    )
    if not selected or len(set(selected)) != len(selected):
        raise RademacherAdapterError(
            "Rademacher training indexes must be nonempty and unique"
        )
    if any(index < 0 or index >= len(cells) for index in selected):
        raise RademacherAdapterError("Rademacher training index is out of range")
    expected = _exact_nonnegative_integer(
        expected_rank, "expected Rademacher rank"
    )
    if expected < 1:
        raise RademacherAdapterError("expected Rademacher rank must be positive")

    try:
        physical_provenance = m2._require_physical_fold_provenance(
            physical_prepared,
            cells,
            moments,
            candidates,
            geometry,
            physical_pairs,
            selected,
        )
    except m2.M2Error as error:
        raise RademacherAdapterError(
            "physical fold does not match exact same-pass provenance"
        ) from error
    physical_transform = physical_prepared.transform
    prepared_m1 = physical_prepared.m1_data
    if (
        physical_provenance.physical_rank != expected
        or physical_transform.diagnostics.get("training_cells") != len(selected)
        or physical_provenance.training_groups
        != len({cells[index].group for index in selected})
    ):
        raise RademacherAdapterError(
            "physical fold lacks the required stable rank provenance"
        )
    gram_ff = physical_provenance.gram_ff
    inverse_ff = physical_provenance.inverse_ff
    ff_tolerance = physical_provenance.ff_tolerance
    ff_rank = physical_provenance.ff_rank

    gram_fr = np.zeros((m1.PARAMETER_DIMENSION, COLUMN_COUNT), dtype=float)
    gram_rr = np.zeros((COLUMN_COUNT, COLUMN_COUNT), dtype=float)
    for local_index, cell in enumerate(prepared_m1.cells):
        weight = float(prepared_m1.weights[local_index])
        statistics = features.statistics(cell.k)
        target_cross = np.einsum(
            "a,rp->arp",
            prepared_m1.x[local_index],
            statistics.contrast_cross_covariance,
            optimize=True,
        ).reshape(m1.B_PARAMETER_DIMENSION, COLUMN_COUNT)
        gram_fr[: m1.B_PARAMETER_DIMENSION] += weight * target_cross

        basis = candidates.basis(cell.k)
        geometry_cross = np.zeros(
            (m1.GEOMETRY_DIMENSION, COLUMN_COUNT), dtype=float
        )
        for start in range(0, basis.count, m1.BATCH_MAXIMUM):
            stop = min(start + m1.BATCH_MAXIMUM, basis.count)
            raw = geometry.raw_batch(cell, start, stop)
            z = (
                raw - prepared_m1.geometry_centers[local_index]
            ) / prepared_m1.geometry_scales
            geometry_cross += z.T @ features.basis(cell.k).indicators[start:stop]
        geometry_cross /= float(basis.count)
        gram_fr[m1.B_PARAMETER_DIMENSION :] += weight * geometry_cross
        gram_rr += weight * statistics.covariance

    gram_ff = m2._finite(0.5 * (gram_ff + gram_ff.T), "physical G_FF")
    gram_fr = m2._finite(gram_fr, "G_FR")
    gram_rr = m2._finite(0.5 * (gram_rr + gram_rr.T), "G_RR")
    projection = inverse_ff @ gram_fr
    orthogonality = gram_fr - gram_ff @ projection
    orthogonality_error = float(np.max(np.abs(orthogonality)))
    if orthogonality_error > m2.PROJECTION_TOLERANCE:
        raise RademacherAdapterError(
            "Rademacher residual features are not orthogonal to the M1 span"
        )

    residual_gram_schur = (
        gram_rr
        - projection.T @ gram_fr
        - gram_fr.T @ projection
        + projection.T @ gram_ff @ projection
    )
    residual_gram_schur = m2._finite(
        0.5 * (residual_gram_schur + residual_gram_schur.T),
        "Schur Rademacher residual Gram",
    )

    residual_gram = np.zeros((COLUMN_COUNT, COLUMN_COUNT), dtype=float)
    projected_gram = np.zeros((COLUMN_COUNT, COLUMN_COUNT), dtype=float)
    projection_target = projection[: m1.B_PARAMETER_DIMENSION].reshape(
        m1.DESIGN_DIMENSION,
        m1.TARGET_CONTRAST_DIMENSION,
        COLUMN_COUNT,
    )
    projection_geometry = projection[m1.B_PARAMETER_DIMENSION :]
    physical_scales = physical_transform.pair_residual_scales
    physical_projector = physical_transform.range_projector
    for local_index, cell in enumerate(prepared_m1.cells):
        basis = candidates.basis(cell.k)
        contrast_mean = np.mean(basis.contrasts, axis=0)
        cell_target_projection = np.einsum(
            "a,arp->rp",
            prepared_m1.x[local_index],
            projection_target,
            optimize=True,
        )
        cell_residual_second = np.zeros(
            (COLUMN_COUNT, COLUMN_COUNT), dtype=float
        )
        cell_projected_second = np.zeros(
            (COLUMN_COUNT, COLUMN_COUNT), dtype=float
        )
        for start in range(0, basis.count, m1.BATCH_MAXIMUM):
            stop = min(start + m1.BATCH_MAXIMUM, basis.count)
            raw = geometry.raw_batch(cell, start, stop)
            z = (
                raw - prepared_m1.geometry_centers[local_index]
            ) / prepared_m1.geometry_scales
            a_centered = basis.contrasts[start:stop] - contrast_mean
            residual = features.basis(cell.k).indicators[start:stop] - (
                a_centered @ cell_target_projection
                + z @ projection_geometry
            )
            projected = (residual / physical_scales) @ physical_projector
            cell_residual_second += residual.T @ residual
            cell_projected_second += projected.T @ projected
        weight = float(prepared_m1.weights[local_index]) / float(basis.count)
        residual_gram += weight * cell_residual_second
        projected_gram += weight * cell_projected_second

    residual_gram = m2._finite(
        0.5 * (residual_gram + residual_gram.T),
        "direct Rademacher residual Gram",
    )
    projected_gram = m2._finite(
        0.5 * (projected_gram + projected_gram.T),
        "physical-range-projected Rademacher training Gram",
    )
    schur_direct_error = float(
        np.max(np.abs(residual_gram - residual_gram_schur))
    )

    eigenvalues, _ = m2._symmetric_eigensystem(
        projected_gram,
        "physical-range-projected Rademacher training Gram",
    )
    largest = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
    rank_tolerance = (
        COLUMN_COUNT * np.finfo(float).eps * max(1.0, largest)
    )
    if eigenvalues.size and float(np.min(eigenvalues)) < -10.0 * rank_tolerance:
        raise RademacherAdapterError(
            "projected Rademacher training Gram is materially indefinite"
        )
    ranks = {
        "tau_over_10": int(np.sum(eigenvalues > rank_tolerance / 10.0)),
        "tau": int(np.sum(eigenvalues > rank_tolerance)),
        "ten_tau": int(np.sum(eigenvalues > 10.0 * rank_tolerance)),
    }
    if len(set(ranks.values())) != 1 or ranks["tau"] != expected:
        raise RademacherAdapterError(
            "projected Rademacher rank is not the frozen stable rank: "
            f"{ranks}, expected {expected}"
        )

    diagnostics: dict[str, Any] = {
        "feature_kind": "centered_Rademacher_noise_pseudofeatures",
        "training_cells": len(selected),
        "training_groups": len({cells[index].group for index in selected}),
        "dictionary_rows": ROW_COUNT,
        "dictionary_columns": COLUMN_COUNT,
        "dictionary_read_only": not features.matrix.flags.writeable,
        "seed_route": features.seed_route.record(),
        "uniform_Omega_centering": True,
        "equal_group_weighting": True,
        "G_FF_rank": ff_rank,
        "G_FF_tolerance": ff_tolerance,
        "G_FF_same_pass_physical_provenance": True,
        "G_FF_pseudoinverse_same_pass_physical_provenance": True,
        "physical_geometry_centers_reused": True,
        "G_FR_projection_maximum_absolute_orthogonality_error": orthogonality_error,
        "Schur_vs_direct_residual_Gram_maximum_absolute_error": schur_direct_error,
        "physical_pair_residual_RMS_reused": True,
        "physical_range_basis_reused": True,
        "physical_range_projector_reused": True,
        "rank_tolerance": rank_tolerance,
        "rank_at_tau_over_10_tau_ten_tau": ranks,
        "rank_stable": True,
        "rank": ranks["tau"],
        "expected_rank": expected,
        "expected_rank_match": True,
    }
    transform = m2.ResidualTransform(
        design_transform=physical_transform.design_transform,
        geometry_scales=physical_transform.geometry_scales,
        projection_f_to_h=m2._finite(projection, "F-to-R projection"),
        pair_residual_scales=physical_transform.pair_residual_scales,
        range_basis=physical_transform.range_basis,
        range_projector=physical_transform.range_projector,
        residual_gram=residual_gram,
        scaled_residual_gram=projected_gram,
        diagnostics=diagnostics,
    )
    prepared = m2.PreparedM2(prepared_m1, transform)
    fitted = RademacherFoldFit(
        features=features,
        physical_prepared=physical_prepared,
        transform=transform,
        prepared=prepared,
    )
    selected_cells = tuple(cells[index] for index in selected)
    selected_moments = tuple(moments[index] for index in selected)
    prepared_snapshot = _snapshot_prepared(
        prepared, selected_cells, selected_moments, candidates
    )
    issuance = _FoldIssuance(
        fitted_reference=_weak_reference(fitted, "Rademacher fold fit"),
        features_reference=_weak_reference(features, "Rademacher features"),
        physical_prepared_reference=_weak_reference(
            physical_prepared, "physical PreparedM2"
        ),
        transform_reference=_weak_reference(transform, "Rademacher transform"),
        prepared_reference=_weak_reference(prepared, "Rademacher training preparation"),
        candidates_reference=_weak_reference(candidates, "Rademacher candidates"),
        geometry_reference=_weak_reference(geometry, "Rademacher geometry"),
        physical_pairs_reference=_weak_reference(
            physical_pairs, "physical pair features"
        ),
        selected_indexes=selected,
        physical_candidate_ks=physical_provenance.candidate_ks,
        projection_reference=_weak_reference(
            transform.projection_f_to_h, "Rademacher F-to-R projection"
        ),
        projection_snapshot=_readonly(transform.projection_f_to_h),
        residual_gram_reference=_weak_reference(
            transform.residual_gram, "Rademacher residual Gram"
        ),
        residual_gram_snapshot=_readonly(transform.residual_gram),
        scaled_gram_reference=_weak_reference(
            transform.scaled_residual_gram, "Rademacher projected Gram"
        ),
        scaled_gram_snapshot=_readonly(transform.scaled_residual_gram),
        prepared_snapshot=prepared_snapshot,
    )
    _register_identity(_FOLD_ISSUANCE, fitted, issuance)
    _register_identity(_PREPARED_ISSUANCE, prepared, issuance)
    m2._register_exact_rademacher_fold(fitted)
    return fitted


def apply_rademacher_validation(
    fitted: RademacherFoldFit,
    cells: Sequence[m1.Cell],
    moments: Sequence[m1.UniformCellMoments],
) -> m2.PreparedM2:
    """Apply the frozen training-only noise transform to held-out cells."""
    issuance = _require_issued_fold(fitted)
    validation_ks = {cell.k for cell in cells}
    if not validation_ks.issubset(set(issuance.physical_candidate_ks)):
        raise RademacherAdapterError(
            "validation K was not anchored by the physical fit's full cell view"
        )
    prepared = m2.apply_residual_transform(cells, moments, fitted.transform)
    if prepared.transform is not fitted.transform:
        raise RademacherAdapterError("validation application copied the fold transform")
    snapshot = _snapshot_prepared(
        prepared, cells, moments, fitted.features.candidates
    )
    issuance = _ValidationIssuance(
        prepared_reference=_weak_reference(prepared, "Rademacher validation PreparedM2"),
        fitted_reference=_weak_reference(fitted, "Rademacher fold fit"),
        prepared_snapshot=snapshot,
    )
    _register_identity(_PREPARED_ISSUANCE, prepared, issuance)
    m2._register_exact_rademacher_prepared(fitted, prepared)
    return prepared
