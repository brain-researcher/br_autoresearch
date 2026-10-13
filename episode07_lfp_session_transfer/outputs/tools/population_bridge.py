"""Fitting-only EP07 population bases and the registered history bridge.

Candidate convention: centered, unscaled PCA of unresidualized native counts,
with equal weight per fitting trial/native-bin row. Direction/time anchors are
formed before residualization, projected into that day's basis, then centered
across their 8 * native_bins rows. Numeric neuron identities are never matched
between days. Alignment is an explicit candidate choice, not a chosen winner.

A history bridge and its source-dependent selected rank are HISTORY-ONLY
state. The separate today-only basis function accepts target calibration only;
this module does not transfer the history-selected rank or objects to it.
Final capacity matching and predictor assembly remain the caller's scientific
integration work. No input reader, evaluation, residual fitter or launch exists
here. Calls require an admitted fitting context and its resource check.
"""
from dataclasses import dataclass
import math

import numpy as np

from native_trials import DayTrials, _snapshot, _validate_day
from view_access import EARLIER


RANKS = (16, 12, 8, 4)
ALIGNMENTS = ("none", "orthogonal", "regularized_correlation")


class BridgeUnavailable(ValueError):
    """Required scientific rank/support is absent; do not drop a source."""


@dataclass(frozen=True)
class PopulationBasis:
    animal: str
    day: str
    target_day: str
    region: str
    stream: str
    role: str
    trial_ids: tuple
    bin_offsets: np.ndarray
    unit_guide: np.ndarray
    rank: int
    mean: np.ndarray
    components: np.ndarray
    centered_anchors: np.ndarray
    anchor_condition: float


@dataclass(frozen=True)
class HistoryLink:
    source: PopulationBasis
    matrix: np.ndarray
    support_ratio: float


@dataclass(frozen=True)
class HistoryBridge:
    target: PopulationBasis
    sources: tuple[HistoryLink, ...]
    rank: int
    alignment: str
    regularization: float | None
    rejected_ranks: tuple
    fitting_stream: str = "history_assisted"


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned resource check required")
    check()


def _fit_arrays(day, check):
    if not isinstance(day, DayTrials) or day.view_role != "fit":
        raise PermissionError("population coordinates use fitting responses only")
    directions, _ = _validate_day(day, check)
    _check(check)
    y = np.array(day.spikes, dtype=np.float64, copy=True)
    rows = y.reshape(-1, y.shape[-1])
    mean = np.mean(rows, axis=0)
    centered = rows - mean
    if not np.isfinite(centered).all():
        raise BridgeUnavailable("nonfinite_centered_fitting_counts")
    _check(check)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    tolerance = (np.finfo(np.float64).eps * max(centered.shape) * singular[0]
                 if len(singular) else 0.0)
    numerical_rank = int(np.count_nonzero(singular > tolerance))
    # All eight directions have already passed the complete fitting-role gate.
    anchors = np.stack([np.mean(y[directions == direction], axis=0)
                        for direction in range(8)]).reshape(-1, y.shape[-1])
    _check(check)
    return mean, vt, numerical_rank, anchors


def _basis(day, arrays, rank, check):
    if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)) or rank not in RANKS:
        raise ValueError("rank must be one of 4, 8, 12, 16")
    rank = int(rank)
    mean, vt, numerical_rank, anchors = arrays
    if numerical_rank < rank:
        raise BridgeUnavailable("fitting_population_rank_below_candidate")
    if len(anchors) < 64 or len(anchors) < 4 * rank:
        raise BridgeUnavailable("insufficient_native_anchor_cells")
    components = vt[:rank].T.copy()
    # Sign convention is reproducibility only; no physiological axis identity.
    for column in range(rank):
        pivot = int(np.argmax(np.abs(components[:, column])))
        if components[pivot, column] < 0:
            components[:, column] *= -1
    projected = (anchors - mean) @ components
    projected -= np.mean(projected, axis=0)
    _check(check)
    singular = np.linalg.svd(projected, compute_uv=False)
    tolerance = np.finfo(np.float64).eps * max(projected.shape) * singular[0]
    if (len(singular) != rank or singular[-1] <= tolerance
            or not np.isfinite(singular).all()):
        raise BridgeUnavailable("centered_anchor_rank_below_candidate")
    condition = float(singular[0] / singular[-1])
    if condition > 30:
        raise BridgeUnavailable("centered_anchor_condition_above_30")
    return PopulationBasis(day.animal, day.day, day.target_day, day.region,
        day.stream, day.role, day.trial_ids, _snapshot(day.bin_offsets),
        _snapshot(day.unit_guide), rank, _snapshot(mean), _snapshot(components),
        _snapshot(projected), condition)


def fit_today_calibration_basis(day, *, rank, check):
    """Separate target-calibration state for an explicitly supplied candidate rank.

    The caller must choose the rank without consulting a history-fitted object.
    No fallback driven by source links, earlier-day basis or source responses
    enters this function. This does not itself resolve final capacity matching.
    """
    _check(check)
    if (not isinstance(day, DayTrials) or day.stream != "today_only"
            or day.role != "calibration" or day.day != day.target_day):
        raise PermissionError("today-only basis requires its own target calibration pack")
    return _basis(day, _fit_arrays(day, check), rank, check)


def _matrix(source, target, alignment, regularization, check):
    a, b = source.centered_anchors, target.centered_anchors
    if a.shape != b.shape or not np.array_equal(source.bin_offsets, target.bin_offsets):
        raise ValueError("source/target native anchor cells must correspond exactly")
    _check(check)
    qs = np.linalg.qr(a, mode="reduced")[0]
    qt = np.linalg.qr(b, mode="reduced")[0]
    singular = np.linalg.svd(qs.T @ qt, compute_uv=False)
    ratio = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.0
    if not math.isfinite(ratio) or ratio < 0.05:
        raise BridgeUnavailable("source_target_Q_support_ratio_below_0p05")
    if alignment == "none":
        matrix = np.eye(source.rank, dtype=np.float64)
    elif alignment == "orthogonal":
        u, _, vt = np.linalg.svd(a.T @ b, full_matrices=False)
        matrix = u @ vt
    else:
        # Ridge-regularized whitening, canonical rotation, target unwhitening.
        # The caller supplies the positive ridge value as a candidate parameter.
        covariance_a = a.T @ a / (len(a) - 1) + regularization * np.eye(source.rank)
        covariance_b = b.T @ b / (len(b) - 1) + regularization * np.eye(source.rank)
        eigen_a, vectors_a = np.linalg.eigh(covariance_a)
        eigen_b, vectors_b = np.linalg.eigh(covariance_b)
        if min(eigen_a[0], eigen_b[0]) <= 0:
            raise BridgeUnavailable("nonpositive_regularized_anchor_covariance")
        whiten_a = (vectors_a / np.sqrt(eigen_a)) @ vectors_a.T
        whiten_b = (vectors_b / np.sqrt(eigen_b)) @ vectors_b.T
        restore_b = (vectors_b * np.sqrt(eigen_b)) @ vectors_b.T
        u, _, vt = np.linalg.svd(whiten_a @ (a.T @ b / (len(a) - 1)) @ whiten_b,
                                 full_matrices=False)
        matrix = whiten_a @ u @ vt @ restore_b
    if not np.isfinite(matrix).all():
        raise BridgeUnavailable("nonfinite_alignment_matrix")
    _check(check)
    return HistoryLink(source, _snapshot(matrix), ratio)


def fit_history_bridge(days, *, alignment, regularization=None, check):
    """Largest eligible rank, requiring all three source links at that rank.

    All four packs are independently fitted in their native neuron spaces.
    Only an eligibility failure tries the next registered rank. Input/cancellation
    or numerical decomposition failures propagate; none triggers another model,
    source omission, smaller session set, new fit role or relaxed support gate.
    """
    _check(check)
    if alignment not in ALIGNMENTS:
        raise ValueError("unknown registered alignment family")
    if alignment == "regularized_correlation":
        if (isinstance(regularization, bool) or not isinstance(regularization, (int, float))
                or not math.isfinite(regularization) or regularization <= 0):
            raise ValueError("regularized correlation needs an explicit positive ridge")
        regularization = float(regularization)
    elif regularization is not None:
        raise ValueError("unused regularization is not an alignment parameter")
    days = tuple(days)
    if len(days) != 4 or any(not isinstance(day, DayTrials) for day in days):
        raise ValueError("exactly three source days and target calibration are required")
    target = days[-1]
    scope = (target.animal, target.target_day, target.region)
    expected = EARLIER.get(target.animal, ()) + (target.target_day,)
    if (tuple(day.day for day in days) != expected
            or any((day.animal, day.target_day, day.region) != scope
                   or day.stream != "history_assisted" or day.view_role != "fit"
                   for day in days)):
        raise PermissionError("history bridge requires its exact permitted four-day fitting roster")
    arrays = tuple(_fit_arrays(day, check) for day in days)
    rejected = []
    for rank in RANKS:
        _check(check)
        try:
            bases = tuple(_basis(day, data, rank, check) for day, data in zip(days, arrays))
            links = tuple(_matrix(source, bases[-1], alignment, regularization, check)
                          for source in bases[:-1])
        except BridgeUnavailable as error:
            rejected.append((rank, str(error)))
            continue
        return HistoryBridge(bases[-1], links, rank, alignment, regularization, tuple(rejected))
    raise BridgeUnavailable("no_registered_rank_passes_every_day_and_source_link: " + repr(tuple(rejected)))
