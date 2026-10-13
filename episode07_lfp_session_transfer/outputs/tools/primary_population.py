"""Primary EP07 candidate population coordinates, with explicit candidate rank.

This module does not apply the conditional explanation's anchor-condition and
QR support gates. Those remain in population_bridge.py. A primary candidate
uses its specified rank without a source-driven fallback, per-day centered
unscaled PCA on permitted unresidualized counts, and a specified source map.
Numerical population rank must support the requested dimension in every day.
Today-only fits its own calibration basis and accepts no history-fitted state.

Each source's native neuron axis is independent. Source coordinates are mapped
into the target's coordinates before pooling; all predictions are reconstructed
on the target's complete native neuron axis. 'none' means identity in separately
fitted coordinates, not matching neuron or electrode identities. Parameters
are fitted before primary response residualization. No I/O, score, search,
automatic retry, rank selection or final-evaluation opening occurs here.
"""
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from native_trials import DayTrials, EARLIER, _snapshot, _validate_day
from primary_fit_scope import bind_fit_days


class PrimaryPopulationUnavailable(ValueError):
    """The explicit candidate rank cannot be fitted; do not omit a day."""


@dataclass(frozen=True)
class PrimaryBasis:
    animal: str
    day: str
    target_day: str
    region: str
    stream: str
    role: str
    trial_ids: tuple
    bin_offsets: np.ndarray
    unit_guide: np.ndarray
    mean: np.ndarray
    components: np.ndarray
    centered_anchors: np.ndarray


@dataclass(frozen=True)
class PrimaryPopulation:
    animal: str
    target_day: str
    region: str
    stream: str
    rank: int
    alignment: str
    regularization: float | None
    bases: tuple[PrimaryBasis, ...]
    source_maps: tuple[np.ndarray, ...]


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned resource check required")
    check()


def _scope(day):
    return day.animal, day.target_day, day.region, day.stream


def _fit_basis(day, rank, check):
    directions, _ = _validate_day(day, check)
    if day.view_role != "fit":
        raise PermissionError("population fitting requires training/calibration responses")
    y = np.asarray(day.spikes, dtype=np.float64)
    rows = y.reshape(-1, y.shape[-1])
    mean = np.mean(rows, axis=0)
    centered = rows - mean
    _check(check)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    tolerance = (np.finfo(np.float64).eps * max(centered.shape) * singular[0]
                 if len(singular) else 0.0)
    if np.count_nonzero(singular > tolerance) < rank:
        raise PrimaryPopulationUnavailable("native_fitting_rank_below_explicit_candidate_rank")
    components = vt[:rank].T.copy()
    for column in range(rank):
        pivot = int(np.argmax(np.abs(components[:, column])))
        if components[pivot, column] < 0:
            components[:, column] *= -1
    anchors = np.stack([np.mean(y[directions == direction], axis=0)
                        for direction in range(8)]).reshape(-1, y.shape[-1])
    anchors = (anchors - mean) @ components
    anchors -= np.mean(anchors, axis=0)
    if not all(np.isfinite(value).all() for value in (mean, components, anchors)):
        raise FloatingPointError("nonfinite population fitting state")
    _check(check)
    return PrimaryBasis(day.animal, day.day, day.target_day, day.region,
        day.stream, day.role, day.trial_ids, _snapshot(day.bin_offsets),
        _snapshot(day.unit_guide), _snapshot(mean), _snapshot(components),
        _snapshot(anchors))


def _map(source, target, alignment, regularization, check):
    a, b = source.centered_anchors, target.centered_anchors
    if a.shape != b.shape or not np.array_equal(source.bin_offsets, target.bin_offsets):
        raise ValueError("source and target must have matching direction/native-bin cells")
    _check(check)
    if alignment == "none":
        result = np.eye(a.shape[1])
    elif alignment == "orthogonal":
        u, _, vt = np.linalg.svd(a.T @ b, full_matrices=False)
        result = u @ vt
    else:
        covariance_a = a.T @ a / (len(a) - 1) + regularization * np.eye(a.shape[1])
        covariance_b = b.T @ b / (len(b) - 1) + regularization * np.eye(b.shape[1])
        ea, va = np.linalg.eigh(covariance_a)
        eb, vb = np.linalg.eigh(covariance_b)
        if min(ea[0], eb[0]) <= 0:
            raise FloatingPointError("nonpositive regularized anchor covariance")
        wa = (va / np.sqrt(ea)) @ va.T
        wb = (vb / np.sqrt(eb)) @ vb.T
        restore_b = (vb * np.sqrt(eb)) @ vb.T
        u, _, vt = np.linalg.svd(wa @ (a.T @ b / (len(a) - 1)) @ wb,
                                full_matrices=False)
        result = wa @ u @ vt @ restore_b
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite primary source map")
    _check(check)
    return _snapshot(result)


def fit_primary_population(days, *, rank, alignment, regularization=None, check, _control=None):
    """Fit one full-roster primary candidate; no implicit rank/model search.

    For today-only, alignment must be 'none': it has no earlier-day link. The
    rank is its own explicit candidate parameter, never copied from a fitted
    history bridge. Primary model streams may select different global recipes
    under equal development opportunities.
    """
    _check(check)
    if (isinstance(rank, (bool, np.bool_)) or not isinstance(rank, Integral)
            or rank not in (4, 8, 12, 16)):
        raise ValueError("rank must be an explicit 4, 8, 12 or 16")
    rank = int(rank)
    if alignment not in ("none", "orthogonal", "regularized_correlation"):
        raise ValueError("unknown primary alignment family")
    if alignment == "regularized_correlation":
        if (isinstance(regularization, (bool, np.bool_))
                or not isinstance(regularization, Real)
                or not np.isfinite(regularization) or regularization <= 0):
            raise ValueError("regularized correlation needs an explicit positive ridge")
        regularization = float(regularization)
    elif regularization is not None:
        raise ValueError("regularization is unused by this alignment")
    days = tuple(days)
    # Preserve the ordinary public population fitter's ordered-roster refusal;
    # the shared top-level assembly canonicalizes before calling this helper.
    if _control is None and days and all(isinstance(day, DayTrials) for day in days):
        last = days[-1]
        expected = ((EARLIER.get(last.animal, ()) + (last.target_day,))
                    if last.stream == "history_assisted" else (last.target_day,))
        if tuple(day.day for day in days) != expected:
            raise PermissionError("population fit requires the exact permitted stream roster")
    days = bind_fit_days(days, _control=_control)
    target = days[-1]
    if target.stream == "today_only" and alignment != "none":
        raise ValueError("today-only has no source-to-target alignment")
    # A remove-history control retains its history recipe's declared alignment
    # but has no source bases/maps. Its target basis is fitted anew here.
    bases = tuple(_fit_basis(day, rank, check) for day in days)
    maps = tuple(_map(source, bases[-1], alignment, regularization, check)
                 for source in bases[:-1])
    return PrimaryPopulation(*_scope(target), rank, alignment, regularization,
                             bases, maps)


def project_primary_responses(fitted, day, values, *, residualized, check):
    """Project permitted fitting responses on their own native population axis.

    `values` are either native counts or native day-cell residuals from the
    primary design layer, with the exact complete fitting trial/bin order.
    The role-bound caller supplies this explicit representation flag; this
    numerical module cannot authenticate arbitrary arrays as those responses.
    """
    _check(check)
    if (not isinstance(fitted, PrimaryPopulation) or not isinstance(day, DayTrials)
            or day.view_role != "fit" or _scope(day) != (
                fitted.animal, fitted.target_day, fitted.region, fitted.stream)):
        raise PermissionError("response projection is bound to the fitted stream")
    if not isinstance(residualized, bool):
        raise TypeError("response representation must be explicit")
    matches = [i for i, basis in enumerate(fitted.bases) if basis.day == day.day]
    if len(matches) != 1:
        raise PermissionError("day was not in this population fit")
    index = matches[0]
    basis = fitted.bases[index]
    if (day.role != basis.role or day.trial_ids != basis.trial_ids
            or not np.array_equal(day.bin_offsets, basis.bin_offsets)
            or not np.array_equal(day.unit_guide, basis.unit_guide)):
        raise PermissionError("fitting trial or native neuron axes changed")
    value = np.asarray(values)
    shape = (len(basis.trial_ids), len(basis.bin_offsets), len(basis.unit_guide))
    if value.dtype.kind not in "fiu" or value.shape != shape or not np.isfinite(value).all():
        raise ValueError("finite complete native fitting response tensor required")
    projected = ((value if residualized else value - basis.mean) @ basis.components)
    if index < len(fitted.source_maps):
        projected = projected @ fitted.source_maps[index]
    if not np.isfinite(projected).all():
        raise FloatingPointError("nonfinite population projection")
    _check(check)
    return _snapshot(projected)


def reconstruct_primary_responses(fitted, coordinates, *, residualized, check):
    """Restore target-native axes; primary design restores cell means later.

    Explicit primary responses restore the PCA center. Residual responses use
    zero origin here; their target calibration cell means belong to the primary
    design layer. No clipping or transformation of predicted counts is applied.
    """
    _check(check)
    if not isinstance(fitted, PrimaryPopulation) or not isinstance(residualized, bool):
        raise TypeError("fitted population and explicit representation required")
    target = fitted.bases[-1]
    coordinates = np.asarray(coordinates)
    if (coordinates.dtype.kind not in "fiu" or coordinates.ndim != 3
            or coordinates.shape[0] == 0
            or coordinates.shape[1:] != (len(target.bin_offsets), fitted.rank)
            or not np.isfinite(coordinates).all()):
        raise ValueError("finite trial-by-native-bin-by-rank prediction required")
    result = coordinates @ target.components.T
    if not residualized:
        result += target.mean
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite native prediction")
    _check(check)
    return _snapshot(result)
