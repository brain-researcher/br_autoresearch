"""EP07 electrode-invariant LFP feature choices; no I/O or response access.

Both families preserve each trial and native time bin and retain all nine
guide bands separately. Numeric electrode labels only identify duplicate
columns within a day; they never define a feature or a cross-day match.
Different electrode counts change the empirical distribution, not feature
dimension. No electrode is filtered, imputed, clipped or silently discarded.

``moments_quantiles`` emits the electrode mean, population SD (ddof=0), then
the requested quantiles, per band. Quantiles use the inverse empirical CDF:
the smallest observed value with cumulative mass >= q (q=0 is the minimum).
``robust_histogram`` learns lower/upper quantiles per band and divides that
interval into equal-width bins. It emits normalized counts for underflow,
each interior bin, and overflow; internal edges go to the bin on their right,
and the upper edge belongs to the last interior bin. Repeated edges are
allowed: for a collapsed interval, equality occupies the last interior bin,
with lower/upper values still represented by the two tails.

Fitted quantiles give each permitted day equal total weight, then each
trial/bin/electrode within that day equal weight. Optional output scaling
uses each day's trial/bin feature values with equal day weight: subtract the
median and divide by the IQR. A zero IQR uses scale 1, retaining the feature
and any future departure. These feature-fit weights are separate from the
searchable source-day weights of a predictive model. All arithmetic is
float64; nonfinite inputs or derived features fail without dropping columns.

The ordinary full-pack API requires all three earlier days plus target
calibration for history, and calibration alone for today-only. It accepts
only the existing adapter's fit/development feature roles. primary_control's
explicit private context permits its exact zero/two-source fitting roster;
feature-fit day weights then apply equally to the actual retained days.
An explicit held-out transform reuses frozen parameters and preserves held-out
tags; it grants no opening authority. Inner-fold interfaces remain separate.
It never reads DayTrials.spikes or unit_guide. Role tags are procedural access
controls, not proof of data provenance or an authorization to open a payload.
These are configurable primitives in the broad adaptive grammar, not a final
model shortlist. The trainable set encoder is a separate implementation.
"""
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from native_trials import BANDS, DayTrials, EARLIER, NEW, _directions
from primary_fit_scope import bind_fit_days


@dataclass(frozen=True)
class FeatureRecipe:
    family: str = "moments_quantiles"
    quantiles: tuple[float, ...] = (.1, .25, .5, .75, .9)
    histogram_bins: int = 8
    histogram_quantiles: tuple[float, float] = (.01, .99)
    scaling: str = "none"


@dataclass(frozen=True)
class FeatureAxis:
    band: tuple[float, float]
    statistic: str


@dataclass(frozen=True)
class FittedLFPFeatures:
    recipe: FeatureRecipe
    animal: str
    target_day: str
    region: str
    stream: str
    fit_trial_ids: tuple[tuple[str, tuple[str, ...]], ...]
    bin_offsets: np.ndarray
    axes: tuple[FeatureAxis, ...]
    histogram_edges: tuple[np.ndarray, ...] | None
    center: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class TrialFeatures:
    animal: str
    day: str
    target_day: str
    region: str
    stream: str
    view_role: str
    role: str
    trial_ids: tuple[str, ...]
    directions: np.ndarray
    bin_offsets: np.ndarray
    axes: tuple[FeatureAxis, ...]
    values: np.ndarray


def _readonly(value, dtype=None):
    value = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _recipe(value):
    if not isinstance(value, FeatureRecipe) or value.family not in (
            "moments_quantiles", "robust_histogram") or value.scaling not in ("none", "robust"):
        raise ValueError("unknown LFP feature family or output scaling")
    for values in (value.quantiles, value.histogram_quantiles):
        if (not isinstance(values, tuple) or not values
                or any(isinstance(q, bool) or not isinstance(q, Real)
                       or not np.isfinite(q) or not 0 <= q <= 1 for q in values)
                or any(a >= b for a, b in zip(values, values[1:]))):
            raise ValueError("quantiles must be an increasing tuple in [0, 1]")
    if (len(value.histogram_quantiles) != 2
            or isinstance(value.histogram_bins, bool)
            or not isinstance(value.histogram_bins, Integral) or value.histogram_bins < 1):
        raise ValueError("histogram needs two bound quantiles and a positive integer bin count")
    return value


def _metadata(day, *, _phase="candidate"):
    if (not isinstance(day, DayTrials) or day.animal not in NEW
            or day.target_day not in NEW[day.animal] or day.region not in ("M1", "PMd")
            or day.stream not in ("history_assisted", "today_only")):
        raise ValueError("features require an adapter DayTrials in one permitted scope")
    if _phase not in ("candidate", "held_out"):
        raise ValueError("unknown feature transformation phase")
    if _phase == "held_out":
        if day.view_role != "held_out_features" or day.role != "held_out":
            raise PermissionError("final transformation requires explicit held-out features")
        if day.spikes is not None or day.unit_guide is not None:
            raise PermissionError("held-out feature transforms cannot receive neural targets")
        expected, permitted = "held_out", (day.target_day,)
        _directions(day.directions, len(day.trial_ids), "held_out")
    elif day.view_role == "fit":
        expected = "calibration" if day.day == day.target_day else "training"
        permitted = ((EARLIER[day.animal] + (day.target_day,))
                     if day.stream == "history_assisted" else (day.target_day,))
    elif day.view_role == "development_features":
        expected, permitted = "development", (day.target_day,)
    else:
        raise PermissionError("this feature interface accepts fit or development features only")
    if day.day not in permitted or day.role != expected:
        raise PermissionError("day or trial role is not permitted for this model stream")
    return day.animal, day.target_day, day.region, day.stream


def _data(day):
    """Validate every LFP column; deliberately never request spike fields."""
    values, guide = np.asarray(day.lfp), np.asarray(day.lfp_guide)
    directions, offsets = np.asarray(day.directions), np.asarray(day.bin_offsets)
    if (values.dtype.kind not in "fiu" or values.ndim != 3 or 0 in values.shape
            or guide.dtype.kind not in "fiu" or guide.shape != (values.shape[2], 3)
            or not np.all(np.isfinite(values)) or not np.all(np.isfinite(guide))):
        raise ValueError("finite trial-by-native-bin-by-column LFP and complete guide are required")
    if (not isinstance(day.trial_ids, tuple) or len(day.trial_ids) != values.shape[0]
            or any(not isinstance(key, str) or not key for key in day.trial_ids)
            or len(set(day.trial_ids)) != len(day.trial_ids)
            or directions.dtype.kind not in "iu" or directions.shape != (values.shape[0],)
            or np.any((directions < 0) | (directions >= 8))
            or offsets.dtype.kind not in "iu"
            or not np.array_equal(offsets, np.arange(5, 15) if day.region == "M1" else np.arange(-15, 0))):
        raise ValueError("trial/direction identities and the unchanged native bin axis are required")
    if values.shape[1] != len(offsets):
        raise ValueError("LFP bin axis does not match its native offsets")
    bands, covered = [], np.zeros(values.shape[2], dtype=bool)
    for band in BANDS:
        columns = np.flatnonzero(np.all(guide[:, 1:] == band, axis=1))
        if not len(columns) or len(np.unique(guide[columns, 0])) != len(columns):
            raise ValueError("each of the nine guide bands needs distinct electrode columns")
        covered[columns] = True
        # Sorting values, not labels, makes numerical reductions independent of
        # provider column order. No guide identity enters a model coordinate.
        with np.errstate(over="ignore", invalid="ignore"):
            block = np.sort(values[:, :, columns].astype(np.float64), axis=2)
        if not np.all(np.isfinite(block)):
            raise ValueError("LFP values are not finite in the declared float64 arithmetic")
        bands.append(block)
    if not np.all(covered):
        raise ValueError("unknown LFP guide band; no column may be silently discarded")
    return tuple(bands)


def _weighted_quantiles(parts, quantiles):
    """Equal-day empirical CDF; merging exact ties avoids duplicate-count drift."""
    values, weights = [], []
    for part in parts:
        unique, counts = np.unique(part, return_counts=True)
        values.append(unique)
        weights.append(counts.astype(np.float64) / part.size / len(parts))
    values, weights = np.concatenate(values), np.concatenate(weights)
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    values, weights = values[starts], np.add.reduceat(weights, starts)
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return values[np.minimum(np.searchsorted(cumulative, quantiles, side="left"), len(values) - 1)]


def _axes(recipe):
    names = (("mean", "population_sd") + tuple(f"quantile_{float(q)!r}" for q in recipe.quantiles)
             if recipe.family == "moments_quantiles" else
             ("underflow",) + tuple(f"histogram_bin_{i}" for i in range(recipe.histogram_bins))
             + ("overflow",))
    return tuple(FeatureAxis(tuple(band), name) for band in BANDS for name in names)


def _raw(bands, recipe, edges, check):
    result = []
    for index, values in enumerate(bands):
        check()
        if recipe.family == "moments_quantiles":
            positions = np.maximum(0, np.ceil(np.asarray(recipe.quantiles) * values.shape[2]).astype(int) - 1)
            with np.errstate(over="ignore", invalid="ignore"):
                result.append(np.concatenate((np.mean(values, axis=2)[:, :, None],
                    np.std(values, axis=2, ddof=0)[:, :, None], values[:, :, positions]), axis=2))
        else:
            boundary = edges[index]
            bins = np.searchsorted(boundary, values, side="right")
            bins[values == boundary[-1]] = recipe.histogram_bins
            result.append(np.stack([np.mean(bins == position, axis=2)
                for position in range(recipe.histogram_bins + 2)], axis=2))
    output = np.concatenate(result, axis=2)
    if not np.all(np.isfinite(output)):
        raise ValueError("nonfinite derived LFP feature; no feature was dropped")
    return output


def fit_lfp_features(fit_days, *, recipe=FeatureRecipe(), check, _control=None):
    """Fit one stream/target's transform; authenticate all roles before LFP access."""
    check()
    recipe = _recipe(recipe)
    days = bind_fit_days(fit_days, _control=_control)
    scopes = tuple(_metadata(day) for day in days)
    if any(scope != scopes[0] for scope in scopes) or any(day.view_role != "fit" for day in days):
        raise PermissionError("feature fitting cannot mix scopes or use development features")
    animal, target, region, stream = scopes[0]
    bands = []
    for day in days:
        check()
        bands.append(_data(day))
    edges = None
    if recipe.family == "robust_histogram":
        learned = []
        for band in range(len(BANDS)):
            check()
            lower, upper = _weighted_quantiles([values[band] for values in bands], recipe.histogram_quantiles)
            with np.errstate(over="ignore", invalid="ignore"):
                boundary = np.linspace(lower, upper, recipe.histogram_bins + 1)
            if not np.all(np.isfinite(boundary)):
                raise ValueError("nonfinite learned histogram edges")
            learned.append(_readonly(boundary))
        edges = tuple(learned)
    axes = _axes(recipe)
    center, scale = np.zeros(len(axes)), np.ones(len(axes))
    if recipe.scaling == "robust":
        raw = [_raw(values, recipe, edges, check) for values in bands]
        for column in range(len(axes)):
            check()
            q25, center[column], q75 = _weighted_quantiles(
                [value[:, :, column] for value in raw], (.25, .5, .75))
            with np.errstate(over="ignore", invalid="ignore"):
                width = q75 - q25
            if not np.isfinite(width):
                raise ValueError("nonfinite fitted feature IQR")
            scale[column] = width if width > 0 else 1.
    return FittedLFPFeatures(recipe, animal, target, region, stream,
        tuple((day.day, day.trial_ids) for day in days), _readonly(days[0].bin_offsets, np.int64),
        axes, edges, _readonly(center), _readonly(scale))


def transform_lfp_features(fitted, day, *, check):
    """Apply frozen features; this function cannot update bounds or scaling."""
    return _transform_lfp_features(fitted, day, check=check)


def transform_lfp_held_out(fitted, day, *, check):
    """Explicit frozen transform for held-out features; never fit from them."""
    return _transform_lfp_features(fitted, day, check=check, _phase="held_out")


def _transform_lfp_features(fitted, day, *, check, _phase="candidate"):
    check()
    if not isinstance(fitted, FittedLFPFeatures):
        raise TypeError("a fitted LFP transform is required")
    if _metadata(day, _phase=_phase) != (fitted.animal, fitted.target_day, fitted.region, fitted.stream):
        raise PermissionError("a fitted transform cannot cross animal, target, region or model stream")
    if day.view_role in ("development_features", "held_out_features"):
        calibration_ids = dict(fitted.fit_trial_ids)[fitted.target_day]
        if set(day.trial_ids).intersection(calibration_ids):
            raise PermissionError("prediction trials overlap fitted target calibration trials")
    elif dict(fitted.fit_trial_ids).get(day.day) != day.trial_ids:
        raise PermissionError("fitting day/trial identities were not in this frozen transform")
    bands = _data(day)
    if not np.array_equal(day.bin_offsets, fitted.bin_offsets):
        raise ValueError("native bin identities changed")
    raw = _raw(bands, fitted.recipe, fitted.histogram_edges, check)
    with np.errstate(over="ignore", invalid="ignore"):
        values = (raw - fitted.center) / fitted.scale
    if not np.all(np.isfinite(values)):
        raise ValueError("nonfinite scaled feature; no trial/bin/column was removed")
    check()
    return TrialFeatures(day.animal, day.day, day.target_day, day.region, day.stream,
        day.view_role, day.role, day.trial_ids, _readonly(day.directions, np.int64),
        _readonly(day.bin_offsets, np.int64), fitted.axes, _readonly(values))
