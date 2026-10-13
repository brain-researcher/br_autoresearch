"""Dormant fitting-only source-day weighting candidates for EP07.

The full history roster is three earlier training days plus target calibration,
all in one animal/region. These functions neither fit a predictor nor select a
recipe, open data, score development outcomes, or alter the starting-model run.
DayTrials and fitted-feature roles are procedural bindings, not authentication
of arbitrary constructed objects. The eventual caller must bind returned day
masses to this exact fitting scope before using them in a candidate recipe.

Equal weighting assigns each retained source 1/n_sources of source_total_mass.
The ordinary interface requires three; primary_control's explicit omission
context permits exactly the other two. Zero-source controls bypass this fitter.
Reliability sorts
the integer index in each source trial ID, alternates whole trials within each
direction, and forms two native-neuron mean trajectories over all 8*B cells.
After centering each neuron across cells, r = clip(2<A,B>/(||A||²+||B||²),0,1).
The sums pool cells and native neurons, not per-neuron correlations. Constant
anchors give r=0. This is split-half mean-trajectory reproducibility, without a
Spearman--Brown correction; it is not a residual-coupling measure. Native axes
remain independent between days. Reliability weights are (weight_floor+r)^power.

Hierarchical shrinkage is explicitly a precision-weighted heuristic:
 r_shrunk = (n_eff*r + kappa*mean_source_r)/(n_eff+kappa),
 n_eff = 64 / sum_direction(1/n_direction).
The common mean gives the retained sources equal weight. For balanced directions,
n_eff equals the number of whole trials. This is not a posterior or a learned
hyperparameter model. The same positive-floor reliability weighting follows.

Similarity uses one history-bound FittedLFPFeatures transform, without fitting
it again. Direction/native-bin prototypes have equal cell and feature weight.
Target scale is its calibration feature population SD over trial/bin rows,
maximized with the explicit feature_scale_floor. For squared standardized
prototype distance D, source weight is weight_floor + exp(-D/temperature).
All features/cells remain present, including constant target features.

Only source relative weights sum to one. source_total_mass and target_mass are
separate explicit candidate parameters; values 3 and 1 with equal weights
recover four equal day masses. They are neither feature-fit weights nor the
animal-balanced scientific scoring weights. Every source must retain positive
finite mass; numerically collapsed weights fail instead of dropping a day.
Leave-one-day stacking requires a separate inner-fold interface and is absent.
"""
from dataclasses import dataclass
from numbers import Real

import numpy as np

from lfp_features import FittedLFPFeatures, transform_lfp_features
from native_trials import (DayTrials, EARLIER, NEW, _directions, _guide,
                           _offsets, _snapshot, _trial_ids)
from primary_fit_scope import bind_fit_days


@dataclass(frozen=True)
class SourceWeightRecipe:
    family: str
    target_mass: float
    source_total_mass: float
    weight_floor: float | None = None
    reliability_power: float | None = None
    similarity_temperature: float | None = None
    feature_scale_floor: float | None = None
    shrinkage_strength: float | None = None


@dataclass(frozen=True)
class SourceWeights:
    recipe: SourceWeightRecipe
    animal: str
    target_day: str
    region: str
    stream: str
    fit_trial_ids: tuple[tuple[str, tuple[str, ...]], ...]
    bin_offsets: np.ndarray
    source_days: tuple[str, ...]
    relative_weights: np.ndarray
    source_masses: tuple[float, ...]
    target_mass: float
    reliability: np.ndarray | None
    shrunk_reliability: np.ndarray | None
    effective_trials: np.ndarray | None
    prototype_distances: np.ndarray | None
    feature_scale: np.ndarray | None


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned cancellation/resource check required")
    check()


def _positive(value, name, *, zero=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0 or (value == 0 and not zero)):
        raise ValueError(f"{name} must be an explicit finite {'nonnegative' if zero else 'positive'} scalar")


def _recipe(recipe):
    if not isinstance(recipe, SourceWeightRecipe):
        raise TypeError("an explicit source-weight recipe is required")
    if recipe.family == "leave_one_day_stacking":
        raise NotImplementedError("stacking requires a separate permitted inner-fold interface")
    fields = {
        "equal": (),
        "reliability": ("weight_floor", "reliability_power"),
        "calibration_similarity": ("weight_floor", "similarity_temperature", "feature_scale_floor"),
        "hierarchical_shrinkage": ("weight_floor", "reliability_power", "shrinkage_strength"),
    }
    if recipe.family not in fields:
        raise ValueError("unknown source-weight family")
    _positive(recipe.target_mass, "target_mass")
    _positive(recipe.source_total_mass, "source_total_mass")
    for name in ("weight_floor", "reliability_power", "similarity_temperature",
                 "feature_scale_floor", "shrinkage_strength"):
        value = getattr(recipe, name)
        if name in fields[recipe.family]:
            _positive(value, name, zero=name == "shrinkage_strength")
        elif value is not None:
            raise ValueError("unused source-weight parameters must be None")
    return recipe


def _roster(fit_days, *, _control=None):
    """Check every role and scope before interpreting any supplied array."""
    days = bind_fit_days(fit_days, _control=_control)
    if days[-1].stream != "history_assisted":
        raise PermissionError("source weighting accepts one history-only fitting scope")
    if len(days) == 1:
        raise ValueError("zero-source controls must bypass source-weight fitting")
    return days


def _identities(day):
    _trial_ids(day.trial_ids)
    directions, counts = _directions(day.directions, len(day.trial_ids), day.role)
    offsets = np.asarray(day.bin_offsets)
    if offsets.dtype.kind not in "iu" or not np.array_equal(offsets, _offsets(day.region)):
        raise ValueError("unchanged native time offsets are required")
    return directions, counts


def _reliability(day, directions, check):
    units = _guide(day.unit_guide, 2, "unit_guide")
    raw = np.asarray(day.spikes)
    if (raw.dtype.kind not in "fiu" or raw.shape !=
            (len(day.trial_ids), len(day.bin_offsets), len(units))
            or not np.isfinite(raw).all() or np.any(raw < 0)
            or np.any(raw != np.floor(raw))):
        raise ValueError("source reliability requires its own complete native count/neuronal axes")
    halves = ([], [])
    for direction in range(8):
        _check(check)
        indices = sorted(np.flatnonzero(directions == direction),
                         key=lambda index: int(day.trial_ids[index][6:]))
        for half in (0, 1):
            with np.errstate(over="ignore", invalid="ignore"):
                block = raw[indices[half::2]].astype(np.float64)
                mean = np.mean(block, axis=0)
            if not np.isfinite(mean).all():
                raise FloatingPointError("nonfinite split-half native mean")
            halves[half].append(mean)
    with np.errstate(over="ignore", invalid="ignore"):
        a, b = (np.concatenate(parts, axis=0) for parts in halves)
        a = a - np.mean(a, axis=0)
        b = b - np.mean(b, axis=0)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError("nonfinite centered split-half anchors")
    # A common scaling preserves the ratio and avoids overflow in products.
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    if scale == 0:
        return 0.0
    a, b = a / scale, b / scale
    denominator = np.sum(a * a) + np.sum(b * b)
    value = 2 * np.sum(a * b) / denominator
    if not np.isfinite(value):
        raise FloatingPointError("nonfinite split-half reproducibility")
    _check(check)
    return float(np.clip(value, 0, 1))


def _prototype(feature, check):
    means = []
    for direction in range(8):
        _check(check)
        with np.errstate(over="ignore", invalid="ignore"):
            means.append(np.mean(feature.values[feature.directions == direction], axis=0))
    result = np.stack(means)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite fitting feature prototype")
    return result


def _similarity(days, feature_model, recipe, check):
    expected_ids = tuple((day.day, day.trial_ids) for day in days)
    first = days[0]
    if (not isinstance(feature_model, FittedLFPFeatures)
            or (feature_model.animal, feature_model.target_day, feature_model.region,
                feature_model.stream) !=
               (first.animal, first.target_day, first.region, "history_assisted")
            or feature_model.fit_trial_ids != expected_ids):
        raise PermissionError("similarity requires the same history fitting scope and exact feature-fit IDs")
    target = transform_lfp_features(feature_model, days[-1], check=check)
    target_mean = _prototype(target, check)
    with np.errstate(over="ignore", invalid="ignore"):
        scale = np.std(target.values.reshape(-1, target.values.shape[-1]), axis=0, ddof=0)
        scale = np.maximum(scale, recipe.feature_scale_floor)
    if not np.isfinite(scale).all() or not np.all(scale > 0):
        raise FloatingPointError("nonfinite target calibration feature scale")
    del target
    distances = []
    for day in days[:-1]:
        _check(check)
        feature = transform_lfp_features(feature_model, day, check=check)
        source_mean = _prototype(feature, check)
        with np.errstate(over="ignore", invalid="ignore"):
            difference = (source_mean - target_mean) / scale
            distance = np.mean(difference * difference)
        if not np.isfinite(distance):
            raise FloatingPointError("nonfinite calibration prototype distance")
        distances.append(float(distance))
        del feature, source_mean, difference
    return np.asarray(distances, dtype=np.float64), scale


def fit_source_weights(fit_days, *, recipe, feature_model=None, check, _control=None):
    """Derive one history candidate's weights without fitting/scoring a model.

    Equal uses metadata only. Reliability/shrinkage read only earlier training
    counts, never target calibration counts. Similarity reads only fitting LFP
    through the supplied frozen history feature model, never any count array.
    """
    _check(check)
    recipe = _recipe(recipe)
    days = _roster(fit_days, _control=_control)
    if recipe.family != "calibration_similarity" and feature_model is not None:
        raise ValueError("this weighting family does not use fitted feature state")
    identity = []
    for day in days:
        _check(check)
        identity.append(_identities(day))
    reliability = shrunk = effective = distances = scale = None
    if recipe.family == "equal":
        unnormalized = np.ones(len(days) - 1, dtype=np.float64)
    elif recipe.family == "calibration_similarity":
        distances, scale = _similarity(days, feature_model, recipe, check)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            unnormalized = recipe.weight_floor + np.exp(-distances / recipe.similarity_temperature)
    else:
        reliability = np.asarray([_reliability(day, directions, check)
            for day, (directions, _) in zip(days[:-1], identity[:-1])], dtype=np.float64)
        effective = np.asarray([64 / np.sum(1 / counts.astype(np.float64))
                                for _, counts in identity[:-1]])
        values = reliability
        if recipe.family == "hierarchical_shrinkage":
            # Convex form avoids overflowing n_eff*r + kappa*mean(r).
            with np.errstate(over="ignore", invalid="ignore"):
                blend = 1 / (1 + recipe.shrinkage_strength / effective)
                shrunk = blend * reliability + (1 - blend) * np.mean(reliability)
            values = shrunk
        with np.errstate(over="ignore", invalid="ignore"):
            logs = recipe.reliability_power * np.log(recipe.weight_floor + values)
        if not np.isfinite(logs).all():
            raise FloatingPointError("source-weight power is not finite in float64")
        with np.errstate(under="ignore"):
            unnormalized = np.exp(logs - np.max(logs))
    if (not np.isfinite(unnormalized).all() or not np.all(unnormalized > 0)):
        raise FloatingPointError("every retained source must have positive finite weight")
    weights = unnormalized / np.max(unnormalized)
    weights /= np.sum(weights)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        masses = weights * recipe.source_total_mass
    if not np.isfinite(masses).all() or not np.all(masses > 0):
        raise FloatingPointError("every retained source must have positive finite mass")
    _check(check)
    saved = lambda value: None if value is None else _snapshot(value)
    return SourceWeights(recipe, days[0].animal, days[0].target_day, days[0].region,
        "history_assisted", tuple((day.day, day.trial_ids) for day in days),
        _snapshot(days[0].bin_offsets), tuple(day.day for day in days[:-1]),
        _snapshot(weights), tuple(float(value) for value in masses), float(recipe.target_mass),
        saved(reliability), saved(shrunk), saved(effective), saved(distances), saved(scale))
