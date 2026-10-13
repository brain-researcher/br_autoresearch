"""EP07 primary predictor design; this is not the residual-only follow-up.

Three searchable adjustments use independently fitted day-specific means:
``explicit`` keeps features/native counts and appends direction/time cells;
``residualize`` subtracts fitting direction-by-bin means from both features
and native counts; ``both`` additionally appends the explicit cell predictors.
The explicit basis covers all eight directions crossed with every native bin.
It omits direction 0 / first native bin: regression must supply an intercept.
No other indicator is fitted, standardized, selected or dropped.

Each cell mean gives that day's fitting trials in the cell equal weight.
Earlier training days and target calibration remain separate; no neuron axes
are pooled or matched. Feature axes must agree across days, and the caller
must supply features from the same permitted upstream feature transform.
This module does not fit that transform, a bridge, a regression or weights.

Population bases/alignments are learned from UNRESIDUALIZED counts upstream.
FitDesign.responses remain in their own day's native neuron coordinates;
population transport is separate. After transport decodes a prediction into
target native coordinates, restore_primary_prediction adds the target's
calibration cell mean only for residualize/both. It never adds a PCA mean,
clips a predicted count, or casts to integer. Upstream transport owns the PCA
origin: centered counts for explicit, zero-origin residuals for the other two.
The explicit remove-nuisance control alone allows 'none': no fitted nuisance
means, indicators, residualization or cell-mean restoration; population fitting
and decoding still center/restore the unresidualized native counts.

The stricter conditional residual follow-up forbids predictor labels/templates
and does not restore means for residual scoring. No such qualification or
mechanism claim is made here. Ordinary interfaces accept fit/development;
explicit held-out wrappers preserve that phase while applying frozen means.
primary_control supplies exact source-removal scope. Opening authority and
inner-fold interfaces remain separate.
Role bindings are procedural, not proof of provenance or opening authority.
"""
from dataclasses import dataclass

import numpy as np

from lfp_features import FeatureAxis, TrialFeatures
from native_trials import DayTrials, NativePredictions, EARLIER, NEW, _snapshot, _validate_day
from primary_fit_scope import bind_fit_days, validate_control


ADJUSTMENTS = ("explicit", "residualize", "both")


@dataclass(frozen=True)
class DesignAxis:
    kind: str
    feature: FeatureAxis | None
    direction: int | None
    bin_offset: int | None


@dataclass(frozen=True)
class DayMeans:
    day: str
    role: str
    trial_ids: tuple
    directions: np.ndarray
    unit_guide: np.ndarray
    feature_means: np.ndarray | None
    response_means: np.ndarray | None
    trials_per_direction: np.ndarray


@dataclass(frozen=True)
class PrimaryDesign:
    animal: str
    target_day: str
    region: str
    stream: str
    adjustment: str
    feature_axes: tuple[FeatureAxis, ...]
    axes: tuple[DesignAxis, ...]
    bin_offsets: np.ndarray
    day_means: tuple[DayMeans, ...]


@dataclass(frozen=True)
class DesignMatrix:
    animal: str
    day: str
    target_day: str
    region: str
    stream: str
    view_role: str
    role: str
    trial_ids: tuple
    directions: np.ndarray
    bin_offsets: np.ndarray
    axes: tuple[DesignAxis, ...]
    adjustment: str
    values: np.ndarray

    @property
    def X(self):
        """Regression rows: trial-major, native-bin-minor; no temporal pooling."""
        return self.values.reshape(-1, self.values.shape[-1])


@dataclass(frozen=True)
class FitDesign:
    design: DesignMatrix
    unit_guide: np.ndarray
    responses: np.ndarray

    @property
    def Y(self):
        """Native-day response rows, aligned exactly to design.X."""
        return self.responses.reshape(-1, self.responses.shape[-1])


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned resource check required")
    check()


def _scope(value, *, _phase="candidate"):
    if (value.animal not in NEW or value.target_day not in NEW[value.animal]
            or value.region not in ("M1", "PMd")
            or value.stream not in ("history_assisted", "today_only")):
        raise ValueError("one permitted animal/target/region/stream is required")
    if _phase not in ("candidate", "held_out"):
        raise ValueError("unknown primary design phase")
    expected = ((EARLIER[value.animal] + (value.target_day,))
                if value.stream == "history_assisted" else (value.target_day,))
    if _phase == "held_out":
        if (value.view_role != "held_out_features" or value.role != "held_out"
                or value.day != value.target_day):
            raise PermissionError("final primary design requires target held-out features")
    elif value.view_role == "fit":
        role = "calibration" if value.day == value.target_day else "training"
        if value.day not in expected or value.role != role:
            raise PermissionError("day or role is outside this fitting stream")
    elif value.view_role == "development_features":
        if value.day != value.target_day or value.role != "development":
            raise PermissionError("development features must belong to the target day")
    else:
        raise PermissionError("primary design accepts only fit or development features")
    return value.animal, value.target_day, value.region, value.stream


def _trial_metadata(value):
    ids, directions, offsets = value.trial_ids, np.asarray(value.directions), np.asarray(value.bin_offsets)
    if (not isinstance(ids, tuple) or not ids or len(set(ids)) != len(ids)
            or any(not isinstance(key, str) or not key for key in ids)
            or directions.dtype.kind not in "iu" or directions.shape != (len(ids),)
            or np.any((directions < 0) | (directions >= 8))
            or offsets.dtype.kind not in "iu"
            or not np.array_equal(offsets, np.arange(5, 15) if value.region == "M1" else np.arange(-15, 0))):
        raise ValueError("whole-trial/direction identities and exact native bins are required")
    return directions


def _features(value):
    if (not isinstance(value.axes, tuple) or not value.axes
            or any(not isinstance(axis, FeatureAxis) for axis in value.axes)
            or len(set(value.axes)) != len(value.axes)):
        raise ValueError("unique immutable feature axes are required")
    values = np.asarray(value.values)
    if (values.dtype.kind not in "fiu" or values.shape != (
            len(value.trial_ids), len(value.bin_offsets), len(value.axes))
            or not np.isfinite(values).all()):
        raise ValueError("finite trial-by-native-bin-by-feature values are required")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(values, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("features are nonfinite in the declared float64 arithmetic")
    return result


def _bind_pair(day, features):
    if not isinstance(day, DayTrials) or not isinstance(features, TrialFeatures):
        raise TypeError("fit design requires DayTrials and its TrialFeatures")
    if day.view_role != "fit" or features.view_role != "fit":
        raise PermissionError("primary means and fitting responses use fitting roles only")
    if (_scope(day) != _scope(features) or day.day != features.day or day.role != features.role
            or day.trial_ids != features.trial_ids
            or not np.array_equal(day.directions, features.directions)
            or not np.array_equal(day.bin_offsets, features.bin_offsets)):
        raise ValueError("native fitting responses and features must have identical scope and rows")
    _trial_metadata(features)


def _means(fitted, day):
    return next((value for value in fitted.day_means if value.day == day), None)


def _bind_fitted(fitted, value, *, _phase="candidate"):
    if not isinstance(fitted, PrimaryDesign):
        raise TypeError("an existing primary design is required")
    if _scope(value, _phase=_phase) != (fitted.animal, fitted.target_day, fitted.region, fitted.stream):
        raise PermissionError("primary fitted state cannot cross animal/target/region/stream")
    means = _means(fitted, value.day)
    if means is None:
        raise ValueError("day has no permitted fitted means")
    if value.view_role in ("development_features", "held_out_features"):
        if set(value.trial_ids).intersection(means.trial_ids):
            raise PermissionError("prediction trials overlap fitted target calibration")
    elif (value.trial_ids != means.trial_ids or value.role != means.role
          or not np.array_equal(value.directions, means.directions)):
        raise ValueError("fitting trial identity/order/directions differ from the frozen means")
    directions = _trial_metadata(value)
    if not np.array_equal(value.bin_offsets, fitted.bin_offsets):
        raise ValueError("native bin identities differ from the fitted design")
    return means, directions


def fit_primary_design(fit_days, fit_features, *, adjustment, check, _control=None):
    """Fit all per-day cell means; validate the complete scope before any spikes."""
    _check(check)
    if adjustment not in ADJUSTMENTS and not (adjustment == "none"
            and _control is not None and validate_control(_control).kind == "remove_nuisance"):
        raise ValueError("unknown primary direction/time adjustment")
    days, features = tuple(fit_days), tuple(fit_features)
    if (not days or len(days) != len(features) or any(not isinstance(day, DayTrials) for day in days)
            or any(not isinstance(value, TrialFeatures) for value in features)):
        raise ValueError("one feature pack per permitted fitting day is required")
    if any(value.view_role != "fit" for value in (*days, *features)):
        raise PermissionError("development or held-out data cannot fit primary means")
    scope = _scope(days[0])
    if any(_scope(value) != scope for value in (*days, *features)):
        raise PermissionError("fitting cannot mix animals, targets, regions or streams")
    animal, target, region, stream = scope
    days = bind_fit_days(days, _control=_control)
    expected = tuple(day.day for day in days)
    native, inputs = {day.day: day for day in days}, {value.day: value for value in features}
    if (len(native) != len(days) or len(inputs) != len(features)
            or set(native) != set(expected) or set(inputs) != set(expected)):
        raise ValueError("the complete fixed fitting roster is required")
    feature_axes = inputs[target].axes
    for key in expected:
        _bind_pair(native[key], inputs[key])
        if inputs[key].axes != feature_axes:
            raise ValueError("pooled predictor feature axes must agree across days")
    fitted = []
    for key in expected:
        _check(check)
        day, feature = native[key], inputs[key]
        directions, counts = _validate_day(day, check)
        x = _features(feature)
        y = np.asarray(day.spikes, dtype=np.float64)
        if adjustment == "none":
            fitted.append(DayMeans(key, day.role, day.trial_ids, _snapshot(directions),
                _snapshot(day.unit_guide), None, None, _snapshot(counts)))
            continue
        feature_means, response_means = [], []
        for direction in range(8):
            _check(check)
            with np.errstate(over="ignore", invalid="ignore"):
                feature_means.append(np.mean(x[directions == direction], axis=0))
                response_means.append(np.mean(y[directions == direction], axis=0))
        if not np.isfinite(feature_means).all() or not np.isfinite(response_means).all():
            raise ValueError("nonfinite fitted cell mean; no cell may be dropped or imputed")
        fitted.append(DayMeans(key, day.role, day.trial_ids, _snapshot(directions),
            _snapshot(day.unit_guide), _snapshot(feature_means), _snapshot(response_means), _snapshot(counts)))
    offsets = _snapshot(native[target].bin_offsets)
    axes = tuple(DesignAxis("lfp", axis, None, None) for axis in feature_axes)
    if adjustment in ("explicit", "both"):
        axes += tuple(DesignAxis("direction_time", None, direction, int(offset))
                      for direction in range(8) for bin_index, offset in enumerate(offsets)
                      if direction != 0 or bin_index != 0)
    _check(check)
    return PrimaryDesign(animal, target, region, stream, adjustment, feature_axes, axes, offsets, tuple(fitted))


def transform_primary_features(fitted, features, *, check):
    """Build predictors without receiving native responses or changing fitted means."""
    return _transform_primary_features(fitted, features, check=check)


def transform_primary_held_out_features(fitted, features, *, check):
    """Apply frozen design to held-out TrialFeatures, preserving both role tags."""
    return _transform_primary_features(fitted, features, check=check, _phase="held_out")


def _transform_primary_features(fitted, features, *, check, _phase="candidate"):
    _check(check)
    if not isinstance(features, TrialFeatures):
        raise TypeError("prediction requires a TrialFeatures object, never native targets")
    means, directions = _bind_fitted(fitted, features, _phase=_phase)
    if features.axes != fitted.feature_axes:
        raise ValueError("feature axes differ from the fitted primary design")
    values = _features(features)
    if fitted.adjustment in ("residualize", "both"):
        with np.errstate(over="ignore", invalid="ignore"):
            values -= means.feature_means[directions]
    if fitted.adjustment in ("explicit", "both"):
        bins = len(fitted.bin_offsets)
        cells = directions[:, None] * bins + np.arange(bins)[None, :]
        indicators = (cells[:, :, None] == np.arange(1, 8 * bins)).astype(np.float64)
        values = np.concatenate((values, indicators), axis=2)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite primary predictor design")
    _check(check)
    return DesignMatrix(features.animal, features.day, features.target_day, features.region,
        features.stream, features.view_role, features.role, features.trial_ids,
        _snapshot(directions), _snapshot(features.bin_offsets), fitted.axes,
        fitted.adjustment, _snapshot(values))


def make_primary_fit_design(fitted, day, features, *, check):
    """Provide authorized native-day fitting targets; no cross-day neuron pooling."""
    _check(check)
    _bind_pair(day, features)
    means, directions = _bind_fitted(fitted, features)
    if not np.array_equal(day.unit_guide, means.unit_guide):
        raise ValueError("native fitting neuron order differs from that day's frozen guide")
    _validate_day(day, check)
    design = transform_primary_features(fitted, features, check=check)
    responses = np.array(day.spikes, dtype=np.float64, copy=True)
    if fitted.adjustment in ("residualize", "both"):
        with np.errstate(over="ignore", invalid="ignore"):
            responses -= means.response_means[directions]
    if not np.isfinite(responses).all():
        raise ValueError("nonfinite native fitting responses after primary adjustment")
    _check(check)
    return FitDesign(design, _snapshot(means.unit_guide), _snapshot(responses))


def restore_primary_prediction(fitted, design, predicted_native, *, check):
    """Restore a decoded TARGET prediction to primary native count coordinates.

    The caller has already decoded the declared target unit-guide order, with
    its PCA origin handled separately. No labels/templates enter the mapping
    here; only the primary response cell mean is restored when appropriate.
    """
    return _restore_primary_prediction(fitted, design, predicted_native, check=check)


def restore_primary_held_out_prediction(fitted, design, predicted_native, *, check):
    """Restore the frozen primary target mean for an explicit held-out design."""
    return _restore_primary_prediction(fitted, design, predicted_native, check=check,
                                       _phase="held_out")


def _restore_primary_prediction(fitted, design, predicted_native, *, check, _phase="candidate"):
    _check(check)
    if not isinstance(design, DesignMatrix):
        raise TypeError("a primary design matrix is required")
    means, directions = _bind_fitted(fitted, design, _phase=_phase)
    if (design.day != fitted.target_day or design.axes != fitted.axes
            or design.adjustment != fitted.adjustment):
        raise ValueError("prediction must bind the fitted primary target design")
    prediction = np.asarray(predicted_native)
    if (prediction.dtype.kind not in "fiu" or prediction.shape != (
            len(design.trial_ids), len(fitted.bin_offsets), len(means.unit_guide))
            or not np.isfinite(prediction).all()):
        raise ValueError("finite target-native trial/bin/neuron predictions are required")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.array(prediction, dtype=np.float64, copy=True)
        if fitted.adjustment in ("residualize", "both"):
            values += means.response_means[directions]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite restored primary prediction")
    _check(check)
    return NativePredictions(fitted.animal, fitted.target_day, fitted.region, design.trial_ids,
        _snapshot(directions), _snapshot(fitted.bin_offsets), _snapshot(means.unit_guide),
        _snapshot(values), means.trial_ids)
