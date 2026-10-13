"""Complete EP07 candidate-role tensors and the calibration-only comparator.

The ordinary API uses only CandidateViews' fit and development_features roles.
Explicit held-out wrappers reuse the numerical adapter with a private phase;
they preserve held_out_features/held_out tags and never accept neural targets.
Each
day retains its own neural axes; numeric neuron/electrode IDs across days never
establish correspondence. Native values and whole-trial order are unchanged.
The complete-role minima do not authorize silently truncated inner-fold data;
such folds need their own explicit interface. This module performs no scoring,
serialization, process launch, or direct file access. Access and resource
authorization remain with the caller and the existing procedural reader.
"""
from dataclasses import dataclass
import re

import numpy as np

from view_access import CandidateViews, EARLIER, NEW


BANDS = ((0.0, 0.0), (0.5, 4.0), (4.0, 8.0), (8.0, 12.0),
         (12.0, 25.0), (25.0, 50.0), (50.0, 100.0),
         (100.0, 200.0), (200.0, 400.0))
_MINIMUM = {"training": 15, "calibration": 3, "development": 6, "held_out": 6}
_BASE_KEYS = {"animal", "day", "region", "trial_id", "role", "direction",
              "lfp", "lfp_guide", "bin_offsets"}


@dataclass(frozen=True)
class DayTrials:
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
    lfp: np.ndarray
    lfp_guide: np.ndarray
    spikes: np.ndarray | None
    unit_guide: np.ndarray | None


@dataclass(frozen=True)
class CalibrationMean:
    animal: str
    target_day: str
    region: str
    calibration_trial_ids: tuple
    trials_per_direction: np.ndarray
    bin_offsets: np.ndarray
    unit_guide: np.ndarray
    means: np.ndarray


@dataclass(frozen=True)
class NativePredictions:
    animal: str
    target_day: str
    region: str
    trial_ids: tuple
    directions: np.ndarray
    bin_offsets: np.ndarray
    unit_guide: np.ndarray
    values: np.ndarray
    calibration_trial_ids: tuple


def _check(check):
    if not callable(check):
        raise TypeError("a caller-owned cancellation/resource check is required")
    check()


def _snapshot(value):
    # Immutable bytes back the snapshot: callers cannot re-enable writes on it.
    array = np.asarray(value)
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _scope(view_role, animal, target_day, region, stream, *, _phase="candidate"):
    if _phase not in ("candidate", "held_out"):
        raise ValueError("unknown native adapter phase")
    allowed = ("fit", "development_features") if _phase == "candidate" else ("held_out_features",)
    if view_role not in allowed:
        raise PermissionError("native adapter accepts only candidate fit or development features")
    CandidateViews._check_scope(animal, target_day, region)
    if stream not in ("history_assisted", "today_only"):
        raise ValueError("unknown candidate stream")
    if view_role == "fit" and stream == "history_assisted":
        return EARLIER[animal] + (target_day,)
    return (target_day,)


def _offsets(region):
    return np.arange(5, 15, dtype=np.int64) if region == "M1" else np.arange(-15, 0, dtype=np.int64)


def _role(view_role, day, target_day):
    if view_role == "held_out_features":
        return "held_out"
    if view_role == "development_features":
        return "development"
    return "calibration" if day == target_day else "training"


def _numeric(value, ndim, name):
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite real native data with {ndim} axes")
    return array


def _guide(value, width, name):
    array = _numeric(value, 2, name)
    if array.shape[1] != width or array.shape[0] == 0:
        raise ValueError(f"invalid {name} shape")
    if len({tuple(row) for row in array.tolist()}) != len(array):
        raise ValueError(f"duplicate {name} identities")
    if name == "lfp_guide" and {tuple(row) for row in array[:, 1:].tolist()} != set(BANDS):
        raise ValueError("LFP guide must retain all nine declared band classes")
    return array


def _trial_ids(ids):
    if not isinstance(ids, tuple) or not ids or len(set(ids)) != len(ids):
        raise ValueError("whole-trial identities must be nonempty and unique within day")
    if any(not isinstance(x, str) or re.fullmatch(r"trial_[0-9]{6,}", x) is None
           or int(x[6:]) < 1 for x in ids):
        raise ValueError("invalid source whole-trial identity")


def _directions(value, trial_count, role):
    array = np.asarray(value)
    if array.shape != (trial_count,) or array.dtype.kind not in "iu" or np.any((array < 0) | (array > 7)):
        raise ValueError("one integer reach direction in 0..7 is required per trial")
    counts = np.bincount(array.astype(np.int64), minlength=8)
    if np.any(counts < _MINIMUM[role]):
        raise ValueError("complete role pack lacks required trials in one or more directions")
    return array, counts


def _validate_day(day, check, *, _phase="candidate"):
    _check(check)
    if not isinstance(day, DayTrials):
        raise TypeError("expected a day-specific native trial pack")
    allowed = _scope(day.view_role, day.animal, day.target_day, day.region, day.stream, _phase=_phase)
    if day.day not in allowed or day.role != _role(day.view_role, day.day, day.target_day):
        raise ValueError("day or role is outside the declared candidate scope")
    _trial_ids(day.trial_ids)
    directions, counts = _directions(day.directions, len(day.trial_ids), day.role)
    offsets = np.asarray(day.bin_offsets)
    if offsets.dtype.kind not in "iu" or not np.array_equal(offsets, _offsets(day.region)):
        raise ValueError("native time offsets differ from the fixed region window")
    guide = _guide(day.lfp_guide, 3, "lfp_guide")
    if np.asarray(day.lfp).shape != (len(day.trial_ids), len(offsets), len(guide)):
        raise ValueError("LFP tensor axes disagree with whole trials, native time or guide")
    for row in day.lfp:
        _check(check)
        _numeric(row, 2, "lfp")
    if day.view_role in ("development_features", "held_out_features"):
        if day.spikes is not None or day.unit_guide is not None:
            raise PermissionError("feature-only trial packs cannot contain neural targets")
    else:
        units = _guide(day.unit_guide, 2, "unit_guide")
        if np.asarray(day.spikes).shape != (len(day.trial_ids), len(offsets), len(units)):
            raise ValueError("spike tensor axes disagree with its own day-specific neuron guide")
        for row in day.spikes:
            _check(check)
            values = _numeric(row, 2, "spikes")
            if np.any(values < 0) or np.any(values != np.floor(values)):
                raise ValueError("spikes must retain nonnegative native integer counts")
    _check(check)
    return directions, counts


def adapt_records(records, *, view_role, animal, target_day, region, stream, check):
    """Consume complete allowed records into separate immutable day tensors.

    This is an adapter of the existing reader contract, not authentication of
    arbitrary dictionaries. It refuses missing expected days and malformed
    roles/axes without dropping observations or matching any cross-day units.
    """
    return _adapt_records(records, view_role=view_role, animal=animal, target_day=target_day,
                          region=region, stream=stream, check=check)


def _adapt_records(records, *, view_role, animal, target_day, region, stream, check,
                   _phase="candidate"):
    _check(check)
    days = _scope(view_role, animal, target_day, region, stream, _phase=_phase)
    expected_keys = _BASE_KEYS | ({"spikes", "unit_guide"} if view_role == "fit" else set())
    grouped = {day: [] for day in days}
    identities = {day: set() for day in days}
    iterator = iter(records)
    try:
        while True:
            _check(check)
            try:
                record = next(iterator)
            except StopIteration:
                break
            # Bind metadata before interpreting any supplied neural array.
            if set(record) != expected_keys:
                raise ValueError("record fields differ from the permitted candidate view")
            day = record["day"]
            if (record["animal"] != animal or record["region"] != region or day not in grouped
                    or record["role"] != _role(view_role, day, target_day)):
                raise ValueError("mixed or unauthorized animal/day/region/role")
            trial_id = record["trial_id"]
            _trial_ids((trial_id,))
            if trial_id in identities[day]:
                raise ValueError("duplicate whole trial within a day")
            direction = record["direction"]
            if isinstance(direction, (bool, np.bool_)) or not isinstance(direction, (int, np.integer)) or not 0 <= direction < 8:
                raise ValueError("invalid reach direction")
            offsets = np.asarray(record["bin_offsets"])
            if offsets.dtype.kind not in "iu" or not np.array_equal(offsets, _offsets(region)):
                raise ValueError("native time offsets differ from the fixed region window")
            guide = _guide(record["lfp_guide"], 3, "lfp_guide")
            lfp = _numeric(record["lfp"], 2, "lfp")
            if lfp.shape != (len(offsets), len(guide)):
                raise ValueError("LFP native axes differ from its guide")
            units = spikes = None
            if view_role == "fit":
                units = _guide(record["unit_guide"], 2, "unit_guide")
                spikes = _numeric(record["spikes"], 2, "spikes")
                if spikes.shape != (len(offsets), len(units)):
                    raise ValueError("spike native axes differ from its day-specific guide")
                if np.any(spikes < 0) or np.any(spikes != np.floor(spikes)):
                    raise ValueError("spikes must retain nonnegative native integer counts")
            if grouped[day]:
                first = grouped[day][0]
                if (guide.dtype != first["lfp_guide"].dtype or not np.array_equal(guide, first["lfp_guide"])
                        or lfp.dtype != first["lfp"].dtype):
                    raise ValueError("LFP axis or native dtype changed within one day")
                if view_role == "fit" and (units.dtype != first["unit_guide"].dtype
                        or not np.array_equal(units, first["unit_guide"])
                        or spikes.dtype != first["spikes"].dtype):
                    raise ValueError("neuron axis or native count dtype changed within one day")
            _check(check)
            grouped[day].append({"trial_id": trial_id, "direction": int(direction),
                                 "lfp": _snapshot(lfp), "lfp_guide": _snapshot(guide),
                                 "spikes": None if spikes is None else _snapshot(spikes),
                                 "unit_guide": None if units is None else _snapshot(units)})
            identities[day].add(trial_id)
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    result = []
    for day in days:
        _check(check)
        rows = grouped.pop(day)
        if not rows:
            raise ValueError("complete candidate view is missing a required day")
        first = rows[0]
        pack = DayTrials(animal, day, target_day, region, stream, view_role,
                         _role(view_role, day, target_day),
                         tuple(row["trial_id"] for row in rows),
                         _snapshot(np.asarray([row["direction"] for row in rows], dtype=np.int64)),
                         _snapshot(_offsets(region)),
                         _snapshot(np.stack([row["lfp"] for row in rows])), first["lfp_guide"],
                         None if view_role != "fit" else _snapshot(np.stack([row["spikes"] for row in rows])),
                         first["unit_guide"])
        _validate_day(pack, check, _phase=_phase)
        result.append(pack)
        del rows, first, pack
    return tuple(result)


def load_native_trials(views, *, view_role, animal, target_day, region, stream, check):
    """Use the existing candidate reader only after validating its exact scope."""
    _check(check)
    _scope(view_role, animal, target_day, region, stream)
    records = views.iter_records(view_role, animal=animal, target_day=target_day,
                                 region=region, stream=stream)
    return adapt_records(records, view_role=view_role, animal=animal,
                         target_day=target_day, region=region, stream=stream, check=check)


def adapt_held_out_records(records, *, animal, target_day, region, stream, check):
    """Explicit final feature adapter; preserve the held-out role without targets."""
    return _adapt_records(records, view_role="held_out_features", animal=animal,
        target_day=target_day, region=region, stream=stream, check=check, _phase="held_out")


def load_held_out_trials(views, *, animal, target_day, region, stream, check):
    """Load a target-only HeldOutFeatureViews iterator; no candidate role changes."""
    _check(check)
    _scope("held_out_features", animal, target_day, region, stream, _phase="held_out")
    records = views.iter_records(animal=animal, target_day=target_day,
                                region=region, stream=stream, check=check)
    return adapt_held_out_records(records, animal=animal, target_day=target_day,
                                 region=region, stream=stream, check=check)


def fit_calibration_mean(day, *, check):
    """Arithmetic native-count means for each target direction/time/neuron."""
    _check(check)
    if (not isinstance(day, DayTrials) or day.view_role != "fit"
            or day.role != "calibration" or day.day != day.target_day):
        raise PermissionError("direction/time baseline fits target calibration only")
    directions, counts = _validate_day(day, check)
    means = np.empty((8, len(day.bin_offsets), len(day.unit_guide)), dtype=np.float64)
    for direction in range(8):
        _check(check)
        means[direction] = np.mean(day.spikes[directions == direction], axis=0, dtype=np.float64)
    if not np.isfinite(means).all():
        raise ValueError("calibration mean is not finite")
    _check(check)
    return CalibrationMean(day.animal, day.target_day, day.region, day.trial_ids,
                           _snapshot(counts), _snapshot(day.bin_offsets),
                           _snapshot(day.unit_guide), _snapshot(means))


def predict_calibration_mean(model, day, *, check):
    """Predict matching candidate development trials, carrying the fitted axis.

    No development target or unit-guide read is requested. The target neuron
    axis comes solely from calibration; the steward fixed it across eligible
    target-day trials. Held-out deployment is outside this fitting interface.
    """
    return _predict_calibration_mean(model, day, check=check)


def predict_calibration_mean_held_out(model, day, *, check):
    """Apply frozen calibration means to explicit held-out FEATURES only."""
    return _predict_calibration_mean(model, day, check=check, _phase="held_out")


def _predict_calibration_mean(model, day, *, check, _phase="candidate"):
    _check(check)
    if not isinstance(model, CalibrationMean) or not isinstance(day, DayTrials):
        raise TypeError("expected a calibration mean and a day-specific trial pack")
    expected = (("development_features", "development") if _phase == "candidate" else
                ("held_out_features", "held_out") if _phase == "held_out" else None)
    if expected is None or (day.view_role, day.role) != expected:
        raise PermissionError("baseline deployment requires its explicit feature-only phase")
    if (day.animal, day.day, day.target_day, day.region) != (
            model.animal, model.target_day, model.target_day, model.region):
        raise ValueError("prediction day does not match the target calibration model")
    _trial_ids(model.calibration_trial_ids)
    if set(model.calibration_trial_ids).intersection(day.trial_ids):
        raise ValueError("calibration and prediction whole-trial roles overlap")
    directions, _ = _validate_day(day, check, _phase=_phase)
    if not np.array_equal(model.bin_offsets, day.bin_offsets):
        raise ValueError("prediction native time axis differs from calibration")
    units = _guide(model.unit_guide, 2, "unit_guide")
    means = _numeric(model.means, 3, "calibration means")
    counts = np.asarray(model.trials_per_direction)
    if (counts.shape != (8,) or counts.dtype.kind not in "iu" or np.any(counts < 3)
            or int(counts.sum()) != len(model.calibration_trial_ids)):
        raise ValueError("calibration support is incomplete")
    if means.shape != (8, len(day.bin_offsets), len(units)) or np.any(means < 0):
        raise ValueError("calibration mean native axes or count scale are invalid")
    _check(check)
    return NativePredictions(day.animal, day.target_day, day.region, day.trial_ids,
                             _snapshot(directions), _snapshot(day.bin_offsets),
                             _snapshot(units), _snapshot(means[directions]),
                             model.calibration_trial_ids)
