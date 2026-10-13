"""Scorer-owned complete-panel EP07 development feedback.

The candidate fitter must never receive this module's target reader or scorer
directory. This remains a procedural same-account boundary. The only file it
can open is the prepared development_targets.h5; no held-out interface exists.
Report aggregate development feedback, not trial/neuron errors or targets.

Joint R2 is 1 - sum(SSE) / sum(SST), with each neuron's scored-response mean
computed over all trials and native bins. It is not an average of neuron R2s.
Zero total SST is unavailable; constant neurons remain in the full SSE and no
column is dropped or stabilized with epsilon. Negative scores remain visible.
"""
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from native_trials import NativePredictions, NEW


@dataclass(frozen=True)
class DevelopmentFeedback:
    region: str
    model_r2: float | None
    calibration_mean_r2: float | None
    improvement_over_mean: float | None
    available: bool
    unavailable_reason: str | None
    complete_new_days: int
    independent_animals: int = 2


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned resource check required")
    check()


def joint_r2(observed, predicted, *, check):
    """One complete native day/region score, preserving every neuron and bin."""
    _check(check)
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    if (observed.dtype.kind not in "fiu" or predicted.dtype.kind not in "fiu"
            or observed.ndim != 3 or 0 in observed.shape
            or observed.shape != predicted.shape
            or not np.isfinite(observed).all() or not np.isfinite(predicted).all()):
        raise ValueError("complete finite matched trial/bin/neuron arrays required")
    y, p = observed.astype(np.float64), predicted.astype(np.float64)
    mean = np.mean(y, axis=(0, 1), keepdims=True)
    sst = float(np.sum(np.square(y - mean), dtype=np.float64))
    sse = float(np.sum(np.square(y - p), dtype=np.float64))
    if not np.isfinite(sst) or not np.isfinite(sse):
        raise FloatingPointError("nonfinite complete-day score arithmetic")
    _check(check)
    return None if sst == 0 else 1.0 - sse / sst


def _prediction(prediction):
    if not isinstance(prediction, NativePredictions):
        raise TypeError("native-axis predictions required")
    if (prediction.animal not in NEW
            or prediction.target_day not in NEW[prediction.animal]
            or prediction.region not in ("M1", "PMd")):
        raise ValueError("prediction outside the retained panel")
    ids = prediction.trial_ids
    if (not isinstance(ids, tuple) or not ids or len(set(ids)) != len(ids)
            or any(not isinstance(key, str) or not key for key in ids)
            or set(ids).intersection(prediction.calibration_trial_ids)):
        raise ValueError("unique development trials disjoint from calibration required")
    directions, offsets = np.asarray(prediction.directions), np.asarray(prediction.bin_offsets)
    if (directions.dtype.kind not in "iu" or directions.shape != (len(ids),)
            or np.any((directions < 0) | (directions > 7))
            or np.any(np.bincount(directions.astype(np.int64), minlength=8) < 6)
            or offsets.dtype.kind not in "iu"
            or not np.array_equal(offsets, np.arange(5, 15) if prediction.region == "M1"
                                  else np.arange(-15, 0))):
        raise ValueError("complete eight-direction native development window required")
    units = np.asarray(prediction.unit_guide)
    values = np.asarray(prediction.values)
    if (units.dtype.kind not in "fiu" or units.ndim != 2 or units.shape[1] != 2
            or len(units) == 0 or not np.isfinite(units).all()
            or len({tuple(row) for row in units.tolist()}) != len(units)
            or values.dtype.kind not in "fiu"
            or values.shape != (len(ids), len(offsets), len(units))
            or not np.isfinite(values).all()):
        raise ValueError("finite complete target-native prediction axis required")
    return prediction.animal, prediction.target_day, prediction.region


def _panel(predictions):
    predictions = tuple(predictions)
    result = {}
    for prediction in predictions:
        key = _prediction(prediction)
        if key in result:
            raise ValueError("duplicate day/region prediction")
        result[key] = prediction
    expected = {(animal, day, region) for animal in NEW for day in NEW[animal]
                for region in ("M1", "PMd")}
    if set(result) != expected:
        raise ValueError("all five target days and both regions required; no renormalization")
    return result


def _read_targets(handle, prediction, check):
    group = handle[f"{prediction.animal}/{prediction.target_day}/{prediction.region}"]
    if set(group) != set(prediction.trial_ids):
        raise ValueError("predicted trials do not equal the complete scorer day")
    targets = []
    for index, trial in enumerate(prediction.trial_ids):
        _check(check)
        record = group[trial]
        if (set(record) != {"spikes", "unit_guide"}
                or record.attrs["role"] != "development"
                or int(record.attrs["direction"]) != int(prediction.directions[index])
                or float(record.attrs["native_bin_seconds"]) != .03):
            raise PermissionError("target record is not the bound native development trial")
        if not np.array_equal(np.asarray(record["unit_guide"]), prediction.unit_guide):
            raise ValueError("scorer and prediction native neuron axes differ")
        values = np.asarray(record["spikes"])
        if (values.dtype.kind not in "fiu" or values.shape != prediction.values.shape[1:]
                or not np.isfinite(values).all() or np.any(values < 0)
                or np.any(values != np.floor(values))):
            raise ValueError("scorer target lost its complete native integer representation")
        targets.append(values)
    return np.stack(targets)


def score_development_panel(scorer_directory, predictions, calibration_predictions, *, check):
    """Open scorer targets only after both complete prediction panels validate.

    Feedback averages target days within animal, then animals equally. Only
    per-region aggregate scores/differences are returned. This function neither
    selects a model nor pairs it with a history/today model from another stream.
    """
    _check(check)
    model = _panel(predictions)
    baseline = _panel(calibration_predictions)
    for key, prediction in model.items():
        reference = baseline[key]
        if (prediction.trial_ids != reference.trial_ids
                or prediction.calibration_trial_ids != reference.calibration_trial_ids
                or not np.array_equal(prediction.directions, reference.directions)
                or not np.array_equal(prediction.bin_offsets, reference.bin_offsets)
                or not np.array_equal(prediction.unit_guide, reference.unit_guide)):
            raise ValueError("candidate/comparator trial or native axes are not paired")
    scores = {}
    with h5py.File(Path(scorer_directory) / "development_targets.h5", "r") as handle:
        for key, prediction in model.items():
            y = _read_targets(handle, prediction, check)
            scores[key] = (joint_r2(y, prediction.values, check=check),
                           joint_r2(y, baseline[key].values, check=check))
    feedback = []
    for region in ("M1", "PMd"):
        values = [scores[(animal, day, region)] for animal in NEW for day in NEW[animal]]
        if any(model_score is None or baseline_score is None
               for model_score, baseline_score in values):
            feedback.append(DevelopmentFeedback(region, None, None, None, False,
                "zero_joint_native_response_variance_in_required_day", 5))
            continue
        aggregates = []
        for position in (0, 1):
            aggregates.append(float(np.mean([
                np.mean([scores[(animal, day, region)][position] for day in NEW[animal]])
                for animal in NEW])))
        feedback.append(DevelopmentFeedback(region, aggregates[0], aggregates[1],
            aggregates[0] - aggregates[1], True, None, 5))
    _check(check)
    return tuple(feedback)
