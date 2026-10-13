"""Paired native held-out prediction publication, without a response reader.

Both finalists and the calibration comparator must cover all five fixed days
and both regions before the artifact is complete. This distinct held-out
format never relabels a development artifact. NativePredictions is a
phase-neutral array record; the final caller owns authentic held trial roles,
frozen fitting state and opening authority. Structural completion alone grants
none of those. No fitting data, evaluation targets or fitted state are read.

Mathematical native-axis checks are shared with candidate_predictions, whose
writer and development-kind loader are not used. All output creation is
exclusive; incomplete artifacts remain incomplete and cannot be resumed here.
"""
from dataclasses import dataclass

import h5py
import numpy as np

from candidate_predictions import (_check, _identity, _object, _pair,
                                   _session_binding, PANEL_KEYS)
from native_trials import NEW, NativePredictions


KIND = "EP07_paired_held_out_predictions"
ROLE = "held_out"
_AXES = ("directions", "bin_offsets", "unit_guide")
_FIELDS = {"trial_ids", "calibration_trial_ids", *_AXES,
           "history_prediction", "today_prediction", "calibration_mean"}


@dataclass(frozen=True)
class HeldOutPredictions:
    history_candidate_id: str
    today_candidate_id: str
    history: tuple[NativePredictions, ...]
    today: tuple[NativePredictions, ...]
    calibration_means: tuple[NativePredictions, ...]


def _triple(history, today, baseline):
    history, baseline = _pair(history, baseline)
    today, _ = _pair(today, baseline)
    return history, today, baseline


class HeldOutPredictionWriter:
    """Publish a BOTH-stream full panel before any final scoring process."""

    def __init__(self, path, *, history_candidate_id, today_candidate_id, check):
        _identity(history_candidate_id, "history_assisted")
        _identity(today_candidate_id, "today_only")
        self.path, self.check = path, check
        self.history_id, self.today_id = history_candidate_id, today_candidate_id
        self._handle, self._entered, self._failed = None, False, False
        self._seen, self._sessions = set(), {}

    def __enter__(self):
        _check(self.check)
        if self._entered:
            raise RuntimeError("a final publication context cannot be entered twice")
        self._entered = True
        handle = h5py.File(self.path, "x")
        try:
            handle.attrs.update(kind=KIND, role=ROLE, completed=False,
                                history_candidate_id=self.history_id,
                                today_candidate_id=self.today_id)
            handle.create_group("predictions")
            handle.flush()
        except BaseException:
            handle.close()
            raise
        self._handle = handle
        return self

    def write(self, history, today, calibration_mean):
        if self._handle is None or self._failed:
            raise RuntimeError("final publication is closed or failed")
        try:
            _check(self.check)
            history, today, baseline = _triple(history, today, calibration_mean)
            key = history.animal, history.target_day, history.region
            if key in self._seen:
                raise ValueError("a final panel cell cannot be duplicated")
            _session_binding(self._sessions, history)
            group = self._handle["predictions"].require_group(history.animal).require_group(
                history.target_day).create_group(history.region)
            string = h5py.string_dtype(encoding="utf-8")
            group.create_dataset("trial_ids", data=history.trial_ids, dtype=string)
            group.create_dataset("calibration_trial_ids", data=history.calibration_trial_ids,
                                 dtype=string)
            for field in _AXES:
                group.create_dataset(field, data=getattr(history, field))
            for field, value in (("history_prediction", history),
                                 ("today_prediction", today), ("calibration_mean", baseline)):
                group.create_dataset(field, data=value.values)
            self._handle.flush()
            _check(self.check)
            self._seen.add(key)
        except BaseException:
            self._failed = True
            raise

    def __exit__(self, exception_type, exception, traceback):
        handle = self._handle
        if handle is None:
            return False
        try:
            if exception_type is None:
                _check(self.check)
                if self._failed or self._seen != set(PANEL_KEYS):
                    raise ValueError("both finalists require all ten native cells")
                handle.attrs["complete_cells"] = len(PANEL_KEYS)
                handle.flush()
                _check(self.check)
                handle.attrs["completed"] = True
                handle.flush()
        finally:
            handle.close()
            self._handle = None
        return False


def _read(group, key):
    if set(group) != _FIELDS:
        raise ValueError("paired final cell has unexpected or missing fields")
    datasets = {field: _object(group, field, h5py.Dataset) for field in _FIELDS}
    if any(datasets[field].ndim != 1 for field in ("trial_ids", "calibration_trial_ids")):
        raise ValueError("final trial identifiers must be one-dimensional")
    ids = tuple(datasets["trial_ids"].asstr()[...].tolist())
    calibration = tuple(datasets["calibration_trial_ids"].asstr()[...].tolist())
    common = (*key, ids, *(datasets[field][...] for field in _AXES))
    return _triple(*(NativePredictions(*common, datasets[field][...], calibration)
                     for field in ("history_prediction", "today_prediction", "calibration_mean")))


def load_held_out_predictions(path, *, history_candidate_id, today_candidate_id, check):
    """Read a complete paired prediction artifact; never final response data."""
    _identity(history_candidate_id, "history_assisted")
    _identity(today_candidate_id, "today_only")
    _check(check)
    history, today, baselines, sessions = [], [], [], {}
    with h5py.File(path, "r") as handle:
        complete = handle.attrs.get("completed")
        count = handle.attrs.get("complete_cells")
        if (not isinstance(complete, (bool, np.bool_)) or not complete
                or handle.attrs.get("kind") != KIND or handle.attrs.get("role") != ROLE
                or handle.attrs.get("history_candidate_id") != history_candidate_id
                or handle.attrs.get("today_candidate_id") != today_candidate_id
                or isinstance(count, (bool, np.bool_))
                or not isinstance(count, (int, np.integer)) or count != len(PANEL_KEYS)
                or set(handle) != {"predictions"}):
            raise ValueError("final prediction artifact is incomplete or has a different identity")
        root = _object(handle, "predictions", h5py.Group)
        if set(root) != {"Mihili", "Chewie-L"}:
            raise ValueError("both fixed animals are required")
        for animal in ("Mihili", "Chewie-L"):
            animal_group = _object(root, animal, h5py.Group)
            if set(animal_group) != set(NEW[animal]):
                raise ValueError("all and only the fixed new days are required")
            for day in NEW[animal]:
                day_group = _object(animal_group, day, h5py.Group)
                if set(day_group) != {"M1", "PMd"}:
                    raise ValueError("both regions are required for each final day")
                for region in ("M1", "PMd"):
                    _check(check)
                    h, t, baseline = _read(_object(day_group, region, h5py.Group),
                                           (animal, day, region))
                    _session_binding(sessions, h)
                    history.append(h)
                    today.append(t)
                    baselines.append(baseline)
    _check(check)
    return HeldOutPredictions(history_candidate_id, today_candidate_id,
                              tuple(history), tuple(today), tuple(baselines))
