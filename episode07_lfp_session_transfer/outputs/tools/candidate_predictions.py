"""Exclusive EP07 development-prediction publication, with no outcome reader.

A caller supplies one explicit output path and declared candidate/stream.
The context writes one animal/day/region at a time, retaining only small
identity records between writes. It requires the exact five new days in both
regions, with whole-trial roles shared across M1/PMd, before setting completed
as its final publication marker. A failed or incomplete file is retained and
never resumed, overwritten or accepted by the loader. Existing completed
files are immutable through this API, not through operating-system isolation.

Files contain candidate predictions, the calibration-mean comparator, their
shared native axes, and a structural publication check record. No source,
scorer, evaluation or fitted-response input path is accepted. Loading follows
only hard links inside the explicit file; external/soft links, virtual arrays
and external-storage datasets are refused before value reads.

Checks establish complete record/axis support, not that predictions arose
from a claimed fit, that calibration IDs have per-direction support, that a
model passed scientific checks, or that an outcome opening is authorized.
Negative candidate predictions are retained. The calibration comparator must
be nonnegative and constant across trials of the same direction/native bin,
but its equality to an actual fitting mean is the upstream caller's duty.
"""
from dataclasses import dataclass
from pathlib import Path
import re

import h5py
import numpy as np

from native_trials import NEW, NativePredictions, _snapshot, _trial_ids


PANEL_KEYS = tuple((animal, day, region) for animal in ("Mihili", "Chewie-L")
                   for day in NEW[animal] for region in ("M1", "PMd"))
KIND = "EP07_development_prediction"
CHECK_KIND = "structural_publication_only"
_FIELDS = {"prediction", "calibration_mean", "trial_ids", "directions", "bin_offsets",
           "unit_guide", "calibration_trial_ids"}


@dataclass(frozen=True)
class PublicationCheck:
    kind: str
    complete_count: int


@dataclass(frozen=True)
class CandidatePredictions:
    candidate_id: str
    stream: str
    predictions: tuple[NativePredictions, ...]
    calibration_means: tuple[NativePredictions, ...]
    check_record: PublicationCheck


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned resource check required")
    check()


def _identity(candidate_id, stream):
    if not isinstance(candidate_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", candidate_id) is None:
        raise ValueError("a nonempty simple candidate identifier is required")
    if stream not in ("history_assisted", "today_only"):
        raise ValueError("unknown declared model stream")


def _prediction(value):
    if not isinstance(value, NativePredictions):
        raise TypeError("a native development prediction record is required")
    key = value.animal, value.target_day, value.region
    if key not in PANEL_KEYS:
        raise ValueError("prediction does not belong to the fixed five-day/two-region panel")
    _trial_ids(value.trial_ids)
    _trial_ids(value.calibration_trial_ids)
    if (len(value.calibration_trial_ids) < 24
            or set(value.calibration_trial_ids).intersection(value.trial_ids)):
        raise ValueError("distinct fitting calibration IDs and development IDs are required")
    directions, offsets, guide = map(np.asarray, (value.directions, value.bin_offsets, value.unit_guide))
    if (directions.dtype.kind not in "iu" or directions.shape != (len(value.trial_ids),)
            or np.any((directions < 0) | (directions > 7))
            or np.any(np.bincount(directions.astype(np.int64), minlength=8) < 6)):
        raise ValueError("all eight development directions need at least six whole trials")
    expected = np.arange(5, 15) if value.region == "M1" else np.arange(-15, 0)
    if offsets.dtype.kind not in "iu" or not np.array_equal(offsets, expected):
        raise ValueError("prediction native bin identities differ from the fixed setting")
    if (guide.dtype.kind not in "fiu" or guide.ndim != 2 or guide.shape[1] != 2
            or len(guide) == 0 or not np.isfinite(guide).all()
            or len({tuple(row) for row in guide.tolist()}) != len(guide)):
        raise ValueError("a complete unique native neuron guide is required")
    raw = np.asarray(value.values)
    if raw.dtype.kind not in "fiu" or raw.shape != (len(value.trial_ids), len(offsets), len(guide)):
        raise ValueError("prediction trial/bin/neuron dimensions disagree with their identities")
    with np.errstate(over="ignore", invalid="ignore"):
        values = _snapshot(np.asarray(raw, dtype=np.float64))
    if not np.isfinite(values).all():
        raise ValueError("prediction values must be finite in float64 storage")
    return NativePredictions(value.animal, value.target_day, value.region, value.trial_ids,
        _snapshot(directions.astype(np.int64)), _snapshot(offsets.astype(np.int64)),
        _snapshot(guide), values, value.calibration_trial_ids)


def _pair(prediction, calibration_mean):
    prediction, calibration_mean = _prediction(prediction), _prediction(calibration_mean)
    if (prediction.animal, prediction.target_day, prediction.region, prediction.trial_ids,
            prediction.calibration_trial_ids) != (calibration_mean.animal, calibration_mean.target_day,
                calibration_mean.region, calibration_mean.trial_ids, calibration_mean.calibration_trial_ids):
        raise ValueError("candidate and calibration comparator identities differ")
    if any(not np.array_equal(getattr(prediction, field), getattr(calibration_mean, field))
           for field in ("directions", "bin_offsets", "unit_guide")):
        raise ValueError("candidate and calibration comparator native axes differ")
    if np.any(calibration_mean.values < 0):
        raise ValueError("a native calibration count mean cannot be negative")
    for direction in range(8):
        rows = calibration_mean.values[calibration_mean.directions == direction]
        if not np.all(rows == rows[0]):
            raise ValueError("calibration comparator must repeat its direction/native-bin mean")
    return prediction, calibration_mean


def _session_binding(records, prediction):
    """The simultaneous regions share trial roles, not native neuron/bin axes."""
    key = prediction.animal, prediction.target_day
    value = (prediction.trial_ids, prediction.calibration_trial_ids, tuple(prediction.directions))
    if key in records and records[key] != value:
        raise ValueError("M1 and PMd must share whole-trial identities, directions and calibration IDs")
    records[key] = value


class CandidatePredictionWriter:
    """One exclusive-create full-panel context; write each day/region once."""

    def __init__(self, path, *, candidate_id, stream, check):
        _identity(candidate_id, stream)
        self.path = Path(path)
        self.candidate_id, self.stream, self.check = candidate_id, stream, check
        self._handle, self._entered, self._failed = None, False, False
        self._seen, self._sessions = set(), {}

    def __enter__(self):
        _check(self.check)
        if self._entered:
            raise RuntimeError("a publication context cannot be entered twice")
        self._entered = True
        handle = h5py.File(self.path, "x")
        try:
            handle.attrs["kind"] = KIND
            handle.attrs["candidate_id"] = self.candidate_id
            handle.attrs["stream"] = self.stream
            handle.attrs["completed"] = False
            handle.create_group("predictions")
            handle.flush()
        except BaseException:
            handle.close()
            raise
        self._handle = handle
        return self

    def write(self, prediction, calibration_mean):
        if self._handle is None or self._failed:
            raise RuntimeError("publication context is closed or failed")
        try:
            _check(self.check)
            prediction, calibration_mean = _pair(prediction, calibration_mean)
            key = prediction.animal, prediction.target_day, prediction.region
            if key in self._seen:
                raise ValueError("a prediction cell cannot be overwritten or duplicated")
            _session_binding(self._sessions, prediction)
            _check(self.check)
            group = self._handle["predictions"].require_group(prediction.animal).require_group(
                prediction.target_day).create_group(prediction.region)
            text_type = h5py.string_dtype(encoding="utf-8")
            group.create_dataset("trial_ids", data=prediction.trial_ids, dtype=text_type)
            group.create_dataset("calibration_trial_ids", data=prediction.calibration_trial_ids, dtype=text_type)
            for field in ("directions", "bin_offsets", "unit_guide"):
                group.create_dataset(field, data=getattr(prediction, field))
            group.create_dataset("prediction", data=prediction.values)
            group.create_dataset("calibration_mean", data=calibration_mean.values)
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
                    raise ValueError("publication failed or lacks the complete five-day/two-region panel")
                record = handle.create_group("check_record")
                record.attrs["kind"] = CHECK_KIND
                record.attrs["complete_count"] = len(PANEL_KEYS)
                handle.flush()
                _check(self.check)
                handle.attrs["completed"] = True
                handle.flush()
        finally:
            handle.close()
            self._handle = None
        return False


def _object(parent, name, expected_type):
    if not isinstance(parent.get(name, getlink=True), h5py.HardLink):
        raise ValueError("prediction artifacts cannot follow missing, soft or external HDF5 links")
    result = parent[name]
    if not isinstance(result, expected_type):
        raise ValueError("unexpected prediction artifact object type")
    if isinstance(result, h5py.Dataset) and (result.is_virtual or result.external):
        raise ValueError("prediction arrays cannot use virtual or external storage")
    return result


def _read_cell(group, key):
    if set(group) != _FIELDS:
        raise ValueError("prediction cell has missing or unexpected fields")
    datasets = {field: _object(group, field, h5py.Dataset) for field in _FIELDS}
    if datasets["trial_ids"].ndim != 1 or datasets["calibration_trial_ids"].ndim != 1:
        raise ValueError("prediction trial identity arrays must be one-dimensional")
    ids = tuple(datasets["trial_ids"].asstr()[...].tolist())
    calibration_ids = tuple(datasets["calibration_trial_ids"].asstr()[...].tolist())
    directions, offsets, guide = (datasets[field][...] for field in ("directions", "bin_offsets", "unit_guide"))
    animal, day, region = key
    common = (animal, day, region, ids, directions, offsets, guide)
    return _pair(NativePredictions(*common, datasets["prediction"][...], calibration_ids),
                 NativePredictions(*common, datasets["calibration_mean"][...], calibration_ids))


def load_candidate_predictions(path, *, expected_candidate_id, expected_stream, check):
    """Load only a complete declared panel from one explicit prediction file."""
    _identity(expected_candidate_id, expected_stream)
    _check(check)
    predictions, baselines, sessions = [], [], {}
    with h5py.File(path, "r") as handle:
        completed = handle.attrs.get("completed")
        if (not isinstance(completed, (bool, np.bool_)) or not completed
                or handle.attrs.get("kind") != KIND
                or handle.attrs.get("candidate_id") != expected_candidate_id
                or handle.attrs.get("stream") != expected_stream):
            raise ValueError("incomplete prediction artifact or candidate/stream identity mismatch")
        if set(handle) != {"predictions", "check_record"}:
            raise ValueError("prediction artifact lacks its exact complete publication structure")
        record = _object(handle, "check_record", h5py.Group)
        count = record.attrs.get("complete_count")
        if (record.attrs.get("kind") != CHECK_KIND or isinstance(count, (bool, np.bool_))
                or not isinstance(count, (int, np.integer)) or count != len(PANEL_KEYS)):
            raise ValueError("structural publication check is incomplete")
        root = _object(handle, "predictions", h5py.Group)
        if set(root) != {"Mihili", "Chewie-L"}:
            raise ValueError("both fixed animals are required")
        for animal in ("Mihili", "Chewie-L"):
            animal_group = _object(root, animal, h5py.Group)
            if set(animal_group) != set(NEW[animal]):
                raise ValueError("all and only the five fixed new days are required")
            for day in NEW[animal]:
                day_group = _object(animal_group, day, h5py.Group)
                if set(day_group) != {"M1", "PMd"}:
                    raise ValueError("both fixed regions are required for every new day")
                for region in ("M1", "PMd"):
                    _check(check)
                    pair = _read_cell(_object(day_group, region, h5py.Group), (animal, day, region))
                    _session_binding(sessions, pair[0])
                    predictions.append(pair[0])
                    baselines.append(pair[1])
    _check(check)
    return CandidatePredictions(expected_candidate_id, expected_stream, tuple(predictions),
        tuple(baselines), PublicationCheck(CHECK_KIND, len(PANEL_KEYS)))
