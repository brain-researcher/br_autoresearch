"""Separate EP07 held-count transport into the fixed primary inference engine.

The complete BOTH-stream prediction artifact is loaded and validated before
the caller's ``begin_opening()`` callback. That callback must durably claim the
single authorized primary opening, refusing any earlier claim, before returning.
If it raises, this module never opens the supplied held-out file. The caller
also owns freeze/scientific-check authority and durable success/failure records;
this procedural callback does not establish those facts or OS-level isolation.

After that claim, all ten original held-out cell/trial/native-axis bindings are
checked before any spike values are read. Only bin offsets, neuron identities
and native counts are dereferenced; no LFP dataset, development adapter, fitting
state or selection input is used. No trial/neuron is dropped or reordered by
its values. Predictions fix row order; infer_primary separately implements the
registered numerical trial order and shared bootstrap schedule.

There is no retry, writer, CLI or alternative inferential rule here. Returned
arithmetic_pass excludes external scientific checks and authorizes no opening.
"""
import h5py
import numpy as np

from final_predictions import load_held_out_predictions
from final_view_access import _object
from native_trials import _snapshot
from primary_inference import DAY_ORDER, REGIONS, HeldOutCell, infer_primary


_FIELDS = {"lfp", "lfp_guide", "bin_offsets", "spikes", "unit_guide"}


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned cancellation/resource check required")
    check()


def _read_cells(handle, panel, check):
    """Bind the full original panel before reading the first count value."""
    expected_days = {animal: {day for owner, day in DAY_ORDER if owner == animal}
                     for animal, _ in DAY_ORDER}
    if set(handle) != set(expected_days):
        raise ValueError("held-out targets must contain exactly both fixed animals")
    groups = {}
    for animal, days in expected_days.items():
        animal_group = _object(handle, animal, h5py.Group)
        if set(animal_group) != days:
            raise ValueError("held-out targets must retain all and only the five fixed days")
        for day in sorted(days):
            day_group = _object(animal_group, day, h5py.Group)
            if set(day_group) != set(REGIONS):
                raise ValueError("both fixed regions are required for every held-out day")
            for region in REGIONS:
                groups[animal, day, region] = _object(day_group, region, h5py.Group)
    bound = []
    for history, today, baseline in zip(panel.history, panel.today, panel.calibration_means):
        _check(check)
        key = history.animal, history.target_day, history.region
        group = groups[key]
        if set(group) != set(history.trial_ids):
            raise ValueError("predictions must cover exactly every original held-out trial")
        targets = []
        for index, trial in enumerate(history.trial_ids):
            _check(check)
            record = _object(group, trial, h5py.Group)
            direction = record.attrs.get("direction")
            seconds = record.attrs.get("native_bin_seconds")
            if (record.attrs.get("role") != "held_out"
                    or isinstance(direction, (bool, np.bool_))
                    or not isinstance(direction, (int, np.integer))
                    or direction != int(history.directions[index])
                    or isinstance(seconds, (bool, np.bool_))
                    or not isinstance(seconds, (int, float, np.integer, np.floating))
                    or seconds != .03):
                raise PermissionError("target is not the bound native held-out trial")
            if set(record) != _FIELDS:
                raise ValueError("held-out record has unexpected or missing fields")
            # Inspect requested storage before values; LFP objects stay closed.
            offsets = _object(record, "bin_offsets", h5py.Dataset)
            units = _object(record, "unit_guide", h5py.Dataset)
            spikes = _object(record, "spikes", h5py.Dataset)
            if (offsets.dtype.kind not in "iu" or offsets.shape != history.bin_offsets.shape
                    or units.dtype.kind not in "fiu" or units.shape != history.unit_guide.shape
                    or spikes.dtype.kind not in "fiu" or spikes.shape != history.values.shape[1:]):
                raise ValueError("held-out target native array dimensions or representation differ")
            if (not np.array_equal(offsets[...], history.bin_offsets)
                    or not np.array_equal(units[...], history.unit_guide)):
                raise ValueError("held-out and frozen prediction native axes differ")
            targets.append(spikes)
        bound.append((history, today, baseline, targets))
    cells = []
    for history, today, baseline, targets in bound:
        values = []
        for target in targets:
            _check(check)
            raw = target[...]
            if (not np.isfinite(raw).all() or np.any(raw < 0)
                    or np.any(raw != np.floor(raw))):
                raise ValueError("held-out responses must remain finite nonnegative native integer counts")
            values.append(raw)
        cells.append(HeldOutCell(history.animal, history.target_day, history.region,
            "held_out", history.trial_ids, history.directions, history.bin_offsets,
            history.unit_guide, _snapshot(np.stack(values)), history, today, baseline))
    _check(check)
    return tuple(cells)


def score_primary_held_out(heldout_path, prediction_path, *, history_candidate_id,
                          today_candidate_id, begin_opening, check):
    """Load paired predictions, claim one opening, then return PrimaryInference.

    ``begin_opening`` takes no arguments and must raise if a durable opening
    record already exists. Its implementation belongs to the final caller;
    passing a no-op does not certify a valid single opening. Failure after the
    callback remains an opened/technically incomplete result, never a retry.
    Paths are explicit files, never directories to scan or infer from indexes.
    """
    _check(check)
    if not callable(begin_opening):
        raise TypeError("a caller-owned durable single-opening callback is required")
    panel = load_held_out_predictions(prediction_path,
        history_candidate_id=history_candidate_id, today_candidate_id=today_candidate_id,
        check=check)
    _check(check)
    begin_opening()
    _check(check)
    with h5py.File(heldout_path, "r") as handle:
        cells = _read_cells(handle, panel, check)
    return infer_primary(cells, check=check)
