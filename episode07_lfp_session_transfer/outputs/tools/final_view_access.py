"""Explicit EP07 held-out FEATURE reader; no neural response dataset access.

The caller supplies the exact prepared heldout.h5 path after the applicable
freeze/opening authorization. This interface neither establishes that authority
nor changes CandidateViews. The prepared file contains features and responses;
only feature objects are dereferenced. Response child names may be inspected as
structural metadata, but spikes/unit_guide objects and values are never fetched.
This is a procedural boundary under the same account, not OS isolation.
"""
from pathlib import Path

import h5py
import numpy as np

from native_trials import _trial_ids
from view_access import CandidateViews, FEATURE_KEYS, FIT_KEYS


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned cancellation/resource check required")
    check()


def _object(parent, name, expected_type):
    if not isinstance(parent.get(name, getlink=True), h5py.HardLink):
        raise ValueError("held-out feature reads cannot follow missing, soft or external links")
    value = parent[name]
    if not isinstance(value, expected_type):
        raise ValueError("unexpected held-out feature object type")
    if isinstance(value, h5py.Dataset) and (value.is_virtual or value.external):
        raise ValueError("held-out features cannot use virtual or external storage")
    return value


class HeldOutFeatureViews:
    """One explicit sealed-file path, with target-only feature iteration."""

    def __init__(self, heldout_path):
        self._path = Path(heldout_path)

    def iter_records(self, *, animal, target_day, region, stream, check):
        _check(check)
        CandidateViews._check_scope(animal, target_day, region)
        if stream not in ("history_assisted", "today_only"):
            raise ValueError("unknown frozen model stream")
        with h5py.File(self._path, "r") as handle:
            parent = _object(handle, animal, h5py.Group)
            parent = _object(parent, target_day, h5py.Group)
            group = _object(parent, region, h5py.Group)
            ids = tuple(group.keys())
            _trial_ids(ids)
            for trial in sorted(ids, key=lambda value: int(value[6:])):
                _check(check)
                record = _object(group, trial, h5py.Group)
                if record.attrs.get("role") != "held_out":
                    raise PermissionError("the final feature reader accepts held-out trials only")
                direction = record.attrs.get("direction")
                if (isinstance(direction, (bool, np.bool_))
                        or not isinstance(direction, (int, np.integer)) or not 0 <= direction < 8):
                    raise ValueError("held-out direction must be an integer in 0..7")
                if set(record.keys()) != set(FIT_KEYS):
                    raise ValueError("prepared held-out record has unexpected fields")
                # Validate every requested feature object's storage before any
                # feature value read. Do not fetch either response object.
                fields = {key: _object(record, key, h5py.Dataset) for key in FEATURE_KEYS}
                _check(check)
                yield {"animal": animal, "day": target_day, "region": region,
                       "trial_id": trial, "role": "held_out", "direction": int(direction),
                       **{key: np.asarray(fields[key]) for key in FEATURE_KEYS}}
