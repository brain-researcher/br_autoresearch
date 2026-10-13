"""Role-scoped readers for EP06's completed seven-session development export.

Candidate training returns native counts and the fixed training-selected 15
electrodes. Candidate calibration/evaluation access returns LFP only. A separate
outcome reader is bound to exactly one nontraining role. Both preserve whole
trials, native bins and all native units; no scores, fits or role reassignment
occur here. The full 96-electrode export remains intact for separately admitted
controls, but this ordinary candidate API never substitutes another subset.

The shared-account boundary is procedural, not OS isolation. The exporter owns
the EP07 handoff prerequisite; these readers require its completed artifact.
The caller still owns scientific admission, outcome-opening records, resource
limits and cancellation. Metadata/array headers are checked before neural value
access. These readers do not open reserved dates or source MATLAB files.
"""
from dataclasses import dataclass
from pathlib import Path
import math

import h5py
import numpy as np

from bandwise_prediction import BandwiseLFP
from component_scoring import BoundActivity, FrozenDesign
from prepare_development_views import (
    ANGLE_TOLERANCE, BANDS, KIND, OFFSETS, REGIONS, ROLES, SESSIONS, _compute_only,
)


@dataclass(frozen=True)
class DevelopmentRole:
    animal: str
    region: str
    design: FrozenDesign
    source_trial_indices: tuple
    direction_raw: tuple
    calibration_halves: tuple
    window_starts: tuple
    window_stops: tuple
    native_electrode_ids: tuple
    selected_electrode_positions: tuple
    selected_electrode_ids: tuple
    native_unit_guide: tuple


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned resource/cancellation check required")
    check()


def _node(parent, name, cls):
    if name not in parent or not isinstance(parent.get(name, getlink=True), h5py.HardLink):
        raise ValueError("development views require local hard-linked objects")
    node = parent[name]
    if not isinstance(node, cls):
        raise ValueError("development object has the wrong declared type")
    if isinstance(node, h5py.Dataset) and (node.is_virtual or node.external):
        raise ValueError("external or virtual dataset storage is not a development view")
    return node


def _true(value):
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _header(handle):
    if (handle.attrs.get("kind") != KIND or not _true(handle.attrs.get("completed"))
            or handle.attrs.get("expected_sessions") != len(SESSIONS)):
        raise PermissionError("a completed seven-session development export is required")
    sessions = _node(handle, "sessions", h5py.Group)
    if set(sessions) != {spec.session_id for spec in SESSIONS}:
        raise PermissionError("all and only the seven fixed development sessions are permitted")
    for spec in SESSIONS:
        session = _node(sessions, spec.session_id, h5py.Group)
        if (not _true(session.attrs.get("complete"))
                or session.attrs.get("animal") != spec.animal
                or session.attrs.get("source_filename") != spec.filename
                or session.attrs.get("original_position") != spec.original_position):
            raise ValueError("development session identity/completion is inconsistent")
    return sessions


def _numbers(group, name, shape, *, integer=False, boolean=False):
    node = _node(group, name, h5py.Dataset)
    allowed = "b" if boolean else "iu" if integer else "fiu"
    if node.shape != shape or node.dtype.kind not in allowed:
        raise ValueError(f"invalid development metadata shape/type: {name}")
    value = node[()]
    if not np.isfinite(value).all():
        raise ValueError(f"nonfinite development metadata: {name}")
    return value


def _array_header(group, name, shape):
    node = _node(group, name, h5py.Dataset)
    if node.shape != shape or node.dtype.kind not in "fiu":
        raise ValueError(f"native development array support differs: {name}")
    return node


def _session_metadata(session, spec, check):
    """Read metadata only, validating both regional partners and all roles."""
    _check(check)
    if set(session) != {"bands", "bin_offsets", *REGIONS}:
        raise ValueError("unexpected or missing development session objects")
    if (tuple(map(tuple, _numbers(session, "bands", (9, 2)))) != BANDS
            or tuple(_numbers(session, "bin_offsets", (10,), integer=True)) != OFFSETS):
        raise ValueError("the registered nine bands and native movement window are required")
    result = {}
    for region in REGIONS:
        group = _node(session, region, h5py.Group)
        if set(group) != {"electrode_ids", "training_finite_electrodes",
                         "selected_electrode_positions", "unit_guide", *ROLES}:
            raise ValueError("unexpected or missing development region objects")
        electrodes = tuple(float(x) for x in _numbers(group, "electrode_ids", (96,)))
        finite = _numbers(group, "training_finite_electrodes", (96,), boolean=True)
        selected = tuple(int(x) for x in _numbers(group, "selected_electrode_positions", (15,), integer=True))
        if (electrodes != tuple(float(x) for x in range(1, 97))
                or tuple(sorted(set(selected))) != selected or min(selected) < 0 or max(selected) >= 96
                or not finite[list(selected)].all()):
            raise ValueError("the fixed 15 training-finite physical electrodes are inconsistent")
        units_node = _node(group, "unit_guide", h5py.Dataset)
        if len(units_node.shape) != 2 or units_node.shape[1] != 2 or not 1 <= units_node.shape[0] <= 4096:
            raise ValueError("complete native unit-guide support is required")
        units = tuple(tuple(float(x) for x in row) for row in
                      _numbers(group, "unit_guide", units_node.shape))
        if len(set(units)) != len(units):
            raise ValueError("native unit identities must be unique without filtering")
        axes = tuple(f"{region}:electrode_{a!r}:unit_{b!r}" for a, b in units)
        coverage, directions_by_role = [], {}
        for role in ROLES:
            _check(check)
            rg = _node(group, role, h5py.Group)
            if set(rg) != {"source_trial_indices", "trial_ids", "direction", "direction_raw",
                           "calibration_half", "window_start", "window_stop", "lfp", "spikes"}:
                raise ValueError("unexpected or missing development role objects")
            indices_node = _node(rg, "source_trial_indices", h5py.Dataset)
            if len(indices_node.shape) != 1 or not 1 <= indices_node.shape[0] <= spec.trials:
                raise ValueError("a nonempty bounded whole-trial role is required")
            count = indices_node.shape[0]
            indices = tuple(int(x) for x in _numbers(rg, "source_trial_indices", (count,), integer=True))
            if tuple(sorted(set(indices))) != indices or indices[0] < 0 or indices[-1] >= spec.trials:
                raise ValueError("whole trials must retain unique original source order")
            ids = _node(rg, "trial_ids", h5py.Dataset)
            if ids.shape != (count,) or h5py.check_string_dtype(ids.dtype) is None:
                raise ValueError("one structural identity per whole trial is required")
            trial_ids = tuple(ids.asstr()[()])
            expected_ids = tuple(f"{spec.session_id}:row_{index + 1:06d}" for index in indices)
            if trial_ids != expected_ids:
                raise ValueError("structural trial identities do not match their source positions")
            direction = tuple(int(x) for x in _numbers(rg, "direction", (count,), integer=True))
            raw = tuple(float(x) for x in _numbers(rg, "direction_raw", (count,)))
            halves = tuple(int(x) for x in _numbers(rg, "calibration_half", (count,), integer=True))
            starts = tuple(int(x) for x in _numbers(rg, "window_start", (count,), integer=True))
            stops = tuple(int(x) for x in _numbers(rg, "window_stop", (count,), integer=True))
            for label, angle, start, stop in zip(direction, raw, starts, stops):
                nearest = math.floor(angle / (math.pi / 4) + .5)
                if (label != nearest % 8 or abs(angle - nearest * math.pi / 4) > ANGLE_TOLERANCE
                        or start < 0 or stop - start != 10):
                    raise ValueError("direction or complete native window metadata differs")
            minimum = 7 if role == "training" else 4
            if set(direction) != set(range(8)) or any(direction.count(d) < minimum for d in range(8)):
                raise ValueError("all eight directions need complete minimum whole-trial support")
            for d in range(8):
                assigned_halves = tuple(h for h, label in zip(halves, direction) if label == d)
                expected = tuple(i % 2 for i in range(len(assigned_halves))) if role == "calibration" else (-1,) * len(assigned_halves)
                if assigned_halves != expected:
                    raise ValueError("calibration halves must alternate source-ordered rows within direction")
            design = FrozenDesign(spec.filename, role, tuple(range(8)), trial_ids, direction, OFFSETS, axes)
            _array_header(rg, "lfp", (count, 10, 96, 9))
            _array_header(rg, "spikes", (count, 10, len(units)))
            result[region, role] = DevelopmentRole(spec.animal, region, design, indices, raw, halves,
                starts, stops, electrodes, selected, tuple(electrodes[i] for i in selected), units)
            coverage.extend(indices)
            directions_by_role[role] = direction
        if sorted(coverage) != list(range(spec.trials)):
            raise ValueError("every original trial must occur exactly once; no role overlap or drop")
        for d in range(8):
            train, cal, evaluation = (directions_by_role[role].count(d) for role in ROLES)
            total = train + cal + evaluation
            if train != total // 2 or cal != (total - train) // 2:
                raise ValueError("whole-trial role counts differ from the frozen direction fractions")
    for role in ROLES:
        left, right = (result[region, role] for region in REGIONS)
        for field in ("source_trial_indices", "direction_raw", "calibration_halves", "window_starts", "window_stops"):
            if getattr(left, field) != getattr(right, field):
                raise ValueError("paired regions must use identical trials, roles, directions and windows")
        if (left.design.trial_ids != right.design.trial_ids
                or left.design.trial_directions != right.design.trial_directions):
            raise ValueError("paired regional trial identities differ")
    return result


def _readonly(value):
    array = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class _DevelopmentViews:
    def __init__(self, path, *, check):
        _compute_only()
        _check(check)
        self._check = check
        self._path = Path(path)
        if not self._path.is_absolute() or self._path.name != "development_views.h5":
            raise ValueError("the explicit completed development_views.h5 path is required")
        with h5py.File(self._path, "r") as handle:
            sessions = _header(handle)
            for spec in SESSIONS:
                _session_metadata(sessions[spec.session_id], spec, self._check)

    def _read(self, session, region, role, *, features=False, outcomes=False):
        _check(self._check)
        matches = [spec for spec in SESSIONS if session in (spec.session_id, spec.filename)]
        if len(matches) != 1 or region not in REGIONS or role not in ROLES:
            raise PermissionError("only a named development session, region and role are readable")
        spec = matches[0]
        with h5py.File(self._path, "r") as handle:
            sessions = _header(handle)
            metadata = _session_metadata(sessions[spec.session_id], spec, self._check)[region, role]
            group = sessions[spec.session_id][region][role]
            lfp = counts = None
            if features:
                _check(self._check)
                values = group["lfp"][:, :, list(metadata.selected_electrode_positions), :]
                if not np.isfinite(values).all():
                    raise ValueError("fixed selected LFP is nonfinite; no redraw or trial dropping")
                lfp = BandwiseLFP(region, metadata.design, metadata.selected_electrode_ids, BANDS, _readonly(values))
            if outcomes:
                _check(self._check)
                values = group["spikes"][()]
                if not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
                    raise ValueError("native counts must remain finite nonnegative integers without filtering")
                counts = BoundActivity(metadata.design, _readonly(values))
            return metadata, lfp, counts


class CandidateDevelopmentViews(_DevelopmentViews):
    def training(self, session, region):
        _, lfp, counts = self._read(session, region, "training", features=True, outcomes=True)
        return lfp, counts

    def features(self, session, region, role):
        if role not in ("calibration", "evaluation"):
            raise PermissionError("candidate nontraining access exposes calibration/evaluation features only")
        return self._read(session, region, role, features=True)[1]

    def metadata(self, session, region, role):
        return self._read(session, region, role)[0]


class DevelopmentOutcomeViews(_DevelopmentViews):
    def __init__(self, path, *, role, check):
        if role not in ("calibration", "evaluation"):
            raise PermissionError("an outcome reader must bind exactly calibration or evaluation")
        self._role = role
        super().__init__(path, check=check)

    @property
    def role(self):
        return self._role

    def targets(self, session, region):
        return self._read(session, region, self.role, outcomes=True)[2]

    def metadata(self, session, region):
        return self._read(session, region, self.role)[0]
