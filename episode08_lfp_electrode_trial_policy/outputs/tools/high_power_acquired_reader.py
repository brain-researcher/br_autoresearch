"""Native high-power-only trajectories; no LMP numerical reads."""
import h5py
import numpy as np

from prepare_development_trial_roles import DEVELOPMENT
from source_support import field, mclass

CHECKPOINTS = (32, 48, 64, 80, 96)
BANDS = ((100., 200.), (200., 400.))


def balanced_ids(role, total):
    if total not in CHECKPOINTS:
        raise ValueError("Unselected acquisition checkpoint")
    return [t for pool in role["fitting_pool_by_direction"] for t in pool[:total//8]]


class Exposure:
    """Source-read sum and unique logical source elements are distinct."""
    def __init__(self, report):
        self.report = report
        self.lfp_seen, self.target_seen = set(), set()

    def neural_read(self, filename, trial, channels, lfp, spikes, evaluation=False):
        if lfp.shape != (10, 2*len(channels)) or spikes.ndim != 2 or len(spikes) != 10:
            raise ValueError("Exposure must count actual ten-bin two-band arrays")
        self.report["logical_neural_bytes"] += int(lfp.nbytes+spikes.nbytes)
        self.report["evaluation_trials_read" if evaluation else "fitting_trial_read_calls"] += 1
        for channel in channels:
            key = filename, trial, channel
            if key not in self.lfp_seen:
                self.lfp_seen.add(key)
                self.report["unique_logical_neural_bytes"] += int(20*lfp.dtype.itemsize)
        key = filename, trial
        if key not in self.target_seen:
            self.target_seen.add(key)
            self.report["unique_logical_neural_bytes"] += int(spikes.nbytes)


def read_trials(path, source, role, records, ids, channels, check, exposure, reference=None, evaluation=False):
    """Trusted internal slice reader; trajectory object validates phase first."""
    if (path.name not in DEVELOPMENT or path.name != source["filename"] or path.name != role["filename"]
            or source["session_role"] != "development" or role["session_role"] != "development"):
        raise ValueError("Development source identity required")
    allowed = set(role["evaluation_trials"] if evaluation else role["nested_training_sets"]["96"]["training_trials"])
    if len(ids) != len(set(ids)) or not set(ids) <= allowed:
        raise ValueError("Requested identities exceed the exact permitted role view")
    if (not evaluation and set(ids) & set(role["evaluation_trials"])) or not set(channels) <= set(source["common_physical_channels"]):
        raise ValueError("Closed evaluation identity or unavailable electrode requested")
    if len(channels) != len(set(channels)) or not channels:
        raise ValueError("Unique physical electrodes required")
    x, y = [], []
    ref_guide, ref_units = reference if reference is not None else (None, None)
    with h5py.File(path, "r") as handle:
        groups = [handle[name] for name in handle if not name.startswith("#") and isinstance(handle[name], h5py.Group) and mclass(handle[name]) == "struct"]
        if len(groups) != 1:
            raise ValueError("Single native trial struct required")
        group = groups[0]
        for trial in ids:
            check()
            guide = field(handle, group, "M1_lfp_guide", trial)[...].T
            units = field(handle, group, "M1_unit_guide", trial)[...].T
            if (guide.ndim != 2 or guide.shape[1] != 3 or units.ndim != 2 or units.shape[1] != 2
                    or len(units) < 1 or not np.isfinite(units).all() or len(np.unique(units, axis=0)) != len(units)):
                raise ValueError("Invalid guide identity")
            if ref_guide is None:
                ref_guide, ref_units = guide.copy(), units.copy()
            elif not np.array_equal(guide, ref_guide) or not np.array_equal(units, ref_units):
                raise ValueError("Trial electrode/population identity changed")
            selected = []
            for channel in channels:
                for band in BANDS:
                    match = np.flatnonzero((guide[:, 0] == channel) & (guide[:, 1] == band[0]) & (guide[:, 2] == band[1]))
                    if len(match) != 1:
                        raise ValueError("Selected electrode/band absent or duplicated")
                    selected.append(int(match[0]))
            selected_sorted = np.sort(selected)
            undo = np.searchsorted(selected_sorted, selected)
            info = records[trial]
            start, end = info["window_start_zero_based"], info["window_end_exclusive"]
            lfp_node, spike_node = field(handle, group, "M1_lfp", trial), field(handle, group, "M1_spikes", trial)
            if (end-start != 10 or list(reversed(lfp_node.shape)) != info["array_shapes"]["M1_lfp"]
                    or list(reversed(spike_node.shape)) != info["array_shapes"]["M1_spikes"]
                    or lfp_node.shape[0] != len(guide) or spike_node.shape[0] != len(units)):
                raise ValueError("Frozen window/source shape disagrees")
            # These are the only neural deserialization expressions.
            lfp = lfp_node[selected_sorted, start:end].T[:, undo]
            spikes = spike_node[:, start:end].T
            exposure.neural_read(path.name, trial, channels, lfp, spikes, evaluation)
            if not np.isfinite(lfp).all() or not np.isfinite(spikes).all():
                raise ValueError("Nonfinite observation; no exclusions")
            x.append(np.asarray(lfp, dtype=float).mean(axis=0))
            y.append(np.asarray(spikes, dtype=float).mean(axis=0))
    return np.stack(x), np.stack(y), (ref_guide, ref_units)


class Trajectory:
    """Chosen set is immutable; all post-initial data are selected before read."""
    def __init__(self, path, source, role, records, channels, check, exposure):
        self.path, self.source, self.role, self.records = path, source, role, records
        self.channels = tuple(channels)
        self.check, self.exposure = check, exposure
        self.rows, self.targets, self.guide = {}, {}, None
        self.current = 0

    def initialize_random(self):
        if self.current:
            raise ValueError("Initializer already consumed")
        ids = balanced_ids(self.role, 32)
        x, y, self.guide = read_trials(self.path, self.source, self.role, self.records, ids, list(self.channels), self.check, self.exposure)
        self.rows, self.targets = dict(zip(ids, x)), dict(zip(ids, y))
        self.current = 32

    def initialize_selected(self, x, y, all_channels, guide):
        if self.current or list(all_channels) != self.source["common_physical_channels"]:
            raise ValueError("Only the qualified32-row all96initializer may seed selected trajectory")
        ids = balanced_ids(self.role, 32)
        if x.shape != (32, 192) or len(y) != 32:
            raise ValueError("Exact initial arrays required")
        pos = {channel: i for i, channel in enumerate(all_channels)}
        cols = [2*pos[c]+band for c in self.channels for band in range(2)]
        self.rows, self.targets = dict(zip(ids, x[:, cols].copy())), dict(zip(ids, y.copy()))
        self.guide, self.current = guide, 32

    def acquire_to(self, total):
        if total != self.current+16 or total not in CHECKPOINTS:
            raise ValueError("Only the next balanced checkpoint may be acquired")
        ids = [t for t in balanced_ids(self.role, total) if t not in self.rows]
        if len(ids) != 16:
            raise ValueError("Expected exactly16new acquired trials")
        # Immutable self.channels is the only post-initial neural input view.
        x, y, self.guide = read_trials(self.path, self.source, self.role, self.records, ids, list(self.channels), self.check, self.exposure, self.guide)
        self.rows.update(zip(ids, x))
        self.targets.update(zip(ids, y))
        self.current = total

    def observed(self):
        ids = balanced_ids(self.role, self.current)
        if set(ids) != set(self.rows) or set(ids) != set(self.targets):
            raise ValueError("Observed history differs from acquired checkpoint")
        return np.stack([self.rows[t] for t in ids]), np.stack([self.targets[t] for t in ids]), ids
