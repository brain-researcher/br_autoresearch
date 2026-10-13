"""Trusted chosen-trial source views; policies receive acquired arrays only."""
import h5py
import numpy as np

from acquired_only_reader import balanced_ids
from development_reader import BANDS
from prepare_development_trial_roles import DEVELOPMENT
from source_support import field, mclass


class FittingSource:
    """One bounded development-session handle; no evaluation read entrypoint."""
    def __init__(self, path, source, role, records, check, exposure):
        if (path.name not in DEVELOPMENT or path.name != source["filename"] or path.name != role["filename"]
                or source["session_role"] != "development" or role["session_role"] != "development"):
            raise ValueError("Fixed development session required")
        self.path, self.source, self.role, self.records = path, source, role, records
        self.check, self.exposure = check, exposure
        self.allowed = set(role["nested_training_sets"]["96"]["training_trials"])
        queues = role["fitting_pool_by_direction"]
        if (len(queues) != 8 or any(len(q) < 12 for q in queues)
                or [t for q in queues for t in q[:12]] != role["nested_training_sets"]["96"]["training_trials"]
                or len(self.allowed) != 96 or self.allowed & set(role["evaluation_trials"])):
            raise ValueError("Exact disjoint96trial queues required before source opening")
        self.reference = None

    def __enter__(self):
        self.handle = h5py.File(self.path, "r")
        groups = [self.handle[name] for name in self.handle if not name.startswith("#")
                  and isinstance(self.handle[name], h5py.Group) and mclass(self.handle[name]) == "struct"]
        if len(groups) != 1:
            self.handle.close()
            raise ValueError("Single native trial struct required")
        self.group = groups[0]
        return self

    def __exit__(self, *_):
        self.handle.close()

    def read(self, ids, channels):
        if (not ids or len(ids) != len(set(ids)) or not set(ids) <= self.allowed
                or not channels or len(channels) != len(set(channels))
                or not set(channels) <= set(self.source["common_physical_channels"])):
            raise ValueError("Only named fitting trials and chosen electrodes may be read")
        x, y = [], []
        for trial in ids:
            self.check()
            guide = field(self.handle, self.group, "M1_lfp_guide", trial)[...].T
            units = field(self.handle, self.group, "M1_unit_guide", trial)[...].T
            if (guide.ndim != 2 or guide.shape[1] != 3 or not np.isfinite(guide).all()
                    or units.ndim != 2 or units.shape[1] != 2 or not len(units)
                    or not np.isfinite(units).all() or len(np.unique(units, axis=0)) != len(units)):
                raise ValueError("Finite physical guides required")
            if self.reference is None:
                self.reference = guide.copy(), units.copy()
            elif not np.array_equal(guide, self.reference[0]) or not np.array_equal(units, self.reference[1]):
                raise ValueError("Trial channel or target identity changed")
            selected = []
            for channel in channels:
                for band in BANDS:
                    matching = np.flatnonzero((guide[:, 0] == channel) & (guide[:, 1] == band[0]) & (guide[:, 2] == band[1]))
                    if len(matching) != 1:
                        raise ValueError("Chosen physical electrode/band lacks unique support")
                    selected.append(int(matching[0]))
            sorted_rows = np.sort(selected)
            undo = np.searchsorted(sorted_rows, selected)
            info = self.records[trial]
            start, end = info["window_start_zero_based"], info["window_end_exclusive"]
            lfp_node, spike_node = field(self.handle, self.group, "M1_lfp", trial), field(self.handle, self.group, "M1_spikes", trial)
            if (end-start != 10 or list(reversed(lfp_node.shape)) != info["array_shapes"]["M1_lfp"]
                    or list(reversed(spike_node.shape)) != info["array_shapes"]["M1_spikes"]
                    or lfp_node.shape[0] != len(guide) or spike_node.shape[0] != len(units)):
                raise ValueError("Frozen window/shape differs")
            # Only chosen fitting IDs/channel rows reach these expressions.
            lfp = lfp_node[sorted_rows, start:end].T[:, undo]
            spikes = spike_node[:, start:end].T
            self.exposure.neural_read(self.path.name, trial, channels, lfp, spikes)
            if not np.isfinite(lfp).all() or not np.isfinite(spikes).all():
                raise ValueError("Nonfinite acquired observation; no row exclusion")
            x.append(np.asarray(lfp, dtype=float).mean(axis=0))
            y.append(np.asarray(spikes, dtype=float).mean(axis=0))
        return np.stack(x), np.stack(y)


class AllocationTrajectory:
    def __init__(self, source, channels, initial_x, initial_y):
        self.source, self.channels = source, tuple(channels)
        self.queues = tuple(tuple(q[:12]) for q in source.role["fitting_pool_by_direction"])
        ids = balanced_ids(source.role, 32)
        if initial_x.shape != (32, 3*len(channels)) or len(initial_y) != 32:
            raise ValueError("The common32trial initializer must match chosen electrodes")
        self.x = dict(zip(ids, np.array(initial_x, copy=True)))
        self.y = dict(zip(ids, np.array(initial_y, copy=True)))
        self.counts = np.full(8, 4, dtype=int)

    def acquire(self, direction):
        if type(direction) is not int or direction not in range(8) or self.counts[direction] >= 12:
            raise ValueError("A legal unexhausted direction must be chosen before reading")
        trial = self.queues[direction][self.counts[direction]]
        if trial in self.x or trial in self.y:
            raise ValueError("Next queue observation was already acquired")
        x, y = self.source.read([trial], list(self.channels))
        self.x[trial], self.y[trial] = x[0], y[0]
        self.counts[direction] += 1
        return trial

    def observed(self):
        # Canonical direction/queue order makes identical observed sets produce
        # identical fits regardless of the chronological acquisition order.
        ids = [t for d, q in enumerate(self.queues) for t in q[:self.counts[d]]]
        if set(ids) != set(self.x) or set(ids) != set(self.y):
            raise ValueError("Observation history differs from selected queue prefixes")
        labels = np.array([d for d, q in enumerate(self.queues) for _ in q[:self.counts[d]]])
        return np.stack([self.x[t] for t in ids]), np.stack([self.y[t] for t in ids]), labels, ids
