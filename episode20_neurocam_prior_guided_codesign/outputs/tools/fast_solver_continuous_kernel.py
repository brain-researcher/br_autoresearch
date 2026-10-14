"""Exact-duration adapters for the continuous EP20 driver; no model changes."""
from collections import OrderedDict
from collections.abc import MutableMapping
from types import MethodType

import numpy as np

from fast_solver_engine import JMAX, _selection_key
from fast_solver_multirate import MultirateSolver
from fast_solver_node_weights import NodeWeights
from fast_solver_phi import shift_monomials


def array_storage(value):
    """Reachable numeric backing allocations, including parents of array views."""
    owners, seen = {}, set()

    def visit(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            owner = obj
            while getattr(owner, "base", None) is not None:
                owner = owner.base
            size = getattr(owner, "nbytes", None)
            if size is None:
                size = memoryview(owner).nbytes
            owners[id(owner)] = (owner, int(size))
        elif isinstance(obj, dict):
            for item in obj.values():
                visit(item)
        elif isinstance(obj, (tuple, list)):
            for item in obj:
                visit(item)
        elif hasattr(obj, "__dict__"):
            visit(vars(obj))

    visit(value)
    return owners


class BoundedWeightCache:
    """One LRU budget across weight scopes; shared backing allocations count once.

    The budget covers retained numeric allocations, not Python bookkeeping or
    temporary construction memory. The driver's RSS limit covers the latter.
    Oversized values are usable by the caller but are not retained.
    """
    def __init__(self, max_bytes):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("positive integer cache byte limit required")
        self.max_bytes = max_bytes
        self.entries, self.owners, self.counts = OrderedDict(), {}, {}
        self.bytes = self.peak_bytes = self.hits = self.misses = self.evictions = 0

    def scope(self, name):
        return _CacheScope(self, name)

    def remove(self, key):
        _, allocations = self.entries.pop(key)
        self.counts[key[0]] -= 1
        for identity in allocations:
            owner, size, count = self.owners[identity]
            if count == 1:
                self.bytes -= size
                del self.owners[identity]
            else:
                self.owners[identity] = (owner, size, count - 1)

    def put(self, key, value):
        if key in self.entries:
            self.remove(key)
        allocations = array_storage(value)
        if sum(size for _, size in allocations.values()) > self.max_bytes:
            return
        additional = lambda: sum(size for identity, (_, size) in allocations.items()
                                 if identity not in self.owners)
        while self.entries and self.bytes + additional() > self.max_bytes:
            self.remove(next(iter(self.entries)))
            self.evictions += 1
        for identity, (owner, size) in allocations.items():
            if identity in self.owners:
                old, old_size, count = self.owners[identity]
                self.owners[identity] = (old, old_size, count + 1)
            else:
                self.owners[identity] = (owner, size, 1)
                self.bytes += size
        self.entries[key] = (value, tuple(allocations))
        self.counts[key[0]] = self.counts.get(key[0], 0) + 1
        self.peak_bytes = max(self.peak_bytes, self.bytes)

    def report(self):
        return dict(limit_bytes=self.max_bytes, retained_numeric_bytes=self.bytes,
                    peak_retained_numeric_bytes=self.peak_bytes, entries=len(self.entries),
                    hits=self.hits, misses=self.misses, evictions=self.evictions)


class _CacheScope(MutableMapping):
    def __init__(self, cache, name):
        self.cache, self.name = cache, name

    def __getitem__(self, key):
        full = (self.name, key)
        try:
            value = self.cache.entries[full][0]
        except KeyError:
            self.cache.misses += 1
            raise
        self.cache.entries.move_to_end(full)
        self.cache.hits += 1
        return value

    def __setitem__(self, key, value):
        self.cache.put((self.name, key), value)

    def __delitem__(self, key):
        self.cache.remove((self.name, key))

    def __iter__(self):
        return iter([key for name, key in self.cache.entries if name == self.name])

    def __len__(self):
        return self.cache.counts.get(self.name, 0)


class ContinuousMultirateSolver(MultirateSolver):
    _audit_optimize_tones = True

    def _audit_unit_response(self, force, node_force):
        # Keep modes separate and retain the existing phi0*coupling products;
        # only the polynomial reduction uses batched matrix multiplication.
        return (force.transpose(1, 0, 2) @ node_force).transpose(1, 0, 2)

    def _audit_weights(self, lam, h, nodes):
        key = (id(lam), self._key(h), JMAX, tuple(map(float, nodes)))
        cached = self._audit_weights_cache.get(key)
        if cached is not None:
            self.stats["audit_weight_cache_hits"] += 1
            return cached[1]
        self.stats["audit_weight_cache_misses"] += 1
        weights = NodeWeights(lam, h, JMAX, nodes)
        # Modal rates are fixed after construction. Retain their array so its
        # identity cannot be reused while this entry survives; the shared
        # cache accounts for its backing allocation along with the weights.
        self._audit_weights_cache[key] = (lam, weights)
        return weights

    def __init__(self, *args, cache_bytes, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_cache = BoundedWeightCache(cache_bytes)
        self._a_weights = self.weight_cache.scope("fine")
        self._fourier_unit = self.weight_cache.scope("fourier_unit")
        self._fourier_a = self.weight_cache.scope("fourier_active")
        self._audit_weights_cache = self.weight_cache.scope("audit")
        self.stats.update(audit_weight_cache_hits=0, audit_weight_cache_misses=0)
        self._prepared_blocks = set()
        self._down_selections = {}
        for block in self.base.unit_blocks:
            self.prepare_block(block)

    def _key(self, h):
        return float(h).hex()

    def prepare_block(self, block):
        if id(block) in self._prepared_blocks:
            return
        self._prepared_blocks.add(id(block))
        scope = self.weight_cache.scope(("modal", id(block)))
        block._weights = scope

        def weights(owner, key, h, nodes, nodes_key):
            identity = (float(h).hex(), JMAX, nodes_key, tuple(map(float, nodes)))
            value = scope.get(identity)
            if value is None:
                value = NodeWeights(owner.lam, h, JMAX, nodes)
                scope[identity] = value
            return value

        block.weights = MethodType(weights, block)

    def down_matrices(self, selection):
        identity = None if selection is None else tuple(sorted((k, _selection_key(v)) for k, v in selection.items()))
        if identity in self._down_selections:
            return self._down_selections[identity]
        down = self.base._down_matrices(selection)
        # One owner per selected circuit, even if the base's small LRU clears.
        # Its lifetime also keeps every id-keyed modal weight tied to its basis.
        self._down_selections[identity] = down
        down["weights"] = self.weight_cache.scope(("downstream", id(down)))
        return down

    def coarse_piece(self, entry, t0, t1):
        """Reuse the existing autonomous coarse equations on one actual interval."""
        h = float(t1) - float(t0)
        if not np.isfinite(h) or h <= 0:
            raise ValueError("coarse interval must have finite positive duration")
        self.prepare_block(entry["block"])
        dwell, fractions, count = self.dwell, self.fractions, self.stats["dwells"]
        try:
            self.dwell, self.fractions = h, (0.0, 1.0)
            record, = super().coarse_dwell(entry, float(t0))
        finally:
            self.dwell, self.fractions = dwell, fractions
            self.stats["dwells"] = count  # The driver counts physical dwells once.
        record.update(t0=float(t0), t1=float(t1))
        return record

    def fine_piece(self, entry, record, u0, u1):
        """Fine active advance with exact physical duration and unrounded restriction."""
        if not record["t0"] <= u0 < u1 <= record["t1"]:
            raise ValueError("fine interval must lie inside its coarse interval")
        h = float(u1) - float(u0)
        offset = (float(u0) - record["t0"]) / record["h"]
        scale = h / record["h"]
        base, block = self.base, entry["block"]
        key = (id(block), h.hex(), JMAX, tuple(map(float, base.nodes)))
        weights = self._a_weights.get(key)
        if weights is None:
            weights = NodeWeights(block.lam, h, JMAX, base.nodes)
            self._a_weights[key] = weights
        whole = u0 == record["t0"] and u1 == record["t1"]
        coupling = (record["a_coupling"] if whole else
                    shift_monomials(record["a_coupling"], offset, scale))
        forcing = block.phi[entry["port_pos"]].T @ coupling
        initial = base.a_y
        values = (weights.node_exp * initial[:, None]
                  + np.einsum("kj,kji->ki", forcing, weights.node_force))
        sources = block.phi[entry["source_pos"]] @ values
        base.a_y = weights.end_exp * initial + np.einsum("kj,kj->k", forcing, weights.end_force)
        self.stats["fine_pieces"] += 1
        return sources @ base.vinv.T, h

    def downstream_exact_piece(self, down, h, t0, homogeneous, row):
        # The inherited downstream cache is local to the selected matrices;
        # its key additionally identifies the *actual* duration exactly.
        return self.downstream_piece(down, h, ("continuous", float(h).hex()), t0, homogeneous, row)


def dwell_spans(window, template):
    """One linear-size split map; no repeated scan of the full split collection."""
    actions = window.hold_action[window.split_hold]
    if template.actions % 25 or np.any(np.diff(actions) < 0):
        raise ValueError("selected recipe requires ordered whole 25-action dwells")
    boundaries = np.arange(0, template.actions + 1, 25)
    splits = np.searchsorted(actions, boundaries)
    for first, last, lo, hi in zip(boundaries[:-1], boundaries[1:], splits[:-1], splits[1:]):
        row = int(template.action_row[first])
        if lo == hi or np.any(template.action_row[first:last] != row):
            raise ValueError("one selected row is required within each physical dwell")
        yield int(first), int(last), int(lo), int(hi), row


def coarse_bounds(start, end, fractions, extra=()):
    values = {float(start), float(end)}
    values.update(float(start + (end - start) * f) for f in fractions if 0 < f < 1)
    values.update(float(t) for t in extra if start < t < end)
    result = np.array(sorted(values))
    if np.any(np.diff(result) <= 0) or not np.all(np.isfinite(result)):
        raise ValueError("coarse grid must be finite and strictly ordered")
    return result
