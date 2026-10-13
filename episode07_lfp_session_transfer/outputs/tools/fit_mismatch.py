"""Fixed whole-trial LFP mismatch controls on one complete EP07 fitting panel.

Only the LFP trial axis is permuted. Counts, directions, recipient trial IDs,
neuron/electrode guides and native time axes remain unchanged. Each returned
pack has an explicit parallel donor-ID tuple: row i contains LFP originally
belonging to that donor, not necessarily the recipient trial_ids[i]. These
are declared control inputs, never genuine re-paired observations.

The caller supplies all ten immutable original DayTrials packs for ONE animal
and target: three history sources plus calibration, and today calibration,
in each region. Shared trial IDs/directions bind before any neural values are
read. Original packs undergo the normal native validation; calibration values
must also be identical across streams within each region. No data are loaded,
models fitted or scores computed; payload admission remains caller-owned.

Within each direction, numeric-suffix-sorted trial IDs receive one uniform
nonzero cyclic shift. Earlier-source seed is 7007101; calibration seed is
7007102. SHA256 of compact ASCII JSON [seed,animal,original_day,direction]
supplies eight little-endian uint32 SeedSequence words for a local PCG64.
The key excludes region, stream and target-day context: the same original
fitting group gets the same donors wherever it is used. Neither Python hash
nor global NumPy RNG state is used. There is no redraw, tuning or seed override.

This is one fixed bounded disruption control, NOT a uniform draw from all
derangements or the registered held-out mismatch bank. Means and marginal
distributions within direction are preserved; a score loss is not guaranteed.
The returned metadata do not authenticate arbitrary arrays or grant access.
"""
from dataclasses import dataclass, replace
import hashlib
import json

import numpy as np

from native_trials import (DayTrials, EARLIER, NEW, _check, _directions,
                           _offsets, _snapshot, _trial_ids, _validate_day)


_SEEDS = {"earlier": 7007101, "calibration": 7007102}
_REGIONS = ("M1", "PMd")
_STREAMS = ("history_assisted", "today_only")


@dataclass(frozen=True)
class FitMismatch:
    scope: str
    seed: int
    packs: tuple[DayTrials, ...]
    lfp_source_trial_ids: tuple[tuple[str, ...], ...]


def _roster(packs, check):
    """Bind the entire ten-pack metadata roster before neural-array access."""
    if (not isinstance(packs, tuple) or len(packs) != 10
            or any(not isinstance(day, DayTrials) for day in packs)):
        raise ValueError("one complete ten-pack paired fitting tuple is required")
    animal, target = packs[0].animal, packs[0].target_day
    if animal not in NEW or target not in NEW[animal]:
        raise ValueError("one permitted animal and target day is required")
    expected = {(region, stream, date)
        for region in _REGIONS for stream in _STREAMS
        for date in (EARLIER[animal] + (target,) if stream == "history_assisted" else (target,))}
    seen, groups = {}, {}
    for day in packs:
        _check(check)
        if day.view_role != "fit":
            raise PermissionError("mismatch controls accept original fitting packs only")
        key = day.region, day.stream, day.day
        if ((day.animal, day.target_day) != (animal, target)
                or key not in expected or key in seen):
            raise ValueError("mixed, duplicate or incomplete paired fitting scope")
        if day.role != ("calibration" if day.day == target else "training"):
            raise PermissionError("fitting day and role must retain their original binding")
        _trial_ids(day.trial_ids)
        directions, _ = _directions(day.directions, len(day.trial_ids), day.role)
        offsets = np.asarray(day.bin_offsets)
        if offsets.dtype.kind not in "iu" or not np.array_equal(offsets, _offsets(day.region)):
            raise ValueError("native time offsets must retain their region binding")
        signature = day.trial_ids, tuple(int(value) for value in directions)
        if day.day in groups and groups[day.day] != signature:
            raise ValueError("regions and streams must share exact original trial IDs and directions")
        groups[day.day] = signature
        seen[key] = day
    if set(seen) != expected:
        raise ValueError("all and only the ten paired fitting packs are required")
    return animal, target, seen, groups


def _original_values(packs, by_key, target, check):
    for day in packs:
        _validate_day(day, check)
        for name in ("directions", "bin_offsets", "lfp", "lfp_guide", "spikes", "unit_guide"):
            if np.asarray(getattr(day, name)).flags.writeable:
                raise ValueError("original fitting arrays must be immutable adapter snapshots")
    # Metadata alone must not let the two streams see different calibration.
    for region in _REGIONS:
        history, today = (by_key[(region, stream, target)] for stream in _STREAMS)
        for name in ("lfp", "lfp_guide", "spikes", "unit_guide"):
            _check(check)
            left, right = np.asarray(getattr(history, name)), np.asarray(getattr(today, name))
            if left.dtype != right.dtype or not np.array_equal(left, right):
                raise ValueError("both streams must share identical original calibration values and axes")


def _donors(seed, animal, original_day, ids, directions, check):
    donors = np.arange(len(ids), dtype=np.int64)
    for direction in range(8):
        _check(check)
        rows = sorted((index for index, value in enumerate(directions) if value == direction),
                      key=lambda index: int(ids[index][6:]))
        if len(rows) < 2:
            raise ValueError("a direction with fewer than two trials cannot be deranged")
        encoded = json.dumps([seed, animal, original_day, direction],
                             ensure_ascii=True, separators=(",", ":")).encode("ascii")
        digest = hashlib.sha256(encoded).digest()
        words = tuple(int.from_bytes(digest[offset:offset + 4], "little")
                      for offset in range(0, 32, 4))
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(words)))
        shift = int(rng.integers(1, len(rows)))
        for position, recipient in enumerate(rows):
            donors[recipient] = rows[(position + shift) % len(rows)]
    return _snapshot(donors)


def derange_fit_lfp(packs, *, scope, check):
    """Return the same ordered packs plus explicit donor identities per pack.

    ``scope='earlier'`` perturbs only history's three training days, while
    ``scope='calibration'`` perturbs the target calibration in both streams.
    Unaffected packs are reused unchanged with identity donor mappings. The
    transformed LFP is an immutable native-dtype snapshot. All other fields
    retain their original immutable objects; no neural axes are matched
    across days. This helper does not extend the normal fitting roster.
    """
    _check(check)
    if not isinstance(scope, str) or scope not in _SEEDS:
        raise ValueError("mismatch scope must be earlier or calibration")
    animal, target, by_key, groups = _roster(packs, check)
    _original_values(packs, by_key, target, check)
    seed = _SEEDS[scope]
    affected = EARLIER[animal] if scope == "earlier" else (target,)
    permutations = {date: _donors(seed, animal, date, *groups[date], check)
                    for date in affected}
    result, mappings = [], []
    for day in packs:
        _check(check)
        donors = permutations.get(day.day)
        if donors is None:
            result.append(day)
            mappings.append(day.trial_ids)
        else:
            result.append(replace(day, lfp=_snapshot(day.lfp[donors, :, :])))
            mappings.append(tuple(day.trial_ids[int(index)] for index in donors))
    _check(check)
    return FitMismatch(scope, seed, tuple(result), tuple(mappings))
