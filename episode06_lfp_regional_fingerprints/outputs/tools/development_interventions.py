"""Prospective paired within-direction whole-trial LFP mismatch for EP06.

This array-only helper has no reader, spike/count interface, scorer or fitting
authority. Declare the mapping from metadata before giving it any LFP arrays.
For each direction in the caller's explicit order, PCG64 permutes the existing
trial positions and draws a uniform offset in1..n-1. A cyclic shift of that
ordering supplies a bijection with no fixed trial. This random cyclic
construction does not generally sample all possible derangements uniformly.

The planned caller seed is (6006201, animal_index, original_session_position,
role_index), with role indices training0/calibration1/evaluation2. The caller
must fix and bind that seed prospectively; this primitive never infers an
animal, selects an empirical control, or chooses a seed from results. A region
key must not be added: one declared mapping is applied to BOTH physical regions.

Target whole-trial IDs and all native time/electrode/band order remain intact;
only which same-direction trial supplies the entire LFP trajectory changes.
Each region retains its own native neuron and electrode identities. Cross-
region matching concerns session, role, direction, trial and time metadata,
not equality of different arrays' physical axes. No trial is dropped or filled.
"""
from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np

from bandwise_prediction import _lfp_metadata, _lfp_values, _readonly
from component_scoring import _ids


class TrialMismatchUnavailable(ValueError):
    """A registered direction cannot support a complete no-fixed-point map."""


@dataclass(frozen=True)
class TrialMismatchMapping:
    session_id: str
    role: str
    directions: tuple
    trial_ids: tuple
    trial_directions: tuple
    seed: tuple
    source_indices: tuple       # Target position -> source position.
    source_trial_ids: tuple     # Explicit identity corresponding to each source.


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned cancellation/resource check required")
    check()


def _metadata(session_id, role, directions, trial_ids, trial_directions, seed):
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("an explicit session identity is required")
    if role not in ("training", "calibration", "evaluation"):
        raise ValueError("one explicit whole-trial role is required")
    if not all(isinstance(value, tuple) for value in (directions, trial_ids, trial_directions)):
        raise TypeError("ordered immutable trial and direction identities are required")
    directions = _ids(directions, "directions")
    trial_ids = _ids(trial_ids, "trial_ids")
    trial_directions = _ids(trial_directions, "trial_directions", unique=False)
    if (len(directions) != 8 or len(trial_ids) != len(trial_directions)
            or set(trial_directions) != set(directions)):
        raise ValueError("every whole trial must retain exactly one of all eight declared directions")
    if (not isinstance(seed, tuple) or not seed
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                   or value < 0 for value in seed)):
        raise ValueError("the caller must explicitly supply a tuple of nonnegative integer seed words")
    for direction in directions:
        if trial_directions.count(direction) < 2:
            raise TrialMismatchUnavailable("every direction needs at least two whole trials; no trial dropping")
    return directions, trial_ids, trial_directions, tuple(int(value) for value in seed)


def _validate_mapping(mapping):
    if not isinstance(mapping, TrialMismatchMapping):
        raise TypeError("an explicit frozen whole-trial mapping is required")
    _metadata(mapping.session_id, mapping.role, mapping.directions, mapping.trial_ids,
              mapping.trial_directions, mapping.seed)
    count = len(mapping.trial_ids)
    source = mapping.source_indices
    if (not isinstance(source, tuple) or len(source) != count
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in source)
            or tuple(sorted(source)) != tuple(range(count))):
        raise ValueError("the whole-trial mapping must be a complete positional bijection")
    if any(index == supplied for index, supplied in enumerate(source)):
        raise ValueError("the within-direction mismatch may have no fixed trial")
    if any(mapping.trial_directions[index] != mapping.trial_directions[supplied]
           for index, supplied in enumerate(source)):
        raise ValueError("the mismatch must preserve each target trial's registered direction")
    if (not isinstance(mapping.source_trial_ids, tuple)
            or mapping.source_trial_ids != tuple(mapping.trial_ids[index] for index in source)):
        raise ValueError("explicit source trial identities disagree with the declared positional mapping")


def make_within_direction_mapping(*, session_id, role, directions, trial_ids,
                                  trial_directions, seed, check):
    """Declare target->source identity mapping without accessing any arrays."""
    _check(check)
    directions, trial_ids, trial_directions, seed = _metadata(
        session_id, role, directions, trial_ids, trial_directions, seed)
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    source = [-1] * len(trial_ids)
    for direction in directions:
        _check(check)
        positions = tuple(index for index, value in enumerate(trial_directions) if value == direction)
        order = generator.permutation(positions)
        offset = int(generator.integers(1, len(positions)))
        for index, target in enumerate(order):
            source[int(target)] = int(order[(index + offset) % len(order)])
    result = TrialMismatchMapping(session_id, role, directions, trial_ids, trial_directions,
        seed, tuple(source), tuple(trial_ids[index] for index in source))
    _validate_mapping(result)
    _check(check)
    return result


def apply_paired_lfp_mismatch(m1, pmd, *, mapping, check):
    """Apply the same declared whole-trajectory source map to both regions."""
    _check(check)
    _validate_mapping(mapping)
    left, _, left_bands = _lfp_metadata(m1)
    right, _, right_bands = _lfp_metadata(pmd)
    if m1.region != "M1" or pmd.region != "PMd":
        raise ValueError("the paired inputs must be explicitly ordered M1 then PMd")
    common = ("session_id", "role", "directions", "trial_ids", "trial_directions", "time_ids")
    if any(getattr(left, field) != getattr(right, field) for field in common):
        raise ValueError("paired regions must have exact session/role/direction/trial/time metadata")
    if left_bands != right_bands:
        raise ValueError("paired inputs must retain the same explicit band order without hidden reordering")
    if any(getattr(left, field) != getattr(mapping, field) for field in common if field != "time_ids"):
        raise ValueError("the declared mapping does not bind these target whole-trial identities")
    # All mapping and BOTH input metadata are bound before either array access.
    _check(check)
    left_values = _lfp_values(m1)
    _check(check)
    right_values = _lfp_values(pmd)
    indices = np.asarray(mapping.source_indices, dtype=np.int64)
    result = (replace(m1, values=_readonly(left_values[indices, :, :, :])),
              replace(pmd, values=_readonly(right_values[indices, :, :, :])))
    _check(check)
    return result
