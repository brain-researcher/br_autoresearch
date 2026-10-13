"""Dormant secondary Finding A summaries of a supplied original point panel.

No reader, fitting, grant, CLI, job launch or change to source-primary results.
The caller supplies the completed point PanelResult and its resource check.
Only the original 224-image identity order is accepted: a primary image draw
must not silently become a nested image-by-dyad bootstrap.

Target-incoming sensitivity averages four sources per target. The secondary
source-style bootstrap independently proposes source then target, rejects
self-pairs, and retains duplicate edges until twenty rows have been accepted.
Within-source values follow each edge's SOURCE for this descriptive weighting;
they are never target baselines or independent edge observations. Contrasts
come from the previously support-qualified point table, never subtraction of
unmatched absolute means.

The adopted local RandomState(42) schedule is shared across all endpoints and
layers. It does not reproduce the source script's historically unspecified
global/multiprocessing seed. Exactly 1,000 draws and a 1,000-proposal cap per
draw are fixed. Failed draws are recorded, never replaced. These pointwise
percentile intervals neither replace source-participant primary inference nor
support population, independence, equivalence or non-inferiority claims.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np

import finding_a_inference as inference
from schedule import PARTICIPANTS, PAIR_ORDER


LAYERS = inference.LAYERS
ENDPOINTS = inference.ENDPOINTS
DYADIC_DRAWS = 1000
PROPOSAL_CAP = 1000
SEED = 42


@dataclass(frozen=True)
class IncomingMetric:
    layer_target_means: object
    target_means: object
    layer_group_means: object
    group_means: object


@dataclass(frozen=True)
class IncomingResult:
    participants: tuple[str, ...]
    layers: tuple[str, ...]
    image_ids: tuple[str, ...]
    pattern: IncomingMetric
    profile: IncomingMetric


@dataclass(frozen=True)
class DyadicMetric:
    pair_values: object
    layer_means: object
    group_means: object
    layer_intervals: object
    group_intervals: object
    layer_reasons: object
    group_reasons: object


@dataclass(frozen=True)
class DyadicBootstrapResult:
    participants: tuple[str, ...]
    layers: tuple[str, ...]
    image_ids: tuple[str, ...]
    pair_order: tuple[tuple[str, str], ...]
    draw_indices: np.ndarray
    proposal_counts: np.ndarray
    draw_available: np.ndarray
    pattern: DyadicMetric
    profile: DyadicMetric


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _readonly(value):
    value.setflags(write=False)
    return value


def _freeze(values):
    return MappingProxyType({key: _readonly(value) if isinstance(value, np.ndarray) else value
                             for key, value in values.items()})


def _point_tables(panel, check):
    """Snapshot only compact layer slots; original fitted metric payloads stay outside."""
    check()
    _require(isinstance(panel, inference.PanelResult)
             and panel.participants == PARTICIPANTS and panel.layers == LAYERS,
             "original_fixed_five_participant_nineteen_layer_panel_required")
    ids = panel.image_ids
    _require(isinstance(ids, tuple) and len(ids) == 224
             and all(isinstance(value, str) and bool(value) for value in ids)
             and len(set(ids)) == 224, "original_224_unique_image_IDs_required")
    draw = panel.image_indices
    _require(isinstance(draw, np.ndarray) and draw.dtype == np.dtype("int64")
             and draw.shape == (224,) and not draw.flags.writeable
             and np.array_equal(draw, np.arange(224, dtype=np.int64)),
             "original_identity_image_positions_required_no_nested_image_bootstrap")
    tables = {}
    for metric in ("pattern", "profile"):
        values = getattr(panel, metric)
        _require(isinstance(values, inference.MetricPanel)
                 and set(values.layer_means) == set(ENDPOINTS),
                 "complete_endpoint_layer_slot_tables_required")
        # Keep caller callbacks from replacing mapping entries halfway through
        # this small snapshot. Finished point arrays themselves are readonly.
        supplied = dict(values.layer_means)
        tables[metric] = {}
        for endpoint in ENDPOINTS:
            check()
            value = supplied[endpoint]
            shape = (19, 5) if endpoint == "within_source" else (19, 5, 5)
            _require(isinstance(value, np.ndarray) and value.dtype == np.dtype("float64")
                     and value.shape == shape and not value.flags.writeable
                     and not bool(np.isinf(value).any()),
                     "readonly_complete_float64_layer_slots_with_explicit_missing_values_required")
            if endpoint != "within_source":
                _require(bool(np.isnan(value[:, np.arange(5), np.arange(5)]).all()),
                         "inapplicable_self_pair_diagonal_must_remain_blank")
            tables[metric][endpoint] = _readonly(value.copy())
    return tables


def _mean_required(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values) or not bool(np.isfinite(values).all()):
        return np.nan
    return math.fsum(float(value) for value in values) / len(values)


def _as_group(value):
    return float(value) if np.isfinite(value) else None


def _incoming_metric(tables, check):
    layer_targets, targets, layer_groups, groups = {}, {}, {}, {}
    for endpoint in ENDPOINTS:
        check()
        values = tables[endpoint]
        incoming = np.full((19, 5), np.nan, dtype=np.float64)
        for target in range(5):
            sources = [source for source in range(5) if source != target]
            for layer in range(19):
                cells = (values[layer, sources] if endpoint == "within_source"
                         else values[layer, sources, target])
                incoming[layer, target] = _mean_required(cells)
        target_means = np.asarray([_mean_required(incoming[:, target]) for target in range(5)])
        layer_targets[endpoint] = incoming
        targets[endpoint] = target_means
        layer_groups[endpoint] = np.asarray([_mean_required(row) for row in incoming])
        groups[endpoint] = _as_group(_mean_required(target_means))
    return IncomingMetric(_freeze(layer_targets), _freeze(targets),
                          _freeze(layer_groups), _freeze(groups))


def target_incoming(panel, *, check):
    """Equal four-source incoming means, with all nineteen layers required.

    No source or layer is dropped to recover an otherwise unavailable target.
    Unrelated complete targets/endpoints remain visible. This result leaves
    the supplied source-primary panel unchanged.
    """
    tables = _point_tables(panel, check)
    return IncomingResult(PARTICIPANTS, LAYERS, panel.image_ids,
        _incoming_metric(tables["pattern"], check), _incoming_metric(tables["profile"], check))


def _rng():
    """Fixed local generator; private seam for a forced cap-failure fixture."""
    return np.random.RandomState(SEED)


def _draw_schedule(rng, check):
    lookup = {(PARTICIPANTS.index(source), PARTICIPANTS.index(target)): index
              for index, (source, target) in enumerate(PAIR_ORDER)}
    draws = np.full((DYADIC_DRAWS, 20), -1, dtype=np.int64)
    proposals = np.empty(DYADIC_DRAWS, dtype=np.int64)
    available = np.zeros(DYADIC_DRAWS, dtype=bool)
    for draw in range(DYADIC_DRAWS):
        check()
        accepted = 0
        for proposal in range(1, PROPOSAL_CAP + 1):
            if proposal % 64 == 0:
                check()
            # Match the source's two sequential choice calls. Batched source/
            # target proposals or overdrawing past twenty changes RNG use.
            source = int(rng.choice(5, 1, replace=True)[0])
            target = int(rng.choice(5, 1, replace=True)[0])
            if source == target:
                continue
            draws[draw, accepted] = lookup[source, target]
            accepted += 1
            if accepted == 20:
                available[draw] = True
                break
        proposals[draw] = proposal
    check()
    return _readonly(draws), _readonly(proposals), _readonly(available)


def _dyadic_metric(tables, draws, available, check):
    pair_values, layer_means, group_means = {}, {}, {}
    layer_intervals, group_intervals, layer_reasons, group_reasons = {}, {}, {}, {}
    source_positions = [PARTICIPANTS.index(source) for source, _ in PAIR_ORDER]
    target_positions = [PARTICIPANTS.index(target) for _, target in PAIR_ORDER]
    for endpoint in ENDPOINTS:
        check()
        values = tables[endpoint]
        original = (values[:, source_positions] if endpoint == "within_source"
                    else values[:, source_positions, target_positions])
        # Exactly one original value per directed dyad per layer before any
        # resampling. Within-source expansion is descriptive edge weighting.
        original = np.array(original, dtype=np.float64, copy=True)
        complete = np.isfinite(original).all(axis=1)
        point_layers = np.asarray([_mean_required(row) for row in original])
        point_group = _as_group(_mean_required(point_layers))
        layer_samples = np.full((DYADIC_DRAWS, 19), np.nan, dtype=np.float64)
        for layer in range(19):
            check()
            if complete[layer]:
                # Only complete ORIGINAL tables qualify. A draw which happens
                # to omit a missing edge cannot make an endpoint evaluable.
                for draw in np.flatnonzero(available):
                    layer_samples[draw, layer] = _mean_required(original[layer, draws[draw]])
        group_samples = np.asarray([_mean_required(row) for row in layer_samples])
        pair_values[endpoint] = original
        layer_means[endpoint] = point_layers
        group_means[endpoint] = point_group
        layer_intervals[endpoint] = inference._interval(point_layers, layer_samples, check)
        group_intervals[endpoint] = inference._interval(point_group, group_samples, check)
        layer_reasons[endpoint] = np.asarray([
            "incomplete_original_pair_table" if not good else
            ("complete_fixed_dyadic_bootstrap" if bool(available.all()) else "proposal_cap_exhausted")
            for good in complete], dtype=object)
        group_reasons[endpoint] = ("required_layer_unavailable" if point_group is None else
            ("complete_fixed_dyadic_bootstrap" if bool(available.all()) else "proposal_cap_exhausted"))
    return DyadicMetric(_freeze(pair_values), _freeze(layer_means), _freeze(group_means),
        MappingProxyType(layer_intervals), MappingProxyType(group_intervals),
        _freeze(layer_reasons), _freeze(group_reasons))


def dyadic_bootstrap(panel, *, check):
    """Secondary source-style descriptive intervals; no primary image resampling.

    Draw indices address the explicit canonical pair_order. Duplicates remain
    duplicate rows. A capped incomplete row retains its accepted prefix and
    -1 unused positions for reporting, but contributes no partial draw value.
    No public seed/count/cap override, retry or available-pair subset exists.
    """
    tables = _point_tables(panel, check)
    draws, proposals, available = _draw_schedule(_rng(), check)
    return DyadicBootstrapResult(PARTICIPANTS, LAYERS, panel.image_ids, PAIR_ORDER,
        draws, proposals, available, _dyadic_metric(tables["pattern"], draws, available, check),
        _dyadic_metric(tables["profile"], draws, available, check))
