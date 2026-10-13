"""Dormant full Finding A point panel and synchronized image bootstrap.

The caller supplies an admitted block provider and resource/cancellation check.
This module reads no files, launches no jobs, fits no models and grants no data
access. It uses every registered layer, all five fixed source participants,
and all four outgoing targets, without an available-case replacement panel.

The primary biological unit is the source participant. Targets are repeated
measures: average four targets equally, then five sources equally. The within-
source baseline is scored once per source/layer. Paired defined supports are
checked before any contrast aggregation; absolute means cannot substitute for
that check. Cross-layer endpoints average all nineteen layer values equally.

Exactly 1,000 synchronized RandomState(42) image draws recompute complete
endpoints, including profile correlations. Pointwise percentile intervals are
conditional on these five participants and the fixed fitted objects, not
population, simultaneous, hypothesis-test or comparability claims. Secondary
source-style dyadic bootstrap and target-incoming sensitivity are separate.

Only the within-source and current content/brain full-layer results are held
by the panel loop. Compact slots preserve means, counts and missingness, not
full image/unit metric payloads. The optional caller-owned sink receives point
LayerScores for later persistence; it owns any added storage/memory. Bootstrap
retains compact draws only (roughly tens of MiB), never 1,000 full feature banks.
It still requires 1,001 full panel passes (855,855 method/layer scores when all
inputs are present). That recomputation cost is unmeasured; bounded memory does
not establish a feasible production runtime or authorize its execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np

import finding_a_metrics as metrics
from schedule import PARTICIPANTS


LAYERS = metrics.LAYERS
METHODS = ("within_source", "content_loss", "brain_loss")
CONTRASTS = ("content_minus_brain", "content_minus_within")
ENDPOINTS = METHODS + CONTRASTS
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 42
_METRICS = ("pattern", "profile")


@dataclass(frozen=True)
class UnavailableLayer:
    """An explicit missing input/model; ordinary provider exceptions propagate."""
    reason: str


@dataclass(frozen=True)
class MetricPanel:
    # Within layer slots: [layer, source]. Other slots: [layer, source, target].
    # Complete 5x5 dyad matrices have NaN diagonals (self-pairs are inapplicable).
    layer_means: object
    layer_defined_counts: object
    layer_undefined_counts: object
    layer_reasons: object
    layer_source_means: object
    layer_group_means: object
    dyad_means: object
    source_means: object
    group_means: object


@dataclass(frozen=True)
class PanelResult:
    image_ids: tuple[str, ...]
    image_indices: np.ndarray
    participants: tuple[str, ...]
    layers: tuple[str, ...]
    pattern: MetricPanel
    profile: MetricPanel


@dataclass(frozen=True)
class Interval:
    # Matching endpoint shapes; group fields are zero-dimensional arrays.
    lower: np.ndarray
    upper: np.ndarray
    available: np.ndarray
    valid_draws: np.ndarray


@dataclass(frozen=True)
class MetricIntervals:
    layer_dyad: object
    layer_source: object
    layer_group: object
    dyad: object
    source: object
    group: object


@dataclass(frozen=True)
class BootstrapResult:
    point: PanelResult
    draw_indices: np.ndarray
    pattern: MetricIntervals
    profile: MetricIntervals


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _readonly(value):
    value.setflags(write=False)
    return value


def _freeze(values):
    return MappingProxyType({name: _readonly(value) if isinstance(value, np.ndarray) else value
                             for name, value in values.items()})


def _image_ids(values, count):
    _require(isinstance(values, tuple) and len(values) == count
             and all(isinstance(value, str) and bool(value) for value in values)
             and len(set(values)) == count, "complete_unique_image_ID_tuple_required")


def _context(training_ids, image_ids, image_indices, check):
    check()
    _image_ids(image_ids, 224)
    _require(set(training_ids) == set(PARTICIPANTS), "all_five_decoder_training_owners_required")
    # Callbacks may retain and change the original mapping. The tuple values
    # are immutable; freeze their five owner bindings before any provider call.
    training_ids = MappingProxyType({owner: training_ids[owner] for owner in PARTICIPANTS})
    for owner in PARTICIPANTS:
        _image_ids(training_ids[owner], 4712)
        _require(not set(image_ids).intersection(training_ids[owner]),
                 "decoder_training_test_image_overlap_forbidden")
    if image_indices is None:
        indices = np.arange(224, dtype=np.int64)
    else:
        _require(isinstance(image_indices, np.ndarray) and image_indices.shape == (224,)
                 and image_indices.dtype.kind in "iu" and not image_indices.flags.writeable
                 and bool((image_indices >= 0).all()) and bool((image_indices < 224).all()),
                 "immutable_224_image_draw_positions_required")
        indices = image_indices.astype(np.int64, copy=True)
    return training_ids, _readonly(indices)


def _empty_metric():
    means, reasons = {}, {}
    for endpoint in ENDPOINTS:
        shape = (len(LAYERS), 5) if endpoint == "within_source" else (len(LAYERS), 5, 5)
        means[endpoint] = np.full(shape, np.nan, dtype=np.float64)
        reasons[endpoint] = np.full(shape, "not_evaluated", dtype=object)
        if endpoint != "within_source":
            for person in range(5):
                reasons[endpoint][:, person, person] = "self_pair_not_applicable"
    defined = {method: np.full(means[method].shape, -1, dtype=np.int64) for method in METHODS}
    undefined = {method: np.full(means[method].shape, -1, dtype=np.int64) for method in METHODS}
    return means, defined, undefined, reasons


def _score(provider, training_ids, image_ids, indices, *, method, source, target, layer,
           check, result_sink):
    check()
    supplied = provider(method=method, source=source, target=target, layer=layer)
    if isinstance(supplied, UnavailableLayer):
        _require(isinstance(supplied.reason, str) and bool(supplied.reason),
                 "explicit_unavailable_layer_reason_required")
        return supplied
    owner = source if method == "within_source" else target
    score = metrics.score_layer(supplied, method=method, source=source, target=target,
        layer=layer, expected_image_ids=image_ids, decoder_training_image_ids=training_ids[owner],
        image_indices=indices, check=check)
    check()
    if result_sink is not None:
        result_sink(score)
        check()
    return score


def _record_method(records, method, position, score):
    for metric in _METRICS:
        means, defined, undefined, reasons = records[metric]
        if isinstance(score, UnavailableLayer):
            reasons[method][position] = score.reason
            continue
        summary = getattr(score, metric)
        means[method][position] = summary.mean if summary.available else np.nan
        defined[method][position] = summary.defined_count
        undefined[method][position] = summary.undefined_count
        reasons[method][position] = "defined_correlations" if summary.available else "all_correlations_undefined"


def _record_contrast(records, endpoint, position, first, second, check):
    if isinstance(first, UnavailableLayer) or isinstance(second, UnavailableLayer):
        for metric in _METRICS:
            records[metric][3][endpoint][position] = "required_method_unavailable"
        return
    contrast = metrics.paired_contrast(first, second, check=check)
    for metric in _METRICS:
        result = getattr(contrast, metric)
        records[metric][0][endpoint][position] = result.difference if result.available else np.nan
        records[metric][3][endpoint][position] = result.reason


def _required_mean(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values) or not bool(np.isfinite(values).all()):
        return np.nan
    return math.fsum(float(value) for value in values) / len(values)


def _reduce(records, check):
    means, defined, undefined, reasons = records
    layer_sources, layer_groups, dyads, sources, groups = {}, {}, {}, {}, {}
    for endpoint in ENDPOINTS:
        check()
        layer_source = np.full((len(LAYERS), 5), np.nan, dtype=np.float64)
        source_values = np.full(5, np.nan, dtype=np.float64)
        if endpoint == "within_source":
            layer_source[:] = means[endpoint]
        else:
            dyad = np.full((5, 5), np.nan, dtype=np.float64)
            for source in range(5):
                targets = [target for target in range(5) if target != source]
                for target in targets:
                    dyad[source, target] = _required_mean(means[endpoint][:, source, target])
                for layer in range(len(LAYERS)):
                    layer_source[layer, source] = _required_mean(means[endpoint][layer, source, targets])
            dyads[endpoint] = dyad
        for source in range(5):
            source_values[source] = _required_mean(layer_source[:, source])
        layer_sources[endpoint] = layer_source
        layer_groups[endpoint] = np.asarray([_required_mean(row) for row in layer_source], dtype=np.float64)
        sources[endpoint] = source_values
        group = _required_mean(source_values)
        groups[endpoint] = float(group) if np.isfinite(group) else None
    return MetricPanel(_freeze(means), _freeze(defined), _freeze(undefined), _freeze(reasons),
        _freeze(layer_sources), _freeze(layer_groups), _freeze(dyads), _freeze(sources), _freeze(groups))


def evaluate_panel(block_provider, decoder_training_image_ids_by_owner, expected_image_ids,
                   image_indices=None, *, check, result_sink=None):
    """Evaluate all 855 method/layer slots, or retain explicitly missing slots.

    block_provider(method=..., source=..., target=..., layer=...) returns an
    iterable of MetricBlocks or UnavailableLayer. Within-source target is None.
    Only an explicit sentinel represents missing scientific inputs; provider,
    scorer, sink and cancellation exceptions propagate without retry. Calls
    reuse the same image draw across every layer, source, target and method.
    """
    training_ids, indices = _context(decoder_training_image_ids_by_owner,
                                    expected_image_ids, image_indices, check)
    _require(callable(block_provider) and (result_sink is None or callable(result_sink)),
             "caller_block_provider_and_optional_result_sink_required")
    records = {metric: _empty_metric() for metric in _METRICS}
    for layer_index, layer in enumerate(LAYERS):
        for source_index, source in enumerate(PARTICIPANTS):
            within = _score(block_provider, training_ids,
                expected_image_ids, indices, method="within_source", source=source,
                target=None, layer=layer, check=check, result_sink=result_sink)
            _record_method(records, "within_source", (layer_index, source_index), within)
            for target_index, target in enumerate(PARTICIPANTS):
                if source == target:
                    continue
                position = (layer_index, source_index, target_index)
                content = _score(block_provider, training_ids,
                    expected_image_ids, indices, method="content_loss", source=source,
                    target=target, layer=layer, check=check, result_sink=result_sink)
                brain = _score(block_provider, training_ids,
                    expected_image_ids, indices, method="brain_loss", source=source,
                    target=target, layer=layer, check=check, result_sink=result_sink)
                _record_method(records, "content_loss", position, content)
                _record_method(records, "brain_loss", position, brain)
                _record_contrast(records, "content_minus_brain", position, content, brain, check)
                _record_contrast(records, "content_minus_within", position, content, within, check)
                # Do not retain preceding target results while scoring the next.
                del content, brain
            del within
    check()
    return PanelResult(expected_image_ids, indices, PARTICIPANTS, LAYERS,
                       _reduce(records["pattern"], check), _reduce(records["profile"], check))


_INTERVAL_FIELDS = (
    ("layer_dyad", "layer_means"), ("layer_source", "layer_source_means"),
    ("layer_group", "layer_group_means"), ("dyad", "dyad_means"),
    ("source", "source_means"), ("group", "group_means"),
)


def _endpoint_array(value):
    return np.asarray(np.nan if value is None else value, dtype=np.float64)


def _interval(point, draws, check):
    check()
    point = _endpoint_array(point)
    _require(draws.shape == (BOOTSTRAP_DRAWS,) + point.shape, "exact_thousand_endpoint_draws_required")
    valid_draws = np.count_nonzero(np.isfinite(draws), axis=0)
    available = np.isfinite(point) & (valid_draws == BOOTSTRAP_DRAWS)
    lower, upper = np.full(point.shape, np.nan), np.full(point.shape, np.nan)
    # A point failure cannot be rescued, nor can one unavailable draw be
    # omitted/replaced. Other endpoints retain their own availability.
    flat_valid = available.reshape(-1)
    if bool(flat_valid.any()):
        values = draws.reshape(BOOTSTRAP_DRAWS, -1)[:, flat_valid]
        bounds = np.percentile(values, [2.5, 97.5], axis=0, method="linear")
        lower.reshape(-1)[flat_valid], upper.reshape(-1)[flat_valid] = bounds
    check()
    return Interval(_readonly(lower), _readonly(upper), _readonly(np.asarray(available)),
                    _readonly(np.asarray(valid_draws, dtype=np.int64)))


def bootstrap_panel(block_provider, decoder_training_image_ids_by_owner, expected_image_ids,
                    *, check, result_sink=None):
    """Point panel plus exactly 1,000 fully recomputed synchronized draws.

    The same immutable 224-position row is passed to the full panel for each
    draw; repeated occurrences remain repeated. The caller's RNG is untouched.
    Only the point evaluation invokes result_sink. No count, seed, panel-size,
    invalid-draw replacement, or data-dependent early-stop option is exposed.
    """
    check()
    training_ids, _ = _context(decoder_training_image_ids_by_owner, expected_image_ids, None, check)
    point = evaluate_panel(block_provider, training_ids,
                           expected_image_ids, check=check, result_sink=result_sink)
    schedule = _readonly(np.random.RandomState(BOOTSTRAP_SEED).randint(0, 224,
        size=(BOOTSTRAP_DRAWS, 224)).astype(np.int64, copy=False))
    samples = {}
    for metric in _METRICS:
        panel = getattr(point, metric)
        samples[metric] = {}
        for interval_name, panel_name in _INTERVAL_FIELDS:
            samples[metric][interval_name] = {
                endpoint: np.empty((BOOTSTRAP_DRAWS,) + _endpoint_array(value).shape, dtype=np.float64)
                for endpoint, value in getattr(panel, panel_name).items()}
    for draw in range(BOOTSTRAP_DRAWS):
        check()
        result = evaluate_panel(block_provider, training_ids,
            expected_image_ids, image_indices=schedule[draw], check=check)
        for metric in _METRICS:
            panel = getattr(result, metric)
            for interval_name, panel_name in _INTERVAL_FIELDS:
                for endpoint, value in getattr(panel, panel_name).items():
                    samples[metric][interval_name][endpoint][draw] = _endpoint_array(value)
        del result
    intervals = {}
    for metric in _METRICS:
        panel = getattr(point, metric)
        fields = []
        for interval_name, panel_name in _INTERVAL_FIELDS:
            fields.append(MappingProxyType({endpoint: _interval(value,
                samples[metric][interval_name][endpoint], check)
                for endpoint, value in getattr(panel, panel_name).items()}))
        intervals[metric] = MetricIntervals(*fields)
    check()
    return BootstrapResult(point, schedule, intervals["pattern"], intervals["profile"])
