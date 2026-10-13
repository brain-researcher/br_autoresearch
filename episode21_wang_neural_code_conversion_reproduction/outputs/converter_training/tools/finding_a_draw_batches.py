"""Dormant array-only partition of the original synchronized image bootstrap.

The point is produced separately by finding_a_inference.evaluate_panel (and
its optional point-vector sink). Each of 63 tasks evaluates one canonical
slice of the original 1,000 RandomState(42) draws: 62 slices of 16 and one of
8. Every slice visits the full five-participant, nineteen-layer, 855-slot
panel using the existing independent scorers and paired-support checks.

The merger accepts completed batches in any arrival order, validates their
original image/roster/draw identities and compact reductions, and calls the
original interval implementation. A missing or failed batch cannot be dropped
or replaced. Incoming and dyadic summaries remain separate existing calls.

Caller prerequisites: identical admitted fitted objects, truth, owner moments
and training identities across the point and every task. PanelResult does not
encode those identities, and this adapter cannot authenticate them. Compact
contrast tables are trusted outputs of the existing support-qualified scorer;
they are never reconstructed by subtracting unmatched absolute means.

There is no I/O, access grant, scheduler, model loading or persistence here.
Stream batches to the merger to retain only one batch plus the original
roughly 43 MiB numeric interval samples; resource checks remain caller-owned.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

import finding_a_batched_inference as batched
import finding_a_inference as original


DRAW_BATCH_SIZE = batched.DRAW_BATCH_SIZE
BATCH_COUNT = (original.BOOTSTRAP_DRAWS + DRAW_BATCH_SIZE - 1) // DRAW_BATCH_SIZE


@dataclass(frozen=True)
class DrawBatch:
    batch_index: int
    panels: tuple[original.PanelResult, ...]


def _batch_start(batch_index):
    original._require(type(batch_index) is int and 0 <= batch_index < BATCH_COUNT,
                      "canonical_draw_batch_index_required")
    return batch_index * DRAW_BATCH_SIZE


def _schedule():
    return original._readonly(np.random.RandomState(original.BOOTSTRAP_SEED).randint(
        0, 224, size=(original.BOOTSTRAP_DRAWS, 224)).astype(np.int64, copy=False))


def evaluate_draw_batch(block_provider, decoder_training_image_ids_by_owner,
                        expected_image_ids, *, batch_index, check):
    """Evaluate one fixed full-panel slice; no arbitrary draw/seed/count option."""
    check()
    first = _batch_start(batch_index)
    # The existing batch engine snapshots the five training-ID bindings before
    # any provider access and owns iterator cleanup on failure/cancellation.
    panels = batched._evaluate_panel_batch(block_provider,
        decoder_training_image_ids_by_owner, expected_image_ids,
        _schedule()[first:first + DRAW_BATCH_SIZE], check=check)
    check()
    return DrawBatch(batch_index, panels)


def _mapping(value, keys):
    original._require(isinstance(value, Mapping) and set(value) == set(keys),
                      "complete_original_endpoint_mapping_required")


def _layer_array(value, shape, kind):
    original._require(isinstance(value, np.ndarray) and value.shape == shape
                      and value.dtype == np.dtype(kind) and not value.flags.writeable,
                      "immutable_complete_layer_table_required")


def _metric(metric, check):
    """Bind supplied compact summaries to their unchanged original reductions."""
    original._require(isinstance(metric, original.MetricPanel), "original_metric_panel_required")
    _mapping(metric.layer_means, original.ENDPOINTS)
    _mapping(metric.layer_reasons, original.ENDPOINTS)
    _mapping(metric.layer_defined_counts, original.METHODS)
    _mapping(metric.layer_undefined_counts, original.METHODS)
    for endpoint in original.ENDPOINTS:
        check()
        shape = (19, 5) if endpoint == "within_source" else (19, 5, 5)
        means, reasons = metric.layer_means[endpoint], metric.layer_reasons[endpoint]
        _layer_array(means, shape, np.float64)
        _layer_array(reasons, shape, object)
        original._require(not bool(np.isinf(means).any()) and all(
            isinstance(reason, str) and bool(reason) and reason != "not_evaluated"
            for reason in reasons.flat), "all_original_layer_slots_must_be_terminal")
        if endpoint != "within_source":
            diagonal = np.arange(5)
            original._require(bool(np.isnan(means[:, diagonal, diagonal]).all())
                and bool((reasons[:, diagonal, diagonal] == "self_pair_not_applicable").all()),
                "self_dyads_must_remain_inapplicable")
        if endpoint in original.METHODS:
            for counts in (metric.layer_defined_counts[endpoint], metric.layer_undefined_counts[endpoint]):
                _layer_array(counts, shape, np.int64)
                original._require(bool((counts >= -1).all()), "invalid_layer_support_counts")
                if endpoint != "within_source":
                    original._require(bool((counts[:, diagonal, diagonal] == -1).all()),
                                      "self_dyads_must_remain_inapplicable")
    reduced = original._reduce((metric.layer_means, metric.layer_defined_counts,
                               metric.layer_undefined_counts, metric.layer_reasons), check)
    for _, name in original._INTERVAL_FIELDS[1:]:
        supplied, expected = getattr(metric, name), getattr(reduced, name)
        _mapping(supplied, expected)
        for endpoint, value in expected.items():
            check()
            original._require(np.array_equal(original._endpoint_array(supplied[endpoint]),
                original._endpoint_array(value), equal_nan=True),
                "compact_summary_disagrees_with_original_reduction")


def _panel(panel, image_ids, draw, check):
    check()
    original._require(isinstance(panel, original.PanelResult), "original_panel_result_required")
    original._require(panel.image_ids == image_ids and panel.participants == original.PARTICIPANTS
                      and panel.layers == original.LAYERS, "original_panel_identity_mismatch")
    indices = panel.image_indices
    original._require(isinstance(indices, np.ndarray) and indices.dtype == np.dtype(np.int64)
                      and not indices.flags.writeable and np.array_equal(indices, draw),
                      "canonical_panel_draw_order_required")
    for name in original._METRICS:
        _metric(getattr(panel, name), check)


def assemble_bootstrap(point, batches, *, check):
    """Join all 63 completed batches; no access to features/models or rescoring.

    Batches may arrive out of order. Each batch's internal panel sequence must
    match its exact original draw positions. An iterable is closed on success
    or failure, including refusal before iteration. The original point object
    is retained; no point sink or secondary analysis is invoked here.
    """
    with batched._source_iterator(batches) as begin:
        check()
        original._require(isinstance(point, original.PanelResult), "original_point_panel_required")
        original._image_ids(point.image_ids, 224)
        _panel(point, point.image_ids, np.arange(224, dtype=np.int64), check)
        schedule = _schedule()
        samples = {}
        for metric in original._METRICS:
            panel = getattr(point, metric)
            samples[metric] = {}
            for interval_name, panel_name in original._INTERVAL_FIELDS:
                check()
                samples[metric][interval_name] = {
                    endpoint: np.empty((original.BOOTSTRAP_DRAWS,) + original._endpoint_array(value).shape,
                                       dtype=np.float64)
                    for endpoint, value in getattr(panel, panel_name).items()}
        seen = set()
        for batch in begin():
            check()
            original._require(isinstance(batch, DrawBatch), "completed_draw_batch_required")
            first = _batch_start(batch.batch_index)
            original._require(batch.batch_index not in seen, "duplicate_draw_batch")
            draws = schedule[first:first + DRAW_BATCH_SIZE]
            original._require(isinstance(batch.panels, tuple) and len(batch.panels) == len(draws),
                              "complete_canonical_draw_batch_required")
            for offset, result in enumerate(batch.panels):
                _panel(result, point.image_ids, draws[offset], check)
                for metric in original._METRICS:
                    panel = getattr(result, metric)
                    for interval_name, panel_name in original._INTERVAL_FIELDS:
                        for endpoint, value in getattr(panel, panel_name).items():
                            check()
                            samples[metric][interval_name][endpoint][first + offset] = original._endpoint_array(value)
            seen.add(batch.batch_index)
            del batch, result
        original._require(len(seen) == BATCH_COUNT, "all_63_canonical_draw_batches_required")
        check()
    intervals = {}
    for metric in original._METRICS:
        panel = getattr(point, metric)
        fields = []
        for interval_name, panel_name in original._INTERVAL_FIELDS:
            fields.append(MappingProxyType({endpoint: original._interval(value,
                samples[metric][interval_name][endpoint], check)
                for endpoint, value in getattr(panel, panel_name).items()}))
        intervals[metric] = original.MetricIntervals(*fields)
    check()
    return original.BootstrapResult(point, schedule, intervals["pattern"], intervals["profile"])
