"""Bounded replay of the original Finding A image bootstrap.

The point panel uses the unchanged original evaluator and sink. For bootstrap
draws, one supplied block feeds at most sixteen independent scorer states.
Every state follows score_layer's float32 normalization, float64 centering,
block order, C-ordered summaries and exact support checks. There is no pattern
cache, weighted-moment shortcut, draw replacement or available-case reduction.
The original metrics/inference modules remain untouched as scalar references.

One point pass plus 63 draw batches replaces 1,001 provider passes, but still
performs every draw's numerical calculation. At the largest registered layer,
three sixteen-draw states/results require roughly 2.6 GiB, before runtime/I/O
overhead. Caller checks must enforce an admitted finite memory/time budget;
these estimates neither qualify throughput nor authorize empirical access.
This array-only module opens no files, starts no processes and grants nothing.
The first-readout composition uses this helper under its separate caller admission.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MappingProxyType

import numpy as np

import finding_a_inference as original
import finding_a_metrics as metrics


DRAW_BATCH_SIZE = 16


@contextmanager
def _source_iterator(blocks):
    iterator = None

    def begin():
        nonlocal iterator
        iterator = iter(blocks)
        return iterator

    try:
        # Calling iter() may itself open a caller-owned data source. Defer it
        # until the scorer has checked admission, identities and draw bounds.
        yield begin
    except BaseException as error:
        close = getattr(iterator if iterator is not None else blocks, "close", None)
        if callable(close):
            try:
                close()
            except BaseException as cleanup_error:
                error.add_note("batched source cleanup failed: " + type(cleanup_error).__name__)
        raise
    else:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


def _draws(indices):
    original._require(isinstance(indices, np.ndarray) and indices.ndim == 2
        and indices.shape[1] == metrics.TEST_IMAGES and 1 <= len(indices) <= DRAW_BATCH_SIZE
        and indices.dtype.kind in "iu" and not indices.flags.writeable,
        "one_to_sixteen_readonly_224_position_draws_required")
    result = []
    for row in indices:
        metrics._validate_image_indices(row)
        result.append(metrics._readonly(row.astype(np.int64, copy=True)))
    return tuple(result)


class _LayerState:
    """One draw, with the original scalar scorer's independent arithmetic state."""
    def __init__(self, units, draw):
        self.draw = draw
        self.seen = np.zeros(units, dtype=bool)
        self.profile = np.empty(units, dtype=np.float64)
        self.training_mean = np.empty(units, dtype=np.float32)
        self.training_std = np.empty(units, dtype=np.float32)
        self.count = 0
        self.mean_x = np.zeros(metrics.TEST_IMAGES, dtype=np.float64)
        self.mean_y = np.zeros(metrics.TEST_IMAGES, dtype=np.float64)
        self.xx = np.zeros(metrics.TEST_IMAGES, dtype=np.float64)
        self.yy = np.zeros(metrics.TEST_IMAGES, dtype=np.float64)
        self.xy = np.zeros(metrics.TEST_IMAGES, dtype=np.float64)

    def consume(self, block, owner, image_ids, units, check):
        check()
        ids = metrics._block(block, owner, image_ids, units, self.seen)
        # Keep the scalar scorer's operation sequence and array shapes. In
        # particular, do not reduce over a new draw axis or group duplicates.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            predicted, truth = block.predicted[self.draw], block.truth[self.draw]
            raw_x, raw_y = predicted.astype(np.float64), truth.astype(np.float64)
            _, _, px, py, pxy = metrics._centered(raw_x, raw_y, axis=0)
            self.profile[ids] = metrics._correlations(px, py, pxy)
            check()
            x = ((predicted - block.training_mean) / block.training_std).astype(np.float64)
            y = ((truth - block.training_mean) / block.training_std).astype(np.float64)
            bx, by, bxx, byy, bxy = metrics._centered(x, y, axis=1)
            total = self.count + len(ids)
            dx, dy = bx - self.mean_x, by - self.mean_y
            factor = self.count * len(ids) / total
            self.xx += bxx + dx * dx * factor
            self.yy += byy + dy * dy * factor
            self.xy += bxy + dx * dy * factor
            self.mean_x += dx * (len(ids) / total)
            self.mean_y += dy * (len(ids) / total)
        self.count = total
        self.training_mean[ids], self.training_std[ids] = block.training_mean, block.training_std
        self.seen[ids] = True
        check()

    def finish(self, *, method, source, target, owner, layer, image_ids, training_ids, units, check):
        check()
        original._require(self.count == units and bool(self.seen.all()),
                          "complete_all_channel_layer_C_unit_coverage_required")
        pattern = metrics._summary(metrics._correlations(self.xx, self.yy, self.xy), check)
        profile = metrics._summary(self.profile, check)
        return metrics.LayerScores(method, source, target, owner, layer, image_ids,
            pattern, profile, training_ids, metrics._readonly(self.training_mean),
            metrics._readonly(self.training_std), self.draw)


def score_layer_batch(blocks, *, method, source, target, layer, expected_image_ids,
                      decoder_training_image_ids, check, image_indices):
    """Consume one complete layer once for 1..16 supplied original image draws.

    The small draw interface supports the final eight-draw batch and direct
    array comparisons. It does not change bootstrap_panel's fixed 1,000 draws.
    Source iterators close on success, invalid coverage, cancellation or failure.
    """
    with _source_iterator(blocks) as begin:
        check()
        owner = metrics._owner(method, source, target)
        units = metrics._unit_count(layer)
        metrics._ids(expected_image_ids, metrics.TEST_IMAGES)
        metrics._ids(decoder_training_image_ids, 4712)
        original._require(not set(expected_image_ids).intersection(decoder_training_image_ids),
                          "test_decoder_training_image_overlap_forbidden")
        draws = _draws(image_indices)
        states = []
        for draw in draws:
            check()
            states.append(_LayerState(units, draw))
        for block in begin():
            check()
            for state in states:
                state.consume(block, owner, expected_image_ids, units, check)
        return tuple(state.finish(method=method, source=source, target=target, owner=owner,
            layer=layer, image_ids=expected_image_ids, training_ids=decoder_training_image_ids,
            units=units, check=check) for state in states)


def _score_batch(provider, training_ids, image_ids, indices, *, method, source, target, layer, check):
    check()
    supplied = provider(method=method, source=source, target=target, layer=layer)
    if isinstance(supplied, original.UnavailableLayer):
        original._require(isinstance(supplied.reason, str) and bool(supplied.reason),
                          "explicit_unavailable_layer_reason_required")
        return (supplied,) * len(indices)
    owner = source if method == "within_source" else target
    return score_layer_batch(supplied, method=method, source=source, target=target,
        layer=layer, expected_image_ids=image_ids, decoder_training_image_ids=training_ids[owner],
        image_indices=indices, check=check)


def _evaluate_panel_batch(block_provider, training_ids, image_ids, indices, *, check):
    # Snapshot caller identity bindings before any callback/provider can alter
    # its mapping. Each returned panel keeps the corresponding immutable row.
    training_ids, _ = original._context(training_ids, image_ids, None, check)
    draws = _draws(indices)
    indices = original._readonly(np.stack(draws))
    original._require(callable(block_provider), "caller_block_provider_required")
    records = [{metric: original._empty_metric() for metric in original._METRICS} for _ in draws]
    for layer_index, layer in enumerate(original.LAYERS):
        for source_index, source in enumerate(original.PARTICIPANTS):
            within = _score_batch(block_provider, training_ids, image_ids, indices,
                method="within_source", source=source, target=None, layer=layer, check=check)
            for draw in range(len(draws)):
                original._record_method(records[draw], "within_source", (layer_index, source_index), within[draw])
            for target_index, target in enumerate(original.PARTICIPANTS):
                if source == target:
                    continue
                position = layer_index, source_index, target_index
                content = _score_batch(block_provider, training_ids, image_ids, indices,
                    method="content_loss", source=source, target=target, layer=layer, check=check)
                brain = _score_batch(block_provider, training_ids, image_ids, indices,
                    method="brain_loss", source=source, target=target, layer=layer, check=check)
                for draw in range(len(draws)):
                    check()
                    original._record_method(records[draw], "content_loss", position, content[draw])
                    original._record_method(records[draw], "brain_loss", position, brain[draw])
                    original._record_contrast(records[draw], "content_minus_brain", position,
                                             content[draw], brain[draw], check)
                    original._record_contrast(records[draw], "content_minus_within", position,
                                             content[draw], within[draw], check)
                del content, brain
            del within
    results = []
    for draw, record in zip(draws, records):
        check()
        results.append(original.PanelResult(image_ids, draw, original.PARTICIPANTS, original.LAYERS,
            original._reduce(record["pattern"], check), original._reduce(record["profile"], check)))
    return tuple(results)


def bootstrap_panel(block_provider, decoder_training_image_ids_by_owner, expected_image_ids,
                    *, check, result_sink=None):
    """Original point/sink, then exactly 1,000 synchronized draws in batches of16."""
    check()
    training_ids, _ = original._context(decoder_training_image_ids_by_owner, expected_image_ids, None, check)
    point = original.evaluate_panel(block_provider, training_ids, expected_image_ids,
                                    check=check, result_sink=result_sink)
    schedule = original._readonly(np.random.RandomState(original.BOOTSTRAP_SEED).randint(0, 224,
        size=(original.BOOTSTRAP_DRAWS, 224)).astype(np.int64, copy=False))
    samples = {}
    for metric in original._METRICS:
        panel = getattr(point, metric)
        samples[metric] = {}
        for interval_name, panel_name in original._INTERVAL_FIELDS:
            samples[metric][interval_name] = {
                endpoint: np.empty((original.BOOTSTRAP_DRAWS,) + original._endpoint_array(value).shape,
                                   dtype=np.float64)
                for endpoint, value in getattr(panel, panel_name).items()}
    for first in range(0, original.BOOTSTRAP_DRAWS, DRAW_BATCH_SIZE):
        check()
        results = _evaluate_panel_batch(block_provider, training_ids, expected_image_ids,
            schedule[first:first + DRAW_BATCH_SIZE], check=check)
        for offset, result in enumerate(results):
            check()
            for metric in original._METRICS:
                panel = getattr(result, metric)
                for interval_name, panel_name in original._INTERVAL_FIELDS:
                    for endpoint, value in getattr(panel, panel_name).items():
                        samples[metric][interval_name][endpoint][first + offset] = original._endpoint_array(value)
        del results, result
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
