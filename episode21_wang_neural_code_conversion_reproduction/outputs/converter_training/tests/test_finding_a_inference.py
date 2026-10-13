"""Four orchestration checks; execute only in coordinated Slurm compute.

Fabricated compact LayerScores isolate roster, aggregation and interval logic.
They do not qualify full convolutional metric axes. The seven separate metric
tests cover real complete-fc8 arithmetic, including repeated-image resampling.
No empirical arrays, files, model fits or scientific scores are used here.
"""
from collections import Counter
from dataclasses import replace
import math
from pathlib import Path
import sys
from types import MappingProxyType
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import finding_a_inference as inference
import finding_a_metrics as metrics


PEOPLE = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
LAYERS = ("conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2",
          "conv3_3", "conv3_4", "conv4_1", "conv4_2", "conv4_3", "conv4_4",
          "conv5_1", "conv5_2", "conv5_3", "conv5_4", "fc6", "fc7", "fc8")
METHODS = ("within_source", "content_loss", "brain_loss")
ENDPOINTS = METHODS + ("content_minus_brain", "content_minus_within")
TEST_IDS = tuple(f"fabricated-image-{index:03d}" for index in range(224))
TRAIN_IDS = {owner: tuple(f"fabricated-{owner}-train-{index:04d}" for index in range(4712))
             for owner in PEOPLE}
FIELD_PAIRS = (("layer_dyad", "layer_means"), ("layer_source", "layer_source_means"),
               ("layer_group", "layer_group_means"), ("dyad", "dyad_means"),
               ("source", "source_means"), ("group", "group_means"))
SLOPES = {"pattern": dict(zip(ENDPOINTS, (.2, .5, -.1, .6, .3))),
          "profile": dict(zip(ENDPOINTS, (.1, -.2, .3, -.5, -.3)))}
BAD_POSITIONS = {"layer_means": (3, 1, 4), "layer_source_means": (3, 1),
                 "layer_group_means": 3, "dyad_means": (1, 4),
                 "source_means": 1, "group_means": ()}


def readonly(values, dtype=None):
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def freeze(values):
    return MappingProxyType({key: readonly(value) if isinstance(value, np.ndarray) else value
                             for key, value in values.items()})


def known_mean(metric, method, layer, source, target):
    k, i = LAYERS.index(layer), PEOPLE.index(source)
    j = None if target is None else PEOPLE.index(target)
    if metric == "pattern":
        if method == "within_source":
            return .05 + .01 * i + .002 * k
        if method == "content_loss":
            return .25 + .03 * i + .017 * j + .004 * k
        return .1 + .012 * i - .009 * j + .001 * k
    if method == "within_source":
        return -.15 + .013 * i - .003 * k
    if method == "content_loss":
        return .15 + .007 * i + .02 * j + .002 * k
    return .05 - .005 * i + .011 * j + .001 * k


def compact_score(token, *, method, source, target, layer, expected_image_ids,
                  decoder_training_image_ids, image_indices, check):
    """Mock only the correlation engine; preserve real paired-contrast checks."""
    check()
    assert token == (method, source, target, layer)
    assert expected_image_ids == TEST_IDS
    owner = source if method == "within_source" else target
    assert decoder_training_image_ids == TRAIN_IDS[owner]
    assert image_indices.shape == (224,) and not image_indices.flags.writeable
    i, k = PEOPLE.index(source), LAYERS.index(layer)
    summaries = {}
    for metric, width in (("pattern", 224), ("profile", 3 + k % 3)):
        value = known_mean(metric, method, layer, source, target)
        mask = np.ones(width, dtype=bool)
        if i % 2:
            mask[-1] = False
        values = np.where(mask, value, np.nan)
        summaries[metric] = metrics.CorrelationSummary(readonly(values), readonly(mask), value,
                                                       int(mask.sum()), int((~mask).sum()))
    owner_index = PEOPLE.index(owner)
    return metrics.LayerScores(method=method, source=source, target=target,
        decoder_participant=owner, layer=layer, image_ids=TEST_IDS, **summaries,
        decoder_training_image_ids=TRAIN_IDS[owner],
        training_mean=readonly([owner_index], np.float32),
        training_std=readonly([owner_index + 1], np.float32),
        image_indices=readonly(image_indices, np.int64))


def provider(**request):
    return tuple(request[name] for name in ("method", "source", "target", "layer"))


def oracle_metric(metric):
    """Direct complete-array arithmetic for the registered 4/5/19 weights."""
    means = {"within_source": np.array([[known_mean(metric, "within_source", layer, source, None)
                                         for source in PEOPLE] for layer in LAYERS])}
    for method in METHODS[1:]:
        means[method] = np.array([[[np.nan if source == target else
            known_mean(metric, method, layer, source, target) for target in PEOPLE]
            for source in PEOPLE] for layer in LAYERS])
    means["content_minus_brain"] = means["content_loss"] - means["brain_loss"]
    means["content_minus_within"] = means["content_loss"] - means["within_source"][:, :, None]
    layer_sources, layer_groups, dyads, sources, groups = {}, {}, {}, {}, {}
    for endpoint, values in means.items():
        if endpoint == "within_source":
            layer_sources[endpoint] = values.copy()
        else:
            layer_sources[endpoint] = np.column_stack([
                values[:, i, [j for j in range(5) if j != i]].mean(axis=1) for i in range(5)])
            dyads[endpoint] = values.mean(axis=0)
        layer_groups[endpoint] = layer_sources[endpoint].mean(axis=1)
        sources[endpoint] = layer_sources[endpoint].mean(axis=0)
        groups[endpoint] = float(sources[endpoint].mean())
    counts, undefined = {}, {}
    for method in METHODS:
        counts[method] = np.full(means[method].shape, -1, dtype=np.int64)
        undefined[method] = np.full(means[method].shape, -1, dtype=np.int64)
        for k in range(19):
            for i in range(5):
                n = 224 if metric == "pattern" else 3 + k % 3
                positions = [(k, i)] if method == "within_source" else [
                    (k, i, j) for j in range(5) if j != i]
                for position in positions:
                    counts[method][position], undefined[method][position] = n - i % 2, i % 2
    reasons = {key: np.full(value.shape, "fabricated", dtype=object) for key, value in means.items()}
    return inference.MetricPanel(**{name: freeze(value) for name, value in {
        "layer_means": means, "layer_defined_counts": counts, "layer_undefined_counts": undefined,
        "layer_reasons": reasons, "layer_source_means": layer_sources,
        "layer_group_means": layer_groups, "dyad_means": dyads,
        "source_means": sources, "group_means": groups}.items()})


def fixture_panel():
    return inference.PanelResult(TEST_IDS, readonly(np.arange(224)), PEOPLE, LAYERS,
                                 oracle_metric("pattern"), oracle_metric("profile"))


def shift_panel(point, indices, delta):
    """Cheap, internally consistent compact draw endpoint fixtures."""
    changed = {}
    for metric in ("pattern", "profile"):
        panel = getattr(point, metric)
        fields = {}
        for _, field in FIELD_PAIRS:
            fields[field] = freeze({endpoint: value + SLOPES[metric][endpoint] * delta
                                    for endpoint, value in getattr(panel, field).items()})
        changed[metric] = replace(panel, **fields)
    return replace(point, image_indices=readonly(indices, np.int64), **changed)


def poison_contrast(panel):
    """One unavailable layer/dyad propagates through only its dependents."""
    endpoint = "content_minus_brain"
    metric = panel.pattern
    fields = {}
    for field, position in BAD_POSITIONS.items():
        changed = dict(getattr(metric, field))
        if field == "group_means":
            changed[endpoint] = None
        else:
            values = changed[endpoint].copy()
            values[position] = np.nan
            changed[endpoint] = values
        fields[field] = freeze(changed)
    return replace(panel, pattern=replace(metric, **fields))


def literal_percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


class FindingAInferenceTests(unittest.TestCase):
    def test_full_roster_within_reuse_equal_weights_and_point_sink(self):
        calls, received, draw_objects = [], [], []
        draw = readonly(np.roll(np.repeat(np.arange(112), 2), 7))
        caller_training = dict(TRAIN_IDS)

        def supplied(**request):
            # Retained caller storage must not change the admitted bindings
            # after this panel starts invoking provider callbacks.
            caller_training[PEOPLE[-1]] = ("mutated caller binding",)
            calls.append(provider(**request))
            return calls[-1]

        def scored(*args, **kwargs):
            draw_objects.append(kwargs["image_indices"])
            return compact_score(*args, **kwargs)

        with mock.patch.object(inference.metrics, "score_layer", side_effect=scored) as mocked:
            actual = inference.evaluate_panel(supplied, caller_training, TEST_IDS, image_indices=draw,
                                               check=lambda: None, result_sink=received.append)
        self.assertEqual(mocked.call_count, 855)
        self.assertEqual(Counter(key[0] for key in calls),
                         {"within_source": 95, "content_loss": 380, "brain_loss": 380})
        expected_keys = {(method, source, target, layer) for layer in LAYERS for source in PEOPLE
                         for method in METHODS for target in ([None] if method == "within_source"
                         else [value for value in PEOPLE if value != source])}
        self.assertEqual(set(calls), expected_keys)
        self.assertEqual(len(calls), len(set(calls)))
        self.assertEqual(len(received), 855)
        self.assertTrue(all(isinstance(value, metrics.LayerScores) for value in received))
        self.assertEqual({id(value) for value in draw_objects}, {id(draw_objects[0])})
        np.testing.assert_array_equal(actual.image_indices, draw)
        self.assertFalse(actual.image_indices.flags.writeable)
        self.assertEqual((actual.image_ids, actual.participants, actual.layers), (TEST_IDS, PEOPLE, LAYERS))
        for metric in ("pattern", "profile"):
            observed, expected = getattr(actual, metric), oracle_metric(metric)
            for field in ("layer_means", "layer_defined_counts", "layer_undefined_counts",
                          "layer_source_means", "layer_group_means", "dyad_means", "source_means"):
                self.assertEqual(set(getattr(observed, field)), set(getattr(expected, field)))
                for endpoint, values in getattr(expected, field).items():
                    np.testing.assert_allclose(getattr(observed, field)[endpoint], values,
                                               rtol=0, atol=1e-14, equal_nan=True)
                    self.assertFalse(getattr(observed, field)[endpoint].flags.writeable)
            for endpoint, value in expected.group_means.items():
                self.assertAlmostEqual(observed.group_means[endpoint], value, places=14)

    def test_missing_inputs_and_support_mismatch_propagate_without_available_case_reweighting(self):
        absent = ("brain_loss", PEOPLE[0], PEOPLE[1], LAYERS[2])
        mismatch = ("brain_loss", PEOPLE[2], PEOPLE[4], LAYERS[7])
        calls, received = [], []

        def supplied(**request):
            key = provider(**request)
            calls.append(key)
            return inference.UnavailableLayer("fabricated missing model") if key == absent else key

        def scored(token, **kwargs):
            result = compact_score(token, **kwargs)
            if token == mismatch:
                summary = result.pattern
                mask = summary.defined_mask.copy()
                mask[0] = False
                values = summary.correlations.copy()
                values[0] = np.nan
                result = replace(result, pattern=metrics.CorrelationSummary(readonly(values), readonly(mask),
                    summary.mean, summary.defined_count - 1, summary.undefined_count + 1))
            return result

        with mock.patch.object(inference.metrics, "score_layer", side_effect=scored) as mocked:
            actual = inference.evaluate_panel(supplied, TRAIN_IDS, TEST_IDS,
                                               check=lambda: None, result_sink=received.append)
        self.assertEqual(len(calls), 855)
        self.assertEqual(mocked.call_count, 854)
        self.assertEqual(len(received), 854)
        for metric in (actual.pattern, actual.profile):
            self.assertEqual(metric.layer_defined_counts["brain_loss"][2, 0, 1], -1)
            self.assertEqual(metric.layer_reasons["brain_loss"][2, 0, 1], "fabricated missing model")
            self.assertTrue(np.isnan(metric.dyad_means["brain_loss"][0, 1]))
            self.assertTrue(np.isnan(metric.source_means["brain_loss"][0]))
            self.assertIsNone(metric.group_means["brain_loss"])
            self.assertIsNotNone(metric.group_means["content_loss"])
            self.assertIsNotNone(metric.group_means["content_minus_within"])
            self.assertTrue(np.isfinite(metric.source_means["brain_loss"][1:]).all())
        self.assertTrue(np.isnan(actual.pattern.layer_means["content_minus_brain"][7, 2, 4]))
        self.assertTrue(np.isfinite(actual.pattern.layer_means["brain_loss"][7, 2, 4]))
        self.assertTrue(np.isfinite(actual.profile.layer_means["content_minus_brain"][7, 2, 4]))
        self.assertTrue(np.isnan(actual.pattern.source_means["content_minus_brain"][2]))
        self.assertTrue(np.isfinite(actual.profile.source_means["content_minus_brain"][2]))

        for stage, error in (("provider", LookupError("missing caller object")),
                             ("scorer", ArithmeticError("scorer failure")),
                             ("sink", OSError("sink failure"))):
            with self.subTest(stage=stage):
                source = mock.Mock(side_effect=error) if stage == "provider" else provider
                sink = mock.Mock(side_effect=error) if stage == "sink" else None
                scorer = mock.Mock(side_effect=error) if stage == "scorer" else compact_score
                with mock.patch.object(inference.metrics, "score_layer", side_effect=scorer):
                    with self.assertRaises(type(error)):
                        inference.evaluate_panel(source, TRAIN_IDS, TEST_IDS,
                                                 check=lambda: None, result_sink=sink)

    def test_thousand_shared_draws_linear_percentiles_and_one_bad_draw_are_preserved(self):
        point = fixture_panel()
        observed, deltas = [], []
        caller_training = dict(TRAIN_IDS)
        sink = mock.Mock()
        marker = object()
        caller_state = np.random.get_state()

        def evaluated(supplied, training_ids, image_ids, image_indices=None, *, check, result_sink=None):
            self.assertIs(supplied, provider)
            self.assertEqual(dict(training_ids), TRAIN_IDS)
            self.assertEqual(image_ids, TEST_IDS)
            check()
            if image_indices is None:
                caller_training[PEOPLE[-1]] = ("mutated after point",)
                self.assertIs(result_sink, sink)
                result_sink(marker)
                return point
            self.assertIsNone(result_sink)
            self.assertFalse(image_indices.flags.writeable)
            observed.append(image_indices.copy())
            delta = math.fsum((i + 1) * int(value) for i, value in enumerate(image_indices)) / 1.e8
            deltas.append(delta)
            result = shift_panel(point, image_indices, delta)
            return poison_contrast(result) if len(observed) == 418 else result

        with mock.patch.object(inference, "evaluate_panel", side_effect=evaluated) as mocked:
            actual = inference.bootstrap_panel(provider, caller_training, TEST_IDS,
                                                check=lambda: None, result_sink=sink)
        self.assertEqual(mocked.call_count, 1001)
        self.assertEqual(len(observed), 1000)
        expected_draws = np.random.RandomState(42).randint(0, 224, size=(1000, 224))
        np.testing.assert_array_equal(np.asarray(observed), expected_draws)
        np.testing.assert_array_equal(actual.draw_indices, expected_draws)
        self.assertFalse(actual.draw_indices.flags.writeable)
        sink.assert_called_once_with(marker)
        after = np.random.get_state()
        self.assertEqual(caller_state[0], after[0])
        np.testing.assert_array_equal(caller_state[1], after[1])
        self.assertEqual(caller_state[2:], after[2:])
        for metric in ("pattern", "profile"):
            for interval_field, panel_field in FIELD_PAIRS:
                intervals = getattr(getattr(actual, metric), interval_field)
                references = getattr(getattr(point, metric), panel_field)
                self.assertEqual(set(intervals), set(references))
                for endpoint, reference in references.items():
                    expected = np.asarray(reference, dtype=np.float64)
                    available = np.asarray(np.isfinite(expected)).copy()
                    valid = np.where(available, 1000, 0)
                    if metric == "pattern" and endpoint == "content_minus_brain":
                        available = available.copy()
                        valid = valid.copy()
                        available[BAD_POSITIONS[panel_field]] = False
                        valid[BAD_POSITIONS[panel_field]] = 999
                    shifts = [SLOPES[metric][endpoint] * value for value in deltas]
                    lower = np.where(available, expected + literal_percentile(shifts, .025), np.nan)
                    upper = np.where(available, expected + literal_percentile(shifts, .975), np.nan)
                    result = intervals[endpoint]
                    np.testing.assert_array_equal(result.available, available)
                    np.testing.assert_array_equal(result.valid_draws, valid)
                    np.testing.assert_allclose(result.lower, lower, rtol=0, atol=1e-14, equal_nan=True)
                    np.testing.assert_allclose(result.upper, upper, rtol=0, atol=1e-14, equal_nan=True)
                    for values in (result.lower, result.upper, result.available, result.valid_draws):
                        self.assertFalse(values.flags.writeable)

    def test_unavailable_point_cannot_be_rescued_and_cancellation_or_bad_roster_stops(self):
        complete = fixture_panel()
        failed_point = poison_contrast(complete)
        calls = []

        def evaluated(supplied, training_ids, image_ids, image_indices=None, *, check, result_sink=None):
            calls.append(image_indices)
            return failed_point if image_indices is None else replace(complete, image_indices=image_indices)

        with mock.patch.object(inference, "evaluate_panel", side_effect=evaluated):
            actual = inference.bootstrap_panel(provider, TRAIN_IDS, TEST_IDS, check=lambda: None)
        self.assertEqual(len(calls), 1001)
        bad = actual.pattern.group["content_minus_brain"]
        self.assertFalse(bool(bad.available))
        self.assertEqual(int(bad.valid_draws), 1000)
        self.assertTrue(np.isnan(bad.lower))
        self.assertTrue(np.isnan(bad.upper))
        self.assertTrue(bool(actual.profile.group["content_minus_brain"].available))
        self.assertTrue(bool(actual.pattern.group["content_loss"].available))

        def cancelled():
            raise TimeoutError("caller cancelled")

        untouched = mock.Mock()
        for function in (inference.evaluate_panel, inference.bootstrap_panel):
            with self.subTest(function=function.__name__), self.assertRaisesRegex(TimeoutError, "caller cancelled"):
                function(untouched, TRAIN_IDS, TEST_IDS, check=cancelled)
        with self.assertRaises(ValueError):
            inference.evaluate_panel(untouched, {key: value for key, value in TRAIN_IDS.items()
                                                 if key != PEOPLE[-1]}, TEST_IDS, check=lambda: None)
        untouched.assert_not_called()


if __name__ == "__main__":
    unittest.main()
