"""Three source-free replay checks; execute only in coordinated Slurm compute.

Real complete-fc8 fixtures reuse the qualified scalar metric test data, not a
new recovery study. Whole-bootstrap roster/schedule checks use explicitly
fabricated small summaries and preserve the production 1,000-draw count.
"""
from collections import Counter
from dataclasses import fields, is_dataclass
from pathlib import Path
import sys
from collections.abc import Mapping
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import finding_a_batched_inference as batched
import finding_a_inference as original
import finding_a_metrics as metrics
from test_finding_a_metrics import TEST_IDS, TRAIN_IDS, fixture, blocks


def readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def equal(test, actual, expected):
    if isinstance(expected, np.ndarray):
        test.assertEqual(actual.dtype, expected.dtype)
        np.testing.assert_array_equal(actual, expected)
    elif is_dataclass(expected):
        test.assertIs(type(actual), type(expected))
        for field in fields(expected):
            equal(test, getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(expected, Mapping):
        test.assertEqual(set(actual), set(expected))
        for key in expected:
            equal(test, actual[key], expected[key])
    else:
        test.assertEqual(actual, expected)


def percentile(values, q):
    ordered = sorted(map(float, values))
    location = (len(ordered) - 1) * q
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


class FindingABatchedInferenceTests(unittest.TestCase):
    def test_all_sixteen_real_draws_equal_scalar_arithmetic_and_C_support(self):
        self.assertEqual(batched.DRAW_BATCH_SIZE, 16)
        predicted, truth, mean, std = fixture()
        truth[:, 0] = mean[0]
        truth[0] = mean
        predicted[3, 17] = np.nan
        # This normalization overflows float32 for one image without changing
        # that column's finite raw values. Profiles must still use raw values.
        predicted[4, 19] = np.finfo(np.float32).max
        std[19] = np.float32(.125)
        order = np.roll(np.arange(1000, dtype=np.int64)[::-1], 147)
        supplied = blocks(predicted, truth, mean, std, order=order, width=113)[::-1]
        draws = np.random.RandomState(42).randint(0, 224, size=(16, 224))
        draws[0] = np.arange(224)
        draws[1] = np.arange(223, -1, -1)
        draws[2] = 0  # constant sampled image: every raw profile is undefined
        draws[3] = np.tile(np.array([1, 2]), 112)
        draws = readonly(draws, np.int64)
        calls, closed = [], []

        def once():
            try:
                for number, block in enumerate(supplied):
                    calls.append(number)
                    yield block
            finally:
                closed.append(True)

        arguments = dict(method="content_loss", source="sub-03", target="sub-01", layer="fc8",
                         expected_image_ids=TEST_IDS, decoder_training_image_ids=TRAIN_IDS, check=lambda: None)
        actual = batched.score_layer_batch(once(), image_indices=draws, **arguments)
        self.assertEqual(calls, list(range(len(supplied))))
        self.assertEqual(closed, [True])
        self.assertEqual(len(actual), 16)
        for number, score in enumerate(actual):
            expected = metrics.score_layer(iter(supplied), image_indices=draws[number], **arguments)
            equal(self, score, expected)
            for value in (score.image_indices, score.pattern.correlations, score.pattern.defined_mask,
                          score.profile.correlations, score.profile.defined_mask,
                          score.training_mean, score.training_std):
                self.assertFalse(value.flags.writeable)
        self.assertIsNone(actual[2].profile.mean)
        self.assertEqual(actual[2].profile.undefined_count, 1000)
        self.assertFalse(actual[0].pattern.defined_mask[4])
        self.assertTrue(actual[0].profile.defined_mask[19])
        self.assertFalse(np.shares_memory(actual[0].profile.correlations, actual[1].profile.correlations))
        self.assertFalse(np.shares_memory(actual[0].image_indices, draws))

    def test_full_bootstrap_fixed_schedule_roster_sink_support_and_intervals(self):
        self.assertEqual((original.BOOTSTRAP_DRAWS, original.BOOTSTRAP_SEED), (1000, 42))
        schedule = readonly(np.random.RandomState(42).randint(0, 224, size=(1000, 224)), np.int64)
        original_training = {owner: TRAIN_IDS for owner in original.PARTICIPANTS}
        caller_training = dict(original_training)
        counts, batch_sizes, scored_draws = Counter(), [], []
        first_call = True

        def provider(*, method, source, target, layer):
            nonlocal first_call
            if first_call:
                caller_training["sub-03"] = tuple("changed-" + value for value in TRAIN_IDS)
                first_call = False
            key = method, source, target, layer
            counts[key] += 1
            if source == "sub-03" and (method == "within_source" or target == "sub-01"):
                return key
            return original.UnavailableLayer("fixture_missing")

        def score(token, *, method, source, target, layer, expected_image_ids,
                  decoder_training_image_ids, image_indices, check):
            check()
            self.assertEqual(token, (method, source, target, layer))
            self.assertEqual(expected_image_ids, TEST_IDS)
            self.assertEqual(decoder_training_image_ids, TRAIN_IDS)
            self.assertFalse(image_indices.flags.writeable)
            z = np.count_nonzero(image_indices == 0) / 224
            base, coefficient = {"within_source": (.1, .01), "content_loss": (.5, .03),
                                 "brain_loss": (.3, -.02)}[method]
            value = base + .001 * original.LAYERS.index(layer) + coefficient * z
            summaries = []
            for metric, width in (("pattern", 224), ("profile", 3)):
                mask = np.ones(width, dtype=bool)
                if metric == "profile" and method == "brain_loss" and np.array_equal(image_indices, schedule[17]):
                    mask[-1] = False
                correlation = np.where(mask, value, np.nan)
                summaries.append(metrics.CorrelationSummary(readonly(correlation), readonly(mask), value,
                    int(mask.sum()), int((~mask).sum())))
            return metrics.LayerScores(method, source, target, source if target is None else target,
                layer, TEST_IDS, *summaries, TRAIN_IDS, readonly([0], np.float32),
                readonly([1], np.float32), readonly(image_indices, np.int64))

        def score_batch(token, *, image_indices, **kwargs):
            batch_sizes.append(len(image_indices))
            if token == ("within_source", "sub-03", None, original.LAYERS[0]):
                scored_draws.extend(np.array(row, copy=True) for row in image_indices)
            return tuple(score(token, image_indices=row, **kwargs) for row in image_indices)

        sink = []
        rng_before = np.random.get_state()
        with mock.patch.object(original.metrics, "score_layer", side_effect=score), \
                mock.patch.object(batched, "score_layer_batch", side_effect=score_batch):
            result = batched.bootstrap_panel(provider, caller_training, TEST_IDS, check=lambda: None,
                                             result_sink=sink.append)
            # Original independent panel path agrees for point, boundary draw
            # seventeen (its profile support differs), and the final draw.
            def reference_provider(**request):
                if request["source"] == "sub-03" and (request["method"] == "within_source" or request["target"] == "sub-01"):
                    return tuple(request[name] for name in ("method", "source", "target", "layer"))
                return original.UnavailableLayer("fixture_missing")
            point = original.evaluate_panel(reference_provider, original_training, TEST_IDS, check=lambda: None)
            equal(self, result.point, point)
            for number in (0, 17, 999):
                reference = original.evaluate_panel(reference_provider, original_training, TEST_IDS,
                    image_indices=schedule[number], check=lambda: None)
                actual = batched._evaluate_panel_batch(reference_provider, original_training, TEST_IDS,
                    schedule[number:number + 1], check=lambda: None)[0]
                equal(self, actual, reference)
        rng_after = np.random.get_state()
        self.assertEqual(rng_before[0], rng_after[0])
        np.testing.assert_array_equal(rng_before[1], rng_after[1])
        self.assertEqual(rng_before[2:], rng_after[2:])
        np.testing.assert_array_equal(result.draw_indices, schedule)
        # Exclude the three explicit one-row panel comparisons above.
        np.testing.assert_array_equal(np.stack(scored_draws[:1000]), schedule)
        self.assertFalse(result.draw_indices.flags.writeable)
        expected_slots = {(method, source, target, layer) for layer in original.LAYERS
            for source in original.PARTICIPANTS for method, target in [("within_source", None)] + [
                (method, target) for target in original.PARTICIPANTS if target != source
                for method in ("content_loss", "brain_loss")]}
        self.assertEqual(set(counts), expected_slots)
        self.assertEqual(set(counts.values()), {64})  # point + ceil(1000/16)
        self.assertEqual(Counter(batch_sizes[:63 * 57]), {16: 62 * 57, 8: 57})
        self.assertEqual(len(sink), 57)
        self.assertTrue(all(np.array_equal(value.image_indices, np.arange(224)) for value in sink))
        k, i, j = 18, original.PARTICIPANTS.index("sub-03"), original.PARTICIPANTS.index("sub-01")
        z = np.count_nonzero(schedule == 0, axis=1) / 224
        contrast = result.pattern.layer_dyad["content_minus_brain"]
        self.assertTrue(contrast.available[k, i, j])
        self.assertEqual(contrast.valid_draws[k, i, j], 1000)
        self.assertAlmostEqual(contrast.lower[k, i, j], percentile(.2 + .05 * z, .025), places=14)
        self.assertAlmostEqual(contrast.upper[k, i, j], percentile(.2 + .05 * z, .975), places=14)
        unavailable = result.profile.layer_dyad["content_minus_brain"]
        self.assertFalse(unavailable.available[k, i, j])
        self.assertEqual(unavailable.valid_draws[k, i, j], 999)
        self.assertTrue(np.isnan(unavailable.lower[k, i, j]))
        self.assertTrue(result.profile.layer_dyad["content_minus_within"].available[k, i, j])
        within = result.profile.source["within_source"]
        self.assertTrue(within.available[i])
        self.assertAlmostEqual(within.lower[i], percentile(.109 + .01 * z, .025), places=14)
        self.assertFalse(result.profile.group["within_source"].available)
        self.assertEqual(result.profile.group["within_source"].valid_draws, 0)

    def test_real_paired_support_coverage_and_cancellation_close_source(self):
        predicted, truth, mean, std = fixture()
        original_blocks = blocks(predicted, truth, mean, std)
        arguments = dict(method="content_loss", source="sub-03", target="sub-01", layer="fc8",
                         expected_image_ids=TEST_IDS, decoder_training_image_ids=TRAIN_IDS)
        draws = readonly([np.arange(224), np.tile(np.arange(112), 2)], np.int64)

        class Unopened:
            def __init__(self):
                self.iterated = self.closed = False

            def __iter__(self):
                self.iterated = True
                raise AssertionError("iterable opened before admission/draw validation")

            def close(self):
                self.closed = True

        for indices, callback, error in (
                (draws, mock.Mock(side_effect=PermissionError("fixture denied")), PermissionError),
                (readonly(np.zeros((17, 224), dtype=np.int64)), lambda: None, ValueError)):
            unopened = Unopened()
            with self.assertRaises(error):
                batched.score_layer_batch(unopened, image_indices=indices, check=callback, **arguments)
            self.assertFalse(unopened.iterated)
            self.assertTrue(unopened.closed)
        for bad_blocks in (original_blocks[:-1], original_blocks + original_blocks[:1]):
            closed = []

            def invalid():
                try:
                    yield from bad_blocks
                finally:
                    closed.append(True)

            with self.assertRaises(ValueError):
                batched.score_layer_batch(invalid(), image_indices=draws, check=lambda: None, **arguments)
            self.assertEqual(closed, [True])
        started, closed = [], []

        def cancel_source():
            try:
                started.append(True)
                yield original_blocks[0]
                raise AssertionError("source resumed after cancellation")
            finally:
                closed.append(True)

        def check():
            if started:
                raise TimeoutError("fixed fixture cancellation")

        with self.assertRaisesRegex(TimeoutError, "fixed fixture cancellation"):
            batched.score_layer_batch(cancel_source(), image_indices=draws, check=check, **arguments)
        self.assertEqual(closed, [True])
        brain_prediction = predicted.copy()
        brain_prediction[:, 7] = np.float32(5)  # one additional undefined profile, not an intersection
        brain_blocks = blocks(brain_prediction, truth, mean, std)
        training = {owner: TRAIN_IDS for owner in original.PARTICIPANTS}
        read_counts, closed = Counter(), []

        def provider(*, method, source, target, layer):
            if (source, target, layer) != ("sub-03", "sub-01", "fc8") or method not in ("content_loss", "brain_loss"):
                return original.UnavailableLayer("fixture_missing")
            data = original_blocks if method == "content_loss" else brain_blocks

            def values():
                try:
                    for item in data:
                        read_counts[method] += 1
                        yield item
                finally:
                    closed.append(method)
            return values()

        actual = batched._evaluate_panel_batch(provider, training, TEST_IDS, draws, check=lambda: None)
        self.assertEqual(dict(read_counts), {"content_loss": len(original_blocks), "brain_loss": len(brain_blocks)})
        self.assertEqual(Counter(closed), {"content_loss": 1, "brain_loss": 1})
        for number, panel in enumerate(actual):
            reference = original.evaluate_panel(provider, training, TEST_IDS, image_indices=draws[number], check=lambda: None)
            equal(self, panel, reference)
            position = (18, original.PARTICIPANTS.index("sub-03"), original.PARTICIPANTS.index("sub-01"))
            self.assertEqual(panel.profile.layer_reasons["content_minus_brain"][position],
                             "different_defined_supports_no_intersection_permitted")
            self.assertTrue(np.isnan(panel.profile.layer_means["content_minus_brain"][position]))
            self.assertTrue(np.isfinite(panel.pattern.layer_means["content_minus_brain"][position]))


if __name__ == "__main__":
    unittest.main()
