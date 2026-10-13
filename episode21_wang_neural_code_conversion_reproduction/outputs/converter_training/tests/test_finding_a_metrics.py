"""Seven array-only metric checks; run only in coordinated Slurm compute.

The numerical fixtures contain all 224 fabricated image IDs and all 1,000 fc8
units. They open no empirical payload. Only the pure nineteen-layer reducer
uses small, explicitly fabricated summaries, never a reduced scientific layer.
The scalar oracle independently follows pinned BdPy's saved-float32 boundary:
pattern normalization occurs in float32, then Pearson centering in float64;
profile correlations instead use the raw saved float32 values.
"""
from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import finding_a_metrics as metrics


TEST_IDS = tuple(f"fabricated-tau-test-{index:03d}" for index in range(224))
TRAIN_IDS = tuple(f"fabricated-decoder-train-{index:04d}" for index in range(4712))
LAYERS = ("conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2",
          "conv3_3", "conv3_4", "conv4_1", "conv4_2", "conv4_3", "conv4_4",
          "conv5_1", "conv5_2", "conv5_3", "conv5_4", "fc6", "fc7", "fc8")


def fixture():
    rng = np.random.default_rng(1948)
    mean = np.linspace(-40., 80., 1000, dtype=np.float32)
    std = np.linspace(.15, 3.5, 1000, dtype=np.float32)
    latent = rng.normal(size=(224, 1000)).astype(np.float32)
    noise = rng.normal(size=(224, 1000)).astype(np.float32)
    truth = mean + std * latent
    predicted = mean + std * (np.float32(.8) * latent + np.float32(.6) * noise)
    return predicted, truth, mean, std


def blocks(predicted, truth, mean, std, *, order=None, width=127, decoder="sub-01"):
    order = np.arange(1000, dtype=np.int64) if order is None else order
    return [metrics.MetricBlock(decoder_participant=decoder, image_ids=TEST_IDS,
            full_layer_C_index=order[first:first + width].copy(),
            predicted=predicted[:, order[first:first + width]].copy(),
            truth=truth[:, order[first:first + width]].copy(),
            training_mean=mean[order[first:first + width]].copy(),
            training_std=std[order[first:first + width]].copy())
            for first in range(0, len(order), width)]


def score(data, *, method="content_loss", **changes):
    arguments = dict(method=method, source="sub-03", target="sub-01", layer="fc8",
                     expected_image_ids=TEST_IDS, decoder_training_image_ids=TRAIN_IDS,
                     check=lambda: None)
    arguments.update(changes)
    return metrics.score_layer(data, **arguments)


def scalar_pearson(x, y):
    """Centered scalar reference; no block merging or scorer helper is reused."""
    x, y = tuple(map(float, x)), tuple(map(float, y))
    if not all(math.isfinite(value) for value in x + y):
        return math.nan
    xmean, ymean = math.fsum(x) / len(x), math.fsum(y) / len(y)
    xc, yc = tuple(value - xmean for value in x), tuple(value - ymean for value in y)
    xx, yy = math.fsum(value * value for value in xc), math.fsum(value * value for value in yc)
    if xx == 0. or yy == 0.:
        return math.nan
    value = math.fsum(a * b for a, b in zip(xc, yc)) / math.sqrt(xx * yy)
    return min(1., max(-1., value))


def scalar_reference(predicted, truth, mean, std):
    normalized_prediction = (predicted - mean) / std
    normalized_truth = (truth - mean) / std
    pattern = np.array([scalar_pearson(x, y)
                        for x, y in zip(normalized_prediction, normalized_truth)])
    profile = np.array([scalar_pearson(predicted[:, unit], truth[:, unit])
                        for unit in range(truth.shape[1])])
    return pattern, profile


def fabricated_summary(mean, count):
    """A reducer fixture, not the claimed score of any complete feature axis."""
    if mean is None:
        return metrics.CorrelationSummary(np.full(count, np.nan), np.zeros(count, dtype=bool),
                                          None, 0, count)
    return metrics.CorrelationSummary(np.full(count, mean), np.ones(count, dtype=bool),
                                      mean, count, 0)


def readonly_indices(values, dtype=np.int64):
    result = np.array(values, dtype=dtype)
    result.setflags(write=False)
    return result


class FindingAMetricsTests(unittest.TestCase):
    def assert_summary_matches(self, actual, expected):
        mask = np.isfinite(expected)
        self.assertEqual(actual.correlations.dtype, np.dtype("float64"))
        self.assertEqual(actual.defined_mask.dtype, np.dtype("bool"))
        self.assertFalse(actual.correlations.flags.writeable)
        self.assertFalse(actual.defined_mask.flags.writeable)
        np.testing.assert_allclose(actual.correlations, expected, rtol=0, atol=2e-12, equal_nan=True)
        np.testing.assert_array_equal(actual.defined_mask, mask)
        self.assertEqual(actual.defined_count, int(mask.sum()))
        self.assertEqual(actual.undefined_count, len(mask) - int(mask.sum()))
        self.assertEqual(actual.available, bool(mask.any()))
        if mask.any():
            self.assertAlmostEqual(actual.mean, math.fsum(expected[mask]) / int(mask.sum()), places=12)
        else:
            self.assertIsNone(actual.mean)

    def test_complete_fc8_matches_independent_saved_float32_correlation_reference(self):
        predicted, truth, mean, std = fixture()
        actual = score(blocks(predicted, truth, mean, std))
        pattern, profile = scalar_reference(predicted, truth, mean, std)
        self.assert_summary_matches(actual.pattern, pattern)
        self.assert_summary_matches(actual.profile, profile)
        self.assertEqual(actual.image_ids, TEST_IDS)
        np.testing.assert_array_equal(actual.image_indices, np.arange(224))
        self.assertFalse(actual.image_indices.flags.writeable)
        self.assertEqual(actual.decoder_training_image_ids, TRAIN_IDS)
        np.testing.assert_array_equal(actual.training_mean, mean)
        np.testing.assert_array_equal(actual.training_std, std)
        self.assertEqual(actual.training_mean.dtype, np.dtype("float32"))
        self.assertFalse(actual.training_mean.flags.writeable)
        self.assertFalse(actual.training_std.flags.writeable)

        # Pin the independent scalar reference to public BdPy's np.corrcoef
        # arithmetic at selected rows/units without importing that source.
        normalized_prediction, normalized_truth = (predicted - mean) / std, (truth - mean) / std
        self.assertEqual(normalized_prediction.dtype, np.dtype("float32"))
        for row in (0, 81, 223):
            self.assertAlmostEqual(pattern[row], np.corrcoef(
                normalized_prediction[row], normalized_truth[row])[0, 1], places=13)
        for unit in (0, 401, 999):
            self.assertAlmostEqual(profile[unit], np.corrcoef(
                predicted[:, unit], truth[:, unit])[0, 1], places=13)
        raw_pattern = np.array([scalar_pearson(x, y) for x, y in zip(predicted, truth)])
        self.assertGreater(float(np.max(np.abs(pattern - raw_pattern))), .1)
        # Premature float64 normalization changes source rounding, even though
        # correlation itself must use float64 centered sums.
        wrong64 = np.array([scalar_pearson(x, y) for x, y in zip(
            (predicted.astype(np.float64) - mean) / std,
            (truth.astype(np.float64) - mean) / std)])
        self.assertGreater(float(np.max(np.abs(pattern - wrong64))), 1e-10)

    def test_full_C_unit_identity_survives_shuffled_blocks_and_refuses_gaps_duplicates(self):
        predicted, truth, mean, std = fixture()
        canonical = score(blocks(predicted, truth, mean, std))
        order = np.roll(np.arange(1000, dtype=np.int64)[::-1], 147)
        shuffled = blocks(predicted, truth, mean, std, order=order, width=113)[::-1]
        actual = score(iter(shuffled))
        np.testing.assert_allclose(actual.pattern.correlations, canonical.pattern.correlations,
                                   rtol=0, atol=2e-12)
        np.testing.assert_allclose(actual.profile.correlations, canonical.profile.correlations,
                                   rtol=0, atol=2e-12)
        np.testing.assert_array_equal(actual.training_mean, mean)
        np.testing.assert_array_equal(actual.training_std, std)
        duplicate_ids = shuffled[0].full_layer_C_index.copy()
        duplicate_ids[1] = duplicate_ids[0]
        invalid = {
            "missing block": shuffled[:-1],
            "duplicate block": shuffled + shuffled[:1],
            "duplicate within block": [replace(shuffled[0], full_layer_C_index=duplicate_ids)] + shuffled[1:],
            "wrong image order": [replace(shuffled[0], image_ids=TEST_IDS[::-1])] + shuffled[1:],
            "wrong decoder owner": [replace(shuffled[0], decoder_participant="sub-02")] + shuffled[1:],
            "missing image row": [replace(shuffled[0], predicted=shuffled[0].predicted[:-1])] + shuffled[1:],
        }
        for name, data in invalid.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                score(data)

    def test_undefined_correlations_preserve_support_and_never_intersect_for_contrasts(self):
        predicted, truth, mean, std = fixture()
        truth[:, 0] = mean[0]
        truth[0] = mean
        first = score(blocks(predicted, truth, mean, std))
        second = score(blocks(predicted * np.float32(.9), truth, mean, std), method="brain_loss")
        self.assertEqual((first.pattern.defined_count, first.profile.defined_count), (223, 999))
        self.assertFalse(first.pattern.defined_mask[0])
        self.assertFalse(first.profile.defined_mask[0])
        paired = metrics.paired_contrast(first, second, check=lambda: None)
        for name in ("pattern", "profile"):
            self.assertTrue(getattr(paired, name).available)
            self.assertAlmostEqual(getattr(paired, name).difference,
                getattr(first, name).mean - getattr(second, name).mean, places=12)

        # The reusable within-source baseline belongs to the source decoder,
        # whose training identities and feature scales legitimately differ.
        source_ids = tuple(f"fabricated-source-train-{index:04d}" for index in range(4712))
        within = score(blocks(predicted, truth, mean, std * np.float32(2.), decoder="sub-03"),
                       method="within_source", target=None, decoder_training_image_ids=source_ids)
        baseline = metrics.paired_contrast(first, within, check=lambda: None)
        for name in ("pattern", "profile"):
            self.assertTrue(getattr(baseline, name).available)
            self.assertAlmostEqual(getattr(baseline, name).difference,
                getattr(first, name).mean - getattr(within, name).mean, places=12)

        # Equal defined counts are insufficient: the two methods now lose
        # different image and C-unit identities, with all other IDs retained.
        different = []
        for index, method in ((1, "content_loss"), (2, "brain_loss")):
            altered = predicted.copy()
            altered[index] = mean
            altered[:, index] = mean[index]
            different.append(score(blocks(altered, truth, mean, std), method=method))
        mismatch = metrics.paired_contrast(*different, check=lambda: None)
        for name in ("pattern", "profile"):
            left, right = (getattr(value, name) for value in different)
            self.assertEqual(left.defined_count, right.defined_count)
            self.assertFalse(np.array_equal(left.defined_mask, right.defined_mask))
            contrast = getattr(mismatch, name)
            self.assertFalse(contrast.available)
            self.assertIsNone(contrast.difference)
            self.assertTrue(contrast.reason)

        nonfinite = predicted.copy()
        nonfinite[7, 7] = np.nan
        affected = score(blocks(nonfinite, truth, mean, std))
        self.assertEqual((affected.pattern.undefined_count, affected.profile.undefined_count), (2, 2))
        self.assertFalse(affected.pattern.defined_mask[7])
        self.assertFalse(affected.profile.defined_mask[7])
        constants = np.broadcast_to(mean, (224, 1000)).copy()
        unavailable = score(blocks(constants, constants, mean, std))
        self.assert_summary_matches(unavailable.pattern, np.full(224, np.nan))
        self.assert_summary_matches(unavailable.profile, np.full(1000, np.nan))
        missing = metrics.paired_contrast(first, unavailable, check=lambda: None)
        self.assertFalse(missing.pattern.available)
        self.assertFalse(missing.profile.available)

    def test_nineteen_layer_summary_is_equal_weight_and_requires_every_available_layer(self):
        layers = []
        for index, layer in enumerate(LAYERS):
            layers.append(metrics.LayerScores(method="content_loss", source="sub-03", target="sub-01",
                decoder_participant="sub-01", layer=layer, image_ids=TEST_IDS,
                pattern=fabricated_summary(-.3 + .025 * index, index + 1),
                profile=fabricated_summary(.4 - .02 * index, 19 - index),
                decoder_training_image_ids=TRAIN_IDS,
                training_mean=np.zeros(1, dtype=np.float32), training_std=np.ones(1, dtype=np.float32)))
        actual = metrics.summarize_layers(layers[::-1], check=lambda: None)
        self.assertTrue(actual.pattern.available)
        self.assertTrue(actual.profile.available)
        self.assertAlmostEqual(actual.pattern.mean, math.fsum(x.pattern.mean for x in layers) / 19, places=14)
        self.assertAlmostEqual(actual.profile.mean, math.fsum(x.profile.mean for x in layers) / 19, places=14)
        weighted = math.fsum(x.pattern.mean * x.pattern.defined_count for x in layers) / sum(
            x.pattern.defined_count for x in layers)
        self.assertGreater(abs(actual.pattern.mean - weighted), .05)
        missing = metrics.summarize_layers(layers[:-1], check=lambda: None)
        self.assertFalse(missing.pattern.available)
        self.assertFalse(missing.profile.available)
        self.assertIsNone(missing.pattern.mean)
        self.assertIsNone(missing.profile.mean)
        with self.assertRaises(ValueError):
            metrics.summarize_layers(layers[:-1] + layers[:1], check=lambda: None)
        unavailable = list(layers)
        unavailable[5] = replace(unavailable[5], pattern=fabricated_summary(None, 6))
        partial = metrics.summarize_layers(unavailable, check=lambda: None)
        self.assertFalse(partial.pattern.available)
        self.assertIsNone(partial.pattern.mean)
        self.assertTrue(partial.profile.available)
        self.assertAlmostEqual(partial.profile.mean, actual.profile.mean, places=14)

    def test_large_offset_saved_dtype_moment_overlap_and_cancellation_boundaries(self):
        rng = np.random.default_rng(833)
        latent = rng.integers(-4, 5, size=(224, 1000)).astype(np.float32)
        noise = rng.integers(-2, 3, size=(224, 1000)).astype(np.float32)
        mean, std = np.full(1000, 2 ** 30, dtype=np.float32), np.full(1000, 128., dtype=np.float32)
        truth, predicted = mean + std * latent, mean + std * (latent + noise)
        actual = score(blocks(predicted, truth, mean, std))
        pattern, profile = scalar_reference(predicted, truth, mean, std)
        self.assert_summary_matches(actual.pattern, pattern)
        self.assert_summary_matches(actual.profile, profile)
        # A float64 pre-serialization prediction can retain information that
        # the source float32 file does not. The scorer must not accept it.
        perturbation = np.clip(truth.astype(np.float64) - predicted, -15., 15.)
        before_save = predicted.astype(np.float64) + perturbation
        np.testing.assert_array_equal(before_save.astype(np.float32), predicted)
        self.assertGreater(abs(scalar_pearson(before_save[:, 0], truth[:, 0]) - profile[0]), 1e-5)
        data = blocks(predicted, truth, mean, std)
        bad_std = data[0].training_std.copy()
        bad_std[0] = 0.
        bad_mean = data[0].training_mean.copy()
        bad_mean[0] = np.inf
        invalid = {
            "float64 intermediate": blocks(before_save, truth, mean, std),
            "float64 truth": blocks(predicted, truth.astype(np.float64), mean, std),
            "zero training std": [replace(data[0], training_std=bad_std)] + data[1:],
            "nonfinite training mean": [replace(data[0], training_mean=bad_mean)] + data[1:],
        }
        for name, values in invalid.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                score(values)
        with self.assertRaises(ValueError):
            score(data, decoder_training_image_ids=(TEST_IDS[0],) + TRAIN_IDS[1:])
        with self.assertRaises(ValueError):
            metrics.paired_contrast(actual, replace(actual, training_mean=mean + np.float32(128.)),
                                    check=lambda: None)

        def cancelled():
            raise TimeoutError("caller budget exhausted")

        with self.assertRaisesRegex(TimeoutError, "caller budget exhausted"):
            score(data, check=cancelled)
        with self.assertRaisesRegex(TimeoutError, "caller budget exhausted"):
            metrics.paired_contrast(actual, actual, check=cancelled)
        with self.assertRaisesRegex(TimeoutError, "caller budget exhausted"):
            metrics.summarize_layers([], check=cancelled)

    def test_image_resample_recomputes_full_fc8_from_repeated_raw_rows(self):
        predicted, truth, mean, std = fixture()
        truth[0] = mean
        # This bad row is absent from the fixed fabricated draw below; its
        # old full-sample profile undefinedness must not be carried forward.
        predicted[1, 17] = np.nan
        point = score(blocks(predicted, truth, mean, std))
        backing = np.roll(np.repeat(np.arange(0, 224, 2, dtype=np.int32), 2), 17)
        requested = backing.view()
        requested.setflags(write=False)
        expected_indices = backing.copy()
        actual = score(blocks(predicted, truth, mean, std), image_indices=requested)
        # A readonly caller view can still have a mutable base. Results must
        # retain their own immutable draw snapshot for subsequent comparisons.
        backing[0] = 223
        self.assertEqual(actual.image_ids, TEST_IDS)
        self.assertEqual(actual.image_indices.dtype, np.dtype("int64"))
        self.assertFalse(actual.image_indices.flags.writeable)
        self.assertFalse(np.shares_memory(actual.image_indices, requested))
        np.testing.assert_array_equal(actual.image_indices, expected_indices)
        self.assertEqual(len(np.unique(actual.image_indices)), 112)

        pattern, profile = scalar_reference(predicted[expected_indices], truth[expected_indices], mean, std)
        self.assert_summary_matches(actual.pattern, pattern)
        self.assert_summary_matches(actual.profile, profile)
        self.assertEqual(actual.pattern.undefined_count, 2)
        np.testing.assert_array_equal(np.flatnonzero(~actual.pattern.defined_mask),
                                      np.flatnonzero(expected_indices == 0))
        self.assertTrue(np.isnan(point.profile.correlations[17]))
        self.assertTrue(actual.profile.defined_mask[17])
        self.assertGreater(float(np.max(np.abs(actual.profile.correlations[:17]
                                               - point.profile.correlations[:17]))), 1e-3)
        np.testing.assert_allclose(actual.pattern.correlations,
                                   point.pattern.correlations[expected_indices],
                                   rtol=0, atol=2e-12, equal_nan=True)

    def test_degenerate_draw_profiles_are_undefined_and_draw_identity_is_required(self):
        predicted, truth, mean, std = fixture()
        data = blocks(predicted, truth, mean, std)
        draw = readonly_indices(np.full(224, 31))
        actual = score(data, image_indices=draw)
        expected_pattern = scalar_pearson((predicted[31] - mean) / std, (truth[31] - mean) / std)
        self.assert_summary_matches(actual.pattern, np.full(224, expected_pattern))
        self.assert_summary_matches(actual.profile, np.full(1000, np.nan))
        matched = metrics.paired_contrast(actual, actual, check=lambda: None)
        self.assertTrue(matched.pattern.available)
        self.assertEqual(matched.pattern.difference, 0.)
        self.assertFalse(matched.profile.available)

        different = readonly_indices(np.full(224, 32))
        with self.assertRaises(ValueError):
            metrics.paired_contrast(actual, replace(actual, image_indices=different), check=lambda: None)
        # These are explicitly fabricated reducer metadata fixtures. Their
        # purpose is to bind one shared draw across all nineteen summaries.
        layers = [replace(actual, layer=layer) for layer in LAYERS]
        aggregate = metrics.summarize_layers(layers, check=lambda: None)
        self.assertTrue(aggregate.pattern.available)
        self.assertFalse(aggregate.profile.available)
        layers[8] = replace(layers[8], image_indices=different)
        with self.assertRaises(ValueError):
            metrics.summarize_layers(layers, check=lambda: None)

        invalid = {
            "mutable draw": np.arange(224),
            "wrong length": readonly_indices(np.arange(223)),
            "fractional dtype": readonly_indices(np.arange(224), dtype=np.float64),
            "negative position": readonly_indices([-1] + list(range(223))),
            "position outside original IDs": readonly_indices(list(range(223)) + [224]),
        }
        for name, indices in invalid.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                score(data, image_indices=indices)
        with self.assertRaises(ValueError):
            score(data, image_indices=draw, expected_image_ids=TEST_IDS[:-1] + TEST_IDS[:1])


if __name__ == "__main__":
    unittest.main()
