"""Four secondary-summary checks; run only in coordinated Slurm compute.

The imported inference fixtures are explicitly fabricated compact summaries,
not evidence of full-feature qualification or empirical decoding performance.
The local seeded draw oracle implements the adopted reproducible source-style
rule; it does not claim to recover the original paper's unseeded historical draws.
"""
from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import finding_a_secondary as secondary
from test_finding_a_inference import (
    ENDPOINTS, LAYERS, PEOPLE, TEST_IDS, fixture_panel, known_mean,
    literal_percentile, poison_contrast, readonly,
)


PAIR_ORDER = tuple((source, target) for source in PEOPLE for target in PEOPLE if source != target)


def literal_pairs(metric, endpoint):
    """One original row per dyad; within-source follows each row's SOURCE."""
    values = []
    for layer in LAYERS:
        row = []
        for source, target in PAIR_ORDER:
            within = known_mean(metric, "within_source", layer, source, None)
            content = known_mean(metric, "content_loss", layer, source, target)
            brain = known_mean(metric, "brain_loss", layer, source, target)
            row.append({"within_source": within, "content_loss": content, "brain_loss": brain,
                        "content_minus_brain": content - brain,
                        "content_minus_within": content - within}[endpoint])
        values.append(row)
    return np.array(values)


def literal_schedule():
    """Pinned sequential choice order, with every rejected proposal counted."""
    rng = np.random.RandomState(42)
    draws = np.full((1000, 20), -1, dtype=np.int64)
    proposals = np.zeros(1000, dtype=np.int64)
    for draw in range(1000):
        accepted = 0
        for proposal in range(1000):
            source = int(rng.choice(5, 1, replace=True)[0])
            target = int(rng.choice(5, 1, replace=True)[0])
            proposals[draw] = proposal + 1
            if source == target:
                continue
            draws[draw, accepted] = PAIR_ORDER.index((PEOPLE[source], PEOPLE[target]))
            accepted += 1
            if accepted == 20:
                break
        if accepted != 20:
            raise AssertionError("fixed literal fixture unexpectedly exhausted its cap")
    return draws, proposals


class FindingASecondaryTests(unittest.TestCase):
    def assert_readonly_interval(self, interval):
        for value in (interval.lower, interval.upper, interval.available, interval.valid_draws):
            self.assertFalse(value.flags.writeable)

    def test_incoming_uses_four_sources_and_nineteen_layers_with_source_bound_baseline(self):
        actual = secondary.target_incoming(fixture_panel(), check=lambda: None)
        self.assertEqual((actual.participants, actual.layers, actual.image_ids), (PEOPLE, LAYERS, TEST_IDS))
        for metric in ("pattern", "profile"):
            result = getattr(actual, metric)
            for endpoint in ENDPOINTS:
                rows = literal_pairs(metric, endpoint)
                layer_targets = np.array([[math.fsum(rows[k, edge] for edge, (_, target) in
                    enumerate(PAIR_ORDER) if target == person) / 4 for person in PEOPLE]
                    for k in range(19)])
                targets = np.array([math.fsum(layer_targets[:, j]) / 19 for j in range(5)])
                layer_groups = np.array([math.fsum(row) / 5 for row in layer_targets])
                np.testing.assert_allclose(result.layer_target_means[endpoint], layer_targets, rtol=0, atol=1e-14)
                np.testing.assert_allclose(result.target_means[endpoint], targets, rtol=0, atol=1e-14)
                np.testing.assert_allclose(result.layer_group_means[endpoint], layer_groups, rtol=0, atol=1e-14)
                self.assertAlmostEqual(result.group_means[endpoint], math.fsum(targets) / 5, places=14)
                for field in ("layer_target_means", "target_means", "layer_group_means"):
                    self.assertFalse(getattr(result, field)[endpoint].flags.writeable)
            # A target's own within-source value is not its incoming baseline.
            wrong_target_baseline = known_mean(metric, "within_source", LAYERS[0], PEOPLE[0], None)
            self.assertGreater(abs(result.layer_target_means["within_source"][0, 0]
                                   - wrong_target_baseline), .01)

        missing = secondary.target_incoming(poison_contrast(fixture_panel()), check=lambda: None)
        self.assertTrue(np.isnan(missing.pattern.layer_target_means["content_minus_brain"][3, 4]))
        self.assertTrue(np.isnan(missing.pattern.target_means["content_minus_brain"][4]))
        self.assertIsNone(missing.pattern.group_means["content_minus_brain"])
        self.assertTrue(np.isfinite(missing.pattern.target_means["content_minus_brain"][:4]).all())
        self.assertIsNotNone(missing.profile.group_means["content_minus_brain"])
        self.assertIsNotNone(missing.pattern.group_means["content_loss"])

    def test_all_dyadic_draws_match_literal_rng_and_scalar_percentile_oracle(self):
        before = np.random.get_state()
        actual = secondary.dyadic_bootstrap(fixture_panel(), check=lambda: None)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        draws, proposals = literal_schedule()
        self.assertEqual((actual.participants, actual.layers, actual.image_ids), (PEOPLE, LAYERS, TEST_IDS))
        self.assertEqual(actual.pair_order, PAIR_ORDER)
        np.testing.assert_array_equal(actual.draw_indices, draws)
        np.testing.assert_array_equal(actual.proposal_counts, proposals)
        np.testing.assert_array_equal(actual.draw_available, np.ones(1000, dtype=bool))
        self.assertGreater(int(proposals.max()), 20)
        self.assertTrue(any(len(set(row)) < 20 for row in draws))
        for values in (actual.draw_indices, actual.proposal_counts, actual.draw_available):
            self.assertFalse(values.flags.writeable)
        for metric in ("pattern", "profile"):
            result = getattr(actual, metric)
            for endpoint in ENDPOINTS:
                original = literal_pairs(metric, endpoint)
                np.testing.assert_allclose(result.pair_values[endpoint], original, rtol=0, atol=1e-14)
                layer_points = np.array([math.fsum(row) / 20 for row in original])
                np.testing.assert_allclose(result.layer_means[endpoint], layer_points, rtol=0, atol=1e-14)
                self.assertAlmostEqual(result.group_means[endpoint], math.fsum(layer_points) / 19, places=14)
                self.assertFalse(result.pair_values[endpoint].flags.writeable)
                # Every repeated edge contributes another row, with the SAME
                # schedule for all methods, layers, metrics and contrasts.
                sampled_layers = np.array([[math.fsum(original[k, edge] for edge in draw) / 20
                                            for k in range(19)] for draw in draws])
                sampled_groups = np.array([math.fsum(row) / 19 for row in sampled_layers])
                layer_interval, group_interval = result.layer_intervals[endpoint], result.group_intervals[endpoint]
                np.testing.assert_array_equal(layer_interval.available, np.ones(19, dtype=bool))
                np.testing.assert_array_equal(layer_interval.valid_draws, np.full(19, 1000))
                np.testing.assert_allclose(layer_interval.lower,
                    [literal_percentile(sampled_layers[:, k], .025) for k in range(19)], rtol=0, atol=1e-14)
                np.testing.assert_allclose(layer_interval.upper,
                    [literal_percentile(sampled_layers[:, k], .975) for k in range(19)], rtol=0, atol=1e-14)
                self.assertTrue(bool(group_interval.available))
                self.assertEqual(int(group_interval.valid_draws), 1000)
                self.assertAlmostEqual(float(group_interval.lower), literal_percentile(sampled_groups, .025), places=14)
                self.assertAlmostEqual(float(group_interval.upper), literal_percentile(sampled_groups, .975), places=14)
                self.assert_readonly_interval(layer_interval)
                self.assert_readonly_interval(group_interval)

    def test_missing_original_edge_and_proposal_cap_failure_cannot_be_reweighted_or_replaced(self):
        missing = secondary.dyadic_bootstrap(poison_contrast(fixture_panel()), check=lambda: None)
        edge = PAIR_ORDER.index((PEOPLE[1], PEOPLE[4]))
        self.assertTrue(np.any(np.all(missing.draw_indices != edge, axis=1)))
        result = missing.pattern
        self.assertTrue(np.isnan(result.pair_values["content_minus_brain"][3, edge]))
        interval = result.layer_intervals["content_minus_brain"]
        self.assertFalse(bool(interval.available[3]))
        self.assertEqual(int(interval.valid_draws[3]), 0)
        self.assertTrue(np.isnan(interval.lower[3]))
        self.assertTrue(np.isnan(interval.upper[3]))
        self.assertEqual(result.layer_reasons["content_minus_brain"][3], "incomplete_original_pair_table")
        self.assertTrue(np.all(np.delete(interval.available, 3)))
        self.assertFalse(bool(result.group_intervals["content_minus_brain"].available))
        self.assertEqual(result.group_reasons["content_minus_brain"], "required_layer_unavailable")
        self.assertTrue(bool(result.group_intervals["content_loss"].available))
        self.assertTrue(bool(missing.profile.group_intervals["content_minus_brain"].available))

        class OneCappedDraw:
            def __init__(self):
                self.calls = 0

            def choice(self, population, size, replace):
                if (population, size, replace) != (5, 1, True):
                    raise AssertionError("literal scalar source/target choice required")
                call = self.calls
                self.calls += 1
                # Draw0 exhausts1000 self proposals. Every later draw receives
                # exactly20 repeated0->1 edges; there is no replacement draw.
                value = 0 if call < 2000 else (call - 2000) % 2
                return np.array([value], dtype=np.int64)

        rng = OneCappedDraw()
        with mock.patch.object(secondary, "_rng", return_value=rng) as factory:
            capped = secondary.dyadic_bootstrap(fixture_panel(), check=lambda: None)
        factory.assert_called_once_with()
        self.assertEqual(rng.calls, 2000 + 2 * 20 * 999)
        np.testing.assert_array_equal(capped.draw_indices[0], np.full(20, -1))
        np.testing.assert_array_equal(capped.draw_indices[1:],
                                      np.full((999, 20), PAIR_ORDER.index((PEOPLE[0], PEOPLE[1]))))
        np.testing.assert_array_equal(capped.proposal_counts, np.array([1000] + [20] * 999))
        np.testing.assert_array_equal(capped.draw_available, np.array([False] + [True] * 999))
        for metric in (capped.pattern, capped.profile):
            for endpoint in ENDPOINTS:
                layer, group = metric.layer_intervals[endpoint], metric.group_intervals[endpoint]
                self.assertFalse(bool(layer.available.any()))
                np.testing.assert_array_equal(layer.valid_draws, np.full(19, 999))
                self.assertTrue(np.isnan(layer.lower).all())
                self.assertTrue(np.isnan(layer.upper).all())
                self.assertFalse(bool(group.available))
                self.assertEqual(int(group.valid_draws), 999)
                self.assertTrue(np.isnan(group.lower))
                self.assertTrue(np.isnan(group.upper))
                np.testing.assert_array_equal(metric.layer_reasons[endpoint],
                                              np.full(19, "proposal_cap_exhausted", dtype=object))
                self.assertEqual(metric.group_reasons[endpoint], "proposal_cap_exhausted")

    def test_only_complete_original_point_roster_is_accepted_and_cancellation_propagates(self):
        panel = fixture_panel()
        malformed = {
            "participant order": replace(panel, participants=PEOPLE[::-1]),
            "missing participant": replace(panel, participants=PEOPLE[:-1]),
            "missing layer": replace(panel, layers=LAYERS[:-1]),
            "duplicate image identity": replace(panel, image_ids=TEST_IDS[:-1] + TEST_IDS[:1]),
            "nonidentity image draw": replace(panel, image_indices=readonly(np.roll(np.arange(224), 1))),
            "mutable image draw": replace(panel, image_indices=np.arange(224)),
        }
        for name, value in malformed.items():
            for function in (secondary.target_incoming, secondary.dyadic_bootstrap):
                with self.subTest(name=name, function=function.__name__), self.assertRaises(ValueError):
                    function(value, check=lambda: None)

        def cancelled():
            raise TimeoutError("caller cancelled secondary summary")

        with mock.patch.object(secondary, "_rng") as untouched:
            for function in (secondary.target_incoming, secondary.dyadic_bootstrap):
                with self.subTest(function=function.__name__), self.assertRaisesRegex(TimeoutError, "caller cancelled"):
                    function(panel, check=cancelled)
            untouched.assert_not_called()


if __name__ == "__main__":
    unittest.main()
