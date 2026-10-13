"""Four focused reducer checks; run only in the coordinator's Slurm allocation."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
from forecast_scoring import UnresolvedScore
from forecast_uncertainty import (
    CONDITIONS, ENDPOINTS, FAMILIES, HORIZONS, PARTICIPANTS, SOURCES,
    FactorialScorePanel, factorial_equivalence_intervals,
)

REGISTRY = json.loads((CODE.parent / "model_recipe_registry.json").read_text())


def panel(source, scores):
    return FactorialScorePanel(source, PARTICIPANTS[source], FAMILIES, HORIZONS, CONDITIONS, scores)


def varied_panels():
    generator = np.random.default_rng(81)
    return tuple(panel(source, -3. + generator.normal(0, .04, (len(PARTICIPANTS[source]), 5, 2, 5)))
                 for source in SOURCES)


def reduce(panels, *, seed=19019, full_reference_margin=.2, interaction_margin=.2, registry=REGISTRY):
    return factorial_equivalence_intervals(registry=registry, panels=panels, seed=seed,
        full_reference_margin=full_reference_margin, interaction_margin=interaction_margin)


def scalar_reference(panels, seed):
    """Independent scalar summation/order-statistic reference, no reducer helpers."""
    tables, means, indices = [], [], []
    children = np.random.SeedSequence(seed).spawn(2)
    for source, child in zip(("WAY", "self_paced"), children):
        supplied = next(item for item in panels if item.source == source)
        table = []
        for person in supplied.scores:
            row = []
            for family in person:
                for horizon in family:
                    original, q11, q01, q10, q00 = map(float, horizon)
                    row.extend((q11 - original, q11 - q10 - q01 + q00))
            table.append(row)
        tables.append(table)
        means.append([math.fsum(row[j] for row in table) / len(table) for j in range(20)])
        rng = np.random.Generator(np.random.PCG64(child))
        indices.append(rng.integers(0, len(table), size=(10_000, len(table))))
    maxima = []
    for draw in range(10_000):
        largest = 0.
        for source_index, table in enumerate(tables):
            selected = indices[source_index][draw]
            for endpoint in range(20):
                sampled_mean = math.fsum(table[index][endpoint] for index in selected) / len(table)
                largest = max(largest, abs(sampled_mean - means[source_index][endpoint]))
        maxima.append(largest)
    radius = sorted(maxima)[math.ceil(.95 * (10_000 - 1))]
    return np.asarray(means).reshape(-1), radius


class ForecastUncertaintyTests(unittest.TestCase):
    def test_all_forty_intervals_match_independent_scalar_reference(self):
        supplied = varied_panels()
        originals = tuple(item.scores.copy() for item in supplied)
        expected_means, expected_radius = scalar_reference(supplied, 19019)
        result = reduce(supplied)
        self.assertEqual(result["endpoint_count"], 40)
        self.assertEqual(result["draws"], 10_000)
        self.assertEqual(result["confidence_level"], .95)
        self.assertEqual(result["quantile_method"], "higher")
        expected_keys = [(source, family, horizon, endpoint) for source in SOURCES
                         for family in FAMILIES for horizon in HORIZONS for endpoint in ENDPOINTS]
        self.assertEqual([(item["source"], item["family"], item["horizon"], item["endpoint"])
                          for item in result["intervals"]], expected_keys)
        np.testing.assert_allclose([item["mean"] for item in result["intervals"]], expected_means, rtol=0, atol=1e-14)
        np.testing.assert_allclose([item["lower"] for item in result["intervals"]], expected_means - expected_radius, rtol=0, atol=1e-14)
        np.testing.assert_allclose([item["upper"] for item in result["intervals"]], expected_means + expected_radius, rtol=0, atol=1e-14)
        self.assertAlmostEqual(result["simultaneous_half_width"], expected_radius, places=14)
        for item, original in zip(supplied, originals):
            np.testing.assert_array_equal(item.scores, original)

    def test_correlated_endpoints_share_draws_and_both_sources_enter_one_maximum(self):
        supplied, source_errors = [], []
        children = np.random.SeedSequence(72).spawn(2)
        for source, child in zip(SOURCES, children):
            n = len(PARTICIPANTS[source])
            effect = np.arange(n, dtype=float) - (n - 1) / 2
            effect *= .01 * np.sqrt(n) / effect.std()
            values = np.full((n, 5, 2, 5), -3.)
            values[..., 1] += effect[:, None, None]
            values[..., 3] += 2 * effect[:, None, None]
            supplied.append(panel(source, values))
            # Twenty endpoints are exact copies/opposites, so shared sampling
            # reduces each source's maximum to one absolute centered mean.
            effect = values[:, 0, 0, 1] - values[:, 0, 0, 0]
            rng = np.random.Generator(np.random.PCG64(child))
            selected = rng.integers(0, n, size=(10_000, n))
            source_errors.append(np.abs(effect[selected].mean(axis=1) - effect.mean()))
        joint = np.maximum(*source_errors)
        expected = np.quantile(joint, .95, method="higher")
        separate = [np.quantile(errors, .95, method="higher") for errors in source_errors]
        self.assertGreater(expected, max(separate))
        result = reduce(tuple(reversed(supplied)), seed=72)
        self.assertAlmostEqual(result["simultaneous_half_width"], expected, places=13)
        for item in result["intervals"]:
            self.assertAlmostEqual((item["upper"] - item["lower"]) / 2, expected, places=13)
        self.assertEqual(result, reduce(supplied, seed=72))

    def test_missing_margins_stay_unresolved_and_boundary_equality_does_not_pass(self):
        supplied = []
        for source in SOURCES:
            values = np.full((len(PARTICIPANTS[source]), 5, 2, 5), -3.)
            values[..., 1] += .125
            supplied.append(panel(source, values))
        unresolved = reduce(supplied, full_reference_margin=None, interaction_margin=.25)
        self.assertEqual(unresolved["full_reference_equivalence"]["status"], "unresolved_missing_margin")
        self.assertIsNone(unresolved["full_reference_equivalence"]["equivalence_established"])
        self.assertTrue(unresolved["interaction_equivalence"]["equivalence_established"])
        self.assertTrue(all(item["inside_margin"] is None for item in unresolved["intervals"]
                            if item["endpoint"] == ENDPOINTS[0]))
        boundary = reduce(supplied, full_reference_margin=.125, interaction_margin=None)
        self.assertEqual(boundary["simultaneous_half_width"], 0.)
        self.assertEqual(boundary["full_reference_equivalence"]["status"], "not_established")
        self.assertIsNone(boundary["interaction_equivalence"]["equivalence_established"])
        self.assertFalse(boundary["full_temporal_explanation_evaluated"])
        for margin in (0, -1., float("nan"), float("inf"), True, ".2"):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                reduce(supplied, full_reference_margin=margin)
        for seed in (None, -1, 1.5, True):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                reduce(supplied, seed=seed)

    def test_incomplete_cohorts_wrong_ids_axes_and_nonfinite_scores_are_refused(self):
        way, self_paced = varied_panels()
        bad_panels = [
            (way,), (way, way), (way, replace(self_paced, source="development")),
            (replace(way, participants=way.participants[:-1], scores=way.scores[:-1]), self_paced),
            (replace(way, participants=(1,) + way.participants[:-1]), self_paced),
            (replace(way, participants=(True,) + way.participants[1:]), self_paced),
            (way, replace(self_paced, participants=("sub-02",) + self_paced.participants[1:])),
            (replace(way, families=FAMILIES[:-1] + (FAMILIES[0],)), self_paced),
            (replace(way, horizons=("near_event", "longer_diagnostic")), self_paced),
            (replace(way, conditions=CONDITIONS[::-1]), self_paced),
            (replace(way, scores=way.scores[:, :, :, :-1]), self_paced),
            (replace(way, scores=way.scores.astype(complex)), self_paced),
        ]
        for value in (float("nan"), float("inf")):
            corrupt = way.scores.copy()
            corrupt[-1, -1, -1, -1] = value
            bad_panels.append((replace(way, scores=corrupt), self_paced))
        for supplied in bad_panels:
            with self.subTest(supplied=supplied[0].source), self.assertRaises((ValueError, UnresolvedScore)):
                reduce(supplied)
        registry = deepcopy(REGISTRY)
        registry["folds"]["WAY_participants"][-1] = 13
        with self.assertRaisesRegex(ValueError, "exact registered"):
            reduce((way, self_paced), registry=registry)


if __name__ == "__main__":
    unittest.main()
