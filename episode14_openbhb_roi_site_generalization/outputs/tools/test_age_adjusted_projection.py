#!/usr/bin/env python3
"""Focused fixed-age-adjustment checks, run only in the owned Slurm session."""
from copy import deepcopy
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

import run_age_adjusted_projection as diagnostic


def fixture(seed=14):
    rng = np.random.RandomState(seed)
    sites = np.repeat(np.arange(6).astype(str), 16)
    age = 30 + np.repeat(np.arange(6), 16) * 2 + rng.normal(0, 5, 96)
    x = rng.normal(size=(96, 12))
    x[:, :3] += age[:, None] / 10
    return x, sites, age


def appearance(person, age, prediction, domain="a", view=0, support=True):
    return {"id": person, "age": float(age), "domain": domain, "view": view,
            "role": "external", "on_common_support": support,
            "predictions": {name: prediction for name in diagnostic.MODELS},
            "site_correctness": None}


class FocusedChecks(unittest.TestCase):
    def test_sham_preserves_site_composition_and_ols_uses_original_metric(self):
        x, sites, age = fixture()
        sham = diagnostic.within_site_sham(age, sites)
        np.testing.assert_array_equal(sham, diagnostic.within_site_sham(age, sites))
        self.assertTrue(np.any(sham != age))
        for site in set(sites):
            np.testing.assert_array_equal(np.sort(sham[sites == site]), np.sort(age[sites == site]))
        z = diagnostic.legacy.Scale().fit(x).transform(x)
        residual, state = diagnostic.regress_age(z, age)
        basis = np.column_stack((np.ones(len(age)), (age - age.mean()) / age.std()))
        np.testing.assert_allclose(basis.T @ residual, 0, atol=1e-10)
        np.testing.assert_allclose(z, residual + basis @ state["coefficients"], atol=1e-12)
        with self.assertRaises(diagnostic.prior.SupportFailure):
            diagnostic.regress_age(z, np.ones(len(age)))

    def test_feature_only_transform_projects_original_z_and_covariance(self):
        x, sites, age = fixture()
        maps = diagnostic.DirectionMaps().fit(x, sites, age)
        before = pickle.dumps(maps)
        held = x[:7] + 7
        transformed = maps.transform(held)
        self.assertEqual(before, pickle.dumps(maps))
        z_held = maps.pre.transform(held)
        z_fit = maps.pre.transform(x)
        covariance = z_fit.T @ (age - age.mean()) / len(age)
        for arm, directions in maps.directions.items():
            np.testing.assert_allclose(transformed[arm], z_held - (z_held @ directions.T) @ directions,
                                       atol=1e-12)
            np.testing.assert_allclose(transformed[arm] @ directions.T, 0, atol=1e-11)
            fraction = np.sum((directions @ covariance)**2) / np.sum(covariance**2)
            self.assertAlmostEqual(fraction, maps.metadata["arms"][arm]["removed_age_covariance_squared_norm_fraction"])
            self.assertGreaterEqual(fraction, 0)
            self.assertLessEqual(fraction, 1 + 1e-12)
        self.assertLess(maps.metadata["arms"]["age_adjusted"]["training_residual_covariance_with_actual_age_norm"], 1e-10)

    def test_outer_fold_fit_inputs_and_no_evaluation_age_fit(self):
        x, sites, age = fixture()
        # Keep all six training classes and evaluate one internal/external row
        # per class; this synthetic fold tests boundaries, not real role counts.
        fit = np.flatnonzero(np.arange(len(x)) % 16 < 12)
        internal = np.flatnonzero(np.arange(len(x)) % 16 == 12)
        external = np.flatnonzero(np.arange(len(x)) % 16 == 13)
        bundle = SimpleNamespace(splits=[{"fit": fit, "internal": internal, "external": external}],
                                 sites=sites, age=age, ids=np.arange(len(x)).astype(str))
        recorded = []
        original = diagnostic.DirectionMaps.fit

        def traced_fit(instance, features, labels, ages):
            recorded.append((features.copy(), labels.copy(), ages.copy()))
            return original(instance, features, labels, ages)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(diagnostic.DirectionMaps, "fit", traced_fit):
                first = diagnostic.outer_fold(bundle, x, "fixture", 0, Path(directory) / "first")
                changed = deepcopy(bundle)
                changed.age[np.concatenate((internal, external))] += 1000
                second = diagnostic.outer_fold(changed, x, "fixture", 0, Path(directory) / "second")
            state = np.load(Path(directory) / "first/outer/fixture_view0.npz")
            self.assertEqual(state["directions_age_adjusted"].shape, (4, x.shape[1]))
        for features, labels, fitted_ages in recorded:
            np.testing.assert_array_equal(features, x[fit])
            np.testing.assert_array_equal(labels, sites[fit])
            np.testing.assert_array_equal(fitted_ages, age[fit])
        for a, b in zip(first["appearances"], second["appearances"]):
            self.assertEqual(a["predictions"], b["predictions"])
            self.assertEqual(a["site_correctness"], b["site_correctness"])
        self.assertEqual(first["direction_metadata"], second["direction_metadata"])

    def test_rank_failure_is_not_silently_substituted(self):
        x, sites, age = fixture()
        with patch.object(diagnostic.legacy, "site_directions", return_value=(np.eye(12)[:3], 3)):
            with self.assertRaises(diagnostic.prior.SupportFailure):
                diagnostic.DirectionMaps().fit(x, sites, age)

    def test_age_scoring_matches_prior_draws_and_preserves_support(self):
        rows = [appearance("repeat", 0, 10, view=0), appearance("repeat", 0, -10, view=1),
                appearance("single", 20, 20)]
        group = diagnostic.group_arrays(rows)
        self.assertAlmostEqual(group["absolute"][0], 5)
        self.assertAlmostEqual(group["centered"][0], 5)
        first = diagnostic.group_arrays([appearance("p", 0, 0), appearance("q", 0, 10)])
        second = deepcopy(first)
        boot = diagnostic.bootstrap_models([first, second], centered=True, replicates=128)
        self.assertEqual(float(boot[:, 0].min()), 0)
        np.testing.assert_array_equal(boot[:, 0], boot[:, 1])
        rng = np.random.RandomState(diagnostic.SEED)
        draws = rng.randint(2, size=(128, 2))
        expected = np.zeros_like(boot)
        for domain in range(2):
            for replicate, _ in np.argwhere(draws == domain):
                weight = rng.multinomial(2, [0.5, 0.5]) / 2
                center = weight @ first["residuals"]
                expected[replicate] += np.sum(weight[:, None] * np.abs(first["residuals"] - center), axis=0) / 2
        np.testing.assert_allclose(boot, expected, atol=1e-12)
        rows = [appearance(str(i), i, i + 1) for i in range(10)]
        rows += [appearance("small", 20, 21, domain="small"),
                 appearance("outside", 21, 22, domain="outside", support=False)]
        summary = diagnostic.summarize_population(rows, common=True, intervals=False)
        self.assertEqual(summary["centered"]["included_domains"], ["a"])
        self.assertEqual(summary["absolute_on_centered_eligible_domains"]["included_domains"], ["a"])
        self.assertEqual(set(summary["centered"]["excluded_domains"]), {"small", "outside"})

    def test_site_correctness_appearance_weighting_and_paired_intervals(self):
        rows = []
        for person, domain, view, correct in (("repeat", "a", 0, True), ("repeat", "a", 1, False),
                                               ("single", "a", 0, True), ("other", "b", 0, False)):
            row = appearance(person, 20, 20, domain=domain, view=view)
            row.update(role="internal", site_correctness={arm: correct for arm in diagnostic.ARMS})
            rows.append(row)
        report = diagnostic.site_probe_summary(rows, intervals=True, replicates=128)
        self.assertAlmostEqual(report["domains"]["a"]["class_recall"]["raw"], .75)
        self.assertAlmostEqual(report["balanced_accuracy"]["raw"], .375)
        for value in report["contrasts"].values():
            self.assertEqual(value["delta_balanced_accuracy"], 0)
            self.assertEqual(value["descriptive_percentiles_2_5_97_5"], [0, 0])


if __name__ == "__main__":
    diagnostic.legacy.compute_only()
    with threadpool_limits(limits=1):
        unittest.main(verbosity=2)
