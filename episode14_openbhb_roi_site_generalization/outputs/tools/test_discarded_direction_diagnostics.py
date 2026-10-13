#!/usr/bin/env python3
"""Focused EP14 diagnostics checks; execute only in the owned Slurm session."""
from copy import deepcopy
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

import run_discarded_direction_diagnostics as diagnostic


def appearance(person, age, value, domain="a", view=0, support=True):
    predictions = {name: value for name in diagnostic.MODELS}
    predictions["centered_zero"] = 0.0
    return {"id": person, "age": float(age), "domain": domain, "view": view,
            "role": "external", "on_common_support": support, "predictions": predictions}


class FocusedChecks(unittest.TestCase):
    def test_fold_fit_boundary_and_features_only_state(self):
        rng = np.random.RandomState(42)
        x = rng.normal(size=(72, 12))
        sites = np.repeat(np.arange(6).astype(str), 12)
        age = 30 + 2 * x[:, 0] + rng.normal(size=len(x))
        bundle = SimpleNamespace(splits=[{"fit": np.arange(72)}], sites=sites, age=age)
        recorded = []
        original = diagnostic.DirectionMaps.fit

        def traced_fit(instance, features, labels):
            recorded.append((features.copy(), labels.copy()))
            return original(instance, features, labels)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(diagnostic.DirectionMaps, "fit", traced_fit):
                diagnostic.inner_fold(bundle, x, "fixture", 0, 0, Path(directory))
        self.assertEqual(len(recorded), 1)
        np.testing.assert_array_equal(recorded[0][0], x[12:])
        np.testing.assert_array_equal(recorded[0][1], sites[12:])
        fitted = diagnostic.DirectionMaps().fit(x[12:], sites[12:])
        before = pickle.dumps(fitted)
        scores, _ = fitted.transform(x[:12] + 100)
        self.assertEqual(before, pickle.dumps(fitted))
        self.assertTrue(all(v.shape == (12, 4) for v in scores.values()))

    def test_restored_geometry_at_every_prespecified_alpha(self):
        rng = np.random.RandomState(43)
        train = rng.normal(size=(96, 12))
        held = rng.normal(size=(13, 12))
        sites = np.repeat(np.arange(6).astype(str), 16)
        age = 40 + 3 * train[:, 1] + rng.normal(size=len(train))
        fitted = diagnostic.DirectionMaps().fit(train, sites)
        _, geometry = fitted.transform(train, include_geometry=True)
        _, test_geometry = fitted.transform(held, include_geometry=True)
        require = np.testing.assert_allclose
        require(geometry["restored"] @ geometry["restored"].T,
                geometry["raw"] @ geometry["raw"].T, atol=1e-9)
        require(test_geometry["restored"] @ geometry["restored"].T,
                test_geometry["raw"] @ geometry["raw"].T, atol=1e-9)
        raw, _ = diagnostic.legacy.ridge_grid(geometry["raw"], age, test_geometry["raw"], diagnostic.ALPHAS)
        restored, _ = diagnostic.legacy.ridge_grid(geometry["restored"], age, test_geometry["restored"], diagnostic.ALPHAS)
        require(raw, restored, atol=1e-9)

    def test_fit_defined_support_and_visible_insufficient_domains(self):
        ages = np.array([5.] * 25 + [10.] * 20 + [100.] * 25)
        sites = np.concatenate((np.repeat(np.arange(5), 5), np.repeat(np.arange(4), 5), np.repeat(np.arange(5), 5))).astype(str)
        support = diagnostic.fit_support(ages, sites)
        self.assertEqual(support["eligible_bins"], [1, 19])
        np.testing.assert_array_equal(diagnostic.age_bins(np.array([-1., 0., 4.999, 5., 95., 100., 100.1])), [-1, 0, 0, 1, 19, 19, -1])
        rows = [appearance(str(i), i, i + 1) for i in range(10)]
        rows += [appearance("small", 20, 21, domain="small")]
        rows += [appearance("outside", 21, 22, domain="outside", support=False)]
        summary = diagnostic.summarize_population(rows, common=True, intervals=False)
        self.assertEqual(summary["centered"]["included_domains"], ["a"])
        self.assertEqual(summary["absolute_on_centered_eligible_domains"]["included_domains"], ["a"])
        self.assertEqual(set(summary["absolute"]["included_domains"]), {"a", "small"})
        self.assertEqual(set(summary["centered"]["excluded_domains"]), {"small", "outside"})
        self.assertEqual(summary["domains"]["outside"]["participants"], 0)
        self.assertEqual(summary["domains"]["small"]["status"], "insufficient_common_centered_support")
        self.assertEqual(diagnostic.choose_alpha([[1.] * 5])[0], 4)

    def test_appearance_losses_and_bootstrap_recentering_are_paired(self):
        rows = [appearance("repeat", 0, 10, view=0), appearance("repeat", 0, -10, view=1),
                appearance("single", 20, 20)]
        group = diagnostic.group_arrays(rows)
        index = diagnostic.MODELS.index("discarded_site")
        self.assertAlmostEqual(group["absolute"][index], 5.)
        self.assertAlmostEqual(group["centered"][index], 5.)
        # Two residuals {0,10}: sampling one participant twice gives zero
        # centered error. Bootstrapping frozen centered losses would stay at5.
        first = diagnostic.group_arrays([appearance("p", 0, 0), appearance("q", 0, 10)])
        second = deepcopy(first)
        boot = diagnostic.bootstrap_models([first, second], centered=True, replicates=128)
        self.assertIsNotNone(boot)
        self.assertEqual(float(np.min(boot[:, index])), 0.)
        self.assertTrue(np.all(boot[:, index] <= 5.))
        np.testing.assert_array_equal(boot[:, index], boot[:, diagnostic.MODELS.index("discarded_random")])
        # A scalar paired reference also checks repeated domain occurrences:
        # each occurrence gets a fresh participant draw and accumulates.
        rng = np.random.RandomState(diagnostic.SEED)
        domain_draws = rng.randint(2, size=(128, 2))
        self.assertTrue(np.any(domain_draws[:, 0] == domain_draws[:, 1]))
        expected = np.zeros_like(boot)
        for domain in range(2):
            for replicate, _ in np.argwhere(domain_draws == domain):
                multiplicity = rng.multinomial(2, [0.5, 0.5])
                weight = multiplicity / 2.0
                center = weight @ first["residuals"]
                expected[replicate] += np.sum(weight[:, None] * np.abs(first["residuals"] - center), axis=0) / 2.0
        np.testing.assert_allclose(boot, expected, atol=1e-12)
        # The centered-zero reference removes the target-domain mean even
        # when the absolute baseline's outer-training median differs by view.
        zero = diagnostic.MODELS.index("centered_zero")
        self.assertAlmostEqual(group["centered"][zero], 10.)
        self.assertIsNone(diagnostic.bootstrap_models([first], centered=True, replicates=2))


if __name__ == "__main__":
    diagnostic.legacy.compute_only()
    with threadpool_limits(limits=1):
        unittest.main(verbosity=2)
