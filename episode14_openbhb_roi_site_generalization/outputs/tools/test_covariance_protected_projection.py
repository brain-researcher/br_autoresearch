"""Focused mathematical and population checks; run only in owned Slurm."""
import unittest
from types import SimpleNamespace

import numpy as np

import run_covariance_protected_projection as run


class CovarianceProtectionTests(unittest.TestCase):
    def test_exact_protection_rank_and_features_only_transform(self):
        rng = np.random.RandomState(17)
        z = rng.normal(size=(80, 12))
        y = z[:, 0] + 2 * z[:, 7] + rng.normal(size=80)
        original = np.eye(4, 12)
        c = run.covariance(z, y)
        v, checks = run.protected_directions(original, c)
        self.assertEqual(checks["projected_row_rank"], 4)
        np.testing.assert_allclose(v @ v.T, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(v @ c, np.zeros(4), atol=1e-12)
        np.testing.assert_allclose(run.covariance(run.project(z, v), y), c, atol=1e-12)
        held = rng.normal(size=(10, 12))
        before = v.copy()
        transformed = run.project(held, v)
        np.testing.assert_array_equal(v, before)
        self.assertEqual(transformed.shape, held.shape)
        meta = run.direction_metadata(z, v, original, c)
        direct = np.linalg.norm(v.T @ v - original.T @ original, ord="fro")
        self.assertAlmostEqual(meta["projector_frobenius_displacement_vs_original"], direct, places=12)
        self.assertLess(meta["removed_age_covariance_squared_norm_fraction"], 1e-25)

    def test_rank_loss_explicitly_fails_and_sham_preserves_sites(self):
        with self.assertRaises(run.prior.SupportFailure):
            run.protected_directions(np.eye(4, 9), np.eye(9)[0])
        ages = np.arange(30, dtype=float)
        sites = np.repeat(["a", "b", "c"], 10)
        sham = run.source.within_site_sham(ages, sites)
        self.assertFalse(np.array_equal(ages, sham))
        for site in set(sites):
            np.testing.assert_array_equal(np.sort(ages[sites == site]), np.sort(sham[sites == site]))

    def test_exact_borrowed_rows_and_fit_complement(self):
        bundle = SimpleNamespace(ids=np.array(["a", "b", "c", "d"]),
                                 age=np.array([20., 25., 30., 35.]), sites=np.array(["1", "1", "1", "2"]),
                                 splits=[{"fit": np.array([0, 1]), "internal": np.array([2]), "external": np.array([3])}])
        support = {"eligible_bins": [6, 7]}
        rows = [{"id": "c", "domain": "1", "view": 0, "role": "internal", "age": 30., "age_bin": 6,
                 "on_common_support": True},
                {"id": "d", "domain": "2", "view": 0, "role": "external", "age": 35., "age_bin": 7,
                 "on_common_support": True}]
        record = {"view": 0, "support": support, "appearances": rows}
        np.testing.assert_array_equal(run.verify_cached_rows(record, bundle, 0, support), [2, 3])
        rows[0]["id"] = "a"
        with self.assertRaises(ValueError):
            run.verify_cached_rows(record, bundle, 0, support)

    def test_same_support_and_participant_balancing_with_explicit_models(self):
        rows = []
        for domain, n in (("large", 12), ("small", 3), ("empty", 2)):
            for i in range(n):
                for view in range(2 if i == 0 else 1):
                    age = 20. + i * 2
                    residual = float(i + 3 * view)
                    rows.append({"id": f"{domain}{i}", "domain": domain, "age": age,
                                 "on_common_support": domain != "empty", "view": view,
                                 "predictions": {name: age + residual + j for j, name in enumerate(run.MODELS)}})
        panel = run.summarize_population(rows, True, False)
        self.assertEqual(panel["absolute"]["included_domains"], ["large", "small"])
        self.assertEqual(panel["centered"]["included_domains"], ["large"])
        self.assertEqual(panel["absolute_on_centered_eligible_domains"]["included_domains"], ["large"])
        self.assertEqual(panel["domains"]["empty"]["participants"], 0)
        selected = [r for r in rows if r["domain"] == "large"]
        group = run.source.group_arrays(selected, models=run.MODELS)
        self.assertAlmostEqual(group["absolute"][0], (sum(range(12)) + 1.5) / 12)
        boot = run.source.bootstrap_models([group, group], True, models=run.MODELS, replicates=20)
        # All arms differ by offsets only, which each bootstrap domain recenters.
        np.testing.assert_allclose(boot - boot[:, :1], 0, atol=1e-12)


if __name__ == "__main__":
    run.legacy.compute_only()
    unittest.main()
