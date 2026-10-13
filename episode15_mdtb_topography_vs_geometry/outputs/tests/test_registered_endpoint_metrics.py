"""Small deterministic identities for selected metrics; no scientific worlds."""
from itertools import combinations
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from registered_endpoint_metrics import (
    CONDITIONS, DOMAINS, condition_weights, crossvalidated_geometries,
    geometry_covariance, geometry_inner, geometry_weights, map_covariance,
    oriented_contrasts, project_gain_direction,
)


class RegisteredEndpointMetricsTests(unittest.TestCase):
    def test_complete_condition_and_domain_pair_weights(self):
        self.assertEqual(len(CONDITIONS), 18)
        self.assertEqual(len(set(CONDITIONS)), 18)
        labels = [g for g, domain in enumerate(DOMAINS) for _ in domain]
        cw, gw = condition_weights(), geometry_weights()
        for g in range(7):
            idx = np.flatnonzero(np.asarray(labels) == g)
            self.assertAlmostEqual(cw[idx].sum(), 1 / 7)
            for h in range(g, 7):
                block = 0.
                for c in range(18):
                    for d in range(c, 18):
                        if (labels[c], labels[d]) == (g, h):
                            block += gw[c, d] if c == d else 2 * gw[c, d]
                self.assertAlmostEqual(block, 1 / 28)
        self.assertEqual(np.count_nonzero(np.triu(gw)), 171)
        self.assertTrue(np.all(np.diag(gw) > 0))
        self.assertAlmostEqual(gw.sum(), 1.)
        np.testing.assert_array_equal(gw, gw.T)

    def test_crossvalidated_pairing_spatial_centering_and_signed_diagonal(self):
        pattern = np.arange(1., 19.)[:, None] * np.array([-1., 0., 1.])
        maps = np.stack((pattern, -pattern, 2 * pattern, 3 * pattern))[None]
        actual = crossvalidated_geometries(maps)
        outer = pattern @ pattern.T / 3
        np.testing.assert_allclose(actual[0, 0], -outer, rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual[0, 1], 6 * outer, rtol=0, atol=1e-12)
        self.assertTrue(np.all(np.diag(actual[0, 0]) < 0))
        offsets = np.arange(72.).reshape(1, 4, 18, 1)
        np.testing.assert_allclose(crossvalidated_geometries(maps + offsets), actual,
                                   rtol=0, atol=1e-12)

    def test_gain_projection_uses_selected_metric_and_each_fold_source(self):
        source = np.eye(18)
        signal = np.diag(np.arange(1., 19.))
        projected = project_gain_direction(signal, source)
        self.assertAlmostEqual(float(geometry_inner(projected, source)), 0., places=13)
        np.testing.assert_allclose(project_gain_direction(projected, source), projected,
                                   rtol=0, atol=2e-14)
        # Domain imbalance makes this different from a uniform/Frobenius projection.
        self.assertGreater(np.max(np.abs(projected - (signal - signal.trace() / 18 * source))), .1)
        sources = np.stack((source, np.diag(np.arange(1., 19.))))[:, None]
        matrices = np.stack((signal, signal[::-1, ::-1]))[:, None] + np.zeros((2, 2, 18, 18))
        result = project_gain_direction(matrices, sources)
        np.testing.assert_allclose(geometry_inner(result, sources), 0., rtol=0, atol=2e-13)
        for i in range(2):
            np.testing.assert_allclose(result[i], project_gain_direction(matrices[i], sources[i]),
                                       rtol=0, atol=0)

    def test_map_contraction_matches_explicit_participant_pairs(self):
        a, b = np.array([1., 3., 5.]), np.array([2., 1., 6.])
        pattern = np.arange(1., 19.)[:, None] * np.array([-1., 0., 1.])
        x, y = a[:, None, None] * pattern, b[:, None, None] * pattern
        metric_energy = np.sum(np.mean(pattern ** 2, axis=-1) * condition_weights())
        self.assertAlmostEqual(map_covariance(x, y), 4 * metric_energy)
        pair_sum = sum(np.sum(np.mean((x[i] - x[j]) * (y[i] - y[j]), axis=-1)
                              * condition_weights()) for i, j in combinations(range(3), 2)) / 6
        self.assertAlmostEqual(map_covariance(x, y), pair_sum)
        offsets = np.arange(54.).reshape(3, 18, 1)
        self.assertAlmostEqual(map_covariance(x + offsets, y - 2 * offsets), pair_sum)
        self.assertLess(map_covariance(x, -x), 0.)

    def test_projected_geometry_covariance_and_exact_residual_removal(self):
        source = np.eye(18)
        direction = project_gain_direction(np.diag(np.arange(18.)), source)
        x = np.array([1., 3., 5.])[:, None, None] * direction
        y = np.array([2., 1., 6.])[:, None, None] * direction
        expected = 4 * float(geometry_inner(direction, direction))
        self.assertAlmostEqual(geometry_covariance(x, y), expected)
        pair_sum = sum(float(geometry_inner(x[i] - x[j], y[i] - y[j]))
                       for i, j in combinations(range(3), 2)) / 6
        self.assertAlmostEqual(geometry_covariance(x, y), pair_sum)
        self.assertEqual(geometry_covariance(x - x, x - x), 0.)
        self.assertLess(geometry_covariance(x, -x), 0.)

    def contrast_inputs(self):
        return dict(v_map=10., v_geometry=5., s_map=2., s_geometry=3.,
                    r_map=dict(M1=4., M2=3., M3=2., M4=1.),
                    r_geometry=dict(M1=3., M2=2., M3=1., M4=.5),
                    margins=dict(tau_signal=.2, tau_remaining=.1, tau_increment=.05,
                                 tau_geometry=.3, tau_boundary=.4, tau_alignment=.5,
                                 tau_ablation=.6),
                    boundary=dict(V=8., R=2., S=3.), interior=dict(V=4., R=1., S=2.),
                    alignment_advantage=.25,
                    ablation_gains={"mode0": [1., 2.], "mode1": [-1., 0.]})

    def test_all_oriented_primitives_preserve_both_ablations_and_negative_signs(self):
        out = oriented_contrasts(**self.contrast_inputs())
        self.assertEqual(len(out), 18 + 2 * 2)
        self.assertAlmostEqual(out["G_map"], 9.6)
        self.assertAlmostEqual(out["G_geom"], 4.1)
        self.assertEqual(out["A_map,M1"], -3.)
        self.assertNotIn("A_geom,M3", out)
        self.assertAlmostEqual(out["I_geom,M4|M3"], .25)
        self.assertAlmostEqual(out["M1_interior_adequacy"], -.6)
        self.assertAlmostEqual(out["M1_boundary_enrichment"], .1)
        self.assertAlmostEqual(out["M2_alignment_advantage"], -.25)
        self.assertAlmostEqual(out["M4_ablation,mode1,replicate1"], -1.6)
        self.assertAlmostEqual(out["M4_ablation,mode1,replicate2"], -.6)
        self.assertEqual(sum(key.startswith("I_") for key in out), 6)

    def test_missing_signed_support_scales_or_margins_are_not_repaired(self):
        asymmetric = np.eye(18); asymmetric[0, 1] = 1
        with self.assertRaises(ValueError):
            geometry_inner(asymmetric, np.eye(18))
        with self.assertRaises(ValueError):
            project_gain_direction(np.eye(18), np.zeros((18, 18)))
        with self.assertRaises(ValueError):
            crossvalidated_geometries(np.zeros((3, 4, 17, 2)))
        with self.assertRaises(ValueError):
            map_covariance(np.full((3, 18, 2), np.nan), np.zeros((3, 18, 2)))
        for change in ("margin", "scale", "replicate", "class", "nonfinite"):
            args = self.contrast_inputs()
            if change == "margin":
                del args["margins"]["tau_signal"]
            elif change == "scale":
                args["boundary"]["S"] = 0
            elif change == "replicate":
                args["ablation_gains"]["mode1"] = [1.]
            elif change == "class":
                del args["r_geometry"]["M3"]
            else:
                args["alignment_advantage"] = np.inf
            with self.subTest(change=change), self.assertRaises(ValueError):
                oriented_contrasts(**args)


if __name__ == "__main__":
    unittest.main()
