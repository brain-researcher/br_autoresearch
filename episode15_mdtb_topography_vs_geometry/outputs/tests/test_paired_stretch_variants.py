"""Five bounded M4 recipe/certificate fixtures; no participant data or sweep."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from development_lopo import (
    DEVELOPMENT_IDS, FoldContext, M4Prediction, M4_VARIANTS, SharedM3Source,
    exploratory_points, fit_development_lopo,
)
from fit_m4_task_a import PairedM4TaskAObjective
from local_anatomical_basis import BasisTolerances, build_local_anatomical_basis
from local_functional_source import LocalFunctionalSource, LocalSourceGeometry, TASK_A_CONDITIONS
import paired_stretch_variants as stretch
from registered_endpoint_metrics import (
    condition_weights, crossvalidated_geometries, geometry_covariance,
    map_covariance, project_gain_direction,
)
from spatial_deformation_operators import apply_m3
import test_development_lopo as lopo_fixture


class PairedStretchVariantsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        coordinates = 3*np.indices((13, 4, 4)).reshape(3, -1).T.astype(float)
        tolerance = BasisTolerances(eigen_tie_atol=1e-10, eigen_tie_rtol=1e-10,
                                    rank_rtol=1e-8, numerical_atol=1e-10)
        cls.anatomy = build_local_anatomical_basis(coordinates, np.arange(500, 708), tolerances=tolerance)
        cls.recipe = LocalFunctionalSource(cls.anatomy, numerical_atol=1e-10, rank_rtol=1e-8)
        cls.B = cls.anatomy.B
        cls.specifications = stretch.stretch_specifications()
        cls.context = FoldContext(DEVELOPMENT_IDS[0], DEVELOPMENT_IDS[1:], None)
        complement = cls.anatomy.cell_labels.astype(float)
        complement -= complement.mean()
        cls.complement = complement/np.linalg.norm(complement)
        cls.X = np.sqrt(29.)*np.eye(29, 16)@cls.B.T+4*cls.complement[None]
        maps = (np.eye(18, 16)+.1)@cls.B.T
        cls.partitions = np.stack([(1+k/10)*maps+(k*k/25)*np.arange(1., 19.)[:, None]*cls.complement
                                   for k in range(4)])
        K0 = crossvalidated_geometries(cls.partitions[None])[0].mean(axis=0)
        halves = cls.partitions.reshape(2, 2, 18, 208).mean(axis=1)
        scale = float(np.sum(condition_weights()*np.mean(halves[0]*halves[1], axis=-1)))
        state = LocalSourceGeometry(cls.partitions, cls.X,
                                    tuple(np.eye(16) for _ in cls.context.training_ids),
                                    tuple({} for _ in cls.context.training_ids), cls.context.training_ids)
        cls.source = SharedM3Source(cls.context.training_ids, TASK_A_CONDITIONS, cls.recipe.voxel_ids,
                                    cls.X, cls.partitions.mean(axis=0), K0, scale, cls.B,
                                    cls.recipe.weights, True, 1e-8, 1e-10, state)
        cls.C, cls.O = np.zeros((16, 16)), np.zeros((16, 16))
        for k, (start, stop) in enumerate(stretch.BLOCKS):
            angle = .05*(1+k/16)
            cls.O[start:stop, start:stop] = [[np.cos(angle), -np.sin(angle)],
                                           [np.sin(angle), np.cos(angle)]]
            cls.C[start:stop, start:stop] = np.diag([.004*(k+1)/8, -.004*(k+1)/8])
        # Choose only the orthogonal-complement target amplitude to make the
        # frozen pre-operator common gain exactly the desired scalar. The
        # active target is the known, noncommuting paired O exp(C) transform.
        gain = 1.7
        active_target = gain*np.sqrt(29.)*np.eye(29, 16)@cls.O@expm(cls.C)@cls.B.T
        complement_multiplier = gain*(32-np.trace(cls.O@expm(cls.C)))/16
        cls.Y = active_target+complement_multiplier*4*cls.complement[None]
        cls.reference = cls.recipe.calibrate_m3(cls.source, cls.Y, cls.context)

    def callback(self, check=lambda: None):
        return stretch.PairedStretchVariants(self.recipe, self.anatomy, check=check)

    def exact_numerical_result(self, X, Y, B, O, **settings):
        self.assertEqual(settings.pop("max_iterations"), 500)
        self.assertEqual(settings.pop("optimizer_ftol"), 1e-12)
        rho = settings.pop("spectral_norm_bound")
        self.assertIn(rho, (stretch.REFERENCE_RHO, stretch.STRESS_RHO))
        self.assertEqual(settings["rank_bound"], 16)
        np.testing.assert_array_equal(B, self.B)
        np.testing.assert_array_equal(O, self.reference["O"])
        self.assertEqual(settings["global_gain"], self.reference["gain"])
        objective = PairedM4TaskAObjective(X, Y, B, O, **settings)
        theta = np.einsum("pij,ij->p", settings["mode_matrices"], self.C)
        return {"C": objective.matrix(theta), "coefficients": theta, "O": O.copy(),
                "gain": settings["global_gain"], "iterations": 1, "local_optimizer_converged": True,
                "global_optimum_certified": False, "unique_solution_certified": False}

    def test_fixed_modes_are_orthonormal_trace_free_and_preserve_local_capacities(self):
        self.assertEqual(tuple(item.name for item in self.specifications), M4_VARIANTS)
        self.assertEqual(tuple(item.parameter_count for item in self.specifications), (15, 8, 16, 16))
        allowed = np.zeros((16, 16), dtype=bool)
        for start, stop in stretch.BLOCKS:
            allowed[start:stop, start:stop] = True
        for specification in self.specifications:
            M, p = specification.modes, specification.parameter_count
            np.testing.assert_allclose(M.reshape(p, -1)@M.reshape(p, -1).T, np.eye(p), atol=1e-14, rtol=0)
            np.testing.assert_allclose(np.trace(M, axis1=1, axis2=2), 0., atol=1e-14, rtol=0)
            np.testing.assert_array_equal(M, M.transpose(0, 2, 1))
            np.testing.assert_array_equal(M[:, ~allowed], 0.)
            self.assertEqual(np.linalg.matrix_rank(np.concatenate(list(M), axis=1)), 16)
            self.assertFalse(M.flags.writeable)
        np.testing.assert_array_equal(self.specifications[2].modes, self.specifications[3].modes)
        self.assertAlmostEqual(np.exp(self.specifications[0].spectral_norm_bound), 1.25)
        self.assertAlmostEqual(np.exp(self.specifications[3].spectral_norm_bound), 1.5)

    def test_curvature_and_gap_match_independent_analytic_oracles(self):
        rho = stretch.STRESS_RHO
        isotropic = stretch.block_curvature_bounds(np.eye(2), np.eye(2), global_gain=1., spectral_norm_bound=rho)
        self.assertLess(isotropic["residual_hessian_lower_bound"], 0)
        self.assertAlmostEqual(isotropic["refined_hessian_lower_bound"], 4/9, places=13)
        self.assertGreater(isotropic["hessian_lower_bound"], isotropic["required_positive_margin"])
        # For a non-isotropic, noncommuting block, directly evaluate the
        # directional second derivative of the unexpanded SSE at three fixed
        # directions. This is a tiny arithmetic fixture, not a fit/sweep.
        A = np.array([[1., .015], [0., 1.01], [.02, -.01]])
        Y = A@np.array([[1.005, .002], [.002, .995]])
        bounds = stretch.block_curvature_bounds(A, Y, global_gain=1., spectral_norm_bound=rho)
        C = np.array([[.04, .025], [.025, -.04]])
        step = 1e-4
        for H in (np.eye(2)/np.sqrt(2), np.diag([1., -1.])/np.sqrt(2),
                  np.array([[0., 1.], [1., 0.]])/np.sqrt(2)):
            def value(displacement):
                return np.sum((A@expm(C+displacement*H)-Y)**2)
            hessian = (value(step)-2*value(0)+value(-step))/step**2
            self.assertGreaterEqual(hessian+1e-6, bounds["hessian_lower_bound"])
        for specification in self.specifications:
            p = specification.parameter_count
            gradient = np.linspace(-.8, 1.1, p)
            theta = .003*np.sin(np.arange(p)+1)
            radius = np.sqrt(2)*specification.spectral_norm_bound
            if specification.name == M4_VARIANTS[0]:
                diagonal_gradient = np.diag(np.einsum("p,pij->ij", gradient, specification.modes))
                minimizer = np.empty(16)
                order = np.argsort(diagonal_gradient)
                minimizer[order[:8]] = specification.spectral_norm_bound
                minimizer[order[8:]] = -specification.spectral_norm_bound
                vertex = np.einsum("pij,ij->p", specification.modes, np.diag(minimizer))
            elif specification.name == M4_VARIANTS[1]:
                vertex = -radius*np.sign(gradient)
            else:
                pairs = gradient.reshape(8, 2)
                vertex = (-radius*pairs/np.linalg.norm(pairs, axis=1)[:, None]).ravel()
            self.assertAlmostEqual(stretch.first_order_gap(specification, theta, gradient),
                                   float(gradient@(theta-vertex)), places=12)

    def test_unverified_curvature_and_nonoptimal_fit_refuse_without_fallback(self):
        # Same positive pre-model gain=1, but active target amplitude4 and a
        # compensating complement make the sufficient curvature fail. The
        # unique M3 anchor itself is valid; no C fit may run after this refusal.
        bad_target = 4*np.sqrt(29.)*np.eye(29, 16)@self.B.T-8*self.complement[None]
        reference = self.recipe.calibrate_m3(self.source, bad_target, self.context)
        with patch.object(stretch, "fit_m4_task_a") as fit:
            with self.assertRaisesRegex(stretch.M4CertificateUnverified, "uniqueness curvature"):
                self.callback()(self.source, bad_target, reference, self.context)
            fit.assert_not_called()
        def not_optimal(X, Y, B, O, **settings):
            result = self.exact_numerical_result(X, Y, B, O, **settings)
            result["C"] = np.zeros((16, 16))
            result["coefficients"] = np.zeros_like(result["coefficients"])
            return result
        with patch.object(stretch, "fit_m4_task_a", side_effect=not_optimal) as fit:
            with self.assertRaisesRegex(stretch.M4CertificateUnverified, "optimality gap"):
                self.callback()(self.source, self.Y, self.reference, self.context)
            self.assertEqual(fit.call_count, 1)
            self.assertEqual(fit.call_args.kwargs["spectral_norm_bound"], stretch.REFERENCE_RHO)

    def test_prediction_keeps_reference_orientation_gain_and_actual_four_partition_geometry(self):
        calls = []
        originals = [item.copy() for item in (self.X, self.Y, self.B, self.partitions, self.reference["O"])]
        with patch.object(stretch, "fit_m4_task_a", side_effect=self.exact_numerical_result) as fit:
            predictions = self.callback(lambda: calls.append("check"))(self.source, self.Y, self.reference, self.context)
        self.assertEqual(fit.call_count, 4)
        self.assertGreaterEqual(len(calls), 9)
        self.assertEqual(tuple(item.name for item in predictions), M4_VARIANTS)
        self.assertGreater(np.linalg.norm(self.reference["O"]@expm(self.C)-expm(self.C)@self.reference["O"]), 1e-4)
        L = np.eye(208)+self.B@(self.reference["O"]@expm(self.C)-np.eye(16))@self.B.T
        expected_partitions = self.reference["gain"]*self.partitions@L
        expected_geometry = crossvalidated_geometries(expected_partitions[None])[0].mean(axis=0)
        for prediction in predictions:
            np.testing.assert_allclose(prediction.maps, self.reference["gain"]*self.source.G@L, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(prediction.geometry, expected_geometry, atol=1e-12, rtol=1e-12)
            self.assertGreater(np.linalg.norm(prediction.geometry-prediction.maps@prediction.maps.T/208), 1e-4)
            d = prediction.diagnostics
            self.assertTrue(d["unique_minimizer_sufficient_curvature"])
            self.assertTrue(d["solution_within_frozen_gap"])
            self.assertTrue(d["locality_certified"])
            self.assertFalse(d["global_optimum_certified"])
            self.assertFalse(d["exact_global_optimizer"])
            self.assertFalse(d["interval_arithmetic_proof"])
            self.assertFalse(d["native_stability_and_ablation_qualified"])
            self.assertLessEqual(d["first_order_gap"], d["required_gap_upper_bound"])
            self.assertAlmostEqual(d["coefficient_distance_upper_bound"],
                                   np.sqrt(2*d["first_order_gap"]/d["hessian_lower_bound"]))
            self.assertEqual(d["training_ids"], self.context.training_ids)
            self.assertEqual(d["source_partitions_used"], 4)
            np.testing.assert_array_equal(d["O"], self.reference["O"])
            self.assertEqual(d["gain"], self.reference["gain"])
        for current, before in zip((self.X, self.Y, self.B, self.partitions, self.reference["O"]), originals):
            np.testing.assert_array_equal(current, before)
        # The actual wrapper was exercised above. This inexpensive stub now
        # exercises its new LOPO record/callback/aggregation boundary using
        # the existing two-mode LOPO fixture; it is not another M4 model fit.
        fixture = lopo_fixture.DevelopmentLOPOTests()
        fixture.setUp()
        contexts = []
        def variants(source, target_a, numerical, context):
            contexts.append(context)
            self.assertEqual(target_a.shape, (3, 208))
            self.assertNotIn(context.target_id, source.training_ids)
            self.assertNotIn(context.deleted_id, source.training_ids)
            maps = numerical["gain"]*apply_m3(source.G, source.B, numerical["O"])
            geometry = fixture.geometry(source, numerical, context)
            return tuple(M4Prediction(name, (1+k/5)*maps, (1+k/5)**2*geometry, {})
                         for k, name in enumerate(M4_VARIANTS))
        def run(records, callback=variants):
            return fit_development_lopo(records, fit_shared=fixture.shared,
                calibrate_m3=fixture.calibrate, certify_m3=fixture.certificate,
                predict_m3_geometry=fixture.geometry, check=lambda: None,
                include_participant_deletions=False, predict_m4=callback).full
        full = run(fixture.records)
        self.assertEqual(len(contexts), 12)
        points = exploratory_points(full)
        observed = np.stack([record.task_b_partitions for record in fixture.records])
        halves = observed.reshape(12, 2, 2, 18, 208).mean(axis=2)
        observed_geometry = crossvalidated_geometries(observed)
        K0 = np.stack([fold.K0 for fold in full.folds])
        for k, name in enumerate(M4_VARIANTS):
            maps = np.stack([fold.m4_predictions[k].maps for fold in full.folds])
            geometry = np.stack([fold.m4_predictions[k].geometry for fold in full.folds])
            self.assertAlmostEqual(points["R_map"][name],
                map_covariance(halves[:, 0]-maps, halves[:, 1]-maps), places=10)
            residual = project_gain_direction(observed_geometry-geometry[:, None], K0[:, None])
            self.assertAlmostEqual(points["R_geometry"][name],
                geometry_covariance(residual[:, 0], residual[:, 1]), places=10)
        altered = list(fixture.records)
        altered[0] = replace(altered[0], task_b_partitions=-5*altered[0].task_b_partitions)
        after = run(altered)
        self.assertFalse(np.array_equal(full.folds[0].observed_partitions, after.folds[0].observed_partitions))
        for before_value, after_value in zip(full.folds[0].m4_predictions, after.folds[0].m4_predictions):
            np.testing.assert_array_equal(before_value.maps, after_value.maps)
            np.testing.assert_array_equal(before_value.geometry, after_value.geometry)
        with self.assertRaisesRegex(ValueError, "four.*M4"):
            run(fixture.records, callback=lambda *args: variants(*args)[:-1])

    def test_full_operator_and_reference_boundaries_and_cancellation(self):
        callback = self.callback()
        cross_cell = self.C.copy()
        cross_cell[0, 2] = cross_cell[2, 0] = .01
        with self.assertRaisesRegex(ValueError, "algebra/locality/spectral"):
            callback.operator_certificate(self.reference["O"], cross_cell, self.specifications[2])
        excessive = np.diag(np.tile([1., -1.], 8))
        with self.assertRaisesRegex(ValueError, "algebra/locality/spectral"):
            callback.operator_certificate(self.reference["O"], excessive, self.specifications[3])
        with self.assertRaisesRegex(ValueError, "unchanged reference"):
            stretch.PairedStretchVariants(self.recipe, replace(self.anatomy, B=self.B[:, ::-1]), check=lambda: None)
        with patch.object(stretch, "fit_m4_task_a") as fit:
            with self.assertRaisesRegex(ValueError, "actual current-fold"):
                callback(replace(self.source, training_ids=(self.context.target_id,)), self.Y, self.reference, self.context)
            with self.assertRaisesRegex(ValueError, "pre-operator target-A common gain"):
                callback(self.source, self.Y, {**self.reference, "gain": 2*self.reference["gain"]}, self.context)
            def cancelled():
                raise RuntimeError("caller deadline")
            with self.assertRaisesRegex(RuntimeError, "caller deadline"):
                self.callback(cancelled)(self.source, self.Y, self.reference, self.context)
            fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
