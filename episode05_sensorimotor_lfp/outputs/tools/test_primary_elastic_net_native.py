"""Scientific and engineering equivalence checks for the single-sweep C kernel.

The coordinator builds the reviewed kernel and runs these only on compute.
"""
import ctypes
import unittest
from unittest.mock import patch

import numpy as np

import primary_elastic_net as original
import primary_elastic_net_native as native
from test_primary_elastic_net import enumerated_solution, example


def objective(gram, cross, coefficients, penalty, scale):
    return (.5*np.sum(coefficients*(gram@coefficients))-np.sum(coefficients*cross)
            +.5*penalty*np.sum(coefficients**2)+penalty*scale*np.abs(coefficients).sum())


class NativeElasticNetTest(unittest.TestCase):
    def test_one_sweep_matches_sequential_physical_order_from_nonzero_start(self):
        g = np.array([[1., .4, -.2], [.4, 1.3, .3], [-.2, .3, .8]])
        c = np.array([[.7, -.4], [-.2, 1.3], [.9, -.8]])
        start = np.array([[.4, -.1], [-.6, .3], [.2, -.5]])
        expected, actual = start.copy(), start.copy()
        penalty, scale = .1, .7
        for electrode in range(3):
            partial = c[electrode]-g[electrode]@expected+g[electrode, electrode]*expected[electrode]
            expected[electrode] = np.sign(partial)*np.maximum(np.abs(partial)-penalty*scale, 0)/(g[electrode, electrode]+penalty)
        before_g, before_c = g.copy(), c.copy()
        native.cyclic_sweep(g, c, penalty, scale, actual)
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-15)
        np.testing.assert_array_equal(g, before_g)
        np.testing.assert_array_equal(c, before_c)
        # Updating every channel from the old W (Jacobi) would be different.
        partial = c-g@start+np.diag(g)[:, None]*start
        simultaneous = np.sign(partial)*np.maximum(np.abs(partial)-penalty*scale, 0)/(np.diag(g)[:, None]+penalty)
        self.assertGreater(np.linalg.norm(actual-simultaneous), .01)

    def test_correlated_path_matches_original_and_independent_sign_oracle(self):
        gram = np.array([[1., .85, -.25], [.85, 1.1, -.1], [-.25, -.1, .8]])
        cross = np.array([[1.2, -.2], [.8, 1.6], [-.9, .45]])
        scale = .7
        old, old_info = original.solve_path(gram, cross, scale)
        new, new_info = native.solve_path(gram, cross, scale)
        for i, penalty in enumerate(original.LAMBDAS):
            exact = enumerated_solution(gram, cross, penalty, scale)
            for coefficients, info in ((old[i], old_info[i]), (new[i], new_info[i])):
                certificate = original.kkt_certificate(gram, cross, coefficients, penalty, scale)
                self.assertTrue(certificate["converged"])
                self.assertEqual(info["KKT_tolerance"], certificate["KKT_tolerance"])
                gap = objective(gram, cross, coefficients, penalty, scale)-objective(gram, cross, exact, penalty, scale)
                self.assertLessEqual(gap, certificate["objective_error_upper_bound"]+2e-12)
                # Strong convexity bounds distance to the unique optimum.
                self.assertLessEqual(np.linalg.norm(coefficients-exact),
                    certificate["KKT_residual_Frobenius_norm"]/penalty+2e-12)
            bound = (old_info[i]["KKT_residual_Frobenius_norm"]
                     +new_info[i]["KKT_residual_Frobenius_norm"])/penalty
            self.assertLessEqual(np.linalg.norm(new[i]-old[i]), bound+4e-12)
            self.assertEqual(new_info[i]["lambda"], penalty)
            self.assertEqual(new_info[i]["sweep_backend"], native.BACKEND)
            self.assertEqual(new_info[i]["maximum_sweeps"], 10000)

    def test_zero_rank_deficient_and_noncontiguous_inputs_keep_original_solution(self):
        gram = np.array([[1., 1., 0.], [1., 1., 0.], [0., 0., 0.]])
        cross = np.array([[.3, -.4], [.3, -.4], [0., 0.]])
        # Transposes and sliced covariance arrays must be normalized safely
        # before passing a contiguous float64 pointer to the C ABI.
        cross_storage = np.zeros((3, 4))
        cross_storage[:, ::2] = cross
        fitted, info = native.solve_path(gram.T, cross_storage[:, ::2], 0.)
        for i, penalty in enumerate(original.LAMBDAS):
            exact = np.linalg.solve(gram+penalty*np.eye(3), cross)
            bound = info[i]["KKT_residual_Frobenius_norm"]/penalty
            self.assertLessEqual(np.linalg.norm(fitted[i]-exact), bound+3e-12)
        zero, diagnostics = native.solve_path(gram, np.zeros_like(cross), 0.)
        np.testing.assert_array_equal(zero, np.zeros((5, 3, 2)))
        self.assertTrue(all(d["completed_sweeps"] == 1 and d["active_channel_union"] == 0 for d in diagnostics))

    def test_every_sweep_has_deadline_then_original_KKT_and_descending_warm_start(self):
        gram = np.array([[1., .7], [.7, 1.1]])
        cross = np.array([[.8, -.2], [.4, 1.3]])
        events, starts, endings = [], {}, {}
        sweep = native.cyclic_sweep
        certificate = native.kkt_certificate
        def check():
            events.append("deadline")
        def observed_sweep(g, c, penalty, scale, w):
            events.append("sweep")
            starts.setdefault(penalty, w.copy())
            sweep(g, c, penalty, scale, w)
        def observed_certificate(g, c, w, penalty, scale):
            events.append("certificate")
            result = certificate(g, c, w, penalty, scale)
            if result["converged"]:
                endings[penalty] = w.copy()
            return result
        self.assertIs(certificate, original.kkt_certificate)
        with patch.object(native, "cyclic_sweep", side_effect=observed_sweep), \
                patch.object(native, "kkt_certificate", side_effect=observed_certificate):
            _, diagnostics = native.solve_path(gram, cross, .7, check)
        count = sum(row["completed_sweeps"] for row in diagnostics)
        self.assertEqual(events, ["deadline", "sweep", "certificate"]*count)
        order = list(reversed(original.LAMBDAS))
        self.assertEqual(list(starts), order)
        np.testing.assert_array_equal(starts[order[0]], np.zeros((2, 2)))
        for before, after in zip(order, order[1:]):
            np.testing.assert_array_equal(starts[after], endings[before])

    def test_nonconvergence_and_deadline_abort_without_accepting_partial_path(self):
        gram = np.array([[1., .9], [.9, 1.]])
        cross = np.array([[2., 1.2], [1., 2.]])
        with self.assertRaisesRegex(RuntimeError, "KKT"):
            native.solve_path(gram, cross, .1, max_sweeps=1)
        with self.assertRaises(ValueError):
            native.solve_path(gram, cross, .1, max_sweeps=10001)
        class DeadlineReached(Exception):
            pass
        checks = []
        def check():
            checks.append(1)
            if len(checks) == 3:
                raise DeadlineReached()
        with patch.object(native, "cyclic_sweep", wraps=native.cyclic_sweep) as sweeps:
            with self.assertRaises(DeadlineReached):
                native.solve_path(gram, cross, .1, check)
        self.assertEqual(sweeps.call_count, 2)

    def test_C_ABI_rejects_bad_shapes_dtypes_readonly_and_nonfinite_arithmetic(self):
        gram, cross, w = np.eye(2), np.ones((2, 2)), np.zeros((2, 2))
        with self.assertRaises(ValueError):
            native.cyclic_sweep(gram, cross, .1, .2, np.zeros((2, 1)))
        with self.assertRaises((TypeError, ctypes.ArgumentError)):
            native.cyclic_sweep(gram.astype(np.float32), cross, .1, .2, w)
        readonly = w.copy()
        readonly.flags.writeable = False
        with self.assertRaises((TypeError, ctypes.ArgumentError)):
            native.cyclic_sweep(gram, cross, .1, .2, readonly)
        poisoned = cross.copy()
        poisoned[0, 0] = np.nan
        with self.assertRaises(ArithmeticError):
            native.cyclic_sweep(gram, poisoned, .1, .2, w.copy())
        for penalty, scale in ((0., .2), (.1, -1.), (.1, np.nan)):
            with self.subTest(penalty=penalty, scale=scale), self.assertRaises(ArithmeticError):
                native.cyclic_sweep(gram, cross, penalty, scale, w.copy())

    def test_synthetic_native_ID_banks_preserve_controls_predictions_and_scoring(self):
        x, y, labels, ids = example(85011)
        old_banks, old_baseline, old_predictions = original.fit_banks(x, y, labels, ids, "example")
        new_banks, new_baseline, new_predictions = native.fit_banks(x, y, labels, ids, "example")
        np.testing.assert_array_equal(new_baseline, old_baseline)
        for recipe in original.RECIPES:
            np.testing.assert_allclose(new_predictions[recipe], old_predictions[recipe], rtol=2e-7, atol=2e-7)
            for key in ("recipe", "lambdas", "baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
                self.assertEqual(new_banks[recipe][key], old_banks[recipe][key])
            for old_row, new_row in zip(old_banks[recipe]["lambda_rows"], new_banks[recipe]["lambda_rows"]):
                self.assertEqual(new_row.keys(), old_row.keys())
                for key in old_row:
                    np.testing.assert_allclose(new_row[key], old_row[key], rtol=3e-8, atol=2e-8)
        old_metadata = old_banks[original.RECIPES[0]]["direction_fit_metadata"]
        new_metadata = new_banks[original.RECIPES[0]]["direction_fit_metadata"]
        for before, after in zip(old_metadata, new_metadata):
            for key in ("held_direction", "physical_mean", "physical_sd", "physical_zero_SD_columns",
                        "common_velocity_penalty_scale", "training_trial_maps_by_replicate"):
                self.assertEqual(before[key], after[key])
            diagnostic_groups = [after["paired_fit_diagnostics_by_lambda"],
                                 *after["control_fit_diagnostics_by_replicate_lambda"]]
            self.assertEqual(sum(map(len, diagnostic_groups)), 45)
            for group in diagnostic_groups:
                self.assertTrue(all(row["converged"] for row in group))
                self.assertTrue(all(row["maximum_KKT_residual"] <= row["KKT_tolerance"] for row in group))


if __name__ == "__main__":
    unittest.main()
