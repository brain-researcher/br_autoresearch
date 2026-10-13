"""Five focused fixed brain-CV checks, only in coordinated Slurm compute.

Small artificial matrices test source arithmetic and role/axis interfaces.
The sole actual CV fit uses897 rows but only3 source/2 target coordinates.
No tau payload, empirical reader, decoder, endpoint or scheduler is accessed.
"""
import ast
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import brain_loss as brain


class BrainLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute ONLY the actual pinned source's tiny no-selection branch in
        # these fixture calls; do not import its optional package dependencies.
        wheel = ROOT / "outputs/source_pipeline_binding/fastl2lir_0_9/fastl2lir-0.9-py2.py3-none-any.whl"
        data = wheel.read_bytes()
        if hashlib.sha256(data).hexdigest() != "2abd8f46c8faa69aaea4e56568c8c4048178c24cc1ea88ad7c33f661151a0412":
            raise AssertionError("pinned FastL2LiR0.9 source required")
        with zipfile.ZipFile(wheel) as archive:
            code = archive.read("fastl2lir/fastl2lir.py").decode("utf-8")
        tree = ast.parse(code)
        definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FastL2LiR")
        fit = next(node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == "__sub_fit")
        fit.name = "source_fit"
        namespace = {"np": np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fit], type_ignores=[])),
                     "PINNED_fastl2lir_0_9_no_selection", "exec"), namespace)
        cls.pinned_fit = staticmethod(namespace["source_fit"])

    def fixture(self):
        ids = tuple(f"image-{index:04d}" for index in range(897))
        u = np.linspace(-2., 3., 897, dtype=np.float32)
        x = np.column_stack((u, np.sin(2 * u), np.cos(3 * u))).astype(np.float32)
        y = np.column_stack((.4 * u + np.sin(u), -.7 * u + np.cos(2 * u))).astype(np.float32)
        order = np.random.RandomState(19).permutation(897)
        source = brain.TauTrainResponses("sub-01", tuple(ids[index] for index in order), (12, 3, 8), x[order])
        target = brain.TauTrainResponses("sub-03", ids[::-1], (900, 4), y[::-1])
        return ids, x, y, source, target

    def pinned_matrix(self, x, y, alpha):
        weights, bias = self.pinned_fit(None, x, y, alpha=alpha, n_feat=x.shape[1],
                                       use_all_features=True, dtype=np.float32)
        return np.vstack((weights, bias))

    def test_common_fold_membership_exact_ID_join_and_fixed_native_axes(self):
        ids, x, y, source, target = self.fixture()
        state = np.random.get_state()
        folds = brain.make_common_folds(ids[::-1])
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        self.assertEqual(folds.image_ids, tuple(sorted(ids, key=lambda value: value.encode())))
        shuffled = np.random.RandomState(42).permutation(897)
        expected = tuple(tuple(sorted(int(index) for index in shuffled[fold::5])) for fold in range(5))
        self.assertEqual(folds.validation_indices, expected)
        self.assertEqual(tuple(map(len, expected)), (180, 180, 179, 179, 179))
        self.assertEqual(sorted(index for fold in expected for index in fold), list(range(897)))
        np.testing.assert_array_equal(brain._aligned_values(source, folds, (12, 3, 8))[0], x)
        np.testing.assert_array_equal(brain._aligned_values(target, folds, (900, 4))[0], y)
        for changed in (replace(source, role="tau_test"),
                        replace(source, aggregation="unaveraged_presentations"),
                        replace(source, image_ids=source.image_ids[:-1]),
                        replace(source, image_ids=(source.image_ids[1],) + source.image_ids[1:]),
                        replace(source, image_ids=("foreign-image",) + source.image_ids[1:]),
                        replace(source, native_voxel_ids=(3, 12, 8)),
                        replace(source, native_voxel_ids=(12, 3, True))):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                brain._aligned_values(changed, folds, (12, 3, 8))
        with self.assertRaisesRegex(ValueError, "common_tau_train_fold"):
            brain._validate_folds(replace(folds, validation_indices=folds.validation_indices[::-1]))

    def test_actual_pinned_primal_penalized_bias_and_float64_prediction_promotion(self):
        # More features than rows still requires a feature-space primal solve.
        x = np.array([[1., 2., -1., .5], [2., -1., .5, 3.], [3., .25, 2., -2.]], dtype=np.float32)
        y = np.array([[3., -1.], [6., 4.], [2., 7.]], dtype=np.float32)
        gram, right = brain._primal_system(x, y, lambda: None)
        solve = np.linalg.solve
        with mock.patch.object(np.linalg, "solve", wraps=solve) as called:
            actual = brain._solve_primal(gram, right, 620, lambda: None)
        self.assertEqual(called.call_args.args[0].shape, (5, 5))
        self.assertEqual(called.call_args.args[0].dtype, np.float32)
        np.testing.assert_array_equal(actual, self.pinned_matrix(x, y, 620))
        self.assertEqual(actual.dtype, np.float32)
        without_bias_penalty = gram + 620 * np.eye(5, dtype=np.float32)
        without_bias_penalty[-1, -1] -= 620
        self.assertGreater(float(np.linalg.norm(actual - solve(without_bias_penalty, right))), .01)
        prediction = brain._predict_standardized(x, actual)
        self.assertEqual(prediction.dtype, np.float64)
        np.testing.assert_array_equal(prediction, np.hstack((x, np.ones((3, 1)))) @ actual)

    def test_one_actual_fixed_CV_fit_has_fold_only_moments_and_all_image_refit(self):
        ids, x, y, source, target = self.fixture()
        folds = brain.make_common_folds(ids)
        systems, checks = [], []
        original_system = brain._primal_system
        def system(a, b, check):
            systems.append((a.copy(), b.copy()))
            return original_system(a, b, check)
        with mock.patch.object(brain, "_primal_system", side_effect=system), \
                mock.patch.object(brain, "_solve_primal", wraps=brain._solve_primal) as solve:
            result = brain.fit_brain_pair(source, target, folds, source_native_ids=(12, 3, 8),
                target_native_ids=(900, 4), check=lambda: checks.append(True))
        self.assertEqual(solve.call_count, 51)
        self.assertEqual([call.args[2] for call in solve.call_args_list[:50]], list(brain.ALPHAS) * 5)
        self.assertGreaterEqual(len(checks), 51)
        self.assertEqual([len(a) for a, b in systems], [717, 717, 718, 718, 718, 897])
        for fold, validation in enumerate(folds.validation_indices):
            keep = np.ones(897, dtype=bool)
            keep[list(validation)] = False
            for actual, raw in zip(systems[fold], (x, y)):
                train = raw[keep]
                expected = (train - np.mean(train, axis=0)[None]) / np.std(train, axis=0, ddof=1)[None]
                np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(result.source_mean, np.mean(x, axis=0)[None])
        np.testing.assert_array_equal(result.source_std, np.std(x, axis=0, ddof=1)[None])
        np.testing.assert_array_equal(result.target_mean, np.mean(y, axis=0)[None])
        np.testing.assert_array_equal(result.target_std, np.std(y, axis=0, ddof=1)[None])
        np.testing.assert_array_equal(result.matrix,
            self.pinned_matrix(systems[-1][0], systems[-1][1], result.summary.selected_alpha))
        self.assertTrue(result.summary.available)
        self.assertEqual(result.summary.completed_fits, 51)
        self.assertEqual(result.source_native_voxel_ids, (12, 3, 8))
        self.assertEqual(result.target_native_voxel_ids, (900, 4))
        self.assertEqual(result.image_ids, ids)
        self.assertFalse(result.matrix.flags.writeable)
        self.assertEqual(np.asarray(result.summary.fold_mse_by_alpha).shape, (10, 5))
        # Independently recompute one held-out cell using only its fitting rows.
        keep = np.ones(897, dtype=bool)
        validation = np.asarray(folds.validation_indices[0])
        keep[validation] = False
        xv = (x[validation] - np.mean(x[keep], axis=0)[None]) / np.std(x[keep], axis=0, ddof=1)[None]
        yv = (y[validation] - np.mean(y[keep], axis=0)[None]) / np.std(y[keep], axis=0, ddof=1)[None]
        mapping = self.pinned_matrix(*systems[0], 620)
        prediction = np.hstack((xv, np.ones((len(validation), 1)))) @ mapping
        expected_mse = np.mean(np.mean((prediction - yv) ** 2, axis=1))
        self.assertEqual(result.summary.fold_mse_by_alpha[0][0], expected_mse)

    def test_image_weighted_selection_exact_ties_and_no_scale_or_score_repair(self):
        scores = np.full((10, 5), 2.)
        scores[0], scores[1] = [1., 0., 0., 0., 0.], [0., 0., 1., 0., 0.]
        selected, aggregate = brain._choose_alpha(scores)
        self.assertEqual(selected, 640)  # Equal fold means would incorrectly tie.
        self.assertEqual(aggregate[:2], (180/897, 179/897))
        self.assertEqual(brain._choose_alpha(np.ones((10, 5)))[0], 620)
        scores[9, 4] = np.nan
        with self.assertRaises(brain.BrainLossNotEvaluable):
            brain._choose_alpha(scores)
        ids, x, y, source, target = self.fixture()
        folds = brain.make_common_folds(ids)
        for side in ("source", "target"):
            response = source if side == "source" else target
            altered = response.values.copy()
            altered[:, 0] = 1.
            with mock.patch.object(brain, "_solve_primal") as forbidden, \
                    self.assertRaisesRegex(brain.BrainLossNotEvaluable, "required_training_scale"):
                brain.fit_brain_pair(replace(source, values=altered) if side == "source" else source,
                    replace(target, values=altered) if side == "target" else target, folds,
                    source_native_ids=(12, 3, 8), target_native_ids=(900, 4), check=lambda: None)
            forbidden.assert_not_called()
        with mock.patch.object(brain, "_solve_primal") as forbidden, self.assertRaisesRegex(TimeoutError, "deadline"):
            brain.fit_brain_pair(source, target, folds, source_native_ids=(12, 3, 8), target_native_ids=(900, 4),
                check=mock.Mock(side_effect=TimeoutError("deadline")))
        forbidden.assert_not_called()

    def test_panel_keeps_all20_dispositions_and_propagates_budget_callback_and_arbitrary_errors(self):
        ids, x, y, source, target = self.fixture()
        responses = {participant: replace(source, participant=participant) for participant in brain.PARTICIPANTS}
        axes = {participant: (12, 3, 8) for participant in brain.PARTICIPANTS}
        calls, consumed = [], []
        def fit(source, target, folds, **kwargs):
            pair = source.participant, target.participant
            calls.append(pair)
            if pair == brain.PAIR_ORDER[2]:
                raise brain.BrainLossNotEvaluable("fixture_required_zero_scale")
            return SimpleNamespace(summary=brain.BrainPairSummary(*pair, True, "fixture_complete", 620, completed_fits=51))
        with mock.patch.object(brain, "fit_brain_pair", side_effect=fit):
            results = brain.fit_brain_panel(responses, authenticated_tau_train_image_ids=ids,
                expected_native_axes=axes, check=lambda: None, on_pair_completed=consumed.append)
        self.assertEqual(calls, list(brain.PAIR_ORDER))
        self.assertEqual(len(results), 20)
        self.assertEqual(len(consumed), 19)
        self.assertFalse(results[2].available)
        self.assertIsNone(results[2].selected_alpha)
        self.assertIsNone(results[2].completed_fits)
        for error in (TimeoutError("allocation ended"), RuntimeError("programming failure")):
            with mock.patch.object(brain, "fit_brain_pair", side_effect=error) as stopped, self.assertRaises(type(error)):
                brain.fit_brain_panel(responses, authenticated_tau_train_image_ids=ids,
                    expected_native_axes=axes, check=lambda: None, on_pair_completed=lambda value: None)
            self.assertEqual(stopped.call_count, 1)
        # A callback error of the dedicated type must ALSO propagate, never be
        # mistaken for a scientifically unavailable fit and silently continued.
        result = SimpleNamespace(summary=brain.BrainPairSummary(*brain.PAIR_ORDER[0], True, "fixture_complete"))
        with mock.patch.object(brain, "fit_brain_pair", return_value=result) as stopped, \
                self.assertRaisesRegex(brain.BrainLossNotEvaluable, "consumer failed"):
            brain.fit_brain_panel(responses, authenticated_tau_train_image_ids=ids,
                expected_native_axes=axes, check=lambda: None,
                on_pair_completed=mock.Mock(side_effect=brain.BrainLossNotEvaluable("consumer failed")))
        self.assertEqual(stopped.call_count, 1)


if __name__ == "__main__":
    unittest.main()
