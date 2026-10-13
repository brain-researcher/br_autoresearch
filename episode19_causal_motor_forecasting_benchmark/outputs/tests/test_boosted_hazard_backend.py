"""Five focused boosted-baseline checks, only in coordinated compute.

One small full-schedule fit checks the actual tree backend; other fixtures
exercise censored targets, weighting and interfaces without repeated fitting.
No source data, model search, scientific score or readiness claim is produced.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import boosted_hazard_backend as backend
from classical_forecast_backend import FittingScope, UnresolvedFit, hazard_loss_and_gradient
from non_neural_forecast_backend import ContextFeatures
import test_non_neural_forecast_backend as wrapper_fixtures

REGISTRY = json.loads((CODE.parent / "model_recipe_registry.json").read_text())


class BoostedHazardTests(unittest.TestCase):
    def tiny(self):
        # Every head sees an event and a known negative, plus zero-information
        # and partially censored examples. These are likelihood fixtures only.
        categories = np.r_[np.arange(19), -1, -1]
        complete = np.r_[np.full(19, 18), 0, 5]
        features = np.column_stack((np.arange(21)/20., np.sin(np.arange(21))))
        weights = np.arange(1, 22, dtype=float)
        weights /= weights.sum()
        return features, categories, complete, weights

    def test_conditional_masks_equal_existing_censored_likelihood(self):
        x, category, complete, _ = self.tiny()
        logits = np.linspace(-2., 2., len(x)*18).reshape(len(x), 18)
        risk, event = backend.conditional_hazard_targets(category, complete, len(x))
        explicit = np.sum(risk*(np.logaddexp(0., logits)-event*logits), axis=1)
        loss, _ = hazard_loss_and_gradient(logits, category, complete)
        np.testing.assert_allclose(explicit, loss, atol=1e-14, rtol=0)
        np.testing.assert_array_equal(risk[0], [True]+[False]*17)
        self.assertEqual(int(event[0].sum()), 1)
        self.assertFalse(risk[-2].any())
        np.testing.assert_array_equal(risk[-1], [True]*5+[False]*13)
        self.assertFalse(event[-1].any())
        self.assertTrue(risk[18].all())
        self.assertFalse(event[18].any())

    def test_each_head_receives_only_observed_intervals_and_unchanged_person_weights(self):
        x, category, complete, weights = self.tiny()
        risk, event = backend.conditional_hazard_targets(category, complete, len(x))
        heads = []
        class Head:
            n_estimators_ = 200
            estimators_ = np.empty((200, 1), dtype=object)
            classes_ = np.array([0, 1])
            def fit(self, features, targets, *, sample_weight):
                self.observed = (features.copy(), targets.copy(), sample_weight.copy())
                return self
            def decision_function(self, features):
                return np.zeros(len(features))
        def create(depth, seed):
            self.assertEqual((depth, seed), (3, 1902))
            head = Head()
            heads.append(head)
            return head
        with mock.patch.object(backend, "_new_head", side_effect=create):
            result = backend.fit_boosted_hazard(x, category, complete, weights, depth=3, seed=1902)
        self.assertTrue(result.completed)
        self.assertEqual(len(heads), 18)
        for index, head in enumerate(heads):
            got_x, got_y, got_weight = head.observed
            np.testing.assert_array_equal(got_x, x[risk[:, index]])
            np.testing.assert_array_equal(got_y, event[risk[:, index], index])
            np.testing.assert_allclose(got_weight, weights[risk[:, index]], atol=1e-15, rtol=0)
        self.assertAlmostEqual(result.objective, result.initial_objective)

    def test_missing_or_single_class_head_stays_unresolved_without_any_tree_fit_or_floor(self):
        x, category, complete, weights = self.tiny()
        cases = [(np.full(len(x), -1), np.zeros(len(x), dtype=int), "no_observed"),
                 (np.full(len(x), 18), complete, "single_class"),
                 (np.zeros(len(x), dtype=int), complete, "single_class")]
        with mock.patch.object(backend, "_new_head") as forbidden:
            for labels, bins, reason in cases:
                result = backend.fit_boosted_hazard(x, labels, bins, weights, depth=2, seed=1901)
                self.assertFalse(result.completed)
                self.assertIn(reason, result.reason)
                self.assertEqual(result.unresolved_head, 0)
                self.assertEqual(result.heads, ())
                with self.assertRaises(UnresolvedFit):
                    result.logits(x)
            forbidden.assert_not_called()

    def test_actual_fixed_tree_schedule_returns_finite_joint_probabilities_without_extra_search(self):
        x, category, complete, weights = self.tiny()
        fitted = backend.fit_boosted_hazard(x, category, complete, weights, depth=2, seed=1901)
        self.assertTrue(fitted.completed)
        self.assertLess(fitted.objective, fitted.initial_objective)
        self.assertEqual(len(fitted.heads), 18)
        for head in fitted.heads:
            self.assertEqual(head.n_estimators_, 200)
            self.assertEqual(head.get_params()["init"], "zero")
            self.assertIsNone(head.get_params()["n_iter_no_change"])
            self.assertEqual(head.get_params()["subsample"], 1.)
        logp = backend.hazard_log_probabilities(fitted.logits(x))
        self.assertEqual(logp.shape, (len(x), 19))
        self.assertTrue(np.all(np.isfinite(logp)))
        np.testing.assert_allclose(np.exp(logp).sum(axis=1), 1., atol=1e-13, rtol=0)

    def test_model_keeps_canonical_roles_fitting_weights_and_past_only_frozen_offsets(self):
        fixture = wrapper_fixtures.NonNeuralWrapperTests()
        schema = wrapper_fixtures.SCHEMA
        recipe = backend.boosted_hazard_recipes(REGISTRY, source="self_paced")[1]
        fitted = SimpleNamespace(completed=True, reason="fixture_complete", objective=1., unresolved_head=None,
                                 logits=lambda matrix: np.tile(np.linspace(-1., 1., 18), (len(matrix), 1)))
        model = backend.BoostedHazardModel()
        with mock.patch.object(backend, "fit_boosted_hazard", return_value=fitted) as fit:
            report = model.fit([fixture.examples("self_paced")], recipe=recipe, seed=1902, registry=REGISTRY,
                fitting_scope=fixture.scope("self_paced"), feature_schema=schema)
        weights = fit.call_args.args[3]
        self.assertAlmostEqual(float(weights[:3].sum()), 1/12)
        np.testing.assert_allclose(weights[3:], 1/12)
        self.assertEqual(fit.call_args.kwargs, {"depth": 3, "seed": 1902})
        self.assertFalse(report["source_admission_established_by_backend"])
        model.reset(boundary_id="validation-person")
        context = ContextFeatures([.5, 2.], schema, "self_paced", "sub-02", "validation-person", 20.)
        mean, scale = model.standardizer.mean.copy(), model.standardizer.scale.copy()
        for invalid in (replace(context, available_through=21.), replace(context, boundary_id="other"),
                        replace(context, feature_schema=tuple(reversed(schema)))):
            with self.assertRaises(ValueError):
                model.predict_log_probabilities(invalid, decision_time=20.)
            self.assertEqual(model.last_decision_time, -np.inf)
        logp = model.predict_log_probabilities(context, decision_time=20.)
        self.assertAlmostEqual(float(np.exp(logp).sum()), 1., places=14)
        np.testing.assert_array_equal(model.standardizer.mean, mean)
        np.testing.assert_array_equal(model.standardizer.scale, scale)
        with self.assertRaisesRegex(ValueError, "increasing"):
            model.predict_log_probabilities(context, decision_time=20.)
        way_recipe = backend.boosted_hazard_recipes(REGISTRY, source="WAY")[0]
        with mock.patch.object(backend, "fit_boosted_hazard", return_value=fitted):
            way = backend.BoostedHazardModel()
            way.fit([fixture.examples()], recipe=way_recipe, seed=1901, registry=REGISTRY,
                    fitting_scope=fixture.scope(), feature_schema=schema)
        way.reset(boundary_id="new-series")
        with self.assertRaisesRegex(ValueError, "participant"):
            way.predict_log_probabilities(ContextFeatures([.5, 2.], schema, "WAY", 2, "new-series", 20.),
                                          decision_time=20.)
        def unopened():
            self.fail("role rejection must precede consuming batches")
            yield []
        with self.assertRaisesRegex(ValueError, "canonical development fold"):
            backend.BoostedHazardModel().fit(unopened(), recipe=way_recipe, seed=1901, registry=REGISTRY,
                fitting_scope=FittingScope("WAY", "WAY.P1.validation_series6", (1,), (8, 9)), feature_schema=schema)
        with self.assertRaisesRegex(ValueError, "registered"):
            backend.BoostedHazardModel().fit(unopened(), recipe=replace(way_recipe, trees=201), seed=1901,
                registry=REGISTRY, fitting_scope=fixture.scope(), feature_schema=schema)


if __name__ == "__main__":
    unittest.main()
