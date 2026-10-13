"""Four small wrapper/interface checks; no optimizer reruns or source payloads.

Run only inside the coordinator's substantive Slurm allocation. The low-level
optimizer already has its own focused checks; these fixtures mock that fit and
exercise its new role/schema/past-only/output integration.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import non_neural_forecast_backend as backend
from classical_forecast_backend import (
    FittingScope, GroupIdentity, LinearFit, UnresolvedFit, normalized_log_probabilities,
)
from forecast_components import calibrated_log_probabilities

REGISTRY = json.loads((CODE.parent / "model_recipe_registry.json").read_text())
SCHEMA = ("prepared_feature_a", "prepared_feature_b")


class NonNeuralWrapperTests(unittest.TestCase):
    def scope(self, source="WAY"):
        return FittingScope.development(REGISTRY, source=source,
            fold_id="WAY.P1.validation_series6" if source == "WAY" else "self_paced.f0",
            participant=1 if source == "WAY" else None)

    def examples(self, source="WAY"):
        scope = self.scope(source)
        groups = scope.allowed_series if source == "WAY" else scope.allowed_participants
        examples = []
        for index, group in enumerate(groups):
            person = 1 if source == "WAY" else group
            run = group if source == "WAY" else "run1"
            boundary = f"{source}:{person}:{run}"
            # Unequal point counts ensure the wrapper preserves person weighting.
            count = 3 if source == "self_paced" and index == 0 else 1
            for point in range(count):
                when = 10. + point
                context = backend.ContextFeatures(np.asarray([index, index + point + 2.]), SCHEMA,
                                                   source, person, boundary, when)
                examples.append(backend.BaselineFittingExample(context, when,
                    GroupIdentity(source, person, run, boundary), 0 if point == 0 else -1, 0 if point == 0 else 6))
        return examples

    def fitted(self, source="WAY", converged=True):
        recipe = backend.logistic_hazard_recipes(REGISTRY, source=source)[0]
        fit_result = LinearFit(np.array([[.1] * 18, [-.2] * 18]), np.linspace(-.3, .3, 18),
            converged, 3, 1., 2., "relative_objective_tolerance" if converged else "iteration_limit")
        model = backend.LogisticHazardModel()
        with mock.patch.object(backend, "fit_logistic_hazard", return_value=fit_result) as fit:
            report = model.fit([self.examples(source)], recipe=recipe, seed=1901, registry=REGISTRY,
                              fitting_scope=self.scope(source), feature_schema=SCHEMA)
        return model, report, fit

    def test_registered_roles_weights_and_scope_rejection_precede_fit(self):
        self.assertEqual([recipe.l2 for recipe in backend.logistic_hazard_recipes(REGISTRY, source="WAY")],
                         [.001, .01, .1])
        model, report, fit = self.fitted("self_paced")
        weights = fit.call_args.args[3]
        self.assertAlmostEqual(float(weights[:3].sum()), 1/12)
        np.testing.assert_allclose(weights[3:], 1/12)
        self.assertEqual(report["source"], "self_paced")
        self.assertFalse(report["source_admission_established_by_backend"])
        recipe = backend.logistic_hazard_recipes(REGISTRY, source="WAY")[0]
        def unopened():
            self.fail("forged scope must be rejected before batches are consumed")
            yield []
        with mock.patch.object(backend, "fit_logistic_hazard") as forbidden:
            with self.assertRaisesRegex(ValueError, "canonical development fold"):
                backend.LogisticHazardModel().fit(unopened(), recipe=recipe, seed=1901,
                    registry=REGISTRY, fitting_scope=FittingScope("WAY", "WAY.P1.validation_series6", (1,), (8, 9)),
                    feature_schema=SCHEMA)
            with self.assertRaises(UnresolvedFit):
                backend.LogisticHazardModel().fit([self.examples()[:-1]], recipe=recipe, seed=1901,
                    registry=REGISTRY, fitting_scope=self.scope(), feature_schema=SCHEMA)
            with self.assertRaisesRegex(ValueError, "registered"):
                backend.LogisticHazardModel().fit([self.examples()], recipe=replace(recipe, l2=.02), seed=1901,
                    registry=REGISTRY, fitting_scope=self.scope(), feature_schema=SCHEMA)
            mismatched = self.examples()
            mismatched[0] = replace(mismatched[0], context=replace(mismatched[0].context, participant=2))
            with self.assertRaisesRegex(ValueError, "participant"):
                backend.LogisticHazardModel().fit([mismatched], recipe=recipe, seed=1901,
                    registry=REGISTRY, fitting_scope=self.scope(), feature_schema=SCHEMA)
            forbidden.assert_not_called()

    def test_frozen_19_category_output_is_a_valid_EEG_offset_and_calibration_input(self):
        model, report, fit = self.fitted()
        model.reset(boundary_id="validation-series")
        context = backend.ContextFeatures([.25, 2.5], SCHEMA, "WAY", 1, "validation-series", 20.)
        logp = model.predict_log_probabilities(context, decision_time=20.)
        self.assertEqual(logp.shape, (19,))
        self.assertTrue(np.all(np.isfinite(logp)))
        self.assertAlmostEqual(float(np.exp(logp).sum()), 1., places=14)
        # This is the existing EEG correction-head operation with zero correction.
        np.testing.assert_allclose(normalized_log_probabilities(logp[None, :] + np.zeros((1, 19)))[0],
                                   logp, atol=1e-13, rtol=0)
        np.testing.assert_allclose(calibrated_log_probabilities(logp[None, :], [1., 0., 0., 0.])[0],
                                   logp, atol=1e-13, rtol=0)
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(fit.call_args.kwargs["penalty"], .001)
        grouped, _, _ = self.fitted("self_paced")
        grouped.reset(boundary_id="held-development-person")
        unseen = backend.ContextFeatures([.25, 2.5], SCHEMA, "self_paced", "sub-02",
                                         "held-development-person", 20.)
        self.assertNotIn("sub-02", grouped.fitting_scope.allowed_participants)
        self.assertAlmostEqual(float(np.exp(grouped.predict_log_probabilities(unseen, decision_time=20.)).sum()),
                               1., places=14)

    def test_schema_source_future_and_boundary_errors_do_not_update_state_or_scaling(self):
        model, _, _ = self.fitted()
        model.reset(boundary_id="validation-series")
        context = backend.ContextFeatures([1., 3.], SCHEMA, "WAY", 1, "validation-series", 20.)
        mean, scale = model.standardizer.mean.copy(), model.standardizer.scale.copy()
        invalid = [replace(context, feature_schema=tuple(reversed(SCHEMA))),
                   replace(context, source="self_paced"), replace(context, participant=2),
                   replace(context, boundary_id="different-run"),
                   replace(context, available_through=20.001), replace(context, values=[np.nan, 3.])]
        for item in invalid:
            with self.subTest(context=item), self.assertRaises(ValueError):
                model.predict_log_probabilities(item, decision_time=20.)
            self.assertEqual(model.last_decision_time, -np.inf)
        model.reset(boundary_id="different-person-series")
        with self.assertRaisesRegex(ValueError, "participant"):
            model.predict_log_probabilities(replace(context, participant=2, boundary_id="different-person-series"),
                                            decision_time=20.)
        self.assertEqual(model.last_decision_time, -np.inf)
        model.reset(boundary_id="validation-series")
        first = model.predict_log_probabilities(context, decision_time=20.)
        with self.assertRaisesRegex(ValueError, "increasing"):
            model.predict_log_probabilities(context, decision_time=20.)
        model.reset(boundary_id="validation-series")
        np.testing.assert_array_equal(first, model.predict_log_probabilities(context, decision_time=20.))
        np.testing.assert_array_equal(mean, model.standardizer.mean)
        np.testing.assert_array_equal(scale, model.standardizer.scale)

    def test_nonconverged_fit_cannot_supply_offsets_or_trigger_a_retry(self):
        model, report, fit = self.fitted(converged=False)
        self.assertFalse(report["converged"])
        self.assertEqual(report["reason"], "iteration_limit")
        model.reset(boundary_id="validation-series")
        context = backend.ContextFeatures([1., 3.], SCHEMA, "WAY", 1, "validation-series", 20.)
        with self.assertRaisesRegex(UnresolvedFit, "cannot supply EEG offsets"):
            model.predict_log_probabilities(context, decision_time=20.)
        self.assertEqual(fit.call_count, 1)


if __name__ == "__main__":
    unittest.main()
