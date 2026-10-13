"""Five focused ensemble/calibration integration checks; Slurm execution only.

Known probability arrays and instrumented interface stubs test ordering and
state isolation. These are not fitted EEG models or scientific simulations.
"""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
from scipy.special import logsumexp

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import ensemble_forecast_backend as backend
from classical_forecast_backend import (
    ClassicalForecastModel, FittingExample, FittingScope, GroupIdentity,
    PastWindow, UnresolvedFit,
)
from forecast_components import calibrated_log_probabilities
from model_recipes import reference_recipes

REGISTRY = json.loads((CODE.parent / "model_recipe_registry.json").read_text())


class ForecastStub:
    """Existing fit/reset/predict interface only; every prediction is explicit."""
    def __init__(self, seed, recipe, scope, logp, *, classical_report=False):
        self.seed, self.recipe, self.rate, self.adapter = seed, recipe, 125., "original"
        self.fitted, self.boundary_id, self.calls, self.failure = True, None, [], None
        self.logp = np.asarray(logp, dtype=float).copy()
        self.fitting_scope = scope
        self.report = {"seed": seed, "recipe_id": recipe.recipe_id, "fold_id": scope.fold_id,
            "converged" if classical_report else "completed": True, "examples": 5}

    def reset(self, *, boundary_id):
        self.boundary_id = boundary_id
        self.last_decision_time = -np.inf

    def predict_log_probabilities(self, past_input, *, decision_time, context_log_probabilities):
        if self.failure is not None:
            raise self.failure
        if self.boundary_id is None or decision_time <= self.last_decision_time:
            raise ValueError("stub requires reset and increasing decisions")
        self.calls.append((past_input, decision_time, context_log_probabilities, self.boundary_id))
        self.last_decision_time = decision_time
        return self.logp.copy()


class EnsembleForecastTests(unittest.TestCase):
    @staticmethod
    def probabilities(rows=5):
        values = {}
        for index, seed in enumerate(backend.SEEDS):
            logits = np.tile(np.linspace(-3., 2., 19), (rows, 1))
            logits[:, 3 + 4 * index] += 3. + index
            logits[:, 0] = -1200. - index
            logits += np.arange(rows)[:, None] * np.linspace(-.1, .1, 19)
            values[seed] = logits - logsumexp(logits, axis=1, keepdims=True)
        return values

    @staticmethod
    def row_keys(rows=5):
        identity = GroupIdentity("self_paced", "sub-02", "run-1", "sub-02-run-1")
        return tuple((identity, 1. + index * .05) for index in range(rows))

    def members(self, *, classical=False):
        scope = FittingScope.development(REGISTRY, source="self_paced", fold_id="self_paced.f0")
        family = "past_only_covariance_riemannian_tangent_model" if classical else "small_one_directional_GRU"
        recipe = next(item for item in reference_recipes(REGISTRY) if item.family == family)
        probabilities = self.probabilities(1)
        models = [ForecastStub(seed, recipe, scope, probabilities[seed][0], classical_report=classical)
                  for seed in backend.SEEDS]
        return [(model, model.report) for model in models], scope

    def test_exact_equal_probability_mean_and_all_seed_row_alignment(self):
        predictions, rows = self.probabilities(), self.row_keys()
        keys = {seed: rows for seed in backend.SEEDS}
        result = backend.mean_seed_log_probabilities(predictions, seed_row_keys=keys, expected_row_keys=rows)
        np.testing.assert_allclose(np.exp(result), np.mean([np.exp(predictions[seed]) for seed in backend.SEEDS], axis=0),
                                   rtol=2e-14, atol=1e-16)
        self.assertTrue(np.all(np.isfinite(result)))  # Category0 must not underflow into -inf.
        np.testing.assert_allclose(logsumexp(result, axis=1), 0., rtol=0, atol=1e-14)
        wrong = np.mean(list(predictions.values()), axis=0)
        wrong -= logsumexp(wrong, axis=1, keepdims=True)
        self.assertGreater(float(np.max(np.abs(result - wrong))), .1)
        shuffled = {**keys, 1902: rows[::-1]}
        with self.assertRaisesRegex(ValueError, "admitted row order"):
            backend.mean_seed_log_probabilities(predictions, seed_row_keys=shuffled, expected_row_keys=rows)
        with self.assertRaisesRegex(UnresolvedFit, "all_three_seed_predictions"):
            backend.mean_seed_log_probabilities({seed: predictions[seed] for seed in (1901, 1902)},
                seed_row_keys=keys, expected_row_keys=rows)
        with self.assertRaisesRegex(UnresolvedFit, "all_three_seed_row_orders"):
            backend.mean_seed_log_probabilities(predictions, seed_row_keys={1901: rows, 1902: rows}, expected_row_keys=rows)
        bad = {**predictions, 1902: predictions[1902] + .1}
        with self.assertRaisesRegex(UnresolvedFit, "normalized_19_category"):
            backend.mean_seed_log_probabilities(bad, seed_row_keys=keys, expected_row_keys=rows)

    def test_fitted_seed_reports_bind_recipe_fold_grid_and_incomplete_members(self):
        for classical in (False, True):
            members, scope = self.members(classical=classical)
            model = backend.EnsembleForecastModel(members[::-1], registry=REGISTRY, fitting_scope=scope)
            self.assertTrue(model.fitted)
            self.assertEqual(set(model.models), set(backend.SEEDS))
        members, scope = self.members()
        with self.assertRaisesRegex(UnresolvedFit, "all_three_fitted_seed_models"):
            backend.EnsembleForecastModel(members[:-1], registry=REGISTRY, fitting_scope=scope)
        members[1][0].fitted = False
        with self.assertRaisesRegex(UnresolvedFit, "incomplete_seed_fit"):
            backend.EnsembleForecastModel(members, registry=REGISTRY, fitting_scope=scope)
        members[1][0].fitted = True
        members[1][1]["fold_id"] = "self_paced.f1"
        with self.assertRaisesRegex(ValueError, "fitting fold"):
            backend.EnsembleForecastModel(members, registry=REGISTRY, fitting_scope=scope)
        members[1][1]["fold_id"] = scope.fold_id
        members[1][0].adapter = "factorial"
        with self.assertRaisesRegex(ValueError, "share recipe"):
            backend.EnsembleForecastModel(members, registry=REGISTRY, fitting_scope=scope)
        members[1][0].adapter = "original"
        with self.assertRaisesRegex(UnresolvedFit, "exactly_one_fitted_model"):
            backend.EnsembleForecastModel([members[0], members[0], members[2]], registry=REGISTRY, fitting_scope=scope)

    def test_calibration_after_mean_and_paired_suffix_share_one_reset_inference(self):
        members, scope = self.members()
        ensemble = backend.EnsembleForecastModel(members, registry=REGISTRY, fitting_scope=scope)
        window, context = object(), np.full(19, -np.log(19.))
        parameters = [1.7, .3, -.4, .1]
        with self.assertRaisesRegex(ValueError, "reset boundary"):
            ensemble.predict_log_probabilities(window, decision_time=3., context_log_probabilities=context)
        ensemble.reset(boundary_id="suffix-run")
        pair = ensemble.predict_paired_log_probabilities(window, decision_time=3.,
            context_log_probabilities=context, calibration_parameters=parameters)
        expected = logsumexp(np.stack([model.logp for model, _ in members]), axis=0) - np.log(3.)
        np.testing.assert_allclose(pair["zero_label"], expected, rtol=0, atol=1e-14)
        np.testing.assert_allclose(pair["calibrated"], calibrated_log_probabilities(expected[None], parameters)[0],
                                   rtol=0, atol=1e-14)
        wrong = logsumexp(np.stack([calibrated_log_probabilities(model.logp[None], parameters)[0]
                                   for model, _ in members]), axis=0) - np.log(3.)
        self.assertGreater(float(np.max(np.abs(pair["calibrated"] - wrong))), .1)
        for model, _ in members:
            self.assertEqual(len(model.calls), 1)
            self.assertIs(model.calls[0][0], window)
            self.assertIs(model.calls[0][2], context)
            self.assertEqual(model.calls[0][3], "suffix-run")
        with self.assertRaises(ValueError):
            ensemble.predict_paired_log_probabilities(window, decision_time=4.,
                context_log_probabilities=context, calibration_parameters=[0., 0., 0., 0.])
        self.assertTrue(all(len(model.calls) == 1 for model, _ in members))
        members[1][0].failure = UnresolvedFit("one required seed failed")
        with self.assertRaisesRegex(UnresolvedFit, "required seed failed"):
            ensemble.predict_log_probabilities(window, decision_time=4., context_log_probabilities=context)
        self.assertIsNone(ensemble.boundary_id)
        members[1][0].failure = None
        with self.assertRaisesRegex(ValueError, "reset boundary"):
            ensemble.predict_log_probabilities(window, decision_time=5., context_log_probabilities=context)
        ensemble.reset(boundary_id="suffix-run")
        identity = ensemble.predict_paired_log_probabilities(window, decision_time=3.,
            context_log_probabilities=context, calibration_parameters=[1., 0., 0., 0.])
        np.testing.assert_allclose(identity["zero_label"], identity["calibrated"], rtol=0, atol=1e-14)

    def test_actual_classical_fits_bind_scope_and_fixed_causal_filters(self):
        recipe = next(item for item in reference_recipes(REGISTRY)
                      if item.family == "past_only_filterbank_bandpower_elastic_net")
        offset = np.full(19, -np.log(19.))
        def fit(person, seed):
            scope = FittingScope.development(REGISTRY, source="WAY", participant=person,
                                             fold_id=f"WAY.P{person}.validation_series6")
            examples = []
            for series in scope.allowed_series:
                boundary = f"P{person}S{series}"
                for category in range(19):
                    window = PastWindow(np.ones((5, 29)), np.arange(5) / 5., [boundary] * 5)
                    examples.append(FittingExample(window, 1.,
                        GroupIdentity("WAY", person, series, boundary), category, 0, offset))
            model = ClassicalForecastModel()
            report = model.fit([examples], recipe=recipe, seed=seed, semantics={"registry": REGISTRY,
                "fitting_scope": scope, "analysis_sample_rate_hz": 5., "adapter": "original",
                "causal_filter_coefficients": [[1.]] * 5})
            self.assertTrue(report["converged"], report)
            self.assertEqual(model.fitting_scope, scope)
            return model, report
        members = [fit(1, seed) for seed in backend.SEEDS]
        scope = members[0][0].fitting_scope
        ensemble = backend.EnsembleForecastModel(members, registry=REGISTRY, fitting_scope=scope)
        ensemble.reset(boundary_id="P1S6")
        window = PastWindow(np.ones((5, 29)), np.arange(5) / 5., ["P1S6"] * 5)
        result = ensemble.predict_log_probabilities(window, decision_time=1., context_log_probabilities=offset)
        np.testing.assert_allclose(result, offset, rtol=0, atol=1e-12)
        other_model, other_report = fit(2, 1902)
        # A report accidentally associated with the wrong person must not stand
        # in for the fitted model's actual scope, even if its fold text matches.
        mislabeled = {**other_report, "fold_id": scope.fold_id}
        with self.assertRaisesRegex(ValueError, "fitting fold"):
            backend.EnsembleForecastModel([members[0], (other_model, mislabeled), members[2]],
                registry=REGISTRY, fitting_scope=scope)
        del members[1][0].fitting_scope
        with self.assertRaisesRegex(ValueError, "fitting fold"):
            backend.EnsembleForecastModel(members, registry=REGISTRY, fitting_scope=scope)
        members[1][0].fitting_scope = scope
        members[1][0].features.coefficients[0][0] = .5
        with self.assertRaisesRegex(ValueError, "causal FIRs"):
            backend.EnsembleForecastModel(members, registry=REGISTRY, fitting_scope=scope)

    def test_prefix_labels_weights_and_identity_fallback_are_preserved_without_models(self):
        predictions, rows = self.probabilities(), self.row_keys()
        keys = {seed: rows for seed in backend.SEEDS}
        categories, complete = np.array([0, 6, 18, -1, -1]), np.array([0, 0, 18, 0, 9])
        weights = np.ones(5)
        before = copy.deepcopy((predictions, categories, complete, weights))
        actual_fit = backend.fit_four_parameter_calibration
        with mock.patch.object(backend, "fit_four_parameter_calibration", wraps=actual_fit) as fitter:
            report = backend.fit_ensemble_calibration(predictions, seed_row_keys=keys, prefix_row_keys=rows,
                observed_category=categories, complete_bins=complete, weights=weights)
        expected = backend.mean_seed_log_probabilities(predictions, seed_row_keys=keys, expected_row_keys=rows)
        np.testing.assert_allclose(fitter.call_args.args[0], expected, rtol=0, atol=0)
        for observed, original in zip(fitter.call_args.args[1:], (categories, complete, weights)):
            np.testing.assert_array_equal(observed, original)
        self.assertEqual(report["prefix_rows"], 5)
        self.assertFalse(report["prefix_eligibility_and_timing_established_by_backend"])
        self.assertFalse(report["recipe_frozen_for_empirical_use"])
        self.assertEqual(report["seeds"], [1901, 1902, 1903])
        fallback = {"parameters": [1., 0., 0., 0.], "accepted": False, "identity_fallback": True,
            "optimizer_message": "fixture: iteration cap", "recipe_frozen_for_empirical_use": False}
        with mock.patch.object(backend, "fit_four_parameter_calibration", return_value=fallback) as fitter:
            failed = backend.fit_ensemble_calibration(predictions, seed_row_keys=keys, prefix_row_keys=rows,
                observed_category=categories, complete_bins=complete, weights=weights)
        self.assertEqual(fitter.call_count, 1)
        self.assertTrue(failed["identity_fallback"])
        self.assertEqual(failed["parameters"], [1., 0., 0., 0.])
        for seed in backend.SEEDS:
            np.testing.assert_array_equal(predictions[seed], before[0][seed])
        for observed, original in zip((categories, complete, weights), before[1:]):
            np.testing.assert_array_equal(observed, original)
        mixed = (*rows[:-1], (GroupIdentity("self_paced", "sub-03", "run-1", "other"), 2.))
        with self.assertRaisesRegex(ValueError, "one self-paced participant"):
            backend.fit_ensemble_calibration(predictions, seed_row_keys=keys, prefix_row_keys=mixed,
                observed_category=categories, complete_bins=complete, weights=weights)


if __name__ == "__main__":
    unittest.main()
