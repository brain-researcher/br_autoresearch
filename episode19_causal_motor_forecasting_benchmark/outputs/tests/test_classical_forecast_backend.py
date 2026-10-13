"""Small arithmetic/interface checks, to run only in a coordinated allocation.

These fixtures are not physiological recovery, calibration, source qualification
or an empirical model comparison. No EP19 source or protected outcome is read.
"""
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
from scipy.special import logsumexp

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
from classical_forecast_backend import (
    BandpowerFeatures, ClassicalForecastModel, FittingExample, FittingScope,
    FittingStandardizer, GroupIdentity, PastWindow, TangentFeatures,
    UnresolvedFit, _proximal_fit, affine_invariant_mean,
    ensemble_log_probabilities, exact_samples, fit_logistic_hazard,
    hazard_log_probabilities, hazard_loss_and_gradient, participant_weights,
    shrunk_covariance,
)
from forecast_components import censoring_loss_and_logit_gradient, FactorizedBlocks
from model_recipes import reference_recipes

REGISTRY = json.loads((CODE.parent / "model_recipe_registry.json").read_text())


class ClassicalBackendTests(unittest.TestCase):
    def test_hazard_matches_existing_censored_likelihood_and_gradient(self):
        logits = np.linspace(-3, 2, 90).reshape(5, 18)
        category = np.array([0, 5, 18, -1, -1])
        complete = np.array([0, 0, 18, 9, 0])
        logp = hazard_log_probabilities(logits)
        loss, gradient = hazard_loss_and_gradient(logits, category, complete)
        old_loss, _ = censoring_loss_and_logit_gradient(logp, category, complete)
        np.testing.assert_allclose(loss, old_loss, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(logsumexp(logp, axis=1), 0, atol=2e-15)
        for row, column in ((0, 0), (1, 5), (2, 17), (3, 8), (3, 9), (4, 0)):
            plus, minus = logits.copy(), logits.copy()
            plus[row, column] += 1e-6
            minus[row, column] -= 1e-6
            numerical = (hazard_loss_and_gradient(plus, category, complete)[0][row]
                         - hazard_loss_and_gradient(minus, category, complete)[0][row]) / 2e-6
            self.assertAlmostEqual(gradient[row, column], numerical, places=7)
        np.testing.assert_array_equal(gradient[3, 9:], 0)
        np.testing.assert_array_equal(gradient[4], 0)
        with self.assertRaises(ValueError):
            hazard_loss_and_gradient(logits, category.astype(float) + .1, complete)

    def test_natural_prevalence_weights_and_fitting_only_scaling(self):
        weights = participant_weights(["a", "a", "a", "b"])
        np.testing.assert_allclose(weights, [1/6, 1/6, 1/6, 1/2])
        standardizer = FittingStandardizer().fit([[0, 7], [2, 7]], [.25, .75])
        old_mean, old_scale = standardizer.mean.copy(), standardizer.scale.copy()
        standardizer.transform([[1e9, -1e8]])
        np.testing.assert_array_equal(old_mean, standardizer.mean)
        np.testing.assert_array_equal(old_scale, standardizer.scale)
        np.testing.assert_allclose(standardizer.transform([[0, 7], [2, 7]])[:, 1], 0)

    def test_group_roles_and_missing_groups_cannot_be_silently_dropped(self):
        scope = FittingScope.development(REGISTRY, source="WAY",
            participant=1, fold_id="WAY.P1.validation_series6")
        identities = [GroupIdentity("WAY", 1, series, f"P1S{series}") for series in range(1, 6)]
        np.testing.assert_allclose(scope.validate(identities), .2)
        for bad in (GroupIdentity("WAY", 1, 6, "validation"),
                    GroupIdentity("WAY", 1, 8, "held-out"),
                    GroupIdentity("WAY", 2, 1, "other-person")):
            with self.assertRaises(ValueError):
                scope.validate([*identities, bad])
        with self.assertRaises(UnresolvedFit):
            scope.validate(identities[:-1])
        self_scope = FittingScope.development(REGISTRY, source="self_paced", fold_id="self_paced.f0")
        people = [GroupIdentity("self_paced", person, 1, person + "run1")
                  for person in self_scope.allowed_participants]
        self_scope.validate(people)
        with self.assertRaises(ValueError):
            self_scope.validate([*people, GroupIdentity("self_paced", "sub-01", 1, "held")])
        with self.assertRaises(ValueError):
            self_scope.validate([GroupIdentity(item.source, item.participant, 1, "same") for item in people])

    def test_timestamped_window_rejects_future_gaps_and_cross_boundary(self):
        values = np.arange(2900., dtype=float).reshape(100, 29)
        times = np.arange(100) / 100
        kwargs = dict(decision_time=1., sample_rate_hz=100., history_seconds=1., boundary_id="run")
        window = PastWindow(values, times, ["run"] * 100)
        np.testing.assert_array_equal(window.validated(**kwargs), values)
        with self.assertRaises(ValueError):
            PastWindow(values, times + .01, ["run"] * 100).validated(**kwargs)
        gap = times.copy()
        gap[35] += .003
        with self.assertRaises(ValueError):
            PastWindow(values, gap, ["run"] * 100).validated(**kwargs)
        with self.assertRaises(ValueError):
            PastWindow(values, times, ["run"] * 99 + ["new-run"]).validated(**kwargs)

    def test_forged_fitting_scope_is_rejected_before_batches_or_features(self):
        recipe = next(reference_recipes(REGISTRY))
        genuine_self = FittingScope.development(REGISTRY, source="self_paced",
                                               fold_id="self_paced.f0")
        forged = [
            FittingScope("WAY", "WAY.P1.validation_series6", (1,), (8, 9)),
            FittingScope("WAY", "WAY.P1.validation_series6", (1,), (1, 2, 3, 4, 5, 6)),
            FittingScope("self_paced", genuine_self.fold_id,
                         (*genuine_self.allowed_participants[:-1], "sub-01"), None),
        ]
        def forbidden_batches():
            self.fail("batches must not be consumed for a forged fitting scope")
            yield []
        for scope in forged:
            with self.subTest(source=scope.source, participants=scope.allowed_participants), \
                    mock.patch("classical_forecast_backend.BandpowerFeatures") as features:
                with self.assertRaisesRegex(ValueError, "canonical development fold"):
                    ClassicalForecastModel().fit(forbidden_batches(), recipe=recipe, seed=1901,
                        semantics={"registry": REGISTRY, "fitting_scope": scope})
                features.assert_not_called()
    def test_explicit_rate_and_FIR_support(self):
        self.assertEqual(exact_samples(.128, 250.), 32)
        with self.assertRaises(ValueError):
            exact_samples(.128, 256.)
        feature = BandpowerFeatures([[.25, .75]] * 5)
        history = np.arange(6.)[:, None] * np.ones((1, 29))
        observed = feature.transform_original([history], recent_samples=2)[0]
        expected_older = np.log1p(np.mean(np.array([.25, 1.25, 2.25]) ** 2))
        expected_recent = np.log1p(np.mean(np.array([3.25, 4.25]) ** 2))
        np.testing.assert_allclose(observed[:145], expected_older)
        np.testing.assert_allclose(observed[145:], expected_recent)
        reset = feature.transform_factorial([(history[1:4], history[4:])])[0]
        np.testing.assert_allclose(reset[145:], np.log1p(4.25 ** 2))
        self.assertFalse(np.allclose(reset[145:], observed[145:]))
        # Original FIR may carry older state into the recent slot; a factorial
        # reset may not. The older modification does not change recent input.
        changed = history.copy()
        changed[3] += 10
        self.assertFalse(np.allclose(feature.transform_original([changed], 2)[0, 145:], observed[145:]))
        np.testing.assert_array_equal(feature.transform_factorial([(changed[1:4], changed[4:])])[0, 145:], reset[145:])
        with self.assertRaises(ValueError):
            feature.transform_factorial([(np.ones((3, 29)), np.ones((1, 29)))])

    def test_factorial_mask_is_applied_before_both_feature_paths(self):
        rng = np.random.default_rng(17)
        blocks = FactorizedBlocks(rng.normal(size=(40, 29)), rng.normal(size=(40, 29)), rng.normal(size=(20, 29)))
        changed_recent = FactorizedBlocks(blocks.complement, blocks.older_slow, blocks.recent + 100)
        changed_slow = FactorizedBlocks(blocks.complement, blocks.older_slow - 500, blocks.recent)
        band = BandpowerFeatures([[1., -.5]] * 5)
        def features(value, recent, slow):
            reconstructed = value.reconstruct(recent, slow)
            slots = [(reconstructed[:40], reconstructed[40:])]
            return band.transform_factorial(slots), [shrunk_covariance(slot, .1) for slot in slots[0]]
        for slow in (False, True):
            expected = features(blocks, False, slow)
            mutated = features(changed_recent, False, slow)
            np.testing.assert_array_equal(expected[0], mutated[0])
            np.testing.assert_array_equal(expected[1], mutated[1])
        for recent in (False, True):
            expected = features(blocks, recent, False)
            mutated = features(changed_slow, recent, False)
            np.testing.assert_array_equal(expected[0], mutated[0])
            np.testing.assert_array_equal(expected[1], mutated[1])
        self.assertFalse(np.allclose(features(blocks, True, True)[0], features(changed_recent, True, True)[0]))

    def test_weighted_affine_mean_and_tangent_geometry(self):
        diagonal1 = np.linspace(1, 2, 29)
        diagonal2 = np.linspace(3, 7, 29)
        mean = affine_invariant_mean([np.diag(diagonal1), np.diag(diagonal2)], [.25, .75])
        np.testing.assert_allclose(mean, np.diag(diagonal1 ** .25 * diagonal2 ** .75), atol=1e-10)
        rng = np.random.default_rng(11)
        pair = (rng.normal(size=(50, 29)), rng.normal(size=(20, 29)))
        tangent = TangentFeatures(.5).fit([pair], [1.])
        original_reference = tangent.reference.copy()
        transformed = tangent.transform([pair])
        self.assertEqual(transformed.shape, (1, 870))
        self.assertTrue(np.all(np.isfinite(transformed)))
        tangent.transform([(pair[0] * 3, pair[1] * 4)])
        np.testing.assert_array_equal(tangent.reference, original_reference)
        zero = shrunk_covariance(np.zeros((3, 29)), .1)
        np.testing.assert_allclose(zero, np.eye(29) * 1e-6)
        with self.assertRaises(UnresolvedFit):
            affine_invariant_mean([np.zeros((29, 29))], [1])

    def test_proximal_fit_improves_and_iteration_failure_remains_visible(self):
        x = np.tile([[-1.], [1.]], (20, 1))
        category = np.tile([0, 1], 20)
        complete = np.zeros(40, dtype=int)
        offsets = np.full((40, 19), -np.log(19.))
        fit = _proximal_fit(x, category, complete, np.ones(40), penalty=.1,
                           l1_ratio=.1, offsets=offsets)
        self.assertTrue(fit.converged, fit.reason)
        self.assertLess(fit.objective, fit.initial_objective - .5)
        self.assertGreater(fit.coefficients[0, 1], fit.coefficients[0, 0])
        incomplete = _proximal_fit(x, category, complete, np.ones(40), penalty=.1,
            l1_ratio=.1, offsets=offsets, maximum_iterations=1, relative_tolerance=1e-14)
        self.assertFalse(incomplete.converged)
        self.assertEqual(incomplete.reason, "iteration_limit")
        with self.assertRaises(UnresolvedFit):
            incomplete.require_converged()

    def test_logistic_hazard_fit_keeps_censoring_unknown_intervals(self):
        category = np.tile(np.arange(19), 3)
        x = np.zeros((len(category), 1))
        fit = fit_logistic_hazard(x, category, np.zeros(len(category), dtype=int),
                                 np.ones(len(category)), penalty=.1)
        self.assertTrue(fit.converged, fit.reason)
        probability = np.exp(hazard_log_probabilities(fit.intercept[None, :]))[0]
        np.testing.assert_allclose(probability, 1/19, atol=.006)
        # Zero-bin censored examples carry no known negative/event intervals.
        loss, gradient = hazard_loss_and_gradient(np.ones((2, 18)), [-1, -1], [0, 0])
        np.testing.assert_array_equal(loss, 0)
        np.testing.assert_array_equal(gradient, 0)

    def test_seed_ensemble_is_probability_average_not_logit_average(self):
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(3, 4, 19))
        logs = logits - logsumexp(logits, axis=2, keepdims=True)
        actual = ensemble_log_probabilities(logs)
        np.testing.assert_allclose(np.exp(actual), np.exp(logs).mean(axis=0), atol=1e-15)
        with self.assertRaises(ValueError):
            ensemble_log_probabilities(logs[:2])

    def test_end_to_end_classical_fit_and_prediction_reset(self):
        recipe = next(item for item in reference_recipes(REGISTRY)
                      if item.family == "past_only_filterbank_bandpower_elastic_net")
        scope = FittingScope.development(REGISTRY, source="WAY", participant=1,
                                        fold_id="WAY.P1.validation_series6")
        offset = np.full(19, -np.log(19.))
        examples = []
        for series in range(1, 6):
            boundary = f"P1S{series}"
            for category in range(19):
                window = PastWindow(np.ones((100, 29)), np.arange(100) / 100,
                                    [boundary] * 100)
                examples.append(FittingExample(window, 1., GroupIdentity("WAY", 1, series, boundary),
                                                category, 0, offset))
        model = ClassicalForecastModel()
        result = model.fit([examples], recipe=recipe, seed=1901,
            semantics={"registry": REGISTRY, "fitting_scope": scope,
                       "analysis_sample_rate_hz": 100., "adapter": "original",
                       "causal_filter_coefficients": [[1.]] * 5})
        self.assertTrue(result["converged"], result)
        window = PastWindow(np.ones((100, 29)), np.arange(100) / 100, ["future-run"] * 100)
        with self.assertRaises(ValueError):
            model.predict_log_probabilities(window, decision_time=1., context_log_probabilities=offset)
        model.reset(boundary_id="future-run")
        first = model.predict_log_probabilities(window, decision_time=1., context_log_probabilities=offset)
        np.testing.assert_allclose(first, offset, atol=1e-12)
        with self.assertRaises(ValueError):
            model.predict_log_probabilities(window, decision_time=1., context_log_probabilities=offset)
        model.reset(boundary_id="future-run")
        repeated = model.predict_log_probabilities(window, decision_time=1., context_log_probabilities=offset)
        np.testing.assert_array_equal(first, repeated)
        # A source buffer's later samples cannot enter the supplied past window.
        stream = np.ones((200, 29))
        past = PastWindow(stream[:100], np.arange(100) / 100, ["future-run"] * 100)
        stream[100:] = -1e9
        model.reset(boundary_id="future-run")
        np.testing.assert_array_equal(first, model.predict_log_probabilities(
            past, decision_time=1., context_log_probabilities=offset))


if __name__ == "__main__":
    unittest.main()
