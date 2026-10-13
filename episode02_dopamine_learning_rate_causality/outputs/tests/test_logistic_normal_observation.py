"""Focused deterministic likelihood identities; no generated mouse cohorts."""
from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from scipy.integrate import quad
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import logistic_normal_observation as observation_module
from development_score_core import (DEVELOPMENT_RECORDS, MouseFold,
                                   score_loso_panel, session_binomial_logpmf)
from logistic_normal_observation import (
    FinalObservationFit, IntegrationSettings, ObservationError, ObservationFit, ObservationRecipe, PopulationMouseFold,
    PopulationPrediction, TrainingMouse, fit_observation_model, predict_new_mouse,
    refit_final_observation_model, score_population_loso_panel)


SETTINGS = IntegrationSettings((64, 128, 256), 1e-8, 1e-7, 1e-10, 60, 10000)
RECIPE = ObservationRecipe((-4.0, 4.0), (0.0, 0.5, 1.0), 1e-8, SETTINGS)
TRAINING = DEVELOPMENT_RECORDS[1:]


def prediction(*, sd=0.0, logits=(0.0, 0.0), offset=0.0, recipe=RECIPE):
    return PopulationPrediction(1, TRAINING, tuple(f"s{i}" for i in range(len(logits))),
                                logits, offset, sd, recipe)


def training_rows(*, successes=3, trials=4):
    return tuple(TrainingMouse(m, ("s1",), (0.0,), (successes,), (trials,)) for m in TRAINING)


class LogisticNormalObservationTests(unittest.TestCase):
    def test_final_nine_refit_known_optimum_and_distinct_unchanged_fold_interface(self):
        rows = (TrainingMouse(1, ("s1",), (0.,), (3,), (4,)),) + training_rows()
        checks = []
        final = refit_final_observation_model(training_mice=tuple(reversed(rows)), recipe=RECIPE,
                                               check=lambda: checks.append(None))
        self.assertIsInstance(final, FinalObservationFit)
        self.assertFalse(hasattr(final, "held_mouse_id"))
        self.assertEqual(final.training_mouse_ids, DEVELOPMENT_RECORDS)
        self.assertEqual(final.recipe, RECIPE)
        self.assertGreater(len(checks), 9)
        self.assertEqual(final.selected.mouse_sd, 0.)
        self.assertAlmostEqual(final.selected.population_offset, math.log(3), places=8)
        self.assertAlmostEqual(final.selected.training_log_probability,
                               9 * session_binomial_logpmf(3, 4, math.log(3)), places=10)
        fold = fit_observation_model(training_mice=training_rows(), held_mouse_id=1, recipe=RECIPE)
        self.assertIsInstance(fold, ObservationFit)
        self.assertEqual((fold.held_mouse_id, fold.training_mouse_ids), (1, TRAINING))
        self.assertEqual(fold.selected.mouse_sd, 0.)
        self.assertAlmostEqual(fold.selected.population_offset, math.log(3), places=8)
        self.assertAlmostEqual(fold.selected.training_log_probability,
                               8 * session_binomial_logpmf(3, 4, math.log(3)), places=10)
        with self.assertRaises(ObservationError):
            predict_new_mouse(final, held_mouse_id=1, session_ids=("s1",), mechanism_logits=(0.,))

    def test_final_refit_requires_exact_control_roster_recipe_and_callback_before_profile(self):
        rows = (TrainingMouse(1, ("s1",), (0.,), (3,), (4,)),) + training_rows()
        bad_rosters = (rows[:-1], rows + (rows[0],), rows[:-1] + (rows[0],),
                       rows[:-1] + (TrainingMouse(3, (), (), (), ()),),
                       rows[:-1] + (TrainingMouse(21, (), (), (), ()),))
        with patch.object(observation_module, "_fit_profile") as profile:
            for roster in bad_rosters:
                with self.subTest(roster=tuple(row.mouse_id for row in roster)), self.assertRaisesRegex(
                        ObservationError, "nine fixed development mice"):
                    refit_final_observation_model(training_mice=roster, recipe=RECIPE, check=lambda: None)
            with self.assertRaises(ObservationError):
                refit_final_observation_model(training_mice=rows, recipe=None, check=lambda: None)
            with self.assertRaisesRegex(ObservationError, "budget callback"):
                refit_final_observation_model(training_mice=rows, recipe=RECIPE, check=None)
            with self.assertRaises(TypeError):
                refit_final_observation_model(training_mice=rows, recipe=RECIPE, check=lambda: None,
                                               held_mouse_id=1)
            profile.assert_not_called()
        with self.assertRaises(ObservationError):
            fit_observation_model(training_mice=rows, held_mouse_id=1, recipe=RECIPE)

    def test_final_refit_budget_abort_inside_root_evaluation_propagates_without_result(self):
        rows = (TrainingMouse(1, ("s1",), (0.,), (3,), (4,)),) + training_rows()
        integrated = []
        original = observation_module._integrate
        stop = RuntimeError("selected resource limit")
        def counted(*args):
            result = original(*args)
            integrated.append(None)
            return result
        def check():
            # Left/right bracket likelihoods use 18 mouse evaluations. Abort
            # during the following root evaluation, rather than only at entry.
            if len(integrated) >= 20:
                raise stop
        with patch.object(observation_module, "_integrate", side_effect=counted):
            with self.assertRaises(RuntimeError) as caught:
                refit_final_observation_model(training_mice=rows,
                    recipe=replace(RECIPE, mouse_sd_grid=(0.,)), check=check)
        self.assertIs(caught.exception, stop)
        self.assertEqual(len(integrated), 20)

    def test_zero_variance_is_exact_session_binomial_product(self):
        logits, k, n = (1000.0, -1000.0, 0.7), (2, 0, 3), (2, 4, 8)
        p = prediction(logits=logits)
        expected = math.fsum(session_binomial_logpmf(a, b, c) for a, b, c in zip(k, n, logits))
        self.assertAlmostEqual(p.log_probability(successes=k, trials=n), expected, places=12)
        split = prediction(logits=(0.7, 0.7))
        combined = prediction(logits=(0.7,))
        # Splitting counts changes the combinatorial constant, not the kernel.
        a = split.log_probability(successes=(1, 2), trials=(3, 5))
        b = combined.log_probability(successes=(3,), trials=(8,))
        self.assertAlmostEqual(a - b, math.log(math.comb(3, 1) * math.comb(5, 2) / math.comb(8, 3)), places=12)
        with_zero = prediction(logits=(0.7, -1000.0, 1000.0))
        self.assertAlmostEqual(with_zero.log_probability(successes=(3, 0, 0), trials=(8, 0, 0)), b, places=12)
        self.assertEqual(with_zero.session_ids, ("s0", "s1", "s2"))

    def test_one_intercept_shared_across_sessions_matches_independent_integral(self):
        p = prediction(sd=1.0)
        expected, error = quad(lambda z: float(expit(z))**2 * math.exp(-z*z/2)
                              / math.sqrt(2*math.pi), -12, 12, epsabs=1e-12, epsrel=1e-12)
        actual = math.exp(p.log_probability(successes=(1, 1), trials=(1, 1)))
        self.assertLess(error, 1e-11)
        self.assertAlmostEqual(actual, expected, places=10)
        self.assertGreater(actual, 0.25)  # Independent per-session effects would give .25.
        one = prediction(sd=1.0, logits=(0.0,))
        self.assertAlmostEqual(math.exp(one.log_probability(successes=(1,), trials=(1,))), 0.5, places=12)
        opposite = math.exp(p.log_probability(successes=(1, 0), trials=(1, 1)))
        self.assertAlmostEqual(actual + opposite, 0.5, places=11)

    def test_training_profile_recovers_known_zero_variance_optimum(self):
        fit = fit_observation_model(training_mice=training_rows(), held_mouse_id=1, recipe=RECIPE)
        self.assertEqual(fit.selected.mouse_sd, 0.0)
        self.assertAlmostEqual(fit.selected.population_offset, math.log(3), places=8)
        expected = 8 * session_binomial_logpmf(3, 4, math.log(3))
        self.assertAlmostEqual(fit.selected.training_log_probability, expected, places=10)
        self.assertEqual(fit.training_mouse_ids, TRAINING)
        self.assertFalse(fit.selected.offset_on_boundary)
        bound_recipe = replace(RECIPE, offset_bounds=(-0.5, 0.5), mouse_sd_grid=(0.0,))
        bound = fit_observation_model(training_mice=training_rows(), held_mouse_id=1, recipe=bound_recipe)
        self.assertEqual(bound.selected.population_offset, 0.5)
        self.assertTrue(bound.selected.offset_on_boundary)

    def test_grouped_heterogeneity_selects_nonzero_grid_and_prediction_never_refits(self):
        rows = tuple(TrainingMouse(m, ("s1", "s2"), (0.0, 0.0),
                                   ((0, 0) if i < 4 else (4, 4)), (4, 4))
                     for i, m in enumerate(TRAINING))
        fit = fit_observation_model(training_mice=rows, held_mouse_id=1, recipe=RECIPE)
        self.assertEqual(fit.selected.mouse_sd, 1.0)
        self.assertAlmostEqual(fit.selected.population_offset, 0.0, places=8)
        p = predict_new_mouse(fit, held_mouse_id=1, session_ids=("s1", "s2"), mechanism_logits=(0.0, 0.0))
        before = repr(fit), repr(p)
        p.log_probability(successes=(0, 0), trials=(4, 4))
        p.log_probability(successes=(4, 4), trials=(4, 4))
        self.assertEqual((repr(fit), repr(p)), before)
        with self.assertRaises(ObservationError):
            predict_new_mouse(fit, held_mouse_id=2, session_ids=("s1",), mechanism_logits=(0.0,))

    def test_folds_counts_recipes_and_numerical_failures_are_not_silently_dropped(self):
        with self.assertRaises(ObservationError):
            fit_observation_model(training_mice=training_rows()[:-1], held_mouse_id=1, recipe=RECIPE)
        with self.assertRaises(ObservationError):
            fit_observation_model(training_mice=(replace(training_rows()[0], mouse_id=1),) + training_rows()[1:],
                                  held_mouse_id=1, recipe=RECIPE)
        for k, n in [((3, 1), (2, 2)), ((True, 0), (2, 2)), ((0, 0), (0, 0)), ((0,), (2, 2))]:
            with self.subTest(k=k, n=n), self.assertRaises(ObservationError):
                prediction().log_probability(successes=k, trials=n)
        for recipe in [replace(RECIPE, mouse_sd_grid=(1.0,)),
                       replace(RECIPE, offset_bounds=(4.0, -4.0)),
                       replace(RECIPE, integration=replace(SETTINGS, maximum_matrix_cells=1)),
                       replace(RECIPE, integration=replace(SETTINGS, maximum_root_iterations=1))]:
            with self.subTest(recipe=recipe), self.assertRaises(ObservationError):
                fit_observation_model(training_mice=training_rows(), held_mouse_id=1, recipe=recipe)
        inaccurate = replace(RECIPE, integration=replace(SETTINGS, orders=(2, 4),
                                                        log_probability_tolerance=1e-15))
        with self.assertRaisesRegex(ObservationError, "quadrature orders did not converge"):
            prediction(sd=1, recipe=inaccurate).log_probability(successes=(1, 1), trials=(1, 1))

    def test_complete_panel_preserves_seed_order_and_equal_mouse_scoring(self):
        marginal, old = [], []
        seeds = (9, 4, 1, 7, 2)
        for index, mouse in enumerate(DEVELOPMENT_RECORDS):
            roster = tuple(m for m in DEVELOPMENT_RECORDS if m != mouse)
            sessions, k, n = ("early", "late"), (index % 3, 2), (3, 4 + index)
            candidate = tuple((0.05 * seed, 0.1 * seed) for seed in seeds)
            null = tuple((0.0, 0.0) for _ in seeds)
            def make(logits):
                return tuple(PopulationPrediction(mouse, roster, sessions, row, 0.0, 0.0, RECIPE) for row in logits)
            marginal.append(PopulationMouseFold(mouse, sessions, k, n, seeds, make(candidate), make(null)))
            old.append(MouseFold(mouse, roster, sessions, k, n, seeds, candidate, null))
        actual = score_population_loso_panel(marginal, frozen_complexity=3, configuration_id="fixed")
        expected = score_loso_panel(old, frozen_complexity=3, configuration_id="fixed")
        for name in ("minimum_mouse_gain", "median_mouse_gain", "mean_mouse_gain"):
            self.assertAlmostEqual(getattr(actual, name), getattr(expected, name), places=12)
        self.assertEqual(score_population_loso_panel(tuple(reversed(marginal)), frozen_complexity=3,
                                                   configuration_id="fixed"), actual)
        changed = replace(marginal[0].candidate_predictions[0], recipe=replace(RECIPE, profile_tie_tolerance=0.1))
        bad = [replace(marginal[0], candidate_predictions=(changed,) + marginal[0].candidate_predictions[1:])] + marginal[1:]
        with self.assertRaises(ObservationError):
            score_population_loso_panel(bad, frozen_complexity=3, configuration_id="fixed")


if __name__ == "__main__":
    unittest.main()
