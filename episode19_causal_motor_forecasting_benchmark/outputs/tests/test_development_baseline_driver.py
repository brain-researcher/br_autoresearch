"""Focused composition fixtures; no source data and no repeated backend fitting.

Instrumented existing fit/reset/predict interfaces replace numerical fitting.
The full registered participant/fold/recipe/seed panel is retained, with tiny
explicit prepared arrays. Prefix partitions exercise the already-qualified
packet interface; they do not simulate or certify thirty-two real movements.
Run only in the coordinator's allocation; checks are initially UNRUN.
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
import development_baseline_driver as driver
from classical_forecast_backend import GroupIdentity, UnresolvedFit
from forecast_components import calibrated_log_probabilities
from forecast_scoring import UnresolvedScore
from non_neural_forecast_backend import ContextFeatures
from temporal_hazard_backend import ContextSequence

REGISTRY = json.loads((CODE.parent / "model_recipe_registry.json").read_text())
SCHEMA = ("explicit_fixture_context",)


def row(source, person, series, index, *, category=7, coverage=18):
    boundary = f"{source}:{person}:{series}"
    time = 3. + index * .05
    identity = GroupIdentity(source, person, series, boundary)
    summary = ContextFeatures(np.array([float(series) if source == "WAY" else 1.]),
        SCHEMA, source, person, boundary, time)
    times = time - np.array([2., 1.])
    sequence = ContextSequence(np.zeros((2, 1)), SCHEMA, source, person,
                               (boundary, boundary), times, times.copy())
    return driver.PreparedBaselineRow(identity, time, summary, sequence, category, 0, coverage)


def people(source):
    if source == "WAY":
        return tuple(driver.PreparedParticipant(person, tuple(
            row(source, person, series, index, coverage=11 if series == 1 else 18)
            for series in range(1, 7) for index in range(series)), None)
            for person in REGISTRY["folds"]["WAY_participants"])
    ids = sorted(person for fold in REGISTRY["folds"]["self_paced_validation_participants"] for person in fold)
    return tuple(driver.PreparedParticipant(person, tuple(row(source, person, "run-1", index)
        for index in range(5)), driver.CalibrationPartition(2, (-1, -1), (1, 0), 3)) for person in ids)


def probabilities(seed, series=1., recipe_index=0, *, same_recipes=False):
    primary = (.42 if same_recipes else (.46 if recipe_index == 0 else .42 if recipe_index == 1 else .25))
    primary += .01 * series + .06 * (seed - 1902)
    result = np.full(19, (1. - primary) / 13.)
    fraction = .8 if same_recipes or recipe_index == 0 else .2
    result[6:12] = primary * (1. - fraction) / 5.
    result[7] = primary * fraction
    return result


class BackendStub:
    def __init__(self, tracker, *, same_recipes=False):
        self.tracker, self.same_recipes = tracker, same_recipes

    def fit(self, batches, **kwargs):
        examples = tuple(example for batch in batches for example in batch)
        self.recipe, self.seed = kwargs["recipe"], kwargs["seed"]
        self.fitting_scope = kwargs["fitting_scope"]
        self.fitting_scope.validate([example.identity for example in examples])
        self.fitted = True
        self.recipe_index = list(item.recipe_id for item in driver.baseline_recipes(
            REGISTRY, source=self.recipe.source)).index(self.recipe.recipe_id)
        self.tracker["fits"].append((self.recipe.recipe_id, self.seed, self.fitting_scope,
                                     tuple(example.identity for example in examples)))
        self.boundary_id, self.last_time = None, -np.inf
        return {"completed": True, "recipe_id": self.recipe.recipe_id,
                "seed": self.seed, "fold_id": self.fitting_scope.fold_id}

    def reset(self, *, boundary_id):
        self.boundary_id, self.last_time = boundary_id, -np.inf
        self.tracker["resets"].append((self.seed, boundary_id))

    def predict_log_probabilities(self, *args, decision_time):
        context = args[-1]
        if self.boundary_id != context.boundary_id or decision_time <= self.last_time:
            raise AssertionError("driver failed to reset measured/prefix state before inference")
        self.last_time = decision_time
        self.tracker["predictions"].append((self.seed, context.participant, decision_time))
        return np.log(probabilities(self.seed, context.values[0], self.recipe_index,
                                    same_recipes=self.same_recipes))


class DevelopmentBaselineDriverTests(unittest.TestCase):
    def harness(self, source, *, same_recipes=False, measure=None, calibration=None, cohort=None):
        tracker = {"fits": [], "resets": [], "predictions": [], "measurements": [], "checks": 0}

        def check():
            tracker["checks"] += 1

        def measured(*, models, recipe, fitting_scope):
            self.assertEqual(tuple(model.seed for model in models), driver.SEEDS)
            self.assertTrue(all(model.fitted and model.fitting_scope == fitting_scope for model in models))
            tracker["measurements"].append((recipe.recipe_id, fitting_scope.fold_id))
            # Deliberately disturb inference state: the driver must reset before
            # any future prefix/suffix prediction, even on the same boundary.
            for model in models:
                model.boundary_id, model.last_time = "measurement-state", np.inf
            return (measure(models=models, recipe=recipe, fitting_scope=fitting_scope)
                    if measure else driver.SelectionCost(100, .01))

        settings = dict(registry=REGISTRY, source=source, participants=people(source) if cohort is None else cohort,
            summary_schema=SCHEMA, sequence_schema=SCHEMA, analysis_sample_rate_hz=1.,
            measure_ensemble=measured, parameter_count_basis="fixture actual-ensemble counting convention",
            latency_basis="fixture comparable ensemble inference measurement", budget_check=check)
        factory = mock.patch.object(driver, "_new_model", side_effect=lambda recipe:
                                    BackendStub(tracker, same_recipes=same_recipes))
        with factory:
            if calibration is None:
                result = driver.run_development_baseline_selection(**settings)
            else:
                with mock.patch.object(driver, "fit_ensemble_calibration", side_effect=calibration):
                    result = driver.run_development_baseline_selection(**settings)
        return result, tracker

    def test_complete_WAY_panel_pooling_seed_mean_primary_selection_and_NLL(self):
        recipes = driver.baseline_recipes(REGISTRY, source="WAY")
        result, tracker = self.harness("WAY", measure=lambda recipe, **kwargs:
            driver.SelectionCost(1000 if recipe == recipes[0] else 1, .1))
        self.assertEqual((result["recipes"], result["folds_per_recipe"], result["fits_completed"]), (9, 72, 1944))
        self.assertEqual(len(tracker["fits"]), 1944)
        self.assertEqual(len(tracker["measurements"]), 9 * 72)
        self.assertEqual(result["selected_recipe_id"], recipes[0].recipe_id)
        self.assertEqual(len(result["selected_oof"]), 12)
        for _, _, scope, identities in tracker["fits"]:
            self.assertEqual({item.participant for item in identities}, set(scope.allowed_participants))
            self.assertEqual(len(scope.allowed_participants), 1)
            self.assertEqual({item.series_or_run for item in identities}, set(scope.allowed_series))
            self.assertEqual(len(scope.allowed_series), 5)
        selected = result["selected_oof"][0]
        series = np.array([identity.series_or_run for identity, _ in selected["row_keys"]])
        expected = np.stack([np.mean([probabilities(seed, value) for seed in driver.SEEDS], axis=0)
                             for value in series])
        np.testing.assert_allclose(np.exp(selected["primary"]), expected, rtol=1e-13, atol=0)
        np.testing.assert_array_equal(selected["primary"], selected["zero_label"])
        scores = result["recipe_results"][0]["scores"]["primary"]
        observed = scores["horizons"]["primary_strict"]["participant_macro_log_score"]
        expected_score = np.log(expected[series != 1, 6:12].sum(axis=1)).mean()
        self.assertAlmostEqual(observed, expected_score, places=14)
        equal_series = np.mean([np.log(.46 + .01 * value) for value in range(2, 7)])
        self.assertGreater(abs(observed - equal_series), .001)
        # Count favors other models, but the registered primary score wins.
        # NLL retains all21 natural rows, including the first series whose
        # observed event lacks full primary-horizon follow-up.
        other = result["recipe_results"][1]["scores"]["primary"]
        self.assertGreater(observed, other["horizons"]["primary_strict"]["participant_macro_log_score"])
        self.assertAlmostEqual(scores["participant_macro_19_category_nll"], -np.log(expected[:, 7]).mean(), places=14)
        self.assertLess(scores["participant_macro_19_category_nll"], other["participant_macro_19_category_nll"])
        self.assertEqual(result["primary_adaptation"], "unadapted")
        self.assertFalse(result["source_admission_established"])
        self.assertFalse(result["final_refit_performed"])

    def test_self_paced_whole_people_prefix_labels_paired_suffix_and_actual_fold_costs(self):
        calibration_calls = []
        parameters = [1.3, .2, -.1, .4]

        def calibration(seed_predictions, **kwargs):
            keys = kwargs["prefix_row_keys"]
            self.assertEqual(len(keys), 2)
            self.assertEqual(set(seed_predictions), set(driver.SEEDS))
            self.assertEqual(len({identity.participant for identity, _ in keys}), 1)
            np.testing.assert_array_equal(kwargs["observed_category"], [-1, -1])
            np.testing.assert_array_equal(kwargs["complete_bins"], [1, 0])
            np.testing.assert_array_equal(kwargs["weights"], [1., 1.])
            calibration_calls.append(keys)
            # Uses the same already-tested probability-mean primitive; the
            # fixture supplies fixed calibration coefficients, never refits.
            mean = driver.mean_seed_log_probabilities(seed_predictions,
                seed_row_keys=kwargs["seed_row_keys"], expected_row_keys=keys)
            np.testing.assert_allclose(np.exp(mean), np.tile(np.mean([
                probabilities(seed, same_recipes=True) for seed in driver.SEEDS], axis=0), (2, 1)))
            return {"parameters": parameters, "identity_fallback": False}

        recipes = driver.baseline_recipes(REGISTRY, source="self_paced")

        def measure(*, models, recipe, fitting_scope):
            index = [item.recipe_id for item in recipes].index(recipe.recipe_id)
            fold = int(fitting_scope.fold_id[-1])
            counts = (100, 90, 90, 90, 200, 200, 200, 200, 200)
            times = (.010, .009, .008, .008, .001, .001, .001, .001, .001)
            return driver.SelectionCost(counts[index] - (4 - fold), times[index] * (fold + 1) / 5.)

        result, tracker = self.harness("self_paced", same_recipes=True,
                                        measure=measure, calibration=calibration)
        self.assertEqual(result["fits_completed"], 135)
        self.assertEqual(len(tracker["measurements"]), 45)
        self.assertEqual(len(calibration_calls), 9 * 15)
        self.assertEqual(len(tracker["predictions"]), 9 * 3 * 15 * 4)  # prefix2 + suffix2, once per seed
        self.assertEqual(result["selected_recipe_id"], min(recipes[2].recipe_id, recipes[3].recipe_id))
        for _, _, scope, identities in tracker["fits"]:
            self.assertEqual(len(scope.allowed_participants), 12)
            self.assertEqual({item.participant for item in identities}, set(scope.allowed_participants))
            self.assertEqual(len(identities), 12 * 5)  # full fitting-person records, including their own guard
        expected_zero = np.log(np.mean([probabilities(seed, same_recipes=True) for seed in driver.SEEDS], axis=0))
        for output in result["selected_oof"]:
            self.assertEqual([time for _, time in output["row_keys"]], [3.15, 3.2])
            np.testing.assert_allclose(output["zero_label"], np.tile(expected_zero, (2, 1)), atol=1e-14)
            np.testing.assert_allclose(output["primary"], calibrated_log_probabilities(
                np.tile(expected_zero, (2, 1)), parameters), atol=1e-14)
        for report in result["recipe_results"]:
            self.assertEqual(len(report["fold_costs"]), 5)
            self.assertEqual(report["parameter_count"], max(item["parameter_count"] for item in report["fold_costs"]))
            self.assertEqual(report["latency_seconds"], max(item["latency_seconds"] for item in report["fold_costs"]))
        self.assertEqual(result["primary_adaptation"], "calibrated32")

    def test_missing_person_series_prefix_roles_or_measurement_convention_refuses_before_fits(self):
        way = people("WAY")
        selfpaced = people("self_paced")
        cases = [
            ("WAY", way[:-1], UnresolvedFit),
            ("WAY", (replace(way[0], rows=way[0].rows[:-6]), *way[1:]), UnresolvedFit),
            ("self_paced", (replace(selfpaced[0], calibration=None), *selfpaced[1:]), ValueError),
            ("self_paced", (replace(selfpaced[0], participant="sub-01"), *selfpaced[1:]), UnresolvedFit),
            ("self_paced", (replace(selfpaced[0], calibration=driver.CalibrationPartition(2, (8, -1), (1, 0), 3)),
                            *selfpaced[1:]), ValueError),
        ]
        with mock.patch.object(driver, "_new_model") as factory:
            for source, cohort, error in cases:
                with self.assertRaises(error):
                    driver.run_development_baseline_selection(registry=REGISTRY, source=source,
                        participants=cohort, summary_schema=SCHEMA, sequence_schema=SCHEMA,
                        analysis_sample_rate_hz=1., measure_ensemble=lambda **kwargs: driver.SelectionCost(1, .1),
                        parameter_count_basis="fixed", latency_basis="fixed", budget_check=lambda: None)
            with self.assertRaisesRegex(ValueError, "measurement callable"):
                driver.run_development_baseline_selection(registry=REGISTRY, source="WAY", participants=way,
                    summary_schema=SCHEMA, sequence_schema=SCHEMA, analysis_sample_rate_hz=1.,
                    measure_ensemble=None, parameter_count_basis="fixed", latency_basis="fixed", budget_check=lambda: None)
            factory.assert_not_called()

    def test_late_failed_fit_or_unmeasured_actual_ensemble_returns_no_partial_selection(self):
        tracker = {"fits": [], "resets": [], "predictions": []}

        class FailedLast(BackendStub):
            def fit(self, batches, **kwargs):
                report = super().fit(batches, **kwargs)
                if len(tracker["fits"]) == 216:
                    self.fitted = False
                    report["completed"] = False
                return report

        settings = dict(registry=REGISTRY, source="WAY", participants=people("WAY"),
            summary_schema=SCHEMA, sequence_schema=SCHEMA, analysis_sample_rate_hz=1.,
            measure_ensemble=lambda **kwargs: driver.SelectionCost(100, .01),
            parameter_count_basis="fixed", latency_basis="fixed", budget_check=lambda: None)
        with mock.patch.object(driver, "_new_model", side_effect=lambda recipe: FailedLast(tracker)):
            with self.assertRaisesRegex(UnresolvedFit, "baseline_cell_unresolved"):
                driver.run_development_baseline_selection(**settings)
        self.assertEqual(len(tracker["fits"]), 216)
        self.assertGreater(len(tracker["predictions"]), 0)
        for missing in (None, driver.SelectionCost(100, np.nan)):
            with self.assertRaisesRegex(UnresolvedFit, "measurement_required"):
                self.harness("WAY", measure=lambda **kwargs: missing)

    def test_full_followup_masks_do_not_become_training_censoring_or_zero_scores(self):
        cohort = list(people("WAY"))
        cohort = tuple(replace(person, rows=tuple(replace(value,
            observed_through_bins=12 if value.identity.series_or_run > 1 else 11)
            for value in person.rows)) for person in cohort)
        scope, prepared = driver._prepare(REGISTRY, "WAY", cohort, SCHEMA, SCHEMA, 1., lambda: None)
        predictions = {person: {condition: np.tile(np.log(probabilities(1902)), (len(item["rows"]), 1))
                                for condition in ("primary", "zero_label")}
                       for person, item in prepared.items()}
        result = driver._summarize(scope, prepared, predictions)
        primary = result["primary"]["horizons"]["primary_strict"]
        self.assertTrue(primary["complete"])
        self.assertTrue(all(item["eligible_rows"] == 20 for item in primary["participants"]))
        diagnostic = result["primary"]["horizons"]["longer_diagnostic"]
        self.assertFalse(diagnostic["complete"])
        self.assertIsNone(diagnostic["participant_macro_log_score"])
        self.assertTrue(all(item["log_score"] is None for item in diagnostic["participants"]))
        self.assertTrue(all(item["natural_grid_rows"] == 21 for item in result["primary"]["nll_participants"]))
        no_primary = (replace(cohort[0], rows=tuple(replace(value, observed_through_bins=11)
                                                   for value in cohort[0].rows)), *cohort[1:])
        with self.assertRaises(UnresolvedScore):
            driver._prepare(REGISTRY, "WAY", no_primary, SCHEMA, SCHEMA, 1., lambda: None)


if __name__ == "__main__":
    unittest.main()
