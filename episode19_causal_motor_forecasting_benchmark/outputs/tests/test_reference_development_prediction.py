"""Three array-interface checks; coordinator Slurm execution only, UNRUN.

Instrumented already-fitted models verify composition, not neural recovery or
calibration-optimizer qualification. Existing numerical checks remain reusable.
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
import reference_development_prediction as driver
import ensemble_forecast_backend as ensemble_api
from baseline_final_refit import FittedBaselineEnsemble
from classical_forecast_backend import FittingScope, GroupIdentity, PastWindow, UnresolvedFit
from development_baseline_driver import CalibrationPartition, PreparedBaselineRow, PreparedParticipant
from model_recipes import reference_recipes, self_paced_folds
from non_neural_forecast_backend import ContextFeatures, logistic_hazard_recipes
from temporal_hazard_backend import ContextSequence

REGISTRY = json.loads((CODE.parent / "model_recipe_registry.json").read_text())
SCHEMA = ("prepared_context",)
PARAMETERS = [1.25, .2, -.1, .05]


def baseline_probabilities(seed):
    first = {1901: .1, 1902: .2, 1903: .4}[seed]
    return np.r_[first, np.full(18, (1. - first) / 18.)]


def reference_probabilities(condition, seed):
    base = sum(baseline_probabilities(item) for item in driver.SEEDS) / 3.
    coefficient = (driver.EEG_CONDITIONS.index(condition) + 1) * (driver.SEEDS.index(seed) + 1) * .025
    values = base * np.exp(np.arange(19) * coefficient)
    return values / values.sum()


def calibrated_oracle(probabilities, parameters):
    shifts = np.r_[np.repeat(parameters[1:], 6), 0.]
    values = probabilities ** parameters[0] * np.exp(shifts)
    return values / values.sum()


class FittedStub:
    def __init__(self, recipe, seed, scope, condition, events):
        self.recipe, self.seed, self.fitting_scope = recipe, seed, scope
        self.condition, self.events = condition, events
        self.fitted, self.fitting_stage, self.rate, self.adapter = True, "development", 5., "original"
        self.feature_schema = SCHEMA
        if recipe.family == "past_only_filterbank_bandpower_elastic_net":
            self.features = SimpleNamespace(coefficients=tuple(np.array([1.]) for _ in range(5)))
        self.failure = None

    def reset(self, *, boundary_id):
        self.boundary_id, self.last_time = boundary_id, -np.inf
        self.events["resets"].append((self.condition, self.seed, boundary_id))

    def predict_log_probabilities(self, value, *, decision_time, context_log_probabilities=None):
        if decision_time <= self.last_time:
            raise AssertionError("prediction without required reset or duplicate state advance")
        self.last_time = decision_time
        offset = None if context_log_probabilities is None else context_log_probabilities.copy()
        self.events["predictions"].append((self.condition, self.seed, self.boundary_id, decision_time, offset))
        if self.failure is not None:
            raise self.failure
        if self.condition == "non_neural":
            value.validated(schema=SCHEMA, source=self.fitting_scope.source,
                participant=value.participant, boundary_id=self.boundary_id, decision_time=decision_time)
            return np.log(baseline_probabilities(self.seed))
        value.validated(decision_time=decision_time, sample_rate_hz=self.rate,
            history_seconds=self.recipe.history_seconds, boundary_id=self.boundary_id)
        if context_log_probabilities.flags.writeable:
            raise AssertionError("shared baseline offsets must remain frozen")
        coefficient = (driver.EEG_CONDITIONS.index(self.condition) + 1) * (driver.SEEDS.index(self.seed) + 1) * .025
        logits = context_log_probabilities + coefficient * np.arange(19)
        values = np.exp(logits)
        return np.log(values / values.sum())


def report(model):
    return {"completed": True, "seed": model.seed, "recipe_id": model.recipe.recipe_id,
            "fold_id": model.fitting_scope.fold_id, "fitting_stage": "development"}


def fixture(source, family="past_only_covariance_riemannian_tangent_model"):
    events = {"predictions": [], "resets": []}
    scope = (FittingScope.development(REGISTRY, source="WAY", participant=1,
        fold_id="WAY.P1.validation_series6") if source == "WAY" else
        FittingScope.development(REGISTRY, source=source, fold_id="self_paced.f0"))
    people = (1,) if source == "WAY" else self_paced_folds(REGISTRY)[0].validation
    baseline_recipe = logistic_hazard_recipes(REGISTRY, source=source)[0]
    base_models = tuple(FittedStub(baseline_recipe, seed, scope, "non_neural", events) for seed in driver.SEEDS)
    baseline = FittedBaselineEnsemble(baseline_recipe, scope, base_models, tuple(map(report, base_models)))
    baseline_fit = {"completed": True, "stage": "development_baseline_inside_EEG_fitting_fold",
        "fits_completed": 3, "seeds": driver.SEEDS, "ensemble": baseline,
        "selected_recipe_id": baseline_recipe.recipe_id, "source": source}
    eeg_recipe = next(recipe for recipe in reference_recipes(REGISTRY)
                      if recipe.family == family)
    references = {}
    for condition in driver.EEG_CONDITIONS:
        models = tuple(FittedStub(eeg_recipe, seed, scope, condition, events) for seed in driver.SEEDS)
        reports = tuple(map(report, models))
        ensemble = ensemble_api.EnsembleForecastModel(tuple(zip(models, reports)),
            registry=REGISTRY, fitting_scope=scope)
        references[condition] = {"completed_scope": True, "stage": "development_reference_fitting_fold",
            "source": source, "condition": condition, "fitting_scope": scope, "fits_completed": 3,
            "seeds": driver.SEEDS, "baseline_recipe_id": baseline_recipe.recipe_id,
            "selected_recipe_id": eeg_recipe.recipe_id, "ensemble": ensemble, "fit_reports": reports}
    participants, windows = [], {condition: {} for condition in driver.EEG_CONDITIONS}
    for person in people:
        series = 6 if source == "WAY" else "run1"
        boundary = f"{source}:{person}:{series}"
        identity = GroupIdentity(source, person, series, boundary)
        rows = []
        for index in range(5):
            time = 3. + index * .05
            context = ContextFeatures(np.array([float(index)]), SCHEMA, source, person, boundary, time)
            times = time - np.arange(10, 0, -1) / 5.
            sequence = ContextSequence(np.zeros((10, 1)), SCHEMA, source, person,
                                       (boundary,) * 10, times, times.copy())
            category = (0, 18, -1, 6, 18)[index]
            rows.append(PreparedBaselineRow(identity, time, context, sequence, category, 18, 18))
            if source == "self_paced" and index == 2:
                # A guard context that would fail if evaluated; it is unused.
                context.values[:] = np.nan
                continue
            for condition in driver.EEG_CONDITIONS:
                windows[condition][(identity, time)] = PastWindow(np.zeros((5, 29)),
                    time - np.arange(5, 0, -1) / 5., (boundary,) * 5)
        partition = None if source == "WAY" else CalibrationPartition(2, (0, -1), (0, 4), 3)
        participants.append(PreparedParticipant(person, tuple(rows), partition))
    kwargs = dict(registry=REGISTRY, baseline_fit=baseline_fit, reference_fits=references,
        validation_participants=tuple(participants), eeg_windows_by_condition=windows, budget_check=lambda: None)
    return kwargs, events


class DevelopmentReferencePredictionTests(unittest.TestCase):
    def test_both_sources_share_offsets_seed_mean_prefix_calibration_and_single_suffix(self):
        for source in ("WAY", "self_paced"):
            with self.subTest(source=source):
                kwargs, events = fixture(source)
                calibration_calls = []
                def calibration(logp, categories, complete, weights):
                    calibration_calls.append((logp.copy(), categories.copy(), complete.copy(), weights.copy()))
                    return {"parameters": PARAMETERS, "accepted": True, "identity_fallback": False}
                with mock.patch.object(ensemble_api, "fit_four_parameter_calibration", side_effect=calibration):
                    result = driver.predict_reference_development_fold(**kwargs)
                self.assertTrue(result["completed_validation_fold"])
                self.assertEqual(result["encoder_fits_performed"], 0)
                for key in ("complete_source_panel", "aggregate_feedback_produced", "source_admission_established",
                            "condition_construction_established", "calibration_prefix_qualification_established"):
                    self.assertFalse(result[key])
                self.assertEqual(result["conditions"], driver.CONDITIONS)
                base = sum(baseline_probabilities(seed) for seed in driver.SEEDS) / 3.
                expected_people = 1 if source == "WAY" else 3
                active, scored = (5, 5) if source == "WAY" else (4, 2)
                self.assertEqual(len(events["predictions"]), expected_people * 5 * 3 * active)
                self.assertEqual(len(events["resets"]), expected_people * 5 * 3 * (1 if source == "WAY" else 2))
                self.assertEqual(len(calibration_calls), 0 if source == "WAY" else 15)
                for participant, output in result["participant_predictions"].items():
                    self.assertEqual(len(output["row_keys"]), scored)
                    offsets = output["evaluation_offsets"]
                    self.assertEqual(len(offsets["row_keys"]), active)
                    self.assertFalse(offsets["log_probabilities"].flags.writeable)
                    np.testing.assert_allclose(np.exp(offsets["log_probabilities"]),
                        np.tile(base, (active, 1)), atol=1e-15, rtol=1e-14)
                    for condition in driver.CONDITIONS:
                        mean = base if condition == "non_neural" else sum(
                            reference_probabilities(condition, seed) for seed in driver.SEEDS) / 3.
                        prediction = output["predictions"][condition]
                        np.testing.assert_allclose(np.exp(prediction["zero_label"]),
                            np.tile(mean, (scored, 1)), atol=1e-15, rtol=1e-13)
                        expected = mean if source == "WAY" else calibrated_oracle(mean, PARAMETERS)
                        np.testing.assert_allclose(np.exp(prediction["primary"]),
                            np.tile(expected, (scored, 1)), atol=1e-15, rtol=1e-13)
                        self.assertFalse(prediction["primary"].flags.writeable)
                        if source == "WAY":
                            self.assertIsNone(output["calibration_reports"][condition])
                for condition, _, _, _, offset in events["predictions"]:
                    if condition != "non_neural":
                        np.testing.assert_allclose(np.exp(offset), base, atol=1e-15, rtol=1e-14)
                if source == "self_paced":
                    self.assertTrue(all(event[3] != 3.1 for event in events["predictions"]))
                    for index, (logp, categories, complete, weights) in enumerate(calibration_calls):
                        condition = driver.CONDITIONS[index % 5]
                        mean = base if condition == "non_neural" else sum(
                            reference_probabilities(condition, seed) for seed in driver.SEEDS) / 3.
                        np.testing.assert_allclose(np.exp(logp), np.tile(mean, (2, 1)), atol=1e-15, rtol=1e-13)
                        np.testing.assert_array_equal(categories, [0, -1])
                        np.testing.assert_array_equal(complete, [0, 4])
                        np.testing.assert_array_equal(weights, [1., 1.])

    def test_role_condition_seed_and_row_joins_refuse_before_any_prediction(self):
        for defect in ("missing_person", "training_person", "missing_condition", "wrong_fold", "reused_models",
                       "missing_seed", "missing_window", "guard_window", "future_prefix_label", "different_firs",
                       "final_stage"):
            with self.subTest(defect=defect):
                family = ("past_only_filterbank_bandpower_elastic_net" if defect == "different_firs" else
                          "past_only_covariance_riemannian_tangent_model")
                kwargs, events = fixture("self_paced", family=family)
                people = kwargs["validation_participants"]
                references = kwargs["reference_fits"]
                if defect == "missing_person":
                    kwargs["validation_participants"] = people[:-1]
                elif defect == "training_person":
                    person = kwargs["baseline_fit"]["ensemble"].fitting_scope.allowed_participants[0]
                    kwargs["validation_participants"] = (replace(people[0], participant=person), *people[1:])
                elif defect == "missing_condition":
                    del references["phase_1"]
                elif defect == "wrong_fold":
                    references["real_EEG"]["fitting_scope"] = FittingScope.development(REGISTRY,
                        source="self_paced", fold_id="self_paced.f1")
                elif defect == "reused_models":
                    references["zero_EEG"]["ensemble"] = references["real_EEG"]["ensemble"]
                elif defect == "missing_seed":
                    baseline = kwargs["baseline_fit"]["ensemble"]
                    kwargs["baseline_fit"]["ensemble"] = replace(baseline, models=baseline.models[:-1])
                elif defect == "missing_window":
                    values = kwargs["eeg_windows_by_condition"]["zero_EEG"]
                    del values[next(iter(values))]
                elif defect == "guard_window":
                    row = people[0].rows[2]
                    values = kwargs["eeg_windows_by_condition"]["zero_EEG"]
                    values[(row.identity, row.decision_time)] = next(iter(values.values()))
                elif defect == "future_prefix_label":
                    # Full row0 is onset bin0: it cannot become four known negative bins.
                    partition = replace(people[0].calibration, prefix_observed_category=(-1, -1),
                                        prefix_complete_bins=(4, 4))
                    kwargs["validation_participants"] = (replace(people[0], calibration=partition), *people[1:])
                elif defect == "different_firs":
                    # Each condition still has internally identical seed FIRs;
                    # only cross-condition preprocessing matching is violated.
                    for model in references["phase_1"]["ensemble"].models.values():
                        model.features.coefficients = tuple(np.array([.5, .5]) for _ in range(5))
                else:
                    kwargs["baseline_fit"]["stage"] = "final_baseline_refit_on_development_people"
                with self.assertRaises((ValueError, UnresolvedFit)):
                    driver.predict_reference_development_fold(**kwargs)
                self.assertEqual(events["predictions"], [])
                self.assertEqual(events["resets"], [])
        for series in (1, 7, 8):
            kwargs, events = fixture("WAY")
            person = kwargs["validation_participants"][0]
            rows = tuple(replace(row, identity=replace(row.identity, series_or_run=series)) for row in person.rows)
            kwargs["validation_participants"] = (replace(person, rows=rows),)
            with self.subTest(series=series), self.assertRaises(ValueError):
                driver.predict_reference_development_fold(**kwargs)
            self.assertEqual(events["predictions"], [])

    def test_failure_cancellation_and_flagged_calibration_fallback_do_not_retry(self):
        kwargs, events = fixture("WAY")
        kwargs["baseline_fit"]["ensemble"].models[1].failure = UnresolvedFit("fixture failed seed")
        with self.assertRaisesRegex(UnresolvedFit, "fixture failed seed"):
            driver.predict_reference_development_fold(**kwargs)
        self.assertEqual([event[1] for event in events["predictions"]], [1901] * 5 + [1902])
        self.assertTrue(all(event[0] == "non_neural" for event in events["predictions"]))
        kwargs, events = fixture("WAY")
        def cancel():
            if events["predictions"]:
                raise TimeoutError("allocation ended")
        kwargs["budget_check"] = cancel
        with self.assertRaisesRegex(TimeoutError, "allocation ended"):
            driver.predict_reference_development_fold(**kwargs)
        self.assertEqual(len(events["predictions"]), 1)
        kwargs, events = fixture("self_paced")
        fallback = {"parameters": [1., 0., 0., 0.], "accepted": False, "identity_fallback": True}
        with mock.patch.object(ensemble_api, "fit_four_parameter_calibration", return_value=fallback) as fitter:
            result = driver.predict_reference_development_fold(**kwargs)
        self.assertEqual(fitter.call_count, 15)
        self.assertEqual(len(events["predictions"]), 3 * 5 * 3 * 4)
        for person in result["participant_predictions"].values():
            for condition in driver.CONDITIONS:
                self.assertTrue(person["calibration_reports"][condition]["identity_fallback"])
                np.testing.assert_allclose(person["predictions"][condition]["primary"],
                    person["predictions"][condition]["zero_label"], rtol=0, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
