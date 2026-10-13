"""Final/fold composition checks; numerical fitting is instrumented, not repeated.

Run only in the coordinator's Slurm allocation. These prepared arrays exercise
group roles and existing interfaces; they are not source or timing validation.
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
import baseline_final_refit as driver
import boosted_hazard_backend as boosted
import non_neural_forecast_backend as logistic
import temporal_hazard_backend as temporal
from classical_forecast_backend import FittingScope, GroupIdentity, UnresolvedFit
from development_baseline_driver import (
    CalibrationPartition, PreparedBaselineRow, PreparedParticipant, baseline_recipes,
)

REGISTRY = json.loads((CODE.parent / "model_recipe_registry.json").read_text())
SCHEMA = ("prepared_fixture_context",)


def row(source, person, series, index):
    boundary = f"{source}:{person}:{series}"
    time = 3. + index * .05
    identity = GroupIdentity(source, person, series, boundary)
    context = logistic.ContextFeatures(np.array([float(index)]), SCHEMA, source,
                                        person, boundary, time)
    times = time - np.array([2., 1.])
    sequence = temporal.ContextSequence(np.zeros((2, 1)), SCHEMA, source, person,
                                        (boundary, boundary), times, times.copy())
    return PreparedBaselineRow(identity, time, context, sequence, 18, 18, 18)


def people(source):
    if source == "WAY":
        return tuple(PreparedParticipant(person, tuple(row(source, person, series, index)
            for series in range(1, 8) for index in range(series)), None)
            for person in REGISTRY["folds"]["WAY_participants"])
    ids = sorted(person for group in REGISTRY["folds"]["self_paced_validation_participants"]
                 for person in group)
    return tuple(PreparedParticipant(person, tuple(row(source, person, "run-1", index)
        for index in range(3 + order % 3)), CalibrationPartition(1, (-1,), (0,), 2))
        for order, person in enumerate(ids))


def probabilities(seed):
    first = {1901: .1, 1902: .2, 1903: .4}[seed]
    return np.r_[first, np.full(18, (1. - first) / 18.)]


class BackendStub:
    def __init__(self, tracker, failure_seed=None):
        self.tracker, self.failure_seed = tracker, failure_seed

    def fit(self, batches, **kwargs):
        examples = tuple(example for batch in batches for example in batch)
        self.recipe, self.seed = kwargs["recipe"], kwargs["seed"]
        self.fitting_scope = logistic._canonical_scope(REGISTRY, kwargs["fitting_scope"],
            self.recipe.source, allow_final_refit=True)
        identities = tuple(example.identity for example in examples)
        weights = self.fitting_scope.validate(identities)
        self.tracker["fits"].append((self.seed, self.fitting_scope, identities, weights))
        self.fitted = self.seed != self.failure_seed
        self.boundary_id, self.last_time = None, -np.inf
        return {"completed": self.fitted, "recipe_id": self.recipe.recipe_id,
                "seed": self.seed, "fold_id": self.fitting_scope.fold_id}

    def reset(self, *, boundary_id):
        self.boundary_id, self.last_time = boundary_id, -np.inf
        self.tracker["resets"].append((self.seed, boundary_id))

    def predict_log_probabilities(self, *args, decision_time):
        context = args[-1]
        if self.boundary_id != context.boundary_id or decision_time <= self.last_time:
            raise AssertionError("missing reset or repeated state advance")
        self.last_time = decision_time
        self.tracker["predictions"].append((self.seed, context.participant, context.boundary_id))
        return np.log(probabilities(self.seed))


class BaselineFinalRefitTests(unittest.TestCase):
    def harness(self, source, *, cohort=None, scope=None, failure_seed=None, check=None):
        tracker = {"fits": [], "resets": [], "predictions": []}
        arguments = dict(registry=REGISTRY,
            selected_recipe_id=baseline_recipes(REGISTRY, source=source)[0].recipe_id,
            participants=people(source) if cohort is None else cohort,
            summary_schema=SCHEMA, sequence_schema=SCHEMA, analysis_sample_rate_hz=1.,
            budget_check=check or (lambda: None))
        with mock.patch.object(driver, "_new_model", side_effect=lambda recipe:
                BackendStub(tracker, failure_seed=failure_seed)):
            if scope is None:
                result = driver.fit_final_baselines(source=source, **arguments)
            else:
                result = driver.fit_fitting_fold_baseline(fitting_scope=scope, **arguments)
        return result, tracker

    def assert_offsets(self, result, expected_people):
        expected = np.mean([probabilities(seed) for seed in driver.SEEDS], axis=0)
        self.assertFalse(result["offsets_are_out_of_fold"])
        self.assertFalse(result["calibration_performed"])
        self.assertEqual(len(result["fitting_offsets"]), len(expected_people))
        for offset, person in zip(result["fitting_offsets"], expected_people):
            self.assertEqual(offset.participant, person.participant)
            self.assertEqual(offset.row_keys, tuple((item.identity, item.decision_time) for item in person.rows))
            np.testing.assert_allclose(np.exp(offset.log_probabilities),
                np.broadcast_to(expected, offset.log_probabilities.shape), rtol=1e-13, atol=0)
            self.assertFalse(offset.log_probabilities.flags.writeable)
            self.assertEqual(offset.prediction_role, "training_rows_of_frozen_baseline_not_out_of_fold")
            self.assertEqual(offset.calibration, "uncalibrated_before_EEG_fusion")

    def test_all_twelve_WAY_final_fits_use_seven_series_and_probability_mean(self):
        result, tracker = self.harness("WAY")
        self.assertEqual(result["fits_completed"], 36)
        self.assertEqual(len(result["ensembles"]), 12)
        self.assertEqual(len(tracker["fits"]), 36)
        for seed, scope, identities, weights in tracker["fits"]:
            self.assertIn(seed, driver.SEEDS)
            self.assertEqual(len(scope.allowed_participants), 1)
            self.assertEqual(scope.allowed_series, tuple(range(1, 8)))
            self.assertEqual({identity.series_or_run for identity in identities}, set(range(1, 8)))
            np.testing.assert_allclose(weights, np.full(len(identities), 1. / len(identities)))
        self.assert_offsets(result, people("WAY"))
        self.assertFalse(result["final_evaluation_performed"])

    def test_self_paced_refit_is_one_all_fifteen_group_with_equal_person_weights(self):
        result, tracker = self.harness("self_paced")
        self.assertEqual(result["fits_completed"], 3)
        self.assertEqual(len(result["ensembles"]), 1)
        for _, scope, identities, weights in tracker["fits"]:
            self.assertEqual(len(scope.allowed_participants), 15)
            self.assertIsNone(scope.allowed_series)
            for person in scope.allowed_participants:
                mask = np.array([identity.participant == person for identity in identities])
                self.assertAlmostEqual(float(weights[mask].sum()), 1. / 15.)
            # Fitting keeps prefixes and guards, irrespective of scoring partition.
            self.assertEqual(len(identities), sum(len(person.rows) for person in people("self_paced")))
        self.assert_offsets(result, people("self_paced"))

    def test_development_fitting_offsets_exclude_outer_validation_groups(self):
        way_scope = FittingScope.development(REGISTRY, source="WAY",
            fold_id="WAY.P1.validation_series6", participant=1)
        first = people("WAY")[0]
        way_people = (replace(first, rows=tuple(item for item in first.rows
                                               if item.identity.series_or_run in way_scope.allowed_series)),)
        way, tracker = self.harness("WAY", cohort=way_people, scope=way_scope)
        self.assertEqual(way["fits_completed"], 3)
        self.assert_offsets(way, way_people)
        self.assertTrue(all(set(identity.series_or_run for identity in fit[2]) == set(range(1, 6))
                            for fit in tracker["fits"]))
        group_scope = FittingScope.development(REGISTRY, source="self_paced", fold_id="self_paced.f0")
        group_people = tuple(person for person in people("self_paced")
                             if person.participant in group_scope.allowed_participants)
        group, _ = self.harness("self_paced", cohort=group_people, scope=group_scope)
        self.assertEqual(group["fits_completed"], 3)
        self.assert_offsets(group, group_people)
        with self.assertRaises(UnresolvedFit):
            self.harness("self_paced", scope=group_scope)
        with self.assertRaises(ValueError):
            self.harness("WAY", cohort=(first,), scope=way_scope)

    def test_only_baseline_interfaces_opt_into_explicit_final_scope(self):
        scope = FittingScope.final_refit(REGISTRY, source="WAY", participant=1)
        with self.assertRaises(ValueError):
            logistic._canonical_scope(REGISTRY, scope, "WAY")
        with self.assertRaises(ValueError):
            logistic._canonical_scope(REGISTRY, replace(scope, allowed_series=(1, 2, 3, 4, 5, 6)),
                                      "WAY", allow_final_refit=True)
        with self.assertRaises(ValueError):
            FittingScope.final_refit(REGISTRY, source="self_paced", participant="sub-02")
        rows = people("WAY")[0].rows
        examples = tuple(logistic.BaselineFittingExample(item.context, item.decision_time,
            item.identity, item.observed_category, item.complete_bins) for item in rows)
        linear_fit = SimpleNamespace(converged=True, iterations=1, objective=0., reason="instrumented")
        boosted_fit = SimpleNamespace(completed=True, reason="instrumented", unresolved_head=None, objective=0.)
        with mock.patch.object(logistic, "fit_logistic_hazard", return_value=linear_fit) as kernel:
            model = logistic.LogisticHazardModel()
            model.fit((examples,), recipe=logistic.logistic_hazard_recipes(REGISTRY, source="WAY")[0],
                seed=1901, registry=REGISTRY, fitting_scope=scope, feature_schema=SCHEMA)
            self.assertTrue(model.fitted)
            self.assertEqual(kernel.call_count, 1)
        with mock.patch.object(boosted, "fit_boosted_hazard", return_value=boosted_fit) as kernel:
            model = boosted.BoostedHazardModel()
            model.fit((examples,), recipe=boosted.boosted_hazard_recipes(REGISTRY, source="WAY")[0],
                seed=1901, registry=REGISTRY, fitting_scope=scope, feature_schema=SCHEMA)
            self.assertTrue(model.fitted)
            self.assertEqual(kernel.call_count, 1)
        temporal_examples = tuple(temporal.TemporalFittingExample(item.sequence, item.context,
            item.decision_time, item.identity, item.observed_category, item.complete_bins) for item in rows)
        with mock.patch.object(temporal, "_train_fixed_schedule", return_value=(object(), 1, 0.)) as kernel:
            model = temporal.TemporalHazardModel()
            model.fit((temporal_examples,), recipe=temporal.temporal_hazard_recipes(REGISTRY, source="WAY")[0],
                seed=1901, registry=REGISTRY, fitting_scope=scope, sequence_schema=SCHEMA,
                summary_schema=SCHEMA, analysis_sample_rate_hz=1., budget_check=lambda: None)
            self.assertTrue(model.fitted)
            self.assertEqual(kernel.call_count, 1)

    def test_missing_groups_held_out_rows_budget_and_failed_seed_refuse(self):
        cohort = people("WAY")
        with self.assertRaises(UnresolvedFit):
            self.harness("WAY", cohort=cohort[:-1])
        missing_seven = replace(cohort[0], rows=tuple(item for item in cohort[0].rows
                                                     if item.identity.series_or_run != 7))
        with self.assertRaises(UnresolvedFit):
            self.harness("WAY", cohort=(missing_seven, *cohort[1:]))
        held_out = replace(cohort[0], rows=(*cohort[0].rows, row("WAY", 1, 8, 0)))
        with self.assertRaises(ValueError):
            self.harness("WAY", cohort=(held_out, *cohort[1:]))
        group = people("self_paced")
        with self.assertRaises(UnresolvedFit):
            self.harness("self_paced", cohort=(replace(group[0], participant="sub-01"), *group[1:]))
        with self.assertRaises(UnresolvedFit):
            self.harness("self_paced", failure_seed=1902)
        with self.assertRaisesRegex(RuntimeError, "allocation deadline"):
            self.harness("self_paced", check=lambda: (_ for _ in ()).throw(RuntimeError("allocation deadline")))


if __name__ == "__main__":
    unittest.main()
