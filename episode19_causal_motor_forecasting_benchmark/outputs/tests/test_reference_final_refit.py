"""Three development/final scope checks; coordinator Slurm execution only.

Instrumented fits test actual dispatch and identities without repeating model
training or inventing a scientific recovery case. Existing backend arithmetic
checks remain the numerical oracle. No empirical inputs are used.
"""
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import reference_final_refit as driver
from baseline_final_refit import FrozenFittingOffsets
from classical_forecast_backend import (
    FittingExample, FittingScope, GroupIdentity, PastWindow, UnresolvedFit,
    normalized_log_probabilities,
)
from ensemble_forecast_backend import EnsembleForecastModel, SEEDS
from model_recipes import reference_recipes, self_paced_folds
from non_neural_forecast_backend import logistic_hazard_recipes

REGISTRY = json.loads((CODE.parent / "model_recipe_registry.json").read_text())
CHANNELS = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
            "T7", "C3", "C4", "T8", "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
            "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2")


def fixture(source, stage="final_refit"):
    if stage == "final_refit":
        scope = FittingScope.final_refit(REGISTRY, source=source,
                                         participant=1 if source == "WAY" else None)
    else:
        scope = (FittingScope.development(REGISTRY, source="WAY", participant=1,
            fold_id="WAY.P1.validation_series6") if source == "WAY" else
            FittingScope.development(REGISTRY, source=source,
                fold_id=self_paced_folds(REGISTRY)[0].fold_id))
    rows, packets = [], []
    baseline = logistic_hazard_recipes(REGISTRY, source=source)[0].recipe_id
    for person_index, person in enumerate(scope.allowed_participants):
        person_rows = []
        for group in (scope.allowed_series if source == "WAY" else ("run1",)):
            boundary = f"{source}:{person}:{group}"
            for index in range(1 + person_index % 2):
                time = 3. + index * .05
                identity = GroupIdentity(source, person, group, boundary)
                probabilities = normalized_log_probabilities(
                    (np.linspace(-1., 1., 19) * (person_index + index + 1))[None, :])[0]
                window = PastWindow(np.zeros((5, 29)), time - np.arange(5, 0, -1) / 5.,
                                    (boundary,) * 5)
                person_rows.append(FittingExample(window, time, identity, 18, 18, probabilities))
        rows.extend(person_rows)
        # Deliberately reverse packet order: join by identity, not row ordinal.
        keys = tuple((row.identity, row.decision_time) for row in person_rows[::-1])
        values = np.asarray([row.context_log_probabilities for row in person_rows[::-1]])
        values.setflags(write=False)
        packets.append(FrozenFittingOffsets(person, keys, values, baseline, scope))
    return scope, tuple(rows), tuple(packets[::-1])


def seed_probabilities(seed):
    first = {1901: .1, 1902: .2, 1903: .6}[seed]
    return np.r_[first, np.full(18, (1. - first) / 18.)]


class BackendStub:
    def __init__(self, calls, failure_seed=None):
        self.calls, self.failure_seed = calls, failure_seed

    def fit(self, batches, *, recipe, seed, semantics):
        self.recipe, self.seed = recipe, seed
        self.fitting_scope = semantics["fitting_scope"]
        self.fitting_stage = semantics["fitting_stage"]
        self.rate, self.adapter = semantics["analysis_sample_rate_hz"], semantics["adapter"]
        rows = tuple(row for batch in batches for row in batch)
        weights = self.fitting_scope.validate([row.identity for row in rows])
        self.calls.append((seed, rows, weights, semantics))
        self.fitted = seed != self.failure_seed
        self.report = {"completed": self.fitted, "seed": seed, "recipe_id": recipe.recipe_id,
            "fold_id": self.fitting_scope.fold_id, "fitting_stage": self.fitting_stage}
        return self.report

    def reset(self, *, boundary_id):
        self.boundary_id, self.last_time = boundary_id, -np.inf

    def predict_log_probabilities(self, past_input, *, decision_time, context_log_probabilities):
        if decision_time <= self.last_time:
            raise AssertionError("prediction state must advance once per seed")
        self.last_time = decision_time
        return np.log(seed_probabilities(self.seed))


def arguments(source="WAY", stage="final_refit"):
    scope, rows, packets = fixture(source, stage)
    recipe = next(recipe for recipe in reference_recipes(REGISTRY)
                  if recipe.family == "past_only_covariance_riemannian_tangent_model")
    return rows, dict(registry=REGISTRY, selected_recipe_id=recipe.recipe_id,
        condition="real_EEG", fitting_scope=scope, fitting_offsets=packets,
        semantics={"analysis_sample_rate_hz": 5., "adapter": "original", "channel_order": CHANNELS},
        budget_check=lambda: None)


def run_ensemble(batches, stage, kwargs):
    if stage == "final_refit":
        return driver.fit_final_reference_ensemble(batches, **kwargs)
    kwargs = dict(kwargs)
    kwargs["recipe_id"] = kwargs.pop("selected_recipe_id")
    return driver.fit_development_reference_ensemble(batches, **kwargs)


class FinalReferenceRefitTests(unittest.TestCase):
    def test_complete_cohorts_exact_offset_join_three_seed_dispatch_and_probability_ensemble(self):
        for source, stage in product(("WAY", "self_paced"), ("development", "final_refit")):
            with self.subTest(source=source, stage=stage):
                rows, kwargs = arguments(source, stage)
                calls = []
                with mock.patch.object(driver, "_new_model", side_effect=lambda recipe: BackendStub(calls)):
                    result = run_ensemble((rows[:3], rows[3:]), stage, kwargs)
                self.assertEqual(tuple(call[0] for call in calls), SEEDS)
                self.assertEqual(result["fits_completed"], 3)
                self.assertTrue(result["completed_scope"])
                for field in ("complete_reference_panel", "offsets_are_out_of_fold", "calibration_performed",
                              "final_evaluation_performed", "source_admission_established",
                              "empirical_method_freeze_established", "condition_construction_established"):
                    self.assertFalse(result[field])
                self.assertEqual(result["fitting_scope"], kwargs["fitting_scope"])
                self.assertEqual(result["ensemble"].fitting_stage, stage)
                self.assertEqual(result["stage"], "final_reference_refit" if stage == "final_refit"
                                 else "development_reference_fitting_fold")
                if source == "WAY":
                    self.assertEqual(kwargs["fitting_scope"].allowed_series,
                                     (1, 2, 3, 4, 5) if stage == "development" else (1, 2, 3, 4, 5, 6, 7))
                else:
                    self.assertEqual(len(kwargs["fitting_scope"].allowed_participants),
                                     12 if stage == "development" else 15)
                for seed, joined, weights, semantics in calls:
                    self.assertEqual(semantics["fitting_stage"], stage)
                    self.assertIs(semantics["budget_check"], kwargs["budget_check"])
                    for original, row in zip(rows, joined):
                        self.assertIs(row.past_input, original.past_input)
                        self.assertEqual((row.identity, row.decision_time, row.observed_category, row.complete_bins),
                            (original.identity, original.decision_time, original.observed_category, original.complete_bins))
                        np.testing.assert_array_equal(row.context_log_probabilities, original.context_log_probabilities)
                        self.assertFalse(row.context_log_probabilities.flags.writeable)
                    for person in kwargs["fitting_scope"].allowed_participants:
                        total = sum(weight for row, weight in zip(joined, weights)
                                    if row.identity.participant == person)
                        self.assertAlmostEqual(total, 1. / len(kwargs["fitting_scope"].allowed_participants))
                ensemble = result["ensemble"]
                ensemble.reset(boundary_id="future-admitted-boundary")
                actual = ensemble.predict_log_probabilities(None, decision_time=5.,
                                                            context_log_probabilities=rows[0].context_log_probabilities)
                # Independent literal probability average, never average logits.
                expected = (seed_probabilities(1901) + seed_probabilities(1902) + seed_probabilities(1903)) / 3.
                np.testing.assert_allclose(np.exp(actual), expected, atol=1e-15, rtol=1e-14)
                members = [(ensemble.models[seed], result["fit_reports"][index])
                           for index, seed in enumerate(SEEDS)]
                if stage == "final_refit":
                    with self.assertRaises(ValueError):
                        EnsembleForecastModel(members, registry=REGISTRY, fitting_scope=kwargs["fitting_scope"])
                else:
                    accepted = EnsembleForecastModel(members, registry=REGISTRY,
                                                     fitting_scope=kwargs["fitting_scope"])
                    self.assertEqual(accepted.fitting_stage, "development")

    def test_each_actual_backend_requires_explicit_final_stage_before_consuming_batches(self):
        class ReachedRows(RuntimeError):
            pass
        families = {}
        for recipe in reference_recipes(REGISTRY):
            families.setdefault(recipe.family, recipe)
        self.assertEqual(len(families), 5)
        for source in ("WAY", "self_paced"):
            final, _, _ = fixture(source)
            development = (FittingScope.development(REGISTRY, source="WAY", participant=1,
                fold_id="WAY.P1.validation_series6") if source == "WAY" else
                FittingScope.development(REGISTRY, source=source, fold_id=self_paced_folds(REGISTRY)[0].fold_id))
            for recipe in families.values():
                with self.subTest(source=source, family=recipe.family):
                    touched = []
                    def batches():
                        touched.append(True)
                        raise ReachedRows("scope accepted before numerical work")
                        yield ()
                    semantics = dict(registry=REGISTRY, fitting_scope=final, analysis_sample_rate_hz=500.,
                        channel_order=CHANNELS, adapter="original", budget_check=lambda: None,
                        causal_filter_coefficients=tuple(np.array([1.]) for _ in range(5)))
                    with self.assertRaises(ValueError):
                        driver._new_model(recipe).fit(batches(), recipe=recipe, seed=1901, semantics=semantics)
                    self.assertEqual(touched, [])
                    with self.assertRaises(ReachedRows):
                        driver._new_model(recipe).fit(batches(), recipe=recipe, seed=1901,
                            semantics={**semantics, "fitting_stage": "final_refit"})
                    self.assertEqual(touched, [True])
                    # The same unmodified default remains usable for a real development scope.
                    with self.assertRaises(ReachedRows):
                        driver._new_model(recipe).fit(batches(), recipe=recipe, seed=1901,
                            semantics={**semantics, "fitting_scope": development})

    def test_incomplete_or_wrong_role_offsets_refuse_before_fit_and_failures_never_retry(self):
        for stage in ("development", "final_refit"):
            with self.subTest(stage=stage):
                self._assert_refusals_and_failures(stage)

    def _assert_refusals_and_failures(self, stage):
        rows, original = arguments(stage=stage)
        packet = original["fitting_offsets"][0]
        cases = [
            (rows[:-1], {**original, "fitting_offsets": (replace(packet,
                row_keys=packet.row_keys[1:], log_probabilities=packet.log_probabilities[1:]),)}),
            (rows, {**original, "fitting_scope": replace(original["fitting_scope"], allowed_series=(1, 2, 3, 4, 5, 6))}),
            (rows, {**original, "condition": "unknown"}),
            (rows, {**original, "condition": "Q11"}),
            (rows, {**original, "semantics": {**original["semantics"],
                "fitting_stage": "final_refit" if stage == "development" else "development"}}),
            (rows, {**original, "fitting_offsets": (replace(packet, row_keys=packet.row_keys[:-1] + (packet.row_keys[0],)),)}),
        ]
        for field, wrong in (("prediction_role", "out_of_fold"), ("calibration", "already_calibrated"),
                              ("seed_aggregation", "mean_logits"), ("recipe_id", "unknown")):
            cases.append((rows, {**original, "fitting_offsets": (replace(packet, **{field: wrong}),)}))
        changed = normalized_log_probabilities(packet.log_probabilities + np.arange(19)[None, :] / 10.)
        cases.append((rows, {**original, "fitting_offsets": (replace(packet, log_probabilities=changed),)}))
        self_rows, self_kwargs = arguments("self_paced", stage)
        missing_person = self_kwargs["fitting_scope"].allowed_participants[0]
        cases.append((tuple(row for row in self_rows if row.identity.participant != missing_person),
            {**self_kwargs, "fitting_offsets": tuple(packet for packet in self_kwargs["fitting_offsets"]
                                                     if packet.participant != missing_person)}))
        wrong_identity = replace(rows[-1].identity, series_or_run=8)
        cases.append((rows[:-1] + (replace(rows[-1], identity=wrong_identity),), original))
        other_stage = "final_refit" if stage == "development" else "development"
        other_rows, other_kwargs = arguments(stage=other_stage)
        cases.append((other_rows, other_kwargs))
        if stage == "development":
            for series in (6, 7):
                wrong_identity = replace(rows[-1].identity, series_or_run=series)
                cases.append((rows[:-1] + (replace(rows[-1], identity=wrong_identity),), original))
            validation_person = self_paced_folds(REGISTRY)[0].validation[0]
            wrong_identity = replace(self_rows[-1].identity, participant=validation_person)
            cases.append((self_rows[:-1] + (replace(self_rows[-1], identity=wrong_identity),), self_kwargs))
            final_only = next(condition for condition in REGISTRY["matched_conditions"]["primary"]
                if condition not in ("real_EEG", "zero_EEG", *REGISTRY["matched_conditions"]["development_surrogates"]))
            cases.append((rows, {**original, "condition": final_only}))
        for index, (supplied, kwargs) in enumerate(cases):
            with self.subTest(defect=index), mock.patch.object(driver, "_new_model") as factory:
                with self.assertRaises((ValueError, UnresolvedFit)):
                    run_ensemble((supplied,), stage, kwargs)
                factory.assert_not_called()
        calls = []
        with mock.patch.object(driver, "_new_model", side_effect=lambda recipe: BackendStub(calls, failure_seed=1902)):
            with self.assertRaisesRegex(UnresolvedFit, "seed_did_not_complete"):
                run_ensemble((rows,), stage, original)
        self.assertEqual([call[0] for call in calls], [1901, 1902])
        calls.clear()
        def cancel():
            if calls:
                raise TimeoutError("allocation ended")
        with mock.patch.object(driver, "_new_model", side_effect=lambda recipe: BackendStub(calls)):
            with self.assertRaisesRegex(TimeoutError, "allocation ended"):
                run_ensemble((rows,), stage, {**original, "budget_check": cancel})
        self.assertEqual([call[0] for call in calls], [1901])


if __name__ == "__main__":
    unittest.main()
