"""Three focused integration checks; coordinator Slurm execution only.

Instrumented EEG fits exercise the existing seed loop, actual fold prediction,
calibration and complete-source scorer. These are small software fixtures, not
a simulated scientific study or a repetition of backend numerical training.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest import mock
import weakref

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import reference_development_driver as driver
import reference_final_refit as fit_api
from baseline_final_refit import FittedBaselineEnsemble, FrozenFittingOffsets
from classical_forecast_backend import FittingScope, GroupIdentity, PastWindow, UnresolvedFit
from development_baseline_driver import CalibrationPartition, PreparedBaselineRow, PreparedParticipant
from ensemble_forecast_backend import SEEDS
from forecast_scoring import ScoringScope
from model_recipes import reference_recipes, self_paced_folds, way_inner_folds
from non_neural_forecast_backend import ContextFeatures, logistic_hazard_recipes
from temporal_hazard_backend import ContextSequence

REGISTRY = json.loads((CODE.parent / "model_recipe_registry.json").read_text())
SCHEMA = ("prepared_context",)
CODES = {"real_EEG": 1., "zero_EEG": 0., "shift_1": 2., "phase_1": 3.}


def baseline_probabilities(seed):
    first = {1901: .1, 1902: .2, 1903: .4}[seed]
    return np.r_[first, np.full(18, (1. - first) / 18.)]


class BaselineStub:
    def __init__(self, recipe, scope, seed):
        self.recipe, self.fitting_scope, self.seed = recipe, scope, seed
        self.fitted = True

    def reset(self, *, boundary_id):
        self.boundary_id, self.last_time = boundary_id, -np.inf

    def predict_log_probabilities(self, value, *, decision_time):
        if decision_time <= self.last_time:
            raise AssertionError("missing baseline boundary/prefix reset")
        self.last_time = decision_time
        value.validated(schema=SCHEMA, source=self.fitting_scope.source,
            participant=value.participant, boundary_id=self.boundary_id, decision_time=decision_time)
        return np.log(baseline_probabilities(self.seed))


class EEGStub:
    def __init__(self, events):
        self.events = events

    def fit(self, batches, *, recipe, seed, semantics):
        rows = tuple(row for batch in batches for row in batch)
        self.recipe, self.seed, self.fitting_scope = recipe, seed, semantics["fitting_scope"]
        self.rate, self.adapter, self.fitting_stage = semantics["analysis_sample_rate_hz"], "original", "development"
        self.condition = next(name for name, code in CODES.items() if code == rows[0].past_input.values[0, 0])
        expected = np.log(sum(baseline_probabilities(s) for s in SEEDS) / 3.)
        signature = []
        for row in rows:
            np.testing.assert_allclose(row.context_log_probabilities, expected, atol=1e-15, rtol=0)
            if row.context_log_probabilities.flags.writeable:
                raise AssertionError("fitting offsets must remain frozen")
            row.past_input.validated(decision_time=row.decision_time, sample_rate_hz=self.rate,
                history_seconds=recipe.history_seconds, boundary_id=row.identity.boundary_id)
            signature.append((row.identity, row.decision_time, row.observed_category, row.complete_bins))
        self.events["calls"].append((self.fitting_scope.fold_id, self.condition, seed, tuple(signature)))
        if len(self.events["calls"]) == self.events.get("fail_at"):
            raise RuntimeError("observed required seed failure")
        self.fitted = True
        return {"completed": True, "seed": seed, "recipe_id": recipe.recipe_id,
                "fold_id": self.fitting_scope.fold_id, "fitting_stage": "development"}

    def reset(self, *, boundary_id):
        self.boundary_id, self.last_time = boundary_id, -np.inf

    def predict_log_probabilities(self, value, *, decision_time, context_log_probabilities):
        if decision_time <= self.last_time:
            raise AssertionError("missing EEG boundary/prefix reset")
        self.last_time = decision_time
        value.validated(decision_time=decision_time, sample_rate_hz=self.rate,
            history_seconds=self.recipe.history_seconds, boundary_id=self.boundary_id)
        logits = context_log_probabilities + .005 * CODES[self.condition] * np.arange(19)
        probabilities = np.exp(logits)
        return np.log(probabilities / probabilities.sum())


def fixture(source):
    scoring = ScoringScope.from_registry(REGISTRY, source=source, stage="development")
    baseline_recipe = logistic_hazard_recipes(REGISTRY, source=source)[0]
    reference = next(recipe for recipe in reference_recipes(REGISTRY)
        if recipe.family == "past_only_covariance_riemannian_tangent_model")
    all_people, expected_keys, category, complete, coverage = {}, [], [], [], []
    for person in scoring.participants:
        rows = []
        for series in (range(1, 7) if source == "WAY" else ("run1",)):
            boundary = f"{source}:{person}:{series}"
            identity = GroupIdentity(source, person, series, boundary)
            for index in range(2 if source == "WAY" else 5):
                time = 3. + .05 * index
                context = ContextFeatures(np.array([float(index)]), SCHEMA, source, person, boundary, time)
                times = time - np.arange(10, 0, -1) / 5.
                sequence = ContextSequence(np.zeros((10, 1)), SCHEMA, source, person,
                    (boundary,) * 10, times, times.copy())
                event = (7, -1)[index] if source == "WAY" else (0, 18, -1, 7, -1)[index]
                row = PreparedBaselineRow(identity, time, context, sequence, event, 0 if event >= 0 else 18, 18)
                rows.append(row)
                if source == "WAY" or index >= 3:
                    expected_keys.append((identity, time))
                    category.append(row.observed_category)
                    complete.append(row.complete_bins)
                    coverage.append(row.observed_through_bins)
        partition = None if source == "WAY" else CalibrationPartition(2, (0, -1), (0, 4), 3)
        all_people[person] = PreparedParticipant(person, tuple(rows), partition)

    def windows(people, *, validation):
        result = {condition: {} for condition in CODES}
        for person in people:
            for index, row in enumerate(person.rows):
                if validation and source == "self_paced" and index == 2:
                    continue
                key = row.identity, row.decision_time
                for condition, code in CODES.items():
                    result[condition][key] = PastWindow(np.full((5, 29), code),
                        row.decision_time - np.arange(5, 0, -1) / 5., (row.identity.boundary_id,) * 5)
        return result

    roster = (tuple((person, fold) for person in scoring.participants for fold in way_inner_folds(REGISTRY, person))
              if source == "WAY" else tuple((None, fold) for fold in self_paced_folds(REGISTRY)))
    folds = {}
    for person, fold in roster:
        scope = FittingScope.development(REGISTRY, source=source, participant=person, fold_id=fold.fold_id)
        if source == "WAY":
            whole = all_people[person]
            fitting = (replace(whole, rows=tuple(row for row in whole.rows if row.identity.series_or_run in fold.fit)),)
            validation = (replace(whole, rows=tuple(row for row in whole.rows if row.identity.series_or_run in fold.validation)),)
        else:
            fitting = tuple(all_people[p] for p in fold.fit)
            validation = tuple(all_people[p] for p in fold.validation)
        models = tuple(BaselineStub(baseline_recipe, scope, seed) for seed in SEEDS)
        reports = tuple({"completed": True, "seed": seed, "recipe_id": baseline_recipe.recipe_id,
                         "fold_id": scope.fold_id} for seed in SEEDS)
        offsets = []
        for prepared in fitting:
            keys = tuple((row.identity, row.decision_time) for row in prepared.rows)
            values = np.tile(np.log(sum(baseline_probabilities(s) for s in SEEDS) / 3.), (len(keys), 1))
            values.setflags(write=False)
            offsets.append(FrozenFittingOffsets(prepared.participant, keys, values, baseline_recipe.recipe_id, scope))
        baseline = {"completed": True, "stage": "development_baseline_inside_EEG_fitting_fold",
            "source": source, "selected_recipe_id": baseline_recipe.recipe_id, "fits_completed": 3,
            "seeds": SEEDS, "ensemble": FittedBaselineEnsemble(baseline_recipe, scope, models, reports),
            "fitting_offsets": tuple(offsets)}
        folds[fold.fold_id] = {"baseline_fit": baseline, "fitting_participants": fitting,
            "fitting_windows_by_condition": windows(fitting, validation=False),
            "semantics": {"analysis_sample_rate_hz": 5., "adapter": "original"},
            "validation_participants": validation,
            "validation_windows_by_condition": windows(validation, validation=True)}
    return dict(registry=REGISTRY, source=source, baseline_recipe_id=baseline_recipe.recipe_id,
        reference_recipe_id=reference.recipe_id, fold_inputs=dict(reversed(tuple(folds.items()))),
        expected_row_keys=tuple(expected_keys), observed_category=np.asarray(category),
        complete_bins=np.asarray(complete), observed_through_bins=np.asarray(coverage), budget_check=lambda: None)


class ReferenceDevelopmentDriverTests(unittest.TestCase):
    def test_complete_sources_use_exact_existing_seed_loops_and_release_each_fold(self):
        for source in ("WAY", "self_paced"):
            arguments = fixture(source)
            events = {"calls": [], "references": []}
            def make_backend(recipe):
                alive = sum(reference() is not None for reference in events["references"])
                self.assertLess(alive, 12)
                if len(events["calls"]) % 12 == 0:
                    self.assertEqual(alive, 0, "previous fold's EEG models were retained")
                model = EEGStub(events)
                events["references"].append(weakref.ref(model))
                return model
            with mock.patch.object(fit_api, "_new_model", side_effect=make_backend):
                result = driver.run_reference_development_recipe(**arguments)
            count = 72 if source == "WAY" else 5
            self.assertEqual(result["fold_count"], count)
            self.assertEqual(result["execution"]["EEG_fits_completed"], count * 12)
            self.assertEqual(result["execution"]["baseline_fits_performed"], 0)
            self.assertEqual(len(events["calls"]), count * 12)
            self.assertTrue(all(reference() is None for reference in events["references"]))
            for index, fold_id in enumerate(result["fold_ids"]):
                calls = events["calls"][index * 12:(index + 1) * 12]
                self.assertEqual([(call[0], call[1], call[2]) for call in calls],
                    [(fold_id, condition, seed) for condition in driver.EEG_CONDITIONS for seed in SEEDS])
                self.assertTrue(all(call[3] == calls[0][3] for call in calls))
            self.assertEqual(result["row_keys"], arguments["expected_row_keys"])
            self.assertTrue(result["complete_source_panel"])
            self.assertFalse(result["source_admission_established"])
            self.assertFalse(result["recipe_selected"])
            self.assertFalse(result["held_out_evaluated"])
            self.assertEqual(len(result["calibration_reports"]), count if source == "WAY" else 15)
            self.assertTrue(result["scores"]["primary"]["primary_strict"]["complete"])

    def test_last_fold_roster_condition_offset_and_label_faults_refuse_before_any_fit(self):
        original = fixture("self_paced")
        for fault in ("missing_fold", "condition", "fit_window", "offset_recipe", "offset_scope",
                      "offset_missing", "offset_nonfinite", "label", "coverage", "baseline"):
            with self.subTest(fault=fault):
                args = deepcopy(original)
                last = args["fold_inputs"]["self_paced.f4"]
                if fault == "missing_fold":
                    del args["fold_inputs"]["self_paced.f4"]
                elif fault == "condition":
                    del last["fitting_windows_by_condition"]["phase_1"]
                elif fault == "fit_window":
                    last["fitting_windows_by_condition"]["phase_1"].pop(next(iter(last["fitting_windows_by_condition"]["phase_1"])))
                elif fault.startswith("offset"):
                    packets = list(last["baseline_fit"]["fitting_offsets"])
                    packet = packets[0]
                    if fault == "offset_recipe":
                        packets[0] = replace(packet, recipe_id="unknown")
                    elif fault == "offset_scope":
                        packets[0] = replace(packet, fitting_scope=FittingScope.final_refit(REGISTRY, source="self_paced"))
                    elif fault == "offset_missing":
                        packets.pop()
                    else:
                        values = packet.log_probabilities.copy()
                        values[0, 0] = np.nan
                        packets[0] = replace(packet, log_probabilities=values)
                    last["baseline_fit"]["fitting_offsets"] = tuple(packets)
                elif fault in ("label", "coverage"):
                    participants = list(last["validation_participants"])
                    rows = list(participants[0].rows)
                    rows[-1] = replace(rows[-1], **({"observed_category": 7} if fault == "label"
                                                   else {"observed_through_bins": 12}))
                    participants[0] = replace(participants[0], rows=tuple(rows))
                    last["validation_participants"] = tuple(participants)
                else:
                    last["baseline_fit"]["fits_completed"] = 2
                with mock.patch.object(fit_api, "_new_model") as backend, mock.patch.object(driver, "score_reference_development_panel") as score:
                    with self.assertRaises((ValueError, UnresolvedFit)):
                        driver.run_reference_development_recipe(**args)
                    backend.assert_not_called()
                    score.assert_not_called()

    def test_required_seed_failure_and_budget_stop_never_retry_or_score_partial_source(self):
        for cause in ("fit", "budget"):
            arguments = fixture("self_paced")
            events = {"calls": [], "fail_at": 4 if cause == "fit" else None}
            def check():
                if cause == "budget" and len(events["calls"]) >= 4:
                    raise TimeoutError("allocation budget exhausted")
            arguments["budget_check"] = check
            with mock.patch.object(fit_api, "_new_model", side_effect=lambda recipe: EEGStub(events)), \
                    mock.patch.object(driver, "score_reference_development_panel") as score:
                with self.assertRaises(RuntimeError if cause == "fit" else TimeoutError):
                    driver.run_reference_development_recipe(**arguments)
                self.assertEqual(len(events["calls"]), 4)
                score.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
