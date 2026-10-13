"""Two complete-roster composition checks; coordinator Slurm execution only.

Small explicit probability arrays exercise the original 72/5 fold support and
independent scalar scoring arithmetic. No model fit, source or simulation study.
"""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import reference_development_panel as panel
from classical_forecast_backend import FittingScope, GroupIdentity
from forecast_scoring import ScoringScope, UnresolvedScore
from model_recipes import reference_recipes, self_paced_folds, way_inner_folds
from non_neural_forecast_backend import logistic_hazard_recipes

REGISTRY = json.loads((CODE.parent / "model_recipe_registry.json").read_text())


def fixture(source):
    scope = ScoringScope.from_registry(REGISTRY, source=source, stage="development")
    keys, category, complete, coverage = [], [], [], []
    probability_rows = {mode: {condition: [] for condition in scope.conditions}
                        for mode in ("primary", "zero_label")}
    for person_index, person in enumerate(scope.participants):
        for series in (range(1, 7) if source == "WAY" else ("run-1",)):
            count = 9 if person_index == 0 and series in (1, "run-1") else 2
            for index in range(count):
                keys.append((GroupIdentity(source, person, series, f"{source}:{person}:{series}"), 3. + .05 * index))
                event = 7 if index % 2 else -1
                category.append(event)
                complete.append(0 if event >= 0 else 18)
                coverage.append(18)
                for condition_index, condition in enumerate(scope.conditions):
                    # Different series lengths and people have different risks.
                    strict = (.58 if person_index == 0 and series in (1, "run-1") else .12)
                    strict += .02 * condition_index
                    near, longer = .1, .08
                    probabilities = np.r_[np.full(6, near / 6), np.full(6, strict / 6),
                                          np.full(6, longer / 6), 1. - near - strict - longer]
                    primary = probabilities
                    if source == "self_paced" and person_index != 0:
                        # Independent literal alpha=1/coarse-offset calibration.
                        factors = np.exp(np.r_[np.repeat([.2, -.1, .05], 6), 0.])
                        primary = probabilities * factors
                        primary = primary / primary.sum()
                    probability_rows["zero_label"][condition].append(probabilities)
                    probability_rows["primary"][condition].append(primary)
    keys = tuple(keys)
    category, complete, coverage = map(lambda value: np.asarray(value, dtype=np.int64),
                                       (category, complete, coverage))
    probabilities = {mode: {name: np.asarray(value) for name, value in conditions.items()}
                     for mode, conditions in probability_rows.items()}
    baseline_id = logistic_hazard_recipes(REGISTRY, source=source)[0].recipe_id
    reference_id = next(reference_recipes(REGISTRY)).recipe_id
    folds = []
    roster = (tuple((person, fold) for person in scope.participants for fold in way_inner_folds(REGISTRY, person))
              if source == "WAY" else tuple((None, fold) for fold in self_paced_folds(REGISTRY)))
    for person, fold in roster:
        people = (person,) if source == "WAY" else fold.validation
        fitting_scope = FittingScope.development(REGISTRY, source=source, fold_id=fold.fold_id, participant=person)
        participants = {}
        for person in people:
            indices = [index for index, (identity, _) in enumerate(keys)
                       if identity.participant == person and (source != "WAY" or identity.series_or_run in fold.validation)]
            reports = {}
            for condition in scope.conditions:
                fallback = person == scope.participants[0]
                reports[condition] = (None if source == "WAY" else {
                    "accepted": not fallback, "identity_fallback": fallback,
                    "parameters": [1., 0., 0., 0.] if fallback else [1., .2, -.1, .05],
                    "optimizer_message": "fixture failure retained" if fallback else "fixture convergence",
                    "iterations": 2, "identity_objective": 2., "selected_objective": 2. if fallback else 1.,
                    "recipe_frozen_for_empirical_use": False,
                    "seed_aggregation": "equal_probability_mean_before_calibration",
                    "seeds": [1901, 1902, 1903], "prefix_rows": 3,
                    "prefix_eligibility_and_timing_established_by_backend": False})
            participants[person] = {
                "row_keys": tuple(keys[index] for index in indices),
                "observed_category": category[indices].copy(), "complete_bins": complete[indices].copy(),
                "observed_through_bins": coverage[indices].copy(),
                "predictions": {name: {mode: np.log(probabilities[mode][name][indices])
                                        for mode in ("primary", "zero_label")} for name in scope.conditions},
                "calibration_reports": reports,
            }
        folds.append({"source": source, "stage": "development_fold_validation_predictions",
            "fitting_scope": fitting_scope, "participants": people, "conditions": scope.conditions,
            "baseline_recipe_id": baseline_id, "reference_recipe_id": reference_id,
            "completed_validation_fold": True, "complete_source_panel": False,
            "aggregate_feedback_produced": False, "encoder_fits_performed": 0,
            "source_admission_established": False, "condition_construction_established": False,
            "calibration_prefix_qualification_established": False, "participant_predictions": participants})
    arguments = dict(registry=REGISTRY, source=source, baseline_recipe_id=baseline_id,
        reference_recipe_id=reference_id, expected_row_keys=keys, observed_category=category,
        complete_bins=complete, observed_through_bins=coverage, budget_check=lambda: None)
    return folds, arguments, probabilities


class ReferenceDevelopmentPanelTests(unittest.TestCase):
    def test_complete_cohorts_scalar_weights_alignment_calibration_and_horizon_support(self):
        for source in ("WAY", "self_paced"):
            folds, arguments, probabilities = fixture(source)
            result = panel.score_reference_development_panel(tuple(reversed(folds)), **arguments)
            self.assertEqual(result["fold_count"], 72 if source == "WAY" else 5)
            self.assertEqual(len(result["participants"]), 12 if source == "WAY" else 15)
            self.assertEqual(result["row_keys"], arguments["expected_row_keys"])
            self.assertTrue(result["complete_source_panel"])
            self.assertFalse(result["recipe_selected"])
            self.assertFalse(result["held_out_evaluated"])
            self.assertEqual(result["adaptation_modes"]["primary"], "unadapted" if source == "WAY" else "calibrated32")
            self.assertEqual(result["adaptation_modes"]["zero_label"], "unadapted" if source == "WAY" else "zero_label")
            for mode in ("primary", "zero_label"):
                for horizon, (low, high) in (("near_event", (0, 6)), ("primary_strict", (6, 12)),
                                             ("longer_diagnostic", (12, 18))):
                    actual = result["scores"][mode][horizon]
                    self.assertTrue(actual["complete"])
                    person_scores = {}
                    for condition in result["conditions"]:
                        oracle = []
                        for person in result["participants"]:
                            scalar_rows = []
                            for index, (identity, _) in enumerate(arguments["expected_row_keys"]):
                                if identity.participant != person or arguments["observed_through_bins"][index] < high:
                                    continue
                                p = probabilities[mode][condition][index]
                                inside = low <= arguments["observed_category"][index] < high
                                scalar_rows.append(math.log(sum(p[low:high]) if inside else sum(p[:low]) + sum(p[high:])))
                            oracle.append(sum(scalar_rows) / len(scalar_rows))
                        person_scores[condition] = oracle
                        np.testing.assert_allclose(actual["result"]["participant_log_scores"][condition], oracle,
                                                   atol=1e-14, rtol=0)
                        self.assertAlmostEqual(actual["result"]["macro_log_scores"][condition], sum(oracle) / len(oracle))
                    strongest = [max(a, b) for a, b in zip(person_scores["shift_1"], person_scores["phase_1"])]
                    np.testing.assert_allclose(actual["result"]["strongest_surrogate_log_scores"], strongest,
                                               atol=1e-14, rtol=0)
                    for comparator in ("non_neural", "zero_EEG"):
                        expected = [a - b for a, b in zip(person_scores["real_EEG"], person_scores[comparator])]
                        np.testing.assert_allclose(actual["result"]["participant_increments"][f"real_minus_{comparator}"],
                                                   expected, atol=1e-14, rtol=0)
                np.testing.assert_allclose(np.exp(result["predictions"][mode]["real_EEG"]),
                                           probabilities[mode]["real_EEG"], atol=1e-15, rtol=0)
                self.assertFalse(result["predictions"][mode]["real_EEG"].flags.writeable)
            actual = result["scores"]["primary"]["primary_strict"]["result"]
            counts = actual["participant_row_counts"]
            wrong_row_mean = sum(value * counts[person] for person, value in zip(result["participants"],
                actual["participant_log_scores"]["real_EEG"])) / sum(counts.values())
            self.assertGreater(abs(wrong_row_mean - actual["macro_log_scores"]["real_EEG"]), .001)
            if source == "WAY":
                self.assertEqual(result["calibration_identity_fallbacks"], ())
            else:
                self.assertEqual(len(result["calibration_identity_fallbacks"]), 5)
                retained = next(item for item in result["calibration_reports"]
                                if item["participant"] == result["participants"][0])
                self.assertEqual(retained["reports"]["real_EEG"]["optimizer_message"], "fixture failure retained")
                folds[0]["participant_predictions"][result["participants"][0]]["calibration_reports"]["real_EEG"]["optimizer_message"] = "changed caller record"
                self.assertEqual(retained["reports"]["real_EEG"]["optimizer_message"], "fixture failure retained")
            # Full primary support, but no 900ms follow-up for one retained person.
            selected = result["participants"][-1]
            for index, (identity, _) in enumerate(arguments["expected_row_keys"]):
                if identity.participant == selected:
                    arguments["observed_through_bins"][index] = 12
            for fold in folds:
                if selected in fold["participant_predictions"]:
                    fold["participant_predictions"][selected]["observed_through_bins"][:] = 12
            limited = panel.score_reference_development_panel(folds, **arguments)
            self.assertTrue(limited["scores"]["primary"]["primary_strict"]["complete"])
            self.assertFalse(limited["scores"]["primary"]["longer_diagnostic"]["complete"])
            self.assertIsNone(limited["scores"]["primary"]["longer_diagnostic"]["result"])

    def test_incomplete_drifted_role_control_rows_labels_support_and_cancellation_refuse(self):
        for source in ("WAY", "self_paced"):
            original, arguments, _ = fixture(source)
            for fault in ("missing", "duplicate", "scope", "participant", "baseline", "reference", "condition",
                          "row_order", "category", "complete", "coverage", "adaptation", "calibration", "nonfinite"):
                folds = deepcopy(original)
                fold = folds[0]
                person = fold["participants"][0]
                item = fold["participant_predictions"][person]
                if fault == "missing": folds.pop()
                elif fault == "duplicate": folds[-1] = deepcopy(fold)
                elif fault == "scope": fold["fitting_scope"] = replace(fold["fitting_scope"], allowed_participants=())
                elif fault == "participant": fold["participant_predictions"].pop(person)
                elif fault == "baseline": fold["baseline_recipe_id"] += "-changed"
                elif fault == "reference": fold["reference_recipe_id"] += "-changed"
                elif fault == "condition": item["predictions"].pop("phase_1")
                elif fault == "row_order": item["row_keys"] = item["row_keys"][::-1]
                elif fault == "category": item["observed_category"][0] = 6
                elif fault == "complete": item["complete_bins"][0] -= 1
                elif fault == "coverage": item["observed_through_bins"][0] -= 1
                elif fault == "adaptation": item["predictions"]["real_EEG"].pop("zero_label")
                elif fault == "calibration": item["calibration_reports"].pop("zero_EEG")
                elif fault == "nonfinite": item["predictions"]["real_EEG"]["primary"][0, 0] = np.nan
                with self.subTest(source=source, fault=fault), mock.patch.object(panel, "score_matched_forecasts",
                        side_effect=AssertionError("malformed panel reached scoring")):
                    with self.assertRaises((ValueError, UnresolvedScore)):
                        panel.score_reference_development_panel(folds, **arguments)
            # A separately authenticated expected row may not silently disappear.
            wrong_keys = list(arguments["expected_row_keys"])
            identity, time = wrong_keys[0]
            wrong_keys[0] = (replace(identity, participant=99 if source == "WAY" else "sub-01"), time)
            with self.assertRaisesRegex(ValueError, "role"):
                panel.score_reference_development_panel(original, **{**arguments, "expected_row_keys": wrong_keys})
            with self.assertRaisesRegex(ValueError, "duplicated"):
                panel.score_reference_development_panel(original, **{**arguments, "expected_row_keys": (*arguments["expected_row_keys"][:-1], arguments["expected_row_keys"][0])})
            folds = deepcopy(original)
            coverage = arguments["observed_through_bins"].copy()
            selected = folds[0]["participants"][0]
            for index, (identity, _) in enumerate(arguments["expected_row_keys"]):
                if identity.participant == selected: coverage[index] = 11
            for fold in folds:
                if selected in fold["participant_predictions"]:
                    fold["participant_predictions"][selected]["observed_through_bins"][:] = 11
            with self.assertRaisesRegex(UnresolvedScore, "observed support"):
                panel.score_reference_development_panel(folds, **{**arguments, "observed_through_bins": coverage})
            calls = 0
            def cancel():
                nonlocal calls
                calls += 1
                if calls == 5: raise TimeoutError("bounded cancellation")
            with mock.patch.object(panel, "score_matched_forecasts", side_effect=AssertionError("cancelled panel scored")):
                with self.assertRaisesRegex(TimeoutError, "bounded cancellation"):
                    panel.score_reference_development_panel(original, **{**arguments, "budget_check": cancel})
        folds, arguments, _ = fixture("WAY")
        item = folds[0]["participant_predictions"][1]
        item["predictions"]["real_EEG"]["primary"] = np.roll(item["predictions"]["real_EEG"]["primary"], 1, axis=1)
        with self.assertRaisesRegex(ValueError, "identical_unadapted"):
            panel.score_reference_development_panel(folds, **arguments)


if __name__ == "__main__":
    unittest.main()
