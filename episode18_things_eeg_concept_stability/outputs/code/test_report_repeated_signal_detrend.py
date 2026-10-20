"""Focused paired-person/complete-cohort reporting checks, no EEG input."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

import report_repeated_signal_detrend as reporter


def structure(person, *, adjusted=False, negative=False, offset=0.):
    matched, shift = ((.07, .01) if negative else (.15, .03) if adjusted else (.1, .02))
    matched += offset
    definition = {"lag_bin_edges_ms": list(range(0, 801, 20)), "final_bin_includes_800ms": True,
        "interpolation": False, "event_snapping": False, "prestimulus_correction": False,
        "sample_bin_minimum_by_sequence": [5] * 12, "sample_bin_maximum_by_sequence": [6] * 12}
    if adjusted:
        definition.update(lag_window_relative_to_original_event_ms=[0, 800], diagnostic_onset_shift_ms=0,
                          boundary_overlap="positive and negative windows share an observed original0ms sample")
    if negative:
        definition.update(lag_bin_edges_ms=list(range(-800, 1, 20)), final_bin_includes_0ms=True,
                          final_bin_includes_800ms=False, diagnostic_onset_shift_ms=-800,
                          lag_window_relative_to_original_event_ms=[-800, 0])
    return {"schema": "ep18_repeated_signal_structure_v1", "status": "complete_repeated_signal_structure",
        "participant_id": person, "data_role": reporter.DATA_ROLE,
        "physical_sequence_count": 12, "image_identity_count": 200,
        "recorded_channel_count": 62, "sensor_contrast_count": 61,
        "raw_EEG_detrending": adjusted, "model_fit_performed": False, "inference_performed": False,
        "original_control_threshold_changed": False, "primary_rescue": False,
        "audit_access": False, "concept_or_false_scores_computed": False,
        "sequence_metrics": [{"sequence_id": f"sequence-{i}", "ordinal": i, "retained_samples": 5376}
                             for i in range(12)],
        "template_definition": definition, "image_order": {"fixed_fixture_order": True},
        "template_metrics": {"cosine_defined": True, "matched_cosine": matched,
            "shift37_cosine": shift, "matched_minus_shift37_cosine": matched - shift}}


def rows(person, offset=0.):
    original = {"status": "complete_repeated_signal_diagnostic", "participant_id": person,
        "audit_access": False, "concept_or_false_scores_computed": False,
        "structure": structure(person, offset=offset)}
    adjusted = {"schema": reporter.model.SCHEMA, "status": "complete_repeated_signal_detrend",
        "participant_id": person, "data_role": reporter.DATA_ROLE,
        "configuration": {"analysis_id": "repeated_signal_detrend", "fixture": True},
        "projection_recipe": deepcopy(reporter.model.PROJECTION_RECIPE),
        "original_R_changed": False, "original_control_threshold_changed": False,
        "primary_rescue": False, "out_of_sample_prediction": False,
        "audit_access": False, "concept_or_false_scores_computed": False,
        "projection_accounting": {"sequence_rows": [{"sequence_id": f"sequence-{i}"} for i in range(12)]},
        "structure": structure(person, adjusted=True, offset=offset),
        "negative_lag_structure": structure(person, adjusted=True, negative=True, offset=offset)}
    return original, adjusted


class PairedDetrendReportTests(unittest.TestCase):
    def test_both_changes_pair_the_same_people_and_keep_df29(self):
        metrics = []
        for index, person in enumerate(reporter.DEVELOPMENT_PARTICIPANTS):
            original, adjusted = rows(person, index / 1000)
            metrics.append(reporter._participant(adjusted, original, person))
        raw = reporter._describe([item["raw_matched_minus_shift37_cosine"] for item in metrics])
        change = reporter._describe([item["paired_change_matched_minus_shift37_cosine"] for item in metrics])
        lag = reporter._describe([item["positive_minus_negative_matched_minus_shift37_cosine"] for item in metrics])
        self.assertEqual(change["participant_count"], 30)
        self.assertGreater(raw["standard_error"], 0.)
        self.assertAlmostEqual(change["mean"], .04)
        self.assertAlmostEqual(change["standard_error"], 0.)
        self.assertAlmostEqual(lag["mean"], .06)
        self.assertAlmostEqual(lag["standard_error"], 0.)
        # Treating repeated sequences or directions as people is forbidden.
        with self.assertRaisesRegex(ValueError, "exact30"):
            reporter._describe([item["paired_change_matched_cosine"] for item in metrics] * 2)

    def test_changed_window_order_recipe_or_primary_claim_rejected(self):
        person = reporter.DEVELOPMENT_PARTICIPANTS[0]
        original, adjusted = rows(person)
        changes = (
            ("projection_recipe", {"degree": 2}), ("original_R_changed", True),
            ("out_of_sample_prediction", True),
        )
        for key, value in changes:
            bad = deepcopy(adjusted)
            bad[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "fixed_degree_three"):
                reporter._participant(bad, original, person)
        bad = deepcopy(adjusted)
        bad["negative_lag_structure"]["template_definition"]["diagnostic_onset_shift_ms"] = -400
        with self.assertRaisesRegex(ValueError, "preserve_sequences"):
            reporter._participant(bad, original, person)
        bad = deepcopy(adjusted)
        bad["structure"]["sequence_metrics"][-1]["retained_samples"] -= 1
        with self.assertRaisesRegex(ValueError, "preserve_sequences"):
            reporter._participant(bad, original, person)
        bad = deepcopy(adjusted)
        bad["negative_lag_structure"]["image_order"] = {"changed": True}
        with self.assertRaisesRegex(ValueError, "preserve_sequences"):
            reporter._participant(bad, original, person)

    def test_undefined_negative_keeps_all_people_and_invalidates_its_paired_estimate(self):
        person = reporter.DEVELOPMENT_PARTICIPANTS[0]
        original, adjusted = rows(person)
        template = adjusted["negative_lag_structure"]["template_metrics"]
        template.update(cosine_defined=False, **dict.fromkeys(reporter.TEMPLATE_METRICS))
        metrics = reporter._participant(adjusted, original, person)
        self.assertIsNotNone(metrics["paired_change_matched_minus_shift37_cosine"])
        self.assertIsNone(metrics["positive_minus_negative_matched_minus_shift37_cosine"])
        estimate = reporter._describe([.06] * 29 + [None])
        self.assertEqual(estimate["participant_count"], 30)
        self.assertEqual(estimate["undefined_participant_count"], 1)
        self.assertIsNone(estimate["mean"])
        self.assertIsNone(estimate["descriptive_95_percent_interval"])

    def test_complete_cohort_required_before_publication_and_exact_three_lines(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            people = reporter.DEVELOPMENT_PARTICIPANTS
            manifest = {"role": "development", "participants": people, "event_count": 739440,
                        "frozen_event_join": "development_event_image_join.tsv.gz"}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            released = {"status": "complete_30_person_repeated_signal_diagnostic_report",
                "participants": people, "participant_count": 30, "participant_diagnostics": {},
                "participant_metrics": {}, "audit_access": False, "concept_or_false_scores_computed": False,
                "primary_result_rescued": False, "estimates": {"original_R": {"mean": -.000045}}}
            for index, person in enumerate(people):
                original, adjusted = rows(person, index / 1000)
                released["participant_diagnostics"][person] = original
                released["participant_metrics"][person] = {
                    key: original["structure"]["template_metrics"][key] for key in reporter.TEMPLATE_METRICS}
                if index == 29:
                    template = adjusted["negative_lag_structure"]["template_metrics"]
                    template.update(cosine_defined=False, **dict.fromkeys(reporter.TEMPLATE_METRICS))
                    final = adjusted
                    continue
                directory = root / "diagnostics" / person
                directory.mkdir(parents=True)
                (directory / "complete.json").write_text(json.dumps(adjusted))
            original_path = root / "original.json"
            original_path.write_text(json.dumps(released))
            args = dict(development_manifest=manifest_path, diagnostic_root=root / "diagnostics",
                original_report=original_path, output_root=root / "report", figure_root=root / "figures")
            with self.assertRaisesRegex(ValueError, "all30_completed"):
                reporter.report(**args)
            self.assertFalse(args["output_root"].exists())
            directory = root / "diagnostics" / people[-1]
            directory.mkdir()
            (directory / "complete.json").write_text(json.dumps(final))
            result = reporter.report(**args)
            self.assertEqual(result["participant_count"], 30)
            self.assertEqual(result["degrees_of_freedom"], 29)
            self.assertEqual(len(result["participant_diagnostics"]), 30)
            self.assertIsNone(result["estimates"]["positive_minus_negative_matched_minus_shift37_cosine"]["mean"])
            self.assertAlmostEqual(result["estimates"]["paired_change_matched_minus_shift37_cosine"]["mean"], .04)
            self.assertFalse(result["primary_result_rescued"])
            self.assertFalse(result["out_of_sample_prediction"])
            self.assertFalse(result["simultaneous_or_confirmatory_inference"])
            self.assertIsNone(result["new_pass_rule"])
            self.assertEqual(len((args["output_root"] / "summary.md").read_text().splitlines()), 3)
            for extension in ("png", "pdf"):
                self.assertGreater((args["figure_root"] / ("repeated_signal_detrend." + extension)).stat().st_size, 0)
            with self.assertRaisesRegex(ValueError, "fresh_distinct"):
                reporter.report(**args)


if __name__ == "__main__":
    unittest.main()
