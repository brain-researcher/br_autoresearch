"""Focused complete-cohort/independent-person diagnostic reporting checks."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from scipy.special import stdtrit

import report_repeated_signal_diagnostic as reporter
import repeated_signal_diagnostic as decomposition


def participant(person, gain_offset=0.):
    directions = []
    for name, denominator, poly, orth, fraction in zip(reporter.DIRECTIONS,
            (10., 100.), (.2, -.05), (.1, .15), (.8, .2)):
        poly += gain_offset
        gain = poly + orth
        directions.append({"direction": name, "gain_original_R": gain, "gain_recomputed_R": gain,
            "polynomial_gain_contribution": poly, "orthogonal_gain_contribution": orth,
            "nuisance_polynomial_energy_fraction": fraction,
            "original_total_nuisance_denominator": denominator,
            "sequence_rows": [{"sequence_id": f"sequence-{i}"} for i in range(6)],
            "released_losses_reproduced": True})
    component = {"status": "complete_repeated_signal_diagnostic", "participant_id": person,
        "schema": decomposition.SCHEMA, "diagnostic_recipe": deepcopy(decomposition.DIAGNOSTIC_RECIPE),
        "data_role": "development", "primary_R_changed": False, "inner_retuning_performed": False,
        "audit_access": False, "concept_or_false_models_computed": False, "directions": directions}
    for key in reporter.DECOMPOSITION_METRICS:
        component[key] = sum(direction[key] for direction in directions) / 2
    template = {"matched_cosine": .4, "shift37_cosine": .1, "matched_minus_shift37_cosine": .3,
        "cosine_defined": True, "matched_cross_covariance_V2": .4, "shift37_cross_covariance_V2": .1,
        "matched_minus_shift37_cross_covariance_V2": .3, "odd_energy_V2": 1., "even_energy_V2": 1.}
    sequences = [{"mean_square_dc_V2": 4., "centered_variance_V2": 2., "total_mean_square_V2": 6.,
        "spectral_power_V2": dict(zip(reporter.BANDS, (1., .75, .25))),
        "spectral_fraction_centered": dict(zip(reporter.BANDS, (.5, .375, .125)))} for _ in range(12)]
    structure = {"status": "complete_repeated_signal_structure", "participant_id": person,
        "schema": "ep18_repeated_signal_structure_v1",
        "data_role": "development_repeated_signal_recovery_diagnostic_only",
        "audit_access": False, "concept_or_false_scores_computed": False,
        "raw_EEG_detrending": False, "primary_rescue": False,
        "model_fit_performed": False, "inference_performed": False,
        "original_control_threshold_changed": False,
        "template_metrics": template, "sequence_metrics": sequences}
    return {"participant_id": person, "status": "complete_repeated_signal_diagnostic",
        "configuration": {"analysis_id": "repeated_signal_diagnostic"}, "recipe": {"fixture": True},
        "decomposition": component, "structure": structure,
        "audit_access": False, "concept_or_false_scores_computed": False}


class SignalDiagnosticReportTests(unittest.TestCase):
    def test_direction_ratios_and_components_use_original_denominator(self):
        row = participant("sub-07")
        metrics = reporter._participant(row, "sub-07", .2)
        self.assertAlmostEqual(metrics["polynomial_gain_contribution"], .075)
        self.assertAlmostEqual(metrics["orthogonal_gain_contribution"], .125)
        self.assertAlmostEqual(metrics["nuisance_polynomial_energy_fraction"], .5)
        # Opposite directional errors cannot cancel at the participant level.
        bad = deepcopy(row)
        bad["decomposition"]["directions"][0]["polynomial_gain_contribution"] += .1
        bad["decomposition"]["directions"][1]["polynomial_gain_contribution"] -= .1
        with self.assertRaisesRegex(ValueError, "each_direction_requires_original_denominator"):
            reporter._participant(bad, "sub-07", .2)
        with self.assertRaisesRegex(ValueError, "unchanged_original_R"):
            reporter._participant(row, "sub-07", .21)
        # Near-zero R is compared on the original loss-ratio reproduction scale.
        original = row["decomposition"]["gain_original_R"]
        reporter._participant(row, "sub-07", original + 5e-10)

    def test_uncertainty_has_30_people_not_60_directions_and_keeps_undefined(self):
        values = [i / 100 for i in range(30)]
        estimate = reporter._describe(values)
        self.assertEqual(estimate["participant_count"], 30)
        self.assertAlmostEqual(estimate["mean"], .145)
        # A progression has sample variance 30*31/(12*100^2).
        se = math.sqrt(31 / 120000.)
        self.assertAlmostEqual(estimate["standard_error"], se)
        self.assertAlmostEqual(estimate["descriptive_95_percent_interval"][0], .145 - stdtrit(29, .975)*se)
        with self.assertRaisesRegex(ValueError, "exact30"):
            reporter._describe(values * 2)
        values[-1] = None
        unavailable = reporter._describe(values)
        self.assertEqual(unavailable["participant_count"], 30)
        self.assertEqual(unavailable["undefined_participant_count"], 1)
        self.assertIsNone(unavailable["mean"])
        self.assertIsNone(unavailable["descriptive_95_percent_interval"])

    def test_no_partial_cohort_release_and_no_dropping_zero_energy_person(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            people = reporter.DEVELOPMENT_PARTICIPANTS
            manifest = {"role": "development", "participants": people, "event_count": 739440,
                        "frozen_event_join": "development_event_image_join.tsv.gz"}
            (root / "manifest.json").write_text(json.dumps(manifest))
            released = {"status": "complete_30_person_repeated_image_control_report",
                "participants": people, "participant_count": 30, "directions_per_participant": 2,
                "participant_G_T_R": {person: [-.0001, -.0005, -.2] for person in people},
                "audit_access": False, "concept_or_false_scores_computed": False,
                "primary_result_rescued": False, "all_three_positive_controls_pass": False,
                "endpoint_control_pass": {"G": False, "T": False, "R": False}}
            (root / "released.json").write_text(json.dumps(released))
            args = dict(development_manifest=root/"manifest.json", diagnostic_root=root/"diagnostics",
                repeated_report=root/"released.json", output_root=root/"report", figure_root=root/"figures")
            for person in people[:-1]:
                directory = root / "diagnostics" / person
                directory.mkdir(parents=True)
                (directory / "complete.json").write_text(json.dumps(participant(person, -.4)))
            with self.assertRaisesRegex(ValueError, "all30_completed"):
                reporter.report(**args)
            self.assertFalse(args["output_root"].exists())
            row = participant(people[-1], -.4)
            row["participant_id"] = "sub-99"
            directory = root / "diagnostics" / people[-1]
            directory.mkdir()
            path = directory / "complete.json"
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "complete_matching_development"):
                reporter.report(**args)
            self.assertFalse(args["output_root"].exists())
            row["participant_id"] = people[-1]
            template = row["structure"]["template_metrics"]
            template.update({key: None for key in reporter.TEMPLATE_METRICS})
            template.update(cosine_defined=False, odd_energy_V2=0., matched_cross_covariance_V2=0.,
                            shift37_cross_covariance_V2=0., matched_minus_shift37_cross_covariance_V2=0.)
            path.write_text(json.dumps(row))
            result = reporter.report(**args)
            self.assertEqual(result["participant_count"], 30)
            self.assertEqual(result["degrees_of_freedom"], 29)
            self.assertFalse(result["simultaneous_or_confirmatory_inference"])
            self.assertIsNone(result["new_pass_rule"])
            estimate = result["estimates"]["matched_minus_shift37_cosine"]
            self.assertIsNone(estimate["mean"])
            self.assertEqual(estimate["undefined_participant_count"], 1)
            self.assertEqual(len(result["participant_diagnostics"]), 30)
            self.assertEqual(len((args["output_root"] / "summary.md").read_text().splitlines()), 3)
            for extension in ("png", "pdf"):
                self.assertGreater((args["figure_root"] / ("repeated_signal_diagnostic." + extension)).stat().st_size, 0)
            with self.assertRaisesRegex(ValueError, "fresh_distinct"):
                reporter.report(**args)


if __name__ == "__main__":
    unittest.main()
