"""The fitted recovery screen must not confuse significance with the margin."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code"))
from run_criterion_recovery import summarize_paired_scores
from batch_criterion_recovery import implementation_digest, summarize_reports


def scores(effects):
    return {f"sub-{i:03d}": {"version": "EI" if i < 54 else "ER",
                             "M0": -1., "MC": -1.+effect}
            for i, effect in enumerate(effects)}


class RecoveryComparisonTests(unittest.TestCase):
    def test_precise_submargin_gain_fails(self):
        result = summarize_paired_scores(scores([.005]*108), 1)
        self.assertFalse(result["requirements"]["mean_gain_at_least_0p01"])
        self.assertTrue(result["requirements"]["mean_gain_greater_than_two_participant_SE"])
        self.assertFalse(result["necessary_primary_comparison_pass"])

    def test_mean_margin_without_precision_fails(self):
        result = summarize_paired_scores(scores([.011+(.1 if i%2 else -.1) for i in range(108)]), 1)
        self.assertTrue(result["requirements"]["mean_gain_at_least_0p01"])
        self.assertFalse(result["requirements"]["mean_gain_greater_than_two_participant_SE"])
        self.assertFalse(result["necessary_primary_comparison_pass"])

    def test_both_strata_and_all_registered_conditions_needed(self):
        good = summarize_paired_scores(scores([.02]*108), 1)
        self.assertTrue(good["necessary_primary_comparison_pass"])
        self.assertIsNone(good["full_recovery_pass"])
        mixed = summarize_paired_scores(scores([.05]*54+[-.001]*54), 1)
        self.assertFalse(mixed["requirements"]["positive_in_both_task_versions"])
        self.assertFalse(mixed["necessary_primary_comparison_pass"])

    def test_fixed_cell_bound_excludes_technical_failures_and_duplicate_successes(self):
        spec = {"fixed": "test"}
        digest = implementation_digest(spec)
        def report(i, valid=True):
            return (str(i), {"phase": "MC_SESOI_fitted_recovery_component",
                "replicate_index": i, "planned_cell_replicates": 200,
                "implementation": spec,
                "calibration": "five_point_decrease_in_four_second_mixture_level_acceptance_at_zero_reference_drift",
                "valid_recovery_component": valid,
                "comparison": {"necessary_primary_comparison_pass": False,
                    "requirements": {"mean_gain_at_least_0p01": False,
                                     "mean_gain_greater_than_two_participant_SE": True}}})
        first20 = [report(i) for i in range(20)]
        technical = summarize_reports(first20+[report(20, False)], 200, digest)
        self.assertEqual(technical["best_possible_completed_success_count"], 180)
        self.assertFalse(technical["necessary_90pct_cell_criterion_unreachable"])
        failed21 = summarize_reports(first20+[report(20)], 200, digest)
        self.assertEqual(failed21["best_possible_completed_success_count"], 179)
        self.assertTrue(failed21["necessary_90pct_cell_criterion_unreachable"])
        self.assertFalse(failed21["episode_terminal"])
        with self.assertRaises(ValueError):
            summarize_reports([report(0), report(0)], 200, digest)

    def test_extra_diagnostic_failure_alone_cannot_curtail_core_family_selection(self):
        spec = {"fixed": "test"}
        reports = []
        for index in range(21):
            reports.append((str(index), {
                "phase": "MC_SESOI_fitted_recovery_component", "replicate_index": index,
                "planned_cell_replicates": 200, "implementation": spec,
                "calibration": "five_point_decrease_in_four_second_mixture_level_acceptance_at_zero_reference_drift",
                "valid_recovery_component": True,
                "comparison": {"necessary_primary_comparison_pass": False,
                    "requirements": {"mean_gain_at_least_0p01": True,
                                     "mean_gain_greater_than_two_participant_SE": True,
                                     "positive_in_both_task_versions": False}}}))
        result = summarize_reports(reports, 200, implementation_digest(spec))
        self.assertEqual(result["valid_failures"], 0)
        self.assertEqual(result["valid_successes"], 21)
        self.assertEqual(result["valid_richer_primary_failures"], 21)
        self.assertEqual(result["best_possible_completed_success_count"], 200)
        self.assertFalse(result["necessary_90pct_cell_criterion_unreachable"])
        self.assertTrue(all(row["core_family_selection_pass"] for row in result["valid_reports"]))
        self.assertTrue(all(not row["counts_as_curtailment_failure"] for row in result["valid_reports"]))

    def test_two_SE_only_failure_counts_and_missing_core_flags_fail_closed(self):
        spec = {"fixed": "test"}
        value = {"phase": "MC_SESOI_fitted_recovery_component", "replicate_index": 0,
            "planned_cell_replicates": 200, "implementation": spec,
            "calibration": "five_point_decrease_in_four_second_mixture_level_acceptance_at_zero_reference_drift",
            "valid_recovery_component": True,
            "comparison": {"necessary_primary_comparison_pass": False,
                "requirements": {"mean_gain_at_least_0p01": True,
                                 "mean_gain_greater_than_two_participant_SE": False}}}
        result = summarize_reports([("one", value)], 200, implementation_digest(spec))
        self.assertEqual(result["valid_failures"], 1)
        del value["comparison"]["requirements"]["mean_gain_greater_than_two_participant_SE"]
        with self.assertRaises(ValueError):
            summarize_reports([("one", value)], 200, implementation_digest(spec))


if __name__ == "__main__":
    unittest.main()
