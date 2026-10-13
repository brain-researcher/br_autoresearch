from __future__ import annotations

import unittest

import numpy as np

from ep12_executor.coverage_successor import (
    CoverageSuccessorError,
    assess_cell_coverage,
    derive_error_budget,
    independent_cell_gaussian_max_t,
    next_oracle_look,
    oracle_enclosure,
    project_oracle_cost,
)


class CoverageSuccessorTests(unittest.TestCase):
    def test_error_budget_spends_one_five_percent_budget(self) -> None:
        budget = derive_error_budget(
            familywise_confidence_level=0.95,
            cell_count=35,
            planned_look_count=9,
        )
        self.assertAlmostEqual(budget.total_alpha, 0.05)
        self.assertAlmostEqual(budget.oracle_enclosure_alpha, 0.025)
        self.assertAlmostEqual(budget.oracle_model_alpha, 0.0125)
        self.assertAlmostEqual(budget.oracle_monte_carlo_alpha, 0.0125)
        self.assertAlmostEqual(budget.binomial_compatibility_alpha, 0.025)
        self.assertAlmostEqual(
            budget.oracle_look_confidence_level,
            1.0 - 0.0125 / 9.0,
        )
        self.assertAlmostEqual(
            budget.binomial_readout_confidence_level,
            1.0 - 0.025 / (35 * 4),
        )
        self.assertAlmostEqual(
            budget.oracle_model_alpha
            + budget.oracle_monte_carlo_alpha
            + budget.binomial_compatibility_alpha,
            budget.total_alpha,
        )

    def test_only_frozen_level_is_accepted(self) -> None:
        with self.assertRaisesRegex(CoverageSuccessorError, "frozen 0.95"):
            derive_error_budget(
                familywise_confidence_level=0.90,
                cell_count=1,
                planned_look_count=1,
            )

    def test_certain_possible_count_gate_passes_and_fails_closed(self) -> None:
        oracle = {
            "lower": [-0.1, -0.1, -0.1],
            "upper": [0.1, 0.1, 0.1],
            "precision_met": True,
            "maximum_half_width": 0.001,
        }
        containing = {"lower": [-1.0] * 3, "upper": [1.0] * 3}
        disjoint = {"lower": [1.0] * 3, "upper": [2.0] * 3}
        possible_only = {"lower": [0.0] * 3, "upper": [1.0] * 3}

        passed = assess_cell_coverage(
            outer_intervals=[containing] * 72,
            oracle=oracle,
            nominal_coverage=0.95,
            binomial_confidence_level=0.99375,
        )
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["status"], "coverage_qualified")

        failed = assess_cell_coverage(
            outer_intervals=[disjoint] * 72,
            oracle=oracle,
            nominal_coverage=0.95,
            binomial_confidence_level=0.99375,
        )
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["status"], "coverage_incompatible")

        ambiguous = assess_cell_coverage(
            outer_intervals=[containing] * 60 + [possible_only] * 12,
            oracle=oracle,
            nominal_coverage=0.95,
            binomial_confidence_level=0.99375,
        )
        self.assertFalse(ambiguous["passed"])
        self.assertEqual(ambiguous["status"], "unresolved_oracle_ambiguity")

    def test_imprecise_oracle_never_passes(self) -> None:
        result = assess_cell_coverage(
            outer_intervals=[{"lower": [-1.0] * 3, "upper": [1.0] * 3}] * 72,
            oracle={
                "lower": [-0.1] * 3,
                "upper": [0.1] * 3,
                "precision_met": False,
                "maximum_half_width": 0.003,
            },
            nominal_coverage=0.95,
            binomial_confidence_level=0.99375,
        )
        self.assertEqual(result["status"], "unresolved_oracle_precision")
        self.assertFalse(result["passed"])

    def test_oracle_enclosure_uses_fixed_tolerance(self) -> None:
        rows = np.column_stack(
            [np.linspace(-0.01, 0.01, 24), np.linspace(0.01, -0.01, 24)]
        )
        result = oracle_enclosure(
            rows,
            confidence_level=0.95,
            bootstrap_replicates=199,
            seed=7,
            integration_tolerance=0.002,
        )
        self.assertEqual(result["estimand"], "infinite_design_expected_fixed_finite_procedure_gain")
        self.assertTrue(result["independent_of_outer_qualification"])
        with self.assertRaisesRegex(CoverageSuccessorError, "frozen 0.002"):
            oracle_enclosure(
                rows,
                confidence_level=0.95,
                bootstrap_replicates=199,
                seed=7,
                integration_tolerance=0.003,
            )

    def test_global_multiplier_keeps_cell_streams_independent(self) -> None:
        rows = [
            np.column_stack(
                [np.linspace(-0.02, 0.02, 24), np.linspace(0.01, -0.01, 24)]
            ),
            np.column_stack(
                [np.linspace(0.03, -0.01, 24), np.linspace(-0.02, 0.02, 24)]
            ),
        ]
        first, summary = independent_cell_gaussian_max_t(
            rows,
            confidence_level=0.975,
            monte_carlo_draws=4095,
            monte_carlo_failure_probability=0.0125,
            seed=91,
            integration_tolerance=0.002,
        )
        second, _ = independent_cell_gaussian_max_t(
            rows,
            confidence_level=0.975,
            monte_carlo_draws=4095,
            monte_carlo_failure_probability=0.0125,
            seed=91,
            integration_tolerance=0.002,
        )
        self.assertEqual(first, second)
        self.assertTrue(summary["independent_cell_multiplier_streams"])
        self.assertFalse(summary["raw_index_bootstrap_used"])
        self.assertEqual(summary["coordinate_count"], 6)

    def test_bounded_adaptive_actions(self) -> None:
        looks = [72, 144, 288]
        self.assertEqual(
            next_oracle_look(
                completed_rows=72,
                planned_rows=looks,
                precision_met=False,
            )["next_rows"],
            144,
        )
        self.assertEqual(
            next_oracle_look(
                completed_rows=288,
                planned_rows=looks,
                precision_met=False,
            )["action"],
            "unresolved_at_bounded_oracle_cost",
        )
        self.assertEqual(
            next_oracle_look(
                completed_rows=144,
                planned_rows=looks,
                precision_met=True,
            )["action"],
            "oracle_precision_complete",
        )

    def test_cost_projection_is_explicit_and_bounded(self) -> None:
        report = project_oracle_cost(
            measured_cpu_seconds_per_whole_type=1.0,
            cell_count=35,
            whole_types_per_oracle_replicate=8,
            maximum_oracle_replicates_per_cell=12500,
        )
        self.assertEqual(report["maximum_generated_whole_types"], 3_500_000)
        self.assertAlmostEqual(report["projected_cpu_core_hours"], 3_500_000 / 3600)
        self.assertTrue(report["projection_required_before_full_bank_submission"])


if __name__ == "__main__":
    unittest.main()
