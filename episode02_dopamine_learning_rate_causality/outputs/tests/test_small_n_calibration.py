#!/usr/bin/env python3
"""Mechanical tests for EP02's outcome-blind small-n calibration."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


EPISODE = Path(__file__).resolve().parents[2]
CALIBRATION = Path(
    os.environ.get("EP02_GENERIC_CALIBRATION_DIR", EPISODE / "outputs" / "calibration")
)
MODULE_PATH = EPISODE / "outputs" / "code" / "calibrate_small_n.py"
SPEC = importlib.util.spec_from_file_location("calibrate_small_n", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CalibrationFunctionTests(unittest.TestCase):
    def test_observed_on_is_required(self) -> None:
        with patch.object(sys, "argv", ["calibrate_small_n.py", "--output-dir", "output"]), self.assertRaises(SystemExit):
            MODULE.parse_args()

    @staticmethod
    def development_row() -> np.ndarray:
        return np.linspace(-1.0, 1.0, 9)[None, :]

    def test_assignment_space_is_complete(self) -> None:
        weights, assignments = MODULE.assignment_weights()
        self.assertEqual(len(assignments), math.comb(11, 6))
        self.assertEqual(weights.shape, (462, 11))
        np.testing.assert_allclose(weights.sum(axis=1), 0.0, atol=1e-12)
        self.assertEqual(assignments[0], tuple(range(6)))

    def test_perfect_oriented_separation_has_minimum_p(self) -> None:
        weights, _ = MODULE.assignment_weights()
        scores = np.array([[1.0] * 6 + [-1.0] * 5])
        metrics = MODULE.scenario_metrics(
            self.development_row(), scores, weights, (0.50,), (0.05, 0.10)
        )
        self.assertAlmostEqual(float(metrics["p_zero"][0]), 1.0 / 462.0)
        self.assertAlmostEqual(float(metrics["p_margin_0.50"][0]), 1.0 / 462.0)
        self.assertGreater(float(metrics["loo_minimum_delta"][0]), 0.0)

    def test_reversed_separation_does_not_pass(self) -> None:
        weights, _ = MODULE.assignment_weights()
        scores = np.array([[-1.0] * 6 + [1.0] * 5])
        metrics = MODULE.scenario_metrics(
            self.development_row(),
            scores,
            weights,
            (0.25, 0.50, 0.75),
            (0.05, 0.10),
        )
        self.assertEqual(float(metrics["p_zero"][0]), 1.0)
        for rule in MODULE.rule_grid():
            self.assertFalse(bool(MODULE.passes_rule(metrics, rule)[0]))

    def test_scenario_seed_is_order_independent(self) -> None:
        first = MODULE.stable_scenario_seed(20260922, "rate_moderate")
        second = MODULE.stable_scenario_seed(20260922, "rate_moderate")
        other = MODULE.stable_scenario_seed(20260922, "null_gaussian")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_shifted_exact_p_matches_independent_brute_force(self) -> None:
        weights, assignments = MODULE.assignment_weights()
        scores = np.array([[2.0, 1.5, 1.0, 0.5, 0.0, -0.5, 1.0, 0.0, -1.0, -1.5, -2.0]])
        margin = 0.50
        metrics = MODULE.scenario_metrics(
            self.development_row(), scores, weights, (margin,), (0.05,)
        )
        standardized = (scores[0] - self.development_row().mean()) / self.development_row().std(ddof=1)
        shifted = standardized.copy()
        shifted[:6] -= margin
        observed = shifted[:6].mean() - shifted[6:].mean()
        brute = []
        all_indices = set(range(11))
        for chosen in assignments:
            other = sorted(all_indices - set(chosen))
            brute.append(shifted[list(chosen)].mean() - shifted[other].mean())
        expected = np.mean(np.asarray(brute) >= observed - 1e-14)
        self.assertAlmostEqual(float(metrics["p_margin_0.50"][0]), float(expected))

    def test_exact_p_includes_ties_as_extreme(self) -> None:
        weights, assignments = MODULE.assignment_weights()
        scores = np.array([[2, 2, 1, 1, 0, 0, 0, 0, -1, -1, -2]], dtype=float)
        metrics = MODULE.scenario_metrics(
            self.development_row(), scores, weights, (0.50,), (0.05,)
        )
        observed = scores[0, :6].mean() - scores[0, 6:].mean()
        all_indices = set(range(11))
        brute = []
        for chosen in assignments:
            other = sorted(all_indices - set(chosen))
            brute.append(scores[0, list(chosen)].mean() - scores[0, other].mean())
        self.assertAlmostEqual(
            float(metrics["p_zero"][0]),
            float(np.mean(np.asarray(brute) >= observed - 1e-14)),
        )
        self.assertGreater(np.sum(np.isclose(brute, observed)), 1)

    def test_loo_delta_matches_direct_recomputation(self) -> None:
        scores = np.array([[3, 2, 2, 1, 1, 0, 0, -1, -1, -2, -3]], dtype=float)
        observed = float(MODULE.loo_minimum_delta(scores)[0])
        direct = []
        for index in range(11):
            if index < 6:
                direct.append(np.delete(scores[0, :6], index).mean() - scores[0, 6:].mean())
            else:
                direct.append(scores[0, :6].mean() - np.delete(scores[0, 6:], index - 6).mean())
        self.assertAlmostEqual(observed, min(direct))

    def test_rule_grid_is_unique_and_complete(self) -> None:
        rules = MODULE.rule_grid()
        self.assertEqual(len(rules), 216)
        self.assertEqual(len({rule.rule_id for rule in rules}), len(rules))

    def test_small_simulation_is_deterministic(self) -> None:
        scenarios = MODULE.SCENARIOS[:2]
        rules = MODULE.rule_grid()[:4]
        first = MODULE.evaluate_grid(64, 32, 12345, scenarios, rules)
        second = MODULE.evaluate_grid(64, 32, 12345, scenarios, rules)
        self.assertEqual(first, second)

    def test_equal_selection_and_validation_seeds_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "must be independent"):
                MODULE.write_outputs(
                    Path(directory),
                    selection_draws=8,
                    validation_draws=8,
                    batch_size=4,
                    selection_seed=17,
                    validation_seed=17,
                    observed_on="2026-09-23",
                )

    def test_diagnostic_selector_is_deterministic_under_ties(self) -> None:
        rules = MODULE.rule_grid()[:3]
        template = {
            "eligible": False,
            "safety_eligible": True,
            "strong_pattern_support_rate": 0.1,
            "minimum_strong_robustness_support_rate": 0.08,
            "minimum_moderate_pattern_support_rate": 0.03,
            "max_one_arm_wilson_95_upper": 0.04,
            "max_safety_support_rate": 0.001,
        }
        diagnostics = {rule.rule_id: dict(template) for rule in rules}
        first = MODULE.choose_diagnostic_rule(rules, diagnostics)
        second = MODULE.choose_diagnostic_rule(rules, diagnostics)
        self.assertEqual(first.rule_id, second.rule_id)


@unittest.skipUnless((CALIBRATION / "calibration.json").exists(), "production packet not generated")
class CalibrationPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.packet = json.loads((CALIBRATION / "calibration.json").read_text())

    def test_packet_is_explicitly_outcome_blind_and_noncanonical(self) -> None:
        self.assertTrue(self.packet["outcome_blind"])
        self.assertFalse(self.packet["audit_outcome_values_accessed"])
        self.assertEqual(self.packet["empirical_outcome_files_read"], [])
        self.assertFalse(self.packet["canonical_state_changed"])

    def test_mouse_level_design_and_exact_space(self) -> None:
        design = self.packet["design"]
        self.assertEqual(design["inferential_unit"], "mouse")
        self.assertEqual(design["n_development_controls"], 9)
        self.assertEqual((design["n_stim_lick_minus"], design["n_stim_lick_plus"]), (6, 5))
        self.assertEqual(design["assignment_count"], 462)
        self.assertFalse(design["trial_level_pseudoreplication_allowed"])

    def test_no_binding_generic_rule_and_endpoint_calibration_required(self) -> None:
        self.assertIsNone(self.packet["selected_rule"])
        self.assertFalse(self.packet["selection"]["binding_rule_selected"])
        self.assertFalse(self.packet["terminal_guidance"]["positive_mechanistic_terminal_authorized"])
        self.assertEqual(
            self.packet["required_endpoint_specific_calibration"]["families"],
            ["binomial", "beta_binomial"],
        )

    def test_diagnostic_rule_uses_studentized_simultaneous_arm_guards(self) -> None:
        diagnostic = self.packet["diagnostic_rule"]
        self.assertIn("Bonferroni-Welch", diagnostic["arm_direction_method"])
        self.assertLessEqual(diagnostic["arm_direction_familywise_alpha"], 0.10)
        self.assertGreaterEqual(diagnostic["practical_margin_development_sd"], 0.50)

    def test_exact_test_scope_is_not_weak_null(self) -> None:
        family = self.packet["exact_randomization_family"]
        self.assertFalse(family["zero_test_is_exact_for_heterogeneous_weak_mean_null"])
        self.assertTrue(family["design_exact_requires_assignment_mechanism_receipt"])
        self.assertIn("sharp", family["practical_test_interpretation"])

    def test_independent_validation_and_extreme_one_arm_scenarios(self) -> None:
        simulation = self.packet["simulation"]
        self.assertTrue(simulation["selection_and_validation_seeds_independent"])
        self.assertIsInstance(simulation["selection_master_seed_hex"], str)
        self.assertTrue(simulation["selection_master_seed_hex"].startswith("0x"))
        scenarios = simulation["validation_scenarios"]
        self.assertIn("one_arm_minus_extreme_inactive_sd2", scenarios)
        self.assertIn("one_arm_plus_extreme_inactive_t3_sd2", scenarios)
        for item in scenarios.values():
            self.assertIsInstance(item["seed_hex"], str)

    def test_no_empirical_or_audit_path_is_embedded(self) -> None:
        text = (CALIBRATION / "calibration.json").read_text().lower()
        self.assertNotIn("seshmerge.mat", text)
        self.assertNotIn("role_vault", text)
        self.assertNotIn("primary_audit", text)

    def test_manifest(self) -> None:
        rows = (CALIBRATION / "SHA256SUMS").read_text().splitlines()
        self.assertEqual(len(rows), 4)
        for row in rows:
            expected, name = row.split("  ", 1)
            observed = hashlib.sha256((CALIBRATION / name).read_bytes()).hexdigest()
            self.assertEqual(observed, expected, name)


if __name__ == "__main__":
    unittest.main()
