#!/usr/bin/env python3
"""Mechanical tests for EP02's bounded endpoint calibration."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


EPISODE = Path(__file__).resolve().parents[2]
CALIBRATION = Path(
    os.environ.get(
        "EP02_ENDPOINT_CALIBRATION_DIR", EPISODE / "outputs" / "endpoint_calibration"
    )
)
MODULE_PATH = EPISODE / "outputs" / "code" / "calibrate_endpoint_small_n.py"
SPEC = importlib.util.spec_from_file_location("calibrate_endpoint_small_n", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EndpointCalibrationFunctionTests(unittest.TestCase):
    def test_assignment_space_is_complete(self) -> None:
        weights, indicators = MODULE.assignment_weights()
        self.assertEqual(weights.shape, (math.comb(11, 6), 11))
        self.assertEqual(indicators.shape, weights.shape)
        np.testing.assert_allclose(weights.sum(axis=1), 0.0, atol=1e-12)
        np.testing.assert_array_equal(indicators[0], [1] * 6 + [0] * 5)

    def test_rule_grid_is_unique_and_complete(self) -> None:
        rules = MODULE.rule_grid()
        self.assertEqual(len(rules), 384)
        self.assertEqual(len({rule.rule_id for rule in rules}), 384)

    def test_binomial_and_beta_binomial_scores_are_bounded(self) -> None:
        for scenario in (MODULE.SCENARIOS[0], MODULE.SCENARIOS[1]):
            values = MODULE.simulate_scenario(np.random.default_rng(17), scenario, 200)
            development, audit, structural = values
            self.assertTrue(np.all(development >= -1.0))
            self.assertTrue(np.all(development <= 1.0))
            self.assertTrue(np.all(audit >= -1.0))
            self.assertTrue(np.all(audit <= 1.0))
            self.assertTrue(np.all(structural))

    def test_beta_binomial_is_more_overdispersed_than_binomial(self) -> None:
        binomial = MODULE.SCENARIOS[0]
        beta = MODULE.SCENARIOS[2]
        development_bin, _, _ = MODULE.simulate_scenario(
            np.random.default_rng(123), binomial, 5000
        )
        development_beta, _, _ = MODULE.simulate_scenario(
            np.random.default_rng(123), beta, 5000
        )
        self.assertGreater(development_beta.var(), development_bin.var() * 2.0)

    def test_perfect_oriented_separation_has_minimum_exact_p(self) -> None:
        weights, _ = MODULE.assignment_weights()
        development = np.linspace(0.30, 0.70, 9)[None, :]
        audit = np.array([[0.90] * 6 + [0.10] * 5])
        metrics = MODULE.scenario_metrics(
            development, audit, np.array([True]), weights, (0.15,)
        )
        self.assertAlmostEqual(float(metrics["p_zero"][0]), 1.0 / 462.0)
        self.assertAlmostEqual(float(metrics["p_margin_0.15"][0]), 1.0 / 462.0)

    def test_shifted_p_matches_independent_brute_force(self) -> None:
        weights, _ = MODULE.assignment_weights()
        development = np.linspace(0.2, 0.8, 9)[None, :]
        audit = np.array([[0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.6, 0.4, 0.2, 0.1, 0.0]])
        margin = 0.15
        metrics = MODULE.scenario_metrics(
            development, audit, np.array([True]), weights, (margin,)
        )
        shifted = audit[0].copy()
        shifted[:6] -= margin
        observed = shifted[:6].mean() - shifted[6:].mean()
        brute = []
        all_indices = set(range(11))
        for chosen in itertools.combinations(range(11), 6):
            other = sorted(all_indices - set(chosen))
            brute.append(shifted[list(chosen)].mean() - shifted[other].mean())
        expected = np.mean(np.asarray(brute) >= observed - 1e-14)
        self.assertAlmostEqual(float(metrics["p_margin_0.15"][0]), float(expected))

    def test_ties_are_included(self) -> None:
        weights, _ = MODULE.assignment_weights()
        development = np.linspace(0.2, 0.8, 9)[None, :]
        audit = np.array([[0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.2]])
        metrics = MODULE.scenario_metrics(
            development, audit, np.array([True]), weights, (0.10,)
        )
        observed = audit[0, :6].mean() - audit[0, 6:].mean()
        all_statistics = audit @ weights.T
        expected = np.mean(all_statistics[0] >= observed - 1e-14)
        self.assertAlmostEqual(float(metrics["p_zero"][0]), float(expected))
        self.assertGreater(np.sum(np.isclose(all_statistics[0], observed)), 1)

    def test_structural_failure_forces_abstention(self) -> None:
        weights, _ = MODULE.assignment_weights()
        development = np.linspace(0.2, 0.8, 9)[None, :]
        audit = np.array([[0.9] * 6 + [0.1] * 5])
        metrics = MODULE.scenario_metrics(
            development, audit, np.array([False]), weights, (0.10, 0.15, 0.20, 0.25)
        )
        for rule in MODULE.rule_grid():
            self.assertFalse(bool(MODULE.passes_rule(metrics, rule)[0]))

    def test_prerequisite_inventory_is_always_nonbinding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = MODULE.receipt_preflight(root / "receipts", root / "signoff.json")
        self.assertEqual(inventory["status"], "NONBINDING_PREREQUISITE_INVENTORY_ONLY")
        self.assertFalse(inventory["binding_eligible"])
        self.assertFalse(inventory["cryptographic_authority_verified"])
        self.assertIn("scientist_signoff_missing", inventory["failures"])
        self.assertIn(
            "endpoint_receipt_authorities_not_cryptographically_verified",
            inventory["failures"],
        )
        self.assertTrue(any(item.startswith("missing_receipt:") for item in inventory["failures"]))

    def test_declared_ready_fixture_still_lacks_cryptographic_authority(self) -> None:
        receipt_schemas = {
            "trialID_codebook_receipt.json": "ep02.dudman.endpoint_receipt.v1",
            "seshID_training_day_semantics_receipt.json": "ep02.dudman.endpoint_receipt.v1",
            "lickState_750ms_trigger_semantics_receipt.json": "ep02.dudman.endpoint_receipt.v1",
            "raw_701_sample_axis_and_event_alignment_receipt.json": "ep02.dudman.endpoint_receipt.v1",
            "randomization_mechanism_receipt.json": "ep02.dudman.randomization_receipt.v1",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipts = root / "receipts"
            receipts.mkdir()
            for name, schema in receipt_schemas.items():
                (receipts / name).write_text(
                    json.dumps(
                        {
                            "schema_version": schema,
                            "status": "RESOLVED",
                            "launch_gate_satisfied": True,
                        }
                    )
                )
            signoff = root / "signoff.json"
            signoff.write_text(
                json.dumps(
                    {
                        "schema_version": "ep02.dudman.scientist_signoff.v1",
                        "status": "SIGNED",
                        "scientist_signature": "signature-shaped-but-unverified",
                    }
                )
            )
            inventory = MODULE.receipt_preflight(receipts, signoff)
        self.assertTrue(inventory["all_declared_readiness_fields_satisfied"])
        self.assertFalse(inventory["binding_eligible"])
        self.assertFalse(inventory["cryptographic_authority_verified"])
        self.assertEqual(
            inventory["declared_readiness_failures"],
            [],
        )

    def test_small_simulation_is_deterministic(self) -> None:
        scenarios = MODULE.SCENARIOS[:2]
        rules = MODULE.rule_grid()[:3]
        first = MODULE.evaluate_grid(64, 32, 12345, scenarios, rules)
        second = MODULE.evaluate_grid(64, 32, 12345, scenarios, rules)
        self.assertEqual(first, second)

    def test_equal_selection_and_validation_seeds_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be independent"):
            MODULE.run(SimpleNamespace(selection_seed=17, validation_seed=17))

    def test_diagnostic_selector_is_deterministic(self) -> None:
        rules = MODULE.rule_grid()[:3]
        diagnostics = {
            rule.rule_id: {
                "safety_eligible": False,
                "minimum_strong_robustness_support_rate": 0.1,
                "strong_binomial_support_rate": 0.2,
                "moderate_binomial_support_rate": 0.1,
                "near_boundary_support_rate": 0.1,
                "max_one_arm_wilson_95_upper": 0.02,
                "max_safety_wilson_95_upper": 0.01,
            }
            for rule in rules
        }
        first = MODULE.choose_diagnostic_rule(rules, diagnostics)
        second = MODULE.choose_diagnostic_rule(rules, diagnostics)
        self.assertEqual(first.rule_id, second.rule_id)


@unittest.skipUnless(
    (CALIBRATION / "calibration.json").exists(),
    "runtime endpoint-calibration packet is not present",
)
class EndpointCalibrationPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.packet = json.loads((CALIBRATION / "calibration.json").read_text())

    def test_packet_is_outcome_blind_nonbinding_and_provisional(self) -> None:
        self.assertTrue(self.packet["outcome_blind"])
        self.assertEqual(self.packet["empirical_outcome_files_read"], [])
        self.assertFalse(self.packet["audit_outcome_values_accessed"])
        self.assertIsNone(self.packet["binding_rule"])
        self.assertFalse(self.packet["audit_opening_authorized"])
        inventory = self.packet["prerequisite_inventory"]
        self.assertEqual(inventory["status"], "NONBINDING_PREREQUISITE_INVENTORY_ONLY")
        self.assertFalse(inventory["binding_eligible"])
        self.assertFalse(inventory["cryptographic_authority_verified"])
        report = (CALIBRATION / "CALIBRATION.md").read_text()
        self.assertIn("Provisional endpoint-shaped", report)
        self.assertNotIn("# Endpoint-faithful", report)

    def test_packet_code_hash_and_manifest_replay(self) -> None:
        observed_code = hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest()
        self.assertEqual(self.packet["runtime"]["code_sha256"], observed_code)
        rows = (CALIBRATION / "SHA256SUMS").read_text().splitlines()
        self.assertEqual(len(rows), 4)
        for row in rows:
            expected, name = row.split("  ", 1)
            self.assertEqual(hashlib.sha256((CALIBRATION / name).read_bytes()).hexdigest(), expected)

if __name__ == "__main__":
    unittest.main()
