#!/usr/bin/env python3
"""Mechanical tests for the generated Dudman Phase-0 packet."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


EPISODE = Path(__file__).resolve().parents[2]
PHASE0 = Path(os.environ.get("EP02_PHASE0_PACKET_DIR", EPISODE / "outputs" / "phase0"))
CODE = EPISODE / "outputs" / "code" / "phase0_inventory.py"
SPEC = importlib.util.spec_from_file_location("phase0_inventory", CODE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Phase0ArgumentTests(unittest.TestCase):
    def test_observed_on_is_required(self) -> None:
        argv = [
            "phase0_inventory.py",
            "--mat",
            "source.mat",
            "--cohort-map",
            "cohort.json",
            "--output-dir",
            "output",
            "--expected-bytes",
            "1",
            "--expected-md5",
            "0" * 32,
            "--expected-sha256",
            "0" * 64,
        ]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            MODULE.parse_args()


@unittest.skipUnless(
    (PHASE0 / "source_integrity.json").exists()
    and (PHASE0 / "structural_inventory.json").exists()
    and (PHASE0 / "role_feasibility.json").exists()
    and (PHASE0 / "access_receipt.json").exists(),
    "runtime Phase-0 packet is not present in the tracked specification",
)
class Phase0PacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.integrity = json.loads((PHASE0 / "source_integrity.json").read_text())
        cls.inventory = json.loads((PHASE0 / "structural_inventory.json").read_text())
        cls.roles = json.loads((PHASE0 / "role_feasibility.json").read_text())
        cls.receipt = json.loads((PHASE0 / "access_receipt.json").read_text())

    def test_source_pins(self) -> None:
        self.assertTrue(self.integrity["bytes_match"])
        self.assertTrue(self.integrity["md5_matches"])
        self.assertTrue(self.integrity["sha256_matches"])
        self.assertTrue(
            self.integrity["mat_container"]["single_compressed_payload_reaches_eof"]
        )

    def test_record_and_cohort_counts(self) -> None:
        self.assertEqual(self.inventory["schema_version"], "ep02.dudman.phase0_inventory.v2")
        self.assertEqual(self.inventory["record_count"], 24)
        self.assertEqual(len(self.inventory["development_records"]), 9)
        self.assertFalse(self.inventory["audit_individual_record_details_emitted"])
        self.assertNotIn("records", self.inventory)
        self.assertEqual(
            sum(item["record_count"] for item in self.inventory["audit_role_summary"].values()),
            15,
        )
        self.assertTrue(
            all(
                not item["individual_record_details_emitted"]
                for item in self.inventory["audit_role_summary"].values()
            )
        )
        counts = {name: item["mice"] for name, item in self.roles["cohorts"].items()}
        self.assertEqual(
            counts,
            {
                "development_control": 9,
                "audit_calibrated_stim_lick_minus": 6,
                "audit_calibrated_stim_lick_plus": 5,
                "boundary_supraphysiological_stim_lick_plus": 4,
            },
        )

    def test_mouse_is_inferential_unit(self) -> None:
        self.assertEqual(self.roles["primary_inferential_unit"], "mouse")
        self.assertFalse(self.roles["trial_level_pseudoreplication_allowed"])
        self.assertEqual(self.roles["primary_audit_assignment_count"], 462)

    def test_access_receipt_contains_no_claimed_outcome_emission(self) -> None:
        for key in (
            "candidate_discriminating_values_emitted",
            "behavioral_or_latency_magnitudes_emitted",
            "photometry_magnitudes_emitted",
            "perturbation_effects_emitted",
            "model_scores_emitted",
            "intervention_individual_outcome_magnitudes_inspected",
        ):
            self.assertFalse(self.receipt[key], key)
        self.assertFalse(
            self.receipt["intervention_behavioral_state_codes_used_for_contingency_qc"]
        )
        prior = self.receipt["superseded_v1_prelaunch_exposure"]
        self.assertFalse(prior["candidate_search_or_controller_started"])
        self.assertTrue(prior["values_must_be_treated_as_exposed_regardless_of_whether_viewed"])
        self.assertIn("permanently_excluded", prior["scientific_use"])

    def test_no_per_record_audit_treatment_or_contingency_fields(self) -> None:
        serialized_inventory = json.dumps(self.inventory, sort_keys=True)
        self.assertNotIn("stimulated_trial_count", serialized_inventory)
        self.assertNotIn("declared_contingency_structurally_consistent", serialized_inventory)
        self.assertEqual(
            self.roles["gate"]["checks"][4]["check_id"],
            "audit_record_nondisclosure",
        )

    def test_gate_is_nonlaunching(self) -> None:
        self.assertEqual(self.roles["gate"]["verdict"], "PASS_WITH_CONDITIONS")
        self.assertFalse(self.roles["gate"]["launch_authorized"])
        self.assertFalse(self.roles["gate"]["canonical_state_changed"])

    def test_sha256_manifest(self) -> None:
        rows = (PHASE0 / "SHA256SUMS").read_text().splitlines()
        self.assertGreaterEqual(len(rows), 6)
        for row in rows:
            expected, name = row.split("  ", 1)
            path = PHASE0 / name
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(observed, expected, name)


if __name__ == "__main__":
    unittest.main()
