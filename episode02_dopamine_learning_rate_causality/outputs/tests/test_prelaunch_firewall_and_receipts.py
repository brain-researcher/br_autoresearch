#!/usr/bin/env python3
"""Fail-closed tests for EP02 firewall and documentary receipts."""

from __future__ import annotations

import copy
import base64
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


EPISODE = Path(__file__).resolve().parents[2]
CODE = EPISODE / "outputs" / "code" / "verify_permission_firewall.py"
FIREWALL = EPISODE / "outputs" / "firewall"
RECEIPTS = EPISODE / "outputs" / "receipts"
SIGNOFF = EPISODE / "outputs" / "signoff" / "SCIENTIST_SIGNOFF_TEMPLATE.json"
PRELAUNCH = EPISODE / "outputs" / "prelaunch"
POLICY = EPISODE / "SEARCH_POLICY.json"
SPEC = importlib.util.spec_from_file_location("verify_permission_firewall", CODE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FirewallFunctionTests(unittest.TestCase):
    ASSESSMENT_TIME = datetime(2026, 9, 22, 22, 30, tzinfo=timezone.utc)

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_contract = json.loads((FIREWALL / "FIREWALL_CONTRACT.json").read_text())

    def pinned_contract(self) -> dict:
        contract = copy.deepcopy(self.base_contract)
        contract["current_environment_status"] = "PROVISIONED_EXTERNALLY_ATTESTED"
        contract["trusted_signers_status"] = "PINNED_AND_REVIEWED"
        contract["primary_audit_pack_root_sha256"] = "a" * 64
        contract["policy_sha256"] = "b" * 64
        contract["configuration_lock_sha256"] = "c" * 64
        contract["trusted_signers"] = {
            "infrastructure_operator": {
                "signer_id": "infra-independent",
                "ed25519_public_key_der_sha256": "1" * 64,
            },
            "data_steward": {
                "signer_id": "steward-independent",
                "ed25519_public_key_der_sha256": "2" * 64,
            },
        }
        return contract

    def valid_bundle(self, contract: dict) -> dict:
        payload = {
            "contract_sha256": MODULE.sha256_bytes(MODULE.canonical_json_bytes(contract)),
            "episode_id": contract["episode_id"],
            "principals": {
                "candidate": "candidate-principal",
                "evaluator": "evaluator-principal",
                "steward": "steward-principal",
            },
            "checks": {
                name: {"status": "PASS", "external_evidence_id": f"evidence:{name}"}
                for name in MODULE.REQUIRED_CHECKS
            },
            "source_sha256": contract["source_sha256"],
            "primary_audit_pack_root_sha256": contract["primary_audit_pack_root_sha256"],
            "policy_sha256": contract["policy_sha256"],
            "configuration_lock_sha256": contract["configuration_lock_sha256"],
            "candidate_runtime_id": "runtime-candidate-1",
            "evaluator_runtime_id": "runtime-evaluator-1",
            "observed_at_utc": "2026-09-22T22:00:00Z",
            "expires_at_utc": "2026-09-22T23:00:00Z",
        }
        return {
            "schema_version": "ep02.dudman.permission_firewall_receipt_bundle.v1",
            "status": "SIGNED",
            "receipt_payload": payload,
            "attestations": [
                {"role": "infrastructure_operator", "signer_id": "infra-independent"},
                {"role": "data_steward", "signer_id": "steward-independent"},
            ],
        }

    @staticmethod
    def accepts_test_signatures(openssl: str, payload: bytes, attestation: dict, expected: str) -> bool:
        return bool(openssl and payload and attestation and expected)

    def test_same_uid_contract_is_deliberately_not_passable(self) -> None:
        failures = MODULE.validate_contract(self.base_contract)
        self.assertIn("environment_not_provisioned", failures)
        self.assertIn("trusted_signers_not_pinned_and_reviewed", failures)
        self.assertIn("trusted_signer_not_pinned:data_steward", failures)
        self.assertIn("trusted_signer_not_pinned:infrastructure_operator", failures)
        self.assertFalse(self.base_contract["same_uid_controls_are_sufficient"])

    def test_semantically_complete_independently_signed_bundle_can_pass(self) -> None:
        contract = self.pinned_contract()
        assessment = MODULE.assess_signed_bundle(
            contract,
            self.valid_bundle(contract),
            "test-openssl",
            signature_checker=self.accepts_test_signatures,
            assessment_time=self.ASSESSMENT_TIME,
        )
        self.assertEqual(assessment["status"], "PASS")
        self.assertTrue(assessment["launch_authorized"])
        self.assertEqual(assessment["outcome_files_read"], [])

    def test_pending_contract_status_cannot_pass(self) -> None:
        contract = self.pinned_contract()
        contract["current_environment_status"] = "BLOCKED_PENDING_EXTERNAL_PROVISIONING"
        assessment = MODULE.assess_signed_bundle(
            contract,
            self.valid_bundle(contract),
            "test-openssl",
            signature_checker=self.accepts_test_signatures,
            assessment_time=self.ASSESSMENT_TIME,
        )
        self.assertEqual(assessment["status"], "BLOCKED")
        self.assertIn("environment_not_provisioned", assessment["failures"])

    def test_contract_and_runtime_commitment_mismatches_fail(self) -> None:
        for field in (
            "episode_id",
            "source_sha256",
            "primary_audit_pack_root_sha256",
            "policy_sha256",
            "configuration_lock_sha256",
        ):
            with self.subTest(field=field):
                contract = self.pinned_contract()
                bundle = self.valid_bundle(contract)
                bundle["receipt_payload"][field] = "wrong" if field == "episode_id" else "d" * 64
                assessment = MODULE.assess_signed_bundle(
                    contract,
                    bundle,
                    "test-openssl",
                    signature_checker=self.accepts_test_signatures,
                    assessment_time=self.ASSESSMENT_TIME,
                )
                self.assertIn(f"{field}_mismatch", assessment["failures"])

        contract = self.pinned_contract()
        bundle = self.valid_bundle(contract)
        bundle["receipt_payload"]["contract_sha256"] = "0" * 64
        assessment = MODULE.assess_signed_bundle(
            contract,
            bundle,
            "test-openssl",
            signature_checker=self.accepts_test_signatures,
            assessment_time=self.ASSESSMENT_TIME,
        )
        self.assertIn("contract_hash_mismatch", assessment["failures"])

    def test_malformed_container_and_timing_types_return_blocked(self) -> None:
        contract = self.pinned_contract()
        contract["receipt_validity_seconds_max"] = "86400"
        contract["clock_skew_seconds_max"] = []
        contract["trusted_signers"] = []
        bundle = self.valid_bundle(contract)
        bundle["receipt_payload"]["principals"] = []
        bundle["receipt_payload"]["checks"] = []
        assessment = MODULE.assess_signed_bundle(
            contract,
            bundle,
            "test-openssl",
            signature_checker=self.accepts_test_signatures,
            assessment_time=self.ASSESSMENT_TIME,
        )
        self.assertEqual(assessment["status"], "BLOCKED")
        for failure in (
            "receipt_validity_seconds_max_invalid",
            "clock_skew_seconds_max_invalid",
            "trusted_signers_not_object",
            "principals_not_object",
            "checks_not_object",
        ):
            self.assertIn(failure, assessment["failures"])

        assessment = MODULE.assess_signed_bundle(
            [],
            [],
            "test-openssl",
            signature_checker=self.accepts_test_signatures,
            assessment_time=self.ASSESSMENT_TIME,
        )
        self.assertEqual(assessment["status"], "BLOCKED")
        self.assertIn("contract_not_object", assessment["failures"])
        self.assertIn("receipt_bundle_not_object", assessment["failures"])

    def test_evidence_and_runtime_ids_must_be_nonempty_strings(self) -> None:
        contract = self.pinned_contract()
        bundle = self.valid_bundle(contract)
        bundle["receipt_payload"]["candidate_runtime_id"] = 7
        bundle["receipt_payload"]["evaluator_runtime_id"] = []
        bundle["receipt_payload"]["checks"]["atomic_no_partial_output"][
            "external_evidence_id"
        ] = 99
        assessment = MODULE.assess_signed_bundle(
            contract,
            bundle,
            "test-openssl",
            signature_checker=self.accepts_test_signatures,
            assessment_time=self.ASSESSMENT_TIME,
        )
        self.assertIn("candidate_runtime_id_missing", assessment["failures"])
        self.assertIn("evaluator_runtime_id_missing", assessment["failures"])
        self.assertIn("external_evidence_missing:atomic_no_partial_output", assessment["failures"])

    def test_same_principal_fails_even_with_signatures(self) -> None:
        contract = self.pinned_contract()
        bundle = self.valid_bundle(contract)
        bundle["receipt_payload"]["principals"]["evaluator"] = "candidate-principal"
        assessment = MODULE.assess_signed_bundle(
            contract,
            bundle,
            "test-openssl",
            signature_checker=self.accepts_test_signatures,
            assessment_time=self.ASSESSMENT_TIME,
        )
        self.assertEqual(assessment["status"], "BLOCKED")
        self.assertIn("principals_not_distinct", assessment["failures"])

    def test_missing_network_denial_fails(self) -> None:
        contract = self.pinned_contract()
        bundle = self.valid_bundle(contract)
        bundle["receipt_payload"]["checks"]["candidate_network_egress_denial"]["status"] = "BLOCKED"
        assessment = MODULE.assess_signed_bundle(
            contract,
            bundle,
            "test-openssl",
            signature_checker=self.accepts_test_signatures,
            assessment_time=self.ASSESSMENT_TIME,
        )
        self.assertIn(
            "check_not_passed:candidate_network_egress_denial", assessment["failures"]
        )

    def test_signer_id_and_key_must_both_be_distinct(self) -> None:
        contract = self.pinned_contract()
        contract["trusted_signers"]["data_steward"] = copy.deepcopy(
            contract["trusted_signers"]["infrastructure_operator"]
        )
        failures = MODULE.validate_contract(contract)
        self.assertIn("trusted_signer_ids_not_distinct", failures)
        self.assertIn("trusted_signer_keys_not_distinct", failures)

    def test_invalid_or_stale_time_binding_fails(self) -> None:
        contract = self.pinned_contract()
        invalid = self.valid_bundle(contract)
        invalid["receipt_payload"]["observed_at_utc"] = "not-a-time"
        assessment = MODULE.assess_signed_bundle(
            contract,
            invalid,
            "test-openssl",
            signature_checker=self.accepts_test_signatures,
            assessment_time=self.ASSESSMENT_TIME,
        )
        self.assertIn("observation_time_invalid", assessment["failures"])

        stale = self.valid_bundle(contract)
        assessment = MODULE.assess_signed_bundle(
            contract,
            stale,
            "test-openssl",
            signature_checker=self.accepts_test_signatures,
            assessment_time=datetime(2026, 9, 23, 22, 30, tzinfo=timezone.utc),
        )
        self.assertIn("receipt_expired", assessment["failures"])

    @unittest.skipUnless(shutil.which("openssl"), "OpenSSL not available")
    def test_real_ed25519_verification_and_tamper_rejection(self) -> None:
        openssl = shutil.which("openssl")
        assert openssl is not None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / "private.pem"
            public = root / "public.pem"
            payload_path = root / "payload.json"
            signature = root / "signature.bin"
            subprocess.run(
                [openssl, "genpkey", "-algorithm", "ED25519", "-out", str(private)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            subprocess.run(
                [openssl, "pkey", "-in", str(private), "-pubout", "-out", str(public)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            payload = MODULE.canonical_json_bytes({"receipt": "synthetic"})
            payload_path.write_bytes(payload)
            subprocess.run(
                [
                    openssl,
                    "pkeyutl",
                    "-sign",
                    "-rawin",
                    "-inkey",
                    str(private),
                    "-in",
                    str(payload_path),
                    "-out",
                    str(signature),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            expected = MODULE._public_key_der_sha256(openssl, public.read_bytes())
            assert expected is not None
            attestation = {
                "public_key_pem": public.read_text(),
                "signature_base64": base64.b64encode(signature.read_bytes()).decode("ascii"),
            }
            self.assertTrue(
                MODULE.verify_ed25519_attestation(openssl, payload, attestation, expected)
            )
            self.assertFalse(
                MODULE.verify_ed25519_attestation(
                    openssl, payload + b"tampered", attestation, expected
                )
            )

    @unittest.skipUnless(shutil.which("openssl"), "OpenSSL not available")
    def test_rsa_public_key_is_rejected_even_if_openssl_parses_it(self) -> None:
        openssl = shutil.which("openssl")
        assert openssl is not None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / "rsa-private.pem"
            public = root / "rsa-public.pem"
            subprocess.run(
                [openssl, "genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048", "-out", str(private)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            subprocess.run(
                [openssl, "pkey", "-in", str(private), "-pubout", "-out", str(public)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertIsNone(MODULE._public_key_der_sha256(openssl, public.read_bytes()))
            attestation = {
                "public_key_pem": public.read_text(),
                "signature_base64": base64.b64encode(b"not-an-ed25519-signature").decode("ascii"),
            }
            self.assertFalse(
                MODULE.verify_ed25519_attestation(openssl, b"payload", attestation, "0" * 64)
            )


@unittest.skipUnless(
    (FIREWALL / "current_firewall_assessment.json").exists()
    and RECEIPTS.exists()
    and SIGNOFF.exists()
    and (PRELAUNCH / "PRELAUNCH_EVIDENCE_MANIFEST.json").exists(),
    "runtime prelaunch packet is not present in the tracked specification",
)
class CurrentPacketTests(unittest.TestCase):
    def test_current_assessment_is_blocked_without_content_read(self) -> None:
        assessment = json.loads((FIREWALL / "current_firewall_assessment.json").read_text())
        self.assertEqual(assessment["status"], "BLOCKED")
        self.assertFalse(assessment["launch_authorized"])
        self.assertFalse(assessment["source_probe"]["content_read"])
        self.assertTrue(assessment["source_probe"]["same_uid"])
        self.assertTrue(assessment["source_probe"]["candidate_readable_by_access_probe"])
        self.assertFalse(assessment["same_uid_mode_bits_treated_as_firewall"])

    def test_endpoint_receipts_do_not_overclaim(self) -> None:
        expected = {
            "trialID_codebook_receipt.json": "UNRESOLVED",
            "seshID_training_day_semantics_receipt.json": "CONDITIONAL",
            "lickState_750ms_trigger_semantics_receipt.json": "CONDITIONAL",
            "raw_701_sample_axis_and_event_alignment_receipt.json": "UNRESOLVED",
            "randomization_mechanism_receipt.json": "BLOCKED_DOCUMENTED_SMALL_COHORT_RANDOMIZATION_AND_POSTCOLLECTION_EXCLUSIONS_ROSTER_MISSING",
        }
        for name, status in expected.items():
            value = json.loads((RECEIPTS / name).read_text())
            self.assertEqual(value["status"], status, name)
            self.assertFalse(value["launch_gate_satisfied"], name)
            self.assertFalse(value["values_or_effects_read"], name)

    def test_randomization_receipt_forbids_design_exact_label(self) -> None:
        value = json.loads((RECEIPTS / "randomization_mechanism_receipt.json").read_text())
        self.assertFalse(value["design_exact_462_allocation_claim_allowed"])
        self.assertEqual(value["allocation_count_if_conditionally_exchangeable"], 462)

    def test_signoff_is_unsigned(self) -> None:
        value = json.loads(SIGNOFF.read_text())
        self.assertEqual(value["status"], "UNSIGNABLE_NO_ELIGIBLE_RULE")
        self.assertIsNone(value["scientist_signature"])
        self.assertFalse(value["current_calibration_has_signable_rule"])
        self.assertTrue(value["new_calibration_required_before_signature"])
        self.assertTrue(value["unsigned_template_cannot_authorize_launch"])

    def test_packet_manifests(self) -> None:
        for directory in (FIREWALL, RECEIPTS, SIGNOFF.parent, PRELAUNCH):
            for row in (directory / "SHA256SUMS").read_text().splitlines():
                expected, name = row.split("  ", 1)
                observed = hashlib.sha256((directory / name).read_bytes()).hexdigest()
                self.assertEqual(observed, expected, str(directory / name))

    def test_prelaunch_manifest_is_nonbinding_and_outcome_blind(self) -> None:
        manifest_path = PRELAUNCH / "PRELAUNCH_EVIDENCE_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        self.assertFalse(manifest["launch_authorized"])
        self.assertFalse(manifest["audit_opening_authorized"])
        self.assertFalse(manifest["positive_mechanistic_terminal_authorized"])
        self.assertEqual(manifest["empirical_outcome_files_read"], [])
        self.assertFalse(manifest["audit_outcome_values_accessed"])
        firewall = manifest["permission_firewall"]
        self.assertEqual(firewall["status"], "BLOCKED")
        self.assertFalse(firewall["local_probe_read_source_content"])
        randomization = manifest["endpoint_and_design_receipts"]["randomization"]
        self.assertFalse(randomization["all_462_labels_are_documented_randomization_space"])
        self.assertFalse(randomization["all_462_label_result_is_intention_to_treat"])
        calibration = manifest["provisional_endpoint_shaped_calibration"]
        self.assertEqual(calibration["candidate_rule_count"], 384)
        self.assertEqual(calibration["selection_eligible_rule_count"], 0)
        self.assertEqual(calibration["independent_validation_eligible_rule_count"], 0)
        self.assertIsNone(calibration["binding_rule"])
        self.assertFalse(calibration["endpoint_faithful_claim_allowed"])
        self.assertFalse(calibration["audit_opening_authorized"])
        self.assertEqual(manifest["scientist_signoff"]["status"], "UNSIGNABLE_NO_ELIGIBLE_RULE")

    def test_policy_binds_prelaunch_packet_only_as_nonbinding_evidence(self) -> None:
        policy = json.loads(POLICY.read_text())
        manifest_path = PRELAUNCH / "PRELAUNCH_EVIDENCE_MANIFEST.json"
        observed = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        pin = policy["source_pins"]["prelaunch_evidence_packet"]
        self.assertEqual(pin["manifest_sha256"], observed)
        self.assertFalse(pin["binding_authority"])
        self.assertFalse(pin["launch_authorized"])
        nonbinding = policy["canonical_binding"]["nonbinding_prelaunch_artifact_hashes"]
        self.assertEqual(nonbinding["prelaunch_evidence_manifest"], observed)
        required = policy["canonical_binding"]["required_manifest_hashes"]
        self.assertIsNone(required["endpoint_faithful_small_n_calibration"])
        self.assertIsNone(required["scientist_calibration_signoff"])


if __name__ == "__main__":
    unittest.main()
