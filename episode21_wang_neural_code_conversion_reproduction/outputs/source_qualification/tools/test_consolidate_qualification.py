#!/usr/bin/env python3

import copy
import importlib.util
import json
import unittest
from pathlib import Path


TOOLS = Path(__file__).parent
EPISODE = TOOLS.parents[2]
RUNNER = TOOLS / "consolidate_qualification.py"
SPEC = importlib.util.spec_from_file_location("ep21_consolidate_qualification", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def public_model_payload_fixture(attempt: int) -> dict:
    assets = {}
    for asset_id, pin in MODULE.PUBLIC_MODEL_PINS.items():
        assets[asset_id] = {
            "pinned_identity": {
                "figshare_file_id": pin["figshare_file_id"],
                "filename": pin["filename"],
                "size_bytes": pin["size_bytes"],
                "md5": pin["md5"],
            },
            "cache": {
                "size_matches": True,
                "md5_matches": True,
                "cache_regular_file_verified": True,
                "cache_symlink_rejected": True,
                "cache_mode": "0440",
            },
            "archive_inventory": {
                "unsafe_member_names": [],
                "non_regular_member_names": [],
                "members": [
                    {
                        "path": pin["checkpoint_member"],
                        "member_type": "regular_file",
                        "sha256": "a" * 64,
                    }
                ],
            },
        }
    return {
        "schema_version": "ep21.public_model_payload_audit.v2",
        "astra_analysis_id": "source_qualification",
        "attempt": attempt,
        "outcome": "completed_public_reconstruction_assets_authenticated_evaluator_checkpoints_unbound",
        "empirical_or_decoded_payload_accessed": False,
        "ood_or_ep04_payload_accessed": False,
        "checkpoint_deserialization_performed": False,
        "payload_bytes_downloaded_this_attempt": 0,
        "assets": assets,
        "public_reconstruction_bindings": {
            "vgg19": {
                "checkpoint_member_verified": True,
                "checkpoint_member": assets["vgg19"]["archive_inventory"]["members"][0],
            },
            "caffenet": {
                "candidate_checkpoint_member_verified": True,
                "candidate_checkpoint_member": assets["caffenet"]["archive_inventory"]["members"][0],
            },
        },
        "evaluator_checkpoint_references": {
            "vgg19": {
                "lab_local_byte_identity_verified": False,
                "strict_bdpy_runtime_load_verified": False,
            },
            "alexnet": {
                "exact_public_bvlc_alexnet_named_member_verified": False,
                "caffenet_equivalence_claim_allowed": False,
            },
        },
    }


class QualificationConsolidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        verified_source = json.loads(
            (
                EPISODE
                / "outputs/source_qualification/attempt-002/verified_public_source_inventory.json"
            ).read_text()
        )
        verified_source["source_release"]["inventory_release_matches_pin"] = True
        cls.arguments = dict(
            attempt=5,
            structural=json.loads(
                (EPISODE / "inputs/conversion_structural_manifest.json").read_text()
            ),
            initial_qualification=json.loads(
                (
                    EPISODE
                    / "outputs/source_qualification/attempt-002/qualification_report.json"
                ).read_text()
            ),
            verified_source=verified_source,
            asset_dependencies=json.loads(
                (
                    EPISODE
                    / "outputs/source_qualification/attempt-004/public_asset_dependency_report.json"
                ).read_text()
            ),
            asset_metadata=json.loads(
                (
                    EPISODE
                    / "outputs/source_qualification/attempt-004/public_asset_metadata_report.json"
                ).read_text()
            ),
            runtime=json.loads(
                (
                    EPISODE
                    / "outputs/source_qualification/attempt-004/runtime_environment_report.json"
                ).read_text()
            ),
            unmanifested_assets=json.loads(
                (
                    EPISODE
                    / "outputs/source_qualification/attempt-006/unmanifested_source_asset_report.json"
                ).read_text()
            ),
            started_at="2026-09-29T00:00:00+00:00",
            process_id=1,
            manifest_metadata=json.loads(
                (
                    EPISODE
                    / "outputs/source_qualification/attempt-009/figshare_manifest_reconciliation.json"
                ).read_text()
            ),
        )
        cls.reports = MODULE.build_reports(**cls.arguments)

    def test_declared_outputs_share_attempt_and_analysis(self) -> None:
        for report in self.reports:
            self.assertEqual(report["attempt"], 5)
            self.assertEqual(report["astra_analysis_id"], "source_qualification")

    def test_structural_matrix_preserves_unknowns(self) -> None:
        matrix = self.reports[0]["structural_qualification_matrix"]
        required = {
            "release_package_identifier",
            "license_and_dua",
            "sherlock_risk_classification",
            "participant_ids",
            "session_run_trial_image_repetition_joins",
            "beta_variant",
            "array_shapes_dtypes_finiteness_missingness",
            "native_roi_and_mask_ids",
            "stimulus_roles_and_tau_partition",
            "exact_overlap_and_repetition_gates",
            "directed_laion_pair_table",
            "prior_exposure",
            "declared_ep21_input_boundary",
        }
        self.assertTrue(required.issubset(matrix))
        self.assertEqual(matrix["release_package_identifier"]["state"], "unknown")
        self.assertEqual(matrix["license_and_dua"]["state"], "unresolved")

    def test_source_and_asset_evidence_is_connected(self) -> None:
        report = self.reports[0]
        self.assertEqual(report["source_identity"]["state"], "verified")
        self.assertEqual(
            report["source_pipeline_assets"]["verified_public_model_asset_count"],
            4,
        )
        self.assertEqual(report["source_pipeline_assets"]["payload_bytes_downloaded"], 0)
        reconciliation = report["source_pipeline_assets"][
            "figshare_manifest_metadata_reconciliation"
        ]
        self.assertEqual(reconciliation["resolved_file_id_count"], 62)
        self.assertEqual(
            reconciliation["md5_reconciliation"][
                "manifest_vs_figshare_computed_match_count"
            ],
            62,
        )
        self.assertEqual(reconciliation["readiness_effect"], "none_laion_remains_blocked")
        self.assertEqual(
            report["source_pipeline_assets"]["unmanifested_or_unpinned_gaps"][
                "alexnet_identification_checkpoint"
            ]["state"],
            "required_unmanifested",
        )

    def test_empirical_boundaries_remain_closed(self) -> None:
        qualification, laion, external = self.reports
        self.assertFalse(qualification["empirical_or_ood_payload_accessed"])
        self.assertFalse(laion["empirical_data_accessed"])
        self.assertFalse(external["empirical_data_accessed"])
        self.assertEqual(laion["status"], "blocked_not_runnable")

    def test_consolidation_rejects_every_release_pin_substitution(self) -> None:
        for field, expected in MODULE.PINNED_SOURCE_RELEASE.items():
            arguments = copy.deepcopy(self.arguments)
            arguments["verified_source"]["source_release"][field] = (
                -1 if isinstance(expected, int) else f"wrong-{field}"
            )
            with self.subTest(field=field), self.assertRaises(ValueError):
                MODULE.build_reports(**arguments)

    def test_branch_safety_claims_are_derived_from_structural_state(self) -> None:
        arguments = copy.deepcopy(self.arguments)
        structural = arguments["structural"]
        structural["physical_exclusion"]["ep21_input_tree_contains_ood_payload"] = True
        structural["stable_interfaces"]["laion_regular_feature_handoff"] = "present"
        structural["stable_interfaces"]["laion_regular_scoring_handoff"] = "present"
        arguments["initial_qualification"]["declared_interface_state"][
            "laion_regular_feature_handoff"
        ] = "present"
        arguments["initial_qualification"]["declared_interface_state"][
            "laion_regular_scoring_handoff"
        ] = "present"
        structural["authorization"]["protocol_lock_present"] = True
        qualification, laion, _ = MODULE.build_reports(**arguments)
        self.assertTrue(qualification["protocol_lock_present"])
        self.assertFalse(
            laion["verified"]["declared_input_tree_excludes_ood_interface"]
        )
        self.assertFalse(laion["absent"]["laion_regular_feature_handoff"])
        self.assertFalse(laion["absent"]["laion_regular_scoring_handoff"])
        self.assertFalse(laion["absent"]["protocol_lock"])

    def test_manifest_and_direct_interface_state_must_agree(self) -> None:
        for location in ("structural", "initial_qualification"):
            arguments = copy.deepcopy(self.arguments)
            if location == "structural":
                arguments["structural"]["stable_interfaces"][
                    "laion_regular_feature_handoff"
                ] = "present"
            else:
                arguments["initial_qualification"]["declared_interface_state"][
                    "laion_regular_feature_handoff"
                ] = "present"
            with self.subTest(location=location), self.assertRaises(ValueError):
                MODULE.build_reports(**arguments)

    def test_public_payload_audit_is_narrowly_consolidated(self) -> None:
        arguments = copy.deepcopy(self.arguments)
        arguments["public_model_payload"] = public_model_payload_fixture(5)
        qualification, laion, _ = MODULE.build_reports(**arguments)
        assets = qualification["source_pipeline_assets"]
        self.assertEqual(assets["payload_verified_public_model_asset_count"], 2)
        self.assertEqual(assets["public_model_payload_bytes_inspected"], 760_409_145)
        gaps = assets["unmanifested_or_unpinned_gaps"]
        self.assertFalse(
            gaps["vgg19_identification_checkpoint_binding"][
                "evaluator_lab_local_byte_identity_verified"
            ]
        )
        self.assertFalse(
            gaps["alexnet_identification_checkpoint"][
                "caffenet_equivalence_claim_allowed"
            ]
        )
        self.assertNotIn(
            "bdpy_release_provenance", qualification["runtime_and_reconstruction"]
        )
        self.assertEqual(gaps["main_bdpy_dependency"]["state"], "declared_but_unpinned")
        self.assertEqual(
            gaps["main_bdpy_dependency"]["required_action"],
            "Authenticate the bdpy revision/API compatibility used by the tagged pipeline before the runtime environment can be locked.",
        )
        self.assertIn(
            "VGG19_evaluator_lab_copy_and_strict_runtime_load",
            laion["unknown_or_unresolved"],
        )
        self.assertIn("main_bdpy_revision", laion["unknown_or_unresolved"])
        self.assertNotIn(
            "main_bdpy_historical_installed_revision",
            laion["unknown_or_unresolved"],
        )

    def test_public_payload_audit_rejects_overclaim_and_attempt_mismatch(self) -> None:
        for mutation in ("attempt", "vgg_identity", "alexnet_equivalence"):
            arguments = copy.deepcopy(self.arguments)
            report = public_model_payload_fixture(5)
            if mutation == "attempt":
                report["attempt"] = 6
            elif mutation == "vgg_identity":
                report["evaluator_checkpoint_references"]["vgg19"][
                    "lab_local_byte_identity_verified"
                ] = True
            else:
                report["evaluator_checkpoint_references"]["alexnet"][
                    "caffenet_equivalence_claim_allowed"
                ] = True
            arguments["public_model_payload"] = report
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                MODULE.build_reports(**arguments)


if __name__ == "__main__":
    unittest.main()
