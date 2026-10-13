from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from ep12_executor import scientific_transition_binding as binding
from ep12_executor import post36_unattended_controller as unattended
from ep12_executor import replacement_attestation_terminal as replacement_terminal
from ep12_executor.qualification_launch import _TRANSITION_IDENTITY_FIELDS
from ep12_executor.replacement_attestation_launch import (
    DYNAMIC_TRANSITION_ARCHIVE_FILES,
    REQUIRED_ARCHIVE_FILES,
)


def _bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _dynamic_authority(root: Path) -> dict:
    archive = {
        "archived_root": "/synthetic/archive",
        "archive_parent": "/synthetic",
        "archive_receipt_path": "/synthetic/archive/receipt.json",
        "archive_receipt_file_sha256": "11" * 32,
        "archive_receipt_sha256": "12" * 32,
        "manifest_path": "/synthetic/archive/manifest.json",
        "manifest_file_sha256": "13" * 32,
        "manifest_sha256": "14" * 32,
        "tree_sha256": "15" * 32,
        "surface_allowlist": ["outputs/executor"],
        "inputs_included": False,
        "development_outputs_included": False,
        "prelaunch_outputs_included": False,
        "bootstrap_sha256": "16" * 32,
        "parent_first_publication_bound": True,
        "complete": True,
    }
    final = {
        "status": "absent_at_publication",
        "path": str(root / "outputs/prelaunch/post36_orchestration/receipt.json"),
        "file_sha256": None,
        "receipt_sha256": None,
        "required_before_irreversible_transition": True,
    }
    core = {
        "schema_version": binding.ATTESTATION_AUTHORITY_SCHEMA,
        "authoritative_job_id": "987654",
        "authoritative_job_name": "ep12-observed-attestation-replacement",
        "replacement_terminal_identity_path": str(root / "outputs/prelaunch/replacement_attestation_launch/terminal_identity.json"),
        "replacement_terminal_identity_file_sha256": "21" * 32,
        "replacement_terminal_identity_sha256": "22" * 32,
        "replacement_launch_receipt_file_sha256": "23" * 32,
        "replacement_launch_receipt_sha256": "24" * 32,
        "replacement_release_receipt_file_sha256": "25" * 32,
        "replacement_release_receipt_sha256": "26" * 32,
        "post36_orchestration_intent_path": str(root / "outputs/prelaunch/post36_orchestration/intent.json"),
        "post36_orchestration_intent_file_sha256": "27" * 32,
        "post36_orchestration_intent_sha256": "28" * 32,
        "post36_start_authorization_path": str(root / "outputs/prelaunch/post36_orchestration/start_authorization.json"),
        "post36_start_authorization_file_sha256": "29" * 32,
        "post36_start_authorization_sha256": "2a" * 32,
        "obsolete_cancellation_receipt_path": str(root / "outputs/prelaunch/replacement_attestation_launch/obsolete_cancellation/receipt.json"),
        "obsolete_cancellation_receipt_file_sha256": "2b" * 32,
        "obsolete_cancellation_receipt_sha256": "2c" * 32,
        "obsolete_job_id": "45332084",
        "obsolete_job_name": "ep12-attest36",
        "obsolete_terminal_scheduler_state": "CANCELLED by 12345",
        "obsolete_identity_file_sha256": "2d" * 32,
        "obsolete_identity_sha256": "2e" * 32,
        "obsolete_result_authoritative": False,
        "authoritative_result": True,
        "source_archive": archive,
        "controller_final_receipt": final,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return {**core, "authority_sha256": binding._canonical_digest(core)}


class DynamicTransitionArchiveClosureTests(unittest.TestCase):
    def test_dynamic_runtime_config_and_tests_are_required(self) -> None:
        self.assertTrue(DYNAMIC_TRANSITION_ARCHIVE_FILES <= REQUIRED_ARCHIVE_FILES)
        self.assertNotIn(
            "outputs/executor/ep12_executor/readiness_join_launch.py",
            REQUIRED_ARCHIVE_FILES,
        )

    def test_observed_interface_and_failed_attempt_recovery_are_required(self) -> None:
        self.assertTrue(
            {
                "outputs/executor/ep12_executor/adaptive_search.py",
                "outputs/executor/ep12_executor/adaptive_successors.py",
                "outputs/executor/ep12_executor/attest_observed_trials.py",
                "outputs/executor/ep12_executor/post36_failed_attempt_recovery.py",
                "outputs/executor/tests/test_post36_failed_attempt_recovery.py",
                "outputs/executor/tests/test_production_readiness.py",
            }
            <= REQUIRED_ARCHIVE_FILES
        )


class DynamicAttestationAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.terminal_path = (
            self.root
            / "outputs/prelaunch/replacement_attestation_launch/terminal_identity.json"
        )
        self.cancellation_path = (
            self.root
            / "outputs/prelaunch/replacement_attestation_launch/obsolete_cancellation/receipt.json"
        )
        self.start_path = (
            self.root / "outputs/prelaunch/post36_orchestration/start_authorization.json"
        )
        self.intent_path = (
            self.root / "outputs/prelaunch/post36_orchestration/intent.json"
        )
        self.final_path = (
            self.root / "outputs/prelaunch/post36_orchestration/receipt.json"
        )
        self.final_path.parent.mkdir(parents=True)
        self.archive = {
            "archived_root": "/synthetic/archive",
            "archive_parent": "/synthetic",
            "archive_receipt_path": "/synthetic/archive/receipt.json",
            "archive_receipt_file_sha256": "11" * 32,
            "archive_receipt_sha256": "12" * 32,
            "manifest_path": "/synthetic/archive/manifest.json",
            "manifest_file_sha256": "13" * 32,
            "manifest_sha256": "14" * 32,
            "tree_sha256": "15" * 32,
            "surface_allowlist": ["outputs/executor"],
            "inputs_included": False,
            "development_outputs_included": False,
            "prelaunch_outputs_included": False,
            "bootstrap_sha256": "16" * 32,
            "parent_first_publication_bound": True,
            "complete": True,
        }
        self.spec = {
            "required_observed_attestation_authority": {
                "schema_version": "ep12.dynamic_observed_attestation_authority_requirement.v1",
                "terminal_identity_relative_path": str(
                    self.terminal_path.relative_to(self.root)
                ),
                "cancellation_receipt_relative_path": str(
                    self.cancellation_path.relative_to(self.root)
                ),
                "start_authorization_relative_path": str(
                    self.start_path.relative_to(self.root)
                ),
                "controller_receipt_relative_path": str(
                    self.final_path.relative_to(self.root)
                ),
                "controller_receipt_policy": "bind_if_present_require_before_irreversible_transition",
                "dynamic_replacement_required": True,
                "source_archive_required": True,
                "obsolete_job_id": "45332084",
                "obsolete_result_authoritative": False,
            }
        }
        self.terminal = {
            "terminal_identity_path": str(self.terminal_path),
            "terminal_identity_file_sha256": "21" * 32,
            "terminal_identity_sha256": "22" * 32,
            "slurm_job_id": "987654",
            "slurm_job_name": "ep12-observed-attestation-replacement",
            "launch_receipt_file_sha256": "23" * 32,
            "launch_receipt_sha256": "24" * 32,
            "release_receipt_file_sha256": "25" * 32,
            "release_receipt_sha256": "26" * 32,
            "archive": self.archive,
            "attestation_manifest_file_sha256": "31" * 32,
            "attestation_artifact_file_sha256": "32" * 32,
            "attestation_sha256": "33" * 32,
            "superseded_result_authoritative": False,
            "attestation_result_authoritative": True,
        }
        self.cancellation = {
            "cancellation_receipt_path": str(self.cancellation_path),
            "cancellation_receipt_file_sha256": "2b" * 32,
            "cancellation_receipt_sha256": "2c" * 32,
            "obsolete_job_id": "45332084",
            "obsolete_job_name": "ep12-attest36",
            "replacement_job_id": self.terminal["slurm_job_id"],
            "terminal_scheduler_state": "CANCELLED by 12345",
            "superseded_result_authoritative": False,
        }
        self.start = {
            "intent_path": str(self.intent_path),
            "intent_file_sha256": "27" * 32,
            "intent_sha256": "28" * 32,
            "authorization_path": str(self.start_path),
            "authorization_file_sha256": "29" * 32,
            "start_authorization_sha256": "2a" * 32,
            "replacement": {
                "slurm_job_id": self.terminal["slurm_job_id"],
                "archive": self.archive,
            },
            "obsolete_cancellation": self.cancellation,
        }
        self.superseded = {
            "superseded_job_id": "45332084",
            "superseded_identity_file_sha256": "2d" * 32,
            "superseded_identity_sha256": "2e" * 32,
            "superseded_result_authoritative": False,
        }
        self.final = {
            "intent_path": str(self.intent_path),
            "intent_file_sha256": self.start["intent_file_sha256"],
            "intent_sha256": self.start["intent_sha256"],
            "receipt_path": str(self.final_path),
            "receipt_file_sha256": "41" * 32,
            "receipt_sha256": "42" * 32,
            "replacement": {"slurm_job_id": self.terminal["slurm_job_id"]},
            "obsolete_cancellation": {
                "cancellation_receipt_sha256": self.cancellation[
                    "cancellation_receipt_sha256"
                ]
            },
            "start_authorization": {
                "start_authorization_sha256": self.start[
                    "start_authorization_sha256"
                ]
            },
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _verify(
        self,
        *,
        final_side_effect: Exception | None = None,
        start_path: Path | None = None,
        cancellation_path: Path | None = None,
    ) -> dict:
        with mock.patch.object(
            replacement_terminal,
            "verify_replacement_attestation_terminal_identity",
            return_value=self.terminal,
        ), mock.patch.object(
            unattended,
            "verify_post36_start_authorization",
            return_value=self.start,
        ), mock.patch.object(
            unattended,
            "verify_post36_orchestration_receipt",
            return_value=self.final,
            side_effect=final_side_effect,
        ), mock.patch.object(
            binding,
            "_verify_superseded_attestation_job_identity",
            return_value=self.superseded,
        ):
            return binding._verify_dynamic_attestation_authority(
                episode_root=self.root,
                transition_spec=self.spec,
                terminal_identity_path=self.terminal_path,
                start_authorization_path=start_path or self.start_path,
                cancellation_receipt_path=(
                    cancellation_path or self.cancellation_path
                ),
                attestation_manifest_file_sha256="31" * 32,
                attestation_artifact_file_sha256="32" * 32,
                attestation_sha256="33" * 32,
            )

    def test_absent_final_receipt_succeeds_with_deferred_requirement(self) -> None:
        authority = self._verify()
        self.assertEqual(
            authority["controller_final_receipt"]["status"],
            "absent_at_publication",
        )
        self.assertTrue(
            authority["controller_final_receipt"][
                "required_before_irreversible_transition"
            ]
        )
        self.assertTrue(authority["authoritative_result"])
        self.assertFalse(authority["obsolete_result_authoritative"])

    def test_noncanonical_top_start_authorization_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            binding.ScientificTransitionBindingError,
            "start authorization path is not canonical",
        ):
            self._verify(start_path=self.root / "wrong-start.json")

    def test_noncanonical_exact_cancellation_receipt_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            binding.ScientificTransitionBindingError,
            "cancellation receipt path is not canonical",
        ):
            self._verify(cancellation_path=self.root / "wrong-cancellation.json")

    def test_present_valid_final_receipt_is_bound(self) -> None:
        self.final_path.write_bytes(b"synthetic present receipt\n")
        authority = self._verify()
        self.assertEqual(
            authority["controller_final_receipt"],
            {
                "status": "present_verified",
                "path": str(self.final_path),
                "file_sha256": "41" * 32,
                "receipt_sha256": "42" * 32,
                "required_before_irreversible_transition": True,
            },
        )

    def test_present_tampered_final_receipt_fails_closed(self) -> None:
        self.final_path.write_bytes(b"synthetic tampered receipt\n")
        with self.assertRaisesRegex(
            binding.ScientificTransitionBindingError,
            "present post36 controller receipt is invalid",
        ):
            self._verify(final_side_effect=ValueError("synthetic tampering"))


class ScientificTransitionBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.executor = self.root / "outputs/executor"
        self.executor.mkdir(parents=True)
        self.transition_root = self.root / binding.TRANSITION_ROOT_RELATIVE
        self.transition_root.mkdir(parents=True)
        self.attestation_dir = self.root / binding.OBSERVED_ATTESTATION_RELATIVE
        self.attestation_dir.mkdir(parents=True)
        self.finalization_dir = self.root / binding.OBSERVED_FINALIZATION_RELATIVE
        self.finalization_dir.mkdir(parents=True)
        self.run_root = self.root / "outputs/development_run002"

        self.old_model_sha = "1" * 64
        model_bytes = b"# active scientific model\n"
        self.active_model_sha = _sha(model_bytes)
        model_path = self.executor / "ep12_executor/scientific_models.py"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(model_bytes)
        self.active_contract_sha = "2" * 64
        self.active_spec_sha = "3" * 64
        self.transition_spec_sha = "4" * 64
        (self.executor / "CONTRACT_TRANSITION_PROBE_SPEC.json").write_text(
            "{}\n", encoding="utf-8"
        )
        (self.executor / "CONTRACT_TRANSITION_SPEC.json").write_bytes(
            _bytes(
                {
                    "implementation_lineage": {
                        "scientific_models": {
                            "path": "ep12_executor/scientific_models.py",
                            "old_sha256": self.old_model_sha,
                            "new_sha256": self.active_model_sha,
                        }
                    }
                }
            )
        )

        self.attestation_semantic_sha = "5" * 64
        self.attestation_rows: list[dict] = []
        self.declaration_rows: list[dict] = []
        for index in range(36):
            trial_id = f"trial_{index:02d}"
            relative = f"synthetic_trials/{trial_id}"
            report = {
                "schema_version": "ep12.development_trial.v1",
                "trial_id": trial_id,
                "status": "completed",
                "scientific_contract_sha256": binding.OBSERVED_PREFIX_CONTRACT_SHA256,
                "data_role": "development_only",
                "final_connectivity_accessed": False,
                "synthetic_objective_never_emitted": index / 10.0,
            }
            report_bytes = _bytes(report)
            report_path = self.run_root / relative / "trial_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_bytes(report_bytes)
            report_sha = _sha(report_bytes)
            row = {
                "trial_id": trial_id,
                "source_trial_directory": relative,
                "source_files_sha256": {"trial_report.json": report_sha},
            }
            self.attestation_rows.append(row)
            self.declaration_rows.append(
                {
                    "trial_id": trial_id,
                    "source_report_schema": report["schema_version"],
                    "source_report_status": report["status"],
                    "source_trial_report_sha256": report_sha,
                    "declared_scientific_contract_sha256": report[
                        "scientific_contract_sha256"
                    ],
                    "data_role": "development_only",
                    "final_connectivity_accessed": False,
                }
            )
        attestation = {
            "attestation_sha256": self.attestation_semantic_sha,
            "trials": self.attestation_rows,
        }
        attestation_bytes = _bytes(attestation)
        (self.attestation_dir / binding.OBSERVED_ATTESTATION_NAME).write_bytes(
            attestation_bytes
        )
        attestation_manifest = {
            "schema_version": "ep12.observed_trial_attestation_set_manifest.v1",
            "files": {
                binding.OBSERVED_ATTESTATION_NAME: _sha(attestation_bytes)
            },
            "attestation_sha256": self.attestation_semantic_sha,
            "trial_count": 36,
            "source_trial_files_modified": False,
            "objective_values_emitted": False,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }
        attestation_manifest_bytes = _bytes(attestation_manifest)
        (self.attestation_dir / "manifest.json").write_bytes(
            attestation_manifest_bytes
        )
        self.attestation_manifest_sha = _sha(attestation_manifest_bytes)
        self.attestation_artifact_sha = _sha(attestation_bytes)

        # Invalid JSON is intentional: the verifier must hash but never decode it.
        archive_bytes = b"opaque synthetic archive; not JSON\n"
        (self.finalization_dir / binding.OBSERVED_PREFIX_ARCHIVE_NAME).write_bytes(
            archive_bytes
        )
        limitation_manifest = {
            "schema_version": "ep12.observed_minimum_finalization_manifest.v1",
            "files": {binding.OBSERVED_PREFIX_ARCHIVE_NAME: _sha(archive_bytes)},
            "final_connectivity_accessed": False,
        }
        limitation_bytes = _bytes(limitation_manifest)
        (self.finalization_dir / "manifest.json").write_bytes(limitation_bytes)
        self.archive_sha = _sha(archive_bytes)
        self.limitation_sha = _sha(limitation_bytes)

        self.probe_records: dict[str, dict] = {}
        for phase, payload_sha in (("legacy", "6" * 64), ("post36", "6" * 64)):
            record = {
                "phase": phase,
                "artifact_sha256": ("7" if phase == "legacy" else "8") * 64,
                "scientific_payload_sha256": payload_sha,
            }
            path = self.root / f"outputs/prelaunch/{phase}_probe.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_bytes(record))
            self.probe_records[phase] = {"path": path, "record": record}
        additions = ["diagnostics.representation_rank"]
        comparison = {
            "schema_version": "ep12.contract_transition_invariance_comparison.v1",
            "artifact_sha256": "9" * 64,
            "legacy": {
                "path": str(self.probe_records["legacy"]["path"]),
                "file_sha256": _sha(
                    self.probe_records["legacy"]["path"].read_bytes()
                ),
                "artifact_sha256": self.probe_records["legacy"]["record"][
                    "artifact_sha256"
                ],
            },
            "post36": {
                "path": str(self.probe_records["post36"]["path"]),
                "file_sha256": _sha(
                    self.probe_records["post36"]["path"].read_bytes()
                ),
                "artifact_sha256": self.probe_records["post36"]["record"][
                    "artifact_sha256"
                ],
            },
            "checks": {
                "input_payload_byte_identical": True,
                "environment_byte_identical": True,
                "scientific_payload_byte_identical": True,
                "diagnostic_changes_limited_to_added_representation_rank": True,
                "diagnostic_additions": additions,
            },
        }
        self.comparison = comparison
        comparison_bytes = _bytes(comparison)
        (self.transition_root / binding.SYNTHETIC_INVARIANCE_NAME).write_bytes(
            comparison_bytes
        )

        self.declaration_sha = binding._canonical_digest(self.declaration_rows)
        declarations = {
            "schema_version": binding.REPORT_DECLARATIONS_SCHEMA,
            "transition_id": binding.TRANSITION_ID,
            "observed_attestation_manifest_sha256": self.attestation_manifest_sha,
            "observed_attestation_sha256": self.attestation_semantic_sha,
            "trial_count": 36,
            "rows": self.declaration_rows,
            "objective_values_read": False,
            "objective_values_compared": False,
            "objective_values_emitted": False,
            "final_connectivity_accessed": False,
            "declarations_sha256": self.declaration_sha,
        }
        declaration_bytes = _bytes(declarations)
        (self.transition_root / binding.REPORT_DECLARATIONS_NAME).write_bytes(
            declaration_bytes
        )

        self.receipt_semantic_sha = "a" * 64
        self.attestation_authority = _dynamic_authority(self.root)
        self.exposure_evidence = {
            "operator_exposure_incident_file_sha256": "e" * 64,
            "operator_exposure_incident_semantic_sha256": "f" * 64,
            "operator_development_outcome_values_displayed": True,
            "programmatic_transition_extraction_outcome_fields_accessed": False,
            "values_used_for_transition_decisions": False,
            "final_connectivity_accessed": False,
        }
        receipt = {
            "transition_receipt_sha256": self.receipt_semantic_sha,
            "old_contract_sha256": binding.OBSERVED_PREFIX_CONTRACT_SHA256,
            "active_contract_sha256": self.active_contract_sha,
            "old_qualification_spec_sha256": binding.OBSERVED_PREFIX_QUALIFICATION_SPEC_SHA256,
            "active_qualification_spec_sha256": self.active_spec_sha,
            "old_scientific_models_sha256": self.old_model_sha,
            "active_scientific_models_sha256": self.active_model_sha,
            "observed_attestation_manifest_sha256": self.attestation_manifest_sha,
            "observed_attestation_sha256": self.attestation_semantic_sha,
            "observed_attestation_authority": self.attestation_authority,
            "observed_report_contract_declarations_sha256": self.declaration_sha,
            "synthetic_invariance_evidence_sha256": comparison["artifact_sha256"],
            "operator_exposure_incident_file_sha256": self.exposure_evidence[
                "operator_exposure_incident_file_sha256"
            ],
            "operator_exposure_incident_semantic_sha256": self.exposure_evidence[
                "operator_exposure_incident_semantic_sha256"
            ],
        }
        receipt_bytes = _bytes(receipt)
        (self.transition_root / binding.TRANSITION_RECEIPT_NAME).write_bytes(
            receipt_bytes
        )

        probe_evidence = {
            "legacy_probe_file_sha256": _sha(
                self.probe_records["legacy"]["path"].read_bytes()
            ),
            "legacy_probe_artifact_sha256": self.probe_records["legacy"][
                "record"
            ]["artifact_sha256"],
            "legacy_probe_scientific_payload_sha256": "6" * 64,
            "post36_probe_file_sha256": _sha(
                self.probe_records["post36"]["path"].read_bytes()
            ),
            "post36_probe_artifact_sha256": self.probe_records["post36"][
                "record"
            ]["artifact_sha256"],
            "post36_probe_scientific_payload_sha256": "6" * 64,
            "allowed_diagnostics_delta_sha256": binding._canonical_digest(additions),
        }
        manifest = {
            "schema_version": binding.TRANSITION_MANIFEST_SCHEMA,
            "files": {
                binding.TRANSITION_RECEIPT_NAME: _sha(receipt_bytes),
                binding.REPORT_DECLARATIONS_NAME: _sha(declaration_bytes),
                binding.SYNTHETIC_INVARIANCE_NAME: _sha(comparison_bytes),
            },
            "transition_id": binding.TRANSITION_ID,
            "transition_receipt_sha256": self.receipt_semantic_sha,
            "transition_spec_sha256": self.transition_spec_sha,
            "observed_prefix_contract_sha256": binding.OBSERVED_PREFIX_CONTRACT_SHA256,
            "active_contract_sha256": self.active_contract_sha,
            "observed_prefix_qualification_spec_sha256": binding.OBSERVED_PREFIX_QUALIFICATION_SPEC_SHA256,
            "active_qualification_spec_sha256": self.active_spec_sha,
            "observed_prefix_scientific_models_sha256": self.old_model_sha,
            "active_scientific_models_sha256": self.active_model_sha,
            "observed_attestation_manifest_file_sha256": self.attestation_manifest_sha,
            "observed_attestation_artifact_file_sha256": self.attestation_artifact_sha,
            "observed_attestation_sha256": self.attestation_semantic_sha,
            "observed_attestation_authority_sha256": self.attestation_authority[
                "authority_sha256"
            ],
            "authoritative_attestation_job_id": self.attestation_authority[
                "authoritative_job_id"
            ],
            "obsolete_attestation_job_id": self.attestation_authority[
                "obsolete_job_id"
            ],
            "source_archive_receipt_file_sha256": self.attestation_authority[
                "source_archive"
            ]["archive_receipt_file_sha256"],
            "source_archive_receipt_sha256": self.attestation_authority[
                "source_archive"
            ]["archive_receipt_sha256"],
            "source_archive_manifest_file_sha256": self.attestation_authority[
                "source_archive"
            ]["manifest_file_sha256"],
            "source_archive_manifest_sha256": self.attestation_authority[
                "source_archive"
            ]["manifest_sha256"],
            "source_archive_tree_sha256": self.attestation_authority[
                "source_archive"
            ]["tree_sha256"],
            "controller_final_receipt_status": self.attestation_authority[
                "controller_final_receipt"
            ]["status"],
            "observed_trial_count": 36,
            "observed_report_contract_declarations_sha256": self.declaration_sha,
            "observed_prefix_archive_file_sha256": self.archive_sha,
            "observed_prefix_limitation_manifest_file_sha256": self.limitation_sha,
            **probe_evidence,
            "operator_exposure_incident_file_sha256": self.exposure_evidence[
                "operator_exposure_incident_file_sha256"
            ],
            "operator_exposure_incident_semantic_sha256": self.exposure_evidence[
                "operator_exposure_incident_semantic_sha256"
            ],
            "source_trial_reports_reverified": True,
            "operator_development_outcome_values_displayed": True,
            "programmatic_transition_extraction_outcome_fields_accessed": False,
            "values_used_for_transition_decisions": False,
            "final_connectivity_accessed": False,
            "complete": True,
        }
        self.manifest_path = self.transition_root / binding.TRANSITION_MANIFEST_NAME
        self.manifest_path.write_bytes(_bytes(manifest))
        self.manifest_sha = _sha(self.manifest_path.read_bytes())

        self.fake_transition = types.SimpleNamespace(
            verify_transition_receipt=lambda receipt: None,
            verify_contract_transition_definition=lambda **kwargs: {
                "transition_spec_sha256": self.transition_spec_sha,
                "old_contract_sha256": binding.OBSERVED_PREFIX_CONTRACT_SHA256,
                "new_contract_sha256": self.active_contract_sha,
                "old_qualification_spec_sha256": binding.OBSERVED_PREFIX_QUALIFICATION_SPEC_SHA256,
                "new_qualification_spec_sha256": self.active_spec_sha,
            },
            _verify_attestation=lambda attestation: {
                row["trial_id"]: row for row in self.attestation_rows
            },
            _verify_report_declarations=self._verify_declarations,
            extract_authenticated_report_contract_declarations=self._extract,
        )
        self.fake_probe = types.SimpleNamespace(verify_artifact=self._verify_probe)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _verify_declarations(self, rows: list[dict], attested: dict) -> list[dict]:
        self.assertEqual(len(rows), 36)
        for row in rows:
            self.assertEqual(
                row["source_trial_report_sha256"],
                attested[row["trial_id"]]["source_files_sha256"]["trial_report.json"],
            )
            if row["declared_scientific_contract_sha256"] != binding.OBSERVED_PREFIX_CONTRACT_SHA256:
                raise ValueError("wrong observed-prefix contract")
        return sorted(rows, key=lambda row: row["trial_id"])

    def _extract(self, *, attested_by_id: dict, authenticated_trial_report_bytes: dict) -> list[dict]:
        rows = []
        for trial_id in sorted(attested_by_id):
            raw = authenticated_trial_report_bytes[trial_id]
            self.assertEqual(
                _sha(raw),
                attested_by_id[trial_id]["source_files_sha256"]["trial_report.json"],
            )
            report = json.loads(raw)
            rows.append(
                {
                    "trial_id": trial_id,
                    "source_report_schema": report["schema_version"],
                    "source_report_status": report["status"],
                    "source_trial_report_sha256": _sha(raw),
                    "declared_scientific_contract_sha256": report[
                        "scientific_contract_sha256"
                    ],
                    "data_role": report["data_role"],
                    "final_connectivity_accessed": report[
                        "final_connectivity_accessed"
                    ],
                }
            )
        return self._verify_declarations(rows, attested_by_id)

    def _verify_probe(self, *, artifact_path: Path, spec_path: Path) -> dict:
        self.assertEqual(spec_path, self.executor / "CONTRACT_TRANSITION_PROBE_SPEC.json")
        return json.loads(artifact_path.read_text())

    def _verify(self) -> dict:
        with mock.patch.object(binding, "_transition_module", return_value=self.fake_transition), mock.patch.object(
            binding, "_probe_module", return_value=self.fake_probe
        ), mock.patch.object(
            binding,
            "_verify_dynamic_attestation_authority",
            return_value=self.attestation_authority,
        ), mock.patch.object(
            binding,
            "_verify_operator_exposure_incident",
            return_value=self.exposure_evidence,
        ):
            return binding.verify_scientific_contract_transition(
                episode_root=self.root,
                executor_root=self.executor,
                manifest_path=self.manifest_path,
                expected_manifest_file_sha256=self.manifest_sha,
                expected_transition_receipt_sha256=self.receipt_semantic_sha,
            )

    def test_exact_bundle_passes_and_opaque_archive_is_not_decoded(self) -> None:
        evidence = self._verify()
        self.assertEqual(set(evidence), _TRANSITION_IDENTITY_FIELDS)
        self.assertTrue(
            evidence["exact_36_attestation_set_matches_transition_evidence"]
        )
        self.assertEqual(evidence["observed_trial_count"], 36)
        self.assertEqual(
            evidence["observed_prefix_contract_sha256"],
            binding.OBSERVED_PREFIX_CONTRACT_SHA256,
        )
        self.assertEqual(evidence["active_contract_sha256"], self.active_contract_sha)
        self.assertTrue(evidence["operator_development_outcome_values_displayed"])
        self.assertFalse(
            evidence["programmatic_transition_extraction_outcome_fields_accessed"]
        )
        self.assertFalse(evidence["values_used_for_transition_decisions"])

    def test_one_report_byte_mutation_fails_closed(self) -> None:
        report = self.run_root / "synthetic_trials/trial_00/trial_report.json"
        report.write_bytes(report.read_bytes() + b" ")
        with self.assertRaises(AssertionError):
            self._verify()

    def test_manifest_file_hash_and_exact_36_are_independent_gates(self) -> None:
        with mock.patch.object(binding, "_transition_module", return_value=self.fake_transition), mock.patch.object(
            binding, "_probe_module", return_value=self.fake_probe
        ), self.assertRaisesRegex(
            binding.ScientificTransitionBindingError, "manifest file hash mismatch"
        ):
            binding.verify_scientific_contract_transition(
                episode_root=self.root,
                executor_root=self.executor,
                manifest_path=self.manifest_path,
                expected_manifest_file_sha256="0" * 64,
                expected_transition_receipt_sha256=self.receipt_semantic_sha,
            )


if __name__ == "__main__":
    unittest.main()
