from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ep12_executor import scientific_transition_binding as binding
from ep12_executor import scientific_transition_publish as publisher


def _json_bytes(value: object) -> bytes:
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


class TransitionMetadataVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()
        self.source = self.outputs / "executor/ep12_executor/live_import.py"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"synthetic live import\n")

        incident_core = {
            "schema_version": binding.OPERATOR_EXPOSURE_SCHEMA,
            "episode_id": "ep12",
            "episode_root": str(self.root),
            "observed_at": "2026-09-26T06:12:01-07:00",
            "actor": "synthetic_operator",
            "access_class": "terminal_display",
            "intended_purpose": "synthetic operational diagnosis",
            "cause": "synthetic terminal output",
            "displayed_content_classes": ["synthetic development score"],
            "displayed_path_patterns": ["synthetic/trial_report.json"],
            "development_connectivity_outcome_values_displayed": True,
            "development_connectivity_outcome_values_intentionally_requested": False,
            "development_connectivity_outcome_values_analyzed": False,
            "development_connectivity_outcome_values_used_for_scientific_decisions": False,
            "scientific_contract_or_threshold_changed_after_display": False,
            "final_connectivity_accessed": False,
            "remediation": ["record exact incident"],
        }
        self.incident = {
            **incident_core,
            "incident_sha256": binding._canonical_digest(incident_core),
        }
        self.incident_path = self.outputs / "prelaunch/exposure_incident.json"
        self.incident_path.parent.mkdir(parents=True)
        self.incident_path.write_bytes(_json_bytes(self.incident))

        identity_core = {
            "schema_version": binding.ATTESTATION_JOB_IDENTITY_SCHEMA,
            "slurm_job_id": "12345",
            "slurm_job_name": "synthetic-attestation",
            "submission_user": "synthetic",
            "submitted_at": "2026-09-26T05:00:00-07:00",
            "identity_captured_at": "2026-09-26T05:01:00-07:00",
            "dependency": "afterok:12344",
            "partition": "normal",
            "cpus_per_task": 1,
            "memory_gib": 2,
            "time_limit": "00:20:00",
            "spooled_batch_script_sha256": "1" * 64,
            "current_on_disk_wrapper_sha256": "1" * 64,
            "spooled_script_matches_current_on_disk_wrapper": False,
            "spooled_script_uses_live_oak_python_imports": True,
            "pre_run_live_import_source_sha256": {
                str(self.source.relative_to(self.root)): _sha(self.source.read_bytes())
            },
            "identity_limitations": ["synthetic fixture"],
            "real_connectivity_outcome_values_read_to_build_identity": False,
            "final_connectivity_accessed": False,
        }
        identity_semantic_bytes = (
            json.dumps(
                identity_core,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        self.identity = {
            **identity_core,
            "identity_sha256": binding._file_sha(identity_semantic_bytes),
        }
        self.identity_path = self.outputs / "prelaunch/attestation_identity.json"
        self.identity_path.write_bytes(_json_bytes(self.identity))
        self.spec = {
            "outcome_firewall": {
                "operator_exposure_incident": {
                    "relative_path": str(self.incident_path.relative_to(self.root)),
                    "file_sha256": _sha(self.incident_path.read_bytes()),
                    "semantic_sha256": self.incident["incident_sha256"],
                    "schema_version": binding.OPERATOR_EXPOSURE_SCHEMA,
                },
                "operator_development_outcome_values_displayed": True,
                "programmatic_transition_extraction_outcome_fields_accessed": False,
                "values_used_for_transition_decisions": False,
                "final_connectivity_accessed": False,
            },
            "superseded_observed_attestation_job_identity": {
                "relative_path": str(self.identity_path.relative_to(self.root)),
                "file_sha256": _sha(self.identity_path.read_bytes()),
                "identity_sha256": self.identity["identity_sha256"],
                "schema_version": binding.ATTESTATION_JOB_IDENTITY_SCHEMA,
                "slurm_job_id": "12345",
                "authoritative": False,
                "disposition": "terminally_cancelled_superseded_non_authoritative",
            },
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_incident_and_superseded_identity_pass(self) -> None:
        exposure = binding._verify_operator_exposure_incident(
            episode_root=self.root, transition_spec=self.spec
        )
        identity = binding._verify_superseded_attestation_job_identity(
            episode_root=self.root, transition_spec=self.spec
        )
        self.assertTrue(exposure["operator_development_outcome_values_displayed"])
        self.assertFalse(
            exposure["programmatic_transition_extraction_outcome_fields_accessed"]
        )
        self.assertFalse(exposure["values_used_for_transition_decisions"])
        self.assertEqual(identity["superseded_job_id"], "12345")
        self.assertFalse(identity["superseded_result_authoritative"])

    def test_live_import_mutation_cannot_promote_superseded_identity(self) -> None:
        self.source.write_bytes(b"mutated after attestation\n")
        evidence = binding._verify_superseded_attestation_job_identity(
            episode_root=self.root, transition_spec=self.spec
        )
        self.assertFalse(evidence["superseded_result_authoritative"])

    def test_intermediate_symlink_cannot_redirect_evidence_read(self) -> None:
        redirected = self.outputs / "redirected"
        redirected.symlink_to(self.source.parent, target_is_directory=True)
        with self.assertRaisesRegex(
            binding.ScientificTransitionBindingError, "cannot safely open"
        ):
            binding._read_bytes(
                redirected / self.source.name,
                boundary=self.root,
                label="synthetic redirected evidence",
            )

    def test_analyzed_operator_incident_cannot_be_relabelled_as_safe(self) -> None:
        core = {
            key: value
            for key, value in self.incident.items()
            if key != "incident_sha256"
        }
        core["development_connectivity_outcome_values_analyzed"] = True
        unsafe = {**core, "incident_sha256": binding._canonical_digest(core)}
        self.incident_path.write_bytes(_json_bytes(unsafe))
        exposure = self.spec["outcome_firewall"]["operator_exposure_incident"]
        exposure["file_sha256"] = _sha(self.incident_path.read_bytes())
        exposure["semantic_sha256"] = unsafe["incident_sha256"]
        with self.assertRaisesRegex(
            binding.ScientificTransitionBindingError, "safety facts changed"
        ):
            binding._verify_operator_exposure_incident(
                episode_root=self.root, transition_spec=self.spec
            )

    def test_finalization_archive_is_hashed_without_json_decode(self) -> None:
        finalization = self.root / binding.OBSERVED_FINALIZATION_RELATIVE
        finalization.mkdir(parents=True)
        archive = b"opaque synthetic archive; deliberately not JSON\n"
        (finalization / binding.OBSERVED_PREFIX_ARCHIVE_NAME).write_bytes(archive)
        manifest = {
            "schema_version": "ep12.observed_minimum_finalization_manifest.v1",
            "files": {binding.OBSERVED_PREFIX_ARCHIVE_NAME: _sha(archive)},
            "final_connectivity_accessed": False,
        }
        manifest_bytes = _json_bytes(manifest)
        (finalization / "manifest.json").write_bytes(manifest_bytes)
        observed = binding._verify_observed_prefix_archive(
            episode_root=self.root, finalization_dir=finalization
        )
        self.assertEqual(observed, (_sha(archive), _sha(manifest_bytes)))


class TransitionExactOncePublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        (self.root / "outputs").mkdir()
        self.executor = self.root / "executor"
        self.executor.mkdir()
        self.payloads = {
            name: f"synthetic:{name}\n".encode("ascii")
            for name in publisher._BUNDLE_NAMES
        }
        self.summary = {
            "transition_receipt_sha256": "a" * 64,
            "observed_attestation_authority": _dynamic_authority(self.root),
            "operator_development_outcome_values_displayed": True,
            "programmatic_transition_extraction_outcome_fields_accessed": False,
            "values_used_for_transition_decisions": False,
            "final_connectivity_accessed": False,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _publish(self) -> dict:
        with mock.patch.object(
            publisher,
            "_collect_bundle",
            return_value=(dict(self.payloads), dict(self.summary)),
        ), mock.patch.object(
            binding,
            "verify_scientific_contract_transition",
            return_value={"synthetic_verification": True},
        ):
            return publisher.publish_scientific_transition(
                episode_root=self.root,
                executor_root=self.executor,
                synthetic_invariance_path=self.root / "synthetic.json",
                replacement_terminal_identity_path=self.root / "terminal.json",
                post36_start_authorization_path=self.root / "start.json",
                obsolete_cancellation_receipt_path=self.root / "cancel.json",
            )

    def test_exact_four_directory_publish_and_identical_retry(self) -> None:
        first = self._publish()
        second = self._publish()
        target = self.root / binding.TRANSITION_ROOT_RELATIVE
        self.assertEqual(first["publication_status"], "created")
        self.assertEqual(second["publication_status"], "reused_identical")
        self.assertEqual({path.name for path in target.iterdir()}, publisher._BUNDLE_NAMES)
        for name, expected in self.payloads.items():
            self.assertEqual((target / name).read_bytes(), expected)

    def test_existing_partial_tree_is_never_overwritten(self) -> None:
        target = self.root / binding.TRANSITION_ROOT_RELATIVE
        target.mkdir(parents=True)
        sentinel = target / binding.TRANSITION_RECEIPT_NAME
        sentinel.write_bytes(b"partial sentinel\n")
        with self.assertRaisesRegex(
            publisher.ScientificTransitionPublishError, "partial or has extra"
        ):
            publisher._publish_exact_once(
                episode_root=self.root, payloads=self.payloads
            )
        self.assertEqual(sentinel.read_bytes(), b"partial sentinel\n")
        self.assertEqual(list(target.iterdir()), [sentinel])

    def test_existing_complete_but_different_tree_is_never_overwritten(self) -> None:
        status = publisher._publish_exact_once(
            episode_root=self.root, payloads=self.payloads
        )
        self.assertEqual(status, "created")
        target = self.root / binding.TRANSITION_ROOT_RELATIVE
        before = {
            path.name: path.read_bytes() for path in target.iterdir()
        }
        changed = dict(self.payloads)
        changed[binding.TRANSITION_RECEIPT_NAME] = b"different retry\n"
        with self.assertRaisesRegex(
            publisher.ScientificTransitionPublishError,
            "differs from this exact retry",
        ):
            publisher._publish_exact_once(
                episode_root=self.root, payloads=changed
            )
        after = {path.name: path.read_bytes() for path in target.iterdir()}
        self.assertEqual(after, before)

    def test_input_change_between_prepublication_verifications_publishes_nothing(self) -> None:
        changed = dict(self.payloads)
        changed[binding.TRANSITION_RECEIPT_NAME] = b"changed\n"
        with mock.patch.object(
            publisher,
            "_collect_bundle",
            side_effect=[
                (dict(self.payloads), dict(self.summary)),
                (changed, dict(self.summary)),
            ],
        ):
            with self.assertRaisesRegex(
                publisher.ScientificTransitionPublishError,
                "changed between the two",
            ):
                publisher.publish_scientific_transition(
                    episode_root=self.root,
                    executor_root=self.executor,
                    synthetic_invariance_path=self.root / "synthetic.json",
                    replacement_terminal_identity_path=self.root / "terminal.json",
                    post36_start_authorization_path=self.root / "start.json",
                    obsolete_cancellation_receipt_path=self.root / "cancel.json",
                )
        self.assertFalse((self.root / binding.TRANSITION_ROOT_RELATIVE).exists())


class TransitionBundleAssemblyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.executor = self.root / "outputs/executor"
        self.executor.mkdir(parents=True)
        (self.executor / "CONTRACT_TRANSITION_SPEC.json").write_bytes(
            _json_bytes({"synthetic": "transition spec"})
        )
        (self.executor / "CONTRACT_TRANSITION_PROBE_SPEC.json").write_bytes(b"{}\n")

        run_root = self.root / "outputs/development_run002"
        self.attestation_rows = []
        for index in range(36):
            trial_id = f"synthetic_{index:02d}"
            relative = f"trials/{trial_id}"
            report = {
                "schema_version": "ep12.development_trial.v1",
                "trial_id": trial_id,
                "status": "completed",
                "scientific_contract_sha256": binding.OBSERVED_PREFIX_CONTRACT_SHA256,
                "data_role": "development_only",
                "final_connectivity_accessed": False,
                "synthetic_candidate_outcome_must_not_be_emitted": index + 0.25,
            }
            report_bytes = _json_bytes(report)
            report_path = run_root / relative / "trial_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_bytes(report_bytes)
            self.attestation_rows.append(
                {
                    "trial_id": trial_id,
                    "source_trial_directory": relative,
                    "source_files_sha256": {"trial_report.json": _sha(report_bytes)},
                }
            )
        attestation = {
            "attestation_sha256": "1" * 64,
            "trials": self.attestation_rows,
        }
        attestation_bytes = _json_bytes(attestation)
        attestation_dir = self.root / binding.OBSERVED_ATTESTATION_RELATIVE
        attestation_dir.mkdir(parents=True)
        (attestation_dir / binding.OBSERVED_ATTESTATION_NAME).write_bytes(
            attestation_bytes
        )
        attestation_manifest = {
            "schema_version": "ep12.observed_trial_attestation_set_manifest.v1",
            "files": {binding.OBSERVED_ATTESTATION_NAME: _sha(attestation_bytes)},
            "attestation_sha256": "1" * 64,
            "trial_count": 36,
            "source_trial_files_modified": False,
            "objective_values_emitted": False,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }
        (attestation_dir / "manifest.json").write_bytes(
            _json_bytes(attestation_manifest)
        )

        finalization = self.root / binding.OBSERVED_FINALIZATION_RELATIVE
        finalization.mkdir(parents=True)
        archive = b"opaque outcome-bearing bytes; this is intentionally invalid JSON\n"
        (finalization / binding.OBSERVED_PREFIX_ARCHIVE_NAME).write_bytes(archive)
        (finalization / "manifest.json").write_bytes(
            _json_bytes(
                {
                    "schema_version": "ep12.observed_minimum_finalization_manifest.v1",
                    "files": {binding.OBSERVED_PREFIX_ARCHIVE_NAME: _sha(archive)},
                    "final_connectivity_accessed": False,
                }
            )
        )
        self.comparison_path = self.root / "outputs/prelaunch/synthetic_comparison.json"
        self.comparison_path.parent.mkdir(parents=True)
        self.comparison = {
            "schema_version": "ep12.contract_transition_invariance_comparison.v1",
            "artifact_sha256": "2" * 64,
        }
        self.comparison_bytes = _json_bytes(self.comparison)
        self.comparison_path.write_bytes(self.comparison_bytes)
        self.attestation_authority = _dynamic_authority(self.root)
        self.exposure_evidence = {
            "operator_exposure_incident_file_sha256": "6" * 64,
            "operator_exposure_incident_semantic_sha256": "7" * 64,
            "operator_development_outcome_values_displayed": True,
            "programmatic_transition_extraction_outcome_fields_accessed": False,
            "values_used_for_transition_decisions": False,
            "final_connectivity_accessed": False,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _extract(self, *, attested_by_id: dict, authenticated_trial_report_bytes: dict) -> list[dict]:
        rows = []
        for trial_id in sorted(attested_by_id):
            raw = authenticated_trial_report_bytes[trial_id]
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
        return rows

    def _receipt(self, **kwargs: object) -> dict:
        declarations = self._extract(
            attested_by_id={row["trial_id"]: row for row in self.attestation_rows},
            authenticated_trial_report_bytes=kwargs["authenticated_trial_report_bytes"],
        )
        return {
            "transition_receipt_sha256": "8" * 64,
            "old_scientific_models_sha256": "9" * 64,
            "active_scientific_models_sha256": "a" * 64,
            "observed_attestation_authority": self.attestation_authority,
            "operator_exposure_incident_file_sha256": self.exposure_evidence[
                "operator_exposure_incident_file_sha256"
            ],
            "operator_exposure_incident_semantic_sha256": self.exposure_evidence[
                "operator_exposure_incident_semantic_sha256"
            ],
            "observed_report_contract_declarations_sha256": binding._canonical_digest(
                declarations
            ),
            "operator_development_outcome_values_displayed": True,
            "programmatic_transition_extraction_outcome_fields_accessed": False,
            "values_used_for_transition_decisions": False,
            "final_connectivity_accessed": False,
        }

    def test_bundle_assembly_preserves_probe_bytes_and_emits_only_declarations(self) -> None:
        definition = {
            "transition_spec_sha256": "b" * 64,
            "old_contract_sha256": binding.OBSERVED_PREFIX_CONTRACT_SHA256,
            "new_contract_sha256": "c" * 64,
            "old_qualification_spec_sha256": binding.OBSERVED_PREFIX_QUALIFICATION_SPEC_SHA256,
            "new_qualification_spec_sha256": "d" * 64,
        }
        probe_evidence = {
            "legacy_probe_file_sha256": "e" * 64,
            "legacy_probe_artifact_sha256": "f" * 64,
            "legacy_probe_scientific_payload_sha256": "0" * 64,
            "post36_probe_file_sha256": "1" * 64,
            "post36_probe_artifact_sha256": "2" * 64,
            "post36_probe_scientific_payload_sha256": "0" * 64,
            "allowed_diagnostics_delta_sha256": "3" * 64,
        }
        with mock.patch.object(
            publisher.scientific_contract_transition,
            "verify_contract_transition_definition",
            return_value=definition,
        ), mock.patch.object(
            publisher.scientific_contract_transition,
            "_verify_attestation",
            return_value={row["trial_id"]: row for row in self.attestation_rows},
        ), mock.patch.object(
            publisher.scientific_contract_transition,
            "extract_authenticated_report_contract_declarations",
            side_effect=self._extract,
        ), mock.patch.object(
            publisher.scientific_contract_transition,
            "verify_synthetic_invariance_evidence",
            return_value=self.comparison["artifact_sha256"],
        ), mock.patch.object(
            publisher.scientific_contract_transition,
            "build_transition_receipt",
            side_effect=self._receipt,
        ), mock.patch.object(
            publisher.scientific_contract_transition,
            "verify_transition_receipt",
            return_value=None,
        ), mock.patch.object(
            publisher.contract_invariance_probe,
            "verify_artifact",
            return_value=self.comparison,
        ), mock.patch.object(
            binding,
            "_verify_dynamic_attestation_authority",
            return_value=self.attestation_authority,
        ), mock.patch.object(
            binding,
            "_verify_operator_exposure_incident",
            return_value=self.exposure_evidence,
        ), mock.patch.object(
            binding,
            "_verify_probe_pair",
            return_value=probe_evidence,
        ):
            payloads, summary = publisher._collect_bundle(
                episode_root=self.root,
                executor_root=self.executor,
                synthetic_invariance_path=self.comparison_path,
                replacement_terminal_identity_path=Path(
                    self.attestation_authority["replacement_terminal_identity_path"]
                ),
                post36_start_authorization_path=Path(
                    self.attestation_authority["post36_start_authorization_path"]
                ),
                obsolete_cancellation_receipt_path=Path(
                    self.attestation_authority["obsolete_cancellation_receipt_path"]
                ),
            )
        self.assertEqual(set(payloads), publisher._BUNDLE_NAMES)
        self.assertEqual(
            payloads[binding.SYNTHETIC_INVARIANCE_NAME], self.comparison_bytes
        )
        declarations = json.loads(payloads[binding.REPORT_DECLARATIONS_NAME])
        self.assertEqual(len(declarations["rows"]), 36)
        self.assertNotIn(
            "synthetic_candidate_outcome_must_not_be_emitted",
            json.dumps(declarations),
        )
        manifest = json.loads(payloads[binding.TRANSITION_MANIFEST_NAME])
        self.assertEqual(set(manifest), binding._MANIFEST_FIELDS)
        self.assertTrue(manifest["operator_development_outcome_values_displayed"])
        self.assertFalse(
            manifest["programmatic_transition_extraction_outcome_fields_accessed"]
        )
        self.assertFalse(summary["values_used_for_transition_decisions"])


if __name__ == "__main__":
    unittest.main()
