from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ep12_executor.contract_snapshot import (
    ContractSnapshotError,
    V1_CONTRACT_SHA256,
    assert_immutable_v1_contract,
    capture_yaml_contract,
)
from ep12_executor.scientific_contract_transition import (
    ScientificContractTransitionError,
    build_transition_receipt,
    verify_contract_transition_definition,
    verify_transition_receipt,
)
from ep12_executor.yaml_subset import YamlSubsetError, parse_yaml_subset_bytes

try:
    import numpy as np

    import ep12_executor.scientific_models as scientific_models
    from ep12_executor.pre_null_materializer import (
        PreNullMaterializerError,
        _component_rank,
    )

    SCIENTIFIC_STACK_AVAILABLE = True
except (ImportError, RuntimeError):
    SCIENTIFIC_STACK_AVAILABLE = False


EXECUTOR_ROOT = Path(__file__).resolve().parents[1]


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def dynamic_attestation_authority() -> dict:
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
        "path": "/synthetic/episode/outputs/prelaunch/post36_orchestration/receipt.json",
        "file_sha256": None,
        "receipt_sha256": None,
        "required_before_irreversible_transition": True,
    }
    core = {
        "schema_version": "ep12.dynamic_observed_attestation_authority.v1",
        "authoritative_job_id": "987654",
        "authoritative_job_name": "ep12-observed-attestation-replacement",
        "replacement_terminal_identity_path": "/synthetic/episode/outputs/prelaunch/replacement_attestation_launch/terminal_identity.json",
        "replacement_terminal_identity_file_sha256": "21" * 32,
        "replacement_terminal_identity_sha256": "22" * 32,
        "replacement_launch_receipt_file_sha256": "23" * 32,
        "replacement_launch_receipt_sha256": "24" * 32,
        "replacement_release_receipt_file_sha256": "25" * 32,
        "replacement_release_receipt_sha256": "26" * 32,
        "post36_orchestration_intent_path": "/synthetic/episode/outputs/prelaunch/post36_orchestration/intent.json",
        "post36_orchestration_intent_file_sha256": "27" * 32,
        "post36_orchestration_intent_sha256": "28" * 32,
        "post36_start_authorization_path": "/synthetic/episode/outputs/prelaunch/post36_orchestration/start_authorization.json",
        "post36_start_authorization_file_sha256": "29" * 32,
        "post36_start_authorization_sha256": "2a" * 32,
        "obsolete_cancellation_receipt_path": "/synthetic/episode/outputs/prelaunch/replacement_attestation_launch/obsolete_cancellation/receipt.json",
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
    return {**core, "authority_sha256": canonical_digest(core)}


class ContractSnapshotTests(unittest.TestCase):
    def test_yaml_is_parsed_from_the_exact_digest_bytes(self) -> None:
        raw = b"schema_version: synthetic.v1\nvalue: [1, 2, true]\n"
        self.assertEqual(
            parse_yaml_subset_bytes(raw),
            {"schema_version": "synthetic.v1", "value": [1, 2, True]},
        )
        with self.assertRaisesRegex(YamlSubsetError, "valid UTF-8"):
            parse_yaml_subset_bytes(b"value: \xff\n")

    def test_v1_contract_is_byte_immutable_and_snapshot_detects_change(self) -> None:
        source = EXECUTOR_ROOT / "SCIENTIFIC_CONTRACT.yaml"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "SCIENTIFIC_CONTRACT.yaml"
            shutil.copyfile(source, path)
            snapshot = assert_immutable_v1_contract(path)
            self.assertEqual(snapshot.sha256, V1_CONTRACT_SHA256)
            path.write_bytes(snapshot.content + b"# mutation\n")
            with self.assertRaisesRegex(ContractSnapshotError, "changed after snapshot"):
                snapshot.assert_unchanged()
            with self.assertRaisesRegex(ContractSnapshotError, "digest mismatch"):
                assert_immutable_v1_contract(path)

    def test_leaf_symlink_is_rejected(self) -> None:
        source = EXECUTOR_ROOT / "SCIENTIFIC_CONTRACT.yaml"
        with tempfile.TemporaryDirectory() as temporary:
            link = Path(temporary) / "contract.yaml"
            link.symlink_to(source)
            with self.assertRaisesRegex(ContractSnapshotError, "without following links"):
                capture_yaml_contract(link)


class TransitionDefinitionTests(unittest.TestCase):
    def test_post36_definition_has_only_the_allowlisted_delta(self) -> None:
        verified = verify_contract_transition_definition(executor_root=EXECUTOR_ROOT)
        self.assertEqual(verified["old_contract_sha256"], V1_CONTRACT_SHA256)
        self.assertNotEqual(
            verified["old_contract_sha256"], verified["new_contract_sha256"]
        )
        self.assertFalse(verified["final_connectivity_accessed"])

    def _attestation_and_reports(self):
        rows = []
        reports = {}
        for index in range(36):
            trial_id = f"synthetic_trial_{index:02d}"
            report = {
                "schema_version": "ep12.development_trial.v1",
                "trial_id": trial_id,
                "status": "completed",
                "data_role": "development_only",
                "final_connectivity_accessed": False,
                "scientific_contract_sha256": V1_CONTRACT_SHA256,
                "aggregate": {"synthetic_outcome_not_accessed": float(index)},
            }
            report_bytes = json.dumps(report, sort_keys=True).encode("utf-8")
            reports[trial_id] = report_bytes
            report_hash = hashlib.sha256(report_bytes).hexdigest()
            rows.append(
                {
                    "trial_id": trial_id,
                    "source_files_sha256": {"trial_report.json": report_hash},
                    "source_report_schema": "ep12.development_trial.v1",
                    "source_report_status": "completed",
                    "data_role": "development_only",
                    "final_connectivity_accessed": False,
                }
            )
        core = {
            "schema_version": "ep12.observed_trial_attestation_set.v1",
            "episode_id": "ep12",
            "trial_count": 36,
            "stage_trial_counts": {
                "coverage": 18,
                "adaptive_cycle_1": 9,
                "adaptive_cycle_2": 9,
            },
            "trials": rows,
            "objective_values_read": False,
            "objective_values_compared": False,
            "objective_values_emitted": False,
            "development_only": True,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }
        return {**core, "attestation_sha256": canonical_digest(core)}, reports

    def _invariance_evidence(self):
        transition_spec = json.loads(
            (EXECUTOR_ROOT / "CONTRACT_TRANSITION_SPEC.json").read_text(
                encoding="utf-8"
            )
        )
        required = transition_spec["required_synthetic_invariance_evidence"]
        core = {
            "schema_version": required["schema_version"],
            "status": "pass",
            "outcome_firewall": {
                "generated_examples_only": True,
                "real_connectivity_accessed": False,
                "connectivity_paths_accepted": False,
                "annotation_or_outcome_files_accessed": False,
            },
            "spec_sha256": required["probe_spec_sha256"],
            "legacy": {
                "contract_sha256": transition_spec["old_contract"]["sha256"]
            },
            "post36": {
                "contract_sha256": transition_spec["new_contract"]["sha256"]
            },
            "checks": {
                "input_payload_byte_identical": True,
                "environment_byte_identical": True,
                "scientific_payload_byte_identical": True,
                "diagnostic_changes_limited_to_added_representation_rank": True,
                "diagnostic_additions": required[
                    "allowed_diagnostic_additions"
                ],
            },
        }
        return {**core, "artifact_sha256": canonical_digest(core)}

    def test_synthetic_outcome_blind_receipt_round_trip(self) -> None:
        attestation, reports = self._attestation_and_reports()
        receipt = build_transition_receipt(
            executor_root=EXECUTOR_ROOT,
            observed_attestation=attestation,
            observed_attestation_manifest_sha256="ab" * 32,
            observed_attestation_authority=dynamic_attestation_authority(),
            authenticated_trial_report_bytes=reports,
            synthetic_invariance_evidence=self._invariance_evidence(),
        )
        verify_transition_receipt(receipt)
        self.assertTrue(receipt["every_observed_report_declares_old_contract"])
        self.assertTrue(receipt["operator_development_outcome_values_displayed"])
        self.assertFalse(
            receipt["programmatic_transition_extraction_outcome_fields_accessed"]
        )
        self.assertFalse(receipt["values_used_for_transition_decisions"])
        self.assertFalse(receipt["final_connectivity_accessed"])

    def test_report_declarations_are_exact_authenticated_and_outcome_blind(self) -> None:
        attestation, reports = self._attestation_and_reports()
        first_id = sorted(reports)[0]
        reports[first_id] += b"\n"
        with self.assertRaisesRegex(
            ScientificContractTransitionError, "authenticated hash"
        ):
            build_transition_receipt(
                executor_root=EXECUTOR_ROOT,
                observed_attestation=attestation,
                observed_attestation_manifest_sha256="ab" * 32,
                observed_attestation_authority=dynamic_attestation_authority(),
                authenticated_trial_report_bytes=reports,
                synthetic_invariance_evidence=self._invariance_evidence(),
            )
        attestation, reports = self._attestation_and_reports()
        receipt = build_transition_receipt(
            executor_root=EXECUTOR_ROOT,
            observed_attestation=attestation,
            observed_attestation_manifest_sha256="ab" * 32,
            observed_attestation_authority=dynamic_attestation_authority(),
            authenticated_trial_report_bytes=reports,
            synthetic_invariance_evidence=self._invariance_evidence(),
        )
        self.assertNotIn("synthetic_outcome_not_accessed", json.dumps(receipt))

    def test_wrong_contract_or_invariance_evidence_fails_closed(self) -> None:
        attestation, reports = self._attestation_and_reports()
        trial_id = sorted(reports)[4]
        report = json.loads(reports[trial_id].decode("utf-8"))
        report["scientific_contract_sha256"] = "ef" * 32
        reports[trial_id] = json.dumps(report, sort_keys=True).encode("utf-8")
        for row in attestation["trials"]:
            if row["trial_id"] == trial_id:
                row["source_files_sha256"]["trial_report.json"] = hashlib.sha256(
                    reports[trial_id]
                ).hexdigest()
        attestation["attestation_sha256"] = canonical_digest(
            {
                key: value
                for key, value in attestation.items()
                if key != "attestation_sha256"
            }
        )
        with self.assertRaisesRegex(
            ScientificContractTransitionError, "immutable v1 contract"
        ):
            build_transition_receipt(
                executor_root=EXECUTOR_ROOT,
                observed_attestation=attestation,
                observed_attestation_manifest_sha256="ab" * 32,
                observed_attestation_authority=dynamic_attestation_authority(),
                authenticated_trial_report_bytes=reports,
                synthetic_invariance_evidence=self._invariance_evidence(),
            )
        attestation, reports = self._attestation_and_reports()
        evidence = self._invariance_evidence()
        evidence["checks"]["scientific_payload_byte_identical"] = False
        evidence["artifact_sha256"] = canonical_digest(
            {key: value for key, value in evidence.items() if key != "artifact_sha256"}
        )
        with self.assertRaisesRegex(
            ScientificContractTransitionError, "check failed"
        ):
            build_transition_receipt(
                executor_root=EXECUTOR_ROOT,
                observed_attestation=attestation,
                observed_attestation_manifest_sha256="ab" * 32,
                observed_attestation_authority=dynamic_attestation_authority(),
                authenticated_trial_report_bytes=reports,
                synthetic_invariance_evidence=evidence,
            )

    def test_receipt_tampering_fails_self_hash(self) -> None:
        attestation, reports = self._attestation_and_reports()
        receipt = build_transition_receipt(
            executor_root=EXECUTOR_ROOT,
            observed_attestation=attestation,
            observed_attestation_manifest_sha256="ab" * 32,
            observed_attestation_authority=dynamic_attestation_authority(),
            authenticated_trial_report_bytes=reports,
            synthetic_invariance_evidence=self._invariance_evidence(),
        )
        receipt["observed_trial_count"] = 35
        with self.assertRaisesRegex(
            ScientificContractTransitionError, "self-hash mismatch"
        ):
            verify_transition_receipt(receipt)


@unittest.skipUnless(
    SCIENTIFIC_STACK_AVAILABLE,
    "needs the pinned Sherlock numpy/scipy/scikit-learn module environment",
)
class RepresentationRankTests(unittest.TestCase):
    @staticmethod
    def _counts_and_covariates():
        rng = np.random.default_rng(20260926)
        counts = rng.poisson(7.0, size=(10, 10)).astype(np.int64) + 1
        covariates = np.linspace(-1.0, 1.0, 10)[:, None]
        target = rng.poisson(8.0, size=(6, 10)).astype(np.int64) + 1
        target_covariates = np.linspace(-0.7, 0.7, 6)[:, None]
        return counts, covariates, target, target_covariates

    def test_count_aware_effective_rank_is_residual_dimension_bounded_by_n(self) -> None:
        counts, covariates, _, _ = self._counts_and_covariates()
        model = scientific_models.fit_predictive_model(
            counts,
            covariates,
            "U_linear",
            seed=17,
            integration_draws=4,
            representation="count_aware_log_ratio",
            covariance_mode="diagonal",
            covariance_rank=1,
        )
        self.assertEqual(model.diagnostics["representation_rank"], 9)
        self.assertEqual(
            model.diagnostics["representation_rank_definition"],
            "representation_latent_effective_rank",
        )

    def test_low_rank_diagnostic_follows_representation_not_covariance(self) -> None:
        counts, covariates, _, _ = self._counts_and_covariates()
        for rank in (2, 4):
            for covariance in ("diagonal", "shrinkage", "low_rank"):
                model = scientific_models.fit_predictive_model(
                    counts,
                    covariates,
                    "U_linear",
                    seed=20 + rank,
                    integration_draws=4,
                    latent_rank=rank,
                    representation="low_rank_composition",
                    covariance_mode=covariance,
                    covariance_rank=1,
                )
                self.assertEqual(model.diagnostics["representation_rank"], rank)
                self.assertEqual(model.diagnostics["covariance_rank"], 1)

    def test_component_rank_uses_required_representation_diagnostic(self) -> None:
        fitted = SimpleNamespace(
            M=SimpleNamespace(
                prior_means=np.zeros((2, 9)),
                diagnostics={
                    "representation": "low_rank_count_model",
                    "representation_rank": 4,
                    "covariance_rank": 1,
                },
            )
        )
        configuration = {
            "representation": "low_rank_count_model",
            "rank": 4,
            "covariance_rank": 1,
        }
        repeated_side_component_ranks = [
            _component_rank(
                rule="representation_latent_effective_rank",
                fitted=fitted,
                source_neuron_count=source_neuron_count,
                configuration=configuration,
            )
            for source_neuron_count in (10, 12)
            for _component in range(3)
        ]
        self.assertEqual(repeated_side_component_ranks, [4] * 6)
        fitted.M.diagnostics.pop("representation_rank")
        with self.assertRaisesRegex(
            PreNullMaterializerError, "lacks the required representation-rank"
        ):
            _component_rank(
                rule="representation_latent_effective_rank",
                fitted=fitted,
                source_neuron_count=10,
                configuration=configuration,
            )
        with self.assertRaisesRegex(
            PreNullMaterializerError, "not the frozen post-36 choice"
        ):
            _component_rank(
                rule="shared_within_component_covariance_effective_rank",
                fitted=fitted,
                source_neuron_count=10,
                configuration=configuration,
            )

    def test_diagnostic_only_change_is_bitwise_numerically_invariant(self) -> None:
        counts, covariates, target, target_covariates = self._counts_and_covariates()
        original = scientific_models._representation_residuals

        def legacy_representation_residuals(*args, **kwargs):
            residuals, diagnostics = original(*args, **kwargs)
            diagnostics = dict(diagnostics)
            diagnostics.pop("representation_rank_definition", None)
            if diagnostics.get("representation") == "count_aware_log_ratio":
                diagnostics.pop("representation_rank", None)
            return residuals, diagnostics

        arrays = (
            "basis",
            "nuisance_coefficients",
            "latent_offsets",
            "log_weights",
            "prior_means",
            "prior_covariances",
            "prior_weights",
        )
        for representation in (
            "count_aware_log_ratio",
            "low_rank_composition",
            "low_rank_count_model",
        ):
            for covariance in ("diagonal", "shrinkage", "low_rank"):
                options = {
                    "seed": 222,
                    "integration_draws": 4,
                    "latent_rank": 2,
                    "representation": representation,
                    "covariance_mode": covariance,
                    "covariance_rank": 1,
                }
                current = scientific_models.fit_predictive_model(
                    counts, covariates, "U_linear", **options
                )
                with mock.patch.object(
                    scientific_models,
                    "_representation_residuals",
                    side_effect=legacy_representation_residuals,
                ):
                    legacy = scientific_models.fit_predictive_model(
                        counts, covariates, "U_linear", **options
                    )
                for field in arrays:
                    np.testing.assert_array_equal(
                        getattr(current, field), getattr(legacy, field)
                    )
                np.testing.assert_array_equal(
                    current.score_neurons(target, target_covariates),
                    legacy.score_neurons(target, target_covariates),
                )


if __name__ == "__main__":
    unittest.main()
