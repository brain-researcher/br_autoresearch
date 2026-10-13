from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import observed_tail_plan

from ep12_executor import adaptive_successors
from ep12_executor.adaptive_search import coverage_configurations
from ep12_executor.attest_observed_trials import (
    ObservedAttestationError,
    build_observed_attestation,
    verify_attestation_root,
    write_attestation_artifacts,
)
from ep12_executor.policy import (
    EpisodePolicy,
    canonical_json,
    digest_bytes,
    digest_object,
)
from ep12_executor.qualification_launch import (
    QUALIFICATION_HASH_TARGETS,
    OUTPUT_RELATIVE,
    RECEIPT_RELATIVE,
    _TRANSITION_IDENTITY_FIELDS,
    launch_qualification003,
)
from ep12_executor.production_readiness import (
    ReadinessError,
    _expected_qualification_cells,
    claim_is_permitted,
    evaluate_production_readiness,
    scientific_configuration_sha256,
    verify_falsifier_plan,
    write_readiness_artifacts,
)
from ep12_executor.yaml_subset import load_yaml_subset
from ep12_executor.test_launch import TEST_WRAPPER_RELATIVE, launch_executor_tests


EPISODE_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = EPISODE_ROOT / "SEARCH_POLICY.yaml"
FROZEN_SPEC_PATH = EPISODE_ROOT / "outputs" / "executor" / "SYNTHETIC_QUALIFICATION_SPEC_POST36.yaml"
FROZEN_BUDGET_PATH = EPISODE_ROOT / "outputs" / "executor" / "DEVELOPMENT_BUDGET_OVERRIDE.yaml"


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _write_json(path: Path, value: object) -> None:
    _write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _file_hash(path: Path) -> str:
    return digest_bytes(path.read_bytes())


class ProductionReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "outputs"
        self.root.mkdir(parents=True)
        self.policy = EpisodePolicy.load(POLICY_PATH)
        transition_patch = mock.patch(
            "ep12_executor.qualification_launch._transition_identity",
            side_effect=self._transition_identity,
        )
        transition_patch.start()
        self.addCleanup(transition_patch.stop)
        gate_patch = mock.patch(
            "ep12_executor.production_readiness.verify_scientific_contract_transition_gate",
            side_effect=self._transition_gate,
        )
        gate_patch.start()
        self.addCleanup(gate_patch.stop)
        self.budget_path = self.root / "executor" / "DEVELOPMENT_BUDGET_OVERRIDE.yaml"
        _write(self.budget_path, FROZEN_BUDGET_PATH.read_bytes())
        self.plan_path = self.root / "executor" / "OBSERVED_POST36_TAIL_PLAN.json"
        _write_json(self.plan_path, observed_tail_plan.build_plan())
        self.launch_path, self.qualification_dir = self._qualification_fixture()
        self.run_root, self.finalization_dir = self._observed_chain_fixture()
        self.attestation_dir = self.run_root / "observed_attestation"
        attestation = build_observed_attestation(
            run_root=self.run_root,
            policy=self.policy,
            outputs_root=self.root,
        )
        write_attestation_artifacts(
            run_root=self.run_root,
            output_dir=self.attestation_dir,
            attestation=attestation,
            outputs_root=self.root,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _transition_identity(self, *, episode_root: Path, executor_root: Path) -> dict:
        value: dict[str, object] = {
            field: "a" * 64 for field in _TRANSITION_IDENTITY_FIELDS
        }
        value.update(
            {
                "transition_id": "ep12_post36_representation_rank_v1",
                "transition_manifest_path": str(
                    episode_root
                    / "outputs/prelaunch/scientific_contract_transition/manifest.json"
                ),
                "transition_receipt_path": str(
                    episode_root
                    / "outputs/prelaunch/scientific_contract_transition/receipt.json"
                ),
                "observed_trial_count": 36,
                "exact_36_attestation_set_matches_transition_evidence": True,
                "source_trial_reports_reverified": True,
                "operator_development_outcome_values_displayed": True,
                "programmatic_transition_extraction_outcome_fields_accessed": False,
                "values_used_for_transition_decisions": False,
                "final_connectivity_accessed": False,
                "active_contract_sha256": _file_hash(
                    episode_root / QUALIFICATION_HASH_TARGETS["scientific_contract_sha256"]
                ),
                "active_qualification_spec_sha256": _file_hash(
                    episode_root / QUALIFICATION_HASH_TARGETS["qualification_spec_sha256"]
                ),
                "active_scientific_models_sha256": _file_hash(
                    episode_root / QUALIFICATION_HASH_TARGETS["scientific_models_code_sha256"]
                ),
            }
        )
        self.assertEqual(executor_root, episode_root / "outputs/executor")
        return value

    def _transition_gate(self, **kwargs: object) -> dict[str, object]:
        outputs_root = Path(kwargs["outputs_root"])
        executor_root = Path(kwargs["executor_root"])
        value = self._transition_identity(
            episode_root=outputs_root.parent,
            executor_root=executor_root,
        )
        value["observed_prefix_contract_sha256"] = "b" * 64
        value["transition_receipt_file_sha256"] = "c" * 64
        return value

    def _qualification_fixture(self) -> tuple[Path, Path]:
        episode_root = self.root.parent
        qualification_dir = episode_root / OUTPUT_RELATIVE
        launch_path = episode_root / RECEIPT_RELATIVE
        source_files = {
            field: episode_root / relative
            for field, relative in QUALIFICATION_HASH_TARGETS.items()
        }
        for index, (field, path) in enumerate(source_files.items(), start=1):
            content = (
                FROZEN_SPEC_PATH.read_bytes()
                if field == "qualification_spec_sha256"
                else f"frozen-source-{index}\n".encode("ascii")
            )
            _write(path, content)
        _write(episode_root / "SEARCH_POLICY.yaml", POLICY_PATH.read_bytes())
        _write(
            episode_root / TEST_WRAPPER_RELATIVE,
            b"#!/bin/bash\nexit 0\n",
        )
        test_receipt = launch_executor_tests(
            episode_root=episode_root,
            submitter=lambda command: "70001;sherlock",
            job_resolver=lambda name, user: None,
        )
        launch_qualification003(
            test_job_id=test_receipt["slurm_job_id"],
            episode_root=episode_root,
            submitter=lambda command: "70002;sherlock",
            job_resolver=lambda name, user: None,
        )
        receipts = qualification_dir / "fit_and_score_receipts.jsonl"
        _write(receipts, b'{"synthetic":true}\n')
        repetitions = 72
        spec = load_yaml_subset(source_files["qualification_spec_sha256"])
        expected_cells = _expected_qualification_cells(spec)
        cells = [
            {
                "cell_id": cell_id,
                **cell,
                "replicates": repetitions,
                "passed": True,
                "failure_reasons": [],
                "terminal_counts": {
                    "M_supported_before_full_search_null": 0,
                    "single_population_adequate_before_sensitivity_gate": 0,
                    "unresolved": repetitions,
                    "numerical_failure": 0,
                },
            }
            for cell_id, cell in expected_cells.items()
        ]
        cost_profile = {
            "scope": "generated_data_measured_projection_not_real_runtime",
            "projected_cpu_core_hours": 2812.5,
            "policy_cpu_core_hour_ceiling": 1000,
            "cost_gate_passed": False,
        }
        stability = {"passed": True, "synthetic_fixture": True}
        _write_json(qualification_dir / "cost_profile.json", cost_profile)
        _write_json(qualification_dir / "integration_stability.json", stability)
        _write(qualification_dir / "result_table.csv", b"synthetic-table\n")
        _write(qualification_dir / "result_table.md", b"synthetic-table\n")
        _write_json(qualification_dir / "environment_lock.json", {"synthetic": True})
        transition = json.loads(launch_path.read_text(encoding="utf-8"))[
            "scientific_contract_transition"
        ]
        report = {
            "schema_version": "ep12.scientific_qualification_report.v2",
            "tier": "qualification",
            "qualification_passed": False,
            "qualification_scope": "actual_T_U_M_generated_data_only",
            "real_connectivity_accessed": False,
            "full_search_null_executed": False,
            "scientific_contract_enforced": True,
            "contract_configuration": True,
            "contract_sha256": _file_hash(source_files["scientific_contract_sha256"]),
            "spec_sha256": _file_hash(source_files["qualification_spec_sha256"]),
            "scientific_contract_transition": transition,
            "observed_prefix_contract_sha256": transition[
                "observed_prefix_contract_sha256"
            ],
            "active_contract_sha256": transition["active_contract_sha256"],
            "active_qualification_spec_sha256": transition[
                "active_qualification_spec_sha256"
            ],
            "settings": {
                "repetitions_per_cell": repetitions,
                "whole_types_per_replicate": 8,
            },
            "cells": cells,
            "cost_profile": cost_profile,
            "integration_stability": stability,
            "receipt_sha256": _file_hash(receipts),
        }
        _write_json(qualification_dir / "qualification_report.json", report)
        manifest = {
            "report_sha256": _file_hash(qualification_dir / "qualification_report.json"),
            "receipts_sha256": _file_hash(receipts),
            "result_table_csv_sha256": _file_hash(qualification_dir / "result_table.csv"),
            "result_table_markdown_sha256": _file_hash(qualification_dir / "result_table.md"),
            "cost_profile_sha256": _file_hash(qualification_dir / "cost_profile.json"),
            "environment_lock_sha256": _file_hash(qualification_dir / "environment_lock.json"),
            "integration_stability_sha256": _file_hash(
                qualification_dir / "integration_stability.json"
            ),
            "transition_manifest_file_sha256": transition[
                "transition_manifest_file_sha256"
            ],
            "transition_receipt_sha256": transition["transition_receipt_sha256"],
            "observed_prefix_contract_sha256": transition[
                "observed_prefix_contract_sha256"
            ],
            "active_contract_sha256": transition["active_contract_sha256"],
            "qualification_passed": False,
            "cost_gate_passed": False,
            "real_connectivity_accessed": False,
        }
        _write_json(qualification_dir / "manifest.json", manifest)
        return launch_path, qualification_dir

    def _write_plan_manifest(self, plan_dir: Path, names: list[str]) -> None:
        _write_json(
            plan_dir / "manifest.json",
            {
                "schema_version": "fixture.plan.manifest.v1",
                "files": {name: _file_hash(plan_dir / name) for name in names},
                "final_connectivity_accessed": False,
            },
        )

    def _write_trial(
        self,
        directory: Path,
        configuration: dict[str, object],
        *,
        synthetic_objective: float = 0.0,
    ) -> None:
        trial_id = str(configuration["trial_id"])
        left_to_right = float(synthetic_objective)
        right_to_left = left_to_right - 0.0001
        bilateral = (left_to_right + right_to_left) / 2.0
        _write(directory / "result_table.csv", b"opaque-real-outcome-table\n")
        _write(
            directory / "type_direction_receipts.jsonl",
            (
                json.dumps(
                    {
                        "status": "valid_complete_T_U_M_triplet",
                        "left_to_right_delta": "OBJECTIVE_MUST_NOT_BE_COMPARED",
                        "right_to_left_scores": {"OBJECTIVE": "MUST_NOT_BE_EMITTED"},
                    },
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8"),
        )
        _write_json(
            directory / "trial_report.json",
            {
                "schema_version": "ep12.development_trial.v1",
                "trial_id": trial_id,
                "status": "completed",
                "data_role": "development_only",
                "final_connectivity_accessed": False,
                "configuration": {
                    **configuration,
                    "configuration_sha256": digest_object(configuration),
                },
                "aggregate": {
                    "valid_type_count": 1,
                    "unscorable_type_count": 0,
                    "technical_failure_count": 0,
                    # Synthetic values exist only so the real adaptive planner
                    # can rank parents.  The attester must authenticate the
                    # report without reading or emitting any of them.
                    "mean_left_to_right_delta": left_to_right,
                    "mean_right_to_left_delta": right_to_left,
                    "mean_bilateral_delta": bilateral,
                    "types_with_both_point_estimates_above_margin": int(
                        min(left_to_right, right_to_left) > 0.02
                    ),
                },
                "unscorable": [],
                "failures": [],
                "objective_payload": {"MUST_NOT": "BE_EMITTED"},
            },
        )
        files = {
            name: _file_hash(directory / name)
            for name in (
                "result_table.csv",
                "trial_report.json",
                "type_direction_receipts.jsonl",
            )
        }
        _write_json(
            directory / "manifest.json",
            {
                "schema_version": "ep12.development_trial_manifest.v1",
                "files": files,
                "final_connectivity_accessed": False,
            },
        )

    def _observed_chain_fixture(self) -> tuple[Path, Path]:
        run_root = self.root / "development_run002"
        coverage = coverage_configurations(self.policy)
        coverage_path = run_root / "search_plan" / "coverage_configurations.json"
        _write_json(coverage_path, coverage)
        _write_json(run_root / "search_plan" / "search_plan.json", {"outcome_blind": True})
        self._write_plan_manifest(
            run_root / "search_plan",
            ["coverage_configurations.json", "search_plan.json"],
        )
        for index, configuration in enumerate(coverage, start=1):
            trial_id = str(configuration["trial_id"])
            self._write_trial(
                run_root / "observed" / trial_id,
                configuration,
                synthetic_objective=index / 1000.0,
            )

        temporary_episode_root = self.root.parent
        _write(temporary_episode_root / "SEARCH_POLICY.yaml", POLICY_PATH.read_bytes())
        source_specs = [(coverage_path, run_root / "observed")]
        self.real_adaptive_successor_payloads: list[dict[str, object]] = []
        with mock.patch.object(
            adaptive_successors, "EPISODE_ROOT", temporary_episode_root
        ):
            for cycle in (1, 2):
                plan_dir = run_root / "adaptive" / f"cycle{cycle}" / "plan"
                adaptive_successors.plan_successor_cycle(
                    cycle=cycle,
                    source_specs=source_specs,
                    output_dir=plan_dir,
                )
                configuration_path = plan_dir / "successor_configurations.json"
                payload = json.loads(configuration_path.read_text(encoding="utf-8"))
                self.real_adaptive_successor_payloads.append(payload)
                configurations = payload["configurations"]
                observed_root = (
                    run_root / "adaptive" / f"cycle{cycle}" / "observed"
                )
                for index, configuration in enumerate(configurations, start=1):
                    trial_id = str(configuration["trial_id"])
                    self._write_trial(
                        observed_root / trial_id,
                        configuration,
                        synthetic_objective=index / 100.0,
                    )
                source_specs.append((configuration_path, observed_root))

        _write_json(
            run_root / "adaptive_dag_launch.json",
            {
                "schema_version": "ep12.adaptive_dag_launch.v1",
                "minimum_observed_trial_count_after_success": 36,
                "final_connectivity_access_authorized": False,
            },
        )
        finalization = run_root / "observed_finalization"
        # This archive would contain real objectives in production.  Keep it
        # invalid JSON to ensure the readiness code never decodes it.
        _write(finalization / "observed_search_archive.json", b"opaque-observed-archive\n")
        _write_json(
            finalization / "manifest.json",
            {
                "schema_version": "ep12.observed_minimum_finalization_manifest.v1",
                "files": {
                    "observed_search_archive.json": _file_hash(
                        finalization / "observed_search_archive.json"
                    )
                },
                "final_connectivity_accessed": False,
            },
        )
        return run_root, finalization

    def test_attester_consumes_real_adaptive_successor_metadata(self) -> None:
        attestation = build_observed_attestation(
            run_root=self.run_root,
            policy=self.policy,
            outputs_root=self.root,
        )
        self.assertEqual(
            attestation["stage_trial_counts"],
            {"coverage": 18, "adaptive_cycle_1": 9, "adaptive_cycle_2": 9},
        )
        for cycle, payload in enumerate(
            self.real_adaptive_successor_payloads, start=1
        ):
            self.assertEqual(
                payload["schema_version"],
                "ep12.adaptive_successor_configurations.v1",
            )
            self.assertEqual(payload["cycle"], cycle)
            self.assertIn("proposal_rule", payload)
            self.assertTrue(
                all(
                    configuration["stage"] == "adaptive"
                    for configuration in payload["configurations"]
                )
            )
        self.assertFalse(attestation["objective_values_read"])
        self.assertFalse(attestation["objective_values_compared"])
        self.assertFalse(attestation["objective_values_emitted"])

    def _evaluate(self) -> dict[str, object]:
        launch = json.loads(self.launch_path.read_text(encoding="utf-8"))
        transition = launch["scientific_contract_transition"]
        return evaluate_production_readiness(
            qualification_launch_path=self.launch_path,
            qualification_dir=self.qualification_dir,
            observed_run_root=self.run_root,
            observed_finalization_dir=self.finalization_dir,
            observed_attestation_dir=self.attestation_dir,
            transition_manifest_path=Path(transition["transition_manifest_path"]),
            transition_manifest_file_sha256=transition["transition_manifest_file_sha256"],
            transition_receipt_sha256=transition["transition_receipt_sha256"],
            falsifier_plan_path=self.plan_path,
            budget_override_path=self.budget_path,
            policy_path=self.root.parent / "SEARCH_POLICY.yaml",
            outputs_root=self.root,
            executor_root=self.root / "executor",
        )

    def _rewrite_attestation(self, payload: dict[str, object]) -> None:
        artifact_path = self.attestation_dir / "observed_trial_attestations.json"
        material = {
            key: value for key, value in payload.items() if key != "attestation_sha256"
        }
        payload["attestation_sha256"] = digest_object(material)
        _write_json(artifact_path, payload)
        manifest_path = self.attestation_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["observed_trial_attestations.json"] = _file_hash(
            artifact_path
        )
        manifest["attestation_sha256"] = payload["attestation_sha256"]
        _write_json(manifest_path, manifest)

    def test_foundational_readiness_passes_without_reading_outcomes(self) -> None:
        report = self._evaluate()
        self.assertTrue(report["production_readiness_passed"])
        self.assertFalse(report["observed_outcome_values_read"])
        observed = report["checks"][2]["evidence"]
        self.assertEqual(observed["verified_trial_count"], 36)
        self.assertTrue(observed["source_status_metadata_parsed"])
        self.assertFalse(observed["outcome_values_read"])
        self.assertFalse(observed["outcome_values_compared"])
        self.assertFalse(observed["outcome_values_emitted"])
        self.assertFalse(observed["observed_search_archive_parsed"])
        self.assertTrue(observed["trial_attestations_verified"])
        self.assertTrue(observed["unique_scientific_configuration_hashes"])
        self.assertEqual(report["checks"][1]["evidence"]["cell_count"], 35)
        self.assertEqual(
            report["checks"][4]["evidence"]["user_rounded_planning_core_hours"],
            2750,
        )
        self.assertEqual(
            report["checks"][4]["evidence"][
                "qualification_projected_cpu_core_hours"
            ],
            2812.5,
        )
        self.assertTrue(
            report["checks"][4]["evidence"][
                "qualification_projection_recorded_not_used_as_hard_ceiling"
            ]
        )
        self.assertTrue(claim_is_permitted(report, "observed_tail"))

    def test_provisional_readiness_never_claims_lock_null_or_final_access(self) -> None:
        report = self._evaluate()
        self.assertFalse(claim_is_permitted(report, "procedure_lock"))
        self.assertFalse(claim_is_permitted(report, "full_search_null"))
        self.assertFalse(claim_is_permitted(report, "final_connectivity"))
        self.assertFalse(report["procedure_locked"])
        self.assertFalse(report["full_search_null_completed"])
        self.assertFalse(report["final_connectivity_access_authorized"])

    def test_qualification_hash_tamper_fails_closed(self) -> None:
        with (self.qualification_dir / "qualification_report.json").open("ab") as handle:
            handle.write(b"tamper")
        report = self._evaluate()
        self.assertFalse(report["production_readiness_passed"])
        self.assertIn("full_synthetic_qualification", report["failed_foundational_checks"])

    def test_readiness_rejects_noncanonical_launch_receipt(self) -> None:
        launch = json.loads(self.launch_path.read_text(encoding="utf-8"))
        _write_json(self.launch_path, launch)
        report = self._evaluate()
        self.assertFalse(report["production_readiness_passed"])
        self.assertIn("full_synthetic_qualification", report["failed_foundational_checks"])

    def test_readiness_rejects_rehashed_launch_semantic_mutation(self) -> None:
        launch = json.loads(self.launch_path.read_text(encoding="utf-8"))
        launch["whole_types_per_replicate"] = 9
        core = {
            key: value for key, value in launch.items() if key != "launch_sha256"
        }
        launch["launch_sha256"] = digest_object(core)
        _write(self.launch_path, (canonical_json(launch) + "\n").encode("utf-8"))
        report = self._evaluate()
        self.assertFalse(report["production_readiness_passed"])
        self.assertIn("full_synthetic_qualification", report["failed_foundational_checks"])

    def test_missing_observed_trial_manifest_fails_closed(self) -> None:
        (self.run_root / "observed" / "coverage_018" / "manifest.json").unlink()
        report = self._evaluate()
        self.assertFalse(report["production_readiness_passed"])
        self.assertIn(
            "observed_36_trial_provisional_chain", report["failed_foundational_checks"]
        )

    def test_attester_rejects_source_technical_failure_status(self) -> None:
        trial_dir = self.run_root / "observed" / "coverage_001"
        report_path = trial_dir / "trial_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["status"] = "completed_with_technical_failures"
        report["aggregate"]["technical_failure_count"] = 1
        report["failures"] = [{"failure_type": "SyntheticFailure"}]
        _write_json(report_path, report)
        manifest_path = trial_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["trial_report.json"] = _file_hash(report_path)
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(
            ObservedAttestationError, "not technically complete"
        ):
            build_observed_attestation(
                run_root=self.run_root,
                policy=self.policy,
                outputs_root=self.root,
            )

    def test_attester_rejects_missing_T_U_M_or_wrong_data_role(self) -> None:
        trial_dir = self.run_root / "observed" / "coverage_001"
        report_path = trial_dir / "trial_report.json"
        original = json.loads(report_path.read_text(encoding="utf-8"))
        mutations = (
            ("data_role", "final"),
            ("models", list(self.policy.model_triplet[:-1])),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                report = json.loads(json.dumps(original))
                if field == "models":
                    report["configuration"]["models"] = value
                else:
                    report[field] = value
                _write_json(report_path, report)
                manifest_path = trial_dir / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["files"]["trial_report.json"] = _file_hash(report_path)
                _write_json(manifest_path, manifest)
                with self.assertRaises(ObservedAttestationError):
                    build_observed_attestation(
                        run_root=self.run_root,
                        policy=self.policy,
                        outputs_root=self.root,
                    )

    def test_attester_never_emits_or_compares_objective_payloads(self) -> None:
        trial_dir = self.run_root / "observed" / "coverage_001"
        report_path = trial_dir / "trial_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["aggregate"]["mean_bilateral_delta"] = {
            "arbitrary": "changed objective payload"
        }
        report["objective_payload"] = ["arbitrary", "unread", "values"]
        _write_json(report_path, report)
        receipt_path = trial_dir / "type_direction_receipts.jsonl"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["left_to_right_delta"] = {"arbitrary": "changed"}
        _write(
            receipt_path,
            (json.dumps(receipt, sort_keys=True) + "\n").encode("utf-8"),
        )
        manifest_path = trial_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["trial_report.json"] = _file_hash(report_path)
        manifest["files"]["type_direction_receipts.jsonl"] = _file_hash(receipt_path)
        _write_json(manifest_path, manifest)
        attestation = build_observed_attestation(
            run_root=self.run_root,
            policy=self.policy,
            outputs_root=self.root,
        )
        serialized = json.dumps(attestation, sort_keys=True)
        self.assertNotIn("changed objective payload", serialized)
        self.assertNotIn('"arbitrary": "changed"', serialized)
        self.assertFalse(attestation["objective_values_read"])
        self.assertFalse(attestation["objective_values_compared"])
        self.assertFalse(attestation["objective_values_emitted"])

    def test_attestation_write_is_idempotent_and_preserves_source_hashes(self) -> None:
        source_paths = sorted(
            path
            for path in self.run_root.rglob("*")
            if path.is_file() and self.attestation_dir not in path.parents
        )
        before = {str(path): _file_hash(path) for path in source_paths}
        attestation = build_observed_attestation(
            run_root=self.run_root,
            policy=self.policy,
            outputs_root=self.root,
        )
        write_attestation_artifacts(
            run_root=self.run_root,
            output_dir=self.attestation_dir,
            attestation=attestation,
            outputs_root=self.root,
        )
        evidence = verify_attestation_root(
            run_root=self.run_root,
            attestation_dir=self.attestation_dir,
            policy=self.policy,
            outputs_root=self.root,
        )
        after = {str(path): _file_hash(path) for path in source_paths}
        self.assertEqual(before, after)
        self.assertEqual(evidence["verified_trial_count"], 36)
        self.assertFalse(evidence["source_trial_files_modified"])

    def test_missing_frozen_qualification_cell_fails_even_when_rehashed(self) -> None:
        report_path = self.qualification_dir / "qualification_report.json"
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        report_payload["cells"].pop()
        _write_json(report_path, report_payload)
        manifest_path = self.qualification_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["report_sha256"] = _file_hash(report_path)
        _write_json(manifest_path, manifest)
        evaluated = self._evaluate()
        self.assertIn(
            "full_synthetic_qualification", evaluated["failed_foundational_checks"]
        )

    def test_rehashed_separate_attestation_config_tamper_fails_closed(self) -> None:
        artifact_path = self.attestation_dir / "observed_trial_attestations.json"
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        payload["trials"][0]["configuration_sha256"] = "00" * 32
        self._rewrite_attestation(payload)
        report = self._evaluate()
        self.assertIn(
            "observed_36_trial_provisional_chain", report["failed_foundational_checks"]
        )

    def test_missing_separate_attestation_root_fails_closed(self) -> None:
        (self.attestation_dir / "manifest.json").unlink()
        report = self._evaluate()
        self.assertIn(
            "observed_36_trial_provisional_chain", report["failed_foundational_checks"]
        )

    def test_incomplete_or_technically_failed_attestation_fails_closed(self) -> None:
        artifact_path = self.attestation_dir / "observed_trial_attestations.json"
        original = json.loads(artifact_path.read_text(encoding="utf-8"))
        mutations = (
            ("complete_T_U_M_triplet", False),
            ("technical_failure_count", 1),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                manifest = json.loads(json.dumps(original))
                manifest["trials"][0][field] = value
                self._rewrite_attestation(manifest)
                report = self._evaluate()
                self.assertIn(
                    "observed_36_trial_provisional_chain",
                    report["failed_foundational_checks"],
                )
        self._rewrite_attestation(original)

    def test_identity_and_stage_cannot_hide_duplicate_scientific_config(self) -> None:
        path = self.run_root / "search_plan" / "coverage_configurations.json"
        configurations = json.loads(path.read_text(encoding="utf-8"))
        duplicate = dict(configurations[0])
        duplicate["trial_id"] = configurations[-1]["trial_id"]
        duplicate["stage"] = configurations[-1]["stage"]
        self.assertNotEqual(digest_object(duplicate), digest_object(configurations[0]))
        self.assertEqual(
            scientific_configuration_sha256(duplicate),
            scientific_configuration_sha256(configurations[0]),
        )
        configurations[-1] = duplicate
        _write_json(path, configurations)
        self._write_plan_manifest(
            path.parent, ["coverage_configurations.json", "search_plan.json"]
        )
        self._write_trial(
            self.run_root / "observed" / str(duplicate["trial_id"]), duplicate
        )
        report = self._evaluate()
        self.assertIn(
            "observed_36_trial_provisional_chain", report["failed_foundational_checks"]
        )

    def test_plan_missing_nontrial_policy_control_fails_closed(self) -> None:
        payload = json.loads(self.plan_path.read_text(encoding="utf-8"))
        payload["non_trial_required_gates"] = payload["non_trial_required_gates"][1:]
        payload.pop("plan_sha256")
        payload["plan_sha256"] = digest_object(payload)
        _write_json(self.plan_path, payload)
        with self.assertRaisesRegex(ValueError, "frozen declarative plan"):
            verify_falsifier_plan(
                plan_path=self.plan_path,
                policy=self.policy,
                outputs_root=self.root,
            )

    def test_rehashed_semantic_plan_mutation_fails_exact_validator(self) -> None:
        payload = json.loads(self.plan_path.read_text(encoding="utf-8"))
        payload["all_slots"][0]["selector"]["tie_rule"] = "mutated_but_deterministic"
        slot = dict(payload["all_slots"][0])
        slot.pop("slot_sha256")
        payload["all_slots"][0]["slot_sha256"] = digest_object(slot)
        payload.pop("plan_sha256")
        payload["plan_sha256"] = digest_object(payload)
        _write_json(self.plan_path, payload)
        with self.assertRaisesRegex(ValueError, "frozen declarative plan"):
            verify_falsifier_plan(
                plan_path=self.plan_path,
                policy=self.policy,
                outputs_root=self.root,
            )

    def test_budget_override_tamper_fails_closed(self) -> None:
        with self.budget_path.open("ab") as handle:
            handle.write(b"\n# tamper\n")
        report = self._evaluate()
        self.assertIn(
            "development_budget_override", report["failed_foundational_checks"]
        )

    def test_readiness_artifact_retry_recovers_after_report_only(self) -> None:
        output_dir = self.root / "recovery"
        report = {
            "schema_version": "synthetic.readiness.v1",
            "production_readiness_passed": True,
        }
        _write_json(output_dir / "production_readiness.json", report)
        write_readiness_artifacts(output_dir, report, outputs_root=self.root)
        self.assertTrue((output_dir / "manifest.json").is_file())
        first_manifest = (output_dir / "manifest.json").read_bytes()
        write_readiness_artifacts(output_dir, report, outputs_root=self.root)
        self.assertEqual((output_dir / "manifest.json").read_bytes(), first_manifest)
        _write_json(
            output_dir / "production_readiness.json",
            {**report, "production_readiness_passed": False},
        )
        with self.assertRaisesRegex(ReadinessError, "differs from retry payload"):
            write_readiness_artifacts(output_dir, report, outputs_root=self.root)


if __name__ == "__main__":
    unittest.main()
