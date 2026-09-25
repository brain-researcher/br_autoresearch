from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ep12_executor.controller import (
    AuditError,
    RoleLeakageError,
    SimulatedInterruption,
    StateTransitionError,
    SyntheticSourceManifest,
    WholeTypeRoleManifest,
    balanced_coverage_configs,
)
from ep12_executor.synthetic import (
    EPISODE_ROOT,
    SCRATCH_ROOT,
    SyntheticAuditVault,
    _verify_completed_artifacts,
    _evaluation,
    atomic_json,
    final_result,
    make_controller,
    null_receipts,
    run_budget_exhaustion,
    run_complete_terminal_scenario,
    run_input_failure,
    run_no_valid_comparison,
)
from ep12_executor.journal import HashChainedJournal
from ep12_executor.policy import digest_bytes, digest_object


class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=SCRATCH_ROOT)
        self.root = Path(self.temporary.name)
        self.policy_path = EPISODE_ROOT / "SEARCH_POLICY.yaml"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _register_first(self, controller, trial_id: str = "trial-0001") -> dict:
        config = balanced_coverage_configs(controller.policy)[0]
        controller.register_hypothesis(
            trial_id=trial_id,
            parent_trial_ids=(),
            proposal_mode="prespecified_coverage",
            proposal_context_hash=controller.journal.prefix_hash(),
            outcome_evidence_refs=(),
            successor_cycle_id=None,
            prediction="synthetic prediction",
            falsifier="synthetic falsifier",
            changed_operator="coverage factor",
            unchanged_operators=("T_U_M_triplet",),
            expected_information_gain="state-machine coverage",
            expected_cost={"synthetic_core_hours": 0.0},
            config=config,
            stage="coverage",
        )
        return config

    def test_role_manifest_is_whole_type_and_synthetic_source_is_fail_closed(self) -> None:
        with self.assertRaises(RoleLeakageError):
            WholeTypeRoleManifest(
                development_types=("same",), final_types=("same",)
            ).validate()
        with self.assertRaises(Exception):
            SyntheticSourceManifest(
                source_id="bad",
                contains_real_connectivity=True,
                real_connectivity_access_count=1,
            ).validate()

    def test_interruption_resumes_without_double_counting(self) -> None:
        run_dir = self.root / "resume"
        controller = make_controller(run_dir, self.policy_path)
        self._register_first(controller)
        evaluation = _evaluation(controller, selection_statistic=0.001)
        with self.assertRaises(SimulatedInterruption):
            controller.run_development_trial(
                "trial-0001", evaluation, interrupt_after_running=True
            )
        resumed = make_controller(run_dir, self.policy_path)
        resumed.run_development_trial("trial-0001", evaluation)
        resumed.run_development_trial("trial-0001", evaluation)
        resumed.update_archive(
            "trial-0001", "INCUMBENT", rationale="synthetic improvement", decision=True
        )
        state = resumed.replay()
        self.assertEqual(state.valid_trial_count, 1)
        self.assertEqual(
            sum(
                record["state"] == "SCORED"
                for record in state.trials["trial-0001"]
            ),
            1,
        )

    def test_incomplete_triplet_and_role_leak_do_not_count(self) -> None:
        controller = make_controller(self.root / "invalid", self.policy_path)
        self._register_first(controller)
        incomplete = _evaluation(controller, selection_statistic=0.0)
        incomplete["models_completed"] = list(controller.policy.model_triplet[:2])
        controller.run_development_trial("trial-0001", incomplete)
        self.assertEqual(controller.replay().valid_trial_count, 0)

        leaking = make_controller(self.root / "leaking", self.policy_path)
        self._register_first(leaking)
        evaluation = _evaluation(leaking, selection_statistic=0.0)
        evaluation["accessed_types"].append(leaking.role_manifest.final_types[0])
        with self.assertRaises(RoleLeakageError):
            leaking.run_development_trial("trial-0001", evaluation)
        self.assertEqual(leaking.replay().valid_trial_count, 0)
        resumed = make_controller(self.root / "leaking", self.policy_path)
        with self.assertRaises(StateTransitionError):
            resumed.run_development_trial("trial-0001", _evaluation(resumed, selection_statistic=0.0))
        terminal = resumed.close_without_audit(
            "technical_failure", "durably recorded synthetic role leakage"
        )
        self.assertEqual(terminal["detailed_outcome"], "technical_failure")

    def test_terminal_rules_use_only_frozen_policy_margin(self) -> None:
        controller = make_controller(self.root / "decisions", self.policy_path)
        positive = controller._decide_terminal(final_result("positive", controller), 0.01)
        adequate = controller._decide_terminal(final_result("adequate", controller), 0.01)
        unresolved = controller._decide_terminal(final_result("unresolved", controller), 0.01)
        self.assertEqual(positive[0], "candidate_ready_residual_modes")
        self.assertEqual(adequate[0], "closed_single_population_adequate_within_margin")
        self.assertEqual(unresolved[0], "closed_unresolved")

        not_adequate = final_result("adequate", controller)
        not_adequate["adequacy_sensitivity_qualified"] = False
        self.assertEqual(
            controller._decide_terminal(not_adequate, 0.01)[0], "closed_unresolved"
        )
        failed_control = final_result("positive", controller)
        failed_control["residual_support"][controller.RESIDUAL_SUPPORT_KEYS[0]] = False
        self.assertEqual(
            controller._decide_terminal(failed_control, 0.01)[0], "closed_unresolved"
        )
        self.assertEqual(
            controller._decide_terminal(final_result("positive", controller), 0.06)[0],
            "closed_unresolved",
        )
        missing_type = final_result("positive", controller)
        missing_type["reported_final_types"] = missing_type["reported_final_types"][:-1]
        self.assertEqual(
            controller._decide_terminal(missing_type, 0.01)[0], "technical_failure"
        )
        missing_direction = final_result("positive", controller)
        del missing_direction["per_type_results"][0]["directions"]["right_to_left"]
        self.assertEqual(
            controller._decide_terminal(missing_direction, 0.01)[0], "technical_failure"
        )
        nonfinite = final_result("positive", controller)
        nonfinite["directions"]["left_to_right"]["gain"] = float("inf")
        self.assertEqual(
            controller._decide_terminal(nonfinite, 0.01)[0], "technical_failure"
        )

    def test_failure_terminals_are_distinct_and_never_open_final(self) -> None:
        no_valid = run_no_valid_comparison(self.root / "no-valid", self.policy_path)
        budget = run_budget_exhaustion(self.root / "budget", self.policy_path)
        input_failure = run_input_failure(self.root / "input", self.policy_path)
        self.assertEqual(
            no_valid["terminal"]["detailed_outcome"], "closed_no_valid_comparison"
        )
        self.assertEqual(budget["terminal"]["detailed_outcome"], "incomplete_search")
        self.assertEqual(input_failure["terminal"]["detailed_outcome"], "technical_failure")
        self.assertTrue(
            all(
                scenario["status"]["final_open_count"] == 0
                for scenario in (no_valid, budget, input_failure)
            )
        )

        premature = make_controller(self.root / "premature", self.policy_path)
        with self.assertRaises(StateTransitionError):
            premature.close_without_audit(
                "closed_no_valid_comparison", "no development was attempted"
            )

    def test_full_positive_recovery_and_exactly_once_final(self) -> None:
        scenario = run_complete_terminal_scenario(
            kind="positive",
            run_dir=self.root / "positive",
            policy_path=self.policy_path,
            inject_trial_interruption=True,
            inject_lost_final_acknowledgement=True,
        )
        self.assertEqual(
            scenario["terminal"]["detailed_outcome"], "candidate_ready_residual_modes"
        )
        self.assertEqual(scenario["status"]["valid_trials"], 52)
        self.assertTrue(scenario["status"]["coverage_complete"])
        self.assertEqual(scenario["null_replicates"], 99)
        self.assertGreaterEqual(
            scenario["protocol_scale_model_family_fit_count"], 10800
        )
        self.assertTrue(scenario["trial_interruption_recovered"])
        self.assertTrue(scenario["lost_acknowledgement_reconciled"])
        self.assertEqual(scenario["status"]["final_open_count"], 1)
        self.assertAlmostEqual(
            scenario["status"]["post_coverage_falsifier_fraction"], 14 / 34
        )
        self.assertEqual(scenario["evaluator_invocation_count"], 1)
        self.assertEqual(scenario["evaluator_access_count"], 1)
        persisted_receipts = json.loads(
            Path(scenario["synthetic_null_receipts_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(len(persisted_receipts), 99)
        self.assertTrue(
            all(
                receipt["receipt_scope"] == "asserted_synthetic_not_executed"
                and receipt["complete_controller_rerun"] is False
                and receipt["deterministic_replay_passed"] is False
                for receipt in persisted_receipts
            )
        )

        controller = make_controller(self.root / "positive", self.policy_path)
        self.assertEqual(
            controller.lock_configuration(
                "trial-0036",
                synthetic_selection_rule_id="synthetic_qualification_only",
                final_output_allowlist=controller.SYNTHETIC_FINAL_OUTPUT_FIELDS,
            ),
            scenario["lock_hash"],
        )
        with self.assertRaises(StateTransitionError):
            self._register_first(controller, trial_id="post-audit")
        vault = SyntheticAuditVault(
            self.root / "positive" / "second-vault.json",
            final_result("positive", controller),
        )
        with self.assertRaises(AuditError):
            controller.run_audit(
                scenario["lock_hash"], vault, operation_id="different-final-operation"
            )
        with self.assertRaises(AuditError):
            controller.run_audit(
                "0" * 64, vault, operation_id="synthetic-final-positive"
            )

        required = {
            "trial_id",
            "timestamp_utc",
            "previous_record_hash",
            "record_hash",
            "parent_trial_ids",
            "proposal_mode",
            "proposal_context_hash",
            "outcome_evidence_refs",
            "successor_cycle_id",
            "configuration_lock_hash",
            "stage",
            "hypothesis",
            "directional_prediction",
            "falsifier",
            "changed_operator",
            "unchanged_operators",
            "expected_information_gain",
            "expected_cost",
            "config",
            "hashes",
            "state",
            "resources",
            "belief_update",
            "successor_rationale",
            "disposition",
        }
        self.assertTrue(
            all(required.issubset(record) for record in controller.journal.read())
        )

        selection_program_hash = digest_object(
            {
                "id": "synthetic_qualification_only",
                "policy_hash": controller.policy.policy_hash,
            }
        )
        receipts = null_receipts(
            controller,
            observed_statistic=0.036,
            selection_program_hash=selection_program_hash,
        )
        replayed_null = controller.run_full_search_null(
            generator_lock_hash=digest_object({"synthetic_generator": "U"}),
            replicate_manifest=receipts,
            selected_trial_id="trial-0036",
            observed_statistic=0.036,
            selection_program_hash=selection_program_hash,
        )
        self.assertEqual(replayed_null["event_id"], "operation:full-search-null")

    def test_full_adequate_and_unresolved_paths(self) -> None:
        expected = {
            "adequate": "closed_single_population_adequate_within_margin",
            "unresolved": "closed_unresolved",
        }
        for kind, detailed in expected.items():
            with self.subTest(kind=kind):
                scenario = run_complete_terminal_scenario(
                    kind=kind,
                    run_dir=self.root / kind,
                    policy_path=self.policy_path,
                )
                self.assertEqual(scenario["terminal"]["detailed_outcome"], detailed)
                self.assertEqual(scenario["status"]["final_open_count"], 1)
                self.assertEqual(scenario["null_replicates"], 99)

    def test_malformed_final_becomes_durable_technical_failure(self) -> None:
        fixture_controller = make_controller(self.root / "fixture", self.policy_path)
        malformed = final_result("positive", fixture_controller)
        malformed["undeclared_candidate_diagnostic"] = "must not escape"
        scenario = run_complete_terminal_scenario(
            kind="positive",
            run_dir=self.root / "malformed-final",
            policy_path=self.policy_path,
            final_payload=malformed,
        )
        self.assertEqual(scenario["terminal"]["detailed_outcome"], "technical_failure")
        self.assertFalse(scenario["terminal"]["final_evaluation_valid"])
        controller = make_controller(self.root / "malformed-final", self.policy_path)
        receipt = controller.replay().audit_result_record["audit_receipt"]
        self.assertNotIn("undeclared_candidate_diagnostic", receipt["result"])
        self.assertTrue(receipt["result_validation_errors"])

    def test_paths_are_restricted_to_ep12_scratch_or_outputs(self) -> None:
        with self.assertRaises(ValueError):
            make_controller(Path("/tmp/ep12-forbidden"), self.policy_path)
        with self.assertRaises(ValueError):
            make_controller(self.root / "wrong-policy", self.root / "SEARCH_POLICY.yaml")

    def test_invalid_attempts_still_exhaust_resource_budget(self) -> None:
        controller = make_controller(self.root / "invalid-budget", self.policy_path)
        configs = balanced_coverage_configs(controller.policy)
        for index in range(1, 13):
            trial_id = f"invalid-budget-{index:04d}"
            controller.register_hypothesis(
                trial_id=trial_id,
                parent_trial_ids=(),
                proposal_mode="prespecified_coverage",
                proposal_context_hash=controller.journal.prefix_hash(),
                outcome_evidence_refs=(),
                successor_cycle_id=None,
                prediction="synthetic invalid budget attempt",
                falsifier="synthetic readiness check",
                changed_operator="coverage factor",
                unchanged_operators=("T_U_M_triplet",),
                expected_information_gain="resource accounting",
                expected_cost={"synthetic_core_hours": 84.0},
                config=configs[(index - 1) % len(configs)],
                stage="coverage",
            )
            controller.run_development_trial(
                trial_id,
                _evaluation(
                    controller,
                    selection_statistic=0.0,
                    cpu_core_hours=84.0,
                    comparison_gate_overrides={
                        "T_U_M_capacity_comparison_is_fair": False
                    },
                ),
            )
        self.assertEqual(controller.replay().valid_trial_count, 0)
        self.assertEqual(controller.check_stop(), ("stop", "cpu_budget_exhausted"))
        with self.assertRaises(StateTransitionError):
            self._register_first(controller, "after-budget")

    def test_stop_activating_score_retry_is_idempotent(self) -> None:
        controller = make_controller(self.root / "stop-retry", self.policy_path)
        configs = balanced_coverage_configs(controller.policy)
        evaluations = []
        for index, core_hours in enumerate((333.0, 333.0, 334.0), start=1):
            trial_id = f"stop-retry-{index:04d}"
            controller.register_hypothesis(
                trial_id=trial_id,
                parent_trial_ids=(),
                proposal_mode="prespecified_coverage",
                proposal_context_hash=controller.journal.prefix_hash(),
                outcome_evidence_refs=(),
                successor_cycle_id=None,
                prediction="synthetic stop-boundary retry",
                falsifier="synthetic readiness check",
                changed_operator="coverage factor",
                unchanged_operators=("T_U_M_triplet",),
                expected_information_gain="idempotent recovery",
                expected_cost={"synthetic_core_hours": core_hours},
                config=configs[index - 1],
                stage="coverage",
            )
            evaluation = _evaluation(
                controller,
                selection_statistic=index / 1000.0,
                cpu_core_hours=core_hours,
            )
            evaluations.append(evaluation)
            controller.run_development_trial(trial_id, evaluation)
            if index < 3:
                controller.update_archive(
                    trial_id,
                    "INCUMBENT",
                    rationale="synthetic improvement",
                    decision=True,
                )
        self.assertEqual(controller.check_stop(), ("stop", "cpu_budget_exhausted"))
        replayed = controller.run_development_trial("stop-retry-0003", evaluations[-1])
        self.assertEqual(replayed["state"], "SCORED")
        self.assertEqual(controller.replay().valid_trial_count, 3)

    def test_malformed_numeric_receipts_are_durable_failures(self) -> None:
        bad_metric = make_controller(self.root / "bad-metric", self.policy_path)
        self._register_first(bad_metric)
        evaluation = _evaluation(bad_metric, selection_statistic=0.0)
        evaluation["metrics"]["selection_statistic"] = float("nan")
        record = bad_metric.run_development_trial("trial-0001", evaluation)
        self.assertEqual(record["state"], "INVALID")
        self.assertEqual(bad_metric.replay().valid_trial_count, 0)

        bad_resource = make_controller(self.root / "bad-resource", self.policy_path)
        self._register_first(bad_resource)
        evaluation = _evaluation(bad_resource, selection_statistic=0.0)
        evaluation["resources"]["cpu_core_hours"] = "not-a-number"
        with self.assertRaises(Exception):
            bad_resource.run_development_trial("trial-0001", evaluation)
        self.assertIsNotNone(bad_resource.replay().input_failure_record)
        resumed = make_controller(self.root / "bad-resource", self.policy_path)
        with self.assertRaises(StateTransitionError):
            resumed.run_development_trial("trial-0001", evaluation)

    def test_proposal_metadata_is_fail_closed(self) -> None:
        controller = make_controller(self.root / "metadata", self.policy_path)
        config = balanced_coverage_configs(controller.policy)[0]
        base = {
            "trial_id": "metadata-0001",
            "parent_trial_ids": (),
            "proposal_mode": "prespecified_coverage",
            "proposal_context_hash": controller.journal.prefix_hash(),
            "outcome_evidence_refs": (),
            "successor_cycle_id": None,
            "prediction": "synthetic prediction",
            "falsifier": "synthetic falsifier",
            "changed_operator": "coverage factor",
            "unchanged_operators": ("T_U_M_triplet",),
            "expected_information_gain": "metadata validation",
            "expected_cost": {"synthetic_core_hours": 0.0},
            "config": config,
            "stage": "coverage",
        }
        for field, value in (
            ("trial_id", ""),
            ("stage", "made_up_stage"),
            ("proposal_mode", "outcome_adaptive_successor"),
            ("prediction", "  "),
            ("unchanged_operators", ()),
            ("expected_cost", {}),
        ):
            with self.subTest(field=field):
                malformed = dict(base)
                malformed[field] = value
                with self.assertRaises(StateTransitionError):
                    controller.register_hypothesis(**malformed)
        self.assertEqual(controller.journal.read(), [])

    def test_completed_artifact_verification_detects_tampering(self) -> None:
        output_dir = self.root / "completed-artifacts"
        output_dir.mkdir()
        journal_path = output_dir / "trial_ledger.jsonl"
        journal = HashChainedJournal(journal_path)
        journal.append(
            event_id="synthetic-event",
            timestamp_utc="2026-09-24T00:00:00Z",
            payload={"fixture": True},
        )
        vault_path = output_dir / "synthetic_final_vault.json"
        atomic_json(vault_path, {"fixture": True})
        receipt_path = output_dir / "synthetic_null_receipts.json"
        receipts = [
            {
                "receipt_scope": "asserted_synthetic_not_executed",
                "complete_controller_rerun": False,
                "deterministic_replay_passed": False,
            }
            for _ in range(99)
        ]
        atomic_json(receipt_path, receipts)
        base = {
            "journal_path": str(journal_path),
            "journal_hash": digest_object(
                journal_path.read_text(encoding="utf-8").splitlines()
            ),
        }
        complete = {
            **base,
            "synthetic_final_vault_path": str(vault_path),
            "synthetic_final_vault_sha256": digest_bytes(vault_path.read_bytes()),
            "synthetic_null_receipts_path": str(receipt_path),
            "synthetic_null_receipts_sha256": digest_bytes(receipt_path.read_bytes()),
        }
        report = {
            "policy": {"null_replicates": 99},
            "scenarios": [
                {**complete, "scenario": "positive"},
                {**complete, "scenario": "adequate"},
                {**complete, "scenario": "unresolved"},
                {**base, "scenario": "no_valid_comparison"},
                {**base, "scenario": "budget_exhaustion"},
                {**base, "scenario": "input_failure"},
            ],
        }
        _verify_completed_artifacts(output_dir, report)
        journal_path.write_text("tampered\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            _verify_completed_artifacts(output_dir, report)


if __name__ == "__main__":
    unittest.main()
