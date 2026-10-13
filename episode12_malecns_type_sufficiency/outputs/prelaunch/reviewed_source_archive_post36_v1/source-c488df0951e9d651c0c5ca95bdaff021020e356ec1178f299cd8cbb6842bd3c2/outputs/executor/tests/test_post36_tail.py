from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import observed_tail_plan

from ep12_executor.adaptive_search import coverage_configurations
from ep12_executor.adaptive_successors import _ranking_key, _successors
from ep12_executor.contract_provenance import (
    ACTIVE_EPOCH,
    PREFIX_EPOCH,
    PROVENANCE_SCHEMA,
    RANK_RULE,
)
from ep12_executor.policy import EpisodePolicy, digest_object
from ep12_executor.post36_tail import (
    OBSERVED_SEARCH_STARTED_UTC,
    OBSERVED_SEARCH_WALL_DEADLINE_UTC,
    Post36TailController,
    TAIL_REALIZATION_SCHEMA,
    TailExecutionError,
    WALL_DEADLINE_STOP_REASON,
    assess_observed_search_wall_budget,
    capability_report,
    controller_code_identity,
    observed_search_wall_budget,
    production_control_capabilities,
    promotion_decision,
)
from ep12_executor.scientific_contract_transition import TRANSITION_RECEIPT_SCHEMA
from ep12_executor.synthetic import EPISODE_ROOT
from tests.test_contract_transition import dynamic_attestation_authority


def _synthetic_contract_provenance() -> dict[str, Any]:
    authority = dynamic_attestation_authority()
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "prefix_contract": {
            "epoch": PREFIX_EPOCH, "path": "synthetic-prefix", "sha256": "1" * 64,
        },
        "active_contract": {
            "epoch": ACTIVE_EPOCH, "path": "synthetic-active", "sha256": "2" * 64,
        },
        "transition": {
            "schema_version": TRANSITION_RECEIPT_SCHEMA,
            "complete": True,
            "path": "synthetic-transition",
            "semantic_sha256": "3" * 64,
            "file_sha256": "4" * 64,
            "spec_path": "synthetic-spec",
            "spec_sha256": "5" * 64,
            "spec_file_sha256": "6" * 64,
            "observed_attestation_authority": authority,
            "controller_final_receipt_required_before_irreversible_transition": True,
            "authoritative_attestation_job_id": authority[
                "authoritative_job_id"
            ],
            "obsolete_attestation_job_id": authority["obsolete_job_id"],
        },
        "prevalence_parameter_rank_definition": RANK_RULE,
        "operator_development_outcome_values_displayed": True,
        "values_used_for_transition_decisions": False,
        "final_connectivity_accessed": False,
    }


def _objective(value: float) -> dict[str, Any]:
    return {
        "left_to_right": value,
        "right_to_left": value - 0.001,
        "bilateral": value - 0.0005,
        "bilateral_floor": value - 0.001,
        "direction_disagreement": 0.001,
        "types_both_above_margin": 0,
    }


def _record(configuration: Mapping[str, Any], value: float) -> dict[str, Any]:
    return {
        "trial_id": configuration["trial_id"],
        "configuration": dict(configuration),
        "configuration_sha256": digest_object(configuration),
        "objective": _objective(value),
        "valid_type_count": 8,
        "unscorable_type_count": 0,
        "evidence": {
            "trial_report_sha256": digest_object([configuration["trial_id"], "report"]),
            "manifest_sha256": digest_object([configuration["trial_id"], "manifest"]),
        },
    }


def _prefix(policy: EpisodePolicy) -> list[dict[str, Any]]:
    coverage = coverage_configurations(policy)
    coverage_records = [
        _record(configuration, 0.010 + index / 10000.0)
        for index, configuration in enumerate(coverage)
    ]
    cycle1, _ = _successors(
        policy=policy,
        cycle=1,
        ranked_parents=sorted(coverage_records, key=_ranking_key),
        seen_configurations=coverage,
        count=9,
    )
    cycle1_records = [
        _record(configuration, 0.020 + index / 10000.0)
        for index, configuration in enumerate(cycle1)
    ]
    cycle2, _ = _successors(
        policy=policy,
        cycle=2,
        ranked_parents=sorted(cycle1_records, key=_ranking_key),
        seen_configurations=coverage + cycle1,
        count=9,
    )
    cycle2_records = [
        _record(configuration, 0.030 + index / 10000.0)
        for index, configuration in enumerate(cycle2)
    ]
    return coverage_records + cycle1_records + cycle2_records


class FakeExecutor:
    def __init__(
        self,
        plan: Mapping[str, Any],
        *,
        challenger_improves: bool = False,
        fail_after_durable_once: bool = False,
        malformed: bool = False,
    ):
        countable = plan["required_falsifier_classification"][
            "countable_observed_trial_ids"
        ]
        self._capabilities = {
            str(value): {
                "supported": True,
                "implementation": f"synthetic::{value}",
            }
            for value in countable
        }
        self.challenger_improves = challenger_improves
        self.fail_after_durable_once = fail_after_durable_once
        self.malformed = malformed
        self.durable: dict[str, Mapping[str, Any]] = {}
        self.executions: list[str] = []
        self.failed_once = False

    def capabilities(self) -> Mapping[str, Mapping[str, Any]]:
        return self._capabilities

    def recover(self, proposal: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return self.durable.get(str(proposal["trial_id"]))

    def execute(self, proposal: Mapping[str, Any]) -> Mapping[str, Any]:
        trial_id = str(proposal["trial_id"])
        self.executions.append(trial_id)
        if proposal["promotion_eligible"] is True:
            value = 10.0 if self.challenger_improves else -10.0
        else:
            # Controls are intentionally made much stronger than the prefix.
            # They still must never replace the primary incumbent.
            value = 100.0
        falsifier_id = proposal["registered_falsifier_id"]
        realization: dict[str, Any] = {
            "schema_version": TAIL_REALIZATION_SCHEMA,
            "trial_id": trial_id,
            "slot_sha256": proposal["slot_sha256"],
            "configuration_sha256": proposal["configuration_sha256"],
            "proposal_context_hash": proposal["proposal_context_hash"],
            "status": "completed",
            "valid_complete_T_U_M_triplet": True,
            "data_role": "development_only",
            "final_connectivity_accessed": False,
            "technical_failure_count": 0,
            "valid_type_count": 8,
            "unscorable_type_count": 0,
            "objective": _objective(value),
            **{
                key: proposal[key] for key in (
                    "contract_epoch", "scientific_contract_sha256",
                    "contract_transition_semantic_sha256",
                    "contract_transition_file_sha256",
                )
            },
            "control_receipt": {
                "falsifier_id": falsifier_id,
                "transforms": [
                    {
                        "transform": (
                            f"synthetic::{falsifier_id}"
                            if falsifier_id
                            else "identity_primary_development_snapshot"
                        )
                    }
                ],
                **{
                    key: proposal[key] for key in (
                        "contract_epoch", "scientific_contract_sha256",
                        "contract_transition_semantic_sha256",
                        "contract_transition_file_sha256",
                    )
                },
                "applicability_checks_passed": True,
                "conservation_checks_passed": True,
                "control_passed": True,
            },
            "resources": {
                "wall_hours": 0.001,
                "cpu_core_hours": 0.001,
                "gpu_hours": 0.0,
            },
            "evidence": {
                "bundle_manifest_sha256": digest_object([trial_id, "bundle"]),
                "trial_report_sha256": digest_object([trial_id, "report"]),
            },
        }
        if self.malformed:
            realization["technical_failure_count"] = 1
        self.durable[trial_id] = realization
        if self.fail_after_durable_once and not self.failed_once:
            self.failed_once = True
            raise TailExecutionError(
                "synthetic lost acknowledgement after durable bundle",
                evidence={"durable_bundle_present": True},
            )
        return realization


class Post36TailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.plan = observed_tail_plan.build_plan()
        self.prefix = _prefix(self.policy)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _controller(
        self,
        executor: FakeExecutor,
        *,
        plan: Mapping[str, Any] | None = None,
        now_utc: Callable[[], datetime] | None = None,
    ) -> Post36TailController:
        selected_plan = dict(plan or self.plan)
        controller = Post36TailController(
            policy=self.policy,
            plan=selected_plan,
            journal_path=self.root / "ledger.jsonl",
            status_path=self.root / "controller_status.json",
            executor=executor,
            identity={
                "policy_sha256": self.policy.policy_hash,
                "plan_sha256": selected_plan["plan_sha256"],
                "fixture": True,
                "final_connectivity_accessed": False,
                "observed_search_wall_budget": observed_search_wall_budget(
                    self.policy
                ),
                "contract_provenance": _synthetic_contract_provenance(),
            },
            now_utc=now_utc
            or (lambda: datetime(2026, 9, 26, 6, 0, tzinfo=timezone.utc)),
        )
        controller.initialize(
            prefix_records=self.prefix,
            prefix_evidence={
                "synthetic_fixture": True,
                "final_connectivity_accessed": False,
            },
        )
        return controller

    def test_wall_budget_is_exactly_the_authenticated_120_hour_interval(self) -> None:
        budget = observed_search_wall_budget(self.policy)
        self.assertEqual(
            budget["observed_search_started_utc"], OBSERVED_SEARCH_STARTED_UTC
        )
        self.assertEqual(
            budget["overall_wall_deadline_utc"],
            OBSERVED_SEARCH_WALL_DEADLINE_UTC,
        )
        self.assertEqual(budget["wall_clock_hour_ceiling"], 120.0)
        self.assertEqual(budget["wall_hours_per_trial_maximum"], 12.0)
        self.assertEqual(
            budget["decision_scope"], "outcome_blind_tail_slot_admission"
        )

    def test_wall_budget_boundary_admits_exact_latest_start_only(self) -> None:
        budget = observed_search_wall_budget(self.policy)
        exact = assess_observed_search_wall_budget(
            policy=self.policy,
            wall_budget=budget,
            now_utc=datetime(2026, 9, 30, 17, 57, 34, tzinfo=timezone.utc),
        )
        late = assess_observed_search_wall_budget(
            policy=self.policy,
            wall_budget=budget,
            now_utc=datetime(2026, 9, 30, 17, 57, 35, tzinfo=timezone.utc),
        )
        self.assertTrue(exact["can_start_next_tail_slot"])
        self.assertEqual(
            exact["projected_next_slot_completion_utc"],
            OBSERVED_SEARCH_WALL_DEADLINE_UTC,
        )
        self.assertFalse(late["can_start_next_tail_slot"])
        self.assertTrue(
            late["deadline_would_be_violated_by_next_maximum_duration_slot"]
        )
        self.assertFalse(late["outcome_values_read_for_decision"])

    def test_late_start_writes_durable_pending_stop_without_execution(self) -> None:
        executor = FakeExecutor(self.plan)
        late = datetime(2026, 9, 30, 17, 57, 35, tzinfo=timezone.utc)
        controller = self._controller(executor, now_utc=lambda: late)
        first = controller.run_one()
        self.assertEqual(first["action"], "wall_deadline_pending")
        self.assertEqual(first["reason"], WALL_DEADLINE_STOP_REASON)
        self.assertEqual(first["current_valid_ordinal"], 36)
        self.assertFalse(first["next_tail_slot_submission_permitted"])
        self.assertEqual(executor.executions, [])
        self.assertEqual(controller.replay().stop_record["reason"], WALL_DEADLINE_STOP_REASON)

        receipt_path = self.root / "wall_deadline_stop.json"
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
        receipt_core = dict(receipt)
        receipt_sha256 = receipt_core.pop("receipt_sha256")
        self.assertEqual(receipt_sha256, digest_object(receipt_core))
        self.assertFalse(receipt["next_tail_slot_submission_permitted"])
        self.assertFalse(receipt["procedure_lock_submission_permitted"])
        self.assertFalse(receipt["outcome_values_read_for_deadline_decision"])
        self.assertTrue((self.root / "realized_schedule.json").is_file())

        record_count = controller.journal.verify()["record_count"]
        second = controller.run_one()
        self.assertEqual(second["action"], "wall_deadline_pending")
        self.assertEqual(controller.journal.verify()["record_count"], record_count)
        self.assertEqual(receipt_path.read_bytes(), receipt_bytes)
        self.assertEqual(executor.executions, [])

    def test_completion_stops_before_chaining_slot_that_no_longer_fits(self) -> None:
        executor = FakeExecutor(self.plan)
        moments = iter(
            (
                datetime(2026, 9, 30, 17, 57, 34, tzinfo=timezone.utc),
                datetime(2026, 9, 30, 17, 57, 35, tzinfo=timezone.utc),
            )
        )
        controller = self._controller(executor, now_utc=lambda: next(moments))
        status = controller.run_one()
        self.assertEqual(status["action"], "wall_deadline_pending")
        self.assertEqual(status["current_valid_ordinal"], 37)
        self.assertEqual(len(executor.executions), 1)
        self.assertFalse(status["next_tail_slot_submission_permitted"])
        receipt = json.loads((self.root / "wall_deadline_stop.json").read_text())
        self.assertEqual(receipt["current_valid_ordinal"], 37)
        self.assertEqual(receipt["next_tail_slot_valid_ordinal"], 38)

    def test_controller_identity_binds_complete_decision_import_path(self) -> None:
        identity = controller_code_identity()
        self.assertTrue(
            {
                "post36_tail.py",
                "adaptive_successors.py",
                "journal.py",
                "policy.py",
                "production_readiness.py",
                "scientific_models.py",
                "scientific_qualification.py",
                "scientific_synthetic.py",
                "synthetic.py",
                "yaml_subset.py",
                "observed_tail_plan.py",
                "FALSIFIER_CONTROL_IMPLEMENTATION_SPEC.json",
            }
            <= set(identity)
        )
        self.assertTrue(all(len(value) == 64 for value in identity.values()))

    def test_no_improvement_reaches_exact_qualified_patience_stop(self) -> None:
        executor = FakeExecutor(self.plan)
        controller = self._controller(executor)
        status: Mapping[str, Any] = {}
        for _ in range(16):
            status = controller.run_one()
        self.assertEqual(status["action"], "stopped")
        self.assertEqual(status["reason"], "patience_exhausted")
        self.assertEqual(status["current_valid_ordinal"], 52)
        expected_falsifiers = sum(
            slot["slot_kind"] == "mandatory_falsifier"
            for slot in self.plan["all_slots"]
            if 37 <= int(slot["valid_trial_ordinal"]) <= 52
        )
        self.assertEqual(
            status["counted_post_coverage_falsifiers"], expected_falsifiers
        )
        self.assertGreaterEqual(expected_falsifiers * 5, (52 - 18) * 2)
        self.assertTrue(status["falsifier_fraction_gate"])
        self.assertTrue(status["all_countable_falsifier_ids_gate"])
        replay = controller.replay()
        self.assertEqual(replay.incumbent["trial_id"], self.prefix[-1]["trial_id"])
        self.assertTrue((self.root / "realized_schedule.json").is_file())
        schedule = json.loads(
            (self.root / "realized_schedule.json").read_text(encoding="utf-8")
        )
        self.assertEqual(schedule["realized_slot_count"], 16)
        self.assertFalse(schedule["procedure_locked"])
        self.assertFalse(schedule["non_trial_required_gates_completed"])

    def test_promotion_eligible_improvement_resets_patience_and_enters_reserve(self) -> None:
        early_challengers = [
            slot
            for slot in self.plan["all_slots"]
            if 37 <= int(slot["valid_trial_ordinal"]) <= 52
            and slot["promotion_eligible"] is True
        ]
        if not early_challengers:
            self.skipTest("waiting for corrected frozen schedule with early challengers")
        executor = FakeExecutor(self.plan, challenger_improves=True)
        controller = self._controller(executor)
        status: Mapping[str, Any] = {}
        for _ in range(16):
            status = controller.run_one()
        self.assertEqual(status["action"], "continue")
        self.assertEqual(status["current_valid_ordinal"], 52)
        replay = controller.replay()
        self.assertEqual(
            replay.last_promotion_improvement_ordinal,
            int(early_challengers[0]["valid_trial_ordinal"]),
        )
        next_status = controller.run_one()
        self.assertEqual(next_status["current_valid_ordinal"], 53)

    def test_control_result_cannot_replace_incumbent(self) -> None:
        incumbent = self.prefix[-1]
        control = dict(incumbent)
        control["trial_id"] = "synthetic_control"
        control["configuration_sha256"] = "0" * 64
        control["objective"] = _objective(1000.0)
        decision = promotion_decision(
            incumbent=incumbent,
            candidate=control,
            promotion_eligible=False,
        )
        self.assertFalse(decision["improved"])
        self.assertEqual(decision["incumbent_trial_id"], incumbent["trial_id"])

    def test_lost_acknowledgement_recovers_without_double_counting(self) -> None:
        executor = FakeExecutor(self.plan, fail_after_durable_once=True)
        controller = self._controller(executor)
        first = controller.run_one()
        self.assertEqual(first["action"], "retry_required")
        self.assertEqual(first["current_valid_ordinal"], 36)
        second = controller.run_one()
        self.assertEqual(second["current_valid_ordinal"], 37)
        replay = controller.replay()
        self.assertEqual(len(replay.dispositions), 1)
        self.assertEqual(len(executor.executions), 1)

    def test_invalid_attempt_does_not_advance_valid_ordinal(self) -> None:
        executor = FakeExecutor(self.plan, malformed=True)
        controller = self._controller(executor)
        status = controller.run_one()
        self.assertEqual(status["action"], "retry_required")
        self.assertEqual(status["current_valid_ordinal"], 36)
        replay = controller.replay()
        self.assertEqual(len(replay.dispositions), 0)
        self.assertEqual(len(replay.failures), 1)

    def test_production_capability_audit_requires_exact_all_control_support(self) -> None:
        report = capability_report(self.plan, production_control_capabilities())
        self.assertTrue(report["whole_plan_executable"])
        self.assertEqual(report["unsupported_falsifiers"], [])
        self.assertEqual(report["unsupported_slot_ids"], [])
        self.assertEqual(
            set(report["supported_falsifier_ids"]),
            set(
                self.plan["required_falsifier_classification"][
                    "countable_observed_trial_ids"
                ]
            ),
        )
        self.assertFalse(report["substitution_allowed"])


if __name__ == "__main__":
    unittest.main()
