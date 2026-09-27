from __future__ import annotations

import copy
import unittest
from typing import Any, Mapping, Sequence

import observed_tail_plan as tail

from ep12_executor.adaptive_search import coverage_configurations
from ep12_executor.adaptive_successors import _ranking_key, _successors
from ep12_executor.null_controller import (
    NullControllerError,
    disposition_for_result,
    freeze_null_controller_program,
    propose_next_trial,
    records_from_steps,
    stop_status,
    verify_null_controller_program,
)
from ep12_executor.policy import EpisodePolicy, digest_object
from ep12_executor.runtime import EPISODE_ROOT


def objective(value: float, *, supported_types: int = 0) -> dict[str, Any]:
    """Return a complete, internally simple synthetic ranking objective."""

    return {
        "left_to_right": value,
        "right_to_left": value,
        "bilateral": value,
        "bilateral_floor": value,
        "direction_disagreement": 0.0,
        "types_both_above_margin": supported_types,
    }


def committed_step(
    *,
    proposal: Mapping[str, Any],
    value: Mapping[str, Any],
    prior_steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the minimal immutable step shape consumed by the controller."""

    disposition = disposition_for_result(
        proposal=proposal,
        objective=value,
        prior_steps=prior_steps,
    )
    fit_core = {
        "objective": dict(value),
        "synthetic": True,
        "final_connectivity_accessed": False,
    }
    fit_result = {**fit_core, "result_sha256": digest_object(fit_core)}
    core = {
        "proposal": copy.deepcopy(dict(proposal)),
        "fit_result": fit_result,
        "disposition": disposition,
        "previous_step_sha256": (
            prior_steps[-1]["step_sha256"] if prior_steps else None
        ),
    }
    return {**core, "step_sha256": digest_object(core)}


class NullControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")
        cls.coverage = coverage_configurations(cls.policy)
        cls.countable = list(tail.COUNTABLE_OBSERVED_FALSIFIERS)
        cls.non_trial = [
            str(value["falsifier_id"]) for value in tail.NON_TRIAL_REQUIRED_GATES
        ]
        cls.plan = tail.build_plan()
        cls.program = freeze_null_controller_program(
            policy=cls.policy,
            coverage_configurations=cls.coverage,
            post36_tail_plan=cls.plan,
            countable_falsifier_ids=cls.countable,
            non_trial_required_falsifier_ids=cls.non_trial,
        )

    def proposal_ledger(
        self,
        count: int,
        *,
        values: Mapping[int, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Run only the outcome-blind controller against synthetic objectives."""

        overrides = dict(values or {})
        steps: list[dict[str, Any]] = []
        while len(steps) < count:
            proposal = propose_next_trial(
                program=self.program,
                policy=self.policy,
                prior_steps=steps,
            )
            ordinal = len(steps) + 1
            value = objective(overrides.get(ordinal, ordinal / 1000.0))
            steps.append(
                committed_step(
                    proposal=proposal,
                    value=value,
                    prior_steps=steps,
                )
            )
        return steps

    def arithmetic_ledger(
        self,
        count: int,
        *,
        tail_improvements: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Make policy-valid record shapes without exercising model fitting."""

        improvements = set(tail_improvements or ())
        steps: list[dict[str, Any]] = []
        base = copy.deepcopy(self.coverage[0])
        for ordinal in range(1, count + 1):
            configuration = copy.deepcopy(base)
            configuration["trial_id"] = f"synthetic_{ordinal:03d}"
            configuration["stage"] = "coverage" if ordinal <= 18 else "adaptive"
            if ordinal <= 18:
                slot_kind = "coverage"
                falsifier_id = None
                eligible = True
            elif ordinal <= 36:
                slot_kind = "adaptive_challenger"
                falsifier_id = None
                eligible = True
            else:
                slot = self.program["post36_slots"][ordinal - 37]
                slot_kind = str(slot["slot_kind"])
                falsifier_id = slot["falsifier_id"]
                eligible = slot["promotion_eligible"] is True
            proposal_core = {
                "valid_trial_ordinal": ordinal,
                "trial_id": configuration["trial_id"],
                "configuration": configuration,
                "configuration_sha256": digest_object(configuration),
                "slot_kind": slot_kind,
                "falsifier_id": falsifier_id,
                "promotion_eligible": eligible,
                "derived_from_null_outcomes_only": True,
            }
            proposal = {
                **proposal_core,
                "proposal_sha256": digest_object(proposal_core),
            }
            value = objective(
                2.0 if ordinal in improvements else 1.0 if ordinal == 1 else 0.0
            )
            step = committed_step(
                proposal=proposal,
                value=value,
                prior_steps=steps,
            )
            if ordinal in improvements:
                self.assertTrue(step["disposition"]["improved"])
            steps.append(step)
        return steps

    def assert_exhaustion_probe_is_deterministic_when_exposed(
        self,
        left: Mapping[str, Any],
        right: Mapping[str, Any],
    ) -> None:
        """Cover the optional deterministic exhaustion fields as they land."""

        field = "admissible_configuration_space_exhausted"
        if field not in left and field not in right:
            return
        self.assertIn(field, left)
        self.assertIn(field, right)
        self.assertEqual(left[field], right[field])
        self.assertEqual(
            left.get("exhaustion_probe_sha256"),
            right.get("exhaustion_probe_sha256"),
        )
        probe = left.get("exhaustion_probe_sha256")
        if probe is not None:
            self.assertIsInstance(probe, str)
            self.assertEqual(len(probe), 64)

    def test_program_and_prefix_are_exact_18_plus_9_plus_9(self) -> None:
        verified = verify_null_controller_program(self.program, policy=self.policy)
        self.assertEqual(len(verified["coverage_configurations"]), 18)
        self.assertEqual(
            [value["successor_count"] for value in verified["adaptive_prefix"]],
            [9, 9],
        )

        steps = self.proposal_ledger(36)
        records = records_from_steps(steps)
        self.assertEqual(
            [value["trial_id"] for value in records[:18]],
            [f"coverage_{ordinal:03d}" for ordinal in range(1, 19)],
        )
        self.assertEqual(
            [value["trial_id"] for value in records[18:27]],
            [f"adaptive_c1_{ordinal:03d}" for ordinal in range(1, 10)],
        )
        self.assertEqual(
            [value["trial_id"] for value in records[27:36]],
            [f"adaptive_c2_{ordinal:03d}" for ordinal in range(1, 10)],
        )
        self.assertEqual(
            [value["slot_kind"] for value in records],
            ["coverage"] * 18 + ["adaptive_challenger"] * 18,
        )

    def test_first_successor_of_each_cycle_matches_shared_successor_engine(self) -> None:
        coverage_steps = self.proposal_ledger(18)
        coverage_records = records_from_steps(coverage_steps)
        expected_cycle1, expected_lineage1 = _successors(
            policy=self.policy,
            cycle=1,
            ranked_parents=sorted(coverage_records, key=_ranking_key),
            seen_configurations=[value["configuration"] for value in coverage_records],
            count=9,
        )
        actual_cycle1 = propose_next_trial(
            program=self.program,
            policy=self.policy,
            prior_steps=coverage_steps,
        )
        self.assertEqual(actual_cycle1["configuration"], expected_cycle1[0])
        self.assertEqual(
            actual_cycle1["lineage"]["parent_configuration_sha256"],
            expected_lineage1[0]["parent_configuration_sha256"],
        )

        cycle1_steps = self.proposal_ledger(27)
        all_records = records_from_steps(cycle1_steps)
        expected_cycle2, expected_lineage2 = _successors(
            policy=self.policy,
            cycle=2,
            ranked_parents=sorted(all_records[18:27], key=_ranking_key),
            seen_configurations=[value["configuration"] for value in all_records],
            count=9,
        )
        actual_cycle2 = propose_next_trial(
            program=self.program,
            policy=self.policy,
            prior_steps=cycle1_steps,
        )
        self.assertEqual(actual_cycle2["configuration"], expected_cycle2[0])
        self.assertEqual(
            actual_cycle2["lineage"]["parent_configuration_sha256"],
            expected_lineage2[0]["parent_configuration_sha256"],
        )

    def test_adversarial_objective_ranking_changes_the_successor(self) -> None:
        first_wins = self.proposal_ledger(
            18,
            values={
                ordinal: (10.0 if ordinal == 1 else 0.0)
                for ordinal in range(1, 19)
            },
        )
        second_wins = self.proposal_ledger(
            18,
            values={
                ordinal: (10.0 if ordinal == 2 else 0.0)
                for ordinal in range(1, 19)
            },
        )
        first = propose_next_trial(
            program=self.program,
            policy=self.policy,
            prior_steps=first_wins,
        )
        second = propose_next_trial(
            program=self.program,
            policy=self.policy,
            prior_steps=second_wins,
        )
        self.assertEqual(first["lineage"]["parent_trial_id"], "coverage_001")
        self.assertEqual(second["lineage"]["parent_trial_id"], "coverage_002")
        self.assertNotEqual(
            first["lineage"]["parent_configuration_sha256"],
            second["lineage"]["parent_configuration_sha256"],
        )
        self.assertNotEqual(
            first["configuration_sha256"], second["configuration_sha256"]
        )

    def test_no_tail_improvement_stops_at_52(self) -> None:
        steps = self.arithmetic_ledger(52)
        status = stop_status(program=self.program, policy=self.policy, steps=steps)
        replay = stop_status(
            program=self.program,
            policy=self.policy,
            steps=copy.deepcopy(steps),
        )
        self.assertEqual(status, replay)
        self.assertEqual(status["valid_trial_count"], 52)
        self.assertEqual(status["last_promotion_eligible_improvement_ordinal"], 36)
        self.assertEqual(
            status["trailing_valid_trials_without_eligible_improvement"], 16
        )
        self.assertTrue(status["patience_actionable"])
        self.assertEqual(
            (status["action"], status["reason"]),
            ("stop", "patience_exhausted"),
        )
        self.assert_exhaustion_probe_is_deterministic_when_exposed(status, replay)

    def test_improvement_at_49_or_50_delays_patience_stop(self) -> None:
        for improvement, expected_stop in ((49, 65), (50, 66)):
            with self.subTest(improvement=improvement):
                at_52 = self.arithmetic_ledger(52, tail_improvements={improvement})
                status_52 = stop_status(
                    program=self.program,
                    policy=self.policy,
                    steps=at_52,
                )
                replay_52 = stop_status(
                    program=self.program,
                    policy=self.policy,
                    steps=copy.deepcopy(at_52),
                )
                self.assertEqual(status_52, replay_52)
                self.assertEqual(status_52["action"], "continue")
                self.assertFalse(status_52["patience_actionable"])
                self.assertEqual(
                    status_52["last_promotion_eligible_improvement_ordinal"],
                    improvement,
                )
                self.assert_exhaustion_probe_is_deterministic_when_exposed(
                    status_52,
                    replay_52,
                )
                if "admissible_configuration_space_exhausted" in status_52:
                    self.assertFalse(
                        status_52["admissible_configuration_space_exhausted"]
                    )
                    self.assertIsInstance(
                        status_52["exhaustion_probe_sha256"],
                        str,
                    )

                delayed = self.arithmetic_ledger(
                    expected_stop,
                    tail_improvements={improvement},
                )
                delayed_status = stop_status(
                    program=self.program,
                    policy=self.policy,
                    steps=delayed,
                )
                self.assertEqual(
                    delayed_status["trailing_valid_trials_without_eligible_improvement"],
                    16,
                )
                self.assertTrue(delayed_status["patience_actionable"])
                self.assertEqual(delayed_status["action"], "stop")

    def test_fraction_and_every_countable_id_gate_use_exact_trial_roles(self) -> None:
        steps = self.arithmetic_ledger(52)
        at_45 = stop_status(
            program=self.program,
            policy=self.policy,
            steps=steps[:45],
        )
        self.assertEqual(at_45["counted_post_coverage_falsifier_count"], 9)
        self.assertEqual(at_45["post_coverage_valid_trial_count"], 27)
        self.assertFalse(at_45["falsifier_fraction_gate"])
        self.assertTrue(at_45["all_countable_falsifier_ids_gate"])

        at_48 = stop_status(
            program=self.program,
            policy=self.policy,
            steps=steps[:48],
        )
        self.assertEqual(at_48["counted_post_coverage_falsifier_count"], 12)
        self.assertEqual(at_48["post_coverage_valid_trial_count"], 30)
        self.assertTrue(at_48["falsifier_fraction_gate"])
        self.assertTrue(at_48["all_countable_falsifier_ids_gate"])
        self.assertEqual(
            set(at_48["completed_countable_falsifier_ids"]),
            set(self.countable),
        )
        self.assertFalse(
            set(self.non_trial) & set(at_48["completed_countable_falsifier_ids"])
        )

        at_51 = stop_status(
            program=self.program,
            policy=self.policy,
            steps=steps[:51],
        )
        self.assertEqual(at_51["counted_post_coverage_falsifier_count"], 13)
        self.assertEqual(at_51["post_coverage_valid_trial_count"], 33)
        self.assertFalse(at_51["falsifier_fraction_gate"])

        at_52 = stop_status(
            program=self.program,
            policy=self.policy,
            steps=steps,
        )
        self.assertEqual(at_52["counted_post_coverage_falsifier_count"], 14)
        self.assertEqual(at_52["post_coverage_valid_trial_count"], 34)
        self.assertTrue(at_52["falsifier_fraction_gate"])
        self.assertTrue(at_52["all_countable_falsifier_ids_gate"])

    def test_program_and_proposal_replay_are_deterministic_and_tamper_fails(self) -> None:
        replay_program = freeze_null_controller_program(
            policy=self.policy,
            coverage_configurations=self.coverage,
            post36_tail_plan=self.plan,
            countable_falsifier_ids=self.countable,
            non_trial_required_falsifier_ids=self.non_trial,
        )
        self.assertEqual(self.program, replay_program)

        ledger = self.proposal_ledger(18)
        first = propose_next_trial(
            program=self.program,
            policy=self.policy,
            prior_steps=ledger,
        )
        replay = propose_next_trial(
            program=copy.deepcopy(self.program),
            policy=self.policy,
            prior_steps=copy.deepcopy(ledger),
        )
        self.assertEqual(first, replay)

        tampered = copy.deepcopy(self.program)
        tampered["observed_realized_schedule_embedded"] = True
        tampered["program_sha256"] = digest_object(
            {key: value for key, value in tampered.items() if key != "program_sha256"}
        )
        with self.assertRaisesRegex(NullControllerError, "observed decisions"):
            verify_null_controller_program(tampered, policy=self.policy)


if __name__ == "__main__":
    unittest.main()
