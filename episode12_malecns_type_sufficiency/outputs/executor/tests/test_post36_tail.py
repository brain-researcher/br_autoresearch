from __future__ import annotations

import unittest
from datetime import datetime, timezone

from ep12_executor.policy import EpisodePolicy
from ep12_executor.post36_tail import (
    assess_observed_search_wall_budget,
    capability_report,
    observed_search_wall_budget,
    production_control_capabilities,
    promotion_decision,
)
from ep12_executor.runtime import EPISODE_ROOT

import observed_tail_plan


def _record(trial_id: str, score: float, configuration_sha256: str) -> dict:
    return {
        "trial_id": trial_id,
        "configuration_sha256": configuration_sha256,
        "objective": {
            "bilateral_floor": score,
            "bilateral": score,
            "direction_disagreement": 0.0,
            "types_both_above_margin": 2,
        },
    }


class Post36TailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")

    def test_controls_cover_every_registered_countable_falsifier(self) -> None:
        report = capability_report(
            observed_tail_plan.build_plan(), production_control_capabilities()
        )
        self.assertTrue(report["whole_plan_executable"])
        self.assertEqual(report["unsupported_falsifiers"], [])

    def test_control_cannot_replace_the_primary_incumbent(self) -> None:
        incumbent = _record("incumbent", 0.01, "a" * 64)
        stronger_control = _record("control", 1.0, "b" * 64)
        decision = promotion_decision(
            incumbent=incumbent,
            candidate=stronger_control,
            promotion_eligible=False,
        )
        self.assertFalse(decision["improved"])
        self.assertEqual(decision["incumbent_trial_id"], "incumbent")

    def test_wall_deadline_gate_is_outcome_blind(self) -> None:
        budget = observed_search_wall_budget(self.policy)
        early = assess_observed_search_wall_budget(
            policy=self.policy,
            wall_budget=budget,
            now_utc=datetime(2026, 9, 26, 6, tzinfo=timezone.utc),
        )
        late = assess_observed_search_wall_budget(
            policy=self.policy,
            wall_budget=budget,
            now_utc=datetime(2026, 10, 1, 5, tzinfo=timezone.utc),
        )
        self.assertTrue(early["can_start_next_tail_slot"])
        self.assertFalse(late["can_start_next_tail_slot"])
        self.assertFalse(late["outcome_values_read_for_decision"])


if __name__ == "__main__":
    unittest.main()
