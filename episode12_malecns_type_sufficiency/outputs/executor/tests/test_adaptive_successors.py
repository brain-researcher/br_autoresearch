from __future__ import annotations

import unittest

from ep12_executor.adaptive_search import coverage_configurations
from ep12_executor.adaptive_successors import (
    _ranking_key,
    _scientific_signature,
    _successors,
)
from ep12_executor.policy import EpisodePolicy, digest_object
from ep12_executor.runtime import EPISODE_ROOT


def record(configuration: dict[str, object], value: float) -> dict[str, object]:
    return {
        "trial_id": configuration["trial_id"],
        "configuration": configuration,
        "configuration_sha256": digest_object(configuration),
        "objective": {
            "left_to_right": value,
            "right_to_left": value - 0.001,
            "bilateral": value - 0.0005,
            "bilateral_floor": value - 0.001,
            "direction_disagreement": 0.001,
            "types_both_above_margin": 0,
        },
    }


class AdaptiveSuccessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")

    def test_two_cycles_are_deterministic_disjoint_one_operator_neighbors(self) -> None:
        coverage = coverage_configurations(self.policy)
        coverage_records = [
            record(configuration, float(index) / 1000.0)
            for index, configuration in enumerate(coverage)
        ]
        ranked_coverage = sorted(coverage_records, key=_ranking_key)
        cycle1, lineage1 = _successors(
            policy=self.policy,
            cycle=1,
            ranked_parents=ranked_coverage,
            seen_configurations=coverage,
            count=9,
        )
        self.assertEqual(len(cycle1), 9)
        self.assertEqual(len({_scientific_signature(value) for value in cycle1}), 9)
        self.assertTrue(all(value["stage"] == "adaptive" for value in cycle1))
        self.assertEqual({value["parent_trial_id"] for value in lineage1}, {"coverage_018"})

        cycle1_records = [
            record(configuration, float(index) / 100.0)
            for index, configuration in enumerate(cycle1)
        ]
        cycle2, lineage2 = _successors(
            policy=self.policy,
            cycle=2,
            ranked_parents=sorted(cycle1_records, key=_ranking_key),
            seen_configurations=coverage + cycle1,
            count=9,
        )
        signatures1 = {_scientific_signature(value) for value in cycle1}
        signatures2 = {_scientific_signature(value) for value in cycle2}
        self.assertEqual(len(cycle2), 9)
        self.assertFalse(signatures1 & signatures2)
        self.assertFalse(
            {_scientific_signature(value) for value in coverage} & signatures2
        )
        self.assertEqual({value["parent_trial_id"] for value in lineage2}, {"adaptive_c1_009"})

    def test_bilateral_floor_precedes_mean_in_parent_ranking(self) -> None:
        coverage = coverage_configurations(self.policy)
        first = record(coverage[0], 0.01)
        second = record(coverage[1], 0.02)
        second["objective"]["right_to_left"] = -0.5
        second["objective"]["bilateral_floor"] = -0.5
        self.assertEqual(sorted([second, first], key=_ranking_key)[0]["trial_id"], "coverage_001")


if __name__ == "__main__":
    unittest.main()
