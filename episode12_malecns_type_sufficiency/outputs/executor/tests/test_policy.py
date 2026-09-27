from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ep12_executor.adaptive_search import coverage_configurations, coverage_proof
from ep12_executor.policy import ConfigurationError, EpisodePolicy, PolicyError
from ep12_executor.runtime import EPISODE_ROOT, SCRATCH_ROOT


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
        cls.policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")

    def test_pinned_episode_contract_is_loaded_without_external_yaml(self) -> None:
        self.assertEqual(self.policy.episode_id, "ep12")
        self.assertEqual(self.policy.meaningful_margin, 0.02)
        self.assertEqual(self.policy.minimum_valid_trials, 36)
        self.assertEqual(self.policy.maximum_valid_trials, 96)
        self.assertEqual(self.policy.coverage_minimum, 18)
        self.assertEqual(self.policy.patience_valid_trials, 16)
        self.assertEqual(self.policy.null_replicates, 99)
        self.assertEqual(self.policy.maximum_open_count, 1)
        self.assertFalse(self.policy.contract_summary()["real_data_enabled"])

    def test_real_development_covering_array_has_all_required_pairs(self) -> None:
        rows = coverage_configurations(self.policy)
        self.assertEqual(len(rows), 18)
        self.assertTrue(coverage_proof(self.policy, rows)["all_required_pairs_covered"])
        self.assertEqual({row["data_mode"] for row in rows}, {"development"})
        self.assertEqual(
            {row["stage"] for row in rows},
            {"coverage"},
        )
        self.assertEqual(
            {row["regularization"] for row in rows},
            set(self.policy.grammar_choices["regularization"]),
        )

    def test_real_mode_and_forbidden_annotation_are_rejected(self) -> None:
        config = coverage_configurations(self.policy)[0]
        real_config = dict(config)
        real_config["data_mode"] = "real"
        with self.assertRaises(ConfigurationError):
            self.policy.validate_config(real_config)
        forbidden = dict(config)
        forbidden["initialization"] = "provider_group"
        with self.assertRaises(ConfigurationError):
            self.policy.validate_config(forbidden)
        extra = dict(config)
        extra["new_unregistered_estimator_knob"] = 1
        with self.assertRaises(ConfigurationError):
            self.policy.validate_config(extra)

    def test_yaml_subset_rejects_anchors(self) -> None:
        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as temporary:
            policy_path = Path(temporary) / "bad.yaml"
            policy_path.write_text("schema_version: &anchor bad\n", encoding="utf-8")
            with self.assertRaises(Exception):
                EpisodePolicy.load(policy_path)

    def test_more_than_one_final_opening_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as temporary:
            policy_path = Path(temporary) / "unsafe.yaml"
            text = (EPISODE_ROOT / "SEARCH_POLICY.yaml").read_text(encoding="utf-8")
            policy_path.write_text(
                text.replace("  maximum_open_count: 1\n", "  maximum_open_count: 2\n"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PolicyError, "exactly one final opening"):
                EpisodePolicy.load(policy_path)


if __name__ == "__main__":
    unittest.main()
