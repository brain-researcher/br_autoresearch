from __future__ import annotations

import itertools
import tempfile
import unittest
from pathlib import Path

from ep12_executor.controller import balanced_coverage_configs
from ep12_executor.policy import ConfigurationError, EpisodePolicy
from ep12_executor.synthetic import EPISODE_ROOT, SCRATCH_ROOT


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

    def test_synthetic_covering_array_has_all_required_pairs(self) -> None:
        rows = balanced_coverage_configs(self.policy)
        self.assertEqual(len(rows), 18)
        factors = (
            "partner_vocabulary",
            "representation",
            "covariance",
            "component_count",
        )
        for left_index, left in enumerate(factors):
            for right in factors[left_index + 1 :]:
                observed = {(row[left], row[right]) for row in rows}
                expected = set(
                    itertools.product(
                        self.policy.grammar_choices[left],
                        self.policy.grammar_choices[right],
                    )
                )
                self.assertEqual(observed, expected)

    def test_real_mode_and_forbidden_annotation_are_rejected(self) -> None:
        config = balanced_coverage_configs(self.policy)[0]
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


if __name__ == "__main__":
    unittest.main()
