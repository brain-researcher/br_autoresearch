from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ep12_executor.cli import build_parser
from ep12_executor.policy import EpisodePolicy
from ep12_executor.resource_authorization import (
    AUTHORIZATION_SCOPE,
    AUTHORIZATION_SOURCE,
    ResourceAuthorizationError,
    assess_projected_core_hours,
    verify_resource_authorization,
)
from ep12_executor.runtime import EPISODE_ROOT


class ResourceAuthorizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")

    def test_cli_module_entrypoint_is_executable(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "ep12_executor.cli", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("qualify-methods", result.stdout)

    def _authorization(self) -> dict:
        return {
            "scope": AUTHORIZATION_SCOPE,
            "authorization_source": AUTHORIZATION_SOURCE,
            "measured_planning_core_hours": 2750.0,
            "cpu_budget_stop_disabled_until_minimum_full_search_complete": True,
            "new_hard_cpu_ceiling": None,
            "unchanged_limits": {
                "minimum_valid_trials": self.policy.minimum_valid_trials,
                "maximum_valid_trials": self.policy.maximum_valid_trials,
                "cpu_cores_per_trial_max": self.policy.cpu_cores_per_trial_maximum,
                "memory_gib_per_trial_max": self.policy.memory_gib_per_trial_maximum,
                "wall_hours_per_trial_max": self.policy.wall_hours_per_trial_maximum,
                "wall_clock_hour_ceiling": self.policy.raw["budgets"][
                    "wall_clock_hour_ceiling"
                ],
                "gpu_hour_ceiling": self.policy.gpu_hour_ceiling,
            },
            "final_connectivity_access_authorized": False,
        }

    def _verify(self, value: dict) -> dict:
        with tempfile.TemporaryDirectory(prefix="ep12-resource-auth-") as directory:
            path = Path(directory) / "authorization.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            return verify_resource_authorization(
                authorization_path=path,
                policy=self.policy,
            )

    def test_authorization_source_must_be_explicitly_present(self) -> None:
        value = self._authorization()
        value.pop("authorization_source")
        with self.assertRaisesRegex(ResourceAuthorizationError, "explicit user instruction"):
            self._verify(value)

    def test_explicit_authorization_source_is_retained(self) -> None:
        result = self._verify(self._authorization())
        self.assertEqual(result["authorization_source"], AUTHORIZATION_SOURCE)
        self.assertFalse(result["final_connectivity_access_authorized"])

    def test_symlink_authorization_is_rejected_before_resolution(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-resource-auth-link-") as directory:
            root = Path(directory)
            target = root / "authorization.json"
            target.write_text(json.dumps(self._authorization()), encoding="utf-8")
            link = root / "authorization-link.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(ResourceAuthorizationError, "symbolic link"):
                verify_resource_authorization(
                    authorization_path=link,
                    policy=self.policy,
                )

    def test_projection_uses_explicit_authorization_not_old_policy_ceiling(self) -> None:
        authorization = self._verify(self._authorization())
        assessment = assess_projected_core_hours(
            projected_core_hours=2000.0,
            authorization=authorization,
        )
        self.assertGreater(2000.0, self.policy.cpu_core_hour_ceiling)
        self.assertTrue(assessment["within_explicit_authorization"])
        self.assertFalse(assessment["original_policy_cpu_ceiling_enforced"])

    def test_projection_above_planning_value_is_recorded_without_a_new_ceiling(self) -> None:
        authorization = self._verify(self._authorization())
        assessment = assess_projected_core_hours(
            projected_core_hours=2750.1,
            authorization=authorization,
        )
        self.assertTrue(assessment["within_explicit_authorization"])
        self.assertTrue(assessment["projection_is_recorded_not_a_hard_gate"])
        self.assertIsNone(assessment["new_hard_cpu_ceiling"])

    def test_non_null_new_hard_ceiling_is_rejected(self) -> None:
        value = self._authorization()
        value["new_hard_cpu_ceiling"] = 2750.0
        with self.assertRaisesRegex(ResourceAuthorizationError, "no new hard CPU ceiling"):
            self._verify(value)

    def test_overall_wall_clock_limit_may_not_change(self) -> None:
        value = self._authorization()
        value["unchanged_limits"]["wall_clock_hour_ceiling"] += 1
        with self.assertRaisesRegex(ResourceAuthorizationError, "bounded-search"):
            self._verify(value)

    def test_runtime_maximum_projection_is_recorded_not_capped(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-resource-runtime-") as directory:
            root = Path(directory)
            authorization_path = root / "authorization.json"
            authorization_path.write_text(
                json.dumps(self._authorization()), encoding="utf-8"
            )
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "resource_plan.json").write_text(
                json.dumps(
                    {"projected_maximum_program_allocated_core_hours": 4000.0}
                ),
                encoding="utf-8",
            )
            result = verify_resource_authorization(
                authorization_path=authorization_path,
                policy=self.policy,
                runtime_bundle=bundle,
            )
        self.assertEqual(
            result["runtime_projected_maximum_program_allocated_core_hours"],
            4000.0,
        )
        self.assertIsNone(result["new_hard_cpu_ceiling"])

    def test_qualification_cli_defaults_to_the_existing_authorization(self) -> None:
        arguments = build_parser().parse_args(
            [
                "qualify-methods",
                "--contract",
                str(
                    EPISODE_ROOT
                    / "outputs/executor/SCIENTIFIC_CONTRACT_POST36_V3.yaml"
                ),
                "--qualification-spec",
                str(
                    EPISODE_ROOT
                    / "outputs/executor/SYNTHETIC_QUALIFICATION_SPEC_POST36_V3.yaml"
                ),
                "--oracle-aggregate",
                str(
                    EPISODE_ROOT
                    / "outputs/scientific_qualification/stochastic_oracle/aggregate.json"
                ),
            ]
        )
        self.assertEqual(
            arguments.resource_authorization,
            EPISODE_ROOT / "outputs/executor/DEVELOPMENT_BUDGET_OVERRIDE.yaml",
        )


if __name__ == "__main__":
    unittest.main()
