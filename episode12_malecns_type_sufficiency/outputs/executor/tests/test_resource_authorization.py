from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ep12_executor.policy import EpisodePolicy
from ep12_executor.resource_authorization import (
    AUTHORIZATION_SCOPE,
    AUTHORIZATION_SOURCE,
    ResourceAuthorizationError,
    verify_resource_authorization,
)
from ep12_executor.runtime import EPISODE_ROOT


class ResourceAuthorizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")

    def _authorization(self) -> dict:
        return {
            "scope": AUTHORIZATION_SCOPE,
            "authorization_source": AUTHORIZATION_SOURCE,
            "measured_planning_core_hours": 2750.0,
            "allow_full_search_null_dispatch": True,
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


if __name__ == "__main__":
    unittest.main()
