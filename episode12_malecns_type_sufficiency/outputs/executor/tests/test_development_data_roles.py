from __future__ import annotations

import copy
import unittest

from ep12_executor.development_data import validated_role_types
from ep12_executor.policy import digest_object


class DevelopmentRoleManifestTests(unittest.TestCase):
    def _manifest(self) -> dict:
        manifest = {
            "assignment_unit": "whole_provider_type",
            "eligible_type_count": 5,
            "development_type_count": 4,
            "final_type_count": 1,
            "rows": [
                {"provider_type": name, "role": "development"}
                for name in ("d1", "d2", "d3", "d4")
            ]
            + [{"provider_type": "f1", "role": "final"}],
        }
        manifest["rows_hash"] = digest_object(manifest["rows"])
        return manifest

    def test_valid_whole_type_counts_are_returned(self) -> None:
        development, final = validated_role_types(self._manifest())
        self.assertEqual(development, frozenset({"d1", "d2", "d3", "d4"}))
        self.assertEqual(final, frozenset({"f1"}))

    def test_duplicate_type_is_rejected_before_roles_are_collapsed(self) -> None:
        manifest = self._manifest()
        manifest["rows"].append({"provider_type": "f1", "role": "development"})
        manifest["eligible_type_count"] += 1
        manifest["development_type_count"] += 1
        manifest["rows_hash"] = digest_object(manifest["rows"])
        with self.assertRaisesRegex(ValueError, "duplicate provider type"):
            validated_role_types(manifest)

    def test_unknown_role_is_rejected(self) -> None:
        manifest = self._manifest()
        manifest["rows"][0]["role"] = "holdoutish"
        manifest["rows_hash"] = digest_object(manifest["rows"])
        with self.assertRaisesRegex(ValueError, "unknown role"):
            validated_role_types(manifest)

    def test_rows_must_match_the_frozen_role_identity(self) -> None:
        manifest = self._manifest()
        manifest["rows"][0]["provider_type"] = "changed"
        with self.assertRaisesRegex(ValueError, "frozen row identity"):
            validated_role_types(manifest)

    def test_declared_counts_must_match_rows(self) -> None:
        for key in (
            "eligible_type_count",
            "development_type_count",
            "final_type_count",
        ):
            with self.subTest(key=key):
                manifest = copy.deepcopy(self._manifest())
                manifest[key] += 1
                with self.assertRaisesRegex(ValueError, key):
                    validated_role_types(manifest)


if __name__ == "__main__":
    unittest.main()
