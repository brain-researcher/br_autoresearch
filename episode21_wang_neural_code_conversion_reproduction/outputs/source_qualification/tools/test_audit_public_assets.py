#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


RUNNER = Path(__file__).with_name("audit_public_assets.py")
SPEC = importlib.util.spec_from_file_location("ep21_public_asset_audit", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PublicAssetAuditTest(unittest.TestCase):
    def test_reconstruction_models_are_head_probe_eligible(self) -> None:
        result = MODULE.classify_asset(
            "reconstruction/data/files.json",
            "recon_demo",
            "VGG_ILSVRC_19_layers.zip",
        )
        self.assertTrue(result["head_probe_allowed"])

    def test_decoded_feature_fixture_is_excluded(self) -> None:
        result = MODULE.classify_asset(
            "reconstruction/data/files.json",
            "recon_demo",
            "decoded_features.zip",
        )
        self.assertFalse(result["head_probe_allowed"])
        self.assertEqual(result["ep21_use"], "forbidden_for_lock_and_current_episode")

    def test_empirical_and_source_derived_assets_are_not_probed(self) -> None:
        for group in (
            "fmri_training",
            "fmri_test",
            "stimulus_feature",
            "pre-trained-converters",
            "pre-trained-decoders",
            "test_image-true_features",
        ):
            result = MODULE.classify_asset("data/files.json", group, "asset.zip")
            self.assertFalse(result["head_probe_allowed"], group)

    def test_url_checksum_parts_preserve_cardinality(self) -> None:
        record = {
            "name": "split.zip",
            "url": ["https://example.test/1", "https://example.test/2"],
            "md5sum": ["a", "b"],
        }
        self.assertEqual(
            MODULE.url_checksum_pairs(record),
            [
                {"url": "https://example.test/1", "md5": "a"},
                {"url": "https://example.test/2", "md5": "b"},
            ],
        )

    def test_url_checksum_mismatch_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.url_checksum_pairs(
                {
                    "name": "bad.zip",
                    "url": ["https://example.test/1", "https://example.test/2"],
                    "md5sum": ["a"],
                }
            )

    def test_figshare_file_id(self) -> None:
        self.assertEqual(
            MODULE.figshare_file_id(
                "https://figshare.com/ndownloader/files/38225868"
            ),
            38225868,
        )

    def test_non_figshare_url_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.figshare_file_id("https://example.test/no-id")


if __name__ == "__main__":
    unittest.main()
