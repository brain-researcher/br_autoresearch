#!/usr/bin/env python3
"""Focused tests for metadata-only manifest reconciliation."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("audit_manifest_metadata.py")
SPEC = importlib.util.spec_from_file_location("audit_manifest_metadata", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ManifestMetadataAuditTests(unittest.TestCase):
    def test_main_and_reconstruction_manifests_are_unambiguous(self) -> None:
        names = [
            "root/data/files.json",
            "root/reconstruction/data/files.json",
        ]
        self.assertEqual(
            MODULE.select_manifest_name(names, "data/files.json"),
            "root/data/files.json",
        )
        self.assertEqual(
            MODULE.select_manifest_name(names, "reconstruction/data/files.json"),
            "root/reconstruction/data/files.json",
        )

    def test_url_checksum_pairs_preserve_parts(self) -> None:
        record = {
            "url": [
                "https://figshare.com/ndownloader/files/1",
                "https://figshare.com/ndownloader/files/2",
            ],
            "md5sum": ["a" * 32, "b" * 32],
        }
        self.assertEqual(len(MODULE.url_checksum_pairs(record)), 2)

    def test_url_checksum_cardinality_and_domain_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.url_checksum_pairs(
                {
                    "url": ["https://figshare.com/ndownloader/files/1"],
                    "md5sum": ["a" * 32, "b" * 32],
                }
            )
        with self.assertRaises(ValueError):
            MODULE.url_checksum_pairs(
                {"url": "https://example.org/files/1", "md5sum": "a" * 32}
            )

    def test_reconcile_requires_both_supplier_md5_fields(self) -> None:
        part = {
            "source_manifest": "data/files.json",
            "group": "fixture",
            "logical_asset": "fixture.zip",
            "logical_key": "fixture",
            "part_index": 0,
            "figshare_file_id": 99,
            "manifest_url": "https://figshare.com/ndownloader/files/99",
            "manifest_md5": "a" * 32,
        }
        good_file = {
            "id": 99,
            "name": "fixture.zip",
            "size": 1,
            "supplied_md5": "a" * 32,
            "computed_md5": "a" * 32,
        }
        report = MODULE.reconcile(
            [part],
            1,
            {999: {"title": "fixture", "license": {}, "files": [good_file]}},
        )
        self.assertEqual(report["resolved_file_id_count"], 1)
        bad_file = {**good_file, "computed_md5": "b" * 32}
        with self.assertRaises(ValueError):
            MODULE.reconcile(
                [part],
                1,
                {999: {"title": "fixture", "license": {}, "files": [bad_file]}},
            )

    def test_unresolved_file_id_fails_closed(self) -> None:
        part = {
            "source_manifest": "data/files.json",
            "group": "fixture",
            "logical_asset": "fixture.zip",
            "logical_key": "fixture",
            "part_index": 0,
            "figshare_file_id": 99,
            "manifest_url": "https://figshare.com/ndownloader/files/99",
            "manifest_md5": "a" * 32,
        }
        with self.assertRaises(ValueError):
            MODULE.reconcile([part], 1, {999: {"files": []}})


if __name__ == "__main__":
    unittest.main()
