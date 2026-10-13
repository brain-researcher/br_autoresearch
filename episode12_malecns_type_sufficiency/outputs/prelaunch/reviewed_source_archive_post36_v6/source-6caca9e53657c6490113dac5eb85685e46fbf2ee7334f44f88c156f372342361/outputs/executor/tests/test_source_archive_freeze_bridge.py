from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from ep12_executor import source_archive_freeze_bridge as bridge
from ep12_executor.policy import canonical_json
from ep12_executor.source_archive import publish_source_archive
from ep12_executor.source_freeze import (
    SOURCE_FREEZE_RELATIVE,
    verify_source_freeze_manifest,
)


class SourceArchiveFreezeBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.reviewed = self.root / "reviewed"
        self.live = self.root / "live"
        self.archive_parent = self.root / "archive"
        self.archive_parent.mkdir(mode=0o700)
        executor = self.reviewed / "outputs/executor/ep12_executor"
        executor.mkdir(parents=True)
        (self.reviewed / "SEARCH_POLICY.yaml").write_text(
            "schema: synthetic-test\n", encoding="utf-8"
        )
        (executor / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
        wrapper = self.reviewed / "outputs/executor/slurm/test_executor.sbatch"
        wrapper.parent.mkdir(parents=True)
        wrapper.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        wrapper.chmod(0o750)
        shutil.copytree(self.reviewed, self.live)
        self.publication = publish_source_archive(
            source_root=self.reviewed, archive_parent=self.archive_parent
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _bridge(self) -> dict[str, object]:
        value = self.publication
        return bridge.bridge_source_archive_to_live_freeze(
            archive_root=Path(value["archived_root"]),
            archive_receipt_file_sha256=value["archive_receipt_file_sha256"],
            archive_receipt_sha256=value["archive_receipt_sha256"],
            archive_manifest_file_sha256=value["manifest_file_sha256"],
            archive_manifest_sha256=value["manifest_sha256"],
            archive_tree_sha256=value["tree_sha256"],
            live_episode_root=self.live,
        )

    def test_exact_bridge_is_canonical_and_retry_is_identical(self) -> None:
        first = self._bridge()
        self.assertEqual(first["publication_status"], "created")
        self.assertTrue(first["archive_and_live_file_maps_identical"])
        self.assertFalse(first["real_connectivity_accessed"])
        self.assertFalse(first["final_connectivity_access_authorized"])

        receipt_path = self.live / bridge.BRIDGE_RELATIVE
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(
            receipt_path.read_bytes(),
            (canonical_json(receipt) + "\n").encode("utf-8"),
        )
        freeze = first["source_freeze_identity"]
        self.assertEqual(
            verify_source_freeze_manifest(
                Path(freeze["manifest_path"]),
                episode_root=self.live,
                expected_file_sha256=freeze["manifest_file_sha256"],
            )["manifest_sha256"],
            freeze["manifest_sha256"],
        )

        second = self._bridge()
        self.assertEqual(second["publication_status"], "reused_identical")
        for key in first:
            if key != "publication_status":
                self.assertEqual(first[key], second[key])

    def test_live_difference_fails_before_freeze_or_receipt(self) -> None:
        (self.live / "outputs/executor/ep12_executor/model.py").write_text(
            "VALUE = 2\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            bridge.SourceArchiveFreezeBridgeError,
            "live executable surface differs",
        ):
            self._bridge()
        self.assertFalse((self.live / SOURCE_FREEZE_RELATIVE).exists())
        self.assertFalse((self.live / bridge.BRIDGE_RELATIVE).exists())

    def test_later_live_mutation_fails_closed_and_preserves_receipt(self) -> None:
        first = self._bridge()
        receipt_before = (self.live / bridge.BRIDGE_RELATIVE).read_bytes()
        (self.live / "outputs/executor/ep12_executor/model.py").write_text(
            "VALUE = 999\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            bridge.SourceArchiveFreezeBridgeError,
            "live executable surface differs",
        ):
            self._bridge()
        self.assertEqual(
            (self.live / bridge.BRIDGE_RELATIVE).read_bytes(), receipt_before
        )
        self.assertEqual(
            json.loads(receipt_before)["receipt_sha256"], first["receipt_sha256"]
        )

    def test_bad_archive_identity_fails_before_live_freeze(self) -> None:
        value = self.publication
        with self.assertRaisesRegex(
            bridge.SourceArchiveFreezeBridgeError,
            "archive verification failed",
        ):
            bridge.bridge_source_archive_to_live_freeze(
                archive_root=Path(value["archived_root"]),
                archive_receipt_file_sha256=value["archive_receipt_file_sha256"],
                archive_receipt_sha256=value["archive_receipt_sha256"],
                archive_manifest_file_sha256=value["manifest_file_sha256"],
                archive_manifest_sha256=value["manifest_sha256"],
                archive_tree_sha256="0" * 64,
                live_episode_root=self.live,
            )
        self.assertFalse((self.live / SOURCE_FREEZE_RELATIVE).exists())

    def test_production_mode_rejects_noncanonical_live_root(self) -> None:
        value = self.publication
        with self.assertRaisesRegex(
            bridge.SourceArchiveFreezeBridgeError, "live root is not canonical"
        ):
            bridge.bridge_source_archive_to_live_freeze(
                archive_root=Path(value["archived_root"]),
                archive_receipt_file_sha256=value["archive_receipt_file_sha256"],
                archive_receipt_sha256=value["archive_receipt_sha256"],
                archive_manifest_file_sha256=value["manifest_file_sha256"],
                archive_manifest_sha256=value["manifest_sha256"],
                archive_tree_sha256=value["tree_sha256"],
                live_episode_root=self.live,
                require_canonical_live_root=True,
            )
        self.assertFalse((self.live / SOURCE_FREEZE_RELATIVE).exists())

    def test_live_mutation_between_manifest_and_final_check_has_no_receipt(self) -> None:
        real_create = bridge.create_source_freeze_manifest

        def create_then_mutate(*args: object, **kwargs: object) -> dict[str, object]:
            result = real_create(*args, **kwargs)
            (self.live / "outputs/executor/ep12_executor/model.py").write_text(
                "VALUE = 3\n", encoding="utf-8"
            )
            return result

        with mock.patch.object(
            bridge, "create_source_freeze_manifest", side_effect=create_then_mutate
        ):
            with self.assertRaisesRegex(Exception, "surface changed"):
                self._bridge()
        self.assertFalse((self.live / bridge.BRIDGE_RELATIVE).exists())

    def test_tampered_existing_receipt_is_not_replaced(self) -> None:
        self._bridge()
        receipt_path = self.live / bridge.BRIDGE_RELATIVE
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
        value["file_count"] += 1
        tampered = (canonical_json(value) + "\n").encode("utf-8")
        receipt_path.write_bytes(tampered)
        with self.assertRaisesRegex(
            bridge.SourceArchiveFreezeBridgeError, "self-hash mismatch"
        ):
            self._bridge()
        self.assertEqual(receipt_path.read_bytes(), tampered)


if __name__ == "__main__":
    unittest.main()
