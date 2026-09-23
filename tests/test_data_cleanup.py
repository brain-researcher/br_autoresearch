import contextlib
import copy
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
import pathlib
import stat
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin" / "data-cleanup"
LOADER = importlib.machinery.SourceFileLoader("data_cleanup", str(SCRIPT))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
data_cleanup = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(data_cleanup)


class DataCleanupTests(unittest.TestCase):
    def test_catalog_is_exact_and_non_authorizing(self):
        catalog, raw, digest = data_cleanup.load_catalog()
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
        self.assertEqual(
            [item["logical_id"] for item in catalog["candidates"]],
            [
                "scratch_things_eeg1_ingest",
                "scratch_ep05_payload",
                "personal_link_001201_full",
                "personal_link_001201_sentinels",
                "personal_epic_ptir_coating",
            ],
        )
        self.assertFalse(catalog["authority"]["quarantine_authorized"])
        self.assertFalse(catalog["authority"]["permanent_deletion_authorized"])

    def test_snapshot_does_not_follow_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp) / "source"
            outside = pathlib.Path(temp) / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "payload").write_bytes(b"abc")
            (outside / "secret").write_bytes(b"not counted")
            (root / "pointer").symlink_to(outside, target_is_directory=True)
            snapshot = data_cleanup.snapshot_tree(root)
            self.assertEqual(snapshot["regular_files"], 1)
            self.assertEqual(snapshot["symlinks"], 1)
            self.assertEqual(snapshot["logical_file_bytes"], 3)

    def test_build_receipt_is_read_only_and_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            source = pathlib.Path(temp) / "source"
            counterpart = pathlib.Path(temp) / "counterpart"
            source.mkdir()
            counterpart.mkdir()
            (source / "payload").write_bytes(b"abc")
            before = sorted(path.relative_to(source) for path in source.rglob("*"))
            observed = data_cleanup.snapshot_tree(source)
            candidate = {
                "logical_id": "fixture",
                "source_path": str(source),
                "source_snapshot": {
                    key: observed[key]
                    for key in data_cleanup.SNAPSHOT_MATCH_KEYS
                },
                "scientific_payload": {
                    "source_path": str(source),
                    "authoritative_path": str(counterpart),
                },
                "evidence_paths": [],
                "required_equivalence": "fixture",
                "quarantine_root": next(iter(data_cleanup.QUARANTINE_ROOTS)),
            }
            receipt = data_cleanup.build_receipt(
                candidate, "0" * 64, "fixture-2"
            )
            after = sorted(path.relative_to(source) for path in source.rglob("*"))
            self.assertEqual(before, after)
            self.assertTrue(receipt["source_snapshot_matches_catalog"])
            self.assertFalse(receipt["authority"]["quarantine_authorized"])
            self.assertEqual(receipt["receipt_id"], "fixture-2")
            self.assertEqual(
                receipt["proposed_quarantine_destination"],
                str(
                    pathlib.Path(candidate["quarantine_root"])
                    / "fixture-2"
                    / "fixture"
                ),
            )

    def test_receipt_write_is_private_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            catalog_bytes = b'{"fixture":true}\n'
            path = data_cleanup.write_receipt(
                root, "fixture-1", {"status": "pending"}, catalog_bytes
            )
            self.assertEqual(json.loads(path.read_text()), {"status": "pending"})
            self.assertEqual(
                (path.parent / "candidate_catalog.json").read_bytes(), catalog_bytes
            )
            self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaises(SystemExit):
                data_cleanup.write_receipt(
                    root, "fixture-1", {"status": "pending"}, catalog_bytes
                )

    def test_parser_has_no_mutating_command(self):
        parser = data_cleanup.build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["quarantine", "fixture"])
            with self.assertRaises(SystemExit):
                parser.parse_args(["delete", "fixture"])

    def test_literal_path_rejects_broad_glob_and_parent(self):
        for value in ("relative", "/", "/tmp/../data", "/tmp/*"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                data_cleanup.validate_literal_path(value, "fixture")

    def test_root_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            real = pathlib.Path(temp) / "real"
            link = pathlib.Path(temp) / "link"
            real.mkdir()
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaises(SystemExit):
                data_cleanup.snapshot_tree(link)

    def test_metadata_digest_is_not_a_content_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp) / "source"
            root.mkdir()
            payload = root / "payload"
            payload.write_bytes(b"abc")
            before_stat = payload.stat()
            before = data_cleanup.snapshot_tree(root)
            payload.write_bytes(b"xyz")
            os.utime(
                payload,
                ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns),
            )
            after = data_cleanup.snapshot_tree(root)
            self.assertEqual(before["inventory_sha256"], after["inventory_sha256"])

    def test_catalog_authority_and_manifest_hash_are_enforced(self):
        catalog, _, _ = data_cleanup.load_catalog()
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "catalog.json"
            disabled = copy.deepcopy(catalog)
            disabled["authority"]["receipt_preparation_authorized"] = False
            path.write_text(json.dumps(disabled))
            with self.assertRaises(SystemExit):
                data_cleanup.load_catalog(path)

            stale = copy.deepcopy(catalog)
            stale["location_manifest_sha256_at_review"] = "0" * 64
            path.write_text(json.dumps(stale))
            with self.assertRaises(SystemExit):
                data_cleanup.load_catalog(path)

    def test_candidate_rejects_payload_escape_and_authoritative_overlap(self):
        catalog, _, _ = data_cleanup.load_catalog()
        candidate = copy.deepcopy(catalog["candidates"][0])
        candidate["scientific_payload"]["source_path"] = "/tmp/outside"
        with self.assertRaises(SystemExit):
            data_cleanup.validate_candidate(candidate)

        candidate = copy.deepcopy(catalog["candidates"][0])
        candidate["scientific_payload"]["authoritative_path"] = candidate[
            "source_path"
        ]
        with self.assertRaises(SystemExit):
            data_cleanup.validate_candidate(candidate)


if __name__ == "__main__":
    unittest.main()
