#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("verified_source_archive.py")
SPEC = importlib.util.spec_from_file_location("ep21_verified_source_archive", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_zip(path: Path, payload: bytes) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("release/evidence.txt", payload)


class VerifiedSourceArchiveTests(unittest.TestCase):
    def test_hash_and_zip_parse_cannot_mix_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            archive_path = directory / "source.zip"
            replacement = directory / "replacement.zip"
            write_zip(archive_path, b"authenticated-A")
            write_zip(replacement, b"different-B")
            pinned_md5 = hashlib.md5(archive_path.read_bytes()).hexdigest()
            pinned_size = archive_path.stat().st_size
            original_stream = MODULE._stream_md5

            def stream_then_replace(handle):
                result = original_stream(handle)
                os.replace(replacement, archive_path)
                return result

            observed_payload = None
            try:
                with mock.patch.object(
                    MODULE, "_stream_md5", side_effect=stream_then_replace
                ), MODULE.verified_zip(
                    archive_path, pinned_md5, pinned_size
                ) as verified:
                    observed_payload = verified.archive.read("release/evidence.txt")
            except ValueError:
                pass
            self.assertIn(observed_payload, (None, b"authenticated-A"))
            self.assertNotEqual(observed_payload, b"different-B")

    def test_symlink_and_hardlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            target = directory / "target.zip"
            write_zip(target, b"payload")
            expected_md5 = hashlib.md5(target.read_bytes()).hexdigest()
            symlink = directory / "symlink.zip"
            symlink.symlink_to(target)
            hardlink = directory / "hardlink.zip"
            os.link(target, hardlink)
            for path in (symlink, hardlink):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    with MODULE.verified_zip(path, expected_md5):
                        pass

    def test_in_place_mutation_before_exit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            archive_path = Path(name) / "source.zip"
            write_zip(archive_path, b"payload")
            expected_md5 = hashlib.md5(archive_path.read_bytes()).hexdigest()
            expected_size = archive_path.stat().st_size
            with self.assertRaises(ValueError):
                with MODULE.verified_zip(
                    archive_path, expected_md5, expected_size
                ) as verified:
                    self.assertEqual(
                        verified.archive.read("release/evidence.txt"), b"payload"
                    )
                    with archive_path.open("ab") as handle:
                        handle.write(b"mutation")
                        handle.flush()
                        os.fsync(handle.fileno())


if __name__ == "__main__":
    unittest.main()
