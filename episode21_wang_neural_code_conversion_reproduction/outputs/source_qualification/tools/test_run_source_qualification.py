#!/usr/bin/env python3

import ast
import copy
import hashlib
import io
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


RUNNER = Path(__file__).with_name("run_source_qualification.py")


class SourceQualificationRunnerTest(unittest.TestCase):
    def test_python_does_not_contain_json_literal_names(self) -> None:
        tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
        invalid = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id in {"false", "true", "null"}
        }
        self.assertEqual(invalid, set())

    def test_archive_member_safety(self) -> None:
        spec = importlib.util.spec_from_file_location("ep21_source_qualification", RUNNER)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(module.safe_archive_name("release/code.py"))
        self.assertFalse(module.safe_archive_name("../outside"))
        self.assertFalse(module.safe_archive_name("/absolute/path"))

    def test_source_identity_is_anchored_to_every_ep21_pin(self) -> None:
        spec = importlib.util.spec_from_file_location("ep21_source_identity", RUNNER)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = copy.deepcopy(module.PINNED_SOURCE_RELEASE)
        common = {
            "source": source,
            "tag_commit": module.PINNED_SOURCE_RELEASE["expected_commit"],
            "archive_md5": module.PINNED_SOURCE_RELEASE["expected_archive_md5"],
            "archive_size": module.PINNED_SOURCE_RELEASE[
                "expected_archive_size_bytes"
            ],
            "archive_inspection": {"unsafe_member_names": []},
            "errors": [],
        }
        self.assertTrue(
            module.source_identity_matches(
                zenodo={"id": 14910040, "doi": "10.5281/zenodo.14910040"},
                **common,
            )
        )
        self.assertFalse(
            module.source_identity_matches(
                zenodo={"id": 14910041, "doi": "10.5281/zenodo.14910040"},
                **common,
            )
        )
        self.assertFalse(
            module.source_identity_matches(
                zenodo={"id": 14910040, "doi": "10.5281/zenodo.wrong"},
                **common,
            )
        )
        for field in module.PINNED_SOURCE_RELEASE:
            mutated = copy.deepcopy(source)
            mutated[field] = (
                -1
                if isinstance(module.PINNED_SOURCE_RELEASE[field], int)
                else f"wrong-{field}"
            )
            with self.subTest(field=field):
                self.assertTrue(module.source_contract_errors(mutated))
                self.assertFalse(
                    module.source_identity_matches(
                        source=mutated,
                        zenodo={
                            "id": 14910040,
                            "doi": "10.5281/zenodo.14910040",
                        },
                        tag_commit=module.PINNED_SOURCE_RELEASE["expected_commit"],
                        archive_md5=module.PINNED_SOURCE_RELEASE[
                            "expected_archive_md5"
                        ],
                        archive_size=module.PINNED_SOURCE_RELEASE[
                            "expected_archive_size_bytes"
                        ],
                        archive_inspection={"unsafe_member_names": []},
                        errors=[],
                    )
                )

    def test_direct_runner_rejects_archive_path_substitution(self) -> None:
        spec = importlib.util.spec_from_file_location("ep21_source_path", RUNNER)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for unsafe in ["/tmp/escape.zip", "../escape.zip", "a/../../escape.zip"]:
            source = copy.deepcopy(module.PINNED_SOURCE_RELEASE)
            source["expected_archive_name"] = unsafe
            with self.subTest(unsafe=unsafe):
                self.assertTrue(module.source_contract_errors(source))
                self.assertFalse(module.safe_archive_name(unsafe))
        self.assertIn(
            'archive_path = args.scratch_dir / PINNED_SOURCE_RELEASE["expected_archive_name"]',
            RUNNER.read_text(encoding="utf-8"),
        )

    def test_attempt_is_required_and_not_hard_coded(self) -> None:
        tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
        attempt_values = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "attempt":
                    attempt_values.append(value)
        self.assertGreaterEqual(len(attempt_values), 3)
        self.assertTrue(
            all(
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "args"
                and value.attr == "attempt"
                for value in attempt_values
            )
        )
        self.assertIn('add_argument("--attempt", type=int, required=True)', RUNNER.read_text())

    def test_download_is_bounded_atomic_and_does_not_follow_destination(self) -> None:
        spec = importlib.util.spec_from_file_location("ep21_source_download", RUNNER)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Response(io.BytesIO):
            def __init__(self, body: bytes) -> None:
                super().__init__(body)
                self.headers = {"Content-Length": str(len(body))}

        body = b"authenticated-public-source"
        expected_md5 = hashlib.md5(body).hexdigest()
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            sentinel = directory / "sentinel"
            sentinel.write_bytes(b"keep-me")
            destination = directory / "source.zip"
            destination.symlink_to(sentinel)
            with mock.patch.object(
                module.urllib.request, "urlopen", return_value=Response(body)
            ):
                observed = module.download(
                    "https://example.test/source.zip",
                    destination,
                    expected_md5,
                    len(body),
                )
            self.assertEqual(observed, (expected_md5, len(body)))
            self.assertEqual(destination.read_bytes(), body)
            self.assertFalse(destination.is_symlink())
            self.assertEqual(sentinel.read_bytes(), b"keep-me")
            self.assertEqual(destination.stat().st_mode & 0o777, 0o440)
            self.assertFalse(any(directory.glob(".source.zip.part-*")))

    def test_failed_download_preserves_existing_destination(self) -> None:
        spec = importlib.util.spec_from_file_location("ep21_source_bad_download", RUNNER)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Response(io.BytesIO):
            def __init__(self, body: bytes) -> None:
                super().__init__(body)
                self.headers = {"Content-Length": str(len(body))}

        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            destination = directory / "source.zip"
            destination.write_bytes(b"existing")
            with mock.patch.object(
                module.urllib.request, "urlopen", return_value=Response(b"bad")
            ), self.assertRaises(ValueError):
                module.download(
                    "https://example.test/source.zip",
                    destination,
                    hashlib.md5(b"good").hexdigest(),
                    3,
                )
            self.assertEqual(destination.read_bytes(), b"existing")
            self.assertFalse(any(directory.glob(".source.zip.part-*")))


if __name__ == "__main__":
    unittest.main()
