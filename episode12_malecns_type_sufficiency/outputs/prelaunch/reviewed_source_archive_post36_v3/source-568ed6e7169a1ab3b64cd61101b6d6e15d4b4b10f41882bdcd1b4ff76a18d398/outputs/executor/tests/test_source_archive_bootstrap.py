from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ep12_executor import source_archive
from ep12_executor import source_archive_bootstrap as bootstrap


class SourceArchiveBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "reviewed"
        self.archives = self.root / "archives"
        self.stages = self.root / "stages"
        (self.source / "outputs/executor/ep12_executor").mkdir(parents=True)
        self.archives.mkdir(mode=0o750)
        self.stages.mkdir(mode=0o700)
        (self.source / "SEARCH_POLICY.yaml").write_bytes(b"episode_id: ep12\n")
        (self.source / "outputs/executor/ep12_executor/module.py").write_bytes(
            b"SYNTHETIC = True\n"
        )
        self.publication = source_archive.publish_source_archive(
            source_root=self.source,
            archive_parent=self.archives,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _arguments(
        self,
        *,
        publication: dict[str, object] | None = None,
        stage_parent: Path | None = None,
    ) -> argparse.Namespace:
        value = publication or self.publication
        return argparse.Namespace(
            archive_root=Path(str(value["archived_root"])),
            stage_parent=stage_parent or self.stages,
            expected_receipt_file_sha256=value[
                "archive_receipt_file_sha256"
            ],
            expected_receipt_sha256=value["archive_receipt_sha256"],
            expected_manifest_file_sha256=value["manifest_file_sha256"],
            expected_manifest_sha256=value["manifest_sha256"],
            expected_tree_sha256=value["tree_sha256"],
        )

    def test_verify_stage_and_identical_retry(self) -> None:
        first = bootstrap.verify_and_stage(self._arguments())
        second = bootstrap.verify_and_stage(self._arguments())
        self.assertEqual(first["stage_status"], "created")
        self.assertEqual(second["stage_status"], "reused_identical")
        self.assertEqual(first["staged_root"], second["staged_root"])
        self.assertEqual(first["tree_sha256"], self.publication["tree_sha256"])
        staged = Path(first["staged_root"])
        self.assertEqual(
            (staged / "SEARCH_POLICY.yaml").read_bytes(), b"episode_id: ep12\n"
        )
        self.assertEqual(
            (staged / "outputs/executor/ep12_executor/module.py").read_bytes(),
            b"SYNTHETIC = True\n",
        )

    def test_wrong_expected_identity_fails_before_stage(self) -> None:
        arguments = self._arguments()
        arguments.expected_tree_sha256 = "0" * 64
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap.verify_and_stage(arguments)
        self.assertFalse(
            any(path.name.startswith(bootstrap.PREFIX) for path in self.stages.iterdir())
        )

    def test_stage_parent_bound_to_other_archive_fails(self) -> None:
        bootstrap.verify_and_stage(self._arguments())
        second_source = self.root / "reviewed-second"
        second_archives = self.root / "archives-second"
        (second_source / "outputs/executor").mkdir(parents=True)
        second_archives.mkdir(mode=0o750)
        (second_source / "SEARCH_POLICY.yaml").write_bytes(b"episode_id: ep12\n")
        (second_source / "outputs/executor/changed.py").write_bytes(b"CHANGED = True\n")
        second = source_archive.publish_source_archive(
            source_root=second_source,
            archive_parent=second_archives,
        )
        with self.assertRaisesRegex(
            bootstrap.BootstrapError, "already bound to a different archive"
        ):
            bootstrap.verify_and_stage(
                self._arguments(publication=second, stage_parent=self.stages)
            )
        self.assertEqual(
            json.loads((self.stages / bootstrap.RECEIPT_NAME).read_text())["tree_sha256"],
            self.publication["tree_sha256"],
        )

    def test_mutated_archive_fails_closed(self) -> None:
        archive = Path(str(self.publication["archived_root"]))
        source = archive / "outputs/executor/ep12_executor/module.py"
        os.chmod(archive, 0o750)
        os.chmod(source, 0o640)
        source.write_bytes(b"MUTATED = True\n")
        os.chmod(source, 0o440)
        os.chmod(archive, 0o550)
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap.verify_and_stage(self._arguments())

    def test_bootstrap_has_only_stdlib_imports(self) -> None:
        path = Path(bootstrap.__file__).resolve()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        relative_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    relative_imports.append(node)
                elif node.module:
                    imported.add(node.module.split(".", 1)[0])
        self.assertEqual(relative_imports, [])
        self.assertTrue(imported <= sys.stdlib_module_names | {"__future__"})

    def test_isolated_cli_does_not_need_package_imports(self) -> None:
        isolated_stage = self.root / "isolated-stage"
        isolated_stage.mkdir(mode=0o700)
        arguments = self._arguments(stage_parent=isolated_stage)
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(Path(bootstrap.__file__).resolve()),
                "verify-and-stage",
                "--archive-root",
                str(arguments.archive_root),
                "--stage-parent",
                str(arguments.stage_parent),
                "--expected-receipt-file-sha256",
                str(arguments.expected_receipt_file_sha256),
                "--expected-receipt-sha256",
                str(arguments.expected_receipt_sha256),
                "--expected-manifest-file-sha256",
                str(arguments.expected_manifest_file_sha256),
                "--expected-manifest-sha256",
                str(arguments.expected_manifest_sha256),
                "--expected-tree-sha256",
                str(arguments.expected_tree_sha256),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={
                "PATH": os.environ.get("PATH", ""),
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
                "LANG": "C",
            },
        )
        result = json.loads(completed.stdout)
        self.assertTrue(result["complete"])
        self.assertEqual(result["tree_sha256"], self.publication["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
