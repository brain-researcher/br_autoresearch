from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

from ep12_executor import source_archive


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _surface(root: Path) -> dict[str, tuple[str, int]]:
    result: dict[str, tuple[str, int]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            result[relative] = (f"symlink:{os.readlink(path)}", 0)
        elif path.is_file():
            result[relative] = (_sha(path), stat.S_IMODE(path.stat().st_mode))
        elif path.is_dir():
            result[relative] = ("directory", stat.S_IMODE(path.stat().st_mode))
    return result


class SourceArchivePublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "reviewed-source"
        self.archives = self.root / "durable-archives"
        self.stages = self.root / "scratch-stages"
        self.source.mkdir()
        self.archives.mkdir()
        self.stages.mkdir()
        self.executor = self.source / "outputs/executor"
        self.executor.mkdir(parents=True)
        (self.executor / "ep12_executor").mkdir()
        (self.executor / "slurm").mkdir()
        (self.executor / "empty").mkdir()
        (self.executor / "ep12_executor/module.py").write_bytes(
            b"VALUE = 'synthetic-only'\n"
        )
        (self.executor / "slurm/run.sbatch").write_bytes(
            b"#!/bin/bash\necho synthetic\n"
        )
        os.chmod(self.executor / "slurm/run.sbatch", 0o750)
        (self.executor / "notes.txt").write_bytes(b"reviewed synthetic note\n")
        (self.source / "SEARCH_POLICY.yaml").write_bytes(b"episode_id: ep12\n")
        (self.executor / "launch.py").write_bytes(
            b"EPISODE_LEVEL = True\n"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _publish(self) -> dict:
        return source_archive.publish_source_archive(
            source_root=self.source,
            archive_parent=self.archives,
        )

    def _stage(self, publication: dict) -> dict:
        return source_archive.stage_source_archive(
            archive_root=Path(publication["archived_root"]),
            stage_parent=self.stages,
            expected_receipt_file_sha256=publication[
                "archive_receipt_file_sha256"
            ],
            expected_receipt_sha256=publication["archive_receipt_sha256"],
            expected_manifest_file_sha256=publication[
                "manifest_file_sha256"
            ],
            expected_manifest_sha256=publication["manifest_sha256"],
            expected_tree_sha256=publication["tree_sha256"],
        )

    def test_publish_verify_and_identical_retry_preserve_source(self) -> None:
        before = _surface(self.source)
        first = self._publish()
        after = _surface(self.source)
        self.assertEqual(before, after)
        self.assertEqual(first["publication_status"], "created")
        archive = Path(first["archived_root"])
        self.assertEqual(
            archive.name,
            f"{source_archive.SOURCE_ARCHIVE_PREFIX}{first['tree_sha256']}",
        )
        self.assertEqual(stat.S_IMODE(archive.stat().st_mode), 0o550)
        self.assertEqual(
            stat.S_IMODE((archive / "outputs/executor/ep12_executor/module.py").stat().st_mode),
            0o440,
        )
        self.assertEqual(
            stat.S_IMODE((archive / "outputs/executor/slurm/run.sbatch").stat().st_mode),
            0o550,
        )
        self.assertTrue((archive / "SEARCH_POLICY.yaml").is_file())
        self.assertTrue((archive / "outputs/executor/launch.py").is_file())

        manifest_path = archive / source_archive.SOURCE_ARCHIVE_MANIFEST
        receipt_path = self.archives / source_archive.SOURCE_ARCHIVE_RECEIPT
        self.assertEqual(stat.S_IMODE(receipt_path.stat().st_mode), 0o440)
        encoded = manifest_path.read_bytes()
        manifest = json.loads(encoded)
        self.assertEqual(
            encoded,
            (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        )
        self.assertEqual(_sha(manifest_path), first["manifest_file_sha256"])
        self.assertEqual(manifest["manifest_sha256"], first["manifest_sha256"])
        self.assertEqual(manifest["tree_sha256"], first["tree_sha256"])
        self.assertEqual(_sha(receipt_path), first["archive_receipt_file_sha256"])

        verified = source_archive.verify_source_archive(
            archive,
            expected_manifest_file_sha256=first["manifest_file_sha256"],
            expected_manifest_sha256=first["manifest_sha256"],
            expected_tree_sha256=first["tree_sha256"],
        )
        self.assertTrue(verified["complete"])
        self.assertTrue(verified["read_only"])
        second = self._publish()
        self.assertEqual(second["publication_status"], "reused_identical")
        for key in (
            "archived_root",
            "manifest_file_sha256",
            "manifest_sha256",
            "tree_sha256",
        ):
            self.assertEqual(second[key], first[key])

    def test_live_episode_and_result_surfaces_fail_before_publication(self) -> None:
        forbidden = (
            "inputs",
            "outputs/prelaunch",
            "outputs/development_run002",
        )
        for relative in forbidden:
            with self.subTest(relative=relative):
                path = self.source / relative
                path.mkdir(parents=True)
                with self.assertRaisesRegex(
                    source_archive.SourceArchiveError,
                    "exactly SEARCH_POLICY|exactly executor",
                ):
                    self._publish()
                path.rmdir()
        self.assertFalse(
            any(
                path.name.startswith(source_archive.SOURCE_ARCHIVE_PREFIX)
                for path in self.archives.iterdir()
            )
        )

    def test_excluded_source_artifacts_are_rejected(self) -> None:
        for name in ("failed.orig", "failed.rej", "failed.patch"):
            with self.subTest(name=name):
                rejected = self.executor / name
                rejected.write_bytes(b"must not be archived\n")
                with self.assertRaisesRegex(
                    source_archive.SourceArchiveError, "excluded artifact"
                ):
                    self._publish()
                rejected.unlink()
        cache = self.executor / ".pycache-check"
        cache.mkdir()
        with self.assertRaisesRegex(
            source_archive.SourceArchiveError, "excluded directory"
        ):
            self._publish()
        cache.rmdir()
        self.assertFalse(
            any(
                path.name.startswith(source_archive.SOURCE_ARCHIVE_PREFIX)
                for path in self.archives.iterdir()
            )
        )

    def test_merged_source_surface_has_no_backup_or_cache_artifacts(self) -> None:
        merged = Path(__file__).resolve().parents[1]
        forbidden_files = sorted(
            path.relative_to(merged).as_posix()
            for path in merged.rglob("*")
            if path.is_file()
            and path.name.endswith((".orig", ".rej", ".patch", ".pyc"))
        )
        forbidden_directories = sorted(
            path.relative_to(merged).as_posix()
            for path in merged.rglob("*")
            if path.is_dir()
            and path.name in {"__pycache__", ".pytest_cache", ".pycache-check"}
        )
        self.assertEqual(forbidden_files, [])
        self.assertEqual(forbidden_directories, [])

    def test_changed_source_cannot_publish_second_tree_under_bound_parent(self) -> None:
        first = self._publish()
        receipt = self.archives / source_archive.SOURCE_ARCHIVE_RECEIPT
        receipt_before = receipt.read_bytes()
        (self.executor / "notes.txt").write_bytes(b"different reviewed content\n")
        with self.assertRaisesRegex(
            source_archive.SourceArchiveError, "already bound to different"
        ):
            self._publish()
        self.assertEqual(receipt.read_bytes(), receipt_before)
        content_archives = [
            path
            for path in self.archives.iterdir()
            if path.name.startswith(source_archive.SOURCE_ARCHIVE_PREFIX)
        ]
        self.assertEqual(content_archives, [Path(first["archived_root"])])
        verified = source_archive.verify_source_archive_publication(self.archives)
        self.assertEqual(verified["tree_sha256"], first["tree_sha256"])

    def test_concurrent_identical_retry_serializes_to_one_archive(self) -> None:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: self._publish(), range(2)))
        self.assertEqual(
            sorted(result["publication_status"] for result in results),
            ["created", "reused_identical"],
        )
        self.assertEqual(
            {result["tree_sha256"] for result in results},
            {results[0]["tree_sha256"]},
        )
        content_archives = [
            path
            for path in self.archives.iterdir()
            if path.name.startswith(source_archive.SOURCE_ARCHIVE_PREFIX)
        ]
        self.assertEqual(len(content_archives), 1)
        self.assertTrue(
            (self.archives / source_archive.SOURCE_ARCHIVE_RECEIPT).is_file()
        )

    def test_stage_uses_only_immutable_archive_and_reuses_exact_identity(self) -> None:
        publication = self._publish()
        durable_receipt = (
            self.archives / source_archive.SOURCE_ARCHIVE_RECEIPT
        ).read_bytes()
        # A post-publication live-source change must not enter downstream staging.
        (self.executor / "notes.txt").write_bytes(b"live source changed later\n")
        first = self._stage(publication)
        self.assertEqual(first["stage_status"], "created")
        staged = Path(first["staged_root"])
        self.assertEqual(
            (staged / "outputs/executor/notes.txt").read_bytes(),
            b"reviewed synthetic note\n",
        )
        self.assertEqual(
            (self.stages / source_archive.SOURCE_ARCHIVE_RECEIPT).read_bytes(),
            durable_receipt,
        )
        self.assertEqual(stat.S_IMODE(staged.stat().st_mode), 0o550)
        self.assertEqual(
            stat.S_IMODE((staged / "outputs/executor/notes.txt").stat().st_mode), 0o440
        )
        second = self._stage(publication)
        self.assertEqual(second["stage_status"], "reused_identical")
        self.assertEqual(second["staged_root"], first["staged_root"])
        self.assertEqual(second["tree_sha256"], publication["tree_sha256"])

    def test_stage_rejects_extra_or_wrong_expected_identity(self) -> None:
        publication = self._publish()
        with self.assertRaisesRegex(source_archive.SourceArchiveError, "mismatch"):
            source_archive.stage_source_archive(
                archive_root=Path(publication["archived_root"]),
                stage_parent=self.stages,
                expected_receipt_file_sha256="0" * 64,
                expected_receipt_sha256=publication["archive_receipt_sha256"],
                expected_manifest_file_sha256=publication[
                    "manifest_file_sha256"
                ],
                expected_manifest_sha256=publication["manifest_sha256"],
                expected_tree_sha256=publication["tree_sha256"],
            )
        self.assertEqual(list(self.stages.iterdir()), [])
        staged_result = self._stage(publication)
        staged = Path(staged_result["staged_root"])
        os.chmod(staged, 0o750)
        extra = staged / "extra.py"
        extra.write_bytes(b"extra\n")
        os.chmod(extra, 0o440)
        os.chmod(staged, 0o550)
        with self.assertRaisesRegex(source_archive.SourceArchiveError, "extra files"):
            self._stage(publication)

    def test_symlink_rejected_in_executor_source(self) -> None:
        target = self.executor / "notes.txt"
        link = self.executor / "redirect.py"
        link.symlink_to(target)
        with self.assertRaisesRegex(source_archive.SourceArchiveError, "symlink"):
            self._publish()
        self.assertEqual(
            [path.name for path in self.archives.iterdir()],
            [source_archive.SOURCE_ARCHIVE_LOCK],
        )

    def test_symlink_receipt_cannot_redirect_first_publication(self) -> None:
        outside = self.root / "outside-receipt.json"
        outside.write_bytes(b"not a receipt\n")
        (self.archives / source_archive.SOURCE_ARCHIVE_RECEIPT).symlink_to(outside)
        with self.assertRaisesRegex(
            source_archive.SourceArchiveError, "cannot safely read"
        ):
            self._publish()
        self.assertFalse(
            any(
                path.name.startswith(source_archive.SOURCE_ARCHIVE_PREFIX)
                for path in self.archives.iterdir()
            )
        )

    def test_source_mutation_after_copy_is_detected_and_private_tree_removed(self) -> None:
        original = source_archive._copy_regular_once
        mutated = False

        def mutate_after_copy(**kwargs: object) -> dict[str, object]:
            nonlocal mutated
            record = original(**kwargs)  # type: ignore[arg-type]
            if not mutated:
                descriptor = os.open(
                    str(kwargs["name"]),
                    os.O_WRONLY | os.O_APPEND,
                    dir_fd=int(kwargs["source_directory_fd"]),
                )
                try:
                    os.write(descriptor, b"mutation\n")
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                mutated = True
            return record

        with mock.patch.object(
            source_archive, "_copy_regular_once", side_effect=mutate_after_copy
        ):
            with self.assertRaisesRegex(source_archive.SourceArchiveError, "mutated"):
                self._publish()
        self.assertTrue(mutated)
        self.assertEqual(
            [path.name for path in self.archives.iterdir()],
            [source_archive.SOURCE_ARCHIVE_LOCK],
        )

    def test_verifier_rejects_extra_file_and_retry_does_not_overwrite(self) -> None:
        first = self._publish()
        archive = Path(first["archived_root"])
        os.chmod(archive, 0o750)
        extra = archive / "undeclared.txt"
        extra.write_bytes(b"not declared\n")
        os.chmod(extra, 0o440)
        os.chmod(archive, 0o550)
        with self.assertRaisesRegex(source_archive.SourceArchiveError, "extra files"):
            source_archive.verify_source_archive(archive)
        with self.assertRaisesRegex(source_archive.SourceArchiveError, "extra files"):
            self._publish()
        self.assertEqual(extra.read_bytes(), b"not declared\n")
        self.assertFalse(
            any(path.name.startswith(".source-tmp") for path in self.archives.iterdir())
        )

    def test_verifier_rejects_content_mode_and_expected_digest_changes(self) -> None:
        first = self._publish()
        archive = Path(first["archived_root"])
        copied = archive / "outputs/executor/notes.txt"
        os.chmod(copied, 0o640)
        with self.assertRaisesRegex(source_archive.SourceArchiveError, "file changed"):
            source_archive.verify_source_archive(archive)
        os.chmod(copied, 0o440)
        with self.assertRaisesRegex(
            source_archive.SourceArchiveError, "manifest file hash mismatch"
        ):
            source_archive.verify_source_archive(
                archive, expected_manifest_file_sha256="0" * 64
            )
        with self.assertRaisesRegex(source_archive.SourceArchiveError, "invalid"):
            source_archive.verify_source_archive(
                archive, expected_tree_sha256="not-a-digest"
            )

    def test_group_writable_archive_parent_is_rejected(self) -> None:
        os.chmod(self.archives, 0o770)
        try:
            with self.assertRaisesRegex(
                source_archive.SourceArchiveError, "not group/world writable"
            ):
                self._publish()
        finally:
            os.chmod(self.archives, 0o750)

    def test_source_and_archive_parent_must_be_existing_absolute_disjoint_dirs(self) -> None:
        with self.assertRaisesRegex(source_archive.SourceArchiveError, "absolute"):
            source_archive.publish_source_archive(
                source_root=Path("relative"), archive_parent=self.archives
            )
        with self.assertRaisesRegex(source_archive.SourceArchiveError, "disjoint"):
            source_archive.publish_source_archive(
                source_root=self.source, archive_parent=self.source / "nested"
            )


if __name__ == "__main__":
    unittest.main()
