#!/usr/bin/env python3
"""Stdlib-only trust anchor for verifying and staging an EP12 source archive.

This file is executed with python3 -I -S only after the spooled Slurm wrapper
has independently checked its exact SHA-256. It intentionally imports no EP12
package code. It verifies the parent publication receipt, canonical manifest,
complete archive surface, every file hash/mode, and the tree digest before
publishing a read-only content-addressed stage.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
import sys
from typing import Any, Mapping, Sequence

RECEIPT_NAME = "source_archive_receipt.json"
MANIFEST_NAME = "source_archive_manifest.json"
PREFIX = "source-"
LOCK_NAME = ".source_archive_bootstrap.lock"
RECEIPT_SCHEMA = "ep12.reviewed_source_archive_receipt.v1"
MANIFEST_SCHEMA = "ep12.reviewed_source_archive_manifest.v1"
HEX = frozenset("0123456789abcdef")
EXCLUDED_DIRECTORIES = frozenset({
    ".git", ".hg", ".mypy_cache", ".nox", ".pytest_cache", ".ruff_cache",
    ".svn", ".tox", "__pycache__", ".pycache-check", ".temp", "temp", "tmp", "patch",
})
EXCLUDED_FILES = frozenset({".coverage", ".DS_Store", "patch", "temp", "tmp"})
EXCLUDED_SUFFIXES = (
    ".diff", ".orig", ".patch", ".pyc", ".pyo", ".rej", ".swo", ".swp",
    ".temp", ".tmp", "~",
)
MANIFEST_FIELDS = {
    "schema_version", "archive_kind", "surface_rule", "directory_count",
    "file_count", "total_source_bytes", "directories", "files", "tree_sha256",
    "source_files_read_once", "symlinks_allowed", "complete", "manifest_sha256",
}
RECEIPT_FIELDS = {
    "schema_version", "archive_name", "manifest_name", "manifest_file_sha256",
    "manifest_sha256", "tree_sha256", "directory_count", "file_count",
    "total_source_bytes", "complete", "receipt_sha256",
}


class BootstrapError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BootstrapError(message)


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BootstrapError("noncanonical JSON value") from exc


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_object(value: Any) -> str:
    return digest_bytes(canonical(value))


def is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def surface_rule() -> dict[str, Any]:
    return {
        "include": ["SEARCH_POLICY.yaml", "outputs/executor/**"],
        "source_root_layout": "exact_clean_episode_source_v1",
        "excluded_directory_names": sorted(EXCLUDED_DIRECTORIES),
        "excluded_file_names": sorted(EXCLUDED_FILES),
        "excluded_suffixes": list(EXCLUDED_SUFFIXES),
        "reserved_archive_manifest": MANIFEST_NAME,
        "symlinks_allowed": False,
        "special_files_allowed": False,
    }


def lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def secure_existing(path: Path, *, directory: bool, label: str) -> Path:
    require(path.is_absolute(), f"{label} must be absolute")
    candidate = lexical(path)
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current = current / component
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise BootstrapError(f"cannot inspect {label} component {current}: {exc}") from exc
        require(not stat.S_ISLNK(metadata.st_mode), f"{label} traverses symlink: {current}")
    metadata = os.lstat(candidate)
    require(
        stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode),
        f"{label} has wrong file type",
    )
    return candidate


def read_regular(path: Path, *, label: str) -> tuple[bytes, os.stat_result]:
    secure_existing(path, directory=False, label=label)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"{label} is not regular")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            == (named.st_dev, named.st_ino, named.st_size, named.st_mtime_ns, named.st_ctime_ns),
            f"{label} changed while read",
        )
        return b"".join(chunks), after
    finally:
        os.close(descriptor)


def parse_canonical(encoded: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError(f"{label} is malformed") from exc
    require(isinstance(value, dict), f"{label} is not an object")
    require(encoded == canonical(value) + b"\n", f"{label} is not canonical JSON")
    return value


def relative(value: Any, *, label: str) -> str:
    require(isinstance(value, str) and value, f"{label} is invalid")
    path = PurePosixPath(value)
    require(
        not path.is_absolute() and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"{label} is unsafe",
    )
    return value


def excluded_directory(name: str) -> bool:
    return name in EXCLUDED_DIRECTORIES or name.endswith(
        (".diff", ".orig", ".rej", ".patch", ".temp", ".tmp")
    )


def excluded_file(name: str) -> bool:
    return (
        name in EXCLUDED_FILES or name.startswith(".tmp.")
        or name.endswith(EXCLUDED_SUFFIXES)
    )


FORBIDDEN_SOURCE_COMPONENTS = frozenset({"inputs", "outcomes", "prelaunch"})


def forbidden_source_component(name: str) -> bool:
    return name in FORBIDDEN_SOURCE_COMPONENTS or name.startswith("development_run")


def parse_receipt(encoded: bytes) -> dict[str, Any]:
    receipt = parse_canonical(encoded, label="source archive receipt")
    require(set(receipt) == RECEIPT_FIELDS, "source archive receipt schema changed")
    core = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    for key in ("manifest_file_sha256", "manifest_sha256", "tree_sha256", "receipt_sha256"):
        require(is_sha(receipt.get(key)), f"source archive receipt {key} is invalid")
    for key in ("directory_count", "file_count", "total_source_bytes"):
        require(type(receipt.get(key)) is int and receipt[key] >= 0, f"source archive receipt {key} is invalid")
    require(
        receipt.get("schema_version") == RECEIPT_SCHEMA
        and receipt.get("manifest_name") == MANIFEST_NAME
        and receipt.get("archive_name") == PREFIX + receipt["tree_sha256"]
        and "/" not in receipt["archive_name"]
        and receipt.get("complete") is True
        and receipt.get("receipt_sha256") == digest_object(core),
        "source archive receipt identity mismatch",
    )
    return receipt


def parse_manifest(encoded: bytes) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    manifest = parse_canonical(encoded, label="source archive manifest")
    require(set(manifest) == MANIFEST_FIELDS, "source archive manifest schema changed")
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    require(
        manifest.get("schema_version") == MANIFEST_SCHEMA
        and manifest.get("archive_kind") == "content_addressed_reviewed_executor_source"
        and manifest.get("surface_rule") == surface_rule()
        and manifest.get("source_files_read_once") is True
        and manifest.get("symlinks_allowed") is False
        and manifest.get("complete") is True
        and manifest.get("manifest_sha256") == digest_object(core),
        "source archive manifest identity mismatch",
    )
    raw_directories = manifest.get("directories")
    raw_files = manifest.get("files")
    require(isinstance(raw_directories, list) and isinstance(raw_files, list), "archive surface lists are invalid")
    directories = [relative(value, label="archive directory") for value in raw_directories]
    require(
        directories == sorted(set(directories))
        and manifest.get("directory_count") == len(directories),
        "archive directory list is not exact",
    )
    for item in directories:
        require(
            not any(excluded_directory(part) for part in PurePosixPath(item).parts),
            f"archive contains excluded directory {item}",
        )
    files: list[dict[str, Any]] = []
    prior = ""
    for raw in raw_files:
        require(isinstance(raw, dict), "archive file record is invalid")
        record = dict(raw)
        require(set(record) == {"path", "size_bytes", "sha256", "archive_mode"}, "archive file record schema changed")
        name = relative(record.get("path"), label="archive file")
        require(name > prior and name != MANIFEST_NAME, "archive file order or name is invalid")
        prior = name
        require(
            not any(excluded_directory(part) for part in PurePosixPath(name).parts[:-1])
            and not excluded_file(PurePosixPath(name).name),
            f"archive contains excluded file {name}",
        )
        require(type(record.get("size_bytes")) is int and record["size_bytes"] >= 0, "archive file size is invalid")
        require(is_sha(record.get("sha256")), "archive file hash is invalid")
        require(record.get("archive_mode") in {"0440", "0550"}, "archive file mode is invalid")
        files.append(record)
    require(
        manifest.get("file_count") == len(files)
        and manifest.get("total_source_bytes") == sum(item["size_bytes"] for item in files),
        "archive file count or byte total mismatch",
    )
    directory_set = set(directories)
    for item in directories:
        parent = PurePosixPath(item).parent.as_posix()
        require(parent == "." or parent in directory_set, f"archive directory lacks parent: {item}")
    for record in files:
        parent = PurePosixPath(record["path"]).parent.as_posix()
        require(parent == "." or parent in directory_set, f"archive file lacks parent: {record['path']}")
    directory_set = set(directories)
    require(
        {"outputs", "outputs/executor"} <= directory_set,
        "archive lacks exact episode executor directories",
    )
    for item in directories:
        parts = PurePosixPath(item).parts
        require(
            item in {"outputs", "outputs/executor"}
            or item.startswith("outputs/executor/"),
            f"archive directory is outside exact executor allowlist: {item}",
        )
        require(
            not any(forbidden_source_component(part) for part in parts),
            f"archive directory is a forbidden episode artifact: {item}",
        )
    file_paths = [str(record["path"]) for record in files]
    require(
        "SEARCH_POLICY.yaml" in file_paths
        and any(path.startswith("outputs/executor/") for path in file_paths),
        "archive lacks SEARCH_POLICY.yaml or executor source files",
    )
    for item in file_paths:
        parts = PurePosixPath(item).parts
        require(
            item == "SEARCH_POLICY.yaml" or item.startswith("outputs/executor/"),
            f"archive file is outside exact executor allowlist: {item}",
        )
        require(
            not any(forbidden_source_component(part) for part in parts),
            f"archive file is a forbidden episode artifact: {item}",
        )
    tree = {"surface_rule": surface_rule(), "directories": directories, "files": files}
    require(is_sha(manifest.get("tree_sha256")) and digest_object(tree) == manifest["tree_sha256"], "archive tree hash mismatch")
    return manifest, directories, files


def scan_tree(root: Path) -> tuple[list[str], list[str]]:
    directories: list[str] = []
    files: list[str] = []
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        relative_root = current_path.relative_to(root)
        for name in sorted(dirnames):
            path = current_path / name
            metadata = os.lstat(path)
            require(stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), f"archive has unsafe directory {path}")
            directories.append((relative_root / name).as_posix())
        for name in sorted(filenames):
            path = current_path / name
            metadata = os.lstat(path)
            require(stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), f"archive has unsafe file {path}")
            files.append((relative_root / name).as_posix())
    return sorted(directories), sorted(files)


def metadata_snapshot(root: Path) -> tuple[tuple[Any, ...], ...]:
    """Bind the complete namespace and inode metadata across verify and copy."""

    secure_existing(root, directory=True, label="archive snapshot root")
    records: list[tuple[Any, ...]] = []
    for current, dirnames, filenames in os.walk(
        root, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        relative_root = current_path.relative_to(root)
        names = sorted(dirnames + filenames)
        if current_path == root:
            value = os.lstat(root)
            records.append((
                ".", value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
                value.st_size, value.st_mtime_ns, value.st_ctime_ns,
            ))
        for name in names:
            path = current_path / name
            relative_name = (relative_root / name).as_posix()
            value = os.lstat(path)
            require(
                not stat.S_ISLNK(value.st_mode),
                f"archive snapshot found symlink: {relative_name}",
            )
            records.append((
                relative_name, value.st_dev, value.st_ino, value.st_mode,
                value.st_nlink, value.st_size, value.st_mtime_ns,
                value.st_ctime_ns,
            ))
    return tuple(records)


def verify_archive(
    archive_root: Path,
    *,
    receipt_file_sha256: str,
    receipt_sha256: str,
    manifest_file_sha256: str,
    manifest_sha256: str,
    tree_sha256: str,
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes, list[str], list[dict[str, Any]]]:
    for value in (receipt_file_sha256, receipt_sha256, manifest_file_sha256, manifest_sha256, tree_sha256):
        require(is_sha(value), "expected archive digest is invalid")
    archive = secure_existing(archive_root, directory=True, label="archive root")
    require(archive.name == PREFIX + tree_sha256, "archive root name is not expected content address")
    receipt_path = archive.parent / RECEIPT_NAME
    receipt_bytes, receipt_stat = read_regular(receipt_path, label="archive publication receipt")
    require(stat.S_IMODE(receipt_stat.st_mode) == 0o440, "archive receipt mode changed")
    receipt = parse_receipt(receipt_bytes)
    require(
        digest_bytes(receipt_bytes) == receipt_file_sha256
        and receipt["receipt_sha256"] == receipt_sha256
        and receipt["manifest_file_sha256"] == manifest_file_sha256
        and receipt["manifest_sha256"] == manifest_sha256
        and receipt["tree_sha256"] == tree_sha256
        and receipt["archive_name"] == archive.name,
        "archive receipt binding mismatch",
    )
    content_names: list[str] = []
    for item in archive.parent.iterdir():
        if item.name.startswith(PREFIX):
            metadata = os.lstat(item)
            require(stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), "archive parent content entry is unsafe")
            require(is_sha(item.name[len(PREFIX):]), "archive parent content name is invalid")
            content_names.append(item.name)
    require(sorted(content_names) == [archive.name], "archive parent is not bound to exactly one content archive")
    manifest_bytes, manifest_stat = read_regular(archive / MANIFEST_NAME, label="archive manifest")
    require(stat.S_IMODE(manifest_stat.st_mode) == 0o440, "archive manifest mode changed")
    manifest, directories, files = parse_manifest(manifest_bytes)
    require(
        digest_bytes(manifest_bytes) == manifest_file_sha256
        and manifest["manifest_sha256"] == manifest_sha256
        and manifest["tree_sha256"] == tree_sha256,
        "archive manifest binding mismatch",
    )
    observed_directories, observed_files = scan_tree(archive)
    require(observed_directories == directories, "archive has missing or extra directories")
    require(observed_files == sorted([MANIFEST_NAME] + [item["path"] for item in files]), "archive has missing or extra files")
    require(stat.S_IMODE(os.lstat(archive).st_mode) == 0o550, "archive root mode changed")
    for item in directories:
        require(stat.S_IMODE(os.lstat(archive / item).st_mode) == 0o550, f"archive directory mode changed: {item}")
    for record in files:
        encoded, metadata = read_regular(archive / record["path"], label=f"archive file {record['path']}")
        require(
            len(encoded) == record["size_bytes"]
            and digest_bytes(encoded) == record["sha256"]
            and stat.S_IMODE(metadata.st_mode) == int(record["archive_mode"], 8),
            f"archive file changed: {record['path']}",
        )
    require(
        receipt["directory_count"] == len(directories)
        and receipt["file_count"] == len(files)
        and receipt["total_source_bytes"] == manifest["total_source_bytes"],
        "archive receipt counts mismatch",
    )
    return receipt, receipt_bytes, manifest, manifest_bytes, directories, files


def _clean_private_fd(directory_fd: int) -> None:
    os.fchmod(directory_fd, 0o700)
    for name in sorted(os.listdir(directory_fd)):
        value = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISREG(value.st_mode):
            os.chmod(
                name, 0o600, dir_fd=directory_fd, follow_symlinks=False
            )
            os.unlink(name, dir_fd=directory_fd)
        elif stat.S_ISDIR(value.st_mode) and not stat.S_ISLNK(value.st_mode):
            child_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                opened = os.fstat(child_fd)
                require(
                    (opened.st_dev, opened.st_ino) == (value.st_dev, value.st_ino),
                    "private cleanup directory raced",
                )
                _clean_private_fd(child_fd)
            finally:
                os.close(child_fd)
            os.rmdir(name, dir_fd=directory_fd)
        else:
            raise BootstrapError("private cleanup found symlink or special file")
    os.fsync(directory_fd)


def remove_private(path: Path) -> None:
    parent = secure_existing(path.parent, directory=True, label="private stage parent")
    parent_fd = os.open(
        parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    child_fd = -1
    try:
        try:
            named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        require(
            stat.S_ISDIR(named.st_mode) and not stat.S_ISLNK(named.st_mode),
            "private cleanup target is not a directory",
        )
        child_fd = os.open(
            path.name,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(child_fd)
        require(
            (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino),
            "private cleanup target raced",
        )
        _clean_private_fd(child_fd)
        os.close(child_fd)
        child_fd = -1
        os.rmdir(path.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        if child_fd >= 0:
            os.close(child_fd)
        os.close(parent_fd)


def verify_staged(target: Path, manifest_bytes: bytes, directories: list[str], files: list[dict[str, Any]]) -> None:
    secure_existing(target, directory=True, label="staged archive")
    observed_directories, observed_files = scan_tree(target)
    require(observed_directories == directories, "staged archive directories differ")
    require(observed_files == sorted([MANIFEST_NAME] + [item["path"] for item in files]), "staged archive files differ")
    manifest_copy, metadata = read_regular(target / MANIFEST_NAME, label="staged manifest")
    require(manifest_copy == manifest_bytes and stat.S_IMODE(metadata.st_mode) == 0o440, "staged manifest differs")
    for item in directories:
        require(stat.S_IMODE(os.lstat(target / item).st_mode) == 0o550, f"staged directory mode differs: {item}")
    for record in files:
        encoded, metadata = read_regular(target / record["path"], label=f"staged file {record['path']}")
        require(
            len(encoded) == record["size_bytes"] and digest_bytes(encoded) == record["sha256"]
            and stat.S_IMODE(metadata.st_mode) == int(record["archive_mode"], 8),
            f"staged file differs: {record['path']}",
        )
    require(stat.S_IMODE(os.lstat(target).st_mode) == 0o550, "staged root mode differs")


def copy_stage(archive: Path, private: Path, manifest_bytes: bytes, directories: list[str], files: list[dict[str, Any]]) -> None:
    os.mkdir(private, 0o700)
    for item in directories:
        (private / item).mkdir(mode=0o700)
    records = [{"path": MANIFEST_NAME, "size_bytes": len(manifest_bytes), "sha256": digest_bytes(manifest_bytes), "archive_mode": "0440"}, *files]
    for record in sorted(records, key=lambda value: value["path"]):
        source = archive / record["path"]
        encoded, before = read_regular(source, label=f"archive copy source {record['path']}")
        require(len(encoded) == record["size_bytes"] and digest_bytes(encoded) == record["sha256"], f"archive copy source changed: {record['path']}")
        destination = private / record["path"]
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            view = memoryview(encoded)
            while view:
                count = os.write(descriptor, view)
                require(count > 0, "short staged archive write")
                view = view[count:]
            os.fsync(descriptor)
            os.fchmod(descriptor, int(record["archive_mode"], 8))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        after = os.stat(source, follow_symlinks=False)
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            f"archive source raced: {record['path']}",
        )
    for item in reversed(directories):
        os.chmod(private / item, 0o550)
    os.chmod(private, 0o550)
    verify_staged(private, manifest_bytes, directories, files)


def write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        count = os.write(descriptor, view)
        require(count > 0, "short receipt write")
        view = view[count:]


def fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_receipt(stage_parent: Path, receipt_bytes: bytes) -> None:
    path = stage_parent / RECEIPT_NAME
    if path.exists():
        existing, metadata = read_regular(path, label="staged archive receipt")
        require(existing == receipt_bytes and stat.S_IMODE(metadata.st_mode) == 0o440, "staged archive receipt differs")
        return
    temporary = stage_parent / f".{RECEIPT_NAME}.{os.getpid()}.{secrets.token_hex(8)}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        write_all(descriptor, receipt_bytes)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o440)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
        fsync_directory(stage_parent)
    except FileExistsError:
        existing, _ = read_regular(path, label="staged archive receipt")
        require(existing == receipt_bytes, "concurrent staged archive receipt differs")
    finally:
        os.unlink(temporary)
        fsync_directory(stage_parent)


def verify_and_stage(args: argparse.Namespace) -> dict[str, Any]:
    archive = lexical(args.archive_root)
    archive_parent = secure_existing(
        archive.parent, directory=True, label="archive publication parent"
    )
    archive_parent_before = os.lstat(archive_parent)
    require(
        archive_parent_before.st_uid == os.geteuid()
        and not (archive_parent_before.st_mode & 0o022),
        "archive publication parent is not private and owned by effective user",
    )
    stage_parent = secure_existing(args.stage_parent, directory=True, label="stage parent")
    parent_stat = os.lstat(stage_parent)
    require(parent_stat.st_uid == os.geteuid(), "stage parent is not owned by effective user")
    require(
        not (parent_stat.st_mode & 0o022),
        "stage parent is group- or world-writable",
    )
    require(
        archive.parent != stage_parent and archive.parent not in stage_parent.parents
        and stage_parent not in archive.parent.parents,
        "archive and stage parents overlap",
    )
    archive_before = metadata_snapshot(archive)
    receipt, receipt_bytes, manifest, manifest_bytes, directories, files = verify_archive(
        archive,
        receipt_file_sha256=args.expected_receipt_file_sha256,
        receipt_sha256=args.expected_receipt_sha256,
        manifest_file_sha256=args.expected_manifest_file_sha256,
        manifest_sha256=args.expected_manifest_sha256,
        tree_sha256=args.expected_tree_sha256,
    )
    target = stage_parent / (PREFIX + args.expected_tree_sha256)
    lock_path = stage_parent / LOCK_NAME
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    private: Path | None = None
    try:
        require(stat.S_ISREG(os.fstat(lock_fd).st_mode), "bootstrap lock is not regular")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        content_names: list[str] = []
        for child in stage_parent.iterdir():
            if child.name.startswith(PREFIX):
                child_stat = os.lstat(child)
                require(
                    stat.S_ISDIR(child_stat.st_mode)
                    and not stat.S_ISLNK(child_stat.st_mode),
                    "stage parent content entry is unsafe",
                )
                content_names.append(child.name)
        content_names.sort()
        staged_receipt_path = stage_parent / RECEIPT_NAME
        if staged_receipt_path.exists():
            existing_receipt, existing_receipt_stat = read_regular(
                staged_receipt_path, label="existing staged archive receipt"
            )
            require(
                existing_receipt == receipt_bytes
                and stat.S_IMODE(existing_receipt_stat.st_mode) == 0o440,
                "stage parent is already bound to a different archive",
            )
            require(
                content_names == [target.name],
                "bound stage parent has missing or extra content archives",
            )
            verify_staged(target, manifest_bytes, directories, files)
            status = "reused_identical"
        else:
            require(
                content_names in ([], [target.name]),
                "unbound stage parent contains conflicting content archives",
            )
            if content_names:
                verify_staged(target, manifest_bytes, directories, files)
                status = "recovered_identical"
            else:
                private = stage_parent / f".{PREFIX}stage-tmp.{os.getpid()}.{secrets.token_hex(8)}"
                copy_stage(archive, private, manifest_bytes, directories, files)
                try:
                    os.rename(private, target)
                    fsync_directory(stage_parent)
                    private = None
                    status = "created"
                except FileExistsError:
                    verify_staged(target, manifest_bytes, directories, files)
                    remove_private(private)
                    private = None
                    status = "recovered_identical"
            publish_receipt(stage_parent, receipt_bytes)
        verify_staged(target, manifest_bytes, directories, files)
        require(
            metadata_snapshot(archive) == archive_before,
            "source archive mutated across verify and stage",
        )
        verified_after = verify_archive(
            archive,
            receipt_file_sha256=args.expected_receipt_file_sha256,
            receipt_sha256=args.expected_receipt_sha256,
            manifest_file_sha256=args.expected_manifest_file_sha256,
            manifest_sha256=args.expected_manifest_sha256,
            tree_sha256=args.expected_tree_sha256,
        )
        require(
            verified_after[1] == receipt_bytes
            and verified_after[3] == manifest_bytes,
            "archive receipt or manifest mutated across verify and stage",
        )
        archive_parent_after = os.lstat(archive_parent)
        require(
            (
                archive_parent_after.st_dev,
                archive_parent_after.st_ino,
                archive_parent_after.st_mode,
                archive_parent_after.st_uid,
                archive_parent_after.st_mtime_ns,
                archive_parent_after.st_ctime_ns,
            )
            == (
                archive_parent_before.st_dev,
                archive_parent_before.st_ino,
                archive_parent_before.st_mode,
                archive_parent_before.st_uid,
                archive_parent_before.st_mtime_ns,
                archive_parent_before.st_ctime_ns,
            ),
            "archive publication parent mutated across verify and stage",
        )
    finally:
        if private is not None:
            remove_private(private)
        os.close(lock_fd)
    return {
        "schema_version": "ep12.source_archive_bootstrap_stage.v1",
        "stage_status": status,
        "staged_root": str(target),
        "upstream_archived_root": str(archive),
        "archive_receipt_file_sha256": digest_bytes(receipt_bytes),
        "archive_receipt_sha256": receipt["receipt_sha256"],
        "manifest_file_sha256": digest_bytes(manifest_bytes),
        "manifest_sha256": manifest["manifest_sha256"],
        "tree_sha256": manifest["tree_sha256"],
        "complete": True,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("verify-and-stage")
    stage.add_argument("--archive-root", type=Path, required=True)
    stage.add_argument("--stage-parent", type=Path, required=True)
    stage.add_argument("--expected-receipt-file-sha256", required=True)
    stage.add_argument("--expected-receipt-sha256", required=True)
    stage.add_argument("--expected-manifest-file-sha256", required=True)
    stage.add_argument("--expected-manifest-sha256", required=True)
    stage.add_argument("--expected-tree-sha256", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        payload = verify_and_stage(args)
    except (BootstrapError, OSError, ValueError) as exc:
        print(f"source archive bootstrap failed: {exc}", file=sys.stderr)
        return 2
    print(canonical(payload).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
