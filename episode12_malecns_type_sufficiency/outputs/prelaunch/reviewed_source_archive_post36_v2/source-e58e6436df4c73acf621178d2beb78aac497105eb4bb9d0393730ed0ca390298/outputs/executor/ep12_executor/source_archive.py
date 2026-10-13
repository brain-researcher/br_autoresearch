"""Publish and verify a read-only, content-addressed EP12 source archive.

The caller supplies one explicitly reviewed source root and one durable archive
parent.  Every included source file is opened without following links and read
exactly once while it is copied.  A metadata-only second traversal rejects a
source tree that changed during the copy.  The completed private tree is made
read-only and then published with one same-parent directory rename.

Archive identity depends only on the declared source surface and bytes, not on
the machine-specific reviewed-source path.  Therefore an identical retry can
reuse the same content address, while a partial, mutated, or extra archive
fails closed and is never overwritten.

Publication is deliberately restricted to a clean episode-level source root:
SEARCH_POLICY.yaml plus outputs/executor/**, with no other episode data or
result directories.  Callers must assemble and review that isolated source
root before publication; passing the live EP12 episode root fails closed before
any source file bytes are copied.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
from typing import Any, Mapping, Sequence

from .policy import canonical_json, digest_object
from .secure_transaction import open_regular_lock_at, open_secure_directory


SOURCE_ARCHIVE_SCHEMA = "ep12.reviewed_source_archive_manifest.v1"
SOURCE_ARCHIVE_RECEIPT_SCHEMA = "ep12.reviewed_source_archive_receipt.v1"
SOURCE_ARCHIVE_MANIFEST = "source_archive_manifest.json"
SOURCE_ARCHIVE_RECEIPT = "source_archive_receipt.json"
SOURCE_ARCHIVE_PREFIX = "source-"
SOURCE_ARCHIVE_LOCK = ".source_archive_publish.lock"

_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".nox",
        ".pycache-check",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        "__pycache__",
        ".temp",
        "temp",
        "tmp",
        "patch",
    }
)
_EXCLUDED_FILE_NAMES = frozenset(
    {".coverage", ".DS_Store", "patch", "temp", "tmp"}
)
_EXCLUDED_SUFFIXES = (
    ".diff",
    ".orig",
    ".patch",
    ".pyc",
    ".pyo",
    ".rej",
    ".swo",
    ".swp",
    ".temp",
    ".tmp",
    "~",
)
_HEX_DIGITS = frozenset("0123456789abcdef")
_MANIFEST_FIELDS = {
    "schema_version",
    "archive_kind",
    "surface_rule",
    "directory_count",
    "file_count",
    "total_source_bytes",
    "directories",
    "files",
    "tree_sha256",
    "source_files_read_once",
    "symlinks_allowed",
    "complete",
    "manifest_sha256",
}
_RECEIPT_FIELDS = {
    "schema_version",
    "archive_name",
    "manifest_name",
    "manifest_file_sha256",
    "manifest_sha256",
    "tree_sha256",
    "directory_count",
    "file_count",
    "total_source_bytes",
    "complete",
    "receipt_sha256",
}


class SourceArchiveError(RuntimeError):
    """A reviewed source or durable archive is unsafe or inconsistent."""


@dataclass(frozen=True)
class _CopiedArchive:
    manifest: dict[str, Any]
    manifest_bytes: bytes
    manifest_file_sha256: str
    tree_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceArchiveError(message)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _HEX_DIGITS
    )


def _lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _surface_rule() -> dict[str, Any]:
    return {
        "include": ["SEARCH_POLICY.yaml", "outputs/executor/**"],
        "source_root_layout": "exact_clean_episode_source_v1",
        "excluded_directory_names": sorted(_EXCLUDED_DIRECTORY_NAMES),
        "excluded_file_names": sorted(_EXCLUDED_FILE_NAMES),
        "excluded_suffixes": list(_EXCLUDED_SUFFIXES),
        "reserved_archive_manifest": SOURCE_ARCHIVE_MANIFEST,
        "symlinks_allowed": False,
        "special_files_allowed": False,
    }


def _excluded_directory(name: str) -> bool:
    return name in _EXCLUDED_DIRECTORY_NAMES or name.endswith(
        (".diff", ".orig", ".rej", ".patch", ".temp", ".tmp")
    )


def _excluded_file(name: str) -> bool:
    return (
        name in _EXCLUDED_FILE_NAMES
        or name.startswith(".tmp.")
        or name.endswith(_EXCLUDED_SUFFIXES)
    )


_FORBIDDEN_SOURCE_COMPONENTS = frozenset({"inputs", "outcomes", "prelaunch"})


def _forbidden_source_component(name: str) -> bool:
    return (
        name in _FORBIDDEN_SOURCE_COMPONENTS
        or name.startswith("development_run")
    )


def _scan_clean_executor_source(directory_fd: int, relative_parent: str) -> int:
    file_count = 0
    try:
        names = sorted(os.listdir(directory_fd))
    except OSError as exc:
        raise SourceArchiveError(
            f"cannot enumerate reviewed executor source {relative_parent}: {exc}"
        ) from exc
    for name in names:
        relative = _relative_text(relative_parent, name)
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
            raise SourceArchiveError(f"reviewed source contains a symlink: {relative}")
        _require(
            not _forbidden_source_component(name),
            f"reviewed source contains a forbidden episode artifact: {relative}",
        )
        if stat.S_ISDIR(metadata.st_mode):
            _require(
                not _excluded_directory(name),
                f"reviewed source contains an excluded directory: {relative}",
            )
            child = os.open(
                name,
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                _require(
                    _same_inode(metadata, os.fstat(child)),
                    f"source directory changed while opening: {relative}",
                )
                file_count += _scan_clean_executor_source(child, relative)
            finally:
                os.close(child)
        elif stat.S_ISREG(metadata.st_mode):
            _require(
                not _excluded_file(name)
                and name not in {SOURCE_ARCHIVE_MANIFEST, SOURCE_ARCHIVE_RECEIPT},
                f"reviewed source contains an excluded artifact: {relative}",
            )
            file_count += 1
        else:
            raise SourceArchiveError(
                f"reviewed source contains a special file: {relative}"
            )
    return file_count


def _assert_clean_episode_source_root(root_fd: int) -> None:
    names = set(os.listdir(root_fd))
    _require(
        names == {"SEARCH_POLICY.yaml", "outputs"},
        "reviewed source root must contain exactly SEARCH_POLICY.yaml and outputs/",
    )
    policy_metadata = os.stat(
        "SEARCH_POLICY.yaml", dir_fd=root_fd, follow_symlinks=False
    )
    outputs_metadata = os.stat("outputs", dir_fd=root_fd, follow_symlinks=False)
    _require(
        stat.S_ISREG(policy_metadata.st_mode),
        "reviewed SEARCH_POLICY.yaml must be one regular file",
    )
    _require(
        stat.S_ISDIR(outputs_metadata.st_mode),
        "reviewed outputs entry must be a directory",
    )
    outputs_fd = os.open(
        "outputs",
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=root_fd,
    )
    try:
        _require(
            _same_inode(outputs_metadata, os.fstat(outputs_fd)),
            "reviewed outputs directory changed while opening",
        )
        _require(
            set(os.listdir(outputs_fd)) == {"executor"},
            "reviewed outputs/ must contain exactly executor/",
        )
        executor_metadata = os.stat(
            "executor", dir_fd=outputs_fd, follow_symlinks=False
        )
        _require(
            stat.S_ISDIR(executor_metadata.st_mode),
            "reviewed outputs/executor entry must be a directory",
        )
        executor_fd = os.open(
            "executor",
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=outputs_fd,
        )
        try:
            _require(
                _same_inode(executor_metadata, os.fstat(executor_fd)),
                "reviewed outputs/executor directory changed while opening",
            )
            _require(
                _scan_clean_executor_source(executor_fd, "outputs/executor") > 0,
                "reviewed outputs/executor source must contain at least one file",
            )
        finally:
            os.close(executor_fd)
    finally:
        os.close(outputs_fd)


def _assert_manifest_episode_surface(
    directories: Sequence[str], files: Sequence[Mapping[str, Any]]
) -> None:
    directory_set = set(directories)
    _require(
        {"outputs", "outputs/executor"} <= directory_set,
        "source archive lacks the exact episode executor directories",
    )
    for relative in directories:
        _require(
            relative in {"outputs", "outputs/executor"}
            or relative.startswith("outputs/executor/"),
            f"source archive directory is outside the executor allowlist: {relative}",
        )
        _require(
            not any(
                _forbidden_source_component(component)
                for component in PurePosixPath(relative).parts
            ),
            f"source archive directory is a forbidden episode artifact: {relative}",
        )
    file_paths = [str(record["path"]) for record in files]
    _require(
        "SEARCH_POLICY.yaml" in file_paths
        and any(path.startswith("outputs/executor/") for path in file_paths),
        "source archive lacks SEARCH_POLICY.yaml or executor source files",
    )
    for relative in file_paths:
        _require(
            relative == "SEARCH_POLICY.yaml"
            or relative.startswith("outputs/executor/"),
            f"source archive file is outside the executor allowlist: {relative}",
        )
        _require(
            not any(
                _forbidden_source_component(component)
                for component in PurePosixPath(relative).parts
            ),
            f"source archive file is a forbidden episode artifact: {relative}",
        )


def _relative_text(parent: str, name: str) -> str:
    _require(
        name not in {"", ".", ".."}
        and "/" not in name
        and "\x00" not in name,
        f"unsafe source entry name: {name!r}",
    )
    return name if not parent else f"{parent}/{name}"


def _stat_identity(value: os.stat_result, kind: str) -> tuple[Any, ...]:
    return (
        kind,
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        _require(written > 0, "short write while constructing source archive")
        view = view[written:]


def _copy_regular_once(
    *,
    source_directory_fd: int,
    destination_directory_fd: int,
    name: str,
    initial: os.stat_result,
    relative: str,
) -> dict[str, Any]:
    source_fd = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=source_directory_fd,
    )
    archive_mode = "0550" if initial.st_mode & 0o111 else "0440"
    destination_fd = -1
    digest = hashlib.sha256()
    byte_count = 0
    try:
        opened = os.fstat(source_fd)
        _require(
            stat.S_ISREG(opened.st_mode)
            and _same_inode(initial, opened),
            f"source file changed while opening: {relative}",
        )
        destination_fd = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=destination_directory_fd,
        )
        while True:
            block = os.read(source_fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            byte_count += len(block)
            _write_all(destination_fd, block)
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        named = os.stat(name, dir_fd=source_directory_fd, follow_symlinks=False)
        expected = _stat_identity(initial, "file")
        _require(
            expected == _stat_identity(opened, "file")
            == _stat_identity(after, "file")
            == _stat_identity(named, "file")
            and byte_count == initial.st_size,
            f"source file mutated during its one-time copy: {relative}",
        )
    finally:
        os.close(source_fd)
        if destination_fd >= 0:
            os.close(destination_fd)
    return {
        "path": relative,
        "size_bytes": byte_count,
        "sha256": digest.hexdigest(),
        "archive_mode": archive_mode,
    }


def _snapshot_and_copy(
    *,
    source_fd: int,
    destination_fd: int | None,
    relative_parent: str,
    included_surface: bool,
    state: dict[str, tuple[Any, ...]],
    directories: list[str],
    files: list[dict[str, Any]],
) -> None:
    try:
        names = sorted(os.listdir(source_fd))
    except OSError as exc:
        raise SourceArchiveError(
            f"cannot enumerate reviewed source directory {relative_parent or '.'}: {exc}"
        ) from exc
    for name in names:
        relative = _relative_text(relative_parent, name)
        try:
            metadata = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        except OSError as exc:
            raise SourceArchiveError(
                f"cannot inspect reviewed source entry {relative}: {exc}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise SourceArchiveError(f"reviewed source contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            state[relative] = _stat_identity(metadata, "directory")
            child_source = os.open(
                name,
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=source_fd,
            )
            child_destination = -1
            child_included = included_surface and not _excluded_directory(name)
            try:
                _require(
                    _same_inode(metadata, os.fstat(child_source)),
                    f"source directory changed while opening: {relative}",
                )
                if child_included:
                    _require(destination_fd is not None, "missing archive destination")
                    os.mkdir(name, 0o750, dir_fd=destination_fd)
                    child_destination = os.open(
                        name,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=destination_fd,
                    )
                    directories.append(relative)
                _snapshot_and_copy(
                    source_fd=child_source,
                    destination_fd=(
                        child_destination if child_destination >= 0 else None
                    ),
                    relative_parent=relative,
                    included_surface=child_included,
                    state=state,
                    directories=directories,
                    files=files,
                )
                if child_destination >= 0:
                    os.fsync(child_destination)
            finally:
                os.close(child_source)
                if child_destination >= 0:
                    os.close(child_destination)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise SourceArchiveError(
                f"reviewed source contains a special file: {relative}"
            )
        state[relative] = _stat_identity(metadata, "file")
        if not included_surface or _excluded_file(name):
            continue
        _require(
            relative != SOURCE_ARCHIVE_MANIFEST,
            f"reviewed source uses reserved archive path: {relative}",
        )
        _require(destination_fd is not None, "missing archive destination")
        files.append(
            _copy_regular_once(
                source_directory_fd=source_fd,
                destination_directory_fd=destination_fd,
                name=name,
                initial=metadata,
                relative=relative,
            )
        )


def _metadata_state(source_root: Path) -> dict[str, tuple[Any, ...]]:
    root, descriptor = open_secure_directory(
        source_root, root=source_root, create=False
    )
    state: dict[str, tuple[Any, ...]] = {
        ".": _stat_identity(os.fstat(descriptor), "directory")
    }
    try:
        _snapshot_and_copy(
            source_fd=descriptor,
            destination_fd=None,
            relative_parent="",
            included_surface=False,
            state=state,
            directories=[],
            files=[],
        )
    finally:
        os.close(descriptor)
    _require(root == source_root, "reviewed source root identity changed")
    return state


def _manifest_from_records(
    directories: Sequence[str], files: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    normalized_directories = sorted(str(value) for value in directories)
    normalized_files = sorted(
        (dict(value) for value in files), key=lambda value: str(value["path"])
    )
    _require(
        len(normalized_directories) == len(set(normalized_directories)),
        "reviewed source repeats a directory path",
    )
    _assert_manifest_episode_surface(normalized_directories, normalized_files)
    file_paths = [str(value["path"]) for value in normalized_files]
    _require(
        len(file_paths) == len(set(file_paths)),
        "reviewed source repeats a file path",
    )
    tree_payload = {
        "surface_rule": _surface_rule(),
        "directories": normalized_directories,
        "files": normalized_files,
    }
    tree_sha256 = digest_object(tree_payload)
    core = {
        "schema_version": SOURCE_ARCHIVE_SCHEMA,
        "archive_kind": "content_addressed_reviewed_executor_source",
        "surface_rule": _surface_rule(),
        "directory_count": len(normalized_directories),
        "file_count": len(normalized_files),
        "total_source_bytes": sum(
            int(record["size_bytes"]) for record in normalized_files
        ),
        "directories": normalized_directories,
        "files": normalized_files,
        "tree_sha256": tree_sha256,
        "source_files_read_once": True,
        "symlinks_allowed": False,
        "complete": True,
    }
    return {**core, "manifest_sha256": digest_object(core)}


def _write_manifest(directory_fd: int, manifest: Mapping[str, Any]) -> bytes:
    encoded = (canonical_json(manifest) + "\n").encode("utf-8")
    descriptor = os.open(
        SOURCE_ARCHIVE_MANIFEST,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)
    return encoded


def _chmod_nofollow(path: Path, mode: int, *, directory: bool) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    if directory:
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        kind_matches = (
            stat.S_ISDIR(opened.st_mode)
            if directory
            else stat.S_ISREG(opened.st_mode)
        )
        _require(
            kind_matches
            and _same_inode(opened, named)
            and not stat.S_ISLNK(named.st_mode),
            f"archive path changed while setting its mode: {path}",
        )
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _set_archive_read_only(root: Path, manifest: Mapping[str, Any]) -> None:
    modes = {
        str(record["path"]): int(str(record["archive_mode"]), 8)
        for record in manifest["files"]
    }
    for current, directory_names, file_names in os.walk(
        root, topdown=False, followlinks=False
    ):
        current_path = Path(current)
        _require(not current_path.is_symlink(), "archive construction produced a symlink")
        for name in file_names:
            path = current_path / name
            _require(path.is_file() and not path.is_symlink(), "archive file became unsafe")
            relative = path.relative_to(root).as_posix()
            mode = 0o440 if relative == SOURCE_ARCHIVE_MANIFEST else modes[relative]
            _chmod_nofollow(path, mode, directory=False)
        for name in directory_names:
            path = current_path / name
            _require(path.is_dir() and not path.is_symlink(), "archive directory became unsafe")
            _chmod_nofollow(path, 0o550, directory=True)
        _chmod_nofollow(current_path, 0o550, directory=True)


def _canonical_relative(value: Any, *, label: str) -> str:
    _require(isinstance(value, str) and value, f"{label} path is invalid")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and path.as_posix() == value
        and all(component not in {"", ".", ".."} for component in path.parts),
        f"{label} path is not canonical: {value!r}",
    )
    return value


def _open_relative_file(root_fd: int, relative: str) -> tuple[int, int]:
    parts = PurePosixPath(relative).parts
    directory_fd = os.dup(root_fd)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    try:
        for component in parts[:-1]:
            child = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        return descriptor, directory_fd
    except Exception:
        os.close(directory_fd)
        raise


def _read_relative_bytes(root_fd: int, relative: str) -> tuple[bytes, os.stat_result]:
    descriptor, directory_fd = _open_relative_file(root_fd, relative)
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"archive entry is not regular: {relative}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        named = os.stat(
            PurePosixPath(relative).name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        _require(
            stat.S_ISREG(named.st_mode) and _same_inode(metadata, named),
            f"archive entry changed while reading: {relative}",
        )
        return b"".join(chunks), metadata
    finally:
        os.close(descriptor)
        os.close(directory_fd)


def _scan_archive(
    root_fd: int, relative_parent: str = ""
) -> tuple[dict[str, os.stat_result], dict[str, os.stat_result]]:
    directories: dict[str, os.stat_result] = {}
    files: dict[str, os.stat_result] = {}
    for name in sorted(os.listdir(root_fd)):
        relative = _relative_text(relative_parent, name)
        metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
            raise SourceArchiveError(f"archive contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            directories[relative] = metadata
            child = os.open(
                name,
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_fd,
            )
            try:
                child_directories, child_files = _scan_archive(child, relative)
            finally:
                os.close(child)
            directories.update(child_directories)
            files.update(child_files)
        elif stat.S_ISREG(metadata.st_mode):
            files[relative] = metadata
        else:
            raise SourceArchiveError(f"archive contains a special file: {relative}")
    return directories, files


def _parse_manifest(encoded: bytes) -> dict[str, Any]:
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceArchiveError("source archive manifest is malformed") from exc
    _require(isinstance(value, Mapping), "source archive manifest is not an object")
    manifest = dict(value)
    _require(
        encoded == (canonical_json(manifest) + "\n").encode("utf-8"),
        "source archive manifest is not canonical JSON",
    )
    _require(set(manifest) == _MANIFEST_FIELDS, "source archive manifest schema changed")
    core = {key: item for key, item in manifest.items() if key != "manifest_sha256"}
    _require(
        manifest.get("schema_version") == SOURCE_ARCHIVE_SCHEMA
        and manifest.get("archive_kind")
        == "content_addressed_reviewed_executor_source"
        and manifest.get("surface_rule") == _surface_rule()
        and manifest.get("source_files_read_once") is True
        and manifest.get("symlinks_allowed") is False
        and manifest.get("complete") is True
        and manifest.get("manifest_sha256") == digest_object(core),
        "source archive manifest identity or self-hash mismatch",
    )
    return manifest


def _validate_manifest_surface(
    manifest: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    raw_directories = manifest.get("directories")
    raw_files = manifest.get("files")
    _require(isinstance(raw_directories, list), "archive directory list is invalid")
    _require(isinstance(raw_files, list), "archive file list is invalid")
    directories = [
        _canonical_relative(value, label="archive directory")
        for value in raw_directories
    ]
    _require(
        type(manifest.get("directory_count")) is int
        and manifest["directory_count"] >= 0
        and directories == sorted(set(directories))
        and manifest.get("directory_count") == len(directories),
        "archive directories are not exact, unique, and sorted",
    )
    for directory in directories:
        _require(
            not any(
                _excluded_directory(component)
                for component in PurePosixPath(directory).parts
            ),
            f"archive includes an excluded source directory: {directory}",
        )
    files: list[dict[str, Any]] = []
    prior = ""
    for index, raw in enumerate(raw_files):
        _require(isinstance(raw, Mapping), f"archive file record {index} is invalid")
        record = dict(raw)
        _require(
            set(record) == {"path", "size_bytes", "sha256", "archive_mode"},
            f"archive file record {index} schema changed",
        )
        relative = _canonical_relative(record.get("path"), label="archive file")
        _require(relative > prior, "archive file paths are not strictly sorted")
        prior = relative
        _require(
            relative != SOURCE_ARCHIVE_MANIFEST
            and not any(
                _excluded_directory(component)
                for component in PurePosixPath(relative).parts[:-1]
            )
            and not _excluded_file(PurePosixPath(relative).name),
            f"archive includes an excluded source artifact: {relative}",
        )
        _require(
            type(record.get("size_bytes")) is int and record["size_bytes"] >= 0,
            f"archive size is invalid: {relative}",
        )
        _require(_is_sha256(record.get("sha256")), f"archive digest is invalid: {relative}")
        _require(
            record.get("archive_mode") in {"0440", "0550"},
            f"archive mode is invalid: {relative}",
        )
        files.append(record)
    _require(
        type(manifest.get("file_count")) is int
        and manifest["file_count"] >= 0
        and type(manifest.get("total_source_bytes")) is int
        and manifest["total_source_bytes"] >= 0
        and manifest.get("file_count") == len(files)
        and manifest.get("total_source_bytes")
        == sum(record["size_bytes"] for record in files),
        "archive file count or byte total mismatch",
    )
    _require(
        _is_sha256(manifest.get("tree_sha256")),
        "archive tree digest is invalid",
    )
    _assert_manifest_episode_surface(directories, files)
    directory_set = set(directories)
    for directory in directories:
        parent = PurePosixPath(directory).parent.as_posix()
        _require(
            parent == "." or parent in directory_set,
            f"archive directory lacks a declared parent: {directory}",
        )
    for record in files:
        parent = PurePosixPath(record["path"]).parent.as_posix()
        _require(
            parent == "." or parent in directory_set,
            f"archive file lacks a declared parent: {record['path']}",
        )
    return directories, files


def _receipt_from_copy(copied: _CopiedArchive) -> dict[str, Any]:
    core = {
        "schema_version": SOURCE_ARCHIVE_RECEIPT_SCHEMA,
        "archive_name": f"{SOURCE_ARCHIVE_PREFIX}{copied.tree_sha256}",
        "manifest_name": SOURCE_ARCHIVE_MANIFEST,
        "manifest_file_sha256": copied.manifest_file_sha256,
        "manifest_sha256": copied.manifest["manifest_sha256"],
        "tree_sha256": copied.tree_sha256,
        "directory_count": copied.manifest["directory_count"],
        "file_count": copied.manifest["file_count"],
        "total_source_bytes": copied.manifest["total_source_bytes"],
        "complete": True,
    }
    return {**core, "receipt_sha256": digest_object(core)}


def _receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    return (canonical_json(receipt) + "\n").encode("utf-8")


def _parse_receipt(encoded: bytes) -> dict[str, Any]:
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceArchiveError("source archive receipt is malformed") from exc
    _require(isinstance(value, Mapping), "source archive receipt is not an object")
    receipt = dict(value)
    _require(
        encoded == _receipt_bytes(receipt),
        "source archive receipt is not canonical JSON",
    )
    _require(set(receipt) == _RECEIPT_FIELDS, "source archive receipt schema changed")
    core = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    for key in (
        "manifest_file_sha256",
        "manifest_sha256",
        "tree_sha256",
        "receipt_sha256",
    ):
        _require(_is_sha256(receipt.get(key)), f"source archive receipt {key} is invalid")
    for key in ("directory_count", "file_count", "total_source_bytes"):
        _require(
            type(receipt.get(key)) is int and receipt[key] >= 0,
            f"source archive receipt {key} is invalid",
        )
    _require(
        receipt.get("schema_version") == SOURCE_ARCHIVE_RECEIPT_SCHEMA
        and receipt.get("manifest_name") == SOURCE_ARCHIVE_MANIFEST
        and receipt.get("archive_name")
        == f"{SOURCE_ARCHIVE_PREFIX}{receipt['tree_sha256']}"
        and "/" not in receipt["archive_name"]
        and receipt.get("complete") is True
        and receipt.get("receipt_sha256") == digest_object(core),
        "source archive receipt identity or self-hash mismatch",
    )
    return receipt


def _read_receipt_at(parent_fd: int) -> tuple[dict[str, Any], bytes, str]:
    try:
        encoded, metadata = _read_relative_bytes(parent_fd, SOURCE_ARCHIVE_RECEIPT)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise SourceArchiveError(
            f"cannot safely read source archive receipt: {exc}"
        ) from exc
    _require(
        stat.S_IMODE(metadata.st_mode) == 0o440,
        "source archive receipt mode changed",
    )
    return _parse_receipt(encoded), encoded, _sha_bytes(encoded)


def _content_archive_names(parent_fd: int) -> list[str]:
    names: list[str] = []
    try:
        entries = sorted(os.listdir(parent_fd))
    except OSError as exc:
        raise SourceArchiveError(f"cannot enumerate archive parent: {exc}") from exc
    for name in entries:
        if not name.startswith(SOURCE_ARCHIVE_PREFIX):
            continue
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        _require(
            stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
            f"content-addressed archive entry is not a directory: {name}",
        )
        digest = name[len(SOURCE_ARCHIVE_PREFIX) :]
        _require(
            _is_sha256(digest),
            f"content-addressed archive directory name is invalid: {name}",
        )
        names.append(name)
    return names


def _publish_receipt_once(parent_fd: int, encoded: bytes) -> None:
    temporary = f".source-receipt-tmp.{os.getpid()}.{secrets.token_hex(8)}"
    descriptor = -1
    linked = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o440)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(
                temporary,
                SOURCE_ARCHIVE_RECEIPT,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            linked = True
        except FileExistsError as exc:
            raise SourceArchiveError(
                "source archive receipt appeared during exact-once publication"
            ) from exc
        os.fsync(parent_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        if linked:
            os.fsync(parent_fd)


def _verify_archive(
    archive_root: Path,
    *,
    expected_manifest_file_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_tree_sha256: str | None = None,
    require_content_address_name: bool = True,
) -> dict[str, Any]:
    archive_root = _lexical(archive_root)
    root, root_fd = open_secure_directory(
        archive_root, root=archive_root, create=False
    )
    try:
        root_mode = stat.S_IMODE(os.fstat(root_fd).st_mode)
        _require(root_mode == 0o550, "source archive root mode changed")
        manifest_bytes, manifest_stat = _read_relative_bytes(
            root_fd, SOURCE_ARCHIVE_MANIFEST
        )
        _require(
            stat.S_IMODE(manifest_stat.st_mode) == 0o440,
            "source archive manifest mode changed",
        )
        manifest_file_sha256 = _sha_bytes(manifest_bytes)
        manifest = _parse_manifest(manifest_bytes)
        directories, files = _validate_manifest_surface(manifest)
        if expected_manifest_file_sha256 is not None:
            _require(
                manifest_file_sha256 == expected_manifest_file_sha256,
                "source archive manifest file hash mismatch",
            )
        if expected_manifest_sha256 is not None:
            _require(
                manifest["manifest_sha256"] == expected_manifest_sha256,
                "source archive manifest semantic hash mismatch",
            )
        if expected_tree_sha256 is not None:
            _require(
                manifest["tree_sha256"] == expected_tree_sha256,
                "source archive tree hash mismatch",
            )
        if require_content_address_name:
            _require(
                root.name == f"{SOURCE_ARCHIVE_PREFIX}{manifest['tree_sha256']}",
                "source archive directory name is not its content address",
            )
        observed_directories, observed_files = _scan_archive(root_fd)
        expected_file_paths = {record["path"] for record in files}
        _require(
            set(observed_directories) == set(directories),
            "source archive has missing or extra directories",
        )
        _require(
            set(observed_files) == expected_file_paths | {SOURCE_ARCHIVE_MANIFEST},
            "source archive has missing or extra files",
        )
        for directory, metadata in observed_directories.items():
            _require(
                stat.S_IMODE(metadata.st_mode) == 0o550,
                f"source archive directory mode changed: {directory}",
            )
        recomputed_files: list[dict[str, Any]] = []
        by_path = {record["path"]: record for record in files}
        for relative in sorted(expected_file_paths):
            content, metadata = _read_relative_bytes(root_fd, relative)
            expected = by_path[relative]
            _require(
                len(content) == expected["size_bytes"]
                and _sha_bytes(content) == expected["sha256"]
                and stat.S_IMODE(metadata.st_mode)
                == int(expected["archive_mode"], 8),
                f"source archive file changed: {relative}",
            )
            recomputed_files.append(dict(expected))
        tree_payload = {
            "surface_rule": _surface_rule(),
            "directories": directories,
            "files": recomputed_files,
        }
        _require(
            digest_object(tree_payload) == manifest["tree_sha256"],
            "source archive tree digest mismatch",
        )
        return {
            "archived_root": str(root),
            "manifest_path": str(root / SOURCE_ARCHIVE_MANIFEST),
            "manifest_file_sha256": manifest_file_sha256,
            "manifest_sha256": manifest["manifest_sha256"],
            "tree_sha256": manifest["tree_sha256"],
            "directory_count": len(directories),
            "file_count": len(files),
            "total_source_bytes": manifest["total_source_bytes"],
            "read_only": True,
            "complete": True,
        }
    finally:
        os.close(root_fd)


def _validate_expected_digests(
    *,
    expected_receipt_file_sha256: str | None,
    expected_receipt_sha256: str | None,
    expected_manifest_file_sha256: str | None,
    expected_manifest_sha256: str | None,
    expected_tree_sha256: str | None,
) -> None:
    for value, label in (
        (expected_receipt_file_sha256, "expected receipt file sha256"),
        (expected_receipt_sha256, "expected receipt semantic sha256"),
        (expected_manifest_file_sha256, "expected manifest file sha256"),
        (expected_manifest_sha256, "expected manifest semantic sha256"),
        (expected_tree_sha256, "expected tree sha256"),
    ):
        if value is not None:
            _require(_is_sha256(value), f"{label} is invalid")


def verify_source_archive_publication(
    archive_parent: Path,
    *,
    expected_receipt_file_sha256: str | None = None,
    expected_receipt_sha256: str | None = None,
    expected_manifest_file_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_tree_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify the one archive permanently bound to an archive parent."""

    _validate_expected_digests(
        expected_receipt_file_sha256=expected_receipt_file_sha256,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_tree_sha256=expected_tree_sha256,
    )
    _require(archive_parent.is_absolute(), "archive parent must be absolute")
    parent = _lexical(archive_parent)
    opened_parent, parent_fd = open_secure_directory(
        parent, root=parent, create=False
    )
    _require(opened_parent == parent, "archive parent changed")
    parent_metadata = os.fstat(parent_fd)
    _require(
        parent_metadata.st_uid == os.geteuid()
        and not (parent_metadata.st_mode & 0o022),
        "archive parent must be effective-user-owned and not group/world writable",
    )
    try:
        receipt, receipt_encoded, receipt_file_sha256 = _read_receipt_at(parent_fd)
        _require(
            _content_archive_names(parent_fd) == [receipt["archive_name"]],
            "archive parent is not bound to exactly one content-addressed archive",
        )
    finally:
        os.close(parent_fd)
    if expected_receipt_file_sha256 is not None:
        _require(
            receipt_file_sha256 == expected_receipt_file_sha256,
            "source archive receipt file hash mismatch",
        )
    if expected_receipt_sha256 is not None:
        _require(
            receipt["receipt_sha256"] == expected_receipt_sha256,
            "source archive receipt semantic hash mismatch",
        )
    for expected, key, label in (
        (
            expected_manifest_file_sha256,
            "manifest_file_sha256",
            "source archive manifest file hash mismatch",
        ),
        (
            expected_manifest_sha256,
            "manifest_sha256",
            "source archive manifest semantic hash mismatch",
        ),
        (expected_tree_sha256, "tree_sha256", "source archive tree hash mismatch"),
    ):
        if expected is not None:
            _require(receipt[key] == expected, label)
    archive_root = parent / receipt["archive_name"]
    verified = _verify_archive(
        archive_root,
        expected_manifest_file_sha256=receipt["manifest_file_sha256"],
        expected_manifest_sha256=receipt["manifest_sha256"],
        expected_tree_sha256=receipt["tree_sha256"],
        require_content_address_name=True,
    )
    _require(
        verified["directory_count"] == receipt["directory_count"]
        and verified["file_count"] == receipt["file_count"]
        and verified["total_source_bytes"] == receipt["total_source_bytes"],
        "source archive receipt counts differ from its manifest",
    )
    return {
        **verified,
        "archive_parent": str(parent),
        "archive_receipt_path": str(parent / SOURCE_ARCHIVE_RECEIPT),
        "archive_receipt_file_sha256": receipt_file_sha256,
        "archive_receipt_sha256": receipt["receipt_sha256"],
        "parent_first_publication_bound": True,
    }


def verify_source_archive(
    archive_root: Path,
    *,
    expected_receipt_file_sha256: str | None = None,
    expected_receipt_sha256: str | None = None,
    expected_manifest_file_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_tree_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify an exact archive and its immutable parent publication receipt."""

    _require(archive_root.is_absolute(), "archive root must be absolute")
    requested = _lexical(archive_root)
    verified = verify_source_archive_publication(
        requested.parent,
        expected_receipt_file_sha256=expected_receipt_file_sha256,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_tree_sha256=expected_tree_sha256,
    )
    _require(
        Path(verified["archived_root"]) == requested,
        "requested archive root is not the parent-bound archive",
    )
    return verified


def _cleanup_private_tree(path: Path) -> None:
    if not path.exists():
        return
    for current, directory_names, file_names in os.walk(
        path, topdown=False, followlinks=False
    ):
        current_path = Path(current)
        _chmod_nofollow(current_path, 0o700, directory=True)
        for name in file_names:
            file_path = current_path / name
            _require(not file_path.is_symlink(), "private archive cleanup found a symlink")
            os.unlink(file_path)
        for name in directory_names:
            directory_path = current_path / name
            _require(not directory_path.is_symlink(), "private archive cleanup found a symlink")
            _chmod_nofollow(directory_path, 0o700, directory=True)
            os.rmdir(directory_path)
    os.rmdir(path)


def _copy_to_private_archive(source_root: Path, private_root: Path) -> _CopiedArchive:
    _, source_fd = open_secure_directory(
        source_root, root=source_root, create=False
    )
    _, destination_fd = open_secure_directory(
        private_root, root=private_root, create=False
    )
    before_state: dict[str, tuple[Any, ...]] = {
        ".": _stat_identity(os.fstat(source_fd), "directory")
    }
    _assert_clean_episode_source_root(source_fd)
    directories: list[str] = []
    files: list[dict[str, Any]] = []
    try:
        _snapshot_and_copy(
            source_fd=source_fd,
            destination_fd=destination_fd,
            relative_parent="",
            included_surface=True,
            state=before_state,
            directories=directories,
            files=files,
        )
        manifest = _manifest_from_records(directories, files)
        manifest_bytes = _write_manifest(destination_fd, manifest)
    finally:
        os.close(source_fd)
        os.close(destination_fd)
    after_state = _metadata_state(source_root)
    _require(
        before_state == after_state,
        "reviewed source tree mutated while its one-time archive copy was built",
    )
    _set_archive_read_only(private_root, manifest)
    verification = _verify_archive(
        private_root,
        expected_manifest_file_sha256=_sha_bytes(manifest_bytes),
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_tree_sha256=manifest["tree_sha256"],
        require_content_address_name=False,
    )
    _require(
        verification["manifest_file_sha256"] == _sha_bytes(manifest_bytes),
        "private source archive verification changed its manifest identity",
    )
    return _CopiedArchive(
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        manifest_file_sha256=_sha_bytes(manifest_bytes),
        tree_sha256=manifest["tree_sha256"],
    )


def _open_relative_directory_fd(root_fd: int, relative: str) -> int:
    if relative in {"", "."}:
        return os.dup(root_fd)
    canonical = _canonical_relative(relative, label="archive directory")
    descriptor = os.dup(root_fd)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    try:
        for component in PurePosixPath(canonical).parts:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _copy_verified_archive_file_once(
    *,
    source_root_fd: int,
    destination_root_fd: int,
    record: Mapping[str, Any],
) -> None:
    relative = str(record["path"])
    source_fd, source_parent_fd = _open_relative_file(source_root_fd, relative)
    parent = PurePosixPath(relative).parent.as_posix()
    destination_parent_fd = _open_relative_directory_fd(
        destination_root_fd, parent
    )
    destination_fd = -1
    digest = hashlib.sha256()
    byte_count = 0
    try:
        before = os.fstat(source_fd)
        named_before = os.stat(
            PurePosixPath(relative).name,
            dir_fd=source_parent_fd,
            follow_symlinks=False,
        )
        expected_mode = int(str(record["archive_mode"]), 8)
        _require(
            stat.S_ISREG(before.st_mode)
            and _same_inode(before, named_before)
            and stat.S_IMODE(before.st_mode) == expected_mode,
            f"verified source archive file changed before staging: {relative}",
        )
        destination_fd = os.open(
            PurePosixPath(relative).name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=destination_parent_fd,
        )
        while True:
            block = os.read(source_fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            byte_count += len(block)
            _write_all(destination_fd, block)
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        named_after = os.stat(
            PurePosixPath(relative).name,
            dir_fd=source_parent_fd,
            follow_symlinks=False,
        )
        _require(
            _stat_identity(before, "file")
            == _stat_identity(after, "file")
            == _stat_identity(named_after, "file")
            and byte_count == record["size_bytes"]
            and digest.hexdigest() == record["sha256"],
            f"verified source archive file mutated while staging: {relative}",
        )
        os.fsync(destination_parent_fd)
    finally:
        os.close(source_fd)
        os.close(source_parent_fd)
        os.close(destination_parent_fd)
        if destination_fd >= 0:
            os.close(destination_fd)


def _copy_verified_archive_to_private(
    *,
    archive_root: Path,
    private_root: Path,
    expected_manifest_file_sha256: str,
    expected_manifest_sha256: str,
    expected_tree_sha256: str,
) -> None:
    before_state = _metadata_state(archive_root)
    _, source_fd = open_secure_directory(
        archive_root, root=archive_root, create=False
    )
    _, destination_fd = open_secure_directory(
        private_root, root=private_root, create=False
    )
    try:
        manifest_bytes, manifest_metadata = _read_relative_bytes(
            source_fd, SOURCE_ARCHIVE_MANIFEST
        )
        manifest = _parse_manifest(manifest_bytes)
        directories, files = _validate_manifest_surface(manifest)
        _require(
            _sha_bytes(manifest_bytes) == expected_manifest_file_sha256
            and manifest["manifest_sha256"] == expected_manifest_sha256
            and manifest["tree_sha256"] == expected_tree_sha256
            and stat.S_IMODE(manifest_metadata.st_mode) == 0o440,
            "verified source archive identity changed before staging",
        )
        for relative in directories:
            path = PurePosixPath(relative)
            parent_fd = _open_relative_directory_fd(
                destination_fd, path.parent.as_posix()
            )
            try:
                os.mkdir(path.name, 0o750, dir_fd=parent_fd)
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        records = [
            {
                "path": SOURCE_ARCHIVE_MANIFEST,
                "size_bytes": len(manifest_bytes),
                "sha256": expected_manifest_file_sha256,
                "archive_mode": "0440",
            },
            *files,
        ]
        for record in sorted(records, key=lambda item: str(item["path"])):
            _copy_verified_archive_file_once(
                source_root_fd=source_fd,
                destination_root_fd=destination_fd,
                record=record,
            )
        os.fsync(destination_fd)
    finally:
        os.close(source_fd)
        os.close(destination_fd)
    _require(
        before_state == _metadata_state(archive_root),
        "verified source archive mutated while staging",
    )
    _set_archive_read_only(private_root, manifest)
    _verify_archive(
        private_root,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_tree_sha256=expected_tree_sha256,
        require_content_address_name=False,
    )


def stage_source_archive(
    *,
    archive_root: Path,
    stage_parent: Path,
    expected_receipt_file_sha256: str,
    expected_receipt_sha256: str,
    expected_manifest_file_sha256: str,
    expected_manifest_sha256: str,
    expected_tree_sha256: str,
) -> dict[str, Any]:
    """Stage only from a verified immutable archive into content-addressed Scratch."""

    _validate_expected_digests(
        expected_receipt_file_sha256=expected_receipt_file_sha256,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_tree_sha256=expected_tree_sha256,
    )
    _require(archive_root.is_absolute(), "archive root must be absolute")
    _require(stage_parent.is_absolute(), "stage parent must be absolute")
    archive = _lexical(archive_root)
    source_parent = archive.parent
    stage = _lexical(stage_parent)
    _require(
        source_parent != stage
        and source_parent not in stage.parents
        and stage not in source_parent.parents,
        "archive parent and stage parent must be disjoint",
    )
    upstream = verify_source_archive(
        archive,
        expected_receipt_file_sha256=expected_receipt_file_sha256,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_tree_sha256=expected_tree_sha256,
    )
    _, upstream_parent_fd = open_secure_directory(
        source_parent, root=source_parent, create=False
    )
    try:
        upstream_receipt, upstream_receipt_bytes, upstream_receipt_file_sha256 = (
            _read_receipt_at(upstream_parent_fd)
        )
    finally:
        os.close(upstream_parent_fd)
    _require(
        upstream_receipt_file_sha256 == expected_receipt_file_sha256
        and upstream_receipt["receipt_sha256"] == expected_receipt_sha256,
        "upstream source archive receipt changed before staging",
    )

    opened_stage, stage_fd = open_secure_directory(
        stage, root=stage, create=False
    )
    _require(opened_stage == stage, "stage parent changed")
    lock_fd = -1
    private_path: Path | None = None
    try:
        lock_fd = open_regular_lock_at(stage_fd, SOURCE_ARCHIVE_LOCK)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        private_name = (
            f".{SOURCE_ARCHIVE_PREFIX}stage-tmp."
            f"{os.getpid()}.{secrets.token_hex(8)}"
        )
        os.mkdir(private_name, 0o750, dir_fd=stage_fd)
        private_path = stage / private_name
        _copy_verified_archive_to_private(
            archive_root=archive,
            private_root=private_path,
            expected_manifest_file_sha256=expected_manifest_file_sha256,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_tree_sha256=expected_tree_sha256,
        )
        upstream_after = verify_source_archive(
            archive,
            expected_receipt_file_sha256=expected_receipt_file_sha256,
            expected_receipt_sha256=expected_receipt_sha256,
            expected_manifest_file_sha256=expected_manifest_file_sha256,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_tree_sha256=expected_tree_sha256,
        )
        _require(
            upstream_after == upstream,
            "upstream source archive binding changed while staging",
        )
        target_name = f"{SOURCE_ARCHIVE_PREFIX}{expected_tree_sha256}"
        target = stage / target_name
        try:
            existing_receipt, existing_receipt_bytes, _ = _read_receipt_at(
                stage_fd
            )
        except FileNotFoundError:
            existing_archives = _content_archive_names(stage_fd)
            _require(
                existing_archives in ([], [target_name]),
                "stage parent contains an unbound conflicting source archive",
            )
            if existing_archives:
                _verify_archive(
                    target,
                    expected_manifest_file_sha256=expected_manifest_file_sha256,
                    expected_manifest_sha256=expected_manifest_sha256,
                    expected_tree_sha256=expected_tree_sha256,
                    require_content_address_name=True,
                )
                _cleanup_private_tree(private_path)
                private_path = None
                status = "recovered_identical"
            else:
                os.rename(
                    private_name,
                    target_name,
                    src_dir_fd=stage_fd,
                    dst_dir_fd=stage_fd,
                )
                private_path = None
                os.fsync(stage_fd)
                status = "created"
            _publish_receipt_once(stage_fd, upstream_receipt_bytes)
        else:
            _require(
                existing_receipt_bytes == upstream_receipt_bytes
                and existing_receipt == upstream_receipt,
                "stage parent is already bound to different source archive content",
            )
            _require(
                _content_archive_names(stage_fd) == [target_name],
                "stage parent is not bound to exactly one source archive",
            )
            _verify_archive(
                target,
                expected_manifest_file_sha256=expected_manifest_file_sha256,
                expected_manifest_sha256=expected_manifest_sha256,
                expected_tree_sha256=expected_tree_sha256,
                require_content_address_name=True,
            )
            _cleanup_private_tree(private_path)
            private_path = None
            status = "reused_identical"
        verified = verify_source_archive(
            target,
            expected_receipt_file_sha256=expected_receipt_file_sha256,
            expected_receipt_sha256=expected_receipt_sha256,
            expected_manifest_file_sha256=expected_manifest_file_sha256,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_tree_sha256=expected_tree_sha256,
        )
        return {
            "stage_status": status,
            "staged_root": verified["archived_root"],
            "upstream_archived_root": upstream["archived_root"],
            **{
                key: value
                for key, value in verified.items()
                if key != "archived_root"
            },
        }
    finally:
        if private_path is not None:
            try:
                _cleanup_private_tree(private_path)
            except FileNotFoundError:
                pass
        if lock_fd >= 0:
            os.close(lock_fd)
        os.close(stage_fd)


def publish_source_archive(
    *, source_root: Path, archive_parent: Path
) -> dict[str, Any]:
    """Copy once, verify, and atomically publish a content-addressed archive."""

    source = _lexical(source_root)
    parent = _lexical(archive_parent)
    _require(source_root.is_absolute(), "reviewed source root must be absolute")
    _require(archive_parent.is_absolute(), "archive parent must be absolute")
    _require(
        source != parent and source not in parent.parents and parent not in source.parents,
        "reviewed source root and archive parent must be disjoint",
    )
    open_source, source_fd = open_secure_directory(
        source, root=source, create=False
    )
    os.close(source_fd)
    _require(open_source == source, "reviewed source root changed")
    open_parent, parent_fd = open_secure_directory(
        parent, root=parent, create=False
    )
    _require(open_parent == parent, "archive parent changed")
    parent_metadata = os.fstat(parent_fd)
    _require(
        parent_metadata.st_uid == os.geteuid()
        and not (parent_metadata.st_mode & 0o022),
        "archive parent must be effective-user-owned and not group/world writable",
    )
    lock_fd = -1
    private_path: Path | None = None
    try:
        lock_fd = open_regular_lock_at(parent_fd, SOURCE_ARCHIVE_LOCK)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        private_name = f".{SOURCE_ARCHIVE_PREFIX}tmp.{os.getpid()}.{secrets.token_hex(8)}"
        os.mkdir(private_name, 0o750, dir_fd=parent_fd)
        private_path = parent / private_name
        copied = _copy_to_private_archive(source, private_path)
        target_name = f"{SOURCE_ARCHIVE_PREFIX}{copied.tree_sha256}"
        target = parent / target_name
        candidate_receipt = _receipt_from_copy(copied)
        candidate_receipt_bytes = _receipt_bytes(candidate_receipt)
        candidate_receipt_file_sha256 = _sha_bytes(candidate_receipt_bytes)
        try:
            existing_receipt, existing_receipt_bytes, _ = _read_receipt_at(
                parent_fd
            )
        except FileNotFoundError:
            existing_archives = _content_archive_names(parent_fd)
            _require(
                existing_archives in ([], [target_name]),
                "archive parent contains an unbound conflicting source archive",
            )
            if existing_archives:
                _verify_archive(
                    target,
                    expected_manifest_file_sha256=copied.manifest_file_sha256,
                    expected_manifest_sha256=copied.manifest["manifest_sha256"],
                    expected_tree_sha256=copied.tree_sha256,
                    require_content_address_name=True,
                )
                _cleanup_private_tree(private_path)
                private_path = None
                status = "recovered_identical"
            else:
                os.rename(
                    private_name,
                    target_name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                private_path = None
                os.fsync(parent_fd)
                status = "created"
            _publish_receipt_once(parent_fd, candidate_receipt_bytes)
        else:
            _require(
                existing_receipt_bytes == candidate_receipt_bytes
                and existing_receipt == candidate_receipt,
                "archive parent is already bound to different reviewed source content",
            )
            _require(
                _content_archive_names(parent_fd) == [target_name],
                "archive parent is not bound to exactly one content-addressed archive",
            )
            existing = _verify_archive(
                target,
                expected_manifest_file_sha256=copied.manifest_file_sha256,
                expected_manifest_sha256=copied.manifest["manifest_sha256"],
                expected_tree_sha256=copied.tree_sha256,
                require_content_address_name=True,
            )
            _require(
                existing["file_count"] == copied.manifest["file_count"]
                and existing["total_source_bytes"]
                == copied.manifest["total_source_bytes"],
                "existing content-addressed archive differs from identical retry",
            )
            _cleanup_private_tree(private_path)
            private_path = None
            status = "reused_identical"
        verified = verify_source_archive(
            target,
            expected_receipt_file_sha256=candidate_receipt_file_sha256,
            expected_receipt_sha256=candidate_receipt["receipt_sha256"],
            expected_manifest_file_sha256=copied.manifest_file_sha256,
            expected_manifest_sha256=copied.manifest["manifest_sha256"],
            expected_tree_sha256=copied.tree_sha256,
        )
        return {"publication_status": status, **verified}
    finally:
        if private_path is not None:
            try:
                _cleanup_private_tree(private_path)
            except FileNotFoundError:
                pass
        if lock_fd >= 0:
            os.close(lock_fd)
        os.close(parent_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ep12-source-archive",
        description="Publish or verify a content-addressed reviewed EP12 source archive",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish = subparsers.add_parser("publish")
    publish.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help=(
            "explicit reviewed source root; production EP12 callers use the "
            "frozen episode-level root containing SEARCH_POLICY.yaml and "
            "outputs/executor"
        ),
    )
    publish.add_argument("--archive-parent", type=Path, required=True)
    stage = subparsers.add_parser("stage")
    stage.add_argument("--archive-root", type=Path, required=True)
    stage.add_argument("--stage-parent", type=Path, required=True)
    stage.add_argument("--expected-receipt-file-sha256", required=True)
    stage.add_argument("--expected-receipt-sha256", required=True)
    stage.add_argument("--expected-manifest-file-sha256", required=True)
    stage.add_argument("--expected-manifest-sha256", required=True)
    stage.add_argument("--expected-tree-sha256", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--archive-root", type=Path, required=True)
    verify.add_argument("--expected-receipt-file-sha256")
    verify.add_argument("--expected-receipt-sha256")
    verify.add_argument("--expected-manifest-file-sha256")
    verify.add_argument("--expected-manifest-sha256")
    verify.add_argument("--expected-tree-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "publish":
        result = publish_source_archive(
            source_root=arguments.source_root,
            archive_parent=arguments.archive_parent,
        )
    elif arguments.command == "stage":
        result = stage_source_archive(
            archive_root=arguments.archive_root,
            stage_parent=arguments.stage_parent,
            expected_receipt_file_sha256=arguments.expected_receipt_file_sha256,
            expected_receipt_sha256=arguments.expected_receipt_sha256,
            expected_manifest_file_sha256=arguments.expected_manifest_file_sha256,
            expected_manifest_sha256=arguments.expected_manifest_sha256,
            expected_tree_sha256=arguments.expected_tree_sha256,
        )
    else:
        result = verify_source_archive(
            arguments.archive_root,
            expected_receipt_file_sha256=arguments.expected_receipt_file_sha256,
            expected_receipt_sha256=arguments.expected_receipt_sha256,
            expected_manifest_file_sha256=arguments.expected_manifest_file_sha256,
            expected_manifest_sha256=arguments.expected_manifest_sha256,
            expected_tree_sha256=arguments.expected_tree_sha256,
        )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SOURCE_ARCHIVE_MANIFEST",
    "SOURCE_ARCHIVE_PREFIX",
    "SOURCE_ARCHIVE_RECEIPT",
    "SOURCE_ARCHIVE_RECEIPT_SCHEMA",
    "SOURCE_ARCHIVE_SCHEMA",
    "SourceArchiveError",
    "publish_source_archive",
    "stage_source_archive",
    "verify_source_archive",
    "verify_source_archive_publication",
]
