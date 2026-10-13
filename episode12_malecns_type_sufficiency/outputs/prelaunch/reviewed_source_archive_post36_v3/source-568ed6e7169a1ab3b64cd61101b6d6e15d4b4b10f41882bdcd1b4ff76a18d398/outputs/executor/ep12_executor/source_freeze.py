"""Canonical, fail-closed source freeze for the EP12 executable surface.

The freeze covers ``SEARCH_POLICY.yaml`` and every durable regular file below
``outputs/executor`` other than explicitly named cache or transient artifacts.
It is deliberately independent of Git: the bytes submitted to Slurm are the
bytes authenticated here, including tests and Slurm wrappers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any, Mapping, Sequence

from .policy import canonical_json, digest_object
from .synthetic import EPISODE_ROOT, SCRATCH_ROOT


SOURCE_FREEZE_SCHEMA = "ep12.executable_source_freeze_manifest.v1"
SOURCE_FREEZE_RELATIVE = Path(
    "outputs/prelaunch/executable_source_freeze_manifest.json"
)

_EXECUTOR_RELATIVE = Path("outputs/executor")
_SEARCH_POLICY_RELATIVE = Path("SEARCH_POLICY.yaml")
_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "runtime",
        "tmp",
        "temp",
    }
)
_EXCLUDED_FILE_NAMES = frozenset({".coverage", ".DS_Store"})
_EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".tmp", ".swp", ".swo", "~")
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class SourceFreezeError(RuntimeError):
    """The executable surface or its manifest is unsafe or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceFreezeError(message)


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def secure_source_path(path: Path, *, episode_root: Path, label: str) -> Path:
    """Return an in-episode lexical path after rejecting symlink components."""

    _require(episode_root.is_absolute(), f"{label} EP12 root must be absolute")
    _require(path.is_absolute(), f"{label} must be absolute")
    root = _lexical_absolute(episode_root)
    candidate = _lexical_absolute(path)
    _require(
        candidate == root or root in candidate.parents,
        f"{label} escapes EP12 root: {candidate}",
    )
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise SourceFreezeError(
                f"cannot inspect {label} path component {current}: {exc}"
            ) from exc
        _require(
            not stat.S_ISLNK(metadata.st_mode),
            f"{label} contains a symlink component: {current}",
        )
    return candidate


def _read_bytes_nofollow(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SourceFreezeError(f"cannot open {label} {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{label} is not regular: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                return b"".join(chunks)
            chunks.append(block)
    finally:
        os.close(descriptor)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _path_exists_nofollow(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise SourceFreezeError(f"cannot inspect path {path}: {exc}") from exc
    return True


def _excluded_file(name: str) -> bool:
    return name in _EXCLUDED_FILE_NAMES or name.endswith(_EXCLUDED_SUFFIXES)


def _surface_rule() -> dict[str, Any]:
    return {
        "required_root_file": _SEARCH_POLICY_RELATIVE.as_posix(),
        "executor_root": _EXECUTOR_RELATIVE.as_posix(),
        "include": "every durable regular file",
        "excluded_directory_names": sorted(_EXCLUDED_DIRECTORY_NAMES),
        "excluded_file_names": sorted(_EXCLUDED_FILE_NAMES),
        "excluded_suffixes": list(_EXCLUDED_SUFFIXES),
        "symlinks_allowed": False,
    }


def _validated_manifest_relative(value: object) -> PurePosixPath:
    _require(isinstance(value, str) and bool(value), "source-freeze path is invalid")
    relative = PurePosixPath(value)
    _require(not relative.is_absolute(), "source-freeze path is absolute")
    _require(
        relative.as_posix() == value
        and all(component not in {"", ".", ".."} for component in relative.parts),
        f"source-freeze path is not normalized: {value}",
    )
    is_policy = relative == PurePosixPath(_SEARCH_POLICY_RELATIVE.as_posix())
    is_executor = (
        len(relative.parts) > len(_EXECUTOR_RELATIVE.parts)
        and relative.parts[: len(_EXECUTOR_RELATIVE.parts)]
        == _EXECUTOR_RELATIVE.parts
    )
    _require(is_policy or is_executor, f"source-freeze path is outside the surface: {value}")
    if is_executor:
        nested_parts = relative.parts[len(_EXECUTOR_RELATIVE.parts) :]
        _require(
            not any(part in _EXCLUDED_DIRECTORY_NAMES for part in nested_parts[:-1]),
            f"source-freeze path enters an excluded directory: {value}",
        )
        _require(
            not _excluded_file(nested_parts[-1]),
            f"source-freeze path names an excluded file: {value}",
        )
    return relative


def _validated_manifest_records(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    _require(value.get("surface_rule") == _surface_rule(), "source-freeze surface rule changed")
    files = value.get("files")
    _require(isinstance(files, list), "source-freeze file list is invalid")
    _require(
        type(value.get("file_count")) is int and value.get("file_count") == len(files),
        "source-freeze file count changed",
    )
    records: list[dict[str, Any]] = []
    prior = ""
    for record in files:
        _require(isinstance(record, Mapping), "source-freeze file record is invalid")
        _require(
            set(record) == {"path", "size_bytes", "sha256"},
            "source-freeze file record schema changed",
        )
        relative = _validated_manifest_relative(record.get("path"))
        relative_text = relative.as_posix()
        _require(relative_text > prior, "source-freeze paths are not strictly sorted")
        prior = relative_text
        _require(
            type(record.get("size_bytes")) is int and record["size_bytes"] >= 0,
            "source-freeze size is invalid",
        )
        _require(
            isinstance(record.get("sha256"), str)
            and _HEX_RE.fullmatch(record["sha256"]) is not None,
            "source-freeze file digest is invalid",
        )
        records.append(dict(record))
    _require(
        any(record["path"] == _SEARCH_POLICY_RELATIVE.as_posix() for record in records),
        "source-freeze manifest omits SEARCH_POLICY.yaml",
    )
    return records


def _surface_paths(episode_root: Path) -> list[Path]:
    root = secure_source_path(
        _lexical_absolute(episode_root),
        episode_root=_lexical_absolute(episode_root),
        label="EP12 root",
    )
    _require(root.is_dir(), f"EP12 root is not a directory: {root}")
    policy = secure_source_path(
        root / _SEARCH_POLICY_RELATIVE,
        episode_root=root,
        label="SEARCH_POLICY.yaml",
    )
    try:
        policy_metadata = os.lstat(policy)
    except OSError as exc:
        raise SourceFreezeError(f"missing SEARCH_POLICY.yaml: {exc}") from exc
    _require(stat.S_ISREG(policy_metadata.st_mode), "SEARCH_POLICY.yaml is not regular")

    executor = secure_source_path(
        root / _EXECUTOR_RELATIVE,
        episode_root=root,
        label="executor surface",
    )
    try:
        executor_metadata = os.lstat(executor)
    except OSError as exc:
        raise SourceFreezeError(f"missing executor surface: {exc}") from exc
    _require(stat.S_ISDIR(executor_metadata.st_mode), "executor surface is not a directory")

    paths = [policy]

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise SourceFreezeError(f"cannot enumerate executor surface {directory}: {exc}") from exc
        for entry in entries:
            child = Path(entry.path)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise SourceFreezeError(f"cannot inspect executor entry {child}: {exc}") from exc
            _require(
                not stat.S_ISLNK(metadata.st_mode),
                f"executor surface contains a symlink: {child}",
            )
            if stat.S_ISDIR(metadata.st_mode):
                if entry.name not in _EXCLUDED_DIRECTORY_NAMES:
                    visit(child)
                continue
            if stat.S_ISREG(metadata.st_mode):
                if not _excluded_file(entry.name):
                    paths.append(child)
                continue
            raise SourceFreezeError(f"executor surface contains a non-regular entry: {child}")

    visit(executor)
    _require(len(paths) > 1, "executor surface contains no durable files")
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _surface_records(episode_root: Path) -> list[dict[str, Any]]:
    root = _lexical_absolute(episode_root)
    records: list[dict[str, Any]] = []
    for path in _surface_paths(root):
        encoded = _read_bytes_nofollow(path, label="source-freeze input")
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": len(encoded),
                "sha256": _sha256_bytes(encoded),
            }
        )
    return records


def _manifest_value(episode_root: Path) -> dict[str, Any]:
    root = _lexical_absolute(episode_root)
    files = _surface_records(root)
    core = {
        "schema_version": SOURCE_FREEZE_SCHEMA,
        "episode_root": str(root),
        "surface_rule": _surface_rule(),
        "files": files,
        "file_count": len(files),
        "real_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return {**core, "manifest_sha256": digest_object(core)}


def source_freeze_manifest_value(
    *, episode_root: Path = EPISODE_ROOT
) -> dict[str, Any]:
    """Return the canonical current executable-surface identity without writing it."""

    return _manifest_value(_lexical_absolute(episode_root))


def _atomic_write(path: Path, encoded: bytes, *, episode_root: Path) -> None:
    secured = secure_source_path(path, episode_root=episode_root, label="source-freeze manifest")
    secured.parent.mkdir(parents=True, exist_ok=True)
    secure_source_path(
        secured.parent,
        episode_root=episode_root,
        label="source-freeze manifest directory",
    )
    temporary = secured.with_name(
        f".{secured.name}.{os.getpid()}-{_sha256_bytes(encoded)[:12]}.tmp"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o640,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SourceFreezeError(f"short write while creating {secured}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, secured)
    parent_descriptor = os.open(secured.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def source_freeze_file_sha256(path: Path, *, episode_root: Path) -> str:
    secured = secure_source_path(
        path, episode_root=_lexical_absolute(episode_root), label="source-freeze manifest"
    )
    return _sha256_bytes(_read_bytes_nofollow(secured, label="source-freeze manifest"))


def verify_source_freeze_manifest(
    manifest_path: Path,
    *,
    episode_root: Path = EPISODE_ROOT,
    expected_file_sha256: str | None = None,
    require_current_surface: bool = True,
) -> dict[str, Any]:
    """Verify canonical bytes, the self hash, and optionally the live surface."""

    root = _lexical_absolute(episode_root)
    secured = secure_source_path(
        manifest_path, episode_root=root, label="source-freeze manifest"
    )
    encoded = _read_bytes_nofollow(secured, label="source-freeze manifest")
    if expected_file_sha256 is not None:
        _require(
            isinstance(expected_file_sha256, str)
            and _HEX_RE.fullmatch(expected_file_sha256) is not None,
            "expected source-freeze file SHA-256 is invalid",
        )
        _require(
            _sha256_bytes(encoded) == expected_file_sha256,
            "source-freeze manifest file SHA-256 mismatch",
        )
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceFreezeError(f"malformed source-freeze manifest: {exc}") from exc
    _require(isinstance(value, Mapping), "source-freeze manifest is not an object")
    _require(
        encoded == (canonical_json(value) + "\n").encode("utf-8"),
        "source-freeze manifest is not canonical JSON",
    )
    expected_keys = {
        "schema_version",
        "episode_root",
        "surface_rule",
        "files",
        "file_count",
        "real_connectivity_accessed",
        "final_connectivity_access_authorized",
        "manifest_sha256",
    }
    _require(set(value) == expected_keys, "source-freeze manifest schema changed")
    core = {key: item for key, item in value.items() if key != "manifest_sha256"}
    _require(value.get("schema_version") == SOURCE_FREEZE_SCHEMA, "source-freeze schema mismatch")
    _require(value.get("episode_root") == str(root), "source-freeze episode root changed")
    _require(
        isinstance(value.get("manifest_sha256"), str)
        and value.get("manifest_sha256") == digest_object(core),
        "source-freeze self hash mismatch",
    )
    _require(
        value.get("real_connectivity_accessed") is False
        and value.get("final_connectivity_access_authorized") is False,
        "source freeze crosses a real-connectivity boundary",
    )
    _validated_manifest_records(value)
    if require_current_surface:
        expected = _manifest_value(root)
        _require(dict(value) == expected, "source-freeze executable surface changed")
    return dict(value)


def _open_directory_nofollow(path: Path, *, label: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SourceFreezeError(f"cannot open {label} {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISDIR(metadata.st_mode), f"{label} is not a directory: {path}")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _read_relative_bytes_nofollow(
    root_descriptor: int,
    relative: PurePosixPath,
    *,
    label: str,
) -> bytes:
    """Read one relative regular file without following any path component."""

    current = os.dup(root_descriptor)
    descriptor: int | None = None
    try:
        for component in relative.parts[:-1]:
            try:
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=current,
                )
            except OSError as exc:
                raise SourceFreezeError(
                    f"cannot open {label} directory {relative}: {exc}"
                ) from exc
            os.close(current)
            current = next_descriptor
        try:
            descriptor = os.open(
                relative.parts[-1],
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current,
            )
        except OSError as exc:
            raise SourceFreezeError(f"cannot open {label} {relative}: {exc}") from exc
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{label} is not regular: {relative}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                return b"".join(chunks)
            chunks.append(block)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(current)


def _write_relative_bytes_exclusive(
    root_descriptor: int,
    relative: PurePosixPath,
    encoded: bytes,
) -> None:
    """Create one staged file under a private directory, without symlink traversal."""

    current = os.dup(root_descriptor)
    descriptor: int | None = None
    try:
        for component in relative.parts[:-1]:
            try:
                os.mkdir(component, 0o700, dir_fd=current)
            except FileExistsError:
                pass
            except OSError as exc:
                raise SourceFreezeError(
                    f"cannot create staged source directory {relative}: {exc}"
                ) from exc
            try:
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=current,
                )
            except OSError as exc:
                raise SourceFreezeError(
                    f"cannot open staged source directory {relative}: {exc}"
                ) from exc
            os.close(current)
            current = next_descriptor
        try:
            descriptor = os.open(
                relative.parts[-1],
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=current,
            )
        except OSError as exc:
            raise SourceFreezeError(f"cannot create staged source {relative}: {exc}") from exc
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SourceFreezeError(f"short write while staging {relative}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(current)


def _secure_existing_stage_directory(
    path: Path,
    *,
    scratch_root: Path,
    label: str,
) -> Path:
    scratch = _lexical_absolute(scratch_root)
    candidate = secure_source_path(path, episode_root=scratch, label=label)
    try:
        metadata = os.lstat(candidate)
    except OSError as exc:
        raise SourceFreezeError(f"cannot inspect {label} {candidate}: {exc}") from exc
    _require(stat.S_ISDIR(metadata.st_mode), f"{label} is not a directory: {candidate}")
    return candidate


def _enumerate_staged_tree(stage_root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise SourceFreezeError(f"cannot enumerate staged source {directory}: {exc}") from exc
        for entry in entries:
            child = Path(entry.path)
            relative = child.relative_to(stage_root).as_posix()
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise SourceFreezeError(f"cannot inspect staged source {child}: {exc}") from exc
            _require(
                not stat.S_ISLNK(metadata.st_mode),
                f"staged source contains a symlink: {child}",
            )
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative)
                visit(child)
            elif stat.S_ISREG(metadata.st_mode):
                files.add(relative)
            else:
                raise SourceFreezeError(f"staged source contains a non-regular entry: {child}")

    visit(stage_root)
    return files, directories


def _verify_staged_records(stage_root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    expected_files = {str(record["path"]) for record in records}
    expected_directories: set[str] = set()
    for path_text in expected_files:
        relative = PurePosixPath(path_text)
        for parent in relative.parents:
            if parent != PurePosixPath("."):
                expected_directories.add(parent.as_posix())
    observed_files, observed_directories = _enumerate_staged_tree(stage_root)
    _require(observed_files == expected_files, "staged source file set differs from manifest")
    _require(
        observed_directories == expected_directories,
        "staged source directory set differs from manifest",
    )
    root_descriptor = _open_directory_nofollow(stage_root, label="staged source root")
    try:
        for record in records:
            relative = _validated_manifest_relative(record["path"])
            encoded = _read_relative_bytes_nofollow(
                root_descriptor, relative, label="staged source"
            )
            _require(
                len(encoded) == record["size_bytes"],
                f"staged source size differs from manifest: {relative}",
            )
            _require(
                _sha256_bytes(encoded) == record["sha256"],
                f"staged source digest differs from manifest: {relative}",
            )
    finally:
        os.close(root_descriptor)


def verify_staged_source(
    stage_root: Path,
    manifest_path: Path,
    *,
    episode_root: Path = EPISODE_ROOT,
    scratch_root: Path = SCRATCH_ROOT,
    expected_file_sha256: str,
) -> dict[str, Any]:
    """Verify an exact, symlink-free staged tree against its bound manifest."""

    manifest = verify_source_freeze_manifest(
        manifest_path,
        episode_root=episode_root,
        expected_file_sha256=expected_file_sha256,
        require_current_surface=False,
    )
    records = _validated_manifest_records(manifest)
    secured_stage = _secure_existing_stage_directory(
        stage_root, scratch_root=scratch_root, label="staged source root"
    )
    _verify_staged_records(secured_stage, records)
    return manifest


def _make_stage_read_only(stage: Path) -> None:
    """Remove write permission after a complete staged tree is verified."""

    for path in sorted(stage.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        metadata = os.lstat(path)
        if stat.S_ISREG(metadata.st_mode):
            os.chmod(path, 0o400)
        elif stat.S_ISDIR(metadata.st_mode):
            # The tree was already walked with lstat and rejects symlinks.
            # Some Sherlock Python builds cannot combine chmod with
            # follow_symlinks=False, so use the authenticated path directly.
            os.chmod(path, 0o500)
        else:
            raise SourceFreezeError(f"staged source contains an unsafe entry: {path}")
    os.chmod(stage, 0o500)


def _discard_private_stage(stage: Path) -> None:
    """Make a private unpublished stage removable, then discard it."""

    for path in stage.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            os.chmod(path, 0o700)
    os.chmod(stage, 0o700)
    shutil.rmtree(stage)


def stage_source_freeze(
    manifest_path: Path,
    *,
    expected_file_sha256: str,
    stage_parent: Path,
    episode_root: Path = EPISODE_ROOT,
    scratch_root: Path = SCRATCH_ROOT,
) -> Path:
    """Create once and reuse an immutable content-addressed Scratch source tree."""

    root = _lexical_absolute(episode_root)
    manifest = verify_source_freeze_manifest(
        manifest_path,
        episode_root=root,
        expected_file_sha256=expected_file_sha256,
        require_current_surface=False,
    )
    records = _validated_manifest_records(manifest)
    parent = _secure_existing_stage_directory(
        stage_parent, scratch_root=scratch_root, label="source-stage parent"
    )
    target = parent / f"source-{expected_file_sha256}"
    if _path_exists_nofollow(target):
        secured_target = _secure_existing_stage_directory(
            target, scratch_root=scratch_root, label="content-addressed source stage"
        )
        _verify_staged_records(secured_target, records)
        return secured_target

    expected_paths = [str(record["path"]) for record in records]
    observed_paths = [
        path.relative_to(root).as_posix() for path in _surface_paths(root)
    ]
    _require(
        observed_paths == expected_paths,
        "live source file set differs from source-freeze manifest",
    )
    job_token = re.sub(
        r"[^A-Za-z0-9_.-]", "_", os.environ.get("SLURM_JOB_ID", "nojob")
    )[:64]
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".source-{expected_file_sha256[:16]}-{job_token}-", dir=parent
        )
    )
    stage = _secure_existing_stage_directory(
        stage, scratch_root=scratch_root, label="new source stage"
    )
    os.chmod(stage, 0o700)
    source_descriptor = _open_directory_nofollow(root, label="EP12 source root")
    stage_descriptor = _open_directory_nofollow(stage, label="new source stage")
    try:
        for record in records:
            relative = _validated_manifest_relative(record["path"])
            encoded = _read_relative_bytes_nofollow(
                source_descriptor, relative, label="live source"
            )
            _require(
                len(encoded) == record["size_bytes"],
                f"live source size differs from manifest: {relative}",
            )
            _require(
                _sha256_bytes(encoded) == record["sha256"],
                f"live source digest differs from manifest: {relative}",
            )
            _write_relative_bytes_exclusive(stage_descriptor, relative, encoded)
    except Exception:
        os.close(stage_descriptor)
        os.close(source_descriptor)
        _discard_private_stage(stage)
        raise
    else:
        os.close(stage_descriptor)
        os.close(source_descriptor)

    try:
        verify_source_freeze_manifest(
            manifest_path,
            episode_root=root,
            expected_file_sha256=expected_file_sha256,
            require_current_surface=True,
        )
    except Exception:
        _discard_private_stage(stage)
        raise
    try:
        _verify_staged_records(stage, records)
        _make_stage_read_only(stage)
        os.rename(stage, target)
    except OSError:
        if _path_exists_nofollow(target):
            _discard_private_stage(stage)
            secured_target = _secure_existing_stage_directory(
                target,
                scratch_root=scratch_root,
                label="concurrent content-addressed source stage",
            )
            _verify_staged_records(secured_target, records)
            return secured_target
        _discard_private_stage(stage)
        raise
    except Exception:
        _discard_private_stage(stage)
        raise
    parent_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    secured_target = _secure_existing_stage_directory(
        target, scratch_root=scratch_root, label="content-addressed source stage"
    )
    _verify_staged_records(secured_target, records)
    return secured_target


def create_source_freeze_manifest(
    *,
    episode_root: Path = EPISODE_ROOT,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Create the canonical manifest once, or verify an identical existing one."""

    root = _lexical_absolute(episode_root)
    target = manifest_path or (root / SOURCE_FREEZE_RELATIVE)
    target = secure_source_path(target, episode_root=root, label="source-freeze manifest")
    expected = _manifest_value(root)
    encoded = (canonical_json(expected) + "\n").encode("utf-8")
    if _path_exists_nofollow(target):
        observed = verify_source_freeze_manifest(
            target,
            episode_root=root,
            expected_file_sha256=_sha256_bytes(encoded),
            require_current_surface=True,
        )
        _require(observed == expected, "existing source-freeze manifest changed")
        return observed
    _atomic_write(target, encoded, episode_root=root)
    return verify_source_freeze_manifest(
        target,
        episode_root=root,
        expected_file_sha256=_sha256_bytes(encoded),
        require_current_surface=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--manifest", type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--expected-file-sha256", required=True)
    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("--manifest", type=Path, required=True)
    stage_parser.add_argument("--expected-file-sha256", required=True)
    stage_parser.add_argument("--stage-parent", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "create":
        value = create_source_freeze_manifest(manifest_path=arguments.manifest)
    elif arguments.command == "verify":
        value = verify_source_freeze_manifest(
            arguments.manifest,
            expected_file_sha256=arguments.expected_file_sha256,
        )
    else:
        stage = stage_source_freeze(
            arguments.manifest,
            expected_file_sha256=arguments.expected_file_sha256,
            stage_parent=arguments.stage_parent,
        )
        print(stage)
        return 0
    print(canonical_json(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
