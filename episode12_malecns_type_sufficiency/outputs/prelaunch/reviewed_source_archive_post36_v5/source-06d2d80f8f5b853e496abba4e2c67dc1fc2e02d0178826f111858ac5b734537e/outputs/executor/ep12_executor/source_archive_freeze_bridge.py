"""Bind one reviewed source archive to the canonical live source freeze.

The reviewed archive is the authority for executable bytes. This module is
the narrow handoff to the already-qualified EP12 source-freeze pipeline: it
verifies the complete archive publication, proves that the live
``SEARCH_POLICY.yaml`` plus ``outputs/executor/**`` file map is byte-for-byte
identical, creates/verifies the canonical live freeze, and publishes one
exact-once receipt. Downstream jobs can then carry only the source-freeze
identity; every stage still rechecks the live surface and fails closed on a
later mutation.

The bridge never reads connectivity inputs or outcome artifacts.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

from .policy import canonical_json, digest_object
from .secure_transaction import open_regular_lock_at, open_secure_directory
from .source_archive import (
    SOURCE_ARCHIVE_MANIFEST,
    SOURCE_ARCHIVE_SCHEMA,
    verify_source_archive,
)
from .source_freeze import (
    SOURCE_FREEZE_RELATIVE,
    SOURCE_FREEZE_SCHEMA,
    create_source_freeze_manifest,
    source_freeze_file_sha256,
    source_freeze_manifest_value,
    verify_source_freeze_manifest,
)


BRIDGE_SCHEMA = "ep12.source_archive_freeze_bridge_receipt.v1"
CANONICAL_EP12_ROOT = Path(
    "/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/"
    "episode12_malecns_type_sufficiency"
)
BRIDGE_RELATIVE = Path(
    "outputs/prelaunch/source_archive_freeze_bridge/receipt.json"
)
BRIDGE_LOCK = ".source_archive_freeze_bridge.lock"
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_ARCHIVE_IDENTITY_FIELDS = {
    "archived_root",
    "archive_parent",
    "archive_receipt_path",
    "archive_receipt_file_sha256",
    "archive_receipt_sha256",
    "manifest_path",
    "manifest_file_sha256",
    "manifest_sha256",
    "tree_sha256",
    "directory_count",
    "file_count",
    "total_source_bytes",
}
_FREEZE_IDENTITY_FIELDS = {
    "schema_version",
    "manifest_path",
    "manifest_file_sha256",
    "manifest_sha256",
    "file_count",
}
_RECEIPT_FIELDS = {
    "schema_version",
    "archive_identity",
    "live_episode_root",
    "source_freeze_identity",
    "executable_record_map_sha256",
    "file_count",
    "archive_and_live_file_maps_identical",
    "archive_modes_bound_upstream_not_required_live_equal",
    "archive_reverified_after_freeze",
    "live_surface_reverified_after_freeze",
    "real_connectivity_accessed",
    "final_connectivity_access_authorized",
    "receipt_sha256",
}


class SourceArchiveFreezeBridgeError(RuntimeError):
    """The reviewed archive cannot be safely bound to the live source freeze."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceArchiveFreezeBridgeError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _read_regular_nofollow(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SourceArchiveFreezeBridgeError(
            f"cannot safely open {label} {path}: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{label} is not regular: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _archive_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        _ARCHIVE_IDENTITY_FIELDS <= set(value),
        "verified archive identity is incomplete",
    )
    return {key: value[key] for key in sorted(_ARCHIVE_IDENTITY_FIELDS)}


def _normalized_records(
    raw_records: object, *, archive_records: bool
) -> list[dict[str, Any]]:
    _require(isinstance(raw_records, list), "executable record list is invalid")
    required = {"path", "size_bytes", "sha256"}
    expected = required | ({"archive_mode"} if archive_records else set())
    records: list[dict[str, Any]] = []
    prior = ""
    for index, raw in enumerate(raw_records):
        _require(isinstance(raw, Mapping), f"executable record {index} is invalid")
        _require(set(raw) == expected, f"executable record {index} schema changed")
        path = raw.get("path")
        size = raw.get("size_bytes")
        sha256 = raw.get("sha256")
        _require(
            isinstance(path, str)
            and path > prior
            and (
                path == "SEARCH_POLICY.yaml"
                or path.startswith("outputs/executor/")
            ),
            f"executable record {index} path is invalid",
        )
        _require(
            type(size) is int and size >= 0,
            f"executable record {index} size is invalid",
        )
        _require(
            isinstance(sha256, str) and _HEX_RE.fullmatch(sha256) is not None,
            f"executable record {index} digest is invalid",
        )
        records.append({"path": path, "size_bytes": size, "sha256": sha256})
        prior = path
    _require(bool(records), "executable record list is empty")
    return records


def _read_verified_archive_manifest(
    verified_archive: Mapping[str, Any],
) -> dict[str, Any]:
    archived_root = _lexical(Path(str(verified_archive["archived_root"])))
    manifest_path = _lexical(Path(str(verified_archive["manifest_path"])))
    _require(
        manifest_path == archived_root / SOURCE_ARCHIVE_MANIFEST,
        "verified archive manifest path changed",
    )
    encoded = _read_regular_nofollow(manifest_path, label="archive manifest")
    _require(
        _sha256_bytes(encoded) == verified_archive["manifest_file_sha256"],
        "archive manifest changed after archive verification",
    )
    try:
        parsed = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceArchiveFreezeBridgeError("archive manifest is malformed") from exc
    _require(isinstance(parsed, Mapping), "archive manifest is not an object")
    manifest = dict(parsed)
    _require(
        encoded == (canonical_json(manifest) + "\n").encode("utf-8"),
        "archive manifest is not canonical JSON",
    )
    _require(
        manifest.get("schema_version") == SOURCE_ARCHIVE_SCHEMA
        and manifest.get("manifest_sha256") == verified_archive["manifest_sha256"]
        and manifest.get("tree_sha256") == verified_archive["tree_sha256"]
        and manifest.get("file_count") == verified_archive["file_count"],
        "archive manifest identity differs from verified publication",
    )
    return manifest


def _verify_archive_exact(
    *,
    archive_root: Path,
    receipt_file_sha256: str,
    receipt_sha256: str,
    manifest_file_sha256: str,
    manifest_sha256: str,
    tree_sha256: str,
) -> dict[str, Any]:
    values = {
        "archive receipt file SHA-256": receipt_file_sha256,
        "archive receipt semantic SHA-256": receipt_sha256,
        "archive manifest file SHA-256": manifest_file_sha256,
        "archive manifest semantic SHA-256": manifest_sha256,
        "archive tree SHA-256": tree_sha256,
    }
    for label, value in values.items():
        _require(
            isinstance(value, str) and _HEX_RE.fullmatch(value) is not None,
            f"{label} is invalid",
        )
    try:
        return verify_source_archive(
            _lexical(archive_root),
            expected_receipt_file_sha256=receipt_file_sha256,
            expected_receipt_sha256=receipt_sha256,
            expected_manifest_file_sha256=manifest_file_sha256,
            expected_manifest_sha256=manifest_sha256,
            expected_tree_sha256=tree_sha256,
        )
    except Exception as exc:
        raise SourceArchiveFreezeBridgeError(
            f"reviewed source archive verification failed: {exc}"
        ) from exc


def _receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    return (canonical_json(receipt) + "\n").encode("utf-8")


def _validate_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(value)
    _require(set(receipt) == _RECEIPT_FIELDS, "bridge receipt schema changed")
    core = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    _require(
        receipt.get("schema_version") == BRIDGE_SCHEMA
        and receipt.get("receipt_sha256") == digest_object(core),
        "bridge receipt identity or self-hash mismatch",
    )
    _require(
        isinstance(receipt.get("archive_identity"), Mapping)
        and set(receipt["archive_identity"]) == _ARCHIVE_IDENTITY_FIELDS,
        "bridge archive identity schema changed",
    )
    freeze = receipt.get("source_freeze_identity")
    _require(
        isinstance(freeze, Mapping)
        and set(freeze) == _FREEZE_IDENTITY_FIELDS
        and freeze.get("schema_version") == SOURCE_FREEZE_SCHEMA,
        "bridge source-freeze identity schema changed",
    )
    for label, digest in (
        ("bridge executable record map SHA-256", receipt.get("executable_record_map_sha256")),
        ("bridge receipt SHA-256", receipt.get("receipt_sha256")),
        ("source-freeze file SHA-256", freeze.get("manifest_file_sha256")),
        ("source-freeze semantic SHA-256", freeze.get("manifest_sha256")),
    ):
        _require(
            isinstance(digest, str) and _HEX_RE.fullmatch(digest) is not None,
            f"{label} is invalid",
        )
    _require(
        type(receipt.get("file_count")) is int
        and receipt["file_count"] > 0
        and receipt["file_count"] == freeze.get("file_count")
        and receipt.get("archive_and_live_file_maps_identical") is True
        and receipt.get("archive_modes_bound_upstream_not_required_live_equal") is True
        and receipt.get("archive_reverified_after_freeze") is True
        and receipt.get("live_surface_reverified_after_freeze") is True
        and receipt.get("real_connectivity_accessed") is False
        and receipt.get("final_connectivity_access_authorized") is False,
        "bridge receipt does not record a closed, outcome-blind source transition",
    )
    return receipt


def _read_receipt_at(directory_fd: int) -> tuple[dict[str, Any], bytes]:
    descriptor = os.open(
        BRIDGE_RELATIVE.name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), "bridge receipt is not regular")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        encoded = b"".join(chunks)
    finally:
        os.close(descriptor)
    try:
        parsed = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceArchiveFreezeBridgeError("bridge receipt is malformed") from exc
    _require(isinstance(parsed, Mapping), "bridge receipt is not an object")
    receipt = _validate_receipt(parsed)
    _require(encoded == _receipt_bytes(receipt), "bridge receipt is not canonical JSON")
    return receipt, encoded


def _publish_receipt_once(
    directory_fd: int, receipt: Mapping[str, Any]
) -> tuple[str, str]:
    encoded = _receipt_bytes(receipt)
    temporary = f".{BRIDGE_RELATIVE.name}.{os.getpid()}.{_sha256_bytes(encoded)[:16]}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o640,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, "short write while publishing bridge receipt")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(
            temporary,
            BRIDGE_RELATIVE.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.fsync(directory_fd)
    except FileExistsError as exc:
        raise SourceArchiveFreezeBridgeError(
            "bridge receipt appeared during exact-once publication"
        ) from exc
    finally:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
    return receipt["receipt_sha256"], _sha256_bytes(encoded)


def bridge_source_archive_to_live_freeze(
    *,
    archive_root: Path,
    archive_receipt_file_sha256: str,
    archive_receipt_sha256: str,
    archive_manifest_file_sha256: str,
    archive_manifest_sha256: str,
    archive_tree_sha256: str,
    live_episode_root: Path,
    require_canonical_live_root: bool = False,
) -> dict[str, Any]:
    """Publish/reuse the exact bridge receipt after two-sided verification."""

    live_root = _lexical(live_episode_root)
    _require(live_episode_root.is_absolute(), "live episode root must be absolute")
    if require_canonical_live_root:
        _require(
            live_root == CANONICAL_EP12_ROOT,
            "production bridge live root is not canonical EP12",
        )
    bridge_dir = live_root / BRIDGE_RELATIVE.parent
    try:
        _opened, directory_fd = open_secure_directory(
            bridge_dir, root=live_root, create=True
        )
        lock_fd = open_regular_lock_at(directory_fd, BRIDGE_LOCK)
    except Exception as exc:
        raise SourceArchiveFreezeBridgeError(
            f"cannot open secure bridge transaction: {exc}"
        ) from exc
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        archive_before = _verify_archive_exact(
            archive_root=archive_root,
            receipt_file_sha256=archive_receipt_file_sha256,
            receipt_sha256=archive_receipt_sha256,
            manifest_file_sha256=archive_manifest_file_sha256,
            manifest_sha256=archive_manifest_sha256,
            tree_sha256=archive_tree_sha256,
        )
        archive_manifest = _read_verified_archive_manifest(archive_before)
        archive_records = _normalized_records(
            archive_manifest.get("files"), archive_records=True
        )

        live_candidate = source_freeze_manifest_value(episode_root=live_root)
        live_records = _normalized_records(
            live_candidate.get("files"), archive_records=False
        )
        _require(
            archive_records == live_records,
            "canonical live executable surface differs from reviewed archive",
        )

        freeze_manifest_path = live_root / SOURCE_FREEZE_RELATIVE
        create_source_freeze_manifest(
            episode_root=live_root, manifest_path=freeze_manifest_path
        )
        freeze_file_sha256 = source_freeze_file_sha256(
            freeze_manifest_path, episode_root=live_root
        )
        freeze_manifest = verify_source_freeze_manifest(
            freeze_manifest_path,
            episode_root=live_root,
            expected_file_sha256=freeze_file_sha256,
            require_current_surface=True,
        )
        freeze_records = _normalized_records(
            freeze_manifest.get("files"), archive_records=False
        )
        _require(archive_records == freeze_records, "source freeze differs from reviewed archive")

        archive_after = _verify_archive_exact(
            archive_root=archive_root,
            receipt_file_sha256=archive_receipt_file_sha256,
            receipt_sha256=archive_receipt_sha256,
            manifest_file_sha256=archive_manifest_file_sha256,
            manifest_sha256=archive_manifest_sha256,
            tree_sha256=archive_tree_sha256,
        )
        archive_manifest_after = _read_verified_archive_manifest(archive_after)
        archive_records_after = _normalized_records(
            archive_manifest_after.get("files"), archive_records=True
        )
        _require(
            _archive_identity(archive_before) == _archive_identity(archive_after)
            and archive_records_after == archive_records,
            "reviewed archive changed during source-freeze bridge",
        )
        freeze_after = verify_source_freeze_manifest(
            freeze_manifest_path,
            episode_root=live_root,
            expected_file_sha256=freeze_file_sha256,
            require_current_surface=True,
        )
        _require(
            dict(freeze_after) == dict(freeze_manifest),
            "canonical live surface changed during source-freeze bridge",
        )

        core = {
            "schema_version": BRIDGE_SCHEMA,
            "archive_identity": _archive_identity(archive_after),
            "live_episode_root": str(live_root),
            "source_freeze_identity": {
                "schema_version": SOURCE_FREEZE_SCHEMA,
                "manifest_path": str(freeze_manifest_path),
                "manifest_file_sha256": freeze_file_sha256,
                "manifest_sha256": freeze_manifest["manifest_sha256"],
                "file_count": len(freeze_records),
            },
            "executable_record_map_sha256": digest_object(freeze_records),
            "file_count": len(freeze_records),
            "archive_and_live_file_maps_identical": True,
            # The archive verifier still authenticates its file modes. The
            # downstream source-freeze contract intentionally binds only path,
            # size, and bytes because sbatch and Python read their inputs;
            # live execute bits are not an authority.
            "archive_modes_bound_upstream_not_required_live_equal": True,
            "archive_reverified_after_freeze": True,
            "live_surface_reverified_after_freeze": True,
            "real_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }
        candidate = _validate_receipt(
            {**core, "receipt_sha256": digest_object(core)}
        )
        try:
            existing, encoded = _read_receipt_at(directory_fd)
        except FileNotFoundError:
            receipt_sha256, receipt_file_sha256 = _publish_receipt_once(
                directory_fd, candidate
            )
            _require(
                receipt_sha256 == candidate["receipt_sha256"],
                "bridge receipt hash changed",
            )
            status = "created"
        else:
            _require(
                existing == candidate,
                "existing bridge receipt binds different source identities",
            )
            receipt_file_sha256 = _sha256_bytes(encoded)
            status = "reused_identical"
        return {
            "publication_status": status,
            "receipt_path": str(live_root / BRIDGE_RELATIVE),
            "receipt_file_sha256": receipt_file_sha256,
            **candidate,
        }
    finally:
        os.close(lock_fd)
        os.close(directory_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--archive-receipt-file-sha256", required=True)
    parser.add_argument("--archive-receipt-sha256", required=True)
    parser.add_argument("--archive-manifest-file-sha256", required=True)
    parser.add_argument("--archive-manifest-sha256", required=True)
    parser.add_argument("--archive-tree-sha256", required=True)
    parser.add_argument("--live-episode-root", type=Path, required=True)
    parser.add_argument("--require-canonical-live-root", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = bridge_source_archive_to_live_freeze(
        archive_root=arguments.archive_root,
        archive_receipt_file_sha256=arguments.archive_receipt_file_sha256,
        archive_receipt_sha256=arguments.archive_receipt_sha256,
        archive_manifest_file_sha256=arguments.archive_manifest_file_sha256,
        archive_manifest_sha256=arguments.archive_manifest_sha256,
        archive_tree_sha256=arguments.archive_tree_sha256,
        live_episode_root=arguments.live_episode_root,
        require_canonical_live_root=arguments.require_canonical_live_root,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
