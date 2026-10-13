#!/usr/bin/env python3
"""Build the stable EP21 source-qualification report from declared inputs.

The wrapper composes the focused qualification tools in an isolated Scratch
workspace and publishes only the requested ASTRA output.  It reads public
source/metadata, two pinned public non-empirical model archives, and six
declared outcome-blind contract/manifest inputs; it never opens empirical handoffs or any
decoded-feature, OOD, or EP04 payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping


MAX_DECLARED_INPUT_BYTES = 10 * 1024 * 1024


def run_checked(arguments: list[str]) -> None:
    subprocess.run(arguments, check=True)


def publish(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def safe_relative_archive(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("source archive name must be a safe relative path")
    return relative


def read_declared_input(path: Path, role: str) -> tuple[bytes, dict[str, Any]]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot securely open declared input {role}: {path}") from error
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
            raise ValueError(f"declared input {role} is not a single-link regular file")
        if initial.st_size <= 0 or initial.st_size > MAX_DECLARED_INPUT_BYTES:
            raise ValueError(f"declared input {role} has an invalid size")
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            byte_count += len(chunk)
            if byte_count > MAX_DECLARED_INPUT_BYTES:
                raise ValueError(f"declared input {role} exceeded the size bound")
        final = os.fstat(descriptor)
        identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(initial, field) != getattr(final, field) for field in identity_fields):
            raise ValueError(f"declared input {role} changed while reading")
        if byte_count != initial.st_size:
            raise ValueError(f"declared input {role} byte count changed while reading")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    payload.decode("utf-8", errors="strict")
    return payload, {
        "path": str(path),
        "role": role,
        "size_bytes": byte_count,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def write_private_snapshot(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o400)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("private snapshot write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_recipe_report(
    source: Path,
    destination: Path,
    declared_inputs: Mapping[str, Mapping[str, Any]],
) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["evidence_inputs"] = {
        **{key: dict(value) for key, value in declared_inputs.items()},
        "public_source_identity_and_inventory": "recipe_intermediate:run_source_qualification.py",
        "public_asset_dependencies_metadata_and_runtime": "recipe_intermediate:audit_public_assets.py",
        "public_model_payload_audit": "recipe_intermediate:audit_public_model_payloads.py",
        "unmanifested_source_assets": "recipe_intermediate:audit_unmanifested_assets.py",
        "figshare_manifest_reconciliation": "recipe_intermediate:audit_manifest_metadata.py",
    }
    payload["recipe_replay"] = {
        "ephemeral_recipe_workspace_lifecycle": "cleanup_pending_at_publication_and_automatic_on_normal_context_exit",
        "persistent_public_model_cache_retained": True,
        "declared_contract_inputs_identity_checked": [
            "goal",
            "datasets",
            "protocol_table",
            "input_contract",
        ],
        "mutable_declared_inputs_snapshotted_for_children": [
            "conversion_structural_manifest",
            "upstream_pipeline_asset_inventory",
        ],
        "component_tools": [
            "run_source_qualification.py",
            "audit_public_assets.py",
            "audit_public_model_payloads.py",
            "audit_unmanifested_assets.py",
            "audit_manifest_metadata.py",
            "consolidate_qualification.py",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--protocol-table", type=Path, required=True)
    parser.add_argument("--input-contract", type=Path, required=True)
    parser.add_argument("--structural-manifest", type=Path, required=True)
    parser.add_argument("--asset-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    if args.attempt <= 0:
        parser.error("--attempt must be the positive attempt prepared in the ledger")

    declared_paths = {
        "goal": (args.goal, "scientific_question_evidence_roles_and_claim_limits"),
        "datasets": (args.datasets, "dataset_and_branch_contract"),
        "protocol_table": (args.protocol_table, "outcome_independent_protocol_fields"),
        "input_contract": (args.input_contract, "stable_interfaces_and_access_restrictions"),
        "conversion_structural_manifest": (
            args.structural_manifest,
            "redacted_structural_metadata",
        ),
        "upstream_pipeline_asset_inventory": (
            args.asset_inventory,
            "outcome_blind_source_asset_inventory",
        ),
    }
    declared_payloads: dict[str, bytes] = {}
    declared_inputs: dict[str, dict[str, Any]] = {}
    for input_id, (path, role) in declared_paths.items():
        payload, identity = read_declared_input(path, role)
        declared_payloads[input_id] = payload
        declared_inputs[input_id] = identity

    tools_dir = Path(__file__).resolve().parent
    episode_dir = tools_dir.parents[2]
    scratch_root = args.scratch_root
    if scratch_root is None:
        scratch_value = os.environ.get("SCRATCH")
        if not scratch_value:
            raise RuntimeError("SCRATCH is required for source-qualification replay")
        scratch_root = Path(scratch_value) / "br_autoresearch" / episode_dir.name
    scratch_root.mkdir(parents=True, exist_ok=True)
    public_model_cache = scratch_root / "public_model_assets"

    inventory = json.loads(
        declared_payloads["upstream_pipeline_asset_inventory"].decode("utf-8")
    )
    expected_archive_name = inventory["source_release"]["expected_archive_name"]
    safe_relative_archive(expected_archive_name)

    with tempfile.TemporaryDirectory(
        prefix="source-qualification-replay-", dir=scratch_root
    ) as temporary_name:
        workspace = Path(temporary_name)
        declared_dir = workspace / "declared-inputs"
        structural_snapshot = declared_dir / "conversion_structural_manifest.json"
        inventory_snapshot = declared_dir / "upstream_pipeline_asset_inventory.json"
        write_private_snapshot(
            structural_snapshot, declared_payloads["conversion_structural_manifest"]
        )
        write_private_snapshot(
            inventory_snapshot,
            declared_payloads["upstream_pipeline_asset_inventory"],
        )
        initial_dir = workspace / "initial"
        public_assets_dir = workspace / "public-assets"
        unmanifested_dir = workspace / "unmanifested"
        manifest_metadata_path = workspace / "figshare-manifest-metadata.json"
        public_model_payload_path = workspace / "public-model-payload-audit.json"
        consolidated_dir = workspace / "consolidated"
        source_cache = workspace / "source-cache"

        run_checked(
            [
                sys.executable,
                str(tools_dir / "run_source_qualification.py"),
                "--episode-dir",
                str(episode_dir),
                "--structural-manifest",
                str(structural_snapshot),
                "--asset-inventory",
                str(inventory_snapshot),
                "--output-dir",
                str(initial_dir),
                "--scratch-dir",
                str(source_cache),
                "--attempt",
                str(args.attempt),
            ]
        )
        archive_path = source_cache / expected_archive_name
        run_checked(
            [
                sys.executable,
                str(tools_dir / "audit_public_assets.py"),
                "--archive",
                str(archive_path),
                "--output-dir",
                str(public_assets_dir),
                "--attempt",
                str(args.attempt),
            ]
        )
        run_checked(
            [
                sys.executable,
                str(tools_dir / "audit_unmanifested_assets.py"),
                "--archive",
                str(archive_path),
                "--output-dir",
                str(unmanifested_dir),
                "--attempt",
                str(args.attempt),
            ]
        )
        run_checked(
            [
                sys.executable,
                str(tools_dir / "audit_public_model_payloads.py"),
                "--source-archive",
                str(archive_path),
                "--cache-dir",
                str(public_model_cache),
                "--output",
                str(public_model_payload_path),
                "--attempt",
                str(args.attempt),
            ]
        )
        run_checked(
            [
                sys.executable,
                str(tools_dir / "audit_manifest_metadata.py"),
                "--archive",
                str(archive_path),
                "--output",
                str(manifest_metadata_path),
                "--attempt",
                str(args.attempt),
            ]
        )
        run_checked(
            [
                sys.executable,
                str(tools_dir / "consolidate_qualification.py"),
                "--attempt",
                str(args.attempt),
                "--structural-manifest",
                str(structural_snapshot),
                "--attempt-2-dir",
                str(initial_dir),
                "--attempt-4-dir",
                str(public_assets_dir),
                "--attempt-6-dir",
                str(unmanifested_dir),
                "--manifest-metadata-report",
                str(manifest_metadata_path),
                "--public-model-payload-report",
                str(public_model_payload_path),
                "--output-dir",
                str(consolidated_dir),
            ]
        )
        publish_recipe_report(
            consolidated_dir / "qualification_report.json",
            args.output,
            declared_inputs,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
