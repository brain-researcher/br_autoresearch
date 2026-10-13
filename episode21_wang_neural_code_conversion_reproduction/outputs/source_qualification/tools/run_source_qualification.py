#!/usr/bin/env python3
"""Outcome-blind EP21 public-source qualification.

This program downloads and inspects only the pinned public source archive and
its public metadata. It never opens any empirical EP21, EP04, neural, stimulus,
feature, reconstruction, or scoring payload.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
import verified_source_archive as source_archive_verifier


RELEVANT_SUFFIXES = {
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".md",
    ".txt",
    ".sh",
}
EVIDENCE_PATTERNS = {
    "activation": re.compile(r"LeakyReLU|(?<!Leaky)ReLU", re.IGNORECASE),
    "content_loss": re.compile(
        r"content.?loss|feature.?map|inverse|norm|weight", re.IGNORECASE
    ),
    "reconstruction": re.compile(
        r"DISTS|generator|reconstruct|conv[1-5]_[1-4]|fc[678]", re.IGNORECASE
    ),
    "decoder": re.compile(r"ridge|voxel|500|decoder", re.IGNORECASE),
    "external_dependency": re.compile(
        r"figshare|deeprecon|download|checkpoint|pre.?train", re.IGNORECASE
    ),
}

PINNED_SOURCE_RELEASE = {
    "repository": "https://github.com/KamitaniLab/InterSiteNeuralCodeConversion",
    "tag": "V1.0.0",
    "expected_commit": "edbff02edd6a08d3c6c829af45a957fd14bdc460",
    "zenodo_record": "14910040",
    "zenodo_doi": "10.5281/zenodo.14910040",
    "expected_archive_name": "KamitaniLab/InterSiteNeuralCodeConversion-V1.0.0.zip",
    "expected_archive_size_bytes": 23_110_908,
    "expected_archive_md5": "bb80dcc1a038737d6d8fc2c95b3e5586",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def fetch_json(url: str, timeout: int = 60) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "EP21-source-qualification/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def download(
    url: str,
    destination: Path,
    expected_md5: str,
    expected_size: int,
    timeout: int = 120,
) -> tuple[str, int]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "EP21-source-qualification/1.0"},
    )
    digest = hashlib.md5()  # Required reproduction identity, not security use.
    byte_count = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=destination.parent,
        prefix=f".{destination.name}.part-",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            os.fchmod(handle.fileno(), 0o600)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) != expected_size:
                    raise ValueError("source archive Content-Length mismatch")
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    byte_count += len(chunk)
                    if byte_count > expected_size:
                        raise ValueError("source archive exceeded the pinned size")
                    handle.write(chunk)
                    digest.update(chunk)
            handle.flush()
            os.fsync(handle.fileno())
            observed_md5 = digest.hexdigest()
            if byte_count != expected_size or observed_md5 != expected_md5:
                raise ValueError("downloaded source archive identity mismatch")
            os.fchmod(handle.fileno(), 0o440)
        except Exception:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return observed_md5, byte_count


def git_tag_commit(repository: str, tag: str) -> str:
    completed = subprocess.run(
        ["git", "ls-remote", "--refs", repository, f"refs/tags/{tag}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    fields = completed.stdout.strip().split()
    if len(fields) < 2:
        raise RuntimeError(f"tag {tag!r} did not resolve")
    return fields[0]


def safe_archive_name(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def inspect_archive(archive: zipfile.ZipFile) -> dict[str, Any]:
    evidence: dict[str, list[dict[str, Any]]] = {
        key: [] for key in EVIDENCE_PATTERNS
    }
    files: list[dict[str, Any]] = []
    unsafe_names: list[str] = []
    environment_files: list[str] = []
    model_weight_files: list[str] = []
    archive_uncompressed_bytes = 0

    for info in archive.infolist():
        if info.is_dir():
            continue
        if not safe_archive_name(info.filename):
            unsafe_names.append(info.filename)
            continue
        archive_uncompressed_bytes += info.file_size
        suffix = PurePosixPath(info.filename).suffix.lower()
        files.append(
            {
                "path": info.filename,
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
            }
        )
        if PurePosixPath(info.filename).name.lower() in {
            "env.yaml",
            "environment.yml",
            "environment.yaml",
            "requirements.txt",
        }:
            environment_files.append(info.filename)
        if suffix in {".pt", ".pth", ".caffemodel", ".ckpt", ".h5"}:
            model_weight_files.append(info.filename)
        if suffix not in RELEVANT_SUFFIXES or info.file_size > 2_000_000:
            continue
        try:
            text = archive.read(info).decode("utf-8", errors="replace")
        except (KeyError, RuntimeError):
            continue
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            for category, pattern in EVIDENCE_PATTERNS.items():
                if len(evidence[category]) >= 120:
                    continue
                if pattern.search(line):
                    evidence[category].append(
                        {
                            "path": info.filename,
                            "line": line_number,
                            "text": line[:400],
                        }
                    )

    return {
        "file_count": len(files),
        "uncompressed_bytes": archive_uncompressed_bytes,
        "unsafe_member_names": unsafe_names,
        "environment_files": sorted(environment_files),
        "bundled_model_weight_files": sorted(model_weight_files),
        "files": sorted(files, key=lambda item: item["path"]),
        "source_evidence": evidence,
    }


def direct_interface_state(episode_dir: Path) -> dict[str, str]:
    inputs = episode_dir / "inputs"
    names = [
        "laion_regular_feature_handoff",
        "laion_regular_scoring_handoff",
        "external_site_feature_handoff",
        "external_site_scoring_handoff",
    ]
    return {
        name: "present" if (inputs / name).exists() else "absent" for name in names
    }


def source_contract_errors(source: dict[str, Any]) -> list[str]:
    """Compare the supplied inventory to the EP21 contract's immutable release."""
    errors: list[str] = []
    for field, expected in PINNED_SOURCE_RELEASE.items():
        observed = source.get(field)
        matches = (
            str(observed) == str(expected)
            if field == "zenodo_record"
            else observed == expected
        )
        if not matches:
            errors.append(
                f"source_inventory_pin_mismatch:{field}: expected {expected!r}, got {observed!r}"
            )
    return errors


def source_identity_matches(
    source: dict[str, Any],
    zenodo: dict[str, Any],
    tag_commit: str,
    archive_md5: str,
    archive_size: int,
    archive_inspection: dict[str, Any],
    errors: list[str],
) -> bool:
    return (
        not source_contract_errors(source)
        and tag_commit == PINNED_SOURCE_RELEASE["expected_commit"]
        and str(zenodo.get("id")) == PINNED_SOURCE_RELEASE["zenodo_record"]
        and zenodo.get("doi") == PINNED_SOURCE_RELEASE["zenodo_doi"]
        and archive_md5 == PINNED_SOURCE_RELEASE["expected_archive_md5"]
        and archive_size == PINNED_SOURCE_RELEASE["expected_archive_size_bytes"]
        and not archive_inspection.get("unsafe_member_names")
        and not errors
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--structural-manifest", type=Path, required=True)
    parser.add_argument("--asset-inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()

    started_at = utc_now()
    print(
        json.dumps(
            {
                "event": "ep21_source_qualification_started",
                "pid": os.getpid(),
                "started_at": started_at,
            }
        ),
        flush=True,
    )

    structural = load_json(args.structural_manifest)
    inventory = load_json(args.asset_inventory)
    source = inventory["source_release"]
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inventory_errors = source_contract_errors(source)
    errors: list[str] = list(inventory_errors)
    tag_commit = ""
    zenodo: dict[str, Any] = {}
    archive_inspection: dict[str, Any] = {}
    archive_md5 = ""
    archive_size = 0
    archive_reused = False
    archive_path = args.scratch_dir / PINNED_SOURCE_RELEASE["expected_archive_name"]

    if not inventory_errors:
        try:
            tag_commit = git_tag_commit(
                PINNED_SOURCE_RELEASE["repository"] + ".git",
                PINNED_SOURCE_RELEASE["tag"],
            )
        except Exception as exc:  # Preserve a complete diagnostic report.
            errors.append(f"git_tag_resolution: {type(exc).__name__}: {exc}")

        try:
            zenodo = fetch_json(
                f"https://zenodo.org/api/records/{PINNED_SOURCE_RELEASE['zenodo_record']}"
            )
            matching_files = [
                item
                for item in zenodo.get("files", [])
                if item.get("key") == PINNED_SOURCE_RELEASE["expected_archive_name"]
            ]
            if len(matching_files) != 1:
                raise RuntimeError("expected exactly one matching Zenodo archive")
            archive_url = matching_files[0]["links"]["self"]
            if os.path.lexists(archive_path):
                try:
                    with source_archive_verifier.verified_zip(
                        archive_path,
                        PINNED_SOURCE_RELEASE["expected_archive_md5"],
                        PINNED_SOURCE_RELEASE["expected_archive_size_bytes"],
                    ) as verified:
                        archive_md5 = verified.md5
                        archive_size = verified.size
                        archive_inspection = inspect_archive(verified.archive)
                    archive_reused = True
                except (OSError, ValueError, zipfile.BadZipFile):
                    archive_reused = False
            if not archive_reused:
                download(
                    archive_url,
                    archive_path,
                    PINNED_SOURCE_RELEASE["expected_archive_md5"],
                    PINNED_SOURCE_RELEASE["expected_archive_size_bytes"],
                )
                with source_archive_verifier.verified_zip(
                    archive_path,
                    PINNED_SOURCE_RELEASE["expected_archive_md5"],
                    PINNED_SOURCE_RELEASE["expected_archive_size_bytes"],
                ) as verified:
                    archive_md5 = verified.md5
                    archive_size = verified.size
                    archive_inspection = inspect_archive(verified.archive)
        except Exception as exc:  # Preserve a complete diagnostic report.
            errors.append(f"zenodo_archive: {type(exc).__name__}: {exc}")

    expected_commit = PINNED_SOURCE_RELEASE["expected_commit"]
    expected_md5 = PINNED_SOURCE_RELEASE["expected_archive_md5"]
    expected_size = PINNED_SOURCE_RELEASE["expected_archive_size_bytes"]
    source_identity_pass = source_identity_matches(
        source,
        zenodo,
        tag_commit,
        archive_md5,
        archive_size,
        archive_inspection,
        errors,
    )

    interfaces = direct_interface_state(args.episode_dir)
    empirical_ready = (
        structural["authorization"]["empirical_handoff_access_authorized"]
        and structural["authorization"]["neural_scoring_authorized"]
        and structural["authorization"]["protocol_lock_present"]
        and interfaces["laion_regular_feature_handoff"] == "present"
        and interfaces["laion_regular_scoring_handoff"] == "present"
    )

    verified_inventory = {
        "schema_version": "ep21.verified_public_source_inventory.v1",
        "astra_analysis_id": "source_qualification",
        "started_at": started_at,
        "completed_at": utc_now(),
        "process_id": os.getpid(),
        "scope": "public_source_code_and_metadata_only",
        "forbidden_payload_accessed": False,
        "source_release": {
            "repository": PINNED_SOURCE_RELEASE["repository"],
            "tag": PINNED_SOURCE_RELEASE["tag"],
            "expected_commit": expected_commit,
            "observed_commit": tag_commit,
            "tag_commit_matches": tag_commit == expected_commit,
            "zenodo_record": zenodo.get("id"),
            "zenodo_record_matches": str(zenodo.get("id"))
            == PINNED_SOURCE_RELEASE["zenodo_record"],
            "zenodo_doi": zenodo.get("doi"),
            "zenodo_doi_matches": zenodo.get("doi")
            == PINNED_SOURCE_RELEASE["zenodo_doi"],
            "expected_archive_name": PINNED_SOURCE_RELEASE[
                "expected_archive_name"
            ],
            "observed_archive_size_bytes": archive_size,
            "archive_size_matches": archive_size == expected_size,
            "observed_archive_md5": archive_md5,
            "archive_md5_matches": archive_md5 == expected_md5,
            "archive_reused_from_verified_scratch_cache": archive_reused,
            "inventory_release_matches_pin": not inventory_errors,
            "identity_status": "verified" if source_identity_pass else "failed",
        },
        "archive": archive_inspection,
        "errors": errors,
    }
    atomic_json(args.output_dir / "verified_public_source_inventory.json", verified_inventory)

    qualification = {
        "schema_version": "ep21.source_qualification_report.v1",
        "astra_analysis_id": "source_qualification",
        "attempt": args.attempt,
        "started_at": started_at,
        "completed_at": utc_now(),
        "process_id": os.getpid(),
        "scope": "public_source_and_redacted_structure_only",
        "source_identity": "pass" if source_identity_pass else "fail",
        "structural_manifest_status": structural["qualification_disposition"],
        "declared_interface_state": interfaces,
        "paper_code_discrepancy_evidence": archive_inspection.get(
            "source_evidence", {}
        ),
        "asset_availability": {
            "source_environment_files": archive_inspection.get(
                "environment_files", []
            ),
            "bundled_model_weight_files": archive_inspection.get(
                "bundled_model_weight_files", []
            ),
            "nonbundled_assets": "require_separate_authentication_before_protocol_lock",
        },
        "empirical_readiness": "ready" if empirical_ready else "blocked",
        "empirical_blockers": []
        if empirical_ready
        else [
            "LAION DUA, release identity, and Sherlock risk status unresolved",
            "role-filtered LAION feature and scoring handoffs absent",
            "image roles, tau split, overlaps, participants, betas, and masks unverified",
            "protocol lock absent",
            "neural scoring not authorized by the current structural manifest",
        ],
        "external_readiness": "blocked_unprovisioned",
        "ood_disposition": "not_evaluable_under_current_contract",
        "forbidden_payload_accessed": False,
        "errors": errors,
        "outcome": "completed_with_empirical_blockers"
        if source_identity_pass
        else "technical_failure",
    }
    atomic_json(args.output_dir / "qualification_report.json", qualification)

    laion = {
        "schema_version": "ep21.laion_branch_qualification.v1",
        "astra_analysis_id": "source_qualification",
        "attempt": args.attempt,
        "status": "blocked_pending_dua_role_filtered_handoffs_and_protocol_lock",
        "empirical_data_accessed": False,
        "planning_participants": 5,
        "planning_directed_pairs": 20,
        "eligible_directed_pairs": [],
        "interface_state": {
            key: value for key, value in interfaces.items() if key.startswith("laion_")
        },
        "next_action": "Provision a DUA-cleared redacted structural manifest and role-filtered regular-only feature handoff; then resolve the protocol lock before any scoring handoff is visible.",
    }
    atomic_json(args.output_dir / "laion_branch_qualification.json", laion)

    external = {
        "schema_version": "ep21.external_branch_qualification.v1",
        "astra_analysis_id": "source_qualification",
        "attempt": args.attempt,
        "status": "blocked_unprovisioned_optional_branch",
        "empirical_data_accessed": False,
        "planning_directed_pairs": 70,
        "eligible_directed_pairs": [],
        "interface_state": {
            key: value
            for key, value in interfaces.items()
            if key.startswith("external_")
        },
        "next_action": "Provision and qualify exact NSD/THINGS releases independently; absence does not block the LAION branch.",
    }
    atomic_json(args.output_dir / "external_branch_qualification.json", external)

    summary = {
        "event": "ep21_source_qualification_finished",
        "pid": os.getpid(),
        "outcome": qualification["outcome"],
        "source_identity": qualification["source_identity"],
        "empirical_readiness": qualification["empirical_readiness"],
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary), flush=True)
    return 0 if source_identity_pass else 1


if __name__ == "__main__":
    sys.exit(main())
