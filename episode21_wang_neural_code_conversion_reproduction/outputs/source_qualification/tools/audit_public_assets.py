#!/usr/bin/env python3
"""Inventory EP21's public source dependencies without downloading payloads."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
import verified_source_archive as source_archive_verifier


ARCHIVE_MD5 = "bb80dcc1a038737d6d8fc2c95b3e5586"
ARCHIVE_SIZE_BYTES = 23_110_908
ARCHIVE_ROOT = "KamitaniLab-InterSiteNeuralCodeConversion-edbff02/"
MAIN_FILES = ARCHIVE_ROOT + "data/files.json"
RECON_FILES = ARCHIVE_ROOT + "reconstruction/data/files.json"
MAIN_ENV = ARCHIVE_ROOT + "env.yaml"
RECON_ENV = ARCHIVE_ROOT + "reconstruction/env.yaml"
RECON_CODE = ARCHIVE_ROOT + "reconstruction/recon_icnn_image_gd.py"
FIGSHARE_ARTICLE_ID = 21564384


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def url_checksum_pairs(record: dict[str, Any]) -> list[dict[str, str]]:
    urls = record["url"] if isinstance(record["url"], list) else [record["url"]]
    md5s = (
        record["md5sum"]
        if isinstance(record["md5sum"], list)
        else [record["md5sum"]]
    )
    if len(urls) != len(md5s):
        raise ValueError(f"URL/checksum cardinality mismatch for {record['name']}")
    return [{"url": url, "md5": md5} for url, md5 in zip(urls, md5s)]


def classify_asset(source_manifest: str, group: str, name: str) -> dict[str, Any]:
    if source_manifest == "reconstruction/data/files.json":
        if name == "decoded_features.zip":
            return {
                "role": "source_study_decoded_outcome_fixture",
                "ep21_use": "forbidden_for_lock_and_current_episode",
                "head_probe_allowed": False,
            }
        return {
            "role": "public_non_neural_reconstruction_model_or_statistic",
            "ep21_use": "candidate_source_pipeline_asset_pending_checksum_verification",
            "head_probe_allowed": True,
        }

    if group in {"fmri_training", "fmri_test"}:
        role = "source_study_empirical_fmri"
    elif group in {"stimulus_feature", "test_image-true_features"}:
        role = "source_study_stimulus_feature_payload"
    elif group in {"pre-trained-converters", "pre-trained-decoders"}:
        role = "source_study_fitted_object_reference"
    else:
        role = "unclassified_source_asset"
    return {
        "role": role,
        "ep21_use": "inventory_reference_only_not_an_adapted_LAION_input",
        "head_probe_allowed": False,
    }


def head_probe(url: str, timeout: int = 60) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "EP21-public-asset-audit/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {
                "requested_method": "HEAD",
                "requested_url": url,
                "status": response.status,
                "final_url": response.geturl(),
                "content_length": response.headers.get("Content-Length"),
                "content_type": response.headers.get("Content-Type"),
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "body_bytes_read": 0,
                "error": None,
            }
    except (urllib.error.URLError, TimeoutError) as exc:
        return {
            "requested_method": "HEAD",
            "requested_url": url,
            "status": None,
            "final_url": None,
            "content_length": None,
            "content_type": None,
            "etag": None,
            "last_modified": None,
            "body_bytes_read": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def fetch_json(url: str, timeout: int = 60) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        method="GET",
        headers={"User-Agent": "EP21-public-asset-audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def figshare_file_id(url: str) -> int:
    match = re.search(r"/files/(\d+)(?:$|[/?#])", url)
    if match is None:
        raise ValueError(f"cannot resolve Figshare file id from {url}")
    return int(match.group(1))


def parse_environment(name: str, text: str) -> dict[str, Any]:
    stripped = [line.strip() for line in text.splitlines()]
    dependencies = [line[2:] for line in stripped if line.startswith("- ")]
    key_patterns = {
        "python": re.compile(r"^python(?:=|$)"),
        "pytorch": re.compile(r"^pytorch(?:=|$)"),
        "cuda": re.compile(r"^(?:cuda|cudatoolkit|pytorch-cuda)(?:=|$)"),
        "torchvision": re.compile(r"^torchvision(?:=|$)"),
        "bdpy": re.compile(r"^bdpy(?:=|$)"),
        "fastl2lir": re.compile(r"^fastl2lir(?:=|$)"),
        "hydra_core": re.compile(r"^hydra-core(?:=|$)"),
    }
    selected = {
        key: [item for item in dependencies if pattern.search(item)]
        for key, pattern in key_patterns.items()
    }
    return {
        "source_path": name,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "dependency_entry_count": len(dependencies),
        "selected_versions": selected,
        "reproduction_note": "Declared upstream environment; availability on Sherlock is not established by this inventory.",
    }


def literal_dict_assignment(source: str, variable: str) -> dict[str, Any]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == variable for target in node.targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                raise TypeError(f"{variable} is not a literal dictionary")
            return value
    raise KeyError(variable)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()

    started_at = utc_now()
    print(
        json.dumps(
            {
                "event": "ep21_public_asset_audit_started",
                "pid": os.getpid(),
                "started_at": started_at,
            }
        ),
        flush=True,
    )

    inventory: list[dict[str, Any]] = []
    safe_model_parts: list[dict[str, Any]] = []
    with source_archive_verifier.verified_zip(
        args.archive, ARCHIVE_MD5, ARCHIVE_SIZE_BYTES
    ) as verified:
        observed_md5 = verified.md5
        archive = verified.archive
        main_manifest = json.loads(archive.read(MAIN_FILES))
        recon_manifest = json.loads(archive.read(RECON_FILES))
        main_env_text = archive.read(MAIN_ENV).decode("utf-8")
        recon_env_text = archive.read(RECON_ENV).decode("utf-8")
        recon_code_text = archive.read(RECON_CODE).decode("utf-8")

    for source_name, manifest in [
        ("data/files.json", main_manifest),
        ("reconstruction/data/files.json", recon_manifest),
    ]:
        for group, group_record in manifest.items():
            for file_record in group_record["files"]:
                classification = classify_asset(source_name, group, file_record["name"])
                pairs = url_checksum_pairs(file_record)
                record = {
                    "source_manifest": source_name,
                    "group": group,
                    "name": file_record["name"],
                    "save_in": group_record.get("save_in"),
                    "postprocessing": file_record.get("postproc", []),
                    "parts": pairs,
                    **classification,
                }
                inventory.append(record)
                if classification["head_probe_allowed"]:
                    for part_index, pair in enumerate(pairs):
                        safe_model_parts.append(
                            {
                                "asset_name": file_record["name"],
                                "part_index": part_index,
                                "declared_md5": pair["md5"],
                                "declared_url": pair["url"],
                                "figshare_file_id": figshare_file_id(pair["url"]),
                            }
                        )

    figshare_metadata = fetch_json(
        f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"
    )
    metadata_by_id = {item["id"]: item for item in figshare_metadata.get("files", [])}
    verified_model_assets: list[dict[str, Any]] = []
    for part in safe_model_parts:
        observed = metadata_by_id.get(part["figshare_file_id"])
        verified_model_assets.append(
            {
                **part,
                "metadata_present": observed is not None,
                "observed_name": None if observed is None else observed.get("name"),
                "observed_size_bytes": None if observed is None else observed.get("size"),
                "observed_md5": None
                if observed is None
                else observed.get("supplied_md5"),
                "observed_download_url": None
                if observed is None
                else observed.get("download_url"),
                "observed_is_link_only": None
                if observed is None
                else observed.get("is_link_only"),
                "name_matches": observed is not None
                and observed.get("name") == part["asset_name"],
                "md5_matches": observed is not None
                and observed.get("supplied_md5") == part["declared_md5"],
            }
        )

    runtime_report = {
        "schema_version": "ep21.public_runtime_environment_report.v1",
        "astra_analysis_id": "source_qualification",
        "attempt": args.attempt,
        "started_at": started_at,
        "completed_at": utc_now(),
        "process_id": os.getpid(),
        "archive_md5": observed_md5,
        "environments": [
            parse_environment("env.yaml", main_env_text),
            parse_environment("reconstruction/env.yaml", recon_env_text),
        ],
        "dists_like_custom_loss": {
            "implementation_path": "reconstruction/recon_icnn_image_gd.py",
            "alpha_coefficients": literal_dict_assignment(
                recon_code_text, "dists_alpha"
            ),
            "beta_coefficients": literal_dict_assignment(recon_code_text, "dists_beta"),
            "external_dists_weight_file_referenced": False,
            "interpretation": "The tagged reconstruction code embeds 19 alpha and 19 beta coefficients in a custom DIST-style loss; it does not reference a separately bundled DISTS checkpoint.",
        },
        "execution_environment_ready": False,
        "blocker": "Sherlock module/environment compatibility has not been qualified and no environment installation is authorized by this audit.",
    }
    atomic_json(args.output_dir / "runtime_environment_report.json", runtime_report)

    dependency_report = {
        "schema_version": "ep21.public_asset_dependency_report.v1",
        "astra_analysis_id": "source_qualification",
        "attempt": args.attempt,
        "started_at": started_at,
        "completed_at": utc_now(),
        "process_id": os.getpid(),
        "archive_md5": observed_md5,
        "scope": "pinned_public_source_metadata_only",
        "asset_count": len(inventory),
        "assets": inventory,
        "classification_counts": {
            role: sum(1 for item in inventory if item["role"] == role)
            for role in sorted({item["role"] for item in inventory})
        },
        "safe_model_asset_metadata_count": len(safe_model_parts),
        "excluded_payload_download_count": sum(
            1 for item in inventory if not item["head_probe_allowed"]
        ),
        "payload_bytes_downloaded": 0,
        "empirical_or_decoded_payload_accessed": False,
    }
    atomic_json(args.output_dir / "public_asset_dependency_report.json", dependency_report)

    metadata_report = {
        "schema_version": "ep21.public_asset_metadata_report.v1",
        "astra_analysis_id": "source_qualification",
        "attempt": args.attempt,
        "started_at": started_at,
        "completed_at": utc_now(),
        "process_id": os.getpid(),
        "method": "official_Figshare_article_metadata_API",
        "article": {
            "id": figshare_metadata.get("id"),
            "title": figshare_metadata.get("title"),
            "license": figshare_metadata.get("license"),
            "url": figshare_metadata.get("url_public_api"),
        },
        "assets": verified_model_assets,
        "verified_asset_count": sum(
            1
            for item in verified_model_assets
            if item["metadata_present"] and item["name_matches"] and item["md5_matches"]
        ),
        "failed_asset_count": sum(
            1
            for item in verified_model_assets
            if not (
                item["metadata_present"]
                and item["name_matches"]
                and item["md5_matches"]
            )
        ),
        "payload_bytes_downloaded": 0,
    }
    atomic_json(args.output_dir / "public_asset_metadata_report.json", metadata_report)

    outcome = "completed" if metadata_report["failed_asset_count"] == 0 else "failed"
    print(
        json.dumps(
            {
                "event": "ep21_public_asset_audit_finished",
                "pid": os.getpid(),
                "outcome": outcome,
                "asset_count": len(inventory),
                "model_asset_metadata_records": len(verified_model_assets),
                "verified_model_assets": metadata_report["verified_asset_count"],
                "payload_bytes_downloaded": 0,
                "output_dir": str(args.output_dir),
            }
        ),
        flush=True,
    )
    return 0 if metadata_report["failed_asset_count"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
