#!/usr/bin/env python3
"""Reconcile every tagged public dependency with Figshare metadata only.

No listed file body is requested.  Source-study empirical, feature, fitted,
decoded, and reconstruction payloads remain unopened; only the two public
``files.json`` manifests and three official article-metadata JSON responses
are read.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import urlparse

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
import verified_source_archive as source_archive_verifier


ARCHIVE_MD5 = "bb80dcc1a038737d6d8fc2c95b3e5586"
ARCHIVE_SIZE_BYTES = 23_110_908
MANIFEST_SUFFIXES = {
    "data/files.json": "data/files.json",
    "reconstruction/data/files.json": "reconstruction/data/files.json",
}
ARTICLE_EXPECTATIONS = {
    17985578: {
        "title": "Inter-individual deep image reconstruction",
        "license": {
            "value": 1,
            "name": "CC BY 4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/",
        },
        "logical_asset_count": 15,
        "physical_file_count": 15,
        "declared_size_bytes": 4561847436,
    },
    26860954: {
        "title": "Inter-individual and inter-site neural code conversion",
        "license": {
            "value": 1,
            "name": "CC BY 4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/",
        },
        "logical_asset_count": 41,
        "physical_file_count": 43,
        "declared_size_bytes": 248372559801,
    },
    21564384: {
        "title": "brain-decoding-cookbook",
        "license": {
            "value": 3,
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        "logical_asset_count": 4,
        "physical_file_count": 4,
        "declared_size_bytes": 1647561742,
    },
}
EXPECTED_GROUP_ROWS = {
    ("fmri_training", 17985578): (5, 5, 3384947172),
    ("fmri_test", 17985578): (10, 10, 1176900264),
    ("stimulus_feature", 26860954): (19, 19, 65101770416),
    ("pre-trained-converters", 26860954): (1, 1, 18040345242),
    ("pre-trained-decoders", 26860954): (19, 21, 154947898244),
    ("test_image-true_features", 26860954): (1, 1, 2696807715),
    ("recon_demo", 26860954): (1, 1, 7585738184),
    ("recon_demo", 21564384): (4, 4, 1647561742),
}
FIGSHARE_FILE_ID = re.compile(r"/files/(\d+)(?:[/?#].*)?$")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def safe_archive_name(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def select_manifest_name(names: list[str], label: str) -> str:
    suffix = MANIFEST_SUFFIXES[label]
    matches = [name for name in names if name.endswith(suffix)]
    if label == "data/files.json":
        matches = [
            name
            for name in matches
            if not name.endswith("/reconstruction/data/files.json")
        ]
    if len(matches) != 1:
        raise ValueError(f"expected one safe archive member for {label!r}, got {matches}")
    return matches[0]


def read_manifests(archive: zipfile.ZipFile) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    safe_names = [name for name in archive.namelist() if safe_archive_name(name)]
    for label in MANIFEST_SUFFIXES:
        manifests[label] = json.loads(
            archive.read(select_manifest_name(safe_names, label))
        )
    return manifests


def url_checksum_pairs(file_record: Mapping[str, Any]) -> list[tuple[str, str]]:
    raw_urls = file_record["url"]
    raw_md5s = file_record["md5sum"]
    urls = [raw_urls] if isinstance(raw_urls, str) else list(raw_urls)
    md5s = [raw_md5s] if isinstance(raw_md5s, str) else list(raw_md5s)
    if len(urls) != len(md5s) or not urls:
        raise ValueError("manifest URL/MD5 cardinality mismatch")
    pairs: list[tuple[str, str]] = []
    for url, checksum in zip(urls, md5s):
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in {
            "figshare.com",
            "www.figshare.com",
            "ndownloader.figshare.com",
        }:
            raise ValueError(f"non-Figshare file URL in pinned manifest: {url}")
        match = FIGSHARE_FILE_ID.search(url)
        if match is None:
            raise ValueError(f"non-Figshare file URL in pinned manifest: {url}")
        if not re.fullmatch(r"[0-9a-f]{32}", checksum):
            raise ValueError(f"invalid manifest MD5 for {url}")
        pairs.append((url, checksum))
    return pairs


def flatten_manifests(
    manifests: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    parts: list[dict[str, Any]] = []
    logical_count = 0
    for source_manifest, manifest in manifests.items():
        for group, group_record in manifest.items():
            for file_record in group_record["files"]:
                logical_count += 1
                logical_key = f"{source_manifest}:{group}:{file_record['name']}"
                pairs = url_checksum_pairs(file_record)
                for part_index, (url, checksum) in enumerate(pairs):
                    match = FIGSHARE_FILE_ID.search(url)
                    assert match is not None
                    parts.append(
                        {
                            "source_manifest": source_manifest,
                            "group": group,
                            "logical_asset": file_record["name"],
                            "logical_key": logical_key,
                            "part_index": part_index,
                            "figshare_file_id": int(match.group(1)),
                            "manifest_url": url,
                            "manifest_md5": checksum,
                        }
                    )
    ids = [part["figshare_file_id"] for part in parts]
    if len(ids) != len(set(ids)):
        raise ValueError("a Figshare file ID occurs more than once in the manifests")
    return parts, logical_count


def fetch_article(article_id: int, timeout: int = 60) -> dict[str, Any]:
    url = f"https://api.figshare.com/v2/articles/{article_id}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "EP21-Figshare-metadata-audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.geturl() != url:
            raise ValueError(f"unexpected Figshare API redirect for article {article_id}")
        return json.load(response)


def reconcile(
    parts: list[dict[str, Any]],
    logical_count: int,
    articles: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    metadata_by_file_id: dict[int, tuple[int, Mapping[str, Any]]] = {}
    for article_id, article in articles.items():
        for file_record in article.get("files", []):
            file_id = int(file_record["id"])
            if file_id in metadata_by_file_id:
                raise ValueError(f"Figshare file ID {file_id} appears in multiple articles")
            metadata_by_file_id[file_id] = (article_id, file_record)

    reconciled_parts: list[dict[str, Any]] = []
    unresolved: list[int] = []
    supplied_mismatches: list[int] = []
    computed_mismatches: list[int] = []
    for part in parts:
        file_id = part["figshare_file_id"]
        resolved = metadata_by_file_id.get(file_id)
        if resolved is None:
            unresolved.append(file_id)
            continue
        article_id, metadata = resolved
        supplied_matches = metadata.get("supplied_md5") == part["manifest_md5"]
        computed_matches = metadata.get("computed_md5") == part["manifest_md5"]
        if not supplied_matches:
            supplied_mismatches.append(file_id)
        if not computed_matches:
            computed_mismatches.append(file_id)
        reconciled_parts.append(
            {
                **part,
                "article_id": article_id,
                "figshare_name": metadata.get("name"),
                "figshare_size_bytes": int(metadata["size"]),
                "manifest_vs_figshare_supplied_md5_match": supplied_matches,
                "manifest_vs_figshare_computed_md5_match": computed_matches,
            }
        )

    if unresolved or supplied_mismatches or computed_mismatches:
        raise ValueError(
            "Figshare reconciliation failed: "
            f"unresolved={unresolved}, supplied={supplied_mismatches}, "
            f"computed={computed_mismatches}"
        )

    article_rows: list[dict[str, Any]] = []
    for article_id, article in articles.items():
        expected = ARTICLE_EXPECTATIONS.get(article_id)
        matched = [part for part in reconciled_parts if part["article_id"] == article_id]
        logical_assets = {part["logical_key"] for part in matched}
        row = {
            "id": article_id,
            "url": f"https://api.figshare.com/v2/articles/{article_id}",
            "title": article.get("title"),
            "license": article.get("license"),
            "logical_asset_count": len(logical_assets),
            "physical_file_count": len(matched),
            "declared_size_bytes": sum(part["figshare_size_bytes"] for part in matched),
        }
        if expected is not None:
            for field in [
                "title",
                "license",
                "logical_asset_count",
                "physical_file_count",
                "declared_size_bytes",
            ]:
                if row[field] != expected[field]:
                    raise ValueError(
                        f"article {article_id} {field} mismatch: {row[field]!r}"
                    )
        article_rows.append(row)

    group_rows: list[dict[str, Any]] = []
    group_keys = sorted({(part["group"], part["article_id"]) for part in reconciled_parts})
    for group, article_id in group_keys:
        matched = [
            part
            for part in reconciled_parts
            if part["group"] == group and part["article_id"] == article_id
        ]
        row = {
            "group": group,
            "article_id": article_id,
            "logical_asset_count": len({part["logical_key"] for part in matched}),
            "physical_file_count": len(matched),
            "declared_size_bytes": sum(part["figshare_size_bytes"] for part in matched),
        }
        expected = EXPECTED_GROUP_ROWS.get((group, article_id))
        observed = (
            row["logical_asset_count"],
            row["physical_file_count"],
            row["declared_size_bytes"],
        )
        if expected is not None and observed != expected:
            raise ValueError(f"group reconciliation mismatch for {(group, article_id)}")
        group_rows.append(row)

    return {
        "state": "complete_metadata_only_payload_unopened",
        "method": "official_Figshare_article_metadata_API",
        "embedded_manifest_logical_asset_count": logical_count,
        "physical_url_count": len(parts),
        "unique_figshare_file_id_count": len({part["figshare_file_id"] for part in parts}),
        "resolved_file_id_count": len(reconciled_parts),
        "unresolved_file_ids": unresolved,
        "total_declared_size_bytes": sum(
            part["figshare_size_bytes"] for part in reconciled_parts
        ),
        "articles": sorted(article_rows, key=lambda row: row["id"]),
        "group_reconciliation": group_rows,
        "md5_reconciliation": {
            "algorithm": "md5",
            "entries_compared": len(reconciled_parts),
            "manifest_vs_figshare_supplied_match_count": len(reconciled_parts),
            "manifest_vs_figshare_supplied_mismatch_count": 0,
            "manifest_vs_figshare_computed_match_count": len(reconciled_parts),
            "manifest_vs_figshare_computed_mismatch_count": 0,
            "supplied_mismatches": supplied_mismatches,
            "computed_mismatches": computed_mismatches,
            "payload_hashes_computed_by_ep21": 0,
        },
        "physical_files": sorted(
            reconciled_parts, key=lambda row: row["figshare_file_id"]
        ),
        "payload_file_bodies_downloaded": 0,
        "payload_bytes_downloaded": 0,
        "empirical_or_decoded_payload_accessed": False,
        "license_scope_limitation": "Article-level Figshare metadata only; this does not establish LAION DUA/risk eligibility or downstream checkpoint/stimulus rights.",
        "readiness_effect": "none_laion_remains_blocked",
    }


def validate_expected(report: Mapping[str, Any]) -> None:
    required = {
        "embedded_manifest_logical_asset_count": 60,
        "physical_url_count": 62,
        "unique_figshare_file_id_count": 62,
        "resolved_file_id_count": 62,
        "total_declared_size_bytes": 254581968979,
    }
    for field, expected in required.items():
        if report.get(field) != expected:
            raise ValueError(f"{field} mismatch: {report.get(field)!r} != {expected}")
    observed_groups = {
        (row["group"], row["article_id"]) for row in report["group_reconciliation"]
    }
    if observed_groups != set(EXPECTED_GROUP_ROWS):
        raise ValueError("group/article mapping differs from the pinned expectation")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    started_at = utc_now()
    process_id = os.getpid()
    print(
        json.dumps(
            {
                "event": "ep21_figshare_manifest_metadata_audit_started",
                "pid": process_id,
                "started_at": started_at,
            }
        ),
        flush=True,
    )
    with source_archive_verifier.verified_zip(
        args.archive, ARCHIVE_MD5, ARCHIVE_SIZE_BYTES
    ) as verified:
        observed_md5 = verified.md5
        parts, logical_count = flatten_manifests(read_manifests(verified.archive))
    articles = {article_id: fetch_article(article_id) for article_id in ARTICLE_EXPECTATIONS}
    reconciliation = reconcile(parts, logical_count, articles)
    validate_expected(reconciliation)
    report = {
        "schema_version": "ep21.figshare_manifest_metadata_reconciliation.v1",
        "astra_analysis_id": "source_qualification",
        "attempt": args.attempt,
        "started_at": started_at,
        "completed_at": utc_now(),
        "process_id": process_id,
        "archive_md5": observed_md5,
        **reconciliation,
    }
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "event": "ep21_figshare_manifest_metadata_audit_completed",
                "pid": process_id,
                "output": str(args.output),
                "payload_bytes_downloaded": 0,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
