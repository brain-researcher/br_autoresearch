#!/usr/bin/env python3
"""Inspect only ZIP directory metadata in one named public author-code archive.

No archive member is read, decompressed or extracted. A server that does not
honor exact byte ranges stops this lookup; there is no full-download fallback.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import struct
import urllib.error
import urllib.request


ARCHIVE_NAME = "Qt_Cplusplus_LED_ECoG_2048_64x16grid_for_Phase_Reversal.zip"
ARCHIVE_URL = f"https://zenodo.org/api/records/10825275/files/{ARCHIVE_NAME}/content"
ARCHIVE_BYTES = 293071180  # Official record 10825275 file metadata.
MAX_DIRECTORY_BYTES = 8 * 1024 * 1024  # I/O bound, not a scientific trial cap.
TARGET = "eleccoords_rathd.mat"
CENTRAL = struct.Struct("<4s6H3I5H2I")
END = struct.Struct("<4s4H2IH")


class LookupStopped(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise LookupStopped(message)


def exact_range(first, last, requests):
    requested = f"bytes={first}-{last}"
    item = {"requested_range": requested, "response_status": None,
            "response_content_range": None, "response_content_length": None,
            "body_bytes_read": 0}
    requests.append(item)
    request = urllib.request.Request(ARCHIVE_URL, headers={
        "Range": requested, "Accept-Encoding": "identity",
        "User-Agent": "EP20-public-source-directory-inspection",
    })
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as exc:
        item["response_status"] = exc.code
        item["response_content_range"] = exc.headers.get("Content-Range")
        item["response_content_length"] = exc.headers.get("Content-Length")
        exc.close()  # Do not read an error body or retry with a full request.
        raise LookupStopped(f"HTTP {item['response_status']}; no response body read") from None
    except urllib.error.URLError as exc:
        raise LookupStopped(f"range request unavailable: {type(exc.reason).__name__}") from None
    with response:
        item["response_status"] = response.status
        item["response_content_range"] = response.headers.get("Content-Range")
        item["response_content_length"] = response.headers.get("Content-Length")
        require(response.status == 206, "server did not return206; body not read, no fallback")
        expected_range = f"bytes {first}-{last}/{ARCHIVE_BYTES}"
        require(item["response_content_range"] == expected_range,
                "returned byte range or archive size differs; body not read")
        require(response.headers.get("Content-Encoding", "identity") == "identity",
                "encoded response is not the requested raw byte range")
        count = last - first + 1
        if item["response_content_length"] is not None:
            require(item["response_content_length"] == str(count),
                    "Content-Length differs from exact metadata range")
        body = response.read(count)
        item["body_bytes_read"] = len(body)
        require(len(body) == count, "incomplete metadata range")
        return body


def directory_entries(raw, count):
    position = 0
    for _ in range(count):
        require(position + CENTRAL.size <= len(raw), "truncated central-directory header")
        fields = CENTRAL.unpack_from(raw, position)
        require(fields[0] == b"PK\x01\x02", "unexpected central-directory record")
        flags, compressed, uncompressed = fields[3], fields[8], fields[9]
        name_length, extra_length, comment_length = fields[10:13]
        require(fields[13] == 0, "multidisk member not supported in this bounded lookup")
        record_end = position + CENTRAL.size + name_length + extra_length + comment_length
        require(record_end <= len(raw), "truncated central-directory record")
        filename = raw[position + CENTRAL.size:position + CENTRAL.size + name_length]
        filename = filename.decode("utf-8" if flags & 0x800 else "cp437", errors="strict")
        yield {"member_name": filename, "compressed_bytes": compressed,
               "uncompressed_bytes": uncompressed,
               "encrypted_flag": bool(flags & 1)}
        position = record_end
    require(position == len(raw), "unparsed central-directory trailing metadata")


def inspect(report):
    # Reading the final22bytes reaches only ZIP end/comment metadata. Unlike a
    # generic65KiB tail request, it cannot overlap preceding member contents.
    tail = exact_range(ARCHIVE_BYTES - END.size, ARCHIVE_BYTES - 1,
                       report["requests"])
    fields = END.unpack(tail)
    require(fields[0] == b"PK\x05\x06",
            "EOCD not at final22bytes (e.g. archive comment); stop without wider tail")
    _, disk, directory_disk, entries_on_disk, entries_total, size, offset, comment_length = fields
    require(disk == directory_disk == 0 and entries_on_disk == entries_total,
            "multipart ZIP unsupported")
    require(comment_length == 0, "commented ZIP unsupported in this bounded lookup")
    require(entries_total != 0xFFFF and size != 0xFFFFFFFF and offset != 0xFFFFFFFF,
            "ZIP64 directory unsupported in this bounded lookup")
    require(0 < size <= MAX_DIRECTORY_BYTES, "central directory exceeds selected metadata I/O bound")
    require(offset + size == ARCHIVE_BYTES - END.size,
            "directory does not end at verified EOCD; stop without extra ranges")
    report["central_directory"] = {"offset": offset, "bytes": size,
                                   "declared_member_count": entries_total}
    central = exact_range(offset, offset + size - 1, report["requests"])
    matches, documentary, suffix_counts = [], [], Counter()
    for item in directory_entries(central, entries_total):
        name = PurePosixPath(item["member_name"].replace("\\", "/"))
        basename = name.name.casefold()
        suffix_counts[name.suffix.casefold() or "<no_suffix>"] += 1
        if basename == TARGET:
            matches.append(item)
        if name.suffix.casefold() in {".m", ".py", ".cpp", ".h", ".txt", ".md", ".json"}:
            if re.search(r"rat|rhd|nwb|coord|convert|readme|filter|reference", basename):
                documentary.append(item)
    report.update({
        "lookup_completed": True, "exact_rat_map_members": matches,
        "documentary_filename_candidates": documentary,
        "member_suffix_counts": dict(sorted(suffix_counts.items())),
        "member_count": entries_total,
        "rat5_channel_mapping_established": False,
        "source_qualification_complete": False,
        "next_interpretation": (
            "A filename match would need authenticated Rat5/NWB channel linkage and "
            "separate authorized metadata inspection; absence here is limited to this archive"),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname(),
            "run this lookup only in coordinator-owned Slurm compute")
    require(not args.out.exists(), "attempt output exists; preserve it")
    args.out.mkdir(parents=True)
    report = {
        "analysis_id": "model_specification", "task": "rat_map_archive_directory_lookup",
        "source_record": "https://zenodo.org/records/10825275",
        "source_archive": ARCHIVE_NAME, "declared_archive_bytes": ARCHIVE_BYTES,
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "requests": [], "lookup_completed": False,
        "archive_member_payloads_read": False, "full_archive_downloaded": False,
        "neural_values_read": False, "final_animal_accessed": False,
        "archive_members_extracted": False, "rat5_channel_mapping_established": False,
        "source_qualification_complete": False,
    }
    try:
        inspect(report)
    except (LookupStopped, OSError, UnicodeError, struct.error) as exc:
        report["stopped_reason"] = str(exc)
    (args.out / "directory_lookup.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"lookup_completed": report["lookup_completed"],
                      "exact_rat_map_members": len(report.get("exact_rat_map_members", [])),
                      "stopped_reason": report.get("stopped_reason"),
                      "source_qualification_complete": False}))
    return 0 if report["lookup_completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
