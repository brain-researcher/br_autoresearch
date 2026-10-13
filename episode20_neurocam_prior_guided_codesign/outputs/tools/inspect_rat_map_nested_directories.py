#!/usr/bin/env python3
"""Bounded directory-only lookup inside the two recorded author-archive ZIPs.

Only stored (method 0) nested containers permit direct directory byte ranges.
Compressed containers remain unresolved; no member extraction or fallback.
The outer ZIP member bytes read are nested ZIP *metadata*, never inner payloads.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import socket
import struct
import time
import urllib.error
import urllib.request


ARCHIVE_NAME = "Qt_Cplusplus_LED_ECoG_2048_64x16grid_for_Phase_Reversal.zip"
ARCHIVE_URL = f"https://zenodo.org/api/records/10825275/files/{ARCHIVE_NAME}/content"
ARCHIVE_BYTES = 293071180
OUTER_OFFSET = 291334742
OUTER_DIRECTORY_BYTES = 1736416
OUTER_COUNT = 8960
MAX_DIRECTORY_BYTES = 8 * 1024 * 1024
MAX_TRANSFER_BYTES = 20 * 1024 * 1024
MAX_SECONDS = 240
TARGET = "eleccoords_rathd.mat"
CENTRAL = struct.Struct("<4s6H3I5H2I")
LOCAL = struct.Struct("<4s5H3I2H")
END = struct.Struct("<4s4H2IH")


class LookupStopped(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise LookupStopped(message)


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Return redirect errors to the caller without draining their bodies."""

    def http_error_302(self, request, response, code, message, headers):
        raise urllib.error.HTTPError(request.full_url, code, message, headers, response)

    http_error_301 = http_error_302
    http_error_303 = http_error_302
    http_error_307 = http_error_302
    http_error_308 = http_error_302


class ExactRanges:
    """No retry, full response, error body or encoded body is consumed."""

    def __init__(self, requests, max_bytes, deadline):
        self.requests = requests
        self.max_bytes = max_bytes
        self.deadline = deadline
        self.requested_bytes = 0
        self.opener = urllib.request.build_opener(RejectRedirects())

    def get(self, first, last, purpose):
        remaining = self.deadline - time.monotonic()
        require(remaining > 0, "internal wall-time budget exhausted")
        require(0 <= first <= last < ARCHIVE_BYTES, "range outside selected archive")
        count = last - first + 1
        require(self.requested_bytes + count <= self.max_bytes,
                "selected metadata transfer budget would be exceeded")
        self.requested_bytes += count
        item = {"purpose": purpose, "requested_range": f"bytes={first}-{last}",
                "response_status": None, "body_bytes_read": 0}
        self.requests.append(item)
        request = urllib.request.Request(ARCHIVE_URL, headers={
            "Range": item["requested_range"], "Accept-Encoding": "identity",
            "User-Agent": "EP20-bounded-nested-directory-inspection",
        })
        try:
            response = self.opener.open(request, timeout=min(30, remaining))
        except urllib.error.HTTPError as exc:
            item["response_status"] = exc.code
            exc.close()
            raise LookupStopped(f"HTTP {exc.code}; error body not read") from None
        except urllib.error.URLError as exc:
            raise LookupStopped(f"range unavailable: {type(exc.reason).__name__}") from None
        with response:
            item.update(response_status=response.status,
                        response_content_range=response.headers.get("Content-Range"),
                        response_content_length=response.headers.get("Content-Length"))
            require(response.status == 206, "server did not honor range; body not read")
            require(item["response_content_range"] == f"bytes {first}-{last}/{ARCHIVE_BYTES}",
                    "range/size mismatch; body not read")
            require(response.headers.get("Content-Encoding", "identity") == "identity",
                    "encoded response; body not read")
            require(item["response_content_length"] in (None, str(count)),
                    "Content-Length mismatch; body not read")
            chunks = []
            while item["body_bytes_read"] < count:
                require(time.monotonic() < self.deadline, "internal wall-time budget exhausted")
                chunk = response.read(min(65536, count - item["body_bytes_read"]))
                require(bool(chunk), "truncated metadata range")
                item["body_bytes_read"] += len(chunk)
                chunks.append(chunk)
            return b"".join(chunks)


def directory_entries(raw, count):
    position = 0
    entries = []
    for _ in range(count):
        require(position + CENTRAL.size <= len(raw), "truncated central header")
        fields = CENTRAL.unpack_from(raw, position)
        require(fields[0] == b"PK\x01\x02", "invalid central header")
        flags, method, crc, compressed, uncompressed = fields[3:5] + fields[7:10]
        name_len, extra_len, comment_len = fields[10:13]
        end = position + CENTRAL.size + name_len + extra_len + comment_len
        require(end <= len(raw), "truncated central record")
        name_bytes = raw[position + CENTRAL.size:position + CENTRAL.size + name_len]
        name = name_bytes.decode("utf-8" if flags & 0x800 else "cp437", errors="strict")
        entries.append({"member_name": name, "flags": flags, "method": method,
                        "crc32": crc, "compressed_bytes": compressed,
                        "uncompressed_bytes": uncompressed,
                        "local_header_offset": fields[16], "disk": fields[13]})
        position = end
    require(position == len(raw), "central directory trailing metadata")
    return entries


def directory_from_eocd(raw, container_bytes):
    require(len(raw) == END.size, "incomplete EOCD")
    fields = END.unpack(raw)
    require(fields[0] == b"PK\x05\x06", "EOCD not at final22 bytes; no wider tail read")
    _, disk, central_disk, disk_count, count, size, offset, comment = fields
    require(disk == central_disk == 0 and disk_count == count, "multipart ZIP unsupported")
    require(comment == 0, "commented ZIP unsupported")
    require(count != 0xFFFF and size != 0xFFFFFFFF and offset != 0xFFFFFFFF,
            "ZIP64 unsupported")
    require(0 < size <= MAX_DIRECTORY_BYTES, "nested directory outside metadata size bound")
    require(offset + size == container_bytes - END.size, "directory does not end at EOCD")
    return {"offset": offset, "bytes": size, "declared_member_count": count}


def basename(name):
    return PurePosixPath(name.replace("\\", "/")).name.casefold()


def stored_container_start(entry, reader):
    require(entry["disk"] == 0, "multipart member unsupported")
    require(not entry["flags"] & 0x41, "encrypted container unsupported")
    require(entry["method"] == 0, "compressed container unresolved; no decompression")
    size = entry["compressed_bytes"]
    require(size == entry["uncompressed_bytes"] and END.size <= size < 0xFFFFFFFF,
            "stored container size invalid or ZIP64")
    offset = entry["local_header_offset"]
    require(0 <= offset and offset + LOCAL.size <= OUTER_OFFSET,
            "local header outside outer payload region")
    raw = reader.get(offset, offset + LOCAL.size - 1, "nested_container_local_header")
    fields = LOCAL.unpack(raw)
    require(fields[0] == b"PK\x03\x04", "invalid local header")
    require(fields[2] == entry["flags"] and fields[3] == entry["method"],
            "local/central flags or method mismatch")
    expected = (entry["crc32"], size, entry["uncompressed_bytes"])
    observed = fields[6:9]
    if entry["flags"] & 8:  # A subsequent descriptor is not read.
        require(all(value in (0, wanted) for value, wanted in zip(observed, expected)),
                "descriptor-style local size/CRC mismatch")
    else:
        require(observed == expected, "local/central size or CRC mismatch")
    name_len, extra_len = fields[9:11]
    header_end = offset + LOCAL.size + name_len + extra_len
    require(name_len > 0 and header_end + size <= OUTER_OFFSET,
            "container extent exceeds outer member region")
    variable = reader.get(offset + LOCAL.size, header_end - 1,
                          "nested_container_local_name_extra")
    local_name = variable[:name_len].decode("utf-8" if entry["flags"] & 0x800 else "cp437",
                                           errors="strict")
    require(local_name == entry["member_name"], "local/central member name mismatch")
    return header_end


def inspect_nested(entry, reader):
    result = dict(entry, directory_inspected=False, inner_member_payloads_read=False,
                  compressed_container_decoded=False)
    try:
        start = stored_container_start(entry, reader)
        size = entry["uncompressed_bytes"]
        result["outer_data_start"] = start
        tail = reader.get(start + size - END.size, start + size - 1, "nested_EOCD_only")
        directory = directory_from_eocd(tail, size)
        result["central_directory"] = directory
        first = start + directory["offset"]
        raw = reader.get(first, first + directory["bytes"] - 1, "nested_central_directory_only")
        entries = directory_entries(raw, directory["declared_member_count"])
        exact, documentary = [], []
        suffixes = Counter()
        for item in entries:
            name = basename(item["member_name"])
            suffix = PurePosixPath(name).suffix
            suffixes[suffix or "<no_suffix>"] += 1
            if name == TARGET:
                exact.append(item)
            if suffix in {".m", ".py", ".cpp", ".h", ".txt", ".md", ".json"} and re.search(
                    r"rat.?5|coord|convert|rhd.*nwb|nwb.*rhd", name):
                documentary.append(item)
        result.update(directory_inspected=True, exact_rat_map_members=exact,
                      documentary_filename_candidates=documentary,
                      member_suffix_counts=dict(sorted(suffixes.items())),
                      match_scope="this directly embedded ZIP directory only; no recursion")
    except LookupStopped as exc:
        result["unresolved_reason"] = str(exc)
    return result


def inspect(report, reader):
    raw = reader.get(OUTER_OFFSET, OUTER_OFFSET + OUTER_DIRECTORY_BYTES - 1,
                     "previously_verified_outer_central_directory")
    entries = directory_entries(raw, OUTER_COUNT)
    nested = [item for item in entries if basename(item["member_name"]).endswith(".zip")]
    require(len(nested) == 2, "outer directory no longer identifies exactly two nested ZIPs")
    report["nested_zip_members"] = nested
    report["nested_directories"] = []
    for entry in nested:
        report["nested_directories"].append(inspect_nested(entry, reader))
    report["inspection_completed"] = True
    report["all_nested_directories_inspected"] = all(
        item["directory_inspected"] for item in report["nested_directories"])


def validate_prior(report):
    require(report.get("lookup_completed") is True
            and report.get("source_archive") == ARCHIVE_NAME
            and report.get("declared_archive_bytes") == ARCHIVE_BYTES
            and report.get("central_directory") == {
                "offset": OUTER_OFFSET, "bytes": OUTER_DIRECTORY_BYTES,
                "declared_member_count": OUTER_COUNT}
            and report.get("member_suffix_counts", {}).get(".zip") == 2,
            "prior directory evidence differs from selected exact source")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    parser.add_argument("--max-transfer-bytes", type=int, default=MAX_TRANSFER_BYTES)
    args = parser.parse_args()
    require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname(),
            "run only in coordinator-owned Slurm compute")
    require(0 < args.max_seconds <= MAX_SECONDS and 0 < args.max_transfer_bytes <= MAX_TRANSFER_BYTES,
            "resource bounds may be reduced, not enlarged")
    validate_prior(json.loads(args.outer_report.read_text(encoding="utf-8")))
    require(not args.out.exists(), "attempt output exists; preserve it")
    args.out.mkdir(parents=True)
    started = time.monotonic()
    report = {
        "analysis_id": "model_specification", "task": "rat_map_nested_directory_lookup",
        "source_archive": ARCHIVE_NAME, "source_url": ARCHIVE_URL,
        "declared_archive_bytes": ARCHIVE_BYTES, "prior_report": str(args.outer_report),
        "hostname": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "max_seconds": args.max_seconds, "max_transfer_bytes": args.max_transfer_bytes,
        "requests": [], "inspection_completed": False,
        "all_nested_directories_inspected": False,
        "inner_member_payloads_read": False, "compressed_container_decoded": False,
        "outer_member_access_scope": "stored nested ZIP directory metadata only",
        "full_archive_downloaded": False, "archive_members_extracted": False,
        "neural_values_read": False, "final_animal_accessed": False,
        "rat5_channel_mapping_established": False, "source_qualification_complete": False,
        "interpretation": "Filename evidence only; map contents and Rat5 channel linkage remain unqualified.",
    }

    def deadline_expired(_signum, _frame):
        raise LookupStopped("internal wall-time budget exhausted")

    previous_handler = signal.signal(signal.SIGALRM, deadline_expired)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    try:
        reader = ExactRanges(report["requests"], args.max_transfer_bytes, started + args.max_seconds)
        inspect(report, reader)
    except (LookupStopped, OSError, UnicodeError, struct.error) as exc:
        report["stopped_reason"] = str(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    report["elapsed_seconds"] = time.monotonic() - started
    report["body_bytes_read"] = sum(item["body_bytes_read"] for item in report["requests"])
    (args.out / "nested_directory_lookup.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in (
        "inspection_completed", "all_nested_directories_inspected", "body_bytes_read", "stopped_reason")}))
    return 0 if report["inspection_completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
