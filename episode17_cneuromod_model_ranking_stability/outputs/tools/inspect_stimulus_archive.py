#!/usr/bin/env python3
"""Read a bounded ZIP directory and select existing development image identities.

No member stream is opened. ZipFile receives an in-memory directory-only ZIP,
never the source archive. Source length and the existing acquisition manifest
are reused; this operation does not recompute the archive's content digest.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import os
from pathlib import Path, PurePosixPath
import resource
import signal
import socket
import stat
import struct
import time
import zipfile

EPISODE = Path(__file__).resolve().parents[2]
ACQUISITION = EPISODE.parent.parent / "br_autoresearch_data/steward_acquisition/cneuromod_images_fmri_pinned_f8f5162_20260922"
ARCHIVE = ACQUISITION / "archives/images_fmri.zip"
MANIFEST = ACQUISITION / "PROVIDER_MANIFEST.tsv"
PROVIDER = ACQUISITION / "provider_git/image_paths_fmri.csv"
ROLES = EPISODE / "outputs/runs/concept_role_assignment/full001"
OUTPUT_PARENT = EPISODE / "outputs/runs/stimulus_features"
EXPECTED_SIZE = 917286854
EXPECTED_MD5 = "bb6aa66c55247e4660cb66cb133e5bd6"
EXPECTED_SHA256 = "f3c41fa59485faab5debb882b15f046253acfc144564abfdcd0782fba7d90912"
TAIL_CAP = 65557
CD_CAP = 16 * 1024 * 1024
ENTRY_CAP = 16384
METADATA_CAP = 32 * 1024 * 1024
OUTPUT_CAP = 256 * 1024
INTERNAL_SECONDS = 60
SELECTION_FIELDS = ["feature_row_zero_based", "provider_image_path", "image_name",
                    "things_category_nr", "uniqueID", "role", "outer_fold",
                    "development_block", "eligible_image", "member_match_count"]


class Refusal(Exception):
    """An intentionally short, nonsecret operational error code."""


def require(condition, code):
    if not condition:
        raise Refusal(code)


def integer(value):
    require(isinstance(value, str) and value.isascii() and value.isdecimal(), "invalid_numeric_identity")
    result = int(value)
    require(result > 0, "nonpositive_identity")
    return result


def bounded_bytes(path, cap):
    with Path(path).open("rb") as stream:
        require(os.fstat(stream.fileno()).st_size <= cap, "metadata_size_cap")
        data = stream.read(cap + 1)
    require(len(data) <= cap, "metadata_size_cap")
    return data


def read_csv(path, delimiter=",", cap=METADATA_CAP):
    data = bounded_bytes(path, cap).decode("utf-8")
    reader = csv.DictReader(io.StringIO(data, newline=""), delimiter=delimiter)
    fields = reader.fieldnames or []
    require(fields and len(fields) == len(set(fields)), "metadata_header")
    result = list(reader)
    require(all(None not in row and all(v is not None for v in row.values()) for row in result), "metadata_row_width")
    return result


def validate_manifest():
    rows = read_csv(MANIFEST, delimiter="\t", cap=16384)
    require(rows == [{"relative_path": "archives/images_fmri.zip", "bytes": str(EXPECTED_SIZE),
                     "provider_md5": EXPECTED_MD5, "local_sha256": EXPECTED_SHA256}], "manifest_identity_changed")


def safe_name(name):
    require(isinstance(name, str) and name and "\0" not in name and "\\" not in name,
            "unsafe_member_name")
    parts = name.split("/")
    require(not name.startswith("/") and all(part not in {".", ".."} for part in parts)
            and ":" not in parts[0] and all(parts[:-1]), "unsafe_member_name")
    return name


def select_development_images(concepts, images, provider, count=32):
    """Exact metadata join; no new role/eligibility decision and no image fallback."""
    concept_by_id = {}
    for row in concepts:
        cid = integer(row["things_category_nr"])
        require(cid not in concept_by_id, "duplicate_concept_identity")
        require(row["uniqueID"] and row["role"] in {"development", "support_calibration", "sealed_audit"}, "invalid_concept_role")
        concept_by_id[cid] = row
    require(len({row["uniqueID"] for row in concepts}) == len(concepts), "duplicate_concept_name")
    provider_by_path = {}
    for row in provider:
        key = row["image_path"]
        require(key not in provider_by_path, "duplicate_provider_path")
        provider_by_path[key] = row
    eligible = defaultdict(list)
    seen = set()
    qualified_counts = Counter()
    for row in images:
        key = safe_name(row["provider_image_path"])
        require(key not in seen and key in provider_by_path, "image_provider_join")
        seen.add(key)
        cid = integer(row["things_category_nr"])
        require(cid in concept_by_id, "image_concept_join")
        concept, source = concept_by_id[cid], provider_by_path[key]
        path = PurePosixPath(key)
        require(len(path.parts) == 3 and path.parts[0] == "images_fmri"
                and path.name == row["image_name"] and path.parts[1] == row["concept_name"] == concept["uniqueID"], "image_path_identity")
        require(row["condition"] == source["condition"] == "exp", "retained_condition_changed")
        for local, remote in (("things_category_nr", "things_category_nr"),
                              ("provider_category_nr", "category_nr"),
                              ("provider_exemplar_nr", "exemplar_nr")):
            require(integer(row[local]) == integer(source[remote]), "provider_numeric_identity")
        require(integer(row["provider_category_nr"]) == integer(concept["provider_category_nr"]), "concept_provider_identity")
        for field in ("role", "outer_fold", "development_block", "audit_block"):
            require(row[field] == concept[field], "inherited_role_mismatch")
        require(row["eligible_image"] in {"True", "False"}, "eligibility_not_boolean")
        if row["eligible_image"] == "True":
            require(row["common_to_four"] == "True" and row["failure_reasons"] == "", "qualified_image_identity")
            qualified_counts[cid] += 1
            if row["role"] == "development":
                eligible[cid].append(row)
    for cid, concept in concept_by_id.items():
        require(qualified_counts[cid] == integer(concept["qualified_image_count"]), "qualified_count_changed")
    ids = sorted(cid for cid, row in concept_by_id.items() if row["role"] == "development")
    require(len(ids) >= count, "insufficient_development_concepts")
    selected = []
    for index, cid in enumerate(ids[:count]):
        require(eligible[cid], "missing_selected_eligible_image")
        row = min(eligible[cid], key=lambda item: item["provider_image_path"].encode("utf-8"))
        selected.append({"feature_row_zero_based": index, "provider_image_path": row["provider_image_path"],
                         "image_name": row["image_name"], "things_category_nr": cid,
                         "uniqueID": concept_by_id[cid]["uniqueID"], "role": "development",
                         "outer_fold": row["outer_fold"], "development_block": row["development_block"],
                         "eligible_image": True, "member_match_count": ""})
    return selected


def load_selection():
    report = json.loads(bounded_bytes(ROLES / "assignment_report.json", 256 * 1024))
    require(report.get("status") == "completed" and report.get("mode") == "full"
            and report.get("metadata_assignments_complete") is True, "assignments_incomplete")
    require(report.get("preserved_metadata") == {"concepts": 720, "images": 4320, "presentations": 49980, "runs": 833}, "assignment_scope_changed")
    require(report.get("role_counts") == {"development": 480, "support_calibration": 120, "sealed_audit": 120}, "assignment_capacity_changed")
    concepts = read_csv(ROLES / "concept_assignments.csv")
    images = read_csv(ROLES / "image_assignments.csv")
    require(len(concepts) == 720 and len(images) == 4320, "assignment_row_count")
    require(Counter(row["role"] for row in concepts) == report["role_counts"], "assignment_role_counts")
    require(sum(row["eligible_image"] == "True" for row in images) == 3634, "qualified_image_count_changed")
    return select_development_images(concepts, images, read_csv(PROVIDER))


def find_eocd(tail, file_size):
    candidates = []
    start = 0
    while True:
        pos = tail.find(b"PK\x05\x06", start)
        if pos < 0:
            break
        if pos + 22 <= len(tail):
            fields = struct.unpack_from("<4s4H2IH", tail, pos)
            if pos + 22 + fields[-1] == len(tail):
                candidates.append((pos, fields))
        start = pos + 1
    require(len(candidates) == 1, "ambiguous_or_missing_classic_eocd")
    pos, (_, disk, cd_disk, disk_entries, entries, size, offset, _) = candidates[0]
    require(pos < 20 or tail[pos - 20:pos - 16] != b"PK\x06\x07", "zip64_locator")
    require(disk == cd_disk == 0 and disk_entries == entries, "multidisk_zip")
    require(entries != 0xFFFF and size != 0xFFFFFFFF and offset != 0xFFFFFFFF, "zip64_eocd")
    require(0 < entries <= ENTRY_CAP and 0 < size <= CD_CAP, "central_directory_cap")
    eocd_offset = file_size - len(tail) + pos
    require(offset >= 0 and offset + size == eocd_offset, "unknown_directory_layout")
    return {"entry_count": entries, "central_directory_bytes": size,
            "central_directory_offset": offset, "eocd_offset": eocd_offset,
            "eocd_comment_bytes": len(tail) - pos - 22}


def extra_facts(extra):
    pos, ids, aes = 0, [], []
    while pos < len(extra):
        require(pos + 4 <= len(extra), "truncated_extra_header")
        ident, length = struct.unpack_from("<HH", extra, pos)
        pos += 4
        require(pos + length <= len(extra), "truncated_extra_value")
        value = extra[pos:pos + length]
        pos += length
        require(ident != 0x0001, "zip64_entry")
        ids.append(ident)
        if ident == 0x9901:
            require(length == 7, "aes_extra_length")
            version, vendor, strength, method = struct.unpack("<H2sBH", value)
            aes.append({"vendor_version": version, "vendor_is_AE": vendor == b"AE",
                        "strength_code": strength, "actual_compression_method": method})
    require(len(aes) <= 1, "duplicate_aes_extra")
    return {"extra_field_ids": sorted(set(ids))[:32], "extra_field_distinct_count": len(set(ids)),
            "extra_field_ids_truncated": len(set(ids)) > 32, "aes": aes[0] if aes else None}


def parse_directory(directory, facts):
    """Fence every raw record, then let ZipFile decode only the bounded directory."""
    pos, parsed_extra = 0, []
    for _ in range(facts["entry_count"]):
        require(pos + 46 <= len(directory) and directory[pos:pos + 4] == b"PK\x01\x02", "central_record_layout")
        compressed, uncompressed = struct.unpack_from("<II", directory, pos + 20)
        nlen, xlen, clen, disk = struct.unpack_from("<4H", directory, pos + 28)
        local_offset = struct.unpack_from("<I", directory, pos + 42)[0]
        require(disk == 0, "multidisk_entry")
        require(compressed != 0xFFFFFFFF and uncompressed != 0xFFFFFFFF and local_offset != 0xFFFFFFFF, "zip64_entry")
        end = pos + 46 + nlen + xlen + clen
        require(nlen > 0 and end <= len(directory), "truncated_central_record")
        require(local_offset < facts["central_directory_offset"], "member_offset_outside_payload_region")
        parsed_extra.append(extra_facts(directory[pos + 46 + nlen:pos + 46 + nlen + xlen]))
        pos = end
    require(pos == len(directory), "unknown_central_trailer")
    wrapper = directory + struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0,
                                      facts["entry_count"], facts["entry_count"], len(directory), 0, 0)
    with zipfile.ZipFile(io.BytesIO(wrapper), "r", allowZip64=False) as archive:
        infos = archive.infolist()
    require(len(infos) == facts["entry_count"], "zip_parser_count")
    entries = []
    for info, extras in zip(infos, parsed_extra):
        require(info.filename == info.orig_filename, "truncated_member_name")
        safe_name(info.filename)
        entries.append({"name": info.filename, "is_directory": info.is_dir(),
                        "compressed_bytes": info.compress_size, "uncompressed_bytes": info.file_size,
                        "crc32": f"{info.CRC:08x}", "general_purpose_flags": info.flag_bits,
                        "encrypted": bool(info.flag_bits & 1), "strong_encryption_flag": bool(info.flag_bits & 64),
                        "compression_method": info.compress_type, "extract_version": info.extract_version,
                        **extras})
    return entries


def inspect_stream(stream, file_size):
    tail_bytes = min(file_size, TAIL_CAP)
    require(file_size >= 22, "archive_too_short")
    stream.seek(file_size - tail_bytes)
    tail = stream.read(tail_bytes)
    require(len(tail) == tail_bytes, "short_eocd_read")
    facts = find_eocd(tail, file_size)
    stream.seek(facts["central_directory_offset"])
    directory = stream.read(facts["central_directory_bytes"])
    require(len(directory) == facts["central_directory_bytes"], "short_directory_read")
    entries = parse_directory(directory, facts)
    facts.update({"tail_bytes_read": tail_bytes, "directory_bytes_read": len(directory),
                  "total_requested_source_bytes": tail_bytes + len(directory),
                  "zipfile_received_directory_only": True})
    return facts, entries


def inspect_archive():
    with ARCHIVE.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size == EXPECTED_SIZE, "archive_length_or_type_changed")
        facts, entries = inspect_stream(stream, before.st_size)
        after = os.fstat(stream.fileno())
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "archive_changed_during_read")
    return facts, entries


def histogram(values):
    counts = Counter(values)
    return {"counts": [{"value": key, "count": counts[key]} for key in sorted(counts)[:32]],
            "distinct_count": len(counts), "truncated": len(counts) > 32}


def match_selection(selected, entries):
    by_name = defaultdict(list)
    for entry in entries:
        by_name[entry["name"]].append(entry)
    matches = []
    for row in selected:
        found = by_name[row["provider_image_path"]]
        row["member_match_count"] = len(found)
        matches.append({"feature_row_zero_based": row["feature_row_zero_based"], "match_count": len(found),
                        "member": {k: v for k, v in found[0].items() if k != "name"} if len(found) == 1 else None})
    complete = all(row["match_count"] == 1 and not row["member"]["is_directory"] for row in matches)
    return complete, matches


def base_report():
    return {"analysis_id": "stimulus_archive_metadata", "status": "started",
            "archive_inspection_complete": False, "selected_member_mapping_complete": False,
            "selected_image_count": 32, "source_archive": str(ARCHIVE), "source_archive_bytes": EXPECTED_SIZE,
            "recorded_provider_md5": EXPECTED_MD5, "recorded_local_sha256": EXPECTED_SHA256,
            "archive_digest_recomputed": False, "identity_basis": "existing_exact_provider_manifest_and_current_length",
            "prefix_alias_applied": False, "member_streams_opened": False, "password_supplied": False,
            "decryption_selected": False, "image_pixels_decoded": False, "model_features_computed": False,
            "audit_pixels_or_features_opened": False, "neural_values_read": False,
            "source_roles_complete": False, "feature_pilot_selected": False,
            "caps": {"tail_bytes": TAIL_CAP, "central_directory_bytes": CD_CAP, "entries": ENTRY_CAP,
                     "metadata_csv_bytes_each": METADATA_CAP, "combined_output_bytes": OUTPUT_CAP,
                     "internal_seconds": INTERNAL_SECONDS}, "errors": []}


def mark_complete(report, mapping_complete):
    report["archive_inspection_complete"] = True
    report["selected_member_mapping_complete"] = mapping_complete
    report["status"] = "completed" if mapping_complete else "unresolved"


def render_outputs(report, selected):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SELECTION_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(selected)
    csv_bytes = buffer.getvalue().encode("utf-8")
    report_bytes = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    require(len(csv_bytes) + len(report_bytes) <= OUTPUT_CAP, "output_size_cap")
    return csv_bytes, report_bytes


def run_attempt(out, arm_deadline=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    report, selected = base_report(), []
    started = time.monotonic()

    def check_deadline():
        require(time.monotonic() - started < INTERNAL_SECONDS, "internal_deadline")

    def deadline_handler(_signum, _frame):
        raise Refusal("internal_deadline_or_termination")

    previous = {}
    if arm_deadline:
        for sig in (signal.SIGALRM, signal.SIGTERM):
            previous[sig] = signal.signal(sig, deadline_handler)
        signal.setitimer(signal.ITIMER_REAL, INTERNAL_SECONDS)
    try:
        validate_manifest()
        selected = load_selection()
        check_deadline()
        facts, entries = inspect_archive()
        report["directory"] = facts
        report["compression_methods"] = histogram(entry["compression_method"] for entry in entries)
        report["general_purpose_flags"] = histogram(entry["general_purpose_flags"] for entry in entries)
        report["encrypted_entry_count"] = sum(entry["encrypted"] for entry in entries)
        report["directory_entry_count"] = sum(entry["is_directory"] for entry in entries)
        mapping_complete, report["selected_members"] = match_selection(selected, entries)
        del entries
        mark_complete(report, mapping_complete)
        check_deadline()
        render_outputs(report, selected)
    except (Refusal, OSError, ValueError, KeyError, TypeError, UnicodeError, NotImplementedError,
            zipfile.BadZipFile, struct.error) as exc:
        report["status"] = "stopped"
        report["archive_inspection_complete"] = False
        report["selected_member_mapping_complete"] = False
        report["errors"].append(str(exc) if isinstance(exc, Refusal) else type(exc).__name__)
    finally:
        if arm_deadline:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if report["elapsed_seconds"] >= INTERNAL_SECONDS:
        report.update(status="stopped", archive_inspection_complete=False, selected_member_mapping_complete=False)
        report["errors"].append("internal_deadline")
    try:
        csv_bytes, report_bytes = render_outputs(report, selected)
    except Refusal:
        report.update(status="stopped", archive_inspection_complete=False, selected_member_mapping_complete=False)
        report["errors"].append("output_size_cap")
        report.pop("selected_members", None)
        csv_bytes, report_bytes = render_outputs(report, [])
    (out / "selected_images.csv").write_bytes(csv_bytes)
    (out / "archive_metadata_report.json").write_bytes(report_bytes)
    print(json.dumps({key: report[key] for key in ("status", "archive_inspection_complete", "selected_member_mapping_complete", "elapsed_seconds", "errors")}))
    return 0 if report["status"] == "completed" and report["archive_inspection_complete"] and report["selected_member_mapping_complete"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_ID") and "-ln" not in socket.gethostname(), "compute_allocation_required")
    require(args.out.parent.resolve() == OUTPUT_PARENT.resolve(), "output_parent_not_selected")
    return run_attempt(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
