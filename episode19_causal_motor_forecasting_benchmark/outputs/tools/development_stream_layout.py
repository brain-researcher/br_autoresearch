#!/usr/bin/env python3
"""Development-only binary layout and New Segment metadata, without signal reads.

The Brain Products BVCDF specification defines BINARY/MULTIPLEXED storage,
IEEE_FLOAT_32 (four bytes), INT_16 (two bytes), and the optional 20-digit
YYYYMMDDhhmmssffffff New Segment date. Positions are one-based samples.
Sources: https://www.brainproducts.com/support-resources/brainvision-core-data-format-1-0/
https://www.fieldtriptoolbox.org/assets/pdf/BrainVisionCoreFileFormat_1.0_2018-08-02.pdf
https://pressrelease.brainproducts.com/markers/
The headerless binary layout is also implemented by the format writer:
https://pybv.readthedocs.io/en/stable/_modules/pybv/io.html#_write_bveeg_file

Only the already materialized, exact 155 development triplets are admitted.
Original embedded-reference defects remain in every report. The default mode
does not access either file of a header-reference-unresolved triplet. Explicit
publisher_same_entity_filename_contract mode instead associates the unique
frozen triplet by the provider's published subject/session/run naming rule:
https://www.nature.com/articles/s41597-025-06039-9 (Data Records).
This documentary association neither repairs source bytes nor verifies the
embedded references; a reference to another frozen identified run blocks it.
Binary files are stat'ed, never opened. Marker bytes are streamed to identify
New Segment records; all other marker fields remain opaque and are discarded.
Descriptions are never decoded, retained, reported, or used scientifically.
Segment metadata supplies partial discontinuity information, not gap-free,
clock synchronization, filtering-delay, calibration, or forecasting evidence.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import socket
import stat
import sys
import time

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "outputs/code"))
from inspect_development_acquisition import (  # noqa: E402
    PRIOR_HEADERS, PRIOR_VIEW, RUNS, VIEW_NAME, read_bounded, require, select_headers,
)
from inventory_source_members import DEVELOPMENT_PEOPLE, classify_member  # noqa: E402

MARKER_LIMIT = 2_000_000
LINE_LIMIT = 65_536
WIDTHS = {"IEEE_FLOAT_32": 4, "INT_16": 2}
ORIGINAL_MAPPING = "original_references"
PUBLISHER_MAPPING = "publisher_same_entity_filename_contract"
MAPPING_BASES = (ORIGINAL_MAPPING, PUBLISHER_MAPPING)
PUBLISHER_SOURCE = "https://www.nature.com/articles/s41597-025-06039-9"
MARKER_SIGNATURES = {
    b"Brain Vision Data Exchange Marker File, Version 1.0",
    b"Brain Vision Data Exchange Marker File Version 1.0",
    # MNE's maintained _check_bv_version explicitly accepts Brain ?Vision.
    # https://github.com/mne-tools/mne-python/blob/main/mne/io/brainvision/brainvision.py
    # Admit the observed spelling only; do not adopt its wider version support.
    b"BrainVision Data Exchange Marker File Version 1.0",
}


def admit_runs(prior, view, view_root, *, mapping_basis=ORIGINAL_MAPPING):
    """Bind every triplet/role before any development file is stat'ed/opened."""
    require(mapping_basis in MAPPING_BASES, "explicit_supported_mapping_basis_required")
    selected = select_headers(prior, view, view_root)
    rows = [row for row in view["files"] if row.get("source") == "self_paced"]
    require(len(rows) == 465, "exact_465_development_triplet_members_required")
    members = {row["member"]: row for row in rows}
    require(len(members) == 465, "duplicate_development_triplet_member")
    basenames = {suffix: tuple(PurePosixPath(member).name for member in members
                              if PurePosixPath(member).suffix == suffix)
                 for suffix in (".eeg", ".vmrk")}
    require(all(len(names) == len(set(names)) == 155 for names in basenames.values()),
            "unique_155_frozen_triplet_basenames_required")
    expected_members, admitted = set(), []
    for header_path, _, header in selected:
        header_member = PurePosixPath(header["member"])
        run = {"header": header, "header_member": str(header_member),
               "participant": header["participant"],
               "triplet_binding": {"status": "verified_original_references",
                                   "reference_verified": True, "original_references": {},
                                   "unresolved": []},
               "filename_mapping": {"basis": mapping_basis,
                                    "publisher_source": PUBLISHER_SOURCE if mapping_basis == PUBLISHER_MAPPING else None,
                                    "accepted": False, "competing_references": [],
                                    "synchronization_verified": False,
                                    "trace_identity_independently_verified": False},
               "known_data_files": basenames[".eeg"]}
        expected_members.add(str(header_member))
        for kind, suffix in (("data", ".eeg"), ("marker", ".vmrk")):
            filename = header.get(kind + "_file")
            require(isinstance(filename, str) and filename not in ("", ".", "..")
                    and "/" not in filename and "\\" not in filename,
                    "safe_basename_header_reference_required")
            binding = run["triplet_binding"]
            expected_name = header_member.with_suffix(suffix).name
            binding["original_references"][kind] = {
                "filename": filename, "member_present": header.get(kind + "_member_present"),
                "matches_manifest": filename == expected_name}
            if filename != expected_name:
                binding["unresolved"].append(kind + "_header_reference_differs_from_manifest")
                if filename in basenames[suffix]:
                    run["filename_mapping"]["competing_references"].append(filename)
            if header.get(kind + "_member_present") is not True:
                binding["unresolved"].append(kind + "_original_reference_not_present")
            member = str(header_member.with_suffix(suffix))
            expected_members.add(member)
            require(member in members, "missing_original_triplet_member")
            row = members[member]
            role = classify_member("self_paced", None, member)
            expected = header_path.with_suffix(suffix)
            require(row.get("role") == role.get("role") == "development"
                    and role.get("participant") == header["participant"]
                    and row.get("destination") == str(expected)
                    and type(row.get("bytes")) is int and row["bytes"] > 0,
                    "fixed_development_triplet_identity_and_size_required")
            run[kind + "_path"] = expected
            run[kind + "_bytes"] = row["bytes"]
        if run["triplet_binding"]["unresolved"]:
            run["triplet_binding"]["reference_verified"] = False
            run["triplet_binding"]["status"] = (
                "unresolved_no_source_access" if mapping_basis == ORIGINAL_MAPPING
                else "unresolved_original_references")
        mapping = run["filename_mapping"]
        if mapping["competing_references"]:
            mapping["status"] = "blocked_competing_frozen_run_reference"
        elif mapping_basis == PUBLISHER_MAPPING:
            mapping.update(accepted=True, status="accepted_publisher_filename_contract")
        elif run["triplet_binding"]["reference_verified"]:
            mapping.update(accepted=True, status="accepted_original_references")
        else:
            mapping["status"] = "unavailable_original_references"
        admitted.append(run)
    require(expected_members == set(members), "only_exact_original_triplets_allowed")
    return admitted


def binary_layout(header, size):
    """Integer divisibility only; unsupported declarations remain unresolved."""
    result = {"file_bytes": size, "data_format": header.get("data_format"),
              "data_orientation": header.get("data_orientation"),
              "binary_format": header.get("binary_format"),
              "channel_count": header.get("channel_count"),
              "bytes_per_value": None, "binary_prefix_bytes": None,
              "bytes_per_sample_frame": None, "remainder_bytes": None,
              "sample_count": None, "sampling_interval_microseconds":
              header.get("sampling_interval_microseconds"),
              "nominal_stored_duration_seconds": None, "unresolved": []}
    problems = result["unresolved"]
    if header.get("data_format") != "BINARY":
        problems.append("unsupported_or_missing_data_format")
    if header.get("data_orientation") != "MULTIPLEXED":
        problems.append("unsupported_or_missing_data_orientation")
    width = WIDTHS.get(header.get("binary_format"))
    if width is None:
        problems.append("unsupported_or_missing_binary_format")
    channels = header.get("channel_count")
    if type(channels) is not int or channels <= 0:
        problems.append("invalid_channel_count")
    if type(size) is not int or size <= 0:
        problems.append("invalid_binary_file_size")
    interval = header.get("sampling_interval_microseconds")
    interval_valid = (type(interval) in (int, float) and math.isfinite(interval)
                      and interval > 0)
    if not problems:
        count, remainder = divmod(size, width * channels)
        result.update(bytes_per_value=width, binary_prefix_bytes=0,
                      bytes_per_sample_frame=width * channels, remainder_bytes=remainder)
        if remainder or count < 1:
            problems.append("binary_bytes_not_positive_integral_sample_frames")
        else:
            result["sample_count"] = count
            if interval_valid:
                duration = count * interval / 1e6
                if math.isfinite(duration):
                    result["nominal_stored_duration_seconds"] = duration
                else:
                    problems.append("nonfinite_nominal_stored_duration")
    if not interval_valid:
        problems.append("invalid_sampling_interval")
    return result


def _checked_stat(path, exact_size):
    """Reject redirected/nonregular paths; never open binary signal files."""
    path = Path(path)
    require(path.resolve() == path.absolute(), "nonsymlink_development_path_required")
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_size == exact_size,
            "original_regular_development_file_size_required")
    return info.st_size


def inspect_binary(path, expected_size, header):
    return binary_layout(header, _checked_stat(path, expected_size))


def _integer(token):
    return int(token) if re.fullmatch(rb"[0-9]{1,20}", token) else None


def parse_new_segments(lines, expected_data_file, sample_count, sampling_interval, *,
                       mapping_basis=ORIGINAL_MAPPING, known_data_files=()):
    """Parse only safe structural tokens and New Segment positions/dates.

    Caller supplies bounded byte lines. Non-New-Segment values and descriptions
    are never decoded. Only the structural DataFile reference is retained along
    with New Segment metadata; no marker description enters reports or errors.
    """
    require(mapping_basis in MAPPING_BASES, "explicit_supported_mapping_basis_required")
    if mapping_basis == PUBLISHER_MAPPING:
        require(expected_data_file in known_data_files
                and len(known_data_files) == len(set(known_data_files)),
                "documentary_marker_mapping_requires_unique_frozen_identity")
    result = {"new_segments": [], "unresolved": [],
              "data_file_binding": {"reference_verified": False, "original_references": [],
                                    "reference_count": 0, "competing_references": [],
                                    "documentary_association_accepted": False,
                                    "status": "not_inspected"},
              "gap_information": "partial_new_segment_metadata_only",
              "absence_proves_gap_free": False, "timestamp_timezone": "unspecified"}
    issues = result["unresolved"]
    section, first, common_count, marker_count, data_files = None, True, 0, 0, []
    previous_position = None
    for raw in lines:
        require(isinstance(raw, bytes) and len(raw) <= LINE_LIMIT,
                "bounded_marker_byte_line_required")
        line = raw.strip()
        if first:
            first = False
            if line.startswith(b"\xef\xbb\xbf"):
                line = line[3:]
            if line not in MARKER_SIGNATURES:
                issues.append("unsupported_marker_signature")
                return result
            continue
        if not line or line.startswith(b";"):
            continue
        if line.startswith(b"[") and line.endswith(b"]"):
            section = line
            common_count += int(section == b"[Common Infos]")
            marker_count += int(section == b"[Marker Infos]")
            continue
        key, separator, value = line.partition(b"=")
        key = key.strip()
        if section == b"[Common Infos]" and key == b"DataFile" and separator:
            # Structural reference only, never an opened path or event field.
            token = value.strip()
            try:
                filename = token.decode("ascii")
            except UnicodeDecodeError:
                filename = None
            data_files.append(filename)
        elif section == b"[Marker Infos]" and separator:
            marker_type, comma, remainder = value.partition(b",")
            if marker_type.strip() != b"New Segment":
                continue
            entry = {"segment_ordinal": len(result["new_segments"]) + 1,
                     "position_one_based": None, "sample_offset_zero_based": None,
                     "nominal_offset_seconds": None, "timestamp": None,
                     "timestamp_status": "absent", "unresolved": []}
            problems = entry["unresolved"]
            if not re.fullmatch(rb"Mk[0-9]+", key) or not comma:
                problems.append("malformed_new_segment_record")
            # Discard the description without decoding or retaining it.
            _, delimiter, coordinates = remainder.partition(b",")
            fields = coordinates.split(b",") if delimiter else []
            if len(fields) not in (3, 4):
                problems.append("unsupported_new_segment_field_count")
            else:
                position = _integer(fields[0].strip())
                points, channel = _integer(fields[1].strip()), _integer(fields[2].strip())
                if position is None or position < 1:
                    problems.append("invalid_new_segment_position")
                else:
                    entry.update(position_one_based=position, sample_offset_zero_based=position - 1)
                    if sample_count is None:
                        problems.append("position_sample_bound_unverified")
                    elif position > sample_count:
                        problems.append("position_exceeds_sample_count")
                    if previous_position is not None and position <= previous_position:
                        problems.append("new_segment_positions_not_strictly_increasing")
                    previous_position = position
                    if (type(sampling_interval) in (int, float)
                            and math.isfinite(sampling_interval) and sampling_interval > 0):
                        offset = (position - 1) * sampling_interval / 1e6
                        if math.isfinite(offset):
                            entry["nominal_offset_seconds"] = offset
                    else:
                        problems.append("nominal_offset_unavailable")
                if points is None or points < 1 or channel != 0:
                    problems.append("unsupported_new_segment_extent_or_channel")
                timestamp = fields[3].strip() if len(fields) == 4 else b""
                if timestamp:
                    entry["timestamp_status"] = "invalid"
                    if re.fullmatch(rb"[0-9]{20}", timestamp):
                        try:
                            parsed = datetime.strptime(timestamp.decode("ascii"), "%Y%m%d%H%M%S%f")
                        except ValueError:
                            pass
                        else:
                            entry["timestamp"] = parsed.isoformat(timespec="microseconds")
                            entry["timestamp_status"] = "present_timezone_unspecified"
                    if entry["timestamp"] is None:
                        problems.append("invalid_new_segment_timestamp")
            result["new_segments"].append(entry)
    if first:
        issues.append("empty_marker_file")
    if common_count != 1 or marker_count != 1:
        issues.append("missing_or_duplicate_marker_sections")
    binding = result["data_file_binding"]
    binding.update(original_references=data_files, reference_count=len(data_files),
                   reference_verified=data_files == [expected_data_file])
    unsafe = any(not isinstance(name, str) or not name or "/" in name or "\\" in name
                 or name in (".", "..") for name in data_files)
    binding["competing_references"] = sorted({name for name in data_files
                                               if name in known_data_files and name != expected_data_file})
    if not binding["reference_verified"]:
        issues.append("marker_data_file_binding_unverified")
    if not data_files:
        issues.append("marker_data_file_reference_missing")
    elif len(data_files) > 1:
        issues.append("marker_data_file_reference_duplicated")
    if unsafe:
        issues.append("unsafe_or_unsupported_marker_data_file_reference")
    if binding["competing_references"]:
        issues.append("marker_data_file_references_competing_frozen_run")
        binding["status"] = "blocked_competing_frozen_run_reference"
    elif unsafe:
        binding["status"] = "blocked_unsafe_reference"
    elif mapping_basis == PUBLISHER_MAPPING:
        binding.update(documentary_association_accepted=True,
                       status="accepted_publisher_filename_contract")
    else:
        binding["status"] = ("verified_original_reference" if binding["reference_verified"]
                             else "unresolved_original_reference")
    if not result["new_segments"]:
        issues.append("no_new_segment_metadata_available")
    if any(entry["unresolved"] for entry in result["new_segments"]):
        issues.append("new_segment_records_have_unresolved_metadata")
    return result


def inspect_markers(path, expected_size, header, sample_count, *,
                    mapping_basis=ORIGINAL_MAPPING, known_data_files=()):
    _checked_stat(path, expected_size)
    if expected_size > MARKER_LIMIT:
        return {"new_segments": [], "unresolved": ["marker_file_exceeds_metadata_read_bound"],
                "gap_information": "partial_new_segment_metadata_only",
                "absence_proves_gap_free": False, "timestamp_timezone": "unspecified"}
    with Path(path).open("rb") as source:
        consumed = 0
        def lines():
            nonlocal consumed
            while True:
                line = source.readline(LINE_LIMIT + 1)
                if not line:
                    break
                consumed += len(line)
                require(consumed <= expected_size and len(line) <= LINE_LIMIT,
                        "marker_file_changed_or_line_bound_exceeded")
                yield line
        # The expected data identity comes from the already admitted physical
        # filename. A stale header DataFile is preserved, never substituted here.
        expected_data_file = Path(path).with_suffix(".eeg").name
        result = parse_new_segments(lines(), expected_data_file, sample_count,
                                    header.get("sampling_interval_microseconds"),
                                    mapping_basis=mapping_basis, known_data_files=known_data_files)
        # Unsupported signatures stop early and remain unresolved; do not read
        # the rest merely to complete a byte count.
        if "unsupported_marker_signature" not in result["unresolved"]:
            require(consumed == expected_size, "marker_file_size_changed_during_read")
        _checked_stat(path, expected_size)
    return result


def inspect_run(item):
    """Inspect only a selected, unambiguous admitted file association."""
    row = {"header_member": item["header_member"], "participant": item["participant"],
           "triplet_binding": item["triplet_binding"],
           "filename_mapping": item["filename_mapping"], "source_files_checked": False,
           "metadata_file_association_accepted": False}
    if not item["filename_mapping"]["accepted"]:
        layout = binary_layout(item["header"], None)
        layout["unresolved"] = [reason for reason in layout["unresolved"]
                                if reason != "invalid_binary_file_size"]
        layout["unresolved"].append("triplet_binding_unresolved_binary_size_not_checked")
        segments = {"new_segments": None,
                    "unresolved": ["triplet_binding_unresolved_marker_not_read"],
                    "gap_information": "partial_new_segment_metadata_only",
                    "absence_proves_gap_free": False, "timestamp_timezone": "unspecified"}
    else:
        layout = inspect_binary(item["data_path"], item["data_bytes"], item["header"])
        segments = inspect_markers(item["marker_path"], item["marker_bytes"],
                                   item["header"], layout["sample_count"],
                                   mapping_basis=item["filename_mapping"]["basis"],
                                   known_data_files=item["known_data_files"])
        row["source_files_checked"] = True
        marker_binding = segments.get("data_file_binding", {})
        row["metadata_file_association_accepted"] = (
            marker_binding.get("documentary_association_accepted", False)
            if item["filename_mapping"]["basis"] == PUBLISHER_MAPPING
            else marker_binding.get("reference_verified", False))
    row.update(binary_layout=layout, segment_metadata=segments)
    return row


def run(args):
    job, host = os.environ.get("SLURM_JOB_ID", ""), socket.gethostname()
    require(job.isdigit() and "-ln" not in host and os.environ.get("SLURM_CPUS_PER_TASK") == "1",
            "coordinator_owned_one_CPU_compute_allocation_required")
    require(Path.cwd().resolve() == BASE, "exact_episode_working_directory_required")
    require(type(args.max_seconds) is int and 0 < args.max_seconds <= 60, "at_most_60_seconds_required")
    output = Path(args.output_dir).absolute()
    require(output.parent == RUNS and output.resolve() == output and not output.exists(),
            "fresh_direct_child_source_views_and_timing_output_required")
    started = time.monotonic()
    previous = signal.getsignal(signal.SIGALRM)
    def deadline(signum, frame):
        raise TimeoutError("bounded_stream_metadata_deadline")
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    record, error = None, None
    try:
        prior = json.loads(read_bounded(PRIOR_HEADERS, 4_000_000))
        view = json.loads(read_bounded(PRIOR_VIEW, 1_000_000))
        admitted = admit_runs(prior, view, Path(os.environ["SCRATCH"]) / VIEW_NAME,
                              mapping_basis=args.mapping_basis)
        output.mkdir()
        record = {"analysis_id": "source_views_and_timing",
                  "status": "development_stream_layout_started",
                  "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                  "job_id": job, "host": host,
                  "source_header_report": str(PRIOR_HEADERS.relative_to(BASE)),
                  "source_development_view": str(PRIOR_VIEW.relative_to(BASE)),
                  "mapping_basis": args.mapping_basis,
                  "publisher_naming_source": PUBLISHER_SOURCE if args.mapping_basis == PUBLISHER_MAPPING else None,
                  "embedded_reference_repair_performed": False,
                  "source_files_rewritten": False,
                  "physical_synchronization_verified": False,
                  "trace_identity_independently_verified": False,
                  "expected_run_count": 155, "development_people": sorted(DEVELOPMENT_PEOPLE),
                  "runs": [], "binary_signal_file_opened": False,
                  "marker_bytes_scanned_for_new_segments": True,
                  "non_segment_marker_descriptions_decoded_or_retained": False,
                  "stimulus_labels_used": False, "way_source_access": False,
                  "held_out_source_access": False, "archive_access": False,
                  "gap_information": "partial_new_segment_metadata_only",
                  "absence_proves_gap_free": False,
                  "physical_acceleration_calibration_verified": False,
                  "filter_transfer_or_delay_verified": False, "timing_bound_verified": False,
                  "forecast_scoring_ready": False}
        for item in admitted:
            record["runs"].append(inspect_run(item))
        record["status"] = "complete_development_stream_layout_metadata"
    except Exception as exc:
        error = exc
        if record is not None:
            # No exception text: malformed source fields must never reach logs.
            record.update(status="partial_development_stream_layout_metadata",
                          error_type=type(exc).__name__)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if record is not None:
            record["elapsed_seconds"] = time.monotonic() - started
            record["runs_completed"] = len(record["runs"])
            record["runs_with_both_source_files_checked"] = sum(
                row["source_files_checked"] for row in record["runs"])
            record["source_files_checked_in_completed_rows"] = (
                2 * record["runs_with_both_source_files_checked"])
            record["runs_skipped_for_unresolved_triplet_binding"] = sum(
                bool(row["triplet_binding"]["unresolved"]) and not row["source_files_checked"]
                for row in record["runs"])
            record["runs_with_unresolved_embedded_header_references"] = sum(
                not row["triplet_binding"]["reference_verified"] for row in record["runs"])
            record["runs_with_accepted_metadata_file_association"] = sum(
                row["metadata_file_association_accepted"] for row in record["runs"])
            completed = {row["header_member"] for row in record["runs"]}
            record["remaining_members"] = [item["header_member"] for item in admitted
                                           if item["header_member"] not in completed]
            record["runs_with_unresolved_metadata"] = sum(
                bool(row["triplet_binding"]["unresolved"]
                     or row["binary_layout"]["unresolved"] or row["segment_metadata"]["unresolved"])
                for row in record["runs"])
            record["sample_count_counts"] = dict(Counter(
                str(row["binary_layout"]["sample_count"]) for row in record["runs"]))
            with (output / "development_stream_layout.json").open("x") as target:
                json.dump(record, target, indent=2, allow_nan=False)
                target.write("\n")
    summary = {"status": record["status"] if record else "development_stream_layout_refused",
               "runs_completed": len(record["runs"]) if record else 0,
               "error_type": type(error).__name__ if error else None}
    print(json.dumps(summary), flush=True)
    return 0 if error is None else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seconds", type=int, default=60)
    parser.add_argument("--mapping-basis", choices=MAPPING_BASES, default=ORIGINAL_MAPPING)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
