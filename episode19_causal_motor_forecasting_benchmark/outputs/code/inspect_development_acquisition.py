#!/usr/bin/env python3
"""Extract omitted acquisition settings from the 155 existing development headers.

Reads only the fixed previous reports and their exact materialized .vhdr paths.
Never opens an archive, EEG array, marker, event, or held-out header. Recorded
cutoffs are configuration evidence, not a filter transfer function or delay bound.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import socket
import time

from inspect_development_headers import parse_header
from inventory_source_members import DEVELOPMENT_PEOPLE, classify_member


BASE = Path(__file__).resolve().parents[2]
RUNS = BASE / "outputs/runs/source_views_and_timing"
PRIOR_HEADERS = RUNS / "job_46425841_headers/development_headers.json"
PRIOR_VIEW = RUNS / "job_46425841_development_view/development_view.json"
VIEW_NAME = "br_autoresearch/episode19_causal_motor_forecasting_benchmark/development_view_job_46425841"
HEADER_LIMIT = 100_000  # Same text-only limit as the completed header reader.
TABLE_COLUMNS = "# Name Phys. Chn. Resolution / Unit Low Cutoff [s] High Cutoff [Hz] Notch [Hz] Gradient Offset"
CHANNEL_ROW = re.compile(
    r"^(\d+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)"
    r"(?:\s+(\S+)\s+(\S+))?$"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_acquisition(text, header):
    """Keep explicit table tokens; absent/unsupported settings remain unknown."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "[Comment]"]
    result = {"recording_software": None, "amplifier_model": None,
              "software_filters": None, "channel_settings": [], "unresolved": []}
    if len(starts) != 1:
        result["unresolved"].append("missing_or_duplicate_comment_section")
        return result
    comments = []
    for i in range(starts[0] + 1, len(lines)):
        if re.fullmatch(r"\[[^]]+\]", lines[i].strip()):
            break
        comments.append((i + 1, lines[i].strip()))
    for index, (line_number, line) in enumerate(comments):
        if line.startswith("BrainVision Recorder "):
            result["recording_software"] = {"text": line, "line": line_number}
        if line == "Amplifier":
            following = next(((n, value) for n, value in comments[index + 1:] if value), None)
            if following:
                result["amplifier_model"] = {"text": following[1], "line": following[0]}
        if "".join(line.split()).lower() == "softwarefilters":
            following = next(((n, value) for n, value in comments[index + 1:]
                              if value and not set(value) <= {"=", "-"}), None)
            if following and following[1] == "Disabled":
                result["software_filters"] = {"text": "Disabled", "line": following[0]}
            else:
                result["unresolved"].append("software_filter_configuration_not_supported")
    tables = [i for i, (_, line) in enumerate(comments) if " ".join(line.split()) == TABLE_COLUMNS]
    if len(tables) == 1:
        rows = []
        for line_number, line in comments[tables[0] + 1:]:
            if not line:
                break
            match = CHANNEL_ROW.fullmatch(line)
            if not match:
                result["unresolved"].append("unsupported_channel_table_row")
                rows = []
                break
            index, name, physical, resolution, unit, low, high, notch, gradient, offset = match.groups()
            rows.append({"index": int(index), "name": name, "physical_channel": int(physical),
                         "resolution": resolution, "unit": unit,
                         "low_cutoff_seconds_token": low, "high_cutoff_hz_token": high,
                         "notch_hz_token": notch, "gradient_token": gradient,
                         "offset_token": offset, "line": line_number})
        expected = header["channels"]
        matched = len(rows) == header["channel_count"] and all(
            row["index"] == index and all(row[key] == old[key] for key in ("name", "resolution", "unit"))
            for index, (row, old) in enumerate(zip(rows, expected), 1)
        )
        if matched:
            result["channel_settings"] = rows
        else:
            result["unresolved"].append("channel_table_incomplete_or_differs_from_channel_infos")
    else:
        result["unresolved"].append("missing_or_unsupported_channel_table_columns")
    for key in ("recording_software", "amplifier_model", "software_filters"):
        if result[key] is None:
            result["unresolved"].append(key + "_unknown")
    return result


def select_headers(prior, view, view_root):
    """Admit all 155 exact development header paths before any header is read."""
    require(prior.get("status") == "development_header_inspection_complete"
            and prior.get("header_count") == 155 and len(prior.get("headers", [])) == 155,
            "exact_completed_155_header_report_required")
    require(view.get("status") == "development_view_materialized", "completed_development_view_required")
    headers = {row["member"]: row for row in prior["headers"]}
    require(len(headers) == 155, "duplicate_prior_header_member")
    selected = [row for row in view["files"] if row.get("source") == "self_paced"
                and PurePosixPath(row["member"]).suffix.lower() == ".vhdr"]
    require(len(selected) == 155 and {row["member"] for row in selected} == set(headers),
            "exact_same_155_materialized_header_members_required")
    admitted, people = [], set()
    for row in sorted(selected, key=lambda item: item["member"]):
        member = row["member"]
        role = classify_member("self_paced", None, member)
        prior_row = headers[member]
        require(role.get("role") == row.get("role") == "development"
                and role.get("participant") == prior_row.get("participant"),
                "only_original_development_participant_headers_allowed")
        person = role["participant"]
        require(re.fullmatch(rf"{person}/ses-\d{{2}}/eeg/{person}_ses-\d{{2}}_task-reachingandgrasping_run-\d{{4}}_eeg\.vhdr", member),
                "exact_raw_development_header_name_required")
        expected = Path(view_root) / "self_paced/development" / member
        require(row["destination"] == str(expected) and type(row["bytes"]) is int
                and 0 < row["bytes"] <= HEADER_LIMIT, "fixed_materialized_header_path_and_size_required")
        people.add(person)
        admitted.append((expected, row["bytes"], prior_row))
    require(people == DEVELOPMENT_PEOPLE, "all_15_original_development_people_required")
    return admitted


def read_bounded(path, limit, exact_size=None):
    path = Path(path)
    require(path.resolve() == path.absolute() and path.is_file(), "regular_nonsymlink_metadata_file_required")
    size = path.stat().st_size
    require(0 < size <= limit and (exact_size is None or size == exact_size), "metadata_file_size_mismatch")
    with path.open("rb") as source:
        value = source.read(limit + 1)
    require(len(value) == size, "metadata_file_changed_or_exceeded_bound")
    return value.decode("utf-8-sig")


def run(args):
    job, host = os.environ.get("SLURM_JOB_ID", ""), socket.gethostname()
    require(job.isdigit() and "-ln" not in host and os.environ.get("SLURM_CPUS_PER_TASK") == "1",
            "coordinator_owned_one_CPU_compute_allocation_required")
    require(Path.cwd().resolve() == BASE, "exact_episode_working_directory_required")
    require(type(args.max_seconds) is int and 0 < args.max_seconds <= 60, "at_most_60_seconds_required")
    output = Path(args.output_dir).absolute()
    require(output.parent == RUNS and output.resolve() == output and not output.exists(),
            "fresh_direct_child_source_views_and_timing_output_required")
    view_root = Path(os.environ["SCRATCH"]) / VIEW_NAME
    prior = json.loads(read_bounded(PRIOR_HEADERS, 4_000_000))
    view = json.loads(read_bounded(PRIOR_VIEW, 1_000_000))
    admitted = select_headers(prior, view, view_root)
    output.mkdir()
    record = {"analysis_id": "source_views_and_timing", "status": "metadata_extraction_started",
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "job_id": job, "host": host,
              "source_header_report": str(PRIOR_HEADERS.relative_to(BASE)),
              "source_development_view": str(PRIOR_VIEW.relative_to(BASE)),
              "expected_header_count": 155, "headers": [],
              "signal_array_access": False, "marker_or_event_access": False,
              "held_out_header_access": False, "archive_access": False,
              "physical_acceleration_calibration_verified": False,
              "filter_transfer_or_delay_verified": False, "timing_bound_verified": False,
              "forecast_scoring_ready": False}
    started, error = time.monotonic(), None
    previous = signal.getsignal(signal.SIGALRM)
    def deadline(signum, frame):
        raise TimeoutError("bounded_header_extraction_deadline")
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    try:
        for path, size, old in admitted:
            text = read_bounded(path, HEADER_LIMIT, size)
            current = parse_header(text)
            require(all(current[key] == old[key] for key in current), "header_differs_from_original_report")
            record["headers"].append({"member": old["member"], "participant": old["participant"],
                "sample_rate_hz": current["sample_rate_hz"], "channel_count": current["channel_count"],
                "acquisition": parse_acquisition(text, current)})
        record["status"] = "complete_development_acquisition_metadata_extraction"
    except Exception as exc:
        error = exc
        record.update(status="partial_development_acquisition_metadata_extraction",
                      error_type=type(exc).__name__, error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        record["elapsed_seconds"] = time.monotonic() - started
        record["headers_completed"] = len(record["headers"])
        completed = {row["member"] for row in record["headers"]}
        record["remaining_members"] = [old["member"] for _, _, old in admitted if old["member"] not in completed]
        record["headers_with_unresolved_settings"] = sum(bool(row["acquisition"]["unresolved"])
                                                        for row in record["headers"])
        record["sampling_rate_counts"] = dict(Counter(str(row["sample_rate_hz"]) for row in record["headers"]))
        with (output / "acquisition_metadata.json").open("x") as target:
            json.dump(record, target, indent=2)
            target.write("\n")
    print(json.dumps({key: value for key, value in record.items() if key not in {"headers", "remaining_members"}}), flush=True)
    return 0 if error is None else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seconds", type=int, default=60)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
