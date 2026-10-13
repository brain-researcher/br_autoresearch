#!/usr/bin/env python3
"""Explain observed EP18 clock discontinuities using event and marker metadata.

All events remain present. A >100-ms offset change is a descriptive breakpoint
for this 100-ms RSVP stream, not an exclusion or eligibility threshold. Report
all such changes, every BrainVision segment and every physical-sequence span.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket


def diagnose(root, participant):
    prefix = root/participant/"eeg"/f"{participant}_task-rsvp"
    segments = []
    with Path(str(prefix)+"_eeg.vmrk").open(encoding="utf-8-sig") as source:
        for line in source:
            if not re.match(r"^Mk[0-9]+=", line):
                continue
            key, body = line.strip().split("=", 1)
            fields = body.split(",")
            if fields[0] == "New Segment":
                stamp = datetime.strptime(fields[5], "%Y%m%d%H%M%S%f") if len(fields) > 5 and fields[5] else None
                segments.append({"marker_id": key, "sample_zero_based": int(fields[2])-1,
                                 "timestamp": stamp})
    if not segments:
        raise ValueError("No BrainVision recording segment marker")
    segments.sort(key=lambda item: item["sample_zero_based"])
    positions = [segment["sample_zero_based"] for segment in segments]
    source_fields = ["sample", "onset", "time_stimon", "isteststim", "blocksequencenumber",
                     "sequencenumber", "presentationnumber"]
    events = []
    with Path(str(prefix)+"_events.tsv").open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            item = {key: float(row[key]) for key in source_fields}
            sample = int(item["sample"])-1
            item["sample_zero_based"] = sample
            item["recording_segment"] = bisect_right(positions, sample)-1
            item["recording_minus_display_seconds"] = sample/1000-item["time_stimon"]
            events.append(item)
    groups = defaultdict(list)
    discontinuities, largest = [], []
    offset_segment = 0
    for index, event in enumerate(events):
        if index:
            previous = events[index-1]
            delta = event["recording_minus_display_seconds"]-previous["recording_minus_display_seconds"]
            change = {"previous_event_index": index-1, "event_index": index,
                      "offset_change_seconds": delta,
                      "recording_interval_seconds": (event["sample_zero_based"]-previous["sample_zero_based"])/1000,
                      "display_interval_seconds": event["time_stimon"]-previous["time_stimon"],
                      "previous_location": {key: previous[key] for key in ["isteststim", "blocksequencenumber", "sequencenumber", "presentationnumber", "recording_segment"]},
                      "current_location": {key: event[key] for key in ["isteststim", "blocksequencenumber", "sequencenumber", "presentationnumber", "recording_segment"]},
                      "physical_sequence_changes": event["sequencenumber"] != previous["sequencenumber"] or event["isteststim"] != previous["isteststim"],
                      "recording_segment_changes": event["recording_segment"] != previous["recording_segment"]}
            largest.append(change)
            if abs(delta) > .1 or change["recording_segment_changes"]:
                discontinuities.append(change)
                offset_segment += 1
        event["descriptive_offset_segment"] = offset_segment
        groups[(int(event["isteststim"]), int(event["blocksequencenumber"]),
                int(event["sequencenumber"]), event["recording_segment"], offset_segment)].append(event)
    sequence_reports = []
    for key, rows in groups.items():
        offsets = [row["recording_minus_display_seconds"] for row in rows]
        sequence_reports.append({"is_repeated": key[0], "block": key[1], "physical_sequence": key[2],
                                 "recording_segment": key[3], "descriptive_offset_segment": key[4],
                                 "events": len(rows), "offset_range_seconds": [min(offsets), max(offsets)],
                                 "offset_span_seconds": max(offsets)-min(offsets),
                                 "first_sample_zero_based": rows[0]["sample_zero_based"],
                                 "last_sample_zero_based": rows[-1]["sample_zero_based"]})
    segment_reports = []
    first_stamp = segments[0]["timestamp"]
    for index, segment in enumerate(segments):
        elapsed = (segment["timestamp"]-first_stamp).total_seconds() if first_stamp and segment["timestamp"] else None
        record = {"recording_segment": index, "marker_id": segment["marker_id"],
                  "sample_zero_based": segment["sample_zero_based"],
                  "timestamp_elapsed_seconds_from_first_segment": elapsed}
        if index and elapsed is not None and segments[index-1]["timestamp"]:
            record["timestamp_interval_seconds"] = (segment["timestamp"]-segments[index-1]["timestamp"]).total_seconds()
            record["recorded_sample_interval_seconds"] = (segment["sample_zero_based"]-segments[index-1]["sample_zero_based"])/1000
            record["timestamp_minus_recorded_interval_seconds"] = record["timestamp_interval_seconds"]-record["recorded_sample_interval_seconds"]
        segment_reports.append(record)
    offsets = [event["recording_minus_display_seconds"] for event in events]
    return {"participant": participant, "events": len(events),
            "global_offset_span_seconds": max(offsets)-min(offsets),
            "recording_segments": segment_reports,
            "discontinuities": discontinuities,
            "ten_largest_adjacent_offset_changes": sorted(largest, key=lambda row: -abs(row["offset_change_seconds"]))[:10],
            "per_sequence_and_segment": sequence_reports,
            "maximum_within_sequence_segment_offset_span_seconds": max(row["offset_span_seconds"] for row in sequence_reports),
            "all_large_discontinuities_at_physical_sequence_boundaries": all(row["physical_sequence_changes"] for row in discontinuities),
            "every_large_discontinuity_matches_recording_segment": all(row["recording_segment_changes"] for row in discontinuities)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--participants", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use the coordinator's verified compute allocation")
    if any(not re.fullmatch(r"sub-[0-9]{2}", participant) for participant in args.participants):
        raise ValueError("Invalid explicit participant ID")
    results = [diagnose(args.bids_root, participant) for participant in args.participants]
    report = {"analysis_id": "input_role_preparation", "phase": "recording_display_clock_diagnosis",
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "observed_utc": datetime.now(timezone.utc).isoformat(), "participant_diagnostics": results,
              "all_source_events_retained": True, "neural_or_image_values_opened": False,
              "participant_eligibility_changed": False, "timing_verified": False}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"clock_segment_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"participant_summaries": [{key: value for key, value in result.items()
                                                if key not in {"per_sequence_and_segment", "ten_largest_adjacent_offset_changes"}}
                                               for result in results]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
