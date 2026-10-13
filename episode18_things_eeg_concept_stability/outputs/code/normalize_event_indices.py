#!/usr/bin/env python3
"""Repair the observed one-based TSV sample convention in an EP18 working join.

Reuse all passing structural checks. Verify the release-wide marker diagnostic,
then preserve source sample/onset and add explicit zero-based recording indices.
Only event text and the previously qualified filename join are read.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import socket
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-report", type=Path, required=True)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--working-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use the coordinator's verified compute allocation")
    previous = json.loads(args.qualification_report.read_text())
    expected_failure = "event_samples_match_one_based_markers"
    diagnostics = {}
    for result in previous["participant_results"]:
        if "structural_error" in result:
            raise ValueError("Unrepaired structural error in earlier qualification")
        failures = {key for key, passed in result["checks"].items() if not passed}
        count = result["main_events"]+result["repeated_events"]
        matches = result["marker_match_counts_by_added_sample_offset"]
        if failures != {expected_failure} or matches != {"0": count, "1": 0}:
            raise ValueError("Earlier report does not support the same unique sample-origin repair")
        diagnostics[result["participant"]] = result
    if len(diagnostics) != 46 or not previous["common_sensor_inventory_has_62"]:
        raise ValueError("Cohort or common-channel contract is not qualified")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.working_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    output_path = args.working_dir/"event_image_join_normalized.tsv.gz"
    counts, clock_spans = {}, {}
    active_participant, event_source, event_reader = None, None, None
    try:
        with gzip.open(previous["event_image_join"], "rt", newline="", encoding="utf-8") as source:
            joined = csv.DictReader(source, delimiter="\t")
            fields = list(joined.fieldnames)+["sample_zero_based", "source_onset_seconds",
                                             "onset_recording_seconds", "source_time_stimon_seconds"]
            with gzip.open(output_path, "wt", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                for row in joined:
                    participant = row["participant"]
                    if participant != active_participant:
                        if event_source:
                            if next(event_reader, None) is not None:
                                raise ValueError("The earlier join omitted source events")
                            event_source.close()
                        active_participant = participant
                        event_path = args.bids_root/participant/"eeg"/f"{participant}_task-rsvp_events.tsv"
                        event_source = event_path.open(newline="", encoding="utf-8-sig")
                        event_reader = csv.DictReader(event_source, delimiter="\t")
                        counts[participant] = 0
                    event = next(event_reader)
                    sample = int(row["sample"])
                    if float(event["sample"]) != sample:
                        raise ValueError("Source event ordering differs from the existing join")
                    sampling_hz = diagnostics[participant]["sampling_hz"]
                    if round(float(event["onset"])*sampling_hz) != sample:
                        raise ValueError("Source onset no longer agrees with the observed sample convention")
                    zero = sample-1
                    if not 0 <= zero < diagnostics[participant]["recording_samples"]:
                        raise ValueError("Normalized sample is outside recording support")
                    row.update({"sample_zero_based": zero,
                                "source_onset_seconds": event["onset"],
                                "onset_recording_seconds": format(zero/sampling_hz, ".9f"),
                                "source_time_stimon_seconds": event["time_stimon"]})
                    writer.writerow(row)
                    counts[participant] += 1
                if event_reader is not None and next(event_reader, None) is not None:
                    raise ValueError("The earlier join omitted final source events")
    finally:
        if event_source:
            event_source.close()
    for participant, result in diagnostics.items():
        if counts.get(participant) != result["main_events"]+result["repeated_events"]:
            raise ValueError("Normalized join count differs from qualified source")
        clock_range = result["onset_minus_time_stimon_seconds_range"]
        clock_spans[participant] = clock_range[1]-clock_range[0]
    report = {"analysis_id": "input_role_preparation", "phase": "explicit_sample_origin_normalization",
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "observed_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic()-started,
              "previous_qualification_report": str(args.qualification_report),
              "status": "source_structure_and_sample_origin_reconciled",
              "source_sample_origin": "one_based_BrainVision_marker_position",
              "working_sample_rule": "sample_zero_based = source sample - 1",
              "working_onset_rule": "onset_recording_seconds = sample_zero_based / source sampling_hz",
              "source_sample_and_onset_retained": True,
              "normalized_join": str(output_path), "participant_event_counts": counts,
              "participants": len(counts), "events": sum(counts.values()),
              "common_recorded_channel_order": previous["common_recorded_channel_order"],
              "display_vs_recording_clock_offset_span_seconds": clock_spans,
              "display_clock_tolerance_adjudicated": False,
              "eeg_or_image_values_opened": False, "roles_assigned": False,
              "audit_seal_verified": False, "join_frozen": False,
              "marker_position_reference": "https://pressrelease.brainproducts.com/markers/"}
    (args.output_dir/"sample_origin_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
