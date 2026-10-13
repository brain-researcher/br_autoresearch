#!/usr/bin/env python3
"""EP18 outcome-blind event/header/marker/image-filename qualification.

Never opens EEG data or image members. Reads text event metadata, BrainVision
headers/markers and the ZIP central directory; only stats the EEG binary.
No participant split, EEG quality decision or candidate comparison is made.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import re
import socket
import time
import zipfile


EXCLUDED = {"sub-01", "sub-06", "sub-18", "sub-23"}
FIELDS = ["onset", "duration", "blocksequencenumber", "isteststim", "objectnumber",
          "presentationnumber", "sequencenumber", "stim", "stimnumber", "teststimnumber",
          "withinsequencenumber", "stimname", "time_stimon", "sample"]


def integer(value):
    result = float(value)
    if not math.isfinite(result) or not result.is_integer():
        raise ValueError("Expected a finite integer metadata field")
    return int(result)


def ini(path):
    sections, section = {}, None
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                sections[section] = {}
            elif section and "=" in line:
                key, value = line.split("=", 1)
                sections[section][key] = value
    return sections


def image_key(path):
    pieces = path.replace("\\", "/").split("/")
    if len(pieces) < 2:
        return None
    return "/".join(pieces[-2:])


def qualify_participant(root, participant, metadata_images, archive_images, join):
    prefix = root/participant/"eeg"/f"{participant}_task-rsvp"
    header_path = Path(str(prefix)+"_eeg.vhdr")
    header = ini(header_path)
    common = header["Common Infos"]
    channels = [value.split(",", 1)[0].replace("\\1", ",")
                for key, value in sorted(header["Channel Infos"].items(), key=lambda item: int(item[0][2:]))]
    n_channels = integer(common["NumberOfChannels"])
    sampling_hz = 1e6/float(common["SamplingInterval"])
    data_path = header_path.parent/common["DataFile"]
    marker_path = header_path.parent/common["MarkerFile"]
    if data_path.parent != header_path.parent or marker_path.parent != header_path.parent:
        raise ValueError("Unexpected source path in BrainVision header")
    if header["Binary Infos"]["BinaryFormat"] != "IEEE_FLOAT_32" or common["DataOrientation"] != "MULTIPLEXED":
        raise ValueError("Recording-length stat requires documented float32 multiplexed format")
    data_bytes = data_path.stat().st_size
    sample_count, leftover = divmod(data_bytes, n_channels*4)
    # Brain Products specifies one-based marker positions. Keep explicit
    # diagnostics for both offsets; do not silently repair a source convention.
    marker_onsets = []
    with marker_path.open(encoding="utf-8-sig") as source:
        for line in source:
            if re.match(r"^Mk[0-9]+=", line):
                parts = line.split("=", 1)[1].strip().split(",")
                if parts[0] == "Event" and re.fullmatch(r"E\s*1", parts[1]):
                    marker_onsets.append(integer(parts[2]))
    marker_set = set(marker_onsets)
    with Path(str(prefix)+"_events.tsv").open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source, delimiter="\t")
        missing = set(FIELDS)-set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing structural event fields: {sorted(missing)}")
        events = [{field: row[field] for field in FIELDS} for row in reader]
    main, repeated, joins_failed = [], [], []
    for event in events:
        for key in ["blocksequencenumber", "isteststim", "objectnumber", "presentationnumber",
                    "sequencenumber", "stimnumber", "teststimnumber", "withinsequencenumber", "sample"]:
            # Main and repeat-specific identifiers may be absent outside their own session.
            if event[key] not in ("n/a", "nan", "", "None"):
                event[key] = integer(event[key])
        (repeated if event["isteststim"] else main).append(event)
        key = image_key(event["stim"])
        if key not in metadata_images or len(archive_images.get(key, [])) != 1:
            joins_failed.append({"sample": event["sample"], "stim": event["stim"], "metadata_key": key,
                                 "metadata_match": key in metadata_images,
                                 "archive_matches": len(archive_images.get(key, []))})
        else:
            join.writerow([participant, event["isteststim"], event["blocksequencenumber"],
                           event["sequencenumber"], event["presentationnumber"], event["sample"],
                           event["objectnumber"], event["stimnumber"], key, archive_images[key][0],
                           "things_metadata_osf_jum2f_20260922", "0.05"])
        event["image_key"] = key
    blocks, sequences, concepts = defaultdict(list), defaultdict(list), defaultdict(list)
    for event in main:
        blocks[event["blocksequencenumber"]].append(event)
        sequences[event["sequencenumber"]].append(event)
        concepts[event["objectnumber"]].append(event)
    repeated_images = Counter(event["image_key"] for event in repeated)
    samples = [event["sample"] for event in events]
    duration_values = sorted({event["duration"] for event in events})
    onset_error = max(abs(float(event["onset"])*sampling_hz-event["sample"]) for event in events)
    clock_offsets = [float(event["onset"])-float(event["time_stimon"]) for event in events]
    checks = {
        "header_1000_hz": sampling_hz == 1000,
        "channel_count_and_labels": len(channels) == n_channels and len(set(channels)) == n_channels,
        "binary_length_has_whole_samples": leftover == 0,
        "all_event_samples_in_recording": all(0 <= sample < sample_count for sample in samples),
        "event_samples_strictly_increasing": all(a < b for a, b in zip(samples[:-1], samples[1:])),
        "main_event_count_22248": len(main) == 22248,
        "main_blocks_0_to_11": set(blocks) == set(range(12)),
        "each_block_has_each_concept_once": all(len(rows) == 1854 and len({row["objectnumber"] for row in rows}) == 1854 for rows in blocks.values()),
        "1854_concepts_with_12_distinct_exemplars": len(concepts) == 1854 and all(len(rows) == 12 and len({row["stimnumber"] for row in rows}) == 12 and len({row["image_key"] for row in rows}) == 12 for rows in concepts.values()),
        "main_sequences_0_to_71": set(sequences) == set(range(72)),
        "six_sequences_per_block": all(len({row["sequencenumber"] for row in rows}) == 6 for rows in blocks.values()),
        "each_sequence_309_positions": all(len(rows) == 309 and {row["presentationnumber"] for row in rows} == set(range(309)) and len({row["blocksequencenumber"] for row in rows}) == 1 for rows in sequences.values()),
        "within_block_sequence_positions_0_to_5": all({row["withinsequencenumber"] for row in rows} == set(range(6)) for rows in blocks.values()),
        "repeated_200_images_12_times": len(repeated) == 2400 and len(repeated_images) == 200 and set(repeated_images.values()) == {12},
        "source_duration_is_50_ms": all(float(value) == 50 for value in duration_values),
        "exact_image_filename_join": not joins_failed,
        "event_samples_match_one_based_markers": all(sample+1 in marker_set for sample in samples),
    }
    return {"participant": participant, "checks": checks, "all_checks_pass": all(checks.values()),
            "channels": channels, "n_channels": n_channels, "sampling_hz": sampling_hz,
            "eeg_bytes_statted_not_opened": data_bytes, "recording_samples": sample_count,
            "main_events": len(main), "repeated_events": len(repeated),
            "main_blocks": len(blocks), "main_sequences": len(sequences),
            "onset_minus_sample_max_absolute_samples": onset_error,
            "onset_minus_time_stimon_seconds_range": [min(clock_offsets), max(clock_offsets)],
            "image_onset_markers": len(marker_onsets),
            "marker_match_counts_by_added_sample_offset": {str(offset): sum(sample+offset in marker_set for sample in samples) for offset in (0, 1)},
            "failed_image_join_count": len(joins_failed), "failed_image_join_examples": joins_failed[:10]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--image-archive", type=Path, required=True)
    parser.add_argument("--working-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use the coordinator's verified compute allocation")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.working_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with (args.bids_root/"participants.tsv").open(newline="", encoding="utf-8-sig") as source:
        participants = list(csv.DictReader(source, delimiter="\t"))
    provider_excluded = {row["participant_id"] for row in participants if integer(row["exclude"]) == 1}
    if provider_excluded != EXCLUDED:
        raise ValueError("Provider exclusions differ from fixed contract; no roles may be assigned")
    eligible = [row["participant_id"] for row in participants if row["participant_id"] not in EXCLUDED]
    if len(eligible) != 46:
        raise ValueError("Eligible cohort count differs from 46; contract requires revision before analysis")
    with (args.metadata_root/"payload/image-paths.csv").open(encoding="utf-8-sig") as source:
        metadata_images = {image_key(line.strip()) for line in source if line.strip()}
    archive_images = defaultdict(list)
    with zipfile.ZipFile(args.image_archive) as archive:
        for member in archive.infolist():
            if not member.is_dir() and member.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                archive_images[image_key(member.filename)].append(member.filename)
    results = []
    join_path = args.working_dir/"event_image_join.tsv.gz"
    with gzip.open(join_path, "wt", newline="", encoding="utf-8") as target:
        join = csv.writer(target, delimiter="\t")
        join.writerow(["participant", "is_repeated", "block", "physical_sequence", "presentation", "sample",
                       "concept_number", "exemplar_number", "canonical_image", "archive_member", "metadata_snapshot", "duration_seconds"])
        for participant in eligible:
            try:
                result = qualify_participant(args.bids_root, participant, metadata_images, archive_images, join)
            except (ValueError, KeyError, OSError) as error:
                result = {"participant": participant, "all_checks_pass": False, "structural_error": str(error)}
            results.append(result)
            print(json.dumps({"participant": participant, "all_checks_pass": result["all_checks_pass"],
                              "failed_checks": [key for key, passed in result.get("checks", {}).items() if not passed],
                              "structural_error": result.get("structural_error")}), flush=True)
    channel_sets = [set(result["channels"]) for result in results if "channels" in result]
    common = sorted(set.intersection(*channel_sets)) if len(channel_sets) == 46 else []
    report = {"analysis_id": "input_role_preparation", "phase": "outcome_blind_structural_qualification",
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "observed_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic()-started,
              "provider_exclusions": sorted(provider_excluded), "potentially_eligible_participants": eligible,
              "all_structural_checks_pass": all(result["all_checks_pass"] for result in results),
              "common_recorded_channel_count": len(common), "common_recorded_channel_order": common,
              "common_sensor_inventory_has_62": len(common) == 62,
              "metadata_image_keys": len(metadata_images), "archive_image_keys": len(archive_images),
              "event_image_join": str(join_path), "participant_results": results,
              "eeg_or_image_values_opened": False, "roles_assigned": False, "audit_seal_verified": False,
              "join_frozen": False, "timing_rule_finalized": False,
              "marker_position_reference": "https://pressrelease.brainproducts.com/markers/",
              "next_action": "Resolve any structural/timing diagnostics before exact contract role optimization; no EEG access."}
    (args.output_dir/"structural_input_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: value for key, value in report.items() if key != "participant_results"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
