#!/usr/bin/env python3
"""Read only development BrainVision text headers and three source documents."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import socket
import zipfile

from inventory_source_members import classify_member, DEVELOPMENT_PEOPLE


DOCUMENTS = ("README.md", "task-reachingandgrasping_events.json", "code/experiment/PlayStartEnd.m")
SCIENTIFIC_CHANNELS = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "C4", "T8", "TP9", "CP5", "CP1", "CP2", "CP6", "TP10", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2")


def parse_header(text):
    sections, section = {}, None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            sections.setdefault(section, {})
        elif section and "=" in line:
            key, value = line.split("=", 1)
            sections[section][key.strip()] = value.strip()
    common = sections["Common Infos"]
    channel_fields = sections["Channel Infos"]
    channel_count = int(common["NumberOfChannels"])
    channels = []
    for channel in range(1, channel_count + 1):
        pieces = channel_fields[f"Ch{channel}"].split(",")
        channels.append({"name": pieces[0].replace("\\1", ","),
                         "reference": pieces[1] if len(pieces) > 1 else "",
                         "resolution": pieces[2] if len(pieces) > 2 else "",
                         "unit": pieces[3] if len(pieces) > 3 else ""})
    interval = float(common["SamplingInterval"])
    return {"channel_count": channel_count, "sampling_interval_microseconds": interval,
            "sample_rate_hz": 1e6 / interval, "data_format": common["DataFormat"],
            "data_orientation": common["DataOrientation"],
            "binary_format": sections.get("Binary Infos", {}).get("BinaryFormat"),
            "data_file": common["DataFile"], "marker_file": common["MarkerFile"],
            "channels": channels,
            "missing_scientific_channels": sorted(set(SCIENTIFIC_CHANNELS) - {item["name"] for item in channels})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("header inspection requires the task-owned compute allocation")
    args.output_root.mkdir(parents=True, exist_ok=False)
    documents_root = args.output_root / "documents"
    documents_root.mkdir()
    reports = []
    with zipfile.ZipFile(args.archive) as archive:
        members = {item.filename: item for item in archive.infolist()}
        for name in sorted(members):
            role = classify_member("self_paced", None, name)
            if role.get("role") != "development" or PurePosixPath(name).suffix.lower() != ".vhdr":
                continue
            if members[name].file_size > 100000:
                raise ValueError("unexpected header size; inspect source identity before reading")
            text = archive.read(name).decode("utf-8-sig", errors="replace")
            header = parse_header(text)
            parent = PurePosixPath(name).parent
            data_name, marker_name = str(parent / header["data_file"]), str(parent / header["marker_file"])
            header.update({"participant": role["participant"], "member": name,
                           "data_member_present": data_name in members, "marker_member_present": marker_name in members})
            reports.append(header)
        for name in DOCUMENTS:
            if name not in members or members[name].file_size > 100000:
                raise ValueError("documentary member identity/size differs from observed inventory")
            target = documents_root / (name.replace("/", "__"))
            target.write_bytes(archive.read(name))
    report = {
        "analysis_id": "source_views_and_timing", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
        "status": "development_header_inspection_complete", "header_count": len(reports),
        "participants_seen": sorted({item["participant"] for item in reports}),
        "missing_development_participants": sorted(DEVELOPMENT_PEOPLE - {item["participant"] for item in reports}),
        "sampling_rate_counts": dict(Counter(str(item["sample_rate_hz"]) for item in reports)),
        "channel_count_counts": dict(Counter(str(item["channel_count"]) for item in reports)),
        "headers_missing_scientific_channels": sum(bool(item["missing_scientific_channels"]) for item in reports),
        "source_documents": list(DOCUMENTS), "signal_member_access": False,
        "event_value_access": False, "held_out_header_access": False,
        "timing_bound_verified": False, "valid_run_eligibility_verified": False,
        "production_scoring_ready": False, "headers": reports,
    }
    (args.output_root / "development_headers.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "headers"}, indent=2))


if __name__ == "__main__":
    main()
