#!/usr/bin/env python3
"""Read ZIP central directories only; never open or extract member contents."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import zipfile


DEVELOPMENT_PEOPLE = {"sub-02", "sub-03", "sub-04", "sub-06", "sub-08", "sub-09",
                      "sub-11", "sub-12", "sub-14", "sub-15", "sub-17", "sub-18",
                      "sub-20", "sub-21", "sub-22"}
HELD_OUT_PEOPLE = {"sub-01", "sub-05", "sub-07", "sub-10", "sub-13", "sub-16", "sub-19", "sub-23"}


def classify_member(source, archive_person, name):
    """Only exact declared identity tokens classify; unknown/mixed stays closed."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return {"role": "unresolved_unsafe_path"}
    if "__MACOSX" in path.parts or path.name.startswith("._"):
        return {"role": "non_signal_archive_metadata"}
    if source == "way":
        matched = re.search(r"(?:^|[_/])P(\d+)_S(\d+)(?:[_.]|$)", name, flags=re.IGNORECASE)
        if not matched:
            return {"role": "unresolved_or_mixed_series", "participant": archive_person}
        person, series = map(int, matched.groups())
        if person != archive_person or series not in range(1, 10):
            return {"role": "unresolved_identity_conflict"}
        role = "fit_and_inner_selection" if series <= 6 else ("aggregate_development_feedback" if series == 7 else "held_out")
        return {"role": role, "participant": person, "series": series}
    people = {part.lower() for part in path.parts if re.fullmatch(r"sub-\d{2}", part, flags=re.IGNORECASE)}
    if len(people) != 1:
        return {"role": "unresolved_or_dataset_metadata"}
    person = next(iter(people))
    if person in DEVELOPMENT_PEOPLE:
        return {"role": "development", "participant": person}
    if person in HELD_OUT_PEOPLE:
        return {"role": "held_out", "participant": person}
    return {"role": "unresolved_participant", "participant": person}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("source inventory requires the coordinated compute allocation")
    args.output_root.mkdir(parents=True, exist_ok=False)
    sources = [("way", person, args.source_root / "archives" / "way" / f"P{person}.zip") for person in range(1, 13)]
    sources.append(("self_paced", None, args.source_root / "archives" / "freewill" / "Freewill_EEG_Reaching_Grasping.zip"))
    rows, archives = [], []
    for source, person, archive_path in sources:
        with zipfile.ZipFile(archive_path) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            for member in members:
                rows.append({"source": source, "archive": str(archive_path), "member": member.filename,
                             "uncompressed_bytes": member.file_size, "compressed_bytes": member.compress_size,
                             "suffix": PurePosixPath(member.filename).suffix.lower(),
                             **classify_member(source, person, member.filename)})
            archives.append({"source": source, "archive": str(archive_path), "members": len(members),
                             "uncompressed_bytes": sum(member.file_size for member in members)})
    report = {
        "analysis_id": "source_views_and_timing", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
        "status": "structural_inventory_complete", "archives": archives,
        "member_count": len(rows),
        "source_role_counts": dict(Counter(f"{row['source']}:{row['role']}" for row in rows)),
        "source_suffix_counts": dict(Counter(f"{row['source']}:{row['suffix']}" for row in rows)),
        "member_payload_read": False, "members_extracted": 0, "held_out_signal_access": False,
        "physical_role_separation_provisioned": False, "timing_bound_verified": False,
        "forecast_scoring_ready": False,
        "next": "Resolve member naming, then inspect only development text headers and prepare trusted role-selected exports.",
    }
    (args.output_root / "archive_member_index.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_root / "source_inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "member_count", "source_role_counts", "source_suffix_counts")}, indent=2))


if __name__ == "__main__":
    main()
