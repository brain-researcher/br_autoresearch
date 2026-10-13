#!/usr/bin/env python3
"""Trusted builder: copy only raw, policy-selected development archive members.

Decompression reads selected development payload bytes. It performs no signal
calculation and never decompresses a held-out, WS, AllLifts or derivative-MAT
member. Output read-only mode is convenience, not proof of user isolation.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import time
import zipfile

from inventory_source_members import classify_member


def copy_selected(archive, member, destination, source_kind, role, report_rows):
    info = archive.getinfo(member)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info, "r") as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    size = destination.stat().st_size
    if size != info.file_size:
        raise RuntimeError("incomplete development-member copy")
    destination.chmod(0o444)
    report_rows.append({"source": source_kind, "role": role, "member": member,
                        "destination": str(destination), "bytes": size})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--member-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("source preparation requires coordinator-owned compute")
    plan = json.loads(args.member_plan.read_text())
    planned = plan["members"]
    if len(planned) != 549 or plan["member_count"] != 549 or sum(row["uncompressed_bytes"] for row in planned) != 9038196400 or plan["uncompressed_bytes"] != 9038196400:
        raise ValueError("selected member list/size differs from observed development-only plan")
    expected = {}
    for row in planned:
        path = PurePosixPath(row["member"])
        if row["source"] == "way":
            match = re.fullmatch(r"HS_P(\d+)_S([1-7])\.mat", row["member"])
            if not match or int(match[1]) not in range(1, 13):
                raise ValueError("noncontinuous/nondevelopment WAY member in plan")
            archive_path = args.source_root / "archives" / "way" / f"P{int(match[1])}.zip"
            role = "fit_and_inner_selection" if int(match[2]) <= 6 else "aggregate_development_feedback"
        elif row["source"] == "self_paced":
            classification = classify_member("self_paced", None, row["member"])
            if classification.get("role") != "development" or not row["member"].startswith("sub-") or path.suffix.lower() not in (".eeg", ".vhdr", ".vmrk"):
                raise ValueError("nonraw/nondevelopment self-paced member in plan")
            archive_path = args.source_root / "archives" / "freewill" / "Freewill_EEG_Reaching_Grasping.zip"
            role = "development"
        else:
            raise ValueError("unknown source in selected plan")
        key = (str(archive_path), row["member"])
        if str(archive_path) != row["archive"] or role != row["role"] or key in expected:
            raise ValueError("source/role mismatch or duplicate in selected plan")
        expected[key] = row["uncompressed_bytes"]
    # Validate every selected central-directory size before any decompression.
    for archive_path in sorted({key[0] for key in expected}):
        with zipfile.ZipFile(archive_path) as archive:
            for (source_path, member), size in expected.items():
                if source_path == archive_path and archive.getinfo(member).file_size != size:
                    raise ValueError("source member size changed since structural inventory")
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "selected_member_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    started = time.monotonic()
    rows = []
    for participant in range(1, 13):
        archive_path = args.source_root / "archives" / "way" / f"P{participant}.zip"
        with zipfile.ZipFile(archive_path) as archive:
            for series in range(1, 8):
                member = f"HS_P{participant}_S{series}.mat"
                if (str(archive_path), member) not in expected:
                    raise ValueError("required WAY member absent from selected plan")
                role = "fit_and_inner_selection" if series <= 6 else "aggregate_development_feedback"
                destination = args.output_root / "way" / role / member
                copy_selected(archive, member, destination, "way", role, rows)
        print(f"copied WAY P{participant} continuous development series1-7", flush=True)
    archive_path = args.source_root / "archives" / "freewill" / "Freewill_EEG_Reaching_Grasping.zip"
    with zipfile.ZipFile(archive_path) as archive:
        for member in sorted(archive.namelist()):
            role = classify_member("self_paced", None, member)
            path = PurePosixPath(member)
            if (str(archive_path), member) not in expected:
                continue
            destination = args.output_root / "self_paced" / "development" / str(path)
            copy_selected(archive, member, destination, "self_paced", "development", rows)
    if len(rows) != len(planned) or sum(row["bytes"] for row in rows) != plan["uncompressed_bytes"]:
        raise RuntimeError("development extraction did not complete the selected plan")
    report = {
        "analysis_id": "source_views_and_timing", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
        "status": "development_view_materialized", "seconds": time.monotonic() - started,
        "members_copied": len(rows), "uncompressed_bytes_copied": sum(row["bytes"] for row in rows),
        "source_role_counts": dict(Counter(f"{row['source']}:{row['role']}" for row in rows)),
        "source_payload_bytes_read": True, "held_out_member_payload_read": False,
        "signal_values_computed_or_displayed": False, "eeg_forecast_scores_computed": False,
        "read_only_file_mode": "0444", "os_level_user_isolation_claim": False,
        "provider_windowed_or_derivative_payload_read": False,
        "timing_bound_verified": False, "onset_rules_frozen": False, "candidate_scoring_ready": False,
        "storage": "Regenerable scratch projection; durable immutable acquisition archives are unchanged.",
        "files": rows,
    }
    (args.output_root / "development_view.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
