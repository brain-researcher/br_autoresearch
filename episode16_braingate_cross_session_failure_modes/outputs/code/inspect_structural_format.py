#!/usr/bin/env python3
"""Read the MAT format header of one already-development T10 session only.

No variable, neural value, cursor trajectory, impedance or target is loaded.
The purpose is to choose a structural-field reader for the required support
assessment without opening unassigned participant outcomes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import stat
import tarfile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use the coordinator's verified compute allocation")
    if args.output_dir.exists():
        raise FileExistsError("Choose a new immutable attempt directory")
    archive = args.source_root/"source/decoding_T10.tar.gz"
    record = None
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            name = member.name.removeprefix("./")
            if not member.isfile() or not re.fullmatch(r"decoding/T10/T10_day_[0-9]+_decoding\.mat", name):
                continue
            source = stream.extractfile(member)
            if source is None:
                raise ValueError("Session member is unreadable")
            header = source.read(128)
            if len(header) != 128:
                raise ValueError("Truncated MAT header")
            description = header[:116].decode("ascii", errors="replace").rstrip(" \x00")
            if header.startswith(b"MATLAB 7.3 MAT-file"):
                kind = "MATLAB_7p3_HDF5"
            elif header.startswith(b"MATLAB 5.0 MAT-file") and header[126:128] in (b"IM", b"MI"):
                kind = "MATLAB_5_structured_binary"
            else:
                kind = "unrecognized_requires_documented_reader"
            record = {"member": name, "uncompressed_bytes": member.size,
                      "format": kind, "header_description": description,
                      "header_bytes_read": 128}
            break
    if record is None:
        raise ValueError("The expected T10 decoding member was not found")
    report = {"analysis_id": "structural_support", "phase": "source_format_only",
              "participant": "T10", "participant_role": "preexisting_development",
              "archive": str(archive), "archive_mode": oct(stat.S_IMODE(archive.stat().st_mode)),
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "observed_utc": datetime.now(timezone.utc).isoformat(), "session_header": record,
              "variables_deserialized": [], "held_out_outcomes_opened": False,
              "role_assignment_complete": False}
    args.output_dir.mkdir(parents=True)
    (args.output_dir/"structural_format_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
