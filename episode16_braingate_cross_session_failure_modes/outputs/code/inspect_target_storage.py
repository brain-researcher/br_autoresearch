#!/usr/bin/env python3
"""Record target_position MAT storage headers for the nine fixed EP16 archives.

No target, neural, cursor, electrode or impedance numeric payload is decoded.
Storage precision does not establish task-controller precision or eligibility.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import struct
import tarfile
import tempfile
import time

from mat5_structural import Inflated, element, exact


EXPECTED = {"T2": 42, "T3": 8, "T5": 92, "T6": 124, "T7": 35,
            "T8": 58, "T9": 117, "T10": 57, "T11": 196}
CLASSES = {6: "mxDOUBLE_CLASS", 7: "mxSINGLE_CLASS", 8: "mxINT8_CLASS",
           9: "mxUINT8_CLASS", 10: "mxINT16_CLASS", 11: "mxUINT16_CLASS",
           12: "mxINT32_CLASS", 13: "mxUINT32_CLASS", 14: "mxINT64_CLASS",
           15: "mxUINT64_CLASS"}
PRIMITIVES = {1: ("miINT8", "i1", 1), 2: ("miUINT8", "u1", 1),
              3: ("miINT16", "i2", 2), 4: ("miUINT16", "u2", 2),
              5: ("miINT32", "i4", 4), 6: ("miUINT32", "u4", 4),
              7: ("miSINGLE", "f4", 4), 9: ("miDOUBLE", "f8", 8),
              12: ("miINT64", "i8", 8), 13: ("miUINT64", "u8", 8)}


def header_field(stream, endian, maximum_bytes, label):
    item = element(stream, endian)
    if item.data.remaining > maximum_bytes:
        raise ValueError(f"Unexpected {label} header length")
    raw = exact(item.data, item.data.remaining)
    item.finish()
    return item.kind, raw


def inspect_session(source):
    source.seek(0)
    header = exact(source, 128)
    if not header.startswith(b"MATLAB 5.0 MAT-file"):
        raise ValueError("Only the already-observed MATLAB Level5 format is supported")
    endian = {b"IM": "<", b"MI": ">"}.get(header[126:128])
    if endian is None:
        raise ValueError("Unknown MAT byte-order tag")
    selected = None
    while True:
        outer = element(source, endian, allow_eof=True)
        if outer is None:
            break
        compressed = outer.kind == 15
        if compressed:
            inflated = Inflated(outer.data)
            matrix = element(inflated, endian)
            if matrix.kind != 14:
                raise ValueError("Compressed top-level element is not miMATRIX")
            stream = matrix.data
        elif outer.kind == 14:
            stream = outer.data
        else:
            raise ValueError("Unsupported top-level MAT encoding")
        flag_kind, flags_raw = header_field(stream, endian, 8, "array flags")
        shape_kind, shape_raw = header_field(stream, endian, 32, "dimensions")
        _, name_raw = header_field(stream, endian, 256, "array name")
        if flag_kind != 6 or len(flags_raw) != 8 or shape_kind != 5 or len(shape_raw) % 4:
            raise ValueError("Invalid MAT array-header encoding")
        name = name_raw.decode("utf-8")
        if name == "target_position":
            if selected is not None:
                raise ValueError("Duplicate target_position variable")
            flags = struct.unpack(endian + "II", flags_raw)[0]
            array_class = flags & 255
            shape = list(struct.unpack(endian + "i" * (len(shape_raw) // 4), shape_raw))
            if array_class not in CLASSES or flags & (0x800 | 0x200):
                raise ValueError("Target storage must be real nonlogical dense numeric")
            if len(shape) != 2 or shape[0] < 1 or shape[1] != 2:
                raise ValueError("Target storage differs from documented n_time_bins by2")
            # Read the primitive tag only. Never read/finish this numeric body.
            primitive = element(stream, endian)
            if primitive.kind not in PRIMITIVES:
                raise ValueError("Unknown target numeric primitive tag")
            primitive_name, dtype_code, itemsize = PRIMITIVES[primitive.kind]
            body_bytes = primitive.data.remaining
            if body_bytes != math.prod(shape) * itemsize:
                raise ValueError("Target numeric byte count differs from header shape")
            selected = {"variable": name, "matlab_array_class_id": array_class,
                "matlab_array_class": CLASSES[array_class],
                "numeric_primitive_tag_id": primitive.kind,
                "numeric_primitive_tag": primitive_name,
                "storage_dtype": endian + dtype_code, "itemsize_bytes": itemsize,
                "shape": shape, "declared_numeric_body_bytes": body_bytes,
                "top_level_compressed": compressed,
                "numeric_values_decoded": 0}
        # Crucial: do not finish the inflated matrix. Skip remaining compressed
        # bytes on the opaque temporary file; no numeric body is decompressed.
        # For uncompressed variables, this similarly seeks over their bodies.
        outer.finish()
    if selected is None:
        raise ValueError("target_position header absent")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned compute allocation required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.scratch_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    begin = time.monotonic()
    records, counts, errors = [], Counter(), []
    opaque_bytes, largest_member = 0, 0
    current = {"participant": None, "member": None}
    try:
        with tempfile.TemporaryDirectory(prefix="ep16-target-headers-", dir=args.scratch_dir) as private:
            with (args.output_dir / "target_storage_headers.jsonl").open("x") as output:
                for participant in EXPECTED:
                    archive = args.source_root / "source" / f"decoding_{participant}.tar.gz"
                    current = {"participant": participant, "member": None}
                    seen = set()
                    pattern = re.compile(rf"decoding/{participant}/{participant}_day_([0-9]+)_decoding\.mat")
                    with tarfile.open(archive, "r|gz") as members:
                        for member in members:
                            name = member.name.removeprefix("./")
                            match = pattern.fullmatch(name)
                            if not member.isfile() or match is None:
                                continue
                            current = {"participant": participant, "member": name}
                            if name in seen:
                                raise ValueError("Duplicate exact session member")
                            seen.add(name)
                            with tempfile.TemporaryFile(dir=private) as opaque:
                                body = members.extractfile(member)
                                if body is None:
                                    raise ValueError("Unreadable session member")
                                shutil.copyfileobj(body, opaque, length=1024 * 1024)
                                info = inspect_session(opaque)
                            opaque_bytes += member.size
                            largest_member = max(largest_member, member.size)
                            info.update({"participant": participant, "day": int(match.group(1)),
                                "archive": str(archive), "member": name})
                            records.append(info)
                            counts[participant] += 1
                            output.write(json.dumps(info, allow_nan=False) + "\n")
                            output.flush()
                    if counts[participant] != EXPECTED[participant]:
                        raise ValueError("Session count differs from fixed provider census")
                    print(json.dumps({"participant": participant, "storage_headers": counts[participant],
                        "elapsed_seconds": time.monotonic() - begin}), flush=True)
    except Exception as exc:
        errors.append({**current, "error_type": type(exc).__name__, "message": str(exc)})
    report = {"analysis_id": "structural_support", "phase": "target_storage_headers_only",
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "elapsed_seconds": time.monotonic() - begin,
        "complete": not errors and dict(counts) == EXPECTED,
        "sessions": len(records), "session_counts": dict(counts), "expected_session_counts": EXPECTED,
        "storage_dtype_counts": dict(Counter(row["storage_dtype"] for row in records)),
        "matlab_class_counts": dict(Counter(row["matlab_array_class"] for row in records)),
        "opaque_member_bytes_copied": opaque_bytes, "largest_temporary_member_bytes": largest_member,
        "errors": errors, "numeric_values_decoded": 0,
        "target_values_decoded": False, "neural_cursor_impedance_values_decoded": False,
        "task_controller_resolution_established": False, "schedule_tolerance_changed": False,
        "task_size_classification_established": False, "roles_assigned": False,
        "source_archives_modified": False, "retained_raw_member_copies": 0}
    (args.output_dir / "target_storage_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
