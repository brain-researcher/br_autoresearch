#!/usr/bin/env python3
"""Steward-only EP02 development pack: the nine controls and nothing else.

The provider MAT holds every evidence role in one compressed variable, so the
steward must decompress it in memory. This program serializes only the nine
development rows, reusing the trusted `materialize_roles.write_role_pack`
format that `development_role_reader.py` admits. It creates no evaluator vault,
audit-structural packet or public manifest, and it prints only structural
metadata (role, identities, shapes, pack root), never array values.
Run it inside a Slurm allocation with a job-specific `$SCRATCH` destination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import materialize_roles as trusted  # noqa: E402

from scipy.io import loadmat, whosmat  # noqa: E402

ROLE = "development_full"


def verify_source(path: Path, expected_bytes: int, expected_md5: str, expected_sha256: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("source must be a regular, non-symlink file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("source descriptor is not a regular file")
        if os.fstat(stream.fileno()).st_size != expected_bytes:
            raise ValueError("source byte count differs from the frozen value")
        md5, sha256 = hashlib.md5(), hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    if md5.hexdigest() != expected_md5 or sha256.hexdigest() != expected_sha256:
        raise ValueError("source digest differs from the frozen provider/local value")
    return sha256.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--cohort-map", type=Path, required=True)
    parser.add_argument("--development-dir", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--expected-md5", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-cohort-map-sha256", required=True)
    parser.add_argument("--observed-on", required=True)
    args = parser.parse_args()

    source = args.mat.absolute()
    destination = args.development_dir.absolute()
    source_sha256 = verify_source(source, args.expected_bytes, args.expected_md5, args.expected_sha256)
    cohort_bytes = trusted.read_regular_bytes(args.cohort_map.absolute())
    cohort_sha256 = trusted.sha256_bytes(cohort_bytes)
    if cohort_sha256 != args.expected_cohort_map_sha256:
        raise ValueError("cohort-map SHA-256 differs from the frozen value")
    _, lookup = trusted.load_cohort_map(cohort_bytes, source_sha256)
    if trusted.role_records(lookup, ROLE) != [1, 2, 4, 6, 9, 11, 15, 16, 19]:
        raise ValueError("cohort map does not yield the nine fixed development controls")
    if whosmat(str(source)) != [("seshMerge", (1, 24), "struct")]:
        raise ValueError("unexpected MAT top-level layout")
    trusted.require_new_directory(destination, "development pack")

    loaded = loadmat(str(source), variable_names=["seshMerge"], struct_as_record=True, squeeze_me=False)
    sessions = loaded.pop("seshMerge")
    if sessions.shape != (1, 24) or tuple(sessions.dtype.names or ()) != trusted.EXPECTED_FIELDS:
        raise ValueError("loaded MAT schema differs from the frozen schema")

    staging = Path(tempfile.mkdtemp(prefix=".development_full.", dir=destination.parent))
    try:
        manifest = trusted.write_role_pack(
            staging / "pack", sessions, lookup, ROLE, args.observed_on,
            source_sha256, cohort_sha256, trusted.file_digest(Path(trusted.__file__).resolve()))
        del sessions, loaded
        (staging / "pack").replace(destination)
        os.chmod(destination, 0o550)
        staging.rmdir()
    finally:
        if staging.exists():
            for child in staging.rglob("*"):
                child.chmod(0o700 if child.is_dir() else 0o600)
            shutil.rmtree(staging)

    record = {
        "role": ROLE,
        "development_dir": str(destination),
        "pack_root_sha256": manifest["pack_root_sha256"],
        "records": [
            {"source_index_1_based": entry["source_index_1_based"],
             "shapes": {field: entry["fields"][field]["shape"] for field in ("trialID", "seshID", "lickState", "lick")},
             "dtypes": {field: entry["fields"][field]["dtype"] for field in ("trialID", "seshID", "lickState", "lick")}}
            for entry in manifest["record_entries"]
        ],
        "audit_or_evaluator_pack_created": False,
        "array_values_printed": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    print(json.dumps(record, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
