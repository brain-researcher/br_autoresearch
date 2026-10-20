#!/usr/bin/env python3
"""Extract EP16 development features (window means, cursor at start) for one development participant.

Refuses any non-development participant before opening an archive (ep16_roles). Reads only
the sessions of the participant's registered passing P0 component. One opaque MAT member is
spooled at a time into an unlinked private file; per-session compact arrays go to a transient
$SCRATCH directory, and a count-only report goes to the episode run directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import socket
import tarfile
import tempfile
import time

import numpy as np

from ep16_roles import DEVELOPMENT, SUPPLEMENTARY_DEVELOPMENT, authorize_protected_read
from mat5_neural import read_session_windows
from mat5_structural import field_vector


def per_electrode_arrays(electrodes, count):
    """Array label per electrode; a single stored label applies to every electrode."""
    records = electrodes if isinstance(electrodes, list) else [electrodes]
    labels = []
    for record in records:
        value = record["array"]
        labels.extend([value] if isinstance(value, str) else [str(v) for v in value])
    if len(labels) == 1 and count > 1:
        labels = labels * count
    if len(labels) != count:
        raise ValueError("Array labels do not match electrode count")
    return np.asarray(labels)


def component_days(report_path, participant):
    report = json.loads(Path(report_path).read_text())
    days = [d["session_days"] for g in report["variants"]["P0"]["groups_with_designs"] if g["participant"] == participant
            for d in g["designs"] if d["conditional_numerical_pass"]]
    if len(days) != 1:
        raise ValueError(f"Expected one passing P0 component for {participant}, found {len(days)}")
    return sorted(int(d) for d in days[0])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", required=True, choices=sorted(DEVELOPMENT | SUPPLEMENTARY_DEVELOPMENT))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--support-report", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True, help="transient $SCRATCH destination")
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Compute allocation required")
    authorize_protected_read(args.participant)  # before the archive is opened
    days = set(component_days(args.support_report, args.participant))
    args.report_dir.mkdir(parents=True, exist_ok=False)
    args.feature_dir.mkdir(parents=True, exist_ok=False)
    args.scratch_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    archive = args.source_root / "source" / f"decoding_{args.participant}.tar.gz"
    pattern = re.compile(rf"decoding/{args.participant}/{args.participant}_day_([0-9]+)_decoding\.mat")
    started, sessions, decoded, member_bytes = time.monotonic(), [], set(), 0
    with tempfile.TemporaryDirectory(prefix="ep16-dev-", dir=args.scratch_dir) as private, tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            match = pattern.fullmatch(member.name.removeprefix("./"))
            if not member.isfile() or match is None or int(match.group(1)) not in days:
                continue
            day = int(match.group(1))
            t0 = time.monotonic()
            with tempfile.TemporaryFile(dir=private) as opaque:
                shutil.copyfileobj(stream.extractfile(member), opaque, length=1024 * 1024)
                values, out, index, access = read_session_windows(opaque, args.participant)
            if int(np.asarray(values["post_implant_day"]).reshape(-1)[0]) != day:
                raise ValueError("In-file day differs from member name")
            electrodes = values["electrodes"]
            ids = field_vector(electrodes, "electrode_id").astype(np.int64)
            arrays = per_electrode_arrays(electrodes, len(ids))
            np.savez_compressed(args.feature_dir / f"{args.participant}_day_{day}.npz", day=day,
                trial_start=index["starts"], trial_end=index["ends"], trial_block=index["trial_blocks"],
                whole=index["whole"], window_inside_block=index["window_inside_block"],
                target_start=np.asarray(values["trial_targets"], dtype=np.float64), cursor_start=out["cursor_start"],
                sbp=out["sbp"], tx_4_5=out["tx_4_5"], sbp_odd_even=out["sbp_odd_even"],
                electrode_id=ids, electrode_array=arrays,
                block_id=field_vector(values["blocks"], "block_id").astype(np.int64),
                block_start=field_vector(values["blocks"], "sample_start_index").astype(np.int64),
                block_end=field_vector(values["blocks"], "sample_end_index").astype(np.int64))
            decoded |= set(access["protected_decoded_paths"])
            member_bytes += member.size
            sessions.append({"day": day, "trials": int(len(index["starts"])), "electrodes": int(out["sbp"].shape[1]),
                             "windows_inside_block": int(index["window_inside_block"].sum()),
                             "nonfinite_sbp_channel_windows": int(np.isnan(out["sbp"][index["window_inside_block"]]).sum()),
                             "nonfinite_tx_4_5_channel_windows": int(np.isnan(out["tx_4_5"][index["window_inside_block"]]).sum()),
                             "seconds": round(time.monotonic() - t0, 3)})
            print(json.dumps(sessions[-1]), flush=True)
    missing = sorted(days - {s["day"] for s in sessions})
    report = {"analysis_id": "analysis_specification", "phase": "development_feature_extraction",
              "participant": args.participant, "role": "development", "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "hostname": socket.gethostname(), "observed_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": round(time.monotonic() - started, 3), "archive": str(archive),
              "component_days_requested": len(days), "sessions_extracted": len(sessions), "missing_days": missing,
              "opaque_member_bytes": member_bytes, "protected_decoded_paths": sorted(decoded),
              "feature_dir_transient": str(args.feature_dir), "sessions": sessions,
              "other_participants_opened": False}
    (args.report_dir / "feature_extraction_report.json").write_text(json.dumps(report, indent=1) + "\n")
    if missing:
        raise SystemExit(f"Missing component days: {missing}")


if __name__ == "__main__":
    main()
