#!/usr/bin/env python3
"""Coordinator's closed-metadata handoff followed by EP06 development export.

Only EP07's completed freeze/opening timestamps and closed trial-role CSV are
projected across episodes. No scientific score, fitted state, prediction or
neural array is imported. The EP06 exporter subsequently reads its own fixed
public source. This is the selected procedural access order, not OS isolation.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import time


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("explicit UTC stage timestamps are required")
    return parsed


def record_handoff(episode, specs, handoff_path, check):
    # This coordinator-only projection reads closed operational metadata;
    # candidate/scorer/exporter modules do not read sibling artifacts.
    check()
    ep07 = episode.parent / "episode07_lfp_session_transfer"
    primary = ep07 / "outputs/runs/primary_evaluation/attempt001"
    # The mixed analysis freeze contains development scores and is deliberately
    # not read. EP07 projects only freeze/check completion facts into its
    # score-free opening status before contact, then marks that opening closed.
    opening = json.loads((primary / "opening.json").read_text())
    if (opening.get("analysis_freeze_complete") is not True
            or opening.get("scientific_checks_complete") is not True
            or opening.get("ten_paired_controls_complete") is not True
            or opening.get("four_fixed_recoveries_complete") is not True
            or opening.get("complete") is not True
            or opening.get("state") != "primary_evaluation_complete"
            or opening.get("held_out_spike_contact") is not True):
        raise PermissionError("EP07 must have a full freeze and completed primary opening")
    times = opening["analysis_frozen_utc"], opening["started_utc"], opening["ended_utc"]
    if not timestamp(times[0]) <= timestamp(times[1]) <= timestamp(times[2]):
        raise ValueError("EP07 stage chronology is inconsistent")
    by_session = {(spec.animal, spec.day): spec for spec in specs}
    records, observed = [], set()
    with (ep07 / "outputs/runs/source_support/attempt002/trial_roles.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            check()
            key = row["animal_implant"], row["day"]
            if key not in by_session:
                continue
            spec = by_session[key]
            index = int(row["source_index_zero_based"])
            trial_key = spec.animal, spec.filename, index
            if (row["ep06_session_role"] != "development" or row["regions"] != "M1;PMd"
                    or int(row["original_session_position"]) != spec.original_position
                    or row["role"] not in ("training", "calibration", "development", "held_out")
                    or row["trial_id"] != f"trial_{index + 1:06d}"
                    or not 0 <= index < spec.trials or trial_key in observed):
                raise ValueError("closed EP07 roles do not bind the complete EP06 development roster")
            observed.add(trial_key)
            records.append({"animal": spec.animal, "session_name": spec.filename,
                            "source_trial_index_zero_based": index, "ep07_role": row["role"]})
    expected = {(spec.animal, spec.filename, index) for spec in specs for index in range(spec.trials)}
    if observed != expected:
        raise ValueError("closed role projection must retain all1954 development trials")
    handoff = {"source_episode": "episode07_lfp_session_transfer",
        "analysis_frozen_utc": times[0], "primary_opening_started_utc": times[1],
        "primary_opening_ended_utc": times[2], "primary_opening_complete": True,
        "ep07_views": "source_support_attempt002", "excluded_session": "Mihili_CO_20140306.mat",
        "records": records, "projection_scope": "closed_role_metadata_and_stage_timestamps_only",
        "created_utc": datetime.now(timezone.utc).isoformat(), "job_id": os.environ["SLURM_JOB_ID"]}
    check()
    with handoff_path.open("x") as handle:
        json.dump(handoff, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    started = time.time()
    if (not os.environ.get("SLURM_JOB_ID", "").isdigit() or "-ln" in socket.gethostname()
            or os.environ.get("SLURM_CPUS_PER_TASK") not in ("1", "2")
            or os.environ.get("SLURM_MEM_PER_NODE") != "7168"
            or not 0 < args.deadline_epoch - started <= 900):
        raise SystemExit("the selected1 CPU/7GiB allocation and at most900 seconds are required")
    def check():
        if time.time() >= args.deadline_epoch:
            raise TimeoutError("EP06 preparation deadline reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > int(6.5 * 2**30):
            raise MemoryError("EP06 preparation peak RSS exceeds6.5GiB")
    from prepare_development_views import EPISODE, HANDOFF_PATH, SESSIONS, prepare_development_views
    record_handoff(EPISODE, SESSIONS, HANDOFF_PATH, check)
    directory = Path(os.environ["SCRATCH"]) / "br_autoresearch/episode06_lfp_regional_fingerprints/development001"
    durable = EPISODE / "outputs/runs/source_session_support/development001"
    if not directory.parent.is_dir() or not durable.is_dir():
        raise ValueError("the selected task-owned preparation parents must exist")
    try:
        path = prepare_development_views(opening_handoff=HANDOFF_PATH,
                                         output_directory=directory, check=check)
        print(json.dumps({"state": "development_views_complete", "path": str(path)}), flush=True)
    finally:
        # Preserve small role/preparation records on Oak, including a failed
        # export. The reproducible neural role view remains on scratch.
        for name in ("report.json", "trial_roles.csv"):
            source = directory / name
            if source.exists():
                with source.open("rb") as src, (durable / name).open("xb") as dst:
                    shutil.copyfileobj(src, dst)


if __name__ == "__main__":
    main()
