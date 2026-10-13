#!/usr/bin/env python3
"""Score one complete paired successor in a separate, bounded process.

This entrypoint opens development targets only after both published panels
validate. It never fits a candidate or opens held-out data. Starting08 remains
an exhausted, unscored starting opportunity; adaptive progress cannot erase
that deficit or authorize final evaluation. Each selected pair is contacted
at most once, with durable contact state written before target access.
Indices7..13 append under adaptive002 and replay all earlier adaptive001
feedback plus earlier adaptive002 records without resetting the controller.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import socket
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scorer-directory", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    if (not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname()
            or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) not in (1, 2)
            or os.environ.get("SLURM_MEM_PER_NODE") != "7168"
            or args.index not in range(14)):
        raise SystemExit("scorer requires the selected bounded compute allocation and pair")
    episode = Path(__file__).resolve().parents[2]
    expected = Path(os.environ["SCRATCH"]) / "br_autoresearch/episode07_lfp_session_transfer/source_support_attempt002/scorer"
    run_name = "adaptive001" if args.index < 7 else "adaptive002"
    selected_out = episode / "outputs/runs/matched_search" / run_name
    if (args.out.resolve() != selected_out or not args.out.is_dir()
            or args.scorer_directory.resolve() != expected.resolve()):
        raise SystemExit("scorer accepts only the selected attempt and development view")
    remaining = args.deadline_epoch - time.time()
    if not math.isfinite(remaining) or not 0 < remaining <= 3600:
        raise SystemExit("scorer requires a future deadline at most one hour away")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    from adaptive_progress import assess_adaptive_progress

    def check():
        if time.time() >= args.deadline_epoch:
            raise TimeoutError("adaptive scorer deadline reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > int(6.5 * 2**30):
            raise MemoryError("adaptive scorer peak RSS exceeds 6.5 GiB")

    check()
    starts = []
    for attempt, indices in (("attempt001", range(16)), ("attempt002", (8, 10))):
        check()
        report = json.loads((selected_out.parent / attempt / "development_feedback.json").read_text())
        records = report.get("pairs", [])
        expected_ids = {f"starting_{value:02d}" for value in indices}
        if (report.get("evidence_stage") != "development_only"
                or report.get("held_out_opened") is not False or not report.get("ended_utc")
                or len(records) != len(expected_ids)
                or {value.get("candidate_id") for value in records} != expected_ids):
            raise SystemExit("starting report identity or closed development scope differs")
        starts.extend(records)
    previous = []
    for index in range(args.index):
        check()
        directory = selected_out.parent / ("adaptive001" if index < 7 else "adaptive002")
        report = json.loads((directory / f"adaptive_{index:02d}_feedback.json").read_text())
        if (report.get("evidence_stage") != "development_only"
                or report.get("held_out_opened") is not False
                or not report.get("ended_utc")
                or report.get("pair", {}).get("candidate_id") != f"adaptive_{index:02d}"):
            raise SystemExit("prior successor contact is not closed")
        previous.append(report["pair"])
    progress = assess_adaptive_progress(starts, previous)
    if progress.stop_reason is not None:
        raise SystemExit("adaptive stage cannot continue past its recorded stop condition")
    from candidate_predictions import load_candidate_predictions
    from development_score import score_development_panel
    candidate_id = f"adaptive_{args.index:02d}"
    path = args.out / f"{candidate_id}_feedback.json"
    pair = {"candidate_id": candidate_id, "outcome_contact": False,
            "state": "paired_prediction_incomplete_unscored", "streams": {}}
    report = {"evidence_stage": "development_only", "held_out_opened": False,
              "job_id": os.environ["SLURM_JOB_ID"],
              "started_utc": datetime.now(timezone.utc).isoformat(), "pair": pair}
    with path.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)

    def save(target, value):
        temporary = target.with_suffix(".tmp")
        with temporary.open("w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)

    panels = []
    try:
        for stream in ("history_assisted", "today_only"):
            panels.append(load_candidate_predictions(args.out / f"{candidate_id}__{stream}.h5",
                expected_candidate_id=candidate_id, expected_stream=stream, check=check))
    except (FileNotFoundError, ValueError) as error:
        pair.update(error_type=type(error).__name__, error=str(error))
    else:
        pair.update(outcome_contact=True, state="paired_development_contact_started")
        save(path, report)
        for stream, panel in zip(("history_assisted", "today_only"), panels):
            pair["streams"][stream] = {"state": "development_contact_started"}
            save(path, report)
            feedback = score_development_panel(args.scorer_directory, panel.predictions,
                                               panel.calibration_means, check=check)
            available = all(value.available for value in feedback)
            pair["streams"][stream] = {
                "regions": [asdict(value) for value in feedback], "complete": available,
                "global_selection_r2": sum(value.model_r2 for value in feedback) / 2 if available else None}
            save(path, report)
        pair["state"] = "paired_development_scored"
    report["ended_utc"] = datetime.now(timezone.utc).isoformat()
    save(path, report)
    progress = assess_adaptive_progress(starts, previous + [pair])
    save(args.out / "progress.json", asdict(progress))
    print(json.dumps({"candidate_id": candidate_id, "status": progress.status,
                      "stop_reason": progress.stop_reason,
                      "starting_complete": progress.starting_complete,
                      "valid_successor_pairs": progress.valid_successor_pairs,
                      "incumbents": [asdict(value) if value is not None else None
                                     for value in progress.incumbents]}, allow_nan=False), flush=True)
    # An unscored numerical failure consumes an attempt but can be followed by
    # another genuinely distinct successor. Incomplete contact stops the job.
    return 1 if progress.incomplete_contact_ids else 0


if __name__ == "__main__":
    sys.exit(main())
