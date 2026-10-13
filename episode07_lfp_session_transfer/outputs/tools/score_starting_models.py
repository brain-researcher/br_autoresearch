#!/usr/bin/env python3
"""Separate scorer process for paired complete EP07 starting predictions.

Only complete history/today pairs make development contact; a missing/failed
counterpart receives no extra feedback. No candidate fitting code is imported
or called here. The output is aggregate development evidence, never a held-out
decision or a new-day/animal claim. There is no candidate selection or retry.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--scorer-directory", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    if (not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname()
            or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) not in
               (range(2, 4) if args.repair else range(4, 7))
            or os.environ.get("SLURM_MEM_PER_NODE") != ("14336" if args.repair else "28672")):
        raise SystemExit("scorer requires the selected compute allocation")
    episode = Path(__file__).resolve().parents[2]
    expected = Path(os.environ["SCRATCH"]) / "br_autoresearch/episode07_lfp_session_transfer/source_support_attempt002/scorer"
    attempt = "attempt002" if args.repair else "attempt001"
    if (args.out.resolve() != episode / "outputs/runs/matched_search" / attempt
            or args.scorer_directory.resolve() != expected.resolve()):
        raise SystemExit("scorer paths differ from the selected attempt and development view")
    indices = (8, 10) if args.repair else tuple(range(16))
    if args.repair:
        prior = json.loads((args.out.parent / "attempt001/development_feedback.json").read_text())
        if prior.get("completed_paired_contacts") != 14:
            raise SystemExit("repair requires exactly fourteen prior paired contacts")
        for index in indices:
            pair = next((v for v in prior["pairs"] if v["candidate_id"] == f"starting_{index:02d}"), None)
            if pair is None or pair["outcome_contact"] is not False:
                raise SystemExit("repair cannot rescore an already contacted candidate")
    from candidate_predictions import load_candidate_predictions
    from development_score import score_development_panel

    def check():
        if time.time() >= args.deadline_epoch:
            raise TimeoutError("development-scorer deadline reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > int(6.5 * 2**30):
            raise MemoryError("development-scorer peak RSS exceeds 6.5 GiB")

    check()
    report_path = args.out / "development_feedback.json"
    report = {"evidence_stage": "development_only", "held_out_opened": False,
        "job_id": os.environ["SLURM_JOB_ID"], "started_utc": datetime.now(timezone.utc).isoformat(),
        "metric": "one_minus_joint_SSE_over_sum_of_per_neuron_SST_across_scored_trials_and_native_bins",
        "global_selection_objective": "equal_mean_of_two_region_animal_balanced_development_R2",
        "paired_starting_opportunities": len(indices), "completed_paired_contacts": 0,
        "previous_paired_contacts": 14 if args.repair else 0,
        "previous_feedback": "../attempt001/development_feedback.json" if args.repair else None,
        "pairs": [], "complete": False}
    with report_path.open("x") as handle:
        json.dump(report, handle, indent=2)
    def save():
        temporary = report_path.with_suffix(".tmp")
        with temporary.open("w") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, report_path)
    for index in indices:
        check()
        candidate_id = f"starting_{index:02d}"
        panels = []
        result = {"candidate_id": candidate_id, "outcome_contact": False, "streams": {}}
        report["pairs"].append(result)
        try:
            for stream in ("history_assisted", "today_only"):
                panels.append(load_candidate_predictions(args.out / f"{candidate_id}__{stream}.h5",
                    expected_candidate_id=candidate_id, expected_stream=stream, check=check))
        except (FileNotFoundError, ValueError) as error:
            result.update(state="paired_prediction_incomplete_unscored", error_type=type(error).__name__, error=str(error))
        else:
            result["outcome_contact"] = True
            result["state"] = "paired_development_contact_started"
            save()
            for stream, panel in zip(("history_assisted", "today_only"), panels):
                result["streams"][stream] = {"state": "development_contact_started"}
                save()
                feedback = score_development_panel(args.scorer_directory, panel.predictions,
                                                     panel.calibration_means, check=check)
                available = all(value.available for value in feedback)
                result["streams"][stream] = {"regions": [asdict(value) for value in feedback],
                    "complete": available,
                    "global_selection_r2": sum(value.model_r2 for value in feedback) / 2 if available else None}
                save()
            result["state"] = "paired_development_scored"
            report["completed_paired_contacts"] += 1
        save()
    report["complete"] = report["completed_paired_contacts"] == len(indices)
    report["accumulated_starting_paired_contacts"] = report["previous_paired_contacts"] + report["completed_paired_contacts"]
    report["ended_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
