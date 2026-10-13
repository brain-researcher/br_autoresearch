#!/usr/bin/env python3
"""Bounded EP07 starting-model worker; candidate data only, no scorer import.

Each of four workers owns four predefined pairs. It loads one permitted
animal/day/region/stream pack at a time and fits its four candidates, publishing
complete ten-setting prediction panels. Fitted state never crosses streams.
Scientific/numerical failures preserve the incomplete artifact and do not
trigger another rank, omitted day, retry, recipe or outcome opening.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--repair", action="store_true",
                        help="second/final attempts of unscored starting08/10 only")
    parser.add_argument("--candidate-directory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    workers = 2 if args.repair else 4
    allowed_cpus = range(2, 4) if args.repair else range(4, 7)
    memory = "14336" if args.repair else "28672"
    if (not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname()
            or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) not in allowed_cpus
            or os.environ.get("SLURM_MEM_PER_NODE") != memory
            or args.worker not in range(workers)):
        raise SystemExit("worker requires the selected worker count and compute allocation")
    expected = Path(os.environ["SCRATCH"]) / "br_autoresearch/episode07_lfp_session_transfer/source_support_attempt002/candidate"
    if args.candidate_directory.resolve() != expected.resolve():
        raise SystemExit("worker accepts only the published candidate directory")
    episode = Path(__file__).resolve().parents[2]
    attempt = "attempt002" if args.repair else "attempt001"
    if args.out.resolve() != episode / "outputs/runs/matched_search" / attempt:
        raise SystemExit("worker output differs from the selected attempt")
    if not args.out.is_dir() or args.deadline_epoch <= time.time():
        raise SystemExit("selected attempt or usable deadline is absent")
    if args.repair:
        prior = args.out.parent / "attempt001"
        feedback = json.loads((prior / "development_feedback.json").read_text())
        if feedback.get("completed_paired_contacts") != 14:
            raise SystemExit("repair requires the recorded fourteen complete starting pairs")
        failed_id = f"starting_{(8, 10)[args.worker]:02d}"
        pair = next((v for v in feedback["pairs"] if v["candidate_id"] == failed_id), None)
        if (pair is None or pair["state"] != "paired_prediction_incomplete_unscored"
                or pair["outcome_contact"] is not False):
            raise SystemExit("selected repair candidate must have no prior outcome contact")
        old_worker = json.loads((prior / f"worker_{(0, 2)[args.worker]}.json").read_text())
        for stream in ("history_assisted", "today_only"):
            record = old_worker["models"][failed_id + "__" + stream]
            if record["state"] != "failed" or record.get("error_type") != "RegressionNotConverged":
                raise SystemExit("repair requires the diagnosed elastic-net failure in both streams")
    # No scientific imports or payloads before the actual allocation guard.
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from candidate_predictions import CandidatePredictionWriter
    from native_trials import (NEW, load_native_trials, fit_calibration_mean,
                               predict_calibration_mean)
    from primary_model import fit_primary_model, predict_primary_development
    from starting_recipes import starting_pair
    from view_access import CandidateViews

    def check():
        if time.time() >= args.deadline_epoch:
            raise TimeoutError("starting-worker deadline reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > int(6.5 * 2**30):
            raise MemoryError("starting-worker peak RSS exceeds 6.5 GiB")

    check()
    indices = ((8, 10)[args.worker],) if args.repair else tuple(range(args.worker, 16, 4))
    def recipe(index, stream_index):
        chosen = starting_pair(index)[stream_index]
        if args.repair:
            chosen = replace(chosen, regression=replace(chosen.regression, max_iterations=100000))
        return chosen
    status_path = args.out / f"worker_{args.worker}.json"
    status = {"worker": args.worker, "started_utc": utc(), "job_id": os.environ["SLURM_JOB_ID"],
              "candidate_pairs": list(indices), "model_attempt": 2 if args.repair else 1,
              "state": "running", "models": {}}
    with status_path.open("x") as handle:
        json.dump(status, handle, indent=2)

    def record():
        status["updated_utc"] = utc()
        with status_path.open("w") as handle:
            json.dump(status, handle, indent=2, allow_nan=False)
            handle.write("\n")

    views = CandidateViews(args.candidate_directory)
    overall_failure = False
    for stream_index, stream in enumerate(("history_assisted", "today_only")):
        writers = {}
        try:
            for index in indices:
                candidate_id = f"starting_{index:02d}"
                key = candidate_id + "__" + stream
                chosen = recipe(index, stream_index)
                writer = CandidatePredictionWriter(args.out / (key + ".h5"),
                    candidate_id=candidate_id, stream=stream, check=check)
                writer.__enter__()
                writers[index] = writer
                status["models"][key] = {"recipe": asdict(chosen), "completed_settings": 0,
                    "state": "fitting", "outcome_contact": False}
            record()
            for animal, dates in NEW.items():
                for target_day in dates:
                    for region in ("M1", "PMd"):
                        if not writers:
                            break
                        check()
                        scope = dict(animal=animal, target_day=target_day, region=region, stream=stream)
                        fit_days = load_native_trials(views, view_role="fit", check=check, **scope)
                        development = load_native_trials(views, view_role="development_features",
                                                         check=check, **scope)[0]
                        baseline = predict_calibration_mean(
                            fit_calibration_mean(fit_days[-1], check=check), development, check=check)
                        for index in tuple(writers):
                            key = f"starting_{index:02d}__{stream}"
                            stage_start = time.monotonic()
                            status["models"][key]["current_setting"] = [animal, target_day, region]
                            record()
                            fitted = prediction = None
                            try:
                                fitted = fit_primary_model(fit_days,
                                    recipe=recipe(index, stream_index), check=check)
                                prediction = predict_primary_development(fitted, development, check=check)
                                writers[index].write(prediction, baseline)
                            except (ValueError, FloatingPointError, RuntimeError) as error:
                                # A timeout or memory ceiling stops the whole worker below.
                                if isinstance(error, (TimeoutError, MemoryError)):
                                    raise
                                writers.pop(index).__exit__(type(error), error, error.__traceback__)
                                status["models"][key].update(state="failed", error_type=type(error).__name__,
                                                              error=str(error))
                                overall_failure = True
                            else:
                                status["models"][key]["completed_settings"] += 1
                            finally:
                                fitted = prediction = None
                            status["models"][key]["last_setting_seconds"] = time.monotonic() - stage_start
                            record()
                            print(f"worker={args.worker} candidate={key} setting={animal}/{target_day}/{region} state={status['models'][key]['state']}", flush=True)
                        del fit_days, development, baseline
            for index in tuple(writers):
                writer = writers.pop(index)
                writer.__exit__(None, None, None)
                status["models"][f"starting_{index:02d}__{stream}"]["state"] = "predictions_complete_unscored"
                record()
        except BaseException as error:
            for writer in writers.values():
                writer.__exit__(type(error), error, error.__traceback__)
            status.update(state="stopped", error_type=type(error).__name__, error=str(error), ended_utc=utc())
            record()
            raise
    status.update(state="closed_with_candidate_failures" if overall_failure else "predictions_complete_unscored",
                  ended_utc=utc(), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    record()
    return 1 if overall_failure else 0


if __name__ == "__main__":
    sys.exit(main())
