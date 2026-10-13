#!/usr/bin/env python3
"""One bounded EP07 adaptive pair worker; no scorer or held-out input.

The coordinator invokes index0..13 serially and owns scoring and stopping.
Indices0..6 belong to adaptive001; indices7..13 append under adaptive002.
Before fitting, the complete preceding feedback is replayed through the fixed
controller, preserving the original opportunities, incumbents and patience.
This worker never launches successors, retries a fit, substitutes a setting,
or interprets search completion. Both streams use one declared global recipe,
fitting independent state. A failed stream retains an incomplete HDF file;
its paired stream may finish, but only the coordinator can admit a complete
pair for development scoring. Ensemble members are read from their original
starting/adaptive directories with their completed permitted feedback.

Only candidate fit/development-feature packs are opened. Published predictions
contain no development responses. New artifacts use exclusive creation; the
worker updates only the status file it claimed, and never replaces a prior
prediction. Source-weight diagnostics are training/calibration quantities.
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


STREAMS = ("history_assisted", "today_only")


def utc():
    return datetime.now(timezone.utc).isoformat()


def _json_array(value):
    # The only non-JSON values emitted below are NumPy diagnostic arrays/scalars.
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"unsupported worker metadata type: {type(value).__name__}")


def _prior_feedback(base, index, check):
    """Read only the fixed earlier scopes; retain each member's original path."""
    starts, previous, members = [], [], {}
    for attempt, indices in (("attempt001", range(16)), ("attempt002", (8, 10))):
        check()
        directory = base / attempt
        report = json.loads((directory / "development_feedback.json").read_text())
        records = report.get("pairs", [])
        expected = {f"starting_{value:02d}" for value in indices}
        if (report.get("evidence_stage") != "development_only"
                or report.get("held_out_opened") is not False or not report.get("ended_utc")
                or len(records) != len(expected)
                or {value.get("candidate_id") for value in records} != expected):
            raise ValueError("starting report identity or closed development scope differs")
        starts.extend(records)
        members.update((value["candidate_id"], (value, directory)) for value in records)
    for prior_index in range(index):
        check()
        directory = base / ("adaptive001" if prior_index < 7 else "adaptive002")
        name = f"adaptive_{prior_index:02d}"
        report = json.loads((directory / f"{name}_feedback.json").read_text())
        if (report.get("evidence_stage") != "development_only"
                or report.get("held_out_opened") is not False or not report.get("ended_utc")
                or report.get("pair", {}).get("candidate_id") != name):
            raise ValueError("prior successor identity or closed development scope differs")
        previous.append(report["pair"])
        members[name] = report["pair"], directory
    return starts, previous, members


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True, choices=range(14))
    parser.add_argument("--candidate-directory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    if (not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname()
            or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) not in (1, 2)
            or os.environ.get("SLURM_MEM_PER_NODE") != "7168"):
        raise SystemExit("adaptive worker requires the selected1–2CPU/7GiB compute allocation")
    expected = (Path(os.environ["SCRATCH"]) /
        "br_autoresearch/episode07_lfp_session_transfer/source_support_attempt002/candidate")
    if args.candidate_directory.resolve() != expected.resolve():
        raise SystemExit("worker accepts only the published candidate directory")
    episode = Path(__file__).resolve().parents[2]
    run_name = "adaptive001" if args.index < 7 else "adaptive002"
    selected_out = episode / "outputs/runs/matched_search" / run_name
    if args.out.resolve() != selected_out or not args.out.is_dir():
        raise SystemExit("worker requires the selected index's adaptive output directory")
    remaining = args.deadline_epoch - time.time()
    if not math.isfinite(remaining) or not 0 < remaining <= 3600:
        raise SystemExit("worker requires a future deadline at most one hour away")
    # Set all numerical thread budgets before importing any numerical package.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"

    def check():
        if time.time() >= args.deadline_epoch:
            raise TimeoutError("adaptive-worker deadline reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > int(6.5 * 2**30):
            raise MemoryError("adaptive-worker peak RSS exceeds6.5GiB")

    check()
    from adaptive_progress import assess_adaptive_progress
    starts, previous, member_records = _prior_feedback(selected_out.parent, args.index, check)
    progress = assess_adaptive_progress(starts, previous)
    if progress.stop_reason is not None:
        raise SystemExit("adaptive stage cannot continue past its recorded stop condition")
    candidate_id = f"adaptive_{args.index:02d}"
    if ((args.out / f"{candidate_id}_feedback.json").exists()
            or any((args.out / f"{candidate_id}__{stream}.h5").exists() for stream in STREAMS)):
        raise SystemExit("this adaptive opportunity already has output; no retry is selected")
    valid_members = {value.candidate_id for value in progress.starting_records + progress.successor_records
                     if value.status == "valid_pair"}
    # No numerical imports or payload reads before the actual allocation guard.
    from adaptive_recipes import adaptive_pair
    from candidate_predictions import CandidatePredictionWriter, load_candidate_predictions
    from native_trials import (NEW, load_native_trials, fit_calibration_mean,
                               predict_calibration_mean)
    from primary_model import fit_primary_model, predict_primary_development
    from weighted_primary_model import (fit_weighted_primary_model,
                                       predict_weighted_primary_development)
    from ensemble_predictions import compose_ensemble
    from view_access import CandidateViews

    pair = adaptive_pair(args.index)
    if any(value.primary is not None and value.primary.regression.family == "shallow_network"
           for value in pair.streams):
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    check()
    status = {"index": args.index, "candidate_id": pair.candidate_id, "label": pair.label,
        "candidate_pairs": [args.index], "model_attempt": 1,
        "started_utc": utc(), "job_id": os.environ["SLURM_JOB_ID"],
        "state": "running", "outcome_contact": False,
        "progress_before": asdict(progress),
        "whole_search_complete": False, "authorizes_held_out": False, "models": {}}
    status_path = args.out / f"worker_{args.index}.json"
    # Keep the exclusively claimed handle: updates never open another file.
    with status_path.open("x") as status_handle:
        def record():
            status["updated_utc"] = utc()
            status_handle.seek(0)
            json.dump(status, status_handle, indent=2, allow_nan=False, default=_json_array)
            status_handle.write("\n")
            status_handle.truncate()
            status_handle.flush()

        record()
        overall_failure = False
        try:
            views = (CandidateViews(args.candidate_directory)
                     if any(value.kind != "ensemble" for value in pair.streams) else None)
            for stream, declaration in zip(STREAMS, pair.streams):
                check()
                key = pair.candidate_id + "__" + stream
                model = {"recipe": asdict(declaration), "completed_settings": 0,
                    "state": "composing" if declaration.kind == "ensemble" else "fitting",
                    "outcome_contact": False, "settings": []}
                status["models"][key] = model
                record()
                try:
                    with CandidatePredictionWriter(args.out / (key + ".h5"),
                            candidate_id=pair.candidate_id, stream=stream, check=check) as writer:
                        if declaration.kind == "ensemble":
                            check()
                            if any(name not in valid_members for name in declaration.member_ids):
                                raise ValueError("each declared member requires a completed available paired evaluation")
                            records = tuple(member_records[name][0] for name in declaration.member_ids)
                            members = tuple(load_candidate_predictions(
                                member_records[name][1] / (name + "__" + stream + ".h5"),
                                expected_candidate_id=name, expected_stream=stream, check=check)
                                for name in declaration.member_ids)
                            composed = compose_ensemble(members, candidate_id=pair.candidate_id,
                                stream=stream, member_ids=declaration.member_ids,
                                feedback_records=tuple(records), weights=declaration.member_weights, check=check)
                            for prediction, baseline in zip(composed.predictions, composed.calibration_means):
                                writer.write(prediction, baseline)
                                model["completed_settings"] += 1
                                record()
                            del members, composed, records
                        else:
                            for animal in ("Mihili", "Chewie-L"):
                                for target_day in NEW[animal]:
                                    for region in ("M1", "PMd"):
                                        check()
                                        scope = dict(animal=animal, target_day=target_day, region=region, stream=stream)
                                        model["current_setting"] = [animal, target_day, region]
                                        record()
                                        started = time.monotonic()
                                        fit_days = development = baseline = fitted = prediction = None
                                        try:
                                            fit_days = load_native_trials(views, view_role="fit", check=check, **scope)
                                            development = load_native_trials(views, view_role="development_features",
                                                                             check=check, **scope)[0]
                                            baseline = predict_calibration_mean(
                                                fit_calibration_mean(fit_days[-1], check=check), development, check=check)
                                            if declaration.kind == "weighted_primary":
                                                fitted = fit_weighted_primary_model(fit_days, recipe=declaration.primary,
                                                    weight_recipe=declaration.source_weighting, check=check)
                                                prediction = predict_weighted_primary_development(fitted, development, check=check)
                                                diagnostics = {"effective_recipe": asdict(fitted.effective_recipe),
                                                               "source_weights": asdict(fitted.source_weights)}
                                            else:
                                                fitted = fit_primary_model(fit_days, recipe=declaration.primary, check=check)
                                                prediction = predict_primary_development(fitted, development, check=check)
                                                diagnostics = {}
                                            writer.write(prediction, baseline)
                                            model["completed_settings"] += 1
                                            model["settings"].append({"setting": [animal, target_day, region],
                                                "seconds": time.monotonic() - started, **diagnostics})
                                        finally:
                                            fit_days = development = baseline = fitted = prediction = None
                                        record()
                                        print(f"candidate={key} setting={animal}/{target_day}/{region} complete", flush=True)
                except (ValueError, FloatingPointError, RuntimeError) as error:
                    model.update(state="failed", error_type=type(error).__name__, error=str(error))
                    overall_failure = True
                else:
                    model["state"] = "predictions_complete_unscored"
                record()
        except BaseException as error:
            status.update(state="stopped", error_type=type(error).__name__, error=str(error), ended_utc=utc())
            record()
            raise
        status.update(state="closed_with_candidate_failures" if overall_failure else "predictions_complete_unscored",
            ended_utc=utc(), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        record()
    return 1 if overall_failure else 0


if __name__ == "__main__":
    sys.exit(main())
