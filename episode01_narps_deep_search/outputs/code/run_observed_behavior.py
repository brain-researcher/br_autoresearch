#!/usr/bin/env python3
"""Frozen exploratory observed behavior: descriptions or one independent fit.

No synthetic recovery, model selection, retries, parameter tuning, or logistic
benchmark. Fitting receives only the selected training rows. A completed half
is a score contribution; comparison requires both halves and all 108 people.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import time

EPISODE = Path(__file__).resolve().parents[2]
SOURCE = EPISODE / "outputs/runs/public_asset_qualification/20261003-attempt01/original_events.csv.gz"
IMPLEMENTATION = Path(__file__).with_name("behavioral_implementation.json")
MODELS = ("M0", "MV", "MZ", "MC", "MVZ", "MVC")
HALVES = (((1, 4), (2, 3)), ((2, 3), (1, 4)))
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
CATEGORIES = {"strong_accept": (1, 1), "weak_accept": (1, 0),
              "weak_reject": (0, 0), "strong_reject": (0, 1), "no_response": (-1, -1)}
FIELDS = ("participant", "task_version", "run", "trial_order", "onset", "gain", "loss",
          "response_category", "choice", "strength", "response_time", "no_response", "sequence_template")
SCORE_FIELDS = ("participant", "task_version", "run", "trial_order", "log_density")
REPORTS = {"describe": "observed_behavior_descriptive.json", "fit": "observed_behavior_fit.json"}


def require_environment(command):
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Observed behavior requires an actual compute-node Slurm step")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if cpus < 1 or (command == "fit" and cpus != 4):
        raise RuntimeError("Fit requires exactly four allocated CPUs; descriptions require one or more")
    if any(os.environ.get(name) != "1" for name in THREAD_ENV):
        raise RuntimeError("Set all six numerical thread variables to 1 before imports")
    return {"hostname": socket.gethostname(), "job_id": os.environ["SLURM_JOB_ID"],
            "allocated_cpus": cpus, "thread_environment": {name: os.environ[name] for name in THREAD_ENV}}


def load_helpers():
    # Import only after checking the actual environment, including legacy modules
    # that set thread environment variables when imported.
    import numpy as np
    import hierarchical_behavior as behavior
    import hierarchical_behavior_parallel as parallel
    from run_single_process_recovery import validate_fit
    return np, behavior, parallel, validate_fit


def finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite source value")
    return number


def validate_rows(rows, check=lambda: None):
    """Check the qualified table's recorded mapping and full support, without recoding."""
    people, seen = {}, set()
    for row in rows:
        check()
        person, version = row["participant"], row["task_version"]
        if not re.fullmatch(r"sub-[A-Za-z0-9]+", person) or version not in ("EI", "ER"):
            raise ValueError("Invalid participant or task version")
        run, trial = int(row["run"]), int(row["trial_order"])
        key = person, run, trial
        if key in seen or run not in range(1, 5) or trial not in range(1, 65):
            raise ValueError("Duplicate or invalid participant/run/trial")
        seen.add(key)
        expected = CATEGORIES.get(row["response_category"])
        if expected != (int(row["choice"]), int(row["strength"])):
            raise ValueError("Original five-category choice/strength mapping changed")
        missed = expected[0] == -1
        if int(row["no_response"]) != int(missed):
            raise ValueError("Miss indicator disagrees with response category")
        if missed:
            if row["response_time"] != "":
                raise ValueError("Misses must retain the empty RT sentinel")
        elif not 0 < finite(row["response_time"]) <= 4:
            raise ValueError("Responded RT outside the original four-second deadline")
        if finite(row["onset"]) < 0 or not row["sequence_template"]:
            raise ValueError("Missing qualified onset/template field")
        finite(row["gain"]), finite(row["loss"])
        panel = people.setdefault(person, {"version": version, "rows": []})
        if panel["version"] != version:
            raise ValueError("Participant task version changes")
        panel["rows"].append(row)
    if len(rows) != 27648 or len(people) != 108 or Counter(p["version"] for p in people.values()) != {"EI": 54, "ER": 54}:
        raise ValueError("Require all 27648 trials and 108 participants, 54 EI / 54 ER")
    for panel in people.values():
        check()
        panel["rows"].sort(key=lambda r: (int(r["run"]), int(r["trial_order"])))
        gains = range(10, 41, 2) if panel["version"] == "EI" else range(5, 21)
        grid = {(float(g), float(l)) for g in gains for l in range(5, 21)}
        if len(panel["rows"]) != 256 or {(finite(r["gain"]), finite(r["loss"])) for r in panel["rows"]} != grid:
            raise ValueError("Original full gain/loss grid changed")
        for run in range(1, 5):
            part = [r for r in panel["rows"] if int(r["run"]) == run]
            if len(part) != 64 or any(finite(b["onset"]) <= finite(a["onset"]) for a, b in zip(part, part[1:])):
                raise ValueError("Each run requires 64 ordered trials with increasing onsets")
    return {person: people[person] for person in sorted(people)}


def read_source(check):
    with gzip.open(SOURCE, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Qualified original-event schema changed")
        rows = []
        for row in reader:
            check()
            rows.append(row)
    return validate_rows(rows, check)


def split_people(people, direction):
    train_runs, held_runs = HALVES[direction]
    training, held = {}, {}
    for person, panel in people.items():
        training[person] = {"version": panel["version"], "rows": [r for r in panel["rows"] if int(r["run"]) in train_runs]}
        held[person] = {"version": panel["version"], "rows": [r for r in panel["rows"] if int(r["run"]) in held_runs]}
        if len(training[person]["rows"]) != 128 or len(held[person]["rows"]) != 128:
            raise ValueError("Primary split must retain 128 training and 128 held-out trials")
    return training, held


def describe_rows(rows, np):
    categories = Counter(r["response_category"] for r in rows)
    n = len(rows)
    accept = sum(int(r["choice"]) == 1 for r in rows)
    reject = sum(int(r["choice"]) == 0 for r in rows)
    missed = categories["no_response"]
    quantiles = {}
    for label, choice in (("accept", 1), ("reject", 0)):
        rt = [float(r["response_time"]) for r in rows if int(r["choice"]) == choice]
        quantiles[label] = {"responded_trials": len(rt), "seconds_q10_q50_q90":
                           np.quantile(rt, [.1, .5, .9], method="linear").tolist() if rt else None}
    return {"trials": n, "category_counts": {key: categories[key] for key in CATEGORIES},
            "accept_count": accept, "reject_count": reject, "miss_count": missed,
            "accept_fraction_all_trials": accept/n, "reject_fraction_all_trials": reject/n,
            "miss_fraction_all_trials": missed/n,
            "accept_fraction_responded": accept/(accept+reject) if accept+reject else None,
            "responded_rt_by_choice": quantiles}


def describe(people, np, check):
    versions, grid, participant, participant_run = defaultdict(list), defaultdict(list), {}, {}
    for person, panel in people.items():
        check()
        participant[person] = {"version": panel["version"], **describe_rows(panel["rows"], np)}
        participant_run[person] = {str(run): describe_rows([r for r in panel["rows"] if int(r["run"]) == run], np)
                                   for run in range(1, 5)}
        for row in panel["rows"]:
            versions[panel["version"]].append(row)
            grid[(panel["version"], float(row["gain"]), float(row["loss"]))].append(row)
    grid_rows = []
    for (version, gain, loss), rows in sorted(grid.items()):
        check()
        grid_rows.append({"version": version, "gain": gain, "loss": loss, **describe_rows(rows, np)})
    return {"overall": describe_rows(versions["EI"] + versions["ER"], np),
            "by_version": {v: describe_rows(rows, np) for v, rows in versions.items()},
            "by_participant": participant, "by_participant_run": participant_run,
            "by_version_gain_loss": grid_rows, "rt_quantile_method": "linear",
            "strong_weak_interpretation": "Recorded response category, not independent confidence"}


def atomic_report(output, name, report):
    temporary = output / (name + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / name)


def score_fit(fit, held, spec, helpers, check, report, checkpoint, output):
    np, behavior, _, _ = helpers
    scores, trial_rows = {}, []
    for person, theta in zip(fit["participants"], fit["parameters"]):
        check()
        panel = held[person]
        logp = behavior.trial_likelihood(np.asarray(theta), fit["parameter_names"],
                                        behavior.make_data(panel["rows"]), spec, "fixed")
        check()
        if np.shape(logp) != (128,) or not np.isfinite(logp).all():
            raise ValueError("Held-out scores require all 128 finite trial log densities")
        total = float(logp.sum())
        scores[person] = {"version": panel["version"], "heldout_trials": 128,
                          "log_sum": total, "contribution_sum_over_256": total/256}
        trial_rows.extend({"participant": person, "task_version": panel["version"],
                           "run": int(row["run"]), "trial_order": int(row["trial_order"]),
                           "log_density": float(value)} for row, value in zip(panel["rows"], logp))
        report["scored_participants"] = len(scores)
        if len(scores) % 12 == 0:
            checkpoint()
    path = output / "heldout_trial_scores.csv.gz"
    temporary = path.with_name(path.name + ".partial")
    with gzip.open(temporary, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        for row in trial_rows:
            check()
            writer.writerow(row)
    check()
    temporary.replace(path)
    report["trial_score_artifact"] = str(path)
    check()
    return scores


def run(args):
    environment = require_environment(args.command)
    cap = 120 if args.command == "describe" else 1800
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= cap:
        raise ValueError(f"Whole-process cap must be 1 through {cap} seconds")
    if args.command == "fit" and (args.model not in MODELS or args.direction not in (0, 1)
                                 or args.implementation.resolve() != IMPLEMENTATION.resolve()):
        raise ValueError("Use a frozen model, primary direction and original implementation")
    output = args.output_dir.resolve()
    if output.exists() or EPISODE / "outputs/runs" not in output.parents or SOURCE.parent in output.parents:
        raise FileExistsError("Use a fresh episode outputs/runs directory outside the qualified source")
    started = time.monotonic()
    deadline = started + args.max_seconds
    output.mkdir(parents=True)
    name = REPORTS[args.command]
    report = {"analysis_id": "exploratory_observed_behavior", "phase": "observed_behavior_" + args.command,
              **environment, "source": str(SOURCE), "source_validated": False,
              "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "internal_seconds": args.max_seconds, "observed_choices_or_RTs_used": True,
              "logistic_benchmark": "registered_but_not_implemented_in_this_report",
              "interpretation": "Exploratory within-session prediction; no psychological winner or confirmatory claim",
              "recovery_gate_passed": False, "full_behavioral_gate": "not_evaluated",
              "model_comparison": "not_evaluated_requires_both_directions_all_108",
              "automatic_retry": False, "tuning_performed": False, "direction_complete": False,
              "scores_available": False, "fit": None, "fresh_fit_calls": 0,
              "workers_cleaned_up": True, "scored_participants": 0}
    if args.command == "fit":
        report.update(model=args.model, rival="fixed", direction=args.direction,
                      training_runs=list(HALVES[args.direction][0]), scoring_runs=list(HALVES[args.direction][1]),
                      workers=4, score_normalization="direction_log_sum/256; sum both directions per participant")
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Observed-behavior whole-process deadline exceeded")
    def checkpoint():
        report["process_elapsed_seconds"] = time.monotonic() - started
        atomic_report(output, name, report)
    def interrupted(signum, frame):
        raise TimeoutError(f"Observed-behavior process interrupted by signal {signum}")
    def failed(error):
        report.update(status="technical_failure", direction_complete=False, scores_available=False,
                      error=f"{type(error).__name__}: {error}")
        report.pop("participant_scores", None)
        report.pop("descriptions", None)
        progress = getattr(error, "parallel_progress", None)
        if progress is not None:
            report["parallel_failure_progress"] = progress
            report["workers_cleaned_up"] = progress.get("workers_cleaned_up", False)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    success = False
    for sig in previous:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    try:
        checkpoint()
        check()
        helpers = load_helpers()
        check()
        people = read_source(check)
        report.update(source_validated=True, participants=list(people), participants_count=108,
                      version_counts={"EI": 54, "ER": 54}, source_trials=27648)
        checkpoint()
        if args.command == "describe":
            report["descriptions"] = describe(people, helpers[0], check)
            report["status"] = "descriptions_completed"
        else:
            spec = json.loads(args.implementation.read_text())
            if (spec.get("policy_version") != 6 or spec.get("state") != "recorded_before_model_scoring"
                    or spec["optimizer"]["maximum_sweeps"] != 500
                    or spec["recovery_plan"]["primary_splits"] != [[1, 4], [2, 3]]):
                raise ValueError("Original implementation or 500-sweep primary split changed")
            report["implementation"] = spec
            training, held = split_people(people, args.direction)
            report.update(stage="fitting")
            checkpoint()
            check()
            remaining = deadline - time.monotonic() - (helpers[2].CLEANUP_SECONDS + 1)
            if remaining <= 0:
                raise TimeoutError("No fit time remains after the child-cleanup reserve")
            def progress(event):
                check()
                if event.get("stage") == "sweep":
                    report["fit_progress"] = event
                    checkpoint()
            began = time.monotonic()
            report.update(fresh_fit_calls=1, workers_cleaned_up=False)
            fit = helpers[2].fit_partially_pooled_parallel(training, args.model, "fixed", spec,
                                                          workers=4, max_seconds=remaining, progress=progress)
            report["workers_cleaned_up"] = True
            fit.update(training_runs=list(HALVES[args.direction][0]), scoring_runs=list(HALVES[args.direction][1]),
                       elapsed_seconds=time.monotonic()-began)
            report["fit"] = fit
            checkpoint()  # Preserve every returned fit, including nonconvergence.
            check()
            helpers[3](fit, args.model, *HALVES[args.direction], spec, people)
            report["stage"] = "scoring"
            scores = score_fit(fit, held, spec, helpers, check, report, checkpoint, output)
            report.update(participant_scores=scores, direction_complete=True, scores_available=True,
                          scored_trials=13824, status="direction_completed")
        check()
        success = True
    except Exception as error:
        failed(error)
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    # Restoration and final publication are inside the declared whole-process
    # accounting. A late expiry must retract success even if scores exist.
    try:
        check()
    except Exception as error:
        failed(error)
        success = False
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint()
    if success:
        try:
            check()
        except Exception as error:
            failed(error)
            success = False
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            checkpoint()
    return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    desc = sub.add_parser("describe")
    fit = sub.add_parser("fit")
    for command, default in ((desc, 120), (fit, 1800)):
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--max-seconds", type=int, default=default)
    fit.add_argument("--model", choices=MODELS, required=True)
    fit.add_argument("--direction", type=int, choices=(0, 1), required=True)
    fit.add_argument("--implementation", type=Path, default=IMPLEMENTATION)
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
