#!/usr/bin/env python3
"""Fixed MC/MV rival diagnostics on the completed MZ replicate zero cohort.

Reuse the saved simulation and M0/MZ fits. This four-fit add-on cannot qualify
family selection, confusion rates, group recovery, or the full behavioral gate.
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
import signal
import time

import numpy as np

from hierarchical_behavior import fit_partially_pooled, load_split, make_data, trial_likelihood
from qualify_assets import require_compute
from run_criterion_recovery import summarize_paired_scores as original_screen
from run_single_process_recovery import DESIGN_FIELDS, RESPONSE_FIELDS, DESIGN_SOURCE, HALVES, IMPLEMENTATION, focal_truth, validate_fit
from summarize_single_process_recovery import summarize_reports


EPISODE = Path(__file__).resolve().parents[2]
SOURCE = EPISODE / "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-replicate000-attempt001"
REPORT_NAME = "single_process_rivals.json"
RIVALS = ("MC", "MV")
SEED = 910031
REPLAY_ABSOLUTE_TOLERANCE = 1e-12  # nats/trial; numerical replay only, never a selection margin.


def load_cohort(path, check=lambda: None):
    """Validate the saved synthetic cohort, including its response channels."""
    people, slots, counts, versions = defaultdict(list), set(), Counter(), {}
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(DESIGN_FIELDS + RESPONSE_FIELDS):
            raise ValueError("Use the exact saved seven-column-design synthetic response table")
        for row in reader:
            check()
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed synthetic row")
            person, version = row["participant"], row["task_version"]
            run, trial = int(row["run"]), int(row["trial_order"])
            slot = person, run, trial
            if not person or slot in slots or run not in (1, 2, 3, 4) or not 1 <= trial <= 64:
                raise ValueError("Invalid or repeated participant/run/trial identity")
            if version not in ("EI", "ER") or versions.setdefault(person, version) != version:
                raise ValueError("Invalid or changing task version")
            if not all(math.isfinite(float(row[key])) for key in ("gain", "loss", "onset")):
                raise ValueError("Nonfinite saved design")
            choice, missing = int(row["choice"]), int(row["no_response"])
            if (choice not in (-1, 0, 1) or missing != int(choice == -1) or int(row["strength"]) != -1
                    or row["response_category"] != {-1: "no_response", 0: "simulated_reject", 1: "simulated_accept"}[choice]):
                raise ValueError("Saved synthetic response identity changed")
            if choice == -1:
                if row["response_time"] != "":
                    raise ValueError("A deadline miss cannot have a response time")
            elif not (math.isfinite(float(row["response_time"])) and 0 < float(row["response_time"]) <= 4):
                raise ValueError("Saved response time is outside the four-second deadline")
            slots.add(slot)
            counts[person, run] += 1
            people[person].append(row)
            if len(slots) > 27648:
                raise ValueError("Synthetic cohort exceeds fixed support")
    if (len(people) != 108 or len(slots) != 27648 or len(counts) != 432
            or set(counts.values()) != {64} or Counter(versions.values()) != {"EI": 54, "ER": 54}):
        raise ValueError("Exact 108-participant/432-run/27648-trial support is required")
    for person, rows in people.items():
        gains = range(10, 41, 2) if versions[person] == "EI" else range(5, 21)
        expected = {(float(gain), float(loss)) for gain in gains for loss in range(5, 21)}
        if {(float(row["gain"]), float(row["loss"])) for row in rows} != expected:
            raise ValueError("Saved cohort changed the full version-specific offer matrix")
    return dict(people)


def validate_source(report, truth, people, spec):
    summary = summarize_reports([("saved_MZ0", report)], "MZ", spec)
    if (summary["valid_replicates"] != 1 or report["replicate_index"] != 0 or report["seed"] != SEED
            or not report.get("finished_utc") or not report.get("started_utc")
            or [report.get(key) for key in ("participants", "runs", "trials")] != [108, 432, 27648]
            or report.get("observed_choices_or_RTs_used") is not False
            or report.get("old_synthetic_responses_used") is not False
            or report.get("design_fields") != list(DESIGN_FIELDS)
            or Path(report.get("design_source", "")).resolve() != DESIGN_SOURCE.resolve()):
        raise ValueError("Only the completed saved MZ replicate zero is selected")
    if (spec.get("policy_version") != 6 or spec.get("state") != "recorded_before_model_scoring"
            or spec["recovery_plan"]["primary_splits"] != [[1, 4], [2, 3]]
            or spec["contaminant"]["fraction"] != .02 or spec["contaminant"]["miss_probability"] != .01):
        raise ValueError("The frozen implementation changed")
    if set(truth) != set(people):
        raise ValueError("Generating truth participant identities changed")
    for params in truth.values():
        if set(params) != {"kappa", "log_a", "nondecision_logit", "eta", "rho", "criterion"}:
            raise ValueError("Generating truth schema changed")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in params.values()):
            raise ValueError("Nonfinite generating truth")
        if any(params[key] != value for key, value in focal_truth("MZ").items()):
            raise ValueError("The saved cohort is not the selected MZ truth")
        for name in ("kappa", "log_a", "nondecision_logit"):
            low, high = spec["parameters"][name]["bounds"]
            if not low <= params[name] <= high:
                raise ValueError("Saved nuisance truth is outside the frozen support")
    if report.get("simulated_deadline_misses") != sum(int(row["no_response"]) for rows in people.values() for row in rows):
        raise ValueError("Saved cohort deadline-miss accounting changed")
    expected = [(model, train, score) for train, score in HALVES for model in ("M0", "MZ")]
    if len(report.get("fits", [])) != len(expected):
        raise ValueError("The saved M0/MZ fit panel is incomplete")
    for fit, (model, train, score) in zip(report["fits"], expected):
        validate_fit(fit, model, train, score, spec, people)
    scores = report.get("participant_scores", {})
    if set(scores) != set(people):
        raise ValueError("Saved score participant identities changed")
    for person, record in scores.items():
        if (set(record) != {"M0", "MZ", "version"} or record["version"] != people[person][0]["task_version"]
                or any(type(record[key]) not in (int, float) or not math.isfinite(record[key]) for key in ("M0", "MZ"))):
            raise ValueError("Saved baseline scores are incomplete or nonfinite")


def score_fit(fit, scoring, spec, check=lambda: None):
    result = {}
    for person, theta in zip(fit["participants"], fit["parameters"]):
        check()
        logp = np.asarray(trial_likelihood(theta, fit["parameter_names"], make_data(scoring[person]["rows"]), spec, "fixed"))
        if logp.shape != (128,) or not np.isfinite(logp).all():
            raise ValueError("Each held-out half requires 128 finite trial scores")
        result[person] = float(logp.sum()) / 256
    return result


def replay_saved_scores(source, splits, spec, check=lambda: None):
    scores = {person: {"M0": 0., "MZ": 0., "version": row["version"]}
              for person, row in source["participant_scores"].items()}
    for position, fit in enumerate(source["fits"]):
        for person, value in score_fit(fit, splits[position // 2][1], spec, check).items():
            scores[person][fit["model"]] += value
    saved = source["participant_scores"]
    if (set(scores) != set(saved) or any(set(scores[person]) != set(saved[person])
            or scores[person]["version"] != saved[person]["version"] for person in scores)):
        raise ValueError("Saved score participant, version or model identities changed")
    discrepancy = max(abs(scores[person][model] - saved[person][model])
                      for person in scores for model in ("M0", "MZ"))
    if not math.isfinite(discrepancy) or discrepancy > REPLAY_ABSOLUTE_TOLERANCE:
        raise ValueError("Saved M0/MZ held-out scores exceed the numerical replay tolerance")
    # Preserve the original recorded baseline even if arithmetic differs in its last bits.
    return {person: dict(row) for person, row in saved.items()}, discrepancy


def rival_diagnostics(scores, model):
    if model not in RIVALS or len(scores) != 108:
        raise ValueError("Only the selected full-participant rival diagnostics are defined")
    if any(not math.isfinite(row[key]) for row in scores.values() for key in ("M0", "MZ", model)):
        raise ValueError("Nonfinite paired diagnostic score")
    mapped = {person: {"M0": row["M0"], "MC": row[model], "version": row["version"]} for person, row in scores.items()}
    screen = original_screen(mapped, SEED + 4099)
    screen.update(fitted_model=model, baseline_model="M0", generating_model="MZ",
                  interpretation="Original necessary screen applied as a diagnostic on this one MZ-generated cohort; not a generating-cell recovery result")
    people = sorted(scores)
    effects = np.array([scores[person]["MZ"] - scores[person][model] for person in people])
    versions = np.array([scores[person]["version"] for person in people])
    return {"rival_vs_saved_M0": screen, "saved_MZ_minus_rival": {
        "contrast": f"MZ-{model}", "mean_gain_nats_per_trial": float(effects.mean()),
        "participant_SE": float(effects.std(ddof=1) / np.sqrt(108)),
        "version_means": {version: float(effects[versions == version].mean()) for version in ("EI", "ER")},
        "participant_median": float(np.median(effects)), "participants": 108,
        "interpretation": "Paired descriptive contrast only; no family-selection threshold or confusion-rate classification"}}


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"Rival diagnostics interrupted by signal {signum}; unfinished work is technical")


def run(args):
    require_compute()
    if args.source_dir.resolve() != SOURCE.resolve() or args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use only the selected saved MZ0 cohort and unchanged implementation")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= 3000:
        raise ValueError("The whole-process internal cap is at most 3000 seconds")
    if any(os.environ.get(name) != "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("All four numerical thread limits must be explicitly one")
    output = args.output_dir.resolve()
    if output.exists() or SOURCE.resolve() == output or SOURCE.resolve() in output.parents:
        raise FileExistsError("Use a fresh destination outside the preserved MZ0 cohort")
    output.mkdir(parents=True)
    started = time.perf_counter()
    deadline = time.monotonic() + args.max_seconds
    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process deadline exceeded")
    report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": "MZ0_MC_MV_rival_diagnostics",
        "generating_model": "MZ", "replicate_index": 0, "seed": SEED, "source_dir": str(args.source_dir),
        "job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat(),
        "internal_seconds": args.max_seconds, "status": "running", "rival_diagnostics_complete": False,
        "simulation_regenerated": False, "saved_M0_MZ_refitted": False, "saved_scores_replayed": False,
        "family_selection": "not_evaluated", "confusion_rate": "not_evaluated", "group_recovery": "not_evaluated",
        "full_gate": "not_evaluated", "full_recovery_pass": None, "episode_terminal": False,
        "fits": [], "completed_rivals": [], "diagnostics": {}}
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    succeeded = False
    try:
        write_report(output, report)
        spec = json.loads(args.implementation.read_text())
        source = json.loads((args.source_dir / "single_process_recovery.json").read_text())
        truth = json.loads((args.source_dir / "generating_parameters.json").read_text())
        events = args.source_dir / "simulated_events.csv.gz"
        people = load_cohort(events, check)
        validate_source(source, truth, people, spec)
        splits = [load_split(events, list(train), list(score)) for train, score in HALVES]
        scores, replay_discrepancy = replay_saved_scores(source, splits, spec, check)
        report.update(implementation=spec, source_finished_utc=source["finished_utc"], saved_scores_replayed=True,
                      saved_score_replay_max_absolute_discrepancy=replay_discrepancy,
                      saved_score_replay_absolute_tolerance=REPLAY_ABSOLUTE_TOLERANCE,
                      participants=108, runs=432, trials=27648)
        write_report(output, report)
        for model in RIVALS:
            for person in scores:
                scores[person][model] = 0.
            for (train, held), (training, scoring) in zip(HALVES, splits):
                check()
                fit_started = time.perf_counter()
                fit = fit_partially_pooled(training, model, "fixed", spec)
                fit.update(training_runs=list(train), scoring_runs=list(held), elapsed_seconds=time.perf_counter() - fit_started)
                json.dumps(fit, allow_nan=False)
                report["fits"].append(fit)
                write_report(output, report)
                check()
                validate_fit(fit, model, train, held, spec, people)
                for person, value in score_fit(fit, scoring, spec, check).items():
                    scores[person][model] += value
                check()
                print(json.dumps({"model": model, "training_runs": list(train), "fit_completed": True,
                                  "elapsed_seconds": fit["elapsed_seconds"]}), flush=True)
            diagnostics = rival_diagnostics(scores, model)
            check()
            report["diagnostics"][model] = diagnostics
            report["completed_rivals"].append(model)
            report["participant_scores"] = {person: {key: row[key] for key in ("version", "M0", "MZ", *report["completed_rivals"])}
                                            for person, row in scores.items()}
            write_report(output, report)
        check()
        succeeded = True
    except Exception as error:
        report.update(status="technical_rival_failure", rival_diagnostics_complete=False,
                      error=f"{type(error).__name__}: {error}")
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if succeeded and time.monotonic() >= deadline:
            succeeded = False
            report.update(status="technical_rival_failure", error="RuntimeError: Internal deadline reached during finalization")
        report.update(status="rival_diagnostics_completed" if succeeded else report["status"],
                      rival_diagnostics_complete=succeeded, elapsed_seconds=time.perf_counter() - started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_report(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=3000)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("status", "rival_diagnostics_complete", "completed_rivals", "diagnostics", "elapsed_seconds", "error")
                      if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["rival_diagnostics_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
