#!/usr/bin/env python3
"""Fit one already-declared MVZ/MVC rival in both halves of the saved MZ0 cohort."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np

from hierarchical_behavior import fit_partially_pooled, load_split
from qualify_assets import require_compute
from run_single_process_recovery import HALVES, IMPLEMENTATION, validate_fit
import run_single_process_rivals as shared


SOURCE = shared.SOURCE
MODELS = ("MVZ", "MVC")
REPORT_NAME = "mz_complex_rival.json"


def paired_diagnostics(scores, model):
    """Descriptive paired arithmetic; no selection rule or new margin."""
    if model not in MODELS or len(scores) != 108:
        raise ValueError("Select one declared complex rival with all 108 participants")
    if Counter(row["version"] for row in scores.values()) != {"EI": 54, "ER": 54}:
        raise ValueError("Both unchanged 54-person task versions are required")
    if any(not math.isfinite(row[key]) for row in scores.values() for key in ("M0", "MZ", model)):
        raise ValueError("Paired scores must all be finite")
    people = sorted(scores)
    versions = np.array([scores[person]["version"] for person in people])
    result = {}
    for baseline in ("M0", "MZ"):
        effects = np.array([scores[person][model] - scores[person][baseline] for person in people])
        result[f"model_minus_saved_{baseline}"] = {
            "contrast": f"{model}-{baseline}", "mean_gain_nats_per_trial": float(effects.mean()),
            "participant_SE": float(effects.std(ddof=1) / np.sqrt(108)),
            "version_means": {version: float(effects[versions == version].mean()) for version in ("EI", "ER")},
            "participant_median": float(np.median(effects)), "participants": 108,
            "nested_comparison": baseline == "M0" or model == "MVZ",
            "interpretation": "Paired descriptive contrast only; full declared-menu selection remains unevaluated"}
    return result


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"Complex-rival work interrupted by signal {signum}; unfinished work is technical")


def run(args):
    require_compute()
    if args.model not in MODELS:
        raise ValueError("Only one MVZ or MVC model is selected per invocation")
    if args.source_dir.resolve() != SOURCE.resolve() or args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use only the selected completed MZ0 cohort and unchanged implementation")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= 5400:
        raise ValueError("The whole-process internal cap is at most 5400 seconds")
    if any(os.environ.get(name) != "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("All four numerical thread limits must be explicitly one")
    output = args.output_dir.resolve()
    if output.exists() or SOURCE.resolve() == output or SOURCE.resolve() in output.parents:
        raise FileExistsError("Use a fresh destination outside the preserved source cohort")
    output.mkdir(parents=True)
    started = time.perf_counter()
    deadline = time.monotonic() + args.max_seconds
    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process deadline exceeded")
    report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": "MZ0_complex_rival_diagnostics",
        "generating_model": "MZ", "fitted_model": args.model, "replicate_index": 0, "seed": shared.SEED,
        "source_dir": str(args.source_dir), "job_id": os.environ.get("SLURM_JOB_ID"),
        "started_utc": datetime.now(timezone.utc).isoformat(), "internal_seconds": args.max_seconds,
        "status": "running", "rival_diagnostics_complete": False, "saved_scores_replayed": False,
        "simulation_regenerated": False, "saved_M0_MZ_refitted": False,
        "family_selection": "not_evaluated", "confusion_rate": "not_evaluated", "group_recovery": "not_evaluated",
        "full_gate": "not_evaluated", "full_recovery_pass": None, "episode_terminal": False,
        "missing_nested_model_comparisons": ["MV"] if args.model == "MVZ" else ["MV", "MC"],
        "fits": [], "scored_primary_directions": [], "completed_rivals": [], "diagnostics": {}}
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
        people = shared.load_cohort(events, check)
        shared.validate_source(source, truth, people, spec)
        splits = [load_split(events, list(train), list(held)) for train, held in HALVES]
        scores, discrepancy = shared.replay_saved_scores(source, splits, spec, check)
        report.update(implementation=spec, source_finished_utc=source["finished_utc"], saved_scores_replayed=True,
            saved_score_replay_max_absolute_discrepancy=discrepancy,
            saved_score_replay_absolute_tolerance=shared.REPLAY_ABSOLUTE_TOLERANCE,
            participants=108, runs=432, trials=27648)
        write_report(output, report)
        for row in scores.values():
            row[args.model] = 0.
        for (train, held), (training, scoring) in zip(HALVES, splits):
            check()
            fit_started = time.perf_counter()
            fit = fit_partially_pooled(training, args.model, "fixed", spec)
            fit.update(training_runs=list(train), scoring_runs=list(held), elapsed_seconds=time.perf_counter() - fit_started)
            json.dumps(fit, allow_nan=False)
            report["fits"].append(fit)
            write_report(output, report)
            check()
            validate_fit(fit, args.model, train, held, spec, people)
            for person, value in shared.score_fit(fit, scoring, spec, check).items():
                scores[person][args.model] += value
            check()
            report["scored_primary_directions"].append({"training_runs": list(train), "scoring_runs": list(held)})
            write_report(output, report)
            print(json.dumps({"model": args.model, "training_runs": list(train), "fit_completed": True,
                              "elapsed_seconds": fit["elapsed_seconds"]}), flush=True)
        diagnostics = paired_diagnostics(scores, args.model)
        check()
        report.update(participant_scores=scores, diagnostics={args.model: diagnostics}, completed_rivals=[args.model])
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
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=5400)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("status", "fitted_model", "rival_diagnostics_complete", "diagnostics", "elapsed_seconds", "error")
                      if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["rival_diagnostics_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
