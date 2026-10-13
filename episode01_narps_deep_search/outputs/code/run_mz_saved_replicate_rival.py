#!/usr/bin/env python3
"""Fit one declared rival on one unchanged saved MZ-SESOI replicate1–199.

Original source index, seed, simulation, M0/MZ fits and recorded scores remain
unchanged. No model-family selection, confusion rate, interval or gate verdict
is computed. Scheduling/model selection belongs to the coordinator.
"""
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
from run_single_process_recovery import DESIGN_FIELDS, DESIGN_SOURCE, HALVES, IMPLEMENTATION, cell_seed, focal_truth, validate_fit
from summarize_single_process_recovery import summarize_reports
import run_single_process_rivals as shared


RUNS_ROOT = Path(__file__).resolve().parents[2] / "outputs/runs/behavioral_mechanism_kill_test"
MODELS = ("MC", "MV", "MVZ", "MVC")
REPORT_NAME = "mz_saved_replicate_rival.json"


def source_directory(index):
    if type(index) is not int or not 1 <= index <= 199:
        raise ValueError("Only the fixed saved MZ replicate indices1 through199 are selected")
    array = 46538652 if index <= 20 else 46539590
    return RUNS_ROOT / f"20261005-MZ-SESOI-array-{array}" / f"replicate{index}"


def validate_source(report, truth, people, spec, index):
    """Adapt the existing MZ0 validator to the real fixed index; never relabel."""
    source_directory(index)
    summary = summarize_reports([(f"saved_MZ_replicate{index}", report)], "MZ", spec)
    if (summary["valid_replicates"] != 1 or report["replicate_index"] != index
            or report["seed"] != cell_seed("MZ", index)
            or [report.get(key) for key in ("participants", "runs", "trials")] != [108, 432, 27648]
            or report.get("observed_choices_or_RTs_used") is not False
            or report.get("old_synthetic_responses_used") is not False
            or report.get("design_fields") != list(DESIGN_FIELDS)
            or Path(report.get("design_source", "")).resolve() != DESIGN_SOURCE.resolve()):
        raise ValueError("Only the completed original MZ report for the selected index/seed is allowed")
    for field in ("started_utc", "finished_utc"):
        value = report.get(field)
        if not isinstance(value, str) or datetime.fromisoformat(value).utcoffset() is None:
            raise ValueError("Original source report must have explicit dated terminal provenance")
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
            raise ValueError("The saved cohort is not the fixed MZ-SESOI truth")
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


def paired_diagnostics(scores, model):
    """Same descriptive contrasts for all four rivals, without a screen/CI."""
    if model not in MODELS or len(scores) != 108:
        raise ValueError("Select one declared rival and retain all108 participants")
    if Counter(row["version"] for row in scores.values()) != {"EI": 54, "ER": 54}:
        raise ValueError("Both unchanged54-person task versions are required")
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
            "interpretation": "Paired descriptive contrast only; declared-menu selection and confusion remain unevaluated"}
    return result


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"Saved-replicate rival interrupted by signal{signum}; unfinished work is technical")


def run(args):
    require_compute()
    expected_source = source_directory(args.replicate_index)
    if args.model not in MODELS:
        raise ValueError("Select exactly one MC/MV/MVZ/MVC model")
    if args.source_dir.resolve() != expected_source.resolve() or args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use only the exact selected index's saved array directory and unchanged implementation")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= 5400:
        raise ValueError("The whole-process internal cap is at most5400 seconds")
    if any(os.environ.get(name) != "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("All four numerical thread limits must be explicitly one")
    output = args.output_dir.resolve()
    if output.exists() or output == expected_source.resolve() or expected_source.resolve() in output.parents:
        raise FileExistsError("Use a fresh destination outside the preserved source cohort")
    output.mkdir(parents=True)
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds
    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process deadline exceeded")
    report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": "MZ_SESOI_saved_replicate_rival_diagnostics",
        "generating_model": "MZ", "fitted_model": args.model,
        "replicate_index": args.replicate_index, "seed": cell_seed("MZ", args.replicate_index),
        "source_dir": str(expected_source), "job_id": os.environ.get("SLURM_JOB_ID"),
        "started_utc": datetime.now(timezone.utc).isoformat(), "internal_seconds": args.max_seconds,
        "status": "running", "rival_diagnostics_complete": False, "saved_scores_replayed": False,
        "simulation_regenerated": False, "saved_M0_MZ_refitted": False,
        "observed_choices_or_RTs_used": False, "family_selection": "not_evaluated",
        "confusion_rate": "not_evaluated", "group_recovery": "not_evaluated", "interval_coverage": "not_evaluated",
        "full_gate": "not_evaluated", "full_recovery_pass": None, "episode_terminal": False,
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
        validate_source(source, truth, people, spec, args.replicate_index)
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
            print(json.dumps({"model": args.model, "replicate_index": args.replicate_index,
                              "training_runs": list(train), "fit_completed": True,
                              "elapsed_seconds": fit["elapsed_seconds"]}), flush=True)
        diagnostics = paired_diagnostics(scores, args.model)
        check()
        report.update(participant_scores=scores, diagnostics={args.model: diagnostics}, completed_rivals=[args.model])
        write_report(output, report)
        check()
        succeeded = True
    except Exception as error:
        report.update(status="technical_rival_failure", error=f"{type(error).__name__}: {error}")
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if succeeded and time.monotonic() >= deadline:
            succeeded = False
            report.update(status="technical_rival_failure", error="RuntimeError: Internal deadline reached during finalization")
        if not succeeded:
            report.pop("participant_scores", None)
            report.update(completed_rivals=[], diagnostics={})
        report.update(status="rival_diagnostics_completed" if succeeded else report["status"],
                      rival_diagnostics_complete=succeeded, elapsed_seconds=time.perf_counter() - started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_report(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--replicate-index", type=int, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=5400)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("status", "fitted_model", "replicate_index", "seed",
        "rival_diagnostics_complete", "diagnostics", "elapsed_seconds", "error") if key in report},
        indent=2, allow_nan=False), flush=True)
    return 0 if report["rival_diagnostics_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
