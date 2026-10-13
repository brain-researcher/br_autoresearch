#!/usr/bin/env python3
"""One nonlinear-value M0 or MZ component on the preserved synthetic MZ0.

Unequal gain/loss powers make nonlinear M0 a curvature-adjusted reference,
not globally symmetric valuation. Both fits are fresh and training-only;
source primary fits are validated/replayed but never relabeled or refitted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from hierarchical_behavior import parameter_names
from qualify_assets import require_compute
import run_single_process_rivals as shared
import summarize_mz_fixed_menu as fixed


SOURCE, IMPLEMENTATION, HALVES = shared.SOURCE, shared.IMPLEMENTATION, shared.HALVES
SOURCE_REPORT = "single_process_recovery.json"
MODELS, RIVAL, MAX_SECONDS = ("M0", "MZ"), "nonlinear", 5400
PHASE = "MZ0_nonlinear_value_component"
REPORT_NAME = "mz_nonlinear.json"
SCORE_KEY = "nonlinear_score"
NORMALIZATION = "sum_of_two128_trial_heldout_log_likelihoods_divided_by256_per_participant"
CLAIM_FIELDS = ("family_selection", "family_recovery_rate", "false_selection_rate", "confusion_rate",
                "group_recovery", "group_intervals", "interval_coverage", "individual_recovery",
                "full_menu_selection", "full_gate", "early_kill", "curtailment", "nonlinear_robustness")
fit_partially_pooled, load_split = shared.fit_partially_pooled, shared.load_split


def validate_nonlinear_fit(fit, model, train, held, spec, people):
    """Original fit checks with truthful nonlinear identity and5/6 parameters."""
    if model not in MODELS:
        raise ValueError("Only nonlinear M0 and MZ components are selected")
    names = parameter_names(model, RIVAL)
    if (fit.get("model") != model or fit.get("rival") != RIVAL
            or fit.get("training_runs") != list(train) or fit.get("scoring_runs") != list(held)
            or fit.get("participants") != sorted(people) or fit.get("parameter_names") != names
            or fit.get("version_order") != ["EI", "ER"] or fit.get("inference") != spec["method"]):
        raise ValueError("Nonlinear fit model, rival, split, participant or parameter identity changed")
    if fit.get("converged") is not True or fit.get("subject_optimizer_failures") != []:
        raise RuntimeError("Nonconverged nonlinear fit is technical; it is not a recovery failure")
    theta = np.asarray(fit["parameters"], dtype=float)
    if theta.shape != (108, len(names)) or not np.isfinite(theta).all():
        raise ValueError("Incomplete or nonfinite nonlinear participant parameters")
    for column, name in enumerate(names):
        low, high = spec["parameters"][name]["bounds"]
        if np.any(theta[:, column] < low) or np.any(theta[:, column] > high):
            raise ValueError("Fitted nonlinear parameter outside unchanged support")
    for field, shape in (("global_mean", (len(names),)), ("version_means", (2, len(names))),
                         ("version_standard_deviations", (2, len(names)))):
        values = np.asarray(fit.get(field), dtype=float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Incomplete or nonfinite fitted hyperparameters: {field}")
        if field == "version_standard_deviations" and np.any(values <= 0):
            raise ValueError("Nonpositive fitted population scale")
    objective = np.asarray(fit["objective_history"], dtype=float)
    if (objective.ndim != 1 or not 2 <= len(objective) <= spec["optimizer"]["maximum_sweeps"]
            or not np.isfinite(objective).all()
            or abs(objective[-1] - objective[-2]) > spec["optimizer"]["joint_relative_tolerance"] * (1 + abs(objective[-2]))):
        raise ValueError("Nonlinear fit lacks the unchanged convergence evidence")
    if not isinstance(fit.get("support_contacts"), list):
        raise ValueError("Missing numeric-support diagnostics")


def score_nonlinear_fit(fit, scoring, spec, check=lambda: None):
    scores = {}
    for person, theta in zip(fit["participants"], fit["parameters"]):
        check()
        rows = scoring[person]["rows"]
        if len(rows) != 128:
            raise ValueError("Nonlinear scoring requires128 held-out rows per half")
        logp = np.asarray(shared.trial_likelihood(theta, fit["parameter_names"], shared.make_data(rows), spec, RIVAL))
        if logp.shape != (128,) or not np.isfinite(logp).all():
            raise ValueError("Nonlinear held-out likelihood must contain128 finite trial scores")
        scores[person] = float(logp.sum()) / 256
    return scores


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"Nonlinear component interrupted by signal {signum}; unfinished work is technical")


def run(args):
    require_compute()
    if args.model not in MODELS:
        raise ValueError("Select only nonlinear M0 or MZ")
    if args.source_dir.resolve() != SOURCE.resolve() or args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the fixed original MZ0 source and unchanged implementation")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("The whole-process cap must be1 through5400 seconds")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1" or any(os.environ.get(name) != "1" for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("Exactly one allocated CPU and all four numerical thread limits of one are required")
    output = args.output_dir.resolve()
    if output.exists() or output == SOURCE.resolve() or SOURCE.resolve() in output.parents:
        raise FileExistsError("Use a fresh output outside the preserved original MZ0 source")
    output.mkdir(parents=True)
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process deadline exceeded")

    report = {
        "analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "generating_model": "MZ", "replicate_index": 0, "seed": shared.SEED,
        "fitted_model": args.model, "rival": RIVAL,
        "source_dir": str(SOURCE), "source_report": str(SOURCE / SOURCE_REPORT),
        "source_baseline_validated": False, "saved_primary_scores_replayed": False,
        "job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat(),
        "internal_seconds": args.max_seconds, "status": "running", "nonlinear_component_complete": False,
        "simulation_regenerated": False, "simulation_copied": False, "saved_primary_fits_reused_for_nonlinear": False,
        "observed_choices_or_RTs_used": False, "selected_replicates": [0], "wave_selected": False,
        "automatic_retry": False, "full_recovery_pass": None, "episode_terminal": False,
        **{field: "not_evaluated" for field in CLAIM_FIELDS},
        "score_normalization": NORMALIZATION,
        "interpretation": "One nonlinear M0 or MZ component on saved MZ0. Independent gain/loss powers make nonlinear M0 a curvature-adjusted reference, not globally symmetric valuation. Incremental rho can be assessed only after both nonlinear components complete. Original fixed scores qualify the source; no fixed-versus-nonlinear arbitration, family, scientific-rate, interval, curtailment or gate verdict is supplied.",
        "cost_accounting": "elapsed_seconds covers this validation, primary replay and two-fit nonlinear process; prior cohort cost is not charged again",
        "fits": [], "scored_primary_directions": [],
    }
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    succeeded = False
    try:
        write_report(output, report)
        check()
        spec = json.loads(args.implementation.read_text())
        if spec["optimizer"]["maximum_sweeps"] != 500:
            raise ValueError("Both fresh nonlinear fits require the original500 implementation")
        report["implementation"] = spec
        source = json.loads((SOURCE / SOURCE_REPORT).read_text())
        truth = json.loads((SOURCE / "generating_parameters.json").read_text())
        events = SOURCE / "simulated_events.csv.gz"
        people = shared.load_cohort(events, check)
        shared.validate_source(source, truth, people, spec)
        check()
        report.update(source_baseline_validated=True, source_job_id=source["job_id"],
                      source_finished_utc=source["finished_utc"], participants=108, runs=432, trials=27648,
                      heldout_trials_per_direction_per_participant=128, combined_scored_trials_per_participant=256)
        write_report(output, report)
        splits = [load_split(events, list(train), list(held)) for train, held in HALVES]
        check()
        primary_scores, discrepancy = shared.replay_saved_scores(source, splits, spec, check)
        report.update(saved_primary_scores_replayed=True, saved_primary_participant_scores=primary_scores,
                      saved_score_replay_max_absolute_discrepancy=discrepancy,
                      saved_score_replay_absolute_tolerance=shared.REPLAY_ABSOLUTE_TOLERANCE)
        write_report(output, report)
        scores = {person: {SCORE_KEY: 0., "version": rows[0]["task_version"]} for person, rows in people.items()}
        for (train, held), (training, scoring) in zip(HALVES, splits):
            check()
            fit_started = time.perf_counter()
            fit = fit_partially_pooled(training, args.model, RIVAL, spec)
            fit.update(training_runs=list(train), scoring_runs=list(held), elapsed_seconds=time.perf_counter() - fit_started)
            json.dumps(fit, allow_nan=False)
            report["fits"].append(fit)
            write_report(output, report)
            check()
            validate_nonlinear_fit(fit, args.model, train, held, spec, people)
            for person, value in score_nonlinear_fit(fit, scoring, spec, check).items():
                scores[person][SCORE_KEY] += value
            check()
            report["scored_primary_directions"].append({"training_runs": list(train), "scoring_runs": list(held)})
            write_report(output, report)
            print(json.dumps({"model": args.model, "rival": RIVAL, "training_runs": list(train),
                              "fit_completed": True, "elapsed_seconds": fit["elapsed_seconds"]}), flush=True)
        fixed.score_panel({"participant_scores": scores}, (SCORE_KEY,))
        check()
        report.update(participant_scores=scores, status="nonlinear_component_completed", nonlinear_component_complete=True,
                      elapsed_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        write_report(output, report)
        check()
        succeeded = True
    except Exception as error:
        report.update(status="technical_nonlinear_failure", error=f"{type(error).__name__}: {error}")
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if not succeeded:
            report.pop("participant_scores", None)
            report.update(status="technical_nonlinear_failure", nonlinear_component_complete=False,
                          elapsed_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
            write_report(output, report)  # Only failure closure uses the wrapper's finite grace.
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "fitted_model", "rival", "status", "nonlinear_component_complete",
        "elapsed_seconds", "error") if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["nonlinear_component_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
