#!/usr/bin/env python3
"""One registered LOO M0/MZ diagnostic on saved synthetic MZ0.

The original primary fit panel only qualifies the saved source. Two fresh
original500 fits use the other three runs; the held-out run contributes its
64-trial mean log score. No across-run aggregation or recovery gate is supplied.
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

from qualify_assets import require_compute
import run_single_process_rivals as shared
import summarize_mz_fixed_menu as fixed


SOURCE, IMPLEMENTATION = shared.SOURCE, shared.IMPLEMENTATION
SOURCE_REPORT = "single_process_recovery.json"
MAX_SECONDS = 5400
PHASE = "MZ0_leave_one_run_out_M0_MZ_nested_diagnostic"
REPORT_NAME = "mz_leave_one_run_out.json"
CLAIM_FIELDS = ("family_selection", "family_recovery_rate", "false_selection_rate", "confusion_rate",
                "group_recovery", "group_intervals", "interval_coverage", "individual_recovery",
                "full_menu_selection", "full_gate", "early_kill", "curtailment", "across_run_robustness")
fit_partially_pooled, load_split, validate_fit = shared.fit_partially_pooled, shared.load_split, shared.validate_fit


def selected_runs(held_out_run):
    if type(held_out_run) is not int or held_out_run not in (1, 2, 3, 4):
        raise ValueError("Select exactly one of the four registered held-out runs")
    return tuple(run for run in (1, 2, 3, 4) if run != held_out_run), (held_out_run,)


def validate_split(training, scoring, people, train, held):
    if set(training) != set(people) or set(scoring) != set(people):
        raise ValueError("Both LOO panels require the original108 participants")
    for person, rows in people.items():
        version = rows[0]["task_version"]
        for panel, runs, size in ((training, train, 192), (scoring, held, 64)):
            record = panel[person]
            if (record["version"] != version or len(record["rows"]) != size
                    or {int(row["run"]) for row in record["rows"]} != set(runs)):
                raise ValueError("LOO requires192 training and64 held-out trials with unchanged run/version identity")


def score_run(fit, scoring, spec, check=lambda: None):
    """Unchanged joint likelihood, normalized within exactly one64-trial run."""
    scores = {}
    for person, theta in zip(fit["participants"], fit["parameters"]):
        check()
        rows = scoring[person]["rows"]
        if len(rows) != 64:
            raise ValueError("LOO scoring requires exactly64 held-out rows per participant")
        logp = np.asarray(shared.trial_likelihood(theta, fit["parameter_names"], shared.make_data(rows), spec, "fixed"))
        if logp.shape != (64,) or not np.isfinite(logp).all():
            raise ValueError("LOO likelihood must contain64 finite held-out trial scores")
        scores[person] = float(logp.sum()) / 64
    return scores


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"LOO diagnostic interrupted by signal {signum}; unfinished work is technical")


def run(args):
    require_compute()
    train, held = selected_runs(args.held_out_run)
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
        "held_out_run": args.held_out_run, "training_runs": list(train), "scoring_runs": list(held),
        "fitted_models": ["M0", "MZ"], "source_dir": str(SOURCE), "source_report": str(SOURCE / SOURCE_REPORT),
        "source_baseline_validated": False, "saved_primary_scores_replayed": False,
        "job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat(),
        "internal_seconds": args.max_seconds, "status": "running", "loo_diagnostic_complete": False,
        "simulation_regenerated": False, "simulation_copied": False, "saved_primary_fits_reused_for_LOO": False,
        "observed_choices_or_RTs_used": False, "selected_replicates": [0], "wave_selected": False,
        "automatic_retry": False, "LOO_runs_are_independent_replications": False,
        "full_recovery_pass": None, "episode_terminal": False,
        **{field: "not_evaluated" for field in CLAIM_FIELDS},
        "score_normalization": "sum_of64_heldout_trial_log_likelihoods_divided_by64_per_participant",
        "interpretation": "One saved MZ0 synthetic-cohort LOO sensitivity. Each M0/MZ model is fitted afresh on the other three runs only. Scores average the64 trials of the omitted run; original primary scores only qualify the source. No across-run aggregation, independent-replication, family, scientific-rate, interval, curtailment or gate verdict.",
        "cost_accounting": "elapsed_seconds covers this validation, primary replay and two-fit process; prior cohort cost is not charged again",
        "fits": [], "scored_fits": [],
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
            raise ValueError("Both fresh LOO fits require the original500 implementation")
        report["implementation"] = spec
        source = json.loads((SOURCE / SOURCE_REPORT).read_text())
        truth = json.loads((SOURCE / "generating_parameters.json").read_text())
        events = SOURCE / "simulated_events.csv.gz"
        people = shared.load_cohort(events, check)
        shared.validate_source(source, truth, people, spec)
        check()
        report.update(source_baseline_validated=True, source_job_id=source["job_id"],
                      source_finished_utc=source["finished_utc"], participants=108, runs=432, trials=27648,
                      training_trials_per_participant=192, heldout_trials_per_participant=64)
        write_report(output, report)
        # Replay still uses the unchanged14<->23 source panel and scorer.
        primary_splits = [load_split(events, list(a), list(b)) for a, b in shared.HALVES]
        check()
        primary_scores, discrepancy = shared.replay_saved_scores(source, primary_splits, spec, check)
        report.update(saved_primary_scores_replayed=True, saved_primary_participant_scores=primary_scores,
                      saved_score_replay_max_absolute_discrepancy=discrepancy,
                      saved_score_replay_absolute_tolerance=shared.REPLAY_ABSOLUTE_TOLERANCE)
        write_report(output, report)
        training, scoring = load_split(events, list(train), list(held))
        validate_split(training, scoring, people, train, held)
        scores = {person: {"version": rows[0]["task_version"]} for person, rows in people.items()}
        for model in ("M0", "MZ"):
            check()
            fit_started = time.perf_counter()
            fit = fit_partially_pooled(training, model, "fixed", spec)
            fit.update(training_runs=list(train), scoring_runs=list(held), elapsed_seconds=time.perf_counter() - fit_started)
            json.dumps(fit, allow_nan=False)
            report["fits"].append(fit)
            write_report(output, report)
            check()
            validate_fit(fit, model, train, held, spec, people)
            for person, value in score_run(fit, scoring, spec, check).items():
                scores[person][model] = value
            check()
            report["scored_fits"].append({"model": model, "training_runs": list(train), "scoring_runs": list(held)})
            write_report(output, report)
            print(json.dumps({"held_out_run": args.held_out_run, "model": model, "training_runs": list(train),
                              "fit_completed": True, "elapsed_seconds": fit["elapsed_seconds"]}), flush=True)
        fixed.score_panel({"participant_scores": scores}, ("M0", "MZ"))
        diagnostic = fixed.nested_contrast(scores, "MZ", "M0")
        check()
        report.update(participant_scores=scores, nested_diagnostic=diagnostic,
                      status="loo_diagnostic_completed", loo_diagnostic_complete=True,
                      elapsed_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        write_report(output, report)
        check()
        succeeded = True
    except Exception as error:
        report.update(status="technical_loo_failure", error=f"{type(error).__name__}: {error}")
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if not succeeded:
            report.pop("participant_scores", None)
            report.pop("nested_diagnostic", None)
            report.update(status="technical_loo_failure", loo_diagnostic_complete=False,
                          elapsed_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
            write_report(output, report)  # Only failure closure uses the wrapper's finite grace.
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-out-run", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "held_out_run", "status", "loo_diagnostic_complete",
        "nested_diagnostic", "elapsed_seconds", "error") if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["loo_diagnostic_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
