#!/usr/bin/env python3
"""One complementary two-run sensitivity on the preserved synthetic MZ0 cohort.

Qualify the original primary panel before four fresh original500 M0/MZ fits.
Only the alternate-partition nested contrast is reported; no recovery gate,
independent replication, family selection, interval or scientific rate follows.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time

from qualify_assets import require_compute
import run_single_process_rivals as shared
import summarize_mz_fixed_menu as fixed


SOURCE, IMPLEMENTATION = shared.SOURCE, shared.IMPLEMENTATION
SOURCE_REPORT = "single_process_recovery.json"
PARTITIONS = {"12_34": (((1, 2), (3, 4)), ((3, 4), (1, 2))),
              "13_24": (((1, 3), (2, 4)), ((2, 4), (1, 3)))}
MAX_SECONDS = 5400
PHASE = "MZ0_other_partition_M0_MZ_nested_diagnostics"
REPORT_NAME = "mz_other_partitions.json"
CLAIM_FIELDS = ("family_selection", "family_recovery_rate", "false_selection_rate", "confusion_rate",
                "group_recovery", "group_intervals", "interval_coverage", "individual_recovery",
                "full_menu_selection", "full_gate", "early_kill", "curtailment")
fit_partially_pooled, load_split, validate_fit = shared.fit_partially_pooled, shared.load_split, shared.validate_fit


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"Other-partition diagnostics interrupted by signal {signum}; unfinished work is technical")


def run(args):
    require_compute()
    if args.partition not in PARTITIONS:
        raise ValueError("Only the two nonprimary complementary two-run partitions are selected")
    if args.source_dir.resolve() != SOURCE.resolve() or args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the fixed original MZ0 source and unchanged implementation")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("The whole-process cap must be 1 through5400 seconds")
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

    directions = PARTITIONS[args.partition]
    report = {
        "analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "generating_model": "MZ", "replicate_index": 0, "seed": shared.SEED,
        "partition": args.partition, "fitted_models": ["M0", "MZ"],
        "partition_directions": [{"training_runs": list(train), "scoring_runs": list(held)} for train, held in directions],
        "source_dir": str(SOURCE), "source_report": str(SOURCE / SOURCE_REPORT),
        "source_baseline_validated": False, "saved_primary_scores_replayed": False,
        "job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat(),
        "internal_seconds": args.max_seconds, "status": "running", "partition_diagnostic_complete": False,
        "simulation_regenerated": False, "simulation_copied": False, "saved_primary_fits_reused_for_alternate_partition": False,
        "observed_choices_or_RTs_used": False, "selected_replicates": [0], "wave_selected": False,
        "automatic_retry": False, "partitions_are_independent_replications": False,
        "full_recovery_pass": None, "episode_terminal": False,
        **{field: "not_evaluated" for field in CLAIM_FIELDS},
        "interpretation": "One saved MZ0 synthetic-cohort sensitivity. Both alternate directions have fresh training-only M0/MZ fits; each128-trial score contributes sum(logp)/256. Original primary scores only qualify the source and are never added to the alternate panel. No full-menu, recovery-rate, interval, curtailment or gate verdict; these partitions are not independent replications.",
        "remaining_leave_one_run_out_obligation": "not_evaluated",
        "cost_accounting": "elapsed_seconds is this source-validation, replay and four-fit sensitivity process only; original cohort cost is not charged again",
        "fits": [], "scored_fits": [], "scored_partition_directions": [],
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
            raise ValueError("All four new fits require the unchanged original500 implementation")
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
        # Keep shared.HALVES untouched: saved source fits/scores belong ONLY to
        # the original14<->23 primary partition.
        primary_splits = [load_split(events, list(train), list(held)) for train, held in shared.HALVES]
        check()
        primary_scores, discrepancy = shared.replay_saved_scores(source, primary_splits, spec, check)
        report.update(saved_primary_scores_replayed=True, saved_primary_participant_scores=primary_scores,
                      saved_score_replay_max_absolute_discrepancy=discrepancy,
                      saved_score_replay_absolute_tolerance=shared.REPLAY_ABSOLUTE_TOLERANCE)
        write_report(output, report)
        scores = {person: {"M0": 0., "MZ": 0., "version": rows[0]["task_version"]} for person, rows in people.items()}
        for train, held in directions:
            check()
            training, scoring = load_split(events, list(train), list(held))
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
                for person, value in shared.score_fit(fit, scoring, spec, check).items():
                    scores[person][model] += value
                check()
                report["scored_fits"].append({"model": model, "training_runs": list(train), "scoring_runs": list(held)})
                write_report(output, report)
                print(json.dumps({"partition": args.partition, "model": model, "training_runs": list(train),
                                  "fit_completed": True, "elapsed_seconds": fit["elapsed_seconds"]}), flush=True)
            report["scored_partition_directions"].append({"training_runs": list(train), "scoring_runs": list(held)})
            write_report(output, report)
        fixed.score_panel({"participant_scores": scores}, ("M0", "MZ"))
        diagnostic = fixed.nested_contrast(scores, "MZ", "M0")
        check()
        report.update(participant_scores=scores, nested_diagnostic=diagnostic,
                      status="partition_diagnostic_completed", partition_diagnostic_complete=True,
                      elapsed_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        write_report(output, report)
        check()  # Successful finalization is inside the same whole-process cap.
        succeeded = True
    except Exception as error:
        report.update(status="technical_partition_failure", error=f"{type(error).__name__}: {error}")
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if not succeeded:
            report.pop("participant_scores", None)
            report.pop("nested_diagnostic", None)
            report.update(status="technical_partition_failure", partition_diagnostic_complete=False,
                          elapsed_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
            # Retain all returned fits and completed scoring prefixes. Only
            # failure closure uses the external wrapper's finite grace.
            write_report(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=tuple(PARTITIONS), required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "partition", "status", "partition_diagnostic_complete",
        "nested_diagnostic", "elapsed_seconds", "error") if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["partition_diagnostic_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
