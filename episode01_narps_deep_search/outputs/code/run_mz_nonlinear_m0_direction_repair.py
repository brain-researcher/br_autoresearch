#!/usr/bin/env python3
"""Independent fresh nonlinear M0 directions after exact first-half exhaustion.

Direction0 uses2000 sweeps after its1500 failure; unattempted direction1 uses
1500. Four participant workers preserve the original serial population updates
and objective addition order. Each report supplies only one128/256 contribution.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from qualify_assets import require_compute
import hierarchical_behavior_parallel as parallel
import run_mz_nonlinear_m0_repair as previous
import summarize_mz_nonlinear_repaired as previous_pool


nonlinear = previous.nonlinear
SOURCE, IMPLEMENTATION = nonlinear.SOURCE, nonlinear.IMPLEMENTATION
FAILED_JOB = "46611338"
FAILED_REPORT = SOURCE.parent / f"20261005-MZ-nonlinear-M0-repair-{FAILED_JOB}" / previous.REPORT_NAME
FAILED_FINISHED = "2026-10-05T13:20:40.127557+00:00"
FAILED_PROCESS_SECONDS = 6302.497318990994
FAILED_FIT_SECONDS = 6301.2349644051865
FAILED_TAIL = [7870.31500201187, 7870.314182004591, 7870.313346104512]
PHASE = "MZ0_nonlinear_M0_independent_direction_numerical_repair"
REPORT_NAME = "mz_nonlinear_m0_direction_repair.json"
NORMALIZATION = "one128_trial_heldout_log_likelihood_sum_divided_by256_per_participant"
MAX_SECONDS, WORKERS = 4800, 4
SWEEPS = {0: 2000, 1: 1500}


def selected_runs(direction):
    if type(direction) is not int or direction not in SWEEPS:
        raise ValueError("Select only the explicit primary direction0 or1")
    return nonlinear.HALVES[direction]


def runtime_specification(original, direction):
    selected_runs(direction)
    if original["optimizer"]["maximum_sweeps"] != 500:
        raise ValueError("Preserve the original500 implementation")
    runtime = deepcopy(original)
    runtime["optimizer"]["maximum_sweeps"] = SWEEPS[direction]
    return runtime


def numerical_override(direction):
    selected_runs(direction)
    return {"parameter": "optimizer.maximum_sweeps", "original": 500, "runtime": SWEEPS[direction],
        "scope": "one_fresh_nonlinear_M0_primary_direction", "direction": direction,
        "execution_kind": "first_direction_1500_exhaustion_repair" if direction == 0 else "previously_unattempted_reverse_direction",
        "likelihood_priors_support_starts_tolerances_and_first_convergence_changed": False,
        "nonconverged_optimizer_state_reused": False,
        "parallel_arithmetic": "fixed_four_workers_sorted_collection_original_serial_population_updates_and_objective_reduction"}


def _validated_contexts(failed, source, primary, original):
    """Validate source-key identity and both exact terminal prior attempts."""
    runtime_specification(original, 0)
    if primary != source["participant_scores"]:
        raise ValueError("Use independently validated original source score last bits")
    original_failed = json.loads(previous.FAILED_REPORT.read_text())
    original_context = previous.repair_context(original_failed, source, primary, original)
    _, scores, replayed_primary, identity = previous_pool.validate_repair(failed, FAILED_JOB, original, original_context)
    if (scores is not None or replayed_primary != primary or identity != (str(source["job_id"]), source["finished_utc"])
            or failed.get("status") != "technical_nonlinear_failure" or failed.get("error") != previous.NONCONVERGENCE
            or failed.get("finished_utc") != FAILED_FINISHED or failed.get("elapsed_seconds") != FAILED_PROCESS_SECONDS
            or failed.get("internal_seconds") != 15600 or failed.get("repair_context_validated") is not True
            or failed.get("fresh_fit_calls") != 1 or len(failed["fits"]) != 1
            or failed.get("scored_primary_directions") != []):
        raise ValueError("Require exact46611338 first-half1500 exhaustion with no scored direction")
    fit = failed["fits"][0]
    names = nonlinear.parameter_names("M0", "nonlinear")
    if (fit.get("parameter_names") != names or fit.get("version_order") != ["EI", "ER"]
            or fit.get("inference") != original["method"] or fit.get("converged") is not False
            or fit.get("subject_optimizer_failures") != [] or fit.get("elapsed_seconds") != FAILED_FIT_SECONDS):
        raise ValueError("Retain the diagnosed finite first nonlinear M0 fit")
    theta = np.asarray(fit.get("parameters"), dtype=float)
    if theta.shape != (108, len(names)) or not np.isfinite(theta).all():
        raise ValueError("Retain complete finite failed participant parameters")
    for column, name in enumerate(names):
        low, high = original["parameters"][name]["bounds"]
        if np.any(theta[:, column] < low) or np.any(theta[:, column] > high):
            raise ValueError("The failed fit must retain original parameter support")
    for field, shape in (("global_mean", (len(names),)), ("version_means", (2, len(names))),
                         ("version_standard_deviations", (2, len(names)))):
        value = np.asarray(fit.get(field), dtype=float)
        if value.shape != shape or not np.isfinite(value).all() or (field == "version_standard_deviations" and np.any(value <= 0)):
            raise ValueError("Retain finite full failed hyperparameter evidence")
    history = np.asarray(fit.get("objective_history"), dtype=float)
    if history.shape != (1500,) or not np.isfinite(history).all() or history[-3:].tolist() != FAILED_TAIL:
        raise ValueError("Require the exact1500-sweep exhaustion trajectory endpoint")
    change = float(abs(history[-1] - history[-2]))
    threshold = original["optimizer"]["joint_relative_tolerance"] * (1 + abs(float(history[-2])))
    if not change > threshold:
        raise ValueError("Retain failure of the unchanged first-convergence criterion")
    contacts = fit.get("support_contacts")
    expected = {("EI", "gamma_gain"), ("EI", "gamma_loss"),
                ("ER", "kappa"), ("ER", "gamma_gain"), ("ER", "gamma_loss")}
    if (not isinstance(contacts, list) or len(contacts) != 5
            or {(row.get("version"), row.get("parameter")) for row in contacts} != expected):
        raise ValueError("Retain the same five diagnosed lower population-scale contacts")
    for contact in contacts:
        name = contact["parameter"]
        value = -3.9999998503354774 if name == "kappa" else -3.999999830370791
        if (contact.get("level") != "version_log_standard_deviation"
                or contact.get("bounds") != original["parameters"][name]["individual_log_scale_bounds"]
                or contact.get("value") != value):
            raise ValueError("Preserve exact recorded lower support-contact values")
    context = {"kind": "nonlinear_M0_first_half1500_exhaustion_reverse_unattempted",
        "failed_report": str(FAILED_REPORT), "failed_job_id": FAILED_JOB,
        "failed_started_utc": failed["started_utc"], "failed_finished_utc": FAILED_FINISHED,
        "failed_error": failed["error"], "source_job_id": source["job_id"], "source_finished_utc": source["finished_utc"],
        "original_maximum_sweeps": 500, "failed_runtime_maximum_sweeps": 1500,
        "previous_process_elapsed_seconds": failed["elapsed_seconds"],
        "previous_fit_elapsed_seconds": fit["elapsed_seconds"],
        "previous_cumulative_process_elapsed_seconds": failed["cumulative_process_elapsed_seconds"],
        "previous_support_contacts": deepcopy(contacts), "final_objective_change": change,
        "unchanged_convergence_threshold": threshold, "completed_first_fit_available": False,
        "reverse_fit_previously_attempted": False, "nonconverged_checkpoint_state_reused": False,
        "initialization": "unchanged_fresh_prior_centers_and_recorded_first_sweep_offsets",
        "cost_accounting": "Both prior failed processes are shared references. Count each once across the direction pair; never add the cumulative reference again to its two component costs."}
    return context, original_context


def repair_context(failed, source, primary, original):
    return _validated_contexts(failed, source, primary, original)[0]


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"Nonlinear M0 direction repair interrupted by signal {signum}; unfinished work is technical")


def run(args):
    require_compute()
    train, held = selected_runs(args.direction)
    if args.source_dir.resolve() != SOURCE.resolve() or args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use only the fixed original MZ0 source and implementation")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("Keep the whole-process cap within1 through4800 seconds")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "4" or any(os.environ.get(name) != "1" for name in parallel.THREAD_ENV):
        raise ValueError("Require four actually allocated CPUs and all six numerical thread limits of one")
    output = args.output_dir.resolve()
    protected = (SOURCE, FAILED_REPORT.parent, previous.FAILED_REPORT.parents[1])
    if output.exists() or any(output == root.resolve() or root.resolve() in output.parents for root in protected):
        raise FileExistsError("Preserve all original sources/failures; use a fresh output directory")
    output.mkdir(parents=True)
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process deadline exceeded")

    report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "generating_model": "MZ", "replicate_index": 0, "seed": nonlinear.shared.SEED,
        "fitted_model": "M0", "rival": "nonlinear", "direction": args.direction,
        "training_runs": list(train), "scoring_runs": list(held), "direction_contribution_only": True,
        "score_normalization": NORMALIZATION, "heldout_trials_per_direction_per_participant": 128,
        "combined_scored_trials_per_participant": 256,
        "source_dir": str(SOURCE), "source_report": str(SOURCE / nonlinear.SOURCE_REPORT),
        "failed_report": str(FAILED_REPORT), "repair_context_validated": False,
        "source_baseline_validated": False, "saved_primary_scores_replayed": False,
        "prior_1500_failed_context": None, "original_500_failed_context": None,
        "original_implementation": None, "runtime_implementation": None,
        "original_maximum_sweeps": 500, "fitted_maximum_sweeps": SWEEPS[args.direction],
        "numerical_override": numerical_override(args.direction),
        "workers": WORKERS, "allocated_cpus": 4, "parallel_execution": True,
        "parallel_fit_max_seconds": None, "parallel_cleanup_reserve_seconds": parallel.CLEANUP_SECONDS + 1.,
        "parallel_progress": None, "parallel_completed_sweeps": [], "parallel_failure": None,
        "parallel_workers_cleaned_up": None, "fresh_fit_calls": 0,
        "job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat(),
        "internal_seconds": args.max_seconds, "status": "running", "nonlinear_direction_complete": False,
        "simulation_regenerated": False, "simulation_copied": False, "saved_primary_fits_reused_for_nonlinear": False,
        "observed_choices_or_RTs_used": False, "selected_replicates": [0], "wave_selected": False,
        "retry_wave_selected": False, "automatic_retry": False, "full_recovery_pass": None, "episode_terminal": False,
        **{field: "not_evaluated" for field in nonlinear.CLAIM_FIELDS},
        "cost_accounting": "elapsed_seconds is this new four-CPU direction process only. Both earlier M0 failures are shared references counted once by the pool/coordinator, not added anew for each direction or recharged as allocation cost.",
        "interpretation": "One fresh nonlinear curvature-adjusted M0 direction after the exact first-half1500 exhaustion. Direction0 uses2000, unattempted direction1 uses1500; only ceilings change. Four participant workers preserve sorted population updates and objective addition order. No failed state, full-range fixed fit, or sibling nonlinear fit is reused. This128-trial score contributes sum/256; both directions and nonlinear MZ are needed for the separate diagnostic. No robustness, family, rate, interval, recovery, curtailment or gate verdict.",
        "fits": [], "scored_primary_directions": [], "completed_direction_scores": {}}

    def checkpoint():
        report["elapsed_seconds"] = time.perf_counter() - started
        write_report(output, report)

    def progress(event):
        check()
        report["parallel_progress"] = deepcopy(event)
        if event["stage"] == "sweep":
            report["parallel_completed_sweeps"].append(deepcopy(event))
            checkpoint()

    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in previous_handlers:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    succeeded = False
    try:
        checkpoint()
        check()
        original = json.loads(args.implementation.read_text())
        runtime = runtime_specification(original, args.direction)
        report.update(implementation=original, original_implementation=original, runtime_implementation=runtime)
        source = json.loads((SOURCE / nonlinear.SOURCE_REPORT).read_text())
        truth = json.loads((SOURCE / "generating_parameters.json").read_text())
        events = SOURCE / "simulated_events.csv.gz"
        people = nonlinear.shared.load_cohort(events, check)
        nonlinear.shared.validate_source(source, truth, people, original)
        check()
        report.update(source_baseline_validated=True, source_job_id=source["job_id"],
            source_finished_utc=source["finished_utc"], participants=108, runs=432, trials=27648)
        checkpoint()
        splits = [nonlinear.load_split(events, list(a), list(b)) for a, b in nonlinear.HALVES]
        primary, discrepancy = nonlinear.shared.replay_saved_scores(source, splits, original, check)
        report.update(saved_primary_scores_replayed=True, saved_primary_participant_scores=primary,
            saved_score_replay_max_absolute_discrepancy=discrepancy,
            saved_score_replay_absolute_tolerance=nonlinear.shared.REPLAY_ABSOLUTE_TOLERANCE)
        failed = json.loads(FAILED_REPORT.read_text())
        context, original_context = _validated_contexts(failed, source, primary, original)
        check()
        report.update(repair_context_validated=True, prior_1500_failed_context=context, original_500_failed_context=original_context)
        training, scoring = splits[args.direction]
        if set(training) != set(people) or set(scoring) != set(people):
            raise ValueError("Retain all108 actual source participants in each direction")
        for person in people:
            for panel, runs in ((training, train), (scoring, held)):
                if (panel[person]["version"] != people[person][0]["task_version"]
                        or len(panel[person]["rows"]) != 128
                        or {int(row["run"]) for row in panel[person]["rows"]} != set(runs)):
                    raise ValueError("Each train/held panel must retain its original128 rows and complementary runs")
        checkpoint()
        check()
        remaining = deadline - time.monotonic() - report["parallel_cleanup_reserve_seconds"]
        if remaining <= 0:
            raise RuntimeError("Insufficient whole-process time remains for bounded worker cleanup")
        report["parallel_fit_max_seconds"] = remaining
        fit_started = time.perf_counter()
        report["fresh_fit_calls"] = 1
        try:
            fit = parallel.fit_partially_pooled_parallel(training, "M0", "nonlinear", runtime,
                workers=WORKERS, max_seconds=remaining, progress=progress)
        except BaseException as error:
            report["parallel_failure"] = deepcopy(getattr(error, "parallel_progress", None))
            if report["parallel_failure"] is not None:
                report["parallel_workers_cleaned_up"] = report["parallel_failure"].get("workers_cleaned_up")
            raise
        report["parallel_workers_cleaned_up"] = True
        fit.update(training_runs=list(train), scoring_runs=list(held), elapsed_seconds=time.perf_counter() - fit_started)
        json.dumps(fit, allow_nan=False)
        report["fits"].append(fit)
        checkpoint()
        check()
        nonlinear.validate_nonlinear_fit(fit, "M0", train, held, runtime, people)
        contribution = nonlinear.score_nonlinear_fit(fit, scoring, original, check)
        panel = {person: {"version": people[person][0]["task_version"], "nonlinear_score": contribution[person]}
                 for person in sorted(people)}
        nonlinear.fixed.score_panel({"participant_scores": panel}, ("nonlinear_score",))
        check()
        report["completed_direction_scores"] = panel
        report["scored_primary_directions"].append({"training_runs": list(train), "scoring_runs": list(held)})
        report["participant_scores"] = deepcopy(panel)
        report.update(status="nonlinear_direction_completed", nonlinear_direction_complete=True,
            finished_utc=datetime.now(timezone.utc).isoformat())
        checkpoint()
        check()
        succeeded = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        signal.alarm(0)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if succeeded:
            try:
                check()
            except BaseException as error:
                report["error"] = f"{type(error).__name__}: {error}"
                succeeded = False
        if not succeeded:
            report.pop("participant_scores", None)
            report.update(status="technical_nonlinear_failure", nonlinear_direction_complete=False,
                finished_utc=datetime.now(timezone.utc).isoformat())
            checkpoint()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", type=int, choices=(0, 1), required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "status", "direction", "nonlinear_direction_complete",
        "fresh_fit_calls", "parallel_workers_cleaned_up", "elapsed_seconds", "error") if key in report}, indent=2), flush=True)
    return 0 if report["nonlinear_direction_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
