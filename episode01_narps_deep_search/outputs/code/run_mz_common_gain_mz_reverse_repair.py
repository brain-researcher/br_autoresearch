#!/usr/bin/env python3
"""Fresh dir1/MZ common-gain repair with four bounded participant workers.

The sole numerical change is the diagnosed MZ sweep ceiling500-to1500.
Source qualification, replay and scoring retain the original500 specification.
The three other completed components remain separate immutable pool inputs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np

from qualify_assets import require_compute
import hierarchical_behavior_parallel as parallel
import run_mz_common_gain_range_repair as previous
import summarize_mz_common_gain_range_repaired as previous_pool


common = previous.common
SOURCE, IMPLEMENTATION = common.SOURCE, common.IMPLEMENTATION
FAILED_JOB = "46615238"
FAILED_REPORT = SOURCE.parent / f"20261005-MZ-common-gain-repair-{FAILED_JOB}" / "dir1/MZ" / previous.REPORT_NAME
FAILED_FINISHED = "2026-10-05T13:00:53.053307+00:00"
FAILED_PROCESS_SECONDS = 1160.472135021817
FAILED_FIT_SECONDS = 1158.414262315724
FAILED_TAIL = [5256.943583964956, 5256.942879696079]
PHASE = "MZ0_common_gain_direction1_MZ_runtime1500_parallel_repair"
REPORT_NAME = "mz_common_gain_mz_reverse_repair.json"
MAX_SECONDS, WORKERS = 3600, 4
DIRECTION, MODEL = 1, "MZ"


def runtime_specification(original):
    if original["optimizer"]["maximum_sweeps"] != 500:
        raise ValueError("Preserve the original500 scientific implementation")
    runtime = deepcopy(original)
    runtime["optimizer"]["maximum_sweeps"] = 1500
    return runtime


def numerical_override():
    return {"parameter": "optimizer.maximum_sweeps", "original_value": 500, "runtime_value": 1500,
        "scope": "one_fresh_common_gain_direction1_MZ_fit",
        "likelihood_priors_support_starts_tolerances_and_first_convergence_changed": False,
        "nonconverged_optimizer_state_reused": False,
        "parallel_arithmetic": "fixed_four_workers_sorted_collection_original_serial_population_updates_and_objective_reduction"}


def _validated_contexts(failed, source, primary, spec):
    if primary != source["participant_scores"]:
        raise ValueError("Use the original source score last bits as participant identity")
    original_failed = json.loads(previous.failed_path(DIRECTION).read_text())
    original_context = previous.repair_context(original_failed, source, primary, spec, DIRECTION)
    _, panel, support = previous_pool.validate_component(failed, DIRECTION, MODEL, FAILED_JOB,
                                                        spec, primary, source, original_context)
    if (panel is not None or failed.get("status") != "technical_common_gain_failure"
            or failed.get("error") != previous.NONCONVERGENCE or failed.get("finished_utc") != FAILED_FINISHED
            or failed.get("elapsed_seconds") != FAILED_PROCESS_SECONDS or failed.get("internal_seconds") != 4200
            or failed.get("source_context_validated") is not True or failed.get("source_baseline_validated") is not True
            or failed.get("saved_primary_scores_replayed") is not True or failed.get("fresh_fit_calls") != 1
            or len(failed.get("fits", [])) != 1 or failed.get("scored_fits") != []
            or failed.get("completed_model_scores") != {}):
        raise ValueError("Require exact terminal46615238 dir1/MZ original500 nonconvergence with no scored prefix")
    fit = failed["fits"][0]
    names = ["kappa", "log_a", "nondecision_logit", "rho"]
    if (fit.get("converged") is not False or fit.get("subject_optimizer_failures") != []
            or fit.get("parameter_names") != names or fit.get("version_order") != ["EI", "ER"]
            or fit.get("inference") != spec["method"] or fit.get("elapsed_seconds") != FAILED_FIT_SECONDS):
        raise ValueError("Only the diagnosed finite original500 MZ failure is eligible")
    values = np.asarray(fit.get("parameters"), dtype=float)
    if values.shape != (108, 4) or not np.isfinite(values).all():
        raise ValueError("Preserve the full finite failed MZ parameter panel")
    for column, name in enumerate(names):
        low, high = spec["parameters"][name]["bounds"]
        if np.any(values[:, column] < low) or np.any(values[:, column] > high):
            raise ValueError("Failed parameters left the original scientific support")
    for field, shape in (("global_mean", (4,)), ("version_means", (2, 4)), ("version_standard_deviations", (2, 4))):
        values = np.asarray(fit.get(field), dtype=float)
        if (values.shape != shape or not np.isfinite(values).all()
                or field == "version_standard_deviations" and np.any(values <= 0)):
            raise ValueError("Preserve complete finite failed hyperparameter evidence")
    history = np.asarray(fit.get("objective_history"), dtype=float)
    if history.shape != (500,) or not np.isfinite(history).all() or history[-2:].tolist() != FAILED_TAIL:
        raise ValueError("Retain the exact500-sweep history and diagnosed final objective changes")
    gap = float(abs(history[-1] - history[-2]))
    threshold = spec["optimizer"]["joint_relative_tolerance"] * (1 + abs(float(history[-2])))
    if not gap > threshold:
        raise ValueError("The failed fit must still fail the original first-convergence criterion")
    contacts = fit.get("support_contacts")
    expected = {(version, name) for version in ("EI", "ER") for name in ("kappa", "rho")}
    if (not isinstance(contacts, list) or len(contacts) != 4
            or {(row.get("version"), row.get("parameter")) for row in contacts} != expected):
        raise ValueError("Retain all four original EI/ER kappa/rho lower scale contacts")
    for contact in contacts:
        low, high = spec["parameters"][contact["parameter"]]["individual_log_scale_bounds"]
        value = contact.get("value")
        if (contact.get("level") != "version_log_standard_deviation" or contact.get("bounds") != [low, high]
                or type(value) not in (int, float) or not math.isfinite(value)
                or not low <= value <= low + 1e-6 * (1 + high - low)):
            raise ValueError("A preserved contact must remain at its declared original lower scale bound")
    return {"kind": "common_gain_direction1_MZ_original500_exhaustion_fresh1500",
        "failed_report": str(FAILED_REPORT), "failed_job_id": FAILED_JOB,
        "failed_started_utc": failed["started_utc"], "failed_finished_utc": FAILED_FINISHED,
        "failed_error": failed["error"], "previous_process_elapsed_seconds": failed["elapsed_seconds"],
        "previous_fit_elapsed_seconds": fit["elapsed_seconds"], "previous_support_contacts": deepcopy(contacts),
        "source_job_id": source["job_id"], "source_finished_utc": source["finished_utc"],
        "retained_support": deepcopy(support), "original_maximum_sweeps": 500,
        "final_objective_change": gap, "unchanged_convergence_threshold": threshold,
        "nonconverged_checkpoint_state_reused": False,
        "initialization": "unchanged_fresh_prior_centers_and_recorded_first_sweep_offsets",
        "cost_accounting": "This exact failed MZ process and the original shared M0 direction failure remain separate prior references; neither is recharged as new work."}, original_context


def repair_context(failed, source, primary, spec):
    """Reconstruct both prior lineages; only the new MZ context is returned."""
    return _validated_contexts(failed, source, primary, spec)[0]


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"Common-gain parallel MZ repair interrupted by signal {signum}; unfinished work is technical")


def run(args):
    require_compute()
    if args.source_dir.resolve() != SOURCE.resolve() or args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use only the fixed original MZ0 source and implementation")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("The whole repair process is capped at3600 seconds")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "4" or any(os.environ.get(name) != "1" for name in parallel.THREAD_ENV):
        raise ValueError("Require four actually allocated CPUs and all six numerical thread limits of one")
    output = args.output_dir.resolve()
    protected = (SOURCE, previous.FAILED_ROOT, FAILED_REPORT.parents[2])
    if output.exists() or any(output == path.resolve() or path.resolve() in output.parents for path in protected):
        raise FileExistsError("Use a fresh output outside all preserved source/failed attempts")
    output.mkdir(parents=True)
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds
    train, held = common.selected_runs(DIRECTION)

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process deadline exceeded")

    report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "generating_model": "MZ", "replicate_index": 0, "seed": common.shared.SEED,
        "direction": DIRECTION, "fitted_model": MODEL, "fitted_models": [MODEL], "rival": "fixed",
        "training_runs": list(train), "scoring_runs": list(held), "gain_range_dollars": [10, 20],
        "filter_boundary_inclusive": True, "score_normalization": common.NORMALIZATION,
        "source_dir": str(SOURCE), "source_report": str(SOURCE / common.SOURCE_REPORT),
        "failed_report": str(FAILED_REPORT), "source_context_validated": False,
        "source_baseline_validated": False, "saved_primary_scores_replayed": False,
        "prior_MZ_failed_context": None, "original_M0_failed_context": None,
        "original_implementation": None, "runtime_implementation": None,
        "original_maximum_sweeps": 500, "fitted_maximum_sweeps": 1500, "numerical_override": numerical_override(),
        "workers": WORKERS, "allocated_cpus": 4, "parallel_execution": True,
        "parallel_fit_max_seconds": None, "parallel_cleanup_reserve_seconds": parallel.CLEANUP_SECONDS + 1.,
        "parallel_progress": None, "parallel_completed_sweeps": [], "parallel_failure": None,
        "parallel_workers_cleaned_up": None, "fresh_fit_calls": 0,
        "job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat(),
        "internal_seconds": args.max_seconds, "status": "running", "common_gain_component_complete": False,
        "simulation_regenerated": False, "simulation_copied": False, "saved_primary_fits_reused_for_common_gain": False,
        "observed_choices_or_RTs_used": False, "selected_replicates": [0], "wave_selected": False,
        "retry_wave_selected": False, "automatic_retry": False, "task_versions_are_independent_replications": False,
        "full_recovery_pass": None, "episode_terminal": False,
        **{field: "not_evaluated" for field in common.CLAIM_FIELDS},
        "cost_accounting": "elapsed_seconds is this new four-CPU process's wall duration only. Prior MZ and shared M0 failure processes remain distinct references, counted once by the coordinator/pool. Do not add per-worker timing or recharge old parent allocations.",
        "interpretation": "Fresh common-gain dir1/MZ runtime1500 repair after its exact original500 exhaustion. Four participant workers retain original sorted population updates and objective addition order. All108 original participants and selected rows remain; no failed state or full-range fit is reused. Original500 source replay and held-out likelihood remain unchanged. No family, recovery, robustness, interval or gate verdict.",
        "fits": [], "scored_fits": [], "completed_model_scores": {}}

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
        spec = json.loads(args.implementation.read_text())
        runtime = runtime_specification(spec)
        report.update(implementation=spec, original_implementation=spec, runtime_implementation=runtime)
        source = json.loads((SOURCE / common.SOURCE_REPORT).read_text())
        truth = json.loads((SOURCE / "generating_parameters.json").read_text())
        events = SOURCE / "simulated_events.csv.gz"
        people = common.shared.load_cohort(events, check)
        common.shared.validate_source(source, truth, people, spec)
        check()
        report.update(source_baseline_validated=True, source_job_id=source["job_id"], source_finished_utc=source["finished_utc"],
            participants=108, runs=432, trials=27648)
        checkpoint()
        splits = [common.load_split(events, list(a), list(b)) for a, b in common.HALVES]
        primary, discrepancy = common.shared.replay_saved_scores(source, splits, spec, check)
        report.update(saved_primary_scores_replayed=True, saved_primary_participant_scores=primary,
            saved_score_replay_max_absolute_discrepancy=discrepancy,
            saved_score_replay_absolute_tolerance=common.shared.REPLAY_ABSOLUTE_TOLERANCE)
        failed = json.loads(FAILED_REPORT.read_text())
        mz_context, m0_context = _validated_contexts(failed, source, primary, spec)
        check()
        report.update(source_context_validated=True, prior_MZ_failed_context=mz_context, original_M0_failed_context=m0_context)
        training, scoring, support = common.filtered_panels(people, DIRECTION)
        if support != mz_context["retained_support"]:
            raise ValueError("Actual newly filtered source support must equal the exact failed MZ support")
        report["retained_support"] = support
        checkpoint()
        check()
        # Calculate after the last pre-fit publication, so its I/O cannot
        # consume the parallel fitter's separate bounded-cleanup reserve.
        remaining = deadline - time.monotonic() - report["parallel_cleanup_reserve_seconds"]
        if remaining <= 0:
            raise RuntimeError("Insufficient whole-process time remains for bounded worker cleanup")
        report["parallel_fit_max_seconds"] = remaining
        fit_started = time.perf_counter()
        report["fresh_fit_calls"] = 1
        try:
            fit = parallel.fit_partially_pooled_parallel(training, MODEL, "fixed", runtime,
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
        common.validate_fit(fit, MODEL, train, held, runtime, people)
        scores = common.score_filtered(fit, scoring, spec, check)
        check()
        report["completed_model_scores"][MODEL] = scores
        report["scored_fits"].append({"model": MODEL, "training_runs": list(train), "scoring_runs": list(held)})
        report["participant_scores"] = {person: {"version": support[person]["version"],
            "heldout_trials": value["trials"], "MZ_log_sum": value["log_sum"]} for person, value in scores.items()}
        report.update(status="common_gain_component_completed", common_gain_component_complete=True,
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
            report.update(status="technical_common_gain_failure", common_gain_component_complete=False,
                finished_utc=datetime.now(timezone.utc).isoformat())
            checkpoint()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "status", "common_gain_component_complete",
        "fresh_fit_calls", "parallel_workers_cleaned_up", "elapsed_seconds", "error") if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["common_gain_component_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
