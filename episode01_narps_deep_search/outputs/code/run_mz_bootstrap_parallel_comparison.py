#!/usr/bin/env python3
"""Repeat preserved bootstrap0 with four workers for arithmetic/cost comparison.

The completed serial draw remains the sole statistical bootstrap0. This is
neither another draw nor an interval-method/count selection. No serial fit is
rerun. Cross-hardware timing observations are not a controlled speedup.
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
import socket
import time

import numpy as np

import hierarchical_behavior_parallel as parallel
from qualify_assets import require_compute
from run_single_process_recovery import HALVES, IMPLEMENTATION, validate_fit


EPISODE = Path(__file__).resolve().parents[2]
SOURCE_REPORT = EPISODE / "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-replicate000-attempt001/single_process_recovery.json"
SERIAL_REPORT = EPISODE / "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-closeout-pilot-46547708/bootstrap000/mz_bootstrap_pilot.json"
SERIAL_FINISHED = "2026-10-05T02:00:35.130180+00:00"
REPORT_NAME = "mz_bootstrap_parallel_comparison.json"
PHASE = "MZ_SESOI_bootstrap0_parallel_engineering_comparison"
MAX_SECONDS, WORKERS = 900, 4
ABSOLUTE_TOLERANCE = 1e-12
NUMERIC_FIELDS = ("parameters", "global_mean", "version_means", "version_standard_deviations", "objective_history")


def finite_seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Retain finite nonnegative recorded process/fit costs")
    return value


def validate_serial(reference, source, halves, spec, legacy):
    """Admit the exact terminal draw against the actual original whole people."""
    if (reference.get("phase") != "MZ_SESOI_participant_bootstrap_resource_pilot"
            or reference.get("status") != "paired_refit_draw_completed"
            or reference.get("job_id") != "46547708" or reference.get("finished_utc") != SERIAL_FINISHED
            or reference.get("source_report") != str(SOURCE_REPORT)
            or reference.get("source_events") != str(SOURCE_REPORT.parent / "simulated_events.csv.gz")
            or reference.get("source_replicate_index") != 0 or reference.get("source_seed") != 910031
            or reference.get("bootstrap_index") != 0
            or reference.get("seed_sequence_components") != [1200037, 910031, 0]
            or reference.get("implementation") != spec or spec["optimizer"]["maximum_sweeps"] != 500
            or reference.get("bootstrap_draw_complete") is not True
            or reference.get("same_draw_in_both_halves") is not True
            or reference.get("bootstrap_participants") != 108
            or reference.get("bootstrap_draw_count_selected") != 1
            or reference.get("interval_bootstrap_count_selected") is not None
            or reference.get("interval_convention_selected") is not None
            or reference.get("full_recovery_gate") != "not_evaluated"
            or any(reference.get(field) is not False for field in (
                "observed_choices_or_RTs_used", "new_responses_simulated", "heldout_scoring_performed",
                "source_files_modified", "interval_coverage_evaluated", "group_interval_coverage_qualified"))):
        raise ValueError("Require the exact completed serial bootstrap0 and its unchanged scope/specification")
    versions = {person: record["version"] for person, record in source["participant_scores"].items()}
    legacy.validate_source_halves(halves, versions)
    # Reconstruct only to verify identity; the saved serial membership drives fits.
    draws = reference.get("participant_draws")
    if draws != legacy.paired_draw(versions):
        raise ValueError("The saved bootstrap0 membership/order must match its fixed original seed")
    unique = {version: len({row["source_participant"] for row in draws if row["task_version"] == version})
              for version in ("EI", "ER")}
    if reference.get("unique_source_participants_by_version") != unique:
        raise ValueError("Serial distinct-person counts disagree with preserved membership")
    fits = reference.get("fits")
    if not isinstance(fits, list) or len(fits) != 2:
        raise ValueError("Both completed original serial MZ fits are required")
    for fit, half, (train, held) in zip(fits, halves, HALVES):
        training = legacy.resampled_half(half, draws)
        validate_fit(fit, "MZ", train, held, spec, training)
        if (fit.get("group_interval_coverage_qualified") is not False
                or fit.get("individual_rank_recovery_qualified") is not False):
            raise ValueError("Serial point fits cannot qualify intervals or individual recovery")
        finite_seconds(fit.get("elapsed_seconds"))
    finite_seconds(reference.get("elapsed_seconds"))
    if reference.get("total_fit_seconds") != math.fsum(fit["elapsed_seconds"] for fit in fits):
        raise ValueError("Serial total fit time must retain its two recorded component durations")
    if reference.get("physical_rejection_displacement") != legacy.physical_group_effects(fits, draws):
        raise ValueError("Serial physical estimand must match its preserved fitted people and both halves")
    return deepcopy(draws)


def numeric_comparison(actual, expected):
    left, right = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Only finite completed estimates can enter the engineering comparison")
    same_shape = left.shape == right.shape
    maximum = float(np.max(np.abs(left - right))) if same_shape and left.size else (0. if same_shape else None)
    return {"parallel_shape": list(left.shape), "serial_shape": list(right.shape),
            "same_shape": same_shape, "exactly_equal": bool(same_shape and np.array_equal(left, right)),
            "max_absolute_difference": maximum,
            "parallel_minus_serial": float(left - right) if same_shape and left.ndim == 0 else None,
            "within_absolute_tolerance": bool(same_shape and maximum <= ABSOLUTE_TOLERANCE)}


def compare_results(fits, reference, effects):
    components = []
    for direction, (fit, old) in enumerate(zip(fits, reference["fits"])):
        metrics = {field: numeric_comparison(fit[field], old[field]) for field in NUMERIC_FIELDS}
        metrics["final_objective"] = numeric_comparison(fit["objective_history"][-1], old["objective_history"][-1])
        components.append({"direction": direction, "training_runs": fit["training_runs"],
            "serial_sweeps": len(old["objective_history"]), "parallel_sweeps": len(fit["objective_history"]),
            "numeric_fields": metrics, "support_contacts_exactly_equal": fit["support_contacts"] == old["support_contacts"],
            "scientific_fit_dictionary_exactly_equal_excluding_elapsed_seconds":
                {key: value for key, value in fit.items() if key != "elapsed_seconds"} ==
                {key: value for key, value in old.items() if key != "elapsed_seconds"}})
    effect_metrics = {version: numeric_comparison(effects[version], reference["physical_rejection_displacement"][version])
                      for version in ("EI", "ER")}
    metrics = [item for component in components for item in component["numeric_fields"].values()] + list(effect_metrics.values())
    return {"absolute_tolerance": ABSOLUTE_TOLERANCE, "relative_tolerance": 0.,
        "criterion_scope": "prospectively selected engineering description only; not a scientific margin or qualified equivalence",
        "fits": components, "physical_effects": effect_metrics,
        "all_numeric_values_exactly_equal": all(item["exactly_equal"] for item in metrics),
        "engineering_agreement_at_absolute_1e_minus12": all(item["within_absolute_tolerance"] for item in metrics)}


def write_report(output, report):
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if len(payload.encode("utf-8")) > 20 * 1024 * 1024:
        raise ValueError("Comparison checkpoint exceeds the existing20MiB pilot envelope")
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(payload)
    temporary.replace(output / REPORT_NAME)


def run(args):
    require_compute()
    if (args.source_report.resolve() != SOURCE_REPORT.resolve()
            or args.serial_report.resolve() != SERIAL_REPORT.resolve()
            or args.implementation.resolve() != IMPLEMENTATION.resolve()):
        raise ValueError("Use only original MZ0, completed serial bootstrap0 and the original implementation")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("The explicit whole-process limit must be within1..900 seconds")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "4" or any(os.environ.get(name) != "1" for name in parallel.THREAD_ENV):
        raise ValueError("Require four allocated CPUs and all six numerical thread limits of one")
    output = args.output_dir.resolve()
    if (output.exists() or not output.is_relative_to((EPISODE / "outputs/runs").resolve())
            or any(output == parent.resolve() or parent.resolve() in output.parents
                   for parent in (SOURCE_REPORT.parent, SERIAL_REPORT.parent))):
        raise FileExistsError("Use a fresh episode run outside both preserved input attempts")
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds
    output.mkdir(parents=True)

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process comparison deadline exceeded")

    report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "source_report": str(SOURCE_REPORT), "serial_report": str(SERIAL_REPORT),
        "bootstrap_index": 0, "source_replicate_index": 0, "source_seed": 910031,
        "seed_sequence_components": [1200037, 910031, 0], "source_baseline_validated": False,
        "serial_reference_validated": False, "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "internal_seconds": args.max_seconds, "workers": WORKERS, "allocated_cpus": 4,
        "thread_environment": {name: os.environ[name] for name in parallel.THREAD_ENV},
        "original_maximum_sweeps": 500, "parallel_cleanup_reserve_seconds": parallel.CLEANUP_SECONDS + 1.,
        "status": "running", "comparison_complete": False, "fresh_fit_calls": 0,
        "active_fit": None, "parallel_progress": None, "parallel_failure": None,
        "parallel_workers_cleaned_up": None, "parallel_completed_sweeps": [], "fit_budgets": [],
        "fits": [], "validated_fit_directions": [], "new_statistical_draws": 0,
        "serial_bootstrap0_remains_statistical_result": True, "interval_bootstrap_count_selected": None,
        "interval_convention_selected": None, "interval_coverage_evaluated": False,
        "group_interval_coverage_qualified": False, "full_recovery_gate": "not_evaluated",
        "scientific_equivalence_qualified": False, "controlled_speedup_established": False,
        "automatic_core_hour_savings_claimed": False, "automatic_retry": False,
        "observed_choices_or_RTs_used": False, "new_responses_simulated": False,
        "heldout_scoring_performed": False, "source_files_modified": False,
        "serial_fit_rerun": False, "saved_fit_parameters_reused": False,
        "interpretation": "One engineering repeat of the identical preserved bootstrap0. Completed finite fits remain a completed comparison even if arithmetic agreement is false. No new statistical draw, interval, coverage or gate verdict.",
        "cost_accounting": "Serial and new process/fit times are separate observations on potentially different hardware. Four allocated CPUs do not imply fourfold savings. Root accounts actual allocated cost once; old serial cost is not recharged."}

    def checkpoint():
        report["elapsed_seconds"] = time.perf_counter() - started
        write_report(output, report)

    def progress(event):
        check()
        report["parallel_progress"] = deepcopy(event)
        if event["stage"] == "sweep":
            report["parallel_completed_sweeps"].append({"direction": report["active_fit"]["direction"], **deepcopy(event)})
            checkpoint()

    def interrupted(signum, frame):
        raise RuntimeError(f"Parallel comparison interrupted by signal {signum}; unfinished work is technical")

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    succeeded = False
    try:
        checkpoint()
        check()
        # The legacy CLI sets thread limits at import. Import only AFTER the real
        # caller environment passes the guard above; no resource guard is spoofed.
        import run_mz_bootstrap_pilot as legacy
        spec = json.loads(args.implementation.read_text())
        report["implementation"] = spec
        source = json.loads(args.source_report.read_text())
        summary = legacy.summarize_group_reports([(args.source_report, source)], spec)
        if (source.get("replicate_index") != 0 or source.get("seed") != 910031
                or summary["valid_replicates"] != 1 or source.get("valid_recovery_component") is not True):
            raise ValueError("Require the completed original MZ0 source")
        report["source_baseline_validated"] = True
        report["source_finished_utc"] = source["finished_utc"]
        check()
        halves = legacy.load_split(SOURCE_REPORT.parent / "simulated_events.csv.gz", [1, 4], [2, 3])
        reference = json.loads(args.serial_report.read_text())
        draws = validate_serial(reference, source, halves, spec, legacy)
        report.update(serial_reference_validated=True, participant_draws=draws,
            same_draw_in_both_halves=True, bootstrap_participants=108,
            unique_source_participants_by_version=reference["unique_source_participants_by_version"],
            serial_finished_utc=reference["finished_utc"], serial_job_id=reference["job_id"],
            serial_process_elapsed_seconds=reference["elapsed_seconds"],
            serial_fit_elapsed_seconds=[fit["elapsed_seconds"] for fit in reference["fits"]],
            serial_physical_rejection_displacement=reference["physical_rejection_displacement"])
        checkpoint()
        for direction, (half, (train, held)) in enumerate(zip(halves, HALVES)):
            check()
            training = legacy.resampled_half(half, draws)
            report.update(active_fit={"direction": direction, "model": "MZ",
                "training_runs": list(train), "scoring_runs": list(held)},
                parallel_progress=None, parallel_workers_cleaned_up=None)
            checkpoint()
            check()
            remaining = deadline - time.monotonic() - report["parallel_cleanup_reserve_seconds"]
            if remaining <= 0:
                raise RuntimeError("Insufficient time remains for bounded participant-worker cleanup")
            report["fit_budgets"].append({"direction": direction, "max_seconds": remaining})
            report["fresh_fit_calls"] += 1
            fit_started = time.perf_counter()
            try:
                fit = parallel.fit_partially_pooled_parallel(training, "MZ", "fixed", spec,
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
            validate_fit(fit, "MZ", train, held, spec, training)
            if (fit.get("group_interval_coverage_qualified") is not False
                    or fit.get("individual_rank_recovery_qualified") is not False):
                raise ValueError("New point fits cannot qualify intervals or individual recovery")
            report["validated_fit_directions"].append(direction)
            checkpoint()
        check()
        effects = legacy.physical_group_effects(report["fits"], draws)
        comparison = compare_results(report["fits"], reference, effects)
        check()
        report.update(physical_rejection_displacement=effects, comparison=comparison,
            total_fit_seconds=math.fsum(fit["elapsed_seconds"] for fit in report["fits"]),
            active_fit=None, status="parallel_comparison_completed", comparison_complete=True,
            finished_utc=datetime.now(timezone.utc).isoformat())
        checkpoint()
        check()
        succeeded = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        signal.alarm(0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if succeeded:
            try:
                check()
            except BaseException as error:
                report["error"] = f"{type(error).__name__}: {error}"
                succeeded = False
        if not succeeded:
            report.pop("comparison", None)
            report.pop("physical_rejection_displacement", None)
            report.update(status="technical_parallel_comparison_failure", comparison_complete=False,
                finished_utc=datetime.now(timezone.utc).isoformat())
            checkpoint()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--serial-report", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("status", "comparison_complete", "fresh_fit_calls",
        "elapsed_seconds", "comparison", "error") if key in report}, indent=2), flush=True)
    return 0 if report["comparison_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
