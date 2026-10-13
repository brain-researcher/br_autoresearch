#!/usr/bin/env python3
"""One registered offer-sequence omission refit on the saved synthetic MZ0.

Omit the same complete version-specific group from training and scoring.
All retained subject and population parameters are fitted afresh, separately
on each primary training half. This is deletion sensitivity, not prediction
for unseen participants or a template-cluster interval qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np

from qualify_assets import require_compute
from hierarchical_behavior import parameter_names
import hierarchical_behavior_parallel as parallel
import run_single_process_rivals as shared
import summarize_mz0_offer_sequence_influence as influence


SOURCE, IMPLEMENTATION = shared.SOURCE, shared.IMPLEMENTATION
SOURCE_REPORT = "single_process_recovery.json"
PHASE = "MZ0_leave_one_offer_sequence_out_refit"
REPORT_NAME = "mz0_offer_sequence_refit.json"
MAX_SECONDS, WORKERS = 5400, 4
CLAIMS = ("cluster_intervals", "template_generality", "unseen_template_prediction",
          "family_selection", "recovery_rates", "group_interval_coverage", "full_gate")


def read_membership():
    schedule = json.loads(influence.SCHEDULE.read_text())
    with influence.MEMBERSHIP.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["participant", "task_version", "offer_sequence_id"]:
            raise ValueError("Use the unchanged qualified three-column membership")
        return schedule, list(reader)


def retained_cohort(source, schedule, members, people, omitted_group):
    if omitted_group not in influence.GROUP_IDS:
        raise ValueError("Select one of the fixed16 version-specific offer groups")
    _, versions, groups = influence.validate_inputs(source, schedule, members)
    if set(people) != set(versions) or any(
            rows[0]["task_version"] != versions[person] for person, rows in people.items()):
        raise ValueError("Qualified membership must match the actual saved cohort")
    omitted = sorted(groups[omitted_group])
    retained = {person: people[person] for person in sorted(people) if person not in omitted}
    counts = Counter(versions[person] for person in retained)
    if set(counts) != {"EI", "ER"} or len(retained) + len(omitted) != 108:
        raise ValueError("The exact omitted-group complement must retain both task versions")
    return retained, omitted, dict(counts)


def retained_split(training, scoring, retained, train, held):
    """Filter both panels identically; retain every original slot for each person."""
    panels = []
    for original, runs in ((training, train), (scoring, held)):
        panel = {person: original[person] for person in retained}
        expected_slots = {(run, trial) for run in runs for trial in range(1, 65)}
        for person, item in panel.items():
            rows = item["rows"]
            if (item["version"] != retained[person][0]["task_version"] or len(rows) != 128
                    or {(int(row["run"]), int(row["trial_order"])) for row in rows} != expected_slots
                    or any(row["participant"] != person or row["task_version"] != item["version"] for row in rows)):
                raise ValueError("Retained halves require the exact128 original run/trial slots and versions")
        panels.append(panel)
    return tuple(panels)


def validate_retained_fit(fit, model, train, held, spec, retained):
    """Original fixed-fit criteria, with the declared retained participant count."""
    names = parameter_names(model, "fixed")
    if (fit.get("model") != model or fit.get("rival") != "fixed"
            or fit.get("training_runs") != list(train) or fit.get("scoring_runs") != list(held)
            or fit.get("participants") != sorted(retained) or fit.get("parameter_names") != names
            or fit.get("version_order") != ["EI", "ER"] or fit.get("inference") != spec["method"]):
        raise ValueError("Retained fit model, split, participant or parameter identity changed")
    if fit.get("converged") is not True or fit.get("subject_optimizer_failures") != []:
        raise RuntimeError("Nonconverged retained fit is technical; it is not sensitivity failure")
    theta = np.asarray(fit.get("parameters"), dtype=float)
    if theta.shape != (len(retained), len(names)) or not np.isfinite(theta).all():
        raise ValueError("Incomplete or nonfinite retained participant parameters")
    for column, name in enumerate(names):
        low, high = spec["parameters"][name]["bounds"]
        if np.any(theta[:, column] < low) or np.any(theta[:, column] > high):
            raise ValueError("Fitted parameter outside unchanged support")
    for field, shape in (("global_mean", (len(names),)), ("version_means", (2, len(names))),
                         ("version_standard_deviations", (2, len(names)))):
        values = np.asarray(fit.get(field), dtype=float)
        if (values.shape != shape or not np.isfinite(values).all()
                or field == "version_standard_deviations" and np.any(values <= 0)):
            raise ValueError("Incomplete or invalid retained hyperparameters")
    history = np.asarray(fit.get("objective_history"), dtype=float)
    if (history.ndim != 1 or not 2 <= len(history) <= spec["optimizer"]["maximum_sweeps"]
            or not np.isfinite(history).all()
            or abs(history[-1] - history[-2]) > spec["optimizer"]["joint_relative_tolerance"] * (1 + abs(history[-2]))):
        raise ValueError("Retained fit lacks unchanged first-convergence evidence")
    if not isinstance(fit.get("support_contacts"), list):
        raise ValueError("Retain numeric-support diagnostics")


def paired_diagnostic(scores, retained):
    if set(scores) != set(retained):
        raise ValueError("Score panel must equal the declared retained complement")
    people = sorted(retained)
    for person in people:
        row = scores[person]
        if (set(row) != {"version", "M0", "MZ"} or row["version"] != retained[person][0]["task_version"]
                or any(type(row[key]) not in (int, float) or not math.isfinite(row[key]) for key in ("M0", "MZ"))):
            raise ValueError("Retained scores must be finite and version matched")
    differences = np.array([scores[person]["MZ"] - scores[person]["M0"] for person in people])
    mean = float(differences.mean())
    se = float(differences.std(ddof=1) / np.sqrt(len(people)))
    if not np.isfinite(differences).all() or not math.isfinite(mean) or not math.isfinite(se):
        raise ValueError("Nonfinite retained paired arithmetic")
    versions = np.array([scores[person]["version"] for person in people])
    version_means = {version: float(differences[versions == version].mean()) for version in ("EI", "ER")}
    return {"participants": len(people), "participants_by_version": dict(Counter(versions)),
        "mean_gain_nats_per_trial": mean, "participant_SE": se,
        "participant_median": float(np.median(differences)), "version_means": version_means,
        "version_mean_positive": {version: value > 0 for version, value in version_means.items()},
        "requirements": {"mean_gain_at_least_0p01": mean >= .01,
                         "mean_gain_greater_than_two_participant_SE": mean > 2 * se},
        "interpretation": "Equal retained-person descriptive paired contrast; participant SE is not a template-cluster interval or a generality verdict."}


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def run(args):
    require_compute()
    if (args.source_dir.resolve() != SOURCE.resolve() or args.implementation.resolve() != IMPLEMENTATION.resolve()
            or args.omitted_group not in influence.GROUP_IDS):
        raise ValueError("Use the original MZ0 source/spec and one qualified offer-group ID")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("Whole refit cap must be within1..5400 seconds")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "4" or any(os.environ.get(name) != "1" for name in parallel.THREAD_ENV):
        raise ValueError("Require four allocated CPUs and all six numerical thread limits of one")
    output = args.output_dir.resolve()
    if (output.exists() or not output.is_relative_to((shared.EPISODE / "outputs/runs").resolve())
            or any(output == path.resolve() or path.resolve() in output.parents for path in (SOURCE, influence.SCHEDULE.parent))):
        raise FileExistsError("Use a fresh episode run outside the preserved source and schedule attempts")
    output.mkdir(parents=True)
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process deadline exceeded")

    report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "generating_model": "MZ", "replicate_index": 0, "seed": shared.SEED,
        "omitted_offer_sequence_id": args.omitted_group, "registered_offer_sequence_ids": list(influence.GROUP_IDS),
        "source_dir": str(SOURCE), "source_report": str(SOURCE / SOURCE_REPORT),
        "schedule_report": str(influence.SCHEDULE), "membership_table": str(influence.MEMBERSHIP),
        "source_baseline_validated": False, "saved_primary_scores_replayed": False, "membership_validated": False,
        "job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat(),
        "internal_seconds": args.max_seconds, "status": "running", "offer_sequence_refit_complete": False,
        "workers": WORKERS, "allocated_cpus": 4, "original_maximum_sweeps": 500,
        "parallel_cleanup_reserve_seconds": parallel.CLEANUP_SECONDS + 1., "fresh_fit_calls": 0,
        "active_fit": None, "parallel_progress": None, "parallel_failure": None,
        "parallel_workers_cleaned_up": None, "parallel_completed_sweeps": [], "fit_budgets": [],
        "fits": [], "completed_components": [], "scored_fits": [],
        "simulation_regenerated": False, "saved_fit_parameters_reused": False, "observed_choices_or_RTs_used": False,
        "automatic_retry": False, "automatic_ceiling_increase": False, "subsequent_omissions_selected": False,
        "task_versions_are_independent_replications": False, "full_recovery_pass": None, "episode_terminal": False,
        **{field: "not_evaluated" for field in CLAIMS},
        "score_normalization": "each128_trial_half_log_sum_divided_by256_then_sum_both_halves_per_retained_person",
        "interpretation": "Delete the complete named offer group from fit and score support; refit all remaining subject and population parameters on training runs only. Equal retained-person weights, distinct EI/ER IDs, original offers/onsets/responses unchanged. One synthetic-cohort deletion sensitivity, not unseen-template prediction or an independent replication.",
        "cost_accounting": "elapsed_seconds is this new process wall time; fit times are component costs within it. Prior source fits are preserved references, not recharged. Coordinator accounts allocated CPU cost once."}

    def checkpoint():
        report["elapsed_seconds"] = time.perf_counter() - started
        write_report(output, report)

    def progress(event):
        check()
        report["parallel_progress"] = deepcopy(event)
        if event["stage"] == "sweep":
            report["parallel_completed_sweeps"].append({"fit": deepcopy(report["active_fit"]), **deepcopy(event)})
            checkpoint()

    def interrupted(signum, frame):
        raise RuntimeError(f"Offer-sequence refit interrupted by signal {signum}; unfinished work is technical")

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    succeeded = False
    try:
        checkpoint()
        check()
        spec = json.loads(args.implementation.read_text())
        if spec["optimizer"]["maximum_sweeps"] != 500:
            raise ValueError("All four new fits preserve the original500 implementation")
        report["implementation"] = spec
        source = json.loads((SOURCE / SOURCE_REPORT).read_text())
        truth = json.loads((SOURCE / "generating_parameters.json").read_text())
        events = SOURCE / "simulated_events.csv.gz"
        people = shared.load_cohort(events, check)
        shared.validate_source(source, truth, people, spec)
        check()
        report.update(source_baseline_validated=True, source_job_id=source["job_id"],
            source_finished_utc=source["finished_utc"], original_participants=108,
            source_fit_elapsed_seconds=[fit.get("elapsed_seconds") for fit in source["fits"]],
            source_process_elapsed_seconds=source.get("elapsed_seconds"))
        splits = [shared.load_split(events, list(train), list(held)) for train, held in shared.HALVES]
        primary, discrepancy = shared.replay_saved_scores(source, splits, spec, check)
        report.update(saved_primary_scores_replayed=True, saved_primary_participant_scores=primary,
            saved_score_replay_max_absolute_discrepancy=discrepancy,
            saved_score_replay_absolute_tolerance=shared.REPLAY_ABSOLUTE_TOLERANCE)
        schedule, members = read_membership()
        retained, omitted, counts = retained_cohort(source, schedule, members, people, args.omitted_group)
        report.update(membership_validated=True, omitted_participants=omitted,
            retained_participants=sorted(retained), participants=len(retained), participants_by_version=counts,
            heldout_trials_per_direction_per_participant=128, combined_scored_trials_per_participant=256,
            fixed_prediction_retained_diagnostic=paired_diagnostic({p: primary[p] for p in retained}, retained))
        checkpoint()
        scores = {person: {"version": rows[0]["task_version"], "M0": 0., "MZ": 0.} for person, rows in retained.items()}
        for direction, ((train, held), original_panels) in enumerate(zip(shared.HALVES, splits)):
            training, scoring = retained_split(*original_panels, retained, train, held)
            for model in ("M0", "MZ"):
                report["active_fit"] = {"direction": direction, "model": model,
                    "training_runs": list(train), "scoring_runs": list(held)}
                report["parallel_progress"] = None
                report["parallel_workers_cleaned_up"] = None
                checkpoint()
                check()
                remaining = deadline - time.monotonic() - report["parallel_cleanup_reserve_seconds"]
                if remaining <= 0:
                    raise RuntimeError("Insufficient time remains for bounded participant-worker cleanup")
                report["fit_budgets"].append({**report["active_fit"], "max_seconds": remaining})
                report["fresh_fit_calls"] += 1
                fit_started = time.perf_counter()
                try:
                    fit = parallel.fit_partially_pooled_parallel(training, model, "fixed", spec,
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
                validate_retained_fit(fit, model, train, held, spec, retained)
                contribution = shared.score_fit(fit, scoring, spec, check)
                check()
                if set(contribution) != set(retained):
                    raise ValueError("Every retained participant needs a held-out contribution")
                report["completed_components"].append({**report["active_fit"], "participant_score_contributions": contribution})
                for person, value in contribution.items():
                    scores[person][model] += value
                report["scored_fits"].append(deepcopy(report["active_fit"]))
                checkpoint()
        diagnostic = paired_diagnostic(scores, retained)
        report.update(participant_scores=scores, nested_diagnostic=diagnostic, active_fit=None,
            status="offer_sequence_refit_completed", offer_sequence_refit_complete=True,
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
            report.pop("participant_scores", None)
            report.pop("nested_diagnostic", None)
            report.update(status="technical_offer_sequence_refit_failure", offer_sequence_refit_complete=False,
                finished_utc=datetime.now(timezone.utc).isoformat())
            checkpoint()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omitted-group", choices=influence.GROUP_IDS, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("status", "omitted_offer_sequence_id", "offer_sequence_refit_complete",
        "fresh_fit_calls", "elapsed_seconds", "error") if key in report}, indent=2), flush=True)
    return 0 if report["offer_sequence_refit_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
