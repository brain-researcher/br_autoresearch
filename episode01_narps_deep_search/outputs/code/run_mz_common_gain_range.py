#!/usr/bin/env python3
"""One direction of the registered common-gain sensitivity on saved MZ0.

Fresh M0/MZ fits use only unchanged rows with10<=gain<=20. Held-out log
likelihoods remain sums with actual counts; only a separate complete two-
direction pool may form participant means or a nested diagnostic.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from qualify_assets import require_compute
import run_single_process_rivals as shared


SOURCE, IMPLEMENTATION, HALVES = shared.SOURCE, shared.IMPLEMENTATION, shared.HALVES
SOURCE_REPORT = "single_process_recovery.json"
MODELS, MAX_SECONDS = ("M0", "MZ"), 5400
PHASE = "MZ0_common_gain_10_20_direction_component"
REPORT_NAME = "mz_common_gain_range.json"
NORMALIZATION = "unnormalized_heldout_log_sums_with_actual_participant_counts"
CLAIM_FIELDS = ("family_selection", "family_recovery_rate", "false_selection_rate", "confusion_rate",
                "group_recovery", "group_intervals", "interval_coverage", "individual_recovery",
                "full_menu_selection", "full_gate", "early_kill", "curtailment", "common_gain_robustness",
                "template_robustness", "missingness_robustness")
fit_partially_pooled, load_split, validate_fit = shared.fit_partially_pooled, shared.load_split, shared.validate_fit


def selected_runs(direction):
    if type(direction) is not int or direction not in (0, 1):
        raise ValueError("Select exactly one of the two original complementary directions")
    return HALVES[direction]


def filtered_panels(people, direction):
    """Select original row objects without changing any response or design field."""
    train, held = selected_runs(direction)
    if (len(people) != 108 or Counter(rows[0]["task_version"] for rows in people.values()) != {"EI": 54, "ER": 54}):
        raise ValueError("Retain all108 original participants and both54-person versions")
    training, scoring, support = {}, {}, {}
    for person, rows in people.items():
        version = rows[0]["task_version"]
        selected = [row for row in rows if 10 <= float(row["gain"]) <= 20]
        expected = 96 if version == "EI" else 176
        if len(selected) != expected:
            raise ValueError("Filtered full-session counts must match the unchanged version-specific offer grid")
        panels = [[row for row in selected if int(row["run"]) in runs] for runs in (train, held)]
        slots = [[[int(row["run"]), int(row["trial_order"])] for row in panel] for panel in panels]
        if (not all(panels) or len(panels[0]) + len(panels[1]) != expected
                or len({tuple(slot) for group in slots for slot in group}) != expected):
            raise ValueError("Filtered train/held slots must be nonempty, disjoint and exhaust the original selected rows")
        training[person] = {"version": version, "rows": panels[0]}
        scoring[person] = {"version": version, "rows": panels[1]}
        support[person] = {"version": version, "full_session_trials": expected,
            "training_trials": len(panels[0]), "heldout_trials": len(panels[1]),
            "training_slots": slots[0], "heldout_slots": slots[1]}
    return training, scoring, support


def score_filtered(fit, scoring, spec, check=lambda: None):
    """Use the same fixed likelihood, retaining sums and actual row counts."""
    scores = {}
    for person, theta in zip(fit["participants"], fit["parameters"]):
        check()
        rows = scoring[person]["rows"]
        if not rows or any(not 10 <= float(row["gain"]) <= 20 for row in rows):
            raise ValueError("Common-gain scoring requires nonempty retained10-through20 rows")
        values = np.asarray(shared.trial_likelihood(theta, fit["parameter_names"], shared.make_data(rows), spec, "fixed"))
        if values.shape != (len(rows),) or not np.isfinite(values).all():
            raise ValueError("Each retained trial requires one finite held-out log likelihood")
        scores[person] = {"log_sum": float(values.sum()), "trials": len(rows)}
    return scores


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"Common-gain component interrupted by signal {signum}; unfinished work is technical")


def run(args):
    require_compute()
    train, held = selected_runs(args.direction)
    if args.source_dir.resolve() != SOURCE.resolve() or args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the fixed original MZ0 source and unchanged implementation")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("The whole-process cap must be1 through5400 seconds")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1" or any(os.environ.get(name) != "1" for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("Exactly one allocated CPU and all four numerical thread limits of one are required")
    output = args.output_dir.resolve()
    if output.exists() or output == SOURCE.resolve() or SOURCE.resolve() in output.parents:
        raise FileExistsError("Use a fresh output outside the preserved original source")
    output.mkdir(parents=True)
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process deadline exceeded")

    report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "generating_model": "MZ", "replicate_index": 0, "seed": shared.SEED, "direction": args.direction,
        "training_runs": list(train), "scoring_runs": list(held), "fitted_models": list(MODELS), "rival": "fixed",
        "gain_range_dollars": [10, 20], "filter_boundary_inclusive": True,
        "source_dir": str(SOURCE), "source_report": str(SOURCE / SOURCE_REPORT),
        "source_baseline_validated": False, "saved_primary_scores_replayed": False,
        "job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat(),
        "internal_seconds": args.max_seconds, "status": "running", "common_gain_component_complete": False,
        "simulation_regenerated": False, "simulation_copied": False, "saved_primary_fits_reused_for_common_gain": False,
        "observed_choices_or_RTs_used": False, "selected_replicates": [0], "wave_selected": False,
        "automatic_retry": False, "task_versions_are_independent_replications": False,
        "full_recovery_pass": None, "episode_terminal": False,
        **{field: "not_evaluated" for field in CLAIM_FIELDS},
        "score_normalization": NORMALIZATION,
        "interpretation": "One saved MZ0 common-gain10-through20 direction. All108 participants remain; unchanged selected rows retain their losses, responses, deadlines and slots. Two fresh original500 M0/MZ fits use only training rows. Held-out scores remain sums/counts pending both complementary directions. Original full-range scores only qualify the source. This does not establish recovery, generality, heterogeneity, template or missingness robustness, intervals or a gate verdict.",
        "cost_accounting": "elapsed_seconds covers this source validation, primary replay and fresh two-fit process once; original source cost is not charged again",
        "fits": [], "scored_fits": [], "completed_model_scores": {}}
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
            raise ValueError("Both fresh common-gain fits require the original500 implementation")
        report["implementation"] = spec
        source = json.loads((SOURCE / SOURCE_REPORT).read_text())
        truth = json.loads((SOURCE / "generating_parameters.json").read_text())
        events = SOURCE / "simulated_events.csv.gz"
        people = shared.load_cohort(events, check)
        shared.validate_source(source, truth, people, spec)
        check()
        report.update(source_baseline_validated=True, source_job_id=source["job_id"],
            source_finished_utc=source["finished_utc"], participants=108, runs=432, trials=27648)
        write_report(output, report)
        primary_splits = [load_split(events, list(a), list(b)) for a, b in HALVES]
        primary, discrepancy = shared.replay_saved_scores(source, primary_splits, spec, check)
        report.update(saved_primary_scores_replayed=True, saved_primary_participant_scores=primary,
            saved_score_replay_max_absolute_discrepancy=discrepancy,
            saved_score_replay_absolute_tolerance=shared.REPLAY_ABSOLUTE_TOLERANCE)
        check()
        training, scoring, support = filtered_panels(people, args.direction)
        report["retained_support"] = support
        write_report(output, report)
        for model in MODELS:
            check()
            fit_started = time.perf_counter()
            fit = fit_partially_pooled(training, model, "fixed", spec)
            fit.update(training_runs=list(train), scoring_runs=list(held), elapsed_seconds=time.perf_counter() - fit_started)
            json.dumps(fit, allow_nan=False)
            report["fits"].append(fit)
            write_report(output, report)
            check()
            validate_fit(fit, model, train, held, spec, people)
            scores = score_filtered(fit, scoring, spec, check)
            check()
            report["completed_model_scores"][model] = scores
            report["scored_fits"].append({"model": model, "training_runs": list(train), "scoring_runs": list(held)})
            write_report(output, report)
            print(json.dumps({"direction": args.direction, "model": model, "fit_completed": True,
                "elapsed_seconds": fit["elapsed_seconds"]}), flush=True)
        report["participant_scores"] = {person: {"version": row["version"], "heldout_trials": row["heldout_trials"],
            **{model + "_log_sum": report["completed_model_scores"][model][person]["log_sum"] for model in MODELS}}
            for person, row in support.items()}
        check()
        report.update(status="common_gain_component_completed", common_gain_component_complete=True,
            elapsed_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        write_report(output, report)
        check()
        succeeded = True
    except Exception as error:
        report.update(status="technical_common_gain_failure", error=f"{type(error).__name__}: {error}")
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if not succeeded:
            report.pop("participant_scores", None)
            report.update(status="technical_common_gain_failure", common_gain_component_complete=False,
                elapsed_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
            write_report(output, report)  # Finite failure closure retains returned fits and completed score prefixes.
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", type=int, choices=(0, 1), required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "direction", "status", "common_gain_component_complete",
        "elapsed_seconds", "error") if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["common_gain_component_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
