#!/usr/bin/env python3
"""Finite urgency likelihood timing/mesh pilot on preserved synthetic MZ0.

Two deterministic participants and existing initialization vectors are probes,
not fitted support or an inferential sample. No optimizer or model fit runs.
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

import hierarchical_behavior as behavior
from qualify_assets import require_compute
import run_single_process_rivals as shared


SOURCE, IMPLEMENTATION, HALVES = shared.SOURCE, shared.IMPLEMENTATION, shared.HALVES
SOURCE_REPORT = "single_process_recovery.json"
MODELS, MAX_SECONDS = ("M0", "MZ"), 1800
PHASE = "MZ0_urgency_cost_precision_pilot"
REPORT_NAME = "mz_urgency_cost_precision_pilot.json"
MESHES = (("coarse", .002, 200), ("refined", .001, 400))
EXPECTED_PAIRS, EXPECTED_CALLS = 24, 48
CLAIM_FIELDS = ("family_selection", "family_recovery_rate", "false_selection_rate", "confusion_rate",
                "group_recovery", "group_intervals", "interval_coverage", "individual_recovery",
                "full_menu_selection", "full_gate", "early_kill", "curtailment", "urgency_robustness",
                "fitted_support_precision_qualification")


def initialization_vectors(model, spec):
    """The exact center and two clipped starts used in the original first sweep."""
    if model not in MODELS:
        raise ValueError("Only the selected M0/MZ initialization probes are defined")
    names = behavior.parameter_names(model, "urgency")
    priors = [spec["parameters"][name] for name in names]
    centers = np.array([prior["population_center"] for prior in priors])
    scales = np.array([prior["individual_scale_center"] for prior in priors])
    bounds = [prior["bounds"] for prior in priors]
    offsets = spec["optimizer"]["initial_standardized_offsets"]
    if offsets != [[-.5] * 8, [.5] * 8]:
        raise ValueError("Preserve the two existing deterministic standardized offsets")
    starts = [centers.copy()]
    for offset in offsets:
        starts.append(np.clip(centers + np.asarray(offset[:len(names)]) * scales,
                              [bound[0] for bound in bounds], [bound[1] for bound in bounds]))
    return names, starts


def drift_and_horizon(theta, names, data):
    """Use exactly the urgency likelihood's drift and decision horizon."""
    params = dict(zip(names, theta))
    drift = behavior.deadline_ddm.drift_rate(data["gain"], data["loss"], params["kappa"],
        params.get("eta", 0.), params.get("criterion", 0.), params.get("gamma_gain", 1.), params.get("gamma_loss", 1.))
    return drift, float(4 - 4 * behavior.expit(params["nondecision_logit"]))


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"Urgency numerical pilot interrupted by signal {signum}; unfinished probes remain incomplete")


def run(args):
    require_compute()
    if args.source_dir.resolve() != SOURCE.resolve() or args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use only the fixed original MZ0 source and unchanged implementation")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("The whole-process cap must be1 through1800 seconds")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1" or any(os.environ.get(name) != "1" for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("Exactly one allocated CPU and all four numerical thread limits of one are required")
    output = args.output_dir.resolve()
    if output.exists() or output == SOURCE.resolve() or SOURCE.resolve() in output.parents:
        raise FileExistsError("Use a fresh output outside the preserved source")
    output.mkdir(parents=True)
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process deadline exceeded")

    report = {
        "analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "generating_model": "MZ", "replicate_index": 0, "seed": shared.SEED, "rival": "urgency",
        "source_dir": str(SOURCE), "source_report": str(SOURCE / SOURCE_REPORT),
        "source_baseline_validated": False, "saved_primary_scores_replayed": False,
        "job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat(),
        "internal_seconds": args.max_seconds, "status": "running", "pilot_complete": False,
        "simulation_regenerated": False, "simulation_copied": False, "observed_choices_or_RTs_used": False,
        "optimizer_called": False, "model_fitted": False, "actual_fitted_support_evaluated": False,
        "wave_selected": False, "automatic_retry": False, "full_recovery_pass": None, "episode_terminal": False,
        **{field: "not_evaluated" for field in CLAIM_FIELDS},
        "participant_selection": "first_sorted_participant_within_each_EI_and_ER_after_full108_source_validation",
        "probe_scope": "both_primary_training_halves_at_the_three_original_first_sweep_initializations",
        "expected_probe_pairs": EXPECTED_PAIRS, "expected_likelihood_calls": EXPECTED_CALLS,
        "completed_probe_pairs": 0, "completed_likelihood_calls": 0, "probes": [],
        "interpretation": "Likelihood timing and mesh differences at deterministic initializations only. The two-person probe is not a reduced recovery cohort or model fit. No numerical acceptance threshold, model comparison or gate verdict is supplied. The required .001/400 repeat at future fitted support remains unevaluated; eventual fits must retain all108 participants.",
        "cost_accounting": "elapsed_seconds covers full source validation, primary replay and this finite probe process once; prior cohort fitting cost is not charged again",
        "cost_scaling": "each_likelihood_call_uses_actual_unique_drifts_times_ceil_decision_horizon_over_dt_times_spatial_intervals; optimizer_objective_call_counts_and_converged_fit_cost_remain_unknown",
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
        if (spec["optimizer"]["maximum_sweeps"] != 500
                or spec["parameters"]["urgency"]["bounds"] != [0., 2.]
                or spec["urgency_solver"] != {"dt_seconds": .002, "spatial_intervals": 200,
                    "required_precision_check": "repeat_at_dt_0.001_and_400_intervals_at_fitted_support_before_rival_inference"}):
            raise ValueError("Preserve the original500 implementation and declared urgency support/meshes")
        report["implementation"] = spec
        refined = deepcopy(spec)
        refined["urgency_solver"].update(dt_seconds=.001, spatial_intervals=400)
        report["refined_numerical_implementation"] = refined
        source = json.loads((SOURCE / SOURCE_REPORT).read_text())
        truth = json.loads((SOURCE / "generating_parameters.json").read_text())
        check()
        events = SOURCE / "simulated_events.csv.gz"
        people = shared.load_cohort(events, check)
        shared.validate_source(source, truth, people, spec)
        check()
        report.update(source_baseline_validated=True, source_job_id=source["job_id"], source_finished_utc=source["finished_utc"],
                      source_participants=108, source_runs=432, source_trials=27648)
        write_report(output, report)
        splits = [shared.load_split(events, list(train), list(held)) for train, held in HALVES]
        check()
        primary_scores, discrepancy = shared.replay_saved_scores(source, splits, spec, check)
        report.update(saved_primary_scores_replayed=True, saved_primary_participant_scores=primary_scores,
                      saved_score_replay_max_absolute_discrepancy=discrepancy,
                      saved_score_replay_absolute_tolerance=shared.REPLAY_ABSOLUTE_TOLERANCE)
        selected = {version: min(person for person, rows in people.items() if rows[0]["task_version"] == version)
                    for version in ("EI", "ER")}
        report["selected_probe_participants"] = selected
        write_report(output, report)
        for (train, held), (training, scoring) in zip(HALVES, splits):
            for version, person in selected.items():
                rows = training[person]["rows"]
                if (len(rows) != 128 or {int(row["run"]) for row in rows} != set(train)
                        or training[person]["version"] != version):
                    raise ValueError("Each numerical probe requires its unchanged128-row training half")
                data = shared.make_data(rows)
                for model in MODELS:
                    names, starts = initialization_vectors(model, spec)
                    for start_index, theta in enumerate(starts):
                        check()
                        drift, horizon = drift_and_horizon(theta, names, data)
                        probe = {"participant": person, "task_version": version, "model": model,
                            "training_runs": list(train), "excluded_scoring_runs": list(held), "trials": 128,
                            "trial_slots": [[int(row["run"]), int(row["trial_order"])] for row in rows],
                            "start_index": start_index, "parameter_names": names, "parameters": theta.tolist(),
                            "actual_unique_drift_count": int(len(np.unique(drift))), "decision_horizon_seconds": horizon,
                            "meshes": {}, "pair_complete": False}
                        report["probes"].append(probe)
                        for mesh, dt, intervals in MESHES:
                            check()
                            report["active_call"] = {"probe_index": len(report["probes"]) - 1, "mesh": mesh}
                            write_report(output, report)
                            call_started = time.perf_counter()
                            values = np.asarray(shared.trial_likelihood(theta, names, data,
                                spec if mesh == "coarse" else refined, "urgency"))
                            elapsed = time.perf_counter() - call_started
                            if values.shape != (128,) or not np.isfinite(values).all():
                                raise ValueError("An urgency probe must return128 finite joint trial log scores")
                            probe["meshes"][mesh] = {"dt_seconds": dt, "spatial_intervals": intervals,
                                "time_steps_per_drift": math.ceil(horizon / dt), "elapsed_seconds": elapsed,
                                "trial_log_scores": values.tolist(), "mean_log_score": float(values.mean())}
                            report["completed_likelihood_calls"] += 1
                            report.pop("active_call", None)
                            write_report(output, report)  # Preserve the completed vector before checking the cap.
                            check()
                        differences = (np.asarray(probe["meshes"]["refined"]["trial_log_scores"])
                                       - np.asarray(probe["meshes"]["coarse"]["trial_log_scores"]))
                        probe.update(pair_complete=True, mesh_difference={
                            "direction": "refined_minus_coarse", "per_trial_log_score_difference": differences.tolist(),
                            "mean_log_score_difference": float(differences.mean()),
                            "maximum_absolute_log_score_difference": float(np.max(np.abs(differences)))})
                        report["completed_probe_pairs"] += 1
                        write_report(output, report)
        check()
        if report["completed_probe_pairs"] != EXPECTED_PAIRS or report["completed_likelihood_calls"] != EXPECTED_CALLS:
            raise ValueError("The finite selected probe panel is incomplete")
        report.update(status="urgency_cost_precision_pilot_completed", pilot_complete=True,
                      elapsed_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        write_report(output, report)
        check()
        succeeded = True
    except Exception as error:
        report["incomplete_reason"] = f"{type(error).__name__}: {error}"
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if not succeeded:
            report.update(status="urgency_cost_precision_pilot_incomplete", pilot_complete=False,
                          elapsed_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
            write_report(output, report)  # Finite wrapper grace is reserved for incomplete closure.
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "status", "pilot_complete", "completed_probe_pairs",
        "completed_likelihood_calls", "elapsed_seconds", "incomplete_reason") if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["pilot_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
