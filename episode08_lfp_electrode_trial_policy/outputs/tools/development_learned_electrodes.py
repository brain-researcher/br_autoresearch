#!/usr/bin/env python3
"""Finite complete pilot or remaining cohort for the adopted learned selector."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time

import numpy as np

from acquired_only_reader import Exposure, balanced_ids, read_trials
from development_reader import selected_metadata
from learned_electrode_experiment import aggregate, attach_reference, donor_records, fit_target, score_target
from learned_electrode_value import CONTROL, MC_SEEDS, POLICY
from prepare_development_trial_roles import DEVELOPMENT
from ridge_training import new_progress

PILOT = DEVELOPMENT[0]
REPORT_NAME = "learned_electrode_value.json"


def role_record(source, role):
    return {"physical_channels": source["common_physical_channels"],
            "initial_trial_ids": balanced_ids(role, 32), "fitting_trial_ids": balanced_ids(role, 96),
            "evaluation_trial_ids": role["evaluation_trials"],
            "initial_inner_folds": role["nested_training_sets"]["32"]["inner_folds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "remaining"), required=True)
    for flag in ("source-root", "metadata-report", "trial-roles", "output-dir"):
        parser.add_argument("--"+flag, type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned verified compute allocation required")
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread numerical settings required")
    if (args.max_seconds <= 0 or (args.mode == "pilot" and args.max_seconds > 270)
            or args.output_dir.exists()
            or (args.mode == "remaining" and (args.pilot_report is None or args.reference_report is None))):
        raise ValueError("Fresh output, finite selected deadline and declared completed reference paths required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def deadline(*_):
        raise TimeoutError("Selected learned-electrode experiment deadline reached")
    def check():
        if time.monotonic()-started >= args.max_seconds:
            deadline()
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    progress = new_progress()
    progress.update(donor_label_fits=0, learned_ranker_fits=0, capacity_inner_candidate_scores=0,
                    direction_only_models=0)
    report = {"analysis_id": "development_search", "recipe": "outputs/learned_electrode_value_proposal.md",
        "status": "running", "mode": args.mode, "policy": POLICY, "control": CONTROL,
        "MC_seeds": list(MC_SEEDS), "pilot_target": PILOT, "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "fit_progress": progress, "sessions": [], "completed_session_ids": [],
        "logical_neural_bytes": 0, "unique_logical_neural_bytes": 0,
        "fitting_trial_read_calls": 0, "evaluation_trials_read": 0,
        "valid_evaluations_charged": 0, "held_openings_charged": 0, "held_session_accessed": False,
        "global_search_completed": False, "named_explanatory_amendment_activated": False,
        "raw_arrays_or_decoder_coefficients_written": False, "small_ranker_parameters_written": True,
        "target_definition": "one ten-bin time mean per original native population unit; no averaging across units",
        "donor_objective": "||Y_residual-X_residual B||_F^2/n + 1*||B||_F^2",
        "donor_helper": "ridge_training.ridge_coefficients with direction_training residual statistics",
        "ranker_objective": "weighted mean squared label error + 1*||beta||_2^2; unpenalized intercept",
        "equal_target_session_acquisition_cost": True, "equal_total_training_information_or_cost": False,
        "boundary": "target excluded from donor ranker inputs/statistics; source shared globally as donor for other sessions; not OS isolation",
        "transfer_limit": "leave-one-session-out within two exposed animals, not new-animal or chronological causal transfer"}
    exposure = Exposure(report)
    pilot = None
    try:
        metadata, roles = json.loads(args.metadata_report.read_text()), json.loads(args.trial_roles.read_text())
        if len(roles["sessions"]) != 8 or {s["filename"] for s in roles["sessions"]} != set(DEVELOPMENT):
            raise ValueError("All eight fixed development roles required")
        sources = {name: selected_metadata(metadata, roles, name) for name in DEVELOPMENT}
        target_names = (PILOT,) if args.mode == "pilot" else tuple(name for name in DEVELOPMENT if name != PILOT)
        donor_names = tuple(name for name in DEVELOPMENT if name != PILOT) if args.mode == "pilot" else DEVELOPMENT
        t = time.monotonic()
        training = donor_records(args.source_root, sources, donor_names, progress, check, exposure)
        report["donor_training_records"] = training
        report["donor_preparation_seconds"] = time.monotonic()-t
        report["donor_phase_cost"] = {key: report[key] for key in
            ("logical_neural_bytes", "unique_logical_neural_bytes", "fitting_trial_read_calls")}
        if len(training) != len(donor_names)*5 or progress["donor_label_fits"] != len(donor_names)*5*96:
            raise ArithmeticError("Every declared donor view/electrode label required")
        for filename in target_names:
            check()
            source, role, records = sources[filename]
            t, before = time.monotonic(), progress["outer_training_refits"]
            models, nulls, guide, selections = fit_target(args.source_root/filename, source, role, records,
                                                         training, progress, check, exposure)
            if progress["outer_training_refits"]-before != 65:
                raise ArithmeticError("Exactly65 actual target models required before evaluation")
            fit_end = time.monotonic()
            evaluation_union_before = report["unique_logical_neural_bytes"]
            ex, ey, _ = read_trials(args.source_root/filename, source, role, records, role["evaluation_trials"],
                source["common_physical_channels"], check, exposure, guide, evaluation=True)
            labels = np.asarray([records[trial]["direction_index"] for trial in role["evaluation_trials"]])
            panels, seed_panels = score_target(models, nulls, ex, ey, labels, source["common_physical_channels"])
            report["sessions"].append({"filename": filename, "animal": role["animal"], "panels": panels,
                "seed_panels": seed_panels, "selection_records": selections,
                "source_roles": role_record(source, role), "target_unit_count": ey.shape[1],
                "evaluation_unique_logical_bytes": report["unique_logical_neural_bytes"]-evaluation_union_before,
                "all65_target_models_and20_rankers_fixed_before_evaluation": True,
                "all_target_orders_fixed_before_post_initial_reads": True,
                "training_read_and_fit_seconds": fit_end-t, "elapsed_seconds": time.monotonic()-t})
            report["completed_session_ids"].append(filename)
            print(json.dumps({"session_completed": filename, "elapsed_seconds": time.monotonic()-t,
                              "actual_target_models": 65, "scientific_scores_released": False}), flush=True)
            del models, nulls, selections, ex, ey, guide
        if (progress["outer_training_refits"] != 65*len(target_names)
                or progress["inner_ridge_fits"] != 1300*len(target_names)
                or progress["learned_ranker_fits"] != 20*len(target_names)
                or report["fitting_trial_read_calls"] != 96*len(donor_names)+864*len(target_names)
                or report["evaluation_trials_read"] != 40*len(target_names)):
            raise ArithmeticError("Complete declared fit/read counts differ")
        if args.mode == "pilot":
            report.update(status="completed_learned_selector_pilot_scores_withheld",
                          scientific_scores_released=False, reference_replay_deferred_until_full_cohort=True)
        else:
            # No prior scientific score or reference enters any new fit/ranker.
            pilot = json.loads(args.pilot_report.read_text())
            if (pilot["status"] != "completed_learned_selector_pilot_scores_withheld"
                    or pilot["mode"] != "pilot" or pilot["policy"] != POLICY or pilot["control"] != CONTROL
                    or pilot["MC_seeds"] != list(MC_SEEDS) or pilot["completed_session_ids"] != [PILOT]
                    or pilot["held_session_accessed"] or len(pilot["sessions"]) != 1
                    or pilot["sessions"][0]["source_roles"] != role_record(*sources[PILOT][:2])
                    or pilot["fit_progress"]["donor_label_fits"] != 3360
                    or pilot["fit_progress"]["outer_training_refits"] != 65
                    or pilot["fit_progress"]["inner_ridge_fits"] != 1300
                    or pilot["fit_progress"]["learned_ranker_fits"] != 20
                    or pilot["evaluation_trials_read"] != 40):
                raise ValueError("Exact complete prospective first-target pilot required; no partial substitution")
            report["sessions"] = [pilot["sessions"][0], *report["sessions"]]
            previous = json.loads(args.reference_report.read_text())
            report["full_electrode_score_replay_max_absolute_difference"] = attach_reference(report["sessions"], previous)
            report["panels"] = aggregate(report["sessions"])
            report["fixed_prediction_session_omission"] = [
                {"excluded_session": name, "panels": aggregate([s for s in report["sessions"] if s["filename"] != name], False)}
                for name in DEVELOPMENT]
            report["reference_reuse"] = {"path": str(args.reference_report), "status": previous["status"],
                "refitted": False, "old_prediction_array_identity_claimed": False,
                "role_basis": "unchanged frozen role manifest/qualified reader; old score packet does not store trial arrays"}
            report["pilot_reuse"] = {"path": str(args.pilot_report), "refitted": False,
                "elapsed_seconds": pilot["elapsed_seconds"], "maximum_RSS_KiB": pilot["maximum_RSS_KiB"]}
            report["combined_fit_progress"] = {key: progress[key]+pilot["fit_progress"][key] for key in progress}
            report["combined_component_costs"] = {key: report[key]+pilot[key] for key in
                ("logical_neural_bytes", "fitting_trial_read_calls", "evaluation_trials_read")}
            # Remaining reads all8 full96 fitting pools as donors and only its
            # seven targets' evaluation. Add the pilot40 evaluation identities
            # using the measured dtype-aware Exposure increment, not a byte guess.
            report["unique_source_union_across_components_bytes"] = (
                report["unique_logical_neural_bytes"]+pilot["sessions"][0]["evaluation_unique_logical_bytes"])
            report["unique_source_union_note"] = "Remaining full training/evaluation union plus the disjoint measured pilot evaluation union; overlapping donor reads counted once."
            report.update(status="completed_two_learned_selector_comparisons", scientific_scores_released=True,
                valid_evaluations_charged=2, valid_evaluations_total=23, search_progress_resolution_Q=.001,
                completed_session_ids=list(DEVELOPMENT))
    except Exception as error:
        report.update(status="technical_failure_learned_selector_incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
                      maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      observed_utc=datetime.now(timezone.utc).isoformat())
        if pilot is not None:
            report["combined_component_elapsed_seconds"] = report["elapsed_seconds"]+pilot["elapsed_seconds"]
        path = args.output_dir/REPORT_NAME
        report["retained_report_bytes"] = 0
        for _ in range(4):
            payload = json.dumps(report, indent=2, allow_nan=False)+"\n"
            size = len(payload.encode("utf-8"))
            if report["retained_report_bytes"] == size:
                break
            report["retained_report_bytes"] = size
        path.write_text(payload)
    print(json.dumps({key: report[key] for key in
        ("status", "elapsed_seconds", "maximum_RSS_KiB", "valid_evaluations_charged")}), flush=True)


if __name__ == "__main__":
    main()
