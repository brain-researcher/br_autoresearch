#!/usr/bin/env python3
"""Finite first-session pilot or remaining cohort of conditional-gain selection."""
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

from acquired_only_reader import Exposure, read_trials
from development_learned_electrodes import role_record
from development_reader import selected_metadata
from initializer_conditional_gain import CONTROL, MC_SEEDS, POLICY
from initializer_conditional_experiment import aggregate, attach_references, fit_target, score_target
from prepare_development_trial_roles import DEVELOPMENT
from ridge_training import new_progress

PILOT = DEVELOPMENT[0]
RECIPE = "outputs/initializer_conditional_gain_proposal.md"


def validate_pilot(pilot, sources):
    counts = {"outer_training_refits": 95, "inner_ridge_fits": 1900,
              "capacity_inner_candidate_scores": 9500, "selector_orderings": 30,
              "conditional_selection_steps": 960, "direction_permutation_maps": 200}
    if (pilot["status"] != "completed_initializer_conditional_pilot_scores_withheld"
            or pilot["mode"] != "pilot" or pilot["recipe"] != RECIPE
            or pilot["policy"] != POLICY or pilot["control"] != CONTROL
            or pilot["MC_seeds"] != list(MC_SEEDS) or pilot["held_session_accessed"]
            or pilot["scientific_scores_released"] or pilot["valid_evaluations_charged"] != 0
            or pilot["completed_session_ids"] != [PILOT] or len(pilot["sessions"]) != 1
            or pilot["fitting_trial_read_calls"] != 1248 or pilot["evaluation_trials_read"] != 40
            or any(pilot["fit_progress"][key] != value for key, value in counts.items())):
        raise ValueError("Complete unchanged predetermined resource pilot required")
    session = pilot["sessions"][0]
    if (session["filename"] != PILOT or session["source_roles"] != role_record(*sources[PILOT][:2])
            or session["animal"] != sources[PILOT][1]["animal"]
            or not session["all95_target_models_fixed_before_evaluation"]
            or not session["all30_target_orders_fixed_before_post_initial_reads"]):
        raise ValueError("Pilot recipe/source roles or information state differs; no refit/fallback")
    return session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "remaining"), required=True)
    for flag in ("source-root", "metadata-report", "trial-roles", "output-dir"):
        parser.add_argument("--"+flag, type=Path, required=True)
    for flag in ("pilot-report", "target-reference", "uniform-reference"):
        parser.add_argument("--"+flag, type=Path)
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned verified compute allocation required")
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread numerical settings required")
    if (args.max_seconds <= 0 or (args.mode == "pilot" and args.max_seconds > 270)
            or args.output_dir.exists() or (args.mode == "remaining" and any(path is None for path in
                (args.pilot_report, args.target_reference, args.uniform_reference)))):
        raise ValueError("Fresh preserved output, bounded pilot or selected finite remainder and required references")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.monotonic(), time.process_time()
    def deadline(*_):
        raise TimeoutError("Selected initializer conditional-gain phase deadline reached")
    def check():
        if time.monotonic()-started >= args.max_seconds:
            deadline()
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    progress = new_progress()
    progress.update(selector_orderings=0, direction_permutation_maps=0, conditional_selection_steps=0, selection_seconds=0.,
                    capacity_inner_candidate_scores=0, direction_only_models=0)
    report = {"analysis_id": "development_search", "recipe": RECIPE, "mode": args.mode, "pilot_target": PILOT,
        "status": "running", "policy": POLICY, "control": CONTROL, "MC_seeds": list(MC_SEEDS),
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "fit_progress": progress,
        "sessions": [], "completed_session_ids": [], "stage": "new_target_fits",
        "logical_neural_bytes": 0, "unique_logical_neural_bytes": 0,
        "fitting_trial_read_calls": 0, "evaluation_trials_read": 0,
        "valid_evaluations_charged": 0, "held_openings_charged": 0, "held_session_accessed": False,
        "global_search_completed": False, "named_explanatory_amendment_activated": False,
        "raw_arrays_or_decoder_coefficients_written": False,
        "target_definition": "one ten-bin time mean per original native population unit; no averaging across units",
        "selector_objective": "fixed alpha1 conditional mean-loss ridge gain via rank-three updates; exactly32 steps",
        "input_coordinates_per_electrode": 3, "population_targets_used_for_selection": True,
        "population_targets_acquired_and_used_for_decoding": True,
        "three_band_decoder_capacity_matched": True,
        "decoder_target_labels_shuffled": False,
        "equal_acquisition_cost_vs_own_target_label_control": True,
        "equal_acquisition_cost_vs_reused_uniform_control": True,
        "equal_physical_electrode_trial_cost_vs_three_band_initial_marginal": True,
        "equal_feature_bandwidth_vs_three_band_initial_marginal": True,
        "equal_acquisition_cost_vs_historical_fixed_random": False,
        "donor_supervision_used": False,
        "boundary": "trusted original initializer/later selected-channel source views; same UID is not OS isolation",
        "principal_contrast": "initializer conditional Q versus own mean-five target-label control and original marginal",
        "control_definition": "exact original within-direction whole-target maps; independent greedy state; true-Y decoder",
        "claim_limit": "algorithmic conditional proxy, not activated electrode-removal/complementarity/coupling endpoint"}
    exposure = Exposure(report)
    pilot = None
    try:
        metadata, roles = json.loads(args.metadata_report.read_text()), json.loads(args.trial_roles.read_text())
        if len(roles["sessions"]) != 8 or {session["filename"] for session in roles["sessions"]} != set(DEVELOPMENT):
            raise ValueError("All eight original development roles required")
        sources = {name: selected_metadata(metadata, roles, name) for name in DEVELOPMENT}
        target_names = (PILOT,) if args.mode == "pilot" else tuple(name for name in DEVELOPMENT if name != PILOT)
        for filename in target_names:
            check()
            source, role, trial_records = sources[filename]
            before_models, before_orders = progress["outer_training_refits"], progress["selector_orderings"]
            t = time.monotonic()
            models, nulls, guide, selections = fit_target(args.source_root/filename, source, role, trial_records,
                                                         progress, check, exposure)
            if (progress["outer_training_refits"]-before_models != 95
                    or progress["selector_orderings"]-before_orders != 30):
                raise ArithmeticError("All95 distinct models/30 selector orderings required before evaluation")
            fitted_at = time.monotonic()
            ex, ey, _ = read_trials(args.source_root/filename, source, role, trial_records,
                role["evaluation_trials"], source["common_physical_channels"], check, exposure, guide, evaluation=True)
            labels = np.asarray([trial_records[trial]["direction_index"] for trial in role["evaluation_trials"]])
            panels, seeds = score_target(models, nulls, ex, ey, labels, source["common_physical_channels"])
            report["sessions"].append({"filename": filename, "animal": role["animal"], "panels": panels,
                "seed_panels": seeds, "selection_records": selections, "source_roles": role_record(source, role),
                "target_unit_count": ey.shape[1], "all95_target_models_fixed_before_evaluation": True,
                "all30_target_orders_fixed_before_post_initial_reads": True,
                "training_read_and_fit_seconds": fitted_at-t, "elapsed_seconds": time.monotonic()-t})
            report["completed_session_ids"].append(filename)
            print(json.dumps({"session_completed": filename, "elapsed_seconds": time.monotonic()-t,
                              "actual_target_models": 95, "scientific_scores_released": False}), flush=True)
            del models, nulls, guide, selections, ex, ey
        count = len(target_names)
        if (progress["outer_training_refits"] != 95*count or progress["inner_ridge_fits"] != 1900*count
                or progress["selector_orderings"] != 30*count or progress["direction_permutation_maps"] != 200*count
                or progress["conditional_selection_steps"] != 960*count
                or progress["capacity_inner_candidate_scores"] != 9500*count
                or report["fitting_trial_read_calls"] != 1248*count or report["evaluation_trials_read"] != 40*count):
            raise ArithmeticError("Complete adopted target/selector/read counts differ")
        check()
        if args.mode == "pilot":
            report.update(status="completed_initializer_conditional_pilot_scores_withheld", stage="pilot_complete",
                scientific_scores_released=False, reference_replay_deferred_until_full_cohort=True)
            return
        report["stage"] = "post_fit_reference_comparison"
        # Pilot outcomes and all historical references remain unread until the
        # other seven target sessions have completed every new fit.
        pilot_packet = json.loads(args.pilot_report.read_text())
        pilot_session = validate_pilot(pilot_packet, sources)
        pilot = pilot_packet
        report["sessions"] = [pilot_session, *report["sessions"]]
        report["completed_session_ids"] = [session["filename"] for session in report["sessions"]]
        report["combined_fit_progress"] = {key: progress[key]+pilot["fit_progress"][key] for key in progress}
        previous = json.loads(args.target_reference.read_text())
        uniform = json.loads(args.uniform_reference.read_text())
        replay, target_replay = attach_references(report["sessions"], previous, uniform)
        report["full_electrode_historical_and_within_run_score_replay_max_absolute_difference"] = replay
        report["historical_direction_null_Q_replay_max_absolute_difference"] = target_replay
        report["panels"] = aggregate(report["sessions"])
        report["fixed_prediction_session_omission"] = [
            {"excluded_session": name, "panels": aggregate([s for s in report["sessions"] if s["filename"] != name], False)}
            for name in DEVELOPMENT]
        true = next(panel for panel in report["panels"] if panel["policy"] == POLICY)
        report["prospective_directional_screens"] = {
            "conditional_positive_vs_own_target_label_control_in_both_animals":
                all(animal["mean_delta_Q_from_"+CONTROL] > 0 for animal in true["animals"]),
            "conditional_positive_vs_initial_marginal_in_both_animals":
                all(animal["mean_delta_Q_from_initial_marginal"] > 0 for animal in true["animals"]),
            "significance_or_promotion_claim": False}
        report["seed_summaries"] = []
        for seed in MC_SEEDS:
            animals = []
            for animal in ("Chewie", "Mihili"):
                subset = [s for s in report["sessions"] if s["animal"] == animal]
                seed_q = float(np.mean([next(entry["panel"]["mean_Q"] for entry in s["seed_panels"]
                                             if entry["mc_seed"] == seed) for s in subset]))
                true_q = float(np.mean([s["panels"][POLICY]["mean_Q"] for s in subset]))
                animals.append({"animal": animal, "mean_Q": seed_q, "true_minus_seed_Q": true_q-seed_q})
            report["seed_summaries"].append({"mc_seed": seed, "animals": animals,
                "mean_Q": float(np.mean([animal["mean_Q"] for animal in animals])),
                "true_minus_seed_Q": float(np.mean([animal["true_minus_seed_Q"] for animal in animals]))})
        report["reference_reuse"] = {"target_reference": str(args.target_reference), "uniform_reference": str(args.uniform_reference),
            "pilot_report": str(args.pilot_report), "pilot_reused_without_refit": True,
            "read_only_after_all_new_fits": True, "old_prediction_array_identity_claimed": False,
            "old_serialized_source_role_identity_verified": True,
            "old_native_unit_guide_identity_claimed": False, "old_model_refits_charged": 0}
        report.update(status="completed_two_initializer_conditional_comparisons", stage="complete",
            valid_evaluations_charged=2, valid_evaluations_total=32, search_progress_resolution_Q=.001,
            scientific_scores_released=True)
    except Exception as error:
        report.update(status="technical_failure_initializer_conditional_incomplete",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started, process_CPU_seconds=time.process_time()-cpu_started,
                      maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      observed_utc=datetime.now(timezone.utc).isoformat())
        if pilot is not None:
            report["inherited_pilot_retained_report_bytes"] = pilot["retained_report_bytes"]
            report["combined_resource_cost"] = {key: report[key]+pilot[key] for key in
                ("elapsed_seconds", "process_CPU_seconds", "logical_neural_bytes", "unique_logical_neural_bytes",
                 "fitting_trial_read_calls", "evaluation_trials_read")}
            report["combined_resource_cost"]["maximum_RSS_KiB"] = max(report["maximum_RSS_KiB"], pilot["maximum_RSS_KiB"])
            report["combined_resource_cost"]["scope"] = "sum of disjoint-session pilot and remaining executions; no queue/idle wall time"
        report["retained_report_bytes"] = 0
        for _ in range(4):
            payload = json.dumps(report, indent=2, allow_nan=False)+"\n"
            size = len(payload.encode("utf-8"))
            if report["retained_report_bytes"] == size:
                break
            report["retained_report_bytes"] = size
        (args.output_dir/"initializer_conditional_gain.json").write_text(payload)
        print(json.dumps({key: report[key] for key in
            ("status", "elapsed_seconds", "maximum_RSS_KiB", "valid_evaluations_charged")}), flush=True)


if __name__ == "__main__":
    main()
