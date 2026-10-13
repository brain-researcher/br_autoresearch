#!/usr/bin/env python3
"""Three ordinary paid initializers for fixed residual marginal and matched controls."""
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
from initializer_robustness import CONTROL, MC_SEEDS, POLICY
from initializer_robustness_experiment import (aggregate, attach_original_context, fit_target, score_target,
    validate_order_panels, average_orders, add_contrasts)
from initializer_robustness_roles import ORDERS, validate_prepared
from prepare_development_trial_roles import DEVELOPMENT
from ridge_training import new_progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-root", "metadata-report", "trial-roles", "prepared-roles", "reference-report", "output-dir"):
        parser.add_argument("--"+flag, type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned verified compute allocation required")
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread numerical settings required")
    if not 0 < args.max_seconds <= 600 or args.output_dir.exists():
        raise ValueError("Fresh preserved output and selected deadline at most 600 seconds required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.monotonic(), time.process_time()
    def deadline(*_):
        raise TimeoutError("Selected three-initializer full-cohort deadline reached")
    def check():
        if time.monotonic()-started >= args.max_seconds:
            deadline()
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    progress = new_progress()
    progress.update(selector_orderings=0, direction_permutation_maps=0, marginal_score_solves=0, residual_normalization_views=0, selection_seconds=0.,
                    capacity_inner_candidate_scores=0, direction_only_models=0)
    report = {"analysis_id": "development_search", "recipe": "outputs/initializer_robustness_proposal.md",
        "status": "running", "policy": POLICY, "control": CONTROL, "MC_seeds": list(MC_SEEDS),
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "fit_progress": progress,
        "sessions": [], "completed_session_ids": [], "completed_order_fits": [], "initializer_orders": list(ORDERS), "stage": "new_target_fits",
        "logical_neural_bytes": 0, "unique_logical_neural_bytes": 0,
        "fitting_trial_read_calls": 0, "evaluation_trials_read": 0,
        "valid_evaluations_charged": 0, "held_openings_charged": 0, "held_session_accessed": False,
        "global_search_completed": False, "named_explanatory_amendment_activated": False,
        "raw_arrays_or_decoder_coefficients_written": False,
        "target_definition": "one ten-bin time mean per original native population unit; no averaging across units",
        "selector_objective": "fixed incumbent alpha1 residual marginal gain in each paid initializer",
        "input_coordinates_per_electrode": 3, "population_targets_used_for_selection": True,
        "population_targets_acquired_and_used_for_decoding": True,
        "three_band_decoder_capacity_matched": True,
        "decoder_target_labels_shuffled": False,
        "equal_acquisition_cost_vs_own_target_label_control": True,
        "equal_physical_electrode_trial_cost_vs_three_band_initial_marginal": True,
        "equal_feature_bandwidth_vs_three_band_initial_marginal": True,
        "equal_acquisition_cost_vs_historical_fixed_random": False,
        "donor_supervision_used": False,
        "boundary": "trusted original initializer/later selected-channel source views; same UID is not OS isolation",
        "principal_contrast": "mean-three-order residual marginal Q minus mean-three-order mean-five-control Q",
        "control_definition": "five within-direction whole-Y maps with new order-specific literals; true decoder Y",
        "selector_direction_means_subtracted": True,
        "selector_input_SD": "residual fitting population SD",
        "decoder_direction_means_subtracted": True,
        "claim_limit": "initializer/acquisition-path robustness; no isolated initializer cause/removal/coupling",
        "order_aggregation": "mean Q after within-order mean-five-SSE control; never pool different denominators",
        "original_fitting_whitelist_and_evaluation_preserved": True,
        "original_role_files_modified": False}
    exposure = Exposure(report)
    try:
        metadata, roles = json.loads(args.metadata_report.read_text()), json.loads(args.trial_roles.read_text())
        if len(roles["sessions"]) != 8 or {session["filename"] for session in roles["sessions"]} != set(DEVELOPMENT):
            raise ValueError("All eight original development roles required")
        original_sources = {name: selected_metadata(metadata, roles, name) for name in DEVELOPMENT}
        prepared = validate_prepared(json.loads(args.prepared_roles.read_text()), metadata, roles)
        by_order = {entry["initializer_order"]: entry for entry in prepared["orders"]}
        report["prepared_roles"] = str(args.prepared_roles)
        report["within_order_full96_score_replay_max_absolute_difference"] = 0.
        for filename in DEVELOPMENT:
            check()
            t = time.monotonic()
            before_session_models, before_session_orders = progress["outer_training_refits"], progress["selector_orderings"]
            pending = []
            for order in ORDERS:
                source, role, trial_records = selected_metadata(metadata, by_order[order], filename)
                before_models, before_orders = progress["outer_training_refits"], progress["selector_orderings"]
                fitted_start = time.monotonic()
                models, nulls, guide, selections = fit_target(args.source_root/filename, source, role, trial_records,
                                                             progress, check, exposure, order)
                if (progress["outer_training_refits"]-before_models != 95
                        or progress["selector_orderings"]-before_orders != 30):
                    raise ArithmeticError("All95 distinct models/30 orders required in every initializer condition")
                pending.append((order, source, role, trial_records, models, nulls, guide, selections,
                                time.monotonic()-fitted_start))
                report["completed_order_fits"].append({"filename": filename, "initializer_order": order})
            if (progress["outer_training_refits"]-before_session_models != 285
                    or progress["selector_orderings"]-before_session_orders != 90):
                raise ArithmeticError("All285 models/90 orders must precede the shared evaluation read")
            reference_guide = pending[0][6]
            if (len(reference_guide) != 2 or any(len(item[6]) != 2 for item in pending)
                    or any(not all(np.array_equal(a, b) for a, b in zip(reference_guide, item[6])) for item in pending)):
                raise ArithmeticError("Native guides must agree across initializer conditions before evaluation")
            _, source, role, trial_records, *_ = pending[0]
            ex, ey, _ = read_trials(args.source_root/filename, source, role, trial_records,
                role["evaluation_trials"], source["common_physical_channels"], check, exposure,
                reference_guide, evaluation=True)
            labels = np.asarray([trial_records[trial]["direction_index"] for trial in role["evaluation_trials"]])
            order_results = []
            for order, source, role, trial_records, models, nulls, guide, selections, fit_seconds in pending:
                panels, seeds = score_target(models, nulls, ex, ey, labels, source["common_physical_channels"])
                replay = validate_order_panels(panels, seeds, ey.shape[1])
                report["within_order_full96_score_replay_max_absolute_difference"] = max(
                    report["within_order_full96_score_replay_max_absolute_difference"], replay)
                order_results.append({"initializer_order": order, "panels": add_contrasts(panels),
                    "seed_panels": seeds, "selection_records": selections, "source_roles": role_record(source, role),
                    "all30_target_orders_fixed_before_post_initial_reads": True,
                    "actual_target_models": 95, "training_read_and_fit_seconds": fit_seconds})
            panels, seeds = average_orders(order_results)
            report["sessions"].append({"filename": filename, "animal": role["animal"], "panels": panels,
                "seed_panels": seeds, "order_results": order_results, "target_unit_count": ey.shape[1],
                "all285_target_models_fixed_before_evaluation": True,
                "all90_target_orders_fixed_before_evaluation": True,
                "evaluation_read_shared_after_all_order_fits": True, "elapsed_seconds": time.monotonic()-t})
            report["completed_session_ids"].append(filename)
            print(json.dumps({"session_completed": filename, "elapsed_seconds": time.monotonic()-t,
                              "actual_target_models": 285, "scientific_scores_released": False}), flush=True)
            del pending, models, nulls, guide, selections, ex, ey
        if (progress["outer_training_refits"] != 2280 or progress["inner_ridge_fits"] != 45600
                or progress["selector_orderings"] != 720 or progress["direction_permutation_maps"] != 4800
                or progress["marginal_score_solves"] != 69120 or progress["residual_normalization_views"] != 120
                or progress["direction_only_models"] != 120
                or progress["capacity_inner_candidate_scores"] != 228000
                or report["fitting_trial_read_calls"] != 29952 or report["evaluation_trials_read"] != 320):
            raise ArithmeticError("Complete adopted three-order model/selector/read counts differ")
        check()
        report["stage"] = "post_fit_original_context"
        previous = json.loads(args.reference_report.read_text())
        attach_original_context(report["sessions"], previous, original_sources)
        report["panels"] = aggregate(report["sessions"])
        report["fixed_prediction_session_omission"] = [
            {"excluded_session": name, "panels": aggregate([s for s in report["sessions"] if s["filename"] != name], False)}
            for name in DEVELOPMENT]
        report["order_panels"] = []
        for order in ORDERS:
            rows = [{"filename": session["filename"], "animal": session["animal"],
                     "panels": session["order_results"][order]["panels"]} for session in report["sessions"]]
            report["order_panels"].append({"initializer_order": order, "panels": aggregate(rows),
                "fixed_prediction_session_omission": [
                    {"excluded_session": name, "panels": aggregate([s for s in rows if s["filename"] != name], False)}
                    for name in DEVELOPMENT]})
        true = next(panel for panel in report["panels"] if panel["policy"] == POLICY)
        report["prospective_directional_screens"] = {
            "mean_three_order_correspondence_positive_in_both_animals":
                all(animal["mean_delta_Q_from_"+CONTROL] > 0 for animal in true["animals"]),
            "correspondence_positive_in_both_animals_in_every_order": all(
                animal["mean_delta_Q_from_"+CONTROL] > 0
                for entry in report["order_panels"] for panel in entry["panels"] if panel["policy"] == POLICY
                for animal in panel["animals"]), "significance_or_promotion_claim": False}
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
        report["reference_reuse"] = {"reference_report": str(args.reference_report),
            "read_only_after_all_new_fits": True, "old_model_refits_charged": 0,
            "original_roles_verified_for_original_context_only": True,
            "historical_or_crossorder_score_capacity_replay_claimed": False,
            "crossorder_common_denominator_claimed": False,
            "old_native_unit_guide_identity_claimed": False,
            "legacy_initial_marginal_SSE_and_denominator_replay_claimed": False}
        report.update(status="completed_two_initializer_robustness_comparisons", stage="complete",
            valid_evaluations_charged=2, valid_evaluations_total=41, search_progress_resolution_Q=.001,
            scientific_scores_released=True)
    except Exception as error:
        report.update(status="technical_failure_initializer_robustness_incomplete",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started, process_CPU_seconds=time.process_time()-cpu_started,
                      maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      observed_utc=datetime.now(timezone.utc).isoformat())
        report["retained_report_bytes"] = 0
        for _ in range(4):
            payload = json.dumps(report, indent=2, allow_nan=False)+"\n"
            size = len(payload.encode("utf-8"))
            if report["retained_report_bytes"] == size:
                break
            report["retained_report_bytes"] = size
        (args.output_dir/"initializer_robustness.json").write_text(payload)
    print(json.dumps({key: report[key] for key in
        ("status", "elapsed_seconds", "maximum_RSS_KiB", "valid_evaluations_charged")}), flush=True)


if __name__ == "__main__":
    main()
