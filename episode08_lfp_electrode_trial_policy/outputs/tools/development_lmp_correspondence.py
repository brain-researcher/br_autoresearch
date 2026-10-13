#!/usr/bin/env python3
"""Finite native-LMP candidate and matched correspondence control."""
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

from lmp_acquired_reader import Exposure, read_trials
from development_learned_electrodes import role_record
from development_reader import selected_metadata
from lmp_correspondence import CONTROL, MC_SEEDS, POLICY
from lmp_correspondence_experiment import aggregate, attach_references, fit_target, score_target
from prepare_development_trial_roles import DEVELOPMENT
from ridge_training import new_progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-root", "metadata-report", "trial-roles", "reference-report", "acquired-reference", "output-dir"):
        parser.add_argument("--"+flag, type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned verified compute allocation required")
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread numerical settings required")
    if not 0 < args.max_seconds <= 270 or args.output_dir.exists():
        raise ValueError("Fresh preserved output and selected deadline at most 270 seconds required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.monotonic(), time.process_time()
    def deadline(*_):
        raise TimeoutError("Selected native-LMP correspondence full-cohort deadline reached")
    def check():
        if time.monotonic()-started >= args.max_seconds:
            deadline()
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    progress = new_progress()
    progress.update(selector_orderings=0, direction_permutation_maps=0, selection_seconds=0.,
                    capacity_inner_candidate_scores=0, direction_only_models=0)
    report = {"analysis_id": "development_search", "recipe": "outputs/lmp_only_correspondence_proposal.md",
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
        "selector_objective": "scalar native-LMP alpha1 mean-loss marginal gain; only control selector target correspondence is shuffled",
        "input_coordinates_per_electrode": 1, "native_LMP_guide_pair": [0., 0.],
        "high_frequency_neural_values_read": False, "high_frequency_columns_zero_padded": False,
        "three_band_decoder_capacity_matched": False, "native_LMP_certified_spike_free": False,
        "decoder_target_labels_shuffled": False,
        "equal_acquisition_cost_vs_own_LMP_control": True,
        "equal_physical_electrode_trial_cost_vs_three_band_initial_marginal": True,
        "equal_feature_bandwidth_vs_three_band_initial_marginal": False,
        "equal_acquisition_cost_vs_historical_fixed_random": False,
        "donor_supervision_used": False,
        "boundary": "trusted original initializer/later selected-channel source views; same UID is not OS isolation",
        "principal_contrast": "native-LMP true marginal Q minus mean-five native-LMP target-label control Q",
        "claim_limit": "whole-pipeline removal of100-400Hz power; no isolated spectral cause/spike-free certification/causal mechanism/coupling"}
    exposure = Exposure(report)
    try:
        metadata, roles = json.loads(args.metadata_report.read_text()), json.loads(args.trial_roles.read_text())
        if len(roles["sessions"]) != 8 or {session["filename"] for session in roles["sessions"]} != set(DEVELOPMENT):
            raise ValueError("All eight original development roles required")
        sources = {name: selected_metadata(metadata, roles, name) for name in DEVELOPMENT}
        for filename in DEVELOPMENT:
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
        if (progress["outer_training_refits"] != 760 or progress["inner_ridge_fits"] != 15200
                or progress["selector_orderings"] != 240 or progress["direction_permutation_maps"] != 1600
                or progress["capacity_inner_candidate_scores"] != 76000
                or report["fitting_trial_read_calls"] != 9984 or report["evaluation_trials_read"] != 320):
            raise ArithmeticError("Complete adopted target/selector/read counts differ")
        check()
        report["stage"] = "post_fit_reference_comparison"
        # No completed reference JSON is parsed until all new fitting is done.
        previous = json.loads(args.reference_report.read_text())
        acquired = json.loads(args.acquired_reference.read_text())
        replay, target_replay = attach_references(report["sessions"], previous, acquired)
        report["within_run_full_LMP_score_replay_max_absolute_difference"] = replay
        report["historical_direction_null_Q_replay_max_absolute_difference"] = target_replay
        report["panels"] = aggregate(report["sessions"])
        report["fixed_prediction_session_omission"] = [
            {"excluded_session": name, "panels": aggregate([s for s in report["sessions"] if s["filename"] != name], False)}
            for name in DEVELOPMENT]
        true = next(panel for panel in report["panels"] if panel["policy"] == POLICY)
        report["prospective_directional_screens"] = {
            "true_LMP_marginal_positive_vs_control_in_both_animals":
                all(animal["mean_delta_Q_from_"+CONTROL] > 0 for animal in true["animals"]),
            "true_LMP_marginal_positive_vs_direction_null_in_both_animals":
                all(animal["mean_delta_Q_from_direction_only"] > 0 for animal in true["animals"]),
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
        report["correspondence_spectral_contrast"] = {
            "definition": "LMP(true-control) minus three-band(true-control); capacity differs",
            "animals": [{"animal": animal, "contrast": float(np.mean([
                s["correspondence_contrast_LMP_minus_three_band"] for s in report["sessions"] if s["animal"] == animal]))}
                for animal in ("Chewie", "Mihili")]}
        report["correspondence_spectral_contrast"]["mean"] = float(np.mean([
            animal["contrast"] for animal in report["correspondence_spectral_contrast"]["animals"]]))
        report["reference_reuse"] = {"reference_report": str(args.reference_report), "acquired_reference": str(args.acquired_reference),
            "read_only_after_all_new_fits": True, "old_prediction_array_identity_claimed": False,
            "old_serialized_source_role_identity_verified": True,
            "old_native_unit_guide_identity_claimed": False, "old_model_refits_charged": 0}
        report.update(status="completed_two_LMP_correspondence_comparisons", stage="complete",
            valid_evaluations_charged=2, valid_evaluations_total=28, search_progress_resolution_Q=.001,
            scientific_scores_released=True)
    except Exception as error:
        report.update(status="technical_failure_LMP_correspondence_incomplete",
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
        (args.output_dir/"lmp_correspondence.json").write_text(payload)
    print(json.dumps({key: report[key] for key in
        ("status", "elapsed_seconds", "maximum_RSS_KiB", "valid_evaluations_charged")}), flush=True)


if __name__ == "__main__":
    main()
