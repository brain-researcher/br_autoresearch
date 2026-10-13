#!/usr/bin/env python3
"""Finite full-cohort reverse-animal learned-selector comparison."""
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
from prepare_development_trial_roles import DEVELOPMENT
from reverse_animal_electrode_experiment import aggregate, attach_references, fit_target, score_target
from reverse_animal_electrode_value import CONTROL, MC_SEEDS, POLICY, build_ranker_bank, training_records
from ridge_training import new_progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-root", "metadata-report", "trial-roles", "training-report", "reference-report", "output-dir"):
        parser.add_argument("--"+flag, type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned verified compute allocation required")
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread numerical settings required")
    if not 0 < args.max_seconds <= 270 or args.output_dir.exists():
        raise ValueError("Fresh preserved output and selected finite deadline at most270 seconds required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def deadline(*_):
        raise TimeoutError("Selected reverse-animal full-cohort deadline reached")
    def check():
        if time.monotonic()-started >= args.max_seconds:
            deadline()
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    progress = new_progress()
    progress.update(donor_label_fits=0, reverse_ranker_fits=0, target_orderings=0,
                    capacity_inner_candidate_scores=0, direction_only_models=0)
    report = {"analysis_id": "development_search", "recipe": "outputs/reverse_animal_electrode_value_proposal.md",
        "status": "running", "policy": POLICY, "control": CONTROL, "MC_seeds": list(MC_SEEDS),
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "fit_progress": progress,
        "sessions": [], "completed_session_ids": [], "stage": "declared_training_record_reuse",
        "logical_neural_bytes": 0, "unique_logical_neural_bytes": 0,
        "fitting_trial_read_calls": 0, "evaluation_trials_read": 0,
        "valid_evaluations_charged": 0, "held_openings_charged": 0, "held_session_accessed": False,
        "global_search_completed": False, "named_explanatory_amendment_activated": False,
        "raw_arrays_or_decoder_coefficients_written": False, "small_ranker_parameters_written": True,
        "target_definition": "one ten-bin time mean per original native population unit; no averaging across units",
        "ranker_objective": "weighted mean squared label error + squared coefficient norm; unpenalized intercept",
        "equal_target_session_acquisition_cost": True, "equal_total_training_information_or_cost": False,
        "training_report": str(args.training_report),
        "combined_training_and_outcome_JSON_physically_parsed_before_fits": True,
        "outcome_fields_used_for_new_fits": False,
        "boundary": "other-animal donor training fields only; logical numerical separation, not OS isolation",
        "transfer_limit": "both directions between two exposed animals; current-target calibration still required",
        "seven_donor_contrast_limit": "four versus seven donors changes count/composition; no isolated animal-identity effect"}
    exposure = Exposure(report)
    try:
        metadata, roles = json.loads(args.metadata_report.read_text()), json.loads(args.trial_roles.read_text())
        if len(roles["sessions"]) != 8 or {s["filename"] for s in roles["sessions"]} != set(DEVELOPMENT):
            raise ValueError("All eight original development roles required")
        sources = {name: selected_metadata(metadata, roles, name) for name in DEVELOPMENT}
        # The old combined JSON is an explicitly declared training input. The
        # whitelist discards outcome panels; no unopened-packet claim is made.
        packet = json.loads(args.training_report.read_text())
        records, animals, inherited = training_records(packet, sources)
        del packet
        report["inherited_calibration_cost"] = inherited
        t = time.monotonic()
        rankers = build_ranker_bank(records, animals, progress, check)
        if len(rankers) != 40 or progress["reverse_ranker_fits"] != 40:
            raise ArithmeticError("Exactly forty shared other-animal ranker states required")
        report["ranker_fit_seconds"] = time.monotonic()-t
        report["ranker_states"] = [rankers[key] for key in rankers]
        del records
        report["stage"] = "new_target_fits"
        for filename in DEVELOPMENT:
            check()
            source, role, trial_records = sources[filename]
            before_models, before_orders = progress["outer_training_refits"], progress["target_orderings"]
            t = time.monotonic()
            models, nulls, guide, selections = fit_target(args.source_root/filename, source, role, trial_records,
                                                         rankers, progress, check, exposure)
            if (progress["outer_training_refits"]-before_models != 65
                    or progress["target_orderings"]-before_orders != 20):
                raise ArithmeticError("All sixty-five models/twenty target orderings required before evaluation")
            fitted_at = time.monotonic()
            ex, ey, _ = read_trials(args.source_root/filename, source, role, trial_records, role["evaluation_trials"],
                source["common_physical_channels"], check, exposure, guide, evaluation=True)
            labels = np.asarray([trial_records[trial]["direction_index"] for trial in role["evaluation_trials"]])
            panels, seed_panels = score_target(models, nulls, ex, ey, labels, source["common_physical_channels"])
            report["sessions"].append({"filename": filename, "animal": role["animal"], "panels": panels,
                "seed_panels": seed_panels, "selection_records": selections,
                "source_roles": role_record(source, role), "target_unit_count": ey.shape[1],
                "all65_target_models_fixed_before_evaluation": True,
                "all20_target_orders_fixed_before_post_initial_reads": True,
                "training_read_and_fit_seconds": fitted_at-t, "elapsed_seconds": time.monotonic()-t})
            report["completed_session_ids"].append(filename)
            print(json.dumps({"session_completed": filename, "elapsed_seconds": time.monotonic()-t,
                              "actual_target_models": 65, "scientific_scores_released": False}), flush=True)
            del models, nulls, guide, selections, ex, ey
        if (progress["outer_training_refits"] != 520 or progress["inner_ridge_fits"] != 10400
                or progress["reverse_ranker_fits"] != 40 or progress["target_orderings"] != 160
                or progress["donor_label_fits"] != 0 or report["fitting_trial_read_calls"] != 6912
                or report["evaluation_trials_read"] != 320):
            raise ArithmeticError("Complete adopted target/transfer work counts differ")
        check()
        report["stage"] = "post_fit_reference_comparison"
        # Only now may either completed packet supply scientific score fields.
        previous = json.loads(args.training_report.read_text())
        acquired = json.loads(args.reference_report.read_text())
        report["full_electrode_score_replay_max_absolute_difference"] = attach_references(report["sessions"], acquired, previous)
        report["panels"] = aggregate(report["sessions"])
        report["fixed_prediction_session_omission"] = [
            {"excluded_session": name, "panels": aggregate([s for s in report["sessions"] if s["filename"] != name], False)}
            for name in DEVELOPMENT]
        candidate = next(panel for panel in report["panels"] if panel["policy"] == POLICY)
        report["prospective_directional_screens"] = {
            "both_transfer_directions_positive_vs_own_label_control":
                all(a["mean_delta_Q_from_"+CONTROL] > 0 for a in candidate["animals"]),
            "both_transfer_directions_positive_vs_initial_marginal":
                all(a["mean_delta_Q_from_initial_marginal"] > 0 for a in candidate["animals"]),
            "significance_or_promotion_claim": False}
        report["reference_reuse"] = {"learned_report": str(args.training_report), "acquired_report": str(args.reference_report),
            "old_outcomes_used_only_after_all_new_fits": True, "old_prediction_array_identity_claimed": False,
            "new_target_fits_reused_from_old_reports": False, "donor_training_fields_reused_before_fits": True}
        report.update(status="completed_two_reverse_animal_selector_comparisons", stage="complete",
            valid_evaluations_charged=2, valid_evaluations_total=25, search_progress_resolution_Q=.001,
            scientific_scores_released=True)
    except Exception as error:
        report.update(status="technical_failure_reverse_animal_selector_incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
                      maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      observed_utc=datetime.now(timezone.utc).isoformat())
        report["retained_report_bytes"] = 0
        for _ in range(4):
            payload = json.dumps(report, indent=2, allow_nan=False)+"\n"
            size = len(payload.encode("utf-8"))
            if report["retained_report_bytes"] == size:
                break
            report["retained_report_bytes"] = size
        (args.output_dir/"reverse_animal_electrode_value.json").write_text(payload)
    print(json.dumps({key: report[key] for key in
        ("status", "elapsed_seconds", "maximum_RSS_KiB", "valid_evaluations_charged")}), flush=True)


if __name__ == "__main__":
    main()
