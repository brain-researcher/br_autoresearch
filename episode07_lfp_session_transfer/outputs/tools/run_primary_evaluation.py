#!/usr/bin/env python3
"""One selected primary comparison: freeze, fit, predict, then score.

Modes are separate processes with separate data arguments. All completed
development checks precede the full analysis freeze; all fitted state precedes
held-feature access; both complete prediction panels precede a durable single
outcome-opening claim. No mode searches, retries, substitutes a finalist or
changes a role. Same-account separation remains procedural, not OS isolation.
The residual-coupling follow-up is not performed by this primary analysis.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import socket
from types import SimpleNamespace
import time


STREAMS = ("history_assisted", "today_only")


def utc():
    return datetime.now(timezone.utc).isoformat()


def plain(value):
    """Represent unavailable numeric results as null, never as a valid score."""
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_claimed(handle, value):
    target = Path(handle.name)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("x") as output:
        json.dump(plain(value), output, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, target)


def eligible_pair(episode, bind_deadline, check):
    """Reuse the original controller and ten closed paired-control records."""
    from run_finalist_controls import _selected, _previous_controls
    from known_answer_cases import CASES
    from resume_known_answer_cases import validate_completed_report
    controls = episode / "outputs/runs/alternative_explanations/attempt001"
    selected, specs, progress, _ = _selected(SimpleNamespace(out=controls),
        episode / "outputs/runs/matched_search", bind_deadline, check)
    _previous_controls(controls, 11, specs, check)
    prior = json.loads((controls / "known_answer_report.json").read_text())
    recovery_path = controls.parent / "qualification_repair001/known_answer_report.json"
    report = json.loads(recovery_path.read_text())
    validate_completed_report(report, prior, tuple(specs[stream] for stream in STREAMS))
    ids = {stream: specs[stream].candidate_id for stream in STREAMS}
    if (report.get("state") != "all_four_cases_passed"
            or report.get("known_answer_recovery_complete") is not True
            or report.get("evidence_stage") != "generated_known_answer_only"
            or report.get("held_out_opened") is not False or report.get("finalists") != ids):
        raise ValueError("both unchanged finalists require all four completed recoveries")
    expected = {(case, region, stream) for case in CASES for region in ("M1", "PMd") for stream in STREAMS}
    results = report.get("results", [])
    if (len(results) != 16
            or {(item.get("case"), item.get("region"), item.get("stream")) for item in results} != expected
            or any(item.get("passed") is not True
                   or item.get("finalist_id") != ids[item["stream"]] for item in results)):
        raise ValueError("every fixed recovery cell must pass for its selected finalist")
    expected_fits = 8 * sum(len(spec.members) for spec in specs.values())
    if report.get("completed_member_predictions") != expected_fits:
        raise ValueError("recoveries must complete the exact unchanged finalist member roster")
    separation = json.loads((episode / "outputs/runs/source_support/attempt002/data_separation_report.json").read_text())
    if (separation.get("analysis_id") != "data_separation" or separation.get("passed") is not True
            or separation.get("candidate_closed_role_requests_rejected") is not True
            or separation.get("closed_neural_values_read_by_access_check") is not False
            or separation.get("proof_scope") != "separate_files_and_runner_interface_not_OS_isolation"):
        raise ValueError("the existing passed candidate/closed-target separation demonstration is required")
    check()
    return selected, tuple(specs[stream] for stream in STREAMS), progress


def analysis_choices(finalists):
    from primary_inference import DAY_ORDER, REGIONS, COMPARATORS, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, MARGIN, ALPHA
    return plain({"finalists": [asdict(spec) for spec in finalists],
        "day_order": DAY_ORDER, "regions": REGIONS, "comparators": COMPARATORS,
        "score": "joint_native_SSE_over_per_neuron_centered_SST_negative_scores_retained",
        "weighting": "equal_days_within_animal_equal_animals",
        "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
        "rng_algorithm": "PCG64",
        "rng_order": "draw_then_day_Mihili_then_Chewie_chronological_then_direction0_to7_numeric_trial_ID_order",
        "bootstrap": "paired_whole_trials_within_day_direction_shared_across_regions_and_models_recomputed_SST",
        "margin": MARGIN, "familywise_alpha": ALPHA,
        "test": "one_sided_centered_bootstrap_plus_one_regional_IUT_two_region_Holm",
        "observed_gates": ["positive_both_comparators_in_each_animal",
            "same_day_both_comparators_positive_on_both_Mihili_and_at_least_two_Chewie_days",
            "all_five_frozen_target_day_omissions_positive_in_each_remaining_animal_and_aggregate"],
        "undefined_draw": "inference_unavailable_no_deletion_redraw_or_denominator_clipping",
        "inference_scope": "conditional_on_five_fixed_days_two_animals_and_fixed_fits",
        "residual_followup": "not_performed_in_this_primary_evaluation",
        "claim_limit": "prediction_transfer_only_no_coupling_or_mean_only_mechanism_claim",
        "later_explanatory_choices": "exploratory_on_these_outcomes_new_sequestered_days_required_for_confirmation",
        "public_outcome_exposure": "public_release_and_prior_campaign_were_previously_outcome_exposed",
        "access_boundary": "same_account_procedural_not_OS_isolation"})


def load_freeze(out, finalists, selected, check):
    check()
    value = json.loads((out / "analysis_freeze.json").read_text())
    if (value.get("complete") is not True or value.get("held_out_opened") is not False
            or value.get("scientific_checks_complete") is not True
            or value.get("choices") != analysis_choices(finalists)
            or value.get("search_clock") != selected["search_clock"]
            or value.get("selection_completed_utc") != selected["selection_completed_utc"]):
        raise ValueError("complete unchanged analysis freeze required")
    return value


def execute(args, episode, state_path, bind_deadline, check):
    selected, finalists, progress = eligible_pair(episode, bind_deadline, check)
    if args.mode == "freeze":
        record = {"complete": True, "scientific_checks_complete": True,
            "held_out_opened": False, "frozen_utc": utc(), "job_id": os.environ["SLURM_JOB_ID"],
            "search_clock": selected["search_clock"],
            "selection_completed_utc": selected["selection_completed_utc"],
            "choices": analysis_choices(finalists), "search_progress": asdict(progress),
            "ten_paired_controls_complete": True, "four_fixed_recoveries_complete": True,
            "known_answer_report": "outputs/runs/alternative_explanations/qualification_repair001/known_answer_report.json"}
        with (args.out / "analysis_freeze.json").open("x") as handle:
            save_claimed(handle, record)
        return
    freeze = load_freeze(args.out, finalists, selected, check)
    if args.mode != "score" and (args.out / "opening.json").exists():
        raise PermissionError("an outcome opening already exists; never refit or repredict")
    if args.mode == "fit":
        from candidate_predictions import PANEL_KEYS
        from final_fit import fit_final_cell, assemble_final_pair, save_final_pair
        from native_trials import load_native_trials
        from view_access import CandidateViews
        if (args.out / "feature_opening.json").exists():
            raise PermissionError("fitted state must precede any held feature access")
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        status = {"state": "fitting", "complete": False, "held_out_opened": False,
            "started_utc": utc(), "job_id": os.environ["SLURM_JOB_ID"], "completed_cells": []}
        with (state_path.parent / "fit_status.json").open("x") as handle:
            save_claimed(handle, status)
            try:
                views, cells = CandidateViews(args.candidate_directory), []
                for finalist in finalists:
                    for animal, day, region in PANEL_KEYS:
                        check()
                        days = load_native_trials(views, view_role="fit", animal=animal,
                            target_day=day, region=region, stream=finalist.stream, check=check)
                        cells.append(fit_final_cell(finalist, days, check=check))
                        del days
                        status["completed_cells"].append([finalist.stream, animal, day, region])
                        save_claimed(handle, status)
                pair = assemble_final_pair(finalists, cells, check=check)
                save_final_pair(state_path, pair, check=check)
                status.update(state="frozen_fits_complete", complete=True)
            except BaseException as error:
                status.update(state="failed_incomplete", error_type=type(error).__name__, error=str(error))
                raise
            finally:
                status["ended_utc"] = utc()
                save_claimed(handle, status)
        return
    if args.mode == "predict":
        from final_fit import load_final_pair, predict_final_cell
        from final_predictions import HeldOutPredictionWriter
        from final_view_access import HeldOutFeatureViews
        from native_trials import load_held_out_trials
        from candidate_predictions import PANEL_KEYS
        import numpy as np
        pair = load_final_pair(state_path, finalists, check=check)
        fit_status = json.loads((state_path.parent / "fit_status.json").read_text())
        if (fit_status.get("state") != "frozen_fits_complete" or fit_status.get("complete") is not True
                or fit_status.get("held_out_opened") is not False or not fit_status.get("ended_utc")):
            raise ValueError("completed frozen fitting stage required before held feature access")
        fits = {(cell.stream, cell.animal, cell.target_day, cell.region): cell for cell in pair.cells}
        status = {"state": "held_features_contact_started", "complete": False,
            "held_out_spike_contact": False, "started_utc": utc(),
            "job_id": os.environ["SLURM_JOB_ID"], "analysis_frozen_utc": freeze["frozen_utc"]}
        with (args.out / "feature_opening.json").open("x") as handle:
            save_claimed(handle, status)
            try:
                views = HeldOutFeatureViews(args.heldout)
                with HeldOutPredictionWriter(args.out / "paired_predictions.h5",
                        history_candidate_id=finalists[0].candidate_id,
                        today_candidate_id=finalists[1].candidate_id, check=check) as writer:
                    for animal, day, region in PANEL_KEYS:
                        values = []
                        for stream in STREAMS:
                            features = load_held_out_trials(views, animal=animal, target_day=day,
                                region=region, stream=stream, check=check)[0]
                            values.append(predict_final_cell(fits[stream, animal, day, region], features, check=check))
                        if not np.array_equal(values[0][1].values, values[1][1].values):
                            raise ValueError("both frozen streams require the identical calibration comparator")
                        writer.write(values[0][0], values[1][0], values[0][1])
                status.update(state="both_prediction_panels_complete", complete=True)
            except BaseException as error:
                status.update(state="failed_incomplete", error_type=type(error).__name__, error=str(error))
                raise
            finally:
                status["ended_utc"] = utc()
                save_claimed(handle, status)
        return
    features = json.loads((args.out / "feature_opening.json").read_text())
    if (features.get("state") != "both_prediction_panels_complete" or features.get("complete") is not True
            or features.get("held_out_spike_contact") is not False or not features.get("ended_utc")
            or features.get("analysis_frozen_utc") != freeze["frozen_utc"]):
        raise ValueError("completed paired prediction stage for this exact freeze required before scoring")
    from final_score import score_primary_held_out
    status = {"state": "before_opening", "complete": False,
        "held_out_spike_contact": False, "job_id": os.environ["SLURM_JOB_ID"],
        "analysis_frozen_utc": freeze["frozen_utc"],
        # Score-free operational facts for the separately scoped EP06 handoff.
        # Do not export search_progress, model choices or scientific results.
        "analysis_freeze_complete": freeze["complete"],
        "scientific_checks_complete": freeze["scientific_checks_complete"],
        "ten_paired_controls_complete": freeze["ten_paired_controls_complete"],
        "four_fixed_recoveries_complete": freeze["four_fixed_recoveries_complete"]}
    claim = []
    def begin_opening():
        check()
        if claim:
            raise PermissionError("only one primary outcome opening is allowed")
        handle = (args.out / "opening.json").open("x")
        claim.append(handle)
        status.update(state="held_out_spike_contact_started", held_out_spike_contact=True, started_utc=utc())
        save_claimed(handle, status)
    try:
        result = score_primary_held_out(args.heldout, args.out / "paired_predictions.h5",
            history_candidate_id=finalists[0].candidate_id, today_candidate_id=finalists[1].candidate_id,
            begin_opening=begin_opening, check=check)
        report = asdict(result)
        report.update(scientific_checks_certified=True, authorizes_opening=False,
            evidence_stage="single_primary_held_out_evaluation", choices=freeze["choices"],
            supported_regions=[region for region, passed in zip(result.regions, result.arithmetic_pass) if passed],
            regional_decisions={region: ("inference_unavailable" if not available else
                "supported" if passed else "prespecified_support_rule_not_met")
                for region, available, passed in zip(result.regions, result.margins.available, result.arithmetic_pass)})
        with (args.out / "primary_report.json").open("x") as output:
            save_claimed(output, report)
        status.update(state="primary_evaluation_complete", complete=True)
    except BaseException as error:
        status.update(state="failed_incomplete", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if claim:
            status["ended_utc"] = utc()
            try:
                save_claimed(claim[0], status)
            finally:
                claim[0].close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("freeze", "fit", "predict", "score"))
    parser.add_argument("--candidate-directory", type=Path)
    parser.add_argument("--heldout", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    if (not os.environ.get("SLURM_JOB_ID", "").isdigit() or "-ln" in socket.gethostname()
            or os.environ.get("SLURM_CPUS_PER_TASK") not in ("1", "2")
            or os.environ.get("SLURM_MEM_PER_NODE") != "7168"):
        raise SystemExit("primary execution requires its selected bounded compute allocation")
    expected = {"freeze": (False, False), "fit": (True, False), "predict": (False, True), "score": (False, True)}
    if (args.candidate_directory is not None, args.heldout is not None) != expected[args.mode]:
        raise SystemExit("mode arguments cannot cross fitting and held roles")
    episode = Path(__file__).resolve().parents[2]
    state_path = episode / "outputs/runs/final_fit/attempt001/frozen_final_pair.pkl"
    if (args.out.resolve() != episode / "outputs/runs/primary_evaluation/attempt001"
            or not args.out.is_dir() or not state_path.parent.is_dir()):
        raise SystemExit("use the selected final-fit and primary-evaluation directories")
    roles = Path(os.environ["SCRATCH"]) / "br_autoresearch/episode07_lfp_session_transfer/source_support_attempt002"
    for path, allowed in ((args.candidate_directory, roles / "candidate"), (args.heldout, roles / "sealed/heldout.h5")):
        if path is not None and path.resolve() != allowed.resolve():
            raise SystemExit("only the original published role path is accepted")
    remaining = args.deadline_epoch - time.time()
    if not math.isfinite(remaining) or not 0 < remaining <= 3600:
        raise SystemExit("a future deadline of at most one hour is required")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    deadline = [args.deadline_epoch]
    def bind_deadline(expiry):
        deadline[0] = min(deadline[0], expiry)
    def check():
        if time.time() >= deadline[0]:
            raise TimeoutError("primary allocation or original search deadline reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > int(6.5 * 2**30):
            raise MemoryError("primary peak RSS exceeds6.5GiB")
    check()
    execute(args, episode, state_path, bind_deadline, check)
    print(json.dumps({"mode": args.mode, "completed_utc": utc()}), flush=True)


if __name__ == "__main__":
    main()
