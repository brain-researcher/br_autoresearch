#!/usr/bin/env python3
"""Fresh1500-sweep M0-only repair of the exact nonlinear MZ0 first-half failure.

The original nonlinear process still owns validation/replay, both fresh fits,
scoring and the whole-process timer. Only its process-local numerical bindings
and checkpoint metadata change; the frozen original500 implementation remains.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np

from qualify_assets import require_compute
import run_mz_nonlinear as nonlinear


SOURCE, IMPLEMENTATION = nonlinear.SOURCE, nonlinear.IMPLEMENTATION
FAILED_JOB = "46606082"
FAILED_FINISHED_UTC = "2026-10-05T11:10:26.656203+00:00"
FAILED_REPORT = SOURCE.parent / f"20261005-MZ-nonlinear-{FAILED_JOB}" / "M0" / nonlinear.REPORT_NAME
MAX_SECONDS, RUNTIME_SWEEPS = 15600, 1500
PHASE = "MZ0_nonlinear_M0_runtime1500_repair"
REPORT_NAME = "mz_nonlinear_m0_repair.json"
NONCONVERGENCE = "RuntimeError: Nonconverged nonlinear fit is technical; it is not a recovery failure"


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def runtime_specification(original):
    if original["optimizer"]["maximum_sweeps"] != 500:
        raise ValueError("Retain the unchanged original500 specification")
    runtime = deepcopy(original)
    runtime["optimizer"]["maximum_sweeps"] = RUNTIME_SWEEPS
    return runtime


def numerical_override():
    return {"parameter": "optimizer.maximum_sweeps", "original": 500, "runtime": RUNTIME_SWEEPS,
        "likelihood_priors_support_starts_tolerances_and_first_convergence_changed": False,
        "nonconverged_optimizer_state_reused": False,
        "scope": "fresh_M0_nonlinear_both_primary_directions_only"}


def repair_context(failed, source, people, original):
    """Validate the fixed failed attempt against the independently qualified source.

Only participant keys are consumed here. The producer first validates the full
actual source cohort/truth; admission can supply independently validated keys.
    """
    runtime_specification(original)
    if (failed.get("analysis_id") != "behavioral_mechanism_kill_test" or failed.get("phase") != nonlinear.PHASE
            or failed.get("generating_model") != "MZ" or failed.get("replicate_index") != 0
            or failed.get("seed") != nonlinear.shared.SEED or failed.get("fitted_model") != "M0"
            or failed.get("rival") != "nonlinear" or str(failed.get("job_id")) != FAILED_JOB
            or failed.get("finished_utc") != FAILED_FINISHED_UTC
            or failed.get("source_dir") != str(SOURCE) or failed.get("source_report") != str(SOURCE / nonlinear.SOURCE_REPORT)
            or failed.get("implementation") != original or failed.get("internal_seconds") != 5400
            or failed.get("status") != "technical_nonlinear_failure" or failed.get("error") != NONCONVERGENCE
            or failed.get("nonlinear_component_complete") is not False or "participant_scores" in failed
            or failed.get("source_baseline_validated") is not True or failed.get("saved_primary_scores_replayed") is not True
            or failed.get("source_job_id") != source["job_id"] or failed.get("source_finished_utc") != source["finished_utc"]
            or failed.get("saved_primary_participant_scores") != source["participant_scores"]
            or failed.get("scored_primary_directions") != [] or failed.get("selected_replicates") != [0]
            or failed.get("score_normalization") != nonlinear.NORMALIZATION
            or [failed.get(field) for field in ("participants", "runs", "trials",
                "heldout_trials_per_direction_per_participant", "combined_scored_trials_per_participant")] != [108, 432, 27648, 128, 256]
            or not isinstance(failed.get("fits"), list) or len(failed["fits"]) != 1):
        raise ValueError("Only the exact selected nonlinear M0 first-half500 failure is admitted")
    if (any(failed.get(field) != "not_evaluated" for field in nonlinear.CLAIM_FIELDS)
            or any(failed.get(field) is not False for field in ("simulation_regenerated", "simulation_copied",
                "saved_primary_fits_reused_for_nonlinear", "observed_choices_or_RTs_used", "wave_selected", "automatic_retry", "episode_terminal"))
            or failed.get("full_recovery_pass") is not None
            or failed.get("saved_score_replay_absolute_tolerance") != nonlinear.shared.REPLAY_ABSOLUTE_TOLERANCE
            or not finite(failed.get("saved_score_replay_max_absolute_discrepancy"))
            or not 0 <= failed["saved_score_replay_max_absolute_discrepancy"] <= nonlinear.shared.REPLAY_ABSOLUTE_TOLERANCE
            or not finite(failed.get("elapsed_seconds")) or failed["elapsed_seconds"] < 0):
        raise ValueError("Retain the original technical/replay/claim/cost boundary")
    try:
        began, finished = (datetime.fromisoformat(failed[key]) for key in ("started_utc", "finished_utc"))
        if began.utcoffset() is None or finished.utcoffset() is None or finished < began:
            raise ValueError("Invalid failed-attempt timestamps")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Preserve complete terminal failure timestamps") from error
    fit = failed["fits"][0]
    names = nonlinear.parameter_names("M0", "nonlinear")
    train, held = nonlinear.HALVES[0]
    if (fit.get("model") != "M0" or fit.get("rival") != "nonlinear" or fit.get("parameter_names") != names
            or fit.get("participants") != sorted(people) or len(people) != 108
            or fit.get("training_runs") != list(train) or fit.get("scoring_runs") != list(held)
            or fit.get("inference") != original["method"] or fit.get("version_order") != ["EI", "ER"]
            or fit.get("converged") is not False or fit.get("subject_optimizer_failures") != []
            or not finite(fit.get("elapsed_seconds")) or not 0 <= fit["elapsed_seconds"] <= failed["elapsed_seconds"]):
        raise ValueError("Preserve the actual participants and exact returned first nonlinear M0 fit")
    theta = np.asarray(fit.get("parameters"), dtype=float)
    if theta.shape != (108, len(names)) or not np.isfinite(theta).all():
        raise ValueError("Incomplete or nonfinite failed-fit parameters")
    for column, name in enumerate(names):
        low, high = original["parameters"][name]["bounds"]
        if np.any(theta[:, column] < low) or np.any(theta[:, column] > high):
            raise ValueError("The failed fit changed the original nonlinear support")
    for field, shape in (("global_mean", (len(names),)), ("version_means", (2, len(names))),
                         ("version_standard_deviations", (2, len(names)))):
        values = np.asarray(fit.get(field), dtype=float)
        if values.shape != shape or not np.isfinite(values).all() or (field == "version_standard_deviations" and np.any(values <= 0)):
            raise ValueError("Incomplete or nonfinite failed-fit hyperparameters")
    history = fit.get("objective_history")
    if not isinstance(history, list) or len(history) != 500 or any(not finite(value) for value in history):
        raise ValueError("Require exactly500 finite objective sweeps")
    change = abs(history[-1] - history[-2])
    threshold = original["optimizer"]["joint_relative_tolerance"] * (1 + abs(history[-2]))
    if not finite(change) or not finite(threshold) or change <= threshold:
        raise ValueError("Require exhaustion above the unchanged convergence threshold")
    contacts = fit.get("support_contacts")
    if not isinstance(contacts, list) or len(contacts) != 5:
        raise ValueError("Preserve all five recorded lower scale-support contacts")
    for contact in contacts:
        name = contact.get("parameter")
        if (contact.get("level") != "version_log_standard_deviation" or contact.get("version") not in ("EI", "ER")
                or name not in names or contact.get("bounds") != original["parameters"][name]["individual_log_scale_bounds"]
                or not finite(contact.get("value"))):
            raise ValueError("The recorded scale-contact identity changed")
        low, high = contact["bounds"]
        if not low <= contact["value"] <= low + 1e-6 * (1 + high - low):
            raise ValueError("The diagnosed five contacts were at the original lower support")
    return {"kind": "nonlinear_M0_first_half_500_sweep_exhaustion_fresh_both",
        "failed_report": str(FAILED_REPORT), "failed_job_id": str(failed["job_id"]),
        "failed_started_utc": failed["started_utc"], "failed_finished_utc": failed["finished_utc"],
        "failed_error": failed["error"], "source_job_id": source["job_id"], "source_finished_utc": source["finished_utc"],
        "original_maximum_sweeps": 500, "runtime_maximum_sweeps": RUNTIME_SWEEPS,
        "previous_process_elapsed_seconds": failed["elapsed_seconds"], "previous_fit_elapsed_seconds": [fit["elapsed_seconds"]],
        "previous_support_contacts": deepcopy(contacts), "final_objective_change": change,
        "unchanged_convergence_threshold": threshold, "completed_first_fit_available": False,
        "nonconverged_checkpoint_state_reused": False,
        "fresh_fit_directions": [{"training_runs": list(train), "scoring_runs": list(held)} for train, held in nonlinear.HALVES],
        "initialization": "unchanged_fresh_prior_centers_and_recorded_first_sweep_offsets",
        "tolerances_priors_likelihood_support_and_first_convergence_rule_changed": False,
        "cost_accounting": "prior_failed_M0_process_plus_new_M0_process_once; shared_baseline_and_sibling_MZ_costs_not_added"}


def run(args):
    require_compute()
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("Keep the whole-process cap within1 through15600 seconds")
    if getattr(args, "model", "M0") != "M0":
        raise ValueError("Only the selected nonlinear M0 repair is authorized")
    output = args.output_dir.resolve()
    if output == FAILED_REPORT.parent.resolve() or FAILED_REPORT.parent.resolve() in output.parents:
        raise FileExistsError("Preserve the original failed attempt")
    source_validator, fitter = nonlinear.shared.validate_source, nonlinear.fit_partially_pooled
    fit_validator, writer = nonlinear.validate_nonlinear_fit, nonlinear.write_report
    metadata = None
    fit_calls = 0

    def validate_source(source, truth, people, original):
        nonlocal metadata
        source_validator(source, truth, people, original)
        failed = json.loads(FAILED_REPORT.read_text())
        metadata = repair_context(failed, source, people, original)

    def fit_fresh(training, model, rival, original):
        nonlocal fit_calls
        if metadata is None or model != "M0" or rival != "nonlinear" or fit_calls >= 2:
            raise ValueError("Validate the fixed failure before at most two fresh nonlinear M0 fits")
        fit_calls += 1
        return fitter(training, model, rival, runtime_specification(original))

    def validate_fit(fit, model, train, held, original, people):
        if model != "M0":
            raise ValueError("Only new nonlinear M0 fits use runtime1500")
        return fit_validator(fit, model, train, held, runtime_specification(original), people)

    def checkpoint(directory, report):
        original = report.get("implementation")
        report.update(phase=PHASE, failed_report=str(FAILED_REPORT), repair_context_validated=metadata is not None,
            numerical_repair=deepcopy(metadata), original_implementation=deepcopy(original),
            runtime_implementation=runtime_specification(original) if original is not None else None,
            numerical_override=numerical_override(), original_maximum_sweeps=500, fitted_maximum_sweeps=RUNTIME_SWEEPS,
            fresh_fit_calls=fit_calls, wave_selected=False, retry_wave_selected=False, automatic_retry=False,
            interpretation="Only the exact nonlinear M0 first-half500 exhaustion is repaired. Both halves start fresh under1500 with unchanged priors, support, likelihood and first-convergence rule. The original500 source is validated/replayed; no failed fit state or sibling MZ fit is reused. Original five scale contacts remain visible separately from new fit contacts. No family, robustness, rate, interval, curtailment or gate verdict.",
            cost_accounting="elapsed_seconds is this new M0 process; cumulative adds the exact failed M0 process once and does not duplicate allocation, baseline or sibling costs")
        if metadata is not None and "elapsed_seconds" in report:
            report["cumulative_process_elapsed_seconds"] = metadata["previous_process_elapsed_seconds"] + report["elapsed_seconds"]
        writer(directory, report)

    selected = argparse.Namespace(model="M0", source_dir=args.source_dir, implementation=args.implementation,
                                  output_dir=args.output_dir, max_seconds=args.max_seconds)
    with ExitStack() as stack:
        for owner, name, value in ((nonlinear, "MAX_SECONDS", MAX_SECONDS), (nonlinear, "REPORT_NAME", REPORT_NAME),
                (nonlinear.shared, "validate_source", validate_source), (nonlinear, "fit_partially_pooled", fit_fresh),
                (nonlinear, "validate_nonlinear_fit", validate_fit), (nonlinear, "write_report", checkpoint)):
            stack.enter_context(patch.object(owner, name, value))
        return nonlinear.run(selected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "status", "nonlinear_component_complete",
        "repair_context_validated", "elapsed_seconds", "cumulative_process_elapsed_seconds", "error") if key in report}, indent=2), flush=True)
    return 0 if report["nonlinear_component_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
