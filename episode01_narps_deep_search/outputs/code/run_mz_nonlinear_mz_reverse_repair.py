#!/usr/bin/env python3
"""Preserve nonlinear MZ's first500 fit and freshly repair reverse under1500.

The unchanged nonlinear runner owns source replay, both score directions,
checkpoint closure and the finite timer. Its first fitter dispatch returns a
validated copy; only its second dispatch performs numerical fitting.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from qualify_assets import require_compute
import run_mz_nonlinear as nonlinear
import run_mz_nonlinear_m0_repair as m0
import summarize_mz_nonlinear as admission


SOURCE, IMPLEMENTATION = nonlinear.SOURCE, nonlinear.IMPLEMENTATION
FAILED_JOB = "46606082"
FAILED_FINISHED_UTC = "2026-10-05T11:26:40.630841+00:00"
FAILED_REPORT = SOURCE.parent / f"20261005-MZ-nonlinear-{FAILED_JOB}" / "MZ" / nonlinear.REPORT_NAME
MAX_SECONDS, RUNTIME_SWEEPS = 7800, 1500
PHASE = "MZ0_nonlinear_MZ_reverse_runtime1500_repair"
REPORT_NAME = "mz_nonlinear_mz_reverse_repair.json"
NONCONVERGENCE = m0.NONCONVERGENCE
runtime_specification = m0.runtime_specification
_validate_original_fit = nonlinear.validate_nonlinear_fit


def numerical_override():
    return {"parameter": "optimizer.maximum_sweeps", "original": 500, "runtime": RUNTIME_SWEEPS,
        "preserved_first_validation_maximum_sweeps": 500,
        "likelihood_priors_support_starts_tolerances_and_first_convergence_changed": False,
        "nonconverged_optimizer_state_reused": False, "scope": "fresh_MZ_nonlinear_reverse_direction_only"}


def per_half_provenance():
    return [{"training_runs": list(train), "scoring_runs": list(held),
        "fit_source": str(FAILED_REPORT) if position == 0 else "current_reverse_repair",
        "action": "reuse_validated_fit_and_rescore" if position == 0 else "fresh_fit",
        "validation_maximum_sweeps": 500 if position == 0 else RUNTIME_SWEEPS}
        for position, (train, held) in enumerate(nonlinear.HALVES)]


def repair_context(failed, source, people, original):
    """Return a copied validated first fit and exact failed-attempt metadata.

Only participant keys are needed here; the producer first validates the full
actual source/truth, while admission supplies independently validated keys.
    """
    runtime_specification(original)
    if (failed.get("finished_utc") != FAILED_FINISHED_UTC
            or failed.get("status") != "technical_nonlinear_failure" or failed.get("error") != NONCONVERGENCE
            or failed.get("internal_seconds") != 5400
            or failed.get("source_baseline_validated") is not True or failed.get("saved_primary_scores_replayed") is not True
            or not isinstance(failed.get("fits"), list) or len(failed["fits"]) != 2
            or failed.get("scored_primary_directions") != [{"training_runs": [1, 4], "scoring_runs": [2, 3]}]):
        raise ValueError("Only the exact selected original nonlinear MZ reverse500 failure is admitted")
    # The outer repair has installed its mixed-direction validator already.
    # The immutable old panel must still use the original500 validator here.
    with patch.object(nonlinear, "validate_nonlinear_fit", _validate_original_fit):
        row, scores, primary, identity = admission.validate_component(failed, "MZ", FAILED_JOB, original)
    if (row["status"] != "technical_unknown" or scores is not None
            or identity != (str(source["job_id"]), source["finished_utc"])
            or primary != source["participant_scores"] or set(primary) != set(people) or len(people) != 108):
        raise ValueError("The failed MZ component must retain the actual original source/replay last bits")
    first, reverse = failed["fits"]
    _validate_original_fit(first, "MZ", *nonlinear.HALVES[0], original, people)
    if len(first["objective_history"]) != 153:
        raise ValueError("Preserve the exact completed153-sweep first fit")
    names = nonlinear.parameter_names("MZ", "nonlinear")
    if (reverse.get("parameter_names") != names or reverse.get("inference") != original["method"]
            or reverse.get("version_order") != ["EI", "ER"] or reverse.get("converged") is not False
            or reverse.get("subject_optimizer_failures") != []):
        raise ValueError("The reverse failure must retain its original nonlinear fit identity and no subject failures")
    theta = np.asarray(reverse.get("parameters"), dtype=float)
    if theta.shape != (108, len(names)) or not np.isfinite(theta).all():
        raise ValueError("Incomplete or nonfinite failed reverse parameters")
    for column, name in enumerate(names):
        low, high = original["parameters"][name]["bounds"]
        if np.any(theta[:, column] < low) or np.any(theta[:, column] > high):
            raise ValueError("The failed reverse fit changed the original support")
    for field, shape in (("global_mean", (len(names),)), ("version_means", (2, len(names))),
                         ("version_standard_deviations", (2, len(names)))):
        values = np.asarray(reverse.get(field), dtype=float)
        if values.shape != shape or not np.isfinite(values).all() or (field == "version_standard_deviations" and np.any(values <= 0)):
            raise ValueError("Incomplete or nonfinite failed reverse hyperparameters")
    history = reverse.get("objective_history")
    if not isinstance(history, list) or len(history) != 500 or any(not m0.finite(value) for value in history):
        raise ValueError("Require exactly500 finite failed reverse sweeps")
    change = abs(history[-1] - history[-2])
    threshold = original["optimizer"]["joint_relative_tolerance"] * (1 + abs(history[-2]))
    if not m0.finite(change) or not m0.finite(threshold) or change <= threshold:
        raise ValueError("Reverse exhaustion must exceed the unchanged convergence threshold")
    if any(admission.duration(fit["elapsed_seconds"]) > row["elapsed_seconds"] for fit in failed["fits"]):
        raise ValueError("Retain fit costs within the original process cost")
    expected_contacts = {("EI", name) for name in ("rho", "gamma_gain", "gamma_loss")} | {
        ("ER", name) for name in ("kappa", "rho", "gamma_gain", "gamma_loss")}
    for fit in failed["fits"]:
        contacts = fit.get("support_contacts")
        if (not isinstance(contacts, list) or len(contacts) != 7 or any(not isinstance(item, dict) for item in contacts)
                or {(item.get("version"), item.get("parameter")) for item in contacts} != expected_contacts):
            raise ValueError("Preserve the seven identified scale contacts in each original fit")
        for contact in contacts:
            name = contact["parameter"]
            if (contact.get("level") != "version_log_standard_deviation"
                    or contact.get("bounds") != original["parameters"][name]["individual_log_scale_bounds"]
                    or not m0.finite(contact.get("value"))):
                raise ValueError("The original lower-scale contact identity changed")
            low, high = contact["bounds"]
            if not low <= contact["value"] <= low + 1e-6 * (1 + high - low):
                raise ValueError("Both diagnosed contact panels were at the original lower support")
    metadata = {"kind": "nonlinear_MZ_reverse_500_sweep_exhaustion_preserve_first_fresh_reverse1500",
        "failed_report": str(FAILED_REPORT), "failed_job_id": str(failed["job_id"]),
        "failed_started_utc": failed["started_utc"], "failed_finished_utc": failed["finished_utc"], "failed_error": failed["error"],
        "source_job_id": source["job_id"], "source_finished_utc": source["finished_utc"],
        "original_maximum_sweeps": 500, "preserved_first_validation_maximum_sweeps": 500,
        "fresh_reverse_maximum_sweeps": RUNTIME_SWEEPS, "previous_process_elapsed_seconds": row["elapsed_seconds"],
        "previous_fit_elapsed_seconds": [fit["elapsed_seconds"] for fit in failed["fits"]],
        "previous_support_contacts": [deepcopy(fit["support_contacts"]) for fit in failed["fits"]],
        "first_half_original_fit_elapsed_seconds": first["elapsed_seconds"],
        "first_half_score_handling": "recomputed_from_preserved_parameters_no_saved_nonlinear_score_agreement_claim",
        "new_first_half_fit_elapsed_seconds_meaning": "copy_return_overhead_not_original_fit_cost",
        "final_reverse_objective_change": change, "unchanged_convergence_threshold": threshold,
        "nonconverged_checkpoint_state_reused": False,
        "second_half_initialization": "unchanged_fresh_prior_centers_and_recorded_first_sweep_offsets",
        "tolerances_priors_likelihood_support_and_first_convergence_rule_changed": False,
        "cost_accounting": "prior_failed_MZ_process_plus_new_MZ_process_once; no_duplicate_first_fit_baseline_or_sibling_cost"}
    return deepcopy(first), metadata


def run(args):
    require_compute()
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("Keep the whole-process cap within1 through7800 seconds")
    if getattr(args, "model", "MZ") != "MZ":
        raise ValueError("Only the selected nonlinear MZ reverse repair is authorized")
    output = args.output_dir.resolve()
    if output == FAILED_REPORT.parent.resolve() or FAILED_REPORT.parent.resolve() in output.parents:
        raise FileExistsError("Preserve the original nonlinear MZ failed attempt")
    source_validator, fitter = nonlinear.shared.validate_source, nonlinear.fit_partially_pooled
    fit_validator, scorer, writer = nonlinear.validate_nonlinear_fit, nonlinear.score_nonlinear_fit, nonlinear.write_report
    first = metadata = None
    calls, fresh_calls, rescore_seconds = 0, 0, None

    def validate_source(source, truth, people, original):
        nonlocal first, metadata
        source_validator(source, truth, people, original)
        first, metadata = repair_context(json.loads(FAILED_REPORT.read_text()), source, people, original)

    def dispatch_fit(training, model, rival, original):
        nonlocal calls, fresh_calls
        if metadata is None or model != "MZ" or rival != "nonlinear" or calls >= 2:
            raise ValueError("Only one validated first fit and one fresh reverse MZ fit are selected")
        calls += 1
        if calls == 1:
            return deepcopy(first)
        fresh_calls += 1
        return fitter(training, model, rival, runtime_specification(original))

    def validate_fit(fit, model, train, held, original, people):
        direction = (tuple(train), tuple(held))
        if model != "MZ" or metadata is None or direction not in nonlinear.HALVES:
            raise ValueError("Validate only the two selected nonlinear MZ directions")
        spec = original if direction == nonlinear.HALVES[0] else runtime_specification(original)
        return fit_validator(fit, model, train, held, spec, people)

    def score_fit(fit, scoring, original, check):
        nonlocal rescore_seconds
        began = nonlinear.time.perf_counter()
        values = scorer(fit, scoring, original, check)
        if tuple(fit["training_runs"]) == nonlinear.HALVES[0][0]:
            rescore_seconds = nonlinear.time.perf_counter() - began
        return values

    def checkpoint(directory, report):
        original = report.get("implementation")
        report.update(phase=PHASE, failed_report=str(FAILED_REPORT), repair_context_validated=metadata is not None,
            numerical_repair=deepcopy(metadata), original_implementation=deepcopy(original),
            runtime_implementation=runtime_specification(original) if original is not None else None,
            numerical_override=numerical_override(), original_maximum_sweeps=500,
            preserved_first_validation_maximum_sweeps=500, fresh_reverse_maximum_sweeps=RUNTIME_SWEEPS,
            fresh_fit_calls=fresh_calls, first_half_fit_reused=calls >= 1, fresh_second_half_fit_called=fresh_calls == 1,
            first_half_rescored=rescore_seconds is not None, first_half_rescore_elapsed_seconds=rescore_seconds,
            first_half_copy_elapsed_seconds=report["fits"][0]["elapsed_seconds"] if report["fits"] else None,
            per_half_provenance=per_half_provenance(), wave_selected=False, retry_wave_selected=False, automatic_retry=False,
            interpretation="Only the exact nonlinear MZ reverse500 exhaustion is repaired. Preserve and rescore first500, freshly fit reverse1500 with unchanged priors/support/likelihood/starts/tolerance. Original seven lower scale contacts in each fit remain explicit. No failed optimizer state, source simulation or baseline fit is reused as a new fit. No family, robustness, rate, interval, curtailment or gate verdict.",
            cost_accounting="elapsed_seconds is this reverse completion; cumulative adds prior MZ process once without duplicating its preserved first fit, baseline or sibling costs")
        if metadata is not None and "elapsed_seconds" in report:
            report["cumulative_process_elapsed_seconds"] = metadata["previous_process_elapsed_seconds"] + report["elapsed_seconds"]
        writer(directory, report)

    selected = argparse.Namespace(model="MZ", source_dir=args.source_dir, implementation=args.implementation,
                                  output_dir=args.output_dir, max_seconds=args.max_seconds)
    with ExitStack() as stack:
        for owner, name, value in ((nonlinear, "MAX_SECONDS", MAX_SECONDS), (nonlinear, "REPORT_NAME", REPORT_NAME),
                (nonlinear.shared, "validate_source", validate_source), (nonlinear, "fit_partially_pooled", dispatch_fit),
                (nonlinear, "validate_nonlinear_fit", validate_fit), (nonlinear, "score_nonlinear_fit", score_fit),
                (nonlinear, "write_report", checkpoint)):
            stack.enter_context(patch.object(owner, name, value))
        return nonlinear.run(selected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "status", "nonlinear_component_complete", "first_half_fit_reused",
        "fresh_fit_calls", "elapsed_seconds", "cumulative_process_elapsed_seconds", "error") if key in report}, indent=2), flush=True)
    return 0 if report["nonlinear_component_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
