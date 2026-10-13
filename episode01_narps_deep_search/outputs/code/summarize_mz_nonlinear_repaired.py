#!/usr/bin/env python3
"""Pair exact original nonlinear MZ500 with a selected fresh M0 repair1500.

The reports keep their actual phases and original/runtime specifications.
Only the existing component validator's process-local admission context changes;
no source scores, report identities or scientific comparisons are rewritten.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import patch

from qualify_assets import require_compute
import run_mz_nonlinear_m0_repair as repair
import summarize_mz_nonlinear as original


nonlinear = repair.nonlinear
PHASE = "MZ0_nonlinear_value_repaired_M0_MZ_nested_diagnostic"
SOURCE_REPORT = repair.SOURCE / nonlinear.SOURCE_REPORT
MZ_REPORT = repair.SOURCE.parent / f"20261005-MZ-nonlinear-{repair.FAILED_JOB}" / "MZ" / nonlinear.REPORT_NAME


def report_paths(repair_dir, repair_job_id):
    job = original.parent_job(repair_job_id)
    directory = Path(repair_dir).resolve()
    expected = repair.SOURCE.parent / f"20261005-MZ-nonlinear-M0-repair-{job}"
    if directory != expected.resolve():
        raise ValueError("Use the explicit selected nonlinear M0 repair directory and parent ID")
    return {"M0": directory / repair.REPORT_NAME, "MZ": MZ_REPORT}


def validate_repair(report, job_id, spec, context):
    """Reuse original score/fit/technical-prefix checks with truthful repair admission."""
    runtime = repair.runtime_specification(spec)
    validated = report.get("repair_context_validated")
    calls = report.get("fresh_fit_calls")
    fits = report.get("fits")
    if (type(validated) is not bool or report.get("failed_report") != str(repair.FAILED_REPORT)
            or report.get("numerical_override") != repair.numerical_override()
            or report.get("original_maximum_sweeps") != 500 or report.get("fitted_maximum_sweeps") != 1500
            or report.get("retry_wave_selected") is not False
            or type(calls) is not int or not 0 <= calls <= 2
            or not isinstance(fits, list) or not len(fits) <= calls <= len(fits) + 1):
        raise ValueError("Retain exact fresh-only M0 repair identity, override and fit-call prefix")
    if "implementation" in report:
        if (report["implementation"] != spec or report.get("original_implementation") != spec
                or report.get("runtime_implementation") != runtime):
            raise ValueError("Keep original500 and M0 runtime1500 specifications separately")
    elif report.get("original_implementation") is not None or report.get("runtime_implementation") is not None:
        raise ValueError("A pre-specification technical checkpoint has no loaded implementations")
    if validated:
        if (report.get("numerical_repair") != context or report.get("implementation") != spec
                or original.duration(report.get("cumulative_process_elapsed_seconds")) !=
                context["previous_process_elapsed_seconds"] + original.duration(report.get("elapsed_seconds"))):
            raise ValueError("Reconstruct exact failure/source/contacts and add prior plus new process once")
    elif (report.get("numerical_repair") is not None or calls or fits
            or report.get("source_baseline_validated") is not False
            or "cumulative_process_elapsed_seconds" in report):
        raise ValueError("No new fit or prior-cost claim may precede exact repair-context validation")
    if report.get("nonlinear_component_complete") is True and (not validated or calls != 2):
        raise ValueError("Both fresh nonlinear M0 fits require complete repair qualification")

    # Admission follows the actual repair phase; the supplied report is untouched.
    # The original500 implementation remains in the report and source checks.
    # Only each new M0 fit's maximum history length is evaluated under1500.
    fit_validator = nonlinear.validate_nonlinear_fit

    def runtime_fit(fit, model, train, held, original_spec, people):
        if model != "M0" or original_spec != spec:
            raise ValueError("Only this selected M0 component has a numerical override")
        return fit_validator(fit, model, train, held, runtime, people)

    with ExitStack() as stack:
        stack.enter_context(patch.object(nonlinear, "PHASE", repair.PHASE))
        stack.enter_context(patch.object(nonlinear, "MAX_SECONDS", repair.MAX_SECONDS))
        stack.enter_context(patch.object(nonlinear, "validate_nonlinear_fit", runtime_fit))
        row, scores, primary, identity = original.validate_component(report, "M0", job_id, spec)
    row.update(phase=report["phase"], original_maximum_sweeps=500, fitted_maximum_sweeps=1500,
        fresh_fit_calls=calls, repair_context_validated=validated,
        current_fit_support_contacts=[deepcopy(fit.get("support_contacts")) for fit in fits])
    if validated:
        row.update(numerical_repair=deepcopy(context),
            cumulative_process_elapsed_seconds=report["cumulative_process_elapsed_seconds"])
    return row, scores, primary, identity


def pool(repair_dir, repair_job_id, spec):
    paths = report_paths(repair_dir, repair_job_id)
    runtime = repair.runtime_specification(spec)
    source = json.loads(SOURCE_REPORT.read_text())
    primary = nonlinear.fixed.validate_base(source, spec)
    failed = json.loads(repair.FAILED_REPORT.read_text())
    context = repair.repair_context(failed, source, primary, spec)
    identity = (str(source["job_id"]), source["finished_utc"])
    rows, panels = [], {}
    for model in original.MODELS:
        try:
            report = json.loads(paths[model].read_text())
        except FileNotFoundError:
            rows.append({"fitted_model": model, "status": "missing_unknown", "report": str(paths[model])})
            continue
        if model == "M0":
            row, scores, saved, actual_identity = validate_repair(report, repair_job_id, spec, context)
        else:
            row, scores, saved, actual_identity = original.validate_component(report, model, repair.FAILED_JOB, spec)
            row.update(phase=report["phase"], fitted_maximum_sweeps=500,
                current_fit_support_contacts=[deepcopy(fit.get("support_contacts")) for fit in report["fits"]])
        if actual_identity is not None and actual_identity != identity:
            raise ValueError("Both components must retain the exact independently validated MZ0 source completion")
        if saved is not None and saved != primary:
            raise ValueError("Keep original primary score last bits in both components")
        row["report"] = str(paths[model])
        rows.append(row)
        if scores is not None:
            panels[model] = scores
    complete = len(panels) == 2
    result = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "generating_model": "MZ", "replicate_index": 0, "seed": nonlinear.shared.SEED, "rival": "nonlinear",
        "repair_dir": str(Path(repair_dir).resolve()), "repair_job_id": str(repair_job_id),
        "original_nonlinear_job_id": repair.FAILED_JOB,
        "source_dir": str(repair.SOURCE), "source_report": str(SOURCE_REPORT),
        "source_job_id": source["job_id"], "source_finished_utc": source["finished_utc"],
        "original_implementation": spec, "model_fit_implementations": {"M0": runtime, "MZ": spec},
        "numerical_override": repair.numerical_override(), "prior_M0_failure": context,
        "status": "nonlinear_pair_diagnostic_completed" if complete else "incomplete_nonlinear_components",
        "pooled_diagnostic_complete": complete, "complete_models": sorted(panels),
        "unknown_models": [model for model in original.MODELS if model not in panels], "components": rows,
        "fixed_primary_participant_scores": primary, "selected_replicates": [0],
        "wave_selected": False, "retry_wave_selected": False, "automatic_retry": False,
        "full_recovery_pass": None, "episode_terminal": False,
        **{field: "not_evaluated" for field in nonlinear.CLAIM_FIELDS},
        "fixed_vs_nonlinear_arbitration": "not_evaluated",
        "cost_accounting": "Each component retains its own process cost; M0 cumulative cost adds its exact failed process once. No shared baseline or sibling cost is added to it.",
        "interpretation": "Nonlinear MZ-minus-M0 on one saved MZ0 cohort, with original MZ500 and fresh repaired M0 both halves1500. Unequal powers make nonlinear M0 a curvature-adjusted reference, not globally symmetric valuation. The five lower scale-support contacts in the original M0 failure remain explicit in prior_M0_failure, separately from current fit contacts; extending the sweep limit does not resolve or erase them. Fixed-primary scores remain separate. No nonnested winner, family, recovery rate, interval, curtailment, robustness or gate verdict."}
    if complete:
        scores = {person: {"version": primary[person]["version"],
            **{model: panels[model][person][nonlinear.SCORE_KEY] for model in original.MODELS}} for person in sorted(primary)}
        nonlinear.fixed.score_panel({"participant_scores": scores}, original.MODELS)
        result.update(participants=108, combined_heldout_trials_per_participant=256,
            score_normalization=nonlinear.NORMALIZATION, nonlinear_participant_scores=scores,
            nonlinear_nested_diagnostic=nonlinear.fixed.nested_contrast(scores, "MZ", "M0"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-dir", type=Path, required=True)
    parser.add_argument("--repair-job-id", type=original.parent_job, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.implementation.resolve() != repair.IMPLEMENTATION.resolve():
        raise ValueError("Use the unchanged original implementation path")
    if args.output.exists():
        raise FileExistsError("Use a fresh summary output")
    result = pool(args.repair_dir, args.repair_job_id, json.loads(args.implementation.read_text()))
    result["recorded_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "complete_models", "unknown_models", "pooled_diagnostic_complete")}, indent=2), flush=True)
    return 0 if result["pooled_diagnostic_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
