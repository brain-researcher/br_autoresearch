#!/usr/bin/env python3
"""Compare explicit repaired nonlinear M0 and MZ on the unchanged saved MZ0.

M0 has two fresh1500 fits. MZ preserves its original500 first fit and has one
fresh1500 reverse fit. These numerical histories and both failed attempts stay
separate; only two complete repaired components supply the nested diagnostic.
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
import run_mz_nonlinear_mz_reverse_repair as reverse
import summarize_mz_nonlinear_repaired as m0_pool


nonlinear, original = m0_pool.nonlinear, m0_pool.original
M0_REPAIR_JOB = "46611338"
M0_REPORT = reverse.SOURCE.parent / f"20261005-MZ-nonlinear-M0-repair-{M0_REPAIR_JOB}" / m0_pool.repair.REPORT_NAME
PHASE = "MZ0_nonlinear_value_both_repaired_nested_diagnostic"


def report_paths(repair_dir, repair_job_id):
    job = original.parent_job(repair_job_id)
    directory = Path(repair_dir).resolve()
    expected = reverse.SOURCE.parent / f"20261005-MZ-nonlinear-MZ-reverse-repair-{job}"
    if directory != expected.resolve():
        raise ValueError("Use the explicit selected MZ reverse repair directory and parent ID")
    return {"M0": M0_REPORT, "MZ": directory / reverse.REPORT_NAME}


def validate_reverse(report, job_id, spec, first, context):
    """Retain exact first-fit contents and mixed original500/runtime1500 validation."""
    runtime = reverse.runtime_specification(spec)
    validated = report.get("repair_context_validated")
    reused, fresh, rescored = (report.get(key) for key in
        ("first_half_fit_reused", "fresh_second_half_fit_called", "first_half_rescored"))
    fits, calls = report.get("fits"), report.get("fresh_fit_calls")
    if (type(validated) is not bool or any(type(value) is not bool for value in (reused, fresh, rescored))
            or report.get("failed_report") != str(reverse.FAILED_REPORT)
            or report.get("numerical_override") != reverse.numerical_override()
            or report.get("per_half_provenance") != reverse.per_half_provenance()
            or report.get("original_maximum_sweeps") != 500
            or report.get("preserved_first_validation_maximum_sweeps") != 500
            or report.get("fresh_reverse_maximum_sweeps") != 1500
            or report.get("retry_wave_selected") is not False
            or type(calls) is not int or calls not in (0, 1) or calls != int(fresh)
            or not isinstance(fits, list) or len(fits) > 2
            or (fits and not reused) or (rescored and not reused)
            or rescored != (report.get("first_half_rescore_elapsed_seconds") is not None)
            or (not fits and report.get("first_half_copy_elapsed_seconds") is not None)
            or (fresh and (not rescored or len(fits) < 1)) or (len(fits) == 2 and not fresh)):
        raise ValueError("Retain the exact reused-first/fresh-reverse numerical provenance")
    if "implementation" in report:
        if (report["implementation"] != spec or report.get("original_implementation") != spec
                or report.get("runtime_implementation") != runtime):
            raise ValueError("Keep original500 and reverse runtime1500 implementations separately")
    elif report.get("original_implementation") is not None or report.get("runtime_implementation") is not None:
        raise ValueError("A pre-specification technical checkpoint has no loaded implementations")
    if validated:
        if (report.get("numerical_repair") != context or report.get("implementation") != spec
                or original.duration(report.get("cumulative_process_elapsed_seconds")) !=
                context["previous_process_elapsed_seconds"] + original.duration(report.get("elapsed_seconds"))):
            raise ValueError("Reconstruct the exact failure, scale contacts and prior-plus-new process cost")
    elif (report.get("numerical_repair") is not None or reused or fresh or rescored or fits
            or report.get("source_baseline_validated") is not False
            or "cumulative_process_elapsed_seconds" in report):
        raise ValueError("No fitted evidence or cost lineage may precede reverse-repair qualification")
    if fits:
        if ({key: value for key, value in fits[0].items() if key != "elapsed_seconds"} !=
                {key: value for key, value in first.items() if key != "elapsed_seconds"}
                or original.duration(report.get("first_half_copy_elapsed_seconds")) != fits[0].get("elapsed_seconds")):
            raise ValueError("Preserve all original first-fit contents except its separately recorded copy duration")
    for field in ("first_half_copy_elapsed_seconds", "first_half_rescore_elapsed_seconds"):
        if report.get(field) is not None:
            original.duration(report[field])
    if rescored:
        original.duration(report.get("first_half_rescore_elapsed_seconds"))
    if report.get("nonlinear_component_complete") is True and not (validated and reused and fresh and rescored):
        raise ValueError("A completed repaired MZ requires original-first rescoring and fresh reverse fitting")

    fit_validator = nonlinear.validate_nonlinear_fit

    def mixed_fit(fit, model, train, held, original_spec, people):
        if model != "MZ" or original_spec != spec:
            raise ValueError("Only the selected MZ reverse fit has this numerical override")
        fit_spec = spec if (tuple(train), tuple(held)) == nonlinear.HALVES[0] else runtime
        return fit_validator(fit, model, train, held, fit_spec, people)

    with ExitStack() as stack:
        stack.enter_context(patch.object(nonlinear, "PHASE", reverse.PHASE))
        stack.enter_context(patch.object(nonlinear, "MAX_SECONDS", reverse.MAX_SECONDS))
        stack.enter_context(patch.object(nonlinear, "validate_nonlinear_fit", mixed_fit))
        row, scores, primary, identity = original.validate_component(report, "MZ", job_id, spec)
    if report["scored_primary_directions"] and not rescored:
        raise ValueError("A recorded scored first direction requires explicit rescoring evidence")
    row.update(phase=report["phase"], per_half_provenance=reverse.per_half_provenance(),
        first_half_fit_reused=reused, first_half_rescored=rescored, fresh_fit_calls=calls,
        first_half_copy_elapsed_seconds=report.get("first_half_copy_elapsed_seconds"),
        first_half_rescore_elapsed_seconds=report.get("first_half_rescore_elapsed_seconds"),
        repair_context_validated=validated,
        current_fit_support_contacts=[deepcopy(fit.get("support_contacts")) for fit in fits])
    if validated:
        row.update(numerical_repair=deepcopy(context),
            cumulative_process_elapsed_seconds=report["cumulative_process_elapsed_seconds"])
    return row, scores, primary, identity


def pool(repair_dir, repair_job_id, spec):
    paths = report_paths(repair_dir, repair_job_id)
    runtime = reverse.runtime_specification(spec)
    source = json.loads(m0_pool.SOURCE_REPORT.read_text())
    primary = nonlinear.fixed.validate_base(source, spec)
    m0_failed = json.loads(m0_pool.repair.FAILED_REPORT.read_text())
    m0_context = m0_pool.repair.repair_context(m0_failed, source, primary, spec)
    mz_failed = json.loads(reverse.FAILED_REPORT.read_text())
    first, mz_context = reverse.repair_context(mz_failed, source, primary, spec)
    identity = (str(source["job_id"]), source["finished_utc"])
    rows, panels = [], {}
    for model in original.MODELS:
        try:
            report = json.loads(paths[model].read_text())
        except FileNotFoundError:
            rows.append({"fitted_model": model, "status": "missing_unknown", "report": str(paths[model])})
            continue
        if model == "M0":
            row, scores, saved, actual_identity = m0_pool.validate_repair(report, M0_REPAIR_JOB, spec, m0_context)
        else:
            row, scores, saved, actual_identity = validate_reverse(report, repair_job_id, spec, first, mz_context)
        if actual_identity is not None and actual_identity != identity:
            raise ValueError("Both repaired components must retain the exact original source completion")
        if saved is not None and saved != primary:
            raise ValueError("Keep identical original fixed-primary score last bits in both repairs")
        row["report"] = str(paths[model])
        rows.append(row)
        if scores is not None:
            panels[model] = scores
    complete = len(panels) == 2
    result = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "generating_model": "MZ", "replicate_index": 0, "seed": nonlinear.shared.SEED, "rival": "nonlinear",
        "repair_dir": str(Path(repair_dir).resolve()), "M0_repair_job_id": M0_REPAIR_JOB,
        "MZ_reverse_repair_job_id": str(repair_job_id),
        "source_dir": str(reverse.SOURCE), "source_report": str(m0_pool.SOURCE_REPORT),
        "source_job_id": source["job_id"], "source_finished_utc": source["finished_utc"],
        "original_implementation": spec, "runtime_implementation": runtime,
        "per_model_half_maximum_sweeps": {"M0": [1500, 1500], "MZ": [500, 1500]},
        "prior_failures": {"M0": m0_context, "MZ": mz_context},
        "status": "nonlinear_pair_diagnostic_completed" if complete else "incomplete_nonlinear_components",
        "pooled_diagnostic_complete": complete, "complete_models": sorted(panels),
        "unknown_models": [model for model in original.MODELS if model not in panels], "components": rows,
        "fixed_primary_participant_scores": primary, "selected_replicates": [0],
        "wave_selected": False, "retry_wave_selected": False, "automatic_retry": False,
        "full_recovery_pass": None, "episode_terminal": False,
        **{field: "not_evaluated" for field in nonlinear.CLAIM_FIELDS},
        "fixed_vs_nonlinear_arbitration": "not_evaluated",
        "cost_accounting": "Each component's cumulative duration adds its own exact failed process and new process once. Preserved first-fit cost, copy/rescore cost, source baseline and sibling processes are never added again.",
        "interpretation": "Nonlinear MZ-minus-M0 on one saved MZ0 cohort: M0 uses two fresh1500 fits; MZ preserves its first500 fit with recomputed scores and freshly fits only reverse1500. Unequal powers make nonlinear M0 a curvature-adjusted reference, not globally symmetric valuation. Both original failures' scale contacts remain explicit, including five in failed M0 and seven in failed MZ reverse; numerical completion does not erase them. Fixed-primary scores remain separate. No nonnested winner, family, recovery rate, interval, curtailment, robustness or gate verdict."}
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
    if args.implementation.resolve() != reverse.IMPLEMENTATION.resolve():
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
