#!/usr/bin/env python3
"""Resolve unique fixed-menu maxima in two preserved synthetic cell summaries.

Interpret the nine already-qualified contrasts in each saved menu. Recheck
identities, margin flags and the nested partial order; do not independently
recompute paired means or standard errors. Incomparable maxima and missing or
technical menus remain unknown in the fixed 200-cohort denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace

from qualify_assets import require_compute


EPISODE = Path(__file__).resolve().parents[2]
IMPLEMENTATION = Path(__file__).with_name("behavioral_implementation.json")
SOURCE_ROOT = EPISODE / "outputs/runs/behavioral_mechanism_kill_test/20261005-terminal-summaries-46607960"
SOURCES = {"MZ_SESOI": SOURCE_ROOT / "mz199.json", "M0_null": SOURCE_ROOT / "null200.json"}
GENERATORS = {"MZ_SESOI": "MZ", "M0_null": "M0"}
PHASE = "resolved_fixed_menu_counts"
MAX_SECONDS, DENOMINATOR = 240, 200
MZ_FRESH, MZ_REVERSE, NULL_ADDITIONAL = 46597540, 46597546, 46596399
NULL_REPAIRS = {43: 46599566, 61: 46599991, 168: 46599991}
NULL_REVERSES = {12: 46602112, 31: 46601362, 37: 46602112, 109: 46602112, 135: 46602112,
                 140: 46602543, 159: 46602543, 164: 46602112, 183: 46600829, 188: 46602897}
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
DIAGNOSTIC_FIELDS = ("nested_comparisons", "nested_comparison_eligibility",
                     "maximal_eligible_models_diagnostic", "multiple_incomparable_eligible_models")


def load_helpers():
    # Historical modules may set thread environment variables. Load them only
    # after checking the actual caller's compute/thread environment in main.
    import summarize_mz_fixed_menu as fixed
    import summarize_mz_repaired_menu_wave as mz
    import summarize_mz_null_indexed_menu as null
    return SimpleNamespace(fixed=fixed, mz=mz, null=null)


def resolve_contrasts(contrasts, helpers):
    """Pure resolver of stored statistics, with no incomparable tie-break."""
    menu = helpers.fixed
    if not isinstance(contrasts, dict) or set(contrasts) != {f"{model}-{base}" for model, base in menu.EDGES}:
        raise ValueError("Require exactly the original nine nested contrast edges")
    finite = lambda value: type(value) in (int, float) and math.isfinite(value)
    fields = {"model", "nested_baseline", "participants", "mean_gain_nats_per_trial", "participant_SE",
              "version_means", "participant_median", "requirements", "qualifies_nested_comparison"}
    for model, base in menu.EDGES:
        contrast = contrasts[f"{model}-{base}"]
        if (not isinstance(contrast, dict) or set(contrast) != fields or contrast.get("model") != model
                or contrast.get("nested_baseline") != base or type(contrast.get("participants")) is not int
                or contrast["participants"] != 108 or not finite(contrast.get("mean_gain_nats_per_trial"))
                or not finite(contrast.get("participant_SE")) or contrast["participant_SE"] < 0
                or not finite(contrast.get("participant_median"))
                or not isinstance(contrast.get("version_means"), dict) or set(contrast["version_means"]) != {"EI", "ER"}
                or not all(finite(value) for value in contrast["version_means"].values())):
            raise ValueError("Stored contrast identity, 108-person support or finite statistics changed")
        requirements = {"mean_gain_at_least_0p01": contrast["mean_gain_nats_per_trial"] >= .01,
                        "mean_gain_greater_than_two_participant_SE": contrast["mean_gain_nats_per_trial"] > 2 * contrast["participant_SE"]}
        if (contrast.get("requirements") != requirements
                or any(type(value) is not bool for value in contrast["requirements"].values())
                or type(contrast.get("qualifies_nested_comparison")) is not bool
                or contrast["qualifies_nested_comparison"] != all(requirements.values())):
            raise ValueError("Stored contrast flags differ from the unchanged >=0.01 and >2SE margins")
    eligibility = {model: all(contrasts[f"{model}-{base}"]["qualifies_nested_comparison"]
                              for base in menu.NESTED[model]) for model in menu.MODELS}
    maxima = [model for model in menu.MODELS if eligibility[model] and not any(
        eligibility[other] and menu.MECHANISMS[model] < menu.MECHANISMS[other] for other in menu.MODELS)]
    diagnostics = {"nested_comparisons": deepcopy(contrasts), "nested_comparison_eligibility": eligibility,
                   "eligible_models_diagnostic": [model for model in menu.MODELS if eligibility[model]],
                   "maximal_eligible_models_diagnostic": maxima,
                   "multiple_incomparable_eligible_models": len(maxima) > 1}
    if not maxima:
        raise ValueError("The fixed menu always contains the eligible empty M0 mechanism set")
    model = maxima[0] if len(maxima) == 1 else None
    return {**diagnostics, "resolved_model": model,
            "resolved_mechanisms": sorted(helpers.fixed.MECHANISMS[model]) if model is not None else None,
            "decision_reason": "unique_maximal_eligible_mechanism_set" if model is not None else "incomparable_maximal_sets"}


def expected_identity(cell, index, helpers):
    if cell == "MZ_SESOI":
        paths = helpers.mz.input_paths(index, MZ_FRESH, MZ_REVERSE)
        phase = ("MZ0_fixed_menu_nested_diagnostics" if index == 0 else
                 helpers.mz.repaired42.PHASE if index == 42 else
                 "MZ_SESOI_saved_replicate_menu_nested_diagnostics" if index in (92, 105) else
                 "MZ_runtime1500_repaired_menu_nested_diagnostics")
        return {"seed": helpers.mz.saved.cell_seed("MZ", index), "phase": phase,
                "paths": {} if paths is None else {key: str(path) for key, path in paths.items()},
                "source_dir": None if paths is None else str(paths["baseline"].parent)}
    paths = helpers.null.input_paths(index, NULL_ADDITIONAL, NULL_REPAIRS, NULL_REVERSES)
    directory = helpers.null.menu.null.SOURCE if index == 0 else helpers.null.simple.source_directory(index)
    return {"seed": helpers.null.baseline.indexed.null_seed(index),
            "phase": helpers.null.pilot_menu.PHASE if index == 0 else "M0_null_indexed_complete_menu_nested_diagnostics",
            "paths": {key: str(path) for key, path in paths.items()}, "source_dir": str(directory)}


def validate_scope(report, spec):
    if (report.get("analysis_id") != "behavioral_mechanism_kill_test" or report.get("implementation") != spec
            or report.get("family_selection") != "not_evaluated" or report.get("confusion_rate") != "not_evaluated"
            or report.get("group_recovery") != "not_evaluated" or report.get("full_gate") != "not_evaluated"
            or report.get("full_recovery_pass") is not None or report.get("episode_terminal") is not False):
        raise ValueError("Only the unchanged historically diagnostic-only source summaries are admitted")


def validate_saved_menu(summary, cell, index, expected, spec, helpers):
    validate_scope(summary, spec)
    if (summary.get("phase") != expected["phase"] or summary.get("generating_model") != GENERATORS[cell]
            or type(summary.get("replicate_index")) is not int or summary["replicate_index"] != index
            or type(summary.get("seed")) is not int or summary["seed"] != expected["seed"]
            or summary.get("source_dir") != expected["source_dir"]
            or [summary.get(field) for field in ("participants", "runs", "trials")] != [108, 432, 27648]
            or summary.get("model_menu") != list(helpers.fixed.MODELS)
            or ("selected_replicates" in summary and summary["selected_replicates"] != [index])):
        raise ValueError("Saved menu phase, cell, index, seed, original source or support changed")
    try:
        if datetime.fromisoformat(summary["source_finished_utc"]).utcoffset() is None:
            raise ValueError("Source completion requires a timezone")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Retain the validated source completion timestamp") from error
    resolved = resolve_contrasts(summary.get("nested_comparisons"), helpers)
    if (any(summary.get(field) != resolved[field] for field in DIAGNOSTIC_FIELDS)
            or any(type(value) is not bool for value in summary["nested_comparison_eligibility"].values())
            or type(summary.get("multiple_incomparable_eligible_models")) is not bool
            or ("eligible_models_diagnostic" in summary
                and summary["eligible_models_diagnostic"] != resolved["eligible_models_diagnostic"])):
        raise ValueError("Stored nested flags, eligibility or maxima disagree with their qualified contrasts")
    return resolved


def initial_rows(cell):
    return [{"replicate_index": index, "status": "unknown", "decision_reason": "source_not_yet_validated",
             "source_summary": str(SOURCES[cell])} for index in range(DENOMINATOR)]


def validate_cell(document, cell, spec, helpers, *, check=lambda: None, rows=None):
    validate_scope(document, spec)
    expected_phase = helpers.mz.PHASE if cell == "MZ_SESOI" else helpers.null.PHASE
    selected = [i for i in range(200) if cell != "MZ_SESOI" or i != 1]
    if (document.get("phase") != expected_phase or document.get("generating_model") != GENERATORS[cell]
            or type(document.get("fixed_cohort_count")) is not int or document["fixed_cohort_count"] != DENOMINATOR
            or document.get("selected_terminal_indices") != selected):
        raise ValueError("Use the exact preserved fixed200 cell and its original terminal selection")
    if cell == "MZ_SESOI":
        if document.get("fresh_array_job_id") != MZ_FRESH or document.get("reverse_array_job_id") != MZ_REVERSE:
            raise ValueError("The preserved MZ repair source arrays changed")
    elif (document.get("additional_mvc_array_job_id") != NULL_ADDITIONAL
            or document.get("mvc_repair_arrays") != {str(i): array for i, array in NULL_REPAIRS.items()}
            or document.get("mvc_reverse_arrays") != {str(i): array for i, array in NULL_REVERSES.items()}):
        raise ValueError("The preserved null completion/repair source arrays changed")
    incoming = document.get("cohorts")
    if (not isinstance(incoming, list) or len(incoming) != 200
            or any(not isinstance(row, dict) or type(row.get("replicate_index")) is not int for row in incoming)
            or sorted(row["replicate_index"] for row in incoming) != list(range(200))):
        raise ValueError("Every fixed index0..199 must appear exactly once")
    rows = initial_rows(cell) if rows is None else rows
    complete, maxima, unknown = [], Counter(), []
    complete_status = "nested_diagnostics_completed" if cell == "MZ_SESOI" else "complete_menu_diagnostic"
    for saved in sorted(incoming, key=lambda row: row["replicate_index"]):
        check()
        index = saved["replicate_index"]
        expected = expected_identity(cell, index, helpers)
        row = rows[index]
        row.update(seed=expected["seed"], source_status=saved.get("status"))
        if ("seed" in saved and saved["seed"] != expected["seed"]
                or "input_reports" in saved and saved["input_reports"] != expected["paths"]):
            raise ValueError("Cohort seed or explicit source-report mapping changed")
        if saved.get("status") == "unknown":
            if not isinstance(saved.get("reason"), str) or not saved["reason"] or "summary" in saved:
                raise ValueError("Unknown menus require a reason and cannot carry a completed summary")
            if cell == "MZ_SESOI" and index == 1 and saved["reason"] != "MC1_interrupted_retry_held":
                raise ValueError("Held MZ1 must remain the original unresolved attempt")
            row.update(decision_reason=saved["reason"], source_evidence=deepcopy(saved),
                       input_reports=deepcopy(saved.get("input_reports", expected["paths"])))
            unknown.append(index)
            continue
        if (saved.get("status") != complete_status or index not in selected
                or saved.get("input_reports") != expected["paths"] or not isinstance(saved.get("summary"), dict)):
            raise ValueError("Only the original validated complete-menu status supplies a decision")
        summary = saved["summary"]
        resolved = validate_saved_menu(summary, cell, index, expected, spec, helpers)
        complete.append(index)
        maxima[tuple(resolved["maximal_eligible_models_diagnostic"])] += 1
        row.update(decision_reason=resolved["decision_reason"], input_reports=deepcopy(expected["paths"]),
            source_menu_phase=summary["phase"], source_finished_utc=summary["source_finished_utc"],
            nested_comparisons=resolved["nested_comparisons"],
            nested_comparison_eligibility=resolved["nested_comparison_eligibility"],
            maximal_eligible_models=resolved["maximal_eligible_models_diagnostic"],
            resolved_model=resolved["resolved_model"], resolved_mechanisms=resolved["resolved_mechanisms"],
            cost_references={key: deepcopy(value) for key, value in summary.items() if key.endswith("elapsed_seconds")})
        if resolved["resolved_model"] is not None:
            row.update(status="resolved", correct_mechanism_set=resolved["resolved_model"] == GENERATORS[cell])
            if cell == "M0_null":
                mechanisms = helpers.fixed.MECHANISMS[resolved["resolved_model"]]
                row["false_components"] = {"valuation": "eta" in mechanisms,
                    "bias": bool(mechanisms & {"z", "c"}), "any_component": bool(mechanisms)}
    check()
    if document.get("completed_indices") != complete:
        raise ValueError("Stored completed-index list differs from its actual menu rows")
    if cell == "MZ_SESOI":
        expected_counts = {"completed_menus": len(complete), "unknown_menus": len(unknown),
            "maximal_eligible_model_sets_diagnostic": [{"models": list(models), "menus": count} for models, count in sorted(maxima.items())],
            "menus_with_multiple_incomparable_eligible_models": sum(count for models, count in maxima.items() if len(models) > 1)}
        unresolved = document.get("unknown_indices")
    else:
        expected_counts = {"complete_menus": len(complete), "M0_only_maximal_diagnostics": maxima[("M0",)],
            "multiple_incomparable_diagnostics": sum(count for models, count in maxima.items() if len(models) > 1),
            "unknown_cohorts": len(unknown)}
        unresolved = document.get("unresolved_indices")
    if document.get("diagnostic_counts") != expected_counts or unresolved != unknown:
        raise ValueError("Stored menu counts or unresolved-index list disagree with the recomputed rows")
    return rows


def cell_counts(cell, rows, qualified):
    resolved = [row for row in rows if row["status"] == "resolved"]
    correct = sum(row["correct_mechanism_set"] for row in resolved)
    unknown = DENOMINATOR - len(resolved)
    result = {"generating_model": GENERATORS[cell], "fixed_cohort_count": DENOMINATOR,
        "source_summary": str(SOURCES[cell]), "source_validation_complete": qualified,
        "counts_provisional": not qualified, "known_correct": correct, "known_incorrect": len(resolved) - correct,
        "unknown": unknown, "correct_fraction_bounds": [correct / DENOMINATOR, (correct + unknown) / DENOMINATOR] if qualified else None,
        "incomparable_count": sum(row["decision_reason"] == "incomparable_maximal_sets" for row in rows),
        "unknown_indices": [row["replicate_index"] for row in rows if row["status"] == "unknown"], "cohorts": rows}
    if cell == "M0_null":
        result["false_component_counts"] = {}
        for component in ("valuation", "bias", "any_component"):
            count = sum(row["false_components"][component] for row in resolved)
            result["false_component_counts"][component] = {"known_false_selections": count,
                "known_absent_selections": len(resolved) - count, "unknown": unknown,
                "fraction_bounds": [count / DENOMINATOR, (count + unknown) / DENOMINATOR] if qualified else None}
    return result


def result_document(rows, qualified, spec):
    return {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE, "implementation": spec,
        "status": "resolved_counts_completed" if all(qualified.values()) else "provisional_resolved_counts",
        "cells": {cell: cell_counts(cell, rows[cell], qualified[cell]) for cell in SOURCES},
        "mechanism_selection_rule": "unique maximal eligible set under unchanged nine nested contrasts; no choice among incomparable maxima",
        "bound_definition": "Deterministic fractions over the fixed200 indices, assigning every unknown both possible outcomes. Not confidence intervals or population probabilities.",
        "validation_scope": "Derived decision/count interpretation of two frozen qualified summaries. Rechecks saved identities, original implementation, 108-person contrast support, finite statistics, unchanged margin flags, eligibility and maxima. Preserves stored means and SEs without independently recomputing score mathematics. Source fits, participant rosters and replay retain prior qualification; optional score panels are not consumed and no source attempt or event is reread.",
        "paired_statistics_independently_recomputed": False, "participant_score_panels_consumed": False,
        "incomparable_arbitration": "not_evaluated", "interval_coverage": "not_evaluated", "group_recovery": "not_evaluated",
        "z_c_confusion_across_generators": "not_evaluated", "full_gate": "not_evaluated", "full_recovery_pass": None,
        "episode_terminal": False, "new_fits": False, "new_simulation": False, "saved_scores_rescored": False,
        "bootstrap_performed": False, "confidence_intervals": False, "automatic_retry": False,
        "cost_accounting": "Historical summary and attempt costs are references only; no input allocation is recharged.",
        "interpretation": "These two uniform synthetic cells support only resolved six-model fixed-bound counts. Unknowns are not excluded or declared failures. Other cells, robustness competitors, interval coverage and the full behavioral gate remain separate obligations."}


def write_report(output, result):
    temporary = output.with_name(output.name + ".partial")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the unchanged implementation path")
    if not 1 <= args.max_seconds <= MAX_SECONDS or any(os.environ.get(key) != "1" for key in THREAD_ENV):
        raise ValueError("Require a1..240-second cap and all six actual numerical thread limits one")
    if args.output.exists() or args.output.with_name(args.output.name + ".partial").exists():
        raise FileExistsError("Preserve all previous outputs; choose a fresh report")
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds
    rows, qualified = {cell: initial_rows(cell) for cell in SOURCES}, {cell: False for cell in SOURCES}
    spec = None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Whole-process resolved-count deadline exceeded")

    def interrupted(signum, frame):
        raise RuntimeError(f"Resolved-count publication interrupted by signal {signum}")

    def publish(result):
        result.update(recorded_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started,
                      internal_seconds=args.max_seconds, job_id=os.environ.get("SLURM_JOB_ID"))
        write_report(args.output, result)

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    succeeded, failure = False, None
    try:
        check()
        helpers = load_helpers()
        spec = json.loads(IMPLEMENTATION.read_text())
        if (spec.get("policy_version") != 6 or spec.get("state") != "recorded_before_model_scoring"
                or spec["optimizer"]["maximum_sweeps"] != 500
                or spec["recovery_plan"]["replicates_per_qualified_generating_cell"] != 200
                or spec["recovery_plan"]["primary_splits"] != [[1, 4], [2, 3]]
                or spec.get("joint_z_c_recovery_passed") is not False):
            raise ValueError("Only the original fixed200 six-model specification is admitted")
        for cell, path in SOURCES.items():
            check()
            document = json.loads(path.read_text())
            check()
            validate_cell(document, cell, spec, helpers, check=check, rows=rows[cell])
            qualified[cell] = True
            publish(result_document(rows, qualified, spec))
        check()
        result = result_document(rows, qualified, spec)
        publish(result)
        check()
        succeeded = True
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        signal.alarm(0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if succeeded:
            try:
                check()
            except BaseException as error:
                succeeded, failure = False, f"{type(error).__name__}: {error}"
        if not succeeded:
            result = result_document(rows, {cell: False for cell in SOURCES}, spec)
            result.update(status="technical_resolved_count_failure", error=failure)
            publish(result)
    print(json.dumps({"status": result["status"], "cells": {cell: {key: value for key, value in summary.items()
        if key in ("known_correct", "known_incorrect", "unknown", "correct_fraction_bounds")}
        for cell, summary in result["cells"].items()}}, allow_nan=False), flush=True)
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
