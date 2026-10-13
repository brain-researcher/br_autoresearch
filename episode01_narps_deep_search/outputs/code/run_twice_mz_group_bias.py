#!/usr/bin/env python3
"""Group point bias of the preserved 200 twice-SESOI MZ baseline cohorts.

Reuse the qualified twice-source validators and physical effect extractor.
Every valid cohort enters, including failed predictive screens. This computes
no interval, model fit, family-selection rate or full recovery verdict.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import time

from qualify_assets import require_compute
import summarize_mz_group_recovery as group
import summarize_mz_twice_wave as twice


IMPLEMENTATION = twice.IMPLEMENTATION
PHASE = "MZ_twice_SESOI_group_point_bias_summary"
TRUE_EFFECT, RELATIVE_BIAS_LIMIT = .10, group.RELATIVE_BIAS_LIMIT
MAX_SECONDS = 240
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def validate_selection(indices, first_array, remaining_array, spec):
    if (any(type(index) is not int for index in indices) or sorted(indices) != list(range(twice.PLANNED))
            or type(first_array) is not int or first_array != twice.FIRST_ARRAY
            or type(remaining_array) is not int or remaining_array != twice.REMAINING_ARRAY):
        raise ValueError("Select exactly indices 0 through 199 and the preserved first/remaining arrays")
    if (spec.get("policy_version") != 6 or spec.get("state") != "recorded_before_model_scoring"
            or spec["optimizer"]["maximum_sweeps"] != 500
            or spec["recovery_plan"]["replicates_per_qualified_generating_cell"] != twice.PLANNED
            or spec["recovery_plan"]["primary_splits"] != [[1, 4], [2, 3]]
            or spec["contaminant"]["fraction"] != .02 or spec["contaminant"]["miss_probability"] != .01):
        raise ValueError("Use the unchanged original 500-sweep, 200-cohort implementation")


def unknown_rows():
    return [{"replicate_index": index, "seed": twice.levels.level_seed("twice", index),
             "status": "unknown", "reason": "selected_terminal_report_not_yet_validated"}
            for index in range(twice.PLANNED)]


def summarize_rows(rows, spec, *, allow_final=True):
    valid = [row for row in rows if row["status"] == "point_effect_validated"]
    complete = allow_final and len(valid) == twice.PLANNED
    statistics = {}
    for version in group.VERSIONS:
        mean = math.fsum(row["rejection_displacement"][version] for row in valid) / len(valid) if valid else None
        signed = mean - TRUE_EFFECT if mean is not None else None
        relative = abs(signed) / TRUE_EFFECT if signed is not None else None
        statistics[version] = {"mean_rejection_displacement": mean, "signed_bias": signed,
            "absolute_relative_bias": relative,
            "relative_bias_pass": relative <= RELATIVE_BIAS_LIMIT if complete else None}
    return {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "generating_model": "MZ", "effect_level": "twice", "SESOI_level": 2., "generating_z": .40,
        "focal_generating_truth": twice.levels.focal_truth("twice"), "implementation": spec,
        "fixed_cohort_count": twice.PLANNED, "planned_cell_replicates": twice.PLANNED,
        "selected_terminal_indices": list(range(twice.PLANNED)),
        "source_arrays": {"index1": twice.FIRST_ARRAY, "indices2_through199": twice.REMAINING_ARRAY},
        "representative0_report": str(twice.report_path(0, twice.FIRST_ARRAY, twice.REMAINING_ARRAY)),
        "status": "complete_fixed_cell_point_bias" if complete else "provisional_incomplete_fixed_cell",
        "valid_replicates": len(valid), "unknown_replicates": twice.PLANNED - len(valid),
        "unknown_indices": [row["replicate_index"] for row in rows if row["status"] == "unknown"],
        "group_point_bias_complete": complete, "statistics_provisional": not complete,
        "true_rejection_displacement": TRUE_EFFECT, "relative_bias_limit": RELATIVE_BIAS_LIMIT,
        "estimator": "0.5 minus mean(expit(rho_hat)) over all 54 participants and both primary halves, separately EI/ER",
        "bias_definition": "abs(mean across all 200 fixed replicate displacements minus 0.10)/0.10; not mean absolute error",
        "version_statistics": statistics,
        "both_versions_relative_bias_pass": all(statistics[v]["relative_bias_pass"] for v in group.VERSIONS) if complete else None,
        "valid_core_failures_included": sum(not row["necessary_nested_MZ_vs_M0_pass"] for row in valid),
        "cohorts": rows, "interval_coverage_evaluated": False, "group_interval_coverage_qualified": False,
        "group_recovery": "not_evaluated", "family_selection": "not_evaluated", "family_recovery_rate": "not_evaluated",
        "full_recovery_gate": "not_evaluated", "full_recovery_pass": None, "episode_terminal": False,
        "new_simulation": False, "new_model_fits": False, "bootstrap_performed": False,
        "automatic_retry": False, "observed_choices_or_RTs_used": False,
        "interpretation": "Uniform twice-SESOI fixed-MZ point bias only. All valid cohorts enter regardless of their predictive screen. Technical/missing inputs remain unknown; no final bias verdict before 200 valid unique cohorts. Broader nuisance regimes, intervals, family selection and the full gate remain unevaluated.",
        "cost_accounting": "Input process/fit costs are retained as references, not summed or recharged. The coordinator accounts the new summary step once."}


def pool(indices, first_array, remaining_array, spec, *, check=lambda: None, rows=None):
    validate_selection(indices, first_array, remaining_array, spec)
    rows = unknown_rows() if rows is None else rows
    original_versions = None
    for index, row in enumerate(rows):
        check()
        path = twice.report_path(index, first_array, remaining_array)
        row["report"] = str(path)
        try:
            report = json.loads(path.read_text())
        except FileNotFoundError:
            row["reason"] = "selected_terminal_report_missing"
            continue
        check()
        twice.validate_identity(report, index, spec)
        row.update(job_id=str(report["job_id"]), elapsed_seconds=report["elapsed_seconds"], finished_utc=report["finished_utc"])
        if report.get("status") == "technical_recovery_failure":
            row.update(reason="technical_component_failure", **twice.technical_metadata(report))
            continue
        diagnostic, _ = twice.completed_diagnostic(report, spec)
        effects, versions, contacts = group.extract_effect(report, spec)
        if original_versions is None:
            original_versions = versions
        elif versions != original_versions:
            raise ValueError("Participant identities or version assignments changed across the fixed cohort set")
        check()
        row.pop("reason", None)
        row.update(status="point_effect_validated", rejection_displacement=effects,
            fit_support_contact_counts=contacts,
            source_fit_elapsed_seconds=[fit["elapsed_seconds"] for fit in report["fits"]],
            necessary_nested_MZ_vs_M0_pass=diagnostic["qualifies_nested_comparison"])
    check()
    return summarize_rows(rows, spec)


def write_report(output, result):
    temporary = output.with_name(output.name + ".partial")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-indices", type=twice.terminal_indices, required=True)
    parser.add_argument("--first-array-job-id", type=twice.array_id, required=True)
    parser.add_argument("--remaining-array-job-id", type=twice.array_id, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the unchanged original implementation path")
    if not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("The whole-process summary cap must be within 1 through 240 seconds")
    if any(os.environ.get(name) != "1" for name in THREAD_ENV):
        raise ValueError("Run this single-process summary with all six numerical thread limits set to one")
    if args.output.exists() or args.output.with_name(args.output.name + ".partial").exists():
        raise FileExistsError("Preserve earlier artifacts and select a fresh summary output")
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds
    rows, spec, result = unknown_rows(), None, None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Whole-process twice-MZ group-bias deadline exceeded")

    def interrupted(signum, frame):
        raise RuntimeError(f"Group-bias summary interrupted by signal {signum}")

    def publish(value):
        value.update(recorded_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.perf_counter() - started, internal_seconds=args.max_seconds,
            job_id=os.environ.get("SLURM_JOB_ID"))
        write_report(args.output, value)

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    succeeded, failure = False, None
    try:
        check()
        spec = json.loads(IMPLEMENTATION.read_text())
        validate_selection(args.terminal_indices, args.first_array_job_id, args.remaining_array_job_id, spec)
        publish(summarize_rows(rows, spec, allow_final=False))
        check()
        result = pool(args.terminal_indices, args.first_array_job_id, args.remaining_array_job_id, spec,
                      check=check, rows=rows)
        check()
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
                failure, succeeded = f"{type(error).__name__}: {error}", False
        if not succeeded:
            result = summarize_rows(rows, spec, allow_final=False)
            result.update(status="technical_group_bias_summary_failure", error=failure)
            publish(result)
    print(json.dumps({key: result[key] for key in ("status", "valid_replicates", "unknown_replicates",
        "version_statistics", "both_versions_relative_bias_pass", "error") if key in result}, indent=2), flush=True)
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
