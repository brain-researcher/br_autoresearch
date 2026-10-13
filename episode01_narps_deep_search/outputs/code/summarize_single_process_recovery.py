#!/usr/bin/env python3
"""Pool explicitly supplied MV or MZ recovery reports; never fit or launch.

The fixed cell contains200 indices and requires180 successes. Only the two
registered core selection conditions can count a scientific failure. Technical
attempts and richer-diagnostic failures alone cannot curtail a recovery cell.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from qualify_assets import require_compute


IMPLEMENTATION = Path(__file__).with_name("behavioral_implementation.json")
BASE_SEEDS = {"MV": 810031, "MZ": 910031}
CORE = ("mean_gain_at_least_0p01", "mean_gain_greater_than_two_participant_SE")
DIAGNOSTICS = ("positive_in_both_task_versions", "positive_participant_median",
               "participant_median_95CI_lower_positive")
PLANNED = 200
REQUIRED = 180


def boolean(record, name):
    value = record.get(name)
    if type(value) is not bool:
        raise ValueError(f"Explicit Boolean required: {name}")
    return value


def summarize_reports(reports, model, implementation):
    """Pure aggregate arithmetic over (explicit_path, report_dict) pairs.

    Multiple technical attempts may be retained for an index, including one
    later completed valid attempt. Two valid attempts for one index are refused;
    the caller cannot choose the favorable result. No missing index is failed.
    """
    if model not in BASE_SEEDS:
        raise ValueError("Select exactly one registered MV or MZ cell")
    if implementation["recovery_plan"]["replicates_per_qualified_generating_cell"] != PLANNED:
        raise ValueError("Only the unchanged200-replicate recovery cell is defined")
    expected_truth = {"eta": math.log(1.15) if model == "MV" else 0.,
                      "rho": math.log(.45/.55) if model == "MZ" else 0., "criterion": 0.}
    indexed, technical = {}, []
    for path, report in reports:
        index = report.get("replicate_index")
        if type(index) is not int or not 0 <= index < PLANNED:
            raise ValueError("Replicate index must be an integer0 through199")
        if (report.get("analysis_id") != "behavioral_mechanism_kill_test"
                or report.get("phase") != f"{model}_SESOI_fitted_recovery_component"
                or report.get("generating_model") != model
                or type(report.get("planned_cell_replicates")) is not int
                or report["planned_cell_replicates"] != PLANNED
                or type(report.get("seed")) is not int
                or report["seed"] != BASE_SEEDS[model] + 10007 * index
                or isinstance(report.get("SESOI_level"), bool) or report.get("SESOI_level") != 1.):
            raise ValueError("Do not pool different generating models, phases, counts, seeds or levels")
        if report.get("implementation") != implementation:
            raise ValueError("Do not pool different inference implementations")
        truth = report.get("focal_generating_truth")
        if (not isinstance(truth, dict) or set(truth) != set(expected_truth)
                or any(type(truth[key]) not in (int, float) or not math.isfinite(truth[key])
                       or truth[key] != value for key, value in expected_truth.items())):
            raise ValueError("Do not pool different focal generating truths")
        if (report.get("full_behavioral_gate") != "not_evaluated"
                or report.get("full_recovery_pass") is not None):
            raise ValueError("These reports cannot supply a full recovery gate verdict")
        valid = boolean(report, "valid_recovery_component")
        counted_failure = boolean(report, "counts_as_curtailment_failure")
        if not valid:
            if counted_failure or report.get("status") == "necessary_primary_comparison_completed":
                raise ValueError("Technical/incomplete attempt contradicts its completion or failure flag")
            technical.append({"replicate_index": index, "report": str(path),
                "status": report.get("status", "incomplete"), "error": report.get("error", "incomplete"),
                "counts_as_curtailment_failure": False})
            continue
        if report.get("status") != "necessary_primary_comparison_completed":
            raise ValueError("Valid scientific comparison lacks completed status")
        if index in indexed:
            raise ValueError("Multiple valid attempts for the same replicate are prohibited")
        comparison = report.get("comparison", {})
        requirements = comparison.get("requirements", {})
        flags = {name: boolean(requirements, name) for name in CORE + DIAGNOSTICS}
        core_passed = all(flags[name] for name in CORE)
        richer_passed = all(flags.values())
        if (comparison.get("focal_model") != model
                or boolean(comparison, "core_family_selection_pass") != core_passed
                or boolean(comparison, "counts_as_curtailment_failure") != (not core_passed)
                or counted_failure != (not core_passed)
                or boolean(comparison, "necessary_primary_comparison_pass") != richer_passed
                or boolean(comparison, "richer_primary_diagnostic_pass") != richer_passed
                or comparison.get("full_recovery_pass") is not None):
            raise ValueError("Core or richer diagnostic Boolean results are inconsistent")
        indexed[index] = {"replicate_index": index, "report": str(path),
            "core_family_selection_pass": core_passed, "counts_as_curtailment_failure": not core_passed,
            "richer_primary_comparison_pass": richer_passed, "requirements": flags}
    failures = sum(row["counts_as_curtailment_failure"] for row in indexed.values())
    successes = len(indexed) - failures
    best_possible = PLANNED - failures
    return {"analysis_id": "behavioral_mechanism_kill_test", "phase": f"{model}_SESOI_fixed_cell_summary",
        "generating_model": model, "planned_cell_replicates": PLANNED, "required_successes": REQUIRED,
        "valid_replicates": len(indexed), "valid_successes": successes, "valid_failures": failures,
        "remaining_unique_replicates": PLANNED - len(indexed),
        "best_possible_completed_success_count": best_possible,
        "best_possible_completed_success_fraction": best_possible / PLANNED,
        "necessary_90pct_cell_criterion_unreachable": best_possible < REQUIRED,
        "valid_richer_primary_successes": sum(row["richer_primary_comparison_pass"] for row in indexed.values()),
        "valid_richer_primary_failures": sum(not row["richer_primary_comparison_pass"] for row in indexed.values()),
        "valid_reports": [indexed[index] for index in sorted(indexed)], "technical_attempts": technical,
        "curtailment_counting_rule": "Only explicit0.01-nat or two-participant-SE core failures count; richer diagnostic failures alone do not",
        "full_recovery_gate": "not_evaluated", "episode_terminal": False,
        "interpretation": "At21 valid core failures, even success at every other fixed index cannot reach180/200. This is a necessary cell bound, not a full gate pass or episode verdict; no new fits are authorized."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("MV", "MZ"), required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the unchanged saved behavioral implementation")
    if args.output.exists():
        raise FileExistsError("Preserve earlier summaries; select a fresh output JSON")
    implementation = json.loads(args.implementation.read_text())
    reports = [(path, json.loads(path.read_text())) for path in args.report]
    result = summarize_reports(reports, args.model, implementation)
    result.update(recorded_utc=datetime.now(timezone.utc).isoformat(),
                  implementation=str(args.implementation), input_reports=[str(path) for path in args.report])
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(payload)
    print(json.dumps({key: result[key] for key in ("generating_model", "valid_replicates", "valid_successes",
        "valid_failures", "remaining_unique_replicates", "best_possible_completed_success_count",
        "required_successes", "necessary_90pct_cell_criterion_unreachable", "full_recovery_gate", "episode_terminal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
