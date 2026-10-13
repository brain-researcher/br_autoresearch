#!/usr/bin/env python3
"""Execute an explicit bounded list of MC recovery replicates in one Slurm job.

This does not submit jobs, monitor a scheduler, retry failures, or change the
recorded recovery count. Mathematical curtailment uses only valid necessary-
comparison failures: even success on every other planned replicate cannot
reach the existing 90-percent empirical criterion. Other scientific gates and
fallback branches remain unevaluated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from qualify_assets import require_compute


def implementation_digest(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def summarize_reports(reports, planned_count, expected_digest):
    indexed = {}
    technical = []
    for path, report in reports:
        index = report["replicate_index"]
        if report.get("phase") != "MC_SESOI_fitted_recovery_component":
            raise ValueError("Only the registered MC SESOI necessary comparison can enter this cell")
        if report.get("planned_cell_replicates") != planned_count or not 0 <= index < planned_count:
            raise ValueError("Replicate count or index differs from the recorded cell")
        if implementation_digest(report["implementation"]) != expected_digest:
            raise ValueError("Do not pool different inference implementations")
        if report.get("calibration") != "five_point_decrease_in_four_second_mixture_level_acceptance_at_zero_reference_drift":
            raise ValueError("Do not pool different generating calibrations")
        if not report.get("valid_recovery_component", False):
            technical.append({"replicate_index": index, "report": str(path),
                              "error": report.get("error", "incomplete")})
            continue
        if index in indexed:
            raise ValueError("Do not count a replicate twice or choose between two valid attempts")
        comparison = report["comparison"]
        richer_passed = comparison["necessary_primary_comparison_pass"]
        requirements = comparison.get("requirements", {})
        core_names = ("mean_gain_at_least_0p01", "mean_gain_greater_than_two_participant_SE")
        if not isinstance(richer_passed, bool) or any(not isinstance(requirements.get(name), bool) for name in core_names):
            raise ValueError("A valid comparison requires explicit Boolean core-margin and richer results")
        # Only the two requirements explicitly named by behavioral_recovery's
        # family-selection rule can make the 200-replicate criterion unreachable.
        # Task-version/median checks remain useful recorded diagnostics but do
        # not strengthen this necessary recovery screen after fitting.
        passed = all(requirements[name] for name in core_names)
        if richer_passed and not passed:
            raise ValueError("Richer all-requirements pass contradicts a core-margin failure")
        indexed[index] = {"replicate_index": index, "report": str(path), "passed": passed,
                          "core_family_selection_pass": passed,
                          "counts_as_curtailment_failure": not passed,
                          "richer_primary_comparison_pass": richer_passed}
    failures = sum(not r["passed"] for r in indexed.values())
    successes = len(indexed)-failures
    required_successes = math.ceil(.9*planned_count)
    upper_count = planned_count-failures
    return {"planned_cell_replicates": planned_count,
            "valid_replicates": len(indexed), "valid_successes": successes,
            "valid_failures": failures, "technical_attempts": technical,
            "curtailment_counting_rule": "Only explicit 0.01-nat or two-participant-SE family-selection failures count; richer diagnostic failures alone do not",
            "valid_richer_primary_successes": sum(r["richer_primary_comparison_pass"] for r in indexed.values()),
            "valid_richer_primary_failures": sum(not r["richer_primary_comparison_pass"] for r in indexed.values()),
            "valid_reports": [indexed[i] for i in sorted(indexed)],
            "remaining_unique_replicates": planned_count-len(indexed),
            "required_successes": required_successes,
            "best_possible_completed_success_count": upper_count,
            "best_possible_completed_success_fraction": upper_count/planned_count,
            "necessary_90pct_cell_criterion_unreachable": upper_count < required_successes,
            "full_recovery_gate": "not_evaluated",
            "episode_terminal": False,
            "interpretation": "A necessary-comparison failure cannot be rescued by adding rivals; other rules and fallback prerequisites remain unevaluated."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--replicate-indices", type=int, nargs="+", required=True)
    parser.add_argument("--prior-report", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    spec = json.loads(args.implementation.read_text())
    count = spec["recovery_plan"]["replicates_per_qualified_generating_cell"]
    digest = implementation_digest(spec)
    if len(set(args.replicate_indices)) != len(args.replicate_indices):
        raise ValueError("Repeated indices would not constitute independent replicates")
    if any(not 0 <= i < count for i in args.replicate_indices):
        raise ValueError("Replicate indices exceed the recorded cell")
    reports = [(path, json.loads(path.read_text())) for path in args.prior_report]
    summary = summarize_reports(reports, count, digest)
    already_valid = {r["replicate_index"] for r in summary["valid_reports"]}
    if already_valid & set(args.replicate_indices):
        raise ValueError("A previously valid replicate cannot be rerun for selection")
    if args.output_dir.exists():
        raise FileExistsError("Preserve batch attempts; choose a fresh output directory")
    args.output_dir.mkdir(parents=True)
    record = {"analysis_id": "behavioral_mechanism_kill_test",
              "phase": "bounded_MC_SESOI_recovery_batch", "job_id": os.environ["SLURM_JOB_ID"],
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "implementation_sha256": digest,
              "requested_replicate_indices": args.replicate_indices,
              "launched_replicate_indices": [], "summary": summary,
              "status": "running"}
    output = args.output_dir/"batch_report.json"
    def save():
        output.write_text(json.dumps(record, indent=2)+"\n")
    save()
    try:
        for index in args.replicate_indices:
            if record["summary"]["necessary_90pct_cell_criterion_unreachable"]:
                record["status"] = "mathematically_curtailed_necessary_comparison"
                break
            destination = args.output_dir/f"replicate{index:03d}"
            command = [sys.executable, str(Path(__file__).with_name("run_criterion_recovery.py")),
                       "--events", str(args.events), "--implementation", str(args.implementation),
                       "--replicate-index", str(index), "--output-dir", str(destination)]
            record["launched_replicate_indices"].append(index)
            save()
            completed = subprocess.run(command, check=False)
            path = destination/"criterion_recovery.json"
            if path.exists():
                reports.append((path, json.loads(path.read_text())))
                record["summary"] = summarize_reports(reports, count, digest)
            if completed.returncode != 0:
                record.update(status="stopped_for_technical_failure", returncode=completed.returncode)
                save()
                return completed.returncode or 1
            save()
        else:
            record["status"] = ("mathematically_curtailed_necessary_comparison"
                if record["summary"]["necessary_90pct_cell_criterion_unreachable"]
                else "requested_replicates_completed")
    except Exception as exc:
        record.update(status="batch_technical_failure", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps({"status": record["status"], "summary": record["summary"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
