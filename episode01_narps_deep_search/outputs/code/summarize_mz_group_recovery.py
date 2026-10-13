#!/usr/bin/env python3
"""Extract the adopted physical MZ group effect from explicit terminal reports.

EI and ER each use all54 participants and both primary half-fits. This is
fixed-MZ point-bias recovery, regardless of the replicate's selection result.
No interval, fitting, retry, observed-data access or full gate is implemented.
The coordinator must supply terminal jobs' original reports; this program does
not query the scheduler. A finalized report is necessary but is not evidence of
the scheduler's terminal exit status.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from scipy.special import expit

from qualify_assets import require_compute
from run_single_process_recovery import HALVES, validate_fit
from summarize_single_process_recovery import IMPLEMENTATION, PLANNED, summarize_reports


VERSIONS = ("EI", "ER")
TRUE_EFFECT = .05
RELATIVE_BIAS_LIMIT = .10


def finalized(report):
    """Refuse a live checkpoint even when its scientific flag is already set."""
    stamp = report.get("finished_utc")
    if not isinstance(stamp, str):
        raise ValueError("An explicit terminal report needs finished_utc")
    try:
        parsed = datetime.fromisoformat(stamp)
    except ValueError as error:
        raise ValueError("Invalid finished_utc in terminal report") from error
    if parsed.utcoffset() is None:
        raise ValueError("finished_utc must have an explicit timezone")
    if report.get("status") not in ("necessary_primary_comparison_completed", "technical_recovery_failure"):
        raise ValueError("Only finalized completed or technical original attempts are inputs")


def numeric_array(value):
    """Fit JSON arrays must contain numbers, not bools or coerced text."""
    if isinstance(value, list):
        return all(numeric_array(item) for item in value)
    return type(value) in (int, float) and math.isfinite(value)


def extract_effect(report, implementation):
    """Validate one complete original four-fit report, then read its MZ theta."""
    if (report.get("participants") != 108 or type(report.get("participants")) is not int
            or report.get("runs") != 432 or type(report.get("runs")) is not int
            or report.get("trials") != 27648 or type(report.get("trials")) is not int
            or report.get("observed_choices_or_RTs_used") is not False
            or report.get("old_synthetic_responses_used") is not False
            or report.get("focal_effect_between_participant_variance") != 0.
            or type(report.get("focal_effect_between_participant_variance")) not in (int, float)):
        raise ValueError("Changed original fixed-cohort recovery support or source scope")
    scores = report.get("participant_scores")
    if not isinstance(scores, dict) or len(scores) != 108:
        raise ValueError("The original108-person version mapping is required")
    versions = {}
    for participant, record in scores.items():
        if not isinstance(participant, str) or not participant or not isinstance(record, dict):
            raise ValueError("Invalid participant identity/version record")
        versions[participant] = record.get("version")
    if Counter(versions.values()) != {"EI": 54, "ER": 54}:
        raise ValueError("Exactly54 original participants per task version are required")
    fits = report.get("fits")
    if not isinstance(fits, list) or len(fits) != 4:
        raise ValueError("An original completed report has exactly four fitted models/halves")
    cells = {}
    contacts = []
    for fit in fits:
        if not isinstance(fit, dict):
            raise ValueError("Invalid fitted model record")
        model = fit.get("model")
        training, scoring = fit.get("training_runs"), fit.get("scoring_runs")
        if model not in ("M0", "MZ") or not any(
                training == list(a) and scoring == list(b) for a, b in HALVES):
            raise ValueError("Unexpected model or primary partition in original report")
        key = model, tuple(training)
        if key in cells:
            raise ValueError("Duplicate fitted model/half")
        for field in ("parameters", "global_mean", "version_means", "version_standard_deviations", "objective_history"):
            if not isinstance(fit.get(field), list) or not numeric_array(fit[field]):
                raise ValueError(f"Nonfinite or invalid numeric fit schema: {field}")
        validate_fit(fit, model, training, scoring, implementation, versions)
        if (fit.get("group_interval_coverage_qualified") is not False
                or fit.get("individual_rank_recovery_qualified") is not False):
            raise ValueError("Original point-fit report cannot claim interval or individual qualification")
        cells[key] = fit
        # Boundary contact is retained evidence, not an invented exclusion.
        contacts.append({"model": model, "training_runs": training,
                         "count": len(fit["support_contacts"])})
    expected = {(model, a) for a, _ in HALVES for model in ("M0", "MZ")}
    if set(cells) != expected:
        raise ValueError("Missing original fitted model/half")
    values = {version: [] for version in VERSIONS}
    for training, _ in HALVES:
        fit = cells["MZ", training]
        column = fit["parameter_names"].index("rho")
        for participant, parameters in zip(fit["participants"], fit["parameters"]):
            values[versions[participant]].append(float(expit(parameters[column])))
    effects = {version: .5 - math.fsum(values[version]) / 108 for version in VERSIONS}
    return effects, versions, contacts


def summarize_group_reports(reports, implementation):
    """Pure aggregation of explicit (path, original-report) pairs; no I/O."""
    reports = list(reports)
    seen = set()
    for _, report in reports:
        if not isinstance(report, dict):
            raise ValueError("Original report must be an object")
        finalized(report)
        index = report.get("replicate_index")
        if type(index) is not int or index in seen:
            raise ValueError("Duplicate or invalid replicate index; select one terminal attempt per index")
        seen.add(index)
    # Reuse fixed truth/seed/implementation and scientific-versus-technical
    # validation. Core failures are deliberately retained below.
    selection_summary = summarize_reports(reports, "MZ", implementation)
    rows, original_versions = [], None
    for path, report in sorted(reports, key=lambda item: item[1]["replicate_index"]):
        if not report["valid_recovery_component"]:
            continue
        effects, versions, contacts = extract_effect(report, implementation)
        if original_versions is None:
            original_versions = versions
        elif versions != original_versions:
            raise ValueError("Participant identities or version assignments changed across replicates")
        rows.append({"replicate_index": report["replicate_index"], "report": str(path),
                     "rejection_displacement": effects, "fit_support_contact_counts": contacts,
                     "core_family_selection_pass": report["comparison"]["core_family_selection_pass"]})
    complete = len(rows) == PLANNED
    statistics = {}
    for version in VERSIONS:
        mean = math.fsum(row["rejection_displacement"][version] for row in rows) / len(rows) if rows else None
        signed = mean - TRUE_EFFECT if mean is not None else None
        relative = abs(signed) / TRUE_EFFECT if signed is not None else None
        statistics[version] = {"mean_rejection_displacement": mean, "signed_bias": signed,
                               "absolute_relative_bias": relative,
                               "relative_bias_pass": relative <= RELATIVE_BIAS_LIMIT if complete else None}
    return {"analysis_id": "behavioral_mechanism_kill_test", "phase": "MZ_SESOI_group_point_bias_summary",
            "generating_model": "MZ", "planned_cell_replicates": PLANNED,
            "valid_replicates": len(rows), "remaining_unique_valid_replicates": PLANNED - len(rows),
            "status": "complete_fixed_cell_point_bias" if complete else "provisional_incomplete_fixed_cell",
            "statistics_provisional": not complete, "group_point_bias_complete": complete,
            "true_rejection_displacement": TRUE_EFFECT, "relative_bias_limit": RELATIVE_BIAS_LIMIT,
            "estimator": "0.5 minus mean(expit(rho_hat)) over all54 participants and both primary halves, separately EI/ER",
            "bias_definition": "abs(mean across all200 fixed replicates of displacement minus0.05)/0.05; not mean absolute error",
            "version_statistics": statistics,
            "both_versions_relative_bias_pass": all(statistics[v]["relative_bias_pass"] for v in VERSIONS) if complete else None,
            "valid_reports": rows, "technical_attempts": selection_summary["technical_attempts"],
            "core_selection_failures_included": sum(not row["core_family_selection_pass"] for row in rows),
            "interval_coverage_evaluated": False, "group_interval_coverage_qualified": False,
            "full_recovery_gate": "not_evaluated", "episode_terminal": False,
            "interpretation": "Fixed-MZ point recovery only. Technical attempts are unresolved. No final bias verdict before200 unique valid replicates; no interval, post-selection or full-gate claim."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the unchanged saved behavioral implementation")
    if args.output.exists():
        raise FileExistsError("Preserve previous summaries; select a fresh output JSON")
    implementation = json.loads(args.implementation.read_text())
    result = summarize_group_reports([(path, json.loads(path.read_text())) for path in args.report], implementation)
    result.update(recorded_utc=datetime.now(timezone.utc).isoformat(),
                  implementation=str(args.implementation), input_reports=[str(path) for path in args.report])
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(payload)
    print(json.dumps({key: result[key] for key in ("status", "valid_replicates", "statistics_provisional",
        "version_statistics", "both_versions_relative_bias_pass", "group_interval_coverage_qualified")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
