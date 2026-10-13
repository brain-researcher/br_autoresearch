#!/usr/bin/env python3
"""Frozen summaries of complete observed-data fits; no selection or refitting."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import statistics

EPISODE = Path(__file__).resolve().parents[2]
RUNS = EPISODE / "outputs/runs/exploratory_observed_behavior"
SOURCE = EPISODE / "outputs/runs/public_asset_qualification/20261003-attempt01/original_events.csv.gz"
MODELS = ("M0", "MV", "MZ", "MC", "MVZ", "MVC")
HALVES = (((1, 4), (2, 3)), ((2, 3), (1, 4)))


def moments(values, versions):
    if len(values) != 108 or not all(math.isfinite(x) for x in values):
        raise ValueError("Only complete finite 108-person summaries are defined")
    return {
        "participants": 108,
        "mean_nats_per_trial": statistics.fmean(values),
        "participant_SE": statistics.stdev(values) / math.sqrt(108),
        "participant_median": statistics.median(values),
        "version_means": {v: statistics.fmean(x for x, group in zip(values, versions) if group == v)
                          for v in ("EI", "ER")},
    }


def summarize(paths, implementation):
    records, reference = {}, None
    for path in paths:
        path = path.resolve()
        if RUNS not in path.parents or path.name != "observed_behavior_fit.json":
            raise ValueError("Use only exact observed-fit reports in the declared run tree")
        report = json.loads(path.read_text())
        model, direction = report.get("model"), report.get("direction")
        if model not in MODELS or type(direction) is not int or direction not in (0, 1):
            raise ValueError("Unregistered model or direction")
        key = model, direction
        if key in records:
            raise ValueError("Duplicate attempt for a frozen model/direction")
        complete = report.get("status") == "direction_completed"
        record = {"path": str(path), "status": report.get("status"), "complete": complete}
        records[key] = record
        if not complete:
            record["reason"] = report.get("error", "missing_or_unfinished_direction")
            continue
        if not (report.get("source_validated") is True and report.get("direction_complete") is True
                and report.get("scores_available") is True and report.get("workers_cleaned_up") is True
                and report.get("source") == str(SOURCE) and report.get("implementation") == implementation
                and report.get("training_runs") == list(HALVES[direction][0])
                and report.get("scoring_runs") == list(HALVES[direction][1])
                and report.get("scored_participants") == 108 and report.get("scored_trials") == 13824):
            raise ValueError("Completed direction violates frozen support or implementation")
        scores = report["participant_scores"]
        people = sorted(scores)
        versions = [scores[p]["version"] for p in people]
        if (len(people) != 108 or report.get("participants") != people
                or versions.count("EI") != 54 or versions.count("ER") != 54):
            raise ValueError("Incomplete or changed participant/version support")
        identity = (people, versions)
        fit = report["fit"]
        if (fit.get("converged") is not True or fit.get("subject_optimizer_failures") != []
                or fit.get("model") != model or fit.get("rival") != "fixed"
                or fit.get("participants") != people
                or fit.get("training_runs") != list(HALVES[direction][0])
                or fit.get("scoring_runs") != list(HALVES[direction][1])
                or not isinstance(fit.get("support_contacts"), list)
                or not isinstance(fit.get("objective_history"), list)
                or not 2 <= len(fit["objective_history"]) <= 500):
            raise ValueError("Missing or inconsistent completed-fit convergence metadata")
        if reference is None:
            reference = identity
        elif reference != identity:
            raise ValueError("Different participant/version identities across fits")
        values = []
        for person in people:
            row = scores[person]
            value, total = row["contribution_sum_over_256"], row["log_sum"]
            if (row["heldout_trials"] != 128 or not math.isfinite(value) or not math.isfinite(total)
                    or not math.isclose(value * 256, total, rel_tol=1e-12, abs_tol=1e-9)):
                raise ValueError("Invalid held-out half contribution")
            values.append(value)
        record["values"] = values
        record["fit_support_contacts"] = fit["support_contacts"]
        record["converged"] = fit["converged"]
        record["sweeps"] = len(fit["objective_history"])
        record["subject_optimizer_failures"] = fit["subject_optimizer_failures"]
    result = {
        "analysis_id": "exploratory_observed_behavior",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "Exploratory plug-in within-session prediction; no mechanism winner, recovery verdict or confirmation",
        "logistic_benchmark": "registered_but_not_implemented_in_this_report",
        "denominator": "256 trials per participant after summing both primary directions",
        "uncertainty": "Participant standard errors are descriptive; no cluster interval or gate inference",
        "models": {}, "paired_differences_from_M0": {},
    }
    full = {}
    for model in MODELS:
        halves = [records.get((model, direction)) for direction in (0, 1)]
        sources = [{k: v for k, v in r.items() if k != "values"} if r else
                   {"status": "not_submitted_or_report_absent", "complete": False} for r in halves]
        result["models"][model] = {"directions": sources, "available": False}
        if all(r and r["complete"] for r in halves):
            values = [a + b for a, b in zip(halves[0]["values"], halves[1]["values"])]
            full[model] = values
            result["models"][model].update(available=True, summary=moments(values, reference[1]),
                participant_scores={p: {"version": v, "nats_per_trial": x}
                                    for p, v, x in zip(reference[0], reference[1], values)})
    for model in MODELS[1:]:
        contrast = model + "-M0"
        result["paired_differences_from_M0"][contrast] = {"available": False}
        if model in full and "M0" in full:
            values = [a - b for a, b in zip(full[model], full["M0"])]
            result["paired_differences_from_M0"][contrast] = {
                "available": True, "summary": moments(values, reference[1]),
                "participant_differences": dict(zip(reference[0], values)),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-report", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Scientific aggregation requires an actual Slurm compute shell")
    if len(args.fit_report) > 12:
        raise ValueError("At most twelve frozen fit reports")
    output = args.output.resolve()
    if RUNS not in output.parents or output.exists():
        raise ValueError("Use a fresh output inside the observed-data run tree")
    implementation = json.loads((EPISODE / "outputs/code/behavioral_implementation.json").read_text())
    report = summarize(args.fit_report, implementation)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "complete_models": [m for m in MODELS if report["models"][m]["available"]]}))


if __name__ == "__main__":
    main()
