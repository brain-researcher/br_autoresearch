#!/usr/bin/env python3
"""Saved MZ0 influence of the qualified16 version-specific offer sequences."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from itertools import combinations
import json
import math
import os
from pathlib import Path
import signal
import time

from qualify_assets import require_compute

EPISODE = Path(__file__).resolve().parents[2]
RUNS = EPISODE / "outputs/runs"
PRIMARY = RUNS / "behavioral_mechanism_kill_test/20261005-MZ-SESOI-replicate000-attempt001/single_process_recovery.json"
SCHEDULE = RUNS / "public_asset_qualification/20261003-schedule01/schedule_support.json"
MEMBERSHIP = SCHEDULE.with_name("offer_sequence_membership.tsv")
PHASE = "MZ0_saved_prediction_offer_sequence_influence"
MAX_SECONDS = 60
VERSIONS = ("EI", "ER")
GROUP_IDS = tuple(f"{version}-offers-{index:02d}" for version in VERSIONS for index in range(1, 9))


def read_inputs():
    primary, schedule = (json.loads(path.read_text()) for path in (PRIMARY, SCHEDULE))
    with MEMBERSHIP.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["participant", "task_version", "offer_sequence_id"]:
            raise ValueError("Use the original three-column membership table")
        members = list(reader)
    return primary, schedule, members


def validate_inputs(primary, schedule, members):
    expected = {"phase": "MZ_SESOI_fitted_recovery_component", "generating_model": "MZ",
        "replicate_index": 0, "seed": 910031, "participants": 108,
        "status": "necessary_primary_comparison_completed"}
    if any(primary.get(key) != value for key, value in expected.items()):
        raise ValueError("Require the completed original108-person synthetic MZ0 comparison")
    expected_schedule = {"analysis_id": "public_asset_qualification", "job_id": "46424502",
        "source_event_table": "outputs/runs/public_asset_qualification/20261003-attempt01/original_events.csv.gz",
        "choice_RT_neural_values_used": False, "grouping": "exact_whole_participant_ordered_run_trial_gain_loss",
        "offer_only_template_counts": {"EI": 8, "ER": 8}, "expected_nominal_templates_per_version": 8,
        "matches_documented_eight_offer_sequences": True}
    if any(schedule.get(key) != value for key, value in expected_schedule.items()):
        raise ValueError("Use the completed design-only8+8 offer-sequence qualification")
    scores = primary["participant_scores"]
    if not isinstance(scores, dict) or len(scores) != 108:
        raise ValueError("Retain every original participant score")
    gains, versions = {}, {}
    for person, row in scores.items():
        if (not isinstance(person, str) or not person or set(row) != {"version", "M0", "MZ"}
                or row["version"] not in VERSIONS or any(type(row[key]) not in (int, float)
                or not math.isfinite(row[key]) for key in ("M0", "MZ"))):
            raise ValueError("Require the finite saved M0/MZ panel and task versions")
        gains[person], versions[person] = row["MZ"] - row["M0"], row["version"]
        if not math.isfinite(gains[person]):
            raise ValueError("The saved-score difference must be finite")
    if Counter(versions.values()) != {"EI": 54, "ER": 54}:
        raise ValueError("Keep all54 EI and54 ER participants")
    groups, qualified = {}, {}
    for row in schedule["groups"]:
        group = row["offer_sequence_id"]
        if group not in GROUP_IDS or group in groups or not row["participants"]:
            raise ValueError("Require exactly the16 distinct qualified offer IDs")
        groups[group] = tuple(row["participants"])
        for person in groups[group]:
            if person in qualified or person not in scores or versions[person] != group[:2]:
                raise ValueError("Qualified membership must join participants and versions one-to-one")
            qualified[person] = (group[:2], group)
    joined = {}
    for row in members:
        if set(row) != {"participant", "task_version", "offer_sequence_id"}:
            raise ValueError("Membership rows must keep the original schema")
        person = row["participant"]
        if person in joined:
            raise ValueError("Duplicate participant in membership table")
        joined[person] = (row["task_version"], row["offer_sequence_id"])
    if set(groups) != set(GROUP_IDS) or set(qualified) != set(scores) or joined != qualified:
        raise ValueError("The JSON groups, TSV membership and all108 scores must agree exactly")
    return gains, versions, groups


def panel_summary(people, gains, versions):
    people = sorted(people)
    if not people:
        raise ValueError("Cannot summarize an empty retained panel")
    counts = Counter(versions[person] for person in people)
    means = {version: math.fsum(gains[person] for person in people if versions[person] == version)/counts[version]
             for version in VERSIONS if counts[version]}
    return {"participants": len(people), "participants_by_version": dict(counts),
        "mean_gain_nats_per_trial": math.fsum(gains[person] for person in people)/len(people),
        "version_means": means, "version_mean_positive": {version: value > 0 for version, value in means.items()}}


def fill_report(report, primary, schedule, members, check, publish):
    gains, versions, groups = validate_inputs(primary, schedule, members)
    report["source_panel_validated"] = True
    report["full_panel_reference"] = panel_summary(gains, gains, versions)
    check()
    publish()
    for group in GROUP_IDS:
        check()
        report["template_groups"].append({"offer_sequence_id": group, "task_version": group[:2],
            "participant_ids": sorted(groups[group]), **panel_summary(groups[group], gains, versions)})
        report["progress"]["template_groups"] = len(report["template_groups"])
    publish()
    for count, target in ((1, "single_omissions"), (2, "pair_omissions")):
        for omitted in combinations(GROUP_IDS, count):
            check()
            removed = {person for group in omitted for person in groups[group]}
            summary = panel_summary(set(gains) - removed, gains, versions)
            report[target].append({"omitted_offer_sequence_ids": list(omitted),
                "omitted_participants": len(removed),
                "pair_type": None if count == 1 else (
                    "within_version" if omitted[0][:2] == omitted[1][:2] else "cross_version"),
                **summary, "mean_change_from_full_panel": summary["mean_gain_nats_per_trial"]
                    - report["full_panel_reference"]["mean_gain_nats_per_trial"]})
            report["progress"][target] = len(report[target])
            if len(report[target]) % 16 == 0:
                publish()
        publish()
    check()


def new_report():
    return {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "primary_report": str(PRIMARY), "schedule_report": str(SCHEDULE), "membership_table": str(MEMBERSHIP),
        "status": "running", "diagnostic_complete": False, "source_panel_validated": False,
        "fixed_scope": {"template_groups": 16, "single_omissions": 16, "pair_omissions": 120,
                        "within_version_pairs": 56, "cross_version_pairs": 64},
        "progress": {"template_groups": 0, "single_omissions": 0, "pair_omissions": 0},
        "full_panel_reference": None, "template_groups": [], "single_omissions": [], "pair_omissions": [],
        "weighting": "Equal participant weight within each retained panel; divide by its actual retained count. Report EI/ER separately; never average template means.",
        "interpretation": "Fixed-prediction influence on one saved synthetic MZ0 cohort. The selected scope includes all120 unordered pairs of16 version-specific IDs; progress records actual completed subsets. Cross-version pairs do not match ordinal template labels. Measured onsets and all predictions remain unchanged. These are overlapping sensitivity subsets, not independent replications or refitted template generalization.",
        "cluster_intervals": "not_evaluated", "template_generality": "not_evaluated",
        "full_gate": "not_evaluated", "full_recovery_pass": None, "automatic_retry": False,
        "source_events_read": False, "saved_fit_parameters_used": False, "models_refitted": False, "predictions_modified": False,
        "observed_response_or_neural_values_read": False}


def write_report(path, report):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(output):
    require_compute()
    output = output.resolve()
    if (output.exists() or output.with_name(output.name + ".partial").exists()
            or EPISODE / "outputs" not in output.parents
            or any(path.parent == output.parent or path.parent in output.parents for path in (PRIMARY, SCHEDULE))):
        raise FileExistsError("Use a fresh episode output outside both preserved input attempts")
    began, deadline = time.perf_counter(), time.monotonic() + MAX_SECONDS
    report = new_report()
    report.update(job_id=os.environ.get("SLURM_JOB_ID"), internal_seconds=MAX_SECONDS,
        started_utc=datetime.now(timezone.utc).isoformat(),
        cost_accounting="Only this saved-report join and finite descriptive aggregation; no historical fitting cost is recharged.")

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Whole-process60-second deadline exceeded")

    def publish():
        report["elapsed_seconds"] = time.perf_counter() - began
        write_report(output, report)

    def interrupted(signum, frame):
        raise RuntimeError(f"Offer-sequence influence interrupted by signal {signum}")

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, interrupted)
    signal.alarm(MAX_SECONDS)
    complete = False
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        publish()
        check()
        primary, schedule, members = read_inputs()
        check()
        fill_report(report, primary, schedule, members, check, publish)
        check()
        complete = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        signal.alarm(0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if time.monotonic() >= deadline:
            complete = False
            report["error"] = "RuntimeError: Whole-process60-second deadline exceeded before final publication"
        report.update(status="offer_sequence_influence_completed" if complete else "technical_offer_sequence_influence_failure",
                      diagnostic_complete=complete, finished_utc=datetime.now(timezone.utc).isoformat())
        publish()
        if complete and time.monotonic() >= deadline:
            report.update(status="technical_offer_sequence_influence_failure", diagnostic_complete=False,
                error="RuntimeError: Whole-process60-second deadline exceeded during final publication",
                finished_utc=datetime.now(timezone.utc).isoformat())
            publish()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    report = run(parser.parse_args(argv).output)
    print(json.dumps({key: report[key] for key in ("status", "diagnostic_complete", "progress", "elapsed_seconds", "error") if key in report}, indent=2))
    return 0 if report["diagnostic_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
