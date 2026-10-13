#!/usr/bin/env python3
"""Reconcile the fixed EP17 retained roster, reusing all three pilot event files."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import signal
import socket
import time

import prepare_released_event_manifest as manifest_tool
import reconcile_released_events as core


OUTPUT_PARENT = core.EPISODE / "outputs/runs/released_event_reconciliation"
PILOT_OUTPUT = OUTPUT_PARENT / "pilot001"
INTERNAL_SECONDS = 1080


def validate_manifest(document, roster):
    core.require(document.get("complete") is True and document.get("errors") == [] and
                 document.get("events_commit") == core.EVENT_COMMIT and
                 document.get("release_commit") == core.RELEASE_COMMIT and
                 document.get("fmriprep_commit") == core.FMRIPREP_COMMIT,
                 "unqualified_annex_manifest")
    entries = document.get("entries", [])
    core.require([manifest_tool.run_key(row) for row in entries] ==
                 [manifest_tool.run_key(row) for row in roster], "manifest_roster_mismatch")
    total = 0
    for item, retained in zip(entries, roster):
        for field in retained:
            core.require(item.get(field) == retained[field], "manifest_retained_metadata_mismatch")
        core.require(item.get("source_path") == manifest_tool.event_path(retained), "manifest_source_path_mismatch")
        core.require(isinstance(item.get("directory"), str) and
                     re.fullmatch(r"[A-Za-z0-9]{2}/[A-Za-z0-9]{2}", item["directory"]) and
                     isinstance(item.get("md5"), str) and re.fullmatch(r"[0-9a-f]{32}", item["md5"]) and
                     isinstance(item.get("pointer_blob_sha"), str) and
                     manifest_tool.SHA.fullmatch(item["pointer_blob_sha"]), "invalid_manifest_annex_identity")
        core.require(type(item.get("bytes")) is int and 0 < item["bytes"] <= manifest_tool.EVENT_OBJECT_CAP,
                     "event_object_exceeds_selected_budget")
        total += item["bytes"]
        participant, session, run = manifest_tool.run_key(item)
        reused = participant == "sub-01" and session == 1 and run in (1, 2, 3)
        core.require(item.get("reuse_pilot") is reused, "pilot_reuse_scope_mismatch")
        if reused:
            spec = core.PILOT[run - 1]
            core.require(all(item[field] == spec[field] for field in ("directory", "bytes", "md5")),
                         "pilot_annex_identity_changed")
    core.require(total <= manifest_tool.FULL_EVENT_CAP, "aggregate_event_bytes_exceed_selected_budget")
    core.require(sum(item["reuse_pilot"] for item in entries) == 3, "pilot_reuse_count_mismatch")
    return entries


def load_presentations(roster):
    grouped = {manifest_tool.run_key(row): [] for row in roster}
    with (core.METADATA / "presentation_identity.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = manifest_tool.run_key(row)
            core.require(key in grouped, "presentation_outside_retained_roster")
            grouped[key].append(row)
    for record in roster:
        rows = grouped[manifest_tool.run_key(record)]
        count = record["presentations"]
        start = record["first_glmsingle_trial_index_zero_based"]
        core.require(len(rows) == count and
                     [int(row["trial_in_run_zero_based"]) for row in rows] == list(range(count)) and
                     [int(row["glmsingle_trial_index_zero_based"]) for row in rows] == list(range(start, start + count)) and
                     all(int(row["run_order"]) == record["run_order"] for row in rows),
                     "retained_presentation_order_mismatch")
    return grouped


def validate_pilot_rows(rows, expected):
    core.require(len(rows) == len(expected), "pilot_projection_count_mismatch")
    for projected, previous in zip(rows, expected):
        for field in ("participant", "session", "run", "run_order", "trial_in_run_zero_based",
                      "glmsingle_trial_index_zero_based", "things_category_nr",
                      "provider_category_nr", "provider_exemplar_nr", "onset_TR_after_two_volume_removal"):
            core.require(str(projected[field]) == str(previous[field]), "pilot_projection_identity_mismatch")
        core.require(projected["expected_provider_image_path"] == previous["provider_image_path"] and
                     projected["source_image_basename"] == previous["image_name"] and
                     projected["identity_and_onset_match"] == "True" and
                     projected["image_path_suffix_matches_expected"] == "True",
                     "pilot_projection_does_not_match")


def load_pilot(expected):
    with (PILOT_OUTPUT / "reconciliation_report.json").open(encoding="utf-8") as stream:
        report = json.load(stream)
    core.require(report.get("selected_pilot_reconciliation_complete") is True and
                 report.get("status") == "completed" and report.get("errors") == [] and
                 report.get("events_commit") == core.EVENT_COMMIT and
                 report.get("projected_rows") == 180, "pilot_not_reusable")
    results = {}
    with (PILOT_OUTPUT / "event_metadata.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        core.require(reader.fieldnames == list(core.OUTPUT_FIELDS), "pilot_projection_columns_changed")
        for row in reader:
            key = manifest_tool.run_key(row)
            core.require(key in (("sub-01", 1, 1), ("sub-01", 1, 2), ("sub-01", 1, 3)),
                         "pilot_projection_scope_changed")
            results.setdefault(key, []).append(row)
    core.require(len(results) == 3, "incomplete_pilot_projection")
    run_reports = {row["run"]: row for row in report["runs"]}
    for key, rows in results.items():
        original = run_reports[key[2]]
        core.require(original.get("status") == "matched" and original.get("mismatches") == [] and
                     original.get("annex_content_identity_verified") is True,
                     "pilot_run_not_reusable")
        validate_pilot_rows(rows, expected[key])
    return results, run_reports


def process_run(spec, expected, pilot_rows, pilot_reports, deadline):
    participant, session, run = manifest_tool.run_key(spec)
    key = participant, session, run
    report = {"participant": participant, "session": session, "run": run,
              "annex_bytes": spec["bytes"], "annex_md5": spec["md5"],
              "reused_pilot": spec["reuse_pilot"], "status": "started"}
    start = time.monotonic()
    try:
        if spec["reuse_pilot"]:
            rows = pilot_rows[key]
            mismatches = []
            unknowns = pilot_reports[run]["unknown_metadata"]
            report["annex_content_identity_verified_in_pilot"] = True
            report["new_event_bytes"] = 0
        else:
            payload = core.download(spec, deadline, report)
            rows, mismatches, unknowns = core.reconcile(
                payload, expected[key], run, participant=participant, session=session)
            del payload
            report["new_event_bytes"] = spec["bytes"]
        report.update({"status": "matched" if not mismatches else "mismatch",
                       "projected_rows": len(rows), "mismatches": mismatches,
                       "unknown_metadata": unknowns})
        return rows, report
    except core.Refusal as error:
        report.update({"status": "failed", "error": error.detail})
        return [], report
    finally:
        report["elapsed_seconds"] = time.monotonic() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    core.require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname(),
                 "requires_coordinator_owned_compute")
    manifest_path = args.manifest.resolve(strict=True)
    core.require(manifest_path.name == "annex_manifest.json" and
                 manifest_path.parent.parent == manifest_tool.OUTPUT_PARENT.resolve(),
                 "requires_episode_manifest_attempt")
    out = args.out.resolve()
    core.require(out.parent == OUTPUT_PARENT.resolve() and not out.exists(), "fresh_episode_attempt_required")
    out.mkdir(parents=True)
    start = time.monotonic()
    report = {"analysis_id": "released_event_reconciliation", "stage": "retained_roster_reconciliation",
              "events_commit": core.EVENT_COMMIT, "release_commit": core.RELEASE_COMMIT,
              "fmriprep_commit": core.FMRIPREP_COMMIT, "manifest": str(manifest_path),
              "pilot_reused_from": str(PILOT_OUTPUT), "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "hostname": socket.gethostname(), "internal_seconds": INTERNAL_SECONDS,
              "request_seconds": core.REQUEST_SECONDS, "automatic_retries": 0,
              "maximum_new_event_requests": 830,
              "retained_roster_event_reconciliation_complete": False,
              "all_source_events_reconciled": False,
              "source_roles_complete": False, "eligible_720_concept_universe_verified": False,
              "roles_assigned": False, "folds_created": False, "audit_opened": False,
              "neural_values_read": False, "image_values_read": False, "model_features_read": False,
              "candidate_scoring_run": False, "raw_event_files_written": False,
              "behavioral_columns_selected": False, "behavioral_values_persisted_or_reported": False,
              "mixed_event_body_read_attempted": False, "projected_rows": 0,
              "runs": [], "errors": [], "status": "running"}
    signal.signal(signal.SIGALRM, core.deadline_alarm)
    signal.alarm(INTERNAL_SECONDS)
    try:
        roster = manifest_tool.load_roster()
        with manifest_path.open(encoding="utf-8") as stream:
            entries = validate_manifest(json.load(stream), roster)
        expected = load_presentations(roster)
        pilot_rows, pilot_reports = load_pilot(expected)
        report["selected_retained_runs"] = len(entries)
        report["selected_total_event_bytes"] = sum(item["bytes"] for item in entries)
        report["selected_new_event_bytes"] = sum(item["bytes"] for item in entries if not item["reuse_pilot"])
        with (out / "event_metadata.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=core.OUTPUT_FIELDS)
            writer.writeheader()
            for spec in entries:
                core.require(time.monotonic() < start + INTERNAL_SECONDS, "attempt_deadline")
                rows, run_report = process_run(spec, expected, pilot_rows, pilot_reports, start + INTERNAL_SECONDS)
                report["runs"].append(run_report)
                writer.writerows(rows)
                stream.flush()
                report["projected_rows"] += len(rows)
                core.require(run_report["status"] == "matched", "retained_run_reconciliation_stopped")
        core.require(report["projected_rows"] == sum(row["presentations"] for row in roster),
                     "retained_roster_projection_incomplete")
        report["retained_roster_event_reconciliation_complete"] = True
        report["status"] = "completed"
    except core.Refusal as error:
        report["errors"].append(error.detail)
        report["status"] = "stopped"
    except (OSError, ValueError, KeyError, TypeError):
        report["errors"].append({"code": "retained_metadata_processing_failure"})
        report["status"] = "stopped"
    finally:
        signal.alarm(0)
    report["elapsed_seconds"] = time.monotonic() - start
    report["mixed_event_body_read_attempted"] = any(
        row.get("mixed_event_body_read_attempted", False) for row in report["runs"])
    report["reused_pilot_runs"] = sum(row.get("reused_pilot", False) for row in report["runs"])
    report["newly_matched_runs"] = sum(row["status"] == "matched" and not row["reused_pilot"] for row in report["runs"])
    report["remaining_requirements"] = [
        "Prospective repeat-completeness and taxonomy/balance assignment choices",
        "Qualified common universe, whole-concept roles, folds and role-filtered handoffs",
        "Later authorized neural/geometry/calibration/model-registry requirements",
    ]
    with (out / "reconciliation_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "projected_rows": report["projected_rows"],
                      "matched_new_runs": report["newly_matched_runs"],
                      "reused_pilot_runs": report["reused_pilot_runs"]}))
    return 0 if report["retained_roster_event_reconciliation_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
