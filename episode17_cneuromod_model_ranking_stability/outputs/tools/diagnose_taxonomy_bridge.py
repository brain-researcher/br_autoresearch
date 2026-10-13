#!/usr/bin/env python3
"""Diagnose the pinned taxonomy row-order discrepancy using source IDs only.

No eligibility, normalization, membership projection, role assignment or
source repair. Run only in the coordinator-owned Slurm allocation.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path, PurePosixPath
import signal
import socket
import time

import qualify_source_roles as stage
import reconcile_released_events as core


INTERNAL_SECONDS = 120
OUTPUT_CAP = 256 * 1024
OUTPUT_NAME = "taxonomy_bridge_diagnostic002.json"


def source_anchors(payload):
    _, rows = stage.table_rows(payload, stage.PRESENTATION_FIELDS)
    core.require(len(rows) == stage.ROW_COUNT, "fixed_source_row_count_changed")
    anchors, names = {}, {}
    participants = set()
    for row in rows:
        core.require(row["condition"] == "exp", "retained_source_condition_scope_changed")
        core.require(row["participant"] in stage.SUBJECTS, "source_participant_changed")
        participants.add(row["participant"])
        path = PurePosixPath(row["provider_image_path"])
        name = row["concept_name"]
        core.require(len(path.parts) == 3 and path.parts[0] == "images_fmri" and
                     ".." not in path.parts and path.name == row["image_name"] and
                     bool(name) and name == path.parent.name,
                     "source_concept_folder_identity_mismatch")
        cid = core.integer(row["things_category_nr"], "things_category_nr")
        core.require(1 <= cid <= 1854, "source_concept_index_out_of_range")
        core.require(anchors.setdefault(cid, name) == name and names.setdefault(name, cid) == cid,
                     "conflicting_source_CID_name_anchor")
    core.require(participants == set(stage.SUBJECTS) and len(anchors) == len(names) == 720,
                 "source_anchor_scope_changed")
    return anchors


def unique_ids(payload):
    """Immediately discard every field except uniqueID, including all ratings."""
    try:
        reader = csv.reader(io.StringIO(payload.decode("utf-8-sig"), newline=""),
                            delimiter="\t", strict=True)
        header = next(reader)
        core.require(len(header) == len(set(header)) and "uniqueID" in header, "invalid_UID_header")
        position = header.index("uniqueID")
        result = []
        for index, values in enumerate(reader):
            core.require(len(values) == len(header), "UID_table_row_width_mismatch", row=index)
            result.append(values[position])
    except (UnicodeError, csv.Error, StopIteration):
        raise core.Refusal("invalid_UID_table") from None
    core.require(len(result) == 1854 and all(result) and len(set(result)) == 1854,
                 "UID_table_not_1854_nonempty_unique_keys")
    return result


def compare(anchors, categories, concepts):
    category_set, concept_set = set(categories), set(concepts)
    core.require(category_set == concept_set, "pinned_UID_sets_differ")
    category_mismatches, concept_mismatches = [], []
    exact_category = exact_concept = 0
    for cid, name in sorted(anchors.items()):
        exact_category += name in category_set
        exact_concept += name in concept_set
        if categories[cid - 1] != name:
            category_mismatches.append([cid, name, categories[cid - 1]])
        if concepts[cid - 1] != name:
            concept_mismatches.append([cid, name, concepts[cid - 1]])
    examples = ([{"table": "category53", "CID_sourceName_numericRowUID": item}
                 for item in category_mismatches] +
                [{"table": "concepts", "CID_sourceName_numericRowUID": item}
                 for item in concept_mismatches])[:10]
    return {
        "source_anchor_count": len(anchors),
        "source_anchor_basis": "Exact concept_name equals images_fmri parent directory; one-to-one things_category_nr/name pairs",
        "exact_uniqueID_coverage": {"category53": exact_category, "concepts": exact_concept},
        "numeric_CID_minus_one_mismatch_counts": {"category53": len(category_mismatches),
                                                 "concepts": len(concept_mismatches)},
        "mismatch_examples_maximum": 10, "mismatch_examples": examples,
        "table_rowwise_UID_mismatch_count": sum(a != b for a, b in zip(categories, concepts)),
        "table_uniqueID_sets_equal": True,
        "verified_source_anchors": [[cid, name] for cid, name in sorted(anchors.items())],
        "verified_source_anchor_columns": ["things_category_nr", "source_concept_name"],
        "verified_uniqueID_tables_in_source_row_order": {"category53": categories, "concepts": concepts},
    }


def encoded_report(report):
    # JSON escaping preserves exact punctuation/whitespace without terminal controls.
    return (json.dumps(report, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def diagnose(out):
    started = time.monotonic()
    report = {
        "analysis_id": "source_roles", "stage": "taxonomy_ID_bridge_diagnostic",
        "status": "running", "diagnostic_complete": False,
        "source_metadata_attempt": str(core.METADATA),
        "stimulus_commit": stage.STIMULUS_COMMIT,
        "annotation_code_commit": stage.ANNOTATION_CODE_COMMIT,
        "internal_seconds": INTERNAL_SECONDS, "external_seconds_required": 180,
        "existing_CSV_byte_cap": stage.FILE_CAP, "output_byte_cap": OUTPUT_CAP,
        "request_seconds": 20, "automatic_retries": 0,
        "maximum_taxonomy_requests": 2, "selected_taxonomy_bytes": 724963,
        "maximum_body_bytes_including_oversize_sentinels": 724965,
        "eligibility_evaluated": False, "eligibility_complete": False,
        "source_roles_complete": False, "roles_assigned": False, "normalization_applied": False,
        "crosswalk_adopted": False, "neural_values_read": False, "image_values_read": False,
        "model_features_read": False, "participant_task_response_payloads_read": False,
        "taxonomy_memberships_selected": False, "normative_ratings_selected": False,
        "public_table_projection": "Only uniqueID selected; other cells transiently parsed then discarded",
        "taxonomy_license": "CC BY; Hebart et al. (2019), Stoinski et al. (2024)",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(),
        "taxonomy_transfers": [], "errors": [],
    }
    signal.signal(signal.SIGALRM, core.deadline_alarm)
    signal.alarm(INTERNAL_SECONDS)
    try:
        anchors = source_anchors(stage.bounded_read(core.METADATA / "presentation_identity.csv"))
        uid_tables = []
        for spec in stage.TAXONOMY:
            stage.verify_pinned_pointer(spec)
            receipt = {**spec, "request_attempted": False, "body_read_attempted": False,
                       "received_bytes": 0, "annex_content_identity_verified": False}
            report["taxonomy_transfers"].append(receipt)
            payload = stage.download_taxonomy(spec, started + INTERNAL_SECONDS, receipt)
            uid_tables.append(unique_ids(payload))
            del payload
        result = compare(anchors, *uid_tables)
        prospective = {**report, **result, "status": "completed", "diagnostic_complete": True,
                       "elapsed_seconds": time.monotonic() - started}
        core.require(len(encoded_report(prospective)) + 128 <= OUTPUT_CAP, "diagnostic_output_exceeds_cap")
        core.require(time.monotonic() < started + INTERNAL_SECONDS, "attempt_deadline")
        report.update(result)
        report.update({"status": "completed", "diagnostic_complete": True})
    except core.Refusal as error:
        report["errors"].append(error.detail)
        report.update({"status": "stopped", "diagnostic_complete": False})
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        report["errors"].append({"code": "bounded_taxonomy_bridge_diagnostic_failed"})
        report.update({"status": "stopped", "diagnostic_complete": False})
    finally:
        signal.alarm(0)
    report["elapsed_seconds"] = time.monotonic() - started
    payload = encoded_report(report)
    core.require(len(payload) <= OUTPUT_CAP, "diagnostic_output_exceeds_cap")
    with out.open("xb") as stream:
        stream.write(payload)
    print(json.dumps({"status": report["status"], "source_anchor_count": report.get("source_anchor_count"),
                      "exact_uniqueID_coverage": report.get("exact_uniqueID_coverage"),
                      "numeric_CID_minus_one_mismatch_counts": report.get("numeric_CID_minus_one_mismatch_counts"),
                      "diagnostic_complete": report["diagnostic_complete"]}))
    return 0 if report["status"] == "completed" and report["diagnostic_complete"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    core.require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname(),
                 "requires_coordinator_owned_compute")
    out = args.out.resolve()
    core.require(out.parent == stage.OUTPUT_PARENT.resolve() and out.name == OUTPUT_NAME and not out.exists(),
                 "fresh_selected_diagnostic_path_required")
    core.require(out.parent.is_dir(), "existing_source_qualification_parent_required")
    return diagnose(out)


if __name__ == "__main__":
    raise SystemExit(main())
