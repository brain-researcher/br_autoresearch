#!/usr/bin/env python3
"""Bounded EP17 source eligibility and taxonomy metadata; no roles or outcomes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import signal
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

import reconcile_released_events as core


EVENTS = core.EPISODE / "outputs/runs/released_event_reconciliation/retained001"
OUTPUT_PARENT = core.EPISODE / "outputs/runs/source_qualification"
SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-06")
RUN_COUNTS = dict(zip(SUBJECTS, (213, 213, 213, 194)))
ROW_COUNT = 49980
INTERNAL_SECONDS = 120
FILE_CAP = 32 * 1024 * 1024
STIMULUS_COMMIT = "f8f5162f79fb2e9fb0d8cbb8bac610db11a54180"
ANNOTATION_CODE_COMMIT = "b84bf6d5c18e53e78c6278bfcb5d4e0e6afff214"
CONP = ("https://sftp.conp.ca/users/cneuromod/ria-conp/550/"
        "4289b-0ac5-4f7c-a6cd-86af1f5c2135/annex/objects/")
TAXONOMY = (
    {"name": "category53_wideFormat.tsv", "directory": "fK/Wk", "bytes": 227496,
     "md5": "f24be61211cc5f6b600cf0d13dab8bba",
     "pointer_blob": "209d6b464a6bf0df4a471a5505e1d3a7325432c7"},
    {"name": "things_concepts.tsv", "directory": "2x/X8", "bytes": 497467,
     "md5": "cf6dc25d0aa272c386041a8ab80f07c1",
     "pointer_blob": "c4b0a6f9b274d660c4c866990f14839dffb9992e"},
)
PRESENTATION_FIELDS = (
    "participant", "session", "run", "run_order", "trial_in_run_zero_based",
    "glmsingle_trial_index_zero_based", "onset_TR_after_two_volume_removal",
    "provider_image_path", "image_name", "concept_name", "things_category_nr",
    "provider_category_nr", "provider_exemplar_nr", "condition",
    "image_presentation_number", "previous_same_image_index_gap",
)
RUN_FIELDS = ("participant", "session", "run", "run_order", "presentations",
              "first_glmsingle_trial_index_zero_based", "TR_seconds", "TR_count")
IDENTITY_FIELDS = ("provider_image_path", "image_name", "concept_name",
                   "things_category_nr", "provider_category_nr", "provider_exemplar_nr", "condition")
MATCH_FIELDS = ("participant", "session", "run", "run_order", "trial_in_run_zero_based",
                "glmsingle_trial_index_zero_based", "things_category_nr", "provider_category_nr",
                "provider_exemplar_nr", "onset_TR_after_two_volume_removal")
IMAGE_FIELDS = IDENTITY_FIELDS + tuple(f"retained_count_{s}" for s in SUBJECTS) + (
    "common_to_four", "eligible_image", "failure_reasons", "source_ordinals_positive_unique_in_each_participant")
CONCEPT_FIELDS = ("things_category_nr", "provider_category_nr", "concept_name", "uniqueID",
                  "original_common_image_count", "original_five_six_status", "qualified_image_count",
                  "eligible_concept", "taxonomy_membership_count", "taxonomy_unmapped")


def bounded_read(path, cap=FILE_CAP):
    """Fixed metadata inputs only; bound both stat and actual bytes read."""
    core.require(path.is_file() and path.stat().st_size <= cap, "metadata_file_exceeds_cap")
    with path.open("rb") as stream:
        payload = stream.read(cap + 1)
    core.require(len(payload) <= cap, "metadata_file_exceeds_cap")
    return payload


def table_rows(payload, fields=None, delimiter=","):
    try:
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""),
                                delimiter=delimiter, strict=True)
        header = reader.fieldnames
        core.require(header is not None and len(header) == len(set(header)), "invalid_metadata_header")
        if fields is not None:
            core.require(header == list(fields), "metadata_columns_changed")
        rows = []
        for i, row in enumerate(reader):
            core.require(None not in row and all(value is not None for value in row.values()),
                         "metadata_row_width_mismatch", row=i)
            rows.append(row)
        return header, rows
    except (UnicodeError, csv.Error):
        raise core.Refusal("invalid_metadata_csv") from None


def key(row):
    return row["participant"], core.integer(row["glmsingle_trial_index_zero_based"], "trial_index")


def run_key(row):
    return (row["participant"], core.integer(row["session"], "session"),
            core.integer(row["run"], "run"))


def check_reports(basis, event_report):
    core.require(basis.get("analysis_id") == "source_metadata" and
                 basis.get("structural_errors") == [] and basis.get("roles_assigned") is False,
                 "invalid_completed_source_metadata")
    people = basis.get("participants", [])
    core.require(len(people) == 4 and {x["participant"]: x["runs"] for x in people} == RUN_COUNTS,
                 "source_participant_scope_changed")
    for person in people:
        core.require(person.get("presentations_by_condition") == {"exp": 60 * RUN_COUNTS[person["participant"]]},
                     "retained_source_condition_scope_changed")
        axes = person.get("stage_axes", {})
        core.require(person.get("structural_issues") == [] and
                     person.get("B_C_D_stored_shapes_equal") is True and set(axes) == {"B", "C", "D"},
                     "unverified_BCD_metadata")
        shapes = []
        for stage in ("B", "C", "D"):
            axis = axes[stage]
            core.require(axis.get("trial_axis_matches_design_count") is True and
                         axis.get("neural_values_read") is False and
                         axis.get("singleton_removed_shape", [None])[0] == person["presentations"] ==
                         60 * RUN_COUNTS[person["participant"]], "BCD_trial_axis_mismatch")
            shapes.append(axis["stored_shape"])
        core.require(shapes[0] == shapes[1] == shapes[2], "BCD_shapes_differ")
    core.require(event_report.get("retained_roster_event_reconciliation_complete") is True and
                 event_report.get("status") == "completed" and event_report.get("errors") == [] and
                 event_report.get("projected_rows") == ROW_COUNT and
                 event_report.get("events_commit") == core.EVENT_COMMIT and
                 event_report.get("release_commit") == core.RELEASE_COMMIT and
                 event_report.get("fmriprep_commit") == core.FMRIPREP_COMMIT and
                 event_report.get("roles_assigned") is False,
                 "event_reconciliation_incomplete_or_changed")


def check_roster(runs, event_report):
    core.require(len(runs) == 833 and Counter(row["participant"] for row in runs) == RUN_COUNTS,
                 "retained_run_scope_changed")
    seen = set()
    for subject in SUBJECTS:
        selected = [row for row in runs if row["participant"] == subject]
        for i, row in enumerate(selected):
            k = run_key(row)
            core.require(k not in seen and k[1] >= 1 and 1 <= k[2] <= 6 and
                         core.integer(row["run_order"], "run_order") == i and
                         core.integer(row["presentations"], "presentations") == 60 and
                         core.integer(row["first_glmsingle_trial_index_zero_based"], "first_trial") == 60 * i and
                         core.integer(row["TR_count"], "TR_count") == 188 and
                         core.finite_number(row["TR_seconds"], "TR_seconds") == 1.49,
                         "retained_run_structure_changed")
            seen.add(k)
    evidence = event_report.get("runs", [])
    core.require(len(evidence) == 833 and len({run_key(row) for row in evidence}) == 833 and
                 {run_key(row) for row in evidence} == seen, "event_report_run_scope_changed")
    for row in evidence:
        core.require(row.get("status") == "matched" and row.get("mismatches") == [] and
                     row.get("projected_rows") == 60 and
                     (row.get("annex_content_identity_verified") is True or
                      row.get("annex_content_identity_verified_in_pilot") is True),
                     "event_run_not_verified")


def check_presentations(presentations, events, runs):
    """Check every old B/C/D coordinate against the completed projected source."""
    expected = {}
    run_rows = {run_key(row): row for row in runs}
    seen_per_run = defaultdict(set)
    for row in presentations:
        core.require(row["condition"] == "exp", "retained_source_condition_scope_changed")
        k, rk = key(row), run_key(row)
        core.require(k not in expected and rk in run_rows, "duplicate_or_unretained_presentation")
        trial = core.integer(row["trial_in_run_zero_based"], "trial_in_run")
        record = run_rows[rk]
        core.require(0 <= trial < core.integer(record["presentations"], "presentations") and
                     row["run_order"] == record["run_order"] and
                     k[1] == core.integer(record["first_glmsingle_trial_index_zero_based"], "first_trial") + trial,
                     "presentation_BCD_coordinate_changed")
        core.require(trial not in seen_per_run[rk], "duplicate_run_trial")
        seen_per_run[rk].add(trial)
        expected[k] = row
    core.require(all(len(seen_per_run[rk]) == core.integer(row["presentations"], "presentations")
                     for rk, row in run_rows.items()), "presentation_roster_incomplete")
    joined, seen = [], set()
    for event in events:
        k = key(event)
        core.require(k in expected and k not in seen, "event_coordinate_mismatch")
        previous = expected[k]
        for field in MATCH_FIELDS:
            core.require(event[field] == previous[field], "event_identity_coordinate_mismatch", field=field)
        core.require(event["expected_provider_image_path"] == previous["provider_image_path"] and
                     event["source_image_basename"] == previous["image_name"] and
                     event["identity_and_onset_match"] == "True" and
                     event["image_path_suffix_matches_expected"] == "True" and
                     event["retained_image_presentation_number"] == previous["image_presentation_number"] and
                     event["retained_previous_same_image_index_gap"] == previous["previous_same_image_index_gap"],
                     "event_image_mapping_changed")
        core.require(core.integer(event["source_order"], "source_order") ==
                     core.integer(previous["trial_in_run_zero_based"], "trial_in_run") and
                     core.integer(event["source_trial_number"], "source_trial_number") ==
                     core.integer(event["source_order"], "source_order") + 1 and
                     core.provider_onset_tr(event["onset_seconds"]) ==
                     core.integer(previous["onset_TR_after_two_volume_removal"], "onset_TR") and
                     core.finite_number(event["duration_seconds"], "duration") > 0,
                     "event_onset_or_order_changed")
        # Revalidate only the approved optional schedule columns. No responses exist here.
        for source, field in (("source_repetition", "repetition"), ("atypical", "atypical"),
                              ("not_for_memory", "not_for_memory"), ("delay_days", "delay_days"),
                              ("delay_seconds", "delay_seconds"), ("trials_since_lastrep", "trials_since_lastrep")):
            core.optional_value({field: event[source]}, field, None, [])
        joined.append({**event, "concept_name": previous["concept_name"], "condition": previous["condition"]})
        seen.add(k)
    core.require(seen == set(expected), "event_projection_incomplete")
    return joined


def load_sources():
    basis = json.loads(bounded_read(core.METADATA / "metadata_report.json"))
    event_report = json.loads(bounded_read(EVENTS / "reconciliation_report.json"))
    check_reports(basis, event_report)
    _, runs = table_rows(bounded_read(core.METADATA / "run_metadata.csv"), RUN_FIELDS)
    check_roster(runs, event_report)
    _, presentations = table_rows(bounded_read(core.METADATA / "presentation_identity.csv"), PRESENTATION_FIELDS)
    _, events = table_rows(bounded_read(EVENTS / "event_metadata.csv"), core.OUTPUT_FIELDS)
    core.require(len(presentations) == len(events) == ROW_COUNT, "fixed_presentation_count_changed")
    return check_presentations(presentations, events, runs)


def download_taxonomy(spec, deadline, receipt):
    """Exactly one public pinned object; existing annex digest verifies identity."""
    key_name = f"MD5E-s{spec['bytes']}--{spec['md5']}.tsv"
    url = f"{CONP}{spec['directory']}/{key_name}/{key_name}"
    remaining = deadline - time.monotonic()
    core.require(remaining > 0, "attempt_deadline")
    request = Request(url, headers={"User-Agent": "EP17-metadata-steward/1", "Accept-Encoding": "identity"})
    receipt["request_attempted"] = True
    try:
        with build_opener(core.NoRedirect()).open(request, timeout=min(20, remaining)) as response:
            core.require(response.status == 200, "unexpected_taxonomy_http_status")
            declared = response.headers.get("Content-Length")
            core.require(declared is not None and declared.isdigit() and int(declared) == spec["bytes"],
                         "taxonomy_content_length_mismatch")
            core.require(response.headers.get("Content-Encoding", "identity").lower() == "identity",
                         "taxonomy_content_encoding_changed")
            receipt["body_read_attempted"] = True
            payload = response.read(spec["bytes"] + 1)
            receipt["received_bytes"] = len(payload)
    except HTTPError as error:
        error.close()
        raise core.Refusal(f"taxonomy_http_status_{error.code}") from None
    except (URLError, TimeoutError, OSError):
        raise core.Refusal("bounded_taxonomy_transfer_failed") from None
    core.require(time.monotonic() < deadline, "attempt_deadline")
    core.require(len(payload) == spec["bytes"], "taxonomy_payload_length_mismatch")
    core.require(hashlib.md5(payload).hexdigest() == spec["md5"], "taxonomy_annex_md5_mismatch")
    receipt["annex_content_identity_verified"] = True
    return payload


def verify_pinned_pointer(spec):
    key_name = f"MD5E-s{spec['bytes']}--{spec['md5']}.tsv"
    payload = f"../../.git/annex/objects/{spec['directory']}/{key_name}/{key_name}".encode("ascii")
    blob = b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    core.require(len(payload) == 132 and hashlib.sha1(blob).hexdigest() == spec["pointer_blob"],
                 "pinned_taxonomy_pointer_mismatch")


def parse_taxonomy(category_bytes, concept_bytes, expected_rows=1854, expected_labels=53):
    header, categories = table_rows(category_bytes, delimiter="\t")
    # Project the concepts table immediately: unrelated rating cells stay in
    # the temporary CSV row, never in the returned metadata or diagnostics.
    try:
        reader = csv.reader(io.StringIO(concept_bytes.decode("utf-8-sig"), newline=""),
                            delimiter="\t", strict=True)
        concept_header = next(reader)
        core.require(len(concept_header) == len(set(concept_header)) and "uniqueID" in concept_header,
                     "invalid_concept_table_header")
        uid_position = concept_header.index("uniqueID")
        concepts = []
        for number, values in enumerate(reader):
            core.require(len(values) == len(concept_header), "concept_table_row_width_mismatch", row=number)
            concepts.append(values[uid_position])
    except (UnicodeError, csv.Error, StopIteration):
        raise core.Refusal("invalid_concept_table") from None
    core.require(header[:2] == ["uniqueID", "Word"] and len(header) == expected_labels + 2 and
                 len(categories) == len(concepts) == expected_rows,
                 "taxonomy_shape_or_header_changed")
    concept_ids = set(concepts)
    core.require(len(concept_ids) == expected_rows and
                 all(uid and uid == uid.strip() for uid in concepts),
                 "concept_table_uniqueIDs_invalid")
    labels = header[2:]
    seen, projected = set(), {}
    for number, category in enumerate(categories, 1):
        uid = category["uniqueID"]
        # The pinned tables have eight harmless row permutations outside EP17.
        # Join exact unique IDs, retaining category53's original numeric rows.
        core.require(bool(uid) and uid == uid.strip() and uid not in seen and uid in concept_ids,
                     "taxonomy_uniqueID_bridge_mismatch", row=number - 1)
        memberships = {}
        for label in labels:
            value = core.integer(category[label], "taxonomy_membership", number - 1)
            core.require(value in (0, 1), "taxonomy_not_binary", row=number - 1)
            memberships[label] = value
        # All non-ID columns in things_concepts, including ratings, are discarded.
        projected[number] = {"things_category_nr": number, "uniqueID": uid, **memberships}
        seen.add(uid)
    return labels, projected


def qualify(rows, taxonomy, labels):
    images, repeats, ordinals = {}, defaultdict(Counter), defaultdict(lambda: defaultdict(list))
    concept_identity = {}
    for row in rows:
        core.require(row["condition"] == "exp", "retained_source_condition_scope_changed")
        subject = row["participant"]
        core.require(subject in SUBJECTS, "unexpected_participant")
        path = row["expected_provider_image_path"]
        ppath = PurePosixPath(path)
        core.require(".." not in ppath.parts and ppath.name == row["source_image_basename"], "invalid_image_identity")
        identity = {"provider_image_path": path, "image_name": row["source_image_basename"],
                    **{field: row[field] for field in IDENTITY_FIELDS if field not in ("provider_image_path", "image_name")}}
        old = images.setdefault(path, identity)
        core.require(identity == old, "image_identity_differs_between_presentations")
        cid = core.integer(row["things_category_nr"], "things_category_nr")
        core.require(cid in taxonomy, "concept_absent_from_pinned_taxonomy")
        core.require(row["concept_name"] == ppath.parent.name == taxonomy[cid]["uniqueID"],
                     "selected_source_concept_taxonomy_anchor_mismatch")
        identity_c = (row["provider_category_nr"], row["concept_name"])
        core.require(concept_identity.setdefault(cid, identity_c) == identity_c, "concept_identity_disagrees")
        repeats[path][subject] += 1
        token = row["source_repetition"]
        ordinals[path][subject].append(None if token == "" else core.integer(token, "repetition"))
    image_rows, concept_images = [], defaultdict(list)
    chronology_unknown = []
    for path in sorted(images):
        identity = images[path]
        counts = repeats[path]
        reasons = []
        for subject in SUBJECTS:
            if counts[subject] == 0:
                reasons.append(f"absent:{subject}")
            elif counts[subject] < 2:
                reasons.append(f"fewer_than_two_retained_presentations:{subject}")
        ordinal_ok = True
        for subject in SUBJECTS:
            values = ordinals[path].get(subject, [])
            if not values or any(v is None or v < 1 for v in values) or len(set(values)) != len(values):
                ordinal_ok = False
                chronology_unknown.append({"provider_image_path": path, "participant": subject,
                                           "reason": "missing_or_nonunique_source_ordinals"})
        image = {**identity, **{f"retained_count_{s}": counts[s] for s in SUBJECTS},
                 "common_to_four": all(counts[s] > 0 for s in SUBJECTS),
                 "eligible_image": not reasons, "failure_reasons": ";".join(reasons),
                 "source_ordinals_positive_unique_in_each_participant": ordinal_ok}
        image_rows.append(image)
        concept_images[core.integer(identity["things_category_nr"], "things_category_nr")].append(image)
    concept_rows = []
    for cid, group in sorted(concept_images.items()):
        common = sum(row["common_to_four"] for row in group)
        qualified = sum(row["eligible_image"] for row in group)
        entry = taxonomy[cid]
        membership_count = sum(entry[label] for label in labels)
        concept_rows.append({"things_category_nr": cid, "provider_category_nr": concept_identity[cid][0],
                             "concept_name": concept_identity[cid][1], "uniqueID": entry["uniqueID"],
                             "original_common_image_count": common,
                             "original_five_six_status": str(common) if common in (5, 6) else "other",
                             "qualified_image_count": qualified, "eligible_concept": qualified >= 1,
                             "taxonomy_membership_count": membership_count,
                             "taxonomy_unmapped": membership_count == 0})
    return image_rows, concept_rows, chronology_unknown


def write_table(path, fields, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def publish(out, rows, images, concepts, taxonomy, labels):
    write_table(out / "image_qualification.csv", IMAGE_FIELDS, images)
    write_table(out / "concept_qualification.csv", CONCEPT_FIELDS, concepts)
    ids = {row["things_category_nr"] for row in concepts}
    write_table(out / "taxonomy_crosswalk.csv", ("things_category_nr", "uniqueID", *labels),
                (taxonomy[cid] for cid in sorted(ids)))
    by_image = {row["provider_image_path"]: row for row in images}
    fields = (*core.OUTPUT_FIELDS, "concept_name", "condition", "uniqueID", "eligible_image")
    def augmented():
        for row in rows:
            yield {**row, "uniqueID": taxonomy[core.integer(row["things_category_nr"], "things_category_nr")]["uniqueID"],
                   "eligible_image": by_image[row["expected_provider_image_path"]]["eligible_image"]}
    write_table(out / "presentation_qualification.csv", fields, augmented())


def complete_eligibility(report):
    report.update({"eligibility_complete": True, "eligible_720_concept_universe_verified": True,
                   "status": "completed"})


def run_attempt(out):
    start = time.monotonic()
    report = {"analysis_id": "source_eligibility", "stage": "metadata_eligibility_and_taxonomy",
              "status": "running", "eligibility_complete": False,
              "eligible_720_concept_universe_verified": False, "source_roles_complete": False,
              "roles_assigned": False, "folds_created": False, "repeat_halves_assigned": False,
              "actual_acquisition_order_verified": False, "chronology_qualified": False,
              "neural_values_read": False, "image_values_read": False, "model_features_read": False,
              "behavioral_outcomes_read": False, "candidate_scoring_run": False, "audit_opened": False,
              "unselected_public_normative_columns_persisted_or_reported": False,
              "public_concepts_projection": "Only uniqueID selected; other public rating cells transiently parsed then discarded. No CNeuroMod participant task responses read.",
              "BCD_metadata_mapping_verified": False, "neural_units_or_voxel_identity_verified": False,
              "repeat_rule_adopted_at": "2026-10-04 21:20 UTC",
              "minimum_retained_presentations_per_common_image_and_participant": 2,
              "minimum_qualified_images_per_concept": 1, "required_eligible_concepts": 720,
              "source_metadata_attempt": str(core.METADATA), "event_projection_attempt": str(EVENTS),
              "stimulus_commit": STIMULUS_COMMIT, "annotation_code_commit": ANNOTATION_CODE_COMMIT,
              "taxonomy_crosswalk_rule": "Preserve source CID; category53[CID-1].uniqueID equals exact source concept folder and occurs exactly once in concepts.uniqueID; no normalization or table-wide row-order assumption",
              "internal_seconds": INTERNAL_SECONDS, "external_seconds_required": 180,
              "local_input_byte_cap": FILE_CAP, "request_seconds": 20, "automatic_retries": 0,
              "maximum_taxonomy_requests": 2, "maximum_taxonomy_bytes": 724963,
              "maximum_taxonomy_body_bytes_including_oversize_sentinels": 724965,
              "taxonomy_license": "CC BY; Hebart et al. (2019), Stoinski et al. (2024)",
              "taxonomy_transfers": [], "errors": [], "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "hostname": socket.gethostname()}
    signal.signal(signal.SIGALRM, core.deadline_alarm)
    signal.alarm(INTERNAL_SECONDS)
    try:
        rows = load_sources()
        report["projected_presentations"] = len(rows)
        report["BCD_metadata_mapping_verified"] = True
        payloads = []
        for spec in TAXONOMY:
            verify_pinned_pointer(spec)
            receipt = {**spec, "request_attempted": False, "body_read_attempted": False,
                       "received_bytes": 0, "annex_content_identity_verified": False}
            report["taxonomy_transfers"].append(receipt)
            payloads.append(download_taxonomy(spec, start + INTERNAL_SECONDS, receipt))
        labels, taxonomy = parse_taxonomy(*payloads)
        del payloads
        images, concepts, chronology_unknown = qualify(rows, taxonomy, labels)
        report.update({"observed_image_count": len(images),
                       "common_experimental_image_count": sum(x["common_to_four"] and x["condition"] == "exp" for x in images),
                       "qualified_image_count": sum(x["eligible_image"] for x in images),
                       "qualified_concept_count": sum(x["eligible_concept"] for x in concepts),
                       "taxonomy_crosswalk_concepts": len(concepts),
                       "taxonomy_unmapped_concepts": sum(x["taxonomy_unmapped"] for x in concepts),
                       "original_common_exemplar_histogram": dict(Counter(x["original_common_image_count"] for x in concepts)),
                       "qualified_exemplar_histogram": dict(Counter(x["qualified_image_count"] for x in concepts)),
                       "chronology_unresolved_coordinates": chronology_unknown,
                       "chronology_note": "Source ordinal availability is reported; no global ordering or repeat-half assignment is asserted."})
        publish(out, rows, images, concepts, taxonomy, labels)
        core.require(report["qualified_concept_count"] == 720, "eligible_concept_count_not_720_no_rescue")
        core.require(time.monotonic() < start + INTERNAL_SECONDS, "attempt_deadline")
        complete_eligibility(report)
    except core.Refusal as error:
        report["errors"].append(error.detail)
        report["status"] = "stopped"
        report["eligibility_complete"] = report["eligible_720_concept_universe_verified"] = False
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        report["errors"].append({"code": "source_qualification_processing_failure"})
        report["status"] = "stopped"
        report["eligibility_complete"] = report["eligible_720_concept_universe_verified"] = False
    finally:
        signal.alarm(0)
    report["elapsed_seconds"] = time.monotonic() - start
    report["remaining_requirements"] = ["chronology qualification and repeat halves",
        "prospective balance precision and finite optimizer bounds", "whole-concept roles, folds and blocks",
        "materialized role-separated handoffs and later authorized neural/model qualification"]
    with (out / "qualification_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "eligible_concepts": report.get("qualified_concept_count"),
                      "eligibility_complete": report["eligibility_complete"], "roles_assigned": False}))
    return 0 if (report["status"] == "completed" and report["eligibility_complete"] and
                 report["eligible_720_concept_universe_verified"]) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    core.require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname(),
                 "requires_coordinator_owned_compute")
    out = args.out.resolve()
    core.require(out.parent == OUTPUT_PARENT.resolve() and not out.exists(), "fresh_episode_attempt_required")
    out.mkdir(parents=True)
    return run_attempt(out)


if __name__ == "__main__":
    raise SystemExit(main())
