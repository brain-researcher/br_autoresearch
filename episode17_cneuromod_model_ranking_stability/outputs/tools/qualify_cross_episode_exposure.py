#!/usr/bin/env python3
"""Bounded EP17/EP18 metadata overlap; never reads arrays, pixels or weights."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path, PurePosixPath
import resource
import signal
import socket
import os
import time


EPISODE = Path(__file__).resolve().parents[2]
ROLES = ("development", "support_calibration", "sealed_audit")
ROLE_COUNTS = dict(zip(ROLES, (480, 120, 120)))
ASSIGNMENT = EPISODE / "outputs/runs/concept_role_assignment/full001"
HANDOFF = EPISODE / "outputs/runs/cross_episode_exposure/ep18_handoff001"
ACQUISITION = Path(os.environ["OAK"]) / "users/zijiao/br_autoresearch_data/steward_acquisition"
PROVIDER = ACQUISITION / "cneuromod_images_fmri_pinned_f8f5162_20260922/provider_git/image_paths_fmri.csv"
THINGS = ACQUISITION / "things_metadata_osf_jum2f_20260922/payload/image-paths.csv"
INPUT_CAP = 10 * 1024 * 1024
OUTPUT_CAP = 8 * 1024 * 1024
SECONDS = 60
ROLE_FIELDS = ("role", "outer_fold", "development_block", "audit_block")


class Refusal(Exception):
    pass


def require(condition, code):
    if not condition:
        raise Refusal(code)


def utc():
    return datetime.now(timezone.utc).isoformat()


def alarm(signum, frame):
    raise Refusal("deadline_or_termination")


def check(deadline):
    require(time.monotonic() < deadline, "internal_deadline")


def text_input(path, ledger, deadline):
    check(deadline)
    require(path.is_file() and path.stat().st_size <= INPUT_CAP, "input_missing_or_byte_cap")
    with path.open("rb") as stream:
        data = stream.read(INPUT_CAP + 1)
    require(len(data) <= INPUT_CAP, "input_byte_cap")
    ledger.append({"path": str(path), "bytes": len(data)})
    check(deadline)
    return data.decode("utf-8-sig")


def csv_input(path, ledger, deadline):
    reader = csv.DictReader(io.StringIO(text_input(path, ledger, deadline)))
    require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), "CSV_header_invalid")
    rows = list(reader)
    require(all(None not in row and all(value is not None for value in row.values()) for row in rows),
            "CSV_row_width_invalid")
    return rows


def integer(token):
    require(isinstance(token, str), "integer_type")
    value = int(token)
    require(value > 0 and str(value) == token, "positive_integer_invalid")
    return value


def boolean(token):
    require(token in ("True", "False"), "boolean_invalid")
    return token == "True"


def unique_strings(items, expected):
    require(isinstance(items, list) and len(items) == expected
            and all(isinstance(item, str) and item for item in items)
            and len(set(items)) == expected, "overlap_identity_list_invalid")
    return set(items)


def canonical_path(value, prefix):
    require(isinstance(value, str) and "\\" not in value, "image_path_invalid")
    path = PurePosixPath(value)
    require(len(path.parts) == 3 and path.parts[0] == prefix
            and str(path) == value and not any(x in (".", "..") for x in path.parts),
            "image_path_structure_invalid")
    return path.parts[1], path.parts[2], "/".join(path.parts[1:])


def overlap_summary(rows, main_images, repeated_images, main_concepts, repeated_concepts):
    result = {}
    for role in ("all",) + ROLES:
        selected = rows if role == "all" else [row for row in rows if row["role"] == role]
        identities = sorted(row["canonical_THINGS_image"] for row in selected)
        names = sorted({row["uniqueID"] for row in selected})
        image_set, concept_set = set(identities), set(names)
        result[role] = {
            "image_count": len(identities), "concept_count": len(names),
            "canonical_image_identities": identities, "concept_identities": names,
            "EP18_main_image_overlap_count": len(image_set & main_images),
            "EP18_main_image_overlap_identities": sorted(image_set & main_images),
            "EP18_repeated_image_overlap_count": len(image_set & repeated_images),
            "EP18_repeated_image_overlap_identities": sorted(image_set & repeated_images),
            "EP18_main_concept_overlap_count": len(concept_set & main_concepts),
            "EP18_repeated_concept_overlap_count": len(concept_set & repeated_concepts),
            "EP18_repeated_concept_overlap_identities": sorted(concept_set & repeated_concepts),
        }
    return result


def compare(report, deadline):
    ledger = report["inputs"]
    handoff = {name: json.loads(text_input(HANDOFF / (name + ".json"), ledger, deadline))
               for name in ("source_overlap", "frozen_inputs_and_roles", "ep17_ep18_exposure_record", "feature_registry")}
    overlap, freeze, exposure, features = (handoff[key] for key in
        ("source_overlap", "frozen_inputs_and_roles", "ep17_ep18_exposure_record", "feature_registry"))
    require(freeze.get("roles_frozen") is True and freeze.get("join_frozen") is True,
            "EP18_metadata_not_frozen")
    development, audit = freeze["roles"]["development"], freeze["roles"]["audit"]
    require(len(development) == len(set(development)) == 30 and len(audit) == len(set(audit)) == 16
            and not set(development) & set(audit), "EP18_role_roster_invalid")
    require(exposure["source_identity"]["CNeuroMod_stimuli_revision"] ==
            "f8f5162f79fb2e9fb0d8cbb8bac610db11a54180"
            and exposure["source_identity"]["EEG"] == "OpenNeuro ds003825 v1.2.0"
            and overlap["cneuromod_image_csv"] == str(PROVIDER)
            and overlap["things_image_paths"] == str(THINGS), "source_identity_changed")
    require(overlap["main"]["EP18_unique_images"] == 22248
            and overlap["main"]["EP18_unique_concepts"] == 1854
            and overlap["repeated"]["EP18_unique_images"] == 200
            and overlap["repeated"]["EP18_unique_concepts"] == 200, "EP18_inventory_changed")
    main_images = unique_strings(overlap["main"]["shared_image_identities"], 8640)
    main_concepts = unique_strings(overlap["main"]["shared_concept_identities"], 720)
    repeated_images = unique_strings(overlap["repeated"]["shared_image_identities"], 100)
    repeated_concepts = unique_strings(overlap["repeated"]["shared_concept_identities"], 200)

    concepts = csv_input(ASSIGNMENT / "concept_assignments.csv", ledger, deadline)
    images = csv_input(ASSIGNMENT / "image_assignments.csv", ledger, deadline)
    provider_rows = csv_input(PROVIDER, ledger, deadline)
    things_rows = list(csv.reader(io.StringIO(text_input(THINGS, ledger, deadline))))
    require(all(len(row) == 1 for row in things_rows), "THINGS_single_column_required")
    require(len(provider_rows) == overlap["source_rows"] == 8840, "provider_inventory_changed")
    provider = {row["image_path"]: row for row in provider_rows}
    require(len(provider) == len(provider_rows), "provider_image_duplicate")
    require(len(concepts) == 720 and Counter(row["role"] for row in concepts) == ROLE_COUNTS,
            "concept_role_counts_changed")
    by_id = {integer(row["things_category_nr"]): row for row in concepts}
    require(len(by_id) == 720 and len({row["uniqueID"] for row in concepts}) == 720,
            "concept_identity_duplicate")
    require({row["uniqueID"] for row in concepts} == main_concepts, "concept_universe_overlap_changed")
    require(len(images) == 4320 and len({row["provider_image_path"] for row in images}) == 4320,
            "observed_image_roster_changed")
    mapped = []
    for row in images:
        check(deadline)
        source = provider[row["provider_image_path"]]
        number = integer(source["things_image_nr"])
        require(number <= len(things_rows), "THINGS_row_out_of_bounds")
        folder, filename, key = canonical_path(things_rows[number - 1][0], "images")
        source_folder, source_name, source_key = canonical_path(row["provider_image_path"], "images_fmri")
        concept = by_id[integer(row["things_category_nr"])]
        require(source["condition"] == row["condition"] == "exp"
                and source["things_category_nr"] == row["things_category_nr"]
                and source["category_nr"] == row["provider_category_nr"] == concept["provider_category_nr"]
                and source["exemplar_nr"] == row["provider_exemplar_nr"]
                and folder == source_folder == row["concept_name"] == concept["uniqueID"]
                and filename == source_name == row["image_name"] and key == source_key,
                "provider_THINGS_assignment_identity_mismatch")
        require(all(row[field] == concept[field] for field in ROLE_FIELDS), "image_role_inheritance_mismatch")
        common, eligible = boolean(row["common_to_four"]), boolean(row["eligible_image"])
        require(not eligible or common, "eligible_image_not_common")
        require(key in main_images, "observed_image_absent_from_frozen_EP18_main_overlap")
        mapped.append({"provider_image_path": row["provider_image_path"], "things_image_nr": number,
                       "canonical_THINGS_image": key, "things_category_nr": integer(row["things_category_nr"]),
                       "uniqueID": concept["uniqueID"], "common_to_four": common, "eligible_image": eligible,
                       **{field: row[field] for field in ROLE_FIELDS}})
    require(len({row["canonical_THINGS_image"] for row in mapped}) == 4320, "canonical_image_duplicate")
    common = [row for row in mapped if row["common_to_four"]]
    eligible = [row for row in mapped if row["eligible_image"]]
    require(len(common) == 3840 and len(eligible) == 3634, "qualified_image_counts_changed")
    common_counts = Counter(row["things_category_nr"] for row in common)
    eligible_counts = Counter(row["things_category_nr"] for row in eligible)
    for number, row in by_id.items():
        require(common_counts[number] == integer(row["original_common_image_count"])
                and eligible_counts[number] == integer(row["qualified_image_count"]),
                "concept_image_qualification_counts_changed")
    report["concept_role_counts"] = dict(Counter(row["role"] for row in concepts))
    report["concept_role_identities"] = {role: sorted(row["uniqueID"] for row in concepts if row["role"] == role)
                                        for role in ROLES}
    report["frozen_EP18_participant_roles"] = freeze["roles"]
    report["EP18_audit_seal_observed_in_handoff"] = freeze["audit_seal_verified"]
    report["image_identity_mapping"] = mapped
    report["scope_comparisons"] = {
        name: overlap_summary(rows, main_images, repeated_images, main_concepts, repeated_concepts)
        for name, rows in (("all_observed", mapped), ("common_to_four", common), ("repeat_qualified", eligible))}
    checkpoint_pairs = (("supervised_resnet50_v1", "SUP_R50_V1",
        "https://download.pytorch.org/models/resnet50-0676ba61.pth"),
        ("dino_resnet50", "DINO_R50",
         "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"))
    checkpoint_records = []
    for ep18_id, ep17_id, url in checkpoint_pairs:
        record = features["features"][ep18_id]
        require(record["checkpoint"] == url and ep18_id in features["core_image_panel"],
                "shared_checkpoint_recipe_changed")
        history = [entry for entry in exposure["checkpoint_reuse_candidates_from_official_sources"]
                   if entry["official_source"] == url]
        require(len(history) == 1 and isinstance(history[0]["EP18_loaded"], bool),
                "checkpoint_access_history_ambiguous")
        checkpoint_records.append({"EP17_model": ep17_id, "EP18_model": ep18_id, "official_checkpoint": url,
            "EP18_preprocessing": record["preprocessing"], "EP18_selected_layer": record["selected_layer"],
            "EP18_output_dimension": record["dimension"], "EP18_loaded_as_of_handoff": history[0]["EP18_loaded"],
            "feature_array_reuse_authorized": False, "identical_feature_values_established": False})
    report["checkpoint_reuse"] = checkpoint_records
    report["feature_recipe_comparison"] = {
        "EP17_frozen_recipe": "Common RGB bilinear shortest-edge256/center224/ImageNet normalization; separate GAP layer1/2/3/4 vectors.",
        "EP18_frozen_recipe": "Supervised final avgpool uses bilinear; DINO final avgpool uses bicubic; training transforms remain EP18-local.",
        "shared_checkpoint_is_not_identical_feature": True,
        "EP17_live_cache_or_feature_consumption": False,
        "SWAV_selected_in_EP18": False,
        "pretraining_THINGS_exact_exposure": "unknown; this metadata comparison cannot establish clean pretraining exposure"}
    report["first_access_history_from_EP18_handoff"] = exposure["first_access_history"]
    report["first_access_history_limit"] = exposure["history_limit"]
    report["EP17_access_history_reference"] = "Coordinator-maintained EP17 experiment_log.md; not read by this metadata-only comparison."
    report["pixel_identity_limit"] = overlap["pixel_identity"]
    report["interpretation"] = (
        "Maintain both frozen designs and roles. EP17 and EP18 are correlated, design-exposed THINGS studies; "
        "new EEG participants or a different modality do not create independent stimulus-family confirmation. "
        "Exact concept/image identities are compared, not identical archive bytes. No selection changes, "
        "neural evidence, audit opening, runtime isolation or clean pretraining exposure are established.")
    check(deadline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    require(out.parent == EPISODE / "outputs/runs/cross_episode_exposure", "output_scope_invalid")
    out.mkdir(mode=0o700, exist_ok=False)
    started = time.monotonic()
    deadline = started + SECONDS
    report = {"analysis_id": "cross_episode_exposure", "status": "running",
        "exposure_comparison_complete": False, "exposure_decision": "correlated_design_exposed_unchanged_designs",
        "observed_utc": utc(), "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "internal_seconds": SECONDS, "external_seconds_required": 90, "requested_cpu_count": 1,
        "requested_memory_gib": 1, "input_file_byte_cap": INPUT_CAP, "output_byte_cap": OUTPUT_CAP,
        "network_requests": 0, "automatic_retries": 0, "inputs": [], "errors": [],
        "neural_values_read": False, "pixel_values_read": False, "feature_values_read": False,
        "checkpoint_payloads_read": False, "audit_opened_by_this_attempt": False,
        "selection_or_role_changes": False, "physical_access_separation_established": False,
        "independent_stimulus_family_confirmation": False, "identical_pixel_bytes_claimed": False}
    signal.signal(signal.SIGALRM, alarm)
    signal.signal(signal.SIGTERM, alarm)
    signal.alarm(SECONDS)
    try:
        compare(report, deadline)
        check(deadline)
        report.update(status="completed", exposure_comparison_complete=True)
        # Budget serialization too; success is never inferred from partial lists.
        payload = json.dumps(report, indent=2, allow_nan=False).encode("utf-8")
        require(len(payload) + 2048 <= OUTPUT_CAP, "output_byte_cap")
        check(deadline)
    except Exception as error:
        report.update(status="refused", exposure_comparison_complete=False)
        report["errors"].append({"type": type(error).__name__, "message": str(error)[:1000]})
        # Large partially assembled lists are not a completed overlap release.
        report.pop("image_identity_mapping", None)
        report.pop("scope_comparisons", None)
    finally:
        signal.alarm(0)
    report["elapsed_seconds"] = time.monotonic() - started
    report["finished_utc"] = utc()
    report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = json.dumps(report, indent=2, allow_nan=False).encode("utf-8")
    require(len(payload) <= OUTPUT_CAP, "terminal_report_byte_cap")
    with (out / "exposure_report.json").open("xb") as stream:
        stream.write(payload)
    return 0 if report["status"] == "completed" and report["exposure_comparison_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
