#!/usr/bin/env python3
"""Freeze qualified metadata roles/join and source-only EP17/18 overlap.

Read the existing normalized event join and pinned public stimulus CSV only.
No EEG/image/checkpoint payload, live sibling artifact, or source-file copy is
opened. Role-filtered metadata files do not establish runtime access isolation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path, PurePosixPath
import socket
import time


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def canonical(path):
    parts = PurePosixPath(path).parts
    if len(parts) < 2 or ".." in parts:
        raise ValueError("Malformed provider image identity")
    return "/".join(parts[-2:])


def source_image_identity(row, things_images):
    """Resolve provider THINGS indices, including flattened test-image paths."""
    number = row["things_image_nr"].strip()
    category = row["things_category_nr"].strip()
    if not number and not category:
        return canonical(row["image_path"]), None
    if not number or not category:
        raise ValueError("Partially mapped provider THINGS image/concept identity")
    index = int(number) - 1
    if not 0 <= index < len(things_images):
        raise ValueError("Provider one-based THINGS image index is outside metadata")
    identity = things_images[index]
    if PurePosixPath(identity).name != PurePosixPath(row["image_path"]).name:
        raise ValueError("Provider THINGS image index and filename disagree")
    if row["condition"] == "exp" and canonical(row["image_path"]) != identity:
        raise ValueError("Experimental concept folder disagrees with THINGS identity")
    return identity, identity.split("/", 1)[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalization-report", type=Path, required=True)
    parser.add_argument("--role-proposal", type=Path, required=True)
    parser.add_argument("--clock-report", type=Path, required=True)
    parser.add_argument("--cneuromod-image-csv", type=Path, required=True)
    parser.add_argument("--things-image-paths", type=Path, required=True)
    parser.add_argument("--exposure-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use the coordinator's verified compute allocation")
    started = time.monotonic()
    normalized = read_json(args.normalization_report)
    proposal = read_json(args.role_proposal)
    clocks = read_json(args.clock_report)
    exposure = read_json(args.exposure_record)
    if normalized["status"] != "source_structure_and_sample_origin_reconciled":
        raise ValueError("Source normalization is not qualified")
    if normalized["eeg_or_image_values_opened"] or proposal["eeg_or_image_values_opened"]:
        raise ValueError("Unexpected payload exposure in metadata prerequisites")
    if proposal["sumsq_exact"] != proposal["sumsq_lower_bound_exact"] or proposal["sumsq_exact"] != proposal["sumsq_upper_bound_exact"]:
        raise ValueError("Role proposal lacks completed exact objective bounds")
    roles = {"development": proposal["development_proposal"], "audit": proposal["audit_proposal"]}
    eligible = sorted(normalized["participant_event_counts"])
    assigned = roles["development"] + roles["audit"]
    if len(eligible) != 46 or len(roles["development"]) != 30 or len(roles["audit"]) != 16 or sorted(assigned) != eligible or len(set(assigned)) != 46:
        raise ValueError("Whole-person 30/16 assignment differs from qualified cohort")
    for quartile in (eligible[:12], eligible[12:24], eligible[24:35], eligible[35:]):
        if len(set(quartile) & set(roles["audit"])) != 4:
            raise ValueError("Audit quartile constraint differs from proposal")
    if len({"sub-49", "sub-50"} & set(roles["audit"])) != 1:
        raise ValueError("Montage pair is not split")
    if {r["participant"] for r in clocks["participant_diagnostics"]} != {"sub-36", "sub-38"}:
        raise ValueError("Required clock-discontinuity diagnosis is missing")
    for result in clocks["participant_diagnostics"]:
        if result["events"] != 24648 or not result["discontinuities"] or not all(d["physical_sequence_changes"] and d["recording_segment_changes"] for d in result["discontinuities"]):
            raise ValueError("Observed large clock discontinuities are not contained by fixed sequence boundaries")
    if exposure["current_session_eeg_or_image_values_opened"] or not exposure["metadata_freeze_authorized_under_existing_contract"]:
        raise ValueError("Exposure review does not support metadata freeze")

    source_images = defaultdict(set)
    source_concepts = defaultdict(set)
    unmapped_images = defaultdict(set)
    with args.things_image_paths.open(encoding="utf-8-sig") as stream:
        things_images = [canonical(line.strip()) for line in stream if line.strip()]
    if len(things_images) != 26107 or len(set(things_images)) != 26107:
        raise ValueError("THINGS identity inventory differs from qualified metadata")
    with args.cneuromod_image_csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"image_path", "condition", "things_category_nr", "things_image_nr"}.issubset(reader.fieldnames or []):
            raise ValueError("Pinned source image CSV columns changed")
        source_rows = 0
        image_to_source_concept = {}
        source_number_to_concept = {}
        for row in reader:
            identity, concept = source_image_identity(row, things_images)
            previous = image_to_source_concept.setdefault(identity, row["things_category_nr"])
            if previous != row["things_category_nr"]:
                raise ValueError("Provider image maps to conflicting concept numbers")
            source_images[row["condition"]].add(identity)
            if concept is None:
                unmapped_images[row["condition"]].add(identity)
            else:
                if source_number_to_concept.setdefault(row["things_category_nr"], concept) != concept:
                    raise ValueError("Provider THINGS concept number maps to conflicting folder identities")
                source_concepts[row["condition"]].add(concept)
            source_rows += 1

    args.output_dir.mkdir(parents=True, exist_ok=False)
    role_by_participant = {person: role for role, persons in roles.items() for person in persons}
    counts = Counter()
    per_person_images = defaultdict(lambda: {"main": set(), "repeated": set()})
    per_person_counts = defaultdict(Counter)
    concept_number_to_key = {}
    key_to_concept_number = {}
    with ExitStack() as stack:
        source = stack.enter_context(gzip.open(normalized["normalized_join"], "rt", newline="", encoding="utf-8"))
        reader = csv.DictReader(source, delimiter="\t")
        required = {"participant", "is_repeated", "canonical_image", "concept_number", "sample", "sample_zero_based", "source_onset_seconds", "source_time_stimon_seconds", "onset_recording_seconds"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Normalized join lacks required original/normalized fields")
        writers = {}
        for role in roles:
            stream = stack.enter_context(gzip.open(args.output_dir / f"{role}_event_image_join.tsv.gz", "wt", newline="", encoding="utf-8"))
            writers[role] = csv.DictWriter(stream, fieldnames=reader.fieldnames, delimiter="\t")
            writers[role].writeheader()
        for row in reader:
            person = row["participant"]
            if person not in role_by_participant or row["is_repeated"] not in {"0", "1"}:
                raise ValueError("Unexpected participant or event class")
            if int(row["sample_zero_based"]) != int(row["sample"]) - 1:
                raise ValueError("Normalized sample rule differs from qualified source")
            kind = "repeated" if row["is_repeated"] == "1" else "main"
            identity = canonical(row["canonical_image"])
            concept = identity.split("/", 1)[0]
            if kind == "main":
                number = row["concept_number"]
                if concept_number_to_key.setdefault(number, concept) != concept or key_to_concept_number.setdefault(concept, number) != number:
                    raise ValueError("Main concept-number/identity mapping is not one-to-one")
            per_person_images[person][kind].add(identity)
            per_person_counts[person][kind] += 1
            counts[person] += 1
            writers[role_by_participant[person]].writerow(row)
    if dict(counts) != normalized["participant_event_counts"]:
        raise ValueError("Frozen views do not retain every qualified event")
    reference = per_person_images[eligible[0]]
    for person in eligible:
        if per_person_counts[person] != {"main": 22248, "repeated": 2400} or per_person_images[person] != reference:
            raise ValueError("Person event counts or image inventory differ from qualified common design")
    if len(reference["main"]) != 22248 or len(reference["repeated"]) != 200 or len(concept_number_to_key) != 1854:
        raise ValueError("Main/repeated image or concept inventory differs from contract")
    images_union = set().union(*source_images.values())
    concepts_union = set().union(*source_concepts.values())
    overlap = {"scope": "pinned CNeuroMod source inventory, not an EP17 analysis universe",
               "cneuromod_image_csv": str(args.cneuromod_image_csv), "source_rows": source_rows,
               "things_image_paths": str(args.things_image_paths),
               "source_unique_images": len(images_union), "source_unique_concepts": len(concepts_union),
               "source_images_without_THINGS_mapping": sum(len(values) for values in unmapped_images.values()),
               "unmapped_image_identities_by_condition": {condition: sorted(values) for condition, values in unmapped_images.items()},
               "source_by_condition": {condition: {"images": len(source_images[condition]), "concepts": len(source_concepts[condition]), "unmapped_images": len(unmapped_images[condition])} for condition in sorted(source_images)},
               "canonical_identity_rule": "Provider one-based things_image_nr selects the preserved THINGS image-paths row; verify filename and experimental folder. Concept keys derive only from verified mapped rows; blank catch mappings remain nonconcept identities.",
               "exact_analysis_universe_overlap_with_EP17": "unresolved_until_EP17_qualifies_its_source_event_inventory",
               "pixel_identity": "existing September central-directory receipt shows all 8640 experimental archive entries differ from THINGS in size and/or CRC; this does not cover flattened test/catch CSV rows; no new payload check"}
    for kind in ("main", "repeated"):
        images = reference[kind]
        concepts = {item.split("/", 1)[0] for item in images}
        overlap[kind] = {"EP18_unique_images": len(images), "EP18_unique_concepts": len(concepts),
                         "shared_image_count": len(images & images_union), "shared_concept_count": len(concepts & concepts_union),
                         "shared_image_identities": sorted(images & images_union),
                         "shared_concept_identities": sorted(concepts & concepts_union),
                         "by_source_condition": {condition: {"shared_images": len(images & source_images[condition]), "shared_concepts": len(concepts & source_concepts[condition])} for condition in sorted(source_images)}}
    write_json(args.output_dir / "source_overlap.json", overlap)
    for role, persons in roles.items():
        manifest = {"role": role, "participants": persons,
                    "frozen_event_join": f"{role}_event_image_join.tsv.gz",
                    "event_count": sum(counts[p] for p in persons),
                    "EEG_relative_paths_for_controlled_mount": [f"{p}/eeg/{p}_task-rsvp_eeg.{ext}" for p in persons for ext in ("eeg", "vhdr", "vmrk", "json")],
                    "paths_are_a_deployment_manifest_not_existing_symlinks_or_access_controls": True,
                    "runtime_root": "/inputs/development_eeg" if role == "development" else "/inputs/audit_eeg",
                    "audit_seal_verified": False, "EEG_access_authorized_by_this_manifest": False}
        write_json(args.output_dir / f"{role}_manifest.json", manifest)
    report = {"analysis_id": "input_role_preparation", "phase": "freeze_outcome_blind_metadata_roles_and_join",
              "observed_utc": datetime.now(timezone.utc).isoformat(), "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "elapsed_seconds": time.monotonic() - started, "role_proposal": str(args.role_proposal),
              "normalization_report": str(args.normalization_report), "clock_report": str(args.clock_report), "exposure_record": str(args.exposure_record),
              "source_overlap": "source_overlap.json", "roles": roles, "roles_frozen": True, "join_frozen": True,
              "frozen_join_representation": "disjoint development and audit metadata joins; their union preserves all qualified event rows and original columns",
              "events": sum(counts.values()), "participant_event_counts": dict(counts),
              "common_recorded_channel_order": normalized["common_recorded_channel_order"],
              "scientific_claim": "none; metadata assignment and stimulus inventory only",
              "cross_episode_independence": "EP17/EP18 are design-exposed and stimulus-related, not independent stimulus-family confirmation",
              "audit_seal_verified": False, "EEG_driven_candidate_comparison_authorized": False,
              "eeg_image_checkpoint_payload_values_opened": False,
              "raw_source_fields_preserved": True, "display_clock_absolute_accuracy_claimed": False}
    write_json(args.output_dir / "frozen_inputs_and_roles.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "participant_event_counts"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
