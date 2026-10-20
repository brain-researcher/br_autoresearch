"""Check the source validation-image index against the existing repeated join.

Metadata only: authenticate the original E3/E1 binding, then test whether the
documented teststimnumber (0..199) names the same image in all 12 sequences.
Presentation numbering is descriptive because its source meaning is ambiguous.
No EEG amplitude reader, score, preprocessing change or participant exclusion.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from production_core_direct import AllocatedBudget, atomic_json
from repeated_image_sequences import (BoundRepeatedParticipant, bind_participant,
    _json, _source_repeated, require)
from prepare_continuous_eeg import _integer
from qualify_structural_inputs import image_key
from report_repeated_image_control import DEVELOPMENT_PARTICIPANTS


EPISODE = Path(__file__).resolve().parents[2]
INDEX_DESCRIPTION = "stimulus number in the validation image set (0-199)"
SCHEMA = "ep18_repeated_identity_metadata_v1"
MISSING = (None, "", "n/a", "nan", "None")


def _descriptive(rows, field):
    values = [row.get(field) for row in rows]
    present = [value for value in values if value not in MISSING]
    return {"unique_value_count": len(set(present)),
        "missing_value_count": len(values) - len(present),
        "first_value": values[0], "last_value": values[-1]}


def check_bound_identity(bound, source_rows, *, budget_check):
    """Pure check of already authenticated metadata; never infer image labels."""
    require(isinstance(bound, BoundRepeatedParticipant), "authentic repeated metadata binding required")
    require(len(source_rows) == 2400 and len(bound.sequences) == 12
            and tuple(span.ordinal for span in bound.sequences) == tuple(range(12))
            and len(bound.image_ids) == len(set(bound.image_ids)) == 200,
            "all 12 original sequences and 200 images are required")
    by_sample = {}
    for row in source_rows:
        budget_check()
        sample = _integer(row["sample"], "source sample") - 1
        require(sample not in by_sample, "duplicate repeated source sample")
        by_sample[sample] = row
    number_to_image, image_to_number, sequence_records = {}, {}, []
    used_samples = set()
    for span in bound.sequences:
        budget_check()
        require(len(span.event_samples) == len(span.image_indices) == 200,
                "every bound E3 sequence must retain 200 events")
        rows, labels = [], []
        for sample, image_index in zip(span.event_samples, span.image_indices):
            require(sample in by_sample and sample not in used_samples,
                    "source samples must match the disjoint bound E3/E1 intervals")
            used_samples.add(sample)
            row = by_sample[sample]
            identity = image_key(row["stim"])
            require(identity == bound.image_ids[image_index]
                    and _integer(row["isteststim"], "source event class") == 1,
                    "source image/class differs from the authenticated repeated binding")
            require(row.get("teststimnumber") not in MISSING,
                    f"missing teststimnumber in {span.sequence_id} at sample {sample + 1}")
            number = _integer(row["teststimnumber"], "teststimnumber")
            require(0 <= number < 200,
                    f"teststimnumber outside 0..199 in {span.sequence_id} at sample {sample + 1}")
            previous_image = number_to_image.setdefault(number, identity)
            previous_number = image_to_number.setdefault(identity, number)
            require(previous_image == identity and previous_number == number,
                    f"unstable teststimnumber/image mapping in {span.sequence_id} at sample {sample + 1}: "
                    f"index {number}, image {identity}, prior image {previous_image}, prior index {previous_number}")
            rows.append(row)
            labels.append(number)
        require(sorted(labels) == list(range(200)),
                f"{span.sequence_id} lacks exactly one occurrence of every index 0..199")
        sequence_records.append({"sequence_id": span.sequence_id, "ordinal": span.ordinal,
            "events": 200, "teststimnumber_coverage_0_through_199_once": True,
            "source_sequencenumber": _descriptive(rows, "sequencenumber"),
            "source_presentationnumber": _descriptive(rows, "presentationnumber"),
            "source_withinsequencenumber": _descriptive(rows, "withinsequencenumber")})
    require(len(used_samples) == len(by_sample) == 2400
            and set(number_to_image) == set(range(200))
            and set(image_to_number) == set(bound.image_ids), "incomplete repeated index/image coverage")
    return {"schema": SCHEMA, "status": "complete_repeated_identity_metadata",
        "participant_id": bound.participant_id, "stable_bijection": True,
        "physical_sequence_count": 12, "image_count": 200, "event_count": 2400,
        "image_by_teststimnumber": [number_to_image[number] for number in range(200)],
        "sequences": sequence_records, "presentation_numbering_used_as_admission_rule": False,
        "EEG_amplitudes_read": False, "audit_access": False,
        "concept_or_false_scores_computed": False, "primary_analysis_changed": False}


def check_participant(*, development_manifest, bids_root, frozen_channel_order,
                      coefficient_record, participant_id, budget_check):
    """Use the existing metadata-only reader and original source codebook."""
    codebook_path = Path(bids_root) / "task-rsvp_events.json"
    codebook = _json(codebook_path, 128 * 1024)
    require(codebook.get("teststimnumber", {}).get("Description") == INDEX_DESCRIPTION,
            "source codebook does not establish the selected validation-image index")
    bound = bind_participant(development_manifest=development_manifest, bids_root=bids_root,
        frozen_channel_order=frozen_channel_order, coefficient_record=coefficient_record,
        participant_id=participant_id, budget_check=budget_check)
    source = _source_repeated(bound.recording.paths, budget_check)
    result = check_bound_identity(bound, source, budget_check=budget_check)
    result["source_codebook"] = str(codebook_path)
    result["teststimnumber_description"] = INDEX_DESCRIPTION
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default=str(EPISODE / "outputs/config/development_eeg_release_20261012_001.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--rss-limit-bytes", type=int, default=384 * 1024**2)
    args = parser.parse_args()
    budget = AllocatedBudget(deadline_epoch=args.deadline_epoch, rss_limit_bytes=args.rss_limit_bytes,
        durable_byte_limit=16 * 1024**2, record_byte_limit=2 * 1024**2, threads=1)
    budget.require_compute_threads()
    release = _json(args.release, 64 * 1024)
    manifest = _json(release["development_manifest"], 8 * 1024**2)
    require(manifest.get("role") == "development"
            and tuple(manifest.get("participants", ())) == DEVELOPMENT_PARTICIPANTS,
            "exact original 30-person development cohort required")
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    started, records = time.monotonic(), []
    for person in DEVELOPMENT_PARTICIPANTS:
        budget.check()
        try:
            record = check_participant(**{key: release[key] for key in
                ("development_manifest", "bids_root", "frozen_channel_order", "coefficient_record")},
                participant_id=person, budget_check=budget.check)
        except ValueError as error:
            # Complete the metadata diagnosis for every original participant;
            # failures remain explicit, never silently remove a participant.
            record = {"participant_id": person, "status": "metadata_check_failed",
                "stable_bijection": False, "error": str(error), "EEG_amplitudes_read": False}
        records.append(record)
        atomic_json(directory / (person + ".json"), record)
    report = {"schema": SCHEMA, "status": "complete_30_person_repeated_identity_metadata_diagnosis",
        "participants": list(DEVELOPMENT_PARTICIPANTS), "participant_count": 30,
        "all_participants_pass": all(row["stable_bijection"] for row in records),
        "participant_results": records, "job_id": os.environ["SLURM_JOB_ID"],
        "elapsed_seconds": time.monotonic() - started, "EEG_amplitudes_read": False,
        "audit_access": False, "concept_or_false_scores_computed": False, "primary_analysis_changed": False,
        "interpretation": "Index/image consistency narrows a metadata-label explanation; it does not verify absolute display timing or establish image-signal recovery."}
    budget.check()
    atomic_json(directory / "repeated_identity_metadata.complete.json", report)
    return 0 if report["all_participants_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
