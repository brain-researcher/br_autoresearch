#!/usr/bin/env python3
"""Prepare private source routing and separate concept-text metadata, without pixels.

The private routing manifest is for the trusted builder, never an image worker.
Writing it is not an access seal. This command reads only already-frozen EP18
metadata and never opens EEG, images, model weights or live sibling outputs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import socket
import uuid

from feature_extractors import require_compute


def name_prompt(row):
    context, word = row["provider_WordContext"].strip(), row["provider_Word"].strip()
    selected = context or word
    if not selected:
        raise ValueError(f"No provider name/context for {row['uniqueID']}; do not invent/drop")
    return f"a photo of {selected}.", not bool(context)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-record", required=True, type=Path)
    parser.add_argument("--human-report", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    require_compute()
    frozen = json.loads(args.frozen_record.read_text())
    human = json.loads(args.human_report.read_text())
    registry = json.loads(args.registry.read_text())
    if not frozen["roles_frozen"] or not frozen["join_frozen"] or human["frozen_main_concepts"] != 1854:
        raise ValueError("Use the qualified metadata freeze and full human crosswalk")
    if registry["record_id"] != "ep18_feature_recipe_20261003":
        raise ValueError("Unexpected feature recipe")
    if args.output_dir.exists():
        raise FileExistsError("Preserve any existing attempt and its opaque-ID mapping")

    images, first_person, events = {}, None, 0
    join = args.frozen_record.parent / "development_event_image_join.tsv.gz"
    with gzip.open(join, "rt", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if first_person is None:
                first_person = row["participant"]
            if row["participant"] != first_person:
                break
            events += 1
            canonical = row["canonical_image"]
            entry = images.setdefault(canonical, {"opaque_image_id": uuid.uuid4().hex, "canonical_image": canonical,
                                                   "main": False, "repeated": False})
            entry["repeated" if row["is_repeated"] == "1" else "main"] = True
    if events != 24648 or sum(value["main"] for value in images.values()) != 22248 or sum(value["repeated"] for value in images.values()) != 200:
        raise ValueError("Unexpected already-frozen image inventory")

    prompts, seen = [], set()
    with (args.human_report.parent / "concept_crosswalk.tsv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            key = row["uniqueID"]
            if key in seen:
                raise ValueError("Duplicate provider concept key")
            seen.add(key)
            prompt, fallback = name_prompt(row)
            prompts.append({"uniqueID": key, "name_prompt": prompt, "provider_Word_fallback": int(fallback)})
    if seen != {key.split("/", 1)[0] for key, value in images.items() if value["main"]} or len(seen) != 1854:
        raise ValueError("Name crosswalk and frozen main concepts differ")

    args.output_dir.mkdir(parents=True)
    # Different files serve different trusted roles; they do not establish OS isolation.
    for filename, fields, rows in [
        ("private_source_routing.tsv", ["opaque_image_id", "canonical_image", "main", "repeated"],
         [images[key] for key in sorted(images)]),
        ("label_name_prompts.tsv", ["uniqueID", "name_prompt", "provider_Word_fallback"], prompts),
    ]:
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    report = {"analysis_id": "feature_preparation", "phase": "private_routing_and_label_text",
              "observed_utc": datetime.now(timezone.utc).isoformat(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "hostname": socket.gethostname(), "main_image_identities": 22248, "repeated_image_identities": 200,
              "union_image_identities": len(images), "concepts": len(prompts),
              "provider_Word_fallbacks": sum(row["provider_Word_fallback"] for row in prompts),
              "opaque_ID_rule": "Random UUID4, fixed by this preserved routing output; never regenerate for an existing feature artifact",
              "private_routing_is_image_worker_input": False, "actual_worker_delivery_verified": False,
              "EEG_pixels_checkpoints_opened": False, "audit_seal": False,
              "frozen_feature_record_emitted": False, "registry": str(args.registry)}
    (args.output_dir / "feature_metadata_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
