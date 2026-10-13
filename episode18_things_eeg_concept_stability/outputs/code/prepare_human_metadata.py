#!/usr/bin/env python3
"""Join the frozen EP18 concept inventory to public aggregate human metadata.

No EEG, image pixels, live sibling outputs, subject responses or model scores
are consumed. Aggregate metadata values are retained without fitted transforms.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import socket
import time


def table(path, required):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = list(reader.fieldnames or [])
        if not set(required).issubset(fields):
            raise ValueError(f"Required source columns missing in {path.name}")
        rows = {}
        for row in reader:
            key = row["uniqueID"]
            if not key or key in rows:
                raise ValueError(f"Blank or duplicate uniqueID in {path.name}")
            rows[key] = {field: row[field] for field in required}
    return fields, rows


def number(value):
    if value.strip().lower() in {"", "na", "n/a", "nan", "none", "null"}:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite value is not an explicit missing-value token")
    return result


def profile(values):
    present = [value for value in values if value is not None]
    return {"present": len(present), "missing": len(values) - len(present),
            "minimum": min(present) if present else None,
            "maximum": max(present) if present else None,
            "source_mean": sum(present) / len(present) if present else None,
            "distinct_observed_values": len(set(present))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-record", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned compute allocation required")
    started = time.monotonic()
    frozen = json.loads(args.frozen_record.read_text())
    recipe = json.loads(args.recipe.read_text())
    if not frozen["roles_frozen"] or not frozen["join_frozen"]:
        raise ValueError("Use the completed outcome-blind metadata freeze")
    # The freeze already proved all 46 share the same presented image/concept
    # inventory. Read only the first complete development person's metadata.
    join = args.frozen_record.parent / "development_event_image_join.tsv.gz"
    concepts, counts = {}, Counter()
    first_person, events = None, 0
    with gzip.open(join, "rt", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if first_person is None:
                first_person = row["participant"]
            if row["participant"] != first_person:
                break
            events += 1
            if row["is_repeated"] == "0":
                key = row["canonical_image"].split("/", 1)[0]
                if concepts.setdefault(key, row["concept_number"]) != row["concept_number"]:
                    raise ValueError("Frozen main concept identity is inconsistent")
                counts[key] += 1
    if first_person not in frozen["roles"]["development"] or events != 24648 or len(concepts) != 1854 or set(counts.values()) != {12}:
        raise ValueError("Frozen complete main concept inventory is not established")
    payload = args.metadata_root / "payload"
    _, names = table(payload / recipe["identity_table"], ["uniqueID", "Word"])
    property_columns = recipe["property_columns"]
    _, properties = table(payload / recipe["property_table"], ["uniqueID", "WordContext", *property_columns])
    category_path = payload / recipe["category_table"]
    with category_path.open(newline="", encoding="utf-8-sig") as stream:
        category_fields = next(csv.reader(stream, delimiter="\t"))
    categories = [field for field in category_fields if field not in {"uniqueID", "Word"}]
    if len(categories) != 53 or len(set(categories)) != 53:
        raise ValueError("Selected category taxonomy is not the 53-column provider membership table")
    _, membership = table(category_path, ["uniqueID", *categories])
    support = {name: {"source_concepts": len(rows), "missing_frozen_concepts": sorted(set(concepts) - set(rows)),
                      "extra_source_concepts": sorted(set(rows) - set(concepts))}
               for name, rows in (("identity", names), ("properties", properties), ("categories", membership))}
    if any(value["missing_frozen_concepts"] for value in support.values()):
        raise ValueError("A frozen concept lacks a required source-metadata row; no exclusion or imputation is allowed")
    property_values, category_values = {}, {}
    for key in sorted(concepts):
        property_values[key] = {column: number(properties[key][column]) for column in property_columns}
        category_values[key] = {category: number(membership[key][category]) for category in categories}
        if any(value is not None and value not in (0., 1.) for value in category_values[key].values()):
            raise ValueError("Broad-category membership has a nonbinary observed source value")
    missing = [{"uniqueID": key, "family": family, "field": field}
               for family, values in (("properties", property_values), ("categories", category_values))
               for key, fields in values.items() for field, value in fields.items() if value is None]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "concept_crosswalk.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["uniqueID", "frozen_concept_number", "provider_Word", "provider_WordContext"])
        for key in sorted(concepts):
            writer.writerow([key, concepts[key], names[key]["Word"], properties[key]["WordContext"]])
    with (args.output_dir / "human_features_untransformed.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["uniqueID", *property_columns, *("category53::" + name for name in categories)])
        for key in sorted(concepts):
            values = [*property_values[key].values(), *category_values[key].values()]
            writer.writerow([key, *("" if value is None else format(value, ".17g") for value in values)])
    report = {"analysis_id": "feature_preparation", "phase": "human_metadata_crosswalk",
              "observed_utc": datetime.now(timezone.utc).isoformat(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "hostname": socket.gethostname(), "elapsed_seconds": time.monotonic() - started,
              "frozen_record": str(args.frozen_record), "recipe": recipe, "metadata_root": str(args.metadata_root),
              "frozen_main_concepts": len(concepts), "source_support": support,
              "property_columns": property_columns, "category_columns": categories,
              "property_profiles": {column: profile([property_values[key][column] for key in concepts]) for column in property_columns},
              "category_profiles": {category: profile([category_values[key][category] for key in concepts]) for category in categories},
              "missing_value_locations": missing,
              "concepts_with_no_recorded_category_membership": [key for key in concepts if all(value == 0 for value in category_values[key].values())],
              "multiple_category_memberships_retained": True,
              "join_key_is_not_a_predictor": True, "one_hot_concept_features_created": False,
              "imputation_applied": False, "scaling_or_PCA_applied": False,
              "complete_feature_panel_frozen": False, "concept_embedding_recipe_selected": False,
              "EEG_image_checkpoint_payload_values_opened": False,
              "audit_seal_verified": False, "EEG_candidate_comparison_authorized": False,
              "status": "source_crosswalk_complete_missing_values_require_prospective_rule" if missing else "source_crosswalk_complete_no_missing_values"}
    (args.output_dir / "human_metadata_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("analysis_id", "phase", "status", "frozen_main_concepts", "elapsed_seconds", "audit_seal_verified")}), flush=True)


if __name__ == "__main__":
    main()
