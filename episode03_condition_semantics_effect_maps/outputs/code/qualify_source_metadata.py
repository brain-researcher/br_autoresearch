#!/usr/bin/env python3
"""Census the known NeuroEffect spatial-lineage metadata without loading maps.

This reads only explicitly named JSON/JSONL lineage records in the preserved
source pack. BOLD arrays, transforms, map bytes, results text and model scores
are never opened. Dataset/task labels are not authenticated independent groups.
The source quarantine explicitly says this pack is not the signed-map handoff.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket


def require_compute():
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Run this metadata census in the coordinator's verified Slurm allocation")


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def subject_token(value):
    """BIDS numeric padding is not a different participant within one accession."""
    value = str(value).removeprefix("sub-")
    return str(int(value)) if value.isdecimal() else value


def census(inventory, design_mapping, aggregate, roots):
    acquisitions = inventory["acquisitions"]
    participants = defaultdict(set)
    tasks = defaultdict(set)
    task_subjects = defaultdict(set)
    keys = []
    for row in acquisitions:
        key = row["source_acquisition_key"]
        if len(key) != 6:
            raise ValueError("Unexpected acquisition-key shape; do not guess its lineage semantics")
        accession, subject, session, task, run, extra = map(str, key)
        subject = subject_token(subject)
        participants[accession].add(subject)
        tasks[accession].add(task)
        task_subjects[(accession, task)].add(subject)
        keys.append((accession, subject, session, task, run, extra))
    counts = Counter(k[0] for k in keys)
    pinned = {row["dataset_id"]: row for row in aggregate["receipts"]}
    source_roots = {row["dataset_id"]: row for row in roots["datasets"]}
    overlaps = []
    for accession in sorted(tasks):
        names = sorted(tasks[accession])
        for index, first in enumerate(names):
            for second in names[index+1:]:
                common = task_subjects[(accession, first)] & task_subjects[(accession, second)]
                if common:
                    overlaps.append({"dataset_id": accession, "first_task": first,
                                     "second_task": second, "shared_participant_labels": len(common)})
    mapping_keys = Counter(tuple(map(str, row["source_acquisition_key"])) for row in design_mapping)
    inventory_keys = Counter(tuple(map(str, row["source_acquisition_key"])) for row in acquisitions)
    dataset_rows = [{"dataset_id": accession, "acquisitions": counts[accession],
                     "participant_labels_within_accession": len(participants[accession]),
                     "task_count": len(tasks[accession]), "tasks": sorted(tasks[accession]),
                     "preserved_public_repository_commit": pinned.get(accession, {}).get("pinned_commit"),
                     "preserved_payload_files": pinned.get(accession, {}).get("path_count", 0),
                     "source_roots_recorded": accession in source_roots}
                    for accession in sorted(tasks)]
    return {"source_acquisition_rows": len(acquisitions),
            "reported_acquisition_count": inventory.get("acquisition_count"),
            "unique_acquisition_keys": len(set(keys)),
            "duplicate_acquisition_keys": len(keys)-len(set(keys)),
            "distinct_openneuro_accessions": len(tasks),
            "dataset_task_pairs": len(task_subjects),
            "participant_labels_summed_within_accession": sum(map(len, participants.values())),
            "within_accession_task_pairs_sharing_participant_labels": overlaps,
            "design_mapping_rows": len(design_mapping),
            "design_mapping_exact_key_multiset_matches_inventory": mapping_keys == inventory_keys,
            "preserved_pinned_repositories": len(pinned),
            "preserved_selected_payload_files": aggregate.get("path_count"),
            "historical_byte_verification_status": aggregate.get("status"),
            "historical_pinned_heads_exact": aggregate.get("all_pinned_heads_exact"),
            "datasets": dataset_rows,
            "authenticated_independent_biological_groups": None,
            "eligible_signed_group_Z_maps": None,
            "family_support": None,
            "role": "historically_exposed_spatial_lineage_only",
            "candidate_handoff_qualified": False,
            "audit_available": False,
            "unknowns": ["historical_51_group_identity_and_46_task_group_discrepancy",
                         "cross_accession_participant_and_derivative_overlap",
                         "signed_unthresholded_group_Z_map_manifest_and_observability",
                         "methods_only_family_and_canonical_polarity_taxonomy",
                         "licenses_release_ids_common_space_mask_and_repeatability",
                         "prospective_audit_not_present"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh attempt; preserve prior metadata evidence")
    paths = {"inventory": args.source_root/"lineage/source_inventory.json",
             "design_mapping": args.source_root/"source/full51_runtime_design_mapping.jsonl",
             "aggregate": args.source_root/"source/aggregate_validation.json",
             "roots": args.source_root/"source/source_roots.json"}
    inventory = json.loads(paths["inventory"].read_text())
    report = census(inventory, read_jsonl(paths["design_mapping"]),
                    json.loads(paths["aggregate"].read_text()), json.loads(paths["roots"].read_text()))
    report.update(analysis_id="source_handoff_qualification",
                  status="metadata_census_completed_signed_map_handoff_unresolved",
                  observed_utc=datetime.now(timezone.utc).isoformat(),
                  slurm_job_id=os.environ["SLURM_JOB_ID"], hostname=socket.gethostname(),
                  metadata_sources={key: str(path) for key, path in paths.items()},
                  map_or_bold_payload_opened=False)
    # Check only the three explicitly named metadata pointers. Do not walk
    # their directories, follow a live cache, or open any referenced payload.
    report["historical_metadata_pointers"] = {
        key: {"path": inventory[key], "file_exists": Path(inventory[key]).is_file()}
        for key in ["design_sidecars_path", "source_roots_path", "support_universe_path"]}
    args.output_dir.mkdir(parents=True)
    (args.output_dir/"source_metadata_report.json").write_text(json.dumps(report, indent=2)+"\n")
    with (args.output_dir/"dataset_support.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["datasets"][0]), delimiter="\t")
        writer.writeheader()
        for row in report["datasets"]:
            writer.writerow({**row, "tasks": ";".join(row["tasks"])})
    print(json.dumps({key: report[key] for key in ["status", "distinct_openneuro_accessions",
        "dataset_task_pairs", "source_acquisition_rows", "design_mapping_rows",
        "design_mapping_exact_key_multiset_matches_inventory", "candidate_handoff_qualified",
        "historical_metadata_pointers"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
