#!/usr/bin/env python3
"""Join the surviving design metadata to the declared spatial source support.

No image, tensor, operator array, training score or map result is opened. The
upstream confirmatory flags do not become EP03 roles. Output contrast names and
weights are source definitions, not an inferred task taxonomy or model input.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from qualify_source_metadata import require_compute, subject_token


def normalized_key(dataset, subject, session, task, run):
    return (str(dataset), subject_token(subject), subject_token(session), str(task), subject_token(run))


def spatial_key(key):
    if len(key) != 6:
        raise ValueError("Unrecognized declared source acquisition key")
    if key[5] not in [None, ""]:
        raise ValueError("Additional acquisition key requires explicit matching; do not discard it")
    return normalized_key(*key[:5])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-sidecars", type=Path, required=True)
    parser.add_argument("--support-universe", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.output_dir.exists():
        raise FileExistsError("Preserve attempts; choose a fresh output directory")
    receipt = json.loads(args.design_sidecars.read_text())
    support = json.loads(args.support_universe.read_text())
    source_keys = set()
    target_keys = set()
    source_identifiers = defaultdict(set)
    for sample in support["samples"]:
        source_key = spatial_key(sample["source_acquisition_key"])
        source_keys.add(source_key)
        source_identifiers[source_key].add(sample["bold_artifact_id"])
        target_keys.add(spatial_key(sample["target_acquisition_key"]))
    acquisitions = receipt["content"]["acquisitions"]
    by_key = defaultdict(list)
    for row in acquisitions:
        key = row["key"]
        by_key[normalized_key(row["dataset"], key["subject"], key.get("session", ""),
                              key["task"], key.get("run", ""))].append(row)
    matched_rows = {key: by_key[key][0] for key in source_keys if len(by_key.get(key, ())) == 1}
    identity_repairs = []
    # The observed ds004656 discrepancy is literal am/pm versus AM/PM.
    # Permit this diagnosed alias only when the untouched physical artifact
    # suffix and declared BOLD stem prove the same acquisition uniquely.
    for source_key in sorted(source_keys - by_key.keys()):
        if source_key[0] != "ds004656" or source_key[2] not in {"am", "pm"}:
            continue
        candidates = []
        for design_key, rows in by_key.items():
            if (design_key[:2] + design_key[3:] != source_key[:2] + source_key[3:]
                    or design_key[2].lower() != source_key[2] or len(rows) != 1):
                continue
            row = rows[0]
            artifacts = source_identifiers[source_key]
            data_path = row.get("data", {}).get("path", "")
            if artifacts and all(data_path.endswith("/" + artifact)
                    and Path(artifact).parent.name == row.get("physical_bold_stem")
                    for artifact in artifacts):
                candidates.append((design_key, row))
        if len(candidates) == 1:
            design_key, row = candidates[0]
            matched_rows[source_key] = row
            identity_repairs.append({"source_key": list(source_key), "design_key": list(design_key),
                                     "evidence": "exact_unchanged_artifact_suffix_and_physical_BOLD_stem",
                                     "bold_artifact_ids": sorted(source_identifiers[source_key])})
    matched_keys = set(matched_rows)
    definitions = {}
    definition_variants = defaultdict(set)
    dataset_tasks = defaultdict(set)
    for key in sorted(matched_keys):
        row = matched_rows[key]
        dataset_tasks[key[0]].add(key[3])
        for query in row["queries"]:
            condition = query["condition"]
            name = (key[0], condition["task"], condition["contrast"])
            signature = json.dumps({"conditions": condition["conditions"],
                                    "weights": condition["weights"]}, sort_keys=True)
            definition_variants[name].add(signature)
            entry = definitions.setdefault((name, signature), {
                "dataset_id": key[0], "task": condition["task"],
                "contrast": condition["contrast"], "conditions": condition["conditions"],
                "weights": condition["weights"], "acquisition_support": 0,
                "source_estimable_acquisitions": 0, "definition_source_paths": set(),
                "upstream_role_not_EP03_audit": "historically_exposed_development_lineage"})
            entry["acquisition_support"] += 1
            entry["source_estimable_acquisitions"] += query.get("estimable") is True
            if query.get("definition_source_path"):
                entry["definition_source_paths"].add(query["definition_source_path"])
    # Source existence is a metadata observation, not proof of same contents.
    # Test only exact named definition paths and their documented FitLins layout.
    definition_paths = {}
    for entry in definitions.values():
        entry["definition_source_paths"] = sorted(entry["definition_source_paths"])
        for source in entry["definition_source_paths"]:
            if source in definition_paths:
                continue
            original = Path(source)
            relocated = (args.analysis_root/entry["dataset_id"]/"code/openneuro_glmfitlins/statsmodel_specs"
                         /entry["dataset_id"]/original.name)
            definition_paths[source] = {"recorded_path_exists": original.is_file(),
                                       "candidate_current_path": str(relocated),
                                       "candidate_current_path_exists": relocated.is_file(),
                                       "contents_compared": False}
    all_tasks = {(key[0], key[3]) for key in by_key}
    spatial_tasks = {(key[0], key[3]) for key in source_keys}
    missing_keys = source_keys-by_key.keys()-matched_keys
    missing_tasks = {(key[0], key[3]) for key in missing_keys}
    report = {"analysis_id": "source_handoff_qualification",
        "status": "design_spatial_metadata_reconciled_group_map_handoff_still_unqualified",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "design_sidecars_path": str(args.design_sidecars),
        "support_universe_path": str(args.support_universe),
        "producer": receipt.get("producer"), "spec_id": receipt.get("spec_id"),
        "upstream_bindings": receipt.get("bindings"),
        "upstream_summary": receipt["content"].get("summary"),
        "design_acquisition_records": len(acquisitions),
        "design_distinct_accessions": len({key[0] for key in by_key}),
        "design_dataset_task_pairs": len(all_tasks),
        "spatial_acquisition_keys": len(source_keys), "spatial_dataset_task_pairs": len(spatial_tasks),
        "source_and_target_acquisition_support_equal": source_keys == target_keys,
        "uniquely_matched_acquisitions": len(matched_keys),
        "physical_identity_supported_key_repairs": identity_repairs,
        "missing_design_keys": [list(key) for key in sorted(missing_keys)],
        "unmatched_source_identifiers": [{"key": list(key), "bold_artifact_ids": sorted(source_identifiers[key])}
                                         for key in sorted(missing_keys)],
        "same_task_design_identifiers_for_unmatched_keys": [
            {"key": list(key), "physical_bold_stem": row.get("physical_bold_stem"),
             "declared_data_path": row.get("data", {}).get("path")}
            for key, acquisitions_for_key in sorted(by_key.items()) if (key[0], key[3]) in missing_tasks
            for row in acquisitions_for_key],
        "ambiguous_design_keys": [{"key": list(key), "records": len(by_key[key])}
                                  for key in sorted(source_keys) if len(by_key.get(key, ())) > 1],
        "design_only_dataset_task_pairs": [list(key) for key in sorted(all_tasks-spatial_tasks)],
        "supported_source_contrast_names": len(definition_variants),
        "supported_source_contrast_definitions": len(definitions),
        "conflicting_definition_names": [{"key": list(name), "variants": len(variants)}
                                         for name, variants in sorted(definition_variants.items()) if len(variants) > 1],
        "definition_path_status": definition_paths,
        "source_condition_and_weight_definitions": list(definitions.values()),
        "eligible_group_Z_map_count": None, "authenticated_biological_group_count": None,
        "map_or_tensor_payload_opened": False, "historical_scores_opened": False,
        "candidate_handoff_qualified": False, "audit_opened": False}
    args.output_dir.mkdir(parents=True)
    (args.output_dir/"design_support_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ["status", "design_acquisition_records",
        "design_distinct_accessions", "design_dataset_task_pairs", "spatial_acquisition_keys",
        "spatial_dataset_task_pairs", "uniquely_matched_acquisitions", "missing_design_keys",
        "ambiguous_design_keys", "supported_source_contrast_names", "supported_source_contrast_definitions",
        "conflicting_definition_names", "design_only_dataset_task_pairs"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
