#!/usr/bin/env python3
"""Locate dataset-level Z-map filenames through explicit matched source routes.

Reads source statistics-model JSON and dataset-description metadata only.
Never opens an image, image header, generated report, prediction or score.
Filename labels are potential source support, not authenticated map eligibility.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

from qualify_source_metadata import require_compute


SPACE_KEYS = {"SpatialReference", "ReferenceSpace", "Template", "space"}


def declared_spaces(value, location=""):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            where = f"{location}.{key}" if location else key
            if key in SPACE_KEYS and item not in (None, "", [], {}):
                found.append({"field": where, "declared_value": item})
            elif isinstance(item, (dict, list)):
                found.extend(declared_spaces(item, where))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            found.extend(declared_spaces(item, f"{location}[{i}]"))
    return found


def camel_name(name):
    return re.sub(r"_([A-Za-z])", lambda match: match.group(1).upper(), name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-support-report", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.output_dir.exists():
        raise FileExistsError("Preserve prior source attempts")
    reconciliation = json.loads(args.design_support_report.read_text())
    grouped = defaultdict(list)
    for row in reconciliation["source_condition_and_weight_definitions"]:
        grouped[(row["dataset_id"], row["task"])].append(row)
    rows = []
    all_maps = []
    for (dataset, task), definitions in sorted(grouped.items()):
        dataset_root = args.analysis_root/dataset
        task_root = dataset_root/f"task-{task}"
        spec_root = dataset_root/"code/openneuro_glmfitlins/statsmodel_specs"/dataset
        spec_paths = {spec_root/f"{dataset}-{task}_specs.json"}
        for definition in definitions:
            for original in definition["definition_source_paths"]:
                filename = Path(original).name
                if filename.endswith("_contrasts.json"):
                    spec_paths.add(spec_root/filename.replace("_contrasts.json", "_specs.json"))
                elif filename.endswith("_specs.json"):
                    spec_paths.add(spec_root/filename)
        existing_specs = sorted(path for path in spec_paths if path.is_file())
        row = {"dataset_id": dataset, "task": task, "task_directory": str(task_root),
               "task_directory_exists": task_root.is_dir(),
               "candidate_spec_paths": [str(path) for path in sorted(spec_paths)],
               "existing_spec_paths": [str(path) for path in existing_specs],
               "source_contrast_count": len(definitions), "dataset_nodes": [],
               "declared_space_evidence": [], "map_filename_candidates": [],
               "missing_filename_contrasts_observability": "unknown"}
        if len(existing_specs) != 1:
            row["source_binding"] = "missing_or_ambiguous_statistics_model"
            rows.append(row)
            continue
        spec_path = existing_specs[0]
        spec = json.loads(spec_path.read_text())
        row["source_binding"] = "one_source_statistics_model_found_contents_not_yet_frozen"
        row["declared_input"] = {key: spec.get("Input", {}).get(key)
                                 for key in ["task", "subject", "session", "run", "space"]}
        row["declared_space_evidence"].extend(
            {"metadata_path": str(spec_path), **item} for item in declared_spaces(spec))
        # Dataset descriptions may declare template/source provenance. Never
        # infer an exact template from a map name, a grid or a generic MNI label.
        for metadata_path in [dataset_root/"dataset_description.json", task_root/"dataset_description.json"]:
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text())
                row["declared_space_evidence"].extend(
                    {"metadata_path": str(metadata_path), **item} for item in declared_spaces(metadata))
        source_contrasts = {}
        for node in spec.get("Nodes", []):
            if str(node.get("Level", "")).lower() == "run":
                for contrast in node.get("Contrasts", []):
                    source_contrasts.setdefault(contrast.get("Name"), []).append(contrast)
            if str(node.get("Level", "")).lower() != "dataset":
                continue
            name = node["Name"]
            directories = sorted({task_root/f"node-{name}", task_root/f"node-{camel_name(name)}"})
            existing = [path for path in directories if path.is_dir()]
            node_record = {"name": name, "declared_level": node["Level"],
                           "group_by": node.get("GroupBy", []),
                           "model": node.get("Model"), "dummy_contrasts": node.get("DummyContrasts"),
                           "existing_directories": [str(path) for path in existing],
                           "candidate_directories": [str(path) for path in directories]}
            row["dataset_nodes"].append(node_record)
            # A declared dataset node grouped by participant or run cannot be
            # promoted as a population group effect by its directory name.
            if any(str(field).lower() in {"subject", "run"} for field in node.get("GroupBy", [])):
                node_record["population_group_effect_eligible"] = False
                node_record["reason"] = "node_retains_subject_or_run_grouping"
                continue
            node_record["population_group_effect_eligible"] = None
            for directory in existing:
                # Exact declared node, at depth zero or one session component.
                # Do not traverse reports, participant nodes, or arbitrary trees.
                map_paths = sorted(set(directory.glob("*_stat-z_statmap.nii.gz")) |
                                   set(directory.glob("ses-*/*_stat-z_statmap.nii.gz")))
                for path in map_paths:
                    match = re.search(r"(?:^|_)contrast-(.+)_stat-z_statmap\.nii\.gz$", path.name)
                    if not match:
                        continue
                    contrast_name = match.group(1)
                    if contrast_name not in {entry["contrast"] for entry in definitions}:
                        continue
                    candidate = {"dataset_id": dataset, "task": task, "contrast": contrast_name,
                                 "declared_node_level": "Dataset", "declared_node_name": name,
                                 "path": str(path), "filename_statistic_label": "z",
                                 "image_opened": False, "group_map_eligibility_authenticated": False,
                                 "canonical_polarity": None, "template_space_authenticated": False}
                    row["map_filename_candidates"].append(candidate)
                    all_maps.append(candidate)
        matches = []
        for definition in definitions:
            candidates = source_contrasts.get(definition["contrast"], [])
            same = [contrast for contrast in candidates
                    if contrast.get("ConditionList") == definition["conditions"]
                    and contrast.get("Weights") == definition["weights"]]
            matches.append({"contrast": definition["contrast"],
                            "receipt_conditions": definition["conditions"],
                            "receipt_weights": definition["weights"],
                            "model_contrast_definitions": candidates,
                            "exact_conditions_and_weights_match": len(same) == 1})
        row["contrast_definition_comparison"] = matches
        names_found = {item["contrast"] for item in row["map_filename_candidates"]}
        row["source_contrasts_without_candidate_filename"] = sorted(
            {item["contrast"] for item in definitions}-names_found)
        rows.append(row)
    report = {"analysis_id": "source_handoff_qualification",
              "status": "dataset_node_filename_census_completed_not_map_eligibility",
              "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "observed_utc": datetime.now(timezone.utc).isoformat(),
              "source_reconciliation": str(args.design_support_report),
              "explicit_dataset_task_pairs": len(grouped),
              "tasks_with_exact_source_model": sum(len(row["existing_spec_paths"]) == 1 for row in rows),
              "candidate_group_Z_filenames": len(all_maps),
              "accessions_with_candidate_group_Z_filenames": len({row["dataset_id"] for row in all_maps}),
              "task_pairs_with_candidate_group_Z_filenames": len({(row["dataset_id"], row["task"]) for row in all_maps}),
              "task_pairs_with_declared_space_metadata": sum(bool(row["declared_space_evidence"]) for row in rows),
              "tasks": rows, "images_or_headers_opened": False, "scored_artifacts_opened": False,
              "candidate_handoff_qualified": False, "audit_available": False,
              "unresolved": ["historical_51_identity_and_independence", "source_snapshot_and_license",
                             "actual_group_sample_and_derivative_lineage", "unthresholded_statistic_and_source_sign",
                             "exact_template_and_common_mask", "methods_only_family_redaction_and_canonical_polarity",
                             "pre_search_inference_and_negative_control_identity"]}
    args.output_dir.mkdir(parents=True)
    (args.output_dir/"group_map_metadata_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ["status", "explicit_dataset_task_pairs",
        "tasks_with_exact_source_model", "candidate_group_Z_filenames", "accessions_with_candidate_group_Z_filenames",
        "task_pairs_with_candidate_group_Z_filenames", "task_pairs_with_declared_space_metadata",
        "candidate_handoff_qualified"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
