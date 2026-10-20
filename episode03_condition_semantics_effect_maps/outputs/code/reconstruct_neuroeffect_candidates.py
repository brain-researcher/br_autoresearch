#!/usr/bin/env python3
"""Reconstruct what the staged NeuroEffect metadata says about its "51 groups".

Reads the staged policy-3 subject partitions and release metadata, the stored
summary of the staged design receipt, EP03's completed metadata census, and
directory listings of the shared FitLins analyses. It counts Dataset-level
Z-map filenames only: no image, header, PNG, report or score is opened.
"""

import argparse
import json
import os
import socket
import time
from pathlib import Path

Z_SUFFIX = "_stat-z_statmap.nii.gz"


def group_z_map_names(ds_dir):
    """Dataset-level Z-map filenames per task and node; None if absent."""
    if not ds_dir.is_dir():
        return None
    out = {}
    for task in sorted(e.name for e in os.scandir(ds_dir) if e.name.startswith("task-")):
        tdir = ds_dir / task
        nodes = {}
        for node in sorted(e.name for e in os.scandir(tdir) if e.name.startswith("node-")):
            names = []
            for e in os.scandir(tdir / node):
                if e.name.startswith("sub-"):
                    continue  # run/subject outputs; never descended or statted
                if e.name.endswith(Z_SUFFIX):
                    names.append(e.name)
                elif e.name.startswith("ses-") and e.is_dir():
                    names += [f"{e.name}/{f.name}" for f in os.scandir(tdir / node / e.name)
                              if f.name.endswith(Z_SUFFIX) and not f.name.startswith("sub-")]
            if names:
                nodes[node] = len(names)
        out[task] = nodes
    return out


def stored_counts(obj, depth=0, found=None):
    """Integer summary fields of the design receipt, shallow below `content`."""
    found = {} if found is None else found
    if isinstance(obj, dict) and depth < 4:
        for key, val in obj.items():
            if isinstance(val, int) and not isinstance(val, bool):
                found[key] = val
            elif isinstance(val, dict):
                stored_counts(val, depth + 1, found)
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partitions", type=Path, required=True)
    ap.add_argument("--release-metadata", type=Path, required=True)
    ap.add_argument("--design-receipt", type=Path, required=True)
    ap.add_argument("--metadata-census", type=Path, required=True)
    ap.add_argument("--analysis-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()

    parts = json.loads(args.partitions.read_text())
    release = json.loads(args.release_metadata.read_text())
    census = json.loads(args.metadata_census.read_text())
    receipt = json.loads(args.design_receipt.read_text())
    content = receipt.get("content", {})
    receipt_summary = {k: v for k, v in stored_counts(content).items()}
    acq = content.get("acquisitions", [])
    receipt_pairs = sorted({(a["dataset"], a["key"]["task"]) for a in acq})
    del receipt, content, acq

    included = {i["dataset_id"]: i for i in parts["included"]}
    excluded = {e["dataset_id"]: e for e in parts["excluded"]}
    census_tasks = {d["dataset_id"]: d.get("tasks", []) for d in census.get("datasets", [])}

    rows, totals = [], {"present": 0, "group_z_maps": 0, "fitlins_task_routes": 0,
                        "included_group_z_maps": 0, "excluded_group_z_maps": 0}
    for ds in parts["candidate_dataset_ids"]:
        names = group_z_map_names(args.analysis_root / ds)
        n_maps = sum(sum(nodes.values()) for nodes in names.values()) if names else 0
        row = {"dataset_id": ds,
               "partition_status": "included" if ds in included else "excluded",
               "subject_count": (included.get(ds) or excluded.get(ds) or {}).get("subject_count"),
               "exclusion_reason": (excluded.get(ds) or {}).get("reason"),
               "split_counts": (included.get(ds) or {}).get("counts"),
               "spatial_census_tasks": census_tasks.get(ds),
               "fitlins_analysis_dir_present": names is not None,
               "fitlins_task_routes": sorted(names) if names else [],
               "dataset_level_z_map_filenames": n_maps,
               "z_maps_by_task_and_node": names}
        rows.append(row)
        if names is not None:
            totals["present"] += 1
            totals["fitlins_task_routes"] += len(names)
        totals["group_z_maps"] += n_maps
        totals["included_group_z_maps" if ds in included else "excluded_group_z_maps"] += n_maps

    report = {
        "analysis_id": "source_handoff_qualification",
        "check": "neuroeffect_51_candidate_provenance",
        "job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(),
        "inputs": {k: str(v) for k, v in vars(args).items() if k != "output"},
        "partitions_schema": parts.get("schema"),
        "partitions_policy": parts.get("policy"),
        "partitions_generated_from": parts.get("generated_from"),
        "release_metadata": release,
        "candidate_count": len(parts["candidate_dataset_ids"]),
        "included_count": len(included), "excluded_count": len(excluded),
        "included_subjects": sum(i["subject_count"] for i in included.values()),
        "spatial_census_dataset_task_pairs": census.get("dataset_task_pairs"),
        "design_receipt_stored_counts": receipt_summary,
        "design_receipt_dataset_task_pairs": len(receipt_pairs),
        "design_receipt_accessions": len({p[0] for p in receipt_pairs}),
        "totals": totals,
        "matches_reported_577": totals["group_z_maps"] == 577,
        "candidates": rows,
        "payloads_headers_pngs_reports_or_scores_opened": False,
        "elapsed_seconds": time.time() - t0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    brief = {k: report[k] for k in ("candidate_count", "included_count", "excluded_count",
                                     "included_subjects", "design_receipt_dataset_task_pairs",
                                     "design_receipt_accessions", "totals", "matches_reported_577")}
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
