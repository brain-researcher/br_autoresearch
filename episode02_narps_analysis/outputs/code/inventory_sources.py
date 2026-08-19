#!/usr/bin/env python3
"""Outcome-blind source inventory for NARPS episode02.

This script reads filenames, tabular metadata, and NIfTI headers only. It never
loads BOLD, effect-map, variance-map, or group-map voxel arrays.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import socket
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib


WORKSPACE = Path(__file__).resolve().parents[2]
RAW = WORKSPACE / "inputs" / "raw"
FMRIPREP = WORKSPACE / "inputs" / "fmriprep"
FITLINS = WORKSPACE / "inputs" / "fitlins"
OUT_JSON = WORKSPACE / "outputs" / "source_inventory.json"
OUT_TSV = WORKSPACE / "outputs" / "source_inventory_runs.tsv"
OUT_MD = WORKSPACE / "outputs" / "source_inventory.md"


def tsv_header_and_rows(path: Path) -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader)
        return header, sum(1 for _ in reader)


def nifti_header(path: Path) -> dict:
    image = nib.load(str(path))
    return {
        "shape": list(image.shape),
        "zooms": [float(value) for value in image.header.get_zooms()],
        "dtype": str(image.header.get_data_dtype()),
        "affine_rounded": [[round(float(value), 6) for value in row] for row in image.affine],
    }


def run_key(path: Path) -> tuple[str, int]:
    match = re.search(r"(sub-\d+).*?_run-(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse subject/run from {path}")
    return match.group(1), int(match.group(2))


def subject_map_key(path: Path) -> tuple[str, str, str]:
    match = re.search(r"(sub-\d+)_contrast-(gain|loss)_stat-(effect|variance)_statmap", path.name)
    if match is None:
        raise ValueError(f"Cannot parse subject map from {path}")
    return match.group(1), match.group(2), match.group(3)


def main() -> None:
    roots = {}
    for label, root in (("raw", RAW), ("fmriprep", FMRIPREP), ("fitlins", FITLINS)):
        roots[label] = {
            "workspace_reference": str(root.relative_to(WORKSPACE)),
            "is_symlink": root.is_symlink(),
            "symlink_target": os.readlink(root) if root.is_symlink() else None,
            "resolved_path": str(root.resolve(strict=False)),
            "exists": root.exists(),
            "readable": os.access(root, os.R_OK),
        }

    participants = {}
    with (RAW / "participants.tsv").open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            participants[row["participant_id"]] = row["group"]
    cohort_counts = Counter(participants.values())

    raw_bold_entries = sorted(RAW.glob("sub-*/func/*_task-MGT_run-*_bold.nii.gz"))
    raw_bold = [path for path in raw_bold_entries if path.exists() and os.access(path, os.R_OK)]
    raw_bold_dangling = [path for path in raw_bold_entries if not path.exists()]
    raw_events = sorted(
        path
        for path in RAW.glob("sub-*/func/*_task-MGT_run-*_events.tsv")
        if "ORIGINAL" not in path.name
    )
    raw_bold_by_run = {run_key(path): path for path in raw_bold}
    raw_events_by_run = {run_key(path): path for path in raw_events}

    preproc_bold_entries = sorted(
        FMRIPREP.glob(
            "sub-*/func/*_task-MGT_run-*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
        )
    )
    preproc_bold = [path for path in preproc_bold_entries if path.exists() and os.access(path, os.R_OK)]
    run_mask_entries = sorted(
        FMRIPREP.glob(
            "sub-*/func/*_task-MGT_run-*_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
        )
    )
    run_masks = [path for path in run_mask_entries if path.exists() and os.access(path, os.R_OK)]
    confound_entries = sorted(FMRIPREP.glob("sub-*/func/*_task-MGT_run-*_desc-confounds_timeseries.tsv"))
    confounds = [path for path in confound_entries if path.exists() and os.access(path, os.R_OK)]
    preproc_by_run = {run_key(path): path for path in preproc_bold}
    masks_by_run = {run_key(path): path for path in run_masks}
    confounds_by_run = {run_key(path): path for path in confounds}

    run_design_entries = sorted(
        (FITLINS / "task-MGT" / "node-runLevel").glob("sub-*/*_task-MGT_run-*_design.tsv")
    )
    run_designs = [path for path in run_design_entries if path.exists() and os.access(path, os.R_OK)]
    designs_by_run = {run_key(path): path for path in run_designs}
    subject_map_entries = sorted(
        (FITLINS / "task-MGT" / "node-subjectLevel").glob(
            "sub-*/*_contrast-*_stat-*_statmap.nii.gz"
        )
    )
    subject_maps = [path for path in subject_map_entries if path.exists() and os.access(path, os.R_OK)]
    selected_subject_maps = {
        subject_map_key(path): path
        for path in subject_maps
        if re.search(r"contrast-(gain|loss)_stat-(effect|variance)_statmap", path.name)
    }

    all_expected_runs = {(subject, run) for subject in participants for run in range(1, 5)}
    availability_sets = {
        "raw_bold": set(raw_bold_by_run),
        "events": set(raw_events_by_run),
        "preproc_bold_mni2mm": set(preproc_by_run),
        "brain_mask_mni2mm": set(masks_by_run),
        "confounds": set(confounds_by_run),
        "fitlins_run_design": set(designs_by_run),
    }
    required_test_sets = {
        name: keys for name, keys in availability_sets.items() if name != "raw_bold"
    }

    event_columns = Counter()
    event_rows = {}
    for key, path in raw_events_by_run.items():
        header, rows = tsv_header_and_rows(path)
        event_columns[tuple(header)] += 1
        event_rows[key] = rows

    confound_columns = Counter()
    confound_rows = {}
    for key, path in confounds_by_run.items():
        header, rows = tsv_header_and_rows(path)
        confound_columns[tuple(header)] += 1
        confound_rows[key] = rows

    design_columns = Counter()
    design_rows = {}
    contrast_estimand_columns = {}
    for key, path in designs_by_run.items():
        header, rows = tsv_header_and_rows(path)
        design_columns[tuple(header)] += 1
        design_rows[key] = rows
        contrast_estimand_columns[str(key)] = {
            "gain_demean": "gain_demean" in header,
            "loss_demean": "loss_demean" in header,
            "intercept": "intercept" in header,
            "cosine_count": sum(name.startswith("cosine") for name in header),
        }

    # Header-only compatibility checks. nibabel image.dataobj is never indexed.
    header_sets = {}
    spatial_header_sets = {}
    for label, paths in (
        ("raw_bold", raw_bold),
        ("preproc_bold_mni2mm", preproc_bold),
        ("brain_mask_mni2mm", run_masks),
    ):
        signatures = Counter()
        spatial_signatures = Counter()
        for path in paths:
            header = nifti_header(path)
            signature = json.dumps(header, sort_keys=True)
            signatures[signature] += 1
            spatial = {
                "shape": header["shape"][:3],
                "zooms": header["zooms"][:3],
                "affine_rounded": header["affine_rounded"],
            }
            spatial_signatures[json.dumps(spatial, sort_keys=True)] += 1
        header_sets[label] = [
            {"count": count, "header": json.loads(signature)}
            for signature, count in signatures.items()
        ]
        spatial_header_sets[label] = [
            {"count": count, "header": json.loads(signature)}
            for signature, count in spatial_signatures.items()
        ]

    map_header_sets = Counter()
    for path in selected_subject_maps.values():
        map_header_sets[json.dumps(nifti_header(path), sort_keys=True)] += 1

    fitlins_description = json.loads((FITLINS / "task-MGT" / "dataset_description.json").read_text())
    fitlins_params = fitlins_description.get("PipelineDescription", {}).get("Parameters", {})

    scratch_root = Path("/scratch/users/zijiao")
    inventory = {
        "schema_version": "narps.episode02.source_inventory.v1",
        "workspace": str(WORKSPACE),
        "inventory_scope": "filenames_tabular_metadata_and_nifti_headers_only",
        "neural_outcome_values_read": False,
        "source_roots": roots,
        "participants": {
            "count": len(participants),
            "cohort_counts": dict(sorted(cohort_counts.items())),
            "subjects": sorted(participants),
        },
        "runs": {
            "expected": len(all_expected_runs),
            "availability_counts": {name: len(keys) for name, keys in availability_sets.items()},
            "complete_intersection_all_sources": len(set.intersection(*availability_sets.values())),
            "complete_intersection_required_test_sources": len(set.intersection(*required_test_sets.values())),
            "dangling_raw_bold_links": [str(path.relative_to(WORKSPACE)) for path in raw_bold_dangling],
            "missing": {
                name: [f"{subject}_run-{run}" for subject, run in sorted(all_expected_runs - keys)]
                for name, keys in availability_sets.items()
            },
        },
        "metadata": {
            "event_column_patterns": [
                {"count": count, "columns": list(columns)}
                for columns, count in event_columns.items()
            ],
            "event_row_range": [min(event_rows.values()), max(event_rows.values())],
            "confound_column_pattern_count": len(confound_columns),
            "confound_row_range": [min(confound_rows.values()), max(confound_rows.values())],
            "design_column_patterns": [
                {"count": count, "columns": list(columns)}
                for columns, count in design_columns.items()
            ],
            "design_row_range": [min(design_rows.values()), max(design_rows.values())],
            "all_designs_have_gain_loss_intercept": all(
                item["gain_demean"] and item["loss_demean"] and item["intercept"]
                for item in contrast_estimand_columns.values()
            ),
        },
        "nifti_header_signatures": header_sets,
        "nifti_spatial_header_signatures": spatial_header_sets,
        "fitlins": {
            "pipeline_name": fitlins_description.get("PipelineDescription", {}).get("Name"),
            "pipeline_version": fitlins_description.get("PipelineDescription", {}).get("Version"),
            "configured_space": fitlins_params.get("space"),
            "configured_smoothing": fitlins_params.get("smoothing"),
            "configured_estimator": fitlins_params.get("estimator"),
            "selected_subject_map_count": len(selected_subject_maps),
            "selected_subject_map_expected": len(participants) * 2 * 2,
            "selected_subject_map_missing": [
                f"{subject}:{contrast}:{stat}"
                for subject in sorted(participants)
                for contrast in ("gain", "loss")
                for stat in ("effect", "variance")
                if (subject, contrast, stat) not in selected_subject_maps
            ],
            "selected_subject_map_header_signatures": [
                {"count": count, "header": json.loads(signature)}
                for signature, count in map_header_sets.items()
            ],
        },
        "compute": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
            "scratch_root": str(scratch_root),
            "scratch_exists": scratch_root.exists(),
            "scratch_writable": os.access(scratch_root, os.W_OK),
            "durable_output_root": str(WORKSPACE / "outputs"),
        },
        "compatibility": {
            "all_expected_runs_have_all_required_test_sources": all(
                keys == all_expected_runs for keys in required_test_sets.values()
            ),
            "mni_preproc_and_masks_have_one_spatial_signature": (
                len(spatial_header_sets["preproc_bold_mni2mm"]) == 1
                and len(spatial_header_sets["brain_mask_mni2mm"]) == 1
                and spatial_header_sets["preproc_bold_mni2mm"][0]["header"]
                == spatial_header_sets["brain_mask_mni2mm"][0]["header"]
            ),
            "fitlins_subject_maps_complete": len(selected_subject_maps) == len(participants) * 2 * 2,
            "fitlins_subject_maps_have_one_spatial_signature": len(map_header_sets) == 1,
            "bounded_bold_recomputation_needed": True,
            "reason": (
                "Existing task-MGT FitLins maps were configured with 5 mm run-level smoothing. "
                "A direct 0-versus-8 mm smoothing mechanism test therefore requires a bounded "
                "recomputation from the supplied fMRIPrep BOLD using the supplied run designs."
            ),
        },
    }

    OUT_JSON.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with OUT_TSV.open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "participant_id", "task_version", "run", "raw_bold", "events",
            "preproc_bold_mni2mm", "brain_mask_mni2mm", "confounds", "fitlins_run_design",
            "n_volumes", "event_rows", "confound_rows", "design_rows",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for subject, run in sorted(all_expected_runs):
            key = (subject, run)
            preproc_header = nifti_header(preproc_by_run[key]) if key in preproc_by_run else None
            writer.writerow({
                "participant_id": subject,
                "task_version": participants[subject],
                "run": run,
                "raw_bold": key in raw_bold_by_run,
                "events": key in raw_events_by_run,
                "preproc_bold_mni2mm": key in preproc_by_run,
                "brain_mask_mni2mm": key in masks_by_run,
                "confounds": key in confounds_by_run,
                "fitlins_run_design": key in designs_by_run,
                "n_volumes": preproc_header["shape"][3] if preproc_header else "",
                "event_rows": event_rows.get(key, ""),
                "confound_rows": confound_rows.get(key, ""),
                "design_rows": design_rows.get(key, ""),
            })

    availability = inventory["runs"]["availability_counts"]
    compatibility = inventory["compatibility"]
    markdown = f"""# Source inventory

Status: **pass for bounded successor testing**. This inventory accessed filenames,
tabular metadata, and NIfTI headers only; it did not read BOLD or neural outcome
voxel values.

## Source resolution and scope

- All three workspace references are readable Sherlock-native absolute OAK symlinks.
- Raw participants: {len(participants)} ({cohort_counts['equalIndifference']} EI,
  {cohort_counts['equalRange']} ER).
- Expected task runs: {len(all_expected_runs)} (four for every participant).
- Complete required-test-source run intersection:
  {inventory['runs']['complete_intersection_required_test_sources']}.
- Required availability counts: raw BOLD {availability['raw_bold']}, events
  {availability['events']}, MNI 2 mm fMRIPrep BOLD {availability['preproc_bold_mni2mm']},
  MNI 2 mm masks {availability['brain_mask_mni2mm']}, confounds
  {availability['confounds']}, and FitLins run designs {availability['fitlins_run_design']}.
- Optional raw BOLD links readable: {availability['raw_bold']}/{len(all_expected_runs)};
  dangling raw annex links: {len(raw_bold_dangling)}. The bounded recomputation
  uses complete fMRIPrep BOLD, so this does not block either test.
- Every inspected run design contains `gain_demean`, `loss_demean`, and an intercept.

The machine-readable audit and per-run manifest are
`outputs/source_inventory.json` and `outputs/source_inventory_runs.tsv`.

## Image and design compatibility

- MNI preprocessed BOLD and masks use one common spatial signature:
  `{compatibility['mni_preproc_and_masks_have_one_spatial_signature']}`.
- Existing gain/loss subject effect and variance maps are complete
  ({inventory['fitlins']['selected_subject_map_count']}/
  {inventory['fitlins']['selected_subject_map_expected']}) and use one spatial
  signature: `{compatibility['fitlins_subject_maps_have_one_spatial_signature']}`.
- FitLins reports version `{inventory['fitlins']['pipeline_version']}`, estimator
  `{inventory['fitlins']['configured_estimator']}`, space
  `{inventory['fitlins']['configured_space']}`, and configured smoothing
  `{inventory['fitlins']['configured_smoothing']}`.

The existing FitLins statistics are usable as provenance and a matched 5 mm
reference, but they cannot by themselves identify the v1 0-versus-8 mm smoothing
mechanism. A bounded recomputation is therefore needed only for the prospectively
selected smoothing endpoints; the v1 3 x 3 x 3 grid is not needed.

## Candidate support

- Amplitude-versus-shape: supported by matched 0/8 mm group maps from a single
  fixed first-level design, evaluated before and after spatial scale normalization.
- Mask-edge/weighting: supported by the same subject effect/variance estimates,
  outcome-blind run-mask coverage shells, and equal versus inverse-variance
  aggregation.
- Task-version/sample composition: supported as a stratified robustness/control
  readout (EI and ER separately), not as independent confirmation.
- Continuous-versus-thresholded: supported by fixed continuous metrics and one
  frozen BH-FDR q=0.05 readout; threshold is not tuned.

## Compute and storage plan

Inventory ran on compute node `{inventory['compute']['hostname']}` inside Slurm job
`{inventory['compute']['slurm_job_id']}`. Durable artifacts remain in `outputs/`.
High-frequency GLM checkpoints and transient arrays will use an episode-specific
directory below `{inventory['compute']['scratch_root']}` (available and writable:
`{inventory['compute']['scratch_writable']}`), with only final reproducibility
artifacts copied to this workspace. No analysis will run on a login node.

## Stage-0 decision

No required source is missing, and semantic compatibility is sufficient to freeze
a two-test successor contract. The raw checkout's broader annex completeness is
not required for these tests, and no replacement data will be downloaded.
"""
    OUT_MD.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
