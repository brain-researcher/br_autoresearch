#!/usr/bin/env python3
"""Build the frozen outcome-blind common, core, and periphery masks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


WORKSPACE = Path(__file__).resolve().parents[2]
FMRIPREP = WORKSPACE / "inputs" / "fmriprep"
PARTICIPANTS = WORKSPACE / "inputs" / "raw" / "participants.tsv"
OUT = WORKSPACE / "outputs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    groups = {}
    with PARTICIPANTS.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            groups[row["participant_id"]] = row["group"]

    mask_paths = sorted(
        FMRIPREP.glob(
            "sub-*/func/*_task-MGT_run-*_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
        )
    )
    if len(mask_paths) != 432 or any(not path.exists() for path in mask_paths):
        raise RuntimeError(f"Expected 432 readable masks, found {sum(path.exists() for path in mask_paths)}")

    reference = nib.load(str(mask_paths[0]))
    shape = reference.shape
    affine = reference.affine
    total = np.zeros(shape, dtype=np.uint16)
    by_group = {
        "equalIndifference": np.zeros(shape, dtype=np.uint16),
        "equalRange": np.zeros(shape, dtype=np.uint16),
    }
    run_sizes = []
    group_run_counts = {key: 0 for key in by_group}

    for path in mask_paths:
        subject = path.name.split("_", 1)[0]
        group = groups[subject]
        image = nib.load(str(path))
        if image.shape != shape or not np.allclose(image.affine, affine, atol=1e-6, rtol=0):
            raise RuntimeError(f"Mask grid mismatch: {path}")
        data = np.asanyarray(image.dataobj) > 0
        total += data
        by_group[group] += data
        run_sizes.append(int(data.sum()))
        group_run_counts[group] += 1

    overall_required = math.ceil(0.95 * len(mask_paths))
    group_required = {
        group: math.ceil(0.95 * count) for group, count in group_run_counts.items()
    }
    common = total >= overall_required
    for group in by_group:
        common &= by_group[group] >= group_required[group]

    structure = ndimage.generate_binary_structure(3, 1)
    core = ndimage.binary_erosion(common, structure=structure, iterations=1, border_value=0)
    periphery = common & ~core

    median_run_size = float(np.median(run_sizes))
    counts = {
        "common": int(common.sum()),
        "core": int(core.sum()),
        "periphery": int(periphery.sum()),
        "median_run_mask": median_run_size,
    }
    if counts["common"] == 0 or counts["common"] < 0.5 * median_run_size:
        raise RuntimeError(f"Invalid common mask: {counts}")
    if counts["core"] < 0.5 * counts["common"]:
        raise RuntimeError(f"Core mask below 50% of common: {counts}")
    if counts["periphery"] < 0.01 * counts["common"]:
        raise RuntimeError(f"Periphery below 1% of common: {counts}")
    if np.any(core & ~common):
        raise RuntimeError("Core is not a subset of common")

    paths = {}
    for label, data in (("common", common), ("core", core), ("periphery", periphery)):
        path = OUT / f"{label}_analysis_mask.nii.gz"
        nib.save(nib.Nifti1Image(data.astype(np.uint8), affine, reference.header), str(path))
        paths[label] = path

    summary = {
        "schema_version": "narps.episode02.mask_summary.v1",
        "outcome_blind": True,
        "reference_mask": str(mask_paths[0].relative_to(WORKSPACE)),
        "shape": list(shape),
        "zooms": [float(value) for value in reference.header.get_zooms()[:3]],
        "run_mask_count": len(mask_paths),
        "group_run_counts": group_run_counts,
        "coverage_thresholds": {
            "overall": overall_required,
            **group_required,
        },
        "core_definition": "one six-neighbor binary erosion of the fixed common mask",
        "counts": counts,
        "files": {
            label: {
                "path": str(path.relative_to(WORKSPACE)),
                "sha256": sha256(path),
            }
            for label, path in paths.items()
        },
    }
    (OUT / "mask_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
