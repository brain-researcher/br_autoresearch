#!/usr/bin/env python
"""Outcome-blind source/QC/design audit and frozen common-mask construction."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import nibabel as nib
import nilearn
import numpy as np
import pandas as pd
import scipy
from nibabel.processing import resample_from_to

from narps_common import (
    ACOMPCOR6,
    CONFOUNDS,
    CONTRACT_HASHES,
    CONTRASTS,
    EXPECTED_EXCLUSIONS_Q1,
    EXPECTED_EXCLUSIONS_Q2,
    OUTPUTS,
    QC_SETS,
    RUNS,
    SMOOTHINGS,
    assert_frozen_contract,
    ceil_fraction,
    estimability,
    load_frozen_design,
    participants,
    run_paths,
    save_json_atomic,
    save_tsv_atomic,
    sha256,
    validate_acompcor_metadata,
)


def same_grid(image: nib.spatialimages.SpatialImage, reference) -> bool:
    return image.shape == reference[0] and np.allclose(image.affine, reference[1])


def main() -> int:
    assert_frozen_contract()
    people = participants()
    if len(people) != 108:
        raise RuntimeError(f"Expected 108 subjects, observed {len(people)}")
    group_counts = {
        key: int(value)
        for key, value in people["group"].value_counts().to_dict().items()
    }
    expected_groups = {"equalIndifference": 54, "equalRange": 54}
    if group_counts != expected_groups:
        raise RuntimeError(f"Unexpected group counts: {group_counts}")

    subject_rows: list[dict] = []
    run_rows: list[dict] = []
    reference_image = None
    total_coverage = None
    cohort_coverage = {
        "equalIndifference": None,
        "equalRange": None,
    }
    mask_voxel_counts: list[int] = []

    for person in people.itertuples(index=False):
        subject = person.participant_id
        group = person.group
        total_fd = 0.0
        total_fd_frames = 0
        max_fd_fraction = -np.inf
        run_fd_means = []
        dvars_means = []
        for run in RUNS:
            paths = run_paths(subject, run)
            missing = [name for name, path in paths.items() if not path.is_file()]
            if missing:
                raise RuntimeError(f"{subject} run {run}: missing source files {missing}")

            confounds = pd.read_csv(paths["confounds_tsv"], sep="\t")
            with paths["confounds_json"].open() as stream:
                confound_metadata = json.load(stream)
            validate_acompcor_metadata(confound_metadata, paths["confounds_json"])
            if any(column not in confounds for column in ACOMPCOR6):
                raise RuntimeError(f"{paths['confounds_tsv']}: missing frozen aCompCor")

            bold = nib.load(paths["bold"])
            if bold.ndim != 4 or bold.shape[3] != len(confounds):
                raise RuntimeError(
                    f"{paths['bold']}: shape {bold.shape} versus {len(confounds)} confounds"
                )
            design_rows = sum(1 for _ in paths["design"].open()) - 1
            if design_rows != bold.shape[3]:
                raise RuntimeError(
                    f"{paths['design']}: {design_rows} rows versus {bold.shape[3]} volumes"
                )

            fd = pd.to_numeric(confounds["framewise_displacement"], errors="coerce")
            if not pd.isna(fd.iloc[0]):
                raise RuntimeError(f"{paths['confounds_tsv']}: first FD is not NaN")
            defined_fd = fd.iloc[1:]
            if defined_fd.isna().any():
                raise RuntimeError(
                    f"{paths['confounds_tsv']}: non-first-frame undefined FD"
                )
            fd_sum = float(defined_fd.sum())
            fd_n = int(defined_fd.size)
            fd_mean = fd_sum / fd_n
            fd_fraction = float((defined_fd > 0.5).mean())
            total_fd += fd_sum
            total_fd_frames += fd_n
            max_fd_fraction = max(max_fd_fraction, fd_fraction)
            run_fd_means.append(fd_mean)
            run_std_dvars = np.nan
            if "std_dvars" in confounds:
                run_std_dvars = float(pd.to_numeric(
                    confounds["std_dvars"], errors="coerce"
                ).iloc[1:].mean())
                dvars_means.append(run_std_dvars)

            contrast_details = {}
            for confound_model in CONFOUNDS:
                design = load_frozen_design(subject, run, confound_model)
                model_details = {
                    "columns": design.shape[1],
                    "rank": None,
                }
                for contrast in CONTRASTS:
                    is_estimable, rank, error = estimability(design, contrast)
                    model_details["rank"] = rank
                    model_details[f"{contrast}_estimable"] = is_estimable
                    model_details[f"{contrast}_projection_error"] = error
                    if not is_estimable:
                        raise RuntimeError(
                            f"{subject} run {run} {confound_model}: "
                            f"{contrast} is not estimable (projection error {error})"
                        )
                contrast_details[confound_model] = model_details

            mask_image = nib.load(paths["mask"])
            if reference_image is None:
                reference_image = mask_image
                reference = (reference_image.shape, reference_image.affine)
                total_coverage = np.zeros(reference_image.shape, dtype=np.uint16)
                cohort_coverage = {
                    key: np.zeros(reference_image.shape, dtype=np.uint16)
                    for key in cohort_coverage
                }
            if not same_grid(mask_image, reference):
                mask_image = resample_from_to(mask_image, reference, order=0)
            mask = np.asarray(mask_image.dataobj, dtype=np.uint8) > 0
            mask_voxel_counts.append(int(mask.sum()))
            total_coverage += mask
            cohort_coverage[group] += mask

            run_row = {
                "participant_id": subject,
                "group": group,
                "run": run,
                "n_volumes": bold.shape[3],
                "defined_fd_frames": fd_n,
                "mean_fd_mm": fd_mean,
                "proportion_fd_gt_0_5": fd_fraction,
                "mean_std_dvars_diagnostic": run_std_dvars,
                "mask_voxels": int(mask.sum()),
            }
            for model, details in contrast_details.items():
                for key, value in details.items():
                    run_row[f"{model}_{key}"] = value
            run_rows.append(run_row)

        subject_mean_fd = total_fd / total_fd_frames
        q1_computed = subject_mean_fd <= 0.30 and max_fd_fraction <= 0.20
        q2_computed = (
            q1_computed
            and subject_mean_fd <= 0.20
            and max_fd_fraction <= 0.10
        )
        subject_rows.append({
            "participant_id": subject,
            "group": group,
            "n_runs": 4,
            "defined_fd_frames": total_fd_frames,
            "weighted_mean_fd_mm": subject_mean_fd,
            "max_run_proportion_fd_gt_0_5": max_fd_fraction,
            "mean_std_dvars_diagnostic": (
                float(np.nanmean(dvars_means)) if dvars_means else np.nan
            ),
            "Q0_all": True,
            "Q1_lenient": q1_computed,
            "Q2_strict": q2_computed,
            "frozen_exclusion_stage": (
                "Q0_to_Q1" if subject in EXPECTED_EXCLUSIONS_Q1
                else "Q1_to_Q2" if subject in EXPECTED_EXCLUSIONS_Q2
                else "retained_Q2"
            ),
        })

    subject_table = pd.DataFrame(subject_rows)
    run_table = pd.DataFrame(run_rows)
    observed_q1_excluded = set(
        subject_table.loc[~subject_table["Q1_lenient"], "participant_id"]
    )
    observed_q2_excluded = set(
        subject_table.loc[~subject_table["Q2_strict"], "participant_id"]
    )
    if observed_q1_excluded != EXPECTED_EXCLUSIONS_Q1:
        raise RuntimeError(
            f"Q1 mismatch: expected {sorted(EXPECTED_EXCLUSIONS_Q1)}, "
            f"observed {sorted(observed_q1_excluded)}"
        )
    if observed_q2_excluded != EXPECTED_EXCLUSIONS_Q2:
        raise RuntimeError(
            f"Q2 mismatch: expected {sorted(EXPECTED_EXCLUSIONS_Q2)}, "
            f"observed {sorted(observed_q2_excluded)}"
        )
    if not (
        (subject_table["Q2_strict"] <= subject_table["Q1_lenient"]).all()
        and (subject_table["Q1_lenient"] <= subject_table["Q0_all"]).all()
    ):
        raise RuntimeError("QC nesting violation")
    qc_counts = {
        qc: {
            "total": int(subject_table[qc].sum()),
            "equalIndifference": int(
                subject_table.loc[
                    subject_table[qc] & (subject_table.group == "equalIndifference")
                ].shape[0]
            ),
            "equalRange": int(
                subject_table.loc[
                    subject_table[qc] & (subject_table.group == "equalRange")
                ].shape[0]
            ),
        }
        for qc in QC_SETS
    }
    expected_qc_counts = {
        "Q0_all": {"total": 108, "equalIndifference": 54, "equalRange": 54},
        "Q1_lenient": {"total": 106, "equalIndifference": 54, "equalRange": 52},
        "Q2_strict": {"total": 103, "equalIndifference": 54, "equalRange": 49},
    }
    if qc_counts != expected_qc_counts:
        raise RuntimeError(f"QC count mismatch: {qc_counts}")

    total_threshold = ceil_fraction(0.95, len(run_rows))
    cohort_runs = {
        group: int((run_table.group == group).sum()) for group in cohort_coverage
    }
    common = total_coverage >= total_threshold
    for group, coverage in cohort_coverage.items():
        common &= coverage >= ceil_fraction(0.95, cohort_runs[group])
    common_voxels = int(common.sum())
    median_run_mask_voxels = float(np.median(mask_voxel_counts))
    if common_voxels == 0 or common_voxels < 0.5 * median_run_mask_voxels:
        raise RuntimeError(
            f"Frozen common-mask validity failure: common={common_voxels}, "
            f"median_run_mask={median_run_mask_voxels}"
        )
    header = reference_image.header.copy()
    header.set_data_dtype(np.uint8)
    mask_output = nib.Nifti1Image(common.astype(np.uint8), reference_image.affine, header)
    nib.save(mask_output, OUTPUTS / "common_analysis_mask.nii.gz")

    save_tsv_atomic(OUTPUTS / "qc_subject_sets.tsv", subject_table)
    save_tsv_atomic(OUTPUTS / "run_design_qc_audit.tsv", run_table)
    manifest_rows = []
    for contrast in CONTRASTS:
        for confound_model in CONFOUNDS:
            for qc in QC_SETS:
                for smoothing in SMOOTHINGS:
                    manifest_rows.append({
                        "contrast": contrast,
                        "confound": confound_model,
                        "qc": qc,
                        "smoothing": smoothing,
                        "fwhm_mm": {"S0": 0, "S1": 5, "S2": 8}[smoothing],
                        "expected_subjects": qc_counts[qc]["total"],
                        "status": "pending_subject_fits",
                        "effect_map": "",
                        "t_map": "",
                        "fdr_map": "",
                        "error": "",
                    })
    save_tsv_atomic(OUTPUTS / "multiverse_manifest.tsv", pd.DataFrame(manifest_rows))
    audit_payload = {
        "schema_version": "narps.source_audit.v1",
        "status": "pass",
        "contract_hashes": {
            name: sha256(OUTPUTS / name) for name in CONTRACT_HASHES
        },
        "inputs_frozen_grid_present": (
            OUTPUTS.parent / "inputs/frozen_grid.json"
        ).is_file(),
        "subjects": len(subject_table),
        "runs": len(run_table),
        "group_counts": group_counts,
        "qc_counts": qc_counts,
        "common_mask": {
            "voxels": common_voxels,
            "median_run_mask_voxels": median_run_mask_voxels,
            "overall_required_runs": total_threshold,
            "cohort_runs": cohort_runs,
            "cohort_required_runs": {
                group: ceil_fraction(0.95, count)
                for group, count in cohort_runs.items()
            },
            "shape": [int(value) for value in reference_image.shape],
            "zooms": [
                float(value) for value in reference_image.header.get_zooms()[:3]
            ],
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "nibabel": nib.__version__,
            "nilearn": nilearn.__version__,
        },
        "notes": [
            "std_dvars was read only as a diagnostic.",
            "No BOLD voxel values or neural outcome maps were read by this audit.",
            "The workspace inputs directory did not supply frozen_grid.json; the user-supplied frozen grid was locked verbatim in the output contracts."
                if not (OUTPUTS.parent / "inputs/frozen_grid.json").is_file()
                else "inputs/frozen_grid.json was present.",
        ],
    }
    save_json_atomic(OUTPUTS / "source_audit.json", audit_payload)
    print(json.dumps(audit_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
