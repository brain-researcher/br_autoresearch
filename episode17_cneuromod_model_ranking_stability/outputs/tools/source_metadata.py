#!/usr/bin/env python3
"""EP17 steward metadata preparation, with no neural/image/behavior value reader.

The original acquisition lacks THINGS/fmriprep events. The provider's
imgDesignNumbers, sparse design coordinates and runlist nevertheless preserve
image identity and GLMsingle presentation order. This command reports that
metadata without treating it as completed event reconciliation or eligibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path, PurePosixPath
import socket
import struct
from collections import Counter

import h5py
import numpy as np


SUBJECTS = ("01", "02", "03", "06")
STAGES = {
    "B": "TYPEB_FITHRF.mat",
    "C": "TYPEC_FITHRF_GLMDENOISE.mat",
    "D": "TYPED_FITHRF_GLMDENOISE_RR.mat",
}
PRESENTATION_FIELDS = (
    "participant", "session", "run", "run_order", "trial_in_run_zero_based",
    "glmsingle_trial_index_zero_based", "onset_TR_after_two_volume_removal",
    "provider_image_path", "image_name", "concept_name", "things_category_nr",
    "provider_category_nr", "provider_exemplar_nr", "condition",
    "image_presentation_number", "previous_same_image_index_gap",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def original_file(path: Path, source_root: Path) -> Path:
    """Allow original git-annex links only when they remain inside this source."""
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(source_root), "source link leaves source root")
    require(resolved.is_file(), "expected a regular source file")
    return resolved


def dataset_at(handle, key):
    """Do not follow HDF5 soft/external links or virtual/external payloads."""
    obj = handle
    for part in key.split("/"):
        require(isinstance(obj, h5py.Group), "expected HDF5 group")
        require(isinstance(obj.get(part, getlink=True), h5py.HardLink),
                f"missing or nonlocal HDF5 member: {key}")
        obj = obj[part]
    require(isinstance(obj, h5py.Dataset), f"not a dataset: {key}")
    require(not obj.is_virtual and obj.external is None,
            f"nonlocal dataset storage: {key}")
    return obj


def integer_metadata(handle, key, columns=None):
    """Only caller-named runlist/design metadata reaches this value reader."""
    data = dataset_at(handle, key)
    require(data.dtype.kind in "iuf", f"nonnumeric metadata: {key}")
    if columns is not None:
        require(data.ndim == 2 and data.shape[1] == columns,
                f"unexpected coordinate axes: {key}")
    else:
        require(data.ndim <= 2 and (data.ndim < 2 or 1 in data.shape),
                f"unexpected metadata vector: {key}")
    values = data[()]
    require(np.isfinite(values).all() and np.equal(values, np.floor(values)).all(),
            f"noninteger metadata: {key}")
    return values.astype(np.int64)


def read_image_identity(path):
    """Pinned provider identity CSV only; no image entry is opened."""
    required = {"image_path", "condition", "category_nr", "exemplar_nr",
                "things_category_nr"}
    by_name = {}
    concept_names = {}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        require(required.issubset(rows.fieldnames or ()),
                "pinned image identity CSV lacks required columns")
        for row in rows:
            rel = PurePosixPath(row["image_path"])
            # Keep catch and practice/test identities: parsing is not eligibility.
            condition = row["condition"]
            path_classes = {
                "exp": ("images_fmri", 3),
                "catch": ("images_catch_fmri", 2),
                "test": ("images_test_fmri", 2),
            }
            require(condition in path_classes, "unknown provider image condition")
            prefix, depth = path_classes[condition]
            require(len(rel.parts) == depth and rel.parts[0] == prefix
                    and ".." not in rel.parts, "unexpected provider image path")
            require(rel.name not in by_name, "ambiguous provider image basename")
            if condition != "catch":
                require(bool(row["things_category_nr"]), "missing source concept ID")
            if condition == "exp":
                previous_name = concept_names.setdefault(
                    row["things_category_nr"], rel.parent.name)
                require(previous_name == rel.parent.name, "ambiguous provider concept name")
            by_name[rel.name] = {
                "provider_image_path": str(rel), "image_name": rel.name,
                "concept_name": rel.parent.name if condition == "exp" else "",
                "things_category_nr": row["things_category_nr"],
                "provider_category_nr": row["category_nr"],
                "provider_exemplar_nr": row["exemplar_nr"],
                "condition": row["condition"],
            }
    for item in by_name.values():
        if item["condition"] == "test":
            item["concept_name"] = concept_names.get(item["things_category_nr"], "")
    return by_name


def nifti1_header(path):
    """Read exactly the fixed 348-byte NIfTI-1 header, never its voxel payload."""
    with path.open("rb") as stream:
        raw = stream.read(348)
    require(len(raw) == 348, "short NIfTI header")
    endian = "<" if struct.unpack("<i", raw[:4])[0] == 348 else ">"
    require(struct.unpack(endian + "i", raw[:4])[0] == 348,
            "not a NIfTI-1 fixed header")
    require(raw[344:348] == b"n+1\x00", "not a single-file NIfTI-1 header")
    dims = struct.unpack_from(endian + "8h", raw, 40)
    require(1 <= dims[0] <= 7, "invalid NIfTI dimensions")
    return {
        "shape": list(dims[1:1 + dims[0]]),
        "datatype_code": struct.unpack_from(endian + "h", raw, 70)[0],
        "pixdim": list(struct.unpack_from(endian + "8f", raw, 76)),
        "qform_code": struct.unpack_from(endian + "h", raw, 252)[0],
        "sform_code": struct.unpack_from(endian + "h", raw, 254)[0],
        "quaternion_and_offset": list(struct.unpack_from(endian + "6f", raw, 256)),
        "srow_xyz": [list(struct.unpack_from(endian + "4f", raw, offset))
                     for offset in (280, 296, 312)],
        "voxel_values_read": False,
    }


def participant_metadata(source_root, subject, identities, runlist):
    label = "sub-" + subject
    glm = source_root / "THINGS/glmsingle"
    inputs = glm / label / "glmsingle/input"
    image_map_path = original_file(inputs / f"{label}_task-things_imgDesignNumbers.json",
                                   source_root)
    image_map = json.loads(image_map_path.read_text(encoding="utf-8"))
    require(isinstance(image_map, dict) and bool(image_map), "empty image mapping")
    require(all(isinstance(k, str) and Path(k).name == k and
                type(v) is int for k, v in image_map.items()),
            "unexpected image-to-design mapping")
    require(set(image_map.values()) == set(range(len(image_map))),
            "image design indices are not unique contiguous zero-based IDs")
    inverse = {value: name for name, value in image_map.items()}
    require(set(image_map).issubset(identities), "design image absent from pinned CSV")
    sessions = integer_metadata(runlist, f"{subject}/sessions").ravel().tolist()
    require(len(set(sessions)) == len(sessions), "duplicate runlist session")
    design_path = original_file(
        inputs / f"{label}_task-things_model-glmsingle_desc-sparse_design.h5",
        source_root)
    presentations, run_rows = [], []
    repeat_counts, previous = Counter(), {}
    with h5py.File(design_path, "r") as design:
        total_conditions = integer_metadata(design, "total_conditions").ravel()
        tr_count = integer_metadata(design, "TR_count").ravel()
        require(total_conditions.size == tr_count.size == 1, "invalid design scalars")
        require(int(total_conditions[0]) == len(image_map), "condition map mismatch")
        tr_dataset = dataset_at(design, "TR")
        require(tr_dataset.size == 1 and tr_dataset.dtype.kind in "iuf",
                "invalid TR metadata")
        tr_seconds = float(np.asarray(tr_dataset[()]).item())
        require(np.isfinite(tr_seconds) and tr_seconds > 0, "invalid TR duration")
        for session in sessions:
            runs = integer_metadata(runlist, f"{subject}/{session}").ravel().tolist()
            require(len(set(runs)) == len(runs), "duplicate runlist run")
            for run in runs:
                key = f"{session:02d}/{run:02d}/design_coord"
                coords = integer_metadata(design, key, columns=2)
                require(len(coords) > 0, "empty retained run")
                require(np.all(np.diff(coords[:, 0]) > 0),
                        "design onsets not strictly increasing; trial order unresolved")
                require(np.all((coords[:, 0] >= 0) & (coords[:, 0] < tr_count[0])),
                        "design onset outside recorded run")
                run_order = len(run_rows)
                start_index = len(presentations)
                for trial_index, (onset, image_index) in enumerate(coords.tolist()):
                    require(image_index in inverse, "unmapped design image index")
                    name = inverse[image_index]
                    global_index = len(presentations)
                    repeat_counts[name] += 1
                    gap = global_index - previous[name] if name in previous else ""
                    previous[name] = global_index
                    presentations.append({
                        "participant": label, "session": session, "run": run,
                        "run_order": run_order,
                        "trial_in_run_zero_based": trial_index,
                        "glmsingle_trial_index_zero_based": global_index,
                        "onset_TR_after_two_volume_removal": onset,
                        **identities[name],
                        "image_presentation_number": repeat_counts[name],
                        "previous_same_image_index_gap": gap,
                    })
                run_rows.append({
                    "participant": label, "session": session, "run": run,
                    "run_order": run_order, "presentations": len(coords),
                    "first_glmsingle_trial_index_zero_based": start_index,
                    "TR_seconds": tr_seconds, "TR_count": int(tr_count[0]),
                })

    stage_axes, issues = {}, []
    for stage, filename in STAGES.items():
        path = glm / label / "glmsingle/output/T1w" / filename
        try:
            with h5py.File(original_file(path, source_root), "r") as handle:
                # The sole neural object reached here is described, never sliced.
                betas = dataset_at(handle, "modelmd")
                shape = list(betas.shape)
                squeezed = [size for size in shape if size != 1]
                matches = len(squeezed) == 2 and squeezed[0] == len(presentations)
                stage_axes[stage] = {
                    "file": str(path.relative_to(source_root)),
                    "dataset": "modelmd", "stored_shape": shape,
                    "dtype": str(betas.dtype), "singleton_removed_shape": squeezed,
                    "trial_axis_matches_design_count": matches,
                    "neural_values_read": False,
                }
                if not matches:
                    issues.append(f"stage {stage} trial axis requires reconciliation")
        except (OSError, ValueError, KeyError) as exc:
            stage_axes[stage] = {"structural_error": str(exc)}
            issues.append(f"stage {stage} structural metadata unavailable")
    shapes = [x.get("stored_shape") for x in stage_axes.values()]
    equal_shapes = all(x is not None and x == shapes[0] for x in shapes)
    if not equal_shapes:
        issues.append("B/C/D stored axes are missing or differ")
    masks, optional_metadata_warnings = {}, []
    for mask_name in ("union", "unionNonNaN", "unionNaN"):
        path = inputs / f"{label}_task-things_space-T1w_label-brain_desc-{mask_name}_mask.nii"
        try:
            masks[mask_name] = nifti1_header(original_file(path, source_root))
        except (OSError, ValueError) as exc:
            masks[mask_name] = {"structural_error": str(exc)}
            if mask_name == "unionNaN":
                # Provider diagnostic complement, not either model's input mask.
                # Its absence does not prevent this identity/header-only task.
                optional_metadata_warnings.append("unionNaN diagnostic mask header unavailable")
            else:
                issues.append(f"{mask_name} mask header unavailable")
    summary = {
        "participant": label, "sessions": len(sessions), "runs": len(run_rows),
        "presentations": len(presentations), "unique_design_images": len(image_map),
        "unique_retained_run_images": len(repeat_counts),
        "image_repeat_count_histogram": dict(sorted(Counter(repeat_counts.values()).items())),
        "retained_run_concepts_with_provider_ID": len({identities[n]["things_category_nr"]
                                      for n in repeat_counts if identities[n]["things_category_nr"]}),
        "unique_retained_run_images_by_condition": dict(Counter(
            identities[n]["condition"] for n in repeat_counts)),
        "presentations_by_condition": dict(Counter(row["condition"] for row in presentations)),
        "stage_axes": stage_axes, "B_C_D_stored_shapes_equal": equal_shapes,
        "mask_headers": masks, "structural_issues": issues,
        "optional_metadata_warnings": optional_metadata_warnings,
        "finite_voxel_identity_or_response_units_verified": False,
    }
    return summary, presentations, run_rows, repeat_counts


def write_csv(path, fields, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--stimulus-metadata-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname(),
            "run metadata processing in a coordinator-owned Slurm compute allocation")
    source = args.source.resolve(strict=True)
    require(source.name == "cneuromod-things-1.0.1-restricted-raw", "wrong source release")
    require(args.stimulus_metadata_csv.name == "image_paths_fmri.csv",
            "use the pinned provider image identity table")
    require(not args.out.exists(), "attempt output exists; preserve it and select a new attempt")
    args.out.mkdir(parents=True)
    report = {
        "analysis_id": "source_metadata", "source": str(source),
        "stimulus_metadata_csv": str(args.stimulus_metadata_csv),
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "source_roles_complete": False, "roles_assigned": False,
        "eligible_720_concept_universe_verified": False,
        "neural_values_read": False, "mask_voxel_values_read": False,
        "image_values_read": False, "behavioral_outcomes_read": False,
        "audit_opened": False, "candidate_scoring_run": False,
        "source_basis": "provider sparse design, runlist and image identity metadata",
        "acquisition_order_basis": "provider GLMsingle_run.m runlist traversal then ascending design onset",
        "repetition_gap_units": "retained-presentation index difference, not elapsed wall time",
        "assignment_numerical_rules": "prospective; no eligibility or role rule inferred here",
        "source_raw_event_file_count": 0,
        "raw_event_reconciliation_complete": False,
        "participants": [], "structural_errors": [],
    }
    presentations, run_rows, image_rows, seen_images = [], [], [], []
    try:
        # Inspect only filenames in the one documented event location, not TSV rows.
        report["source_raw_event_file_count"] = sum(
            len(list((source / "THINGS/fmriprep/sourcedata/things" / f"sub-{s}").glob(
                "ses-*/func/*_task-things_run-*_events.tsv"))) for s in SUBJECTS)
        identities = read_image_identity(args.stimulus_metadata_csv)
        report["pinned_csv_image_count"] = len(identities)
        report["pinned_csv_images_by_condition"] = dict(Counter(
            item["condition"] for item in identities.values()))
        runlist_path = original_file(source / "THINGS/glmsingle/task-things_runlist.h5", source)
        with h5py.File(runlist_path, "r") as runlist:
            for subject in SUBJECTS:
                try:
                    summary, subject_rows, subject_runs, counts = participant_metadata(
                        source, subject, identities, runlist)
                    report["participants"].append(summary)
                    presentations.extend(subject_rows)
                    run_rows.extend(subject_runs)
                    seen_images.append(set(counts))
                    for name, count in sorted(counts.items()):
                        image_rows.append({"participant": "sub-" + subject,
                                           **identities[name], "presentations": count})
                    report["structural_errors"].extend(
                        f"sub-{subject}: {issue}" for issue in summary["structural_issues"])
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    report["structural_errors"].append(f"sub-{subject}: {type(exc).__name__}: {exc}")
        if len(seen_images) == len(SUBJECTS):
            common = set.intersection(*seen_images)
            report["unqualified_common_image_count"] = len(common)
            report["unqualified_common_concept_count"] = len(
                {identities[name]["things_category_nr"] for name in common
                 if identities[name]["things_category_nr"]})
            report["common_images_by_condition"] = dict(Counter(
                identities[name]["condition"] for name in common))
            report["common_images_per_concept_histogram"] = dict(sorted(Counter(Counter(
                identities[name]["things_category_nr"] for name in common
                if identities[name]["things_category_nr"]).values()).items()))
            experimental_common = {name for name in common if identities[name]["condition"] == "exp"}
            report["unqualified_experimental_common_image_count"] = len(experimental_common)
            report["unqualified_experimental_common_concept_count"] = len({
                identities[name]["things_category_nr"] for name in experimental_common})
            report["catch_test_identity_handling"] = (
                "All retained-run identities preserved; catch empty concept IDs are not "
                "counted as concepts; condition-specific counts are descriptive, not eligibility")
        write_csv(args.out / "presentation_identity.csv", PRESENTATION_FIELDS, presentations)
        write_csv(args.out / "run_metadata.csv", (
            "participant", "session", "run", "run_order", "presentations",
            "first_glmsingle_trial_index_zero_based", "TR_seconds", "TR_count"), run_rows)
        write_csv(args.out / "image_repeat_summary.csv", (
            "participant", "provider_image_path", "image_name", "concept_name",
            "things_category_nr", "provider_category_nr", "provider_exemplar_nr",
            "condition", "presentations"), image_rows)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["structural_errors"].append(f"source: {type(exc).__name__}: {exc}")
    report["remaining_requirements"] = [
        "Raw-event identity reconciliation, including provider schedule exceptions",
        "Prospective repeat-completeness and taxonomy/balance assignment specifications",
        "Qualified exact720 concepts and deterministic480/120/120 roles, folds and blocks",
        "Materialized role-separated handoffs with declared reader enforcement",
        "B/C/D voxel identity, finite response and unit checks in authorized later builder stage",
    ]
    (args.out / "metadata_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"analysis_id": "source_metadata", "source_roles_complete": False,
                      "participants_processed": len(report["participants"]),
                      "structural_error_count": len(report["structural_errors"])}))
    return 2 if report["structural_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
