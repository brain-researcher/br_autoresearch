#!/usr/bin/env python
"""Audit edge-scaling diagnostics from immutable subject checkpoints only.

This diagnostic runs after all 108 subject fits and before group aggregation.
It never opens source BOLD images and never modifies subject or primary map
artifacts.  It validates the complete NPZ/JSON inventory and frozen axes, then
summarizes fixed-effect variances and any *available* sidecar counts produced
from the separately calculated float64 temporal mean below 1.  Those sidecar
counts are diagnostics; they are not asserted to equal Nilearn's internal
mean-scaling clamp count, which is calculated from the fit array's own dtype.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from narps_common import (
    CONFOUNDS,
    CONTRASTS,
    OUTPUTS,
    RUNS,
    SMOOTHINGS,
    assert_frozen_contract,
    load_mask_vector,
    sha256,
)


SCHEMA_VERSION = "narps.edge_scaling_diagnostic.v1"
EXPECTED_SUBJECTS = 108
VARIANCE_THRESHOLDS = (1e-20, 1e-30)
DIAGNOSTIC_SOURCE_FIELD = "mean_scaling_clamped_voxels"
DIAGNOSTIC_INTERPRETATION = (
    "Available values summarize voxels whose separately computed float64 "
    "temporal mean was below 1 before mean_scaling. They are diagnostic only "
    "and are not claimed to be exact counts of Nilearn's internal clamp."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic checks under outputs/tmp; do not inspect production maps.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_bytes_once(path: Path, payload: bytes) -> None:
    """Create an artifact atomically, or verify a byte-identical prior copy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to overwrite nonidentical artifact: {path}")
        return
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Linking, rather than replacing, preserves create-once semantics if
            # another process creates the destination concurrently.
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(
                    f"Concurrent nonidentical artifact exists: {path}"
                )
    finally:
        if temporary.exists():
            temporary.unlink()


def save_table_once(path: Path, table: pd.DataFrame) -> None:
    payload = table.to_csv(
        sep="\t",
        index=False,
        lineterminator="\n",
        na_rep="",
        float_format="%.17g",
    ).encode("utf-8")
    save_bytes_once(path, payload)


def save_json_once(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    save_bytes_once(path, encoded)


def expected_subjects_from_qc(path: Path) -> list[str]:
    if not path.is_file():
        raise RuntimeError(f"Missing frozen QC subject table: {path}")
    table = pd.read_csv(path, sep="\t", dtype={"participant_id": str})
    if "participant_id" not in table.columns:
        raise RuntimeError(f"{path} lacks participant_id")
    subjects = table["participant_id"].astype(str).tolist()
    if len(subjects) != EXPECTED_SUBJECTS or len(set(subjects)) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Frozen QC table must contain exactly {EXPECTED_SUBJECTS} unique subjects"
        )
    malformed = [subject for subject in subjects if not subject.startswith("sub-")]
    if malformed:
        raise RuntimeError(f"Malformed subject IDs in QC table: {malformed[:8]}")
    return sorted(subjects, key=lambda value: int(value.removeprefix("sub-")))


def exact_inventory(subject_dir: Path, subjects: Iterable[str]) -> dict[str, tuple[Path, Path]]:
    expected = {
        subject: (
            subject_dir / f"{subject}_fixed_effects.npz",
            subject_dir / f"{subject}_fixed_effects.json",
        )
        for subject in subjects
    }
    expected_npz = {pair[0] for pair in expected.values()}
    expected_json = {pair[1] for pair in expected.values()}
    observed_npz = set(subject_dir.glob("sub-*_fixed_effects.npz"))
    observed_json = set(subject_dir.glob("sub-*_fixed_effects.json"))
    missing_npz = sorted(str(path) for path in expected_npz - observed_npz)
    missing_json = sorted(str(path) for path in expected_json - observed_json)
    extra_npz = sorted(str(path) for path in observed_npz - expected_npz)
    extra_json = sorted(str(path) for path in observed_json - expected_json)
    if (
        len(observed_npz) != EXPECTED_SUBJECTS
        or len(observed_json) != EXPECTED_SUBJECTS
        or missing_npz
        or missing_json
        or extra_npz
        or extra_json
    ):
        raise RuntimeError(
            "Subject checkpoint inventory must be exactly 108 NPZ+JSON pairs; "
            f"observed_npz={len(observed_npz)}, observed_json={len(observed_json)}, "
            f"missing_npz={missing_npz[:8]}, missing_json={missing_json[:8]}, "
            f"extra_npz={extra_npz[:8]}, extra_json={extra_json[:8]}"
        )
    return expected


def strict_int(value: Any, label: str, lower: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RuntimeError(f"{label} must be an integer, observed {value!r}")
    result = int(value)
    if result < lower:
        raise RuntimeError(f"{label} must be >= {lower}, observed {result}")
    return result


def validate_sidecar(
    path: Path,
    subject: str,
    n_voxels: int,
    mask_hash: str,
) -> tuple[dict[tuple[str, str], list[int]] | None, dict[str, Any]]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unreadable checkpoint sidecar {path}: {error}") from error
    expected_scalars = {
        "schema_version": "narps.subject_fixed_effects.v1",
        "subject": subject,
        "pilot": False,
        "n_voxels": n_voxels,
        "mask_sha256": mask_hash,
        "nilearn_ar1_bins": 100,
    }
    for key, expected in expected_scalars.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"{path}: invalid {key}: {metadata.get(key)!r} != {expected!r}"
            )
    expected_axes = {
        "runs": list(RUNS),
        "confounds": list(CONFOUNDS),
        "smoothings": list(SMOOTHINGS),
        "contrasts": list(CONTRASTS),
    }
    for key, expected in expected_axes.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"{path}: {key} axis mismatch")

    records = metadata.get("run_records")
    if not isinstance(records, list) or len(records) != (
        len(RUNS) * len(CONFOUNDS) * len(SMOOTHINGS)
    ):
        raise RuntimeError(f"{path}: expected exactly 36 run records")
    indexed: dict[tuple[int, str, str], dict[str, Any]] = {}
    availability: list[bool] = []
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise RuntimeError(f"{path}: run record {record_index} is not an object")
        key = (record.get("run"), record.get("confound"), record.get("smoothing"))
        if (
            key[0] not in RUNS
            or key[1] not in CONFOUNDS
            or key[2] not in SMOOTHINGS
            or key in indexed
        ):
            raise RuntimeError(f"{path}: invalid/duplicate run-record key {key!r}")
        indexed[key] = record
        availability.append(DIAGNOSTIC_SOURCE_FIELD in record)
    expected_keys = {
        (run, confound, smoothing)
        for run in RUNS
        for confound in CONFOUNDS
        for smoothing in SMOOTHINGS
    }
    if set(indexed) != expected_keys:
        raise RuntimeError(f"{path}: incomplete run-record axes")
    if any(availability) and not all(availability):
        raise RuntimeError(f"{path}: partially available temporal-mean diagnostic")
    if not any(availability):
        return None, metadata

    by_cell: dict[tuple[str, str], list[int]] = {}
    for confound in CONFOUNDS:
        for smoothing in SMOOTHINGS:
            values = [
                strict_int(
                    indexed[(run, confound, smoothing)][DIAGNOSTIC_SOURCE_FIELD],
                    f"{path}:{run}/{confound}/{smoothing}/{DIAGNOSTIC_SOURCE_FIELD}",
                )
                for run in RUNS
            ]
            if any(value > n_voxels for value in values):
                raise RuntimeError(f"{path}: temporal-mean diagnostic exceeds mask size")
            by_cell[(confound, smoothing)] = values

    # This diagnostic was calculated before the confound design was fit, so its
    # three replicated C records must agree within every run and smoothing.
    for run in RUNS:
        for smoothing in SMOOTHINGS:
            values = {
                strict_int(
                    indexed[(run, confound, smoothing)][DIAGNOSTIC_SOURCE_FIELD],
                    f"{path}:{run}/{confound}/{smoothing}/{DIAGNOSTIC_SOURCE_FIELD}",
                )
                for confound in CONFOUNDS
            }
            if len(values) != 1:
                raise RuntimeError(
                    f"{path}: replicated temporal-mean diagnostic disagrees across C"
                )
    return by_cell, metadata


def audit_checkpoint_set(
    inventory: dict[str, tuple[Path, Path]],
    n_voxels: int,
    mask_hash: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected_shape = (
        len(CONFOUNDS),
        len(SMOOTHINGS),
        len(CONTRASTS),
        n_voxels,
    )
    rows: list[dict[str, Any]] = []
    input_hash_records: list[str] = []
    diagnostic_available_subjects: list[str] = []
    diagnostic_unavailable_subjects: list[str] = []

    for subject_index, (subject, (npz_path, json_path)) in enumerate(inventory.items()):
        checkpoint_hash = file_sha256(npz_path)
        sidecar_hash = file_sha256(json_path)
        input_hash_records.append(f"{subject}\t{checkpoint_hash}\t{sidecar_hash}\n")
        diagnostic_by_cell, _ = validate_sidecar(
            json_path, subject, n_voxels, mask_hash
        )
        if diagnostic_by_cell is None:
            diagnostic_unavailable_subjects.append(subject)
        else:
            diagnostic_available_subjects.append(subject)

        with np.load(npz_path, allow_pickle=False) as payload:
            required = {
                "effect",
                "variance",
                "confounds",
                "smoothings",
                "contrasts",
                "subject",
                "mask_sha256",
            }
            if not required.issubset(payload.files):
                raise RuntimeError(
                    f"{npz_path}: missing arrays {sorted(required - set(payload.files))}"
                )
            if str(payload["subject"].item()) != subject:
                raise RuntimeError(f"{npz_path}: subject identity mismatch")
            if str(payload["mask_sha256"].item()) != mask_hash:
                raise RuntimeError(f"{npz_path}: frozen-mask hash mismatch")
            if tuple(payload["confounds"].astype(str)) != tuple(CONFOUNDS):
                raise RuntimeError(f"{npz_path}: confound axis mismatch")
            if tuple(payload["smoothings"].astype(str)) != tuple(SMOOTHINGS):
                raise RuntimeError(f"{npz_path}: smoothing axis mismatch")
            if tuple(payload["contrasts"].astype(str)) != tuple(CONTRASTS):
                raise RuntimeError(f"{npz_path}: contrast axis mismatch")
            effect = np.asarray(payload["effect"])
            variance = np.asarray(payload["variance"])
            if effect.shape != expected_shape or variance.shape != expected_shape:
                raise RuntimeError(
                    f"{npz_path}: shapes {effect.shape}/{variance.shape} "
                    f"!= {expected_shape}"
                )
            if effect.dtype.kind != "f" or not np.isfinite(effect).all():
                raise RuntimeError(f"{npz_path}: effects must be finite floating point")
            if variance.dtype.kind != "f":
                raise RuntimeError(f"{npz_path}: variances must be floating point")

            for ci, confound in enumerate(CONFOUNDS):
                for si, smoothing in enumerate(SMOOTHINGS):
                    diagnostic_values = (
                        None
                        if diagnostic_by_cell is None
                        else diagnostic_by_cell[(confound, smoothing)]
                    )
                    for ki, contrast in enumerate(CONTRASTS):
                        values = np.asarray(variance[ci, si, ki], dtype=np.float64)
                        finite = np.isfinite(values)
                        positive = finite & (values > 0)
                        finite_count = int(finite.sum())
                        positive_count = int(positive.sum())
                        nonfinite_count = int((~finite).sum())
                        nonpositive_count = int((finite & (values <= 0)).sum())
                        finite_values = values[finite]
                        minimum = (
                            float(np.min(finite_values))
                            if finite_values.size
                            else None
                        )
                        row: dict[str, Any] = {
                            "subject": subject,
                            "confound": confound,
                            "smoothing": smoothing,
                            "contrast": contrast,
                            "n_voxels": n_voxels,
                            "variance_min": minimum,
                            "variance_lt_1e_20_count": int(
                                np.count_nonzero(finite & (values < VARIANCE_THRESHOLDS[0]))
                            ),
                            "variance_lt_1e_30_count": int(
                                np.count_nonzero(finite & (values < VARIANCE_THRESHOLDS[1]))
                            ),
                            "variance_finite_count": finite_count,
                            "variance_positive_count": positive_count,
                            "variance_nonfinite_count": nonfinite_count,
                            "variance_nonpositive_count": nonpositive_count,
                            "variance_all_finite": finite_count == n_voxels,
                            "variance_all_positive": positive_count == n_voxels,
                            "sidecar_available": True,
                            "float64_temporal_mean_lt1_diagnostic_available": (
                                diagnostic_values is not None
                            ),
                            "float64_temporal_mean_lt1_run_count_available": (
                                len(diagnostic_values)
                                if diagnostic_values is not None
                                else None
                            ),
                            "float64_temporal_mean_lt1_voxel_run_sum": (
                                sum(diagnostic_values)
                                if diagnostic_values is not None
                                else None
                            ),
                            "float64_temporal_mean_lt1_min_per_run": (
                                min(diagnostic_values)
                                if diagnostic_values is not None
                                else None
                            ),
                            "float64_temporal_mean_lt1_max_per_run": (
                                max(diagnostic_values)
                                if diagnostic_values is not None
                                else None
                            ),
                            "float64_temporal_mean_lt1_runs_nonzero": (
                                sum(value > 0 for value in diagnostic_values)
                                if diagnostic_values is not None
                                else None
                            ),
                        }
                        rows.append(row)

        if (subject_index + 1) % 12 == 0 or subject_index + 1 == len(inventory):
            print(
                f"Validated edge diagnostic inputs {subject_index + 1}/"
                f"{len(inventory)}",
                flush=True,
            )

    table = pd.DataFrame(rows)
    expected_rows = len(inventory) * len(CONFOUNDS) * len(SMOOTHINGS) * len(CONTRASTS)
    if len(table) != expected_rows:
        raise RuntimeError(f"Diagnostic row count {len(table)} != {expected_rows}")
    invalid_variance_rows = table.loc[
        ~table["variance_all_finite"] | ~table["variance_all_positive"]
    ]
    if not invalid_variance_rows.empty:
        raise RuntimeError(
            "Fixed-effect variance validation failed in "
            f"{len(invalid_variance_rows)} subject/cell/contrast rows"
        )

    inventory_digest = hashlib.sha256(
        "".join(input_hash_records).encode("utf-8")
    ).hexdigest()
    available = table["float64_temporal_mean_lt1_diagnostic_available"]
    available_rows = table.loc[available]
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scope": "diagnostic_only_subject_checkpoint_audit",
        "source_bold_accessed": False,
        "primary_maps_modified": False,
        "subject_count": len(inventory),
        "npz_count": len(inventory),
        "json_sidecar_count": len(inventory),
        "sidecars_available": len(inventory),
        "sidecars_missing": 0,
        "row_count": len(table),
        "expected_row_count": expected_rows,
        "n_voxels": n_voxels,
        "mask_sha256": mask_hash,
        "confounds": list(CONFOUNDS),
        "smoothings": list(SMOOTHINGS),
        "contrasts": list(CONTRASTS),
        "variance_thresholds_strictly_below": list(VARIANCE_THRESHOLDS),
        "variance_global_min": float(table["variance_min"].min()),
        "variance_lt_1e_20_total": int(table["variance_lt_1e_20_count"].sum()),
        "variance_lt_1e_30_total": int(table["variance_lt_1e_30_count"].sum()),
        "variance_finite_total": int(table["variance_finite_count"].sum()),
        "variance_positive_total": int(table["variance_positive_count"].sum()),
        "variance_nonfinite_total": int(table["variance_nonfinite_count"].sum()),
        "variance_nonpositive_total": int(table["variance_nonpositive_count"].sum()),
        "all_variances_finite_and_positive": True,
        "checkpoint_inventory_sha256": inventory_digest,
        "float64_temporal_mean_lt1_diagnostic": {
            "source_sidecar_field": DIAGNOSTIC_SOURCE_FIELD,
            "interpretation": DIAGNOSTIC_INTERPRETATION,
            "subjects_available_count": len(diagnostic_available_subjects),
            "subjects_available": diagnostic_available_subjects,
            "subjects_unavailable_count": len(diagnostic_unavailable_subjects),
            "subjects_unavailable": diagnostic_unavailable_subjects,
            "rows_available_count": int(available.sum()),
            "rows_unavailable_count": int((~available).sum()),
            "voxel_run_sum_across_available_rows": int(
                available_rows["float64_temporal_mean_lt1_voxel_run_sum"].sum()
            ),
            "max_per_run_across_available_rows": (
                int(available_rows["float64_temporal_mean_lt1_max_per_run"].max())
                if not available_rows.empty
                else None
            ),
        },
    }
    return table, summary


def synthetic_sidecar(subject: str, n_voxels: int, mask_hash: str) -> dict[str, Any]:
    records = []
    for run in RUNS:
        for smoothing_index, smoothing in enumerate(SMOOTHINGS):
            diagnostic = run + smoothing_index
            for confound in CONFOUNDS:
                records.append(
                    {
                        "run": run,
                        "confound": confound,
                        "smoothing": smoothing,
                        DIAGNOSTIC_SOURCE_FIELD: diagnostic,
                    }
                )
    return {
        "schema_version": "narps.subject_fixed_effects.v1",
        "subject": subject,
        "pilot": False,
        "n_voxels": n_voxels,
        "mask_sha256": mask_hash,
        "nilearn_ar1_bins": 100,
        "runs": list(RUNS),
        "confounds": list(CONFOUNDS),
        "smoothings": list(SMOOTHINGS),
        "contrasts": list(CONTRASTS),
        "run_records": records,
    }


def run_self_test() -> int:
    temporary_root = OUTPUTS / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    tests: list[str] = []
    with tempfile.TemporaryDirectory(
        prefix="audit_edge_scaling_selftest_", dir=temporary_root
    ) as directory_name:
        directory = Path(directory_name)
        subject = "sub-001"
        n_voxels = 8
        mask_hash = "a" * 64
        shape = (len(CONFOUNDS), len(SMOOTHINGS), len(CONTRASTS), n_voxels)
        effect = np.zeros(shape, dtype=np.float32)
        variance = np.full(shape, 1e-10, dtype=np.float32)
        variance[0, 0, 0, :4] = np.asarray(
            [1e-31, 1e-25, 1e-19, 1.0], dtype=np.float32
        )
        npz_path = directory / f"{subject}_fixed_effects.npz"
        json_path = directory / f"{subject}_fixed_effects.json"
        np.savez(
            npz_path,
            effect=effect,
            variance=variance,
            confounds=np.asarray(CONFOUNDS),
            smoothings=np.asarray(SMOOTHINGS),
            contrasts=np.asarray(CONTRASTS),
            subject=np.asarray(subject),
            mask_sha256=np.asarray(mask_hash),
        )
        json_path.write_text(
            json.dumps(synthetic_sidecar(subject, n_voxels, mask_hash)),
            encoding="utf-8",
        )
        table, summary = audit_checkpoint_set(
            {subject: (npz_path, json_path)}, n_voxels, mask_hash
        )
        target = table.loc[
            table["subject"].eq(subject)
            & table["confound"].eq("C0")
            & table["smoothing"].eq("S0")
            & table["contrast"].eq("gain_demean")
        ].iloc[0]
        assert int(target["variance_lt_1e_20_count"]) == 2
        assert int(target["variance_lt_1e_30_count"]) == 1
        assert bool(target["variance_all_finite"])
        assert bool(target["variance_all_positive"])
        assert int(target["float64_temporal_mean_lt1_voxel_run_sum"]) == 10
        assert summary["row_count"] == 18
        tests.append("synthetic_threshold_and_sidecar_summary")

        once_path = directory / "write_once.json"
        save_bytes_once(once_path, b"same\n")
        save_bytes_once(once_path, b"same\n")
        try:
            save_bytes_once(once_path, b"different\n")
        except FileExistsError:
            pass
        else:
            raise AssertionError("Nonidentical immutable write was not rejected")
        tests.append("immutable_writer")

        legacy = synthetic_sidecar(subject, n_voxels, mask_hash)
        for record in legacy["run_records"]:
            record.pop(DIAGNOSTIC_SOURCE_FIELD)
        legacy_path = directory / "legacy.json"
        legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
        diagnostic, _ = validate_sidecar(legacy_path, subject, n_voxels, mask_hash)
        assert diagnostic is None
        tests.append("legacy_sidecar_explicit_unavailability")

        incomplete = directory / "incomplete_inventory"
        incomplete.mkdir()
        synthetic_subjects = [f"sub-{index:03d}" for index in range(1, 109)]
        try:
            exact_inventory(incomplete, synthetic_subjects)
        except RuntimeError as error:
            assert "exactly 108 NPZ+JSON pairs" in str(error)
        else:
            raise AssertionError("Partial inventory was not rejected")
        tests.append("partial_inventory_rejected_before_production")

    report = {
        "schema_version": f"{SCHEMA_VERSION}.self_test",
        "status": "passed",
        "tests": tests,
        "production_subject_outputs_accessed": False,
        "temporary_root": str(temporary_root),
    }
    report_path = temporary_root / "audit_edge_scaling_selftest.json"
    temporary = report_path.with_name(report_path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    # These gates execute before any production diagnostic is written.  In
    # particular, a partial fit array cannot produce a partial audit artifact.
    assert_frozen_contract()
    subjects = expected_subjects_from_qc(OUTPUTS / "qc_subject_sets.tsv")
    inventory = exact_inventory(OUTPUTS / "subject_maps", subjects)
    _, mask = load_mask_vector()
    n_voxels = int(mask.sum())
    mask_hash = sha256(OUTPUTS / "common_analysis_mask.nii.gz")

    table, summary = audit_checkpoint_set(inventory, n_voxels, mask_hash)
    save_table_once(OUTPUTS / "edge_scaling_diagnostic.tsv", table)
    save_json_once(OUTPUTS / "edge_scaling_diagnostic_summary.json", summary)
    print(
        "Wrote or verified immutable edge-scaling diagnostic artifacts: "
        f"{len(table)} rows across {len(inventory)} subjects",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
