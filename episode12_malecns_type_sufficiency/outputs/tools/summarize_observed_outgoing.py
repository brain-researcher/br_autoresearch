#!/usr/bin/env python3
"""Describe the fixed, already-open EP12 development outgoing snapshot.

This separate descriptive calculation does not call the historical T/U/M
controller, read body statistics, reconstruct incoming edges, or open final
focal profiles. Run only in the coordinator's selected Slurm allocation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time

import numpy as np
import pyarrow
import pyarrow.parquet as parquet
import scipy
from scipy import sparse


EPISODE = "episode12_malecns_type_sufficiency"
EXPECTED_SHAPE = (84735, 11771)
EXPECTED_NNZ = 4688114


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def deadline(_signum: int, _frame: object) -> None:
    raise TimeoutError("selected descriptive wall-time limit reached")


def read_role_rows(root: Path, materialization: dict) -> dict[str, dict]:
    path = root / "outputs/prelaunch/source_preflight/annotation_role_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest["rows"]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    actual_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    require(actual_hash == manifest["rows_hash"] == materialization["role_rows_hash"],
            "frozen role rows differ from the materialized snapshot")
    require(manifest["assignment_unit"] == "whole_provider_type", "wrong role unit")
    require(len(rows) == 1020 and manifest["development_type_count"] == 816
            and manifest["final_type_count"] == 204, "whole-type counts changed")
    require(len({row["provider_type"] for row in rows}) == len(rows), "duplicate type role")
    require(all(row["role"] in {"development", "final"} for row in rows), "unknown role")
    development = {row["provider_type"]: row for row in rows if row["role"] == "development"}
    require(len(development) == 816, "development type roster changed")
    return development


def checked_snapshot_file(snapshot: Path, name: str, materialization: dict) -> Path:
    """Reuse the existing snapshot identities; never create a new identity chain."""
    path = snapshot / name
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    require(digest.hexdigest() == materialization["scratch_artifact_sha256"][name],
            f"recorded snapshot identity mismatch: {name}")
    return path


def side_description(matrix: sparse.csr_matrix, unknown_columns: np.ndarray) -> tuple[dict, np.ndarray | None]:
    """Return descriptive TV on positive-mass rows, with every zero retained in accounting."""
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    positive = totals > 0
    result = {
        "neurons": int(matrix.shape[0]),
        "positive_mass_neurons": int(positive.sum()),
        "zero_mass_neurons": int((~positive).sum()),
        "outgoing_count": int(totals.sum()),
        "mean_tv_to_side_centroid": None,
        "mean_untyped_or_missing_fraction": None,
    }
    if not positive.any():
        return result, None
    profiles = matrix[positive].astype(np.float64)
    profiles.data *= np.repeat(1.0 / totals[positive], np.diff(profiles.indptr))
    centroid = np.asarray(profiles.mean(axis=0)).ravel()
    require(np.isfinite(centroid).all() and abs(float(centroid.sum()) - 1.0) < 1e-10,
            "normalized side centroid is not a probability vector")
    # For probability vectors, TV(p,q)=1-sum_j min(p_j,q_j). Sparse zeros
    # contribute zero to that overlap, so no neuron-by-partner dense array is needed.
    overlap = profiles.copy()
    overlap.data = np.minimum(profiles.data, centroid[profiles.indices])
    distances = 1.0 - np.asarray(overlap.sum(axis=1)).ravel()
    require(np.isfinite(distances).all() and np.all(distances >= -1e-10)
            and np.all(distances <= 1.0 + 1e-10), "TV arithmetic outside its range")
    # Only floating-point roundoff at the mathematical endpoints is clipped.
    result["mean_tv_to_side_centroid"] = float(np.clip(distances, 0.0, 1.0).mean())
    result["mean_untyped_or_missing_fraction"] = float(centroid[unknown_columns].sum())
    return result, centroid


def check_equations() -> dict:
    """One tiny closed-form check of sparse TV and undefined zero-count rows."""
    counts = np.array([[3, 0, 1], [0, 2, 2], [0, 0, 0]], dtype=np.int64)
    unknown = np.array([False, False, True])
    result, centroid = side_description(sparse.csr_matrix(counts), unknown)
    expected = np.array([0.375, 0.25, 0.375])
    dense_profiles = counts[:2] / counts[:2].sum(axis=1, keepdims=True)
    dense_tv = float((0.5 * np.abs(dense_profiles - dense_profiles.mean(axis=0)).sum(axis=1)).mean())
    require(np.allclose(centroid, expected, atol=1e-14, rtol=0), "closed-form centroid check failed")
    require(abs(result["mean_tv_to_side_centroid"] - 0.375) < 1e-14
            and abs(result["mean_tv_to_side_centroid"] - dense_tv) < 1e-14,
            "sparse/dense all-bin TV check failed")
    require(result["neurons"] == 3 and result["positive_mass_neurons"] == 2
            and result["zero_mass_neurons"] == 1
            and abs(result["mean_untyped_or_missing_fraction"] - 0.375) < 1e-14,
            "zero-count or unknown-bin accounting check failed")
    empty, undefined = side_description(sparse.csr_matrix(np.zeros((1, 3), dtype=np.int64)), unknown)
    require(undefined is None and empty["mean_tv_to_side_centroid"] is None
            and empty["zero_mass_neurons"] == 1, "all-zero side was incorrectly described")
    singleton, _ = side_description(sparse.csr_matrix([[0, 4, 0]], dtype=np.int64), unknown)
    require(singleton["mean_tv_to_side_centroid"] == 0.0, "singleton dispersion is not zero")
    return {"passed": True, "check": "closed_form_sparse_dense_tv_zero_mass_unknown_bin_singleton",
            "sparse_and_dense_tv": dense_tv}


def summarize(root: Path, snapshot: Path, output: Path, started: float) -> dict:
    metadata_path = root / "outputs/development_run001/materialization/development_materialization.json"
    materialization = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(materialization["scratch_root"] == str(snapshot), "unexpected snapshot location")
    require(materialization["scratch_artifacts_are_development_only"] is True
            and materialization["final_focal_connectivity_materialized_or_summarized"] is False,
            "materialization does not establish development-only focal inputs")
    roles = read_role_rows(root, materialization)
    focal_path = checked_snapshot_file(snapshot, "development_focal_neurons.parquet", materialization)
    columns = ["bodyId", "type", "somaSide", "status"]
    schema = parquet.read_schema(focal_path)
    require(set(columns).issubset(schema.names), "required focal columns are missing")
    # This metadata-only read validates every focal role before opening count values.
    focal = parquet.read_table(focal_path, columns=columns).to_pydict()
    require(len(focal["bodyId"]) == EXPECTED_SHAPE[0], "focal row count changed")
    require(len(set(focal["bodyId"])) == EXPECTED_SHAPE[0], "duplicate focal body ID")
    require(all(isinstance(value, str) and value in roles for value in focal["type"]),
            "non-development or missing focal type")
    require(set(focal["type"]) == set(roles), "not every frozen development type is present")
    require(all(value in {"L", "R"} for value in focal["somaSide"]), "invalid focal side")
    require(all(value == "Traced" for value in focal["status"]), "focal status changed")
    groups: dict[tuple[str, str], list[int]] = {}
    for index, (provider_type, side) in enumerate(zip(focal["type"], focal["somaSide"])):
        groups.setdefault((provider_type, side), []).append(index)
    for provider_type, role in roles.items():
        for side, count_key in (("L", "left_neurons"), ("R", "right_neurons")):
            require(len(groups.get((provider_type, side), [])) == role[count_key],
                    "type/side support differs from the frozen annotation roster")
    vocabulary_path = checked_snapshot_file(snapshot, "raw_partner_vocabulary.json", materialization)
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    require(isinstance(vocabulary, list) and len(vocabulary) == EXPECTED_SHAPE[1]
            and all(isinstance(key, str) for key in vocabulary)
            and len(set(vocabulary)) == len(vocabulary), "partner vocabulary changed")
    require("missing_annotation" in vocabulary and all(
        key.startswith("typed::") or key.startswith("untyped_status::") or key == "missing_annotation"
        for key in vocabulary), "unrecognized partner accounting category")
    unknown_columns = np.asarray([not key.startswith("typed::") for key in vocabulary])
    matrix_path = checked_snapshot_file(snapshot, "development_outgoing_counts.npz", materialization)
    matrix = sparse.load_npz(matrix_path).tocsr()
    require(matrix.shape == EXPECTED_SHAPE and matrix.nnz == EXPECTED_NNZ, "matrix dimensions changed")
    require(matrix.has_canonical_format, "count matrix is not canonical CSR")
    require(np.issubdtype(matrix.dtype, np.integer) and np.all(matrix.data >= 0), "invalid raw counts")
    total_counts = np.asarray(matrix.sum(axis=1)).ravel()
    require(np.all(total_counts >= 0)
            and int(total_counts.sum()) == materialization["development_observed_outgoing_weight"],
            "count totals differ from the fixed materialization")

    rows = []
    for provider_type in sorted(roles):
        row = {"provider_type": provider_type}
        centroids = []
        for side in ("L", "R"):
            result, centroid = side_description(matrix[groups[(provider_type, side)]], unknown_columns)
            row.update({f"{side}_{key}": value for key, value in result.items()})
            centroids.append(centroid)
        if all(value is not None for value in centroids):
            row["between_side_centroid_tv"] = float(0.5 * np.abs(centroids[0] - centroids[1]).sum())
            row["status"] = "both_sides_described"
        else:
            row["between_side_centroid_tv"] = None
            row["status"] = "zero_mass_prevents_one_or_both_side_descriptions"
        rows.append(row)
        if len(rows) % 100 == 0:
            write_json(output / "progress.json", {"complete": False, "types_completed": len(rows),
                       "elapsed_seconds": time.monotonic() - started})
    require(len(rows) == 816, "incomplete all-development table")
    side_rows = []
    bilateral_rows = []
    for row in rows:
        for side in ("L", "R"):
            side_row = {"provider_type": row["provider_type"], "side": side}
            side_row.update({key[2:]: value for key, value in row.items() if key.startswith(side + "_")})
            side_row["status"] = ("positive_mass_profiles_described" if side_row["positive_mass_neurons"]
                                  else "zero_mass_composition_undefined")
            side_rows.append(side_row)
        bilateral_rows.append({
            "provider_type": row["provider_type"],
            **{key: row[key] for key in ("L_neurons", "L_positive_mass_neurons", "L_zero_mass_neurons",
                                        "R_neurons", "R_positive_mass_neurons", "R_zero_mass_neurons")},
            "between_side_centroid_tv": row["between_side_centroid_tv"],
            "status": row["status"],
        })
    require(len(side_rows) == 1632 and len(bilateral_rows) == 816, "incomplete side/bilateral accounting")
    for filename, table_rows in (("outgoing_type_side_descriptions.csv", side_rows),
                                 ("outgoing_bilateral_descriptions.csv", bilateral_rows)):
        with (output / filename).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
            writer.writeheader()
            writer.writerows(table_rows)
            handle.flush()
            os.fsync(handle.fileno())
    report = {
        "complete": True,
        "scope": "descriptive_already_open_development_outgoing_only_one_male",
        "snapshot": str(snapshot),
        "types_reported": len(rows),
        "type_side_rows_reported": len(side_rows),
        "type_sides_with_positive_mass_profiles": sum(row["positive_mass_neurons"] > 0 for row in side_rows),
        "types_with_two_positive_mass_sides": sum(row["status"] == "both_sides_described" for row in rows),
        "focal_neurons": EXPECTED_SHAPE[0],
        "partner_bins": EXPECTED_SHAPE[1],
        "zero_mass_neurons": int((total_counts == 0).sum()),
        "matrix_nonzero_cells": matrix.nnz,
        "neuron_weighting": "equal_among_positive_mass_neurons_within_type_and_side",
        "zero_mass_rule": "undefined_composition_counted_explicitly_never_imputed_or_replaced",
        "centroid_rule": "neuron_equal_arithmetic_mean_in_full_raw_partner_vocabulary",
        "within_side_metric": "mean_TV_of_each_positive_mass_profile_to_its_own_side_centroid",
        "between_side_metric": "TV_between_the_two_neuron_equal_side_centroids",
        "interpretation": "finite_sample_descriptive_dispersion_not_heldout_prediction_or_a_group_test",
        "self_inclusion": "each_neuron_contributes_to_its_own_side_centroid_so_spread_is_in_sample_and_sample_size_dependent",
        "final_focal_profiles_opened": False,
        "incoming_profiles_constructed": False,
        "historical_search_or_generated_chain_resumed": False,
        "elapsed_seconds": time.monotonic() - started,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "pyarrow": pyarrow.__version__},
    }
    write_json(output / "descriptive_report.json", report)
    write_json(output / "progress.json", {"complete": True, "types_completed": len(rows),
               "elapsed_seconds": time.monotonic() - started})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-seconds", type=int, default=1680)
    parser.add_argument("--check-equations-only", action="store_true")
    args = parser.parse_args()
    require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname(),
            "descriptive execution requires the selected compute-node Slurm shell")
    if args.check_equations_only:
        require(args.output_dir is None, "the closed-form check does not write a run directory")
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(30)
        try:
            print(json.dumps(check_equations(), sort_keys=True), flush=True)
        finally:
            signal.alarm(0)
        return
    require(args.output_dir is not None, "the selected descriptive pass requires --output-dir")
    require(1 <= args.max_seconds <= 1680, "selected internal cap is at most 1680 seconds")
    root = Path(os.environ["OAK"]) / "users/zijiao/br_autoresearch" / EPISODE
    snapshot = Path(os.environ["SCRATCH"]) / "br_autoresearch" / EPISODE / "development_snapshot_v1"
    output = args.output_dir.resolve()
    require((root / "outputs").resolve() in output.parents, "output must remain in this episode")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    try:
        summarize(root, snapshot, output, started)
    except Exception as error:
        write_json(output / "failure.json", {"complete": False, "error_type": type(error).__name__,
                   "message": str(error), "elapsed_seconds": time.monotonic() - started})
        raise
    finally:
        signal.alarm(0)
    print("EP12 descriptive all-development outgoing pass complete", flush=True)


if __name__ == "__main__":
    main()
