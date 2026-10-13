"""Materialize EP12's development-only MaleCNS view behind the role firewall.

The flat Feather file is not partitioned by EP12 role, so the trusted
materializer must physically scan its record batches.  It discards every
non-development focal row before constructing any matrix or durable summary.
Final-type focal connectivity is never written, summarized, or returned.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as parquet
import scipy
from scipy import sparse

from .policy import digest_bytes, digest_object
from .preflight import (
    ANNOTATION_FILE,
    BODY_STATS_FILE,
    SOURCE_ROOT,
    WEIGHTS_FILE,
)
from .synthetic import EPISODE_ROOT, SCRATCH_ROOT, atomic_json
from .yaml_subset import load_yaml_subset


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_filtered_batches(
    path: Path,
    *,
    id_column: str,
    allowed_ids: pa.Array,
    columns: list[str],
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        schema_names = list(reader.schema.names)
        indices = [schema_names.index(name) for name in columns]
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index).select(indices)
            mask = pc.is_in(batch[id_column], value_set=allowed_ids)
            filtered = batch.filter(mask)
            if filtered.num_rows:
                frames.append(filtered.to_pandas())
    return frames


def _partner_key_table(annotations: pd.DataFrame) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in annotations.itertuples(index=False):
        body = int(row.bodyId)
        provider_type = "" if pd.isna(row.type) else str(row.type).strip()
        if provider_type:
            result[body] = f"typed::{provider_type}"
            continue
        status = "missing" if pd.isna(row.status) else str(row.status).strip()
        status_label = (
            "missing" if pd.isna(row.statusLabel) else str(row.statusLabel).strip()
        )
        result[body] = f"untyped_status::{status}::{status_label}"
    return result


def materialize_development_view(
    *,
    output_dir: Path,
    scratch_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    scratch_dir = scratch_dir.resolve()
    allowed_output = (EPISODE_ROOT / "outputs").resolve()
    allowed_scratch = SCRATCH_ROOT.resolve()
    if output_dir != allowed_output and allowed_output not in output_dir.parents:
        raise ValueError("development records must stay under this EP12 outputs root")
    if scratch_dir != allowed_scratch and allowed_scratch not in scratch_dir.parents:
        raise ValueError("development matrices must stay under this EP12 scratch root")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"development output directory is not empty: {output_dir}")
    if scratch_dir.exists() and any(scratch_dir.iterdir()):
        raise FileExistsError(f"development scratch directory is not empty: {scratch_dir}")

    contract_path = EPISODE_ROOT / "outputs" / "executor" / "SCIENTIFIC_CONTRACT.yaml"
    contract = load_yaml_subset(contract_path)
    if contract["real_connectivity_access_authorized"] != "development_only":
        raise ValueError("scientific contract has not authorized development connectivity")
    if contract.get("final_connectivity_access_authorized", False) is not False:
        raise ValueError("final connectivity must remain locked")
    role_path = (
        EPISODE_ROOT
        / "outputs"
        / "prelaunch"
        / "source_preflight"
        / "annotation_role_manifest.json"
    )
    exposure_path = role_path.parent / "exposure_record.json"
    role_manifest = json.loads(role_path.read_text(encoding="utf-8"))
    role_by_type = {
        str(row["provider_type"]): str(row["role"])
        for row in role_manifest["rows"]
    }
    development_types = {
        provider_type
        for provider_type, role in role_by_type.items()
        if role == "development"
    }
    final_types = {
        provider_type for provider_type, role in role_by_type.items() if role == "final"
    }
    if development_types & final_types:
        raise ValueError("development/final type roles overlap")

    source_dir = SOURCE_ROOT / "source"
    annotation_columns = [
        "bodyId",
        "type",
        "somaSide",
        "status",
        "statusLabel",
        "somaLocation",
        "assignedOlHex1",
        "assignedOlHex2",
        "mcnsSerial",
        "somaNeuromere",
        "serialMotif",
    ]
    annotations = feather.read_table(
        source_dir / ANNOTATION_FILE,
        columns=annotation_columns,
        memory_map=True,
    ).to_pandas()
    type_values = annotations["type"].astype("string").fillna("").str.strip()
    side_values = annotations["somaSide"].astype("string").fillna("")
    status_values = annotations["status"].astype("string").fillna("")
    focal_mask = (
        type_values.isin(development_types)
        & side_values.isin(["L", "R"])
        & status_values.eq("Traced")
    )
    focal = annotations.loc[focal_mask].copy()
    focal["type"] = type_values[focal_mask].astype(str)
    focal["somaSide"] = side_values[focal_mask].astype(str)
    focal_ids = focal["bodyId"].astype(np.int64).to_numpy()
    if len(focal_ids) != len(np.unique(focal_ids)):
        raise ValueError("development focal body IDs are not unique")
    if not set(focal["type"]).issubset(development_types):
        raise ValueError("final type entered the development focal table")
    row_index = {int(body): index for index, body in enumerate(focal_ids)}
    allowed_ids = pa.array(focal_ids, type=pa.int64())

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    stats_frames = _read_filtered_batches(
        source_dir / BODY_STATS_FILE,
        id_column="body",
        allowed_ids=allowed_ids,
        columns=[
            "body",
            "pre",
            "post",
            "status_fine",
            "downstream",
            "synweight",
            "rank",
        ],
    )
    development_stats = (
        pd.concat(stats_frames, ignore_index=True)
        if stats_frames
        else pd.DataFrame(
            columns=[
                "body",
                "pre",
                "post",
                "status_fine",
                "downstream",
                "synweight",
                "rank",
            ]
        )
    )
    if not set(development_stats["body"].astype(int)).issubset(set(row_index)):
        raise ValueError("non-development body entered filtered body statistics")

    partner_key_by_body = _partner_key_table(
        annotations[["bodyId", "type", "status", "statusLabel"]]
    )
    vocabulary = sorted(set(partner_key_by_body.values()) | {"missing_annotation"})
    partner_index = {key: index for index, key in enumerate(vocabulary)}
    partner_column_by_body = {
        body: partner_index[key] for body, key in partner_key_by_body.items()
    }
    missing_annotation_column = partner_index["missing_annotation"]
    edge_rows: list[np.ndarray] = []
    edge_columns: list[np.ndarray] = []
    edge_weights: list[np.ndarray] = []
    with pa.memory_map(str(source_dir / WEIGHTS_FILE), "r") as source:
        reader = ipc.open_file(source)
        names = list(reader.schema.names)
        indices = [names.index(name) for name in ("body_pre", "body_post", "weight")]
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index).select(indices)
            mask = pc.is_in(batch["body_pre"], value_set=allowed_ids)
            filtered = batch.filter(mask)
            if not filtered.num_rows:
                continue
            frame = filtered.to_pandas()
            rows = frame["body_pre"].map(row_index)
            if rows.isna().any():
                raise ValueError("role firewall retained a non-development source body")
            partner_columns = (
                frame["body_post"]
                .map(partner_column_by_body)
                .fillna(missing_annotation_column)
            )
            edge_rows.append(rows.to_numpy(dtype=np.int64))
            edge_columns.append(partner_columns.to_numpy(dtype=np.int64))
            edge_weights.append(frame["weight"].to_numpy(dtype=np.int64))

    if not edge_rows:
        raise ValueError("no development outgoing connectivity was materialized")
    rows = np.concatenate(edge_rows)
    columns = np.concatenate(edge_columns)
    weights = np.concatenate(edge_weights)
    if np.any(weights < 0):
        raise ValueError("negative connectome weight encountered")
    matrix = sparse.coo_matrix(
        (weights, (rows, columns)),
        shape=(len(focal), len(vocabulary)),
        dtype=np.int64,
    ).tocsr()
    matrix.sum_duplicates()
    if matrix.data.size and np.any(matrix.data < 0):
        raise ValueError("aggregated development matrix contains negative counts")

    scratch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = scratch_dir / "development_outgoing_counts.npz"
    focal_path = scratch_dir / "development_focal_neurons.parquet"
    stats_path = scratch_dir / "development_body_statistics.parquet"
    vocabulary_path = scratch_dir / "raw_partner_vocabulary.json"
    sparse.save_npz(matrix_path, matrix, compressed=True)
    parquet.write_table(pa.Table.from_pandas(focal, preserve_index=False), focal_path)
    parquet.write_table(
        pa.Table.from_pandas(development_stats, preserve_index=False), stats_path
    )
    vocabulary_path.write_text(
        json.dumps(vocabulary, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    scratch_hashes = {
        path.name: _stream_sha256(path)
        for path in (matrix_path, focal_path, stats_path, vocabulary_path)
    }
    prior_exposure = json.loads(exposure_path.read_text(encoding="utf-8"))
    exposure = {
        "schema_version": "ep12.development_exposure.v1",
        "first_development_connectivity_access_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "authorization_source": "explicit_user_instruction_in_active_EP12_Codex_task",
        "annotation_values_opened": True,
        "development_body_statistics_values_opened": True,
        "development_connectome_weight_values_opened": True,
        "final_focal_connectivity_materialized_or_summarized": False,
        "final_connectivity_access_authorized": False,
        "flat_file_physical_scan_note": (
            "unpartitioned Arrow record batches were scanned inside the trusted role "
            "materializer; non-development focal rows were discarded before any "
            "matrix, summary, log, or returned value was constructed"
        ),
        "prior_prelaunch_exposure_sha256": digest_bytes(exposure_path.read_bytes()),
        "role_rows_hash": role_manifest["rows_hash"],
        "scientific_contract_sha256": digest_bytes(contract_path.read_bytes()),
    }
    manifest = {
        "schema_version": "ep12.development_materialization.v1",
        "episode_root": str(EPISODE_ROOT),
        "scratch_root": str(scratch_dir),
        "source_root": str(SOURCE_ROOT),
        "role_rows_hash": role_manifest["rows_hash"],
        "development_type_count": len(development_types),
        "final_type_count_locked": len(final_types),
        "development_focal_neuron_count": int(len(focal)),
        "development_focal_left_count": int((focal["somaSide"] == "L").sum()),
        "development_focal_right_count": int((focal["somaSide"] == "R").sum()),
        "development_weight_rows_retained": int(len(weights)),
        "development_observed_outgoing_weight": int(matrix.sum()),
        "raw_partner_key_count": int(len(vocabulary)),
        "matrix_shape": [int(value) for value in matrix.shape],
        "matrix_nonzero_cells": int(matrix.nnz),
        "scratch_artifact_sha256": scratch_hashes,
        "scratch_artifacts_are_development_only": True,
        "final_focal_connectivity_materialized_or_summarized": False,
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pyarrow": pa.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
            "numeric_thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
    }
    atomic_json(output_dir / "development_exposure.json", exposure)
    atomic_json(output_dir / "development_materialization.json", manifest)
    atomic_json(
        output_dir / "manifest.json",
        {
            "schema_version": "ep12.development_materialization_manifest.v1",
            "files": {
                name: digest_bytes((output_dir / name).read_bytes())
                for name in (
                    "development_exposure.json",
                    "development_materialization.json",
                )
            },
            "scratch_artifact_sha256": scratch_hashes,
            "final_focal_connectivity_materialized_or_summarized": False,
        },
    )
    return manifest


def materialize_partner_hierarchy(
    *,
    output_dir: Path,
    snapshot_dir: Path,
) -> dict[str, Any]:
    """Freeze annotation-only partner type parents for the hierarchy branch."""

    output_dir = output_dir.resolve()
    snapshot_dir = snapshot_dir.resolve()
    allowed_output = (EPISODE_ROOT / "outputs").resolve()
    allowed_scratch = SCRATCH_ROOT.resolve()
    if output_dir != allowed_output and allowed_output not in output_dir.parents:
        raise ValueError("hierarchy records must stay under this EP12 outputs root")
    if snapshot_dir != allowed_scratch and allowed_scratch not in snapshot_dir.parents:
        raise ValueError("hierarchy snapshot must stay under this EP12 scratch root")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"hierarchy output directory is not empty: {output_dir}")
    hierarchy_path = snapshot_dir / "partner_type_hierarchy.json"
    if hierarchy_path.exists():
        raise FileExistsError(f"partner hierarchy already exists: {hierarchy_path}")

    contract = load_yaml_subset(
        EPISODE_ROOT / "outputs" / "executor" / "SCIENTIFIC_CONTRACT.yaml"
    )
    if contract["real_connectivity_access_authorized"] != "development_only":
        raise ValueError("development execution is not authorized")
    if contract.get("final_connectivity_access_authorized", False) is not False:
        raise ValueError("final connectivity must remain locked")

    table = feather.read_table(
        SOURCE_ROOT / "source" / ANNOTATION_FILE,
        columns=["type", "subclass", "class"],
        memory_map=True,
    ).to_pandas()
    table["type"] = table["type"].astype("string").fillna("").str.strip()
    table = table.loc[table["type"].ne("")].copy()
    mapping: dict[str, dict[str, str]] = {}
    conflict_counts = {"subclass": 0, "class": 0}
    for provider_type, group in table.groupby("type", sort=True):
        record: dict[str, str] = {}
        for field in ("subclass", "class"):
            values = group[field].astype("string").fillna("").str.strip()
            counts = values.loc[values.ne("")].value_counts()
            if counts.empty:
                record[field] = ""
                continue
            top_count = int(counts.max())
            candidates = sorted(str(value) for value in counts[counts.eq(top_count)].index)
            record[field] = candidates[0]
            if len(counts) > 1:
                conflict_counts[field] += 1
        mapping[str(provider_type)] = record

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    hierarchy_path.write_text(
        json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "ep12.partner_type_hierarchy.v1",
        "source": str(SOURCE_ROOT / "source" / ANNOTATION_FILE),
        "source_fields": ["type", "subclass", "class"],
        "selection": "modal_nonempty_annotation_with_lexical_tie_break",
        "provider_type_count": len(mapping),
        "conflicting_type_counts": conflict_counts,
        "scratch_artifact": str(hierarchy_path),
        "scratch_artifact_sha256": _stream_sha256(hierarchy_path),
        "annotation_only": True,
        "connectivity_values_accessed": False,
        "final_connectivity_accessed": False,
    }
    atomic_json(output_dir / "partner_hierarchy_materialization.json", report)
    atomic_json(
        output_dir / "manifest.json",
        {
            "schema_version": "ep12.partner_type_hierarchy_manifest.v1",
            "files": {
                "partner_hierarchy_materialization.json": digest_bytes(
                    (output_dir / "partner_hierarchy_materialization.json").read_bytes()
                )
            },
            "scratch_artifact_sha256": report["scratch_artifact_sha256"],
            "final_connectivity_accessed": False,
        },
    )
    return report
