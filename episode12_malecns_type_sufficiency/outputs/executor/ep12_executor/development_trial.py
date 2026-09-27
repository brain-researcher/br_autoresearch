"""Run a complete real-data T/U/M trial on EP12 development roles only."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet
from scipy import sparse

from .policy import EpisodePolicy, digest_bytes, digest_object
from .contract_roles import active_record_fields, validate_contract_provenance
from .scientific_models import reciprocal_transfer, seed_from
from .scientific_qualification import _contract_settings
from .runtime import EPISODE_ROOT, SCRATCH_ROOT, atomic_json
from .yaml_subset import load_yaml_subset


ENDPOINT_BINS = (
    "other_typed",
    "unknown",
    "untyped",
    "fragment",
    "proofreading",
    "missing_annotation",
    "out_of_vocabulary",
)


def _endpoint_bin(raw_key: str) -> str:
    if raw_key == "missing_annotation":
        return "missing_annotation"
    if raw_key.startswith("typed::"):
        return "other_typed"
    if not raw_key.startswith("untyped_status::"):
        return "out_of_vocabulary"
    _, status, status_label = raw_key.split("::", 2)
    normalized = f"{status} {status_label}".lower()
    if "orphan" in normalized or "leaves" in normalized:
        return "fragment"
    if status in {"Assign", "Anchor"} or any(
        token in normalized
        for token in ("hard to trace", "partially traced", "anchor", "assign")
    ):
        return "proofreading"
    if status == "Traced":
        return "untyped"
    return "unknown"


def _fixed_covariates(
    focal: pd.DataFrame,
    stats: pd.DataFrame,
    *,
    spline_df: int = 0,
) -> np.ndarray:
    stats = stats.drop_duplicates("body", keep="first").set_index("body")
    joined = focal[["bodyId"]].join(stats, on="bodyId")
    soma = focal["somaLocation"].apply(
        lambda value: list(value) if isinstance(value, (list, tuple, np.ndarray)) else []
    )
    raw_columns: list[np.ndarray] = []
    for index in range(3):
        raw_columns.append(
            np.asarray(
                [float(value[index]) if len(value) > index else np.nan for value in soma],
                dtype=float,
            )
            / 100000.0
        )
    for name, scale in (
        ("assignedOlHex1", 100.0),
        ("assignedOlHex2", 100.0),
        ("mcnsSerial", 100.0),
    ):
        raw_columns.append(pd.to_numeric(focal[name], errors="coerce").to_numpy(float) / scale)
    for name in ("pre", "post", "downstream", "synweight"):
        values = pd.to_numeric(joined[name], errors="coerce").to_numpy(float)
        raw_columns.append(np.log1p(np.maximum(values, 0.0)) / 10.0)
    raw_columns.append(
        pd.to_numeric(joined["rank"], errors="coerce").to_numpy(float) / 100000.0
    )
    values = np.column_stack(raw_columns)
    missing = ~np.isfinite(values)
    values[missing] = 0.0
    design = [values, missing.astype(float)]
    if spline_df:
        if spline_df not in {3, 5}:
            raise ValueError("nuisance_spline_df must be one of 0, 3, or 5")
        bounded = np.tanh(values[:, :6])
        design.extend(bounded**degree for degree in range(2, spline_df + 1))
    return np.column_stack(design)


def _collapse_vocabulary(
    matrix: sparse.csr_matrix,
    focal_types: np.ndarray,
    raw_vocabulary: list[str],
    *,
    strategy: str,
    minimum_development_type_count: int,
    rare_partner_mass_fraction: float,
    hierarchy_depth: int = 1,
    partner_hierarchy: dict[str, dict[str, str]] | None = None,
) -> tuple[sparse.csr_matrix, list[str], dict[str, Any]]:
    type_names = sorted(set(str(value) for value in focal_types))
    type_index = {name: index for index, name in enumerate(type_names)}
    group = sparse.coo_matrix(
        (
            np.ones(len(focal_types), dtype=np.int8),
            (
                np.fromiter(
                    (type_index[str(value)] for value in focal_types),
                    dtype=np.int64,
                    count=len(focal_types),
                ),
                np.arange(len(focal_types), dtype=np.int64),
            ),
        ),
        shape=(len(type_names), matrix.shape[0]),
    ).tocsr()
    presence = group @ matrix.sign()
    support = np.asarray((presence > 0).sum(axis=0)).ravel()
    mass = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
    typed_total = float(
        sum(value for key, value in zip(raw_vocabulary, mass) if key.startswith("typed::"))
    )
    typed_statistics = {
        key: (int(type_support), float(type_mass) / typed_total if typed_total else 0.0)
        for key, type_support, type_mass in zip(raw_vocabulary, support, mass)
        if key.startswith("typed::")
    }
    if strategy == "development_frequency_pooling":
        retained_typed = {
            key
            for key, (type_support, type_mass_fraction) in typed_statistics.items()
            if type_support >= minimum_development_type_count
            and type_mass_fraction >= rare_partner_mass_fraction
        }
    elif strategy == "explicit_unknown_untyped_fragment_and_proofreading_bins":
        retained_typed = {
            key
            for key, (_, type_mass_fraction) in typed_statistics.items()
            if type_mass_fraction >= rare_partner_mass_fraction
        }
    elif strategy == "bounded_hierarchical_partner_categories":
        if partner_hierarchy is None:
            raise ValueError("hierarchical vocabulary requires the frozen partner hierarchy")
        if hierarchy_depth not in {1, 2}:
            raise ValueError("partner hierarchy depth must be 1 or 2")
        retained_typed = {
            key
            for key, (type_support, type_mass_fraction) in typed_statistics.items()
            if type_support >= minimum_development_type_count
            and type_mass_fraction >= rare_partner_mass_fraction
        }
    else:
        raise ValueError(f"unknown partner vocabulary strategy: {strategy}")

    hierarchy_bins: set[str] = set()
    hierarchy_field = "subclass" if hierarchy_depth == 1 else "class"
    if strategy == "bounded_hierarchical_partner_categories":
        for key in typed_statistics:
            if key in retained_typed:
                continue
            provider_type = key.removeprefix("typed::")
            label = str(partner_hierarchy.get(provider_type, {}).get(hierarchy_field, "")).strip()
            hierarchy_bins.add(f"hierarchy_{hierarchy_depth}::{label or 'unclassified'}")
    collapsed_vocabulary = (
        sorted(retained_typed) + sorted(hierarchy_bins) + list(ENDPOINT_BINS)
    )
    collapsed_index = {key: index for index, key in enumerate(collapsed_vocabulary)}
    target_columns: list[int] = []
    for key in raw_vocabulary:
        if key in retained_typed:
            target_key = key
        elif key.startswith("typed::") and strategy == "bounded_hierarchical_partner_categories":
            provider_type = key.removeprefix("typed::")
            label = str(partner_hierarchy.get(provider_type, {}).get(hierarchy_field, "")).strip()
            target_key = f"hierarchy_{hierarchy_depth}::{label or 'unclassified'}"
        else:
            target_key = _endpoint_bin(key)
        target_columns.append(collapsed_index[target_key])
    mapping = sparse.coo_matrix(
        (
            np.ones(len(raw_vocabulary), dtype=np.int8),
            (
                np.arange(len(raw_vocabulary), dtype=np.int64),
                np.asarray(target_columns, dtype=np.int64),
            ),
        ),
        shape=(len(raw_vocabulary), len(collapsed_vocabulary)),
    ).tocsr()
    collapsed = (matrix @ mapping).tocsr()
    return collapsed, collapsed_vocabulary, {
        "partner_vocabulary": strategy,
        "minimum_development_type_count": minimum_development_type_count,
        "rare_partner_mass_fraction": rare_partner_mass_fraction,
        "partner_hierarchy_depth": hierarchy_depth,
        "partner_hierarchy_field": (
            hierarchy_field if strategy == "bounded_hierarchical_partner_categories" else None
        ),
        "raw_partner_key_count": len(raw_vocabulary),
        "retained_typed_partner_count": len(retained_typed),
        "hierarchy_bin_count": len(hierarchy_bins),
        "collapsed_partner_count": len(collapsed_vocabulary),
        "retained_typed_partners": sorted(
            key.removeprefix("typed::") for key in retained_typed
        ),
        "endpoint_mapping": (
            "typed partners failing the frozen support/mass branch pool to "
            "other_typed; untyped provider status/statusLabel maps deterministically "
            "to unknown/untyped/fragment/proofreading"
        ),
    }


def run_development_trial(
    *,
    output_dir: Path,
    snapshot_dir: Path,
    configuration: dict[str, Any] | None = None,
    contract_path: Path | None = None,
    contract_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    snapshot_dir = snapshot_dir.resolve()
    allowed_output = (EPISODE_ROOT / "outputs").resolve()
    allowed_scratch = SCRATCH_ROOT.resolve()
    if (
        output_dir != allowed_output
        and allowed_output not in output_dir.parents
        and output_dir != allowed_scratch
        and allowed_scratch not in output_dir.parents
    ):
        raise ValueError("development trial outputs must stay under EP12 outputs or scratch")
    if snapshot_dir != allowed_scratch and allowed_scratch not in snapshot_dir.parents:
        raise ValueError("development snapshot must stay under this EP12 scratch root")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"development trial output directory is not empty: {output_dir}")

    contract_path = (
        contract_path.resolve(strict=True)
        if contract_path is not None
        else EPISODE_ROOT / "outputs" / "executor" / "SCIENTIFIC_CONTRACT.yaml"
    )
    contract = load_yaml_subset(contract_path)
    contract_fields: dict[str, Any] = {}
    if contract_provenance is not None:
        contract_fields = active_record_fields(
            validate_contract_provenance(contract_provenance)
        )
    settings = _contract_settings(contract)
    if contract["real_connectivity_access_authorized"] != "development_only":
        raise ValueError("development execution is not authorized")
    if contract.get("final_connectivity_access_authorized", False) is not False:
        raise ValueError("final connectivity must remain locked")
    policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")
    if configuration is None:
        configuration = {
            "trial_id": "development_trial_001",
            "stage": "initial_real_configuration",
            "partner_vocabulary": "development_frequency_pooling",
            "partner_minimum_development_type_count": 2,
            "rare_partner_pooling_mass_fraction": 0.01,
            "partner_hierarchy_depth": 1,
            "representation": "count_aware_log_ratio",
            "rank": 2,
            "composition_pseudocount": 0.5,
            "nuisance": list(policy.nuisance_terms),
            "nuisance_spline_df": 0,
            "covariance": "shrinkage",
            "covariance_rank": 2,
            "component_count": 2,
            "regularization": 0.1,
        }
    configuration = dict(configuration)
    trial_id = str(configuration["trial_id"])
    matrix = sparse.load_npz(snapshot_dir / "development_outgoing_counts.npz").tocsr()
    focal = parquet.read_table(snapshot_dir / "development_focal_neurons.parquet").to_pandas()
    stats = parquet.read_table(snapshot_dir / "development_body_statistics.parquet").to_pandas()
    raw_vocabulary = json.loads(
        (snapshot_dir / "raw_partner_vocabulary.json").read_text(encoding="utf-8")
    )
    if matrix.shape != (len(focal), len(raw_vocabulary)):
        raise ValueError("development snapshot dimensions disagree")
    hierarchy_path = snapshot_dir / "partner_type_hierarchy.json"
    partner_hierarchy = (
        json.loads(hierarchy_path.read_text(encoding="utf-8"))
        if hierarchy_path.is_file()
        else None
    )
    collapsed, vocabulary, vocabulary_record = _collapse_vocabulary(
        matrix,
        focal["type"].astype(str).to_numpy(),
        raw_vocabulary,
        strategy=str(configuration["partner_vocabulary"]),
        minimum_development_type_count=int(
            configuration["partner_minimum_development_type_count"]
        ),
        rare_partner_mass_fraction=float(
            configuration["rare_partner_pooling_mass_fraction"]
        ),
        hierarchy_depth=int(configuration["partner_hierarchy_depth"]),
        partner_hierarchy=partner_hierarchy,
    )
    covariates = _fixed_covariates(
        focal,
        stats,
        spline_df=int(configuration["nuisance_spline_df"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "type_direction_receipts.jsonl"
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    rows: list[dict[str, Any]] = []
    unscorable: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with receipt_path.open("x", encoding="utf-8") as handle:
        for provider_type in sorted(focal["type"].astype(str).unique()):
            type_mask = focal["type"].astype(str).eq(provider_type).to_numpy()
            left = type_mask & focal["somaSide"].astype(str).eq("L").to_numpy()
            right = type_mask & focal["somaSide"].astype(str).eq("R").to_numpy()
            record: dict[str, Any] = {
                "provider_type": provider_type,
                "left_neurons": int(left.sum()),
                "right_neurons": int(right.sum()),
            }
            left_counts = collapsed[left].toarray().astype(np.int64)
            right_counts = collapsed[right].toarray().astype(np.int64)
            zero_left = int(np.sum(left_counts.sum(axis=1) == 0))
            zero_right = int(np.sum(right_counts.sum(axis=1) == 0))
            if zero_left or zero_right:
                record.update(
                    {
                        "status": "unscorable",
                        "reason_code": "zero_observed_target_count",
                        "zero_count_left_neurons": zero_left,
                        "zero_count_right_neurons": zero_right,
                    }
                )
                unscorable.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                continue
            component_count = int(configuration["component_count"])
            source_folds = int(settings["source_cv_folds"])
            minimum_source_neurons = component_count * source_folds
            if int(left.sum()) < minimum_source_neurons or int(right.sum()) < minimum_source_neurons:
                record.update(
                    {
                        "status": "unscorable",
                        "reason_code": "source_side_below_frozen_support_after_integrity_filter",
                        "minimum_source_neurons_for_component_cv": minimum_source_neurons,
                    }
                )
                unscorable.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                continue
            try:
                result = reciprocal_transfer(
                    left_counts,
                    covariates[left],
                    right_counts,
                    covariates[right],
                    seed=seed_from("ep12-development", trial_id, provider_type),
                    integration_draws=int(settings["integration_draws"]),
                    folds=int(settings["source_cv_folds"]),
                    latent_rank=int(configuration["rank"]),
                    ridge=float(configuration["regularization"]),
                    pseudocount=float(configuration["composition_pseudocount"]),
                    representation=str(configuration["representation"]),
                    covariance_mode=str(configuration["covariance"]),
                    covariance_rank=int(configuration["covariance_rank"]),
                    component_candidates=(component_count,),
                )
                record.update(
                    {
                        "status": "valid_complete_T_U_M_triplet",
                        "left_to_right_delta": float(
                            result["left_to_right"]["mean_delta_M_minus_reference"]
                        ),
                        "right_to_left_delta": float(
                            result["right_to_left"]["mean_delta_M_minus_reference"]
                        ),
                        "left_to_right_scores": result["left_to_right"]["mean_scores"],
                        "right_to_left_scores": result["right_to_left"]["mean_scores"],
                        "source_fit_metadata": result["source_fit_metadata"],
                    }
                )
                rows.append(record)
            except Exception as exc:
                record.update(
                    {
                        "status": "technical_failure",
                        "failure_type": type(exc).__name__,
                        "failure_message": str(exc),
                    }
                )
                failures.append(
                    {
                        "provider_type": provider_type,
                        "failure_type": type(exc).__name__,
                        "failure_message": str(exc),
                    }
                )
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
        os.fsync(handle.fileno())

    if rows:
        delta_values = np.asarray(
            [[row["left_to_right_delta"], row["right_to_left_delta"]] for row in rows]
        )
        aggregate = {
            "valid_type_count": len(rows),
            "unscorable_type_count": len(unscorable),
            "technical_failure_count": len(failures),
            "mean_left_to_right_delta": float(delta_values[:, 0].mean()),
            "mean_right_to_left_delta": float(delta_values[:, 1].mean()),
            "mean_bilateral_delta": float(delta_values.mean()),
            "types_with_both_point_estimates_above_margin": int(
                np.sum(np.all(delta_values > float(policy.meaningful_margin), axis=1))
            ),
        }
    else:
        aggregate = {
            "valid_type_count": 0,
            "unscorable_type_count": len(unscorable),
            "technical_failure_count": len(failures),
        }
    report = {
        "schema_version": "ep12.development_trial.v1",
        "trial_id": trial_id,
        "status": "completed" if not failures else "completed_with_technical_failures",
        "data_role": "development_only",
        "final_connectivity_accessed": False,
        "configuration": {
            **configuration,
            **vocabulary_record,
            "component_count_candidates": [int(configuration["component_count"])],
            "integration_draws": int(settings["integration_draws"]),
            "source_cross_validation_folds": int(settings["source_cv_folds"]),
            "configuration_sha256": digest_object(configuration),
        },
        "aggregate": aggregate,
        "unscorable": unscorable,
        "failures": failures,
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        **contract_fields,
        "scientific_contract_id": contract.get("contract_id"),
    }
    atomic_json(output_dir / "trial_report.json", report)
    with (output_dir / "result_table.csv").open("x", encoding="utf-8", newline="") as handle:
        fields = [
            "provider_type",
            "left_neurons",
            "right_neurons",
            "left_to_right_delta",
            "right_to_left_delta",
            "left_source_U",
            "left_source_M",
            "right_source_U",
            "right_source_M",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metadata = row["source_fit_metadata"]
            writer.writerow(
                {
                    "provider_type": row["provider_type"],
                    "left_neurons": row["left_neurons"],
                    "right_neurons": row["right_neurons"],
                    "left_to_right_delta": row["left_to_right_delta"],
                    "right_to_left_delta": row["right_to_left_delta"],
                    "left_source_U": metadata["left"]["U"],
                    "left_source_M": metadata["left"]["M"],
                    "right_source_U": metadata["right"]["U"],
                    "right_source_M": metadata["right"]["M"],
                }
            )
    atomic_json(
        output_dir / "manifest.json",
        {
            "schema_version": "ep12.development_trial_manifest.v1",
            "files": {
                name: digest_bytes((output_dir / name).read_bytes())
                for name in (
                    "type_direction_receipts.jsonl",
                    "trial_report.json",
                    "result_table.csv",
                )
            },
            "final_connectivity_accessed": False,
        },
    )
    return report


def reconcile_zero_count_failures(
    *,
    source_trial_dir: Path,
    output_dir: Path,
    snapshot_dir: Path,
) -> dict[str, Any]:
    """Reclassify only verified zero-count failures without changing scores."""

    source_trial_dir = source_trial_dir.resolve()
    output_dir = output_dir.resolve()
    snapshot_dir = snapshot_dir.resolve()
    allowed_output = (EPISODE_ROOT / "outputs").resolve()
    allowed_scratch = SCRATCH_ROOT.resolve()
    for path in (source_trial_dir, output_dir):
        if path != allowed_output and allowed_output not in path.parents:
            raise ValueError("trial reconciliation must stay under this EP12 outputs root")
    if snapshot_dir != allowed_scratch and allowed_scratch not in snapshot_dir.parents:
        raise ValueError("development snapshot must stay under this EP12 scratch root")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"reconciled output directory is not empty: {output_dir}")

    source_report_path = source_trial_dir / "trial_report.json"
    source_receipt_path = source_trial_dir / "type_direction_receipts.jsonl"
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    matrix = sparse.load_npz(snapshot_dir / "development_outgoing_counts.npz").tocsr()
    focal = parquet.read_table(snapshot_dir / "development_focal_neurons.parquet").to_pandas()
    raw_vocabulary = json.loads(
        (snapshot_dir / "raw_partner_vocabulary.json").read_text(encoding="utf-8")
    )
    collapsed, _, _ = _collapse_vocabulary(
        matrix,
        focal["type"].astype(str).to_numpy(),
        raw_vocabulary,
        strategy=str(source_report["configuration"]["partner_vocabulary"]),
        minimum_development_type_count=int(
            source_report["configuration"]["minimum_development_type_count"]
        ),
        rare_partner_mass_fraction=float(
            source_report["configuration"]["rare_partner_mass_fraction"]
        ),
        hierarchy_depth=int(
            source_report["configuration"].get("partner_hierarchy_depth", 1)
        ),
        partner_hierarchy=(
            json.loads(
                (snapshot_dir / "partner_type_hierarchy.json").read_text(encoding="utf-8")
            )
            if (snapshot_dir / "partner_type_hierarchy.json").is_file()
            else None
        ),
    )
    receipts = [
        json.loads(line)
        for line in source_receipt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reconciled: list[dict[str, Any]] = []
    remaining_failures: list[dict[str, str]] = []
    for record in receipts:
        if record.get("status") != "technical_failure":
            reconciled.append(record)
            continue
        exact_zero_failure = (
            record.get("failure_type") == "ScientificModelError"
            and record.get("failure_message")
            == "every scored neuron needs positive observed mass"
        )
        if not exact_zero_failure:
            remaining_failures.append(
                {
                    "provider_type": str(record["provider_type"]),
                    "failure_type": str(record.get("failure_type", "unknown")),
                    "failure_message": str(record.get("failure_message", "")),
                }
            )
            reconciled.append(record)
            continue
        provider_type = str(record["provider_type"])
        type_mask = focal["type"].astype(str).eq(provider_type).to_numpy()
        left = type_mask & focal["somaSide"].astype(str).eq("L").to_numpy()
        right = type_mask & focal["somaSide"].astype(str).eq("R").to_numpy()
        zero_left = int(np.sum(np.asarray(collapsed[left].sum(axis=1)).ravel() == 0))
        zero_right = int(np.sum(np.asarray(collapsed[right].sum(axis=1)).ravel() == 0))
        if zero_left + zero_right == 0:
            raise ValueError(
                f"cannot reconcile {provider_type}: snapshot has no zero-count neuron"
            )
        repaired = {
            key: value
            for key, value in record.items()
            if key not in {"failure_type", "failure_message"}
        }
        repaired.update(
            {
                "status": "unscorable",
                "reason_code": "zero_observed_target_count",
                "zero_count_left_neurons": zero_left,
                "zero_count_right_neurons": zero_right,
                "reconciled_from_exact_failure": True,
            }
        )
        reconciled.append(repaired)

    unscorable = [record for record in reconciled if record.get("status") == "unscorable"]
    valid = [
        record
        for record in reconciled
        if record.get("status") == "valid_complete_T_U_M_triplet"
    ]
    if len(reconciled) != 816 or len(valid) + len(unscorable) + len(remaining_failures) != 816:
        raise ValueError("reconciled whole-type accounting is incomplete")
    report = dict(source_report)
    report["schema_version"] = "ep12.development_trial_reconciliation.v1"
    report["status"] = (
        "completed_with_unscorable_types"
        if not remaining_failures
        else "completed_with_technical_failures"
    )
    report["aggregate"] = dict(source_report["aggregate"])
    report["aggregate"]["valid_type_count"] = len(valid)
    report["aggregate"]["unscorable_type_count"] = len(unscorable)
    report["aggregate"]["technical_failure_count"] = len(remaining_failures)
    report["unscorable"] = unscorable
    report["failures"] = remaining_failures
    report["reconciliation"] = {
        "source_trial_report_sha256": digest_bytes(source_report_path.read_bytes()),
        "source_receipts_sha256": digest_bytes(source_receipt_path.read_bytes()),
        "operation": (
            "reclassify only the exact zero-positive-mass ScientificModelError "
            "after verifying zero-count neurons in the hashed development snapshot"
        ),
        "scores_refit_or_changed": False,
        "reconciled_type_count": len(unscorable),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "type_direction_receipts.jsonl").open(
        "x", encoding="utf-8"
    ) as handle:
        for record in reconciled:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    shutil.copyfile(source_trial_dir / "result_table.csv", output_dir / "result_table.csv")
    atomic_json(output_dir / "trial_report.json", report)
    atomic_json(
        output_dir / "manifest.json",
        {
            "schema_version": "ep12.development_trial_reconciliation_manifest.v1",
            "files": {
                name: digest_bytes((output_dir / name).read_bytes())
                for name in (
                    "type_direction_receipts.jsonl",
                    "trial_report.json",
                    "result_table.csv",
                )
            },
            "scores_refit_or_changed": False,
            "final_connectivity_accessed": False,
        },
    )
    return report
