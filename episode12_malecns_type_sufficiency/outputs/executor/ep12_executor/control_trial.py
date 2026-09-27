"""Complete development-only T/U/M runner for frozen falsifier branches.

This is separate from ``development_trial.py`` so already-launched observed
jobs keep their code identity.  It uses the same likelihood, source-only
fitting, configuration, and aggregation, while serializing the source-side M
component contrasts and neuron-level scores required by prespecified controls.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet
from scipy import sparse

from .development_trial import _collapse_vocabulary, _fixed_covariates
from .contract_roles import active_record_fields, validate_contract_provenance
from .falsifier_controls import (
    FalsifierControlError,
    ablate_nuisance_block,
    component_partner_contrasts,
    endpoint_bin,
    pooled_multinomial_noise_sparse,
)
from .policy import EpisodePolicy, digest_bytes, digest_object
from .scientific_models import fit_triplet, seed_from
from .scientific_qualification import _contract_settings
from .runtime import EPISODE_ROOT, SCRATCH_ROOT, atomic_json
from .yaml_subset import load_yaml_subset


CONTROL_TRIAL_SCHEMA = "ep12.control_development_trial.v1"
ALLOWED_CONTROL_SPEC_KEYS = {
    "branch_id",
    "matrix_transform",
    "transform_seed",
    "nuisance_ablation_block",
    "partner_pool_keys",
    "endpoint_pool_bin",
}


def _control_spec(value: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(value or {})
    unknown = set(result) - ALLOWED_CONTROL_SPEC_KEYS
    if unknown:
        raise FalsifierControlError(f"unknown control-spec keys: {sorted(unknown)}")
    result.setdefault("branch_id", "weighted_baseline")
    result.setdefault("matrix_transform", "identity")
    result.setdefault("transform_seed", 0)
    result.setdefault("nuisance_ablation_block", None)
    result.setdefault("partner_pool_keys", [])
    result.setdefault("endpoint_pool_bin", None)
    if result["matrix_transform"] not in {
        "identity",
        "binary",
        "pooled_multinomial_noise",
    }:
        raise FalsifierControlError("unknown matrix transform")
    if result["partner_pool_keys"] and result["endpoint_pool_bin"] is not None:
        raise FalsifierControlError("partner and endpoint pooling cannot share one branch")
    return result


def _pool_after_collapse(
    *,
    matrix: sparse.csr_matrix,
    focal_types: np.ndarray,
    raw_vocabulary: list[str],
    configuration: Mapping[str, Any],
    partner_hierarchy: dict[str, dict[str, str]] | None,
    selected_columns: list[int],
    pool_name: str,
) -> tuple[sparse.csr_matrix, list[str], dict[str, Any], dict[str, Any]]:
    if not selected_columns:
        raise FalsifierControlError("pool branch selected no raw partner columns")
    selected = sorted(set(int(value) for value in selected_columns))
    pooled_mass = np.asarray(matrix[:, selected].sum(axis=1)).ravel().astype(np.int64)
    if int(pooled_mass.sum()) <= 0:
        raise FalsifierControlError("pool branch selected zero observed mass")
    retained = matrix.tolil(copy=True)
    retained[:, selected] = 0
    retained = retained.tocsr()
    collapsed, vocabulary, record = _collapse_vocabulary(
        retained,
        focal_types,
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
    collapsed = sparse.hstack(
        [collapsed, sparse.csr_matrix(pooled_mass[:, None])], format="csr"
    )
    vocabulary = list(vocabulary) + [pool_name]
    before = np.asarray(matrix.sum(axis=1)).ravel()
    after = np.asarray(collapsed.sum(axis=1)).ravel()
    if not np.array_equal(before, after):
        raise FalsifierControlError("explicit pool branch failed to preserve row mass")
    receipt = {
        "transform": "explicit_retained_mass_pool",
        "pool_name": pool_name,
        "selected_raw_columns": selected,
        "selected_raw_keys": [raw_vocabulary[index] for index in selected],
        "pooled_mass": int(pooled_mass.sum()),
        "row_totals_preserved": True,
        "total_mass_preserved": int(matrix.sum()) == int(collapsed.sum()),
    }
    return collapsed, vocabulary, record, receipt


def _prepare_arrays(
    *,
    snapshot_dir: Path,
    configuration: Mapping[str, Any],
    control_spec: Mapping[str, Any],
) -> tuple[sparse.csr_matrix, pd.DataFrame, np.ndarray, list[str], dict[str, Any], list[dict[str, Any]]]:
    matrix = sparse.load_npz(snapshot_dir / "development_outgoing_counts.npz").tocsr()
    focal = parquet.read_table(snapshot_dir / "development_focal_neurons.parquet").to_pandas()
    stats = parquet.read_table(snapshot_dir / "development_body_statistics.parquet").to_pandas()
    raw_vocabulary = json.loads(
        (snapshot_dir / "raw_partner_vocabulary.json").read_text(encoding="utf-8")
    )
    if matrix.shape != (len(focal), len(raw_vocabulary)):
        raise FalsifierControlError("development snapshot dimensions disagree")
    hierarchy_path = snapshot_dir / "partner_type_hierarchy.json"
    hierarchy = (
        json.loads(hierarchy_path.read_text(encoding="utf-8"))
        if hierarchy_path.is_file()
        else None
    )
    transforms: list[dict[str, Any]] = []
    matrix_transform = str(control_spec["matrix_transform"])
    deferred_noise = matrix_transform == "pooled_multinomial_noise"
    if matrix_transform == "binary":
        matrix = matrix.sign().astype(np.int64).tocsr()
        transforms.append(
            {
                "transform": "binary_edge_counts",
                "same_shape": True,
                "nonzero_support_preserved": True,
                "row_mass_preservation_required": False,
            }
        )
    elif matrix_transform == "pooled_multinomial_noise":
        # Defer the draw until after the frozen baseline vocabulary collapse.
        # This preserves the exact modeled partner dimension as well as every
        # neuron row total, so the T/U/M candidate capacities are identical.
        pass
    elif matrix_transform != "identity":
        raise FalsifierControlError("unsupported matrix transform")

    selected_columns: list[int] = []
    pool_name: str | None = None
    partner_keys = set(map(str, control_spec.get("partner_pool_keys", [])))
    endpoint = control_spec.get("endpoint_pool_bin")
    if partner_keys:
        missing = partner_keys - set(map(str, raw_vocabulary))
        if missing:
            raise FalsifierControlError(f"partner pool keys are absent: {sorted(missing)}")
        selected_columns = [
            index for index, key in enumerate(raw_vocabulary) if str(key) in partner_keys
        ]
        pool_name = f"control_partner_family_pool::{control_spec['branch_id']}"
    elif endpoint is not None:
        selected_columns = [
            index
            for index, key in enumerate(raw_vocabulary)
            if endpoint_bin(str(key)) == str(endpoint)
        ]
        pool_name = f"control_endpoint_pool::{endpoint}"

    focal_types = focal["type"].astype(str).to_numpy()
    if selected_columns:
        collapsed, vocabulary, vocabulary_record, receipt = _pool_after_collapse(
            matrix=matrix,
            focal_types=focal_types,
            raw_vocabulary=list(map(str, raw_vocabulary)),
            configuration=configuration,
            partner_hierarchy=hierarchy,
            selected_columns=selected_columns,
            pool_name=str(pool_name),
        )
        transforms.append(receipt)
    else:
        collapsed, vocabulary, vocabulary_record = _collapse_vocabulary(
            matrix,
            focal_types,
            list(map(str, raw_vocabulary)),
            strategy=str(configuration["partner_vocabulary"]),
            minimum_development_type_count=int(
                configuration["partner_minimum_development_type_count"]
            ),
            rare_partner_mass_fraction=float(
                configuration["rare_partner_pooling_mass_fraction"]
            ),
            hierarchy_depth=int(configuration["partner_hierarchy_depth"]),
            partner_hierarchy=hierarchy,
        )
    if deferred_noise:
        collapsed, receipt = pooled_multinomial_noise_sparse(
            collapsed, seed=int(control_spec["transform_seed"])
        )
        transforms.append(receipt)
    vocabulary_record = {
        **vocabulary_record,
        "collapsed_partner_count": len(vocabulary),
        "modeled_partner_vocabulary": list(vocabulary),
        "modeled_partner_vocabulary_sha256": digest_object(list(vocabulary)),
    }
    covariates = _fixed_covariates(
        focal,
        stats,
        spline_df=int(configuration["nuisance_spline_df"]),
    )
    nuisance = control_spec.get("nuisance_ablation_block")
    if nuisance is not None:
        covariates, receipt = ablate_nuisance_block(
            covariates,
            spline_df=int(configuration["nuisance_spline_df"]),
            block_id=str(nuisance),
        )
        transforms.append(receipt)
    return collapsed, focal, covariates, vocabulary, vocabulary_record, transforms


def _source_metadata(fitted: Any, vocabulary: list[str]) -> dict[str, Any]:
    contrasts = component_partner_contrasts(
        basis=fitted.M.basis,
        component_means=fitted.M.prior_means,
        component_weights=fitted.M.prior_weights,
    )
    contrasts["partner_vocabulary"] = list(vocabulary)
    return {
        "U": fitted.U.model_id,
        "M": fitted.M.model_id,
        "reference": fitted.reference_model_id,
        "null_generator": fitted.null_generator_model_id,
        "parameter_counts": {
            "T": fitted.T.parameter_count,
            "U": fitted.U.parameter_count,
            "M": fitted.M.parameter_count,
        },
        "M_component_partner_contrasts": contrasts,
    }


def _target_neuron_rows(
    *,
    scored: Mapping[str, Any],
    body_ids: list[str],
    counts: np.ndarray,
) -> list[dict[str, Any]]:
    arrays = {
        key: np.asarray(scored[key], dtype=float)
        for key in ("T", "U", "M", "reference", "delta_M_minus_reference")
    }
    if any(len(values) != len(body_ids) for values in arrays.values()):
        raise FalsifierControlError("target score length differs from neuron identities")
    strengths = counts.sum(axis=1).astype(np.int64)
    return [
        {
            "body_id": body_id,
            "strength": int(strengths[index]),
            "T": float(arrays["T"][index]),
            "U": float(arrays["U"][index]),
            "M": float(arrays["M"][index]),
            "reference": float(arrays["reference"][index]),
            "delta": float(arrays["delta_M_minus_reference"][index]),
        }
        for index, body_id in enumerate(body_ids)
    ]


def verify_development_trial_artifacts(
    *,
    result_root: Path,
    configuration: Mapping[str, Any],
    configuration_hash: str,
) -> dict[str, Any]:
    """Authenticate a trial under EP12 outputs or its dedicated scratch root."""

    resolved = result_root.resolve()
    outputs_root = (EPISODE_ROOT / "outputs").resolve()
    scratch_root = SCRATCH_ROOT.resolve()
    if not any(
        resolved == root or root in resolved.parents
        for root in (outputs_root, scratch_root)
    ):
        raise FalsifierControlError("trial result root leaves EP12 outputs/scratch")
    trial_id = str(configuration["trial_id"])
    trial_dir = resolved / trial_id
    manifest_path = trial_dir / "manifest.json"
    report_path = trial_dir / "trial_report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        raise FalsifierControlError(f"missing durable artifacts for {trial_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("final_connectivity_accessed") is not False:
        raise FalsifierControlError(f"final-connectivity boundary missing for {trial_id}")
    for name, expected in manifest.get("files", {}).items():
        artifact = trial_dir / str(name)
        if not artifact.is_file() or digest_bytes(artifact.read_bytes()) != str(expected):
            raise FalsifierControlError(f"artifact digest mismatch for {trial_id}/{name}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("trial_id") != trial_id:
        raise FalsifierControlError(f"trial identity mismatch for {trial_id}")
    if report.get("final_connectivity_accessed") is not False:
        raise FalsifierControlError(f"final connectivity entered {trial_id}")
    aggregate = report.get("aggregate", {})
    if report.get("status") != "completed":
        raise FalsifierControlError(f"trial did not complete cleanly: {trial_id}")
    if int(aggregate.get("technical_failure_count", -1)) != 0:
        raise FalsifierControlError(f"trial has technical failures: {trial_id}")
    if int(aggregate.get("valid_type_count", 0)) <= 0:
        raise FalsifierControlError(f"trial has no valid comparisons: {trial_id}")
    if report.get("configuration", {}).get("configuration_sha256") != configuration_hash:
        raise FalsifierControlError(f"executed configuration identity mismatch: {trial_id}")
    directional = (
        float(aggregate["mean_left_to_right_delta"]),
        float(aggregate["mean_right_to_left_delta"]),
    )
    bilateral = float(aggregate["mean_bilateral_delta"])
    if not all(math.isfinite(value) for value in (*directional, bilateral)):
        raise FalsifierControlError(f"nonfinite objective for {trial_id}")
    return {
        "trial_id": trial_id,
        "configuration": dict(configuration),
        "configuration_sha256": configuration_hash,
        "objective": {
            "left_to_right": directional[0],
            "right_to_left": directional[1],
            "bilateral": bilateral,
            "bilateral_floor": min(directional),
            "direction_disagreement": abs(directional[0] - directional[1]),
            "types_both_above_margin": int(
                aggregate["types_with_both_point_estimates_above_margin"]
            ),
        },
        "valid_type_count": int(aggregate["valid_type_count"]),
        "unscorable_type_count": int(aggregate["unscorable_type_count"]),
        "trial_report_sha256": digest_bytes(report_path.read_bytes()),
        "manifest_sha256": digest_bytes(manifest_path.read_bytes()),
    }


def run_control_development_trial(
    *,
    output_dir: Path,
    snapshot_dir: Path,
    configuration: Mapping[str, Any],
    control_spec: Mapping[str, Any] | None = None,
    contract_path: Path | None = None,
    contract_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit and serialize one complete control branch on development roles."""

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
        raise FalsifierControlError("control trial output leaves EP12 roots")
    if snapshot_dir != allowed_scratch and allowed_scratch not in snapshot_dir.parents:
        raise FalsifierControlError("control trial snapshot leaves EP12 scratch")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"control trial output directory is not empty: {output_dir}")
    config = dict(configuration)
    spec = _control_spec(control_spec)
    trial_id = str(config["trial_id"])
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
    if contract.get("real_connectivity_access_authorized") != "development_only":
        raise FalsifierControlError("development connectivity is not authorized")
    if contract.get("final_connectivity_access_authorized", False) is not False:
        raise FalsifierControlError("final connectivity must remain locked")
    settings = _contract_settings(contract)
    policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")
    collapsed, focal, covariates, vocabulary, vocabulary_record, transforms = _prepare_arrays(
        snapshot_dir=snapshot_dir,
        configuration=config,
        control_spec=spec,
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
            if np.any(left_counts.sum(axis=1) == 0) or np.any(right_counts.sum(axis=1) == 0):
                record.update({"status": "unscorable", "reason_code": "zero_observed_target_count"})
                unscorable.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                continue
            component_count = int(config["component_count"])
            minimum_source = component_count * int(settings["source_cv_folds"])
            if int(left.sum()) < minimum_source or int(right.sum()) < minimum_source:
                record.update(
                    {
                        "status": "unscorable",
                        "reason_code": "source_side_below_frozen_support_after_integrity_filter",
                        "minimum_source_neurons_for_component_cv": minimum_source,
                    }
                )
                unscorable.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                continue
            try:
                options = dict(
                    integration_draws=int(settings["integration_draws"]),
                    folds=int(settings["source_cv_folds"]),
                    latent_rank=int(config["rank"]),
                    ridge=float(config["regularization"]),
                    pseudocount=float(config["composition_pseudocount"]),
                    representation=str(config["representation"]),
                    covariance_mode=str(config["covariance"]),
                    covariance_rank=int(config["covariance_rank"]),
                    component_candidates=(component_count,),
                )
                seed = seed_from("ep12-control", trial_id, provider_type)
                left_fit = fit_triplet(
                    left_counts, covariates[left], seed=seed_from(seed, "left-source"), **options
                )
                right_fit = fit_triplet(
                    right_counts, covariates[right], seed=seed_from(seed, "right-source"), **options
                )
                left_to_right = left_fit.score_target(right_counts, covariates[right])
                right_to_left = right_fit.score_target(left_counts, covariates[left])
                record.update(
                    {
                        "status": "valid_complete_T_U_M_triplet",
                        "left_to_right_delta": float(
                            left_to_right["mean_delta_M_minus_reference"]
                        ),
                        "right_to_left_delta": float(
                            right_to_left["mean_delta_M_minus_reference"]
                        ),
                        "left_to_right_scores": left_to_right["mean_scores"],
                        "right_to_left_scores": right_to_left["mean_scores"],
                        "source_fit_metadata": {
                            "left": _source_metadata(left_fit, vocabulary),
                            "right": _source_metadata(right_fit, vocabulary),
                        },
                        "target_neuron_scores": {
                            "left_to_right": _target_neuron_rows(
                                scored=left_to_right,
                                body_ids=focal.loc[right, "bodyId"].astype(str).tolist(),
                                counts=right_counts,
                            ),
                            "right_to_left": _target_neuron_rows(
                                scored=right_to_left,
                                body_ids=focal.loc[left, "bodyId"].astype(str).tolist(),
                                counts=left_counts,
                            ),
                        },
                    }
                )
                rows.append(record)
            except Exception as error:
                record.update(
                    {
                        "status": "technical_failure",
                        "failure_type": type(error).__name__,
                        "failure_message": str(error),
                    }
                )
                failures.append(
                    {
                        "provider_type": provider_type,
                        "failure_type": type(error).__name__,
                        "failure_message": str(error),
                    }
                )
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
        os.fsync(handle.fileno())

    if rows:
        values = np.asarray(
            [[row["left_to_right_delta"], row["right_to_left_delta"]] for row in rows]
        )
        aggregate = {
            "valid_type_count": len(rows),
            "unscorable_type_count": len(unscorable),
            "technical_failure_count": len(failures),
            "mean_left_to_right_delta": float(values[:, 0].mean()),
            "mean_right_to_left_delta": float(values[:, 1].mean()),
            "mean_bilateral_delta": float(values.mean()),
            "types_with_both_point_estimates_above_margin": int(
                np.sum(np.all(values > float(policy.meaningful_margin), axis=1))
            ),
        }
    else:
        aggregate = {
            "valid_type_count": 0,
            "unscorable_type_count": len(unscorable),
            "technical_failure_count": len(failures),
        }
    report = {
        "schema_version": CONTROL_TRIAL_SCHEMA,
        "trial_id": trial_id,
        "status": "completed" if not failures else "completed_with_technical_failures",
        "data_role": "development_only",
        "final_connectivity_accessed": False,
        "configuration": {
            **config,
            **vocabulary_record,
            "configuration_sha256": digest_object(config),
        },
        "control_spec": spec,
        "control_spec_sha256": digest_object(spec),
        "control_transforms": transforms,
        "aggregate": aggregate,
        "complete_type_records": rows,
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
            "schema_version": "ep12.control_development_trial_manifest.v1",
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
