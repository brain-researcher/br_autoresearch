#!/usr/bin/env python3
"""Aggregate frozen episode02 checkpoints and evaluate both mechanism tests."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import stats


WORKSPACE = Path(__file__).resolve().parents[2]
OUT = WORKSPACE / "outputs"
SCRATCH = Path("/scratch/users/zijiao/episode02_narps_analysis")
CHECKPOINTS = SCRATCH / "checkpoints"
PARTICIPANTS = WORKSPACE / "inputs" / "raw" / "participants.tsv"
CONTRACT = OUT / "frozen_successor_contract.json"
CONTRASTS = ("gain_demean", "loss_demean")
SMOOTHINGS = ("S0", "S8")
STRATEGIES = ("ivw", "equal", "stabilized")
CONTRACT_SHA256 = "27a58a09ee043e71b7d1fca93de0ef727b79fc6ecbd9fef2eaf800208317829f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_participants() -> tuple[list[str], dict[str, str]]:
    groups = {}
    with PARTICIPANTS.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            groups[row["participant_id"]] = row["group"]
    return sorted(groups), groups


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def region_metric(y0: np.ndarray, y8: np.ndarray, region: np.ndarray) -> dict[str, float]:
    a = np.asarray(y0[region], dtype=np.float64)
    b = np.asarray(y8[region], dtype=np.float64)
    ss0 = float(a @ a)
    delta = b - a
    dss = float(delta @ delta)
    d_raw = math.sqrt(dss / ss0) if ss0 > 0 else math.nan
    ac = a - a.mean()
    bc = b - b.mean()
    centered0 = float(ac @ ac)
    centered8 = float(bc @ bc)
    centered_cross = float(ac @ bc)
    denom = math.sqrt(centered0 * centered8)
    pearson = centered_cross / denom if denom > 0 else math.nan
    # Evaluate the declared OLS fit directly.  The algebraically equivalent
    # centered0*(1-r**2) loses many digits for the affine negative control.
    if centered8 > 0:
        beta = centered_cross / centered8
        residual = ac - beta * bc
        residual_ss = float(residual @ residual)
    else:
        residual_ss = math.nan
    d_aligned = math.sqrt(residual_ss / ss0) if ss0 > 0 and residual_ss >= 0 else math.nan
    aligned_ratio = d_aligned / d_raw if d_raw > 0 else math.nan
    return {
        "d_raw": d_raw,
        "d_aligned": d_aligned,
        "aligned_ratio": aligned_ratio,
        "scale_attenuation": 1.0 - aligned_ratio,
        "pearson_r": pearson,
        "delta_mean_square": dss / len(a),
    }


def gram_pack(m0: np.ndarray, m8: np.ndarray, region: np.ndarray) -> dict[str, np.ndarray | float]:
    a = np.asarray(m0[:, region], dtype=np.float64)
    b = np.asarray(m8[:, region], dtype=np.float64)
    return {
        "n_voxels": float(a.shape[1]),
        "sum0": a.sum(axis=1),
        "sum8": b.sum(axis=1),
        "g00": a @ a.T,
        "g88": b @ b.T,
        "g08": a @ b.T,
    }


def quadratic(weights: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum("bi,ij,bj->b", weights, matrix, weights, optimize=True)


def weighted_metrics(pack: dict, weights: np.ndarray) -> dict[str, np.ndarray]:
    if weights.ndim == 1:
        weights = weights[None, :]
    nvox = float(pack["n_voxels"])
    sum0 = weights @ pack["sum0"]
    sum8 = weights @ pack["sum8"]
    ss0 = quadratic(weights, pack["g00"])
    ss8 = quadratic(weights, pack["g88"])
    cross = quadratic(weights, pack["g08"])
    dss = np.maximum(0.0, ss0 + ss8 - 2.0 * cross)
    d_raw = np.sqrt(dss / ss0)
    centered0 = np.maximum(0.0, ss0 - sum0 * sum0 / nvox)
    centered8 = np.maximum(0.0, ss8 - sum8 * sum8 / nvox)
    centered_cross = cross - sum0 * sum8 / nvox
    pearson = centered_cross / np.sqrt(centered0 * centered8)
    pearson = np.clip(pearson, -1.0, 1.0)
    # Schur-complement form of the same OLS residual.  It is more stable than
    # centered0*(1-r**2) and is available from the frozen Gram summaries.
    explained_ss = np.full_like(centered0, np.nan)
    np.divide(
        centered_cross * centered_cross,
        centered8,
        out=explained_ss,
        where=centered8 > 0,
    )
    residual_ss = np.maximum(0.0, centered0 - explained_ss)
    d_aligned = np.sqrt(residual_ss / ss0)
    aligned_ratio = d_aligned / d_raw
    return {
        "d_raw": d_raw,
        "d_aligned": d_aligned,
        "aligned_ratio": aligned_ratio,
        "scale_attenuation": 1.0 - aligned_ratio,
        "pearson_r": pearson,
        "delta_mean_square": dss / nvox,
    }


def equal_cohort_weights(subjects: list[str], groups: dict[str, str], selected: set[str] | None = None) -> np.ndarray:
    if selected is None:
        selected = set(subjects)
    weights = np.zeros(len(subjects), dtype=np.float64)
    for group in ("equalIndifference", "equalRange"):
        indices = [i for i, subject in enumerate(subjects) if subject in selected and groups[subject] == group]
        if not indices:
            raise ValueError(f"No selected subjects in {group}")
        weights[indices] = 0.5 / len(indices)
    return weights


def stratum_weights(subjects: list[str], groups: dict[str, str], group: str) -> np.ndarray:
    indices = [i for i, subject in enumerate(subjects) if groups[subject] == group]
    weights = np.zeros(len(subjects), dtype=np.float64)
    weights[indices] = 1.0 / len(indices)
    return weights


def bootstrap_weights(subjects: list[str], groups: dict[str, str], replicates: int = 1000) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(20260817))
    weights = np.zeros((replicates, len(subjects)), dtype=np.float64)
    for group in ("equalIndifference", "equalRange"):
        indices = np.array([i for i, subject in enumerate(subjects) if groups[subject] == group])
        counts = rng.multinomial(len(indices), np.repeat(1.0 / len(indices), len(indices)), size=replicates)
        weights[:, indices] = 0.5 * counts / len(indices)
    return weights


def leave_one_out_weights(subjects: list[str], groups: dict[str, str]) -> np.ndarray:
    weights = np.zeros((len(subjects), len(subjects)), dtype=np.float64)
    for excluded_index, excluded in enumerate(subjects):
        for group in ("equalIndifference", "equalRange"):
            indices = [
                i for i, subject in enumerate(subjects)
                if groups[subject] == group and subject != excluded
            ]
            weights[excluded_index, indices] = 0.5 / len(indices)
    return weights


def bh_mask(t_values: np.ndarray, df: int, q: float = 0.05) -> np.ndarray:
    p_values = 2.0 * stats.t.sf(np.abs(t_values), df=df)
    order = np.argsort(p_values)
    ranked = p_values[order]
    passed = ranked <= q * np.arange(1, len(ranked) + 1) / len(ranked)
    reject = np.zeros(len(ranked), dtype=bool)
    if np.any(passed):
        threshold = ranked[np.flatnonzero(passed)[-1]]
        reject = p_values <= threshold
    return reject


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.sum(a | b))
    return float(np.sum(a & b) / union) if union else math.nan


def threshold_pair_metrics(
    m0: np.ndarray,
    m8: np.ndarray,
    task_codes: np.ndarray | None,
) -> dict[str, float | int]:
    """Evaluate the frozen FDR and matched-rank readouts for one subject set."""
    t0, df0 = group_t(np.asarray(m0, dtype=np.float64), task_codes)
    t8, df8 = group_t(np.asarray(m8, dtype=np.float64), task_codes)
    if df0 != df8:
        raise RuntimeError(f"Smoothing degrees of freedom differ: {df0} versus {df8}")
    reject0 = bh_mask(t0, df=df0, q=0.05)
    reject8 = bh_mask(t8, df=df8, q=0.05)
    fdr_j = jaccard(reject0, reject8)
    k = min(int(reject0.sum()), int(reject8.sum()))
    if k > 0:
        top0 = np.zeros(len(t0), dtype=bool)
        top8 = np.zeros(len(t8), dtype=bool)
        top0[np.argpartition(np.abs(t0), -k)[-k:]] = True
        top8[np.argpartition(np.abs(t8), -k)[-k:]] = True
        top_j = jaccard(top0, top8)
    else:
        top_j = math.nan
    return {
        "df": df0,
        "s0_rejected": int(reject0.sum()),
        "s8_rejected": int(reject8.sum()),
        "fdr_jaccard": fdr_j,
        "matched_k": k,
        "top_abs_t_jaccard": top_j,
    }


def group_t(subject_maps: np.ndarray, task_codes: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    n = subject_maps.shape[0]
    if task_codes is None:
        mean = subject_maps.mean(axis=0, dtype=np.float64)
        sd = subject_maps.std(axis=0, ddof=1, dtype=np.float64)
        t_values = mean / (sd / math.sqrt(n))
        return t_values, n - 1
    x = np.column_stack([np.ones(n), task_codes])
    pinv = np.linalg.pinv(x)
    beta = pinv @ subject_maps
    residual = subject_maps - x @ beta
    df = n - x.shape[1]
    sigma2 = np.sum(residual * residual, axis=0, dtype=np.float64) / df
    covariance00 = np.linalg.inv(x.T @ x)[0, 0]
    t_values = beta[0] / np.sqrt(sigma2 * covariance00)
    return t_values, df


def save_masked_map(values: np.ndarray, mask_img: nib.spatialimages.SpatialImage, path: Path) -> None:
    mask = np.asanyarray(mask_img.dataobj) > 0
    data = np.zeros(mask.shape, dtype=np.float32)
    data[mask] = values.astype(np.float32)
    header = mask_img.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data, mask_img.affine, header), str(path))


def percentile_row(contrast: str, strategy: str, metric: str, values: np.ndarray, estimate: float) -> dict:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "analysis": "paired_cohort_stratified_bootstrap",
        "contrast": contrast,
        "strategy": strategy,
        "metric": metric,
        "estimate": estimate,
        "lower_95": float(lower),
        "upper_95": float(upper),
        "replicates": len(values),
        "seed": 20260817,
        "threshold": "",
        "any_threshold_crossing": "",
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    if sha256(CONTRACT) != CONTRACT_SHA256:
        raise RuntimeError("Frozen contract hash mismatch")
    subjects, groups = read_participants()
    if len(subjects) != 108:
        raise RuntimeError(f"Expected 108 subjects, found {len(subjects)}")

    manifest = []
    maps = {
        contrast: {strategy: {} for strategy in STRATEGIES}
        for contrast in CONTRASTS
    }
    loaded = {contrast: {strategy: [] for strategy in STRATEGIES} for contrast in CONTRASTS}
    for subject in subjects:
        checkpoint = CHECKPOINTS / f"{subject}.npz"
        sidecar = CHECKPOINTS / f"{subject}.json"
        if not checkpoint.exists() or not sidecar.exists():
            raise RuntimeError(f"Missing checkpoint for {subject}")
        metadata = json.loads(sidecar.read_text())
        if metadata.get("status") != "complete" or metadata.get("contract_sha256") != CONTRACT_SHA256:
            raise RuntimeError(f"Invalid sidecar for {subject}")
        actual_hash = sha256(checkpoint)
        if actual_hash != metadata.get("checkpoint_sha256"):
            raise RuntimeError(f"Checkpoint hash mismatch for {subject}")
        manifest.append({
            "participant_id": subject,
            "task_version": groups[subject],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": actual_hash,
            "elapsed_seconds": metadata.get("elapsed_seconds"),
            "hostname": metadata.get("hostname"),
            "slurm_job_id": metadata.get("slurm_job_id"),
            "slurm_array_task_id": metadata.get("slurm_array_task_id"),
            "status": metadata.get("status"),
        })
        with np.load(checkpoint) as data:
            if int(data["mask_voxels"][0]) != 221116:
                raise RuntimeError(f"Mask voxel mismatch for {subject}")
            for contrast in CONTRASTS:
                for strategy in STRATEGIES:
                    for smoothing in SMOOTHINGS:
                        key = f"{contrast}__{smoothing}__{strategy}"
                        maps[contrast][strategy].setdefault(smoothing, []).append(
                            np.asarray(data[key], dtype=np.float32)
                        )
    write_tsv(OUT / "analysis_manifest.tsv", manifest)
    for contrast in CONTRASTS:
        for strategy in STRATEGIES:
            for smoothing in SMOOTHINGS:
                maps[contrast][strategy][smoothing] = np.stack(maps[contrast][strategy][smoothing])

    common_img = nib.load(str(OUT / "common_analysis_mask.nii.gz"))
    common_volume = np.asanyarray(common_img.dataobj) > 0
    core_volume = np.asanyarray(nib.load(str(OUT / "core_analysis_mask.nii.gz")).dataobj) > 0
    periphery_volume = np.asanyarray(nib.load(str(OUT / "periphery_analysis_mask.nii.gz")).dataobj) > 0
    core = core_volume[common_volume]
    periphery = periphery_volume[common_volume]
    regions = {"common": np.ones(int(common_volume.sum()), dtype=bool), "core": core, "periphery": periphery}

    sorted_by_group = {
        group: [subject for subject in subjects if groups[subject] == group]
        for group in ("equalIndifference", "equalRange")
    }
    split_a = set(sorted_by_group["equalIndifference"][::2] + sorted_by_group["equalRange"][::2])
    split_b = set(sorted_by_group["equalIndifference"][1::2] + sorted_by_group["equalRange"][1::2])
    scope_weights = {
        "full": equal_cohort_weights(subjects, groups),
        "EI": stratum_weights(subjects, groups, "equalIndifference"),
        "ER": stratum_weights(subjects, groups, "equalRange"),
        "split_A": equal_cohort_weights(subjects, groups, split_a),
        "split_B": equal_cohort_weights(subjects, groups, split_b),
    }
    scope_n = {"full": 108, "EI": 54, "ER": 54, "split_A": 54, "split_B": 54}

    group_dir = OUT / "group_maps"
    group_dir.mkdir(exist_ok=True)
    point_rows = []
    point_lookup = {}
    group_map_cache = {}
    for contrast in CONTRASTS:
        for scope, weights in scope_weights.items():
            for strategy in STRATEGIES:
                y0 = weights @ maps[contrast][strategy]["S0"]
                y8 = weights @ maps[contrast][strategy]["S8"]
                group_map_cache[(contrast, scope, strategy, "S0")] = y0
                group_map_cache[(contrast, scope, strategy, "S8")] = y8
                if scope == "full":
                    save_masked_map(y0, common_img, group_dir / f"{contrast}_{strategy}_S0_effect.nii.gz")
                    save_masked_map(y8, common_img, group_dir / f"{contrast}_{strategy}_S8_effect.nii.gz")
                region_values = {
                    region_name: region_metric(y0, y8, region)
                    for region_name, region in regions.items()
                }
                edge_enrichment = (
                    region_values["periphery"]["delta_mean_square"]
                    / region_values["core"]["delta_mean_square"]
                )
                core_ratio = region_values["core"]["d_raw"] / region_values["common"]["d_raw"]
                ivw_full_d = region_metric(
                    scope_weights[scope] @ maps[contrast]["ivw"]["S0"],
                    scope_weights[scope] @ maps[contrast]["ivw"]["S8"],
                    regions["common"],
                )["d_raw"]
                aggregation_attenuation = 0.0 if strategy == "ivw" else 1.0 - region_values["common"]["d_raw"] / ivw_full_d
                for region_name, values in region_values.items():
                    row = {
                        "contrast": contrast,
                        "scope": scope,
                        "strategy": strategy,
                        "region": region_name,
                        "n_subjects": scope_n[scope],
                        **values,
                        "edge_enrichment": edge_enrichment if region_name == "common" else "",
                        "core_ratio": core_ratio if region_name == "common" else "",
                        "aggregation_attenuation": aggregation_attenuation if region_name == "common" else "",
                    }
                    point_rows.append(row)
                    point_lookup[(contrast, scope, strategy, region_name)] = row
    write_tsv(OUT / "mechanism_results.tsv", point_rows)

    bootstrap = bootstrap_weights(subjects, groups)
    loo = leave_one_out_weights(subjects, groups)
    robustness_rows = []
    gram = {}
    for contrast in CONTRASTS:
        for strategy in STRATEGIES:
            needed_regions = regions if strategy == "ivw" else {"common": regions["common"]}
            for region_name, region in needed_regions.items():
                gram[(contrast, strategy, region_name)] = gram_pack(
                    maps[contrast][strategy]["S0"], maps[contrast][strategy]["S8"], region
                )

        boot_metrics = {
            (strategy, region): weighted_metrics(gram[(contrast, strategy, region)], bootstrap)
            for strategy in STRATEGIES
            for region in (("common", "core", "periphery") if strategy == "ivw" else ("common",))
        }
        for metric in ("scale_attenuation", "pearson_r", "d_raw"):
            values = boot_metrics[("ivw", "common")][metric]
            estimate = float(point_lookup[(contrast, "full", "ivw", "common")][metric])
            robustness_rows.append(percentile_row(contrast, "ivw", metric, values, estimate))
        edge_values = (
            boot_metrics[("ivw", "periphery")]["delta_mean_square"]
            / boot_metrics[("ivw", "core")]["delta_mean_square"]
        )
        edge_est = float(point_lookup[(contrast, "full", "ivw", "common")]["edge_enrichment"])
        robustness_rows.append(percentile_row(contrast, "ivw", "edge_enrichment", edge_values, edge_est))
        core_ratio_values = boot_metrics[("ivw", "core")]["d_raw"] / boot_metrics[("ivw", "common")]["d_raw"]
        core_est = float(point_lookup[(contrast, "full", "ivw", "common")]["core_ratio"])
        robustness_rows.append(percentile_row(contrast, "ivw", "core_ratio", core_ratio_values, core_est))
        for strategy in ("equal", "stabilized"):
            attenuation_values = 1.0 - (
                boot_metrics[(strategy, "common")]["d_raw"]
                / boot_metrics[("ivw", "common")]["d_raw"]
            )
            estimate = float(point_lookup[(contrast, "full", strategy, "common")]["aggregation_attenuation"])
            robustness_rows.append(
                percentile_row(contrast, strategy, "aggregation_attenuation", attenuation_values, estimate)
            )

        loo_metrics = {
            (strategy, region): weighted_metrics(gram[(contrast, strategy, region)], loo)
            for strategy in STRATEGIES
            for region in (("common", "core", "periphery") if strategy == "ivw" else ("common",))
        }
        loo_series = {
            ("ivw", "scale_attenuation", 0.50): loo_metrics[("ivw", "common")]["scale_attenuation"],
            ("ivw", "pearson_r", 0.80): loo_metrics[("ivw", "common")]["pearson_r"],
            ("ivw", "edge_enrichment", 2.0): (
                loo_metrics[("ivw", "periphery")]["delta_mean_square"]
                / loo_metrics[("ivw", "core")]["delta_mean_square"]
            ),
            ("ivw", "core_ratio", 0.75): (
                loo_metrics[("ivw", "core")]["d_raw"] / loo_metrics[("ivw", "common")]["d_raw"]
            ),
        }
        for strategy in ("equal", "stabilized"):
            loo_series[(strategy, "aggregation_attenuation", 0.25)] = 1.0 - (
                loo_metrics[(strategy, "common")]["d_raw"]
                / loo_metrics[("ivw", "common")]["d_raw"]
            )
        for (strategy, metric, threshold), values in loo_series.items():
            estimate = float(
                point_lookup[(contrast, "full", strategy, "common")].get(metric, math.nan)
                if metric != "edge_enrichment" and metric != "core_ratio"
                else point_lookup[(contrast, "full", "ivw", "common")][metric]
            )
            full_side = estimate >= threshold
            crossing = bool(np.any((values >= threshold) != full_side))
            robustness_rows.append({
                "analysis": "leave_one_subject_out",
                "contrast": contrast,
                "strategy": strategy,
                "metric": metric,
                "estimate": estimate,
                "lower_95": "",
                "upper_95": "",
                "replicates": len(values),
                "seed": "",
                "threshold": threshold,
                "any_threshold_crossing": crossing,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            })
        d_raw_values = loo_metrics[("ivw", "common")]["d_raw"]
        robustness_rows.append({
            "analysis": "leave_one_subject_out",
            "contrast": contrast,
            "strategy": "ivw",
            "metric": "d_raw",
            "estimate": float(point_lookup[(contrast, "full", "ivw", "common")]["d_raw"]),
            "lower_95": "",
            "upper_95": "",
            "replicates": len(d_raw_values),
            "seed": "",
            "threshold": "",
            "any_threshold_crossing": "",
            "minimum": float(np.min(d_raw_values)),
            "maximum": float(np.max(d_raw_values)),
        })

    negative_control_rows = []
    for contrast in CONTRASTS:
        y0 = group_map_cache[(contrast, "full", "ivw", "S0")]
        control = 0.1 * np.std(y0) + 1.25 * y0
        values = region_metric(y0, control, regions["common"])
        residual_rms = values["d_aligned"] * math.sqrt(float(y0 @ y0) / len(y0))
        reference_rms = math.sqrt(float(y0 @ y0) / len(y0))
        passed = abs(values["pearson_r"] - 1.0) <= 1e-12 and residual_rms <= 1e-10 * reference_rms
        negative_control_rows.append({
            "analysis": "amplitude_alignment_negative_control",
            "contrast": contrast,
            "strategy": "ivw",
            "metric": "negative_control_pass",
            "estimate": passed,
            "lower_95": "",
            "upper_95": "",
            "replicates": "",
            "seed": "",
            "threshold": "r within 1e-12 of 1 and residual_rms <= 1e-10*reference_rms",
            "any_threshold_crossing": "",
            "minimum": values["pearson_r"],
            "maximum": residual_rms / reference_rms,
        })
        if not passed:
            raise RuntimeError(f"Amplitude alignment negative control failed for {contrast}: {values}")
    robustness_rows.extend(negative_control_rows)

    threshold_rows = []
    task_codes = np.array([0.5 if groups[s] == "equalIndifference" else -0.5 for s in subjects])
    scope_indices = {
        "full": np.arange(len(subjects)),
        "EI": np.array([i for i, s in enumerate(subjects) if groups[s] == "equalIndifference"]),
        "ER": np.array([i for i, s in enumerate(subjects) if groups[s] == "equalRange"]),
        "split_A": np.array([i for i, s in enumerate(subjects) if s in split_a]),
        "split_B": np.array([i for i, s in enumerate(subjects) if s in split_b]),
    }
    for contrast in CONTRASTS:
        for scope, indices in scope_indices.items():
            codes = None if scope in ("EI", "ER") else task_codes[indices]
            pair = threshold_pair_metrics(
                maps[contrast]["ivw"]["S0"][indices],
                maps[contrast]["ivw"]["S8"][indices],
                codes,
            )
            if scope == "full":
                for smoothing in SMOOTHINGS:
                    t_values, _ = group_t(
                        np.asarray(maps[contrast]["ivw"][smoothing][indices], dtype=np.float64),
                        codes,
                    )
                    save_masked_map(t_values, common_img, group_dir / f"{contrast}_ivw_{smoothing}_t.nii.gz")
            pearson = float(point_lookup[(contrast, scope, "ivw", "common")]["pearson_r"])
            threshold_rows.append({
                "contrast": contrast,
                "scope": scope,
                "n_subjects": len(indices),
                "df": pair["df"],
                "q": 0.05,
                "s0_rejected": pair["s0_rejected"],
                "s8_rejected": pair["s8_rejected"],
                "fdr_jaccard": pair["fdr_jaccard"],
                "matched_k": pair["matched_k"],
                "top_abs_t_jaccard": pair["top_abs_t_jaccard"],
                "continuous_pearson_r": pearson,
                "threshold_crossing_disconnect": bool(
                    pair["matched_k"] > 0
                    and pair["fdr_jaccard"] < 0.25
                    and pair["top_abs_t_jaccard"] >= 0.50
                ),
                "broad_spatial_relocation": bool(
                    pair["matched_k"] > 0
                    and pair["fdr_jaccard"] < 0.25
                    and pair["top_abs_t_jaccard"] < 0.25
                ),
            })
    write_tsv(OUT / "threshold_stability.tsv", threshold_rows)

    # Frozen decision evaluation.
    boot_lookup = {
        (row["contrast"], row["strategy"], row["metric"]): row
        for row in robustness_rows if row["analysis"] == "paired_cohort_stratified_bootstrap"
    }
    loo_lookup = {
        (row["contrast"], row["strategy"], row["metric"]): row
        for row in robustness_rows if row["analysis"] == "leave_one_subject_out"
    }

    def split_direction(metric: str, strategy: str, relation: str, value: float) -> bool:
        for contrast in CONTRASTS:
            for scope in ("split_A", "split_B"):
                row = point_lookup[(contrast, scope, strategy, "common")]
                observed = float(row[metric])
                if relation == "ge" and not observed >= value:
                    return False
                if relation == "le" and not observed <= value:
                    return False
        return True

    amplitude_support = True
    edge_support = True
    amplitude_guard = True
    edge_guard = True
    for contrast in CONTRASTS:
        full = point_lookup[(contrast, "full", "ivw", "common")]
        amp_boot = boot_lookup[(contrast, "ivw", "scale_attenuation")]
        r_boot = boot_lookup[(contrast, "ivw", "pearson_r")]
        amplitude_support &= (
            float(full["scale_attenuation"]) >= 0.50
            and float(amp_boot["lower_95"]) >= 0.50
            and float(full["pearson_r"]) >= 0.80
            and float(r_boot["lower_95"]) >= 0.80
        )
        for scope in ("EI", "ER"):
            row = point_lookup[(contrast, scope, "ivw", "common")]
            amplitude_support &= float(row["scale_attenuation"]) >= 0.50 and float(row["pearson_r"]) >= 0.80
        amplitude_guard &= not bool(loo_lookup[(contrast, "ivw", "scale_attenuation")]["any_threshold_crossing"])
        amplitude_guard &= not bool(loo_lookup[(contrast, "ivw", "pearson_r")]["any_threshold_crossing"])

        edge_boot = boot_lookup[(contrast, "ivw", "edge_enrichment")]
        core_boot = boot_lookup[(contrast, "ivw", "core_ratio")]
        alternative_checks = []
        for strategy in ("equal", "stabilized"):
            alt = point_lookup[(contrast, "full", strategy, "common")]
            alt_boot = boot_lookup[(contrast, strategy, "aggregation_attenuation")]
            support = (
                float(alt["aggregation_attenuation"]) >= 0.25
                and float(alt_boot["lower_95"]) >= 0.25
            )
            guard = not bool(
                loo_lookup[(contrast, strategy, "aggregation_attenuation")]["any_threshold_crossing"]
            )
            alternative_checks.append((support, guard))
        alt_pass = any(support for support, _ in alternative_checks)
        alt_guard = any(support and guard for support, guard in alternative_checks)
        edge_support &= (
            float(full["edge_enrichment"]) >= 2.0
            and float(edge_boot["lower_95"]) >= 2.0
            and float(full["core_ratio"]) <= 0.75
            and float(core_boot["upper_95"]) <= 0.75
            and alt_pass
        )
        for scope in ("EI", "ER"):
            row = point_lookup[(contrast, scope, "ivw", "common")]
            edge_support &= float(row["edge_enrichment"]) >= 1.5 and float(row["core_ratio"]) <= 0.85
        edge_guard &= not bool(loo_lookup[(contrast, "ivw", "edge_enrichment")]["any_threshold_crossing"])
        edge_guard &= not bool(loo_lookup[(contrast, "ivw", "core_ratio")]["any_threshold_crossing"])
        edge_guard &= alt_guard

    amplitude_split = split_direction("scale_attenuation", "ivw", "ge", 0.0) and split_direction("pearson_r", "ivw", "ge", 0.0)
    edge_split = split_direction("edge_enrichment", "ivw", "ge", 1.0) and split_direction("core_ratio", "ivw", "le", 1.0)
    edge_candidate = edge_support and edge_split and edge_guard
    amplitude_candidate = amplitude_support and not edge_support and amplitude_split and amplitude_guard

    residual_support = not amplitude_support and not edge_support
    residual_guard = True
    for contrast in CONTRASTS:
        for scope in ("full", "EI", "ER"):
            row = point_lookup[(contrast, scope, "equal", "core")]
            residual_support &= float(row["d_raw"]) >= 0.25 and float(row["pearson_r"]) <= 0.80
        for scope in ("split_A", "split_B"):
            row = point_lookup[(contrast, scope, "equal", "core")]
            residual_support &= float(row["d_raw"]) >= 0.0 and float(row["pearson_r"]) <= 1.0
        # Guard relevant thresholds using direct leave-one-out equal/core metrics.
        equal_core_pack = gram_pack(
            maps[contrast]["equal"]["S0"], maps[contrast]["equal"]["S8"], regions["core"]
        )
        equal_core_loo = weighted_metrics(equal_core_pack, loo)
        core_point = point_lookup[(contrast, "full", "equal", "core")]
        d_crossing = bool(np.any(
            (equal_core_loo["d_raw"] >= 0.25) != (float(core_point["d_raw"]) >= 0.25)
        ))
        r_crossing = bool(np.any(
            (equal_core_loo["pearson_r"] <= 0.80) != (float(core_point["pearson_r"]) <= 0.80)
        ))
        residual_guard &= not d_crossing and not r_crossing
        for metric, values, threshold, crossing in (
            ("core_d_raw", equal_core_loo["d_raw"], 0.25, d_crossing),
            ("core_pearson_r", equal_core_loo["pearson_r"], 0.80, r_crossing),
        ):
            estimate = float(core_point["d_raw" if metric == "core_d_raw" else "pearson_r"])
            robustness_rows.append({
                "analysis": "leave_one_subject_out",
                "contrast": contrast,
                "strategy": "equal",
                "metric": metric,
                "estimate": estimate,
                "lower_95": "",
                "upper_95": "",
                "replicates": len(values),
                "seed": "",
                "threshold": threshold,
                "any_threshold_crossing": crossing,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            })
    residual_candidate = residual_support and residual_guard

    threshold_candidate = True
    threshold_lookup = {(row["contrast"], row["scope"]): row for row in threshold_rows}
    for contrast in CONTRASTS:
        for scope in ("full", "EI", "ER"):
            row = threshold_lookup[(contrast, scope)]
            threshold_candidate &= (
                float(row["continuous_pearson_r"]) >= 0.80
                and float(row["fdr_jaccard"]) < 0.25
                and float(row["top_abs_t_jaccard"]) >= 0.50
            )
    threshold_split = True
    for contrast in CONTRASTS:
        for scope in ("split_A", "split_B"):
            row = threshold_lookup[(contrast, scope)]
            threshold_split &= (
                int(row["matched_k"]) > 0
                and float(row["continuous_pearson_r"]) >= 0.0
                and float(row["fdr_jaccard"]) < float(row["top_abs_t_jaccard"])
            )
    threshold_guard = all(
        not bool(loo_lookup[(contrast, "ivw", "pearson_r")]["any_threshold_crossing"])
        for contrast in CONTRASTS
    )
    threshold_candidate &= threshold_split and threshold_guard

    # Persist after the residual-core influence rows have also been appended.
    write_tsv(OUT / "robustness_results.tsv", robustness_rows)

    if edge_candidate:
        terminal_status, candidate = "candidate_ready", "edge_weighting_mechanism"
    elif amplitude_candidate:
        terminal_status, candidate = "candidate_ready", "amplitude_offset_mechanism"
    elif residual_candidate:
        terminal_status, candidate = "candidate_ready", "residual_substantive_spatial_sensitivity"
    elif threshold_candidate:
        terminal_status, candidate = "candidate_ready", "continuous_threshold_disconnect"
    else:
        terminal_status, candidate = "closed_no_candidate", None

    decision = {
        "schema_version": "narps.episode02.frozen_decision.v1",
        "terminal_status": terminal_status,
        "selected_candidate": candidate,
        "rules": {
            "amplitude_support": bool(amplitude_support),
            "amplitude_split_direction": bool(amplitude_split),
            "amplitude_small_subject_guard": bool(amplitude_guard),
            "edge_support": bool(edge_support),
            "edge_split_direction": bool(edge_split),
            "edge_small_subject_guard": bool(edge_guard),
            "residual_support": bool(residual_support),
            "residual_small_subject_guard": bool(residual_guard),
            "threshold_disconnect_support": bool(threshold_candidate),
            "threshold_disconnect_split_direction": bool(threshold_split),
            "threshold_disconnect_small_subject_guard": bool(threshold_guard),
        },
        "contract_sha256": CONTRACT_SHA256,
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    # Compact summary figure.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    x = np.arange(len(CONTRASTS))
    attenuation = [point_lookup[(c, "full", "ivw", "common")]["scale_attenuation"] for c in CONTRASTS]
    correlation = [point_lookup[(c, "full", "ivw", "common")]["pearson_r"] for c in CONTRASTS]
    axes[0, 0].bar(x - 0.18, attenuation, width=0.36, label="scale attenuation")
    axes[0, 0].bar(x + 0.18, correlation, width=0.36, label="Pearson r")
    axes[0, 0].axhline(0.5, color="C0", ls="--", lw=1)
    axes[0, 0].axhline(0.8, color="C1", ls="--", lw=1)
    axes[0, 0].set_title("Test 1: amplitude/shape")
    axes[0, 0].legend(fontsize=8)
    edge = [point_lookup[(c, "full", "ivw", "common")]["edge_enrichment"] for c in CONTRASTS]
    core_ratio = [point_lookup[(c, "full", "ivw", "common")]["core_ratio"] for c in CONTRASTS]
    axes[0, 1].bar(x - 0.18, edge, width=0.36, label="edge enrichment")
    axes[0, 1].bar(x + 0.18, core_ratio, width=0.36, label="core/full ratio")
    axes[0, 1].axhline(2.0, color="C0", ls="--", lw=1)
    axes[0, 1].axhline(0.75, color="C1", ls="--", lw=1)
    axes[0, 1].set_title("Test 2: edge")
    axes[0, 1].legend(fontsize=8)
    width = 0.24
    for j, strategy in enumerate(STRATEGIES):
        vals = [point_lookup[(c, "full", strategy, "common")]["d_raw"] for c in CONTRASTS]
        axes[1, 0].bar(x + (j - 1) * width, vals, width=width, label=strategy)
    axes[1, 0].set_title("Full-mask smoothing difference")
    axes[1, 0].legend(fontsize=8)
    fdr = [threshold_lookup[(c, "full")]["fdr_jaccard"] for c in CONTRASTS]
    top = [threshold_lookup[(c, "full")]["top_abs_t_jaccard"] for c in CONTRASTS]
    axes[1, 1].bar(x - 0.18, fdr, width=0.36, label="BH-FDR Jaccard")
    axes[1, 1].bar(x + 0.18, top, width=0.36, label="top-|t| Jaccard")
    axes[1, 1].set_title("Threshold/rank stability")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.set_xticks(x, ["gain", "loss"])
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("NARPS episode02 frozen mechanism tests")
    figure_dir = OUT / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir / "mechanism_summary.png", dpi=180)
    plt.close(fig)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
