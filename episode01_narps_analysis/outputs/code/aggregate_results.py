#!/usr/bin/env python
"""Aggregate immutable subject checkpoints for the frozen NARPS multiverse.

This is the only group/analysis stage.  It validates and loads all 108 subject
fixed-effect checkpoints, writes the 54 group intercept/t/BH-FDR map triplets,
and evaluates the frozen continuous-map, uncertainty, calibration, and
exploratory analyses.  Source datasets and ``inputs/`` are never written.

The Bayesian bootstrap and matched-size calibration use exact map inner
products from a subject-by-cell Gram matrix.  This is algebraically identical
to materializing each replicate map but avoids thousands of full-volume
replicate arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import traceback
from collections import OrderedDict
from pathlib import Path


# Keep caches underneath outputs even when this file is run outside sbatch.
SCRIPT_OUTPUTS = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_OUTPUTS / "cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_OUTPUTS / "cache"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
from scipy import stats

from narps_common import (
    CONFOUNDS,
    CONTRACT_HASHES,
    CONTRASTS,
    EXPECTED_EXCLUSIONS_Q1,
    EXPECTED_EXCLUSIONS_Q2,
    OUTPUTS,
    QC_SETS,
    SMOOTHINGS,
    SMOOTHING_FWHM,
    WORKSPACE,
    assert_frozen_contract,
    load_mask_vector,
    participants,
    save_json_atomic,
    save_tsv_atomic,
    sha256,
)


N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260815
N_DELETION = 1000
DELETION_SEED = 20260814
GROUPS = ("equalIndifference", "equalRange")
Q_EXPECTED = {
    "Q0_all": (108, 54, 54),
    "Q1_lenient": (106, 54, 52),
    "Q2_strict": (103, 54, 49),
}
Q_SHORT = {"Q0_all": "Q0", "Q1_lenient": "Q1", "Q2_strict": "Q2"}
BASE_CELLS = tuple((c, s) for c in CONFOUNDS for s in SMOOTHINGS)
# Internal order is C, S, Q.  This keeps each base first-level cell contiguous.
CELLS = tuple((c, q, s) for c, s in BASE_CELLS for q in QC_SETS)
INTERACTIONS = {"C:Q", "C:S", "Q:S", "C:Q:S"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gram-voxel-chunk",
        type=int,
        default=8192,
        help="Voxel block used for float64 subject/cell Gram accumulation.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic algebra/unit checks only; do not access outcomes.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def boolean_column(series: pd.Series, name: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false"}
    if not set(normalized).issubset(allowed):
        raise RuntimeError(f"{name} is not a strict Boolean column")
    return normalized.eq("true").to_numpy(dtype=bool)


def save_bytes_once(path: Path, payload: bytes) -> None:
    """Create an immutable artifact, or verify an identical prior artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to overwrite nonidentical result: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def save_text_once(path: Path, text: str) -> None:
    save_bytes_once(path, text.encode("utf-8"))


def save_table_once(path: Path, table: pd.DataFrame) -> None:
    text = table.to_csv(sep="\t", index=False, lineterminator="\n")
    save_text_once(path, text)


def save_json_once(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    save_text_once(path, text)


def save_figure_once(path: Path, figure: plt.Figure, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(
        temporary,
        dpi=dpi,
        bbox_inches="tight",
        metadata={"Software": "NARPS aggregate_results.py"},
    )
    plt.close(figure)
    if path.exists():
        if file_sha256(path) != file_sha256(temporary):
            temporary.unlink()
            raise FileExistsError(f"Refusing to overwrite nonidentical figure: {path}")
        temporary.unlink()
        return
    temporary.replace(path)


def save_masked_nifti_once(
    path: Path,
    vector: np.ndarray,
    mask_image: nib.Nifti1Image,
    mask: np.ndarray,
    dtype: np.dtype,
) -> None:
    expected = np.asarray(vector, dtype=dtype)
    if expected.shape != (int(mask.sum()),):
        raise ValueError(f"{path}: vector has shape {expected.shape}")
    if path.exists():
        old = nib.load(path)
        if old.shape != mask.shape or not np.allclose(old.affine, mask_image.affine):
            raise FileExistsError(f"Existing result has a different grid: {path}")
        observed = np.asarray(old.dataobj)[mask].astype(dtype, copy=False)
        if not np.array_equal(observed, expected, equal_nan=False):
            raise FileExistsError(f"Refusing to overwrite nonidentical map: {path}")
        return
    data = np.zeros(mask.shape, dtype=dtype)
    data[mask] = expected
    header = mask_image.header.copy()
    header.set_data_dtype(dtype)
    header.set_slope_inter(None, None)
    image = nib.Nifti1Image(data, mask_image.affine, header)
    temporary = path.with_name(path.name.removesuffix(".nii.gz") + ".tmp.nii.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, temporary)
    temporary.replace(path)


def load_and_validate_qc() -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    people = participants()
    qc = pd.read_csv(OUTPUTS / "qc_subject_sets.tsv", sep="\t")
    required = {"participant_id", "group", *QC_SETS}
    if not required.issubset(qc.columns):
        raise RuntimeError(f"QC table lacks {sorted(required - set(qc.columns))}")
    if qc["participant_id"].duplicated().any() or len(qc) != 108:
        raise RuntimeError("QC table must contain 108 unique subjects")
    qc = qc.set_index("participant_id").loc[people.participant_id].reset_index()
    if not np.array_equal(qc["group"].astype(str), people["group"].astype(str)):
        raise RuntimeError("QC cohort labels disagree with participants.tsv")
    memberships = {q: boolean_column(qc[q], q) for q in QC_SETS}
    if not (
        np.all(memberships["Q2_strict"] <= memberships["Q1_lenient"])
        and np.all(memberships["Q1_lenient"] <= memberships["Q0_all"])
    ):
        raise RuntimeError("Frozen QC nesting is violated")
    is_ei = qc["group"].eq("equalIndifference").to_numpy()
    for q, (total, ei, er) in Q_EXPECTED.items():
        observed = (
            int(memberships[q].sum()),
            int((memberships[q] & is_ei).sum()),
            int((memberships[q] & ~is_ei).sum()),
        )
        if observed != (total, ei, er):
            raise RuntimeError(f"{q} count mismatch: {observed} != {(total, ei, er)}")
    ids = qc["participant_id"].astype(str)
    q1_excluded = set(ids[~memberships["Q1_lenient"]])
    q2_excluded = set(ids[~memberships["Q2_strict"]])
    if q1_excluded != EXPECTED_EXCLUSIONS_Q1:
        raise RuntimeError(f"Q1 exclusions differ: {sorted(q1_excluded)}")
    if q2_excluded != EXPECTED_EXCLUSIONS_Q2:
        raise RuntimeError(f"Q2 exclusions differ: {sorted(q2_excluded)}")
    if any(is_ei[ids.eq(subject).to_numpy()][0] for subject in q2_excluded):
        raise RuntimeError("Every frozen exclusion must be an ER subject")
    return qc, is_ei, memberships


def observed_group_weights(
    is_ei: np.ndarray, memberships: dict[str, np.ndarray]
) -> np.ndarray:
    weights = np.zeros((len(QC_SETS), len(is_ei)), dtype=np.float64)
    for qi, q in enumerate(QC_SETS):
        eligible = memberships[q]
        for cohort in (is_ei, ~is_ei):
            selected = eligible & cohort
            if not selected.any():
                raise RuntimeError(f"{q} has an empty task-version cohort")
            weights[qi, selected] = 0.5 / selected.sum()
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-14):
        raise RuntimeError("Observed group weights do not sum to one")
    return weights


def load_subject_checkpoints(
    qc: pd.DataFrame, n_voxels: int, mask_hash: str
) -> tuple[np.ndarray, dict[str, str]]:
    subject_dir = OUTPUTS / "subject_maps"
    expected_paths = {
        subject: subject_dir / f"{subject}_fixed_effects.npz"
        for subject in qc.participant_id.astype(str)
    }
    observed_paths = set(subject_dir.glob("sub-*_fixed_effects.npz"))
    missing = [str(path) for path in expected_paths.values() if not path.is_file()]
    extra = sorted(str(path) for path in observed_paths - set(expected_paths.values()))
    if missing or extra:
        raise RuntimeError(
            f"Checkpoint inventory must be exactly 108 (missing={missing[:8]}, "
            f"extra={extra[:8]})"
        )
    shape = (
        len(CONTRASTS), len(qc), len(BASE_CELLS), n_voxels
    )
    effects = np.empty(shape, dtype=np.float32)
    hashes: dict[str, str] = {}
    expected_shape = (
        len(CONFOUNDS), len(SMOOTHINGS), len(CONTRASTS), n_voxels
    )
    for subject_index, subject in enumerate(qc.participant_id.astype(str)):
        path = expected_paths[subject]
        with np.load(path, allow_pickle=False) as payload:
            required = {
                "effect", "variance", "confounds", "smoothings", "contrasts",
                "subject", "mask_sha256",
            }
            if not required.issubset(payload.files):
                raise RuntimeError(f"{path} lacks {sorted(required - set(payload.files))}")
            if str(payload["subject"].item()) != subject:
                raise RuntimeError(f"{path}: subject identity mismatch")
            if str(payload["mask_sha256"].item()) != mask_hash:
                raise RuntimeError(f"{path}: frozen-mask hash mismatch")
            if tuple(payload["confounds"].astype(str)) != tuple(CONFOUNDS):
                raise RuntimeError(f"{path}: confound axis mismatch")
            if tuple(payload["smoothings"].astype(str)) != tuple(SMOOTHINGS):
                raise RuntimeError(f"{path}: smoothing axis mismatch")
            if tuple(payload["contrasts"].astype(str)) != tuple(CONTRASTS):
                raise RuntimeError(f"{path}: contrast axis mismatch")
            effect = np.asarray(payload["effect"])
            variance = np.asarray(payload["variance"])
            if effect.shape != expected_shape or variance.shape != expected_shape:
                raise RuntimeError(
                    f"{path}: shapes {effect.shape}/{variance.shape} != {expected_shape}"
                )
            if effect.dtype.kind != "f" or not np.isfinite(effect).all():
                raise RuntimeError(f"{path}: nonfinite or non-floating effects")
            if not np.isfinite(variance).all() or not np.all(variance > 0):
                raise RuntimeError(f"{path}: invalid fixed-effect variances")
            for contrast_index in range(len(CONTRASTS)):
                effects[contrast_index, subject_index] = np.asarray(
                    effect[:, :, contrast_index, :], dtype=np.float32
                ).reshape(len(BASE_CELLS), n_voxels)
        companion = path.with_suffix(".json")
        if not companion.is_file():
            raise RuntimeError(f"Missing checkpoint provenance: {companion}")
        metadata = json.loads(companion.read_text(encoding="utf-8"))
        if (
            metadata.get("schema_version") != "narps.subject_fixed_effects.v1"
            or metadata.get("subject") != subject
            or metadata.get("pilot") is not False
            or int(metadata.get("n_voxels", -1)) != n_voxels
            or metadata.get("mask_sha256") != mask_hash
        ):
            raise RuntimeError(f"Invalid checkpoint provenance: {companion}")
        hashes[subject] = sha256(path)
        if (subject_index + 1) % 12 == 0 or subject_index + 1 == len(qc):
            print(f"Validated and loaded {subject_index + 1}/108 checkpoints", flush=True)
    return effects, hashes


def group_intercept_t(
    values: np.ndarray,
    eligible: np.ndarray,
    is_ei: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """OLS intercept and t for coding EI=+0.5, ER=-0.5."""
    if values.ndim != 2 or values.shape[0] != len(eligible):
        raise ValueError("values must be subject x voxel")
    ei = eligible & is_ei
    er = eligible & ~is_ei
    n_ei, n_er = int(ei.sum()), int(er.sum())
    n_total = n_ei + n_er
    df = n_total - 2
    if n_ei < 2 or n_er < 2 or df <= 0:
        raise RuntimeError("Group OLS has insufficient cohort sample size")
    y_ei = values[ei]
    y_er = values[er]
    sum_ei = np.sum(y_ei, axis=0, dtype=np.float64)
    sum_er = np.sum(y_er, axis=0, dtype=np.float64)
    mean_ei = sum_ei / n_ei
    mean_er = sum_er / n_er
    intercept = 0.5 * (mean_ei + mean_er)
    ss_ei = np.einsum("ij,ij->j", y_ei, y_ei, dtype=np.float64) - (
        sum_ei * sum_ei / n_ei
    )
    ss_er = np.einsum("ij,ij->j", y_er, y_er, dtype=np.float64) - (
        sum_er * sum_er / n_er
    )
    sse = ss_ei + ss_er
    tolerance = np.finfo(np.float64).eps * np.maximum(
        1.0,
        np.einsum("ij,ij->j", y_ei, y_ei, dtype=np.float64)
        + np.einsum("ij,ij->j", y_er, y_er, dtype=np.float64),
    ) * 100.0
    if np.any(sse < -tolerance):
        raise RuntimeError("Negative group residual sum of squares beyond roundoff")
    sse = np.maximum(sse, 0.0)
    intercept_variance = (sse / df) * 0.25 * (1.0 / n_ei + 1.0 / n_er)
    if np.any(~np.isfinite(intercept_variance)) or np.any(intercept_variance <= 0):
        count = int((~np.isfinite(intercept_variance) | (intercept_variance <= 0)).sum())
        raise RuntimeError(f"Group intercept t undefined in {count} mask voxels")
    t_stat = intercept / np.sqrt(intercept_variance)
    if not np.isfinite(intercept).all() or not np.isfinite(t_stat).all():
        raise RuntimeError("Nonfinite group intercept or t statistic")
    return intercept, t_stat, df


def bh_rejections(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("BH input must be a finite one-dimensional p-value vector")
    order = np.argsort(p, kind="mergesort")
    ordered = p[order]
    passed = ordered <= q * np.arange(1, len(p) + 1, dtype=np.float64) / len(p)
    if not passed.any():
        return np.zeros(len(p), dtype=bool)
    cutoff = ordered[np.flatnonzero(passed)[-1]]
    return p <= cutoff


def projection_matrices() -> OrderedDict[str, np.ndarray]:
    identity = np.eye(3, dtype=np.float64)
    mean = np.ones((3, 3), dtype=np.float64) / 3.0
    centered = identity - mean
    # Internal cell order is C, S, Q.
    matrices = OrderedDict([
        ("C", np.kron(np.kron(centered, mean), mean)),
        ("Q", np.kron(np.kron(mean, mean), centered)),
        ("S", np.kron(np.kron(mean, centered), mean)),
        ("C:Q", np.kron(np.kron(centered, mean), centered)),
        ("C:S", np.kron(np.kron(centered, centered), mean)),
        ("Q:S", np.kron(np.kron(mean, centered), centered)),
        ("C:Q:S", np.kron(np.kron(centered, centered), centered)),
    ])
    grand = np.kron(np.kron(mean, mean), mean)
    total = np.eye(27) - grand
    if not np.allclose(sum(matrices.values()), total, atol=2e-14):
        raise RuntimeError("Functional-ANOVA projections do not span total variation")
    for name, matrix in matrices.items():
        if not (
            np.allclose(matrix, matrix.T, atol=2e-14)
            and np.allclose(matrix @ matrix, matrix, atol=2e-14)
        ):
            raise RuntimeError(f"Invalid ANOVA projection {name}")
    return matrices


def nonnegative_scalar(value: float, scale: float, label: str) -> float:
    tolerance = 1e-9 * max(1.0, abs(scale))
    if value < -tolerance:
        raise RuntimeError(f"{label} is negative beyond roundoff: {value}")
    return max(0.0, float(value))


def functional_anova(
    gram: np.ndarray,
    projections: OrderedDict[str, np.ndarray],
) -> dict:
    if gram.shape != (27, 27) or not np.isfinite(gram).all():
        raise ValueError("Functional ANOVA needs a finite 27 x 27 Gram matrix")
    total_projection = sum(projections.values())
    scale = float(np.trace(gram))
    ss = {
        name: nonnegative_scalar(
            float(np.einsum("ij,ji->", matrix, gram)), scale, name
        )
        for name, matrix in projections.items()
    }
    total_direct = nonnegative_scalar(
        float(np.einsum("ij,ji->", total_projection, gram)), scale, "total"
    )
    total_terms = float(sum(ss.values()))
    if not np.isclose(total_terms, total_direct, rtol=1e-9, atol=1e-10 * max(1.0, scale)):
        raise RuntimeError(
            f"ANOVA partition failure: terms={total_terms}, direct={total_direct}"
        )
    if total_direct <= np.finfo(float).eps * max(1.0, scale):
        raise RuntimeError("Across-cell map variance is numerically zero")
    attributed = {
        "C": ss["C"] + 0.5 * ss["C:Q"] + 0.5 * ss["C:S"]
             + ss["C:Q:S"] / 3.0,
        "Q": ss["Q"] + 0.5 * ss["C:Q"] + 0.5 * ss["Q:S"]
             + ss["C:Q:S"] / 3.0,
        "S": ss["S"] + 0.5 * ss["C:S"] + 0.5 * ss["Q:S"]
             + ss["C:Q:S"] / 3.0,
    }
    if not np.isclose(sum(attributed.values()), total_direct, rtol=1e-9, atol=1e-10):
        raise RuntimeError("Shapley attribution does not sum to total SS")
    if attributed["S"] <= np.finfo(float).eps * total_direct:
        raise RuntimeError("Smoothing-attributed variance is numerically zero")
    return {
        "ss": ss,
        "total": total_direct,
        "attributed": attributed,
        "term_share": {key: value / total_direct for key, value in ss.items()},
        "factor_share": {
            key: value / total_direct for key, value in attributed.items()
        },
        "ratios": {
            "C_over_S": attributed["C"] / attributed["S"],
            "Q_over_S": attributed["Q"] / attributed["S"],
            "C_plus_Q_over_S": (
                attributed["C"] + attributed["Q"]
            ) / attributed["S"],
        },
        "partition_relative_error": abs(total_terms - total_direct) / total_direct,
    }


def subject_cell_gram(
    effects: np.ndarray, voxel_chunk: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return G[b,n,d,m]=dot(map[n,b], map[m,d]) and spatial sums."""
    if effects.ndim != 3:
        raise ValueError("effects must be subject x base-cell x voxel")
    n_subjects, n_base, n_voxels = effects.shape
    if n_base != 9 or voxel_chunk < 64:
        raise ValueError("Unexpected base-cell count or Gram chunk")
    n_flat = n_base * n_subjects
    gram = np.zeros((n_flat, n_flat), dtype=np.float64)
    for start in range(0, n_voxels, voxel_chunk):
        stop = min(start + voxel_chunk, n_voxels)
        block = np.ascontiguousarray(
            effects[:, :, start:stop].transpose(1, 0, 2).reshape(n_flat, stop - start),
            dtype=np.float64,
        )
        gram += block @ block.T
        print(f"  Gram voxels {stop}/{n_voxels}", flush=True)
    gram = 0.5 * (gram + gram.T)
    spatial_sums = np.sum(effects, axis=2, dtype=np.float64)
    return gram.reshape(n_base, n_subjects, n_base, n_subjects), spatial_sums


def paired_bootstrap_weights(
    is_ei: np.ndarray,
    memberships: dict[str, np.ndarray],
) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    raw = rng.exponential(scale=1.0, size=(N_BOOTSTRAP, len(is_ei)))
    coefficients = np.zeros(
        (N_BOOTSTRAP, len(QC_SETS), len(is_ei)), dtype=np.float64
    )
    for qi, q in enumerate(QC_SETS):
        eligible = memberships[q]
        for cohort in (is_ei, ~is_ei):
            selected = eligible & cohort
            denominator = raw[:, selected].sum(axis=1)
            if np.any(denominator <= 0):
                raise RuntimeError("Bayesian-bootstrap cohort weight sum is zero")
            coefficients[:, qi, selected] = (
                0.5 * raw[:, selected] / denominator[:, None]
            )
    if not np.allclose(coefficients.sum(axis=2), 1.0, atol=2e-14):
        raise RuntimeError("Bayesian-bootstrap group weights do not sum to one")
    return coefficients


def group_grams_from_subject_gram(
    subject_gram: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    """Exact group-map Gram matrices for many weight replicates."""
    n_rep, n_q, n_subjects = coefficients.shape
    n_base = subject_gram.shape[0]
    if subject_gram.shape != (n_base, n_subjects, n_base, n_subjects):
        raise ValueError("Subject Gram shape disagrees with coefficients")
    result = np.empty(
        (n_rep, n_base, n_q, n_base, n_q), dtype=np.float64
    )
    flat_coefficients = coefficients.reshape(n_rep * n_q, n_subjects)
    for b in range(n_base):
        for d in range(b, n_base):
            left = (flat_coefficients @ subject_gram[b, :, d, :]).reshape(
                n_rep, n_q, n_subjects
            )
            block = np.einsum(
                "rqn,rpn->rqp", left, coefficients, optimize=True
            )
            result[:, b, :, d, :] = block
            if d != b:
                result[:, d, :, b, :] = block.transpose(0, 2, 1)
    return result.reshape(n_rep, n_base * n_q, n_base * n_q)


def bootstrap_anova(
    grams: np.ndarray,
    projections: OrderedDict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    scale = np.trace(grams, axis1=1, axis2=2)
    term_ss: dict[str, np.ndarray] = {}
    for name, matrix in projections.items():
        value = np.einsum("ij,rji->r", matrix, grams, optimize=True)
        tolerance = 1e-8 * np.maximum(1.0, np.abs(scale))
        if np.any(value < -tolerance):
            raise RuntimeError(f"Bootstrap {name} SS is negative beyond roundoff")
        term_ss[name] = np.maximum(value, 0.0)
    total = sum(term_ss.values())
    if np.any(total <= np.finfo(float).eps * np.maximum(1.0, np.abs(scale))):
        raise RuntimeError("Bootstrap across-cell variance is numerically zero")
    attributed = {
        "C": term_ss["C"] + 0.5 * term_ss["C:Q"]
             + 0.5 * term_ss["C:S"] + term_ss["C:Q:S"] / 3.0,
        "Q": term_ss["Q"] + 0.5 * term_ss["C:Q"]
             + 0.5 * term_ss["Q:S"] + term_ss["C:Q:S"] / 3.0,
        "S": term_ss["S"] + 0.5 * term_ss["C:S"]
             + 0.5 * term_ss["Q:S"] + term_ss["C:Q:S"] / 3.0,
    }
    if np.any(attributed["S"] <= np.finfo(float).eps * total):
        raise RuntimeError("Bootstrap smoothing attribution is numerically zero")
    ratios = {
        "C_over_S": attributed["C"] / attributed["S"],
        "Q_over_S": attributed["Q"] / attributed["S"],
        "C_plus_Q_over_S": (
            attributed["C"] + attributed["Q"]
        ) / attributed["S"],
    }
    return {
        "term_ss": term_ss,
        "term_share": {key: value / total for key, value in term_ss.items()},
        "attributed": attributed,
        "factor_share": {key: value / total for key, value in attributed.items()},
        "ratios": ratios,
        "total": {"total": total},
    }


def percentile_interval(values: np.ndarray, coverage: float = 0.95) -> tuple[float, float]:
    tail = (1.0 - coverage) * 50.0
    low, high = np.percentile(np.asarray(values), [tail, 100.0 - tail])
    return float(low), float(high)


def spatial_metrics_from_gram(
    gram: np.ndarray,
    spatial_sums: np.ndarray,
    n_voxels: int,
    reference_index: int = 0,
) -> dict[str, np.ndarray]:
    """Pairwise Pearson/cosine/NRMSD for one or many Gram matrices."""
    many = gram.ndim == 3
    if not many:
        gram = gram[None, :, :]
        spatial_sums = spatial_sums[None, :]
    n_rep, n_cells, _ = gram.shape
    if spatial_sums.shape != (n_rep, n_cells):
        raise ValueError("Spatial-sum shape disagrees with Gram matrices")
    diagonal = np.diagonal(gram, axis1=1, axis2=2)
    centered = diagonal - spatial_sums * spatial_sums / n_voxels
    scale = np.maximum(1.0, np.abs(diagonal))
    if np.any(centered < -1e-9 * scale):
        raise RuntimeError("Negative centered map norm beyond roundoff")
    centered = np.maximum(centered, 0.0)
    if np.any(centered <= np.finfo(float).eps * scale):
        raise RuntimeError("Spatial Pearson correlation undefined for constant map")
    if np.any(diagonal <= np.finfo(float).eps):
        raise RuntimeError("Cosine similarity undefined for zero-norm map")
    reference_norm = diagonal[:, reference_index]
    if np.any(reference_norm <= np.finfo(float).eps):
        raise RuntimeError("Frozen NRMSD reference denominator is zero")
    ii, jj = np.triu_indices(n_cells, k=1)
    dot = gram[:, ii, jj]
    cosine = dot / np.sqrt(diagonal[:, ii] * diagonal[:, jj])
    centered_dot = dot - spatial_sums[:, ii] * spatial_sums[:, jj] / n_voxels
    pearson = centered_dot / np.sqrt(centered[:, ii] * centered[:, jj])
    difference_norm = diagonal[:, ii] + diagonal[:, jj] - 2.0 * dot
    tolerance = 1e-9 * np.maximum(
        1.0, diagonal[:, ii] + diagonal[:, jj] + 2.0 * np.abs(dot)
    )
    if np.any(difference_norm < -tolerance):
        raise RuntimeError("Negative pairwise squared difference beyond roundoff")
    nrmsd = np.sqrt(np.maximum(difference_norm, 0.0) / reference_norm[:, None])
    for name, values in {"pearson": pearson, "cosine": cosine}.items():
        if np.any(~np.isfinite(values)) or np.any(np.abs(values) > 1.0 + 1e-7):
            raise RuntimeError(f"Invalid {name} values")
        values[:] = np.clip(values, -1.0, 1.0)
    result = {
        "i": ii,
        "j": jj,
        "pearson_r": pearson,
        "cosine_similarity": cosine,
        "normalized_rms_difference": nrmsd,
    }
    if not many:
        return {
            key: value[0] if key not in {"i", "j"} else value
            for key, value in result.items()
        }
    return result


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.count_nonzero(a | b))
    return 1.0 if union == 0 else float(np.count_nonzero(a & b) / union)


def load_manifest() -> pd.DataFrame:
    path = OUTPUTS / "multiverse_manifest.tsv"
    manifest = pd.read_csv(path, sep="\t", keep_default_na=False)
    required = {
        "contrast", "confound", "qc", "smoothing", "fwhm_mm",
        "expected_subjects", "status", "effect_map", "t_map", "fdr_map", "error",
    }
    if not required.issubset(manifest.columns) or len(manifest) != 54:
        raise RuntimeError("Multiverse manifest schema/count is invalid")
    expected = {
        (contrast, c, q, s)
        for contrast in CONTRASTS for c in CONFOUNDS
        for q in QC_SETS for s in SMOOTHINGS
    }
    observed = set(zip(
        manifest.contrast, manifest.confound, manifest.qc, manifest.smoothing
    ))
    if observed != expected or len(observed) != len(manifest):
        raise RuntimeError("Multiverse manifest cells differ from frozen grid")
    return manifest


def update_manifest_cell(
    manifest: pd.DataFrame,
    contrast: str,
    c: str,
    q: str,
    s: str,
    status: str,
    paths: tuple[Path, Path, Path] | None = None,
    error: str = "",
) -> None:
    selector = (
        manifest.contrast.eq(contrast) & manifest.confound.eq(c)
        & manifest.qc.eq(q) & manifest.smoothing.eq(s)
    )
    if int(selector.sum()) != 1:
        raise RuntimeError("Cannot resolve one frozen manifest cell")
    manifest.loc[selector, "status"] = status
    manifest.loc[selector, "error"] = error[:1000]
    if paths is not None:
        manifest.loc[selector, "effect_map"] = str(paths[0].relative_to(WORKSPACE))
        manifest.loc[selector, "t_map"] = str(paths[1].relative_to(WORKSPACE))
        manifest.loc[selector, "fdr_map"] = str(paths[2].relative_to(WORKSPACE))
    save_tsv_atomic(OUTPUTS / "multiverse_manifest.tsv", manifest)


def map_paths(contrast: str, c: str, q: str, s: str) -> tuple[Path, Path, Path]:
    stem = f"{contrast}_{c}_{Q_SHORT[q]}_{s}"
    root = OUTPUTS / "group_maps"
    return (
        root / f"{stem}_effect.nii.gz",
        root / f"{stem}_t.nii.gz",
        root / f"{stem}_fdr_q-0p05.nii.gz",
    )


def generate_random_deletions(
    qc: pd.DataFrame,
    is_ei: np.ndarray,
    memberships: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    transitions = OrderedDict([
        ("Q0_to_Q1", ("Q0_all", 2)),
        ("Q1_to_Q2", ("Q1_lenient", 3)),
    ])
    rng = np.random.Generator(np.random.PCG64(DELETION_SEED))
    targets = {
        name: np.zeros((N_DELETION, len(qc)), dtype=np.float64)
        for name in transitions
    }
    sample_rows: list[dict] = []
    ids = qc.participant_id.astype(str).to_numpy()
    # Replicate-major draw order is frozen here and recorded in the sample table.
    for replicate in range(N_DELETION):
        for name, (base_q, n_remove) in transitions.items():
            base = memberships[base_q]
            er_pool = np.flatnonzero(base & ~is_ei)
            removed = np.sort(rng.choice(er_pool, size=n_remove, replace=False))
            target = base.copy()
            target[removed] = False
            for cohort in (is_ei, ~is_ei):
                selected = target & cohort
                targets[name][replicate, selected] = 0.5 / selected.sum()
            sample_rows.append({
                "replicate": replicate + 1,
                "transition": name,
                "base_qc": base_q,
                "removed_ei": 0,
                "removed_er": n_remove,
                "removed_subjects": ";".join(ids[removed]),
                "rng": "PCG64",
                "seed": DELETION_SEED,
                "draw_order": "replicate_major_Q0_to_Q1_then_Q1_to_Q2",
            })
    for name, weights in targets.items():
        if not np.allclose(weights.sum(axis=1), 1.0, atol=2e-14):
            raise RuntimeError(f"{name} deletion weights do not sum to one")
    return targets, pd.DataFrame(sample_rows)


def matched_calibration(
    contrast: str,
    subject_gram: np.ndarray,
    subject_sums: np.ndarray,
    observed_maps: np.ndarray,
    observed_weights: np.ndarray,
    target_weights: dict[str, np.ndarray],
    n_voxels: int,
) -> tuple[list[dict], dict[tuple[str, str, str, str], np.ndarray]]:
    transition_info = OrderedDict([
        ("Q0_to_Q1", (0, 1, 2)),
        ("Q1_to_Q2", (1, 2, 3)),
    ])
    reference_norm = float(np.dot(observed_maps[0, 0], observed_maps[0, 0]))
    if reference_norm <= np.finfo(float).eps:
        raise RuntimeError(f"{contrast}: zero NRMSD reference")
    rows: list[dict] = []
    distributions: dict[tuple[str, str, str, str], np.ndarray] = {}
    for transition, (base_qi, target_qi, n_remove) in transition_info.items():
        base_weights = observed_weights[base_qi]
        random_weights = target_weights[transition]
        for b, (c, s) in enumerate(BASE_CELLS):
            gram = subject_gram[b, :, b, :]
            base_gram_vector = gram @ base_weights
            base_norm = float(base_weights @ base_gram_vector)
            random_left = random_weights @ gram
            random_norm = np.einsum(
                "rn,rn->r", random_left, random_weights, optimize=True
            )
            cross = random_weights @ base_gram_vector
            difference = random_norm + base_norm - 2.0 * cross
            tolerance = 1e-9 * np.maximum(
                1.0, random_norm + base_norm + 2.0 * np.abs(cross)
            )
            if np.any(difference < -tolerance):
                raise RuntimeError(f"{contrast} {transition} {c}/{s}: negative distance")
            null_nrmsd = np.sqrt(np.maximum(difference, 0.0) / reference_norm)

            base_sum = float(base_weights @ subject_sums[:, b])
            random_sum = random_weights @ subject_sums[:, b]
            centered_base = base_norm - base_sum * base_sum / n_voxels
            centered_random = random_norm - random_sum * random_sum / n_voxels
            centered_cross = cross - random_sum * base_sum / n_voxels
            if centered_base <= 0 or np.any(centered_random <= 0):
                raise RuntimeError("Matched calibration Pearson denominator is zero")
            null_correlation = centered_cross / np.sqrt(
                centered_random * centered_base
            )
            if np.any(~np.isfinite(null_correlation)) or np.any(
                np.abs(null_correlation) > 1.0 + 1e-7
            ):
                raise RuntimeError("Invalid matched-calibration Pearson correlation")
            null_one_minus_r = 1.0 - np.clip(null_correlation, -1.0, 1.0)

            before = observed_maps[b, base_qi]
            after = observed_maps[b, target_qi]
            observed_nrmsd = float(
                np.sqrt(np.dot(after - before, after - before) / reference_norm)
            )
            before_centered = before - before.mean()
            after_centered = after - after.mean()
            denom = np.linalg.norm(before_centered) * np.linalg.norm(after_centered)
            if denom <= 0:
                raise RuntimeError("Observed matched calibration Pearson is undefined")
            observed_correlation = float(
                np.dot(before_centered, after_centered) / denom
            )
            if not np.isfinite(observed_correlation) or abs(observed_correlation) > 1.0 + 1e-7:
                raise RuntimeError("Invalid observed matched-calibration correlation")
            observed_one_minus_r = float(
                1.0 - np.clip(observed_correlation, -1.0, 1.0)
            )
            for metric, observed, null in (
                ("normalized_rms_difference", observed_nrmsd, null_nrmsd),
                ("one_minus_pearson_r", observed_one_minus_r, null_one_minus_r),
            ):
                null = np.asarray(null, dtype=np.float64)
                if not np.isfinite(observed) or not np.isfinite(null).all():
                    raise RuntimeError(
                        f"{contrast} {transition} {c}/{s}: nonfinite {metric}"
                    )
                median = float(np.median(null))
                low, high = percentile_interval(null, 0.95)
                ratio = observed / median if median > 0 else (
                    1.0 if observed == 0 else float("inf")
                )
                empirical = float((1 + np.count_nonzero(null >= observed)) / 1001.0)
                rows.append({
                    "transition": transition,
                    "contrast": contrast,
                    "confound": c,
                    "smoothing": s,
                    "metric": metric,
                    "base_qc": QC_SETS[base_qi],
                    "observed_target_qc": QC_SETS[target_qi],
                    "removed_ei": 0,
                    "removed_er": n_remove,
                    "replicates": N_DELETION,
                    "rng": "PCG64",
                    "seed": DELETION_SEED,
                    "observed": observed,
                    "null_median": median,
                    "null_percentile_2p5": low,
                    "null_percentile_97p5": high,
                    "observed_to_null_median_ratio": ratio,
                    "upper_tail_empirical_p": empirical,
                    "interpretation": "sample_composition_calibration_not_causal_QC",
                })
                distributions[(contrast, transition, metric, f"{c}/{s}")] = null
    return rows, distributions


def er_leave_one_out_influence(
    contrast: str,
    subject_gram: np.ndarray,
    qc: pd.DataFrame,
    is_ei: np.ndarray,
) -> list[dict]:
    er_indices = np.flatnonzero(~is_ei)
    n_er = len(er_indices)
    if n_er != 54:
        raise RuntimeError("Exploratory ER diagnostic requires all 54 ER subjects")
    ids = qc.participant_id.astype(str).to_numpy()
    excluded = EXPECTED_EXCLUSIONS_Q2
    rows: list[dict] = []
    for b, (c, s) in enumerate(BASE_CELLS):
        gram = subject_gram[b, :, b, :][np.ix_(er_indices, er_indices)]
        mean_weights = np.full(n_er, 1.0 / n_er)
        mean_dot = gram @ mean_weights
        mean_norm = float(mean_weights @ mean_dot)
        if mean_norm <= np.finfo(float).eps:
            raise RuntimeError(f"{contrast} {c}/{s}: ER full-map norm is zero")
        residual_norm = np.diag(gram) - 2.0 * mean_dot + mean_norm
        tolerance = 1e-9 * np.maximum(1.0, np.abs(np.diag(gram)) + abs(mean_norm))
        if np.any(residual_norm < -tolerance):
            raise RuntimeError("Negative ER leave-one-out influence norm")
        influence = np.sqrt(np.maximum(residual_norm, 0.0) / mean_norm) / (n_er - 1)
        retained = np.array([ids[index] not in excluded for index in er_indices])
        retained_median = float(np.median(influence[retained]))
        retained_p95 = float(np.percentile(influence[retained], 95.0))
        for local_index, subject_index in enumerate(er_indices):
            subject = ids[subject_index]
            rows.append({
                "label": "exploratory_post_primary",
                "contrast": contrast,
                "confound": c,
                "smoothing": s,
                "participant_id": subject,
                "frozen_status": (
                    "frozen_excluded_ER" if subject in excluded else "retained_ER"
                ),
                "normalized_rms_loo_influence": float(influence[local_index]),
                "retained_er_median": retained_median,
                "retained_er_percentile_95": retained_p95,
                "influence_to_retained_median_ratio": (
                    float(influence[local_index] / retained_median)
                    if retained_median > 0 else float("inf")
                ),
                "full_er_n": n_er,
                "interpretation": "exploratory_descriptive_not_primary",
            })
    return rows


def variance_rows_for_contrast(
    contrast: str,
    observed: dict,
    bootstrap: dict[str, dict[str, np.ndarray]],
    n_voxels: int,
) -> list[dict]:
    rows: list[dict] = [{
        "contrast": contrast,
        "record_type": "anova_total",
        "component": "total_across_cell",
        "point_estimate": observed["total"],
        "sum_squares": observed["total"],
        "sum_squares_per_voxel": observed["total"] / n_voxels,
        "share": 1.0,
        "ci_coverage": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "interval_type": "not_applicable",
        "material_interaction": False,
        "bootstrap_replicates": N_BOOTSTRAP,
        "rng": "PCG64",
        "seed": BOOTSTRAP_SEED,
        "partition_relative_error": observed["partition_relative_error"],
    }]
    for term, value in observed["ss"].items():
        low, high = percentile_interval(bootstrap["term_share"][term], 0.95)
        share = observed["term_share"][term]
        rows.append({
            "contrast": contrast,
            "record_type": "anova_term",
            "component": term,
            "point_estimate": share,
            "sum_squares": value,
            "sum_squares_per_voxel": value / n_voxels,
            "share": share,
            "ci_coverage": 0.95,
            "ci_lower": low,
            "ci_upper": high,
            "interval_type": "paired_cohort_stratified_Bayesian_bootstrap_percentile",
            "material_interaction": bool(term in INTERACTIONS and share >= 0.05),
            "bootstrap_replicates": N_BOOTSTRAP,
            "rng": "PCG64",
            "seed": BOOTSTRAP_SEED,
            "partition_relative_error": observed["partition_relative_error"],
        })
    for factor, value in observed["attributed"].items():
        low, high = percentile_interval(bootstrap["factor_share"][factor], 0.95)
        share = observed["factor_share"][factor]
        rows.append({
            "contrast": contrast,
            "record_type": "shapley_factor",
            "component": factor,
            "point_estimate": share,
            "sum_squares": value,
            "sum_squares_per_voxel": value / n_voxels,
            "share": share,
            "ci_coverage": 0.95,
            "ci_lower": low,
            "ci_upper": high,
            "interval_type": "paired_cohort_stratified_Bayesian_bootstrap_percentile",
            "material_interaction": False,
            "bootstrap_replicates": N_BOOTSTRAP,
            "rng": "PCG64",
            "seed": BOOTSTRAP_SEED,
            "partition_relative_error": observed["partition_relative_error"],
        })
    for ratio, value in observed["ratios"].items():
        coverage = 0.975 if ratio in {"C_over_S", "Q_over_S"} else 0.95
        low, high = percentile_interval(bootstrap["ratios"][ratio], coverage)
        rows.append({
            "contrast": contrast,
            "record_type": "primary_ratio",
            "component": ratio,
            "point_estimate": value,
            "sum_squares": np.nan,
            "sum_squares_per_voxel": np.nan,
            "share": np.nan,
            "ci_coverage": coverage,
            "ci_lower": low,
            "ci_upper": high,
            "interval_type": (
                "Bonferroni_simultaneous_Bayesian_bootstrap_percentile"
                if coverage == 0.975
                else "paired_cohort_stratified_Bayesian_bootstrap_percentile"
            ),
            "material_interaction": False,
            "bootstrap_replicates": N_BOOTSTRAP,
            "rng": "PCG64",
            "seed": BOOTSTRAP_SEED,
            "partition_relative_error": observed["partition_relative_error"],
        })
    return rows


def pairwise_rows_for_contrast(
    contrast: str,
    observed_maps: np.ndarray,
    fdr_masks: np.ndarray,
    bootstrap_grams: np.ndarray,
    bootstrap_sums: np.ndarray,
    n_voxels: int,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    flat_maps = observed_maps.reshape(27, n_voxels)
    gram = flat_maps @ flat_maps.T
    sums = flat_maps.sum(axis=1, dtype=np.float64)
    observed_metrics = spatial_metrics_from_gram(gram, sums, n_voxels)
    bootstrap_metrics = spatial_metrics_from_gram(
        bootstrap_grams, bootstrap_sums, n_voxels
    )
    ii, jj = observed_metrics["i"], observed_metrics["j"]
    pearson_matrix = np.eye(27)
    jaccard_matrix = np.eye(27)
    rows: list[dict] = []
    metric_names = (
        "pearson_r", "cosine_similarity", "normalized_rms_difference"
    )
    for pair_index, (i, j) in enumerate(zip(ii, jj)):
        c1, q1, s1 = CELLS[int(i)]
        c2, q2, s2 = CELLS[int(j)]
        axes = [
            name for name, a, b in (
                ("C", c1, c2), ("Q", q1, q2), ("S", s1, s2)
            ) if a != b
        ]
        row = {
            "record_type": "pairwise_cell_stability",
            "contrast": contrast,
            "confound": "",
            "qc": "",
            "smoothing": "",
            "n_subjects": np.nan,
            "df": np.nan,
            "fdr_rejected_voxels": np.nan,
            "fdr_rejected_proportion": np.nan,
            "fdr_positive_voxels": np.nan,
            "fdr_negative_voxels": np.nan,
            "fdr_any_rejection": "",
            "cell_a": f"{c1}/{Q_SHORT[q1]}/{s1}",
            "cell_b": f"{c2}/{Q_SHORT[q2]}/{s2}",
            "differing_axes": ":".join(axes),
            "n_differing_axes": len(axes),
        }
        for metric in metric_names:
            point = float(observed_metrics[metric][pair_index])
            low, high = percentile_interval(
                bootstrap_metrics[metric][:, pair_index], 0.95
            )
            row[metric] = point
            row[f"{metric}_ci_lower"] = low
            row[f"{metric}_ci_upper"] = high
        signs_a = np.sign(flat_maps[i])
        signs_b = np.sign(flat_maps[j])
        row["sign_agreement"] = float(
            np.count_nonzero((signs_a == signs_b) & (signs_a != 0)) / n_voxels
        )
        fdr_j = jaccard(fdr_masks.reshape(27, n_voxels)[i],
                         fdr_masks.reshape(27, n_voxels)[j])
        row["fdr_jaccard"] = fdr_j
        row["bootstrap_replicates"] = N_BOOTSTRAP
        row["bootstrap_seed"] = BOOTSTRAP_SEED
        row["notes"] = "sign agreement and FDR Jaccard are descriptive"
        rows.append(row)
        pearson_matrix[i, j] = pearson_matrix[j, i] = row["pearson_r"]
        jaccard_matrix[i, j] = jaccard_matrix[j, i] = fdr_j
    return rows, pearson_matrix, jaccard_matrix


def cell_summary_row(
    contrast: str,
    c: str,
    q: str,
    s: str,
    n_subjects: int,
    df: int,
    effect: np.ndarray,
    rejected: np.ndarray,
) -> dict:
    count = int(rejected.sum())
    return {
        "record_type": "cell_bh_fdr_summary",
        "contrast": contrast,
        "confound": c,
        "qc": q,
        "smoothing": s,
        "n_subjects": n_subjects,
        "df": df,
        "fdr_rejected_voxels": count,
        "fdr_rejected_proportion": count / len(rejected),
        "fdr_positive_voxels": int(np.count_nonzero(rejected & (effect > 0))),
        "fdr_negative_voxels": int(np.count_nonzero(rejected & (effect < 0))),
        "fdr_any_rejection": bool(count),
        "cell_a": "",
        "cell_b": "",
        "differing_axes": "",
        "n_differing_axes": np.nan,
        "pearson_r": np.nan,
        "pearson_r_ci_lower": np.nan,
        "pearson_r_ci_upper": np.nan,
        "cosine_similarity": np.nan,
        "cosine_similarity_ci_lower": np.nan,
        "cosine_similarity_ci_upper": np.nan,
        "normalized_rms_difference": np.nan,
        "normalized_rms_difference_ci_lower": np.nan,
        "normalized_rms_difference_ci_upper": np.nan,
        "sign_agreement": np.nan,
        "fdr_jaccard": np.nan,
        "bootstrap_replicates": np.nan,
        "bootstrap_seed": np.nan,
        "notes": "BH-FDR q<=0.05 is secondary; threshold is not a factor",
    }


def plot_variance_attribution(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    colors = {"C": "#4477AA", "Q": "#CC6677", "S": "#228833"}
    for axis, contrast in zip(axes, CONTRASTS):
        subset = table[
            (table.contrast == contrast) & (table.record_type == "shapley_factor")
        ].set_index("component").loc[list("CQS")]
        x = np.arange(3)
        values = subset["share"].to_numpy(float)
        low = subset["ci_lower"].to_numpy(float)
        high = subset["ci_upper"].to_numpy(float)
        axis.bar(x, values, color=[colors[key] for key in "CQS"], width=0.7)
        axis.errorbar(
            x, values, yerr=np.vstack([values - low, high - values]),
            fmt="none", ecolor="black", capsize=4, linewidth=1.2,
        )
        axis.set_xticks(x, ["Confound C", "QC set Q", "Smoothing S"])
        axis.set_title(contrast)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Shapley-attributed share of across-cell map SS")
    fig.suptitle(
        "Primary exploratory variance attribution\n"
        "Bars: observed; intervals: paired cohort-stratified Bayesian bootstrap 95%"
    )
    save_figure_once(OUTPUTS / "figures" / "variance_attribution.png", fig)


def plot_heatmaps(
    matrices: dict[str, np.ndarray], filename: str, title: str,
    vmin: float, vmax: float, cmap: str,
) -> None:
    labels = [f"{c}-{Q_SHORT[q]}-{s}" for c, q, s in CELLS]
    fig, axes = plt.subplots(1, 2, figsize=(20, 8.5))
    image = None
    for axis, contrast in zip(axes, CONTRASTS):
        image = axis.imshow(
            matrices[contrast], vmin=vmin, vmax=vmax, cmap=cmap,
            interpolation="nearest", aspect="equal",
        )
        axis.set_xticks(np.arange(27), labels, rotation=90, fontsize=6)
        axis.set_yticks(np.arange(27), labels, fontsize=6)
        axis.set_title(contrast)
    fig.colorbar(image, ax=axes, shrink=0.8, pad=0.02)
    fig.suptitle(title)
    save_figure_once(OUTPUTS / "figures" / filename, fig)


def plot_fdr(
    conclusion: pd.DataFrame, jaccard_matrices: dict[str, np.ndarray]
) -> None:
    labels = [f"{c}-{Q_SHORT[q]}-{s}" for c, q, s in CELLS]
    fig, axes = plt.subplots(2, 2, figsize=(20, 13))
    image = None
    cells = conclusion[conclusion.record_type == "cell_bh_fdr_summary"].copy()
    for column, contrast in enumerate(CONTRASTS):
        lookup = cells[cells.contrast == contrast].set_index(
            ["confound", "qc", "smoothing"]
        )
        counts = np.array([
            lookup.loc[(c, q, s), "fdr_rejected_voxels"] for c, q, s in CELLS
        ], dtype=float)
        axes[0, column].bar(np.arange(27), counts, color="#777777")
        axes[0, column].set_xticks(np.arange(27), labels, rotation=90, fontsize=6)
        axes[0, column].set_ylabel("BH-FDR rejected voxels")
        axes[0, column].set_title(f"{contrast}: q <= 0.05 counts (secondary)")
        image = axes[1, column].imshow(
            jaccard_matrices[contrast], vmin=0, vmax=1, cmap="viridis",
            interpolation="nearest",
        )
        axes[1, column].set_xticks(np.arange(27), labels, rotation=90, fontsize=6)
        axes[1, column].set_yticks(np.arange(27), labels, fontsize=6)
        axes[1, column].set_title(f"{contrast}: rejected-set Jaccard")
    fig.colorbar(image, ax=axes[1, :].tolist(), shrink=0.8, pad=0.02)
    fig.suptitle("Secondary conclusion stability; threshold is not a variance factor")
    save_figure_once(OUTPUTS / "figures" / "fdr_stability.png", fig)


def plot_matched_calibration(
    matched: pd.DataFrame,
    distributions: dict[tuple[str, str, str, str], np.ndarray],
) -> None:
    cell_labels = [f"{c}/{s}" for c, s in BASE_CELLS]
    row_keys = [
        (contrast, transition)
        for contrast in CONTRASTS
        for transition in ("Q0_to_Q1", "Q1_to_Q2")
    ]
    metric_keys = (
        "normalized_rms_difference", "one_minus_pearson_r"
    )
    fig, axes = plt.subplots(4, 2, figsize=(18, 18), sharex=True)
    for row_index, (contrast, transition) in enumerate(row_keys):
        for column_index, metric in enumerate(metric_keys):
            axis = axes[row_index, column_index]
            values = [
                distributions[(contrast, transition, metric, label)]
                for label in cell_labels
            ]
            axis.boxplot(values, showfliers=False, widths=0.65)
            subset = matched[
                (matched.contrast == contrast)
                & (matched.transition == transition)
                & (matched.metric == metric)
            ].set_index(["confound", "smoothing"])
            observed = [
                subset.loc[(c, s), "observed"] for c, s in BASE_CELLS
            ]
            axis.scatter(np.arange(1, 10), observed, color="#CC3311", s=24,
                         zorder=3, label="observed frozen-Q change")
            axis.set_title(f"{contrast}; {transition}; {metric}")
            axis.grid(axis="y", alpha=0.2)
            if row_index == 3:
                axis.set_xticks(np.arange(1, 10), cell_labels, rotation=45,
                                ha="right")
            if row_index == 0 and column_index == 0:
                axis.legend(loc="best", fontsize=8)
    fig.suptitle(
        "Matched-size cohort-stratified random deletion calibration\n"
        "Boxes: 1,000 random deletions; red: observed QC-set change; not causal QC"
    )
    save_figure_once(OUTPUTS / "figures" / "matched_n_calibration.png", fig)


def plot_er_influence(table: pd.DataFrame) -> None:
    labels = [f"{c}/{s}" for c, s in BASE_CELLS]
    excluded_ids = sorted(EXPECTED_EXCLUSIONS_Q2)
    colors = plt.cm.tab10(np.linspace(0, 1, len(excluded_ids)))
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), sharey=True)
    for axis, contrast in zip(axes, CONTRASTS):
        subset = table[table.contrast == contrast]
        retained_values = []
        for c, s in BASE_CELLS:
            retained_values.append(subset[
                (subset.confound == c) & (subset.smoothing == s)
                & (subset.frozen_status == "retained_ER")
            ].normalized_rms_loo_influence.to_numpy(float))
        axis.boxplot(retained_values, showfliers=False, widths=0.65)
        for color, subject in zip(colors, excluded_ids):
            values = []
            for c, s in BASE_CELLS:
                selected = subset[
                    (subset.confound == c) & (subset.smoothing == s)
                    & (subset.participant_id == subject)
                ].normalized_rms_loo_influence
                if len(selected) != 1:
                    raise RuntimeError("Missing excluded-subject influence row")
                values.append(float(selected.iloc[0]))
            axis.plot(np.arange(1, 10), values, marker="o", linewidth=1,
                      markersize=3, color=color, label=subject)
        axis.set_xticks(np.arange(1, 10), labels, rotation=45, ha="right")
        axis.set_title(contrast)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("NRMS leave-one-out influence from full ER map")
    axes[1].legend(title="Frozen excluded ER", fontsize=8)
    fig.suptitle(
        "Exploratory post-primary ER leave-one-out influence\n"
        "Boxes are retained ER subjects; lines are the five frozen exclusions"
    )
    save_figure_once(OUTPUTS / "figures" / "exploratory_er_loo_influence.png", fig)


def format_number(value: float) -> str:
    if not np.isfinite(value):
        return str(value)
    return f"{value:.4g}"


def load_edge_scaling_summary() -> dict:
    path = OUTPUTS / "edge_scaling_diagnostic_summary.json"
    if not path.is_file():
        raise RuntimeError(
            "Missing diagnostic-only edge scaling summary; run "
            "outputs/code/audit_edge_scaling.py before aggregation"
        )
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "narps.edge_scaling_diagnostic.v1",
        "scope": "diagnostic_only_subject_checkpoint_audit",
        "source_bold_accessed": False,
        "primary_maps_modified": False,
        "subject_count": 108,
        "npz_count": 108,
        "json_sidecar_count": 108,
        "sidecars_missing": 0,
        "row_count": 1944,
        "expected_row_count": 1944,
        "all_variances_finite_and_positive": True,
        "variance_nonfinite_total": 0,
        "variance_nonpositive_total": 0,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(
                f"Edge scaling diagnostic summary mismatch for {key}: "
                f"{summary.get(key)!r} != {value!r}"
            )
    diagnostic = summary.get("float64_temporal_mean_lt1_diagnostic")
    if not isinstance(diagnostic, dict):
        raise RuntimeError("Edge scaling diagnostic availability record is missing")
    if (
        int(diagnostic.get("subjects_available_count", -1))
        + int(diagnostic.get("subjects_unavailable_count", -1))
        != 108
    ):
        raise RuntimeError("Edge scaling diagnostic subject availability is incomplete")
    expected_legacy = {
        "sub-001", "sub-002", "sub-003", "sub-008", "sub-009",
        "sub-010", "sub-011", "sub-014", "sub-015", "sub-016",
        "sub-017", "sub-018", "sub-019", "sub-022",
    }
    unavailable = set(diagnostic.get("subjects_unavailable", []))
    if (
        int(diagnostic.get("subjects_unavailable_count", -1)) != 14
        or int(diagnostic.get("subjects_available_count", -1)) != 94
        or unavailable != expected_legacy
        or int(diagnostic.get("rows_unavailable_count", -1)) != 252
        or int(diagnostic.get("rows_available_count", -1)) != 1692
    ):
        raise RuntimeError(
            "Edge scaling diagnostic legacy-sidecar provenance is inconsistent"
        )
    for key in (
        "variance_global_min", "variance_lt_1e_20_total",
        "variance_lt_1e_30_total",
    ):
        if key not in summary or not np.isfinite(float(summary[key])):
            raise RuntimeError(f"Edge scaling diagnostic lacks finite {key}")
    return summary


def result_markdown(
    variance: pd.DataFrame,
    conclusion: pd.DataFrame,
    matched: pd.DataFrame,
    influence: pd.DataFrame,
    n_voxels: int,
    edge_summary: dict,
) -> str:
    lines = [
        "# NARPS analysis-choice attribution result",
        "",
        "## Scope and claim boundary",
        "",
        "This is an **exploratory-only analysis-choice attribution study** in public "
        "NARPS ds001734 data. It is not independent confirmation, does not establish "
        "a biological effect, and does not estimate a causal effect of QC. Independent "
        "fresh confirmation remains required.",
        "",
        "Gain (`gain_demean`) and loss (`loss_demean`) were analyzed separately. "
        "No pooled EI/ER gain-minus-loss contrast is used as a primary result.",
        "",
        "## Completion and estimand",
        "",
        f"All 54 contrast-by-grid cells (27 per contrast) used the frozen common mask "
        f"of {n_voxels:,} voxels and the frozen group intercept estimand: the equally "
        "weighted mean of EI and ER cohort means. All multiverse cells overlap in "
        "subjects and are not treated as independent observations.",
        "",
        "## Primary continuous-map variance attribution",
        "",
        "| Contrast | C share (95% interval) | Q share (95% interval) | S share "
        "(95% interval) | C/S (97.5% simultaneous interval) | Q/S (97.5% "
        "simultaneous interval) | (C+Q)/S (95% interval) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    full_supported: list[str] = []
    for contrast in CONTRASTS:
        factors = variance[
            (variance.contrast == contrast)
            & (variance.record_type == "shapley_factor")
        ].set_index("component")
        ratios = variance[
            (variance.contrast == contrast)
            & (variance.record_type == "primary_ratio")
        ].set_index("component")
        factor_text = []
        for key in "CQS":
            row = factors.loc[key]
            factor_text.append(
                f"{format_number(row['share'])} "
                f"[{format_number(row['ci_lower'])}, {format_number(row['ci_upper'])}]"
            )
        ratio_text = []
        for key in ("C_over_S", "Q_over_S", "C_plus_Q_over_S"):
            row = ratios.loc[key]
            ratio_text.append(
                f"{format_number(row['point_estimate'])} "
                f"[{format_number(row['ci_lower'])}, {format_number(row['ci_upper'])}]"
            )
        lines.append(
            f"| {contrast} | {factor_text[0]} | {factor_text[1]} | {factor_text[2]} "
            f"| {ratio_text[0]} | {ratio_text[1]} | {ratio_text[2]} |"
        )
        if (
            ratios.loc["C_over_S", "point_estimate"] > 1
            and ratios.loc["Q_over_S", "point_estimate"] > 1
            and ratios.loc["C_over_S", "ci_lower"] > 1
            and ratios.loc["Q_over_S", "ci_lower"] > 1
        ):
            full_supported.append(contrast)
    lines.extend(["", "The exact functional-ANOVA term maps partitioned total "
                  "across-cell map sum of squares within numerical tolerance. "
                  "Interactions were allocated equally to their participating "
                  "factors (the frozen Shapley rule)."])
    material = variance[
        (variance.record_type == "anova_term") & variance.material_interaction.astype(bool)
    ]
    lines.extend(["", "Material interactions (observed SS share >= 0.05):"])
    if material.empty:
        lines.append("none.")
    else:
        lines.append(", ".join(
            f"{row.contrast} {row.component}={format_number(row.share)}"
            for row in material.itertuples()
        ) + ".")
    lines.extend(["", "## Frozen decision rule", ""])
    if full_supported:
        joined = ", ".join(full_supported)
        lines.extend([
            f"The frozen numerical full-objective rule is met for: {joined}. This "
            "makes the exploratory direction eligible for the separate terminal "
            "candidate/review workflow; it is not scientific acceptance.",
            "",
            "A specific fresh-confirmation prediction would be that, for the stated "
            "contrast(s), both attributed C/S and Q/S remain above 1 under the same "
            "frozen estimand and controls. The direction is falsified if either ratio "
            "is <=1 or its prespecified simultaneous uncertainty lower bound is <=1.",
        ])
    else:
        lines.append(
            "No contrast met the complete frozen rule requiring both C/S and Q/S "
            "point estimates and both simultaneous interval lower bounds to exceed 1. "
            "The exact partial patterns are the table values above; they must not be "
            "described as showing that both confound and exclusion sensitivity exceed "
            "smoothing."
        )
    lines.extend(["", "## Descriptive continuous and BH-FDR stability", ""])
    pairs = conclusion[
        (conclusion.record_type == "pairwise_cell_stability")
        & (conclusion.n_differing_axes == 1)
    ]
    lines.extend([
        "One-axis pair summaries (median over matched cell pairs):",
        "",
        "| Contrast | Axis changed | Pearson r | cosine | NRMSD | FDR Jaccard |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for contrast in CONTRASTS:
        for axis in ("C", "Q", "S"):
            part = pairs[(pairs.contrast == contrast) & (pairs.differing_axes == axis)]
            lines.append(
                f"| {contrast} | {axis} | {format_number(part.pearson_r.median())} "
                f"| {format_number(part.cosine_similarity.median())} "
                f"| {format_number(part.normalized_rms_difference.median())} "
                f"| {format_number(part.fdr_jaccard.median())} |"
            )
    cells = conclusion[conclusion.record_type == "cell_bh_fdr_summary"]
    lines.extend(["", "BH-FDR at q<=0.05 is secondary only and was not tuned or "
                  "treated as a fourth factor."])
    for contrast in CONTRASTS:
        part = cells[cells.contrast == contrast]
        lines.append(
            f"For {contrast}, rejected counts ranged from "
            f"{int(part.fdr_rejected_voxels.min()):,} to "
            f"{int(part.fdr_rejected_voxels.max()):,}; "
            f"{int(part.fdr_any_rejection.astype(bool).sum())}/27 cells had at least "
            "one rejection."
        )
    lines.extend(["", "## Matched-size deletion calibration", ""])
    lines.append(
        "The frozen-Q changes were calibrated against 1,000 cohort-stratified "
        "random deletions using the same sampled subject sets for every C×S cell and "
        "both contrasts (PCG64 seed 20260814). This is sample-composition calibration, "
        "not a causal estimate of QC."
    )
    for contrast in CONTRASTS:
        for transition in ("Q0_to_Q1", "Q1_to_Q2"):
            for metric in ("normalized_rms_difference", "one_minus_pearson_r"):
                part = matched[
                    (matched.contrast == contrast)
                    & (matched.transition == transition)
                    & (matched.metric == metric)
                ]
                lines.append(
                    f"- {contrast}, {transition}, {metric}: median observed/null-median "
                    f"ratio across C×S = "
                    f"{format_number(part.observed_to_null_median_ratio.median())}; "
                    f"{int((part.upper_tail_empirical_p <= 0.05).sum())}/9 cells had "
                    "upper-tail empirical p<=0.05."
                )
    lines.extend(["", "## Exploratory post-primary diagnostic", ""])
    lines.append(
        "ER leave-one-out influence was computed for all 54 ER subjects and compared "
        "the five frozen exclusions (sub-016, sub-018, sub-030, sub-088, sub-100) "
        "descriptively with retained ER participants. This diagnostic does not alter "
        "the frozen primary analysis."
    )
    for contrast in CONTRASTS:
        part = influence[influence.contrast == contrast]
        excluded = part[part.frozen_status == "frozen_excluded_ER"]
        retained = part[part.frozen_status == "retained_ER"]
        lines.append(
            f"- {contrast}: median influence across subjects and C×S was "
            f"{format_number(excluded.normalized_rms_loo_influence.median())} for "
            f"frozen-excluded ER and "
            f"{format_number(retained.normalized_rms_loo_influence.median())} for "
            "retained ER."
        )
    edge_availability = edge_summary["float64_temporal_mean_lt1_diagnostic"]
    lines.extend([
        "",
        "## Exploratory edge-scaling diagnostic",
        "",
        "A diagnostic-only audit of all 1,944 subject × C × S × contrast "
        "fixed-effect variance arrays found "
        f"{int(edge_summary['variance_lt_1e_20_total']):,} array-voxel entries "
        "below 1e-20 and "
        f"{int(edge_summary['variance_lt_1e_30_total']):,} below 1e-30; the "
        f"global minimum was {format_number(float(edge_summary['variance_global_min']))}. "
        "All stored variances remained finite and strictly positive. Float64 "
        "temporal-mean-below-1 logging diagnostics were available for "
        f"{int(edge_availability['subjects_available_count'])}/108 subjects; "
        f"the {int(edge_availability['subjects_unavailable_count'])} earlier "
        "compatible sidecars predated diagnostic logging and were not recomputed. "
        "The float64 counts are not asserted to equal Nilearn's internal float32 "
        "clamp decisions in every rounding-edge case. These thresholds and counts "
        "are exploratory engineering diagnostics, not a change to, filter "
        "on, or sensitivity redefinition of the frozen primary analysis.",
    ])
    lines.extend([
        "",
        "## Limitations and reproducibility",
        "",
        "All five exclusions are ER participants, so Q sensitivity conflates the "
        "prespecified analysis set with sample composition. EI and ER are task "
        "versions with different gain support; any direct task-version coefficient "
        "would be exploratory. Bayesian-bootstrap intervals quantify paired "
        "subject-sampling sensitivity of map-level metrics and are not voxelwise "
        "inference. The same dataset generated every exploratory result here. The "
        "frozen 95%-coverage mask permits low-intensity edge voxels in individual "
        "runs; exact Nilearn signal_scaling=0 behavior clamps temporal means below "
        "1 before percent scaling. Those run/voxel estimates are therefore not "
        "literally scaled by their own temporal mean. An exact-zero series becomes "
        "constant -100 and can have numerically tiny within-run contrast variance, "
        "which can pull its inverse-variance subject fixed effect toward zero. "
        "Smoothing reduces many such edge cases, so this boundary behavior can "
        "contribute specifically to S attribution. The diagnostic audit is reported "
        "separately; no post-hoc mask trimming or variance floor was applied.",
        "",
        "Reproduce after all immutable subject checkpoints exist with:",
        "",
        "```bash",
        "module load python/3.12.1",
        "/home/users/zijiao/pcvenv/bin/python "
        "outputs/code/audit_edge_scaling.py",
        "/home/users/zijiao/pcvenv/bin/python outputs/code/aggregate_results.py "
        "--gram-voxel-chunk 8192",
        "/home/users/zijiao/pcvenv/bin/python "
        "outputs/code/finalize_candidate.py",
        "```",
        "",
        "Frozen randomness: paired Bayesian bootstrap PCG64 seed 20260815 "
        "(2,000 replicates); matched-size deletion PCG64 seed 20260814 "
        "(1,000 replicates). The Gram-matrix implementation is an exact algebraic "
        "evaluation of the frozen map-space sums, correlations, and distances; it "
        "does not change the estimand. No frozen-analysis deviation was introduced "
        "by this aggregation stage.",
        "",
    ])
    return "\n".join(lines)


def run_self_test() -> int:
    rng = np.random.default_rng(147)
    projections = projection_matrices()
    maps = rng.normal(size=(27, 31))
    observed = functional_anova(maps @ maps.T, projections)
    centered = maps - maps.mean(axis=0, keepdims=True)
    direct = float(np.sum(centered * centered))
    np.testing.assert_allclose(observed["total"], direct, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(sum(observed["attributed"].values()), direct,
                               rtol=2e-13, atol=2e-13)

    n_subjects, n_voxels = 7, 19
    subject_maps = rng.normal(size=(n_subjects, 9, n_voxels))
    flat = subject_maps.transpose(1, 0, 2).reshape(9 * n_subjects, n_voxels)
    subject_gram = (flat @ flat.T).reshape(9, n_subjects, 9, n_subjects)
    coefficients = rng.exponential(size=(5, 3, n_subjects))
    coefficients /= coefficients.sum(axis=2, keepdims=True)
    algebraic = group_grams_from_subject_gram(subject_gram, coefficients)
    direct_maps = np.einsum(
        "rqn,nbv->rbqv", coefficients, subject_maps, optimize=True
    ).reshape(5, 27, n_voxels)
    direct_gram = np.einsum(
        "riv,rjv->rij", direct_maps, direct_maps, optimize=True
    )
    np.testing.assert_allclose(algebraic, direct_gram, rtol=2e-13, atol=2e-13)

    is_ei = np.array([True, True, True, False, False, False, False])
    eligible = np.ones(n_subjects, dtype=bool)
    y = rng.normal(size=(n_subjects, 13))
    intercept, t_stat, df = group_intercept_t(y, eligible, is_ei)
    design = np.column_stack([np.ones(n_subjects), np.where(is_ei, 0.5, -0.5)])
    inverse = np.linalg.inv(design.T @ design)
    beta = inverse @ design.T @ y
    residual = y - design @ beta
    sigma2 = np.sum(residual * residual, axis=0) / (n_subjects - 2)
    direct_t = beta[0] / np.sqrt(sigma2 * inverse[0, 0])
    np.testing.assert_allclose(intercept, beta[0], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(t_stat, direct_t, rtol=2e-13, atol=2e-13)
    assert df == n_subjects - 2

    p = np.array([0.001, 0.01, 0.03, 0.2, 0.9])
    np.testing.assert_array_equal(bh_rejections(p), [True, True, True, False, False])

    sums = direct_maps.sum(axis=2)
    metrics = spatial_metrics_from_gram(direct_gram, sums, n_voxels)
    a = direct_maps[0, 0]
    b = direct_maps[0, 1]
    expected_r = np.corrcoef(a, b)[0, 1]
    np.testing.assert_allclose(metrics["pearson_r"][0, 0], expected_r, atol=2e-13)
    print("aggregate_results.py synthetic self-test: PASS")
    return 0


def run_analysis(args: argparse.Namespace) -> int:
    if args.gram_voxel_chunk < 64:
        raise ValueError("--gram-voxel-chunk must be >=64")
    assert_frozen_contract()
    mask_image, mask = load_mask_vector()
    n_voxels = int(mask.sum())
    mask_hash = sha256(OUTPUTS / "common_analysis_mask.nii.gz")
    qc, is_ei, memberships = load_and_validate_qc()
    observed_weights = observed_group_weights(is_ei, memberships)
    bootstrap_weights = paired_bootstrap_weights(is_ei, memberships)
    deletion_weights, deletion_samples = generate_random_deletions(
        qc, is_ei, memberships
    )
    effects, checkpoint_hashes = load_subject_checkpoints(qc, n_voxels, mask_hash)
    manifest = load_manifest()
    projections = projection_matrices()

    group_maps = np.empty((len(CONTRASTS), 9, 3, n_voxels), dtype=np.float64)
    fdr_masks = np.empty((len(CONTRASTS), 9, 3, n_voxels), dtype=bool)
    cell_rows: list[dict] = []
    for contrast_index, contrast in enumerate(CONTRASTS):
        data = effects[contrast_index]
        for b, (c, s) in enumerate(BASE_CELLS):
            for qi, q in enumerate(QC_SETS):
                paths = map_paths(contrast, c, q, s)
                try:
                    intercept, t_stat, df = group_intercept_t(
                        data[:, b, :], memberships[q], is_ei
                    )
                    # Check the closed-form OLS result against the frozen weight map.
                    weighted = observed_weights[qi] @ data[:, b, :]
                    if not np.allclose(intercept, weighted, rtol=2e-12, atol=2e-12):
                        raise RuntimeError("Group intercept disagrees with cohort weights")
                    p_values = 2.0 * stats.t.sf(np.abs(t_stat), df=df)
                    rejected = bh_rejections(p_values, q=0.05)
                    group_maps[contrast_index, b, qi] = intercept
                    fdr_masks[contrast_index, b, qi] = rejected
                    save_masked_nifti_once(
                        paths[0], intercept.astype(np.float32), mask_image, mask,
                        np.dtype(np.float32),
                    )
                    save_masked_nifti_once(
                        paths[1], t_stat.astype(np.float32), mask_image, mask,
                        np.dtype(np.float32),
                    )
                    save_masked_nifti_once(
                        paths[2], rejected.astype(np.uint8), mask_image, mask,
                        np.dtype(np.uint8),
                    )
                    cell_rows.append(cell_summary_row(
                        contrast, c, q, s, int(memberships[q].sum()), df,
                        intercept, rejected,
                    ))
                    update_manifest_cell(
                        manifest, contrast, c, q, s, "complete", paths=paths
                    )
                    print(
                        f"Completed group cell {contrast} {c}/{q}/{s}: "
                        f"FDR={int(rejected.sum())}", flush=True,
                    )
                except Exception as exc:
                    update_manifest_cell(
                        manifest, contrast, c, q, s, "failed_group_aggregation",
                        paths=None, error=f"{type(exc).__name__}: {exc}",
                    )
                    raise

    if len(cell_rows) != 54 or not np.isfinite(group_maps).all():
        raise RuntimeError("The complete 54-cell group grid was not produced")

    variance_rows: list[dict] = []
    pairwise_rows: list[dict] = []
    matched_rows: list[dict] = []
    influence_rows: list[dict] = []
    pearson_matrices: dict[str, np.ndarray] = {}
    jaccard_matrices: dict[str, np.ndarray] = {}
    all_deletion_distributions: dict[tuple[str, str, str, str], np.ndarray] = {}
    for contrast_index, contrast in enumerate(CONTRASTS):
        print(f"Building exact subject/cell Gram matrix for {contrast}", flush=True)
        subject_gram, subject_sums = subject_cell_gram(
            effects[contrast_index], args.gram_voxel_chunk
        )
        print(f"Evaluating {N_BOOTSTRAP} paired bootstrap replicates", flush=True)
        bootstrap_grams = group_grams_from_subject_gram(
            subject_gram, bootstrap_weights
        )
        bootstrap_sums = np.einsum(
            "rqn,nb->rbq", bootstrap_weights, subject_sums, optimize=True
        ).reshape(N_BOOTSTRAP, 27)
        flat_maps = group_maps[contrast_index].reshape(27, n_voxels)
        observed_gram = flat_maps @ flat_maps.T
        observed_anova = functional_anova(observed_gram, projections)
        boot_anova = bootstrap_anova(bootstrap_grams, projections)
        variance_rows.extend(variance_rows_for_contrast(
            contrast, observed_anova, boot_anova, n_voxels
        ))
        rows, pearson_matrix, jaccard_matrix = pairwise_rows_for_contrast(
            contrast, group_maps[contrast_index], fdr_masks[contrast_index],
            bootstrap_grams, bootstrap_sums, n_voxels,
        )
        pairwise_rows.extend(rows)
        pearson_matrices[contrast] = pearson_matrix
        jaccard_matrices[contrast] = jaccard_matrix
        rows, distributions = matched_calibration(
            contrast, subject_gram, subject_sums, group_maps[contrast_index],
            observed_weights, deletion_weights, n_voxels,
        )
        matched_rows.extend(rows)
        all_deletion_distributions.update(distributions)
        influence_rows.extend(er_leave_one_out_influence(
            contrast, subject_gram, qc, is_ei
        ))
        del subject_gram, bootstrap_grams

    variance_table = pd.DataFrame(variance_rows)
    conclusion_table = pd.DataFrame(cell_rows + pairwise_rows)
    matched_table = pd.DataFrame(matched_rows)
    influence_table = pd.DataFrame(influence_rows)
    save_table_once(OUTPUTS / "variance_attribution.tsv", variance_table)
    save_table_once(OUTPUTS / "conclusion_stability.tsv", conclusion_table)
    save_table_once(OUTPUTS / "matched_n_calibration.tsv", matched_table)
    save_table_once(OUTPUTS / "matched_n_random_sets.tsv", deletion_samples)
    save_table_once(
        OUTPUTS / "exploratory_er_loo_influence.tsv", influence_table
    )

    plot_variance_attribution(variance_table)
    plot_heatmaps(
        pearson_matrices, "continuous_pearson_heatmaps.png",
        "Primary continuous-map spatial Pearson correlations (27-cell grid)",
        -1.0, 1.0, "coolwarm",
    )
    plot_fdr(conclusion_table, jaccard_matrices)
    plot_matched_calibration(matched_table, all_deletion_distributions)
    plot_er_influence(influence_table)

    edge_summary = load_edge_scaling_summary()
    result = result_markdown(
        variance_table, conclusion_table, matched_table, influence_table,
        n_voxels, edge_summary,
    )
    save_text_once(OUTPUTS / "RESULT.md", result)
    repair_path = OUTPUTS / "engineering_repairs.json"
    repair_log = (
        json.loads(repair_path.read_text(encoding="utf-8"))
        if repair_path.is_file() else {}
    )
    repair_failures = repair_log.get("failed_attempts", [])
    repair_deviations = repair_log.get("deviations", [])
    if not isinstance(repair_failures, list) or not all(
        isinstance(item, str) and item.strip() for item in repair_failures
    ):
        raise RuntimeError("engineering_repairs.json failed_attempts is invalid")
    if not isinstance(repair_deviations, list) or not all(
        isinstance(item, str) and item.strip() for item in repair_deviations
    ):
        raise RuntimeError("engineering_repairs.json deviations is invalid")
    provenance = {
        "schema_version": "narps.aggregation_provenance.v1",
        "status": "complete",
        "scientific_status": "exploratory_only",
        "confirmation_eligible": False,
        "scientific_acceptance": False,
        "confirmation_requirement": "independent_fresh_confirmation_required",
        "contract_hashes": {
            name: sha256(OUTPUTS / name) for name in CONTRACT_HASHES
        },
        "common_mask_sha256": mask_hash,
        "common_mask_voxels": n_voxels,
        "subject_checkpoints": 108,
        "subject_checkpoint_sha256": checkpoint_hashes,
        "completed_grid_cells": 54,
        "group_map_triplets": 54,
        "functional_anova_terms": list(projections),
        "bayesian_bootstrap": {
            "replicates": N_BOOTSTRAP,
            "rng": "PCG64",
            "seed": BOOTSTRAP_SEED,
            "paired_across": ["C", "Q", "S", "contrast"],
            "implementation": "exact subject_by_cell_Gram_matrix_algebra",
        },
        "matched_size_random_deletion": {
            "replicates": N_DELETION,
            "rng": "PCG64",
            "seed": DELETION_SEED,
            "draw_order": "replicate_major_Q0_to_Q1_then_Q1_to_Q2",
            "sample_table": "outputs/matched_n_random_sets.tsv",
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "nibabel": nib.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "analysis_code_sha256": sha256(Path(__file__)),
        "fit_code_sha256": sha256(OUTPUTS / "code" / "fit_subject.py"),
        "engineering_repairs": repair_log.get("repairs", []),
        "development_validation": [
            "Python byte-compilation passed after loading python/3.12.1.",
            "Built-in synthetic projection/Shapley/OLS/bootstrap-Gram/metric self-test passed.",
            "Independent outputs/tests/test_math_review.py audit passed without neural outcome access.",
        ],
        "failed_attempts": [
            "A development-only direct pcvenv invocation before loading the cluster "
            "python/3.12.1 module could not locate libpython3.12.so.1.0; rerunning "
            "inside the declared module environment passed. No neural outcome was "
            "accessed by either validation attempt."
        ] + repair_failures,
        "deviations_from_frozen_analysis": repair_deviations,
    }
    save_json_once(OUTPUTS / "aggregation_provenance.json", provenance)
    print("Aggregation completed with all frozen analyses and artifacts", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    try:
        return run_analysis(args)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
