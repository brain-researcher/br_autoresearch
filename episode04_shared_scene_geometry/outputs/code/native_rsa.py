"""Determinate EP04 equations; no data loading, source binding, or audit access.

Callers must supply image-aligned, role-filtered data after the registered
training-fold preprocessing. This module does not select preprocessing, folds,
low-level features, model arrays, ROI masks, or an empirical candidate.
"""
from dataclasses import dataclass
import math

import numpy as np


VLM_ORDER = ("openclip", "pecore", "siglip2")
MANDATORY_SENSITIVITIES = ("low_level_partial_rsa", "near_neighbour_exclusion",
                           "equal_source_strata", "complete_repeats", "neural_cosine")
REQUIRED_DIAGNOSTICS = ("identical_fold_encoding", "evc_reliability_context")


class NotEvaluable(ValueError):
    """Required geometry or score is undefined; never silently omit a row."""


def finite_array(value, ndim=None):
    result = np.asarray(value, dtype=float)
    if (ndim is not None and result.ndim != ndim) or not np.all(np.isfinite(result)):
        raise NotEvaluable("required array has wrong dimensions or nonfinite values")
    return result


def l2_rows(value):
    value = finite_array(value, 2)
    norms = np.linalg.norm(value, axis=1)
    if np.any(norms == 0):
        raise NotEvaluable("zero-length row has no cosine geometry")
    return value / norms[:, None]


def rdm(value, distance="cosine"):
    value = finite_array(value, 2)
    if value.shape[0] < 3:
        raise NotEvaluable("at least three images are needed for RSA")
    if distance == "correlation":
        value = value - value.mean(axis=1, keepdims=True)
    elif distance != "cosine":
        raise ValueError("registered distances are cosine and correlation")
    normalized = l2_rows(value)
    result = 1 - normalized @ normalized.T
    np.fill_diagonal(result, 0)
    return result


def upper_triangle(matrix):
    matrix = finite_array(matrix, 2)
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 3:
        raise NotEvaluable("RDM must be square with at least three images")
    return matrix[np.triu_indices(matrix.shape[0], 1)]


def average_ranks(values):
    values = finite_array(values, 1)
    order = np.argsort(values, kind="stable")
    ranked = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranked[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranked


def pearson(left, right):
    left, right = finite_array(left, 1), finite_array(right, 1)
    if left.shape != right.shape or len(left) < 3:
        raise NotEvaluable("correlation needs aligned vectors with at least three entries")
    left, right = left - left.mean(), right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        raise NotEvaluable("constant geometry has undefined correlation")
    return float(np.clip(left @ right / denominator, -1, 1))


def spearman_rsa(model_rdm, neural_rdm):
    return pearson(average_ranks(upper_triangle(model_rdm)),
                   average_ranks(upper_triangle(neural_rdm)))


def partial_spearman_rsa(model_rdm, neural_rdm, low_level_rdm):
    model, neural, low = [average_ranks(upper_triangle(item))
                          for item in (model_rdm, neural_rdm, low_level_rdm)]
    if model.shape != neural.shape or model.shape != low.shape:
        raise NotEvaluable("partial RSA requires identical image support")
    design = np.column_stack([np.ones(len(low)), low])
    model_residual = model - design @ np.linalg.lstsq(design, model, rcond=None)[0]
    neural_residual = neural - design @ np.linalg.lstsq(design, neural, rcond=None)[0]
    # Algebraically zero residuals are undefined, not an epsilon-rescued score.
    tolerance = np.finfo(float).eps * max(design.shape) * max(np.linalg.norm(model), np.linalg.norm(neural))
    if min(np.linalg.norm(model_residual), np.linalg.norm(neural_residual)) <= tolerance:
        raise NotEvaluable("low-level ranks fully explain a required rank vector")
    return pearson(model_residual, neural_residual)


def fisher_z(correlation):
    if not math.isfinite(correlation) or abs(correlation) >= 1:
        raise NotEvaluable("Fisher z is not finite; no arbitrary correlation clipping")
    return math.atanh(correlation)


def whole_image_permutation(strata, seed):
    """One index map, reused for every model and ROI in this image fold."""
    strata = np.asarray(strata, dtype=str)
    rng = np.random.Generator(np.random.PCG64(seed))
    permutation = np.arange(len(strata))
    for stratum in sorted(set(strata.tolist())):
        indices = np.flatnonzero(strata == stratum)
        permutation[indices] = rng.permutation(indices)
    return permutation


def relabel_rdm(matrix, permutation):
    matrix = finite_array(matrix, 2)
    permutation = np.asarray(permutation, dtype=int)
    if matrix.shape != (len(permutation), len(permutation)) or sorted(permutation.tolist()) != list(range(len(permutation))):
        raise ValueError("whole-image relabeling requires one bijective row-and-column index")
    return matrix[np.ix_(permutation, permutation)]


def covariance_matched_geometry(raw_embeddings, model_index, draw_index):
    """Frozen centered SVD/Haar analogue before the primary L2 normalization.

    Centering bounds rank by n-1. When dimensions>=n, the mathematically zero
    last singular direction is omitted so the projected QR stays feasible.
    No fitted rank choice is made.
    """
    value = finite_array(raw_embeddings, 2)
    if value.shape[0] < 3 or model_index not in range(3) or draw_index not in range(1000):
        raise ValueError("invalid registered random-geometry indices or shape")
    mean = value.mean(axis=0)
    centered = value - mean
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    count = min(value.shape[0] - 1, value.shape[1])
    rng = np.random.Generator(np.random.PCG64(100000 + 1000 * model_index + draw_index))
    gaussian = rng.standard_normal((len(value), count))
    gaussian -= gaussian.mean(axis=0)
    orthogonal, triangular = np.linalg.qr(gaussian, mode="reduced")
    signs = np.where(np.diag(triangular) < 0, -1, 1)
    orthogonal *= signs
    raw = mean + (orthogonal * singular[:count]) @ right[:count]
    return raw, l2_rows(raw)


def repeat_average(standardized_presentations, image_ids, valid):
    """Average all explicitly valid presentations, retaining >=2 per image."""
    presentations = np.asarray(standardized_presentations, dtype=float)
    image_ids = np.asarray(image_ids, dtype=str)
    valid = np.asarray(valid, dtype=bool)
    if presentations.ndim != 2 or len(presentations) != len(image_ids) or valid.shape != image_ids.shape:
        raise ValueError("presentation arrays are not aligned")
    finite_array(presentations[valid], 2)
    retained, averages, counts = [], [], []
    for image_id in sorted(set(image_ids.tolist())):
        selected = (image_ids == image_id) & valid
        count = int(selected.sum())
        if count >= 2:
            retained.append(image_id)
            averages.append(presentations[selected].mean(axis=0))
            counts.append(count)
    return retained, np.asarray(averages), np.asarray(counts, dtype=int)


@dataclass(frozen=True)
class ModelIdentity:
    released_id: str
    dimension: int


def frozen_argmax(scores, identities):
    scores = finite_array(scores, 1)
    if len(scores) != 3 or len(identities) != 3:
        raise NotEvaluable("all three fixed VLM rows are required")
    tied = np.flatnonzero(scores.max() - scores <= 1e-6)
    return int(min(tied, key=lambda index: (identities[index].dimension, identities[index].released_id)))


def panel_effects(vlm_z, dino_z):
    """Inputs are [model,participant,fold,sector] and [participant,fold,sector]."""
    vlm_z, dino_z = finite_array(vlm_z, 4), finite_array(dino_z, 3)
    if vlm_z.shape[0] != 3 or dino_z.shape[0] != 5 or dino_z.shape[2] != 3 or dino_z.shape[1] < 1 or vlm_z.shape[1:] != dino_z.shape:
        raise NotEvaluable("require complete three-model, five-participant, three-sector panel")
    gain = vlm_z - dino_z[None]
    participant = gain.mean(axis=(2, 3))
    sector = gain.mean(axis=(1, 2))
    return {"gain": participant.mean(axis=1), "participant": participant,
            "participant_loo": (participant.sum(axis=1, keepdims=True) - participant) / 4,
            "sector_loo": (sector.sum(axis=1, keepdims=True) - sector) / 2,
            "unsubtracted_alignment": vlm_z.mean(axis=(1, 2, 3))}


def familywise_null_statistic(vlm_z, dino_z, identities):
    """Unconditional gain argmax; there is intentionally no eligibility input."""
    effects = panel_effects(vlm_z, dino_z)
    selected = frozen_argmax(effects["gain"], identities)
    return selected, float(effects["unsubtracted_alignment"][selected])


def screen_complete_panel(vlm_z, dino_z, identities, image_null_threshold,
                          geometry_null_threshold, sensitivities, diagnostics):
    """Apply the frozen primary gates after caller supplies completed controls.

    This returns a mathematical decision only. Source coverage, full rerun and
    write-once lock are external stages; this function cannot authorize them.
    Threshold quantile convention remains a separately bound pre-score choice.
    """
    effects = panel_effects(vlm_z, dino_z)
    for threshold in (image_null_threshold, geometry_null_threshold):
        if not math.isfinite(threshold):
            raise NotEvaluable("both complete family-wise null thresholds are required")
    for name in MANDATORY_SENSITIVITIES:
        if name not in sensitivities or len(sensitivities[name]) != 3 or any(value not in ("pass", "fail") for value in sensitivities[name]):
            raise NotEvaluable(f"required sensitivity incomplete:{name}")
    if any(diagnostics.get(name) is not True for name in REQUIRED_DIAGNOSTICS):
        raise NotEvaluable("required encoding/EVC diagnostics incomplete")
    gates = []
    for row in range(3):
        row_gates = {
            "practical_gain": bool(effects["gain"][row] >= .02),
            "participant_direction": bool(np.count_nonzero(effects["participant"][row] > 0) >= 4),
            "participant_influence": bool(np.all(effects["participant_loo"][row] > 0)),
            "sector_influence": bool(np.all(effects["sector_loo"][row] > 0)),
            "positive_alignment": bool(effects["unsubtracted_alignment"][row] > 0),
            "image_null": bool(effects["unsubtracted_alignment"][row] > image_null_threshold),
            "geometry_null": bool(effects["unsubtracted_alignment"][row] > geometry_null_threshold),
        }
        row_gates.update({name: sensitivities[name][row] == "pass" for name in MANDATORY_SENSITIVITIES})
        gates.append(row_gates)
    eligible = [row for row in range(3) if all(gates[row].values())]
    selected = None
    if eligible:
        maximum = max(effects["gain"][row] for row in eligible)
        tied = [row for row in eligible if maximum - effects["gain"][row] <= 1e-6]
        selected = min(tied, key=lambda row: (identities[row].dimension, identities[row].released_id))
    return {"selected_index": selected, "gates": gates,
            "effects": {key: value.tolist() for key, value in effects.items()},
            "configuration_lock_authorized": False}
