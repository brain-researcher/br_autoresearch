#!/usr/bin/env python3
"""Outcome-blind synthetic recovery screen for EP11.

The script reads only the compact soma/source metadata inventory.  Every
projection outcome used here is simulated.  It compares a flexible continuous
density route with a finite-group route under whole-brain development/audit
splits, and never emits brain identities or a role assignment.

This is a five-target Stage-0 surrogate screen, not the frozen 56-coordinate
mixed-support likelihood or the production B/P/C/D/H evaluator. Its purpose is
to falsify an under-powered or anti-continuum design before any real regional
projection outcome is constructed. A pass cannot activate outcome-bearing
search.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

# Avoid nested BLAS pools when Slurm gives this program several worker CPUs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from scipy.stats import beta, t as student_t
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold


NO_GROUP_SCENARIOS = (
    "smooth_spatial_continuum",
    "curved_continuum",
    "nonuniform_continuum",
    "depleted_middle_continuum",
    "animal_confounded_continuum",
)
GROUP_SCENARIOS = (
    "reusable_groups",
    "groups_plus_continuum",
    "spatially_interdigitated_groups",
)
ALL_SCENARIOS = NO_GROUP_SCENARIOS + GROUP_SCENARIOS
EFFECT_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)
CONTINUOUS_BANDWIDTHS = (0.45, 0.60, 0.80, 1.05, 1.35, 1.70)
GROUP_CONFIGS = (
    (2, "diag"),
    (2, "full"),
    (3, "diag"),
    (3, "full"),
    (4, "diag"),
    (4, "full"),
)

OUTCOME_DIM = 4  # four ALR coordinates -> five synthetic target proportions
GROUP_BASE_SEPARATION = 0.55
DEFAULT_MARGIN = 0.05  # nats/neuron on the common synthetic composition
# Keep the completed 12/8 simulation stream stable after its display label was
# clarified to include the 16 units that never receive simulated outcomes.
RNG_DESIGN_KEY = "12_dev_8_audit"

_META: "MetadataLayout | None" = None


@dataclass(frozen=True)
class MetadataLayout:
    frame: pd.DataFrame
    brains: np.ndarray
    lines: np.ndarray
    b_design: np.ndarray
    p_design: np.ndarray
    coords: np.ndarray
    pc: np.ndarray
    layer_depth: np.ndarray


@dataclass
class DensityModels:
    beta_b: np.ndarray
    beta_p: np.ndarray
    gaussian_b: tuple[np.ndarray, np.ndarray]
    gaussian_p: tuple[np.ndarray, np.ndarray]
    c_centers: np.ndarray
    c_cov: np.ndarray
    gmm_d: GaussianMixture
    gmm_h: GaussianMixture
    h_config: tuple[int, str]
    c_bandwidth: float


def stable_u64(*parts: object) -> int:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def layer_depth_from_label(label: str) -> float:
    suffix = str(label).replace("SSp-tr", "")
    return {
        "1": 0.08,
        "2/3": 0.27,
        "4": 0.46,
        "5": 0.67,
        "6a": 0.88,
        "6b": 0.97,
    }.get(suffix, 0.50)


def one_hot(values: Sequence[str]) -> np.ndarray:
    values = np.asarray(values, dtype=str)
    levels = sorted(set(values.tolist()))
    # Keep every outcome-blind level. Ridge regularization handles collinearity.
    return np.column_stack([(values == level).astype(float) for level in levels])


def standardize_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mean = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd < 1e-8] = 1.0
    return (np.nan_to_num(x, nan=mean) - mean) / sd


def load_metadata(path: Path) -> MetadataLayout:
    df = pd.read_csv(path, dtype={"fMOST_brain_id": str})
    df = df.loc[
        df["primary_axon_dendrite_distinguished_eligible"].eq("yes")
    ].copy()
    if len(df) != 206 or df["fMOST_brain_id"].nunique() != 36:
        raise ValueError("Expected 206 primary cells in 36 brain/sample units")

    coords = standardize_columns(
        df[["soma_x_ccfv3_um", "soma_y_ccfv3_um", "soma_z_ccfv3_um"]].to_numpy()
    )
    _, _, vt = np.linalg.svd(coords, full_matrices=False)
    pc = coords @ vt.T
    layer_depth = np.array([layer_depth_from_label(x) for x in df["source_acronym"]])
    age = standardize_columns(df[["sample_age_days"]].to_numpy())
    gender = standardize_columns(df[["sample_gender_code"]].to_numpy())

    b_design = np.column_stack(
        [
            np.ones(len(df)),
            age,
            gender,
            one_hot(df["strain_or_cre_line"].astype(str)),
            one_hot(df["hemisphere"].astype(str)),
            one_hot(df["sample_lab_name"].astype(str)),
        ]
    )
    x, y, z = coords.T
    position_basis = np.column_stack(
        [
            coords,
            coords**2,
            coords**3,
            x * y,
            x * z,
            y * z,
            x * y * z,
            layer_depth,
            layer_depth**2,
            one_hot(df["source_acronym"].astype(str)),
        ]
    )
    p_design = np.column_stack([b_design, position_basis])
    return MetadataLayout(
        frame=df.reset_index(drop=True),
        brains=df["fMOST_brain_id"].astype(str).to_numpy(),
        lines=df["strain_or_cre_line"].astype(str).to_numpy(),
        b_design=b_design,
        p_design=p_design,
        coords=coords,
        pc=pc,
        layer_depth=layer_depth,
    )


def role_split(layout: MetadataLayout, n_dev: int, n_audit: int, rng: np.random.Generator):
    brain_rows = layout.frame.groupby("fMOST_brain_id", sort=True).first().reset_index()
    singleton_line = "Tle4-2A-CreER"
    eligible = brain_rows.loc[brain_rows["strain_or_cre_line"].ne(singleton_line)].copy()
    line_to_brains = {
        line: group["fMOST_brain_id"].astype(str).tolist()
        for line, group in eligible.groupby("strain_or_cre_line", sort=True)
    }
    if len(line_to_brains) != 8 or n_audit < len(line_to_brains):
        raise ValueError("The current role sampler expects eight non-singleton lines")

    # One audit and one development unit per non-singleton line. Extra units are
    # sampled without using synthetic or real outcomes. Identities are not saved.
    audit: list[str] = []
    dev: list[str] = []
    for line in sorted(line_to_brains):
        choices = np.asarray(line_to_brains[line], dtype=object)
        picks = rng.choice(choices, size=2, replace=False)
        audit.append(str(picks[0]))
        dev.append(str(picks[1]))

    remaining = [
        x
        for x in eligible["fMOST_brain_id"].astype(str).tolist()
        if x not in set(audit) and x not in set(dev)
    ]
    rng.shuffle(remaining)
    if n_audit > len(audit):
        audit.extend(remaining[: n_audit - len(audit)])
        remaining = remaining[n_audit - len(audit) :]
    dev.extend(remaining[: n_dev - len(dev)])
    if len(dev) != n_dev or len(audit) != n_audit or set(dev) & set(audit):
        raise RuntimeError("Could not construct a disjoint line-balanced role split")
    dev_idx = np.flatnonzero(np.isin(layout.brains, np.asarray(dev)))
    audit_idx = np.flatnonzero(np.isin(layout.brains, np.asarray(audit)))
    return dev_idx, audit_idx


def fixed_line_effect(lines: np.ndarray) -> np.ndarray:
    out = np.empty((len(lines), OUTCOME_DIM), dtype=float)
    for i, line in enumerate(lines):
        seed = stable_u64("line", line) % (2**32)
        rng = np.random.default_rng(seed)
        out[i] = rng.normal(0.0, 0.07, size=OUTCOME_DIM)
    return out


def softmax_composition(z: np.ndarray) -> np.ndarray:
    logits = np.column_stack([z, np.zeros(len(z))])
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    y = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return np.clip(y, 1e-12, 1.0)


def alr(y: np.ndarray) -> np.ndarray:
    return np.log(y[:, :-1]) - np.log(y[:, [-1]])


def log_alr_jacobian(y: np.ndarray) -> np.ndarray:
    return -np.log(y).sum(axis=1)


def generate_outcomes(
    layout: MetadataLayout,
    indices: np.ndarray,
    scenario: str,
    effect_multiplier: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    pc = layout.pc[indices]
    coords = layout.coords[indices]
    depth = layout.layer_depth[indices]
    brains = layout.brains[indices]
    lines = layout.lines[indices]
    n = len(indices)

    u = 0.72 * pc[:, 0] + 0.28 * pc[:, 1]
    v = pc[:, 1]
    smooth = np.column_stack(
        [
            0.30 * np.tanh(u),
            0.24 * np.sin(0.85 * u) + 0.10 * v,
            0.24 * (depth - 0.5) + 0.08 * coords[:, 2] ** 2,
            0.18 * np.cos(0.65 * u) - 0.10 * coords[:, 0] * coords[:, 1],
        ]
    )
    z = np.array([-0.25, 0.05, 0.20, -0.10]) + smooth + fixed_line_effect(lines)
    latent_raw = np.tanh(0.70 * u + rng.normal(0.0, 0.45, size=n))

    if scenario == "smooth_spatial_continuum":
        t = latent_raw
        z += 0.24 * np.column_stack([t, -0.5 * t, 0.4 * t, -0.2 * t])
    elif scenario == "curved_continuum":
        t = latent_raw
        z += 0.48 * np.column_stack(
            [t, 1.25 * (t**2 - 0.33), np.sin(np.pi * t), np.cos(np.pi * t)]
        )
    elif scenario == "nonuniform_continuum":
        q = np.clip(0.5 * (latent_raw + 1.0), 1e-4, 1 - 1e-4)
        t = 2.0 * q**2.4 - 1.0
        z += 0.48 * np.column_stack(
            [t, t**2 - np.mean(t**2), np.sin(np.pi * t), -0.5 * t]
        )
    elif scenario == "depleted_middle_continuum":
        t = np.sign(latent_raw) * (0.42 + 0.58 * np.abs(latent_raw))
        z += 0.48 * np.column_stack(
            [t, t**2 - np.mean(t**2), np.sin(np.pi * t), -0.6 * t]
        )
    elif scenario == "animal_confounded_continuum":
        # Continuous animal shifts plus animal-specific position coverage can
        # look clustered, but there is no reusable cell group.
        for brain in np.unique(brains):
            mask = brains == brain
            shift = rng.multivariate_normal(
                np.zeros(OUTCOME_DIM),
                0.12 * np.eye(OUTCOME_DIM) + 0.07 * np.ones((OUTCOME_DIM, OUTCOME_DIM)),
            )
            z[mask] += shift
        t = np.tanh(latent_raw + 0.35 * np.array([np.mean(u[brains == b]) for b in brains]))
        z += 0.30 * np.column_stack([t, -t, 0.5 * t, -0.25 * t])
    elif scenario in GROUP_SCENARIOS:
        if scenario == "spatially_interdigitated_groups":
            probability = np.full(n, 0.5)
        else:
            probability = expit(0.38 * u - 0.22 * (depth - 0.5))
        g01 = rng.binomial(1, probability, size=n)
        sign = 2.0 * g01 - 1.0
        signature = np.array([1.0, -0.85, 0.70, -0.45])
        signature /= np.linalg.norm(signature)
        z += (
            GROUP_BASE_SEPARATION
            * effect_multiplier
            * sign[:, None]
            * signature[None, :]
        )
        if scenario == "groups_plus_continuum":
            t = latent_raw
            z += 0.42 * np.column_stack(
                [t, t**2 - 0.33, np.sin(np.pi * t), -0.45 * t]
            )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Shared random effect is sampled once per brain and integrated implicitly
    # by every predictive density. Audit outcomes never condition on other cells.
    for brain in np.unique(brains):
        mask = brains == brain
        animal_shift = rng.multivariate_normal(
            np.zeros(OUTCOME_DIM),
            0.018 * np.eye(OUTCOME_DIM) + 0.010 * np.ones((OUTCOME_DIM, OUTCOME_DIM)),
        )
        z[mask] += animal_shift

    noise_scale = 0.23 * (0.75 + 0.45 * np.abs(np.tanh(u)))
    if scenario == "smooth_spatial_continuum":
        noise = rng.standard_t(df=5, size=(n, OUTCOME_DIM)) / math.sqrt(5 / 3)
    else:
        noise = rng.normal(size=(n, OUTCOME_DIM))
    z += noise_scale[:, None] * noise
    truth = np.full(n, -1, dtype=int)
    if scenario in GROUP_SCENARIOS:
        truth = g01.astype(int)
    return softmax_composition(z), truth


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 3.0) -> np.ndarray:
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 1e-8
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def fit_gaussian(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = residual.mean(axis=0)
    covariance = LedoitWolf().fit(residual).covariance_ + 1e-5 * np.eye(residual.shape[1])
    return mean, covariance


def logpdf_gaussian(x: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    delta = x - mean
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise FloatingPointError("Non-positive covariance")
    solved = np.linalg.solve(covariance, delta.T).T
    return -0.5 * (
        x.shape[1] * math.log(2 * math.pi)
        + logdet
        + np.einsum("ij,ij->i", delta, solved)
    )


def fit_kde(residual: np.ndarray, bandwidth: float) -> tuple[np.ndarray, np.ndarray]:
    base_cov = LedoitWolf().fit(residual).covariance_
    covariance = (bandwidth**2) * base_cov + 1e-5 * np.eye(residual.shape[1])
    return residual.copy(), covariance


def logpdf_kde(x: np.ndarray, centers: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise FloatingPointError("Non-positive KDE covariance")
    inv = np.linalg.inv(covariance)
    # Audit sets are small, so one dense (n_test, n_centers, dim) block is safe.
    delta = x[:, None, :] - centers[None, :, :]
    quad = np.einsum("nkd,de,nke->nk", delta, inv, delta)
    component = -0.5 * (x.shape[1] * math.log(2 * math.pi) + logdet + quad)
    return logsumexp(component, axis=1) - math.log(len(centers))


def equal_animal_mean(values: np.ndarray, brains: np.ndarray) -> float:
    return float(np.mean([values[brains == b].mean() for b in np.unique(brains)]))


def choose_candidates(
    layout: MetadataLayout,
    dev_idx: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[float, tuple[int, str]]:
    z = alr(y)
    groups = layout.brains[dev_idx]
    n_splits = min(3, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    c_by_brain = {bw: {} for bw in CONTINUOUS_BANDWIDTHS}
    h_by_brain = {cfg: {} for cfg in GROUP_CONFIGS}

    for fold, (train_local, test_local) in enumerate(
        splitter.split(layout.p_design[dev_idx], z, groups)
    ):
        train_idx = dev_idx[train_local]
        test_idx = dev_idx[test_local]
        beta_p = ridge_fit(layout.p_design[train_idx], z[train_local])
        residual_train = z[train_local] - layout.p_design[train_idx] @ beta_p
        residual_test = z[test_local] - layout.p_design[test_idx] @ beta_p
        test_brains = layout.brains[test_idx]
        jac = log_alr_jacobian(y[test_local])

        for bw in CONTINUOUS_BANDWIDTHS:
            centers, cov = fit_kde(residual_train, bw)
            scores = logpdf_kde(residual_test, centers, cov) + jac
            for brain in np.unique(test_brains):
                c_by_brain[bw][brain] = float(scores[test_brains == brain].mean())

        for config_index, cfg in enumerate(GROUP_CONFIGS):
            k, covariance_type = cfg
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=covariance_type,
                reg_covar=0.008,
                n_init=1,
                max_iter=250,
                random_state=(seed + 97 * fold + 17 * config_index) % (2**32 - 1),
            ).fit(residual_train)
            scores = gmm.score_samples(residual_test) + jac
            for brain in np.unique(test_brains):
                h_by_brain[cfg][brain] = float(scores[test_brains == brain].mean())

    c_scores = {bw: float(np.mean(list(v.values()))) for bw, v in c_by_brain.items()}
    h_scores = {cfg: float(np.mean(list(v.values()))) for cfg, v in h_by_brain.items()}
    best_c = sorted(CONTINUOUS_BANDWIDTHS, key=lambda bw: (-c_scores[bw], bw))[0]
    best_h = sorted(GROUP_CONFIGS, key=lambda cfg: (-h_scores[cfg], cfg[0], cfg[1]))[0]
    return best_c, best_h


def fit_models(
    layout: MetadataLayout,
    dev_idx: np.ndarray,
    y_dev: np.ndarray,
    seed: int,
) -> DensityModels:
    z = alr(y_dev)
    best_c, best_h = choose_candidates(layout, dev_idx, y_dev, seed)
    beta_b = ridge_fit(layout.b_design[dev_idx], z)
    beta_p = ridge_fit(layout.p_design[dev_idx], z)
    residual_b = z - layout.b_design[dev_idx] @ beta_b
    residual_p = z - layout.p_design[dev_idx] @ beta_p
    gaussian_b = fit_gaussian(residual_b)
    gaussian_p = fit_gaussian(residual_p)
    c_centers, c_cov = fit_kde(residual_p, best_c)
    k, covariance_type = best_h
    gmm_h = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        reg_covar=0.008,
        n_init=3,
        max_iter=400,
        random_state=seed % (2**32 - 1),
    ).fit(residual_p)
    gmm_d = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        reg_covar=0.008,
        n_init=3,
        max_iter=400,
        random_state=(seed + 1) % (2**32 - 1),
    ).fit(residual_b)
    return DensityModels(
        beta_b=beta_b,
        beta_p=beta_p,
        gaussian_b=gaussian_b,
        gaussian_p=gaussian_p,
        c_centers=c_centers,
        c_cov=c_cov,
        gmm_d=gmm_d,
        gmm_h=gmm_h,
        h_config=best_h,
        c_bandwidth=best_c,
    )


def score_models(
    layout: MetadataLayout,
    models: DensityModels,
    idx: np.ndarray,
    y: np.ndarray,
) -> dict[str, np.ndarray]:
    z = alr(y)
    jac = log_alr_jacobian(y)
    residual_b = z - layout.b_design[idx] @ models.beta_b
    residual_p = z - layout.p_design[idx] @ models.beta_p
    return {
        "B": logpdf_gaussian(residual_b, *models.gaussian_b) + jac,
        "P": logpdf_gaussian(residual_p, *models.gaussian_p) + jac,
        "C": logpdf_kde(residual_p, models.c_centers, models.c_cov) + jac,
        "D": models.gmm_d.score_samples(residual_b) + jac,
        "H": models.gmm_h.score_samples(residual_p) + jac,
    }


def animal_contrasts(scores: dict[str, np.ndarray], brains: np.ndarray) -> dict[str, np.ndarray]:
    keys = {"P_minus_B": ("P", "B"), "C_minus_P": ("C", "P"),
            "H_minus_C": ("H", "C"), "H_minus_D": ("H", "D")}
    result: dict[str, np.ndarray] = {}
    for name, (left, right) in keys.items():
        delta = scores[left] - scores[right]
        result[name] = np.array([delta[brains == b].mean() for b in np.unique(brains)])
    return result


def interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, float("inf"), float("inf")
    se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    one_sided_half = float(student_t.ppf(confidence, len(values) - 1) * se)
    two_sided_half = float(student_t.ppf(0.5 + confidence / 2, len(values) - 1) * se)
    return mean, one_sided_half, two_sided_half


def component_support(resp: np.ndarray, brains: np.ndarray, stage: str) -> bool:
    min_cells = 20.0 if stage == "development" else 10.0
    min_animal_ess = 4.0 if stage == "development" else 3.0
    min_animals = 4 if stage == "development" else 3
    max_fraction = 0.35 if stage == "development" else 0.50
    min_prevalence = 0.15
    unique_brains = np.unique(brains)
    for k in range(resp.shape[1]):
        weights = resp[:, k]
        prevalence = float(weights.mean())
        cell_ess = float(weights.sum() ** 2 / max(np.square(weights).sum(), 1e-12))
        by_animal_mean = np.array([weights[brains == b].mean() for b in unique_brains])
        animal_ess = float(
            by_animal_mean.sum() ** 2 / max(np.square(by_animal_mean).sum(), 1e-12)
        )
        effective_animals = int(np.sum(by_animal_mean >= 0.20))
        by_animal_mass = np.array([weights[brains == b].sum() for b in unique_brains])
        max_share = float(by_animal_mass.max() / max(by_animal_mass.sum(), 1e-12))
        if not (
            prevalence >= min_prevalence
            and cell_ess >= min_cells
            and animal_ess >= min_animal_ess
            and effective_animals >= min_animals
            and max_share <= max_fraction
        ):
            return False
    return True


def fit_and_evaluate(task: tuple) -> dict[str, object]:
    if _META is None:
        raise RuntimeError("Worker metadata not initialized")
    (
        design,
        n_dev,
        n_audit,
        scenario,
        effect_multiplier,
        replicate,
        seed,
        margin,
        oracle_draws,
    ) = task
    rng = np.random.default_rng(seed)
    dev_idx, audit_idx = role_split(_META, n_dev, n_audit, rng)
    all_idx = np.concatenate([dev_idx, audit_idx])
    y_all, _ = generate_outcomes(_META, all_idx, scenario, effect_multiplier, rng)
    n_dev_cells = len(dev_idx)
    y_dev = y_all[:n_dev_cells]
    y_audit = y_all[n_dev_cells:]

    try:
        models = fit_models(_META, dev_idx, y_dev, seed)
        scores = score_models(_META, models, audit_idx, y_audit)
        contrasts = animal_contrasts(scores, _META.brains[audit_idx])
        hc_mean, hc_one_half, hc_two_half = interval(contrasts["H_minus_C"])
        hd_mean, _, _ = interval(contrasts["H_minus_D"])
        pb_mean = float(contrasts["P_minus_B"].mean())
        cp_mean = float(contrasts["C_minus_P"].mean())

        z_dev = alr(y_dev)
        z_audit = alr(y_audit)
        dev_resid = z_dev - _META.p_design[dev_idx] @ models.beta_p
        audit_resid = z_audit - _META.p_design[audit_idx] @ models.beta_p
        dev_support = component_support(
            models.gmm_h.predict_proba(dev_resid), _META.brains[dev_idx], "development"
        )
        audit_support = component_support(
            models.gmm_h.predict_proba(audit_resid), _META.brains[audit_idx], "audit"
        )
        reusable_group_route = bool(
            hc_mean - hc_one_half > margin and dev_support and audit_support
        )
        continuous_adequate = bool(hc_mean + hc_one_half < margin)

        oracle_delta: list[float] = []
        for draw in range(oracle_draws):
            draw_rng = np.random.default_rng(stable_u64(seed, "oracle", draw))
            y_draw, _ = generate_outcomes(
                _META, audit_idx, scenario, effect_multiplier, draw_rng
            )
            draw_scores = score_models(_META, models, audit_idx, y_draw)
            draw_hc = animal_contrasts(draw_scores, _META.brains[audit_idx])["H_minus_C"]
            oracle_delta.append(float(draw_hc.mean()))
        oracle = float(np.mean(oracle_delta))
        coverage = bool(hc_mean - hc_two_half <= oracle <= hc_mean + hc_two_half)
        return {
            "design": design,
            "scenario": scenario,
            "scenario_class": "no_group" if scenario in NO_GROUP_SCENARIOS else "group",
            "effect_multiplier": effect_multiplier,
            "replicate": replicate,
            "valid": 1,
            "n_dev_cells": len(dev_idx),
            "n_audit_cells": len(audit_idx),
            "selected_c_bandwidth": models.c_bandwidth,
            "selected_h_k": models.h_config[0],
            "selected_h_covariance": models.h_config[1],
            "P_minus_B": pb_mean,
            "C_minus_P": cp_mean,
            "H_minus_C": hc_mean,
            "H_minus_D": hd_mean,
            "one_sided_half_width": hc_one_half,
            "two_sided_coverage": int(coverage),
            "oracle_H_minus_C": oracle,
            "bias": hc_mean - oracle,
            "development_group_support": int(dev_support),
            "audit_group_support": int(audit_support),
            "reusable_group_route": int(reusable_group_route),
            "continuous_adequate": int(continuous_adequate),
            "error": "",
        }
    except Exception as exc:  # preserve failures for the validity denominator
        return {
            "design": design,
            "scenario": scenario,
            "scenario_class": "no_group" if scenario in NO_GROUP_SCENARIOS else "group",
            "effect_multiplier": effect_multiplier,
            "replicate": replicate,
            "valid": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def clopper_pearson_upper(successes: int, trials: int, confidence: float = 0.95) -> float:
    if trials == 0:
        return float("nan")
    if successes == trials:
        return 1.0
    return float(beta.ppf(confidence, successes + 1, trials - successes))


def summarize(replicates: pd.DataFrame, margin: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["design", "scenario", "scenario_class", "effect_multiplier"]
    for keys, block_all in replicates.groupby(group_cols, sort=True):
        block = block_all.loc[block_all["valid"].eq(1)]
        n = len(block)
        route_count = int(block["reusable_group_route"].sum()) if n else 0
        row = {
            "record_type": "condition",
            "design": keys[0],
            "scenario": keys[1],
            "scenario_class": keys[2],
            "effect_multiplier": keys[3],
            "attempted_replicates": len(block_all),
            "valid_replicates": n,
            "valid_fraction": n / len(block_all),
            "mean_development_cells": block["n_dev_cells"].mean() if n else np.nan,
            "mean_audit_cells": block["n_audit_cells"].mean() if n else np.nan,
            "mean_H_minus_C": block["H_minus_C"].mean() if n else np.nan,
            "mean_H_minus_D": block["H_minus_D"].mean() if n else np.nan,
            "mean_P_minus_B": block["P_minus_B"].mean() if n else np.nan,
            "mean_C_minus_P": block["C_minus_P"].mean() if n else np.nan,
            "reusable_group_route_rate": route_count / n if n else np.nan,
            "route_rate_one_sided_95pct_MC_upper": clopper_pearson_upper(route_count, n),
            "continuous_adequacy_rate": block["continuous_adequate"].mean() if n else np.nan,
            "two_sided_95pct_interval_coverage": block["two_sided_coverage"].mean() if n else np.nan,
            "mean_bias": block["bias"].mean() if n else np.nan,
            "absolute_mean_bias_fraction_of_margin": abs(block["bias"].mean()) / margin if n else np.nan,
            "mean_absolute_bias_fraction_of_margin": block["bias"].abs().mean() / margin if n else np.nan,
            "precision_probability_one_sided_half_width_below_margin": (
                block["one_sided_half_width"].lt(margin).mean() if n else np.nan
            ),
            "mean_one_sided_half_width": block["one_sided_half_width"].mean() if n else np.nan,
            "meaningful_margin_nats_per_neuron": margin,
            "gate_status": "not_evaluated_by_surrogate",
            "surrogate_check_status": "not_applicable",
        }
        rows.append(row)

    condition = pd.DataFrame(rows)
    for design, block in replicates.groupby("design", sort=True):
        valid = block.loc[block["valid"].eq(1)]
        hard_null = valid.loc[
            valid["scenario_class"].eq("no_group") & valid["effect_multiplier"].eq(0.0)
        ]
        hard_by_scenario = hard_null.groupby("scenario")["reusable_group_route"].mean()
        false_count = int(hard_null["reusable_group_route"].sum())
        false_rate = false_count / len(hard_null) if len(hard_null) else np.nan
        false_upper = clopper_pearson_upper(false_count, len(hard_null))
        zero = valid.loc[valid["effect_multiplier"].eq(0.0)]
        at_two = valid.loc[
            valid["scenario_class"].eq("group") & valid["effect_multiplier"].eq(2.0)
        ]
        core = at_two.loc[at_two["scenario"].isin(["reusable_groups", "spatially_interdigitated_groups"])]
        stress = at_two.loc[at_two["scenario"].eq("groups_plus_continuum")]
        coverage = valid["two_sided_coverage"].mean()
        precision = valid["one_sided_half_width"].lt(margin).mean()
        bias_fraction = abs(valid["bias"].mean()) / margin
        max_individual = float(hard_by_scenario.max()) if len(hard_by_scenario) else np.nan
        gates = {
            "valid_fraction": len(valid) / len(block) >= 0.99,
            "coverage": coverage >= 0.94,
            "pooled_false_group_rate": false_rate <= 0.05,
            "pooled_false_group_mc_upper": false_upper <= 0.075,
            "individual_hard_null": max_individual <= 0.10,
            "core_recovery_at_maximum_effect_multiplier": core["reusable_group_route"].mean() >= 0.80,
            "stress_recovery_at_maximum_effect_multiplier": stress["reusable_group_route"].mean() >= 0.70,
            "continuous_adequacy_zero": zero["continuous_adequate"].mean() >= 0.80,
            "bias": bias_fraction <= 0.20,
            "precision": precision >= 0.80,
        }
        power_curve = (
            valid.loc[valid["scenario_class"].eq("group")]
            .groupby(["scenario", "effect_multiplier"])["reusable_group_route"]
            .mean()
        )
        mde_by_scenario = []
        for scenario in GROUP_SCENARIOS:
            reached = [
                effect
                for effect in EFFECT_GRID
                if power_curve.get((scenario, effect), 0.0) >= 0.80
            ]
            mde_by_scenario.append(min(reached) if reached else float("inf"))
        rows.append(
            {
                "record_type": "design_summary",
                "design": design,
                "scenario": "all_registered_scenarios",
                "scenario_class": "aggregate",
                "effect_multiplier": np.nan,
                "attempted_replicates": len(block),
                "valid_replicates": len(valid),
                "valid_fraction": len(valid) / len(block),
                "mean_development_cells": valid["n_dev_cells"].mean(),
                "mean_audit_cells": valid["n_audit_cells"].mean(),
                "reusable_group_route_rate": false_rate,
                "route_rate_one_sided_95pct_MC_upper": false_upper,
                "continuous_adequacy_rate": zero["continuous_adequate"].mean(),
                "two_sided_95pct_interval_coverage": coverage,
                "absolute_mean_bias_fraction_of_margin": bias_fraction,
                "precision_probability_one_sided_half_width_below_margin": precision,
            "core_group_route_rate_at_maximum_effect_multiplier": core["reusable_group_route"].mean(),
            "stress_group_route_rate_at_maximum_effect_multiplier": stress["reusable_group_route"].mean(),
                "maximum_individual_hard_null_false_group_rate": max_individual,
                "worst_case_MDE80_effect_multiplier": max(mde_by_scenario),
                "meaningful_margin_nats_per_neuron": margin,
                "gate_status": "not_full_registered_qualification",
                "surrogate_check_status": "pass" if all(gates.values()) else "fail",
                "failed_gates": ";".join(k for k, passed in gates.items() if not passed),
            }
        )
    return pd.DataFrame(rows)


def format_num(value: object, digits: int = 3) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return "NA" if np.isnan(x) else "Inf"
    return f"{x:.{digits}f}"


def write_report(
    path: Path,
    summary: pd.DataFrame,
    replicates: pd.DataFrame,
    args: argparse.Namespace,
    metadata: MetadataLayout,
) -> None:
    design_rows = summary.loc[summary["record_type"].eq("design_summary")]
    lines = [
        "# EP11 outcome-blind five-target recovery surrogate screen",
        "",
        f"**Completed:** {date.today().isoformat()}",
        "",
        "## Scope",
        "",
        "This screen used the observed 206-cell, 36-brain/sample metadata layout ",
        "(cell counts, soma coordinates, provider layer, line, hemisphere, age, and lab), ",
        "but every five-target projection composition was simulated. No real axon ",
        "projection value, provider projection class, or development/audit identity was ",
        "read or emitted. Role allocations were newly sampled from metadata for each ",
        "replicate and therefore do not lock the eventual real-data roles.",
        "",
        "The comparison used one common five-target compositional outcome and its exact ",
        "additive-log-ratio Jacobian. `C` was a flexible continuous kernel density after ",
        "the shared position model; `H` was a development-fitted finite mixture after the ",
        "same position model. Candidate bandwidths and mixture configurations were chosen ",
        "by grouped development cross-validation. Audit brains were scored directly: no ",
        "audit refit, reclustering, label matching, or prevalence recalibration occurred.",
        "",
        "This is a recovery-screen surrogate for the planned evaluator. A surrogate pass ",
        "cannot activate outcome-bearing search and is not proof that the frozen 56-D ",
        "mixed-support likelihood, production B/P/C/D/H implementation, or 99-replicate ",
        "full-search null is qualified.",
        "",
        "## Registered synthetic regimes",
        "",
        "No-group stress cases: smooth spatial, curved, strongly nonuniform, ",
        "depleted-middle, and animal-confounded continua. Positive cases: reusable groups ",
        "without a within-group continuum, reusable groups plus a continuum, and ",
        "spatially interdigitated reusable groups. Positive cases used effect multipliers ",
        "0, 0.5, 1, 1.5, and 2; the fixed screening margin was ",
        f"{args.margin:.3f} nats/neuron. These are generator-amplitude multipliers, ",
        "not oracle predictive gains calibrated to multiples of that margin; therefore ",
        "the maximum-effect route rates below are diagnostics rather than power at 2x ",
        "the scientific margin.",
        "",
        "## Design-level surrogate result",
        "",
        "| Design | Valid | False-group rate (95% MC upper) | Max hard-null rate | "
            "Core group route at max multiplier | Stress group route at max multiplier | Coverage | Precision < margin | "
        "Continuous adequacy at zero | Surrogate check | Registered gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, row in design_rows.iterrows():
        lines.append(
            "| {design} | {valid}/{attempted} | {false_rate} ({false_upper}) | "
            "{max_null} | {core} | {stress} | {coverage} | {precision} | "
            "{adequacy} | {surrogate} | {gate} |".format(
                design=row["design"],
                valid=int(row["valid_replicates"]),
                attempted=int(row["attempted_replicates"]),
                false_rate=format_num(row["reusable_group_route_rate"]),
                false_upper=format_num(row["route_rate_one_sided_95pct_MC_upper"]),
                max_null=format_num(row["maximum_individual_hard_null_false_group_rate"]),
                core=format_num(row["core_group_route_rate_at_maximum_effect_multiplier"]),
                stress=format_num(row["stress_group_route_rate_at_maximum_effect_multiplier"]),
                coverage=format_num(row["two_sided_95pct_interval_coverage"]),
                precision=format_num(row["precision_probability_one_sided_half_width_below_margin"]),
                adequacy=format_num(row["continuous_adequacy_rate"]),
                surrogate=row["surrogate_check_status"],
                gate=row["gate_status"],
            )
        )
    lines.extend(["", "Failed surrogate checks are reported literally; a failure is a design result, not evidence against the biology.", ""])
    for _, row in design_rows.iterrows():
        failed = row.get("failed_gates", "") or "none"
        lines.append(f"- `{row['design']}` failed gates: `{failed}`")

    condition = summary.loc[summary["record_type"].eq("condition")]
    lines.extend(
        [
            "",
            "## Scenario readout",
            "",
            "| Design | Scenario | Effect | N | Group route | Continuous adequate | H-C | Coverage | Half-width < margin |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in condition.iterrows():
        lines.append(
            "| {design} | {scenario} | {effect} | {n} | {route} | {adequacy} | "
            "{hc} | {coverage} | {precision} |".format(
                design=row["design"],
                scenario=row["scenario"],
                effect=format_num(row["effect_multiplier"], 1),
                n=int(row["valid_replicates"]),
                route=format_num(row["reusable_group_route_rate"]),
                adequacy=format_num(row["continuous_adequacy_rate"]),
                hc=format_num(row["mean_H_minus_C"]),
                coverage=format_num(row["two_sided_95pct_interval_coverage"]),
                precision=format_num(row["precision_probability_one_sided_half_width_below_margin"]),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A valid comparison may end on the continuous route. Separation, minimum ",
            "component prevalence, and recurrence are applied only to a positive reusable-",
            "group route. Failure to select groups is never recorded as an invalid comparison.",
            "",
            "Before outcome-bearing search, the frozen 56-D mixed-support likelihood, the full ",
            "production B/P/C/D/H evaluator, and the 99-replicate full-search null must receive ",
            "registered qualification. Any failed recovery or precision check must be resolved ",
            "prospectively. These simulations neither expose ",
            "nor answer the real SSp-tr groups-versus-continuum question.",
            "",
            "## Reproduction",
            "",
            f"- Primary cells: {len(metadata.frame)} in {len(np.unique(metadata.brains))} brain/sample units",
            f"- Replicates requested per condition: {args.replicates}",
            f"- Oracle redraws per replicate: {args.oracle_draws}",
            f"- Root seed: {args.seed}",
            "- Detailed condition metrics: `synthetic_recovery_qualification.csv`",
            "- Script: `tools/run_synthetic_recovery_qualification.py`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def task_grid(args: argparse.Namespace) -> list[tuple]:
    designs = [("12_dev_8_audit_16_unassigned", 12, 8)]
    tasks: list[tuple] = []
    for design, n_dev, n_audit in designs:
        for scenario in args.scenarios:
            effects = EFFECT_GRID if scenario in GROUP_SCENARIOS else (0.0,)
            for effect in effects:
                for replicate in range(args.replicates):
                    seed = stable_u64(
                        args.seed,
                        RNG_DESIGN_KEY,
                        scenario,
                        effect,
                        replicate,
                    ) % (2**32 - 1)
                    tasks.append(
                        (
                            design,
                            n_dev,
                            n_audit,
                            scenario,
                            effect,
                            replicate,
                            seed,
                            args.margin,
                            args.oracle_draws,
                        )
                    )
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--replicate-output", type=Path)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--oracle-draws", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--scenarios", nargs="+", choices=ALL_SCENARIOS, default=list(ALL_SCENARIOS))
    return parser.parse_args()


def main() -> None:
    global _META
    args = parse_args()
    if args.replicates < 1 or args.oracle_draws < 1 or args.workers < 1:
        raise ValueError("replicates, oracle-draws, and workers must be positive")
    _META = load_metadata(args.inventory)
    tasks = task_grid(args)
    if args.workers == 1:
        records = [fit_and_evaluate(task) for task in tasks]
    else:
        import multiprocessing as mp

        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=mp.get_context("fork")
        ) as pool:
            records = list(pool.map(fit_and_evaluate, tasks, chunksize=1))
    replicates = pd.DataFrame(records)
    summary = summarize(replicates, args.margin)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.output_md, summary, replicates, args, _META)
    if args.replicate_output:
        args.replicate_output.parent.mkdir(parents=True, exist_ok=True)
        replicates.to_csv(args.replicate_output, index=False, quoting=csv.QUOTE_MINIMAL)


if __name__ == "__main__":
    main()
