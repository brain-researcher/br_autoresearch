#!/usr/bin/env python3
"""56-coordinate mixed-support synthetic smoke subset for EP11.

Every projection outcome used by this program is synthetic.  The only real
data read are the frozen target vocabulary and the compact, outcome-blind soma
metadata inventory.  The program preserves the observed 206-cell/36-unit
layout and uses exactly 12 development, 8 audit, and 16 unassigned units.

The scored outcome is a 56-coordinate composition with exact zeros.  Its
dominating measure is counting measure over nonempty active-coordinate sets
times the face-specific Euclidean Hausdorff measure.  Each model therefore
scores both

  1. the probability of the observed active set, and
  2. an ordered-Helmert ILR density on that active simplex face, returned to
     Hausdorff measure with the observation-specific Jacobian.

B/P/C/D/H use this one evaluator and the same audit cells.  The fitted models
are an explicitly factorized comparison set:

  * B is the adjustment-only backbone;
  * P is B plus a development-fitted position block, without refitting B;
  * C is P plus one connected development-fitted continuous latent backbone;
  * D is B plus one development-only fixed K=2 group factor; and
  * H is the *identical* C position/continuous backbone plus D's identical
    group signatures and constant development prevalence.

Thus H-C adds only the group factor, while H-D adds only the same position and
continuous factor used by P/C.  Development brain/sample units are divided
into outcome-independent, unit-held-out folds.  The folds evaluate the whole
nuisance/continuous construction and select one noise scale from one fixed
grid by C's equal-animal reference score, then freeze that value into H.  H
never influences C's nuisance choice.  C and H consequently receive
the same nuisance estimates, noise-selection protocol, candidate grid, and
candidate-evaluation budget.  The selected scale is then frozen while the
factorized set is refit to all development units.  Nothing is refit,
reclustered, label-matched, or prevalence-recalibrated on audit units.

For grouped synthetic regimes, the public effect axis is not the generator's
arbitrary coefficient.  A separate synthetic-only projected-oracle *proxy*
maps a fixed amplitude grid to expected H-minus-C log-score gain after fitting
a connected C projection, then interpolates the amplitude required for
prespecified multiples of the 0.05-nat/neuron margin.  It is not the final
projection onto the complete registered C family.  The mapping, brackets,
Monte Carlo budget, and fold separation are emitted as machine-readable result
metadata.

This is deliberately a synthetic smoke *subset*, not a registered
qualification.  It does not authorize outcome-bearing search.  Remaining
blockers are written into every report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

# Prevent nested numerical-library pools in a future Slurm invocation.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, ndtri
from scipy.stats import t as student_t


N_COORDINATES = 56
N_DEVELOPMENT_UNITS = 12
N_AUDIT_UNITS = 8
N_UNASSIGNED_UNITS = 16
K_GROUPS = 2
DEFAULT_MARGIN = 0.05  # nats/neuron on the single frozen outcome measure
CONTINUOUS_QUADRATURE_ORDER = 7
DEVELOPMENT_CROSSFIT_FOLDS = 4
GENERATOR_NOISE_SIGMA = 0.42
SHARED_SIGMA_GRID = (0.30, 0.36, 0.42, 0.50, 0.60, 0.72, 0.88, 1.05, 1.25)
DEFAULT_TARGET_MARGIN_MULTIPLIERS = (0.0, 0.5, 1.0, 1.5, 2.0)
ORACLE_CALIBRATION_AMPLITUDES = (
    0.0,
    0.025,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.30,
    0.45,
    0.65,
    0.90,
    1.25,
    1.75,
    2.50,
)

NO_GROUP_SCENARIOS = (
    "smooth_spatial_continuum",
    "curved_horseshoe_continuum",
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

REMAINING_BLOCKERS = (
    "provider certification that each brain/sample unit is an independent animal",
    "a locked real development/audit role assignment",
    "the final per-cell pial-geodesic and streamline-depth common-support solver",
    "real-axon completeness and truncation eligibility",
    "marginalization of unknown, missing, censored, or truncated target coordinates",
    "exact conditional-nonempty maximum likelihood for support parameters",
    "the final hierarchical new-animal random-effect integral",
    "production qualification of the fixed unit-held-out nuisance/noise protocol",
    "equal-animal weighting inside coefficient residual-PCA and group-signature fitting",
    "absolute calibration and trajectory adequacy for P and C",
    "the full position-standardized reuse ESS overlap influence and technical-decoding gates",
    "smooth position-dependent group gating beyond this subset's fixed constant gate",
    "precision qualification of the synthetic oracle-gain calibration",
    "projection of each grouped fixture onto the full registered C family",
    "separate oracle calibration of the H-minus-D continuum amplitude",
    "the full registered hyperparameter and K search grammar",
    "development-family reference selection under equal adaptive-search budgets",
    "calibrated animal-level intervals for the final evaluator",
    "the 99 complete full-search null controllers",
    "deterministic replay and the global compute envelope",
)


@dataclass(frozen=True)
class MetadataLayout:
    frame: pd.DataFrame
    units: np.ndarray
    lines: np.ndarray
    b_design: np.ndarray
    p_design: np.ndarray
    coords: np.ndarray
    pc: np.ndarray
    layer_depth: np.ndarray


@dataclass(frozen=True)
class RoleSplit:
    development: np.ndarray
    audit: np.ndarray
    unassigned: np.ndarray


@dataclass(frozen=True)
class FaceDatum:
    active: np.ndarray
    helmert: np.ndarray
    ilr: np.ndarray
    log_jacobian: float


@dataclass
class MixedSupportModel:
    name: str
    design_name: str
    support_beta: np.ndarray
    value_beta: np.ndarray
    state_support_shift: np.ndarray
    state_value_shift: np.ndarray
    state_weights: np.ndarray
    sigma: float
    backbone_parent: str = "none"
    continuous_support_backbone: np.ndarray | None = None
    continuous_value_backbone: np.ndarray | None = None
    continuous_weights: np.ndarray | None = None
    group_support_signature: np.ndarray | None = None
    group_value_signature: np.ndarray | None = None
    group_weights: np.ndarray | None = None
    group_gating: str = "none"
    group_signature_protocol: str = "none"
    noise_protocol_id: str = "unassigned"
    paired_training_protocol_id: str = "factorized_shared_backbone_v2"


@dataclass
class FitDiagnostics:
    development_group_labels: np.ndarray
    group_weights: np.ndarray
    continuous_nodes: np.ndarray
    continuous_weights: np.ndarray


@dataclass(frozen=True)
class UnitFold:
    fold_index: int
    training_positions: np.ndarray
    validation_positions: np.ndarray
    training_units: tuple[str, ...]
    validation_units: tuple[str, ...]


@dataclass(frozen=True)
class CrossFitDiagnostics:
    protocol_id: str
    folds: tuple[UnitFold, ...]
    sigma_grid: tuple[float, ...]
    selected_sigma: float
    family_unit_mean_scores: dict[str, tuple[float, ...]]
    selection_reference_family: str
    selection_objective_unit_mean_scores: tuple[float, ...]
    selected_family_unit_mean_scores: dict[str, float]
    fit_count_by_family: dict[str, int]
    sigma_score_count_by_family: dict[str, int]
    unit_coverage_count: int
    train_validation_overlap_count: int
    fold_manifest_fingerprint: str
    unseen_line_rule: str


@dataclass(frozen=True)
class SyntheticTruth:
    groups: np.ndarray
    group_probabilities: np.ndarray
    base_support_logits: np.ndarray
    base_value_mean: np.ndarray
    group_support_shift: np.ndarray
    group_value_shift: np.ndarray
    group_support_direction: np.ndarray
    group_value_direction: np.ndarray
    noise_sigma: float


@dataclass(frozen=True)
class OracleCalibrationPoint:
    scenario: str
    target_margin_multiplier: float
    target_gain_nats_per_neuron: float
    generator_amplitude: float
    estimated_gain_nats_per_neuron: float
    lower_amplitude: float
    upper_amplitude: float
    lower_gain_nats_per_neuron: float
    upper_gain_nats_per_neuron: float
    independent_verification_mean_gain: float
    independent_verification_se: float
    independent_verification_lower95: float
    independent_verification_upper95: float
    projected_C_parameters: tuple[float, ...]


@dataclass(frozen=True)
class OracleCalibration:
    protocol_id: str
    margin_nats_per_neuron: float
    replicates: int
    amplitude_grid: tuple[float, ...]
    scenarios: tuple[str, ...]
    raw_mean_gain_by_scenario: dict[str, tuple[float, ...]]
    raw_se_by_scenario: dict[str, tuple[float, ...]]
    monotone_mean_gain_by_scenario: dict[str, tuple[float, ...]]
    points: tuple[OracleCalibrationPoint, ...]
    projection_family: str
    fixed_role_manifest: dict[str, tuple[str, ...]]
    fixed_role_manifest_fingerprint: str
    fixed_role_is_final_geometry_locked: bool
    projection_diagnostics_by_scenario: dict[str, tuple[dict[str, object], ...]]


@dataclass(frozen=True)
class ConditionSpec:
    scenario: str
    generator_amplitude: float
    target_margin_multiplier: float
    target_oracle_gain_nats_per_neuron: float
    calibrated_oracle_gain_nats_per_neuron: float


@dataclass(frozen=True)
class OracleSample:
    indices: np.ndarray
    outcomes: np.ndarray
    truth: SyntheticTruth


def stable_u64(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
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
    array = np.asarray(values, dtype=str)
    levels = sorted(set(array.tolist()))
    return np.column_stack([(array == level).astype(float) for level in levels])


def standardize_columns(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    mean = np.nanmean(array, axis=0)
    sd = np.nanstd(array, axis=0)
    sd[sd < 1.0e-8] = 1.0
    return (np.nan_to_num(array, nan=mean) - mean) / sd


def validate_vocabulary(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != N_COORDINATES:
        raise ValueError(f"Expected exactly 56 frozen coordinates, found {len(rows)}")
    indices = [int(row["coordinate_index"]) for row in rows]
    if indices != list(range(N_COORDINATES)):
        raise ValueError("Frozen coordinate_index must be exactly 0..55 in file order")
    coordinate_ids = [row["coordinate_id"] for row in rows]
    if len(set(coordinate_ids)) != N_COORDINATES:
        raise ValueError("Frozen coordinate_id values are not unique")
    targets: dict[str, set[str]] = {}
    for row in rows:
        targets.setdefault(row["target_id"], set()).add(row["laterality"])
    expected_laterality = {"ipsilateral", "contralateral"}
    if len(targets) != 28 or any(value != expected_laterality for value in targets.values()):
        raise ValueError("Vocabulary must be 28 targets crossed with ipsi/contralateral")
    return {"coordinate_count": len(rows), "target_count": len(targets)}


def load_metadata(path: Path) -> MetadataLayout:
    frame = pd.read_csv(path, dtype={"fMOST_brain_id": str, "provider_sample_id": str})
    frame = frame.loc[
        frame["primary_axon_dendrite_distinguished_eligible"].eq("yes")
    ].copy()
    if len(frame) != 206:
        raise ValueError(f"Expected 206 primary cells, found {len(frame)}")
    if frame["fMOST_brain_id"].nunique() != 36:
        raise ValueError("Expected 36 distinct fMOST brain/sample units")
    pairs = frame[["fMOST_brain_id", "provider_sample_id"]].drop_duplicates()
    if len(pairs) != 36:
        raise ValueError("Expected a one-to-one 36-pair brain/sample layout")
    if pairs["fMOST_brain_id"].nunique() != 36 or pairs["provider_sample_id"].nunique() != 36:
        raise ValueError("Brain and provider-sample identifiers are not one-to-one")

    coords = standardize_columns(
        frame[["soma_x_ccfv3_um", "soma_y_ccfv3_um", "soma_z_ccfv3_um"]].to_numpy()
    )
    _, _, right = np.linalg.svd(coords, full_matrices=False)
    pc = coords @ right.T
    layer_depth = np.array([layer_depth_from_label(x) for x in frame["source_acronym"]])
    age = standardize_columns(frame[["sample_age_days"]].to_numpy())
    gender = standardize_columns(frame[["sample_gender_code"]].to_numpy())

    b_design = np.column_stack(
        [
            np.ones(len(frame)),
            age,
            gender,
            one_hot(frame["strain_or_cre_line"].astype(str)),
            one_hot(frame["hemisphere"].astype(str)),
            one_hot(frame["sample_lab_name"].astype(str)),
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
            one_hot(frame["source_acronym"].astype(str)),
        ]
    )
    p_design = np.column_stack([b_design, position_basis])
    return MetadataLayout(
        frame=frame.reset_index(drop=True),
        units=frame["fMOST_brain_id"].astype(str).to_numpy(),
        lines=frame["strain_or_cre_line"].astype(str).to_numpy(),
        b_design=b_design,
        p_design=p_design,
        coords=coords,
        pc=pc,
        layer_depth=layer_depth,
    )


def role_split(layout: MetadataLayout, rng: np.random.Generator) -> RoleSplit:
    unit_rows = layout.frame.groupby("fMOST_brain_id", sort=True).first().reset_index()
    singleton_line = "Tle4-2A-CreER"
    eligible = unit_rows.loc[unit_rows["strain_or_cre_line"].ne(singleton_line)].copy()
    line_to_units = {
        line: group["fMOST_brain_id"].astype(str).tolist()
        for line, group in eligible.groupby("strain_or_cre_line", sort=True)
    }
    if len(line_to_units) != N_AUDIT_UNITS:
        raise ValueError("Role sampler expects eight represented nonsingleton lines")

    audit_units: list[str] = []
    development_units: list[str] = []
    for line in sorted(line_to_units):
        choices = np.asarray(line_to_units[line], dtype=object)
        if len(choices) < 2:
            raise ValueError(f"Line {line!r} cannot support development and audit")
        selected = rng.choice(choices, size=2, replace=False)
        audit_units.append(str(selected[0]))
        development_units.append(str(selected[1]))

    used = set(audit_units) | set(development_units)
    remaining = [unit for unit in eligible["fMOST_brain_id"].astype(str) if unit not in used]
    rng.shuffle(remaining)
    development_units.extend(remaining[: N_DEVELOPMENT_UNITS - len(development_units)])
    all_units = set(unit_rows["fMOST_brain_id"].astype(str))
    unassigned_units = sorted(all_units - set(development_units) - set(audit_units))
    if (
        len(development_units) != N_DEVELOPMENT_UNITS
        or len(audit_units) != N_AUDIT_UNITS
        or len(unassigned_units) != N_UNASSIGNED_UNITS
        or set(development_units) & set(audit_units)
    ):
        raise RuntimeError("Could not construct an exact disjoint 12/8/16 split")

    return RoleSplit(
        development=np.flatnonzero(np.isin(layout.units, development_units)),
        audit=np.flatnonzero(np.isin(layout.units, audit_units)),
        unassigned=np.flatnonzero(np.isin(layout.units, unassigned_units)),
    )


def ordered_helmert(size: int) -> np.ndarray:
    """Return the fixed ordered Helmert ILR basis (columns are contrasts)."""
    if size < 1:
        raise ValueError("An active face must contain at least one coordinate")
    basis = np.zeros((size, max(size - 1, 0)), dtype=float)
    for k in range(1, size):
        denominator = math.sqrt(k * (k + 1.0))
        basis[:k, k - 1] = 1.0 / denominator
        basis[k, k - 1] = -k / denominator
    return basis


def face_log_jacobian(values: np.ndarray) -> float:
    """log |d ILR / d H_face| for Euclidean Hausdorff measure.

    For an m-part composition, the ordered Helmert ILR Jacobian relative to
    (m-1)-dimensional Hausdorff measure on the simplex is
    1 / (m * product_j p_j).  For m=1 this correctly gives the unit point mass.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 1 or np.any(values <= 0.0):
        raise ValueError("Face values must be strictly positive")
    return float(-math.log(len(values)) - np.log(values).sum())


def make_face_data(composition: np.ndarray) -> FaceDatum:
    composition = np.asarray(composition, dtype=float)
    if composition.shape != (N_COORDINATES,):
        raise ValueError("Composition must have exactly 56 coordinates")
    active = np.flatnonzero(composition > 0.0)
    if len(active) == 0:
        raise ValueError("The all-zero support face is excluded")
    values = composition[active]
    if not np.isclose(values.sum(), 1.0, atol=1.0e-10):
        raise ValueError("Positive face values must sum to one")
    helmert = ordered_helmert(len(active))
    ilr = helmert.T @ np.log(values)
    return FaceDatum(
        active=active,
        helmert=helmert,
        ilr=ilr,
        log_jacobian=face_log_jacobian(values),
    )


def log1mexp(log_value: float) -> float:
    """Stable log(1-exp(log_value)) for log_value <= 0."""
    if log_value > 0.0:
        raise ValueError("log1mexp domain requires a nonpositive value")
    if log_value < math.log(0.5):
        return math.log1p(-math.exp(log_value))
    return math.log(-math.expm1(log_value))


def log_nonempty_support_probability(logits: np.ndarray) -> float:
    """Stable log probability that independent Bernoulli support is nonempty."""
    logits = np.asarray(logits, dtype=float)
    if logits.shape != (N_COORDINATES,) or not np.isfinite(logits).all():
        raise ValueError("Support logits must be 56 finite values")
    log_softplus = np.empty_like(logits)
    low = logits < -35.0
    high = logits > 35.0
    middle = ~(low | high)
    log_softplus[low] = logits[low]
    log_softplus[high] = np.log(logits[high])
    log_softplus[middle] = np.log(np.logaddexp(0.0, logits[middle]))
    log_total_hazard = float(logsumexp(log_softplus))
    if log_total_hazard < -20.0:
        total_hazard = math.exp(log_total_hazard)
        return log_total_hazard + math.log1p(
            -0.5 * total_hazard + total_hazard**2 / 6.0
        )
    total_hazard = math.exp(log_total_hazard)
    return math.log(-math.expm1(-total_hazard))


def active_set_log_probability(mask: np.ndarray, logits: np.ndarray) -> float:
    """Independent Bernoulli support probability, conditioned on nonempty."""
    mask = np.asarray(mask, dtype=bool)
    logits = np.asarray(logits, dtype=float)
    if mask.shape != (N_COORDINATES,) or logits.shape != (N_COORDINATES,):
        raise ValueError("Support vectors must both have length 56")
    if not np.isfinite(logits).all():
        raise ValueError("Support logits must be finite")
    if not mask.any():
        return -math.inf
    # log sigmoid(x)=-logaddexp(0,-x); log(1-sigmoid(x))=-logaddexp(0,x)
    log_on = -np.logaddexp(0.0, -logits)
    log_off = -np.logaddexp(0.0, logits)
    log_unconditioned = float(np.where(mask, log_on, log_off).sum())
    return log_unconditioned - log_nonempty_support_probability(logits)


def build_clr_targets(outcomes: np.ndarray) -> np.ndarray:
    targets = np.full_like(outcomes, np.nan, dtype=float)
    for i, composition in enumerate(outcomes):
        active = composition > 0.0
        logs = np.log(composition[active])
        targets[i, active] = logs - logs.mean()
    return targets


def fit_logistic_majorizer(
    design: np.ndarray,
    binary: np.ndarray,
    ridge: float,
    offset: np.ndarray | None = None,
    unpenalized_first: bool = True,
    max_iter: int = 120,
) -> np.ndarray:
    """Vectorized ridge-logistic MM fit for all 56 support coordinates."""
    design = np.asarray(design, dtype=float)
    binary = np.asarray(binary, dtype=float)
    n_rows, n_terms = design.shape
    if binary.shape != (n_rows, N_COORDINATES):
        raise ValueError("Binary support matrix has the wrong shape")
    fixed = np.zeros_like(binary) if offset is None else np.asarray(offset, dtype=float)
    if fixed.shape != binary.shape:
        raise ValueError("Logistic offset has the wrong shape")

    penalty = np.full(n_terms, float(ridge))
    if unpenalized_first and n_terms:
        penalty[0] = 0.0
    hessian_bound = 0.25 * (design.T @ design) + np.diag(penalty + 1.0e-8)
    coefficients = np.zeros((n_terms, N_COORDINATES), dtype=float)
    if unpenalized_first and n_terms and np.allclose(design[:, 0], 1.0):
        prevalence = np.clip(binary.mean(axis=0), 0.01, 0.99)
        coefficients[0] = np.log(prevalence / (1.0 - prevalence))

    for _ in range(max_iter):
        fitted = expit(np.clip(fixed + design @ coefficients, -18.0, 18.0))
        gradient = design.T @ (fitted - binary) + penalty[:, None] * coefficients
        step = np.linalg.solve(hessian_bound, gradient)
        coefficients -= step
        if float(np.max(np.abs(step))) < 2.0e-7:
            break
    return coefficients


def ridge_solve(design: np.ndarray, response: np.ndarray, ridge: float) -> np.ndarray:
    penalty = np.eye(design.shape[1]) * ridge
    if design.shape[1] and np.allclose(design[:, 0], 1.0):
        penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty + np.eye(design.shape[1]) * 1.0e-8,
                           design.T @ response)


def fit_value_regression(
    design: np.ndarray,
    clr_targets: np.ndarray,
    active: np.ndarray,
    ridge: float,
    offset: np.ndarray | None = None,
) -> np.ndarray:
    n_terms = design.shape[1]
    coefficients = np.zeros((n_terms, N_COORDINATES), dtype=float)
    fixed = np.zeros_like(clr_targets) if offset is None else np.asarray(offset, dtype=float)
    for coordinate in range(N_COORDINATES):
        observed = active[:, coordinate]
        if int(observed.sum()) < max(3, n_terms // 5):
            # A heavily regularized intercept-like estimate is safer than an
            # unidentified high-dimensional target-specific fit.
            if observed.any():
                coefficients[0, coordinate] = float(
                    np.mean(clr_targets[observed, coordinate] - fixed[observed, coordinate])
                ) if n_terms else 0.0
            continue
        coefficients[:, coordinate] = ridge_solve(
            design[observed],
            clr_targets[observed, coordinate] - fixed[observed, coordinate],
            ridge,
        )
    return coefficients


def estimate_sigma(outcomes: np.ndarray, mean_log_intensity: np.ndarray) -> float:
    residual_sum = 0.0
    dimensions = 0
    for composition, mean in zip(outcomes, mean_log_intensity):
        active = composition > 0.0
        count = int(active.sum())
        if count <= 1:
            continue
        residual = np.log(composition[active]) - mean[active]
        residual -= residual.mean()
        residual_sum += float(residual @ residual)
        dimensions += count - 1
    if dimensions == 0:
        return 0.5
    return float(np.clip(math.sqrt(residual_sum / dimensions), 0.18, 1.50))


def fit_base_model(
    name: str,
    design_name: str,
    design: np.ndarray,
    outcomes: np.ndarray,
    support_ridge: float,
    value_ridge: float,
) -> MixedSupportModel:
    active = outcomes > 0.0
    clr_targets = build_clr_targets(outcomes)
    support_beta = fit_logistic_majorizer(design, active, support_ridge)
    value_beta = fit_value_regression(design, clr_targets, active, value_ridge)
    mean = design @ value_beta
    return MixedSupportModel(
        name=name,
        design_name=design_name,
        support_beta=support_beta,
        value_beta=value_beta,
        state_support_shift=np.zeros((1, N_COORDINATES)),
        state_value_shift=np.zeros((1, N_COORDINATES)),
        state_weights=np.ones(1),
        sigma=estimate_sigma(outcomes, mean),
    )


def residual_features(
    outcomes: np.ndarray,
    support_logits: np.ndarray,
    mean_log_intensity: np.ndarray,
) -> np.ndarray:
    active = outcomes > 0.0
    probabilities = np.clip(expit(support_logits), 0.02, 0.98)
    support_residual = (active - probabilities) / np.sqrt(probabilities * (1.0 - probabilities))
    value_residual = np.zeros_like(outcomes)
    for i, composition in enumerate(outcomes):
        on = active[i]
        observed = np.log(composition[on])
        residual = observed - mean_log_intensity[i, on]
        residual -= residual.mean()
        value_residual[i, on] = residual
    features = np.column_stack([0.45 * support_residual, value_residual])
    features -= features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 0.10] = 1.0
    return features / scale


def principal_scores(features: np.ndarray, n_components: int = 6) -> tuple[np.ndarray, np.ndarray]:
    _, singular, right = np.linalg.svd(features, full_matrices=False)
    count = min(n_components, len(singular))
    scores = features @ right[:count].T
    if count and float(right[0, :N_COORDINATES].sum()) < 0.0:
        scores[:, 0] *= -1.0
        right[0] *= -1.0
    return scores, right[:count]


def rank_normal(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(len(order), dtype=float)
    probabilities = (ranks + 0.5) / len(order)
    return ndtri(np.clip(probabilities, 1.0e-5, 1.0 - 1.0e-5))


def two_means(features: np.ndarray) -> np.ndarray:
    scores, _ = principal_scores(features, n_components=4)
    if scores.shape[1] == 0:
        raise ValueError("No residual dimensions available for K=2 grouping")
    centers = np.vstack([scores[np.argmin(scores[:, 0])], scores[np.argmax(scores[:, 0])]])
    labels = np.zeros(len(scores), dtype=int)
    for _ in range(100):
        distance = ((scores[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        updated = np.argmin(distance, axis=1)
        if np.array_equal(updated, labels) and np.all(np.bincount(updated, minlength=2) > 0):
            break
        labels = updated
        if np.any(np.bincount(labels, minlength=2) == 0):
            labels = (scores[:, 0] > np.median(scores[:, 0])).astype(int)
        centers = np.vstack([scores[labels == group].mean(axis=0) for group in range(2)])
    if min(np.bincount(labels, minlength=2)) < 4:
        labels = (scores[:, 0] > np.median(scores[:, 0])).astype(int)
        centers = np.vstack([scores[labels == group].mean(axis=0) for group in range(2)])
    if centers[0, 0] > centers[1, 0]:
        labels = 1 - labels
    return labels


def fit_adjustments(
    outcomes: np.ndarray,
    fixed_support_logits: np.ndarray,
    fixed_value_mean: np.ndarray,
    latent_design: np.ndarray,
    support_ridge: float = 5.0,
    value_ridge: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    active = outcomes > 0.0
    clr_targets = build_clr_targets(outcomes)
    support_coef = fit_logistic_majorizer(
        latent_design,
        active,
        support_ridge,
        offset=fixed_support_logits,
        unpenalized_first=False,
    )
    value_coef = fit_value_regression(
        latent_design,
        clr_targets,
        active,
        value_ridge,
        offset=fixed_value_mean,
    )
    return support_coef, value_coef


def continuous_basis(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.column_stack([values, (values**2 - 1.0) / math.sqrt(2.0)])


def quadrature() -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite.hermgauss(CONTINUOUS_QUADRATURE_ORDER)
    return nodes * math.sqrt(2.0), weights / math.sqrt(math.pi)


def assert_factorized_model_set(
    models: dict[str, MixedSupportModel],
    b_term_count: int,
) -> None:
    """Assert the exact B/P/C/D/H factorization used by this subset.

    These are executable contract checks, not approximate capacity checks.
    In particular, they ensure that neither matched comparison can acquire a
    separately learned backbone, group definition, prevalence, or noise scale.
    """
    if set(models) != {"B", "P", "C", "D", "H"}:
        raise AssertionError("The factorized comparison must contain exactly B/P/C/D/H")
    model_b, model_p = models["B"], models["P"]
    model_c, model_d, model_h = models["C"], models["D"], models["H"]

    if model_b.design_name != "B" or model_b.backbone_parent != "adjustment_only":
        raise AssertionError("B is not the adjustment-only backbone")
    if model_p.design_name != "P" or model_p.backbone_parent != "B":
        raise AssertionError("P is not represented as B plus position")
    if model_c.design_name != "P" or model_c.backbone_parent != "P":
        raise AssertionError("C is not represented as P plus a continuous backbone")
    if model_d.design_name != "B" or model_d.backbone_parent != "B":
        raise AssertionError("D must be based on B, not P")
    if model_h.design_name != "P" or model_h.backbone_parent != "C":
        raise AssertionError("H must be based on the complete C backbone")

    # P retains B element-for-element and only appends the position block.
    np.testing.assert_array_equal(model_p.support_beta[:b_term_count], model_b.support_beta)
    np.testing.assert_array_equal(model_p.value_beta[:b_term_count], model_b.value_beta)
    # D is B plus groups; C is P plus the connected continuous factor; H is C
    # plus groups.  The underlying adjustment/position coefficients are not
    # independently re-estimated for either matched comparison.
    np.testing.assert_array_equal(model_d.support_beta, model_b.support_beta)
    np.testing.assert_array_equal(model_d.value_beta, model_b.value_beta)
    np.testing.assert_array_equal(model_c.support_beta, model_p.support_beta)
    np.testing.assert_array_equal(model_c.value_beta, model_p.value_beta)
    np.testing.assert_array_equal(model_h.support_beta, model_c.support_beta)
    np.testing.assert_array_equal(model_h.value_beta, model_c.value_beta)

    required_continuous = (
        model_c.continuous_support_backbone,
        model_c.continuous_value_backbone,
        model_c.continuous_weights,
        model_h.continuous_support_backbone,
        model_h.continuous_value_backbone,
        model_h.continuous_weights,
    )
    if any(item is None for item in required_continuous):
        raise AssertionError("C/H must expose their paired continuous backbone")
    np.testing.assert_array_equal(
        model_h.continuous_support_backbone,
        model_c.continuous_support_backbone,
    )
    np.testing.assert_array_equal(
        model_h.continuous_value_backbone,
        model_c.continuous_value_backbone,
    )
    np.testing.assert_array_equal(model_h.continuous_weights, model_c.continuous_weights)
    np.testing.assert_array_equal(model_c.state_support_shift, model_c.continuous_support_backbone)
    np.testing.assert_array_equal(model_c.state_value_shift, model_c.continuous_value_backbone)
    np.testing.assert_array_equal(model_c.state_weights, model_c.continuous_weights)

    required_group = (
        model_d.group_support_signature,
        model_d.group_value_signature,
        model_d.group_weights,
        model_h.group_support_signature,
        model_h.group_value_signature,
        model_h.group_weights,
    )
    if any(item is None for item in required_group):
        raise AssertionError("D/H must expose their paired group factor")
    np.testing.assert_array_equal(model_h.group_support_signature, model_d.group_support_signature)
    np.testing.assert_array_equal(model_h.group_value_signature, model_d.group_value_signature)
    np.testing.assert_array_equal(model_h.group_weights, model_d.group_weights)
    if model_d.group_gating != "constant_development_only" or model_h.group_gating != model_d.group_gating:
        raise AssertionError("D/H do not share the same development-only group gating")
    expected_group_protocol = "labels_and_signatures_conditional_on_frozen_C_training_backbone"
    if (
        model_d.group_signature_protocol != expected_group_protocol
        or model_h.group_signature_protocol != expected_group_protocol
    ):
        raise AssertionError("D/H group signatures were not estimated conditionally on C")
    np.testing.assert_array_equal(model_d.state_support_shift, model_d.group_support_signature)
    np.testing.assert_array_equal(model_d.state_value_shift, model_d.group_value_signature)
    np.testing.assert_array_equal(model_d.state_weights, model_d.group_weights)

    # H is the Cartesian product of exactly those two frozen factors.  These
    # assertions prove elementwise that H-C adds groups and H-D adds the same
    # position/continuous backbone used by P/C.
    expected_support = np.vstack(
        [
            model_d.group_support_signature[group] + continuous
            for group in range(K_GROUPS)
            for continuous in model_c.continuous_support_backbone
        ]
    )
    expected_value = np.vstack(
        [
            model_d.group_value_signature[group] + continuous
            for group in range(K_GROUPS)
            for continuous in model_c.continuous_value_backbone
        ]
    )
    expected_weights = np.concatenate(
        [model_d.group_weights[group] * model_c.continuous_weights for group in range(K_GROUPS)]
    )
    np.testing.assert_array_equal(model_h.state_support_shift, expected_support)
    np.testing.assert_array_equal(model_h.state_value_shift, expected_value)
    np.testing.assert_array_equal(model_h.state_weights, expected_weights)
    for group in range(K_GROUPS):
        start = group * len(model_c.continuous_weights)
        stop = start + len(model_c.continuous_weights)
        np.testing.assert_allclose(
            model_h.state_support_shift[start:stop] - model_d.group_support_signature[group],
            model_c.continuous_support_backbone,
            atol=1.0e-14,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            model_h.state_value_shift[start:stop] - model_d.group_value_signature[group],
            model_c.continuous_value_backbone,
            atol=1.0e-14,
            rtol=0.0,
        )

    if not (model_b.sigma == model_p.sigma == model_c.sigma == model_d.sigma == model_h.sigma):
        raise AssertionError("A model-specific noise scale would break the strict factor comparison")
    protocol_ids = {
        model_b.noise_protocol_id,
        model_p.noise_protocol_id,
        model_c.noise_protocol_id,
        model_d.noise_protocol_id,
        model_h.noise_protocol_id,
    }
    if len(protocol_ids) != 1:
        raise AssertionError("Models do not share one registered noise-estimation protocol")
    if model_c.paired_training_protocol_id != model_h.paired_training_protocol_id:
        raise AssertionError("C/H do not share one paired training protocol")


def fit_models(
    layout: MetadataLayout,
    development: np.ndarray,
    outcomes: np.ndarray,
    shared_sigma_override: float | None = None,
    noise_protocol_id: str = "provisional_training_only",
) -> tuple[dict[str, MixedSupportModel], FitDiagnostics]:
    b_design = layout.b_design[development]
    p_design = layout.p_design[development]
    position_design = p_design[:, b_design.shape[1] :]

    # B is fitted once and its adjustment coefficients are frozen into every
    # other member.  A provisional call uses B's residual scale only to build
    # fold candidates; the final call supplies C's unit-held-out selected scale
    # and freezes that common value into every member, including H.
    model_b = fit_base_model("B", "B", b_design, outcomes, 7.0, 7.0)
    model_b.backbone_parent = "adjustment_only"
    shared_sigma = (
        model_b.sigma if shared_sigma_override is None else float(shared_sigma_override)
    )
    if not np.isfinite(shared_sigma) or shared_sigma <= 0.0:
        raise ValueError("The shared noise scale must be finite and positive")
    model_b.sigma = shared_sigma
    model_b.noise_protocol_id = noise_protocol_id
    b_support = b_design @ model_b.support_beta
    b_value = b_design @ model_b.value_beta

    # P is exactly B plus a position block.  B is not jointly refitted here.
    position_support_coef, position_value_coef = fit_adjustments(
        outcomes,
        b_support,
        b_value,
        position_design,
        support_ridge=9.0,
        value_ridge=9.0,
    )
    model_p = MixedSupportModel(
        name="P",
        design_name="P",
        support_beta=np.vstack([model_b.support_beta, position_support_coef]),
        value_beta=np.vstack([model_b.value_beta, position_value_coef]),
        state_support_shift=np.zeros((1, N_COORDINATES)),
        state_value_shift=np.zeros((1, N_COORDINATES)),
        state_weights=np.ones(1),
        sigma=shared_sigma,
        backbone_parent="B",
        noise_protocol_id=noise_protocol_id,
    )
    p_support = p_design @ model_p.support_beta
    p_value = p_design @ model_p.value_beta
    p_residual = residual_features(outcomes, p_support, p_value)

    # C: one connected latent variable, with a quadratic smooth term, integrated
    # over a fixed standard-normal quadrature rather than discretely selected.
    p_scores, _ = principal_scores(p_residual, n_components=1)
    u_c = rank_normal(p_scores[:, 0])
    z_c = continuous_basis(u_c)
    c_support_coef, c_value_coef = fit_adjustments(outcomes, p_support, p_value, z_c)
    nodes, node_weights = quadrature()
    node_basis = continuous_basis(nodes)
    c_support_shift = node_basis @ c_support_coef
    c_value_shift = node_basis @ c_value_coef
    model_c = MixedSupportModel(
        name="C",
        design_name="P",
        support_beta=model_p.support_beta,
        value_beta=model_p.value_beta,
        state_support_shift=c_support_shift,
        state_value_shift=c_value_shift,
        state_weights=node_weights,
        sigma=shared_sigma,
        backbone_parent="P",
        continuous_support_backbone=c_support_shift.copy(),
        continuous_value_backbone=c_value_shift.copy(),
        continuous_weights=node_weights.copy(),
        noise_protocol_id=noise_protocol_id,
    )

    # D/H share one development-only K=2 definition, but the group definition
    # and signature are learned *after* conditioning on the frozen C training
    # backbone.  This prevents H from receiving a B-residual signature that
    # merely rediscovers the continuous structure already present in C.  The
    # resulting signature is then transported unchanged to D, preserving the
    # D/H factorial comparison even though D deliberately omits C's backbone.
    conditional_c_support = p_support + z_c @ c_support_coef
    conditional_c_value = p_value + z_c @ c_value_coef
    c_residual = residual_features(outcomes, conditional_c_support, conditional_c_value)
    labels = two_means(c_residual)
    counts = np.bincount(labels, minlength=K_GROUPS).astype(float)
    group_weights = (counts + 1.0) / (counts.sum() + K_GROUPS)
    p_one = group_weights[1]
    p_zero = group_weights[0]
    group_codes = np.array([-math.sqrt(p_one / p_zero), math.sqrt(p_zero / p_one)])
    z_group = group_codes[labels, None]
    group_support_coef, group_value_coef = fit_adjustments(
        outcomes,
        conditional_c_support,
        conditional_c_value,
        z_group,
    )
    group_support_shift = group_codes[:, None] @ group_support_coef
    group_value_shift = group_codes[:, None] @ group_value_coef
    model_d = MixedSupportModel(
        name="D",
        design_name="B",
        support_beta=model_b.support_beta,
        value_beta=model_b.value_beta,
        state_support_shift=group_support_shift,
        state_value_shift=group_value_shift,
        state_weights=group_weights,
        sigma=shared_sigma,
        backbone_parent="B",
        group_support_signature=group_support_shift.copy(),
        group_value_signature=group_value_shift.copy(),
        group_weights=group_weights.copy(),
        group_gating="constant_development_only",
        group_signature_protocol="labels_and_signatures_conditional_on_frozen_C_training_backbone",
        noise_protocol_id=noise_protocol_id,
    )

    # H is C plus D's group factor.  It does not learn a second latent axis,
    # smooth, signature, prevalence, gate, or noise scale.
    h_support_states = np.vstack(
        [group_support_shift[group] + shift for group in range(K_GROUPS) for shift in c_support_shift]
    )
    h_value_states = np.vstack(
        [group_value_shift[group] + shift for group in range(K_GROUPS) for shift in c_value_shift]
    )
    h_weights = np.concatenate([group_weights[group] * node_weights for group in range(K_GROUPS)])
    model_h = MixedSupportModel(
        name="H",
        design_name="P",
        support_beta=model_p.support_beta,
        value_beta=model_p.value_beta,
        state_support_shift=h_support_states,
        state_value_shift=h_value_states,
        state_weights=h_weights,
        sigma=shared_sigma,
        backbone_parent="C",
        continuous_support_backbone=c_support_shift.copy(),
        continuous_value_backbone=c_value_shift.copy(),
        continuous_weights=node_weights.copy(),
        group_support_signature=group_support_shift.copy(),
        group_value_signature=group_value_shift.copy(),
        group_weights=group_weights.copy(),
        group_gating="constant_development_only",
        group_signature_protocol="labels_and_signatures_conditional_on_frozen_C_training_backbone",
        noise_protocol_id=noise_protocol_id,
    )

    models = {model.name: model for model in (model_b, model_p, model_c, model_d, model_h)}
    assert_factorized_model_set(models, b_design.shape[1])
    return models, FitDiagnostics(labels, group_weights, nodes, node_weights)


def design_for(model: MixedSupportModel, layout: MetadataLayout, indices: np.ndarray) -> np.ndarray:
    return layout.b_design[indices] if model.design_name == "B" else layout.p_design[indices]


def score_model(
    model: MixedSupportModel,
    design: np.ndarray,
    outcomes: np.ndarray,
    return_components: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Score exact-zero compositions under the one common mixed-support measure."""
    base_support = design @ model.support_beta
    base_value = design @ model.value_beta
    faces = [make_face_data(row) for row in outcomes]
    log_weights = np.log(np.clip(model.state_weights, 1.0e-300, 1.0))
    components = np.empty((len(outcomes), len(model.state_weights)), dtype=float)
    variance = model.sigma**2
    for i, face in enumerate(faces):
        mask = np.zeros(N_COORDINATES, dtype=bool)
        mask[face.active] = True
        dimension = len(face.active) - 1
        normal_constant = dimension * math.log(2.0 * math.pi * variance)
        for state in range(len(model.state_weights)):
            support_logits = base_support[i] + model.state_support_shift[state]
            support_logp = active_set_log_probability(mask, support_logits)
            mean_active = base_value[i, face.active] + model.state_value_shift[state, face.active]
            mean_ilr = face.helmert.T @ mean_active
            residual = face.ilr - mean_ilr
            value_logp = -0.5 * (normal_constant + float(residual @ residual) / variance)
            value_logp += face.log_jacobian
            components[i, state] = log_weights[state] + support_logp + value_logp
    scores = logsumexp(components, axis=1)
    return scores, components if return_components else None


def make_development_unit_folds(
    layout: MetadataLayout,
    development: np.ndarray,
    seed: int,
    n_folds: int = DEVELOPMENT_CROSSFIT_FOLDS,
) -> tuple[UnitFold, ...]:
    """Create deterministic outcome-independent folds of whole development units."""
    development = np.asarray(development, dtype=int)
    development_units = layout.units[development]
    unique_units = sorted(set(development_units.tolist()))
    if len(unique_units) != N_DEVELOPMENT_UNITS:
        raise ValueError("Cross-fitting requires exactly 12 development units")
    if n_folds < 2 or len(unique_units) % n_folds:
        raise ValueError("Development units must divide exactly across cross-fit folds")

    # The allocation depends only on the prespecified seed and unit metadata,
    # never on a projection outcome.  Repeated line identities are spread over
    # folds before load balancing.  A singleton line is necessarily unseen
    # when its sole unit is validated; the model's explicit rule then sets its
    # unavailable line deviation to zero relative to the fitted intercept.
    unit_to_line = {
        unit: str(layout.lines[development][np.flatnonzero(development_units == unit)[0]])
        for unit in unique_units
    }
    line_to_units: dict[str, list[str]] = {}
    for unit in unique_units:
        line_to_units.setdefault(unit_to_line[unit], []).append(unit)
    fold_lists: list[list[str]] = [[] for _ in range(n_folds)]
    fold_lines: list[set[str]] = [set() for _ in range(n_folds)]
    ordered_lines = sorted(
        line_to_units,
        key=lambda line: (-len(line_to_units[line]), stable_u64(seed, "line", line)),
    )
    for line in ordered_lines:
        units_for_line = sorted(
            line_to_units[line], key=lambda unit: stable_u64(seed, "unit", unit)
        )
        for unit in units_for_line:
            chosen = min(
                range(n_folds),
                key=lambda fold_index: (
                    line in fold_lines[fold_index],
                    len(fold_lists[fold_index]),
                    stable_u64(seed, "tie", line, unit, fold_index),
                ),
            )
            fold_lists[chosen].append(unit)
            fold_lines[chosen].add(line)
    fold_units = [tuple(sorted(values)) for values in fold_lists]
    expected_units_per_fold = len(unique_units) // n_folds
    if any(len(values) != expected_units_per_fold for values in fold_units):
        raise AssertionError("Line-aware cross-fit allocation is not unit-balanced")
    folds: list[UnitFold] = []
    validation_union: set[str] = set()
    for fold_index, validation_units in enumerate(fold_units):
        validation_set = set(validation_units)
        training_units = tuple(sorted(set(unique_units) - validation_set))
        training_positions = np.flatnonzero(np.isin(development_units, training_units))
        validation_positions = np.flatnonzero(np.isin(development_units, validation_units))
        if set(training_units) & validation_set:
            raise AssertionError("Cross-fit training and validation units overlap")
        if len(training_units) != N_DEVELOPMENT_UNITS - len(validation_units):
            raise AssertionError("Cross-fit fold has the wrong number of training units")
        if not len(training_positions) or not len(validation_positions):
            raise AssertionError("Cross-fit fold has no training or validation cells")
        validation_union.update(validation_set)
        folds.append(
            UnitFold(
                fold_index=fold_index,
                training_positions=training_positions,
                validation_positions=validation_positions,
                training_units=training_units,
                validation_units=validation_units,
            )
        )
    if validation_union != set(unique_units):
        raise AssertionError("Cross-fit validation folds do not cover every development unit")
    if sum(len(fold.validation_units) for fold in folds) != len(unique_units):
        raise AssertionError("A development unit appears in more than one validation fold")
    return tuple(folds)


def equal_unit_mean(layout: MetadataLayout, indices: np.ndarray, values: np.ndarray) -> float:
    indices = np.asarray(indices, dtype=int)
    values = np.asarray(values, dtype=float)
    if len(indices) != len(values):
        raise ValueError("Unit-weighted score received misaligned cells and values")
    units = layout.units[indices]
    unit_means = [float(values[units == unit].mean()) for unit in sorted(set(units.tolist()))]
    if not unit_means:
        raise ValueError("Cannot aggregate an empty unit-weighted score")
    return float(np.mean(unit_means))


def fit_models_with_unit_crossfit(
    layout: MetadataLayout,
    development: np.ndarray,
    outcomes: np.ndarray,
    fold_seed: int,
    sigma_grid: Sequence[float] = SHARED_SIGMA_GRID,
) -> tuple[dict[str, MixedSupportModel], FitDiagnostics, CrossFitDiagnostics]:
    """Select C's reference noise scale and freeze it unchanged into H.

    Each fold refits B, P, C, and the C-conditional group factor without its
    validation units.  C and H are then scored on precisely the same validation
    cells at precisely the same candidate sigmas.  The common sigma maximizes
    C's equal-unit validation score; H is evaluated on the same grid for a
    budget check but cannot influence C's nuisance choice.  This is a shared
    C-reference protocol, not an H-specific or in-sample residual estimate.
    """
    development = np.asarray(development, dtype=int)
    outcomes = np.asarray(outcomes, dtype=float)
    if len(development) != len(outcomes):
        raise ValueError("Development indices and outcomes are misaligned")
    sigma_values = tuple(float(value) for value in sigma_grid)
    if not sigma_values or any(not np.isfinite(value) or value <= 0.0 for value in sigma_values):
        raise ValueError("Shared sigma grid must contain finite positive values")
    if tuple(sorted(set(sigma_values))) != sigma_values:
        raise ValueError("Shared sigma grid must be strictly increasing and unique")

    folds = make_development_unit_folds(layout, development, fold_seed)
    per_family_per_sigma: dict[str, list[list[float]]] = {
        family: [[] for _ in sigma_values] for family in ("C", "H")
    }
    fit_count = {"C": 0, "H": 0}
    sigma_score_count = {"C": 0, "H": 0}
    for fold in folds:
        training = development[fold.training_positions]
        validation = development[fold.validation_positions]
        fold_models, _ = fit_models(
            layout,
            training,
            outcomes[fold.training_positions],
            noise_protocol_id="crossfit_fold_provisional_then_common_grid",
        )
        validation_outcomes = outcomes[fold.validation_positions]
        for family in ("C", "H"):
            fit_count[family] += 1
            for sigma_index, sigma in enumerate(sigma_values):
                candidate = replace(fold_models[family], sigma=sigma)
                score, _ = score_model(
                    candidate,
                    design_for(candidate, layout, validation),
                    validation_outcomes,
                )
                # Retain one value per held-out unit so large units cannot
                # dominate the common noise selection.
                validation_units = layout.units[validation]
                for unit in fold.validation_units:
                    selected = validation_units == unit
                    per_family_per_sigma[family][sigma_index].append(
                        float(score[selected].mean())
                    )
                sigma_score_count[family] += 1

    family_scores: dict[str, tuple[float, ...]] = {}
    for family in ("C", "H"):
        family_scores[family] = tuple(
            float(np.mean(per_family_per_sigma[family][sigma_index]))
            for sigma_index in range(len(sigma_values))
        )
        if any(
            len(per_family_per_sigma[family][sigma_index]) != N_DEVELOPMENT_UNITS
            for sigma_index in range(len(sigma_values))
        ):
            raise AssertionError("Cross-fit score does not contain one mean per development unit")
    selection_scores = family_scores["C"]
    selected_index = int(np.argmax(np.asarray(selection_scores)))
    selected_sigma = sigma_values[selected_index]

    manifest_payload = "|".join(
        f"{fold.fold_index}:{','.join(fold.training_units)}>{','.join(fold.validation_units)}"
        for fold in folds
    )
    manifest_fingerprint = hashlib.sha256(manifest_payload.encode("utf-8")).hexdigest()
    overlap_count = sum(
        len(set(fold.training_units) & set(fold.validation_units)) for fold in folds
    )
    validation_coverage = len(
        set().union(*(set(fold.validation_units) for fold in folds))
    )
    diagnostics = CrossFitDiagnostics(
        protocol_id="development_brain_unit_heldout_4fold_C_reference_sigma_frozen_into_H_v1",
        folds=folds,
        sigma_grid=sigma_values,
        selected_sigma=selected_sigma,
        family_unit_mean_scores=family_scores,
        selection_reference_family="C",
        selection_objective_unit_mean_scores=selection_scores,
        selected_family_unit_mean_scores={
            family: family_scores[family][selected_index] for family in ("C", "H")
        },
        fit_count_by_family=fit_count,
        sigma_score_count_by_family=sigma_score_count,
        unit_coverage_count=validation_coverage,
        train_validation_overlap_count=overlap_count,
        fold_manifest_fingerprint=manifest_fingerprint,
        unseen_line_rule="zero_deviation_from_population_intercept_when_training_fold_lacks_line",
    )
    if diagnostics.train_validation_overlap_count != 0:
        raise AssertionError("Cross-fit metadata reports train/validation leakage")
    if diagnostics.unit_coverage_count != N_DEVELOPMENT_UNITS:
        raise AssertionError("Cross-fit metadata reports incomplete unit coverage")
    if diagnostics.fit_count_by_family["C"] != diagnostics.fit_count_by_family["H"]:
        raise AssertionError("C/H received unequal fold-fit budgets")
    if diagnostics.sigma_score_count_by_family["C"] != diagnostics.sigma_score_count_by_family["H"]:
        raise AssertionError("C/H received unequal noise-grid evaluation budgets")

    final_models, final_fit = fit_models(
        layout,
        development,
        outcomes,
        shared_sigma_override=selected_sigma,
        noise_protocol_id=diagnostics.protocol_id,
    )
    assert_factorized_model_set(final_models, layout.b_design.shape[1])
    return final_models, final_fit, diagnostics


def truth_directions() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index = np.arange(N_COORDINATES, dtype=float)
    continuous_value = np.vstack(
        [
            np.sin(2.0 * math.pi * (index + 0.5) / N_COORDINATES),
            np.cos(4.0 * math.pi * (index + 0.5) / N_COORDINATES),
            np.sin(6.0 * math.pi * (index + 0.5) / N_COORDINATES + 0.4),
        ]
    )
    continuous_support = np.vstack(
        [
            np.cos(2.0 * math.pi * (index + 0.5) / N_COORDINATES),
            np.sin(4.0 * math.pi * (index + 0.5) / N_COORDINATES),
            np.cos(6.0 * math.pi * (index + 0.5) / N_COORDINATES + 0.2),
        ]
    )
    group_value = np.where((index.astype(int) // 4) % 2 == 0, 1.0, -1.0)
    group_value += 0.25 * np.sin(index * 0.7)
    group_support = np.where((index.astype(int) // 7) % 2 == 0, 1.0, -1.0)
    return continuous_value, continuous_support, group_value, group_support


def standardized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = float(values.std())
    return (values - values.mean()) / (scale if scale > 1.0e-8 else 1.0)


def continuous_truth_features(
    layout: MetadataLayout,
    indices: np.ndarray,
    scenario: str,
    rng: np.random.Generator,
) -> np.ndarray:
    # Define the synthetic population once on the complete metadata layout.
    # Standardizing again inside a role subset would make the DGP depend on
    # whether calibration generated 20 assigned units or a design diagnostic
    # generated all 36 units before subsetting.
    pc = layout.pc
    population_base = standardized(
        0.78 * pc[:, 0] + 0.34 * pc[:, 1] - 0.18 * pc[:, 2]
    )
    base = population_base[indices]
    if scenario == "smooth_spatial_continuum":
        latent = base
        return np.column_stack([latent, np.zeros(len(base)), np.zeros(len(base))])
    if scenario == "curved_horseshoe_continuum":
        latent = base
        return np.column_stack(
            [latent, (latent**2 - 1.0) / math.sqrt(2.0), np.sin(1.35 * latent)]
        )
    if scenario == "nonuniform_continuum":
        latent = standardized(
            np.sign(population_base) * np.abs(population_base) ** 1.7
        )[indices]
        return np.column_stack([latent, (latent**2 - 1.0) / math.sqrt(2.0), np.zeros(len(base))])
    if scenario == "depleted_middle_continuum":
        # A continuous latent path is sampled on two separated intervals; no
        # discrete biological state is introduced.
        latent = standardized(
            population_base
            + np.where(population_base >= 0.0, 0.95, -0.95)
        )[indices]
        return np.column_stack([latent, np.sin(latent), np.zeros(len(base))])
    if scenario == "animal_confounded_continuum":
        effects: dict[str, float] = {}
        for unit in sorted(set(layout.units.tolist())):
            effects[unit] = float(rng.normal(0.0, 0.75))
        population_effect = np.array([effects[unit] for unit in layout.units])
        latent = standardized(population_base + population_effect)[indices]
        return np.column_stack([latent, (latent**2 - 1.0) / math.sqrt(2.0), np.zeros(len(base))])
    if scenario == "reusable_groups":
        return np.column_stack([0.20 * base, np.zeros(len(base)), np.zeros(len(base))])
    if scenario == "groups_plus_continuum":
        return np.column_stack([base, (base**2 - 1.0) / math.sqrt(2.0), np.sin(base)])
    if scenario == "spatially_interdigitated_groups":
        return np.column_stack([0.40 * base, np.zeros(len(base)), np.zeros(len(base))])
    raise ValueError(f"Unknown scenario: {scenario}")


def synthetic_groups(
    layout: MetadataLayout,
    indices: np.ndarray,
    scenario: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if scenario not in GROUP_SCENARIOS:
        labels = np.zeros(len(indices), dtype=int)
        probabilities = np.column_stack([np.ones(len(indices)), np.zeros(len(indices))])
        return labels, probabilities
    base = standardized(layout.pc[:, 0])[indices]
    if scenario == "spatially_interdigitated_groups":
        labels = np.zeros(len(indices), dtype=int)
        units = layout.units[indices]
        phase_by_unit = {
            unit: int(rng.integers(0, 2))
            for unit in sorted(set(layout.units.tolist()))
        }
        for unit in sorted(set(units.tolist())):
            local = np.flatnonzero(units == unit)
            order = local[np.argsort(base[local], kind="mergesort")]
            phase = phase_by_unit[unit]
            labels[order] = (np.arange(len(order)) + phase) % 2
        # The unit-specific phase is unobserved for a new unit.  Marginally,
        # either alternating label has probability 1/2 at every position.
        probability_one = np.full(len(indices), 0.5)
    else:
        probability_one = expit(0.30 * base)
        # Draw the full population vector before subsetting so a cell's planted
        # label is invariant to the role/index scope supplied by the caller.
        labels = (rng.random(len(layout.frame))[indices] < probability_one).astype(int)
    probabilities = np.column_stack([1.0 - probability_one, probability_one])
    return labels, probabilities


def fixed_line_effect(lines: np.ndarray) -> np.ndarray:
    output = np.empty((len(lines), N_COORDINATES), dtype=float)
    for i, line in enumerate(lines):
        local_rng = np.random.default_rng(stable_u64("line", line) % (2**32))
        output[i] = local_rng.normal(0.0, 0.08, size=N_COORDINATES)
    return output


def generate_synthetic_outcomes(
    layout: MetadataLayout,
    indices: np.ndarray,
    scenario: str,
    group_effect: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, SyntheticTruth]:
    """Generate only role-assigned outcomes; unassigned units stay untouched."""
    continuous_value, continuous_support, group_value, group_support = truth_directions()
    features = continuous_truth_features(layout, indices, scenario, rng)
    groups, group_probabilities = synthetic_groups(layout, indices, scenario, rng)
    signed_group = 2.0 * groups - 1.0
    if scenario not in GROUP_SCENARIOS:
        signed_group[:] = 0.0
        group_effect = 0.0

    coordinate = np.arange(N_COORDINATES, dtype=float)
    support_intercept = -1.75 + 0.50 * np.cos(2.0 * math.pi * coordinate / N_COORDINATES)
    value_intercept = 0.25 * np.sin(2.0 * math.pi * coordinate / N_COORDINATES + 0.3)
    line_effect = fixed_line_effect(layout.lines[indices])
    position = layout.coords[indices]
    position_loading = np.vstack(
        [
            0.16 * np.sin(coordinate * 0.21),
            0.13 * np.cos(coordinate * 0.17),
            0.11 * np.sin(coordinate * 0.11 + 0.5),
        ]
    )
    base_support_logits = (
        support_intercept[None, :]
        + 0.32 * features @ continuous_support
        + 0.20 * position @ position_loading
        + 0.50 * line_effect
    )
    base_value_mean = (
        value_intercept[None, :]
        + 0.46 * features @ continuous_value
        + 0.25 * position @ position_loading
        + line_effect
    )
    group_support_shift = np.vstack(
        [
            -(0.58 * group_effect) * group_support,
            +(0.58 * group_effect) * group_support,
        ]
    )
    group_value_shift = np.vstack(
        [
            -(0.72 * group_effect) * group_value,
            +(0.72 * group_effect) * group_value,
        ]
    )
    support_logits = base_support_logits + group_support_shift[groups]
    value_mean = base_value_mean + group_value_shift[groups]

    probability = expit(support_logits)
    active = rng.random(probability.shape) < probability
    empty = np.flatnonzero(~active.any(axis=1))
    while len(empty):
        active[empty] = rng.random((len(empty), N_COORDINATES)) < probability[empty]
        empty = empty[~active[empty].any(axis=1)]

    noisy_log_intensity = value_mean + rng.normal(
        0.0, GENERATOR_NOISE_SIGMA, size=value_mean.shape
    )
    outcomes = np.zeros_like(noisy_log_intensity)
    for i in range(len(outcomes)):
        on = active[i]
        values = noisy_log_intensity[i, on]
        values -= values.max()
        values = np.exp(values)
        outcomes[i, on] = values / values.sum()
    if not np.allclose(outcomes.sum(axis=1), 1.0, atol=1.0e-12):
        raise AssertionError("Synthetic compositions do not sum to one")
    if not np.any(outcomes == 0.0):
        raise AssertionError("Synthetic generator did not preserve exact zeros")
    truth = SyntheticTruth(
        groups=groups,
        group_probabilities=group_probabilities,
        base_support_logits=base_support_logits,
        base_value_mean=base_value_mean,
        group_support_shift=group_support_shift,
        group_value_shift=group_value_shift,
        group_support_direction=group_support.copy(),
        group_value_direction=group_value.copy(),
        noise_sigma=GENERATOR_NOISE_SIGMA,
    )
    return outcomes, truth


def subset_synthetic_truth(
    truth: SyntheticTruth,
    positions: np.ndarray | slice,
) -> SyntheticTruth:
    return SyntheticTruth(
        groups=truth.groups[positions],
        group_probabilities=truth.group_probabilities[positions],
        base_support_logits=truth.base_support_logits[positions],
        base_value_mean=truth.base_value_mean[positions],
        group_support_shift=truth.group_support_shift,
        group_value_shift=truth.group_value_shift,
        group_support_direction=truth.group_support_direction,
        group_value_direction=truth.group_value_direction,
        noise_sigma=truth.noise_sigma,
    )


def oracle_mixed_support_scores(
    outcomes: np.ndarray,
    truth: SyntheticTruth,
    include_groups: bool,
) -> np.ndarray:
    """Score outcomes under the known population-marginal synthetic density.

    Labels are integrated using their generator probabilities; truth labels
    are never conditioned on.  The continuous and positional generator means
    are held fixed in both densities, so H-oracle minus C-oracle isolates only
    the finite group factor on the same mixed-support measure as the fitted
    evaluator.
    """
    outcomes = np.asarray(outcomes, dtype=float)
    if len(outcomes) != len(truth.groups):
        raise ValueError("Oracle truth and outcomes are misaligned")
    faces = [make_face_data(row) for row in outcomes]
    variance = float(truth.noise_sigma**2)
    scores = np.empty(len(outcomes), dtype=float)
    states = range(K_GROUPS) if include_groups else range(1)
    for i, face in enumerate(faces):
        mask = np.zeros(N_COORDINATES, dtype=bool)
        mask[face.active] = True
        dimension = len(face.active) - 1
        normal_constant = dimension * math.log(2.0 * math.pi * variance)
        components: list[float] = []
        for state in states:
            if include_groups:
                weight = float(truth.group_probabilities[i, state])
                support_shift = truth.group_support_shift[state]
                value_shift = truth.group_value_shift[state]
            else:
                weight = 1.0
                support_shift = np.zeros(N_COORDINATES)
                value_shift = np.zeros(N_COORDINATES)
            if weight <= 0.0:
                components.append(-math.inf)
                continue
            support_logits = truth.base_support_logits[i] + support_shift
            support_logp = active_set_log_probability(mask, support_logits)
            mean_active = truth.base_value_mean[i, face.active] + value_shift[face.active]
            mean_ilr = face.helmert.T @ mean_active
            residual = face.ilr - mean_ilr
            value_logp = -0.5 * (normal_constant + float(residual @ residual) / variance)
            value_logp += face.log_jacobian
            components.append(math.log(weight) + support_logp + value_logp)
        scores[i] = float(logsumexp(np.asarray(components)))
    return scores


def projected_continuous_oracle_scores(
    outcomes: np.ndarray,
    truth: SyntheticTruth,
    parameters: Sequence[float],
) -> np.ndarray:
    """Score a generator-aligned projection onto the registered C geometry.

    The four parameters multiply the linear/quadratic connected-latent basis
    along the known support and value directions.  This contains the exact
    no-group generator at zero, but unlike the old no-group-base contrast it
    lets C absorb a multi-peaked grouped marginal through its connected
    seven-node quadrature.  Because the synthetic group perturbation itself is
    rank one in each outcome part, the population-optimal perturbation lies in
    these known directions; fitting arbitrary orthogonal directions would add
    Monte Carlo noise without enlarging the relevant population projection.
    """
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (4,) or not np.isfinite(parameters).all():
        raise ValueError("Projected C requires four finite parameters")
    outcomes = np.asarray(outcomes, dtype=float)
    if len(outcomes) != len(truth.groups):
        raise ValueError("Projected C truth and outcomes are misaligned")
    nodes, weights = quadrature()
    basis = continuous_basis(nodes)
    support_scalars = basis @ parameters[:2]
    value_scalars = basis @ parameters[2:]
    support_shifts = support_scalars[:, None] * truth.group_support_direction[None, :]
    value_shifts = value_scalars[:, None] * truth.group_value_direction[None, :]
    log_weights = np.log(np.clip(weights, 1.0e-300, 1.0))
    variance = float(truth.noise_sigma**2)
    scores = np.empty(len(outcomes), dtype=float)
    for i, face in enumerate(make_face_data(row) for row in outcomes):
        mask = np.zeros(N_COORDINATES, dtype=bool)
        mask[face.active] = True
        dimension = len(face.active) - 1
        normal_constant = dimension * math.log(2.0 * math.pi * variance)
        components = np.empty(len(weights), dtype=float)
        for state in range(len(weights)):
            support_logits = truth.base_support_logits[i] + support_shifts[state]
            support_logp = active_set_log_probability(mask, support_logits)
            mean_active = truth.base_value_mean[i, face.active] + value_shifts[
                state, face.active
            ]
            mean_ilr = face.helmert.T @ mean_active
            residual = face.ilr - mean_ilr
            value_logp = -0.5 * (
                normal_constant + float(residual @ residual) / variance
            )
            value_logp += face.log_jacobian
            components[state] = log_weights[state] + support_logp + value_logp
        scores[i] = float(logsumexp(components))
    return scores


def fit_projected_continuous_oracle(
    layout: MetadataLayout,
    samples: Sequence[OracleSample],
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit the population-projection C parameters on synthetic calibration data."""
    if not samples:
        raise ValueError("Projected C calibration needs at least one synthetic sample")

    def objective(parameters: np.ndarray) -> float:
        replicate_scores = []
        for sample in samples:
            scores = projected_continuous_oracle_scores(
                sample.outcomes, sample.truth, parameters
            )
            replicate_scores.append(equal_unit_mean(layout, sample.indices, scores))
        return -float(np.mean(replicate_scores))

    starts = (
        np.zeros(4),
        np.array([0.25, 0.0, 0.25, 0.0]),
        np.array([-0.25, 0.0, -0.25, 0.0]),
    )
    results = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=[(-3.0, 3.0)] * 4,
            options={"maxiter": 80, "ftol": 1.0e-9, "maxls": 20},
        )
        for start in starts
    ]
    best = min(results, key=lambda result: float(result.fun))
    if not np.isfinite(best.fun) or not np.isfinite(best.x).all():
        raise RuntimeError("Projected C oracle optimization returned nonfinite values")
    if not bool(best.success):
        raise RuntimeError(
            f"Projected C oracle optimization did not converge: {best.message}"
        )
    diagnostics = {
        "success": bool(best.success),
        "message": str(best.message),
        "objective": float(best.fun),
        "iterations": int(getattr(best, "nit", -1)),
        "function_evaluations": int(getattr(best, "nfev", -1)),
        "starts": len(starts),
    }
    return np.asarray(best.x, dtype=float), diagnostics


def isotonic_non_decreasing(values: Sequence[float]) -> np.ndarray:
    """Equal-weight pooled-adjacent-violators fit used only for calibration."""
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 1 or len(raw) == 0 or not np.isfinite(raw).all():
        raise ValueError("Isotonic input must be a finite nonempty vector")
    blocks: list[list[float | int]] = []
    for index, value in enumerate(raw):
        blocks.append([float(value), 1.0, index, index + 1])
        while len(blocks) >= 2 and float(blocks[-2][0]) > float(blocks[-1][0]):
            right = blocks.pop()
            left = blocks.pop()
            weight = float(left[1]) + float(right[1])
            mean = (
                float(left[0]) * float(left[1])
                + float(right[0]) * float(right[1])
            ) / weight
            blocks.append([mean, weight, int(left[2]), int(right[3])])
    fitted = np.empty_like(raw)
    for mean, _, start, stop in blocks:
        fitted[int(start) : int(stop)] = float(mean)
    # The zero-amplitude nested densities are exactly equal.  Enforce that
    # known anchor and nonnegative KL scale after finite-Monte-Carlo smoothing.
    fitted = np.maximum.accumulate(np.maximum(fitted, 0.0))
    fitted[0] = 0.0
    return fitted


def calibrate_oracle_group_effects(
    layout: MetadataLayout,
    scenarios: Sequence[str],
    target_margin_multipliers: Sequence[float],
    margin: float,
    seed: int,
    replicates: int,
    amplitude_grid: Sequence[float] = ORACLE_CALIBRATION_AMPLITUDES,
) -> OracleCalibration:
    """Map amplitude to a projected population-marginal oracle H-C proxy.

    For each amplitude, one synthetic sample fits the population-projection C
    approximation and an independent sample estimates its expected gap from
    the exact label-integrated H generator.  All samples use one fixed,
    outcome-blind provisional 12/8/16 role split.  This avoids averaging over
    favorable splits, but the result remains provisional until that split is
    replaced by the final geometry-eligible role lock.
    """
    selected_scenarios = tuple(sorted(set(scenarios)))
    if any(scenario not in GROUP_SCENARIOS for scenario in selected_scenarios):
        raise ValueError("Oracle group calibration accepts grouped scenarios only")
    multipliers = tuple(float(value) for value in target_margin_multipliers)
    if not multipliers or any(not np.isfinite(value) or value < 0.0 for value in multipliers):
        raise ValueError("Target margin multipliers must be finite and nonnegative")
    amplitudes = tuple(float(value) for value in amplitude_grid)
    if not amplitudes or amplitudes[0] != 0.0:
        raise ValueError("Oracle calibration amplitude grid must start at zero")
    if tuple(sorted(set(amplitudes))) != amplitudes or any(value < 0.0 for value in amplitudes):
        raise ValueError("Oracle calibration amplitude grid must be unique and increasing")
    if replicates < 1:
        raise ValueError("Oracle calibration requires at least one replicate")

    fixed_roles = role_split(
        layout,
        np.random.default_rng(stable_u64(seed, "oracle_fixed_roles") % (2**32)),
    )
    fixed_manifest = {
        "development": tuple(sorted(set(layout.units[fixed_roles.development].tolist()))),
        "audit": tuple(sorted(set(layout.units[fixed_roles.audit].tolist()))),
        "unassigned": tuple(sorted(set(layout.units[fixed_roles.unassigned].tolist()))),
    }
    manifest_payload = "|".join(
        f"{role}:{','.join(units)}" for role, units in sorted(fixed_manifest.items())
    )
    manifest_fingerprint = hashlib.sha256(manifest_payload.encode("utf-8")).hexdigest()

    def make_samples(namespace: str, scenario: str, amplitude: float) -> list[OracleSample]:
        assigned = np.concatenate([fixed_roles.development, fixed_roles.audit])
        n_development = len(fixed_roles.development)
        samples: list[OracleSample] = []
        for replicate in range(replicates):
            # The same random numbers are reused along each amplitude curve,
            # while fit and verification namespaces remain independent.
            outcome_rng = np.random.default_rng(
                stable_u64(seed, namespace, scenario, replicate) % (2**32)
            )
            assigned_outcomes, assigned_truth = generate_synthetic_outcomes(
                layout, assigned, scenario, amplitude, outcome_rng
            )
            samples.append(
                OracleSample(
                    indices=fixed_roles.audit,
                    outcomes=assigned_outcomes[n_development:],
                    truth=subset_synthetic_truth(
                        assigned_truth, slice(n_development, None)
                    ),
                )
            )
        return samples

    def fit_and_verify(
        scenario: str,
        amplitude: float,
        fit_namespace: str,
        verification_namespace: str,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        fit_samples = make_samples(fit_namespace, scenario, amplitude)
        if amplitude == 0.0:
            parameters = np.zeros(4)
            optimizer = {
                "success": True,
                "message": "exact_zero_amplitude_nested_solution",
                "objective": None,
                "iterations": 0,
                "function_evaluations": 0,
                "starts": 0,
            }
        else:
            parameters, optimizer = fit_projected_continuous_oracle(
                layout, fit_samples
            )
        gains = []
        for sample in make_samples(verification_namespace, scenario, amplitude):
            oracle_h = oracle_mixed_support_scores(
                sample.outcomes, sample.truth, include_groups=True
            )
            projected_c = projected_continuous_oracle_scores(
                sample.outcomes, sample.truth, parameters
            )
            gains.append(
                equal_unit_mean(layout, sample.indices, oracle_h - projected_c)
            )
        return parameters, np.asarray(gains, dtype=float), optimizer

    raw_by_scenario: dict[str, tuple[float, ...]] = {}
    se_by_scenario: dict[str, tuple[float, ...]] = {}
    monotone_by_scenario: dict[str, tuple[float, ...]] = {}
    diagnostics_by_scenario: dict[str, tuple[dict[str, object], ...]] = {}
    points: list[OracleCalibrationPoint] = []
    for scenario in selected_scenarios:
        amplitude_means: list[float] = []
        amplitude_ses: list[float] = []
        amplitude_diagnostics: list[dict[str, object]] = []
        for amplitude in amplitudes:
            _, replicate_gains, optimizer = fit_and_verify(
                scenario,
                amplitude,
                "oracle_curve_projection_fit",
                "oracle_curve_independent_verification",
            )
            mean_gain = float(np.mean(replicate_gains))
            se = (
                float(np.std(replicate_gains, ddof=1) / math.sqrt(replicates))
                if replicates > 1
                else 0.0
            )
            if amplitude == 0.0 and not np.isclose(mean_gain, 0.0, atol=2.0e-12):
                raise AssertionError("Zero-amplitude oracle H and C are not nested")
            amplitude_means.append(mean_gain)
            amplitude_ses.append(se)
            amplitude_diagnostics.append(optimizer)

        raw = np.asarray(amplitude_means, dtype=float)
        monotone = isotonic_non_decreasing(raw)
        raw_by_scenario[scenario] = tuple(float(value) for value in raw)
        se_by_scenario[scenario] = tuple(float(value) for value in amplitude_ses)
        monotone_by_scenario[scenario] = tuple(float(value) for value in monotone)
        diagnostics_by_scenario[scenario] = tuple(amplitude_diagnostics)
        for multiplier in multipliers:
            target = float(multiplier * margin)
            if target > float(monotone[-1]) + 1.0e-12:
                raise ValueError(
                    f"Oracle calibration for {scenario} does not bracket target gain {target:.6g}; "
                    f"maximum calibrated gain is {monotone[-1]:.6g}"
                )
            upper = int(np.searchsorted(monotone, target, side="left"))
            if upper == 0:
                lower = 0
                calibrated_amplitude = amplitudes[0]
            else:
                lower = upper - 1
                lower_gain = float(monotone[lower])
                upper_gain = float(monotone[upper])
                if upper_gain <= lower_gain + 1.0e-15:
                    calibrated_amplitude = amplitudes[upper]
                else:
                    fraction = (target - lower_gain) / (upper_gain - lower_gain)
                    calibrated_amplitude = amplitudes[lower] + fraction * (
                        amplitudes[upper] - amplitudes[lower]
                    )
            estimated = float(
                np.interp(calibrated_amplitude, np.asarray(amplitudes), monotone)
            )
            parameters, verification_gains, _ = fit_and_verify(
                scenario,
                float(calibrated_amplitude),
                "oracle_target_projection_fit",
                "oracle_target_independent_verification",
            )
            verification_mean = float(np.mean(verification_gains))
            verification_se = (
                float(np.std(verification_gains, ddof=1) / math.sqrt(replicates))
                if replicates > 1
                else 0.0
            )
            critical = (
                float(student_t.ppf(0.975, replicates - 1))
                if replicates > 1
                else 0.0
            )
            points.append(
                OracleCalibrationPoint(
                    scenario=scenario,
                    target_margin_multiplier=multiplier,
                    target_gain_nats_per_neuron=target,
                    generator_amplitude=float(calibrated_amplitude),
                    estimated_gain_nats_per_neuron=estimated,
                    lower_amplitude=float(amplitudes[lower]),
                    upper_amplitude=float(amplitudes[upper]),
                    lower_gain_nats_per_neuron=float(monotone[lower]),
                    upper_gain_nats_per_neuron=float(monotone[upper]),
                    independent_verification_mean_gain=verification_mean,
                    independent_verification_se=verification_se,
                    independent_verification_lower95=verification_mean
                    - critical * verification_se,
                    independent_verification_upper95=verification_mean
                    + critical * verification_se,
                    projected_C_parameters=tuple(float(value) for value in parameters),
                )
            )

    return OracleCalibration(
        protocol_id="projected_oracle_proxy_label_integrated_H_vs_direction_aligned_connected_C_v2",
        margin_nats_per_neuron=float(margin),
        replicates=int(replicates),
        amplitude_grid=amplitudes,
        scenarios=selected_scenarios,
        raw_mean_gain_by_scenario=raw_by_scenario,
        raw_se_by_scenario=se_by_scenario,
        monotone_mean_gain_by_scenario=monotone_by_scenario,
        points=tuple(points),
        projection_family="generator_direction_aligned_registered_1D_quadratic_7node_C",
        fixed_role_manifest=fixed_manifest,
        fixed_role_manifest_fingerprint=manifest_fingerprint,
        fixed_role_is_final_geometry_locked=False,
        projection_diagnostics_by_scenario=diagnostics_by_scenario,
    )


def paired_unit_summary(
    layout: MetadataLayout,
    audit: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, float]:
    differences = []
    audit_units = layout.units[audit]
    for unit in sorted(set(audit_units.tolist())):
        selected = audit_units == unit
        differences.append(float(np.mean(left[selected] - right[selected])))
    values = np.asarray(differences)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else math.inf
    critical = float(student_t.ppf(0.95, len(values) - 1)) if len(values) > 1 else math.inf
    return {
        "mean": mean,
        "se": se,
        "lower90": mean - critical * se,
        "upper90": mean + critical * se,
        "unit_count": float(len(values)),
    }


def group_reuse_check(
    layout: MetadataLayout,
    roles: RoleSplit,
    diagnostics: FitDiagnostics,
    d_components: np.ndarray,
    model_d: MixedSupportModel,
) -> dict[str, object]:
    development_labels = diagnostics.development_group_labels
    audit_labels = np.argmax(d_components, axis=1)
    audit_responsibilities = np.exp(
        d_components - logsumexp(d_components, axis=1, keepdims=True)
    )
    development_responsibilities = np.eye(K_GROUPS)[development_labels]

    def represented_units(indices: np.ndarray, labels: np.ndarray) -> list[int]:
        return [
            len(set(layout.units[indices][labels == group].tolist()))
            for group in range(K_GROUPS)
        ]

    development_units = represented_units(roles.development, development_labels)
    audit_units = represented_units(roles.audit, audit_labels)

    def effective_membership(
        indices: np.ndarray, responsibilities: np.ndarray
    ) -> dict[str, list[float]]:
        units = layout.units[indices]
        cell_ess = []
        animal_ess = []
        prevalence = []
        maximum_animal_fraction = []
        for group in range(K_GROUPS):
            values = responsibilities[:, group]
            cell_ess.append(float(values.sum() ** 2 / max(float(values @ values), 1.0e-12)))
            per_animal = np.array(
                [float(values[units == unit].mean()) for unit in sorted(set(units.tolist()))]
            )
            animal_ess.append(
                float(per_animal.sum() ** 2 / max(float(per_animal @ per_animal), 1.0e-12))
            )
            prevalence.append(float(per_animal.mean()))
            maximum_animal_fraction.append(
                float(per_animal.max() / max(float(per_animal.sum()), 1.0e-12))
            )
        return {
            "cell_ESS": cell_ess,
            "animal_ESS": animal_ess,
            "equal_animal_prevalence": prevalence,
            "maximum_single_animal_membership_fraction": maximum_animal_fraction,
        }

    development_effective = effective_membership(
        roles.development, development_responsibilities
    )
    audit_effective = effective_membership(roles.audit, audit_responsibilities)
    signature = float(
        np.sqrt(
            np.mean((model_d.group_value_signature[1] - model_d.group_value_signature[0]) ** 2)
            + 0.25
            * np.mean((model_d.group_support_signature[1] - model_d.group_support_signature[0]) ** 2)
        )
    )
    subset_screen_passed = (
        min(development_units) >= 4
        and min(audit_units) >= 3
        and min(development_effective["equal_animal_prevalence"]) >= 0.15
        and min(audit_effective["equal_animal_prevalence"]) >= 0.15
        and min(development_effective["cell_ESS"]) >= 20.0
        and min(development_effective["animal_ESS"]) >= 4.0
        and max(development_effective["maximum_single_animal_membership_fraction"]) <= 0.35
        and min(audit_effective["cell_ESS"]) >= 10.0
        and min(audit_effective["animal_ESS"]) >= 3.0
        and max(audit_effective["maximum_single_animal_membership_fraction"]) <= 0.50
        and signature >= 0.10
    )
    return {
        # This smoke evaluator lacks position-standardized membership, local
        # overlap, technical-decoding, and influence gates.  It therefore
        # cannot assert the full reusable-group policy even when the numeric
        # subset screen passes.
        "passed": False,
        "subset_screen_passed": bool(subset_screen_passed),
        "full_policy_reuse_gate_evaluated": False,
        "development_units_per_fixed_label": development_units,
        "audit_units_per_fixed_label": audit_units,
        "development_effective_membership": development_effective,
        "audit_effective_membership": audit_effective,
        "signature_rms": signature,
        "audit_operation": "fixed_definition_posterior_assignment_without_refit_or_reclustering",
        "missing_full_policy_elements": [
            "position_standardized_membership",
            "local_soma_overlap",
            "technical_decoding",
            "animal_and_territory_influence",
            "standardized_target_signature_similarity",
        ],
    }


def decide_route(
    comparison_valid: bool,
    hc: dict[str, float],
    hd: dict[str, float],
    reuse_passed: bool,
    continuous_adequacy_passed: bool,
    margin: float,
) -> str:
    """Decision path in which continuous validity never depends on group gates."""
    if not comparison_valid:
        return "invalid_comparison"
    if hc["lower90"] > margin and reuse_passed:
        if hd["lower90"] > margin:
            return "reusable_groups_with_within_group_continuum"
        if hd["upper90"] < margin:
            return "reusable_groups_without_supported_within_group_continuum"
        return "reusable_groups_with_internal_organization_unresolved"
    if hc["upper90"] < margin and continuous_adequacy_passed:
        return "continuous_description_no_meaningful_reusable_group_gain"
    if hc["upper90"] < margin:
        return "no_meaningful_group_gain_continuous_adequacy_not_established"
    return "valid_comparison_unresolved"


def evaluate_replicate(
    layout: MetadataLayout,
    condition: ConditionSpec,
    replicate: int,
    seed: int,
    margin: float,
) -> dict[str, object]:
    scenario = condition.scenario
    effect = condition.generator_amplitude
    role_rng = np.random.default_rng(stable_u64(seed, "roles", replicate) % (2**32))
    roles = role_split(layout, role_rng)
    assigned = np.concatenate([roles.development, roles.audit])
    outcome_rng = np.random.default_rng(
        stable_u64(seed, "outcome", scenario, effect, replicate) % (2**32)
    )
    assigned_outcomes, truth = generate_synthetic_outcomes(
        layout, assigned, scenario, effect, outcome_rng
    )
    n_development = len(roles.development)
    development_outcomes = assigned_outcomes[:n_development]
    audit_outcomes = assigned_outcomes[n_development:]
    models, diagnostics, crossfit = fit_models_with_unit_crossfit(
        layout,
        roles.development,
        development_outcomes,
        fold_seed=stable_u64(seed, "development_crossfit", replicate) % (2**32),
    )

    scores: dict[str, np.ndarray] = {}
    d_components: np.ndarray | None = None
    for name, model in models.items():
        values, components = score_model(
            model,
            design_for(model, layout, roles.audit),
            audit_outcomes,
            return_components=name == "D",
        )
        scores[name] = values
        if name == "D":
            d_components = components
    assert d_components is not None
    finite = all(np.isfinite(value).all() for value in scores.values())
    hc = paired_unit_summary(layout, roles.audit, scores["H"], scores["C"])
    hd = paired_unit_summary(layout, roles.audit, scores["H"], scores["D"])
    cp = paired_unit_summary(layout, roles.audit, scores["C"], scores["P"])
    pb = paired_unit_summary(layout, roles.audit, scores["P"], scores["B"])
    reuse = group_reuse_check(layout, roles, diagnostics, d_components, models["D"])
    continuous_adequacy_passed = False
    route = decide_route(
        finite,
        hc,
        hd,
        bool(reuse["passed"]),
        continuous_adequacy_passed,
        margin,
    )

    assigned_units = len(set(layout.units[assigned].tolist()))
    if assigned_units != N_DEVELOPMENT_UNITS + N_AUDIT_UNITS:
        raise AssertionError("Assigned outcome generation did not use exactly 20 units")
    if len(set(layout.units[roles.unassigned].tolist())) != N_UNASSIGNED_UNITS:
        raise AssertionError("Unassigned role does not contain exactly 16 units")

    truth_audit = truth.groups[n_development:]
    inferred_audit = np.argmax(d_components, axis=1)
    direct = float(np.mean(inferred_audit == truth_audit))
    flipped = float(np.mean((1 - inferred_audit) == truth_audit))
    truth_accuracy = max(direct, flipped) if scenario in GROUP_SCENARIOS else math.nan
    fold_manifest = [
        {
            "fold_index": fold.fold_index,
            "training_units": list(fold.training_units),
            "validation_units": list(fold.validation_units),
            "training_cells": int(len(fold.training_positions)),
            "validation_cells": int(len(fold.validation_positions)),
        }
        for fold in crossfit.folds
    ]
    audit_unit_sd = float(hc["se"] * math.sqrt(N_AUDIT_UNITS))
    one_sided_critical = float(student_t.ppf(0.95, N_AUDIT_UNITS - 1))
    required_sd_for_halfwidth_below_margin = float(
        margin * math.sqrt(N_AUDIT_UNITS) / one_sided_critical
    )
    return {
        "scenario": scenario,
        "effect": float(effect),
        "generator_amplitude": float(effect),
        "target_margin_multiplier": condition.target_margin_multiplier,
        "target_oracle_H_minus_C": condition.target_oracle_gain_nats_per_neuron,
        "calibrated_oracle_H_minus_C": condition.calibrated_oracle_gain_nats_per_neuron,
        "replicate": int(replicate),
        "development_units": N_DEVELOPMENT_UNITS,
        "audit_units": N_AUDIT_UNITS,
        "unassigned_units": N_UNASSIGNED_UNITS,
        "development_cells": int(len(roles.development)),
        "audit_cells": int(len(roles.audit)),
        "unassigned_cells_not_simulated": int(len(roles.unassigned)),
        "comparison_valid": bool(finite),
        "continuous_route_eligible_without_group_gate": bool(finite),
        "subset_reuse_check": bool(reuse["subset_screen_passed"]),
        "full_policy_reuse_gate_evaluated": bool(
            reuse["full_policy_reuse_gate_evaluated"]
        ),
        "route": route,
        "H_minus_C": hc["mean"],
        "H_minus_C_lower90": hc["lower90"],
        "H_minus_C_upper90": hc["upper90"],
        "H_minus_D": hd["mean"],
        "H_minus_D_lower90": hd["lower90"],
        "H_minus_D_upper90": hd["upper90"],
        "C_minus_P": cp["mean"],
        "P_minus_B": pb["mean"],
        "fixed_group_truth_accuracy_diagnostic": truth_accuracy,
        "development_units_fixed_label_0": reuse["development_units_per_fixed_label"][0],
        "development_units_fixed_label_1": reuse["development_units_per_fixed_label"][1],
        "audit_units_fixed_label_0": reuse["audit_units_per_fixed_label"][0],
        "audit_units_fixed_label_1": reuse["audit_units_per_fixed_label"][1],
        "fixed_signature_rms": reuse["signature_rms"],
        "crossfit_protocol_id": crossfit.protocol_id,
        "crossfit_fold_manifest_fingerprint": crossfit.fold_manifest_fingerprint,
        "crossfit_fold_manifest_json": json.dumps(fold_manifest, sort_keys=True),
        "crossfit_train_validation_overlap_count": crossfit.train_validation_overlap_count,
        "crossfit_validation_unit_coverage": crossfit.unit_coverage_count,
        "crossfit_selected_sigma": crossfit.selected_sigma,
        "crossfit_sigma_grid_json": json.dumps(crossfit.sigma_grid),
        "crossfit_selection_reference_family": crossfit.selection_reference_family,
        "crossfit_C_fit_count": crossfit.fit_count_by_family["C"],
        "crossfit_H_fit_count": crossfit.fit_count_by_family["H"],
        "crossfit_C_sigma_score_count": crossfit.sigma_score_count_by_family["C"],
        "crossfit_H_sigma_score_count": crossfit.sigma_score_count_by_family["H"],
        "crossfit_unseen_line_rule": crossfit.unseen_line_rule,
        "continuous_absolute_adequacy_evaluated": continuous_adequacy_passed,
        "H_minus_C_audit_unit_sd": audit_unit_sd,
        "H_minus_C_one_sided95_halfwidth": one_sided_critical * hc["se"],
        "required_audit_unit_sd_for_halfwidth_below_margin": required_sd_for_halfwidth_below_margin,
        "precision_halfwidth_below_margin": bool(
            one_sided_critical * hc["se"] < margin
        ),
    }


def scenario_conditions(
    scenarios: Iterable[str],
    calibration: OracleCalibration,
) -> list[ConditionSpec]:
    conditions: list[ConditionSpec] = []
    point_lookup = {
        (point.scenario, point.target_margin_multiplier): point
        for point in calibration.points
    }
    for scenario in scenarios:
        if scenario in NO_GROUP_SCENARIOS:
            conditions.append(
                ConditionSpec(
                    scenario=scenario,
                    generator_amplitude=0.0,
                    target_margin_multiplier=0.0,
                    target_oracle_gain_nats_per_neuron=0.0,
                    calibrated_oracle_gain_nats_per_neuron=0.0,
                )
            )
        else:
            scenario_multipliers = sorted(
                point.target_margin_multiplier
                for point in calibration.points
                if point.scenario == scenario
            )
            for multiplier in scenario_multipliers:
                point = point_lookup[(scenario, float(multiplier))]
                conditions.append(
                    ConditionSpec(
                        scenario=scenario,
                        generator_amplitude=point.generator_amplitude,
                        target_margin_multiplier=point.target_margin_multiplier,
                        target_oracle_gain_nats_per_neuron=point.target_gain_nats_per_neuron,
                        calibrated_oracle_gain_nats_per_neuron=(
                            point.independent_verification_mean_gain
                        ),
                    )
                )
    return conditions


def summarize(rows: list[dict[str, object]], margin: float) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    conditions: list[dict[str, object]] = []
    for (scenario, multiplier), group in frame.groupby(
        ["scenario", "target_margin_multiplier"], sort=True
    ):
        routes = group["route"].astype(str)
        conditions.append(
            {
                "scenario": str(scenario),
                "target_margin_multiplier": float(multiplier),
                "target_oracle_H_minus_C": float(group["target_oracle_H_minus_C"].iloc[0]),
                "generator_amplitude": float(group["generator_amplitude"].iloc[0]),
                "independently_verified_oracle_H_minus_C": float(
                    group["calibrated_oracle_H_minus_C"].iloc[0]
                ),
                "replicates": int(len(group)),
                "comparison_valid_rate": float(group["comparison_valid"].mean()),
                "continuous_route_eligible_rate": float(
                    group["continuous_route_eligible_without_group_gate"].mean()
                ),
                "subset_reuse_check_rate": float(group["subset_reuse_check"].mean()),
                "full_policy_reuse_gate_evaluated": bool(
                    group["full_policy_reuse_gate_evaluated"].all()
                ),
                "reusable_group_route_rate": float(routes.str.startswith("reusable_groups").mean()),
                "predictive_group_gain_lower_bound_rate": float(
                    (group["H_minus_C_lower90"] > margin).mean()
                ),
                "continuous_conclusion_rate": float(
                    routes.eq("continuous_description_no_meaningful_reusable_group_gain").mean()
                ),
                "no_meaningful_group_gain_rate": float(
                    (group["H_minus_C_upper90"] < margin).mean()
                ),
                "unresolved_rate": float(
                    routes.isin(
                        [
                            "valid_comparison_unresolved",
                            "no_meaningful_group_gain_continuous_adequacy_not_established",
                        ]
                    ).mean()
                ),
                "mean_H_minus_C": float(group["H_minus_C"].mean()),
                "mean_H_minus_D": float(group["H_minus_D"].mean()),
                "mean_C_minus_P": float(group["C_minus_P"].mean()),
                "mean_H_minus_C_audit_unit_sd": float(
                    group["H_minus_C_audit_unit_sd"].mean()
                ),
                "precision_halfwidth_below_margin_rate": float(
                    group["precision_halfwidth_below_margin"].mean()
                ),
            }
        )

    no_group = frame[frame["scenario"].isin(NO_GROUP_SCENARIOS)]
    grouped = frame[frame["scenario"].isin(GROUP_SCENARIOS)]
    maximum_multiplier = (
        float(grouped["target_margin_multiplier"].max()) if len(grouped) else math.nan
    )
    strong = (
        grouped[np.isclose(grouped["target_margin_multiplier"], maximum_multiplier)]
        if len(grouped)
        else grouped
    )
    enough_replicates = bool(
        frame.groupby(["scenario", "target_margin_multiplier"]).size().min() >= 20
    )
    checks = {
        "all_comparisons_numerically_valid": bool(frame["comparison_valid"].mean() >= 0.95),
        "continuous_route_runs_without_group_gate": bool(
            frame["continuous_route_eligible_without_group_gate"].mean() >= 0.95
        ),
        "no_group_false_predictive_gain_at_most_0_10": bool(
            len(no_group) and (no_group["H_minus_C_lower90"] > margin).mean() <= 0.10
        ),
        "strong_predictive_group_gain_and_subset_screen_at_least_0_70": bool(
            len(strong)
            and (
                (strong["H_minus_C_lower90"] > margin)
                & strong["subset_reuse_check"].astype(bool)
            ).mean()
            >= 0.70
        ),
        "precision_halfwidth_below_margin_in_at_least_0_80_of_repeats": bool(
            frame["precision_halfwidth_below_margin"].mean() >= 0.80
        ),
    }
    if not enough_replicates:
        subset_status = "descriptive_smoke_only_too_few_replicates"
    else:
        subset_status = "pass" if all(checks.values()) else "revise"
    return {
        "schema_version": "ep11_mixed_support_56d_smoke_result_v2",
        "qualification_class": "synthetic_smoke_subset_not_registered_qualification",
        "full_registered_gate_status": "not_evaluated",
        "outcome_bearing_search_authorized": False,
        "subset_check_status": subset_status,
        "replicate_count": int(len(frame)),
        "margin_nats_per_neuron": float(margin),
        "precision_diagnostic": {
            "audit_units": N_AUDIT_UNITS,
            "one_sided_confidence_level": 0.95,
            "required_audit_unit_SD_for_halfwidth_below_0_05": float(
                margin
                * math.sqrt(N_AUDIT_UNITS)
                / student_t.ppf(0.95, N_AUDIT_UNITS - 1)
            ),
            "approximate_population_SD_for_80_percent_precision_probability": 0.063,
            "rule_is_not_weakened": True,
        },
        "fixed_design": {
            "coordinates": N_COORDINATES,
            "development_units": N_DEVELOPMENT_UNITS,
            "audit_units": N_AUDIT_UNITS,
            "unassigned_units": N_UNASSIGNED_UNITS,
            "audit_refit_or_reclustering": False,
            "role_assignment_is_final_geometry_locked": False,
            "development_crossfit_folds": DEVELOPMENT_CROSSFIT_FOLDS,
            "shared_sigma_grid": list(SHARED_SIGMA_GRID),
            "noise_selection_reference": "C_equal_animal_heldout_score_then_frozen_into_H",
            "cell_level_fit_weighting": "pooled_cells_current_subset",
            "cell_level_fit_weighting_limitation": (
                "model coefficients and residual PCA remain cell-weighted; "
                "fold tuning and reported prediction are equal-unit"
            ),
        },
        "crossfit_verification": {
            "protocol_ids": sorted(set(frame["crossfit_protocol_id"].astype(str))),
            "all_train_validation_overlap_counts_zero": bool(
                (frame["crossfit_train_validation_overlap_count"] == 0).all()
            ),
            "all_12_development_units_covered_once_for_validation": bool(
                (frame["crossfit_validation_unit_coverage"] == N_DEVELOPMENT_UNITS).all()
            ),
            "C_H_fold_fit_budget_equal": bool(
                (frame["crossfit_C_fit_count"] == frame["crossfit_H_fit_count"]).all()
            ),
            "C_H_sigma_score_budget_equal": bool(
                (
                    frame["crossfit_C_sigma_score_count"]
                    == frame["crossfit_H_sigma_score_count"]
                ).all()
            ),
            "selection_reference_family": sorted(
                set(frame["crossfit_selection_reference_family"].astype(str))
            ),
            "unseen_line_rules": sorted(set(frame["crossfit_unseen_line_rule"].astype(str))),
        },
        "full_policy_reuse_gate_evaluated": False,
        "continuous_absolute_adequacy_evaluated": False,
        "subset_checks": checks,
        "conditions": conditions,
        "remaining_blockers": list(REMAINING_BLOCKERS),
    }


def oracle_calibration_to_dict(calibration: OracleCalibration) -> dict[str, object]:
    return {
        "protocol_id": calibration.protocol_id,
        "margin_nats_per_neuron": calibration.margin_nats_per_neuron,
        "fit_and_independent_verification_replicates_each": calibration.replicates,
        "amplitude_grid": list(calibration.amplitude_grid),
        "scenarios": list(calibration.scenarios),
        "projection_family": calibration.projection_family,
        "raw_mean_gain_by_scenario": {
            key: list(value) for key, value in calibration.raw_mean_gain_by_scenario.items()
        },
        "raw_se_by_scenario": {
            key: list(value) for key, value in calibration.raw_se_by_scenario.items()
        },
        "monotone_mean_gain_by_scenario": {
            key: list(value)
            for key, value in calibration.monotone_mean_gain_by_scenario.items()
        },
        "points": [
            {
                "scenario": point.scenario,
                "target_margin_multiplier": point.target_margin_multiplier,
                "target_gain_nats_per_neuron": point.target_gain_nats_per_neuron,
                "generator_amplitude": point.generator_amplitude,
                "interpolated_gain_nats_per_neuron": point.estimated_gain_nats_per_neuron,
                "lower_amplitude": point.lower_amplitude,
                "upper_amplitude": point.upper_amplitude,
                "lower_gain_nats_per_neuron": point.lower_gain_nats_per_neuron,
                "upper_gain_nats_per_neuron": point.upper_gain_nats_per_neuron,
                "independent_verification_mean_gain": point.independent_verification_mean_gain,
                "independent_verification_se": point.independent_verification_se,
                "independent_verification_lower95": point.independent_verification_lower95,
                "independent_verification_upper95": point.independent_verification_upper95,
                "projected_C_parameters": list(point.projected_C_parameters),
            }
            for point in calibration.points
        ],
        "fixed_role_manifest": {
            key: list(value) for key, value in calibration.fixed_role_manifest.items()
        },
        "fixed_role_manifest_fingerprint": calibration.fixed_role_manifest_fingerprint,
        "fixed_role_is_final_geometry_locked": calibration.fixed_role_is_final_geometry_locked,
        "projection_diagnostics_by_scenario": {
            key: list(value)
            for key, value in calibration.projection_diagnostics_by_scenario.items()
        },
        "labels_integrated_not_conditioned": True,
        "calibration_status": (
            "projected_oracle_proxy_only_until_final_geometry_eligible_role_split_and_full_C_family_projection"
        ),
    }


def write_outputs(output_dir: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "mixed_support_56d_replicates.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "mixed_support_56d_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    lines = [
        "# EP11 56-coordinate mixed-support synthetic smoke subset",
        "",
        f"- Subset status: `{summary['subset_check_status']}`",
        "- Qualification class: `synthetic_smoke_subset_not_registered_qualification`",
        "- Full registered gate: `not_evaluated`",
        "- Outcome-bearing search authorized: **no**",
        "- Real projection outcomes read: **no**",
        "",
        "All B/P/C/D/H values are nats per neuron on the same measure: counting",
        "measure over nonempty support sets times face-specific Hausdorff measure.",
        "The continuous route is evaluated whenever the common comparison is valid;",
        "it does not require a group-separation or group-reuse gate. This subset does",
        "not call a continuous conclusion because absolute C/P calibration is not yet",
        "implemented, and it does not call a reusable-group conclusion because the full",
        "position-standardized reuse policy is not evaluated.",
        "",
        "## Condition summary",
        "",
        "| Scenario | Target margin multiple | Calibrated amplitude | n | Valid | Predictive group gain | No meaningful group gain | H-C | H-D |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["conditions"]:
        lines.append(
            f"| {row['scenario']} | {row['target_margin_multiplier']:.2f} | "
            f"{row['generator_amplitude']:.4f} | {row['replicates']} | "
            f"{row['comparison_valid_rate']:.3f} | "
            f"{row['predictive_group_gain_lower_bound_rate']:.3f} | "
            f"{row['no_meaningful_group_gain_rate']:.3f} | "
            f"{row['mean_H_minus_C']:.3f} | {row['mean_H_minus_D']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Remaining blockers",
            "",
            *[f"- {item}" for item in summary["remaining_blockers"]],
            "",
            "Passing this subset would validate the executable score direction and",
            "precision diagnostics only. It cannot substitute for the registered evaluator,",
            "complete null controllers, role lock, real-data eligibility, or independent-",
            "animal certification.",
            "",
        ]
    )
    (output_dir / "mixed_support_56d_report.md").write_text("\n".join(lines), encoding="utf-8")


def mathematical_self_test() -> None:
    rng = np.random.default_rng(9127)
    for size in range(1, 9):
        basis = ordered_helmert(size)
        if size > 1:
            np.testing.assert_allclose(basis.T @ basis, np.eye(size - 1), atol=1.0e-12)
            np.testing.assert_allclose(basis.T @ np.ones(size), 0.0, atol=1.0e-12)
        raw = rng.uniform(0.5, 2.0, size=size)
        values = raw / raw.sum()
        if size > 1:
            coordinates = values[:-1].copy()
            epsilon = 1.0e-7
            numeric = np.empty((size - 1, size - 1))
            for column in range(size - 1):
                plus = coordinates.copy()
                minus = coordinates.copy()
                plus[column] += epsilon
                minus[column] -= epsilon
                p_plus = np.append(plus, 1.0 - plus.sum())
                p_minus = np.append(minus, 1.0 - minus.sum())
                z_plus = basis.T @ np.log(p_plus)
                z_minus = basis.T @ np.log(p_minus)
                numeric[:, column] = (z_plus - z_minus) / (2.0 * epsilon)
            numerical_relative_hausdorff = abs(np.linalg.det(numeric)) / math.sqrt(size)
            analytic = math.exp(face_log_jacobian(values))
            np.testing.assert_allclose(numerical_relative_hausdorff, analytic, rtol=2.0e-6)
        else:
            assert face_log_jacobian(values) == 0.0

    # Exhaustively verify conditional-nonempty active-set normalization in a
    # small dimension, using the same formula as the 56-coordinate score.
    small_logits = np.array([-1.2, 0.3, 1.1, -0.4])
    log_on = -np.logaddexp(0.0, -small_logits)
    log_off = -np.logaddexp(0.0, small_logits)
    log_nonempty = log1mexp(float(log_off.sum()))
    total = 0.0
    for bits in range(1, 2 ** len(small_logits)):
        mask = np.array([(bits >> index) & 1 for index in range(len(small_logits))], dtype=bool)
        total += math.exp(float(np.where(mask, log_on, log_off).sum()) - log_nonempty)
    np.testing.assert_allclose(total, 1.0, atol=1.0e-12)

    dummy_hc = {"lower90": -0.10, "upper90": 0.01, "mean": -0.02}
    dummy_hd = {"lower90": -0.10, "upper90": 0.10, "mean": 0.00}
    assert decide_route(
        True,
        dummy_hc,
        dummy_hd,
        False,
        True,
        DEFAULT_MARGIN,
    ).startswith("continuous_")
    assert decide_route(
        True,
        dummy_hc,
        dummy_hd,
        False,
        False,
        DEFAULT_MARGIN,
    ) == "no_meaningful_group_gain_continuous_adequacy_not_established"


def executable_self_test(layout: MetadataLayout) -> dict[str, object]:
    mathematical_self_test()
    roles = role_split(layout, np.random.default_rng(771))
    assert len(set(layout.units[roles.development])) == N_DEVELOPMENT_UNITS
    assert len(set(layout.units[roles.audit])) == N_AUDIT_UNITS
    assert len(set(layout.units[roles.unassigned])) == N_UNASSIGNED_UNITS
    assigned = np.concatenate([roles.development, roles.audit])
    all_indices = np.arange(len(layout.frame), dtype=int)

    # Population truth must not change when a caller generates only the
    # assigned role subset. This is essential because calibration and design
    # diagnostics may use different index scopes.
    for scenario in ALL_SCENARIOS:
        full_features = continuous_truth_features(
            layout,
            all_indices,
            scenario,
            np.random.default_rng(stable_u64("subset_invariance_features", scenario) % (2**32)),
        )
        subset_features = continuous_truth_features(
            layout,
            assigned,
            scenario,
            np.random.default_rng(stable_u64("subset_invariance_features", scenario) % (2**32)),
        )
        np.testing.assert_array_equal(subset_features, full_features[assigned])
        full_groups, full_probabilities = synthetic_groups(
            layout,
            all_indices,
            scenario,
            np.random.default_rng(stable_u64("subset_invariance_groups", scenario) % (2**32)),
        )
        subset_groups, subset_probabilities = synthetic_groups(
            layout,
            assigned,
            scenario,
            np.random.default_rng(stable_u64("subset_invariance_groups", scenario) % (2**32)),
        )
        np.testing.assert_array_equal(subset_groups, full_groups[assigned])
        np.testing.assert_array_equal(subset_probabilities, full_probabilities[assigned])

    # Every generator is exercised, while only one model fit is needed to
    # validate the common B/P/C/D/H executable path.
    for scenario in ALL_SCENARIOS:
        local_rng = np.random.default_rng(stable_u64("selftest", scenario) % (2**32))
        effect = 1.25 if scenario in GROUP_SCENARIOS else 0.0
        outcomes, _ = generate_synthetic_outcomes(layout, assigned, scenario, effect, local_rng)
        assert outcomes.shape == (len(assigned), N_COORDINATES)
        assert np.any(outcomes == 0.0)

    outcomes, _ = generate_synthetic_outcomes(
        layout,
        assigned,
        "groups_plus_continuum",
        1.25,
        np.random.default_rng(3341),
    )
    n_development = len(roles.development)
    models, diagnostics, crossfit = fit_models_with_unit_crossfit(
        layout,
        roles.development,
        outcomes[:n_development],
        fold_seed=81173,
    )
    audit_outcomes = outcomes[n_development:]
    finite_by_model: dict[str, bool] = {}
    audit_scores: dict[str, np.ndarray] = {}
    for name, model in models.items():
        score, _ = score_model(model, design_for(model, layout, roles.audit), audit_outcomes)
        audit_scores[name] = score
        finite_by_model[name] = bool(np.isfinite(score).all())
    assert set(finite_by_model) == {"B", "P", "C", "D", "H"}
    assert all(finite_by_model.values())
    assert_factorized_model_set(models, layout.b_design.shape[1])

    # Numerical nesting check: when the H group amplitude is set to zero, its
    # K copies of the shared C backbone integrate back to C exactly (up to
    # floating-point log-sum-exp roundoff).  This catches accidental H-only
    # latent fitting, prevalence recalibration, or unmatched noise parameters.
    model_c = models["C"]
    model_h = models["H"]
    assert model_h.group_weights is not None
    zero_group_support = np.vstack(
        [model_c.state_support_shift for _ in range(K_GROUPS)]
    )
    zero_group_value = np.vstack(
        [model_c.state_value_shift for _ in range(K_GROUPS)]
    )
    zero_group_weights = np.concatenate(
        [model_h.group_weights[group] * model_c.state_weights for group in range(K_GROUPS)]
    )
    zero_group_h = replace(
        model_h,
        state_support_shift=zero_group_support,
        state_value_shift=zero_group_value,
        state_weights=zero_group_weights,
        group_support_signature=np.zeros_like(model_h.group_support_signature),
        group_value_signature=np.zeros_like(model_h.group_value_signature),
    )
    zero_group_score, _ = score_model(
        zero_group_h,
        design_for(zero_group_h, layout, roles.audit),
        audit_outcomes,
    )
    np.testing.assert_allclose(
        zero_group_score,
        audit_scores["C"],
        atol=2.0e-12,
        rtol=0.0,
    )
    assert len(diagnostics.continuous_nodes) == CONTINUOUS_QUADRATURE_ORDER
    assert crossfit.train_validation_overlap_count == 0
    assert crossfit.unit_coverage_count == N_DEVELOPMENT_UNITS
    assert crossfit.fit_count_by_family["C"] == crossfit.fit_count_by_family["H"]
    assert (
        crossfit.sigma_score_count_by_family["C"]
        == crossfit.sigma_score_count_by_family["H"]
    )
    assert crossfit.selection_reference_family == "C"

    zero_calibration = calibrate_oracle_group_effects(
        layout,
        scenarios=["reusable_groups"],
        target_margin_multipliers=[0.0],
        margin=DEFAULT_MARGIN,
        seed=7331,
        replicates=1,
        amplitude_grid=[0.0],
    )
    assert len(zero_calibration.points) == 1
    assert zero_calibration.points[0].generator_amplitude == 0.0
    assert np.isclose(
        zero_calibration.points[0].independent_verification_mean_gain,
        0.0,
        atol=2.0e-12,
    )
    return {
        "status": "pass",
        "real_projection_outcomes_read": False,
        "metadata_cells": len(layout.frame),
        "metadata_units": int(len(set(layout.units.tolist()))),
        "roles": {"development": 12, "audit": 8, "unassigned": 16},
        "coordinates": N_COORDINATES,
        "models": finite_by_model,
        "factorization_checks": {
            "P_is_B_plus_position_without_B_refit": True,
            "C_and_H_continuous_backbone_elementwise_identical": True,
            "D_and_H_group_signatures_weights_and_gate_elementwise_identical": True,
            "D_is_based_on_B": True,
            "H_is_based_on_C": True,
            "group_signatures_fit_conditionally_on_C_then_shared_by_D_H": True,
            "zero_group_amplitude_H_score_equals_C_elementwise": True,
        },
        "crossfit_checks": {
            "protocol_id": crossfit.protocol_id,
            "folds": len(crossfit.folds),
            "train_validation_overlap_count": crossfit.train_validation_overlap_count,
            "validation_unit_coverage": crossfit.unit_coverage_count,
            "C_H_fit_budget_equal": True,
            "C_H_sigma_grid_budget_equal": True,
            "selection_reference_family": crossfit.selection_reference_family,
            "selected_sigma": crossfit.selected_sigma,
            "unseen_line_rule": crossfit.unseen_line_rule,
        },
        "oracle_calibration_checks": {
            "protocol_id": zero_calibration.protocol_id,
            "zero_amplitude_projected_H_minus_C": 0.0,
            "labels_integrated_not_conditioned": True,
            "independent_verification_path_exercised": True,
            "synthetic_truth_is_invariant_to_full_vs_assigned_index_scope": True,
        },
        "known_subset_limitations": [
            "role split is not the final geometry-eligible lock",
            "model coefficient and residual-PCA fits are pooled-cell weighted",
            "continuous absolute adequacy is not implemented",
            "full position-standardized reuse policy is not implemented",
            "oracle C projection is generator-direction aligned rather than the full registered C family",
            "H-minus-D continuum amplitude is not oracle-calibrated",
        ],
        "qualification_class": "synthetic_smoke_subset_not_registered_qualification",
        "outcome_bearing_search_authorized": False,
    }


def parse_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_nonnegative_grid(text: str, name: str) -> list[float]:
    values = [float(item) for item in parse_list(text)]
    if not values or any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{name} must contain finite nonnegative values")
    if values != sorted(set(values)):
        raise ValueError(f"{name} must be sorted and unique")
    return values


def main() -> None:
    episode_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=episode_root / "outputs/stage0/whole_cortex_ssptr_inventory.csv",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=episode_root / "outputs/stage0/allen_target_vocabulary.csv",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replicates", type=int, default=40)
    parser.add_argument(
        "--target-margin-multipliers",
        default=",".join(str(value) for value in DEFAULT_TARGET_MARGIN_MULTIPLIERS),
        help="Oracle H-C targets expressed as multiples of the fixed score margin",
    )
    parser.add_argument(
        "--calibration-amplitude-grid",
        default=",".join(str(value) for value in ORACLE_CALIBRATION_AMPLITUDES),
        help="Fixed generator-amplitude grid used only to build the oracle mapping",
    )
    parser.add_argument(
        "--calibration-replicates",
        type=int,
        default=8,
        help="Independent synthetic fit and verification replicates per calibration point",
    )
    parser.add_argument("--scenarios", default=",".join(ALL_SCENARIOS))
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one replicate of curved, grouped, and grouped+continuous regimes",
    )
    args = parser.parse_args()

    vocabulary = validate_vocabulary(args.vocabulary)
    layout = load_metadata(args.inventory)
    if args.self_test:
        result = executable_self_test(layout)
        result["vocabulary"] = vocabulary
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return

    scenarios = parse_list(args.scenarios)
    unknown = sorted(set(scenarios) - set(ALL_SCENARIOS))
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}")
    target_multipliers = parse_nonnegative_grid(
        args.target_margin_multipliers, "Target margin multiplier grid"
    )
    amplitude_grid = parse_nonnegative_grid(
        args.calibration_amplitude_grid, "Calibration amplitude grid"
    )
    if amplitude_grid[0] != 0.0:
        raise ValueError("Calibration amplitude grid must start at zero")
    calibration_replicates = args.calibration_replicates
    if calibration_replicates < 1:
        raise ValueError("--calibration-replicates must be positive")
    replicates = args.replicates
    if args.smoke:
        scenarios = [
            "curved_horseshoe_continuum",
            "reusable_groups",
            "groups_plus_continuum",
        ]
        target_multipliers = [0.0]
        amplitude_grid = [0.0]
        calibration_replicates = 1
        replicates = 1
    if replicates < 1:
        raise ValueError("--replicates must be positive")
    if args.output_dir is None and not args.smoke:
        raise ValueError("--output-dir is required for a non-smoke run")

    grouped_scenarios = [scenario for scenario in scenarios if scenario in GROUP_SCENARIOS]
    calibration = calibrate_oracle_group_effects(
        layout,
        scenarios=grouped_scenarios,
        target_margin_multipliers=target_multipliers,
        margin=args.margin,
        seed=args.seed,
        replicates=calibration_replicates,
        amplitude_grid=amplitude_grid,
    )

    rows: list[dict[str, object]] = []
    for condition in scenario_conditions(scenarios, calibration):
        for replicate in range(replicates):
            rows.append(
                evaluate_replicate(
                    layout,
                    condition,
                    replicate,
                    args.seed,
                    args.margin,
                )
            )
    summary = summarize(rows, args.margin)
    summary["vocabulary"] = vocabulary
    summary["oracle_effect_calibration"] = oracle_calibration_to_dict(calibration)
    if args.output_dir is not None:
        write_outputs(args.output_dir, rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
