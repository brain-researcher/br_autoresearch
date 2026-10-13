#!/usr/bin/env python3
"""Run EP10 v3 M0 with the V6 weak-grid exact-Hessian Newton branch.

Every fit whose ridge lambda is at least 1e-5 is forwarded directly to the
immutable V5 implementation.  Only the frozen weak-grid values 1e-6, 1e-7,
and 1e-8 use an exact conditional fixed-K Hessian and a deterministic
line-search Newton direction.  The scientific objective, analytic gradient,
ridge path, warm starts, tolerances, V4 strong-Wolfe search, and V5 eligible
finite-precision-stall certificate are unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(os.path.abspath(__file__))
if (
    not SCRIPT_PATH.is_file()
    or SCRIPT_PATH.is_symlink()
    or SCRIPT_PATH != SCRIPT_PATH.resolve(strict=True)
):
    raise ImportError("v6 solver wrapper must be an unsymlinked regular file")
SCRIPT_DIR = SCRIPT_PATH.parent
STAGE0_DIR = SCRIPT_DIR.parent
V5_ENGINE_PATH = SCRIPT_DIR / "fit_v3_m0_v5.py"
if (
    not V5_ENGINE_PATH.is_file()
    or V5_ENGINE_PATH.is_symlink()
    or V5_ENGINE_PATH != V5_ENGINE_PATH.resolve(strict=True)
):
    raise ImportError("v5 predecessor wrapper must be an unsymlinked stage0 file")

_V5_SPEC = importlib.util.spec_from_file_location(
    "ep10_v3_m0_solver_v5_for_v6", V5_ENGINE_PATH
)
if _V5_SPEC is None or _V5_SPEC.loader is None:
    raise ImportError(f"cannot import v5 solver wrapper: {V5_ENGINE_PATH}")
v5 = importlib.util.module_from_spec(_V5_SPEC)
sys.modules[_V5_SPEC.name] = v5
_V5_SPEC.loader.exec_module(v5)
v4 = v5.v4
v3 = v5.v3
v2 = v5.v2
base = v5.base


BASE_CONTRACT_PATH = STAGE0_DIR / "v3_m0_model_contract.yaml"
V5_ADDENDUM_PATH = STAGE0_DIR / "v3_m0_solver_addendum_v5.yaml"
V6_ADDENDUM_PATH = STAGE0_DIR / "v3_m0_solver_addendum_v6.yaml"

PARAMETER_DIMENSION = v5.PARAMETER_DIMENSION
HISTORY_SIZE = v5.HISTORY_SIZE
MAXIMUM_ITERATIONS = v5.MAXIMUM_ITERATIONS
GRADIENT_TOLERANCE = v5.GRADIENT_TOLERANCE
RELATIVE_OBJECTIVE_TOLERANCE = v5.RELATIVE_OBJECTIVE_TOLERANCE
DESCENDING_LAMBDAS = v5.DESCENDING_LAMBDAS
WEAK_NEWTON_LAMBDAS = (1.0e-6, 1.0e-7, 1.0e-8)
STRONG_V5_LAMBDAS = tuple(
    value for value in DESCENDING_LAMBDAS if value not in WEAK_NEWTON_LAMBDAS
)

PROBE_OUTER_HELD_GROUP = v5.PROBE_OUTER_HELD_GROUP
PROBE_INNER_HELD_GROUP = v5.PROBE_INNER_HELD_GROUP
PROBE_EXPECTED_CELLS = v5.PROBE_EXPECTED_CELLS
PROBE_EXPECTED_GROUPS = v5.PROBE_EXPECTED_GROUPS
WARM_PATH_MAX_ITERATIONS_EXCLUSIVE = v5.WARM_PATH_MAX_ITERATIONS_EXCLUSIVE
WEAK_PREFIX_LAMBDAS = tuple(
    value for value in DESCENDING_LAMBDAS if value >= 1.0e-7
)
WEAK_PREFIX_REQUIRED = (1.0e-6, 1.0e-7)

PAIR_IDENTITY_TOLERANCE = 5.0e-9
HESSIAN_SYMMETRY_RELATIVE_TOLERANCE = 5.0e-12
SOLVE_BACKWARD_ERROR_TOLERANCE = 5.0e-11
DAMPING_MULTIPLIER = 10.0
DAMPING_MAXIMUM_POSITIVE_ATTEMPTS = 8
DAMPING_MAXIMUM_RELATIVE_TO_SCALE = 1.0e-4
ITERATIVE_REFINEMENT_STEPS = 2
PRODUCTION_OUTPUT_BASENAME = "v3_m0_fit_v6"

V5_FIT_LBFGS_ORIGINAL = v5.fit_lbfgs_v5
V4_STRONG_WOLFE_SEARCH = v5.V4_STRONG_WOLFE_SEARCH
FIT_TELEMETRY = v5.FIT_TELEMETRY


def _require_exact_unsymlinked_file(
    observed: Path, expected: Path, label: str
) -> Path:
    absolute = Path(os.path.abspath(observed))
    expected_absolute = Path(os.path.abspath(expected))
    if (
        not absolute.is_file()
        or absolute.is_symlink()
        or absolute != absolute.resolve(strict=True)
        or expected_absolute != expected_absolute.resolve(strict=True)
        or absolute != expected_absolute
    ):
        raise base.ModelError(
            f"{label} must be the exact unsymlinked EP10 stage0 file"
        )
    return absolute


def _as_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise base.ModelError(f"v6 solver addendum has invalid {label}") from error
    if not math.isfinite(result):
        raise base.ModelError(f"v6 solver addendum has nonfinite {label}")
    return result


def validate_solver_addendum(
    base_contract_path: Path, addendum_path: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate only the model-preserving V6 numerical choices."""
    _require_exact_unsymlinked_file(
        base_contract_path, BASE_CONTRACT_PATH, "v1 M0 model contract"
    )
    _require_exact_unsymlinked_file(
        V5_ENGINE_PATH, V5_ENGINE_PATH, "v5 predecessor wrapper"
    )
    _require_exact_unsymlinked_file(
        V5_ADDENDUM_PATH, V5_ADDENDUM_PATH, "v5 predecessor addendum"
    )
    _require_exact_unsymlinked_file(
        addendum_path, V6_ADDENDUM_PATH, "v6 solver addendum"
    )
    base_contract, predecessor = v5.validate_solver_addendum(
        base_contract_path, V5_ADDENDUM_PATH
    )
    addendum = base._load_yaml(addendum_path)
    if (
        addendum.get("schema_version")
        != "episode10_v3_m0_solver_addendum_v6"
        or addendum.get("status")
        != "frozen_before_v6_weak_grid_probe_and_execution"
        or addendum.get("scientific_authority") is not False
        or addendum.get("scope", {}).get("role") != "development_only"
        or addendum.get("scope", {}).get("final_or_unassigned_access")
        != "forbidden"
    ):
        raise base.ModelError("v6 addendum identity or sealed scope changed")

    reference = addendum.get("predecessor_solver_addendum", {})
    if (
        reference.get("required_schema_version")
        != predecessor.get("schema_version")
        or reference.get("required_status") != predecessor.get("status")
    ):
        raise base.ModelError("v6 predecessor reference changed")

    solver = addendum.get("solver", {})
    branch = solver.get("branch", {})
    newton = solver.get("newton", {})
    line_search = solver.get("line_search", {})
    convergence = solver.get("convergence", {})
    strong_lambdas = tuple(
        float(value) for value in branch.get("strong_grid", {}).get("lambdas", ())
    )
    weak_lambdas = tuple(
        float(value) for value in branch.get("weak_grid", {}).get("lambdas", ())
    )
    if (
        solver.get("algorithm") != "deterministic_versioned_hybrid"
        or int(solver.get("parameter_dimension", -1)) != PARAMETER_DIMENSION
        or int(solver.get("maximum_iterations", -1)) != MAXIMUM_ITERATIONS
        or strong_lambdas != STRONG_V5_LAMBDAS
        or weak_lambdas != WEAK_NEWTON_LAMBDAS
        or branch.get("strong_grid", {}).get("implementation")
        != "immutable_V5_bitwise_forward"
        or branch.get("weak_grid", {}).get("implementation")
        != "exact_Hessian_line_search_Newton"
        or newton.get("factorization") != "deterministic_Cholesky"
        or int(newton.get("maximum_positive_shift_attempts", -1))
        != DAMPING_MAXIMUM_POSITIVE_ATTEMPTS
        or _as_float(
            newton.get("positive_shift_multiplier"), "damping multiplier"
        )
        != DAMPING_MULTIPLIER
        or _as_float(
            newton.get("maximum_shift_relative_to_H_scale"),
            "maximum relative damping",
        )
        != DAMPING_MAXIMUM_RELATIVE_TO_SCALE
        or int(newton.get("iterative_refinement_steps", -1))
        != ITERATIVE_REFINEMENT_STEPS
        or _as_float(
            newton.get("maximum_scaled_backward_error"),
            "solve backward-error tolerance",
        )
        != SOLVE_BACKWARD_ERROR_TOLERANCE
        or line_search.get("implementation") != "unchanged_v4_byte_for_byte"
        or _as_float(
            convergence.get("gradient_infinity_tolerance"),
            "gradient tolerance",
        )
        != GRADIENT_TOLERANCE
        or _as_float(
            convergence.get("relative_objective_tolerance"),
            "relative objective tolerance",
        )
        != RELATIVE_OBJECTIVE_TOLERANCE
        or solver.get("fallback_optimizer") != "forbidden"
    ):
        raise base.ModelError("v6 solver branch or numerical settings changed")

    ridge_path = addendum.get("ridge_path", {})
    initialization = ridge_path.get("initialization", {})
    if (
        tuple(float(value) for value in ridge_path.get("ordered_lambdas", ()))
        != DESCENDING_LAMBDAS
        or initialization.get("first_lambda") != "all_zero_coefficients"
        or initialization.get("later_lambdas")
        != "immediately_preceding_converged_coefficients"
        or ridge_path.get("auxiliary_lambdas") != "forbidden"
    ):
        raise base.ModelError("v6 ridge grid or warm-start order changed")

    narrow = addendum.get("narrow_weak_prefix_probe", {})
    full = addendum.get("full_targeted_probe", {})
    if (
        str(narrow.get("outer_held_group")) != PROBE_OUTER_HELD_GROUP
        or str(narrow.get("inner_held_group")) != PROBE_INNER_HELD_GROUP
        or int(narrow.get("expected_training_cells", -1))
        != PROBE_EXPECTED_CELLS
        or int(narrow.get("expected_training_groups", -1))
        != PROBE_EXPECTED_GROUPS
        or tuple(float(value) for value in narrow.get("required_weak_lambdas", ()))
        != WEAK_PREFIX_REQUIRED
        or int(
            narrow.get(
                "each_required_weak_lambda_max_iterations_exclusive", -1
            )
        )
        != WARM_PATH_MAX_ITERATIONS_EXCLUSIVE
        or full.get("runs_only_after_narrow_weak_prefix_probe_passes") is not True
        or int(full.get("deterministic_warm_path_repeats", -1)) != 2
        or full.get("zero_start_comparison_at_every_lambda") is not True
        or int(full.get("warm_path_max_iterations_exclusive", -1))
        != WARM_PATH_MAX_ITERATIONS_EXCLUSIVE
        or narrow.get("final_or_unassigned_access") != "forbidden"
        or full.get("final_or_unassigned_access") != "forbidden"
    ):
        raise base.ModelError("v6 probe contract changed")

    telemetry = addendum.get("telemetry", {})
    if (
        telemetry.get("scalar_only") is not True
        or telemetry.get("coefficients_prediction_vectors_or_target_contrasts")
        != "forbidden"
    ):
        raise base.ModelError("v6 scalar-only telemetry seal changed")
    return base_contract, addendum


def _finite_or_none(value: float) -> float | None:
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


def _max_abs(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _pair_moments_exact_dp(
    eta: np.ndarray, k_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return exact-DP inclusion covariance for every fixed-K row."""
    eta = np.asarray(eta, dtype=float)
    k_values = np.asarray(k_values, dtype=int)
    if eta.ndim != 2 or k_values.shape != (eta.shape[0],):
        raise ValueError("eta and K arrays are not aligned for pair moments")
    if not np.all(np.isfinite(eta)):
        raise base.ModelError("pair-moment logits contain a nonfinite value")
    n, p = eta.shape
    if np.any(k_values < 0) or np.any(k_values > p):
        raise ValueError("a pair-moment K value is outside the panel")

    _, marginals = base.batch_fixed_k_logz_marginals(eta, k_values)
    joint_directional = np.zeros((n, p, p), dtype=float)
    diagonal = np.arange(p)
    joint_directional[:, diagonal, diagonal] = marginals
    active = np.flatnonzero(k_values >= 1)
    if active.size:
        for target in range(p):
            keep = np.asarray(
                [index for index in range(p) if index != target], dtype=int
            )
            _, conditional = base.batch_fixed_k_logz_marginals(
                eta[active][:, keep], k_values[active] - 1
            )
            joint_directional[
                active[:, None], target, keep[None, :]
            ] = marginals[active, target, None] * conditional

    directional_asymmetry = _max_abs(
        joint_directional - np.swapaxes(joint_directional, 1, 2)
    )
    joint = 0.5 * (
        joint_directional + np.swapaxes(joint_directional, 1, 2)
    )
    joint[:, diagonal, diagonal] = marginals
    covariance = joint - marginals[:, :, None] * marginals[:, None, :]
    covariance = 0.5 * (covariance + np.swapaxes(covariance, 1, 2))

    degenerate = (k_values == 0) | (k_values == p)
    covariance[degenerate] = 0.0
    if not (
        np.all(np.isfinite(marginals))
        and np.all(np.isfinite(joint))
        and np.all(np.isfinite(covariance))
    ):
        raise base.ModelError("exact pair-moment DP produced a nonfinite value")

    marginal_count_error = _max_abs(
        np.sum(marginals, axis=1) - k_values.astype(float)
    )
    joint_count_error = _max_abs(
        np.sum(joint, axis=2) - k_values[:, None] * marginals
    )
    covariance_row_sum_error = _max_abs(np.sum(covariance, axis=2))
    lower_bound_error = max(0.0, -float(np.min(joint))) if joint.size else 0.0
    pair_upper = np.minimum(marginals[:, :, None], marginals[:, None, :])
    upper_bound_error = (
        max(0.0, float(np.max(joint - pair_upper))) if joint.size else 0.0
    )
    maximum_identity_error = max(
        directional_asymmetry,
        marginal_count_error,
        joint_count_error,
        covariance_row_sum_error,
        lower_bound_error,
        upper_bound_error,
    )
    if maximum_identity_error > PAIR_IDENTITY_TOLERANCE:
        raise base.ModelError(
            "exact pair-moment DP violated a fixed-K identity"
        )
    diagnostics = {
        "directional_pair_asymmetry": directional_asymmetry,
        "marginal_count_identity_error": marginal_count_error,
        "joint_count_identity_error": joint_count_error,
        "covariance_row_sum_error": covariance_row_sum_error,
        "joint_lower_bound_error": lower_bound_error,
        "joint_upper_bound_error": upper_bound_error,
        "maximum_identity_error": maximum_identity_error,
    }
    return covariance, marginals, diagnostics


def _exact_hessian_v6(
    coefficient_vector: np.ndarray,
    x: np.ndarray,
    k_values: np.ndarray,
    groups: Sequence[str],
    target_contrasts: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Construct the full analytic fixed-K Hessian in coefficient order."""
    vector = np.asarray(coefficient_vector, dtype=float)
    x = np.asarray(x, dtype=float)
    target_contrasts = np.asarray(target_contrasts, dtype=float)
    if x.ndim != 2 or target_contrasts.ndim != 2:
        raise ValueError("Hessian design and contrasts must be matrices")
    n, d = x.shape
    p, q = target_contrasts.shape
    dimension = d * q
    if (
        vector.shape != (dimension,)
        or k_values.shape != (n,)
        or len(groups) != n
        or q != p - 1
    ):
        raise ValueError("Hessian inputs have incompatible shapes")
    if (
        not np.all(np.isfinite(vector))
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(target_contrasts))
        or not math.isfinite(float(ridge_lambda))
        or ridge_lambda <= 0.0
    ):
        raise base.ModelError("Hessian inputs must be finite with positive ridge")

    coefficients = vector.reshape(d, q)
    eta = (x @ coefficients) @ target_contrasts.T
    covariance, _, pair_diagnostics = _pair_moments_exact_dp(eta, k_values)
    contrast_covariance = np.einsum(
        "tr,ntu,us->nrs",
        target_contrasts,
        covariance,
        target_contrasts,
        optimize=True,
    )
    weights = base.equal_group_weights(groups)
    hessian_four = np.einsum(
        "n,na,nb,nrs->arbs",
        weights,
        x,
        x,
        contrast_covariance,
        optimize=True,
    )
    hessian = hessian_four.reshape(dimension, dimension)
    hessian.flat[:: dimension + 1] += float(ridge_lambda)
    raw_symmetry_error = _max_abs(hessian - hessian.T)
    scale = max(1.0, float(np.linalg.norm(hessian, ord=np.inf)))
    if not math.isfinite(scale):
        raise base.ModelError("exact Hessian scale is nonfinite")
    if raw_symmetry_error > HESSIAN_SYMMETRY_RELATIVE_TOLERANCE * scale:
        raise base.ModelError("exact Hessian exceeded its symmetry tolerance")
    hessian = 0.5 * (hessian + hessian.T)
    if not np.all(np.isfinite(hessian)):
        raise base.ModelError("exact Hessian contains a nonfinite value")
    diagonal = np.diag(hessian)
    if np.any(diagonal <= 0.0):
        raise base.ModelError("exact ridge Hessian has a nonpositive diagonal")
    diagnostics = {
        **pair_diagnostics,
        "raw_Hessian_symmetry_error": raw_symmetry_error,
        "Hessian_infinity_norm": float(np.linalg.norm(hessian, ord=np.inf)),
        "Hessian_minimum_diagonal": float(np.min(diagonal)),
        "Hessian_maximum_diagonal": float(np.max(diagonal)),
    }
    return hessian, diagnostics


def _newton_direction_v6(
    hessian: np.ndarray,
    gradient: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Solve a boundedly damped Newton system with fail-closed checks."""
    hessian = np.asarray(hessian, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    if (
        hessian.ndim != 2
        or hessian.shape[0] != hessian.shape[1]
        or gradient.shape != (hessian.shape[0],)
        or not np.all(np.isfinite(hessian))
        or not np.all(np.isfinite(gradient))
        or not math.isfinite(float(ridge_lambda))
        or ridge_lambda <= 0.0
    ):
        raise base.ModelError("Newton system is malformed or nonfinite")
    scale = max(1.0, float(np.linalg.norm(hessian, ord=np.inf)))
    if not math.isfinite(scale):
        raise base.ModelError("Newton Hessian scale is nonfinite")
    symmetry_error = _max_abs(hessian - hessian.T)
    if symmetry_error > HESSIAN_SYMMETRY_RELATIVE_TOLERANCE * scale:
        raise base.ModelError("Newton Hessian is not numerically symmetric")
    hessian = 0.5 * (hessian + hessian.T)

    positive_base = max(
        float(ridge_lambda), np.finfo(float).eps * scale
    )
    shifts = [0.0]
    shift = positive_base
    for _ in range(DAMPING_MAXIMUM_POSITIVE_ATTEMPTS):
        if shift > DAMPING_MAXIMUM_RELATIVE_TO_SCALE * scale:
            break
        shifts.append(float(shift))
        shift *= DAMPING_MULTIPLIER

    identity = np.eye(hessian.shape[0], dtype=float)
    right_hand_side = -gradient
    last_reason = "no admissible shift"
    for attempt, diagonal_shift in enumerate(shifts, start=1):
        system = hessian + diagonal_shift * identity
        try:
            factor = np.linalg.cholesky(system)
            intermediate = np.linalg.solve(factor, right_hand_side)
            direction = np.linalg.solve(factor.T, intermediate)
        except np.linalg.LinAlgError:
            last_reason = "Cholesky factorization failed"
            continue
        if not np.all(np.isfinite(direction)):
            last_reason = "Newton solution became nonfinite"
            continue
        for _ in range(ITERATIVE_REFINEMENT_STEPS):
            residual = right_hand_side - system @ direction
            if not np.all(np.isfinite(residual)):
                break
            correction = np.linalg.solve(
                factor.T, np.linalg.solve(factor, residual)
            )
            if not np.all(np.isfinite(correction)):
                break
            direction = direction + correction

        residual = system @ direction + gradient
        denominator = max(
            np.finfo(float).tiny,
            float(np.linalg.norm(system, ord=np.inf))
            * float(np.linalg.norm(direction, ord=np.inf))
            + float(np.linalg.norm(gradient, ord=np.inf)),
        )
        if not math.isfinite(denominator):
            last_reason = "Newton backward-error scale became nonfinite"
            continue
        backward_error = float(np.linalg.norm(residual, ord=np.inf)) / denominator
        directional_derivative = float(np.dot(gradient, direction))
        predicted_decrease = float(
            -directional_derivative
            - 0.5 * np.dot(direction, hessian @ direction)
        )
        if not math.isfinite(backward_error) or (
            backward_error > SOLVE_BACKWARD_ERROR_TOLERANCE
        ):
            last_reason = "Newton solve backward error exceeded tolerance"
            continue
        if not math.isfinite(directional_derivative) or not (
            directional_derivative < 0.0
        ):
            last_reason = "Newton solve did not produce a descent direction"
            continue
        if not math.isfinite(predicted_decrease) or not (
            predicted_decrease > 0.0
        ):
            last_reason = "Newton quadratic predicted decrease was not positive"
            continue
        return direction, {
            "factorization_attempts": attempt,
            "diagonal_shift": float(diagonal_shift),
            "Hessian_scale": scale,
            "scaled_backward_error": backward_error,
            "directional_derivative": directional_derivative,
            "quadratic_predicted_decrease": predicted_decrease,
        }
    raise base.ModelError(
        "bounded exact-Hessian Newton factorization failed closed: " + last_reason
    )


def _diagnostic_summary(
    records: Sequence[Mapping[str, float | int]], key: str
) -> dict[str, float | int | None]:
    values = [float(record[key]) for record in records if key in record]
    return v2._numeric_summary(values)


def _telemetry_v6(
    *,
    status: str,
    ridge_lambda: float,
    x: np.ndarray,
    groups: Sequence[str],
    vector: np.ndarray,
    objective: float,
    gradient: np.ndarray,
    gradient_norm: float,
    relative_change: float,
    iterations: int,
    accepted_steps: int,
    step_sizes: Sequence[float],
    line_search_evaluations: int,
    expansion_counts: Sequence[float],
    zoom_counts: Sequence[float],
    armijo_slacks: Sequence[float],
    curvature_ratios: Sequence[float],
    hessian_diagnostics: Sequence[Mapping[str, float | int]],
    solve_diagnostics: Sequence[Mapping[str, float | int]],
    convergence_witness: str,
    line_search_failure: Mapping[str, Any] | None = None,
    certificate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = v3._telemetry(
        status=status,
        ridge_lambda=ridge_lambda,
        x=x,
        groups=groups,
        vector=vector,
        objective=objective,
        gradient_norm=gradient_norm,
        relative_change=relative_change,
        iterations=iterations,
        accepted_steps=accepted_steps,
        step_sizes=step_sizes,
        h0_scales=[],
        line_search_evaluations=line_search_evaluations,
        curvature_updates=0,
        curvature_skips=0,
        expansion_counts=expansion_counts,
        zoom_counts=zoom_counts,
        armijo_slacks=armijo_slacks,
        curvature_ratios=curvature_ratios,
        line_search_failure=line_search_failure,
    )
    gradient_l2 = (
        v5._stable_l2(np.asarray(gradient, dtype=float))
        if np.all(np.isfinite(gradient))
        else math.inf
    )
    payload.update(
        {
            "solver_version": "v6_weak_grid_exact_Hessian_Newton",
            "solver_algorithm": "exact_Hessian_line_search_Newton",
            "solver_branch": "weak_grid_exact_Hessian_Newton",
            "weak_grid_lambdas": list(WEAK_NEWTON_LAMBDAS),
            "v4_line_search_unchanged": True,
            "v5_certificate_unchanged": True,
            "history_size": 0,
            "v5_history_size_for_strong_grid": HISTORY_SIZE,
            "exact_Hessian_evaluations": len(hessian_diagnostics),
            "convergence_witness": convergence_witness,
            "exact_gradient_l2_norm": _finite_or_none(gradient_l2),
            "gradient_infinity_tolerance_pass": bool(
                math.isfinite(gradient_norm)
                and gradient_norm <= GRADIENT_TOLERANCE
            ),
            "successive_relative_objective_change_tolerance_pass": bool(
                math.isfinite(relative_change)
                and relative_change <= RELATIVE_OBJECTIVE_TOLERANCE
            ),
            "accepted_step_values": [float(step) for step in step_sizes],
            "pair_DP_maximum_identity_error_summary": _diagnostic_summary(
                hessian_diagnostics, "maximum_identity_error"
            ),
            "Hessian_symmetry_error_summary": _diagnostic_summary(
                hessian_diagnostics, "raw_Hessian_symmetry_error"
            ),
            "Hessian_minimum_diagonal_summary": _diagnostic_summary(
                hessian_diagnostics, "Hessian_minimum_diagonal"
            ),
            "Hessian_maximum_diagonal_summary": _diagnostic_summary(
                hessian_diagnostics, "Hessian_maximum_diagonal"
            ),
            "Cholesky_factorization_attempt_summary": _diagnostic_summary(
                solve_diagnostics, "factorization_attempts"
            ),
            "Newton_diagonal_shift_summary": _diagnostic_summary(
                solve_diagnostics, "diagonal_shift"
            ),
            "Newton_scaled_backward_error_summary": _diagnostic_summary(
                solve_diagnostics, "scaled_backward_error"
            ),
            "Newton_directional_derivative_summary": _diagnostic_summary(
                solve_diagnostics, "directional_derivative"
            ),
            "Newton_quadratic_predicted_decrease_summary": _diagnostic_summary(
                solve_diagnostics, "quadratic_predicted_decrease"
            ),
            "strong_convexity_certificate": (
                dict(certificate)
                if certificate is not None
                else {"attempted": False, "certificate_pass": False}
            ),
            "coefficient_prediction_or_target_contrast_vectors_emitted": False,
        }
    )
    return payload


def _fit_newton_v6_core(
    x: np.ndarray,
    y: np.ndarray,
    k_values: np.ndarray,
    groups: Sequence[str],
    target_contrasts: np.ndarray,
    ridge_lambda: float,
    initial: np.ndarray | None,
    *,
    maximum_iterations: int,
    gradient_tolerance: float,
    relative_objective_tolerance: float,
) -> v3.OptimizerResultV3:
    dimension = x.shape[1] * target_contrasts.shape[1]
    if (
        maximum_iterations != MAXIMUM_ITERATIONS
        or gradient_tolerance != GRADIENT_TOLERANCE
        or relative_objective_tolerance != RELATIVE_OBJECTIVE_TOLERANCE
    ):
        raise base.ModelError("a caller attempted to override frozen v6 settings")
    if initial is None:
        vector = np.zeros(dimension, dtype=float)
    else:
        initial_array = np.asarray(initial, dtype=float)
        if initial_array.shape == (x.shape[1], target_contrasts.shape[1]):
            vector = initial_array.ravel().copy()
        elif initial_array.shape == (dimension,):
            vector = initial_array.copy()
        else:
            raise ValueError("optimizer initial value has the wrong shape")

    def evaluate(candidate: np.ndarray) -> tuple[float, np.ndarray]:
        return base.objective_and_gradient(
            candidate,
            x,
            y,
            k_values,
            groups,
            target_contrasts,
            ridge_lambda,
        )

    value, gradient = evaluate(vector)
    gradient_norm = float(np.max(np.abs(gradient)))
    relative_change = math.inf
    step_sizes: list[float] = []
    expansion_counts: list[float] = []
    zoom_counts: list[float] = []
    armijo_slacks: list[float] = []
    curvature_ratios: list[float] = []
    hessian_diagnostics: list[dict[str, float | int]] = []
    solve_diagnostics: list[dict[str, float | int]] = []
    line_search_evaluations = 0
    accepted_steps = 0

    if gradient_norm <= gradient_tolerance:
        telemetry = _telemetry_v6(
            status="converged_at_initial_value",
            ridge_lambda=ridge_lambda,
            x=x,
            groups=groups,
            vector=vector,
            objective=value,
            gradient=gradient,
            gradient_norm=gradient_norm,
            relative_change=0.0,
            iterations=0,
            accepted_steps=0,
            step_sizes=step_sizes,
            line_search_evaluations=0,
            expansion_counts=expansion_counts,
            zoom_counts=zoom_counts,
            armijo_slacks=armijo_slacks,
            curvature_ratios=curvature_ratios,
            hessian_diagnostics=hessian_diagnostics,
            solve_diagnostics=solve_diagnostics,
            convergence_witness="initial_gradient",
        )
        FIT_TELEMETRY.append(telemetry)
        return v3.OptimizerResultV3(
            coefficients=vector.reshape(x.shape[1], target_contrasts.shape[1]),
            objective=value,
            gradient_infinity_norm=gradient_norm,
            iterations=0,
            accepted_steps=0,
            relative_objective_change=0.0,
            solver_telemetry=telemetry,
        )

    for iteration in range(1, maximum_iterations + 1):
        try:
            hessian, hessian_diag = _exact_hessian_v6(
                vector,
                x,
                k_values,
                groups,
                target_contrasts,
                ridge_lambda,
            )
            direction, solve_diag = _newton_direction_v6(
                hessian, gradient, ridge_lambda
            )
        except (base.ModelError, np.linalg.LinAlgError, ValueError) as error:
            telemetry = _telemetry_v6(
                status="Hessian_or_factorization_failure",
                ridge_lambda=ridge_lambda,
                x=x,
                groups=groups,
                vector=vector,
                objective=value,
                gradient=gradient,
                gradient_norm=gradient_norm,
                relative_change=relative_change,
                iterations=iteration - 1,
                accepted_steps=accepted_steps,
                step_sizes=step_sizes,
                line_search_evaluations=line_search_evaluations,
                expansion_counts=expansion_counts,
                zoom_counts=zoom_counts,
                armijo_slacks=armijo_slacks,
                curvature_ratios=curvature_ratios,
                hessian_diagnostics=hessian_diagnostics,
                solve_diagnostics=solve_diagnostics,
                convergence_witness="none",
            )
            FIT_TELEMETRY.append(telemetry)
            raise base.ModelError(
                "v6 exact-Hessian construction or bounded Cholesky solve failed"
            ) from error
        hessian_diagnostics.append(dict(hessian_diag))
        solve_diagnostics.append(dict(solve_diag))

        try:
            line = V4_STRONG_WOLFE_SEARCH(
                vector, value, gradient, direction, evaluate
            )
        except v3.StrongWolfeFailure as error:
            line_search_evaluations += error.evaluations
            vector_before = vector.copy()
            gradient_before = gradient.copy()
            value_before = value
            certificate = v5._strong_convexity_certificate(
                failure_message=str(error),
                objective=value,
                gradient=gradient,
                ridge_lambda=ridge_lambda,
                parameter_dimension=dimension,
                prior_relative_change=relative_change,
            )
            certificate.update(
                {
                    "failed_search_evaluations": int(error.evaluations),
                    "failed_search_expansions": int(error.expansions),
                    "failed_search_zoom_iterations": int(error.zoom_iterations),
                    "accepted_steps_before_certificate": int(accepted_steps),
                    "accepted_steps_after_certificate": int(accepted_steps),
                    "history_pairs_before_certificate": 0,
                    "history_pairs_after_certificate": 0,
                    "iterate_array_unchanged": bool(
                        np.array_equal(vector, vector_before)
                    ),
                    "objective_unchanged": bool(value == value_before),
                    "gradient_array_unchanged": bool(
                        np.array_equal(gradient, gradient_before)
                    ),
                    "no_pseudo_step": bool(
                        np.array_equal(vector, vector_before)
                        and value == value_before
                        and np.array_equal(gradient, gradient_before)
                    ),
                }
            )
            failure_payload = {
                "message": str(error),
                "evaluations": error.evaluations,
                "expansions": error.expansions,
                "zoom_iterations": error.zoom_iterations,
            }
            status = (
                "converged_after_line_search_precision_failure_by_"
                "strong_convexity_certificate"
                if certificate["certificate_pass"]
                else "line_search_failure"
            )
            telemetry = _telemetry_v6(
                status=status,
                ridge_lambda=ridge_lambda,
                x=x,
                groups=groups,
                vector=vector,
                objective=value,
                gradient=gradient,
                gradient_norm=gradient_norm,
                relative_change=relative_change,
                iterations=iteration - 1,
                accepted_steps=accepted_steps,
                step_sizes=step_sizes,
                line_search_evaluations=line_search_evaluations,
                expansion_counts=expansion_counts,
                zoom_counts=zoom_counts,
                armijo_slacks=armijo_slacks,
                curvature_ratios=curvature_ratios,
                hessian_diagnostics=hessian_diagnostics,
                solve_diagnostics=solve_diagnostics,
                convergence_witness=(
                    "strong_convexity_gap_after_v4_zoom_stall"
                    if certificate["certificate_pass"]
                    else "none"
                ),
                line_search_failure=failure_payload,
                certificate=certificate,
            )
            FIT_TELEMETRY.append(telemetry)
            if certificate["certificate_pass"]:
                return v3.OptimizerResultV3(
                    coefficients=vector.reshape(
                        x.shape[1], target_contrasts.shape[1]
                    ),
                    objective=value,
                    gradient_infinity_norm=gradient_norm,
                    iterations=iteration - 1,
                    accepted_steps=accepted_steps,
                    relative_objective_change=relative_change,
                    solver_telemetry=telemetry,
                )
            raise base.ModelError(
                "v6 unchanged strong-Wolfe search failed without an eligible "
                "strong-convexity certificate"
            ) from error

        line_search_evaluations += line.evaluations
        step_sizes.append(line.step)
        expansion_counts.append(float(line.expansions))
        zoom_counts.append(float(line.zoom_iterations))
        armijo_slacks.append(line.armijo_slack)
        curvature_ratios.append(line.curvature_ratio)
        new_vector, new_value, new_gradient = (
            line.vector,
            line.value,
            line.gradient,
        )
        relative_change = abs(new_value - value) / max(
            1.0, abs(value), abs(new_value)
        )
        vector, value, gradient = new_vector, new_value, new_gradient
        accepted_steps += 1
        gradient_norm = float(np.max(np.abs(gradient)))
        if (
            gradient_norm <= gradient_tolerance
            and relative_change <= relative_objective_tolerance
        ):
            telemetry = _telemetry_v6(
                status="converged",
                ridge_lambda=ridge_lambda,
                x=x,
                groups=groups,
                vector=vector,
                objective=value,
                gradient=gradient,
                gradient_norm=gradient_norm,
                relative_change=relative_change,
                iterations=iteration,
                accepted_steps=accepted_steps,
                step_sizes=step_sizes,
                line_search_evaluations=line_search_evaluations,
                expansion_counts=expansion_counts,
                zoom_counts=zoom_counts,
                armijo_slacks=armijo_slacks,
                curvature_ratios=curvature_ratios,
                hessian_diagnostics=hessian_diagnostics,
                solve_diagnostics=solve_diagnostics,
                convergence_witness="successive_relative_objective_change",
            )
            FIT_TELEMETRY.append(telemetry)
            return v3.OptimizerResultV3(
                coefficients=vector.reshape(
                    x.shape[1], target_contrasts.shape[1]
                ),
                objective=value,
                gradient_infinity_norm=gradient_norm,
                iterations=iteration,
                accepted_steps=accepted_steps,
                relative_objective_change=relative_change,
                solver_telemetry=telemetry,
            )

    telemetry = _telemetry_v6(
        status="iteration_cap_exhausted",
        ridge_lambda=ridge_lambda,
        x=x,
        groups=groups,
        vector=vector,
        objective=value,
        gradient=gradient,
        gradient_norm=gradient_norm,
        relative_change=relative_change,
        iterations=maximum_iterations,
        accepted_steps=accepted_steps,
        step_sizes=step_sizes,
        line_search_evaluations=line_search_evaluations,
        expansion_counts=expansion_counts,
        zoom_counts=zoom_counts,
        armijo_slacks=armijo_slacks,
        curvature_ratios=curvature_ratios,
        hessian_diagnostics=hessian_diagnostics,
        solve_diagnostics=solve_diagnostics,
        convergence_witness="none",
    )
    FIT_TELEMETRY.append(telemetry)
    raise base.ModelError(
        "v6 exact-Hessian Newton did not meet the unchanged convergence "
        f"witness in {maximum_iterations} iterations"
    )


def fit_lbfgs_v6(
    x: np.ndarray,
    y: np.ndarray,
    k_values: np.ndarray,
    groups: Sequence[str],
    target_contrasts: np.ndarray,
    ridge_lambda: float,
    initial: np.ndarray | None = None,
    *,
    history_size: int = HISTORY_SIZE,
    maximum_iterations: int = MAXIMUM_ITERATIONS,
    gradient_tolerance: float = GRADIENT_TOLERANCE,
    relative_objective_tolerance: float = RELATIVE_OBJECTIVE_TOLERANCE,
) -> v3.OptimizerResultV3:
    """Forward strong-grid fits to V5 and solve only the weak grid by Newton."""
    dimension = x.shape[1] * target_contrasts.shape[1]
    if dimension != PARAMETER_DIMENSION and not (
        x.shape[1] == 1 and target_contrasts.shape[0] == 6
    ):
        raise base.ModelError(f"unexpected v6 optimizer dimension {dimension}")
    if (
        history_size != HISTORY_SIZE
        or maximum_iterations != MAXIMUM_ITERATIONS
        or gradient_tolerance != GRADIENT_TOLERANCE
        or relative_objective_tolerance != RELATIVE_OBJECTIVE_TOLERANCE
    ):
        raise base.ModelError("a caller attempted to override frozen v6 settings")
    ridge = float(ridge_lambda)
    if ridge not in WEAK_NEWTON_LAMBDAS:
        return V5_FIT_LBFGS_ORIGINAL(
            x,
            y,
            k_values,
            groups,
            target_contrasts,
            ridge,
            initial=initial,
            history_size=history_size,
            maximum_iterations=maximum_iterations,
            gradient_tolerance=gradient_tolerance,
            relative_objective_tolerance=relative_objective_tolerance,
        )
    return _fit_newton_v6_core(
        x,
        y,
        k_values,
        groups,
        target_contrasts,
        ridge,
        initial,
        maximum_iterations=maximum_iterations,
        gradient_tolerance=gradient_tolerance,
        relative_objective_tolerance=relative_objective_tolerance,
    )


# Every unchanged v1 production route and the retained probe helper resolve
# these globals at call time. The immutable V5 callable above remains the
# direct implementation for every non-weak lambda.
base.fit_lbfgs = fit_lbfgs_v6
v2.fit_lbfgs_v2 = fit_lbfgs_v6
v2.FIT_TELEMETRY = FIT_TELEMETRY


def _exhaustive_pair_moment_self_test() -> dict[str, Any]:
    p = 6
    eta = np.asarray([-0.9, -0.31, 0.07, 0.26, 0.71, 1.03])
    maximum_marginal_error = 0.0
    maximum_covariance_error = 0.0
    maximum_identity_error = 0.0
    for k in range(p + 1):
        _, brute_marginals, states, probabilities = base._brute_fixed_k(eta, k)
        state_matrix = np.zeros((len(states), p), dtype=float)
        for row, state in enumerate(states):
            state_matrix[row, list(state)] = 1.0
        brute_joint = np.einsum(
            "n,nt,nu->tu", probabilities, state_matrix, state_matrix
        )
        brute_covariance = brute_joint - np.outer(brute_marginals, brute_marginals)
        covariance, marginals, diagnostics = _pair_moments_exact_dp(
            eta[None, :], np.asarray([k], dtype=int)
        )
        maximum_marginal_error = max(
            maximum_marginal_error,
            _max_abs(marginals[0] - brute_marginals),
        )
        maximum_covariance_error = max(
            maximum_covariance_error,
            _max_abs(covariance[0] - brute_covariance),
        )
        maximum_identity_error = max(
            maximum_identity_error, diagnostics["maximum_identity_error"]
        )
    if maximum_marginal_error > 1.0e-12 or maximum_covariance_error > 2.0e-12:
        raise AssertionError("exact pair moments disagree with exhaustive subsets")
    return {
        "status": "pass",
        "panel_size": p,
        "K_values_checked": list(range(p + 1)),
        "maximum_marginal_absolute_error": maximum_marginal_error,
        "maximum_covariance_absolute_error": maximum_covariance_error,
        "maximum_fixed_K_identity_error": maximum_identity_error,
    }


def _multinomial_K1_covariance_self_test() -> dict[str, Any]:
    eta = np.asarray([[0.2, -0.8, 0.4, 1.1, -0.1, 0.0]])
    covariance, marginals, _ = _pair_moments_exact_dp(
        eta, np.asarray([1], dtype=int)
    )
    expected = np.diag(marginals[0]) - np.outer(marginals[0], marginals[0])
    error = _max_abs(covariance[0] - expected)
    if error > 1.0e-13:
        raise AssertionError("K=1 covariance did not reduce to multinomial form")
    return {"status": "pass", "maximum_absolute_error": error}


def _hessian_derivative_self_test() -> dict[str, Any]:
    rng = np.random.default_rng(260927)
    n, d, p = 21, 3, 6
    q = p - 1
    x = rng.normal(size=(n, d))
    x[:, 0] = 1.0
    contrasts = base.helmert(p)
    k_values = 1 + (np.arange(n, dtype=int) % (p - 1))
    y = np.zeros((n, p), dtype=float)
    for row, k in enumerate(k_values):
        y[row, (np.arange(k) + row) % p] = 1.0
    groups = [f"g{index % 4}" for index in range(n)]
    vector = rng.normal(scale=0.13, size=d * q)
    ridge_lambda = 1.0e-3
    hessian, diagnostics = _exact_hessian_v6(
        vector, x, k_values, groups, contrasts, ridge_lambda
    )
    epsilon = 2.0e-5
    finite_difference = np.empty_like(hessian)
    for column in range(vector.size):
        perturbation = np.zeros_like(vector)
        perturbation[column] = epsilon
        _, gradient_plus = base.objective_and_gradient(
            vector + perturbation,
            x,
            y,
            k_values,
            groups,
            contrasts,
            ridge_lambda,
        )
        _, gradient_minus = base.objective_and_gradient(
            vector - perturbation,
            x,
            y,
            k_values,
            groups,
            contrasts,
            ridge_lambda,
        )
        finite_difference[:, column] = (
            gradient_plus - gradient_minus
        ) / (2.0 * epsilon)
    derivative_error = _max_abs(hessian - finite_difference)

    direction = rng.normal(size=vector.size)
    coefficients = vector.reshape(d, q)
    eta = (x @ coefficients) @ contrasts.T
    covariance, _, _ = _pair_moments_exact_dp(eta, k_values)
    delta = direction.reshape(d, q)
    delta_eta = (x @ delta) @ contrasts.T
    weights = base.equal_group_weights(groups)
    variance_form = float(
        np.einsum(
            "n,nt,ntu,nu->",
            weights,
            delta_eta,
            covariance,
            delta_eta,
            optimize=True,
        )
        + ridge_lambda * np.dot(direction, direction)
    )
    matrix_form = float(np.dot(direction, hessian @ direction))
    quadratic_form_error = abs(matrix_form - variance_form)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(hessian)))
    if derivative_error > 2.0e-8:
        raise AssertionError("analytic Hessian failed the gradient finite difference")
    if quadratic_form_error > 2.0e-10:
        raise AssertionError("Hessian quadratic form failed variance identity")
    if minimum_eigenvalue < ridge_lambda - 2.0e-12:
        raise AssertionError("exact Hessian violated the ridge lower bound")
    return {
        "status": "pass",
        "dimension": int(vector.size),
        "finite_difference_max_absolute_error": derivative_error,
        "variance_quadratic_form_absolute_error": quadratic_form_error,
        "minimum_eigenvalue": minimum_eigenvalue,
        "ridge_lower_bound": ridge_lambda,
        "pair_DP_maximum_identity_error": diagnostics["maximum_identity_error"],
    }


def _full_dimension_SPD_self_test() -> dict[str, Any]:
    rng = np.random.default_rng(255)
    n, d, p = 36, 17, 16
    x = rng.normal(size=(n, d))
    x[:, 0] = 1.0
    k_values = 1 + (np.arange(n, dtype=int) % 15)
    groups = [f"g{index % 6}" for index in range(n)]
    contrasts = base.helmert(p)
    vector = rng.normal(scale=0.04, size=d * (p - 1))
    ridge_lambda = 1.0e-8
    hessian, diagnostics = _exact_hessian_v6(
        vector, x, k_values, groups, contrasts, ridge_lambda
    )
    eigenvalues = np.linalg.eigvalsh(hessian)
    minimum = float(eigenvalues[0])
    gradient = rng.normal(scale=1.0e-4, size=vector.size)
    direction, solve = _newton_direction_v6(hessian, gradient, ridge_lambda)
    if hessian.shape != (PARAMETER_DIMENSION, PARAMETER_DIMENSION):
        raise AssertionError("full-dimension Hessian shape changed")
    if minimum < ridge_lambda - 2.0e-11:
        raise AssertionError("full Hessian violated its ridge eigenvalue bound")
    if not float(np.dot(gradient, direction)) < 0.0:
        raise AssertionError("full Hessian solve was not descent")
    return {
        "status": "pass",
        "parameter_dimension": PARAMETER_DIMENSION,
        "minimum_eigenvalue": minimum,
        "ridge_lower_bound": ridge_lambda,
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "pair_DP_maximum_identity_error": diagnostics["maximum_identity_error"],
        "solve_scaled_backward_error": solve["scaled_backward_error"],
        "directional_derivative": solve["directional_derivative"],
    }


def _solve_fail_closed_self_test() -> dict[str, Any]:
    eigenvalues = np.geomspace(1.0e-8, 1.0, 12)
    hessian = np.diag(eigenvalues)
    gradient = np.linspace(-0.4, 0.7, 12)
    direction, diagnostics = _newton_direction_v6(
        hessian, gradient, 1.0e-8
    )
    exact = -gradient / eigenvalues
    relative_solution_error = float(
        np.linalg.norm(direction - exact) / np.linalg.norm(exact)
    )
    if relative_solution_error > 1.0e-12:
        raise AssertionError("known diagonal Newton solve was inaccurate")

    negative_controls: list[str] = []
    cases = {
        "nonfinite": np.where(np.eye(12, dtype=bool), math.nan, hessian),
        "asymmetric": hessian + np.triu(np.ones_like(hessian), 1) * 1.0e-3,
        "indefinite_beyond_damping_bound": np.diag(
            np.concatenate(([-1.0], np.ones(11)))
        ),
    }
    for label, candidate in cases.items():
        try:
            _newton_direction_v6(candidate, gradient, 1.0e-8)
        except base.ModelError:
            negative_controls.append(label)
        else:
            raise AssertionError(f"Newton negative control {label} was accepted")
    return {
        "status": "pass",
        "known_diagonal_relative_solution_error": relative_solution_error,
        "scaled_backward_error": diagnostics["scaled_backward_error"],
        "directional_derivative": diagnostics["directional_derivative"],
        "negative_controls": negative_controls,
    }


def _strong_grid_v5_parity_self_test() -> dict[str, Any]:
    contrasts = base.helmert(6)
    x = np.ones((40, 1), dtype=float)
    groups = [f"g{index % 5}" for index in range(40)]
    k_values = 1 + (np.arange(40, dtype=int) % 5)
    planted = np.linspace(-0.31, 0.29, 5)[None, :]
    eta = (x @ planted) @ contrasts.T
    _, soft_targets = base.batch_fixed_k_logz_marginals(eta, k_values)
    expected_path: dict[float, v3.OptimizerResultV3] = {}
    expected_initial: np.ndarray | None = None
    FIT_TELEMETRY.clear()
    for ridge_lambda in STRONG_V5_LAMBDAS:
        expected_path[ridge_lambda] = V5_FIT_LBFGS_ORIGINAL(
            x,
            soft_targets,
            k_values,
            groups,
            contrasts,
            ridge_lambda,
            initial=expected_initial,
        )
        expected_initial = expected_path[ridge_lambda].coefficients

    observed_path: dict[float, v3.OptimizerResultV3] = {}
    observed_initial: np.ndarray | None = None
    FIT_TELEMETRY.clear()
    for ridge_lambda in STRONG_V5_LAMBDAS:
        observed_path[ridge_lambda] = fit_lbfgs_v6(
            x,
            soft_targets,
            k_values,
            groups,
            contrasts,
            ridge_lambda,
            initial=observed_initial,
        )
        observed_initial = observed_path[ridge_lambda].coefficients

    coefficient_equal = all(
        np.array_equal(
            expected_path[value].coefficients,
            observed_path[value].coefficients,
        )
        for value in STRONG_V5_LAMBDAS
    )
    scalar_equal = all(
        getattr(expected_path[value], field)
        == getattr(observed_path[value], field)
        for value in STRONG_V5_LAMBDAS
        for field in (
            "objective",
            "gradient_infinity_norm",
            "iterations",
            "accepted_steps",
            "relative_objective_change",
        )
    )
    telemetry_equal = all(
        expected_path[value].solver_telemetry
        == observed_path[value].solver_telemetry
        for value in STRONG_V5_LAMBDAS
    )
    FIT_TELEMETRY.clear()
    if not coefficient_equal or not scalar_equal or not telemetry_equal:
        raise AssertionError("V6 strong-grid forwarding was not bitwise V5 parity")
    return {
        "status": "pass",
        "ridge_lambdas": list(STRONG_V5_LAMBDAS),
        "fits_compared": len(STRONG_V5_LAMBDAS),
        "coefficient_arrays_bitwise_equal": coefficient_equal,
        "result_scalars_bitwise_equal": scalar_equal,
        "telemetry_equal": telemetry_equal,
        "maximum_iterations": max(
            result.iterations for result in observed_path.values()
        ),
    }


def _weak_newton_recovery_self_test() -> dict[str, Any]:
    contrasts = base.helmert(6)
    x = np.ones((50, 1), dtype=float)
    groups = [f"g{index % 5}" for index in range(50)]
    k_values = 1 + (np.arange(50, dtype=int) % 5)
    planted = np.linspace(-0.55, 0.47, 5)[None, :]
    eta = (x @ planted) @ contrasts.T
    _, soft_targets = base.batch_fixed_k_logz_marginals(eta, k_values)
    ridge_lambda = 1.0e-6
    FIT_TELEMETRY.clear()
    first = fit_lbfgs_v6(
        x, soft_targets, k_values, groups, contrasts, ridge_lambda
    )
    first_telemetry = dict(first.solver_telemetry)
    FIT_TELEMETRY.clear()
    second = fit_lbfgs_v6(
        x, soft_targets, k_values, groups, contrasts, ridge_lambda
    )
    repeat_error = _max_abs(first.coefficients - second.coefficients)
    if (
        repeat_error != 0.0
        or first.objective != second.objective
        or first.gradient_infinity_norm > GRADIENT_TOLERANCE
        or first.relative_objective_change > RELATIVE_OBJECTIVE_TOLERANCE
        or first_telemetry.get("strong_wolfe_all_accepted_steps_pass") is not True
        or first_telemetry.get("solver_branch")
        != "weak_grid_exact_Hessian_Newton"
    ):
        raise AssertionError("weak-grid exact-Hessian recovery regression failed")
    FIT_TELEMETRY.clear()
    return {
        "status": "pass",
        "ridge_lambda": ridge_lambda,
        "iterations": first.iterations,
        "gradient_infinity_norm": first.gradient_infinity_norm,
        "relative_objective_change": first.relative_objective_change,
        "repeat_coefficient_max_absolute_error": repeat_error,
        "all_accepted_steps_satisfied_exact_strong_Wolfe": True,
    }


def run_newton_unit_tests() -> dict[str, Any]:
    result = {
        "status": "pass",
        "stage": "v3_M0_solver_v6_exact_Hessian_unit_tests",
        "exhaustive_pair_moments": _exhaustive_pair_moment_self_test(),
        "multinomial_K1_covariance": _multinomial_K1_covariance_self_test(),
        "Hessian_derivative_and_variance_identity": (
            _hessian_derivative_self_test()
        ),
        "full_255_parameter_SPD": _full_dimension_SPD_self_test(),
        "Newton_solve_and_fail_closed_controls": (
            _solve_fail_closed_self_test()
        ),
        "strong_grid_V5_bitwise_parity": _strong_grid_v5_parity_self_test(),
        "weak_grid_deterministic_recovery": _weak_newton_recovery_self_test(),
        "uses_real_data": False,
        "final_or_unassigned_opened": False,
    }
    FIT_TELEMETRY.clear()
    return result


def run_self_test(
    base_contract_path: Path, addendum_path: Path
) -> dict[str, Any]:
    validate_solver_addendum(base_contract_path, addendum_path)
    newton_tests = run_newton_unit_tests()
    predecessor = v5.run_self_test(base_contract_path, V5_ADDENDUM_PATH)
    FIT_TELEMETRY.clear()
    return {
        "status": "pass",
        "stage": "v3_M0_solver_v6_synthetic_self_test",
        "v6_exact_Hessian_tests": newton_tests,
        "retained_v5_self_test": predecessor,
        "uses_real_data": False,
        "final_or_unassigned_opened": False,
    }


def _prepare_unique_output(path: Path, scratch: Path, *, mode: str) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise base.ModelError("real v6 solver work must run in a Slurm job")
    episode_root = v3._require_unsymlinked_directory(
        scratch / base.EPISODE_SCRATCH_RELATIVE, "EP10 Scratch root"
    )
    if not path.is_absolute():
        raise base.ModelError("v6 output root must be absolute")
    absolute = Path(os.path.abspath(path))
    expected_names = {
        "weak_prefix": f"v3_m0_solver_weak_prefix_v6_job_{job_id}",
        "full_probe": f"v3_m0_solver_probe_v6_job_{job_id}",
        "execute": PRODUCTION_OUTPUT_BASENAME,
    }
    expected_name = expected_names[mode]
    if absolute.parent != episode_root or absolute.name != expected_name:
        raise base.ModelError(
            f"v6 output must be the exact EP10 Scratch child {expected_name!r}"
        )
    if os.path.lexists(absolute):
        raise base.ModelError("v6 output root must be absent")
    if mode != "execute":
        absolute.mkdir(parents=False)
        if absolute != absolute.resolve(strict=True) or absolute.is_symlink():
            raise base.ModelError("v6 probe output root failed its path seal")
    return absolute


def _load_exact_probe_training(
    target_root: Path, interface_root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    x, y, k_values, groups = v5._load_exact_probe_training(
        target_root, interface_root
    )
    if x.shape != (PROBE_EXPECTED_CELLS, 17):
        raise base.ModelError("targeted v6 probe design shape changed")
    return x, y, k_values, groups


def _result_has_valid_witness(result: v3.OptimizerResultV3) -> bool:
    literal = bool(
        result.gradient_infinity_norm <= GRADIENT_TOLERANCE
        and result.relative_objective_change <= RELATIVE_OBJECTIVE_TOLERANCE
    )
    certificate = result.solver_telemetry.get(
        "strong_convexity_certificate", {}
    )
    certified = bool(
        result.gradient_infinity_norm <= GRADIENT_TOLERANCE
        and certificate.get("certificate_pass") is True
        and certificate.get("no_pseudo_step") is True
    )
    return literal or certified


def _fit_warm_prefix(
    x: np.ndarray,
    y: np.ndarray,
    k_values: np.ndarray,
    groups: Sequence[str],
) -> dict[float, v3.OptimizerResultV3]:
    contrasts = base.helmert(16)
    initial: np.ndarray | None = None
    results: dict[float, v3.OptimizerResultV3] = {}
    for ridge_lambda in WEAK_PREFIX_LAMBDAS:
        result = fit_lbfgs_v6(
            x,
            y,
            k_values,
            groups,
            contrasts,
            ridge_lambda,
            initial=initial,
        )
        results[ridge_lambda] = result
        initial = result.coefficients
    return results


def run_weak_prefix_probe(args: argparse.Namespace) -> dict[str, Any]:
    validate_solver_addendum(args.contract, args.solver_addendum)
    scratch, target_root, interface_root = v3._validate_real_input_roots(
        args.target_root, args.interface_root
    )
    output_root = _prepare_unique_output(
        args.output_root, scratch, mode="weak_prefix"
    )
    FIT_TELEMETRY.clear()
    payload: dict[str, Any] = {
        "status": "fail",
        "stage": "v3_M0_solver_v6_weak_grid_warm_prefix_probe",
        "solver_addendum": args.solver_addendum.name,
        "outer_held_group": PROBE_OUTER_HELD_GROUP,
        "inner_held_group": PROBE_INNER_HELD_GROUP,
        "held_out_scores_emitted": False,
        "coefficient_prediction_or_target_contrast_vectors_emitted": False,
        "final_or_unassigned_opened": False,
        "output_root": str(output_root),
    }
    output_path = output_root / "v3_m0_solver_weak_prefix_v6.json"
    try:
        x, y, k_values, groups = _load_exact_probe_training(
            target_root, interface_root
        )
        results = _fit_warm_prefix(x, y, k_values, groups)
        required = [results[value] for value in WEAK_PREFIX_REQUIRED]
        if not all(
            result.iterations < WARM_PATH_MAX_ITERATIONS_EXCLUSIVE
            for result in required
        ):
            raise base.ModelError(
                "v6 weak-prefix fit lacks frozen iteration headroom"
            )
        if not all(_result_has_valid_witness(result) for result in required):
            raise base.ModelError("v6 weak-prefix fit lacks a convergence witness")
        if not all(
            result.solver_telemetry.get("solver_branch")
            == "weak_grid_exact_Hessian_Newton"
            and result.solver_telemetry.get(
                "strong_wolfe_all_accepted_steps_pass"
            )
            is True
            for result in required
        ):
            raise base.ModelError("v6 weak-prefix branch or Wolfe gate failed")
        payload.update(
            {
                "status": "pass",
                "training_cells": int(x.shape[0]),
                "training_groups": int(len(set(groups))),
                "descending_warm_prefix": list(WEAK_PREFIX_LAMBDAS),
                "required_weak_lambdas": list(WEAK_PREFIX_REQUIRED),
                "fit_telemetry": [
                    results[value].solver_telemetry
                    for value in WEAK_PREFIX_LAMBDAS
                ],
                "weak_lambda_iterations": {
                    format(value, ".0e"): results[value].iterations
                    for value in WEAK_PREFIX_REQUIRED
                },
                "weak_lambda_headroom_pass": True,
                "all_required_weak_fits_met_unchanged_convergence_witness": True,
                "all_accepted_steps_satisfied_exact_strong_Wolfe": True,
            }
        )
        base._atomic_json(output_path, payload)
        return payload
    except (base.ModelError, AssertionError, OSError, ValueError) as error:
        payload.update(
            {
                "status": "fail",
                "error": str(error),
                "completed_fit_telemetry": list(FIT_TELEMETRY),
            }
        )
        base._atomic_json(output_path, payload)
        raise


def run_targeted_probe(args: argparse.Namespace) -> dict[str, Any]:
    validate_solver_addendum(args.contract, args.solver_addendum)
    scratch, target_root, interface_root = v3._validate_real_input_roots(
        args.target_root, args.interface_root
    )
    output_root = _prepare_unique_output(
        args.output_root, scratch, mode="full_probe"
    )
    FIT_TELEMETRY.clear()
    payload: dict[str, Any] = {
        "status": "fail",
        "stage": "v3_M0_solver_v6_full_targeted_numerical_probe",
        "solver_addendum": args.solver_addendum.name,
        "requires_prior_weak_prefix_probe_pass": True,
        "outer_held_group": PROBE_OUTER_HELD_GROUP,
        "inner_held_group": PROBE_INNER_HELD_GROUP,
        "held_out_scores_emitted": False,
        "coefficient_prediction_or_target_contrast_vectors_emitted": False,
        "final_or_unassigned_opened": False,
        "output_root": str(output_root),
    }
    output_path = output_root / "v3_m0_solver_probe_v6.json"
    try:
        x, y, k_values, groups = _load_exact_probe_training(
            target_root, interface_root
        )
        warm_first = v2._fit_probe_path(
            x, y, k_values, groups, warm_start=True
        )
        warm_second = v2._fit_probe_path(
            x, y, k_values, groups, warm_start=True
        )
        zero_start = v2._fit_probe_path(
            x, y, k_values, groups, warm_start=False
        )
        repeat_checks, zero_checks = v2._compare_probe_paths(
            warm_first, warm_second, zero_start
        )
        all_results = [
            result
            for path in (warm_first, warm_second, zero_start)
            for result, _ in path.values()
        ]
        warm_path_worst_iterations = max(
            result.iterations
            for path in (warm_first, warm_second)
            for result, _ in path.values()
        )
        if warm_path_worst_iterations >= WARM_PATH_MAX_ITERATIONS_EXCLUSIVE:
            raise base.ModelError(
                "v6 probe lacks frozen warm-path iteration headroom: "
                f"worst={warm_path_worst_iterations}, required <"
                f"{WARM_PATH_MAX_ITERATIONS_EXCLUSIVE}"
            )
        if not all(
            result.solver_telemetry.get("strong_wolfe_all_accepted_steps_pass")
            is True
            for result in all_results
        ):
            raise base.ModelError("a v6 probe fit accepted a non-Wolfe step")
        if not all(_result_has_valid_witness(result) for result in all_results):
            raise base.ModelError("a v6 probe fit lacks a convergence witness")
        if not all(item["pass"] for item in repeat_checks):
            raise base.ModelError("v6 deterministic repeat check failed")
        if not all(item["pass"] for item in zero_checks):
            raise base.ModelError("v6 warm-start versus zero-start check failed")
        for path in (warm_first, warm_second, zero_start):
            for ridge_lambda in WEAK_NEWTON_LAMBDAS:
                telemetry = path[ridge_lambda][0].solver_telemetry
                if telemetry.get("solver_branch") != (
                    "weak_grid_exact_Hessian_Newton"
                ):
                    raise base.ModelError("a weak-grid fit did not use V6 Newton")

        payload.update(
            {
                "status": "pass",
                "training_cells": int(x.shape[0]),
                "training_groups": int(len(set(groups))),
                "descending_lambda_order": list(DESCENDING_LAMBDAS),
                "warm_path_repeat_1": [
                    warm_first[value][0].solver_telemetry
                    for value in DESCENDING_LAMBDAS
                ],
                "warm_path_repeat_2": [
                    warm_second[value][0].solver_telemetry
                    for value in DESCENDING_LAMBDAS
                ],
                "zero_start_path": [
                    zero_start[value][0].solver_telemetry
                    for value in DESCENDING_LAMBDAS
                ],
                "deterministic_repeat_checks": repeat_checks,
                "warm_start_vs_zero_start_checks": zero_checks,
                "warm_path_worst_iterations": warm_path_worst_iterations,
                "warm_path_max_iterations_exclusive": (
                    WARM_PATH_MAX_ITERATIONS_EXCLUSIVE
                ),
                "warm_path_iteration_headroom_pass": True,
                "all_fits_met_gradient_and_literal_or_certified_objective_accuracy": True,
                "all_accepted_steps_satisfied_exact_strong_Wolfe": True,
                "all_weak_grid_fits_used_exact_Hessian_Newton": True,
            }
        )
        base._atomic_json(output_path, payload)
        return payload
    except (base.ModelError, AssertionError, OSError, ValueError) as error:
        payload.update(
            {
                "status": "fail",
                "error": str(error),
                "completed_fit_telemetry": list(FIT_TELEMETRY),
            }
        )
        base._atomic_json(output_path, payload)
        raise


def run_execute(args: argparse.Namespace) -> dict[str, Any]:
    validate_solver_addendum(args.contract, args.solver_addendum)
    scratch, target_root, interface_root = v3._validate_real_input_roots(
        args.target_root, args.interface_root
    )
    output_root = _prepare_unique_output(args.output_root, scratch, mode="execute")
    args.target_root = target_root
    args.interface_root = interface_root
    args.output_root = output_root
    FIT_TELEMETRY.clear()
    summary = base.run_real(args)
    summary["solver_addendum_v6"] = {
        "schema_version": "episode10_v3_m0_solver_addendum_v6",
        "weak_grid_exact_Hessian_Newton_lambdas": list(WEAK_NEWTON_LAMBDAS),
        "strong_grid_V5_bitwise_forward": True,
        "maximum_iterations": MAXIMUM_ITERATIONS,
        "v4_line_search_unchanged": True,
        "v5_certificate_unchanged": True,
        "convergence_tolerances_changed": False,
    }
    base._atomic_json(output_root / "v3_m0_fit_summary.json", summary)
    base._atomic_json(
        output_root / "v3_m0_solver_telemetry_v6.json",
        {
            "status": "pass",
            "solver_addendum": args.solver_addendum.name,
            "fits": list(FIT_TELEMETRY),
            "coefficient_prediction_or_target_contrast_vectors_emitted": False,
            "final_or_unassigned_opened": False,
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--unit-test", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--weak-prefix-probe", action="store_true")
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--contract", type=Path, default=BASE_CONTRACT_PATH)
    parser.add_argument(
        "--solver-addendum", type=Path, default=V6_ADDENDUM_PATH
    )
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--interface-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--temperature-diagnostic", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.unit_test or args.self_test:
        if any(
            value is not None
            for value in (args.target_root, args.interface_root, args.output_root)
        ):
            parser.error("test modes do not accept real input or output paths")
        if args.temperature_diagnostic:
            parser.error("--temperature-diagnostic is unavailable in test modes")
        try:
            validate_solver_addendum(args.contract, args.solver_addendum)
            result = (
                run_newton_unit_tests()
                if args.unit_test
                else run_self_test(args.contract, args.solver_addendum)
            )
        except (base.ModelError, AssertionError, OSError, ValueError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    missing = [
        name
        for name in ("target_root", "interface_root", "output_root")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            "selected real-data mode requires: "
            + ", ".join("--" + value.replace("_", "-") for value in missing)
        )
    if (args.probe or args.weak_prefix_probe) and args.temperature_diagnostic:
        parser.error("numerical probes cannot run the temperature diagnostic")
    try:
        if args.weak_prefix_probe:
            result = run_weak_prefix_probe(args)
        elif args.probe:
            result = run_targeted_probe(args)
        else:
            result = run_execute(args)
        if args.weak_prefix_probe or args.probe:
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "stage": result["stage"],
                        "output_root": result["output_root"],
                    },
                    sort_keys=True,
                )
            )
    except base.BoundaryReview as error:
        print(f"CONTRACT REVIEW REQUIRED: {error}", file=sys.stderr)
        return 3
    except (base.ModelError, AssertionError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
