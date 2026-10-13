#!/usr/bin/env python3
"""Run EP10 v3 M0 with the V7 four-value exact-Hessian route.

V7 is a thin routing wrapper.  Exactly 1e-5, 1e-6, 1e-7, and 1e-8 use the
unchanged V6 exact-Hessian Newton core.  The other eight ridge-grid values are
forwarded directly to immutable V5.  No scientific model, objective, ridge
grid, warm start, line search, certificate, tolerance, or iteration cap is
changed.
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
    raise ImportError("v7 solver wrapper must be an unsymlinked regular file")
SCRIPT_DIR = SCRIPT_PATH.parent
STAGE0_DIR = SCRIPT_DIR.parent
V6_ENGINE_PATH = SCRIPT_DIR / "fit_v3_m0_v6.py"
if (
    not V6_ENGINE_PATH.is_file()
    or V6_ENGINE_PATH.is_symlink()
    or V6_ENGINE_PATH != V6_ENGINE_PATH.resolve(strict=True)
):
    raise ImportError("v6 predecessor wrapper must be an unsymlinked stage0 file")

_V6_SPEC = importlib.util.spec_from_file_location(
    "ep10_v3_m0_solver_v6_for_v7", V6_ENGINE_PATH
)
if _V6_SPEC is None or _V6_SPEC.loader is None:
    raise ImportError(f"cannot import v6 solver wrapper: {V6_ENGINE_PATH}")
v6 = importlib.util.module_from_spec(_V6_SPEC)
sys.modules[_V6_SPEC.name] = v6
_V6_SPEC.loader.exec_module(v6)
v5 = v6.v5
v4 = v6.v4
v3 = v6.v3
v2 = v6.v2
base = v6.base


BASE_CONTRACT_PATH = STAGE0_DIR / "v3_m0_model_contract.yaml"
V6_ADDENDUM_PATH = STAGE0_DIR / "v3_m0_solver_addendum_v6.yaml"
V7_ADDENDUM_PATH = STAGE0_DIR / "v3_m0_solver_addendum_v7.yaml"

PARAMETER_DIMENSION = v6.PARAMETER_DIMENSION
HISTORY_SIZE = v6.HISTORY_SIZE
MAXIMUM_ITERATIONS = v6.MAXIMUM_ITERATIONS
GRADIENT_TOLERANCE = v6.GRADIENT_TOLERANCE
RELATIVE_OBJECTIVE_TOLERANCE = v6.RELATIVE_OBJECTIVE_TOLERANCE
DESCENDING_LAMBDAS = v6.DESCENDING_LAMBDAS
NEWTON_LAMBDAS = (1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8)
V5_FORWARD_LAMBDAS = tuple(
    value for value in DESCENDING_LAMBDAS if value not in NEWTON_LAMBDAS
)
LAMBDA_1E5 = 1.0e-5
LAMBDA_1E5_WARM_PREFIX = tuple(
    value for value in DESCENDING_LAMBDAS if value >= LAMBDA_1E5
)

PROBE_OUTER_HELD_GROUP = v6.PROBE_OUTER_HELD_GROUP
PROBE_INNER_HELD_GROUP = v6.PROBE_INNER_HELD_GROUP
PROBE_EXPECTED_CELLS = v6.PROBE_EXPECTED_CELLS
PROBE_EXPECTED_GROUPS = v6.PROBE_EXPECTED_GROUPS
WARM_PATH_MAX_ITERATIONS_EXCLUSIVE = v6.WARM_PATH_MAX_ITERATIONS_EXCLUSIVE
PRODUCTION_OUTPUT_BASENAME = "v3_m0_fit_v7"

V5_FIT_LBFGS_ORIGINAL = v6.V5_FIT_LBFGS_ORIGINAL
FIT_TELEMETRY = v6.FIT_TELEMETRY


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


def validate_solver_addendum(
    base_contract_path: Path, addendum_path: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    _require_exact_unsymlinked_file(
        base_contract_path, BASE_CONTRACT_PATH, "v1 M0 model contract"
    )
    _require_exact_unsymlinked_file(
        V6_ENGINE_PATH, V6_ENGINE_PATH, "v6 predecessor wrapper"
    )
    _require_exact_unsymlinked_file(
        V6_ADDENDUM_PATH, V6_ADDENDUM_PATH, "v6 predecessor addendum"
    )
    _require_exact_unsymlinked_file(
        addendum_path, V7_ADDENDUM_PATH, "v7 solver addendum"
    )
    base_contract, predecessor = v6.validate_solver_addendum(
        base_contract_path, V6_ADDENDUM_PATH
    )
    addendum = base._load_yaml(addendum_path)
    if (
        addendum.get("schema_version")
        != "episode10_v3_m0_solver_addendum_v7"
        or addendum.get("status")
        != "frozen_before_v7_lambda_1e5_equivalence_probe_and_execution"
        or addendum.get("scientific_authority") is not False
        or addendum.get("scope", {}).get("role") != "development_only"
        or addendum.get("scope", {}).get("final_or_unassigned_access")
        != "forbidden"
    ):
        raise base.ModelError("v7 addendum identity or sealed scope changed")
    reference = addendum.get("predecessor_solver_addendum", {})
    if (
        reference.get("required_schema_version")
        != predecessor.get("schema_version")
        or reference.get("required_status") != predecessor.get("status")
    ):
        raise base.ModelError("v7 predecessor reference changed")

    solver = addendum.get("solver", {})
    routing = solver.get("routing", {})
    convergence = solver.get("convergence", {})
    if (
        solver.get("algorithm") != "deterministic_versioned_hybrid"
        or int(solver.get("parameter_dimension", -1)) != PARAMETER_DIMENSION
        or int(solver.get("maximum_iterations", -1)) != MAXIMUM_ITERATIONS
        or tuple(
            float(value)
            for value in routing.get("immutable_V5_bitwise_forward", ())
        )
        != V5_FORWARD_LAMBDAS
        or tuple(
            float(value)
            for value in routing.get("exact_Hessian_Newton", ())
        )
        != NEWTON_LAMBDAS
        or solver.get("exact_Hessian_implementation") != "unchanged_V6"
        or solver.get("line_search") != "unchanged_V4_exact_strong_Wolfe"
        or solver.get("eligible_stall_certificate") != "unchanged_V5"
        or float(convergence.get("gradient_infinity_tolerance", math.nan))
        != GRADIENT_TOLERANCE
        or float(convergence.get("relative_objective_tolerance", math.nan))
        != RELATIVE_OBJECTIVE_TOLERANCE
        or solver.get("fallback_optimizer") != "forbidden"
    ):
        raise base.ModelError("v7 routing or unchanged numerical settings drifted")

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
        raise base.ModelError("v7 ridge path or warm semantics changed")

    narrow = addendum.get("narrow_lambda_1e5_equivalence_probe", {})
    full = addendum.get("full_targeted_probe", {})
    comparison = narrow.get("comparison", {})
    if (
        str(narrow.get("outer_held_group")) != PROBE_OUTER_HELD_GROUP
        or str(narrow.get("inner_held_group")) != PROBE_INNER_HELD_GROUP
        or int(narrow.get("expected_training_cells", -1))
        != PROBE_EXPECTED_CELLS
        or int(narrow.get("expected_training_groups", -1))
        != PROBE_EXPECTED_GROUPS
        or float(narrow.get("warm_path_stop_lambda_inclusive", math.nan))
        != LAMBDA_1E5
        or float(narrow.get("isolated_zero_start_lambda", math.nan))
        != LAMBDA_1E5
        or float(
            comparison.get("marginal_max_absolute_tolerance", math.nan)
        )
        != v2.ZERO_START_MARGINAL_TOLERANCE
        or float(
            comparison.get(
                "log_probability_max_absolute_tolerance", math.nan
            )
        )
        != v2.ZERO_START_LOG_PROBABILITY_TOLERANCE
        or full.get(
            "runs_only_after_narrow_lambda_1e5_equivalence_probe_passes"
        )
        is not True
        or int(full.get("deterministic_warm_path_repeats", -1)) != 2
        or full.get("zero_start_comparison_at_every_lambda") is not True
        or int(full.get("warm_path_max_iterations_exclusive", -1))
        != WARM_PATH_MAX_ITERATIONS_EXCLUSIVE
        or narrow.get("final_or_unassigned_access") != "forbidden"
        or full.get("final_or_unassigned_access") != "forbidden"
    ):
        raise base.ModelError("v7 probe definition changed")
    return base_contract, addendum


def _v7_newton_result(result: v3.OptimizerResultV3) -> v3.OptimizerResultV3:
    telemetry = dict(result.solver_telemetry)
    telemetry.update(
        {
            "solver_version": "v7_four_value_exact_Hessian_Newton",
            "solver_branch": "v7_exact_Hessian_Newton",
            "v7_exact_Hessian_Newton_lambda_count": len(NEWTON_LAMBDAS),
            "v7_added_lambda_1e5_to_V6_Newton_route": True,
        }
    )
    telemetry.pop("weak_grid_lambdas", None)
    if FIT_TELEMETRY and FIT_TELEMETRY[-1] is result.solver_telemetry:
        FIT_TELEMETRY[-1] = telemetry
    else:
        raise base.ModelError("v7 could not replace inherited V6 telemetry")
    return v3.OptimizerResultV3(
        coefficients=result.coefficients,
        objective=result.objective,
        gradient_infinity_norm=result.gradient_infinity_norm,
        iterations=result.iterations,
        accepted_steps=result.accepted_steps,
        relative_objective_change=result.relative_objective_change,
        solver_telemetry=telemetry,
    )


def fit_lbfgs_v7(
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
    dimension = x.shape[1] * target_contrasts.shape[1]
    if dimension != PARAMETER_DIMENSION and not (
        x.shape[1] == 1 and target_contrasts.shape[0] == 6
    ):
        raise base.ModelError(f"unexpected v7 optimizer dimension {dimension}")
    if (
        history_size != HISTORY_SIZE
        or maximum_iterations != MAXIMUM_ITERATIONS
        or gradient_tolerance != GRADIENT_TOLERANCE
        or relative_objective_tolerance != RELATIVE_OBJECTIVE_TOLERANCE
    ):
        raise base.ModelError("a caller attempted to override frozen v7 settings")
    ridge = float(ridge_lambda)
    if ridge not in NEWTON_LAMBDAS:
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
    inherited = v6._fit_newton_v6_core(
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
    return _v7_newton_result(inherited)


base.fit_lbfgs = fit_lbfgs_v7
v2.fit_lbfgs_v2 = fit_lbfgs_v7
v2.FIT_TELEMETRY = FIT_TELEMETRY


def _synthetic_problem() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray
]:
    contrasts = base.helmert(6)
    x = np.ones((50, 1), dtype=float)
    groups = [f"g{index % 5}" for index in range(50)]
    k_values = 1 + (np.arange(50, dtype=int) % 5)
    planted = np.linspace(-0.55, 0.47, 5)[None, :]
    eta = (x @ planted) @ contrasts.T
    _, soft_targets = base.batch_fixed_k_logz_marginals(eta, k_values)
    return x, soft_targets, k_values, groups, contrasts


def _routing_self_test() -> dict[str, Any]:
    if NEWTON_LAMBDAS != (1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8):
        raise AssertionError("V7 Newton routing set changed")
    if V5_FORWARD_LAMBDAS != DESCENDING_LAMBDAS[:8]:
        raise AssertionError("V7 V5-forward routing set changed")
    return {
        "status": "pass",
        "Newton_lambda_count": len(NEWTON_LAMBDAS),
        "V5_forward_lambda_count": len(V5_FORWARD_LAMBDAS),
        "lambda_1e5_uses_Newton": LAMBDA_1E5 in NEWTON_LAMBDAS,
    }


def _eight_lambda_v5_parity_self_test() -> dict[str, Any]:
    x, y, k_values, groups, contrasts = _synthetic_problem()
    expected: dict[float, v3.OptimizerResultV3] = {}
    observed: dict[float, v3.OptimizerResultV3] = {}
    expected_initial: np.ndarray | None = None
    FIT_TELEMETRY.clear()
    for ridge_lambda in V5_FORWARD_LAMBDAS:
        expected[ridge_lambda] = V5_FIT_LBFGS_ORIGINAL(
            x,
            y,
            k_values,
            groups,
            contrasts,
            ridge_lambda,
            initial=expected_initial,
        )
        expected_initial = expected[ridge_lambda].coefficients
    observed_initial: np.ndarray | None = None
    FIT_TELEMETRY.clear()
    for ridge_lambda in V5_FORWARD_LAMBDAS:
        observed[ridge_lambda] = fit_lbfgs_v7(
            x,
            y,
            k_values,
            groups,
            contrasts,
            ridge_lambda,
            initial=observed_initial,
        )
        observed_initial = observed[ridge_lambda].coefficients
    coefficient_equal = all(
        np.array_equal(expected[value].coefficients, observed[value].coefficients)
        for value in V5_FORWARD_LAMBDAS
    )
    scalar_equal = all(
        getattr(expected[value], field) == getattr(observed[value], field)
        for value in V5_FORWARD_LAMBDAS
        for field in (
            "objective",
            "gradient_infinity_norm",
            "iterations",
            "accepted_steps",
            "relative_objective_change",
        )
    )
    telemetry_equal = all(
        expected[value].solver_telemetry == observed[value].solver_telemetry
        for value in V5_FORWARD_LAMBDAS
    )
    FIT_TELEMETRY.clear()
    if not coefficient_equal or not scalar_equal or not telemetry_equal:
        raise AssertionError("V7 did not forward all eight strong lambdas bitwise")
    return {
        "status": "pass",
        "fits_compared": len(V5_FORWARD_LAMBDAS),
        "coefficient_arrays_bitwise_equal": coefficient_equal,
        "result_scalars_bitwise_equal": scalar_equal,
        "telemetry_equal": telemetry_equal,
    }


def _lambda_1e5_newton_equivalence_self_test() -> dict[str, Any]:
    x, y, k_values, groups, contrasts = _synthetic_problem()
    FIT_TELEMETRY.clear()
    preceding = V5_FIT_LBFGS_ORIGINAL(
        x, y, k_values, groups, contrasts, 1.0e-4
    )
    warm = fit_lbfgs_v7(
        x,
        y,
        k_values,
        groups,
        contrasts,
        LAMBDA_1E5,
        initial=preceding.coefficients,
    )
    zero = fit_lbfgs_v7(
        x, y, k_values, groups, contrasts, LAMBDA_1E5
    )
    repeat_zero = fit_lbfgs_v7(
        x, y, k_values, groups, contrasts, LAMBDA_1E5
    )
    warm_prediction = base.predict(
        warm.coefficients, x, y, k_values, contrasts
    )
    zero_prediction = base.predict(
        zero.coefficients, x, y, k_values, contrasts
    )
    objective_difference = abs(warm.objective - zero.objective)
    coefficient_difference = float(
        np.max(np.abs(warm.coefficients - zero.coefficients))
    )
    marginal_difference = float(
        np.max(np.abs(warm_prediction.marginals - zero_prediction.marginals))
    )
    log_probability_difference = float(
        np.max(
            np.abs(
                warm_prediction.log_probability_nats
                - zero_prediction.log_probability_nats
            )
        )
    )
    objective_tolerance = v2._zero_start_objective_tolerance(LAMBDA_1E5)
    passed = bool(
        objective_difference <= objective_tolerance
        and marginal_difference <= v2.ZERO_START_MARGINAL_TOLERANCE
        and log_probability_difference
        <= v2.ZERO_START_LOG_PROBABILITY_TOLERANCE
    )
    deterministic = bool(
        np.array_equal(zero.coefficients, repeat_zero.coefficients)
        and zero.objective == repeat_zero.objective
    )
    branches = {
        warm.solver_telemetry.get("solver_branch"),
        zero.solver_telemetry.get("solver_branch"),
    }
    FIT_TELEMETRY.clear()
    if not passed or not deterministic or branches != {"v7_exact_Hessian_Newton"}:
        raise AssertionError("V7 lambda-1e-5 Newton equivalence regression failed")
    return {
        "status": "pass",
        "ridge_lambda": LAMBDA_1E5,
        "objective_absolute_difference": objective_difference,
        "objective_absolute_tolerance": objective_tolerance,
        "coefficient_max_absolute_difference_report_only": coefficient_difference,
        "marginal_max_absolute_difference": marginal_difference,
        "marginal_max_absolute_tolerance": v2.ZERO_START_MARGINAL_TOLERANCE,
        "log_probability_max_absolute_difference": log_probability_difference,
        "log_probability_max_absolute_tolerance": (
            v2.ZERO_START_LOG_PROBABILITY_TOLERANCE
        ),
        "deterministic_zero_start_repeat": deterministic,
        "pass": passed,
    }


def run_unit_tests() -> dict[str, Any]:
    result = {
        "status": "pass",
        "stage": "v3_M0_solver_v7_focused_unit_tests",
        "routing": _routing_self_test(),
        "eight_lambda_V5_bitwise_parity": (
            _eight_lambda_v5_parity_self_test()
        ),
        "lambda_1e5_Newton_equivalence": (
            _lambda_1e5_newton_equivalence_self_test()
        ),
        "retained_exact_pair_moments": v6._exhaustive_pair_moment_self_test(),
        "retained_Hessian_derivative": v6._hessian_derivative_self_test(),
        "retained_full_255_SPD_solve": v6._full_dimension_SPD_self_test(),
        "uses_real_data": False,
        "final_or_unassigned_opened": False,
    }
    FIT_TELEMETRY.clear()
    return result


def _prepare_unique_output(path: Path, scratch: Path, *, mode: str) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise base.ModelError("real v7 solver work must run in a Slurm job")
    episode_root = v3._require_unsymlinked_directory(
        scratch / base.EPISODE_SCRATCH_RELATIVE, "EP10 Scratch root"
    )
    expected_names = {
        "lambda_1e5": f"v3_m0_solver_lambda1e5_equivalence_v7_job_{job_id}",
        "full_probe": f"v3_m0_solver_probe_v7_job_{job_id}",
        "execute": PRODUCTION_OUTPUT_BASENAME,
    }
    expected = expected_names[mode]
    absolute = Path(os.path.abspath(path))
    if (
        not path.is_absolute()
        or absolute.parent != episode_root
        or absolute.name != expected
    ):
        raise base.ModelError(
            f"v7 output must be the exact EP10 Scratch child {expected!r}"
        )
    if os.path.lexists(absolute):
        raise base.ModelError("v7 output root must be absent")
    if mode != "execute":
        absolute.mkdir(parents=False)
        if absolute.is_symlink() or absolute != absolute.resolve(strict=True):
            raise base.ModelError("v7 probe output root failed its path seal")
    return absolute


def _load_exact_probe_training(
    target_root: Path, interface_root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    return v6._load_exact_probe_training(target_root, interface_root)


def _result_has_valid_witness(result: v3.OptimizerResultV3) -> bool:
    return v6._result_has_valid_witness(result)


def _fit_lambda_1e5_warm_prefix(
    x: np.ndarray,
    y: np.ndarray,
    k_values: np.ndarray,
    groups: Sequence[str],
) -> dict[float, v3.OptimizerResultV3]:
    contrasts = base.helmert(16)
    initial: np.ndarray | None = None
    results: dict[float, v3.OptimizerResultV3] = {}
    for ridge_lambda in LAMBDA_1E5_WARM_PREFIX:
        result = fit_lbfgs_v7(
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


def _lambda_1e5_comparison(
    warm: v3.OptimizerResultV3,
    zero: v3.OptimizerResultV3,
    x: np.ndarray,
    y: np.ndarray,
    k_values: np.ndarray,
) -> dict[str, float | bool]:
    contrasts = base.helmert(16)
    warm_prediction = base.predict(warm.coefficients, x, y, k_values, contrasts)
    zero_prediction = base.predict(zero.coefficients, x, y, k_values, contrasts)
    objective_difference = abs(warm.objective - zero.objective)
    coefficient_difference = float(
        np.max(np.abs(warm.coefficients - zero.coefficients))
    )
    marginal_difference = float(
        np.max(np.abs(warm_prediction.marginals - zero_prediction.marginals))
    )
    log_probability_difference = float(
        np.max(
            np.abs(
                warm_prediction.log_probability_nats
                - zero_prediction.log_probability_nats
            )
        )
    )
    objective_tolerance = v2._zero_start_objective_tolerance(LAMBDA_1E5)
    passed = bool(
        objective_difference <= objective_tolerance
        and marginal_difference <= v2.ZERO_START_MARGINAL_TOLERANCE
        and log_probability_difference
        <= v2.ZERO_START_LOG_PROBABILITY_TOLERANCE
    )
    return {
        "lambda": LAMBDA_1E5,
        "objective_absolute_difference": objective_difference,
        "objective_absolute_tolerance": objective_tolerance,
        "coefficient_max_absolute_difference_report_only": coefficient_difference,
        "marginal_max_absolute_difference": marginal_difference,
        "marginal_max_absolute_tolerance": v2.ZERO_START_MARGINAL_TOLERANCE,
        "log_probability_max_absolute_difference": log_probability_difference,
        "log_probability_max_absolute_tolerance": (
            v2.ZERO_START_LOG_PROBABILITY_TOLERANCE
        ),
        "pass": passed,
    }


def run_lambda_1e5_probe(args: argparse.Namespace) -> dict[str, Any]:
    validate_solver_addendum(args.contract, args.solver_addendum)
    scratch, target_root, interface_root = v3._validate_real_input_roots(
        args.target_root, args.interface_root
    )
    output_root = _prepare_unique_output(
        args.output_root, scratch, mode="lambda_1e5"
    )
    FIT_TELEMETRY.clear()
    payload: dict[str, Any] = {
        "status": "fail",
        "stage": "v3_M0_solver_v7_lambda_1e5_warm_zero_equivalence_probe",
        "solver_addendum": args.solver_addendum.name,
        "held_out_scores_emitted": False,
        "coefficient_prediction_score_ID_or_target_contrast_vectors_emitted": False,
        "final_or_unassigned_opened": False,
        "output_root": str(output_root),
    }
    output_path = output_root / "v3_m0_solver_lambda1e5_equivalence_v7.json"
    try:
        x, y, k_values, groups = _load_exact_probe_training(
            target_root, interface_root
        )
        warm_prefix = _fit_lambda_1e5_warm_prefix(x, y, k_values, groups)
        warm = warm_prefix[LAMBDA_1E5]
        zero = fit_lbfgs_v7(
            x,
            y,
            k_values,
            groups,
            base.helmert(16),
            LAMBDA_1E5,
        )
        comparison = _lambda_1e5_comparison(
            warm, zero, x, y, k_values
        )
        results = (warm, zero)
        if not all(_result_has_valid_witness(result) for result in results):
            raise base.ModelError("a lambda-1e-5 fit lacks a convergence witness")
        if not all(
            result.solver_telemetry.get("solver_branch")
            == "v7_exact_Hessian_Newton"
            for result in results
        ):
            raise base.ModelError("lambda 1e-5 did not use V7 Newton twice")
        if not all(
            result.solver_telemetry.get("strong_wolfe_all_accepted_steps_pass")
            is True
            for result in results
        ):
            raise base.ModelError("a lambda-1e-5 fit accepted a non-Wolfe step")
        if not comparison["pass"]:
            raise base.ModelError("lambda-1e-5 warm-zero equivalence failed")
        payload.update(
            {
                "status": "pass",
                "training_cells": int(x.shape[0]),
                "training_group_count": int(len(set(groups))),
                "warm_prefix_lambdas": list(LAMBDA_1E5_WARM_PREFIX),
                "warm_prefix_solver_telemetry": [
                    warm_prefix[value].solver_telemetry
                    for value in LAMBDA_1E5_WARM_PREFIX
                ],
                "zero_start_solver_telemetry": zero.solver_telemetry,
                "lambda_1e5_warm_zero_comparison": comparison,
                "unchanged_equivalence_thresholds_pass": True,
                "both_lambda_1e5_fits_used_V7_exact_Hessian_Newton": True,
                "both_lambda_1e5_fits_met_unchanged_convergence_witness": True,
                "all_accepted_steps_satisfied_exact_strong_Wolfe": True,
            }
        )
        base._atomic_json(output_path, payload)
        return payload
    except (base.ModelError, AssertionError, OSError, ValueError) as error:
        payload.update(
            {
                "status": "fail",
                "failure_class": type(error).__name__,
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
        "stage": "v3_M0_solver_v7_full_targeted_numerical_probe",
        "solver_addendum": args.solver_addendum.name,
        "requires_prior_lambda_1e5_equivalence_probe_pass": True,
        "held_out_scores_emitted": False,
        "coefficient_prediction_score_ID_or_target_contrast_vectors_emitted": False,
        "final_or_unassigned_opened": False,
        "output_root": str(output_root),
    }
    output_path = output_root / "v3_m0_solver_probe_v7.json"
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
        worst_warm_iterations = max(
            result.iterations
            for path in (warm_first, warm_second)
            for result, _ in path.values()
        )
        if worst_warm_iterations >= WARM_PATH_MAX_ITERATIONS_EXCLUSIVE:
            raise base.ModelError("v7 warm path lacks frozen iteration headroom")
        if not all(_result_has_valid_witness(result) for result in all_results):
            raise base.ModelError("a v7 probe fit lacks a convergence witness")
        if not all(
            result.solver_telemetry.get("strong_wolfe_all_accepted_steps_pass")
            is True
            for result in all_results
        ):
            raise base.ModelError("a v7 probe fit accepted a non-Wolfe step")
        if not all(item["pass"] for item in repeat_checks):
            raise base.ModelError("v7 deterministic repeat check failed")
        if not all(item["pass"] for item in zero_checks):
            raise base.ModelError("v7 warm-zero comparison failed")
        for path in (warm_first, warm_second, zero_start):
            for ridge_lambda in NEWTON_LAMBDAS:
                if path[ridge_lambda][0].solver_telemetry.get(
                    "solver_branch"
                ) != "v7_exact_Hessian_Newton":
                    raise base.ModelError("a V7 Newton lambda used another branch")

        payload.update(
            {
                "status": "pass",
                "training_cells": int(x.shape[0]),
                "training_group_count": int(len(set(groups))),
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
                "warm_path_worst_iterations": worst_warm_iterations,
                "warm_path_max_iterations_exclusive": (
                    WARM_PATH_MAX_ITERATIONS_EXCLUSIVE
                ),
                "warm_path_iteration_headroom_pass": True,
                "all_fits_met_unchanged_convergence_witness": True,
                "all_accepted_steps_satisfied_exact_strong_Wolfe": True,
                "all_four_Newton_lambdas_used_V7_exact_Hessian": True,
            }
        )
        base._atomic_json(output_path, payload)
        return payload
    except (base.ModelError, AssertionError, OSError, ValueError) as error:
        payload.update(
            {
                "status": "fail",
                "failure_class": type(error).__name__,
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
    summary["solver_addendum_v7"] = {
        "schema_version": "episode10_v3_m0_solver_addendum_v7",
        "exact_Hessian_Newton_lambda_count": len(NEWTON_LAMBDAS),
        "V5_bitwise_forward_lambda_count": len(V5_FORWARD_LAMBDAS),
        "objective_grid_warm_start_tolerances_and_cap_changed": False,
    }
    base._atomic_json(output_root / "v3_m0_fit_summary.json", summary)
    base._atomic_json(
        output_root / "v3_m0_solver_telemetry_v7.json",
        {
            "status": "pass",
            "solver_addendum": args.solver_addendum.name,
            "fits": list(FIT_TELEMETRY),
            "coefficient_prediction_score_ID_or_target_contrast_vectors_emitted": False,
            "final_or_unassigned_opened": False,
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--unit-test", action="store_true")
    mode.add_argument("--lambda-1e5-probe", action="store_true")
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--contract", type=Path, default=BASE_CONTRACT_PATH)
    parser.add_argument("--solver-addendum", type=Path, default=V7_ADDENDUM_PATH)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--interface-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--temperature-diagnostic", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.unit_test:
        if any(
            value is not None
            for value in (args.target_root, args.interface_root, args.output_root)
        ):
            parser.error("unit test does not accept real input or output paths")
        try:
            validate_solver_addendum(args.contract, args.solver_addendum)
            result = run_unit_tests()
        except (base.ModelError, AssertionError, OSError, ValueError) as error:
            print(f"ERROR: {type(error).__name__}", file=sys.stderr)
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
    if (args.lambda_1e5_probe or args.probe) and args.temperature_diagnostic:
        parser.error("numerical probes cannot run the temperature diagnostic")
    try:
        if args.lambda_1e5_probe:
            result = run_lambda_1e5_probe(args)
        elif args.probe:
            result = run_targeted_probe(args)
        else:
            result = run_execute(args)
        if args.lambda_1e5_probe or args.probe:
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
        print(f"ERROR: {type(error).__name__}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
