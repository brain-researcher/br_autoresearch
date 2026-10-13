#!/usr/bin/env python3
"""Lightweight validator for one saved EP10 v3 M2 core qualification report.

The Slurm launcher runs the expensive qualification exactly once. This file
only reads that JSON artifact and checks its frozen scientific gate inventory,
case inventory, thresholds, and sealed scope; it never calls the qualifier.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_STAGE = "v3_M2_target_free_frozen_core_contract_qualification"
EXPECTED_GATES = (
    "exact_all_K_enumeration_and_beta_zero_nesting",
    "independent_brute_force_Gram_cross",
    "all_120_coordinate_central_derivatives",
    "independent_small_convex_prox_KKT",
    "range_null_probability_invariance",
    "lambda_maximum_rho_1_zero_solution_and_zero_gradient",
    "warm_and_zero_deterministic_repeats",
    "frozen_population_recovery",
)
CORE_PENALTY_CASES = ("lasso", "elastic_net", "group_lasso")
LAMBDA_PENALTY_CASES = (
    "lasso",
    "elastic_net_alpha_0.25",
    "elastic_net_alpha_0.5",
    "elastic_net_alpha_0.75",
    "group_lasso",
)


class ReportValidationError(RuntimeError):
    """The saved qualification report does not prove the frozen core gates."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportValidationError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _finite_at_most(value: Any, maximum: float, label: str) -> None:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value)
        and float(value) <= maximum,
        f"{label} does not satisfy <= {maximum:.17g}",
    )


def _exact_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    _require(set(value) == set(expected), f"{label} case inventory changed")


def validate_saved_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(report.get("status") == "pass", "overall qualification status is not pass")
    _require(report.get("overall_gate_status") == "pass", "overall gate status is not pass")
    _require(report.get("mode") == "full", "report is not a full qualification")
    _require(report.get("stage") == EXPECTED_STAGE, "qualification stage changed")

    gates = _mapping(report.get("gates"), "gates")
    _exact_keys(gates, EXPECTED_GATES, "top-level gate")
    inventory = _mapping(report.get("gate_inventory"), "gate_inventory")
    for field in (
        "expected_gate_order",
        "observed_gate_order",
        "passing_gate_order",
    ):
        _require(
            inventory.get(field) == list(EXPECTED_GATES),
            f"gate_inventory.{field} changed",
        )
    _require(
        inventory.get("required_gate_count") == len(EXPECTED_GATES),
        "required gate count changed",
    )
    _require(
        inventory.get("all_required_gates_present_and_passed") is True,
        "required gates were not all present and passed",
    )
    for gate_name in EXPECTED_GATES:
        _require(
            _mapping(gates[gate_name], gate_name).get("status") == "pass",
            f"required gate {gate_name} is not pass",
        )

    expected_case_inventory = {
        "K_values": list(range(17)),
        "candidate_rows": 65_536,
        "physical_pair_coordinates": 120,
        "prox_penalty_cases": list(CORE_PENALTY_CASES),
        "lambda_rho_1_penalty_cases": list(LAMBDA_PENALTY_CASES),
        "deterministic_repeat_penalty_cases": list(CORE_PENALTY_CASES),
        "population_recovery_penalty_cases": list(CORE_PENALTY_CASES),
        "identified_pair_contrasts_per_repeat_case": 120,
    }
    _require(
        report.get("case_inventory") == expected_case_inventory,
        "full qualification case inventory changed or is incomplete",
    )

    enumeration = _mapping(
        gates["exact_all_K_enumeration_and_beta_zero_nesting"], "enumeration gate"
    )
    _require(enumeration.get("K_values_checked") == list(range(17)), "K inventory changed")
    _require(enumeration.get("total_panel_subsets") == 65_536, "candidate total changed")
    _require(
        enumeration.get("candidate_rows_checked_for_exact_lexicographic_order")
        == 65_536,
        "not every candidate row was checked for exact lexicographic order",
    )
    _require(
        enumeration.get("pair_indicator_rows_checked_against_exact_PAIR_INDEX")
        == 65_536,
        "not every pair-indicator row was checked against PAIR_INDEX",
    )
    _require(
        enumeration.get("PAIR_INDEX_exact_lexicographic_order") is True
        and enumeration.get("PAIR_INDEX_coordinates_checked") == 120,
        "PAIR_INDEX exact-order evidence is incomplete",
    )
    for field in ("candidate_row_order_exact_by_K", "pair_indicator_order_exact_by_K"):
        by_k = _mapping(enumeration.get(field), f"enumeration.{field}")
        _require(
            by_k == {str(k): True for k in range(17)},
            f"{field} is not exact for every K",
        )
    _require(
        enumeration.get("maximum_choose_K_2_pair_count_error") == 0,
        "choose(K,2) pair count failed",
    )
    _require(
        enumeration.get("K_0_and_K_16_M2_minus_M1_gain_exact_zero") is True,
        "degenerate-K exact-zero gain failed",
    )
    for field in (
        "beta_zero_M1_log_probability_maximum_absolute_error",
        "beta_zero_M1_candidate_probability_maximum_absolute_error",
        "beta_zero_M1_marginal_maximum_absolute_error",
    ):
        _finite_at_most(enumeration.get(field), 1.0e-12, f"enumeration.{field}")

    gram = _mapping(gates["independent_brute_force_Gram_cross"], "Gram gate")
    _finite_at_most(gram.get("maximum_Gram_or_cross_error"), 1.0e-10, "Gram/cross")
    _finite_at_most(gram.get("maximum_orthogonality_error"), 1.0e-10, "orthogonality")
    _finite_at_most(
        gram.get("projector_idempotence_maximum_absolute_error"),
        1.0e-12,
        "projector idempotence",
    )
    _finite_at_most(
        gram.get("projector_symmetry_maximum_absolute_error"),
        1.0e-12,
        "projector symmetry",
    )

    derivatives = _mapping(
        gates["all_120_coordinate_central_derivatives"], "derivative gate"
    )
    _require(
        derivatives.get("coordinates_checked") == 120
        and derivatives.get("coordinates_passing_absolute_or_relative_rule") == 120
        and derivatives.get("absolute_tolerance") == 1.0e-6
        and derivatives.get("relative_tolerance") == 1.0e-5,
        "120-coordinate derivative inventory or thresholds changed",
    )

    prox = _mapping(gates["independent_small_convex_prox_KKT"], "prox gate")
    _require(prox.get("penalty_case_order") == list(CORE_PENALTY_CASES), "prox case order changed")
    _require(
        prox.get("Dykstra_absolute_infinity_norm_tolerance") == 1.0e-12,
        "Dykstra absolute tolerance changed",
    )
    _finite_at_most(
        prox.get("Dykstra_maximum_certificate_residual"), 1.0e-12, "Dykstra certificate"
    )
    for field in ("coordinate_aligned_penalties", "oblique_noncommuting_penalties"):
        cases = _mapping(prox.get(field), f"prox.{field}")
        _exact_keys(cases, CORE_PENALTY_CASES, f"prox.{field}")
        for case_name, case_value in cases.items():
            case = _mapping(case_value, f"prox.{field}.{case_name}")
            for residual_field in (
                "engine_fixed_point_residual",
                "engine_range_residual",
                "engine_invariant_residual",
            ):
                _finite_at_most(
                    case.get(residual_field),
                    1.0e-12,
                    f"prox.{field}.{case_name}.{residual_field}",
                )

    invariance = _mapping(
        gates["range_null_probability_invariance"], "range-null gate"
    )
    _finite_at_most(
        invariance.get("candidate_probability_maximum_absolute_error"),
        1.0e-10,
        "range-null probability invariance",
    )

    lambda_gate = _mapping(
        gates["lambda_maximum_rho_1_zero_solution_and_zero_gradient"],
        "lambda gate",
    )
    _require(
        lambda_gate.get("penalty_case_order") == list(LAMBDA_PENALTY_CASES),
        "lambda penalty case order changed",
    )
    _require(
        lambda_gate.get("rho_1_KKT_tolerance") == 1.0e-7
        and lambda_gate.get("relative_bracket_tolerance") == 1.0e-12
        and lambda_gate.get("initial_upper_bound") == 1.0
        and lambda_gate.get("maximum_expansions") == 80,
        "lambda thresholds or bracketing constants changed",
    )
    zero_kkt_solver = _mapping(
        lambda_gate.get("zero_KKT_solver"), "lambda zero-KKT solver telemetry"
    )
    _require(
        zero_kkt_solver.get("accelerated_iteration_budget") == 512
        and zero_kkt_solver.get("smooth_minimax_iteration_budget") == 1_024
        and zero_kkt_solver.get("smooth_minimax_smoothing_exponent_range")
        == [1, 18]
        and zero_kkt_solver.get("smooth_minimax_newton_steps_per_smoothing")
        == 40
        and zero_kkt_solver.get("active_face_roundoff_multiplier") == 64.0
        and zero_kkt_solver.get("active_face_backward_error_multiplier")
        == 1_024.0
        and zero_kkt_solver.get("active_face_budget")
        == "shared_smooth_minimax_iteration_budget"
        and zero_kkt_solver.get("shared_total_iteration_cap") == 10_000,
        "zero-KKT phase budgets or shared iteration cap changed",
    )
    _require(
        zero_kkt_solver.get("active_face_multiplier_polish")
        == "candidate_only_guarded_separator_acceptance"
        and
        zero_kkt_solver.get("feasible_acceptance")
        == "direct_dual_witness_only"
        and zero_kkt_solver.get("infeasible_acceptance")
        == "guarded_range_space_support_separator_only",
        "zero-KKT certificate acceptance paths changed",
    )
    _require(
        lambda_gate.get("exact_zero_gradient_bitwise_all_zero") is True,
        "exact-zero gradient behavior was not exercised",
    )
    tiny_norm = lambda_gate.get("tiny_nonzero_gradient_infinity_norm")
    _require(
        isinstance(tiny_norm, (int, float))
        and 0.0 < float(tiny_norm) < 1.0e-9,
        "tiny-nonzero gradient boundary was not exercised below the old floor",
    )
    lambda_cases = _mapping(lambda_gate.get("cases"), "lambda cases")
    _exact_keys(lambda_cases, LAMBDA_PENALTY_CASES, "lambda")
    for case_name, case_value in lambda_cases.items():
        case = _mapping(case_value, f"lambda.{case_name}")
        _require(
            case.get("rho") == 1.0
            and case.get("rho_1_beta_exact_zero") is True
            and case.get("rho_1_fit_iterations") == 0
            and float(case.get("lambda_max", 0.0)) > 0.0
            and case.get("exact_zero_gradient_lambda_max") == 0.0
            and float(case.get("tiny_nonzero_gradient_lambda_max", 0.0)) > 0.0,
            f"lambda boundary behavior failed for {case_name}",
        )
        _finite_at_most(case.get("rho_1_KKT_residual"), 1.0e-7, f"{case_name} rho=1 KKT")
        _finite_at_most(
            case.get("lambda_bracket_relative_width"), 1.0e-12, f"{case_name} bracket"
        )
        _finite_at_most(
            case.get("tiny_nonzero_lambda_bracket_relative_width"),
            1.0e-12,
            f"{case_name} tiny bracket",
        )

    repeats = _mapping(
        gates["warm_and_zero_deterministic_repeats"], "repeat gate"
    )
    _require(
        repeats.get("penalty_case_order") == list(CORE_PENALTY_CASES)
        and repeats.get("zero_vs_warm_objective_tolerance") == 1.0e-12
        and repeats.get("zero_vs_warm_candidate_probability_tolerance") == 1.0e-10
        and repeats.get("zero_vs_warm_identified_pair_contrast_tolerance") == 1.0e-10
        and repeats.get("identified_equal_group_pair_contrasts_per_penalty") == 120,
        "repeat case inventory or thresholds changed",
    )
    repeat_cases = _mapping(repeats.get("penalties"), "repeat cases")
    _exact_keys(repeat_cases, CORE_PENALTY_CASES, "repeat")
    for case_name, case_value in repeat_cases.items():
        case = _mapping(case_value, f"repeat.{case_name}")
        _require(
            case.get("zero_repeat_exact") is True
            and case.get("warm_repeat_exact") is True
            and case.get("identified_equal_group_pair_contrasts_checked") == 120,
            f"repeat identity or contrast inventory failed for {case_name}",
        )
        _finite_at_most(
            case.get("zero_vs_warm_objective_absolute_difference"),
            1.0e-12,
            f"{case_name} repeat objective",
        )
        _finite_at_most(
            case.get("zero_vs_warm_candidate_probability_maximum_absolute_error"),
            1.0e-10,
            f"{case_name} repeat probability",
        )
        _finite_at_most(
            case.get("zero_vs_warm_identified_pair_contrast_maximum_absolute_difference"),
            1.0e-10,
            f"{case_name} repeat identified pair contrast",
        )

    recovery = _mapping(gates["frozen_population_recovery"], "recovery gate")
    _require(
        recovery.get("penalty_case_order") == list(CORE_PENALTY_CASES),
        "recovery case order changed",
    )
    fixture = _mapping(recovery.get("fixture"), "recovery fixture")
    _require(
        fixture.get("seed") == 2026092810
        and fixture.get("groups") == 8
        and fixture.get("rows_per_group") == 17
        and fixture.get("cells") == 136
        and fixture.get("K_schedule") == "each_group_contains_every_K_0_through_16_once",
        "frozen recovery fixture identity changed",
    )
    recovery_cases = _mapping(recovery.get("cases"), "recovery cases")
    _exact_keys(recovery_cases, CORE_PENALTY_CASES, "recovery")
    expected_nonzero = {"lasso": 6, "elastic_net": 12, "group_lasso": 8}
    for case_name, case_value in recovery_cases.items():
        case = _mapping(case_value, f"recovery.{case_name}")
        _require(
            case.get("plant_nonzero_physical_edges") == expected_nonzero[case_name],
            f"{case_name} planted edge inventory changed",
        )
        _finite_at_most(
            case.get("mean_conditional_KL_bits"), 1.0e-6, f"{case_name} recovery KL"
        )
        _finite_at_most(
            case.get("candidate_probability_maximum_absolute_error"),
            1.0e-5,
            f"{case_name} recovery candidate probability",
        )
        _finite_at_most(
            case.get("pair_event_probability_MAE"),
            1.0e-5,
            f"{case_name} recovery pair-event MAE",
        )

    frozen_fixture = _mapping(report.get("frozen_fixture"), "frozen_fixture")
    _require(
        frozen_fixture
        == {
            "seed": 2026092810,
            "groups": 8,
            "rows_per_group": 17,
            "cells": 136,
            "every_K_0_through_16_once_per_group": True,
        },
        "top-level frozen fixture inventory changed",
    )
    for field in (
        "real_target_membership_or_M2_effects_opened",
        "actual_K_opened",
        "M0_full_refit_or_telemetry_opened",
        "final_or_unassigned_or_SWC_opened",
    ):
        _require(report.get(field) is False, f"sealed-scope flag {field} is not false")
    _require(
        report.get("controller_replay_qualification_included") is False,
        "controller replay must remain a separate qualification",
    )

    return {
        "status": "pass",
        "validated_report": str(report.get("stage")),
        "required_gate_count": len(EXPECTED_GATES),
        "lambda_penalty_case_count": len(LAMBDA_PENALTY_CASES),
        "core_penalty_case_count": len(CORE_PENALTY_CASES),
        "expensive_qualification_rerun": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        with arguments.report.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        result = validate_saved_report(_mapping(report, "saved report"))
    except (OSError, json.JSONDecodeError, ReportValidationError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
