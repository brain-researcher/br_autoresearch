"""Run EP12's actual T/U/M fitter on generated biological counterexamples."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import numpy as np
    from scipy.stats import beta
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("scientific qualification needs NumPy and SciPy") from exc

from .policy import EpisodePolicy
from .scientific_models import ScientificModelError, reciprocal_transfer, seed_from
from .scientific_synthetic import SCENARIO_TRUTH, generate_synthetic_type
from .runtime import EPISODE_ROOT, SCRATCH_ROOT, atomic_json
from .yaml_subset import load_yaml_subset


CORE_NO_M_SCENARIOS = {
    "adequate_template",
    "linear_continuum",
    "curved_continuum_dense",
    "curved_continuum_sparse_intermediates",
    "side_specific_endpoint_loss",
    "anatomy_quality_confound",
    "high_strength_or_influential_cells",
}
STRONG_M_SCENARIOS = {"residual_groups_within_continuum"}


@dataclass(frozen=True)
class QualificationCell:
    cell_id: str
    scenario: str
    neurons_per_side: int
    count_total: int
    strength: float
    missing_fraction: float
    acceptance_role: str


def _contract_settings(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the executable supports the frozen contract exactly."""

    expected = {
        "likelihood": "conditional_multinomial",
        "score": "heldout_log_predictive_density_per_observed_outgoing_count",
        "u_candidates": ["U_linear", "U_curved"],
        "m_components": [2, 3],
        "curve_degree": 3,
        "linear_rank": 2,
        "source_cv_folds": 2,
        "integration_draws": 256,
        "nuisance_ridge": 0.1,
        "mixture_initializations": 5,
        "mixture_max_iterations": 150,
        "mixture_covariance_floor": 0.0001,
        "curved_quadrature_nodes": 9,
        "numeric_threads_per_worker": 1,
        "bootstrap_replicates": 1999,
        "fit_pseudocount": 0.5,
        "mixture_initialization": "deterministic_seeded_kmeans",
    }
    observed = {
        "likelihood": contract["likelihood"]["family"],
        "score": contract["likelihood"]["score"],
        "u_candidates": contract["models"]["U_reference"]["candidates"],
        "m_components": contract["models"]["M"]["component_counts"],
        "curve_degree": contract["models"]["U_curved"]["polynomial_degree"],
        "linear_rank": contract["models"]["U_linear"]["latent_dimension_max"],
        "source_cv_folds": contract["execution_parameters"][
            "source_cross_validation_folds"
        ],
        "integration_draws": contract["execution_parameters"][
            "predictive_integration_draws_per_proposal"
        ],
        "nuisance_ridge": contract["execution_parameters"]["nuisance_ridge_penalty"],
        "mixture_initializations": contract["execution_parameters"][
            "mixture_initializations"
        ],
        "mixture_max_iterations": contract["execution_parameters"][
            "mixture_max_iterations"
        ],
        "mixture_covariance_floor": contract["execution_parameters"][
            "mixture_covariance_floor"
        ],
        "curved_quadrature_nodes": contract["execution_parameters"][
            "curved_quadrature_nodes"
        ],
        "numeric_threads_per_worker": contract["execution_parameters"][
            "numeric_threads_per_worker"
        ],
        "bootstrap_replicates": contract["execution_parameters"][
            "simultaneous_bootstrap_replicates"
        ],
        "fit_pseudocount": contract["likelihood"]["fit_pseudocount"],
        "mixture_initialization": contract["execution_parameters"][
            "mixture_initialization"
        ],
    }
    if observed != expected:
        raise ValueError(
            "scientific contract requests an implementation not qualified by this executor: "
            f"{observed!r}"
        )
    if contract["transfer_scoring"]["fit_source_side_only"] is not True:
        raise ValueError("target-side fitting is forbidden")
    if contract["transfer_scoring"]["target_latent_coordinate"] != "marginalized":
        raise ValueError("target latent coordinate must be marginalized")
    if contract["transfer_scoring"]["target_group"] != "marginalized":
        raise ValueError("target group must be marginalized")
    expected_endpoint_bins = {
        "known_partner_type",
        "unknown",
        "untyped",
        "fragment",
        "proofreading",
        "missing_annotation",
        "out_of_vocabulary",
    }
    if set(contract["input_object"]["endpoint_bins"]) != expected_endpoint_bins:
        raise ValueError("canonical endpoint-bin contract differs from the qualified implementation")
    if contract["models"]["M"]["shared_covariance_across_components"] is not True:
        raise ValueError("qualified M implementation requires shared component covariance")
    if contract["real_connectivity_access_authorized"] not in (False, "development_only"):
        raise ValueError("generated qualification permits at most development-only authorization")
    if contract.get("final_connectivity_access_authorized", False) is not False:
        raise ValueError("generated qualification requires final connectivity to remain disabled")
    return expected


def _tier_strata(tier: str, spec: Mapping[str, Any]) -> list[tuple[int, int, float, float]]:
    if tier not in {"smoke", "qualification", "stress"}:
        raise ValueError(f"unknown qualification tier: {tier}")
    run_tiers = spec["run_tiers"]
    fields = (
        run_tiers[f"{tier}_neurons_per_side"],
        run_tiers[f"{tier}_count_totals"],
        run_tiers[f"{tier}_generator_strengths"],
        run_tiers[f"{tier}_missing_fractions"],
    )
    if len({len(field) for field in fields}) != 1:
        raise ValueError(f"{tier} qualification strata have unequal lengths")
    return [
        (int(neurons), int(total), float(strength), float(missing))
        for neurons, total, strength, missing in zip(*fields)
    ]


def _cell_id(
    scenario: str, neurons: int, total: int, strength: float, missing: float
) -> str:
    strength_id = str(strength).replace(".", "p")
    missing_id = str(missing).replace(".", "p")
    return f"{scenario}__n{neurons}__N{total}__s{strength_id}__m{missing_id}"


def qualification_cells(
    tier: str, spec: Mapping[str, Any] | None = None
) -> list[QualificationCell]:
    if spec is None:
        spec = load_yaml_subset(
            EPISODE_ROOT / "outputs" / "executor" / "SYNTHETIC_QUALIFICATION_SPEC.yaml"
        )
    scenarios = list(spec["generators"])
    if set(scenarios) != set(SCENARIO_TRUTH):
        raise ValueError("qualification scenario inventory differs from executable generators")
    strata = _tier_strata(tier, spec)
    cells = [
        QualificationCell(
            cell_id=_cell_id(scenario, neurons, total, strength, missing),
            scenario=scenario,
            neurons_per_side=neurons,
            count_total=total,
            strength=strength,
            missing_fraction=missing,
            acceptance_role="core_operating_characteristic",
        )
        for scenario in scenarios
        for neurons, total, strength, missing in strata
    ]
    if tier in {"qualification", "stress"}:
        grid = spec["resolution_grid"]
        if spec["run_tiers"]["qualification_adds_group_strength_resolution_grid"]:
            for strength in grid["group_logit_strength"]:
                cells.append(
                    QualificationCell(
                        cell_id=_cell_id(
                            "residual_groups_within_continuum", 19, 100, float(strength), 0.1
                        ),
                        scenario="residual_groups_within_continuum",
                        neurons_per_side=19,
                        count_total=100,
                        strength=float(strength),
                        missing_fraction=0.1,
                        acceptance_role="resolution_diagnostic_only",
                    )
                )
        if spec["run_tiers"]["qualification_adds_side_loss_fraction_grid"]:
            for missing in grid["side_loss_fraction"]:
                cells.append(
                    QualificationCell(
                        cell_id=_cell_id(
                            "side_specific_endpoint_loss", 19, 100, 1.8, float(missing)
                        ),
                        scenario="side_specific_endpoint_loss",
                        neurons_per_side=19,
                        count_total=100,
                        strength=1.8,
                        missing_fraction=float(missing),
                        acceptance_role="resolution_diagnostic_only",
                    )
                )
    unique = {cell.cell_id: cell for cell in cells}
    return list(unique.values())


def exact_binomial_interval(successes: int, total: int, level: float = 0.95) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    alpha = 1.0 - level
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(
        beta.ppf(1.0 - alpha / 2, successes + 1, total - successes)
    )
    return lower, upper


def paired_simultaneous_intervals(
    type_direction_deltas: np.ndarray,
    *,
    margin: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(type_direction_deltas, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 2:
        raise ScientificModelError("simultaneous intervals need >=2 whole types and two directions")
    observed = values.mean(axis=0)
    aggregate = float(observed.mean())
    targets = np.array([observed[0], observed[1], aggregate])
    standard_errors = np.array(
        [
            np.std(values[:, 0], ddof=1) / math.sqrt(len(values)),
            np.std(values[:, 1], ddof=1) / math.sqrt(len(values)),
            np.std(values.mean(axis=1), ddof=1) / math.sqrt(len(values)),
        ]
    )
    standard_errors = np.maximum(standard_errors, 1e-9)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    boot = values[indices].mean(axis=1)
    boot_targets = np.column_stack([boot[:, 0], boot[:, 1], boot.mean(axis=1)])
    max_t = np.max(np.abs((boot_targets - targets[None, :]) / standard_errors[None, :]), axis=1)
    critical = float(np.quantile(max_t, 0.95, method="higher"))
    lower = targets - critical * standard_errors
    upper = targets + critical * standard_errors
    return {
        "labels": ["left_to_right", "right_to_left", "aggregate"],
        "estimate": targets.tolist(),
        "standard_error": standard_errors.tolist(),
        "critical_max_t": critical,
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "margin": margin,
        "bootstrap_replicates": replicates,
    }


def _endpoint_side_permutation_p(
    left_unknown: Sequence[float],
    right_unknown: Sequence[float],
    *,
    seed: int,
    permutations: int = 999,
) -> float:
    left = np.asarray(left_unknown, dtype=float)
    right = np.asarray(right_unknown, dtype=float)
    observed = abs(float(right.mean() - left.mean()))
    pooled = np.concatenate([left, right])
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        shuffled = rng.permutation(pooled)
        statistic = abs(float(shuffled[len(left) :].mean() - shuffled[: len(left)].mean()))
        exceed += statistic >= observed - 1e-15
    return (1.0 + exceed) / (permutations + 1.0)


def _terminal_without_full_null(
    intervals: Mapping[str, Any],
    *,
    controls_pass: bool,
    margin: float,
) -> str:
    lower = intervals["lower"]
    upper = intervals["upper"]
    if controls_pass and lower[0] > margin and lower[1] > margin:
        return "M_supported_before_full_search_null"
    if controls_pass and upper[0] < margin and upper[1] < margin:
        return "single_population_adequate_before_sensitivity_gate"
    return "unresolved"


def run_generated_replicate(
    cell: QualificationCell,
    *,
    replicate_index: int,
    whole_types: int,
    margin: float,
    bootstrap_replicates: int,
    integration_draws: int,
    folds: int,
) -> dict[str, Any]:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    type_deltas: list[list[float]] = []
    type_records: list[dict[str, Any]] = []
    left_unknown: list[float] = []
    right_unknown: list[float] = []
    numerical_failures: list[str] = []
    for type_index in range(whole_types):
        fixture_seed = seed_from(cell.cell_id, replicate_index, type_index, "fixture")
        fixture = generate_synthetic_type(
            cell.scenario,
            left_neurons=cell.neurons_per_side,
            right_neurons=cell.neurons_per_side,
            count_total=cell.count_total,
            strength=cell.strength,
            missing_fraction=cell.missing_fraction,
            seed=fixture_seed,
        )
        unknown_index = fixture.partner_names.index("explicit_unknown")
        left_unknown.extend(
            (
                fixture.left.counts[:, unknown_index]
                / fixture.left.counts.sum(axis=1)
            ).tolist()
        )
        right_unknown.extend(
            (
                fixture.right.counts[:, unknown_index]
                / fixture.right.counts.sum(axis=1)
            ).tolist()
        )
        try:
            result = reciprocal_transfer(
                fixture.left.counts,
                fixture.left.covariates,
                fixture.right.counts,
                fixture.right.covariates,
                seed=seed_from(cell.cell_id, replicate_index, type_index, "fit"),
                integration_draws=integration_draws,
                folds=folds,
                latent_rank=2,
                ridge=0.1,
            )
            deltas = [
                float(result["left_to_right"]["mean_delta_M_minus_reference"]),
                float(result["right_to_left"]["mean_delta_M_minus_reference"]),
            ]
            type_deltas.append(deltas)
            type_records.append(
                {
                    "synthetic_type_index": type_index,
                    "deltas": deltas,
                    "source_fit_metadata": result["source_fit_metadata"],
                }
            )
        except Exception as exc:  # fail closed and retain exact synthetic failure
            numerical_failures.append(f"type-{type_index}:{type(exc).__name__}:{exc}")
    if numerical_failures or len(type_deltas) != whole_types:
        return {
            "cell_id": cell.cell_id,
            "scenario": cell.scenario,
            "truth_family": SCENARIO_TRUTH[cell.scenario],
            "replicate_index": replicate_index,
            "terminal": "numerical_failure",
            "numerical_failures": numerical_failures,
            "type_records": type_records,
            "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
        }

    delta_array = np.asarray(type_deltas, dtype=float)
    intervals = paired_simultaneous_intervals(
        delta_array,
        margin=margin,
        replicates=bootstrap_replicates,
        seed=seed_from(cell.cell_id, replicate_index, "bootstrap"),
    )
    endpoint_p = _endpoint_side_permutation_p(
        left_unknown,
        right_unknown,
        seed=seed_from(cell.cell_id, replicate_index, "endpoint-control"),
    )
    direction_means = delta_array.mean(axis=0)
    leave_one_out = np.array(
        [np.delete(delta_array, index, axis=0).mean(axis=0) for index in range(len(delta_array))],
        dtype=float,
    )
    potential_bilateral_support = bool(np.all(direction_means > margin))
    influence_pass = bool(
        not potential_bilateral_support or np.all(leave_one_out > margin)
    )
    endpoint_pass = bool(endpoint_p > 0.05)
    controls_pass = endpoint_pass and influence_pass
    terminal = _terminal_without_full_null(intervals, controls_pass=controls_pass, margin=margin)
    return {
        "cell_id": cell.cell_id,
        "scenario": cell.scenario,
        "truth_family": SCENARIO_TRUTH[cell.scenario],
        "replicate_index": replicate_index,
        "terminal": terminal,
        "intervals": intervals,
        "controls": {
            "endpoint_side_permutation_p": endpoint_p,
            "endpoint_mass_control_pass": endpoint_pass,
            "leave_one_type_out_directional_gain": leave_one_out.tolist(),
            "influence_control_pass": influence_pass,
            "all_pass": controls_pass,
        },
        "numerical_failures": [],
        "type_records": type_records,
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
    }


def _cell_summary(cell: QualificationCell, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    terminal_counts = {
        terminal: sum(record["terminal"] == terminal for record in records)
        for terminal in (
            "M_supported_before_full_search_null",
            "single_population_adequate_before_sensitivity_gate",
            "unresolved",
            "numerical_failure",
        )
    }
    core_acceptance = cell.acceptance_role == "core_operating_characteristic"
    false_m = (
        terminal_counts["M_supported_before_full_search_null"]
        if core_acceptance and cell.scenario in CORE_NO_M_SCENARIOS
        else 0
    )
    false_adequacy = (
        terminal_counts["single_population_adequate_before_sensitivity_gate"]
        if core_acceptance and cell.scenario in STRONG_M_SCENARIOS
        else 0
    )
    estimable = [record for record in records if "intervals" in record]
    coverage_counts = [0, 0]
    coverage_total = len(estimable) if len(estimable) >= 2 else 0
    if coverage_total:
        estimates = np.asarray(
            [record["intervals"]["estimate"][:2] for record in estimable],
            dtype=float,
        )
        for record_index, record in enumerate(estimable):
            center = np.delete(estimates, record_index, axis=0).mean(axis=0)
            for direction in range(2):
                coverage_counts[direction] += (
                    record["intervals"]["lower"][direction]
                    <= center[direction]
                    <= record["intervals"]["upper"][direction]
                )
    false_m_ci = exact_binomial_interval(false_m, total)
    false_adequacy_ci = exact_binomial_interval(false_adequacy, total)
    coverage_rates = [
        coverage_counts[index] / coverage_total if coverage_total else 0.0
        for index in range(2)
    ]
    coverage_intervals = [
        exact_binomial_interval(coverage_counts[index], coverage_total)
        if coverage_total
        else (0.0, 1.0)
        for index in range(2)
    ]
    return {
        **asdict(cell),
        "truth_family": SCENARIO_TRUTH[cell.scenario],
        "replicates": total,
        "terminal_counts": terminal_counts,
        "false_M_rate": false_m / total,
        "false_M_exact_95_interval": list(false_m_ci),
        "false_adequacy_rate": false_adequacy / total,
        "false_adequacy_exact_95_interval": list(false_adequacy_ci),
        "empirical_interval_coverage": {
            "left_to_right": coverage_rates[0],
            "right_to_left": coverage_rates[1],
            "left_to_right_exact_95_interval": list(coverage_intervals[0]),
            "right_to_left_exact_95_interval": list(coverage_intervals[1]),
            "target": 0.95,
            "scope": "coverage_of_leave_one_replicate_out_generated_center_not_oracle_parameter",
        },
        "wall_seconds": float(sum(record["wall_seconds"] for record in records)),
        "cpu_seconds": float(sum(record["cpu_seconds"] for record in records)),
    }


def _accept_cell(
    summary: Mapping[str, Any], acceptance: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary["terminal_counts"]["numerical_failure"]:
        reasons.append("numerical_failure")
    false_m_max = float(acceptance["false_M_rate_max"])
    false_adequacy_max = float(acceptance["false_adequacy_rate_max"])
    if (
        summary["acceptance_role"] == "core_operating_characteristic"
        and summary["scenario"] in CORE_NO_M_SCENARIOS
        and summary["false_M_exact_95_interval"][1] > false_m_max
    ):
        reasons.append("false_M_exact_upper_bound_above_contract_maximum")
    if (
        summary["acceptance_role"] == "core_operating_characteristic"
        and summary["scenario"] in STRONG_M_SCENARIOS
        and summary["false_adequacy_exact_95_interval"][1] > false_adequacy_max
    ):
        reasons.append("false_adequacy_exact_upper_bound_above_contract_maximum")
    nominal = float(acceptance["interval_nominal_coverage"])
    for direction in ("left_to_right", "right_to_left"):
        interval = summary["empirical_interval_coverage"][f"{direction}_exact_95_interval"]
        if not interval[0] <= nominal <= interval[1]:
            reasons.append(f"{direction}_coverage_incompatible_with_contract_nominal")
    return not reasons, reasons


def _write_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "cell_id",
        "scenario",
        "truth_family",
        "neurons_per_side",
        "count_total",
        "strength",
        "missing_fraction",
        "acceptance_role",
        "replicates",
        "false_M_rate",
        "false_adequacy_rate",
        "M_supported",
        "adequate",
        "unresolved",
        "numerical_failure",
        "coverage_left_to_right",
        "coverage_right_to_left",
        "wall_seconds",
        "cpu_seconds",
        "passed",
        "failure_reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "cell_id": summary["cell_id"],
                    "scenario": summary["scenario"],
                    "truth_family": summary["truth_family"],
                    "neurons_per_side": summary["neurons_per_side"],
                    "count_total": summary["count_total"],
                    "strength": summary["strength"],
                    "missing_fraction": summary["missing_fraction"],
                    "acceptance_role": summary["acceptance_role"],
                    "replicates": summary["replicates"],
                    "false_M_rate": summary["false_M_rate"],
                    "false_adequacy_rate": summary["false_adequacy_rate"],
                    "M_supported": summary["terminal_counts"]["M_supported_before_full_search_null"],
                    "adequate": summary["terminal_counts"]["single_population_adequate_before_sensitivity_gate"],
                    "unresolved": summary["terminal_counts"]["unresolved"],
                    "numerical_failure": summary["terminal_counts"]["numerical_failure"],
                    "coverage_left_to_right": summary["empirical_interval_coverage"]["left_to_right"],
                    "coverage_right_to_left": summary["empirical_interval_coverage"]["right_to_left"],
                    "wall_seconds": summary["wall_seconds"],
                    "cpu_seconds": summary["cpu_seconds"],
                    "passed": summary["passed"],
                    "failure_reasons": ";".join(summary["failure_reasons"]),
                }
            )


def _write_markdown(path: Path, summaries: Sequence[Mapping[str, Any]], report: Mapping[str, Any]) -> None:
    lines = [
        "# EP12 scientific synthetic qualification",
        "",
        f"Tier: `{report['tier']}`  ",
        f"Qualification passed: `{str(report['qualification_passed']).lower()}`  ",
        "Real connectivity accessed: `false`",
        "",
        "The table was produced by the actual count-likelihood T/U/M fitter. No scenario was routed to a prewritten terminal.",
        "",
        "| Scenario | n/side | count total | reps | false M | false adequacy | M | adequate | unresolved | failures | pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summaries:
        counts = item["terminal_counts"]
        lines.append(
            "| {scenario} | {n} | {total} | {reps} | {fm:.3f} | {fa:.3f} | {m} | {a} | {u} | {f} | {passed} |".format(
                scenario=item["scenario"],
                n=item["neurons_per_side"],
                total=item["count_total"],
                reps=item["replicates"],
                fm=item["false_M_rate"],
                fa=item["false_adequacy_rate"],
                m=counts["M_supported_before_full_search_null"],
                a=counts["single_population_adequate_before_sensitivity_gate"],
                u=counts["unresolved"],
                f=counts["numerical_failure"],
                passed="yes" if item["passed"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "These are pre-null method-resolution labels. They are not EP12 scientific terminals and cannot support a MaleCNS conclusion.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def integration_stability_diagnostic(
    spec: Mapping[str, Any], *, margin: float, folds: int
) -> dict[str, Any]:
    settings = spec["numerical_stability"]
    draws_grid = [int(value) for value in settings["integration_draws"]]
    if draws_grid != [128, 256, 512]:
        raise ValueError("unsupported integration stability grid")
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    rows: list[dict[str, Any]] = []
    absolute_changes: list[float] = []
    identity_matches: list[bool] = []
    for scenario in settings["scenarios"]:
        if scenario not in SCENARIO_TRUTH:
            raise ValueError(f"unknown numerical-stability scenario: {scenario}")
        for fixture_seed in settings["seeds"]:
            fixture = generate_synthetic_type(
                scenario,
                left_neurons=int(settings["neurons_per_side"]),
                right_neurons=int(settings["neurons_per_side"]),
                count_total=int(settings["count_total"]),
                strength=float(settings["generator_strength"]),
                missing_fraction=float(settings["missing_fraction"]),
                seed=int(fixture_seed),
            )
            by_draws: dict[int, dict[str, Any]] = {}
            fit_seed = seed_from("integration-stability", scenario, fixture_seed)
            for draws in draws_grid:
                result = reciprocal_transfer(
                    fixture.left.counts,
                    fixture.left.covariates,
                    fixture.right.counts,
                    fixture.right.covariates,
                    seed=fit_seed,
                    integration_draws=draws,
                    folds=folds,
                    latent_rank=2,
                    ridge=0.1,
                )
                by_draws[draws] = {
                    "left_to_right_delta": float(
                        result["left_to_right"]["mean_delta_M_minus_reference"]
                    ),
                    "right_to_left_delta": float(
                        result["right_to_left"]["mean_delta_M_minus_reference"]
                    ),
                    "source_fit_metadata": result["source_fit_metadata"],
                }
            contract_result = by_draws[256]
            high_result = by_draws[512]
            changes = [
                abs(
                    contract_result[f"{direction}_delta"]
                    - high_result[f"{direction}_delta"]
                )
                for direction in ("left_to_right", "right_to_left")
            ]
            identity_match = (
                contract_result["source_fit_metadata"]
                == high_result["source_fit_metadata"]
            )
            absolute_changes.extend(changes)
            identity_matches.append(identity_match)
            rows.append(
                {
                    "scenario": scenario,
                    "fixture_seed": int(fixture_seed),
                    "by_integration_draws": {
                        str(draws): by_draws[draws] for draws in draws_grid
                    },
                    "absolute_contract_to_high_delta_change": changes,
                    "selected_model_ids_match": identity_match,
                }
            )
    tolerance = margin * float(
        settings["maximum_absolute_delta_change_relative_to_margin"]
    )
    model_identity_required = bool(
        settings["selected_model_ids_must_match_at_contract_and_high_draws"]
    )
    maximum_change = max(absolute_changes, default=float("inf"))
    return {
        "schema_version": "ep12.integration_stability.v1",
        "rows": rows,
        "fixed_margin": margin,
        "absolute_delta_change_tolerance": tolerance,
        "maximum_observed_absolute_delta_change": maximum_change,
        "all_selected_model_ids_match": all(identity_matches),
        "passed": (
            maximum_change <= tolerance
            and (not model_identity_required or all(identity_matches))
        ),
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "real_connectivity_accessed": False,
    }


def run_scientific_qualification(
    *,
    tier: str,
    output_dir: Path,
    repetitions: int | None = None,
    whole_types: int | None = None,
    integration_draws: int | None = None,
    folds: int | None = None,
    bootstrap_replicates: int | None = None,
    contract_path: Path | None = None,
    qualification_spec_path: Path | None = None,
    episode_root: Path = EPISODE_ROOT,
    policy_path: Path | None = None,
) -> dict[str, Any]:
    episode_root = episode_root.resolve()
    output_dir = output_dir.resolve()
    allowed_root = (episode_root / "outputs").resolve()
    if output_dir != allowed_root and allowed_root not in output_dir.parents:
        raise ValueError("scientific qualification outputs must stay under outputs")
    protected_roots = tuple(
        (allowed_root / name).resolve()
        for name in (
            "prelaunch",
            "development_run001",
            "development_run002",
            "executor_qualification",
            "executor_qualification_v2",
            "executor_qualification_v3",
        )
    )
    if any(output_dir == root or root in output_dir.parents for root in protected_roots):
        raise ValueError("scientific qualification cannot write into a frozen output tree")
    contract_path = (contract_path or (
        episode_root / "outputs" / "executor" / "SCIENTIFIC_CONTRACT_POST36.yaml"
    )).absolute()
    spec_path = (qualification_spec_path or (
        episode_root
        / "outputs"
        / "executor"
        / "SYNTHETIC_QUALIFICATION_SPEC_POST36.yaml"
    )).absolute()
    if contract_path.name != "SCIENTIFIC_CONTRACT_POST36.yaml":
        raise ValueError("qualification requires the active post-36 scientific contract")
    if spec_path.name != "SYNTHETIC_QUALIFICATION_SPEC_POST36.yaml":
        raise ValueError("qualification requires the active post-36 specification")
    if contract_path.parent != spec_path.parent:
        raise ValueError("post-36 contract and qualification spec must share one executor root")
    contract = load_yaml_subset(contract_path)
    spec = load_yaml_subset(spec_path)
    model_program = _contract_settings(contract)
    policy_path = (policy_path or (episode_root / "SEARCH_POLICY.yaml")).absolute()
    policy = EpisodePolicy.load(policy_path)
    integration_draws = int(
        model_program["integration_draws"]
        if integration_draws is None
        else integration_draws
    )
    folds = int(model_program["source_cv_folds"] if folds is None else folds)
    bootstrap_replicates = int(
        model_program["bootstrap_replicates"]
        if bootstrap_replicates is None
        else bootstrap_replicates
    )
    contract_configuration = (
        integration_draws == model_program["integration_draws"]
        and folds == model_program["source_cv_folds"]
        and bootstrap_replicates == model_program["bootstrap_replicates"]
    )
    tier_key = {
        "smoke": "smoke_replicates_per_cell",
        "qualification": "qualification_replicates_per_core_error_cell",
        "stress": "stress_replicates_per_large_cell",
    }[tier]
    repetitions = int(
        spec["run_tiers"][tier_key] if repetitions is None else repetitions
    )
    whole_types = int(
        spec["run_tiers"]["whole_types_per_replicate"]
        if whole_types is None
        else whole_types
    )
    if repetitions < 1 or whole_types < 2:
        raise ValueError("qualification needs >=1 repetition and >=2 whole types")
    margin = float(policy.meaningful_margin)
    cells = qualification_cells(tier, spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    replicate_results_path = output_dir / "replicate_results.jsonl"
    if replicate_results_path.exists():
        raise FileExistsError(
            f"qualification replicate results already exist: {replicate_results_path}"
        )
    all_records: dict[str, list[dict[str, Any]]] = {cell.cell_id: [] for cell in cells}
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    with replicate_results_path.open("x", encoding="utf-8") as result_handle:
        for cell in cells:
            for replicate_index in range(repetitions):
                record = run_generated_replicate(
                    cell,
                    replicate_index=replicate_index,
                    whole_types=whole_types,
                    margin=margin,
                    bootstrap_replicates=bootstrap_replicates,
                    integration_draws=integration_draws,
                    folds=folds,
                )
                all_records[cell.cell_id].append(record)
                result_handle.write(json.dumps(record, sort_keys=True) + "\n")
                result_handle.flush()
        os.fsync(result_handle.fileno())
    summaries: list[dict[str, Any]] = []
    for cell in cells:
        summary = _cell_summary(cell, all_records[cell.cell_id])
        passed, reasons = _accept_cell(summary, spec["acceptance"])
        summary["passed"] = passed
        summary["failure_reasons"] = reasons
        summaries.append(summary)
    total_wall = time.perf_counter() - started_wall
    total_cpu = time.process_time() - started_cpu
    stability = integration_stability_diagnostic(spec, margin=margin, folds=folds)
    measured_triplets = len(cells) * repetitions * whole_types * 2
    seconds_per_triplet = total_cpu / measured_triplets if measured_triplets else float("inf")
    preflight_report_path = (
        episode_root
        / "outputs"
        / "prelaunch"
        / "source_preflight"
        / "preflight_report.json"
    )
    preflight_report = json.loads(preflight_report_path.read_text(encoding="utf-8"))
    if preflight_report["real_connectivity_outcome_values_opened"] is not False:
        raise ValueError("source preflight exposure record is not outcome-blind")
    eligible_types = int(preflight_report["eligible_type_count"])
    development_types = int(preflight_report["development_type_count"])
    observed_searches = int(spec["run_tiers"]["protocol_cost_projection_observed_searches"])
    null_searches = int(spec["run_tiers"]["protocol_cost_projection_null_searches"])
    if observed_searches != 1 or null_searches != policy.null_replicates:
        raise ValueError("qualification cost projection differs from frozen search policy")
    protocol_triplets = (
        development_types
        * 2
        * policy.minimum_valid_trials
        * (observed_searches + null_searches)
    )
    projected_core_hours = seconds_per_triplet * protocol_triplets / 3600.0
    cost_profile = {
        "scope": "generated_data_measured_projection_not_real_runtime",
        "measured_wall_seconds": total_wall,
        "measured_cpu_seconds": total_cpu,
        "measured_type_direction_triplets": measured_triplets,
        "cpu_seconds_per_type_direction_triplet": seconds_per_triplet,
        "annotation_only_eligible_type_count_used": eligible_types,
        "development_type_count_used": development_types,
        "minimum_trials": policy.minimum_valid_trials,
        "observed_plus_null_searches": observed_searches + null_searches,
        "projected_type_direction_triplets": protocol_triplets,
        "projected_cpu_core_hours": projected_core_hours,
        "policy_cpu_core_hour_ceiling": policy.cpu_core_hour_ceiling,
        "cost_gate_passed": projected_core_hours <= policy.cpu_core_hour_ceiling,
        "qualification_configuration": {
            "integration_draws": integration_draws,
            "cross_validation_folds": folds,
            "bootstrap_replicates": bootstrap_replicates,
            "whole_types_per_replicate": whole_types,
        },
    }
    report = {
        "tier": tier,
        "qualification_passed": (
            contract_configuration
            and all(item["passed"] for item in summaries)
            and cost_profile["cost_gate_passed"]
            and stability["passed"]
        ),
        "qualification_scope": "actual_T_U_M_generated_data_only",
        "real_connectivity_accessed": False,
        "real_connectivity_access_authorized": contract[
            "real_connectivity_access_authorized"
        ],
        "full_search_null_executed": False,
        "scientific_contract_enforced": True,
        "model_program": model_program,
        "contract_configuration": contract_configuration,
        "contract_path": str(contract_path),
        "spec_path": str(spec_path),
        "settings": {
            "repetitions_per_cell": repetitions,
            "whole_types_per_replicate": whole_types,
            "integration_draws": integration_draws,
            "folds": folds,
            "bootstrap_replicates": bootstrap_replicates,
            "meaningful_margin": margin,
        },
        "cells": summaries,
        "cost_profile": cost_profile,
        "integration_stability": stability,
        "replicate_results_path": str(replicate_results_path),
    }
    atomic_json(output_dir / "qualification_report.json", report)
    atomic_json(output_dir / "cost_profile.json", cost_profile)
    atomic_json(output_dir / "integration_stability.json", stability)
    _write_csv(output_dir / "result_table.csv", summaries)
    _write_markdown(output_dir / "result_table.md", summaries, report)
    return report
