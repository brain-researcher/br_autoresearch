"""Independent stochastic-oracle coverage gate for the EP12 successor.

The estimand is the infinite-design expectation of the *fixed finite* T/U/M
fit, source-side selection, and reciprocal target-scoring procedure.  An
oracle bank is generated under a seed namespace that is disjoint from the 72
outer qualification replicates.  It is therefore a Monte Carlo approximation
to the same expectation, not a finite-bank replacement estimand.

The fixed 95% familywise level supplies one total error budget.  Its 5% alpha
is split equally between (a) the stochastic-oracle enclosure and (b) the
exact-binomial compatibility checks.  Oracle alpha is split again between the
model-based enclosure and Monte Carlo critical-value error, then spent across
the predeclared looks.  At each look one max-t calculation is simultaneous
over every cell, both directions, and every equal-direction aggregate.
Binomial alpha is Bonferroni-spent across cells and the four coverage readouts.  The union
bound is therefore the original 5%, rather than two independently spent 5%
budgets.  This accounting inherits the contract's paired bootstrap-max-t
assumption for the oracle enclosure; the binomial intervals themselves are
exact.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import beta, binom

from .runtime import EPISODE_ROOT, atomic_json
from .scientific_models import reciprocal_transfer
from .scientific_synthetic import generate_synthetic_type, seed_from
from .simultaneous_inference import (
    SimultaneousInferenceError,
    bilateral_with_aggregate,
    paired_studentized_max_t,
)
from .yaml_subset import load_yaml_subset


SCHEMA_VERSION = "ep12.stochastic_oracle_coverage.v1"
SHARD_SCHEMA_VERSION = "ep12.stochastic_oracle_shard.v1"
READOUTS = ("left_to_right", "right_to_left", "aggregate", "joint")
COORDINATES = ("left_to_right", "right_to_left", "aggregate")


class CoverageSuccessorError(ValueError):
    """The successor coverage evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class CoverageErrorBudget:
    familywise_confidence_level: float
    total_alpha: float
    oracle_enclosure_alpha: float
    oracle_model_alpha: float
    oracle_monte_carlo_alpha: float
    binomial_compatibility_alpha: float
    oracle_look_confidence_level: float
    oracle_monte_carlo_failure_probability_per_look: float
    binomial_readout_confidence_level: float
    cell_count: int
    planned_look_count: int
    readouts_per_cell: int = len(READOUTS)


def derive_error_budget(
    *,
    familywise_confidence_level: float,
    cell_count: int,
    planned_look_count: int,
) -> CoverageErrorBudget:
    """Derive, rather than invent, the successor's multiplicity levels.

    The only supplied level is the frozen familywise 0.95.  The equal split is
    encoded in the V3 contract.  The oracle half is itself split equally
    between model-based enclosure error and Monte Carlo quantile error.  One
    global max-t calculation handles every cell and coordinate at a look, so
    oracle allocation is needed only across planned looks.  Exact binomial
    intervals are allocated across the four
    reported coverage streams in every cell.
    """

    level = float(familywise_confidence_level)
    if not math.isclose(level, 0.95, rel_tol=0.0, abs_tol=1e-12):
        raise CoverageSuccessorError(
            "the successor derives its error budget from the frozen 0.95 level"
        )
    if isinstance(cell_count, bool) or int(cell_count) != cell_count or cell_count < 1:
        raise CoverageSuccessorError("cell_count must be a positive integer")
    if (
        isinstance(planned_look_count, bool)
        or int(planned_look_count) != planned_look_count
        or planned_look_count < 1
    ):
        raise CoverageSuccessorError("planned_look_count must be a positive integer")

    total_alpha = 1.0 - level
    oracle_alpha = total_alpha / 2.0
    binomial_alpha = total_alpha / 2.0
    oracle_model_alpha = oracle_alpha / 2.0
    oracle_monte_carlo_alpha = oracle_alpha / 2.0
    oracle_level = 1.0 - oracle_model_alpha / int(planned_look_count)
    oracle_mc_per_look = oracle_monte_carlo_alpha / int(planned_look_count)
    binomial_level = 1.0 - binomial_alpha / (int(cell_count) * len(READOUTS))
    return CoverageErrorBudget(
        familywise_confidence_level=level,
        total_alpha=total_alpha,
        oracle_enclosure_alpha=oracle_alpha,
        oracle_model_alpha=oracle_model_alpha,
        oracle_monte_carlo_alpha=oracle_monte_carlo_alpha,
        binomial_compatibility_alpha=binomial_alpha,
        oracle_look_confidence_level=oracle_level,
        oracle_monte_carlo_failure_probability_per_look=oracle_mc_per_look,
        binomial_readout_confidence_level=binomial_level,
        cell_count=int(cell_count),
        planned_look_count=int(planned_look_count),
    )


def exact_binomial_interval(
    successes: int,
    total: int,
    *,
    confidence_level: float,
) -> tuple[float, float]:
    """Two-sided Clopper-Pearson interval at an explicitly supplied level."""

    if (
        isinstance(successes, bool)
        or isinstance(total, bool)
        or int(successes) != successes
        or int(total) != total
        or total <= 0
        or successes < 0
        or successes > total
    ):
        raise CoverageSuccessorError("invalid binomial counts")
    level = float(confidence_level)
    if not 0.0 < level < 1.0:
        raise CoverageSuccessorError("binomial confidence level must lie in (0, 1)")
    alpha = 1.0 - level
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2.0, successes, total - successes + 1))
    )
    upper = (
        1.0
        if successes == total
        else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes))
    )
    return lower, upper


def oracle_enclosure(
    directional_rows: Any,
    *,
    confidence_level: float,
    bootstrap_replicates: int,
    seed: int,
    integration_tolerance: float,
) -> dict[str, Any]:
    """Enclose one cell's expected finite-procedure gain.

    Each row is an independent oracle replicate mean.  Its two entries are the
    reciprocal-direction gains; the aggregate is their equal mean.  This path
    never consumes an outer qualification interval.
    """

    rows = np.asarray(directional_rows, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 2 or rows.shape[0] < 2:
        raise CoverageSuccessorError(
            "oracle enclosure needs at least two independent two-direction rows"
        )
    if not np.all(np.isfinite(rows)):
        raise CoverageSuccessorError("oracle rows must be finite")
    tolerance = float(integration_tolerance)
    if not math.isclose(tolerance, 0.002, rel_tol=0.0, abs_tol=1e-12):
        raise CoverageSuccessorError(
            "oracle precision must use the frozen 0.002 integration tolerance"
        )
    try:
        interval = paired_studentized_max_t(
            bilateral_with_aggregate(rows),
            replicates=int(bootstrap_replicates),
            seed=int(seed),
            confidence_level=float(confidence_level),
        )
    except SimultaneousInferenceError as exc:
        raise CoverageSuccessorError(str(exc)) from exc
    lower = np.asarray(interval["lower"], dtype=float)
    upper = np.asarray(interval["upper"], dtype=float)
    half_width = (upper - lower) / 2.0
    return {
        **interval,
        "labels": list(COORDINATES),
        "oracle_replicate_rows": int(rows.shape[0]),
        "estimand": "infinite_design_expected_fixed_finite_procedure_gain",
        "independent_of_outer_qualification": True,
        "integration_tolerance": tolerance,
        "half_width": half_width.tolist(),
        "maximum_half_width": float(np.max(half_width)),
        "precision_met": bool(np.all(half_width <= tolerance)),
    }


def independent_cell_gaussian_max_t(
    rows_by_cell: Sequence[Any],
    *,
    confidence_level: float,
    monte_carlo_draws: int,
    monte_carlo_failure_probability: float,
    seed: int,
    integration_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Efficient global max-t enclosure with independent cell blocks.

    Each cell contributes its empirical 3x3 correlation for the two paired
    directions and their aggregate.  Independent Gaussian multiplier streams
    are simulated for different cells, then one maximum is taken over all
    coordinates.  This avoids both arbitrary cross-cell pairing and the
    O(cells * rows * draws) raw-index bootstrap.  It is the V3 stochastic-
    oracle approximation to the contract's studentized max-t law.
    """

    level = float(confidence_level)
    draws = int(monte_carlo_draws)
    tolerance = float(integration_tolerance)
    mc_failure = float(monte_carlo_failure_probability)
    if not 0.0 < level < 1.0 or draws < 1 or not 0.0 < mc_failure < 1.0:
        raise CoverageSuccessorError("global oracle max-t settings are invalid")
    if not math.isclose(tolerance, 0.002, rel_tol=0.0, abs_tol=1e-12):
        raise CoverageSuccessorError(
            "oracle precision must use the frozen 0.002 integration tolerance"
        )

    estimates: list[np.ndarray] = []
    standard_errors: list[np.ndarray] = []
    correlation_roots: list[np.ndarray] = []
    variable_masks: list[np.ndarray] = []
    row_counts: list[int] = []
    for raw_rows in rows_by_cell:
        values = bilateral_with_aggregate(raw_rows)
        row_counts.append(int(values.shape[0]))
        estimates.append(values.mean(axis=0))
        se = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
        standard_errors.append(se)
        variable = se > 0.0
        variable_masks.append(variable)
        if np.count_nonzero(variable):
            correlation = np.corrcoef(values[:, variable], rowvar=False)
            correlation = np.atleast_2d(np.asarray(correlation, dtype=float))
            eigenvalues, eigenvectors = np.linalg.eigh(correlation)
            root_variable = eigenvectors @ np.diag(
                np.sqrt(np.maximum(eigenvalues, 0.0))
            )
            root = np.zeros((3, root_variable.shape[1]), dtype=float)
            root[variable, :] = root_variable
        else:
            root = np.zeros((3, 1), dtype=float)
        correlation_roots.append(root)

    max_t = np.zeros(draws, dtype=float)
    for cell_index, (root, variable) in enumerate(
        zip(correlation_roots, variable_masks)
    ):
        if not np.any(variable):
            continue
        rng = np.random.Generator(np.random.PCG64(seed_from(seed, cell_index)))
        simulated = rng.standard_normal((draws, root.shape[1])) @ root.T
        max_t = np.maximum(max_t, np.max(np.abs(simulated), axis=1))
    # Let q be the requested population max-t quantile.  The number of Monte
    # Carlo draws below q is Binomial(draws, level).  Select the first order
    # statistic whose probability of falling below q is at most the allocated
    # Monte Carlo error.  This makes simulation error explicit rather than
    # silently treating an empirical quantile as exact.
    if float(binom.sf(draws - 1, draws, level)) > mc_failure:
        raise CoverageSuccessorError(
            "oracle Monte Carlo draws cannot resolve the allocated quantile error"
        )
    low, high = 1, draws
    while low < high:
        candidate = (low + high) // 2
        if float(binom.sf(candidate - 1, draws, level)) <= mc_failure:
            high = candidate
        else:
            low = candidate + 1
    order_one_based = low
    order = order_one_based - 1
    mc_underestimate_bound = float(binom.sf(order_one_based - 1, draws, level))
    critical = float(np.partition(max_t, order)[order])
    if not math.isfinite(critical):
        raise CoverageSuccessorError("global oracle max-t critical value is nonfinite")

    enclosures: list[dict[str, Any]] = []
    for estimate, se, rows in zip(estimates, standard_errors, row_counts):
        lower = estimate - critical * se
        upper = estimate + critical * se
        half_width = (upper - lower) / 2.0
        enclosures.append(
            {
                "method": "independent_cell_gaussian_multiplier_max_t",
                "confidence_level": level,
                "estimate": estimate.tolist(),
                "standard_error": se.tolist(),
                "critical_max_t": critical,
                "lower": lower.tolist(),
                "upper": upper.tolist(),
                "monte_carlo_draws": draws,
                "monte_carlo_failure_probability": mc_failure,
                "monte_carlo_quantile_order_one_based": order_one_based,
                "monte_carlo_underestimate_probability_upper_bound": mc_underestimate_bound,
                "labels": list(COORDINATES),
                "oracle_replicate_rows": rows,
                "estimand": "infinite_design_expected_fixed_finite_procedure_gain",
                "independent_of_outer_qualification": True,
                "independent_cell_multiplier_streams": True,
                "integration_tolerance": tolerance,
                "half_width": half_width.tolist(),
                "maximum_half_width": float(np.max(half_width)),
                "precision_met": bool(np.all(half_width <= tolerance)),
                "simultaneous_cell_count": len(rows_by_cell),
                "simultaneous_coordinate_count": 3 * len(rows_by_cell),
            }
        )
    return enclosures, {
        "method": "independent_cell_gaussian_multiplier_max_t",
        "confidence_level": level,
        "monte_carlo_draws": draws,
        "monte_carlo_failure_probability": mc_failure,
        "monte_carlo_quantile_order_one_based": order_one_based,
        "monte_carlo_underestimate_probability_upper_bound": mc_underestimate_bound,
        "critical_max_t": critical,
        "cell_count": len(rows_by_cell),
        "coordinate_count": 3 * len(rows_by_cell),
        "independent_cell_multiplier_streams": True,
        "raw_index_bootstrap_used": False,
    }


def _coverage_count_bounds(
    outer_intervals: Sequence[Mapping[str, Any]],
    oracle: Mapping[str, Any],
) -> dict[str, tuple[int, int]]:
    oracle_lower = np.asarray(oracle.get("lower"), dtype=float)
    oracle_upper = np.asarray(oracle.get("upper"), dtype=float)
    if (
        oracle_lower.shape != (3,)
        or oracle_upper.shape != (3,)
        or not np.all(np.isfinite(oracle_lower))
        or not np.all(np.isfinite(oracle_upper))
        or np.any(oracle_lower > oracle_upper)
    ):
        raise CoverageSuccessorError("oracle enclosure has invalid bounds")
    if not outer_intervals:
        raise CoverageSuccessorError("coverage needs outer qualification intervals")

    certain = np.zeros(3, dtype=int)
    possible = np.zeros(3, dtype=int)
    joint_certain = 0
    joint_possible = 0
    for item in outer_intervals:
        lower = np.asarray(item.get("lower"), dtype=float)
        upper = np.asarray(item.get("upper"), dtype=float)
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower > upper)
        ):
            raise CoverageSuccessorError("outer qualification interval is invalid")
        row_certain = (lower <= oracle_lower) & (oracle_upper <= upper)
        row_possible = (lower <= oracle_upper) & (oracle_lower <= upper)
        certain += row_certain.astype(int)
        possible += row_possible.astype(int)
        joint_certain += int(np.all(row_certain))
        joint_possible += int(np.all(row_possible))

    return {
        "left_to_right": (int(certain[0]), int(possible[0])),
        "right_to_left": (int(certain[1]), int(possible[1])),
        "aggregate": (int(certain[2]), int(possible[2])),
        "joint": (joint_certain, joint_possible),
    }


def _compatibility_for_count_range(
    *,
    certain_successes: int,
    possible_successes: int,
    total: int,
    nominal_coverage: float,
    confidence_level: float,
) -> dict[str, Any]:
    if not 0 <= certain_successes <= possible_successes <= total:
        raise CoverageSuccessorError("coverage success-count bounds are invalid")
    decisions: list[bool] = []
    intervals: dict[str, list[float]] = {}
    for successes in range(certain_successes, possible_successes + 1):
        bounds = exact_binomial_interval(
            successes,
            total,
            confidence_level=confidence_level,
        )
        intervals[str(successes)] = [float(bounds[0]), float(bounds[1])]
        decisions.append(bounds[0] <= nominal_coverage <= bounds[1])
    if all(decisions):
        status = "compatible_for_every_oracle_consistent_count"
    elif not any(decisions):
        status = "incompatible_for_every_oracle_consistent_count"
    else:
        status = "ambiguous_across_oracle_consistent_counts"
    return {
        "certain_successes": int(certain_successes),
        "possible_successes": int(possible_successes),
        "total": int(total),
        "nominal_coverage": float(nominal_coverage),
        "exact_interval_confidence_level": float(confidence_level),
        "status": status,
        "candidate_count_intervals": intervals,
    }


def assess_cell_coverage(
    *,
    outer_intervals: Sequence[Mapping[str, Any]],
    oracle: Mapping[str, Any],
    nominal_coverage: float,
    binomial_confidence_level: float,
) -> dict[str, Any]:
    """Apply certain/possible coverage counting and fail-closed ambiguity."""

    nominal = float(nominal_coverage)
    if not math.isclose(nominal, 0.95, rel_tol=0.0, abs_tol=1e-12):
        raise CoverageSuccessorError("coverage target must remain the frozen 0.95")
    counts = _coverage_count_bounds(outer_intervals, oracle)
    total = len(outer_intervals)
    readouts = {
        name: _compatibility_for_count_range(
            certain_successes=bounds[0],
            possible_successes=bounds[1],
            total=total,
            nominal_coverage=nominal,
            confidence_level=binomial_confidence_level,
        )
        for name, bounds in counts.items()
    }
    statuses = {value["status"] for value in readouts.values()}
    precision_met = oracle.get("precision_met") is True
    if not precision_met:
        status = "unresolved_oracle_precision"
    elif "ambiguous_across_oracle_consistent_counts" in statuses:
        status = "unresolved_oracle_ambiguity"
    elif "incompatible_for_every_oracle_consistent_count" in statuses:
        status = "coverage_incompatible"
    elif statuses == {"compatible_for_every_oracle_consistent_count"}:
        status = "coverage_qualified"
    else:  # pragma: no cover - closed status vocabulary above
        raise CoverageSuccessorError("coverage readout statuses are inconsistent")
    return {
        "status": status,
        "passed": status == "coverage_qualified",
        "oracle_precision_met": precision_met,
        "oracle_maximum_half_width": oracle.get("maximum_half_width"),
        "readouts": readouts,
        "outer_replicates": total,
        "fail_closed_on_ambiguity": True,
    }


def next_oracle_look(
    *,
    completed_rows: int,
    planned_rows: Sequence[int],
    precision_met: bool,
) -> dict[str, Any]:
    """Return the bounded adaptive action without inventing a stop threshold."""

    looks = [int(value) for value in planned_rows]
    if not looks or looks != sorted(set(looks)) or looks[0] < 2:
        raise CoverageSuccessorError(
            "planned oracle rows must be unique increasing integers starting at >=2"
        )
    if completed_rows not in looks:
        raise CoverageSuccessorError("completed_rows must equal a predeclared look")
    index = looks.index(int(completed_rows))
    if precision_met:
        return {
            "action": "oracle_precision_complete",
            "completed_rows": int(completed_rows),
            "next_rows": None,
            "at_bound": index == len(looks) - 1,
        }
    if index + 1 < len(looks):
        return {
            "action": "refine_oracle_bank",
            "completed_rows": int(completed_rows),
            "next_rows": looks[index + 1],
            "additional_rows": looks[index + 1] - int(completed_rows),
            "at_bound": False,
        }
    return {
        "action": "unresolved_at_bounded_oracle_cost",
        "completed_rows": int(completed_rows),
        "next_rows": None,
        "at_bound": True,
    }


def project_oracle_cost(
    *,
    measured_cpu_seconds_per_whole_type: float,
    cell_count: int,
    whole_types_per_oracle_replicate: int,
    maximum_oracle_replicates_per_cell: int,
) -> dict[str, Any]:
    """Project the full bounded bank from a measured generated-only pilot."""

    seconds = float(measured_cpu_seconds_per_whole_type)
    integers = (
        int(cell_count),
        int(whole_types_per_oracle_replicate),
        int(maximum_oracle_replicates_per_cell),
    )
    if not math.isfinite(seconds) or seconds <= 0.0 or any(value < 1 for value in integers):
        raise CoverageSuccessorError("cost projection inputs must be positive and finite")
    whole_types = math.prod(integers)
    cpu_seconds = seconds * whole_types
    return {
        "measured_cpu_seconds_per_whole_type": seconds,
        "cell_count": integers[0],
        "whole_types_per_oracle_replicate": integers[1],
        "maximum_oracle_replicates_per_cell": integers[2],
        "maximum_generated_whole_types": whole_types,
        "projected_cpu_seconds": cpu_seconds,
        "projected_cpu_core_hours": cpu_seconds / 3600.0,
        "projection_required_before_full_bank_submission": True,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _oracle_replicate(
    *,
    cell: Any,
    replicate_index: int,
    whole_types: int,
    seed_namespace: str,
    integration_draws: int,
    folds: int,
    latent_rank: int,
    ridge: float,
) -> tuple[list[float], float, float]:
    """Run the actual finite T/U/M procedure on fresh generated whole types."""

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    rows: list[list[float]] = []
    for type_index in range(int(whole_types)):
        fixture = generate_synthetic_type(
            cell.scenario,
            left_neurons=cell.neurons_per_side,
            right_neurons=cell.neurons_per_side,
            count_total=cell.count_total,
            strength=cell.strength,
            missing_fraction=cell.missing_fraction,
            seed=seed_from(
                seed_namespace,
                cell.cell_id,
                replicate_index,
                type_index,
                "fixture",
            ),
        )
        result = reciprocal_transfer(
            fixture.left.counts,
            fixture.left.covariates,
            fixture.right.counts,
            fixture.right.covariates,
            seed=seed_from(
                seed_namespace,
                cell.cell_id,
                replicate_index,
                type_index,
                "fit",
            ),
            integration_draws=int(integration_draws),
            folds=int(folds),
            latent_rank=int(latent_rank),
            ridge=float(ridge),
        )
        rows.append(
            [
                float(result["left_to_right"]["mean_delta_M_minus_reference"]),
                float(result["right_to_left"]["mean_delta_M_minus_reference"]),
            ]
        )
    values = np.asarray(rows, dtype=float)
    if values.shape != (int(whole_types), 2) or not np.all(np.isfinite(values)):
        raise CoverageSuccessorError("oracle replicate produced invalid directional gains")
    return (
        values.mean(axis=0).tolist(),
        time.perf_counter() - started_wall,
        time.process_time() - started_cpu,
    )


def _tier_settings(spec: Mapping[str, Any], tier: str) -> Mapping[str, Any]:
    settings = spec.get("stochastic_oracle", {}).get("tiers", {}).get(tier)
    if not isinstance(settings, Mapping):
        raise CoverageSuccessorError(f"unknown stochastic-oracle tier: {tier}")
    return settings


def run_oracle_shard(
    *,
    tier: str,
    look_index: int,
    cell_index: int,
    output_dir: Path,
    contract_path: Path,
    qualification_spec_path: Path,
    episode_root: Path = EPISODE_ROOT,
) -> dict[str, Any]:
    """Generate one cell's incremental independent oracle-bank shard.

    Full-bank parallelism is one array task per cell and look.  A shard covers
    only the rows added at its look, so a failed task can be retried without
    regenerating completed cells or silently absorbing the bank into the
    serial outer-qualification wrapper.
    """

    from .scientific_qualification import _contract_settings, qualification_cells

    episode_root = episode_root.resolve()
    output_dir = output_dir.resolve()
    allowed = (episode_root / "outputs").resolve()
    if allowed not in output_dir.parents:
        raise CoverageSuccessorError("oracle shards must stay under episode outputs")
    contract_path = contract_path.resolve()
    qualification_spec_path = qualification_spec_path.resolve()
    if contract_path.name != "SCIENTIFIC_CONTRACT_POST36_V3.yaml":
        raise CoverageSuccessorError("oracle shards require the additive V3 contract")
    if qualification_spec_path.name != "SYNTHETIC_QUALIFICATION_SPEC_POST36_V3.yaml":
        raise CoverageSuccessorError("oracle shards require the additive V3 specification")
    contract = load_yaml_subset(contract_path)
    spec = load_yaml_subset(qualification_spec_path)
    program = _contract_settings(contract)
    tier_spec = _tier_settings(spec, tier)
    cell_tier = str(tier_spec["qualification_cell_tier"])
    cells = qualification_cells(cell_tier, spec)
    cell_limit = int(tier_spec.get("cell_limit", len(cells)))
    cells = cells[:cell_limit]
    if not 0 <= int(cell_index) < len(cells):
        raise CoverageSuccessorError("cell_index is outside the tier inventory")
    looks = [int(value) for value in tier_spec["planned_oracle_replicates_per_cell"]]
    if not 0 <= int(look_index) < len(looks):
        raise CoverageSuccessorError("look_index is outside the tier schedule")
    stop = looks[int(look_index)]
    start = 0 if int(look_index) == 0 else looks[int(look_index) - 1]
    scientific_gate = bool(tier_spec["scientific_gate"])
    integration_draws = int(
        program["integration_draws"]
        if scientific_gate
        else tier_spec["integration_draws"]
    )
    folds = int(program["source_cv_folds"])
    whole_types = int(tier_spec["whole_types_per_oracle_replicate"])
    seed_namespace = str(spec["stochastic_oracle"]["seed_namespace"])
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for replicate_index in range(start, stop):
        try:
            gain, wall_seconds, cpu_seconds = _oracle_replicate(
                cell=cells[int(cell_index)],
                replicate_index=replicate_index,
                whole_types=whole_types,
                seed_namespace=seed_namespace,
                integration_draws=integration_draws,
                folds=folds,
                latent_rank=int(program["linear_rank"]),
                ridge=float(program["nuisance_ridge"]),
            )
            records.append(
                {
                    "replicate_index": replicate_index,
                    "directional_gain": gain,
                    "wall_seconds": wall_seconds,
                    "cpu_seconds": cpu_seconds,
                }
            )
        except Exception as exc:  # retain and fail closed in aggregation
            failures.append(
                {
                    "replicate_index": replicate_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    report = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "tier": tier,
        "look_index": int(look_index),
        "cell_index": int(cell_index),
        "cell": asdict(cells[int(cell_index)]),
        "replicate_start_inclusive": start,
        "replicate_stop_exclusive": stop,
        "whole_types_per_oracle_replicate": whole_types,
        "records": records,
        "failures": failures,
        "complete": len(records) == stop - start and not failures,
        "scientific_gate": scientific_gate,
        "seed_namespace": seed_namespace,
        "outer_qualification_seed_namespace_used": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "contract_path": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "qualification_spec_path": str(qualification_spec_path),
        "qualification_spec_sha256": _sha256(qualification_spec_path),
        "settings": {
            "integration_draws": integration_draws,
            "folds": folds,
            "latent_rank": int(program["linear_rank"]),
            "ridge": float(program["nuisance_ridge"]),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"cell_{int(cell_index):03d}.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite oracle shard: {destination}")
    atomic_json(destination, report)
    return report


def aggregate_oracle_look(
    *,
    tier: str,
    look_index: int,
    bank_root: Path,
    output_path: Path,
    qualification_spec_path: Path,
    episode_root: Path = EPISODE_ROOT,
) -> dict[str, Any]:
    """Verify all incremental shards through a look and publish enclosures."""

    from .scientific_qualification import qualification_cells

    episode_root = episode_root.resolve()
    bank_root = bank_root.resolve()
    output_path = output_path.resolve()
    allowed = (episode_root / "outputs").resolve()
    if allowed not in bank_root.parents or allowed not in output_path.parents:
        raise CoverageSuccessorError("oracle bank artifacts must stay under outputs")
    spec = load_yaml_subset(qualification_spec_path)
    tier_spec = _tier_settings(spec, tier)
    cells = qualification_cells(str(tier_spec["qualification_cell_tier"]), spec)
    cells = cells[: int(tier_spec.get("cell_limit", len(cells)))]
    looks = [int(value) for value in tier_spec["planned_oracle_replicates_per_cell"]]
    if not 0 <= int(look_index) < len(looks):
        raise CoverageSuccessorError("look_index is outside the tier schedule")
    budget = derive_error_budget(
        familywise_confidence_level=float(
            spec["stochastic_oracle"]["familywise_confidence_level"]
        ),
        cell_count=len(cells),
        planned_look_count=len(looks),
    )
    bootstrap_replicates = int(tier_spec["oracle_monte_carlo_draws"])
    tolerance = float(spec["stochastic_oracle"]["integration_tolerance"])
    expected_stop = looks[int(look_index)]
    rows_by_cell: list[np.ndarray] = []
    total_cpu = 0.0
    total_wall = 0.0
    for cell_index, cell in enumerate(cells):
        records: dict[int, list[float]] = {}
        for prior_look in range(int(look_index) + 1):
            shard_path = (
                bank_root
                / f"look_{prior_look:02d}"
                / f"cell_{cell_index:03d}.json"
            )
            try:
                shard = json.loads(shard_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise CoverageSuccessorError(
                    f"cannot read oracle shard {shard_path}: {exc}"
                ) from exc
            if (
                shard.get("schema_version") != SHARD_SCHEMA_VERSION
                or shard.get("complete") is not True
                or shard.get("real_connectivity_accessed") is not False
                or shard.get("final_connectivity_accessed") is not False
                or int(shard.get("cell_index", -1)) != cell_index
                or shard.get("cell", {}).get("cell_id") != cell.cell_id
            ):
                raise CoverageSuccessorError(
                    f"oracle shard identity or completion failed: {shard_path}"
                )
            for record in shard.get("records", []):
                index = int(record["replicate_index"])
                if index in records:
                    raise CoverageSuccessorError("oracle replicate is duplicated")
                gain = np.asarray(record["directional_gain"], dtype=float)
                if gain.shape != (2,) or not np.all(np.isfinite(gain)):
                    raise CoverageSuccessorError("oracle shard contains an invalid gain")
                records[index] = gain.tolist()
                total_cpu += float(record["cpu_seconds"])
                total_wall += float(record["wall_seconds"])
        if set(records) != set(range(expected_stop)):
            raise CoverageSuccessorError(
                f"oracle cell {cell.cell_id} is incomplete through row {expected_stop}"
            )
        rows_by_cell.append(
            np.asarray([records[index] for index in range(expected_stop)], dtype=float)
        )

    enclosures, global_interval = independent_cell_gaussian_max_t(
        rows_by_cell,
        confidence_level=budget.oracle_look_confidence_level,
        monte_carlo_draws=bootstrap_replicates,
        monte_carlo_failure_probability=(
            budget.oracle_monte_carlo_failure_probability_per_look
        ),
        seed=seed_from(
            spec["stochastic_oracle"]["seed_namespace"],
            tier,
            look_index,
            "global-oracle-enclosure",
        ),
        integration_tolerance=tolerance,
    )
    cell_results: list[dict[str, Any]] = []
    for cell, enclosure in zip(cells, enclosures):
        cell_results.append({"cell": asdict(cell), "oracle_enclosure": enclosure})

    precision_met = all(
        item["oracle_enclosure"]["precision_met"] for item in cell_results
    )
    action = next_oracle_look(
        completed_rows=expected_stop,
        planned_rows=looks,
        precision_met=precision_met,
    )
    measured_whole_types = (
        len(cells)
        * expected_stop
        * int(tier_spec["whole_types_per_oracle_replicate"])
    )
    qualification_tier = _tier_settings(spec, "qualification")
    qualification_cells_count = len(qualification_cells("qualification", spec))
    cost_projection = project_oracle_cost(
        measured_cpu_seconds_per_whole_type=total_cpu / measured_whole_types,
        cell_count=qualification_cells_count,
        whole_types_per_oracle_replicate=int(
            qualification_tier["whole_types_per_oracle_replicate"]
        ),
        maximum_oracle_replicates_per_cell=max(
            int(value)
            for value in qualification_tier["planned_oracle_replicates_per_cell"]
        ),
    )
    qualification_looks = [
        int(value)
        for value in qualification_tier["planned_oracle_replicates_per_cell"]
    ]
    qualification_increments = [
        value - (qualification_looks[index - 1] if index else 0)
        for index, value in enumerate(qualification_looks)
    ]
    maximum_shard_whole_types = max(qualification_increments) * int(
        qualification_tier["whole_types_per_oracle_replicate"]
    )
    projected_maximum_shard_cpu_seconds = (
        cost_projection["measured_cpu_seconds_per_whole_type"]
        * maximum_shard_whole_types
    )
    cost_projection.update(
        {
            "maximum_incremental_oracle_replicates_per_cell": max(
                qualification_increments
            ),
            "maximum_whole_types_in_one_cell_look_shard": maximum_shard_whole_types,
            "projected_maximum_shard_cpu_seconds": projected_maximum_shard_cpu_seconds,
            "explicit_shard_wall_limit_seconds": 12 * 60 * 60,
            "full_bank_submission_ready": (
                projected_maximum_shard_cpu_seconds <= 12 * 60 * 60
            ),
        }
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "tier": tier,
        "look_index": int(look_index),
        "completed_oracle_replicates_per_cell": expected_stop,
        "whole_types_per_oracle_replicate": int(
            tier_spec["whole_types_per_oracle_replicate"]
        ),
        "cell_count": len(cells),
        "cells": cell_results,
        "all_cells_precision_met": precision_met,
        "next_action": action,
        "error_budget": asdict(budget),
        "error_accounting": {
            "rule": "equal_split_then_union_bound",
            "oracle_all_cells_and_coordinates": "independent_cell_gaussian_multiplier_max_t",
            "oracle_across_looks": "bonferroni",
            "binomial_across_cells_and_readouts": "bonferroni",
            "combined_alpha_upper_bound": budget.total_alpha,
            "same_five_percent_spent_twice": False,
            "global_oracle_interval": global_interval,
        },
        "measured_cpu_seconds": total_cpu,
        "summed_task_wall_seconds": total_wall,
        "bounded_full_bank_cost_projection": cost_projection,
        "scientific_gate": bool(tier_spec["scientific_gate"]),
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
    }
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite oracle aggregate: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output_path, report)
    return report


def coverage_evidence_by_cell(
    report: Mapping[str, Any],
    *,
    expected_cell_ids: Sequence[str],
) -> tuple[dict[str, Mapping[str, Any]], CoverageErrorBudget]:
    """Validate an aggregate for use by the serial outer qualifier."""

    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("scientific_gate") is not True
        or report.get("all_cells_precision_met") is not True
        or report.get("real_connectivity_accessed") is not False
        or report.get("final_connectivity_accessed") is not False
    ):
        raise CoverageSuccessorError("oracle aggregate is not qualification-ready")
    cells = report.get("cells")
    if not isinstance(cells, list):
        raise CoverageSuccessorError("oracle aggregate lacks cells")
    indexed: dict[str, Mapping[str, Any]] = {}
    for item in cells:
        if not isinstance(item, Mapping) or not isinstance(item.get("cell"), Mapping):
            raise CoverageSuccessorError("oracle aggregate cell entry is invalid")
        cell_id = str(item["cell"].get("cell_id", ""))
        enclosure = item.get("oracle_enclosure")
        if not cell_id or cell_id in indexed or not isinstance(enclosure, Mapping):
            raise CoverageSuccessorError("oracle aggregate cell identity is invalid")
        indexed[cell_id] = enclosure
    if set(indexed) != set(expected_cell_ids):
        raise CoverageSuccessorError("oracle aggregate cell inventory differs from qualification")
    raw_budget = report.get("error_budget")
    if not isinstance(raw_budget, Mapping):
        raise CoverageSuccessorError("oracle aggregate lacks its error budget")
    try:
        budget = CoverageErrorBudget(**{key: raw_budget[key] for key in CoverageErrorBudget.__dataclass_fields__})
    except (KeyError, TypeError, ValueError) as exc:
        raise CoverageSuccessorError("oracle aggregate error budget is invalid") from exc
    expected = derive_error_budget(
        familywise_confidence_level=budget.familywise_confidence_level,
        cell_count=len(expected_cell_ids),
        planned_look_count=budget.planned_look_count,
    )
    if budget != expected:
        raise CoverageSuccessorError("oracle aggregate error budget was not mechanically derived")
    return indexed, budget


__all__ = [
    "COORDINATES",
    "CoverageErrorBudget",
    "CoverageSuccessorError",
    "READOUTS",
    "SCHEMA_VERSION",
    "SHARD_SCHEMA_VERSION",
    "aggregate_oracle_look",
    "assess_cell_coverage",
    "coverage_evidence_by_cell",
    "derive_error_budget",
    "exact_binomial_interval",
    "independent_cell_gaussian_max_t",
    "next_oracle_look",
    "oracle_enclosure",
    "project_oracle_cost",
    "run_oracle_shard",
]
