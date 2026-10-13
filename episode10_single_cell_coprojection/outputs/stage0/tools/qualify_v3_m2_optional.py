#!/usr/bin/env python3
"""Target-free qualification for the EP10 v3 M2 optional families.

This executable has no empirical data inputs. It uses the frozen synthetic population
fixture from the already-passed core qualification, verifies the optional
trigger diagnostic, and tests identified-probability recovery for every
authorized low-rank rank and for the rank-plus-16-named-edge hybrid family.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import fit_v3_m1 as m1
import fit_v3_m2 as m2
import fit_v3_m2_optional as optional
import qualify_v3_m2_core as core_qualification
import v3_m2_optional_diagnostic_v1 as diagnostic
import v3_search_controller_v1 as controller


RECOVERY_RHO = 0.001
RECOVERY_KL_MAXIMUM_BITS = 1.0e-5
PLANTED_GRADIENT_TOLERANCE = 1.0e-10
LOW_RANK_RANKS = (1, 2, 3, 4, 6, 8)
HYBRID_NAMED_EDGE_COUNT = 16
LN2 = math.log(2.0)
RECOVERY_FAMILIES = (
    "low_rank_population_recovery",
    "sparse_plus_low_rank_population_recovery",
)
# These three pre-existing atomic cases lack fixture metadata. Their original
# producer and unchanged fixture were established before this resume repair.
# Missing context is accepted only when content matches this durable source.
LEGACY_COMPLETED_CASES = (
    Path(__file__).resolve().parents[1]
    / "v3_m2_optional_family_qualification_job_46115632" / "completed_cases"
)


class OptionalQualificationError(RuntimeError):
    """A required optional-family qualification gate failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OptionalQualificationError(message)


def _maximum_absolute(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=float)
    _require(bool(np.all(np.isfinite(array))), "qualification value is nonfinite")
    return float(np.max(np.abs(array))) if array.size else 0.0


def _dct_columns(rank: int) -> np.ndarray:
    rows = np.arange(m1.PANEL_SIZE, dtype=float)[:, np.newaxis]
    columns = np.arange(rank, dtype=float)[np.newaxis, :]
    result = np.cos(math.pi * (rows + 0.5) * columns / m1.PANEL_SIZE)
    result[:, 0] *= 1.0 / math.sqrt(float(m1.PANEL_SIZE))
    if rank > 1:
        result[:, 1:] *= math.sqrt(2.0 / float(m1.PANEL_SIZE))
    return result


def _planted_low_rank_state(rank: int) -> optional.LowRankState:
    signs = np.where(np.arange(rank) % 2 == 0, 1.0, -1.0)
    magnitudes = 0.08 / np.sqrt(np.arange(1, rank + 1, dtype=float))
    return optional.LowRankState(_dct_columns(rank), signs * magnitudes)


def _identified_comparison(
    planted_beta: np.ndarray,
    fitted_beta: np.ndarray,
    fixed_m1_theta: np.ndarray,
    prepared: m2.PreparedM2,
    fixture: Any,
) -> Mapping[str, float]:
    comparison = m2.compare_conditional_probabilities(
        planted_beta,
        fitted_beta,
        fixed_m1_theta,
        prepared,
        fixture.candidates,
        fixture.geometry,
        fixture.pairs,
    )
    result = {
        "mean_conditional_KL_bits": float(comparison["mean_KL_nats"]) / LN2,
        "maximum_candidate_probability_absolute_error": float(
            comparison["maximum_candidate_probability_absolute_error"]
        ),
        "mean_inclusion_marginal_absolute_error": float(
            comparison["mean_inclusion_marginal_absolute_error"]
        ),
    }
    _require(
        all(math.isfinite(value) and value >= 0.0 for value in result.values()),
        "identified-probability comparison is nonfinite or negative",
    )
    return result


def qualify_optional_trigger_diagnostic() -> Mapping[str, Any]:
    membership = np.zeros(m1.PANEL_SIZE, dtype=float)
    membership[:8] = 1.0
    observed = np.asarray(
        [membership[left] * membership[right] for left, right in m2.PAIR_INDEX],
        dtype=float,
    )
    positive = 0.20
    negative = math.comb(8, 2) * positive / (
        m2.PAIR_DIMENSION - math.comb(8, 2)
    )
    residual = np.where(observed == 1.0, positive, -negative)
    diffuse_rows = tuple(
        diagnostic.PairResidualRow(
            f"synthetic_diffuse_{group_index:02d}",
            group,
            8,
            membership,
            observed - residual,
        )
        for group_index, group in enumerate(m1.GROUP_ORDER)
    )
    diffuse = diagnostic.diagnose_optional_families(diffuse_rows)
    localized_contrasts = np.zeros(m2.PAIR_DIMENSION, dtype=float)
    localized_contrasts[:HYBRID_NAMED_EDGE_COUNT] = np.linspace(
        1.0, 0.25, HYBRID_NAMED_EDGE_COUNT
    )
    localized = controller.localized_sparse_selection(localized_contrasts)
    activation = controller.optional_activation(diffuse.diagnostics, localized)
    _require(activation.low_rank, "computed diffuse diagnostic did not trigger low rank")
    _require(
        localized.eligible and activation.sparse_plus_low_rank,
        "computed diffuse diagnostic plus localized contrasts did not trigger hybrid",
    )

    zero_rows = tuple(
        diagnostic.PairResidualRow(
            f"synthetic_zero_{group_index:02d}",
            group,
            8,
            membership,
            observed,
        )
        for group_index, group in enumerate(m1.GROUP_ORDER)
    )
    zero = diagnostic.diagnose_optional_families(zero_rows)
    zero_activation = controller.optional_activation(
        zero.diagnostics,
        controller.localized_sparse_selection(np.zeros(m2.PAIR_DIMENSION)),
    )
    _require(
        not zero_activation.low_rank and not zero_activation.sparse_plus_low_rank,
        "exact-zero diagnostic did not retain its nontrigger disposition",
    )
    return {
        "status": "pass",
        "diffuse_leading_energy": diffuse.diagnostics.leading_energy,
        "diffuse_edge_participation_ratio": (
            diffuse.diagnostics.leading_edge_participation_ratio
        ),
        "diffuse_median_LOGO_cosine": (
            diffuse.diagnostics.median_leave_one_group_cosine
        ),
        "diffuse_LOGO_cosines_at_least_0_70": (
            diffuse.diagnostics.leave_one_group_cosines_at_least_0_70
        ),
        "localized_named_edge_count": len(localized.edges),
        "low_rank_triggered": activation.low_rank,
        "sparse_plus_low_rank_triggered": activation.sparse_plus_low_rank,
        "exact_zero_nontrigger": True,
    }


def qualify_low_rank_population_recovery(
    fixture: Any,
    prepared: m2.PreparedM2,
    *,
    ranks: Sequence[int] = LOW_RANK_RANKS,
) -> Mapping[str, Any]:
    cases: dict[str, Any] = {}
    for rank in ranks:
        planted_state = _planted_low_rank_state(rank)
        planted_beta = optional.low_rank_effective_beta(
            planted_state, prepared.transform
        )
        correction = m2.population_linear_correction(
            planted_beta,
            fixture.fixed_m1_theta,
            prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
        )
        planted_gradient_error = _maximum_absolute(
            m2.evaluate_m2(
                planted_beta,
                fixture.fixed_m1_theta,
                prepared,
                fixture.candidates,
                fixture.geometry,
                fixture.pairs,
                linear_correction=correction,
            ).gradient
        )
        _require(
            planted_gradient_error <= PLANTED_GRADIENT_TOLERANCE,
            f"rank-{rank} population correction did not center the plant",
        )
        scales = optional.fold_low_rank_scales_synthetic_qualification(
            fixture.fixed_m1_theta,
            prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
            linear_correction=correction,
        )
        specification = optional.LowRankSpec(
            rank,
            RECOVERY_RHO,
            scales.lambda_reference,
        )
        arguments = (
            fixture.fixed_m1_theta,
            prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
            specification,
        )
        first = optional.fit_low_rank_synthetic_qualification(
            *arguments, linear_correction=correction
        )
        second = optional.fit_low_rank_synthetic_qualification(
            *arguments, linear_correction=correction
        )
        _require(
            first.status == second.status == "pass",
            f"rank-{rank} optional fit did not pass",
        )
        deterministic_repeat = bool(
            first.objective == second.objective
            and first.telemetry == second.telemetry
            and np.array_equal(first.beta, second.beta)
            and np.array_equal(first.state.U, second.state.U)
            and np.array_equal(first.state.d, second.state.d)
        )
        _require(
            deterministic_repeat,
            f"rank-{rank} low-rank recovery did not repeat bit exactly",
        )
        comparison = _identified_comparison(
            planted_beta,
            first.beta,
            fixture.fixed_m1_theta,
            prepared,
            fixture,
        )
        _require(
            comparison["mean_conditional_KL_bits"] <= RECOVERY_KL_MAXIMUM_BITS,
            f"rank-{rank} recovery KL exceeds the frozen optional threshold",
        )
        cases[str(rank)] = {
            "status": "pass",
            "rho": RECOVERY_RHO,
            "lambda_reference": scales.lambda_reference,
            "population_correction_planted_gradient_maximum_absolute_error": (
                planted_gradient_error
            ),
            "fit_iterations": first.iterations,
            "fit_gradient_infinity": first.gradient_infinity,
            "fit_relative_objective_change": first.relative_objective_change,
            "deterministic_repeat_bit_exact": deterministic_repeat,
            **comparison,
        }
    return {
        "status": "pass",
        "ranks": list(ranks),
        "every_authorized_rank_qualified": set(map(int, cases)) == set(ranks),
        "mean_conditional_KL_maximum_bits": RECOVERY_KL_MAXIMUM_BITS,
        "cases": cases,
    }


def qualify_hybrid_population_recovery(
    fixture: Any,
    prepared: m2.PreparedM2,
    *,
    ranks: Sequence[int] = LOW_RANK_RANKS,
) -> Mapping[str, Any]:
    named_edges = tuple(m2.PAIR_INDEX[:HYBRID_NAMED_EDGE_COUNT])
    sparse = 0.045 * np.where(
        np.arange(HYBRID_NAMED_EDGE_COUNT) % 2 == 0,
        1.0,
        -1.0,
    )
    cases: dict[str, Any] = {}
    for rank in ranks:
        planted_low_rank = _planted_low_rank_state(rank)
        coordinate_specification = optional.HybridSpec(
            rank,
            RECOVERY_RHO,
            RECOVERY_RHO,
            1.0,
            1.0,
            named_edges,
        )
        planted_state = optional.HybridState(planted_low_rank, sparse)
        planted_beta = optional.hybrid_effective_beta(
            planted_state,
            coordinate_specification,
            prepared.transform,
        )
        correction = m2.population_linear_correction(
            planted_beta,
            fixture.fixed_m1_theta,
            prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
        )
        planted_gradient_error = _maximum_absolute(
            m2.evaluate_m2(
                planted_beta,
                fixture.fixed_m1_theta,
                prepared,
                fixture.candidates,
                fixture.geometry,
                fixture.pairs,
                linear_correction=correction,
            ).gradient
        )
        _require(
            planted_gradient_error <= PLANTED_GRADIENT_TOLERANCE,
            f"rank-{rank} hybrid population correction did not center the plant",
        )
        scales = optional.fold_low_rank_scales_synthetic_qualification(
            fixture.fixed_m1_theta,
            prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
            linear_correction=correction,
        )
        lasso_maximum = m2.fold_lambda_maximum(
            fixture.fixed_m1_theta,
            prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
            "lasso",
            linear_correction=correction,
        )
        _require(
            lasso_maximum.value > 0.0,
            f"rank-{rank} hybrid recovery lasso lambda-max is zero",
        )

        anchor_specification = optional.LowRankSpec(
            rank,
            0.1,
            scales.lambda_reference,
        )
        selected_anchor = optional.fit_low_rank_synthetic_qualification(
            fixture.fixed_m1_theta,
            prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
            anchor_specification,
            linear_correction=correction,
        )
        specification = optional.HybridSpec(
            rank,
            RECOVERY_RHO,
            RECOVERY_RHO,
            scales.lambda_reference,
            lasso_maximum.value,
            named_edges,
        )
        arguments = (
            fixture.fixed_m1_theta,
            prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
            specification,
            selected_anchor,
        )
        first = optional.fit_sparse_plus_low_rank_synthetic_qualification(
            *arguments,
            linear_correction=correction,
        )
        second = optional.fit_sparse_plus_low_rank_synthetic_qualification(
            *arguments,
            linear_correction=correction,
        )
        _require(
            first.status == second.status == "pass",
            f"rank-{rank} hybrid recovery fit did not pass",
        )
        deterministic_repeat = bool(
            first.objective == second.objective
            and first.telemetry == second.telemetry
            and np.array_equal(first.beta, second.beta)
            and np.array_equal(first.state.low_rank.U, second.state.low_rank.U)
            and np.array_equal(first.state.low_rank.d, second.state.low_rank.d)
            and np.array_equal(
                first.state.sparse_coefficients,
                second.state.sparse_coefficients,
            )
        )
        _require(
            deterministic_repeat,
            f"rank-{rank} hybrid recovery did not repeat bit exactly",
        )
        comparison = _identified_comparison(
            planted_beta,
            first.beta,
            fixture.fixed_m1_theta,
            prepared,
            fixture,
        )
        _require(
            comparison["mean_conditional_KL_bits"] <= RECOVERY_KL_MAXIMUM_BITS,
            f"rank-{rank} hybrid recovery KL exceeds the frozen optional threshold",
        )
        cases[str(rank)] = {
            "status": "pass",
            "rank": rank,
            "named_sparse_edges": HYBRID_NAMED_EDGE_COUNT,
            "rho_low": RECOVERY_RHO,
            "rho_sparse": RECOVERY_RHO,
            "selected_anchor_rho": 0.1,
            "lambda_reference": scales.lambda_reference,
            "core_lasso_lambda_maximum": lasso_maximum.value,
            "population_correction_planted_gradient_maximum_absolute_error": (
                planted_gradient_error
            ),
            "fit_sweeps": first.sweeps,
            "fit_joint_stationarity_infinity": (
                first.joint_stationarity_infinity
            ),
            "fit_relative_objective_change": first.relative_objective_change,
            "deterministic_repeat_bit_exact": deterministic_repeat,
            **comparison,
        }
    return {
        "status": "pass",
        "ranks": list(ranks),
        "every_possible_activated_rank_qualified": (
            set(map(int, cases)) == set(ranks)
        ),
        "named_sparse_edges": HYBRID_NAMED_EDGE_COUNT,
        "mean_conditional_KL_maximum_bits": RECOVERY_KL_MAXIMUM_BITS,
        "cases": cases,
    }


def _validate_runtime(workers: int = 1) -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not job_id.isdigit():
        raise OptionalQualificationError("optional qualification requires Slurm")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(name) != "1":
            raise OptionalQualificationError(f"{name} must equal 1")
    _require(1 <= workers <= 2 * len(LOW_RANK_RANKS), "invalid worker count")
    if workers > 1:
        _require(
            workers <= int(os.environ.get("SLURM_CPUS_PER_TASK", "0")),
            "qualification workers exceed requested Slurm CPUs",
        )


def _run_recovery_case(family: str, rank: int) -> tuple[str, int, Mapping[str, Any]]:
    """Keep both repeats, and a hybrid's live anchor, in one child process."""
    _validate_runtime()
    fixture = core_qualification._frozen_fixture()
    transform, prepared = m2.fit_residual_transform(
        fixture.cells, fixture.moments, fixture.candidates,
        fixture.geometry, fixture.pairs,
    )
    _require(bool(transform.diagnostics["rank_stable"]), "unstable worker rank")
    if family == "low_rank_population_recovery":
        gate = qualify_low_rank_population_recovery(fixture, prepared, ranks=(rank,))
    elif family == "sparse_plus_low_rank_population_recovery":
        gate = qualify_hybrid_population_recovery(fixture, prepared, ranks=(rank,))
    else:
        raise OptionalQualificationError("unknown recovery family")
    return family, rank, gate


def _case_context(family: str, rank: int) -> Mapping[str, Any]:
    return {
        "family": family, "rank": rank,
        "seed": core_qualification.FROZEN_SEED,
        "groups": core_qualification.FROZEN_GROUPS,
        "rows_per_group": core_qualification.FROZEN_ROWS_PER_GROUP,
        "every_K_0_through_16_once_per_group": True,
        "recovery_rho": RECOVERY_RHO,
        "mean_conditional_KL_maximum_bits": RECOVERY_KL_MAXIMUM_BITS,
        "planted_gradient_tolerance": PLANTED_GRADIENT_TOLERANCE,
        "gradient_tolerance": optional.LOW_RANK_GRADIENT_INFINITY_TOLERANCE,
        "relative_objective_tolerance": optional.LOW_RANK_RELATIVE_OBJECTIVE_TOLERANCE,
        "maximum_iterations": optional.LOW_RANK_MAXIMUM_ITERATIONS,
        "hybrid_stationarity_tolerance": optional.HYBRID_JOINT_STATIONARITY_TOLERANCE,
        "hybrid_relative_objective_tolerance": optional.HYBRID_RELATIVE_OBJECTIVE_TOLERANCE,
        "hybrid_maximum_sweeps": optional.HYBRID_MAXIMUM_SWEEPS,
    }


def _validate_case(gate: Any, family: str, rank: int) -> None:
    """Reject incomplete or changed singleton evidence before it can be reused."""
    _require(isinstance(gate, dict), "recovery case is not an object")
    _require(
        gate.get("status") == "pass" and gate.get("ranks") == [rank]
        and type(gate["ranks"][0]) is int
        and isinstance(gate.get("cases"), dict)
        and set(gate["cases"]) == {str(rank)},
        "recovery case has mismatched rank or nonpassing status",
    )
    _require(
        gate.get("mean_conditional_KL_maximum_bits") == RECOVERY_KL_MAXIMUM_BITS,
        "recovery case has changed KL threshold",
    )
    case = gate["cases"][str(rank)]
    _require(isinstance(case, dict), "recovery case detail is not an object")
    _require(
        case.get("status") == "pass"
        and case.get("deterministic_repeat_bit_exact") is True,
        "recovery case lacks passing exact repeats",
    )

    def number(name: str, maximum: float | None = None, *, positive=False) -> float:
        value = case.get(name)
        _require(
            type(value) in (int, float) and math.isfinite(value)
            and (value > 0.0 if positive else value >= 0.0)
            and (maximum is None or value <= maximum),
            f"recovery case has invalid {name}",
        )
        return float(value)

    number("mean_conditional_KL_bits", RECOVERY_KL_MAXIMUM_BITS)
    number("maximum_candidate_probability_absolute_error", 1.0)
    number("mean_inclusion_marginal_absolute_error", 1.0)
    number("population_correction_planted_gradient_maximum_absolute_error",
           PLANTED_GRADIENT_TOLERANCE)
    number("lambda_reference", positive=True)
    if family == RECOVERY_FAMILIES[0]:
        _require(
            gate.get("every_authorized_rank_qualified") is True
            and case.get("rho") == RECOVERY_RHO
            and "named_sparse_edges" not in gate
            and "rho_sparse" not in case,
            "recovery case has wrong low-rank family or parameters",
        )
        number("fit_gradient_infinity", optional.LOW_RANK_GRADIENT_INFINITY_TOLERANCE)
        number("fit_relative_objective_change", optional.LOW_RANK_RELATIVE_OBJECTIVE_TOLERANCE)
        count, limit = "fit_iterations", optional.LOW_RANK_MAXIMUM_ITERATIONS
    else:
        _require(
            family == RECOVERY_FAMILIES[1]
            and gate.get("every_possible_activated_rank_qualified") is True
            and gate.get("named_sparse_edges") == HYBRID_NAMED_EDGE_COUNT
            and case.get("named_sparse_edges") == HYBRID_NAMED_EDGE_COUNT
            and case.get("rank") == rank
            and case.get("rho_low") == case.get("rho_sparse") == RECOVERY_RHO
            and case.get("selected_anchor_rho") == 0.1
            and "rho" not in case,
            "recovery case has wrong hybrid family or parameters",
        )
        number("core_lasso_lambda_maximum", positive=True)
        number("fit_joint_stationarity_infinity", optional.HYBRID_JOINT_STATIONARITY_TOLERANCE)
        number("fit_relative_objective_change", optional.HYBRID_RELATIVE_OBJECTIVE_TOLERANCE)
        count, limit = "fit_sweeps", optional.HYBRID_MAXIMUM_SWEEPS
    _require(type(case.get(count)) is int and 0 <= case[count] <= limit,
             f"recovery case has invalid {count}")


def _load_completed_cases(directory: Path | None) -> dict[tuple[str, int], Any]:
    if directory is None:
        return {}
    directory = directory.resolve()
    _require(directory.is_dir(), "resume completed-case directory is missing")
    expected = {
        f"{family}_rank_{rank}.json": (family, rank)
        for family in RECOVERY_FAMILIES for rank in LOW_RANK_RANKS
    }
    unexpected = [path.name for path in directory.iterdir()
                  if path.suffix == ".json" and path.name not in expected]
    _require(not unexpected, f"unexpected resume case filenames: {unexpected}")
    loaded = {}
    for name, (family, rank) in expected.items():
        path = directory / name
        if not path.exists():
            continue
        try:
            gate = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise OptionalQualificationError(f"malformed resume case {name}: {error}") from error
        _validate_case(gate, family, rank)
        context = gate.get("qualification_context")
        legacy_source = LEGACY_COMPLETED_CASES.resolve() / name
        legacy = False
        if context is None and family == RECOVERY_FAMILIES[0] and rank in (1, 2, 3):
            try:
                original = json.loads(legacy_source.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise OptionalQualificationError(
                    f"cannot verify original legacy case {name}: {error}"
                ) from error
            legacy = gate == original
        _require(
            legacy or context == _case_context(family, rank),
            f"resume case {name} has missing or changed frozen fixture/family context",
        )
        # Also reject inconsistent explicit identities on legacy records.
        for key, value in (("seed", core_qualification.FROZEN_SEED),
                           ("family", family), ("rank", rank)):
            _require(key not in gate or gate[key] == value,
                     f"resume case {name} has changed {key}")
        loaded[(family, rank)] = {
            "gate": gate,
            "provenance": {
                "artifact": str(path),
                "original_artifact": str(legacy_source if legacy else path),
                "legacy_context_from_recorded_producer": legacy,
                "original_producer_job_id": (
                    "46115632" if legacy else gate.get("producer_job_id")
                ),
            },
        }
    _require(bool(loaded), "resume directory contains no completed recovery cases")
    return loaded


def _parallel_recovery_gates(
    workers: int, progress_dir: Path | None,
    resumed_cases: Mapping[tuple[str, int], Any] | None = None,
) -> Mapping[str, Any]:
    """Parallelize independent cases without sharing solver identities."""
    families = RECOVERY_FAMILIES
    completed: dict[str, dict[str, Any]] = {family: {} for family in families}
    for (family, rank), saved in (resumed_cases or {}).items():
        gate = saved["gate"]
        _require(family in families and rank in LOW_RANK_RANKS,
                 "unexpected resumed family/rank")
        _validate_case(gate, family, rank)
        completed[family][str(rank)] = gate["cases"][str(rank)]
        if progress_dir is not None:
            _atomic_json(progress_dir / f"{family}_rank_{rank}.json", gate)
        print(f"RECOVERY_CASE_REUSED family={family} rank={rank}", flush=True)
    failures: list[str] = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(_run_recovery_case, family, rank): (family, rank)
            for family in families for rank in LOW_RANK_RANKS
            if str(rank) not in completed[family]
        }
        for future in as_completed(futures):
            family, rank = futures[future]
            try:
                returned_family, returned_rank, gate = future.result()
                _require(
                    (returned_family, returned_rank) == (family, rank),
                    "recovery worker returned a mismatched case",
                )
                _validate_case(gate, family, rank)
                if progress_dir is not None:
                    _atomic_json(progress_dir / f"{family}_rank_{rank}.json", {
                        **gate, "qualification_context": _case_context(family, rank),
                        "producer_job_id": os.environ.get("SLURM_JOB_ID"),
                    })
                completed[family][str(rank)] = gate["cases"][str(rank)]
                print(f"RECOVERY_CASE_PASS family={family} rank={rank}", flush=True)
            except Exception as error:
                failures.append(f"{family} rank={rank}: {type(error).__name__}: {error}")
                print(f"RECOVERY_CASE_FAIL {failures[-1]}", flush=True)
    _require(not failures, "; ".join(failures))
    _require(
        all(set(cases) == set(map(str, LOW_RANK_RANKS)) for cases in completed.values()),
        "parallel recovery did not complete every frozen family/rank case",
    )
    ordered = {
        family: {str(rank): completed[family][str(rank)] for rank in LOW_RANK_RANKS}
        for family in families
    }
    return {
        families[0]: {
            "status": "pass", "ranks": list(LOW_RANK_RANKS),
            "every_authorized_rank_qualified": True,
            "mean_conditional_KL_maximum_bits": RECOVERY_KL_MAXIMUM_BITS,
            "cases": ordered[families[0]],
        },
        families[1]: {
            "status": "pass", "ranks": list(LOW_RANK_RANKS),
            "every_possible_activated_rank_qualified": True,
            "named_sparse_edges": HYBRID_NAMED_EDGE_COUNT,
            "mean_conditional_KL_maximum_bits": RECOVERY_KL_MAXIMUM_BITS,
            "cases": ordered[families[1]],
        },
    }


def run_qualification(
    *, workers: int = 1, progress_dir: Path | None = None,
    resume_completed_cases: Path | None = None,
) -> Mapping[str, Any]:
    _validate_runtime(workers)
    started = time.monotonic()
    if resume_completed_cases is not None and progress_dir is not None:
        _require(resume_completed_cases.resolve() != progress_dir.resolve(),
                 "resume source and new progress directory must differ")
    resumed_cases = _load_completed_cases(resume_completed_cases)
    fixture = core_qualification._frozen_fixture()
    transform, prepared = m2.fit_residual_transform(
        fixture.cells,
        fixture.moments,
        fixture.candidates,
        fixture.geometry,
        fixture.pairs,
    )
    _require(
        bool(transform.diagnostics["rank_stable"]),
        "optional recovery residual rank is unstable",
    )
    gates = {"optional_trigger_diagnostic": qualify_optional_trigger_diagnostic()}
    if workers == 1 and not resumed_cases:
        gates.update({
            "low_rank_population_recovery": qualify_low_rank_population_recovery(
                fixture, prepared
            ),
            "sparse_plus_low_rank_population_recovery": (
                qualify_hybrid_population_recovery(fixture, prepared)
            ),
        })
    else:
        print(f"RECOVERY_START workers={workers} new_cases={12 - len(resumed_cases)} "
              f"reused_cases={len(resumed_cases)}", flush=True)
        gates.update(_parallel_recovery_gates(workers, progress_dir, resumed_cases))
    _require(
        all(gate.get("status") == "pass" for gate in gates.values()),
        "optional qualification contains a nonpassing gate",
    )
    return {
        "status": "pass",
        "overall_gate_status": "pass",
        "stage": "v3_M2_target_free_optional_family_qualification_v1",
        "analysis_label": "exploratory_development_only",
        "elapsed_wall_seconds": time.monotonic() - started,
        "execution": {
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            "elapsed_wall_seconds_scope": "current_job_qualification_only_excludes_reused_case_runtime",
            "new_recovery_case_count": 12 - len(resumed_cases),
            "reused_recovery_case_count": len(resumed_cases),
            "reused_recovery_cases": [
                {"family": family, "rank": rank, **saved["provenance"]}
                for (family, rank), saved in resumed_cases.items()
            ],
        },
        "frozen_fixture": {
            "seed": core_qualification.FROZEN_SEED,
            "groups": core_qualification.FROZEN_GROUPS,
            "rows_per_group": core_qualification.FROZEN_ROWS_PER_GROUP,
            "cells": len(fixture.cells),
            "every_K_0_through_16_once_per_group": True,
            "residual_rank": transform.rank,
            "rank_stable": True,
        },
        "gate_inventory": {
            "required": list(gates),
            "passing": list(gates),
            "required_gate_count": len(gates),
            "all_required_gates_present_and_passed": True,
        },
        "gates": gates,
        "real_target_membership_or_M2_effects_opened": False,
        "actual_K_opened": False,
        "M0_full_refit_or_telemetry_opened": False,
        "final_or_unassigned_or_SWC_opened": False,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume-completed-cases", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_qualification(
            workers=args.workers, progress_dir=args.output.parent / "completed_cases",
            resume_completed_cases=args.resume_completed_cases,
        )
        _atomic_json(args.output, report)
    except Exception as error:
        print(
            f"ERROR: optional-family qualification failed ({type(error).__name__}): {error}",
            file=os.sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "gate_count": report["gate_inventory"]["required_gate_count"],
                "real_M2_opened": report[
                    "real_target_membership_or_M2_effects_opened"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
