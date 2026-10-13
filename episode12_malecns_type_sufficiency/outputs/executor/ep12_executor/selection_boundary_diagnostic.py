"""Generated-only diagnostic for the EP12 M2/M3 selection boundary.

The common-random 512-versus-1,024 integration pilot isolated one discrete
model-selection change in the curved-continuum fixture with seed 1,202.  This
module records the numerical quantities around that already-observed boundary.
It does not alter the candidate family, cross-validation rule, tie tolerance,
scientific margin, or any qualification/readiness gate, and it has no path to
MaleCNS connectivity inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .runtime import EPISODE_ROOT, publish_once
from .scientific_models import (
    CandidateSelection,
    ScientificModelError,
    fit_predictive_model,
    fit_triplet,
    seed_from,
    validate_profile_arrays,
)
from .scientific_synthetic import SCENARIO_TRUTH, generate_synthetic_type


SCHEMA_VERSION = "ep12.selection_boundary_diagnostic.v1"
SCENARIO = "curved_continuum_dense"
FIXTURE_SEED = 1202
DRAW_GRID = (512, 1024, 2048)
MODEL_CANDIDATES = ("M2", "M3")
NEURONS_PER_SIDE = 19
COUNT_TOTAL = 100
GENERATOR_STRENGTH = 1.8
MISSING_FRACTION = 0.1
FOLDS = 2
LATENT_RANK = 2
RIDGE = 0.1
TIE_TOLERANCE = 1e-6
REPORT_NAME = "selection_boundary_diagnostic.json"
DEFAULT_OUTPUT_DIR = (
    EPISODE_ROOT
    / "outputs"
    / "scientific_qualification"
    / "pilots"
    / "selection_boundary_curved1202_512_1024_2048"
)
SOURCE_SIDES = ("left", "right")


def _json_copy(value: Any) -> Any:
    """Return a detached JSON-safe value or fail on non-finite numbers."""

    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _validated_configuration(
    scenario: str,
    fixture_seed: int,
    source_side: str,
    draw_grid: Sequence[int],
) -> tuple[str, int, str, tuple[int, ...]]:
    """Validate the bounded generated-fixture diagnostic configuration."""

    if scenario not in SCENARIO_TRUTH:
        raise ValueError(f"unknown generated scenario: {scenario}")
    if isinstance(fixture_seed, bool) or not isinstance(fixture_seed, int):
        raise ValueError("fixture seed must be a nonnegative integer")
    if not 0 <= fixture_seed < 2**32:
        raise ValueError("fixture seed must lie in [0, 2**32)")
    if source_side not in SOURCE_SIDES:
        raise ValueError(f"source side must be one of {SOURCE_SIDES}")
    if any(isinstance(value, bool) for value in draw_grid):
        raise ValueError("integration draw counts must be positive integers")
    grid = tuple(int(value) for value in draw_grid)
    if (
        len(grid) < 2
        or any(value < 1 for value in grid)
        or tuple(sorted(set(grid))) != grid
    ):
        raise ValueError(
            "integration draw grid must contain at least two strictly increasing "
            "positive integers"
        )
    return scenario, fixture_seed, source_side, grid


def _candidate_selection_diagnostics(
    summaries: Sequence[CandidateSelection],
) -> dict[str, Any]:
    """Apply and expose the unchanged production candidate-selection rule."""

    if {item.model_id for item in summaries} != set(MODEL_CANDIDATES):
        raise ScientificModelError("selection diagnostic requires exactly M2 and M3")
    ranked = sorted(
        summaries,
        key=lambda item: (-item.mean_score, item.parameter_count, item.model_id),
    )
    best = ranked[0]
    selection_tolerance = max(TIE_TOLERANCE, best.standard_error)
    candidates: list[dict[str, Any]] = []
    near: list[CandidateSelection] = []
    for item in sorted(summaries, key=lambda value: value.model_id):
        gap = float(best.mean_score - item.mean_score)
        signed_distance = float(selection_tolerance - gap)
        in_near_set = gap <= selection_tolerance
        if in_near_set:
            near.append(item)
        candidates.append(
            {
                "model_id": item.model_id,
                "fold_scores": list(item.fold_scores),
                "mean_score": float(item.mean_score),
                "standard_error": float(item.standard_error),
                "parameter_count": int(item.parameter_count),
                "score_gap_from_best": gap,
                "near_set_boundary_signed_distance": signed_distance,
                "in_near_set": in_near_set,
            }
        )
    selected = min(near, key=lambda item: (item.parameter_count, item.model_id))
    return {
        "rule": (
            "rank by descending mean CV score; define the near set by "
            "best_score - candidate_score <= max(tie_tolerance, best_SE); "
            "select minimum parameter count then model id within that set"
        ),
        "tie_tolerance": TIE_TOLERANCE,
        "best_score_model_id": best.model_id,
        "best_standard_error": float(best.standard_error),
        "selection_tolerance": float(selection_tolerance),
        "signed_distance_definition": (
            "selection_tolerance - score_gap_from_best; nonnegative is inside "
            "the near set and negative is outside"
        ),
        "near_set_model_ids": sorted(item.model_id for item in near),
        "selected_model_id": selected.model_id,
        "candidates": candidates,
    }


def _evaluate_m_candidates(
    counts: np.ndarray,
    covariates: np.ndarray,
    *,
    source_seed: int,
    integration_draws: int,
) -> dict[str, Any]:
    """Record both M candidates under the production CV split and seed schedule."""

    counts, covariates = validate_profile_arrays(counts, covariates)
    folds = max(2, min(FOLDS, counts.shape[0]))
    m_cv_seed = seed_from(source_seed, "M-cv")
    permutation = np.random.default_rng(m_cv_seed).permutation(counts.shape[0])
    fold_ids = np.arange(counts.shape[0]) % folds
    fold_ids = fold_ids[np.argsort(permutation)]
    summaries: list[CandidateSelection] = []
    fold_integration_seeds = [
        seed_from(m_cv_seed, "cv-integration", fold) for fold in range(folds)
    ]

    for candidate in MODEL_CANDIDATES:
        fold_scores: list[float] = []
        parameter_count = 0
        for fold in range(folds):
            validation = fold_ids == fold
            training = ~validation
            if training.sum() < 2 or validation.sum() < 1:
                raise ScientificModelError(
                    "selection diagnostic encountered an invalid CV fold"
                )
            fitted = fit_predictive_model(
                counts[training],
                covariates[training],
                candidate,
                seed=seed_from(m_cv_seed, candidate, fold),
                integration_seed=fold_integration_seeds[fold],
                integration_draws=integration_draws,
                latent_rank=LATENT_RANK,
                ridge=RIDGE,
            )
            fold_scores.append(
                fitted.mean_score(counts[validation], covariates[validation])
            )
            parameter_count = max(parameter_count, fitted.parameter_count)
        values = np.asarray(fold_scores, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ScientificModelError("selection diagnostic CV score is non-finite")
        standard_error = (
            float(np.std(values, ddof=1) / math.sqrt(len(values)))
            if len(values) > 1
            else 0.0
        )
        summaries.append(
            CandidateSelection(
                model_id=candidate,
                mean_score=float(np.mean(values)),
                standard_error=standard_error,
                fold_scores=tuple(float(value) for value in values),
                parameter_count=int(parameter_count),
            )
        )

    result = _candidate_selection_diagnostics(summaries)
    result["cv_seed"] = m_cv_seed
    result["fold_integration_seeds"] = fold_integration_seeds
    result["integration_seed_shared_between_M2_and_M3_within_fold"] = True
    return result


def _evaluate_draw(
    fixture: Any,
    *,
    source_seed: int,
    integration_draws: int,
    source_side: str = "left",
) -> dict[str, Any]:
    """Evaluate one draw count without changing the fitted-model procedure."""

    if source_side not in SOURCE_SIDES:
        raise ValueError(f"source side must be one of {SOURCE_SIDES}")
    source = getattr(fixture, source_side)
    target_side = "right" if source_side == "left" else "left"
    target = getattr(fixture, target_side)
    triplet = fit_triplet(
        source.counts,
        source.covariates,
        seed=source_seed,
        integration_draws=integration_draws,
        folds=FOLDS,
        latent_rank=LATENT_RANK,
        ridge=RIDGE,
        tie_tolerance=TIE_TOLERANCE,
    )
    selection = _evaluate_m_candidates(
        source.counts,
        source.covariates,
        source_seed=source_seed,
        integration_draws=integration_draws,
    )
    production_selected = triplet.M_cv.model_id
    if selection["selected_model_id"] != production_selected:
        raise ScientificModelError(
            "diagnostic reconstruction disagrees with production M selection"
        )

    reference_model = (
        triplet.U if triplet.reference_model_id.startswith("U") else triplet.T
    )
    reference_scores = np.asarray(
        reference_model.score_neurons(
            target.counts, target.covariates
        ),
        dtype=float,
    )
    final_integration_seed = seed_from(source_seed, "final-integration")
    fixed_deltas: dict[str, float] = {}
    fixed_fit_seeds: dict[str, int] = {}
    for model_id in MODEL_CANDIDATES:
        fit_seed = seed_from(source_seed, "M-final", model_id)
        fixed_fit_seeds[model_id] = fit_seed
        fixed_model = fit_predictive_model(
            source.counts,
            source.covariates,
            model_id,
            seed=fit_seed,
            integration_seed=final_integration_seed,
            integration_draws=integration_draws,
            latent_rank=LATENT_RANK,
            ridge=RIDGE,
        )
        fixed_scores = np.asarray(
            fixed_model.score_neurons(
                target.counts, target.covariates
            ),
            dtype=float,
        )
        if fixed_scores.shape != reference_scores.shape:
            raise ScientificModelError(
                f"{model_id} and selected-reference target scores differ in shape"
            )
        delta = float(np.mean(fixed_scores - reference_scores))
        if not math.isfinite(delta):
            raise ScientificModelError(f"{model_id} target delta is non-finite")
        fixed_deltas[model_id] = delta

    selection["production_selected_model_id"] = production_selected
    selection["production_selection_exactly_reproduced"] = True
    direction = f"{source_side}_to_{target_side}"
    return {
        "integration_draws": int(integration_draws),
        "M_candidate_cross_validation": selection,
        "selected_reference_model_id": triplet.reference_model_id,
        "selected_U_model_id": triplet.U.model_id,
        f"fixed_M_{direction}_mean_delta_against_selected_reference": fixed_deltas,
        "seed_schedule": {
            "final_integration_seed": final_integration_seed,
            "fixed_M_fit_seeds": fixed_fit_seeds,
            "same_final_integration_seed_for_selected_reference_and_fixed_M": True,
        },
    }


def _evaluate_once(
    *,
    scenario: str = SCENARIO,
    fixture_seed: int = FIXTURE_SEED,
    source_side: str = "left",
    draw_grid: Sequence[int] = DRAW_GRID,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    scenario, fixture_seed, source_side, grid = _validated_configuration(
        scenario, fixture_seed, source_side, draw_grid
    )
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    fixture = generate_synthetic_type(
        scenario,
        left_neurons=NEURONS_PER_SIDE,
        right_neurons=NEURONS_PER_SIDE,
        count_total=COUNT_TOTAL,
        strength=GENERATOR_STRENGTH,
        missing_fraction=MISSING_FRACTION,
        seed=fixture_seed,
    )
    transfer_seed = seed_from("integration-stability", scenario, fixture_seed)
    source_seed = seed_from(transfer_seed, f"{source_side}-source")
    rows = [
        _evaluate_draw(
            fixture,
            source_seed=source_seed,
            integration_draws=draws,
            source_side=source_side,
        )
        for draws in grid
    ]
    return rows, {
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
    }


def _canonical_rows(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def build_selection_boundary_report(
    *,
    scenario: str = SCENARIO,
    fixture_seed: int = FIXTURE_SEED,
    source_side: str = "left",
    draw_grid: Sequence[int] = DRAW_GRID,
) -> dict[str, Any]:
    """Build the complete non-gating report and exact deterministic replay."""

    scenario, fixture_seed, source_side, grid = _validated_configuration(
        scenario, fixture_seed, source_side, draw_grid
    )
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    evaluation = {
        "scenario": scenario,
        "fixture_seed": fixture_seed,
        "source_side": source_side,
        "draw_grid": grid,
    }
    primary_rows, primary_timing = _evaluate_once(**evaluation)
    replay_rows, replay_timing = _evaluate_once(**evaluation)
    primary_bytes = _canonical_rows(primary_rows)
    replay_bytes = _canonical_rows(replay_rows)
    replay_exact = primary_bytes == replay_bytes
    reference_ids = {
        str(row["selected_reference_model_id"]) for row in primary_rows
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "generated_selection_boundary_diagnostic",
        "pilot_scope": "generated_data_only",
        "scientific_gate_applied": False,
        "acceptance_threshold_applied": False,
        "contract_or_tie_rule_modified": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "fixture_configuration": {
            "scenario": scenario,
            "fixture_seed": fixture_seed,
            "direction": (
                "left_to_right" if source_side == "left" else "right_to_left"
            ),
            "neurons_per_side": NEURONS_PER_SIDE,
            "count_total": COUNT_TOTAL,
            "generator_strength": GENERATOR_STRENGTH,
            "missing_fraction": MISSING_FRACTION,
            "folds": FOLDS,
            "latent_rank": LATENT_RANK,
            "ridge": RIDGE,
        },
        "integration_draws_grid": list(grid),
        "model_candidates": list(MODEL_CANDIDATES),
        "integration_seed_contract": {
            "generator": "nested_scrambled_sobol_standard_normal",
            "seeds_unchanged_across_draw_grid": True,
            "M2_M3_cv_integration_seed_shared_within_fold": True,
            "fixed_M_target_integration_seed_shared_with_selected_reference": True,
            "fit_and_GMM_seeds_remain_model_specific": True,
        },
        "rows": primary_rows,
        "selected_reference_identity_constant_across_draw_grid": len(reference_ids)
        == 1,
        "deterministic_replay": {
            "performed": True,
            "exact_row_replay": replay_exact,
            "primary_sha256": hashlib.sha256(primary_bytes).hexdigest(),
            "replay_sha256": hashlib.sha256(replay_bytes).hexdigest(),
        },
        "interpretation": (
            "diagnostic only: separates fixed-model numerical refinement from "
            "the existing discrete M2/M3 CV selection boundary"
        ),
        "runtime": {
            "primary": primary_timing,
            "deterministic_replay": replay_timing,
            "total_wall_seconds": time.perf_counter() - started_wall,
            "total_cpu_seconds": time.process_time() - started_cpu,
        },
    }


def _validated_output_dir(output_dir: Path, episode_root: Path) -> Path:
    episode_root = episode_root.resolve()
    pilots_root = (
        episode_root / "outputs" / "scientific_qualification" / "pilots"
    ).resolve()
    resolved = output_dir.resolve()
    if resolved == pilots_root or pilots_root not in resolved.parents:
        raise ValueError(
            "selection-boundary output must be a run directory below "
            "outputs/scientific_qualification/pilots"
        )
    if resolved.exists():
        raise FileExistsError(
            f"refusing to overwrite selection-boundary output: {resolved}"
        )
    return resolved


def run_selection_boundary_diagnostic(
    output_dir: Path,
    *,
    episode_root: Path = EPISODE_ROOT,
    scenario: str = SCENARIO,
    fixture_seed: int = FIXTURE_SEED,
    source_side: str = "left",
    draw_grid: Sequence[int] = DRAW_GRID,
) -> dict[str, Any]:
    """Run first, then atomically publish one durable generated-only report."""

    output_dir = _validated_output_dir(output_dir, episode_root)
    report = build_selection_boundary_report(
        scenario=scenario,
        fixture_seed=fixture_seed,
        source_side=source_side,
        draw_grid=draw_grid,
    )
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    if not publish_once(output_dir / REPORT_NAME, encoded):  # pragma: no cover
        raise FileExistsError("selection-boundary report unexpectedly pre-existed")
    return _json_copy(report)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the generated-only EP12 selection-boundary diagnostic."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="new run directory below outputs/scientific_qualification/pilots",
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIO_TRUTH),
        default=SCENARIO,
        help="declared generated-data scenario",
    )
    parser.add_argument(
        "--fixture-seed",
        type=int,
        default=FIXTURE_SEED,
        help="generated fixture seed",
    )
    parser.add_argument(
        "--source-side",
        choices=SOURCE_SIDES,
        default="left",
        help="source side fitted before scoring the opposite target side",
    )
    parser.add_argument(
        "--draw-grid",
        type=int,
        nargs="+",
        default=list(DRAW_GRID),
        help="strictly increasing integration draw counts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_selection_boundary_diagnostic(
        args.output_dir,
        scenario=args.scenario,
        fixture_seed=args.fixture_seed,
        source_side=args.source_side,
        draw_grid=args.draw_grid,
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir.resolve() / REPORT_NAME),
                "draw_grid": report["integration_draws_grid"],
                "deterministic_replay_exact": report["deterministic_replay"][
                    "exact_row_replay"
                ],
                "scientific_gate_applied": False,
            },
            sort_keys=True,
        )
    )
    # Exact replay is a computational-validity condition, not a scientific
    # acceptance threshold.  Publish the diagnostic either way, but fail the
    # job if it cannot reproduce its own rows exactly.
    return 0 if report["deterministic_replay"]["exact_row_replay"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
