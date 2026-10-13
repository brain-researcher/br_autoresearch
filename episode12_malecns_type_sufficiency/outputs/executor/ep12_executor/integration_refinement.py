"""Generated-data-only refinement pilot for EP12 predictive integration.

This is deliberately separate from the frozen scientific qualification path.
It asks whether 512 predictive-integration draws agree with 1,024 draws under
the already-frozen 0.002 tolerance and exact selected-model metadata rule.
No MaleCNS connectivity reader is imported or reachable from this module.
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

from .runtime import EPISODE_ROOT, atomic_json
from .scientific_models import _normal_qmc, reciprocal_transfer, seed_from
from .scientific_synthetic import SCENARIO_TRUTH, generate_synthetic_type


SCHEMA_VERSION = "ep12.integration_refinement.v1"
DRAW_GRID = (256, 512, 1024)
CANDIDATE_DRAWS = 512
HIGH_DRAWS = 1024
ABSOLUTE_DELTA_TOLERANCE = 0.002
SCENARIOS = (
    "adequate_template",
    "curved_continuum_dense",
    "residual_groups_within_continuum",
)
FIXTURE_SEEDS = (1201, 1202)
NEURONS_PER_SIDE = 19
COUNT_TOTAL = 100
GENERATOR_STRENGTH = 1.8
MISSING_FRACTION = 0.1
FOLDS = 2
LATENT_RANK = 2
RIDGE = 0.1
REPORT_NAME = "integration_refinement.json"
DEFAULT_OUTPUT_DIR = (
    EPISODE_ROOT
    / "outputs"
    / "scientific_qualification"
    / "pilots"
    / "integration_refinement_512_1024"
)


def _json_copy(value: Any) -> Any:
    """Return a JSON-safe detached value or fail before publication."""

    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _extract_transfer_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and retain only the pilot's predeclared comparison fields."""

    deltas: dict[str, float] = {}
    for direction in ("left_to_right", "right_to_left"):
        delta = float(result[direction]["mean_delta_M_minus_reference"])
        if not math.isfinite(delta):
            raise ValueError(f"non-finite {direction} transfer delta")
        deltas[f"{direction}_delta"] = delta
    metadata = _json_copy(result["source_fit_metadata"])
    if not isinstance(metadata, dict) or set(metadata) != {"left", "right"}:
        raise ValueError("source_fit_metadata must contain exact left/right records")
    expected_fields = {"U", "M", "reference", "null_generator"}
    for side in ("left", "right"):
        if not isinstance(metadata[side], dict) or set(metadata[side]) != expected_fields:
            raise ValueError(
                f"source_fit_metadata.{side} must contain the exact selected-model fields"
            )
        if not all(isinstance(value, str) and value for value in metadata[side].values()):
            raise ValueError(
                f"source_fit_metadata.{side} selected-model values must be strings"
            )
    return {**deltas, "source_fit_metadata": metadata}


def _validated_draw_configuration(
    draw_grid: Sequence[int], candidate_draws: int, high_draws: int
) -> tuple[tuple[int, ...], int, int]:
    """Validate a bounded refinement comparison without changing its criteria."""

    if any(isinstance(value, bool) for value in (*draw_grid, candidate_draws, high_draws)):
        raise ValueError("integration draw counts must be positive integers")
    grid = tuple(int(value) for value in draw_grid)
    candidate = int(candidate_draws)
    high = int(high_draws)
    if len(grid) < 2 or any(value < 1 for value in grid):
        raise ValueError("integration draw grid needs at least two positive counts")
    if tuple(sorted(set(grid))) != grid:
        raise ValueError("integration draw grid must be strictly increasing")
    if candidate not in grid or high not in grid or candidate >= high:
        raise ValueError(
            "candidate and high draw counts must be increasing members of the grid"
        )
    return grid, candidate, high


def validate_sobol_prefixes(
    draw_grid: Sequence[int] = DRAW_GRID,
) -> dict[str, Any]:
    """Verify that the actual normal-QMC generator nests the pilot draw grid."""

    grid = tuple(int(value) for value in draw_grid)
    if len(grid) < 2 or tuple(sorted(set(grid))) != grid or any(
        value < 1 for value in grid
    ):
        raise ValueError("Sobol prefix grid must be strictly increasing and positive")
    checks: list[dict[str, Any]] = []
    all_match = True
    for dimension in (1, 2, 10):
        probe_seed = seed_from("integration-refinement-prefix", dimension)
        arrays = {
            draws: np.asarray(_normal_qmc(draws, dimension, probe_seed))
            for draws in grid
        }
        for low, high in zip(grid, grid[1:]):
            match = bool(np.array_equal(arrays[low], arrays[high][:low]))
            all_match = all_match and match
            checks.append(
                {
                    "dimension": dimension,
                    "seed": probe_seed,
                    "low_draws": low,
                    "high_draws": high,
                    "exact_prefix_match": match,
                }
            )
    return {
        "generator": "nested_scrambled_sobol_standard_normal",
        "grid": list(grid),
        "checks": checks,
        "passed": all_match,
    }


def _evaluate_grid(
    draw_grid: Sequence[int] = DRAW_GRID,
    candidate_draws: int = CANDIDATE_DRAWS,
    high_draws: int = HIGH_DRAWS,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run the fixed generated fixtures once across all three draw counts."""

    grid, candidate_count, high_count = _validated_draw_configuration(
        draw_grid, candidate_draws, high_draws
    )
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        if scenario not in SCENARIO_TRUTH:
            raise ValueError(f"unknown generated scenario: {scenario}")
        for fixture_seed in FIXTURE_SEEDS:
            fixture = generate_synthetic_type(
                scenario,
                left_neurons=NEURONS_PER_SIDE,
                right_neurons=NEURONS_PER_SIDE,
                count_total=COUNT_TOTAL,
                strength=GENERATOR_STRENGTH,
                missing_fraction=MISSING_FRACTION,
                seed=fixture_seed,
            )
            fit_seed = seed_from("integration-stability", scenario, fixture_seed)
            by_draws: dict[int, dict[str, Any]] = {}
            for draws in grid:
                result = reciprocal_transfer(
                    fixture.left.counts,
                    fixture.left.covariates,
                    fixture.right.counts,
                    fixture.right.covariates,
                    seed=fit_seed,
                    integration_draws=draws,
                    folds=FOLDS,
                    latent_rank=LATENT_RANK,
                    ridge=RIDGE,
                )
                by_draws[draws] = _extract_transfer_result(result)

            candidate = by_draws[candidate_count]
            high = by_draws[high_count]
            changes = {
                direction: abs(
                    candidate[f"{direction}_delta"]
                    - high[f"{direction}_delta"]
                )
                for direction in ("left_to_right", "right_to_left")
            }
            metadata_match = (
                candidate["source_fit_metadata"] == high["source_fit_metadata"]
            )
            rows.append(
                {
                    "scenario": scenario,
                    "fixture_seed": fixture_seed,
                    "fit_seed": fit_seed,
                    "by_integration_draws": {
                        str(draws): by_draws[draws] for draws in grid
                    },
                    "candidate_to_high_absolute_delta_change": changes,
                    "selected_model_metadata_exact_match": metadata_match,
                    "comparison_passed": (
                        max(changes.values()) <= ABSOLUTE_DELTA_TOLERANCE
                        and metadata_match
                    ),
                }
            )
    timing = {
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
    }
    return rows, timing


def _canonical_rows(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def build_refinement_report(
    *,
    draw_grid: Sequence[int] = DRAW_GRID,
    candidate_draws: int = CANDIDATE_DRAWS,
    high_draws: int = HIGH_DRAWS,
) -> dict[str, Any]:
    """Build the complete pilot report without reading or writing real data."""

    grid, candidate_count, high_count = _validated_draw_configuration(
        draw_grid, candidate_draws, high_draws
    )
    total_wall = time.perf_counter()
    total_cpu = time.process_time()
    prefix_validation = validate_sobol_prefixes(grid)
    primary_rows, primary_timing = _evaluate_grid(
        grid, candidate_count, high_count
    )
    replay_rows, replay_timing = _evaluate_grid(
        grid, candidate_count, high_count
    )
    primary_bytes = _canonical_rows(primary_rows)
    replay_bytes = _canonical_rows(replay_rows)
    replay_exact = primary_bytes == replay_bytes

    changes = [
        float(change)
        for row in primary_rows
        for change in row["candidate_to_high_absolute_delta_change"].values()
    ]
    maximum_change = max(changes, default=float("inf"))
    metadata_match = all(
        bool(row["selected_model_metadata_exact_match"])
        for row in primary_rows
    )
    numerical_criterion_passed = (
        maximum_change <= ABSOLUTE_DELTA_TOLERANCE and metadata_match
    )
    passed = (
        numerical_criterion_passed
        and bool(prefix_validation["passed"])
        and replay_exact
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "pilot_scope": "generated_data_only",
        "scientific_qualification_path_modified": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "integration_draws_grid": list(grid),
        "candidate_draws": candidate_count,
        "high_draws": high_count,
        "absolute_delta_change_tolerance": ABSOLUTE_DELTA_TOLERANCE,
        "selected_model_criterion": "exact_source_fit_metadata_equality",
        "fixture_configuration": {
            "scenarios": list(SCENARIOS),
            "seeds": list(FIXTURE_SEEDS),
            "neurons_per_side": NEURONS_PER_SIDE,
            "count_total": COUNT_TOTAL,
            "generator_strength": GENERATOR_STRENGTH,
            "missing_fraction": MISSING_FRACTION,
            "folds": FOLDS,
            "latent_rank": LATENT_RANK,
            "ridge": RIDGE,
        },
        "sobol_prefix_validation": prefix_validation,
        "deterministic_replay": {
            "performed": True,
            "exact_row_replay": replay_exact,
            "primary_sha256": hashlib.sha256(primary_bytes).hexdigest(),
            "replay_sha256": hashlib.sha256(replay_bytes).hexdigest(),
            "passed": replay_exact,
        },
        "rows": primary_rows,
        "maximum_observed_absolute_delta_change": maximum_change,
        "all_selected_model_metadata_exactly_match": metadata_match,
        "numerical_criterion_passed": numerical_criterion_passed,
        "passed": passed,
        "runtime": {
            "primary": primary_timing,
            "deterministic_replay": replay_timing,
            "total_wall_seconds": time.perf_counter() - total_wall,
            "total_cpu_seconds": time.process_time() - total_cpu,
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
            "integration refinement output must be a run directory below "
            "outputs/scientific_qualification/pilots"
        )
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite pilot output: {resolved}")
    return resolved


def run_integration_refinement(
    output_dir: Path,
    *,
    episode_root: Path = EPISODE_ROOT,
    draw_grid: Sequence[int] = DRAW_GRID,
    candidate_draws: int = CANDIDATE_DRAWS,
    high_draws: int = HIGH_DRAWS,
) -> dict[str, Any]:
    """Run, then atomically publish once into a new durable pilot directory."""

    output_dir = _validated_output_dir(output_dir, episode_root)
    report = build_refinement_report(
        draw_grid=draw_grid,
        candidate_draws=candidate_draws,
        high_draws=high_draws,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    atomic_json(output_dir / REPORT_NAME, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the generated-only EP12 integration refinement pilot."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="new run directory below outputs/scientific_qualification/pilots",
    )
    parser.add_argument(
        "--draw-grid",
        type=int,
        nargs="+",
        default=list(DRAW_GRID),
        help="strictly increasing integration draw counts",
    )
    parser.add_argument(
        "--candidate-draws",
        type=int,
        default=CANDIDATE_DRAWS,
        help="lower draw count in the frozen numerical comparison",
    )
    parser.add_argument(
        "--high-draws",
        type=int,
        default=HIGH_DRAWS,
        help="higher draw count in the frozen numerical comparison",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_integration_refinement(
        args.output_dir,
        draw_grid=args.draw_grid,
        candidate_draws=args.candidate_draws,
        high_draws=args.high_draws,
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir.resolve() / REPORT_NAME),
                "passed": report["passed"],
                "maximum_observed_absolute_delta_change": report[
                    "maximum_observed_absolute_delta_change"
                ],
                "all_selected_model_metadata_exactly_match": report[
                    "all_selected_model_metadata_exactly_match"
                ],
                "deterministic_replay_passed": report[
                    "deterministic_replay"
                ]["passed"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
