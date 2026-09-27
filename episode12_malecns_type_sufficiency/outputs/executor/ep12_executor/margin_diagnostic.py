"""Outcome-blind diagnostic for compatibility with EP12's fixed score margin.

This is deliberately not an acceptance test and cannot unlock real connectivity.
It gives M an optimistic advantage by fitting the generator's true component
count to a large generated source sample, then evaluates the unchanged target-
marginal predictive score against the two frozen U references.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .policy import EpisodePolicy
from .scientific_models import (
    cross_validated_candidate,
    fit_predictive_model,
    seed_from,
)
from .scientific_qualification import _contract_settings
from .scientific_synthetic import generate_synthetic_type
from .runtime import EPISODE_ROOT, atomic_json
from .yaml_subset import load_yaml_subset


def _score_direction(
    source_counts: np.ndarray,
    source_covariates: np.ndarray,
    target_counts: np.ndarray,
    target_covariates: np.ndarray,
    *,
    true_components: int,
    seed: int,
    integration_draws: int,
    folds: int,
) -> dict[str, Any]:
    u_selection = cross_validated_candidate(
        source_counts,
        source_covariates,
        ["U_linear", "U_curved"],
        seed=seed_from(seed, "U-cv"),
        folds=folds,
        integration_draws=integration_draws,
        latent_rank=2,
        ridge=0.1,
    )
    u_model = fit_predictive_model(
        source_counts,
        source_covariates,
        u_selection.model_id,
        seed=seed_from(seed, "U-final", u_selection.model_id),
        integration_draws=integration_draws,
        latent_rank=2,
        ridge=0.1,
    )
    m_model_id = f"M{true_components}"
    m_model = fit_predictive_model(
        source_counts,
        source_covariates,
        m_model_id,
        seed=seed_from(seed, "M-truth-K", m_model_id),
        integration_draws=integration_draws,
        latent_rank=2,
        ridge=0.1,
    )
    u_score = u_model.mean_score(target_counts, target_covariates)
    m_score = m_model.mean_score(target_counts, target_covariates)
    return {
        "selected_U": u_model.model_id,
        "optimistic_M": m_model.model_id,
        "U_score_nats_per_observed_count": u_score,
        "M_score_nats_per_observed_count": m_score,
        "delta_M_minus_U_nats_per_observed_count": m_score - u_score,
    }


def run_margin_compatibility_diagnostic(*, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    allowed_root = (EPISODE_ROOT / "outputs" / "prelaunch").resolve()
    if output_dir != allowed_root and allowed_root not in output_dir.parents:
        raise ValueError("margin diagnostic outputs must stay under outputs/prelaunch")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"margin diagnostic output directory is not empty: {output_dir}")

    contract_path = EPISODE_ROOT / "outputs" / "executor" / "SCIENTIFIC_CONTRACT.yaml"
    spec_path = EPISODE_ROOT / "outputs" / "executor" / "SYNTHETIC_QUALIFICATION_SPEC.yaml"
    contract = load_yaml_subset(contract_path)
    settings = _contract_settings(contract)
    if contract["real_connectivity_access_authorized"] not in (False, "development_only"):
        raise ValueError("margin diagnostic permits at most development-only authorization")
    if contract.get("final_connectivity_access_authorized", False) is not False:
        raise ValueError("margin diagnostic requires final connectivity to remain disabled")
    expected_threads = str(settings["numeric_threads_per_worker"])
    thread_variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    if any(os.environ.get(name) != expected_threads for name in thread_variables):
        raise ValueError("numeric thread environment differs from the scientific contract")

    spec = load_yaml_subset(spec_path)
    policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")
    margin = float(policy.meaningful_margin)
    strengths = [
        float(value)
        for value in spec["resolution_grid"]["group_logit_strength"]
        if float(value) > 0.0
    ]
    count_totals = [10, 100, 1000]
    source_neurons = 300
    target_neurons = 500
    seeds = [3101, 3102, 3103]
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    rows: list[dict[str, Any]] = []
    for count_total in count_totals:
        for strength in strengths:
            for fixture_seed in seeds:
                fixture = generate_synthetic_type(
                    "residual_groups_within_continuum",
                    left_neurons=source_neurons,
                    right_neurons=target_neurons,
                    count_total=count_total,
                    strength=strength,
                    missing_fraction=0.0,
                    seed=fixture_seed,
                )
                true_components = int(fixture.generator_parameters["group_components"])
                left_to_right = _score_direction(
                    fixture.left.counts,
                    fixture.left.covariates,
                    fixture.right.counts,
                    fixture.right.covariates,
                    true_components=true_components,
                    seed=seed_from("margin-diagnostic", count_total, strength, fixture_seed, "L2R"),
                    integration_draws=int(settings["integration_draws"]),
                    folds=int(settings["source_cv_folds"]),
                )
                right_to_left = _score_direction(
                    fixture.right.counts,
                    fixture.right.covariates,
                    fixture.left.counts,
                    fixture.left.covariates,
                    true_components=true_components,
                    seed=seed_from("margin-diagnostic", count_total, strength, fixture_seed, "R2L"),
                    integration_draws=int(settings["integration_draws"]),
                    folds=int(settings["source_cv_folds"]),
                )
                direction_deltas = [
                    left_to_right["delta_M_minus_U_nats_per_observed_count"],
                    right_to_left["delta_M_minus_U_nats_per_observed_count"],
                ]
                rows.append(
                    {
                        "count_total": count_total,
                        "group_logit_strength": strength,
                        "fixture_seed": fixture_seed,
                        "true_components": true_components,
                        "left_to_right": left_to_right,
                        "right_to_left": right_to_left,
                        "mean_direction_delta": float(np.mean(direction_deltas)),
                        "minimum_direction_delta": float(np.min(direction_deltas)),
                        "both_directions_exceed_margin": bool(
                            min(direction_deltas) > margin
                        ),
                    }
                )

    grouped: list[dict[str, Any]] = []
    for count_total in count_totals:
        for strength in strengths:
            selected = [
                row
                for row in rows
                if row["count_total"] == count_total
                and row["group_logit_strength"] == strength
            ]
            direction_deltas = [
                row[direction]["delta_M_minus_U_nats_per_observed_count"]
                for row in selected
                for direction in ("left_to_right", "right_to_left")
            ]
            grouped.append(
                {
                    "count_total": count_total,
                    "group_logit_strength": strength,
                    "direction_evaluations": len(direction_deltas),
                    "mean_delta": float(np.mean(direction_deltas)),
                    "minimum_delta": float(np.min(direction_deltas)),
                    "maximum_delta": float(np.max(direction_deltas)),
                    "directions_exceeding_margin": int(
                        sum(value > margin for value in direction_deltas)
                    ),
                    "fixed_margin": margin,
                }
            )

    maximum_observed_delta = max(
        row[direction]["delta_M_minus_U_nats_per_observed_count"]
        for row in rows
        for direction in ("left_to_right", "right_to_left")
    )
    report = {
        "diagnostic_scope": (
            "generated_residual_groups_only_not_an_acceptance_test; M_is_given_the_"
            "generator_true_component_count_and_large_source_sample"
        ),
        "source_neurons": source_neurons,
        "target_neurons": target_neurons,
        "fixture_seeds": seeds,
        "count_totals": count_totals,
        "group_logit_strengths": strengths,
        "fixed_margin_nats_per_observed_count": margin,
        "maximum_observed_direction_delta": maximum_observed_delta,
        "any_direction_exceeded_margin": bool(maximum_observed_delta > margin),
        "any_fixture_exceeded_margin_both_directions": any(
            row["both_directions_exceed_margin"] for row in rows
        ),
        "grouped_results": grouped,
        "rows": rows,
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "real_connectivity_accessed": False,
    }
    atomic_json(output_dir / "margin_compatibility.json", report)
    with (output_dir / "margin_compatibility.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grouped[0]))
        writer.writeheader()
        writer.writerows(grouped)
    lines = [
        "# EP12 fixed-margin compatibility diagnostic",
        "",
        "Real connectivity accessed: `false`  ",
        "Acceptance-test status: `diagnostic only`  ",
        f"Fixed policy margin: `{margin:.6f}` nats per observed outgoing count",
        "",
        "M was given the generator's true component count and a 300-neuron generated source side; target groups remained marginalized exactly as required by the frozen transfer contract.",
        "",
        "| Count total | Group strength | Direction evals | Mean M-U | Min M-U | Max M-U | Above margin |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in grouped:
        lines.append(
            "| {count_total} | {group_logit_strength:.1f} | {direction_evaluations} | {mean_delta:.6f} | {minimum_delta:.6f} | {maximum_delta:.6f} | {directions_exceeding_margin} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "This diagnostic does not change the frozen margin, qualify false-positive error, or authorize real-data access.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return report
