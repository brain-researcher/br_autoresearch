#!/usr/bin/env python3
"""Outcome-blind generic small-sample calibration for the Dudman EP02 audit.

This program never reads empirical development or intervention outcomes.  It
calibrates candidate decision rules for a fixed mouse-level score under the
primary audit's 6-versus-5 design.  Every Monte Carlo replicate also contains
nine synthetic development controls; their sample mean and sample standard
deviation define the reference used for that replicate.  This exposes rather
than ignores reference-center and reference-scale uncertainty.

The between-intervention calculation enumerates all C(11, 6)=462 label
assignments.  It is exact only under the declared randomization/conditional
exchangeability and sharp-null contracts; it is not advertised as an exact
test of a heterogeneous weak mean null.  Bidirectional guards are simultaneous
Bonferroni-Welch one-sided lower bounds against the independent development
reference.  These remain model-based at n=5/6/9 and cannot replace an
endpoint-faithful binomial or beta-binomial calibration once the behavioral
endpoint and absolute raw-unit margin are frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy
from scipy.stats import t as student_t


SCHEMA_VERSION = "ep02.dudman.small_n_calibration.v2"
N_MINUS = 6
N_PLUS = 5
N_DEVELOPMENT = 9
N_TOTAL = N_MINUS + N_PLUS
EXPECTED_ASSIGNMENTS = math.comb(N_TOTAL, N_MINUS)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    family: str
    mean_minus: float
    mean_plus: float
    mean_development: float = 0.0
    sd_development: float = 1.0
    sd_minus: float = 1.0
    sd_plus: float = 1.0
    development_distribution: str = "gaussian"
    minus_distribution: str = "gaussian"
    plus_distribution: str = "gaussian"
    selection_role: str = "diagnostic"


@dataclass(frozen=True)
class Rule:
    rule_id: str
    zero_alpha: float
    practical_margin: float
    practical_alpha: float
    arm_familywise_alpha: float
    arm_lower_margin: float
    loo_delta_floor: float


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("null_gaussian", "null", 0.0, 0.0, selection_role="safety"),
    Scenario(
        "null_heavy_tail_t3",
        "null",
        0.0,
        0.0,
        development_distribution="student_t3_unit_variance",
        minus_distribution="student_t3_unit_variance",
        plus_distribution="student_t3_unit_variance",
        selection_role="safety",
    ),
    Scenario(
        "null_heteroskedastic",
        "null",
        0.0,
        0.0,
        sd_minus=3.0,
        sd_plus=0.5,
        selection_role="safety",
    ),
    Scenario(
        "null_heteroskedastic_reverse",
        "null",
        0.0,
        0.0,
        sd_minus=0.5,
        sd_plus=3.0,
        selection_role="safety",
    ),
    Scenario(
        "null_heavy_development_tail",
        "null",
        0.0,
        0.0,
        development_distribution="student_t3_unit_variance",
        selection_role="safety",
    ),
    Scenario("shared_positive", "same_sign", 0.75, 0.75, selection_role="safety"),
    Scenario("shared_negative", "same_sign", -0.75, -0.75, selection_role="safety"),
    Scenario(
        "reversed_rate_pattern",
        "opposite_direction",
        -0.75,
        0.75,
        selection_role="safety",
    ),
    Scenario(
        "practical_boundary_symmetric",
        "boundary",
        0.25,
        -0.25,
        selection_role="practical_boundary",
    ),
    Scenario("one_arm_minus_only", "one_arm", 1.0, 0.0, selection_role="one_arm"),
    Scenario("one_arm_plus_only", "one_arm", 0.0, -1.0, selection_role="one_arm"),
    Scenario(
        "one_arm_minus_extreme_inactive_sd2",
        "one_arm",
        3.0,
        0.0,
        sd_plus=2.0,
        selection_role="one_arm_extreme",
    ),
    Scenario(
        "one_arm_plus_extreme_inactive_sd2",
        "one_arm",
        0.0,
        -3.0,
        sd_minus=2.0,
        selection_role="one_arm_extreme",
    ),
    Scenario(
        "one_arm_minus_extreme_inactive_t3_sd2",
        "one_arm",
        3.0,
        0.0,
        sd_plus=2.0,
        plus_distribution="student_t3_unit_variance",
        selection_role="one_arm_extreme",
    ),
    Scenario(
        "one_arm_plus_extreme_inactive_t3_sd2",
        "one_arm",
        0.0,
        -3.0,
        sd_minus=2.0,
        minus_distribution="student_t3_unit_variance",
        selection_role="one_arm_extreme",
    ),
    Scenario("rate_weak", "rate_pattern", 0.50, -0.50),
    Scenario(
        "rate_moderate",
        "rate_pattern",
        0.75,
        -0.75,
        selection_role="moderate_target",
    ),
    Scenario(
        "rate_moderate_heteroskedastic",
        "rate_pattern",
        0.75,
        -0.75,
        sd_minus=1.50,
        sd_plus=0.75,
        selection_role="moderate_target",
    ),
    Scenario(
        "rate_moderate_heavy_tail_t3",
        "rate_pattern",
        0.75,
        -0.75,
        development_distribution="student_t3_unit_variance",
        minus_distribution="student_t3_unit_variance",
        plus_distribution="student_t3_unit_variance",
        selection_role="moderate_target",
    ),
    Scenario(
        "rate_strong",
        "rate_pattern",
        1.0,
        -1.0,
        selection_role="strong_target",
    ),
    Scenario(
        "rate_strong_heteroskedastic",
        "rate_pattern",
        1.0,
        -1.0,
        sd_minus=1.50,
        sd_plus=0.75,
        selection_role="strong_robustness",
    ),
    Scenario(
        "rate_strong_heteroskedastic_reverse",
        "rate_pattern",
        1.0,
        -1.0,
        sd_minus=0.75,
        sd_plus=1.50,
        selection_role="strong_robustness",
    ),
    Scenario(
        "rate_strong_heavy_tail_t3",
        "rate_pattern",
        1.0,
        -1.0,
        development_distribution="student_t3_unit_variance",
        minus_distribution="student_t3_unit_variance",
        plus_distribution="student_t3_unit_variance",
        selection_role="strong_robustness",
    ),
    Scenario("rate_very_strong", "rate_pattern", 1.50, -1.50),
    Scenario("rate_extreme", "rate_pattern", 2.0, -2.0),
    Scenario(
        "relative_strong_absolute_tiny",
        "scale_diagnostic",
        0.05,
        -0.05,
        sd_development=0.05,
        sd_minus=0.05,
        sd_plus=0.05,
        selection_role="absolute_margin_failure_diagnostic",
    ),
)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_scenario_seed(master_seed: int, scenario_id: str) -> int:
    material = f"{master_seed}:{scenario_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def assignment_weights() -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    assignments = tuple(itertools.combinations(range(N_TOTAL), N_MINUS))
    if len(assignments) != EXPECTED_ASSIGNMENTS:
        raise AssertionError("unexpected assignment count")
    weights = np.full((len(assignments), N_TOTAL), -1.0 / N_PLUS, dtype=np.float64)
    for row, indices in enumerate(assignments):
        weights[row, list(indices)] = 1.0 / N_MINUS
    if assignments[0] != tuple(range(N_MINUS)):
        raise AssertionError("observed assignment is not the first enumerated row")
    return weights, assignments


def rule_grid() -> tuple[Rule, ...]:
    rules: list[Rule] = []
    for (
        zero_alpha,
        practical_margin,
        practical_alpha,
        arm_familywise_alpha,
        arm_lower_margin,
        loo_floor,
    ) in itertools.product(
        (0.025, 0.05),
        (0.25, 0.50, 0.75),
        (0.05, 0.10),
        (0.05, 0.10),
        (0.0, 0.25, 0.50),
        (0.0, 0.25, 0.50),
    ):
        rule_id = (
            f"z{zero_alpha:.3f}_m{practical_margin:.2f}_"
            f"ma{practical_alpha:.2f}_af{arm_familywise_alpha:.2f}_"
            f"ab{arm_lower_margin:.2f}_l{loo_floor:.2f}"
        )
        rules.append(
            Rule(
                rule_id=rule_id,
                zero_alpha=zero_alpha,
                practical_margin=practical_margin,
                practical_alpha=practical_alpha,
                arm_familywise_alpha=arm_familywise_alpha,
                arm_lower_margin=arm_lower_margin,
                loo_delta_floor=loo_floor,
            )
        )
    return tuple(rules)


def random_noise(
    rng: np.random.Generator,
    distribution: str,
    shape: tuple[int, int],
) -> np.ndarray:
    if distribution == "gaussian":
        return rng.standard_normal(shape)
    if distribution == "student_t3_unit_variance":
        # A t(3) variate has variance 3; division yields unit variance.
        return rng.standard_t(3, size=shape) / math.sqrt(3.0)
    raise ValueError(f"unsupported distribution: {distribution}")


def simulate_scores(
    rng: np.random.Generator,
    scenario: Scenario,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    development = scenario.mean_development + scenario.sd_development * random_noise(
        rng,
        scenario.development_distribution,
        (size, N_DEVELOPMENT),
    )
    minus = scenario.mean_minus + scenario.sd_minus * random_noise(
        rng,
        scenario.minus_distribution,
        (size, N_MINUS),
    )
    plus = scenario.mean_plus + scenario.sd_plus * random_noise(
        rng,
        scenario.plus_distribution,
        (size, N_PLUS),
    )
    return development, np.concatenate((minus, plus), axis=1)


def loo_minimum_delta(scores: np.ndarray) -> np.ndarray:
    minus = scores[:, :N_MINUS]
    plus = scores[:, N_MINUS:]
    sum_minus = minus.sum(axis=1)
    sum_plus = plus.sum(axis=1)
    drops_minus = (sum_minus[:, None] - minus) / (N_MINUS - 1) - sum_plus[:, None] / N_PLUS
    drops_plus = sum_minus[:, None] / N_MINUS - (sum_plus[:, None] - plus) / (N_PLUS - 1)
    return np.minimum(drops_minus.min(axis=1), drops_plus.min(axis=1))


def welch_lower_bound(
    first: np.ndarray,
    second: np.ndarray,
    one_sided_alpha: float,
) -> np.ndarray:
    """Welch-Satterthwaite one-sided lower bound for mean(first-second)."""

    n_first = first.shape[1]
    n_second = second.shape[1]
    mean_difference = first.mean(axis=1) - second.mean(axis=1)
    variance_first = first.var(axis=1, ddof=1)
    variance_second = second.var(axis=1, ddof=1)
    component_first = variance_first / n_first
    component_second = variance_second / n_second
    standard_error = np.sqrt(component_first + component_second)
    denominator = (
        component_first * component_first / (n_first - 1)
        + component_second * component_second / (n_second - 1)
    )
    degrees_freedom = np.divide(
        (component_first + component_second) ** 2,
        denominator,
        out=np.full_like(denominator, np.inf),
        where=denominator > 0,
    )
    critical = student_t.ppf(1.0 - one_sided_alpha, degrees_freedom)
    return mean_difference - critical * standard_error


def scenario_metrics(
    development: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    practical_margins: Iterable[float],
    arm_familywise_alphas: Iterable[float],
) -> dict[str, np.ndarray]:
    development = np.asarray(development, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if development.ndim != 2 or development.shape[1] != N_DEVELOPMENT:
        raise ValueError(f"development must have shape (draws, {N_DEVELOPMENT})")
    if scores.ndim != 2 or scores.shape[1] != N_TOTAL:
        raise ValueError(f"scores must have shape (draws, {N_TOTAL})")
    if scores.shape[0] != development.shape[0]:
        raise ValueError("development and audit draws must have equal row counts")

    development_mean = development.mean(axis=1)
    development_scale = development.std(axis=1, ddof=1)
    if np.any(~np.isfinite(development_scale)) or np.any(development_scale <= 0):
        raise ValueError("synthetic development scale must be finite and positive")
    standardized_scores = (scores - development_mean[:, None]) / development_scale[:, None]

    permutation_statistics = standardized_scores @ weights.T
    observed_delta = permutation_statistics[:, 0]
    tolerance = 32.0 * np.finfo(np.float64).eps
    p_zero = np.mean(
        permutation_statistics >= observed_delta[:, None] - tolerance,
        axis=1,
    )

    observed_minus_indicator = np.concatenate((np.ones(N_MINUS), np.zeros(N_PLUS)))
    assignment_overlap_contrast = weights @ observed_minus_indicator
    metrics: dict[str, np.ndarray] = {
        "observed_delta": observed_delta,
        "mean_minus": standardized_scores[:, :N_MINUS].mean(axis=1),
        "mean_plus": standardized_scores[:, N_MINUS:].mean(axis=1),
        "development_mean": development_mean,
        "development_scale": development_scale,
        "loo_minimum_delta": loo_minimum_delta(standardized_scores),
        "p_zero": p_zero,
    }
    for margin in practical_margins:
        shifted_statistics = permutation_statistics - margin * assignment_overlap_contrast[None, :]
        shifted_observed = observed_delta - margin
        metrics[f"p_margin_{margin:.2f}"] = np.mean(
            shifted_statistics >= shifted_observed[:, None] - tolerance,
            axis=1,
        )
    for familywise_alpha in arm_familywise_alphas:
        per_arm_alpha = familywise_alpha / 2.0
        minus_lower_raw = welch_lower_bound(
            scores[:, :N_MINUS], development, per_arm_alpha
        )
        plus_lower_raw = welch_lower_bound(
            development, scores[:, N_MINUS:], per_arm_alpha
        )
        metrics[f"minus_arm_lower_{familywise_alpha:.2f}"] = (
            minus_lower_raw / development_scale
        )
        metrics[f"plus_arm_lower_{familywise_alpha:.2f}"] = (
            plus_lower_raw / development_scale
        )
    return metrics


def passes_rule(metrics: dict[str, np.ndarray], rule: Rule) -> np.ndarray:
    return (
        (metrics["p_zero"] <= rule.zero_alpha + 1e-15)
        & (metrics[f"p_margin_{rule.practical_margin:.2f}"] <= rule.practical_alpha + 1e-15)
        & (metrics["observed_delta"] >= rule.practical_margin)
        & (
            metrics[f"minus_arm_lower_{rule.arm_familywise_alpha:.2f}"]
            >= rule.arm_lower_margin
        )
        & (
            metrics[f"plus_arm_lower_{rule.arm_familywise_alpha:.2f}"]
            >= rule.arm_lower_margin
        )
        & (metrics["loo_minimum_delta"] > rule.loo_delta_floor)
    )


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(probability * (1.0 - probability) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def evaluate_grid(
    draws: int,
    batch_size: int,
    master_seed: int,
    scenarios: tuple[Scenario, ...],
    rules: tuple[Rule, ...],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, Any]]]:
    if draws <= 0 or batch_size <= 0:
        raise ValueError("draws and batch_size must be positive")
    weights, _ = assignment_weights()
    margins = sorted({rule.practical_margin for rule in rules})
    arm_familywise_alphas = sorted({rule.arm_familywise_alpha for rule in rules})
    counts: dict[str, dict[str, int]] = {
        rule.rule_id: {scenario.scenario_id: 0 for scenario in scenarios} for rule in rules
    }

    scenario_metadata: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        rng = np.random.Generator(np.random.PCG64(stable_scenario_seed(master_seed, scenario.scenario_id)))
        complete = 0
        while complete < draws:
            current = min(batch_size, draws - complete)
            development, scores = simulate_scores(rng, scenario, current)
            metrics = scenario_metrics(
                development,
                scores,
                weights,
                margins,
                arm_familywise_alphas,
            )
            for rule in rules:
                counts[rule.rule_id][scenario.scenario_id] += int(passes_rule(metrics, rule).sum())
            complete += current
        scenario_metadata[scenario.scenario_id] = {
            "family": scenario.family,
            "development_distribution": scenario.development_distribution,
            "minus_distribution": scenario.minus_distribution,
            "plus_distribution": scenario.plus_distribution,
            "mean_development": scenario.mean_development,
            "sd_development": scenario.sd_development,
            "mean_minus": scenario.mean_minus,
            "mean_plus": scenario.mean_plus,
            "sd_minus": scenario.sd_minus,
            "sd_plus": scenario.sd_plus,
            "true_delta": scenario.mean_minus - scenario.mean_plus,
            "selection_role": scenario.selection_role,
            "seed_hex": f"0x{stable_scenario_seed(master_seed, scenario.scenario_id):016x}",
        }

    rates = {
        rule_id: {scenario_id: successes / draws for scenario_id, successes in row.items()}
        for rule_id, row in counts.items()
    }
    return rates, scenario_metadata


def diagnose_rules(
    rules: tuple[Rule, ...],
    rates: dict[str, dict[str, float]],
    draws: int,
) -> dict[str, dict[str, float | bool]]:
    safety_ids = (
        "null_gaussian",
        "null_heavy_tail_t3",
        "null_heteroskedastic",
        "null_heteroskedastic_reverse",
        "null_heavy_development_tail",
        "shared_positive",
        "shared_negative",
        "reversed_rate_pattern",
    )
    one_arm_ids = (
        "one_arm_minus_only",
        "one_arm_plus_only",
        "one_arm_minus_extreme_inactive_sd2",
        "one_arm_plus_extreme_inactive_sd2",
        "one_arm_minus_extreme_inactive_t3_sd2",
        "one_arm_plus_extreme_inactive_t3_sd2",
    )
    moderate_ids = (
        "rate_moderate",
        "rate_moderate_heteroskedastic",
        "rate_moderate_heavy_tail_t3",
    )
    strong_robustness_ids = (
        "rate_strong_heteroskedastic",
        "rate_strong_heteroskedastic_reverse",
        "rate_strong_heavy_tail_t3",
    )

    diagnostics: dict[str, dict[str, float | bool]] = {}
    for rule in rules:
        row = rates[rule.rule_id]
        safety_bounds = [wilson_interval(int(round(row[item] * draws)), draws) for item in safety_ids]
        one_arm_bounds = [wilson_interval(int(round(row[item] * draws)), draws) for item in one_arm_ids]
        boundary_low, boundary_high = wilson_interval(
            int(round(row["practical_boundary_symmetric"] * draws)), draws
        )
        strong_low, strong_high = wilson_interval(
            int(round(row["rate_strong"] * draws)), draws
        )
        robust_bounds = [
            wilson_interval(int(round(row[item] * draws)), draws)
            for item in strong_robustness_ids
        ]
        max_safety = max(row[item] for item in safety_ids)
        max_one_arm = max(row[item] for item in one_arm_ids)
        strong_rate = row["rate_strong"]
        safety_eligible = (
            rule.practical_margin >= 0.50
            and rule.arm_lower_margin >= 0.0
            and rule.loo_delta_floor >= 0.0
            and max(high for _, high in safety_bounds) <= 0.05
            and max(high for _, high in one_arm_bounds) <= 0.05
            and boundary_high <= 0.10
        )
        meets = (
            safety_eligible
            and strong_low >= 0.50
            and min(low for low, _ in robust_bounds) >= 0.35
        )
        diagnostics[rule.rule_id] = {
            "eligible": meets,
            "safety_eligible": safety_eligible,
            "max_safety_support_rate": max_safety,
            "max_safety_wilson_95_upper": max(high for _, high in safety_bounds),
            "max_one_arm_support_rate": max_one_arm,
            "max_one_arm_wilson_95_upper": max(high for _, high in one_arm_bounds),
            "practical_boundary_support_rate": row["practical_boundary_symmetric"],
            "practical_boundary_wilson_95_upper": boundary_high,
            "strong_pattern_support_rate": strong_rate,
            "strong_pattern_wilson_95_lower": strong_low,
            "strong_pattern_wilson_95_upper": strong_high,
            "minimum_strong_robustness_support_rate": min(
                row[item] for item in strong_robustness_ids
            ),
            "minimum_strong_robustness_wilson_95_lower": min(
                low for low, _ in robust_bounds
            ),
            "minimum_moderate_pattern_support_rate": min(row[item] for item in moderate_ids),
            "mean_moderate_pattern_support_rate": sum(row[item] for item in moderate_ids)
            / len(moderate_ids),
        }
    return diagnostics


def choose_diagnostic_rule(
    rules: tuple[Rule, ...],
    diagnostics: dict[str, dict[str, float | bool]],
) -> Rule:
    safety_pool = [rule for rule in rules if diagnostics[rule.rule_id]["safety_eligible"]]
    pool = safety_pool if safety_pool else list(rules)

    def ranking_key(rule: Rule) -> tuple[float, ...]:
        item = diagnostics[rule.rule_id]
        # The diagnostic rule is not an accepted rule.  Among candidates that
        # pass the safety screen it exposes the best power achievable under the
        # generic Gaussian/t3 score model; remaining ties favor stricter bounds.
        return (
            float(item["strong_pattern_support_rate"]),
            float(item["minimum_strong_robustness_support_rate"]),
            float(item["minimum_moderate_pattern_support_rate"]),
            -float(item["max_one_arm_wilson_95_upper"]),
            -float(item["max_safety_support_rate"]),
            -rule.zero_alpha,
            -rule.practical_alpha,
            -rule.arm_familywise_alpha,
            rule.practical_margin,
            rule.arm_lower_margin,
            rule.loo_delta_floor,
        )

    return max(pool, key=ranking_key)


def rule_payload(rule: Rule) -> dict[str, Any]:
    zero_rank = math.floor(rule.zero_alpha * EXPECTED_ASSIGNMENTS + 1e-12)
    practical_rank = math.floor(rule.practical_alpha * EXPECTED_ASSIGNMENTS + 1e-12)
    return {
        "rule_id": rule.rule_id,
        "zero_effect_one_sided_exact_p_max": rule.zero_alpha,
        "zero_effect_max_extreme_allocations": zero_rank,
        "practical_margin_development_sd": rule.practical_margin,
        "practical_margin_one_sided_exact_p_max": rule.practical_alpha,
        "practical_margin_max_extreme_allocations": practical_rank,
        "observed_delta_development_sd_min": rule.practical_margin,
        "arm_direction_method": "Bonferroni-Welch one-sided lower bounds against n=9 development controls",
        "arm_direction_familywise_alpha": rule.arm_familywise_alpha,
        "arm_direction_per_arm_one_sided_alpha": rule.arm_familywise_alpha / 2.0,
        "stim_lick_minus_vs_development_lower_bound_development_sd_min": rule.arm_lower_margin,
        "development_vs_stim_lick_plus_lower_bound_development_sd_min": rule.arm_lower_margin,
        "every_leave_one_mouse_out_delta_development_sd_strictly_above": rule.loo_delta_floor,
        "logical_operator": "all",
    }


def tsv_text(headers: list[str], rows: Iterable[dict[str, Any]]) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=headers, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def render_markdown(payload: dict[str, Any]) -> str:
    diagnostic = payload["diagnostic_rule"]
    diagnostic_results = payload["diagnostic_rule_validation_operating_characteristics"]
    rows = "\n".join(
        "| `{}` | {}/{}/{} | {:+.2f} | {:+.2f} | {:.3f} | [{:.3f}, {:.3f}] |".format(
            item["scenario_id"],
            item["development_distribution"],
            item["minus_distribution"],
            item["plus_distribution"],
            item["mean_minus"],
            item["mean_plus"],
            item["support_rate"],
            item["wilson_95_low"],
            item["wilson_95_high"],
        )
        for item in diagnostic_results
    )
    return f"""# Outcome-blind small-`n` calibration

## Status

This calibration is **synthetic and outcome-blind**.  It read no empirical
mouse outcome, development score, intervention value, or audit result.  It
does not authorize the audit.

**No binding generic rule was selected.**  The generic score simulation either
fails the predeclared safety-plus-power criteria or, regardless of those
criteria, lacks a frozen raw behavioral endpoint and absolute raw-unit margin.
The old point-arm rule is withdrawn.  Endpoint-faithful binomial or
beta-binomial calibration is required after event semantics and denominators
are frozen.

## Design and exact test

- Inferential unit: one mouse.
- Development reference: nine independently simulated controls in every
  replicate; their sample mean and sample SD define that replicate's center
  and scale.
- Primary allocation: `n=6` calibrated `stimLick-` versus `n=5` calibrated
  `stimLick+`.
- Exact assignment space: all `{payload['design']['assignment_count']}`
  six-of-eleven allocations.
- Oriented contrast: `mean(stimLick-) - mean(stimLick+)`, in units of a
  location and scale frozen using development controls only.
- Fisher `p0` is exact only for the unit-level sharp no-effect null under the
  frozen assignment mechanism.  It is not an exact test of a heterogeneous
  weak mean null.
- The shifted test is exact only for its sharp constant-additive-shift null.
- A design-exact interpretation additionally requires an evaluator receipt
  proving the six-of-eleven assignment mechanism; the paper's prose alone is
  insufficient.
- Arm guards use two Bonferroni-adjusted one-sided Welch lower bounds against
  development controls.  They are model-based cohort comparisons, not
  randomized arm-versus-control tests.

## Best generic diagnostic rule—not authorized

The following rule is reported only to show the generic design's operating
boundary:

1. zero-effect sharp-null `p <= {diagnostic['zero_effect_one_sided_exact_p_max']:.3f}`
   (at most {diagnostic['zero_effect_max_extreme_allocations']} of 462
   allocations);
2. the shifted sharp-null test at `{diagnostic['practical_margin_development_sd']:.2f}`
   sample development SD has
   `p <= {diagnostic['practical_margin_one_sided_exact_p_max']:.2f}`
   (at most {diagnostic['practical_margin_max_extreme_allocations']} allocations),
   and the observed contrast is at least that margin;
3. both Bonferroni-Welch one-sided lower bounds, using familywise alpha
   `{diagnostic['arm_direction_familywise_alpha']:.2f}`, are at least
   `{diagnostic['stim_lick_minus_vs_development_lower_bound_development_sd_min']:.2f}`
   sample development SD in their predicted directions;
4. every leave-one-mouse-out contrast is strictly above
   `{diagnostic['every_leave_one_mouse_out_delta_development_sd_strictly_above']:.2f}`.

Even a pass is not eligible for a positive terminal.  The sample-SD margin can
become arbitrarily small when the nine-control scale is small, and a generic
continuous-score simulation cannot represent bounded trial counts,
overdispersion, unequal denominators, or endpoint ties.

## Independent-validation operating characteristics

Rule search and validation used independent seeds.  Each row below uses
{payload['simulation']['validation_draws_per_scenario']:,} validation draws.
Intervals are 95% Wilson intervals for Monte Carlo precision, not confidence
intervals on empirical mice.

| Scenario | Noise dev/minus/plus | Mean minus | Mean plus | Support rate | Monte Carlo 95% |
| --- | --- | ---: | ---: | ---: | ---: |
{rows}

The complete candidate grid and all scenario rates are in `rule_grid.tsv`.
Eligibility uses Wilson bounds, not point Monte Carlo estimates.  The expanded
one-arm family includes `(+3, 0)` and `(0, -3)` with inactive-arm SD 2 under
Gaussian and heavy-tailed noise.

## Required endpoint-specific calibration

Before any positive audit terminal can exist, freeze the behavioral event,
trial inclusion, per-mouse denominator, and absolute raw-unit effect margin;
then simulate bounded binomial and beta-binomial mice across development-only
overdispersion, variable denominators, missing blocks, and ties.  Re-run rule
selection and an independent validation seed without intervention outcomes.

## Interpretation limits

- Neither Fisher exact calculation is a weak mean-null test.
- Welch bounds are approximate under heavy tails and very small samples; their
  performance is displayed, not assumed.
- Eleven mice provide useful sensitivity mainly for large standardized
  patterns.  `closed_unresolved` is a valid and expected terminal.
- Development trials may make a per-mouse score more precise, but they do not
  increase the biological sample size.
- The rule cannot be used until score construction, missingness/abstention,
  development location/scale, codebook/timing, and evaluator behavior are
  frozen without consulting intervention outcomes.
"""


def write_outputs(
    output_dir: Path,
    selection_draws: int,
    validation_draws: int,
    batch_size: int,
    selection_seed: int,
    validation_seed: int,
    observed_on: str,
) -> dict[str, Any]:
    if selection_seed == validation_seed:
        raise ValueError("selection and validation seeds must be independent")
    rules = rule_grid()
    selection_rates, selection_metadata = evaluate_grid(
        selection_draws, batch_size, selection_seed, SCENARIOS, rules
    )
    validation_rates, validation_metadata = evaluate_grid(
        validation_draws, batch_size, validation_seed, SCENARIOS, rules
    )
    selection_diagnostics = diagnose_rules(rules, selection_rates, selection_draws)
    validation_diagnostics = diagnose_rules(rules, validation_rates, validation_draws)
    diagnostic_rule = choose_diagnostic_rule(rules, selection_diagnostics)
    diagnostic_id = diagnostic_rule.rule_id
    selection_eligible_ids = [
        rule.rule_id for rule in rules if selection_diagnostics[rule.rule_id]["eligible"]
    ]
    validation_eligible_ids = [
        rule.rule_id for rule in rules if validation_diagnostics[rule.rule_id]["eligible"]
    ]
    generic_passed_both = bool(
        selection_diagnostics[diagnostic_id]["eligible"]
        and validation_diagnostics[diagnostic_id]["eligible"]
    )

    diagnostic_results: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        rate = validation_rates[diagnostic_id][scenario.scenario_id]
        successes = int(round(rate * validation_draws))
        low, high = wilson_interval(successes, validation_draws)
        diagnostic_results.append(
            {
                "scenario_id": scenario.scenario_id,
                **validation_metadata[scenario.scenario_id],
                "draws": validation_draws,
                "support_count": successes,
                "support_rate": rate,
                "wilson_95_low": low,
                "wilson_95_high": high,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "observed_on": observed_on,
        "status": "no_binding_rule_endpoint_specific_calibration_required",
        "outcome_blind": True,
        "empirical_outcome_files_read": [],
        "audit_outcome_values_accessed": False,
        "canonical_state_changed": False,
        "execution": {
            "scheduler": "slurm" if os.environ.get("SLURM_JOB_ID") else "direct",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "code_sha256": sha256_file(Path(__file__).resolve()),
        },
        "design": {
            "inferential_unit": "mouse",
            "n_development_controls": N_DEVELOPMENT,
            "n_stim_lick_minus": N_MINUS,
            "n_stim_lick_plus": N_PLUS,
            "assignment_count": EXPECTED_ASSIGNMENTS,
            "trial_level_pseudoreplication_allowed": False,
            "contrast": "mean(stimLick-) - mean(stimLick+)",
            "alternative": "greater",
        },
        "score_contract": {
            "status": "not_frozen_generic_calibration_only",
            "generic_unit": "sample standard deviations of nine development controls",
            "reference_location_and_scale_simulated_per_replicate": True,
            "orientation": "higher favors the locked learning-rate-gate prediction",
            "required_arm_pattern": "stimLick- positive and stimLick+ negative around frozen development reference",
            "intervention_values_may_not_set_location_scale_or_orientation": True,
            "score_formula_frozen_by_this_calibration": False,
            "absolute_raw_unit_margin_frozen": False,
        },
        "exact_randomization_family": {
            "enumeration": "all six-of-eleven allocations",
            "zero_test": "Fisher one-sided exact test for the unit-level sharp no-effect null",
            "zero_test_is_exact_for_heterogeneous_weak_mean_null": False,
            "practical_test": "Fisher one-sided exact test after subtracting the practical margin from every observed stimLick- score",
            "practical_test_interpretation": "sharp constant-additive-shift null only",
            "design_exact_requires_assignment_mechanism_receipt": True,
            "ties": "included as at-least-as-extreme",
            "arm_direction_gates_are_randomized_arm_vs_control_tests": False,
        },
        "arm_direction_guard": {
            "method": "two one-sided Welch-Satterthwaite lower bounds with Bonferroni familywise allocation",
            "development_reference_uncertainty_included": True,
            "small_sample_model_based": True,
            "weak_under_heavy_tails": True,
        },
        "simulation": {
            "selection_master_seed_hex": f"0x{selection_seed:x}",
            "validation_master_seed_hex": f"0x{validation_seed:x}",
            "selection_and_validation_seeds_independent": selection_seed != validation_seed,
            "bit_generator": "numpy.random.PCG64",
            "scenario_seed_derivation": "uint64_le(first_8_bytes(SHA256(f'{master_seed}:{scenario_id}')))",
            "selection_draws_per_scenario": selection_draws,
            "validation_draws_per_scenario": validation_draws,
            "batch_size": batch_size,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "python_version": platform.python_version(),
            "selection_scenarios": selection_metadata,
            "validation_scenarios": validation_metadata,
        },
        "selection": {
            "binding_rule_selected": False,
            "selected_rule_id": None,
            "candidate_rule_count": len(rules),
            "selection_eligible_rule_count": len(selection_eligible_ids),
            "validation_eligible_rule_count": len(validation_eligible_ids),
            "generic_diagnostic_rule_passed_both": generic_passed_both,
            "diagnostic_rule_id": diagnostic_id,
            "diagnostic_rule_selection_diagnostics": selection_diagnostics[diagnostic_id],
            "diagnostic_rule_validation_diagnostics": validation_diagnostics[diagnostic_id],
            "constraints_use_wilson_95_bounds": True,
            "constraints": {
                "minimum_practical_margin_sample_development_sd": 0.50,
                "minimum_arm_lower_bound_sample_development_sd": 0.0,
                "minimum_loo_delta_floor_sample_development_sd": 0.0,
                "maximum_safety_scenario_wilson_95_upper": 0.05,
                "maximum_one_arm_scenario_wilson_95_upper": 0.05,
                "maximum_practical_boundary_wilson_95_upper": 0.10,
                "minimum_strong_pattern_wilson_95_lower": 0.50,
                "minimum_strong_robustness_wilson_95_lower": 0.35,
            },
        },
        "selected_rule": None,
        "diagnostic_rule": rule_payload(diagnostic_rule),
        "diagnostic_rule_validation_operating_characteristics": diagnostic_results,
        "terminal_guidance": {
            "positive_mechanistic_terminal_authorized": False,
            "reason": "no endpoint-faithful calibrated binding rule",
            "any_generic_rule_result": "closed_unresolved_or_technical_calibration_failure",
            "invalid_score_missingness_or_evaluator_failure": "technical_failure",
            "boundary_high_amplitude_cohort_may_rescue_primary_failure": False,
        },
        "required_endpoint_specific_calibration": {
            "families": ["binomial", "beta_binomial"],
            "freeze_before_simulation": [
                "behavioral event and lickState/trialID semantics",
                "per-mouse numerator and denominator",
                "trial inclusion missing-block and abstention rules",
                "absolute raw-unit practical margin",
                "development-only overdispersion grid or conservative bound",
                "assignment-mechanism receipt",
            ],
            "stress_cases": [
                "variable denominators",
                "endpoint ties and discreteness",
                "beta-binomial overdispersion",
                "near-boundary probabilities",
                "missing acquisition blocks",
                "one-arm-only changes",
                "reverse heteroskedasticity",
            ],
            "independent_validation_seed_required": True,
        },
        "signoff_blockers": [
            "freeze the per-mouse score formula and its causal time support",
            "replace sample-SD-only margins with a scientific absolute raw-unit floor",
            "complete endpoint-faithful binomial/beta-binomial calibration after semantics freeze",
            "obtain a hash-bound randomization-mechanism receipt for six-of-eleven enumeration",
            "freeze missingness, minimum support, and abstention behavior",
            "freeze the trialID and 701-sample timing convention or declare them unusable",
            "bind the exact evaluator implementation and prove it emits one atomic terminal packet",
            "obtain explicit scientist signoff on the endpoint-specific thresholds and terminal wording",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "calibration.json", payload)

    scenario_headers = [
        "scenario_id",
        "family",
        "selection_role",
        "development_distribution",
        "minus_distribution",
        "plus_distribution",
        "mean_development",
        "mean_minus",
        "mean_plus",
        "true_delta",
        "sd_minus",
        "sd_plus",
        "draws",
        "support_count",
        "support_rate",
        "wilson_95_low",
        "wilson_95_high",
    ]
    scenario_rows = []
    for item in diagnostic_results:
        scenario_rows.append(
            {
                key: f"{item[key]:.6f}" if isinstance(item[key], float) else item[key]
                for key in scenario_headers
            }
        )
    atomic_write_text(output_dir / "scenario_results.tsv", tsv_text(scenario_headers, scenario_rows))

    scenario_ids = [scenario.scenario_id for scenario in SCENARIOS]
    grid_headers = [
        "rule_id",
        "diagnostic_rule",
        "selection_eligible",
        "validation_eligible",
        "zero_alpha",
        "practical_margin",
        "practical_alpha",
        "arm_familywise_alpha",
        "arm_lower_margin",
        "loo_delta_floor",
        "selection_max_safety_wilson_95_upper",
        "selection_max_one_arm_wilson_95_upper",
        "selection_strong_pattern_wilson_95_lower",
        "validation_max_safety_wilson_95_upper",
        "validation_max_one_arm_wilson_95_upper",
        "validation_strong_pattern_wilson_95_lower",
        *[f"selection__{item}" for item in scenario_ids],
        *[f"validation__{item}" for item in scenario_ids],
    ]
    grid_rows: list[dict[str, Any]] = []
    for rule in rules:
        selection_item = selection_diagnostics[rule.rule_id]
        validation_item = validation_diagnostics[rule.rule_id]
        row: dict[str, Any] = {
            "rule_id": rule.rule_id,
            "diagnostic_rule": str(rule.rule_id == diagnostic_id).lower(),
            "selection_eligible": str(selection_item["eligible"]).lower(),
            "validation_eligible": str(validation_item["eligible"]).lower(),
            "zero_alpha": f"{rule.zero_alpha:.3f}",
            "practical_margin": f"{rule.practical_margin:.2f}",
            "practical_alpha": f"{rule.practical_alpha:.2f}",
            "arm_familywise_alpha": f"{rule.arm_familywise_alpha:.2f}",
            "arm_lower_margin": f"{rule.arm_lower_margin:.2f}",
            "loo_delta_floor": f"{rule.loo_delta_floor:.2f}",
            "selection_max_safety_wilson_95_upper": f"{float(selection_item['max_safety_wilson_95_upper']):.6f}",
            "selection_max_one_arm_wilson_95_upper": f"{float(selection_item['max_one_arm_wilson_95_upper']):.6f}",
            "selection_strong_pattern_wilson_95_lower": f"{float(selection_item['strong_pattern_wilson_95_lower']):.6f}",
            "validation_max_safety_wilson_95_upper": f"{float(validation_item['max_safety_wilson_95_upper']):.6f}",
            "validation_max_one_arm_wilson_95_upper": f"{float(validation_item['max_one_arm_wilson_95_upper']):.6f}",
            "validation_strong_pattern_wilson_95_lower": f"{float(validation_item['strong_pattern_wilson_95_lower']):.6f}",
        }
        for scenario_id in scenario_ids:
            row[f"selection__{scenario_id}"] = f"{selection_rates[rule.rule_id][scenario_id]:.6f}"
            row[f"validation__{scenario_id}"] = f"{validation_rates[rule.rule_id][scenario_id]:.6f}"
        grid_rows.append(row)
    atomic_write_text(output_dir / "rule_grid.tsv", tsv_text(grid_headers, grid_rows))
    atomic_write_text(output_dir / "CALIBRATION.md", render_markdown(payload))

    manifest_names = ("CALIBRATION.md", "calibration.json", "rule_grid.tsv", "scenario_results.tsv")
    manifest = "".join(f"{sha256_file(output_dir / name)}  {name}\n" for name in manifest_names)
    atomic_write_text(output_dir / "SHA256SUMS", manifest)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-draws", type=int, default=50_000)
    parser.add_argument("--validation-draws", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--selection-seed", type=int, default=20_260_922)
    parser.add_argument("--validation-seed", type=int, default=120_260_922)
    parser.add_argument("--observed-on", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_outputs(
        output_dir=args.output_dir,
        selection_draws=args.selection_draws,
        validation_draws=args.validation_draws,
        batch_size=args.batch_size,
        selection_seed=args.selection_seed,
        validation_seed=args.validation_seed,
        observed_on=args.observed_on,
    )
    print(
        json.dumps(
            {
                "assignment_count": payload["design"]["assignment_count"],
                "selection_draws_per_scenario": payload["simulation"]["selection_draws_per_scenario"],
                "validation_draws_per_scenario": payload["simulation"]["validation_draws_per_scenario"],
                "selected_rule_id": payload["selection"]["selected_rule_id"],
                "diagnostic_rule_id": payload["selection"]["diagnostic_rule_id"],
                "status": payload["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
