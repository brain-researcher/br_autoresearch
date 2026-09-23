#!/usr/bin/env python3
"""Outcome-blind binomial/beta-binomial calibration for Dudman EP02.

This is an endpoint-faithful *provisional* calibration engine: each synthetic
mouse score is a difference of two bounded session proportions with explicit
integer numerators and denominators.  It never reads empirical outcome files.
Unresolved endpoint receipts and an unsigned scientist decision force the
binding rule to remain null even when a rule has acceptable simulated
operating characteristics.
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
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "ep02.dudman.endpoint_small_n_calibration.v1"
N_DEVELOPMENT = 9
N_MINUS = 6
N_PLUS = 5
N_AUDIT = N_MINUS + N_PLUS
ASSIGNMENT_COUNT = math.comb(N_AUDIT, N_MINUS)
MIN_DENOMINATOR = 30


@dataclass(frozen=True)
class Denominators:
    early_low: int = 75
    early_high: int = 95
    late_low: int = 65
    late_high: int = 85


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    family: str
    p_early_development: float
    p_late_development: float
    p_early_minus: float
    p_late_minus: float
    p_early_plus: float
    p_late_plus: float
    concentration: float | None = None
    development_denominators: Denominators = Denominators()
    minus_denominators: Denominators = Denominators()
    plus_denominators: Denominators = Denominators()
    structural_failure_probability: float = 0.0
    selection_role: str = "diagnostic"


@dataclass(frozen=True)
class Rule:
    rule_id: str
    zero_alpha: float
    practical_margin_raw: float
    practical_alpha: float
    arm_margin_raw: float
    development_scale_floor_raw: float
    loo_delta_floor_raw: float


REGULAR = Denominators()
WIDE = Denominators(30, 95, 30, 85)
MINUS_SPARSE = Denominators(30, 55, 30, 55)
PLUS_DENSE = Denominators(80, 95, 70, 90)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("null_binomial", "binomial", 0.10, 0.60, 0.10, 0.60, 0.10, 0.60, selection_role="safety"),
    Scenario("null_beta_phi20", "beta_binomial", 0.10, 0.60, 0.10, 0.60, 0.10, 0.60, 20.0, selection_role="safety"),
    Scenario("null_beta_phi5", "beta_binomial", 0.10, 0.60, 0.10, 0.60, 0.10, 0.60, 5.0, selection_role="safety"),
    Scenario(
        "null_denominator_heterogeneous",
        "beta_binomial",
        0.10,
        0.60,
        0.10,
        0.60,
        0.10,
        0.60,
        20.0,
        REGULAR,
        MINUS_SPARSE,
        PLUS_DENSE,
        selection_role="safety",
    ),
    Scenario("shared_positive_beta", "beta_binomial", 0.10, 0.60, 0.10, 0.75, 0.10, 0.75, 20.0, selection_role="safety"),
    Scenario("shared_negative_beta", "beta_binomial", 0.10, 0.60, 0.10, 0.45, 0.10, 0.45, 20.0, selection_role="safety"),
    Scenario("reversed_rate_beta", "beta_binomial", 0.10, 0.60, 0.10, 0.45, 0.10, 0.75, 20.0, selection_role="safety"),
    Scenario("near_zero_scale_null", "binomial", 0.10, 0.12, 0.10, 0.12, 0.10, 0.12, selection_role="safety"),
    Scenario("one_arm_minus_beta", "beta_binomial", 0.10, 0.60, 0.10, 0.85, 0.10, 0.60, 20.0, selection_role="one_arm"),
    Scenario("one_arm_plus_beta", "beta_binomial", 0.10, 0.60, 0.10, 0.60, 0.10, 0.35, 20.0, selection_role="one_arm"),
    Scenario(
        "one_arm_minus_sparse",
        "beta_binomial",
        0.10,
        0.60,
        0.10,
        0.90,
        0.10,
        0.60,
        10.0,
        REGULAR,
        MINUS_SPARSE,
        PLUS_DENSE,
        selection_role="one_arm",
    ),
    Scenario(
        "one_arm_plus_sparse",
        "beta_binomial",
        0.10,
        0.60,
        0.10,
        0.60,
        0.10,
        0.30,
        10.0,
        REGULAR,
        PLUS_DENSE,
        MINUS_SPARSE,
        selection_role="one_arm",
    ),
    Scenario("practical_boundary_raw_0p10", "binomial", 0.10, 0.60, 0.10, 0.65, 0.10, 0.55, selection_role="boundary"),
    Scenario("rate_moderate_binomial", "binomial", 0.10, 0.60, 0.10, 0.70, 0.10, 0.50, selection_role="moderate_target"),
    Scenario("rate_strong_binomial", "binomial", 0.10, 0.60, 0.10, 0.75, 0.10, 0.45, selection_role="strong_target"),
    Scenario("rate_strong_beta_phi20", "beta_binomial", 0.10, 0.60, 0.10, 0.75, 0.10, 0.45, 20.0, selection_role="strong_robustness"),
    Scenario("rate_strong_beta_phi5", "beta_binomial", 0.10, 0.60, 0.10, 0.75, 0.10, 0.45, 5.0, selection_role="strong_robustness"),
    Scenario(
        "rate_strong_denominator_heterogeneous",
        "beta_binomial",
        0.10,
        0.60,
        0.10,
        0.75,
        0.10,
        0.45,
        20.0,
        REGULAR,
        MINUS_SPARSE,
        PLUS_DENSE,
        selection_role="strong_robustness",
    ),
    Scenario(
        "rate_strong_structural_failure",
        "beta_binomial",
        0.10,
        0.60,
        0.10,
        0.75,
        0.10,
        0.45,
        20.0,
        WIDE,
        WIDE,
        WIDE,
        structural_failure_probability=0.05,
        selection_role="structural_abstention",
    ),
    Scenario("near_probability_boundary", "beta_binomial", 0.02, 0.20, 0.02, 0.30, 0.02, 0.10, 20.0, selection_role="moderate_target"),
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def stable_seed(master_seed: int, scenario_id: str) -> int:
    material = f"{master_seed}:{scenario_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def assignment_weights() -> tuple[np.ndarray, np.ndarray]:
    assignments = tuple(itertools.combinations(range(N_AUDIT), N_MINUS))
    if len(assignments) != ASSIGNMENT_COUNT or assignments[0] != tuple(range(N_MINUS)):
        raise AssertionError("unexpected assignment enumeration")
    indicators = np.zeros((ASSIGNMENT_COUNT, N_AUDIT), dtype=np.float64)
    for row, indices in enumerate(assignments):
        indicators[row, list(indices)] = 1.0
    weights = indicators / N_MINUS - (1.0 - indicators) / N_PLUS
    return weights, indicators


def rule_grid() -> tuple[Rule, ...]:
    rules: list[Rule] = []
    for zero, margin, margin_alpha, arm, scale, loo in itertools.product(
        (0.025, 0.05),
        (0.10, 0.15, 0.20, 0.25),
        (0.05, 0.10),
        (0.025, 0.05, 0.10, 0.15),
        (0.02, 0.04, 0.06),
        (0.00, 0.05),
    ):
        rule_id = (
            f"z{zero:.3f}_m{margin:.2f}_ma{margin_alpha:.2f}_"
            f"a{arm:.3f}_s{scale:.2f}_l{loo:.2f}"
        )
        rules.append(Rule(rule_id, zero, margin, margin_alpha, arm, scale, loo))
    return tuple(rules)


def _probabilities(
    rng: np.random.Generator, mean: float, concentration: float | None, shape: tuple[int, int]
) -> np.ndarray:
    if concentration is None:
        return np.full(shape, mean, dtype=np.float64)
    clipped = min(max(mean, 1e-8), 1.0 - 1e-8)
    return rng.beta(clipped * concentration, (1.0 - clipped) * concentration, size=shape)


def _mouse_gains(
    rng: np.random.Generator,
    draws: int,
    mice: int,
    p_early: float,
    p_late: float,
    concentration: float | None,
    denominators: Denominators,
    structural_failure_probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    early_n = rng.integers(denominators.early_low, denominators.early_high + 1, size=(draws, mice))
    late_n = rng.integers(denominators.late_low, denominators.late_high + 1, size=(draws, mice))
    if structural_failure_probability:
        failure = rng.random((draws, mice)) < structural_failure_probability
        late_n = late_n.copy()
        late_n[failure] = MIN_DENOMINATOR - 1
    early_p = _probabilities(rng, p_early, concentration, (draws, mice))
    late_p = _probabilities(rng, p_late, concentration, (draws, mice))
    early_k = rng.binomial(early_n, early_p)
    late_k = rng.binomial(late_n, late_p)
    gains = late_k / late_n - early_k / early_n
    valid = (early_n >= MIN_DENOMINATOR) & (late_n >= MIN_DENOMINATOR)
    return gains, valid


def simulate_scenario(
    rng: np.random.Generator, scenario: Scenario, draws: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    concentration = scenario.concentration if scenario.family == "beta_binomial" else None
    if scenario.family not in {"binomial", "beta_binomial"}:
        raise ValueError(f"unsupported family: {scenario.family}")
    development, valid_development = _mouse_gains(
        rng,
        draws,
        N_DEVELOPMENT,
        scenario.p_early_development,
        scenario.p_late_development,
        concentration,
        scenario.development_denominators,
        scenario.structural_failure_probability,
    )
    minus, valid_minus = _mouse_gains(
        rng,
        draws,
        N_MINUS,
        scenario.p_early_minus,
        scenario.p_late_minus,
        concentration,
        scenario.minus_denominators,
        scenario.structural_failure_probability,
    )
    plus, valid_plus = _mouse_gains(
        rng,
        draws,
        N_PLUS,
        scenario.p_early_plus,
        scenario.p_late_plus,
        concentration,
        scenario.plus_denominators,
        scenario.structural_failure_probability,
    )
    structural_valid = np.concatenate((valid_development, valid_minus, valid_plus), axis=1).all(axis=1)
    return development, np.concatenate((minus, plus), axis=1), structural_valid


def loo_minimum_delta(scores: np.ndarray) -> np.ndarray:
    minus = scores[:, :N_MINUS]
    plus = scores[:, N_MINUS:]
    sum_minus = minus.sum(axis=1)
    sum_plus = plus.sum(axis=1)
    drop_minus = (sum_minus[:, None] - minus) / (N_MINUS - 1) - sum_plus[:, None] / N_PLUS
    drop_plus = sum_minus[:, None] / N_MINUS - (sum_plus[:, None] - plus) / (N_PLUS - 1)
    return np.minimum(drop_minus.min(axis=1), drop_plus.min(axis=1))


def scenario_metrics(
    development: np.ndarray,
    audit: np.ndarray,
    structural_valid: np.ndarray,
    weights: np.ndarray,
    practical_margins: Iterable[float],
) -> dict[str, np.ndarray]:
    if development.ndim != 2 or development.shape[1] != N_DEVELOPMENT:
        raise ValueError("development shape mismatch")
    if audit.ndim != 2 or audit.shape[1] != N_AUDIT:
        raise ValueError("audit shape mismatch")
    if development.shape[0] != audit.shape[0] or structural_valid.shape != (audit.shape[0],):
        raise ValueError("draw count mismatch")
    development_mean = development.mean(axis=1)
    development_scale = development.std(axis=1, ddof=1)
    permutation_statistics = audit @ weights.T
    observed_delta = permutation_statistics[:, 0]
    tolerance = 32.0 * np.finfo(np.float64).eps
    metrics: dict[str, np.ndarray] = {
        "structural_valid": structural_valid,
        "development_mean": development_mean,
        "development_scale": development_scale,
        "observed_delta_raw": observed_delta,
        "minus_vs_development_raw": audit[:, :N_MINUS].mean(axis=1) - development_mean,
        "development_vs_plus_raw": development_mean - audit[:, N_MINUS:].mean(axis=1),
        "loo_minimum_delta_raw": loo_minimum_delta(audit),
        "p_zero": np.mean(permutation_statistics >= observed_delta[:, None] - tolerance, axis=1),
    }
    observed_minus = np.concatenate((np.ones(N_MINUS), np.zeros(N_PLUS)))
    overlap = weights @ observed_minus
    for margin in practical_margins:
        shifted = permutation_statistics - margin * overlap[None, :]
        metrics[f"p_margin_{margin:.2f}"] = np.mean(
            shifted >= (observed_delta - margin)[:, None] - tolerance,
            axis=1,
        )
    return metrics


def passes_rule(metrics: dict[str, np.ndarray], rule: Rule) -> np.ndarray:
    finite_scale = np.isfinite(metrics["development_scale"])
    return (
        metrics["structural_valid"]
        & finite_scale
        & (metrics["development_scale"] >= rule.development_scale_floor_raw)
        & (metrics["p_zero"] <= rule.zero_alpha + 1e-15)
        & (metrics[f"p_margin_{rule.practical_margin_raw:.2f}"] <= rule.practical_alpha + 1e-15)
        & (metrics["observed_delta_raw"] >= rule.practical_margin_raw)
        & (metrics["minus_vs_development_raw"] >= rule.arm_margin_raw)
        & (metrics["development_vs_plus_raw"] >= rule.arm_margin_raw)
        & (metrics["loo_minimum_delta_raw"] > rule.loo_delta_floor_raw)
    )


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def evaluate_grid(
    draws: int,
    batch_size: int,
    master_seed: int,
    scenarios: tuple[Scenario, ...],
    rules: tuple[Rule, ...],
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    if draws <= 0 or batch_size <= 0:
        raise ValueError("draws and batch_size must be positive")
    weights, _ = assignment_weights()
    margins = sorted({rule.practical_margin_raw for rule in rules})
    counts = {rule.rule_id: {scenario.scenario_id: 0 for scenario in scenarios} for rule in rules}
    structural_valid_counts = {scenario.scenario_id: 0 for scenario in scenarios}
    for scenario in scenarios:
        rng = np.random.Generator(np.random.PCG64(stable_seed(master_seed, scenario.scenario_id)))
        completed = 0
        while completed < draws:
            current = min(batch_size, draws - completed)
            development, audit, structural_valid = simulate_scenario(rng, scenario, current)
            metrics = scenario_metrics(development, audit, structural_valid, weights, margins)
            structural_valid_counts[scenario.scenario_id] += int(structural_valid.sum())
            for rule in rules:
                counts[rule.rule_id][scenario.scenario_id] += int(passes_rule(metrics, rule).sum())
            completed += current
    return counts, structural_valid_counts


def diagnose(
    rules: tuple[Rule, ...], counts: dict[str, dict[str, int]], draws: int
) -> dict[str, dict[str, Any]]:
    safety = tuple(s.scenario_id for s in SCENARIOS if s.selection_role == "safety")
    one_arm = tuple(s.scenario_id for s in SCENARIOS if s.selection_role == "one_arm")
    robust = tuple(s.scenario_id for s in SCENARIOS if s.selection_role == "strong_robustness")
    diagnostics: dict[str, dict[str, Any]] = {}
    for rule in rules:
        row = counts[rule.rule_id]
        intervals = {name: wilson_interval(row[name], draws) for name in row}
        safety_upper = max(intervals[name][1] for name in safety)
        one_arm_upper = max(intervals[name][1] for name in one_arm)
        boundary_upper = intervals["practical_boundary_raw_0p10"][1]
        strong_lower = intervals["rate_strong_binomial"][0]
        robust_lower = min(intervals[name][0] for name in robust)
        safety_eligible = (
            rule.practical_margin_raw >= 0.15
            and rule.arm_margin_raw >= 0.025
            and rule.development_scale_floor_raw >= 0.02
            and safety_upper <= 0.05
            and one_arm_upper <= 0.05
            and boundary_upper <= 0.10
        )
        diagnostics[rule.rule_id] = {
            "safety_eligible": safety_eligible,
            "selection_eligible": safety_eligible and strong_lower >= 0.50 and robust_lower >= 0.30,
            "max_safety_wilson_95_upper": safety_upper,
            "max_one_arm_wilson_95_upper": one_arm_upper,
            "boundary_wilson_95_upper": boundary_upper,
            "strong_binomial_support_rate": row["rate_strong_binomial"] / draws,
            "strong_binomial_wilson_95_lower": strong_lower,
            "minimum_strong_robustness_support_rate": min(row[name] / draws for name in robust),
            "minimum_strong_robustness_wilson_95_lower": robust_lower,
            "moderate_binomial_support_rate": row["rate_moderate_binomial"] / draws,
            "near_boundary_support_rate": row["near_probability_boundary"] / draws,
        }
    return diagnostics


def choose_provisional_rule(
    rules: tuple[Rule, ...], diagnostics: dict[str, dict[str, Any]]
) -> Rule | None:
    eligible = [rule for rule in rules if diagnostics[rule.rule_id]["selection_eligible"]]
    if not eligible:
        return None

    def rank(rule: Rule) -> tuple[float, ...]:
        item = diagnostics[rule.rule_id]
        return (
            item["minimum_strong_robustness_support_rate"],
            item["strong_binomial_support_rate"],
            item["moderate_binomial_support_rate"],
            -item["max_one_arm_wilson_95_upper"],
            -item["max_safety_wilson_95_upper"],
            rule.practical_margin_raw,
            rule.arm_margin_raw,
            rule.development_scale_floor_raw,
            rule.loo_delta_floor_raw,
            -rule.zero_alpha,
            -rule.practical_alpha,
        )

    return max(eligible, key=rank)


def choose_diagnostic_rule(
    rules: tuple[Rule, ...], diagnostics: dict[str, dict[str, Any]]
) -> Rule:
    safety_pool = [rule for rule in rules if diagnostics[rule.rule_id]["safety_eligible"]]
    pool = safety_pool if safety_pool else list(rules)

    def rank(rule: Rule) -> tuple[float, ...]:
        item = diagnostics[rule.rule_id]
        return (
            item["minimum_strong_robustness_support_rate"],
            item["strong_binomial_support_rate"],
            item["moderate_binomial_support_rate"],
            item["near_boundary_support_rate"],
            -item["max_one_arm_wilson_95_upper"],
            -item["max_safety_wilson_95_upper"],
            rule.practical_margin_raw,
            rule.arm_margin_raw,
            rule.development_scale_floor_raw,
            rule.loo_delta_floor_raw,
            -rule.zero_alpha,
            -rule.practical_alpha,
        )

    return max(pool, key=rank)


def tsv(headers: list[str], rows: Iterable[dict[str, Any]]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=headers, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def evidence_gate(receipt_directory: Path, signoff_path: Path) -> dict[str, Any]:
    receipt_names = (
        "trialID_codebook_receipt.json",
        "seshID_training_day_semantics_receipt.json",
        "lickState_750ms_trigger_semantics_receipt.json",
        "raw_701_sample_axis_and_event_alignment_receipt.json",
        "randomization_mechanism_receipt.json",
    )
    receipts: dict[str, Any] = {}
    failures: list[str] = []
    for name in receipt_names:
        path = receipt_directory / name
        if not path.exists():
            failures.append(f"missing_receipt:{name}")
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        receipts[name] = {"status": value.get("status")}
        if value.get("launch_gate_satisfied") is not True:
            failures.append(f"receipt_not_resolved:{name}")
    if not signoff_path.exists():
        failures.append("scientist_signoff_missing")
        signoff = {}
    else:
        signoff = json.loads(signoff_path.read_text(encoding="utf-8"))
        if signoff.get("status") != "SIGNED" or not signoff.get("scientist_signature"):
            failures.append("scientist_signoff_not_signed")
    return {
        "binding_eligible": not failures,
        "failures": sorted(failures),
        "receipts": receipts,
        "scientist_signoff_status": signoff.get("status"),
    }


def rule_dict(rule: Rule | None) -> dict[str, Any] | None:
    return asdict(rule) if rule else None


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    rules = rule_grid()
    selection_counts, selection_valid = evaluate_grid(
        arguments.draws,
        arguments.batch_size,
        arguments.selection_seed,
        SCENARIOS,
        rules,
    )
    selection_diagnostics = diagnose(rules, selection_counts, arguments.draws)
    provisional = choose_provisional_rule(rules, selection_diagnostics)
    diagnostic = choose_diagnostic_rule(rules, selection_diagnostics)
    validation_counts, validation_valid = evaluate_grid(
        arguments.draws,
        arguments.batch_size,
        arguments.validation_seed,
        SCENARIOS,
        rules,
    )
    validation_diagnostics = diagnose(rules, validation_counts, arguments.draws)
    validation_eligible = (
        provisional is not None and validation_diagnostics[provisional.rule_id]["selection_eligible"]
    )
    reported_rule = provisional or diagnostic
    gate = evidence_gate(arguments.receipt_directory, arguments.signoff)
    binding_allowed = bool(gate["binding_eligible"] and validation_eligible)
    if binding_allowed:
        raise RuntimeError(
            "This v1 engine never self-binds. A signed policy amendment must import the validated rule."
        )

    scenario_rows: list[dict[str, Any]] = []
    for stream, counts, valid in (
        ("selection", selection_counts, selection_valid),
        ("validation", validation_counts, validation_valid),
    ):
        for scenario in SCENARIOS:
            selected_count = counts[reported_rule.rule_id][scenario.scenario_id]
            low, high = wilson_interval(selected_count, arguments.draws)
            scenario_rows.append(
                {
                    "stream": stream,
                    "scenario_id": scenario.scenario_id,
                    "family": scenario.family,
                    "selection_role": scenario.selection_role,
                    "structural_valid_rate": valid[scenario.scenario_id] / arguments.draws,
                    "reported_rule_support_rate": selected_count / arguments.draws,
                    "wilson_95_low": low,
                    "wilson_95_high": high,
                }
            )
    rule_rows: list[dict[str, Any]] = []
    for rule in rules:
        row = asdict(rule)
        row.update({f"selection_{k}": v for k, v in selection_diagnostics[rule.rule_id].items()})
        row.update({f"validation_{k}": v for k, v in validation_diagnostics[rule.rule_id].items()})
        rule_rows.append(row)

    binding_blockers = list(gate["failures"])
    if provisional is None:
        binding_blockers.append("no_selection_eligible_rule")
    if not validation_eligible:
        binding_blockers.append("no_independently_validated_rule")
    signoff_alone_can_bind = bool(
        validation_eligible
        and set(gate["failures"]) == {"scientist_signoff_not_signed"}
    )

    calibration_status = (
        "NO_ELIGIBLE_RULE_PROVISIONAL_NONBINDING_AND_RECEIPTS_SIGNOFF_UNRESOLVED"
        if provisional is None or not validation_eligible
        else "PROVISIONAL_RULE_VALIDATED_BUT_NONBINDING_RECEIPTS_SIGNOFF_UNRESOLVED"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": calibration_status,
        "outcome_blind": True,
        "empirical_outcome_files_read": [],
        "audit_outcome_values_accessed": False,
        "canonical_state_changed": False,
        "design": {
            "inferential_unit": "mouse",
            "n_development_controls": N_DEVELOPMENT,
            "n_stim_lick_minus": N_MINUS,
            "n_stim_lick_plus": N_PLUS,
            "assignment_count": ASSIGNMENT_COUNT,
            "endpoint": "session_8_binary_proportion_minus_session_1_binary_proportion",
            "minimum_eligible_trials_per_epoch": MIN_DENOMINATOR,
            "families": ["binomial", "beta_binomial"],
            "trial_level_pseudoreplication_allowed": False,
        },
        "simulation": {
            "draws_per_scenario_per_stream": arguments.draws,
            "batch_size": arguments.batch_size,
            "selection_master_seed_hex": f"0x{arguments.selection_seed:x}",
            "validation_master_seed_hex": f"0x{arguments.validation_seed:x}",
            "selection_and_validation_seeds_independent": arguments.selection_seed != arguments.validation_seed,
            "scenario_count": len(SCENARIOS),
            "rule_count": len(rules),
            "synthetic_scenario_source": "prespecified_stress_grid_not_fit_to_development_or_audit_outcomes",
            "scenario_definitions": {
                scenario.scenario_id: asdict(scenario) for scenario in SCENARIOS
            },
            "selection_criteria": {
                "practical_margin_raw_min": 0.15,
                "arm_margin_raw_min": 0.025,
                "development_scale_floor_raw_min": 0.02,
                "max_safety_wilson_95_upper": 0.05,
                "max_one_arm_wilson_95_upper": 0.05,
                "raw_0p10_boundary_wilson_95_upper": 0.10,
                "strong_binomial_wilson_95_lower_min": 0.50,
                "minimum_strong_robustness_wilson_95_lower_min": 0.30
            },
        },
        "inference_scope": {
            "p_zero": "one_sided_enumeration_over_all_462_labels_with_ties_included",
            "p_margin": "one_sided_sharp_constant_additive_raw_probability_gain_shift_test",
            "design_exact_claim_allowed": False,
            "intention_to_treat_claim_allowed": False,
            "reason": "the_reporting_summary_documents_randomization_within_repeated_2_to_4_mouse_cohorts_and_four_postcollection_exclusions_but_the_release_lacks_block_rosters_within_block_allocations_and_exclusion_group_membership",
            "allowed_label": "unblocked_all_11_conditional_exchangeability_sensitivity_only",
            "weak_average_effect_null_claimed_exactly_tested": False,
        },
        "evidence_gate": gate,
        "selection_eligible_rule_count": sum(
            bool(item["selection_eligible"]) for item in selection_diagnostics.values()
        ),
        "independent_validation_eligible_rule_count": sum(
            bool(item["selection_eligible"]) for item in validation_diagnostics.values()
        ),
        "provisional_rule": rule_dict(provisional),
        "provisional_rule_independently_validated": validation_eligible,
        "diagnostic_rule": rule_dict(diagnostic),
        "diagnostic_rule_is_safety_eligible": bool(
            selection_diagnostics[diagnostic.rule_id]["safety_eligible"]
        ),
        "diagnostic_rule_selection_operating_characteristics": selection_diagnostics[
            diagnostic.rule_id
        ],
        "diagnostic_rule_validation_operating_characteristics": validation_diagnostics[
            diagnostic.rule_id
        ],
        "binding_rule": None,
        "binding_blockers": sorted(set(binding_blockers)),
        "scientist_signoff_alone_can_bind_a_current_rule": signoff_alone_can_bind,
        "audit_opening_authorized": False,
        "positive_mechanistic_terminal_authorized": False,
        "scientist_action_required": [
            (
                "do_not_sign_the_current_packet_because_no_rule_met_selection_and_independent_validation_eligibility"
                if not validation_eligible
                else "review_the_provisional_rule_without_treating_the_template_as_authority"
            ),
            "resolve_or_replace_each_failed_endpoint_receipt",
            "obtain_the_randomization_block_and_exclusion_roster_or_explicitly_limit_the_462_label_result_to_an_unblocked_exchangeability_sensitivity",
            "presign_absolute_raw_margin_arm_margins_and_development_scale_floor",
            "review_the_independent_validation_operating_characteristics",
            "sign_terminal_wording_and_a_separate_policy_amendment",
        ],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }

    output = arguments.output_directory
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output / "scenario_results.tsv", tsv(list(scenario_rows[0]), scenario_rows))
    atomic_write_text(output / "rule_grid.tsv", tsv(list(rule_rows[0]), rule_rows))
    atomic_write_text(output / "calibration.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    diagnostic_selection = selection_diagnostics[diagnostic.rule_id]
    diagnostic_validation = validation_diagnostics[diagnostic.rule_id]
    report = f"""# Endpoint-faithful small-`n` calibration

## Result

This run is synthetic, outcome-blind, and nonbinding. It models each mouse as
an early/late pair of integer binomial or beta-binomial counts with variable
eligible-trial denominators. It includes ties, overdispersion, near-boundary
probabilities, unequal denominator regimes, one-arm alternatives, a structural
failure/abstention regime, an absolute raw probability-gain margin, an
absolute development-scale floor, and independent selection/validation seeds.

- Candidate rules: `{len(rules)}`
- Scenarios per stream: `{len(SCENARIOS)}`
- Draws per scenario per stream: `{arguments.draws}`
- Selection-eligible rules: `{payload['selection_eligible_rule_count']}`
- Independently validation-eligible rules: `{payload['independent_validation_eligible_rule_count']}`
- Provisional rule: `{provisional.rule_id if provisional else 'none'}`
- Best diagnostic rule: `{diagnostic.rule_id}`
- Diagnostic selection safety-eligible: `{diagnostic_selection['safety_eligible']}`
- Diagnostic validation safety-eligible: `{diagnostic_validation['safety_eligible']}`
- Diagnostic strong-binomial support, selection / validation: `{diagnostic_selection['strong_binomial_support_rate']:.5f}` / `{diagnostic_validation['strong_binomial_support_rate']:.5f}` (required Wilson lower bound: `0.50`)
- Diagnostic minimum strong beta-binomial/denominator-robust support, selection / validation: `{diagnostic_selection['minimum_strong_robustness_support_rate']:.5f}` / `{diagnostic_validation['minimum_strong_robustness_support_rate']:.5f}` (required Wilson lower bound: `0.30`)
- Binding rule: **none**
- Audit opening authorized: **no**

Even a provisionally validated rule cannot bind while field/time evidence is
unresolved, the randomization mechanism is incomplete, and the scientist
signoff is unsigned. The engine itself is incapable of self-binding: once all
preconditions are met, a separately reviewed and signed policy amendment must
import the chosen rule and supporting evidence.
"""
    atomic_write_text(output / "CALIBRATION.md", report)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--receipt-directory", type=Path, required=True)
    parser.add_argument("--signoff", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--selection-seed", type=lambda value: int(value, 0), default=0xE2025A11)
    parser.add_argument("--validation-seed", type=lambda value: int(value, 0), default=0xE2025B22)
    arguments = parser.parse_args()
    run(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
