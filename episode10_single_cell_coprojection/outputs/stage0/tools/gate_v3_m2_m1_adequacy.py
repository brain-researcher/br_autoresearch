#!/usr/bin/env python3
"""Verify the frozen pre-M2 M1 adequacy gate from completed M1 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


SUMMARY = "v3_m1_fit_summary.json"
PREDICTIONS = "v3_m1_cross_fitted_predictions.csv"
MARGINAL_PREFIX = "exact_inclusion_marginal__"
TARGET_CALIBRATION_RMS_MAXIMUM = 0.15
TARGET_CALIBRATION_MAX_ABSOLUTE_MAXIMUM = 0.30
UNIFORM_METRIC_RATIO_MAXIMUM = 0.80
GROUP_SIZES = {
    "212074": 6,
    "220187": 35,
    "220188": 25,
    "220190": 16,
    "221091": 33,
    "221225": 39,
    "221418": 12,
    "221514": 5,
    "221539": 1,
    "221588": 7,
    "221751": 13,
    "233281": 2,
}


class GateError(RuntimeError):
    """The prospective M1 adequacy gate failed closed."""


def _regular_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise GateError(f"required regular non-symlink file is unavailable: {path}")


def _load_json(path: Path) -> Mapping[str, Any]:
    _regular_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateError(f"JSON root is not an object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise GateError(f"{label} is not numeric") from error
    if not math.isfinite(number):
        raise GateError(f"{label} is nonfinite")
    return number


def _rms(values: Sequence[float]) -> float:
    if not values:
        raise GateError("target calibration RMS requires values")
    return math.sqrt(math.fsum(value * value for value in values) / len(values))


def evaluate(m1_root: Path) -> Mapping[str, Any]:
    if not m1_root.is_dir() or m1_root.is_symlink():
        raise GateError("M1 root must be a real directory")
    summary = _load_json(m1_root / SUMMARY)
    prediction_path = m1_root / PREDICTIONS
    _regular_file(prediction_path)
    required_summary = {
        "status": "pass",
        "development_cells": 194,
        "development_groups": 12,
        "target_panel_size": 16,
        "SWC_opened": False,
        "final_or_unassigned_opened": False,
        "M0_prediction_or_fit_artifact_opened": False,
    }
    for key, expected in required_summary.items():
        if summary.get(key) != expected:
            raise GateError(f"M1 summary {key!r} changed from {expected!r}")

    scores = summary.get("scores_and_calibration")
    if not isinstance(scores, dict):
        raise GateError("M1 scores_and_calibration is absent")
    log_gain = _finite(
        scores.get("aggregate_equal_group_paired_M1_minus_uniform_gain_bits_per_cell"),
        "equal-group M1-minus-uniform log-score gain",
    )
    if not log_gain > 0.0:
        raise GateError("M1 does not beat the fixed-K uniform realized-set log score")
    per_group = scores.get("per_group")
    if not isinstance(per_group, dict) or set(per_group) != set(GROUP_SIZES):
        raise GateError("M1 per-group calibration inventory changed")

    with prediction_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = tuple(reader.fieldnames or ())
    marginal_fields = tuple(field for field in fields if field.startswith(MARGINAL_PREFIX))
    targets = tuple(field[len(MARGINAL_PREFIX) :] for field in marginal_fields)
    if (
        len(rows) != 194
        or len(marginal_fields) != 16
        or any(not target for target in targets)
        or len(set(targets)) != 16
    ):
        raise GateError("M1 prediction dimensions changed")
    identifiers: set[str] = set()
    group_counts: Counter[str] = Counter()
    uniform_brier_by_group: dict[str, list[float]] = {
        group: [] for group in GROUP_SIZES
    }
    K_sum_by_group = {group: 0 for group in GROUP_SIZES}
    marginal_sum_by_group = {
        group: {target: 0.0 for target in targets} for group in GROUP_SIZES
    }
    maximum_sum_K_error = 0.0
    for row in rows:
        identifier = str(row.get("provider_neuron_id", ""))
        group = str(row.get("fMOST_brain_id", ""))
        if not identifier or identifier in identifiers or group not in GROUP_SIZES:
            raise GateError("M1 prediction identity/group inventory is malformed")
        identifiers.add(identifier)
        group_counts[group] += 1
        try:
            k = int(str(row.get("K", "")))
        except ValueError as error:
            raise GateError("M1 prediction K is not an integer") from error
        if not 0 <= k <= 16:
            raise GateError("M1 prediction K is outside 0..16")
        marginals = [_finite(row.get(field), field) for field in marginal_fields]
        if any(value < -1.0e-12 or value > 1.0 + 1.0e-12 for value in marginals):
            raise GateError("M1 inclusion marginal is outside probability bounds")
        maximum_sum_K_error = max(maximum_sum_K_error, abs(math.fsum(marginals) - k))
        K_sum_by_group[group] += k
        for target, marginal in zip(targets, marginals, strict=True):
            marginal_sum_by_group[group][target] += marginal
        p = k / 16.0
        uniform_brier_by_group[group].append(p * (1.0 - p))
    if dict(group_counts) != GROUP_SIZES:
        raise GateError("M1 prediction group-size vector changed")
    if maximum_sum_K_error > 1.0e-10:
        raise GateError("M1 inclusion marginals do not sum to K")

    m1_group_brier: dict[str, float] = {}
    uniform_group_brier: dict[str, float] = {}
    per_group_calibration_inputs: dict[str, Mapping[str, Any]] = {}
    for group in GROUP_SIZES:
        record = per_group[group]
        if not isinstance(record, dict) or record.get("cells") != GROUP_SIZES[group]:
            raise GateError(f"M1 calibration record for group {group} changed")
        m1_group_brier[group] = _finite(record.get("marginal_Brier_M1"), "M1 Brier")
        uniform_group_brier[group] = math.fsum(uniform_brier_by_group[group]) / GROUP_SIZES[group]
        residuals = record.get("observed_minus_expected_by_target")
        if not isinstance(residuals, Mapping) or set(residuals) != set(targets):
            raise GateError(f"M1 target residual inventory for group {group} changed")
        mean_K = K_sum_by_group[group] / GROUP_SIZES[group]
        uniform_marginal = mean_K / 16.0
        target_inputs: dict[str, Mapping[str, float]] = {}
        for target in targets:
            mean_m1_marginal = (
                marginal_sum_by_group[group][target] / GROUP_SIZES[group]
            )
            m1_residual = _finite(
                residuals.get(target),
                f"M1 observed-minus-expected residual for group {group} target {target}",
            )
            observed_mean = mean_m1_marginal + m1_residual
            if observed_mean < -1.0e-10 or observed_mean > 1.0 + 1.0e-10:
                raise GateError(
                    f"reconstructed observed mean for group {group} target {target} "
                    "is outside probability bounds"
                )
            target_inputs[target] = {
                "mean_M1_exact_inclusion_marginal": mean_m1_marginal,
                "M1_observed_minus_expected_residual": m1_residual,
                "reconstructed_mean_observed_indicator": observed_mean,
                "fixed_K_uniform_observed_minus_expected_residual": (
                    observed_mean - uniform_marginal
                ),
            }
        per_group_calibration_inputs[group] = {
            "cells": GROUP_SIZES[group],
            "mean_K": mean_K,
            "mean_fixed_K_uniform_marginal": uniform_marginal,
            "targets": target_inputs,
        }
    m1_brier = math.fsum(m1_group_brier.values()) / 12.0
    uniform_brier = math.fsum(uniform_group_brier.values()) / 12.0
    if not m1_brier < uniform_brier:
        raise GateError("M1 marginal Brier does not beat the fixed-K uniform baseline")

    m1_target_residuals: dict[str, float] = {}
    uniform_target_residuals: dict[str, float] = {}
    for target in targets:
        m1_target_residuals[target] = math.fsum(
            float(
                per_group_calibration_inputs[group]["targets"][target][
                    "M1_observed_minus_expected_residual"
                ]
            )
            for group in GROUP_SIZES
        ) / len(GROUP_SIZES)
        uniform_target_residuals[target] = math.fsum(
            float(
                per_group_calibration_inputs[group]["targets"][target][
                    "fixed_K_uniform_observed_minus_expected_residual"
                ]
            )
            for group in GROUP_SIZES
        ) / len(GROUP_SIZES)
    m1_target_rms = _rms(tuple(m1_target_residuals.values()))
    uniform_target_rms = _rms(tuple(uniform_target_residuals.values()))
    m1_target_maximum = max(abs(value) for value in m1_target_residuals.values())
    uniform_target_maximum = max(
        abs(value) for value in uniform_target_residuals.values()
    )
    if m1_target_rms > TARGET_CALIBRATION_RMS_MAXIMUM:
        raise GateError("M1 equal-group target calibration RMS exceeds 0.15")
    if m1_target_maximum > TARGET_CALIBRATION_MAX_ABSOLUTE_MAXIMUM:
        raise GateError("M1 equal-group maximum absolute target residual exceeds 0.30")
    if m1_target_rms > UNIFORM_METRIC_RATIO_MAXIMUM * uniform_target_rms:
        raise GateError("M1 target calibration RMS does not improve 20% over uniform")
    if m1_target_maximum > UNIFORM_METRIC_RATIO_MAXIMUM * uniform_target_maximum:
        raise GateError(
            "M1 maximum absolute target residual does not improve 20% over uniform"
        )

    return {
        "status": "pass",
        "gate": "v3_M2_preaccess_M1_adequacy",
        "development_cells": 194,
        "development_groups": 12,
        "target_panel_size": 16,
        "equal_group_M1_minus_uniform_log_score_gain_bits_per_cell": log_gain,
        "equal_group_marginal_Brier_M1": m1_brier,
        "equal_group_marginal_Brier_fixed_K_uniform": uniform_brier,
        "marginal_Brier_improvement": uniform_brier - m1_brier,
        "maximum_absolute_marginal_sum_K_error": maximum_sum_K_error,
        "per_group_M1_marginal_Brier": m1_group_brier,
        "per_group_fixed_K_uniform_marginal_Brier": uniform_group_brier,
        "target_calibration_in_the_large": {
            "definition": (
                "for each target, one_twelfth_sum_of_group_mean_observed_minus_"
                "expected_residuals"
            ),
            "ordered_targets": list(targets),
            "thresholds": {
                "M1_target_RMS_maximum": TARGET_CALIBRATION_RMS_MAXIMUM,
                "M1_target_maximum_absolute_residual_maximum": (
                    TARGET_CALIBRATION_MAX_ABSOLUTE_MAXIMUM
                ),
                "M1_to_fixed_K_uniform_metric_ratio_maximum": (
                    UNIFORM_METRIC_RATIO_MAXIMUM
                ),
            },
            "M1_equal_group_residual_by_target": m1_target_residuals,
            "fixed_K_uniform_equal_group_residual_by_target": (
                uniform_target_residuals
            ),
            "M1_target_RMS": m1_target_rms,
            "fixed_K_uniform_target_RMS": uniform_target_rms,
            "M1_target_maximum_absolute_residual": m1_target_maximum,
            "fixed_K_uniform_target_maximum_absolute_residual": (
                uniform_target_maximum
            ),
            "M1_to_fixed_K_uniform_RMS_ratio": (
                m1_target_rms / uniform_target_rms
                if uniform_target_rms > 0.0
                else (0.0 if m1_target_rms == 0.0 else None)
            ),
            "M1_to_fixed_K_uniform_maximum_absolute_ratio": (
                m1_target_maximum / uniform_target_maximum
                if uniform_target_maximum > 0.0
                else (0.0 if m1_target_maximum == 0.0 else None)
            ),
            "per_group_inputs": per_group_calibration_inputs,
            "cell_identities_emitted": False,
        },
        "M1_summary_and_outer_predictions_opened": True,
        "real_M2_outcomes_opened": False,
        "M0_full_refit_or_telemetry_opened": False,
        "SWC_opened": False,
        "final_or_unassigned_opened": False,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(args.m1_root)
    _atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
