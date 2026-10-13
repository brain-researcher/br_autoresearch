#!/usr/bin/env python3
"""Target-free fixture tests for the pre-M2 M1 adequacy gate."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "gate_v3_m2_m1_adequacy.py"
)
SPEC = importlib.util.spec_from_file_location("gate_v3_m2_m1_adequacy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


TARGETS = tuple(f"t{index}" for index in range(16))


def _default_marginals(shift: float) -> tuple[float, ...]:
    return tuple([0.25 + shift] * 8 + [0.25 - shift] * 8)


def _default_residuals(value: float) -> tuple[float, ...]:
    return tuple([value] * 8 + [-value] * 8)


def fixture(
    root: Path,
    *,
    gain: float = 0.2,
    brier: float = 0.1,
    marginals: Sequence[float] | None = None,
    residuals: Sequence[float] | None = None,
    per_group_residuals: Mapping[str, Sequence[float]] | None = None,
) -> None:
    marginal_values = tuple(
        _default_marginals(0.10) if marginals is None else marginals
    )
    residual_values = tuple(
        _default_residuals(0.02) if residuals is None else residuals
    )
    if len(marginal_values) != 16 or len(residual_values) != 16:
        raise AssertionError("fixture target vector changed")
    per_group = {
        group: {
            "cells": count,
            "marginal_Brier_M1": brier,
            "observed_minus_expected_by_target": {
                target: float(
                    (per_group_residuals or {}).get(group, residual_values)[index]
                )
                for index, target in enumerate(TARGETS)
            },
        }
        for group, count in gate.GROUP_SIZES.items()
    }
    summary = {
        "status": "pass",
        "development_cells": 194,
        "development_groups": 12,
        "target_panel_size": 16,
        "SWC_opened": False,
        "final_or_unassigned_opened": False,
        "M0_prediction_or_fit_artifact_opened": False,
        "scores_and_calibration": {
            "aggregate_equal_group_paired_M1_minus_uniform_gain_bits_per_cell": gain,
            "per_group": per_group,
        },
    }
    (root / gate.SUMMARY).write_text(json.dumps(summary), encoding="utf-8")
    marginal_fields = [f"exact_inclusion_marginal__{target}" for target in TARGETS]
    fields = ["provider_neuron_id", "fMOST_brain_id", "K", *marginal_fields]
    with (root / gate.PREDICTIONS).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        ordinal = 0
        for group, count in gate.GROUP_SIZES.items():
            for _ in range(count):
                ordinal += 1
                row = {
                    "provider_neuron_id": f"cell_{ordinal}",
                    "fMOST_brain_id": group,
                    "K": 4,
                }
                row.update(
                    {
                        field: marginal_values[index]
                        for index, field in enumerate(marginal_fields)
                    }
                )
                writer.writerow(row)


def test_pass() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture(root)
        result = gate.evaluate(root)
        assert result["status"] == "pass"
        assert result["development_cells"] == 194
        assert math.isclose(result["equal_group_marginal_Brier_M1"], 0.1)
        assert math.isclose(
            result["equal_group_marginal_Brier_fixed_K_uniform"], 0.1875
        )
        calibration = result["target_calibration_in_the_large"]
        assert math.isclose(calibration["M1_target_RMS"], 0.02)
        assert math.isclose(calibration["fixed_K_uniform_target_RMS"], 0.12)
        assert math.isclose(
            calibration["per_group_inputs"]["212074"]["targets"]["t0"][
                "reconstructed_mean_observed_indicator"
            ],
            0.37,
        )
        assert calibration["cell_identities_emitted"] is False
        serialized = json.dumps(result, sort_keys=True)
        assert "provider_neuron_id" not in serialized
        assert all(
            json.dumps(f"cell_{ordinal}") not in serialized
            for ordinal in range(1, 195)
        )
        assert result["real_M2_outcomes_opened"] is False


def test_nonpositive_log_gain_fails() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture(root, gain=0.0)
        try:
            gate.evaluate(root)
        except gate.GateError:
            return
        raise AssertionError("nonpositive M1 log-score gain passed")


def test_worse_brier_fails() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture(root, brier=0.2)
        try:
            gate.evaluate(root)
        except gate.GateError:
            return
        raise AssertionError("worse M1 marginal Brier passed")


def test_sum_K_failure_fails() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture(root)
        path = root / gate.PREDICTIONS
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = list(reader.fieldnames or ())
        rows[0][f"exact_inclusion_marginal__{TARGETS[0]}"] = "0.34"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        try:
            gate.evaluate(root)
        except gate.GateError:
            return
        raise AssertionError("marginals not summing to K passed")


def test_target_RMS_absolute_ceiling_fails() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture(root, marginals=(0.25,) * 16, residuals=_default_residuals(0.16))
        try:
            gate.evaluate(root)
        except gate.GateError as error:
            assert "RMS exceeds 0.15" in str(error)
            return
        raise AssertionError("target RMS above 0.15 passed")


def test_target_maximum_absolute_ceiling_fails() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        marginals = (0.25, 0.40, *([3.35 / 14.0] * 14))
        residuals = (0.31, -0.31, *([0.0] * 14))
        fixture(root, marginals=marginals, residuals=residuals)
        try:
            gate.evaluate(root)
        except gate.GateError as error:
            assert "maximum absolute target residual exceeds 0.30" in str(error)
            return
        raise AssertionError("target maximum absolute residual above 0.30 passed")


def test_uniform_relative_improvement_failure_fails() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture(
            root,
            marginals=_default_marginals(0.004),
            residuals=_default_residuals(0.02),
        )
        try:
            gate.evaluate(root)
        except gate.GateError as error:
            assert "does not improve 20% over uniform" in str(error)
            return
        raise AssertionError("less than 20% calibration improvement passed")


def test_uniform_maximum_relative_improvement_failure_fails() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        residuals = (0.10, *([-0.10 / 15.0] * 15))
        uniform_residuals = tuple([0.05] * 8 + [-0.05] * 8)
        marginals = tuple(
            0.25 + uniform_residual - residual
            for uniform_residual, residual in zip(
                uniform_residuals, residuals, strict=True
            )
        )
        fixture(root, marginals=marginals, residuals=residuals)
        try:
            gate.evaluate(root)
        except gate.GateError as error:
            assert (
                "maximum absolute target residual does not improve 20% over uniform"
                in str(error)
            )
            return
        raise AssertionError("insufficient maximum-residual improvement passed")


def test_equal_group_target_aggregation_not_cell_weighted() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        zero = (0.0,) * 16
        tiny_group = (0.06, *([0.0] * 15))
        fixture(
            root,
            residuals=zero,
            per_group_residuals={"221539": tiny_group},
        )
        result = gate.evaluate(root)
        observed = result["target_calibration_in_the_large"][
            "M1_equal_group_residual_by_target"
        ]["t0"]
        assert math.isclose(observed, 0.06 / 12.0, abs_tol=1.0e-15)
        assert not math.isclose(observed, 0.06 / 194.0, abs_tol=1.0e-6)


def test_summary_target_inventory_mismatch_fails() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture(root)
        path = root / gate.SUMMARY
        summary = json.loads(path.read_text(encoding="utf-8"))
        residuals = summary["scores_and_calibration"]["per_group"]["212074"][
            "observed_minus_expected_by_target"
        ]
        residuals["unexpected_target"] = residuals.pop("t15")
        path.write_text(json.dumps(summary), encoding="utf-8")
        try:
            gate.evaluate(root)
        except gate.GateError:
            return
        raise AssertionError("mismatched summary target inventory passed")


def test_reconstructed_observed_mean_out_of_bounds_fails() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        residuals = list(_default_residuals(0.02))
        residuals[0] = 0.70
        fixture(root, residuals=residuals)
        try:
            gate.evaluate(root)
        except gate.GateError as error:
            assert "outside probability bounds" in str(error)
            return
        raise AssertionError("impossible reconstructed observed prevalence passed")


def run() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"M1 adequacy gate tests: PASS ({len(tests)}/{len(tests)})")


if __name__ == "__main__":
    run()
