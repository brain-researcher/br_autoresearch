#!/usr/bin/env python3
"""Diagnose EP10 V6 warm-path versus zero-start agreement, scalar only.

This diagnostic imports the frozen V6 solver without changing it, fits one
complete descending warm path and one complete zero-start path on the exact
153-cell/10-group development training split, and delegates comparisons to
the existing V2 comparison routine.  It emits no coefficients, predictions,
scores, target contrasts, cell IDs, or group IDs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(os.path.abspath(__file__))
if (
    not SCRIPT_PATH.is_file()
    or SCRIPT_PATH.is_symlink()
    or SCRIPT_PATH != SCRIPT_PATH.resolve(strict=True)
):
    raise ImportError("V6 warm-zero diagnostic must be an unsymlinked file")
SCRIPT_DIR = SCRIPT_PATH.parent
STAGE0_DIR = SCRIPT_DIR.parent
V6_ENGINE_PATH = SCRIPT_DIR / "fit_v3_m0_v6.py"
if (
    not V6_ENGINE_PATH.is_file()
    or V6_ENGINE_PATH.is_symlink()
    or V6_ENGINE_PATH != V6_ENGINE_PATH.resolve(strict=True)
):
    raise ImportError("V6 solver must be an unsymlinked stage0 file")

_V6_SPEC = importlib.util.spec_from_file_location(
    "ep10_v3_m0_v6_for_warm_zero_diagnostic", V6_ENGINE_PATH
)
if _V6_SPEC is None or _V6_SPEC.loader is None:
    raise ImportError(f"cannot import V6 solver: {V6_ENGINE_PATH}")
v6 = importlib.util.module_from_spec(_V6_SPEC)
sys.modules[_V6_SPEC.name] = v6
_V6_SPEC.loader.exec_module(v6)
base = v6.base
v2 = v6.v2
v3 = v6.v3


BASE_CONTRACT_PATH = STAGE0_DIR / "v3_m0_model_contract.yaml"
V6_ADDENDUM_PATH = STAGE0_DIR / "v3_m0_solver_addendum_v6.yaml"
OUTPUT_BASENAME_PREFIX = "v3_m0_solver_warm_zero_diagnostic_v6_job_"
OUTPUT_FILENAME = "v3_m0_solver_warm_zero_diagnostic_v6.json"

COMPARISON_KEYS = (
    "lambda",
    "objective_absolute_difference",
    "objective_absolute_tolerance",
    "coefficient_max_absolute_difference_report_only",
    "marginal_max_absolute_difference",
    "marginal_max_absolute_tolerance",
    "log_probability_max_absolute_difference",
    "log_probability_max_absolute_tolerance",
    "pass",
)


def _scalarize(value: Any) -> Any:
    """Keep only JSON scalar leaves and mappings; drop every vector-like value."""
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, np.generic):
        scalar = value.item()
        return _scalarize(scalar)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            scalarized = _scalarize(child)
            if scalarized is not _DROP:
                result[str(key)] = scalarized
        return result
    return _DROP


class _DropSentinel:
    pass


_DROP = _DropSentinel()


def _scalar_solver_telemetry(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    scalarized = _scalarize(telemetry)
    if not isinstance(scalarized, dict):
        raise base.ModelError("solver telemetry could not be scalarized")
    forbidden = {
        "accepted_step_values",
        "coefficients",
        "predictions",
        "marginals",
        "log_probability_nats",
        "eta",
        "target_contrasts",
    }
    if forbidden.intersection(scalarized):
        raise base.ModelError("vector-like solver telemetry escaped filtering")
    return scalarized


def _comparison_rows(
    comparisons: Sequence[Mapping[str, Any]],
) -> list[dict[str, float | bool]]:
    rows: list[dict[str, float | bool]] = []
    for comparison in comparisons:
        if any(key not in comparison for key in COMPARISON_KEYS):
            raise base.ModelError("warm-zero comparison lacks an expected scalar")
        row: dict[str, float | bool] = {}
        for key in COMPARISON_KEYS:
            value = comparison[key]
            if key == "pass":
                row[key] = bool(value)
            else:
                scalar = float(value)
                if not math.isfinite(scalar):
                    raise base.ModelError(
                        "warm-zero comparison contains a nonfinite scalar"
                    )
                row[key] = scalar
        rows.append(row)
    if tuple(row["lambda"] for row in rows) != v6.DESCENDING_LAMBDAS:
        raise base.ModelError("warm-zero diagnostic lambda order changed")
    return rows


def _prepare_output(path: Path, scratch: Path) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise base.ModelError("real warm-zero diagnostic must run in Slurm")
    episode_root = v3._require_unsymlinked_directory(
        scratch / base.EPISODE_SCRATCH_RELATIVE, "EP10 Scratch root"
    )
    expected_name = OUTPUT_BASENAME_PREFIX + job_id
    absolute = Path(os.path.abspath(path))
    if (
        not path.is_absolute()
        or absolute.parent != episode_root
        or absolute.name != expected_name
    ):
        raise base.ModelError(
            f"diagnostic output must be the exact EP10 Scratch child {expected_name!r}"
        )
    if os.path.lexists(absolute):
        raise base.ModelError("warm-zero diagnostic output root must be absent")
    absolute.mkdir(parents=False)
    if absolute.is_symlink() or absolute != absolute.resolve(strict=True):
        raise base.ModelError("warm-zero diagnostic output path is invalid")
    return absolute


def run_unit_test() -> dict[str, Any]:
    sample = {
        "status": "converged",
        "objective": 1.25,
        "accepted_step_values": [1.0, 0.5],
        "nested_summary": {
            "count": 2,
            "minimum": 0.5,
            "maximum": 1.0,
        },
        "coefficient_array": np.asarray([1.0, 2.0]),
    }
    filtered = _scalar_solver_telemetry(sample)
    if (
        "accepted_step_values" in filtered
        or "coefficient_array" in filtered
        or filtered.get("objective") != 1.25
        or filtered.get("nested_summary", {}).get("count") != 2
    ):
        raise AssertionError("scalar telemetry filter regression failed")

    example_rows = []
    for ridge_lambda in v6.DESCENDING_LAMBDAS:
        example_rows.append(
            {
                "lambda": ridge_lambda,
                "objective_absolute_difference": 0.0,
                "objective_absolute_tolerance": 1.0e-10,
                "coefficient_max_absolute_difference_report_only": 0.0,
                "marginal_max_absolute_difference": 0.0,
                "marginal_max_absolute_tolerance": 1.0e-5,
                "log_probability_max_absolute_difference": 0.0,
                "log_probability_max_absolute_tolerance": 1.0e-4,
                "pass": True,
            }
        )
    rows = _comparison_rows(example_rows)
    if len(rows) != len(v6.DESCENDING_LAMBDAS) or not all(
        row["pass"] for row in rows
    ):
        raise AssertionError("scalar comparison projection regression failed")
    return {
        "status": "pass",
        "stage": "v3_M0_solver_v6_warm_zero_diagnostic_unit_test",
        "lambda_rows_checked": len(rows),
        "vector_like_telemetry_dropped": True,
        "comparison_scalars_preserved": True,
        "uses_real_data": False,
        "final_or_unassigned_opened": False,
    }


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    v6.validate_solver_addendum(args.contract, args.solver_addendum)
    scratch, target_root, interface_root = v3._validate_real_input_roots(
        args.target_root, args.interface_root
    )
    output_root = _prepare_output(args.output_root, scratch)
    output_path = output_root / OUTPUT_FILENAME
    v6.FIT_TELEMETRY.clear()
    payload: dict[str, Any] = {
        "status": "fail",
        "stage": "v3_M0_solver_v6_warm_zero_scalar_diagnostic",
        "solver_addendum": args.solver_addendum.name,
        "held_out_scores_emitted": False,
        "coefficient_prediction_target_contrast_or_ID_vectors_emitted": False,
        "final_or_unassigned_opened": False,
        "output_root": str(output_root),
    }
    try:
        x, y, k_values, groups = v6._load_exact_probe_training(
            target_root, interface_root
        )
        warm = v2._fit_probe_path(x, y, k_values, groups, warm_start=True)
        zero = v2._fit_probe_path(x, y, k_values, groups, warm_start=False)

        # Reusing the warm path as both repeat arguments makes the retained
        # V2 routine compute its standard warm-zero metrics without launching
        # an unnecessary second warm fit.
        warm_self_checks, warm_zero_checks = v2._compare_probe_paths(
            warm, warm, zero
        )
        comparison_rows = _comparison_rows(warm_zero_checks)
        if not all(item.get("pass") is True for item in warm_self_checks):
            raise base.ModelError("warm self-comparison unexpectedly failed")

        warm_telemetry = [
            _scalar_solver_telemetry(warm[value][0].solver_telemetry)
            for value in v6.DESCENDING_LAMBDAS
        ]
        zero_telemetry = [
            _scalar_solver_telemetry(zero[value][0].solver_telemetry)
            for value in v6.DESCENDING_LAMBDAS
        ]
        failing_rows = [row for row in comparison_rows if not row["pass"]]
        payload.update(
            {
                "status": "diagnostic_complete",
                "training_cells": int(x.shape[0]),
                "training_group_count": int(len(set(groups))),
                "lambda_count": len(v6.DESCENDING_LAMBDAS),
                "per_lambda_warm_zero_comparison": comparison_rows,
                "all_warm_zero_comparisons_pass": not failing_rows,
                "failing_lambda_count": len(failing_rows),
                "warm_path_solver_scalar_telemetry": warm_telemetry,
                "zero_start_solver_scalar_telemetry": zero_telemetry,
                "warm_path_reused_for_repeat_argument": True,
            }
        )
        base._atomic_json(output_path, payload)
        return payload
    except (base.ModelError, AssertionError, OSError, ValueError) as error:
        payload.update(
            {
                "status": "fail",
                "failure_class": type(error).__name__,
                "completed_fit_solver_scalar_telemetry": [
                    _scalar_solver_telemetry(item)
                    for item in v6.FIT_TELEMETRY
                ],
            }
        )
        base._atomic_json(output_path, payload)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--unit-test", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--contract", type=Path, default=BASE_CONTRACT_PATH)
    parser.add_argument(
        "--solver-addendum", type=Path, default=V6_ADDENDUM_PATH
    )
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--interface-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.unit_test:
            v6.validate_solver_addendum(args.contract, args.solver_addendum)
            if any(
                value is not None
                for value in (
                    args.target_root,
                    args.interface_root,
                    args.output_root,
                )
            ):
                parser.error("unit test does not accept real paths")
            result = run_unit_test()
        else:
            missing = [
                name
                for name in ("target_root", "interface_root", "output_root")
                if getattr(args, name) is None
            ]
            if missing:
                parser.error(
                    "diagnostic run requires: "
                    + ", ".join(
                        "--" + value.replace("_", "-") for value in missing
                    )
                )
            result = run_diagnostic(args)
    except (base.ModelError, AssertionError, OSError, ValueError) as error:
        print(f"ERROR: {type(error).__name__}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "stage": result["stage"],
                "output_root": result.get("output_root"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
