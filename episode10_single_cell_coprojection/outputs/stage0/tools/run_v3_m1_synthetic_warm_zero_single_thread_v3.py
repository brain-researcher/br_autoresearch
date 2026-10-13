#!/usr/bin/env python3
"""Run the frozen EP10 68-row M1 gate under the v3 one-thread backend."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(os.path.abspath(__file__))
STAGE0 = SCRIPT_PATH.parent.parent
ENGINE_PATH = SCRIPT_PATH.parent / "fit_v3_m1.py"
EXECUTION_VALIDATOR_PATH = (
    SCRIPT_PATH.parent / "validate_v3_m1_numerical_execution_v3.py"
)
CONTRACT_PATH = STAGE0 / "v3_m1_model_contract.yaml"
SYNTHETIC_DESIGN_PATH = STAGE0 / "v3_m1_synthetic_design_addendum_v1.yaml"
SYNTHETIC_RECOVERY_PATH = (
    STAGE0 / "v3_m1_synthetic_recovery_addendum_v2.yaml"
)
SOLVER_ADDENDUM_PATH = STAGE0 / "v3_m1_solver_addendum_v2.yaml"
EXECUTION_ADDENDUM_PATH = STAGE0 / "v3_m1_numerical_execution_addendum_v3.yaml"
SUMMARY_FILENAME = "v3_m1_synthetic_warm_zero_diagnostic.json"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import required module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


execution_policy = _load_module(
    "ep10_v3_m1_single_thread_execution_policy", EXECUTION_VALIDATOR_PATH
)
engine = _load_module("ep10_v3_m1_single_thread_qualification_engine", ENGINE_PATH)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument(
        "--synthetic-design-addendum",
        type=Path,
        default=SYNTHETIC_DESIGN_PATH,
    )
    parser.add_argument(
        "--synthetic-recovery-addendum",
        type=Path,
        default=SYNTHETIC_RECOVERY_PATH,
    )
    parser.add_argument(
        "--solver-addendum", type=Path, default=SOLVER_ADDENDUM_PATH
    )
    parser.add_argument(
        "--execution-addendum", type=Path, default=EXECUTION_ADDENDUM_PATH
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _attach_execution_witness(
    payload: dict[str, Any],
    addendum: Mapping[str, Any],
    backend: Mapping[str, Any],
) -> None:
    for key in (
        "real_target_calls_or_real_K_opened",
        "real_target_calls_or_real_K_or_membership_opened",
        "M0_prediction_or_fit_artifact_opened",
        "final_or_unassigned_or_SWC_opened",
    ):
        if key in payload and payload[key] is not False:
            raise execution_policy.ExecutionPolicyError(
                f"synthetic qualification seal changed: {key}"
            )
    payload["numerical_execution_addendum"] = EXECUTION_ADDENDUM_PATH.name
    payload["numerical_execution_schema_version"] = addendum["schema_version"]
    payload["numeric_backend"] = dict(backend)
    payload["post_freeze_single_thread_qualification"] = True
    payload["real_target_calls_or_real_K_or_membership_opened"] = False
    payload["M0_prediction_or_fit_artifact_opened"] = False
    payload["final_or_unassigned_or_SWC_opened"] = False


def _persist_existing_failure(
    output_root: Path,
    addendum: Mapping[str, Any],
    backend: Mapping[str, Any],
) -> None:
    summary_path = output_root / SUMMARY_FILENAME
    if not summary_path.is_file() or summary_path.is_symlink():
        return
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return
    _attach_execution_witness(payload, addendum, backend)
    engine._atomic_json(summary_path, payload)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        addendum = execution_policy.validate_addendum(args.execution_addendum)
        backend = execution_policy.numeric_backend()
    except Exception as error:
        print(
            f"ERROR: single-thread execution preflight failed closed ({error})",
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "stage": "single_thread_execution_preflight",
                "status": "pass",
                "numeric_backend": backend,
                "final_or_unassigned_or_SWC_opened": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    engine_args = argparse.Namespace(
        contract=args.contract,
        synthetic_design_addendum=args.synthetic_design_addendum,
        synthetic_recovery_addendum=args.synthetic_recovery_addendum,
        solver_addendum=args.solver_addendum,
        output_root=args.output_root,
        interface_root=None,
        geometry_root=None,
    )
    try:
        result = engine.run_synthetic_warm_zero_diagnostic(engine_args)
        _attach_execution_witness(result, addendum, backend)
        engine._atomic_json(Path(result["output_root"]) / SUMMARY_FILENAME, result)
    except Exception as error:
        try:
            _persist_existing_failure(args.output_root, addendum, backend)
        except Exception as persistence_error:
            print(
                "ERROR: could not attach execution witness to failed gate "
                f"({persistence_error})",
                file=sys.stderr,
            )
        print(f"ERROR: 68-row qualification failed ({error})", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "stage": result["stage"],
                "status": result["status"],
                "completed_fit_count": result["completed_fit_count"],
                "output_root": result["output_root"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
