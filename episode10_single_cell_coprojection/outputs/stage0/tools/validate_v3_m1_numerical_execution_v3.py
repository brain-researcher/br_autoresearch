#!/usr/bin/env python3
"""Validate the frozen EP10 M1 single-thread execution policy and runtime."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


SCRIPT_PATH = Path(os.path.abspath(__file__))
STAGE0 = SCRIPT_PATH.parent.parent
ADDENDUM_PATH = STAGE0 / "v3_m1_numerical_execution_addendum_v3.yaml"


class ExecutionPolicyError(RuntimeError):
    """The frozen single-thread execution policy or runtime changed."""


def validate_addendum(path: Path = ADDENDUM_PATH) -> Mapping[str, Any]:
    absolute = Path(os.path.abspath(path))
    if (
        absolute != ADDENDUM_PATH
        or not absolute.is_file()
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != ADDENDUM_PATH.resolve(strict=True)
    ):
        raise ExecutionPolicyError(
            "numerical execution addendum must be the exact stage0 file"
        )
    with absolute.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ExecutionPolicyError("numerical execution addendum is not a mapping")

    authority = payload.get("authority", {})
    trigger = payload.get("trigger", {})
    execution = payload.get("sole_execution_change", {})
    threads = execution.get("numerical_thread_environment", {})
    checkpoint = payload.get("checkpoint_policy", {})
    qualification = payload.get("required_qualification", {})
    resources = payload.get("resource_reconciliation", {})
    production = payload.get("production_boundary", {})
    boundary = payload.get("implementation_boundary", {})
    unchanged_flags = (
        "changes_model_estimand_objective_gradient_or_Hessian",
        "changes_Newton_direction_formula_or_factorization",
        "changes_Wolfe_constants_steps_caps_or_inequalities",
        "changes_convergence_or_certificate_thresholds",
        "changes_lambda_grid_initialization_fold_order_or_selection_rule",
        "adds_retry_acceptance_slack_tie_rule_or_fallback",
    )
    if not (
        payload.get("schema_version")
        == "episode10_v3_m1_numerical_execution_addendum_v3"
        and payload.get("status")
        == "frozen_before_single_thread_qualification_and_clean_production"
        and payload.get("scientific_authority") is False
        and authority.get("base_contract", {}).get("required_schema_version")
        == "episode10_v3_m1_model_v1"
        and authority.get("stable_line_solver_addendum", {}).get(
            "required_schema_version"
        )
        == "episode10_v3_m1_solver_addendum_v1"
        and authority.get("confirmation_solver_addendum", {}).get(
            "required_schema_version"
        )
        == "episode10_v3_m1_solver_addendum_v2"
        and str(trigger.get("failed_production_job")) == "45609750"
        and str(trigger.get("four_thread_original_backend_diagnostic_job"))
        == "45619301"
        and str(trigger.get("single_thread_diagnostic_job")) == "45620682"
        and int(trigger.get("single_thread_completed_fits", -1)) == 24
        and float(
            trigger.get(
                "single_thread_repeat_coefficient_maximum_absolute_difference",
                float("nan"),
            )
        )
        == 0.0
        and execution.get("scope")
        == "every_M1_qualification_and_production_optimizer_or_evaluator_call"
        and execution.get("scheduler_constraint") == "CPU_SKU:7502"
        and int(execution.get("requested_cpus_per_task", -1)) == 1
        and threads
        == {
            "OMP_NUM_THREADS": 1,
            "OPENBLAS_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1,
            "NUMEXPR_NUM_THREADS": 1,
        }
        and int(execution.get("effective_OpenBLAS_threads_required", -1)) == 1
        and execution.get("dispatched_OpenBLAS_core_required") == "Zen"
        and execution.get("dispatched_CPU_model_must_contain")
        == "AMD EPYC 7502"
        and all(execution.get(name) is False for name in unchanged_flags)
        and checkpoint.get("four_thread_job_45609750_checkpoints_reusable")
        is False
        and checkpoint.get("clean_job_id_and_output_root_required") is True
        and checkpoint.get("copy_link_or_read_old_checkpoint_coefficients")
        == "forbidden"
        and qualification.get("data") == "deterministic_synthetic_only"
        and qualification.get("scheduler_constraint") == "CPU_SKU:7502"
        and int(qualification.get("effective_OpenBLAS_threads", -1)) == 1
        and int(qualification.get("seed", -1)) == 20260928
        and int(qualification.get("required_fits", -1)) == 36
        and float(qualification.get("repeat_objective_absolute_tolerance"))
        == 1.0e-12
        and float(
            qualification.get("repeat_coefficient_max_absolute_tolerance")
        )
        == 1.0e-10
        and float(
            qualification.get("repeat_probability_max_absolute_tolerance")
        )
        == 1.0e-10
        and qualification.get("pass_authorizes_only")
        == "clean_single_thread_development_M1_production_fit"
        and qualification.get(
            "real_target_calls_or_real_K_or_membership_opened"
        )
        is False
        and qualification.get("M0_prediction_or_fit_artifact_opened") is False
        and qualification.get("final_or_unassigned_or_SWC_opened") is False
        and resources.get("authorization_basis")
        == "frozen_conservative_path_equivalents"
        and int(resources.get("conservative_path_equivalents", -1)) == 157
        and int(
            resources.get(
                "actual_executed_path_equivalents_with_exact_fold_reuse", -1
            )
        )
        == 91
        and float(resources.get("projected_seconds_for_157_path_equivalents"))
        == 16485.0
        and float(resources.get("frozen_eight_hour_request_80_percent_seconds"))
        == 23040.0
        and resources.get("projected_time_below_80_percent_of_request") is True
        and production.get("scheduler_constraint") == "CPU_SKU:7502"
        and int(production.get("requested_cpus_per_task", -1)) == 1
        and int(production.get("effective_OpenBLAS_threads", -1)) == 1
        and production.get("new_Slurm_job_id_and_clean_job_scoped_output_root")
        == "required"
        and boundary.get("next_qualification_output_basename")
        == "v3_m1_synthetic_warm_zero_diagnostic_v2_job_JOBID"
    ):
        raise ExecutionPolicyError("numerical execution addendum identity changed")
    return payload


def _openblas_runtime() -> Mapping[str, Any]:
    libraries = sorted(
        (Path(np.__file__).resolve().parent.parent / "numpy.libs").glob(
            "libopenblas*.so"
        )
    )
    if len(libraries) != 1:
        raise ExecutionPolicyError(
            f"expected exactly one NumPy OpenBLAS library, found {len(libraries)}"
        )
    blas = ctypes.CDLL(str(libraries[0]))

    def call_text(*names: str) -> str:
        for name in names:
            function = getattr(blas, name, None)
            if function is not None:
                function.restype = ctypes.c_char_p
                return function().decode("ascii", "strict")
        raise ExecutionPolicyError(f"missing OpenBLAS symbol: {names}")

    def call_int(*names: str) -> int:
        for name in names:
            function = getattr(blas, name, None)
            if function is not None:
                function.restype = ctypes.c_int
                return int(function())
        raise ExecutionPolicyError(f"missing OpenBLAS symbol: {names}")

    return {
        "library": libraries[0].name,
        "config": call_text("openblas_get_config64_", "openblas_get_config"),
        "core": call_text("openblas_get_corename64_", "openblas_get_corename"),
        "threads": call_int(
            "openblas_get_num_threads64_",
            "openblas_get_num_threads_64_",
            "openblas_get_num_threads",
        ),
    }


def _cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    raise ExecutionPolicyError("CPU model name is unavailable")


def numeric_backend() -> Mapping[str, Any]:
    for name in (
        "OPENBLAS_CORETYPE",
        "NPY_DISABLE_CPU_FEATURES",
        "NPY_ENABLE_CPU_FEATURES",
    ):
        if name in os.environ:
            raise ExecutionPolicyError(f"forbidden numerical override is set: {name}")
    expected_environment = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    observed_environment = {
        name: os.environ.get(name) for name in expected_environment
    }
    if observed_environment != expected_environment:
        raise ExecutionPolicyError("single-thread numerical environment changed")

    openblas = _openblas_runtime()
    effective_threads = int(openblas["threads"])
    if effective_threads != 1:
        raise ExecutionPolicyError("effective OpenBLAS thread count is not one")
    if openblas["core"] != "Zen":
        raise ExecutionPolicyError("dispatched OpenBLAS core is not Zen")
    cpu_model = _cpu_model()
    if "AMD EPYC 7502" not in cpu_model:
        raise ExecutionPolicyError("dispatched CPU is not AMD EPYC 7502")
    return {
        "hostname": socket.gethostname(),
        "cpu_model": cpu_model,
        "numpy": np.__version__,
        "openblas_library": openblas["library"],
        "openblas_config": openblas["config"],
        "openblas_core": openblas["core"],
        "openblas_threads": effective_threads,
        "thread_environment": observed_environment,
        "SLURM_job_id": os.environ.get("SLURM_JOB_ID"),
        "SLURM_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "SLURM_tres_per_task": os.environ.get("SLURM_TRES_PER_TASK"),
        "SLURM_job_cpus_per_node": os.environ.get("SLURM_JOB_CPUS_PER_NODE"),
        "SLURM_job_constraint": os.environ.get("SLURM_JOB_CONSTRAINTS"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addendum", type=Path, default=ADDENDUM_PATH)
    parser.add_argument("--runtime", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        addendum = validate_addendum(args.addendum)
        backend = numeric_backend() if args.runtime else None
    except Exception as error:
        print(f"ERROR: M1 execution policy failed closed ({error})", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "pass",
                "schema_version": addendum["schema_version"],
                "numeric_backend": backend,
                "M0_prediction_or_fit_artifact_opened": False,
                "final_or_unassigned_or_SWC_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
