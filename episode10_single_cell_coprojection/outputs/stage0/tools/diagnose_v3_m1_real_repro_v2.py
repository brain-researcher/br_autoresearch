#!/usr/bin/env python3
"""Repeat the failed EP10 real-M1 path twice on the original numerical backend."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(os.path.abspath(__file__))
V1_PATH = SCRIPT_PATH.parent / "diagnose_v3_m1_real_failure_v1.py"

SPEC = importlib.util.spec_from_file_location(
    "ep10_fit_v3_m1_real_repro_v2_dependency", V1_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import real-M1 diagnostic: {V1_PATH}")
v1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v1
SPEC.loader.exec_module(v1)
runner = v1.runner
m1 = v1.m1


OUTPUT_PREFIX = "v3_m1_real_warm_repro_v2"
SUMMARY_FILENAME = "v3_m1_real_warm_repro_v2.json"
STAGE = "v3_M1_real_warm_path_original_backend_repro_v2"
REPEAT_COUNT = 2
EXPECTED_OPENBLAS_THREADS = 4
OBJECTIVE_TOLERANCE = 1.0e-12
COEFFICIENT_TOLERANCE = 1.0e-10
PROBABILITY_TOLERANCE = 1.0e-10
EXPECTED_FAILURE_STATUS = "pass_diagnostic_trigger_single_thread"
EXPECTED_FAILURE_DECISION = "trigger_single_thread_arm"
EXPECTED_FAILURE_REASON_BOTH = "original_failure_reproduced_in_both_repeats"
EXPECTED_FAILURE_REASON_ONE = "original_failure_reproduced_in_one_of_two_repeats"
DISAGREEMENT_STATUS = "pass_diagnostic_trigger_single_thread"
DISAGREEMENT_DECISION = "trigger_single_thread_arm"
DISAGREEMENT_REASON = "four_thread_repeat_disagreement"
AGREEMENT_STATUS = "pass_two_repeat_nonreproduction"
AGREEMENT_DECISION = "bounded_nonreproduction_at_four_threads"
AGREEMENT_REASON = "both_paths_and_all_repeat_tolerances_passed"


class ReproDiagnosticError(v1.DiagnosticError):
    """The original-backend repeat diagnostic failed closed."""


def _prepare_output_root(path: Path) -> Path:
    scratch_value = os.environ.get("SCRATCH")
    job_id = os.environ.get("SLURM_JOB_ID")
    if not scratch_value or not job_id:
        raise ReproDiagnosticError("repro diagnostic requires SCRATCH and SLURM_JOB_ID")
    scratch = Path(scratch_value).resolve(strict=True)
    episode = (
        scratch / "br_autoresearch" / "episode10_single_cell_coprojection"
    ).resolve(strict=True)
    absolute = Path(os.path.abspath(path))
    if (
        absolute.parent != episode
        or absolute.name != f"{OUTPUT_PREFIX}_job_{job_id}"
        or absolute.is_symlink()
    ):
        raise ReproDiagnosticError(
            "repro output is not the exact job-scoped Scratch child"
        )
    try:
        absolute.mkdir(mode=0o770)
    except FileExistsError as error:
        raise ReproDiagnosticError("repro diagnostic output already exists") from error
    return absolute


def _numeric_backend() -> Mapping[str, Any]:
    np = runner.np
    libraries = sorted(
        (Path(np.__file__).resolve().parent.parent / "numpy.libs").glob(
            "libopenblas*.so"
        )
    )
    if len(libraries) != 1:
        raise ReproDiagnosticError(
            f"expected exactly one NumPy OpenBLAS library, found {len(libraries)}"
        )
    blas = ctypes.CDLL(str(libraries[0]))

    def call_text(*names: str) -> str:
        for name in names:
            function = getattr(blas, name, None)
            if function is not None:
                function.restype = ctypes.c_char_p
                return function().decode("ascii", "strict")
        raise ReproDiagnosticError(f"missing OpenBLAS symbol: {names}")

    def call_int(*names: str) -> int:
        for name in names:
            function = getattr(blas, name, None)
            if function is not None:
                function.restype = ctypes.c_int
                return int(function())
        raise ReproDiagnosticError(f"missing OpenBLAS symbol: {names}")

    thread_environment = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    effective_threads = call_int(
        "openblas_get_num_threads64_",
        "openblas_get_num_threads_64_",
        "openblas_get_num_threads",
    )
    slurm_cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", "-1"))
    expected_thread_environment = {
        "OMP_NUM_THREADS": str(EXPECTED_OPENBLAS_THREADS),
        "OPENBLAS_NUM_THREADS": str(EXPECTED_OPENBLAS_THREADS),
        "MKL_NUM_THREADS": str(EXPECTED_OPENBLAS_THREADS),
        "NUMEXPR_NUM_THREADS": "1",
    }
    if (
        effective_threads != EXPECTED_OPENBLAS_THREADS
        or thread_environment != expected_thread_environment
    ):
        raise ReproDiagnosticError("requested numerical backend thread count changed")
    return {
        "hostname": socket.gethostname(),
        "numpy": np.__version__,
        "openblas_library": libraries[0].name,
        "openblas_config": call_text(
            "openblas_get_config64_", "openblas_get_config"
        ),
        "openblas_core": call_text(
            "openblas_get_corename64_", "openblas_get_corename"
        ),
        "openblas_threads": effective_threads,
        "SLURM_cpus_per_task": slurm_cpus_per_task,
        "SLURM_tres_per_task": os.environ.get("SLURM_TRES_PER_TASK"),
        "SLURM_job_cpus_per_node": os.environ.get("SLURM_JOB_CPUS_PER_NODE"),
        "thread_environment": thread_environment,
    }


def _prepare_failed_training(args: argparse.Namespace) -> tuple[Any, ...]:
    target_blind_cells, pairs = m1.load_target_free_real_shape(
        args.interface_root, args.geometry_root
    )
    expected_ids = {cell.identifier for cell in target_blind_cells}
    membership, counts = runner._read_membership_inputs(args.target_root, expected_ids)
    library = m1.CandidateLibrary()
    cells = runner._align_membership(target_blind_cells, membership, library)
    del membership
    geometry = m1.GeometryEngine(library, pairs)
    moments = tuple(geometry.uniform_moments(cell) for cell in cells)
    indexes = runner._training_indexes(cells, set(v1.FAILED_GROUPS))
    prepared = runner._prepare_training(cells, moments, indexes)
    training_groups = tuple(
        sorted({cell.group for cell in prepared.cells}, key=m1.GROUP_ORDER.index)
    )
    if not (
        len(cells) == runner.EXPECTED_ROWS
        and len(prepared.cells)
        == runner.EXPECTED_ROWS
        - m1.GROUP_SIZES[v1.FAILED_GROUPS[0]]
        - m1.GROUP_SIZES[v1.FAILED_GROUPS[1]]
        and len(training_groups) == 10
        and not set(v1.FAILED_GROUPS).intersection(training_groups)
        and counts.total_rows == runner.EXPECTED_TARGET_ROWS
        and counts.matched_rows == runner.EXPECTED_ROWS
    ):
        raise ReproDiagnosticError("failed-path training set identity changed")
    return prepared, library, geometry, len(prepared.cells), len(training_groups)


def _repeat_comparison(
    ridge_lambda: float,
    first: Any,
    second: Any,
    prepared: Any,
    library: Any,
    geometry: Any,
) -> Mapping[str, Any]:
    distribution = m1.compare_conditional_distributions(
        first.theta, second.theta, prepared, library, geometry
    )
    objective_difference = abs(float(first.objective) - float(second.objective))
    coefficient_difference = float(
        runner.np.max(runner.np.abs(first.theta - second.theta))
    )
    probability_difference = float(
        distribution["maximum_candidate_probability_absolute_difference"]
    )
    finite = all(
        math.isfinite(value)
        for value in (
            objective_difference,
            coefficient_difference,
            probability_difference,
        )
    )
    passed = bool(
        finite
        and objective_difference <= OBJECTIVE_TOLERANCE
        and coefficient_difference <= COEFFICIENT_TOLERANCE
        and probability_difference <= PROBABILITY_TOLERANCE
    )
    return {
        "lambda": float(ridge_lambda),
        "status": "pass" if passed else "fail",
        "objective_absolute_difference": objective_difference,
        "objective_absolute_tolerance": OBJECTIVE_TOLERANCE,
        "coefficient_maximum_absolute_difference": coefficient_difference,
        "coefficient_maximum_absolute_tolerance": COEFFICIENT_TOLERANCE,
        "probability_maximum_absolute_difference": probability_difference,
        "probability_maximum_absolute_tolerance": PROBABILITY_TOLERANCE,
    }


def run_diagnostic(args: argparse.Namespace) -> Mapping[str, Any]:
    started = time.perf_counter()
    runner.validate_real_contract(args.contract)
    runner.require_actual_k_prerequisite(
        args.actual_k_root, membership_opened_by_job_id=v1.FAILED_JOB_ID
    )
    v1.require_failed_attempt(args.prior_fit_root)
    output_root = _prepare_output_root(args.output_root)
    summary_path = output_root / SUMMARY_FILENAME
    payload: dict[str, Any] = {
        "status": "running",
        "stage": STAGE,
        "analysis_label": "exploratory",
        "SLURM_job_id": os.environ["SLURM_JOB_ID"],
        "failed_production_job_id": v1.FAILED_JOB_ID,
        "training_set": v1.FAILED_KEY,
        "excluded_groups": list(v1.FAILED_GROUPS),
        "lambda_order": list(m1.LAMBDAS_DESCENDING),
        "repeat_count": REPEAT_COUNT,
        "fit_attempts": [],
        "repeat_outcomes": [],
        "repeat_comparisons": [],
        "active_fit": None,
        "completed_fit_count": 0,
        "attempted_repeat_count": 0,
        "completed_repeat_count": 0,
        "realized_development_membership_opened": False,
        "M0_prediction_or_fit_artifact_opened": False,
        "final_or_unassigned_or_SWC_opened": False,
    }

    def checkpoint() -> None:
        payload["elapsed_wall_seconds"] = time.perf_counter() - started
        runner._atomic_json(summary_path, payload)

    checkpoint()
    try:
        payload["numeric_backend"] = _numeric_backend()
        payload["phase"] = "membership_access_started"
        payload["realized_development_membership_opened"] = True
        checkpoint()
        prepared, library, geometry, training_cells, training_groups = (
            _prepare_failed_training(args)
        )
        payload.update(
            {
                "training_cells": training_cells,
                "training_groups": training_groups,
                "target_fields_opened": (
                    "provider_neuron_id_fMOST_brain_id_K_and_detected16_only"
                ),
                "phase": "failed_path_two_repeat_replay",
            }
        )
        checkpoint()

        paths: list[dict[float, Any] | None] = []
        global_ordinal = 0
        for repeat_index in range(1, REPEAT_COUNT + 1):
            payload["attempted_repeat_count"] = repeat_index
            path: dict[float, Any] = {}
            initial = None
            previous_lambda: float | None = None
            repeat_failure: Mapping[str, Any] | None = None
            for path_ordinal, ridge_lambda in enumerate(
                m1.LAMBDAS_DESCENDING, start=1
            ):
                global_ordinal += 1
                initialization_source = (
                    "all_261_parameters_zero"
                    if initial is None
                    else "immediately_preceding_converged_joint_B_and_gamma"
                )
                payload["active_fit"] = {
                    "repeat": repeat_index,
                    "path_ordinal": path_ordinal,
                    "global_ordinal": global_ordinal,
                    "lambda": ridge_lambda,
                    "initialization_source": initialization_source,
                    "initial_from_lambda": previous_lambda,
                }
                checkpoint()
                phase = f"{v1.FAILED_KEY}__repeat_{repeat_index}"
                try:
                    result = m1.fit_newton(
                        prepared,
                        library,
                        geometry,
                        ridge_lambda,
                        initial=initial,
                    )
                except m1.M1Error as error:
                    record = m1._diagnostic_failure_record(
                        phase,
                        ridge_lambda,
                        initialization_source,
                        previous_lambda,
                        path_ordinal,
                        global_ordinal,
                        error,
                    )
                    record["repeat"] = repeat_index
                    payload["fit_attempts"].append(record)
                    payload["active_fit"] = None
                    payload.setdefault("first_failure", record)
                    repeat_failure = record
                    checkpoint()
                    break
                record = m1._diagnostic_success_record(
                    phase,
                    ridge_lambda,
                    initialization_source,
                    previous_lambda,
                    path_ordinal,
                    global_ordinal,
                    result,
                )
                record["repeat"] = repeat_index
                payload["fit_attempts"].append(record)
                payload["completed_fit_count"] = (
                    int(payload["completed_fit_count"]) + 1
                )
                payload["active_fit"] = None
                checkpoint()
                path[float(ridge_lambda)] = result
                initial = result.theta
                previous_lambda = ridge_lambda
            if repeat_failure is None:
                paths.append(path)
                payload["completed_repeat_count"] = (
                    int(payload["completed_repeat_count"]) + 1
                )
                payload["repeat_outcomes"].append(
                    {
                        "repeat": repeat_index,
                        "status": "complete",
                        "completed_fit_count": len(path),
                    }
                )
            else:
                paths.append(None)
                payload["repeat_outcomes"].append(
                    {
                        "repeat": repeat_index,
                        "status": "failed",
                        "completed_fit_count": len(path),
                        "failure_lambda": repeat_failure["lambda"],
                        "failure_class": repeat_failure["failure_class"],
                    }
                )
            checkpoint()

        failures = [
            outcome
            for outcome in payload["repeat_outcomes"]
            if outcome["status"] == "failed"
        ]
        if failures:
            expected = all(
                outcome["failure_class"] == "UncertifiedLineSearchFailure"
                for outcome in failures
            )
            payload["failure_recurrence_count"] = len(failures)
            payload["failure_reproduced"] = expected
            if not expected:
                payload["status"] = "fail_unexpected_failure_class"
                checkpoint()
                raise ReproDiagnosticError(
                    "failed path produced a different failure class"
                )
            payload["status"] = EXPECTED_FAILURE_STATUS
            payload["decision"] = EXPECTED_FAILURE_DECISION
            payload["decision_reason"] = (
                EXPECTED_FAILURE_REASON_BOTH
                if len(failures) == REPEAT_COUNT
                else EXPECTED_FAILURE_REASON_ONE
            )
            payload["repeat_agreement"] = bool(len(failures) == REPEAT_COUNT)
            payload["phase"] = "complete"
            checkpoint()
            return payload

        if any(path is None for path in paths):
            raise ReproDiagnosticError("repeat outcome/path accounting changed")
        first_path = paths[0]
        second_path = paths[1]
        assert first_path is not None and second_path is not None
        for ridge_lambda in m1.LAMBDAS_DESCENDING:
            comparison = _repeat_comparison(
                ridge_lambda,
                first_path[ridge_lambda],
                second_path[ridge_lambda],
                prepared,
                library,
                geometry,
            )
            payload["repeat_comparisons"].append(comparison)
            checkpoint()
            if comparison["status"] != "pass":
                payload["status"] = DISAGREEMENT_STATUS
                payload["first_failure"] = comparison
                payload["failure_reproduced"] = False
                payload["repeat_agreement"] = False
                payload["decision"] = DISAGREEMENT_DECISION
                payload["decision_reason"] = DISAGREEMENT_REASON
                payload["phase"] = "complete"
                checkpoint()
                return payload

        payload["status"] = AGREEMENT_STATUS
        payload["failure_reproduced"] = False
        payload["repeat_agreement"] = True
        payload["decision"] = AGREEMENT_DECISION
        payload["decision_reason"] = AGREEMENT_REASON
        payload["phase"] = "complete"
        checkpoint()
        return payload
    except Exception as error:
        if payload.get("status") == "running":
            payload.update(
                {
                    "status": "fail",
                    "failure_class": type(error).__name__,
                    "failure_code": "real_M1_original_backend_repro_failed_closed",
                }
            )
            checkpoint()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--interface-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--actual-k-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--prior-fit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_diagnostic(args)
    except Exception as error:
        print(
            f"ERROR: original-backend repro failed closed ({type(error).__name__})",
            file=sys.stderr,
        )
        return 2
    first_failure = result.get("first_failure")
    print(
        json.dumps(
            {
                "status": result["status"],
                "training_set": result["training_set"],
                "completed_fit_count": result["completed_fit_count"],
                "attempted_repeat_count": result["attempted_repeat_count"],
                "completed_repeat_count": result["completed_repeat_count"],
                "failure_reproduced": result["failure_reproduced"],
                "failure_lambda": (
                    first_failure.get("lambda") if first_failure else None
                ),
                "failure_class": (
                    first_failure.get("failure_class") if first_failure else None
                ),
                "repeat_agreement": result.get("repeat_agreement"),
                "decision": result.get("decision"),
                "decision_reason": result.get("decision_reason"),
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
