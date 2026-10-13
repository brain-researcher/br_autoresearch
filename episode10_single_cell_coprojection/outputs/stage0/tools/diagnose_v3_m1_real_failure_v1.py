#!/usr/bin/env python3
"""Reproduce the exact EP10 real-M1 warm-path solver failure with scalar telemetry."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(os.path.abspath(__file__))
RUNNER_PATH = SCRIPT_PATH.parent / "fit_v3_m1_real_v1.py"

SPEC = importlib.util.spec_from_file_location(
    "ep10_fit_v3_m1_real_failure_diagnostic", RUNNER_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import real-M1 runner: {RUNNER_PATH}")
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)
m1 = runner.m1


FAILED_JOB_ID = "45609750"
FAILED_ROOT_BASENAME = f"v3_m1_real_fit_v1_job_{FAILED_JOB_ID}"
FAILED_GROUPS = ("221418", "221588")
FAILED_KEY = f"leave_two__{FAILED_GROUPS[0]}__{FAILED_GROUPS[1]}"
EXPECTED_COMPLETED_PATHS = 53
OUTPUT_PREFIX = "v3_m1_real_warm_diagnostic_v1"
SUMMARY_FILENAME = "v3_m1_real_warm_diagnostic.json"


class DiagnosticError(runner.RealM1Error):
    """The narrow real-M1 failure diagnostic failed closed."""


def _open_regular_json(root: Path, name: str) -> Mapping[str, Any]:
    directory_fd = os.open(
        root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise DiagnosticError("diagnostic prerequisite is not a regular file")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    finally:
        os.close(directory_fd)
    if not isinstance(payload, dict):
        raise DiagnosticError("diagnostic prerequisite JSON is not a mapping")
    return payload


def require_failed_attempt(prior_root_argument: Path) -> Path:
    root = m1._require_exact_scratch_child(
        prior_root_argument, FAILED_ROOT_BASENAME, "failed real-M1 root"
    )
    summary = _open_regular_json(root, runner.SUMMARY_FILENAME)
    if not (
        summary.get("status") == "fail"
        and summary.get("stage") == "v3_M1_development_real_fit_v1"
        and summary.get("failure_class") == "UncertifiedLineSearchFailure"
        and summary.get("actual_K_gate_job_id") == runner.ACTUAL_K_JOB_ID
        and summary.get("M0_prediction_or_fit_artifact_opened") is False
        and summary.get("final_or_unassigned_or_SWC_opened") is False
    ):
        raise DiagnosticError("prior real-M1 failure identity changed")

    checkpoint_root = root / runner.CHECKPOINT_DIRECTORY
    if not checkpoint_root.is_dir() or checkpoint_root.is_symlink():
        raise DiagnosticError("prior checkpoint root is absent or symlinked")
    observed: set[str] = set()
    for path in checkpoint_root.iterdir():
        if not path.is_file() or path.is_symlink():
            raise DiagnosticError("prior checkpoint root contains a nonordinary entry")
        if not (
            path.name.startswith("warm__leave_two__") and path.name.endswith(".npz")
        ):
            raise DiagnosticError("prior checkpoint set contains an unexpected fit kind")
        observed.add(path.name.removeprefix("warm__").removesuffix(".npz"))
    ordered = [
        f"leave_two__{left}__{right}"
        for left_index, left in enumerate(m1.GROUP_ORDER[:-1])
        for right in m1.GROUP_ORDER[left_index + 1 :]
    ]
    if not (
        len(observed) == EXPECTED_COMPLETED_PATHS
        and observed == set(ordered[:EXPECTED_COMPLETED_PATHS])
        and ordered[EXPECTED_COMPLETED_PATHS] == FAILED_KEY
    ):
        raise DiagnosticError("prior checkpoint frontier is not the exact failed path")
    return root


def _prepare_output_root(path: Path) -> Path:
    scratch_value = os.environ.get("SCRATCH")
    job_id = os.environ.get("SLURM_JOB_ID")
    if not scratch_value or not job_id:
        raise DiagnosticError("real-M1 diagnostic requires SCRATCH and SLURM_JOB_ID")
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
        raise DiagnosticError("diagnostic output is not the exact job-scoped Scratch child")
    try:
        absolute.mkdir(mode=0o770)
    except FileExistsError as error:
        raise DiagnosticError("diagnostic output already exists") from error
    return absolute


def run_diagnostic(args: argparse.Namespace) -> Mapping[str, Any]:
    started = time.perf_counter()
    runner.validate_real_contract(args.contract)
    runner.require_actual_k_prerequisite(
        args.actual_k_root, membership_opened_by_job_id=FAILED_JOB_ID
    )
    require_failed_attempt(args.prior_fit_root)
    output_root = _prepare_output_root(args.output_root)
    summary_path = output_root / SUMMARY_FILENAME
    payload: dict[str, Any] = {
        "status": "running",
        "stage": "v3_M1_real_warm_path_failure_diagnostic_v1",
        "analysis_label": "exploratory",
        "SLURM_job_id": os.environ["SLURM_JOB_ID"],
        "failed_production_job_id": FAILED_JOB_ID,
        "training_set": FAILED_KEY,
        "excluded_groups": list(FAILED_GROUPS),
        "lambda_order": list(m1.LAMBDAS_DESCENDING),
        "fit_attempts": [],
        "active_fit": None,
        "completed_fit_count": 0,
        "realized_development_membership_opened": False,
        "M0_prediction_or_fit_artifact_opened": False,
        "final_or_unassigned_or_SWC_opened": False,
    }

    def checkpoint() -> None:
        payload["elapsed_wall_seconds"] = time.perf_counter() - started
        runner._atomic_json(summary_path, payload)

    checkpoint()
    try:
        target_blind_cells, pairs = m1.load_target_free_real_shape(
            args.interface_root, args.geometry_root
        )
        expected_ids = {cell.identifier for cell in target_blind_cells}
        payload["phase"] = "membership_access_started"
        payload["realized_development_membership_opened"] = True
        checkpoint()
        membership, counts = runner._read_membership_inputs(
            args.target_root, expected_ids
        )
        library = m1.CandidateLibrary()
        cells = runner._align_membership(target_blind_cells, membership, library)
        del membership
        geometry = m1.GeometryEngine(library, pairs)
        moments = tuple(geometry.uniform_moments(cell) for cell in cells)
        indexes = runner._training_indexes(cells, set(FAILED_GROUPS))
        prepared = runner._prepare_training(cells, moments, indexes)
        training_groups = tuple(
            sorted({cell.group for cell in prepared.cells}, key=m1.GROUP_ORDER.index)
        )
        if not (
            len(cells) == runner.EXPECTED_ROWS
            and len(prepared.cells)
            == runner.EXPECTED_ROWS
            - m1.GROUP_SIZES[FAILED_GROUPS[0]]
            - m1.GROUP_SIZES[FAILED_GROUPS[1]]
            and len(training_groups) == 10
            and not set(FAILED_GROUPS).intersection(training_groups)
            and counts.total_rows == runner.EXPECTED_TARGET_ROWS
            and counts.matched_rows == runner.EXPECTED_ROWS
        ):
            raise DiagnosticError("failed-path training set identity changed")
        payload.update(
            {
                "training_cells": len(prepared.cells),
                "training_groups": len(training_groups),
                "target_fields_opened": (
                    "provider_neuron_id_fMOST_brain_id_K_and_detected16_only"
                ),
                "phase": "failed_path_warm_replay",
            }
        )
        checkpoint()

        initial = None
        previous_lambda: float | None = None
        for ordinal, ridge_lambda in enumerate(m1.LAMBDAS_DESCENDING, start=1):
            initialization_source = (
                "all_261_parameters_zero"
                if initial is None
                else "immediately_preceding_converged_joint_B_and_gamma"
            )
            payload["active_fit"] = {
                "ordinal": ordinal,
                "lambda": ridge_lambda,
                "initialization_source": initialization_source,
                "initial_from_lambda": previous_lambda,
            }
            checkpoint()
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
                    FAILED_KEY,
                    ridge_lambda,
                    initialization_source,
                    previous_lambda,
                    ordinal,
                    ordinal,
                    error,
                )
                payload["fit_attempts"].append(record)
                payload["active_fit"] = None
                payload["first_failure"] = record
                payload["failure_reproduced"] = isinstance(
                    error, m1.UncertifiedLineSearchFailure
                )
                payload["status"] = (
                    "pass"
                    if payload["failure_reproduced"]
                    else "fail_unexpected_failure_class"
                )
                checkpoint()
                if not payload["failure_reproduced"]:
                    raise DiagnosticError(
                        "failed path reproduced a different failure class"
                    ) from error
                return payload
            record = m1._diagnostic_success_record(
                FAILED_KEY,
                ridge_lambda,
                initialization_source,
                previous_lambda,
                ordinal,
                ordinal,
                result,
            )
            payload["fit_attempts"].append(record)
            payload["completed_fit_count"] = ordinal
            payload["active_fit"] = None
            checkpoint()
            initial = result.theta
            previous_lambda = ridge_lambda

        payload["status"] = "fail_failure_not_reproduced"
        payload["failure_reproduced"] = False
        checkpoint()
        raise DiagnosticError("failed warm path unexpectedly completed")
    except Exception as error:
        if payload.get("status") == "running":
            payload.update(
                {
                    "status": "fail",
                    "failure_class": type(error).__name__,
                    "failure_code": "real_M1_warm_diagnostic_failed_closed",
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
            f"ERROR: real-M1 warm diagnostic failed closed ({type(error).__name__})",
            file=sys.stderr,
        )
        return 2
    first_failure = result["first_failure"]
    context = first_failure.get("optimizer_scalar_context") or {}
    print(
        json.dumps(
            {
                "status": result["status"],
                "training_set": result["training_set"],
                "completed_fit_count": result["completed_fit_count"],
                "failure_lambda": first_failure["lambda"],
                "failure_class": first_failure["failure_class"],
                "certificate_pass": (context.get("certificate") or {}).get(
                    "certificate_pass"
                ),
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
