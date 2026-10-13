#!/usr/bin/env python3
"""Run the sealed EP10 v3 M1 actual-K-only pre-fit geometry gate.

This program is deliberately separate from the M1 fitting CLI.  Its only
target-derived inputs are ``provider_neuron_id`` and physical ``K`` from the
frozen development target-call CSV.  It never reads realized set membership,
fits a model, or accepts an M0, SWC, final, or unassigned input route.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import stat
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(os.path.abspath(__file__))
STAGE0_DIR = SCRIPT_PATH.parent.parent
ENGINE_PATH = SCRIPT_PATH.parent / "fit_v3_m1.py"

SPEC = importlib.util.spec_from_file_location("ep10_fit_v3_m1_actual_k_gate", ENGINE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import frozen M1 engine: {ENGINE_PATH}")
m1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m1
SPEC.loader.exec_module(m1)


TARGET_ROOT_BASENAME = "gao_mop_development_targets_v3_ep10_assets"
TARGET_FILENAME = "primary_16_target_calls.csv"
TARGET_GATE_FILENAME = "v3_gate_summary.json"
TARGET_FREE_OUTPUT_PREFIX = "v3_m1_target_free_probe_v4_job_"
TARGET_FREE_SUMMARY_FILENAME = "v3_m1_target_free_probe_summary.json"
OUTPUT_PREFIX = "v3_m1_actual_k_gate_v1"
SUMMARY_FILENAME = "v3_m1_actual_k_gate_summary.json"
PERMITTED_TARGET_FIELDS = ("provider_neuron_id", "K")
READINESS_PATH = STAGE0_DIR / "readiness_status.yaml"
EXPECTED_TARGET_ROWS = 200
EXPECTED_ROWS = 194
EXPECTED_EXCLUDED_ROWS = 6
EXPECTED_FOLDS = 79
TARGET_FIELD_COUNT = 90
TARGET_IDENTIFIER_INDEX = 0
TARGET_K_INDEX = 8
K_PATTERN = re.compile(r"(?:[0-9]|1[0-6])\Z")


class ActualKGateError(m1.M1Error):
    """The sealed actual-K gate failed closed."""


@dataclass(frozen=True)
class ProjectionScanCounts:
    total_rows: int
    matched_rows: int
    extra_rows: int


def validate_actual_k_contract(path: Path) -> Mapping[str, Any]:
    """Validate the already-frozen scientific clauses used by this gate."""
    contract = m1.validate_contract(path)
    required = contract.get("required_inputs", {})
    target = required.get("target_root", {})
    join = required.get("join", {})
    sealed = contract.get("sealed_scope", {})
    general = contract.get("pre_real_M1_gates", {}).get("general", {})
    gate = contract.get("pre_real_M1_gates", {}).get(
        "actual_K_pre_fit_rank_and_condition", {}
    )
    if (
        target.get("required_basename") != TARGET_ROOT_BASENAME
        or target.get("required_file") != TARGET_FILENAME
        or target.get("required_gate_summary") != TARGET_GATE_FILENAME
        or tuple(gate.get("permitted_target_fields", ())) != PERMITTED_TARGET_FIELDS
        or gate.get("semantic_alias") != "K_is_K_i_obs"
        or gate.get("realized_positive_set_membership") != "forbidden"
        or gate.get("folds")
        != "every_actual_outer_inner_and_full_development_training_fold"
        or gate.get("transform_and_Gram")
        != "identical_to_the_within_cell_rank_and_condition_definition_above"
        or int(gate.get("geometry_residual_rank_required", -1)) != 6
        or int(gate.get("joint_numerical_rank_required", -1))
        != m1.PARAMETER_DIMENSION
        or float(gate.get("maximum_joint_condition_number_2", math.nan)) != 1.0e10
        or int(sealed.get("required_cells", -1)) != EXPECTED_ROWS
        or int(sealed.get("required_groups", -1)) != len(m1.GROUP_ORDER)
        or sealed.get("final_or_unassigned_access") != "forbidden"
        or join.get("key") != "provider_neuron_id"
        or join.get("cardinality") != "one_to_one"
        or join.get("exact_identity_set_required") is not True
        or join.get("exact_group_match_required") is not True
        or general.get("every_target_free_gate_must_pass_before_first_target_file_access")
        is not True
        or general.get("actual_K_gate_must_then_pass_before_any_fit_to_real_observed_P_i")
        is not True
        or general.get("observed_target_call_file_access_during_target_free_gates")
        != "forbidden"
        or general.get("M0_prediction_or_fit_artifact_access_during_all_pre_fit_gates")
        != "forbidden"
        or general.get("gate_failure_action")
        != "stop_and_freeze_a_reviewed_addendum_before_any_real_M1"
        or gate.get("failure_action") is None
    ):
        raise ActualKGateError("frozen actual-K gate clauses changed")
    return contract


def _require_target_free_prerequisites(target_free_root_argument: Path) -> str:
    """Require the passed v4 artifact plus the reconciled execution ledger."""
    readiness_absolute = Path(os.path.abspath(READINESS_PATH))
    if (
        readiness_absolute != READINESS_PATH
        or not readiness_absolute.is_file()
        or readiness_absolute.is_symlink()
        or readiness_absolute.resolve(strict=True) != READINESS_PATH.resolve(strict=True)
    ):
        raise ActualKGateError("readiness ledger is not the exact stage0 file")
    readiness_fd = os.open(
        readiness_absolute, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    if not stat.S_ISREG(os.fstat(readiness_fd).st_mode):
        os.close(readiness_fd)
        raise ActualKGateError("readiness ledger is not a regular file")
    with os.fdopen(readiness_fd, "r", encoding="utf-8") as handle:
        readiness = m1.yaml.safe_load(handle)
    if not isinstance(readiness, dict):
        raise ActualKGateError("readiness ledger is not a mapping")
    gates = readiness.get("gates", {})
    access = readiness.get("access", {})
    probe_record = str(gates.get("gao_development_v3_m1_complete_target_free_v4_probe", ""))
    match = re.match(r"pass_job_([0-9]+)_", probe_record)
    if match is None:
        raise ActualKGateError("target-free v4 pass is absent from the readiness ledger")
    job_id = match.group(1)
    if not (
        str(gates.get("gao_development_v3_m1_complete_target_free_v4_postflight", ""))
        .startswith("pass_")
        and str(gates.get("gao_development_v3_remaining_budget_reconciliation", ""))
        .startswith("pass_")
        and access.get("v3_m1_actual_K_opened") is False
        and access.get("v3_m1_realized_membership_opened") is False
        and access.get("v3_m1_M0_artifact_opened") is False
        and access.get("final_group_outcomes_opened") is False
        and int(job_id) in readiness.get("launch", {}).get("slurm_job_ids", [])
    ):
        raise ActualKGateError("target-free postflight or budget seal is absent")

    basename = f"{TARGET_FREE_OUTPUT_PREFIX}{job_id}"
    root = m1._require_exact_scratch_child(
        target_free_root_argument, basename, "target-free v4 root"
    )
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        summary_fd = _open_regular_at(root_fd, TARGET_FREE_SUMMARY_FILENAME)
        try:
            with os.fdopen(summary_fd, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
        except (json.JSONDecodeError, OSError) as error:
            raise ActualKGateError("target-free v4 summary is unreadable") from error
    finally:
        os.close(root_fd)
    benchmark = summary.get("real_shape_benchmark", {})
    mixed_rank = benchmark.get("mixed_K_rank_and_condition", {})
    if not (
        summary.get("status")
        == "pass_pending_post_job_sacct_MaxRSS_and_remaining_budget_reconciliation"
        and str(summary.get("SLURM_job_id")) == job_id
        and summary.get("unit_gates", {}).get("status") == "pass"
        and summary.get("synthetic_gates", {}).get("status") == "pass"
        and benchmark.get("status")
        == "pass_pending_post_job_sacct_MaxRSS_and_remaining_budget_reconciliation"
        and mixed_rank.get("status") == "pass"
        and int(mixed_rank.get("unique_training_sets_evaluated", -1)) == EXPECTED_FOLDS
        and benchmark.get("policy_ceiling_projection_pass") is True
        and summary.get("real_target_calls_or_real_K_opened") is False
        and summary.get("M0_prediction_or_fit_artifact_opened") is False
        and summary.get("final_or_unassigned_or_SWC_opened") is False
        and benchmark.get("real_target_calls_or_real_K_opened") is False
        and benchmark.get("M0_prediction_or_fit_artifact_opened") is False
        and benchmark.get("final_or_unassigned_or_SWC_opened") is False
    ):
        raise ActualKGateError("target-free v4 artifact is not passed and sealed")
    return job_id


def _open_regular_at(directory_fd: int, name: str) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ActualKGateError("sealed target input is not a regular file")
    return descriptor


def _scan_actual_k_projection(
    handle: Any, expected_ids: set[str]
) -> tuple[dict[str, int], ProjectionScanCounts]:
    """Project fixed physical fields without materializing forbidden CSV cells.

    Raw bytes necessarily transit the binary I/O buffer.  Only physical field
    0 (provider_neuron_id) is assembled for every row; physical field 8 (K) is
    assembled only after that ID is known to be in the 194-row interface.
    Every other field is represented only by parser state and a nonempty bit.
    """
    projected: dict[str, int] = {}
    seen_identifiers: set[str] = set()
    expected = set(expected_ids)
    record_index = 0
    data_rows = 0
    field_index = 0
    field_buffer = bytearray()
    field_nonempty = False
    field_at_start = True
    record_active = False
    in_quotes = False
    quote_pending = False
    current_identifier: str | None = None
    current_matched = False
    current_k_seen = False

    def capture_current_field() -> bool:
        if record_index == 0:
            return field_index in {TARGET_IDENTIFIER_INDEX, TARGET_K_INDEX}
        return field_index == TARGET_IDENTIFIER_INDEX or (
            field_index == TARGET_K_INDEX and current_matched
        )

    def append_selected(value: bytes) -> None:
        if capture_current_field():
            field_buffer.extend(value)

    def finish_field() -> None:
        nonlocal field_index, field_nonempty, field_at_start
        nonlocal current_identifier, current_matched, current_k_seen
        value = bytes(field_buffer)
        field_buffer.clear()
        if record_index == 0:
            if field_index == TARGET_IDENTIFIER_INDEX and value != b"provider_neuron_id":
                raise ActualKGateError("physical target identifier column changed")
            if field_index == TARGET_K_INDEX and value != b"K":
                raise ActualKGateError("physical target K column changed")
        elif field_index == TARGET_IDENTIFIER_INDEX:
            try:
                identifier = value.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ActualKGateError("target identifier is not UTF-8") from error
            if not identifier or identifier in seen_identifiers:
                raise ActualKGateError("target identifiers are empty or duplicated")
            seen_identifiers.add(identifier)
            current_identifier = identifier
            current_matched = identifier in expected
        elif field_index == TARGET_K_INDEX:
            if current_identifier is None:
                raise ActualKGateError("target K precedes its identifier")
            if current_matched:
                try:
                    raw_k = value.decode("ascii")
                except UnicodeDecodeError as error:
                    raise ActualKGateError("actual K is not canonical ASCII") from error
                if K_PATTERN.fullmatch(raw_k) is None:
                    raise ActualKGateError("actual-K projection contains a noncanonical K")
                projected[current_identifier] = int(raw_k)
                current_k_seen = True
            elif field_nonempty:
                raise ActualKGateError("excluded target row has a nonempty physical K")
        field_index += 1
        field_nonempty = False
        field_at_start = True

    def finish_record() -> None:
        nonlocal record_index, data_rows, field_index, record_active
        nonlocal current_identifier, current_matched, current_k_seen
        if field_index != TARGET_FIELD_COUNT:
            raise ActualKGateError("development target-call CSV physical width changed")
        if record_index > 0:
            if current_identifier is None or (current_matched and not current_k_seen):
                raise ActualKGateError("actual-K projection row is incomplete")
            data_rows += 1
        record_index += 1
        field_index = 0
        record_active = False
        current_identifier = None
        current_matched = False
        current_k_seen = False

    while True:
        byte = handle.read(1)
        if not byte:
            if in_quotes and not quote_pending:
                raise ActualKGateError("development target-call CSV has an unterminated quote")
            if quote_pending:
                in_quotes = False
                quote_pending = False
            if record_active or field_index or field_nonempty or field_buffer:
                finish_field()
                finish_record()
            break

        closed_quote = False
        if in_quotes:
            record_active = True
            if quote_pending:
                if byte == b'"':
                    field_nonempty = True
                    append_selected(b'"')
                    quote_pending = False
                    continue
                in_quotes = False
                quote_pending = False
                closed_quote = True
            elif byte == b'"':
                quote_pending = True
                continue
            else:
                field_nonempty = True
                append_selected(byte)
                continue
        elif field_at_start and byte == b'"':
            in_quotes = True
            field_at_start = False
            record_active = True
            continue
        elif byte == b'"':
            raise ActualKGateError("development target-call CSV has a stray quote")

        if byte == b",":
            record_active = True
            finish_field()
        elif byte == b"\n":
            finish_field()
            finish_record()
        elif byte == b"\r":
            if handle.read(1) != b"\n":
                raise ActualKGateError("development target-call CSV line ending changed")
            finish_field()
            finish_record()
        elif closed_quote:
            raise ActualKGateError("quoted target field has trailing characters")
        else:
            record_active = True
            field_at_start = False
            field_nonempty = True
            append_selected(byte)

    if record_index == 0:
        raise ActualKGateError("development target-call CSV is empty")
    counts = ProjectionScanCounts(
        total_rows=data_rows,
        matched_rows=len(projected),
        extra_rows=data_rows - len(projected),
    )
    return projected, counts


def _read_actual_k_projection(
    path: Path, expected_ids: set[str]
) -> tuple[dict[str, int], ProjectionScanCounts]:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ActualKGateError("actual-K projection is not a regular file")
    with os.fdopen(descriptor, "rb", buffering=65536) as handle:
        return _scan_actual_k_projection(handle, expected_ids)


def _read_target_inputs(
    target_root_argument: Path, expected_ids: set[str]
) -> tuple[dict[str, int], ProjectionScanCounts]:
    root = m1._require_exact_scratch_child(
        target_root_argument, TARGET_ROOT_BASENAME, "development target root"
    )
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        gate_fd = _open_regular_at(root_fd, TARGET_GATE_FILENAME)
        try:
            with os.fdopen(gate_fd, "r", encoding="utf-8") as handle:
                upstream = json.load(handle)
        except (json.JSONDecodeError, OSError) as error:
            raise ActualKGateError("upstream development target gate is unreadable") from error
        if not (
            isinstance(upstream, dict)
            and upstream.get("status") == "pass"
            and int(upstream.get("input_files", -1)) == EXPECTED_TARGET_ROWS
            and int(upstream.get("eligible_cells", -1)) == EXPECTED_ROWS
            and int(upstream.get("excluded_cells", -1)) == EXPECTED_EXCLUDED_ROWS
            and upstream.get("final_or_unassigned_files_opened") is False
        ):
            raise ActualKGateError("upstream development target gate is not passed and sealed")
        target_fd = _open_regular_at(root_fd, TARGET_FILENAME)
        with os.fdopen(target_fd, "rb", buffering=65536) as handle:
            projected, counts = _scan_actual_k_projection(handle, expected_ids)
    finally:
        os.close(root_fd)
    if (
        counts.total_rows != EXPECTED_TARGET_ROWS
        or counts.matched_rows != EXPECTED_ROWS
        or counts.extra_rows != EXPECTED_EXCLUDED_ROWS
        or set(projected) != expected_ids
    ):
        raise ActualKGateError("target projection does not match the 194-row interface")
    return projected, counts


def _prepare_output_root(path: Path) -> Path:
    scratch_value = os.environ.get("SCRATCH")
    job_id = os.environ.get("SLURM_JOB_ID")
    if not scratch_value or not job_id:
        raise ActualKGateError("actual-K gate requires SCRATCH and SLURM_JOB_ID")
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
        raise ActualKGateError("actual-K output is not the job-scoped Scratch child")
    try:
        absolute.mkdir(mode=0o770)
    except FileExistsError as error:
        raise ActualKGateError("actual-K output root already exists") from error
    return absolute


def _align_actual_k(
    cells: Sequence[m1.Cell], projected: Mapping[str, int]
) -> tuple[m1.Cell, ...]:
    identifiers = tuple(cell.identifier for cell in cells)
    if (
        len(cells) != EXPECTED_ROWS
        or len(set(identifiers)) != EXPECTED_ROWS
        or len(projected) != EXPECTED_ROWS
        or set(projected) != set(identifiers)
    ):
        raise ActualKGateError("actual-K projection and target-blind interface do not join exactly")
    # observed_index is intentionally a fixed inert placeholder: the rank gate
    # never reads a realized candidate identity.
    result: list[m1.Cell] = []
    for cell in cells:
        k = int(projected[cell.identifier])
        # One-past-end is rejected by explicit engine checks and also raises on
        # any accidental direct Python/NumPy indexing; it cannot name a set.
        result.append(
            replace(cell, k=k, observed_index=math.comb(m1.PANEL_SIZE, k))
        )
    return tuple(result)


def _aggregate_rank_result(result: Mapping[str, Any]) -> dict[str, Any]:
    folds = result.get("folds")
    if not isinstance(folds, list) or len(folds) != EXPECTED_FOLDS:
        raise ActualKGateError("actual-K rank gate did not evaluate exactly 79 folds")
    joint_ranks: list[int] = []
    block_ranks: list[int] = []
    residual_ranks: list[int] = []
    conditions: list[float] = []
    finite = True
    passed = 0
    training_cells: list[int] = []
    for fold in folds:
        joint = fold["joint"]
        block = fold["geometry_block"]
        residual = fold["geometry_residual"]
        joint_ranks.append(int(joint["rank"]))
        block_ranks.append(int(block["rank"]))
        residual_ranks.append(int(residual["rank"]))
        training_cells.append(int(fold["training_cells"]))
        condition = joint.get("condition_number_2")
        if condition is not None and math.isfinite(float(condition)):
            conditions.append(float(condition))
        else:
            finite = False
        finite = finite and bool(joint.get("finite")) and bool(block.get("finite"))
        finite = finite and bool(residual.get("finite"))
        passed += int(fold.get("status") == "pass")
    maximum_condition = max(conditions) if conditions else None
    computed_pass = bool(
        result.get("status") == "pass"
        and passed == EXPECTED_FOLDS
        and finite
        and min(joint_ranks) == m1.PARAMETER_DIMENSION
        and min(block_ranks) == m1.GEOMETRY_DIMENSION
        and min(residual_ranks) == m1.GEOMETRY_DIMENSION
        and maximum_condition is not None
        and maximum_condition <= 1.0e10
    )
    return {
        "status": "pass" if computed_pass else "fail",
        "training_folds_evaluated": len(folds),
        "training_folds_passed": passed,
        "minimum_training_cells": min(training_cells),
        "maximum_training_cells": max(training_cells),
        "minimum_joint_rank": min(joint_ranks),
        "joint_rank_required": m1.PARAMETER_DIMENSION,
        "minimum_geometry_block_rank": min(block_ranks),
        "minimum_geometry_residual_rank": min(residual_ranks),
        "geometry_residual_rank_required": m1.GEOMETRY_DIMENSION,
        "maximum_joint_condition_number_2": maximum_condition,
        "joint_condition_number_2_limit": 1.0e10,
        "all_reported_diagnostics_finite": finite,
        "fold_level_rows_emitted": False,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    directory_fd = os.open(
        path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    temporary_name = path.name + ".partial"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o660,
            dir_fd=directory_fd,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_fd)


def run_actual_k_gate(args: argparse.Namespace) -> dict[str, Any]:
    validate_actual_k_contract(args.contract)
    target_free_job_id = _require_target_free_prerequisites(args.target_free_root)
    output_root = _prepare_output_root(args.output_root)
    summary_path = output_root / SUMMARY_FILENAME
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "status": "running",
        "stage": "v3_M1_actual_K_only_pre_fit_rank_and_condition_v1",
        "analysis_label": "exploratory",
        "rows": EXPECTED_ROWS,
        "groups": len(m1.GROUP_ORDER),
        "target_projection_rows": EXPECTED_TARGET_ROWS,
        "modeled_interface_rows": EXPECTED_ROWS,
        "unmatched_target_projection_rows": EXPECTED_EXCLUDED_ROWS,
        "SLURM_job_id": os.environ["SLURM_JOB_ID"],
        "qualified_target_free_job_id": target_free_job_id,
        "target_fields_opened": "provider_neuron_id_and_physical_K_only",
        "provider_neuron_ids_emitted": False,
        "K_values_or_distribution_emitted": False,
        "realized_positive_set_membership_opened": False,
        "candidate_scores_or_model_fit_computed": False,
        "M0_prediction_or_fit_artifact_opened": False,
        "SWC_opened": False,
        "final_or_unassigned_opened": False,
    }
    try:
        target_blind_cells, pairs = m1.load_target_free_real_shape(
            args.interface_root, args.geometry_root
        )
        expected_ids = {cell.identifier for cell in target_blind_cells}
        projected, counts = _read_target_inputs(args.target_root, expected_ids)
        if (
            counts.total_rows != EXPECTED_TARGET_ROWS
            or counts.matched_rows != EXPECTED_ROWS
            or counts.extra_rows != EXPECTED_EXCLUDED_ROWS
        ):
            raise ActualKGateError("actual-K projection counts changed")
        cells = _align_actual_k(target_blind_cells, projected)
        library = m1.CandidateLibrary()
        geometry = m1.GeometryEngine(library, pairs)
        moments = tuple(geometry.uniform_moments(cell) for cell in cells)
        folds = m1._all_training_folds(cells)
        rank = m1.rank_condition_gate(
            cells, moments, folds, raise_on_failure=False
        )
        payload["rank_and_condition"] = _aggregate_rank_result(rank)
        payload["elapsed_wall_seconds"] = time.perf_counter() - started
        payload["status"] = payload["rank_and_condition"]["status"]
        _atomic_json(summary_path, payload)
        if payload["status"] != "pass":
            raise ActualKGateError("actual-K rank or condition gate failed")
        return payload
    except Exception as error:
        if payload.get("status") != "fail":
            payload.update(
                {
                    "status": "fail",
                    "failure_class": type(error).__name__,
                    "failure_code": "actual_K_gate_failed_closed",
                    "elapsed_wall_seconds": time.perf_counter() - started,
                }
            )
            _atomic_json(summary_path, payload)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--interface-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--target-free-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_actual_k_gate(args)
    except Exception as error:
        print(f"ERROR: actual-K gate failed closed ({type(error).__name__})", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "stage": result["stage"],
                "rows": result["rows"],
                "training_folds_evaluated": result["rank_and_condition"][
                    "training_folds_evaluated"
                ],
                "realized_positive_set_membership_opened": False,
                "M0_prediction_or_fit_artifact_opened": False,
                "final_or_unassigned_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
