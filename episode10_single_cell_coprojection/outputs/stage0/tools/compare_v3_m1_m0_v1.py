#!/usr/bin/env python3
"""Compare frozen EP10 v3 M1 and M0 outer-cross-fitted predictions.

The comparator is the component allowed to open the M0 prediction artifact
after the repaired M1 fit. Historical M0 access remains recorded. It validates
the independently postflighted current M1 result first, writes
the access state atomically, and only then opens the two permitted M0 files.
It has no route to target calls, morphology, final/unassigned data, telemetry,
or either model's full-development refit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import stat
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


SCRIPT_PATH = Path(os.path.abspath(__file__))
STAGE0_DIR = SCRIPT_PATH.parent.parent
CONTRACT_PATH = STAGE0_DIR / "v3_m1_model_contract.yaml"
READINESS_PATH = STAGE0_DIR / "readiness_status.yaml"

M1_JOB_ID = "46332190"
M0_JOB_ID = "45497012"
HISTORICAL_M0_ACCESS = "true_job_45760290_post_M1_mechanical_comparison_only"
POSTFLIGHT_KEY = "gao_development_v3_m1_repaired_clean_production_postflight"
M1_ROOT_BASENAME = f"v3_m1_real_fit_single_thread_v3_job_{M1_JOB_ID}"
M0_ROOT_BASENAME = "v3_m0_fit_v7"
OUTPUT_PREFIX = "v3_m1_compare_m0_v1"
REPORT_FILENAME = "v3_m1_minus_m0_comparison.json"
M1_MANIFEST_FILENAME = "v3_m1_real_single_thread_v3_checkpoint_manifest.json"
M1_SUMMARY_FILENAME = "v3_m1_fit_summary.json"
M1_PREDICTIONS_FILENAME = "v3_m1_cross_fitted_predictions.csv"
M0_SUMMARY_FILENAME = "v3_m0_fit_summary.json"
M0_PREDICTIONS_FILENAME = "v3_m0_cross_fitted_predictions.csv"

M1_STAGE = "v3_M1_development_real_fit_single_thread_v3"
M0_STAGE = "v3_M0_development_outer_cross_fit_and_full_refit"
COMPARISON_STAGE = "v3_M1_minus_M0_development_comparison_v1"
ESTIMAND = "mapped_positive_terminal_observation_pattern_conditional_on_recorded_K"
POSTFLIGHT_GATE = (
    "pass_job_46332190_artifact_numerical_and_access_seal_audits_"
    "194_predictions_12_groups_91_checkpoints_949_fits_boundary_0_"
    "M0_SWC_final_unassigned_sealed"
)
M0_GATE = (
    "pass_job_45497012_194_cells_12_groups_12_outer_folds_zero_boundary_"
    "folds_full_lambda_0_1_1741_of_1741_valid_convergence_witnesses"
)

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
GROUP_ORDER = tuple(GROUP_SIZES)
EXPECTED_ROWS = 194
EXPECTED_GROUPS = 12
PANEL_SIZE = 16
PAIRING_KEYS = ("provider_neuron_id", "fMOST_brain_id")
UNIFORM_TOLERANCE = 1.0e-12


class ComparisonError(RuntimeError):
    """A frozen input, access boundary, or paired-score invariant failed."""


@dataclass(frozen=True)
class ScoreRow:
    identifier: str
    group: str
    k: int
    log2_probability: float
    uniform_log2_probability: float


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    if path.is_symlink() or temporary.is_symlink():
        raise ComparisonError("refusing symlinked comparison report")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o660,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                _jsonable(payload),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _open_regular_path(path: Path, expected: Path, label: str) -> int:
    absolute = Path(os.path.abspath(path))
    if absolute != expected:
        raise ComparisonError(f"{label} is not the exact stage0 path")
    try:
        descriptor = os.open(absolute, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise ComparisonError(f"{label} is unavailable") from error
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ComparisonError(f"{label} is not an ordinary file")
    return descriptor


def _read_yaml_path(path: Path, expected: Path, label: str) -> Mapping[str, Any]:
    descriptor = _open_regular_path(path, expected, label)
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        raise ComparisonError(f"{label} is unreadable") from error
    if not isinstance(payload, dict):
        raise ComparisonError(f"{label} is not a mapping")
    return payload


def _open_exact_directory(path: Path, expected: Path, label: str) -> int:
    absolute = Path(os.path.abspath(path))
    if absolute != expected:
        raise ComparisonError(f"{label} is not the exact frozen Scratch root")
    try:
        descriptor = os.open(
            absolute,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
        )
    except OSError as error:
        raise ComparisonError(f"{label} is unavailable or symlinked") from error
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ComparisonError(f"{label} is not a directory")
    return descriptor


def _open_regular_at(directory_fd: int, name: str, label: str) -> int:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    except OSError as error:
        raise ComparisonError(f"{label} is unavailable or symlinked") from error
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ComparisonError(f"{label} is not an ordinary file")
    return descriptor


def _read_json_at(directory_fd: int, name: str, label: str) -> Mapping[str, Any]:
    descriptor = _open_regular_at(directory_fd, name, label)
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonError(f"{label} is unreadable") from error
    if not isinstance(payload, dict):
        raise ComparisonError(f"{label} is not a mapping")
    return payload


def _read_csv_at(
    directory_fd: int,
    name: str,
    label: str,
    model: str,
) -> tuple[ScoreRow, ...]:
    descriptor = _open_regular_at(directory_fd, name, label)
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ComparisonError(f"{label} has no header")
            return _parse_prediction_rows(reader, reader.fieldnames, model=model)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ComparisonError(f"{label} is unreadable") from error


def _validate_contract(contract: Mapping[str, Any]) -> None:
    sealed = contract.get("sealed_scope", {})
    required = contract.get("required_inputs", {})
    comparator = required.get("M0_comparator_root", {})
    join = required.get("join", {})
    group_sizes = required.get("expected_group_sizes", {})
    try:
        normalized_sizes = {
            str(key): int(value) for key, value in group_sizes.items()
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise ComparisonError("M1 contract group sizes are malformed") from error
    if not (
        contract.get("schema_version") == "episode10_v3_m1_model_v1"
        and contract.get("status")
        == "frozen_before_implementation_and_real_M1_execution"
        and sealed.get("role") == "development_only"
        and int(sealed.get("required_cells", -1)) == EXPECTED_ROWS
        and int(sealed.get("required_groups", -1)) == EXPECTED_GROUPS
        and sealed.get("final_or_unassigned_access") == "forbidden"
        and comparator.get("required_basename") == M0_ROOT_BASENAME
        and comparator.get("required_file") == M0_PREDICTIONS_FILENAME
        and comparator.get("required_score_column")
        == "M0_outer_cross_fitted_log2_probability"
        and tuple(comparator.get("required_join_columns", ())) == PAIRING_KEYS
        and comparator.get("required_role")
        == "outer_cross_fitted_development_predictions_only"
        and join.get("key") == "provider_neuron_id"
        and join.get("cardinality") == "one_to_one"
        and join.get("exact_identity_set_required") is True
        and join.get("exact_group_match_required") is True
        and normalized_sizes == GROUP_SIZES
    ):
        raise ComparisonError("frozen M1 comparison contract changed")


def _validate_readiness(readiness: Mapping[str, Any]) -> None:
    gates = readiness.get("gates", {})
    access = readiness.get("access", {})
    launch = readiness.get("launch", {})
    job_ids = {str(value) for value in launch.get("slurm_job_ids", ())}
    if not (
        gates.get(POSTFLIGHT_KEY)
        == POSTFLIGHT_GATE
        and gates.get("gao_development_v3_m0_fit_v7") == M0_GATE
        and access.get("v3_m1_M0_artifact_opened") == HISTORICAL_M0_ACCESS
        and access.get("final_group_outcomes_opened") is False
        and M1_JOB_ID in job_ids
        and M0_JOB_ID in job_ids
    ):
        raise ComparisonError(
            "current independent M1 postflight or truthful historical M0 readiness is not passed"
        )


def _validate_outer_folds(summary: Mapping[str, Any], label: str) -> None:
    folds = summary.get("outer_folds")
    if not isinstance(folds, list) or len(folds) != EXPECTED_GROUPS:
        raise ComparisonError(f"{label} outer-fold accounting changed")
    for expected_group, fold in zip(GROUP_ORDER, folds):
        if not isinstance(fold, dict) or not (
            str(fold.get("held_out_group")) == expected_group
            and int(fold.get("held_out_cells", -1)) == GROUP_SIZES[expected_group]
            and int(fold.get("training_groups", -1)) == EXPECTED_GROUPS - 1
            and int(fold.get("inner_folds", -1)) == EXPECTED_GROUPS - 1
            and fold.get("selected_lambda_is_grid_boundary") is False
        ):
            raise ComparisonError(f"{label} outer-fold identity changed")


def _validate_m1_manifest(manifest: Mapping[str, Any]) -> None:
    if not (
        manifest.get("status") == "complete"
        and manifest.get("stage") == M1_STAGE
        and manifest.get("phase") == "complete"
        and str(manifest.get("SLURM_job_id")) == M1_JOB_ID
        and int(manifest.get("development_cells", -1)) == EXPECTED_ROWS
        and int(manifest.get("development_groups", -1)) == EXPECTED_GROUPS
        and int(manifest.get("completed_warm_paths", -1)) == 78
        and int(manifest.get("completed_zero_refits", -1)) == 13
        and manifest.get("M0_prediction_or_fit_artifact_opened") is False
        and manifest.get("final_or_unassigned_or_SWC_opened") is False
    ):
        raise ComparisonError("M1 terminal manifest is not the passed production run")


def _validate_m1_summary(summary: Mapping[str, Any]) -> None:
    if not (
        summary.get("status") == "pass"
        and summary.get("stage") == M1_STAGE
        and summary.get("analysis_label") == "exploratory"
        and summary.get("estimand") == ESTIMAND
        and int(summary.get("development_cells", -1)) == EXPECTED_ROWS
        and int(summary.get("development_groups", -1)) == EXPECTED_GROUPS
        and int(summary.get("target_panel_size", -1)) == PANEL_SIZE
        and int(summary.get("outer_boundary_fold_count", -1)) == 0
        and summary.get("M0_prediction_or_fit_artifact_opened") is False
        and summary.get("SWC_opened") is False
        and summary.get("final_or_unassigned_opened") is False
        and summary.get("full_refit_used_as_performance_estimate") is False
    ):
        raise ComparisonError("M1 summary is not the passed sealed production result")
    _validate_outer_folds(summary, "M1 summary")


def _validate_m0_summary(summary: Mapping[str, Any], expected_root: Path) -> None:
    if not (
        summary.get("status") == "pass"
        and summary.get("stage") == M0_STAGE
        and summary.get("analysis_label") == "exploratory"
        and summary.get("estimand") == ESTIMAND
        and int(summary.get("development_cells", -1)) == EXPECTED_ROWS
        and int(summary.get("development_groups", -1)) == EXPECTED_GROUPS
        and int(summary.get("target_panel_size", -1)) == PANEL_SIZE
        and int(summary.get("outer_boundary_fold_count", -1)) == 0
        and summary.get("zeros_interpreted_as_absence_or_negative") is False
        and summary.get("final_or_unassigned_opened") is False
        and summary.get("temperature_diagnostic_applied_to_predictions_or_scores")
        is False
        and summary.get("output_root") == str(expected_root)
    ):
        raise ComparisonError("M0 summary is not the passed frozen production result")
    _validate_outer_folds(summary, "M0 summary")


def _parse_prediction_rows(
    rows: Iterable[Mapping[str, str]],
    fieldnames: Sequence[str],
    *,
    model: str,
) -> tuple[ScoreRow, ...]:
    if model not in {"M0", "M1"}:
        raise ComparisonError("prediction model label is invalid")
    score_field = f"{model}_outer_cross_fitted_log2_probability"
    required = {
        "provider_neuron_id",
        "fMOST_brain_id",
        "K",
        score_field,
        "uniform_fixed_K_log2_probability",
    }
    if not required.issubset(set(fieldnames)):
        raise ComparisonError(f"{model} prediction columns are incomplete")

    parsed: list[ScoreRow] = []
    identifiers: set[str] = set()
    pair_keys: set[tuple[str, str]] = set()
    for row in rows:
        identifier = str(row.get("provider_neuron_id", ""))
        group = str(row.get("fMOST_brain_id", ""))
        raw_k = str(row.get("K", ""))
        if not identifier or group not in GROUP_SIZES:
            raise ComparisonError(f"{model} prediction identity is invalid")
        try:
            k = int(raw_k)
            log2_probability = float(row.get(score_field, ""))
            uniform = float(row.get("uniform_fixed_K_log2_probability", ""))
        except (TypeError, ValueError) as error:
            raise ComparisonError(f"{model} prediction numeric field is invalid") from error
        if raw_k != str(k) or not 0 <= k <= PANEL_SIZE:
            raise ComparisonError(f"{model} prediction K is invalid")
        if not math.isfinite(log2_probability) or not math.isfinite(uniform):
            raise ComparisonError(f"{model} prediction score is nonfinite")
        if log2_probability > UNIFORM_TOLERANCE:
            raise ComparisonError(f"{model} log2 probability is positive")
        expected_uniform = -math.log2(math.comb(PANEL_SIZE, k))
        if not math.isclose(
            uniform,
            expected_uniform,
            rel_tol=0.0,
            abs_tol=UNIFORM_TOLERANCE,
        ):
            raise ComparisonError(f"{model} fixed-K uniform baseline changed")
        if k in (0, PANEL_SIZE) and (
            log2_probability != 0.0 or uniform != 0.0
        ):
            raise ComparisonError(f"{model} singleton-K score is not exact zero")
        pair = (identifier, group)
        if identifier in identifiers or pair in pair_keys:
            raise ComparisonError(f"{model} prediction identity is duplicated")
        identifiers.add(identifier)
        pair_keys.add(pair)
        parsed.append(ScoreRow(identifier, group, k, log2_probability, uniform))

    counts = Counter(row.group for row in parsed)
    if len(parsed) != EXPECTED_ROWS or dict(counts) != GROUP_SIZES:
        raise ComparisonError(f"{model} prediction group vector changed")
    return tuple(parsed)


def _load_m1_artifacts(
    m1_root: Path,
    scratch_episode: Path,
) -> tuple[tuple[ScoreRow, ...], Mapping[str, Any]]:
    expected_root = scratch_episode / M1_ROOT_BASENAME
    root_fd = _open_exact_directory(m1_root, expected_root, "M1 root")
    try:
        manifest = _read_json_at(
            root_fd,
            M1_MANIFEST_FILENAME,
            "M1 terminal manifest",
        )
        _validate_m1_manifest(manifest)
        summary = _read_json_at(root_fd, M1_SUMMARY_FILENAME, "M1 summary")
        _validate_m1_summary(summary)
        rows = _read_csv_at(
            root_fd,
            M1_PREDICTIONS_FILENAME,
            "M1 predictions",
            "M1",
        )
    finally:
        os.close(root_fd)
    return rows, summary


def _load_m0_artifacts(
    m0_root: Path,
    scratch_episode: Path,
) -> tuple[tuple[ScoreRow, ...], Mapping[str, Any]]:
    expected_root = scratch_episode / M0_ROOT_BASENAME
    root_fd = _open_exact_directory(m0_root, expected_root, "M0 root")
    try:
        summary = _read_json_at(root_fd, M0_SUMMARY_FILENAME, "M0 summary")
        _validate_m0_summary(summary, expected_root)
        rows = _read_csv_at(
            root_fd,
            M0_PREDICTIONS_FILENAME,
            "M0 predictions",
            "M0",
        )
    finally:
        os.close(root_fd)
    return rows, summary


def _mean(values: Iterable[float], count: int) -> float:
    return math.fsum(values) / count


def compare_rows(
    m1_rows: Sequence[ScoreRow],
    m0_rows: Sequence[ScoreRow],
) -> Mapping[str, Any]:
    m1_by_id = {row.identifier: row for row in m1_rows}
    m0_by_id = {row.identifier: row for row in m0_rows}
    if len(m1_by_id) != len(m1_rows) or len(m0_by_id) != len(m0_rows):
        raise ComparisonError("prediction provider-neuron identity is duplicated")
    if set(m1_by_id) != set(m0_by_id):
        raise ComparisonError("M1 and M0 prediction identity sets differ")
    if {(row.identifier, row.group) for row in m1_rows} != {
        (row.identifier, row.group) for row in m0_rows
    }:
        raise ComparisonError("M1 and M0 pairing-key sets differ")

    grouped: dict[str, list[tuple[ScoreRow, ScoreRow]]] = {
        group: [] for group in GROUP_ORDER
    }
    for identifier, m1_row in m1_by_id.items():
        m0_row = m0_by_id[identifier]
        if m1_row.group != m0_row.group:
            raise ComparisonError("paired prediction group differs")
        if m1_row.k != m0_row.k:
            raise ComparisonError("paired prediction K differs")
        if not math.isclose(
            m1_row.uniform_log2_probability,
            m0_row.uniform_log2_probability,
            rel_tol=0.0,
            abs_tol=UNIFORM_TOLERANCE,
        ):
            raise ComparisonError("paired fixed-K uniform baselines differ")
        grouped[m1_row.group].append((m1_row, m0_row))

    per_group: dict[str, Any] = {}
    group_m1: list[float] = []
    group_m0: list[float] = []
    group_gain: list[float] = []
    degenerate_k0 = 0
    degenerate_k16 = 0
    for group in GROUP_ORDER:
        pairs = sorted(grouped[group], key=lambda pair: pair[0].identifier)
        if len(pairs) != GROUP_SIZES[group]:
            raise ComparisonError("paired prediction group size changed")
        mean_m1 = _mean((pair[0].log2_probability for pair in pairs), len(pairs))
        mean_m0 = _mean((pair[1].log2_probability for pair in pairs), len(pairs))
        mean_gain = _mean(
            (
                pair[0].log2_probability - pair[1].log2_probability
                for pair in pairs
            ),
            len(pairs),
        )
        if not math.isclose(
            mean_gain,
            mean_m1 - mean_m0,
            rel_tol=0.0,
            abs_tol=UNIFORM_TOLERANCE,
        ):
            raise ComparisonError("paired group score identity failed")
        group_m1.append(mean_m1)
        group_m0.append(mean_m0)
        group_gain.append(mean_gain)
        per_group[group] = {
            "cells": len(pairs),
            "mean_M1_log2_probability": mean_m1,
            "mean_M0_log2_probability": mean_m0,
            "paired_M1_minus_M0_gain_bits_per_cell": mean_gain,
        }
        degenerate_k0 += sum(pair[0].k == 0 for pair in pairs)
        degenerate_k16 += sum(pair[0].k == PANEL_SIZE for pair in pairs)

    aggregate_m1 = _mean(group_m1, EXPECTED_GROUPS)
    aggregate_m0 = _mean(group_m0, EXPECTED_GROUPS)
    aggregate_gain = _mean(group_gain, EXPECTED_GROUPS)
    if not math.isclose(
        aggregate_gain,
        aggregate_m1 - aggregate_m0,
        rel_tol=0.0,
        abs_tol=UNIFORM_TOLERANCE,
    ):
        raise ComparisonError("aggregate paired score identity failed")
    return {
        "aggregate_equal_group_mean_M1_log2_probability": aggregate_m1,
        "aggregate_equal_group_mean_M0_log2_probability": aggregate_m0,
        "aggregate_equal_group_paired_M1_minus_M0_gain_bits_per_cell": (
            aggregate_gain
        ),
        "per_group": per_group,
        "degenerate_K_counts": {
            "K_0": degenerate_k0,
            "K_16": degenerate_k16,
            "both_models_exact_zero_for_all_degenerate_rows": True,
        },
    }


def _scratch_episode() -> Path:
    scratch_value = os.environ.get("SCRATCH")
    if not scratch_value:
        raise ComparisonError("comparison requires SCRATCH")
    scratch = Path(os.path.abspath(scratch_value))
    try:
        resolved = scratch.resolve(strict=True)
    except OSError as error:
        raise ComparisonError("SCRATCH is unavailable") from error
    episode = resolved / "br_autoresearch" / "episode10_single_cell_coprojection"
    try:
        episode_fd = os.open(
            episode,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
        )
    except OSError as error:
        raise ComparisonError("EP10 Scratch parent is unavailable") from error
    os.close(episode_fd)
    return episode


def _prepare_output_root(path: Path, scratch_episode: Path) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise ComparisonError("comparison requires SLURM_JOB_ID")
    expected = scratch_episode / f"{OUTPUT_PREFIX}_job_{job_id}"
    absolute = Path(os.path.abspath(path))
    if absolute != expected:
        raise ComparisonError("comparison output is not the exact job-scoped root")
    try:
        os.lstat(absolute)
    except FileNotFoundError:
        pass
    else:
        raise ComparisonError("comparison output root already exists")
    try:
        os.mkdir(absolute, 0o770)
    except OSError as error:
        raise ComparisonError("comparison output root could not be created") from error
    return absolute


def _state_payload(
    *,
    status: str,
    phase: str,
    m0_opened: bool,
    failure: BaseException | None = None,
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "stage": COMPARISON_STAGE,
        "phase": phase,
        "SLURM_job_id": os.environ.get("SLURM_JOB_ID"),
        "M1_production_job_id": M1_JOB_ID,
        "M1_root_basename": M1_ROOT_BASENAME,
        "M0_production_job_id": M0_JOB_ID,
        "M0_root_basename": M0_ROOT_BASENAME,
        "historical_M0_artifact_access": HISTORICAL_M0_ACCESS,
        "M1_fitting_process_M0_artifact_opened": False,
        "M0_prediction_or_fit_artifact_opened": m0_opened,
        "SWC_opened": False,
        "final_or_unassigned_opened": False,
    }
    if failure is not None:
        payload["failure_class"] = type(failure).__name__
        payload["failure_message"] = str(failure)
    return payload


def run_comparison(args: argparse.Namespace) -> Mapping[str, Any]:
    output_root: Path | None = None
    phase = "pre_M0_prerequisites"
    m0_opened = False
    try:
        scratch_episode = _scratch_episode()
        contract = _read_yaml_path(
            args.contract,
            CONTRACT_PATH,
            "M1 contract",
        )
        _validate_contract(contract)
        readiness = _read_yaml_path(
            args.readiness,
            READINESS_PATH,
            "readiness record",
        )
        _validate_readiness(readiness)
        m1_rows, _ = _load_m1_artifacts(args.m1_root, scratch_episode)

        output_root = _prepare_output_root(args.output_root, scratch_episode)
        report_path = output_root / REPORT_FILENAME
        phase = "prerequisites_passed_before_M0"
        _atomic_json(
            report_path,
            _state_payload(
                status="in_progress",
                phase=phase,
                m0_opened=False,
            ),
        )

        # This durable marker is deliberately written immediately before the
        # first operation of any kind on the M0 root.
        phase = "M0_access_started"
        m0_opened = True
        _atomic_json(
            report_path,
            _state_payload(
                status="in_progress",
                phase=phase,
                m0_opened=True,
            ),
        )
        m0_rows, _ = _load_m0_artifacts(args.m0_root, scratch_episode)
        scores = compare_rows(m1_rows, m0_rows)

        phase = "complete"
        report = {
            **_state_payload(
                status="pass",
                phase=phase,
                m0_opened=True,
            ),
            "analysis_label": "exploratory",
            "estimand": ESTIMAND,
            "scoring_scale": "bits",
            "development_cells": EXPECTED_ROWS,
            "development_groups": EXPECTED_GROUPS,
            "pairing_keys": list(PAIRING_KEYS),
            **scores,
            "outer_cross_fitted_only": True,
            "full_refit_used_as_performance_estimate": False,
            "zero_semantics": (
                "no_mapped_positive_terminal_observation_recorded_not_absence"
            ),
            "permitted_interpretation": (
                "exploratory_geometry_associated_observation_pattern_"
                "structure_beyond_M0"
            ),
            "biological_promotion_or_final_claim": False,
        }
        _atomic_json(report_path, report)
        return report
    except Exception as error:
        if output_root is not None:
            try:
                _atomic_json(
                    output_root / REPORT_FILENAME,
                    _state_payload(
                        status="fail",
                        phase=phase,
                        m0_opened=m0_opened,
                        failure=error,
                    ),
                )
            except Exception as record_error:
                raise ComparisonError(
                    "comparison failed and its atomic failure record could not be written"
                ) from record_error
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--m1-root", type=Path, required=True)
    parser.add_argument("--m0-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_comparison(args)
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "failure_class": type(error).__name__,
                    "failure_message": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
