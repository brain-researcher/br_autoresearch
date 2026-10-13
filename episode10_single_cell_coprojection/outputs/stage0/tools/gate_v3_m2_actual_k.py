#!/usr/bin/env python3
"""Run the sealed EP10 v3 M2 actual-K-only residual-rank gate.

The only newly opened target-derived values are the physical cell identifier
and ``K_i_obs`` projection from the frozen development target-call file.
Conservative group and ``segment_unmapped`` come from the already-opened,
target-blind model interface.  The program constructs outcome-free M1/pair
transforms for exactly the 12 frozen leave-one-group-out training folds and
emits aggregate stable-rank diagnostics only.  It never reads a realized
positive-set identity, scores or fits an M2 model, or accepts an M0, SWC,
final, or unassigned input route.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import stat
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(os.path.abspath(__file__))
TOOLS_DIR = SCRIPT_PATH.parent
STAGE0_DIR = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m1 as m1  # noqa: E402
import fit_v3_m2 as m2  # noqa: E402
import gate_v3_m1_actual_k as sealed_k  # noqa: E402


EXPECTED_ROWS = 194
EXPECTED_GROUPS = 12
EXPECTED_FOLDS = 12
EXPECTED_RANK = 104
OUTPUT_PREFIX = "v3_m2_actual_k_gate_v1"
SUMMARY_FILENAME = "v3_m2_actual_k_gate_summary.json"
INVENTORY_FIELDS = ("cell_key", "conservative_group")
PERMITTED_ACTUAL_K_FIELDS = (
    "cell_key",
    "conservative_group",
    "segment_unmapped",
    "K_i_obs",
)
CORE_REPORT_BASENAME = "v3_m2_core_contract_qualification.json"


class M2ActualKGateError(m2.M2Error):
    """The sealed actual-K-only M2 rank gate failed closed."""


def _open_regular(path: Path, mode: str) -> Any:
    """Open one unsymlinked regular file without following its final link."""
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise M2ActualKGateError(f"required input is not a regular file: {path.name}")
    return os.fdopen(descriptor, mode, encoding=None if "b" in mode else "utf-8")


def _read_yaml_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        with _open_regular(path, "r") as handle:
            payload = m1.yaml.safe_load(handle)
    except (OSError, UnicodeError, m1.yaml.YAMLError) as error:
        raise M2ActualKGateError(f"{label} is unreadable") from error
    if not isinstance(payload, dict):
        raise M2ActualKGateError(f"{label} is not a mapping")
    return payload


def validate_gate_contract(contract_path: Path, addendum_path: Path) -> None:
    """Check only the frozen clauses that define this scientific gate."""
    contract = _read_yaml_mapping(contract_path, "M2 contract")
    addendum = _read_yaml_mapping(addendum_path, "M2 numerical addendum")
    scope = contract.get("scope_and_seals", {})
    staged = contract.get("target_free_and_staged_gates", {})
    gate = staged.get("actual_K_only_gate", {})
    resampling = contract.get("resampling_and_M1_offset", {})
    projector = contract.get("pair_design", {}).get("range_projector", {})
    prerequisites = addendum.get("access_and_completion_boundary", {}).get(
        "first_real_M2_outcome_access_requires", []
    )
    if (
        contract.get("status") != "frozen_prospectively_before_any_M2_outcome_access"
        or int(scope.get("eligible_cells", -1)) != EXPECTED_ROWS
        or int(scope.get("conservative_groups", -1)) != EXPECTED_GROUPS
        or int(scope.get("unordered_pair_coordinates", -1)) != m2.PAIR_DIMENSION
        or scope.get("final_or_unassigned_access") != "forbidden"
        or scope.get("SWC_access") != "forbidden"
        or scope.get("M0_full_refit_or_telemetry_access") != "forbidden"
        or tuple(gate.get("permitted_fields", ())) != PERMITTED_ACTUAL_K_FIELDS
        or gate.get("realized_positive_set_identity_permitted") is not False
        or int(gate.get("expected_rank", -1)) != EXPECTED_RANK
        or gate.get("expected_rank_is_solver_dimension_constant") is not False
        or gate.get("operations")
        != [
            "construct every Omega_i from K_i_obs and the frozen panel",
            "construct the outcome-free M1 and pair design transforms",
            "compute the numerical range and rank independently for every training fold",
        ]
        or resampling.get("outer_method") != "leave_one_conservative_group_out"
        or int(resampling.get("outer_folds", -1)) != EXPECTED_FOLDS
        or resampling.get("validation_group_never_enters_training_transform_or_fit")
        is not True
        or projector.get("stability_check")
        != "identical_rank_at_0.1x_1x_and_10x_base_tolerance"
        or projector.get("selected_tolerance")
        != "base_tolerance_after_the_rank_stability_check_passes"
        or int(projector.get("expected_fold_rank", -1)) != EXPECTED_RANK
        or projector.get("rank_is_computed_not_hard_coded") is not True
        or "core_target_free_qualification_pass" not in prerequisites
        or "actual_K_only_rank_104_gate_pass_in_every_required_training_fold"
        not in prerequisites
    ):
        raise M2ActualKGateError("frozen M2 actual-K gate clauses changed")


def _read_inventory(path: Path) -> dict[str, str]:
    """Read the target-free frozen cell/group inventory only."""
    try:
        with _open_regular(path, "r") as handle:
            reader = csv.DictReader(handle, dialect="excel-tab")
            if tuple(reader.fieldnames or ()) != INVENTORY_FIELDS:
                raise M2ActualKGateError("frozen M2 inventory columns changed")
            rows = list(reader)
    except (csv.Error, OSError, UnicodeError) as error:
        raise M2ActualKGateError("frozen M2 inventory is unreadable") from error
    result: dict[str, str] = {}
    for row in rows:
        cell_key = row.get("cell_key", "")
        group = row.get("conservative_group", "")
        if (
            not cell_key
            or not group
            or cell_key in result
            or not cell_key.isdigit()
            or group not in m1.GROUP_ORDER
        ):
            raise M2ActualKGateError("frozen M2 inventory identity/group is invalid")
        result[cell_key] = group
    if (
        len(result) != EXPECTED_ROWS
        or len(set(result.values())) != EXPECTED_GROUPS
        or set(result.values()) != set(m1.GROUP_ORDER)
    ):
        raise M2ActualKGateError("frozen M2 inventory size or group set changed")
    return result


def _validate_interface_inventory(
    cells: Sequence[m1.Cell], inventory: Mapping[str, str]
) -> None:
    identifiers = [cell.identifier for cell in cells]
    if (
        len(cells) != EXPECTED_ROWS
        or len(set(identifiers)) != EXPECTED_ROWS
        or set(identifiers) != set(inventory)
        or any(inventory[cell.identifier] != cell.group for cell in cells)
        or any(cell.group not in m1.GROUP_ORDER for cell in cells)
        or any(cell.segment_unmapped not in (0, 1) for cell in cells)
    ):
        raise M2ActualKGateError(
            "target-free interface does not match the frozen M2 cell inventory"
        )


def _outer_training_folds(cells: Sequence[m1.Cell]) -> tuple[tuple[int, ...], ...]:
    """Return exactly the 12 contracted LOGO training sets, never a 13th refit."""
    if len(cells) != EXPECTED_ROWS:
        raise M2ActualKGateError("M2 actual-K gate requires exactly 194 cells")
    groups = tuple(cell.group for cell in cells)
    if set(groups) != set(m1.GROUP_ORDER):
        raise M2ActualKGateError("M2 actual-K gate group inventory changed")
    folds = tuple(
        tuple(index for index, group in enumerate(groups) if group != held_out)
        for held_out in m1.GROUP_ORDER
    )
    if (
        len(folds) != EXPECTED_FOLDS
        or len(set(folds)) != EXPECTED_FOLDS
        or any(not fold or len(fold) >= len(cells) for fold in folds)
        or any(
            held_out in {cells[index].group for index in fold}
            for held_out, fold in zip(m1.GROUP_ORDER, folds, strict=True)
        )
    ):
        raise M2ActualKGateError("contracted LOGO training-fold construction failed")
    return folds


def _aggregate_fold_diagnostics(diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute pass from aggregate-safe numerical diagnostics only."""
    if len(diagnostics) != EXPECTED_FOLDS:
        raise M2ActualKGateError("M2 rank gate did not evaluate exactly 12 folds")
    ranks: list[int] = []
    ranks_low: list[int] = []
    ranks_high: list[int] = []
    symmetry: list[float] = []
    idempotence: list[float] = []
    orthogonality: list[float] = []
    for item in diagnostics:
        stability = item.get("rank_at_tau_over_10_tau_ten_tau", {})
        if set(stability) != {"tau_over_10", "tau", "ten_tau"}:
            raise M2ActualKGateError("fold rank-stability inventory changed")
        rank_low = int(stability["tau_over_10"])
        rank = int(stability["tau"])
        rank_high = int(stability["ten_tau"])
        if (
            item.get("rank_stable") is not True
            or int(item.get("rank", -1)) != rank
            or int(item.get("expected_rank", -1)) != EXPECTED_RANK
            or item.get("expected_rank_match") is not True
            or rank_low != EXPECTED_RANK
            or rank != EXPECTED_RANK
            or rank_high != EXPECTED_RANK
        ):
            raise M2ActualKGateError("a required fold failed stable rank 104")
        ranks_low.append(rank_low)
        ranks.append(rank)
        ranks_high.append(rank_high)
        symmetry.append(float(item["range_projector_symmetry_maximum_absolute_error"]))
        idempotence.append(float(item["range_projector_idempotence_maximum_absolute_error"]))
        orthogonality.append(float(item["G_FH_projection_maximum_absolute_orthogonality_error"]))
    values = ranks_low + ranks + ranks_high + symmetry + idempotence + orthogonality
    if not all(math.isfinite(float(value)) for value in values):
        raise M2ActualKGateError("aggregate M2 actual-K diagnostics are nonfinite")
    return {
        "status": "pass",
        "training_folds_evaluated": EXPECTED_FOLDS,
        "training_folds_passed": EXPECTED_FOLDS,
        "fold_definition": "12_leave_one_conservative_group_out_training_sets",
        "full_development_thirteenth_gate_added": False,
        "expected_rank": EXPECTED_RANK,
        "computed_rank_at_tau_over_10_unique": sorted(set(ranks_low)),
        "computed_rank_at_tau_unique": sorted(set(ranks)),
        "computed_rank_at_ten_tau_unique": sorted(set(ranks_high)),
        "rank_stable_in_every_fold": True,
        "rank_computed_not_hard_coded": True,
        "maximum_projector_symmetry_error": max(symmetry),
        "maximum_projector_idempotence_error": max(idempotence),
        "maximum_projection_orthogonality_error": max(orthogonality),
        "fold_level_rows_emitted": False,
        "cell_ids_groups_or_K_values_emitted": False,
    }


def _compute_rank_gate(
    cells: Sequence[m1.Cell], pair_geometry: m1.PairGeometry
) -> dict[str, Any]:
    candidates = m1.CandidateLibrary()
    geometry = m1.GeometryEngine(candidates, pair_geometry)
    moments = tuple(geometry.uniform_moments(cell) for cell in cells)
    pair_features = m2.PairFeatureLibrary(candidates)
    diagnostics: list[Mapping[str, Any]] = []
    for indexes in _outer_training_folds(cells):
        transform, _ = m2.fit_residual_transform(
            cells,
            moments,
            candidates,
            geometry,
            pair_features,
            indexes,
            expected_rank=EXPECTED_RANK,
        )
        diagnostics.append(transform.diagnostics)
    return _aggregate_fold_diagnostics(diagnostics)


def _require_core_prerequisite(core_report_argument: Path) -> str:
    """Require the exact passed synthetic-core artifact without broad ledgers."""
    absolute = Path(os.path.abspath(core_report_argument))
    match = re.fullmatch(
        r"v3_m2_core_contract_qualification_job_([0-9]+)", absolute.parent.name
    )
    if match is None:
        raise M2ActualKGateError("M2 comprehensive report root name changed")
    core_job_id = match.group(1)
    basename = absolute.parent.name
    core_root = m1._require_exact_scratch_child(
        absolute.parent, basename, "M2 comprehensive qualification root"
    )
    core_report = core_root / CORE_REPORT_BASENAME
    if core_report != absolute:
        raise M2ActualKGateError("M2 comprehensive report path changed")
    try:
        with _open_regular(core_report, "r") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError, UnicodeError) as error:
        raise M2ActualKGateError("M2 comprehensive report is unreadable") from error
    execution = payload.get("execution", {}) if isinstance(payload, dict) else {}
    inventory = payload.get("gate_inventory", {}) if isinstance(payload, dict) else {}
    if not (
        payload.get("status") == "pass"
        and payload.get("overall_gate_status") == "pass"
        and str(execution.get("slurm_job_id")) == core_job_id
        and inventory.get("all_required_gates_present_and_passed") is True
        and int(inventory.get("required_gate_count", -1)) == 8
        and payload.get("real_target_membership_or_M2_effects_opened") is False
        and payload.get("actual_K_opened") is False
        and payload.get("M0_full_refit_or_telemetry_opened") is False
        and payload.get("final_or_unassigned_or_SWC_opened") is False
        and payload.get("controller_replay_qualification_included") is False
    ):
        raise M2ActualKGateError("M2 comprehensive report is not passed and sealed")
    return core_job_id


def _prepare_output_root(path: Path) -> Path:
    scratch_value = os.environ.get("SCRATCH")
    job_id = os.environ.get("SLURM_JOB_ID")
    if not scratch_value or not job_id:
        raise M2ActualKGateError("actual-K gate requires SCRATCH and SLURM_JOB_ID")
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
        raise M2ActualKGateError("actual-K output is not the job-scoped Scratch child")
    try:
        absolute.mkdir(mode=0o770)
    except FileExistsError as error:
        raise M2ActualKGateError("actual-K output root already exists") from error
    return absolute


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    directory_fd = os.open(
        path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    temporary_name = f".{path.name}.tmp.{os.getpid()}"
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
    validate_gate_contract(args.contract, args.addendum)
    core_job_id = _require_core_prerequisite(args.core_report)
    inventory = _read_inventory(args.inventory)
    output_root = _prepare_output_root(args.output_root)
    summary_path = output_root / SUMMARY_FILENAME
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "status": "running",
        "stage": "v3_M2_actual_K_only_residual_rank_gate_v1",
        "analysis_label": "exploratory_development_only",
        "rows": EXPECTED_ROWS,
        "groups": EXPECTED_GROUPS,
        "physical_pair_coordinates": m2.PAIR_DIMENSION,
        "SLURM_job_id": os.environ["SLURM_JOB_ID"],
        "qualified_core_job_id": core_job_id,
        "new_target_derived_fields_opened": "cell_key_and_physical_K_i_obs_only",
        "target_blind_interface_fields_reused": (
            "cell_key_conservative_group_segment_unmapped_and_frozen_M1_covariates"
        ),
        "actual_K_opened": False,
        "cell_ids_groups_segment_flags_or_K_values_emitted": False,
        "realized_positive_set_membership_opened": False,
        "candidate_scores_or_M2_fit_computed": False,
        "M0_full_refit_or_telemetry_opened": False,
        "SWC_opened": False,
        "final_or_unassigned_opened": False,
    }
    try:
        target_blind_cells, pair_geometry = m1.load_target_free_real_shape(
            args.interface_root, args.geometry_root
        )
        _validate_interface_inventory(target_blind_cells, inventory)
        payload["actual_K_opened"] = True
        projected, counts = sealed_k._read_target_inputs(
            args.target_root, set(inventory)
        )
        if (
            counts.total_rows != sealed_k.EXPECTED_TARGET_ROWS
            or counts.matched_rows != EXPECTED_ROWS
            or counts.extra_rows != sealed_k.EXPECTED_EXCLUDED_ROWS
        ):
            raise M2ActualKGateError("sealed physical K projection counts changed")
        cells = sealed_k._align_actual_k(target_blind_cells, projected)
        _validate_interface_inventory(cells, inventory)
        payload["rank_gate"] = _compute_rank_gate(cells, pair_geometry)
        payload["elapsed_wall_seconds"] = time.perf_counter() - started
        payload["status"] = payload["rank_gate"]["status"]
        _atomic_json(summary_path, payload)
        return payload
    except Exception as error:
        payload.update(
            {
                "status": "fail",
                "failure_class": type(error).__name__,
                "failure_code": "M2_actual_K_rank_gate_failed_closed",
                "elapsed_wall_seconds": time.perf_counter() - started,
            }
        )
        _atomic_json(summary_path, payload)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--addendum", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--core-report", type=Path, required=True)
    parser.add_argument("--interface-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_actual_k_gate(args)
    except Exception as error:
        print(
            f"ERROR: M2 actual-K rank gate failed closed ({type(error).__name__})",
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "stage": result["stage"],
                "training_folds_evaluated": result["rank_gate"][
                    "training_folds_evaluated"
                ],
                "computed_rank_at_tau_unique": result["rank_gate"][
                    "computed_rank_at_tau_unique"
                ],
                "actual_K_opened": result["actual_K_opened"],
                "realized_positive_set_membership_opened": False,
                "candidate_scores_or_M2_fit_computed": False,
                "final_or_unassigned_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
