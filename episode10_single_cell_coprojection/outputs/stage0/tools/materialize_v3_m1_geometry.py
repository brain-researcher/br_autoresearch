#!/usr/bin/env python3
"""Materialize target-blind atlas geometry for the EP10 v3 M1 reference.

The production interface accepts the passed 194-row target-blind model
interface, the frozen anatomical vocabulary, and an EP10-local atlas copy.  It
has no target-call, SWC, final, or unassigned-data argument.  Realized axon
routes and terminal geometry are deliberately outside this program.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PANEL = (
    "allen_993_MOs__ipsilateral",
    "allen_993_MOs__contralateral",
    "allen_453_SS__ipsilateral",
    "allen_453_SS__contralateral",
    "allen_31_ACA__ipsilateral",
    "allen_31_ACA__contralateral",
    "allen_477_STR__ipsilateral",
    "allen_477_STR__contralateral",
    "allen_549_TH__ipsilateral",
    "allen_549_TH__contralateral",
    "allen_313_MB__ipsilateral",
    "allen_313_MB__contralateral",
    "allen_771_P__ipsilateral",
    "allen_771_P__contralateral",
    "allen_354_MY__ipsilateral",
    "allen_354_MY__contralateral",
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
HEMISPHERE_COUNTS = {"right": 158, "left": 36}
INTERFACE_FIELDS = (
    "provider_neuron_id",
    "file",
    "fMOST_brain_id",
    "provider_sample_id",
    "source_acronym",
    "provider_hemisphere",
    "soma_x_ccfv3_um",
    "soma_y_ccfv3_um",
    "soma_z_ccfv3_um",
    "segment_unmapped",
    "terminal_unmapped_report_only",
    "mapping_overlap_class_report_only",
)
ANCHOR_FIELDS = (
    "coordinate_index",
    "coordinate_id",
    "target_index",
    "target_id",
    "allen_structure_id",
    "laterality",
    "physical_left_mask_voxels",
    "physical_right_mask_voxels",
    "physical_left_centroid_ap_mm",
    "physical_left_centroid_dv_mm",
    "physical_left_centroid_ml_mm",
    "physical_right_centroid_ap_mm",
    "physical_right_centroid_dv_mm",
    "physical_right_centroid_ml_mm",
    "symmetric_anchor_ap_mm",
    "symmetric_anchor_dv_mm",
    "symmetric_anchor_lateral_depth_mm",
    "canonical_target_lateral_mm",
)
CELL_TARGET_FIELDS = (
    "provider_neuron_id",
    "fMOST_brain_id",
    "coordinate_index",
    "coordinate_id",
    "target_id",
    "laterality",
    "soma_physical_hemisphere",
    "soma_ap_mm",
    "soma_dv_mm",
    "soma_ml_mm",
    "soma_lateral_depth_mm",
    "symmetric_target_ap_mm",
    "symmetric_target_dv_mm",
    "symmetric_target_lateral_depth_mm",
    "canonical_target_lateral_mm",
    "delta_ap_mm",
    "delta_dv_mm",
    "delta_lateral_mm",
    "soma_target_distance_mm",
)
PAIR_FIELDS = (
    "coordinate_index_a",
    "coordinate_id_a",
    "target_id_a",
    "laterality_a",
    "coordinate_index_b",
    "coordinate_id_b",
    "target_id_b",
    "laterality_b",
    "delta_ap_mm",
    "delta_dv_mm",
    "delta_lateral_mm",
    "target_pair_distance_mm",
    "cross_side",
    "homotopic",
)
MODEL_FEATURES = (
    "mean_soma_target_distance_mm",
    "sd_soma_target_distance_mm",
    "mean_target_pair_distance_mm",
    "cross_side_pair_fraction",
    "homotopic_pair_fraction",
    "mst_mean_edge_length_mm",
)
EPISODE_DIRECTORY = "episode10_single_cell_coprojection"
OUTPUT_DIRECTORY = "v3_m1_geometry_v1"
MIDLINE_MM = 5.7


class GeometryError(RuntimeError):
    """A frozen geometry or sealed-input invariant was violated."""


@dataclass(frozen=True)
class Cell:
    provider_neuron_id: str
    filename: str
    group: str
    sample_id: str
    source_acronym: str
    hemisphere: str
    soma_mm: tuple[float, float, float]


@dataclass(frozen=True)
class TargetAnchor:
    coordinate_index: int
    coordinate_id: str
    target_index: int
    target_id: str
    anchor_structure_id: int
    laterality: str
    left_voxels: int
    right_voxels: int
    left_centroid_mm: tuple[float, float, float]
    right_centroid_mm: tuple[float, float, float]
    symmetric_ap_mm: float
    symmetric_dv_mm: float
    symmetric_lateral_depth_mm: float

    @property
    def canonical_target_lateral_mm(self) -> float:
        if self.laterality == "ipsilateral":
            return self.symmetric_lateral_depth_mm
        if self.laterality == "contralateral":
            return -self.symmetric_lateral_depth_mm
        raise GeometryError(f"invalid target laterality: {self.laterality!r}")


@dataclass(frozen=True)
class PairGeometry:
    coordinate_index_a: int
    coordinate_id_a: str
    target_id_a: str
    laterality_a: str
    coordinate_index_b: int
    coordinate_id_b: str
    target_id_b: str
    laterality_b: str
    delta_ap_mm: float
    delta_dv_mm: float
    delta_lateral_mm: float
    distance_mm: float
    cross_side: int
    homotopic: int


def _finite(values: Iterable[float], label: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise GeometryError(f"nonfinite {label}")
    return result


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_unsymlinked_path(
    path: Path, label: str, *, must_exist: bool
) -> Path:
    absolute = _absolute_path(path)
    resolved = absolute.resolve(strict=must_exist)
    if absolute != resolved:
        raise GeometryError(f"{label} may not use symlinked path components")
    return resolved


def _stage0_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _require_stage0_file(path: Path, basename: str, label: str) -> Path:
    expected = _stage0_root() / basename
    resolved = _require_unsymlinked_path(path, label, must_exist=True)
    expected = _require_unsymlinked_path(expected, label, must_exist=True)
    if resolved != expected or not resolved.is_file():
        raise GeometryError(f"{label} must be the frozen EP10 stage0 {basename}")
    return resolved


def _load_contract(path: Path) -> Mapping[str, Any]:
    path = _require_stage0_file(path, "v3_m1_geometry_contract.yaml", "M1 contract")
    try:
        import yaml  # type: ignore
    except ImportError as error:
        raise GeometryError("real execution requires PyYAML") from error
    with path.open("r", encoding="utf-8") as handle:
        contract = yaml.safe_load(handle)
    if not isinstance(contract, dict):
        raise GeometryError("M1 geometry contract is not a mapping")
    if contract.get("status") != "frozen_before_real_m1_geometry_materialization":
        raise GeometryError("M1 geometry contract is not frozen")
    scope = contract.get("sealed_scope", {})
    if (
        scope.get("role") != "development_only"
        or int(scope.get("required_interface_rows", -1)) != 194
        or scope.get("final_or_unassigned_access") != "forbidden"
        or scope.get("target_call_or_swc_input") != "forbidden"
        or scope.get("realized_axon_route_or_terminal_geometry") != "forbidden"
    ):
        raise GeometryError("sealed M1 geometry scope changed")
    panel = tuple(contract.get("fixed_panel", {}).get("ordered_coordinate_ids", ()))
    if panel != PANEL:
        raise GeometryError("frozen 16-coordinate geometry panel changed")
    if tuple(contract.get("model_set_features", {}).get("ordered_features", ())) != MODEL_FEATURES:
        raise GeometryError("M1 set-feature order changed")
    path_seal = contract.get("path_and_output_seal", {})
    if (
        path_seal.get("exact_stage0_files")
        != {
            "m1_contract": "v3_m1_geometry_contract.yaml",
            "observation_base_contract": "projection_observation_contract.yaml",
            "vocabulary": "allen_target_vocabulary.csv",
        }
        or path_seal.get("scratch_episode_directory")
        != f"br_autoresearch/{EPISODE_DIRECTORY}"
        or path_seal.get("scratch_input_files_and_path_components_may_be_symlinks")
        is not False
        or path_seal.get("stage0_input_files_and_path_components_may_be_symlinks")
        is not False
        or path_seal.get("output_root")
        != {
            "location": "direct_child_of_EP10_local_Scratch_root",
            "required_basename": OUTPUT_DIRECTORY,
            "absent_or_empty_before_execute": True,
            "overlap_with_input_roots": "forbidden",
            "path_components_or_output_files_may_be_symlinks": False,
        }
    ):
        raise GeometryError("M1 path/output seal changed")
    interface_contract = contract.get("required_inputs", {}).get("model_interface", {})
    frozen_ids = tuple(
        value.strip()
        for value in str(interface_contract.get("ordered_provider_neuron_ids_csv", "")).split(",")
        if value.strip()
    )
    if len(frozen_ids) != 194 or len(set(frozen_ids)) != 194:
        raise GeometryError("frozen model-interface identity vector changed")
    return contract


def _ep10_scratch_root() -> Path:
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise GeometryError("SCRATCH is required for sealed real execution")
    scratch_root = _require_unsymlinked_path(
        Path(scratch), "SCRATCH root", must_exist=True
    )
    episode_root = _require_unsymlinked_path(
        scratch_root / "br_autoresearch" / EPISODE_DIRECTORY,
        "EP10 Scratch root",
        must_exist=True,
    )
    if not episode_root.is_dir():
        raise GeometryError("EP10 Scratch root is not a directory")
    return episode_root


def _require_below(
    path: Path, root: Path, label: str, *, must_exist: bool = True
) -> Path:
    resolved = _require_unsymlinked_path(path, label, must_exist=must_exist)
    if resolved == root or root not in resolved.parents:
        raise GeometryError(f"{label} is outside the sealed EP10 Scratch root")
    return resolved


def _assert_ep10_local_assets(
    annotation: Path, ontology: Path, contract: Mapping[str, Any]
) -> tuple[Path, Path]:
    scratch_root = _ep10_scratch_root()
    annotation = _require_below(annotation, scratch_root, "annotation")
    ontology = _require_below(ontology, scratch_root, "ontology")
    if not annotation.is_file() or not ontology.is_file():
        raise GeometryError("atlas inputs must be existing regular files")
    if annotation.name != "annotation_25_ccf_2017.nrrd":
        raise GeometryError("unexpected annotation basename")
    if ontology.name != "structure_graph_1.json":
        raise GeometryError("unexpected ontology basename")
    if annotation.parent != ontology.parent:
        raise GeometryError("annotation and ontology must use one EP10-local atlas directory")
    expected_parent = str(
        contract.get("required_inputs", {})
        .get("atlas", {})
        .get("required_parent_directory", "")
    )
    if not expected_parent or annotation.parent.name != expected_parent:
        raise GeometryError("atlas is not the frozen EP10-local steward copy")
    return annotation, ontology


def _load_interface(
    path: Path, contract: Mapping[str, Any], midline_mm: float
) -> list[Cell]:
    interface_contract = contract.get("required_inputs", {}).get("model_interface", {})
    path = _require_below(path, _ep10_scratch_root(), "model interface")
    if not path.is_file():
        raise GeometryError("model-interface input is not a regular file")
    if path.name != "v3_model_interface.csv":
        raise GeometryError("model-interface input must be v3_model_interface.csv")
    if path.parent.name != interface_contract.get("required_parent_directory"):
        raise GeometryError("model interface is not the passed frozen preflight root")
    summary_path = _require_below(
        path.parent / str(interface_contract.get("required_summary_file", "")),
        _ep10_scratch_root(),
        "model-interface summary",
    )
    if summary_path.parent != path.parent or not summary_path.is_file():
        raise GeometryError("model-interface summary is not a sibling regular file")
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except json.JSONDecodeError as error:
        raise GeometryError("model-interface summary is not valid JSON") from error
    if not isinstance(summary, dict):
        raise GeometryError("model-interface summary is not an object")
    if (
        summary.get("status") != interface_contract.get("required_summary_status")
        or summary.get("stage") != interface_contract.get("required_summary_stage")
        or int(summary.get("interface_rows", -1)) != 194
        or tuple(summary.get("interface_columns", ())) != INTERFACE_FIELDS
        or summary.get("eligible_cells_by_group") != GROUP_SIZES
        or summary.get("final_or_unassigned_files_opened") is not False
        or summary.get("projection_outcomes_opened_or_emitted") is not False
        or summary.get("target_vocabulary_or_calls_loaded") is not False
        or summary.get("raw_lengths_or_terminal_counts_emitted") is not False
    ):
        raise GeometryError("model-interface preflight summary is not the passed sealed result")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != INTERFACE_FIELDS:
            raise GeometryError("v3 model-interface columns/order changed")
        rows = list(reader)
    if len(rows) != 194:
        raise GeometryError(f"v3 model interface has {len(rows)} rows, expected 194")
    expected_ids = tuple(
        value.strip()
        for value in str(interface_contract["ordered_provider_neuron_ids_csv"]).split(",")
        if value.strip()
    )
    observed_ids = tuple(row["provider_neuron_id"].strip() for row in rows)
    if observed_ids != expected_ids:
        raise GeometryError("model-interface ordered development identity vector changed")

    cells: list[Cell] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        neuron_id = row["provider_neuron_id"].strip()
        if not neuron_id or neuron_id in seen:
            raise GeometryError(f"missing or duplicate neuron ID at interface row {row_number}")
        seen.add(neuron_id)
        hemisphere = row["provider_hemisphere"].strip().casefold()
        if hemisphere not in {"left", "right"}:
            raise GeometryError(f"invalid hemisphere at interface row {row_number}")
        try:
            soma_um = _finite(
                (
                    float(row["soma_x_ccfv3_um"]),
                    float(row["soma_y_ccfv3_um"]),
                    float(row["soma_z_ccfv3_um"]),
                ),
                f"soma at interface row {row_number}",
            )
        except ValueError as error:
            raise GeometryError(f"nonnumeric soma at interface row {row_number}") from error
        observed_hemisphere = "left" if soma_um[2] / 1000.0 < midline_mm else "right"
        if observed_hemisphere != hemisphere:
            raise GeometryError(f"soma/hemisphere mismatch at interface row {row_number}")
        if row["segment_unmapped"] not in {"0", "1"}:
            raise GeometryError(f"invalid segment mapping bit at interface row {row_number}")
        if row["terminal_unmapped_report_only"] not in {"0", "1"}:
            raise GeometryError(f"invalid terminal mapping bit at interface row {row_number}")
        cells.append(
            Cell(
                provider_neuron_id=neuron_id,
                filename=row["file"],
                group=row["fMOST_brain_id"],
                sample_id=row["provider_sample_id"],
                source_acronym=row["source_acronym"],
                hemisphere=hemisphere,
                soma_mm=tuple(value / 1000.0 for value in soma_um),  # type: ignore[arg-type]
            )
        )

    observed_groups = dict(sorted(Counter(cell.group for cell in cells).items()))
    if observed_groups != GROUP_SIZES:
        raise GeometryError(f"model-interface group vector changed: {observed_groups}")
    observed_hemispheres = dict(Counter(cell.hemisphere for cell in cells))
    if observed_hemispheres != HEMISPHERE_COUNTS:
        raise GeometryError(
            f"model-interface hemisphere vector changed: {observed_hemispheres}"
        )
    return cells


def derive_atlas_anchors(
    annotation: Any,
    spacing_um: Sequence[float],
    midline_um: float,
    ontology: Any,
    coordinates: Sequence[Any],
    *,
    chunk_ap_voxels: int = 16,
) -> list[TargetAnchor]:
    """Derive equal-side symmetrized anchors from descendant voxel masks."""
    try:
        import numpy as np  # type: ignore
    except ImportError as error:
        raise GeometryError("real anchor materialization requires NumPy") from error

    if getattr(annotation, "ndim", None) != 3:
        raise GeometryError("annotation must be three-dimensional")
    spacing = _finite(spacing_um, "atlas spacing")
    if len(spacing) != 3 or any(value <= 0.0 for value in spacing):
        raise GeometryError("atlas spacing must contain three positive values")
    if not math.isfinite(midline_um):
        raise GeometryError("atlas midline is nonfinite")
    if tuple(coordinate.coordinate_id for coordinate in coordinates) != PANEL:
        raise GeometryError("loaded primary coordinates differ from the frozen panel")

    by_anchor: dict[int, list[Any]] = {}
    anchor_order: list[int] = []
    for coordinate in coordinates:
        anchor = int(coordinate.anchor_structure_id)
        if anchor not in by_anchor:
            by_anchor[anchor] = []
            anchor_order.append(anchor)
        by_anchor[anchor].append(coordinate)
    if len(anchor_order) != 8:
        raise GeometryError(f"primary panel has {len(anchor_order)} anatomical anchors")
    for anchor, pair in by_anchor.items():
        if len(pair) != 2 or {item.laterality for item in pair} != {
            "ipsilateral",
            "contralateral",
        }:
            raise GeometryError(f"anatomical anchor {anchor} lacks its laterality pair")

    sparse_structure_to_anchor: dict[int, int] = {}
    for structure_id in ontology.parent_by_id:
        matches = [
            position
            for position, anchor in enumerate(anchor_order)
            if ontology.is_descendant_or_self(int(structure_id), anchor)
        ]
        if len(matches) > 1:
            raise GeometryError(f"ontology node {structure_id} enters multiple target masks")
        if matches:
            sparse_structure_to_anchor[int(structure_id)] = matches[0]
    if not sparse_structure_to_anchor:
        raise GeometryError("no ontology nodes enter the frozen target masks")
    # Allen structure IDs are sparse and may be very large.  Keep only the
    # descendants of the eight frozen targets and use a sorted sparse remap;
    # never allocate an array indexed by the maximum ontology ID.
    structure_ids = np.asarray(sorted(sparse_structure_to_anchor), dtype=np.int64)
    structure_targets = np.asarray(
        [sparse_structure_to_anchor[int(value)] for value in structure_ids],
        dtype=np.int16,
    )

    counts = np.zeros(16, dtype=np.int64)
    ap_sum = np.zeros(16, dtype=np.float64)
    dv_sum = np.zeros(16, dtype=np.float64)
    ml_sum = np.zeros(16, dtype=np.float64)
    for start in range(0, int(annotation.shape[0]), chunk_ap_voxels):
        stop = min(start + chunk_ap_voxels, int(annotation.shape[0]))
        block = np.asarray(annotation[start:stop])
        if block.size == 0:
            continue
        flat = block.reshape(-1).astype(np.int64, copy=False)
        insertion = np.searchsorted(structure_ids, flat)
        candidates = np.flatnonzero(insertion < structure_ids.size)
        if candidates.size == 0:
            continue
        candidates = candidates[
            structure_ids[insertion[candidates]] == flat[candidates]
        ]
        if candidates.size == 0:
            continue
        target_positions = structure_targets[insertion[candidates]].astype(
            np.int64, copy=False
        )
        local_ap, dv, ml = np.unravel_index(candidates, block.shape)
        ml_um = (ml.astype(np.float64) + 0.5) * spacing[2]
        side = (ml_um >= midline_um).astype(np.int64)
        codes = target_positions * 2 + side
        counts += np.bincount(codes, minlength=16)
        ap_sum += np.bincount(
            codes,
            weights=(local_ap.astype(np.float64) + start + 0.5) * spacing[0] / 1000.0,
            minlength=16,
        )
        dv_sum += np.bincount(
            codes,
            weights=(dv.astype(np.float64) + 0.5) * spacing[1] / 1000.0,
            minlength=16,
        )
        ml_sum += np.bincount(codes, weights=ml_um / 1000.0, minlength=16)

    stats: dict[int, Mapping[str, Any]] = {}
    midline_mm = midline_um / 1000.0
    for position, anchor_structure_id in enumerate(anchor_order):
        left_index = 2 * position
        right_index = left_index + 1
        left_count = int(counts[left_index])
        right_count = int(counts[right_index])
        if left_count <= 0 or right_count <= 0:
            raise GeometryError(
                f"target mask {anchor_structure_id} lacks a physical atlas side: "
                f"left={left_count} right={right_count}"
            )
        left = _finite(
            (
                ap_sum[left_index] / left_count,
                dv_sum[left_index] / left_count,
                ml_sum[left_index] / left_count,
            ),
            f"left centroid for {anchor_structure_id}",
        )
        right = _finite(
            (
                ap_sum[right_index] / right_count,
                dv_sum[right_index] / right_count,
                ml_sum[right_index] / right_count,
            ),
            f"right centroid for {anchor_structure_id}",
        )
        left_depth = midline_mm - left[2]
        right_depth = right[2] - midline_mm
        if left_depth <= 0.0 or right_depth <= 0.0:
            raise GeometryError(f"target {anchor_structure_id} has an invalid physical side")
        symmetric = _finite(
            (
                0.5 * (left[0] + right[0]),
                0.5 * (left[1] + right[1]),
                0.5 * (left_depth + right_depth),
            ),
            f"symmetric centroid for {anchor_structure_id}",
        )
        if symmetric[2] <= 0.0:
            raise GeometryError(f"target {anchor_structure_id} has nonpositive depth")
        stats[anchor_structure_id] = {
            "left_count": left_count,
            "right_count": right_count,
            "left": left,
            "right": right,
            "symmetric": symmetric,
        }

    anchors: list[TargetAnchor] = []
    for coordinate in coordinates:
        values = stats[int(coordinate.anchor_structure_id)]
        symmetric = values["symmetric"]
        anchors.append(
            TargetAnchor(
                coordinate_index=int(coordinate.index),
                coordinate_id=str(coordinate.coordinate_id),
                target_index=int(coordinate.target_index),
                target_id=str(coordinate.target_id),
                anchor_structure_id=int(coordinate.anchor_structure_id),
                laterality=str(coordinate.laterality),
                left_voxels=int(values["left_count"]),
                right_voxels=int(values["right_count"]),
                left_centroid_mm=values["left"],
                right_centroid_mm=values["right"],
                symmetric_ap_mm=float(symmetric[0]),
                symmetric_dv_mm=float(symmetric[1]),
                symmetric_lateral_depth_mm=float(symmetric[2]),
            )
        )
    if len(anchors) != 16:
        raise GeometryError("anchor construction did not produce 16 coordinates")
    return anchors


def anchor_rows(anchors: Sequence[TargetAnchor]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for anchor in anchors:
        row: dict[str, object] = {
            "coordinate_index": anchor.coordinate_index,
            "coordinate_id": anchor.coordinate_id,
            "target_index": anchor.target_index,
            "target_id": anchor.target_id,
            "allen_structure_id": anchor.anchor_structure_id,
            "laterality": anchor.laterality,
            "physical_left_mask_voxels": anchor.left_voxels,
            "physical_right_mask_voxels": anchor.right_voxels,
            "physical_left_centroid_ap_mm": anchor.left_centroid_mm[0],
            "physical_left_centroid_dv_mm": anchor.left_centroid_mm[1],
            "physical_left_centroid_ml_mm": anchor.left_centroid_mm[2],
            "physical_right_centroid_ap_mm": anchor.right_centroid_mm[0],
            "physical_right_centroid_dv_mm": anchor.right_centroid_mm[1],
            "physical_right_centroid_ml_mm": anchor.right_centroid_mm[2],
            "symmetric_anchor_ap_mm": anchor.symmetric_ap_mm,
            "symmetric_anchor_dv_mm": anchor.symmetric_dv_mm,
            "symmetric_anchor_lateral_depth_mm": anchor.symmetric_lateral_depth_mm,
            "canonical_target_lateral_mm": anchor.canonical_target_lateral_mm,
        }
        if tuple(row) != ANCHOR_FIELDS:
            raise GeometryError("atlas-anchor output schema changed")
        rows.append(row)
    return rows


def build_cell_target_geometry(
    cells: Sequence[Cell], anchors: Sequence[TargetAnchor], midline_mm: float
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cell in cells:
        soma_ap, soma_dv, soma_ml = _finite(cell.soma_mm, f"soma {cell.provider_neuron_id}")
        physical_side = "left" if soma_ml < midline_mm else "right"
        if physical_side != cell.hemisphere:
            raise GeometryError(f"soma/hemisphere mismatch for {cell.provider_neuron_id}")
        soma_depth = abs(soma_ml - midline_mm)
        for anchor in anchors:
            delta_ap = anchor.symmetric_ap_mm - soma_ap
            delta_dv = anchor.symmetric_dv_mm - soma_dv
            delta_lateral = anchor.canonical_target_lateral_mm - soma_depth
            distance = math.sqrt(
                delta_ap * delta_ap
                + delta_dv * delta_dv
                + delta_lateral * delta_lateral
            )
            _finite(
                (soma_depth, delta_ap, delta_dv, delta_lateral, distance),
                f"cell-target geometry for {cell.provider_neuron_id}",
            )
            row: dict[str, object] = {
                "provider_neuron_id": cell.provider_neuron_id,
                "fMOST_brain_id": cell.group,
                "coordinate_index": anchor.coordinate_index,
                "coordinate_id": anchor.coordinate_id,
                "target_id": anchor.target_id,
                "laterality": anchor.laterality,
                "soma_physical_hemisphere": cell.hemisphere,
                "soma_ap_mm": soma_ap,
                "soma_dv_mm": soma_dv,
                "soma_ml_mm": soma_ml,
                "soma_lateral_depth_mm": soma_depth,
                "symmetric_target_ap_mm": anchor.symmetric_ap_mm,
                "symmetric_target_dv_mm": anchor.symmetric_dv_mm,
                "symmetric_target_lateral_depth_mm": anchor.symmetric_lateral_depth_mm,
                "canonical_target_lateral_mm": anchor.canonical_target_lateral_mm,
                "delta_ap_mm": delta_ap,
                "delta_dv_mm": delta_dv,
                "delta_lateral_mm": delta_lateral,
                "soma_target_distance_mm": distance,
            }
            if tuple(row) != CELL_TARGET_FIELDS:
                raise GeometryError("cell-target output schema changed")
            rows.append(row)
    expected = len(cells) * len(anchors)
    if len(rows) != expected:
        raise GeometryError(f"cell-target row count changed: {len(rows)} != {expected}")
    return rows


def build_pair_geometry(anchors: Sequence[TargetAnchor]) -> list[PairGeometry]:
    rows: list[PairGeometry] = []
    for left, right in itertools.combinations(anchors, 2):
        delta_ap = right.symmetric_ap_mm - left.symmetric_ap_mm
        delta_dv = right.symmetric_dv_mm - left.symmetric_dv_mm
        delta_lateral = (
            right.canonical_target_lateral_mm - left.canonical_target_lateral_mm
        )
        distance = math.sqrt(
            delta_ap * delta_ap
            + delta_dv * delta_dv
            + delta_lateral * delta_lateral
        )
        cross_side = int(left.laterality != right.laterality)
        homotopic = int(left.target_id == right.target_id and cross_side == 1)
        _finite(
            (delta_ap, delta_dv, delta_lateral, distance),
            f"target pair {left.coordinate_id}, {right.coordinate_id}",
        )
        rows.append(
            PairGeometry(
                coordinate_index_a=left.coordinate_index,
                coordinate_id_a=left.coordinate_id,
                target_id_a=left.target_id,
                laterality_a=left.laterality,
                coordinate_index_b=right.coordinate_index,
                coordinate_id_b=right.coordinate_id,
                target_id_b=right.target_id,
                laterality_b=right.laterality,
                delta_ap_mm=delta_ap,
                delta_dv_mm=delta_dv,
                delta_lateral_mm=delta_lateral,
                distance_mm=distance,
                cross_side=cross_side,
                homotopic=homotopic,
            )
        )
    if len(rows) != 120:
        raise GeometryError(f"target-pair row count changed: {len(rows)}")
    return rows


def pair_rows(pairs: Sequence[PairGeometry]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair in pairs:
        row: dict[str, object] = {
            "coordinate_index_a": pair.coordinate_index_a,
            "coordinate_id_a": pair.coordinate_id_a,
            "target_id_a": pair.target_id_a,
            "laterality_a": pair.laterality_a,
            "coordinate_index_b": pair.coordinate_index_b,
            "coordinate_id_b": pair.coordinate_id_b,
            "target_id_b": pair.target_id_b,
            "laterality_b": pair.laterality_b,
            "delta_ap_mm": pair.delta_ap_mm,
            "delta_dv_mm": pair.delta_dv_mm,
            "delta_lateral_mm": pair.delta_lateral_mm,
            "target_pair_distance_mm": pair.distance_mm,
            "cross_side": pair.cross_side,
            "homotopic": pair.homotopic,
        }
        if tuple(row) != PAIR_FIELDS:
            raise GeometryError("target-pair output schema changed")
        rows.append(row)
    return rows


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def pair_lookup(pairs: Sequence[PairGeometry]) -> dict[tuple[str, str], PairGeometry]:
    result = {
        _pair_key(pair.coordinate_id_a, pair.coordinate_id_b): pair for pair in pairs
    }
    if len(result) != len(pairs):
        raise GeometryError("duplicate target-pair geometry")
    return result


def model_set_features(
    omega: Sequence[str],
    soma_target_distance_mm: Mapping[str, float],
    pairs: Mapping[tuple[str, str], PairGeometry],
) -> dict[str, float]:
    """Return the frozen six M1 geometry features for any candidate subset."""
    selected = tuple(omega)
    if len(selected) != len(set(selected)):
        raise GeometryError("Omega contains a duplicate coordinate")
    if not selected:
        return {feature: 0.0 for feature in MODEL_FEATURES}

    distances = _finite(
        (soma_target_distance_mm[target] for target in selected),
        "Omega soma-target distance",
    )
    mean_soma = sum(distances) / len(distances)
    if len(distances) == 1:
        sd_soma = 0.0
    else:
        sd_soma = math.sqrt(
            sum((distance - mean_soma) ** 2 for distance in distances) / len(distances)
        )

    if len(selected) == 1:
        return {
            "mean_soma_target_distance_mm": mean_soma,
            "sd_soma_target_distance_mm": 0.0,
            "mean_target_pair_distance_mm": 0.0,
            "cross_side_pair_fraction": 0.0,
            "homotopic_pair_fraction": 0.0,
            "mst_mean_edge_length_mm": 0.0,
        }

    selected_pairs: list[PairGeometry] = []
    for left, right in itertools.combinations(selected, 2):
        try:
            selected_pairs.append(pairs[_pair_key(left, right)])
        except KeyError as error:
            raise GeometryError(f"missing pair geometry for {left}, {right}") from error
    pair_count = len(selected_pairs)
    mean_pair = sum(pair.distance_mm for pair in selected_pairs) / pair_count
    cross_fraction = sum(pair.cross_side for pair in selected_pairs) / pair_count
    homotopic_fraction = sum(pair.homotopic for pair in selected_pairs) / pair_count

    parent = {target: target for target in selected}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    mst_total = 0.0
    accepted = 0
    edges = sorted(
        selected_pairs,
        key=lambda pair: (
            pair.distance_mm,
            min(pair.coordinate_index_a, pair.coordinate_index_b),
            max(pair.coordinate_index_a, pair.coordinate_index_b),
        ),
    )
    for edge in edges:
        left_root = find(edge.coordinate_id_a)
        right_root = find(edge.coordinate_id_b)
        if left_root == right_root:
            continue
        parent[right_root] = left_root
        mst_total += edge.distance_mm
        accepted += 1
        if accepted == len(selected) - 1:
            break
    if accepted != len(selected) - 1:
        raise GeometryError("target-pair graph is disconnected")
    result = {
        "mean_soma_target_distance_mm": mean_soma,
        "sd_soma_target_distance_mm": sd_soma,
        "mean_target_pair_distance_mm": mean_pair,
        "cross_side_pair_fraction": cross_fraction,
        "homotopic_pair_fraction": homotopic_fraction,
        "mst_mean_edge_length_mm": mst_total / (len(selected) - 1),
    }
    if tuple(result) != MODEL_FEATURES:
        raise GeometryError("model set-feature order changed")
    _finite(result.values(), "Omega set features")
    return result


def _format_output_value(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GeometryError("refusing to write a nonfinite geometry value")
        return format(value, ".12g")
    return value


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise GeometryError(f"refusing symlink output below {path.parent}")
    if temporary.exists():
        raise GeometryError(f"stale temporary output exists: {temporary}")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if tuple(row) != tuple(fields):
                raise GeometryError(f"row schema changed while writing {path.name}")
            writer.writerow({key: _format_output_value(row[key]) for key in fields})
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise GeometryError(f"refusing symlink output below {path.parent}")
    if temporary.exists():
        raise GeometryError(f"stale temporary output exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _synthetic_anchors() -> list[TargetAnchor]:
    anchors: list[TargetAnchor] = []
    for index, coordinate_id in enumerate(PANEL):
        target_number = index // 2
        laterality = "ipsilateral" if index % 2 == 0 else "contralateral"
        symmetric_ap = 2.0 + 0.2 * target_number
        symmetric_dv = 1.0 + 0.1 * target_number
        symmetric_depth = 2.4 + 0.05 * target_number
        anchors.append(
            TargetAnchor(
                coordinate_index=index,
                coordinate_id=coordinate_id,
                target_index=target_number,
                target_id=f"synthetic_target_{target_number}",
                anchor_structure_id=1000 + target_number,
                laterality=laterality,
                left_voxels=10,
                right_voxels=10,
                left_centroid_mm=(symmetric_ap, symmetric_dv, MIDLINE_MM - symmetric_depth),
                right_centroid_mm=(symmetric_ap, symmetric_dv, MIDLINE_MM + symmetric_depth),
                symmetric_ap_mm=symmetric_ap,
                symmetric_dv_mm=symmetric_dv,
                symmetric_lateral_depth_mm=symmetric_depth,
            )
        )
    return anchors


def run_self_test() -> Mapping[str, Any]:
    anchors = _synthetic_anchors()
    reflected_cells = (
        Cell("right", "right.swc", "g", "s", "MOp5", "right", (4.0, 1.5, 6.9)),
        Cell("left", "left.swc", "g", "s", "MOp5", "left", (4.0, 1.5, 4.5)),
    )
    reflection_rows = build_cell_target_geometry(reflected_cells, anchors, MIDLINE_MM)
    right_rows = reflection_rows[:16]
    left_rows = reflection_rows[16:]
    for right, left in zip(right_rows, left_rows):
        for field in (
            "delta_ap_mm",
            "delta_dv_mm",
            "delta_lateral_mm",
            "soma_target_distance_mm",
        ):
            if not math.isclose(float(right[field]), float(left[field]), abs_tol=1.0e-12):
                raise GeometryError(f"reflection invariance failed for {field}")

    ipsilateral = right_rows[0]
    contralateral = right_rows[1]
    if not (
        float(ipsilateral["delta_lateral_mm"]) > 0.0
        and float(contralateral["delta_lateral_mm"]) < 0.0
        and float(contralateral["soma_target_distance_mm"])
        > float(ipsilateral["soma_target_distance_mm"])
    ):
        raise GeometryError("relative laterality sign/distance test failed")

    synthetic_cells: list[Cell] = []
    for group, count in GROUP_SIZES.items():
        for ordinal in range(count):
            synthetic_cells.append(
                Cell(
                    provider_neuron_id=f"{group}_{ordinal}",
                    filename=f"{group}_{ordinal}.swc",
                    group=group,
                    sample_id="synthetic",
                    source_acronym="MOp5",
                    hemisphere="right",
                    soma_mm=(4.0, 1.5, 6.9),
                )
            )
    if len(build_cell_target_geometry(synthetic_cells, anchors, MIDLINE_MM)) != 3104:
        raise GeometryError("194 by 16 row-count test failed")
    pairs = build_pair_geometry(anchors)
    if len(pairs) != 120:
        raise GeometryError("choose(16,2) row-count test failed")
    lookup = pair_lookup(pairs)
    distances = {
        str(row["coordinate_id"]): float(row["soma_target_distance_mm"])
        for row in right_rows
    }
    k0 = model_set_features((), distances, lookup)
    if any(value != 0.0 for value in k0.values()):
        raise GeometryError("K=0 features are not exact zeros")
    k1 = model_set_features((PANEL[0],), distances, lookup)
    if k1["mean_soma_target_distance_mm"] != distances[PANEL[0]]:
        raise GeometryError("K=1 mean soma-target distance changed")
    if any(k1[feature] != 0.0 for feature in MODEL_FEATURES[1:]):
        raise GeometryError("K=1 undefined features are not exact zeros")

    triangle = {
        _pair_key("a", "b"): PairGeometry(0, "a", "A", "ipsilateral", 1, "b", "B", "ipsilateral", 0.0, 0.0, 0.0, 3.0, 0, 0),
        _pair_key("a", "c"): PairGeometry(0, "a", "A", "ipsilateral", 2, "c", "C", "contralateral", 0.0, 0.0, 0.0, 4.0, 1, 0),
        _pair_key("b", "c"): PairGeometry(1, "b", "B", "ipsilateral", 2, "c", "C", "contralateral", 0.0, 0.0, 0.0, 5.0, 1, 0),
    }
    triangle_features = model_set_features(
        ("a", "b", "c"), {"a": 1.0, "b": 2.0, "c": 3.0}, triangle
    )
    if not math.isclose(
        triangle_features["mst_mean_edge_length_mm"], 3.5, abs_tol=1.0e-12
    ):
        raise GeometryError("deterministic MST mean-edge test failed")

    return {
        "status": "pass",
        "mode": "deterministic_synthetic_only",
        "checks": [
            "left_right_soma_reflection_invariance",
            "relative_ipsilateral_contralateral_sign_and_distance",
            "K0_exact_zero_features",
            "K1_undefined_features_exact_zero",
            "deterministic_MST_mean_edge_length",
            "exact_194_by_16_and_choose_16_2_counts",
        ],
        "target_calls_or_swcs_opened": False,
        "final_or_unassigned_opened": False,
    }


def _validate_output_root(path: Path, forbidden_roots: Sequence[Path]) -> Path:
    scratch_root = _ep10_scratch_root()
    output_root = _require_below(
        path, scratch_root, "output root", must_exist=False
    )
    if output_root.parent != scratch_root or output_root.name != OUTPUT_DIRECTORY:
        raise GeometryError(
            f"output root must be {scratch_root / OUTPUT_DIRECTORY}"
        )
    for forbidden in forbidden_roots:
        if (
            output_root == forbidden
            or output_root in forbidden.parents
            or forbidden in output_root.parents
        ):
            raise GeometryError("output root overlaps an input root")
    if output_root.exists():
        if not output_root.is_dir():
            raise GeometryError("output root exists and is not a directory")
        if any(output_root.iterdir()):
            raise GeometryError(f"refusing nonempty output directory: {output_root}")
    return output_root


def run_execute(args: argparse.Namespace) -> Mapping[str, Any]:
    contract = _load_contract(args.contract)
    stage0_files = contract["path_and_output_seal"]["exact_stage0_files"]
    observation_contract = _require_stage0_file(
        args.observation_contract,
        str(stage0_files["observation_base_contract"]),
        "observation base contract",
    )
    vocabulary = _require_stage0_file(
        args.vocabulary, str(stage0_files["vocabulary"]), "target vocabulary"
    )
    annotation, ontology_path = _assert_ep10_local_assets(
        args.annotation, args.ontology, contract
    )
    model_interface = _require_below(
        args.model_interface, _ep10_scratch_root(), "model interface"
    )
    cells = _load_interface(model_interface, contract, MIDLINE_MM)
    output_root = _validate_output_root(
        args.output_root, (annotation.parent, model_interface.parent)
    )

    import construct_mop_terminal_targets as core

    _, atlas, config, ontology, _, coordinates = core._load_real_assets(
        observation_contract,
        annotation,
        ontology_path,
        vocabulary,
        load_target_vocabulary=True,
    )
    if tuple(float(value) for value in config.spacing_um) != (25.0, 25.0, 25.0):
        raise GeometryError("M1 geometry requires the EP10 25-um atlas")
    if not math.isclose(config.ml_midline_um / 1000.0, MIDLINE_MM, abs_tol=1.0e-12):
        raise GeometryError("atlas midline changed")
    annotation = getattr(atlas, "_data", None)
    if annotation is None:
        raise GeometryError("loaded atlas does not expose its annotation array")

    anchors = derive_atlas_anchors(
        annotation,
        config.spacing_um,
        config.ml_midline_um,
        ontology,
        coordinates,
    )
    anchors_csv = anchor_rows(anchors)
    cell_target_csv = build_cell_target_geometry(cells, anchors, MIDLINE_MM)
    pairs = build_pair_geometry(anchors)
    pairs_csv = pair_rows(pairs)

    output_root.mkdir(exist_ok=True)
    if (
        _require_unsymlinked_path(output_root, "output root", must_exist=True)
        != output_root
        or not output_root.is_dir()
    ):
        raise GeometryError("output root changed while preparing outputs")
    _write_csv(output_root / "atlas_target_anchors.csv", ANCHOR_FIELDS, anchors_csv)
    _write_csv(
        output_root / "cell_target_geometry.csv", CELL_TARGET_FIELDS, cell_target_csv
    )
    _write_csv(output_root / "target_pair_geometry.csv", PAIR_FIELDS, pairs_csv)

    mask_qc = {
        anchor.coordinate_id: {
            "physical_left_voxels": anchor.left_voxels,
            "physical_right_voxels": anchor.right_voxels,
            "symmetric_lateral_depth_mm": anchor.symmetric_lateral_depth_mm,
        }
        for anchor in anchors
    }
    summary: dict[str, Any] = {
        "status": "pass",
        "episode_id": "ep10",
        "stage": "v3_M1_target_blind_atlas_geometry",
        "analysis_label": "exploratory",
        "contract_status": contract["status"],
        "interface_rows": len(cells),
        "eligible_cells_by_group": dict(sorted(Counter(cell.group for cell in cells).items())),
        "physical_soma_hemispheres": dict(Counter(cell.hemisphere for cell in cells)),
        "coordinate_count": len(anchors),
        "cell_target_rows": len(cell_target_csv),
        "target_pair_rows": len(pairs_csv),
        "distance_units": "millimeters",
        "canonical_laterality": {
            "soma": "physical soma mirrored to positive lateral depth",
            "ipsilateral_target": "positive symmetric lateral depth",
            "contralateral_target": "negative symmetric lateral depth",
        },
        "atlas_mask_qc": mask_qc,
        "model_set_features": list(MODEL_FEATURES),
        "degenerate_set_features": {
            "K0": "all exact zero",
            "K1": "mean soma-target distance defined; other five exact zero",
        },
        "outputs": {
            "atlas_target_anchors.csv": len(anchors_csv),
            "cell_target_geometry.csv": len(cell_target_csv),
            "target_pair_geometry.csv": len(pairs_csv),
            "geometry_summary.json": 1,
        },
        "target_calls_or_swcs_opened": False,
        "realized_axon_geometry_opened": False,
        "final_or_unassigned_opened": False,
        "imputed_or_dropped_geometry_rows": 0,
    }
    _write_json(output_root / "geometry_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--observation-contract", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--ontology", type=Path)
    parser.add_argument("--vocabulary", type=Path)
    parser.add_argument("--model-interface", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.self_test:
            result = run_self_test()
        else:
            required = (
                "contract",
                "observation_contract",
                "annotation",
                "ontology",
                "vocabulary",
                "model_interface",
                "output_root",
            )
            missing = [name for name in required if getattr(args, name) is None]
            if missing:
                raise GeometryError(
                    "--execute requires: " + ", ".join(name.replace("_", "-") for name in missing)
                )
            result = run_execute(args)
    except (GeometryError, OSError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
