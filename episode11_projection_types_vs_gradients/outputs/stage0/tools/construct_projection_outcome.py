#!/usr/bin/env python3
"""Construct EP11's fixed regional projection outcome from registered SWCs.

The primary outcome is a 56-coordinate composition: 28 fixed Allen grey-
matter targets crossed with soma-relative ipsilateral/contralateral side.  A
segment is split at every crossed 25-um voxel face and at the ML midline, so
no whole segment is assigned by an endpoint or midpoint.  Grey-matter cable
not claimed by one of the 28 targets is QC-only and is not an outcome
coordinate or part of its normalization denominator.

Root and marker edges are excluded; dendrite edges are separately counted as
excluded QC.  Every input cell must also carry its provider Left/Right label;
construction stops before segment allocation if that label disagrees with the
registered soma's ML side.

This module deliberately has a dependency-free ``--self-test`` path.  Real
construction imports PyYAML, NumPy, and pynrrd only after the explicit
``--execute`` switch.  The self-test uses a tiny in-memory atlas and no EP11
SWC or real projection outcome.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence, TextIO


AXON_TYPES = frozenset((0, 2))
AXON_PARENT_TYPES = frozenset((0, 1, 2))
EXCLUDED_DENDRITE_TYPES = frozenset((3,))
IGNORED_MARKER_TYPES = frozenset((10, 11, 12))
ALLOWED_NODE_TYPES = AXON_TYPES | AXON_PARENT_TYPES | EXCLUDED_DENDRITE_TYPES | IGNORED_MARKER_TYPES
EXCLUDED_STRUCTURE_ROOTS = {
    1009: "fiber_tract_um",
    73: "ventricular_system_um",
    1024: "groove_um",
    304325711: "retina_um",
}


class ConstructionError(RuntimeError):
    """Raised when a frozen construction invariant is violated."""


@dataclass(frozen=True)
class GeometryConfig:
    spacing_um: tuple[float, float, float]
    shape: tuple[int, int, int]
    ml_midline_um: float
    grey_structure_id: int = 8
    zero_tolerance_um: float = 1.0e-9


@dataclass(frozen=True)
class SwcNode:
    node_id: int
    node_type: int
    point_um: tuple[float, float, float]
    radius_um: float
    parent_id: int


@dataclass(frozen=True)
class Coordinate:
    index: int
    coordinate_id: str
    target_index: int
    target_id: str
    anchor_structure_id: int
    laterality: str


@dataclass
class CellQC:
    cell_id: str
    provider_hemisphere: str = ""
    soma_hemisphere: str = ""
    provider_hemisphere_agreement: bool = False
    swc_node_count: int = 0
    axon_edge_count: int = 0
    dendrite_edge_count: int = 0
    ignored_marker_node_count: int = 0
    extra_column_line_count: int = 0
    total_axon_centerline_um: float = 0.0
    named_target_um: float = 0.0
    unclaimed_grey_um: float = 0.0
    fiber_tract_um: float = 0.0
    ventricular_system_um: float = 0.0
    groove_um: float = 0.0
    retina_um: float = 0.0
    other_nongrey_um: float = 0.0
    annotation_zero_um: float = 0.0
    outside_volume_um: float = 0.0
    dendrite_centerline_excluded_um: float = 0.0
    allocation_error_um: float = 0.0
    named_coordinate_sum: float = 0.0


@dataclass(frozen=True)
class CellOutcome:
    cell_id: str
    coordinate_ids: tuple[str, ...]
    lengths_um: tuple[float, ...]
    proportions: tuple[float, ...]
    qc: CellQC


@dataclass(frozen=True)
class ManifestEntry:
    cell_id: str
    swc_path: Path
    provider_hemisphere: str


class Atlas(Protocol):
    shape: tuple[int, int, int]

    def structure_id(self, index: tuple[int, int, int]) -> int:
        """Return the annotation structure ID at an in-bounds index."""


class ArrayAtlas:
    """Adapter for a 3-D NumPy-like annotation array."""

    def __init__(self, data: Any):
        if getattr(data, "ndim", None) != 3:
            raise ConstructionError("annotation must be three-dimensional")
        self._data = data
        self.shape = tuple(int(value) for value in data.shape)

    def structure_id(self, index: tuple[int, int, int]) -> int:
        return int(self._data[index])


class SyntheticAtlas:
    """Small dictionary-backed atlas used only by the deterministic self-test."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        values: Mapping[tuple[int, int, int], int],
    ):
        self.shape = shape
        self._values = dict(values)

    def structure_id(self, index: tuple[int, int, int]) -> int:
        return int(self._values.get(index, 0))


class Ontology:
    """Allen structure ancestry needed for target and exclusion mapping."""

    def __init__(self, parent_by_id: Mapping[int, int | None]):
        self.parent_by_id = dict(parent_by_id)
        self._lineage_cache: dict[int, tuple[int, ...]] = {}
        for structure_id in self.parent_by_id:
            self.lineage(structure_id)

    def lineage(self, structure_id: int) -> tuple[int, ...]:
        if structure_id not in self.parent_by_id:
            raise ConstructionError(
                f"annotation structure ID {structure_id} is absent from the pinned ontology"
            )
        cached = self._lineage_cache.get(structure_id)
        if cached is not None:
            return cached
        result: list[int] = []
        seen: set[int] = set()
        current: int | None = structure_id
        while current is not None:
            if current in seen:
                raise ConstructionError(f"ontology cycle at structure {current}")
            if current not in self.parent_by_id:
                raise ConstructionError(
                    f"ontology parent {current} is missing while tracing {structure_id}"
                )
            seen.add(current)
            result.append(current)
            current = self.parent_by_id[current]
        lineage = tuple(result)
        self._lineage_cache[structure_id] = lineage
        return lineage

    def is_descendant_or_self(self, structure_id: int, ancestor_id: int) -> bool:
        return ancestor_id in self.lineage(structure_id)


class StructureMapper:
    """Map annotation IDs to a unique named target or a QC-only category."""

    def __init__(
        self,
        ontology: Ontology,
        anchor_to_target: Mapping[int, int],
        grey_structure_id: int,
    ):
        if len(anchor_to_target) != len(set(anchor_to_target.values())):
            raise ConstructionError("target anchors do not map one-to-one to targets")
        self.ontology = ontology
        self.anchor_to_target = dict(anchor_to_target)
        self.grey_structure_id = grey_structure_id
        for anchor in self.anchor_to_target:
            if not ontology.is_descendant_or_self(anchor, grey_structure_id):
                raise ConstructionError(f"target anchor {anchor} is not below Allen grey")
        anchors = tuple(self.anchor_to_target)
        for left in anchors:
            for right in anchors:
                if left != right and ontology.is_descendant_or_self(right, left):
                    raise ConstructionError(
                        f"overlapping target anchors: {left} contains {right}"
                    )
        self._cache: dict[int, tuple[str, int | None]] = {}

    def classify(self, structure_id: int) -> tuple[str, int | None]:
        cached = self._cache.get(structure_id)
        if cached is not None:
            return cached
        lineage = set(self.ontology.lineage(structure_id))
        matches = [
            target_index
            for anchor, target_index in self.anchor_to_target.items()
            if anchor in lineage
        ]
        if len(matches) > 1:
            raise ConstructionError(
                f"structure {structure_id} maps to multiple named targets"
            )
        if matches:
            result = ("named", matches[0])
        else:
            exclusion_matches = [
                label for root, label in EXCLUDED_STRUCTURE_ROOTS.items() if root in lineage
            ]
            if len(exclusion_matches) > 1:
                raise ConstructionError(
                    f"structure {structure_id} maps to multiple exclusion roots"
                )
            if exclusion_matches:
                result = (exclusion_matches[0], None)
            elif self.grey_structure_id in lineage:
                result = ("unclaimed_grey_um", None)
            else:
                result = ("other_nongrey_um", None)
        self._cache[structure_id] = result
        return result


def _strict_int(token: str, field: str, line_number: int) -> int:
    try:
        value = int(token)
    except ValueError as error:
        raise ConstructionError(
            f"SWC line {line_number}: {field} must be an integer"
        ) from error
    return value


def _validate_parent_paths(
    nodes: Mapping[int, SwcNode], root_id: int
) -> int:
    """Validate all parent paths in linear time and return parent lookups.

    Once a path reaches the canonical root (or any previously validated node),
    every node on that path is memoized as root-connected.  Thus a valid node
    is dereferenced at most once across the whole tree rather than once for
    every descendant.
    """
    if root_id not in nodes or nodes[root_id].parent_id != -1:
        raise ConstructionError("canonical SWC root is absent or is not a root")
    validated_to_root = {root_id}
    parent_lookups = 0
    for start in nodes:
        if start in validated_to_root:
            continue
        current = start
        path: list[int] = []
        path_nodes: set[int] = set()
        while current not in validated_to_root:
            if current in path_nodes:
                raise ConstructionError(f"SWC contains a cycle through node {current}")
            if current not in nodes:
                raise ConstructionError(
                    f"SWC node {path[-1]} refers to missing parent {current}"
                )
            path.append(current)
            path_nodes.add(current)
            parent = nodes[current].parent_id
            parent_lookups += 1
            if parent == -1:
                if current != root_id:
                    raise ConstructionError(
                        f"node {start} reaches a noncanonical root {current}"
                    )
                break
            current = parent
        validated_to_root.update(path)
    return parent_lookups


def parse_swc_lines(lines: Iterable[str]) -> tuple[dict[int, SwcNode], int]:
    """Parse and validate an SWC tree without interpreting projection outcomes."""
    nodes: dict[int, SwcNode] = {}
    extra_column_line_count = 0
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 7:
            raise ConstructionError(f"SWC line {line_number}: fewer than seven columns")
        if len(fields) > 7:
            extra_column_line_count += 1
        node_id = _strict_int(fields[0], "node_id", line_number)
        node_type = _strict_int(fields[1], "node_type", line_number)
        parent_id = _strict_int(fields[6], "parent_id", line_number)
        if node_id in nodes:
            raise ConstructionError(f"SWC line {line_number}: duplicate node ID {node_id}")
        if node_type not in ALLOWED_NODE_TYPES:
            raise ConstructionError(
                f"SWC line {line_number}: unsupported node type {node_type}"
            )
        try:
            values = tuple(float(fields[index]) for index in range(2, 6))
        except ValueError as error:
            raise ConstructionError(
                f"SWC line {line_number}: coordinates/radius must be numeric"
            ) from error
        if not all(math.isfinite(value) for value in values):
            raise ConstructionError(
                f"SWC line {line_number}: coordinates/radius must be finite"
            )
        if values[3] < 0.0:
            raise ConstructionError(
                f"SWC line {line_number}: radius must be nonnegative"
            )
        nodes[node_id] = SwcNode(
            node_id=node_id,
            node_type=node_type,
            point_um=(values[0], values[1], values[2]),
            radius_um=values[3],
            parent_id=parent_id,
        )

    if not nodes:
        raise ConstructionError("SWC contains no nodes")
    roots = [node for node in nodes.values() if node.parent_id == -1]
    if len(roots) != 1:
        raise ConstructionError(f"SWC must have exactly one root; observed {len(roots)}")
    root = roots[0]
    if root.node_type != 1:
        raise ConstructionError("the unique SWC root must be the soma (type 1)")
    for node in nodes.values():
        if node.parent_id != -1 and node.parent_id not in nodes:
            raise ConstructionError(
                f"SWC node {node.node_id} refers to missing parent {node.parent_id}"
            )

    _validate_parent_paths(nodes, root.node_id)
    return nodes, extra_column_line_count


def parse_swc(path: Path) -> tuple[dict[int, SwcNode], int]:
    with path.open("r", encoding="utf-8") as handle:
        return parse_swc_lines(handle)


def euclidean_length(
    start: tuple[float, float, float], end: tuple[float, float, float]
) -> float:
    return math.sqrt(sum((end[i] - start[i]) ** 2 for i in range(3)))


def split_parameters(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    config: GeometryConfig,
) -> tuple[float, ...]:
    """Return sorted parameters at every crossed voxel and midline plane."""
    parameters = [0.0, 1.0]
    for dimension in range(3):
        delta = end[dimension] - start[dimension]
        if delta == 0.0:
            continue
        spacing = config.spacing_um[dimension]
        extent = config.shape[dimension] * spacing
        low = min(start[dimension], end[dimension])
        high = max(start[dimension], end[dimension])
        first_k = max(0, math.ceil(low / spacing))
        last_k = min(config.shape[dimension], math.floor(high / spacing))
        for boundary_index in range(first_k, last_k + 1):
            boundary = boundary_index * spacing
            t_value = (boundary - start[dimension]) / delta
            if 0.0 < t_value < 1.0:
                parameters.append(t_value)
        # The midline need not coincide with a voxel boundary in a fixture or
        # a future pinned atlas, so include it independently.
        if dimension == 2 and 0.0 <= config.ml_midline_um <= extent:
            t_midline = (config.ml_midline_um - start[dimension]) / delta
            if 0.0 < t_midline < 1.0:
                parameters.append(t_midline)

    parameters.sort()
    unique = [parameters[0]]
    for value in parameters[1:]:
        if value - unique[-1] > 1.0e-13:
            unique.append(value)
    unique[0] = 0.0
    unique[-1] = 1.0
    return tuple(unique)


def _point_at(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    t_value: float,
) -> tuple[float, float, float]:
    return tuple(start[i] + t_value * (end[i] - start[i]) for i in range(3))  # type: ignore[return-value]


def _voxel_index(
    point: tuple[float, float, float], config: GeometryConfig
) -> tuple[int, int, int] | None:
    indexes = tuple(
        math.floor(point[dimension] / config.spacing_um[dimension])
        for dimension in range(3)
    )
    if any(index < 0 or index >= config.shape[d] for d, index in enumerate(indexes)):
        return None
    return indexes  # type: ignore[return-value]


def _relative_laterality(
    segment_ml_um: float, soma_ml_um: float, midline_um: float
) -> str:
    soma_left = soma_ml_um < midline_um
    segment_left = segment_ml_um < midline_um
    return "ipsilateral" if soma_left == segment_left else "contralateral"


def _normalize_provider_hemisphere(value: str) -> str:
    normalized = value.strip().casefold()
    aliases = {"l": "left", "left": "left", "r": "right", "right": "right"}
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ConstructionError(
            "provider hemisphere must be one of Left/Right or L/R"
        ) from error


def _soma_hemisphere(soma_ml_um: float, midline_um: float) -> str:
    return "left" if soma_ml_um < midline_um else "right"


def _assert_soma_hemisphere_agreement(
    cell_id: str,
    provider_hemisphere: str,
    soma_ml_um: float,
    midline_um: float,
) -> tuple[str, str]:
    provider_side = _normalize_provider_hemisphere(provider_hemisphere)
    soma_side = _soma_hemisphere(soma_ml_um, midline_um)
    if provider_side != soma_side:
        raise ConstructionError(
            f"cell {cell_id} provider hemisphere {provider_side} disagrees with "
            f"registered soma hemisphere {soma_side} (ML={soma_ml_um:.12g} um, "
            f"midline={midline_um:.12g} um)"
        )
    return provider_side, soma_side


def allocate_segment(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    soma_ml_um: float,
    atlas: Atlas,
    config: GeometryConfig,
    mapper: StructureMapper,
    coordinate_lookup: Mapping[tuple[int, str], int],
    coordinate_lengths_um: list[float],
    qc: CellQC,
) -> None:
    total_length = euclidean_length(start, end)
    if total_length <= config.zero_tolerance_um:
        return
    allocated = 0.0
    parameters = split_parameters(start, end, config)
    for left, right in zip(parameters, parameters[1:]):
        fraction = right - left
        if fraction <= 0.0:
            continue
        subsegment_length = total_length * fraction
        midpoint = _point_at(start, end, (left + right) / 2.0)
        index = _voxel_index(midpoint, config)
        if index is None:
            qc.outside_volume_um += subsegment_length
        else:
            structure_id = atlas.structure_id(index)
            if structure_id == 0:
                qc.annotation_zero_um += subsegment_length
            else:
                category, target_index = mapper.classify(structure_id)
                if category == "named":
                    assert target_index is not None
                    laterality = _relative_laterality(
                        midpoint[2], soma_ml_um, config.ml_midline_um
                    )
                    try:
                        coordinate_index = coordinate_lookup[(target_index, laterality)]
                    except KeyError as error:
                        raise ConstructionError(
                            f"vocabulary lacks target {target_index} {laterality}"
                        ) from error
                    coordinate_lengths_um[coordinate_index] += subsegment_length
                    qc.named_target_um += subsegment_length
                else:
                    setattr(qc, category, getattr(qc, category) + subsegment_length)
        allocated += subsegment_length

    tolerance = max(config.zero_tolerance_um, total_length * 1.0e-11)
    difference = allocated - total_length
    if abs(difference) > tolerance:
        raise ConstructionError(
            f"segment allocation failed length conservation by {difference:.12g} um"
        )
    qc.allocation_error_um += difference


def construct_cell_outcome(
    cell_id: str,
    provider_hemisphere: str,
    nodes: Mapping[int, SwcNode],
    extra_column_line_count: int,
    atlas: Atlas,
    config: GeometryConfig,
    mapper: StructureMapper,
    coordinates: Sequence[Coordinate],
) -> CellOutcome:
    if tuple(coordinate.index for coordinate in coordinates) != tuple(range(len(coordinates))):
        raise ConstructionError("coordinate indexes must be contiguous and ordered")
    coordinate_lookup = {
        (coordinate.target_index, coordinate.laterality): coordinate.index
        for coordinate in coordinates
    }
    if len(coordinate_lookup) != len(coordinates):
        raise ConstructionError("duplicate target/laterality coordinate")
    roots = [node for node in nodes.values() if node.parent_id == -1]
    if len(roots) != 1 or roots[0].node_type != 1:
        raise ConstructionError("validated SWC root/soma invariant was lost")
    soma_ml_um = roots[0].point_um[2]
    provider_side, soma_side = _assert_soma_hemisphere_agreement(
        cell_id,
        provider_hemisphere,
        soma_ml_um,
        config.ml_midline_um,
    )
    lengths = [0.0] * len(coordinates)
    qc = CellQC(
        cell_id=cell_id,
        provider_hemisphere=provider_side,
        soma_hemisphere=soma_side,
        provider_hemisphere_agreement=True,
        swc_node_count=len(nodes),
        ignored_marker_node_count=sum(
            node.node_type in IGNORED_MARKER_TYPES for node in nodes.values()
        ),
        extra_column_line_count=extra_column_line_count,
    )

    for child in nodes.values():
        if child.parent_id == -1:
            continue
        parent = nodes[child.parent_id]
        edge_length = euclidean_length(parent.point_um, child.point_um)
        if child.node_type in AXON_TYPES:
            if parent.node_type not in AXON_PARENT_TYPES:
                raise ConstructionError(
                    f"axon node {child.node_id} has disallowed parent type {parent.node_type}"
                )
            qc.axon_edge_count += 1
            qc.total_axon_centerline_um += edge_length
            allocate_segment(
                parent.point_um,
                child.point_um,
                soma_ml_um,
                atlas,
                config,
                mapper,
                coordinate_lookup,
                lengths,
                qc,
            )
        elif child.node_type in EXCLUDED_DENDRITE_TYPES:
            qc.dendrite_edge_count += 1
            qc.dendrite_centerline_excluded_um += edge_length

    allocated_total = (
        qc.named_target_um
        + qc.unclaimed_grey_um
        + qc.fiber_tract_um
        + qc.ventricular_system_um
        + qc.groove_um
        + qc.retina_um
        + qc.other_nongrey_um
        + qc.annotation_zero_um
        + qc.outside_volume_um
    )
    conservation_error = allocated_total - qc.total_axon_centerline_um
    tolerance = max(config.zero_tolerance_um, qc.total_axon_centerline_um * 1.0e-11)
    if abs(conservation_error) > tolerance:
        raise ConstructionError(
            f"cell {cell_id} failed total axon length conservation by "
            f"{conservation_error:.12g} um"
        )
    qc.allocation_error_um = conservation_error
    qc.named_coordinate_sum = sum(lengths)
    if abs(qc.named_coordinate_sum - qc.named_target_um) > tolerance:
        raise ConstructionError(f"cell {cell_id} named-coordinate sum mismatch")
    if qc.named_target_um <= config.zero_tolerance_um:
        raise ConstructionError(
            f"cell {cell_id} has zero named-target length; primary composition undefined"
        )
    proportions = tuple(value / qc.named_target_um for value in lengths)
    if abs(sum(proportions) - 1.0) > 1.0e-12:
        raise ConstructionError(f"cell {cell_id} composition does not sum to one")
    return CellOutcome(
        cell_id=cell_id,
        coordinate_ids=tuple(coordinate.coordinate_id for coordinate in coordinates),
        lengths_um=tuple(lengths),
        proportions=proportions,
        qc=qc,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flatten_ontology_tree(root: Mapping[str, Any]) -> dict[int, int | None]:
    parents: dict[int, int | None] = {}
    stack: list[tuple[Mapping[str, Any], int | None]] = [(root, None)]
    while stack:
        node, expected_parent = stack.pop()
        try:
            structure_id = int(node["id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ConstructionError("ontology node lacks integer id") from error
        if structure_id in parents:
            raise ConstructionError(f"duplicate ontology structure ID {structure_id}")
        observed_parent_raw = node.get("parent_structure_id")
        observed_parent = (
            None if observed_parent_raw is None else int(observed_parent_raw)
        )
        if observed_parent != expected_parent:
            raise ConstructionError(
                f"ontology parent mismatch for {structure_id}: "
                f"{observed_parent} != {expected_parent}"
            )
        parents[structure_id] = observed_parent
        children = node.get("children", [])
        if not isinstance(children, list):
            raise ConstructionError(f"ontology children is not a list at {structure_id}")
        for child in reversed(children):
            stack.append((child, structure_id))
    return parents


def load_ontology(path: Path) -> Ontology:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    roots = payload.get("msg") if isinstance(payload, dict) else None
    if not isinstance(roots, list) or len(roots) != 1:
        raise ConstructionError("ontology JSON must contain exactly one msg root")
    return Ontology(_flatten_ontology_tree(roots[0]))


def load_vocabulary(path: Path, expected_count: int = 56) -> tuple[Coordinate, ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_count:
        raise ConstructionError(
            f"vocabulary must contain {expected_count} coordinates; observed {len(rows)}"
        )
    coordinates: list[Coordinate] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            structure_id_text = row["allen_structure_id"].strip()
            if not structure_id_text:
                raise ConstructionError(
                    f"vocabulary row {row_number} is residual/unanchored; "
                    "QC-only grey may not be an outcome coordinate"
                )
            coordinate = Coordinate(
                index=int(row["coordinate_index"]),
                coordinate_id=row["coordinate_id"],
                target_index=int(row["target_index"]),
                target_id=row["target_id"],
                anchor_structure_id=int(structure_id_text),
                laterality=row["laterality"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ConstructionError(f"malformed vocabulary row {row_number}") from error
        if coordinate.laterality not in {"ipsilateral", "contralateral"}:
            raise ConstructionError(
                f"invalid laterality at vocabulary row {row_number}"
            )
        coordinates.append(coordinate)
    coordinates.sort(key=lambda value: value.index)
    if tuple(item.index for item in coordinates) != tuple(range(expected_count)):
        raise ConstructionError("vocabulary coordinate indexes are not 0..55")
    targets: dict[int, list[Coordinate]] = {}
    for coordinate in coordinates:
        targets.setdefault(coordinate.target_index, []).append(coordinate)
    if tuple(sorted(targets)) != tuple(range(28)):
        raise ConstructionError("vocabulary target indexes are not 0..27")
    for target_index, pair in targets.items():
        if len(pair) != 2 or {item.laterality for item in pair} != {
            "ipsilateral",
            "contralateral",
        }:
            raise ConstructionError(
                f"target {target_index} does not have exactly two laterality coordinates"
            )
        if len({item.anchor_structure_id for item in pair}) != 1:
            raise ConstructionError(f"target {target_index} has inconsistent anchors")
    return tuple(coordinates)


def _load_manifest(
    manifest_path: Path | None,
    swc_path: Path | None,
    cell_id: str | None,
    provider_hemisphere: str | None,
) -> list[ManifestEntry]:
    if manifest_path is not None:
        if swc_path is not None or cell_id is not None or provider_hemisphere is not None:
            raise ConstructionError(
                "use either --swc-manifest or --swc with --cell-id and "
                "--provider-hemisphere, not both"
            )
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            required = {"cell_id", "swc_path"}
            missing = sorted(required - fields)
            if missing:
                raise ConstructionError(
                    "manifest lacks required columns: " + ", ".join(missing)
                )
            hemisphere_columns = [
                field
                for field in ("provider_hemisphere", "hemisphere")
                if field in fields
            ]
            if not hemisphere_columns:
                raise ConstructionError(
                    "manifest requires provider_hemisphere (or provider metadata "
                    "column hemisphere)"
                )
            rows = list(reader)
        result: list[ManifestEntry] = []
        seen: set[str] = set()
        for row_number, row in enumerate(rows, start=2):
            current_id = (row.get("cell_id") or "").strip()
            current_path_text = (row.get("swc_path") or "").strip()
            if not current_id or current_id in seen:
                raise ConstructionError(
                    f"manifest row {row_number} has blank or duplicate cell_id"
                )
            if not current_path_text:
                raise ConstructionError(
                    f"manifest row {row_number} has blank swc_path"
                )
            normalized_sides = {
                _normalize_provider_hemisphere(row[column])
                for column in hemisphere_columns
                if (row.get(column) or "").strip()
            }
            if len(normalized_sides) != 1:
                raise ConstructionError(
                    f"manifest row {row_number} has blank or conflicting hemisphere values"
                )
            current_path = Path(current_path_text)
            if not current_path.is_absolute():
                current_path = manifest_path.parent / current_path
            seen.add(current_id)
            result.append(
                ManifestEntry(current_id, current_path, normalized_sides.pop())
            )
        if not result:
            raise ConstructionError("SWC manifest is empty")
        return result
    if swc_path is None or not cell_id or provider_hemisphere is None:
        raise ConstructionError(
            "provide --swc-manifest or --swc, --cell-id, and --provider-hemisphere"
        )
    return [
        ManifestEntry(
            cell_id,
            swc_path,
            _normalize_provider_hemisphere(provider_hemisphere),
        )
    ]


def write_vectors(path: Path, outcomes: Sequence[CellOutcome]) -> None:
    if not outcomes:
        raise ConstructionError("no outcomes to write")
    coordinate_ids = outcomes[0].coordinate_ids
    fields = ["cell_id"] + [f"p__{value}" for value in coordinate_ids]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for outcome in outcomes:
            if outcome.coordinate_ids != coordinate_ids:
                raise ConstructionError("cell outcomes use different coordinate orders")
            row: dict[str, object] = {"cell_id": outcome.cell_id}
            row.update(
                {
                    f"p__{coordinate_id}": format(value, ".17g")
                    for coordinate_id, value in zip(
                        outcome.coordinate_ids, outcome.proportions
                    )
                }
            )
            writer.writerow(row)


def write_qc(path: Path, outcomes: Sequence[CellOutcome]) -> None:
    payload = {
        "schema_version": "ep11_projection_outcome_qc_v1",
        "cell_count": len(outcomes),
        "cells": [
            {
                **asdict(outcome.qc),
                "coordinate_ids": list(outcome.coordinate_ids),
                "coordinate_lengths_um": list(outcome.lengths_um),
            }
            for outcome in outcomes
        ],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _fixture_ontology() -> Ontology:
    return Ontology(
        {
            997: None,
            8: 997,
            10: 8,
            11: 10,
            12: 8,
            20: 8,
            1009: 997,
            73: 997,
            1024: 997,
            304325711: 997,
        }
    )


def run_self_test() -> dict[str, object]:
    """Exercise exact geometry and mapping on a deterministic synthetic cell."""
    config = GeometryConfig(
        spacing_um=(25.0, 25.0, 25.0),
        shape=(4, 1, 4),
        ml_midline_um=50.0,
        grey_structure_id=8,
        zero_tolerance_um=1.0e-9,
    )
    values: dict[tuple[int, int, int], int] = {}
    for ml_index in range(4):
        values[(0, 0, ml_index)] = 11  # child of target A
        values[(1, 0, ml_index)] = 12  # unclaimed grey, QC-only
        values[(2, 0, ml_index)] = 1009  # fiber tract, QC-only
        values[(3, 0, ml_index)] = 20  # target B
    atlas = SyntheticAtlas(config.shape, values)
    coordinates = (
        Coordinate(0, "A__ipsilateral", 0, "A", 10, "ipsilateral"),
        Coordinate(1, "A__contralateral", 0, "A", 10, "contralateral"),
        Coordinate(2, "B__ipsilateral", 1, "B", 20, "ipsilateral"),
        Coordinate(3, "B__contralateral", 1, "B", 20, "contralateral"),
    )
    mapper = StructureMapper(_fixture_ontology(), {10: 0, 20: 1}, 8)
    swc_fixture = io.StringIO(
        """# deterministic EP11 synthetic projection fixture
1 1 5 12.5 25 2 -1
2 2 95 12.5 25 1 1
3 2 5 12.5 75 1 1
4 3 15 12.5 25 1 1
5 2 -5 12.5 25 1 1
"""
    )
    nodes, extra_columns = parse_swc_lines(swc_fixture)
    outcome = construct_cell_outcome(
        "synthetic_cell",
        "Left",
        nodes,
        extra_columns,
        atlas,
        config,
        mapper,
        coordinates,
    )

    expected_lengths = (50.0, 25.0, 20.0, 0.0)
    expected_qc = {
        "total_axon_centerline_um": 150.0,
        "named_target_um": 95.0,
        "unclaimed_grey_um": 25.0,
        "fiber_tract_um": 25.0,
        "outside_volume_um": 5.0,
        "dendrite_centerline_excluded_um": 10.0,
    }
    for observed, expected in zip(outcome.lengths_um, expected_lengths):
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-10):
            raise AssertionError(
                f"synthetic coordinate length mismatch: {observed} != {expected}"
            )
    for field, expected in expected_qc.items():
        observed = float(getattr(outcome.qc, field))
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-10):
            raise AssertionError(f"synthetic QC mismatch {field}: {observed} != {expected}")
    if outcome.proportions[3] != 0.0:
        raise AssertionError("zero named coordinate did not remain exact zero")
    if not math.isclose(sum(outcome.proportions), 1.0, abs_tol=1.0e-12):
        raise AssertionError("synthetic composition does not sum to one")
    if abs(outcome.qc.allocation_error_um) > 1.0e-10:
        raise AssertionError("synthetic length conservation failed")
    if not outcome.qc.provider_hemisphere_agreement:
        raise AssertionError("synthetic soma/provider hemisphere check did not pass")

    # This count is an implementation-independent regression guard against
    # restoring the former per-node full ancestry walk (quadratic on a chain).
    chain_size = 4096
    chain = {
        node_id: SwcNode(
            node_id=node_id,
            node_type=1 if node_id == 1 else 2,
            point_um=(float(node_id), 0.0, 0.0),
            radius_um=1.0,
            parent_id=-1 if node_id == 1 else node_id - 1,
        )
        for node_id in range(1, chain_size + 1)
    }
    parent_lookups = _validate_parent_paths(chain, 1)
    if parent_lookups != chain_size - 1:
        raise AssertionError(
            f"memoized parent validation used {parent_lookups} lookups for "
            f"{chain_size} nodes"
        )

    try:
        _assert_soma_hemisphere_agreement(
            "synthetic_mismatch", "Right", 25.0, config.ml_midline_um
        )
    except ConstructionError:
        pass
    else:
        raise AssertionError("provider/soma hemisphere mismatch was not rejected")
    midline_provider, midline_soma = _assert_soma_hemisphere_agreement(
        "synthetic_midline", "Right", config.ml_midline_um, config.ml_midline_um
    )
    if (midline_provider, midline_soma) != ("right", "right"):
        raise AssertionError("an exact-midline soma did not follow the frozen right rule")
    return {
        "status": "pass",
        "uses_real_atlas_or_swc": False,
        "checks": [
            "voxel_boundary_split",
            "ML_midline_split_and_soma_relative_laterality",
            "descendant_to_named_target_mapping",
            "unclaimed_grey_is_QC_only",
            "fiber_tract_is_QC_only",
            "outside_volume_QC",
            "dendrite_exclusion",
            "exact_zero_preservation",
            "axon_length_conservation",
            "named_composition_normalization",
            "linear_time_memoized_parent_path_validation",
            "provider_soma_hemisphere_agreement",
        ],
        "parent_path_validation": {
            "chain_node_count": chain_size,
            "parent_lookups": parent_lookups,
        },
        "coordinate_lengths_um": list(outcome.lengths_um),
        "proportions": list(outcome.proportions),
        "qc": asdict(outcome.qc),
    }


def _load_real_assets(args: argparse.Namespace) -> tuple[
    Mapping[str, Any], Atlas, GeometryConfig, Ontology, tuple[Coordinate, ...]
]:
    try:
        import nrrd  # type: ignore
        import yaml  # type: ignore
    except ImportError as error:
        raise ConstructionError(
            "real construction requires PyYAML, NumPy, and pynrrd"
        ) from error

    with args.contract.open("r", encoding="utf-8") as handle:
        contract = yaml.safe_load(handle)
    if not isinstance(contract, dict):
        raise ConstructionError("outcome contract is not a mapping")
    target_contract = contract["target_vocabulary"]
    if int(target_contract["named_target_count"]) != 28:
        raise ConstructionError("contract must freeze 28 named targets")
    if int(target_contract["fixed_coordinate_count"]) != 56:
        raise ConstructionError("contract must freeze a 56-coordinate outcome")
    if int(target_contract.get("residual_target_count", 0)) != 0:
        raise ConstructionError("contract may not include a residual outcome target")

    atlas_contract = contract["atlas"]
    expected_annotation_hash = str(atlas_contract["annotation_sha256"])
    expected_ontology_hash = str(atlas_contract["ontology_sha256"])
    if sha256_file(args.annotation) != expected_annotation_hash:
        raise ConstructionError("annotation SHA-256 does not match frozen contract")
    if sha256_file(args.ontology) != expected_ontology_hash:
        raise ConstructionError("ontology SHA-256 does not match frozen contract")

    annotation, header = nrrd.read(str(args.annotation), index_order="F")
    atlas = ArrayAtlas(annotation)
    expected_shape = tuple(int(value) for value in atlas_contract["nrrd_header"]["sizes"])
    if atlas.shape != expected_shape:
        raise ConstructionError(
            f"annotation shape {atlas.shape} does not match frozen {expected_shape}"
        )
    if tuple(int(value) for value in header["sizes"]) != expected_shape:
        raise ConstructionError("NRRD header sizes disagree with loaded array")
    spacing = tuple(
        float(value) for value in atlas_contract["nrrd_header"]["spacing_um"]
    )
    config = GeometryConfig(
        spacing_um=spacing,  # type: ignore[arg-type]
        shape=expected_shape,  # type: ignore[arg-type]
        ml_midline_um=float(
            atlas_contract["provider_coordinate_binding"]["ML_midline_um"]
        ),
        grey_structure_id=int(atlas_contract["grey_structure_id"]),
        zero_tolerance_um=float(
            contract["regional_measure"]["segment_allocation"][
                "numerical_zero_tolerance_um"
            ]
        ),
    )
    ontology = load_ontology(args.ontology)
    coordinates = load_vocabulary(args.vocabulary, expected_count=56)
    return contract, atlas, config, ontology, coordinates


def run_real(args: argparse.Namespace) -> None:
    _, atlas, config, ontology, coordinates = _load_real_assets(args)
    anchor_to_target: dict[int, int] = {}
    for coordinate in coordinates:
        prior = anchor_to_target.setdefault(
            coordinate.anchor_structure_id, coordinate.target_index
        )
        if prior != coordinate.target_index:
            raise ConstructionError("one anchor maps to multiple target indexes")
    mapper = StructureMapper(
        ontology, anchor_to_target, grey_structure_id=config.grey_structure_id
    )
    manifest = _load_manifest(
        args.swc_manifest,
        args.swc,
        args.cell_id,
        args.provider_hemisphere,
    )
    outcomes: list[CellOutcome] = []
    for entry in manifest:
        nodes, extra_columns = parse_swc(entry.swc_path)
        outcomes.append(
            construct_cell_outcome(
                entry.cell_id,
                entry.provider_hemisphere,
                nodes,
                extra_columns,
                atlas,
                config,
                mapper,
                coordinates,
            )
        )
    write_vectors(args.output_vectors, outcomes)
    write_qc(args.output_qc, outcomes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the outcome-blind in-memory synthetic fixture and exit",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="explicitly permit reading supplied annotation and SWC outcomes",
    )
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--ontology", type=Path)
    parser.add_argument("--vocabulary", type=Path)
    parser.add_argument("--swc-manifest", type=Path)
    parser.add_argument("--swc", type=Path)
    parser.add_argument("--cell-id")
    parser.add_argument(
        "--provider-hemisphere",
        help="provider Left/Right hemisphere for single-SWC construction",
    )
    parser.add_argument("--output-vectors", type=Path)
    parser.add_argument("--output-qc", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        if args.execute:
            parser.error("--self-test and --execute are mutually exclusive")
        print(json.dumps(run_self_test(), indent=2, sort_keys=True))
        return 0
    if not args.execute:
        parser.error("real construction is gated; use --execute after EP11 activation")
    required = (
        "contract",
        "annotation",
        "ontology",
        "vocabulary",
        "output_vectors",
        "output_qc",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))
    try:
        run_real(args)
    except ConstructionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
