#!/usr/bin/env python3
"""Construct pre-candidate EP10 development terminal-bearing target calls.

The primary output is a binary 16-coordinate target set.  A coordinate is
detected only when the closure of a positive-length target component contains
a rooted axon leaf and the final positive-length subsegment of that leaf's
incoming edge is assigned to the coordinate.  Positive target length without
such a leaf is reported separately as passing-only.

Only global eligibility and the primary 16-coordinate detected, passing-only,
or no-occupancy descriptive states are exposed.  Both zero-valued subtypes are
explicitly labeled not observed in the eligible released reconstruction.
The omitted 40 coordinates, raw lengths, component/branch counts, and
prespecified sensitivities remain sealed until after candidate freeze.  Real
execution is explicitly gated by ``--execute``.
``--self-test`` is dependency-free and never reads a real annotation or SWC.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence


AXON_TYPES = frozenset((0, 2))
AXON_PARENT_TYPES = frozenset((0, 1, 2))
DENDRITE_TYPES = frozenset((3,))
MARKER_TYPES = frozenset((10, 11, 12))
ALLOWED_NODE_TYPES = AXON_TYPES | AXON_PARENT_TYPES | DENDRITE_TYPES | MARKER_TYPES
EXCLUDED_STRUCTURE_ROOTS = {
    1009: "fiber_tract",
    73: "ventricular_system",
    1024: "groove",
    304325711: "retina",
}
EXPECTED_FILES = 200
EXPECTED_GROUPS = 12
EXPECTED_PRIMARY_MODEL_FILES = 200
FROZEN_GEOMETRY_CONDITION = "exact_polyline_radius_500um_depth_tolerance_0.20"
SAFE_BRAIN_ID = re.compile(r"^[0-9]+$")
SAFE_FILE = re.compile(r"^[A-Za-z0-9.-]+_[A-Za-z0-9.-]+\.swc$")
PRIMARY_COORDINATE_IDS = (
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
MANIFEST_FIELDS = {
    "provider_neuron_id",
    "file",
    "fMOST_brain_id",
    "provider_sample_id",
    "source_acronym",
    "hemisphere",
    "reconstruction_type",
    "soma_x_ccfv3_um",
    "soma_y_ccfv3_um",
    "soma_z_ccfv3_um",
    "episode10_role",
    "EP11_SSp_tr_overlap",
    "role_status",
    "projection_outcome_exposed",
    "primary_common_support",
    "primary_model_eligible",
    "geometry_condition",
}
QC_CATEGORIES = (
    "named_target",
    "unclaimed_grey",
    "fiber_tract",
    "ventricular_system",
    "groove",
    "retina",
    "other_nongrey",
    "annotation_zero",
    "outside_volume",
)


class ConstructionError(RuntimeError):
    """A frozen input or construction invariant was violated."""


class CellIneligible(ConstructionError):
    """One cell is globally unknown under the frozen eligibility rule."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


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


@dataclass(frozen=True)
class ManifestEntry:
    provider_neuron_id: str
    filename: str
    brain_id: str
    sample_id: str
    source_acronym: str
    provider_hemisphere: str
    canonical_soma_um: tuple[float, float, float]
    primary_common_support: bool
    primary_model_eligible: bool


@dataclass(frozen=True)
class Piece:
    piece_id: int
    edge_child_id: int
    ordinal: int
    t0: float
    t1: float
    length_um: float
    category: str
    coordinate_index: int | None
    target_index: int | None
    laterality: str | None
    voxel_index: tuple[int, int, int] | None
    eroded: bool


@dataclass
class CellQC:
    provider_neuron_id: str
    filename: str
    brain_id: str
    sample_id: str
    eligible: bool = False
    failure_code: str = ""
    failure_detail: str = ""
    provider_hemisphere: str = ""
    soma_hemisphere: str = ""
    canonical_soma_agreement: bool = False
    soma_maps_to_MOp: bool = False
    node_count: int = 0
    axon_edge_count: int = 0
    positive_axon_edge_count: int = 0
    rooted_axon_leaf_count: int = 0
    dendrite_edge_count: int = 0
    marker_node_count: int = 0
    extra_column_line_count: int = 0
    total_axon_centerline_um: float = 0.0
    dendrite_centerline_excluded_um: float = 0.0
    allocation_error_um: float = 0.0
    category_lengths_um: dict[str, float] = field(
        default_factory=lambda: {category: 0.0 for category in QC_CATEGORIES}
    )


@dataclass(frozen=True)
class CoordinateQC:
    coordinate_index: int
    coordinate_id: str
    length_um: float
    occupied: bool
    component_count: int
    terminal_leaf_count: int
    terminal_component_count: int
    detected: bool
    passing_only: bool
    eroded_length_um: float
    eroded_component_count: int
    eroded_terminal_leaf_count: int
    eroded_terminal_component_count: int
    eroded_detected: bool
    eroded_passing_only: bool
    branched_terminal_detected: bool


@dataclass(frozen=True)
class CellResult:
    entry: ManifestEntry
    qc: CellQC
    coordinate_qc: tuple[CoordinateQC, ...] | None
    primary_k: int | None


class Atlas(Protocol):
    shape: tuple[int, int, int]

    def structure_id(self, index: tuple[int, int, int]) -> int:
        """Return the annotation structure ID at an in-bounds index."""


class ArrayAtlas:
    def __init__(self, data: Any):
        if getattr(data, "ndim", None) != 3:
            raise ConstructionError("annotation must be three-dimensional")
        self._data = data
        self.shape = tuple(int(value) for value in data.shape)

    def structure_id(self, index: tuple[int, int, int]) -> int:
        return int(self._data[index])


class SyntheticAtlas:
    def __init__(
        self,
        shape: tuple[int, int, int],
        values: Mapping[tuple[int, int, int], int],
        default: int = 0,
    ):
        self.shape = shape
        self._values = dict(values)
        self._default = default

    def structure_id(self, index: tuple[int, int, int]) -> int:
        return int(self._values.get(index, self._default))


class Ontology:
    def __init__(
        self,
        parent_by_id: Mapping[int, int | None],
        acronym_by_id: Mapping[int, str] | None = None,
    ):
        self.parent_by_id = dict(parent_by_id)
        self.acronym_by_id = dict(acronym_by_id or {})
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

    def unique_id_for_acronym(self, acronym: str) -> int:
        matches = [
            structure_id
            for structure_id, value in self.acronym_by_id.items()
            if value == acronym
        ]
        if len(matches) != 1:
            raise ConstructionError(
                f"ontology must contain exactly one {acronym!r}; found {matches}"
            )
        return matches[0]


class StructureMapper:
    def __init__(
        self,
        ontology: Ontology,
        anchor_to_target: Mapping[int, int],
        grey_structure_id: int,
    ):
        self.ontology = ontology
        self.anchor_to_target = dict(anchor_to_target)
        self.grey_structure_id = grey_structure_id
        if len(self.anchor_to_target) != len(set(self.anchor_to_target.values())):
            raise ConstructionError("target anchors do not map one-to-one to targets")
        anchors = tuple(self.anchor_to_target)
        for anchor in anchors:
            if not ontology.is_descendant_or_self(anchor, grey_structure_id):
                raise ConstructionError(f"target anchor {anchor} is not below Allen grey")
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
            result = ("named_target", matches[0])
        else:
            excluded = [
                category
                for root, category in EXCLUDED_STRUCTURE_ROOTS.items()
                if root in lineage
            ]
            if len(excluded) > 1:
                raise ConstructionError(
                    f"structure {structure_id} maps to multiple QC exclusions"
                )
            if excluded:
                result = (excluded[0], None)
            elif self.grey_structure_id in lineage:
                result = ("unclaimed_grey", None)
            else:
                result = ("other_nongrey", None)
        self._cache[structure_id] = result
        return result


class ErosionClassifier:
    """One-voxel 6-neighbor internal erosion of each named target mask."""

    def __init__(self, atlas: Atlas, mapper: StructureMapper):
        self.atlas = atlas
        self.mapper = mapper
        self._cache: dict[tuple[tuple[int, int, int], int], bool] = {}

    def is_interior(
        self, index: tuple[int, int, int], target_index: int
    ) -> bool:
        key = (index, target_index)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        for dimension in range(3):
            for step in (-1, 1):
                neighbor = list(index)
                neighbor[dimension] += step
                if not 0 <= neighbor[dimension] < self.atlas.shape[dimension]:
                    self._cache[key] = False
                    return False
                structure_id = self.atlas.structure_id(tuple(neighbor))
                if structure_id == 0:
                    self._cache[key] = False
                    return False
                category, observed_target = self.mapper.classify(structure_id)
                if category != "named_target" or observed_target != target_index:
                    self._cache[key] = False
                    return False
        self._cache[key] = True
        return True


class DisjointSet:
    def __init__(self, values: Iterable[int]):
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            following = self.parent[value]
            self.parent[value] = root
            value = following
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _strict_int(token: str, field_name: str, line_number: int) -> int:
    try:
        return int(token)
    except ValueError as error:
        raise ConstructionError(
            f"SWC line {line_number}: {field_name} must be an integer"
        ) from error


def _validate_parent_paths(nodes: Mapping[int, SwcNode], root_id: int) -> None:
    validated = {root_id}
    for start in nodes:
        if start in validated:
            continue
        current = start
        path: list[int] = []
        on_path: set[int] = set()
        while current not in validated:
            if current in on_path:
                raise ConstructionError(f"SWC contains a cycle through node {current}")
            if current not in nodes:
                raise ConstructionError(f"SWC refers to missing parent {current}")
            on_path.add(current)
            path.append(current)
            parent = nodes[current].parent_id
            if parent == -1:
                if current != root_id:
                    raise ConstructionError(
                        f"node {start} reaches noncanonical root {current}"
                    )
                break
            current = parent
        validated.update(path)


def parse_swc_lines(lines: Iterable[str]) -> tuple[dict[int, SwcNode], int]:
    nodes: dict[int, SwcNode] = {}
    extra_columns = 0
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 7:
            raise ConstructionError(f"SWC line {line_number}: fewer than seven columns")
        if len(fields) > 7:
            extra_columns += 1
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
            x, y, z, radius = (float(fields[index]) for index in range(2, 6))
        except ValueError as error:
            raise ConstructionError(
                f"SWC line {line_number}: coordinates/radius must be numeric"
            ) from error
        if not all(math.isfinite(value) for value in (x, y, z, radius)):
            raise ConstructionError(
                f"SWC line {line_number}: coordinates/radius must be finite"
            )
        if radius < 0.0:
            raise ConstructionError(
                f"SWC line {line_number}: radius must be nonnegative"
            )
        nodes[node_id] = SwcNode(
            node_id=node_id,
            node_type=node_type,
            point_um=(x, y, z),
            radius_um=radius,
            parent_id=parent_id,
        )
    if not nodes:
        raise ConstructionError("SWC contains no nodes")
    roots = [node for node in nodes.values() if node.parent_id == -1]
    if len(roots) != 1:
        raise ConstructionError(f"SWC must have exactly one root; found {len(roots)}")
    if roots[0].node_type != 1:
        raise ConstructionError("the unique SWC root must be soma type 1")
    soma_nodes = [node for node in nodes.values() if node.node_type == 1]
    if len(soma_nodes) != 1 or soma_nodes[0].node_id != roots[0].node_id:
        raise ConstructionError(
            "SWC must contain exactly one soma type 1 node, and it must be the root"
        )
    for node in nodes.values():
        if node.parent_id != -1 and node.parent_id not in nodes:
            raise ConstructionError(
                f"SWC node {node.node_id} refers to missing parent {node.parent_id}"
            )
    _validate_parent_paths(nodes, roots[0].node_id)
    children: dict[int, list[int]] = defaultdict(list)
    for node in nodes.values():
        if node.parent_id == -1:
            continue
        parent = nodes[node.parent_id]
        children[parent.node_id].append(node.node_id)
        if node.node_type in AXON_TYPES:
            allowed_parents = AXON_PARENT_TYPES
        elif node.node_type in DENDRITE_TYPES:
            allowed_parents = frozenset((1, 3))
        elif node.node_type in MARKER_TYPES:
            allowed_parents = frozenset((0, 1, 2, 3))
        else:
            allowed_parents = frozenset()
        if parent.node_type not in allowed_parents:
            raise ConstructionError(
                f"SWC transition {parent.node_type}->{node.node_type} is not allowed "
                f"for parent {parent.node_id} and child {node.node_id}"
            )
    for marker in (node for node in nodes.values() if node.node_type in MARKER_TYPES):
        if children.get(marker.node_id):
            raise ConstructionError(
                f"SWC marker node {marker.node_id} must be a leaf"
            )
    return nodes, extra_columns


def parse_swc(path: Path) -> tuple[dict[int, SwcNode], int]:
    with path.open("r", encoding="utf-8") as handle:
        return parse_swc_lines(handle)


def euclidean_length(
    start: tuple[float, float, float], end: tuple[float, float, float]
) -> float:
    return math.sqrt(sum((end[index] - start[index]) ** 2 for index in range(3)))


def split_parameters(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    config: GeometryConfig,
) -> tuple[float, ...]:
    parameters = [0.0, 1.0]
    for dimension in range(3):
        delta = end[dimension] - start[dimension]
        if delta == 0.0:
            continue
        spacing = config.spacing_um[dimension]
        extent = config.shape[dimension] * spacing
        low = min(start[dimension], end[dimension])
        high = max(start[dimension], end[dimension])
        first_boundary = max(0, math.ceil(low / spacing))
        last_boundary = min(config.shape[dimension], math.floor(high / spacing))
        for boundary_index in range(first_boundary, last_boundary + 1):
            boundary = boundary_index * spacing
            parameter = (boundary - start[dimension]) / delta
            if 0.0 < parameter < 1.0:
                parameters.append(parameter)
        if dimension == 2 and 0.0 <= config.ml_midline_um <= extent:
            parameter = (config.ml_midline_um - start[dimension]) / delta
            if 0.0 < parameter < 1.0:
                parameters.append(parameter)
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
    parameter: float,
) -> tuple[float, float, float]:
    return tuple(
        start[index] + parameter * (end[index] - start[index])
        for index in range(3)
    )  # type: ignore[return-value]


def _voxel_index(
    point: tuple[float, float, float], config: GeometryConfig
) -> tuple[int, int, int] | None:
    result = tuple(
        math.floor(point[dimension] / config.spacing_um[dimension])
        for dimension in range(3)
    )
    if any(value < 0 or value >= config.shape[index] for index, value in enumerate(result)):
        return None
    return result  # type: ignore[return-value]


def _normalize_hemisphere(value: str) -> str:
    normalized = value.strip().casefold()
    aliases = {"left": "left", "l": "left", "right": "right", "r": "right"}
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ConstructionError(f"invalid provider hemisphere: {value!r}") from error


def _soma_hemisphere(ml_um: float, midline_um: float) -> str:
    return "left" if ml_um < midline_um else "right"


def _relative_laterality(
    segment_ml_um: float, soma_ml_um: float, midline_um: float
) -> str:
    segment_side = _soma_hemisphere(segment_ml_um, midline_um)
    soma_side = _soma_hemisphere(soma_ml_um, midline_um)
    return "ipsilateral" if segment_side == soma_side else "contralateral"


def allocate_edge(
    child_id: int,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    soma_ml_um: float,
    atlas: Atlas,
    config: GeometryConfig,
    mapper: StructureMapper,
    erosion: ErosionClassifier | None,
    coordinate_lookup: Mapping[tuple[int, str], int],
    next_piece_id: int,
    qc: CellQC,
) -> list[Piece]:
    total_length = euclidean_length(start, end)
    if total_length <= config.zero_tolerance_um:
        return []
    pieces: list[Piece] = []
    allocated = 0.0
    parameters = split_parameters(start, end, config)
    for ordinal, (left, right) in enumerate(zip(parameters, parameters[1:])):
        fraction = right - left
        if fraction <= 0.0:
            continue
        length = total_length * fraction
        midpoint = _point_at(start, end, (left + right) / 2.0)
        voxel = _voxel_index(midpoint, config)
        category: str
        target_index: int | None = None
        laterality: str | None = None
        coordinate_index: int | None = None
        eroded = False
        if voxel is None:
            category = "outside_volume"
        else:
            structure_id = atlas.structure_id(voxel)
            if structure_id == 0:
                category = "annotation_zero"
            else:
                category, target_index = mapper.classify(structure_id)
                if category == "named_target":
                    assert target_index is not None
                    laterality = _relative_laterality(
                        midpoint[2], soma_ml_um, config.ml_midline_um
                    )
                    coordinate_index = coordinate_lookup.get(
                        (target_index, laterality)
                    )
                    if coordinate_index is not None and erosion is not None:
                        eroded = erosion.is_interior(voxel, target_index)
        qc.category_lengths_um[category] += length
        pieces.append(
            Piece(
                piece_id=next_piece_id + len(pieces),
                edge_child_id=child_id,
                ordinal=ordinal,
                t0=left,
                t1=right,
                length_um=length,
                category=category,
                coordinate_index=coordinate_index,
                target_index=target_index,
                laterality=laterality,
                voxel_index=voxel,
                eroded=eroded,
            )
        )
        allocated += length
    tolerance = max(config.zero_tolerance_um, total_length * 1.0e-11)
    if abs(allocated - total_length) > tolerance:
        raise ConstructionError(
            f"edge {child_id} allocation failed conservation by "
            f"{allocated - total_length:.12g} um"
        )
    return pieces


def _component_statistics(
    pieces: Sequence[Piece],
    edge_piece_ids: Mapping[int, Sequence[int]],
    nodes: Mapping[int, SwcNode],
    axon_children: Mapping[int, Sequence[int]],
    coordinate_count: int,
    eroded_only: bool,
) -> tuple[list[int], list[int], list[int], dict[int, int]]:
    """Return component, terminal-leaf, and terminal-component counts.

    The returned leaf map records the coordinate of the final positive-length
    incoming leaf-edge subsegment.  Connectivity is by shared subsegment
    closure along an edge or at an SWC node.
    """

    by_id = {piece.piece_id: piece for piece in pieces}

    def retained(piece: Piece) -> bool:
        return piece.coordinate_index is not None and (
            piece.eroded if eroded_only else True
        )

    retained_ids = [piece.piece_id for piece in pieces if retained(piece)]
    disjoint = DisjointSet(retained_ids)

    for piece_ids in edge_piece_ids.values():
        for left_id, right_id in zip(piece_ids, piece_ids[1:]):
            left = by_id[left_id]
            right = by_id[right_id]
            if (
                retained(left)
                and retained(right)
                and left.coordinate_index == right.coordinate_index
            ):
                disjoint.union(left_id, right_id)

    incident: dict[int, list[int]] = defaultdict(list)
    for child_id, piece_ids in edge_piece_ids.items():
        if not piece_ids:
            continue
        parent_id = nodes[child_id].parent_id
        first = by_id[piece_ids[0]]
        last = by_id[piece_ids[-1]]
        if retained(first) and first.t0 == 0.0:
            incident[parent_id].append(first.piece_id)
        if retained(last) and last.t1 == 1.0:
            incident[child_id].append(last.piece_id)
    for incident_ids in incident.values():
        by_coordinate: dict[int, list[int]] = defaultdict(list)
        for piece_id in incident_ids:
            coordinate = by_id[piece_id].coordinate_index
            assert coordinate is not None
            by_coordinate[coordinate].append(piece_id)
        for piece_ids in by_coordinate.values():
            for following in piece_ids[1:]:
                disjoint.union(piece_ids[0], following)

    components: list[set[int]] = [set() for _ in range(coordinate_count)]
    for piece_id in retained_ids:
        coordinate = by_id[piece_id].coordinate_index
        assert coordinate is not None
        components[coordinate].add(disjoint.find(piece_id))

    leaf_coordinate: dict[int, int] = {}
    terminal_roots: list[set[int]] = [set() for _ in range(coordinate_count)]
    terminal_leaf_count = [0] * coordinate_count
    nodes_with_any_child = {
        node.parent_id for node in nodes.values() if node.parent_id != -1
    }
    rooted_leaves = [
        node_id
        for node_id, node in nodes.items()
        if node.node_type in AXON_TYPES and node_id not in nodes_with_any_child
    ]
    for leaf_id in rooted_leaves:
        piece_ids = edge_piece_ids.get(leaf_id, ())
        if not piece_ids:
            continue
        final_piece = by_id[piece_ids[-1]]
        if final_piece.t1 != 1.0 or not retained(final_piece):
            continue
        coordinate = final_piece.coordinate_index
        assert coordinate is not None
        leaf_coordinate[leaf_id] = coordinate
        terminal_leaf_count[coordinate] += 1
        terminal_roots[coordinate].add(disjoint.find(final_piece.piece_id))

    return (
        [len(values) for values in components],
        terminal_leaf_count,
        [len(values) for values in terminal_roots],
        leaf_coordinate,
    )


def _branched_terminal_calls(
    pieces: Sequence[Piece],
    edge_piece_ids: Mapping[int, Sequence[int]],
    nodes: Mapping[int, SwcNode],
    axon_children: Mapping[int, Sequence[int]],
    leaf_coordinate: Mapping[int, int],
    coordinate_count: int,
) -> list[bool]:
    """Apply the frozen same-coordinate branched-terminal sensitivity."""

    by_id = {piece.piece_id: piece for piece in pieces}
    edge_coordinate: dict[int, int | None | str] = {}
    for child_id in (
        node_id for node_id, node in nodes.items() if node.node_type in AXON_TYPES
    ):
        piece_ids = edge_piece_ids.get(child_id, ())
        if not piece_ids:
            edge_coordinate[child_id] = "zero_length"
            continue
        observed = {by_id[piece_id].coordinate_index for piece_id in piece_ids}
        if len(observed) == 1 and None not in observed:
            edge_coordinate[child_id] = next(iter(observed))
        else:
            edge_coordinate[child_id] = None

    detected = [False] * coordinate_count
    for branch_id, children in axon_children.items():
        branch = nodes[branch_id]
        if branch.node_type not in AXON_TYPES or len(children) < 2:
            continue
        qualifying_child_coordinates: list[set[int]] = []
        for initial_child in children:
            child_coordinates: set[int] = set()
            stack: list[tuple[int, int | None]] = [(initial_child, None)]
            while stack:
                node_id, path_coordinate = stack.pop()
                current = edge_coordinate[node_id]
                if current is None:
                    continue
                if current != "zero_length":
                    assert isinstance(current, int)
                    if path_coordinate is None:
                        path_coordinate = current
                    elif path_coordinate != current:
                        continue
                descendants = axon_children.get(node_id, ())
                if not descendants:
                    if (
                        path_coordinate is not None
                        and leaf_coordinate.get(node_id) == path_coordinate
                    ):
                        child_coordinates.add(path_coordinate)
                    continue
                for child_id in descendants:
                    stack.append((child_id, path_coordinate))
            qualifying_child_coordinates.append(child_coordinates)
        support: Counter[int] = Counter()
        for coordinates in qualifying_child_coordinates:
            support.update(coordinates)
        for coordinate, child_branch_count in support.items():
            if child_branch_count >= 2:
                detected[coordinate] = True
    return detected


def construct_cell(
    entry: ManifestEntry,
    nodes: Mapping[int, SwcNode],
    extra_columns: int,
    atlas: Atlas,
    config: GeometryConfig,
    ontology: Ontology,
    mapper: StructureMapper,
    coordinates: Sequence[Coordinate],
    primary_indexes: Sequence[int],
    run_sealed_sensitivities: bool = False,
) -> CellResult:
    qc = CellQC(
        provider_neuron_id=entry.provider_neuron_id,
        filename=entry.filename,
        brain_id=entry.brain_id,
        sample_id=entry.sample_id,
        provider_hemisphere=_normalize_hemisphere(entry.provider_hemisphere),
        node_count=len(nodes),
        marker_node_count=sum(node.node_type in MARKER_TYPES for node in nodes.values()),
        extra_column_line_count=extra_columns,
    )
    roots = [node for node in nodes.values() if node.parent_id == -1]
    if len(roots) != 1 or roots[0].node_type != 1:
        raise CellIneligible("root_soma", "validated root/soma invariant was lost")
    root = roots[0]
    if root.point_um != entry.canonical_soma_um:
        raise CellIneligible(
            "canonical_soma_mismatch",
            f"SWC soma {root.point_um} != frozen soma {entry.canonical_soma_um}",
        )
    qc.canonical_soma_agreement = True
    qc.soma_hemisphere = _soma_hemisphere(
        root.point_um[2], config.ml_midline_um
    )
    if qc.provider_hemisphere != qc.soma_hemisphere:
        raise CellIneligible(
            "provider_soma_hemisphere_mismatch",
            f"provider={qc.provider_hemisphere} soma={qc.soma_hemisphere}",
        )

    soma_index = _voxel_index(root.point_um, config)
    if soma_index is None:
        raise CellIneligible("soma_outside_volume", "registered soma is out of bounds")
    soma_structure = atlas.structure_id(soma_index)
    if soma_structure == 0:
        raise CellIneligible("soma_annotation_zero", "registered soma maps to zero")
    mop_id = ontology.unique_id_for_acronym("MOp")
    if not ontology.is_descendant_or_self(soma_structure, mop_id):
        raise CellIneligible(
            "soma_not_MOp",
            f"registered soma structure {soma_structure} is not MOp",
        )
    qc.soma_maps_to_MOp = True

    coordinate_lookup = {
        (coordinate.target_index, coordinate.laterality): coordinate.index
        for coordinate in coordinates
    }
    if len(coordinate_lookup) != len(coordinates):
        raise ConstructionError("duplicate target/laterality coordinate")
    erosion = ErosionClassifier(atlas, mapper) if run_sealed_sensitivities else None

    axon_children: dict[int, list[int]] = defaultdict(list)
    for child in nodes.values():
        if child.node_type not in AXON_TYPES:
            continue
        if child.parent_id == -1:
            raise CellIneligible("axon_root", "axon node may not be an SWC root")
        parent = nodes[child.parent_id]
        if parent.node_type not in AXON_PARENT_TYPES:
            raise CellIneligible(
                "axon_parent_type",
                f"axon node {child.node_id} has parent type {parent.node_type}",
            )
        axon_children[parent.node_id].append(child.node_id)

    pieces: list[Piece] = []
    edge_piece_ids: dict[int, list[int]] = {}
    for child in nodes.values():
        if child.parent_id == -1:
            continue
        parent = nodes[child.parent_id]
        edge_length = euclidean_length(parent.point_um, child.point_um)
        if child.node_type in AXON_TYPES:
            qc.axon_edge_count += 1
            qc.total_axon_centerline_um += edge_length
            if edge_length > config.zero_tolerance_um:
                qc.positive_axon_edge_count += 1
            edge_pieces = allocate_edge(
                child.node_id,
                parent.point_um,
                child.point_um,
                root.point_um[2],
                atlas,
                config,
                mapper,
                erosion,
                coordinate_lookup,
                len(pieces),
                qc,
            )
            edge_piece_ids[child.node_id] = [piece.piece_id for piece in edge_pieces]
            pieces.extend(edge_pieces)
        elif child.node_type in DENDRITE_TYPES:
            qc.dendrite_edge_count += 1
            qc.dendrite_centerline_excluded_um += edge_length

    if qc.positive_axon_edge_count == 0:
        raise CellIneligible(
            "no_positive_axon_edge", "cell has no positive-length axon edge"
        )
    allocated_total = sum(qc.category_lengths_um.values())
    qc.allocation_error_um = allocated_total - qc.total_axon_centerline_um
    tolerance = max(
        config.zero_tolerance_um, qc.total_axon_centerline_um * 1.0e-11
    )
    if abs(qc.allocation_error_um) > tolerance:
        raise CellIneligible(
            "axon_length_nonconservation",
            f"allocation error={qc.allocation_error_um:.12g} um",
        )
    if qc.category_lengths_um["outside_volume"] > config.zero_tolerance_um:
        raise CellIneligible(
            "axon_outside_volume",
            f"outside length={qc.category_lengths_um['outside_volume']:.12g} um",
        )
    if qc.category_lengths_um["annotation_zero"] > config.zero_tolerance_um:
        raise CellIneligible(
            "axon_annotation_zero",
            f"zero-label length={qc.category_lengths_um['annotation_zero']:.12g} um",
        )

    nodes_with_any_child = {
        node.parent_id for node in nodes.values() if node.parent_id != -1
    }
    rooted_leaves = [
        node_id
        for node_id, node in nodes.items()
        if node.node_type in AXON_TYPES and node_id not in nodes_with_any_child
    ]
    qc.rooted_axon_leaf_count = len(rooted_leaves)
    for leaf_id in rooted_leaves:
        leaf = nodes[leaf_id]
        leaf_index = _voxel_index(leaf.point_um, config)
        if leaf_index is None:
            raise CellIneligible(
                "terminal_leaf_outside_volume", f"leaf {leaf_id} is outside volume"
            )
        structure_id = atlas.structure_id(leaf_index)
        if structure_id == 0:
            raise CellIneligible(
                "terminal_leaf_annotation_zero", f"leaf {leaf_id} maps to zero"
            )
        category, _ = mapper.classify(structure_id)
        if category not in {"named_target", "unclaimed_grey"}:
            raise CellIneligible(
                "terminal_leaf_nongrey",
                f"leaf {leaf_id} maps to {category}",
            )

    qc.eligible = True
    if not entry.primary_model_eligible:
        return CellResult(entry, qc, None, None)

    native_components, native_leaves, native_terminal_components, native_leaf_map = (
        _component_statistics(
            pieces,
            edge_piece_ids,
            nodes,
            axon_children,
            len(coordinates),
            eroded_only=False,
        )
    )
    if run_sealed_sensitivities:
        eroded_components, eroded_leaves, eroded_terminal_components, _ = (
            _component_statistics(
                pieces,
                edge_piece_ids,
                nodes,
                axon_children,
                len(coordinates),
                eroded_only=True,
            )
        )
        branched = _branched_terminal_calls(
            pieces,
            edge_piece_ids,
            nodes,
            axon_children,
            native_leaf_map,
            len(coordinates),
        )
    else:
        eroded_components = [0] * len(coordinates)
        eroded_leaves = [0] * len(coordinates)
        eroded_terminal_components = [0] * len(coordinates)
        branched = [False] * len(coordinates)
    lengths = [0.0] * len(coordinates)
    eroded_lengths = [0.0] * len(coordinates)
    for piece in pieces:
        if piece.coordinate_index is None:
            continue
        lengths[piece.coordinate_index] += piece.length_um
        if piece.eroded:
            eroded_lengths[piece.coordinate_index] += piece.length_um

    coordinate_qc: list[CoordinateQC] = []
    for coordinate in coordinates:
        index = coordinate.index
        detected = native_terminal_components[index] > 0
        eroded_detected = eroded_terminal_components[index] > 0
        coordinate_qc.append(
            CoordinateQC(
                coordinate_index=index,
                coordinate_id=coordinate.coordinate_id,
                length_um=lengths[index],
                occupied=lengths[index] > config.zero_tolerance_um,
                component_count=native_components[index],
                terminal_leaf_count=native_leaves[index],
                terminal_component_count=native_terminal_components[index],
                detected=detected,
                passing_only=(
                    lengths[index] > config.zero_tolerance_um and not detected
                ),
                eroded_length_um=eroded_lengths[index],
                eroded_component_count=eroded_components[index],
                eroded_terminal_leaf_count=eroded_leaves[index],
                eroded_terminal_component_count=eroded_terminal_components[index],
                eroded_detected=eroded_detected,
                eroded_passing_only=(
                    eroded_lengths[index] > config.zero_tolerance_um
                    and not eroded_detected
                ),
                branched_terminal_detected=branched[index],
            )
        )
    primary_k = sum(coordinate_qc[index].detected for index in primary_indexes)
    return CellResult(entry, qc, tuple(coordinate_qc), primary_k)


def validate_manifest_rows(
    rows: Sequence[Mapping[str, str]],
    expected_files: int = EXPECTED_FILES,
    expected_groups: int = EXPECTED_GROUPS,
    expected_primary_model_files: int = EXPECTED_PRIMARY_MODEL_FILES,
) -> list[ManifestEntry]:
    if len(rows) != expected_files:
        raise ConstructionError(
            f"development manifest must contain {expected_files} rows; found {len(rows)}"
        )
    entries: list[ManifestEntry] = []
    provider_ids: set[str] = set()
    filenames: set[str] = set()
    brain_samples: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=2):
        missing = sorted(MANIFEST_FIELDS - set(row))
        if missing:
            raise ConstructionError(
                f"manifest row {row_number} lacks fields: {missing}"
            )
        filename = row["file"]
        brain_id = row["fMOST_brain_id"]
        provider_id = row["provider_neuron_id"]
        if row["episode10_role"] != "development":
            raise ConstructionError(
                f"manifest row {row_number} is not development: {filename}"
            )
        if row["EP11_SSp_tr_overlap"] != "no":
            raise ConstructionError(
                f"manifest row {row_number} overlaps EP11: {filename}"
            )
        if row["role_status"] != "frozen_before_projection_outcomes":
            raise ConstructionError(
                f"manifest row {row_number} role is not frozen: {filename}"
            )
        if row["projection_outcome_exposed"] != "no":
            raise ConstructionError(
                f"manifest row {row_number} was previously exposed: {filename}"
            )
        if row["reconstruction_type"] != "Axon_and_dendrite":
            raise ConstructionError(
                f"manifest row {row_number} is not primary reconstruction: {filename}"
            )
        support = row["primary_common_support"]
        model_eligible = row["primary_model_eligible"]
        if support not in {"yes", "no"}:
            raise ConstructionError(
                f"invalid primary common-support flag at row {row_number}"
            )
        if model_eligible not in {"yes", "no"}:
            raise ConstructionError(
                f"invalid primary model-eligibility flag at row {row_number}"
            )
        if model_eligible != support:
            raise ConstructionError(
                f"support/model eligibility mismatch at row {row_number}"
            )
        if row["geometry_condition"] != FROZEN_GEOMETRY_CONDITION:
            raise ConstructionError(
                f"common-support geometry condition changed at row {row_number}"
            )
        if not row["source_acronym"].startswith("MOp"):
            raise ConstructionError(
                f"manifest row {row_number} is not MOp: {filename}"
            )
        if not SAFE_BRAIN_ID.fullmatch(brain_id):
            raise ConstructionError(f"unsafe fMOST brain ID at row {row_number}")
        if not SAFE_FILE.fullmatch(filename) or filename.split("_", 1)[0] != brain_id:
            raise ConstructionError(f"unsafe or mismatched filename at row {row_number}")
        if not provider_id.isdigit() or provider_id in provider_ids:
            raise ConstructionError(
                f"invalid or duplicate provider neuron ID at row {row_number}"
            )
        if filename in filenames:
            raise ConstructionError(f"duplicate filename at row {row_number}")
        sample_id = row["provider_sample_id"]
        if not sample_id:
            raise ConstructionError(f"empty sample ID at row {row_number}")
        prior_sample = brain_samples.setdefault(brain_id, sample_id)
        if prior_sample != sample_id:
            raise ConstructionError(f"multiple sample IDs for fMOST brain {brain_id}")
        try:
            soma = tuple(
                float(row[field_name])
                for field_name in (
                    "soma_x_ccfv3_um",
                    "soma_y_ccfv3_um",
                    "soma_z_ccfv3_um",
                )
            )
        except ValueError as error:
            raise ConstructionError(
                f"nonnumeric frozen soma at row {row_number}"
            ) from error
        if not all(math.isfinite(value) for value in soma):
            raise ConstructionError(f"nonfinite frozen soma at row {row_number}")
        hemisphere = _normalize_hemisphere(row["hemisphere"])
        provider_ids.add(provider_id)
        filenames.add(filename)
        entries.append(
            ManifestEntry(
                provider_neuron_id=provider_id,
                filename=filename,
                brain_id=brain_id,
                sample_id=sample_id,
                source_acronym=row["source_acronym"],
                provider_hemisphere=hemisphere,
                canonical_soma_um=soma,  # type: ignore[arg-type]
                primary_common_support=support == "yes",
                primary_model_eligible=model_eligible == "yes",
            )
        )
    if len(brain_samples) != expected_groups:
        raise ConstructionError(
            f"development manifest must contain {expected_groups} groups; "
            f"found {len(brain_samples)}"
        )
    if entries != sorted(entries, key=lambda value: (value.brain_id, value.filename)):
        raise ConstructionError("development manifest is not canonically sorted")
    observed_primary = sum(entry.primary_model_eligible for entry in entries)
    if observed_primary != expected_primary_model_files:
        raise ConstructionError(
            "development manifest must contain "
            f"{expected_primary_model_files} primary-model rows; "
            f"found {observed_primary}"
        )
    return entries


def load_manifest(path: Path) -> list[ManifestEntry]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(MANIFEST_FIELDS - fields)
        if missing:
            raise ConstructionError(f"development manifest lacks fields: {missing}")
        rows = list(reader)
    return validate_manifest_rows(rows)


def _validate_scratch_path(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ConstructionError(f"{label} must be an absolute Scratch path")
    if path.is_symlink():
        raise ConstructionError(f"refusing symlink {label}: {path}")
    resolved = path.resolve(strict=False)
    roots = [
        Path(value).resolve(strict=False)
        for value in (os.environ.get("SCRATCH"), os.environ.get("GROUP_SCRATCH"))
        if value
    ]
    if not roots or not any(root != resolved and root in resolved.parents for root in roots):
        raise ConstructionError(
            f"{label} must be below SCRATCH or GROUP_SCRATCH: {resolved}"
        )
    return resolved


def validate_exact_swc_set(
    entries: Sequence[ManifestEntry], swc_root: Path
) -> dict[str, Path]:
    if not swc_root.is_dir() or swc_root.is_symlink():
        raise ConstructionError(f"development SWC root is absent or unsafe: {swc_root}")
    expected = {
        f"{entry.brain_id}/{entry.filename}": entry for entry in entries
    }
    observed: dict[str, Path] = {}
    for directory, directory_names, filenames in os.walk(swc_root, followlinks=False):
        base = Path(directory)
        for name in directory_names:
            if (base / name).is_symlink():
                raise ConstructionError(f"refusing symlink SWC directory: {base / name}")
        for name in filenames:
            path = base / name
            if path.is_symlink():
                raise ConstructionError(f"refusing symlink SWC file: {path}")
            if name.endswith(".swc"):
                observed[path.relative_to(swc_root).as_posix()] = path
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise ConstructionError(
            "development SWC set differs from frozen manifest: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    return {
        entry.provider_neuron_id: observed[f"{entry.brain_id}/{entry.filename}"]
        for entry in entries
    }


def _flatten_ontology_tree(
    root: Mapping[str, Any],
) -> tuple[dict[int, int | None], dict[int, str]]:
    parents: dict[int, int | None] = {}
    acronyms: dict[int, str] = {}
    stack: list[tuple[Mapping[str, Any], int | None]] = [(root, None)]
    while stack:
        node, expected_parent = stack.pop()
        try:
            structure_id = int(node["id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ConstructionError("ontology node lacks integer ID") from error
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
        acronyms[structure_id] = str(node.get("acronym", ""))
        children = node.get("children", [])
        if not isinstance(children, list):
            raise ConstructionError(f"ontology children not a list at {structure_id}")
        for child in reversed(children):
            if not isinstance(child, Mapping):
                raise ConstructionError(f"malformed ontology child at {structure_id}")
            stack.append((child, structure_id))
    return parents, acronyms


def load_ontology(path: Path) -> Ontology:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    roots = payload.get("msg") if isinstance(payload, dict) else None
    if not isinstance(roots, list) or len(roots) != 1:
        raise ConstructionError("ontology JSON must contain exactly one msg root")
    parents, acronyms = _flatten_ontology_tree(roots[0])
    return Ontology(parents, acronyms)


def load_vocabulary(path: Path) -> tuple[Coordinate, ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 56:
        raise ConstructionError(f"full vocabulary must contain 56 rows; found {len(rows)}")
    coordinates: list[Coordinate] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            coordinate = Coordinate(
                index=int(row["coordinate_index"]),
                coordinate_id=row["coordinate_id"],
                target_index=int(row["target_index"]),
                target_id=row["target_id"],
                anchor_structure_id=int(row["allen_structure_id"]),
                laterality=row["laterality"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ConstructionError(f"malformed vocabulary row {row_number}") from error
        if coordinate.laterality not in {"ipsilateral", "contralateral"}:
            raise ConstructionError(f"invalid laterality at vocabulary row {row_number}")
        coordinates.append(coordinate)
    coordinates.sort(key=lambda value: value.index)
    if tuple(value.index for value in coordinates) != tuple(range(56)):
        raise ConstructionError("vocabulary indexes must be 0..55")
    targets: dict[int, list[Coordinate]] = defaultdict(list)
    for coordinate in coordinates:
        targets[coordinate.target_index].append(coordinate)
    if tuple(sorted(targets)) != tuple(range(28)):
        raise ConstructionError("vocabulary target indexes must be 0..27")
    for target_index, pair in targets.items():
        if len(pair) != 2 or {value.laterality for value in pair} != {
            "ipsilateral",
            "contralateral",
        }:
            raise ConstructionError(
                f"target {target_index} lacks one ipsilateral/contralateral pair"
            )
        if len({value.anchor_structure_id for value in pair}) != 1:
            raise ConstructionError(f"target {target_index} has inconsistent anchors")
    return tuple(coordinates)


def _load_real_assets(
    contract_path: Path,
    annotation_path: Path,
    ontology_path: Path,
    vocabulary_path: Path,
) -> tuple[
    Mapping[str, Any],
    Atlas,
    GeometryConfig,
    Ontology,
    tuple[Coordinate, ...],
    tuple[Coordinate, ...],
]:
    try:
        import nrrd  # type: ignore
        import yaml  # type: ignore
    except ImportError as error:
        raise ConstructionError("real execution requires PyYAML and pynrrd") from error

    with contract_path.open("r", encoding="utf-8") as handle:
        contract = yaml.safe_load(handle)
    if not isinstance(contract, dict):
        raise ConstructionError("projection observation contract is not a mapping")
    if contract.get("status") != "frozen_before_any_EP10_projection_outcome_access":
        raise ConstructionError("projection observation contract is not frozen")
    if contract.get("real_SWC_or_projection_outcomes_opened_by_this_work") is not False:
        raise ConstructionError("contract access ledger is not outcome-clean")
    roles = contract["scope"]["roles"]
    if roles != {
        "development_groups": 12,
        "final_groups": 8,
        "unassigned_groups": 56,
    }:
        raise ConstructionError("contract role counts differ from the frozen ledger")
    support = contract["frozen_common_support_and_attrition"][
        "development_primary_common_support"
    ]
    if (
        support.get("geometry_condition")
        != "exact_polyline_radius_500um_depth_tolerance_0.20"
        or int(support.get("development_role_rows", -1)) != 209
        or int(support.get("development_acquisition_and_constructor_rows", -1))
        != EXPECTED_FILES
        or int(support.get("primary_model_eligible_rows", -1))
        != EXPECTED_PRIMARY_MODEL_FILES
        or int(support.get("unsupported_development_rows_not_acquired_or_opened", -1))
        != 9
    ):
        raise ConstructionError("development common-support domain is not frozen")
    panel = contract["primary_exact_model_panel"]
    if int(panel["coordinate_count"]) != 16:
        raise ConstructionError("primary panel must contain 16 coordinates")
    panel_ids = tuple(panel["coordinate_ids"])
    if panel_ids != PRIMARY_COORDINATE_IDS:
        raise ConstructionError("contract primary coordinate IDs/order changed")
    full_map = contract["full_atlas_QC_map"]
    if int(full_map["fixed_coordinates"]) != 56:
        raise ConstructionError("full QC map must contain 56 coordinates")
    exposure = contract.get("pre_candidate_development_exposure", {})
    if (
        exposure.get("input_rows")
        != "exactly_200_primary_common_support_development_rows"
        or int(exposure.get("primary_model_rows_before_global_tree_QC", -1))
        != EXPECTED_PRIMARY_MODEL_FILES
        or int(exposure.get("unsupported_development_rows_not_downloaded_or_opened", -1))
        != 9
    ):
        raise ConstructionError("pre-candidate exposure boundary is not frozen")
    swc_contract = contract.get("SWC_interpretation", {})
    node_types = swc_contract.get("node_types", {})
    transitions = swc_contract.get("node_transition_rule", {})
    if (
        node_types.get("soma") != [1]
        or node_types.get("axon") != [0, 2]
        or node_types.get("excluded_dendrite") != [3]
        or node_types.get("ignored_markers") != [10, 11, 12]
        or transitions.get("axon_0_or_2_to_dendrite_3")
        != "global_unknown_and_exclusion"
        or transitions.get("dendrite_3_to_axon_0_or_2")
        != "global_unknown_and_exclusion"
        or transitions.get("marker_10_11_or_12_with_any_child")
        != "global_unknown_and_exclusion"
        or transitions.get("any_transition_with_child_soma_1")
        != "global_unknown_and_exclusion"
    ):
        raise ConstructionError("typed-edge transition contract is not frozen")
    branched = contract.get("branched_terminal_sensitivity", {})
    if (
        branched.get("sensitivity") != "same_coordinate_branched_terminal"
        or branched.get("sensitivity_only") is not True
        or branched.get("sensitivity_is_primary_or_may_replace_primary") is not False
        or branched.get("execution_stage") != "sealed_post_candidate_only"
        or branched.get("emitted_by_pre_candidate_constructor") is not False
    ):
        raise ConstructionError("branched-terminal sensitivity is not frozen")
    erosion_contract = contract.get("target_boundary_sensitivity", {})
    if (
        erosion_contract.get("sensitivity")
        != "one_voxel_internal_boundary_erosion"
        or erosion_contract.get("sensitivity_is_primary_or_may_replace_primary")
        is not False
        or float(erosion_contract.get("voxel_width_um", -1)) != 25.0
        or erosion_contract.get("execution_stage") != "sealed_post_candidate_only"
        or erosion_contract.get("emitted_by_pre_candidate_constructor") is not False
    ):
        raise ConstructionError("one-voxel erosion sensitivity is not frozen")

    atlas_contract = contract["atlas"]
    expected_shape = tuple(
        int(value) for value in atlas_contract["nrrd_header"]["sizes"]
    )
    expected_spacing = tuple(
        float(value) for value in atlas_contract["nrrd_header"]["spacing_um"]
    )
    annotation, header = nrrd.read(str(annotation_path), index_order="F")
    atlas = ArrayAtlas(annotation)
    if atlas.shape != expected_shape:
        raise ConstructionError(
            f"annotation shape {atlas.shape} != frozen {expected_shape}"
        )
    if tuple(int(value) for value in header["sizes"]) != expected_shape:
        raise ConstructionError("NRRD header sizes disagree with loaded array")
    if int(header["dimension"]) != int(atlas_contract["nrrd_header"]["dimension"]):
        raise ConstructionError("NRRD dimension differs from frozen contract")
    observed_type = str(header.get("type", "")).replace(" ", "_")
    if observed_type != str(atlas_contract["nrrd_header"]["type"]):
        raise ConstructionError("NRRD type differs from frozen contract")
    if str(header.get("space", "")) != str(atlas_contract["nrrd_header"]["space"]):
        raise ConstructionError("NRRD space differs from frozen contract")
    directions = header.get("space directions")
    if directions is None:
        raise ConstructionError("NRRD lacks space directions")
    for row in range(3):
        for column in range(3):
            expected = expected_spacing[row] if row == column else 0.0
            if not math.isclose(
                float(directions[row][column]), expected, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ConstructionError("NRRD spacing/orientation differs from contract")
    origin = header.get("space origin")
    expected_origin = tuple(
        float(value) for value in atlas_contract["nrrd_header"]["origin_um"]
    )
    if origin is None or any(
        not math.isclose(float(origin[index]), expected_origin[index], rel_tol=0.0, abs_tol=1.0e-12)
        for index in range(3)
    ):
        raise ConstructionError("NRRD origin differs from frozen contract")
    config = GeometryConfig(
        spacing_um=expected_spacing,  # type: ignore[arg-type]
        shape=expected_shape,  # type: ignore[arg-type]
        ml_midline_um=float(
            atlas_contract["provider_coordinate_binding"]["ML_midline_um"]
        ),
        grey_structure_id=int(atlas_contract["grey_structure_id"]),
        zero_tolerance_um=float(
            contract["segment_allocation"]["numerical_zero_tolerance_um"]
        ),
    )
    ontology = load_ontology(ontology_path)
    if ontology.unique_id_for_acronym("MOp") != 985:
        raise ConstructionError("pinned ontology no longer resolves MOp to 985")
    coordinates = load_vocabulary(vocabulary_path)
    by_id = {coordinate.coordinate_id: coordinate for coordinate in coordinates}
    try:
        selected_primary = tuple(by_id[value] for value in panel_ids)
    except KeyError as error:
        raise ConstructionError(f"primary coordinate absent from full map: {error}") from error
    if coordinates[2].coordinate_id != "allen_993_MOs__ipsilateral":
        raise ConstructionError("MOs replacement is absent from full vocabulary")
    if any(coordinate.anchor_structure_id in {500, 985} for coordinate in coordinates):
        raise ConstructionError("generic MO or source MOp entered the named vocabulary")
    primary_coordinates = tuple(
        Coordinate(
            index=index,
            coordinate_id=coordinate.coordinate_id,
            target_index=coordinate.target_index,
            target_id=coordinate.target_id,
            anchor_structure_id=coordinate.anchor_structure_id,
            laterality=coordinate.laterality,
        )
        for index, coordinate in enumerate(selected_primary)
    )
    return contract, atlas, config, ontology, coordinates, primary_coordinates


def _identity_row(result: CellResult) -> dict[str, object]:
    return {
        "provider_neuron_id": result.entry.provider_neuron_id,
        "file": result.entry.filename,
        "fMOST_brain_id": result.entry.brain_id,
        "provider_sample_id": result.entry.sample_id,
    }


def _atomic_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    partial = path.with_suffix(path.suffix + ".part")
    if path.is_symlink() or partial.is_symlink():
        raise ConstructionError(f"refusing symlink output below {path.parent}")
    try:
        with partial.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".part")
    if path.is_symlink() or partial.is_symlink():
        raise ConstructionError(f"refusing symlink output below {path.parent}")
    try:
        with partial.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def write_outputs(
    output_root: Path,
    results: Sequence[CellResult],
    coordinates: Sequence[Coordinate],
    primary_indexes: Sequence[int],
) -> None:
    """Write only the frozen pre-candidate exposure surface."""

    if len(results) != EXPECTED_FILES:
        raise ConstructionError(
            f"pre-candidate output requires {EXPECTED_FILES} manifest rows"
        )
    if tuple(primary_indexes) != tuple(range(16)) or len(coordinates) != 16:
        raise ConstructionError("pre-candidate output requires exactly 16 coordinates")
    if any(not result.entry.primary_model_eligible for result in results):
        raise ConstructionError("unsupported development row reached constructor output")

    identity_fields = [
        "provider_neuron_id",
        "file",
        "fMOST_brain_id",
        "provider_sample_id",
    ]
    primary_fields = identity_fields + [
        "primary_common_support",
        "primary_model_eligible",
        "global_eligible",
        "global_eligibility_failure",
        "K",
        "S_coordinate_ids",
    ]
    zero_label = "not_observed_in_the_eligible_released_reconstruction"
    for prefix in (
        "status",
        "detected",
        "passing_only",
        "no_occupancy",
        "binary_zero_semantics",
    ):
        primary_fields.extend(
            f"{prefix}__{coordinate.coordinate_id}" for coordinate in coordinates
        )

    primary_rows: list[dict[str, object]] = []
    for result in results:
        callable_result = result.qc.eligible and result.coordinate_qc is not None
        if result.qc.eligible != (result.coordinate_qc is not None):
            raise ConstructionError(
                "eligible primary-model row lacks its primary-16 observation"
            )
        detected_ids = (
            [
                coordinate.coordinate_id
                for coordinate in coordinates
                if result.coordinate_qc is not None
                and result.coordinate_qc[coordinate.index].detected
            ]
            if callable_result
            else []
        )
        row: dict[str, object] = {
            **_identity_row(result),
            "primary_common_support": "yes",
            "primary_model_eligible": "yes",
            "global_eligible": "yes" if result.qc.eligible else "no",
            "global_eligibility_failure": result.qc.failure_code,
            "K": result.primary_k if callable_result else "",
            "S_coordinate_ids": ";".join(detected_ids) if callable_result else "",
        }
        for coordinate in coordinates:
            status: object = ""
            detected: object = ""
            passing_only: object = ""
            no_occupancy: object = ""
            zero_semantics: object = ""
            if result.coordinate_qc is not None:
                stat = result.coordinate_qc[coordinate.index]
                detected = int(stat.detected)
                passing_only = int(stat.passing_only)
                no_occupancy = int(not stat.occupied)
                if stat.detected:
                    status = "detected"
                elif stat.passing_only:
                    status = "passing_only"
                else:
                    status = "no_occupancy"
                if not stat.detected:
                    zero_semantics = zero_label
            row[f"status__{coordinate.coordinate_id}"] = status
            row[f"detected__{coordinate.coordinate_id}"] = detected
            row[f"passing_only__{coordinate.coordinate_id}"] = passing_only
            row[f"no_occupancy__{coordinate.coordinate_id}"] = no_occupancy
            row[f"binary_zero_semantics__{coordinate.coordinate_id}"] = zero_semantics
        primary_rows.append(row)
    _atomic_csv(
        output_root / "primary_16_target_calls.csv", primary_fields, primary_rows
    )

    eligibility_fields = identity_fields + [
        "primary_common_support",
        "primary_model_eligible",
        "global_eligible",
        "global_eligibility_failure",
        "provider_hemisphere",
        "soma_hemisphere",
        "canonical_soma_agreement",
        "soma_maps_to_MOp",
    ]
    eligibility_rows = [
        {
            **_identity_row(result),
            "primary_common_support": "yes",
            "primary_model_eligible": "yes",
            "global_eligible": "yes" if result.qc.eligible else "no",
            "global_eligibility_failure": result.qc.failure_code,
            "provider_hemisphere": result.qc.provider_hemisphere,
            "soma_hemisphere": result.qc.soma_hemisphere,
            "canonical_soma_agreement": int(result.qc.canonical_soma_agreement),
            "soma_maps_to_MOp": int(result.qc.soma_maps_to_MOp),
        }
        for result in results
    ]
    _atomic_csv(
        output_root / "cell_eligibility_qc.csv",
        eligibility_fields,
        eligibility_rows,
    )

    eligible_results = [result for result in results if result.qc.eligible]
    status_counts: dict[str, dict[str, int]] = {}
    for coordinate in coordinates:
        stats = [
            result.coordinate_qc[coordinate.index]
            for result in eligible_results
            if result.coordinate_qc is not None
        ]
        status_counts[coordinate.coordinate_id] = {
            "detected": sum(stat.detected for stat in stats),
            "passing_only": sum(stat.passing_only for stat in stats),
            "no_occupancy": sum(not stat.occupied for stat in stats),
        }
    summary = {
        "status": "complete",
        "exposure_stage": "pre_candidate_development",
        "role": "development",
        "input_files": len(results),
        "input_groups": len({result.entry.brain_id for result in results}),
        "frozen_primary_common_support_files": EXPECTED_PRIMARY_MODEL_FILES,
        "unsupported_development_files_opened": 0,
        "globally_eligible_primary_model_cells": len(eligible_results),
        "globally_unknown_excluded_cells": len(results) - len(eligible_results),
        "global_eligibility_failure_counts": dict(
            sorted(
                Counter(
                    result.qc.failure_code
                    for result in results
                    if not result.qc.eligible
                ).items()
            )
        ),
        "primary_coordinate_count": len(coordinates),
        "primary_status_counts": status_counts,
        "binary_zero_semantics": zero_label,
        "K_distribution": dict(
            sorted(Counter(result.primary_k for result in eligible_results).items())
        ),
        "all_primary_coordinates_unknown_for_globally_ineligible_cells": True,
        "per_cell_omitted_40_outcomes_emitted": False,
        "raw_lengths_emitted": False,
        "branch_or_component_counts_emitted": False,
        "erosion_or_branched_sensitivities_emitted": False,
        "final_or_unassigned_files_opened": False,
        "EP11_shared_units_opened": False,
    }
    _atomic_json(output_root / "aggregate_summary.json", summary)


def _fixture_ontology() -> Ontology:
    return Ontology(
        {
            997: None,
            8: 997,
            315: 8,
            500: 315,
            985: 500,
            993: 500,
            453: 315,
            1009: 997,
            73: 997,
            1024: 997,
            304325711: 997,
        },
        {
            997: "root",
            8: "grey",
            315: "Isocortex",
            500: "MO",
            985: "MOp",
            993: "MOs",
            453: "SS",
            1009: "fiber tracts",
            73: "VS",
            1024: "grooves",
            304325711: "retina",
        },
    )


def _fixture_coordinates() -> tuple[Coordinate, ...]:
    return (
        Coordinate(0, "MOs__ipsilateral", 0, "MOs", 993, "ipsilateral"),
        Coordinate(1, "MOs__contralateral", 0, "MOs", 993, "contralateral"),
        Coordinate(2, "SS__ipsilateral", 1, "SS", 453, "ipsilateral"),
        Coordinate(3, "SS__contralateral", 1, "SS", 453, "contralateral"),
    )


def _fixture_entry(
    soma: tuple[float, float, float] = (5.0, 5.0, 5.0),
    hemisphere: str = "Left",
    primary_common_support: bool = True,
) -> ManifestEntry:
    return ManifestEntry(
        provider_neuron_id="1",
        filename="100001_001.swc",
        brain_id="100001",
        sample_id="fixture",
        source_acronym="MOp5",
        provider_hemisphere=hemisphere,
        canonical_soma_um=soma,
        primary_common_support=primary_common_support,
        primary_model_eligible=primary_common_support,
    )


def _construct_fixture(
    swc: str,
    atlas: Atlas,
    config: GeometryConfig,
    entry: ManifestEntry | None = None,
    run_sealed_sensitivities: bool = False,
) -> CellResult:
    ontology = _fixture_ontology()
    coordinates = _fixture_coordinates()
    mapper = StructureMapper(ontology, {993: 0, 453: 1}, 8)
    nodes, extra = parse_swc_lines(io.StringIO(swc))
    return construct_cell(
        entry or _fixture_entry(),
        nodes,
        extra,
        atlas,
        config,
        ontology,
        mapper,
        coordinates,
        (0, 1, 2, 3),
        run_sealed_sensitivities=run_sealed_sensitivities,
    )


def run_self_test() -> dict[str, object]:
    config = GeometryConfig((10.0, 10.0, 10.0), (8, 3, 5), 20.0)
    values: dict[tuple[int, int, int], int] = {}
    for x in range(8):
        for y in range(3):
            for z in range(5):
                if x == 0:
                    values[(x, y, z)] = 985
                elif x in {1, 2, 3}:
                    values[(x, y, z)] = 993
                elif x in {4, 5, 6}:
                    values[(x, y, z)] = 453
                else:
                    values[(x, y, z)] = 985
    atlas = SyntheticAtlas(config.shape, values)

    try:
        parse_swc_lines(
            io.StringIO(
                """1 1 5 5 5 1 -1
2 2 15 5 5 1 3
3 2 25 5 5 1 2
"""
            )
        )
    except ConstructionError as error:
        if "cycle" not in str(error):
            raise AssertionError(f"unexpected strict-tree failure: {error}")
    else:
        raise AssertionError("a disconnected SWC cycle passed strict parsing")

    invalid_transitions = {
        "nonroot_soma": """1 1 5 5 5 1 -1
2 1 15 5 5 1 1
""",
        "axon_to_dendrite": """1 1 5 5 5 1 -1
2 2 15 5 5 1 1
3 3 25 5 5 1 2
""",
        "dendrite_to_axon": """1 1 5 5 5 1 -1
2 3 15 5 5 1 1
3 2 25 5 5 1 2
""",
        "marker_parents_neurite": """1 1 5 5 5 1 -1
2 10 15 5 5 1 1
3 2 25 5 5 1 2
""",
    }
    for label, malformed in invalid_transitions.items():
        try:
            parse_swc_lines(io.StringIO(malformed))
        except ConstructionError:
            pass
        else:
            raise AssertionError(f"invalid typed edge passed parsing: {label}")

    try:
        _construct_fixture(
            """1 1 5 5 5 1 -1
2 2 25 5 5 1 1
""",
            atlas,
            config,
            _fixture_entry((5.0, 5.0, 5.5)),
        )
    except CellIneligible as error:
        if error.code != "canonical_soma_mismatch":
            raise AssertionError(f"unexpected soma-agreement failure: {error.code}")
    else:
        raise AssertionError("an SWC/canonical soma mismatch remained eligible")

    terminal_passing = _construct_fixture(
        """1 1 5 5 5 1 -1
2 2 25 5 5 1 1
3 2 75 5 5 1 2
4 2 55 5 5 1 1
""",
        atlas,
        config,
    )
    assert terminal_passing.coordinate_qc is not None
    if not terminal_passing.coordinate_qc[0].passing_only:
        raise AssertionError("MOs traversal was not marked passing-only")
    if not terminal_passing.coordinate_qc[2].detected:
        raise AssertionError("SS terminal leaf was not detected")

    marker_terminated_axon = _construct_fixture(
        """1 1 5 5 5 1 -1
2 2 25 5 5 1 1
3 10 25 5 5 1 2
""",
        atlas,
        config,
    )
    assert marker_terminated_axon.coordinate_qc is not None
    if marker_terminated_axon.coordinate_qc[0].detected:
        raise AssertionError("an axon node with a marker child became a terminal")
    if not marker_terminated_axon.coordinate_qc[0].passing_only:
        raise AssertionError("marker-terminated axon occupancy was not passing-only")

    mop_mos = _construct_fixture(
        """1 1 5 5 5 1 -1
2 2 8 5 5 1 1
3 2 25 5 5 1 1
""",
        atlas,
        config,
    )
    assert mop_mos.coordinate_qc is not None
    if not mop_mos.coordinate_qc[0].detected:
        raise AssertionError("MOs terminal was not detected")
    if mop_mos.qc.category_lengths_um["unclaimed_grey"] <= 0.0:
        raise AssertionError("source MOp was not retained as unclaimed-grey QC")

    midline = _construct_fixture(
        """1 1 5 5 5 1 -1
2 2 25 5 35 1 1
""",
        atlas,
        config,
    )
    assert midline.coordinate_qc is not None
    if not midline.coordinate_qc[1].detected:
        raise AssertionError("contralateral MOs terminal was not detected")
    if midline.coordinate_qc[0].length_um <= 0.0:
        raise AssertionError("midline split lost ipsilateral MOs length")

    exact_midline_soma = _construct_fixture(
        """1 1 5 5 20 1 -1
2 2 25 5 20 1 1
""",
        atlas,
        config,
        _fixture_entry((5.0, 5.0, 20.0), "Right"),
    )
    assert exact_midline_soma.coordinate_qc is not None
    if not exact_midline_soma.coordinate_qc[0].detected:
        raise AssertionError("an exact-midline soma did not follow the frozen right rule")

    unknown_values = dict(values)
    unknown_values[(2, 1, 0)] = 1009
    unknown_atlas = SyntheticAtlas(config.shape, unknown_values)
    try:
        _construct_fixture(
            """1 1 5 5 5 1 -1
2 2 25 15 5 1 1
""",
            unknown_atlas,
            config,
        )
    except CellIneligible as error:
        if error.code != "terminal_leaf_nongrey":
            raise AssertionError(f"unexpected global-unknown code: {error.code}")
    else:
        raise AssertionError("fiber-tract terminal did not make the cell unknown")

    branch = _construct_fixture(
        """1 1 5 5 5 1 -1
2 2 15 15 15 1 1
3 2 25 15 15 1 2
4 2 25 25 15 1 2
""",
        atlas,
        config,
        run_sealed_sensitivities=True,
    )
    assert branch.coordinate_qc is not None
    if not branch.coordinate_qc[0].branched_terminal_detected:
        raise AssertionError("same-coordinate two-leaf branch sensitivity failed")

    mixed_branch = _construct_fixture(
        """1 1 5 5 5 1 -1
2 2 15 15 15 1 1
3 2 25 15 15 1 2
4 2 45 25 15 1 2
""",
        atlas,
        config,
        run_sealed_sensitivities=True,
    )
    assert mixed_branch.coordinate_qc is not None
    if any(stat.branched_terminal_detected for stat in mixed_branch.coordinate_qc):
        raise AssertionError("mixed-coordinate child paths passed branched sensitivity")

    erosion_values = {
        (x, y, z): 993
        for x in range(5)
        for y in range(5)
        for z in range(5)
    }
    erosion_values[(0, 2, 0)] = 985
    erosion_config = GeometryConfig((10.0, 10.0, 10.0), (5, 5, 5), 25.0)
    erosion_atlas = SyntheticAtlas(erosion_config.shape, erosion_values)
    boundary = _construct_fixture(
        """1 1 5 25 5 1 -1
2 2 15 25 5 1 1
""",
        erosion_atlas,
        erosion_config,
        _fixture_entry((5.0, 25.0, 5.0)),
        run_sealed_sensitivities=True,
    )
    assert boundary.coordinate_qc is not None
    if not boundary.coordinate_qc[0].detected or boundary.coordinate_qc[0].eroded_detected:
        raise AssertionError("one-voxel boundary erosion sensitivity failed")

    valid_manifest = {
        "provider_neuron_id": "1",
        "file": "100001_001.swc",
        "fMOST_brain_id": "100001",
        "provider_sample_id": "fixture",
        "source_acronym": "MOp5",
        "hemisphere": "Left",
        "reconstruction_type": "Axon_and_dendrite",
        "soma_x_ccfv3_um": "5",
        "soma_y_ccfv3_um": "5",
        "soma_z_ccfv3_um": "5",
        "episode10_role": "development",
        "EP11_SSp_tr_overlap": "no",
        "role_status": "frozen_before_projection_outcomes",
        "projection_outcome_exposed": "no",
        "primary_common_support": "yes",
        "primary_model_eligible": "yes",
        "geometry_condition": FROZEN_GEOMETRY_CONDITION,
    }
    validate_manifest_rows(
        [valid_manifest],
        expected_files=1,
        expected_groups=1,
        expected_primary_model_files=1,
    )
    for field_name, value in (
        ("episode10_role", "final"),
        ("episode10_role", "unassigned"),
        ("EP11_SSp_tr_overlap", "yes"),
        ("role_status", "not_frozen"),
        ("projection_outcome_exposed", "yes"),
        ("reconstruction_type", "Axon_only"),
        ("primary_common_support", "no"),
        ("primary_model_eligible", "no"),
        ("geometry_condition", "changed_geometry_condition"),
    ):
        invalid = dict(valid_manifest)
        invalid[field_name] = value
        try:
            validate_manifest_rows(
                [invalid],
                expected_files=1,
                expected_groups=1,
                expected_primary_model_files=1,
            )
        except ConstructionError:
            pass
        else:
            raise AssertionError(f"role boundary accepted {field_name}={value}")

    return {
        "status": "pass",
        "uses_real_atlas_or_swc": False,
        "checks": [
            "strict_tree_and_root_parsing",
            "complete_typed_edge_transition_rejection",
            "axon_with_any_child_is_not_terminal",
            "canonical_soma_agreement",
            "terminal_vs_passing_only",
            "MOp_unclaimed_grey_and_MOs_detection",
            "exact_voxel_and_midline_clipping",
            "soma_relative_laterality",
            "global_unknown_for_nongrey_terminal",
            "sealed_post_candidate_one_voxel_erosion_synthetic_only",
            "sealed_post_candidate_branched_terminal_synthetic_only",
            "development_role_freeze_exposure_and_EP11_overlap_rejection",
            "length_conservation",
        ],
    }


def run_real(args: argparse.Namespace) -> None:
    (
        _,
        atlas,
        config,
        ontology,
        full_coordinates,
        coordinates,
    ) = _load_real_assets(
        args.contract,
        args.annotation,
        args.ontology,
        args.vocabulary,
    )
    anchor_to_target: dict[int, int] = {}
    for coordinate in full_coordinates:
        prior = anchor_to_target.setdefault(
            coordinate.anchor_structure_id, coordinate.target_index
        )
        if prior != coordinate.target_index:
            raise ConstructionError("one anchor maps to multiple target indexes")
    mapper = StructureMapper(
        ontology,
        anchor_to_target,
        grey_structure_id=config.grey_structure_id,
    )

    entries = load_manifest(args.manifest)
    swc_root = _validate_scratch_path(args.swc_root, "development SWC root")
    output_root = _validate_scratch_path(args.output_root, "output root")
    if (
        swc_root == output_root
        or swc_root in output_root.parents
        or output_root in swc_root.parents
    ):
        raise ConstructionError("SWC input and constructor output roots must be disjoint")
    swc_paths = validate_exact_swc_set(entries, swc_root)
    if output_root.exists() and not output_root.is_dir():
        raise ConstructionError(f"output root is not a directory: {output_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise ConstructionError(
            "pre-candidate output root must be absent or empty; refusing stale outputs"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink():
        raise ConstructionError(f"refusing symlink output root: {output_root}")

    results: list[CellResult] = []
    for entry in entries:
        try:
            nodes, extra_columns = parse_swc(swc_paths[entry.provider_neuron_id])
            result = construct_cell(
                entry,
                nodes,
                extra_columns,
                atlas,
                config,
                ontology,
                mapper,
                coordinates,
                tuple(range(16)),
                run_sealed_sensitivities=False,
            )
        except CellIneligible as error:
            qc = CellQC(
                provider_neuron_id=entry.provider_neuron_id,
                filename=entry.filename,
                brain_id=entry.brain_id,
                sample_id=entry.sample_id,
                eligible=False,
                failure_code=error.code,
                failure_detail=str(error),
                provider_hemisphere=_normalize_hemisphere(
                    entry.provider_hemisphere
                ),
            )
            result = CellResult(entry, qc, None, None)
        except (ConstructionError, OSError, UnicodeError) as error:
            qc = CellQC(
                provider_neuron_id=entry.provider_neuron_id,
                filename=entry.filename,
                brain_id=entry.brain_id,
                sample_id=entry.sample_id,
                eligible=False,
                failure_code="tree_or_construction_failure",
                failure_detail=str(error),
                provider_hemisphere=_normalize_hemisphere(
                    entry.provider_hemisphere
                ),
            )
            result = CellResult(entry, qc, None, None)
        results.append(result)
    if len(results) != EXPECTED_FILES:
        raise ConstructionError("constructor did not emit one result per manifest row")
    write_outputs(output_root, results, coordinates, tuple(range(16)))
    print(
        json.dumps(
            {
                "status": "complete",
                "development_files": len(results),
                "development_groups": len({entry.brain_id for entry in entries}),
                "eligible_cells": sum(result.qc.eligible for result in results),
                "globally_unknown_excluded_cells": sum(
                    not result.qc.eligible for result in results
                ),
                "output_root": str(output_root),
                "final_or_unassigned_files_opened": False,
                "EP11_shared_units_opened": False,
            },
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run dependency-free synthetic tests and make no real-data access",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="explicitly permit reading the frozen development SWCs",
    )
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--vocabulary", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--ontology", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--swc-root", type=Path)
    parser.add_argument("--output-root", type=Path)
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
        parser.error(
            "real construction is gated; use --execute only after the frozen "
            "development SWCs have been acquired"
        )
    required = (
        "contract",
        "vocabulary",
        "annotation",
        "ontology",
        "manifest",
        "swc_root",
        "output_root",
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
