#!/usr/bin/env python3
"""Test outcome-blind spatial support in the Gao whole-cortex SSp-tr cohort.

The input is the compact inventory produced by ``download_whole_cortex_ssptr.py``.
Only cell identity, brain/sample identity, soma coordinates, provider source
label, hemisphere, and strain/Cre line are used.  Axon geometry and published
projection classifications are neither read into the analysis records nor used
by the support test.

The geometry matches the earlier EP11 SS screen: homologous hemispheres are
folded in Allen CCFv3, tangential distance is a shortest-path distance on the
pial-endpoint voxel surface, and depth is the closest-point distance along an
official 10-um cortical streamline divided by local cortical thickness.

Because 36 brains make exhaustive role enumeration impractical, existence of a
12-development/8-audit allocation is tested with ``scipy.optimize.milp``.  The
script emits aggregate feasibility only; it never writes the solver's role
assignment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from ccf_streamlines.linestring3d import LineString3D
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


REPO_EPISODE = Path(
    "/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/"
    "episode11_projection_types_vs_gradients"
)
SCRATCH_EPISODE = Path(
    "/scratch/users/zijiao/br_autoresearch/"
    "episode11_projection_types_vs_gradients"
)
ASSET_DIR = SCRATCH_EPISODE / "allen_cortical_geometry"
SURFACE_PATHS = ASSET_DIR / "surface_paths_10_v3.h5"
CLOSEST_SURFACE = ASSET_DIR / "closest_surface_voxel_lookup.h5"

DEFAULT_INVENTORY = (
    REPO_EPISODE / "outputs/stage0/whole_cortex_ssptr_inventory.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_EPISODE / "outputs/stage0/whole_cortex_ssptr_geometry_support.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_EPISODE / "outputs/stage0/whole_cortex_ssptr_geometry_support.md"
)

SOURCE_LABEL_TO_LAYER = {
    "SSp-tr1": "L1",
    "SSp-tr2/3": "L2/3",
    "SSp-tr4": "L4",
    "SSp-tr5": "L5",
    "SSp-tr6a": "L6a",
    "SSp-tr6b": "L6b",
}
EXPECTED_CELLS = 214
EXPECTED_BRAINS = 36
EXPECTED_SOURCE = "SSp-tr"
SINGLETON_LINE = "Tle4-2A-CreER"
DEVELOPMENT_BRAINS = 12
AUDIT_BRAINS = 8
RADII_UM = (250, 500, 750, 1000, 1250, 1500)
DEPTH_TOLERANCES = (0.10, 0.20, 0.30, math.inf)
PER_ANIMAL_RETENTION = 0.90
OVERALL_ROLE_RETENTION = 0.95
MINIMUM_MULTI_CELL_DEVELOPMENT_BRAINS = 8
MINIMUM_DISTINCT_DEVELOPMENT_NEIGHBORS = 2

REQUIRED_INPUT_COLUMNS = {
    "provider_neuron_id",
    "fMOST_brain_id",
    "source_acronym",
    "strain_or_cre_line",
    "hemisphere",
    "soma_x_ccfv3_um",
    "soma_y_ccfv3_um",
    "soma_z_ccfv3_um",
}


class GeometrySupportError(RuntimeError):
    """Raised when an input or a geometry/solver invariant fails."""


@dataclass(frozen=True)
class CellRecord:
    """The outcome-blind fields used by the support calculation."""

    cell_id: str
    brain_id: str
    sample_id: str | None
    provider_source_label: str
    layer: str
    line: str
    hemisphere: str
    coordinate: tuple[float, float, float]
    primary_axon_dendrite_distinguished_eligible: bool


def finite_float(value: str, field: str, row_number: int) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise GeometrySupportError(
            f"row {row_number}: {field} is not numeric"
        ) from error
    if not math.isfinite(result):
        raise GeometrySupportError(f"row {row_number}: {field} is not finite")
    return result


def load_inventory(path: Path) -> tuple[list[CellRecord], dict[str, object]]:
    """Read only the allow-listed, outcome-blind inventory fields."""

    records: list[CellRecord] = []
    seen_cells: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_INPUT_COLUMNS - headers)
        if missing:
            raise GeometrySupportError(
                f"inventory is missing required columns: {missing}"
            )
        if "primary_axon_dendrite_distinguished_eligible" not in headers:
            raise GeometrySupportError(
                "inventory lacks primary_axon_dendrite_distinguished_eligible"
            )
        sample_id_column = (
            "provider_sample_id"
            if "provider_sample_id" in headers
            else ("sample_id" if "sample_id" in headers else None)
        )
        for row_number, raw in enumerate(reader, start=2):
            # Deliberately resolve only fields needed for soma support.  In
            # particular, downloader QC node counts are not copied here.
            cell_id = raw["provider_neuron_id"].strip()
            brain_id = raw["fMOST_brain_id"].strip()
            label = raw["source_acronym"].strip()
            line = raw["strain_or_cre_line"].strip()
            hemisphere = raw["hemisphere"].strip()
            primary_flag = raw[
                "primary_axon_dendrite_distinguished_eligible"
            ].strip()
            sample_id = (
                raw[sample_id_column].strip()
                if sample_id_column is not None
                else None
            )
            if not cell_id or cell_id in seen_cells:
                raise GeometrySupportError(
                    f"row {row_number}: missing or duplicate provider neuron ID"
                )
            seen_cells.add(cell_id)
            if not brain_id:
                raise GeometrySupportError(f"row {row_number}: missing brain ID")
            if label not in SOURCE_LABEL_TO_LAYER:
                raise GeometrySupportError(
                    f"row {row_number}: unexpected source label {label!r}"
                )
            if not line:
                raise GeometrySupportError(
                    f"row {row_number}: missing strain/Cre line"
                )
            if hemisphere not in {"Left", "Right"}:
                raise GeometrySupportError(
                    f"row {row_number}: unexpected hemisphere {hemisphere!r}"
                )
            if primary_flag not in {"yes", "no_axon_dendrite_not_distinguished"}:
                raise GeometrySupportError(
                    f"row {row_number}: unexpected primary eligibility flag"
                )
            coordinate = tuple(
                finite_float(raw[field], field, row_number)
                for field in (
                    "soma_x_ccfv3_um",
                    "soma_y_ccfv3_um",
                    "soma_z_ccfv3_um",
                )
            )
            records.append(
                CellRecord(
                    cell_id=cell_id,
                    brain_id=brain_id,
                    sample_id=sample_id or None,
                    provider_source_label=label,
                    layer=SOURCE_LABEL_TO_LAYER[label],
                    line=line,
                    hemisphere=hemisphere,
                    coordinate=coordinate,
                    primary_axon_dendrite_distinguished_eligible=(
                        primary_flag == "yes"
                    ),
                )
            )

    brains = sorted({record.brain_id for record in records})
    if len(records) != EXPECTED_CELLS or len(brains) != EXPECTED_BRAINS:
        raise GeometrySupportError(
            "inventory does not reconcile to 214 SSp-tr cells in 36 brains"
        )
    primary_records = [
        record
        for record in records
        if record.primary_axon_dendrite_distinguished_eligible
    ]
    if len(primary_records) != 206 or len(
        {record.brain_id for record in primary_records}
    ) != EXPECTED_BRAINS:
        raise GeometrySupportError(
            "primary eligibility does not reconcile to 206 cells in 36 brains"
        )

    brain_lines: dict[str, set[str]] = defaultdict(set)
    brain_samples: dict[str, set[str]] = defaultdict(set)
    sample_brains: dict[str, set[str]] = defaultdict(set)
    for record in records:
        brain_lines[record.brain_id].add(record.line)
        if record.sample_id is not None:
            brain_samples[record.brain_id].add(record.sample_id)
            sample_brains[record.sample_id].add(record.brain_id)
    if any(len(lines) != 1 for lines in brain_lines.values()):
        raise GeometrySupportError("a brain is associated with multiple strain/Cre lines")

    sample_crosswalk_status = "not_present_in_inventory"
    if any(record.sample_id is not None for record in records):
        if any(record.sample_id is None for record in records):
            raise GeometrySupportError("sample ID is only partially populated")
        if any(len(values) != 1 for values in brain_samples.values()):
            raise GeometrySupportError("a brain is associated with multiple sample IDs")
        if any(len(values) != 1 for values in sample_brains.values()):
            raise GeometrySupportError("a sample ID is associated with multiple brains")
        if len(sample_brains) != EXPECTED_BRAINS:
            raise GeometrySupportError("sample IDs do not form a 36-brain one-to-one map")
        sample_crosswalk_status = "one_to_one_36_brains"

    line_brains: dict[str, set[str]] = defaultdict(set)
    for brain_id, lines in brain_lines.items():
        line_brains[next(iter(lines))].add(brain_id)
    singleton_lines = sorted(
        line for line, line_members in line_brains.items() if len(line_members) == 1
    )
    if singleton_lines != [SINGLETON_LINE]:
        raise GeometrySupportError(
            "the expected sole singleton line Tle4-2A-CreER was not recovered"
        )

    diagnostics = {
        "downloaded_cells": len(records),
        "primary_axon_dendrite_distinguished_eligible_cells": len(primary_records),
        "axon_only_sensitivity_cells": len(records) - len(primary_records),
        "sample_crosswalk_status": sample_crosswalk_status,
        "layer_cell_counts": dict(
            sorted(Counter(record.layer for record in records).items())
        ),
        "line_cell_counts": dict(
            sorted(Counter(record.line for record in records).items())
        ),
        "line_brain_counts": {
            line: len(line_brains[line]) for line in sorted(line_brains)
        },
    }
    return records, diagnostics


def query_h5_sorted_pair_lookup(
    queries: np.ndarray, dataset: h5py.Dataset
) -> np.ndarray:
    """Query a sorted two-column HDF5 lookup without loading it into RAM.

    The official closest-surface table has more than 61 million rows and is
    about 472 MiB when decompressed.  Reading the whole array creates an
    unnecessary memory spike for only 214 soma lookups.  We first locate the
    relevant HDF5 chunks from their final keys, then read only those chunks.
    """

    if dataset.ndim != 2 or dataset.shape[1] != 2:
        raise GeometrySupportError("closest-surface lookup must have two columns")
    row_count = int(dataset.shape[0])
    chunk_rows = int(dataset.chunks[0]) if dataset.chunks else 1_000_000
    chunk_starts = np.arange(0, row_count, chunk_rows, dtype=np.int64)
    chunk_ends = np.minimum(chunk_starts + chunk_rows, row_count) - 1
    last_keys = np.asarray(dataset[chunk_ends, 0], dtype=np.uint32)
    if last_keys.size > 1 and np.any(last_keys[1:] < last_keys[:-1]):
        raise GeometrySupportError("closest-surface chunk boundaries are not sorted")

    query_chunks = np.searchsorted(last_keys, queries, side="left")
    result = np.zeros(queries.shape[0], dtype=np.uint32)
    for chunk_index in np.unique(query_chunks[query_chunks < chunk_starts.size]):
        start = int(chunk_starts[chunk_index])
        stop = min(start + chunk_rows, row_count)
        block = np.asarray(dataset[start:stop, :], dtype=np.uint32)
        keys = block[:, 0]
        if keys.size > 1 and np.any(keys[1:] < keys[:-1]):
            raise GeometrySupportError("closest-surface lookup is not sorted")
        query_positions = np.flatnonzero(query_chunks == chunk_index)
        within = np.searchsorted(keys, queries[query_positions], side="left")
        in_range = within < keys.size
        safe = np.where(in_range, within, 0)
        matched = in_range & (keys[safe] == queries[query_positions])
        result[query_positions[matched]] = block[safe[matched], 1]
    return result


def h5_sorted_take(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(indices, return_inverse=True)
    values = dataset[unique]
    return values[inverse]


def project_to_polyline(
    point_um: np.ndarray,
    flat_path: np.ndarray,
    volume_shape: tuple[int, int, int],
) -> tuple[float, float, float, float]:
    """Return exact/API normalized depths, physical depth, and offset."""

    flat_path = flat_path[flat_path > 0]
    if flat_path.size < 2:
        return math.nan, math.nan, math.nan, math.nan
    voxels = np.column_stack(np.unravel_index(flat_path, volume_shape)).astype(float)
    centers_um = (voxels + 0.5) * 10.0
    starts = centers_um[:-1]
    segments = centers_um[1:] - starts
    lengths = np.linalg.norm(segments, axis=1)
    valid = lengths > 0
    if not np.any(valid):
        return math.nan, math.nan, math.nan, math.nan
    api_fraction = float(
        LineString3D(centers_um).project(point_um, normalized=True)
    )
    starts = starts[valid]
    segments = segments[valid]
    lengths = lengths[valid]
    fractions = np.sum((point_um - starts) * segments, axis=1) / (lengths**2)
    fractions = np.clip(fractions, 0.0, 1.0)
    projected = starts + fractions[:, None] * segments
    distances = np.linalg.norm(point_um - projected, axis=1)
    segment = int(np.argmin(distances))
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    depth_um = cumulative[segment] + fractions[segment] * lengths[segment]
    thickness_um = cumulative[-1]
    return (
        depth_um / thickness_um,
        api_fraction,
        depth_um,
        float(distances[segment]),
    )


def build_surface_graph(
    endpoints: np.ndarray,
    cell_endpoints: np.ndarray,
    volume_shape: tuple[int, int, int],
) -> tuple[object, np.ndarray, dict[str, object]]:
    unique_endpoints = np.unique(endpoints[endpoints > 0])
    coordinates = np.column_stack(np.unravel_index(unique_endpoints, volume_shape))
    cell_coordinates = np.column_stack(np.unravel_index(cell_endpoints, volume_shape))

    margin_voxels = int(max(RADII_UM) / 10)
    lower = np.maximum(cell_coordinates.min(axis=0) - margin_voxels, 0)
    upper = np.minimum(
        cell_coordinates.max(axis=0) + margin_voxels,
        np.asarray(volume_shape) - 1,
    )
    inside = np.all((coordinates >= lower) & (coordinates <= upper), axis=1)
    nodes = unique_endpoints[inside]
    node_coordinates = coordinates[inside]
    sorter = np.argsort(nodes)
    nodes = nodes[sorter]
    node_coordinates = node_coordinates[sorter]

    cell_positions = np.searchsorted(nodes, cell_endpoints)
    if np.any(cell_positions >= nodes.size) or np.any(
        nodes[cell_positions] != cell_endpoints
    ):
        raise GeometrySupportError("a cell pial endpoint is absent from the graph")

    positive_offsets: list[tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) == (0, 0, 0):
                    continue
                if (
                    dx > 0
                    or (dx == 0 and dy > 0)
                    or (dx == 0 and dy == 0 and dz > 0)
                ):
                    positive_offsets.append((dx, dy, dz))

    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    shape_array = np.asarray(volume_shape)
    for offset in positive_offsets:
        offset_array = np.asarray(offset)
        neighbor_coordinates = node_coordinates + offset_array
        in_bounds = np.all(
            (neighbor_coordinates >= 0) & (neighbor_coordinates < shape_array),
            axis=1,
        )
        source_rows = np.flatnonzero(in_bounds)
        neighbor_flats = np.ravel_multi_index(
            tuple(neighbor_coordinates[in_bounds].T), volume_shape
        )
        positions = np.searchsorted(nodes, neighbor_flats)
        found = positions < nodes.size
        safe_positions = np.where(found, positions, 0)
        found &= nodes[safe_positions] == neighbor_flats
        source_rows = source_rows[found]
        target_rows = positions[found]
        if source_rows.size == 0:
            continue
        weight = 10.0 * math.sqrt(sum(value * value for value in offset))
        row_parts.extend((source_rows, target_rows))
        column_parts.extend((target_rows, source_rows))
        weight_parts.extend(
            (
                np.full(source_rows.size, weight, dtype=np.float32),
                np.full(source_rows.size, weight, dtype=np.float32),
            )
        )

    rows = np.concatenate(row_parts)
    columns = np.concatenate(column_parts)
    weights = np.concatenate(weight_parts)
    graph = coo_matrix(
        (weights, (rows, columns)), shape=(nodes.size, nodes.size)
    ).tocsr()
    component_count, component_labels = connected_components(graph, directed=False)
    cell_components = component_labels[cell_positions]
    if np.unique(cell_components).size != 1:
        raise GeometrySupportError(
            "SSp-tr cell pial endpoints span disconnected graph components"
        )
    diagnostics = {
        "bounded_surface_graph_nodes": int(nodes.size),
        "bounded_surface_graph_undirected_edges": int(graph.nnz // 2),
        "bounded_surface_graph_components": int(component_count),
        "cell_surface_component_size": int(
            np.sum(component_labels == cell_components[0])
        ),
        "bounding_box_min_voxel": lower.tolist(),
        "bounding_box_max_voxel": upper.tolist(),
    }
    return graph, cell_positions, diagnostics


def cell_geodesic_matrix(graph: object, cell_positions: np.ndarray) -> np.ndarray:
    unique_positions, inverse = np.unique(cell_positions, return_inverse=True)
    distances = np.full((unique_positions.size, unique_positions.size), np.inf)
    for index, source in enumerate(unique_positions):
        all_distances = dijkstra(
            graph,
            directed=False,
            indices=int(source),
            limit=float(max(RADII_UM)),
        )
        distances[index] = all_distances[unique_positions]
    matrix = distances[inverse][:, inverse]
    if not np.allclose(matrix, matrix.T, equal_nan=True):
        raise GeometrySupportError("surface geodesic matrix is asymmetric")
    return matrix


def build_geometry(
    records: list[CellRecord],
) -> tuple[
    list[CellRecord],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
    dict[str, object],
]:
    for required in (SURFACE_PATHS, CLOSEST_SURFACE):
        if not required.is_file():
            raise GeometrySupportError(f"missing Allen geometry asset: {required}")

    coordinates = np.asarray([record.coordinate for record in records], dtype=float)
    with h5py.File(SURFACE_PATHS, "r") as path_file:
        volume_shape = tuple(
            int(value)
            for value in path_file["volume lookup flat"].attrs["original shape"]
        )
    if volume_shape != (1320, 800, 1140):
        raise GeometrySupportError("unexpected 10-um CCF volume shape")

    reflected = coordinates.copy()
    right = reflected[:, 2] > volume_shape[2] * 10.0 / 2.0
    reflected[right, 2] = volume_shape[2] * 10.0 - reflected[right, 2]
    hemisphere_right = np.asarray(
        [record.hemisphere == "Right" for record in records], dtype=bool
    )
    hemisphere_coordinate_mismatches = int(np.sum(right != hemisphere_right))
    if hemisphere_coordinate_mismatches:
        raise GeometrySupportError(
            "provider hemisphere labels disagree with the CCF ML midline"
        )

    voxels = np.floor(reflected / 10.0).astype(int)
    if np.any(voxels < 0) or np.any(voxels >= np.asarray(volume_shape)):
        raise GeometrySupportError("an SSp-tr soma lies outside the 10-um atlas")
    flat_voxels = np.ravel_multi_index(tuple(voxels.T), volume_shape).astype(
        np.uint32
    )

    with h5py.File(CLOSEST_SURFACE, "r") as lookup_file:
        matching_surface = query_h5_sorted_pair_lookup(
            flat_voxels, lookup_file["closest surface voxel"]
        )

    with h5py.File(SURFACE_PATHS, "r") as path_file:
        path_indices = h5_sorted_take(
            path_file["volume lookup flat"], matching_surface.astype(np.int64)
        ).astype(int)
        valid_path = (
            (matching_surface > 0)
            & (path_indices >= 0)
            & (path_indices < path_file["paths"].shape[0])
        )
        selected_paths = np.zeros(
            (len(records), path_file["paths"].shape[1]), dtype=np.uint32
        )
        if np.any(valid_path):
            valid_indices = path_indices[valid_path]
            unique_indices, inverse = np.unique(valid_indices, return_inverse=True)
            selected_paths[valid_path] = path_file["paths"][unique_indices, :][inverse]
        endpoints = path_file["paths"][:, 0]

    depth_fraction = np.full(len(records), np.nan)
    api_depth_fraction = np.full(len(records), np.nan)
    physical_depth_um = np.full(len(records), np.nan)
    streamline_offset_um = np.full(len(records), np.nan)
    cell_endpoints = np.zeros(len(records), dtype=np.uint32)
    for index in range(len(records)):
        if not valid_path[index]:
            continue
        fraction, api_fraction, depth_um, offset_um = project_to_polyline(
            reflected[index], selected_paths[index], volume_shape
        )
        if (
            math.isfinite(fraction)
            and 0.0 <= fraction <= 1.0
            and math.isfinite(api_fraction)
            and 0.0 <= api_fraction <= 1.0
        ):
            depth_fraction[index] = fraction
            api_depth_fraction[index] = api_fraction
            physical_depth_um[index] = depth_um
            streamline_offset_um[index] = offset_um
            cell_endpoints[index] = selected_paths[index, 0]
        else:
            valid_path[index] = False

    eligible_indices = np.flatnonzero(valid_path)
    eligible_records = [records[int(index)] for index in eligible_indices]
    eligible_depth = depth_fraction[eligible_indices]
    eligible_api_depth = api_depth_fraction[eligible_indices]
    eligible_offsets = streamline_offset_um[eligible_indices]
    if np.any(cell_endpoints[valid_path] == 0):
        raise GeometrySupportError("a geometry-eligible cell lacks a pial endpoint")
    eligible_brain_counts = Counter(record.brain_id for record in eligible_records)
    if len(eligible_brain_counts) != EXPECTED_BRAINS:
        raise GeometrySupportError("geometry eligibility removes an entire brain")

    graph, cell_positions, graph_diagnostics = build_surface_graph(
        endpoints, cell_endpoints[valid_path], volume_shape
    )
    geodesic = cell_geodesic_matrix(graph, cell_positions)
    geometry_diagnostics = {
        "input_cells": len(records),
        "geometry_eligible_cells": len(eligible_records),
        "geometry_excluded_cells": len(records) - len(eligible_records),
        "geometry_eligible_brains": len(eligible_brain_counts),
        "minimum_geometry_eligible_cells_per_brain": min(
            eligible_brain_counts.values()
        ),
        "hemisphere_coordinate_mismatches": hemisphere_coordinate_mismatches,
        "streamline_offset_median_um": float(np.nanmedian(eligible_offsets)),
        "streamline_offset_p95_um": float(
            np.nanpercentile(eligible_offsets, 95)
        ),
        "streamline_offset_max_um": float(np.nanmax(eligible_offsets)),
        "physical_depth_median_um": float(
            np.nanmedian(physical_depth_um[valid_path])
        ),
        "depth_projection_absolute_difference_median": float(
            np.nanmedian(np.abs(eligible_depth - eligible_api_depth))
        ),
        "depth_projection_absolute_difference_max": float(
            np.nanmax(np.abs(eligible_depth - eligible_api_depth))
        ),
    }
    return (
        eligible_records,
        eligible_depth,
        eligible_api_depth,
        geodesic,
        geometry_diagnostics,
        graph_diagnostics,
    )


class ConstraintRows:
    """Small helper for constructing one sparse SciPy LinearConstraint."""

    def __init__(self, variable_count: int):
        self.variable_count = variable_count
        self.rows: list[int] = []
        self.columns: list[int] = []
        self.values: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []

    def add(
        self,
        coefficients: dict[int, float],
        lower: float = -math.inf,
        upper: float = math.inf,
    ) -> None:
        row = len(self.lower)
        for column, value in coefficients.items():
            if value:
                self.rows.append(row)
                self.columns.append(column)
                self.values.append(float(value))
        self.lower.append(float(lower))
        self.upper.append(float(upper))

    def linear_constraint(self) -> LinearConstraint:
        matrix = coo_matrix(
            (self.values, (self.rows, self.columns)),
            shape=(len(self.lower), self.variable_count),
        ).tocsr()
        return LinearConstraint(
            matrix,
            np.asarray(self.lower, dtype=float),
            np.asarray(self.upper, dtype=float),
        )


def neighbor_brain_sets(
    records: list[CellRecord],
    brain_index: dict[str, int],
    geodesic: np.ndarray,
    depth_fraction: np.ndarray,
    radius_um: int,
    depth_tolerance: float,
) -> list[set[int]]:
    neighbors = [set() for _ in records]
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if records[left].brain_id == records[right].brain_id:
                continue
            if geodesic[left, right] > radius_um + 1e-9:
                continue
            if (
                abs(depth_fraction[left] - depth_fraction[right])
                > depth_tolerance + 1e-12
            ):
                continue
            neighbors[left].add(brain_index[records[right].brain_id])
            neighbors[right].add(brain_index[records[left].brain_id])
    return neighbors


def solve_role_feasibility(
    records: list[CellRecord],
    neighbors: list[set[int]],
    time_limit_seconds: float,
) -> dict[str, object]:
    brains = sorted({record.brain_id for record in records})
    brain_index = {brain: index for index, brain in enumerate(brains)}
    brain_cells: dict[int, list[int]] = defaultdict(list)
    brain_line: dict[int, str] = {}
    for cell_index, record in enumerate(records):
        animal = brain_index[record.brain_id]
        brain_cells[animal].append(cell_index)
        prior = brain_line.setdefault(animal, record.line)
        if prior != record.line:
            raise GeometrySupportError("a brain has multiple strain/Cre lines")

    line_brains: dict[str, list[int]] = defaultdict(list)
    for animal, line in brain_line.items():
        line_brains[line].append(animal)
    singleton_lines = {
        line for line, members in line_brains.items() if len(members) == 1
    }
    if singleton_lines != {SINGLETON_LINE}:
        raise GeometrySupportError("unexpected singleton strain/Cre line")

    brain_count = len(brains)
    cell_count = len(records)
    development_offset = 0
    audit_offset = brain_count
    supported_development_offset = 2 * brain_count
    supported_audit_offset = 2 * brain_count + cell_count
    variable_count = 2 * brain_count + 2 * cell_count

    def development(animal: int) -> int:
        return development_offset + animal

    def audit(animal: int) -> int:
        return audit_offset + animal

    def supported_development(cell: int) -> int:
        return supported_development_offset + cell

    def supported_audit(cell: int) -> int:
        return supported_audit_offset + cell

    constraints = ConstraintRows(variable_count)
    constraints.add(
        {development(animal): 1.0 for animal in range(brain_count)},
        DEVELOPMENT_BRAINS,
        DEVELOPMENT_BRAINS,
    )
    constraints.add(
        {audit(animal): 1.0 for animal in range(brain_count)},
        AUDIT_BRAINS,
        AUDIT_BRAINS,
    )
    for animal in range(brain_count):
        constraints.add(
            {development(animal): 1.0, audit(animal): 1.0}, upper=1.0
        )

    # Every non-singleton strain/Cre line must occur in both roles.  The sole
    # singleton Tle4 brain is fixed unassigned through its variable bounds.
    for line, members in sorted(line_brains.items()):
        if line == SINGLETON_LINE:
            continue
        constraints.add(
            {development(animal): 1.0 for animal in members}, lower=1.0
        )
        constraints.add({audit(animal): 1.0 for animal in members}, lower=1.0)

    for cell, record in enumerate(records):
        owner = brain_index[record.brain_id]
        constraints.add(
            {
                supported_development(cell): 1.0,
                development(owner): -1.0,
            },
            upper=0.0,
        )
        constraints.add(
            {supported_audit(cell): 1.0, audit(owner): -1.0}, upper=0.0
        )
        development_neighbor_coefficients = {
            development(animal): -1.0 for animal in neighbors[cell]
        }
        constraints.add(
            {
                supported_development(cell): float(
                    MINIMUM_DISTINCT_DEVELOPMENT_NEIGHBORS
                ),
                **development_neighbor_coefficients,
            },
            upper=0.0,
        )
        constraints.add(
            {
                supported_audit(cell): float(
                    MINIMUM_DISTINCT_DEVELOPMENT_NEIGHBORS
                ),
                **development_neighbor_coefficients,
            },
            upper=0.0,
        )

    for animal, cell_indices in brain_cells.items():
        required = math.ceil(PER_ANIMAL_RETENTION * len(cell_indices) - 1e-12)
        development_coefficients = {
            supported_development(cell): 1.0 for cell in cell_indices
        }
        development_coefficients[development(animal)] = -float(required)
        constraints.add(development_coefficients, lower=0.0)
        audit_coefficients = {
            supported_audit(cell): 1.0 for cell in cell_indices
        }
        audit_coefficients[audit(animal)] = -float(required)
        constraints.add(audit_coefficients, lower=0.0)

    overall_development = {
        supported_development(cell): 1.0 for cell in range(cell_count)
    }
    overall_audit = {supported_audit(cell): 1.0 for cell in range(cell_count)}
    for animal, cell_indices in brain_cells.items():
        overall_development[development(animal)] = (
            -OVERALL_ROLE_RETENTION * len(cell_indices)
        )
        overall_audit[audit(animal)] = -OVERALL_ROLE_RETENTION * len(cell_indices)
    constraints.add(overall_development, lower=0.0)
    constraints.add(overall_audit, lower=0.0)

    multi_cell_brains = [
        animal for animal, cell_indices in brain_cells.items() if len(cell_indices) >= 2
    ]
    constraints.add(
        {development(animal): 1.0 for animal in multi_cell_brains},
        lower=float(MINIMUM_MULTI_CELL_DEVELOPMENT_BRAINS),
    )

    lower_bounds = np.zeros(variable_count, dtype=float)
    upper_bounds = np.ones(variable_count, dtype=float)
    singleton_animal = line_brains[SINGLETON_LINE][0]
    upper_bounds[development(singleton_animal)] = 0.0
    upper_bounds[audit(singleton_animal)] = 0.0
    linear_constraint = constraints.linear_constraint()
    result = milp(
        c=np.zeros(variable_count, dtype=float),
        integrality=np.ones(variable_count, dtype=np.uint8),
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=linear_constraint,
        options={"disp": False, "time_limit": float(time_limit_seconds)},
    )

    if result.status == 0 and result.x is not None:
        rounded = np.rint(result.x)
        if np.max(np.abs(result.x - rounded)) > 1e-6:
            raise GeometrySupportError("MILP returned a non-integral feasible point")
        values = rounded.astype(int)
        evaluated = linear_constraint.A @ values
        finite_lower = np.isfinite(linear_constraint.lb)
        finite_upper = np.isfinite(linear_constraint.ub)
        if np.any(evaluated[finite_lower] < linear_constraint.lb[finite_lower] - 1e-7):
            raise GeometrySupportError("MILP witness violates a lower constraint")
        if np.any(evaluated[finite_upper] > linear_constraint.ub[finite_upper] + 1e-7):
            raise GeometrySupportError("MILP witness violates an upper constraint")
        if int(values[development_offset:audit_offset].sum()) != DEVELOPMENT_BRAINS:
            raise GeometrySupportError("MILP witness has the wrong development count")
        if int(
            values[audit_offset:supported_development_offset].sum()
        ) != AUDIT_BRAINS:
            raise GeometrySupportError("MILP witness has the wrong audit count")
        # The witness is intentionally discarded here.  Only aggregate
        # existence is returned to the caller.
        return {
            "solver_status_code": int(result.status),
            "solver_status": "feasible",
            "feasible_role_allocation_exists": "yes",
        }
    if result.status == 2:
        return {
            "solver_status_code": int(result.status),
            "solver_status": "infeasible",
            "feasible_role_allocation_exists": "no",
        }
    return {
        "solver_status_code": int(result.status),
        "solver_status": "indeterminate",
        "feasible_role_allocation_exists": "unknown",
    }


def depth_label(depth_tolerance: float) -> str:
    return (
        "unrestricted_diagnostic"
        if math.isinf(depth_tolerance)
        else f"{depth_tolerance:.2f}"
    )


def check_monotonicity(rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    ordered_depths = [depth_label(value) for value in DEPTH_TOLERANCES]
    tracks = sorted({str(row["analysis_track"]) for row in rows})
    for track in tracks:
        conventions = sorted(
            {
                str(row["depth_projection_convention"])
                for row in rows
                if row["analysis_track"] == track
            }
        )
        for convention in conventions:
            track_rows = [
                row
                for row in rows
                if row["analysis_track"] == track
                and row["depth_projection_convention"] == convention
            ]
            known = {
                (
                    int(row["tangential_radius_um"]),
                    str(row["normalized_depth_tolerance"]),
                ): str(row["feasible_role_allocation_exists"])
                for row in track_rows
            }
            for depth in ordered_depths:
                seen_yes = False
                for radius in RADII_UM:
                    value = known[(radius, depth)]
                    if value == "yes":
                        seen_yes = True
                    elif value == "no" and seen_yes:
                        raise GeometrySupportError(
                            "radius feasibility is not monotone in "
                            f"{track}/{convention}"
                        )
            for radius in RADII_UM:
                seen_yes = False
                for depth in ordered_depths:
                    value = known[(radius, depth)]
                    if value == "yes":
                        seen_yes = True
                    elif value == "no" and seen_yes:
                        raise GeometrySupportError(
                            "depth feasibility is not monotone in "
                            f"{track}/{convention}"
                        )


def write_outputs(
    rows: list[dict[str, object]],
    inventory_diagnostics: dict[str, object],
    geometry_diagnostics: dict[str, object],
    graph_diagnostics: dict[str, object],
    output_csv: Path,
    output_md: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# EP11 Gao whole-cortex SSp-tr geometry-support screen",
        "",
        "## Scope and rule",
        "",
        "This is an outcome-blind soma-geometry feasibility screen for the Gao",
        "whole-cortex SSp-tr cohort. The observed provider layer labels are harmonized",
        "to one SSp-tr source; layer is retained as an observed covariate and is",
        "not treated as a discrete support stratum. No axon geometry, projection",
        "class, role identity, or solver witness is emitted.",
        "The primary track uses the 206 cells whose provider reconstruction",
        "distinguishes axon from dendrite. This flag alone does not certify",
        "reconstruction completeness. The eight Axon_only reconstructions enter",
        "only the explicitly labelled 214-cell sensitivity track and never enter",
        "primary retention or neighborhood denominators.",
        "",
        "A feasible allocation contains 12 development and 8 audit brain/sample",
        "units. Their interpretation as animals remains conditional on confirming",
        "that the units are independent. Every retained cell needs direct neighbors",
        "from at least two distinct development",
        "brains within the stated pial-geodesic radius and normalized-depth",
        "tolerance. Each role brain/sample unit must retain at least 90% of its",
        "cells, each role overall must retain at least 95%, and at least eight",
        "development",
        "brains must contain two or more retained cells. Every non-singleton",
        "strain/Cre line must occur in both roles; the singleton Tle4 brain is",
        "forced unassigned. The unrestricted-depth rows are tangential-only",
        "diagnostics.",
        "",
        "## Input and geometry diagnostics",
        "",
        f"- Downloaded inventory: {inventory_diagnostics['downloaded_cells']} cells; "
        f"primary axon/dendrite-distinguished set: "
        f"{inventory_diagnostics['primary_axon_dendrite_distinguished_eligible_cells']}; "
        f"Axon_only sensitivity additions: "
        f"{inventory_diagnostics['axon_only_sensitivity_cells']}; all span "
        f"{EXPECTED_BRAINS} fMOST brains. "
        f"sample-ID check: {inventory_diagnostics['sample_crosswalk_status']}.",
        f"- Geometry eligible: {geometry_diagnostics['geometry_eligible_cells']}/"
        f"{geometry_diagnostics['input_cells']} cells across "
        f"{geometry_diagnostics['geometry_eligible_brains']} brains; minimum "
        f"eligible cells per brain: "
        f"{geometry_diagnostics['minimum_geometry_eligible_cells_per_brain']}.",
        "- Layer cell counts: "
        + ", ".join(
            f"{key}={value}"
            for key, value in inventory_diagnostics["layer_cell_counts"].items()
        )
        + ".",
        "- Strain/Cre-line brain counts: "
        + ", ".join(
            f"{key}={value}"
            for key, value in inventory_diagnostics["line_brain_counts"].items()
        )
        + ".",
        f"- Streamline offset median / 95th percentile / maximum: "
        f"{geometry_diagnostics['streamline_offset_median_um']:.1f} / "
        f"{geometry_diagnostics['streamline_offset_p95_um']:.1f} / "
        f"{geometry_diagnostics['streamline_offset_max_um']:.1f} um.",
        "- Exact-versus-LineString3D normalized-depth absolute difference "
        f"median / maximum: "
        f"{geometry_diagnostics['depth_projection_absolute_difference_median']:.4f} / "
        f"{geometry_diagnostics['depth_projection_absolute_difference_max']:.4f}.",
        f"- Bounded pial graph nodes / undirected edges / components: "
        f"{graph_diagnostics['bounded_surface_graph_nodes']:,} / "
        f"{graph_diagnostics['bounded_surface_graph_undirected_edges']:,} / "
        f"{graph_diagnostics['bounded_surface_graph_components']:,}.",
        "",
        "## Allocation-existence results",
        "",
        "| Analysis track | Depth projection | Cells | Tangential radius (um) | Depth tolerance | Cells with >=2 candidate neighboring brains | MILP status | Feasible allocation exists |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['analysis_track']} | "
            f"{row['depth_projection_convention']} | "
            f"{row['geometry_eligible_cells']} | "
            f"{row['tangential_radius_um']} | "
            f"{row['normalized_depth_tolerance']} | "
            f"{row['cells_with_2plus_candidate_neighbor_brains']} | "
            f"{row['solver_status']} | "
            f"{row['feasible_role_allocation_exists']} |"
        )
    lines.extend(
        (
            "",
            "Feasibility means that at least one allocation satisfies the fixed",
            "metadata-only support rules. It does not select or lock an allocation,",
            "and it is not evidence for a projection group or a continuum.",
        )
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, object]:
    records, inventory_diagnostics = load_inventory(args.inventory.resolve())
    (
        all_geometry_records,
        all_depth_fraction,
        all_api_depth_fraction,
        all_geodesic,
        geometry_diagnostics,
        graph_diagnostics,
    ) = build_geometry(records)

    primary_positions = np.asarray(
        [
            index
            for index, record in enumerate(all_geometry_records)
            if record.primary_axon_dendrite_distinguished_eligible
        ],
        dtype=int,
    )
    primary_records = [
        all_geometry_records[int(index)] for index in primary_positions
    ]
    if len(primary_records) > 206 or len(
        {record.brain_id for record in primary_records}
    ) != EXPECTED_BRAINS:
        raise GeometrySupportError(
            "geometry qualification does not preserve all 36 primary brains"
        )
    scopes = (
        (
            "primary_206_axon_dendrite_distinguished",
            primary_records,
            primary_positions,
            all_geodesic[np.ix_(primary_positions, primary_positions)],
            206,
        ),
        (
            "sensitivity_214_including_8_axon_only",
            all_geometry_records,
            np.arange(len(all_geometry_records), dtype=int),
            all_geodesic,
            214,
        ),
    )
    depth_conventions = (
        ("exact_polyline_closest_point_primary", all_depth_fraction),
        (
            "ccf_streamlines_LineString3D_project_sensitivity",
            all_api_depth_fraction,
        ),
    )

    rows: list[dict[str, object]] = []
    for (
        analysis_track,
        track_records,
        track_positions,
        track_geodesic,
        track_cells_before_geometry,
    ) in scopes:
        brains = sorted({record.brain_id for record in track_records})
        brain_index = {brain: index for index, brain in enumerate(brains)}
        for depth_projection_convention, full_depth_fraction in depth_conventions:
            track_depth_fraction = full_depth_fraction[track_positions]
            for radius_um in RADII_UM:
                for depth_tolerance in DEPTH_TOLERANCES:
                    neighbors = neighbor_brain_sets(
                        track_records,
                        brain_index,
                        track_geodesic,
                        track_depth_fraction,
                        radius_um,
                        depth_tolerance,
                    )
                    solver = solve_role_feasibility(
                        track_records, neighbors, args.milp_time_limit_seconds
                    )
                    rows.append(
                        {
                        "source_region": EXPECTED_SOURCE,
                        "cohort": "Gao_whole_cortex_projectome",
                        "analysis_track": analysis_track,
                        "depth_projection_convention": depth_projection_convention,
                        "downloaded_cells": len(records),
                        "analysis_cells_before_geometry_qc": track_cells_before_geometry,
                        "input_brains": len(brains),
                        "geometry_eligible_cells": len(track_records),
                        "geometry_excluded_cells": track_cells_before_geometry
                        - len(track_records),
                        "harmonized_provider_layer_labels": len(
                            inventory_diagnostics["layer_cell_counts"]
                        ),
                        "strain_or_cre_lines": len(
                            inventory_diagnostics["line_brain_counts"]
                        ),
                        "hemispheres_folded": "yes",
                        "tangential_metric": "pial_endpoint_26_neighbor_surface_graph_geodesic_um",
                        "depth_metric": "pia_to_soma_path_length_divided_by_local_streamline_length",
                        "tangential_radius_um": radius_um,
                        "normalized_depth_tolerance": depth_label(
                            depth_tolerance
                        ),
                        "geometry_track": (
                            "tangential_only_diagnostic"
                            if math.isinf(depth_tolerance)
                            else (
                                "cortex_aware_primary_candidate"
                                if depth_projection_convention
                                == "exact_polyline_closest_point_primary"
                                else "cortex_aware_endpoint_handling_sensitivity"
                            )
                        ),
                        "cells_with_2plus_candidate_neighbor_brains": sum(
                            len(values)
                            >= MINIMUM_DISTINCT_DEVELOPMENT_NEIGHBORS
                            for values in neighbors
                        ),
                        "development_brains": DEVELOPMENT_BRAINS,
                        "audit_brains": AUDIT_BRAINS,
                        "unassigned_brains": len(brains)
                        - DEVELOPMENT_BRAINS
                        - AUDIT_BRAINS,
                        "minimum_per_role_brain_sample_retention": f"{PER_ANIMAL_RETENTION:.2f}",
                        "minimum_overall_retention_within_each_role": f"{OVERALL_ROLE_RETENTION:.2f}",
                        "minimum_development_brains_with_2plus_retained_cells": MINIMUM_MULTI_CELL_DEVELOPMENT_BRAINS,
                        "distinct_development_neighbor_brains_per_retained_cell": MINIMUM_DISTINCT_DEVELOPMENT_NEIGHBORS,
                        "non_singleton_line_in_each_role_minimum": 1,
                        "singleton_line_forced_unassigned": SINGLETON_LINE,
                        "independent_animal_interpretation_required": "yes",
                        **solver,
                        "role_assignment_emitted": "no",
                        "screen_status": "outcome_blind_geometry_feasibility_no_role_assignment",
                        }
                    )

    check_monotonicity(rows)
    write_outputs(
        rows,
        inventory_diagnostics,
        geometry_diagnostics,
        graph_diagnostics,
        args.output_csv.resolve(),
        args.output_md.resolve(),
    )
    return {
        "status": "complete",
        "rows": len(rows),
        "downloaded_cells": len(records),
        "primary_axon_dendrite_distinguished_cells": 206,
        "input_brains": EXPECTED_BRAINS,
        "primary_geometry_eligible_cells": len(primary_records),
        "primary_finite_depth_feasible_conditions": sum(
            row["analysis_track"]
            == "primary_206_axon_dendrite_distinguished"
            and row["geometry_track"] == "cortex_aware_primary_candidate"
            and row["feasible_role_allocation_exists"] == "yes"
            for row in rows
        ),
        "primary_depth_projection_sensitivity_feasible_conditions": sum(
            row["analysis_track"]
            == "primary_206_axon_dendrite_distinguished"
            and row["geometry_track"]
            == "cortex_aware_endpoint_handling_sensitivity"
            and row["feasible_role_allocation_exists"] == "yes"
            for row in rows
        ),
        "all_214_primary_depth_feasible_conditions": sum(
            row["analysis_track"]
            == "sensitivity_214_including_8_axon_only"
            and row["geometry_track"] == "cortex_aware_primary_candidate"
            and row["feasible_role_allocation_exists"] == "yes"
            for row in rows
        ),
        "all_214_depth_projection_sensitivity_feasible_conditions": sum(
            row["analysis_track"]
            == "sensitivity_214_including_8_axon_only"
            and row["geometry_track"]
            == "cortex_aware_endpoint_handling_sensitivity"
            and row["feasible_role_allocation_exists"] == "yes"
            for row in rows
        ),
        "indeterminate_conditions": sum(
            row["feasible_role_allocation_exists"] == "unknown" for row in rows
        ),
        "role_assignment_emitted": False,
        "output_csv": str(args.output_csv.resolve()),
        "output_md": str(args.output_md.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--milp-time-limit-seconds", type=float, default=600.0)
    args = parser.parse_args()
    if args.milp_time_limit_seconds <= 0:
        parser.error("--milp-time-limit-seconds must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), sort_keys=True))
