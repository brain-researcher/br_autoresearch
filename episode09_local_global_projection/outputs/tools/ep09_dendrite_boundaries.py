#!/usr/bin/env python3
"""Freeze EP09 development-dendrite boundaries before axon-field access.

Python 3.11; standard library, NumPy, and SciPy only.
All computations must run inside the orchestrator's Slurm allocation.
"""

from __future__ import annotations

import os

# Apply before importing NumPy/SciPy, including in spawned workers.
_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "OPENBLAS_DEFAULT_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _name in _THREAD_ENV:
    os.environ[_name] = "1"

if __name__ == "__main__" and not os.environ.get("SLURM_JOB_ID", "").strip():
    raise SystemExit("error: SLURM_JOB_ID is required; run inside a Slurm allocation.")

import argparse
import atexit
import csv
import hashlib
import io
import json
import math
import multiprocessing
import platform
import socket
import stat
import sys
import zipfile
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy


EXPECTED_ANIMALS = 27
EXPECTED_DEVELOPMENT_CELLS = 1114
GRID_UM = 25
RADIAL_TOLERANCE_UM = 1e-9
DISCRETE_TOLERANCE = 1e-12
RADIAL_PROBABILITIES = (0.25, 0.50, 0.75)
ORDER_PROBABILITIES = (1.0 / 3.0, 2.0 / 3.0)
CANDIDATE_TYPES = frozenset((1, 3, 4, 10, 11, 12))
DENDRITE_TYPES = frozenset((3, 4))
TREE_TYPES = frozenset((3, 4, 10, 11, 12))
WHITESPACE = b" \t\r\n\v\f"
CDF_CHUNK_SIZE = 262_144

OUTPUT_JSON = "dendrite_boundaries.json"
OUTPUT_CSV = "development_dendrite_qc.csv"
QC_COLUMNS = (
    "cell",
    "brain",
    "valid",
    "reason",
    "stems",
    "reached_nodes",
    "unreachable_dendrite_typed_nodes",
    "total_length_um",
    "max_radius_um",
)

_WORKER_ZIP: zipfile.ZipFile | None = None


class RunError(Exception):
    """Fatal input, allocation, numerical, or output error."""


class InvalidCell(Exception):
    """Dendrite QC failure; messages contain fixed reason codes only."""


@dataclass(frozen=True, slots=True)
class CellSpec:
    ordinal: int
    cell: str
    brain: str
    member: str


@dataclass(frozen=True, slots=True)
class Candidate:
    kind: int
    parent: int
    raw_xyz: tuple[bytes, bytes, bytes]


@dataclass(slots=True)
class Segments:
    length: np.ndarray
    projection: np.ndarray
    perpendicular: np.ndarray
    order: np.ndarray


@dataclass(slots=True)
class CellResult:
    ordinal: int
    cell: str
    brain: str
    valid: bool = False
    reason: str = ""
    stems: int | None = None
    reached_nodes: int | None = None
    unreachable_dendrite_typed_nodes: int | None = None
    total_length_um: float | None = None
    max_radius_um: float | None = None
    segments: Segments | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def file_stamp(path: Path) -> tuple[int, int, int, int, int]:
    st = path.stat()
    if not stat.S_ISREG(st.st_mode):
        raise RunError(f"Input must be a regular file: {path}")
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def member_for_cell(cell: str) -> str:
    # Freeze the final underscore as the B/S separator.
    brain_part, separator, sample_part = cell.rpartition("_")
    if (
        not separator
        or not brain_part
        or not sample_part
        or brain_part in (".", "..")
        or sample_part in (".", "..")
        or any(character in cell for character in ("/", "\\", "\x00"))
    ):
        raise RunError(f"Development cell is not a safe B_S identifier: {cell!r}")
    return f"data_v3/{brain_part}/{sample_part}.swc"


def read_manifest(path: Path) -> list[CellSpec]:
    required = {
        "cell", "brain", "line", "area", "layer",
        "pool_key", "pool_size", "ep09_role",
    }
    cells: list[CellSpec] = []
    seen_cells: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            header = next(reader)
        except StopIteration:
            raise RunError("Manifest is empty.") from None

        if len(header) != len(set(header)) or not required.issubset(header):
            raise RunError("Manifest has missing or duplicate column names.")

        role_index = header.index("ep09_role")
        cell_index = header.index("cell")
        brain_index = header.index("brain")

        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise RunError("Manifest contains a row with the wrong field count.")

            # Inspect the role before using any cell/animal fields.
            role = row[role_index].strip()
            if role == "audit":
                continue
            if role != "development":
                raise RunError("ep09_role must be exactly development or audit.")

            cell = row[cell_index].strip()
            brain = row[brain_index].strip()
            if not cell or not brain:
                raise RunError("A development row has an empty cell or brain.")
            if cell in seen_cells:
                raise RunError(f"Duplicate development cell in manifest: {cell}")
            seen_cells.add(cell)
            cells.append(CellSpec(len(cells), cell, brain, member_for_cell(cell)))

    if len(cells) != EXPECTED_DEVELOPMENT_CELLS:
        raise RunError(
            f"Expected {EXPECTED_DEVELOPMENT_CELLS} development rows; "
            f"found {len(cells)}."
        )
    animal_count = len({cell.brain for cell in cells})
    if animal_count != EXPECTED_ANIMALS:
        raise RunError(
            f"Expected {EXPECTED_ANIMALS} development animals; found {animal_count}."
        )
    return cells


def inspect_archive(path: Path, cells: list[CellSpec]) -> int:
    """Read central-directory metadata only; do not open member payloads."""
    selected = {cell.member: 0 for cell in cells}
    with zipfile.ZipFile(path, mode="r") as archive:
        infos = archive.infolist()
        member_count = len(infos)
        for info in infos:
            if info.filename in selected:
                selected[info.filename] += 1
                if info.is_dir() or info.flag_bits & 1:
                    raise RunError(
                        f"Development member is a directory or encrypted: "
                        f"{info.filename}"
                    )

    for member, count in selected.items():
        if count != 1:
            raise RunError(
                f"Development member must occur exactly once in ZIP: {member}"
            )
    return member_count


def initialize_worker(
    zip_path: str, expected_stamp: tuple[int, int, int, int, int]
) -> None:
    global _WORKER_ZIP
    if not os.environ.get("SLURM_JOB_ID", "").strip():
        raise RunError("Worker lacks SLURM_JOB_ID.")
    path = Path(zip_path)
    if file_stamp(path) != expected_stamp:
        raise RunError("ZIP changed before a worker opened it.")
    _WORKER_ZIP = zipfile.ZipFile(path, mode="r")
    atexit.register(_WORKER_ZIP.close)


def read_type_first(raw: bytes) -> int | None:
    """Locate and parse only the second token, without extracting the ID."""
    size = len(raw)
    position = 0
    while position < size and raw[position] in WHITESPACE:
        position += 1
    if position == size or raw[position] == 35:  # '#'
        return None

    # Traverse the first token solely to locate the type token.
    while (
        position < size
        and raw[position] not in WHITESPACE
        and raw[position] != 35
    ):
        position += 1
    while position < size and raw[position] in WHITESPACE:
        position += 1

    start = position
    while (
        position < size
        and raw[position] not in WHITESPACE
        and raw[position] != 35
    ):
        position += 1
    if start == position:
        raise InvalidCell("missing_type_token")

    try:
        return int(raw[start:position])
    except ValueError:
        raise InvalidCell("invalid_type_token") from None


def read_candidates(member: str) -> dict[int, Candidate]:
    if _WORKER_ZIP is None:
        raise RunError("Worker ZIP was not initialized.")

    nodes: dict[int, Candidate] = {}
    with _WORKER_ZIP.open(member, mode="r") as stream:
        for raw in stream:
            kind = read_type_first(raw)
            if kind is None:
                continue

            # Mandatory boundary: no split, ID/parent/coordinate parsing,
            # record creation, line counter, or other accounting for 0/2.
            if kind == 0 or kind == 2:
                continue

            if kind not in CANDIDATE_TYPES:
                raise InvalidCell("unsupported_type_code")

            fields = raw.split(b"#", 1)[0].split()
            if len(fields) != 7:
                raise InvalidCell("candidate_field_count_not_seven")
            try:
                node_id = int(fields[0])
                parent = int(fields[6])
            except ValueError:
                raise InvalidCell("invalid_candidate_id_or_parent") from None

            if node_id in nodes:
                raise InvalidCell("duplicate_candidate_id")

            # Radius is ignored. Coordinate tokens remain unparsed.
            nodes[node_id] = Candidate(
                kind=kind,
                parent=parent,
                raw_xyz=(fields[2], fields[3], fields[4]),
            )
    return nodes


def has_dendritic_ancestry_cycle(
    nodes: dict[int, Candidate], soma_id: int
) -> bool:
    """Check ancestry of typed dendrites; ignore marker-only components."""
    completed = {soma_id}
    for seed, node in nodes.items():
        if node.kind not in DENDRITE_TYPES or seed in completed:
            continue
        path: set[int] = set()
        current = seed
        while current in nodes and current not in completed:
            if current in path:
                return True
            path.add(current)
            current = nodes[current].parent
        completed.update(path)
    return False


def invalidate(result: CellResult, reasons: list[str]) -> CellResult:
    result.valid = False
    result.reason = ";".join(dict.fromkeys(reasons))
    result.segments = None
    return result


def parse_cell(spec: CellSpec) -> CellResult:
    result = CellResult(spec.ordinal, spec.cell, spec.brain)
    try:
        nodes = read_candidates(spec.member)
    except InvalidCell as error:
        return invalidate(result, [str(error)])

    soma_ids = [node_id for node_id, node in nodes.items() if node.kind == 1]
    if len(soma_ids) != 1:
        return invalidate(result, ["soma_count_not_one"])
    soma_id = soma_ids[0]

    reasons: list[str] = []
    if nodes[soma_id].parent != -1:
        reasons.append("soma_parent_not_minus_one")
    if has_dendritic_ancestry_cycle(nodes, soma_id):
        reasons.append("cyclic_dendritic_ancestry")

    children: dict[int, list[int]] = defaultdict(list)
    for node_id, node in nodes.items():
        if node.kind in TREE_TYPES:
            children[node.parent].append(node_id)

    stems = [
        node_id
        for node_id in children.get(soma_id, ())
        if nodes[node_id].kind in DENDRITE_TYPES
    ]
    result.stems = len(stems)

    reached: set[int] = set()
    ordered_ids: list[int] = []
    incoming_orders: list[int] = []
    queue = deque((node_id, 1) for node_id in stems)

    while queue:
        node_id, incoming_order = queue.popleft()
        if node_id in reached:
            reasons.append("repeated_node_during_dendritic_bfs")
            continue
        reached.add(node_id)
        ordered_ids.append(node_id)
        incoming_orders.append(incoming_order)

        child_ids = children.get(node_id, ())
        outgoing_order = incoming_order + int(len(child_ids) >= 2)
        queue.extend((child_id, outgoing_order) for child_id in child_ids)

    result.reached_nodes = len(ordered_ids)
    result.unreachable_dendrite_typed_nodes = sum(
        node.kind in DENDRITE_TYPES and node_id not in reached
        for node_id, node in nodes.items()
    )
    if result.unreachable_dendrite_typed_nodes:
        reasons.append("unreachable_type_3_or_4_nodes")
    if not stems:
        reasons.append("no_dendritic_stem")

    # Only the soma and BFS-reached nodes cross the coordinate parsing boundary.
    selected_ids = [soma_id, *ordered_ids]
    try:
        coordinates = np.asarray(
            [
                tuple(float(token) for token in nodes[node_id].raw_xyz)
                for node_id in selected_ids
            ],
            dtype=np.float64,
        )
    except (ValueError, OverflowError):
        return invalidate(result, [*reasons, "invalid_reached_coordinate"])

    if not np.isfinite(coordinates).all():
        return invalidate(result, [*reasons, "nonfinite_reached_coordinate"])

    if not ordered_ids:
        result.total_length_um = 0.0
        return invalidate(result, [*reasons, "zero_total_dendritic_length"])

    index_by_id = {node_id: index for index, node_id in enumerate(selected_ids)}
    parent_indices = np.asarray(
        [index_by_id[nodes[node_id].parent] for node_id in ordered_ids],
        dtype=np.intp,
    )

    # Unreachable markers and their raw coordinate tokens are now discarded.
    del nodes, children, reached, index_by_id, selected_ids, ordered_ids

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            centered = coordinates - coordinates[0]
            starts = centered[parent_indices]
            ends = centered[1:]
            delta = ends - starts
            lengths = np.hypot.reduce(delta, axis=1)
            radii = np.hypot.reduce(ends, axis=1)

            if not np.isfinite(lengths).all() or not np.isfinite(radii).all():
                raise FloatingPointError

            total_length = math.fsum(float(value) for value in lengths)
            maximum_radius = float(np.max(radii))
            if not math.isfinite(total_length):
                raise FloatingPointError

            result.total_length_um = total_length
            result.max_radius_um = maximum_radius

            if total_length <= 0.0:
                return invalidate(
                    result, [*reasons, "zero_total_dendritic_length"]
                )

            positive = lengths > 0.0
            positive_lengths = lengths[positive]
            unit_directions = delta[positive] / positive_lengths[:, None]
            positive_starts = starts[positive]

            # Along arc coordinate u in [0, length]:
            # |A + u*direction|^2 = perpendicular^2 + (projection + u)^2.
            projection = np.einsum(
                "ij,ij->i", positive_starts, unit_directions, optimize=False
            )
            perpendicular = np.hypot.reduce(
                np.cross(positive_starts, unit_directions), axis=1
            )
            if (
                not np.isfinite(projection).all()
                or not np.isfinite(perpendicular).all()
            ):
                raise FloatingPointError

    except (FloatingPointError, OverflowError):
        result.total_length_um = None
        result.max_radius_um = None
        return invalidate(result, [*reasons, "nonfinite_dendritic_geometry"])

    if reasons:
        return invalidate(result, reasons)

    result.valid = True
    result.segments = Segments(
        length=positive_lengths,
        projection=projection,
        perpendicular=perpendicular,
        order=np.asarray(incoming_orders, dtype=np.int64)[positive],
    )
    return result


@dataclass(slots=True)
class RadialCDF:
    length: np.ndarray
    projection: np.ndarray
    perpendicular: np.ndarray
    mass: np.ndarray
    maximum_radius: float

    def __call__(self, radius: float) -> float:
        if not math.isfinite(radius):
            raise RunError("CDF radius must be finite.")
        if radius <= 0.0:
            return 0.0
        if radius >= self.maximum_radius:
            # Every endpoint, and consequently every segment, is inside.
            return 1.0

        partial_sums: list[float] = []
        for begin in range(0, self.length.size, CDF_CHUNK_SIZE):
            end = begin + CDF_CHUNK_SIZE
            h = self.perpendicular[begin:end]
            intersects = h < radius
            if not np.any(intersects):
                continue

            h = h[intersects]
            s = self.projection[begin:end][intersects]
            length = self.length[begin:end][intersects]
            mass = self.mass[begin:end][intersects]

            # Analytic quadratic roots, evaluated without squaring radius.
            half_width = radius * np.sqrt(
                ((radius - h) / radius) * (1.0 + h / radius)
            )
            lower = np.maximum(0.0, -s - half_width)
            upper = np.minimum(length, -s + half_width)
            fraction = np.clip((upper - lower) / length, 0.0, 1.0)
            partial_sums.append(float(np.dot(mass, fraction)))

        value = math.fsum(partial_sums)
        if not math.isfinite(value) or not -1e-12 <= value <= 1.0 + 1e-12:
            raise RunError("Radial CDF failed its probability bounds check.")
        return min(1.0, max(0.0, value))

    def quantile(self, probability: float) -> tuple[float, float]:
        """Return a lower/upper bracket; use its upper endpoint as Q."""
        lower = 0.0
        upper = self.maximum_radius
        while upper - lower > RADIAL_TOLERANCE_UM:
            midpoint = lower + (upper - lower) / 2.0
            if midpoint == lower or midpoint == upper:
                raise RunError("Float precision cannot achieve 1e-9 um bisection.")
            if self(midpoint) >= probability:
                upper = midpoint
            else:
                lower = midpoint
        return lower, upper


def check_unit_mass(weights: list[float] | np.ndarray) -> None:
    total = math.fsum(float(weight) for weight in weights)
    if not math.isfinite(total) or abs(total - 1.0) > 1e-12:
        raise RunError("Animal-balanced weights do not sum to one.")


def discrete_quantile(
    values_and_weights: list[tuple[int | float, float]], probability: float
) -> int | float:
    check_unit_mass([weight for _, weight in values_and_weights])
    threshold = probability - DISCRETE_TOLERANCE
    cumulative = 0.0
    correction = 0.0
    for value, weight in sorted(values_and_weights, key=lambda pair: pair[0]):
        if weight < 0.0 or not math.isfinite(weight):
            raise RunError("Invalid discrete-quantile weight.")
        # Compensated cumulative sum, in deterministic sorted order.
        adjusted = weight - correction
        updated = cumulative + adjusted
        correction = (updated - cumulative) - adjusted
        cumulative = updated
        if cumulative >= threshold:
            return value
    raise RunError("Discrete quantile threshold was not reached.")


def assemble_distribution(
    results: list[CellResult], animal_names: list[str]
) -> tuple[
    RadialCDF,
    list[tuple[int, float]],
    list[tuple[float, float]],
    list[dict[str, Any]],
]:
    grouped: dict[str, list[CellResult]] = {animal: [] for animal in animal_names}
    development_counts = dict.fromkeys(animal_names, 0)
    for result in results:
        development_counts[result.brain] += 1
        if result.valid:
            grouped[result.brain].append(result)

    missing = [animal for animal in animal_names if not grouped[animal]]
    if missing:
        raise RunError(
            "No valid positive-length cell remains for animal(s): "
            + ", ".join(missing)
            + "; fixed 1/27 animal weights cannot be preserved."
        )

    per_animal: list[dict[str, Any]] = []
    for animal in animal_names:
        members = grouped[animal]
        mean_length = math.fsum(
            float(result.total_length_um) for result in members
        ) / len(members)
        per_animal.append(
            {
                "brain": animal,
                "development_cells": development_counts[animal],
                "valid_cells": len(members),
                "excluded_cells": development_counts[animal] - len(members),
                "animal_weight": 1.0 / EXPECTED_ANIMALS,
                "weighted_mean_dendritic_length_um": mean_length,
            }
        )

    valid = [result for result in results if result.valid]
    edge_count = sum(
        result.segments.length.size
        for result in valid
        if result.segments is not None
    )
    length = np.empty(edge_count, dtype=np.float64)
    projection = np.empty(edge_count, dtype=np.float64)
    perpendicular = np.empty(edge_count, dtype=np.float64)
    mass = np.empty(edge_count, dtype=np.float64)

    order_contributions: dict[int, list[float]] = defaultdict(list)
    maximum_values: list[tuple[float, float]] = []
    offset = 0

    # Results have already been sorted into development manifest order.
    for result in valid:
        segments = result.segments
        if segments is None or result.total_length_um is None:
            raise RunError("Valid cell lacks its segment geometry.")

        cell_weight = 1.0 / (EXPECTED_ANIMALS * len(grouped[result.brain]))
        count = segments.length.size
        target = slice(offset, offset + count)

        length[target] = segments.length
        projection[target] = segments.projection
        perpendicular[target] = segments.perpendicular
        mass[target] = cell_weight * (
            segments.length / result.total_length_um
        )

        histogram = np.bincount(segments.order, weights=mass[target])
        for order in range(1, histogram.size):
            if histogram[order] > 0.0:
                order_contributions[order].append(float(histogram[order]))

        maximum_values.append((float(result.max_radius_um), cell_weight))
        offset += count

        # Keep QC but release per-cell geometry after copying it.
        result.segments = None

    if offset != edge_count:
        raise RunError("Internal segment aggregation mismatch.")

    check_unit_mass(mass)
    order_values = [
        (order, math.fsum(contributions))
        for order, contributions in sorted(order_contributions.items())
    ]
    check_unit_mass([weight for _, weight in order_values])

    cdf = RadialCDF(
        length=length,
        projection=projection,
        perpendicular=perpendicular,
        mass=mass,
        maximum_radius=max(value for value, _ in maximum_values),
    )
    return cdf, order_values, maximum_values, per_animal


def radial_profiles(cdf: RadialCDF, cap: int) -> dict[str, Any]:
    grid_25 = list(range(0, cap + 1, GRID_UM))
    cdf_by_radius = {radius: cdf(float(radius)) for radius in grid_25}
    profiles: dict[str, Any] = {}

    for spacing in (25, 50, 100):
        grid = [radius for radius in grid_25 if radius % spacing == 0]
        if grid[-1] != cap:
            grid.append(cap)
        fractions = [
            cdf_by_radius[right] - cdf_by_radius[left]
            for left, right in zip(grid, grid[1:])
        ]
        if any(fraction < -1e-12 for fraction in fractions):
            raise RunError("Radial profile is not monotone.")
        profiles[str(spacing)] = {
            "nominal_spacing_um": spacing,
            "grid_um": grid,
            "weighted_length_fractions": [
                max(0.0, fraction) for fraction in fractions
            ],
            "overflow": {
                "lower_um": cap,
                "upper_um": None,
                "weighted_length_fraction": 1.0 - cdf_by_radius[cap],
            },
        }
    return profiles


def qc_record(result: CellResult) -> dict[str, Any]:
    return {name: getattr(result, name) for name in QC_COLUMNS}


def make_qc_bytes(results: list[CellResult]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=QC_COLUMNS, lineterminator="\n"
    )
    writer.writeheader()
    for result in results:
        row = qc_record(result)
        row["valid"] = "true" if result.valid else "false"
        for field in ("total_length_um", "max_radius_um"):
            if row[field] is not None:
                row[field] = format(row[field], ".17g")
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def frozen_conventions() -> dict[str, Any]:
    return {
        "contract": "EP09 measurement contract v1.2",
        "stage": "A; development dendrites before axon-field access",
        "manifest": {
            "selected_role": "development",
            "ignored_role": "audit",
            "expected_development_cells": EXPECTED_DEVELOPMENT_CELLS,
            "expected_development_animals": EXPECTED_ANIMALS,
            "animal_identifier": "brain column",
            "member_mapping": "Split cell at final underscore: data_v3/B/S.swc",
            "manifest_hash": "SHA-256 of complete raw manifest bytes",
        },
        "compartment": (
            "All soma-reachable type-3/type-4 dendritic trees, pooled; "
            "reachable type-10/11/12 markers inherit compartment; "
            "no missing apical/basal annotations are inferred."
        ),
        "coordinate_system": {
            "input_axes": ["AP", "DV", "ML"],
            "units": "micrometres",
            "origin": "Registered soma position subtracted before geometry",
            "absolute_coordinates_in_outputs": False,
        },
        "axon_blind_parser": (
            "Stream each selected member; extract type first; immediately "
            "discard type-0/2 rows without splitting or parsing other fields "
            "and without counting them. Retain only candidate id/type/parent "
            "and raw xyz tokens. Parse coordinates only for soma and BFS-reached "
            "dendrites. Unreachable marker coordinates remain unparsed."
        ),
        "validity": {
            "unique_ids": "All retained candidate IDs; ignored 0/2 IDs unread",
            "soma": "Exactly one candidate soma, with parent -1",
            "acyclic": (
                "Typed-dendrite ancestry in the candidate graph and no repeated "
                "node during dendritic BFS; marker-only components are ignored"
            ),
            "coordinates": "Finite soma and reached-node coordinates/geometry",
            "stems": "At least one soma child of type 3/4",
            "unreachable_type_3_4": "Any such node invalidates the cell",
            "length": "Strictly positive finite total dendritic length",
            "unavailable_qc": "Null in JSON and blank in CSV",
            "missing_valid_animal": (
                "Fatal; never renormalize away a development animal"
            ),
        },
        "geometry": {
            "edges": "Every reached parent-to-child edge, including soma-to-stem",
            "length": "Original Euclidean polyline arc length",
            "zero_length_edges": (
                "Retained in topology; omitted from all length-weighted measures"
            ),
            "maximum_radius": "Maximum soma distance over reached nodes",
        },
        "branch_order": {
            "stem_order": 1,
            "rule": (
                "Edges out of soma have order 1; out of a reached node with "
                "at least two reached children increment incoming order by 1; "
                "with one child preserve incoming order, regardless of type."
            ),
            "multifurcations": (
                "Retain as recorded; imbalance summaries use all unordered "
                "child-subtree pairs. This boundary tool does not estimate "
                "imbalance statistics."
            ),
        },
        "weighting": {
            "animal": "1/27",
            "cell": "1/N_valid within each animal",
            "location": "edge length / cell total length, uniformly along edge",
            "per_animal_mean_length": "Equal valid-cell weighted mean",
        },
        "quantiles": {
            "radial": (
                "Analytic sphere/segment intersection fraction from quadratic "
                "roots; generalized-inverse CDF quantile by bisection"
            ),
            "radial_bracket_width_um": RADIAL_TOLERANCE_UM,
            "reported_radial_quantile": "Upper bisection bracket endpoint",
            "radial_probabilities": list(RADIAL_PROBABILITIES),
            "discrete": "Smallest value whose CDF >= q - 1e-12",
            "discrete_cdf_tolerance": DISCRETE_TOLERANCE,
            "branch_order_probabilities": list(ORDER_PROBABILITIES),
            "cell_maximum_radius_probability": 0.99,
        },
        "boundaries": {
            "radial_rounding": "25 * floor(Q/25 + 0.5), ties upward",
            "radial_monotonicity": "e1>=25; e2>=e1+25; e3>=e2+25",
            "radial_shells": "[0,e1), [e1,e2), [e2,e3), [e3,infinity)",
            "branch_orders": (
                "b1=max(1,Q1); b2=max(b1+1,Q2); ranges 1..b1, "
                "b1+1..b2, >b2; retain empty ranges"
            ),
            "R_cap": "25 * ceil(Q_0.99(cell maximum radius)/25); no epsilon",
        },
        "radial_profiles": {
            "quantity": "Pooled animal-balanced dendritic arc-length fraction",
            "spacings_um": [25, 50, 100],
            "coarse_grid_endpoint": (
                "Append R_cap when needed, making the last interval shorter"
            ),
            "intervals": "[lower,upper); explicit overflow [R_cap,infinity)",
        },
        "numeric_reproducibility": (
            "Float64; one numerical thread per process; aggregate cells in "
            "development manifest order; fixed CDF chunk size"
        ),
    }


def write_outputs_exclusively(
    out_dir: Path, json_bytes: bytes, qc_bytes: bytes
) -> None:
    """Create both files exclusively; remove owned outputs on caught failure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = (
        (out_dir / OUTPUT_JSON, json_bytes),
        (out_dir / OUTPUT_CSV, qc_bytes),
    )
    opened: dict[Path, int] = {}
    identities: dict[Path, tuple[int, int]] = {}

    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)

        # Acquire both names before writing either payload.
        for path, _ in payloads:
            descriptor = os.open(path, flags, 0o600)
            opened[path] = descriptor
            info = os.fstat(descriptor)
            identities[path] = (info.st_dev, info.st_ino)

        for path, payload in payloads:
            descriptor = opened[path]
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise RunError("Output write made no progress.")
                remaining = remaining[written:]
            os.fsync(descriptor)

        for descriptor in opened.values():
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)

        for path in list(opened):
            descriptor = opened.pop(path)
            os.close(descriptor)

    except BaseException as error:
        cleanup_errors: list[str] = []
        for descriptor in opened.values():
            try:
                os.close(descriptor)
            except OSError:
                pass

        for path, identity in identities.items():
            try:
                current = path.lstat()
                if (current.st_dev, current.st_ino) != identity:
                    cleanup_errors.append(f"{path}: ownership changed")
                    continue
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                cleanup_errors.append(f"{path}: removal failed")

        # An fstat failure is unusual, but must not silently claim rollback.
        for path in opened:
            if path not in identities:
                cleanup_errors.append(f"{path}: creation identity unavailable")

        if cleanup_errors:
            raise RunError(
                "Output failure; rollback incomplete: " + "; ".join(cleanup_errors)
            ) from error
        raise


def cpu_budget() -> int:
    raw = os.environ.get("SLURM_CPUS_PER_TASK")
    if raw is None:
        budget = 1
    else:
        try:
            budget = int(raw)
        except ValueError:
            raise RunError("SLURM_CPUS_PER_TASK is not an integer.") from None
        if budget < 1:
            raise RunError("SLURM_CPUS_PER_TASK must be positive.")

    if hasattr(os, "sched_getaffinity"):
        budget = min(budget, len(os.sched_getaffinity(0)))
    if budget < 1:
        raise RunError("No CPU is available to this process.")
    return budget


def run(args: argparse.Namespace) -> None:
    started_utc = utc_now()
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    hostname = socket.gethostname()
    if not job_id:
        raise RunError("SLURM_JOB_ID is required.")

    budget = cpu_budget()
    if args.workers < 1 or args.workers > budget:
        raise RunError(
            f"--workers must be between 1 and {budget}; request sufficient "
            "--cpus-per-task in Slurm."
        )

    zip_path = args.zip.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    script_path = Path(__file__).resolve(strict=True)
    out_dir = args.out_dir.resolve()

    for name in (OUTPUT_JSON, OUTPUT_CSV):
        if os.path.lexists(out_dir / name):
            raise RunError(f"Refusing existing output: {out_dir / name}")

    stamps = {
        path: file_stamp(path)
        for path in (zip_path, manifest_path, script_path)
    }
    manifest_sha256 = sha256_file(manifest_path)
    script_sha256 = sha256_file(script_path)
    cells = read_manifest(manifest_path)
    animal_names = list(dict.fromkeys(cell.brain for cell in cells))
    member_count = inspect_archive(zip_path, cells)

    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker,
        initargs=(str(zip_path), stamps[zip_path]),
    ) as executor:
        results = list(executor.map(parse_cell, cells, chunksize=1))

    results.sort(key=lambda result: result.ordinal)
    if [result.ordinal for result in results] != list(range(len(cells))):
        raise RunError("Worker results do not match manifest order.")

    cdf, order_values, maximum_values, per_animal = assemble_distribution(
        results, animal_names
    )

    radial_brackets = {
        str(probability): cdf.quantile(probability)
        for probability in RADIAL_PROBABILITIES
    }
    radial_quantiles = {
        probability: bracket[1]
        for probability, bracket in radial_brackets.items()
    }
    rounded = [
        GRID_UM * math.floor(radial_quantiles[str(probability)] / GRID_UM + 0.5)
        for probability in RADIAL_PROBABILITIES
    ]
    edges = [
        max(GRID_UM, rounded[0]),
        0,
        0,
    ]
    edges[1] = max(rounded[1], edges[0] + GRID_UM)
    edges[2] = max(rounded[2], edges[1] + GRID_UM)

    q_order_1 = int(discrete_quantile(order_values, ORDER_PROBABILITIES[0]))
    q_order_2 = int(discrete_quantile(order_values, ORDER_PROBABILITIES[1]))
    b1 = max(1, q_order_1)
    b2 = max(b1 + 1, q_order_2)

    q_maximum = float(discrete_quantile(maximum_values, 0.99))
    cap = GRID_UM * math.ceil(q_maximum / GRID_UM)
    if cap <= 0:
        raise RunError("Computed R_cap is not positive.")

    boundaries = {
        "R_cap_um": cap,
        "radial_edges_um": edges,
        "branch_order_boundaries": {"b1": b1, "b2": b2},
    }
    boundary_sha256 = hashlib.sha256(
        canonical_json_bytes(boundaries)
    ).hexdigest()

    radial_cumulative = [0.0, *(cdf(float(edge)) for edge in edges), 1.0]
    radial_bounds: list[int | None] = [0, *edges, None]
    radial_shells = []
    for index in range(4):
        fraction = radial_cumulative[index + 1] - radial_cumulative[index]
        if fraction < -1e-12:
            raise RunError("Radial shell CDF is not monotone.")
        radial_shells.append(
            {
                "lower_um": radial_bounds[index],
                "upper_um": radial_bounds[index + 1],
                "weighted_length_fraction": max(0.0, fraction),
            }
        )

    branch_ranges = [
        {
            "lower_order": 1,
            "upper_order": b1,
            "weighted_length_fraction": math.fsum(
                weight for order, weight in order_values if order <= b1
            ),
        },
        {
            "lower_order": b1 + 1,
            "upper_order": b2,
            "weighted_length_fraction": math.fsum(
                weight for order, weight in order_values if b1 < order <= b2
            ),
        },
        {
            "lower_order": b2 + 1,
            "upper_order": None,
            "weighted_length_fraction": math.fsum(
                weight for order, weight in order_values if order > b2
            ),
        },
    ]

    profiles = radial_profiles(cdf, cap)
    qc_bytes = make_qc_bytes(results)
    used = sum(result.valid for result in results)
    excluded = len(results) - used

    artifact = {
        "artifact": "EP09 development-dendrite explanatory boundaries",
        "schema_version": 1,
        "measurement_contract": "EP09 v1.2",
        "stage": "A",
        "boundaries": boundaries,
        "boundary_sha256": boundary_sha256,
        "boundary_hash_definition": {
            "algorithm": "SHA-256",
            "scope": "The exact boundaries object",
            "serialization": (
                "UTF-8 JSON; sort_keys=True; separators=(',', ':'); "
                "ensure_ascii=True; allow_nan=False; no trailing newline"
            ),
        },
        "unrounded_quantiles": {
            "radial_distance_um": radial_quantiles,
            "radial_bisection_brackets_um": radial_brackets,
            "branch_order": {
                str(ORDER_PROBABILITIES[0]): q_order_1,
                str(ORDER_PROBABILITIES[1]): q_order_2,
            },
            "cell_maximum_radius_um": {"0.99": q_maximum},
        },
        "descriptive": {
            "radial_shells": radial_shells,
            "branch_order_ranges": branch_ranges,
            "radial_profiles": profiles,
            "animal_balanced_mean_dendritic_length_um": math.fsum(
                animal["weighted_mean_dendritic_length_um"]
                for animal in per_animal
            ) / EXPECTED_ANIMALS,
        },
        "per_animal": per_animal,
        "cells_used": used,
        "cells_excluded": excluded,
        "excluded_cells": [
            qc_record(result) for result in results if not result.valid
        ],
        "frozen_conventions": frozen_conventions(),
        "provenance": {
            "SLURM_JOB_ID": job_id,
            "hostname": hostname,
            "run_started_utc": started_utc,
            "artifact_prepared_utc": utc_now(),
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "workers": args.workers,
            "cpu_budget": budget,
            "thread_environment": {
                name: os.environ[name] for name in _THREAD_ENV
            },
            "manifest": {
                "path": str(manifest_path),
                "sha256": manifest_sha256,
            },
            "script": {
                "path": str(script_path),
                "sha256": script_sha256,
            },
            "zip": {
                "path": str(zip_path),
                "size_bytes": stamps[zip_path][2],
                "member_count": member_count,
                "member_count_includes_directory_entries": True,
                "payload_selection": "Development manifest members only",
                "whole_archive_content_hash_computed": False,
            },
            "qc_csv": {
                "filename": OUTPUT_CSV,
                "sha256": hashlib.sha256(qc_bytes).hexdigest(),
            },
        },
    }
    json_bytes = (
        json.dumps(
            artifact,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")

    for path, expected_stamp in stamps.items():
        if file_stamp(path) != expected_stamp:
            raise RunError(f"Input changed during boundary generation: {path}")

    write_outputs_exclusively(out_dir, json_bytes, qc_bytes)

    # A closed stdout does not invalidate successfully committed artifacts.
    try:
        print(
            f"R_cap={cap} um; radial_edges={edges} um; "
            f"b1={b1}; b2={b2}; cells_used={used}; cells_excluded={excluded}"
        )
        print(f"boundary_sha256={boundary_sha256}")
    except (BrokenPipeError, OSError):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze EP09 development-dendrite boundaries in Slurm."
    )
    parser.add_argument("zip", type=Path, metavar="ZIP")
    parser.add_argument("manifest", type=Path, metavar="CELL_MANIFEST_CSV")
    parser.add_argument("out_dir", type=Path, metavar="OUT_DIR")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="K",
        help="Parsing processes; default 1; limited by SLURM_CPUS_PER_TASK.",
    )
    args = parser.parse_args()

    try:
        run(args)
    except RunError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("error: interrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        # Do not echo arbitrary parser/library exceptions containing input data.
        print(
            f"error: {type(error).__name__}; boundary generation aborted.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
