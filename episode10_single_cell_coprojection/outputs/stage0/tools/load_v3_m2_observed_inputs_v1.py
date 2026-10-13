#!/usr/bin/env python3
"""Load the sealed EP10 v3 development inputs for the observed M2 runner.

This module is deliberately a narrow bridge around the already-qualified M1
loaders.  It opens only the exact target-blind interface and atlas-geometry
roots plus the positive-only development membership projection.  There is no
argument or API route for final or unassigned data, SWCs, M0 artifacts, the M1
full refit, predictions, or solver telemetry.

The returned objects use the exact :mod:`fit_v3_m2` class identities required
by the production provenance checks.  The separate outer-M1 importer can then
consume the same ordered cells and uniform moments without reopening target
membership.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


SCRIPT_PATH = Path(os.path.abspath(__file__))
TOOLS_DIR = SCRIPT_PATH.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m1_real_v1 as real_m1  # noqa: E402
import fit_v3_m2 as m2  # noqa: E402
import v3_m2_m1_outer_import_v1 as outer_import  # noqa: E402


EPISODE_SCRATCH_COMPONENTS = (
    "br_autoresearch",
    "episode10_single_cell_coprojection",
)
INTERFACE_ROOT_BASENAME = "v3_model_interface_preflight_v1"
GEOMETRY_ROOT_BASENAME = "v3_m1_geometry_v1"
TARGET_ROOT_BASENAME = real_m1.TARGET_ROOT_BASENAME
EXPECTED_ROWS = real_m1.EXPECTED_ROWS
EXPECTED_GROUPS = real_m1.EXPECTED_GROUPS


class ObservedInputError(RuntimeError):
    """A sealed path, development join, or live M2 input invariant failed."""


@dataclass(frozen=True)
class PositionStrata:
    """Target-blind soma octants frozen before development targets are opened."""

    medians_xyz: tuple[float, float, float]
    cell_keys: tuple[str, ...]
    groups: tuple[str, ...]
    octants: tuple[str, ...]


@dataclass(frozen=True)
class ObservedInputBundle:
    """The five aligned live inputs consumed by ``run_observed_search``."""

    cells: tuple[m2.Cell, ...]
    moments: tuple[m2.UniformCellMoments, ...]
    candidates: m2.CandidateLibrary
    geometry: m2.GeometryEngine
    pairs: m2.PairFeatureLibrary
    position_strata: PositionStrata

    def runner_kwargs(self) -> Mapping[str, object]:
        """Return an immutable keyword mapping for the observed runner."""

        return MappingProxyType(
            {
                "cells": self.cells,
                "moments": self.moments,
                "candidates": self.candidates,
                "geometry": self.geometry,
                "pairs": self.pairs,
                "position_strata": self.position_strata,
            }
        )


def _position_strata_fields(cells: tuple[m2.Cell, ...]) -> tuple[object, ...]:
    if len(cells) != EXPECTED_ROWS:
        raise ObservedInputError("position strata require exactly 194 cells")
    coordinates = np.asarray([cell.soma_um for cell in cells], dtype=float)
    if coordinates.shape != (EXPECTED_ROWS, 3) or not np.all(
        np.isfinite(coordinates)
    ):
        raise ObservedInputError("target-blind soma coordinates are malformed")
    medians_array = np.median(coordinates, axis=0)
    medians = tuple(float(value) for value in medians_array)
    octants = tuple(
        "".join("1" if coordinate >= median else "0" for coordinate, median in zip(row, medians, strict=True))
        for row in coordinates
    )
    cell_keys = tuple(str(cell.identifier) for cell in cells)
    groups = tuple(str(cell.group) for cell in cells)
    if (
        len(set(cell_keys)) != EXPECTED_ROWS
        or any(value not in {"000", "001", "010", "011", "100", "101", "110", "111"} for value in octants)
    ):
        raise ObservedInputError("target-blind position strata are malformed")
    coordinates_tuple = tuple(
        tuple(float(value) for value in row) for row in coordinates
    )
    return medians, cell_keys, groups, octants, coordinates_tuple


def _freeze_position_strata(cells: tuple[m2.Cell, ...]) -> PositionStrata:
    """Freeze the pre-outcome median/tie convention on target-blind cells."""

    fields = _position_strata_fields(cells)
    medians, cell_keys, groups, octants, _ = fields
    result = PositionStrata(
        medians_xyz=medians,
        cell_keys=cell_keys,
        groups=groups,
        octants=octants,
    )
    return result


def _require_position_strata(
    value: object,
    cells: tuple[m2.Cell, ...],
) -> PositionStrata:
    """Reauthenticate the exact pre-target issuance against target-blind fields."""

    if type(value) is not PositionStrata:
        raise ObservedInputError("position strata lack the exact loader type")
    fields = _position_strata_fields(cells)
    if (
        value.medians_xyz != fields[0]
        or value.cell_keys != fields[1]
        or value.groups != fields[2]
        or value.octants != fields[3]
    ):
        raise ObservedInputError(
            "pre-target position strata changed after target alignment"
        )
    return value


def _episode_scratch_root() -> Path:
    scratch_value = os.environ.get("SCRATCH")
    if not scratch_value:
        raise ObservedInputError("SCRATCH is unset")
    try:
        scratch_root = Path(scratch_value).resolve(strict=True)
        episode_root = scratch_root.joinpath(*EPISODE_SCRATCH_COMPONENTS)
        resolved_episode = episode_root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ObservedInputError("the EP10 Scratch root is unavailable") from error
    if resolved_episode != episode_root or not episode_root.is_dir():
        raise ObservedInputError("the EP10 Scratch root is symlinked or not a directory")
    return episode_root


def _require_exact_directory(path: str | os.PathLike[str], expected: Path, label: str) -> Path:
    """Require one exact, existing, wholly unsymlinked directory path."""

    try:
        candidate = Path(os.path.abspath(os.fspath(path)))
    except (TypeError, ValueError, OSError) as error:
        raise ObservedInputError(f"{label} is not a valid filesystem path") from error
    if candidate != expected:
        raise ObservedInputError(f"{label} must be the exact sealed EP10 path")
    if candidate.is_symlink() or not candidate.is_dir():
        raise ObservedInputError(f"{label} is absent, symlinked, or not a directory")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ObservedInputError(f"{label} cannot be resolved safely") from error
    if resolved != candidate:
        raise ObservedInputError(f"{label} contains a symlinked path component")
    return candidate


def _validate_input_roots(
    interface_root: str | os.PathLike[str],
    geometry_root: str | os.PathLike[str],
    target_root: str | os.PathLike[str],
) -> tuple[Path, Path, Path]:
    """Validate all three permitted roots before any payload file is opened."""

    episode_root = _episode_scratch_root()
    interface = _require_exact_directory(
        interface_root,
        episode_root / INTERFACE_ROOT_BASENAME,
        "target-blind interface root",
    )
    geometry = _require_exact_directory(
        geometry_root,
        episode_root / GEOMETRY_ROOT_BASENAME,
        "target-blind geometry root",
    )
    target = _require_exact_directory(
        target_root,
        episode_root / TARGET_ROOT_BASENAME,
        "positive-only development target root",
    )
    return interface, geometry, target


def _validate_bundle(bundle: ObservedInputBundle) -> None:
    cells = bundle.cells
    moments = bundle.moments
    if type(cells) is not tuple or type(moments) is not tuple:
        raise ObservedInputError("observed cells and moments must be immutable tuples")
    if len(cells) != EXPECTED_ROWS or len(moments) != EXPECTED_ROWS:
        raise ObservedInputError("observed inputs do not contain exactly 194 cells")
    if type(bundle.candidates) is not m2.CandidateLibrary:
        raise ObservedInputError("candidate library is not the exact M2 class")
    if (
        type(bundle.geometry) is not m2.GeometryEngine
        or bundle.geometry.library is not bundle.candidates
    ):
        raise ObservedInputError("geometry is not bound to the exact candidate library")
    if (
        type(bundle.pairs) is not m2.PairFeatureLibrary
        or bundle.pairs.candidates is not bundle.candidates
    ):
        raise ObservedInputError("pair features are not bound to the exact candidate library")

    expected_group_sequence = tuple(
        group
        for group in m2.m1.GROUP_ORDER
        for _ in range(m2.m1.GROUP_SIZES[group])
    )
    observed_group_sequence = tuple(cell.group for cell in cells)
    identifiers = tuple(cell.identifier for cell in cells)
    if observed_group_sequence != expected_group_sequence:
        raise ObservedInputError("observed cells changed frozen group order or support")
    if len(set(identifiers)) != EXPECTED_ROWS or any(not value for value in identifiers):
        raise ObservedInputError("observed cell identifiers are empty or duplicated")
    if len(set(observed_group_sequence)) != EXPECTED_GROUPS:
        raise ObservedInputError("observed inputs do not contain exactly 12 groups")
    _require_position_strata(bundle.position_strata, cells)

    for cell, moment in zip(cells, moments, strict=True):
        if type(cell) is not m2.Cell:
            raise ObservedInputError("observed cell is not the exact M2 Cell class")
        if type(cell.k) is not int or not 0 <= cell.k <= m2.m1.PANEL_SIZE:
            raise ObservedInputError("observed K is not an integer in 0..16")
        if type(cell.observed_index) is not int:
            raise ObservedInputError("observed candidate index is not an integer")
        basis = bundle.candidates.basis(cell.k)
        if not 0 <= cell.observed_index < basis.count:
            raise ObservedInputError("observed candidate index is outside fixed-K Omega")
        distances = np.asarray(cell.soma_distance_mm, dtype=float)
        if distances.shape != (m2.m1.PANEL_SIZE,) or not np.all(np.isfinite(distances)):
            raise ObservedInputError("cell-target distance vector is malformed")
        if type(cell.segment_unmapped) is not int or cell.segment_unmapped not in (0, 1):
            raise ObservedInputError("segment_unmapped is not an integer binary flag")
        if type(moment) is not m2.UniformCellMoments:
            raise ObservedInputError("uniform moment is not the exact M2 moment class")
        center = np.asarray(moment.geometry_mean, dtype=float)
        covariance = np.asarray(moment.raw_joint_covariance, dtype=float)
        if (
            center.shape != (m2.m1.GEOMETRY_DIMENSION,)
            or covariance.shape
            != (m2.m1.JOINT_CELL_DIMENSION, m2.m1.JOINT_CELL_DIMENSION)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(covariance))
        ):
            raise ObservedInputError("uniform candidate moments are malformed")


def load_observed_inputs(
    interface_root: str | os.PathLike[str],
    geometry_root: str | os.PathLike[str],
    target_root: str | os.PathLike[str],
) -> ObservedInputBundle:
    """Load and align the exact 194 development cells for observed v3 M2.

    Target membership is projected through the same binary streaming reader as
    the successful real-M1 run: only ID, group, K, and the 16 mapped-positive
    indicators are materialized.  Zeros retain their frozen unknown/not-
    observed semantics, and ``segment_unmapped`` remains a target-blind model
    covariate.
    """

    interface, geometry_root_path, target = _validate_input_roots(
        interface_root, geometry_root, target_root
    )
    try:
        target_blind_cells, pair_geometry = m2.m1.load_target_free_real_shape(
            interface, geometry_root_path
        )
        # This is intentionally before the first development-target read.
        position_strata = _freeze_position_strata(tuple(target_blind_cells))
        expected_ids = {cell.identifier for cell in target_blind_cells}
        membership, counts = real_m1._read_membership_inputs(target, expected_ids)
        candidates = m2.CandidateLibrary()
        cells = real_m1._align_membership(
            target_blind_cells, membership, candidates
        )
        del membership
        if not (
            counts.total_rows == real_m1.EXPECTED_TARGET_ROWS
            and counts.matched_rows == EXPECTED_ROWS
            and counts.extra_rows == real_m1.EXPECTED_EXCLUDED_ROWS
        ):
            raise ObservedInputError("positive-only development projection counts changed")
        geometry = m2.GeometryEngine(candidates, pair_geometry)
        moments = tuple(geometry.uniform_moments(cell) for cell in cells)
        pairs = m2.PairFeatureLibrary(candidates)
    except ObservedInputError:
        raise
    except (real_m1.RealM1Error, m2.m1.M1Error, m2.M2Error) as error:
        raise ObservedInputError("sealed observed-input loading failed") from error

    bundle = ObservedInputBundle(
        cells=tuple(cells),
        moments=moments,
        candidates=candidates,
        geometry=geometry,
        pairs=pairs,
        position_strata=position_strata,
    )
    _validate_bundle(bundle)
    return bundle


def import_outer_m1_for_observed(
    bundle: ObservedInputBundle,
    checkpoint_root: str | os.PathLike[str],
    *,
    execution_inventory: Any,
    expected_rank: int = outer_import.executor.EXPECTED_PRODUCTION_RANK,
) -> outer_import.ImportedOuterM1:
    """Import only the twelve producer outer refits for this live bundle."""

    if type(bundle) is not ObservedInputBundle:
        raise ObservedInputError("M1 import requires an exact observed-input bundle")
    _validate_bundle(bundle)
    episode_root = _episode_scratch_root()
    expected = outer_import.expected_checkpoint_root(
        episode_root.parent.parent
    )
    checkpoint = _require_exact_directory(
        checkpoint_root, expected, "M1 outer-refit checkpoint root"
    )
    try:
        return outer_import.import_outer_m1_offsets(
            checkpoint,
            bundle.cells,
            bundle.moments,
            mode=outer_import.executor.SEALED_PRODUCTION_MODE,
            execution_inventory=execution_inventory,
            expected_rank=expected_rank,
        )
    except outer_import.M1OuterImportError as error:
        raise ObservedInputError("sealed M1 outer-refit import failed") from error


def add_observed_input_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add only the four permitted sealed-development roots to ``parser``."""

    parser.add_argument("--interface-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--m1-checkpoint-root", type=Path, required=True)
    return parser


def build_parser() -> argparse.ArgumentParser:
    return add_observed_input_arguments(argparse.ArgumentParser(description=__doc__))


def load_observed_inputs_from_namespace(args: argparse.Namespace) -> ObservedInputBundle:
    """Bridge parsed launcher arguments to :func:`load_observed_inputs`."""

    return load_observed_inputs(args.interface_root, args.geometry_root, args.target_root)


__all__ = [
    "EXPECTED_GROUPS",
    "EXPECTED_ROWS",
    "GEOMETRY_ROOT_BASENAME",
    "INTERFACE_ROOT_BASENAME",
    "ObservedInputBundle",
    "ObservedInputError",
    "PositionStrata",
    "TARGET_ROOT_BASENAME",
    "add_observed_input_arguments",
    "build_parser",
    "import_outer_m1_for_observed",
    "load_observed_inputs",
    "load_observed_inputs_from_namespace",
]
