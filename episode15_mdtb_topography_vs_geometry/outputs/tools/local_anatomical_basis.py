"""Anatomy-only, fixed-support basis for the prospective local M3 recipe.

No files, participant responses, fitting, scoring or reader release occur here.
Coordinates must be common-space millimetres on the exact supplied voxel
axis (208 by default, or the adopted explicit common subset). The reference
is three median splits and two modes per leaf. Explicit
depth/capacity variants use the same construction, without selecting a panel.

Local support certifies a distance bound only when the consumer constrains O
to the returned blocks. A global orthogonal fit may mix those blocks. Task-A
full global/local design rank and unique fitted operators are separate duties.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BasisTolerances:
    """Caller-frozen tolerances; this module supplies no scientific defaults.

    Eigenvalue ties use atol + rtol * max(abs(pair)); groups are anchored at
    their smallest eigenvalue, not joined by chaining near-equalities.
    rank_rtol is relative to the largest centered Laplacian eigenvalue and is
    also the absolute pivot cutoff for dimensionless orthogonal projectors.
    numerical_atol bounds centering and orthonormality residuals.
    """
    eigen_tie_atol: float
    eigen_tie_rtol: float
    rank_rtol: float
    numerical_atol: float

    def __post_init__(self):
        values = (self.eigen_tie_atol, self.eigen_tie_rtol,
                  self.rank_rtol, self.numerical_atol)
        if not all(np.isfinite(v) for v in values):
            raise ValueError("basis tolerances must be finite")
        if self.eigen_tie_atol < 0 or not 0 <= self.eigen_tie_rtol < 1:
            raise ValueError("eigenvalue tie tolerances must be nonnegative")
        if not 0 < self.rank_rtol < 1 or not 0 < self.numerical_atol < 1:
            raise ValueError("rank and numerical tolerances must be in (0, 1)")


@dataclass(frozen=True)
class LocalAnatomicalBasis:
    B: np.ndarray
    coordinates: np.ndarray
    voxel_ids: tuple
    cell_labels: np.ndarray
    blocks: tuple
    cell_voxel_ids: tuple
    max_cell_diameter_mm: float
    split_depth: int
    modes_per_cell: int
    tolerances: BasisTolerances
    diagnostics: dict


def _positive_integer(value, name, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < (0 if allow_zero else 1):
        raise ValueError(f"{name} is outside its allowed range")
    return int(value)


def _partition(coordinates, voxel_ids, depth):
    """DFS low-to-high leaves; exact axis ties x,y,z and coordinate ties ID."""
    splits = []

    def visit(rows, remaining):
        if remaining == 0:
            return [rows[np.argsort(voxel_ids[rows], kind="stable")]]
        if len(rows) < 2:
            raise ValueError("split depth leaves too few fixed-support voxels")
        extent = np.ptp(coordinates[rows], axis=0)
        axis = int(np.argmax(extent))
        order = np.lexsort((voxel_ids[rows], coordinates[rows, axis]))
        ordered = rows[order]
        middle = len(rows) // 2
        splits.append({"depth": depth-remaining, "axis": axis,
                       "voxel_count": len(rows), "extent_mm": extent.tolist()})
        return (visit(ordered[:middle], remaining-1)
                + visit(ordered[middle:], remaining-1))

    return visit(np.arange(len(voxel_ids)), depth), splits


def _centered_coordinates(size):
    """Deterministic Helmert coordinates on native-ID-sorted cell rows."""
    result = np.zeros((size, size-1), dtype=np.float64)
    for column in range(size-1):
        count = column+1
        denominator = np.sqrt(count*(count+1))
        result[:count, column] = 1/denominator
        result[count, column] = -count/denominator
    return result


def _canonical_eigenspace(vectors, tolerances):
    """Resolve rotations/signs by projected native-ID coordinate vectors.

    The complete eigenspace is supplied even when a retained-mode boundary
    falls inside it. Its projector is invariant to an eigensolver's rotation.
    Two orthogonalization passes limit loss of orthogonality.
    """
    projector = vectors @ vectors.T
    selected = []
    for row in range(len(projector)):
        candidate = projector[:, row].copy()
        for _ in range(2):
            for direction in selected:
                candidate -= direction * np.dot(direction, candidate)
        norm = float(np.linalg.norm(candidate))
        if norm <= tolerances.rank_rtol:
            continue
        candidate /= norm
        pivots = np.flatnonzero(np.abs(candidate) > tolerances.rank_rtol)
        if not len(pivots):
            raise ArithmeticError("canonical eigenspace has no stable sign pivot")
        if candidate[pivots[0]] < 0:
            candidate *= -1
        selected.append(candidate)
        if len(selected) == vectors.shape[1]:
            break
    if len(selected) != vectors.shape[1]:
        raise ArithmeticError("canonical eigenspace loses rank at the frozen cutoff")
    return np.column_stack(selected)


def _low_modes(laplacian, modes, tolerances):
    """Diagonalize the centered space; never identify the constant by position."""
    contrasts = _centered_coordinates(len(laplacian))
    reduced = contrasts.T @ laplacian @ contrasts
    reduced = (reduced+reduced.T)/2
    eigenvalues, eigenvectors = np.linalg.eigh(reduced)
    cutoff = tolerances.rank_rtol * float(np.max(np.abs(eigenvalues)))
    if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues <= cutoff):
        raise ArithmeticError("cell graph lacks full nonconstant rank at frozen cutoff")
    chosen, groups = [], []
    start = 0
    while start < modes:
        stop = start+1
        while stop < len(eigenvalues):
            threshold = (tolerances.eigen_tie_atol
                         + tolerances.eigen_tie_rtol
                         * max(abs(eigenvalues[start]), abs(eigenvalues[stop])))
            if eigenvalues[stop]-eigenvalues[start] > threshold:
                break
            stop += 1
        complete_space = contrasts @ eigenvectors[:, start:stop]
        canonical = _canonical_eigenspace(complete_space, tolerances)
        retained = min(stop, modes)-start
        chosen.append(canonical[:, :retained])
        groups.append({"start": start, "stop": stop, "retained": retained,
                       "minimum_eigenvalue": float(eigenvalues[start]),
                       "maximum_eigenvalue": float(eigenvalues[stop-1])})
        start = stop
    result = np.column_stack(chosen)
    return result, {
        "nonconstant_rank": int(len(eigenvalues)),
        "rank_cutoff": cutoff,
        "eigenvalues": eigenvalues.tolist(),
        "retained_groups": groups,
        "selected_subspace_residual_max_abs": float(np.max(np.abs(
            laplacian @ result-result @ (result.T @ laplacian @ result)))),
    }


def build_local_anatomical_basis(coordinates, voxel_ids, *, tolerances,
                                 split_depth=3, modes_per_cell=2, support_size=208):
    """Build a centered local basis without changing the supplied voxel axis.

    `blocks` contains half-open column ranges. Every selected voxel
    belongs to exactly one leaf, including voxels with a zero retained-mode
    coefficient. No row is dropped or replaced. Deeper/shallower partitions
    and explicit per-cell capacity support future tight/broad variants; they
    do not supply Task-A identifiability or a complete model panel.
    """
    if not isinstance(tolerances, BasisTolerances):
        raise ValueError("explicit BasisTolerances are required")
    depth = _positive_integer(split_depth, "split_depth", allow_zero=True)
    modes = _positive_integer(modes_per_cell, "modes_per_cell")
    support_size = _positive_integer(support_size, "support_size")
    if support_size > 208:
        raise ValueError("selected support cannot expand the original 208 locations")
    raw_coordinates = np.asarray(coordinates)
    raw_ids = np.asarray(voxel_ids)
    if raw_coordinates.shape != (support_size, 3) or raw_coordinates.dtype.kind not in "fiu":
        raise ValueError("coordinates must use the complete selected support in mm")
    if raw_ids.shape != (support_size,) or raw_ids.dtype.kind not in "iu":
        raise ValueError("voxel_ids must be the exact ordered selected integer identities")
    if len(np.unique(raw_ids)) != support_size:
        raise ValueError("voxel identities must be unique on the fixed support")
    xyz = np.array(raw_coordinates, dtype=np.float64, copy=True)
    ids = np.array(raw_ids, copy=True)
    if not np.all(np.isfinite(xyz)):
        raise ValueError("coordinates must be finite")
    cells, splits = _partition(xyz, ids, depth)
    if any(len(rows) <= modes for rows in cells):
        raise ValueError("each fixed cell needs more voxels than nonconstant modes")
    basis = np.zeros((support_size, len(cells)*modes), dtype=np.float64)
    labels = np.full(support_size, -1, dtype=np.int64)
    blocks, cell_ids, cell_diagnostics = [], [], []
    for cell, rows in enumerate(cells):
        points = xyz[rows]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            distances = np.linalg.norm(points[:, None, :]-points[None, :, :], axis=2)
            pair_distances = distances[np.triu_indices(len(rows), k=1)]
            positive = pair_distances[pair_distances > 0]
            if not len(positive):
                raise ValueError("a cell has no strictly positive anatomical distance")
            sigma = float(np.median(positive))
            weights = np.exp(-0.5*(distances/sigma)**2)
            np.fill_diagonal(weights, 0)
            laplacian = np.diag(weights.sum(axis=1))-weights
        local, diagnostics = _low_modes(laplacian, modes, tolerances)
        start, stop = cell*modes, (cell+1)*modes
        basis[rows, start:stop] = local
        labels[rows] = cell
        blocks.append((start, stop))
        cell_ids.append(tuple(int(value) for value in ids[rows]))
        diagnostics.update({"cell": cell, "voxel_count": int(len(rows)),
                            "sigma_mm": sigma,
                            "diameter_mm": float(np.max(pair_distances))})
        cell_diagnostics.append(diagnostics)
    centering = float(np.max(np.abs(basis.sum(axis=0))))
    orthonormality = float(np.max(np.abs(basis.T@basis-np.eye(basis.shape[1]))))
    if centering > tolerances.numerical_atol or orthonormality > tolerances.numerical_atol:
        raise ArithmeticError("basis exceeds frozen centering/orthonormality tolerance")
    if np.any(labels < 0) or sum(len(rows) for rows in cells) != support_size:
        raise ArithmeticError("spatial partition changed the fixed support")
    outside = max(float(np.max(np.abs(basis[labels != cell, start:stop])))
                  if np.any(labels != cell) else 0.0
                  for cell, (start, stop) in enumerate(blocks))
    if outside != 0:
        raise ArithmeticError("basis leaks outside its declared spatial blocks")
    diameter = max(item["diameter_mm"] for item in cell_diagnostics)
    for array in (basis, xyz, labels):
        array.setflags(write=False)
    return LocalAnatomicalBasis(
        B=basis, coordinates=xyz, voxel_ids=tuple(int(value) for value in ids),
        cell_labels=labels, blocks=tuple(blocks), cell_voxel_ids=tuple(cell_ids),
        max_cell_diameter_mm=diameter, split_depth=depth, modes_per_cell=modes,
        tolerances=tolerances,
        diagnostics={"voxel_count": support_size, "basis_columns": basis.shape[1],
                     "centering_max_abs": centering,
                     "orthonormality_max_abs": orthonormality,
                     "outside_cell_support_max_abs": outside,
                     "splits": splits, "cells": cell_diagnostics},
    )
