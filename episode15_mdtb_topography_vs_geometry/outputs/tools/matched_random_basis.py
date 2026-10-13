"""M3 random local subspaces with the same maximum Rayleigh-energy ceiling.

The matched quantity is a bound on graph energy of a unit personalization
increment, not its average energy, full spectrum, or total output-map energy.
The constrained perturbations are not uniform draws over local subspaces.
Anatomy-only draws are shared across all participants and nested deletions.
"""
from dataclasses import replace
from typing import Callable

import numpy as np

from development_lopo import RANDOM_BASIS_DRAWS, RandomBasisPrediction
from local_anatomical_basis import _canonical_eigenspace
from local_functional_source import _array, _center, _frozen
from local_isometry_variants import BlockLocalIsometry, VARIANTS
from registered_endpoint_metrics import crossvalidated_geometries
from spatial_deformation_operators import apply_m3


ALPHA = .5
MASTER_SEED = 20261007
STREAM_TAG = 1503


class RandomBasisUnresolved(ArithmeticError):
    """No complete certified control; diagnostics are not a partial prediction."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _prepare_cell(laplacian, reference_basis, *, cell, tolerances):
    """Prepare actual local Ritz geometry, including canonical near-tie residuals."""
    B = np.asarray(reference_basis)
    if B.ndim != 2 or B.shape[1] != 2 or B.shape[0] < 4:
        raise ValueError("a centered local two-column reference basis is required")
    B = _array(B, B.shape, "local reference basis")
    L = _array(laplacian, (len(B), len(B)), "fixed local Laplacian")
    atol = tolerances.numerical_atol
    if (np.max(np.abs(L-L.T)) > atol or np.max(np.abs(B.T@B-np.eye(2))) > atol
            or np.max(np.abs(B.sum(axis=0))) > atol):
        raise ValueError("local reference graph/basis fails its existing numerical certificate")
    H = B.T@L@B
    energies, rotation = np.linalg.eigh((H+H.T)/2)
    r = B@rotation
    for column in range(2):
        pivots = np.flatnonzero(np.abs(r[:, column]) > tolerances.rank_rtol)
        if not len(pivots):
            raise RandomBasisUnresolved("local Ritz direction has no stable native-ID sign", {"cell": cell})
        if r[pivots[0], column] < 0:
            r[:, column] *= -1
    a, b = map(float, energies)
    energy_tolerance = tolerances.eigen_tie_atol+tolerances.eigen_tie_rtol*max(abs(a), abs(b))
    if not np.isfinite(energies).all() or a <= 0:
        raise RandomBasisUnresolved("local reference has no positive finite Rayleigh scale", {"cell": cell})
    # Column normalization avoids units dominating this dimensionless rank
    # decision. Do not normalize tiny residual vectors into extra constraints.
    constraints = np.column_stack((np.ones(len(B)), B, L@B))
    norms = np.linalg.norm(constraints, axis=0)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise RandomBasisUnresolved("local orthogonality constraints are invalid", {"cell": cell})
    left, singular, _ = np.linalg.svd(constraints/norms, full_matrices=True)
    rank = int(np.sum(singular > tolerances.rank_rtol*singular[0]))
    Z = left[:, rank:]
    if not Z.shape[1]:
        raise RandomBasisUnresolved("no complementary local direction remains", {"cell": cell})
    reduced = Z.T@L@Z
    values, vectors = np.linalg.eigh((reduced+reduced.T)/2)
    # Preserve complete numerical eigenspaces. Pick the lowest whole group
    # whose lower energy is at least b up to the existing eigenvalue tolerance.
    eligible = None
    start = 0
    while start < len(values):
        stop = start+1
        while stop < len(values):
            tie = tolerances.eigen_tie_atol+tolerances.eigen_tie_rtol*max(abs(values[start]), abs(values[stop]))
            if values[stop]-values[start] > tie:
                break
            stop += 1
        if values[start] >= b-energy_tolerance:
            eligible = _canonical_eigenspace(Z@vectors[:, start:stop], tolerances)
            break
        start = stop
    if eligible is None:
        raise RandomBasisUnresolved("no eligible complementary Ritz eigenspace", {"cell": cell, "ceiling": b})
    return {"cell": cell, "L": L, "B": B, "r": r, "a": a, "b": b,
            "energy_tolerance": energy_tolerance, "atol": atol,
            "reference_tied": b-a <= energy_tolerance, "eligible": eligible,
            "constraint_rank": rank, "eligible_ritz_range": (float(values[start]), float(values[stop-1])),
            "rank_rtol": tolerances.rank_rtol}


def _draw_cell(prepared, draw_index):
    cell, L, B, r = (prepared[name] for name in ("cell", "L", "B", "r"))
    a, b, atol = (prepared[name] for name in ("a", "b", "atol"))
    seed = (MASTER_SEED, cell, draw_index, STREAM_TAG)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    weights = rng.standard_normal(prepared["eligible"].shape[1])
    norm = float(np.linalg.norm(weights))
    if not np.isfinite(norm) or norm <= prepared["rank_rtol"]:
        raise RandomBasisUnresolved("degenerate fixed random direction; no resampling", {"cell": cell, "draw": draw_index})
    z = prepared["eligible"]@(weights/norm)
    c = float(z@L@z)
    if not np.isfinite(c) or c < b-prepared["energy_tolerance"]:
        raise RandomBasisUnresolved("selected direction violates its complementary energy requirement",
                                    {"cell": cell, "draw": draw_index, "a": a, "b": b, "c": c})
    tied = prepared["reference_tied"]
    if tied:
        # This branch is accepted only by the strict actual-matrix, trace and
        # ceiling checks below. Numerical grouping is not proof of a true tie.
        if max(abs(b-a), abs(c-b)) > atol:
            raise RandomBasisUnresolved("reference tie has no certified boundary-tied complementary direction",
                                        {"cell": cell, "draw": draw_index, "a": a, "b": b, "c": c})
        squared_sine = ALPHA
    else:
        squared_sine = ALPHA*(b-a)/(c-a)
    if not np.isfinite(squared_sine) or not 0 < squared_sine < 1:
        raise RandomBasisUnresolved("no finite nonidentity Rayleigh-matched rotation",
                                    {"cell": cell, "draw": draw_index})
    mixed = np.column_stack((np.sqrt(1-squared_sine)*r[:, 0]+np.sqrt(squared_sine)*z, r[:, 1]))
    actual_H = mixed.T@L@mixed
    actual_values = np.linalg.eigvalsh((actual_H+actual_H.T)/2)
    expected = np.diag([a if tied else a+ALPHA*(b-a), b])
    trace_increase = float(np.trace(actual_H)-np.trace(B.T@L@B))
    expected_trace_increase = 0. if tied else ALPHA*(b-a)
    residuals = {"centering_max_abs": float(np.max(np.abs(mixed.sum(axis=0)))),
                 "orthonormality_max_abs": float(np.max(np.abs(mixed.T@mixed-np.eye(2)))),
                 "energy_matrix_matching_max_abs": float(np.max(np.abs(actual_H-expected))),
                 "maximum_rayleigh_matching_abs": abs(float(actual_values[-1])-b),
                 "trace_formula_error_abs": abs(trace_increase-expected_trace_increase)}
    distance = float(np.linalg.norm(mixed@mixed.T-B@B.T))
    principal_cosines = np.linalg.svd(B.T@mixed, compute_uv=False)
    principal_sine = np.linalg.svd(mixed-B@(B.T@mixed), compute_uv=False)[0]
    # atan2 keeps very small nonzero changes visible; acos rounds them to zero.
    angle = float(np.arctan2(np.clip(principal_sine, 0., 1.),
                            np.clip(principal_cosines[-1], 0., 1.)))
    if any(not np.isfinite(value) or value > atol for value in residuals.values()):
        raise RandomBasisUnresolved("actual local Rayleigh/algebra certificate failed; no weaker bound",
                                    dict(residuals, cell=cell, draw=draw_index))
    if distance <= atol:
        raise RandomBasisUnresolved("numerically degenerate unchanged local subspace",
                                    {"cell": cell, "draw": draw_index, "projector_distance": distance})
    return _frozen(mixed), dict(residuals, cell=cell, draw_index=draw_index, seed_entropy=seed,
        reference_ritz_energies=(a, b), complementary_rayleigh_energy=c,
        mixed_ritz_energies=tuple(map(float, actual_values)),
        trace_increase=trace_increase, expected_trace_increase=expected_trace_increase,
        projector_frobenius_distance=distance, maximum_principal_angle_radians=angle,
        squared_sine=float(squared_sine), alpha=ALPHA,
        numerical_boundary_tie_branch=tied, exact_eigenvalue_tie_proven=False,
        degenerate_control=False, scientific_effectiveness_certified=False,
        eligible_ritz_range=prepared["eligible_ritz_range"], constraint_rank=prepared["constraint_rank"])


class MatchedRandomLocalBasis:
    """Eight complete anatomy-only controls, with target-only M3 refitting.

    No functional source is refitted. Constructor work belongs in the caller's
    Slurm allocation, as do all fits. All eight draws are required; failures
    propagate without resampling, participant removal or a partial control.
    """

    def __init__(self, reference_source_recipe, reference_anatomy, *, check: Callable):
        if not callable(check):
            raise TypeError("an allocation/deadline callback is required")
        if (reference_anatomy.voxel_ids != reference_source_recipe.voxel_ids
                or not np.array_equal(reference_anatomy.B, reference_source_recipe.B)
                or not np.array_equal(reference_anatomy.cell_labels, reference_source_recipe.labels)):
            raise ValueError("random controls must retain the exact existing reference source anatomy")
        check()
        BlockLocalIsometry(reference_anatomy, VARIANTS[0])
        self.source_recipe, self.check = reference_source_recipe, check
        self.support_size = reference_source_recipe.support_size
        self.reference_anatomy = reference_anatomy
        prepared, cell_rows = [], []
        ids = np.asarray(reference_anatomy.voxel_ids)
        for cell, (start, stop) in enumerate(reference_anatomy.blocks):
            check()
            rows = np.flatnonzero(reference_anatomy.cell_labels == cell)
            rows = rows[np.argsort(ids[rows], kind="stable")]
            points = reference_anatomy.coordinates[rows]
            distances = np.linalg.norm(points[:, None]-points[None], axis=-1)
            pair = distances[np.triu_indices(len(rows), k=1)]
            sigma = float(np.median(pair[pair > 0]))
            weights = np.exp(-.5*(distances/sigma)**2)
            np.fill_diagonal(weights, 0.)
            L = np.diag(weights.sum(axis=1))-weights
            prepared.append(_prepare_cell(L, reference_anatomy.B[rows, start:stop],
                                           cell=cell, tolerances=reference_anatomy.tolerances))
            cell_rows.append(rows)
        operators, diagnostics = [], []
        for draw in range(RANDOM_BASIS_DRAWS):
            B = np.zeros_like(reference_anatomy.B)
            cells = []
            for cell, (start, stop) in enumerate(reference_anatomy.blocks):
                check()
                local, report = _draw_cell(prepared[cell], draw)
                B[cell_rows[cell], start:stop] = local
                cells.append(report)
            randomized = replace(reference_anatomy, B=_frozen(B),
                diagnostics={"construction": "maximum_rayleigh_matched_random_local_basis", "cells": tuple(cells)})
            operators.append(BlockLocalIsometry(randomized, VARIANTS[0]))
            diagnostics.append(tuple(cells))
        self.operators, self.cell_diagnostics = tuple(operators), tuple(diagnostics)
        self.check()

    def __call__(self, source, target_task_a, reference_numerical_fit, context):
        self.source_recipe._source(source, context)
        self.check()
        self.source_recipe.certify_m3(source, reference_numerical_fit, context)
        X = _center(_array(source.task_a_maps, (29, self.support_size), "unchanged source Task A"))
        Y = _center(_array(target_task_a, (29, self.support_size), "target Task A"))
        energy = float(np.sum(self.source_recipe.weights[:, None]*X**2))
        expected_gain = float(np.sum(self.source_recipe.weights[:, None]*X*Y)/energy) if energy > 0 else np.nan
        gain = float(reference_numerical_fit["gain"])
        if (not np.isfinite(expected_gain) or expected_gain <= 0
                or abs(gain-expected_gain) > 1e-10*max(1., abs(expected_gain))):
            raise ValueError("random controls must retain the common pre-operator Task-A gain")
        partitions = _array(source.geometry_state.source_partitions, (4, 18, self.support_size),
                            "unchanged source B partitions")
        results = []
        for draw, operator in enumerate(self.operators):
            self.check()
            fitted = operator.fit(X, Y, common_gain=gain)
            predicted = gain*apply_m3(partitions.reshape(72, self.support_size), operator.B,
                                      fitted["O"]).reshape(4, 18, self.support_size)
            maps = gain*apply_m3(_center(source.G), operator.B, fitted["O"])
            geometry = crossvalidated_geometries(predicted[None])[0].mean(axis=0)
            diagnostics = dict(fitted, B=operator.B, cells=self.cell_diagnostics[draw],
                draw_index=draw, common_gain=gain, voxel_ids=source.voxel_ids,
                training_ids=context.training_ids, target_id=context.target_id, deleted_id=context.deleted_id,
                source_binding="existing_reference_shared_G_A_four_G_B_partitions_and_K0",
                source_refitted=False, draws_shared_across_participants_and_deletions=True,
                matched_quantity="maximum_graph_Rayleigh_energy_of_local_personalization_increment",
                full_spectrum_matched=False, average_roughness_matched=False,
                uniform_random_subspace=False, monte_carlo_p_value_available=False,
                native_signature_certified=False, M4_output_smoothness_signature_certified=False,
                class_adequacy_claimed=False, empirical_execution_authorized=False)
            results.append(RandomBasisPrediction(draw, _frozen(maps), _frozen(geometry), diagnostics))
        self.check()
        return tuple(results)
