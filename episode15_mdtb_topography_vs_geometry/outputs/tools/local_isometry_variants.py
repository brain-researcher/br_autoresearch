"""Four prospective M3 target operators on one unchanged functional source.

Only anatomy-defined target operator bases vary. G_A, all four G_B partitions,
K0, native coordinates and the positive pre-operator gain remain common. Tight
locality has sixteen discrete sign factors and no continuous degrees of freedom;
these four variants are not a capacity-matched test of locality alone.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from development_lopo import M3Prediction, M3_VARIANTS
from local_anatomical_basis import BasisTolerances, build_local_anatomical_basis
from local_functional_source import _array, _center, _frozen, _full_rank
from registered_endpoint_metrics import crossvalidated_geometries
from spatial_deformation_operators import apply_m3


@dataclass(frozen=True)
class IsometryVariant:
    name: str
    split_depth: int
    modes_per_cell: int


VARIANTS = tuple(IsometryVariant(name, depth, modes)
                 for name, depth, modes in zip(M3_VARIANTS, (3, 4, 2, 3), (2, 1, 4, 3)))


class BlockLocalIsometry:
    """Certified block-orthogonal target operator; never a shared source fitter."""

    def __init__(self, anatomy, variant):
        if (variant not in VARIANTS or anatomy.split_depth != variant.split_depth
                or anatomy.modes_per_cell != variant.modes_per_cell):
            raise ValueError("the fixed variant and anatomical basis must agree")
        tolerances = BasisTolerances(1e-10, 1e-10, 1e-8, 1e-10)
        if anatomy.tolerances != tolerances:
            raise ValueError("the four target bases retain the reference numerical tolerances")
        self.variant, self.anatomy = variant, anatomy
        self.support_size = len(anatomy.voxel_ids)
        self.dimension = (2**variant.split_depth)*variant.modes_per_cell
        if self.dimension > 29:
            raise ValueError("target active dimension exceeds the 29-condition global design-rank bound")
        self.B = _frozen(_array(anatomy.B, (self.support_size, self.dimension), "target operator basis"))
        self.coordinates = _frozen(_array(anatomy.coordinates, (self.support_size, 3), "common coordinates"))
        self.blocks = tuple((j*variant.modes_per_cell, (j+1)*variant.modes_per_cell)
                            for j in range(2**variant.split_depth))
        if anatomy.blocks != self.blocks:
            raise ValueError("target operator must retain all fixed anatomical blocks")
        self.labels = np.asarray(anatomy.cell_labels)
        if (self.labels.shape != (self.support_size,) or self.labels.dtype.kind not in "iu"
                or set(self.labels) != set(range(len(self.blocks)))):
            raise ValueError("every common location must retain its fixed anatomical cell")
        self.atol, self.rank_rtol = tolerances.numerical_atol, tolerances.rank_rtol
        self.weights = np.full(29, 1/29)
        self.diameter = float(anatomy.max_cell_diameter_mm)
        actual = 0.
        for cell, (start, stop) in enumerate(self.blocks):
            member = self.labels == cell
            if np.max(np.abs(self.B[~member, start:stop])) > self.atol:
                raise ValueError("target basis has coefficients outside its anatomical block")
            points = self.coordinates[member]
            actual = max(actual, float(np.linalg.norm(points[:, None]-points[None], axis=-1).max()))
        if not np.isfinite(self.diameter) or self.diameter <= 0 or actual > self.diameter+self.atol:
            raise ValueError("target anatomical diameter bound is invalid")
        self.continuous_degrees_of_freedom = sum((stop-start)*(stop-start-1)//2
                                                for start, stop in self.blocks)
        self.operator_certificate(np.eye(self.dimension))

    def operator_certificate(self, O):
        O = _array(O, (self.dimension, self.dimension), "target O")
        allowed = np.zeros(O.shape, dtype=bool)
        for start, stop in self.blocks:
            allowed[start:stop, start:stop] = True
        identity = np.eye(self.support_size)
        Q = identity+self.B@(O-np.eye(self.dimension))@self.B.T
        complement = identity-self.B@self.B.T
        diagnostic = {
            "basis_orthonormality_max_abs": float(np.max(np.abs(self.B.T@self.B-np.eye(self.dimension)))),
            "basis_centering_max_abs": float(np.max(np.abs(self.B.sum(axis=0)))),
            "O_cross_block_max_abs": float(np.max(np.abs(O[~allowed]))),
            "O_orthogonality_max_abs": float(max(np.max(np.abs(O.T@O-np.eye(self.dimension))),
                                                np.max(np.abs(O@O.T-np.eye(self.dimension))))),
            "Q_orthogonality_max_abs": float(max(np.max(np.abs(Q.T@Q-identity)),
                                                np.max(np.abs(Q@Q.T-identity)))),
            "Q_constant_preservation_max_abs": float(np.max(np.abs(Q@np.ones(self.support_size)-1))),
            "Q_identity_complement_max_abs": float(np.max(np.abs((Q-identity)@complement))),
            "Q_cross_cell_max_abs": float(np.max(np.abs(Q[self.labels[:, None] != self.labels[None, :]]))),
        }
        if any(not np.isfinite(value) or value > self.atol for value in diagnostic.values()):
            raise ValueError("target M3 operator fails fixed algebra/locality certification")
        diagnostic.update(max_transport_distance_bound_mm=self.diameter,
            active_dimension=self.dimension, continuous_degrees_of_freedom=self.continuous_degrees_of_freedom,
            determinant_sign_factors=len(self.blocks), gain_outside_isometry=True)
        return diagnostic

    def fit(self, source_task_a, target_task_a, *, common_gain):
        """Unique block polar solution, with the existing global-scale rank rule."""
        source = _center(_array(source_task_a, (29, self.support_size), "common source Task A"))
        target = _center(_array(target_task_a, (29, self.support_size), "target Task A"))
        gain = np.asarray(common_gain)
        if gain.shape != () or gain.dtype.kind not in "fiu" or not np.isfinite(gain) or gain <= 0:
            raise ValueError("one positive finite common gain is required")
        active_source = np.sqrt(self.weights)[:, None]*(source@self.B)
        active_target = np.sqrt(self.weights)[:, None]*(target@self.B)
        singular = np.linalg.svd(active_source, compute_uv=False)
        _full_rank(singular, self.dimension, self.rank_rtol, "global Task-A active source design")
        cross = active_source.T@active_target
        cross_singular = np.linalg.svd(cross, compute_uv=False)
        cutoff = self.rank_rtol*cross_singular[0]
        O = np.zeros((self.dimension, self.dimension))
        local = []
        for start, stop in self.blocks:
            left, values, right_t = np.linalg.svd(cross[start:stop, start:stop], full_matrices=False)
            if not np.isfinite(values).all() or values[-1] <= cutoff:
                raise ValueError("local Task-A cross-product is not full rank at global objective scale")
            O[start:stop, start:stop] = left@right_t
            local.append(_frozen(values))
        certificate = self.operator_certificate(O)
        residual = float(gain)*apply_m3(source, self.B, O)-target
        loss = float(np.sum(self.weights[:, None]*residual**2))
        if not np.isfinite(loss):
            raise ArithmeticError("nonfinite target M3 objective")
        return {"O": _frozen(O), "gain": float(gain), "weighted_task_a_squared_error": loss,
                "source_design_singular_values": _frozen(singular),
                "global_cross_singular_values_diagnostic_only": _frozen(cross_singular),
                "local_cross_rank_cutoff": float(cutoff),
                "local_cross_rank_scale": "global_cross_product_spectral_norm",
                "local_cross_singular_values": tuple(local),
                "operator_certificate": certificate, "unique_within_fixed_block_family": True}


class LocalIsometryVariants:
    """All four target variants, reusing the actual reference source and fit.

    Callback: (source, target_task_a, reference_numerical_fit, context).
    No source fitter is called here. No target B is accepted. Returned geometry
    comes from transforming the unchanged four source partitions, not from
    replacing K0 by the Gram matrix of a mean map.
    """

    def __init__(self, reference_source_recipe, reference_anatomy, *, check: Callable):
        if not callable(check):
            raise TypeError("an allocation/deadline callback is required")
        if (reference_anatomy.voxel_ids != reference_source_recipe.voxel_ids
                or not np.array_equal(reference_anatomy.B, reference_source_recipe.B)
                or not np.array_equal(reference_anatomy.cell_labels, reference_source_recipe.labels)
                or reference_anatomy.split_depth != 3 or reference_anatomy.modes_per_cell != 2):
            raise ValueError("all variants must retain the exact existing reference source anatomy")
        self.source_recipe, self.check = reference_source_recipe, check
        self.support_size = reference_source_recipe.support_size
        bases = [reference_anatomy]
        for variant in VARIANTS[1:]:
            check()
            bases.append(build_local_anatomical_basis(reference_anatomy.coordinates,
                np.asarray(reference_anatomy.voxel_ids), tolerances=reference_anatomy.tolerances,
                split_depth=variant.split_depth, modes_per_cell=variant.modes_per_cell,
                support_size=self.support_size))
        self.operators = tuple(BlockLocalIsometry(anatomy, variant)
                               for anatomy, variant in zip(bases, VARIANTS))
        self.check()

    def __call__(self, source, target_task_a, reference_numerical_fit, context):
        self.source_recipe._source(source, context)
        self.check()
        self.source_recipe.certify_m3(source, reference_numerical_fit, context)
        X = _center(_array(source.task_a_maps, (29, self.support_size), "common source Task A"))
        Y = _center(_array(target_task_a, (29, self.support_size), "target Task A"))
        energy = float(np.sum(self.source_recipe.weights[:, None]*X**2))
        expected_gain = float(np.sum(self.source_recipe.weights[:, None]*X*Y)/energy) if energy > 0 else np.nan
        gain = float(reference_numerical_fit["gain"])
        if (not np.isfinite(expected_gain) or expected_gain <= 0
                or abs(gain-expected_gain) > 1e-10*max(1., abs(expected_gain))):
            raise ValueError("every M3 variant must retain the common pre-operator Task-A gain")
        partitions = _array(source.geometry_state.source_partitions, (4, 18, self.support_size),
                            "unchanged common source B partitions")
        results = []
        for index, operator in enumerate(self.operators):
            self.check()
            fitted = (dict(reference_numerical_fit) if index == 0
                      else operator.fit(X, Y, common_gain=gain))
            certificate = operator.operator_certificate(fitted["O"])
            predicted = gain*apply_m3(partitions.reshape(72, self.support_size), operator.B,
                                      fitted["O"]).reshape(4, 18, self.support_size)
            # Parent map prediction uses source.G directly. Preserve that
            # arithmetic for the reference, rather than recomputing its mean.
            maps = gain*apply_m3(_center(source.G), operator.B, fitted["O"])
            geometry = crossvalidated_geometries(predicted[None])[0].mean(axis=0)
            diagnostics = dict(fitted, variant=operator.variant.name, B=operator.B,
                operator_certificate=certificate, common_gain=gain,
                voxel_ids=source.voxel_ids, training_ids=context.training_ids,
                target_id=context.target_id, deleted_id=context.deleted_id,
                source_binding="existing_reference_shared_G_A_four_G_B_partitions_and_K0",
                source_basis=source.B, source_refitted_for_variant=False,
                reference_fit_reused=index == 0,
                native_signature_certified=False, class_adequacy_claimed=False,
                empirical_execution_authorized=False)
            results.append(M3Prediction(operator.variant.name, _frozen(maps), _frozen(geometry), diagnostics))
        self.check()
        return tuple(results)
