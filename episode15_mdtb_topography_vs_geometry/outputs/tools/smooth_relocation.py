"""Prospective EP15 M2 pull fields and deterministic Task-A calibration.

The pull map phi(x)=x+u(x) queries the common source at each target output
location x. One field acts on every condition/partition. Global geometric
bounds are analytic; optimizer convergence is not a global-fit certificate.
No reader, target Task B, native half-stability or random-warp control here.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from common_cardinal_source import CommonCardinalSource
from development_lopo import M2Prediction, M2_VARIANTS
from local_functional_source import _array, _center, _frozen
from local_anatomical_basis import _partition
from registered_endpoint_metrics import crossvalidated_geometries


@dataclass(frozen=True)
class RelocationVariant:
    name: str
    width_multiplier: float
    displacement_mm: float


VARIANTS = tuple(RelocationVariant(name, width, displacement)
                 for name, width, displacement in zip(M2_VARIANTS, (1., 2., .5, 1.), (3., 3., 3., 6.)))


class BoundedPullField:
    """Eight smooth vector components with a global Frobenius-ball certificate.

    h_j is the variant multiplier times the maximum cell radius. Each native
    point's own-cell basis is at least exp(-1/2) for reference/stress,
    exp(-1/8) for tight, and exp(-2) for loose. Let
    a=sqrt(sum_j exp(-1)/h_j^2), R=min(d/sqrt(8), kappa/a).
    ||T||_F<=R implies sup||u||<=d and sup||Du||_2<=kappa=.25.
    Thus phi is a global bi-Lipschitz diffeomorphism, not merely positive at
    sampled Jacobians. Inverse iteration contracts on all of R^3.
    """

    def __init__(self, anatomy, variant):
        if variant not in VARIANTS:
            raise ValueError("one of the four selected relocation variants is required")
        self.variant = variant
        self.support_size = len(anatomy.voxel_ids)
        self.coordinates = _frozen(_array(anatomy.coordinates, (self.support_size, 3), "common coordinates"))
        labels = np.asarray(anatomy.cell_labels)
        cells, _ = _partition(self.coordinates, np.asarray(anatomy.voxel_ids), 3)
        expected_labels = np.empty(self.support_size, dtype=np.int64)
        for label, rows in enumerate(cells):
            expected_labels[rows] = label
        if (labels.shape != (self.support_size,) or labels.dtype.kind not in "iu"
                or not np.array_equal(labels, expected_labels)):
            raise ValueError("all eight deterministic median-split cells must retain every selected location")
        centers = np.array([self.coordinates[labels == label].mean(axis=0) for label in range(8)])
        radii = np.array([np.linalg.norm(self.coordinates[labels == label]-centers[label], axis=1).max()
                          for label in range(8)])
        if not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError("each anatomical cell must have positive finite radius")
        self.centers, self.widths = _frozen(centers), _frozen(variant.width_multiplier*radii)
        self.kappa, self.atol = .25, 1e-10
        self.derivative_factor = float(np.sqrt(np.sum(np.exp(-1)/self.widths**2)))
        self.radius = min(variant.displacement_mm/np.sqrt(8), self.kappa/self.derivative_factor)
        self.native_basis, _ = self.basis(self.coordinates)

    def basis(self, points):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("field points must be nonempty XYZ rows")
        points = _array(points, points.shape, "field coordinates")
        delta = points[:, None]-self.centers[None]
        values = np.exp(-np.sum(delta*delta, axis=-1)/(2*self.widths**2))
        gradients = -values[..., None]*delta/self.widths[None, :, None]**2
        return values, gradients

    def displacement(self, points, coefficients):
        T = _array(coefficients, (8, 3), "field coefficients in mm")
        values, gradients = self.basis(points)
        return values@T, np.einsum("ja,qjb->qab", T, gradients)

    def certificate(self, coefficients):
        T = _array(coefficients, (8, 3), "field coefficients in mm")
        norm = float(np.linalg.norm(T))
        displacement = float(np.sqrt(8)*norm)
        lipschitz = self.derivative_factor*norm
        if (norm > self.radius+self.atol or displacement > self.variant.displacement_mm+self.atol
                or lipschitz > self.kappa+self.atol or lipschitz >= 1):
            raise ValueError("field exceeds its global displacement/Lipschitz bounds")
        return {"coefficient_frobenius_norm_mm": norm, "coefficient_radius_mm": self.radius,
                "global_displacement_upper_bound_mm": displacement,
                "selected_displacement_cap_mm": self.variant.displacement_mm,
                "global_derivative_norm_upper_bound": lipschitz,
                "selected_derivative_cap": self.kappa,
                "jacobian_singular_value_bounds": (1-lipschitz, 1+lipschitz),
                "jacobian_determinant_bounds": ((1-lipschitz)**3, (1+lipschitz)**3),
                "global_diffeomorphism_certified": True,
                "field_convention": "target_output_to_source_query_pull"}

    def inverse(self, points, coefficients):
        certificate = self.certificate(coefficients)
        target = np.asarray(points)
        if target.ndim != 2 or target.shape[1] != 3 or len(target) == 0:
            raise ValueError("inverse query points must be nonempty XYZ rows")
        target = _array(target, target.shape, "inverse query coordinates")
        estimate = target.copy()
        contraction = certificate["global_derivative_norm_upper_bound"]
        for iteration in range(65):
            displacement, _ = self.displacement(estimate, coefficients)
            residual = estimate+displacement-target
            error_bound = float(np.linalg.norm(residual, axis=1).max()/(1-contraction))
            if error_bound <= self.atol:
                return estimate, {"inverse_iterations": iteration,
                                  "inverse_error_upper_bound_mm": error_bound}
            if iteration < 64:
                estimate = target-displacement
        raise ArithmeticError("contraction inverse failed its fixed residual error bound")


class RelocationTaskAObjective:
    """Full nonlinear cardinal-map objective, not a linearized warp estimator.

    z is the 24-vector of coefficients divided by the field's fixed radius.
    The data term uses mean voxels, equal29-A weights and source-A energy;
    the fixed penalty is 1e-3||z||². The ball and raw interpolation row-L1<=4
    constraints remain distinct. At zeros of a cardinal weight the latter is
    nonsmooth; sign(0)=0 is the prescribed subgradient for SLSQP. This solver
    choice carries no differentiability, global-minimum or uniqueness claim.
    """

    def __init__(self, representation, field, source_a, target_a, *, common_gain):
        self.representation, self.field = representation, field
        self.support_size = representation.support_size
        if not np.array_equal(representation.coordinates, field.coordinates):
            raise ValueError("field and shared representation must use identical ordered coordinates")
        self.source = _center(_array(source_a, (29, self.support_size), "source Task A"))
        self.target = _center(_array(target_a, (29, self.support_size), "target Task A"))
        gain = np.asarray(common_gain)
        if gain.shape != () or gain.dtype.kind not in "fiu" or not np.isfinite(gain) or gain <= 0:
            raise ValueError("one positive finite common gain is required")
        self.gain = float(gain)
        self.weights = np.full(29, 1/29)
        self.energy = float(np.mean(np.sum(self.weights[:, None]*self.source**2, axis=0)))
        if not np.isfinite(self.energy) or self.energy <= 0:
            raise ValueError("source-A mean-voxel energy must be positive")
        expected_gain = float(np.mean(np.sum(self.weights[:, None]*self.source*self.target, axis=0))/self.energy)
        if (not np.isfinite(expected_gain) or expected_gain <= 0
                or abs(self.gain-expected_gain) > 1e-10*max(1., abs(expected_gain))):
            raise ValueError("M2 must retain the common untransformed-source Task-A gain")
        self.target = self.target/self.gain
        if not np.isfinite(self.target).all():
            raise ArithmeticError("nonfinite common-gain calibrated target")
        self.penalty = 1e-3
        self._cached_z, self._cached = None, None

    def coefficients(self, z):
        return self.field.radius*_array(z, (24,), "normalized field coefficients").reshape(8, 3)

    def evaluation(self, z):
        z = _array(z, (24,), "normalized field coefficients")
        if self._cached_z is not None and np.array_equal(z, self._cached_z):
            return self._cached
        T = self.coefficients(z)
        queries = self.field.coordinates+self.field.native_basis@T
        W, dW = self.representation.weights_and_derivatives(queries)
        warped = self.source.copy() if not np.any(z) else self.source@W.T
        # Derivatives are evaluated even when the zero-field value is literal.
        spatial_gradient = np.einsum("ci,qid->cqd", self.source, dW)
        jacobian = (self.field.radius*np.einsum("cqd,qj->cqjd", spatial_gradient,
                                              self.field.native_basis)).reshape(29, self.support_size, 24)
        jacobian -= jacobian.mean(axis=1, keepdims=True)
        residual = _center(warped)-self.target
        value = float(np.sum(self.weights[:, None]*residual**2)/(self.support_size*self.energy)
                      +self.penalty*np.dot(z, z))
        gradient = (2*np.einsum("c,cq,cqp->p", self.weights, residual, jacobian)/(self.support_size*self.energy)
                    +2*self.penalty*z)
        # L1 subgradient is analytic away from zero weights, fixed at zero.
        l1_spatial_gradient = np.einsum("qi,qid->qd", np.sign(W), dW)
        constraint_jacobian = (-self.field.radius*np.einsum("qd,qj->qjd", l1_spatial_gradient,
                                                          self.field.native_basis)).reshape(self.support_size, 24)
        constraints = self.representation.amplification_cap-np.sum(np.abs(W), axis=1)
        if (not np.isfinite(value) or not np.isfinite(gradient).all()
                or not np.isfinite(jacobian).all() or not np.isfinite(constraint_jacobian).all()):
            raise ArithmeticError("nonfinite M2 objective or analytic derivative")
        result = {"value": value, "gradient": gradient, "jacobian": jacobian,
                  "interpolation_constraints": constraints, "interpolation_jacobian": constraint_jacobian,
                  "weights": W, "queries": queries}
        self._cached_z, self._cached = z.copy(), result
        return result

    def value_gradient(self, z):
        result = self.evaluation(z)
        return result["value"], result["gradient"]

    def tangent_singular_values(self, z):
        jacobian = self.evaluation(z)["jacobian"]
        weighted = (np.sqrt(self.weights/(self.support_size*self.energy))[:, None, None]*jacobian).reshape(-1, 24)
        return np.linalg.svd(weighted, compute_uv=False)


def fit_relocation_task_a(representation, field, source_a, target_a, *, common_gain):
    """One zero-start, 500-iteration SLSQP attempt with the selected fixed recipe."""
    objective = RelocationTaskAObjective(representation, field, source_a, target_a, common_gain=common_gain)
    zero = np.zeros(24)
    baseline = objective.value_gradient(zero)[0]
    constraints = [
        {"type": "ineq", "fun": lambda z: 1-np.dot(z, z), "jac": lambda z: -2*z},
        {"type": "ineq", "fun": lambda z: objective.evaluation(z)["interpolation_constraints"],
         "jac": lambda z: objective.evaluation(z)["interpolation_jacobian"]},
    ]
    fitted = minimize(objective.value_gradient, zero, jac=True, method="SLSQP", constraints=constraints,
                      options={"maxiter": 500, "ftol": 1e-12, "disp": False})
    if not fitted.success:
        raise ArithmeticError("M2 SLSQP failed; no retry, returned field or reduced-support rescue")
    z = _array(fitted.x, (24,), "fitted normalized field coefficients")
    actual = objective.evaluation(z)
    if np.dot(z, z) > 1+1e-10 or actual["value"] > baseline+1e-10:
        raise ArithmeticError("M2 output violates its ball or identity-baseline objective")
    T = objective.coefficients(z)
    field_certificate = field.certificate(T)
    interpolation_certificate = representation.certificate(actual["weights"])
    # Rank of the unpenalized map derivative is a local numerical diagnostic.
    # It is not an added M2 admission gate or a general identification claim.
    singular = objective.tangent_singular_values(z)
    if not np.isfinite(singular).all():
        raise ArithmeticError("nonfinite local tangent diagnostic")
    cutoff = 1e-8*singular[0]
    rank = int(np.sum(singular > cutoff))
    inverse, inverse_certificate = field.inverse(actual["queries"], T)
    inverse_error = float(np.linalg.norm(inverse-field.coordinates, axis=1).max())
    if inverse_error > 2e-10:
        raise ArithmeticError("M2 pull/inverse composition failed")
    return {"coefficients": _frozen(T), "common_gain": objective.gain,
            "normalized_penalized_task_a_loss": actual["value"],
            "identity_task_a_loss": baseline, "optimizer_iterations": int(fitted.nit),
            "local_tangent_singular_values": _frozen(singular),
            "local_tangent_rank_diagnostic": rank, "local_tangent_rank_cutoff": float(cutoff),
            "local_tangent_full_rank_diagnostic": rank == 24,
            "general_data_identifiability_certified": False,
            "global_solution_certified": False, "unique_solution_certified": False,
            "field_certificate": field_certificate, "interpolation_certificate": interpolation_certificate,
            "inverse_certificate": inverse_certificate, "inverse_composition_max_mm": inverse_error}


class SmoothRelocationPrediction:
    """All four selected M2 variants, sharing source/support/gain/interpolation."""

    def __init__(self, functional_source, anatomy):
        if (anatomy.voxel_ids != functional_source.voxel_ids
                or not np.array_equal(anatomy.B, functional_source.B)
                or not np.array_equal(anatomy.cell_labels, functional_source.labels)):
            raise ValueError("M2 must retain the selected functional source's exact anatomy")
        self.source_recipe = functional_source
        self.support_size = functional_source.support_size
        self.representation = CommonCardinalSource(anatomy.coordinates, anatomy.voxel_ids,
                                                  support_size=self.support_size)
        self.fields = tuple(BoundedPullField(anatomy, variant) for variant in VARIANTS)

    def predict_partitions(self, field, source_partitions, coefficients, common_gain):
        """No fitting: apply a certified pull to the same four source B maps."""
        if field not in self.fields:
            raise ValueError("prediction must use one of the fixed field variants")
        field.certificate(coefficients)
        partitions = _center(_array(source_partitions, (4, 18, self.support_size), "source B partitions"))
        gain = np.asarray(common_gain)
        if gain.shape != () or gain.dtype.kind not in "fiu" or not np.isfinite(gain) or gain <= 0:
            raise ValueError("one positive finite common gain is required")
        if not np.any(coefficients):
            return _center(float(gain)*partitions)
        displacement, _ = field.displacement(field.coordinates, coefficients)
        prediction, _ = self.representation.evaluate(partitions.reshape(72, self.support_size),
                                                     field.coordinates+displacement)
        return _center(float(gain)*prediction.reshape(4, 18, self.support_size))

    def __call__(self, source, target_task_a, common_gain, context):
        self.source_recipe._source(source, context)
        results = []
        for field in self.fields:
            fitted = fit_relocation_task_a(self.representation, field, source.task_a_maps,
                                          target_task_a, common_gain=common_gain)
            partitions = self.predict_partitions(field, source.geometry_state.source_partitions,
                                                 fitted["coefficients"], common_gain)
            diagnostics = dict(fitted, training_ids=context.training_ids,
                variant=field.variant.name, field_widths_mm=field.widths,
                native_signature_certified=False, matched_random_warp_available=False,
                half_field_stability_available=False, class_adequacy_claimed=False,
                empirical_execution_authorized=False)
            results.append(M2Prediction(field.variant.name, _frozen(partitions.mean(axis=0)),
                _frozen(crossvalidated_geometries(partitions[None])[0].mean(axis=0)), diagnostics))
        return tuple(results)
