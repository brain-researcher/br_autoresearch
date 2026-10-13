"""Four selected, array-only M4 variants paired to the reference local M3.

No reader, empirical access, source refit, Task-B calibration or optimizer
search is supplied. The original M4 objective/solver remains unchanged. Its
local result must additionally pass a sufficient curvature bound and an exact
linear-support optimality-gap calculation at fixed floating-point tolerances.
These are numerical acceptance certificates, not interval-arithmetic proofs
or a claim that the returned floating-point optimizer is exactly global.

All families have rank at most 16 in the fixed selected location space. Local rank
is at most two per cell; "low rank" does not mean a rank reduction relative
to the anisotropic variant. Parameter-count bounds are not estimated
statistical effective degrees of freedom.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from development_lopo import M4Prediction, M4_VARIANTS
from fit_m4_task_a import PairedM4TaskAObjective, fit_m4_task_a
from local_functional_source import LocalFunctionalSource, _array, _center, _frozen
from registered_endpoint_metrics import crossvalidated_geometries
from spatial_deformation_operators import apply_m3, apply_m4


NUMERICAL_ATOL = 1e-10
RANK_RTOL = 1e-8
CURVATURE_MARGIN = 1e-10
GAP_RTOL = 1e-10
OPTIMIZER_FTOL = 1e-12
MAX_ITERATIONS = 500
REFERENCE_RHO = float(np.log(1.25))
STRESS_RHO = float(np.log(1.5))
BLOCKS = tuple((2*k, 2*k+2) for k in range(8))


class M4CertificateUnverified(ValueError):
    """A sufficient certificate failed; non-identifiability is not inferred."""

    def __init__(self, reason, diagnostics):
        super().__init__(reason)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class StretchSpecification:
    name: str
    modes: np.ndarray
    spectral_norm_bound: float

    @property
    def parameter_count(self):
        return len(self.modes)


def stretch_specifications():
    """Canonical, Frobenius-orthonormal modes in fixed anatomical order."""
    anisotropic = np.zeros((15, 16, 16), dtype=np.float64)
    for k in range(1, 16):
        diagonal = np.zeros(16)
        diagonal[:k] = 1/np.sqrt(k*(k+1))
        diagonal[k] = -k/np.sqrt(k*(k+1))
        anisotropic[k-1] = np.diag(diagonal)
    diagonal, shear = [], []
    for start, stop in BLOCKS:
        D, H = np.zeros((16, 16)), np.zeros((16, 16))
        D[start:stop, start:stop] = np.diag([1., -1.])/np.sqrt(2.)
        H[start:stop, start:stop] = np.array([[0., 1.], [1., 0.]])/np.sqrt(2.)
        diagonal.append(D)
        shear.extend((D, H))
    families = (anisotropic, np.stack(diagonal), np.stack(shear), np.stack(shear))
    bounds = (REFERENCE_RHO, REFERENCE_RHO, REFERENCE_RHO, STRESS_RHO)
    return tuple(StretchSpecification(name, _frozen(modes), rho)
                 for name, modes, rho in zip(M4_VARIANTS, families, bounds))


def block_curvature_bounds(A, Y, *, global_gain, spectral_norm_bound):
    """Two global Hessian lower bounds for one weighted two-mode block.

    A=sqrt(w)*(center(G_A)@B@O)[:, block], Y=sqrt(w)*(center(Y_A)@B)[:, block].
    Bounds use Frobenius-normalized symmetric directions, not eigenvalue
    samples. Taking the larger of two valid lower bounds remains valid.
    """
    raw = np.asarray(A)
    if raw.ndim != 2 or raw.shape[1] != 2 or not len(raw):
        raise ValueError("weighted source block must have two columns")
    A = _array(raw, raw.shape, "weighted source block")
    Y = _array(Y, A.shape, "weighted target block")
    gain, rho = float(global_gain), float(spectral_norm_bound)
    if not np.isfinite([gain, rho]).all() or gain <= 0 or rho <= 0:
        raise ValueError("positive finite gain and spectral bound are required")
    singular = np.linalg.svd(A, compute_uv=False)
    P = gain**2*(A.T@A)
    cross = A.T@Y
    D = gain*(cross+cross.T)/2
    p, d = float(np.trace(P)/2), float(np.trace(D)/2)
    scale = float(np.sum((gain*A)**2)+np.sum(Y**2))
    if p <= 0 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("positive finite block source energy is required")
    upper, lower = np.exp(rho), np.exp(-rho)
    residual_bound = (np.linalg.norm(gain*A-Y, "fro")
                      +gain*singular[0]*np.sqrt(2.)*(upper-1))
    old = 2*(gain**2*singular[-1]**2*lower**2
             -gain*singular[0]*upper*residual_bound)
    vertex = min(upper, max(lower, d/(4*p)))
    isotropic = 4*p*vertex**2-2*d*vertex
    refined = (isotropic-4*np.linalg.norm(P-p*np.eye(2), "fro")*upper**2
               -2*np.linalg.norm(D-d*np.eye(2), "fro")*upper)
    result = {"residual_hessian_lower_bound": float(old),
              "refined_hessian_lower_bound": float(refined),
              "hessian_lower_bound": float(max(old, refined)),
              "isotropic_hessian_lower_bound": float(isotropic),
              "objective_scale": scale,
              "required_positive_margin": CURVATURE_MARGIN*scale}
    if not np.isfinite(tuple(result.values())).all():
        raise ArithmeticError("nonfinite sufficient curvature calculation")
    return result


def first_order_gap(specification, coefficients, gradient):
    """Exact support formula on the selected convex coefficient domain.

    Helmert diagonal coordinates give a trace-zero 16-dimensional box; the
    eight scalar or two-coordinate local blocks give intervals or disks.
    No numerical optimization, clipping, projection or fitted-mode selection.
    """
    p = specification.parameter_count
    theta = _array(coefficients, (p,), "stretch coefficients")
    gradient = _array(gradient, (p,), "stretch gradient")
    rho = specification.spectral_norm_bound
    C = np.einsum("p,pij->ij", theta, specification.modes)
    if np.max(np.abs(np.linalg.eigvalsh(C))) > rho+NUMERICAL_ATOL:
        raise M4CertificateUnverified("stretch is outside its fixed spectral domain", {})
    if specification.name == M4_VARIANTS[0]:
        # Adding any common constant to this diagonal gradient leaves the
        # trace-zero linear objective unchanged. Helmert gives its centered form.
        diagonal_gradient = np.einsum("p,pii->i", gradient, specification.modes)
        ordered = np.sort(diagonal_gradient)
        support = rho*(ordered[8:].sum()-ordered[:8].sum())
    elif specification.name == M4_VARIANTS[1]:
        support = np.sqrt(2.)*rho*np.abs(gradient).sum()
    elif specification.name in M4_VARIANTS[2:]:
        support = np.sqrt(2.)*rho*np.linalg.norm(gradient.reshape(8, 2), axis=1).sum()
    else:
        raise ValueError("unknown fixed M4 variant")
    gap = float(np.dot(gradient, theta)+support)
    if not np.isfinite(gap) or gap < 0:
        # A negative computed gap cannot certify a feasible point; do not
        # silently turn a feasibility/roundoff discrepancy into exact optimality.
        raise M4CertificateUnverified("computed first-order gap is invalid", {"first_order_gap": gap})
    return gap


def _curvature_certificate(objective, specification):
    root_weight = np.sqrt(objective.weights)[:, None]
    A, Y = root_weight*objective.rotated_source, root_weight*objective.active_target
    blocks = tuple(block_curvature_bounds(A[:, start:stop], Y[:, start:stop],
                   global_gain=objective.gain, spectral_norm_bound=specification.spectral_norm_bound)
                   for start, stop in BLOCKS)
    result = {"block_curvature": blocks,
              "hessian_lower_bound": min(item["hessian_lower_bound"] for item in blocks),
              "objective_scale": sum(item["objective_scale"] for item in blocks)}
    if any(item["hessian_lower_bound"] <= item["required_positive_margin"] for item in blocks):
        raise M4CertificateUnverified("sufficient M4 uniqueness curvature remains unverified", result)
    return result


def _solution_certificate(objective, specification, numerical_fit, curvature):
    theta = _array(numerical_fit["coefficients"], (specification.parameter_count,), "fitted coefficients")
    C = objective.matrix(theta)
    supplied_C = _array(numerical_fit["C"], (16, 16), "fitted C")
    if (np.max(np.abs(C-supplied_C)) > NUMERICAL_ATOL
            or not np.array_equal(numerical_fit["O"], objective.O)
            or numerical_fit["gain"] != objective.gain):
        raise ValueError("the fitted C, reference O and common gain must remain paired")
    objective.final_stretch_diagnostics(theta, specification.spectral_norm_bound)
    loss, gradient = objective.value_gradient(theta)
    tangent = objective.tangent_singular_values(theta)
    if tangent[-1] <= RANK_RTOL*tangent[0]:
        raise M4CertificateUnverified("fitted M4 tangent is not identified at the fixed tolerance", {})
    gap = first_order_gap(specification, theta, gradient)
    result = {**curvature, "first_order_gap": gap,
              "required_gap_upper_bound": GAP_RTOL*curvature["objective_scale"],
              "coefficient_distance_upper_bound": float(np.sqrt(2*gap/curvature["hessian_lower_bound"])),
              "weighted_task_a_squared_error": loss,
              "fitted_tangent_singular_values": _frozen(tangent),
              "unique_minimizer_sufficient_curvature": True,
              "solution_within_frozen_gap": gap <= GAP_RTOL*curvature["objective_scale"],
              "exact_global_optimizer": False,
              "interval_arithmetic_proof": False}
    if not result["solution_within_frozen_gap"]:
        raise M4CertificateUnverified("M4 constrained optimality gap exceeds the frozen bound", result)
    return C, result


class PairedStretchVariants:
    """LOPO callback for the four fixed variants of the exact reference M3.

    check() belongs to the caller's allocation/deadline enforcement. It runs
    before each bounded solver and again afterward; there is no retry, mode
    dropping or adaptive spectral-bound shrinkage. Failed certificates refuse
    the callback rather than silently returning a different scientific model.
    """

    def __init__(self, reference_source_recipe, reference_anatomy, *, check):
        if not isinstance(reference_source_recipe, LocalFunctionalSource) or not callable(check):
            raise ValueError("the reference LocalFunctionalSource and caller check are required")
        recipe, anatomy = reference_source_recipe, reference_anatomy
        if (recipe.atol != NUMERICAL_ATOL or recipe.rank_rtol != RANK_RTOL
                or anatomy.blocks != BLOCKS or recipe.blocks != BLOCKS
                or anatomy.split_depth != 3 or anatomy.modes_per_cell != 2
                or tuple(anatomy.voxel_ids) != recipe.voxel_ids
                or not np.array_equal(anatomy.B, recipe.B)
                or not np.array_equal(anatomy.cell_labels, recipe.labels)
                or anatomy.max_cell_diameter_mm != recipe.max_cell_diameter_mm):
            raise ValueError("M4 requires the unchanged reference eight-cell anatomy/basis/limits")
        self.reference, self.check = recipe, check
        self.support_size = recipe.support_size
        self.specifications = stretch_specifications()

    def operator_certificate(self, O, C, specification):
        """Certify actual full-space locality and the paired active spectrum."""
        self.reference.operator_certificate(O)
        B, labels, atol = self.reference.B, self.reference.labels, NUMERICAL_ATOL
        C = _array(C, (16, 16), "C")
        active = O@expm(C)
        identity = np.eye(self.support_size)
        L = identity+B@(active-np.eye(16))@B.T
        complement = identity-B@B.T
        S = B@C@B.T
        allowed = np.zeros((16, 16), dtype=bool)
        for start, stop in BLOCKS:
            allowed[start:stop, start:stop] = True
        rho = specification.spectral_norm_bound
        singular = np.linalg.svd(active, compute_uv=False)
        residuals = {
            "C_symmetry_max_abs": float(np.max(np.abs(C-C.T))),
            "C_trace_abs": float(abs(np.trace(C))),
            "C_cross_block_max_abs": float(np.max(np.abs(C[~allowed]))),
            "C_spectral_bound_excess": float(max(0., np.linalg.norm(C, 2)-rho)),
            "S_constant_annihilation_max_abs": float(np.max(np.abs(S@np.ones(self.support_size)))),
            "L_right_constant_preservation_max_abs": float(np.max(np.abs(L@np.ones(self.support_size)-1))),
            "L_left_constant_preservation_max_abs": float(np.max(np.abs(np.ones(self.support_size)@L-1))),
            "L_right_identity_complement_max_abs": float(np.max(np.abs((L-identity)@complement))),
            "L_left_identity_complement_max_abs": float(np.max(np.abs(complement@(L-identity)))),
            "L_cross_cell_max_abs": float(np.max(np.abs(L[labels[:, None] != labels[None, :]]))),
            "active_upper_singular_bound_excess": float(max(0., singular[0]-np.exp(rho))),
            "active_lower_singular_bound_excess": float(max(0., np.exp(-rho)-singular[-1])),
        }
        if any(not np.isfinite(value) or value > atol for value in residuals.values()):
            raise ValueError("actual M4 operator violates its fixed algebra/locality/spectral limits")
        return {**residuals, "max_transport_distance_bound_mm": self.reference.max_cell_diameter_mm,
                "active_singular_values": _frozen(singular), "locality_certified": True}

    def __call__(self, source, targetA, reference_numerical_fit, context):
        self.check()
        self.reference._source(source, context)
        self.reference.certify_m3(source, reference_numerical_fit, context)
        X = _center(_array(source.task_a_maps, (29, self.support_size), "source Task A"))
        Y = _center(_array(targetA, (29, self.support_size), "target Task A"))
        gain = reference_numerical_fit["gain"]
        common_gain = float(np.sum(self.reference.weights[:, None]*X*Y)
                            /np.sum(self.reference.weights[:, None]*X**2))
        if not np.isfinite(common_gain) or abs(common_gain-gain) > NUMERICAL_ATOL*max(1., abs(gain)):
            raise ValueError("reference gain must be the same pre-operator target-A common gain")
        O = _array(reference_numerical_fit["O"], (16, 16), "paired reference O")
        partitions = _array(source.geometry_state.source_partitions, (4, 18, self.support_size), "four source partitions")
        G = _array(source.G, (18, self.support_size), "shared source G")
        if not np.allclose(G, partitions.mean(axis=0), rtol=0, atol=NUMERICAL_ATOL):
            raise ValueError("shared G must be the mean of its actual four source partitions")
        result = []
        for specification in self.specifications:
            self.check()
            settings = dict(global_gain=gain, condition_weights=self.reference.weights,
                            mode_matrices=specification.modes, rank_bound=16,
                            dof_bound=specification.parameter_count,
                            rank_rtol=RANK_RTOL, numerical_atol=NUMERICAL_ATOL)
            objective = PairedM4TaskAObjective(X, Y, self.reference.B, O, **settings)
            curvature = _curvature_certificate(objective, specification)
            self.check()
            numerical = fit_m4_task_a(X, Y, self.reference.B, O, **settings,
                spectral_norm_bound=specification.spectral_norm_bound,
                optimizer_ftol=OPTIMIZER_FTOL, max_iterations=MAX_ITERATIONS)
            self.check()
            C, certificate = _solution_certificate(objective, specification, numerical, curvature)
            operator = self.operator_certificate(O, C, specification)
            zero = np.zeros((16, 16))
            zero_error = float(np.max(np.abs(gain*apply_m4(X, self.reference.B, O, zero)
                                           -gain*apply_m3(X, self.reference.B, O))))
            if zero_error > NUMERICAL_ATOL:
                raise ValueError("zero stretch fails paired reference M3 replay")
            predicted_partitions = gain*apply_m4(partitions.reshape(72, self.support_size), self.reference.B, O, C)
            predicted_partitions = predicted_partitions.reshape(4, 18, self.support_size)
            maps = gain*apply_m4(G, self.reference.B, O, C)
            geometry = crossvalidated_geometries(predicted_partitions[None])[0].mean(axis=0)
            if not np.isfinite(maps).all() or not np.isfinite(geometry).all():
                raise ArithmeticError("nonfinite paired M4 predictions")
            result.append(M4Prediction(specification.name, _frozen(maps), _frozen(geometry), {
                **certificate, **operator,
                "C": _frozen(C), "coefficients": _frozen(numerical["coefficients"]),
                "O": _frozen(O), "gain": gain, "training_ids": context.training_ids,
                "paired_model": "M3_reference_local_isometry",
                "spectral_norm_bound": specification.spectral_norm_bound,
                "rank_bound": 16, "per_cell_rank_bound": 2,
                "parameter_count_bound": specification.parameter_count,
                "degrees_of_freedom_interpretation": "fixed_parameter_count_and_prediction_tangent_rank_not_statistical_edf",
                "source_partitions_used": 4, "zero_stretch_replay_max_abs": zero_error,
                "local_optimizer_converged": numerical["local_optimizer_converged"],
                "optimizer_iterations": numerical["iterations"],
                "optimizer_ftol": OPTIMIZER_FTOL, "optimizer_max_iterations": MAX_ITERATIONS,
                "global_optimum_certified": False, "unique_solution_certified": False,
                "native_stability_and_ablation_qualified": False,
            }))
        self.check()
        return tuple(result)
