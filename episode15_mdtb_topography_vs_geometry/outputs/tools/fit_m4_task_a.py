"""Bounded Task-A stretch/shear fitting for an explicitly supplied paired M3.

No empirical reader or scientific recipe is selected here. Source maps, B, O,
the common global gain, stretch modes, limits and numerical settings must all
be supplied by the caller. A converged local fit is not a unique-solution or
locality certificate and cannot by itself release an empirical comparison.
"""
import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize

from fit_m3_task_a import _tolerance
from spatial_deformation_operators import _matrix, apply_m3, apply_m4


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isrealobj(value):
        raise ValueError(f"{name} must be a positive finite real scalar")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite real scalar")
    return result


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be an explicit positive integer")
    return int(value)


class PairedM4TaskAObjective:
    """Exact weighted objective/gradient with one fixed paired O and common gain.

    C(theta) is a linear combination of supplied independent symmetric,
    trace-free matrices. Their joint image must fit the supplied rank bound;
    this makes every allowed C low rank without truncating a fitted result.
    No initial stretch, target Task-B input, fitted-basis repair, or gain refit.
    """

    def __init__(self, source_task_a, target_task_a, B, O, *, global_gain,
                 condition_weights, mode_matrices, rank_bound, dof_bound,
                 rank_rtol, numerical_atol):
        self.atol = _tolerance(numerical_atol, "numerical_atol")
        self.rank_rtol = _tolerance(rank_rtol, "rank_rtol")
        self.gain = _positive(global_gain, "global_gain")
        self.rank_bound = _integer(rank_bound, "rank_bound")
        self.dof_bound = _integer(dof_bound, "dof_bound")
        source, target = _matrix(source_task_a, "source_task_a"), _matrix(target_task_a, "target_task_a")
        self.B, self.O = _matrix(B, "B").copy(), _matrix(O, "O").copy()
        if source.shape != target.shape or source.shape[0] == 0 or source.shape[1] < 2:
            raise ValueError("matched nonempty Task-A condition/voxel axes are required")
        n, v = source.shape
        if self.B.shape[0] != v or not 1 < self.B.shape[1] < v:
            raise ValueError("B must define at least two centered active directions")
        d = self.B.shape[1]
        if self.O.shape != (d, d) or self.rank_bound > d:
            raise ValueError("paired O and rank bound must use the supplied active basis")
        identity = np.eye(d)
        if max(np.max(np.abs(self.B.T@self.B-identity)), np.max(np.abs(self.B.sum(axis=0))),
               np.max(np.abs(self.O.T@self.O-identity)), np.max(np.abs(self.O@self.O.T-identity))) > self.atol:
            raise ValueError("B/O must already meet centered orthonormal/orthogonal conditions")
        modes = np.asarray(mode_matrices)
        if modes.ndim != 3 or modes.shape[1:] != (d, d) or modes.dtype.kind not in "fiu":
            raise ValueError("mode_matrices must be a real mode-by-active-by-active tensor")
        self.modes = np.asarray(modes, dtype=np.float64).copy()
        p = len(self.modes)
        if not 0 < p <= self.dof_bound or not np.all(np.isfinite(self.modes)):
            raise ValueError("explicit finite mode count must fit the degree-of-freedom bound")
        mode_scale = np.max(np.abs(self.modes), axis=(1, 2))
        if np.any(mode_scale == 0):
            raise ValueError("stretch parameterization includes a zero mode")
        symmetry_error = np.max(np.abs(self.modes-self.modes.transpose(0, 2, 1)), axis=(1, 2))
        trace_error = np.abs(np.trace(self.modes, axis1=1, axis2=2))
        if (np.any(symmetry_error/mode_scale > self.atol)
                or np.any(trace_error/mode_scale > self.atol)):
            raise ValueError("every supplied mode must be symmetric and trace free")
        mode_singular = np.linalg.svd(self.modes.reshape(p, -1), compute_uv=False)
        if len(mode_singular) != p or mode_singular[-1] <= self.rank_rtol*mode_singular[0]:
            raise ValueError("stretch parameterization is not full rank")
        joint_singular = np.linalg.svd(np.concatenate(list(self.modes), axis=1), compute_uv=False)
        self.joint_mode_rank = int(np.sum(joint_singular > self.rank_rtol*joint_singular[0]))
        if self.joint_mode_rank > self.rank_bound:
            raise ValueError("joint stretch-mode image exceeds the supplied rank bound")
        weights = np.asarray(condition_weights)
        if weights.shape != (n,) or weights.dtype.kind not in "fiu":
            raise ValueError("condition_weights must be a real vector on the Task-A axis")
        weights = np.asarray(weights, dtype=np.float64)
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("condition_weights must be finite and strictly positive")
        weights = weights/weights.max()
        weights = weights/weights.sum()
        if np.any(weights == 0):
            raise ValueError("condition weights underflowed; no condition may be removed")
        self.weights = weights
        self.source = source-source.mean(axis=1, keepdims=True)
        self.target = target-target.mean(axis=1, keepdims=True)
        self.rotated_source = self.source@self.B@self.O
        self.active_target = self.target@self.B
        active_design = np.sqrt(weights)[:, None]*(self.source@self.B)
        self.source_singular = np.linalg.svd(active_design, compute_uv=False)
        if len(self.source_singular) != d or self.source_singular[-1] <= self.rank_rtol*self.source_singular[0]:
            raise ValueError("Task-A active source design is not full rank")
        source_complement = self.source-(self.source@self.B)@self.B.T
        target_complement = self.target-(self.target@self.B)@self.B.T
        residual_complement = self.gain*source_complement-target_complement
        self.complement_loss = float(np.sum(weights[:, None]*residual_complement**2))

    def matrix(self, coefficients):
        theta = np.asarray(coefficients)
        if theta.shape != (len(self.modes),) or theta.dtype.kind not in "fiu" or not np.all(np.isfinite(theta)):
            raise ValueError("coefficients must be finite on the fixed mode axis")
        result = np.einsum("p,pij->ij", theta, self.modes)
        if not np.all(np.isfinite(result)):
            raise ArithmeticError("nonfinite stretch matrix")
        return result

    def value_gradient(self, coefficients):
        C = self.matrix(coefficients)
        residual = self.gain*(self.rotated_source@expm(C))-self.active_target
        weighted_residual = self.weights[:, None]*residual
        gradient = np.array([2*self.gain*np.sum(weighted_residual*(self.rotated_source@expm_frechet(C, mode, compute_expm=False)))
                             for mode in self.modes])
        value = float(np.sum(weighted_residual*residual)+self.complement_loss)
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise ArithmeticError("nonfinite Task-A objective or Frechet gradient")
        return value, gradient

    def tangent_singular_values(self, coefficients):
        C = self.matrix(coefficients)
        jacobian = np.column_stack([(np.sqrt(self.weights)[:, None]*self.gain*
                                    (self.rotated_source@expm_frechet(C, mode, compute_expm=False))).ravel()
                                   for mode in self.modes])
        return np.linalg.svd(jacobian, compute_uv=False)

    def predict(self, source_maps, coefficients):
        """Apply the same paired O/gain/C; caller owns all data-role boundaries."""
        maps = _matrix(source_maps, "source_maps")
        maps = maps-maps.mean(axis=1, keepdims=True)
        return self.gain*apply_m4(maps, self.B, self.O, self.matrix(coefficients))

    def final_stretch_diagnostics(self, coefficients, spectral_norm_bound):
        """Check the actual fitted C, including coefficient-amplified defects.

        Input-mode checks do not substitute for these output checks. Singular
        values measure true matrix norm/rank; eigvalsh alone would ignore one
        triangle if finite arithmetic or supplied modes introduced asymmetry.
        No projection, symmetrization, rank truncation or trace repair occurs.
        """
        C = self.matrix(coefficients)
        singular = np.linalg.svd(C, compute_uv=False)
        norm = float(singular[0])
        rank = int(np.sum(singular > self.rank_rtol*norm)) if norm else 0
        symmetry = float(np.max(np.abs(C-C.T)))
        trace = float(np.trace(C))
        if symmetry > self.atol or abs(trace) > self.atol or rank > self.rank_bound:
            raise ArithmeticError("fitted C violates symmetry, trace or numerical rank constraints")
        if norm > spectral_norm_bound+self.atol:
            raise ArithmeticError("fitted C violates its true singular-value spectral norm bound")
        return {"C_symmetry_max_abs": symmetry, "C_trace": trace,
                "C_numerical_rank": rank, "C_spectral_norm": norm}


def fit_m4_task_a(source_task_a, target_task_a, B, O, *, global_gain,
                  condition_weights, mode_matrices, rank_bound, dof_bound,
                  spectral_norm_bound, rank_rtol, numerical_atol,
                  optimizer_ftol, max_iterations):
    """Fit the bounded paired stretch with zero initialization and SLSQP.

    Every scientific limit is explicit. Eigenvalue inequalities impose
    -rho I <= C <= rho I. The fixed mode image imposes the rank bound.
    Failed convergence, infeasibility or an unidentified local tangent refuses.
    SLSQP success is not proof of a global optimum or unique solution; both
    corresponding certificate flags remain false for the downstream recipe.
    """
    rho = _positive(spectral_norm_bound, "spectral_norm_bound")
    ftol = _tolerance(optimizer_ftol, "optimizer_ftol")
    iterations = _integer(max_iterations, "max_iterations")
    objective = PairedM4TaskAObjective(source_task_a, target_task_a, B, O,
        global_gain=global_gain, condition_weights=condition_weights, mode_matrices=mode_matrices,
        rank_bound=rank_bound, dof_bound=dof_bound, rank_rtol=rank_rtol, numerical_atol=numerical_atol)
    zero = np.zeros(len(objective.modes))
    initial_loss, _ = objective.value_gradient(zero)

    def constraints(theta):
        eigenvalues = np.linalg.eigvalsh(objective.matrix(theta))
        return np.concatenate((rho-eigenvalues, rho+eigenvalues))

    result = minimize(objective.value_gradient, zero, jac=True, method="SLSQP",
                      constraints={"type": "ineq", "fun": constraints},
                      options={"ftol": ftol, "maxiter": iterations, "disp": False})
    if not result.success:
        raise ArithmeticError(f"M4 local optimizer did not converge: {result.message}")
    theta = result.x.copy()
    C = objective.matrix(theta)
    stretch_diagnostics = objective.final_stretch_diagnostics(theta, rho)
    eigenvalues = np.linalg.eigvalsh(C)
    loss, gradient = objective.value_gradient(theta)
    if np.max(np.abs(eigenvalues)) > rho+objective.atol or loss > initial_loss+objective.atol:
        raise ArithmeticError("M4 returned infeasible stretch or loss above its paired zero-stretch start")
    tangent_singular = objective.tangent_singular_values(theta)
    if len(tangent_singular) != len(theta) or tangent_singular[-1] <= objective.rank_rtol*tangent_singular[0]:
        raise ValueError("fitted Task-A stretch tangent is not identified at rank_rtol")
    zero_prediction = objective.predict(objective.source, zero)
    paired_prediction = objective.gain*apply_m3(objective.source, objective.B, objective.O)
    zero_error = float(np.max(np.abs(zero_prediction-paired_prediction)))
    if zero_error > objective.atol:
        raise ArithmeticError("zero stretch does not reproduce the paired M3 prediction")
    return {"C": C, "coefficients": theta, "O": objective.O.copy(), "gain": objective.gain,
            "weighted_task_a_squared_error": loss, "paired_zero_stretch_loss": initial_loss,
            "C_eigenvalues": eigenvalues, "joint_mode_rank": objective.joint_mode_rank,
            "source_design_singular_values": objective.source_singular,
            "fitted_tangent_singular_values": tangent_singular,
            "objective_gradient": gradient, "iterations": int(result.nit),
            "zero_stretch_replay_max_abs": zero_error, "local_optimizer_converged": True,
            "global_optimum_certified": False, "unique_solution_certified": False,
            "locality_certified": False, "empirical_recipe_selected": False,
            **stretch_diagnostics}
