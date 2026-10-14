"""Global diagonal M4 certificate without a global-convexity assertion.

For the unchanged trace-zero box, f(c)=sum(p_i exp(2c_i)-2d_i exp(c_i)).
Each scalar Lagrangian f_i(c_i)+lambda*c_i has only two endpoint candidates
and at most one interior local minimum. A common lambda whose unique scalar
minimizers sum to zero therefore identifies the unique constrained global
minimum. A sign-changing, branch-preserving lambda interval also suffices.

We additionally certify scalar quadratic growth about those minimizers over
the whole original interval. This is a VALUE-gap bound, not a Hessian bound:
concave boundary solutions can pass through their one-sided derivative.
The old per-block numerical margin, optimizer, rank and gap limits are retained.
As in the prior certificate, these are outward-expanded floating-point
acceptance calculations, not a formal interval proof of prior moments/libm.
This array-only module reads no data and does not activate a panel route.
"""
import math

import numpy as np

from paired_stretch_variants import (
    BLOCKS, CURVATURE_MARGIN, GAP_RTOL, NUMERICAL_ATOL, RANK_RTOL,
    M4CertificateUnverified, apply_m3, first_order_gap, stretch_specifications,
)
from proposed_m4_certificate import (
    _I, _i, _exp, _sqrt, _square, _nonnegative, _helmert_gram_lower,
)


def _refuse(message, **diagnostics):
    raise M4CertificateUnverified(message, diagnostics)


def _log(value):
    if value.lo <= 0:
        raise ArithmeticError("positive logarithm interval required")
    lo, hi = math.log(value.lo), math.log(value.hi)
    for _ in range(8):
        lo, hi = math.nextafter(lo, -math.inf), math.nextafter(hi, math.inf)
    return _I(lo, hi)


def _value(p, d, lam, c):
    z = _exp(c)
    return p*z*z - 2*d*z + lam*c


def _candidates(p, d, lam, lo, hi):
    """Enclose every scalar global-minimum candidate, including both ends."""
    p, d = float(p), float(d)
    lam = _i(lam)
    answer = [("lower", _i(lo)), ("upper", _i(hi))]
    discriminant = _i(d)*d - 2*p*lam
    if discriminant.hi > 0:
        root = _sqrt(_nonnegative(discriminant))
        # The minus root has negative second derivative, so cannot be an
        # interior minimum. The plus root is evaluated stably when d < 0.
        if d < 0 and lam.hi < 0:
            z = lam / (_i(d)-root)
        else:
            z = (_i(d)+root)/(2*p)
        domain = _exp(_I(lo, hi))
        if z.hi > 0 and z.hi >= domain.lo and z.lo <= domain.hi:
            z = _I(max(z.lo, domain.lo), min(z.hi, domain.hi))
            c = _log(z)
            answer.append(("interior", _I(max(lo, c.lo), min(hi, c.hi))))
    return [(kind, c, _value(p, d, lam, c)) for kind, c in answer]


def _selected(p, d, lam, rho):
    p, d = float(p), float(d)
    rows = _candidates(p, d, lam, -rho, rho)
    best = min(range(len(rows)), key=lambda j: (rows[j][2].lo+rows[j][2].hi)/2)
    chosen = rows[best]
    strict = all(chosen[2].hi < row[2].lo for j, row in enumerate(rows) if j != best)
    return chosen, strict


def _branch_witness(p, d, rho, check):
    p, d = tuple(map(float, p)), tuple(map(float, d))
    # Beyond these derivative bounds each scalar is monotone on the full box.
    bound = 1 + 2*max(p)*math.exp(2*rho) + 2*max(map(abs, d))*math.exp(rho)
    left, right = -bound, bound

    def endpoint(value):
        selections = [_selected(pi, di, value, rho) for pi, di in zip(p, d)]
        trace = sum((item[0][1] for item in selections), _i(0.))
        return selections, trace, all(item[1] for item in selections)

    at_left, left_trace, left_strict = endpoint(left)
    at_right, right_trace, right_strict = endpoint(right)
    if not (left_strict and right_strict and left_trace.lo > 0 and right_trace.hi < 0):
        _refuse("initial scalar dual trace bracket is unresolved")
    for iteration in range(160):
        check()
        middle = (left+right)/2
        selected, trace, strict = endpoint(middle)
        kinds = tuple(item[0][0] for item in selected)
        if (strict and "interior" not in kinds
                and kinds.count("lower") == kinds.count("upper")):
            # Equal counts of exactly +/-rho satisfy trace zero algebraically.
            return _i(middle), tuple(item[0] for item in selected), iteration+1
        if strict and trace.lo > 0:
            left = middle
            at_left, left_trace = selected, trace
        elif strict and trace.hi < 0:
            right = middle
            at_right, right_trace = selected, trace
        else:
            # A trace of an ambiguously selected minimizer cannot establish
            # the dual sign. Preserve the last endpoints with BOTH strict
            # global scalar selections and certified trace signs. This also
            # preserves a real branch jump instead of shrinking through it.
            new_left, new_right = (left+middle)/2, (middle+right)/2
            low_rows, low_trace, low_strict = endpoint(new_left)
            high_rows, high_trace, high_strict = endpoint(new_right)
            changed = False
            if low_strict and low_trace.lo > 0 and new_left > left:
                left, at_left, left_trace = new_left, low_rows, low_trace
                changed = True
            if high_strict and high_trace.hi < 0 and new_right < right:
                right, at_right, right_trace = new_right, high_rows, high_trace
                changed = True
            if not changed:
                break
        if not left < (left+right)/2 < right:
            break
    lam = _I(left, right)
    diagnostics = {
        "lambda_width": right-left,
        "left_nonstrict_coordinate_count": sum(not row[1] for row in at_left),
        "right_nonstrict_coordinate_count": sum(not row[1] for row in at_right),
        "left_trace_interval": (left_trace.lo, left_trace.hi),
        "right_trace_interval": (right_trace.lo, right_trace.hi),
        "branch_switch_count": sum(a[0][0] != b[0][0] for a, b in zip(at_left, at_right)),
        "endpoint_to_endpoint_switch_count": sum(
            a[0][0] != b[0][0] and "interior" not in (a[0][0], b[0][0])
            for a, b in zip(at_left, at_right)),
    }
    if not (all(item[1] for item in at_left+at_right)
            and left_trace.lo > 0 and right_trace.hi < 0):
        _refuse("scalar dual trace bracket is unresolved", **diagnostics)
    if diagnostics["branch_switch_count"]:
        _refuse("scalar dual branch switch prevents the uniqueness witness", **diagnostics)
    rows = []
    for pi, di, first, last in zip(p, d, at_left, at_right):
        candidates = _candidates(pi, di, lam, -rho, rho)
        matching = [row for row in candidates if row[0] == first[0][0]]
        if len(matching) != 1:
            _refuse("scalar dual branch is unresolved", **diagnostics)
        chosen = matching[0]
        if any(chosen[2].hi >= row[2].lo for row in candidates if row[0] != chosen[0]):
            _refuse("scalar dual minimum separation is unresolved", **diagnostics)
        if chosen[0] == "interior":
            # The plus-root branch must stay strictly inside the box. It has
            # positive curvature; its derivative with respect to lambda is
            # negative, so the continuous trace changes sign in this bracket.
            if not -rho < chosen[1].lo <= chosen[1].hi < rho:
                _refuse("interior scalar branch reaches the box boundary", **diagnostics)
        rows.append(chosen)
    return lam, tuple(rows), iteration+1


def _curvature_lower(p, d, lo, hi):
    z = _exp(_I(lo, hi))
    vertex = _i(d)/(4*p)
    at = _I(min(z.hi, max(z.lo, vertex.lo)), min(z.hi, max(z.lo, vertex.hi)))
    return (4*p*at*at - 2*d*at).lo


def _growth(p, d, lam, rho, optimum, required, check, max_cells):
    kind, center, minimum = optimum
    target = 2*float(required)
    slope = None
    if kind != "interior":
        z = _exp(center)
        derivative = 2*p*z*z - 2*d*z + lam
        slope = derivative.lo if kind == "lower" else -derivative.hi
        if slope < 0:
            _refuse("endpoint scalar optimality is unresolved")
    stack, visited = [(-rho, rho)], 0
    while stack:
        check()
        visited += 1
        if visited > max_cells:
            _refuse("scalar quadratic-growth cover reached its finite cell limit",
                    visited_cells=visited, kind=kind)
        lo, hi = stack.pop()
        delta = _I(lo, hi)-center
        distance = max(abs(delta.lo), abs(delta.hi))
        hessian = _curvature_lower(p, d, min(lo, center.lo), max(hi, center.hi))
        lower = hessian
        if slope is not None and distance > 0:
            lower = (_i(hessian)+2*_i(slope)/distance).lo
        if lower > target:
            continue
        candidates = _candidates(p, d, lam, lo, hi)
        scalar_lower = min(row[2].lo for row in candidates)
        # This enclosure is uniform over the lambda bracket. It is used only
        # away from the minimizer; Taylor/one-sided bounds handle its vicinity.
        separation = (_i(scalar_lower)-minimum).lo
        if separation > (_i(target)*_square(_i(distance))/2).hi:
            continue
        mid = (lo+hi)/2
        if not lo < mid < hi:
            _refuse("scalar quadratic-growth subdivision is unresolved")
        stack.extend(((mid, hi), (lo, mid)))
    return target, visited


def scalar_dual_certificate(p, d, block_energies, rho, *, check, max_cells=4095):
    """Normalized moments; returns a uniform numerical global-growth witness."""
    p, d, energies = (np.asarray(x, dtype=float) for x in (p, d, block_energies))
    if (p.shape != (16,) or d.shape != (16,) or energies.shape != (8,)
            or not np.isfinite(np.r_[p, d, energies, rho]).all()
            or np.any(p <= 0) or np.any(energies <= 0) or rho <= 0
            or type(max_cells) is not int or max_cells <= 0 or not callable(check)):
        raise ValueError("finite normalized 16-coordinate diagonal moments required")
    lam, optima, iterations = _branch_witness(p, d, rho, check)
    growth, cells = [], []
    for i, optimum in enumerate(optima):
        value, visited = _growth(float(p[i]), float(d[i]), lam, rho, optimum,
                                CURVATURE_MARGIN*energies[i//2], check, max_cells)
        growth.append(value)
        cells.append(visited)
    return {
        "lambda_interval": (lam.lo, lam.hi),
        "coordinate_branches": tuple(row[0] for row in optima),
        "coordinate_minimizer_intervals": tuple((row[1].lo, row[1].hi) for row in optima),
        "scalar_minimum_intervals": tuple((row[2].lo, row[2].hi) for row in optima),
        "scalar_growth_lower_bounds": tuple(growth),
        "growth_cover_cells": tuple(cells), "dual_bisection_iterations": iterations,
        "unique_global_minimum_sufficient_dual": True,
        "full_domain_positive_curvature_asserted": False,
    }


def diagonal_solution_certificate(objective, specification, numerical_fit, *, check):
    """Certify the original fitted coefficients; never replaces an optimizer."""
    canonical = stretch_specifications()[0]
    if (specification.name != canonical.name
            or specification.spectral_norm_bound != canonical.spectral_norm_bound
            or not np.array_equal(specification.modes, canonical.modes)
            or not np.array_equal(objective.modes, canonical.modes)):
        raise ValueError("only the unchanged canonical diagonal M4 is supported")
    theta = np.asarray(numerical_fit["coefficients"], dtype=float)
    if theta.shape != (15,) or not np.isfinite(theta).all():
        raise ValueError("finite canonical coefficients required")
    C = objective.matrix(theta)
    if (np.max(np.abs(C-numerical_fit["C"])) > NUMERICAL_ATOL
            or not np.array_equal(numerical_fit["O"], objective.O)
            or numerical_fit["gain"] != objective.gain):
        raise ValueError("the fitted C, reference O and common gain must remain paired")
    objective.final_stretch_diagnostics(theta, specification.spectral_norm_bound)
    loss, gradient = objective.value_gradient(theta)
    tangent = objective.tangent_singular_values(theta)
    if tangent[-1] <= RANK_RTOL*tangent[0]:
        _refuse("fitted M4 tangent is not identified at the fixed tolerance")
    stationarity_gap = first_order_gap(specification, theta, gradient)
    root = np.sqrt(objective.weights)[:, None]
    A, Y = objective.gain*root*objective.rotated_source, root*objective.active_target
    p, d = np.sum(A*A, axis=0), np.sum(A*Y, axis=0)
    coordinate_energy = np.sum(A*A+Y*Y, axis=0)
    energy = float(coordinate_energy.sum())
    block_energy = np.array([coordinate_energy[a:b].sum() for a, b in BLOCKS])
    if not math.isfinite(energy) or energy <= 0:
        raise ValueError("positive finite Task-A objective scale required")
    if stationarity_gap > GAP_RTOL*energy:
        _refuse("M4 first-order gap exceeds the unchanged frozen bound",
                first_order_gap=stationarity_gap, required_gap_upper_bound=GAP_RTOL*energy)
    witness = scalar_dual_certificate(p/energy, d/energy, block_energy/energy,
                                     specification.spectral_norm_bound, check=check)
    c = np.diag(C)
    candidate_value, optimum_value, distance_squared = _i(0.), _i(0.), _i(0.)
    for pi, di, ci, bounds in zip(p/energy, d/energy, c,
                                witness["coordinate_minimizer_intervals"]):
        optimum = _I(*bounds)
        candidate_value = candidate_value+_value(float(pi), float(di), _i(0.), _i(float(ci)))
        optimum_value = optimum_value+_value(float(pi), float(di), _i(0.), optimum)
        distance_squared = distance_squared+_square(_i(float(ci))-optimum)
    dual_lower = sum((_I(*row) for row in witness["scalar_minimum_intervals"]), _i(0.)).lo
    gap_upper = max(0., (candidate_value-_i(dual_lower)).hi)*energy
    difference = candidate_value-optimum_value
    absolute_difference_upper = max(abs(difference.lo), abs(difference.hi))*energy
    if (not math.isfinite(gap_upper) or not math.isfinite(absolute_difference_upper)
            or max(gap_upper, absolute_difference_upper) > GAP_RTOL*energy):
        _refuse("M4 global primal-dual gap exceeds the unchanged frozen bound",
                global_gap_upper_bound=gap_upper,
                absolute_objective_difference_upper_bound=absolute_difference_upper,
                required_gap_upper_bound=GAP_RTOL*energy)
    gram_lower = _helmert_gram_lower(specification.modes)
    growth_lower = (_i(min(witness["scalar_growth_lower_bounds"]))*energy*gram_lower).lo
    # The returned floating-point fit obeys the original feasibility tolerance,
    # not exact algebraic equality. Use the explicit unique-minimizer enclosure
    # for distance rather than silently applying a feasible-point growth bound
    # to a point with a tiny trace/box discrepancy. No fit is projected/refitted.
    distance_upper = _sqrt(_nonnegative(distance_squared)/gram_lower).hi
    return C, {
        **witness, "method": "diagonal_scalar_dual_and_global_quadratic_growth",
        "objective_scale": energy, "first_order_gap": stationarity_gap,
        "first_order_gap_is_stationarity_only_without_global_convexity": True,
        "global_gap_upper_bound": gap_upper, "required_gap_upper_bound": GAP_RTOL*energy,
        "absolute_objective_difference_upper_bound": absolute_difference_upper,
        "global_coefficient_growth_lower_bound": growth_lower,
        "coefficient_distance_upper_bound": distance_upper,
        "coefficient_distance_method": "direct_unique_minimizer_interval_and_Helmert_Gram_bound",
        "weighted_task_a_squared_error": loss,
        "fitted_tangent_singular_values": tangent,
        "solution_within_frozen_gap": True, "exact_global_optimizer": False,
        "interval_arithmetic_proof": False,
    }


def fit_diagonal_via_dual(objective, specification, *, check):
    """Construct the same-model numerical fit from the certified scalar root.

    There is no SLSQP call, altered initialization search or changed domain.
    The midpoint of the unique-minimizer enclosure is represented in the
    original Helmert coordinates; all original fit checks still apply. The
    independent post-fit certificate repeats the scalar calculation and gates
    the actual returned coefficients, rather than trusting this construction.
    The caller retains the original final operator/locality checks.
    """
    canonical = stretch_specifications()[0]
    if (specification.name != canonical.name
            or specification.spectral_norm_bound != canonical.spectral_norm_bound
            or not np.array_equal(specification.modes, canonical.modes)
            or not np.array_equal(objective.modes, canonical.modes)):
        raise ValueError("only the unchanged canonical diagonal M4 is supported")
    check()
    root = np.sqrt(objective.weights)[:, None]
    A, Y = objective.gain*root*objective.rotated_source, root*objective.active_target
    p, d = np.sum(A*A, axis=0), np.sum(A*Y, axis=0)
    coordinate_energy = np.sum(A*A+Y*Y, axis=0)
    energy = float(coordinate_energy.sum())
    if not math.isfinite(energy) or energy <= 0:
        raise ValueError("positive finite Task-A objective scale required")
    block_energy = np.array([coordinate_energy[a:b].sum() for a, b in BLOCKS])
    witness = scalar_dual_certificate(p/energy, d/energy, block_energy/energy,
                                     specification.spectral_norm_bound, check=check)
    middle = np.array([(lo+hi)/2 for lo, hi in witness["coordinate_minimizer_intervals"]])
    H = np.diagonal(specification.modes, axis1=1, axis2=2).T
    theta = np.linalg.solve(H.T@H, H.T@middle)
    C = objective.matrix(theta)
    diagnostics = objective.final_stretch_diagnostics(theta, specification.spectral_norm_bound)
    eigenvalues = np.linalg.eigvalsh(C)
    zero = np.zeros(specification.parameter_count)
    initial_loss, _ = objective.value_gradient(zero)
    loss, gradient = objective.value_gradient(theta)
    if (np.max(np.abs(eigenvalues)) > specification.spectral_norm_bound+objective.atol
            or loss > initial_loss+objective.atol):
        raise ArithmeticError("dual M4 fit is infeasible or above its paired zero-stretch loss")
    tangent = objective.tangent_singular_values(theta)
    if len(tangent) != len(theta) or tangent[-1] <= objective.rank_rtol*tangent[0]:
        _refuse("fitted M4 tangent is not identified at the fixed tolerance")
    zero_prediction = objective.predict(objective.source, zero)
    paired_prediction = objective.gain*apply_m3(objective.source, objective.B, objective.O)
    zero_error = float(np.max(np.abs(zero_prediction-paired_prediction)))
    if zero_error > objective.atol:
        raise ArithmeticError("zero stretch does not reproduce the paired M3 prediction")
    numerical = {
        "C": C, "coefficients": theta, "O": objective.O.copy(), "gain": objective.gain,
        "weighted_task_a_squared_error": loss, "paired_zero_stretch_loss": initial_loss,
        "C_eigenvalues": eigenvalues, "joint_mode_rank": objective.joint_mode_rank,
        "source_design_singular_values": objective.source_singular,
        "fitted_tangent_singular_values": tangent, "objective_gradient": gradient,
        "iterations": witness["dual_bisection_iterations"],
        "zero_stretch_replay_max_abs": zero_error,
        "numerical_solver": "separable_scalar_dual_in_unchanged_Helmert_domain",
        "numerical_solver_converged": True, "local_optimizer_converged": False,
        "global_optimum_certified": False, "unique_solution_certified": False,
        "locality_certified": False, "empirical_recipe_selected": False,
        **diagnostics,
    }
    check()
    _, certificate = diagonal_solution_certificate(objective, specification, numerical, check=check)
    return numerical, certificate
