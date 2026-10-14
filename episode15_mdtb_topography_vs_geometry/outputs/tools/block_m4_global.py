"""Isolated global certificates for the three unchanged block-local M4 models.

Eight independent scalar intervals or two-dimensional disks are retained.
Shear orientation is minimized algebraically at each radius. Numerical scalar
minimization proposes candidates only: local growth and exclusion of EVERY
exterior interval establish acceptance. No grid or polynomial-root list is a
certificate. The assurance is the existing outward-expanded floating-point
standard, not a formal interval proof of input moments or libm.
"""
import math
from fractions import Fraction

import numpy as np
from scipy.optimize import minimize_scalar

from diagonal_m4_global import SearchBudget, SearchStop, _quadratic_minimum
from paired_stretch_variants import (
    BLOCKS, CURVATURE_MARGIN, GAP_RTOL, M4CertificateUnverified, NUMERICAL_ATOL,
    RANK_RTOL, apply_m3, stretch_specifications,
)
from proposed_m4_certificate import _I, _i, _exp, _square, _sqrt, _nonnegative


def refuse(reason, **details):
    raise M4CertificateUnverified(reason, details)


def _sinh(x):
    x = _i(x)
    lo = ((_exp(x.lo)-_exp(-x.lo))/2).lo
    hi = ((_exp(x.hi)-_exp(-x.hi))/2).hi
    return _I(max(0., lo) if x.lo >= 0 else lo, hi)


def _cosh(x):
    x = _i(x)
    a = ((_exp(x.lo)+_exp(-x.lo))/2)
    b = ((_exp(x.hi)+_exp(-x.hi))/2)
    return _I(1. if x.lo <= 0 <= x.hi else min(a.lo, b.lo), max(a.hi, b.hi))


def _norm(values):
    return _sqrt(_nonnegative(sum((_square(value) for value in values), _i(0.))))


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), _i(0.))


class BlockLoss:
    def __init__(self, P, D, rho, scale, kind):
        self.P, self.D = np.asarray(P, float), np.asarray(D, float)
        if (self.P.shape != (2, 2) or self.D.shape != (2, 2)
                or not np.isfinite(np.r_[self.P.ravel(), self.D.ravel(), rho, scale]).all()
                or rho <= 0 or scale <= 0 or kind not in ("local_diagonal", "local_shear")):
            raise ValueError("finite paired two-mode moments and original block domain required")
        self.rho, self.scale, self.kind = float(rho), float(scale), kind
        self.a = _i(self.P[0, 0])+self.P[1, 1]
        self.b = _i(self.D[0, 0])+self.D[1, 1]
        self.p = (_i(self.P[0, 0])-self.P[1, 1], 2*_i(self.P[0, 1]))
        self.d = (_i(self.D[0, 0])-self.D[1, 1], 2*_i(self.D[0, 1]))
        # Equality is of the stored moments, not an approximate angular waiver.
        exact = Fraction.from_float
        self.origin_stationary = (exact(float(self.P[0, 0]))-exact(float(self.P[1, 1]))
                                  == exact(float(self.D[0, 0]))-exact(float(self.D[1, 1]))
                                  and self.P[0, 1] == self.D[0, 1])
        self.isotropic = (self.P[0, 0] == self.P[1, 1] and self.P[0, 1] == 0
                          and self.D[0, 0] == self.D[1, 1] and self.D[0, 1] == 0)
        self.domain = (-self.rho, self.rho) if kind == "local_diagonal" else (0., self.rho)
        if self.a.lo <= 0 or not np.array_equal(self.P, self.P.T) or not np.array_equal(self.D, self.D.T):
            raise ValueError("symmetric block moments with positive source energy required")

    def evaluate(self, x, derivatives=True):
        x = _i(x)
        if self.kind == "local_diagonal":
            e, q = _exp(x), _exp(-x)
            p0, p1, d0, d1 = (float(matrix[i, i]) for matrix in (self.P, self.D) for i in range(2))
            value = p0*e*e+p1*q*q-2*d0*e-2*d1*q
            first = 2*p0*e*e-2*p1*q*q-2*d0*e+2*d1*q
            second = 4*p0*e*e+4*p1*q*q-2*d0*e-2*d1*q
            return value, first, second
        c, s = _cosh(x), _sinh(x)
        v = tuple(pi*c-di for pi, di in zip(self.p, self.d))
        q = _norm(v)
        if self.isotropic:
            q = _i(0.)
        value = self.a*_cosh(2*x)-2*self.b*c-2*s*q
        if not derivatives:
            return value, None, None
        if self.isotropic:
            return value, 2*self.a*_sinh(2*x)-2*self.b*s, 4*self.a*_cosh(2*x)-2*self.b*c
        if q.lo <= 0:
            return value, None, None
        vp, vpp = tuple(pi*s for pi in self.p), tuple(pi*c for pi in self.p)
        vv = _dot(v, vp)
        qp = vv/q
        qpp = (_dot(vp, vp)+_dot(v, vpp))/q-_square(vv)/(q*q*q)
        first = 2*self.a*_sinh(2*x)-2*self.b*s-2*c*q-2*s*qp
        second = 4*self.a*_cosh(2*x)-2*self.b*c-2*s*q-4*c*qp-2*s*qpp
        return value, first, second

    def anchor(self, x):
        """Enclose an exactly feasible scalar or exact angular-minimum point."""
        if self.kind == "local_diagonal":
            return (_i(x),)
        if x == 0:
            return (_i(0.), _i(0.))
        v = tuple(pi*_cosh(x)-di for pi, di in zip(self.p, self.d))
        q = _norm(v)
        if q.lo <= 0:
            refuse("positive-radius angular minimum is unresolved or nonunique")
        return tuple(-_i(x)*value/q for value in v)

    def full_growth(self, lo, hi, radial_growth):
        if self.kind == "local_diagonal":
            return (_i(radial_growth)*self.scale*self.scale).lo
        if lo <= 0:
            refuse("positive-radius angular-growth interval reaches the origin")
        x = _I(lo, hi)
        v = tuple(pi*_cosh(x)-di for pi, di in zip(self.p, self.d))
        q, pnorm = _norm(v), _norm(self.p)
        if q.lo <= 0:
            refuse("angular separation is unresolved on the entire local interval")
        # n*(r)=-(p cosh(r)-d)/||p cosh(r)-d||. The derivative of
        # normalization is an orthogonal projection divided by that norm.
        lipschitz = (pnorm*_sinh(x)/q).hi
        wmin = (2*_sinh(x)*q).lo
        radial = (_i(radial_growth)/(2*_square(_i(1.)+hi*lipschitz))).lo
        angular = (_i(wmin)/(2*_square(_i(hi)))).lo
        return (_i(min(radial, angular))*self.scale*self.scale).lo

    def actual_value_gap(self, coordinates):
        """Original block value and outward linear-support gap at returned C."""
        if self.kind == "local_diagonal":
            x = _i(float(coordinates[0]))
            value, gradient, _ = self.evaluate(x)
            gap = (gradient*x+self.rho*max(abs(gradient.lo), abs(gradient.hi))).hi
            return value, max(0., gap)
        x = tuple(_i(float(value)) for value in coordinates)
        if all(value.lo == value.hi == 0 for value in x):
            value, gradient = self.a-2*self.b, tuple(2*(pi-di) for pi, di in zip(self.p, self.d))
        else:
            r = _norm(x)
            if r.lo <= 0:
                refuse("returned shear radius is unresolved")
            n = tuple(value/r for value in x)
            w = tuple(pi*_sinh(2*r)-2*di*_sinh(r) for pi, di in zip(self.p, self.d))
            wp = tuple(2*pi*_cosh(2*r)-2*di*_cosh(r) for pi, di in zip(self.p, self.d))
            value = self.a*_cosh(2*r)-2*self.b*_cosh(r)+_dot(w, n)
            radial = 2*self.a*_sinh(2*r)-2*self.b*_sinh(r)+_dot(wp, n)
            dot = _dot(w, n)
            gradient = tuple(radial*ni+(wi-dot*ni)/r for wi, ni in zip(w, n))
        gap = (_dot(gradient, x)+self.rho*_norm(gradient)).hi
        return value, max(0., gap)


def _root(loss, lo, hi, budget):
    """Maintain strict derivative signs around the unique local radial root."""
    for _ in range(120):
        budget.tick()
        mid = (lo+hi)/2
        if not lo < mid < hi:
            break
        derivative = loss.evaluate(_i(mid))[1]
        if derivative is None:
            refuse("stationary derivative is unresolved")
        if derivative.lo > 0:
            hi = mid
        elif derivative.hi < 0:
            lo = mid
        else:
            a, b = (lo+mid)/2, (mid+hi)/2
            ga, gb = loss.evaluate(_i(a))[1], loss.evaluate(_i(b))[1]
            changed = False
            if ga is not None and ga.hi < 0 and a > lo:
                lo, changed = a, True
            if gb is not None and gb.lo > 0 and b < hi:
                hi, changed = b, True
            if not changed:
                break
    return _I(lo, hi)


def _local(loss, candidate, budget):
    lower, upper = loss.domain
    for level in range(25):
        budget.tick()
        radius = (upper-lower)/(2**level)
        lo, hi = max(lower, candidate-radius), min(upper, candidate+radius)
        if not lo < hi:
            continue
        if loss.kind == "local_shear" and candidate == 0:
            if not loss.origin_stationary:
                continue
            # With p=d, h(r)-h(0)=2(cosh(r)-1)*
            # [a(cosh(r)+1)-b-||p||sinh(r)]. cosh(r)-1 >= r²/2.
            radial_growth = (2*(2*loss.a-loss.b-_norm(loss.p)*_sinh(hi))).lo
            growth = (_i(radial_growth)*loss.scale*loss.scale).lo
            if growth > CURVATURE_MARGIN:
                value = loss.a-2*loss.b
                return dict(lo=0., hi=hi, x=0., value=value, lower=value.lo,
                            growth=growth, radial_growth=radial_growth, method="strict_origin_growth")
            continue
        value, _, curvature = loss.evaluate(_I(lo, hi))
        if curvature is None:
            continue
        at_lower, at_upper = loss.evaluate(_i(lo)), loss.evaluate(_i(hi))
        if at_lower[1] is None or at_upper[1] is None:
            continue
        if candidate == lower and at_lower[1].lo > 0:
            radial_growth = (_i(curvature.lo)+2*_i(at_lower[1].lo)/(hi-lo)).lo
            x, point, bound, method = lo, at_lower, at_lower[0].lo, "strict_lower_endpoint_growth"
        elif candidate == upper and at_upper[1].hi < 0:
            radial_growth = (_i(curvature.lo)-2*_i(at_upper[1].hi)/(hi-lo)).lo
            x, point, bound, method = hi, at_upper, at_upper[0].lo, "strict_upper_endpoint_growth"
        elif curvature.lo > 0 and at_lower[1].hi < 0 and at_upper[1].lo > 0:
            root = _root(loss, lo, hi, budget)
            x, radial_growth = (root.lo+root.hi)/2, curvature.lo
            point = loss.evaluate(_i(x))
            dx, terms = _I(lo, hi)-x, [0.]
            if dx.lo < 0:
                terms.append(_quadratic_minimum(point[1].hi, radial_growth, dx.lo, min(0., dx.hi)))
            if dx.hi > 0:
                terms.append(_quadratic_minimum(point[1].lo, radial_growth, max(0., dx.lo), dx.hi))
            bound, method = (_i(point[0].lo)+min(terms)).lo, "strict_interior_radial_growth"
        else:
            continue
        if radial_growth <= 0:
            continue
        try:
            growth = loss.full_growth(lo, hi, radial_growth)
        except M4CertificateUnverified:
            continue
        if growth > CURVATURE_MARGIN:
            return dict(lo=lo, hi=hi, x=x, value=point[0], lower=bound,
                        growth=growth, radial_growth=radial_growth, method=method)
    refuse("no sufficient local radial/angular growth certificate", **budget.snapshot())


def _bound(loss, lo, hi):
    """Interval, monotonicity, and midpoint Taylor lower bounds combined.

    The Taylor polynomial uses the whole-interval lower second derivative and
    the adverse endpoint of the midpoint gradient enclosure on each side.
    This preserves shared dependence through the nearly cancelling terms of
    a shallow well; independent exponential intervals alone lose that bound.
    """
    value, derivative, curvature = loss.evaluate(_I(lo, hi))
    lower = value.lo
    if derivative is not None and derivative.lo > 0:
        lower = max(lower, loss.evaluate(_i(lo), False)[0].lo)
    if derivative is not None and derivative.hi < 0:
        lower = max(lower, loss.evaluate(_i(hi), False)[0].lo)
    if curvature is not None:
        mid = (lo+hi)/2
        center, gradient, _ = loss.evaluate(_i(mid))
        if gradient is not None:
            delta, terms = _I(lo, hi)-mid, [0.]
            for left, right, g in ((delta.lo, min(0., delta.hi), gradient.hi),
                                   (max(0., delta.lo), delta.hi, gradient.lo)):
                if left >= right:
                    continue
                if curvature.lo > 0:
                    terms.append(_quadratic_minimum(g, curvature.lo, left, right))
                else:
                    # A linear/concave lower polynomial reaches its minimum
                    # at an endpoint; never use its stationary maximum.
                    terms.extend((_i(g)*x+curvature.lo*_square(_i(x))/2).lo
                                 for x in (left, right))
            lower = max(lower, (_i(center.lo)+min(terms)).lo)
    return lower


def certify_block(P, D, rho, scale, kind, *, check, seconds=300., max_cells=4096, budget=None):
    """Normalized block moments; accept only complete-domain uniqueness/gap."""
    import heapq
    budget = SearchBudget(check, seconds, max_cells) if budget is None else budget
    loss = BlockLoss(P, D, rho, scale, kind)
    lower, upper = loss.domain
    def objective(x):
        budget.tick()
        value = loss.evaluate(_i(float(x)), False)[0]
        return (value.lo+value.hi)/2
    proposal = minimize_scalar(objective, bounds=(lower, upper), method="bounded",
                               options={"xatol": 1e-13, "maxiter": 500})
    candidates = [lower, upper, float(proposal.x)]
    candidate = min(candidates, key=lambda x: loss.evaluate(_i(x), False)[0].hi)
    restarts = 0
    while True:
        region = _local(loss, candidate, budget)
        anchor = loss.anchor(region["x"])
        incumbent = region["value"].hi
        diameter = _square(2*_i(rho)/scale).hi
        threshold = (_i(incumbent)+CURVATURE_MARGIN*_i(diameter)/2).hi
        queue, serial, excluded_lower = [], 0, math.inf
        def push(lo, hi):
            nonlocal serial, excluded_lower
            if not lo < hi:
                return
            budget.cell()
            bound = _bound(loss, lo, hi)
            if bound > threshold:
                excluded_lower = min(excluded_lower, bound)
            else:
                heapq.heappush(queue, (bound, serial, lo, hi))
                serial += 1
        push(lower, region["lo"])
        push(region["hi"], upper)
        improved = None
        while queue:
            budget.tick()
            best, _, lo, hi = heapq.heappop(queue)
            budget.diagnostics.update(remaining_intervals=len(queue)+1,
                best_remaining_block_gap=max(0., incumbent-min(best, region["lower"])),
                candidate_restarts=restarts, local_method=region["method"])
            mid = (lo+hi)/2
            if not lo < mid < hi:
                refuse("unexcluded scalar interval reached floating-point resolution", **budget.snapshot())
            point = loss.evaluate(_i(mid), False)[0]
            if point.hi < region["value"].lo:
                # A certainly better feasible value changes only the candidate;
                # the next iteration reproves local growth and the FULL domain.
                trial = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                                        options={"xatol": 1e-13, "maxiter": 500})
                improved = min((mid, lo, hi, float(trial.x)),
                    key=lambda x: loss.evaluate(_i(x), False)[0].hi)
                break
            push(lo, mid)
            push(mid, hi)
        if improved is not None:
            candidate, restarts = improved, restarts+1
            continue
        total_lower = min(region["lower"], excluded_lower)
        gap = (_i(incumbent)-total_lower).hi
        outside_growth = ((2*(_i(excluded_lower)-incumbent)/diameter).lo
                          if math.isfinite(excluded_lower) else math.inf)
        growth = min(region["growth"], outside_growth)
        if gap > GAP_RTOL or growth <= CURVATURE_MARGIN:
            refuse("block global gap or full-coordinate growth is unresolved", **budget.snapshot(),
                   normalized_gap=gap, coefficient_growth_lower_bound=growth)
        budget.diagnostics["remaining_intervals"] = 0
        return {**budget.snapshot(), "coordinate_intervals": anchor,
            "lower_bound": total_lower, "feasible_upper": incumbent,
            "normalized_gap": gap, "coefficient_growth_lower_bound": growth,
            "local_method": region["method"], "local_interval_width": region["hi"]-region["lo"],
            "radial_growth_lower_bound": region["radial_growth"],
            "all_exterior_intervals_excluded": True, "candidate_restarts": restarts,
            "candidate_optimizer_iterations": int(proposal.nit),
            "full_domain_positive_curvature_asserted": False}


def fit_block_global(objective, specification, *, check, seconds=300., max_cells=4096):
    """Same eight-block objective/domain, with independent global certificates.

    The caller still must certify the ACTUAL operator before any prediction.
    The returned growth is an absolute coefficient-space bound; each block also
    retains its original block-energy-normalized positive-growth requirement.
    """
    choices = stretch_specifications()[1:]
    matches = [spec for spec in choices if spec.name == specification.name]
    if len(matches) != 1:
        raise ValueError("one of the three canonical block-local families is required")
    canonical = matches[0]
    if (not np.array_equal(specification.modes, canonical.modes)
            or not np.array_equal(objective.modes, canonical.modes)
            or specification.spectral_norm_bound != canonical.spectral_norm_bound):
        raise ValueError("unchanged canonical block modes and original physical domain required")
    kind = "local_diagonal" if len(canonical.modes) == 8 else "local_shear"
    per_block = 1 if kind == "local_diagonal" else 2
    root = np.sqrt(objective.weights)[:, None]
    A, Y = objective.gain*root*objective.rotated_source, root*objective.active_target
    energy = float(np.sum(A*A+Y*Y))
    if not math.isfinite(energy) or energy <= 0:
        raise ValueError("positive finite Task-A objective energy required")
    budget = SearchBudget(check, seconds, max_cells)
    blocks, theta = [], np.zeros(specification.parameter_count)
    for index, (start, stop) in enumerate(BLOCKS):
        budget.diagnostics.update(completed_blocks=index, block_family=kind)
        aa, yy = A[:, start:stop], Y[:, start:stop]
        P, cross = aa.T@aa, aa.T@yy
        D = (cross+cross.T)/2
        block_energy = float(np.sum(aa*aa+yy*yy))
        if not math.isfinite(block_energy) or block_energy <= 0:
            refuse("nonpositive block objective scale", completed_blocks=index)
        scale = float(canonical.modes[per_block*index, start, start])
        result = certify_block(P/block_energy, D/block_energy,
            specification.spectral_norm_bound, scale, kind, check=check, budget=budget)
        for offset, interval in enumerate(result["coordinate_intervals"]):
            theta[per_block*index+offset] = (interval.lo+interval.hi)/(2*scale)
        blocks.append((block_energy, scale, P/block_energy, D/block_energy, result))
    C = objective.matrix(theta)
    diagnostics = objective.final_stretch_diagnostics(theta, specification.spectral_norm_bound)
    loss, gradient = objective.value_gradient(theta)
    zero = np.zeros(specification.parameter_count)
    initial_loss, _ = objective.value_gradient(zero)
    if loss > initial_loss+NUMERICAL_ATOL:
        refuse("global block fit exceeds its paired zero-stretch loss")
    tangent = objective.tangent_singular_values(theta)
    if len(tangent) != len(theta) or tangent[-1] <= RANK_RTOL*tangent[0]:
        refuse("global block fitted tangent fails the original rank threshold")
    first_order, absolute_gap, distance_squared = _i(0.), _i(0.), _i(0.)
    absolute_growth, normalized_growth, public_blocks = [], [], []
    for index, ((start, stop), (block_energy, scale, P, D, result)) in enumerate(zip(BLOCKS, blocks)):
        actual = (C[start, start],) if per_block == 1 else (C[start, start], C[start, start+1])
        block_loss = BlockLoss(P, D, specification.spectral_norm_bound, scale, kind)
        value, gap = block_loss.actual_value_gap(actual)
        first_order = first_order+block_energy*_i(gap)
        difference = max((value-result["lower_bound"]).hi,
                         (_i(result["feasible_upper"])-value).hi)
        absolute_gap = absolute_gap+block_energy*max(0., difference)
        coefficients = theta[per_block*index:per_block*(index+1)]
        anchor_distance = sum((_square(_i(float(point))-interval/scale)
            for point, interval in zip(coefficients, result["coordinate_intervals"])), _i(0.))
        coefficient_distance = (_sqrt(_nonnegative(anchor_distance))
            +_sqrt(2*_i(result["normalized_gap"])/result["coefficient_growth_lower_bound"]))
        distance_squared = distance_squared+_square(coefficient_distance)
        growth = (_i(block_energy)*result["coefficient_growth_lower_bound"]).lo
        absolute_growth.append(growth)
        normalized_growth.append(result["coefficient_growth_lower_bound"])
        public_blocks.append({key: value for key, value in result.items()
                              if key not in {"coordinate_intervals", "lower_bound", "feasible_upper"}})
        public_blocks[-1].update(objective_scale=block_energy,
            absolute_coefficient_growth_lower_bound=growth,
            absolute_objective_gap_per_block_energy=max(0., difference))
    if first_order.hi > GAP_RTOL*energy:
        refuse("global block fit exceeds the original first-order gap",
               first_order_gap_per_energy=first_order.hi/energy)
    if absolute_gap.hi > GAP_RTOL*energy:
        refuse("returned block coefficients exceed the original absolute global gap",
               absolute_objective_gap_per_energy=absolute_gap.hi/energy)
    zero_error = float(np.max(np.abs(objective.predict(objective.source, zero)
        -objective.gain*apply_m3(objective.source, objective.B, objective.O))))
    if zero_error > NUMERICAL_ATOL:
        refuse("global block zero stretch fails the original M3 replay tolerance")
    numerical = dict(C=C, coefficients=theta, O=objective.O.copy(), gain=objective.gain,
        weighted_task_a_squared_error=loss, paired_zero_stretch_loss=initial_loss,
        iterations=sum(block[4]["candidate_optimizer_iterations"] for block in blocks),
        numerical_solver="independent_block_radial_global_exclusion",
        numerical_solver_converged=True, local_optimizer_converged=False,
        zero_stretch_replay_max_abs=zero_error, locality_certified=False, **diagnostics)
    certificate = dict(method="independent_block_radial_global_exclusion", block_count=len(blocks),
        block_certificates=public_blocks, objective_scale=energy,
        first_order_gap_per_energy=first_order.hi/energy,
        first_order_gap_method="outward_eight_interval_or_disk_linear_oracles",
        absolute_objective_gap_per_energy=absolute_gap.hi/energy,
        global_coefficient_growth_lower_bound=min(absolute_growth),
        minimum_block_energy_normalized_growth=min(normalized_growth),
        coefficient_distance_upper_bound=_sqrt(_nonnegative(distance_squared)).hi,
        all_other_regions_excluded=True, unique_global_minimum_sufficient_profile=True,
        full_domain_positive_curvature_asserted=False, exact_global_optimizer=False,
        interval_arithmetic_proof=False, solution_within_frozen_gap=True,
        search_seconds=budget.snapshot()["search_seconds"], scalar_bound_cells=budget.cells)
    return numerical, certificate
