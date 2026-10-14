"""Isolated same-objective diagonal M4 global search and profile certificate.

No data reader or production activation. Restricted-box scalar duals provide
global lower bounds even when their minimizers jump. A surviving one-coordinate
profile is accepted only with unique conditional rest minima, positive profile
curvature or a strict one-sided boundary slope, quantitative full-coordinate
growth, and global exclusion of EVERY region outside the profile interval.
All arithmetic assurances match the qualified floating-point interval helpers;
this is not a formal interval proof of input moments or libm.
"""
import heapq
import math
import time

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from diagonal_m4_dual import _candidates, _curvature_lower, _growth, _value
from paired_stretch_variants import (
    CURVATURE_MARGIN, GAP_RTOL, MAX_ITERATIONS, NUMERICAL_ATOL, OPTIMIZER_FTOL,
    RANK_RTOL, M4CertificateUnverified, apply_m3, stretch_specifications,
)
from proposed_m4_certificate import _I, _i, _exp, _square, _sqrt, _nonnegative, _helmert_gram_lower


class SearchStop(M4CertificateUnverified):
    pass


def _refuse(reason, **diagnostics):
    raise M4CertificateUnverified(reason, diagnostics)


class SearchBudget:
    def __init__(self, check, seconds, max_cells):
        if not callable(check) or not math.isfinite(seconds) or seconds <= 0 or max_cells <= 0:
            raise ValueError("finite positive search budget required")
        self.check, self.deadline = check, time.monotonic()+seconds
        self.max_cells, self.cells = int(max_cells), 0
        self.started = time.monotonic()
        self.diagnostics = {}

    def tick(self):
        self.check()
        if time.monotonic() >= self.deadline:
            raise SearchStop("bounded global search time measurement reached", self.snapshot())

    def cell(self):
        self.tick()
        if self.cells >= self.max_cells:
            raise SearchStop("bounded global search cell measurement reached", self.snapshot())
        self.cells += 1

    def snapshot(self):
        return {**self.diagnostics, "restricted_dual_cells": self.cells,
                "search_seconds": time.monotonic()-self.started}


def _minimum(p, d, lam, lo, hi):
    rows = _candidates(p, d, lam, lo, hi)
    lower, upper = min(row[2].lo for row in rows), min(row[2].hi for row in rows)
    possible = [row for row in rows if row[2].lo <= upper]
    arguments = _I(min(row[1].lo for row in possible), max(row[1].hi for row in possible))
    return _I(lower, upper), arguments


def restricted_dual_bound(p, d, lower, upper, *, check, target=0.):
    """Weak-duality lower bound; no scalar uniqueness/zero dual gap assumed."""
    bound = 1+2*max(p)*math.exp(2*max(upper))+2*max(abs(d))*math.exp(max(upper))
    left, right, best = -float(bound), float(bound), None
    for iteration in range(160):
        check()
        lam = (left+right)/2
        rows = [_minimum(float(pi), float(di), lam, float(lo), float(hi))
                for pi, di, lo, hi in zip(p, d, lower, upper)]
        value = sum((row[0] for row in rows), _i(0.))-_i(lam)*target
        trace = sum((row[1] for row in rows), _i(0.))-target
        if best is None or value.lo > best["lower_bound"]:
            best = {"lower_bound": value.lo, "lambda": lam,
                    "argument_intervals": tuple(row[1] for row in rows),
                    "trace_interval": (trace.lo, trace.hi)}
        if trace.lo > 0:
            left = lam
        elif trace.hi < 0:
            right = lam
        else:
            break
        if not left < (left+right)/2 < right:
            break
    best["iterations"] = iteration+1
    return best


def _conditional_root(p, d, rho, target, check, lower=None, upper=None):
    """Strict scalar branches plus a certified trace=target root interval."""
    lower = np.full(len(p), -rho) if lower is None else lower
    upper = np.full(len(p), rho) if upper is None else upper
    bound = 1+2*max(p)*math.exp(2*rho)+2*max(abs(d))*math.exp(rho)
    left, right = -float(bound), float(bound)

    def endpoint(lam):
        rows = []
        for pi, di, lo, hi in zip(p, d, lower, upper):
            candidates = _candidates(float(pi), float(di), lam, float(lo), float(hi))
            chosen = min(candidates, key=lambda row: (row[2].lo+row[2].hi)/2)
            rows.append((chosen, all(chosen[2].hi < row[2].lo
                                    for row in candidates if row is not chosen)))
        trace = sum((row[0][1] for row in rows), _i(0.))-target
        return rows, trace, all(row[1] for row in rows)

    a, ta, sa = endpoint(left)
    b, tb, sb = endpoint(right)
    if not (sa and sb and ta.lo > 0 and tb.hi < 0):
        _refuse("conditional rest trace lacks a strict sign bracket")
    for _ in range(160):
        check()
        mid = (left+right)/2
        rows, trace, strict = endpoint(mid)
        if strict and trace.lo > 0:
            left, a, ta = mid, rows, trace
        elif strict and trace.hi < 0:
            right, b, tb = mid, rows, trace
        else:
            ql, qr = (left+mid)/2, (mid+right)/2
            al, tl, sl = endpoint(ql)
            br, tr, sr = endpoint(qr)
            changed = False
            if sl and tl.lo > 0 and ql > left:
                left, a, ta, changed = ql, al, tl, True
            if sr and tr.hi < 0 and qr < right:
                right, b, tb, changed = qr, br, tr, True
            if not changed:
                break
        if not left < (left+right)/2 < right:
            break
    switches = [(x[0], y[0]) for x, y in zip(a, b) if x[0][0] != y[0][0]]
    if switches:
        _refuse("another scalar branch switches in the conditional rest problem",
                conditional_branch_switch_count=len(switches),
                conditional_endpoint_to_endpoint_switch_count=sum(
                    "interior" not in (x[0], y[0]) for x, y in switches),
                conditional_max_switch_argument_separation=max(
                    max(abs((x[1]-y[1]).lo), abs((x[1]-y[1]).hi)) for x, y in switches),
                conditional_lambda_width=right-left,
                conditional_left_nonstrict_count=sum(not row[1] for row in a),
                conditional_right_nonstrict_count=sum(not row[1] for row in b),
                conditional_left_trace_residual_interval=(ta.lo, ta.hi),
                conditional_right_trace_residual_interval=(tb.lo, tb.hi),
                conditional_left_free_coordinate_count=sum(row[0][0] == "interior" for row in a),
                conditional_right_free_coordinate_count=sum(row[0][0] == "interior" for row in b))
    lam = _I(left, right)
    rows = _uniform_branches(p, d, rho, lam, tuple(row[0][0] for row in a), lower, upper)
    if not any(row[0] == "interior" for row in rows):
        _refuse("conditional rest profile has no free coordinate")
    return lam, rows


def _uniform_branches(p, d, rho, lam, kinds, lower=None, upper=None):
    lower = np.full(len(p), -rho) if lower is None else lower
    upper = np.full(len(p), rho) if upper is None else upper
    rows = []
    for pi, di, kind, lo, hi in zip(p, d, kinds, lower, upper):
        candidates = _candidates(float(pi), float(di), lam, float(lo), float(hi))
        selected = [row for row in candidates if row[0] == kind]
        if len(selected) != 1:
            _refuse("conditional branch is not present over the whole interval")
        chosen = selected[0]
        if kind == "interior":
            if not lo < chosen[1].lo <= chosen[1].hi < hi:
                _refuse("conditional interior branch reaches an active-set transition")
            if _curvature_lower(float(pi), float(di), chosen[1].lo, chosen[1].hi) <= 0:
                _refuse("conditional free curvature is not strictly positive")
            # Compare branches at the SAME lambda. Separate enclosures of
            # their values lose that dependence and can overlap even for a
            # strictly convex scalar on a narrow local interval. Uniform
            # positive curvature and endpoint derivative signs instead prove
            # one unique interior minimum for every lambda in the interval.
            separated = (_curvature_lower(float(pi), float(di), float(lo), float(hi)) > 0
                and (_derivative(float(pi), float(di), _i(float(lo)))+lam).hi < 0
                and (_derivative(float(pi), float(di), _i(float(hi)))+lam).lo > 0)
        else:
            # Positive endpoint growth proves the same strict global scalar
            # minimum over this entire local interval, including concave cases.
            try:
                separated = _local_growth(float(pi), float(di), lam, float(lo), float(hi), chosen) > 0
            except M4CertificateUnverified:
                separated = False
        if not separated and any(chosen[2].hi >= row[2].lo for row in candidates if row[0] != kind):
            _refuse("conditional global-minimum separation is unresolved")
        rows.append(chosen)
    return tuple(rows)


def _derivative(p, d, c):
    z = _exp(c)
    return 2*p*z*z-2*d*z


def _profile_point(p, d, rho, axis, t, check, lower=None, upper=None):
    rest = np.arange(len(p)) != axis
    lam, rows = _conditional_root(p[rest], d[rest], rho, -t, check,
        None if lower is None else lower[rest], None if upper is None else upper[rest])
    value = _value(float(p[axis]), float(d[axis]), _i(0.), _i(t))
    for pi, di, row in zip(p[rest], d[rest], rows):
        value = value+_value(float(pi), float(di), _i(0.), row[1])
    derivative = _derivative(float(p[axis]), float(d[axis]), _i(t))+lam
    return {"lambda": lam, "rows": rows, "value": value, "derivative": derivative, "t": t}


def _local_growth(p, d, lam, lo, hi, optimum):
    """Uniform scalar growth on a restricted box, including physical endpoints."""
    kind, center, _ = optimum
    curvature = _curvature_lower(p, d, lo, hi)
    if kind == "interior":
        return curvature
    slope = _derivative(p, d, center)+lam
    inward = slope if kind == "lower" else -slope
    if inward.lo <= 0:
        _refuse("restricted endpoint lacks a strictly inward objective slope")
    delta = _I(lo, hi)-center
    radius = max(abs(delta.lo), abs(delta.hi))
    return (_i(curvature)+2*_i(inward.lo)/radius).lo


def profile_region(p, d, rho, axis, lo, hi, *, check, lower=None, upper=None):
    """An exact conditional profile; cross-coordinate growth is kept explicit."""
    restricted = lower is not None
    lower = np.full(len(p), -rho) if lower is None else lower
    upper = np.full(len(p), rho) if upper is None else upper
    rest = np.arange(len(p)) != axis
    a = _profile_point(p, d, rho, axis, lo, check, lower, upper)
    b = _profile_point(p, d, rho, axis, hi, check, lower, upper)
    kinds = tuple(row[0] for row in a["rows"])
    if kinds != tuple(row[0] for row in b["rows"]):
        _refuse("profile interval crosses a conditional active-set transition")
    lam = _I(a["lambda"].lo, b["lambda"].hi)
    rows = _uniform_branches(p[rest], d[rest], rho, lam, kinds, lower[rest], upper[rest])
    inverse_sum = _i(0.)
    for pi, di, row in zip(p[rest], d[rest], rows):
        if row[0] == "interior":
            h = _curvature_lower(float(pi), float(di), row[1].lo, row[1].hi)
            inverse_sum = inverse_sum+_i(1.)/h
    if inverse_sum.lo <= 0:
        _refuse("profile has no smooth free rest direction")
    q_curvature = (_i(_curvature_lower(float(p[axis]), float(d[axis]), lo, hi))
                   +_i(1.)/inverse_sum).lo
    q_slope = _derivative(float(p[axis]), float(d[axis]), _I(lo, hi))+lam
    if q_curvature > 3*CURVATURE_MARGIN:
        kind, q_growth = "strictly_convex_profile", q_curvature
    elif lo == -rho and q_slope.lo > 0:
        kind, q_growth = "strict_lower_boundary_profile", (2*_i(q_slope.lo)/(hi-lo)).lo
    elif hi == rho and q_slope.hi < 0:
        kind, q_growth = "strict_upper_boundary_profile", (-2*_i(q_slope.hi)/(hi-lo)).lo
    else:
        _refuse("profile has neither certified positive curvature nor a one-sided boundary minimum",
                profile_curvature_lower_bound=q_curvature)
    rest_growth, cells = math.inf, 0
    for pi, di, row, rlo, rhi in zip(p[rest], d[rest], rows, lower[rest], upper[rest]):
        if restricted:
            growth, count = _local_growth(float(pi), float(di), lam, float(rlo), float(rhi), row), 1
        else:
            growth, count = _growth(float(pi), float(di), lam, rho, row,
                                   3*CURVATURE_MARGIN, check, 4095)
        rest_growth, cells = min(rest_growth, growth), cells+count
    # Conditional derivative dr_i*/dt=-(1/h_i)/sum_free(1/h_i), so its Euclidean
    # norm is <=1. Thus ||(dr,dt)||² <= 2||dr-dr*(t)||²+3|dt|². A profile
    # curvature is never mislabeled as a full-coordinate growth coefficient.
    full_growth = min(rest_growth/2, q_growth/3)
    if full_growth <= CURVATURE_MARGIN:
        _refuse("full-coordinate growth is below the retained numerical margin")
    return {"lo": lo, "hi": hi, "kind": kind, "profile_curvature": q_curvature,
            "profile_growth": q_growth, "rest_growth": rest_growth,
            "full_growth": full_growth, "rest_growth_cells": cells,
            "left_point": a, "right_point": b, "lower": lower, "upper": upper}


def _quadratic_minimum(g, mu, lo, hi):
    root = -_i(g)/mu
    x = _I(min(hi, max(lo, root.lo)), min(hi, max(lo, root.hi)))
    return (_i(g)*x+_i(mu)*_square(x)/2).lo


def _profile_optimum(p, d, rho, axis, region, check):
    lo, hi = region["lo"], region["hi"]
    left, right = region["left_point"], region["right_point"]
    if region["kind"] == "strict_lower_boundary_profile" or left["derivative"].lo >= 0:
        point = left
    elif region["kind"] == "strict_upper_boundary_profile" or right["derivative"].hi <= 0:
        point = right
    else:
        point = None
        for _ in range(64):
            check()
            mid = (lo+hi)/2
            point = _profile_point(p, d, rho, axis, mid, check, region["lower"], region["upper"])
            if point["derivative"].lo > 0:
                hi = mid
            elif point["derivative"].hi < 0:
                lo = mid
            else:
                break
            if not lo < (lo+hi)/2 < hi:
                break
    if region["kind"] != "strictly_convex_profile":
        lower = point["value"].lo
    else:
        dx = _I(region["lo"], region["hi"])-point["t"]
        terms = [0.]
        if dx.lo < 0:
            terms.append(_quadratic_minimum(point["derivative"].hi,
                                           region["profile_curvature"], dx.lo, min(0., dx.hi)))
        if dx.hi > 0:
            terms.append(_quadratic_minimum(point["derivative"].lo,
                                           region["profile_curvature"], max(0., dx.lo), dx.hi))
        lower = (_i(point["value"].lo)+min(terms)).lo
    return point, lower


def _primal(p, d, H, rho, check):
    def objective(theta):
        check()
        z = np.exp(H@theta)
        return float(np.sum(p*z*z-2*d*z)), H.T@(2*p*z*z-2*d*z)
    result = minimize(objective, np.zeros(H.shape[1]), jac=True, method="SLSQP",
                      constraints=LinearConstraint(H, -rho, rho),
                      options={"ftol": OPTIMIZER_FTOL, "maxiter": MAX_ITERATIONS})
    if not result.success:
        _refuse("normalized linear-constraint primal solve did not converge",
                primal_message=str(result.message), primal_iterations=int(result.nit))
    if np.max(np.abs(H@result.x)) > rho+NUMERICAL_ATOL:
        _refuse("normalized primal solve exceeded the original feasibility tolerance")
    return H@result.x, int(result.nit)


def _exterior_slabs(lower, upper, inner_lower, inner_upper):
    """Cover the entire exterior of the final inner box, retaining boundaries."""
    a, b = lower.copy(), upper.copy()
    for i in range(len(a)):
        if a[i] < inner_lower[i]:
            edge = b.copy()
            edge[i] = inner_lower[i]
            yield a.copy(), edge
        if inner_upper[i] < b[i]:
            edge = a.copy()
            edge[i] = inner_upper[i]
            yield edge, b.copy()
        a[i], b[i] = inner_lower[i], inner_upper[i]


def _branch_coordinate(p, d, lower, upper, bound):
    """Refine an actual scalar ambiguity, else its possible nonconvex gap."""
    widths = np.array([row.hi-row.lo for row in bound["argument_intervals"]])
    if np.max(widths) > 1e-8:
        return int(np.argmax(widths))
    gaps = np.array([max(0., -_curvature_lower(float(pi), float(di), float(lo), float(hi)))
                     *(hi-lo)**2 for pi, di, lo, hi in zip(p, d, lower, upper)])
    return int(np.argmax(gaps if np.max(gaps) > 0 else upper-lower))


def moment_global_search(p, d, rho, H, *, check, seconds=300., max_cells=4096):
    """Return a complete numerical global certificate or an honest bounded stop."""
    p, d, H = np.asarray(p, float), np.asarray(d, float), np.asarray(H, float)
    n = len(p)
    if (d.shape != p.shape or H.shape != (n, n-1) or n != 16
            or np.any(p <= 0) or not np.isfinite(np.r_[p, d, H.ravel(), rho]).all()):
        raise ValueError("finite canonical 16-coordinate normalized moments required")
    budget = SearchBudget(check, seconds, max_cells)
    lower, upper = np.full(n, -rho), np.full(n, rho)
    budget.cell()
    initial = restricted_dual_bound(p, d, lower, upper, check=budget.tick)
    widths = np.array([row.hi-row.lo for row in initial["argument_intervals"]])
    axis = int(np.argmax(widths))
    budget.diagnostics.update(initial_dual_lower_bound=initial["lower_bound"],
                              initial_ambiguous_coordinate_count=int(np.sum(widths > 1e-8)))
    primal, iterations = _primal(p, d, H, rho, budget.tick)
    z = np.exp(primal)
    free = np.flatnonzero(np.abs(primal) < rho-NUMERICAL_ATOL)
    primal_curvature = 4*p*z*z-2*d*z
    dual_axis = axis
    if len(free):
        # A free coordinate with negative scalar curvature cannot be an
        # interior minimum of f_i(c)+lambda*c for ANY lambda. Leaving it in
        # the scalar-dual rest problem therefore cannot certify this primal
        # candidate. Prefer the least-curved free coordinate for the explicit
        # profile. This only selects a search coordinate: all existing global
        # exclusion, conditional-minimum and growth checks remain mandatory.
        axis = int(free[np.argmin(primal_curvature[free])])
    budget.diagnostics.update(primal_iterations=iterations,
        primal_normalized_objective=float(np.sum(p*z*z-2*d*z)),
        primal_trace_abs=float(abs(np.sum(primal))),
        primal_box_excess=float(max(0., np.max(np.abs(primal))-rho)),
        primal_active_lower_count=int(np.sum(np.abs(primal+rho) <= NUMERICAL_ATOL)),
        primal_active_upper_count=int(np.sum(np.abs(primal-rho) <= NUMERICAL_ATOL)),
        primal_free_coordinate_count=int(len(free)),
        primal_free_nonpositive_curvature_count=int(np.sum(primal_curvature[free] <= 0)),
        initial_dual_axis_is_primal_free=bool(dual_axis in free),
        profile_axis_is_primal_free=bool(axis in free),
        profile_axis_changed_from_dual_ambiguity=bool(axis != dual_axis),
        profile_axis_primal_curvature=float(primal_curvature[axis]),
        profile_axis_rule="lowest_curvature_primal_free" if len(free) else "largest_dual_ambiguity")
    center = min(rho, max(-rho, float(primal[axis])))
    region, last_reason = None, "not_attempted"
    for level in range(20):
        budget.tick()
        radius = 2*rho/(2**level)
        lo, hi = max(-rho, center-radius), min(rho, center+radius)
        # The rest-free coordinate may move by as much as the profile axis;
        # a twice-wider rest box prevents an artificial edge at both profile
        # endpoints. Every actual bound is fixed for this whole profile.
        local_lower = np.maximum(-rho, primal-2*radius)
        local_upper = np.minimum(rho, primal+2*radius)
        local_lower[axis], local_upper[axis] = lo, hi
        try:
            region = profile_region(p, d, rho, axis, lo, hi, check=budget.tick,
                                    lower=local_lower, upper=local_upper)
            break
        except SearchStop:
            raise
        except M4CertificateUnverified as error:
            last_reason = str(error)
            failure = dict(reason=last_reason, diagnostics=error.diagnostics,
                           profile_width=hi-lo, attempt=level+1)
            budget.diagnostics.setdefault("first_profile_failure", failure)
            budget.diagnostics["last_profile_failure"] = failure
    if region is None:
        _refuse("no certified unique conditional profile around the primal candidate",
                **budget.snapshot(), profile_reason=last_reason)
    budget.diagnostics.update(profile_kind=region["kind"], profile_width=region["hi"]-region["lo"],
        profile_curvature_lower_bound=region["profile_curvature"],
        conditional_rest_growth_lower_bound=region["rest_growth"],
        profile_full_coordinate_growth_lower_bound=region["full_growth"],
        profile_scope="final_restricted_coordinate_box",
        local_box_min_width=float(np.min(region["upper"]-region["lower"])),
        local_box_max_width=float(np.max(region["upper"]-region["lower"])))
    point, profile_lower = _profile_optimum(p, d, rho, axis, region, budget.tick)
    feasible_upper = point["value"].hi
    diameter_squared = (_square(2*_i(rho))*n).hi
    threshold = (_i(feasible_upper)+CURVATURE_MARGIN*_i(diameter_squared)/2).hi
    queue, serial, pruned_lower = [], 0, math.inf

    def push(a, b):
        nonlocal serial, pruned_lower
        budget.cell()
        # Sum-zero infeasibility is checked with outward arithmetic. No cell
        # is removed merely because its width or apparent gap is small.
        if sum((_i(float(x)) for x in a), _i(0.)).lo > 0 or sum(
                (_i(float(x)) for x in b), _i(0.)).hi < 0:
            return
        bound = restricted_dual_bound(p, d, a, b, check=budget.tick)
        if bound["lower_bound"] > threshold:
            pruned_lower = min(pruned_lower, bound["lower_bound"])
        else:
            split = _branch_coordinate(p, d, a, b, bound)
            heapq.heappush(queue, (bound["lower_bound"], serial, a, b, split))
            serial += 1

    slabs = tuple(_exterior_slabs(lower, upper, region["lower"], region["upper"]))
    budget.diagnostics["initial_exterior_slabs"] = len(slabs)
    split_counts = np.zeros(n, dtype=int)
    for a, b in slabs:
        push(a, b)
    while queue:
        budget.tick()
        best, _, a, b, split = heapq.heappop(queue)
        budget.diagnostics.update(best_remaining_global_gap=max(0., feasible_upper-min(best, profile_lower)),
                                  remaining_cells=len(queue)+1,
                                  branched_coordinate_count=int(np.sum(split_counts > 0)))
        mid = (a[split]+b[split])/2
        if not a[split] < mid < b[split]:
            _refuse("unexcluded region reached floating-point subdivision resolution", **budget.snapshot())
        split_counts[split] += 1
        left_upper, right_lower = b.copy(), a.copy()
        left_upper[split], right_lower[split] = mid, mid
        push(a, left_upper)
        push(right_lower, b)
    budget.diagnostics["remaining_cells"] = 0
    budget.diagnostics["branched_coordinate_count"] = int(np.sum(split_counts > 0))
    global_lower = min(profile_lower, pruned_lower)
    gap = (_i(feasible_upper)-global_lower).hi
    outside_growth = (2*(_i(pruned_lower)-feasible_upper)/diameter_squared).lo if math.isfinite(pruned_lower) else math.inf
    growth = min(region["full_growth"], outside_growth)
    if gap > GAP_RTOL or growth <= CURVATURE_MARGIN:
        _refuse("global objective gap or full-coordinate growth remains unverified",
                **budget.snapshot(), global_gap_upper_bound=gap, full_coordinate_growth=growth)
    coordinates = []
    rest_rows = iter(point["rows"])
    for i in range(n):
        coordinates.append(_i(point["t"]) if i == axis else next(rest_rows)[1])
    return {"coordinate_intervals": tuple(coordinates), "global_lower_bound": global_lower,
            "feasible_upper_bound": feasible_upper, "global_gap_upper_bound": gap,
            "global_coordinate_growth_lower_bound": growth,
            "profile_kind": region["kind"], "profile_curvature_lower_bound": region["profile_curvature"],
            "rest_growth_lower_bound": region["rest_growth"],
            "profile_full_coordinate_growth_lower_bound": region["full_growth"],
            "all_other_regions_excluded": True, "unique_global_minimum_sufficient_profile": True,
            "primal_iterations": iterations, **budget.snapshot()}


def _diagonal_first_order_gap(p, d, c, rho):
    """Outward upper bound for the same trace-zero-box linear optimality gap.

    For 16 equally bounded coordinates, a linear minimum assigns +rho to the
    eight smallest gradient components and -rho to the other eight. Bound
    rounding of the gradient separately using the rho-L1 support-function
    bound; there is no numerical LP tolerance or subtraction-based sign gate.
    """
    if len(p) != 16 or len(d) != 16 or len(c) != 16:
        raise ValueError("the canonical sixteen-coordinate box is required")
    gradients = [_derivative(float(pi), float(di), _i(float(ci)))
                 for pi, di, ci in zip(p, d, c)]
    middle = np.array([(row.lo+row.hi)/2 for row in gradients])
    vertex = np.full(16, -rho)
    vertex[np.argsort(middle)[:8]] = rho
    candidate, minimum, error = _i(0.), _i(0.), _i(0.)
    for row, mid, ci, vi in zip(gradients, middle, c, vertex):
        candidate = candidate+row*_i(float(ci))
        minimum = minimum+_i(float(mid))*_i(float(vi))
        delta = row-_i(float(mid))
        error = error+max(abs(delta.lo), abs(delta.hi))
    minimum_lower = minimum-_i(rho)*error
    return max(0., (candidate-minimum_lower).hi)


def fit_diagonal_global(objective, specification, *, check, seconds=300., max_cells=4096):
    canonical = stretch_specifications()[0]
    if (specification.name != canonical.name or not np.array_equal(specification.modes, canonical.modes)
            or not np.array_equal(objective.modes, canonical.modes)
            or specification.spectral_norm_bound != canonical.spectral_norm_bound):
        raise ValueError("the canonical diagonal objective and original domain are required")
    root = np.sqrt(objective.weights)[:, None]
    A, Y = objective.gain*root*objective.rotated_source, root*objective.active_target
    p, d = np.sum(A*A, axis=0), np.sum(A*Y, axis=0)
    energy = float(np.sum(A*A+Y*Y))
    H = np.diagonal(specification.modes, axis1=1, axis2=2).T
    result = moment_global_search(p/energy, d/energy, specification.spectral_norm_bound, H,
                                 check=check, seconds=seconds, max_cells=max_cells)
    middle = np.array([(row.lo+row.hi)/2 for row in result["coordinate_intervals"]])
    theta = np.linalg.solve(H.T@H, H.T@middle)
    C = objective.matrix(theta)
    diagnostics = objective.final_stretch_diagnostics(theta, specification.spectral_norm_bound)
    loss, gradient = objective.value_gradient(theta)
    zero = np.zeros(specification.parameter_count)
    initial_loss, _ = objective.value_gradient(zero)
    if loss > initial_loss+NUMERICAL_ATOL:
        _refuse("global diagonal fit exceeds its paired zero-stretch loss")
    tangent = objective.tangent_singular_values(theta)
    if len(tangent) != len(theta) or tangent[-1] <= RANK_RTOL*tangent[0]:
        _refuse("global diagonal fitted tangent fails the original rank threshold")
    stationary_gap = _diagonal_first_order_gap(p, d, np.diag(C), specification.spectral_norm_bound)
    if stationary_gap > GAP_RTOL*energy:
        _refuse("global diagonal fit exceeds the original first-order gap",
                first_order_gap_per_energy=stationary_gap/energy)
    candidate_value, anchor_distance = _i(0.), _i(0.)
    for pi, di, ci, anchor in zip(p/energy, d/energy, np.diag(C), result["coordinate_intervals"]):
        candidate_value = candidate_value+_value(float(pi), float(di), _i(0.), _i(float(ci)))
        anchor_distance = anchor_distance+_square(_i(float(ci))-anchor)
    absolute_gap = max((candidate_value-result["global_lower_bound"]).hi,
                       (_i(result["feasible_upper_bound"])-candidate_value).hi)
    if absolute_gap > GAP_RTOL:
        _refuse("returned coefficient objective exceeds the original absolute global gap",
                absolute_objective_gap_per_energy=absolute_gap)
    gram = _helmert_gram_lower(specification.modes)
    distance = (_sqrt(_nonnegative(anchor_distance)/gram)
                +_sqrt(_i(2*result["global_gap_upper_bound"])/(result["global_coordinate_growth_lower_bound"]*gram))).hi
    zero_error = float(np.max(np.abs(objective.predict(objective.source, zero)
                     -objective.gain*apply_m3(objective.source, objective.B, objective.O))))
    if zero_error > NUMERICAL_ATOL:
        _refuse("global diagonal zero stretch fails the original M3 replay tolerance")
    numerical = dict(C=C, coefficients=theta, O=objective.O.copy(), gain=objective.gain,
        weighted_task_a_squared_error=loss, paired_zero_stretch_loss=initial_loss,
        iterations=result["primal_iterations"], numerical_solver="normalized_linear_primal_and_global_profile",
        numerical_solver_converged=True, local_optimizer_converged=False,
        zero_stretch_replay_max_abs=zero_error, locality_certified=False, **diagnostics)
    public = {key: value for key, value in result.items() if key != "coordinate_intervals"}
    certificate = dict(**public, objective_scale=energy, first_order_gap_per_energy=stationary_gap/energy,
        first_order_gap_method="outward_balanced_box_linear_oracle",
        absolute_objective_gap_per_energy=absolute_gap, coefficient_distance_upper_bound=distance,
        full_domain_positive_curvature_asserted=False, exact_global_optimizer=False,
        interval_arithmetic_proof=False, solution_within_frozen_gap=True)
    return numerical, certificate
