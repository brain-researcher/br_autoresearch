"""Adopted supplemental M4 curvature certificate; no data reader.

Adopted under scientist_amendment_20261013_m4_certificate on October 13, 2026.
This module changes no objective, mode, domain or tolerance.
See ../proposed_m4_certificate_followup.md for the fixed specification.

Bounds cover whole intervals/disks, not just evaluation points. Scalar
arithmetic is expanded by nextafter; exp is expanded by eight ULPs, retaining
the episode's qualified floating-point assurance standard, NOT asserting an
interval-arithmetic proof for libm, prior moment calculations or the optimizer.
The finite limits below are prospective, unmeasured limits, never retry rules.
"""
from dataclasses import dataclass
import math
import time

import numpy as np

from paired_stretch_variants import (
    BLOCKS, CURVATURE_MARGIN, M4CertificateUnverified, block_curvature_bounds,
    stretch_specifications,
)


SERIES_DEGREE = 18


@dataclass(frozen=True)
class CoverLimits:
    max_cells_per_block: int = 16383
    max_depth: int = 12
    max_seconds_total: float = 30.0

    def __post_init__(self):
        for value in (self.max_cells_per_block, self.max_depth):
            if type(value) is not int or value <= 0:
                raise ValueError("positive finite subdivision limits required")
        if not math.isfinite(self.max_seconds_total) or self.max_seconds_total <= 0:
            raise ValueError("positive finite elapsed-time limit required")


def _down(x):
    return math.nextafter(x, -math.inf)


def _up(x):
    return math.nextafter(x, math.inf)


@dataclass(frozen=True)
class _I:
    lo: float
    hi: float

    def __post_init__(self):
        if not (math.isfinite(self.lo) and math.isfinite(self.hi) and self.lo <= self.hi):
            raise ArithmeticError("nonfinite or reversed numerical enclosure")

    def __add__(self, other):
        other = _i(other)
        return _I(_down(self.lo + other.lo), _up(self.hi + other.hi))

    __radd__ = __add__

    def __neg__(self):
        return _I(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + (-_i(other))

    def __rsub__(self, other):
        return _i(other) + (-self)

    def __mul__(self, other):
        other = _i(other)
        products = (self.lo*other.lo, self.lo*other.hi,
                    self.hi*other.lo, self.hi*other.hi)
        return _I(_down(min(products)), _up(max(products)))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = _i(other)
        if other.lo <= 0 <= other.hi:
            raise ArithmeticError("numerical enclosure divisor includes zero")
        return self * _I(_down(1/other.hi), _up(1/other.lo))

    def __pow__(self, exponent):
        if type(exponent) is not int or exponent < 0:
            raise ValueError("nonnegative integer power required")
        result = _i(1.)
        for _ in range(exponent):
            result = result*self
        return result


def _i(x):
    if isinstance(x, _I):
        return x
    value = float(x)
    # Large factorial integers need to include integer-to-double rounding.
    if isinstance(x, int) and abs(x) > 2**53:
        return _I(_down(value), _up(value))
    return _I(value, value)


def _square(x):
    x = _i(x)
    lower = 0. if x.lo <= 0 <= x.hi else _down(min(x.lo*x.lo, x.hi*x.hi))
    return _I(max(0., lower), _up(max(x.lo*x.lo, x.hi*x.hi)))


def _sqrt(x):
    x = _i(x)
    if x.lo < 0:
        raise ArithmeticError("negative square-root enclosure")
    return _I(max(0., _down(math.sqrt(x.lo))), _up(math.sqrt(x.hi)))


def _exp(x):
    x = _i(x)
    lower, upper = math.exp(x.lo), math.exp(x.hi)
    for _ in range(8):
        lower, upper = _down(lower), _up(upper)
    return _I(lower, upper)


def _abs_upper(x):
    x = _i(x)
    return max(abs(x.lo), abs(x.hi))


def _nonnegative(x):
    """Restrict a mathematically nonnegative quantity after outward rounding."""
    return _I(max(0., x.lo), max(0., x.hi))


def _min_eigenvalue(a, b, c):
    radius = _sqrt(_nonnegative(_square(a-c) + 4*_square(b)))
    return (a+c-radius)/2


def _radial_derivative(q, k, odd, derivative):
    """Complete cosh(k sqrt(q)) or sinh(k sqrt(q))/sqrt(q) derivative.

    Fixed polynomial plus positive geometric tail. This includes q=0 without
    division by sqrt(q). For n >= N+1 the term ratio decreases, so its first
    upper bound controls the entire omitted derivative tail.
    """
    if derivative not in (1, 2) or k not in (1, 2) or q.lo < 0:
        raise ValueError("fixed radial derivative domain required")

    def coefficient(n):
        falling = math.factorial(n)//math.factorial(n-derivative)
        return _i(k**(2*n+odd))*falling/_i(math.factorial(2*n+odd))

    value = _i(0.)
    for n in range(SERIES_DEGREE, derivative-1, -1):
        value = value*q + coefficient(n)
    n = SERIES_DEGREE+1
    first = coefficient(n)*(q**(n-derivative))
    ratio = (_i(k*k)*q*(n+1)/(n+1-derivative)
             /((2*n+odd+2)*(2*n+odd+1)))
    if ratio.hi >= 1:
        raise ArithmeticError("radial derivative tail is unresolved")
    tail = first/(_i(1.)-ratio)
    return value + _I(0., max(0., tail.hi))


def _shear_hessian(P, D, theta, s):
    """Entry enclosures in the actual stored two-coefficient coordinates."""
    x = tuple(_i(float(v)) for v in theta)
    s2 = _i(s)*s
    q = _nonnegative(s2*(_square(x[0])+_square(x[1])))
    result = [[_i(0.) for _ in range(2)] for _ in range(2)]
    for matrix, k, multiplier in ((P, 2, 1), (D, 1, -2)):
        trace = _i(matrix[0, 0])+matrix[1, 1]
        vector = (_i(s)*(_i(matrix[0, 0])-matrix[1, 1]),
                  _i(2.)*s*matrix[0, 1])
        linear = vector[0]*x[0]+vector[1]*x[1]
        f1, f2 = (_radial_derivative(q, k, 0, d) for d in (1, 2))
        g1, g2 = (_radial_derivative(q, k, 1, d) for d in (1, 2))
        for i in range(2):
            for j in range(i, 2):
                scalar = 4*s2*s2*f2*x[i]*x[j]
                mixed = 4*s2*s2*g2*x[i]*x[j]
                if i == j:
                    scalar, mixed = scalar+2*s2*f1, mixed+2*s2*g1
                entry = (trace*scalar + linear*mixed
                         + 2*s2*g1*(vector[i]*x[j]+x[i]*vector[j]))
                result[i][j] = result[i][j]+multiplier*entry
                result[j][i] = result[i][j]
    return result


def _shear_lipschitz(P, D, radius, s):
    # radius encloses the entire coefficient square, including centers outside
    # the disk. (sqrt(2)*s)^3 retains actual, not idealized mode normalization.
    trace_p = _i(P[0, 0])+P[1, 1]
    trace_d = _i(D[0, 0])+D[1, 1]
    spread = _sqrt(_nonnegative(_square(_i(D[0, 0])-D[1, 1])
                                + 4*_square(_i(D[0, 1]))))
    nuclear_d = max(_abs_upper(trace_d), spread.hi)
    mode_norm = _sqrt(_nonnegative(2*_i(s)*s))
    return ((8*_exp(2*radius)*trace_p + 2*_exp(radius)*nuclear_d)
            *mode_norm**3).hi


def _diagonal_minima(P, D, rho):
    lower, upper = _exp(-rho).lo, _exp(rho).hi
    values = []
    for i in range(2):
        p, d = _i(P[i, i]), _i(D[i, i])
        if p.lo <= 0:
            raise ValueError("positive source diagonal required")
        vertex = d/(4*p)
        z = _I(min(upper, max(lower, vertex.lo)),
               min(upper, max(lower, vertex.hi)))
        values.append((4*p*z*z-2*d*z).lo)
    return min(values)


def _local_diagonal_interval(P, D, lo, hi, s):
    t = _I(lo, hi)
    return (4*_i(s)*s*(_i(P[0, 0])*_exp(2*t)+_i(P[1, 1])*_exp(-2*t))
            -2*_i(s)*s*(_i(D[0, 0])*_exp(t)+_i(D[1, 1])*_exp(-t)))


class _Budget:
    def __init__(self, limits, check):
        if not isinstance(limits, CoverLimits) or not callable(check):
            raise ValueError("finite limits and caller resource check required")
        self.limits, self.check = limits, check
        self.deadline = time.monotonic()+limits.max_seconds_total

    def tick(self):
        self.check()
        if time.monotonic() >= self.deadline:
            raise M4CertificateUnverified("prospective curvature time cap reached", {})


def _cover_bound(P, D, rho, s, margin, kind, budget):
    """Deterministic depth-first dyadic cover; every unresolved cell refuses."""
    if kind == "local_diagonal":
        stack, lipschitz = [((-rho, rho), 0)], None
    else:
        radius = (_i(rho)/s).hi
        outer_spectral = (_sqrt(_i(2.))*s*radius).hi
        lipschitz = _shear_lipschitz(P, D, outer_spectral, s)
        stack = [((-radius, radius, -radius, radius), 0)]
        radius_squared = _square(_i(radius)).hi
    visited = accepted = discarded = 0
    minimum = math.inf
    while stack:
        budget.tick()
        if visited >= budget.limits.max_cells_per_block:
            raise M4CertificateUnverified("prospective curvature cell cap reached",
                                         {"visited_cells": visited, "kind": kind})
        cell, depth = stack.pop()
        visited += 1
        if kind == "local_diagonal":
            lo, hi = cell
            bound = _local_diagonal_interval(P, D, lo, hi, s).lo
        else:
            a, b, c, d = cell
            nearest_x = 0. if a <= 0 <= b else min(abs(a), abs(b))
            nearest_y = 0. if c <= 0 <= d else min(abs(c), abs(d))
            nearest_squared = (_square(_i(nearest_x))+_square(_i(nearest_y))).lo
            if nearest_squared > radius_squared:
                discarded += 1
                continue
            x, y = (a+b)/2, (c+d)/2
            hessian = _shear_hessian(P, D, (x, y), s)
            center = _min_eigenvalue(hessian[0][0], hessian[0][1], hessian[1][1])
            dx = max(_abs_upper(_i(x)-a), _abs_upper(_i(b)-x))
            dy = max(_abs_upper(_i(y)-c), _abs_upper(_i(d)-y))
            delta = _sqrt(_nonnegative(_square(_i(dx))+_square(_i(dy)))).hi
            bound = (center-_i(lipschitz)*delta).lo
        if bound > margin:
            minimum = min(minimum, bound)
            accepted += 1
            continue
        if depth >= budget.limits.max_depth:
            raise M4CertificateUnverified("prospective curvature depth cap reached",
                {"visited_cells": visited, "kind": kind, "unresolved_lower_bound": bound})
        if kind == "local_diagonal":
            mid = (lo+hi)/2
            if not lo < mid < hi:
                raise M4CertificateUnverified("curvature subdivision stagnated", {})
            stack.extend((((mid, hi), depth+1), ((lo, mid), depth+1)))
        else:
            if not (a < x < b and c < y < d):
                raise M4CertificateUnverified("curvature subdivision stagnated", {})
            # Reverse push makes the lower-left child the first visited.
            stack.extend(((child, depth+1) for child in
                          ((x, b, y, d), (a, x, y, d), (x, b, c, y), (a, x, c, y))))
    if not accepted or not math.isfinite(minimum):
        raise M4CertificateUnverified("empty prospective curvature cover", {})
    return {"hessian_lower_bound": minimum, "visited_cells": visited,
            "certified_cells": accepted, "disjoint_cells": discarded,
            "complete_domain_cover": True, "hessian_lipschitz_upper_bound": lipschitz}


def _helmert_gram_lower(modes):
    H = np.diagonal(modes, axis1=1, axis2=2).T
    rows = []
    for i in range(H.shape[1]):
        diagonal, radius = None, _i(0.)
        for j in range(H.shape[1]):
            dot = sum((_i(H[k, i])*H[k, j] for k in range(16)), _i(0.))
            if i == j:
                diagonal = dot
            else:
                radius = radius+_abs_upper(dot)
        rows.append((diagonal-radius).lo)
    lower = min(rows)
    if lower <= 0:
        raise ValueError("actual Helmert coefficient Gram is not certified positive")
    return lower


def proposed_curvature_certificate(objective, specification, *, check,
                                   limits=CoverLimits()):
    """Array-only alternative for all eight blocks; never fits or reads data.

    Qualification and explicit adoption are prerequisites to any production
    use. Each already-passing original block short-circuits the new calculation;
    a finite-cover cap can never invalidate that block. Otherwise acceptance
    uses max(original, alternative), preserving both diagnostic lower bounds.
    Canonical mode and bound equality prevents this helper from certifying
    a different parameterization. Input moments reuse the existing qualified
    weighted source/target construction; no new empirical input is requested.
    """
    choices = stretch_specifications()
    matches = [i for i, item in enumerate(choices) if item.name == specification.name]
    if len(matches) != 1:
        raise ValueError("unknown fixed M4 variant")
    index = matches[0]
    canonical = choices[index]
    if (specification.spectral_norm_bound != canonical.spectral_norm_bound
            or not np.array_equal(specification.modes, canonical.modes)
            or not np.array_equal(objective.modes, specification.modes)):
        raise ValueError("actual modes and spectral bound must remain canonical and paired")
    budget = _Budget(limits, check)
    budget.tick()
    root_weight = np.sqrt(objective.weights)[:, None]
    A, Y = root_weight*objective.rotated_source, root_weight*objective.active_target
    if (A.ndim != 2 or A.shape[1] != 16 or Y.shape != A.shape
            or not np.isfinite(A).all() or not np.isfinite(Y).all()
            or not math.isfinite(objective.gain) or objective.gain <= 0):
        raise ValueError("qualified finite paired weighted Task-A source/target required")
    rho = specification.spectral_norm_bound
    gram_lower = None
    blocks = []
    for block, (start, stop) in enumerate(BLOCKS):
        budget.tick()
        a, y = A[:, start:stop], Y[:, start:stop]
        original = block_curvature_bounds(a, y, global_gain=objective.gain,
                                         spectral_norm_bound=rho)
        if original["hessian_lower_bound"] > original["required_positive_margin"]:
            blocks.append({**original, "method": "unchanged_original_bound",
                           "alternative_evaluated": False})
            continue
        P = objective.gain**2*(a.T@a)
        cross = a.T@y
        D = objective.gain*(cross+cross.T)/2
        energy = float(np.sum((objective.gain*a)**2)+np.sum(y**2))
        if (not np.isfinite(P).all() or not np.isfinite(D).all()
                or not np.array_equal(P, P.T) or not np.array_equal(D, D.T)
                or np.linalg.eigvalsh(P)[0] <= 0
                or not math.isfinite(energy) or energy <= 0):
            raise ValueError("qualified positive finite source moment and block energy required")
        margin = CURVATURE_MARGIN*energy
        if index == 0:
            if gram_lower is None:
                gram_lower = _helmert_gram_lower(specification.modes)
            diagonal_bound = _diagonal_minima(P, D, rho)
            if diagonal_bound <= 0:
                # Scaling a negative bound would require a Gram upper bound;
                # it cannot meet the positive margin, so refuse before scaling.
                raise M4CertificateUnverified("prospective diagonal curvature is nonpositive",
                    {"block": block, "diagonal_hessian_lower_bound": diagonal_bound})
            result = {"hessian_lower_bound": (_i(diagonal_bound)*gram_lower).lo,
                      "diagonal_hessian_lower_bound": diagonal_bound,
                      "coefficient_gram_lower_bound": gram_lower,
                      "complete_domain_cover": True}
        else:
            s = float(specification.modes[block if index == 1 else 2*block, start, start])
            result = _cover_bound(P, D, rho, s, margin,
                "local_diagonal" if index == 1 else "local_shear", budget)
        result.update(objective_scale=energy, required_positive_margin=margin)
        result["alternative_hessian_lower_bound"] = result["hessian_lower_bound"]
        result["original_hessian_lower_bound"] = original["hessian_lower_bound"]
        result["hessian_lower_bound"] = max(result["hessian_lower_bound"],
                                             original["hessian_lower_bound"])
        result["alternative_evaluated"] = True
        if result["hessian_lower_bound"] <= margin:
            raise M4CertificateUnverified("prospective block curvature remains unverified",
                                         {"block": block, **result})
        blocks.append(result)
    budget.tick()
    return {"block_curvature": tuple(blocks),
            "hessian_lower_bound": min(item["hessian_lower_bound"] for item in blocks),
            "objective_scale": sum(item["objective_scale"] for item in blocks),
            "method": "inactive_model_specific_full_domain_curvature",
            "interval_arithmetic_proof": False, "production_adopted": False,
            "cover_limits": {"max_cells_per_block": limits.max_cells_per_block,
                             "max_depth": limits.max_depth,
                             "max_seconds_total": limits.max_seconds_total}}
