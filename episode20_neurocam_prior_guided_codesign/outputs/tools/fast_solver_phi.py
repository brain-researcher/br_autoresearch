"""Exact-integration kernels for diagonal (modal) linear systems.

For one mode y' = lam*y + sum_j u_j s**j on a piece of length h, with the
dimensionless time s = tau/h in [0, 1] and z = lam*h (lam <= 0 for the passive
EP20 circuit), the exact solution is

    y(s) = exp(z s) y0 + h * sum_j u_j * s**(j+1) * j! * phi_{j+1}(z s),

where phi_k(x) = sum_m x**m/(m+k)!. Moments int_0^1 s**m y(s) ds are

    A_m(z) y0 + h * sum_j u_j B_{m,j}(z),
    A_m(z) = int_0^1 s**m exp(z s) ds,
    B_{m,j}(z) = int_0^1 s**(m+j+1) j! phi_{j+1}(z s) ds.

phi values use argument scaling, a short Taylor series, backward recurrence
and the positive-term doubling identity
    phi_k(2x) = 2**-k [phi_0(x) phi_k(x) + sum_{j=1..k} phi_j(x)/(k-j)!],
so no catastrophic cancellation occurs for x <= 0. Moments use 64-point
Gauss-Legendre quadrature of these exact integrands for |z| <= 40 and closed
forms (stable for |z| > max order) beyond. This module has no physics.
"""
import math

import numpy as np

_FACT = np.array([math.factorial(k) for k in range(40)], dtype=float)
Z_SWITCH = 40.0
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(64)
GAUSS64_NODES = 0.5 * (_GL_NODES + 1.0)
GAUSS64_WEIGHTS = 0.5 * _GL_WEIGHTS


def phi_table(z, kmax):
    """Return phi_0..phi_kmax at real z: array of shape (kmax+1,) + z.shape."""
    z = np.asarray(z, dtype=float)
    shape = z.shape
    flat = z.ravel()
    if not np.all(np.isfinite(flat)):
        raise ValueError("phi arguments must be finite")
    magnitude = np.abs(flat)
    steps = np.zeros(flat.size, dtype=np.int64)
    large = magnitude > 0.5
    steps[large] = np.ceil(np.log2(magnitude[large] / 0.5)).astype(np.int64)
    w = flat / np.ldexp(1.0, steps)
    # Taylor for phi_kmax(w), |w| <= 0.5: 30 terms exceed double precision.
    top = np.zeros(flat.size)
    term = np.full(flat.size, 1.0 / _FACT[kmax])
    for m in range(30):
        top += term
        term = term * w / (m + kmax + 1)
    table = np.empty((kmax + 1, flat.size))
    table[kmax] = top
    for k in range(kmax - 1, -1, -1):  # stable for small |w|
        table[k] = w * table[k + 1] + 1.0 / _FACT[k]
    if steps.max(initial=0) > 0:
        # lower[k, j] = 1/(k-j)! for 1 <= j <= k.
        lower = np.zeros((kmax + 1, kmax + 1))
        for k in range(1, kmax + 1):
            for j in range(1, k + 1):
                lower[k, j] = 1.0 / _FACT[k - j]
        scale = np.ldexp(1.0, -np.arange(kmax + 1))[:, None]
        for step in range(int(steps.max())):
            active = steps > step
            current = table[:, active]
            updated = scale * (current[0][None, :] * current + lower @ current)
            table[:, active] = updated
    return table.reshape((kmax + 1,) + shape)


def chebyshev_lobatto(degree):
    """Nodes on [0, 1] including both endpoints, increasing."""
    i = np.arange(degree + 1)
    return 0.5 * (1.0 - np.cos(np.pi * i / degree))


def vandermonde_inverse(nodes):
    """Inverse of V[i, j] = nodes[i]**j (maps node values to monomial coefficients)."""
    nodes = np.asarray(nodes, dtype=float)
    v = nodes[:, None] ** np.arange(len(nodes))[None, :]
    return np.linalg.inv(v)


def polynomial_values(coefficients, nodes):
    """Evaluate sum_j c[..., j] s**j at nodes; returns (..., len(nodes))."""
    coefficients = np.asarray(coefficients)
    powers = np.asarray(nodes)[None, :] ** np.arange(coefficients.shape[-1])[:, None]
    return coefficients @ powers


def shift_monomials(coefficients, offset, scale):
    """Coefficients in t=(offset + scale*s) -> coefficients in s (last axis)."""
    coefficients = np.asarray(coefficients, dtype=float)
    degree = coefficients.shape[-1] - 1
    result = np.zeros_like(coefficients)
    # (offset + scale s)**k = sum_i C(k,i) offset**(k-i) scale**i s**i
    for k in range(degree + 1):
        for i in range(k + 1):
            result[..., i] += coefficients[..., k] * (math.comb(k, i) * offset ** (k - i) * scale ** i)
    return result


def product_integral(left, right):
    """int_0^1 (sum a_j s^j)(sum b_k s^k) ds for coefficient arrays (..., J)."""
    left, right = np.asarray(left), np.asarray(right)
    j = np.arange(left.shape[-1])[:, None]
    k = np.arange(right.shape[-1])[None, :]
    weights = 1.0 / (j + k + 1.0)
    return np.einsum("...j,jk,...k->...", left, weights, right)


def monomial_integrals(coefficients, mmax):
    """int_0^1 s**m p(s) ds for m=0..mmax; p coefficients on the last axis."""
    coefficients = np.asarray(coefficients)
    j = np.arange(coefficients.shape[-1])[:, None]
    m = np.arange(mmax + 1)[None, :]
    return coefficients @ (1.0 / (j + m + 1.0))


class PieceWeights:
    """Exact response weights of a set of modes for one piece length h.

    lam: modal rates (<= 0), h: piece length (s), jmax: input degree,
    nodes: interior evaluation nodes in [0, 1], mmax: highest moment.
    Attributes (k = mode):
      end_exp[k] = exp(z_k); end_force[k, j] = h j! phi_{j+1}(z_k)
      node_exp[k, i] = exp(z_k s_i); node_force[k, j, i] = h s_i^{j+1} j! phi_{j+1}(z_k s_i)
      mom_exp[k, m] = A_m(z_k); mom_force[k, m, j] = h B_{m,j}(z_k)
    """

    def __init__(self, lam, h, jmax, nodes, mmax):
        lam = np.asarray(lam, dtype=float)
        if np.any(lam > 0) or not np.all(np.isfinite(lam)):
            raise ValueError("modal rates must be finite and nonpositive")
        self.h, self.jmax, self.mmax = float(h), int(jmax), int(mmax)
        self.nodes = np.asarray(nodes, dtype=float)
        z = lam * self.h
        self.z = z
        phis = phi_table(z, jmax + 1)
        self.end_exp = phis[0]
        j = np.arange(jmax + 1)
        self.end_force = self.h * phis[1:jmax + 2].T * _FACT[j][None, :]
        zs = z[:, None] * self.nodes[None, :]
        node_phi = phi_table(zs, jmax + 1)  # (jmax+2, K, I)
        self.node_exp = node_phi[0]
        powers = self.nodes[None, :] ** (j[:, None] + 1)  # (J, I)
        self.node_force = self.h * np.transpose(node_phi[1:jmax + 2], (1, 0, 2)) * (
            _FACT[j][None, :, None] * powers[None, :, :])
        self.mom_exp, self.mom_force = _moments(z, jmax, mmax)
        self.mom_force = self.h * self.mom_force


def _moments(z, jmax, mmax):
    """A[k, m] and B[k, m, j] for real z <= 0."""
    z = np.asarray(z, dtype=float)
    count = z.size
    a = np.empty((count, mmax + 1))
    b = np.empty((count, mmax + 1, jmax + 1))
    small = np.abs(z) <= Z_SWITCH
    if np.any(small):
        zs = z[small]
        nodes, weights = GAUSS64_NODES, GAUSS64_WEIGHTS
        arguments = zs[:, None] * nodes[None, :]
        phis = phi_table(arguments, jmax + 1)  # (jmax+2, K, G)
        exp_part = phis[0]
        m = np.arange(mmax + 1)
        powers_m = nodes[None, :] ** m[:, None]  # (M, G)
        a[small] = (exp_part * weights[None, :]) @ powers_m.T
        jj = np.arange(jmax + 1)
        psi = np.transpose(phis[1:jmax + 2], (1, 0, 2)) * (
            _FACT[jj][None, :, None] * nodes[None, None, :] ** (jj[None, :, None] + 1))  # (K, J, G)
        b[small] = np.einsum("kjg,mg,g->kmj", psi, powers_m, weights)
    big = ~small
    if np.any(big):
        zb = z[big]
        ez = np.exp(zb)
        ab = np.empty((zb.size, mmax + 1))
        ab[:, 0] = np.expm1(zb) / zb
        for m in range(1, mmax + 1):
            ab[:, m] = (ez - m * ab[:, m - 1]) / zb
        a[big] = ab
        for jv in range(jmax + 1):
            for m in range(mmax + 1):
                total = np.zeros(zb.size)
                for k in range(jv + 1):
                    total -= (_FACT[jv] / _FACT[jv - k]) / (zb ** (k + 1) * (m + jv - k + 1))
                total += (_FACT[jv] / zb ** (jv + 1)) * ab[:, m]
                b[big, m, jv] = total
    return a, b
