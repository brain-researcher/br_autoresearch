"""Exact Fourier moments of modal solutions (complex phi functions).

For a mode y' = lam*y + sum_j u_j s**j on a piece of length h (s = tau/h),
the work integrals against a tone exp(i*w*tau) are

    int_0^h exp(i w tau) y(tau) dtau = E(lam, w) y0 + sum_j u_j F_j(lam, w),
    E = h * phi_1(z),  z = (lam + i w) h,
    F_j = int_0^h exp(i w tau) psi_j(tau) dtau,
    psi_j(tau) = int_0^tau exp(lam (tau - sigma)) (sigma/h)**j dsigma.

Complex phi_k use the same argument scaling, Taylor series and doubling
identity as fast_solver_phi (valid for complex arguments). F_j is evaluated
by 64-point Gauss-Legendre quadrature of its exact smooth integrand when
|lam h| <= 40 and, beyond, from the exact split psi_j = P_j(s) + j!/z_r**(j+1)
exp(z_r s) with z_r = lam h, whose polynomial part is integrated against the
tone by the same quadrature (smooth for |w h| <= 8). No physics here.
"""
import math

import numpy as np

from fast_solver_phi import phi_table, GAUSS64_NODES, GAUSS64_WEIGHTS

_FACT = np.array([math.factorial(k) for k in range(40)], dtype=float)


def phi_complex(z, kmax):
    """phi_0..phi_kmax at complex z (Re z <= 0 expected): (kmax+1,) + z.shape."""
    z = np.asarray(z, dtype=complex)
    shape = z.shape
    flat = z.ravel()
    magnitude = np.abs(flat)
    steps = np.zeros(flat.size, dtype=np.int64)
    large = magnitude > 0.5
    steps[large] = np.ceil(np.log2(magnitude[large] / 0.5)).astype(np.int64)
    w = flat / np.ldexp(1.0, steps)
    top = np.zeros(flat.size, dtype=complex)
    term = np.full(flat.size, 1.0 / _FACT[kmax], dtype=complex)
    for m in range(30):
        top += term
        term = term * w / (m + kmax + 1)
    table = np.empty((kmax + 1, flat.size), dtype=complex)
    table[kmax] = top
    for k in range(kmax - 1, -1, -1):
        table[k] = w * table[k + 1] + 1.0 / _FACT[k]
    if steps.max(initial=0) > 0:
        lower = np.zeros((kmax + 1, kmax + 1))
        for k in range(1, kmax + 1):
            for j in range(1, k + 1):
                lower[k, j] = 1.0 / _FACT[k - j]
        scale = np.ldexp(1.0, -np.arange(kmax + 1))[:, None]
        for step in range(int(steps.max())):
            active = steps > step
            current = table[:, active]
            table[:, active] = scale * (current[0][None, :] * current + lower @ current)
    return table.reshape((kmax + 1,) + shape)


class FourierWeights:
    """E[k, m] and F[k, m, j] for modes lam[k] (<= 0), tones omega[m], piece length h."""

    def __init__(self, lam, omega, h, jmax):
        lam = np.asarray(lam, dtype=float)
        omega = np.asarray(omega, dtype=float)
        self.h, self.jmax = float(h), int(jmax)
        z = (lam[:, None] + 1j * omega[None, :]) * self.h          # (K, M)
        self.E = self.h * phi_complex(z, 1)[1]
        K, M, J = len(lam), len(omega), jmax + 1
        F = np.empty((K, M, J), dtype=complex)
        s, wts = GAUSS64_NODES, GAUSS64_WEIGHTS
        theta = omega * self.h                                      # (M,)
        tone = np.exp(1j * theta[:, None] * s[None, :])             # (M, G)
        zr = lam * self.h
        small = np.abs(zr) <= 40.0
        if np.any(small):
            ks = np.flatnonzero(small)
            phis = phi_table(zr[ks][:, None] * s[None, :], jmax + 1)   # (J+1, Ks, G)
            jj = np.arange(J)
            psi = (self.h * np.transpose(phis[1:J + 1], (1, 0, 2))
                   * (_FACT[jj][None, :, None] * s[None, None, :] ** (jj[None, :, None] + 1)))  # (Ks, J, G)
            # int_0^h e^{i w tau} psi(tau) dtau = h * int_0^1 e^{i theta s} psi(s h) ds
            F[ks] = self.h * np.einsum("mg,kjg,g->kmj", tone, psi, wts)
        big = ~small
        if np.any(big):
            kb = np.flatnonzero(big)
            zb = zr[kb]
            for j in range(J):
                # psi_j(s h)/h = P_j(s) + j!/z^{j+1} e^{z s},  P_j(s) = -sum_k j!/(j-k)! s^{j-k}/z^{k+1}
                poly = np.zeros((len(kb), s.size))
                for k in range(j + 1):
                    poly -= (_FACT[j] / _FACT[j - k]) * (s[None, :] ** (j - k)) / (zb[:, None] ** (k + 1))
                part = np.einsum("mg,kg,g->km", tone, poly, wts)
                expo = (_FACT[j] / zb[:, None] ** (j + 1)) * phi_complex(zb[:, None] + 1j * theta[None, :], 1)[1]
                F[kb, :, j] = self.h * self.h * (part + expo)
        self.F = F


def tone_integral(nu, t0, t1):
    """int_{t0}^{t1} exp(i nu t) dt for real nu arrays (exact, stable near nu = 0)."""
    nu = np.asarray(nu, dtype=float)
    h = t1 - t0
    return h * np.exp(1j * nu * t0) * phi_complex(1j * nu * h, 1)[1]
