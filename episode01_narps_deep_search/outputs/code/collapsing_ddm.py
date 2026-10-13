"""Exponential-bound rival's absorbing first-passage distribution.

For X with dX=v dt+dW and bounds +/-a0*exp(-u*t)/2, transform
Y=X/a(t)+1/2. Ito gives dY=[v/a(t)+u(Y-1/2)]dt+dW/a(t).
The finite-volume birth/death generator on Y in (0,1) conserves total
probability, including separately accumulated lower/upper boundary flux.
Implicit time stepping and exponential face rates preserve nonnegative mass.

Grid/time convergence must be checked at fitted parameter values before this
solver supplies final inference. It implements the registered symmetric rival;
it is not an urgency-model selection result or a change in EP01's evidence gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import solve_banded


def _bernoulli(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    near = np.abs(x) < 1e-5
    positive = (x > 50) & ~near
    other = ~near & ~positive
    out[near] = 1-x[near]/2+x[near]**2/12
    out[positive] = x[positive]*np.exp(-x[positive])
    out[other] = x[other]/np.expm1(x[other])
    return out


@dataclass(frozen=True)
class FirstPassageDistribution:
    time_edges: np.ndarray
    reject_mass: np.ndarray
    accept_mass: np.ndarray
    survival_mass: float

    def response_density(self, decision_time, choice):
        """Piecewise mean density; no conditional-on-response renormalization."""
        times, choices = np.broadcast_arrays(np.asarray(decision_time, float), choice)
        if not np.isin(choices, [0, 1]).all():
            raise ValueError("Response choices must be zero or one")
        slots = np.searchsorted(self.time_edges, times, side="right")-1
        slots = np.minimum(slots, len(self.accept_mass)-1)
        active = (times > 0) & (times <= self.time_edges[-1])
        result = np.zeros(times.shape)
        safe = np.maximum(slots, 0)
        mass = np.where(choices == 1, self.accept_mass[safe], self.reject_mass[safe])
        result[active] = (mass / np.diff(self.time_edges)[safe])[active]
        return result


def solve(drift, a0, z, urgency, decision_deadline, *, dt=.002, intervals=200):
    values = [drift, a0, z, urgency, decision_deadline, dt]
    if not all(np.isfinite(values)) or a0 <= 0 or not 0 < z < 1 or urgency < 0:
        raise ValueError("Invalid drift-diffusion parameters")
    if decision_deadline <= 0 or dt <= 0 or intervals < 20:
        raise ValueError("Positive time horizon/step and at least 20 spatial intervals are required")
    if not isinstance(intervals, int):
        raise ValueError("Spatial interval count must be an integer")
    if urgency*decision_deadline > 300:
        raise ValueError("Requested contraction is outside floating-point support")
    h = 1/intervals
    if not h <= z <= 1-h:
        raise ValueError("Refine the spatial grid to represent the starting point")
    n_steps = math.ceil(decision_deadline/dt)
    edges = np.linspace(0., decision_deadline, n_steps+1)
    actual_dt = decision_deadline/n_steps
    interior = np.arange(1, intervals)
    # Linear interpolation of initial mass preserves initial position.
    fractional = z/h
    low = int(np.floor(fractional))
    mass = np.zeros(intervals-1)
    mass[low-1] += 1-(fractional-low)
    if fractional > low:
        mass[low] += fractional-low
    reject, accept = np.zeros(n_steps), np.zeros(n_steps)
    for step in range(n_steps):
        time = (edges[step]+edges[step+1])/2
        width = a0*np.exp(-urgency*time)
        diffusivity = .5/width**2
        drift_right = drift/width + urgency*((interior+.5)*h-.5)
        drift_left = drift/width + urgency*((interior-.5)*h-.5)
        right = diffusivity/h**2 * _bernoulli(-drift_right*h/diffusivity)
        left = diffusivity/h**2 * _bernoulli(drift_left*h/diffusivity)
        matrix = np.zeros((3, intervals-1))
        matrix[1] = 1+actual_dt*(left+right)
        matrix[0, 1:] = -actual_dt*left[1:]
        matrix[2, :-1] = -actual_dt*right[:-1]
        mass = solve_banded((1, 1), matrix, mass, check_finite=False)
        if np.min(mass) < -1e-13:
            raise FloatingPointError("Negative interior probability")
        reject[step] = actual_dt*left[0]*mass[0]
        accept[step] = actual_dt*right[-1]*mass[-1]
    remaining = float(mass.sum())
    if not np.isclose(reject.sum()+accept.sum()+remaining, 1., rtol=0., atol=1e-8):
        raise FloatingPointError("Absorbing solver lost probability mass")
    return FirstPassageDistribution(edges, reject, accept, remaining)
