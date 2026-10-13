"""Candidate batching of the unchanged exponential-bound finite-volume method.

Each exact drift occupies an independent tridiagonal block. This module is an
engineering candidate, not an adopted fitter or a precision qualification.
The scalar collapsing_ddm implementation remains the numerical reference.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import expit

import collapsing_ddm as scalar
import deadline_ddm


def solve_many(drifts, a0, z, urgency, decision_deadline, *, dt=.002, intervals=200):
    """Return scalar-compatible distributions in the exact input drift order.

No drifts are rounded, merged or omitted. Duplicate inputs remain independent
blocks; the likelihood adapter supplies exact np.unique values.
"""
    drifts = np.asarray(drifts, dtype=float)
    values = [a0, z, urgency, decision_deadline, dt]
    if (drifts.ndim != 1 or not np.isfinite(drifts).all() or not all(np.isfinite(values))
            or a0 <= 0 or not 0 < z < 1 or urgency < 0):
        raise ValueError("Invalid drift-diffusion parameters")
    if decision_deadline <= 0 or dt <= 0 or intervals < 20:
        raise ValueError("Positive time horizon/step and at least20 spatial intervals are required")
    if not isinstance(intervals, int):
        raise ValueError("Spatial interval count must be an integer")
    if urgency * decision_deadline > 300:
        raise ValueError("Requested contraction is outside floating-point support")
    h = 1 / intervals
    if not h <= z <= 1 - h:
        raise ValueError("Refine the spatial grid to represent the starting point")
    if len(drifts) == 0:
        return ()
    n_steps = math.ceil(decision_deadline / dt)
    edges = np.linspace(0., decision_deadline, n_steps + 1)
    actual_dt = decision_deadline / n_steps
    interior = np.arange(1, intervals)
    block_size = intervals - 1
    fractional = z / h
    low = int(np.floor(fractional))
    mass = np.zeros((len(drifts), block_size))
    mass[:, low - 1] += 1 - (fractional - low)
    if fractional > low:
        mass[:, low] += fractional - low
    reject = np.zeros((len(drifts), n_steps))
    accept = np.zeros((len(drifts), n_steps))
    # Cross-block upper/lower entries remain zero throughout every time step.
    matrix = np.zeros((3, len(drifts) * block_size))
    blocks = matrix.reshape(3, len(drifts), block_size)
    drift_columns = drifts[:, None]
    for step in range(n_steps):
        time = (edges[step] + edges[step + 1]) / 2
        width = a0 * np.exp(-urgency * time)
        diffusivity = .5 / width**2
        drift_right = drift_columns / width + urgency * ((interior + .5) * h - .5)
        drift_left = drift_columns / width + urgency * ((interior - .5) * h - .5)
        right = diffusivity / h**2 * scalar._bernoulli(-drift_right * h / diffusivity)
        left = diffusivity / h**2 * scalar._bernoulli(drift_left * h / diffusivity)
        blocks[1] = 1 + actual_dt * (left + right)
        blocks[0, :, 1:] = -actual_dt * left[:, 1:]
        blocks[2, :, :-1] = -actual_dt * right[:, :-1]
        mass = solve_banded((1, 1), matrix, mass.reshape(-1), check_finite=False).reshape(len(drifts), block_size)
        if np.any(np.min(mass, axis=1) < -1e-13):
            raise FloatingPointError("Negative interior probability")
        reject[:, step] = actual_dt * left[:, 0] * mass[:, 0]
        accept[:, step] = actual_dt * right[:, -1] * mass[:, -1]
    remaining = mass.sum(axis=1)
    if not np.all(np.isclose(reject.sum(axis=1) + accept.sum(axis=1) + remaining, 1., rtol=0., atol=1e-8)):
        raise FloatingPointError("Absorbing solver lost probability mass")
    return tuple(scalar.FirstPassageDistribution(edges.copy(), reject[index].copy(), accept[index].copy(), float(survival))
                 for index, survival in enumerate(remaining))


def trial_likelihood(theta, names, data, spec):
    """Urgency-only likelihood with the scalar model's unchanged mixture rules."""
    params = dict(zip(names, theta))
    if "urgency" not in params:
        raise ValueError("The batched adapter is only for the explicit urgency rival")
    a = np.exp(params["log_a"])
    z = expit(params.get("rho", 0.))
    t0 = 4 * expit(params["nondecision_logit"])
    drift = deadline_ddm.drift_rate(data["gain"], data["loss"], params["kappa"],
        params.get("eta", 0.), params.get("criterion", 0.), params.get("gamma_gain", 1.), params.get("gamma_loss", 1.))
    eps = spec["contaminant"]["fraction"]
    miss_mass = spec["contaminant"]["miss_probability"]
    values, inverse = np.unique(drift, return_inverse=True)
    distributions = solve_many(values, a, z, params["urgency"], 4 - t0,
        dt=spec["urgency_solver"]["dt_seconds"], intervals=spec["urgency_solver"]["spatial_intervals"])
    logp = np.empty_like(drift)
    for index, dist in enumerate(distributions):
        rows = inverse == index
        rt, choice = data["rt"][rows], data["choice"][rows]
        probabilities = np.empty(len(rt))
        misses = choice == -1
        probabilities[misses] = (1 - eps) * dist.survival_mass + eps * miss_mass
        probabilities[~misses] = ((1 - eps) * dist.response_density(rt[~misses] - t0, choice[~misses])
                                  + eps * (1 - miss_mass) / 8)
        with np.errstate(divide="ignore"):
            logp[rows] = np.log(probabilities)
    return logp
