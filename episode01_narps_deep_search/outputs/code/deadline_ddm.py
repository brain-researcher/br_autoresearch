"""Numerical first-passage primitives for EP01, not a completed inference model.

Fixed-bound density uses the image and eigenfunction series in Navarro & Fuss
(2009), doi:10.1016/j.jmp.2009.02.003. Diffusion variance is one. Survival is
the integral of the absorbing transition density, with an image-CDF expression
for short times. No RT binning, miss deletion or choice-only substitution.

This module deliberately does not assert that the EP01 behavioral gate has
passed: hierarchical inference, the recovery experiment and urgency robustness
remain separate parts of that same required analysis.
"""
from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr


def _inputs(t, drift, a, z):
    if not np.isfinite(a) or a <= 0 or not np.isfinite(z) or not 0 < z < 1:
        raise ValueError("a must be positive and z strictly between the bounds")
    t, drift = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(drift, dtype=float))
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(drift)):
        raise ValueError("time and drift must be finite")
    return t, drift


def lower_density(t, drift, a, z):
    """Density of first hitting zero from z*a in [0,a], including choice mass."""
    t, drift = _inputs(t, drift, a, z)
    shape = t.shape
    t, drift = t.reshape(-1), drift.reshape(-1)
    result = np.zeros_like(t)
    live = t > 0
    for small in (True, False):
        mask = live & ((t / a**2 < 0.15) if small else (t / a**2 >= 0.15))
        times, velocities = t[mask, None], drift[mask, None]
        if not len(times):
            continue
        if small:
            k = np.arange(-12, 13, dtype=float)
            distance = a * (z + 2 * k)
            terms = distance * np.exp(-distance**2 / (2 * times)
                                        - velocities * a * z - velocities**2 * times / 2)
            values = terms.sum(axis=1) / np.sqrt(2 * np.pi * times[:, 0]**3)
        else:
            k = np.arange(1, 81, dtype=float)
            terms = k * np.sin(k * np.pi * z) * np.exp(
                -k**2 * np.pi**2 * times / (2 * a**2)
                - velocities * a * z - velocities**2 * times / 2)
            values = np.pi / a**2 * terms.sum(axis=1)
        if np.any(values < -1e-12):
            raise FloatingPointError("Negative first-passage density")
        result[mask] = np.maximum(values, 0)
    return result.reshape(shape)


def response_density(t, drift, a, z, choice):
    """Choice=1 means accept/upper; choice=0 means reject/lower."""
    t, drift, choice = np.broadcast_arrays(t, drift, choice)
    if not np.isin(choice, [0, 1]).all():
        raise ValueError("Responded choices must be zero or one")
    return np.where(choice == 1, lower_density(t, -drift, a, 1-z),
                    lower_density(t, drift, a, z))


def _short_lower_cdf(t, drift, a, z):
    k = np.arange(-12, 13, dtype=float)
    signed_distance = a * (z + 2*k)
    distance = np.abs(signed_distance)
    time = t[:, None]
    velocity = drift[:, None]
    abs_velocity = np.abs(velocity)
    root_time = np.sqrt(time)
    log_first = -velocity * a * z - abs_velocity * distance + log_ndtr(
        (abs_velocity * time - distance) / root_time)
    log_second = -velocity * a * z + abs_velocity * distance + log_ndtr(
        (-abs_velocity * time - distance) / root_time)
    return ((np.exp(log_first) + np.exp(log_second)) * np.sign(signed_distance)).sum(axis=1)


def survival(t, drift, a, z):
    """Probability neither boundary has been reached at decision time t."""
    t, drift = _inputs(t, drift, a, z)
    shape = t.shape
    t, drift = t.reshape(-1), drift.reshape(-1)
    result = np.ones_like(t)
    short = (t > 0) & (t / a**2 < 0.15)
    if short.any():
        result[short] = (1 - _short_lower_cdf(t[short], drift[short], a, z)
                         - _short_lower_cdf(t[short], -drift[short], a, 1-z))
    long = (t > 0) & ~short
    if long.any():
        time, velocity = t[long, None], drift[long, None]
        k = np.arange(1, 81, dtype=float)
        frequency = k * np.pi / a
        decay = -velocity**2 * time / 2 - frequency**2 * time / 2
        # Combine exponentials before summation to avoid e^(v*a) overflow.
        integrated = frequency * (np.exp(decay - velocity * a*z)
            - (-1.)**k * np.exp(decay + velocity * a*(1-z))) / (velocity**2 + frequency**2)
        result[long] = (2/a * np.sin(k*np.pi*z) * integrated).sum(axis=1)
    if np.any(result < -1e-10) or np.any(result > 1+1e-10):
        raise FloatingPointError("Survival probability outside [0,1]")
    return np.clip(result, 0, 1).reshape(shape)


def eventual_acceptance(drift, a, z):
    """Stable upper-boundary absorption probability without a deadline."""
    _, drift = _inputs(np.zeros_like(np.asarray(drift, float)), drift, a, z)
    original_shape = drift.shape
    drift = drift.reshape(-1)
    out = np.empty_like(drift)
    near = np.abs(drift*a) < 1e-8
    positive, negative = (drift > 0) & ~near, (drift < 0) & ~near
    out[near] = z
    out[positive] = (-np.expm1(-2*drift[positive]*a*z)
                     / -np.expm1(-2*drift[positive]*a))
    v = -drift[negative]
    out[negative] = np.exp(-2*v*a*(1-z)) * (-np.expm1(-2*v*a*z))/(-np.expm1(-2*v*a))
    return out.reshape(original_shape)


def lower_cdf(t, drift, a, z):
    """Unconditional lower-boundary response probability by time t."""
    t, drift = _inputs(t, drift, a, z)
    shape = t.shape
    t, drift = t.reshape(-1), drift.reshape(-1)
    out = np.zeros_like(t)
    short = (t > 0) & (t/a**2 < .15)
    out[short] = _short_lower_cdf(t[short], drift[short], a, z)
    long = (t > 0) & ~short
    if long.any():
        time, velocity = t[long, None], drift[long, None]
        k = np.arange(1, 81, dtype=float)
        rate = (velocity**2 + (k*np.pi/a)**2)/2
        tail = (np.pi/a**2 * k*np.sin(k*np.pi*z)
                * np.exp(-velocity*a*z-rate*time)/rate).sum(axis=1)
        out[long] = eventual_acceptance(-drift[long], a, 1-z)-tail
    if np.any(out < -1e-10) or np.any(out > 1+1e-10):
        raise FloatingPointError("First-passage CDF outside [0,1]")
    return np.clip(out, 0, 1).reshape(shape)


def simulate(rng, drift, a, z, t0, contaminant_fraction, contaminant_miss_probability,
             deadline=4., bisection_steps=36):
    """Invert the exact fixed-bound FPT CDF; preserve endogenous deadline misses.

    No Euler-walk overshoot or imposed empirical miss labels. The same
    censoring/contaminant channel supplies both simulation and likelihood.
    """
    drift = np.asarray(drift, float)
    if not 0 <= t0 < deadline:
        raise ValueError("Invalid nondecision time")
    if not 0 <= contaminant_fraction < 1 or not 0 <= contaminant_miss_probability <= 1:
        raise ValueError("Invalid contaminant channel")
    shape = drift.shape
    v = drift.reshape(-1)
    n = len(v)
    choices, rt = np.full(n, -1, dtype=int), np.full(n, np.nan)
    contaminant = rng.random(n) < contaminant_fraction
    contam_response = contaminant & (rng.random(n) >= contaminant_miss_probability)
    choices[contam_response] = rng.integers(0, 2, contam_response.sum())
    rt[contam_response] = rng.uniform(np.finfo(float).eps, deadline, contam_response.sum())
    lower = lower_cdf(deadline-t0, v, a, z)
    upper = lower_cdf(deadline-t0, -v, a, 1-z)
    uniform = rng.random(n)
    reject = ~contaminant & (uniform < lower)
    accept = ~contaminant & (uniform >= lower) & (uniform < lower+upper)
    choices[reject], choices[accept] = 0, 1
    for mask, sign, start in [(reject, 1, z), (accept, -1, 1-z)]:
        if not mask.any():
            continue
        velocity = sign*v[mask]
        target = rng.random(mask.sum())*lower_cdf(deadline-t0, velocity, a, start)
        lo, hi = np.zeros(mask.sum()), np.full(mask.sum(), deadline-t0)
        for _ in range(bisection_steps):
            mid = (lo+hi)/2
            below = lower_cdf(mid, velocity, a, start) < target
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        rt[mask] = t0+(lo+hi)/2
    return choices.reshape(shape), rt.reshape(shape)


def drift_rate(gain, loss, kappa, eta=0., criterion=0., gamma_gain=1., gamma_loss=1.):
    if not 0.5 <= gamma_gain <= 1.5 or not 0.5 <= gamma_loss <= 1.5:
        raise ValueError("The nonlinear rival exponents are restricted to [0.5,1.5]")
    gain, loss = np.broadcast_arrays(np.asarray(gain, float), np.asarray(loss, float))
    if np.any(gain <= 0) or np.any(loss <= 0):
        raise ValueError("Offers use positive physical magnitudes")
    return (np.exp(kappa-eta/2)*(gain/10)**gamma_gain
            - np.exp(kappa+eta/2)*(loss/10)**gamma_loss + criterion)


def trial_log_likelihood(rt, choice, drift, a, z, t0, contaminant_fraction,
                         contaminant_miss_probability, deadline=4.):
    """Joint continuous choice/RT likelihood and deadline survival for misses.

    The contaminant channel has explicit miss mass and otherwise a uniform
    density over two choices and [0,deadline]. Its values must be recorded in
    the implementation specification; this function chooses neither value.
    """
    if not 0 <= contaminant_fraction < 1 or not 0 <= contaminant_miss_probability <= 1:
        raise ValueError("Invalid contaminant probabilities")
    if not 0 <= t0 < deadline:
        raise ValueError("Nondecision time must lie in [0,deadline)")
    rt, choice, drift = np.broadcast_arrays(np.asarray(rt, float), np.asarray(choice),
                                           np.asarray(drift, float))
    if not np.isin(choice, [-1, 0, 1]).all():
        raise ValueError("Choices must be -1 (miss), 0 (reject), or 1 (accept)")
    missed = choice == -1
    responded = ~missed
    if np.any(~np.isfinite(rt[responded])) or np.any(rt[responded] <= 0) or np.any(rt[responded] > deadline):
        raise ValueError("Responded RTs must be finite and within the deadline")
    probabilities = np.empty_like(drift)
    probabilities[missed] = ((1-contaminant_fraction) * survival(deadline-t0, drift[missed], a, z)
                             + contaminant_fraction * contaminant_miss_probability)
    probabilities[responded] = ((1-contaminant_fraction) * response_density(
        rt[responded]-t0, drift[responded], a, z, choice[responded])
        + contaminant_fraction * (1-contaminant_miss_probability)/(2*deadline))
    with np.errstate(divide="ignore"):
        return np.log(probabilities)
