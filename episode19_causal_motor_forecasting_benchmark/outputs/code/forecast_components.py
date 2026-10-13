"""EP19 prospective input and calibration primitives, without source I/O.

Calibration defaults are the existing provisional component recipe. This is
not a trained forecasting model, onset detector, source-role builder or margin
freeze. A caller must provide the same natural at-risk chronological prefix
for each model and keep guard/suffix labels unavailable.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


class CausalFIR:
    """Past-only finite filter; explicit warm-up and gap reset, no padding."""
    def __init__(self, coefficients, channels):
        self.coefficients = np.asarray(coefficients, dtype=float)
        if self.coefficients.ndim != 1 or not len(self.coefficients) or not np.all(np.isfinite(self.coefficients)):
            raise ValueError("finite one-sided coefficients are required")
        self.channels = channels
        self.reset()

    def reset(self):
        self.history = []

    def update(self, sample):
        sample = np.asarray(sample, dtype=float)
        if sample.shape != (self.channels,) or not np.all(np.isfinite(sample)):
            self.reset()
            return None
        self.history.append(sample.copy())
        self.history = self.history[-len(self.coefficients):]
        if len(self.history) < len(self.coefficients):
            return None
        return self.coefficients @ np.asarray(self.history[::-1])


@dataclass(frozen=True)
class FactorizedBlocks:
    complement: np.ndarray
    older_slow: np.ndarray
    recent: np.ndarray

    def reconstruct(self, recent_present, older_slow_present):
        older = self.complement + int(bool(older_slow_present)) * self.older_slow
        recent = self.recent if recent_present else np.zeros_like(self.recent)
        return np.concatenate([older, recent], axis=0)


def factorized_blocks(raw_history, coefficients, window_start, cutoff_index,
                      end_index, voltage_mean, older_slow_mean, voltage_scale):
    """Construct fixed slots with older s<cutoff and recent cutoff<=s<end.

    Coefficients are caller-supplied development-fixed choices. At least L-1
    preceding samples must exist before window_start; a gap requires a fresh
    history buffer. No filter output is shifted backward or padded.
    """
    raw = np.asarray(raw_history, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if raw.ndim != 2 or coefficients.ndim != 1 or not len(coefficients):
        raise ValueError("two-dimensional voltage and one-dimensional FIR coefficients required")
    if not (len(coefficients) - 1 <= window_start < cutoff_index < end_index <= len(raw)):
        raise ValueError("incomplete history or invalid disjoint block boundaries")
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(raw[window_start-len(coefficients)+1:end_index])):
        raise ValueError("gap/nonfinite history cannot cross a factorized example")
    mean, older_mean, scale = [np.asarray(item, dtype=float) for item in (voltage_mean, older_slow_mean, voltage_scale)]
    if any(item.shape != (raw.shape[1],) or not np.all(np.isfinite(item)) for item in (mean, older_mean, scale)) or np.any(scale <= 0):
        raise ValueError("one finite frozen mean and positive shared scale per channel required")
    older = raw[window_start:cutoff_index]
    slow = sum(coefficient * raw[window_start-lag:cutoff_index-lag]
               for lag, coefficient in enumerate(coefficients))
    complement = (older - slow - (mean - older_mean)) / scale
    slow = (slow - older_mean) / scale
    recent = (raw[cutoff_index:end_index] - mean) / scale
    return FactorizedBlocks(complement, slow, recent)


def calibrated_log_probabilities(logits, parameters):
    logits = np.asarray(logits, dtype=float)
    parameters = np.asarray(parameters, dtype=float)
    if logits.ndim != 2 or logits.shape[1] != 19 or parameters.shape != (4,) or not np.all(np.isfinite(logits)) or not np.all(np.isfinite(parameters)) or parameters[0] <= 0:
        raise ValueError("finite19-category logits and four parameters with positive alpha required")
    offsets = np.r_[np.repeat(parameters[1:], 6), 0.]
    adjusted = parameters[0] * logits + offsets
    return adjusted - logsumexp(adjusted, axis=1, keepdims=True)


def censoring_loss_and_logit_gradient(log_probabilities, observed_category, complete_bins):
    """Return per-grid-point NLL and dNLL/d(adjusted logits).

    observed_category is0..18 for an observed event/no-onset category, or-1
    for administrative censoring. complete_bins0..18 is used only when
    censored. An observed first-bin onset is category0 even if no full bin has
    elapsed; zero-bin censoring has likelihood1 and gradient0.
    """
    logp = np.asarray(log_probabilities, dtype=float)
    categories = np.asarray(observed_category, dtype=int)
    complete = np.asarray(complete_bins, dtype=int)
    if logp.ndim != 2 or logp.shape[1] != 19 or not np.all(np.isfinite(logp)):
        raise ValueError("finite log probabilities required")
    if categories.shape != (len(logp),) or complete.shape != categories.shape or np.any((categories < -1) | (categories > 18)) or np.any((complete < 0) | (complete > 18)):
        raise ValueError("invalid categorical or censoring arrays")
    if not np.allclose(logsumexp(logp, axis=1), 0., rtol=0, atol=1e-12):
        raise ValueError("log probabilities are not normalized")
    probability = np.exp(logp)
    loss = np.zeros(len(logp))
    gradient = probability.copy()
    for row in range(len(logp)):
        if categories[row] >= 0:
            loss[row] = -logp[row, categories[row]]
            gradient[row, categories[row]] -= 1
        elif complete[row] == 0:
            gradient[row] = 0
        else:
            start = complete[row]
            survival_logp = logsumexp(logp[row, start:])
            loss[row] = -survival_logp
            gradient[row, start:] -= np.exp(logp[row, start:] - survival_logp)
    return loss, gradient


def calibration_objective(parameters, logits, observed_category, complete_bins, weights):
    logits = np.asarray(logits, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(logits),) or not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("finite natural-prevalence weights with positive total required")
    logp = calibrated_log_probabilities(logits, parameters)
    loss, logit_gradient = censoring_loss_and_logit_gradient(logp, observed_category, complete_bins)
    weights = weights / weights.sum()
    weighted = logit_gradient * weights[:, None]
    parameters = np.asarray(parameters, dtype=float)
    displacement = parameters - np.array([1., 0., 0., 0.])
    objective = float(weights @ loss + .01 * (displacement @ displacement))
    gradient = np.r_[np.sum(weighted * logits), [weighted[:, start:start+6].sum() for start in (0, 6, 12)]]
    return objective, gradient + .02 * displacement


def fit_four_parameter_calibration(logits, observed_category, complete_bins, weights=None):
    """Existing provisional L-BFGS-B recipe with deterministic identity fallback."""
    logits = np.asarray(logits, dtype=float)
    if weights is None:
        weights = np.ones(len(logits))
    identity = np.array([1., 0., 0., 0.])
    bounds = [(0.25, 4.0)] + [(-3., 3.)] * 3
    arguments = (logits, observed_category, complete_bins, weights)
    identity_objective = calibration_objective(identity, *arguments)[0]
    result = minimize(calibration_objective, identity, args=arguments, method="L-BFGS-B", jac=True,
                      bounds=bounds, options={"maxiter": 200, "gtol": 1e-6, "ftol": 1e-9})
    bounded = all(low <= value <= high for value, (low, high) in zip(result.x, bounds))
    accepted = bool(result.success and np.all(np.isfinite(result.x)) and np.isfinite(result.fun)
                    and bounded and result.fun <= identity_objective + 1e-10)
    parameters = result.x if accepted else identity
    return {"parameters": parameters.tolist(), "accepted": accepted,
            "identity_fallback": not accepted, "optimizer_message": str(result.message),
            "iterations": int(result.nit), "identity_objective": identity_objective,
            "selected_objective": calibration_objective(parameters, *arguments)[0],
            "recipe_frozen_for_empirical_use": False}
