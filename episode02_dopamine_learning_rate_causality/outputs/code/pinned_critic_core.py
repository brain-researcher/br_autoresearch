"""EP02 author-code parity component: the pinned state-value critic arithmetic.

Mathematical source, MIT, commit 3627746957975632cddb05bf8f6b26a4b6f901b6:
https://github.com/dudmanj/RNN_learnDA/blob/3627746957975632cddb05bf8f6b26a4b6f901b6/dlRNN_criticEngine.m

This preserves delta = reward + gamma * (V_next - V_current). It is NOT silently
replaced by canonical TD's reward + gamma * V_next - V_current; those formulas
agree when gamma=1. The source stim_cond>=20 branch multiplies supplied rewards
by ten. No provider arm, event, lick-state or online-trigger meaning is inferred.

The caller supplies all arrays and parameters. There is no source reader, basis
builder, policy/action rule, behavioral readout, fitter or simulation schedule.
This component does not establish a complete on-policy/off-policy TD rival,
direct-reinforcement model, empirical endpoint, or production-model readiness.

Adapted source attribution: Copyright (c) 2021 Joshua Dudman.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of
the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


class CriticError(ValueError):
    """Invalid arrays/parameters or an unrepresentable complete critic call."""


@dataclass(frozen=True)
class CriticResult:
    final_weights: np.ndarray
    values: np.ndarray
    next_values_before_update: np.ndarray
    td_errors: np.ndarray
    final_eligibility: np.ndarray


def _array(value, name: str, ndim: int) -> np.ndarray:
    try:
        supplied = np.asarray(value)
        if supplied.dtype.kind not in "iuf":
            raise CriticError(f"{name} must contain real numeric values")
        with np.errstate(over="raise", invalid="raise"):
            array = np.asarray(supplied, dtype=np.float64)
    except (TypeError, ValueError, FloatingPointError, OverflowError) as error:
        if isinstance(error, CriticError):
            raise
        raise CriticError(f"{name} cannot be represented as a numeric array") from error
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise CriticError(f"{name} must be finite and have {ndim} dimensions")
    return array


def _scalar(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CriticError(f"{name} must be a real scalar")
    try:
        scalar = float(value)
    except (ValueError, OverflowError) as error:
        raise CriticError(f"{name} is not representable") from error
    if not math.isfinite(scalar):
        raise CriticError(f"{name} must be finite")
    return scalar


def pinned_td_critic(initial_weights, basis, rewards, *, alpha, gamma,
                     trace_decay, stim_cond) -> CriticResult:
    """Execute exactly T-1 source transitions without mutating any input.

    Shapes are weights[K], basis[K,T] and rewards[T], K>=1 and T>=2. Scalars
    are mandatory: alpha>=0, gamma and trace_decay in [0,1], finite stim_cond.
    Computation uses binary64 without clipping. Eligibility starts at zero on
    every call; only weights may be carried over by the caller between trials.

    Each transition evaluates both values using the same current weights, then
    computes delta, eligibility and the weight update, in that order. rewards[-1]
    is unused, as in the source loop. There is no terminal update or forced zero
    bootstrap. errors and next-values have T-1 computed entries, not a fabricated
    last error. values[T] preserves the source's overwrite convention: a stored
    next value is recomputed when it becomes the next transition's current value.
    Consequently values is not a value function evaluated with final weights.

    A numerical failure refuses the whole call, leaving caller-owned arrays
    unchanged. Cue/reward indices and source role/trigger meanings stay external.
    """
    weights = _array(initial_weights, "initial_weights", 1).copy()
    features = _array(basis, "basis", 2)
    reward = _array(rewards, "rewards", 1)
    if weights.size < 1 or features.shape[0] != weights.size or features.shape[1] < 2:
        raise CriticError("basis must have shape (number_of_weights, T>=2) with nonempty weights")
    steps = features.shape[1]
    if reward.shape != (steps,):
        raise CriticError("rewards must have exactly T entries")
    learning_rate = _scalar(alpha, "alpha")
    discount = _scalar(gamma, "gamma")
    decay = _scalar(trace_decay, "trace_decay")
    condition = _scalar(stim_cond, "stim_cond")
    if learning_rate < 0 or not 0 <= discount <= 1 or not 0 <= decay <= 1:
        raise CriticError("require alpha>=0 and gamma/trace_decay in [0,1]")
    reward_multiplier = 10.0 if condition >= 20 else 1.0
    eligibility = np.zeros_like(weights)
    values = np.empty(steps, dtype=np.float64)
    next_values = np.empty(steps - 1, dtype=np.float64)
    errors = np.empty(steps - 1, dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for tick in range(steps - 1):
                current_value = np.sum(weights * features[:, tick], dtype=np.float64)
                next_value = np.sum(weights * features[:, tick + 1], dtype=np.float64)
                delta = reward_multiplier * reward[tick] + discount * (next_value - current_value)
                eligibility = discount * decay * eligibility + learning_rate * features[:, tick]
                weights = weights + delta * eligibility
                if (not np.isfinite(current_value) or not np.isfinite(next_value) or
                        not np.isfinite(delta) or not np.all(np.isfinite(eligibility)) or
                        not np.all(np.isfinite(weights))):
                    raise CriticError("critic arithmetic is not finite; the whole call is refused")
                values[tick] = current_value
                values[tick + 1] = next_value
                next_values[tick] = next_value
                errors[tick] = delta
    except (FloatingPointError, OverflowError) as error:
        raise CriticError("critic arithmetic overflowed; the whole call is refused") from error
    return CriticResult(weights, values, next_values, errors, eligibility)
