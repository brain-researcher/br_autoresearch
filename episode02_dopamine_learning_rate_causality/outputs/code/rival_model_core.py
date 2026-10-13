"""EP02 low-parameter rival interfaces: outcome-blind, UNADOPTED specification.

These concrete operators occupy only the existing finite mechanism families.
They are not a selected production recipe, an empirical adapter, or an author
ACTR port. The new mathematical choices are recorded in source_semantics_design.md.
The passing actr_port, pinned_critic_core and development_score_core are unchanged.

All source meanings are caller inputs: shared past-only state features, action
identity, endpoint-positive action, signed performance error/credit orientation,
reward scale, terminal boundary and resolved common stimulation event. Nothing
here reads data, creates a task schedule, interprets lick/trial/session codes,
chooses a stimulation cap, fits parameters, or enables a search. In particular,
the canonical TD error here intentionally differs from the pinned source critic.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np


POLICY_FAMILIES = frozenset({
    "adaptive_rate_gate_ACTR", "constant_rate_direct_policy",
    "signed_PE_bias_ACTR", "amplitude_threshold_hybrid",
})
TD_FAMILIES = frozenset({"on_policy_TD_RPE", "off_policy_TD_RPE"})
FAMILIES = POLICY_FAMILIES | TD_FAMILIES | {"direct_contingent_reinforcement"}
SPECIFICATION_STATUS = "unadopted_low_parameter_mathematical_proposal"


class ModelError(ValueError):
    """Invalid explicit model input or nonfinite arithmetic; no partial update."""


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ModelError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ModelError(f"{name} cannot be represented") from error
    if not math.isfinite(result):
        raise ModelError(f"{name} must be finite")
    return result


def _nonnegative(value, name):
    result = _scalar(value, name)
    if result < 0:
        raise ModelError(f"{name} must be nonnegative")
    return result


def _bool(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise ModelError(f"{name} must be an explicit boolean")
    return bool(value)


def _action(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value not in (0, 1):
        raise ModelError(f"{name} must be action index 0 or 1, with meaning supplied by the caller")
    return int(value)


def _array(value, name, ndim):
    try:
        supplied = np.asarray(value)
        if supplied.dtype.kind not in "iuf":
            raise ModelError(f"{name} must contain real numeric values")
        with np.errstate(over="raise", invalid="raise"):
            result = np.asarray(supplied, dtype=np.float64)
    except (TypeError, ValueError, OverflowError, FloatingPointError) as error:
        if isinstance(error, ModelError):
            raise
        raise ModelError(f"{name} cannot be represented") from error
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ModelError(f"{name} must be finite and have {ndim} dimensions")
    return result


def _finite(*values):
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ModelError("model arithmetic is nonfinite; the whole update is refused")


@dataclass(frozen=True)
class BasisDefinition:
    """The same declared features/action order must be used by every rival.

    Feature values are supplied, never built from source fields here. The
    intercept, if wanted, is an explicitly named/supplied feature. No empirical
    action or endpoint label is presumed. Two rows of K coefficients are stored
    by every low-parameter model; this is 2K coefficients, not a claim about
    identifiable/effective complexity or a replacement for frozen complexity.
    """
    feature_names: tuple[str, ...]
    action_names: tuple[str, str]
    endpoint_action: int

    def __post_init__(self):
        for names, label in ((self.feature_names, "feature_names"),
                             (self.action_names, "action_names")):
            if (not isinstance(names, tuple) or not names or
                    any(not isinstance(name, str) or not name.strip() for name in names) or
                    len(set(names)) != len(names)):
                raise ModelError(f"{label} must be a nonempty tuple of distinct explicit names")
        if len(self.action_names) != 2:
            raise ModelError("this proposal has exactly two explicitly named actions")
        _action(self.endpoint_action, "endpoint_action")

    @property
    def stored_coefficient_count(self):
        return 2 * len(self.feature_names)


def _state(weights, state, definition):
    if not isinstance(definition, BasisDefinition):
        raise ModelError("an explicit BasisDefinition is required")
    w = _array(weights, "weights", 2)
    x = _array(state, "state", 1)
    k = len(definition.feature_names)
    if w.shape != (2, k) or x.shape != (k,):
        raise ModelError("weights/state must match the declared (2,K)/(K,) basis")
    return w, x


@dataclass(frozen=True)
class Readout:
    action_values: np.ndarray
    action_probabilities: np.ndarray
    endpoint_logit: float


def behavioral_readout(weights, state, *, definition, inverse_temperature):
    """Shared two-action linear-softmax readout, before a learning update.

    scores[a]=weights[a] @ state; logit(endpoint)=temperature*(score[e]-score[1-e]).
    Positive temperature is explicit and shared across matched comparisons.
    It is not learned here. No additive readout/intercept parameter is hidden.
    The returned logit can enter the existing scorer only after the separately
    specified session-binomial probability construction; this function does
    not silently average trial logits or define a session likelihood.
    """
    w, x = _state(weights, state, definition)
    temperature = _scalar(inverse_temperature, "inverse_temperature")
    if temperature <= 0:
        raise ModelError("inverse_temperature must be positive")
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            values = w @ x
            logit = temperature * (values[definition.endpoint_action] - values[1 - definition.endpoint_action])
            _finite(values, logit)
            z = math.exp(-abs(float(logit)))
            probability = 1.0 / (1.0 + z) if logit >= 0 else z / (1.0 + z)
            probabilities = np.empty(2, dtype=np.float64)
            probabilities[definition.endpoint_action] = probability
            probabilities[1 - definition.endpoint_action] = 1.0 - probability
    except (FloatingPointError, OverflowError) as error:
        raise ModelError("readout arithmetic overflowed") from error
    return Readout(values, probabilities, float(logit))


def score_function_credit(weights, state, *, definition, action, inverse_temperature):
    """Prospective choice: grad(log pi(action|state)) for direct-policy credit.

    This does not identify the provider's performance-error sign. The caller
    must orient that error consistently with an ascent update, or supply a
    different explicitly selected credit to performance_update. The recurrent
    author anchor retains its own eligibility calculation in actr_port.py.
    """
    w, x = _state(weights, state, definition)
    chosen = _action(action, "action")
    readout = behavioral_readout(w, x, definition=definition,
                                inverse_temperature=inverse_temperature)
    try:
        with np.errstate(over="raise", invalid="raise"):
            factors = -readout.action_probabilities.copy()
            factors[chosen] += 1.0
            credit = float(inverse_temperature) * factors[:, None] * x[None, :]
            _finite(credit)
    except (FloatingPointError, OverflowError) as error:
        raise ModelError("credit arithmetic overflowed") from error
    return credit


@dataclass(frozen=True)
class PerformanceUpdate:
    weights: np.ndarray
    error_before_intervention: float
    error_after_intervention: float
    rate_multiplier: float
    baseline_after: float


def performance_update(weights, credit, *, family, performance_error, baseline,
                       baseline_weight, alpha, beta_da, triggered,
                       stimulation_multiplier, positive_bias,
                       amplitude, amplitude_threshold):
    """Signed error/credit update for the four direct-policy mechanisms.

    All parameters are explicit. e=performance_error-baseline, W'=W+alpha*g*(e+b)*credit.
    Constant: g=1,b=0. Rate gate: g=beta*(multiplier if triggered else 1),b=0.
    Signed-PE rival: g=1,b=bias if triggered else 0. Hybrid: rate gate plus
    that bias only for a triggered amplitude >= the supplied threshold.
    The hybrid's boundary convention and amplitude units remain unadopted.
    Baseline'=(1-weight)*baseline+weight*unperturbed performance_error; no
    intervention bias enters that running baseline. These are prospective
    low-parameter operators, NOT replacements for the author's clipped RNN
    and reactive-path updates. Caller arrays are never modified.
    """
    if family not in POLICY_FAMILIES:
        raise ModelError("family must be one of the four declared direct-policy mechanisms")
    w = _array(weights, "weights", 2)
    eligibility = _array(credit, "credit", 2)
    if w.shape[0] != 2 or w.shape[1] < 1 or eligibility.shape != w.shape:
        raise ModelError("weights and credit must have the same (2,K>=1) shape")
    error = _scalar(performance_error, "performance_error")
    previous = _scalar(baseline, "baseline")
    running_weight = _scalar(baseline_weight, "baseline_weight")
    if not 0 <= running_weight <= 1:
        raise ModelError("baseline_weight must lie in [0,1]")
    rate = _nonnegative(alpha, "alpha")
    beta = _nonnegative(beta_da, "beta_da")
    multiplier = _nonnegative(stimulation_multiplier, "stimulation_multiplier")
    bias = _nonnegative(positive_bias, "positive_bias")
    stimulated = _bool(triggered, "triggered")
    intensity = _nonnegative(amplitude, "amplitude")
    threshold = _scalar(amplitude_threshold, "amplitude_threshold")
    if threshold <= 0:
        raise ModelError("amplitude_threshold must be positive in caller-specified units")
    gate = 1.0
    added_bias = 0.0
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            if family in {"adaptive_rate_gate_ACTR", "amplitude_threshold_hybrid"}:
                gate = beta * (multiplier if stimulated else 1.0)
            if stimulated and (family == "signed_PE_bias_ACTR" or
                               (family == "amplitude_threshold_hybrid" and intensity >= threshold)):
                added_bias = bias
            raw_error = error - previous
            effective_error = raw_error + added_bias
            updated = w + rate * gate * effective_error * eligibility
            next_baseline = (1.0 - running_weight) * previous + running_weight * error
            _finite(gate, raw_error, effective_error, updated, next_baseline)
    except (FloatingPointError, OverflowError) as error:
        raise ModelError("performance update overflowed") from error
    return PerformanceUpdate(updated, raw_error, effective_error, gate, next_baseline)


@dataclass(frozen=True)
class TDUpdate:
    weights: np.ndarray
    readout_before: Readout
    bootstrap_value: float
    target: float
    unperturbed_error: float
    effective_error: float


def td_update(weights, state, next_state, *, definition, family, action,
              next_action, terminal, reward, gamma, alpha,
              triggered, positive_rpe_bias, inverse_temperature):
    """One canonical semi-gradient SARSA(0) or Q-learning(0) transition.

    SARSA uses the caller's actually selected next action; max-Q maximizes the
    same next-state action values. Both use pre-update weights and the explicit
    terminal flag (terminal bootstrap=0). Terminal next_state/next_action must
    both be None; a nonterminal SARSA transition requires its next action.
    For max-Q, next_action must be None: a greedy action is never substituted
    into an on-policy transition. No terminal boundary is inferred from rows.

    target=r+gamma*bootstrap; delta=target-Q(s,a); effective=delta+trigger*bias.
    Only W[a] += alpha*effective*state. A positive additive stimulation bias
    in signed-RPE units is an explicit UNADOPTED counterfactual choice, not the
    pinned critic's reward*10 condition. No hidden eligibility trace is added.
    """
    if family not in TD_FAMILIES:
        raise ModelError("family must be an on-policy or off-policy TD rival")
    w, x = _state(weights, state, definition)
    chosen = _action(action, "action")
    last = _bool(terminal, "terminal")
    stimulated = _bool(triggered, "triggered")
    r = _scalar(reward, "reward")
    discount = _scalar(gamma, "gamma")
    rate = _nonnegative(alpha, "alpha")
    bias = _nonnegative(positive_rpe_bias, "positive_rpe_bias")
    if not 0 <= discount <= 1:
        raise ModelError("gamma must lie in [0,1]")
    before = behavioral_readout(w, x, definition=definition,
                               inverse_temperature=inverse_temperature)
    bootstrap = 0.0
    if last:
        if next_state is not None or next_action is not None:
            raise ModelError("terminal transition requires explicit absent next state and action")
    else:
        following = behavioral_readout(w, next_state, definition=definition,
                                      inverse_temperature=inverse_temperature)
        if family == "on_policy_TD_RPE":
            bootstrap = float(following.action_values[_action(next_action, "next_action")])
        else:
            if next_action is not None:
                raise ModelError("off-policy target does not take a next_action")
            bootstrap = float(np.max(following.action_values))
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            target = r + discount * bootstrap
            delta = target - float(before.action_values[chosen])
            effective = delta + (bias if stimulated else 0.0)
            updated = w.copy()
            updated[chosen] += rate * effective * x
            _finite(target, delta, effective, updated)
    except (FloatingPointError, OverflowError) as error:
        raise ModelError("TD update overflowed") from error
    return TDUpdate(updated, before, bootstrap, target, delta, effective)


@dataclass(frozen=True)
class ReinforcementUpdate:
    preferences: np.ndarray
    readout_before: Readout
    reinforced_action: int
    total_reinforcement: float


def direct_reinforcement_update(preferences, preceding_state, *, definition,
                                detected_action, natural_reinforcement,
                                triggered, stimulation_reinforcement,
                                alpha, inverse_temperature):
    """Non-neural associative preference update of the preceding detected action.

    h'[detected_action] = h[action] + alpha*(natural + trigger*stim)*preceding_state.
    Both reinforcers are explicitly nonnegative. Other action coefficients stay
    unchanged. This has no recurrent/hidden neural weight, critic, RPE, learned
    rate gate or retrospective reassignment to the endpoint-positive action.
    The native event's preceding state/action and natural-reward assignment are
    REQUIRED semantic inputs. This local non-neural rule is an UNADOPTED choice
    inside the existing direct-contingent-reinforcement family.
    """
    h, x = _state(preferences, preceding_state, definition)
    chosen = _action(detected_action, "detected_action")
    natural = _nonnegative(natural_reinforcement, "natural_reinforcement")
    stimulation = _nonnegative(stimulation_reinforcement, "stimulation_reinforcement")
    stimulated = _bool(triggered, "triggered")
    rate = _nonnegative(alpha, "alpha")
    before = behavioral_readout(h, x, definition=definition,
                               inverse_temperature=inverse_temperature)
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            total = natural + (stimulation if stimulated else 0.0)
            updated = h.copy()
            updated[chosen] += rate * total * x
            _finite(total, updated)
    except (FloatingPointError, OverflowError) as error:
        raise ModelError("direct reinforcement update overflowed") from error
    return ReinforcementUpdate(updated, before, chosen, total)
