"""Outcome-blind ACTR simulator primitives with explicit source variants.

The recurrent/plant/update equations are ported from the adjacent MIT oracle.
The seeded initializer and short task fixture replace unavailable source assets;
they are not a production endpoint, fitted model, or empirical replay. Public
functions accept numerical arrays, but the command-line qualifier constructs
only synthetic inputs and has no empirical input-path option.

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
from typing import Literal

import numpy as np


VariantName = Literal["article_25ms_times2", "pinned_code_30ms_set4"]
Arm = Literal["control", "stimLick_minus", "stimLick_plus"]


@dataclass(frozen=True)
class Variant:
    name: VariantName
    tau_ms: float
    stimulation_rule: str


VARIANTS = {
    "article_25ms_times2": Variant("article_25ms_times2", 25.0, "lick_contingent_multiply_two"),
    "pinned_code_30ms_set4": Variant("pinned_code_30ms_set4", 30.0, "author_error_sign_proxy_assign_four"),
}


@dataclass
class Network:
    recurrent: np.ndarray
    inputs: np.ndarray
    feedback: np.ndarray
    original_support: np.ndarray
    bias_indices: tuple[int, ...] = ()
    bias_value: float = 1.0


@dataclass(frozen=True)
class Trial:
    output: np.ndarray
    hidden: np.ndarray
    eligibility: np.ndarray


def gaussian_formula(mu: float, sigma: float, duration: int, dt: float = 1.0) -> np.ndarray:
    """Independent math implementation of the documented TONIC convention.

    The source exponent has denominator4*sigma**2, not2*sigma**2. The Gaussian
    prefactor cancels on normalization. Area uses sample indices (source trapz).
    No TONIC code is copied into this implementation.
    """
    if sigma <= 0 or dt <= 0 or duration < 2 * dt:
        raise ValueError("positive width/step and at least two samples required")
    coordinates = np.arange(1, int(math.floor(duration / dt)) + 1, dtype=float) * dt
    weights = np.exp(-0.25 * np.square((coordinates - mu) / sigma))
    area = float(np.sum(weights) - 0.5 * (weights[0] + weights[-1]))
    if not math.isfinite(area) or area <= 0:
        raise ValueError("Gaussian has no representable positive area")
    return weights / area


def seeded_network(seed: int, units: int = 50, probability: float = 0.9, gain: float = 1.3) -> Network:
    """Documented synthetic replacement, not the missing author initialization.

    Bernoulli connectivity with Gaussian weights SD=g/sqrt(p*N); two independent
    Gaussian input columns, no feedback/bias units, first unit is the output.
    Initial output-unit cue/reward couplings are0/1, matching the source's
    explicit first initialization setting. Seed algorithm is NumPy PCG64.
    """
    if units < 1 or not 0 < probability <= 1:
        raise ValueError("invalid network dimensions/connectivity")
    rng = np.random.Generator(np.random.PCG64(seed))
    support = rng.random((units, units)) < probability
    recurrent = rng.normal(0, gain / math.sqrt(probability * units), (units, units)) * support
    inputs = rng.normal(0, 1 / math.sqrt(2), (units, 2))
    inputs[0] = (0.0, 1.0)
    return Network(recurrent, inputs, np.zeros(units), support)


def recurrent_trial(network: Network, stimulus: np.ndarray, variant: Variant,
                    perturbations: np.ndarray | None) -> Trial:
    """Execute source Euler/tanh dynamics using an explicit perturbation tape.

    None selects the unperturbed source branch. A tape of zeros still selects
    the learning branch (as it does for a perturbed run without selected kicks).
    Eligibility is stored before the running activation mean is updated.
    """
    stimulus = np.asarray(stimulus, dtype=float)
    n, channels = network.inputs.shape
    if stimulus.ndim != 2 or stimulus.shape[0] != channels:
        raise ValueError("input channel mismatch")
    steps = stimulus.shape[1]
    if perturbations is not None and np.shape(perturbations) != (n, steps):
        raise ValueError("perturbation tape shape mismatch")
    x = np.zeros(n)
    x[list(network.bias_indices)] = network.bias_value
    rates = np.tanh(x)
    running = x.copy()
    eligibility = np.zeros((n, n))
    output = np.zeros(steps)
    hidden = np.empty((n, steps))
    history = np.empty((steps, n, n))
    for tick in range(steps):
        excitation = -x + network.recurrent @ rates + network.inputs @ stimulus[:, tick]
        if tick and np.max(np.abs(network.feedback)) > 0:
            excitation = excitation + network.feedback * output[tick - 1]
        x = x + excitation / variant.tau_ms
        if perturbations is not None:
            x = x + perturbations[:, tick]
        x[list(network.bias_indices)] = network.bias_value
        previous_rates = rates
        rates = np.tanh(x)
        if perturbations is not None:
            eligibility = 0.998 * eligibility + np.outer(previous_rates, (x - running) ** 3)
        history[tick] = eligibility
        running = 0.33 * running + 0.67 * x
        output[tick] = rates[0]
        hidden[:, tick] = x
    return Trial(output, hidden, history)


def perturbation_tape(rng: np.random.Generator, units: int, steps: int,
                      probability: float = 0.001) -> np.ndarray:
    # Explicit synthetic tape separates numerical equation parity from differences
    # between MATLAB and NumPy generators and array draw order.
    amplitude = 2 * rng.random((units, steps)) - 1
    return amplitude * (rng.random((units, steps)) <= probability)


def behavioral_plant(activity: np.ndarray, rng: np.random.Generator,
                     reward: np.ndarray | None = None, scale: float = 50.0,
                     transition_uniforms: np.ndarray | None = None,
                     bout_offsets: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Port the source state_simple/pass-thru plant; return 1-based lick times.

    The source uses a 3000-tick standard trial, 150-tick lick intervals and a
    100-tick reaction delay. Supplied random tapes permit independent replay.
    """
    activity = np.asarray(activity, dtype=float)
    steps = len(activity)
    if scale <= 0 or activity.ndim != 1 or steps < 2:
        raise ValueError("invalid plant input")
    background_on = reward is None
    if reward is None:
        if steps != 3000:
            raise ValueError("source default reward schedule requires3000 ticks")
        reward = np.r_[np.ones(1600), np.full(1400, 100.0)]
    reward = np.asarray(reward, dtype=float)
    if reward.shape != activity.shape or np.any(reward < 0):
        raise ValueError("invalid reward schedule")
    probability = np.maximum(activity, 0) / scale
    if background_on:
        probability = probability + np.exp((np.arange(1, steps + 1) - steps - 3) / 100)
    draws = rng.random(steps) if transition_uniforms is None else np.asarray(transition_uniforms)
    if draws.shape != activity.shape:
        raise ValueError("plant random tape shape mismatch")
    state = np.zeros(steps, dtype=np.int8)
    for tick in range(steps - 1):
        if state[tick] == 0:
            state[tick + 1] = int(draws[tick] < probability[tick])
        else:
            off = math.inf if reward[tick] == 0 else 0.005 / reward[tick]
            state[tick + 1] = int(not draws[tick] < off)
    transitions = np.diff(state, prepend=0)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    if len(starts) > len(ends):
        ends = np.r_[ends, steps - 1]
    marks = np.zeros(steps, dtype=bool)
    offset_index = 0
    for start, end in zip(starts, ends):
        if end - start + 1 > 100 or start + 1 > steps - 100:
            if bout_offsets is None:
                offset = int(rng.integers(1, 61))
            else:
                offset = int(bout_offsets[offset_index])
                if not 1 <= offset <= 60:
                    raise ValueError("bout offset outside source1..60")
            offset_index += 1
            marks[np.arange(start + offset, min(end + offset, steps - 1) + 1, 150)] = True
    licks = np.flatnonzero(marks) + 1 + 100
    return licks[licks < steps], state


def article_stimulation_multiplier(*, base_beta: float, triggered: bool) -> float:
    """Apply the adopted article multiplier to an explicitly resolved trigger.

    The source ACTR anchor's unperturbed dopamine factor is at least one.
    A triggered calibrated intervention doubles that current factor; an
    untriggered trial retains it. Neither the pinned code's set-to-four rule
    nor its off-trigger subtraction applies to this article component.

    ``triggered`` must be a Boolean from the common, authenticated task rule,
    including its session-cap decision; it is not inferred from an arm name,
    performance error, a lick proxy, trial order or stimulation counts here.
    The API cannot authenticate that provenance. Use the article_25ms_times2
    dynamics with this component; it is not a complete empirical model or fit.
    """
    if not isinstance(triggered, (bool, np.bool_)):
        raise ValueError("an explicit resolved Boolean trigger is required")
    if isinstance(base_beta, (bool, np.bool_)) or not isinstance(base_beta, Real):
        raise ValueError("source base multiplier must be a real scalar")
    try:
        base = float(base_beta)
    except (OverflowError, ValueError) as error:
        raise ValueError("source base multiplier must be representable") from error
    if not math.isfinite(base) or base < 1.0:
        raise ValueError("source base multiplier must be finite and at least1")
    result = base * (2.0 if triggered else 1.0)
    if not math.isfinite(result):
        raise ValueError("article multiplier must remain finite")
    return result


def stimulation(variant: Variant, arm: Arm, base_beta: float, error_r: float,
                pre_reward_lick: bool, session_trials: int, prior_stimulations: int,
                uniform: float) -> tuple[float, bool]:
    """Preserve the two declared rules, including the code's error proxy.

    Article cap handling is an explicit first-eligible synthetic scenario, not
    a claim about missing online cap-ordering metadata. This function is not
    the adopted empirical trigger implementation. Once that trigger is
    authenticated, pass its result directly to article_stimulation_multiplier;
    do not route empirical primary predictions through this scenario helper.
    """
    if arm not in ("control", "stimLick_minus", "stimLick_plus"):
        raise ValueError("unknown calibrated arm; high-amplitude extension is not qualified here")
    if base_beta < 1 or not math.isfinite(base_beta):
        raise ValueError("source base multiplier must be finite and at least1")
    if arm == "control":
        return base_beta, False
    if variant.name == "pinned_code_30ms_set4":
        trigger = ((error_r > 0 and uniform < 0.6) if arm == "stimLick_minus" else error_r < 0)
        return (4.0 if trigger else base_beta - 0.75), bool(trigger)
    trigger = not pre_reward_lick if arm == "stimLick_minus" else pre_reward_lick
    if arm == "stimLick_plus":
        trigger = trigger and prior_stimulations < session_trials // 2
    return article_stimulation_multiplier(base_beta=base_beta, triggered=bool(trigger)), bool(trigger)


def learning_step(network: Network, eligibility: np.ndarray, error_r: float,
                  baseline: float, beta: float, eta_j: float = 0.0005,
                  tau_trans: float = 2.0, saturation: float = 5.0) -> float:
    """Source signed-performance update with nonnegative dopamine multiplier."""
    if beta < 0 or not math.isfinite(beta):
        raise ValueError("dopamine multiplier cannot set update sign")
    credit = eligibility.T * network.original_support
    delta = np.clip(-eta_j * credit * (error_r - baseline) * beta, -0.01, 0.01)
    network.recurrent += delta
    errors = np.array([0.5 * error_r, error_r])
    eta_input = tau_trans / 125.0
    network.inputs[0] += (0.1 + saturation - network.inputs[0]) * eta_input * errors * beta
    network.inputs[0] = np.clip(network.inputs[0], 0.0, saturation)
    return 0.75 * baseline + 0.25 * error_r


def synthetic_fixture(variant_name: VariantName, arm: Arm, seed: int,
                      trials: int = 4) -> dict:
    """Exercise complete recurrent/plant/modulation/update paths on synthetic inputs.

    This short fixture is not the required800-trial8-session production task,
    model recovery, an intervention prediction, or a valid adaptive trial.
    Its cue/reward pulse schedule is explicitly constructed (not recovered from
    the unavailable author initialization workspace or empirical arrays).
    """
    variant = VARIANTS[variant_name]
    if trials < 1:
        raise ValueError("fixture requires at least one trial")
    network = seeded_network(seed)
    stimulus = np.zeros((2, 3000))
    stimulus[0, 499:599] = 1.0
    stimulus[1, 1599:1609] = 1.0
    reward_tick = 1600
    initial = recurrent_trial(network, stimulus, variant, None)
    initial_activity = 0.5 * initial.output + stimulus[1] * network.inputs[0, 1]
    licks, _ = behavioral_plant(initial_activity, np.random.default_rng(np.random.SeedSequence([seed, 0])))
    after = licks[licks > reward_tick]
    latency = int(after[0] - reward_tick) if len(after) else 1400
    baseline = 1 - math.exp(-latency / 500) + 0.25 * np.sum(np.abs(np.diff(initial.output[499:1600])))
    del initial
    da_cumulative = 3 * np.cumsum(gaussian_formula(700, 125, 1000))
    summary = []
    stim_count = 0
    for ordinal in range(1, trials + 1):
        # Same seed schedule across all arms/variants; tape roles stay separate.
        neural_rng = np.random.default_rng(np.random.SeedSequence([seed, ordinal, 1]))
        plant_rng = np.random.default_rng(np.random.SeedSequence([seed, ordinal, 2]))
        trigger_rng = np.random.default_rng(np.random.SeedSequence([seed, ordinal, 3]))
        trial = recurrent_trial(network, stimulus, variant, perturbation_tape(neural_rng, 50, 3000))
        activity = 0.5 * trial.output + stimulus[1] * network.inputs[0, 1] + 10 * stimulus[0] * network.inputs[0, 0]
        licks, _ = behavioral_plant(activity, plant_rng)
        after = licks[licks > reward_tick]
        latency = int(after[0] - reward_tick) if len(after) else 1400
        error_r = math.exp(-(latency + 50) / 500) - trial.output[1598]
        # MATLAB round is half-away-from-zero. The source clamps this index1..1000.
        raw_index = 100 * (trial.output[1609] - trial.output[1598])
        rounded = math.floor(raw_index + 0.5) if raw_index >= 0 else math.ceil(raw_index - 0.5)
        movement_index = int(np.clip(rounded, 1, 1000)) - 1
        reward_index = int(math.floor(network.inputs[0, 1] * 99.9))
        if not 0 <= reward_index < 1000:
            raise ValueError("source dopamine lookup is outside its declared support")
        base_beta = 1 + da_cumulative[movement_index] + da_cumulative[reward_index]
        prep = bool(np.any((licks >= reward_tick - 750) & (licks < reward_tick)))
        beta, delivered = stimulation(variant, arm, base_beta, error_r, prep, trials, stim_count, float(trigger_rng.random()))
        stim_count += int(delivered)
        baseline = learning_step(network, trial.eligibility[min(reward_tick + latency, 3000) - 1], error_r, baseline, beta)
        summary.append({"synthetic_trial": ordinal, "latency_ticks": latency,
                        "pre_reward_lick": prep, "beta": float(beta), "stimulated": delivered})
        del trial
    if not np.all(np.isfinite(network.recurrent)) or not np.all(np.isfinite(network.inputs)):
        raise ArithmeticError("nonfinite synthetic fixture")
    return {"variant": variant.name, "arm": arm, "seed": seed,
            "synthetic_trials": trials, "stimulations": stim_count,
            "summary": summary, "empirical_payload_access": False,
            "production_model_qualified": False, "valid_adaptive_trials": 0}
