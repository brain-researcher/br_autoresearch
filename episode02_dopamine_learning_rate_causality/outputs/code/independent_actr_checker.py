#!/usr/bin/env python3
"""Independently authored EP02 component reference; no primary-port imports.

The specification and synthetic inputs below were written before comparison
against any primary implementation, tests, outputs or expected values.
Permitted inputs were actr_checker_interface.md, the three pinned MIT MATLAB
sources and LICENSE in author_oracle/, and the pinned Gaussian math URL.

Source attribution: Copyright (c) 2021 Joshua Dudman. This reference uses
the MIT-licensed recurrent, plant and training equations. The Gaussian
function is an independent mathematical implementation, not copied TONIC code.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This is partial component qualification, not production simulation, empirical
recovery, full-trial parity, calibration, or outcome-access authorization.
"""
from __future__ import annotations

import copy
import json
import math
import os
import socket


INDEPENDENT_SPECIFICATION = {
    "authorship": "specified_before_any_primary_port_tests_outputs_or_expected_values",
    "author_source_commit": "3627746957975632cddb05bf8f6b26a4b6f901b6",
    "gaussian_math_url": "https://github.com/DudLab/TONIC/blob/f78fdbf4bca0e5c16859f885550b4e274fc3459d/CreateFunctions/TNC_CreateGaussian.m",
    "gaussian": "t=dt,2dt,...<=duration; exp(-(t-mu)^2/(4*sigma^2)); normalize by unit-spacing trapezoid sum",
    "recurrent": "Euler dt=1ms, simultaneous J*r, previous output feedback, additive supplied perturbations then bias reset",
    "hidden_mapping": "pre_tanh_hidden_x; clarified_by_interface_owner_before_implementation",
    "eligibility": "raw E[a,b] decays by .998 then adds prior_rate[a]*(x[b]-prior_xbar[b])^3; xbar=.33*prior_xbar+.67*x",
    "learning_branch": "null perturbations suppress eligibility; explicit zeros retain learning branch",
    "plant": "rectify/scale; source default reward and back probability only when reward=null; strict transition comparisons; inclusive bout spans; offsets1..60;150tick lick template;100tick reaction;lick<T",
    "stimulation": "pinned switch uses error sign and strict uniform<.6, set4 or subtract.75; article doubles on declared lick gate and caps plus at floor(session_trials/2)",
    "learning": "transpose raw eligibility once; mask original support; subtract eta_j*E_transposed*(error-baseline)*beta;clip increment+/-.01;NaNincrement0;input output-row cue/error*.5 and reward/error, eta=tau_trans/125;clip input[0,saturation];baseline=.75old+.25error",
    "synthetic_design": "self_chosen_nonsymmetric3unit_and2unit_networks;nonzero_bias_feedback;null_zero_and_nonzero_tapes;plant_bout_and_tick_boundaries;stimulation_boundary_cases;clipping_masking_zero_PE_and_input_saturation",
    "comparison_performed_during_authorship": False,
}


def _gaussian(case):
    sigma, dt, duration = (float(case[k]) for k in ("sigma", "dt", "duration"))
    if sigma <= 0 or dt <= 0 or duration < 2 * dt:
        raise ValueError("reference Gaussian needs sigma>0 and at least two samples")
    # The source prefactor is constant across samples and cancels exactly on
    # normalization. Its trapz call does not supply the sample spacing.
    n = math.floor(duration / dt + 8 * math.ulp(duration / dt))
    weights = [math.exp(-((i * dt - case["mu"]) / (2 * sigma)) ** 2)
               for i in range(1, n + 1)]
    integral = sum((left + right) / 2 for left, right in zip(weights, weights[1:]))
    return {"values": [w / integral for w in weights]}


def _recurrent(case):
    net, u, tape = case["network"], case["stimulus"], case["perturbations"]
    n, ticks = len(net["recurrent"]), len(u[0])
    x = [0.0] * n
    for i in net["bias_indices"]:
        x[i] = float(net["bias_value"])
    rates, average = [math.tanh(v) for v in x], list(x)
    e = [[0.0] * n for _ in range(n)]
    hidden, outputs, history = [[] for _ in range(n)], [], []
    for tick in range(ticks):
        next_x = []
        for i in range(n):
            recurrent = sum(net["recurrent"][i][j] * rates[j] for j in range(n))
            drive = sum(net["inputs"][i][channel] * u[channel][tick]
                        for channel in range(len(u)))
            feedback = net["feedback"][i] * outputs[-1] if tick else 0.0
            change = (-x[i] + recurrent + drive + feedback) / case["tau_ms"]
            next_x.append(x[i] + change + (tape[i][tick] if tape is not None else 0.0))
        for i in net["bias_indices"]:
            next_x[i] = float(net["bias_value"])
        if tape is not None:
            fluctuation = [(next_x[j] - average[j]) ** 3 for j in range(n)]
            e = [[0.998 * e[i][j] + rates[i] * fluctuation[j]
                  for j in range(n)] for i in range(n)]
        average = [0.33 * average[i] + 0.67 * next_x[i] for i in range(n)]
        x, rates = next_x, [math.tanh(value) for value in next_x]
        outputs.append(rates[0])
        for i in range(n):
            hidden[i].append(x[i])
        history.append(copy.deepcopy(e))
    return {"output": outputs, "hidden": hidden, "eligibility": history}


def _plant(case):
    activity, uniform = case["activity"], case["transition_uniforms"]
    length = len(activity)
    default_reward = case["reward"] is None
    if default_reward and length > 3000:
        raise ValueError("source default reward has only 3000 ticks")
    reward = ([1.0] * 1600 + [100.0] * 1400) if default_reward else case["reward"]
    state = [0] * length
    for i in range(length - 1):
        if state[i] == 0:
            probability = max(activity[i], 0.0) / case["scale"]
            if default_reward:
                probability += math.exp(((i + 1) - length - 3) / 100)
            state[i + 1] = int(uniform[i] < probability)
        else:
            state[i + 1] = int(not (uniform[i] < 0.005 / reward[i]))
    starts = [i + 1 for i in range(1, length) if state[i] == 1 and state[i - 1] == 0]
    stops = [i + 1 for i in range(1, length) if state[i] == 0 and state[i - 1] == 1]
    if len(starts) > len(stops):
        stops.append(length)
    raw_lick = [0] * length
    offsets = iter(case["bout_offsets"])
    for start, stop in zip(starts, stops):
        if stop - start + 1 > 100 or start > length - 100:
            offset = next(offsets)
            if not 1 <= offset <= 60:
                raise ValueError("source bout offsets are integers in 1..60")
            left, right = start + offset, min(stop + offset, length)
            # Source assignment writes both zeros and ones: overlapping shifted
            # intervals can overwrite previous template entries.
            for position in range(left, right + 1):
                raw_lick[position - 1] = int((position - left) % 150 == 0)
    licks = [index + 1 + 100 for index, value in enumerate(raw_lick)
             if value == 1 and index + 1 + 100 < length]
    return {"licks": licks, "state": state}


def _stimulation(case):
    variant, arm = case["variant"], case["arm"]
    beta, triggered = float(case["base_beta"]), False
    if arm not in ("control", "stimLick_minus", "stimLick_plus"):
        raise ValueError("unknown arm")
    if variant == "pinned_code_30ms_set4":
        if arm == "stimLick_minus":
            triggered = case["error_r"] > 0 and case["uniform"] < 0.6
        elif arm == "stimLick_plus":
            triggered = case["error_r"] < 0
        if arm != "control":
            beta = 4.0 if triggered else beta - 0.75
    elif variant == "article_25ms_times2":
        if arm == "stimLick_minus":
            triggered = not case["pre_reward_lick"]
        elif arm == "stimLick_plus":
            triggered = bool(case["pre_reward_lick"]) and (
                case["prior_stimulations"] < math.floor(case["session_trials"] / 2))
        if triggered:
            beta *= 2
    else:
        raise ValueError("unknown variant")
    return {"beta": beta, "triggered": bool(triggered)}


def _learning(case):
    net, e = case["network"], case["eligibility"]
    recurrent, inputs = copy.deepcopy(net["recurrent"]), copy.deepcopy(net["inputs"])
    pe = case["error_r"] - case["baseline"]
    for i in range(len(recurrent)):
        for j in range(len(recurrent)):
            trace = e[j][i] if net["original_support"][i][j] else 0.0
            delta = -case["eta_j"] * trace * pe * case["beta"]
            if math.isnan(delta):
                delta = 0.0
            recurrent[i][j] += max(-0.01, min(0.01, delta))
    rate = case["tau_trans"] / 125.0
    for channel, error in ((1, case["error_r"]), (0, 0.5 * case["error_r"])):
        old = inputs[0][channel]
        proposed = old + (0.1 + case["saturation"] - old) * rate * error * case["beta"]
        inputs[0][channel] = max(0.0, min(case["saturation"], proposed))
    return {"recurrent": recurrent, "inputs": inputs,
            "baseline": 0.75 * case["baseline"] + 0.25 * case["error_r"]}


def evaluate_case(case: dict) -> dict:
    """Evaluate one JSON-compatible case without importing the main port."""
    functions = {"gaussian": _gaussian, "recurrent": _recurrent, "plant": _plant,
                 "stimulation": _stimulation, "learning": _learning}
    return functions[case["kind"]](case)


def make_cases() -> list[dict]:
    """Self-chosen inputs fixed before any primary/reference comparison."""
    net = {"recurrent": [[0.0, 0.2, -0.3], [0.4, 0.0, 0.15], [-0.2, 0.25, 0.0]],
           "inputs": [[0.31, 0.48], [-0.22, 0.13], [0.09, -0.37]],
           "feedback": [0.12, -0.07, 0.03],
           "original_support": [[False, True, True], [True, False, True], [True, True, False]],
           "bias_indices": [2], "bias_value": -0.17}
    result = [
        {"id": "gaussian_off_center_nonunit_dt", "kind": "gaussian", "mu": 1.3,
         "sigma": 0.8, "duration": 3.0, "dt": 0.25},
        {"id": "gaussian_two_samples", "kind": "gaussian", "mu": -0.7,
         "sigma": 2.4, "duration": 4.0, "dt": 2.0},
        {"id": "gaussian_truncated_endpoint", "kind": "gaussian", "mu": 3.1,
         "sigma": 1.7, "duration": 5.6, "dt": 0.75},
    ]
    stimulus = [[0.7, -0.2, 0.0, 0.4, 0.1, -0.6], [-0.1, 0.6, 0.2, 0.0, -0.8, 0.3]]
    for name, tape in (
        ("null", None), ("explicit_zero", [[0.0] * 6 for _ in range(3)]),
        ("asymmetric", [[0.04, 0.0, -0.12, 0.03, 0.0, 0.09],
                        [-0.02, 0.13, 0.0, -0.04, 0.05, 0.0],
                        [0.3, -0.2, 0.4, -0.1, 0.6, -0.5]]),
    ):
        result.append({"id": f"recurrent_bias_feedback_{name}", "kind": "recurrent",
                       "network": copy.deepcopy(net), "stimulus": copy.deepcopy(stimulus),
                       "tau_ms": 3.7, "perturbations": tape})
    net_two = {"recurrent": [[0.11, -0.28], [0.63, 0.09]],
               "inputs": [[0.14, -0.31], [0.57, 0.08]], "feedback": [0.0, 0.0],
               "original_support": [[True, True], [True, True]],
               "bias_indices": [], "bias_value": 0.0}
    result.append({"id": "recurrent_unbiased_no_feedback", "kind": "recurrent", "network": net_two,
                   "stimulus": [[0.2, 0.5, -0.7, 0.1], [0.9, -0.3, 0.0, 0.4]],
                   "tau_ms": 1.4, "perturbations": [[0.01, 0.04, 0.0, -0.02], [0.0, -0.03, 0.09, 0.0]]})
    for suffix, error, baseline, beta, eta, tau in (
        ("asymmetric_mask", 0.4, -0.15, 1.3, 0.0007, 2.0),
        ("positive_saturation", 0.95, -0.8, 4.0, 0.8, 220.0),
        ("negative_saturation", -0.85, 0.6, 4.0, 0.8, 220.0),
        ("zero_PE_input_still_updates", 0.3, 0.3, 0.6, 0.4, 25.0),
    ):
        result.append({"id": f"learning_{suffix}", "kind": "learning", "network": copy.deepcopy(net),
                       "eligibility": [[7.0, -2.0, 0.3], [1.1, -8.0, 2.2], [-0.9, 4.0, 9.0]],
                       "error_r": error, "baseline": baseline, "beta": beta,
                       "eta_j": eta, "tau_trans": tau, "saturation": 0.82})
    for variant in ("pinned_code_30ms_set4", "article_25ms_times2"):
        for arm in ("control", "stimLick_minus", "stimLick_plus"):
            for idx, (error, lick, prior, uniform) in enumerate((
                (0.2, False, 0, 0.59), (0.2, True, 2, 0.6),
                (-0.3, True, 1, 0.91), (0.0, False, 3, 0.0),
            )):
                result.append({"id": f"stimulation_{variant}_{arm}_{idx}", "kind": "stimulation",
                               "variant": variant, "arm": arm, "base_beta": 0.4 if idx == 3 else 1.7,
                               "error_r": error, "pre_reward_lick": lick, "session_trials": 5,
                               "prior_stimulations": prior, "uniform": uniform})
    # Source ticks below are intentionally one-based; array conversion is here,
    # independent of the plant's transition and output indexing implementation.
    activity, uniform = [0.0] * 490, [0.73] * 490
    for tick in (1, 143, 394):
        activity[tick - 1] = 0.9
    for tick in (113, 170, 410):
        uniform[tick - 1] = 0.001
    activity[250] = -0.9
    result.append({"id": "plant_mixed_bouts_and_late_cutoff", "kind": "plant",
                   "activity": activity, "reward": [1.0] * 490, "scale": 1.0,
                   "transition_uniforms": uniform, "bout_offsets": [13, 47]})
    activity, uniform = [0.0] * 360, [0.81] * 360
    activity[0], uniform[0] = 0.25, 0.125  # equality must not initiate
    activity[1], uniform[1] = 0.25, 0.124
    uniform[100] = 0.004  # inclusive span exactly100, not qualified
    activity[260], uniform[260] = 0.9, 0.1
    result.append({"id": "plant_strict_transition_and_100_tick_boundary", "kind": "plant",
                   "activity": activity, "reward": [1.0] * 360, "scale": 2.0,
                   "transition_uniforms": uniform, "bout_offsets": [60]})
    activity, uniform = [0.0] * 3000, [0.999] * 3000
    activity[1589] = 1.0
    uniform[1602] = 0.003
    uniform[1749] = 0.00001
    result.append({"id": "plant_default_reward_switch", "kind": "plant",
                   "activity": activity, "reward": None, "scale": 1.0,
                   "transition_uniforms": uniform, "bout_offsets": [37]})
    result.append({"id": "plant_rectified_silent", "kind": "plant",
                   "activity": [-0.2] * 230, "reward": [1.0] * 230, "scale": 0.7,
                   "transition_uniforms": [0.0] * 230, "bout_offsets": []})
    return result


SOURCE_DOMAIN_EXTENSION_SPECIFICATION = {
    "scope": "six_additional_cases_after_interface_domain_clarification",
    "independence": "inputs_chosen_from_pinned_source_and_clarified_domain_without_main_port_or_comparison_output_reads",
    "source_basis": "trainer line50:DA_trans=3*cumsum(positiveGaussian);line259:base_beta=1+twoDA_transentries",
    "domain": "base_beta>=1; beta1 tests the lower envelope, not a claimed observed network state",
    "original_39_cases_modified": False,
    "boundary_choices": "strict positive error;uniform just below/equal.6;plus error zero/negative;minus no cap;odd-session plus cap equality",
}


def make_source_domain_cases() -> list[dict]:
    """Additional source-domain boundary inputs, with original cases preserved.

    Source-only extension after discovering the original interface had omitted
    a beta-domain restriction. No primary outputs or expectations were read.
    """
    base = {"kind": "stimulation", "base_beta": 1.0, "session_trials": 5,
            "prior_stimulations": 0, "uniform": 0.2,
            "pre_reward_lick": False, "error_r": 0.0}
    choices = (
        {"id": "source_boundary_pinned_minus_positive_below_uniform",
         "variant": "pinned_code_30ms_set4", "arm": "stimLick_minus",
         "error_r": 1e-9, "uniform": math.nextafter(0.6, 0.0)},
        {"id": "source_boundary_pinned_minus_uniform_equality",
         "variant": "pinned_code_30ms_set4", "arm": "stimLick_minus",
         "error_r": 0.25, "uniform": 0.6},
        {"id": "source_boundary_pinned_plus_error_zero",
         "variant": "pinned_code_30ms_set4", "arm": "stimLick_plus",
         "error_r": 0.0, "pre_reward_lick": True},
        {"id": "source_boundary_pinned_plus_negative_error",
         "variant": "pinned_code_30ms_set4", "arm": "stimLick_plus",
         "error_r": -1e-9, "pre_reward_lick": False, "uniform": 0.9},
        {"id": "source_boundary_article_minus_no_plus_cap",
         "variant": "article_25ms_times2", "arm": "stimLick_minus",
         "pre_reward_lick": False, "prior_stimulations": 3, "error_r": -0.2},
        {"id": "source_boundary_article_plus_odd_session_cap_equality",
         "variant": "article_25ms_times2", "arm": "stimLick_plus",
         "pre_reward_lick": True, "prior_stimulations": 2, "error_r": 0.2},
    )
    return [dict(base, **choice) for choice in choices]


def main():
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("run component computations in the coordinated Slurm allocation")
    cases = make_cases()
    artifact = {"specification": INDEPENDENT_SPECIFICATION,
                "cases": [{"case": case, "reference": evaluate_case(case)} for case in cases]}
    print(json.dumps(artifact, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
