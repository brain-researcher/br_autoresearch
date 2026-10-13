#!/usr/bin/env python3
"""Compare independently authored equations on the checker's synthetic inputs."""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket

import numpy as np

import actr_port as primary
import independent_actr_checker as independent


# Prospective component-replay tolerance, not an empirical endpoint/calibration
# margin or the still-unselected production score-replay tolerance.
ATOL = 1e-12
RTOL = 1e-12


def network_from(value):
    return primary.Network(
        np.asarray(value["recurrent"], dtype=float).copy(),
        np.asarray(value["inputs"], dtype=float).copy(),
        np.asarray(value["feedback"], dtype=float).copy(),
        np.asarray(value["original_support"], dtype=bool).copy(),
        tuple(value.get("bias_indices", [])), float(value.get("bias_value", 1.0)),
    )


def evaluate_primary(case):
    kind = case["kind"]
    if kind == "gaussian":
        return {"values": primary.gaussian_formula(case["mu"], case["sigma"], case["duration"], case["dt"]).tolist()}
    if kind == "recurrent":
        variant = primary.Variant("article_25ms_times2", float(case["tau_ms"]), "explicit_equation_case")
        tape = case["perturbations"]
        result = primary.recurrent_trial(network_from(case["network"]),
                                         np.asarray(case["stimulus"], dtype=float), variant,
                                         None if tape is None else np.asarray(tape, dtype=float))
        return {"output": result.output.tolist(), "hidden": result.hidden.tolist(),
                "eligibility": result.eligibility.tolist()}
    if kind == "plant":
        reward = case["reward"]
        licks, state = primary.behavioral_plant(
            np.asarray(case["activity"], dtype=float), np.random.default_rng(0),
            reward=None if reward is None else np.asarray(reward, dtype=float),
            scale=case["scale"], transition_uniforms=np.asarray(case["transition_uniforms"], dtype=float),
            bout_offsets=np.asarray(case["bout_offsets"], dtype=int),
        )
        return {"licks": licks.tolist(), "state": state.tolist()}
    if kind == "stimulation":
        beta, triggered = primary.stimulation(primary.VARIANTS[case["variant"]],
                                              **{key: case[key] for key in (
                                                  "arm", "base_beta", "error_r", "pre_reward_lick",
                                                  "session_trials", "prior_stimulations", "uniform")})
        return {"beta": float(beta), "triggered": bool(triggered)}
    if kind == "learning":
        network = network_from(case["network"])
        baseline = primary.learning_step(network, np.asarray(case["eligibility"], dtype=float),
                                          **{key: case[key] for key in (
                                              "error_r", "baseline", "beta", "eta_j", "tau_trans", "saturation")})
        return {"recurrent": network.recurrent.tolist(), "inputs": network.inputs.tolist(), "baseline": baseline}
    raise ValueError(f"unknown checker case kind:{kind}")


def compare_values(actual, reference, path="output"):
    if isinstance(reference, dict):
        if not isinstance(actual, dict) or set(actual) != set(reference):
            raise AssertionError(f"{path}: output keys differ")
        pieces = [compare_values(actual[key], reference[key], f"{path}.{key}") for key in reference]
    elif isinstance(reference, list):
        if not isinstance(actual, list) or len(actual) != len(reference):
            raise AssertionError(f"{path}: output dimensions differ")
        pieces = [compare_values(left, right, f"{path}[{i}]") for i, (left, right) in enumerate(zip(actual, reference))]
    elif isinstance(reference, bool):
        if actual is not reference:
            raise AssertionError(f"{path}: Boolean result differs")
        return 0.0, 1
    elif isinstance(reference, (float, int)):
        error = abs(float(actual) - float(reference))
        if not math.isfinite(error) or error > ATOL + RTOL * abs(float(reference)):
            raise AssertionError(f"{path}: numeric discrepancy {error}")
        return error, 1
    else:
        if actual != reference:
            raise AssertionError(f"{path}: value differs")
        return 0.0, 1
    return max((part[0] for part in pieces), default=0.0), sum(part[1] for part in pieces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("independent replay requires task-owned compute allocation")
    args.output_root.mkdir(parents=True, exist_ok=False)
    original_cases = independent.make_cases()
    additional_cases = independent.make_source_domain_cases()
    cases = original_cases + additional_cases
    results = []
    for case in cases:
        try:
            if case["kind"] == "stimulation" and case["base_beta"] < 1:
                # The original interface omitted the source-derived domain.
                # Preserve these original reference fixtures without claiming
                # that rejecting them verifies their algebraic extension.
                try:
                    evaluate_primary(copy.deepcopy(case))
                except ValueError as error:
                    if "source base multiplier" not in str(error):
                        raise
                    results.append({"id": case["id"], "kind": case["kind"],
                                    "status": "out_of_source_domain_rejected",
                                    "base_beta": case["base_beta"],
                                    "numerical_equation_comparison": False,
                                    "reason": "Pinned source constructs base_beta as1 plus two nonnegative DA components."})
                    continue
                raise AssertionError("out-of-source-domain multiplier was accepted")
            expected = independent.evaluate_case(copy.deepcopy(case))
            actual = evaluate_primary(copy.deepcopy(case))
            error, count = compare_values(actual, expected)
            results.append({"id": case["id"], "kind": case["kind"], "status": "pass",
                            "maximum_absolute_difference": error, "compared_scalars": count})
        except Exception as error:
            results.append({"id": case["id"], "kind": case["kind"], "status": "fail",
                            "error_type": type(error).__name__, "error": str(error)})
    passed = bool(cases) and all(item["status"] in ("pass", "out_of_source_domain_rejected") for item in results)
    report = {
        "analysis_id": "implementation_qualification",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(), "job_id": os.environ["SLURM_JOB_ID"],
        "status": "independent_component_replay_pass" if passed else "independent_component_replay_fail",
        "independent_specification": independent.INDEPENDENT_SPECIFICATION,
        "independent_source_domain_extension": independent.SOURCE_DOMAIN_EXTENSION_SPECIFICATION,
        "independence": "Reference author read source equations and interface only, not primary port/tests/results. Primary author supplied this adapter only after reference authorship.",
        "case_count": len(cases), "case_counts_by_kind": dict(Counter(case["kind"] for case in cases)),
        "original_case_count": len(original_cases),
        "independently_added_source_domain_case_count": len(additional_cases),
        "numerical_equation_comparison_count": sum(item["status"] == "pass" for item in results),
        "out_of_source_domain_rejection_count": sum(item["status"] == "out_of_source_domain_rejected" for item in results),
        "domain_resolution": "The first replay is retained. Original base_beta0.4 cases are preserved and explicitly rejected, not counted as equation agreement. Source trainer lines50 and259 establish baseline1 plus nonnegative Gaussian cumulative components. Scientific guards remain unchanged.",
        "component_absolute_tolerance": ATOL, "component_relative_tolerance": RTOL,
        "cases": results,
        "empirical_payload_access": False, "audit_opened": False,
        "production_model_qualified": False, "valid_adaptive_trials": 0,
        "remaining": "Full task/seed recovery, verified source callbacks and initialization/task semantics, complete model panel, endpoint calibration and role separation remain required.",
    }
    (args.output_root / "independent_replay.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
