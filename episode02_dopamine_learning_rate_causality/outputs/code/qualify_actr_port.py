#!/usr/bin/env python3
"""Run outcome-blind component tests and six short source-variant fixtures."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
import unittest

import actr_port


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("qualification requires a verified compute-node allocation")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"set{name}=1 for this serial qualification")
    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    tests = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1] / "tests"),
                                               pattern="test_actr_port.py")
    test_result = unittest.TextTestRunner(verbosity=2).run(tests)
    if not test_result.wasSuccessful():
        return 1
    runs = []
    for variant in actr_port.VARIANTS:
        for arm in ("control", "stimLick_minus", "stimLick_plus"):
            runs.append(actr_port.synthetic_fixture(variant, arm, 20261003, trials=4))
    report = {
        "analysis_id": "implementation_qualification",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(), "job_id": os.environ["SLURM_JOB_ID"],
        "status": "synthetic_component_qualification_pass",
        "test_count": test_result.testsRun, "synthetic_runs": runs,
        "elapsed_seconds": time.perf_counter() - started,
        "empirical_payload_access": False, "valid_adaptive_trials": 0,
        "production_model_qualified": False,
        "remaining_qualification": [
            "independent_equation_implementation_replay",
            "full800_trial_schedule_and_all_five_seed_model_recovery",
            "verified_online_trigger_and_empirical_endpoint_semantics",
            "production_initialization_and_article_code_variant_selection",
            "complete_strong_rival_panel_and_LOSO_fitting",
        ],
        "fixture_assumptions": {
            "initializer": "PCG64_Bernoulli0.9_Gaussian_gain1.3_no_bias_no_feedback_not_author_asset",
            "schedule": "synthetic3000_ticks_cue500to599_reward1600to1609_not_recovered_author_input_asset",
            "article_plus_cap": "first_eligible_until_floor_half_fixture_trials_not_verified_online_ordering",
            "Gaussian": "source_denominator4sigma_squared_trapezoid_area_over_indices",
        },
        "binding_thresholds_activated": False, "audit_opened": False,
    }
    (args.output_root / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("EP02 synthetic component qualification PASS; production and audit remain unqualified.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
