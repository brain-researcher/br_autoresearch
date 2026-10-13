#!/usr/bin/env python3
"""Qualify selected source-independent EP19 equations, without opening signals."""
import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import socket
import time
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("tests require task-owned compute allocation")
    if any(os.environ.get(name) != "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("set numeric threads to1")
    args.output_root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1] / "tests"), pattern="test_forecast_components.py")
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (args.output_root / "tests.txt").write_text(stream.getvalue())
    report = {
        "analysis_id": "past_only_controls", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
        "status": "selected_component_checks_pass" if result.wasSuccessful() else "selected_component_checks_fail",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "seconds": time.monotonic() - start, "source_signal_access": False, "held_out_opened": False,
        "scientific_margins_frozen": False, "onset_timing_qualified": False,
        "production_past_only_controls_complete": False, "valid_development_trials": 0,
        "scope": "Provisional calibration, censoring and fixed input-slot equations only; developer-authored tests, no independent model evaluator.",
        "remaining": ["development source-role views and timing", "onset detector", "five reference families",
                      "model-level state and block mutation tests", "prospective motif and leakage controls",
                      "finite search and scientific margins", "natural32-event source packet and guard qualification"],
    }
    (args.output_root / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(stream.getvalue(), end="")
    print(report["status"])
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
