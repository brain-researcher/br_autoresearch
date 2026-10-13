#!/usr/bin/env python3
"""Run only EP04 mathematical component fixtures in a task-owned allocation."""
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
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("qualification requires the coordinator-owned compute allocation")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"set{name}=1 for the one-CPU qualification")
    args.output_root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1] / "tests"), pattern="test_native_rsa.py")
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (args.output_root / "tests.txt").write_text(stream.getvalue())
    report = {
        "analysis_id": "implementation_qualification", "job_id": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(), "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "component_equations_pass" if result.wasSuccessful() else "component_equations_fail",
        "tests_run": result.testsRun, "errors": len(result.errors), "failures": len(result.failures),
        "seconds": time.monotonic() - start,
        "empirical_payload_access": False, "panel_evaluations": 0, "candidate_decisions": 0,
        "audit_openings": 0, "production_panel_qualified": False,
        "scope": "Analytic invariants and counterexamples only; tests authored with main implementation, not independently authored evaluator.",
        "remaining": ["role-filtered source/model/low-level handoff", "pre-score preprocessing/fold/quantile decisions",
                      "production panel and complete null loops", "all mandatory diagnostics and sensitivities",
                      "exposure review and permission-separated one-shot audit"],
    }
    (args.output_root / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(stream.getvalue(), end="")
    print(report["status"])
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
