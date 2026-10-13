#!/usr/bin/env python3
"""Run the EP10 failed-path two-repeat diagnostic on one-thread AMD BLAS."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_PATH = Path(os.path.abspath(__file__))
V2_PATH = SCRIPT_PATH.parent / "diagnose_v3_m1_real_repro_v2.py"

SPEC = importlib.util.spec_from_file_location(
    "ep10_fit_v3_m1_real_repro_1thread_v3_dependency", V2_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import four-thread repro diagnostic: {V2_PATH}")
v2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v2
SPEC.loader.exec_module(v2)


v2.OUTPUT_PREFIX = "v3_m1_real_warm_repro_1thread_v3"
v2.SUMMARY_FILENAME = "v3_m1_real_warm_repro_1thread_v3.json"
v2.STAGE = "v3_M1_real_warm_path_single_thread_repro_v3"
v2.EXPECTED_OPENBLAS_THREADS = 1
v2.EXPECTED_FAILURE_STATUS = "pass_diagnostic_single_thread_failure_reproduced"
v2.EXPECTED_FAILURE_DECISION = "single_thread_arm_failed_requires_solver_amendment"
v2.EXPECTED_FAILURE_REASON_BOTH = "single_thread_failure_reproduced_in_both_repeats"
v2.EXPECTED_FAILURE_REASON_ONE = "single_thread_failure_reproduced_in_one_of_two_repeats"
v2.DISAGREEMENT_STATUS = "pass_diagnostic_single_thread_repeat_disagreement"
v2.DISAGREEMENT_DECISION = "single_thread_arm_failed_requires_solver_amendment"
v2.DISAGREEMENT_REASON = "single_thread_repeat_disagreement"
v2.AGREEMENT_STATUS = "pass_two_repeat_single_thread"
v2.AGREEMENT_DECISION = "single_thread_backend_pass"
v2.AGREEMENT_REASON = "both_single_thread_paths_and_all_repeat_tolerances_passed"


OUTPUT_PREFIX = v2.OUTPUT_PREFIX
SUMMARY_FILENAME = v2.SUMMARY_FILENAME
EXPECTED_OPENBLAS_THREADS = v2.EXPECTED_OPENBLAS_THREADS
build_parser = v2.build_parser
run_diagnostic = v2.run_diagnostic


def main(argv: Sequence[str] | None = None) -> int:
    return v2.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
